import os
import io
import yaml
import time
import logging
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import requests
import onnxruntime as ort
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

# Configure logging
log_dir = os.path.join('logs', 'api')
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'api_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('config', exist_ok=True)
os.makedirs('models/final', exist_ok=True)
os.makedirs('static', exist_ok=True)


# Load configuration
def load_config():
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.warning(f"Error loading configuration: {e}, using default values")
        # Use default configuration
        return {
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'max_request_size': 10,
                'timeout': 30,
                'cors_origins': ['*']
            },
            'evaluation': {
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_detections': 100
            },
            'logs_dir': 'logs'
        }


config = load_config()


# Model initialization
class ObjectDetectionModel:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Load class names
        try:
            with open('config/data_config.yaml', 'r') as f:
                data_config = yaml.safe_load(f)
                self.class_names = data_config['dataset']['classes']
        except Exception as e:
            logger.warning(f"Error loading class names: {e}, using default classes")
            self.class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                                "dog"]

        # Load ONNX model
        try:
            # Check if GPU is available
            providers = ['CUDAExecutionProvider',
                         'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else [
                'CPUExecutionProvider']
            logger.info(f"Using providers: {providers}")

            self.session = ort.InferenceSession(model_path, providers=providers)

            # Get model metadata
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape

            # Get input dimensions
            _, _, self.input_height, self.input_width = self.input_shape

            logger.info(f"Model loaded successfully. Input shape: {self.input_shape}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def preprocess(self, image):
        """
        Preprocess image for model input

        Args:
            image: PIL Image or numpy array

        Returns:
            numpy.ndarray: Preprocessed image
        """
        if isinstance(image, Image.Image):
            # Convert PIL Image to numpy array
            image = np.array(image)

        # Convert BGR to RGB (if needed)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image
        resized = cv2.resize(image, (self.input_width, self.input_height))

        # Normalize pixel values (0-1)
        normalized = resized / 255.0

        # Transpose to match model input (N,C,H,W)
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, 0).astype(np.float32)

        return batched, image.shape

    def non_max_suppression(self, predictions, conf_threshold, iou_threshold):
        """
        Apply non-maximum suppression to predictions

        Args:
            predictions: Raw model output
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            list: List of detections after NMS
        """
        # Reshape predictions: [batch, num_predictions, num_classes + 5]
        # where 5 is [x, y, w, h, confidence]
        batch_size = predictions.shape[0]

        # Create empty list for detections
        detections = []

        # Process each image in the batch
        for i in range(batch_size):
            pred = predictions[i]

            # Filter by confidence threshold
            mask = pred[:, 4] > conf_threshold
            pred = pred[mask]

            if len(pred) == 0:
                detections.append([])
                continue

            # Calculate class scores
            class_scores = pred[:, 5:]
            class_ids = np.argmax(class_scores, axis=1)

            # Convert to [x, y, w, h, confidence, class_id]
            boxes = []
            for j, box in enumerate(pred):
                x, y, w, h = box[:4]
                confidence = box[4]
                class_id = class_ids[j]
                class_score = class_scores[j, class_id]
                confidence *= class_score  # Multiply confidence by class score

                if confidence > conf_threshold:
                    boxes.append([x, y, w, h, confidence, class_id])

            if len(boxes) == 0:
                detections.append([])
                continue

            boxes = np.array(boxes)

            # Convert to xyxy format for NMS
            xyxy_boxes = np.zeros_like(boxes[:, :4])
            xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
            xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
            xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
            xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                xyxy_boxes.tolist(),
                boxes[:, 4].tolist(),
                conf_threshold,
                iou_threshold
            )

            if len(indices) > 0:
                # Handle different versions of OpenCV
                if isinstance(indices, tuple):
                    indices = indices[0]
                elif isinstance(indices, np.ndarray) and indices.ndim > 1:
                    indices = indices.flatten()

                # Get filtered boxes
                filtered_boxes = [boxes[i] for i in indices]
                detections.append(filtered_boxes)
            else:
                detections.append([])

        return detections

    def postprocess(self, predictions, original_shape, max_detections=100):
        """
        Postprocess predictions and format results

        Args:
            predictions: Raw model output
            original_shape: Original image shape (height, width)
            max_detections: Maximum number of detections to return

        Returns:
            list: List of detection dictionaries
        """
        # Apply NMS
        detections = self.non_max_suppression(
            predictions,
            self.conf_threshold,
            self.iou_threshold
        )[0]  # Get detections for the first (and only) image

        # Limit number of detections
        if max_detections > 0 and len(detections) > max_detections:
            # Sort by confidence and take top N
            detections = sorted(detections, key=lambda x: x[4], reverse=True)[:max_detections]

        # Convert detections to output format
        original_height, original_width = original_shape[:2]
        results = []

        for detection in detections:
            x, y, w, h, confidence, class_id = detection

            # Convert normalized coordinates to pixel coordinates
            x1 = int((x - w / 2) * original_width)
            y1 = int((y - h / 2) * original_height)
            x2 = int((x + w / 2) * original_width)
            y2 = int((y + h / 2) * original_height)

            # Ensure coordinates are within image boundaries
            x1 = max(0, min(x1, original_width - 1))
            y1 = max(0, min(y1, original_height - 1))
            x2 = max(0, min(x2, original_width - 1))
            y2 = max(0, min(y2, original_height - 1))

            class_id = int(class_id)
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

            results.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class_id': int(class_id),
                'class_name': class_name
            })

        return results

    def predict(self, image):
        """
        Run inference on image

        Args:
            image: PIL Image or numpy array

        Returns:
            list: List of detection dictionaries
        """
        # Preprocess image
        preprocessed_image, original_shape = self.preprocess(image)

        # Run inference
        start_time = time.time()
        outputs = self.session.run(
            None,
            {self.input_name: preprocessed_image}
        )
        inference_time = time.time() - start_time

        # Postprocess predictions
        predictions = outputs[0]
        results = self.postprocess(predictions, original_shape)

        return results, inference_time


# Initialize model
model = None


class MockModel:
    """Mock model for testing when the real model is not available"""

    def __init__(self, conf_threshold=0.25, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = ["person", "car", "dog", "cat", "bicycle"]
        logger.info("Initialized mock model for testing")

    def predict(self, image):
        # Generate a single random detection for demo purposes
        width, height = image.size if hasattr(image, 'size') else (image.shape[1], image.shape[0])

        # Create a random detection (20% of the time return empty for realism)
        if np.random.random() > 0.2:
            # Random box dimensions (between 10-30% of image size)
            box_width = int(width * (0.1 + np.random.random() * 0.2))
            box_height = int(height * (0.1 + np.random.random() * 0.2))

            # Random box position
            x1 = int(np.random.random() * (width - box_width))
            y1 = int(np.random.random() * (height - box_height))
            x2 = x1 + box_width
            y2 = y1 + box_height

            # Random class and confidence
            class_id = np.random.randint(0, len(self.class_names))
            confidence = 0.5 + np.random.random() * 0.5  # Between 0.5 and 1.0

            result = [{
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class_id': int(class_id),
                'class_name': self.class_names[class_id]
            }]
        else:
            result = []

        # Simulate inference time (between 20-100ms)
        inference_time = 0.02 + np.random.random() * 0.08

        logger.info(f"Mock model generated {len(result)} detections in {inference_time:.3f}s")
        return result, inference_time


def initialize_model():
    global model
    model_path = os.environ.get('MODEL_PATH', 'models/final/model.onnx')

    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}, using mock model for testing")
        model = MockModel(
            conf_threshold=config['evaluation']['confidence_threshold'],
            iou_threshold=config['evaluation']['iou_threshold']
        )
        return

    try:
        model = ObjectDetectionModel(
            model_path,
            conf_threshold=config['evaluation']['confidence_threshold'],
            iou_threshold=config['evaluation']['iou_threshold']
        )
        logger.info(f"Model initialized successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to initialize real model: {e}, falling back to mock model")
        model = MockModel(
            conf_threshold=config['evaluation']['confidence_threshold'],
            iou_threshold=config['evaluation']['iou_threshold']
        )


# FastAPI models
class DetectionResponse(BaseModel):
    bbox: List[int] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence score")
    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")


class PredictionResponse(BaseModel):
    detections: List[DetectionResponse] = Field(..., description="List of detections")
    inference_time: float = Field(..., description="Inference time in seconds")
    model_name: str = Field(..., description="Model name")
    image_size: List[int] = Field(..., description="Image size [width, height]")


# Create FastAPI app
app = FastAPI(
    title="Object Detection API",
    description="API for object detection using YOLO model",
    version="1.0.0"
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api']['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Ensure OpenAPI version is set
    openapi_schema["openapi"] = "3.0.0"

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation link"""
    return """
    <html>
        <head>
            <title>Object Detection API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                a { color: #0066cc; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>Object Detection API</h1>
            <p>Welcome to the Object Detection API. This API allows you to detect objects in images using a YOLO model.</p>
            <p>Please visit the <a href="/docs">API documentation</a> for more information on how to use the API.</p>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        try:
            initialize_model()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Model not initialized: {str(e)}"}
            )

    # Check if it's a mock model
    is_mock = isinstance(model, MockModel)

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "mock" if is_mock else "real",
        "configuration": {
            "confidence_threshold": model.conf_threshold,
            "iou_threshold": model.iou_threshold
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
        file: UploadFile = File(...),
        conf_threshold: Optional[float] = Query(None, description="Confidence threshold (0-1)"),
        max_detections: Optional[int] = Query(None, description="Maximum number of detections")
):
    """
    Detect objects in an uploaded image

    Args:
        file: Image file
        conf_threshold: Confidence threshold (0-1)
        max_detections: Maximum number of detections to return

    Returns:
        JSON with detected objects
    """
    # Check if model is initialized
    if model is None:
        try:
            initialize_model()
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")

    # Temporary override confidence threshold if provided
    original_conf_threshold = model.conf_threshold
    if conf_threshold is not None:
        model.conf_threshold = float(conf_threshold)

    try:
        # Read image file
        contents = await file.read()
        if len(contents) > config['api']['max_request_size'] * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {config['api']['max_request_size']} MB"
            )

        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))

        # Run inference
        start_time = time.time()
        detections, inference_time = model.predict(image)

        # Override max_detections if provided
        if max_detections is not None and max_detections > 0:
            # Sort by confidence and take top N
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:max_detections]

        # Create response
        model_name = "YOLOv5 (Mock)" if isinstance(model, MockModel) else "YOLOv5"
        response = {
            "detections": detections,
            "inference_time": inference_time,
            "model_name": model_name,
            "image_size": [image.width, image.height]
        }

        logger.info(f"Processed image {file.filename} in {time.time() - start_time:.4f}s. "
                    f"Found {len(detections)} objects.")

        return response

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

    finally:
        # Restore original confidence threshold if it was changed
        if conf_threshold is not None:
            model.conf_threshold = original_conf_threshold


@app.post("/batch_predict")
async def batch_predict(
        files: List[UploadFile] = File(...),
        conf_threshold: Optional[float] = Query(None, description="Confidence threshold (0-1)"),
        max_detections: Optional[int] = Query(None, description="Maximum number of detections")
):
    """
    Detect objects in batch of images

    Args:
        files: List of image files
        conf_threshold: Confidence threshold (0-1)
        max_detections: Maximum number of detections to return

    Returns:
        JSON with detected objects for each image
    """
    # Check if model is initialized
    if model is None:
        try:
            initialize_model()
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")

    # Temporary override confidence threshold if provided
    if conf_threshold is not None:
        original_conf_threshold = model.conf_threshold
        model.conf_threshold = float(conf_threshold)

    try:
        results = []

        for file in files:
            try:
                # Read image file
                contents = await file.read()
                if len(contents) > config['api']['max_request_size'] * 1024 * 1024:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "message": f"File too large. Maximum size is {config['api']['max_request_size']} MB"
                    })
                    continue

                # Convert to PIL Image
                image = Image.open(io.BytesIO(contents))

                # Run inference
                detections, inference_time = model.predict(image)

                # Override max_detections if provided
                if max_detections is not None and max_detections > 0:
                    # Sort by confidence and take top N
                    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:max_detections]

                # Add to results
                model_name = "YOLOv5 (Mock)" if isinstance(model, MockModel) else "YOLOv5"
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "detections": detections,
                    "inference_time": inference_time,
                    "model_name": model_name,
                    "image_size": [image.width, image.height]
                })

                logger.info(f"Processed image {file.filename} in {inference_time:.4f}s. "
                            f"Found {len(detections)} objects.")

            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": str(e)
                })

        return {"results": results}

    except Exception as e:
        logger.error(f"Error processing batch request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing batch request: {str(e)}")

    finally:
        # Restore original confidence threshold if it was changed
        if conf_threshold is not None:
            model.conf_threshold = original_conf_threshold


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    try:
        logger.info("Initializing model on startup...")
        initialize_model()
        logger.info("Model initialization completed")

        # Download Swagger UI assets if they don't exist
        await download_swagger_assets()
    except Exception as e:
        logger.error(f"Failed to initialize model on startup: {e}")
        # Don't raise exception, as we want the API to start
        # Model will be initialized on first request


async def download_swagger_assets():
    """Download Swagger UI assets if they don't exist"""
    assets = {
        "swagger-ui-bundle.js": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.15.5/swagger-ui-bundle.js",
        "swagger-ui.css": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.15.5/swagger-ui.css",
        "redoc.standalone.js": "https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js"
    }

    for filename, url in assets.items():
        filepath = os.path.join("static", filename)
        if not os.path.exists(filepath):
            try:
                logger.info(f"Downloading {filename}...")
                response = requests.get(url)
                response.raise_for_status()

                with open(filepath, "wb") as f:
                    f.write(response.content)

                logger.info(f"Downloaded {filename} successfully")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")


if __name__ == "__main__":
    import uvicorn

    # Start the API server
    uvicorn.run(
        "app:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=False
    )