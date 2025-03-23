from fastapi import APIRouter, File, UploadFile, Form, Query, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import List, Dict, Any, Optional
import os
import time
import logging
import yaml
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

from src.api.utils import (
    load_model, preprocess_image, run_inference_onnx, run_inference_pytorch,
    postprocess_detections, draw_detections, encode_image_to_base64,
    save_upload_file, validate_image_file, get_model_info
)

# Create logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global model object
model = None
model_type = None
class_names = []


# Load configuration
def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


config = load_config()


# Initialize model
def initialize_model():
    global model, model_type, class_names

    # Get model path from environment variable or config
    model_path = os.environ.get('MODEL_PATH', config.get('model_path', 'models/final/model.onnx'))

    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Get device
    device = os.environ.get('DEVICE', config.get('device', 'cuda'))

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path, device=device)

    # Determine model type
    model_type = 'onnx' if hasattr(model, 'get_inputs') else 'pytorch'
    logger.info(f"Model type: {model_type}")

    # Load class names
    try:
        with open('config/data_config.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
            class_names = data_config['dataset']['classes']
    except Exception as e:
        logger.error(f"Error loading class names: {e}")
        # Default COCO class names
        class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                       "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                       "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                       "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                       "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                       "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                       "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                       "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                       "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                       "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    logger.info(f"Loaded {len(class_names)} class names")

    return model


@router.get("/model")
async def get_model_information():
    """
    Get model information

    Returns:
        dict: Model information
    """
    global model, model_type, class_names

    # Initialize model if needed
    if model is None:
        initialize_model()

    # Get model information
    model_info = get_model_info(model)

    # Add additional information
    model_info['num_classes'] = len(class_names)
    model_info['class_names'] = class_names

    return model_info


@router.post("/predict")
async def predict_image(
        file: UploadFile = File(...),
        conf_threshold: Optional[float] = Query(0.25, ge=0.05, le=1.0, description="Confidence threshold"),
        iou_threshold: Optional[float] = Query(0.45, ge=0.1, le=1.0, description="IoU threshold for NMS"),
        max_detections: Optional[int] = Query(100, ge=1, le=1000, description="Maximum number of detections"),
        include_visualization: Optional[bool] = Query(False, description="Include base64-encoded visualization image")
):
    """
    Detect objects in an image

    Args:
        file: Image file
        conf_threshold: Confidence threshold (0-1)
        iou_threshold: IoU threshold for NMS (0-1)
        max_detections: Maximum number of detections to return
        include_visualization: Whether to include visualization in response

    Returns:
        dict: Detection results
    """
    global model, model_type, class_names

    # Initialize model if needed
    if model is None:
        initialize_model()

    # Validate file
    if not validate_image_file(file):
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Read file
    contents = await file.read()
    file.file.seek(0)  # Reset file pointer

    # Check file size
    max_size_mb = config.get('api', {}).get('max_request_size', 10)
    if len(contents) > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_size_mb} MB"
        )

    try:
        # Process image
        image = Image.open(BytesIO(contents))
        image_np = np.array(image)

        # Get image size
        height, width = image_np.shape[:2]

        # Preprocess image
        preprocessed_image, original_size = preprocess_image(image_np)

        # Run inference
        start_time = time.time()

        if model_type == 'onnx':
            outputs, inference_time = run_inference_onnx(model, preprocessed_image)
        else:  # pytorch
            outputs, inference_time = run_inference_pytorch(model, preprocessed_image)

        # Postprocess detections
        detections = postprocess_detections(
            outputs,
            original_size,
            class_names,
            conf_threshold,
            iou_threshold,
            max_detections
        )

        # Create response
        response = {
            "detections": detections,
            "inference_time": inference_time,
            "model_name": "YOLOv5",
            "image_size": [width, height]
        }

        # Add visualization if requested
        if include_visualization:
            # Draw detections on the image
            image_with_detections = draw_detections(image_np, detections)

            # Encode to base64
            vis_base64 = encode_image_to_base64(image_with_detections)

            # Add to response
            response["visualization"] = vis_base64

        logger.info(f"Processed image {file.filename} in {time.time() - start_time:.4f}s. "
                    f"Found {len(detections)} objects with confidence >= {conf_threshold}.")

        return response

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.post("/batch_predict")
async def batch_predict(
        files: List[UploadFile] = File(...),
        conf_threshold: Optional[float] = Query(0.25, ge=0.05, le=1.0, description="Confidence threshold"),
        iou_threshold: Optional[float] = Query(0.45, ge=0.1, le=1.0, description="IoU threshold for NMS"),
        max_detections: Optional[int] = Query(100, ge=1, le=1000, description="Maximum number of detections"),
        include_visualization: Optional[bool] = Query(False, description="Include base64-encoded visualization images")
):
    """
    Detect objects in multiple images

    Args:
        files: List of image files
        conf_threshold: Confidence threshold (0-1)
        iou_threshold: IoU threshold for NMS (0-1)
        max_detections: Maximum number of detections to return
        include_visualization: Whether to include visualization in response

    Returns:
        dict: Detection results for each image
    """
    global model, model_type, class_names

    # Initialize model if needed
    if model is None:
        initialize_model()

    # Check number of files
    max_batch_size = config.get('api', {}).get('max_batch_size', 10)
    if len(files) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum batch size is {max_batch_size}"
        )

    # Process each file
    results = []

    for file in files:
        try:
            # Validate file
            if not validate_image_file(file):
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "Invalid image file"
                })
                continue

            # Read file
            contents = await file.read()
            file.file.seek(0)  # Reset file pointer

            # Check file size
            max_size_mb = config.get('api', {}).get('max_request_size', 10)
            if len(contents) > max_size_mb * 1024 * 1024:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": f"File too large. Maximum size is {max_size_mb} MB"
                })
                continue

            # Process image
            image = Image.open(BytesIO(contents))
            image_np = np.array(image)

            # Get image size
            height, width = image_np.shape[:2]

            # Preprocess image
            preprocessed_image, original_size = preprocess_image(image_np)

            # Run inference
            if model_type == 'onnx':
                outputs, inference_time = run_inference_onnx(model, preprocessed_image)
            else:  # pytorch
                outputs, inference_time = run_inference_pytorch(model, preprocessed_image)

            # Postprocess detections
            detections = postprocess_detections(
                outputs,
                original_size,
                class_names,
                conf_threshold,
                iou_threshold,
                max_detections
            )

            # Create result
            result = {
                "filename": file.filename,
                "status": "success",
                "detections": detections,
                "inference_time": inference_time,
                "image_size": [width, height]
            }

            # Add visualization if requested
            if include_visualization:
                # Draw detections on the image
                image_with_detections = draw_detections(image_np, detections)

                # Encode to base64
                vis_base64 = encode_image_to_base64(image_with_detections)

                # Add to result
                result["visualization"] = vis_base64

            results.append(result)

            logger.info(f"Processed image {file.filename} in {inference_time:.4f}s. "
                        f"Found {len(detections)} objects with confidence >= {conf_threshold}.")

        except Exception as e:
            logger.error(f"Error processing image {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })

    return {"results": results}


@router.get("/visualize/{image_id}")
async def visualize_image(
        image_id: str,
        background_tasks: BackgroundTasks
):
    """
    Get visualization image for a detection result

    Args:
        image_id: Image ID

    Returns:
        StreamingResponse: Image file
    """
    # Check if image exists
    image_path = f"outputs/{image_id}.jpg"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Return image
    def cleanup():
        # Delete file after 5 minutes
        time.sleep(300)
        if os.path.exists(image_path):
            os.remove(image_path)

    background_tasks.add_task(cleanup)

    return FileResponse(image_path, media_type="image/jpeg")


@router.get("/health")
async def health_check():
    """
    Health check endpoint

    Returns:
        dict: Health status
    """
    global model

    # Check if model is loaded
    model_loaded = model is not None

    # Try to initialize model if not loaded
    if not model_loaded:
        try:
            model = initialize_model()
            model_loaded = True
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Model not initialized: {str(e)}"}
            )

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": time.time()
    }


@router.get("/classes")
async def get_classes():
    """
    Get list of class names

    Returns:
        dict: Class names
    """
    global class_names

    # Initialize model if needed (which will load class names)
    if len(class_names) == 0:
        initialize_model()

    return {
        "classes": [{"id": i, "name": name} for i, name in enumerate(class_names)]
    }