import os
import logging
import json
import time
import cv2
import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = "cuda"):
    """
    Load detection model from file

    Args:
        model_path: Path to model file (ONNX or PyTorch)
        device: Device to run model on ('cuda' or 'cpu')

    Returns:
        Model object
    """
    # Check file extension
    ext = os.path.splitext(model_path)[1].lower()

    if ext == '.onnx':
        # Load ONNX model
        import onnxruntime as ort

        # Check available providers
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        available_providers = [p for p in providers if p in ort.get_available_providers()]

        logger.info(f"Using ONNX Runtime with providers: {available_providers}")

        # Create session
        session = ort.InferenceSession(model_path, providers=available_providers)

        return session

    elif ext in ['.pt', '.pth']:
        # Load PyTorch model
        import torch
        from src.models.yolo import create_model
        import yaml

        # Load model configuration
        config_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'config', 'model_config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = yaml.safe_load(f)
        else:
            # Default configuration
            model_config = {
                'model': {
                    'version': 's',
                    'num_classes': 80
                }
            }

        # Create model
        model = create_model(model_config)

        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')

        # Extract state dict if it's a checkpoint
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load weights
        model.load_state_dict(state_dict, strict=False)

        # Move to device
        device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        model = model.to(device)

        # Set model to evaluation mode
        model.eval()

        return model

    else:
        raise ValueError(f"Unsupported model format: {ext}")


def preprocess_image(image: Union[np.ndarray, Image.Image, str], target_size: Tuple[int, int] = (416, 416)) -> Tuple[
    np.ndarray, tuple]:
    """
    Preprocess image for model inference

    Args:
        image: Input image (numpy array, PIL Image, or path to image file)
        target_size: Target size (height, width)

    Returns:
        tuple: (preprocessed_image, original_size)
    """
    # Load image if it's a path
    if isinstance(image, str):
        if os.path.exists(image):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(f"Image file not found: {image}")

    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Store original size
    original_size = image.shape[:2]  # (height, width)

    # Resize image
    height, width = target_size
    resized = cv2.resize(image, (width, height))

    # Normalize pixel values to 0-1
    normalized = resized / 255.0

    # Convert to float32
    normalized = normalized.astype(np.float32)

    return normalized, original_size


def run_inference_onnx(session, image: np.ndarray) -> np.ndarray:
    """
    Run inference using ONNX model

    Args:
        session: ONNX session
        image: Preprocessed image

    Returns:
        numpy.ndarray: Raw model output
    """
    # Transpose to NCHW format for ONNX (batch, channels, height, width)
    input_data = np.transpose(image, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    start_time = time.time()
    outputs = session.run([output_name], {input_name: input_data})
    inference_time = time.time() - start_time

    logger.debug(f"Inference time: {inference_time * 1000:.2f} ms")

    return outputs[0], inference_time


def run_inference_pytorch(model, image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Run inference using PyTorch model

    Args:
        model: PyTorch model
        image: Preprocessed image

    Returns:
        tuple: (outputs, inference_time)
    """
    # Convert to PyTorch tensor
    input_data = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    input_data = input_data.unsqueeze(0)

    # Move to device
    device = next(model.parameters()).device
    input_data = input_data.to(device)

    # Run inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_data)
    inference_time = time.time() - start_time

    logger.debug(f"Inference time: {inference_time * 1000:.2f} ms")

    # Convert output to numpy
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().numpy()

    return outputs, inference_time


def postprocess_detections(outputs: np.ndarray, original_size: tuple, class_names: list,
                           conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                           max_detections: int = 100) -> List[Dict[str, Any]]:
    """
    Postprocess detections from model output

    Args:
        outputs: Raw model output
        original_size: Original image size (height, width)
        class_names: List of class names
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections to return

    Returns:
        List[Dict]: List of detection dictionaries
    """
    # Apply NMS
    detections = []

    # Filter by confidence
    if outputs.shape[2] > 5:  # YOLO format [batch, num_boxes, num_classes+5]
        # Get confidence scores
        scores = outputs[0, :, 4]

        # Filter detections by confidence threshold
        mask = scores >= conf_threshold
        filtered_outputs = outputs[0, mask]

        if len(filtered_outputs) > 0:
            # Get class scores
            class_scores = filtered_outputs[:, 5:]
            class_ids = np.argmax(class_scores, axis=1)

            # Get confidence scores
            confidences = filtered_outputs[:, 4] * class_scores[np.arange(len(class_scores)), class_ids]

            # Get boxes
            boxes = filtered_outputs[:, :4]

            # Convert boxes to xyxy format for NMS
            x1 = boxes[:, 0] - boxes[:, 2] / 2  # center_x - width/2
            y1 = boxes[:, 1] - boxes[:, 3] / 2  # center_y - height/2
            x2 = boxes[:, 0] + boxes[:, 2] / 2  # center_x + width/2
            y2 = boxes[:, 1] + boxes[:, 3] / 2  # center_y + height/2

            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                np.column_stack((x1, y1, x2, y2)).tolist(),
                confidences.tolist(),
                conf_threshold,
                iou_threshold
            )

            if len(indices) > 0:
                # Handle different OpenCV versions
                if isinstance(indices, tuple):
                    indices = indices[0]
                elif isinstance(indices, np.ndarray) and indices.ndim > 1:
                    indices = indices.flatten()

                # Limit number of detections
                if max_detections > 0 and len(indices) > max_detections:
                    # Sort by confidence and take top N
                    indices = indices[np.argsort(confidences[indices])[::-1][:max_detections]]

                # Create detection objects
                for i in indices:
                    # Get box
                    box = boxes[i]

                    # Get confidence and class
                    confidence = float(confidences[i])
                    class_id = int(class_ids[i])

                    # Get class name
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                    # Scale box to original image size
                    original_height, original_width = original_size

                    # Convert normalized coordinates to pixel coordinates
                    x, y, w, h = box
                    x1 = int((x - w / 2) * original_width)
                    y1 = int((y - h / 2) * original_height)
                    x2 = int((x + w / 2) * original_width)
                    y2 = int((y + h / 2) * original_height)

                    # Ensure coordinates are within image boundaries
                    x1 = max(0, min(x1, original_width - 1))
                    y1 = max(0, min(y1, original_height - 1))
                    x2 = max(0, min(x2, original_width - 1))
                    y2 = max(0, min(y2, original_height - 1))

                    # Create detection dictionary
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    }

                    detections.append(detection)

    return detections


def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]], line_thickness: int = 2) -> np.ndarray:
    """
    Draw detections on image

    Args:
        image: Input image
        detections: List of detection dictionaries
        line_thickness: Line thickness for bounding boxes

    Returns:
        numpy.ndarray: Image with detections
    """
    # Make a copy of the image
    img = image.copy()

    # Colors for different classes
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Green (dark)
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128)  # Teal
    ]

    # Draw each detection
    for detection in detections:
        # Get box coordinates
        x1, y1, x2, y2 = detection['bbox']

        # Get class ID and confidence
        class_id = detection['class_id']
        confidence = detection['confidence']
        class_name = detection['class_name']

        # Get color for this class
        color = colors[class_id % len(colors)]

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

        # Draw label background
        text = f"{class_name} {confidence:.2f}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_thickness)
        cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)

        # Draw label text
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def encode_image_to_base64(image: np.ndarray, format: str = 'jpeg') -> str:
    """
    Encode image to base64 string

    Args:
        image: Input image
        format: Image format (jpeg, png)

    Returns:
        str: Base64-encoded image
    """
    # Convert to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))

    # Save to bytes
    buffer = BytesIO()
    image.save(buffer, format=format.upper())

    # Encode to base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f"data:image/{format};base64,{img_str}"


def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to image

    Args:
        base64_string: Base64-encoded image

    Returns:
        numpy.ndarray: Decoded image
    """
    # Remove header if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    # Decode base64
    img_data = base64.b64decode(base64_string)

    # Convert to numpy array
    img = Image.open(BytesIO(img_data))
    img_np = np.array(img)

    return img_np


def process_image_file(file_path: str, model_type: str, model, class_names: list,
                       conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                       max_detections: int = 100) -> Dict[str, Any]:
    """
    Process image file with model

    Args:
        file_path: Path to image file
        model_type: Type of model ('onnx' or 'pytorch')
        model: Model object
        class_names: List of class names
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections to return

    Returns:
        dict: Dictionary with detection results
    """
    # Load and preprocess image
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image size
    height, width = image.shape[:2]

    # Preprocess image
    preprocessed_image, original_size = preprocess_image(image)

    # Run inference
    if model_type == 'onnx':
        outputs, inference_time = run_inference_onnx(model, preprocessed_image)
    elif model_type == 'pytorch':
        outputs, inference_time = run_inference_pytorch(model, preprocessed_image)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Postprocess detections
    detections = postprocess_detections(
        outputs,
        original_size,
        class_names,
        conf_threshold,
        iou_threshold,
        max_detections
    )

    # Draw detections on image
    vis_image = draw_detections(image, detections)

    # Encode image to base64 for web display
    base64_image = encode_image_to_base64(vis_image)

    # Create response
    response = {
        'detections': detections,
        'inference_time': inference_time,
        'image_size': [width, height],
        'visualization': base64_image
    }

    return response


def save_upload_file(upload_file, upload_dir: str = "uploads") -> str:
    """
    Save uploaded file to disk

    Args:
        upload_file: File object
        upload_dir: Directory to save file

    Returns:
        str: Path to saved file
    """
    # Create upload directory if it doesn't exist
    os.makedirs(upload_dir, exist_ok=True)

    # Generate file path
    file_path = os.path.join(upload_dir, upload_file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        buffer.write(upload_file.file.read())

    return file_path


def validate_image_file(file) -> bool:
    """
    Validate image file

    Args:
        file: File object

    Returns:
        bool: True if file is valid image, False otherwise
    """
    # Get file content
    file_content = file.read()
    file.file.seek(0)  # Reset file pointer

    # Check file size (10MB max)
    if len(file_content) > 10 * 1024 * 1024:
        logger.warning(f"File too large: {len(file_content)} bytes")
        return False

    # Check file type
    try:
        img = Image.open(BytesIO(file_content))
        img.verify()  # Verify it's an image
        return True
    except Exception as e:
        logger.warning(f"Invalid image file: {e}")
        return False


def get_model_info(model) -> Dict[str, Any]:
    """
    Get model information

    Args:
        model: Model object

    Returns:
        dict: Dictionary with model information
    """
    model_info = {
        'type': 'unknown',
        'input_shape': None,
        'num_parameters': None,
        'providers': None
    }

    # Check model type
    if hasattr(model, 'get_inputs'):
        # ONNX model
        model_info['type'] = 'onnx'
        model_info['input_shape'] = model.get_inputs()[0].shape
        model_info['providers'] = model.get_providers()
    elif isinstance(model, torch.nn.Module):
        # PyTorch model
        model_info['type'] = 'pytorch'
        model_info['num_parameters'] = sum(p.numel() for p in model.parameters())

        # Get device information
        device = next(model.parameters()).device
        model_info['device'] = str(device)

    return model_info


if __name__ == "__main__":
    # Test utilities
    import argparse
    import sys

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="API utilities for object detection")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to output image")
    parser.add_argument("--threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")

    args = parser.parse_args()

    # Load class names
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

    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_model(args.model, device=args.device)

    # Determine model type
    model_type = 'onnx' if hasattr(model, 'get_inputs') else 'pytorch'
    logger.info(f"Model type: {model_type}")

    # Process image
    logger.info(f"Processing image: {args.image}")
    result = process_image_file(
        args.image,
        model_type,
        model,
        class_names,
        conf_threshold=args.threshold
    )

    # Save output image
    output_image = cv2.cvtColor(draw_detections(cv2.imread(args.image), result['detections']), cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, output_image)
    logger.info(f"Output image saved to {args.output}")

    # Print detection results
    logger.info(f"Found {len(result['detections'])} objects in {result['inference_time'] * 1000:.2f} ms")
    for detection in result['detections']:
        logger.info(f"  {detection['class_name']}: {detection['confidence']:.4f}")