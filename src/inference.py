import os
import yaml
import torch
import argparse
import time
import logging
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import onnxruntime as ort

# Import local modules
from src.models.yolo import create_model
from src.utils.visualization import draw_boxes
from src.api.utils import (
    preprocess_image, run_inference_onnx, run_inference_pytorch,
    postprocess_detections, draw_detections
)


def setup_logging():
    """
    Set up logging configuration

    Returns:
        logger: Configured logger
    """
    log_dir = os.path.join('logs', 'inference')
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'inference.log')),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


def load_model(model_path, device='cuda'):
    """
    Load model from file

    Args:
        model_path (str): Path to model file
        device (str): Device to run on

    Returns:
        tuple: (model, model_type)
    """
    # Check model extension
    if model_path.endswith('.onnx'):
        # Load ONNX model
        # Configure ONNX providers
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        available_providers = [p for p in providers if p in ort.get_available_providers()]

        # Load model
        session = ort.InferenceSession(model_path, providers=available_providers)

        return session, 'onnx'
    else:
        # Load PyTorch model
        # Get model configuration
        config_path = os.path.join('config', 'model_config.yaml')
        with open(config_path, 'r') as f:
            model_config = yaml.safe_load(f)

        # Get number of classes
        data_config_path = os.path.join('config', 'data_config.yaml')
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
            num_classes = len(data_config['dataset']['classes'])

        # Create model
        model = create_model(model_config, num_classes=num_classes)

        # Load weights
        state_dict = torch.load(model_path, map_location='cpu')

        # Extract state dict if it's a checkpoint
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        model.load_state_dict(state_dict)

        # Move to device
        device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        model = model.to(device)
        model.eval()

        return model, 'pytorch'


def load_class_names():
    """
    Load class names from configuration

    Returns:
        list: Class names
    """
    try:
        data_config_path = os.path.join('config', 'data_config.yaml')
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
            return data_config['dataset']['classes']
    except Exception as e:
        print(f"Error loading class names: {e}")
        # Default COCO class names
        return ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def process_image(image_path, model, model_type, class_names, conf_threshold=0.25, iou_threshold=0.45, device='cuda'):
    """
    Process a single image

    Args:
        image_path (str): Path to image file
        model: Model to use
        model_type (str): Type of model ('onnx' or 'pytorch')
        class_names (list): List of class names
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold for NMS
        device (str): Device to run on

    Returns:
        tuple: (detections, visualization, inference_time)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess image
    preprocessed_image, original_size = preprocess_image(image)

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
        iou_threshold
    )

    # Draw detections
    visualization = draw_detections(image, detections)

    return detections, visualization, inference_time


def process_video(video_path, output_path, model, model_type, class_names, conf_threshold=0.25, iou_threshold=0.45,
                  device='cuda'):
    """
    Process a video file

    Args:
        video_path (str): Path to video file
        output_path (str): Path to output video file
        model: Model to use
        model_type (str): Type of model ('onnx' or 'pytorch')
        class_names (list): List of class names
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold for NMS
        device (str): Device to run on

    Returns:
        dict: Processing statistics
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process frames
    frame_count = 0
    total_inference_time = 0
    total_detections = 0

    try:
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Preprocess image
                preprocessed_image, original_size = preprocess_image(frame_rgb)

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
                    iou_threshold
                )

                # Draw detections
                visualization = draw_detections(frame_rgb, detections)

                # Convert back to BGR for writing
                visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

                # Write frame
                out.write(visualization_bgr)

                # Update statistics
                frame_count += 1
                total_inference_time += inference_time
                total_detections += len(detections)

                # Update progress bar
                pbar.update(1)
    finally:
        # Release resources
        cap.release()
        out.release()

    # Calculate statistics
    average_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
    average_fps = 1.0 / average_inference_time if average_inference_time > 0 else 0
    average_detections_per_frame = total_detections / frame_count if frame_count > 0 else 0

    return {
        'frames_processed': frame_count,
        'average_inference_time': average_inference_time,
        'average_fps': average_fps,
        'total_detections': total_detections,
        'average_detections_per_frame': average_detections_per_frame
    }


def process_images_in_directory(input_dir, output_dir, model, model_type, class_names, conf_threshold=0.25,
                                iou_threshold=0.45, device='cuda'):
    """
    Process all images in a directory

    Args:
        input_dir (str): Input directory containing images
        output_dir (str): Output directory for processed images
        model: Model to use
        model_type (str): Type of model ('onnx' or 'pytorch')
        class_names (list): List of class names
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold for NMS
        device (str): Device to run on

    Returns:
        dict: Processing statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(input_dir).glob(f'*{ext.upper()}')))

    # Process images
    total_inference_time = 0
    total_detections = 0

    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Process image
            detections, visualization, inference_time = process_image(
                str(image_file),
                model,
                model_type,
                class_names,
                conf_threshold,
                iou_threshold,
                device
            )

            # Save visualization
            output_file = os.path.join(output_dir, image_file.name)
            cv2.imwrite(output_file, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            # Update statistics
            total_inference_time += inference_time
            total_detections += len(detections)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Calculate statistics
    num_images = len(image_files)
    average_inference_time = total_inference_time / num_images if num_images > 0 else 0
    average_fps = 1.0 / average_inference_time if average_inference_time > 0 else 0
    average_detections_per_image = total_detections / num_images if num_images > 0 else 0

    return {
        'images_processed': num_images,
        'average_inference_time': average_inference_time,
        'average_fps': average_fps,
        'total_detections': total_detections,
        'average_detections_per_image': average_detections_per_image
    }


def main(args):
    # Set up logging
    logger = setup_logging()
    logger.info(f"Starting inference with model: {args.model}")

    # Load model
    model, model_type = load_model(args.model, args.device)
    logger.info(f"Loaded {model_type} model")

    # Load class names
    class_names = load_class_names()
    logger.info(f"Loaded {len(class_names)} class names")

    # Create output directory
    if not args.output:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        args.output = f"output_{timestamp}"

    os.makedirs(args.output, exist_ok=True)

    # Process inputs
    if args.image:
        logger.info(f"Processing image: {args.image}")
        try:
            detections, visualization, inference_time = process_image(
                args.image,
                model,
                model_type,
                class_names,
                args.conf_threshold,
                args.iou_threshold,
                args.device
            )

            # Save visualization
            output_file = os.path.join(args.output, os.path.basename(args.image))
            cv2.imwrite(output_file, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            logger.info(f"Processed image in {inference_time * 1000:.2f} ms ({1.0 / inference_time:.2f} FPS)")
            logger.info(f"Detected {len(detections)} objects")
            logger.info(f"Output saved to {output_file}")

        except Exception as e:
            logger.error(f"Error processing image: {e}")

    elif args.video:
        logger.info(f"Processing video: {args.video}")
        output_file = os.path.join(args.output, os.path.basename(args.video))

        try:
            stats = process_video(
                args.video,
                output_file,
                model,
                model_type,
                class_names,
                args.conf_threshold,
                args.iou_threshold,
                args.device
            )

            logger.info(f"Processed {stats['frames_processed']} frames")
            logger.info(
                f"Average inference time: {stats['average_inference_time'] * 1000:.2f} ms ({stats['average_fps']:.2f} FPS)")
            logger.info(
                f"Detected {stats['total_detections']} objects ({stats['average_detections_per_frame']:.2f} per frame)")
            logger.info(f"Output saved to {output_file}")

        except Exception as e:
            logger.error(f"Error processing video: {e}")

    elif args.dir:
        logger.info(f"Processing images in directory: {args.dir}")

        try:
            stats = process_images_in_directory(
                args.dir,
                args.output,
                model,
                model_type,
                class_names,
                args.conf_threshold,
                args.iou_threshold,
                args.device
            )

            logger.info(f"Processed {stats['images_processed']} images")
            logger.info(
                f"Average inference time: {stats['average_inference_time'] * 1000:.2f} ms ({stats['average_fps']:.2f} FPS)")
            logger.info(
                f"Detected {stats['total_detections']} objects ({stats['average_detections_per_image']:.2f} per image)")
            logger.info(f"Outputs saved to {args.output}")

        except Exception as e:
            logger.error(f"Error processing directory: {e}")

    else:
        logger.error("No input specified. Please provide --image, --video, or --dir.")

    logger.info("Inference completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with object detection model")
    parser.add_argument("--model", type=str, default="models/final/model.onnx", help="Path to model file")
    parser.add_argument("--output", type=str, default="", help="Output directory")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to input image")
    input_group.add_argument("--video", type=str, help="Path to input video")
    input_group.add_argument("--dir", type=str, help="Path to directory containing images")

    args = parser.parse_args()

    main(args)