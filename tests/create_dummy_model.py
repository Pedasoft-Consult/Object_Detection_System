#!/usr/bin/env python
"""
Script to create a dummy ONNX model for testing purposes.
This script creates a minimal ONNX model that can be used for tests
when the real model is not available.
"""

import os
import argparse
import torch
import torch.nn as nn
import logging
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class DummyYOLOModel(nn.Module):
    """
    A simplified YOLO-like model structure that can be exported to ONNX
    """

    def __init__(self, num_classes=20, input_size=416):
        super(DummyYOLOModel, self).__init__()

        # Feature extraction layers (simplified backbone)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Detection head (output: bbox, objectness, class scores)
        # 5 + num_classes outputs per anchor box, 3 anchor boxes per feature map
        self.head = nn.Conv2d(512, 3 * (5 + num_classes), kernel_size=1)

        # Calculate feature map size based on input size
        self.fm_size = input_size // 32

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.features(x)
        output = self.head(features)

        # Reshape output to match YOLO format
        # [batch_size, num_anchors * (5 + num_classes), height, width] ->
        # [batch_size, num_anchors, height, width, 5 + num_classes]
        batch_size = x.size(0)
        anchor_out = output.view(batch_size, 3, -1, self.fm_size, self.fm_size)
        anchor_out = anchor_out.permute(0, 1, 3, 4, 2).contiguous()

        # Flatten all detections
        # [batch_size, num_anchors * height * width, 5 + num_classes]
        return anchor_out.view(batch_size, -1, anchor_out.size(-1))


def create_dummy_model(model_path, input_size=416, num_classes=20):
    """
    Create and save a dummy YOLO model in ONNX format

    Args:
        model_path (str): Path to save the model
        input_size (int): Input image size
        num_classes (int): Number of classes
    """
    # Create directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    # Create model
    model = DummyYOLOModel(num_classes=num_classes, input_size=input_size)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Export model to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            model_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        logger.info(f"Dummy model successfully created and saved to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating dummy model: {e}")
        return False


def load_config():
    """
    Load configuration files

    Returns:
        tuple: (config, data_config, model_config)
    """
    config = {}
    data_config = {}
    model_config = {}

    # Make sure config directory exists
    os.makedirs("config", exist_ok=True)

    # Try to load main config
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Main configuration file not found, using defaults")
        config = {
            "project_name": "custom-object-detection",
            "device": "cuda",
            "models_dir": "./models"
        }

    # Try to load data config
    try:
        with open("config/data_config.yaml", "r") as f:
            data_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Data configuration file not found, using defaults")
        data_config = {
            "dataset": {
                "classes": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                            "bird", "cat", "dog", "horse", "sheep", "cow"]
            },
            "preprocessing": {
                "resize": {"height": 416, "width": 416}
            }
        }

    # Try to load model config
    try:
        with open("config/model_config.yaml", "r") as f:
            model_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Model configuration file not found, using defaults")
        model_config = {
            "model": {
                "name": "yolov5",
                "version": "s",
                "input_size": 416,
                "num_classes": 20
            }
        }

    return config, data_config, model_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dummy ONNX model for testing")
    parser.add_argument("--output", type=str, default="models/final/model.onnx",
                        help="Path to save the model")
    parser.add_argument("--input-size", type=int, default=416,
                        help="Input image size")
    parser.add_argument("--num-classes", type=int, default=20,
                        help="Number of classes")

    args = parser.parse_args()

    # Load configurations to get default values
    config, data_config, model_config = load_config()

    # Override with config values if they exist
    if "preprocessing" in data_config and "resize" in data_config["preprocessing"]:
        input_height = data_config["preprocessing"]["resize"].get("height", args.input_size)
        input_width = data_config["preprocessing"]["resize"].get("width", args.input_size)
        input_size = input_height  # Assuming square input
    else:
        input_size = args.input_size

    if "dataset" in data_config and "classes" in data_config["dataset"]:
        num_classes = len(data_config["dataset"]["classes"])
    elif "model" in model_config and "num_classes" in model_config["model"]:
        num_classes = model_config["model"]["num_classes"]
    else:
        num_classes = args.num_classes

    # Create model
    success = create_dummy_model(
        args.output,
        input_size=input_size,
        num_classes=num_classes
    )

    if success:
        print(f"Dummy model created successfully at {args.output}")
    else:
        print("Failed to create dummy model")