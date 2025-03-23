#!/usr/bin/env python
"""
Setup script for the object detection project.
This script creates the necessary directory structure and files.
"""

import os
import logging
import shutil
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_directories():
    """Create the necessary directory structure"""
    directories = [
        'config',
        'models/final',
        'models/checkpoints',
        'logs/api',
        'logs/training',
        'logs/evaluation',
        'data/processed',
        'data/raw',
        'output'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def create_empty_init_files():
    """Create empty __init__.py files to make directories importable"""
    src_dirs = [
        'src',
        'src/models',
        'src/data',
        'src/utils',
        'src/api'
    ]

    for directory in src_dirs:
        os.makedirs(directory, exist_ok=True)
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                pass
            logger.info(f"Created init file: {init_file}")


def copy_config_files():
    """Copy configuration files to the correct location"""
    # Check if source config files exist
    original_files = [
        ('config.yaml', 'config/config.yaml'),
        ('data_config.yaml', 'config/data_config.yaml'),
        ('model_config.yaml', 'config/model_config.yaml')
    ]

    for src, dst in original_files:
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            logger.info(f"Copied {src} to {dst}")


def setup_python_modules():
    """Create necessary Python module files if they don't exist"""
    # Create a simple placeholder module if it doesn't exist
    yolo_path = 'src/models/yolo.py'
    if not os.path.exists(yolo_path):
        with open(yolo_path, 'w') as f:
            f.write('''import torch
import torch.nn as nn

class YOLOv5(nn.Module):
    """Placeholder YOLOv5 class"""
    def __init__(self, config=None, num_classes=80, input_channels=3):
        super(YOLOv5, self).__init__()
        # Simple module for testing purposes
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Output is [batch_size, boxes, 5 + num_classes]
        # where 5 is [x, y, w, h, confidence]
        self.head = nn.Conv2d(64, 3 * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)

        # Reshape to match expected output format
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = x.view(batch_size, height * width * 3, -1)  # [B, boxes, 5+classes]

        return x

def create_model(model_config, num_classes=80, input_channels=3, pretrained=True, pretrained_weights=None):
    """Create YOLOv5 model"""
    return YOLOv5(config=model_config, num_classes=num_classes, input_channels=input_channels)
''')
        logger.info(f"Created placeholder YOLO module: {yolo_path}")

    # Create other necessary files
    utils_path = 'src/utils/__init__.py'
    if not os.path.exists(utils_path):
        with open(utils_path, 'w') as f:
            f.write('''import torch
import numpy as np
import os

def convert_model_to_onnx(model, input_size=(416, 416), batch_size=1, opset_version=12, output_path=None):
    """Convert PyTorch model to ONNX format"""
    if output_path is None:
        output_path = "models/final/model.onnx"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    # Create dummy input
    height, width = input_size
    dummy_input = torch.randn(batch_size, 3, height, width)

    # Export model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    return output_path
''')
        logger.info(f"Created placeholder utils module: {utils_path}")


def run_model_converter():
    """Run the ONNX model converter script"""
    try:
        # Import the converter script
        from onnx_converter import main as converter_main
        logger.info("Running ONNX model converter...")
        converter_main()
    except ImportError:
        logger.error("ONNX converter module not found. Make sure onnx_converter.py is in the current directory.")
        # Try to run the script directly
        if os.path.exists('onnx_converter.py'):
            logger.info("Running ONNX converter script directly...")
            os.system(f"{sys.executable} onnx_converter.py")
        else:
            logger.error("onnx_converter.py not found. The ONNX model won't be created.")


def main():
    """Main setup function"""
    logger.info("Starting project setup...")

    # Create directories
    create_directories()

    # Create Python module structure
    create_empty_init_files()

    # Copy configuration files
    copy_config_files()

    # Create Python modules
    setup_python_modules()

    # Run model converter
    run_model_converter()

    logger.info("Setup complete!")
    logger.info("You can now run the API server using: python app.py")


if __name__ == "__main__":
    main()