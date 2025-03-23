import os
import torch
import yaml
import logging
import sys
import requests
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Make sure necessary directories exist
os.makedirs('config', exist_ok=True)
os.makedirs('models/final', exist_ok=True)
os.makedirs('logs', exist_ok=True)


# Create basic config files if they don't exist
def create_default_configs():
    """Create default configuration files if they don't exist"""

    # Default config.yaml
    if not os.path.exists('config/config.yaml'):
        config = {
            'project_name': "custom-object-detection",
            'random_seed': 42,
            'device': "cuda" if torch.cuda.is_available() else "cpu",
            'num_workers': 4,
            'data_dir': "./data",
            'logs_dir': "./logs",
            'models_dir': "./models",
            'output_dir': "./output",
            'training': {
                'batch_size': 16,
                'num_epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 0.0005,
                'optimizer': "adam",
                'lr_scheduler': "cosine",
                'early_stopping_patience': 5
            },
            'evaluation': {
                'batch_size': 8,
                'iou_threshold': 0.45,
                'confidence_threshold': 0.25,
                'max_detections': 100
            },
            'api': {
                'host': "0.0.0.0",
                'port': 8000,
                'max_request_size': 10,
                'timeout': 30,
                'cors_origins': ["*"]
            }
        }

        with open('config/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info("Created default config.yaml")

    # Default data_config.yaml
    if not os.path.exists('config/data_config.yaml'):
        data_config = {
            'dataset': {
                'name': "coco",
                'subset': "2017",
                'classes': [
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                    "parking meter", "bench", "bird", "cat", "dog"
                ],
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1
            },
            'preprocessing': {
                'resize': {
                    'height': 416,
                    'width': 416,
                },
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            }
        }

        with open('config/data_config.yaml', 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        logger.info("Created default data_config.yaml")

    # Default model_config.yaml
    if not os.path.exists('config/model_config.yaml'):
        model_config = {
            'model': {
                'name': "yolov5",
                'version': "s",
                'pretrained': True,
                'input_size': 416,
                'channels': 3,
                'num_classes': 17,  # Matches the number of classes in data_config
                'anchors': [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                'strides': [8, 16, 32],
                'loss': {
                    'box_weight': 0.05,
                    'obj_weight': 1.0,
                    'cls_weight': 0.5,
                    'iou_type': "ciou"
                },
                'nms': {
                    'iou_threshold': 0.45,
                    'score_threshold': 0.25
                }
            },
            'optimization': {
                'export': {
                    'format': "onnx",
                    'optimize': True,
                    'opset_version': 12
                }
            }
        }

        with open('config/model_config.yaml', 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False)
        logger.info("Created default model_config.yaml")


def download_pretrained_weights(version='s', save_path=None):
    """
    Download pretrained YOLOv5 weights

    Args:
        version (str): YOLOv5 version (n, s, m, l, x)
        save_path (str): Path to save the weights

    Returns:
        str: Path to saved weights
    """
    if save_path is None:
        save_path = f"models/pretrained/yolov5{version}.pt"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Check if file already exists
    if os.path.exists(save_path):
        logger.info(f"Pretrained weights already exist at {save_path}")
        return save_path

    # Download weights
    url = f"https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5{version}.pt"
    logger.info(f"Downloading pretrained weights from {url}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(save_path, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

        progress_bar.close()

        logger.info(f"Successfully downloaded pretrained weights to {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Failed to download pretrained weights: {e}")
        return None


def try_import_yolo_module():
    """
    Try to import the YOLO module from the project's source

    Returns:
        tuple: (success, module)
    """
    try:
        # Add project root to path
        sys.path.append('.')

        # Try to import the module
        try:
            from src.models.yolo import create_model
            logger.info("Successfully imported yolo module")
            return True, create_model
        except ImportError as e:
            logger.error(f"Failed to import yolo module: {e}")
            return False, None

    except Exception as e:
        logger.error(f"Error importing yolo module: {e}")
        return False, None


def create_onnx_from_external_weights(version='s', pretrained_path=None):
    """
    Create ONNX model using the official YOLOv5 repository

    Args:
        version (str): YOLOv5 version (n, s, m, l, x)
        pretrained_path (str): Path to pretrained weights

    Returns:
        str: Path to ONNX model
    """
    try:
        # Clone YOLOv5 repository if not already cloned
        yolov5_dir = Path('yolov5')
        if not yolov5_dir.exists():
            logger.info("Cloning YOLOv5 repository...")
            os.system("git clone https://github.com/ultralytics/yolov5.git")

        # Install YOLOv5 dependencies
        requirements_file = yolov5_dir / 'requirements.txt'
        if requirements_file.exists():
            logger.info("Installing YOLOv5 dependencies...")
            os.system(f"pip install -r {requirements_file}")

        # Run export.py from YOLOv5
        if pretrained_path is None:
            pretrained_path = download_pretrained_weights(version)

        if pretrained_path and os.path.exists(pretrained_path):
            output_path = 'models/final/model.onnx'
            logger.info(f"Exporting model to ONNX using YOLOv5 export.py...")

            # Get number of classes from data config
            if os.path.exists('config/data_config.yaml'):
                with open('config/data_config.yaml', 'r') as f:
                    data_config = yaml.safe_load(f)
                    num_classes = len(data_config['dataset']['classes'])
            else:
                num_classes = 80  # Default COCO classes

            # Run export command
            command = f"cd yolov5 && python export.py --weights {os.path.abspath(pretrained_path)} --include onnx --simplify --opset 12"

            # If custom classes, add --data argument
            if num_classes != 80:
                # Create temporary data YAML
                with open('yolov5/data/custom.yaml', 'w') as f:
                    yaml.dump({'nc': num_classes}, f)
                command += " --data data/custom.yaml"

            os.system(command)

            # Move the exported model to the right location
            exported_path = Path(pretrained_path).with_suffix('.onnx')
            if exported_path.exists():
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                os.rename(exported_path, output_path)
                logger.info(f"Moved exported model to {output_path}")
                return output_path
            else:
                logger.error(f"Exported model not found at {exported_path}")
                return None
        else:
            logger.error("Pretrained weights not available")
            return None

    except Exception as e:
        logger.error(f"Error exporting model with YOLOv5 repository: {e}")
        return None


def create_simple_model_onnx():
    """
    Create a simple model and export to ONNX

    Returns:
        str: Path to ONNX model
    """
    try:
        import torch.nn as nn

        class SimpleYOLO(nn.Module):
            def __init__(self, num_classes=17):
                super(SimpleYOLO, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
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

        # Get number of classes
        if os.path.exists('config/data_config.yaml'):
            with open('config/data_config.yaml', 'r') as f:
                data_config = yaml.safe_load(f)
                num_classes = len(data_config['dataset']['classes'])
        else:
            num_classes = 17  # Default

        logger.info(f"Creating simple model with {num_classes} classes...")
        model = SimpleYOLO(num_classes=num_classes)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, 416, 416)

        # Export model to ONNX
        output_path = 'models/final/model.onnx'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info(f"Exporting simple model to ONNX: {output_path}")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        logger.info(f"Simple model exported to ONNX successfully")
        return output_path

    except Exception as e:
        logger.error(f"Failed to create simple ONNX model: {e}")
        return None


def main():
    """Main function to create configs and model"""
    logger.info("Starting model setup process...")

    # Create default configs if they don't exist
    create_default_configs()

    # Check if model file already exists
    model_path = Path('models/final/model.onnx')
    if model_path.exists():
        logger.info(f"ONNX model already exists at {model_path}")
        return str(model_path)

    # Try different methods to create ONNX model

    # Method 1: Try using project's YOLO module
    logger.info("Method 1: Trying to use project's YOLO module...")
    success, create_model_func = try_import_yolo_module()

    if success and create_model_func:
        try:
            # Load model configuration
            with open('config/model_config.yaml', 'r') as f:
                model_config = yaml.safe_load(f)

            # Get number of classes
            with open('config/data_config.yaml', 'r') as f:
                data_config = yaml.safe_load(f)
                num_classes = len(data_config['dataset']['classes'])

            # Create model
            logger.info(f"Creating model with {num_classes} classes...")
            model = create_model_func(model_config, num_classes=num_classes, pretrained=True)

            # Set model to evaluation mode
            model.eval()

            # Create dummy input
            input_size = model_config['model'].get('input_size', 416)
            dummy_input = torch.randn(1, 3, input_size, input_size)

            # Export model to ONNX
            output_path = 'models/final/model.onnx'

            logger.info(f"Exporting model to ONNX: {output_path}")
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )

            logger.info(f"Model exported to ONNX successfully")
            return output_path

        except Exception as e:
            logger.error(f"Error creating model with src.models.yolo: {e}")

    # Method 2: Try using official YOLOv5 repository
    logger.info("Method 2: Trying to use official YOLOv5 repository...")
    onnx_path = create_onnx_from_external_weights()

    if onnx_path and os.path.exists(onnx_path):
        logger.info(f"Successfully created ONNX model using YOLOv5 repository")
        return onnx_path

    # Method 3: Create a simple model
    logger.info("Method 3: Creating a simple model...")
    simple_model_path = create_simple_model_onnx()

    if simple_model_path and os.path.exists(simple_model_path):
        logger.info(f"Successfully created simple ONNX model")
        return simple_model_path

    # Last resort: Create a placeholder file
    logger.warning("All methods failed. Creating a placeholder ONNX file")
    output_path = 'models/final/model.onnx'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        # Just write a placeholder
        f.write(b'ONNX PLACEHOLDER')

    logger.warning(f"Created placeholder ONNX file: {output_path}")
    logger.warning("This is not a valid ONNX model and will not work with the API!")
    logger.warning("You need to train a real model or provide a pre-trained model.")

    return output_path


if __name__ == "__main__":
    main()