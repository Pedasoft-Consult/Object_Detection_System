# Custom Object Detection System

This repository contains a custom object detection system using YOLOv5 architecture. The system includes a FastAPI server for deployment and inference.

## Project Structure

```
.
├── app.py                      # FastAPI application
├── config/                     # Configuration files
│   ├── config.yaml             # Main configuration
│   ├── data_config.yaml        # Dataset configuration
│   └── model_config.yaml       # Model configuration
├── evaluate.py                 # Model evaluation script
├── inference.py                # Inference script
├── logs/                       # Log files
├── models/                     # Model files
│   ├── checkpoints/            # Model checkpoints during training
│   └── final/                  # Final model for deployment
│       └── model.onnx          # ONNX model for inference
├── onnx_converter.py           # Script to convert model to ONNX
├── setup.py                    # Setup script
├── src/                        # Source code
│   ├── api/                    # API related code
│   ├── data/                   # Data loading and processing
│   ├── models/                 # Model architecture definitions
│   │   ├── loss.py             # Loss function
│   │   └── yolo.py             # YOLO model implementation
│   └── utils/                  # Utility functions
└── train.py                    # Training script
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- ONNX Runtime
- FastAPI

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/custom-object-detection.git
cd custom-object-detection
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the setup script to create necessary directories and files:
```bash
python setup.py
```

### Using the API

1. Start the API server:
```bash
python app.py
```

2. The API will be available at http://localhost:8000

3. Access the API documentation at http://localhost:8000/docs

### API Endpoints

- `GET /`: Root endpoint with information about the API
- `GET /health`: Health check endpoint
- `POST /predict`: Detect objects in an uploaded image
- `POST /batch_predict`: Detect objects in multiple uploaded images

### Example API Usage

```python
import requests

# Single image prediction
url = "http://localhost:8000/predict"
files = {"file": open("test.jpg", "rb")}
params = {"conf_threshold": 0.3, "max_detections": 10}
response = requests.post(url, files=files, params=params)
predictions = response.json()

# Process predictions
for detection in predictions["detections"]:
    print(f"Found: {detection['class_name']} with confidence {detection['confidence']}")
    print(f"Bounding box: {detection['bbox']}")
```

## Training Your Own Model

1. Prepare your dataset according to the format in data_config.yaml

2. Train the model:
```bash
python train.py --config config/config.yaml
```

3. Evaluate the model:
```bash
python evaluate.py --config config/config.yaml
```

4. Convert the model to ONNX for deployment:
```bash
python onnx_converter.py
```

## Configuration

You can customize the system by modifying the configuration files:

- `config.yaml`: General settings like paths, devices, etc.
- `data_config.yaml`: Dataset configuration, classes, preprocessing, etc.
- `model_config.yaml`: Model architecture, detection settings, etc.

## Troubleshooting

If you encounter issues with the API:

1. Check the logs in the `logs/api` directory
2. Ensure the model file exists at `models/final/model.onnx`
3. Verify that the configuration files are correctly formatted

If the API starts with a mock model, it means the real model file was not found. You should train a model or download a pre-trained one.

## License

This project is licensed under the MIT License - see the LICENSE file for details.