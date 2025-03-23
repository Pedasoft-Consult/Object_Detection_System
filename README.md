# Custom Object Detection System

A complete end-to-end object detection system that can detect and classify objects in images, optimize the model for speed and accuracy, and deploy it to a cloud-based infrastructure.

## Project Overview

This project implements a custom object detection system using YOLO (You Only Look Once) architecture. The system includes:

- Data preprocessing and loading
- Model training and evaluation
- API deployment with FastAPI
- Web interface with React
- Containerization with Docker
- CI/CD pipeline with GitHub Actions
- Cloud deployment

## Project Structure

```
object_detection_project/
├── config/                    # Configuration files
│   ├── config.yaml            # Main configuration
│   ├── model_config.yaml      # Model configuration
│   └── data_config.yaml       # Dataset configuration
├── src/                       # Source code
│   ├── data/                  # Data processing
│   ├── models/                # Model implementation
│   ├── utils/                 # Utility functions
│   └── api/                   # API implementation
├── logs/                      # Log files
├── data/                      # Dataset files
├── models/                    # Model checkpoints
├── frontend/                  # Web interface
├── tests/                     # Unit tests
├── .github/workflows/         # CI/CD pipeline
├── Dockerfile                 # Docker configuration
└── docker-compose.yaml        # Multi-container setup
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended for training)
- Docker and Docker Compose
- Node.js and npm

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/object-detection-project.git
cd object-detection-project
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
cd ..
```

### Data Preparation

1. Download the dataset (COCO, Open Images, etc.)
2. Update `config/data_config.yaml` with the dataset path
3. Preprocess the data:
```bash
python -m src.data.preprocess --config config/data_config.yaml
```

### Training

Train the object detection model:
```bash
python -m src.train --config config/config.yaml
```

The training script will:
- Load and preprocess the data
- Train the YOLO model
- Evaluate the model performance
- Save model checkpoints and export to ONNX

### Evaluation

Evaluate the trained model:
```bash
python -m src.evaluate --config config/config.yaml --model models/final/model.onnx
```

### Running the API

Start the API server:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Running the Frontend

Start the frontend development server:
```bash
cd frontend
npm start
```

Visit `http://localhost:3000` to access the web interface.

### Docker Deployment

Build and run the containers:
```bash
docker-compose up -d
```

This will start:
- The API service on port 8000
- The frontend service on port 80

## API Documentation

The API documentation is available at `http://localhost:8000/docs` when the API is running.

### Endpoints

- `GET /health`: Health check endpoint
- `POST /predict`: Detect objects in an image
- `POST /batch_predict`: Detect objects in multiple images

Example usage:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg" \
  -F "conf_threshold=0.25" \
  -F "max_detections=100"
```

## Model Optimization

The model can be optimized using various techniques:

- **Quantization**: Reduce model size and increase inference speed
- **Pruning**: Remove less important parameters
- **ONNX Export**: Convert the model to ONNX format for efficient inference

To optimize the model:
```bash
python -m src.optimize --config config/model_config.yaml --model models/final/model.pt
```

## Deployment

### Local Deployment with Docker

```bash
docker-compose up -d
```

### Cloud Deployment

#### AWS

1. Update AWS configuration in `.aws/` directory
2. Run the deployment script:
```bash
./scripts/deploy_aws.sh
```

#### Google Cloud Platform

1. Update GCP configuration in `.gcp/` directory
2. Run the deployment script:
```bash
./scripts/deploy_gcp.sh
```

## CI/CD Pipeline

The CI/CD pipeline is configured with GitHub Actions in `.github/workflows/`.

The pipeline includes:
- Code validation and testing
- Building Docker images
- Deployment to cloud platforms

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv5 by Ultralytics
- COCO dataset
- FastAPI
- React