# Object Detection Test Suite

This test suite provides comprehensive testing for the object detection system. It includes tests for the API, data loading, model inference, and performance benchmarking.

## Quick Start

To run all tests and generate a report:

```bash
python run_all_tests.py
```

For more options, you can specify:

```bash
python run_all_tests.py --skip-api --skip-benchmark --output-dir="custom_results"
```

## Setup

Before running tests, the setup script will:

1. Create necessary directories
2. Set up configuration files
3. Create a dummy model if needed

You can run setup manually:

```bash
python setup_tests.py
```

## Test Components

The test suite consists of several components:

### 1. API Tests (`test_api.py`)

Tests the REST API endpoints for object detection, including:
- Health check
- Image prediction
- Batch prediction
- Error handling
- Stress testing

### 2. Data Tests (`test_data.py`)

Tests data loading and preprocessing functionality:
- Dataset loading
- Image preprocessing
- Data augmentation
- Batch creation
- Dataset splitting

### 3. Model Tests (`test_model.py`)

Tests the object detection model:
- Model loading
- Inference
- Performance
- Confidence thresholds
- Batch processing
- Input robustness

### 4. Benchmark (`benchmark.py`)

Measures model performance:
- Inference time
- Throughput (FPS)
- Input size impact
- Hardware acceleration

## Configuration

The test suite uses three main configuration files:

1. `config/config.yaml` - Main configuration
2. `config/data_config.yaml` - Data loading configuration
3. `config/model_config.yaml` - Model architecture configuration

## Dummy Model

If the real model file is not available, the test suite will create a dummy ONNX model for testing.

You can create this model manually:

```bash
python create_dummy_model.py --output="models/final/model.onnx"
```

## Reports

After running tests, a report is generated in the output directory (default: `test_results/test_report.html`).

The report includes:
- Test summary
- Detailed test results
- Benchmark metrics
- Performance charts

## Common Issues and Solutions

### Model File Not Found

If you see this error:
```
WARNING - Model file not found: models/final/model.onnx
```

Solution:
1. Run the setup script: `python setup_tests.py`
2. Or create the dummy model: `python create_dummy_model.py`

### API Tests Failing

If API tests are failing with connection errors:

Solution:
1. Make sure the API server is running at the specified URL
2. Skip API tests if server is not available: `python run_all_tests.py --skip-api`

### Configuration Files Missing

Solution:
1. Run the setup script: `python setup_tests.py`
2. Or copy the provided configuration files to the `config/` directory

## Advanced Usage

### Running Individual Tests

You can run individual test scripts:

```bash
python test_data.py
python test_model.py
```

### Customizing Benchmark

The benchmark script has several options:

```bash
python benchmark.py --runs=10 --warmup=5 --input-sizes="416x416,640x640" --device="cpu"
```

### Custom Report Generation

You can generate a report from individual test results:

```bash
python run_all_tests.py --individual
```

## Additional Resources

- [YOLO Object Detection Documentation](https://github.com/ultralytics/yolov5/wiki)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## Troubleshooting

If you encounter issues:

1. Check log files in the output directory
2. Verify configuration files are correct
3. Ensure all dependencies are installed
4. Try running with the `--skip-setup` flag

## Dependencies

- Python 3.8+
- PyTorch
- ONNX Runtime
- NumPy
- Pillow
- tqdm
- matplotlib

Install dependencies with:

```bash
pip install torch torchvision onnx onnxruntime numpy pillow tqdm matplotlib
```