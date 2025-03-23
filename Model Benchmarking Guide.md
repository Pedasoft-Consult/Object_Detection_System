# Object Detection Model Benchmarking Guide

This document explains how to use the model benchmarking script to evaluate the performance of your object detection model across different hardware configurations and input parameters.

## Overview

The `benchmark.py` script performs comprehensive benchmarking of your ONNX model, measuring:

- Inference speed (milliseconds per image)
- Throughput (frames per second)
- Performance across different hardware (CPU/GPU)
- Impact of batch size on performance
- Impact of input resolution on performance

## Prerequisites

- Python 3.8+
- ONNX Runtime
- NumPy
- PIL (Pillow)
- Matplotlib
- A trained object detection model in ONNX format

## Usage

### Basic Usage

```bash
python benchmark.py
```

This will run the benchmark with default settings:
- Model: `models/final/model.onnx`
- Batch size: 1
- Input size: 640x480
- 50 benchmark runs with 10 warmup runs
- Automatic provider selection (CUDA if available, otherwise CPU)

### Advanced Options

```bash
python benchmark.py --model models/final/model.onnx \
                   --providers cpu cuda \
                   --batch_sizes 1 4 8 16 \
                   --input_sizes 416x416 640x480 1280x720 \
                   --runs 100 \
                   --warmup 20 \
                   --output_dir benchmark_results
```

### Command Line Arguments

- `--model`: Path to the ONNX model file (default: `models/final/model.onnx`)
- `--providers`: Execution providers to benchmark (`cpu`, `cuda`, or `auto`) (default: `auto`)
- `--batch_sizes`: Batch sizes to test (default: `[1]`)
- `--input_sizes`: Input image sizes in WxH format (default: `[640x480]`)
- `--runs`: Number of benchmark iterations (default: `50`)
- `--warmup`: Number of warmup iterations (default: `10`)
- `--output_dir`: Directory to save benchmark results (default: `benchmark_results`)

## Output

The benchmark script generates:

1. **CSV Results File**: A detailed table with all benchmark measurements
2. **Performance Charts**:
   - Inference time by batch size
   - FPS by batch size
   - Inference time by input size
   - FPS by input size
   - Distribution of inference times
3. **Console Output**: Summary of benchmark configuration and key results

## Understanding the Results

### CSV Output Format

The CSV file contains the following columns:

- `provider`: Execution provider (CPU/CUDA)
- `batch_size`: Number of images processed in parallel
- `input_size`: Input image dimensions (WxH)
- `avg_time_ms`: Average inference time in milliseconds
- `min_time_ms`: Minimum inference time in milliseconds
- `max_time_ms`: Maximum inference time in milliseconds
- `median_time_ms`: Median inference time in milliseconds
- `fps`: Frames per second (throughput)

### Key Performance Metrics

- **Inference Time**: Lower is better. This is the time required to process one batch.
- **FPS (Frames Per Second)**: Higher is better. This is throughput, calculated as `batch_size / (avg_time_ms / 1000)`.
- **Scaling Efficiency**: How FPS scales with batch size. Ideally, it should scale linearly.

## Example Results Interpretation

```
Provider: CUDA, batch=1, size=640x480, time=15.23ms, fps=65.66
Provider: CUDA, batch=4, size=640x480, time=32.45ms, fps=123.27
Provider: CPU, batch=1, size=640x480, time=120.34ms, fps=8.31
```

This example shows:
1. CUDA (GPU) is about 7.9x faster than CPU for batch size 1
2. Increasing batch size from 1 to 4 on GPU improves throughput by 1.88x (not quite linear scaling)

## Optimizing Model Performance

Based on benchmark results, you can optimize your model by:

1. **Choosing the right hardware**: If CUDA shows significant speedup, prioritize GPU deployment
2. **Finding optimal batch size**: Look for the batch size with highest FPS without excessive latency
3. **Selecting input resolution**: Balance between accuracy and speed based on your use case
4. **Model quantization**: If performance is still insufficient, consider quantizing your model to INT8

## Advanced Analysis

For deeper analysis, the benchmark script saves all individual inference times in the JSON results, which you can use for:

- Calculating percentiles (p95, p99) for latency-sensitive applications
- Analyzing performance variability
- Comparing performance stability across different hardware

## Troubleshooting

- **"CUDA provider not available"**: Ensure you have a compatible GPU and CUDA installation
- **Out of memory errors**: Reduce batch size or input resolution
- **High variability in results**: Increase the number of runs for more stable measurements
- **Error loading model**: Verify the model path and that it's a valid ONNX file

## Next Steps

After benchmarking, you might want to:

1. Export the model in different formats (TensorRT, OpenVINO) for better performance
2. Apply quantization to reduce model size and improve speed
3. Test the model on edge devices or production servers
4. Use the performance data to set expectations for your API response times

## Complete Benchmarking Script

The following script is a comprehensive tool for benchmarking ONNX models. It handles:

1. Loading models with different providers (CPU/CUDA)
2. Generating test images of various sizes
3. Measuring inference time across multiple runs
4. Creating visualization charts for analysis
5. Exporting detailed results in CSV format

```python
#!/usr/bin/env python
"""
Benchmarking script for the object detection model.

This script performs benchmarking of model inference performance
on various devices and with different batch sizes.
"""

import os
import time
import argparse
import yaml
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw
import random
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def generate_test_image(width=640, height=480, with_objects=True):
    """
    Generate a test image for benchmarking
    
    Args:
        width (int): Image width
        height (int): Image height
        with_objects (bool): Whether to draw objects on the image
        
    Returns:
        PIL.Image: Test image
    """
    # Create blank image
    image = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    
    if with_objects:
        # Draw some "objects" (rectangles) that the detector might find
        num_objects = random.randint(1, 5)
        
        for _ in range(num_objects):
            # Random rectangle
            x1 = random.randint(50, width - 150)
            y1 = random.randint(50, height - 150)
            x2 = x1 + random.randint(50, 150)
            y2 = y1 + random.randint(50, 150)
            
            # Random color
            color = (
                random.randint(0, 200),
                random.randint(0, 200),
                random.randint(0, 200)
            )
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw some internal details
            if random.random() > 0.5:
                draw.line([x1, y1, x2, y2], fill=color, width=1)
            else:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = min(x2 - x1, y2 - y1) // 4
                draw.ellipse([center_x - radius, center_y - radius, 
                              center_x + radius, center_y + radius], 
                             outline=color, width=1)
    
    return image


def preprocess_image(image, input_height, input_width):
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image
        input_height (int): Model input height
        input_width (int): Model input width
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Resize image to match model input dimensions
    if image_np.shape[1] != input_width or image_np.shape[0] != input_height:
        # Using numpy's resize to maintain compatibility
        resized = np.array(Image.fromarray(image_np).resize((input_width, input_height)))
    else:
        resized = image_np
    
    # Normalize pixel values to 0-1
    normalized = resized / 255.0
    
    # Transpose to match the expected input format [C, H, W]
    transposed = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension
    batched = np.expand_dims(transposed, 0).astype(np.float32)
    
    return batched


def load_model(model_path, provider=None):
    """
    Load ONNX model
    
    Args:
        model_path (str): Path to model file
        provider (str): Execution provider (CPU/CUDA)
        
    Returns:
        tuple: (session, input_name, output_name, input_shape)
    """
    # Determine providers
    if provider == "cpu":
        providers = ['CPUExecutionProvider']
    elif provider == "cuda":
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            logger.warning("CUDA provider not available, falling back to CPU")
            providers = ['CPUExecutionProvider']
    else:
        # Auto-select
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    
    logger.info(f"Using providers: {providers}")
    
    # Load model
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Get model metadata
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    return session, input_name, output_name, input_shape


def benchmark_inference(session, input_name, input_batch, num_runs=100, warmup_runs=10):
    """
    Benchmark inference performance
    
    Args:
        session: ONNX session
        input_name (str): Model input name
        input_batch (numpy.ndarray): Input batch
        num_runs (int): Number of benchmark runs
        warmup_runs (int): Number of warmup runs
        
    Returns:
        dict: Benchmarking results
    """
    # Warmup runs
    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: input_batch})
    
    # Benchmark runs
    inference_times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        _ = session.run(None, {input_name: input_batch})
        end_time = time.time()
        inference_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    median_time = sorted(inference_times)[len(inference_times) // 2]
    fps = 1000 / avg_time
    
    return {
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'median_time_ms': median_time,
        'fps': fps,
        'all_times_ms': inference_times
    }


def save_benchmark_results(results, output_dir="benchmark_results"):
    """
    Save benchmark results
    
    Args:
        results (dict): Benchmark results
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV with results
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['provider', 'batch_size', 'input_size', 'avg_time_ms', 'min_time_ms', 'max_time_ms', 'median_time_ms', 'fps']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'provider': result['provider'],
                'batch_size': result['batch_size'],
                'input_size': f"{result['input_width']}x{result['input_height']}",
                'avg_time_ms': f"{result['avg_time_ms']:.2f}",
                'min_time_ms': f"{result['min_time_ms']:.2f}",
                'max_time_ms': f"{result['max_time_ms']:.2f}",
                'median_time_ms': f"{result['median_time_ms']:.2f}",
                'fps': f"{result['fps']:.2f}"
            })
    
    logger.info(f"Results saved to {csv_path}")
    
    # Create comparison charts
    plt.figure(figsize=(12, 8))
    
    # Group by provider
    providers = set(result['provider'] for result in results)
    batch_sizes = sorted(set(result['batch_size'] for result in results))
    input_sizes = set(f"{result['input_width']}x{result['input_height']}" for result in results)
    
    # Plot average inference time by batch size for each provider
    plt.subplot(2, 1, 1)
    for provider in providers:
        provider_results = [r for r in results if r['provider'] == provider]
        if provider_results:
            x = [r['batch_size'] for r in provider_results if r['input_width'] == provider_results[0]['input_width']]
            y = [r['avg_time_ms'] for r in provider_results if r['input_width'] == provider_results[0]['input_width']]
            if x and y:
                plt.plot(x, y, marker='o', label=provider)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Average Inference Time (ms)')
    plt.title('Inference Time by Batch Size')
    plt.grid(True)
    plt.legend()
    
    # Plot FPS by batch size for each provider
    plt.subplot(2, 1, 2)
    for provider in providers:
        provider_results = [r for r in results if r['provider'] == provider]
        if provider_results:
            x = [r['batch_size'] for r in provider_results if r['input_width'] == provider_results[0]['input_width']]
            y = [r['fps'] for r in provider_results if r['input_width'] == provider_results[0]['input_width']]
            if x and y:
                plt.plot(x, y, marker='o', label=provider)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Frames Per Second (FPS)')
    plt.title('Performance by Batch Size')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_by_batch_size.png"))
    
    # If we have multiple input sizes, create chart for input size comparison
    if len(input_sizes) > 1:
        plt.figure(figsize=(12, 8))
        
        # Use the most common batch size
        common_batch_size = max(batch_sizes, key=lambda x: len([r for r in results if r['batch_size'] == x]))
        
        # Plot average inference time by input size for each provider
        plt.subplot(2, 1, 1)
        for provider in providers:
            provider_results = [r for r in results if r['provider'] == provider and r['batch_size'] == common_batch_size]
            if provider_results:
                x = [f"{r['input_width']}x{r['input_height']}" for r in provider_results]
                y = [r['avg_time_ms'] for r in provider_results]
                if x and y:
                    plt.plot(x, y, marker='o', label=provider)
        
        plt.xlabel('Input Size')
        plt.ylabel('Average Inference Time (ms)')
        plt.title(f'Inference Time by Input Size (Batch Size: {common_batch_size})')
        plt.grid(True)
        plt.legend()
        
        # Plot FPS by input size for each provider
        plt.subplot(2, 1, 2)
        for provider in providers:
            provider_results = [r for r in results if r['provider'] == provider and r['batch_size'] == common_batch_size]
            if provider_results:
                x = [f"{r['input_width']}x{r['input_height']}" for r in provider_results]
                y = [r['fps'] for r in provider_results]
                if x and y:
                    plt.plot(x, y, marker='o', label=provider)
        
        plt.xlabel('Input Size')
        plt.ylabel('Frames Per Second (FPS)')
        plt.title(f'Performance by Input Size (Batch Size: {common_batch_size})')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "benchmark_by_input_size.png"))
    
    # Create histogram of inference times
    plt.figure(figsize=(12, 6))
    for provider in providers:
        provider_results = [r for r in results if r['provider'] == provider]
        if provider_results and len(provider_results) > 0:
            # Use the first result's times
            plt.hist(provider_results[0]['all_times_ms'], bins=20, alpha=0.5, label=provider)
    
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Inference Times')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "inference_time_distribution.png"))
    
    logger.info(f"Charts saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark object detection model')
    parser.add_argument('--model', type=str, default='models/final/model.onnx', help='Path to ONNX model')
    parser.add_argument('--providers', type=str, nargs='+', default=['auto'], choices=['auto', 'cpu', 'cuda'], help='Execution providers to benchmark')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1], help='Batch sizes to benchmark')
    parser.add_argument('--input_sizes', type=str, nargs='+', default=['640x480'], help='Input sizes to benchmark (WxH)')
    parser.add_argument('--runs', type=int, default=50, help='Number of benchmark runs')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--output_dir', type=str, default='benchmark_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Verify model path
    model_path = args.model
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Parse providers
    if 'auto' in args.providers:
        providers = ['cuda', 'cpu'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['cpu']
    else:
        providers = args.providers
    
    # Parse input sizes
    input_sizes = []
    for size_str in args.input_sizes:
        try:
            width, height = map(int, size_str.split('x'))
            input_sizes.append((width, height))
        except ValueError:
            logger.error(f"Invalid input size format: {size_str}. Use WxH format (e.g., 640x480)")
            return
    
    logger.info(f"Benchmarking model: {model_path}")
    logger.info(f"Providers: {providers}")
    logger.info(f"Batch sizes: {args.batch_sizes}")
    logger.info(f"Input sizes: {args.input_sizes}")
    logger.info(f"Runs: {args.runs}")
    
    results = []
    
    # Run benchmarks for each configuration
    for provider in providers:
        logger.info(f"Loading model with provider: {provider}")
        try:
            session, input_name, output_name, model_input_shape = load_model(model_path, provider)
            
            # Extract model dimensions from the first run
            _, channels, model_height, model_width = model_input_shape
            
            for input_width, input_height in input_sizes:
                logger.info(f"Preparing input size: {input_width}x{input_height}")
                
                # Generate a test image and preprocess it
                test_image = generate_test_image(width=input_width, height=input_height)
                single_input = preprocess_image(test_image, model_height, model_width)
                
                for batch_size in args.batch_sizes:
                    logger.info(f"Benchmarking batch size: {batch_size}")
                    
                    # Create batch
                    if batch_size == 1:
                        batch = single_input
                    else:
                        batch = np.repeat(single_input, batch_size, axis=0)
                    
                    # Run benchmark
                    benchmark_result = benchmark_inference(
                        session, 
                        input_name, 
                        batch, 
                        num_runs=args.runs, 
                        warmup_runs=args.warmup
                    )
                    
                    # Add metadata to results
                    result = {
                        'provider': provider,
                        'batch_size': batch_size,
                        'input_width': input_width,
                        'input_height': input_height,
                        'model_width': model_width,
                        'model_height': model_height,
                        **benchmark_result
                    }
                    
                    results.append(result)
                    
                    # Log result
                    logger.info(f"Result: {provider}, batch={batch_size}, size={input_width}x{input_height}, "
                                f"time={result['avg_time_ms']:.2f}ms, fps={result['fps']:.2f}")
        
        except Exception as e:
            logger.error(f"Error benchmarking with provider {provider}: {e}")
    
    # Save results
    if results:
        save_benchmark_results(results, args.output_dir)
    else:
        logger.error("No benchmark results to save")


if __name__ == "__main__":
    main()
```

## Conclusion

Benchmarking is a critical step in deploying object detection models in production. It helps you understand performance characteristics, identify bottlenecks, and make informed decisions about hardware requirements and configuration settings. With the comprehensive benchmarking script provided here, you can thoroughly evaluate your model's performance across various scenarios.