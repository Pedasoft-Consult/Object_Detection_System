#!/usr/bin/env python
"""
Benchmark script for the object detection model.

This script measures model performance across various metrics.
"""

import os
import sys
import time
import argparse
import numpy as np
import logging
import json
import csv
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import onnxruntime as ort
import cv2
from pathlib import Path

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

            # Draw some internal details (lines, circles) to make it more recognizable
            if random.random() > 0.5:
                draw.line([x1, y1, x2, y2], fill=color, width=1)
                draw.line([x1, y2, x2, y1], fill=color, width=1)
            else:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = min(x2 - x1, y2 - y1) // 4
                draw.ellipse([center_x - radius, center_y - radius,
                              center_x + radius, center_y + radius],
                             outline=color, width=1)

    return image


def preprocess_image(image, input_shape):
    """
    Preprocess image for model input

    Args:
        image: PIL Image
        input_shape: Model input shape

    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Get input dimensions from model
    _, input_channels, input_height, input_width = input_shape

    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # Resize image to match model input dimensions
    resized = cv2.resize(image_np, (input_width, input_height))

    # Normalize pixel values to 0-1
    normalized = resized / 255.0

    # Transpose to match the expected input format [C, H, W]
    transposed = np.transpose(normalized, (2, 0, 1))

    # Add batch dimension
    batched = np.expand_dims(transposed, 0).astype(np.float32)

    return batched


def run_benchmark(model_path, num_runs=10, warmup_runs=3, batch_size=1, input_sizes=None, output_dir=None,
                  device="cuda"):
    """
    Run benchmark on model

    Args:
        model_path (str): Path to ONNX model
        num_runs (int): Number of benchmark runs
        warmup_runs (int): Number of warmup runs
        batch_size (int): Batch size for inference
        input_sizes (list): List of input sizes to test
        output_dir (str): Directory to save output

    Returns:
        dict: Benchmark results
    """
    logger.info(f"Running benchmark on model: {model_path}")
    logger.info(f"Number of runs: {num_runs}")
    logger.info(f"Warmup runs: {warmup_runs}")

    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load model
    try:
        # Set providers based on device
        if device.lower() == "cuda" and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        logger.info(f"Using providers: {providers}")

        session = ort.InferenceSession(model_path, providers=providers)

        # Get model metadata
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape

        logger.info(f"Model loaded successfully. Input shape: {input_shape}")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return {"error": str(e)}

    # If no input sizes specified, use model's default input dimensions
    if not input_sizes:
        _, _, height, width = input_shape
        input_sizes = [(width, height)]

    # Results dictionary
    results = {
        "model_path": model_path,
        "providers": providers,
        "batch_size": batch_size,
        "num_runs": num_runs,
        "input_sizes": input_sizes,
        "inference_times": {},
        "throughput": {}
    }

    # Run benchmark for each input size
    for size in input_sizes:
        width, height = size
        logger.info(f"Testing input size: {width}x{height}")

        # Generate test image
        test_image = generate_test_image(width=width, height=height, with_objects=True)

        # Save the test image for reference
        if output_dir:
            test_image_path = os.path.join(output_dir, f"test_image_{width}x{height}.jpg")
            test_image.save(test_image_path)
            logger.info(f"Saved test image to {test_image_path}")

        # Preprocess image
        input_data = preprocess_image(test_image, input_shape)

        # Warmup runs
        logger.info(f"Performing {warmup_runs} warmup runs...")
        for _ in range(warmup_runs):
            _ = session.run(None, {input_name: input_data})

        # Benchmark runs
        logger.info(f"Performing {num_runs} benchmark runs...")
        times = []

        for i in range(num_runs):
            start_time = time.time()
            _ = session.run(None, {input_name: input_data})
            end_time = time.time()
            inference_time = end_time - start_time
            times.append(inference_time)
            logger.info(f"Run {i + 1}/{num_runs}: {inference_time * 1000:.2f} ms")

        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        median_time = np.median(times)
        p95_time = np.percentile(times, 95)
        throughput = 1.0 / mean_time

        size_key = f"{width}x{height}"
        results["inference_times"][size_key] = {
            "mean_ms": mean_time * 1000,
            "std_ms": std_time * 1000,
            "min_ms": min_time * 1000,
            "max_ms": max_time * 1000,
            "median_ms": median_time * 1000,
            "p95_ms": p95_time * 1000,
            "individual_runs_ms": [t * 1000 for t in times]
        }
        results["throughput"][size_key] = throughput

        logger.info(f"Results for {width}x{height}:")
        logger.info(f"Mean inference time: {mean_time * 1000:.2f} ms")
        logger.info(f"Standard deviation: {std_time * 1000:.2f} ms")
        logger.info(f"Min inference time: {min_time * 1000:.2f} ms")
        logger.info(f"Max inference time: {max_time * 1000:.2f} ms")
        logger.info(f"Throughput: {throughput:.2f} FPS")

    # Save results if output directory is specified
    if output_dir:
        # Save JSON results
        json_path = os.path.join(output_dir, "benchmark_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {json_path}")

        # Save CSV results
        csv_path = os.path.join(output_dir, "benchmark_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Input Size", "Mean (ms)", "Std (ms)", "Min (ms)", "Max (ms)", "Median (ms)", "95th Percentile (ms)",
                 "Throughput (FPS)"])

            for size_key in results["inference_times"]:
                writer.writerow([
                    size_key,
                    results["inference_times"][size_key]["mean_ms"],
                    results["inference_times"][size_key]["std_ms"],
                    results["inference_times"][size_key]["min_ms"],
                    results["inference_times"][size_key]["max_ms"],
                    results["inference_times"][size_key]["median_ms"],
                    results["inference_times"][size_key]["p95_ms"],
                    results["throughput"][size_key]
                ])
        logger.info(f"CSV results saved to {csv_path}")

        # Generate charts
        plt.figure(figsize=(10, 6))
        plt.barh(list(results["inference_times"].keys()),
                 [results["inference_times"][k]["mean_ms"] for k in results["inference_times"]])
        plt.xlabel("Inference Time (ms)")
        plt.ylabel("Input Size")
        plt.title("Mean Inference Time by Input Size")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "inference_time_chart.png"))

        plt.figure(figsize=(10, 6))
        plt.barh(list(results["throughput"].keys()),
                 [results["throughput"][k] for k in results["throughput"]])
        plt.xlabel("Throughput (FPS)")
        plt.ylabel("Input Size")
        plt.title("Throughput by Input Size")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "throughput_chart.png"))

        logger.info("Charts generated")

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark object detection model')
    parser.add_argument('--model', type=str, default="models/final/model.onnx", help='Path to ONNX model')
    parser.add_argument('--runs', type=int, default=10, help='Number of benchmark runs')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup runs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--output-dir', type=str, default="benchmark_results", help='Output directory for results')
    parser.add_argument('--input-sizes', type=str, default="416x416,640x640",
                        help='Comma-separated list of input sizes (WxH)')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run on (cuda or cpu)')

    args = parser.parse_args()

    # Parse input sizes
    input_sizes = []
    for size_str in args.input_sizes.split(','):
        width, height = map(int, size_str.split('x'))
        input_sizes.append((width, height))

    # Run benchmark
    run_benchmark(
        model_path=args.model,
        num_runs=args.runs,
        warmup_runs=args.warmup,
        batch_size=args.batch_size,
        input_sizes=input_sizes,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()