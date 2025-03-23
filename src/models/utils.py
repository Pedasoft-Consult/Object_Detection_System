import torch
import numpy as np
import onnx
import os
import yaml
import logging
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger(__name__)


def convert_model_to_onnx(model, input_size=(416, 416), batch_size=1, opset_version=12, output_path=None):
    """
    Convert PyTorch model to ONNX format

    Args:
        model: PyTorch model
        input_size (tuple): Input size (height, width)
        batch_size (int): Batch size
        opset_version (int): ONNX opset version
        output_path (str): Path to save ONNX model

    Returns:
        str: Path to saved ONNX model
    """
    if output_path is None:
        output_path = "models/final/model.onnx"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    # Create dummy input
    height, width = input_size
    dummy_input = torch.randn(batch_size, 3, height, width)

    # Move to device if needed
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)

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

    logger.info(f"Model exported to ONNX: {output_path}")

    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    logger.info(f"ONNX model verified successfully")

    return output_path


def quantize_model(model, quantization_config=None):
    """
    Quantize PyTorch model to reduce size and improve inference speed

    Args:
        model: PyTorch model
        quantization_config (dict): Quantization configuration

    Returns:
        torch.nn.Module: Quantized model
    """
    if quantization_config is None:
        quantization_config = {
            "method": "dynamic",
            "dtype": "qint8",
            "per_channel": True
        }

    # Set model to evaluation mode
    model.eval()

    # Configure quantization
    if quantization_config["method"] == "dynamic":
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},  # Layers to quantize
            dtype=torch.qint8 if quantization_config["dtype"] == "qint8" else torch.quint8
        )
    elif quantization_config["method"] == "static":
        # Static quantization (requires calibration data)
        raise NotImplementedError("Static quantization requires calibration data")

    logger.info(f"Model quantized using {quantization_config['method']} quantization")

    return quantized_model


def prune_model(model, pruning_config=None):
    """
    Prune model to reduce parameters

    Args:
        model: PyTorch model
        pruning_config (dict): Pruning configuration

    Returns:
        torch.nn.Module: Pruned model
    """
    import torch.nn.utils.prune as prune

    if pruning_config is None:
        pruning_config = {
            "method": "l1_unstructured",
            "amount": 0.3
        }

    # Layers to prune
    prunable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prunable_layers.append((name, module))

    # Apply pruning
    if pruning_config["method"] == "l1_unstructured":
        for name, module in prunable_layers:
            prune.l1_unstructured(module, 'weight', pruning_config["amount"])
            # Make pruning permanent
            prune.remove(module, 'weight')

    elif pruning_config["method"] == "random_unstructured":
        for name, module in prunable_layers:
            prune.random_unstructured(module, 'weight', pruning_config["amount"])
            # Make pruning permanent
            prune.remove(module, 'weight')

    else:
        raise ValueError(f"Unsupported pruning method: {pruning_config['method']}")

    logger.info(f"Model pruned using {pruning_config['method']} method with amount={pruning_config['amount']}")

    return model


def fuse_model(model):
    """
    Fuse model layers for improved inference performance

    Args:
        model: PyTorch model

    Returns:
        torch.nn.Module: Fused model
    """
    # Set model to evaluation mode
    model.eval()

    # Create a copy of the model
    fused_model = model

    # Find Conv-BN-ReLU patterns and fuse them
    for module_name, module in list(fused_model.named_children()):
        if len(list(module.children())) > 0:
            # Recursive call for nested modules
            setattr(fused_model, module_name, fuse_model(module))
            continue

        # Fuse Conv-BN layers
        if isinstance(module, torch.nn.Sequential):
            # Check if it's a Conv-BN-ReLU or Conv-BN sequence
            if len(module) == 3 and isinstance(module[0], torch.nn.Conv2d) and isinstance(module[1],
                                                                                          torch.nn.BatchNorm2d):
                if isinstance(module[2], torch.nn.ReLU) or isinstance(module[2], torch.nn.LeakyReLU):
                    # Conv-BN-ReLU
                    fused = torch.nn.utils.fuse_conv_bn_relu(module[0], module[1], module[2])
                    setattr(fused_model, module_name, fused)
            elif len(module) == 2 and isinstance(module[0], torch.nn.Conv2d) and isinstance(module[1],
                                                                                            torch.nn.BatchNorm2d):
                # Conv-BN
                fused = torch.nn.utils.fuse_conv_bn_eval(module[0], module[1])
                setattr(fused_model, module_name, fused)

    logger.info(f"Model layers fused for improved inference performance")

    return fused_model


def convert_state_dict(state_dict, model_prefix='model.'):
    """
    Convert state dict from training format to inference format

    Args:
        state_dict (dict): Model state dict
        model_prefix (str): Prefix to remove from keys

    Returns:
        dict: Converted state dict
    """
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if key.startswith(model_prefix):
            # Remove prefix
            new_key = key[len(model_prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def load_model_weights(model, weights_path, strict=False):
    """
    Load weights from checkpoint to model

    Args:
        model: PyTorch model
        weights_path (str): Path to weights file
        strict (bool): Whether to strictly enforce that the keys in state_dict match

    Returns:
        torch.nn.Module: Model with loaded weights
    """
    # Check if file exists
    if not os.path.exists(weights_path):
        logger.error(f"Weights file not found: {weights_path}")
        return model

    # Load weights
    checkpoint = torch.load(weights_path, map_location='cpu')

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

    # Convert state dict if needed
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = convert_state_dict(state_dict)

    # Load weights
    try:
        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Model weights loaded from {weights_path}")
    except Exception as e:
        logger.warning(f"Failed to load weights strictly, trying with strict=False: {e}")
        # Try loading with strict=False
        incompatible = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Model weights loaded with missing/unexpected keys: {incompatible}")

    return model


def calculate_model_size(model):
    """
    Calculate model size (parameters count and memory footprint)

    Args:
        model: PyTorch model

    Returns:
        dict: Dictionary with model size information
    """
    # Calculate parameters count
    params_count = sum(p.numel() for p in model.parameters())

    # Calculate memory footprint
    memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    memory_mb = memory_bytes / (1024 * 1024)

    return {
        "params_count": params_count,
        "memory_mb": memory_mb
    }


def analyze_model_performance(model, input_size=(416, 416), batch_size=1, iterations=100, warmup=10, device='cuda'):
    """
    Analyze model inference performance

    Args:
        model: PyTorch model
        input_size (tuple): Input size (height, width)
        batch_size (int): Batch size
        iterations (int): Number of iterations
        warmup (int): Number of warmup iterations
        device (str): Device to run on (cuda/cpu)

    Returns:
        dict: Dictionary with performance metrics
    """
    # Ensure model is in eval mode
    model.eval()

    # Create dummy input
    height, width = input_size
    dummy_input = torch.randn(batch_size, 3, height, width, device=device)

    # Warm up
    for _ in range(warmup):
        _ = model(dummy_input)

    # Synchronize if using CUDA
    if device == 'cuda':
        torch.cuda.synchronize()

    # Measure inference time
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    inference_times = []

    for _ in range(iterations):
        start_time.record()
        _ = model(dummy_input)
        end_time.record()

        # Synchronize if using CUDA
        if device == 'cuda':
            torch.cuda.synchronize()

        # Calculate inference time in milliseconds
        inference_time = start_time.elapsed_time(end_time)
        inference_times.append(inference_time)

    # Calculate performance metrics
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    fps = 1000 / avg_time  # Convert ms to fps

    return {
        "avg_inference_time_ms": avg_time,
        "std_inference_time_ms": std_time,
        "min_inference_time_ms": min_time,
        "max_inference_time_ms": max_time,
        "fps": fps,
        "batch_size": batch_size,
        "input_size": input_size,
        "device": device
    }


def export_torchscript(model, input_size=(416, 416), batch_size=1, output_path=None):
    """
    Export model to TorchScript format

    Args:
        model: PyTorch model
        input_size (tuple): Input size (height, width)
        batch_size (int): Batch size
        output_path (str): Path to save TorchScript model

    Returns:
        str: Path to saved TorchScript model
    """
    if output_path is None:
        output_path = "models/final/model.pt"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    # Create dummy input
    height, width = input_size
    dummy_input = torch.randn(batch_size, 3, height, width)

    # Move to device if needed
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)

    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)

    # Save the model
    traced_model.save(output_path)

    logger.info(f"Model exported to TorchScript: {output_path}")

    return output_path


if __name__ == "__main__":
    # Test model utilities
    import argparse
    import sys

    sys.path.append('.')  # Add project root to path

    from src.models.yolo import create_model

    parser = argparse.ArgumentParser(description="Model utilities for object detection")
    parser.add_argument("--weights", type=str, default="models/checkpoints/best_model.pt", help="Path to model weights")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to model configuration")
    parser.add_argument("--output", type=str, default="models/final", help="Output directory")
    parser.add_argument("--action", type=str, choices=["onnx", "quantize", "prune", "benchmark"],
                        help="Action to perform")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load model configuration
    with open(args.config, "r") as f:
        model_config = yaml.safe_load(f)

    # Create model
    model = create_model(model_config)

    # Load weights if provided
    if args.weights:
        model = load_model_weights(model, args.weights)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Perform action
    if args.action == "onnx":
        output_path = os.path.join(args.output, "model.onnx")
        convert_model_to_onnx(model, output_path=output_path)

    elif args.action == "quantize":
        quantized_model = quantize_model(model)
        output_path = os.path.join(args.output, "model_quantized.pt")
        torch.save(quantized_model.state_dict(), output_path)
        logger.info(f"Quantized model saved to {output_path}")

        # Print model size comparison
        original_size = calculate_model_size(model)
        quantized_size = calculate_model_size(quantized_model)

        logger.info(f"Original model: {original_size['params_count']} parameters, {original_size['memory_mb']:.2f} MB")
        logger.info(
            f"Quantized model: {quantized_size['params_count']} parameters, {quantized_size['memory_mb']:.2f} MB")
        logger.info(f"Memory reduction: {(1 - quantized_size['memory_mb'] / original_size['memory_mb']) * 100:.2f}%")

    elif args.action == "prune":
        pruned_model = prune_model(model)
        output_path = os.path.join(args.output, "model_pruned.pt")
        torch.save(pruned_model.state_dict(), output_path)
        logger.info(f"Pruned model saved to {output_path}")

        # Print model size comparison
        original_size = calculate_model_size(model)
        pruned_size = calculate_model_size(pruned_model)

        logger.info(f"Original model: {original_size['params_count']} parameters, {original_size['memory_mb']:.2f} MB")
        logger.info(f"Pruned model: {pruned_size['params_count']} parameters, {pruned_size['memory_mb']:.2f} MB")
        logger.info(
            f"Parameter reduction: {(1 - pruned_size['params_count'] / original_size['params_count']) * 100:.2f}%")

    elif args.action == "benchmark":
        # Benchmark model performance
        device = "cuda" if torch.cuda.is_available() else "cpu"
        performance = analyze_model_performance(model, device=device)

        logger.info(f"Model performance on {device}:")
        logger.info(f"- Average inference time: {performance['avg_inference_time_ms']:.2f} ms")
        logger.info(f"- FPS: {performance['fps']:.2f}")
        logger.info(f"- Batch size: {performance['batch_size']}")
        logger.info(f"- Input size: {performance['input_size']}")