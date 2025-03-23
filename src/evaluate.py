import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt

# Import local modules
from src.data.data_loader import create_data_loaders
from src.models.yolo import create_model
from src.utils.metrics import calculate_map, calculate_metrics_on_dataset, calculate_confusion_matrix, \
    calculate_precision_recall_f1
from src.utils.visualization import visualize_batch, create_confusion_matrix_plot


def setup_logging(config):
    """
    Set up logging configuration

    Args:
        config (dict): Main configuration

    Returns:
        logger: Configured logger
    """
    log_dir = os.path.join(config['logs_dir'], 'evaluation')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'evaluation_{timestamp}.log')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


def evaluate_model(model, data_loader, device, config, output_dir):
    """
    Evaluate model on a dataset

    Args:
        model: Model to evaluate
        data_loader: Data loader for the dataset
        device: Device to use (cuda/cpu)
        config: Configuration dictionary
        output_dir: Directory to save evaluation results

    Returns:
        dict: Evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Calculate metrics
    metrics = calculate_metrics_on_dataset(model, data_loader, device)

    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Create visualization of example predictions
    visualizations_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)

    # Get class names from data config
    with open('config/data_config.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
        class_names = data_config['dataset']['classes']

    # Create confusion matrix
    confusion_matrix = None
    all_predictions = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Creating visualization examples")):
            # Only process a few batches for visualization
            if batch_idx >= 10:
                break

            # Move data to device
            images = batch['images'].to(device)
            targets = []
            for boxes, labels in zip(batch['boxes'], batch['labels']):
                boxes = torch.tensor(boxes, device=device)
                labels = torch.tensor(labels, device=device)
                targets.append({'boxes': boxes, 'labels': labels})

            # Forward pass
            predictions = model(images)

            # Store predictions and targets for confusion matrix
            all_predictions.extend(predictions)
            all_targets.extend(targets)

            # Create visualizations
            for i in range(len(images)):
                if i >= 5:  # Limit to 5 images per batch
                    break

                vis_image = visualize_batch(
                    images[i].cpu(),
                    predictions[i].cpu() if isinstance(predictions, list) else predictions[i].cpu(),
                    targets[i],
                    class_names=class_names
                )

                # Save visualization
                plt.imsave(
                    os.path.join(visualizations_dir, f'sample_{batch_idx}_{i}.jpg'),
                    vis_image
                )

    # Calculate confusion matrix
    num_classes = len(class_names)
    confusion_matrix = calculate_confusion_matrix(all_predictions, all_targets, num_classes)

    # Calculate precision, recall, F1 score
    precision, recall, f1 = calculate_precision_recall_f1(confusion_matrix)

    # Save metrics per class
    per_class_metrics = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist()
    }

    per_class_metrics_file = os.path.join(output_dir, 'per_class_metrics.json')
    with open(per_class_metrics_file, 'w') as f:
        json.dump(per_class_metrics, f, indent=4)

    # Create confusion matrix visualization
    confusion_matrix_img = create_confusion_matrix_plot(confusion_matrix, class_names)
    plt.imsave(os.path.join(output_dir, 'confusion_matrix.png'), confusion_matrix_img)

    # Create precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'b-', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()

    # Create F1 score plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    plt.bar(x, f1)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Class')
    plt.xticks(x, class_names, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_score_per_class.png'))
    plt.close()

    return metrics


def main(config_path, model_path=None):
    """
    Main evaluation function

    Args:
        config_path (str): Path to the main configuration file
        model_path (str, optional): Path to model weights
    """
    # Load configurations
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(os.path.join(os.path.dirname(config_path), 'model_config.yaml'), 'r') as f:
        model_config = yaml.safe_load(f)

    with open(os.path.join(os.path.dirname(config_path), 'data_config.yaml'), 'r') as f:
        data_config = yaml.safe_load(f)

    # Set up logging
    logger = setup_logging(config)
    logger.info("Starting evaluation...")

    # Set random seeds for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create data loader (use validation set for evaluation)
    logger.info("Creating data loader...")
    _, val_loader = create_data_loaders(data_config)
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")

    # Create model
    logger.info("Creating model...")
    model = create_model(model_config, num_classes=len(data_config['dataset']['classes']))

    # Load model weights
    if model_path:
        logger.info(f"Loading model weights from {model_path}")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')

            # Extract state dict if it's a checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Log additional information
                if 'epoch' in checkpoint:
                    logger.info(f"Checkpoint from epoch {checkpoint['epoch']} with mAP {checkpoint.get('mAP', 'N/A')}")
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
        else:
            logger.error(f"Model weights file not found: {model_path}")
            return
    else:
        # Try to load best model from default location
        default_model_path = os.path.join(config['models_dir'], 'checkpoints', 'best_model.pt')
        if os.path.exists(default_model_path):
            logger.info(f"Loading model weights from {default_model_path}")
            checkpoint = torch.load(default_model_path, map_location='cpu')

            # Extract state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Log additional information
                if 'epoch' in checkpoint:
                    logger.info(f"Checkpoint from epoch {checkpoint['epoch']} with mAP {checkpoint.get('mAP', 'N/A')}")
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
        else:
            logger.warning(f"No model weights specified and no best model found at {default_model_path}")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config['logs_dir'], 'evaluation', f'run_{timestamp}')

    # Evaluate model
    logger.info("Starting evaluation...")
    metrics = evaluate_model(model, val_loader, device, config, output_dir)

    # Log evaluation results
    logger.info("Evaluation results:")
    logger.info(f"mAP: {metrics['mAP']:.4f}")
    logger.info(f"mAP@0.5: {metrics['mAP_50']:.4f}")
    logger.info(f"mAP@0.75: {metrics['mAP_75']:.4f}")
    logger.info(f"Average inference time: {metrics['avg_inference_time'] * 1000:.2f} ms")
    logger.info(f"FPS: {metrics['fps']:.2f}")

    logger.info(f"Evaluation results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO object detection model")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model weights file")
    args = parser.parse_args()

    main(args.config, args.model)