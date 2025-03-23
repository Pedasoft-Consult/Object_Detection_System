import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import io
import os
import random
from torchvision.transforms.functional import to_pil_image

# Define color palette for consistent visualization
COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),  # Maroon
    (0, 128, 0),  # Green (dark)
    (0, 0, 128),  # Navy
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (255, 165, 0),  # Orange
    (210, 105, 30),  # Chocolate
    (50, 205, 50),  # Lime green
    (220, 20, 60),  # Crimson
    (0, 191, 255),  # Deep sky blue
    (255, 215, 0)  # Gold
]


def get_color(class_id):
    """
    Get color for a specific class ID

    Args:
        class_id (int): Class ID

    Returns:
        tuple: RGB color
    """
    return COLORS[class_id % len(COLORS)]


def draw_boxes(image, boxes, labels=None, scores=None, class_names=None, line_thickness=2, font_scale=0.6):
    """
    Draw bounding boxes on image

    Args:
        image: Image as numpy array (H, W, C)
        boxes: Bounding boxes as numpy array (N, 4) in format [x1, y1, x2, y2]
        labels: Class labels as numpy array (N,)
        scores: Confidence scores as numpy array (N,)
        class_names: List of class names
        line_thickness: Thickness of bounding box lines
        font_scale: Font scale for labels

    Returns:
        numpy.ndarray: Image with boxes
    """
    # Make a copy of the image
    image_draw = image.copy()

    # Get image dimensions
    height, width = image_draw.shape[:2]

    # Draw boxes
    for i, box in enumerate(boxes):
        # Get coordinates
        x1, y1, x2, y2 = [int(coord) for coord in box]

        # Ensure coordinates are within image boundaries
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # Get label and color
        if labels is not None:
            label = int(labels[i])
            color = get_color(label)
        else:
            color = get_color(i)

        # Draw rectangle
        cv2.rectangle(image_draw, (x1, y1), (x2, y2), color, line_thickness)

        # Add label and score
        if labels is not None:
            label_text = class_names[label] if class_names is not None else f"Class {label}"

            if scores is not None:
                label_text += f" {scores[i]:.2f}"

            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
            )

            # Draw label background
            cv2.rectangle(
                image_draw,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )

            # Draw text
            cv2.putText(
                image_draw,
                label_text,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                line_thickness // 2
            )

    return image_draw


def visualize_batch(image_tensor, predictions, targets=None, class_names=None, max_boxes=20):
    """
    Visualize image with predictions and optionally ground truth

    Args:
        image_tensor: Image tensor (C, H, W)
        predictions: Prediction tensor from model or dictionary with 'boxes', 'labels', 'scores'
        targets: Target dictionary with 'boxes', 'labels'
        class_names: List of class names
        max_boxes: Maximum number of boxes to draw

    Returns:
        numpy.ndarray: Visualization image
    """
    # Convert image tensor to numpy array
    if isinstance(image_tensor, torch.Tensor):
        # Denormalize if needed (assuming mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = image_tensor.clone().detach().cpu()
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3,
                                                                                                                     1,
                                                                                                                     1)
        image = image.clamp(0, 1)

        # Convert to PIL Image
        image_pil = to_pil_image(image)

        # Convert to numpy array
        image_np = np.array(image_pil)
    else:
        image_np = image_tensor

    # Process predictions
    if isinstance(predictions, torch.Tensor):
        # Model output with shape [batch, num_predictions, num_classes + 5]
        # where 5 is [x, y, w, h, confidence]
        if predictions.ndim == 2:
            # Extract boxes, scores, and labels
            pred_boxes = predictions[:, :4].cpu().numpy()
            pred_scores = predictions[:, 4].cpu().numpy()
            pred_labels = torch.argmax(predictions[:, 5:], dim=1).cpu().numpy()
        else:
            # No predictions
            pred_boxes = np.zeros((0, 4))
            pred_scores = np.zeros(0)
            pred_labels = np.zeros(0)
    elif isinstance(predictions, dict):
        # Dictionary with keys 'boxes', 'labels', 'scores'
        pred_boxes = predictions.get('boxes', torch.zeros((0, 4))).cpu().numpy()
        pred_scores = predictions.get('scores', torch.zeros(0)).cpu().numpy()
        pred_labels = predictions.get('labels', torch.zeros(0)).cpu().numpy()
    else:
        # No predictions
        pred_boxes = np.zeros((0, 4))
        pred_scores = np.zeros(0)
        pred_labels = np.zeros(0)

    # Limit number of boxes
    if len(pred_boxes) > max_boxes:
        indices = np.argsort(pred_scores)[-max_boxes:]
        pred_boxes = pred_boxes[indices]
        pred_scores = pred_scores[indices]
        pred_labels = pred_labels[indices]

    # Convert xywh to xyxy format if needed
    if pred_boxes.shape[1] == 4 and pred_boxes.size > 0:
        if np.max(pred_boxes[:, 2:]) <= 1.0:
            # Normalized coordinates (x1, y1, x2, y2) format
            # Scale to image size
            height, width = image_np.shape[:2]
            pred_boxes[:, 0] *= width
            pred_boxes[:, 1] *= height
            pred_boxes[:, 2] *= width
            pred_boxes[:, 3] *= height
        elif len(pred_boxes) > 0 and pred_boxes[0, 2] < pred_boxes[0, 0]:
            # xywh format, convert to xyxy
            x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
            y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
            x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
            y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
            pred_boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Create visualization with predictions
    vis_image = draw_boxes(
        image_np,
        pred_boxes,
        pred_labels,
        pred_scores,
        class_names,
        line_thickness=2,
        font_scale=0.6
    )

    # Add ground truth if provided
    if targets is not None:
        # Get target boxes and labels
        if isinstance(targets, dict):
            target_boxes = targets.get('boxes', torch.zeros((0, 4))).cpu().numpy()
            target_labels = targets.get('labels', torch.zeros(0)).cpu().numpy()
        else:
            target_boxes = np.zeros((0, 4))
            target_labels = np.zeros(0)

        # Convert target boxes format if needed (same as predictions)
        if target_boxes.shape[1] == 4 and target_boxes.size > 0:
            if np.max(target_boxes[:, 2:]) <= 1.0:
                # Normalized coordinates (x1, y1, x2, y2) format
                # Scale to image size
                height, width = image_np.shape[:2]
                target_boxes[:, 0] *= width
                target_boxes[:, 1] *= height
                target_boxes[:, 2] *= width
                target_boxes[:, 3] *= height
            elif len(target_boxes) > 0 and target_boxes[0, 2] < target_boxes[0, 0]:
                # xywh format, convert to xyxy
                x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
                y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
                x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
                y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
                target_boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Draw ground truth boxes with dashed lines
        for i, box in enumerate(target_boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            label = int(target_labels[i])
            color = get_color(label)

            # Draw dashed rectangle
            for j in range(0, int(x2 - x1), 10):
                x_start = x1 + j
                x_end = min(x_start + 5, x2)
                cv2.line(vis_image, (x_start, y1), (x_end, y1), color, 2)
                cv2.line(vis_image, (x_start, y2), (x_end, y2), color, 2)

            for j in range(0, int(y2 - y1), 10):
                y_start = y1 + j
                y_end = min(y_start + 5, y2)
                cv2.line(vis_image, (x1, y_start), (x1, y_end), color, 2)
                cv2.line(vis_image, (x2, y_start), (x2, y_end), color, 2)

    # Convert to RGB (for Tensorboard)
    if vis_image.shape[2] == 3:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

    return vis_image


def plot_training_metrics(metrics_file, output_dir='logs'):
    """
    Plot training metrics from a log file

    Args:
        metrics_file: Path to metrics CSV file
        output_dir: Directory to save plots
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load metrics
    metrics = pd.read_csv(metrics_file)

    # Extract columns for plotting
    epochs = metrics['epoch']
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    mAP = metrics['mAP']
    lr = metrics['learning_rate']

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

    # Plot mAP
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, mAP, 'g-', label='mAP')
    plt.title('Mean Average Precision (mAP)')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'map_plot.png'))
    plt.close()

    # Plot learning rate
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, lr, 'm-', label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'lr_plot.png'))
    plt.close()


def visualize_augmentation(image, transformed_image, boxes=None, transformed_boxes=None,
                           labels=None, class_names=None):
    """
    Visualize data augmentation effects

    Args:
        image: Original image
        transformed_image: Augmented image
        boxes: Original boxes
        transformed_boxes: Augmented boxes
        labels: Class labels
        class_names: List of class names

    Returns:
        numpy.ndarray: Visualization image with before/after comparison
    """
    # Convert to numpy if tensors
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
    if isinstance(transformed_image, torch.Tensor):
        transformed_image = transformed_image.cpu().numpy().transpose(1, 2, 0)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image
    ax1.imshow(image)
    ax1.set_title('Original Image')

    # Draw original boxes
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=get_color(labels[i] if labels is not None else i),
                facecolor='none'
            )
            ax1.add_patch(rect)

            # Add label text
            if labels is not None:
                label = class_names[labels[i]] if class_names is not None else f"Class {labels[i]}"
                ax1.text(
                    x1, y1 - 5, label,
                    color='white',
                    fontsize=8,
                    bbox=dict(facecolor=get_color(labels[i]), alpha=0.8)
                )

    # Plot transformed image
    ax2.imshow(transformed_image)
    ax2.set_title('Augmented Image')

    # Draw transformed boxes
    if transformed_boxes is not None:
        for i, box in enumerate(transformed_boxes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=get_color(labels[i] if labels is not None else i),
                facecolor='none'
            )
            ax2.add_patch(rect)

            # Add label text
            if labels is not None:
                label = class_names[labels[i]] if class_names is not None else f"Class {labels[i]}"
                ax2.text(
                    x1, y1 - 5, label,
                    color='white',
                    fontsize=8,
                    bbox=dict(facecolor=get_color(labels[i]), alpha=0.8)
                )

    # Remove axis ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()

    # Convert figure to numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = np.array(Image.open(buf))
    plt.close(fig)

    return img


def create_confusion_matrix_plot(confusion_matrix, class_names=None):
    """
    Create confusion matrix plot

    Args:
        confusion_matrix: Confusion matrix as numpy array
        class_names: List of class names

    Returns:
        numpy.ndarray: Confusion matrix image
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create figure
    plt.figure(figsize=(10, 8))

    # Normalize confusion matrix
    cm_norm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-6)

    # Create heatmap
    sns.heatmap(
        cm_norm,
        annot=True,
        cmap='Blues',
        fmt='.2f',
        square=True,
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto'
    )

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Convert figure to numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = np.array(Image.open(buf))
    plt.close()

    return img


if __name__ == "__main__":
    # Test visualization functions
    # Create sample image and boxes
    image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
    boxes = np.array([
        [50, 50, 150, 150],
        [200, 200, 300, 300]
    ])
    labels = np.array([0, 1])
    scores = np.array([0.9, 0.8])

    # Test draw_boxes
    vis_image = draw_boxes(image, boxes, labels, scores)

    # Save test image
    cv2.imwrite('test_visualization.jpg', vis_image)