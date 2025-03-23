import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes

    Args:
        box1: Box 1 coordinates [x1, y1, x2, y2]
        box2: Box 2 coordinates [x1, y1, x2, y2]

    Returns:
        float: IoU value
    """
    # Get coordinates of intersection
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_width = max(0, x2_min - x1_max)
    intersection_height = max(0, y2_min - y1_max)
    intersection_area = intersection_width * intersection_height

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / max(union_area, 1e-6)

    return iou


def calculate_ap(recalls, precisions):
    """
    Calculate Average Precision (AP) using the 11-point interpolation

    Args:
        recalls: List of recall values
        precisions: List of precision values

    Returns:
        float: AP value
    """
    # Make sure lists are numpy arrays
    recalls = np.array(recalls)
    precisions = np.array(precisions)

    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11

    return ap


def calculate_map(predictions, targets, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP) for object detection

    Args:
        predictions: List of prediction dictionaries with keys 'boxes', 'labels', 'scores'
        targets: List of target dictionaries with keys 'boxes', 'labels'
        iou_threshold: IoU threshold for determining a positive detection (default: 0.5)

    Returns:
        float: mAP value
    """
    if not predictions or not targets:
        return 0.0

    # Convert to list if not already
    if isinstance(predictions, torch.Tensor):
        predictions = [{'boxes': pred[:, :4], 'labels': pred[:, 5].long(), 'scores': pred[:, 4]}
                       for pred in predictions]

    # Initialize counters
    class_metrics = defaultdict(lambda: {'TP': [], 'FP': [], 'scores': [], 'num_gt': 0})

    # Process all images
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']

        target_boxes = target['boxes']
        target_labels = target['labels']

        # Skip if no predictions or no ground truth
        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            continue

        # Convert to numpy if tensors
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().numpy()
        if isinstance(target_boxes, torch.Tensor):
            target_boxes = target_boxes.cpu().numpy()
        if isinstance(target_labels, torch.Tensor):
            target_labels = target_labels.cpu().numpy()

        # Count number of ground truth objects per class
        for label in target_labels:
            class_metrics[int(label)]['num_gt'] += 1

        # For each prediction, determine if it's a true positive or false positive
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            # Add score to the class metrics
            class_metrics[int(label)]['scores'].append(float(score))

            # Find matching ground truth box
            max_iou = -1
            max_idx = -1

            for idx, (gt_box, gt_label) in enumerate(zip(target_boxes, target_labels)):
                if gt_label != label:
                    continue

                iou = calculate_iou(box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = idx

            # If we found a matching ground truth box with high enough IoU, it's a true positive
            if max_iou >= iou_threshold:
                class_metrics[int(label)]['TP'].append(1)
                class_metrics[int(label)]['FP'].append(0)

                # Remove the matched ground truth box to prevent multiple detections
                target_boxes = np.delete(target_boxes, max_idx, axis=0)
                target_labels = np.delete(target_labels, max_idx)
            else:
                # Otherwise, it's a false positive
                class_metrics[int(label)]['TP'].append(0)
                class_metrics[int(label)]['FP'].append(1)

    # Calculate AP for each class
    aps = []

    for class_id, metrics in class_metrics.items():
        # Sort by confidence scores
        scores = metrics['scores']
        tp = metrics['TP']
        fp = metrics['FP']
        num_gt = metrics['num_gt']

        if num_gt == 0 or len(scores) == 0:
            continue

        # Sort by confidence scores
        indices = np.argsort(scores)[::-1]
        tp = np.array(tp)[indices]
        fp = np.array(fp)[indices]

        # Calculate cumulative values
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Calculate precision and recall
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / (num_gt + 1e-6)

        # Add a start point (0, 1) and end point (1, 0) for precision-recall curve
        precisions = np.concatenate(([1], precisions))
        recalls = np.concatenate(([0], recalls))

        # Calculate AP
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)

    # Return mean AP if we have any valid classes
    if len(aps) > 0:
        return np.mean(aps)
    else:
        return 0.0


def calculate_metrics_on_dataset(model, data_loader, device):
    """
    Calculate detection metrics on a dataset

    Args:
        model: Detection model
        data_loader: Data loader for the dataset
        device: Device to run the model on (cuda/cpu)

    Returns:
        dict: Dictionary with metrics (mAP, inference time, etc.)
    """
    model.eval()

    all_predictions = []
    all_targets = []

    total_time = 0
    num_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Calculating metrics")

        for batch in progress_bar:
            # Move data to device
            images = batch['images'].to(device)
            targets = []
            for boxes, labels in zip(batch['boxes'], batch['labels']):
                boxes = torch.tensor(boxes, device=device)
                labels = torch.tensor(labels, device=device)
                targets.append({'boxes': boxes, 'labels': labels})

            # Measure inference time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            predictions = model(images)
            end_time.record()

            # Wait for GPU to sync
            torch.cuda.synchronize()
            batch_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds

            total_time += batch_time
            num_samples += len(images)

            # Store predictions and targets for mAP calculation
            all_predictions.extend(predictions)
            all_targets.extend(targets)

    # Calculate metrics
    mAP = calculate_map(all_predictions, all_targets)

    # Calculate mAP at different IoU thresholds
    mAP_50 = calculate_map(all_predictions, all_targets, iou_threshold=0.5)
    mAP_75 = calculate_map(all_predictions, all_targets, iou_threshold=0.75)

    # Calculate average inference time
    avg_inference_time = total_time / num_samples

    return {
        'mAP': mAP,
        'mAP_50': mAP_50,
        'mAP_75': mAP_75,
        'avg_inference_time': avg_inference_time,
        'fps': 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    }


def calculate_confusion_matrix(predictions, targets, num_classes, iou_threshold=0.5, conf_threshold=0.25):
    """
    Calculate confusion matrix for object detection

    Args:
        predictions: List of prediction dictionaries with keys 'boxes', 'labels', 'scores'
        targets: List of target dictionaries with keys 'boxes', 'labels'
        num_classes: Number of classes
        iou_threshold: IoU threshold for determining a positive detection
        conf_threshold: Confidence threshold for filtering predictions

    Returns:
        numpy.ndarray: Confusion matrix of shape (num_classes, num_classes)
    """
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Process all images
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']

        target_boxes = target['boxes']
        target_labels = target['labels']

        # Skip if no predictions or no ground truth
        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            continue

        # Convert to numpy if tensors
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().numpy()
        if isinstance(target_boxes, torch.Tensor):
            target_boxes = target_boxes.cpu().numpy()
        if isinstance(target_labels, torch.Tensor):
            target_labels = target_labels.cpu().numpy()

        # Filter predictions by confidence
        confidence_mask = pred_scores >= conf_threshold
        pred_boxes = pred_boxes[confidence_mask]
        pred_labels = pred_labels[confidence_mask]
        pred_scores = pred_scores[confidence_mask]

        # For each ground truth box, find matching prediction
        for gt_idx, (gt_box, gt_label) in enumerate(zip(target_boxes, target_labels)):
            gt_label = int(gt_label)

            best_iou = -1
            best_pred_idx = -1

            # Find prediction with highest IoU
            for pred_idx, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
                label = int(label)

                iou = calculate_iou(gt_box, box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx

            # If match found with high enough IoU
            if best_iou >= iou_threshold and best_pred_idx != -1:
                pred_label = int(pred_labels[best_pred_idx])

                # Add to confusion matrix (true label, predicted label)
                confusion_matrix[gt_label, pred_label] += 1

                # Remove matched prediction to prevent double counting
                pred_boxes = np.delete(pred_boxes, best_pred_idx, axis=0)
                pred_labels = np.delete(pred_labels, best_pred_idx)
                pred_scores = np.delete(pred_scores, best_pred_idx)
            else:
                # No match found, count as false negative (true label, background)
                confusion_matrix[gt_label, 0] += 1

        # Remaining predictions are false positives
        for label in pred_labels:
            # Background labeled as something (background, predicted label)
            confusion_matrix[0, int(label)] += 1

    return confusion_matrix


def calculate_precision_recall_f1(confusion_matrix):
    """
    Calculate precision, recall, and F1 score from confusion matrix

    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes)

    Returns:
        tuple: (precision, recall, f1) arrays for each class
    """
    num_classes = confusion_matrix.shape[0]

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        # True positives: diagonal elements
        tp = confusion_matrix[i, i]

        # False positives: column sum - true positives
        fp = np.sum(confusion_matrix[:, i]) - tp

        # False negatives: row sum - true positives
        fn = np.sum(confusion_matrix[i, :]) - tp

        # Calculate metrics
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    return precision, recall, f1


if __name__ == "__main__":
    # Test code for metrics
    import torch
    import numpy as np

    # Create sample predictions and targets
    predictions = [
        {
            'boxes': np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]),
            'labels': np.array([1, 2]),
            'scores': np.array([0.9, 0.8])
        },
        {
            'boxes': np.array([[0.5, 0.5, 0.6, 0.6]]),
            'labels': np.array([1]),
            'scores': np.array([0.7])
        }
    ]

    targets = [
        {
            'boxes': np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]),
            'labels': np.array([1, 2])
        },
        {
            'boxes': np.array([[0.5, 0.5, 0.6, 0.6], [0.7, 0.7, 0.8, 0.8]]),
            'labels': np.array([1, 3])
        }
    ]

    # Calculate mAP
    mAP = calculate_map(predictions, targets)
    print(f"mAP: {mAP:.4f}")

    # Calculate confusion matrix
    confusion_matrix = calculate_confusion_matrix(predictions, targets, num_classes=4)
    print("Confusion Matrix:")
    print(confusion_matrix)

    # Calculate precision, recall, F1
    precision, recall, f1 = calculate_precision_recall_f1(confusion_matrix)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)