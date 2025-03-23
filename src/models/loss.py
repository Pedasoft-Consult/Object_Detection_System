import torch
import torch.nn as nn
import numpy as np
import math


class YOLOLoss(nn.Module):
    """
    YOLOv5 Loss function

    Computes loss between predictions and targets for object detection task
    """

    def __init__(self, num_classes, device='cuda', box_weight=0.05, obj_weight=1.0, cls_weight=0.5, iou_type='ciou'):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device

        # Loss weights
        self.box_weight = box_weight  # box loss weight
        self.obj_weight = obj_weight  # objectness loss weight
        self.cls_weight = cls_weight  # class loss weight

        # Define anchors for each detection level
        self.anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],  # P3/8
            [[30, 61], [62, 45], [59, 119]],  # P4/16
            [[116, 90], [156, 198], [373, 326]]  # P5/32
        ], device=device).float()

        # Number of anchors per level
        self.num_anchors = self.anchors.shape[1]

        # Strides for each detection level
        self.strides = torch.tensor([8, 16, 32], device=device)

        # IOU type for box regression loss
        self.iou_type = iou_type

        # Define loss functions
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')

        # Set class weights (if needed)
        self.class_weights = torch.ones(self.num_classes, device=device)

    def forward(self, predictions, targets):
        """
        Compute YOLOv5 loss

        Args:
            predictions: List of prediction tensors from model
            targets: List of ground truth dictionaries with 'boxes' and 'labels'

        Returns:
            tensor: Total loss
        """
        # Initialize loss components
        box_loss = torch.tensor(0.0, device=self.device)
        obj_loss = torch.tensor(0.0, device=self.device)
        cls_loss = torch.tensor(0.0, device=self.device)

        # Number of batch elements
        batch_size = len(targets)

        # Process each prediction level (small, medium, large objects)
        for level, pred in enumerate(predictions):
            # Get anchors and stride for this level
            anchors = self.anchors[level]
            stride = self.strides[level]

            # Extract dimensions
            batch, _, grid_h, grid_w = pred.shape

            # Reshape prediction to [batch, num_anchors, grid_h, grid_w, 5+num_classes]
            # where 5 is [x, y, w, h, objectness]
            pred = pred.view(batch, self.num_anchors, 5 + self.num_classes, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()

            # Get predicted values
            pred_xy = torch.sigmoid(pred[..., 0:2])  # x, y
            pred_wh = torch.exp(pred[..., 2:4])  # w, h
            pred_obj = pred[..., 4]  # objectness
            pred_cls = pred[..., 5:]  # class probabilities

            # Create grid
            grid_y, grid_x = torch.meshgrid([torch.arange(grid_h, device=self.device),
                                             torch.arange(grid_w, device=self.device)])
            grid = torch.stack((grid_x, grid_y), dim=2).view(1, 1, grid_h, grid_w, 2).float()

            # Calculate predicted box coordinates
            pred_bbox = torch.zeros_like(pred[..., :4])
            pred_bbox[..., 0:2] = (pred_xy + grid) * stride  # xy
            pred_bbox[..., 2:4] = pred_wh * anchors.view(1, self.num_anchors, 1, 1, 2) * stride  # wh

            # Create target tensor for this level
            target_obj = torch.zeros_like(pred_obj)
            target_cls = torch.zeros_like(pred_cls)

            # Process each element in batch
            for batch_idx in range(batch_size):
                # Get target boxes and labels for this batch element
                target_boxes = targets[batch_idx]['boxes']  # [N, 4]
                target_labels = targets[batch_idx]['labels']  # [N]

                if len(target_boxes) == 0:
                    continue

                # Convert target boxes to center coordinates (cx, cy, w, h)
                # Assuming target boxes are in format [x1, y1, x2, y2]
                if target_boxes.shape[1] == 4:
                    cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2.0
                    cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2.0
                    w = target_boxes[:, 2] - target_boxes[:, 0]
                    h = target_boxes[:, 3] - target_boxes[:, 1]
                    target_boxes = torch.stack([cx, cy, w, h], dim=1)

                # Assign targets to specific grid cells
                for target_idx in range(len(target_boxes)):
                    # Get target box and label
                    tx, ty, tw, th = target_boxes[target_idx]
                    tl = target_labels[target_idx]

                    # Skip if width or height is zero
                    if tw <= 0 or th <= 0:
                        continue

                    # Find the grid cell coordinates
                    cell_x = torch.floor(tx / stride).long()
                    cell_y = torch.floor(ty / stride).long()

                    # Skip if outside the grid
                    if cell_x >= grid_w or cell_y >= grid_h or cell_x < 0 or cell_y < 0:
                        continue

                    # Find the best anchor for this target
                    target_wh = torch.tensor([tw, th]).to(self.device)
                    anchor_wh = anchors.to(self.device)

                    # Calculate IoU between target and anchors
                    ratio = target_wh / anchor_wh
                    ratio = torch.max(ratio, 1.0 / ratio)
                    max_ratio = torch.max(ratio, dim=1)[0]
                    best_anchor = torch.argmin(max_ratio)

                    # Set objectness target
                    target_obj[batch_idx, best_anchor, cell_y, cell_x] = 1.0

                    # Set class target
                    if tl < self.num_classes:
                        target_cls[batch_idx, best_anchor, cell_y, cell_x, tl.long()] = 1.0

                    # Calculate box loss for this target
                    if pred_bbox.shape[0] > 0:
                        # Get predicted box for this cell
                        pb = pred_bbox[batch_idx, best_anchor, cell_y, cell_x]

                        # Calculate IoU loss
                        if self.iou_type == 'ciou':
                            iou_loss = 1.0 - self._calculate_ciou(pb, target_boxes[target_idx])
                        elif self.iou_type == 'diou':
                            iou_loss = 1.0 - self._calculate_diou(pb, target_boxes[target_idx])
                        elif self.iou_type == 'giou':
                            iou_loss = 1.0 - self._calculate_giou(pb, target_boxes[target_idx])
                        else:  # 'iou'
                            iou_loss = 1.0 - self._calculate_iou(pb, target_boxes[target_idx])

                        box_loss += iou_loss

            # Calculate objectness loss
            obj_loss += torch.mean(self.bce(pred_obj, target_obj))

            # Calculate class loss for positive samples
            target_mask = target_obj.bool()
            if target_mask.sum() > 0:
                cls_loss += torch.mean(self.bce(pred_cls[target_mask], target_cls[target_mask]))

        # Combine losses with weights
        total_loss = self.box_weight * box_loss + self.obj_weight * obj_loss + self.cls_weight * cls_loss

        return total_loss

    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two boxes

        Args:
            box1: First box [cx, cy, w, h]
            box2: Second box [cx, cy, w, h]

        Returns:
            tensor: IoU value
        """
        # Convert to x1, y1, x2, y2 format
        b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

        # Intersection
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        # Union
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-6)

        return iou

    def _calculate_giou(self, box1, box2):
        """
        Calculate GIoU (Generalized IoU) between two boxes

        Args:
            box1: First box [cx, cy, w, h]
            box2: Second box [cx, cy, w, h]

        Returns:
            tensor: GIoU value
        """
        # Calculate IoU
        iou = self._calculate_iou(box1, box2)

        # Convert to x1, y1, x2, y2 format
        b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

        # Calculate enclosing box
        enclose_x1 = torch.min(b1_x1, b2_x1)
        enclose_y1 = torch.min(b1_y1, b2_y1)
        enclose_x2 = torch.max(b1_x2, b2_x2)
        enclose_y2 = torch.max(b1_y2, b2_y2)

        # Calculate area of enclosing box
        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        enclose_area = enclose_w * enclose_h

        # Calculate GIoU
        giou = iou - (enclose_area - (box1[2] * box1[3] + box2[2] * box2[3] - iou * enclose_area)) / (
                    enclose_area + 1e-6)

        return giou

    def _calculate_diou(self, box1, box2):
        """
        Calculate DIoU (Distance IoU) between two boxes

        Args:
            box1: First box [cx, cy, w, h]
            box2: Second box [cx, cy, w, h]

        Returns:
            tensor: DIoU value
        """
        # Calculate IoU
        iou = self._calculate_iou(box1, box2)

        # Calculate center distance
        center_distance = torch.sum((box1[:2] - box2[:2]) ** 2)

        # Calculate diagonal length of enclosing box
        b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

        enclose_x1 = torch.min(b1_x1, b2_x1)
        enclose_y1 = torch.min(b1_y1, b2_y1)
        enclose_x2 = torch.max(b1_x2, b2_x2)
        enclose_y2 = torch.max(b1_y2, b2_y2)

        diagonal_length = torch.sum((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2)

        # Calculate DIoU
        diou = iou - center_distance / (diagonal_length + 1e-6)

        return diou

    def _calculate_ciou(self, box1, box2):
        """
        Calculate CIoU (Complete IoU) between two boxes

        Args:
            box1: First box [cx, cy, w, h]
            box2: Second box [cx, cy, w, h]

        Returns:
            tensor: CIoU value
        """
        # Calculate DIoU
        diou = self._calculate_diou(box1, box2)

        # Calculate aspect ratio consistency
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]

        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        alpha = v / (1 - diou + v + 1e-6)

        # Calculate CIoU
        ciou = diou - alpha * v

        return ciou


if __name__ == "__main__":
    # Test YOLOLoss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create YOLOLoss
    loss_fn = YOLOLoss(num_classes=80, device=device)

    # Create sample predictions and targets
    predictions = [
        torch.randn(2, 3 * (5 + 80), 10, 10).to(device),  # P3/8
        torch.randn(2, 3 * (5 + 80), 5, 5).to(device),  # P4/16
        torch.randn(2, 3 * (5 + 80), 3, 3).to(device)  # P5/32
    ]

    targets = [
        {
            'boxes': torch.tensor([[50, 50, 100, 100], [150, 150, 200, 200]], device=device),
            'labels': torch.tensor([1, 2], device=device)
        },
        {
            'boxes': torch.tensor([[250, 250, 300, 300]], device=device),
            'labels': torch.tensor([3], device=device)
        }
    ]

    # Calculate loss
    loss = loss_fn(predictions, targets)
    print(f"Loss: {loss.item()}")