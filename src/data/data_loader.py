import os
import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2


class COCODataset(Dataset):
    """
    Dataset class for loading COCO format data
    """

    def __init__(self, root_dir, annotation_file, transforms=None, image_size=(416, 416)):
        """
        Initialize COCODataset

        Args:
            root_dir (str): Directory with the images
            annotation_file (str): Path to the COCO annotation JSON file
            transforms (albumentations.Compose, optional): Albumentations transformations
            image_size (tuple): Target image size (height, width)
        """
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.transforms = transforms
        self.image_size = image_size

        # Get all image ids
        self.image_ids = list(self.coco.imgs.keys())

        # Load categories
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_ids = [category['id'] for category in self.categories]
        self.category_id_to_name = {category['id']: category['name'] for category in self.categories}
        self.category_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.category_ids)}

        print(f"Loaded {len(self.image_ids)} images and {len(self.category_ids)} categories")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Get a data sample: image and annotations

        Args:
            idx (int): Index

        Returns:
            dict: Sample with image and annotations
        """
        # Load image
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotations (bounding boxes)
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # Extract bounding boxes and class labels
        boxes = []
        class_labels = []

        for annotation in annotations:
            # Skip annotations with no area (often crowd annotations)
            if annotation['area'] <= 0:
                continue

            # COCO format: [x_min, y_min, width, height]
            x_min, y_min, width, height = annotation['bbox']

            # Convert to [x_min, y_min, x_max, y_max] format
            x_max = x_min + width
            y_max = y_min + height

            # Normalize to 0-1 range
            x_min = max(0, x_min / image_info['width'])
            y_min = max(0, y_min / image_info['height'])
            x_max = min(1, x_max / image_info['width'])
            y_max = min(1, y_max / image_info['height'])

            # Skip invalid boxes
            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            class_idx = self.category_id_to_idx[annotation['category_id']]
            class_labels.append(class_idx)

        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        class_labels = np.array(class_labels, dtype=np.int64)

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            class_labels = np.array(transformed['class_labels'], dtype=np.int64)

        # Handle the case of no valid boxes
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            class_labels = np.zeros(0, dtype=np.int64)

        return {
            'image': image,
            'boxes': boxes,
            'labels': class_labels,
            'image_id': image_id,
            'image_path': image_path
        }


def yolo_collate_fn(batch):
    """
    Custom collate function for YOLO batch processing

    Args:
        batch (list): List of samples

    Returns:
        dict: Batched data
    """
    images = []
    boxes = []
    labels = []
    image_ids = []
    image_paths = []

    for sample in batch:
        images.append(sample['image'])
        boxes.append(sample['boxes'])
        labels.append(sample['labels'])
        image_ids.append(sample['image_id'])
        image_paths.append(sample['image_path'])

    # Stack images into tensor
    images = torch.stack(images)

    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'image_ids': image_ids,
        'image_paths': image_paths
    }


def get_train_transforms(config):
    """
    Get transformations for training data

    Args:
        config (dict): Data configuration

    Returns:
        albumentations.Compose: Composed transformations
    """
    height, width = config['preprocessing']['resize']['height'], config['preprocessing']['resize']['width']

    transforms = []

    # Resize with letterbox
    if config['preprocessing']['resize']['method'] == 'letterbox':
        transforms.append(A.LongestMaxSize(max_size=max(height, width)))
        transforms.append(A.PadIfNeeded(
            min_height=height,
            min_width=width,
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        ))
    else:
        transforms.append(A.Resize(height=height, width=width))

    # Add augmentations if enabled
    if config['augmentation']['enabled']:
        if config['augmentation']['horizontal_flip']['enabled']:
            transforms.append(A.HorizontalFlip(
                p=config['augmentation']['horizontal_flip']['probability']
            ))

        if config['augmentation']['rotate']['enabled']:
            transforms.append(A.Rotate(
                limit=config['augmentation']['rotate']['max_angle'],
                p=config['augmentation']['rotate']['probability']
            ))

        if config['augmentation']['brightness']['enabled']:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=config['augmentation']['brightness']['factor'],
                contrast_limit=config['augmentation']['contrast']['factor'],
                p=max(config['augmentation']['brightness']['probability'],
                      config['augmentation']['contrast']['probability'])
            ))

        if config['augmentation']['hue']['enabled'] or config['augmentation']['saturation']['enabled']:
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=int(config['augmentation']['hue']['factor'] * 180),
                sat_shift_limit=int(config['augmentation']['saturation']['factor'] * 255),
                p=max(config['augmentation']['hue']['probability'],
                      config['augmentation']['saturation']['probability'])
            ))

    # Normalization and conversion to tensor
    transforms.extend([
        A.Normalize(
            mean=config['preprocessing']['normalize']['mean'],
            std=config['preprocessing']['normalize']['std']
        ),
        ToTensorV2()
    ])

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            min_visibility=0.1,
            label_fields=['class_labels']
        )
    )


def get_val_transforms(config):
    """
    Get transformations for validation data

    Args:
        config (dict): Data configuration

    Returns:
        albumentations.Compose: Composed transformations
    """
    height, width = config['preprocessing']['resize']['height'], config['preprocessing']['resize']['width']

    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(height, width)),
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            ),
            A.Normalize(
                mean=config['preprocessing']['normalize']['mean'],
                std=config['preprocessing']['normalize']['std']
            ),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            min_visibility=0.1,
            label_fields=['class_labels']
        )
    )


def create_data_loaders(config):
    """
    Create training and validation data loaders

    Args:
        config (dict): Data configuration

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load data config
    data_config = config

    # Create transforms
    train_transforms = get_train_transforms(data_config)
    val_transforms = get_val_transforms(data_config)

    # Create datasets
    train_dataset = COCODataset(
        root_dir=data_config['dataset']['train_path'],
        annotation_file=os.path.join(data_config['dataset']['train_path'], 'annotations.json'),
        transforms=train_transforms,
        image_size=(
            data_config['preprocessing']['resize']['height'],
            data_config['preprocessing']['resize']['width']
        )
    )

    val_dataset = COCODataset(
        root_dir=data_config['dataset']['val_path'],
        annotation_file=os.path.join(data_config['dataset']['val_path'], 'annotations.json'),
        transforms=val_transforms,
        image_size=(
            data_config['preprocessing']['resize']['height'],
            data_config['preprocessing']['resize']['width']
        )
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['dataloader']['batch_size'],
        shuffle=data_config['dataloader']['shuffle'],
        num_workers=data_config['dataloader']['num_workers'],
        pin_memory=data_config['dataloader']['pin_memory'],
        drop_last=data_config['dataloader']['drop_last'],
        collate_fn=yolo_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=data_config['dataloader']['num_workers'],
        pin_memory=data_config['dataloader']['pin_memory'],
        drop_last=False,
        collate_fn=yolo_collate_fn
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Load configuration for testing
    with open("config/data_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)

    # Print information about the data
    print(f"Train loader: {len(train_loader.dataset)} samples")
    print(f"Val loader: {len(val_loader.dataset)} samples")

    # Test a batch
    for batch in train_loader:
        print(f"Batch size: {batch['images'].shape}")
        print(f"Number of boxes: {[len(boxes) for boxes in batch['boxes']]}")
        break