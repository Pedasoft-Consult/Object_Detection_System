import cv2
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentation_pipeline(config):
    """
    Create comprehensive augmentation pipeline from config

    Args:
        config (dict): Augmentation configuration

    Returns:
        A.Compose: Composed augmentation pipeline
    """
    height, width = config['preprocessing']['resize']['height'], config['preprocessing']['resize']['width']

    # Basic transformations (always applied)
    basic_transforms = [
        A.LongestMaxSize(max_size=max(height, width)),
        A.PadIfNeeded(
            min_height=height,
            min_width=width,
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
    ]

    # Augmentations (applied with probability)
    aug_transforms = []

    if config['augmentation']['enabled']:
        # Geometric transforms
        if config['augmentation']['horizontal_flip']['enabled']:
            aug_transforms.append(A.HorizontalFlip(
                p=config['augmentation']['horizontal_flip']['probability']
            ))

        if config['augmentation']['vertical_flip']['enabled']:
            aug_transforms.append(A.VerticalFlip(
                p=config['augmentation']['vertical_flip']['probability']
            ))

        if config['augmentation']['rotate']['enabled']:
            aug_transforms.append(A.Rotate(
                limit=config['augmentation']['rotate']['max_angle'],
                p=config['augmentation']['rotate']['probability'],
                border_mode=cv2.BORDER_CONSTANT
            ))

        if config['augmentation']['scale']['enabled']:
            aug_transforms.append(A.RandomScale(
                scale_limit=config['augmentation']['scale']['range'],
                p=config['augmentation']['scale']['probability']
            ))

        if config['augmentation']['translate']['enabled']:
            aug_transforms.append(A.ShiftScaleRotate(
                shift_limit=config['augmentation']['translate']['percent'],
                scale_limit=0,
                rotate_limit=0,
                p=config['augmentation']['translate']['probability'],
                border_mode=cv2.BORDER_CONSTANT
            ))

        # Color transforms
        if config['augmentation']['brightness']['enabled'] or config['augmentation']['contrast']['enabled']:
            aug_transforms.append(A.RandomBrightnessContrast(
                brightness_limit=config['augmentation']['brightness']['factor'],
                contrast_limit=config['augmentation']['contrast']['factor'],
                p=max(config['augmentation']['brightness']['probability'],
                      config['augmentation']['contrast']['probability'])
            ))

        if config['augmentation']['hue']['enabled'] or config['augmentation']['saturation']['enabled']:
            aug_transforms.append(A.HueSaturationValue(
                hue_shift_limit=int(config['augmentation']['hue']['factor'] * 180),
                sat_shift_limit=int(config['augmentation']['saturation']['factor'] * 255),
                p=max(config['augmentation']['hue']['probability'],
                      config['augmentation']['saturation']['probability'])
            ))

        if config['augmentation']['blur']['enabled']:
            aug_transforms.append(A.Blur(
                blur_limit=config['augmentation']['blur']['kernel_size'],
                p=config['augmentation']['blur']['probability']
            ))

        if config['augmentation']['noise']['enabled']:
            if config['augmentation']['noise']['type'] == 'gaussian':
                aug_transforms.append(A.GaussNoise(
                    p=config['augmentation']['noise']['probability']
                ))
            elif config['augmentation']['noise']['type'] == 'salt':
                aug_transforms.append(A.ToGray(
                    p=config['augmentation']['noise']['probability']
                ))

    # Mosaic augmentation
    mosaic_enabled = config['preprocessing'].get('mosaic', False)

    # Add normalization and conversion to tensor (always applied last)
    final_transforms = [
        A.Normalize(
            mean=config['preprocessing']['normalize']['mean'],
            std=config['preprocessing']['normalize']['std']
        ),
        ToTensorV2()
    ]

    # Combine all transforms
    all_transforms = basic_transforms + aug_transforms + final_transforms

    return A.Compose(
        all_transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            min_visibility=0.1,
            label_fields=['class_labels']
        )
    )


def apply_mosaic_augmentation(images, boxes, labels, size):
    """
    Apply mosaic augmentation - combines 4 images into one

    Args:
        images (list): List of images
        boxes (list): List of bounding boxes for each image
        labels (list): List of labels for each image
        size (tuple): Output image size (h, w)

    Returns:
        tuple: (mosaic_image, mosaic_boxes, mosaic_labels)
    """
    height, width = size

    # Initialize mosaic image
    mosaic_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Get four random images
    indices = random.sample(range(len(images)), min(4, len(images)))

    # Scale coordinates
    scaled_boxes = []
    scaled_labels = []

    # Define positions (2x2 grid)
    positions = [
        (0, 0, width // 2, height // 2),  # top-left
        (width // 2, 0, width, height // 2),  # top-right
        (0, height // 2, width // 2, height),  # bottom-left
        (width // 2, height // 2, width, height)  # bottom-right
    ]

    for i, index in enumerate(indices):
        # Get image and its boxes/labels
        img = images[index]
        img_boxes = boxes[index]
        img_labels = labels[index]

        # Get position
        x1, y1, x2, y2 = positions[i]
        img_h, img_w = img.shape[:2]

        # Scale the image to fit the position
        scale_h, scale_w = (y2 - y1) / img_h, (x2 - x1) / img_w

        # Resize the image
        resized_img = cv2.resize(img, (x2 - x1, y2 - y1))

        # Place the image in the mosaic
        mosaic_img[y1:y2, x1:x2] = resized_img

        # Scale boxes
        for box, label in zip(img_boxes, img_labels):
            # Original coordinates (normalized 0-1)
            orig_x1 = box[0] - box[2] / 2
            orig_y1 = box[1] - box[3] / 2
            orig_x2 = box[0] + box[2] / 2
            orig_y2 = box[1] + box[3] / 2

            # Scale and shift coordinates
            new_x1 = orig_x1 * scale_w * img_w + x1
            new_y1 = orig_y1 * scale_h * img_h + y1
            new_x2 = orig_x2 * scale_w * img_w + x1
            new_y2 = orig_y2 * scale_h * img_h + y1

            # Convert back to normalized center format
            new_x = (new_x1 + new_x2) / 2 / width
            new_y = (new_y1 + new_y2) / 2 / height
            new_w = (new_x2 - new_x1) / width
            new_h = (new_y2 - new_y1) / height

            # Add to scaled boxes
            scaled_boxes.append([new_x, new_y, new_w, new_h])
            scaled_labels.append(label)

    return mosaic_img, np.array(scaled_boxes), np.array(scaled_labels)


def mixup_augmentation(image1, boxes1, labels1, image2, boxes2, labels2, alpha=0.5):
    """
    Apply mixup augmentation - blend two images

    Args:
        image1, image2: Input images
        boxes1, boxes2: Bounding boxes for each image
        labels1, labels2: Labels for each image
        alpha: Blending factor

    Returns:
        tuple: (blended_image, combined_boxes, combined_labels)
    """
    # Ensure images are same size
    h, w = image1.shape[:2]
    image2 = cv2.resize(image2, (w, h))

    # Blend images
    blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)

    # Combine boxes and labels
    combined_boxes = np.vstack([boxes1, boxes2])
    combined_labels = np.hstack([labels1, labels2])

    return blended_image, combined_boxes, combined_labels


def cutout_augmentation(image, boxes, labels, num_holes=8, max_h_size=8, max_w_size=8):
    """
    Apply cutout augmentation - randomly mask out regions of the image

    Args:
        image: Input image
        boxes: Bounding boxes
        labels: Labels
        num_holes: Number of cutout regions
        max_h_size, max_w_size: Maximum size of cutout regions

    Returns:
        tuple: (cutout_image, boxes, labels)
    """
    h, w = image.shape[:2]
    cutout_image = image.copy()

    for _ in range(num_holes):
        # Random size
        h_size = np.random.randint(1, max_h_size)
        w_size = np.random.randint(1, max_w_size)

        # Random position
        x1 = np.random.randint(0, w - w_size)
        y1 = np.random.randint(0, h - h_size)
        x2 = x1 + w_size
        y2 = y1 + h_size

        # Apply cutout (set to gray)
        cutout_image[y1:y2, x1:x2] = 128

    return cutout_image, boxes, labels


if __name__ == "__main__":
    # Test augmentation functions
    import yaml
    from matplotlib import pyplot as plt

    # Load configuration
    with open("config/data_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create test image and boxes
    test_image = np.ones((416, 416, 3), dtype=np.uint8) * 128
    cv2.rectangle(test_image, (100, 100), (300, 300), (0, 255, 0), -1)

    test_boxes = np.array([[0.5, 0.5, 0.5, 0.5]])  # center x, center y, width, height
    test_labels = np.array([0])

    # Test augmentation pipeline
    aug_pipeline = get_augmentation_pipeline(config)
    augmented = aug_pipeline(image=test_image, bboxes=test_boxes, class_labels=test_labels)

    # Test mosaic
    mosaic_image, mosaic_boxes, mosaic_labels = apply_mosaic_augmentation(
        [test_image, test_image, test_image, test_image],
        [test_boxes, test_boxes, test_boxes, test_boxes],
        [test_labels, test_labels, test_labels, test_labels],
        (416, 416)
    )

    # Test mixup
    mixup_image, mixup_boxes, mixup_labels = mixup_augmentation(
        test_image, test_boxes, test_labels,
        test_image, test_boxes, test_labels
    )

    # Test cutout
    cutout_image, cutout_boxes, cutout_labels = cutout_augmentation(
        test_image, test_boxes, test_labels
    )

    # Display results
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.imshow(test_image)
    plt.title("Original")

    plt.subplot(222)
    plt.imshow(mosaic_image)
    plt.title("Mosaic")

    plt.subplot(223)
    plt.imshow(mixup_image)
    plt.title("Mixup")

    plt.subplot(224)
    plt.imshow(cutout_image)
    plt.title("Cutout")

    plt.tight_layout()
    plt.savefig("augmentation_test.png")
    plt.close()