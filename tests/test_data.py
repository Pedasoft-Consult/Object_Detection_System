#!/usr/bin/env python
"""
Test script for the data loading and preprocessing components.

This script tests various aspects of data handling functionality.
"""

import os
import sys
import unittest
import yaml
import numpy as np
import logging
from PIL import Image, ImageDraw
import random
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class TestDataLoading(unittest.TestCase):
    """Test cases for data loading and preprocessing functionality"""

    def setUp(self):
        """Set up test environment"""
        # Create test directories
        self.test_data_dir = "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)

        # Generate test images and annotations
        self.generate_test_dataset(num_images=5)

        # Load configuration
        try:
            with open("config/data_config.yaml", "r") as f:
                self.data_config = yaml.safe_load(f)
            logger.info("Loaded data configuration")
        except FileNotFoundError:
            logger.warning("Data configuration not found. Creating default config for testing.")
            self.data_config = {
                'dataset': {
                    'name': "test",
                    'classes': ["person", "car", "dog", "cat", "bicycle"],
                    'train_split': 0.8,
                    'val_split': 0.1,
                    'test_split': 0.1
                },
                'preprocessing': {
                    'resize': {
                        'height': 416,
                        'width': 416
                    },
                    'normalize': {
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]
                    }
                }
            }

    def tearDown(self):
        """Clean up after tests"""
        # Remove test data directory
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def generate_test_dataset(self, num_images=5):
        """
        Generate a small test dataset with images and annotations

        Args:
            num_images (int): Number of test images to generate
        """
        # Create images directory
        images_dir = os.path.join(self.test_data_dir, "images")
        annotations_dir = os.path.join(self.test_data_dir, "annotations")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        # Class names for test
        class_names = ["person", "car", "dog", "cat", "bicycle"]

        # Generate images
        for i in range(num_images):
            # Create a test image
            width, height = 640, 480
            image = Image.new("RGB", (width, height), color=(240, 240, 240))
            draw = ImageDraw.Draw(image)

            # Generate random objects in the image
            num_objects = random.randint(1, 5)
            annotations = []

            for j in range(num_objects):
                # Random rectangle for object
                x1 = random.randint(50, width - 150)
                y1 = random.randint(50, height - 150)
                w = random.randint(50, 150)
                h = random.randint(50, 150)
                x2 = x1 + w
                y2 = y1 + h

                # Random class
                class_id = random.randint(0, len(class_names) - 1)
                class_name = class_names[class_id]

                # Random color
                color = (
                    random.randint(0, 200),
                    random.randint(0, 200),
                    random.randint(0, 200)
                )

                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                # Add annotation
                # [class_id, x_center, y_center, width, height] (normalized)
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                norm_width = w / width
                norm_height = h / height

                annotations.append([class_id, x_center, y_center, norm_width, norm_height])

            # Save image
            image_path = os.path.join(images_dir, f"image_{i:04d}.jpg")
            image.save(image_path)

            # Save annotation (YOLO format)
            annotation_path = os.path.join(annotations_dir, f"image_{i:04d}.txt")
            with open(annotation_path, "w") as f:
                for ann in annotations:
                    f.write(" ".join(map(str, ann)) + "\n")

        logger.info(f"Generated test dataset with {num_images} images")

    def test_image_loading(self):
        """Test basic image loading from the test dataset"""
        # Get list of test images
        images_dir = os.path.join(self.test_data_dir, "images")
        image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

        self.assertGreater(len(image_files), 0, "No test images found")

        # Load a test image
        test_image_path = os.path.join(images_dir, image_files[0])
        image = Image.open(test_image_path)

        # Check image properties
        self.assertIsNotNone(image)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (640, 480))

        logger.info("Image loading test passed")

    def test_annotation_loading(self):
        """Test annotation loading from the test dataset"""
        # Get list of test annotations
        annotations_dir = os.path.join(self.test_data_dir, "annotations")
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]

        self.assertGreater(len(annotation_files), 0, "No test annotations found")

        # Load a test annotation
        test_annotation_path = os.path.join(annotations_dir, annotation_files[0])
        with open(test_annotation_path, "r") as f:
            lines = f.readlines()

        # Parse annotations
        annotations = []
        for line in lines:
            values = list(map(float, line.strip().split()))
            annotations.append(values)

        # Check annotation format
        self.assertGreaterEqual(len(annotations), 0)

        for ann in annotations:
            # Each annotation should have 5 values: class_id, x_center, y_center, width, height
            self.assertEqual(len(ann), 5)

            # Values should be within expected ranges
            class_id = int(ann[0])
            x_center, y_center, width, height = ann[1:5]

            self.assertGreaterEqual(class_id, 0)
            self.assertLess(class_id, len(self.data_config['dataset']['classes']))

            self.assertGreaterEqual(x_center, 0.0)
            self.assertLessEqual(x_center, 1.0)

            self.assertGreaterEqual(y_center, 0.0)
            self.assertLessEqual(y_center, 1.0)

            self.assertGreater(width, 0.0)
            self.assertLessEqual(width, 1.0)

            self.assertGreater(height, 0.0)
            self.assertLessEqual(height, 1.0)

        logger.info("Annotation loading test passed")

    def test_image_preprocessing(self):
        """Test image preprocessing functions"""
        # Test image resizing to model input size
        images_dir = os.path.join(self.test_data_dir, "images")
        image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

        if not image_files:
            self.skipTest("No test images found")

        # Load a test image
        test_image_path = os.path.join(images_dir, image_files[0])
        image = Image.open(test_image_path)

        # Resize image
        target_size = (
            self.data_config['preprocessing']['resize']['width'],
            self.data_config['preprocessing']['resize']['height']
        )
        resized_image = image.resize(target_size)

        # Check resized image
        self.assertEqual(resized_image.size, target_size)

        # Convert to numpy and normalize
        image_np = np.array(resized_image) / 255.0

        # Check normalized image values
        self.assertGreaterEqual(np.min(image_np), 0.0)
        self.assertLessEqual(np.max(image_np), 1.0)

        # Apply mean/std normalization
        mean = self.data_config['preprocessing']['normalize']['mean']
        std = self.data_config['preprocessing']['normalize']['std']

        # Apply normalization
        normalized = (image_np - np.array(mean)) / np.array(std)

        logger.info("Image preprocessing test passed")

    def test_data_augmentation(self):
        """Test basic data augmentation techniques"""
        # Load a test image
        images_dir = os.path.join(self.test_data_dir, "images")
        image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

        if not image_files:
            self.skipTest("No test images found")

        test_image_path = os.path.join(images_dir, image_files[0])
        image = Image.open(test_image_path)

        # Test horizontal flip
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        self.assertEqual(flipped.size, image.size)

        # Test random crop
        width, height = image.size
        crop_size = (width // 2, height // 2)
        left = random.randint(0, width - crop_size[0])
        top = random.randint(0, height - crop_size[1])
        cropped = image.crop((left, top, left + crop_size[0], top + crop_size[1]))
        self.assertEqual(cropped.size, crop_size)

        # Test rotation
        angle = random.randint(-30, 30)
        rotated = image.rotate(angle, expand=False)
        self.assertEqual(rotated.size, image.size)

        logger.info("Data augmentation test passed")

    def test_batch_creation(self):
        """Test creation of training batches"""
        # Get list of test images and annotations
        images_dir = os.path.join(self.test_data_dir, "images")
        annotations_dir = os.path.join(self.test_data_dir, "annotations")

        image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]

        if not image_files or not annotation_files:
            self.skipTest("No test images or annotations found")

        # Create a small batch
        batch_size = min(3, len(image_files))
        batch_images = []
        batch_annotations = []

        for i in range(batch_size):
            # Load image
            image_path = os.path.join(images_dir, image_files[i])
            image = Image.open(image_path)

            # Resize and convert to numpy
            target_size = (
                self.data_config['preprocessing']['resize']['width'],
                self.data_config['preprocessing']['resize']['height']
            )
            resized_image = image.resize(target_size)
            image_np = np.array(resized_image) / 255.0

            # Load annotation
            annotation_path = os.path.join(annotations_dir, annotation_files[i])
            with open(annotation_path, "r") as f:
                lines = f.readlines()

            # Parse annotations
            annotations = []
            for line in lines:
                values = list(map(float, line.strip().split()))
                annotations.append(values)

            batch_images.append(image_np)
            batch_annotations.append(annotations)

        # Check batch
        self.assertEqual(len(batch_images), batch_size)
        self.assertEqual(len(batch_annotations), batch_size)

        # Check batch shape
        batch_array = np.array(batch_images)
        self.assertEqual(batch_array.shape, (batch_size, target_size[1], target_size[0], 3))

        logger.info("Batch creation test passed")

    # Fix for test_data_splits in test_data.py
    # Ensure validation indices are created when the dataset is too small

    # Fix for test_data_splits in test_data.py
    # Ensure validation indices are created when the dataset is too small

    def test_data_splits(self):
        """Test dataset splitting into train/val/test"""
        # Get list of all image files
        images_dir = os.path.join(self.test_data_dir, "images")
        image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

        if len(image_files) < 3:
            self.skipTest("Not enough test images for splitting")

        # Get split ratios from config
        try:
            train_split = self.data_config['dataset']['train_split']
            val_split = self.data_config['dataset']['val_split']
            test_split = self.data_config['dataset']['test_split']
        except KeyError:
            # Default values in case the config doesn't have specific split values
            train_split = 0.8
            val_split = 0.1
            test_split = 0.1
            logger.warning("Using default split ratios: 80% train, 10% val, 10% test")

        # Calculate split indices
        num_samples = len(image_files)

        # Ensure at least one sample in each split if the split ratio is non-zero
        min_samples_per_split = 1

        # Adjust calculation to ensure at least one sample per split if ratio > 0
        if val_split > 0:
            num_val = max(min_samples_per_split, int(val_split * num_samples))
        else:
            num_val = 0

        if test_split > 0:
            num_test = max(min_samples_per_split, int(test_split * num_samples))
        else:
            num_test = 0

        # Remaining samples go to train split
        num_train = num_samples - num_val - num_test

        # If we don't have enough samples, prioritize train over val over test
        if num_train <= 0:
            if num_val > 0:
                num_val = max(0, num_val - 1)
                num_train = 1
            elif num_test > 0:
                num_test = max(0, num_test - 1)
                num_train = 1

        # Create splits
        train_indices = list(range(num_train))
        val_indices = list(range(num_train, num_train + num_val))
        test_indices = list(range(num_train + num_val, num_samples))

        # Check split sizes
        self.assertEqual(len(train_indices) + len(val_indices) + len(test_indices), num_samples)

        # Check that splits are non-empty (if they should be)
        if train_split > 0:
            self.assertGreater(len(train_indices), 0, "Train split should have at least one sample")
        if val_split > 0:
            self.assertGreater(len(val_indices), 0, "Validation split should have at least one sample")
        if test_split > 0:
            self.assertGreater(len(test_indices), 0, "Test split should have at least one sample")

        logger.info("Data splits test passed")

if __name__ == "__main__":
    unittest.main()