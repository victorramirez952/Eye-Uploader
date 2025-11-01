#!/usr/bin/env python3
"""
Ultrasound Classification Module

This module handles preprocessing and classification of images to identify
ultrasound vs non-ultrasound images.
"""

import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class USClassifier:
    def __init__(self, model_path, us_folder="USImages", non_us_folder="NonUS"):
        self.model_path = model_path
        self.us_folder = us_folder
        self.non_us_folder = non_us_folder

        # Load US classification model
        self.model = self._load_model()

        # Setup output folders
        # self._setup_folders()

    def _load_model(self):
        """Load the trained US classification model"""
        try:
            model = tf.keras.models.load_model(self.model_path)
            return model
        except Exception as e:
            print(f"Error loading US classification model: {e}")
            raise

    def _setup_folders(self):
        """Setup output folders for US and non-US images"""
        folders = [self.us_folder, self.non_us_folder]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def process_and_pad_image(self, image, target_size):
        """Resize image maintaining aspect ratio and pad to target size"""
        if image is None:
            return None

        if not isinstance(image, np.ndarray):
            return None

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return None

        scale = min(target_size / h, target_size / w)
        new_size = (int(w * scale), int(h * scale))

        if new_size[0] == 0 or new_size[1] == 0:
            return None

        img_resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        new_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

        y_offset = (target_size - new_size[1]) // 2
        x_offset = (target_size - new_size[0]) // 2

        new_img[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0], :] = img_resized

        return new_img

    def preprocess_image_for_us_classification(self, image_path):
        """Preprocess image for US classification model"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                return None

            # Model expects 224x224 input based on architecture
            model_input_size = 224
            processed_img = self.process_and_pad_image(img, model_input_size)
            if processed_img is None:
                return None

            processed_img = np.array(processed_img, dtype=np.uint8)
            processed_img = processed_img.astype('float32') / 255.0
            processed_img = processed_img.reshape((1, model_input_size, model_input_size, 3))

            return processed_img

        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None

    def classify_us_image(self, preprocessed_image):
        """Classify image using the US classification model"""
        try:
            prediction = self.model.predict(preprocessed_image, verbose=0)
            return (prediction > 0.5).astype(int)[0][0]
        except Exception as e:
            print(f"Error classifying US image: {e}")
            return 0

    def classify_images(self, image_paths):
        """
        Classify US images without saving to disk.
        Returns only paths of original images classified as US.
        """
        us_images = []

        with tqdm(image_paths, desc="Processing for US classification") as pbar:
            for image_path in pbar:
                # Preprocess image
                preprocessed_image = self.preprocess_image_for_us_classification(image_path)
                if preprocessed_image is None:
                    continue

                # Classify image
                classification = self.classify_us_image(preprocessed_image)

                # If classified as ultrasound (1), add to US images
                if classification == 1:
                    us_images.append(image_path)

        return us_images