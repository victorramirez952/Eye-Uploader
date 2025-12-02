#!/usr/bin/env python3
"""
Melanoma Classification Module

This module handles preprocessing and classification of ultrasound images
to identify melanoma vs non-melanoma cases.
"""

import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("Warning: OpenVINO not available. Falling back to TensorFlow/Keras.")


class MelanomaClassifier:
    def __init__(self, model_path, melanoma_folder="MelanomaImages",
                 non_melanoma_folder="NonMelanomaImages", verbose=True):
        self.model_path = model_path
        self.melanoma_folder = melanoma_folder
        self.non_melanoma_folder = non_melanoma_folder
        self.use_openvino = OPENVINO_AVAILABLE
        self.verbose = verbose

        # Load melanoma classification model
        self.model = self._load_model()

        # Setup output folders
        # self._setup_folders()

    def _load_model(self):
        """Load the trained melanoma classification model (OpenVINO or TensorFlow)"""
        try:
            if self.use_openvino:
                # Try OpenVINO first
                try:
                    # Change .keras extension to .xml for OpenVINO model
                    if self.model_path.endswith('.keras'):
                        openvino_model_path = self.model_path.replace('.keras', '.xml')
                    else:
                        openvino_model_path = self.model_path
                    
                    if not os.path.exists(openvino_model_path):
                        print(f"OpenVINO model not found at {openvino_model_path}, falling back to Keras")
                        self.use_openvino = False
                    else:
                        core = ov.Core()
                        ov_model = core.read_model(openvino_model_path)
                        compiled_model = core.compile_model(ov_model, 'CPU')
                        print("Melanoma classifier loaded with OpenVINO")
                        return compiled_model
                except Exception as ov_error:
                    print(f"OpenVINO loading failed: {ov_error}. Falling back to Keras.")
                    self.use_openvino = False
            
            # Fallback to TensorFlow/Keras
            from tensorflow import keras
            keras_model_path = self.model_path.replace('.xml', '.keras') if self.model_path.endswith('.xml') else self.model_path
            
            if not os.path.exists(keras_model_path):
                # Try original path if replacement didn't work
                keras_model_path = self.model_path
            
            model = keras.models.load_model(keras_model_path)
            print("Melanoma classifier loaded with TensorFlow/Keras")
            return model
        except Exception as e:
            print(f"Error loading melanoma classification model: {e}")
            raise

    def _setup_folders(self):
        """Setup output folders for melanoma and non-melanoma images"""
        folders = [self.melanoma_folder, self.non_melanoma_folder]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def process_and_pad_image(self, image, target_size, is_rgb=False):
        """Resize image maintaining aspect ratio and pad to target size"""
        if image is None:
            return None

        if not isinstance(image, np.ndarray):
            return None

        # Handle different image dimensions - preserve RGB if specified
        if len(image.shape) == 3 and not is_rgb:
            # Convert RGB to grayscale only if not preserving RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 2 and is_rgb:
            # Convert grayscale to RGB if RGB is requested
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) not in [2, 3]:
            return None

        if image.dtype != np.uint8:
            # Handle different data types
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        h, w = image.shape[:2]  # Get height and width for both grayscale and color images
        if h == 0 or w == 0:
            return None

        scale = min(target_size / h, target_size / w)
        new_size = (int(w * scale), int(h * scale))

        if new_size[0] == 0 or new_size[1] == 0:
            return None

        img_resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        y_offset = (target_size - new_size[1]) // 2
        x_offset = (target_size - new_size[0]) // 2

        if is_rgb:
            new_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255  # White background for color
            new_img[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0], :] = img_resized
        else:
            new_img = np.ones((target_size, target_size), dtype=np.uint8) * 255  # White background
            new_img[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = img_resized

        return new_img

    def preprocess_image_for_melanoma(self, image_path, target_size=224):
        """Preprocess image for melanoma classification"""
        try:
            with open(image_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Load in RGB color

            if img is None:
                return None

            processed_img = self.process_and_pad_image(img, target_size, is_rgb=True)
            return processed_img

        except Exception as e:
            print(f"Error preprocessing image for melanoma: {e}")
            return None

    def classify_melanoma_images(self, us_image_paths, confidence_threshold=0.5):
        """
        Classify US images for melanoma without saving to disk.
        Returns confidence scores and paths of images with melanoma confidence > threshold.
        
        Args:
            us_image_paths: List of image paths to classify
            confidence_threshold: Minimum confidence for melanoma classification (default: 0.5)
        """
        if not us_image_paths:
            print("No US images to classify for melanoma")
            return None

        # Preprocess all US images for melanoma classification
        processed_images = []
        valid_image_paths = []

        iterator = tqdm(us_image_paths, desc="Preprocessing for melanoma classification", disable=not self.verbose)
        for image_path in iterator:
            processed_img = self.preprocess_image_for_melanoma(image_path, 224)
            if processed_img is not None:
                processed_images.append(processed_img)
                valid_image_paths.append(image_path)

        if not processed_images:
            print("No valid images for melanoma classification")
            return None

        # Convert to numpy array and normalize
        processed_images = np.array(processed_images)
        processed_images_normalized = processed_images.astype(np.float32) / 255.0

        # Make predictions using OpenVINO or TensorFlow
        predictions = []
        batch_size = 8
        
        if self.use_openvino:
            # OpenVINO inference
            for i in range(0, len(processed_images_normalized), batch_size):
                batch = processed_images_normalized[i:i + batch_size]
                # OpenVINO inference - use input index 0
                output = self.model({0: batch})
                # Get the output tensor (first output)
                batch_predictions = list(output.values())[0]
                predictions.extend(batch_predictions)
        else:
            # TensorFlow/Keras inference
            predictions = self.model.predict(processed_images_normalized, batch_size=batch_size, verbose=0)

        # Extract confidence scores
        confidences_melanoma = [float(pred[0]) for pred in predictions]
        
        # Filter images by confidence threshold
        filtered_confidences = []
        filtered_paths = []
        
        for confidence, path in zip(confidences_melanoma, valid_image_paths):
            if confidence > confidence_threshold:
                filtered_confidences.append(confidence)
                filtered_paths.append(path)
        
        filtered_count = len(filtered_paths)
        total_count = len(valid_image_paths)
        print(f"Filtered: {filtered_count}/{total_count} images with melanoma confidence > {confidence_threshold}")

        return filtered_confidences, filtered_paths