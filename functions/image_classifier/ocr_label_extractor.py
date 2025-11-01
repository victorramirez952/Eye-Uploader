#!/usr/bin/env python3
"""
OCR Label Extraction Module

This module handles preprocessing and label extraction from images using OCR
to identify eye labels (OD/OS) for classification purposes.
"""

import os
import shutil
import cv2
import numpy as np
import easyocr
from tqdm import tqdm


class OCRLabelExtractor:
    def __init__(self, sections=24, healthy_eye_folder="HealthyEye",
                 sinister_eye_folder="SinisterEye", no_label_folder="NoLabel"):
        self.sections = sections
        self.healthy_eye_folder = healthy_eye_folder
        self.sinister_eye_folder = sinister_eye_folder
        self.no_label_folder = no_label_folder

        # Initialize OCR reader
        self.ocr_reader = easyocr.Reader(['en', 'es'], gpu=False)

        # Setup output folders
        # self._setup_folders()

    def _setup_folders(self):
        """Setup output folders for eye classification"""
        folders = [self.healthy_eye_folder, self.sinister_eye_folder, self.no_label_folder]
        for folder in folders:
            if folder is not None and isinstance(folder, str):
                if not os.path.exists(folder):
                    os.makedirs(folder)

    def split_image_into_sections(self, image, sections=24):
        """Split an image into n sections. Default is 24 sections."""
        h, w = image.shape

        # Handle special case: 1 section means no splitting
        if sections == 1:
            return [image]

        # Calculate grid dimensions based on sections
        if sections == 6:
            rows, cols = 3, 2
        elif sections == 4:
            rows, cols = 2, 2
        elif sections == 8:
            rows, cols = 2, 4
        elif sections == 9:
            rows, cols = 3, 3
        elif sections == 12:
            rows, cols = 3, 4
        elif sections == 16:
            rows, cols = 4, 4
        elif sections == 20:
            rows, cols = 4, 5
        elif sections == 24:
            rows, cols = 4, 6
        else:
            # For other values, try to find a reasonable grid
            rows = int(sections ** 0.5)
            cols = sections // rows
            if rows * cols < sections:
                cols += 1

        row_height = h // rows
        col_width = w // cols

        sections_list = []
        # Process sections from top to bottom, left to right
        for row in range(rows):
            for col in range(cols):
                if len(sections_list) >= sections:
                    break
                y_start = row * row_height
                y_end = (row + 1) * row_height if row < rows - 1 else h
                x_start = col * col_width
                x_end = (col + 1) * col_width if col < cols - 1 else w

                section = image[y_start:y_end, x_start:x_end]
                sections_list.append(section)

        return sections_list

    def preprocess_image_for_ocr(self, image):
        """Preprocess image for better OCR results."""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image.astype(np.uint8))

        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        return sharpened

    def extract_label_optimized(self, image, valid_labels=['OS', 'OD']):
        """Extract label using optimized OCR processing with startswith check."""
        processed_image = self.preprocess_image_for_ocr(image)
        image_sections = self.split_image_into_sections(processed_image, self.sections)

        # Process each section until a valid label is found
        for i, section in enumerate(image_sections):
            if len(section.shape) == 2:
                section_rgb = cv2.cvtColor(section, cv2.COLOR_GRAY2RGB)
            else:
                section_rgb = section

            try:
                detections = self.ocr_reader.readtext(section_rgb, detail=0, width_ths=0.7, height_ths=0.7)
                for text in detections:
                    text_clean = text.strip().upper()

                    # Check exact match first
                    if text_clean in valid_labels:
                        return text_clean

                    # Check if text starts with any valid label
                    for label in valid_labels:
                        if text_clean.startswith(label):
                            return label

            except Exception as e:
                pass

        return "No Label"

    def classify_images_with_ocr(self, image_paths, confidences_melanoma):
        """
        Classify images based on OCR labels and melanoma predictions.
        Returns paths of original images classified as Affected Eye (melanoma).
        Does not save images to disk.
        """
        if not image_paths:
            print("No images to classify with OCR")
            return []

        # Extract confidence scores for non-melanoma
        confidences_no_melanoma = [1 - conf for conf in confidences_melanoma]

        # Sort indices for pipeline
        sorted_indices_no_melanoma = np.argsort(confidences_no_melanoma)

        # Perform OCR-based classification
        label_cache = {}
        found_label = ""
        classification = {}

        valid_labels = ["OD", "OS"]
        index = 0

        # Find master label
        while found_label not in valid_labels and index < len(sorted_indices_no_melanoma):
            image_index = sorted_indices_no_melanoma[index]

            if image_index < len(image_paths):
                img_path = image_paths[image_index]
                filename = os.path.basename(img_path)

                if filename in label_cache:
                    found_label = label_cache[filename]
                else:
                    try:
                        with open(img_path, 'rb') as f:
                            img_array = np.frombuffer(f.read(), dtype=np.uint8)
                        original_img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

                        if original_img is not None:
                            found_label = self.extract_label_optimized(original_img, valid_labels)
                            label_cache[filename] = found_label
                        else:
                            found_label = "No Label"
                    except Exception as e:
                        found_label = "No Label"

            index += 1

        # Process classification if valid label is found
        if found_label in valid_labels:
            alternate_label = "OS" if found_label == "OD" else "OD"

            classification[found_label] = "melanoma"
            classification[alternate_label] = "NoMelanoma"

            affected_eye_images = []

            # Process all images and collect affected eye paths
            with tqdm(image_paths, desc="Classifying images with OCR") as pbar:
                for i, img_path in enumerate(pbar):
                    filename = os.path.basename(img_path)

                    # Extract OCR label for this image
                    if filename in label_cache:
                        current_ocr_label = label_cache[filename]
                    else:
                        try:
                            with open(img_path, 'rb') as f:
                                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                            current_img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

                            if current_img is not None:
                                current_ocr_label = self.extract_label_optimized(current_img, valid_labels)
                                label_cache[filename] = current_ocr_label
                            else:
                                current_ocr_label = "No Label"
                        except Exception as e:
                            current_ocr_label = "No Label"

                    # Determine classification
                    if current_ocr_label in valid_labels:
                        image_classification = classification[current_ocr_label]
                    else:
                        image_classification = "NA"

                    # Add to affected eye list if classified as melanoma
                    if image_classification == "melanoma":
                        affected_eye_images.append(img_path)

            print(f"\nAffected Eye images identified: {len(affected_eye_images)}")
            return affected_eye_images

        else:
            print(f"✗ ERROR: No valid label found for melanoma classification")
            return []