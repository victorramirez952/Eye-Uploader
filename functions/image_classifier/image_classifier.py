#!/usr/bin/env python3
"""
Image Classifier Module - Main Orchestrator

This module orchestrates the image classification process using US classification,
melanoma classification, and OCR label extraction to identify Affected Eye images.

Input: Directory path containing images to be classified.
Output: List of image paths classified as Affected Eye.
"""

import os
from pathlib import Path


class ImageClassifier:
    def __init__(self, us_model_path=None, melanoma_model_path=None, sections=24):
        """
        Initialize the Image Classifier orchestrator.
        
        Args:
            us_model_path: Path to US classification model
            melanoma_model_path: Path to melanoma classification model
            sections: Number of sections for OCR processing (default: 24)
        """
        self.us_model_path = us_model_path or self._get_default_model_path('melanoma_classifier_mobile_net.keras')
        self.melanoma_model_path = melanoma_model_path or self._get_default_model_path('mobile_net_best_model.keras')
        self.sections = sections
        
        # Lazy loading - components will be initialized when needed
        self._us_classifier = None
        self._melanoma_classifier = None
        self._ocr_extractor = None
    
    def _get_default_model_path(self, model_filename):
        """Get default model path in keras_models directory (cross-platform)"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, 'keras_models', model_filename)
    
    @property
    def us_classifier(self):
        """Lazy load US classifier"""
        if self._us_classifier is None:
            from .us_classifier import USClassifier
            # Don't save images to disk
            self._us_classifier = USClassifier(self.us_model_path, us_folder=None, non_us_folder=None)
        return self._us_classifier
    
    @property
    def melanoma_classifier(self):
        """Lazy load melanoma classifier"""
        if self._melanoma_classifier is None:
            from .melanoma_classifier import MelanomaClassifier
            # Don't save images to disk
            self._melanoma_classifier = MelanomaClassifier(
                self.melanoma_model_path,
                melanoma_folder=None,
                non_melanoma_folder=None
            )
        return self._melanoma_classifier
    
    @property
    def ocr_extractor(self):
        """Lazy load OCR extractor"""
        if self._ocr_extractor is None:
            from .ocr_label_extractor import OCRLabelExtractor
            # Don't save images to disk
            self._ocr_extractor = OCRLabelExtractor(
                self.sections,
                healthy_eye_folder=None,
                sinister_eye_folder=None,
                no_label_folder=None
            )
        return self._ocr_extractor
    
    def classify_images(self, image_directory):
        """
        Classify images from a directory and return paths of Affected Eye images.
        
        Args:
            image_directory: Path to directory containing images to classify
            
        Returns:
            List of image paths classified as Affected Eye
        """
        # Get all image paths from directory
        image_paths = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        for filename in os.listdir(image_directory):
            if Path(filename).suffix.lower() in image_extensions:
                image_paths.append(os.path.join(image_directory, filename))
        
        if not image_paths:
            print("No images found in directory")
            return []
        
        print(f"Found {len(image_paths)} images to classify")
        
        # Step 1: US Classification
        print("\n=== US Classification ===")
        us_images = self.us_classifier.classify_images(image_paths)
        print(f"US images identified: {len(us_images)}")
        
        if not us_images:
            print("No US images found")
            return []
        
        # Step 2: Melanoma Classification
        print("\n=== Melanoma Classification ===")
        melanoma_results = self.melanoma_classifier.classify_melanoma_images(us_images)
        
        if not melanoma_results:
            print("No melanoma classification results")
            return []
        
        confidences_melanoma, valid_image_paths = melanoma_results
        print(f"Melanoma classification completed for {len(valid_image_paths)} images")
        
        # Step 3: OCR-based Eye Classification
        print("\n=== OCR Eye Classification ===")
        affected_eye_images = self.ocr_extractor.classify_images_with_ocr(
            valid_image_paths,
            confidences_melanoma
        )
        
        print(f"Affected Eye images identified: {len(affected_eye_images)}")
        
        return affected_eye_images


def classify_directory(image_directory, us_model_path=None, melanoma_model_path=None, sections=24):
    """
    Convenience function to classify images from a directory.
    
    Args:
        image_directory: Path to directory containing images
        us_model_path: Optional path to US classification model
        melanoma_model_path: Optional path to melanoma classification model
        sections: Number of sections for OCR processing (default: 24)
        
    Returns:
        List of image paths classified as Affected Eye
    """
    us_model_path = r"C:/Users/jairr/Documents/UDEM/9noSemestre/PEF/EYE_UPLOADER/functions/image_classifier/keras_models/mobile_net_best_model.keras"
    melanoma_model_path = r"C:/Users/jairr/Documents/UDEM/9noSemestre/PEF/EYE_UPLOADER/functions/image_classifier/keras_models/melanoma_classifier_mobile_net.keras"
    classifier = ImageClassifier(us_model_path, melanoma_model_path, sections)
    return classifier.classify_images(image_directory)
