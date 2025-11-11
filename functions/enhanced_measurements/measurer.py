import cv2
import numpy as np
import os
from PIL import Image
from .morphology import apply_opening, apply_closing, create_kernel
from .segmentation import apply_mean_shift_to_crop
from .circle_fitting import fit_circle_to_border_points
from .image_processing import (
    crop_bottom_image, 
    crop_left_image, 
    apply_binary_mask,
    remove_rows_with_high_white_percentage,
    pixelate_image,
    apply_marching_squares
)
from .distance_field import (
    create_signed_distance_field,
    create_binary_mask_from_sdf,
    measure_vertical_line_distances
)


class Measurer:
    """
    Main class to handle image processing operations including morphological transformations.
    Can be used as a standalone processor or imported as a library.
    """
    
    def __init__(self, temp_folder="tempImages", kernel_size=(5, 5), kernel_shape="rectangular", verbose=False):
        """
        Initialize the Measurer class.
        
        Args:
            temp_folder (str): Directory for temporary images.
            kernel_size (tuple): Size of morphological kernel.
            kernel_shape (str): Shape of morphological kernel.
            verbose (bool): If True, print milestones and save intermediate images.
        """
        self.temp_folder = temp_folder
        self.kernel_size = kernel_size
        self.kernel_shape = kernel_shape
        self.verbose = verbose
        if self.verbose:
            self.setup_directories()
        
    def setup_directories(self):
        """Create the temporary images directory if it doesn't exist and clean it."""
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        else:
            for filename in os.listdir(self.temp_folder):
                file_path = os.path.join(self.temp_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    pass
    
    def load_image(self, image_path):
        """
        Load an image from the specified path.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            numpy.ndarray: Loaded image or None if loading fails.
        """
        if not os.path.exists(image_path):
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        return image
    
    def save_image(self, image, filename):
        """
        Save an image to the temporary images folder.
        
        Args:
            image (numpy.ndarray): Image to save.
            filename (str): Name of the output file.
        """
        if not self.verbose:
            return
        output_path = os.path.join(self.temp_folder, filename)
        cv2.imwrite(output_path, image)
    
    def cut_bottom_rectangle(self, image_path, output_name, percentage):
        """
        Cut a rectangle from the bottom of the image by removing n% of the height.
        
        Args:
            image_path: Path to the input image
            output_name: Name for the output file
            percentage: Percentage of height to cut from bottom
        
        Returns:
            str: Path to the saved cropped image
        """
        img = Image.open(image_path)
        width, height = img.size
        
        cut_height = int(height * percentage / 100)
        new_height = height - cut_height
        
        cropped_img = img.crop((0, 0, width, new_height))
        
        output_path = os.path.join(self.temp_folder, output_name)
        cropped_img.save(output_path)
        
        return output_path
    
    def _process_ruler_measurement(self, image, equivalencies):
        """
        Process ruler measurement from bottom region of image.
        
        Returns:
            tuple: (mm_per_pixel, satisfactory_results)
        """
        if self.verbose:
            print("Cropping bottom region...")
        bottom_image = crop_bottom_image(image)
        self.save_image(bottom_image, "bottom_image.png")
        
        if self.verbose:
            print("Applying mean shift segmentation to bottom region...")
        bottom_segmented = apply_mean_shift_to_crop(bottom_image)
        self.save_image(bottom_segmented, "bottom_image_segmented.png")
        
        if self.verbose:
            print("Cleaning bottom region...")
        bottom_cleaned = remove_rows_with_high_white_percentage(bottom_segmented, percentage_threshold=0.75)
        self.save_image(bottom_cleaned, "bottom_image_cleaned.png")
        
        if self.verbose:
            print("Creating signed distance field...")
        bottom_sdf = create_signed_distance_field(bottom_cleaned)
        self.save_image(bottom_sdf, "bottom_image_sdf.png")
        
        if self.verbose:
            print("Creating binary mask from SDF...")
        bottom_binary_mask = create_binary_mask_from_sdf(bottom_sdf)
        self.save_image(bottom_binary_mask, "bottom_image_binary_mask.png")
        
        if self.verbose:
            print("Measuring vertical line distances...")
        measurement_results = measure_vertical_line_distances(bottom_binary_mask, num_measurements=4)
        median_distance = measurement_results['median_distance']
        average_distance = measurement_results['average_distance']
        mm_per_pixel = equivalencies['median_rule_mm'] / median_distance if median_distance > 0 else 0

        satisfactory_results = False
        if median_distance > 0 and abs(median_distance - average_distance) / median_distance < 0.01:
            satisfactory_results = True
            if self.verbose:
                print(f"Satisfactory results obtained: median={median_distance:.2f}, average={average_distance:.2f}")
        
        return mm_per_pixel, satisfactory_results
    
    def _process_circle_measurement(self, image, mask_path, circle_fit_params, equivalencies):
        """
        Process circle measurement from main image.
        
        Returns:
            float: mm_per_pixel from circle fitting, or 0 if failed
        """
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if mask_path:
            if self.verbose:
                print("Applying binary mask to grayscale image...")
            grayscale_masked = apply_binary_mask(grayscale, mask_path)
            self.save_image(grayscale_masked, "grayscale_image.jpg")
        else:
            grayscale_masked = grayscale
        
        if self.verbose:
            print("Applying morphological operations...")
        opening_result = apply_opening(grayscale_masked, self.kernel_size, self.kernel_shape)
        final_result = apply_closing(opening_result, self.kernel_size, self.kernel_shape)
        self.save_image(final_result, "morphological_result.jpg")
        
        if self.verbose:
            print("Pixelating image...")
        pixelated_result = pixelate_image(final_result, block_size=10)
        self.save_image(pixelated_result, "pixelated_result.jpg")
        
        if self.verbose:
            print("Applying marching squares...")
        bordered_result, border_points, tallest_x, tallest_y = apply_marching_squares(
            pixelated_result, block_size=10, threshold=30, temp_folder=self.temp_folder
        )
        self.save_image(bordered_result, "marching_squares_borders.jpg")
        
        if circle_fit_params is None:
            circle_fit_params = {}
        
        if self.verbose:
            print("Fitting circle to border points...")
        circle_result, circle_params = fit_circle_to_border_points(
            pixelated_result,
            border_points,
            x_asymptote_ratio=circle_fit_params.get('x_asymptote_ratio', 0.25),
            min_x_ratio=circle_fit_params.get('min_x_ratio', 0.35),
            max_x_ratio=circle_fit_params.get('max_x_ratio', 0.75),
            max_iterations=circle_fit_params.get('max_iterations', 100),
            distance_threshold=circle_fit_params.get('distance_threshold', 5.0),
            tallest_cluster_x_max=tallest_x,
            tallest_cluster_y_center=tallest_y
        )
        self.save_image(circle_result, "circle_fit.jpg")
        
        if circle_params is not None:
            cx, cy, radius = circle_params
            diameter = radius * 2
            image_width = image.shape[1]
            min_diameter = image_width * 0.20
            max_diameter = image_width * 0.75

            if diameter > max_diameter or diameter < min_diameter:
                if self.verbose:
                    print(f"Circle diameter {diameter:.2f} is out of bounds, using fallback...")
                mm_per_pixel = equivalencies['median_ap_axis'] / (image_width * 0.65)
                return mm_per_pixel, 'fallback'
            
            mm_per_pixel = equivalencies['median_ap_axis'] / diameter if diameter > 0 else 0
            
            if self.verbose:
                print(f"Circle fitted: diameter={diameter:.2f}, mm_per_pixel={mm_per_pixel:.6f}")
            
            return mm_per_pixel, 'circle'
        
        return 0, 'none'
    
    def process_image(self, image_path, mask_path=None, 
                     circle_fit_params=None, equivalencies=None, test_mode=False, cut_percentage=5):
        """
        Complete image processing pipeline: load, mask, apply morphological operations, and save results.
        
        Args:
            image_path (str): Path to input image.
            mask_path (str): Path to binary mask image (optional).
            circle_fit_params (dict): Parameters for circle fitting (optional).
            equivalencies (dict): Dictionary with 'median_rule_mm' and 'median_ap_axis' keys.
            test_mode (bool): If True, cuts and reprocesses images when satisfactory results obtained.
            cut_percentage (int): Percentage of height to cut from bottom in test mode.
            
        Returns:
            dict: Dictionary with 'mm_per_pixel' and 'method' keys.
        """
        if self.verbose:
            print("Loading image...")
        image = self.load_image(image_path)
        if image is None:
            return None
        
        self.save_image(image, "original_image.jpg")
        
        # Process ruler measurement
        mm_per_pixel_ruler, satisfactory = self._process_ruler_measurement(image, equivalencies)
        
        # If ruler measurement is satisfactory and test_mode is enabled, process with circle
        if satisfactory and test_mode:
            if self.verbose:
                print("Test mode: cutting and reprocessing image...")
            
            edited_image_path = self.cut_bottom_rectangle(image_path, "original_edited.png", cut_percentage)
            if mask_path and os.path.exists(mask_path):
                edited_mask_path = self.cut_bottom_rectangle(mask_path, "mask_edited.png", cut_percentage)
            else:
                edited_mask_path = None
            
            edited_image = self.load_image(edited_image_path)
            if edited_image is None:
                return {'mm_per_pixel': mm_per_pixel_ruler, 'method': 'rule'}
            
            mm_per_pixel_circle, method = self._process_circle_measurement(
                edited_image, edited_mask_path, circle_fit_params, equivalencies
            )
            
            if method in ['circle', 'fallback']:
                return {'mm_per_pixel': mm_per_pixel_circle, 'method': method}
            
            return {'mm_per_pixel': mm_per_pixel_ruler, 'method': 'rule'}
        
        # If ruler measurement is satisfactory but not test mode
        if satisfactory:
            return {'mm_per_pixel': mm_per_pixel_ruler, 'method': 'rule'}
        
        # If ruler measurement is not satisfactory, try circle measurement
        mm_per_pixel_circle, method = self._process_circle_measurement(
            image, mask_path, circle_fit_params, equivalencies
        )
        
        if method in ['circle', 'fallback']:
            return {'mm_per_pixel': mm_per_pixel_circle, 'method': method}
        
        return {'mm_per_pixel': 0, 'method': 'none'}