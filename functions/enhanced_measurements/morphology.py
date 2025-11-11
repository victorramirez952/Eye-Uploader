import cv2
import numpy as np


def create_kernel(kernel_size=(5, 5), kernel_shape="rectangular"):
    """
    Create a kernel for morphological operations.
    
    Args:
        kernel_size (tuple): Size of the structuring element (kernel).
        kernel_shape (str): Shape of the kernel: "rectangular", "elliptical", "cross".
        
    Returns:
        numpy.ndarray: Structuring element for morphological operations.
    """
    if kernel_shape == "rectangular":
        return np.ones(kernel_size, np.uint8)
    elif kernel_shape == "elliptical":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif kernel_shape == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:
        return np.ones(kernel_size, np.uint8)


def apply_opening(image, kernel_size=(5, 5), kernel_shape="rectangular"):
    """
    Apply morphological opening operation to remove noise.
    
    Args:
        image (numpy.ndarray): Grayscale image.
        kernel_size (tuple): Size of the structuring element.
        kernel_shape (str): Shape of the kernel.
        
    Returns:
        numpy.ndarray: Result after opening operation.
    """
    kernel = create_kernel(kernel_size, kernel_shape)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening


def apply_closing(image, kernel_size=(5, 5), kernel_shape="rectangular"):
    """
    Apply morphological closing operation to fill holes.
    
    Args:
        image (numpy.ndarray): Grayscale image.
        kernel_size (tuple): Size of the structuring element.
        kernel_shape (str): Shape of the kernel.
        
    Returns:
        numpy.ndarray: Result after closing operation.
    """
    kernel = create_kernel(kernel_size, kernel_shape)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing
