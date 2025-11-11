import cv2


def apply_mean_shift_to_crop(image, spatial_radius=21, color_radius=51):
    """
    Apply mean shift segmentation to a cropped image.
    
    Args:
        image (numpy.ndarray): Input image (grayscale or BGR).
        spatial_radius (int): Spatial window radius.
        color_radius (int): Color window radius.
        
    Returns:
        numpy.ndarray: Segmented image.
    """
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()
    
    segmented = cv2.pyrMeanShiftFiltering(image_bgr, spatial_radius, color_radius)
    return segmented
