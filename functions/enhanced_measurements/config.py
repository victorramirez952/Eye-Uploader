# Configuration file for Enhanced Measurements project
TEMP_IMAGES_FOLDER = "tempImages"  # Folder to save processed images

MEDIAN_RULE_MM = 1.0430
MEDIAN_AP_AXIS = 22.12

# Morphological operations configuration
KERNEL_SIZE = (5, 5)  # Size of the structuring element (kernel)
KERNEL_SHAPE = "rectangular"  # Shape of the kernel: "rectangular", "elliptical", "cross"

# Circle fitting configuration
CIRCLE_FIT_X_ASYMPTOTE_RATIO = 0.25  # X asymptote position as ratio of image width (25%)
CIRCLE_FIT_MIN_X_RATIO = 0.35  # Minimum X position to consider points (35%)
CIRCLE_FIT_MAX_X_RATIO = 0.75  # Maximum X position to consider points (75%, fallback if tallest cluster not found)
CIRCLE_FIT_EDGE_THRESHOLD = 50  # Threshold for edge detection to consider as valid edge point
CIRCLE_FIT_MAX_ITERATIONS = 100  # Maximum iterations for RANSAC circle fitting
CIRCLE_FIT_DISTANCE_THRESHOLD = 5.0  # Maximum distance from circle for inliers (pixels)