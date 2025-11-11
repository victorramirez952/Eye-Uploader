import cv2
import numpy as np


def create_signed_distance_field(image, white_threshold=128):
    """
    Create a signed distance field from an image following these rules:
    - Black pixels will be negative values (displayed as blue)
    - White pixels (greater than threshold):
        - If in lower half and near bottom: highest values (red)
        - If in upper half or near top: lower values (blue)
        - Gradient based on distance from bottom
        - Vertical lines should be red surrounded by blue
    
    Args:
        image (numpy.ndarray): Input image (grayscale or BGR).
        white_threshold (int): Threshold to consider pixels as white.
        
    Returns:
        numpy.ndarray: Signed distance field as color image (BGR).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    height, width = gray.shape
    
    white_mask = (gray > white_threshold).astype(np.uint8)
    black_mask = (gray <= white_threshold).astype(np.uint8)
    
    dist_from_white = cv2.distanceTransform(black_mask, cv2.DIST_L2, 5)
    dist_from_black = cv2.distanceTransform(white_mask, cv2.DIST_L2, 5)
    
    sdf = np.full((height, width), -10.0, dtype=np.float32)
    
    sdf[black_mask == 1] = -10.0 - dist_from_white[black_mask == 1]
    
    for y in range(height):
        for x in range(width):
            if white_mask[y, x] == 1:
                dist_from_bottom = (height - y) / height
                edge_distance = dist_from_black[y, x]
                
                if y < (height / 2):
                    sdf[y, x] = -5.0 - edge_distance
                elif y < (height * 0.80):
                    sdf[y, x] = -3.0
                else:
                    if edge_distance >= 1.0:
                        core_strength = (edge_distance ** 2) * 5.0
                        
                        if y > (height * 0.95):
                            bottom_boost = 100.0 * (y - height * 0.95) / (height * 0.05)
                        else:
                            bottom_boost = 0.0
                        
                        position_bonus = (dist_from_bottom ** 2) * 50.0
                        
                        sdf[y, x] = core_strength + position_bonus + bottom_boost
                    else:
                        sdf[y, x] = -2.0
    
    sdf_min = np.min(sdf)
    sdf_max = np.max(sdf)
    
    sdf_normalized = np.zeros_like(sdf, dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            val = sdf[y, x]
            if val < 0:
                normalized = int(np.clip(50 + (val / sdf_min) * 50, 0, 100))
            else:
                if sdf_max > 0:
                    val_normalized = np.sqrt(val / sdf_max)
                    normalized = int(np.clip(180 + val_normalized * 75, 180, 255))
                else:
                    normalized = 120
            sdf_normalized[y, x] = normalized
    
    sdf_colored = cv2.applyColorMap(sdf_normalized, cv2.COLORMAP_JET)
    
    return sdf_colored


def create_binary_mask_from_sdf(sdf_image, high_value_threshold=100):
    """
    Create a binary mask from signed distance field where:
    - White pixels represent areas with high positive values (vertical white lines)
    - Black pixels represent all other areas
    
    Args:
        sdf_image (numpy.ndarray): SDF image (colored or grayscale).
        high_value_threshold (int): Threshold for high values (0-255).
        
    Returns:
        numpy.ndarray: Binary mask (white for high values, black otherwise).
    """
    if len(sdf_image.shape) == 3:
        gray = sdf_image[:, :, 2]
    else:
        gray = sdf_image.copy()
    
    _, binary_mask = cv2.threshold(gray, high_value_threshold, 255, cv2.THRESH_BINARY)
    
    return binary_mask


def find_bottom_left_white_pixel(binary_mask):
    """
    Locate the white pixel nearest to the bottom-left corner in the binary mask.
    
    Args:
        binary_mask (numpy.ndarray): Binary mask image (white=255, black=0).
        
    Returns:
        tuple: (x, y) coordinates of the nearest white pixel, or None if not found.
    """
    height, width = binary_mask.shape
    
    min_distance = float('inf')
    closest_pixel = None
    
    white_pixels = np.argwhere(binary_mask == 255)
    
    if len(white_pixels) == 0:
        return None
    
    corner_x, corner_y = 0, height - 1
    
    for pixel in white_pixels:
        y, x = pixel
        distance = np.sqrt((x - corner_x)**2 + (y - corner_y)**2)
        if distance < min_distance:
            min_distance = distance
            closest_pixel = (x, y)
    
    return closest_pixel


def find_next_white_pixel_on_right(binary_mask, start_x, start_y, min_gap=5):
    """
    Find the nearest non-consecutive white pixel on the right from a starting pixel.
    Skips consecutive white pixels and finds the start of the next white region.
    Ensures at least min_gap black pixels between the two white pixel regions.
    
    Args:
        binary_mask (numpy.ndarray): Binary mask image (white=255, black=0).
        start_x (int): Starting X coordinate.
        start_y (int): Starting Y coordinate.
        min_gap (int): Minimum gap in black pixels to consider non-consecutive.
        
    Returns:
        tuple: (x, y) coordinates of the next white pixel, or None if not found.
    """
    height, width = binary_mask.shape
    
    black_count = 0
    
    for x in range(start_x + 1, width):
        pixel_value = binary_mask[start_y, x]
        
        if pixel_value == 0:
            black_count += 1
        elif pixel_value == 255 and black_count >= min_gap:
            return (x, start_y)
        elif pixel_value == 255:
            black_count = 0
    
    search_radius = 5
    for dy in range(1, search_radius + 1):
        for direction in [1, -1]:
            search_y = start_y + (dy * direction)
            if 0 <= search_y < height:
                black_count = 0
                for x in range(start_x + 1, width):
                    pixel_value = binary_mask[search_y, x]
                    
                    if pixel_value == 0:
                        black_count += 1
                    elif pixel_value == 255 and black_count >= min_gap:
                        return (x, search_y)
                    elif pixel_value == 255:
                        black_count = 0
    
    return None


def measure_vertical_line_distances(binary_mask, num_measurements=4):
    """
    Measure horizontal distances between vertical lines and return median distance.
    
    Args:
        binary_mask (numpy.ndarray): Binary mask with white vertical lines.
        num_measurements (int): Number of measurements to take (default 4).
        
    Returns:
        dict: Dictionary with 'measurements' list, 'median_distance' and 'average_distance' values.
    """
    measurements = []
    
    start_pixel = find_bottom_left_white_pixel(binary_mask)
    
    if start_pixel is None:
        return {'measurements': [], 'median_distance': 0, 'average_distance': 0}
    
    current_pixel = start_pixel
    
    for i in range(num_measurements):
        next_pixel = find_next_white_pixel_on_right(binary_mask, current_pixel[0], current_pixel[1])
        
        if next_pixel is None:
            break
        
        horizontal_distance = next_pixel[0] - current_pixel[0]
        measurements.append(horizontal_distance)
        
        current_pixel = next_pixel
    
    if len(measurements) > 0:
        median_distance = np.median(measurements)
        average_distance = np.mean(measurements)
    else:
        median_distance = 0
        average_distance = 0
    
    return {
        'measurements': measurements,
        'median_distance': median_distance,
        'average_distance': average_distance
    }
