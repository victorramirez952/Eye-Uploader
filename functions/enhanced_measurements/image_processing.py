import cv2
import numpy as np
import os


def crop_bottom_image(image, width_ratio=0.20, height_ratio=0.07):
    """
    Crop the bottom-left portion of the image.
    
    Args:
        image (numpy.ndarray): Input image.
        width_ratio (float): Width ratio (default 20%).
        height_ratio (float): Height ratio (default 7%).
        
    Returns:
        numpy.ndarray: Cropped bottom-left image.
    """
    height, width = image.shape[:2]
    crop_width = int(width * width_ratio)
    crop_height = int(height * height_ratio)
    bottom_image = image[height - crop_height:height, 0:crop_width]
    return bottom_image


def crop_left_image(image, width_ratio=0.10, height_ratio=0.30):
    """
    Crop the left-bottom portion of the image.
    
    Args:
        image (numpy.ndarray): Input image.
        width_ratio (float): Width ratio (default 10%).
        height_ratio (float): Height ratio (default 30%).
        
    Returns:
        numpy.ndarray: Cropped left-bottom image.
    """
    height, width = image.shape[:2]
    crop_width = int(width * width_ratio)
    crop_height = int(height * height_ratio)
    left_image = image[height - crop_height:height, 0:crop_width]
    return left_image


def apply_binary_mask(image, mask_path):
    """
    Apply binary mask to delete regions from image (white=delete, black=keep).
    Deleted regions are filled with mean color from 10x10 area at middle height and 25% x position.
    
    Args:
        image (numpy.ndarray): Input image to mask.
        mask_path (str): Path to binary mask image.
        
    Returns:
        numpy.ndarray: Masked image with deleted regions filled with mean color.
    """
    if mask_path is None or not os.path.exists(mask_path):
        return image
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return image
    
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Calculate mean color inline
    height, width = image.shape[:2]
    center_y = height // 2
    center_x = int(width * 0.25)
    matrix_size = 10
    half_size = matrix_size // 2
    y_start = max(0, center_y - half_size)
    y_end = min(height, center_y + half_size)
    x_start = max(0, center_x - half_size)
    x_end = min(width, center_x + half_size)
    region = image[y_start:y_end, x_start:x_end]
    
    if len(image.shape) == 3:
        mean_color = tuple(np.mean(region, axis=(0, 1)).astype(int))
    else:
        mean_color = int(np.mean(region))
    
    inverted_mask = cv2.bitwise_not(mask)
    
    masked_image = image.copy()
    
    if len(image.shape) == 3:
        masked_region = cv2.bitwise_and(image, image, mask=inverted_mask)
        fill_region = np.zeros_like(image)
        fill_region[:] = mean_color
        fill_masked = cv2.bitwise_and(fill_region, fill_region, mask=mask)
        masked_image = cv2.add(masked_region, fill_masked)
    else:
        masked_region = cv2.bitwise_and(image, image, mask=inverted_mask)
        fill_region = np.full_like(image, mean_color)
        fill_masked = cv2.bitwise_and(fill_region, fill_region, mask=mask)
        masked_image = cv2.add(masked_region, fill_masked)
    
    return masked_image


def remove_rows_with_high_white_percentage(image, white_threshold=128, percentage_threshold=0.75):
    """
    Delete all rows where more than percentage_threshold of pixels are white.
    Sets those rows to black pixels.
    
    Args:
        image (numpy.ndarray): Input image (grayscale or BGR).
        white_threshold (int): Threshold to consider pixels as white.
        percentage_threshold (float): Percentage threshold (0.0-1.0) for white pixels in a row.
        
    Returns:
        numpy.ndarray: Image with high-white-percentage rows set to black.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = image.copy()
        is_color = True
    else:
        gray = image.copy()
        result = image.copy()
        is_color = False
    
    height, width = gray.shape
    rows_removed = 0
    
    for y in range(height):
        row = gray[y, :]
        white_pixels = np.sum(row > white_threshold)
        white_percentage = white_pixels / width
        
        if white_percentage > percentage_threshold:
            if is_color:
                result[y, :] = [0, 0, 0]
            else:
                result[y, :] = 0
            rows_removed += 1
    
    return result


def pixelate_image(image, block_size=100):
    """
    Pixelate an image by dividing it into non-overlapping nxn blocks
    and replacing each block with its mean color.
    
    Args:
        image (numpy.ndarray): Input image (grayscale or BGR).
        block_size (int): Size of the square blocks (default 10x10).
        
    Returns:
        numpy.ndarray: Pixelated image.
    """
    height, width = image.shape[:2]
    pixelated = image.copy()
    
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            
            block = image[y:y_end, x:x_end]
            
            if len(image.shape) == 3:
                mean_color = np.mean(block, axis=(0, 1)).astype(int)
                pixelated[y:y_end, x:x_end] = mean_color
            else:
                mean_color = int(np.mean(block))
                pixelated[y:y_end, x:x_end] = mean_color
    
    return pixelated


def filter_border_points_by_clustering(border_points, image_shape, min_cluster_size=50, x_tolerance=15):
    """
    Filter border points to keep only large vertical regions using DBSCAN clustering.
    Removes isolated points and small scattered regions.
    
    Args:
        border_points (list): List of [x, y] coordinates.
        image_shape (tuple): (height, width) of the image.
        min_cluster_size (int): Minimum number of points in a cluster to keep it.
        x_tolerance (int): Maximum horizontal distance for points to be in same cluster.
        
    Returns:
        list: Filtered border points belonging to large vertical regions.
    """
    if len(border_points) < min_cluster_size:
        return border_points
    
    from sklearn.cluster import DBSCAN
    
    points_array = np.array(border_points, dtype=np.float32)
    
    # Use DBSCAN with custom distance metric favoring vertical stacking
    # Scale X dimension to make vertical stacking more important
    scaled_points = points_array.copy()
    scaled_points[:, 0] = scaled_points[:, 0] * 3  # Make X distance more significant
    
    clustering = DBSCAN(eps=x_tolerance * 3, min_samples=min_cluster_size)
    labels = clustering.fit_predict(scaled_points)
    
    # Count cluster sizes
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # Remove noise label
    
    cluster_info = {}
    for label in unique_labels:
        cluster_points = points_array[labels == label]
        cluster_info[label] = {
            'size': len(cluster_points),
            'x_mean': np.mean(cluster_points[:, 0]),
            'y_span': np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
        }
    
    # Filter: keep only clusters that are large and vertically stacked
    filtered_points = []
    cluster_data = []
    
    for label, info in cluster_info.items():
        if info['size'] >= min_cluster_size and info['y_span'] > image_shape[0] * 0.1:
            cluster_mask = labels == label
            cluster_points = points_array[cluster_mask]
            filtered_points.extend(cluster_points.tolist())
            
            # Calculate bounding box for this cluster
            x_min = int(np.min(cluster_points[:, 0]))
            x_max = int(np.max(cluster_points[:, 0]))
            y_min = int(np.min(cluster_points[:, 1]))
            y_max = int(np.max(cluster_points[:, 1]))
            y_center = int((y_min + y_max) / 2)
            
            cluster_data.append({
                'label': label,
                'points': cluster_points,
                'bbox': (x_min, y_min, x_max, y_max),
                'y_span': info['y_span'],
                'y_center': y_center,
                'size': info['size']
            })
    
    return filtered_points, cluster_data


def apply_marching_squares(image, block_size=10, threshold=200, intensity_threshold=25, 
                          line_color=(0, 255, 0), line_thickness=2, 
                          filter_clusters=True, min_cluster_size=50, temp_folder="tempImages"):
    """
    Apply 2D marching squares algorithm to delimit vertical borders between pixel blocks.
    Only highlights borders where pixel intensity is above intensity_threshold (near white).
    Optionally filters isolated points to keep only large vertical regions.
    
    Args:
        image (numpy.ndarray): Input image (grayscale or BGR).
        block_size (int): Size of the blocks used for pixelation (default 10).
        threshold (int): Intensity difference threshold to detect edges (default 128).
        intensity_threshold (int): Minimum intensity to highlight borders (default 200, near white).
        line_color (tuple): Color for border lines in BGR format (default green).
        line_thickness (int): Thickness of border lines (default 2).
        filter_clusters (bool): Whether to filter border points by clustering (default True).
        min_cluster_size (int): Minimum cluster size for filtering (default 50).
        temp_folder (str): Folder to save filtered visualization (default "tempImages").
        
    Returns:
        tuple: (result_image, border_points, tallest_cluster_x_max, tallest_cluster_y_center) where border_points is a list of [x, y] coordinates,
               tallest_cluster_x_max is the rightmost X position of the tallest cluster (or None),
               and tallest_cluster_y_center is the center Y position of the tallest cluster (or None).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = image.copy()
    else:
        gray = image
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    height, width = gray.shape
    border_points = []
    
    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            
            current_block = gray[y:y_end, x:x_end]
            current_mean = int(np.mean(current_block))
            
            if x + block_size < width:
                right_block = gray[y:y_end, x + block_size:min(x + 2*block_size, width)]
                right_mean = int(np.mean(right_block))
                
                if abs(current_mean - right_mean) > threshold:
                    if current_mean >= intensity_threshold or right_mean >= intensity_threshold:
                        cv2.line(result, (x + block_size, y), (x + block_size, y_end), line_color, line_thickness)
                        for border_y in range(y, y_end):
                            border_points.append([x + block_size, border_y])
    
    # Filter border points by clustering to keep only large vertical regions
    if filter_clusters and len(border_points) > 0:
        filtered_border_points, cluster_data = filter_border_points_by_clustering(
            border_points, (height, width), min_cluster_size=min_cluster_size
        )
        
        # Find cluster with maximum vertical height within 35%-75% X range
        x_min_range = int(width * 0.35)
        x_max_range = int(width * 0.75)
        
        max_y_span_cluster = None
        if cluster_data:
            # Filter clusters within X range
            valid_clusters = [c for c in cluster_data 
                            if x_min_range <= c['bbox'][0] <= x_max_range and 
                               x_min_range <= c['bbox'][2] <= x_max_range]
            if valid_clusters:
                max_y_span_cluster = max(valid_clusters, key=lambda c: c['y_span'])
        
        # Create visualization of filtered borders
        filtered_result = image.copy()
        if len(filtered_result.shape) == 2:
            filtered_result = cv2.cvtColor(filtered_result, cv2.COLOR_GRAY2BGR)
        
        # Draw all filtered points
        for point in filtered_border_points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(filtered_result, (x, y), 1, line_color, -1)
        
        # Draw bounding boxes for all clusters
        for cluster in cluster_data:
            x_min, y_min, x_max, y_max = cluster['bbox']
            
            # Use red color for cluster with maximum vertical height
            if max_y_span_cluster and cluster['label'] == max_y_span_cluster['label']:
                box_color = (0, 0, 255)  # Red for tallest cluster
                thickness = 3
                
                # Paint the region points in red as well
                for point in cluster['points']:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(filtered_result, (x, y), 1, box_color, -1)
            else:
                box_color = (255, 255, 0)  # Cyan for other clusters
                thickness = 2
            
            cv2.rectangle(filtered_result, (x_min, y_min), (x_max, y_max), box_color, thickness)
        
        # Save filtered visualization
        if temp_folder and os.path.exists(temp_folder):
            output_path = os.path.join(temp_folder, "marching_squares_filtered.jpg")
            cv2.imwrite(output_path, filtered_result)
        
        # Return the rightmost X and center Y of tallest cluster, but return ALL border points (unfiltered)
        tallest_x_max = max_y_span_cluster['bbox'][2] if max_y_span_cluster else None
        tallest_y_center = max_y_span_cluster['y_center'] if max_y_span_cluster else None
        return result, border_points, tallest_x_max, tallest_y_center
    
    return result, border_points, None, None
