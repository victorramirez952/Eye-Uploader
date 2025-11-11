import cv2
import numpy as np


def fit_circle_to_three_points(points):
    """
    Fit a circle to exactly three points using algebraic method.
    
    Args:
        points (numpy.ndarray): Array of 3 points [[x1,y1], [x2,y2], [x3,y3]].
        
    Returns:
        tuple: (center_x, center_y, radius) or None if points are collinear.
    """
    if len(points) != 3:
        return None
    
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    
    A = x2 - x1
    B = y2 - y1
    C = x3 - x1
    D = y3 - y1
    
    E = A * (x1 + x2) + B * (y1 + y2)
    F = C * (x1 + x3) + D * (y1 + y3)
    G = 2 * (A * (y3 - y2) - B * (x3 - x2))
    
    if abs(G) < 1e-6:
        return None
    
    cx = (D * E - B * F) / G
    cy = (A * F - C * E) / G
    radius = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
    
    return (cx, cy, radius)


def fit_circle_to_border_points(image, border_points, x_asymptote_ratio=0.25, min_x_ratio=0.35, max_x_ratio=0.75,
                                max_iterations=100, distance_threshold=5.0, tallest_cluster_x_max=None, tallest_cluster_y_center=None):
    """
    Fit a circle to border points detected by marching squares.
    Uses RANSAC with priority-based weighting. Points at tallest cluster's center Y near 50% have highest priority.
    As points move vertically away from cluster center, priority gradually shifts toward tallest cluster's rightmost X.
    Analyzes points between 35% and tallest cluster's rightmost position (or max_x_ratio if not provided).
    
    Args:
        image (numpy.ndarray): Input image (grayscale or BGR).
        border_points (list): List of [x, y] coordinates from marching squares.
        x_asymptote_ratio (float): X position for asymptote as ratio of width (default 0.25 = 25%).
        min_x_ratio (float): Minimum X ratio to include points (default 0.35 = 35%).
        max_x_ratio (float): Maximum X ratio to include points (default 0.75 = 75%, used if tallest_cluster_x_max not provided).
        max_iterations (int): Max RANSAC iterations.
        distance_threshold (float): Max distance for inliers.
        tallest_cluster_x_max (int): Rightmost X position of tallest cluster (optional).
        tallest_cluster_y_center (int): Center Y position of tallest cluster (optional).
        
    Returns:
        tuple: (result_image, circle_params) where circle_params = (center_x, center_y, radius)
    """
    np.random.seed(64)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = image.copy()
    else:
        gray = image.copy()
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    height, width = gray.shape
    
    # Use tallest cluster's center Y as priority center, or fall back to image center
    if tallest_cluster_y_center is not None:
        center_y = tallest_cluster_y_center
    else:
        center_y = height // 2
    
    # Use 50% as the reference asymptote for priority weighting
    x_asymptote_35 = int(width * 0.35)
    x_asymptote_50 = int(width * 0.50)
    
    # Use tallest cluster's rightmost X as the edge asymptote, or fall back to ratio-based calculation
    if tallest_cluster_x_max is not None:
        x_asymptote_edge = tallest_cluster_x_max
    else:
        x_asymptote_edge = int(width * max_x_ratio)
    
    x_max_limit = x_asymptote_edge
    
    if len(border_points) < 3:
        return result, None
    
    filtered_points = []
    point_priorities = []
    
    for point in border_points:
        x, y = point
        if x_asymptote_35 < x < x_max_limit:
            # Calculate vertical distance from center (0 at center, 1 at top/bottom)
            vertical_distance_from_center = abs(y - center_y) / (height / 2.0)
            
            # Calculate optimal X position: 50% at center, gradually shifting to edge asymptote at edges
            optimal_x = x_asymptote_50 + vertical_distance_from_center * (x_asymptote_edge - x_asymptote_50)
            
            # Calculate horizontal deviation from optimal position
            horizontal_deviation = abs(x - optimal_x)
            max_horizontal_deviation = width * 0.5
            
            # Higher priority for points near center Y
            vertical_weight = 1.0 - (vertical_distance_from_center * 0.5)
            
            # Higher priority for points near optimal X for their Y position
            horizontal_weight = 1.0 - (horizontal_deviation / max_horizontal_deviation)
            
            priority = vertical_weight * horizontal_weight
            
            filtered_points.append([x, y])
            point_priorities.append(priority)
    
    if len(filtered_points) < 3:
        return result, None
    
    filtered_points = np.array(filtered_points, dtype=np.float32)
    point_priorities = np.array(point_priorities, dtype=np.float32)
    
    point_priorities = point_priorities / np.sum(point_priorities)
    
    best_circle = None
    best_inliers = []
    best_weighted_score = 0
    
    for _ in range(max_iterations):
        if len(filtered_points) < 3:
            break
        
        indices = np.random.choice(len(filtered_points), 3, replace=False, p=point_priorities)
        sample_points = filtered_points[indices]
        
        circle_params = fit_circle_to_three_points(sample_points)
        if circle_params is None:
            continue
        
        cx, cy, radius = circle_params
        
        distances = np.sqrt((filtered_points[:, 0] - cx)**2 + (filtered_points[:, 1] - cy)**2)
        inliers = np.abs(distances - radius) < distance_threshold
        
        weighted_score = np.sum(point_priorities[inliers])
        
        if weighted_score > best_weighted_score:
            best_weighted_score = weighted_score
            best_circle = circle_params
            best_inliers = filtered_points[inliers]
    
    if best_circle is None:
        return result, None
    
    cx, cy, radius = best_circle
    
    cv2.circle(result, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
    cv2.circle(result, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    
    # Draw three asymptote lines: 35% (blue), 50% (reference, light blue), and edge (tallest cluster rightmost, orange)
    cv2.line(result, (x_asymptote_35, 0), (x_asymptote_35, height), (255, 0, 0), 2)  # Blue line at 35%
    cv2.line(result, (x_asymptote_50, 0), (x_asymptote_50, height), (0, 165, 255), 2)  # Light blue line at 50%
    cv2.line(result, (x_asymptote_edge, 0), (x_asymptote_edge, height), (255, 165, 0), 2)  # Orange line at edge
    
    for point in best_inliers:
        cv2.circle(result, (int(point[0]), int(point[1])), 2, (255, 255, 0), -1)
    
    diameter = radius * 2
    
    return result, (cx, cy, radius)

