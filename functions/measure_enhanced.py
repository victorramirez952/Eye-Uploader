import numpy as np
import cv2
import os

def measure(img_array, mm_per_pixel):
    """
    Measure thickness in millimeters.
    
    Args:
        img_array: Binary mask image as numpy array
        mm_per_pixel: Conversion factor from pixels to millimeters
        
    Returns:
        float: Measured thickness in millimeters
    """
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()
    
    # Get image dimensions
    img_height, img_width = gray.shape
    
    # Create axis-aligned bounding box of the binary mask
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    y_coords_mask, x_coords_mask = np.where(binary > 0)
    
    if len(y_coords_mask) == 0:
        print("Error: No white pixels found in mask")
        return 0.0
    
    bbox_right = int(np.max(x_coords_mask))
    x_asymptote_right = bbox_right
    
    # Draw a curve from top (rightmost x, 0%) through (75%, 50%) to bottom (rightmost x, 100%)
    x_asymptote_percent = x_asymptote_right / img_width
    nearest_pixels = []
    
    # Calculate centroid y coordinate of the binary mask
    centroid_y_mask = int(np.mean(y_coords_mask))
    
    # Base distance threshold
    base_threshold_percent = 10
    base_threshold_pixels = (base_threshold_percent / 100.0) * img_width
    
    for y in range(img_height):
        # Parametric curve using quadratic Bezier curve
        t = y / img_height
        x_percent = ((1 - t) ** 2) * x_asymptote_percent + 2 * (1 - t) * t * 0.75 + (t ** 2) * x_asymptote_percent
        x_curve = x_percent * img_width
        
        # Find the nearest white pixel to the curve at this row
        row_pixels = np.where(binary[y, :] > 0)[0]
        
        if len(row_pixels) > 0:
            distance_from_centroid = abs(y - centroid_y_mask)
            threshold_multiplier = 1.0 + (distance_from_centroid / img_height)
            threshold_pixels = base_threshold_pixels * threshold_multiplier
            
            distances = np.abs(row_pixels - x_curve)
            nearest_idx = np.argmin(distances)
            nearest_distance = distances[nearest_idx]
            nearest_x = row_pixels[nearest_idx]
            
            if nearest_distance <= threshold_pixels:
                nearest_pixels.append([nearest_x, y])
    
    perp_length_pixels = 0
    
    # Draw a line connecting all the nearest pixels
    if len(nearest_pixels) > 1:
        nearest_pixels = np.array(nearest_pixels, dtype=np.int32)
        
        # Find the points with greatest and lowest y values
        max_y_idx = np.argmax(nearest_pixels[:, 1])
        min_y_idx = np.argmin(nearest_pixels[:, 1])
        
        point_max_y = nearest_pixels[max_y_idx]
        point_min_y = nearest_pixels[min_y_idx]
        
        # Calculate the direction vector of the line
        line_vector = point_max_y.astype(np.float64) - point_min_y.astype(np.float64)
        line_length = np.linalg.norm(line_vector)
        
        if line_length > 0:
            line_direction = line_vector / line_length
            perp_direction = np.array([-line_direction[1], line_direction[0]])
            
            # Calculate centroid of the binary mask
            centroid_mask = np.array([np.mean(x_coords_mask), np.mean(y_coords_mask)])
            
            # Trace along perpendicular direction to find endpoints within mask
            def trace_to_mask_edge(start_point, direction, binary_mask, max_distance):
                """Trace along direction until reaching edge of binary mask."""
                step_size = 1.0
                current_distance = 0
                last_valid_point = start_point.copy()
                
                while current_distance < max_distance:
                    current_distance += step_size
                    test_point = start_point + direction * current_distance
                    
                    px, py = int(round(test_point[0])), int(round(test_point[1]))
                    
                    if px < 0 or py < 0 or px >= binary_mask.shape[1] or py >= binary_mask.shape[0]:
                        break
                    
                    if binary_mask[py, px] == 0:
                        break
                    
                    last_valid_point = test_point
                
                return last_valid_point
            
            max_trace_distance = max(img_width, img_height)
            
            perp_start = trace_to_mask_edge(centroid_mask, -perp_direction, binary, max_trace_distance)
            perp_end = trace_to_mask_edge(centroid_mask, perp_direction, binary, max_trace_distance)
            
            # Calculate perpendicular line length (height measurement)
            perp_length_pixels = np.linalg.norm(perp_end - perp_start)
    
    # Convert to millimeters
    perp_length_mm = round(perp_length_pixels * mm_per_pixel, 2)
    return perp_length_mm

def visualize_measure(img_array, mm_per_pixel, thickness_mm):
    """
    Visualize the measurement process with annotations
    
    Args:
        img_array: Binary mask image as numpy array
        mm_per_pixel: Conversion factor from pixels to millimeters
        thickness_mm: Pre-calculated thickness in millimeters to display
    """
    OUTPUT_DIR = "tempImages"
    # Create output directory if it doesn't exist in the current directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    img_color = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    img_height, img_width = img_array.shape
    
    y_coords_mask, x_coords_mask = np.where(img_array > 0)
    
    if len(y_coords_mask) == 0:
        print("No mask pixels found")
        return
    
    bbox_right = int(np.max(x_coords_mask))
    x_asymptote_right = bbox_right
    x_asymptote_percent = x_asymptote_right / img_width
    nearest_pixels = []
    
    centroid_y_mask = int(np.mean(y_coords_mask))
    
    base_threshold_percent = 10
    base_threshold_pixels = (base_threshold_percent / 100.0) * img_width
    
    for y in range(img_height):
        t = y / img_height
        x_percent = ((1 - t) ** 2) * x_asymptote_percent + 2 * (1 - t) * t * 0.75 + (t ** 2) * x_asymptote_percent
        x_curve = x_percent * img_width
        
        row_pixels = np.where(img_array[y, :] > 0)[0]
        
        if len(row_pixels) > 0:
            distance_from_centroid = abs(y - centroid_y_mask)
            threshold_multiplier = 1.0 + (distance_from_centroid / img_height)
            threshold_pixels = base_threshold_pixels * threshold_multiplier
            
            distances = np.abs(row_pixels - x_curve)
            nearest_idx = np.argmin(distances)
            nearest_distance = distances[nearest_idx]
            nearest_x = row_pixels[nearest_idx]
            
            if nearest_distance <= threshold_pixels:
                nearest_pixels.append([nearest_x, y])
                cv2.circle(img_color, (nearest_x, y), 2, (0, 0, 255), -1)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_original_with_edge_line.png"), img_color)
    
    if len(nearest_pixels) > 1:
        nearest_pixels = np.array(nearest_pixels, dtype=np.int32)
        cv2.polylines(img_color, [nearest_pixels], isClosed=False, color=(0, 0, 255), thickness=2)
        
        max_y_idx = np.argmax(nearest_pixels[:, 1])
        min_y_idx = np.argmin(nearest_pixels[:, 1])
        
        point_max_y = nearest_pixels[max_y_idx]
        point_min_y = nearest_pixels[min_y_idx]
        
        line_vector = point_max_y.astype(np.float64) - point_min_y.astype(np.float64)
        line_length = np.linalg.norm(line_vector)
        
        if line_length > 0:
            line_direction = line_vector / line_length
            perp_direction = np.array([-line_direction[1], line_direction[0]])
            
            centroid_mask = np.array([np.mean(x_coords_mask), np.mean(y_coords_mask)])
            
            def trace_to_mask_edge(start_point, direction, binary_mask, max_distance):
                step_size = 1.0
                current_distance = 0
                last_valid_point = start_point.copy()
                
                while current_distance < max_distance:
                    current_distance += step_size
                    test_point = start_point + direction * current_distance
                    
                    px, py = int(round(test_point[0])), int(round(test_point[1]))
                    
                    if px < 0 or py < 0 or px >= binary_mask.shape[1] or py >= binary_mask.shape[0]:
                        break
                    
                    if binary_mask[py, px] == 0:
                        break
                    
                    last_valid_point = test_point
                
                return last_valid_point
            
            max_trace_distance = max(img_width, img_height)
            
            perp_start = trace_to_mask_edge(centroid_mask, -perp_direction, img_array, max_trace_distance)
            perp_end = trace_to_mask_edge(centroid_mask, perp_direction, img_array, max_trace_distance)
            
            distance_pixels = np.linalg.norm(perp_end - perp_start)
            
            perp_start_int = tuple(perp_start.astype(int))
            perp_end_int = tuple(perp_end.astype(int))
            
            cv2.line(img_color, perp_start_int, perp_end_int, (255, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.circle(img_color, perp_start_int, 5, (255, 255, 0), -1)
            cv2.circle(img_color, perp_end_int, 5, (255, 255, 0), -1)
            
            text = f"Thickness: {thickness_mm} mm ({distance_pixels:.1f} px)"
            cv2.putText(img_color, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_final_with_measurement.png"), img_color)