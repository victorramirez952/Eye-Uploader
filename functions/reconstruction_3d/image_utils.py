"""
Utility functions for image processing and geometric operations.
"""

import cv2
import numpy as np
from pathlib import Path


def read_mask(path: str) -> np.ndarray:
    """
    Read PNG mask and return uint8 {0,255}.
    """
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 0).astype(np.uint8) * 255


def largest_contour(mask: np.ndarray) -> np.ndarray:
    """
    Extract the largest contour from a binary mask {0,255}.
    Returns contour as (N, 2) float32 array.
    """
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        raise RuntimeError("No contour found")
    # (N, 2) in float32
    return max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float32)


def resample_contour(cnt: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a contour (N,2) to n_pts uniform by arc length.
    """
    if not np.allclose(cnt[0], cnt[-1]):
        cnt = np.vstack([cnt, cnt[0]])
    d = np.diff(cnt, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.sqrt((d**2).sum(1)))])  # distances between points
    L = s[-1]  # total length
    t = np.linspace(0, L, n, endpoint=False) 
    out, j = [], 0
    for u in t:
        while j < len(s) - 2 and s[j + 1] < u:
            j += 1
        a = (u - s[j]) / max(s[j + 1] - s[j], 1e-6)
        out.append(cnt[j] + a * (cnt[j + 1] - cnt[j]))
    return np.asarray(out, np.float32)


def center2d(pts: np.ndarray) -> np.ndarray:
    """
    Center a set of 2D points at the origin.
    """
    return pts - pts.mean(axis=0, keepdims=True) 


def bbox_wh_from_mask(mask: np.ndarray) -> tuple[float, float]:
    """
    Extract width and height from the bbox of the mask.
    The bbox is the minimum rectangle that contains it.
    """
    cnt = largest_contour(mask)
    w = float(cnt[:, 0].max() - cnt[:, 0].min())
    h = float(cnt[:, 1].max() - cnt[:, 1].min())
    if w <= 0 or h <= 0:
        raise RuntimeError("Invalid BBox")
    return w, h


def pca_angle_2d(points_xy: np.ndarray) -> float:
    """
    PCA in 2D to find the angle of the major axis (from +X towards +Y).
    PCA means Principal Component Analysis.
    """
    P = points_xy - points_xy.mean(axis=0, keepdims=True) 
    C = np.cov(P, rowvar=False)
    w, V = np.linalg.eigh(C)
    v = V[:, np.argmax(w)]
    ang = np.arctan2(v[1], v[0])  # rad from +X towards +Y
    return ang


def rotate_xy(points_xy: np.ndarray, theta_rad: float) -> np.ndarray:
    """
    Rotate a set of 2D points around the origin.
    """
    c, s = np.cos(theta_rad), np.sin(theta_rad) 
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points_xy @ R.T


def get_contour_real_dimensions(mask: np.ndarray) -> tuple[float, float]:
    """
    Extract the real dimensions of the contour after aligning it with PCA
    to find its principal axes.
    """
    # Extract and resample contour
    cnt = largest_contour(mask)
    cnt_resampled = resample_contour(cnt, 300)  # Use default N_CONTOUR_PTS
    
    # Center contour at origin
    cnt_centered = center2d(cnt_resampled)
    
    # Apply PCA to find principal axes
    theta = pca_angle_2d(cnt_centered)
    cnt_aligned = rotate_xy(cnt_centered, -theta)
    
    # Measure dimensions in principal axes
    x_coords = cnt_aligned[:, 0]
    y_coords = cnt_aligned[:, 1]
    
    width = x_coords.max() - x_coords.min()   # Dimension in X (major axis after PCA)
    height = y_coords.max() - y_coords.min()  # Dimension in Y (minor axis after PCA)
    
    # Return in order: (major, minor)
    return (width, height) if width >= height else (height, width)


def analyze_orthogonality(mask_t: np.ndarray, mask_l: np.ndarray) -> bool:
    """
    Verify that the two masks are not rotated almost the same (they should be orthogonal).
    """
    # Extract contours from the masks
    contour_t = largest_contour(mask_t)
    contour_l = largest_contour(mask_l)
    
    # Compare PCA orientations of both masks
    angle_t = pca_angle_2d(contour_t)
    angle_l = pca_angle_2d(contour_l)
    
    angle_diff = abs(np.degrees(angle_t - angle_l))
    if angle_diff > 90:
        angle_diff = 180 - angle_diff
        
    # If difference < 15°, there's a problem
    return angle_diff > 15


def select_file(title: str, filetypes: list = None) -> str:
    """
    Select a file from the explorer using tkinter dialog.
    
    Parameters:
    -----------
    title : str
        Dialog window title
    filetypes : list, optional
        List of tuples (description, pattern) for file filtering
    
    Returns:
    --------
    str : Selected file path
    """
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        if filetypes is None:
            filetypes = [("PNG Images", "*.png"), ("All files", "*.*")]
        
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes
        )
        root.destroy()
        
        if not file_path:
            raise ValueError("No file selected")
        
        return file_path
    except ImportError:
        print("Error: tkinter is not available. Please enter the path manually.")
        return input(f"{title}: ")


def select_masks_only() -> tuple[str, str]:
    """
    Allow the user to select only the masks using file dialogs.
    
    Returns:
    --------
    tuple[str, str] : (transverse_mask_path, longitudinal_mask_path)
    """
    print("="*60)
    print("MASK SELECTION")
    print("="*60)
    
    print("\n1. Select the TRANSVERSE mask")
    mask_trans = select_file("Select Transverse Mask", [("PNG Masks", "*.png"), ("All", "*.*")])
    print(f"   Selected: {Path(mask_trans).name}")
    
    print("\n2. Select the LONGITUDINAL mask")
    mask_long = select_file("Select Longitudinal Mask", [("PNG Masks", "*.png"), ("All", "*.*")])
    print(f"   Selected: {Path(mask_long).name}")
    
    print("\n" + "="*60)
    print("Masks selected successfully")
    print("="*60 + "\n")
    
    return mask_trans, mask_long


def get_manual_measurements() -> tuple[float, float, float]:
    """
    Request the user to enter measurements manually.
    
    Returns:
    --------
    tuple[float, float, float] : (BASE_T, BASE_L, H) in millimeters
    """
    print("\n" + "="*60)
    print("CLINICAL MEASUREMENTS INPUT")
    print("="*60)
    print("\nEnter the measurements obtained from ultrasound:")
    print("(All measurements must be in millimeters)")
    print()
    
    while True:
        try:
            base_t = float(input("BASE_T (transverse basal diameter) [mm]: "))
            base_l = float(input("BASE_L (longitudinal basal diameter) [mm]: "))
            h = float(input("H (height/thickness) [mm]: "))
            
            if base_t <= 0 or base_l <= 0 or h <= 0:
                print("Error: All measurements must be positive.")
                continue
            
            print(f"\n{'='*60}")
            print("ENTERED MEASUREMENTS:")
            print(f"{'='*60}")
            print(f"BASE_T (Transverse): {base_t:.2f} mm")
            print(f"BASE_L (Longitudinal): {base_l:.2f} mm")
            print(f"H (Height): {h:.2f} mm")
            print(f"{'='*60}")
            
            confirm = input("\nAre the measurements correct? (y/n): ").strip().lower()
            
            if confirm == 'y':
                print("Measurements confirmed\n")
                return base_t, base_l, h
            else:
                print("\n--- Please re-enter the measurements ---\n")
        
        except ValueError:
            print("Error: Enter valid numeric values.\n")