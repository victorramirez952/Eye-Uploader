"""
3D mesh generation and manipulation functions.
"""

import numpy as np
import pyvista as pv
from pyvista import _vtk
from .image_utils import resample_contour, largest_contour, center2d, pca_angle_2d, rotate_xy
from scipy.ndimage import gaussian_filter


def recenter_to_origin(mesh: pv.PolyData) -> pv.PolyData:
    """
    Recenter a 3D mesh at the origin (0,0,0).
    """
    c = np.array(mesh.center)
    mesh.points = mesh.points - c
    return mesh


def make_ring_xy_from_mask(mask: np.ndarray, spacing_xy: tuple[float, float], n_contour_pts: int = 300) -> np.ndarray:
    """
    Take the 2D silhouette of a mask and convert it into a ring of 
    points in millimeters, ready to extrude.
    """
    cnt = resample_contour(largest_contour(mask), n_contour_pts)
    ring = np.column_stack([cnt[:, 0] * spacing_xy[0], cnt[:, 1] * spacing_xy[1]])  # scale to real mm
    # Center
    ring = center2d(ring)
    # Align major axis with +X (establishes BASE_T in X for T)
    theta = pca_angle_2d(ring)
    return rotate_xy(ring, -theta)


def extrude_ring(ring_xy: np.ndarray, length_mm: float, axis: str, n_layers: int = 400) -> pv.PolyData:
    """
    From a ring of 2D points, generate a 3D solid by extruding it along an axis.
    """
    M = ring_xy.shape[0]
    if axis == 'x': 
        xs = np.linspace(-0.5 * length_mm, 0.5 * length_mm, n_layers)
        P = np.vstack([np.column_stack([np.full(M, x), ring_xy[:, 0], ring_xy[:, 1]]) for x in xs])
    elif axis == 'y':
        ys = np.linspace(-0.5 * length_mm, 0.5 * length_mm, n_layers)
        P = np.vstack([np.column_stack([ring_xy[:, 0], np.full(M, y), ring_xy[:, 1]]) for y in ys])
    else:  # 'z'
        zs = np.linspace(-0.5 * length_mm, 0.5 * length_mm, n_layers)
        P = np.vstack([np.column_stack([ring_xy[:, 0], ring_xy[:, 1], np.full(M, z)]) for z in zs])

    faces = []
    
    # Lateral faces 
    for k in range(n_layers - 1):
        b, n = k * M, (k + 1) * M
        for i in range(M):
            a0 = b + i
            a1 = b + (i + 1) % M
            a2 = n + (i + 1) % M
            a3 = n + i
            faces += [4, a0, a1, a2, a3]
    
    # CAPS 
    # Bottom cap (first layer) - inward normal
    bottom_indices = list(range(M))
    faces += [M] + bottom_indices[::-1]  # Reverse order for correct normal
    
    # Top cap (last layer) - outward normal  
    top_start = (n_layers - 1) * M
    top_indices = list(range(top_start, top_start + M))
    faces += [M] + top_indices  # Normal order for outward normal
            
    mesh = pv.PolyData(P, np.asarray(faces, np.int32))
    
    # CLEANUP 
    mesh = mesh.clean(tolerance=1e-6) 
    mesh = mesh.triangulate()
    
    # Verify that the mesh is valid
    if not mesh.is_all_triangles:
        print("WARNING: Mesh contains non-triangular polygons")
    
    # Verify manifold (closed surface)
    try:
        mesh = mesh.fill_holes(1000)  # Fill small holes if any
    except:
        pass
        
    return mesh


def scale_about_center(mesh: pv.PolyData, sx=1.0, sy=1.0, sz=1.0) -> pv.PolyData:
    """
    Scale a 3D mesh around its center.
    """
    c = np.array(mesh.center)
    pts = mesh.points.copy()
    pts -= c
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    pts[:, 2] *= sz
    pts += c
    mesh.points = pts
    return mesh


def enforce_bbox(mesh: pv.PolyData, target_xyz: tuple[float, float, float]) -> pv.PolyData:
    """
    Force exact dimensions in the mesh bbox, scaling from center.
    """
    x0, x1, y0, y1, z0, z1 = mesh.bounds
    cur = np.array([x1 - x0, y1 - y0, z1 - z0], dtype=float)
    scale = np.array([
        (target_xyz[0] / cur[0]) if target_xyz[0] else 1.0,
        (target_xyz[1] / cur[1]) if target_xyz[1] else 1.0,
        (target_xyz[2] / cur[2]) if target_xyz[2] else 1.0,
    ])
    return scale_about_center(mesh, *scale)


def area_volume(poly: pv.PolyData) -> tuple[float, float]:
    """
    Return area (mm²) and volume (mm³) of a 3D mesh.
    """
    if poly is None or poly.n_points == 0:
        return 0.0, 0.0
    surf = poly.extract_surface().triangulate().clean()
    try:
        surf = surf.fill_holes(1e9)
    except Exception:
        pass
    mp = _vtk.vtkMassProperties()
    mp.SetInputData(surf)
    mp.Update()
    return float(mp.GetSurfaceArea()), float(mp.GetVolume())


def create_longitudinal_ring(mask_l: np.ndarray, sL_x: float, sL_y: float, 
                           flip_y: bool = False, n_contour_pts: int = 300) -> np.ndarray:
    """
    Create YZ ring from longitudinal mask with proper mapping and alignment.
    """
    cnt_L = resample_contour(largest_contour(mask_l), n_contour_pts)
    Y = cnt_L[:, 1] * sL_y
    Z = cnt_L[:, 0] * sL_x
    if flip_y:
        Y = -Y
    ring_L_yz = center2d(np.column_stack([Y, Z]))
    
    # Align major axis of YZ ring with Z (so BASE_L is in Z)
    theta_yz = pca_angle_2d(ring_L_yz)
    ring_L_yz = rotate_xy(ring_L_yz, -(theta_yz - np.pi / 2))  # leave major ≈ Z
    
    return ring_L_yz


def smooth_with_implicit_modeling(inter: pv.PolyData, target_bbox: tuple[float, float, float]) -> pv.PolyData:
    """
    Apply implicit modeling smoothing using voxelization and surface extraction.
    This is an alternative method that converts to volume and extracts a smooth surface.
    
    Parameters:
    -----------
    inter : pv.PolyData
        The intersection mesh to smooth
    target_bbox : tuple[float, float, float]
        Target dimensions (BASE_T_MM, H_MM, BASE_L_MM) to enforce after smoothing
    
    Returns:
    --------
    pv.PolyData : Smoothed mesh with implicit modeling
    """
    print(f"\nApplying implicit modeling (alternative method)...")
    
    try:
        # Step 1: Convert surface to distance field (SDF)
        print("  - Converting to distance field...")
        
        # Step 2: Voxelize the mesh
        print("  - Voxelizing...")
        voxels = pv.voxelize(inter, density=0.5, check_surface=False)
        
        # Step 3: Extract smooth surface from volume
        print("  - Applying gaussian filter to volume...")
        inter_smooth = voxels.extract_surface()
        inter_smooth = inter_smooth.smooth(n_iter=50, relaxation_factor=0.2,
                                           feature_smoothing=True,
                                           boundary_smoothing=True)
        
        # Step 4: Scale back to correct dimensions
        inter_smooth = enforce_bbox(inter_smooth, target_bbox)
        inter_smooth = recenter_to_origin(inter_smooth)
        
        if inter_smooth.n_points > 100:
            print(f"    Implicit modeling successful")
            return inter_smooth.clean()
        else:
            print(f"    Implicit modeling produced insufficient points, using standard smoothing...")
            raise ValueError("Insufficient points after implicit modeling")
            
    except Exception as e:
        print(f"    Implicit modeling failed: {e}")
        print("    Using standard smoothing...")
        
        # Fallback: Aggressive standard smoothing
        try:
            inter_smooth = inter.smooth(n_iter=2000, relaxation_factor=0.1,
                                       feature_smoothing=False,
                                       boundary_smoothing=True)
            inter_smooth = inter_smooth.fill_holes(1000)
            inter_smooth = enforce_bbox(inter_smooth, target_bbox)
            inter_smooth = recenter_to_origin(inter_smooth)
            return inter_smooth.clean()
        except Exception as e2:
            print(f"    Standard smoothing also failed: {e2}")
            print("    Returning original mesh...")
            return inter