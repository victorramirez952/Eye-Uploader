"""
Analysis and measurement functions.
"""

import numpy as np


def relative_volume_error(vtk_volume: float, length: float, width: float, height: float) -> float:
    """
    Calculate the relative error of the volume.
    """
    theoretical_volume = (np.pi / 6) * length * width * height
    print(f"Theoretical volume: {theoretical_volume:.2f} mm³\n")
    if theoretical_volume == 0:
        raise ValueError("Theoretical volume is zero, cannot calculate relative error.")
    return abs(vtk_volume - theoretical_volume) / theoretical_volume * 100


def debug_calibration_info(sT_x: float, sT_y: float, sL_x: float, sL_y: float):
    """
    Print debugging information about calibration.
    """
    print("\n=== DEBUGGING CALIBRATION ===")
    print(f"T scales: ({sT_x:.4f}, {sT_y:.4f}) mm/px")
    print(f"L scales: ({sL_x:.4f}, {sL_y:.4f}) mm/px")


def debug_mesh_intersection(mesh_T, mesh_L):
    """
    Print debugging information about mesh intersection.
    """
    print(f"\n=== DEBUGGING INTERSECTION ===")
    print(f"Meshes overlap in X? {mesh_T.bounds[0] < mesh_L.bounds[1] and mesh_L.bounds[0] < mesh_T.bounds[1]}")
    print(f"Meshes overlap in Y? {mesh_T.bounds[2] < mesh_L.bounds[3] and mesh_L.bounds[2] < mesh_T.bounds[3]}")
    print(f"Meshes overlap in Z? {mesh_T.bounds[4] < mesh_L.bounds[5] and mesh_L.bounds[4] < mesh_T.bounds[5]}")


def debug_longitudinal_pca(ring_L_yz: np.ndarray, base_l_mm: float):
    """
    Print debugging information about longitudinal PCA alignment.
    """
    print(f"\n=== DEBUGGING LONGITUDINAL PCA ===")
    
    # Before alignment
    y_range_before = ring_L_yz[:, 0].max() - ring_L_yz[:, 0].min()
    z_range_before = ring_L_yz[:, 1].max() - ring_L_yz[:, 1].min()
    print(f"Before alignment - Y: {y_range_before:.2f}, Z: {z_range_before:.2f}")
    print(f"Original major axis: {'Y' if y_range_before > z_range_before else 'Z'}")
    
    # After alignment (this would be called after rotation)
    print(f"BASE_L should be in Z? {z_range_before:.2f} ≈ {base_l_mm}")


def print_final_results(inter, base_t_mm: float, h_mm: float, base_l_mm: float, area_mm2: float, vol_mm3: float):
    """
    Print final measurement results.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = inter.bounds
    dimX, dimY, dimZ = xmax - xmin, ymax - ymin, zmax - zmin
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"[Final] Dim (X,Y,Z)= {dimX:.2f} × {dimY:.2f} × {dimZ:.2f} mm")
    print(f"[Expected]         = {base_t_mm:.2f} × {h_mm:.2f} × {base_l_mm:.2f} mm")
    print(f"[Errors]          = {dimX-base_t_mm:+.2f} × {dimY-h_mm:+.2f} × {dimZ-base_l_mm:+.2f} mm")
    
    print(f"Area: {area_mm2:.2f} mm²   Volume: {vol_mm3:.2f} mm³")
    
    # Volume error
    try:
        err_vol = relative_volume_error(vol_mm3, base_t_mm, base_l_mm, h_mm)
        print(f"Relative volume error: {err_vol:.2f} %")
    except ValueError as ve:
        print(f"Error calculating relative volume: {ve}")


def print_mesh_info(mesh, name: str):
    """
    Print mesh information for debugging.
    """
    print(f"\n=== MESH {name.upper()} ===")
    print(f"Bounds {name}: {mesh.bounds}")
    dim = np.array(mesh.bounds[1::2]) - np.array(mesh.bounds[::2])
    print(f"Dimensions {name}: {dim[0]:.2f} x {dim[1]:.2f} x {dim[2]:.2f}")