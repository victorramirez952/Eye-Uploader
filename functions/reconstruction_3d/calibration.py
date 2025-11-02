"""
Calibration functions for converting between pixel and millimeter units.
"""

import numpy as np
from .image_utils import get_contour_real_dimensions, largest_contour, center2d, resample_contour, pca_angle_2d


def calibrate_from_3measures_robust(m_trans: np.ndarray, m_long: np.ndarray,
                                     base_t_mm: float, base_l_mm: float, h_mm: float):
    """
    Calibration mm/px from the 3 clinical measurements (base_t, base_l, h).
    Returns scaling factors for both transverse and longitudinal views.
    """
    
    # 1. EXTRACT REAL CONTOUR DIMENSIONS
    
    # Transverse view: real aligned contour dimensions
    dim_mayor_T, dim_menor_T = get_contour_real_dimensions(m_trans)
    print(f"DEBUG: real T dimensions: {dim_mayor_T:.1f}, {dim_menor_T:.1f}")
    
    # Calculate PCA angle for transverse view
    cnt_trans = largest_contour(m_trans)
    cnt_trans_centered = center2d(resample_contour(cnt_trans, 300))
    theta_trans = pca_angle_2d(cnt_trans_centered)
    print(f"Transverse PCA Angle: {np.degrees(theta_trans):.2f}°")
    
    # Longitudinal view: real aligned contour dimensions  
    dim_mayor_L, dim_menor_L = get_contour_real_dimensions(m_long)
    print(f"DEBUG: real L dimensions: {dim_mayor_L:.1f}, {dim_menor_L:.1f}")
    
    # Calculate PCA angle for longitudinal view
    cnt_long = largest_contour(m_long)
    cnt_long_centered = center2d(resample_contour(cnt_long, 300))
    theta_long = pca_angle_2d(cnt_long_centered)
    print(f"Longitudinal PCA Angle: {np.degrees(theta_long):.2f}°")
    
    print(f"Transverse contour dimensions: {dim_mayor_T:.1f} x {dim_menor_T:.1f} px")
    print(f"Longitudinal contour dimensions: {dim_mayor_L:.1f} x {dim_menor_L:.1f} px")
    
    # 2. TRANSVERSE CALIBRATION WITH PROPORTION PRESERVATION
    
    # Correspondence hypothesis for transverse view:
    # A: major_dimension → BASE_T, minor_dimension → H
    # B: minor_dimension → BASE_T, major_dimension → H
    
    cand_T = [
        (base_t_mm / dim_mayor_T, h_mm / dim_menor_T),  # A: major→BASE_T, minor→H
        (base_t_mm / dim_menor_T, h_mm / dim_mayor_T),  # B: minor→BASE_T, major→H
    ]
    
    # Target ratio (real melanoma proportion)
    ratio_target_T = base_t_mm / h_mm
    
    def score_transversal(escalas):
        """
        Evaluate how well real geometric proportion is preserved
        """
        sx, sy = escalas
        # Resulting ratio with these scales
        ratio_resultante = (dim_mayor_T * sx) / (dim_menor_T * sy)
        # Error in proportion preservation
        return abs(ratio_resultante - ratio_target_T)
    
    # Choose correspondence that best preserves proportions
    sxT, syT = min(cand_T, key=score_transversal)
    error_T = score_transversal((sxT, syT))
    
    print(f"Transverse - Target ratio: {ratio_target_T:.3f}")
    print(f"Transverse - Selected error: {error_T:.6f}")
    
    # 3. LONGITUDINAL CALIBRATION WITH THE SAME PRINCIPLE
    
    # Correspondence hypothesis for longitudinal view:
    cand_L = [
        (base_l_mm / dim_mayor_L, h_mm / dim_menor_L),  # A: major→BASE_L, minor→H
        (base_l_mm / dim_menor_L, h_mm / dim_mayor_L),  # B: minor→BASE_L, major→H
    ]
    
    # Target ratio for longitudinal view
    ratio_target_L = base_l_mm / h_mm
    
    def score_longitudinal(escalas):
        """
        Evaluate proportion preservation for longitudinal view
        """
        sx, sy = escalas
        ratio_resultante = (dim_mayor_L * sx) / (dim_menor_L * sy)
        return abs(ratio_resultante - ratio_target_L)
    
    # Choose the best correspondence
    sxL, syL = min(cand_L, key=score_longitudinal)
    error_L = score_longitudinal((sxL, syL))
    
    print(f"Longitudinal - Target ratio: {ratio_target_L:.3f}")
    print(f"Longitudinal - Selected error: {error_L:.6f}")
    
    # 4. ANATOMICAL CONSISTENCY VALIDATION
    
    # Verify that resulting dimensions are anatomically plausible
    dims_T = (dim_mayor_T * sxT, dim_menor_T * syT)
    dims_L = (dim_mayor_L * sxL, dim_menor_L * syL)
    
    print(f"\n=== VALIDATION ===")
    print(f"Dimensions from T: {dims_T[0]:.2f} x {dims_T[1]:.2f} mm")
    print(f"Dimensions from L: {dims_L[0]:.2f} x {dims_L[1]:.2f} mm")
    print(f"Clinical targets: {base_t_mm:.2f} x {h_mm:.2f} mm (T), {base_l_mm:.2f} x {h_mm:.2f} mm (L)")
    
    # Verify that heights are consistent
    altura_desde_T = dims_T[1] if dims_T[0] > dims_T[1] else dims_T[0]  # minor dimension
    altura_desde_L = dims_L[1] if dims_L[0] > dims_L[1] else dims_L[0]  # minor dimension
    
    consistencia_altura = abs(altura_desde_T - altura_desde_L)
    print(f"Height from T: {altura_desde_T:.2f} mm")
    print(f"Height from L: {altura_desde_L:.2f} mm") 
    print(f"Height difference: {consistencia_altura:.2f} mm")
    
    if consistencia_altura > 0.5:  # 0.5mm tolerance
        print("WARNING: Height inconsistency between views > 0.5mm")
    
    # 5. RETURN FINAL SCALES
    return (sxT, syT), (sxL, syL)


def verify_calibration(m_trans, m_long, sT_x, sT_y, sL_x, sL_y, base_t_mm, base_l_mm, h_mm):
    """
    Verify calibration by checking if calculated dimensions match expected values.
    """
    # Verify transverse dimensions
    dxT, dyT = get_contour_real_dimensions(m_trans)
    calc_base_t_mm = dxT * sT_x
    calc_h_mm_t = dyT * sT_y

    # Verify longitudinal dimensions
    dxL, dyL = get_contour_real_dimensions(m_long)
    calc_base_l_mm = dxL * sL_x
    calc_h_mm_l = dyL * sL_y

    print("\n=== CALIBRATION VERIFICATION ===")
    print(f"Transverse: Expected base = {base_t_mm:.2f} mm, calculated = {calc_base_t_mm:.2f} mm")
    print(f"Transverse: Expected height = {h_mm:.2f} mm, calculated = {calc_h_mm_t:.2f} mm")
    print(f"Longitudinal: Expected base = {base_l_mm:.2f} mm, calculated = {calc_base_l_mm:.2f} mm")
    print(f"Longitudinal: Expected height = {h_mm:.2f} mm, calculated = {calc_h_mm_l:.2f} mm")

    # Calculate relative errors
    error_base_t = abs(calc_base_t_mm - base_t_mm) / base_t_mm * 100
    error_h_t = abs(calc_h_mm_t - h_mm) / h_mm * 100
    error_base_l = abs(calc_base_l_mm - base_l_mm) / base_l_mm * 100
    error_h_l = abs(calc_h_mm_l - h_mm) / h_mm * 100

    print(f"Relative error Base T: {error_base_t:.2f} %")
    print(f"Relative error Height T: {error_h_t:.2f} %")
    print(f"Relative error Base L: {error_base_l:.2f} %")
    print(f"Relative error Height L: {error_h_l:.2f} %")