"""
Main orchestration script for 3D melanoma reconstruction.
This script coordinates the entire reconstruction pipeline using modular components.
"""

import numpy as np
import argparse
import pyvista as pv
import time
import os
import requests
import tempfile
from firebase_admin import storage

# Import modular components
from .config import *
from .image_utils import read_mask, analyze_orthogonality, select_masks_only, get_manual_measurements
from .calibration import calibrate_from_3measures_robust, verify_calibration
from .mesh_ops import (
    make_ring_xy_from_mask, extrude_ring, enforce_bbox, recenter_to_origin,
    create_longitudinal_ring, area_volume, smooth_with_implicit_modeling
)
from .analysis import (
    debug_calibration_info, debug_mesh_intersection, debug_longitudinal_pca,
    print_final_results, print_mesh_info, relative_volume_error
)
from .visualization import setup_pyvista, visualize_results
from .glb_exporter import GLBExporter

# Setup PyVista
setup_pyvista() 



class ExtrusionReconstruction:
    """
    Class for performing 3D melanoma reconstruction from ultrasound images.
    """
    
    def __init__(self):
        """Initialize the reconstruction class."""
        self.temp_dir = None
        
    def download_image(self, url: str, filename: str) -> str:
        """
        Download an image from URL to tempImages directory in the current directory.
        
        Parameters:
        -----------
        url : str
            URL of the image to download
        filename : str
            Name to save the file as
            
        Returns:
        --------
        str : Path to the downloaded file
        """
        # Create temp directory if not exists in the current directory
        if self.temp_dir is None:
            self.temp_dir = os.path.join(os.getcwd(), "tempImages")
            os.makedirs(self.temp_dir, exist_ok=True)

        response = requests.get(url)
        response.raise_for_status()
        
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return filepath
    
    def upload_to_firebase(self, local_path: str, storage_path: str) -> str:
        """
        Upload a file to Firebase Storage and return public URL.
        
        Parameters:
        -----------
        local_path : str
            Local file path to upload
        storage_path : str
            Destination path in Firebase Storage
            
        Returns:
        --------
        str : Public download URL
        """
        bucket = storage.bucket()
        blob = bucket.blob(storage_path)
        blob.content_type = 'model/gltf-binary'
        blob.upload_from_filename(local_path)
        blob.make_public()
        return blob.public_url
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def reconstruct(self, transversal_image_url: str, longitudinal_image_url: str,
                   base_t_mm: float, base_l_mm: float, h_mm: float) -> dict:
        """
        Main reconstruction pipeline.
        
        Parameters:
        -----------
        transversal_image_url : str
            URL of the transversal mask image
        longitudinal_image_url : str
            URL of the longitudinal mask image
        base_t_mm : float
            Basal thickness of transversal image (mm)
        base_l_mm : float
            Basal length of longitudinal image (mm)
        h_mm : float
            Height (mm)
            
        Returns:
        --------
        dict : Result containing GLB file URL and measurements
        """
        try:
            start_time = time.time()
            
            # Download mask images
            print("Downloading mask images...")
            mask_trans_path = self.download_image(transversal_image_url, "transversal_mask.png")
            mask_long_path = self.download_image(longitudinal_image_url, "longitudinal_mask.png")
            
            # Load masks
            print("Loading masks...")
            mT = read_mask(mask_trans_path)
            mL = read_mask(mask_long_path)
            
            # 1) Calibration mm/px
            print("Calibrating...")
            (sT_x, sT_y), (sL_x, sL_y) = calibrate_from_3measures_robust(
                mT, mL, base_t_mm, base_l_mm, h_mm
            )
            
            # Debug calibration
            debug_calibration_info(sT_x, sT_y, sL_x, sL_y)
            
            # Verify calibration
            verify_calibration(mT, mL, sT_x, sT_y, sL_x, sL_y, base_t_mm, base_l_mm, h_mm)

            # 2) Transverse → XY ring scaled, major axis aligned with X, EXTRUDE along Y=H
            print("Creating transverse mesh...")
            ring_T_xy = make_ring_xy_from_mask(mT, (sT_x, sT_y), N_CONTOUR_PTS)
            mesh_T = extrude_ring(ring_T_xy, h_mm, axis='y', n_layers=N_LAYERS)
            mesh_T = enforce_bbox(mesh_T, (base_t_mm, h_mm, base_l_mm))
            mesh_T = recenter_to_origin(mesh_T)
            
            print_mesh_info(mesh_T, "TRANSVERSE")

            # 3) Longitudinal → build YZ ring with mapping (X_img→Z, Y_img→Y)
            print("Creating longitudinal mesh...")
            ring_L_yz = create_longitudinal_ring(mL, sL_x, sL_y, FLIP_LONGITUDINAL_Y, N_CONTOUR_PTS)
            
            # Debug longitudinal PCA
            debug_longitudinal_pca(ring_L_yz, base_l_mm)

            # Extrude along X=BASE_T
            mesh_L = extrude_ring(ring_L_yz, base_t_mm, axis='x', n_layers=N_LAYERS)
            mesh_L = enforce_bbox(mesh_L, (base_t_mm, h_mm, base_l_mm))
            mesh_L = recenter_to_origin(mesh_L)
            
            print_mesh_info(mesh_L, "LONGITUDINAL")

            # Debug intersection
            debug_mesh_intersection(mesh_T, mesh_L)

            # 4) Boolean intersection
            print("Computing boolean intersection...")
            try:
                inter = mesh_T.boolean_intersection(mesh_L).triangulate().clean()
                # Fill small holes if any
                inter = inter.fill_holes(1000)
                inter = inter.clean() 

                print(f"Successful intersection: {inter.n_points} points")
            except Exception as e:
                print("Boolean operation failed:", e)
                raise RuntimeError(f"Boolean intersection failed: {e}")

            if inter.n_points == 0:
                raise RuntimeError("Empty intersection result")
            
            # Store original intersection before smoothing
            inter_original = inter.copy()

            # 4.5) Apply smoothing to result mesh if enabled
            if SMOOTH_RESULT and USE_IMPLICIT_MODELING:
                print("Applying implicit modeling smoothing...")
                inter = smooth_with_implicit_modeling(inter, (base_t_mm, h_mm, base_l_mm))
                inter = inter.clean()
                print(f"  - Processing completed")
                print(f"  - Final points: {inter.n_points}")
            elif SMOOTH_RESULT:
                # Fallback: Standard aggressive smoothing
                print("Applying standard smoothing to result mesh...")
                inter = inter.smooth(n_iter=SMOOTH_ITERATIONS, relaxation_factor=0.1,
                                    feature_smoothing=False, boundary_smoothing=True)
                inter = inter.fill_holes(1000)
                inter = enforce_bbox(inter, (base_t_mm, h_mm, base_l_mm))
                inter = recenter_to_origin(inter)
                inter = inter.clean()

            # 5) Decide which mesh to measure based on configuration
            mesh_to_measure = inter if MEASURE_SMOOTHED else inter_original
            
            # Measure the selected mesh
            area_mm2, vol_mm3 = area_volume(mesh_to_measure)
            
            print(f"\n=== MEASUREMENT INFO ===")
            print(f"Measuring: {'Smoothed mesh' if MEASURE_SMOOTHED else 'Original intersection (before smoothing)'}")
            
            print_final_results(mesh_to_measure, base_t_mm, h_mm, base_l_mm, area_mm2, vol_mm3)

            # 6) Decide which mesh to export based on configuration
            mesh_to_export = inter if EXPORT_SMOOTHED else inter_original
            
            # Apply additional visualization smoothing if exporting smoothed mesh and viz smoothing is enabled
            if EXPORT_SMOOTHED and SMOOTH_FOR_VIZ:
                print(f"Applying additional visualization smoothing to export...")
                mesh_to_export = mesh_to_export.smooth(n_iter=SMOOTH_ITERATIONS, relaxation_factor=0.1)
            
            print(f"\n=== EXPORT INFO ===")
            print(f"Exporting: {'Smoothed mesh' if EXPORT_SMOOTHED else 'Original intersection (before smoothing)'}")
            if EXPORT_SMOOTHED and SMOOTH_FOR_VIZ:
                print(f"  (with additional visualization smoothing: {SMOOTH_ITERATIONS} iterations)")

            # 6.1) Export to GLB
            print("Exporting to GLB...")
            glb_exporter = GLBExporter()
            vertices = mesh_to_export.points
            faces_array = mesh_to_export.faces
            if len(faces_array) > 0:
                n_faces = mesh_to_export.n_cells
                faces = faces_array.reshape(n_faces, -1)[:, 1:]
            else:
                faces = np.array([])
            
            # Create temporary GLB file
            glb_temp_path = os.path.join(self.temp_dir, "model.glb")
            success = glb_exporter.export_to_glb(vertices, faces, glb_temp_path, EXPORT_SMOOTHED)
            
            if not success:
                raise RuntimeError("Failed to export GLB file")
            
            print(f"✓ Model exported to {glb_temp_path}")
            
            # Upload to Firebase Storage
            print("Uploading to Firebase Storage...")
            import hashlib
            import time as time_module
            hash_str = hashlib.sha256(f"{transversal_image_url}{longitudinal_image_url}{time_module.time()}".encode()).hexdigest()
            storage_path = f"3d_models/{hash_str}.glb"
            download_url = self.upload_to_firebase(glb_temp_path, storage_path)
            
            print(f"✓ Model uploaded: {download_url}")

            end_time = time.time()
            print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
            
            # Return results
            return {
                "success": True,
                "glb_url": download_url,
                "area_mm2": float(area_mm2),
                "volume_mm3": float(vol_mm3),
                "processing_time": float(end_time - start_time)
            }
            
        except Exception as e:
            print(f"Error during reconstruction: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            # Clean up temporary files
            self.cleanup()


# ===================== MAIN PIPELINE (for standalone use) =====================

def main(use_defaults: bool = False):
    """
    Main orchestration function that coordinates the entire 3D reconstruction pipeline.
    
    Parameters:
    -----------
    use_defaults : bool
        If True, use default values from config.py
        If False, prompt user for file selection and measurements
    """
    # Determine whether to use defaults or interactive mode
    if use_defaults:
        print("\n=== USING DEFAULT CONFIGURATION ===\n")
        # Load masks from config
        mask_trans = TRANSVERSE_MASK
        mask_long = LONGITUDINAL_MASK
        base_t_mm = BASE_T_MM
        base_l_mm = BASE_L_MM
        h_mm = H_MM
    else:
        print("\n=== INTERACTIVE MODE ===\n")
        # 1) Select masks interactively
        mask_trans, mask_long = select_masks_only()
        
        # 2) Input measurements manually
        base_t_mm, base_l_mm, h_mm = get_manual_measurements()
    
    # Load masks
    start_time = time.time()
    mT = read_mask(mask_trans)
    mL = read_mask(mask_long)
    
    # Optional: Initial verification
    # print("=== INITIAL VERIFICATIONS ===")
    # if not analyze_orthogonality(mT, mL):
    #     print("WARNING! Masks seem to be rotated almost the same, they should be orthogonal.")
    #     print("Check that the original images are properly oriented.")

    # 1) Calibration mm/px
    (sT_x, sT_y), (sL_x, sL_y) = calibrate_from_3measures_robust(mT, mL, base_t_mm, base_l_mm, h_mm)
    
    # Debug calibration
    debug_calibration_info(sT_x, sT_y, sL_x, sL_y)
    
    # Verify calibration
    verify_calibration(mT, mL, sT_x, sT_y, sL_x, sL_y, base_t_mm, base_l_mm, h_mm)

    # 2) Transverse → XY ring scaled, major axis aligned with X, EXTRUDE along Y=H
    ring_T_xy = make_ring_xy_from_mask(mT, (sT_x, sT_y), N_CONTOUR_PTS)
    mesh_T = extrude_ring(ring_T_xy, h_mm, axis='y', n_layers=N_LAYERS)
    mesh_T = enforce_bbox(mesh_T, (base_t_mm, h_mm, base_l_mm))
    mesh_T = recenter_to_origin(mesh_T)
    
    print_mesh_info(mesh_T, "TRANSVERSE")

    # 3) Longitudinal → build YZ ring with mapping (X_img→Z, Y_img→Y)
    ring_L_yz = create_longitudinal_ring(mL, sL_x, sL_y, FLIP_LONGITUDINAL_Y, N_CONTOUR_PTS)
    
    # Debug longitudinal PCA
    debug_longitudinal_pca(ring_L_yz, base_l_mm)

    # Extrude along X=BASE_T
    mesh_L = extrude_ring(ring_L_yz, base_t_mm, axis='x', n_layers=N_LAYERS)
    mesh_L = enforce_bbox(mesh_L, (base_t_mm, h_mm, base_l_mm))
    mesh_L = recenter_to_origin(mesh_L)
    
    print_mesh_info(mesh_L, "LONGITUDINAL")

    # Debug intersection
    debug_mesh_intersection(mesh_T, mesh_L)

    # 4) Boolean intersection
    try:
        inter = mesh_T.boolean_intersection(mesh_L).triangulate().clean()
        # Fill small holes if any
        inter = inter.fill_holes(1000)
        inter = inter.clean() 

        print(f"Successful intersection: {inter.n_points} points")
    except Exception as e:
        print("Boolean operation failed:", e)
        # Create empty mesh to avoid errors
        import pyvista as pv
        inter = pv.PolyData()

    if inter.n_points == 0:
        print("Empty intersection")
        return
    
    # Store original intersection before smoothing
    inter_original = inter.copy()

    # 4.5) Apply smoothing to result mesh if enabled
    if SMOOTH_RESULT and USE_IMPLICIT_MODELING:
        inter = smooth_with_implicit_modeling(inter, (base_t_mm, h_mm, base_l_mm))
        inter = inter.clean()
        print(f"  - Processing completed")
        print(f"  - Final points: {inter.n_points}")
    elif SMOOTH_RESULT:
        # Fallback: Standard aggressive smoothing
        print("\nApplying standard smoothing to result mesh...")
        inter = inter.smooth(n_iter=SMOOTH_ITERATIONS, relaxation_factor=0.1,
                            feature_smoothing=False, boundary_smoothing=True)
        inter = inter.fill_holes(1000)
        inter = enforce_bbox(inter, (base_t_mm, h_mm, base_l_mm))
        inter = recenter_to_origin(inter)
        inter = inter.clean()

    # 5) Decide which mesh to measure based on configuration
    mesh_to_measure = inter if MEASURE_SMOOTHED else inter_original
    
    # Measure the selected mesh
    area_mm2, vol_mm3 = area_volume(mesh_to_measure)
    
    print(f"\n=== MEASUREMENT INFO ===")
    print(f"Measuring: {'Smoothed mesh' if MEASURE_SMOOTHED else 'Original intersection (before smoothing)'}")
    
    print_final_results(mesh_to_measure, base_t_mm, h_mm, base_l_mm, area_mm2, vol_mm3)

    # 6) Decide which mesh to export based on configuration
    mesh_to_export = inter if EXPORT_SMOOTHED else inter_original
    
    # Apply additional visualization smoothing if exporting smoothed mesh and viz smoothing is enabled
    if EXPORT_SMOOTHED and SMOOTH_FOR_VIZ:
        print(f"\nApplying additional visualization smoothing to export...")
        mesh_to_export = mesh_to_export.smooth(n_iter=SMOOTH_ITERATIONS, relaxation_factor=0.1)
    
    print(f"\n=== EXPORT INFO ===")
    print(f"Exporting: {'Smoothed mesh' if EXPORT_SMOOTHED else 'Original intersection (before smoothing)'}")
    if EXPORT_SMOOTHED and SMOOTH_FOR_VIZ:
        print(f"  (with additional visualization smoothing: {SMOOTH_ITERATIONS} iterations)")

    # 6.1) Export to GLB if enabled
    if EXPORT_TO_GLB:
        glb_exporter = GLBExporter()
        vertices = mesh_to_export.points
        faces_array = mesh_to_export.faces
        if len(faces_array) > 0:
            n_faces = mesh_to_export.n_cells
            faces = faces_array.reshape(n_faces, -1)[:, 1:]
        else:
            faces = np.array([])
        success = glb_exporter.export_to_glb(vertices, faces, GLB_OUTPUT_FILENAME, EXPORT_SMOOTHED)
        if success:
            print(f"✓ Model exported to {GLB_OUTPUT_FILENAME}")

    # 7) Visualization 
    if ENABLE_VISUALIZATION:
        visualize_results(mesh_T, mesh_L, inter, SMOOTH_FOR_VIZ, SMOOTH_ITERATIONS)
        
    else:
        print("Visualization disabled by config")
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='3D Melanoma Reconstruction from Ultrasound Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (select files and input measurements)
  python extrusion_reconstruccion.py
  
  # Use default values from config.py
  python extrusion_reconstruccion.py --default
        """
    )
    parser.add_argument(
        '--default',
        action='store_true',
        help='Use default configuration from config.py (no interactive prompts)'
    )
    
    args = parser.parse_args()
    
    # Run main pipeline with the specified mode
    main(use_defaults=args.default)