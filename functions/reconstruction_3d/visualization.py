"""
Visualization functions for 3D reconstruction results.
"""

import pyvista as pv
import numpy as np


def setup_pyvista():
    """
    Setup PyVista global settings.
    """
    pv.global_theme.allow_empty_mesh = True


def extract_mesh_edges(mesh: pv.PolyData, feature_angle=30):
    """
    Extract different types of edges from a mesh.
    
    Parameters:
    -----------
    mesh : pv.PolyData
        The mesh to extract edges from
    feature_angle : float
        Angle threshold for feature edge detection
    
    Returns:
    --------
    dict : Different edge types as separate PolyData objects
    """
    if mesh.n_points == 0:
        return {
            'boundary_edges': pv.PolyData(),
            'feature_edges': pv.PolyData(),
            'all_edges': pv.PolyData()
        }
    
    try:
        # Extract boundary edges (outer contour)
        boundary_edges = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            non_manifold_edges=False
        )
        
        # Extract feature edges (sharp internal edges)
        feature_edges = mesh.extract_feature_edges(
            boundary_edges=False,
            feature_edges=True,
            feature_angle=feature_angle
        )
        
        # Extract all edges
        try:
            all_edges = mesh.extract_all_edges()
        except AttributeError:
            all_edges = mesh.extract_feature_edges(
                boundary_edges=True,
                feature_edges=True,
                non_manifold_edges=True,
                manifold_edges=True,
                feature_angle=180
            )
        
        return {
            'boundary_edges': boundary_edges,
            'feature_edges': feature_edges,
            'all_edges': all_edges
        }
        
    except Exception as e:
        print(f"Warning: Edge extraction failed: {e}")
        return {
            'boundary_edges': pv.PolyData(),
            'feature_edges': pv.PolyData(),
            'all_edges': pv.PolyData()
        } 


def visualize_results(mesh_T, mesh_L, inter, smooth_for_viz: bool = True, smooth_iterations: int = 100):
    """
    Visualize the reconstruction results in a 3-panel display.
    
    Parameters:
    -----------
    mesh_T : pv.PolyData
        Transverse mesh
    mesh_L : pv.PolyData
        Longitudinal mesh
    inter : pv.PolyData
        Intersection mesh
    smooth_for_viz : bool
        Whether to apply additional smoothing for visualization
    smooth_iterations : int
        Number of smoothing iterations for visualization
    """
    # Apply smoothing for visualization if requested
    if smooth_for_viz:
        print(f"\nApplying additional smoothing for visualization...")
        viz_T = mesh_T.smooth(n_iter=smooth_iterations, relaxation_factor=0.1)
        viz_L = mesh_L.smooth(n_iter=smooth_iterations, relaxation_factor=0.1)
        viz_I = inter.smooth(n_iter=smooth_iterations, relaxation_factor=0.1)
    else:
        viz_T, viz_L, viz_I = mesh_T, mesh_L, inter

    # Create 3-panel visualization
    p = pv.Plotter(shape=(1, 3))
    
    meshes_and_titles = [
        (viz_T, 'T extruded (Y=H)'),
        (viz_L, 'L extruded (X=BASE_T)'),
        (viz_I, 'Intersection')
    ]
    
    for i, (mesh, title) in enumerate(meshes_and_titles):
        p.subplot(0, i)
        p.add_mesh(mesh, show_edges=False, smooth_shading=True)  # smooth_shading improves appearance
        p.add_axes()
        p.show_grid()
        p.add_title(title)
        p.camera_position = 'iso'
    
    p.link_views()
    p.show()