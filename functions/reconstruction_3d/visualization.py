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


def visualize_results(mesh_T, mesh_L, inter, smooth_for_viz: bool = True, smooth_iterations: int = 100, markers: dict = None):
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
    markers : dict, optional
        Anatomical markers to display on the intersection mesh
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
    
    marker_actors = []
    markers_visible = [True]
    
    # Subplot 0: Transverse
    p.subplot(0, 0)
    p.add_mesh(viz_T, color='lightblue', show_edges=False, smooth_shading=True)
    p.add_axes()
    p.show_grid()
    p.add_title('T extruded (Y=H)')
    p.camera_position = 'iso'
    
    # Subplot 1: Longitudinal
    p.subplot(0, 1)
    p.add_mesh(viz_L, color='lightblue', show_edges=False, smooth_shading=True)
    p.add_axes()
    p.show_grid()
    p.add_title('L extruded (X=BASE_T)')
    p.camera_position = 'iso'
    
    # Subplot 2: Intersection with anatomical markers
    p.subplot(0, 2)
    p.add_mesh(viz_I, color='lightgray', show_edges=False, smooth_shading=True)
    
    # Add anatomical markers if provided
    if markers:
        for region, marker_info in markers.items():
            point_actor = p.add_points(
                np.array([marker_info['position']]),
                color=[c/255.0 for c in marker_info['color']],
                point_size=20,
                render_points_as_spheres=True,
                name=f"point_{region}"
            )
            marker_actors.append(point_actor)
            
            label_actor = p.add_point_labels(
                [marker_info['position']], 
                [marker_info['label']],
                font_size=14,
                text_color='white',
                bold=True,
                shape_opacity=0.7,
                shape_color=[c/255.0 for c in marker_info['color']],
                point_size=8,
                name=f"label_{region}"
            )
            marker_actors.append(label_actor)
        
        # Toggle markers callback
        def toggle_markers_slider(value):
            state = bool(round(value))
            markers_visible[0] = state
            for actor in marker_actors:
                actor.SetVisibility(state)
        
        # Add slider widget
        p.add_slider_widget(
            callback=toggle_markers_slider,
            rng=[0, 1],
            value=1,
            title="Markers",
            pointa=(0.025, 0.05),
            pointb=(0.25, 0.05),
            style='modern',
            color='lightgreen',
            tube_width=0.005,
            slider_width=0.02,
            fmt="%.0f",
            title_height=0.025,
            title_opacity=1.0,
            title_color='white',
            interaction_event='always'
        )
    
    p.add_axes()
    p.show_grid()
    p.add_title('3D Reconstruction (Final)')
    p.camera_position = 'iso'
    
    p.link_views()
    p.show()