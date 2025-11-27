#!/usr/bin/env python3
"""
GLB file exporter module for 3D reconstruction.
Handles exporting 3D models to GLB format.
"""

import numpy as np
import trimesh
from typing import Tuple, Optional


class GLBExporter:
    """Handles exporting 3D models to GLB file format."""
    
    def __init__(self):
        pass
    
    def export_to_glb(self, vertices: np.ndarray, faces: np.ndarray, 
                     filename: str = "tomography_model.glb",
                     is_smoothed: bool = True,
                     markers: Optional[dict] = None) -> bool:
        """
        Export 3D model to GLB format with optional anatomical markers.
        
        Parameters:
        -----------
        vertices : np.ndarray
            Vertex coordinates array
        faces : np.ndarray
            Face indices array
        filename : str
            Output filename
        is_smoothed : bool
            Whether the mesh being exported is smoothed
        markers : dict, optional
            Anatomical markers to export as colored spheres
        """
        try:
            # Create main mesh
            main_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Add metadata about mesh type
            mesh_type = "smoothed" if is_smoothed else "original"
            main_mesh.metadata['model_type'] = mesh_type
            
            # If no markers, export simple mesh
            if not markers:
                main_mesh.export(filename)
                print(f"Model exported: {filename}")
                return True
            
            # Create scene with main mesh and markers
            scene = trimesh.Scene()
            scene.add_geometry(main_mesh, node_name='melanoma_mesh')
            
            # Add markers as colored spheres
            for region, marker_info in markers.items():
                position = marker_info['position']
                color = marker_info['color']  # RGB 0-255
                
                # Create sphere for marker (radius = 0.8mm for visibility)
                sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.3)
                sphere.apply_translation(position)
                
                # Set color (RGBA, normalized to 0-1)
                sphere.visual.vertex_colors = [
                    color[0] / 255.0,
                    color[1] / 255.0,
                    color[2] / 255.0,
                    1.0  # Alpha
                ]
                
                scene.add_geometry(sphere, node_name=f'marker_{region}')
            
            # Export scene to GLB
            scene.export(filename)
            
            print(f"Model with markers exported: {filename}")
            print(f"  - Main mesh: {len(vertices)} vertices, {len(faces)} faces")
            print(f"  - Markers: {len(markers)} anatomical labels")
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False
