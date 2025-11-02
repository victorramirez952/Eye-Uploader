#!/usr/bin/env python3
"""
GLB file exporter module for 3D reconstruction.
Handles exporting 3D models to GLB format.
"""

import numpy as np
import trimesh
from typing import Tuple


class GLBExporter:
    """Handles exporting 3D models to GLB file format."""
    
    def __init__(self):
        pass
    
    def export_to_glb(self, vertices: np.ndarray, faces: np.ndarray, 
                     filename: str = "tomography_model.glb",
                     is_smoothed: bool = True) -> bool:
        """
        Export 3D model to GLB format.
        
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
        """
        try:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Add metadata about mesh type
            mesh_type = "smoothed" if is_smoothed else "original"
            mesh.metadata['model_type'] = mesh_type
            
            mesh.export(filename)
            
            print(f"Model exported: {filename}")
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False
