#!/usr/bin/env python3
"""
OBJ file exporter module for 3D reconstruction.
Handles exporting 3D models to OBJ format.
"""

import numpy as np
from typing import Tuple


class OBJExporter:
    """Handles exporting 3D models to OBJ file format."""
    
    def __init__(self):
        pass
    
    def export_to_obj(self, vertices: np.ndarray, faces: np.ndarray, 
                     filename: str = "tomography_model.obj",
                     is_smoothed: bool = True) -> bool:
        """
        Export 3D model to OBJ format.
        
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
            with open(filename, 'w') as f:
                mesh_type = "smoothed" if is_smoothed else "original"
                f.write(f"# Tomography reconstruction model ({mesh_type})\n")
                f.write(f"# {len(vertices)} vertices, {len(faces)} faces\n\n")
                
                for vertex in vertices:
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                
                f.write("\n")
                
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            print(f"Model exported: {filename}")
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False