"""
Configuration settings for the 3D reconstruction pipeline.
"""

# ===================== CONFIGURATION =====================

## Mask file paths (to be set by user)
TRANSVERSE_MASK = r""
LONGITUDINAL_MASK = r""

## Patient measurements (mm)
H_MM = 1.00    # altura/espesor (mm)
BASE_T_MM = 1.00   # diámetro basal medido en Transversal (mm)
BASE_L_MM = 1.00   # diámetro basal medido en Longitudinal (mm)

# Processing settings
FLIP_LONGITUDINAL_Y = False   # if Y axis is inverted in longitudinal view
N_CONTOUR_PTS = 300           # points to sample the contour
N_LAYERS = 400                # layers in extrusion
SMOOTH_FOR_VIZ = True         # smooth only for visualization (not for measurement)
SMOOTH_ITERATIONS = 100       # iterations for visualization smoothing
SMOOTH_RESULT = True          # apply smoothing to result mesh before measurement
USE_IMPLICIT_MODELING = True  # use implicit modeling (voxelization) for smoother results
MEASURE_SMOOTHED = False      # if True, measure the smoothed mesh; if False, measure the original intersection

# Output settings
ENABLE_VISUALIZATION = False   # Toggle PyVista 3D visualization
EXPORT_TO_OBJ = True          # Toggle OBJ file export
OBJ_OUTPUT_FILENAME = ""  # Output filename for OBJ export
EXPORT_TO_GLB = True          # Toggle GLB file export
GLB_OUTPUT_FILENAME = ""  # Output filename for GLB export
EXPORT_SMOOTHED = True        # if True, export the smoothed mesh; if False, export the original intersection