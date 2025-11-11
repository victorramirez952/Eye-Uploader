"""
Enhanced Measurements Library
A modular image processing library for medical imaging measurements.
"""

from .morphology import (
    create_kernel,
    apply_opening,
    apply_closing
)

from .segmentation import (
    apply_mean_shift_to_crop
)

from .circle_fitting import (
    fit_circle_to_border_points
)

from .image_processing import (
    crop_bottom_image,
    crop_left_image,
    apply_binary_mask,
    remove_rows_with_high_white_percentage
)

from .distance_field import (
    create_signed_distance_field,
    create_binary_mask_from_sdf,
    find_bottom_left_white_pixel,
    find_next_white_pixel_on_right,
    measure_vertical_line_distances
)

from .measurer import Measurer

__version__ = '1.0.0'

__all__ = [
    'create_kernel',
    'apply_opening',
    'apply_closing',
    'apply_mean_shift_to_crop',
    'fit_circle_to_border_points',
    'crop_bottom_image',
    'crop_left_image',
    'apply_binary_mask',
    'remove_rows_with_high_white_percentage',
    'create_signed_distance_field',
    'create_binary_mask_from_sdf',
    'find_bottom_left_white_pixel',
    'find_next_white_pixel_on_right',
    'measure_vertical_line_distances',
    'Measurer',
]
