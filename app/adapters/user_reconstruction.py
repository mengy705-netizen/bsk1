"""
Paste or import the validated LC-Kriging implementation here.

Required public interface:
    reconstruct(sensor_coords, strain_values, load_info, grid_z, grid_x, config)
        -> 2-D ndarray with the same shape as grid_z/grid_x

Coordinates use millimetres:
    sensor_coords[:, 0] = span coordinate z
    sensor_coords[:, 1] = chord coordinate x
"""
from __future__ import annotations

import numpy as np


def reconstruct(
    sensor_coords: np.ndarray,
    strain_values: np.ndarray,
    load_info: dict,
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    config: dict,
) -> np.ndarray:
    # Replace the next line with a call to your LC-Kriging function, for example:
    # return lc_kriging(sensor_coords, strain_values, load_info["load_n"], grid_z, grid_x)
    raise RuntimeError(
        "adapters/user_reconstruction.py is a template. Insert your LC-Kriging code "
        "or select Demo load-guided RBF mode."
    )
