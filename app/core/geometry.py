from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class WingGeometry:
    span_mm: float
    root_chord_mm: float
    tip_chord_mm: float

    def local_chord(self, z_mm: np.ndarray | float) -> np.ndarray | float:
        """Return local chord length at span coordinate z."""
        return self.root_chord_mm + (
            self.tip_chord_mm - self.root_chord_mm
        ) * np.asarray(z_mm) / self.span_mm

    def polygon(self) -> np.ndarray:
        """Top-view polygon in (z, x) coordinates."""
        return np.array(
            [
                [0.0, 0.0],
                [self.span_mm, 0.0],
                [self.span_mm, self.tip_chord_mm],
                [0.0, self.root_chord_mm],
            ],
            dtype=float,
        )

    def contains(self, z_mm: np.ndarray, x_mm: np.ndarray) -> np.ndarray:
        return (
            (z_mm >= 0.0)
            & (z_mm <= self.span_mm)
            & (x_mm >= 0.0)
            & (x_mm <= self.local_chord(z_mm))
        )


def build_sensor_coordinates(
    geometry: WingGeometry,
    span_fractions: Iterable[float],
    chord_fractions: Iterable[float],
) -> np.ndarray:
    """
    Build 48 sensor coordinates in span-major order.

    For each span station, all chordwise fractions are listed from leading edge
    to trailing edge. Returned columns are [z_mm, x_mm].
    """
    coords: list[tuple[float, float]] = []
    for span_fraction in span_fractions:
        z_mm = float(span_fraction) * geometry.span_mm
        chord = float(geometry.local_chord(z_mm))
        for chord_fraction in chord_fractions:
            coords.append((z_mm, float(chord_fraction) * chord))
    return np.asarray(coords, dtype=float)


def make_grid(
    geometry: WingGeometry, span_points: int, chord_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.linspace(0.0, geometry.span_mm, span_points)
    x = np.linspace(0.0, geometry.root_chord_mm, chord_points)
    grid_z, grid_x = np.meshgrid(z, x)
    mask = geometry.contains(grid_z, grid_x)
    return grid_z, grid_x, mask
