from __future__ import annotations

"""LC-Kriging compatibility reconstruction module.

This module adapts the reconstruction function from the user's original
seven-model inference script to the software's current geometry and manually
edited 48-sensor coordinates.

The source implementation selects the following interpolation methods in order:
1. PyKrige OrdinaryKriging;
2. SciPy linear RBF;
3. nearest-neighbour fallback.

The predicted load/damage result is accepted for interface compatibility and
future load-control extensions. The supplied source algorithm itself does not
feed the predicted load into the variogram or interpolation equations.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from pykrige.ok import OrdinaryKriging

    HAS_PYKRIGE = True
except Exception:
    OrdinaryKriging = None
    HAS_PYKRIGE = False

try:
    from scipy.interpolate import Rbf

    HAS_RBF = True
except Exception:
    Rbf = None
    HAS_RBF = False


class LCKrigingError(RuntimeError):
    pass


@dataclass(frozen=True)
class LCKrigingResult:
    field: np.ndarray
    method: str


def _validate_inputs(
    sensor_coords: np.ndarray,
    strain_values: np.ndarray,
    grid_z: np.ndarray,
    grid_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(sensor_coords, dtype=float)
    values = np.asarray(strain_values, dtype=float).reshape(-1)
    gz = np.asarray(grid_z, dtype=float)
    gx = np.asarray(grid_x, dtype=float)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise LCKrigingError("传感器坐标必须为 N×2 或 N×3 数组，前两列依次为 z、x 坐标。")
    if coords.shape[0] != values.size:
        raise LCKrigingError(
            f"传感器数量与应变值数量不一致：{coords.shape[0]} vs {values.size}。"
        )
    if gz.shape != gx.shape or gz.ndim != 2:
        raise LCKrigingError("grid_z 与 grid_x 必须是形状一致的二维网格。")

    finite = np.isfinite(values) & np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    if finite.sum() < 4:
        raise LCKrigingError("至少需要4个有效传感器点才能进行应变场重构。")
    return coords[finite, :2], values[finite], gz, gx


def _ordinary_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    variogram: str,
) -> np.ndarray:
    if not HAS_PYKRIGE or OrdinaryKriging is None:
        raise LCKrigingError("当前环境未安装 PyKrige。")

    z_axis = np.asarray(grid_z[0, :], dtype=float)
    x_axis = np.asarray(grid_x[:, 0], dtype=float)
    ok = OrdinaryKriging(
        coords[:, 0],
        coords[:, 1],
        values,
        variogram_model=variogram,
        enable_plotting=False,
        coordinates_type="euclidean",
        verbose=False,
    )
    estimate, _ = ok.execute("grid", z_axis, x_axis)
    field = np.asarray(estimate, dtype=float)
    if field.shape == grid_z.shape:
        return field
    if field.T.shape == grid_z.shape:
        return field.T
    raise LCKrigingError(
        f"OrdinaryKriging 输出形状为 {field.shape}，软件网格形状为 {grid_z.shape}。"
    )


def _linear_rbf(
    coords: np.ndarray,
    values: np.ndarray,
    grid_z: np.ndarray,
    grid_x: np.ndarray,
) -> np.ndarray:
    if not HAS_RBF or Rbf is None:
        raise LCKrigingError("当前环境缺少 SciPy Rbf。")
    rbf = Rbf(coords[:, 0], coords[:, 1], values, function="linear")
    return np.asarray(rbf(grid_z, grid_x), dtype=float)


def _nearest_neighbour(
    coords: np.ndarray,
    values: np.ndarray,
    grid_z: np.ndarray,
    grid_x: np.ndarray,
) -> np.ndarray:
    query = np.column_stack([grid_z.ravel(), grid_x.ravel()])
    # 240×180×48 remains small enough for a direct vectorised calculation.
    distance2 = (
        (query[:, None, 0] - coords[None, :, 0]) ** 2
        + (query[:, None, 1] - coords[None, :, 1]) ** 2
    )
    nearest = np.argmin(distance2, axis=1)
    return values[nearest].reshape(grid_z.shape)


def reconstruct_lc_kriging(
    sensor_coords: np.ndarray,
    strain_values: np.ndarray,
    load_info: dict[str, Any] | None,
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    config: dict[str, Any] | None = None,
) -> LCKrigingResult:
    """Reconstruct the field using the user's original fallback sequence.

    `load_info` is retained in the interface because the software first runs the
    seven-model prediction and then performs reconstruction. The uploaded source
    implementation does not use this dictionary in the interpolation formula.
    """

    coords, values, gz, gx = _validate_inputs(
        sensor_coords, strain_values, grid_z, grid_x
    )
    cfg = config or {}
    recon_cfg = cfg.get("lc_kriging", {}) if isinstance(cfg, dict) else {}
    variogram = str(recon_cfg.get("variogram", "linear"))
    allow_rbf = bool(recon_cfg.get("allow_rbf_fallback", True))
    allow_nearest = bool(recon_cfg.get("allow_nearest_fallback", True))

    errors: list[str] = []
    if HAS_PYKRIGE:
        try:
            field = _ordinary_kriging(coords, values, gz, gx, variogram)
            return LCKrigingResult(field=field, method=f"OrdinaryKriging/{variogram}")
        except Exception as exc:
            errors.append(f"OrdinaryKriging: {exc}")

    if allow_rbf and HAS_RBF:
        try:
            field = _linear_rbf(coords, values, gz, gx)
            return LCKrigingResult(field=field, method="SciPy RBF/linear")
        except Exception as exc:
            errors.append(f"RBF: {exc}")

    if allow_nearest:
        field = _nearest_neighbour(coords, values, gz, gx)
        return LCKrigingResult(field=field, method="Nearest-neighbour fallback")

    detail = "；".join(errors) if errors else "没有可用的插值后端"
    raise LCKrigingError(f"LC-Kriging重构失败：{detail}")
