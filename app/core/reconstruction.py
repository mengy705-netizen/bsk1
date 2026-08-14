from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RBFInterpolator


class ReconstructionInterfaceError(RuntimeError):
    pass


class StrainReconstructor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.last_method = ""

    def reconstruct(
        self,
        sensor_coords: np.ndarray,
        strain_values: np.ndarray,
        load_info: dict[str, float],
        grid_z: np.ndarray,
        grid_x: np.ndarray,
        mode: str,
        code_path: str | None = None,
    ) -> np.ndarray:
        if mode == "User LC-Kriging":
            from .lc_kriging import reconstruct_lc_kriging

            result = reconstruct_lc_kriging(
                sensor_coords, strain_values, load_info, grid_z, grid_x, self.config
            )
            self.last_method = result.method
            return result.field
        if mode == "Demo load-guided RBF":
            self.last_method = "Demo load-guided RBF"
            return self._demo_load_guided_rbf(
                sensor_coords, strain_values, load_info, grid_z, grid_x
            )
        if mode == "Custom Python":
            if not code_path:
                raise ReconstructionInterfaceError(
                    "请选择LC-Kriging的Python代码文件。"
                )
            self.last_method = "Custom Python"
            return self._custom_python(
                Path(code_path), sensor_coords, strain_values, load_info, grid_z, grid_x
            )
        raise ReconstructionInterfaceError(f"不支持的重构方式：{mode}")

    def _custom_python(
        self,
        path: Path,
        sensor_coords: np.ndarray,
        strain_values: np.ndarray,
        load_info: dict[str, float],
        grid_z: np.ndarray,
        grid_x: np.ndarray,
    ) -> np.ndarray:
        if not path.exists():
            raise ReconstructionInterfaceError(f"未找到重构代码文件：{path}")
        spec = importlib.util.spec_from_file_location("wing_user_reconstruction", path)
        if spec is None or spec.loader is None:
            raise ReconstructionInterfaceError(f"无法导入重构代码文件：{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, "reconstruct", None)
        if not callable(fn):
            raise ReconstructionInterfaceError(
                "自定义重构文件必须定义 reconstruct(sensor_coords, "
                "strain_values, load_info, grid_z, grid_x, config)."
            )
        field = fn(
            sensor_coords.copy(),
            np.asarray(strain_values, dtype=float).copy(),
            dict(load_info),
            grid_z.copy(),
            grid_x.copy(),
            self.config,
        )
        field_array = np.asarray(field, dtype=float)
        if field_array.shape != grid_z.shape:
            raise ReconstructionInterfaceError(
                f"重构结果形状为{field_array.shape}，应为{grid_z.shape}。"
            )
        return field_array

    def _demo_load_guided_rbf(
        self,
        sensor_coords: np.ndarray,
        strain_values: np.ndarray,
        load_info: dict[str, float],
        grid_z: np.ndarray,
        grid_x: np.ndarray,
    ) -> np.ndarray:
        """
        Demonstration interpolation, not the user's validated LC-Kriging algorithm.

        It uses a thin-plate RBF and a small load-centred correction that preserves the
        sensor values approximately while sharpening the field near the predicted load.
        """
        coords = np.asarray(sensor_coords, dtype=float)
        values = np.asarray(strain_values, dtype=float).reshape(-1)
        query = np.column_stack([grid_z.ravel(), grid_x.ravel()])

        finite = np.isfinite(values)
        if finite.sum() < 4:
            raise ReconstructionInterfaceError("至少需要4个有效应变值才能进行重构。")
        coords = coords[finite]
        values = values[finite]

        smoothing = max(float(np.var(values)) * 1e-4, 1e-9)
        rbf = RBFInterpolator(coords, values, kernel="thin_plate_spline", smoothing=smoothing)
        base = rbf(query).reshape(grid_z.shape)

        load_z = float(load_info["load_z_mm"])
        load_x = float(load_info["load_x_mm"])
        load_n = max(float(load_info["load_n"]), 0.0)
        span = float(self.config["geometry"]["span_mm"])
        root_chord = float(self.config["geometry"]["root_chord_mm"])
        sigma_z = 0.09 * span
        sigma_x = 0.11 * root_chord
        gaussian = np.exp(
            -0.5
            * (
                ((grid_z - load_z) / max(sigma_z, 1e-9)) ** 2
                + ((grid_x - load_x) / max(sigma_x, 1e-9)) ** 2
            )
        )
        signed_scale = np.sign(np.nanmedian(values) or 1.0)
        correction = signed_scale * np.nanstd(values) * np.log1p(load_n) * 0.12 * gaussian
        return base + correction
