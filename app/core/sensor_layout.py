from __future__ import annotations

from datetime import datetime
import csv
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .mesh_io import PreparedMesh


class SensorLayoutError(RuntimeError):
    pass


def _layout_root(base_dir: str | Path) -> Path:
    root = Path(base_dir).resolve() / "sensor_layout"
    root.mkdir(parents=True, exist_ok=True)
    return root


def active_layout_path(base_dir: str | Path) -> Path:
    return _layout_root(base_dir) / "active_sensor_layout.json"


def _resolved_manifest(base: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    for key in ("layout_path", "mesh_npz_path", "original_geometry_path"):
        value = result.get(key)
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = (base / path).resolve()
            result[f"{key}_resolved"] = str(path)
    return result


def read_active_sensor_layout(base_dir: str | Path) -> dict[str, Any] | None:
    base = Path(base_dir).resolve()
    path = active_layout_path(base)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        raise SensorLayoutError(f"无法读取当前传感器布置文件：{path}") from exc
    return _resolved_manifest(base, manifest)


def validate_sensor_coordinates(coords: np.ndarray, expected: int = 48) -> np.ndarray:
    array = np.asarray(coords, dtype=float)
    if array.shape != (expected, 3):
        raise SensorLayoutError(
            f"传感器坐标必须为 {expected}×3，当前为 {array.shape}。"
        )
    if not np.all(np.isfinite(array)):
        raise SensorLayoutError("传感器坐标包含 NaN 或无穷大。")
    return array


def save_active_sensor_layout(
    base_dir: str | Path,
    prepared_mesh: PreparedMesh,
    sensor_coordinates_mm: np.ndarray,
    *,
    layout_name: str | None = None,
) -> Path:
    base = Path(base_dir).resolve()
    root = _layout_root(base)
    coords = validate_sensor_coordinates(sensor_coordinates_mm)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = layout_name or f"sensor_layout_{timestamp}"
    layout_dir = root / name
    suffix = 1
    while layout_dir.exists():
        layout_dir = root / f"{name}_{suffix:02d}"
        suffix += 1
    layout_dir.mkdir(parents=True, exist_ok=False)

    mesh_target = layout_dir / "wing_mesh.npz"
    geometry_target = layout_dir / prepared_mesh.original_copy_path.name
    shutil.copy2(prepared_mesh.npz_path, mesh_target)
    shutil.copy2(prepared_mesh.original_copy_path, geometry_target)

    csv_path = layout_dir / "sensor_coordinates.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sensor_id", "span_z_mm", "chord_x_mm", "thickness_y_mm"])
        for index, (z, x, y) in enumerate(coords, start=1):
            writer.writerow([index, f"{z:.9g}", f"{x:.9g}", f"{y:.9g}"])

    layout_payload = {
        "format_version": 1,
        "layout_name": layout_dir.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sensor_count": int(coords.shape[0]),
        "coordinate_system": {
            "columns": ["span_z_mm", "chord_x_mm", "thickness_y_mm"],
            "unit": "mm",
        },
        "sensor_coordinates_mm": coords.tolist(),
        "geometry": prepared_mesh.metadata,
        "mesh_npz_path": str(mesh_target.relative_to(base)),
        "original_geometry_path": str(geometry_target.relative_to(base)),
        "coordinates_csv": str(csv_path.relative_to(base)),
    }
    layout_json = layout_dir / "sensor_layout.json"
    with layout_json.open("w", encoding="utf-8") as handle:
        json.dump(layout_payload, handle, ensure_ascii=False, indent=2)

    manifest = {
        "format_version": 1,
        "layout_name": layout_payload["layout_name"],
        "updated_at": layout_payload["created_at"],
        "sensor_count": layout_payload["sensor_count"],
        "sensor_coordinates_mm": layout_payload["sensor_coordinates_mm"],
        "geometry": layout_payload["geometry"],
        "layout_path": str(layout_json.relative_to(base)),
        "mesh_npz_path": str(mesh_target.relative_to(base)),
        "original_geometry_path": str(geometry_target.relative_to(base)),
    }
    active_path = active_layout_path(base)
    temp_path = active_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    temp_path.replace(active_path)
    return active_path



def save_active_sensor_coordinates(
    base_dir: str | Path,
    sensor_coordinates_mm: np.ndarray,
    *,
    layout_name: str | None = None,
    geometry_name: str | None = None,
) -> Path:
    """Save only the 48 sensor coordinates. Wing geometry is managed separately."""
    base = Path(base_dir).resolve()
    root = _layout_root(base)
    coords = validate_sensor_coordinates(sensor_coordinates_mm)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = layout_name or f"sensor_layout_{timestamp}"
    layout_dir = root / name
    suffix = 1
    while layout_dir.exists():
        layout_dir = root / f"{name}_{suffix:02d}"
        suffix += 1
    layout_dir.mkdir(parents=True, exist_ok=False)

    csv_path = layout_dir / "sensor_coordinates.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sensor_id", "span_z_mm", "chord_x_mm", "thickness_y_mm"])
        for index, (z, x, y) in enumerate(coords, start=1):
            writer.writerow([index, f"{z:.9g}", f"{x:.9g}", f"{y:.9g}"])

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    layout_payload = {
        "format_version": 2,
        "layout_name": layout_dir.name,
        "created_at": created_at,
        "sensor_count": int(coords.shape[0]),
        "coordinate_system": {
            "columns": ["span_z_mm", "chord_x_mm", "thickness_y_mm"],
            "unit": "mm",
        },
        "sensor_coordinates_mm": coords.tolist(),
        "geometry_name": geometry_name,
        "coordinates_csv": str(csv_path.relative_to(base)),
    }
    layout_json = layout_dir / "sensor_layout.json"
    with layout_json.open("w", encoding="utf-8") as handle:
        json.dump(layout_payload, handle, ensure_ascii=False, indent=2)

    manifest = {
        "format_version": 2,
        "layout_name": layout_payload["layout_name"],
        "updated_at": created_at,
        "sensor_count": layout_payload["sensor_count"],
        "sensor_coordinates_mm": layout_payload["sensor_coordinates_mm"],
        "geometry_name": geometry_name,
        "layout_path": str(layout_json.relative_to(base)),
    }
    active_path = active_layout_path(base)
    temp_path = active_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    temp_path.replace(active_path)
    return active_path

def load_sensor_coordinates_file(path_like: str | Path, expected: int = 48) -> np.ndarray:
    path = Path(path_like)
    if not path.exists():
        raise SensorLayoutError(f"未找到传感器坐标文件：{path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        coords = payload.get("sensor_coordinates_mm")
        if coords is None:
            raise SensorLayoutError("JSON中缺少 sensor_coordinates_mm。")
        return validate_sensor_coordinates(np.asarray(coords, dtype=float), expected)

    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            values = []
            for cell in row:
                try:
                    values.append(float(cell))
                except Exception:
                    continue
            if len(values) >= 4:
                rows.append(values[-3:])
            elif len(values) == 3:
                rows.append(values)
    return validate_sensor_coordinates(np.asarray(rows, dtype=float), expected)


def export_sensor_coordinates_csv(path_like: str | Path, coords: np.ndarray) -> Path:
    path = Path(path_like)
    array = validate_sensor_coordinates(coords)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sensor_id", "span_z_mm", "chord_x_mm", "thickness_y_mm"])
        for index, row in enumerate(array, start=1):
            writer.writerow([index, *[f"{float(value):.9g}" for value in row]])
    return path
