from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

from .mesh_io import PreparedMesh


class WingGeometryRegistryError(RuntimeError):
    pass


def _geometry_root(base_dir: str | Path) -> Path:
    root = Path(base_dir).resolve() / "wing_geometry"
    root.mkdir(parents=True, exist_ok=True)
    return root


def active_geometry_path(base_dir: str | Path) -> Path:
    return _geometry_root(base_dir) / "active_wing_geometry.json"


def _resolved_manifest(base: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    for key in ("geometry_path", "mesh_npz_path", "original_geometry_path"):
        value = result.get(key)
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = (base / path).resolve()
            result[f"{key}_resolved"] = str(path)
    return result


def read_active_wing_geometry(base_dir: str | Path) -> dict[str, Any] | None:
    base = Path(base_dir).resolve()
    path = active_geometry_path(base)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        raise WingGeometryRegistryError(f"无法读取当前机翼三维模型配置：{path}") from exc
    return _resolved_manifest(base, manifest)


def save_active_wing_geometry(
    base_dir: str | Path,
    prepared_mesh: PreparedMesh,
    *,
    geometry_name: str | None = None,
) -> Path:
    base = Path(base_dir).resolve()
    root = _geometry_root(base)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = geometry_name or f"wing_geometry_{timestamp}"
    geometry_dir = root / name
    suffix = 1
    while geometry_dir.exists():
        geometry_dir = root / f"{name}_{suffix:02d}"
        suffix += 1
    geometry_dir.mkdir(parents=True, exist_ok=False)

    mesh_target = geometry_dir / "wing_mesh.npz"
    geometry_target = geometry_dir / prepared_mesh.original_copy_path.name
    shutil.copy2(prepared_mesh.npz_path, mesh_target)
    shutil.copy2(prepared_mesh.original_copy_path, geometry_target)

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {
        "format_version": 1,
        "geometry_name": geometry_dir.name,
        "created_at": created_at,
        "coordinate_system": {
            "columns": ["span_z_mm", "chord_x_mm", "thickness_y_mm"],
            "unit": "mm",
        },
        "geometry": prepared_mesh.metadata,
        "mesh_npz_path": str(mesh_target.relative_to(base)),
        "original_geometry_path": str(geometry_target.relative_to(base)),
    }
    geometry_json = geometry_dir / "wing_geometry.json"
    with geometry_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    manifest = {
        "format_version": 1,
        "geometry_name": payload["geometry_name"],
        "updated_at": created_at,
        "geometry": payload["geometry"],
        "geometry_path": str(geometry_json.relative_to(base)),
        "mesh_npz_path": str(mesh_target.relative_to(base)),
        "original_geometry_path": str(geometry_target.relative_to(base)),
    }
    active_path = active_geometry_path(base)
    temp_path = active_path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    temp_path.replace(active_path)
    return active_path
