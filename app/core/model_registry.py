from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


class ModelRegistryError(RuntimeError):
    pass


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def active_manifest_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "models" / "active_model.json"


def read_active_manifest(base_dir: str | Path) -> dict[str, Any] | None:
    path = active_manifest_path(base_dir)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise ModelRegistryError(f"无法读取当前激活模型清单：{path}") from exc
    bundle_path = payload.get("bundle_path")
    if bundle_path:
        resolved = Path(bundle_path)
        if not resolved.is_absolute():
            resolved = (Path(base_dir) / resolved).resolve()
        payload["bundle_path_resolved"] = str(resolved)
    mesh_path = payload.get("mesh_npz_path")
    if mesh_path:
        resolved_mesh = Path(mesh_path)
        if not resolved_mesh.is_absolute():
            resolved_mesh = (Path(base_dir) / resolved_mesh).resolve()
        payload["mesh_npz_path_resolved"] = str(resolved_mesh)
    return payload


def activate_model(
    base_dir: str | Path,
    bundle_path: str | Path,
    model_dir: str | Path,
    metadata: dict[str, Any],
    mesh_npz_path: str | Path | None = None,
) -> Path:
    base = Path(base_dir).resolve()
    bundle = Path(bundle_path).resolve()
    directory = Path(model_dir).resolve()
    if not bundle.exists():
        raise ModelRegistryError(f"无法激活不存在的模型包：{bundle}")

    def portable(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.resolve().relative_to(base))
        except ValueError:
            return str(path.resolve())

    payload = {
        "format_version": 1,
        "model_name": metadata.get("model_name", directory.name),
        "model_type": metadata.get("model_type", "AE-FST"),
        "training_script": metadata.get("training_script"),
        "updated_at": utc_timestamp(),
        "bundle_path": portable(bundle),
        "model_dir": portable(directory),
        "mesh_npz_path": portable(Path(mesh_npz_path)) if mesh_npz_path else None,
        "geometry": metadata.get("geometry", {}),
        "metrics": metadata.get("metrics", {}),
        "dataset_summary": metadata.get("dataset_summary", {}),
        "training_config": metadata.get("training_config", {}),
    }
    path = active_manifest_path(base)
    _atomic_write_json(path, payload)
    return path


def write_metadata(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    _atomic_write_json(output, payload)
    return output


def pending_manifest_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "models" / "pending_model.json"


def read_pending_manifest(base_dir: str | Path) -> dict[str, Any] | None:
    path = pending_manifest_path(base_dir)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise ModelRegistryError(f"无法读取待更新模型清单：{path}") from exc

    for key, resolved_key in (
        ("bundle_path", "bundle_path_resolved"),
        ("model_dir", "model_dir_resolved"),
        ("mesh_npz_path", "mesh_npz_path_resolved"),
        ("metadata_path", "metadata_path_resolved"),
    ):
        value = payload.get(key)
        if not value:
            continue
        resolved = Path(value)
        if not resolved.is_absolute():
            resolved = (Path(base_dir) / resolved).resolve()
        payload[resolved_key] = str(resolved)
    return payload


def stage_model(
    base_dir: str | Path,
    bundle_path: str | Path,
    model_dir: str | Path,
    metadata: dict[str, Any],
    mesh_npz_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> Path:
    """Register a validated model as pending without replacing the active model."""
    base = Path(base_dir).resolve()
    bundle = Path(bundle_path).resolve()
    directory = Path(model_dir).resolve()
    if not bundle.exists():
        raise ModelRegistryError(f"无法登记不存在的模型包：{bundle}")

    def portable(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.resolve().relative_to(base))
        except ValueError:
            return str(path.resolve())

    payload = {
        "format_version": 1,
        "status": "validated_pending_activation",
        "model_name": metadata.get("model_name", directory.name),
        "model_type": metadata.get("model_type", "AE-FST"),
        "training_script": metadata.get("training_script"),
        "created_at": metadata.get("created_at", utc_timestamp()),
        "validated_at": utc_timestamp(),
        "bundle_path": portable(bundle),
        "model_dir": portable(directory),
        "mesh_npz_path": portable(Path(mesh_npz_path)) if mesh_npz_path else None,
        "metadata_path": portable(Path(metadata_path)) if metadata_path else None,
        "geometry": metadata.get("geometry", {}),
        "metrics": metadata.get("metrics", {}),
        "dataset_summary": metadata.get("dataset_summary", {}),
        "training_config": metadata.get("training_config", {}),
    }
    path = pending_manifest_path(base)
    _atomic_write_json(path, payload)
    return path


def clear_pending_manifest(base_dir: str | Path) -> None:
    path = pending_manifest_path(base_dir)
    if path.exists():
        path.unlink()
