from __future__ import annotations

from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np


class ExternalModelError(RuntimeError):
    pass


TYPE_LABELS = {
    "keras_v3_bundle": "Keras v3迁移模型包",
    "pretrain_bundle": "BP-CNN预训练模型包",
    "keras_generic": "Keras通用模型",
    "joblib": "Joblib / sklearn模型",
    "torchscript": "TorchScript模型",
    "python_adapter": "Python模型适配器",
    "seven_model_group": "七模型组合",
}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _portable(base: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve(base: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def active_external_manifest_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "external_models" / "active_external_model.json"


def read_active_external_model(base_dir: str | Path) -> dict[str, Any] | None:
    base = Path(base_dir).resolve()
    path = active_external_manifest_path(base)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise ExternalModelError(f"无法读取外部模型清单：{path}") from exc
    for key in ("model_path", "input_scaler_path", "output_scaler_path"):
        payload[key + "_resolved"] = _resolve(base, payload.get(key))
    if payload.get("model_type") == "seven_model_group":
        from .seven_model_group import resolve_seven_model_slots
        payload = resolve_seven_model_slots(base, payload)
    return payload


def clear_active_external_model(base_dir: str | Path) -> None:
    path = active_external_manifest_path(base_dir)
    if path.exists():
        path.unlink()


def _find_bundle(selected: Path, expected_type: str) -> Path:
    if selected.is_file():
        candidates = [selected]
    else:
        names = ["keras_v3_bundle.json"] if expected_type == "Keras-AE-guided-KD-v3" else ["pretrain_bundle.json"]
        candidates = [selected / name for name in names]
        candidates += list(selected.glob("*.json"))
    for candidate in candidates:
        if not candidate.exists() or candidate.suffix.lower() != ".json":
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        if data.get("model_type") == expected_type:
            return candidate
    raise ExternalModelError(f"未找到 model_type={expected_type} 的模型包JSON。")


def normalize_selected_model(model_type: str, selected_path: str | Path) -> Path:
    selected = Path(selected_path).expanduser().resolve()
    if not selected.exists():
        raise ExternalModelError(f"模型路径不存在：{selected}")
    if model_type == "keras_v3_bundle":
        return _find_bundle(selected, "Keras-AE-guided-KD-v3")
    if model_type == "pretrain_bundle":
        return _find_bundle(selected, "BP-CNN-Wing-Pretrain-v1")
    allowed = {
        "keras_generic": {".keras", ".h5", ".hdf5"},
        "joblib": {".joblib", ".pkl", ".pickle", ".save"},
        "torchscript": {".pt", ".pth", ".ts", ".torchscript"},
        "python_adapter": {".py"},
    }
    if model_type not in allowed:
        raise ExternalModelError(f"不支持的外部模型类型：{model_type}")
    if not selected.is_file() or selected.suffix.lower() not in allowed[model_type]:
        raise ExternalModelError(f"所选文件与“{TYPE_LABELS.get(model_type, model_type)}”类型不匹配。")
    return selected


def _copy_bundle(source_json: Path, destination: Path) -> Path:
    source_dir = source_json.parent
    target_dir = destination / source_dir.name
    shutil.copytree(source_dir, target_dir)
    return target_dir / source_json.name


def install_external_model(
    base_dir: str | Path,
    *,
    model_type: str,
    selected_path: str | Path,
    input_scaler_path: str | Path | None = None,
    output_scaler_path: str | Path | None = None,
    output_order: list[str] | None = None,
    copy_into_software: bool = True,
) -> Path:
    base = Path(base_dir).resolve()
    model_path = normalize_selected_model(model_type, selected_path)
    in_scaler = Path(input_scaler_path).expanduser().resolve() if input_scaler_path else None
    out_scaler = Path(output_scaler_path).expanduser().resolve() if output_scaler_path else None
    for label, path in (("输入归一化器", in_scaler), ("输出归一化器", out_scaler)):
        if path is not None and not path.exists():
            raise ExternalModelError(f"{label}不存在：{path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if copy_into_software:
        import_root = base / "external_models" / f"imported_{timestamp}"
        import_root.mkdir(parents=True, exist_ok=False)
        if model_type in {"keras_v3_bundle", "pretrain_bundle"}:
            model_path = _copy_bundle(model_path, import_root)
        else:
            target = import_root / model_path.name
            shutil.copy2(model_path, target)
            model_path = target
        if in_scaler is not None:
            target = import_root / in_scaler.name
            shutil.copy2(in_scaler, target)
            in_scaler = target
        if out_scaler is not None:
            target = import_root / out_scaler.name
            shutil.copy2(out_scaler, target)
            out_scaler = target

    payload = {
        "format_version": 1,
        "status": "active",
        "model_name": model_path.parent.name if model_type in {"keras_v3_bundle", "pretrain_bundle"} else model_path.stem,
        "model_type": model_type,
        "model_type_label": TYPE_LABELS.get(model_type, model_type),
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_path": _portable(base, model_path),
        "input_scaler_path": _portable(base, in_scaler),
        "output_scaler_path": _portable(base, out_scaler),
        "output_order": output_order or ["load_x_mm", "load_z_mm", "load_n"],
        "copied_into_software": bool(copy_into_software),
        "source_path": str(Path(selected_path).expanduser().resolve()),
    }
    manifest = active_external_manifest_path(base)
    _atomic_write_json(manifest, payload)
    return manifest


def _load_scaler(path: str | None):
    if not path:
        return None
    import joblib
    return joblib.load(path)


def _to_dict(raw: Any, order: list[str]) -> dict[str, Any]:
    if isinstance(raw, dict):
        result = {str(k): v for k, v in raw.items()}
    else:
        values = np.asarray(raw).reshape(-1)
        if values.size < len(order):
            raise ExternalModelError(
                f"模型仅输出{values.size}个值，但输出映射配置了{len(order)}个字段：{order}。"
            )
        result = {name: values[index] for index, name in enumerate(order)}
    aliases = {
        "load_x": "load_x_mm", "load_y": "load_x_mm", "load_z": "load_z_mm",
        "load_size": "load_n", "load_magnitude": "load_n",
        "damage_x": "damage_x_mm", "damage_z": "damage_z_mm",
        "hole_x": "damage_x_mm", "hole_z": "damage_z_mm",
        "damage_size": "damage_size_mm", "hole_d": "damage_size_mm",
    }
    for source, target in aliases.items():
        if source in result and target not in result:
            result[target] = result[source]
    cleaned: dict[str, Any] = {}
    for key, value in result.items():
        arr = np.asarray(value)
        cleaned[key] = float(arr.reshape(-1)[0]) if arr.size == 1 else value
    return cleaned


def _predict_pretrain_bundle(bundle_path: Path, values: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    import joblib
    from keras.models import load_model

    with bundle_path.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    root = bundle_path.parent
    x = values.reshape(1, -1).astype(np.float32)

    cls_cfg = bundle["classifier"]
    cls_x = joblib.load(root / cls_cfg["scaler_x"]).transform(x)
    cls_model = load_model(root / cls_cfg["model"], compile=False)
    probability = float(np.clip(np.asarray(cls_model.predict(cls_x, verbose=0)).reshape(-1)[0], 0.0, 1.0))

    def branch_predict(name: str) -> dict[str, float]:
        cfg = bundle[name]
        sx = joblib.load(root / cfg["scaler_x"])
        model = load_model(root / cfg["model"], compile=False)
        raw = model.predict(sx.transform(x), verbose=0)
        outputs = cfg["outputs"]
        if isinstance(raw, dict):
            raw_list = [raw[key] for key in outputs]
        elif isinstance(raw, list):
            raw_list = raw
        else:
            arr = np.asarray(raw)
            raw_list = [arr[:, i:i + 1] for i in range(arr.shape[1])] if arr.ndim == 2 and arr.shape[1] > 1 else [arr]
        result: dict[str, float] = {}
        scaler_candidates = {
            "load_x": ["scaler_y_load_x.pkl"],
            "load_z": ["scaler_y_load_z.pkl", "scaler_y_load_y.pkl"],
            "load_size": ["scaler_y_load_size.pkl"],
            "damage_x": ["scaler_y_damage_x.pkl", "scaler_y_hoald_x.save"],
            "damage_z": ["scaler_y_damage_z.pkl", "scaler_y_hoald_y.pkl"],
            "damage_size": ["scaler_y_damage_size.pkl", "scaler_y_hoald_size.pkl"],
        }
        for output_name, raw_value in zip(outputs, raw_list):
            scaler_file = next((root / f for f in scaler_candidates[output_name] if (root / f).exists()), None)
            value = np.asarray(raw_value).reshape(-1, 1)
            if scaler_file is not None:
                value = joblib.load(scaler_file).inverse_transform(value)
            result[output_name] = float(value.reshape(-1)[0])
        return result

    intact = branch_predict("cnn_intact")
    position_scale = float(bundle.get("position_scale", config.get("transfer_learning", {}).get("position_scale", 1000.0)))
    load_scale = float(bundle.get("load_scale", config.get("transfer_learning", {}).get("load_scale", 1.0)))
    damage_size_scale = float(bundle.get("damage_size_scale", 1.0))
    output = {
        "load_x_mm": intact["load_x"] * position_scale,
        "load_z_mm": intact["load_z"] * position_scale,
        "load_n": max(intact["load_size"] * load_scale, 0.0),
        "damage_probability": probability,
        "prediction_mode": "external/BP-CNN-pretrain",
    }
    if probability >= 0.5:
        damaged = branch_predict("cnn_damage")
        output.update({
            "damage_x_mm": damaged["damage_x"] * position_scale,
            "damage_z_mm": damaged["damage_z"] * position_scale,
            "damage_size_mm": max(damaged["damage_size"] * damage_size_scale, 0.0),
        })
    return output


def predict_external_manifest(manifest: dict[str, Any], strain_values: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    model_type = manifest.get("model_type")
    model_path = Path(manifest.get("model_path_resolved") or manifest.get("model_path", ""))
    if not model_path.exists():
        raise ExternalModelError(f"外部模型文件不存在：{model_path}")
    values = np.asarray(strain_values, dtype=np.float32).reshape(-1)
    if values.size != 48:
        raise ExternalModelError(f"外部模型要求48通道输入，当前为{values.size}通道。")
    order = list(manifest.get("output_order") or ["load_x_mm", "load_z_mm", "load_n"])

    if model_type == "seven_model_group":
        from .seven_model_group import predict_seven_model_manifest
        return predict_seven_model_manifest(manifest, values, config)
    if model_type == "keras_v3_bundle":
        from .keras_v3_backend import predict_keras_v3_bundle
        return predict_keras_v3_bundle(model_path, values)
    if model_type == "pretrain_bundle":
        return _predict_pretrain_bundle(model_path, values, config)
    if model_type == "python_adapter":
        spec = importlib.util.spec_from_file_location("wing_external_model", model_path)
        if spec is None or spec.loader is None:
            raise ExternalModelError(f"无法导入Python模型：{model_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, "predict", None)
        if not callable(fn):
            raise ExternalModelError("Python模型必须定义 predict(strain_values, config) 函数。")
        return _to_dict(fn(values.copy(), config), order)

    x = values.reshape(1, -1)
    sx = _load_scaler(manifest.get("input_scaler_path_resolved"))
    if sx is not None:
        x = sx.transform(x)

    if model_type == "keras_generic":
        from keras.models import load_model
        raw = load_model(model_path, compile=False).predict(x, verbose=0)
    elif model_type == "joblib":
        import joblib
        model = joblib.load(model_path)
        if isinstance(model, dict) and "model" in model:
            estimator = model["model"]
            if sx is None and model.get("scaler") is not None:
                x = model["scaler"].transform(x)
            raw = estimator.predict(x)
        elif hasattr(model, "predict"):
            raw = model.predict(x)
        else:
            raise ExternalModelError("Joblib对象没有可用的predict方法。")
    elif model_type == "torchscript":
        import torch
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        with torch.no_grad():
            raw = model(torch.as_tensor(x, dtype=torch.float32))
        if isinstance(raw, dict):
            raw = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v) for k, v in raw.items()}
        elif hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
    else:
        raise ExternalModelError(f"未知外部模型类型：{model_type}")

    sy = _load_scaler(manifest.get("output_scaler_path_resolved"))
    if sy is not None and not isinstance(raw, dict):
        arr = np.asarray(raw)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        raw = sy.inverse_transform(arr)
    return _to_dict(raw, order)


def predict_active_external_model(base_dir: str | Path, strain_values: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    manifest = read_active_external_model(base_dir)
    if not manifest:
        raise ExternalModelError("尚未导入并应用外部模型。")
    return predict_external_manifest(manifest, strain_values, config)


def preview_external_model(
    *,
    model_type: str,
    selected_path: str | Path,
    strain_values: np.ndarray,
    config: dict[str, Any],
    input_scaler_path: str | Path | None = None,
    output_scaler_path: str | Path | None = None,
    output_order: list[str] | None = None,
) -> dict[str, Any]:
    model_path = normalize_selected_model(model_type, selected_path)
    manifest = {
        "model_type": model_type,
        "model_path_resolved": str(model_path),
        "input_scaler_path_resolved": str(Path(input_scaler_path).resolve()) if input_scaler_path else None,
        "output_scaler_path_resolved": str(Path(output_scaler_path).resolve()) if output_scaler_path else None,
        "output_order": output_order or ["load_x_mm", "load_z_mm", "load_n"],
    }
    return predict_external_manifest(manifest, strain_values, config)
