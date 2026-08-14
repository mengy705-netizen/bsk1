from __future__ import annotations

from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np


class SevenModelGroupError(RuntimeError):
    pass


SLOT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "classifier": {
        "label": "损伤分类（BP）",
        "output": "damage_probability",
        "group": "classification",
        "keywords": ["bp_classifier", "classifier", "damage_cls", "bp_cls", "student_bp_cls"],
    },
    "intact_load_x": {
        "label": "无损—载荷位置 x",
        "output": "load_x_mm",
        "group": "intact",
        "keywords": ["intact_load_x", "load_x", "cnn_load_x"],
    },
    "intact_load_z": {
        "label": "无损—载荷位置 z",
        "output": "load_z_mm",
        "group": "intact",
        "keywords": ["intact_load_z", "load_z", "load_y", "cnn_load_z", "cnn_load_y"],
    },
    "intact_load_size": {
        "label": "无损—载荷大小",
        "output": "load_n",
        "group": "intact",
        "keywords": ["intact_load_size", "load_size", "cnn_load_size"],
    },
    "damage_x": {
        "label": "有损—损伤位置 x",
        "output": "damage_x_mm",
        "group": "damage",
        "keywords": ["damage_x", "dmg_load_x", "hole_x", "hold_x", "hoald_x", "cnn_hold_x"],
    },
    "damage_z": {
        "label": "有损—损伤位置 z",
        "output": "damage_z_mm",
        "group": "damage",
        "keywords": ["damage_z", "dmg_load_z", "hole_z", "hold_z", "hold_y", "hoald_y", "cnn_hold_y"],
    },
    "damage_size": {
        "label": "有损—损伤尺寸",
        "output": "damage_size_mm",
        "group": "damage",
        "keywords": ["damage_size", "dmg_load_d", "hole_d", "hole_size", "hold_size", "hoald_size", "cnn_hold_size"],
    },
}

MODEL_SUFFIXES = {".keras", ".h5", ".hdf5", ".joblib", ".pkl", ".pickle", ".save", ".pt", ".pth", ".ts", ".torchscript", ".py"}
SCALER_SUFFIXES = {".joblib", ".pkl", ".pickle", ".save"}


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


def _detect_model_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".keras", ".h5", ".hdf5"}:
        return "keras"
    if suffix in {".joblib", ".pkl", ".pickle", ".save"}:
        return "joblib"
    if suffix in {".pt", ".pth", ".ts", ".torchscript"}:
        return "torchscript"
    if suffix == ".py":
        return "python_adapter"
    raise SevenModelGroupError(f"无法识别模型格式：{path.name}")


def _load_scaler(path: str | Path | None):
    if not path:
        return None
    import joblib

    return joblib.load(path)


def _first_scalar(raw: Any, *, classifier: bool = False) -> float:
    if isinstance(raw, dict):
        preferred = (
            ["damage_probability", "probability", "prob", "y_prob", "output"]
            if classifier
            else ["value", "prediction", "pred", "output", "y"]
        )
        for key in preferred:
            if key in raw:
                raw = raw[key]
                break
        else:
            if len(raw) != 1:
                raise SevenModelGroupError("模型返回字典，但无法确定应使用哪个输出字段。")
            raw = next(iter(raw.values()))
    if isinstance(raw, (list, tuple)):
        if len(raw) == 1:
            raw = raw[0]
        else:
            raw = np.asarray([np.asarray(item).reshape(-1)[0] for item in raw])
    arr = np.asarray(raw, dtype=float)
    if classifier and arr.ndim >= 2 and arr.shape[-1] == 2:
        value = float(arr.reshape(-1, 2)[0, 1])
    elif classifier and arr.size == 2:
        value = float(arr.reshape(-1)[1])
    else:
        if arr.size < 1:
            raise SevenModelGroupError("模型没有返回数值。")
        value = float(arr.reshape(-1)[0])
    if not math.isfinite(value):
        raise SevenModelGroupError("模型输出不是有限数值。")
    return value


def _predict_python(path: Path, values: np.ndarray, config: dict[str, Any], slot: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"wing_external_{slot}", path)
    if spec is None or spec.loader is None:
        raise SevenModelGroupError(f"无法导入Python模型：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "predict", None)
    if not callable(fn):
        raise SevenModelGroupError(f"Python模型必须定义 predict 函数：{path.name}")
    try:
        return fn(values.copy(), config)
    except TypeError:
        return fn(values.copy())


def _predict_one(slot: str, slot_cfg: dict[str, Any], values: np.ndarray, config: dict[str, Any]) -> float:
    model_path = Path(slot_cfg.get("model_path_resolved") or slot_cfg.get("model_path", ""))
    if not model_path.exists():
        raise SevenModelGroupError(f"{SLOT_DEFINITIONS[slot]['label']}模型不存在：{model_path}")
    x = values.reshape(1, -1).astype(np.float32)
    input_scaler_path = slot_cfg.get("input_scaler_path_resolved") or slot_cfg.get("input_scaler_path")
    output_scaler_path = slot_cfg.get("output_scaler_path_resolved") or slot_cfg.get("output_scaler_path")
    sx = _load_scaler(input_scaler_path)
    if sx is not None:
        if not hasattr(sx, "transform"):
            raise SevenModelGroupError(f"{SLOT_DEFINITIONS[slot]['label']}输入归一化文件没有 transform 方法。")
        x = np.asarray(sx.transform(x), dtype=np.float32)

    model_type = slot_cfg.get("model_format") or _detect_model_type(model_path)
    if model_type == "keras":
        from .legacy_keras_loader import load_model_compat, prepare_keras_input

        prefer_conv1d = slot != "classifier"
        fallback_shape = (48, 1) if prefer_conv1d else (48,)
        model = load_model_compat(model_path, input_shape=fallback_shape)
        x_model = prepare_keras_input(model, x, prefer_conv1d=prefer_conv1d)
        raw = model.predict(x_model, verbose=0)
    elif model_type == "joblib":
        import joblib

        obj = joblib.load(model_path)
        if isinstance(obj, dict) and "model" in obj:
            estimator = obj["model"]
            if sx is None and obj.get("scaler") is not None:
                x = obj["scaler"].transform(x)
            raw = estimator.predict(x)
        elif hasattr(obj, "predict"):
            raw = obj.predict(x)
        elif callable(obj):
            raw = obj(x)
        else:
            raise SevenModelGroupError(f"Joblib模型没有可用的predict方法：{model_path.name}")
    elif model_type == "torchscript":
        import torch

        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        with torch.no_grad():
            raw = model(torch.as_tensor(x, dtype=torch.float32))
        if isinstance(raw, dict):
            raw = {str(k): (v.detach().cpu().numpy() if hasattr(v, "detach") else v) for k, v in raw.items()}
        elif hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
    elif model_type == "python_adapter":
        raw = _predict_python(model_path, values, config, slot)
    else:
        raise SevenModelGroupError(f"未知模型格式：{model_type}")

    classifier = slot == "classifier"
    value = _first_scalar(raw, classifier=classifier)
    if classifier:
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(min(value, 60.0), -60.0)))
        return float(np.clip(value, 0.0, 1.0))

    sy = _load_scaler(output_scaler_path)
    if sy is not None:
        if not hasattr(sy, "inverse_transform"):
            raise SevenModelGroupError(f"{SLOT_DEFINITIONS[slot]['label']}输出归一化文件没有 inverse_transform 方法。")
        value = float(np.asarray(sy.inverse_transform(np.array([[value]], dtype=np.float32))).reshape(-1)[0])
    multiplier = float(slot_cfg.get("multiplier", 1.0))
    return float(value * multiplier)


def _validate_slots(slots: dict[str, dict[str, Any]]) -> None:
    missing = []
    for slot in SLOT_DEFINITIONS:
        cfg = slots.get(slot) or {}
        model_path = cfg.get("model_path_resolved") or cfg.get("model_path")
        if not model_path:
            missing.append(SLOT_DEFINITIONS[slot]["label"])
            continue
        path = Path(model_path)
        if not path.exists() or not path.is_file():
            raise SevenModelGroupError(f"{SLOT_DEFINITIONS[slot]['label']}模型文件不存在：{path}")
        _detect_model_type(path)
        for key, label in (("input_scaler_path_resolved", "输入归一化"), ("output_scaler_path_resolved", "输出归一化")):
            scaler = cfg.get(key) or cfg.get(key.replace("_resolved", ""))
            if scaler and not Path(scaler).exists():
                raise SevenModelGroupError(f"{SLOT_DEFINITIONS[slot]['label']}的{label}文件不存在：{scaler}")
    if missing:
        raise SevenModelGroupError("尚未选择以下模型：" + "、".join(missing))


def predict_seven_model_manifest(manifest: dict[str, Any], strain_values: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(strain_values, dtype=np.float32).reshape(-1)
    if values.size != 48:
        raise SevenModelGroupError(f"七模型组合要求48通道输入，当前为{values.size}通道。")
    slots = manifest.get("slots") or {}
    _validate_slots(slots)
    probability = _predict_one("classifier", slots["classifier"], values, config)
    threshold = float(manifest.get("damage_threshold", 0.5))
    damaged = probability >= threshold

    if damaged:
        damage_x = _predict_one("damage_x", slots["damage_x"], values, config)
        damage_z = _predict_one("damage_z", slots["damage_z"], values, config)
        damage_size = max(_predict_one("damage_size", slots["damage_size"], values, config), 0.0)
        return {
            "damage_probability": probability,
            "is_damage": True,
            "prediction_branch": "damaged",
            "damage_x_mm": damage_x,
            "damage_z_mm": damage_z,
            "damage_size_mm": damage_size,
            # 为现有重构模块提供兼容坐标；界面不会把它显示为载荷。
            "load_x_mm": damage_x,
            "load_z_mm": damage_z,
            "load_n": 0.0,
            "prediction_mode": "external/seven-model/damaged",
        }

    load_x = _predict_one("intact_load_x", slots["intact_load_x"], values, config)
    load_z = _predict_one("intact_load_z", slots["intact_load_z"], values, config)
    load_n = max(_predict_one("intact_load_size", slots["intact_load_size"], values, config), 0.0)
    return {
        "damage_probability": probability,
        "is_damage": False,
        "prediction_branch": "intact",
        "load_x_mm": load_x,
        "load_z_mm": load_z,
        "load_n": load_n,
        "prediction_mode": "external/seven-model/intact",
    }


def inspect_seven_model_group(
    *,
    slots: dict[str, dict[str, Any]],
    strain_values: np.ndarray,
    config: dict[str, Any],
    damage_threshold: float = 0.5,
) -> dict[str, Any]:
    values = np.asarray(strain_values, dtype=np.float32).reshape(-1)
    if values.size != 48:
        raise SevenModelGroupError(f"七模型组合要求48通道输入，当前为{values.size}通道。")
    normalized: dict[str, dict[str, Any]] = {}
    for slot in SLOT_DEFINITIONS:
        cfg = dict(slots.get(slot) or {})
        model = cfg.get("model_path")
        if model:
            model_path = Path(model).expanduser().resolve()
            cfg["model_path_resolved"] = str(model_path)
            cfg["model_format"] = _detect_model_type(model_path)
        for key in ("input_scaler_path", "output_scaler_path"):
            value = cfg.get(key)
            cfg[key + "_resolved"] = str(Path(value).expanduser().resolve()) if value else None
        normalized[slot] = cfg
    _validate_slots(normalized)
    outputs: dict[str, Any] = {}
    for slot, definition in SLOT_DEFINITIONS.items():
        outputs[definition["output"]] = _predict_one(slot, normalized[slot], values, config)
    probability = float(outputs["damage_probability"])
    outputs["is_damage"] = probability >= float(damage_threshold)
    outputs["prediction_branch"] = "damaged" if outputs["is_damage"] else "intact"
    outputs["prediction_mode"] = "external/seven-model/full-check"
    return outputs


def preview_seven_model_group(
    *,
    slots: dict[str, dict[str, Any]],
    strain_values: np.ndarray,
    config: dict[str, Any],
    damage_threshold: float = 0.5,
) -> dict[str, Any]:
    normalized: dict[str, dict[str, Any]] = {}
    for slot, definition in SLOT_DEFINITIONS.items():
        cfg = dict(slots.get(slot) or {})
        model = cfg.get("model_path")
        if model:
            model_path = Path(model).expanduser().resolve()
            cfg["model_path_resolved"] = str(model_path)
            cfg["model_format"] = _detect_model_type(model_path)
        for key in ("input_scaler_path", "output_scaler_path"):
            value = cfg.get(key)
            cfg[key + "_resolved"] = str(Path(value).expanduser().resolve()) if value else None
        normalized[slot] = cfg
    manifest = {
        "model_type": "seven_model_group",
        "damage_threshold": float(damage_threshold),
        "slots": normalized,
    }
    return predict_seven_model_manifest(manifest, strain_values, config)


def install_seven_model_group(
    base_dir: str | Path,
    *,
    slots: dict[str, dict[str, Any]],
    damage_threshold: float = 0.5,
    copy_into_software: bool = True,
) -> Path:
    base = Path(base_dir).resolve()
    absolute: dict[str, dict[str, Any]] = {}
    for slot in SLOT_DEFINITIONS:
        cfg = dict(slots.get(slot) or {})
        model_value = cfg.get("model_path")
        cfg["model_path_resolved"] = str(Path(model_value).expanduser().resolve()) if model_value else None
        for key in ("input_scaler_path", "output_scaler_path"):
            value = cfg.get(key)
            cfg[key + "_resolved"] = str(Path(value).expanduser().resolve()) if value else None
        if cfg.get("model_path_resolved"):
            cfg["model_format"] = _detect_model_type(Path(cfg["model_path_resolved"]))
        absolute[slot] = cfg
    _validate_slots(absolute)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import_root: Path | None = None
    if copy_into_software:
        import_root = base / "external_models" / f"seven_models_{timestamp}"
        import_root.mkdir(parents=True, exist_ok=False)
        models_dir = import_root / "models"
        scalers_dir = import_root / "scalers"
        models_dir.mkdir()
        scalers_dir.mkdir()
        for slot, cfg in absolute.items():
            source_model = Path(cfg["model_path_resolved"])
            model_target = models_dir / f"{slot}{source_model.suffix.lower()}"
            shutil.copy2(source_model, model_target)
            cfg["model_path_resolved"] = str(model_target)
            for key, prefix in (("input_scaler_path_resolved", "input"), ("output_scaler_path_resolved", "output")):
                source_value = cfg.get(key)
                if not source_value:
                    continue
                source = Path(source_value)
                target = scalers_dir / f"{slot}_{prefix}{source.suffix.lower()}"
                shutil.copy2(source, target)
                cfg[key] = str(target)

    stored_slots: dict[str, dict[str, Any]] = {}
    for slot, cfg in absolute.items():
        stored_slots[slot] = {
            "label": SLOT_DEFINITIONS[slot]["label"],
            "model_format": cfg.get("model_format"),
            "model_path": _portable(base, Path(cfg["model_path_resolved"])),
            "input_scaler_path": _portable(base, Path(cfg["input_scaler_path_resolved"])) if cfg.get("input_scaler_path_resolved") else None,
            "output_scaler_path": _portable(base, Path(cfg["output_scaler_path_resolved"])) if cfg.get("output_scaler_path_resolved") else None,
            "multiplier": float(cfg.get("multiplier", 1.0)),
        }

    model_root = import_root or Path(absolute["classifier"]["model_path_resolved"]).parent
    payload = {
        "format_version": 2,
        "status": "active",
        "model_name": f"七模型组合_{timestamp}",
        "model_type": "seven_model_group",
        "model_type_label": "BP分类 + 3个无损回归 + 3个损伤回归",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_path": _portable(base, model_root),
        "damage_threshold": float(damage_threshold),
        "copied_into_software": bool(copy_into_software),
        "slots": stored_slots,
    }
    manifest_path = base / "external_models" / "active_external_model.json"
    _atomic_write_json(manifest_path, payload)
    return manifest_path


def resolve_seven_model_slots(base_dir: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    base = Path(base_dir).resolve()
    slots = payload.get("slots")
    if not isinstance(slots, dict):
        return payload
    for slot, cfg in slots.items():
        if not isinstance(cfg, dict):
            continue
        for key in ("model_path", "input_scaler_path", "output_scaler_path"):
            cfg[key + "_resolved"] = _resolve(base, cfg.get(key))
    return payload


def _score_filename(name: str, slot: str, *, scaler: bool = False, output_scaler: bool = False) -> int:
    lower = name.lower()
    definition = SLOT_DEFINITIONS[slot]
    score = 0
    for index, keyword in enumerate(definition["keywords"]):
        if keyword in lower:
            score += 100 - index
    if slot.startswith("intact_") and any(token in lower for token in ("damage", "dmg", "hole", "hold", "hoald")):
        score -= 150
    if slot.startswith("damage_") and not any(token in lower for token in ("damage", "dmg", "hole", "hold", "hoald")):
        score -= 20
    if slot == "classifier" and any(token in lower for token in ("load_x", "load_z", "load_y", "load_size", "damage_x", "damage_z", "damage_size")):
        score -= 80
    if scaler:
        if "scaler" in lower or "scale" in lower or "normal" in lower:
            score += 35
        else:
            score -= 50
        if output_scaler:
            if any(token in lower for token in ("scaler_y", "output", "target")):
                score += 30
            if any(token in lower for token in ("scaler_x", "input")):
                score -= 30
        else:
            if any(token in lower for token in ("scaler_x", "input")):
                score += 30
            if any(token in lower for token in ("scaler_y", "output", "target")):
                score -= 20
    return score


def auto_match_seven_model_directory(directory: str | Path) -> dict[str, dict[str, str | None]]:
    root = Path(directory).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SevenModelGroupError(f"模型目录不存在：{root}")
    files = [path for path in root.rglob("*") if path.is_file()]
    model_files = [path for path in files if path.suffix.lower() in MODEL_SUFFIXES and not (path.suffix.lower() in SCALER_SUFFIXES and "scaler" in path.name.lower())]
    scaler_files = [path for path in files if path.suffix.lower() in SCALER_SUFFIXES and ("scaler" in path.name.lower() or "scale" in path.name.lower() or "normal" in path.name.lower())]
    result: dict[str, dict[str, str | None]] = {}
    used_models: set[Path] = set()
    for slot in SLOT_DEFINITIONS:
        ranked = sorted((( _score_filename(path.name, slot), path) for path in model_files if path not in used_models), reverse=True, key=lambda item: item[0])
        selected = ranked[0][1] if ranked and ranked[0][0] > 0 else None
        if selected is not None:
            used_models.add(selected)
        in_ranked = sorted(((_score_filename(path.name, slot, scaler=True, output_scaler=False), path) for path in scaler_files), reverse=True, key=lambda item: item[0])
        out_ranked = sorted(((_score_filename(path.name, slot, scaler=True, output_scaler=True), path) for path in scaler_files), reverse=True, key=lambda item: item[0])
        input_scaler = in_ranked[0][1] if in_ranked and in_ranked[0][0] > 40 else None
        output_scaler = None if slot == "classifier" else (out_ranked[0][1] if out_ranked and out_ranked[0][0] > 40 else None)
        result[slot] = {
            "model_path": str(selected) if selected else None,
            "input_scaler_path": str(input_scaler) if input_scaler else None,
            "output_scaler_path": str(output_scaler) if output_scaler else None,
        }
    return result
