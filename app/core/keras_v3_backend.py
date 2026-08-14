from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from threading import Event
from typing import Any, Callable

import numpy as np
import pandas as pd

from .pretrain_backend import read_active_pretrain

from .model_registry import (
    activate_model,
    clear_pending_manifest,
    read_pending_manifest,
    stage_model,
    write_metadata,
)


ProgressCallback = Callable[[float, str], None]


class TrainingCancelled(RuntimeError):
    pass


class KerasV3BackendError(RuntimeError):
    pass


@dataclass
class KerasV3TrainingConfig:
    feature_count: int = 48
    latent_dim: int = 16
    batch_size: int = 256
    ae_epochs: int = 200
    reg_epochs: int = 200
    cls_epochs: int = 150
    finetune_epochs: int = 60
    ae_val: float = 0.1
    finetune_val: float = 0.2
    reg_val: float = 0.1
    cls_val: float = 0.1
    alpha_reg_nd: float = 0.7
    alpha_reg_dmg: float = 0.5
    alpha_cls: float = 0.7
    test_n_nd: int = 1000
    test_n_dmg: int = 1000
    damage_threshold: float = 0.5
    position_scale: float = 1000.0
    load_scale: float = 1.0


@dataclass
class KerasV3TrainingResult:
    model_dir: Path
    bundle_path: Path
    manifest_path: Path
    metadata_path: Path
    mesh_npz_path: Path | None
    metadata: dict[str, Any]


def _progress(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback:
        callback(float(np.clip(fraction, 0.0, 1.0)), message)


def _read_csv_flexible(path_like: str | Path) -> pd.DataFrame:
    path = Path(path_like)
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gbk", "cp936", "latin1"):
        try:
            text = raw.decode(encoding, errors="strict")
        except Exception:
            continue
        for separator in (None, ",", ";", "\t", "|"):
            try:
                if separator is None:
                    from io import StringIO
                    return pd.read_csv(StringIO(text), sep=None, engine="python")
                from io import StringIO
                return pd.read_csv(StringIO(text), sep=separator)
            except Exception:
                pass
    return pd.read_csv(path, sep=None, engine="python")


def _extract_x48_mean(csv_path: str | Path) -> np.ndarray:
    df = _read_csv_flexible(csv_path)
    expected = [str(index) for index in range(1, 49)]
    columns = [str(column) for column in df.columns]
    direct = {name: name for name in expected if name in columns}
    loose: dict[str, str] = {}
    for column in columns:
        match = re.search(r"(\d+)", column)
        if match:
            number = int(match.group(1))
            if 1 <= number <= 48 and str(number) not in direct and str(number) not in loose:
                loose[str(number)] = column
    selected = [direct.get(name, loose.get(name)) for name in expected]
    if any(column is None for column in selected):
        missing = [expected[i] for i, column in enumerate(selected) if column is None]
        raise KerasV3BackendError(f"验证数据缺少48通道列：{missing}")
    values = df[selected].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    for column_index in range(values.shape[1]):
        bad = ~np.isfinite(values[:, column_index])
        values[bad, column_index] = medians[column_index]
    return np.mean(values, axis=0).astype(np.float32)


def _stream_process(
    command: list[str],
    callback: ProgressCallback | None,
    cancel_event: Event | None,
    cwd: Path,
) -> None:
    creationflags = 0
    startupinfo = None
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
        startupinfo=startupinfo,
    )
    line_queue: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            line_queue.put(line.rstrip())
        line_queue.put(None)

    threading.Thread(target=reader, daemon=True).start()
    finished_reader = False
    emitted = 0
    while process.poll() is None or not finished_reader:
        if cancel_event is not None and cancel_event.is_set():
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
            raise TrainingCancelled("用户已取消Keras迁移模型训练；原后台模型未被替换。")
        try:
            line = line_queue.get(timeout=0.15)
        except queue.Empty:
            continue
        if line is None:
            finished_reader = True
            continue
        if line.strip():
            emitted += 1
            # 原训练脚本没有结构化进度事件；这里仅提供单调的界面进度，
            # 不改变脚本中的任何训练与保存逻辑。
            fraction = min(0.90, 0.05 + 0.0015 * emitted)
            _progress(callback, fraction, line)
    return_code = process.wait()
    if return_code != 0:
        raise KerasV3BackendError(f"train_transfer_ae_guided_kd_v3.py 运行失败，退出码 {return_code}。")


def _relative_to_model(path: Path, model_dir: Path) -> str:
    return str(path.resolve().relative_to(model_dir.resolve()))


def _collect_metrics(eval_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metrics_path in sorted(eval_dir.glob("*/metrics_summary.csv")):
        try:
            frame = pd.read_csv(metrics_path)
            if len(frame):
                row = frame.iloc[0].to_dict()
                metrics[metrics_path.parent.name] = {
                    str(key): (float(value) if isinstance(value, (int, float, np.number)) else value)
                    for key, value in row.items()
                }
        except Exception:
            continue
    return metrics


def _required_outputs(model_dir: Path) -> list[Path]:
    mt = model_dir / "models_transfer_v3"
    ms = model_dir / "models_student_v3"
    required = []
    for name in ("ae_load_x", "ae_load_y", "ae_load_size", "ae_cls"):
        required.extend(
            [
                mt / name / "nd_autoencoder.h5",
                mt / name / "nd_scaler.pkl",
            ]
        )
    required.extend(
        [
            ms / "student_load_x" / "student_load_x.h5",
            ms / "student_load_x" / "scaler_y_load_x.pkl",
            ms / "student_load_y" / "student_load_y.h5",
            ms / "student_load_y" / "scaler_y_load_y.pkl",
            ms / "student_load_size" / "student_load_size.h5",
            ms / "student_load_size" / "scaler_y_load_size.pkl",
            ms / "student_bp_cls" / "student_bp_cls.h5",
        ]
    )
    return required


def train_keras_v3(
    *,
    base_dir: str | Path,
    sim_undamaged_path: str | Path,
    sim_damaged_path: str | Path,
    actual_undamaged_path: str | Path,
    actual_damaged_path: str | Path,
    geometry_path: str | Path | None = None,
    span_axis: str = "Z",
    chord_axis: str = "X",
    geometry_unit: str = "mm",
    reverse_span: bool = False,
    training_config: KerasV3TrainingConfig,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> KerasV3TrainingResult:
    """Run the user's v3 Keras training script and stage a validated model for activation."""
    base = Path(base_dir).resolve()
    script_path = base / "backend" / "train_transfer_ae_guided_kd_v3.py"
    if not script_path.exists():
        raise KerasV3BackendError(f"未找到迁移训练脚本：{script_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = base / "models" / f"keras_aefst_v3_{timestamp}"
    transfer_dir = model_dir / "models_transfer_v3"
    student_dir = model_dir / "models_student_v3"
    eval_dir = model_dir / "eval_transfer_v3"
    pretrain_manifest = read_active_pretrain(base)
    if not pretrain_manifest or not pretrain_manifest.get("model_dir_resolved"):
        raise KerasV3BackendError(
            "尚未配置预训练模型。请先在‘模型预训练’页面完成BP分类与CNN回归预训练，"
            "或选择已有 pretrain_bundle.json。"
        )
    teacher_dir = Path(pretrain_manifest["model_dir_resolved"]).resolve()
    if not (teacher_dir / "pretrain_bundle.json").exists():
        raise KerasV3BackendError(f"当前预训练模型目录不完整：{teacher_dir}")
    model_dir.mkdir(parents=True, exist_ok=False)

    cfg = training_config
    _progress(progress_callback, 0.01, "已进入第2步：AE引导知识蒸馏迁移学习。")
    _progress(progress_callback, 0.02, f"迁移脚本：{script_path.name}")
    _progress(progress_callback, 0.03, f"教师模型：{teacher_dir.name}（BP分类 + CNN多任务回归）")

    if getattr(sys, "frozen", False):
        command = [sys.executable, "--internal-worker", "transfer"]
    else:
        command = [sys.executable, "-u", str(script_path)]
    command += [
        "--src_nd_csv", str(Path(sim_undamaged_path).resolve()),
        "--src_dmg_csv", str(Path(sim_damaged_path).resolve()),
        "--tgt_nd_csv", str(Path(actual_undamaged_path).resolve()),
        "--tgt_dmg_csv", str(Path(actual_damaged_path).resolve()),
        "--teachers_dir", str(teacher_dir),
        "--models_transfer_dir", str(transfer_dir),
        "--models_student_dir", str(student_dir),
        "--eval_dir", str(eval_dir),
        "--latent_dim", str(cfg.latent_dim),
        "--ae_epochs", str(cfg.ae_epochs),
        "--ae_batch", str(cfg.batch_size),
        "--ae_val", str(cfg.ae_val),
        "--ft_epochs", str(cfg.finetune_epochs),
        "--ft_batch", str(max(1, min(cfg.batch_size, 128))),
        "--ft_val", str(cfg.finetune_val),
        "--reg_epochs", str(cfg.reg_epochs),
        "--reg_batch", str(cfg.batch_size),
        "--reg_val", str(cfg.reg_val),
        "--cls_epochs", str(cfg.cls_epochs),
        "--cls_batch", str(cfg.batch_size),
        "--cls_val", str(cfg.cls_val),
        "--alpha_reg_nd", str(cfg.alpha_reg_nd),
        "--alpha_reg_dmg", str(cfg.alpha_reg_dmg),
        "--alpha_cls", str(cfg.alpha_cls),
        "--test_n_nd", str(cfg.test_n_nd),
        "--test_n_dmg", str(cfg.test_n_dmg),
    ]
    try:
        _stream_process(command, progress_callback, cancel_event, base)
        missing = [path for path in _required_outputs(model_dir) if not path.exists()]
        if missing:
            formatted = "\n".join(str(path) for path in missing[:12])
            raise KerasV3BackendError(f"训练进程结束，但缺少必要模型文件：\n{formatted}")

        # 机翼三维模型由独立模块管理，不写入迁移模型包。
        _progress(progress_callback, 0.92, "模型训练完成，正在生成独立后台模型包……")

        bundle = {
            "format_version": 1,
            "model_type": "Keras-AE-guided-KD-v3",
            "feature_count": cfg.feature_count,
            "damage_threshold": cfg.damage_threshold,
            "position_scale": cfg.position_scale,
            "load_scale": cfg.load_scale,
            "models_transfer_dir": _relative_to_model(transfer_dir, model_dir),
            "models_student_dir": _relative_to_model(student_dir, model_dir),
            "head_mapping": {
                "undamaged": {
                    "load_x_mm": "student_load_x",
                    "load_z_mm": "student_load_y",
                    "load_n": "student_load_size",
                },
                "damaged": {
                    "load_x_mm": "student_dmg_load_x",
                    "load_z_mm": "student_dmg_load_z",
                    "load_n": "student_dmg_load_d",
                },
            },
            "notes": [
                "训练逻辑来自用户上传的 train_transfer_ae_guided_kd_v3.py，软件包装层未修改其网络、损失函数或数据使用方式。",
                "无损第二坐标 load_y 在在线显示中映射为机翼展向 load_z_mm。",
                "有损 load_d 按该脚本从无损 load_size 分支迁移的约定映射为载荷大小。",
            ],
        }
        bundle_path = model_dir / "keras_v3_bundle.json"
        with bundle_path.open("w", encoding="utf-8") as handle:
            json.dump(bundle, handle, ensure_ascii=False, indent=2)

        metrics = _collect_metrics(eval_dir)
        metadata = {
            "format_version": 1,
            "model_name": model_dir.name,
            "model_type": "Keras-AE-guided-KD-v3",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "training_script": "backend/train_transfer_ae_guided_kd_v3.py",
            "metrics": metrics,
            "dataset_summary": {
                "simulation_undamaged": str(Path(sim_undamaged_path).resolve()),
                "simulation_damaged": str(Path(sim_damaged_path).resolve()),
                "actual_undamaged": str(Path(actual_undamaged_path).resolve()),
                "actual_damaged": str(Path(actual_damaged_path).resolve()),
            },
            "training_config": asdict(cfg),
            "pretrained_model": {
                "model_name": pretrain_manifest.get("model_name"),
                "model_type": pretrain_manifest.get("model_type"),
                "bundle_path": pretrain_manifest.get("bundle_path_resolved"),
                "model_dir": pretrain_manifest.get("model_dir_resolved"),
            },
            "limitations": [
                "该后台严格调用用户给定v3脚本；脚本当前的目标域测试抽样来自已参与训练的数据。",
                "迁移学习使用当前已激活的BP-CNN预训练模型作为教师模型。",
                "48通道顺序必须与训练数据完全一致。",
            ],
        }
        metadata_path = write_metadata(model_dir / "training_report.json", metadata)

        _progress(progress_callback, 0.96, "正在加载新Keras模型并执行一次预测验证……")
        validation_input = _extract_x48_mean(actual_undamaged_path)
        prediction = predict_keras_v3_bundle(bundle_path, validation_input)
        required_keys = ("load_x_mm", "load_z_mm", "load_n", "damage_probability")
        if any(key not in prediction for key in required_keys):
            raise KerasV3BackendError("新模型验证失败：预测输出字段不完整。")
        if not all(np.isfinite(float(prediction[key])) for key in required_keys):
            raise KerasV3BackendError("新模型验证失败：预测结果包含NaN或无穷大。")

        manifest_path = stage_model(
            base,
            bundle_path,
            model_dir,
            metadata,
            mesh_npz_path=None,
            metadata_path=metadata_path,
        )
        _progress(progress_callback, 1.0, "迁移学习完成：模型已生成并通过验证，等待更新后台模型。")
        return KerasV3TrainingResult(
            model_dir=model_dir,
            bundle_path=bundle_path,
            manifest_path=manifest_path,
            metadata_path=metadata_path,
            mesh_npz_path=None,
            metadata=metadata,
        )
    except Exception:
        # 保留失败目录和日志，便于定位；训练阶段只写入 pending_model.json，
        # 不会替换当前 active_model.json。
        raise


def activate_pending_keras_v3(base_dir: str | Path) -> Path:
    """Activate the latest validated pending model as the software backend model."""
    base = Path(base_dir).resolve()
    pending = read_pending_manifest(base)
    if not pending:
        raise KerasV3BackendError("没有可更新的待生效模型，请先完成迁移学习。")

    bundle_path = Path(pending.get("bundle_path_resolved", ""))
    model_dir = Path(pending.get("model_dir_resolved", ""))
    metadata_value = pending.get("metadata_path_resolved")
    metadata: dict[str, Any]
    if metadata_value and Path(metadata_value).exists():
        with Path(metadata_value).open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    else:
        metadata = {
            "model_name": pending.get("model_name", model_dir.name),
            "model_type": pending.get("model_type", "Keras-AE-guided-KD-v3"),
            "training_script": pending.get("training_script"),
            "metrics": pending.get("metrics", {}),
            "dataset_summary": pending.get("dataset_summary", {}),
            "training_config": pending.get("training_config", {}),
        }

    if not bundle_path.exists():
        raise KerasV3BackendError(f"待更新模型包不存在：{bundle_path}")
    # 在写入 active_model.json 前再次加载模型包，避免更新损坏或移动后的模型。
    validation_source = pending.get("dataset_summary", {}).get("actual_undamaged")
    if validation_source and Path(validation_source).exists():
        validation_input = _extract_x48_mean(validation_source)
        prediction = predict_keras_v3_bundle(bundle_path, validation_input)
        required_keys = ("load_x_mm", "load_z_mm", "load_n", "damage_probability")
        if any(key not in prediction for key in required_keys):
            raise KerasV3BackendError("后台更新失败：模型预测输出字段不完整。")
        if not all(np.isfinite(float(prediction[key])) for key in required_keys):
            raise KerasV3BackendError("后台更新失败：模型预测结果包含NaN或无穷大。")

    manifest_path = activate_model(
        base,
        bundle_path,
        model_dir,
        metadata,
        mesh_npz_path=None,
    )
    clear_pending_manifest(base)
    return manifest_path


def activate_existing_keras_v3(
    base_dir: str | Path,
    selected_path: str | Path,
) -> Path:
    """Activate an existing Keras v3 model directory without running migration training.

    ``selected_path`` may be either a model directory or its
    ``keras_v3_bundle.json`` file.  For older output directories that contain
    ``models_transfer_v3`` and ``models_student_v3`` but no bundle JSON, a
    compatible bundle description is created automatically.
    """
    base = Path(base_dir).resolve()
    selected = Path(selected_path).expanduser().resolve()
    if not selected.exists():
        raise KerasV3BackendError(f"所选模型路径不存在：{selected}")
    if selected.is_file() and selected.name != "keras_v3_bundle.json":
        raise KerasV3BackendError("请选择 keras_v3_bundle.json，或选择模型输出目录。")

    # 允许选择：模型根目录、models_transfer_v3、models_student_v3，
    # 以及旧命名 models_transfer / models_student。
    if selected.is_file():
        model_dir = selected.parent
        bundle_path = selected
    else:
        if selected.name in {"models_transfer_v3", "models_student_v3", "models_transfer", "models_student"}:
            model_dir = selected.parent
        else:
            model_dir = selected
        bundle_path = model_dir / "keras_v3_bundle.json"

    if not bundle_path.exists():
        candidates = [
            (model_dir / "models_transfer_v3", model_dir / "models_student_v3"),
            (model_dir / "models_transfer", model_dir / "models_student"),
        ]
        transfer_dir = student_dir = None
        for transfer_candidate, student_candidate in candidates:
            if transfer_candidate.is_dir() and student_candidate.is_dir():
                transfer_dir, student_dir = transfer_candidate, student_candidate
                break
        if transfer_dir is None or student_dir is None:
            raise KerasV3BackendError(
                "所选位置不是可用的迁移模型目录。请选择以下任一种：\n"
                "1. 包含 keras_v3_bundle.json 的目录；\n"
                "2. 同时包含 models_transfer_v3 和 models_student_v3 的目录；\n"
                "3. 直接选择 models_transfer_v3 或 models_student_v3 文件夹。\n\n"
                "三维机翼文件和传感器坐标不属于后台模型，请分别在‘机翼三维模型’和‘传感器布置’页面设置。"
            )
        bundle = {
            "format_version": 1,
            "model_type": "Keras-AE-guided-KD-v3",
            "feature_count": 48,
            "damage_threshold": 0.5,
            "position_scale": 1000.0,
            "load_scale": 1.0,
            "models_transfer_dir": str(transfer_dir.relative_to(model_dir)),
            "models_student_dir": str(student_dir.relative_to(model_dir)),
            "head_mapping": {
                "undamaged": {
                    "load_x_mm": "student_load_x",
                    "load_z_mm": "student_load_y",
                    "load_n": "student_load_size",
                },
                "damaged": {
                    "load_x_mm": "student_dmg_load_x",
                    "load_z_mm": "student_dmg_load_z",
                    "load_n": "student_dmg_load_d",
                },
            },
            "notes": ["由软件在手动更新后台模型时生成的兼容模型包。"],
        }
        with bundle_path.open("w", encoding="utf-8") as handle:
            json.dump(bundle, handle, ensure_ascii=False, indent=2)

    # 完整加载一次，确认AE、分类器、回归器和归一化器可读取。
    _load_bundle(bundle_path)

    metadata_path = model_dir / "training_report.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception as exc:
            raise KerasV3BackendError(f"无法读取模型说明文件：{metadata_path}") from exc
    else:
        metadata = {
            "model_name": model_dir.name,
            "model_type": "Keras-AE-guided-KD-v3",
            "training_script": None,
            "metrics": {},
            "dataset_summary": {},
            "training_config": {},
        }

    metadata.setdefault("model_name", model_dir.name)
    metadata.setdefault("model_type", "Keras-AE-guided-KD-v3")
    return activate_model(
        base,
        bundle_path,
        model_dir,
        metadata,
        mesh_npz_path=None,
    )


def train_and_activate_keras_v3(**kwargs) -> KerasV3TrainingResult:
    """Backward-compatible combined operation; new GUI uses separate buttons."""
    result = train_keras_v3(**kwargs)
    active_path = activate_pending_keras_v3(kwargs["base_dir"])
    result.manifest_path = active_path
    return result


_MODEL_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _encoder_from_ae(ae_path: Path):
    from keras.models import Model, load_model

    autoencoder = load_model(ae_path, compile=False)
    try:
        encoder = Model(autoencoder.input, autoencoder.get_layer("latent").output, name="Encoder")
    except Exception:
        middle = max(1, len(autoencoder.layers) // 2 - 1)
        encoder = Model(autoencoder.input, autoencoder.layers[middle].output, name="Encoder")
    return encoder


def _load_regression(student_dir: Path, name: str):
    import joblib
    from keras.models import load_model

    folder = student_dir / f"student_{name}"
    model_path = folder / f"student_{name}.h5"
    scaler_path = folder / f"scaler_y_{name}.pkl"
    if not model_path.exists() or not scaler_path.exists():
        return None
    return load_model(model_path, compile=False), joblib.load(scaler_path)


def _load_bundle(bundle_path: Path) -> dict[str, Any]:
    import joblib
    from keras.models import load_model

    key = str(bundle_path.resolve())
    modified = bundle_path.stat().st_mtime
    cached = _MODEL_CACHE.get(key)
    if cached and cached[0] == modified:
        return cached[1]

    with bundle_path.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    if bundle.get("model_type") != "Keras-AE-guided-KD-v3":
        raise KerasV3BackendError("所选JSON不是Keras AE-guided KD v3模型包。")
    model_dir = bundle_path.parent
    transfer_dir = model_dir / bundle["models_transfer_dir"]
    student_dir = model_dir / bundle["models_student_dir"]

    loaded: dict[str, Any] = {"bundle": bundle}
    for ae_name in ("ae_cls", "ae_load_x", "ae_load_y", "ae_load_size"):
        folder = transfer_dir / ae_name
        loaded[ae_name] = {
            "scaler": joblib.load(folder / "nd_scaler.pkl"),
            "encoder": _encoder_from_ae(folder / "nd_autoencoder.h5"),
        }
    for ae_name in ("ae_load_x_ft_dmg", "ae_load_y_ft_dmg", "ae_load_size_ft_dmg"):
        folder = transfer_dir / ae_name
        if (folder / "nd_scaler.pkl").exists() and (folder / "nd_autoencoder.h5").exists():
            loaded[ae_name] = {
                "scaler": joblib.load(folder / "nd_scaler.pkl"),
                "encoder": _encoder_from_ae(folder / "nd_autoencoder.h5"),
            }

    cls_path = student_dir / "student_bp_cls" / "student_bp_cls.h5"
    loaded["classifier"] = load_model(cls_path, compile=False)
    for name in ("load_x", "load_y", "load_size", "dmg_load_x", "dmg_load_z", "dmg_load_d"):
        loaded[name] = _load_regression(student_dir, name)
    _MODEL_CACHE[key] = (modified, loaded)
    return loaded


def _predict_head(loaded: dict[str, Any], ae_name: str, reg_name: str, x48: np.ndarray) -> float | None:
    pack = loaded.get(ae_name)
    reg_pack = loaded.get(reg_name)
    if pack is None or reg_pack is None:
        return None
    scaler_x = pack["scaler"]
    encoder = pack["encoder"]
    model, scaler_y = reg_pack
    x_scaled = scaler_x.transform(x48)
    latent = encoder.predict(x_scaled, verbose=0)
    y_scaled = model.predict(latent, verbose=0)
    return float(scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).reshape(-1)[0])


def predict_keras_v3_bundle(bundle_path: str | Path, strain_values: np.ndarray) -> dict[str, float]:
    path = Path(bundle_path)
    if not path.exists():
        raise KerasV3BackendError(f"未找到Keras模型包：{path}")
    loaded = _load_bundle(path)
    bundle = loaded["bundle"]
    values = np.asarray(strain_values, dtype=np.float32).reshape(-1)
    feature_count = int(bundle.get("feature_count", 48))
    if values.size != feature_count:
        raise KerasV3BackendError(f"模型需要{feature_count}个通道，当前输入{values.size}个。")
    x48 = values.reshape(1, -1)

    cls_pack = loaded["ae_cls"]
    latent_cls = cls_pack["encoder"].predict(cls_pack["scaler"].transform(x48), verbose=0)
    probability = float(np.clip(loaded["classifier"].predict(latent_cls, verbose=0).reshape(-1)[0], 0.0, 1.0))
    damaged = probability >= float(bundle.get("damage_threshold", 0.5))

    if damaged and all(loaded.get(name) is not None for name in ("dmg_load_x", "dmg_load_z", "dmg_load_d")):
        load_x = _predict_head(loaded, "ae_load_x_ft_dmg", "dmg_load_x", x48)
        load_z = _predict_head(loaded, "ae_load_y_ft_dmg", "dmg_load_z", x48)
        load_value = _predict_head(loaded, "ae_load_size_ft_dmg", "dmg_load_d", x48)
        branch = "damaged"
    else:
        load_x = _predict_head(loaded, "ae_load_x", "load_x", x48)
        load_z = _predict_head(loaded, "ae_load_y", "load_y", x48)
        load_value = _predict_head(loaded, "ae_load_size", "load_size", x48)
        branch = "undamaged"

    if load_x is None or load_z is None or load_value is None:
        raise KerasV3BackendError("Keras后台模型缺少当前分支所需的回归模型。")
    position_scale = float(bundle.get("position_scale", 1000.0))
    load_scale = float(bundle.get("load_scale", 1.0))
    return {
        "load_x_mm": float(load_x * position_scale),
        "load_z_mm": float(load_z * position_scale),
        "load_n": float(max(load_value * load_scale, 0.0)),
        "damage_probability": probability,
        "prediction_mode": f"Keras-AE-guided-KD-v3/{branch}",
    }
