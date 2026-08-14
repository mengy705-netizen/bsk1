from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from threading import Event
from typing import Any, Callable

import numpy as np

ProgressCallback = Callable[[float, str], None]


class PretrainCancelled(RuntimeError):
    pass


class PretrainBackendError(RuntimeError):
    pass


@dataclass
class PretrainConfig:
    feature_count: int = 48
    bp_trials: int = 10
    bp_epochs: int = 200
    cnn_epochs: int = 250
    batch_size: int = 128


@dataclass
class PretrainResult:
    model_dir: Path
    bundle_path: Path
    manifest_path: Path
    metadata: dict[str, Any]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def active_pretrain_manifest_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "pretrained_models" / "active_pretrain.json"


def read_active_pretrain(base_dir: str | Path) -> dict[str, Any] | None:
    base = Path(base_dir).resolve()
    path = active_pretrain_manifest_path(base)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise PretrainBackendError(f"无法读取当前预训练模型清单：{path}") from exc
    for key, resolved_key in (("bundle_path", "bundle_path_resolved"), ("model_dir", "model_dir_resolved")):
        value = payload.get(key)
        if value:
            resolved = Path(value)
            if not resolved.is_absolute():
                resolved = (base / resolved).resolve()
            payload[resolved_key] = str(resolved)
    return payload


def _portable(base: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_bundle(bundle_path: Path) -> dict[str, Any]:
    if not bundle_path.exists():
        raise PretrainBackendError(f"预训练模型包不存在：{bundle_path}")
    try:
        with bundle_path.open("r", encoding="utf-8") as handle:
            bundle = json.load(handle)
    except Exception as exc:
        raise PretrainBackendError(f"无法读取预训练模型包：{bundle_path}") from exc
    if bundle.get("model_type") != "BP-CNN-Wing-Pretrain-v1":
        raise PretrainBackendError("所选文件不是BP-CNN机翼预训练模型包。")
    root = bundle_path.parent
    required = [
        root / "bp_classifier.h5",
        root / "scaler_X_bp.pkl",
        root / "cnn_intact_multitask.keras",
        root / "cnn_damage_multitask.keras",
        root / "cnn_load_x.h5",
        root / "cnn_load_y.h5",
        root / "cnn_load_size.h5",
        root / "cnn_hold_x.h5",
        root / "cnn_hold_y.h5",
        root / "cnn_hold_size.h5",
    ]
    missing = [str(path.name) for path in required if not path.exists()]
    if missing:
        raise PretrainBackendError("预训练模型包不完整，缺少：" + "、".join(missing))
    return bundle


def activate_existing_pretrain(base_dir: str | Path, selected_path: str | Path) -> Path:
    base = Path(base_dir).resolve()
    selected = Path(selected_path).expanduser().resolve()
    if selected.is_dir():
        bundle_path = selected / "pretrain_bundle.json"
    else:
        bundle_path = selected
    bundle = _validate_bundle(bundle_path)
    model_dir = bundle_path.parent
    payload = {
        "format_version": 1,
        "status": "active",
        "model_name": model_dir.name,
        "model_type": bundle.get("model_type"),
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bundle_path": _portable(base, bundle_path),
        "model_dir": _portable(base, model_dir),
        "metrics": bundle.get("metrics", {}),
        "source_columns": bundle.get("source_columns", {}),
    }
    path = active_pretrain_manifest_path(base)
    _atomic_write_json(path, payload)
    return path


def _stream_process(command: list[str], callback: ProgressCallback | None, cancel_event: Event | None, cwd: Path) -> None:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0
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
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line.rstrip())
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    finished = False
    count = 0
    while process.poll() is None or not finished:
        if cancel_event is not None and cancel_event.is_set():
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
            raise PretrainCancelled("用户已取消预训练。")
        try:
            line = lines.get(timeout=0.15)
        except queue.Empty:
            continue
        if line is None:
            finished = True
            continue
        if line.strip():
            count += 1
            if callback:
                callback(min(0.94, 0.03 + count * 0.0015), line)
    code = process.wait()
    if code != 0:
        raise PretrainBackendError(f"预训练脚本运行失败，退出码 {code}。")


def train_pretrained_models(
    *,
    base_dir: str | Path,
    sim_undamaged_path: str | Path,
    sim_damaged_path: str | Path,
    training_config: PretrainConfig,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> PretrainResult:
    base = Path(base_dir).resolve()
    script = base / "backend" / "train_pretrain_bp_cnn.py"
    if not script.exists():
        raise PretrainBackendError(f"未找到预训练脚本：{script}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = base / "pretrained_models" / f"bp_cnn_pretrain_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=False)
    cfg = training_config
    if progress_callback:
        progress_callback(0.01, "预训练任务已启动：BP损伤分类 + CNN多任务回归。")
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--internal-worker", "pretrain"]
    else:
        command = [sys.executable, "-u", str(script)]
    command += [
        "--intact_csv", str(Path(sim_undamaged_path).resolve()),
        "--damage_csv", str(Path(sim_damaged_path).resolve()),
        "--out_dir", str(model_dir),
        "--bp_trials", str(cfg.bp_trials),
        "--bp_epochs", str(cfg.bp_epochs),
        "--cnn_epochs", str(cfg.cnn_epochs),
        "--batch_size", str(cfg.batch_size),
    ]
    _stream_process(command, progress_callback, cancel_event, base)
    bundle_path = model_dir / "pretrain_bundle.json"
    bundle = _validate_bundle(bundle_path)
    metadata = {
        "model_name": model_dir.name,
        "model_type": bundle["model_type"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_summary": {
            "simulation_undamaged": str(Path(sim_undamaged_path).resolve()),
            "simulation_damaged": str(Path(sim_damaged_path).resolve()),
        },
        "training_config": asdict(cfg),
        "metrics": bundle.get("metrics", {}),
    }
    _atomic_write_json(model_dir / "software_pretrain_report.json", metadata)
    manifest_path = activate_existing_pretrain(base, bundle_path)
    if progress_callback:
        progress_callback(1.0, "预训练完成：模型已生成并设为迁移学习教师模型。")
    return PretrainResult(model_dir, bundle_path, manifest_path, metadata)
