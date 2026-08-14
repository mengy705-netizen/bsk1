from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import math
from pathlib import Path
import random
from threading import Event
from typing import Any, Callable

import numpy as np

from .dataset import DomainDataset, finite_target_rows, load_domain_dataset
from .mesh_io import prepare_mesh
from .model_registry import activate_model, write_metadata


ProgressCallback = Callable[[float, str], None]


class TrainingCancelled(RuntimeError):
    pass


class TransferTrainingError(RuntimeError):
    pass


@dataclass
class TransferTrainingConfig:
    feature_count: int = 48
    latent_dim: int = 16
    batch_size: int = 128
    ae_epochs: int = 60
    pretrain_epochs: int = 80
    finetune_epochs: int = 35
    learning_rate: float = 1e-3
    finetune_learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    damage_threshold: float = 0.5
    seed: int = 42
    device: str = "cpu"
    max_training_rows_per_dataset: int = 0


@dataclass
class TrainingResult:
    model_dir: Path
    bundle_path: Path
    manifest_path: Path
    metadata_path: Path
    mesh_npz_path: Path
    metadata: dict[str, Any]


def _progress(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback:
        callback(float(np.clip(fraction, 0.0, 1.0)), message)


def _check_cancel(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise TrainingCancelled("用户已取消模型迁移与更新。")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subsample(dataset: DomainDataset, max_rows: int, seed: int) -> DomainDataset:
    if max_rows <= 0 or dataset.rows <= max_rows:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(dataset.rows, size=max_rows, replace=False))

    def take(target: np.ndarray | None) -> np.ndarray | None:
        return target[indices] if target is not None else None

    return DomainDataset(
        name=dataset.name,
        path=dataset.path,
        features=dataset.features[indices],
        feature_columns=dataset.feature_columns,
        load_targets=take(dataset.load_targets),
        load_target_columns=dataset.load_target_columns,
        damage_targets=take(dataset.damage_targets),
        damage_target_columns=dataset.damage_target_columns,
    )


def _covariance_sqrt(matrix: np.ndarray, inverse: bool = False) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-8, None)
    powered = values ** (-0.5 if inverse else 0.5)
    return (vectors * powered) @ vectors.T


def coral_align_source_to_target(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align source-domain covariance and mean to the target domain using CORAL."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape[0] < 2 or target.shape[0] < 2:
        return source.astype(np.float32)
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    regularizer = np.eye(source.shape[1], dtype=np.float64) * 1e-5
    source_cov = np.cov(source_centered, rowvar=False) + regularizer
    target_cov = np.cov(target_centered, rowvar=False) + regularizer
    transform = _covariance_sqrt(source_cov, inverse=True) @ _covariance_sqrt(target_cov)
    aligned = source_centered @ transform + target_mean
    return aligned.astype(np.float32)


def _fit_standardizer(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate(arrays, axis=0).astype(np.float64)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _normalise(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def _fit_target_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def _target_normalise(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def _make_loader(
    x: np.ndarray,
    y: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    if y is None:
        dataset = TensorDataset(x_tensor)
    else:
        dataset = TensorDataset(x_tensor, torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(
        dataset,
        batch_size=max(1, min(batch_size, len(dataset))),
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def _configure_torch_threads() -> None:
    import torch

    try:
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    except Exception:
        pass


# The architecture mirrors the article workflow at software level:
# shared AE representation -> BP damage branch -> CNN-load and CNN-hole branches,
# with source-domain pretraining, CORAL alignment and target-domain few-shot tuning.
def build_model_classes():
    import torch
    from torch import nn

    class Encoder(nn.Module):
        def __init__(self, feature_count: int, latent_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_count, 64),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim),
            )

        def forward(self, x):
            return self.net(x)

    class AutoEncoder(nn.Module):
        def __init__(self, feature_count: int, latent_dim: int):
            super().__init__()
            self.encoder = Encoder(feature_count, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, feature_count),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class DamageBranch(nn.Module):
        def __init__(self, feature_count: int, latent_dim: int):
            super().__init__()
            self.encoder = Encoder(feature_count, latent_dim)
            self.bp = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.bp(self.encoder(x)).squeeze(-1)

    class CNNRegressionBranch(nn.Module):
        def __init__(self, feature_count: int, latent_dim: int, output_dim: int):
            super().__init__()
            self.encoder = Encoder(feature_count, latent_dim)
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
            )
            self.head = nn.Sequential(
                nn.Linear(32 * 8 + latent_dim, 96),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Linear(48, output_dim),
            )

        def forward(self, x):
            latent = self.encoder(x)
            conv = self.cnn(x.unsqueeze(1)).flatten(1)
            return self.head(torch.cat([conv, latent], dim=1))

    return Encoder, AutoEncoder, DamageBranch, CNNRegressionBranch


def _train_autoencoder(
    model,
    x: np.ndarray,
    cfg: TransferTrainingConfig,
    callback: ProgressCallback | None,
    cancel_event: Event | None,
    start: float,
    end: float,
) -> list[float]:
    import torch
    from torch import nn

    device = torch.device(cfg.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()
    loader = _make_loader(x, None, cfg.batch_size, shuffle=True)
    history: list[float] = []
    epochs = max(1, cfg.ae_epochs)
    for epoch in range(epochs):
        _check_cancel(cancel_event)
        model.train()
        total = 0.0
        count = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch_x)
            loss = loss_fn(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch_x.shape[0]
            count += int(batch_x.shape[0])
        average = total / max(count, 1)
        history.append(average)
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 10) == 0:
            fraction = start + (end - start) * (epoch + 1) / epochs
            _progress(callback, fraction, f"AE共享表征训练：{epoch + 1}/{epochs}，MSE={average:.6g}")
    return history


def _set_encoder_trainable(model, trainable: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable


def _train_supervised(
    model,
    x: np.ndarray,
    y: np.ndarray,
    cfg: TransferTrainingConfig,
    epochs: int,
    lr: float,
    task: str,
    callback: ProgressCallback | None,
    cancel_event: Event | None,
    start: float,
    end: float,
    encoder_trainable: bool,
) -> list[float]:
    import torch
    from torch import nn

    if x.shape[0] == 0:
        return []
    _set_encoder_trainable(model, encoder_trainable)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    device = torch.device(cfg.device)
    model.to(device)
    loader = _make_loader(x, y, cfg.batch_size, shuffle=True)
    history: list[float] = []
    epochs = max(1, int(epochs))
    for epoch in range(epochs):
        _check_cancel(cancel_event)
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            if task == "classification":
                batch_y = batch_y.reshape(-1)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch_x.shape[0]
            count += int(batch_x.shape[0])
        average = total / max(count, 1)
        history.append(average)
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 8) == 0:
            fraction = start + (end - start) * (epoch + 1) / epochs
            stage = "源域预训练" if not encoder_trainable else "实际数据少样本微调"
            _progress(callback, fraction, f"{stage}：{epoch + 1}/{epochs}，loss={average:.6g}")
    return history


def _predict_numpy(model, x: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        output = model(torch.as_tensor(x, dtype=torch.float32, device=device))
    return output.detach().cpu().numpy()


def _classification_metrics(model, x: np.ndarray, y: np.ndarray, device: str) -> dict[str, float]:
    if x.shape[0] == 0:
        return {}
    logits = _predict_numpy(model, x, device).reshape(-1)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    targets = y.reshape(-1)
    predictions = (probabilities >= 0.5).astype(np.float32)
    accuracy = float(np.mean(predictions == targets))
    eps = 1e-8
    bce = float(
        -np.mean(targets * np.log(probabilities + eps) + (1 - targets) * np.log(1 - probabilities + eps))
    )
    return {"accuracy": accuracy, "binary_cross_entropy": bce}


def _regression_metrics(
    model,
    x: np.ndarray,
    y_original: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: str,
) -> dict[str, Any]:
    if x.shape[0] == 0:
        return {}
    normalised_prediction = _predict_numpy(model, x, device)
    prediction = normalised_prediction * target_std + target_mean
    errors = prediction - y_original
    mae_each = np.mean(np.abs(errors), axis=0)
    rmse_each = np.sqrt(np.mean(errors**2, axis=0))
    return {
        "mae": [float(value) for value in mae_each],
        "rmse": [float(value) for value in rmse_each],
        "mae_mean": float(np.mean(mae_each)),
        "rmse_mean": float(np.mean(rmse_each)),
    }


def _copy_encoder_state(source_autoencoder, target_branch) -> None:
    target_branch.encoder.load_state_dict(deepcopy(source_autoencoder.encoder.state_dict()))


def _valid_target_subset(dataset: DomainDataset, target_name: str) -> tuple[np.ndarray, np.ndarray]:
    values = dataset.load_targets if target_name == "load" else dataset.damage_targets
    if values is None:
        return np.empty((0, dataset.features.shape[1]), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    mask = finite_target_rows(values)
    return dataset.features[mask], values[mask]


def _dataset_summary(dataset: DomainDataset) -> dict[str, Any]:
    return {
        "filename": dataset.path.name,
        "rows": dataset.rows,
        "feature_columns": dataset.feature_columns,
        "load_labels_found": bool(dataset.load_targets is not None),
        "load_target_columns": dataset.load_target_columns,
        "damage_labels_found": bool(dataset.damage_targets is not None),
        "damage_target_columns": dataset.damage_target_columns,
    }


def train_and_activate_aefst(
    *,
    base_dir: str | Path,
    sim_undamaged_path: str | Path,
    sim_damaged_path: str | Path,
    actual_undamaged_path: str | Path,
    actual_damaged_path: str | Path,
    geometry_path: str | Path,
    span_axis: str = "Z",
    chord_axis: str = "X",
    geometry_unit: str = "m",
    reverse_span: bool = False,
    training_config: TransferTrainingConfig | None = None,
    feature_columns: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> TrainingResult:
    import torch

    cfg = training_config or TransferTrainingConfig()
    _configure_torch_threads()
    _seed_everything(cfg.seed)
    base = Path(base_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = base / "models" / f"AEFST_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=False)

    _progress(progress_callback, 0.01, "正在读取四类数据并检查48通道和标签列……")
    _check_cancel(cancel_event)
    datasets = {
        "simulation_undamaged": load_domain_dataset(
            sim_undamaged_path, "仿真未损伤数据", cfg.feature_count, feature_columns
        ),
        "simulation_damaged": load_domain_dataset(
            sim_damaged_path, "仿真损伤数据", cfg.feature_count, feature_columns
        ),
        "actual_undamaged": load_domain_dataset(
            actual_undamaged_path, "实际未损伤数据", cfg.feature_count, feature_columns
        ),
        "actual_damaged": load_domain_dataset(
            actual_damaged_path, "实际损伤数据", cfg.feature_count, feature_columns
        ),
    }
    for index, key in enumerate(list(datasets)):
        datasets[key] = _subsample(
            datasets[key], cfg.max_training_rows_per_dataset, cfg.seed + index
        )
    feature_names = datasets["simulation_undamaged"].feature_columns
    for dataset in datasets.values():
        if dataset.features.shape[1] != cfg.feature_count:
            raise TransferTrainingError(f"{dataset.name} 的应变通道数不是 {cfg.feature_count}。")

    _progress(progress_callback, 0.045, "正在读取并规范化新机翼三维模型……")
    prepared_mesh = prepare_mesh(
        geometry_path,
        model_dir,
        span_axis=span_axis,
        chord_axis=chord_axis,
        unit=geometry_unit,
        reverse_span=reverse_span,
    )
    _check_cancel(cancel_event)

    sim_u = datasets["simulation_undamaged"]
    sim_d = datasets["simulation_damaged"]
    act_u = datasets["actual_undamaged"]
    act_d = datasets["actual_damaged"]

    _progress(progress_callback, 0.07, "正在执行仿真域到实际域的数据对齐（CORAL）……")
    actual_all = np.concatenate([act_u.features, act_d.features], axis=0)
    sim_u_aligned = coral_align_source_to_target(sim_u.features, act_u.features)
    sim_d_aligned = coral_align_source_to_target(sim_d.features, act_d.features)
    sim_all_aligned = np.concatenate([sim_u_aligned, sim_d_aligned], axis=0)

    input_mean, input_std = _fit_standardizer(
        [sim_u_aligned, sim_d_aligned, act_u.features, act_d.features]
    )
    x_sim_u = _normalise(sim_u_aligned, input_mean, input_std)
    x_sim_d = _normalise(sim_d_aligned, input_mean, input_std)
    x_act_u = _normalise(act_u.features, input_mean, input_std)
    x_act_d = _normalise(act_d.features, input_mean, input_std)
    x_ae = np.concatenate([x_sim_u, x_sim_d, x_act_u, x_act_d], axis=0)

    Encoder, AutoEncoder, DamageBranch, CNNRegressionBranch = build_model_classes()
    autoencoder = AutoEncoder(cfg.feature_count, cfg.latent_dim)
    ae_history = _train_autoencoder(
        autoencoder,
        x_ae,
        cfg,
        progress_callback,
        cancel_event,
        0.08,
        0.31,
    )

    # BP damage classification branch. Dataset identity supplies the 0/1 labels,
    # so actual datasets can contribute even when they do not include an explicit
    # is_damage column.
    damage_branch = DamageBranch(cfg.feature_count, cfg.latent_dim)
    _copy_encoder_state(autoencoder, damage_branch)
    x_class_source = np.concatenate([x_sim_u, x_sim_d], axis=0)
    y_class_source = np.concatenate(
        [np.zeros(x_sim_u.shape[0]), np.ones(x_sim_d.shape[0])]
    ).astype(np.float32)
    x_class_target = np.concatenate([x_act_u, x_act_d], axis=0)
    y_class_target = np.concatenate(
        [np.zeros(x_act_u.shape[0]), np.ones(x_act_d.shape[0])]
    ).astype(np.float32)
    _progress(progress_callback, 0.32, "正在训练 BP 损伤识别迁移分支……")
    class_pre_history = _train_supervised(
        damage_branch,
        x_class_source,
        y_class_source,
        cfg,
        cfg.pretrain_epochs,
        cfg.learning_rate,
        "classification",
        progress_callback,
        cancel_event,
        0.32,
        0.43,
        encoder_trainable=False,
    )
    class_fine_history = _train_supervised(
        damage_branch,
        x_class_target,
        y_class_target,
        cfg,
        cfg.finetune_epochs,
        cfg.finetune_learning_rate,
        "classification",
        progress_callback,
        cancel_event,
        0.43,
        0.51,
        encoder_trainable=True,
    )

    # CNN-load branch. Source labels are required because load prediction is a
    # mandatory output in the monitoring page. Actual labels are optional: when
    # present they perform the requested few-shot tuning; when absent the branch
    # remains source-pretrained after domain alignment.
    sim_load_x, sim_load_y = _valid_target_subset(sim_u, "load")
    if sim_load_y.shape[0] == 0:
        raise TransferTrainingError(
            "仿真未损伤数据缺少载荷标签。至少需要 load_x、load_z（或 load_y）和 load_size/load_n 三列。"
        )
    # Use the same row mask against the already aligned/normalised source matrix.
    sim_load_mask = finite_target_rows(sim_u.load_targets)
    x_load_source = x_sim_u[sim_load_mask]
    y_load_source = sim_u.load_targets[sim_load_mask]
    act_load_mask = finite_target_rows(act_u.load_targets)
    x_load_target = x_act_u[act_load_mask] if act_load_mask.size else np.empty((0, cfg.feature_count), dtype=np.float32)
    y_load_target = act_u.load_targets[act_load_mask] if act_load_mask.size else np.empty((0, 3), dtype=np.float32)
    load_scaler_values = (
        np.concatenate([y_load_source, y_load_target], axis=0)
        if y_load_target.shape[0]
        else y_load_source
    )
    load_mean, load_std = _fit_target_scaler(load_scaler_values)
    y_load_source_n = _target_normalise(y_load_source, load_mean, load_std)
    y_load_target_n = _target_normalise(y_load_target, load_mean, load_std) if y_load_target.shape[0] else y_load_target

    load_branch = CNNRegressionBranch(cfg.feature_count, cfg.latent_dim, 3)
    _copy_encoder_state(autoencoder, load_branch)
    _progress(progress_callback, 0.52, "正在训练 CNN-load 载荷识别迁移分支……")
    load_pre_history = _train_supervised(
        load_branch,
        x_load_source,
        y_load_source_n,
        cfg,
        cfg.pretrain_epochs,
        cfg.learning_rate,
        "regression",
        progress_callback,
        cancel_event,
        0.52,
        0.65,
        encoder_trainable=False,
    )
    load_fine_history: list[float] = []
    if x_load_target.shape[0]:
        load_fine_history = _train_supervised(
            load_branch,
            x_load_target,
            y_load_target_n,
            cfg,
            cfg.finetune_epochs,
            cfg.finetune_learning_rate,
            "regression",
            progress_callback,
            cancel_event,
            0.65,
            0.74,
            encoder_trainable=True,
        )
    else:
        _progress(progress_callback, 0.74, "实际未损伤数据没有载荷标签：已保留对齐后的源域载荷模型。")

    # CNN-hole branch. This branch is optional only when neither simulation nor
    # actual damaged data contains hole/damage coordinates and size.
    sim_damage_mask = finite_target_rows(sim_d.damage_targets)
    act_damage_mask = finite_target_rows(act_d.damage_targets)
    has_damage_labels = bool(sim_damage_mask.size and sim_damage_mask.any())
    damage_reg_branch = None
    damage_mean = np.zeros(3, dtype=np.float32)
    damage_std = np.ones(3, dtype=np.float32)
    damage_pre_history: list[float] = []
    damage_fine_history: list[float] = []
    if has_damage_labels:
        x_damage_source = x_sim_d[sim_damage_mask]
        y_damage_source = sim_d.damage_targets[sim_damage_mask]
        x_damage_target = x_act_d[act_damage_mask] if act_damage_mask.size else np.empty((0, cfg.feature_count), dtype=np.float32)
        y_damage_target = act_d.damage_targets[act_damage_mask] if act_damage_mask.size else np.empty((0, 3), dtype=np.float32)
        damage_scaler_values = (
            np.concatenate([y_damage_source, y_damage_target], axis=0)
            if y_damage_target.shape[0]
            else y_damage_source
        )
        damage_mean, damage_std = _fit_target_scaler(damage_scaler_values)
        y_damage_source_n = _target_normalise(y_damage_source, damage_mean, damage_std)
        y_damage_target_n = (
            _target_normalise(y_damage_target, damage_mean, damage_std)
            if y_damage_target.shape[0]
            else y_damage_target
        )
        damage_reg_branch = CNNRegressionBranch(cfg.feature_count, cfg.latent_dim, 3)
        _copy_encoder_state(autoencoder, damage_reg_branch)
        _progress(progress_callback, 0.75, "正在训练 CNN-hole 损伤定位迁移分支……")
        damage_pre_history = _train_supervised(
            damage_reg_branch,
            x_damage_source,
            y_damage_source_n,
            cfg,
            cfg.pretrain_epochs,
            cfg.learning_rate,
            "regression",
            progress_callback,
            cancel_event,
            0.75,
            0.86,
            encoder_trainable=False,
        )
        if x_damage_target.shape[0]:
            damage_fine_history = _train_supervised(
                damage_reg_branch,
                x_damage_target,
                y_damage_target_n,
                cfg,
                cfg.finetune_epochs,
                cfg.finetune_learning_rate,
                "regression",
                progress_callback,
                cancel_event,
                0.86,
                0.93,
                encoder_trainable=True,
            )
        else:
            _progress(progress_callback, 0.93, "实际损伤数据没有孔洞标签：已保留对齐后的源域损伤模型。")
    else:
        _progress(
            progress_callback,
            0.93,
            "仿真损伤数据未发现 hole_x、hole_z、hole_d 标签：本次仅更新损伤分类和载荷模型。",
        )

    _check_cancel(cancel_event)
    _progress(progress_callback, 0.94, "正在计算训练诊断指标并验证模型包……")
    class_metrics = _classification_metrics(
        damage_branch, x_class_target, y_class_target, cfg.device
    )
    class_metrics["evaluation_scope"] = "实际数据训练集诊断（非独立测试集）"
    load_eval_x = x_load_target if x_load_target.shape[0] else x_load_source
    load_eval_y = y_load_target if y_load_target.shape[0] else y_load_source
    load_metrics = _regression_metrics(
        load_branch, load_eval_x, load_eval_y, load_mean, load_std, cfg.device
    )
    load_metrics["evaluation_scope"] = (
        "实际未损伤训练集诊断（非独立测试集）"
        if x_load_target.shape[0]
        else "对齐后的仿真未损伤训练集诊断（非独立测试集）"
    )
    damage_metrics: dict[str, Any] = {}
    if damage_reg_branch is not None:
        x_damage_eval = x_act_d[act_damage_mask] if act_damage_mask.size and act_damage_mask.any() else x_sim_d[sim_damage_mask]
        y_damage_eval = act_d.damage_targets[act_damage_mask] if act_damage_mask.size and act_damage_mask.any() else sim_d.damage_targets[sim_damage_mask]
        damage_metrics = _regression_metrics(
            damage_reg_branch,
            x_damage_eval,
            y_damage_eval,
            damage_mean,
            damage_std,
            cfg.device,
        )
        damage_metrics["evaluation_scope"] = (
            "实际损伤训练集诊断（非独立测试集）"
            if act_damage_mask.size and act_damage_mask.any()
            else "对齐后的仿真损伤训练集诊断（非独立测试集）"
        )

    bundle_path = model_dir / "aefst_model_bundle.pt"
    bundle = {
        "format_version": 2,
        "model_type": "AE-FST",
        "feature_count": cfg.feature_count,
        "feature_columns": feature_names,
        "latent_dim": cfg.latent_dim,
        "damage_threshold": cfg.damage_threshold,
        "input_mean": input_mean,
        "input_std": input_std,
        "damage_classifier_state": damage_branch.state_dict(),
        "load_regressor_state": load_branch.state_dict(),
        "damage_regressor_state": (
            damage_reg_branch.state_dict() if damage_reg_branch is not None else None
        ),
        "load_target_mean": load_mean,
        "load_target_std": load_std,
        "damage_target_mean": damage_mean,
        "damage_target_std": damage_std,
        "geometry": prepared_mesh.metadata,
    }
    torch.save(bundle, bundle_path)

    metrics = {
        "damage_classification": class_metrics,
        "load_regression": load_metrics,
        "damage_regression": damage_metrics,
    }
    dataset_summary = {key: _dataset_summary(value) for key, value in datasets.items()}
    metadata = {
        "format_version": 2,
        "model_name": model_dir.name,
        "model_type": "AE-FST",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "article_workflow": {
            "damage_branch": "BP + normalization + CORAL alignment + AE representation + transfer/fine-tune",
            "load_branch": "CNN-load + normalization + CORAL alignment + AE latent Z + transfer/fine-tune",
            "damage_localization_branch": "CNN-hole + normalization + CORAL alignment + AE latent Z + few-shot tune",
        },
        "geometry": prepared_mesh.metadata,
        "metrics": metrics,
        "dataset_summary": dataset_summary,
        "training_config": asdict(cfg),
        "history": {
            "ae": ae_history,
            "damage_pretrain": class_pre_history,
            "damage_finetune": class_fine_history,
            "load_pretrain": load_pre_history,
            "load_finetune": load_fine_history,
            "hole_pretrain": damage_pre_history,
            "hole_finetune": damage_fine_history,
        },
        "limitations": [
            "诊断指标来自训练数据，不等同于独立测试集性能。",
            "三维模型必须使用与数据标签一致的坐标方向和单位。",
            "48通道排列顺序必须与仿真和实际采集完全一致。",
        ],
    }
    metadata_path = write_metadata(model_dir / "training_report.json", metadata)

    # Validation uses the actual-undamaged mean vector. Activation occurs only
    # after a successful load-and-predict cycle, so a failed update never replaces
    # the previously active model.
    validation_input = np.mean(act_u.features, axis=0)
    validation_prediction = predict_aefst_bundle(bundle_path, validation_input)
    required = ("load_x_mm", "load_z_mm", "load_n", "damage_probability")
    if any(key not in validation_prediction for key in required):
        raise TransferTrainingError("新模型包验证失败：缺少必要预测字段。")
    if not all(np.isfinite(float(validation_prediction[key])) for key in required):
        raise TransferTrainingError("新模型包验证失败：预测结果包含 NaN 或无穷大。")

    manifest_path = activate_model(
        base,
        bundle_path,
        model_dir,
        metadata,
        mesh_npz_path=prepared_mesh.npz_path,
    )
    _progress(progress_callback, 1.0, "迁移学习完成，新模型已通过验证并设为软件当前后台模型。")
    return TrainingResult(
        model_dir=model_dir,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        mesh_npz_path=prepared_mesh.npz_path,
        metadata=metadata,
    )


_MODEL_CACHE: dict[str, tuple[float, Any]] = {}


def _load_bundle_model(bundle_path: Path):
    import torch

    key = str(bundle_path.resolve())
    modified = bundle_path.stat().st_mtime
    cached = _MODEL_CACHE.get(key)
    if cached and cached[0] == modified:
        return cached[1]
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    required = {
        "feature_count",
        "latent_dim",
        "input_mean",
        "input_std",
        "damage_classifier_state",
        "load_regressor_state",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise TransferTrainingError(f"AE-FST模型包缺少字段：{missing}")
    _, _, DamageBranch, CNNRegressionBranch = build_model_classes()
    classifier = DamageBranch(int(bundle["feature_count"]), int(bundle["latent_dim"]))
    load_regressor = CNNRegressionBranch(
        int(bundle["feature_count"]), int(bundle["latent_dim"]), 3
    )
    classifier.load_state_dict(bundle["damage_classifier_state"])
    load_regressor.load_state_dict(bundle["load_regressor_state"])
    classifier.eval()
    load_regressor.eval()
    damage_regressor = None
    if bundle.get("damage_regressor_state") is not None:
        damage_regressor = CNNRegressionBranch(
            int(bundle["feature_count"]), int(bundle["latent_dim"]), 3
        )
        damage_regressor.load_state_dict(bundle["damage_regressor_state"])
        damage_regressor.eval()
    loaded = (bundle, classifier, load_regressor, damage_regressor)
    _MODEL_CACHE[key] = (modified, loaded)
    return loaded


def predict_aefst_bundle(bundle_path: str | Path, strain_values: np.ndarray) -> dict[str, float]:
    import torch

    path = Path(bundle_path)
    if not path.exists():
        raise TransferTrainingError(f"未找到AE-FST模型包：{path}")
    bundle, classifier, load_regressor, damage_regressor = _load_bundle_model(path)
    values = np.asarray(strain_values, dtype=np.float32).reshape(-1)
    feature_count = int(bundle["feature_count"])
    if values.size != feature_count:
        raise TransferTrainingError(
            f"模型要求 {feature_count} 个应变通道，当前输入 {values.size} 个。"
        )
    mean = np.asarray(bundle["input_mean"], dtype=np.float32)
    std = np.asarray(bundle["input_std"], dtype=np.float32)
    x = ((values - mean) / std).reshape(1, -1)
    tensor = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        probability = float(torch.sigmoid(classifier(tensor)).item())
        load_n = load_regressor(tensor).cpu().numpy().reshape(-1)
    load_mean = np.asarray(bundle["load_target_mean"], dtype=float)
    load_std = np.asarray(bundle["load_target_std"], dtype=float)
    load_values = load_n * load_std + load_mean
    result: dict[str, float] = {
        "load_x_mm": float(load_values[0]),
        "load_z_mm": float(load_values[1]),
        "load_n": float(load_values[2]),
        "damage_probability": probability,
        "prediction_mode": "AE-FST",
    }
    if damage_regressor is not None:
        with torch.no_grad():
            damage_n = damage_regressor(tensor).cpu().numpy().reshape(-1)
        damage_mean = np.asarray(bundle["damage_target_mean"], dtype=float)
        damage_std = np.asarray(bundle["damage_target_std"], dtype=float)
        damage_values = damage_n * damage_std + damage_mean
        result.update(
            {
                "damage_x_mm": float(damage_values[0]),
                "damage_z_mm": float(damage_values[1]),
                "damage_size_mm": float(damage_values[2]),
            }
        )
    return result
