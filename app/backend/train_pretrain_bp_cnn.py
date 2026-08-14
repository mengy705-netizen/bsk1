# -*- coding: utf-8 -*-
"""预训练阶段：BP损伤分类 + CNN多任务回归。

输入
----
1. 仿真未损伤CSV：48通道应变 + load_x + load_z/load_y + load_size/load_n
2. 仿真损伤CSV：48通道应变 + hole_x/damage_x + hole_z/hole_y/damage_z + hole_d/damage_size

输出
----
输出目录可直接作为后续AE-KD迁移学习的教师模型目录。为了兼容原迁移脚本，
程序同时保存原脚本使用的旧文件名（cnn_load_y、cnn_hold_x、hoald等）。
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import random
import re
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers, regularizers
from tensorflow.keras.models import load_model

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(payload: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_csv_flexible(path_like: str | Path) -> pd.DataFrame:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gbk", "cp936", "latin1"):
        try:
            text = raw.decode(encoding, errors="strict")
        except Exception:
            continue
        for separator in (None, ",", ";", "\t", "|"):
            try:
                stream = io.StringIO(text)
                if separator is None:
                    return pd.read_csv(stream, sep=None, engine="python")
                return pd.read_csv(stream, sep=separator)
            except Exception:
                pass
    return pd.read_csv(path, sep=None, engine="python")


def extract_x48(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    expected = [str(i) for i in range(1, 49)]
    original = list(df.columns)
    string_to_original = {str(c): c for c in original}
    loose: dict[str, object] = {}
    for column in original:
        match = re.search(r"(\d+)", str(column))
        if match:
            number = int(match.group(1))
            if 1 <= number <= 48 and str(number) not in loose:
                loose[str(number)] = column
    chosen: list[object] = []
    missing: list[str] = []
    for name in expected:
        if name in string_to_original:
            chosen.append(string_to_original[name])
        elif name in loose:
            chosen.append(loose[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"缺少48通道传感器列：{missing}")
    values = df[chosen].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    for index in range(values.shape[1]):
        bad = ~np.isfinite(values[:, index])
        values[bad, index] = medians[index]
    return values, [str(c) for c in chosen]


def find_numeric_target(df: pd.DataFrame, aliases: list[str], semantic_name: str) -> tuple[np.ndarray, str]:
    lower = {str(c).strip().lower(): c for c in df.columns}
    selected = None
    for alias in aliases:
        if alias.lower() in lower:
            selected = lower[alias.lower()]
            break
    if selected is None:
        raise KeyError(f"数据中缺少“{semantic_name}”列；支持列名：{aliases}")
    values = pd.to_numeric(df[selected], errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"列 {selected} 存在空值、NaN或非数值内容。")
    return values, str(selected)


def split_three(X: np.ndarray, Y: np.ndarray, *, seed: int, stratify=None):
    indices = np.arange(len(X))
    X_tv, X_test, Y_tv, Y_test, idx_tv, idx_test = train_test_split(
        X, Y, indices, test_size=0.20, random_state=seed, stratify=stratify
    )
    stratify_tv = Y_tv if stratify is not None else None
    X_train, X_val, Y_train, Y_val, idx_train, idx_val = train_test_split(
        X_tv, Y_tv, idx_tv, test_size=0.20, random_state=seed, stratify=stratify_tv
    )
    return (X_train, Y_train, idx_train), (X_val, Y_val, idx_val), (X_test, Y_test, idx_test)


def build_bp(input_dim: int, arch: list[int], activation: str, dropout: float, l2_value: float, lr: float) -> Model:
    inp = layers.Input(shape=(input_dim,), name="strain_48")
    x = inp
    for index, units in enumerate(arch, start=1):
        x = layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_value),
            name=f"bp_dense_{index}",
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"bp_dropout_{index}")(x)
    out = layers.Dense(1, activation="sigmoid", name="damage_probability")(x)
    model = Model(inp, out, name="BP_Damage_Classifier")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_bp_classifier(
    X_intact: np.ndarray,
    X_damage: np.ndarray,
    out_dir: Path,
    *,
    trials: int,
    epochs: int,
    batch_size: int,
) -> dict:
    print("[PRETRAIN] BP分类模型：开始")
    X = np.vstack([X_intact, X_damage]).astype(np.float32)
    y = np.concatenate([
        np.zeros(len(X_intact), dtype=np.int32),
        np.ones(len(X_damage), dtype=np.int32),
    ])
    train, val, test = split_three(X, y, seed=SEED, stratify=y)
    X_train, y_train, idx_train = train
    X_val, y_val, idx_val = val
    X_test, y_test, idx_test = test

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    archs = [[64], [128], [128, 64], [256, 128], [128, 64, 32]]
    activations = ["relu", "tanh"]
    dropouts = [0.0, 0.1, 0.2, 0.3]
    l2s = [0.0, 1e-6, 1e-5, 1e-4]
    lrs = [1e-4, 3e-4, 1e-3]
    trial_rows = []
    best = None
    best_score = -1.0
    trial_root = ensure_dir(out_dir / "bp_trials")

    for trial_id in range(1, max(1, int(trials)) + 1):
        arch = random.choice(archs)
        activation = random.choice(activations)
        dropout = random.choice(dropouts)
        l2_value = random.choice(l2s)
        lr = random.choice(lrs)
        model = build_bp(48, arch, activation, dropout, l2_value, lr)
        checkpoint = trial_root / f"trial_{trial_id:03d}.keras"
        history = model.fit(
            X_train_s,
            y_train,
            validation_data=(X_val_s, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss", patience=7, factor=0.5, min_lr=1e-6),
                callbacks.ModelCheckpoint(checkpoint, monitor="val_accuracy", mode="max", save_best_only=True),
            ],
        )
        candidate = load_model(checkpoint, compile=False)
        val_prob = candidate.predict(X_val_s, verbose=0).reshape(-1)
        val_pred = (val_prob >= 0.5).astype(np.int32)
        score = float(accuracy_score(y_val, val_pred))
        f1 = float(f1_score(y_val, val_pred, zero_division=0))
        row = {
            "trial": trial_id,
            "arch": str(arch),
            "activation": activation,
            "dropout": dropout,
            "l2": l2_value,
            "learning_rate": lr,
            "val_accuracy": score,
            "val_f1": f1,
            "epochs_ran": len(history.history.get("loss", [])),
        }
        trial_rows.append(row)
        print(f"[PRETRAIN][BP] trial {trial_id}/{trials}: val_acc={score:.4f}, val_f1={f1:.4f}")
        if score > best_score or (math.isclose(score, best_score) and best is not None and f1 > best["f1"]):
            best_score = score
            best = {"path": checkpoint, "row": row, "f1": f1}

    if best is None:
        raise RuntimeError("BP分类模型训练失败：没有可用trial。")
    model = load_model(best["path"], compile=False)
    model.save(out_dir / "bp_classifier.h5")
    model.save(out_dir / "bp_classifier.keras")
    joblib.dump(scaler, out_dir / "scaler_X_bp.pkl")

    test_prob = model.predict(X_test_s, verbose=0).reshape(-1)
    test_pred = (test_prob >= 0.5).astype(np.int32)
    metrics = {
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist(),
        "best_hyperparameters": best["row"],
    }
    pd.DataFrame(trial_rows).sort_values(["val_accuracy", "val_f1"], ascending=False).to_csv(
        out_dir / "bp_trial_results.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({
        "global_index": np.concatenate([idx_train, idx_val, idx_test]),
        "split": (["train"] * len(idx_train)) + (["val"] * len(idx_val)) + (["test"] * len(idx_test)),
    }).to_csv(out_dir / "bp_data_split.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"y_true": y_test, "y_prob_damage": test_prob, "y_pred": test_pred}).to_csv(
        out_dir / "bp_predictions_test.csv", index=False, encoding="utf-8-sig"
    )
    save_json(metrics, out_dir / "bp_metrics.json")
    print(f"[PRETRAIN] BP分类模型：完成，test_acc={metrics['accuracy']:.4f}")
    return metrics


def build_cnn_multitask(head_names: list[str], lr: float) -> Model:
    inp = layers.Input(shape=(48,), name="strain_48")
    x = layers.Reshape((48, 1), name="reshape_1d")(inp)
    x = layers.Conv1D(32, 5, padding="same", activation="relu", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling1D(2, name="pool1")(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Conv1D(96, 3, padding="same", activation="relu", name="conv3")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="shared_dense")(x)
    x = layers.Dropout(0.15, name="shared_dropout")(x)
    outputs = [layers.Dense(1, activation="linear", name=name)(x) for name in head_names]
    model = Model(inp, outputs, name="CNN_MultiTask_Regressor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={name: "mse" for name in head_names},
        metrics={name: ["mae"] for name in head_names},
    )
    return model


def train_cnn_branch(
    X: np.ndarray,
    targets: dict[str, np.ndarray],
    out_dir: Path,
    branch: str,
    *,
    epochs: int,
    batch_size: int,
) -> tuple[Model, dict[str, StandardScaler], StandardScaler, dict]:
    head_names = list(targets)
    Y = np.column_stack([targets[name] for name in head_names]).astype(np.float32)
    train, val, test = split_three(X, Y, seed=SEED + (11 if branch == "damage" else 7))
    X_train, Y_train, idx_train = train
    X_val, Y_val, idx_val = val
    X_test, Y_test, idx_test = test

    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler_x.transform(X_val).astype(np.float32)
    X_test_s = scaler_x.transform(X_test).astype(np.float32)

    scaler_y: dict[str, StandardScaler] = {}
    y_train_dict = {}
    y_val_dict = {}
    y_test_dict = {}
    for index, name in enumerate(head_names):
        scaler = StandardScaler()
        y_train_dict[name] = scaler.fit_transform(Y_train[:, index:index + 1]).astype(np.float32)
        y_val_dict[name] = scaler.transform(Y_val[:, index:index + 1]).astype(np.float32)
        y_test_dict[name] = scaler.transform(Y_test[:, index:index + 1]).astype(np.float32)
        scaler_y[name] = scaler

    model = build_cnn_multitask(head_names, lr=1e-3)
    checkpoint = out_dir / f"cnn_{branch}_multitask.keras"
    history = model.fit(
        X_train_s,
        y_train_dict,
        validation_data=(X_val_s, y_val_dict),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", patience=8, factor=0.5, min_lr=1e-6),
            callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
        ],
    )
    model = load_model(checkpoint, compile=False)
    model.save(out_dir / f"cnn_{branch}_multitask.h5")
    joblib.dump(scaler_x, out_dir / f"scaler_X_{branch}.pkl")

    predictions = model.predict(X_test_s, verbose=0)
    if not isinstance(predictions, list):
        predictions = [predictions]
    metrics: dict[str, dict] = {}
    pred_frame: dict[str, np.ndarray] = {}
    for index, name in enumerate(head_names):
        pred_raw = scaler_y[name].inverse_transform(np.asarray(predictions[index]).reshape(-1, 1)).reshape(-1)
        true_raw = Y_test[:, index]
        rmse = float(np.sqrt(mean_squared_error(true_raw, pred_raw)))
        metrics[name] = {
            "n_test": int(len(true_raw)),
            "mae": float(mean_absolute_error(true_raw, pred_raw)),
            "rmse": rmse,
            "r2": float(r2_score(true_raw, pred_raw)),
        }
        pred_frame[f"true_{name}"] = true_raw
        pred_frame[f"pred_{name}"] = pred_raw
        pred_frame[f"abs_error_{name}"] = np.abs(pred_raw - true_raw)
    pd.DataFrame(pred_frame).to_csv(out_dir / f"cnn_{branch}_predictions_test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history.history).to_csv(out_dir / f"cnn_{branch}_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "global_index": np.concatenate([idx_train, idx_val, idx_test]),
        "split": (["train"] * len(idx_train)) + (["val"] * len(idx_val)) + (["test"] * len(idx_test)),
    }).to_csv(out_dir / f"cnn_{branch}_data_split.csv", index=False, encoding="utf-8-sig")
    save_json(metrics, out_dir / f"cnn_{branch}_metrics.json")
    print(f"[PRETRAIN] CNN {branch} 分支：完成")
    return model, scaler_y, scaler_x, metrics


def save_teacher_head(
    master: Model,
    head_name: str,
    model_paths: list[Path],
    scaler_x: StandardScaler,
    scaler_x_paths: list[Path],
    scaler_y: StandardScaler,
    scaler_y_paths: list[Path],
) -> None:
    submodel = Model(master.input, master.get_layer(head_name).output, name=f"Teacher_{head_name}")
    for path in model_paths:
        submodel.save(path)
    for path in scaler_x_paths:
        joblib.dump(scaler_x, path)
    for path in scaler_y_paths:
        joblib.dump(scaler_y, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intact_csv", required=True)
    parser.add_argument("--damage_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bp_trials", type=int, default=10)
    parser.add_argument("--bp_epochs", type=int, default=200)
    parser.add_argument("--cnn_epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    set_seed(SEED)
    out_dir = ensure_dir(args.out_dir)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    df_intact = read_csv_flexible(args.intact_csv)
    df_damage = read_csv_flexible(args.damage_csv)
    X_intact, sensor_cols_intact = extract_x48(df_intact)
    X_damage, sensor_cols_damage = extract_x48(df_damage)

    load_x, col_load_x = find_numeric_target(df_intact, ["load_x"], "无损加载x坐标")
    load_z, col_load_z = find_numeric_target(df_intact, ["load_z", "load_y"], "无损加载z坐标")
    load_size, col_load_size = find_numeric_target(df_intact, ["load_size", "load_n", "force", "load"], "无损载荷大小")

    damage_x, col_damage_x = find_numeric_target(
        df_damage, ["hole_x", "damage_x", "hoald_x", "load_x"], "损伤x坐标"
    )
    damage_z, col_damage_z = find_numeric_target(
        df_damage, ["hole_z", "damage_z", "hole_y", "hoald_y", "load_z", "load_y"], "损伤z坐标"
    )
    damage_size, col_damage_size = find_numeric_target(
        df_damage,
        ["hole_d_mm", "hole_d", "hole_size", "damage_size", "hoald_size", "load_d"],
        "损伤大小",
    )

    bp_metrics = train_bp_classifier(
        X_intact, X_damage, out_dir,
        trials=args.bp_trials, epochs=args.bp_epochs, batch_size=args.batch_size,
    )
    intact_model, intact_scalers_y, intact_scaler_x, intact_metrics = train_cnn_branch(
        X_intact,
        {"load_x": load_x, "load_z": load_z, "load_size": load_size},
        out_dir,
        "intact",
        epochs=args.cnn_epochs,
        batch_size=args.batch_size,
    )
    damage_model, damage_scalers_y, damage_scaler_x, damage_metrics = train_cnn_branch(
        X_damage,
        {"damage_x": damage_x, "damage_z": damage_z, "damage_size": damage_size},
        out_dir,
        "damage",
        epochs=args.cnn_epochs,
        batch_size=args.batch_size,
    )

    # 兼容原迁移脚本使用的教师文件名。
    save_teacher_head(
        intact_model, "load_x",
        [out_dir / "cnn_load_x.h5"],
        intact_scaler_x, [out_dir / "scaler_X_x_load.pkl"],
        intact_scalers_y["load_x"], [out_dir / "scaler_y_load_x.pkl"],
    )
    save_teacher_head(
        intact_model, "load_z",
        [out_dir / "cnn_load_y.h5", out_dir / "cnn_load_z.h5"],
        intact_scaler_x, [out_dir / "scaler_X_y_load.pkl", out_dir / "scaler_X_z_load.pkl"],
        intact_scalers_y["load_z"], [out_dir / "scaler_y_load_y.pkl", out_dir / "scaler_y_load_z.pkl"],
    )
    save_teacher_head(
        intact_model, "load_size",
        [out_dir / "cnn_load_size.h5"],
        intact_scaler_x, [out_dir / "scaler_X_load.pkl"],
        intact_scalers_y["load_size"], [out_dir / "scaler_y_load_size.pkl"],
    )
    save_teacher_head(
        damage_model, "damage_x",
        [out_dir / "cnn_hold_x.h5", out_dir / "cnn_damage_x.h5"],
        damage_scaler_x, [out_dir / "scaler_X_x_damage.save", out_dir / "scaler_X_x_damage.pkl"],
        damage_scalers_y["damage_x"], [out_dir / "scaler_y_hoald_x.save", out_dir / "scaler_y_damage_x.pkl"],
    )
    save_teacher_head(
        damage_model, "damage_z",
        [out_dir / "cnn_hold_y.h5", out_dir / "cnn_damage_z.h5"],
        damage_scaler_x, [out_dir / "scaler_X_y_damage.pkl", out_dir / "scaler_X_z_damage.pkl"],
        damage_scalers_y["damage_z"], [out_dir / "scaler_y_hoald_y.pkl", out_dir / "scaler_y_damage_z.pkl"],
    )
    save_teacher_head(
        damage_model, "damage_size",
        [out_dir / "cnn_hold_size.h5", out_dir / "cnn_damage_size.h5"],
        damage_scaler_x, [out_dir / "scaler_X_s_damage.pkl"],
        damage_scalers_y["damage_size"], [out_dir / "scaler_y_hoald_size.pkl", out_dir / "scaler_y_damage_size.pkl"],
    )

    bundle = {
        "format_version": 1,
        "model_type": "BP-CNN-Wing-Pretrain-v1",
        "feature_count": 48,
        "classifier": {
            "model": "bp_classifier.h5",
            "scaler_x": "scaler_X_bp.pkl",
            "outputs": ["damage_probability"],
        },
        "cnn_intact": {
            "model": "cnn_intact_multitask.keras",
            "scaler_x": "scaler_X_intact.pkl",
            "outputs": ["load_x", "load_z", "load_size"],
        },
        "cnn_damage": {
            "model": "cnn_damage_multitask.keras",
            "scaler_x": "scaler_X_damage.pkl",
            "outputs": ["damage_x", "damage_z", "damage_size"],
        },
        "teacher_compatibility": {
            "transfer_script": "train_transfer_ae_guided_kd_v3.py",
            "enabled": True,
        },
        "source_columns": {
            "sensor_intact": sensor_cols_intact,
            "sensor_damage": sensor_cols_damage,
            "load_x": col_load_x,
            "load_z": col_load_z,
            "load_size": col_load_size,
            "damage_x": col_damage_x,
            "damage_z": col_damage_z,
            "damage_size": col_damage_size,
        },
        "metrics": {
            "bp_classifier": bp_metrics,
            "cnn_intact": intact_metrics,
            "cnn_damage": damage_metrics,
        },
    }
    save_json(bundle, out_dir / "pretrain_bundle.json")
    save_json({
        "model_type": bundle["model_type"],
        "intact_csv": str(Path(args.intact_csv).resolve()),
        "damage_csv": str(Path(args.damage_csv).resolve()),
        "bp_trials": args.bp_trials,
        "bp_epochs": args.bp_epochs,
        "cnn_epochs": args.cnn_epochs,
        "batch_size": args.batch_size,
    }, out_dir / "pretrain_report.json")
    print("[PRETRAIN] 全部完成")
    print(f"PRETRAIN_BUNDLE={out_dir / 'pretrain_bundle.json'}")


if __name__ == "__main__":
    main()
