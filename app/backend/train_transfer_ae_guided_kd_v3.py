# -*- coding: utf-8 -*-
"""
train_transfer_ae_guided_kd_v3.py

本版约定
--------
无损头：保持原来
  - load_x
  - load_y
  - load_size

有损头：改成新损伤文件字段
  - load_x
  - load_z
  - load_d

迁移关系
--------
- 有损 load_x  <- 从无损 load_x 分支初始化
- 有损 load_z  <- 从无损 load_y 分支初始化
- 有损 load_d  <- 从无损 load_size 分支初始化

评估
----
AE训练与KD训练结束后：
- 从新未损伤数据中随机抽 1000 个样本测试
- 从新损伤数据中随机抽 1000 个样本测试

输出
----
1) AE 与微调AE：
   models_transfer_v3/
2) 学生模型：
   models_student_v3/
3) 测试结果（每个头单独目录，包含：
   data_split_index.csv
   history.csv / history.xlsx
   history_curve.(png/pdf/tif)
   metrics_summary.csv
   predictions_test.csv
   scatter_test.png
   residual_hist_test.png
   error_cdf_test.(png/csv)
）
"""

import os
import re
import io
import json
import math
import random
import argparse
import pathlib
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers, models, callbacks, optimizers, metrics
from keras.models import Model, load_model

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

try:
    import joblib
except Exception:
    joblib = None

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# 通用工具
# =========================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_json(o, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2, ensure_ascii=False)


def rmse_score(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def read_csv_flexible(path_like):
    p = pathlib.Path(path_like)
    raw = p.read_bytes()
    for enc in ["utf-8", "utf-8-sig", "gbk", "cp936", "latin1"]:
        try:
            txt = raw.decode(enc, errors="strict")
        except Exception:
            continue
        try:
            return pd.read_csv(io.StringIO(txt), sep=None, engine="python")
        except Exception:
            for sep in [",", ";", "\t", "|"]:
                try:
                    return pd.read_csv(io.StringIO(txt), sep=sep)
                except Exception:
                    pass
    return pd.read_csv(path_like, engine="python", sep=None)


def extract_x48(df: pd.DataFrame):
    expected = [str(i) for i in range(1, 49)]
    cols_str = df.columns.astype(str)

    direct = {e: e for e in expected if e in cols_str}
    loose = {}
    for c in cols_str:
        m = re.search(r"(\d+)", c)
        if m:
            k = int(m.group(1))
            if 1 <= k <= 48 and str(k) not in direct and str(k) not in loose:
                loose[str(k)] = c

    chosen = {}
    for e in expected:
        if e in direct:
            chosen[e] = direct[e]
        elif e in loose:
            chosen[e] = loose[e]

    miss = [e for e in expected if e not in chosen]
    if miss:
        raise ValueError(f"找不到48个传感器列：{miss}")

    X = df[[chosen[e] for e in expected]].to_numpy(dtype=np.float32)
    return X, [chosen[e] for e in expected]


def fix_missing(X):
    X = np.asarray(X, dtype=np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        for j in range(X.shape[1]):
            bj = ~np.isfinite(X[:, j])
            if bj.any():
                X[bj, j] = med[j]
    return X


def collect_labels_v3(df):
    """
    无损：load_x / load_y / load_size
    有损：load_x / load_z / load_d
    分类：is_damage / damage / label / damaged
    """
    cols = {str(c).lower(): c for c in df.columns}

    def find_one(cands):
        for k in cands:
            if k.lower() in cols:
                return cols[k.lower()]
        for c in df.columns:
            lc = str(c).lower()
            for k in cands:
                if k.lower() == lc:
                    return c
        return None

    out = {}

    # 分类
    c = find_one(["is_damage", "damage", "label", "damaged"])
    if c is not None:
        out["is_damage"] = c

    # 无损旧体系
    cx = find_one(["load_x"])
    cy = find_one(["load_y"])
    cs = find_one(["load_size"])

    if cx is not None:
        out["load_x"] = cx
    if cy is not None:
        out["load_y"] = cy
    if cs is not None:
        out["load_size"] = cs

    # 有损新体系
    cz = find_one(["load_z"])
    cd = find_one(["load_d"])

    if cz is not None:
        out["load_z"] = cz
    if cd is not None:
        out["load_d"] = cd

    return out


def load_block(csv_path, fallback_damage_label=None):
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = read_csv_flexible(csv_path)
    X, used = extract_x48(df)
    X = fix_missing(X)
    lab_cols = collect_labels_v3(df)

    labels = {}
    for k, c in lab_cols.items():
        labels[k] = df[c].values

    if "is_damage" not in labels and fallback_damage_label is not None:
        labels["is_damage"] = np.full((len(df),), int(fallback_damage_label), dtype=np.int32)

    return {
        "X": X,
        "labels": labels,
        "used_cols": used,
        "df": df,
    }


def concat_if_exists(*bags):
    Xs = []
    labs = {}
    for b in bags:
        if b is None:
            continue
        Xs.append(b["X"])
        for k, v in b["labels"].items():
            labs.setdefault(k, []).append(v)
    X = np.concatenate(Xs, axis=0) if Xs else None
    for k in list(labs.keys()):
        labs[k] = np.concatenate(labs[k], axis=0)
    return X, labs


def build_head_dataset(*bags, label_key):
    """
    只拼接那些同时具备 X 和指定标签 label_key 的数据块
    保证 X 与 y 严格对齐
    """
    Xs, ys = [], []
    for b in bags:
        if b is None:
            continue
        if "X" not in b or b["X"] is None:
            continue
        if label_key not in b["labels"]:
            continue

        x = np.asarray(b["X"])
        y = np.asarray(b["labels"][label_key])

        if len(x) != len(y):
            raise ValueError(f"{label_key} 对应数据块中 X 与 y 长度不一致: {len(x)} vs {len(y)}")

        Xs.append(x)
        ys.append(y)

    if not Xs:
        return None, None

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    if len(X) != len(y):
        raise ValueError(f"{label_key} 拼接后 X 与 y 长度不一致: {len(X)} vs {len(y)}")

    return X, y


def sample_n_or_all(X, y_dict, n, seed=42):
    if X is None or len(X) == 0:
        return None, None, None
    rs = np.random.RandomState(seed)
    idx_all = np.arange(len(X))
    if len(X) > n:
        idx = rs.choice(idx_all, n, replace=False)
    else:
        idx = idx_all
    idx = np.sort(idx)
    Xs = X[idx]
    ys = {}
    for k, v in y_dict.items():
        ys[k] = np.asarray(v)[idx]
    return Xs, ys, idx


# =========================================================
# AE
# =========================================================
def build_ae(input_dim=48, latent_dim=16):
    inp = layers.Input(shape=(input_dim,), name="ae_input")
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    z = layers.Dense(latent_dim, activation=None, name="latent")(x)
    x = layers.Dense(64, activation="relu")(z)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(input_dim, activation=None)(x)

    ae = Model(inp, out, name="AE")
    ae.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
    enc = Model(inp, z, name="Encoder")
    return ae, enc


def save_history_curves(history, save_prefix, title="Training Curves", ylabel="Loss"):
    hist = history.history if hasattr(history, "history") else {}
    loss = hist.get("loss", None)
    val_loss = hist.get("val_loss", None)
    if loss is None:
        return

    ensure_dir(os.path.dirname(save_prefix))

    plt.figure(figsize=(5.2, 3.2), dpi=200)
    plt.plot(loss, label="train loss")
    if val_loss is not None:
        plt.plot(val_loss, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_prefix + ".png", bbox_inches="tight")
    plt.savefig(save_prefix + ".pdf", bbox_inches="tight")
    plt.savefig(save_prefix + ".tif", bbox_inches="tight")
    plt.close()

    pd.DataFrame({
        "epoch": np.arange(1, len(loss) + 1),
        "loss": loss,
        "val_loss": val_loss if val_loss is not None else [None] * len(loss)
    }).to_csv(save_prefix + ".csv", index=False, encoding="utf-8-sig")


def train_ae_on_data(X_nd, out_dir, latent_dim=16, epochs=200, batch=256, val_split=0.1):
    ensure_dir(out_dir)
    sx = StandardScaler()
    Xs = sx.fit_transform(X_nd.astype(np.float32))

    ae, enc = build_ae(48, latent_dim)
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        callbacks.ModelCheckpoint(os.path.join(out_dir, "nd_autoencoder.h5"),
                                  monitor="val_loss", save_best_only=True, verbose=1)
    ]
    hist = ae.fit(
        Xs, Xs,
        epochs=epochs,
        batch_size=batch,
        validation_split=val_split,
        verbose=2,
        callbacks=cbs
    )
    ae = load_model(os.path.join(out_dir, "nd_autoencoder.h5"), compile=False)
    enc = Model(ae.input, ae.get_layer("latent").output, name="Encoder")

    Xhat = ae.predict(Xs, batch_size=512, verbose=0)
    mse = np.mean((Xs - Xhat) ** 2, axis=1)
    thr = float(np.quantile(mse, 0.95))

    if joblib is not None:
        joblib.dump(sx, os.path.join(out_dir, "nd_scaler.pkl"))

    save_json({"threshold_mse": thr, "latent_dim": latent_dim}, os.path.join(out_dir, "nd_meta.json"))
    save_history_curves(hist, os.path.join(out_dir, "history_curve"), title=f"AE History ({os.path.basename(out_dir)})", ylabel="MSE")
    return sx, ae, enc, thr


def finetune_ae_from(ae, scaler_X, X_dmg, out_dir, epochs=60, batch=128, val_split=0.2, lr=5e-4):
    ensure_dir(out_dir)
    Xs = scaler_X.transform(X_dmg.astype(np.float32))

    ae_ft = models.clone_model(ae)
    ae_ft.build(ae.input_shape)
    ae_ft.set_weights(ae.get_weights())
    ae_ft.compile(optimizer=optimizers.Adam(lr), loss="mse")

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        callbacks.ModelCheckpoint(os.path.join(out_dir, "nd_autoencoder.h5"),
                                  monitor="val_loss", save_best_only=True, verbose=1)
    ]
    hist = ae_ft.fit(
        Xs, Xs,
        epochs=epochs,
        batch_size=batch,
        validation_split=val_split,
        verbose=2,
        callbacks=cbs
    )
    ae_ft = load_model(os.path.join(out_dir, "nd_autoencoder.h5"), compile=False)
    enc_ft = Model(ae_ft.input, ae_ft.get_layer("latent").output, name="Encoder_ft")

    Xhat = ae_ft.predict(Xs, batch_size=512, verbose=0)
    mse = np.mean((Xs - Xhat) ** 2, axis=1)
    thr = float(np.quantile(mse, 0.95))

    if joblib is not None:
        joblib.dump(scaler_X, os.path.join(out_dir, "nd_scaler.pkl"))

    save_json({"threshold_mse": thr, "note": "finetuned_on_damage"}, os.path.join(out_dir, "nd_meta.json"))
    save_history_curves(hist, os.path.join(out_dir, "history_curve"), title=f"AE Finetune ({os.path.basename(out_dir)})", ylabel="MSE")
    return scaler_X, ae_ft, enc_ft, thr


def to_latent(encoder, scaler_X, X_raw):
    Xs = scaler_X.transform(X_raw.astype(np.float32))
    Z = encoder.predict(Xs, batch_size=512, verbose=0)
    return Z


# =========================================================
# 学生模型
# =========================================================
def build_regressor(latent_dim=16, hidden=128):
    inp = layers.Input(shape=(latent_dim,), name="reg_in")
    x = layers.Dense(hidden, activation="relu")(inp)
    x = layers.Dense(hidden // 2, activation="relu")(x)
    y = layers.Dense(1, activation=None)(x)
    m = Model(inp, y)
    return m


def build_classifier(latent_dim=16, hidden=128):
    inp = layers.Input(shape=(latent_dim,), name="cls_in")
    x = layers.Dense(hidden, activation="relu")(inp)
    x = layers.Dense(hidden // 2, activation="relu")(x)
    p = layers.Dense(1, activation="sigmoid")(x)
    m = Model(inp, p)
    return m


def kd_train_regressor(
    Z, y_true_raw, out_h5, out_scaler_pkl, out_dir,
    init_model=None, teacher_y_raw=None, alpha=0.7,
    epochs=200, batch=256, val_split=0.1
):
    ensure_dir(out_dir)

    y_true_raw = np.asarray(y_true_raw).reshape(-1)

    if len(Z) != len(y_true_raw):
        raise ValueError(f"Z 与 y_true_raw 长度不一致: {len(Z)} vs {len(y_true_raw)}")

    if teacher_y_raw is not None:
        teacher_y_raw = np.asarray(teacher_y_raw).reshape(-1)
        if len(teacher_y_raw) != len(y_true_raw):
            raise ValueError(f"KD长度不一致: y_true_raw={len(y_true_raw)}, teacher_y_raw={len(teacher_y_raw)}")

    y_true = y_true_raw.astype(np.float32).reshape(-1, 1)
    sy = StandardScaler()
    ys_true = sy.fit_transform(y_true)

    if teacher_y_raw is not None:
        ys_t = sy.transform(np.asarray(teacher_y_raw, dtype=np.float32).reshape(-1, 1))
        Y = np.concatenate([ys_true, ys_t], axis=1)

        def kd_loss(alpha=alpha):
            def _loss(y_pack, y_pred):
                y_gt = y_pack[:, :1]
                y_te = y_pack[:, 1:2]
                return alpha * tf.reduce_mean(tf.square(y_pred - y_gt)) + (1.0 - alpha) * tf.reduce_mean(tf.square(y_pred - y_te))
            return _loss

        loss_fn = kd_loss(alpha)
    else:
        Y = ys_true
        loss_fn = "mse"

    m = init_model if init_model is not None else build_regressor(latent_dim=Z.shape[1])
    m.compile(optimizer=optimizers.Adam(1e-3), loss=loss_fn, metrics=[metrics.MeanAbsoluteError()])

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
        callbacks.ModelCheckpoint(out_h5, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    hist = m.fit(
        Z, Y,
        epochs=epochs,
        batch_size=batch,
        validation_split=val_split,
        verbose=2,
        callbacks=cbs
    )

    m = load_model(out_h5, compile=False)
    m.compile(optimizer=optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

    if joblib is not None:
        joblib.dump(sy, out_scaler_pkl)

    history_df = pd.DataFrame(hist.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(os.path.join(out_dir, "history.csv"), index=False, encoding="utf-8-sig")
    try:
        history_df.to_excel(os.path.join(out_dir, "history.xlsx"), index=False)
    except Exception:
        pass

    save_history_curves(hist, os.path.join(out_dir, "history_curve"), title=os.path.basename(out_dir), ylabel="Loss")
    return m, sy, history_df


def kd_train_classifier(
    Z, y_true_bin, out_h5, out_dir, teacher_prob=None, alpha=0.7,
    epochs=150, batch=256, val_split=0.1
):
    ensure_dir(out_dir)
    y = np.asarray(y_true_bin).reshape(-1)
    if y.dtype.type is np.str_ or y.dtype.type is np.object_:
        y = np.array([1 if str(v).strip().lower() in ["1", "true", "yes", "y", "damage", "damaged"] else 0 for v in y], dtype=np.int32)

    m = build_classifier(latent_dim=Z.shape[1])

    if teacher_prob is None:
        loss_fn = "binary_crossentropy"
        y_fit = y
    else:
        p_t = np.asarray(teacher_prob, dtype=np.float32).reshape(-1, 1)
        y_pack = np.concatenate([y.reshape(-1, 1).astype(np.float32), p_t], axis=1)

        def kd_bce(alpha=alpha):
            def _loss(y_true_pack, p_s):
                y_true = y_true_pack[:, :1]
                p_tch = tf.clip_by_value(y_true_pack[:, 1:2], 1e-6, 1 - 1e-6)
                p_stu = tf.clip_by_value(p_s, 1e-6, 1 - 1e-6)
                bce = tf.keras.losses.binary_crossentropy(y_true, p_stu)
                ce_soft = -(p_tch * tf.math.log(p_stu) + (1.0 - p_tch) * tf.math.log(1.0 - p_stu))
                return alpha * tf.reduce_mean(bce) + (1.0 - alpha) * tf.reduce_mean(ce_soft)
            return _loss

        loss_fn = kd_bce(alpha)
        y_fit = y_pack

    m.compile(optimizer=optimizers.Adam(1e-3), loss=loss_fn,
              metrics=[metrics.AUC(name="auc"), metrics.BinaryAccuracy(name="acc")])

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        callbacks.ModelCheckpoint(out_h5, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    hist = m.fit(
        Z, y_fit,
        epochs=epochs,
        batch_size=batch,
        validation_split=val_split,
        verbose=2,
        callbacks=cbs
    )

    m = load_model(out_h5, compile=False)
    m.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

    history_df = pd.DataFrame(hist.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(os.path.join(out_dir, "history.csv"), index=False, encoding="utf-8-sig")
    try:
        history_df.to_excel(os.path.join(out_dir, "history.xlsx"), index=False)
    except Exception:
        pass

    save_history_curves(hist, os.path.join(out_dir, "history_curve"), title=os.path.basename(out_dir), ylabel="Loss")
    return m, history_df


# =========================================================
# 教师模型
# =========================================================
class TeacherReg:
    def __init__(self, m_path, sx_path, sy_path):
        if not all(os.path.exists(p) for p in [m_path, sx_path, sy_path]):
            raise FileNotFoundError(f"教师回归缺文件：{m_path} | {sx_path} | {sy_path}")
        self.m = load_model(m_path, compile=False)
        self.sx = joblib.load(sx_path)
        self.sy = joblib.load(sy_path)

    def predict_raw(self, X_raw):
        Xs = self.sx.transform(np.asarray(X_raw, dtype=np.float32))
        y_s = self.m.predict(Xs, verbose=0)
        y = self.sy.inverse_transform(y_s)
        return y.reshape(-1)


class TeacherCls:
    def __init__(self, m_path, sx_path):
        if not all(os.path.exists(p) for p in [m_path, sx_path]):
            raise FileNotFoundError(f"教师分类缺文件：{m_path} | {sx_path}")
        self.m = load_model(m_path, compile=False)
        self.sx = joblib.load(sx_path)

    def predict_prob(self, X_raw):
        Xs = self.sx.transform(np.asarray(X_raw, dtype=np.float32))
        y = self.m.predict(Xs, verbose=0)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            p = y.reshape(-1)
            if (p.min() < 0.0) or (p.max() > 1.0):
                p = 1 / (1 + np.exp(-p))
            return p
        if y.ndim == 2 and y.shape[1] == 2:
            p = y[:, 1]
            if (p.min() < 0.0) or (p.max() > 1.0):
                e = np.exp(y - y.max(axis=1, keepdims=True))
                p = (e / e.sum(axis=1, keepdims=True))[:, 1]
            return p
        return 1 / (1 + np.exp(-y.reshape(-1)))


# =========================================================
# 评估与作图
# =========================================================
def plot_scatter(y_true, y_pred, title, save_path):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = rmse_score(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    pad = 0.05 * (hi - lo + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.75)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.2)
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.title(f"{title}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residual_hist(y_true, y_pred, title, save_path):
    resid = y_pred - y_true
    mae = np.mean(np.abs(resid))
    rmse = np.sqrt(np.mean(resid ** 2))

    plt.figure(figsize=(6.5, 4.8))
    plt.hist(resid, bins=30, alpha=0.85)
    plt.axvline(0, color="k", linewidth=1.2)
    plt.xlabel("Residual (pred - gt)")
    plt.ylabel("Count")
    plt.title(f"{title} Residuals\nMAE={mae:.3f}, RMSE={rmse:.3f}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_error_cdf(y_true, y_pred, title, save_path, save_csv):
    ae = np.abs(y_pred - y_true)
    ae = np.sort(ae)
    cdf = np.linspace(0, 1, len(ae))
    pd.DataFrame({"abs_error": ae, "cdf": cdf}).to_csv(save_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(5.8, 4.8))
    plt.plot(ae, cdf, linewidth=2)
    plt.xlabel("Absolute Error")
    plt.ylabel("CDF")
    plt.title(f"Error CDF ({title})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion(cm, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5.2))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_regression_head(model, sy, Z_test, y_test_raw, out_dir, head_name):
    ensure_dir(out_dir)

    pred_scaled = model.predict(Z_test, verbose=0).reshape(-1, 1)
    y_pred = sy.inverse_transform(pred_scaled).reshape(-1)
    y_true = np.asarray(y_test_raw, dtype=np.float32).reshape(-1)

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": y_pred - y_true,
        "abs_error": np.abs(y_pred - y_true)
    })
    pred_df.to_csv(os.path.join(out_dir, "predictions_test.csv"), index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame([{
        "split": "test",
        "n_samples": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse_score(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }])
    metrics_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False, encoding="utf-8-sig")

    plot_scatter(y_true, y_pred, head_name, os.path.join(out_dir, "scatter_test.png"))
    plot_residual_hist(y_true, y_pred, head_name, os.path.join(out_dir, "residual_hist_test.png"))
    plot_error_cdf(y_true, y_pred, head_name,
                   os.path.join(out_dir, "error_cdf_test.png"),
                   os.path.join(out_dir, "error_cdf_test.csv"))

    return metrics_df


def evaluate_classifier_head(model, Z_test, y_test_raw, out_dir, head_name="bp_cls"):
    ensure_dir(out_dir)

    y_true = np.asarray(y_test_raw).reshape(-1)
    if y_true.dtype.type is np.str_ or y_true.dtype.type is np.object_:
        y_true = np.array([1 if str(v).strip().lower() in ["1", "true", "yes", "y", "damage", "damaged"] else 0 for v in y_true], dtype=np.int32)

    prob = model.predict(Z_test, verbose=0).reshape(-1)
    y_pred = (prob >= 0.5).astype(np.int32)

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "prob_damage": prob,
        "y_pred": y_pred
    })
    pred_df.to_csv(os.path.join(out_dir, "predictions_test.csv"), index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv(os.path.join(out_dir, "confusion_test.csv"), index=False, encoding="utf-8-sig")
    plot_confusion(cm, os.path.join(out_dir, "confusion_test.png"), title=f"Confusion Matrix ({head_name})")

    metrics_df = pd.DataFrame([{
        "split": "test",
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }])
    metrics_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(out_dir, "classification_report_test.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, digits=6))

    return metrics_df


# =========================================================
# 主流程
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--src_nd_csv", type=str, default="probe_data_old_nodamage.csv")
    ap.add_argument("--src_dmg_csv", type=str, default="probe_data_old.csv")
    ap.add_argument("--tgt_nd_csv", type=str, default="probe_pointforceN_int.csv")
    ap.add_argument("--tgt_dmg_csv", type=str, default="probe_forceN_with_hole.csv")

    ap.add_argument("--teachers_dir", type=str, default="./models")
    ap.add_argument("--models_transfer_dir", type=str, default="./models_transfer_v3")
    ap.add_argument("--models_student_dir", type=str, default="./models_student_v3")
    ap.add_argument("--eval_dir", type=str, default="./eval_transfer_v3")

    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--ae_epochs", type=int, default=200)
    ap.add_argument("--ae_batch", type=int, default=256)
    ap.add_argument("--ae_val", type=float, default=0.1)
    ap.add_argument("--ft_epochs", type=int, default=60)
    ap.add_argument("--ft_batch", type=int, default=128)
    ap.add_argument("--ft_val", type=float, default=0.2)

    ap.add_argument("--reg_epochs", type=int, default=200)
    ap.add_argument("--reg_batch", type=int, default=256)
    ap.add_argument("--reg_val", type=float, default=0.1)
    ap.add_argument("--cls_epochs", type=int, default=150)
    ap.add_argument("--cls_batch", type=int, default=256)
    ap.add_argument("--cls_val", type=float, default=0.1)

    ap.add_argument("--alpha_reg_nd", type=float, default=0.7)
    ap.add_argument("--alpha_reg_dmg", type=float, default=0.5)
    ap.add_argument("--alpha_cls", type=float, default=0.7)

    ap.add_argument("--test_n_nd", type=int, default=1000)
    ap.add_argument("--test_n_dmg", type=int, default=1000)

    args = ap.parse_args()

    if joblib is None:
        raise RuntimeError("缺少 joblib，请先安装。")

    MT = args.models_transfer_dir
    MS = args.models_student_dir
    EV = args.eval_dir
    ensure_dir(MT)
    ensure_dir(MS)
    ensure_dir(EV)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    # -------------------------
    # 读数据
    # -------------------------
    SRC_ND = load_block(args.src_nd_csv, fallback_damage_label=0)
    SRC_DG = load_block(args.src_dmg_csv, fallback_damage_label=1)
    TGT_ND = load_block(args.tgt_nd_csv, fallback_damage_label=0)
    TGT_DG = load_block(args.tgt_dmg_csv, fallback_damage_label=1)

    if (SRC_ND is None) and (TGT_ND is None):
        raise RuntimeError("至少需要未损伤CSV用于训练AE。")

    # -------------------------
    # 1) 无损AE：保持原来的三套
    # -------------------------
    X_nd_all, _ = concat_if_exists(SRC_ND, TGT_ND)
    ae_pack = {}

    for tag in ["ae_load_x", "ae_load_y", "ae_load_size"]:
        out_dir = os.path.join(MT, tag)
        sx, ae, enc, thr = train_ae_on_data(
            X_nd_all, out_dir,
            latent_dim=args.latent_dim,
            epochs=args.ae_epochs,
            batch=args.ae_batch,
            val_split=args.ae_val
        )
        ae_pack[tag] = {"sx": sx, "ae": ae, "enc": enc, "thr": thr}

    sx_c, ae_c, enc_c, thr_c = train_ae_on_data(
        X_nd_all, os.path.join(MT, "ae_cls"),
        latent_dim=args.latent_dim,
        epochs=args.ae_epochs,
        batch=args.ae_batch,
        val_split=args.ae_val
    )
    ae_pack["ae_cls"] = {"sx": sx_c, "ae": ae_c, "enc": enc_c, "thr": thr_c}

    # -------------------------
    # 2) 有损AE微调：
    #    load_x <- ae_load_x_ft_dmg
    #    load_z <- ae_load_y_ft_dmg
    #    load_d <- ae_load_size_ft_dmg
    # -------------------------
    if TGT_DG is not None and len(TGT_DG["X"]) > 0:
        X_dmg = TGT_DG["X"]

        for k, base in [
            ("ae_load_x_ft_dmg", "ae_load_x"),
            ("ae_load_y_ft_dmg", "ae_load_y"),
            ("ae_load_size_ft_dmg", "ae_load_size"),
        ]:
            out_dir = os.path.join(MT, k)
            sx_base = ae_pack[base]["sx"]
            ae_base = ae_pack[base]["ae"]
            sx_ft, ae_ft, enc_ft, thr_ft = finetune_ae_from(
                ae_base, sx_base, X_dmg, out_dir,
                epochs=args.ft_epochs,
                batch=args.ft_batch,
                val_split=args.ft_val,
                lr=5e-4
            )
            ae_pack[k] = {"sx": sx_ft, "ae": ae_ft, "enc": enc_ft, "thr": thr_ft}
    else:
        print("[WARN] 无目标域有损数据，跳过有损 AE 微调。")

    # -------------------------
    # 3) 教师绑定
    # -------------------------
    teachers = {}
    TD = args.teachers_dir

    # 无损教师
    try:
        teachers["load_x"] = TeacherReg(
            os.path.join(TD, "cnn_load_x.h5"),
            os.path.join(TD, "scaler_X_x_load.pkl"),
            os.path.join(TD, "scaler_y_load_x.pkl"),
        )
    except Exception:
        print("[INFO] 未找到无损教师 load_x，后续该头用纯监督。")

    try:
        teachers["load_y"] = TeacherReg(
            os.path.join(TD, "cnn_load_y.h5"),
            os.path.join(TD, "scaler_X_y_load.pkl"),
            os.path.join(TD, "scaler_y_load_y.pkl"),
        )
    except Exception:
        print("[INFO] 未找到无损教师 load_y，后续该头用纯监督。")

    try:
        teachers["load_size"] = TeacherReg(
            os.path.join(TD, "cnn_load_size.h5"),
            os.path.join(TD, "scaler_X_load.pkl"),
            os.path.join(TD, "scaler_y_load_size.pkl"),
        )
    except Exception:
        print("[INFO] 未找到无损教师 load_size，后续该头用纯监督。")

    # 有损教师（旧文件名，语义映射到新损伤头）
    try:
        teachers["dmg_load_x"] = TeacherReg(
            os.path.join(TD, "cnn_hold_x.h5"),
            os.path.join(TD, "scaler_X_x_damage.save"),
            os.path.join(TD, "scaler_y_hoald_x.save"),
        )
    except Exception:
        print("[INFO] 未找到有损教师 dmg_load_x，后续该头用纯监督。")

    try:
        teachers["dmg_load_z"] = TeacherReg(
            os.path.join(TD, "cnn_hold_y.h5"),
            os.path.join(TD, "scaler_X_y_damage.pkl"),
            os.path.join(TD, "scaler_y_hoald_y.pkl"),
        )
    except Exception:
        print("[INFO] 未找到有损教师 dmg_load_z，后续该头用纯监督。")

    try:
        teachers["dmg_load_d"] = TeacherReg(
            os.path.join(TD, "cnn_hold_size.h5"),
            os.path.join(TD, "scaler_X_s_damage.pkl"),
            os.path.join(TD, "scaler_y_hoald_size.pkl"),
        )
    except Exception:
        print("[INFO] 未找到有损教师 dmg_load_d，后续该头用纯监督。")

    try:
        cls_m = os.path.join(TD, "bp_classifier.h5")
        cls_s = os.path.join(TD, "scaler_X_bp.pkl")
        if os.path.exists(cls_m) and os.path.exists(cls_s):
            teachers["bp_cls"] = TeacherCls(cls_m, cls_s)
    except Exception:
        pass

    # -------------------------
    # 4) 训练无损学生：保持原来的三头
    # -------------------------
    nd_student = {}

    # load_x
    X_lx, y_lx = build_head_dataset(SRC_ND, TGT_ND, label_key="load_x")
    if X_lx is not None:
        Z = to_latent(ae_pack["ae_load_x"]["enc"], ae_pack["ae_load_x"]["sx"], X_lx)
        t = teachers["load_x"].predict_raw(X_lx) if "load_x" in teachers else None

        out_dir = os.path.join(MS, "student_load_x")
        out_h5 = os.path.join(out_dir, "student_load_x.h5")
        out_sy = os.path.join(out_dir, "scaler_y_load_x.pkl")

        m, sy, _ = kd_train_regressor(
            Z, y_lx, out_h5, out_sy, out_dir,
            teacher_y_raw=t,
            alpha=args.alpha_reg_nd,
            epochs=args.reg_epochs,
            batch=args.reg_batch,
            val_split=args.reg_val
        )
        nd_student["load_x"] = (m, sy)

    # load_y
    X_ly, y_ly = build_head_dataset(SRC_ND, TGT_ND, label_key="load_y")
    if X_ly is not None:
        Z = to_latent(ae_pack["ae_load_y"]["enc"], ae_pack["ae_load_y"]["sx"], X_ly)
        t = teachers["load_y"].predict_raw(X_ly) if "load_y" in teachers else None

        out_dir = os.path.join(MS, "student_load_y")
        out_h5 = os.path.join(out_dir, "student_load_y.h5")
        out_sy = os.path.join(out_dir, "scaler_y_load_y.pkl")

        m, sy, _ = kd_train_regressor(
            Z, y_ly, out_h5, out_sy, out_dir,
            teacher_y_raw=t,
            alpha=args.alpha_reg_nd,
            epochs=args.reg_epochs,
            batch=args.reg_batch,
            val_split=args.reg_val
        )
        nd_student["load_y"] = (m, sy)

    # load_size
    X_ls, y_ls = build_head_dataset(SRC_ND, TGT_ND, label_key="load_size")
    if X_ls is not None:
        Z = to_latent(ae_pack["ae_load_size"]["enc"], ae_pack["ae_load_size"]["sx"], X_ls)
        t = teachers["load_size"].predict_raw(X_ls) if "load_size" in teachers else None

        out_dir = os.path.join(MS, "student_load_size")
        out_h5 = os.path.join(out_dir, "student_load_size.h5")
        out_sy = os.path.join(out_dir, "scaler_y_load_size.pkl")

        m, sy, _ = kd_train_regressor(
            Z, y_ls, out_h5, out_sy, out_dir,
            teacher_y_raw=t,
            alpha=args.alpha_reg_nd,
            epochs=args.reg_epochs,
            batch=args.reg_batch,
            val_split=args.reg_val
        )
        nd_student["load_size"] = (m, sy)

    # 分类学生
    X_cls_list = []
    Y_cls_list = []
    for bag, val in [(SRC_ND, 0), (TGT_ND, 0), (SRC_DG, 1), (TGT_DG, 1)]:
        if bag is None:
            continue
        X_cls_list.append(bag["X"])
        Y_cls_list.append(np.asarray(bag["labels"]["is_damage"]).astype(np.int32))

    if X_cls_list:
        X_cls = np.concatenate(X_cls_list, axis=0)
        Y_cls = np.concatenate(Y_cls_list, axis=0)
        Zc = to_latent(ae_pack["ae_cls"]["enc"], ae_pack["ae_cls"]["sx"], X_cls)
        tprob = teachers["bp_cls"].predict_prob(X_cls) if "bp_cls" in teachers else None

        out_dir = os.path.join(MS, "student_bp_cls")
        out_h5 = os.path.join(out_dir, "student_bp_cls.h5")
        m_cls, _ = kd_train_classifier(
            Zc, Y_cls, out_h5, out_dir,
            teacher_prob=tprob,
            alpha=args.alpha_cls,
            epochs=args.cls_epochs,
            batch=args.cls_batch,
            val_split=args.cls_val
        )
    else:
        m_cls = None

    # -------------------------
    # 5) 训练有损学生：改成新损伤头
    # -------------------------
    dmg_student = {}
    if TGT_DG is not None and len(TGT_DG["X"]) > 0:
        Xd = TGT_DG["X"]
        Ld = TGT_DG["labels"]

        # 有损 load_x <- 无损 load_x
        if ("load_x" in Ld) and ("load_x" in nd_student) and ("ae_load_x_ft_dmg" in ae_pack):
            Zd = to_latent(ae_pack["ae_load_x_ft_dmg"]["enc"], ae_pack["ae_load_x_ft_dmg"]["sx"], Xd)
            y = Ld["load_x"]
            t = teachers["dmg_load_x"].predict_raw(Xd) if "dmg_load_x" in teachers else None

            m_init = build_regressor(latent_dim=Zd.shape[1])
            m_init.set_weights(nd_student["load_x"][0].get_weights())

            out_dir = os.path.join(MS, "student_dmg_load_x")
            out_h5 = os.path.join(out_dir, "student_dmg_load_x.h5")
            out_sy = os.path.join(out_dir, "scaler_y_dmg_load_x.pkl")

            m, sy, _ = kd_train_regressor(
                Zd, y, out_h5, out_sy, out_dir,
                init_model=m_init,
                teacher_y_raw=t,
                alpha=args.alpha_reg_dmg,
                epochs=max(80, args.reg_epochs // 2),
                batch=args.reg_batch,
                val_split=min(0.3, args.reg_val)
            )
            dmg_student["load_x"] = (m, sy)

        # 有损 load_z <- 无损 load_y
        if ("load_z" in Ld) and ("load_y" in nd_student) and ("ae_load_y_ft_dmg" in ae_pack):
            Zd = to_latent(ae_pack["ae_load_y_ft_dmg"]["enc"], ae_pack["ae_load_y_ft_dmg"]["sx"], Xd)
            y = Ld["load_z"]
            t = teachers["dmg_load_z"].predict_raw(Xd) if "dmg_load_z" in teachers else None

            m_init = build_regressor(latent_dim=Zd.shape[1])
            m_init.set_weights(nd_student["load_y"][0].get_weights())

            out_dir = os.path.join(MS, "student_dmg_load_z")
            out_h5 = os.path.join(out_dir, "student_dmg_load_z.h5")
            out_sy = os.path.join(out_dir, "scaler_y_dmg_load_z.pkl")

            m, sy, _ = kd_train_regressor(
                Zd, y, out_h5, out_sy, out_dir,
                init_model=m_init,
                teacher_y_raw=t,
                alpha=args.alpha_reg_dmg,
                epochs=max(80, args.reg_epochs // 2),
                batch=args.reg_batch,
                val_split=min(0.3, args.reg_val)
            )
            dmg_student["load_z"] = (m, sy)

        # 有损 load_d <- 无损 load_size
        if ("load_d" in Ld) and ("load_size" in nd_student) and ("ae_load_size_ft_dmg" in ae_pack):
            Zd = to_latent(ae_pack["ae_load_size_ft_dmg"]["enc"], ae_pack["ae_load_size_ft_dmg"]["sx"], Xd)
            y = Ld["load_d"]
            t = teachers["dmg_load_d"].predict_raw(Xd) if "dmg_load_d" in teachers else None

            m_init = build_regressor(latent_dim=Zd.shape[1])
            m_init.set_weights(nd_student["load_size"][0].get_weights())

            out_dir = os.path.join(MS, "student_dmg_load_d")
            out_h5 = os.path.join(out_dir, "student_dmg_load_d.h5")
            out_sy = os.path.join(out_dir, "scaler_y_dmg_load_d.pkl")

            m, sy, _ = kd_train_regressor(
                Zd, y, out_h5, out_sy, out_dir,
                init_model=m_init,
                teacher_y_raw=t,
                alpha=args.alpha_reg_dmg,
                epochs=max(80, args.reg_epochs // 2),
                batch=args.reg_batch,
                val_split=min(0.3, args.reg_val)
            )
            dmg_student["load_d"] = (m, sy)

    # -------------------------
    # 6) 新数据抽样测试：各1000
    # -------------------------
    X_nd_test, Y_nd_test, idx_nd = sample_n_or_all(
        TGT_ND["X"] if TGT_ND is not None else None,
        TGT_ND["labels"] if TGT_ND is not None else {},
        args.test_n_nd,
        seed=SEED
    )
    X_dmg_test, Y_dmg_test, idx_dmg = sample_n_or_all(
        TGT_DG["X"] if TGT_DG is not None else None,
        TGT_DG["labels"] if TGT_DG is not None else {},
        args.test_n_dmg,
        seed=SEED + 1
    )

    split_rows = []
    if X_nd_test is not None:
        for i in idx_nd:
            split_rows.append({"dataset": "tgt_nd", "global_index": int(i), "split": "test"})
    if X_dmg_test is not None:
        for i in idx_dmg:
            split_rows.append({"dataset": "tgt_dmg", "global_index": int(i), "split": "test"})
    pd.DataFrame(split_rows).to_csv(os.path.join(EV, "data_split_index.csv"), index=False, encoding="utf-8-sig")

    # -------------------------
    # 7) 评估无损三头
    # -------------------------
    if X_nd_test is not None:
        if "load_x" in nd_student and "load_x" in Y_nd_test:
            Z = to_latent(ae_pack["ae_load_x"]["enc"], ae_pack["ae_load_x"]["sx"], X_nd_test)
            evaluate_regression_head(nd_student["load_x"][0], nd_student["load_x"][1], Z, Y_nd_test["load_x"],
                                     os.path.join(EV, "student_load_x"), "student_load_x")

        if "load_y" in nd_student and "load_y" in Y_nd_test:
            Z = to_latent(ae_pack["ae_load_y"]["enc"], ae_pack["ae_load_y"]["sx"], X_nd_test)
            evaluate_regression_head(nd_student["load_y"][0], nd_student["load_y"][1], Z, Y_nd_test["load_y"],
                                     os.path.join(EV, "student_load_y"), "student_load_y")

        if "load_size" in nd_student and "load_size" in Y_nd_test:
            Z = to_latent(ae_pack["ae_load_size"]["enc"], ae_pack["ae_load_size"]["sx"], X_nd_test)
            evaluate_regression_head(nd_student["load_size"][0], nd_student["load_size"][1], Z, Y_nd_test["load_size"],
                                     os.path.join(EV, "student_load_size"), "student_load_size")

    # -------------------------
    # 8) 评估有损三头
    # -------------------------
    if X_dmg_test is not None:
        if "load_x" in dmg_student and "load_x" in Y_dmg_test:
            Z = to_latent(ae_pack["ae_load_x_ft_dmg"]["enc"], ae_pack["ae_load_x_ft_dmg"]["sx"], X_dmg_test)
            evaluate_regression_head(dmg_student["load_x"][0], dmg_student["load_x"][1], Z, Y_dmg_test["load_x"],
                                     os.path.join(EV, "student_dmg_load_x"), "student_dmg_load_x")

        if "load_z" in dmg_student and "load_z" in Y_dmg_test:
            Z = to_latent(ae_pack["ae_load_y_ft_dmg"]["enc"], ae_pack["ae_load_y_ft_dmg"]["sx"], X_dmg_test)
            evaluate_regression_head(dmg_student["load_z"][0], dmg_student["load_z"][1], Z, Y_dmg_test["load_z"],
                                     os.path.join(EV, "student_dmg_load_z"), "student_dmg_load_z")

        if "load_d" in dmg_student and "load_d" in Y_dmg_test:
            Z = to_latent(ae_pack["ae_load_size_ft_dmg"]["enc"], ae_pack["ae_load_size_ft_dmg"]["sx"], X_dmg_test)
            evaluate_regression_head(dmg_student["load_d"][0], dmg_student["load_d"][1], Z, Y_dmg_test["load_d"],
                                     os.path.join(EV, "student_dmg_load_d"), "student_dmg_load_d")

    # -------------------------
    # 9) 评估分类头
    # -------------------------
    if m_cls is not None and (X_nd_test is not None) and (X_dmg_test is not None):
        Xc = np.concatenate([X_nd_test, X_dmg_test], axis=0)
        yc = np.concatenate([
            np.zeros((len(X_nd_test),), dtype=np.int32),
            np.ones((len(X_dmg_test),), dtype=np.int32)
        ], axis=0)
        Zc = to_latent(ae_pack["ae_cls"]["enc"], ae_pack["ae_cls"]["sx"], Xc)
        evaluate_classifier_head(m_cls, Zc, yc, os.path.join(EV, "student_bp_cls"), "student_bp_cls")

    print("\n=== 全部完成 ===")
    print("AE 保存目录：", MT)
    print("学生模型目录：", MS)
    print("测试评估目录：", EV)
    print("无损头保持原来：load_x / load_y / load_size")
    print("有损头改为：load_x / load_z / load_d")
    print("已按新未损/新损伤各抽1000样本测试。")


if __name__ == "__main__":
    main()