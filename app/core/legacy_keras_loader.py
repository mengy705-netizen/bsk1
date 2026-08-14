from __future__ import annotations

"""Compatibility loader for the user's Keras 2 / TensorFlow 2.10 models."""

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class LegacyKerasLoadError(RuntimeError):
    pass


def _custom_objects(tf) -> dict[str, Any]:
    objects: dict[str, Any] = {}
    try:
        objects["DTypePolicy"] = tf.keras.mixed_precision.Policy
        objects["Policy"] = tf.keras.mixed_precision.Policy
    except Exception:
        pass
    return objects


def load_model_compat(path: str | Path, input_shape: tuple[int, ...]):
    """Load a Keras model, with a sequential H5 reconstruction fallback.

    The fallback mirrors the compatibility strategy in the user's supplied
    inference code. It is intended for old sequential BP/Conv1D H5 models.
    """

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    import tensorflow as tf
    from keras.layers import Input
    from keras.layers import deserialize as layer_deserialize
    from keras.models import Model, load_model

    custom_objects = _custom_objects(tf)
    try:
        return load_model(model_path, compile=False, custom_objects=custom_objects)
    except Exception as first_error:
        err1 = first_error

    if model_path.suffix.lower() not in {".h5", ".hdf5"}:
        raise LegacyKerasLoadError(
            f"无法加载Keras模型：{model_path.name} | {type(err1).__name__}: {err1}"
        )

    with h5py.File(model_path, "r") as handle:
        raw = handle.attrs.get("model_config", None)
        if raw is None and "model_config" in handle:
            raw = handle["model_config"][()]
    if raw is None:
        raise LegacyKerasLoadError(f"无法从 {model_path.name} 读取 model_config。")
    if isinstance(raw, (bytes, bytearray)):
        config_text = raw.decode("utf-8")
    elif isinstance(raw, str):
        config_text = raw
    else:
        config_text = bytes(raw).decode("utf-8")

    full_config = json.loads(config_text)
    model_config = full_config.get("config", {})
    layer_configs = model_config.get("layers", [])
    if not isinstance(layer_configs, list) or not layer_configs:
        raise LegacyKerasLoadError("model_config.layers为空，无法执行兼容还原。")

    inp = Input(shape=input_shape, dtype="float32", name="compat_input")
    x = inp
    used_names: set[str] = set()

    def unique_name(base: str) -> str:
        candidate = base
        index = 1
        while candidate in used_names:
            candidate = f"{base}_{index}"
            index += 1
        used_names.add(candidate)
        return candidate

    for index, layer_record in enumerate(layer_configs):
        class_name = layer_record.get("class_name")
        layer_config = dict(layer_record.get("config", {}))
        if class_name == "InputLayer":
            continue
        layer_config.pop("dtype_policy", None)
        layer_config.pop("_dtype_policy", None)
        if "dtype" in layer_config and not isinstance(layer_config["dtype"], str):
            layer_config["dtype"] = "float32"
        raw_name = layer_record.get("name")
        if not isinstance(raw_name, str) or not raw_name:
            raw_name = f"compat_{index}_{str(class_name).lower()}"
        layer_config["name"] = unique_name(raw_name)
        try:
            layer = layer_deserialize(
                {"class_name": class_name, "config": layer_config},
                custom_objects=custom_objects,
            )
            x = layer(x)
        except Exception as exc:
            raise LegacyKerasLoadError(
                f"兼容还原层失败：{class_name}({layer_config['name']}) | {exc}"
            ) from exc

    model = Model(inputs=inp, outputs=x, name=model_config.get("name", "compat_model"))

    try:
        model.load_weights(model_path)
        return model
    except Exception:
        pass

    try:
        with h5py.File(model_path, "r") as handle:
            root = handle["model_weights"] if "model_weights" in handle else handle
            if "layer_names" in root.attrs:
                names = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in root.attrs["layer_names"]
                ]
                groups = [root[name] for name in names if name in root]
            else:
                groups = [root[key] for key in root.keys() if isinstance(root[key], h5py.Group)]

            by_name = {layer.name.split("/")[-1]: layer for layer in model.layers}
            for group in groups:
                layer_name = group.name.split("/")[-1]
                layer = by_name.get(layer_name)
                if layer is None:
                    continue
                datasets: list[Any]
                if "weight_names" in group.attrs:
                    weight_names = [
                        item.decode("utf-8") if isinstance(item, bytes) else str(item)
                        for item in group.attrs["weight_names"]
                    ]
                    datasets = [group[name] for name in weight_names if name in group]
                else:
                    datasets = [
                        group[key] for key in group.keys() if isinstance(group[key], h5py.Dataset)
                    ]
                values = [np.asarray(dataset) for dataset in datasets]
                weights = layer.weights
                if len(values) == len(weights) and all(
                    tuple(weight.shape) == tuple(value.shape)
                    for weight, value in zip(weights, values)
                ):
                    layer.set_weights(values)
        return model
    except Exception as second_error:
        raise LegacyKerasLoadError(
            f"无法兼容加载模型：{model_path.name}\n"
            f"标准加载：{type(err1).__name__}: {err1}\n"
            f"手工还原：{type(second_error).__name__}: {second_error}"
        ) from second_error


def prepare_keras_input(model, x2d: np.ndarray, *, prefer_conv1d: bool) -> np.ndarray:
    """Adapt a single 48-value row to Dense or Conv1D model input shape."""

    x = np.asarray(x2d, dtype=np.float32).reshape(1, -1)
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list) and shape:
        shape = shape[0]
    rank = len(shape) if isinstance(shape, tuple) else None
    if rank == 3:
        return x.reshape(1, x.shape[1], 1)
    if rank == 2:
        return x
    return x.reshape(1, x.shape[1], 1) if prefer_conv1d else x
