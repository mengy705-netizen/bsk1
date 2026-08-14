from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


class ModelInterfaceError(RuntimeError):
    pass


def _as_prediction_dict(raw: Any, output_order: list[str]) -> dict[str, float]:
    if isinstance(raw, dict):
        result = dict(raw)
    else:
        array = np.asarray(raw, dtype=float).reshape(-1)
        if array.size < len(output_order):
            raise ModelInterfaceError(
                f"模型返回了{array.size}个数值，但至少需要"
                f"{len(output_order)}个输出：{output_order}。"
            )
        result = {name: float(array[index]) for index, name in enumerate(output_order)}

    aliases = {
        "load_x": "load_x_mm",
        "load_y": "load_x_mm",
        "load_z": "load_z_mm",
        "load_size": "load_n",
        "load_magnitude": "load_n",
    }
    for source, target in aliases.items():
        if target not in result and source in result:
            result[target] = result[source]

    missing = [key for key in ("load_x_mm", "load_z_mm", "load_n") if key not in result]
    if missing:
        raise ModelInterfaceError(f"模型预测结果缺少必要字段：{missing}")

    for key in list(result):
        value = result[key]
        if isinstance(value, (np.ndarray, list, tuple)) and np.asarray(value).size == 1:
            result[key] = float(np.asarray(value).reshape(-1)[0])
    return result


class ModelRunner:
    """Run active AE-FST, manual bundles, demo, Python, joblib, or TorchScript models."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def predict(
        self,
        strain_values: np.ndarray,
        sensor_coords: np.ndarray,
        mode: str,
        model_path: str | None = None,
    ) -> dict[str, float]:
        values = np.asarray(strain_values, dtype=float).reshape(-1)
        if values.size != sensor_coords.shape[0]:
            raise ModelInterfaceError(
                f"应输入{sensor_coords.shape[0]}个应变值，实际得到{values.size}个。"
            )

        if mode == "Demo":
            return self._demo_predict(values, sensor_coords)
        if mode == "Active External":
            from .external_model_registry import predict_active_external_model

            base_dir = Path(self.config.get("base_dir", Path(__file__).resolve().parents[1]))
            raw = predict_active_external_model(base_dir, values, self.config)
            return _as_prediction_dict(raw, self.config["model"]["output_order"])
        if mode == "Active AE-FST":
            from .model_registry import read_active_manifest

            base_dir = Path(self.config.get("base_dir", Path(__file__).resolve().parents[1]))
            manifest = read_active_manifest(base_dir)
            if not manifest or not manifest.get("bundle_path_resolved"):
                raise ModelInterfaceError("软件当前没有已激活的迁移模型，请先在“迁移学习与模型更新”页完成更新。")
            model_type = str(manifest.get("model_type", "AE-FST"))
            if model_type == "Keras-AE-guided-KD-v3":
                from .keras_v3_backend import predict_keras_v3_bundle

                raw = predict_keras_v3_bundle(manifest["bundle_path_resolved"], values)
            else:
                from .transfer_learning import predict_aefst_bundle

                raw = predict_aefst_bundle(manifest["bundle_path_resolved"], values)
            return _as_prediction_dict(raw, self.config["model"]["output_order"])
        if not model_path:
            raise ModelInterfaceError("请选择模型文件或模型适配器文件。")
        path = Path(model_path)
        if not path.exists():
            raise ModelInterfaceError(f"未找到模型文件：{path}")

        if mode == "AE-FST Bundle":
            if path.suffix.lower() == ".json":
                from .keras_v3_backend import predict_keras_v3_bundle

                raw = predict_keras_v3_bundle(path, values)
            else:
                from .transfer_learning import predict_aefst_bundle

                raw = predict_aefst_bundle(path, values)
        elif mode == "Custom Python":
            raw = self._run_custom_python(path, values)
        elif mode == "Joblib / sklearn":
            raw = self._run_joblib(path, values)
        elif mode == "TorchScript":
            raw = self._run_torchscript(path, values)
        else:
            raise ModelInterfaceError(f"不支持的模型类型：{mode}")

        return _as_prediction_dict(raw, self.config["model"]["output_order"])

    def _demo_predict(self, values: np.ndarray, sensor_coords: np.ndarray) -> dict[str, float]:
        # Weighted strain-energy centroid provides a deterministic visual demo only.
        baseline = float(np.median(values))
        weights = np.abs(values - baseline)
        if not np.any(np.isfinite(weights)):
            raise ModelInterfaceError("应变向量中没有有效数值。")
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0) + 1e-12
        z_mm = float(np.average(sensor_coords[:, 0], weights=weights))
        x_mm = float(np.average(sensor_coords[:, 1], weights=weights))
        load_gain = float(self.config["model"].get("demo_load_gain", 0.55))
        load_n = float(np.percentile(weights, 90) * load_gain)
        damage_probability = float(np.clip(np.std(values) / (np.mean(np.abs(values)) + 1e-9), 0, 1))
        return {
            "load_x_mm": x_mm,
            "load_z_mm": z_mm,
            "load_n": load_n,
            "damage_probability": damage_probability,
            "prediction_mode": "demo",
        }

    def _run_custom_python(self, path: Path, values: np.ndarray) -> Any:
        spec = importlib.util.spec_from_file_location("wing_user_model", path)
        if spec is None or spec.loader is None:
            raise ModelInterfaceError(f"无法导入Python模型适配器：{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predict_fn = getattr(module, "predict", None)
        if not callable(predict_fn):
            raise ModelInterfaceError(
                "自定义Python适配器必须定义 predict(strain_values, config) 函数。"
            )
        return predict_fn(values.copy(), self.config)

    def _run_joblib(self, path: Path, values: np.ndarray) -> Any:
        import joblib

        model = joblib.load(path)
        if isinstance(model, dict) and callable(model.get("predict")):
            return model["predict"](values.reshape(1, -1))
        if isinstance(model, dict) and "model" in model:
            estimator = model["model"]
            x = values.reshape(1, -1)
            scaler = model.get("scaler")
            if scaler is not None:
                x = scaler.transform(x)
            return estimator.predict(x)
        if not hasattr(model, "predict"):
            raise ModelInterfaceError("加载的Joblib对象没有predict方法。")
        return model.predict(values.reshape(1, -1))

    def _run_torchscript(self, path: Path, values: np.ndarray) -> Any:
        import torch

        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
        tensor = torch.as_tensor(values, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            output = model(tensor)
        if isinstance(output, dict):
            converted: dict[str, Any] = {}
            for key, value in output.items():
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                converted[str(key)] = value
            return converted
        return output.detach().cpu().numpy()
