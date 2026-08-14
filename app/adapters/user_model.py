"""
Replace this template with the actual transferred AE-FST model loading code.

Required public interface:
    predict(strain_values, config) -> dict or array

The 48 values follow config.json -> sensors -> input_order.
Recommended return keys:
    load_x_mm: chordwise load coordinate
    load_z_mm: spanwise load coordinate
    load_n: load magnitude in N
Optional keys:
    damage_probability, damage_x_mm, damage_z_mm, damage_size_mm
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


_MODEL_CACHE: dict[str, Any] = {}


def predict(strain_values: np.ndarray, config: dict) -> dict[str, float]:
    values = np.asarray(strain_values, dtype=np.float32).reshape(1, 48)

    # Example integration pattern (edit paths and preprocessing to match your code):
    # import torch
    # base = Path(__file__).resolve().parent
    # encoder = _MODEL_CACHE.setdefault(
    #     "encoder", torch.jit.load(str(base / "models" / "ae_encoder.ts"), map_location="cpu")
    # )
    # regressor = _MODEL_CACHE.setdefault(
    #     "regressor", torch.jit.load(str(base / "models" / "transferred_load_regressor.ts"), map_location="cpu")
    # )
    # scaler = ...  # Load the same scaler used during transfer/fine-tuning.
    # x = scaler.transform(values)
    # with torch.no_grad():
    #     z = encoder(torch.tensor(x, dtype=torch.float32))
    #     output = regressor(z).cpu().numpy().reshape(-1)
    # return {"load_x_mm": output[0], "load_z_mm": output[1], "load_n": output[2]}

    raise RuntimeError(
        "adapters/user_model.py is a template. Insert your transferred-model loading "
        "and preprocessing code, or select Demo mode in the software."
    )
