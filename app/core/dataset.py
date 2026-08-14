from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


class DatasetFormatError(ValueError):
    """Raised when a transfer-learning dataset cannot be interpreted safely."""


@dataclass
class DomainDataset:
    name: str
    path: Path
    features: np.ndarray
    feature_columns: list[str]
    load_targets: np.ndarray | None
    load_target_columns: list[str]
    damage_targets: np.ndarray | None
    damage_target_columns: list[str]

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])


_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "load_x_mm": (
        "load_x_mm",
        "load_x",
        "force_x",
        "loading_x",
        "load_pos_x",
        "load_position_x",
        "x_load",
    ),
    "load_z_mm": (
        "load_z_mm",
        "load_z",
        "load_y",
        "force_z",
        "force_y",
        "loading_z",
        "loading_y",
        "load_pos_z",
        "load_position_z",
        "z_load",
        "y_load",
    ),
    "load_n": (
        "load_n",
        "load_size",
        "load_magnitude",
        "force",
        "force_n",
        "load",
        "f",
    ),
    "damage_x_mm": (
        "damage_x_mm",
        "damage_x",
        "hole_x_mm",
        "hole_x",
        "defect_x",
        "crack_x",
    ),
    "damage_z_mm": (
        "damage_z_mm",
        "damage_z",
        "hole_z_mm",
        "hole_z",
        "hole_y",
        "defect_z",
        "defect_y",
        "crack_z",
    ),
    "damage_size_mm": (
        "damage_size_mm",
        "damage_size",
        "hole_d_mm",
        "hole_d",
        "hole_diameter",
        "diameter",
        "damage_diameter",
        "hole_r",
        "hole_radius",
    ),
}

_FEATURE_PATTERNS = (
    re.compile(r"^(?:strain|sensor|channel|ch|s|ep1|eps|epsilon)[_\- ]*0*(\d+)$", re.I),
    re.compile(r"^0*(\d+)[_\- ]*(?:strain|sensor|channel|ch|s|ep1|eps|epsilon)$", re.I),
)


def _normalise_column_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[\s\-./\\]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _read_delimited(path: Path) -> pd.DataFrame:
    # First try to preserve a real header. sep=None lets pandas sniff comma, tab,
    # semicolon, or whitespace-delimited files.
    try:
        frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, sep=None, engine="python", encoding="gbk")
    except Exception:
        frame = pd.read_csv(path, header=None, sep=None, engine="python", encoding="utf-8-sig")

    # A headerless numeric file is often misread with the first numeric row used
    # as column names. Detect that case and reread it without a header.
    def numeric_header(columns: Iterable[Any]) -> bool:
        converted = 0
        total = 0
        for item in columns:
            total += 1
            try:
                float(str(item).strip())
                converted += 1
            except ValueError:
                pass
        return total > 0 and converted / total > 0.8

    if numeric_header(frame.columns):
        try:
            frame = pd.read_csv(path, header=None, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(path, header=None, sep=None, engine="python", encoding="gbk")
    return frame


def read_table(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise DatasetFormatError(f"未找到数据文件：{file_path}")
    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        frame = pd.read_excel(file_path, engine="openpyxl")
    elif suffix == ".xls":
        frame = pd.read_excel(file_path)
    elif suffix in {".csv", ".txt", ".dat", ".tsv"}:
        frame = _read_delimited(file_path)
    else:
        raise DatasetFormatError(
            f"不支持的数据格式 {suffix or '（无扩展名）'}。请使用 CSV、TXT、TSV、XLS 或 XLSX。"
        )
    if frame.empty:
        raise DatasetFormatError(f"数据文件为空：{file_path.name}")
    # Remove fully empty rows/columns but preserve partially missing columns for
    # explicit validation below.
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        raise DatasetFormatError(f"数据文件没有有效内容：{file_path.name}")
    return frame


def _resolve_label_column(frame: pd.DataFrame, canonical: str) -> str | None:
    normalised = {_normalise_column_name(column): str(column) for column in frame.columns}
    for alias in _LABEL_ALIASES[canonical]:
        match = normalised.get(_normalise_column_name(alias))
        if match is not None:
            return match
    return None


def _detect_feature_columns(
    frame: pd.DataFrame,
    feature_count: int,
    configured_columns: list[str] | None = None,
) -> list[str]:
    if configured_columns:
        missing = [column for column in configured_columns if column not in frame.columns]
        if missing:
            raise DatasetFormatError(f"配置指定的应变列不存在：{missing}")
        if len(configured_columns) != feature_count:
            raise DatasetFormatError(
                f"配置指定了 {len(configured_columns)} 个应变列，但软件要求 {feature_count} 个。"
            )
        return list(configured_columns)

    patterned: list[tuple[int, str]] = []
    for column in frame.columns:
        name = _normalise_column_name(column)
        for pattern in _FEATURE_PATTERNS:
            matched = pattern.match(name)
            if matched:
                patterned.append((int(matched.group(1)), str(column)))
                break
    if len(patterned) >= feature_count:
        patterned.sort(key=lambda item: item[0])
        unique: list[str] = []
        seen: set[str] = set()
        for _, column in patterned:
            if column not in seen:
                unique.append(column)
                seen.add(column)
        if len(unique) >= feature_count:
            return unique[:feature_count]

    label_columns = {
        resolved
        for key in _LABEL_ALIASES
        if (resolved := _resolve_label_column(frame, key)) is not None
    }
    numeric_candidates: list[str] = []
    for column in frame.columns:
        if str(column) in label_columns:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        finite_ratio = float(converted.notna().mean())
        if finite_ratio >= 0.95:
            numeric_candidates.append(str(column))
    if len(numeric_candidates) < feature_count:
        raise DatasetFormatError(
            f"只能识别到 {len(numeric_candidates)} 个数值型应变候选列，少于所需的 {feature_count} 个。"
            "建议将应变列命名为 strain_1～strain_48，或在 config.json 中明确 feature_columns。"
        )
    return numeric_candidates[:feature_count]


def _extract_targets(frame: pd.DataFrame, canonical_names: tuple[str, ...]) -> tuple[np.ndarray | None, list[str]]:
    columns: list[str] = []
    for canonical in canonical_names:
        column = _resolve_label_column(frame, canonical)
        if column is None:
            return None, []
        columns.append(column)
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return values, columns


def load_domain_dataset(
    path: str | Path,
    name: str,
    feature_count: int = 48,
    configured_columns: list[str] | None = None,
) -> DomainDataset:
    file_path = Path(path)
    frame = read_table(file_path)
    feature_columns = _detect_feature_columns(frame, feature_count, configured_columns)
    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    invalid_rows = ~np.all(np.isfinite(features), axis=1)
    if invalid_rows.any():
        count = int(invalid_rows.sum())
        raise DatasetFormatError(
            f"{name} 中有 {count} 行应变数据包含空值、非数值或无穷大。请清理后重新导入。"
        )
    if features.shape[0] < 2:
        raise DatasetFormatError(f"{name} 至少需要 2 行样本，当前只有 {features.shape[0]} 行。")

    load_targets, load_columns = _extract_targets(
        frame, ("load_x_mm", "load_z_mm", "load_n")
    )
    damage_targets, damage_columns = _extract_targets(
        frame, ("damage_x_mm", "damage_z_mm", "damage_size_mm")
    )
    return DomainDataset(
        name=name,
        path=file_path,
        features=features.astype(np.float32, copy=False),
        feature_columns=feature_columns,
        load_targets=load_targets.astype(np.float32, copy=False) if load_targets is not None else None,
        load_target_columns=load_columns,
        damage_targets=(
            damage_targets.astype(np.float32, copy=False) if damage_targets is not None else None
        ),
        damage_target_columns=damage_columns,
    )


def finite_target_rows(values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=bool)
    return np.all(np.isfinite(values), axis=1)
