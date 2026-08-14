from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import struct
from typing import Any

import numpy as np


class MeshFormatError(ValueError):
    pass


@dataclass
class PreparedMesh:
    npz_path: Path
    original_copy_path: Path
    metadata: dict[str, Any]


_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise MeshFormatError("三维模型顶点数据无效。")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] < 1:
        raise MeshFormatError("三维模型必须包含三角形面片。")
    if not np.all(np.isfinite(vertices)):
        raise MeshFormatError("三维模型顶点中包含 NaN 或无穷大。")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise MeshFormatError("三维模型面片索引超出顶点范围。")
    return vertices, faces


def _load_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    size = path.stat().st_size
    if size < 84:
        raise MeshFormatError("STL 文件过小或已损坏。")
    with path.open("rb") as f:
        f.read(80)
        raw_count = f.read(4)
        if len(raw_count) != 4:
            raise MeshFormatError("STL 文件头不完整。")
        count = struct.unpack("<I", raw_count)[0]
        expected = 84 + count * 50
        if expected != size:
            raise MeshFormatError("不是标准二进制 STL。")
        vertices = np.empty((count * 3, 3), dtype=np.float32)
        faces = np.arange(count * 3, dtype=np.int64).reshape(count, 3)
        for i in range(count):
            block = f.read(50)
            if len(block) != 50:
                raise MeshFormatError("STL 三角面片数据不完整。")
            values = struct.unpack("<12fH", block)
            vertices[i * 3:(i + 1) * 3] = np.asarray(values[3:12], dtype=np.float32).reshape(3, 3)
    return _validate_mesh(vertices, faces)


def _load_ascii_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    current: list[int] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0].lower() == "vertex":
                    try:
                        vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    except ValueError as exc:
                        raise MeshFormatError("ASCII STL 中存在无效顶点。") from exc
                    vertices.append(vertex)
                    current.append(len(vertices) - 1)
                    if len(current) == 3:
                        faces.append(current)
                        current = []
    except OSError as exc:
        raise MeshFormatError(f"无法读取 STL 文件：{exc}") from exc
    if not faces:
        raise MeshFormatError("ASCII STL 中没有找到三角面片。")
    return _validate_mesh(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64))


def _load_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Binary STL has a reliable size signature. If that test fails, parse as ASCII.
    try:
        return _load_binary_stl(path)
    except MeshFormatError as binary_error:
        try:
            return _load_ascii_stl(path)
        except MeshFormatError as ascii_error:
            raise MeshFormatError(
                f"STL 文件无法识别。二进制读取结果：{binary_error}；ASCII读取结果：{ascii_error}"
            ) from ascii_error


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if parts[0] == "v" and len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "f" and len(parts) >= 4:
                    polygon: list[int] = []
                    for token in parts[1:]:
                        idx_text = token.split("/")[0]
                        idx = int(idx_text)
                        idx = len(vertices) + idx if idx < 0 else idx - 1
                        polygon.append(idx)
                    for j in range(1, len(polygon) - 1):
                        faces.append([polygon[0], polygon[j], polygon[j + 1]])
    except (OSError, ValueError) as exc:
        raise MeshFormatError(f"OBJ 文件读取失败：{exc}") from exc
    return _validate_mesh(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64))


def _load_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        lines = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                clean = line.split("#", 1)[0].strip()
                if clean:
                    lines.append(clean)
        if not lines or lines[0] != "OFF":
            raise MeshFormatError("OFF 文件头无效。")
        nv, nf, _ = map(int, lines[1].split()[:3])
        vertices = np.asarray([[float(v) for v in lines[i + 2].split()[:3]] for i in range(nv)], dtype=float)
        faces: list[list[int]] = []
        start = 2 + nv
        for i in range(nf):
            values = [int(v) for v in lines[start + i].split()]
            n = values[0]
            polygon = values[1:1 + n]
            for j in range(1, len(polygon) - 1):
                faces.append([polygon[0], polygon[j], polygon[j + 1]])
    except (OSError, ValueError, IndexError) as exc:
        if isinstance(exc, MeshFormatError):
            raise
        raise MeshFormatError(f"OFF 文件读取失败：{exc}") from exc
    return _validate_mesh(vertices, np.asarray(faces, dtype=np.int64))


def _load_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with path.open("r", encoding="utf-8", errors="strict") as f:
            header: list[str] = []
            while True:
                line = f.readline()
                if not line:
                    raise MeshFormatError("PLY 文件头不完整。")
                header.append(line.strip())
                if line.strip() == "end_header":
                    break
            if not header or header[0] != "ply":
                raise MeshFormatError("PLY 文件头无效。")
            if "format ascii 1.0" not in header:
                raise MeshFormatError("内置读取器目前支持 ASCII PLY；二进制 PLY 请转换为 STL。")
            vertex_count = 0
            face_count = 0
            for line in header:
                parts = line.split()
                if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
                    vertex_count = int(parts[2])
                elif len(parts) == 3 and parts[0] == "element" and parts[1] == "face":
                    face_count = int(parts[2])
            vertices = np.asarray(
                [[float(v) for v in f.readline().split()[:3]] for _ in range(vertex_count)],
                dtype=float,
            )
            faces: list[list[int]] = []
            for _ in range(face_count):
                values = [int(v) for v in f.readline().split()]
                n = values[0]
                polygon = values[1:1 + n]
                for j in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[j], polygon[j + 1]])
    except (OSError, UnicodeDecodeError, ValueError, IndexError) as exc:
        if isinstance(exc, MeshFormatError):
            raise
        raise MeshFormatError(f"PLY 文件读取失败：{exc}") from exc
    return _validate_mesh(vertices, np.asarray(faces, dtype=np.int64))


def _load_with_optional_trimesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import trimesh
    except ImportError as exc:
        raise MeshFormatError(
            "GLB/GLTF 需要可选组件 trimesh。当前无需该组件即可直接读取 STL、OBJ、OFF 和 ASCII PLY。"
        ) from exc
    try:
        loaded = trimesh.load(path, force=None, process=False)
        if isinstance(loaded, trimesh.Scene):
            geometries = [geometry for geometry in loaded.geometry.values() if hasattr(geometry, "faces")]
            if not geometries:
                raise MeshFormatError("三维文件中没有可用的三角网格。")
            loaded = trimesh.util.concatenate(geometries)
        return _validate_mesh(np.asarray(loaded.vertices, dtype=float), np.asarray(loaded.faces, dtype=np.int64))
    except MeshFormatError:
        raise
    except Exception as exc:
        raise MeshFormatError(f"三维模型读取失败：{exc}") from exc


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return _load_stl(path)
    if suffix == ".obj":
        return _load_obj(path)
    if suffix == ".off":
        return _load_off(path)
    if suffix == ".ply":
        return _load_ascii_ply(path)
    if suffix in {".glb", ".gltf"}:
        return _load_with_optional_trimesh(path)
    raise MeshFormatError(
        "当前内置读取器支持 STL、OBJ、OFF 和 ASCII PLY。STEP/IGES 请先由 CAD 软件导出为 STL。"
    )


def _estimate_chord(vertices: np.ndarray, span_value: float, span_tolerance: float) -> float:
    mask = np.abs(vertices[:, 0] - span_value) <= span_tolerance
    if int(mask.sum()) < 10:
        order = np.argsort(np.abs(vertices[:, 0] - span_value))
        count = max(10, int(0.05 * len(order)))
        mask = np.zeros(vertices.shape[0], dtype=bool)
        mask[order[:count]] = True
    values = vertices[mask, 1]
    return float(np.percentile(values, 99.0) - np.percentile(values, 1.0))


def prepare_mesh(
    source_path: str | Path,
    output_dir: str | Path,
    span_axis: str = "Z",
    chord_axis: str = "X",
    unit: str = "m",
    reverse_span: bool = False,
    max_faces: int = 60000,
) -> PreparedMesh:
    source = Path(source_path)
    if not source.exists():
        raise MeshFormatError(f"未找到三维机翼文件：{source}")
    span_axis = span_axis.upper()
    chord_axis = chord_axis.upper()
    if span_axis not in _AXIS_INDEX or chord_axis not in _AXIS_INDEX:
        raise MeshFormatError("展向轴和弦向轴必须是 X、Y 或 Z。")
    if span_axis == chord_axis:
        raise MeshFormatError("展向轴和弦向轴不能相同。")
    thickness_axis = ({"X", "Y", "Z"} - {span_axis, chord_axis}).pop()

    raw_vertices, faces = _load_mesh(source)
    scale = {"mm": 1.0, "m": 1000.0, "cm": 10.0}.get(unit.lower())
    if scale is None:
        raise MeshFormatError("模型单位只能选择 mm、cm 或 m。")

    canonical = np.column_stack(
        [
            raw_vertices[:, _AXIS_INDEX[span_axis]],
            raw_vertices[:, _AXIS_INDEX[chord_axis]],
            raw_vertices[:, _AXIS_INDEX[thickness_axis]],
        ]
    ) * scale
    if reverse_span:
        canonical[:, 0] *= -1.0

    canonical[:, 0] -= float(np.min(canonical[:, 0]))
    canonical[:, 1] -= float(np.min(canonical[:, 1]))
    canonical[:, 2] -= float((np.max(canonical[:, 2]) + np.min(canonical[:, 2])) / 2.0)

    span_mm = float(np.max(canonical[:, 0]) - np.min(canonical[:, 0]))
    chord_extent = float(np.max(canonical[:, 1]) - np.min(canonical[:, 1]))
    thickness_mm = float(np.max(canonical[:, 2]) - np.min(canonical[:, 2]))
    if span_mm <= 0 or chord_extent <= 0 or thickness_mm <= 0:
        raise MeshFormatError("三维模型在一个或多个方向上的尺寸为零，无法建立机翼坐标系。")
    tolerance = max(span_mm * 0.04, 1e-6)
    root_chord_mm = _estimate_chord(canonical, 0.0, tolerance)
    tip_chord_mm = _estimate_chord(canonical, span_mm, tolerance)
    if root_chord_mm <= 0 or tip_chord_mm <= 0:
        root_chord_mm = chord_extent
        tip_chord_mm = chord_extent

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original_copy = output / f"original_geometry{source.suffix.lower()}"
    shutil.copy2(source, original_copy)

    display_faces = faces
    if faces.shape[0] > max_faces:
        indices = np.linspace(0, faces.shape[0] - 1, max_faces, dtype=int)
        display_faces = faces[indices]

    npz_path = output / "wing_mesh.npz"
    np.savez_compressed(
        npz_path,
        vertices=canonical.astype(np.float32),
        faces=display_faces.astype(np.int32),
    )
    metadata: dict[str, Any] = {
        "source_filename": source.name,
        "span_axis": span_axis,
        "chord_axis": chord_axis,
        "thickness_axis": thickness_axis,
        "source_unit": unit,
        "reverse_span": bool(reverse_span),
        "vertex_count": int(canonical.shape[0]),
        "face_count_original": int(faces.shape[0]),
        "face_count_display": int(display_faces.shape[0]),
        "span_mm": span_mm,
        "root_chord_mm": root_chord_mm,
        "tip_chord_mm": tip_chord_mm,
        "maximum_chord_extent_mm": chord_extent,
        "thickness_mm": thickness_mm,
        "reader": "内置网格读取器",
    }
    return PreparedMesh(npz_path=npz_path, original_copy_path=original_copy, metadata=metadata)


def load_prepared_mesh(path: str | Path | None) -> dict[str, np.ndarray] | None:
    if not path:
        return None
    mesh_path = Path(path)
    if not mesh_path.exists():
        return None
    with np.load(mesh_path) as payload:
        vertices = np.asarray(payload["vertices"], dtype=float)
        faces = np.asarray(payload["faces"], dtype=np.int64)
    return {"vertices": vertices, "faces": faces}
