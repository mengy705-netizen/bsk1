from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib import cm, colors, font_manager
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.spatial import ConvexHull, cKDTree

from .geometry import WingGeometry


def configure_matplotlib(config: dict[str, Any]) -> None:
    preferred = config["visualization"].get("font_family", "Microsoft YaHei")
    font_files = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
    ]
    selected = preferred
    for font_path in font_files:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
                selected = font_manager.FontProperties(fname=str(font_path)).get_name()
                break
            except Exception:
                continue
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        selected,
        preferred,
        "Microsoft YaHei UI",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _naca_half_thickness(
    geometry: WingGeometry,
    span_mm: np.ndarray | float,
    chord_mm: np.ndarray | float,
    thickness_ratio: float,
) -> np.ndarray:
    span = np.asarray(span_mm, dtype=float)
    chord_pos = np.asarray(chord_mm, dtype=float)
    local_chord = np.asarray(geometry.local_chord(span), dtype=float)
    eta = np.clip(chord_pos / np.maximum(local_chord, 1e-9), 0.0, 1.0)
    profile = 5.0 * thickness_ratio * local_chord * (
        0.2969 * np.sqrt(np.maximum(eta, 0.0))
        - 0.1260 * eta
        - 0.3516 * eta**2
        + 0.2843 * eta**3
        - 0.1015 * eta**4
    )
    return np.maximum(profile, 0.0)


def _finite_limits(field: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    masked = np.ma.array(field, mask=~valid_mask | ~np.isfinite(field))
    finite = masked.compressed()
    if finite.size == 0:
        raise ValueError("机翼范围内的重构应变场没有有效数值。")
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.05, 1.0)
        vmin -= delta
        vmax += delta
    return finite, vmin, vmax


def _mesh_hull(mesh_data: dict[str, np.ndarray] | None) -> np.ndarray | None:
    if not mesh_data:
        return None
    vertices = np.asarray(mesh_data.get("vertices"), dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 4:
        return None
    projected = vertices[:, :2]
    try:
        hull = ConvexHull(projected)
        return projected[hull.vertices]
    except Exception:
        return None


def _field_on_vertices(
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    field: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    z_axis = np.asarray(grid_z[0, :], dtype=float)
    x_axis = np.asarray(grid_x[:, 0], dtype=float)
    safe = np.asarray(field, dtype=float)
    finite = np.isfinite(safe)
    if not finite.all():
        fill = float(np.nanmedian(safe[finite])) if finite.any() else 0.0
        safe = np.where(finite, safe, fill)
    interpolator = RegularGridInterpolator(
        (x_axis, z_axis), safe, bounds_error=False, fill_value=np.nan
    )
    values = interpolator(np.column_stack([vertices[:, 1], vertices[:, 0]]))
    if np.isnan(values).any():
        replacement = float(np.nanmedian(values)) if np.isfinite(values).any() else float(np.nanmedian(safe))
        values = np.where(np.isfinite(values), values, replacement)
    return values


def _nearest_surface_height(vertices: np.ndarray, z: np.ndarray, x: np.ndarray) -> np.ndarray:
    query_z = np.asarray(z, dtype=float).reshape(-1)
    query_x = np.asarray(x, dtype=float).reshape(-1)
    projected = vertices[:, :2]
    result = np.empty(query_z.shape[0], dtype=float)
    for index, (z_value, x_value) in enumerate(zip(query_z, query_x)):
        distance = (projected[:, 0] - z_value) ** 2 + (projected[:, 1] - x_value) ** 2
        nearest = np.argpartition(distance, min(20, len(distance) - 1))[: min(20, len(distance))]
        result[index] = float(np.max(vertices[nearest, 2]))
    return result


def _draw_parametric_wing(
    ax3d,
    geometry: WingGeometry,
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    field: np.ndarray,
    valid_mask: np.ndarray,
    cmap,
    norm,
    vis: dict[str, Any],
) -> tuple[float, callable]:
    stride = max(int(vis.get("surface_stride", 3)), 1)
    zs = grid_z[::stride, ::stride]
    xs = grid_x[::stride, ::stride]
    fs = np.asarray(field, dtype=float)[::stride, ::stride]
    ms = valid_mask[::stride, ::stride] & np.isfinite(fs)
    thickness_ratio = float(vis.get("wing_thickness_ratio", 0.12))
    vertical_exaggeration = float(vis.get("vertical_exaggeration", 1.25))
    half_t = _naca_half_thickness(geometry, zs, xs, thickness_ratio) * vertical_exaggeration
    top = np.where(ms, half_t, np.nan)
    bottom = np.where(ms, -half_t, np.nan)
    facecolors = cmap(norm(np.where(ms, fs, np.nanmin(fs))))
    facecolors[..., 3] = np.where(ms, 0.98, 0.0)
    surface_kwargs = dict(rstride=1, cstride=1, linewidth=0.12, antialiased=True, shade=False)
    ax3d.plot_surface(zs, xs, top, facecolors=facecolors, **surface_kwargs)
    lower_colors = facecolors.copy()
    lower_colors[..., :3] *= 0.70
    lower_colors[..., 3] = np.where(ms, 0.82, 0.0)
    ax3d.plot_surface(zs, xs, bottom, facecolors=lower_colors, **surface_kwargs)

    span_line = np.linspace(0.0, geometry.span_mm, 180)
    leading = np.zeros_like(span_line)
    trailing = np.asarray(geometry.local_chord(span_line), dtype=float)
    zero_top = _naca_half_thickness(geometry, span_line, leading, thickness_ratio)
    zero_trailing = _naca_half_thickness(geometry, span_line, trailing, thickness_ratio)
    ax3d.plot(span_line, leading, zero_top, color="black", linewidth=1.0)
    ax3d.plot(span_line, trailing, zero_trailing, color="black", linewidth=1.0)

    root_chord = np.linspace(0.0, geometry.root_chord_mm, 140)
    root_span = np.zeros_like(root_chord)
    root_top = _naca_half_thickness(geometry, root_span, root_chord, thickness_ratio) * vertical_exaggeration
    ax3d.plot(root_span, root_chord, root_top, color="black", linewidth=2.2)
    ax3d.plot(root_span, root_chord, -root_top, color="black", linewidth=2.2)
    ax3d.text(18.0, 0.88 * geometry.root_chord_mm, 1.45 * float(np.nanmax(root_top)), "固定端", fontsize=9, weight="bold")

    def height(z, x):
        return _naca_half_thickness(geometry, z, x, thickness_ratio) * vertical_exaggeration

    return max(float(np.nanmax(root_top)), 1.0), height


def _surface_envelope_samples(
    vertices: np.ndarray,
    upper: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse duplicate plan-view vertices to an upper/lower surface envelope."""
    vertices = np.asarray(vertices, dtype=float)
    projected = vertices[:, :2]
    extent = max(float(np.ptp(projected[:, 0])), float(np.ptp(projected[:, 1])), 1.0)
    tolerance = max(extent * 1e-7, 1e-6)
    quantized = np.rint(projected / tolerance).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1

    coords = np.zeros((count, 2), dtype=float)
    counts = np.zeros(count, dtype=float)
    np.add.at(coords[:, 0], inverse, projected[:, 0])
    np.add.at(coords[:, 1], inverse, projected[:, 1])
    np.add.at(counts, inverse, 1.0)
    coords /= np.maximum(counts[:, None], 1.0)

    if upper:
        heights = np.full(count, -np.inf, dtype=float)
        np.maximum.at(heights, inverse, vertices[:, 2])
    else:
        heights = np.full(count, np.inf, dtype=float)
        np.minimum.at(heights, inverse, vertices[:, 2])
    return coords, heights


def _make_surface_height_function(vertices: np.ndarray, upper: bool = True):
    """Return a vectorized plan-view -> thickness surface interpolator."""
    coords, heights = _surface_envelope_samples(vertices, upper=upper)
    tree = cKDTree(coords)
    linear = None
    if coords.shape[0] >= 3:
        try:
            linear = LinearNDInterpolator(coords, heights, fill_value=np.nan)
        except Exception:
            linear = None

    def height(z, x):
        z_array, x_array = np.broadcast_arrays(
            np.asarray(z, dtype=float), np.asarray(x, dtype=float)
        )
        points = np.column_stack([z_array.ravel(), x_array.ravel()])
        if linear is not None:
            values = np.asarray(linear(points), dtype=float).reshape(-1)
        else:
            values = np.full(points.shape[0], np.nan, dtype=float)
        missing = ~np.isfinite(values)
        if missing.any():
            _, nearest = tree.query(points[missing], k=1)
            values[missing] = heights[np.asarray(nearest, dtype=int)]
        return values.reshape(z_array.shape)

    return height


def _draw_mesh_wing(
    ax3d,
    mesh_data: dict[str, np.ndarray],
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    field: np.ndarray,
    valid_mask: np.ndarray,
    cmap,
    norm,
    vis: dict[str, Any],
) -> tuple[float, callable]:
    """Texture the imported 3-D wing with the same dense LC-Kriging grid as plan view.

    The old implementation averaged the reconstructed values at each original STL
    triangle and assigned one colour to that entire face.  Coarse CAD/STL files
    therefore appeared as a few large colour triangles.  Here the mesh supplies
    geometry only; colour comes from the shared dense (z, x) reconstruction grid.
    """
    vertices = np.asarray(mesh_data["vertices"], dtype=float)
    stride = max(int(vis.get("surface_stride", 3)), 1)
    zs = np.asarray(grid_z, dtype=float)[::stride, ::stride]
    xs = np.asarray(grid_x, dtype=float)[::stride, ::stride]
    fs = np.asarray(field, dtype=float)[::stride, ::stride]
    ms = np.asarray(valid_mask, dtype=bool)[::stride, ::stride] & np.isfinite(fs)

    top_height = _make_surface_height_function(vertices, upper=True)
    bottom_height = _make_surface_height_function(vertices, upper=False)
    top = np.where(ms, top_height(zs, xs), np.nan)
    bottom = np.where(ms, bottom_height(zs, xs), np.nan)

    finite_field = fs[np.isfinite(fs)]
    fill_value = float(np.nanmedian(finite_field)) if finite_field.size else 0.0
    facecolors = cmap(norm(np.where(ms, fs, fill_value)))
    facecolors[..., 3] = np.where(ms, 0.98, 0.0)
    surface_kwargs = dict(
        rstride=1, cstride=1, linewidth=0.0, antialiased=True, shade=False
    )
    ax3d.plot_surface(zs, xs, top, facecolors=facecolors, **surface_kwargs)

    lower_colors = facecolors.copy()
    lower_colors[..., :3] *= 0.72
    lower_colors[..., 3] = np.where(ms, 0.78, 0.0)
    ax3d.plot_surface(zs, xs, bottom, facecolors=lower_colors, **surface_kwargs)

    hull = _mesh_hull(mesh_data)
    if hull is not None and len(hull) >= 3:
        closed = np.vstack([hull, hull[0]])
        boundary_h = top_height(closed[:, 0], closed[:, 1])
        ax3d.plot(
            closed[:, 0], closed[:, 1], boundary_h,
            color="black", linewidth=1.0, alpha=0.9,
        )

    thickness_extent = max(float(np.ptp(vertices[:, 2])), 1.0)
    root_band = vertices[vertices[:, 0] <= np.percentile(vertices[:, 0], 2.0)]
    if root_band.shape[0]:
        ax3d.text(
            float(np.min(root_band[:, 0])),
            float(np.percentile(root_band[:, 1], 85)),
            float(np.max(root_band[:, 2]) + 0.5 * thickness_extent),
            "固定端", fontsize=9, weight="bold",
        )

    return thickness_extent, top_height


def draw_result(
    figure: Figure,
    geometry: WingGeometry,
    sensor_coords: np.ndarray,
    strain_values: np.ndarray,
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    valid_mask: np.ndarray,
    field: np.ndarray,
    load_info: dict[str, float],
    config: dict[str, Any],
    mesh_data: dict[str, np.ndarray] | None = None,
    sensor_coords_3d: np.ndarray | None = None,
) -> None:
    figure.clear()
    vis = config["visualization"]
    plan_hull = _mesh_hull(mesh_data)
    display_mask = np.asarray(valid_mask, dtype=bool)
    if plan_hull is not None:
        path = MplPath(plan_hull)
        points = np.column_stack([grid_z.ravel(), grid_x.ravel()])
        display_mask = path.contains_points(points).reshape(grid_z.shape)
    finite, vmin, vmax = _finite_limits(field, display_mask)
    cmap = cm.get_cmap(vis.get("colormap", "turbo"))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    grid_spec = figure.add_gridspec(
        3,
        2,
        width_ratios=(3.55, 1.18),
        height_ratios=(1.42, 1.0, 0.095),
        wspace=0.11,
        hspace=0.20,
    )
    ax3d = figure.add_subplot(grid_spec[0:2, 0], projection="3d")
    cbar_ax = figure.add_subplot(grid_spec[2, 0])
    ax_plan = figure.add_subplot(grid_spec[0, 1])
    ax_info = figure.add_subplot(grid_spec[1:3, 1])

    if mesh_data is not None:
        thickness_extent, surface_height = _draw_mesh_wing(
            ax3d, mesh_data, grid_z, grid_x, field, display_mask, cmap, norm, vis
        )
        mesh_vertices = np.asarray(mesh_data["vertices"], dtype=float)
        z_min, z_max = float(mesh_vertices[:, 0].min()), float(mesh_vertices[:, 0].max())
        x_min, x_max = float(mesh_vertices[:, 1].min()), float(mesh_vertices[:, 1].max())
        y_min, y_max = float(mesh_vertices[:, 2].min()), float(mesh_vertices[:, 2].max())
    else:
        thickness_extent, surface_height = _draw_parametric_wing(
            ax3d, geometry, grid_z, grid_x, field, display_mask, cmap, norm, vis
        )
        z_min, z_max = 0.0, geometry.span_mm
        x_min, x_max = 0.0, geometry.root_chord_mm
        y_min, y_max = -0.10 * geometry.root_chord_mm, 0.10 * geometry.root_chord_mm

    if vis.get("show_sensor_points", True):
        sensor_z = sensor_coords[:, 0]
        sensor_x = sensor_coords[:, 1]
        if sensor_coords_3d is not None and np.asarray(sensor_coords_3d).shape == (len(sensor_coords), 3):
            sensor_h = np.asarray(sensor_coords_3d, dtype=float)[:, 2]
        else:
            sensor_h = np.asarray(surface_height(sensor_z, sensor_x), dtype=float) + max(thickness_extent * 0.025, 0.8)
        ax3d.scatter(
            sensor_z,
            sensor_x,
            sensor_h,
            c=np.asarray(strain_values, dtype=float),
            cmap=cmap,
            norm=norm,
            s=13,
            edgecolors="white",
            linewidths=0.4,
            depthshade=False,
        )

    prediction_branch = str(load_info.get("prediction_branch", "intact"))
    damaged_branch = prediction_branch == "damaged" and all(
        key in load_info for key in ("damage_x_mm", "damage_z_mm", "damage_size_mm")
    )
    arrow_height = max(2.8 * thickness_extent, 0.20 * geometry.root_chord_mm, 45.0)
    if damaged_branch:
        load_z = float(np.clip(load_info["damage_z_mm"], z_min, z_max))
        load_x = float(np.clip(load_info["damage_x_mm"], x_min, x_max))
        load_n = 0.0
        damage_size = float(load_info["damage_size_mm"])
        load_surface_h = float(np.asarray(surface_height(np.array([load_z]), np.array([load_x]))).reshape(-1)[0])
        ax3d.scatter(
            [load_z], [load_x], [load_surface_h + max(thickness_extent * 0.03, 1.0)],
            marker="X", s=115, c="darkorange", edgecolors="white",
            linewidths=1.0, depthshade=False,
        )
        ax3d.text(
            load_z, load_x, load_surface_h + max(0.35 * arrow_height, 12.0),
            f"预测损伤 d = {damage_size:.2f} mm\n(z, x) = ({load_z:.1f}, {load_x:.1f}) mm",
            color="darkorange", fontsize=9, weight="bold", ha="center", va="bottom",
        )
        display_top = load_surface_h + max(0.55 * arrow_height, 18.0)
        ax3d.set_title("三维机翼重构应变云图与预测损伤", fontsize=13, weight="bold", pad=10)
    else:
        load_z = float(np.clip(load_info["load_z_mm"], z_min, z_max))
        load_x = float(np.clip(load_info["load_x_mm"], x_min, x_max))
        load_n = float(load_info["load_n"])
        load_surface_h = float(np.asarray(surface_height(np.array([load_z]), np.array([load_x]))).reshape(-1)[0])
        ax3d.quiver(
            load_z, load_x, load_surface_h + arrow_height,
            0.0, 0.0, -arrow_height,
            color="crimson", linewidth=3.2, arrow_length_ratio=0.16,
        )
        ax3d.scatter(
            [load_z], [load_x], [load_surface_h + max(thickness_extent * 0.03, 1.0)],
            marker="X", s=95, c="crimson", edgecolors="white",
            linewidths=0.9, depthshade=False,
        )
        ax3d.text(
            load_z, load_x, load_surface_h + arrow_height * 1.08,
            f"载荷 F = {load_n:.2f} N\n(z, x) = ({load_z:.1f}, {load_x:.1f}) mm",
            color="crimson", fontsize=9, weight="bold", ha="center", va="bottom",
        )
        display_top = load_surface_h + arrow_height * 1.25
        ax3d.set_title("三维机翼重构应变云图与预测载荷", fontsize=13, weight="bold", pad=10)
    ax3d.set_xlabel("展向 z（mm）", fontsize=10, weight="bold", labelpad=8)
    ax3d.set_ylabel("弦向 x（mm）", fontsize=10, weight="bold", labelpad=8)
    ax3d.set_zlabel("厚度方向（mm）", fontsize=9, weight="bold", labelpad=7)
    ax3d.set_xlim(z_min, z_max)
    ax3d.set_ylim(x_max, x_min)
    ax3d.set_zlim(y_min - 0.25 * thickness_extent, max(y_max + 0.5 * thickness_extent, display_top))
    ax3d.view_init(
        elev=float(vis.get("view_elevation", 25.0)),
        azim=float(vis.get("view_azimuth", -62.0)),
    )
    ax3d.set_box_aspect((max(z_max - z_min, 1.0), max(x_max - x_min, 1.0), max(0.35 * (x_max - x_min), thickness_extent)))
    ax3d.tick_params(labelsize=8, pad=1)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_alpha(0.0)
        axis._axinfo["grid"]["linewidth"] = 0.35
        axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.45)

    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    unit = vis.get("strain_unit", "")
    cbar = figure.colorbar(scalar_map, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        f"重构最大主应变 ε1（{unit}）" if unit else "重构最大主应变 ε1",
        fontsize=9.5,
        weight="bold",
        labelpad=3,
    )
    cbar.ax.tick_params(labelsize=8, pad=1)

    masked_field = np.ma.array(field, mask=~display_mask | ~np.isfinite(field))
    levels = np.linspace(vmin, vmax, int(vis.get("levels", 36)))
    ax_plan.contourf(
        grid_z,
        grid_x,
        masked_field,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        antialiased=True,
    )
    outline = plan_hull if plan_hull is not None else geometry.polygon()
    ax_plan.add_patch(
        Polygon(outline, closed=True, fill=False, edgecolor="black", linewidth=1.5, zorder=8)
    )
    if vis.get("show_sensor_points", True):
        ax_plan.scatter(
            sensor_coords[:, 0],
            sensor_coords[:, 1],
            c="white",
            s=10,
            edgecolors="black",
            linewidths=0.35,
            zorder=10,
        )
    if damaged_branch:
        ax_plan.scatter(
            [load_z], [load_x], marker="X", s=85, c="darkorange",
            edgecolors="white", linewidths=0.8, zorder=12,
        )
        ax_plan.annotate(
            f"损伤 {float(load_info['damage_size_mm']):.2f} mm",
            xy=(load_z, load_x),
            xytext=(load_z - 0.26 * max(z_max - z_min, 1.0), load_x + 0.15 * max(x_max - x_min, 1.0)),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.7, color="darkorange"),
            color="darkorange", fontsize=8, weight="bold", ha="center", va="bottom",
        )
    else:
        ax_plan.scatter(
            [load_z], [load_x], marker="X", s=75, c="crimson",
            edgecolors="white", linewidths=0.7, zorder=12,
        )
        ax_plan.annotate(
            f"{load_n:.2f} N",
            xy=(load_z, load_x),
            xytext=(load_z - 0.26 * max(z_max - z_min, 1.0), load_x + 0.15 * max(x_max - x_min, 1.0)),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.7, color="crimson"),
            color="crimson", fontsize=8, weight="bold", ha="center", va="bottom",
        )
    ax_plan.set_title("机翼俯视测量图", fontsize=10, weight="bold", pad=5)
    ax_plan.set_xlabel("展向 z（mm）", fontsize=8, weight="bold")
    ax_plan.set_ylabel("弦向 x（mm）", fontsize=8, weight="bold")
    ax_plan.set_xlim(z_min - 0.02 * (z_max - z_min), z_max + 0.02 * (z_max - z_min))
    ax_plan.set_ylim(x_min - 0.03 * (x_max - x_min), x_max + 0.03 * (x_max - x_min))
    ax_plan.set_aspect("equal", adjustable="box")
    ax_plan.tick_params(labelsize=7)

    damage_probability = load_info.get("damage_probability")
    damage_line = "模型未输出"
    if damage_probability is not None:
        damage_value = float(np.asarray(damage_probability).reshape(-1)[0])
        damage_line = f"{damage_value:.1%}"

    extra_damage_lines: list[str] = []
    if all(key in load_info for key in ("damage_x_mm", "damage_z_mm", "damage_size_mm")):
        extra_damage_lines = [
            f"损伤弦向坐标 x：{float(load_info['damage_x_mm']):.1f} mm",
            f"损伤展向坐标 z：{float(load_info['damage_z_mm']):.1f} mm",
            f"损伤尺寸：      {float(load_info['damage_size_mm']):.2f} mm",
        ]

    ax_info.axis("off")
    if damaged_branch:
        model_lines = [
            "七模型输出（有损分支）",
            "",
            f"损伤概率：      {damage_line}",
            *extra_damage_lines,
        ]
    else:
        model_lines = [
            "七模型输出（无损分支）" if prediction_branch == "intact" else "迁移模型输出",
            "",
            f"载荷大小：      {load_n:.2f} N",
            f"加载展向坐标 z：{load_z:.1f} mm",
            f"加载弦向坐标 x：{load_x:.1f} mm",
            f"损伤概率：      {damage_line}",
            *extra_damage_lines,
        ]
    info_lines = [
        *model_lines,
        "",
        "重构应变场",
        f"最小值：{float(np.nanmin(finite)):.3g} {unit}",
        f"最大值：{float(np.nanmax(finite)):.3g} {unit}",
        f"传感器数量：{sensor_coords.shape[0]}",
        f"三维几何：{'新机翼网格' if mesh_data is not None else '参数化机翼'}",
    ]
    ax_info.text(
        0.03,
        0.96,
        "\n".join(info_lines),
        transform=ax_info.transAxes,
        ha="left",
        va="top",
        fontsize=8.9,
        linespacing=1.34,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="white", edgecolor="0.55", alpha=0.96),
    )
    ax_info.text(
        0.03,
        0.04,
        "彩色云图由稀疏应变测点重构得到；\n红色箭头表示迁移模型预测的加载位置、方向和大小。",
        transform=ax_info.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.7,
        color="0.30",
        linespacing=1.35,
    )

    figure.subplots_adjust(left=0.035, right=0.985, top=0.94, bottom=0.07)


def create_figure(config: dict[str, Any]) -> Figure:
    configure_matplotlib(config)
    return Figure(figsize=(13.7, 7.45), dpi=int(config["visualization"].get("dpi", 140)))
