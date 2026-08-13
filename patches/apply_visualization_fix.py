from __future__ import annotations

import sys
from pathlib import Path


def patch(path: Path) -> None:
    text = path.read_text(encoding='utf-8')

    text = text.replace(
        'from scipy.interpolate import RegularGridInterpolator\nfrom scipy.spatial import ConvexHull\n',
        'from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator\n'
        'from scipy.spatial import ConvexHull, cKDTree\n',
    )

    start = text.index('def _draw_mesh_wing(')
    end = text.index('\n\ndef draw_result(', start)

    replacement = r'''def _surface_envelope_samples(
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
'''
    text = text[:start] + replacement + text[end:]

    old_call = '''        thickness_extent, surface_height = _draw_mesh_wing(\n            ax3d, mesh_data, grid_z, grid_x, field, cmap, norm\n        )'''
    new_call = '''        thickness_extent, surface_height = _draw_mesh_wing(\n            ax3d, mesh_data, grid_z, grid_x, field, display_mask, cmap, norm, vis\n        )'''
    if old_call not in text:
        raise RuntimeError('Expected _draw_mesh_wing call was not found; source version changed.')
    text = text.replace(old_call, new_call, 1)

    path.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    target = Path(sys.argv[1] if len(sys.argv) > 1 else 'slim_project/core/visualization.py')
    patch(target)
    print(f'Applied dense 3-D LC-Kriging texture mapping fix: {target}')
