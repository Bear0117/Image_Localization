"""Utilities for embedding Open3D visualizations inside Gradio outputs.

The helpers in this module generate either an interactive Plotly HTML snippet
or an offscreen-rendered PNG/GIF that can be displayed in the right-side Gradio
column. Camera poses are visualized with either the built-in coordinate frame
or a simple frustum mesh. All camera meshes are transformed by ``T_world_cam``
so that the visualizations stay in world coordinates.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Iterable, List, Optional, Tuple

import numpy as np
import open3d as o3d


CAMERA_COLOR = (1.0, 0.2, 0.2)


def _make_frustum_mesh(size: float = 0.4, color: Tuple[float, float, float] = CAMERA_COLOR) -> o3d.geometry.TriangleMesh:
    """Create a simple pyramid frustum pointing along +Z (Open3D camera convention).

    The frustum is centered at the origin before transformation; callers should
    apply ``transform`` with the desired camera pose. The mesh is watertight so
    that offscreen rendering can generate sensible silhouettes.
    """

    # 5 vertices: camera center + 4 corners on near plane
    half = size * 0.5
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [-half, -half, size],
            [half, -half, size],
            [half, half, size],
            [-half, half, size],
        ],
        dtype=np.float64,
    )

    triangles = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [1, 2, 3],
            [1, 3, 4],
        ],
        dtype=np.int32,
    )

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(triangles)
    )
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def make_camera_mesh(
    T_world_cam: np.ndarray,
    size: float = 0.8,
    mode: str = "frame",
    color: Tuple[float, float, float] = CAMERA_COLOR,
) -> o3d.geometry.TriangleMesh:
    """Create a camera visualization mesh transformed by ``T_world_cam``.

    Parameters
    ----------
    T_world_cam: np.ndarray
        4x4 homogeneous transform for the camera pose in world coordinates.
    size: float
        Overall scale for the camera marker.
    mode: str
        Either ``"frame"`` (Open3D coordinate frame) or ``"frustum"`` for a
        simple pyramid mesh.
    color: tuple
        RGB triple in 0~1 range when ``mode == "frustum"``.
    """

    if mode not in {"frame", "frustum"}:
        raise ValueError("mode must be 'frame' or 'frustum'")

    if mode == "frame":
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    else:
        mesh = _make_frustum_mesh(size=size, color=color)

    mesh.transform(T_world_cam.astype(np.float64))
    return mesh


def load_scene_geometry(path: Optional[Path]) -> Optional[o3d.geometry.Geometry]:
    """Load a scene mesh/point cloud if ``path`` is provided and exists."""

    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None

    ext = path.suffix.lower()
    if ext in {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"}:
        geom = o3d.io.read_point_cloud(str(path))
    else:
        geom = o3d.io.read_triangle_mesh(str(path))

    if isinstance(geom, o3d.geometry.TriangleMesh) and len(geom.triangles) == 0:
        # convert to point cloud if mesh faces are missing
        geom = geom.sample_points_uniformly(5000)
    return geom


def geoms_to_plotly_html(
    geoms: Iterable[o3d.geometry.Geometry],
    width: int = 900,
    height: int = 700,
) -> str:
    """Convert geometries into a Plotly HTML snippet for Gradio HTML blocks."""

    from plotly.io import to_html

    fig = o3d.visualization.draw_plotly(list(geoms), width=width, height=height)
    return to_html(fig, full_html=False, include_plotlyjs="cdn")


def render_geoms_png(
    geoms: List[o3d.geometry.Geometry],
    save_path: Path,
    bg_color: Tuple[float, float, float, float] = (1, 1, 1, 1),
    width: int = 1280,
    height: int = 960,
) -> None:
    """Render geometries offscreen to ``save_path`` as a fallback snapshot."""

    from open3d.visualization import rendering

    bbox_min = None
    bbox_max = None
    for g in geoms:
        b = g.get_axis_aligned_bounding_box()
        mn = b.get_min_bound()
        mx = b.get_max_bound()
        if bbox_min is None:
            bbox_min, bbox_max = mn, mx
        else:
            bbox_min = np.minimum(bbox_min, mn)
            bbox_max = np.maximum(bbox_max, mx)

    bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
    center = bbox.get_center()
    radius = float(np.linalg.norm(bbox.get_extent())) * 1.2 + 1e-3

    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(list(bg_color))

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    for idx, g in enumerate(geoms):
        renderer.scene.add_geometry(f"g{idx}", g, mat)

    eye = center + np.array([radius, radius, radius])
    renderer.scene.camera.look_at(center, eye, np.array([0.0, 1.0, 0.0]))
    img = renderer.render_to_image()
    o3d.io.write_image(str(save_path), img)
    renderer.release()


def prepare_gradio_visuals(
    T_world_cam: np.ndarray,
    scene_path: Optional[Path] = None,
    camera_mode: str = "frame",
    camera_size: float = 0.6,
    width: int = 900,
    height: int = 700,
) -> Tuple[str, Path]:
    """Create both an interactive Plotly HTML and a static PNG for Gradio.

    Returns a tuple of (html_snippet, png_path). The PNG is saved to a temporary
    directory so that callers can feed it into an Image component when
    JavaScript is disabled.
    """

    geoms: List[o3d.geometry.Geometry] = [
        make_camera_mesh(T_world_cam, size=camera_size, mode=camera_mode)
    ]
    scene_geom = load_scene_geometry(scene_path)
    if scene_geom is not None:
        geoms.append(scene_geom)

    try:
        html_snippet = geoms_to_plotly_html(geoms, width=width, height=height)
    except Exception:
        # Plotly is optional; still provide a minimal HTML with the static image
        html_snippet = "<p>Plotly visualization unavailable; showing snapshot instead.</p>"

    tmp_png = Path(tempfile.mkdtemp()) / "scene.png"
    render_geoms_png(geoms, tmp_png)
    return html_snippet, tmp_png
