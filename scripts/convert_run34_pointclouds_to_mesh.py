# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy>=1.26",
#   "open3d==0.19.0",
#   "scipy>=1.13",
#   "trimesh>=4.4",
# ]
# ///
"""Convert the comparable Run 34 point-cloud GLBs into colored BPA meshes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import trimesh


DEFAULT_INPUT_ROOT = Path(
    "downloads/kaggle_run34_qualitative_3d_exports/"
    "outputs/run_34_qualitative_3d_exports"
)
SOURCE_GLOBS = ("01_*.glb", "02_*.glb", "03_*.glb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create visualization-only Ball Pivoting meshes from the three "
            "comparable Run 34 point-cloud exports."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to <input-root>/meshes_ball_pivoting.",
    )
    parser.add_argument("--normal-radius-factor", type=float, default=6.0)
    parser.add_argument(
        "--ball-radius-factors",
        type=float,
        nargs="+",
        default=(1.5, 2.5, 4.0, 6.0),
    )
    parser.add_argument("--max-edge-factor", type=float, default=8.0)
    parser.add_argument("--min-component-triangles", type=int, default=12)
    return parser.parse_args()


def load_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scene = trimesh.load(path, force="scene")
    point_geometries = [
        geometry
        for geometry in scene.dump()
        if isinstance(geometry, trimesh.points.PointCloud)
    ]
    if not point_geometries:
        raise ValueError(f"No point-cloud geometry found in {path}")

    points = np.concatenate(
        [np.asarray(geometry.vertices, dtype=np.float64) for geometry in point_geometries],
        axis=0,
    )
    color_parts = []
    for geometry in point_geometries:
        colors = np.asarray(geometry.colors)
        if len(colors) != len(geometry.vertices):
            colors = np.full((len(geometry.vertices), 4), 200, dtype=np.uint8)
            colors[:, 3] = 255
        color_parts.append(colors[:, :3].astype(np.uint8))
    return points, np.concatenate(color_parts, axis=0)


def remove_small_components(
    mesh: o3d.geometry.TriangleMesh, minimum_triangles: int
) -> int:
    if minimum_triangles <= 1 or not mesh.has_triangles():
        return 0
    labels, counts, _areas = mesh.cluster_connected_triangles()
    labels_array = np.asarray(labels)
    counts_array = np.asarray(counts)
    remove_mask = counts_array[labels_array] < minimum_triangles
    removed = int(remove_mask.sum())
    if removed:
        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()
    return removed


def reconstruct_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    normal_radius_factor: float,
    ball_radius_factors: list[float],
    max_edge_factor: float,
    min_component_triangles: int,
) -> tuple[trimesh.Trimesh, dict[str, float | int]]:
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    if len(points) < 100:
        raise ValueError(f"Need at least 100 finite points, found {len(points)}")

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    cloud.remove_duplicated_points()

    clean_points = np.asarray(cloud.points)
    nearest = np.asarray(cloud.compute_nearest_neighbor_distance())
    positive_nearest = nearest[nearest > 0]
    if not len(positive_nearest):
        raise ValueError("Cannot estimate point spacing from duplicated points")
    spacing = float(np.median(positive_nearest))

    normal_radius = max(spacing * normal_radius_factor, 1e-4)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=50)
    )
    cloud.orient_normals_consistent_tangent_plane(min(30, len(clean_points) - 1))

    radii = o3d.utility.DoubleVector(
        [spacing * factor for factor in ball_radius_factors]
    )
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud, radii
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    removed_long_triangles = 0
    if len(triangles):
        tri_vertices = vertices[triangles]
        edge_lengths = np.stack(
            [
                np.linalg.norm(tri_vertices[:, 0] - tri_vertices[:, 1], axis=1),
                np.linalg.norm(tri_vertices[:, 1] - tri_vertices[:, 2], axis=1),
                np.linalg.norm(tri_vertices[:, 2] - tri_vertices[:, 0], axis=1),
            ],
            axis=1,
        )
        long_mask = edge_lengths.max(axis=1) > max_edge_factor * spacing
        removed_long_triangles = int(long_mask.sum())
        if removed_long_triangles:
            mesh.remove_triangles_by_mask(long_mask)
            mesh.remove_unreferenced_vertices()

    removed_small_triangles = remove_small_components(
        mesh, min_component_triangles
    )
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    mesh_vertices = np.asarray(mesh.vertices)
    mesh_faces = np.asarray(mesh.triangles)
    if not len(mesh_faces):
        raise RuntimeError("Ball Pivoting produced no triangles")

    _distances, nearest_indices = cKDTree(points).query(mesh_vertices, k=1)
    vertex_colors = colors[nearest_indices]
    rgba = np.column_stack(
        [vertex_colors, np.full(len(vertex_colors), 255, dtype=np.uint8)]
    )
    output_mesh = trimesh.Trimesh(
        vertices=mesh_vertices,
        faces=mesh_faces,
        vertex_colors=rgba,
        process=False,
    )
    stats: dict[str, float | int] = {
        "input_points": int(len(points)),
        "median_point_spacing": spacing,
        "mesh_vertices": int(len(mesh_vertices)),
        "mesh_triangles": int(len(mesh_faces)),
        "removed_long_triangles": removed_long_triangles,
        "removed_small_component_triangles": removed_small_triangles,
    }
    return output_mesh, stats


def write_readme(output_root: Path) -> None:
    text = """# Run 34 Ball Pivoting Meshes

These GLB files are visualization-only surface reconstructions generated from
the comparable 3,500-point Run 34 exports:

- `01_*_mesh.glb`: MV-DUSt3R+ RGB-only candidate cloud.
- `02_*_mesh.glb`: Run 30 RGB-D source-depth corrected cloud.
- `03_*_mesh.glb`: direct RGB-D backprojection cloud.

Ball Pivoting connects only nearby samples and intentionally preserves holes.
The meshes do not improve the underlying reconstruction, are not ground truth,
and must not be used to replace point-cloud metrics.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else input_root / "meshes_ball_pivoting"
    )
    source_paths = sorted(
        {
            path
            for pattern in SOURCE_GLOBS
            for path in input_root.glob(f"scene*/{pattern}")
        }
    )
    if not source_paths:
        raise FileNotFoundError(f"No Run 34 comparison GLBs found under {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_path in source_paths:
        group_name = source_path.parent.name
        group_output = output_root / group_name
        group_output.mkdir(parents=True, exist_ok=True)
        output_path = group_output / f"{source_path.stem}_mesh.glb"

        points, colors = load_point_cloud(source_path)
        mesh, stats = reconstruct_mesh(
            points=points,
            colors=colors,
            normal_radius_factor=args.normal_radius_factor,
            ball_radius_factors=list(args.ball_radius_factors),
            max_edge_factor=args.max_edge_factor,
            min_component_triangles=args.min_component_triangles,
        )
        mesh.export(output_path)
        with output_path.open("rb") as handle:
            if handle.read(4) != b"glTF":
                raise RuntimeError(f"Invalid GLB output: {output_path}")

        row = {
            "group": group_name,
            "source_file": source_path.name,
            "mesh_file": str(output_path.relative_to(output_root)),
            "algorithm": "ball_pivoting",
            "normal_radius_factor": args.normal_radius_factor,
            "ball_radius_factors": ",".join(
                str(value) for value in args.ball_radius_factors
            ),
            "max_edge_factor": args.max_edge_factor,
            "min_component_triangles": args.min_component_triangles,
            **stats,
        }
        rows.append(row)
        print(
            f"{group_name}/{source_path.name}: "
            f"{stats['input_points']} points -> "
            f"{stats['mesh_vertices']} vertices, "
            f"{stats['mesh_triangles']} triangles"
        )

    manifest_path = output_root / "mesh_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_readme(output_root)
    print(f"Wrote {len(rows)} meshes to {output_root}")


if __name__ == "__main__":
    main()
