import csv
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


RAW_RUN30 = "https://raw.githubusercontent.com/NguyenNgocMinh30012005/Computer_vision_2025.2/main/scripts/kaggle/kaggle_run30_rgbd_source_depth_correction.py"


def ensure_helper_module(module_name, raw_url):
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        helper_path = Path(f"{module_name}.py")
        print(f"Downloading helper module {module_name} from GitHub raw")
        urllib.request.urlretrieve(raw_url, helper_path)
        sys.path.insert(0, str(Path.cwd()))
        return __import__(module_name)


r30 = ensure_helper_module("kaggle_run30_rgbd_source_depth_correction", RAW_RUN30)
r28 = r30.r28
r27 = r30.r27
base = r30.base

RUN_NAME = "run_32_direct_rgbd_backprojection_baseline"
SEED = 3232
MAX_SCENES = int(os.environ.get("RUN32_MAX_SCENES", "30"))
MAX_EVAL_GROUPS = int(os.environ.get("RUN32_MAX_EVAL_GROUPS", "36"))
MAX_POINTS_PER_GROUP = int(os.environ.get("RUN32_MAX_POINTS_PER_GROUP", "3500"))
DEPTH_PIXEL_STRIDE = int(os.environ.get("RUN32_DEPTH_PIXEL_STRIDE", "1"))
MAX_DEPTH_M = float(os.environ.get("RUN32_MAX_DEPTH_M", "10.0"))
VOXEL_SIZE_M = float(os.environ.get("RUN32_VOXEL_SIZE_M", "0.02"))
GATE_MARGIN_F1 = float(os.environ.get("RUN32_GATE_MARGIN_F1", "0.005"))
CAUTION = (
    "Direct RGB-D backprojection is not source-depth correction. It is a "
    "depth-only/RGB-D baseline. Source-depth correction specifically refers "
    "to correcting MV-DUSt3R+ candidate points using source depth residuals."
)
CIRCULARITY_WARNING = (
    "The current controlled evaluator builds its GT cloud from the same selected "
    "input depth maps. Direct backprojection therefore shares its depth source "
    "with the evaluation target and is not an independent mesh/laser-scan test."
)


def deterministic_sample(points, colors, limit, seed):
    if len(points) <= limit:
        return points, colors
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(points), limit, replace=False))
    return points[indices], colors[indices]


def voxel_downsample(points, colors, voxel_size, limit, seed):
    if not len(points):
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    first_indices = np.sort(first_indices)
    voxel_points = points[first_indices]
    voxel_colors = colors[first_indices]
    return deterministic_sample(voxel_points, voxel_colors, limit, seed)


def load_rgb_at_depth_resolution(rgb_path, depth_shape):
    image = Image.open(rgb_path).convert("RGB")
    height, width = depth_shape
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def backproject_frame(rgb_path, first_pose):
    depth_path = Path(str(rgb_path).replace(".jpg", ".png"))
    pose_path = Path(str(rgb_path).replace(".jpg", ".txt"))
    if not depth_path.exists() or not pose_path.exists():
        print(
            "Run 32 skipping missing frame inputs:",
            {"rgb": str(rgb_path), "depth": str(depth_path), "pose": str(pose_path)},
        )
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    depth = np.asarray(Image.open(depth_path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if depth.size and float(depth.max()) > 100.0:
        depth = depth / 1000.0
    rgb = load_rgb_at_depth_resolution(rgb_path, depth.shape)
    height, width = depth.shape
    fx, fy, cx, cy = r28.intrinsics(depth.shape)
    ys, xs = np.mgrid[0:height:DEPTH_PIXEL_STRIDE, 0:width:DEPTH_PIXEL_STRIDE]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > 0.0) & (z <= MAX_DEPTH_M)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    x_pixels = xs[valid].astype(np.float32)
    y_pixels = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    x = (x_pixels - cx) * z / fx
    y = (y_pixels - cy) * z / fy
    points_camera = np.stack([x, y, z, np.ones_like(z)], axis=1)
    pose = base.parse_pose(pose_path)
    points_world = (pose @ points_camera.T).T
    points_first = (np.linalg.inv(first_pose) @ points_world.T).T[:, :3]
    colors = rgb[ys[valid], xs[valid]]
    finite = np.isfinite(points_first).all(axis=1)
    return points_first[finite].astype(np.float32), colors[finite].astype(np.uint8)


def build_direct_cloud(view_files, group_key):
    first_pose = base.parse_pose(str(view_files[0]).replace(".jpg", ".txt"))
    point_parts = []
    color_parts = []
    for view_file in view_files:
        points, colors = backproject_frame(view_file, first_pose)
        if len(points):
            point_parts.append(points)
            color_parts.append(colors)
    if not point_parts:
        raise RuntimeError(f"No valid source depth points for {group_key}")
    raw_points = np.concatenate(point_parts, axis=0).astype(np.float32)
    raw_colors = np.concatenate(color_parts, axis=0).astype(np.uint8)
    sampled_points, sampled_colors = deterministic_sample(
        raw_points,
        raw_colors,
        MAX_POINTS_PER_GROUP,
        r27.stable_seed(f"run32-sampled-{group_key}"),
    )
    voxel_points, voxel_colors = voxel_downsample(
        raw_points,
        raw_colors,
        VOXEL_SIZE_M,
        MAX_POINTS_PER_GROUP,
        r27.stable_seed(f"run32-voxel-{group_key}"),
    )
    return {
        "raw_count": int(len(raw_points)),
        "sampled": (sampled_points, sampled_colors),
        "voxel": (voxel_points, voxel_colors),
    }


def ensure_trimesh():
    try:
        import trimesh

        return trimesh
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "trimesh"]
        )
        import trimesh

        return trimesh


def export_glb(points, colors, output_path):
    trimesh = ensure_trimesh()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.concatenate(
        [colors.astype(np.uint8), np.full((len(colors), 1), 255, dtype=np.uint8)],
        axis=1,
    )
    cloud = trimesh.points.PointCloud(vertices=points, colors=rgba)
    cloud.export(output_path)
    return str(output_path)


def find_run30_output():
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None
    candidates = sorted(
        input_root.rglob("run_30_rgbd_source_depth_correction/metrics.csv")
    )
    return candidates[0].parent if candidates else None


def read_csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_run30_metric_row(row):
    integer_fields = {"num_views", "num_pred_points", "num_gt_points"}
    float_fields = {
        "occlusion_proxy_ratio",
        "ambiguity_proxy_ratio",
        "runtime_seconds",
        "selected_ratio",
        "accuracy",
        "completeness",
        "precision",
        "recall",
        "fscore",
        "chamfer",
        "threshold_m",
        "conf_percent",
        "conf_threshold",
        "correction_ratio",
        "alpha",
        "residual_threshold_m",
    }
    output = dict(row)
    for field in integer_fields:
        if output.get(field) not in {None, ""}:
            output[field] = int(float(output[field]))
    for field in float_fields:
        if output.get(field) not in {None, ""}:
            output[field] = float(output[field])
    return output


def load_run30_comparison_rows(run30_dir, group_keys):
    if run30_dir is None:
        return [], {}
    rows = read_csv_rows(run30_dir / "metrics.csv")
    methods = {
        "all_candidates",
        "confidence_fixed_final",
        "rgbd_source_depth_selected",
    }
    selected = [
        coerce_run30_metric_row(row)
        for row in rows
        if row.get("group_key") in group_keys and row.get("method") in methods
    ]
    diagnostics = {}
    for row in selected:
        diagnostics[row["group_key"]] = {
            "occlusion_proxy_ratio": r27.as_float(row, "occlusion_proxy_ratio"),
            "ambiguity_proxy_ratio": r27.as_float(row, "ambiguity_proxy_ratio"),
        }
        row["source_run"] = row.get("run", "run_30_rgbd_source_depth_correction")
        row["run"] = RUN_NAME
        row["comparison_source"] = "mounted_run30_output"
        row["circularity_warning"] = 0
    return selected, diagnostics


def direct_metric_row(
    group,
    method,
    implementation_method,
    points,
    gt,
    runtime_seconds,
    raw_count,
    output_glb,
    diagnostics,
):
    metrics = base.compute_metrics(points.astype(np.float32), gt)
    return {
        "run": RUN_NAME,
        "source_run": RUN_NAME,
        "split": group["split"],
        "scene": group["scene"],
        "num_views": int(group["num_views"]),
        "view_policy": group["view_policy"],
        "group_key": group["group_key"],
        "method": method,
        "implementation_method": implementation_method,
        "method_family": "direct_rgbd_depth_baseline",
        "occlusion_proxy_ratio": diagnostics.get("occlusion_proxy_ratio", 0.0),
        "ambiguity_proxy_ratio": diagnostics.get("ambiguity_proxy_ratio", 0.0),
        "runtime_seconds": runtime_seconds,
        "raw_backprojected_points": raw_count,
        "selected_ratio": 1.0,
        "output_glb": output_glb,
        "evaluation_target_source": "same_selected_input_depth_maps",
        "circularity_warning": 1,
        **metrics,
    }


def evaluate_group(scene_lookup, group, output_dir, diagnostics):
    view_files = r27.choose_group_views(scene_lookup, group)
    print(
        "Run 32 group views:",
        {"group": group["group_key"], "views": [path.name for path in view_files]},
    )
    started = time.time()
    clouds = build_direct_cloud(view_files, group["group_key"])
    gt, _stats = base.build_gt_cloud(view_files)
    runtime = time.time() - started
    voxel_points, voxel_colors = clouds["voxel"]
    sampled_points, _sampled_colors = clouds["sampled"]
    glb_path = export_glb(
        voxel_points,
        voxel_colors,
        output_dir / group["group_key"] / "direct_rgbd_backprojection.glb",
    )
    primary = direct_metric_row(
        group,
        "direct_rgbd_backprojection",
        "direct_rgbd_backprojection_voxel",
        voxel_points,
        gt,
        runtime,
        clouds["raw_count"],
        glb_path,
        diagnostics,
    )
    voxel = dict(primary)
    voxel["method"] = "direct_rgbd_backprojection_voxel"
    sampled = direct_metric_row(
        group,
        "direct_rgbd_backprojection_sampled",
        "direct_rgbd_backprojection_sampled",
        sampled_points,
        gt,
        runtime,
        clouds["raw_count"],
        "",
        diagnostics,
    )
    rows = [primary, voxel, sampled]
    for row in rows:
        print("Run 32 metric row:", row)
    return rows


def limit_lookup(rows, split, subset, method):
    candidates = [
        row
        for row in rows
        if row["split"] == split
        and row["limit_subset"] == subset
        and row["method"] == method
    ]
    return candidates[0] if candidates else None


def comparison_gate(limit_rows, run30_available):
    subsets = ["overall", "occlusion_challenging", "ambiguity_challenging"]
    result = {
        "run": RUN_NAME,
        "direct_method": "direct_rgbd_backprojection",
        "run30_method": "rgbd_source_depth_selected",
        "run30_comparison_available": int(run30_available),
        "gate_margin_f1": GATE_MARGIN_F1,
        "circularity_warning": 1,
    }
    if not run30_available:
        result.update(
            {
                "comparison_outcome": "run30_output_unavailable",
                "final_claim_changed": 0,
            }
        )
        return [result]

    deltas = {}
    for split in ["val", "test"]:
        for subset in subsets:
            direct = limit_lookup(
                limit_rows, split, subset, "direct_rgbd_backprojection"
            )
            run30 = limit_lookup(
                limit_rows, split, subset, "rgbd_source_depth_selected"
            )
            if direct is None or run30 is None:
                raise RuntimeError(f"Missing comparison row for {split}/{subset}")
            prefix = f"{split}_{subset}"
            result[f"{prefix}_direct_fscore"] = direct["mean_fscore"]
            result[f"{prefix}_run30_fscore"] = run30["mean_fscore"]
            result[f"{prefix}_direct_minus_run30"] = (
                direct["mean_fscore"] - run30["mean_fscore"]
            )
            deltas[(split, subset)] = result[f"{prefix}_direct_minus_run30"]

    val_delta = deltas[("val", "overall")]
    test_delta = deltas[("test", "overall")]
    if val_delta >= GATE_MARGIN_F1:
        outcome = "direct_depth_dominant"
    elif val_delta <= -GATE_MARGIN_F1:
        outcome = "run30_adds_value_over_direct"
    else:
        outcome = "competitive_close"
    result.update(
        {
            "comparison_outcome": outcome,
            "validation_direction_matches_test": int(val_delta * test_delta >= 0.0),
            "direct_wins_validation": int(val_delta >= GATE_MARGIN_F1),
            "run30_wins_validation": int(val_delta <= -GATE_MARGIN_F1),
            "competitive_close_validation": int(abs(val_delta) < GATE_MARGIN_F1),
            "final_claim_changed": 0,
        }
    )
    return [result]


def qualitative_manifest(metric_rows):
    return [
        {
            "run": RUN_NAME,
            "split": row["split"],
            "scene": row["scene"],
            "group_key": row["group_key"],
            "method": row["method"],
            "output_glb": row.get("output_glb", ""),
        }
        for row in metric_rows
        if row["method"] == "direct_rgbd_backprojection" and row.get("output_glb")
    ]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    r27.validate_static_configuration()
    posed_root = base.find_posed_images_root()
    scene_dirs = r27.discover_scene_dirs(posed_root)[:MAX_SCENES]
    scene_lookup = {path.name: path for path in scene_dirs}
    splits = r27.scene_splits(scene_dirs)
    manifest = r27.build_group_manifest(scene_dirs, splits)
    eval_groups = r27.balanced_group_subset(
        [row for row in manifest if row["split"] in {"val", "test"}],
        MAX_EVAL_GROUPS,
    )
    group_keys = {row["group_key"] for row in eval_groups}
    run30_dir = find_run30_output()
    run30_rows, group_diagnostics = load_run30_comparison_rows(
        run30_dir, group_keys
    )
    print("Run 32 posed images:", posed_root)
    print("Run 32 splits:", splits)
    print("Run 32 eval groups:", len(eval_groups))
    print("Run 32 Run 30 source:", str(run30_dir) if run30_dir else "unavailable")
    print("Run 32 caution:", CAUTION)
    print("Run 32 circularity warning:", CIRCULARITY_WARNING)

    direct_rows = []
    for index, group in enumerate(eval_groups, start=1):
        direct_rows.extend(
            evaluate_group(
                scene_lookup,
                group,
                out_dir / "point_clouds",
                group_diagnostics.get(group["group_key"], {}),
            )
        )
        if index % 6 == 0 or index == len(eval_groups):
            print(f"Run 32 completed groups: {index}/{len(eval_groups)}")

    metric_rows = run30_rows + direct_rows
    summary_rows = r30.summarize(metric_rows)
    limit_rows = r30.limit_summary(metric_rows)
    gate_rows = comparison_gate(limit_rows, bool(run30_rows))
    qualitative_rows = qualitative_manifest(direct_rows)
    r27.write_csv_union(out_dir / "metrics.csv", metric_rows)
    r27.write_csv_union(out_dir / "summary.csv", summary_rows)
    r27.write_csv_union(out_dir / "limit_summary.csv", limit_rows)
    r27.write_csv_union(out_dir / "gate_decision.csv", gate_rows)
    r27.write_csv_union(out_dir / "qualitative_manifest.csv", qualitative_rows)
    config = {
        "run": RUN_NAME,
        "purpose": "Direct RGB-D source-depth backprojection baseline without MV-DUSt3R+.",
        "num_scenes": len(scene_dirs),
        "scene_splits": splits,
        "num_eval_groups": len(eval_groups),
        "max_points_per_group": MAX_POINTS_PER_GROUP,
        "depth_pixel_stride": DEPTH_PIXEL_STRIDE,
        "max_depth_m": MAX_DEPTH_M,
        "voxel_size_m": VOXEL_SIZE_M,
        "primary_method": "direct_rgbd_backprojection",
        "primary_implementation": "direct_rgbd_backprojection_voxel",
        "diagnostic_method": "direct_rgbd_backprojection_sampled",
        "run30_output_dir": str(run30_dir) if run30_dir else None,
        "run30_comparison_available": bool(run30_rows),
        "policy_tuning_performed": False,
        "test_tuning_performed": False,
        "evaluation_target_source": "depth PNGs from the same selected input views",
        "circularity_warning": CIRCULARITY_WARNING,
        "caution": CAUTION,
        "runtime_seconds": time.time() - started,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print("Run 32 config:", config)
    print("Run 32 summary:")
    for row in summary_rows:
        print(row)
    print("Run 32 gate decision:")
    for row in gate_rows:
        print(row)


if __name__ == "__main__":
    main()
