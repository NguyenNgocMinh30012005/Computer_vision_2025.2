import csv
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

try:
    from kaggle_predicted_depth_utils import (
        apply_residual_correction,
        backproject_depth_cloud,
        sample_depth_targets,
    )
except ModuleNotFoundError:
    def _intrinsics(depth_shape):
        height, width = depth_shape
        return (
            577.870605 * (width / 640.0),
            577.870605 * (height / 480.0),
            319.5 * (width / 640.0),
            239.5 * (height / 480.0),
        )

    def sample_depth_targets(
        view_files,
        depth_maps,
        xs,
        ys,
        view_ids,
        candidate_height,
        candidate_width,
        parse_pose,
        scale=1.0,
        min_depth_m=0.10,
    ):
        poses = [
            parse_pose(str(path).replace(".jpg", ".txt")).astype(
                np.float32
            )
            for path in view_files
        ]
        first_pose_inv = np.linalg.inv(poses[0]).astype(np.float32)
        targets = np.zeros((len(xs), 3), dtype=np.float32)
        valid = np.zeros(len(xs), dtype=bool)
        sampled_depth = np.zeros(len(xs), dtype=np.float32)
        for view_index, depth in enumerate(depth_maps):
            indices = np.where(view_ids == view_index)[0]
            if not len(indices):
                continue
            depth = np.asarray(depth, dtype=np.float32) * float(scale)
            height, width = depth.shape
            depth_x = np.clip(
                np.rint(
                    xs[indices]
                    / max(candidate_width - 1, 1)
                    * max(width - 1, 1)
                ).astype(np.int32),
                0,
                width - 1,
            )
            depth_y = np.clip(
                np.rint(
                    ys[indices]
                    / max(candidate_height - 1, 1)
                    * max(height - 1, 1)
                ).astype(np.int32),
                0,
                height - 1,
            )
            z = depth[depth_y, depth_x]
            usable = np.isfinite(z) & (z > float(min_depth_m))
            if not usable.any():
                continue
            usable_indices = indices[usable]
            z = z[usable]
            x = depth_x[usable].astype(np.float32)
            y = depth_y[usable].astype(np.float32)
            fx, fy, cx, cy = _intrinsics(depth.shape)
            camera = np.column_stack(
                [
                    (x - cx) / fx * z,
                    (y - cy) / fy * z,
                    z,
                    np.ones(len(z), dtype=np.float32),
                ]
            )
            world = (poses[view_index] @ camera.T).T
            first_camera = (first_pose_inv @ world.T).T
            targets[usable_indices] = first_camera[:, :3]
            sampled_depth[usable_indices] = z
            valid[usable_indices] = True
        return targets, valid, sampled_depth

    def backproject_depth_cloud(
        view_files,
        depth_maps,
        parse_pose,
        scale=1.0,
        stride=4,
        min_depth_m=0.10,
    ):
        poses = [
            parse_pose(str(path).replace(".jpg", ".txt")).astype(
                np.float32
            )
            for path in view_files
        ]
        first_pose_inv = np.linalg.inv(poses[0]).astype(np.float32)
        clouds = []
        for view_index, depth in enumerate(depth_maps):
            depth = np.asarray(depth, dtype=np.float32) * float(scale)
            height, width = depth.shape
            yy, xx = np.mgrid[0:height:stride, 0:width:stride]
            z = depth[yy, xx]
            valid = np.isfinite(z) & (z > float(min_depth_m))
            if not valid.any():
                continue
            x = xx[valid].astype(np.float32)
            y = yy[valid].astype(np.float32)
            z = z[valid]
            fx, fy, cx, cy = _intrinsics(depth.shape)
            camera = np.column_stack(
                [
                    (x - cx) / fx * z,
                    (y - cy) / fy * z,
                    z,
                    np.ones(len(z), dtype=np.float32),
                ]
            )
            world = (poses[view_index] @ camera.T).T
            first_camera = (first_pose_inv @ world.T).T
            clouds.append(first_camera[:, :3].astype(np.float32))
        if not clouds:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(clouds, axis=0)

    def apply_residual_correction(
        points,
        targets,
        valid,
        tau_pred,
        alpha,
    ):
        points = np.asarray(points, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        residual = np.linalg.norm(points - targets, axis=1).astype(
            np.float32
        )
        mask = valid & (residual >= float(tau_pred))
        corrected = points.copy()
        corrected[mask] = (
            (1.0 - float(alpha)) * points[mask]
            + float(alpha) * targets[mask]
        )
        return corrected, mask, residual


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


r30 = ensure_helper_module(
    "kaggle_run30_rgbd_source_depth_correction",
    RAW_RUN30,
)
r28 = r30.r28
r27 = r30.r27
base = r30.base

RUN_NAME = "run_36_predicted_depth_correction"
SEED = 3636
MAX_SCENES = int(os.environ.get("RUN36_MAX_SCENES", "30"))
MAX_TRAIN_GROUPS = int(os.environ.get("RUN36_MAX_TRAIN_GROUPS", "48"))
MAX_EVAL_GROUPS = int(os.environ.get("RUN36_MAX_EVAL_GROUPS", "36"))
MAX_CANDIDATES_PER_GROUP = int(
    os.environ.get("RUN36_MAX_CANDIDATES_PER_GROUP", "3500")
)
GATE_MARGIN_F1 = float(os.environ.get("RUN36_GATE_MARGIN_F1", "0.005"))
TAU_GRID = [0.10, 0.20, 0.30, 0.50, 0.75]
ALPHA_GRID = [0.25, 0.50, 0.75, 1.00]
SCALE_MODES = ["raw", "global_scale"]


def write_csv_union(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key, default=float("nan")):
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def normalized_metrics(pred, gt, threshold=0.2):
    def normalize(points):
        points = np.asarray(points, dtype=np.float64)
        centered = points - points.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
        return centered / max(float(scale), 1e-8)

    pred_norm = normalize(pred)
    gt_norm = normalize(gt)
    pred_to_gt = cKDTree(gt_norm).query(pred_norm, k=1, workers=-1)[0]
    gt_to_pred = cKDTree(pred_norm).query(gt_norm, k=1, workers=-1)[0]
    return {
        "normalized_distance": float(
            0.5 * (pred_to_gt.mean() + gt_to_pred.mean())
        ),
        "dac_at_0_2_normalized": float(
            np.mean(pred_to_gt <= float(threshold))
        ),
        "normalized_dac_threshold": float(threshold),
    }


def compute_metrics(points, gt):
    sampled = base.downsample(
        np.asarray(points, dtype=np.float32),
        base.MAX_POINTS,
    )
    metrics = base.compute_metrics(sampled, gt)
    if "normalized_distance" not in metrics:
        aligned = base.center_scale_align(sampled, gt)
        metrics.update(normalized_metrics(aligned, gt))
    return metrics


def fast_fscore(points, record, threshold=0.05):
    pred = base.center_scale_align(
        np.asarray(points, dtype=np.float32),
        record["gt"],
    )
    pred_to_gt = record["gt_tree"].query(
        pred,
        k=1,
        workers=-1,
    )[0]
    pred_tree = cKDTree(pred)
    gt_to_pred = pred_tree.query(
        record["gt"],
        k=1,
        workers=-1,
    )[0]
    precision = float(np.mean(pred_to_gt < float(threshold)))
    recall = float(np.mean(gt_to_pred < float(threshold)))
    return (
        0.0
        if precision + recall == 0.0
        else float(2.0 * precision * recall / (precision + recall))
    )


def locate_run_output(run_name, required_file):
    matches = sorted(
        Path("/kaggle/input").rglob(f"{run_name}/{required_file}")
    )
    if not matches:
        raise FileNotFoundError(
            f"Unable to find mounted {run_name}/{required_file}. "
            "Attach the required Kaggle notebook output as a kernel source."
        )
    return matches[0].parent


class PredictedDepthCache:
    def __init__(self, run35_dir):
        self.run35_dir = Path(run35_dir)
        self.config = json.loads(
            (self.run35_dir / "run_config.json").read_text(
                encoding="utf-8"
            )
        )
        rows = read_csv(
            self.run35_dir / "predicted_depth_cache_manifest.csv"
        )
        self.paths = {
            (row["scene"], row["frame"]): self.run35_dir
            / row["cache_relpath"]
            for row in rows
        }
        self.memory = {}

    def get(self, scene, frame):
        key = (scene, frame)
        if key not in self.memory:
            path = self.paths.get(key)
            if path is None or not path.exists():
                raise FileNotFoundError(
                    f"Run 35 predicted depth missing for {scene}/{frame}"
                )
            with np.load(path) as payload:
                self.memory[key] = payload["depth"].astype(np.float32)
        return self.memory[key]


def confidence_mask(conf, num_views):
    percentile = base.FIXED_FINAL_CONF_BY_VIEW_COUNT.get(
        num_views,
        base.CONF_PERCENT,
    )
    threshold = float(np.quantile(conf, percentile / 100.0))
    return conf >= threshold, percentile, threshold


def scale_for_mode(mode, global_scale):
    if mode == "raw":
        return 1.0
    if mode == "global_scale":
        return float(global_scale)
    raise ValueError(f"Unknown predicted-depth scale mode: {mode}")


def build_record(
    backbone,
    root,
    scene_lookup,
    group,
    out_dir,
    depth_cache,
    global_scale,
):
    view_files = r27.choose_group_views(scene_lookup, group)
    print(
        "Run 36 group views:",
        {
            "group": group["group_key"],
            "views": [path.name for path in view_files],
        },
    )
    output, _glb, backbone_runtime = base.run_inference(
        backbone,
        root,
        view_files,
        out_dir,
    )
    points, conf, xs, ys, view_ids, image_h, image_w = (
        r27.output_to_candidates(output)
    )
    points, conf, xs, ys, view_ids = r27.subsample_candidates(
        points,
        conf,
        xs,
        ys,
        view_ids,
        MAX_CANDIDATES_PER_GROUP,
        r27.stable_seed("run36-" + group["group_key"]),
    )
    base_features = r27.build_features(
        points,
        conf,
        xs,
        ys,
        view_ids,
        image_h,
        image_w,
        group,
    )
    features, photo_stats = r27.aggregate_image_pair_features(
        base_features,
        points,
        conf,
        xs,
        ys,
        view_ids,
        view_files,
        group,
        image_h,
        image_w,
    )
    depth_maps = [
        depth_cache.get(group["scene"], path.name)
        for path in view_files
    ]

    method_inputs = {}
    for mode in SCALE_MODES:
        scale = scale_for_mode(mode, global_scale)
        targets, valid, sampled_depth = sample_depth_targets(
            view_files,
            depth_maps,
            xs,
            ys,
            view_ids,
            image_h,
            image_w,
            base.parse_pose,
            scale=scale,
        )
        direct_cloud = backproject_depth_cloud(
            view_files,
            depth_maps,
            base.parse_pose,
            scale=scale,
            stride=4,
        )
        direct_cloud = base.downsample(
            direct_cloud.astype(np.float32),
            MAX_CANDIDATES_PER_GROUP,
        )
        method_inputs[mode] = {
            "targets": targets,
            "valid": valid,
            "sampled_depth": sampled_depth,
            "direct_cloud": direct_cloud,
        }

    # True source depth starts here and is evaluation-only. Nothing below is
    # passed into predicted-depth correction or direct predicted backprojection.
    gt, _gt_stats = base.build_gt_cloud(view_files)
    true_targets, true_valid, _true_depth = r28.source_ray_targets(
        view_files,
        xs,
        ys,
        view_ids,
        image_h,
        image_w,
    )
    _aligned_true, true_residual_m, _alignment = (
        r28.paired_similarity_targets(
            points,
            true_targets,
            true_valid,
        )
    )
    low_support = features[:, r27.SUPPORT_FRAC_010_INDEX] < 0.125
    has_projection = features[:, r27.PHOTO_TARGET_COUNT_INDEX] > 0.0
    occlusion_proxy = (
        true_valid & low_support & has_projection
    ).astype(np.float32)
    ambiguity_proxy = (
        true_valid & (true_residual_m >= 0.15)
    ).astype(np.float32)
    del output
    torch.cuda.empty_cache()
    return {
        "group": group,
        "points": points,
        "conf": conf,
        "method_inputs": method_inputs,
        "gt": gt,
        "gt_tree": cKDTree(gt),
        "occlusion_proxy": occlusion_proxy,
        "ambiguity_proxy": ambiguity_proxy,
        "photo_stats": photo_stats,
        "runtime_seconds": float(backbone_runtime),
        "view_files": view_files,
    }


def corrected_points(record, mode, tau_pred, alpha):
    inputs = record["method_inputs"][mode]
    return apply_residual_correction(
        record["points"],
        inputs["targets"],
        inputs["valid"],
        tau_pred,
        alpha,
    )


def select_policies(records):
    rows = []
    for mode in SCALE_MODES:
        for tau_pred in TAU_GRID:
            for alpha in ALPHA_GRID:
                fscores = []
                correction_ratios = []
                residuals = []
                for record in records:
                    corrected, mask, residual = corrected_points(
                        record,
                        mode,
                        tau_pred,
                        alpha,
                    )
                    fscores.append(fast_fscore(corrected, record))
                    correction_ratios.append(float(mask.mean()))
                    valid = record["method_inputs"][mode]["valid"]
                    if valid.any():
                        residuals.append(float(residual[valid].mean()))
                rows.append(
                    {
                        "run": RUN_NAME,
                        "depth_scale_mode": mode,
                        "tau_pred": float(tau_pred),
                        "alpha": float(alpha),
                        "num_internal_val_groups": len(records),
                        "mean_reconstruction_fscore": float(
                            np.mean(fscores)
                        ),
                        "mean_correction_ratio": float(
                            np.mean(correction_ratios)
                        ),
                        "mean_predicted_residual": float(
                            np.mean(residuals)
                        ),
                    }
                )
    selected_by_mode = {
        mode: max(
            [row for row in rows if row["depth_scale_mode"] == mode],
            key=lambda row: row["mean_reconstruction_fscore"],
        )
        for mode in SCALE_MODES
    }
    selected = max(
        selected_by_mode.values(),
        key=lambda row: row["mean_reconstruction_fscore"],
    )
    return selected, selected_by_mode, rows


def score_points(points, record, method, family, extra):
    group = record["group"]
    return {
        "run": RUN_NAME,
        "split": group["split"],
        "scene": group["scene"],
        "num_views": int(group["num_views"]),
        "view_policy": group["view_policy"],
        "group_key": group["group_key"],
        "method": method,
        "method_family": family,
        "occlusion_proxy_ratio": float(
            record["occlusion_proxy"].mean()
        ),
        "ambiguity_proxy_ratio": float(
            record["ambiguity_proxy"].mean()
        ),
        "runtime_seconds": record["runtime_seconds"],
        "uses_true_source_depth_for_inference": 0,
        "uses_predicted_depth_for_inference": int(
            family != "rgb_only_baseline"
        ),
        "uses_known_pose": 1,
        "uses_known_intrinsics": 1,
        **record["photo_stats"],
        **extra,
        **compute_metrics(points, record["gt"]),
    }


def evaluate_group(record, selected, selected_by_mode):
    group = record["group"]
    conf_mask, conf_percent, conf_threshold = confidence_mask(
        record["conf"],
        int(group["num_views"]),
    )
    rows = [
        score_points(
            record["points"],
            record,
            "mvdust3r_rgb_only_all_candidates",
            "rgb_only_baseline",
            {"selected_ratio": 1.0},
        ),
        score_points(
            record["points"][conf_mask],
            record,
            "mvdust3r_confidence_fixed",
            "rgb_only_baseline",
            {
                "selected_ratio": float(conf_mask.mean()),
                "conf_percent": conf_percent,
                "conf_threshold": conf_threshold,
            },
        ),
    ]

    for mode, method in [
        ("raw", "predicted_depth_correction_raw"),
        ("global_scale", "predicted_depth_correction_scale_aligned"),
    ]:
        policy = selected_by_mode[mode]
        corrected, mask, residual = corrected_points(
            record,
            mode,
            policy["tau_pred"],
            policy["alpha"],
        )
        valid = record["method_inputs"][mode]["valid"]
        rows.append(
            score_points(
                corrected,
                record,
                method,
                "predicted_depth_correction",
                {
                    "depth_scale_mode": mode,
                    "tau_pred": policy["tau_pred"],
                    "alpha": policy["alpha"],
                    "correction_ratio": float(mask.mean()),
                    "valid_predicted_depth_ratio": float(valid.mean()),
                    "mean_predicted_residual": float(
                        residual[valid].mean()
                    )
                    if valid.any()
                    else float("nan"),
                },
            )
        )

    direct_mode = selected["depth_scale_mode"]
    rows.append(
        score_points(
            record["method_inputs"][direct_mode]["direct_cloud"],
            record,
            "predicted_depth_direct_backprojection",
            "predicted_depth_direct",
            {
                "depth_scale_mode": direct_mode,
                "tau_pred": "",
                "alpha": "",
                "correction_ratio": "",
                "valid_predicted_depth_ratio": float(
                    record["method_inputs"][direct_mode]["valid"].mean()
                ),
            },
        )
    )
    for row in rows:
        print("Run 36 metric row:", row)
    return rows


def load_run30_reference(run30_dir):
    metrics_path = Path(run30_dir) / "metrics.csv"
    rows = read_csv(metrics_path)
    output = {}
    for row in rows:
        if row.get("method") != "rgbd_source_depth_selected":
            continue
        output[row["group_key"]] = row
    return output


def reference_row(record, source):
    group = record["group"]
    row = source.get(group["group_key"])
    if row is None:
        return None
    metric_names = [
        "accuracy",
        "completeness",
        "precision",
        "recall",
        "fscore",
        "chamfer",
        "num_pred_points",
        "num_gt_points",
        "threshold_m",
        "normalized_distance",
        "dac_at_0_2_normalized",
        "normalized_dac_threshold",
    ]
    output = {
        "run": RUN_NAME,
        "split": group["split"],
        "scene": group["scene"],
        "num_views": int(group["num_views"]),
        "view_policy": group["view_policy"],
        "group_key": group["group_key"],
        "method": "run30_rgbd_source_depth_selected",
        "method_family": "reference_true_rgbd",
        "occlusion_proxy_ratio": float(
            record["occlusion_proxy"].mean()
        ),
        "ambiguity_proxy_ratio": float(
            record["ambiguity_proxy"].mean()
        ),
        "runtime_seconds": as_float(row, "runtime_seconds"),
        "uses_true_source_depth_for_inference": 1,
        "uses_predicted_depth_for_inference": 0,
        "uses_known_pose": 1,
        "uses_known_intrinsics": 1,
        "reference_only": 1,
    }
    for name in metric_names:
        value = as_float(row, name)
        if math.isfinite(value):
            output[name] = value
    return output


def summarize(metric_rows):
    output = []
    for split, method in sorted(
        {(row["split"], row["method"]) for row in metric_rows}
    ):
        rows = [
            row
            for row in metric_rows
            if row["split"] == split and row["method"] == method
        ]
        summary = {
            "run": RUN_NAME,
            "split": split,
            "method": method,
            "method_family": rows[0]["method_family"],
            "num_groups": len(rows),
        }
        for source, target in [
            ("accuracy", "mean_accuracy"),
            ("completeness", "mean_completeness"),
            ("precision", "mean_precision"),
            ("recall", "mean_recall"),
            ("fscore", "mean_fscore"),
            ("chamfer", "mean_chamfer"),
            ("normalized_distance", "mean_normalized_distance"),
            (
                "dac_at_0_2_normalized",
                "mean_dac_at_0_2_normalized",
            ),
        ]:
            values = [
                float(row[source])
                for row in rows
                if source in row and math.isfinite(float(row[source]))
            ]
            summary[target] = (
                float(np.mean(values)) if values else float("nan")
            )
        output.append(summary)
    return output


def limit_summary(metric_rows):
    def safe_mean(rows, key):
        values = [
            float(row[key])
            for row in rows
            if key in row and math.isfinite(float(row[key]))
        ]
        return float(np.mean(values)) if values else float("nan")

    output = []
    for split in sorted({row["split"] for row in metric_rows}):
        split_rows = [
            row for row in metric_rows if row["split"] == split
        ]
        group_diag = {
            row["group_key"]: (
                float(row["occlusion_proxy_ratio"]),
                float(row["ambiguity_proxy_ratio"]),
            )
            for row in split_rows
        }
        count = max(1, int(math.ceil(len(group_diag) / 3.0)))
        occ_keys = {
            item[0]
            for item in sorted(
                group_diag.items(),
                key=lambda item: (-item[1][0], item[0]),
            )[:count]
        }
        amb_keys = {
            item[0]
            for item in sorted(
                group_diag.items(),
                key=lambda item: (-item[1][1], item[0]),
            )[:count]
        }
        subsets = {
            "overall": lambda row: True,
            "occlusion_challenging": (
                lambda row: row["group_key"] in occ_keys
            ),
            "ambiguity_challenging": (
                lambda row: row["group_key"] in amb_keys
            ),
        }
        for subset_name, predicate in subsets.items():
            subset_rows = [row for row in split_rows if predicate(row)]
            for method in sorted(
                {row["method"] for row in subset_rows}
            ):
                rows = [
                    row
                    for row in subset_rows
                    if row["method"] == method
                ]
                output.append(
                    {
                        "run": RUN_NAME,
                        "split": split,
                        "limit_subset": subset_name,
                        "method": method,
                        "method_family": rows[0]["method_family"],
                        "num_groups": len(rows),
                        "mean_fscore": float(
                            np.mean([row["fscore"] for row in rows])
                        ),
                        "mean_precision": float(
                            np.mean([row["precision"] for row in rows])
                        ),
                        "mean_recall": float(
                            np.mean([row["recall"] for row in rows])
                        ),
                        "mean_normalized_distance": safe_mean(
                            rows,
                            "normalized_distance",
                        ),
                        "mean_dac_at_0_2_normalized": safe_mean(
                            rows,
                            "dac_at_0_2_normalized",
                        ),
                    }
                )
    return output


def find_limit_row(rows, split, subset, method):
    return next(
        row
        for row in rows
        if row["split"] == split
        and row["limit_subset"] == subset
        and row["method"] == method
    )


def gate_decision(limit_rows, selected):
    selected_method = (
        "predicted_depth_correction_raw"
        if selected["depth_scale_mode"] == "raw"
        else "predicted_depth_correction_scale_aligned"
    )
    comparisons = {}
    for subset in [
        "overall",
        "occlusion_challenging",
        "ambiguity_challenging",
    ]:
        candidates = [
            row
            for row in limit_rows
            if row["split"] == "val"
            and row["limit_subset"] == subset
            and row["method_family"] == "rgb_only_baseline"
        ]
        baseline = max(
            candidates,
            key=lambda row: row["mean_fscore"],
        )
        predicted = find_limit_row(
            limit_rows,
            "val",
            subset,
            selected_method,
        )
        comparisons[subset] = (baseline, predicted)
    overall_base, overall_pred = comparisons["overall"]
    occ_base, occ_pred = comparisons["occlusion_challenging"]
    amb_base, amb_pred = comparisons["ambiguity_challenging"]
    overall_delta = (
        overall_pred["mean_fscore"] - overall_base["mean_fscore"]
    )
    occ_delta = occ_pred["mean_fscore"] - occ_base["mean_fscore"]
    amb_delta = amb_pred["mean_fscore"] - amb_base["mean_fscore"]
    passed = (
        overall_delta >= GATE_MARGIN_F1
        and occ_delta >= 0.0
        and amb_delta >= 0.0
    )
    direct = find_limit_row(
        limit_rows,
        "val",
        "overall",
        "predicted_depth_direct_backprojection",
    )
    run30 = find_limit_row(
        limit_rows,
        "val",
        "overall",
        "run30_rgbd_source_depth_selected",
    )
    return [
        {
            "run": RUN_NAME,
            "selected_method": selected_method,
            "selected_depth_scale_mode": selected[
                "depth_scale_mode"
            ],
            "selected_tau_pred": selected["tau_pred"],
            "selected_alpha": selected["alpha"],
            "best_rgb_only_baseline_method": overall_base["method"],
            "validation_rgb_only_baseline_fscore": overall_base[
                "mean_fscore"
            ],
            "validation_predicted_correction_fscore": overall_pred[
                "mean_fscore"
            ],
            "delta_vs_rgb_only": overall_delta,
            "occlusion_delta_vs_rgb_only": occ_delta,
            "ambiguity_delta_vs_rgb_only": amb_delta,
            "validation_direct_predicted_depth_fscore": direct[
                "mean_fscore"
            ],
            "validation_run30_true_rgbd_fscore": run30["mean_fscore"],
            "delta_vs_run30_true_rgbd": (
                overall_pred["mean_fscore"] - run30["mean_fscore"]
            ),
            "overall_pass": int(overall_delta >= GATE_MARGIN_F1),
            "occlusion_non_regression_pass": int(occ_delta >= 0.0),
            "ambiguity_non_regression_pass": int(amb_delta >= 0.0),
            "pass_all_limits": int(passed),
            "gate_margin_f1": GATE_MARGIN_F1,
            "final_project_claim_changed": 0,
        }
    ]


def load_run32_limit_reference():
    matches = sorted(
        Path("/kaggle/input").rglob(
            "run_32_direct_rgbd_backprojection_baseline/limit_summary.csv"
        )
    )
    return read_csv(matches[0]) if matches else []


def comparison_table(limit_rows, run32_rows, selected):
    selected_method = (
        "predicted_depth_correction_raw"
        if selected["depth_scale_mode"] == "raw"
        else "predicted_depth_correction_scale_aligned"
    )
    methods = [
        "mvdust3r_rgb_only_all_candidates",
        "mvdust3r_confidence_fixed",
        selected_method,
        "predicted_depth_direct_backprojection",
        "run30_rgbd_source_depth_selected",
    ]
    output = []
    for split in ["val", "test"]:
        for subset in [
            "overall",
            "occlusion_challenging",
            "ambiguity_challenging",
        ]:
            for method in methods:
                try:
                    row = find_limit_row(
                        limit_rows,
                        split,
                        subset,
                        method,
                    )
                except StopIteration:
                    continue
                output.append(
                    {
                        "run": RUN_NAME,
                        "split": split,
                        "limit_subset": subset,
                        "method": method,
                        "input_contract": (
                            "true RGB-D source depth reference"
                            if method == "run30_rgbd_source_depth_selected"
                            else "RGB + predicted depth + known pose/intrinsics"
                            if "predicted_depth" in method
                            else "RGB-only MV-DUSt3R+"
                        ),
                        "mean_fscore": row["mean_fscore"],
                        "mean_precision": row["mean_precision"],
                        "mean_recall": row["mean_recall"],
                    }
                )
            direct32 = [
                row
                for row in run32_rows
                if row.get("split") == split
                and row.get("limit_subset") == subset
                and row.get("method") == "direct_rgbd_backprojection"
            ]
            if direct32:
                row = direct32[0]
                output.append(
                    {
                        "run": RUN_NAME,
                        "split": split,
                        "limit_subset": subset,
                        "method": "run32_direct_rgbd_backprojection",
                        "input_contract": "true RGB-D direct voxel reference",
                        "mean_fscore": as_float(row, "mean_fscore"),
                        "mean_precision": as_float(
                            row,
                            "mean_precision",
                        ),
                        "mean_recall": as_float(row, "mean_recall"),
                    }
                )
    return output


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    base.require_t4x2()
    run35_dir = locate_run_output(
        "run_35_predicted_depth_quality_diagnostic",
        "run_config.json",
    )
    run30_dir = locate_run_output(
        "run_30_rgbd_source_depth_correction",
        "metrics.csv",
    )
    depth_cache = PredictedDepthCache(run35_dir)
    global_scale = float(
        depth_cache.config["global_scale_fit_median"]
    )
    run30_reference = load_run30_reference(run30_dir)
    run32_limit_reference = load_run32_limit_reference()

    root = base.clone_repo()
    base.install_deps(root)
    posed_root = base.find_posed_images_root()
    scene_dirs = r27.discover_scene_dirs(posed_root)[:MAX_SCENES]
    scene_lookup = {path.name: path for path in scene_dirs}
    splits = r27.scene_splits(scene_dirs)
    manifest = r27.build_group_manifest(scene_dirs, splits)
    train_groups = r27.balanced_group_subset(
        [row for row in manifest if row["split"] == "train"],
        MAX_TRAIN_GROUPS,
    )
    eval_groups = r27.balanced_group_subset(
        [row for row in manifest if row["split"] in {"val", "test"}],
        MAX_EVAL_GROUPS,
    )
    _fit_groups, internal_val_groups, internal_val_scenes = (
        r27.split_internal_train_groups(train_groups)
    )
    print(
        "Run 36 groups:",
        {
            "internal_val": len(internal_val_groups),
            "external_eval": len(eval_groups),
            "internal_val_scenes": internal_val_scenes,
            "global_scale": global_scale,
        },
    )

    checkpoint = base.download_checkpoint(root)
    backbone = base.load_model(root, checkpoint)
    internal_records = [
        build_record(
            backbone,
            root,
            scene_lookup,
            group,
            out_dir / "internal_val_groups" / group["group_key"],
            depth_cache,
            global_scale,
        )
        for group in internal_val_groups
    ]
    selected, selected_by_mode, policy_rows = select_policies(
        internal_records
    )
    print("Run 36 selected policy:", selected)

    metric_rows = []
    cache_manifest_rows = []
    for group in eval_groups:
        record = build_record(
            backbone,
            root,
            scene_lookup,
            group,
            out_dir / "eval_groups" / group["group_key"],
            depth_cache,
            global_scale,
        )
        metric_rows.extend(
            evaluate_group(record, selected, selected_by_mode)
        )
        reference = reference_row(record, run30_reference)
        if reference is not None:
            metric_rows.append(reference)
        for view_file in record["view_files"]:
            cache_manifest_rows.append(
                {
                    "run": RUN_NAME,
                    "split": group["split"],
                    "scene": group["scene"],
                    "group_key": group["group_key"],
                    "frame": view_file.name,
                    "source_run35_dir": str(run35_dir),
                    "depth_model_name": depth_cache.config[
                        "depth_model_name"
                    ],
                    "depth_checkpoint": depth_cache.config[
                        "depth_checkpoint"
                    ],
                }
            )

    summary_rows = summarize(metric_rows)
    limit_rows = limit_summary(metric_rows)
    gate_rows = gate_decision(limit_rows, selected)
    comparison_rows = comparison_table(
        limit_rows,
        run32_limit_reference,
        selected,
    )
    write_csv_union(out_dir / "metrics.csv", metric_rows)
    write_csv_union(out_dir / "summary.csv", summary_rows)
    write_csv_union(out_dir / "limit_summary.csv", limit_rows)
    write_csv_union(out_dir / "policy_selection.csv", policy_rows)
    write_csv_union(out_dir / "gate_decision.csv", gate_rows)
    write_csv_union(out_dir / "comparison_table.csv", comparison_rows)
    write_csv_union(
        out_dir / "predicted_depth_cache_manifest.csv",
        cache_manifest_rows,
    )

    selected_method = gate_rows[0]["selected_method"]
    config = {
        "run": RUN_NAME,
        "purpose": (
            "Replace true RGB-D source depth with monocular predicted depth "
            "for candidate correction under known pose/intrinsics."
        ),
        "source_run35_dir": str(run35_dir),
        "source_run30_dir": str(run30_dir),
        "depth_model_name": depth_cache.config["depth_model_name"],
        "depth_checkpoint": depth_cache.config["depth_checkpoint"],
        "depth_scale_mode": selected["depth_scale_mode"],
        "global_scale_fit_median": global_scale,
        "selected_method": selected_method,
        "selected_tau_pred": selected["tau_pred"],
        "selected_alpha": selected["alpha"],
        "selected_policy_by_scale_mode": selected_by_mode,
        "tau_pred_grid": TAU_GRID,
        "alpha_grid": ALPHA_GRID,
        "num_scenes": len(scene_dirs),
        "scene_splits": splits,
        "num_internal_val_groups": len(internal_records),
        "num_eval_groups": len(eval_groups),
        "internal_val_scenes": internal_val_scenes,
        "max_candidates_per_group": MAX_CANDIDATES_PER_GROUP,
        "uses_true_source_depth_for_inference": False,
        "uses_true_source_depth_for_correction": False,
        "uses_true_source_depth_for_evaluation": True,
        "uses_predicted_depth_for_inference": True,
        "uses_known_pose": True,
        "uses_known_intrinsics": True,
        "input_contract": (
            "RGB images + predicted monocular depth + known "
            "pose/intrinsics"
        ),
        "method_contract": (
            "True source depth is forbidden from predicted-depth correction, "
            "residual gating, and direct predicted-depth backprojection."
        ),
        "evaluation_contract": (
            "The existing depth-derived proxy GT and true-depth hard-group "
            "diagnostics are evaluation-only."
        ),
        "claim_contract": (
            "Run 36 is an RGB-only image-input extension with predicted "
            "monocular depth and known pose/intrinsics. It is not the fully "
            "pose-free MV-DUSt3R+ paper setting."
        ),
        "final_project_claim_changed": False,
        "gate_margin_f1": GATE_MARGIN_F1,
        "runtime_seconds": time.time() - started,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print("Run 36 summary:")
    for row in summary_rows:
        print(row)
    print("Run 36 gate decision:", gate_rows[0])
    print("Run 36 config:", config)
    print("Run 36 output dir:", out_dir)


if __name__ == "__main__":
    main()
