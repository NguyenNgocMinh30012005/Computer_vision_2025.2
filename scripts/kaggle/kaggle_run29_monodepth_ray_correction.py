import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image


RAW_RUN28 = "https://raw.githubusercontent.com/NguyenNgocMinh30012005/Computer_vision_2025.2/main/scripts/kaggle/kaggle_run28_ray_depth_correction.py"


def ensure_helper_module(module_name, raw_url):
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        helper_path = Path(f"{module_name}.py")
        print(f"Downloading helper module {module_name} from GitHub raw")
        urllib.request.urlretrieve(raw_url, helper_path)
        sys.path.insert(0, str(Path.cwd()))
        return __import__(module_name)


r28 = ensure_helper_module("kaggle_run28_ray_depth_correction", RAW_RUN28)
r27 = r28.r27
base = r28.base

RUN_NAME = "run_29_monodepth_ray_correction"
SEED = 2929
MAX_SCENES = int(os.environ.get("RUN29_MAX_SCENES", "30"))
MAX_TRAIN_GROUPS = int(os.environ.get("RUN29_MAX_TRAIN_GROUPS", "48"))
MAX_EVAL_GROUPS = int(os.environ.get("RUN29_MAX_EVAL_GROUPS", "36"))
MAX_CANDIDATES_PER_GROUP = int(os.environ.get("RUN29_MAX_CANDIDATES_PER_GROUP", "3500"))
MONODEPTH_MODEL = os.environ.get(
    "RUN29_MONODEPTH_MODEL",
    "depth-anything/Depth-Anything-V2-Small-hf",
)
GATE_MARGIN_F1 = float(os.environ.get("RUN29_GATE_MARGIN_F1", "0.005"))
POLICY_ALPHAS = [0.10, 0.25, 0.50, 0.75, 1.00]
DEPTH_VARIANTS = ["raw", "inverse", "inv_disp"]
MIN_VALID_DEPTH = 1e-6


def ensure_monodepth_pipeline():
    def import_pipeline():
        from transformers import pipeline

        return pipeline

    try:
        pipeline = import_pipeline()
    except Exception as exc:
        print("Installing transformers for Run 29 monodepth:", repr(exc))
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-cache-dir",
                "transformers>=4.46.0",
                "safetensors",
                "accelerate",
                "timm",
            ]
        )
        pipeline = import_pipeline()

    device = 0 if torch.cuda.is_available() else -1
    try:
        return pipeline("depth-estimation", model=MONODEPTH_MODEL, device=device)
    except Exception as exc:
        print("First monodepth load failed; upgrading transformers:", repr(exc))
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-cache-dir",
                "--upgrade",
                "transformers>=4.46.0",
                "safetensors",
                "accelerate",
                "timm",
            ]
        )
        pipeline = import_pipeline()
        return pipeline("depth-estimation", model=MONODEPTH_MODEL, device=device)


def resize_array(array, width, height, resample=Image.BICUBIC):
    image = Image.fromarray(array.astype(np.float32), mode="F")
    return np.asarray(image.resize((width, height), resample=resample), dtype=np.float32)


def normalize_depth_map(depth):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth[~np.isfinite(depth)] = np.nan
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth, dtype=np.float32)
    lo, hi = np.nanpercentile(depth, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(depth))
        hi = float(np.nanmax(depth)) + 1e-6
    norm = (depth - lo) / max(hi - lo, 1e-6)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def monodepth_map(pipe, image_path, cache):
    key = str(image_path)
    if key in cache:
        return cache[key]
    image = Image.open(image_path).convert("RGB")
    with torch.inference_mode():
        output = pipe(image)
    if "predicted_depth" in output:
        predicted = output["predicted_depth"]
        if isinstance(predicted, torch.Tensor):
            depth = predicted.detach().float().cpu().squeeze().numpy()
        else:
            depth = np.asarray(predicted, dtype=np.float32).squeeze()
    elif "depth" in output:
        depth = np.asarray(output["depth"], dtype=np.float32)
    else:
        raise RuntimeError(f"Unexpected monodepth output keys: {sorted(output.keys())}")
    cache[key] = normalize_depth_map(depth)
    return cache[key]


def variant_depth(depth_norm, variant):
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    if variant == "raw":
        return 0.25 + 5.75 * depth_norm
    if variant == "inverse":
        return 0.25 + 5.75 * (1.0 - depth_norm)
    if variant == "inv_disp":
        disparity = 0.05 + 0.95 * depth_norm
        inv = 1.0 / disparity
        inv = normalize_depth_map(inv)
        return 0.25 + 5.75 * inv
    raise ValueError(f"Unknown depth variant: {variant}")


def monodepth_targets(view_files, xs, ys, view_ids, image_h, image_w, pipe, cache, variant):
    poses = [base.parse_pose(str(path).replace(".jpg", ".txt")).astype(np.float32) for path in view_files]
    first_pose_inv = np.linalg.inv(poses[0]).astype(np.float32)
    targets = np.zeros((len(xs), 3), dtype=np.float32)
    valid = np.zeros(len(xs), dtype=bool)
    sampled_depth = np.zeros(len(xs), dtype=np.float32)

    for view_index, view_file in enumerate(view_files):
        candidate_indices = np.where(view_ids == view_index)[0]
        if not len(candidate_indices):
            continue
        depth_norm = monodepth_map(pipe, view_file, cache)
        depth_norm = resize_array(depth_norm, image_w, image_h)
        pseudo_depth = variant_depth(depth_norm, variant)
        x_idx = np.clip(np.rint(xs[candidate_indices]).astype(np.int32), 0, image_w - 1)
        y_idx = np.clip(np.rint(ys[candidate_indices]).astype(np.int32), 0, image_h - 1)
        z = pseudo_depth[y_idx, x_idx]
        usable = np.isfinite(z) & (z > MIN_VALID_DEPTH)
        if not usable.any():
            continue
        usable_indices = candidate_indices[usable]
        z = z[usable]
        x = x_idx[usable].astype(np.float32)
        y = y_idx[usable].astype(np.float32)
        fx, fy, cx, cy = r28.intrinsics((image_h, image_w))
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


def robust_similarity_targets(points, metric_targets, valid):
    if int(valid.sum()) < 100:
        raise RuntimeError(f"Insufficient valid monodepth targets: {int(valid.sum())}")
    pred_valid = points[valid]
    target_valid = metric_targets[valid]
    pred_center = np.median(pred_valid, axis=0, keepdims=True)
    target_center = np.median(target_valid, axis=0, keepdims=True)
    pred_radius = np.linalg.norm(pred_valid - pred_center, axis=1)
    target_radius = np.linalg.norm(target_valid - target_center, axis=1)
    pred_scale = float(np.quantile(pred_radius, 0.90)) + 1e-8
    target_scale = float(np.quantile(target_radius, 0.90)) + 1e-8
    targets_in_prediction = (
        (metric_targets - target_center) / target_scale * pred_scale + pred_center
    ).astype(np.float32)
    return targets_in_prediction, {
        f"{RUN_NAME}_pred_scale": pred_scale,
        f"{RUN_NAME}_target_scale": target_scale,
        f"{RUN_NAME}_target_valid_ratio": float(valid.mean()),
    }


def run_group(backbone, root, scene_lookup, group_row, out_dir, pipe, depth_cache):
    view_files = r27.choose_group_views(scene_lookup, group_row)
    print("Run 29 group views:", {"group": group_row["group_key"], "views": [path.name for path in view_files]})
    output, _glb, runtime = base.run_inference(backbone, root, view_files, out_dir)
    gt, _stats = base.build_gt_cloud(view_files)
    points, conf, xs, ys, view_ids, image_h, image_w = r27.output_to_candidates(output)
    points, conf, xs, ys, view_ids = r27.subsample_candidates(
        points,
        conf,
        xs,
        ys,
        view_ids,
        MAX_CANDIDATES_PER_GROUP,
        r27.stable_seed("run29-" + group_row["group_key"]),
    )
    base_features = r27.build_features(points, conf, xs, ys, view_ids, image_h, image_w, group_row)
    features, photo_stats = r27.aggregate_image_pair_features(
        base_features,
        points,
        conf,
        xs,
        ys,
        view_ids,
        view_files,
        group_row,
        image_h,
        image_w,
    )
    oracle_metric, oracle_valid, _source_depth = r28.source_ray_targets(
        view_files,
        xs,
        ys,
        view_ids,
        image_h,
        image_w,
    )
    oracle_targets, oracle_residual_m, oracle_alignment_stats = r28.paired_similarity_targets(
        points,
        oracle_metric,
        oracle_valid,
    )
    low_support = features[:, r27.SUPPORT_FRAC_010_INDEX] < 0.125
    has_projection = features[:, r27.PHOTO_TARGET_COUNT_INDEX] > 0.0
    occlusion_proxy = (oracle_valid & low_support & has_projection).astype(np.float32)
    ambiguity_proxy = (oracle_valid & (oracle_residual_m >= 0.15)).astype(np.float32)

    target_by_variant = {}
    valid_by_variant = {}
    variant_stats = {}
    for variant in DEPTH_VARIANTS:
        metric_targets, valid, sampled_depth = monodepth_targets(
            view_files,
            xs,
            ys,
            view_ids,
            image_h,
            image_w,
            pipe,
            depth_cache,
            variant,
        )
        targets, stats = robust_similarity_targets(points, metric_targets, valid)
        target_by_variant[variant] = targets
        valid_by_variant[variant] = valid
        variant_stats[variant] = {
            **stats,
            "sampled_depth_mean": float(sampled_depth[valid].mean()) if valid.any() else 0.0,
        }

    del output
    torch.cuda.empty_cache()
    return {
        "points": points,
        "conf": conf,
        "features": features,
        "gt": gt,
        "runtime_seconds": runtime,
        "view_files": view_files,
        "photo_stats": photo_stats,
        "group_row": group_row,
        "oracle_targets": oracle_targets,
        "oracle_valid": oracle_valid,
        "oracle_residual_m": oracle_residual_m,
        "oracle_alignment_stats": oracle_alignment_stats,
        "occlusion_proxy": occlusion_proxy,
        "ambiguity_proxy": ambiguity_proxy,
        "target_by_variant": target_by_variant,
        "valid_by_variant": valid_by_variant,
        "variant_stats": variant_stats,
    }


def confidence_mask(conf, num_views):
    percentile = base.FIXED_FINAL_CONF_BY_VIEW_COUNT.get(num_views, base.CONF_PERCENT)
    threshold = float(np.quantile(conf, percentile / 100.0))
    return conf >= threshold, percentile, threshold


def corrected_points(record, variant, alpha):
    targets = record["target_by_variant"][variant]
    valid = record["valid_by_variant"][variant]
    corrected = record["points"].copy()
    corrected[valid] = (1.0 - alpha) * record["points"][valid] + alpha * targets[valid]
    return corrected.astype(np.float32)


def score_points(points, record, method, family, extra):
    metrics = base.compute_metrics(base.downsample(points.astype(np.float32), base.MAX_POINTS), record["gt"])
    group = record["group_row"]
    return {
        "run": RUN_NAME,
        "split": group["split"],
        "scene": group["scene"],
        "num_views": int(group["num_views"]),
        "view_policy": group["view_policy"],
        "group_key": group["group_key"],
        "method": method,
        "method_family": family,
        "occlusion_proxy_ratio": float(record["occlusion_proxy"].mean()),
        "ambiguity_proxy_ratio": float(record["ambiguity_proxy"].mean()),
        "runtime_seconds": record["runtime_seconds"],
        **record["photo_stats"],
        **extra,
        **metrics,
    }


def select_policy(records):
    rows = []
    for variant in DEPTH_VARIANTS:
        for alpha in POLICY_ALPHAS:
            fscores = []
            for record in records:
                points = corrected_points(record, variant, alpha)
                metrics = base.compute_metrics(base.downsample(points, base.MAX_POINTS), record["gt"])
                fscores.append(metrics["fscore"])
            rows.append(
                {
                    "variant": variant,
                    "alpha": alpha,
                    "mean_reconstruction_fscore": float(np.mean(fscores)),
                }
            )
    return max(rows, key=lambda row: row["mean_reconstruction_fscore"]), rows


def evaluate_group(record, selected_policy):
    group = record["group_row"]
    conf_mask, conf_percent, conf_threshold = confidence_mask(record["conf"], int(group["num_views"]))
    rows = [
        score_points(
            record["points"],
            record,
            "all_candidates",
            "baseline",
            {"selected_ratio": 1.0},
        ),
        score_points(
            record["points"][conf_mask],
            record,
            "confidence_fixed_final",
            "baseline",
            {
                "selected_ratio": float(conf_mask.mean()),
                "conf_percent": conf_percent,
                "conf_threshold": conf_threshold,
            },
        ),
    ]
    for variant in DEPTH_VARIANTS:
        alpha = selected_policy["alpha"] if variant == selected_policy["variant"] else 0.25
        method = "mdrc_internal_selected" if variant == selected_policy["variant"] else f"mdrc_{variant}_alpha_{alpha:.2f}"
        family = "learned_gate_candidate" if method == "mdrc_internal_selected" else "diagnostic_candidate"
        points = corrected_points(record, variant, alpha)
        rows.append(
            score_points(
                points,
                record,
                method,
                family,
                {
                    "selected_ratio": 1.0,
                    "variant": variant,
                    "alpha": alpha,
                    **record["variant_stats"][variant],
                },
            )
        )
    oracle = record["points"].copy()
    oracle[record["oracle_valid"]] = record["oracle_targets"][record["oracle_valid"]]
    rows.append(
        score_points(
            oracle,
            record,
            "oracle_source_depth_correction",
            "diagnostic_oracle",
            {
                "selected_ratio": 1.0,
                "valid_target_ratio": float(record["oracle_valid"].mean()),
                **record["oracle_alignment_stats"],
            },
        )
    )
    for row in rows:
        print("Run 29 metric row:", row)
    return rows


def summarize(metric_rows):
    output = []
    for split, method in sorted({(row["split"], row["method"]) for row in metric_rows}):
        rows = [row for row in metric_rows if row["split"] == split and row["method"] == method]
        output.append(
            {
                "run": RUN_NAME,
                "split": split,
                "method": method,
                "method_family": rows[0]["method_family"],
                "num_groups": len(rows),
                "mean_fscore": float(np.mean([row["fscore"] for row in rows])),
                "mean_precision": float(np.mean([row["precision"] for row in rows])),
                "mean_recall": float(np.mean([row["recall"] for row in rows])),
                "mean_chamfer": float(np.mean([row["chamfer"] for row in rows])),
            }
        )
    return output


def limit_summary(metric_rows):
    output = []
    for split in sorted({row["split"] for row in metric_rows}):
        split_rows = [row for row in metric_rows if row["split"] == split]
        group_diag = {
            row["group_key"]: (
                float(row["occlusion_proxy_ratio"]),
                float(row["ambiguity_proxy_ratio"]),
            )
            for row in split_rows
        }
        count = max(1, int(math.ceil(len(group_diag) / 3.0)))
        occ_keys = {item[0] for item in sorted(group_diag.items(), key=lambda item: (-item[1][0], item[0]))[:count]}
        amb_keys = {item[0] for item in sorted(group_diag.items(), key=lambda item: (-item[1][1], item[0]))[:count]}
        subsets = {
            "overall": lambda row: True,
            "occlusion_challenging": lambda row: row["group_key"] in occ_keys,
            "ambiguity_challenging": lambda row: row["group_key"] in amb_keys,
        }
        for subset_name, predicate in subsets.items():
            subset_rows = [row for row in split_rows if predicate(row)]
            for method in sorted({row["method"] for row in subset_rows}):
                rows = [row for row in subset_rows if row["method"] == method]
                output.append(
                    {
                        "run": RUN_NAME,
                        "split": split,
                        "limit_subset": subset_name,
                        "method": method,
                        "method_family": rows[0]["method_family"],
                        "num_groups": len(rows),
                        "mean_fscore": float(np.mean([row["fscore"] for row in rows])),
                        "mean_precision": float(np.mean([row["precision"] for row in rows])),
                        "mean_recall": float(np.mean([row["recall"] for row in rows])),
                    }
                )
    return output


def subset_comparison(rows, subset):
    candidates = [row for row in rows if row["split"] == "val" and row["limit_subset"] == subset]
    baseline = max(
        [row for row in candidates if row["method_family"] == "baseline"],
        key=lambda row: row["mean_fscore"],
    )
    learned = next(row for row in candidates if row["method"] == "mdrc_internal_selected")
    return baseline, learned


def gate_decision(limit_rows):
    comparisons = {
        subset: subset_comparison(limit_rows, subset)
        for subset in ["overall", "occlusion_challenging", "ambiguity_challenging"]
    }
    overall_base, overall_learned = comparisons["overall"]
    occ_base, occ_learned = comparisons["occlusion_challenging"]
    amb_base, amb_learned = comparisons["ambiguity_challenging"]
    overall_delta = overall_learned["mean_fscore"] - overall_base["mean_fscore"]
    occ_delta = occ_learned["mean_fscore"] - occ_base["mean_fscore"]
    amb_delta = amb_learned["mean_fscore"] - amb_base["mean_fscore"]
    passed = overall_delta >= GATE_MARGIN_F1 and occ_delta >= 0.0 and amb_delta >= 0.0
    return [
        {
            "run": RUN_NAME,
            "selected_method": "mdrc_internal_selected" if passed else overall_base["method"],
            "best_baseline_method": overall_base["method"],
            "validation_best_baseline_fscore": overall_base["mean_fscore"],
            "validation_learned_fscore": overall_learned["mean_fscore"],
            "delta_vs_best_baseline": overall_delta,
            "occlusion_delta_vs_best_baseline": occ_delta,
            "ambiguity_delta_vs_best_baseline": amb_delta,
            "overall_pass": int(overall_delta >= GATE_MARGIN_F1),
            "occlusion_non_regression_pass": int(occ_delta >= 0.0),
            "ambiguity_non_regression_pass": int(amb_delta >= 0.0),
            "pass_all_limits": int(passed),
            "gate_margin_f1": GATE_MARGIN_F1,
        }
    ]


def correction_summary(record, stage):
    valid = record["oracle_valid"]
    return {
        "run": RUN_NAME,
        "stage": stage,
        "split": record["group_row"]["split"],
        "scene": record["group_row"]["scene"],
        "group_key": record["group_row"]["group_key"],
        "num_views": record["group_row"]["num_views"],
        "view_policy": record["group_row"]["view_policy"],
        "num_candidates": len(record["points"]),
        "oracle_valid_source_depth_ratio": float(valid.mean()),
        "oracle_mean_residual_m": float(record["oracle_residual_m"][valid].mean()) if valid.any() else 0.0,
        "oracle_wrong_depth_ratio": float(record["ambiguity_proxy"].mean()),
        "low_support_valid_ratio": float(record["occlusion_proxy"].mean()),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    r27.validate_static_configuration()
    base.require_t4x2()
    root = base.clone_repo()
    base.install_deps(root)
    pipe = ensure_monodepth_pipeline()
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
    _fit_groups, internal_val_groups, internal_val_scenes = r27.split_internal_train_groups(train_groups)
    print("Run 29 splits:", splits)
    print("Run 29 group counts:", {"train": len(train_groups), "internal_val": len(internal_val_groups), "eval": len(eval_groups)})

    checkpoint = base.download_checkpoint(root)
    backbone = base.load_model(root, checkpoint)
    depth_cache = {}
    internal_val_records = []
    correction_rows = []
    for group in train_groups:
        record = run_group(backbone, root, scene_lookup, group, out_dir / "train_groups" / group["group_key"], pipe, depth_cache)
        stage = "internal_val" if group["scene"] in internal_val_scenes else "fit"
        if stage == "internal_val":
            internal_val_records.append(record)
        correction_rows.append(correction_summary(record, stage))

    selected_policy, policy_rows = select_policy(internal_val_records)
    print("Run 29 selected policy:", selected_policy)

    metric_rows = []
    for group in eval_groups:
        record = run_group(backbone, root, scene_lookup, group, out_dir / "eval_groups" / group["group_key"], pipe, depth_cache)
        correction_rows.append(correction_summary(record, "external_eval"))
        metric_rows.extend(evaluate_group(record, selected_policy))

    summary_rows = summarize(metric_rows)
    limit_rows = limit_summary(metric_rows)
    gate_rows = gate_decision(limit_rows)
    r27.write_csv_union(out_dir / "correction_label_summary.csv", correction_rows)
    r27.write_csv_union(out_dir / "policy_selection.csv", policy_rows)
    r27.write_csv_union(out_dir / "metrics.csv", metric_rows)
    r27.write_csv_union(out_dir / "summary.csv", summary_rows)
    r27.write_csv_union(out_dir / "limit_summary.csv", limit_rows)
    r27.write_csv_union(out_dir / "gate_decision.csv", gate_rows)
    config = {
        "run": RUN_NAME,
        "self_contained": True,
        "source_run28_diagnosis": "Run 28 oracle source-depth correction has large headroom but the learned MLP regresses occlusion.",
        "monodepth_model": MONODEPTH_MODEL,
        "num_scenes": len(scene_dirs),
        "scene_splits": splits,
        "num_internal_val_groups": len(internal_val_records),
        "num_eval_groups": len(eval_groups),
        "max_candidates_per_group": MAX_CANDIDATES_PER_GROUP,
        "depth_variants": DEPTH_VARIANTS,
        "policy_alphas": POLICY_ALPHAS,
        "selected_policy": selected_policy,
        "gate_margin_f1": GATE_MARGIN_F1,
        "runtime_seconds": time.time() - started,
        "inference_contract": "Uses RGB-only pretrained monocular depth and known input camera poses/intrinsics; no GT depth is used for candidate correction.",
        "oracle_contract": "Oracle source-depth correction is diagnostic only and excluded from policy selection and the gate.",
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print("Run 29 config:", config)
    print("Run 29 summary:")
    for row in summary_rows:
        print(row)
    print("Run 29 gate decision:")
    for row in gate_rows:
        print(row)


if __name__ == "__main__":
    main()
