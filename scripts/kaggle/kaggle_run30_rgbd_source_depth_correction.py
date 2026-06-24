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

RUN_NAME = "run_30_rgbd_source_depth_correction"
SEED = 3030
MAX_SCENES = int(os.environ.get("RUN30_MAX_SCENES", "0"))
MAX_TRAIN_GROUPS = int(os.environ.get("RUN30_MAX_TRAIN_GROUPS", "48"))
MAX_EVAL_GROUPS_RAW = os.environ.get("RUN30_MAX_EVAL_GROUPS", "0")
MAX_CANDIDATES_PER_GROUP = int(os.environ.get("RUN30_MAX_CANDIDATES_PER_GROUP", "3500"))
GATE_MARGIN_F1 = float(os.environ.get("RUN30_GATE_MARGIN_F1", "0.005"))

POLICIES = [
    {"name": "rgbd_full_alpha_1.00", "mode": "full", "alpha": 1.00, "residual_threshold_m": 0.0},
    {"name": "rgbd_full_alpha_0.75", "mode": "full", "alpha": 0.75, "residual_threshold_m": 0.0},
    {"name": "rgbd_full_alpha_0.50", "mode": "full", "alpha": 0.50, "residual_threshold_m": 0.0},
    {"name": "rgbd_residual_ge_0.10", "mode": "residual", "alpha": 1.00, "residual_threshold_m": 0.10},
    {"name": "rgbd_residual_ge_0.15", "mode": "residual", "alpha": 1.00, "residual_threshold_m": 0.15},
    {"name": "rgbd_residual_ge_0.30", "mode": "residual", "alpha": 1.00, "residual_threshold_m": 0.30},
    {"name": "rgbd_low_support_or_residual_0.15", "mode": "low_support_or_residual", "alpha": 1.00, "residual_threshold_m": 0.15},
]


def parse_optional_positive_limit(raw_value):
    if raw_value is None:
        return None
    text = str(raw_value).strip().lower()
    if text in {"", "none", "null", "all", "unlimited"}:
        return None
    value = int(text)
    return value if value > 0 else None


MAX_EVAL_GROUPS = parse_optional_positive_limit(MAX_EVAL_GROUPS_RAW)


def confidence_mask(conf, num_views):
    percentile = base.FIXED_FINAL_CONF_BY_VIEW_COUNT.get(num_views, base.CONF_PERCENT)
    threshold = float(np.quantile(conf, percentile / 100.0))
    return conf >= threshold, percentile, threshold


def correction_mask(record, policy):
    valid = record["valid"].astype(bool)
    if policy["mode"] == "full":
        return valid
    if policy["mode"] == "residual":
        return valid & (record["residual_m"] >= float(policy["residual_threshold_m"]))
    if policy["mode"] == "low_support_or_residual":
        low_support = record["occlusion_proxy"] > 0.0
        high_residual = record["residual_m"] >= float(policy["residual_threshold_m"])
        return valid & (low_support | high_residual)
    raise ValueError(f"Unknown policy mode: {policy['mode']}")


def apply_policy(record, policy):
    mask = correction_mask(record, policy)
    corrected = record["points"].copy()
    alpha = float(policy["alpha"])
    corrected[mask] = (1.0 - alpha) * record["points"][mask] + alpha * record["targets"][mask]
    return corrected.astype(np.float32), mask


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
    for policy in POLICIES:
        fscores = []
        correction_ratios = []
        for record in records:
            points, mask = apply_policy(record, policy)
            metrics = base.compute_metrics(base.downsample(points, base.MAX_POINTS), record["gt"])
            fscores.append(metrics["fscore"])
            correction_ratios.append(float(mask.mean()))
        rows.append(
            {
                "policy": policy["name"],
                "mode": policy["mode"],
                "alpha": float(policy["alpha"]),
                "residual_threshold_m": float(policy["residual_threshold_m"]),
                "mean_reconstruction_fscore": float(np.mean(fscores)),
                "mean_correction_ratio": float(np.mean(correction_ratios)),
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

    selected_name = selected_policy["policy"]
    selected_policy_def = next(policy for policy in POLICIES if policy["name"] == selected_name)
    selected_points, selected_mask = apply_policy(record, selected_policy_def)
    rows.append(
        score_points(
            selected_points,
            record,
            "rgbd_source_depth_selected",
            "resource_expanded_depth",
            {
                "selected_ratio": 1.0,
                "correction_ratio": float(selected_mask.mean()),
                "policy": selected_name,
                "mode": selected_policy_def["mode"],
                "alpha": float(selected_policy_def["alpha"]),
                "residual_threshold_m": float(selected_policy_def["residual_threshold_m"]),
            },
        )
    )

    for policy in POLICIES:
        if policy["name"] == selected_name:
            continue
        points, mask = apply_policy(record, policy)
        rows.append(
            score_points(
                points,
                record,
                policy["name"],
                "diagnostic_depth_policy",
                {
                    "selected_ratio": 1.0,
                    "correction_ratio": float(mask.mean()),
                    "policy": policy["name"],
                    "mode": policy["mode"],
                    "alpha": float(policy["alpha"]),
                    "residual_threshold_m": float(policy["residual_threshold_m"]),
                },
            )
        )
    for row in rows:
        print("Run 30 metric row:", row)
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
                "mean_normalized_distance": float(
                    np.mean([row["normalized_distance"] for row in rows])
                ),
                "mean_dac_at_0_2_normalized": float(
                    np.mean([row["dac_at_0_2_normalized"] for row in rows])
                ),
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
                        "mean_normalized_distance": float(
                            np.mean([row["normalized_distance"] for row in rows])
                        ),
                        "mean_dac_at_0_2_normalized": float(
                            np.mean([row["dac_at_0_2_normalized"] for row in rows])
                        ),
                    }
                )
    return output


def subset_comparison(rows, subset):
    candidates = [row for row in rows if row["split"] == "val" and row["limit_subset"] == subset]
    baseline = max(
        [row for row in candidates if row["method_family"] == "baseline"],
        key=lambda row: row["mean_fscore"],
    )
    selected = next(row for row in candidates if row["method"] == "rgbd_source_depth_selected")
    return baseline, selected


def gate_decision(limit_rows):
    comparisons = {
        subset: subset_comparison(limit_rows, subset)
        for subset in ["overall", "occlusion_challenging", "ambiguity_challenging"]
    }
    overall_base, overall_selected = comparisons["overall"]
    occ_base, occ_selected = comparisons["occlusion_challenging"]
    amb_base, amb_selected = comparisons["ambiguity_challenging"]
    overall_delta = overall_selected["mean_fscore"] - overall_base["mean_fscore"]
    occ_delta = occ_selected["mean_fscore"] - occ_base["mean_fscore"]
    amb_delta = amb_selected["mean_fscore"] - amb_base["mean_fscore"]
    passed = overall_delta >= GATE_MARGIN_F1 and occ_delta >= 0.0 and amb_delta >= 0.0
    return [
        {
            "run": RUN_NAME,
            "selected_method": "rgbd_source_depth_selected" if passed else overall_base["method"],
            "best_baseline_method": overall_base["method"],
            "validation_best_baseline_fscore": overall_base["mean_fscore"],
            "validation_rgbd_fscore": overall_selected["mean_fscore"],
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
    valid = record["valid"]
    high_residual = record["residual_m"] >= 0.15
    return {
        "run": RUN_NAME,
        "stage": stage,
        "split": record["group_row"]["split"],
        "scene": record["group_row"]["scene"],
        "group_key": record["group_row"]["group_key"],
        "num_views": record["group_row"]["num_views"],
        "view_policy": record["group_row"]["view_policy"],
        "num_candidates": len(record["points"]),
        "valid_source_depth_ratio": float(valid.mean()),
        "mean_source_depth_m": float(record["source_depth"][valid].mean()) if valid.any() else 0.0,
        "mean_source_residual_m": float(record["residual_m"][valid].mean()) if valid.any() else 0.0,
        "wrong_depth_ratio": float((valid & high_residual).mean()),
        "low_support_valid_ratio": float(record["occlusion_proxy"].mean()),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    r28.MAX_CANDIDATES_PER_GROUP = MAX_CANDIDATES_PER_GROUP
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    r27.validate_static_configuration()
    base.require_t4x2()
    root = base.clone_repo()
    base.install_deps(root)
    posed_root = base.find_posed_images_root()
    all_scene_dirs = r27.discover_scene_dirs(posed_root)
    scene_dirs = all_scene_dirs[:MAX_SCENES] if MAX_SCENES > 0 else all_scene_dirs
    scene_lookup = {path.name: path for path in scene_dirs}
    splits = r27.scene_splits(scene_dirs)
    manifest = r27.build_group_manifest(scene_dirs, splits)
    train_groups = r27.balanced_group_subset(
        [row for row in manifest if row["split"] == "train"],
        MAX_TRAIN_GROUPS,
    )
    eval_group_candidates = [
        row for row in manifest if row["split"] in {"val", "test"}
    ]
    eval_groups = (
        eval_group_candidates
        if MAX_EVAL_GROUPS is None
        else r27.balanced_group_subset(eval_group_candidates, MAX_EVAL_GROUPS)
    )
    evaluated_scene_ids = sorted({row["scene"] for row in eval_groups})
    _fit_groups, internal_val_groups, internal_val_scenes = r27.split_internal_train_groups(train_groups)
    print("Run 30 splits:", splits)
    print("Run 30 group counts:", {"train": len(train_groups), "internal_val": len(internal_val_groups), "eval": len(eval_groups)})

    checkpoint = base.download_checkpoint(root)
    backbone = base.load_model(root, checkpoint)
    internal_val_records = []
    correction_rows = []
    for group in train_groups:
        record = r28.run_group(backbone, root, scene_lookup, group, out_dir / "train_groups" / group["group_key"])
        stage = "internal_val" if group["scene"] in internal_val_scenes else "fit"
        if stage == "internal_val":
            internal_val_records.append(record)
        correction_rows.append(correction_summary(record, stage))

    selected_policy, policy_rows = select_policy(internal_val_records)
    print("Run 30 selected policy:", selected_policy)

    metric_rows = []
    for group in eval_groups:
        record = r28.run_group(backbone, root, scene_lookup, group, out_dir / "eval_groups" / group["group_key"])
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
        "source_run29_diagnosis": "RGB-only monodepth source-ray correction regresses both occlusion and ambiguity; source-depth oracle still has large headroom.",
        "num_scenes": len(scene_dirs),
        "scene_limit": MAX_SCENES if MAX_SCENES > 0 else None,
        "num_discovered_scenes": len(all_scene_dirs),
        "scene_splits": splits,
        "max_eval_groups_raw": MAX_EVAL_GROUPS_RAW,
        "max_eval_groups_resolved": MAX_EVAL_GROUPS,
        "num_total_eval_groups_before_cap": len(eval_group_candidates),
        "num_eval_groups_after_cap": len(eval_groups),
        "evaluated_scene_count": len(evaluated_scene_ids),
        "evaluated_scene_ids": evaluated_scene_ids,
        "num_internal_val_groups": len(internal_val_records),
        "num_eval_groups": len(eval_groups),
        "max_candidates_per_group": MAX_CANDIDATES_PER_GROUP,
        "policies": POLICIES,
        "selected_policy": selected_policy,
        "gate_margin_f1": GATE_MARGIN_F1,
        "runtime_seconds": time.time() - started,
        "inference_contract": "Uses source depth maps from the input posed RGB-D frames plus known camera poses/intrinsics at inference. This is not an RGB-only reconstruction claim.",
        "claim_contract": "If this gate passes, the remaining occlusion/repeated-depth limits are solved under an explicit RGB-D/resource-expanded setting; RGB-only remains unresolved unless a no-depth method also passes.",
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print("Run 30 config:", config)
    print("Run 30 summary:")
    for row in summary_rows:
        print(row)
    print("Run 30 gate decision:")
    for row in gate_rows:
        print(row)


if __name__ == "__main__":
    main()
