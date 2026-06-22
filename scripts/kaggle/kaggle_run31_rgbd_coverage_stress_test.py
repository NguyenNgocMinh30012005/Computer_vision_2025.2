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

RUN_NAME = "run_31_rgbd_coverage_stress_test"
SEED = 3131
MAX_SCENES = int(os.environ.get("RUN31_MAX_SCENES", "30"))
FRAME_POOL_LIMIT = int(os.environ.get("RUN31_FRAME_POOL_LIMIT", "80"))
VARIANTS_PER_COMBINATION = int(os.environ.get("RUN31_VARIANTS_PER_COMBINATION", "2"))
MAX_GROUPS = int(os.environ.get("RUN31_MAX_GROUPS", "360"))
MAX_CANDIDATES_PER_GROUP = int(os.environ.get("RUN31_MAX_CANDIDATES_PER_GROUP", "3500"))
BOOTSTRAP_SAMPLES = int(os.environ.get("RUN31_BOOTSTRAP_SAMPLES", "2000"))
GATE_MARGIN_F1 = float(os.environ.get("RUN31_GATE_MARGIN_F1", "0.005"))
VIEW_COUNTS = [3, 4, 5]
VIEW_POLICIES = ["hybrid", "diversity_aware"]
FIXED_POLICY = {
    "name": "rgbd_residual_ge_0.30",
    "mode": "residual",
    "alpha": 1.0,
    "residual_threshold_m": 0.30,
}
RUN30_REFERENCE = {
    "validation_best_baseline_fscore": 0.1194,
    "validation_rgbd_fscore": 0.1753,
    "validation_delta": 0.0559,
    "validation_occlusion_delta": 0.1458,
    "validation_ambiguity_delta": 0.1377,
    "test_all_candidates_fscore": 0.1758,
    "test_rgbd_fscore": 0.2764,
    "test_delta": 0.1007,
}


def unique_paths(paths):
    output = []
    seen = set()
    for path in paths:
        if path.name not in seen:
            output.append(path)
            seen.add(path.name)
    return output


def camera_center(jpg_path):
    pose = base.parse_pose(str(jpg_path).replace(".jpg", ".txt"))
    return pose[:3, 3].astype(np.float32)


def evenly_spaced_indices(start, end, count):
    raw = np.linspace(start, end, count)
    indices = []
    for value in raw:
        index = int(round(float(value)))
        if not indices or index > indices[-1]:
            indices.append(index)
        else:
            indices.append(indices[-1] + 1)
    overflow = max(0, indices[-1] - end)
    if overflow:
        indices = [index - overflow for index in indices]
    return indices


def select_hybrid_variant(jpgs, num_views, variant_id):
    if len(jpgs) < num_views:
        raise RuntimeError(f"Only {len(jpgs)} frames are available for {num_views} views")
    if VARIANTS_PER_COMBINATION == 1:
        center_fraction = 0.50
    else:
        center_fraction = 0.22 + 0.56 * variant_id / (VARIANTS_PER_COMBINATION - 1)
    span = max(num_views - 1, int(round(0.32 * (len(jpgs) - 1))))
    center = int(round(center_fraction * (len(jpgs) - 1)))
    start = max(0, min(center - span // 2, len(jpgs) - 1 - span))
    end = min(len(jpgs) - 1, start + span)
    indices = evenly_spaced_indices(start, end, num_views)
    return [jpgs[index] for index in indices]


def farthest_point_views(jpgs, centers, num_views, start_index):
    chosen = [int(start_index)]
    while len(chosen) < num_views:
        best_index = None
        best_distance = -1.0
        for index in range(len(jpgs)):
            if index in chosen:
                continue
            distance = float(
                np.linalg.norm(centers[index][None, :] - centers[chosen], axis=1).min()
            )
            if distance > best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            break
        chosen.append(best_index)
    return [jpgs[index] for index in sorted(chosen)]


def select_diversity_variant(jpgs, centers, num_views, variant_id, used_signatures):
    if len(jpgs) < num_views:
        raise RuntimeError(f"Only {len(jpgs)} frames are available for {num_views} views")
    base_fraction = 0.15 + 0.55 * variant_id / max(1, VARIANTS_PER_COMBINATION - 1)
    base_start = int(round(base_fraction * (len(jpgs) - 1)))
    stride = max(1, len(jpgs) // max(3, VARIANTS_PER_COMBINATION + 1))
    for attempt in range(len(jpgs)):
        start_index = (base_start + attempt * stride) % len(jpgs)
        selected = farthest_point_views(jpgs, centers, num_views, start_index)
        signature = tuple(path.name for path in selected)
        if signature not in used_signatures:
            return selected
    raise RuntimeError("Could not construct a unique diversity-aware view group")


def build_coverage_manifest(scene_dirs, splits):
    rows = []
    for scene_dir in scene_dirs:
        jpgs = sorted(scene_dir.glob("*.jpg"))[:FRAME_POOL_LIMIT]
        if len(jpgs) < max(VIEW_COUNTS):
            raise RuntimeError(f"Scene {scene_dir.name} has only {len(jpgs)} usable frames")
        centers = np.stack([camera_center(path) for path in jpgs], axis=0)
        used_by_combination = {}
        for num_views in VIEW_COUNTS:
            for policy in VIEW_POLICIES:
                combination = (num_views, policy)
                used_by_combination[combination] = set()
                for variant_id in range(VARIANTS_PER_COMBINATION):
                    if policy == "hybrid":
                        selected = select_hybrid_variant(jpgs, num_views, variant_id)
                    else:
                        selected = select_diversity_variant(
                            jpgs,
                            centers,
                            num_views,
                            variant_id,
                            used_by_combination[combination],
                        )
                    selected = unique_paths(selected)
                    if len(selected) != num_views:
                        raise RuntimeError(
                            f"{scene_dir.name} {combination} variant {variant_id} "
                            f"contains only {len(selected)} unique frames"
                        )
                    signature = tuple(path.name for path in selected)
                    if signature in used_by_combination[combination]:
                        raise RuntimeError(
                            f"Duplicate sparse-view group for {scene_dir.name} {combination}"
                        )
                    used_by_combination[combination].add(signature)
                    group_key = (
                        f"{scene_dir.name}_{num_views}_{policy}_v{variant_id:02d}"
                    )
                    rows.append(
                        {
                            "run": RUN_NAME,
                            "split": splits[scene_dir.name],
                            "scene": scene_dir.name,
                            "num_views": num_views,
                            "view_policy": policy,
                            "variant_id": variant_id,
                            "group_key": group_key,
                            "selected_images": "|".join(path.name for path in selected),
                            "frame_pool_size": len(jpgs),
                            "group_classes": "coverage_stress_sparse_group",
                            **r27.pairwise_baseline_stats(selected),
                        }
                    )
    return rows


def choose_manifest_views(scene_lookup, group_row):
    scene_dir = scene_lookup.get(group_row["scene"])
    if scene_dir is None:
        raise FileNotFoundError(f"Scene {group_row['scene']} not found")
    names = [name for name in group_row["selected_images"].split("|") if name]
    selected = [scene_dir / name for name in names]
    missing = [str(path) for path in selected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing selected frames: {missing}")
    if len(selected) != int(group_row["num_views"]):
        raise RuntimeError(
            f"Manifest group {group_row['group_key']} has {len(selected)} frames"
        )
    return selected


def evaluate_group(record):
    group = record["group_row"]
    conf_mask, conf_percent, conf_threshold = r30.confidence_mask(
        record["conf"], int(group["num_views"])
    )
    corrected_points, correction_mask = r30.apply_policy(record, FIXED_POLICY)
    rows = [
        r30.score_points(
            record["points"],
            record,
            "all_candidates",
            "baseline",
            {"selected_ratio": 1.0},
        ),
        r30.score_points(
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
        r30.score_points(
            corrected_points,
            record,
            "rgbd_source_depth_fixed_run30",
            "resource_expanded_depth",
            {
                "selected_ratio": 1.0,
                "correction_ratio": float(correction_mask.mean()),
                "policy": FIXED_POLICY["name"],
                "mode": FIXED_POLICY["mode"],
                "alpha": FIXED_POLICY["alpha"],
                "residual_threshold_m": FIXED_POLICY["residual_threshold_m"],
            },
        ),
    ]
    for row in rows:
        row["variant_id"] = int(group["variant_id"])
        row["selected_images"] = group["selected_images"]
        print("Run 31 metric row:", row)
    return rows


def challenging_group_keys(metric_rows, split):
    split_rows = [row for row in metric_rows if row["split"] == split]
    diagnostics = {
        row["group_key"]: (
            float(row["occlusion_proxy_ratio"]),
            float(row["ambiguity_proxy_ratio"]),
        )
        for row in split_rows
    }
    count = max(1, int(math.ceil(len(diagnostics) / 3.0)))
    occlusion = {
        item[0]
        for item in sorted(diagnostics.items(), key=lambda item: (-item[1][0], item[0]))[
            :count
        ]
    }
    ambiguity = {
        item[0]
        for item in sorted(diagnostics.items(), key=lambda item: (-item[1][1], item[0]))[
            :count
        ]
    }
    return {
        "overall": set(diagnostics),
        "occlusion_challenging": occlusion,
        "ambiguity_challenging": ambiguity,
    }


def paired_delta_rows(metric_rows):
    rows_by_group = {}
    for row in metric_rows:
        rows_by_group.setdefault(row["group_key"], {})[row["method"]] = row
    output = []
    for group_key, methods in sorted(rows_by_group.items()):
        baseline = methods["all_candidates"]
        selected = methods["rgbd_source_depth_fixed_run30"]
        output.append(
            {
                "run": RUN_NAME,
                "split": selected["split"],
                "scene": selected["scene"],
                "num_views": selected["num_views"],
                "view_policy": selected["view_policy"],
                "variant_id": selected["variant_id"],
                "group_key": group_key,
                "baseline_fscore": baseline["fscore"],
                "rgbd_fscore": selected["fscore"],
                "paired_delta_fscore": selected["fscore"] - baseline["fscore"],
                "correction_ratio": selected["correction_ratio"],
                "occlusion_proxy_ratio": selected["occlusion_proxy_ratio"],
                "ambiguity_proxy_ratio": selected["ambiguity_proxy_ratio"],
            }
        )
    return output


def cluster_bootstrap(deltas, seed):
    by_scene = {}
    for row in deltas:
        by_scene.setdefault(row["scene"], []).append(float(row["paired_delta_fscore"]))
    scene_means = np.array(
        [np.mean(by_scene[scene]) for scene in sorted(by_scene)], dtype=np.float64
    )
    if not len(scene_means):
        raise RuntimeError("No scene deltas available for bootstrap")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(scene_means), size=(BOOTSTRAP_SAMPLES, len(scene_means)))
    samples = scene_means[indices].mean(axis=1)
    return {
        "num_scenes": int(len(scene_means)),
        "scene_cluster_mean_delta": float(scene_means.mean()),
        "scene_cluster_std_delta": float(scene_means.std(ddof=0)),
        "bootstrap_ci95_low": float(np.quantile(samples, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(samples, 0.975)),
    }


def stability_summary(delta_rows, metric_rows):
    output = []
    for split in sorted({row["split"] for row in delta_rows}):
        subset_keys = challenging_group_keys(metric_rows, split)
        for subset_name, keys in subset_keys.items():
            rows = [
                row
                for row in delta_rows
                if row["split"] == split and row["group_key"] in keys
            ]
            values = np.array(
                [float(row["paired_delta_fscore"]) for row in rows], dtype=np.float64
            )
            output.append(
                {
                    "run": RUN_NAME,
                    "split": split,
                    "limit_subset": subset_name,
                    "num_groups": len(rows),
                    "mean_delta_fscore": float(values.mean()),
                    "median_delta_fscore": float(np.median(values)),
                    "std_delta_fscore": float(values.std(ddof=0)),
                    "positive_group_ratio": float((values > 0.0).mean()),
                    **cluster_bootstrap(
                        rows, r27.stable_seed(f"run31-{split}-{subset_name}")
                    ),
                }
            )
    return output


def view_count_summary(delta_rows):
    output = []
    for split, num_views in sorted(
        {(row["split"], int(row["num_views"])) for row in delta_rows}
    ):
        rows = [
            row
            for row in delta_rows
            if row["split"] == split and int(row["num_views"]) == num_views
        ]
        values = np.array(
            [float(row["paired_delta_fscore"]) for row in rows], dtype=np.float64
        )
        output.append(
            {
                "run": RUN_NAME,
                "split": split,
                "num_views": num_views,
                "num_groups": len(rows),
                "mean_delta_fscore": float(values.mean()),
                "median_delta_fscore": float(np.median(values)),
                "positive_group_ratio": float((values > 0.0).mean()),
                **cluster_bootstrap(
                    rows, r27.stable_seed(f"run31-{split}-{num_views}-views")
                ),
            }
        )
    return output


def coverage_summary(groups):
    output = []
    for scene in sorted({row["scene"] for row in groups}):
        rows = [row for row in groups if row["scene"] == scene]
        output.append(
            {
                "run": RUN_NAME,
                "split": rows[0]["split"],
                "scene": scene,
                "num_groups": len(rows),
                "num_3_view_groups": sum(int(row["num_views"]) == 3 for row in rows),
                "num_4_view_groups": sum(int(row["num_views"]) == 4 for row in rows),
                "num_5_view_groups": sum(int(row["num_views"]) == 5 for row in rows),
                "num_hybrid_groups": sum(row["view_policy"] == "hybrid" for row in rows),
                "num_diversity_groups": sum(
                    row["view_policy"] == "diversity_aware" for row in rows
                ),
                "unique_frame_sets": len(
                    {row["selected_images"] for row in rows}
                ),
            }
        )
    return output


def lookup_stability(rows, split, subset):
    return next(
        row
        for row in rows
        if row["split"] == split and row["limit_subset"] == subset
    )


def gate_decision(groups, coverage_rows, stability_rows, view_rows):
    expected_groups_per_scene = (
        len(VIEW_COUNTS) * len(VIEW_POLICIES) * VARIANTS_PER_COMBINATION
    )
    coverage_complete = (
        len({row["scene"] for row in groups}) == MAX_SCENES
        and len(groups) == MAX_SCENES * expected_groups_per_scene
        and all(row["num_groups"] == expected_groups_per_scene for row in coverage_rows)
        and all(row["unique_frame_sets"] == expected_groups_per_scene for row in coverage_rows)
    )
    val_overall = lookup_stability(stability_rows, "val", "overall")
    test_overall = lookup_stability(stability_rows, "test", "overall")
    val_occ = lookup_stability(stability_rows, "val", "occlusion_challenging")
    val_amb = lookup_stability(stability_rows, "val", "ambiguity_challenging")
    test_occ = lookup_stability(stability_rows, "test", "occlusion_challenging")
    test_amb = lookup_stability(stability_rows, "test", "ambiguity_challenging")
    view_non_regression = all(
        row["mean_delta_fscore"] >= 0.0
        for row in view_rows
        if row["split"] in {"val", "test"}
    )
    checks = {
        "coverage_complete": coverage_complete,
        "policy_frozen": True,
        "validation_overall_margin_pass": val_overall["mean_delta_fscore"]
        >= GATE_MARGIN_F1,
        "test_overall_margin_pass": test_overall["mean_delta_fscore"]
        >= GATE_MARGIN_F1,
        "validation_ci95_low_positive": val_overall["bootstrap_ci95_low"] > 0.0,
        "test_ci95_low_positive": test_overall["bootstrap_ci95_low"] > 0.0,
        "validation_hard_subset_non_regression": min(
            val_occ["mean_delta_fscore"], val_amb["mean_delta_fscore"]
        )
        >= 0.0,
        "test_hard_subset_non_regression": min(
            test_occ["mean_delta_fscore"], test_amb["mean_delta_fscore"]
        )
        >= 0.0,
        "view_count_non_regression": view_non_regression,
    }
    return [
        {
            "run": RUN_NAME,
            "purpose": "coverage_stability_only_no_method_selection",
            "fixed_method": "rgbd_source_depth_fixed_run30",
            "fixed_policy": FIXED_POLICY["name"],
            "num_scenes": len({row["scene"] for row in groups}),
            "num_groups": len(groups),
            "groups_per_scene": expected_groups_per_scene,
            "validation_mean_delta": val_overall["mean_delta_fscore"],
            "validation_ci95_low": val_overall["bootstrap_ci95_low"],
            "validation_ci95_high": val_overall["bootstrap_ci95_high"],
            "test_mean_delta": test_overall["mean_delta_fscore"],
            "test_ci95_low": test_overall["bootstrap_ci95_low"],
            "test_ci95_high": test_overall["bootstrap_ci95_high"],
            **{name: int(value) for name, value in checks.items()},
            "pass_coverage_stability": int(all(checks.values())),
            "gate_margin_f1": GATE_MARGIN_F1,
        }
    ]


def prune_group_artifact(group, group_out_dir):
    keep = (
        int(group["num_views"]) == 4
        and group["view_policy"] == "hybrid"
        and int(group["variant_id"]) == 0
    )
    scene_glb = group_out_dir / "scene.glb"
    if not keep and scene_glb.exists():
        scene_glb.unlink()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if VARIANTS_PER_COMBINATION < 2:
        raise ValueError("Run 31 requires at least two variants per view/policy combination")
    r30.RUN_NAME = RUN_NAME
    r28.MAX_CANDIDATES_PER_GROUP = MAX_CANDIDATES_PER_GROUP
    r27.choose_group_views = choose_manifest_views
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    r27.validate_static_configuration()
    base.require_t4x2()
    root = base.clone_repo()
    base.install_deps(root)
    posed_root = base.find_posed_images_root()
    scene_dirs = r27.discover_scene_dirs(posed_root)[:MAX_SCENES]
    scene_lookup = {path.name: path for path in scene_dirs}
    splits = r27.scene_splits(scene_dirs)
    manifest = build_coverage_manifest(scene_dirs, splits)
    groups = r27.balanced_group_subset(manifest, MAX_GROUPS)
    expected_full_groups = (
        len(scene_dirs)
        * len(VIEW_COUNTS)
        * len(VIEW_POLICIES)
        * VARIANTS_PER_COMBINATION
    )
    if len(groups) != expected_full_groups:
        raise RuntimeError(
            f"Coverage stress test requires all {expected_full_groups} groups, got {len(groups)}"
        )
    print("Run 31 splits:", splits)
    print(
        "Run 31 coverage:",
        {
            "scenes": len(scene_dirs),
            "groups": len(groups),
            "groups_per_scene": len(groups) // len(scene_dirs),
            "views": VIEW_COUNTS,
            "policies": VIEW_POLICIES,
            "variants": VARIANTS_PER_COMBINATION,
            "fixed_policy": FIXED_POLICY,
        },
    )

    checkpoint = base.download_checkpoint(root)
    backbone = base.load_model(root, checkpoint)
    metric_rows = []
    correction_rows = []
    for index, group in enumerate(groups, start=1):
        group_out_dir = out_dir / "groups" / group["group_key"]
        record = r28.run_group(backbone, root, scene_lookup, group, group_out_dir)
        correction_rows.append(r30.correction_summary(record, "coverage_eval"))
        metric_rows.extend(evaluate_group(record))
        prune_group_artifact(group, group_out_dir)
        if index % 25 == 0 or index == len(groups):
            print(f"Run 31 completed groups: {index}/{len(groups)}")

    summary_rows = r30.summarize(metric_rows)
    limit_rows = r30.limit_summary(metric_rows)
    delta_rows = paired_delta_rows(metric_rows)
    stability_rows = stability_summary(delta_rows, metric_rows)
    view_rows = view_count_summary(delta_rows)
    coverage_rows = coverage_summary(groups)
    gate_rows = gate_decision(groups, coverage_rows, stability_rows, view_rows)

    r27.write_csv_union(out_dir / "group_manifest.csv", groups)
    r27.write_csv_union(out_dir / "coverage_summary.csv", coverage_rows)
    r27.write_csv_union(out_dir / "correction_label_summary.csv", correction_rows)
    r27.write_csv_union(out_dir / "metrics.csv", metric_rows)
    r27.write_csv_union(out_dir / "summary.csv", summary_rows)
    r27.write_csv_union(out_dir / "limit_summary.csv", limit_rows)
    r27.write_csv_union(out_dir / "paired_group_deltas.csv", delta_rows)
    r27.write_csv_union(out_dir / "stability_summary.csv", stability_rows)
    r27.write_csv_union(out_dir / "view_count_stability.csv", view_rows)
    r27.write_csv_union(out_dir / "gate_decision.csv", gate_rows)
    config = {
        "run": RUN_NAME,
        "purpose": "Coverage stress test for the frozen Run 30 RGB-D method; no new method or policy selection.",
        "num_scenes": len(scene_dirs),
        "scene_splits": splits,
        "num_groups": len(groups),
        "groups_per_scene": len(groups) // len(scene_dirs),
        "view_counts": VIEW_COUNTS,
        "view_policies": VIEW_POLICIES,
        "variants_per_combination": VARIANTS_PER_COMBINATION,
        "frame_pool_limit": FRAME_POOL_LIMIT,
        "full_dense_frames_used": False,
        "fixed_policy": FIXED_POLICY,
        "policy_selection_performed": False,
        "max_candidates_per_group": MAX_CANDIDATES_PER_GROUP,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "gate_margin_f1": GATE_MARGIN_F1,
        "run30_reference": RUN30_REFERENCE,
        "runtime_seconds": time.time() - started,
        "inference_contract": "Sparse posed RGB-D views with source depth maps and known camera poses/intrinsics at inference.",
        "claim_contract": "Run 31 only tests coverage stability of the frozen Run 30 method. It does not introduce or select a new reconstruction method.",
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print("Run 31 config:", config)
    print("Run 31 stability summary:")
    for row in stability_rows:
        print(row)
    print("Run 31 gate decision:")
    for row in gate_rows:
        print(row)


if __name__ == "__main__":
    main()
