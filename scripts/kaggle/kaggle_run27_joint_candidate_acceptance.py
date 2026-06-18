import csv
import json
import math
import os
import random
import sys
import time
import urllib.request
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree


RAW_BASE = "https://raw.githubusercontent.com/NguyenNgocMinh30012005/Computer_vision_2025.2/main/scripts/kaggle/kaggle_run11_final_validation_3seeds.py"
RAW_RSDH_UTILS = "https://raw.githubusercontent.com/NguyenNgocMinh30012005/Computer_vision_2025.2/main/scripts/kaggle/kaggle_run26_rsdh_v2_diagnostic_gate.py"


def ensure_helper_module(module_name, raw_url):
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        helper_path = Path(f"{module_name}.py")
        print(f"Downloading helper module {module_name} from GitHub raw")
        urllib.request.urlretrieve(raw_url, helper_path)
        sys.path.insert(0, str(Path.cwd()))
        return __import__(module_name)


base = ensure_helper_module("kaggle_run11_final_validation_3seeds", RAW_BASE)
rsdh_utils = ensure_helper_module("kaggle_run26_rsdh_v2_diagnostic_gate", RAW_RSDH_UTILS)


RUN_NAME = "run_27_reconstruction_aware_joint_acceptance"
RUN19_SEED = 1919
SEED = 2727
MAX_SCENES = int(os.environ.get("RUN27_MAX_SCENES", "30"))
MAX_TRAIN_GROUPS = int(os.environ.get("RUN27_MAX_TRAIN_GROUPS", "48"))
MAX_EVAL_GROUPS = int(os.environ.get("RUN27_MAX_EVAL_GROUPS", "36"))
MAX_CANDIDATES_PER_GROUP = int(os.environ.get("RUN27_MAX_CANDIDATES_PER_GROUP", "3500"))
MIN_SELECTED_POINTS = int(os.environ.get("RUN27_MIN_SELECTED_POINTS", "100"))
GATE_MARGIN_F1 = float(os.environ.get("RUN27_GATE_MARGIN_F1", "0.005"))
EPOCHS = int(os.environ.get("RUN27_EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("RUN27_BATCH_SIZE", "65536"))
LR = float(os.environ.get("RUN27_LR", "0.001"))
LABEL_POS_M = float(os.environ.get("RUN27_LABEL_POS_M", str(base.F_SCORE_THRESHOLD_M)))
HARD_NEG_M = float(os.environ.get("RUN27_HARD_NEG_M", "0.15"))
VIEW_COUNTS = [3, 4, 5]
POLICIES = ["hybrid", "diversity_aware"]
MODEL_SEEDS = [2727, 2728, 2729]
TRAIN_RATIOS = [0.95, 0.98, 0.995]
RANK_RATIOS = [0.90, 0.95, 0.97, 0.98, 0.9875, 0.995, 1.0]
RESIDUAL_SCALE = float(os.environ.get("RUN27_RESIDUAL_SCALE", "2.0"))
SOFT_TOPK_TEMPERATURE = float(os.environ.get("RUN27_SOFT_TOPK_TEMPERATURE", "0.20"))
OCCLUSION_PROXY_THRESHOLD = float(os.environ.get("RUN27_OCCLUSION_PROXY_THRESHOLD", "0.05"))
AMBIGUITY_PROXY_THRESHOLD = float(os.environ.get("RUN27_AMBIGUITY_PROXY_THRESHOLD", "0.10"))
LOSS_WEIGHTS = {
    "bce": 0.20,
    "reconstruction_f1": 1.00,
    "ranking": 0.35,
    "ratio": 0.10,
    "residual": 0.01,
}
BASELINE_METHODS = {"confidence_fixed_final", "all_candidates"}

BASE_FEATURE_NAMES = [
    "log_conf",
    "conf_z",
    "conf_rank",
    "src_x_norm",
    "src_y_norm",
    "src_view_norm",
    "num_views_norm",
    "policy_hybrid",
    "policy_diversity",
    "point_x_scaled",
    "point_y_scaled",
    "point_z_scaled",
    "point_radius_scaled",
    "abs_z_scaled",
    "support_nearest_m",
    "support_mean3_m",
    "support_frac_005",
    "support_frac_010",
    "support_frac_020",
    "same_view_conf_rank",
]

PAIR_SIGNAL_NAMES = [
    "pixel_distance",
    "abs_rgb_mean_diff_r",
    "abs_rgb_mean_diff_g",
    "abs_rgb_mean_diff_b",
    "abs_gray_mean_diff",
    "gray_patch_corr",
    "gray_patch_l1",
    "gray_patch_l2",
    "abs_grad_mean_diff",
    "grad_patch_corr",
    "grad_patch_l1",
    "grad_patch_l2",
]
PAIR_AGGREGATIONS = ["mean", "std", "min", "max"]
PHOTO_FEATURE_NAMES = [
    f"photo_{signal}_{aggregation}"
    for signal in PAIR_SIGNAL_NAMES
    for aggregation in PAIR_AGGREGATIONS
]
PHOTO_FEATURE_NAMES += [
    "photo_target_count_norm",
    "photo_has_projection",
    "photo_gray_corr_positive_frac",
    "photo_grad_corr_positive_frac",
    "photo_low_gray_l1_frac",
]
FEATURE_NAMES = BASE_FEATURE_NAMES + PHOTO_FEATURE_NAMES
CONF_RANK_INDEX = FEATURE_NAMES.index("conf_rank")
SUPPORT_FRAC_010_INDEX = FEATURE_NAMES.index("support_frac_010")
PHOTO_TARGET_COUNT_INDEX = FEATURE_NAMES.index("photo_target_count_norm")


class JointCandidateAcceptanceHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(160, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        return RESIDUAL_SCALE * torch.tanh(self.net(x).squeeze(-1))


def write_csv_union(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", "nan", "NaN", None):
        return default
    try:
        x = float(value)
    except ValueError:
        return default
    return x if math.isfinite(x) else default


def as_int(row, key, default=0):
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def stable_seed(text):
    return SEED + (zlib.crc32(str(text).encode("utf-8")) % 100000)


def validate_static_configuration():
    if len(FEATURE_NAMES) != len(set(FEATURE_NAMES)):
        raise RuntimeError("Run 27 feature names are not unique")
    missing_pair_signals = [
        name for name in PAIR_SIGNAL_NAMES if name not in rsdh_utils.FEATURE_NAMES
    ]
    if missing_pair_signals:
        raise RuntimeError(f"Missing pair feature signals: {missing_pair_signals}")
    if not all(0.0 < ratio <= 1.0 for ratio in TRAIN_RATIOS + RANK_RATIOS):
        raise RuntimeError("All train/evaluation keep ratios must be in (0, 1]")
    if sorted(MODEL_SEEDS) != MODEL_SEEDS or len(set(MODEL_SEEDS)) != len(MODEL_SEEDS):
        raise RuntimeError("Model seeds must be unique and sorted")
    print(
        "Run 27 static preflight:",
        {
            "feature_dim": len(FEATURE_NAMES),
            "base_feature_dim": len(BASE_FEATURE_NAMES),
            "photo_feature_dim": len(PHOTO_FEATURE_NAMES),
            "model_seeds": MODEL_SEEDS,
            "train_ratios": TRAIN_RATIOS,
        },
    )


def discover_scene_dirs(posed_root):
    scenes = sorted([p for p in posed_root.glob("scene*") if p.is_dir() and list(p.glob("*.jpg"))])
    if not scenes:
        raise FileNotFoundError(f"No scene directories with JPG frames under {posed_root}")
    return scenes[:MAX_SCENES]


def scene_splits(scene_dirs):
    n = len(scene_dirs)
    if n <= 1:
        return {scene_dirs[0].name: "train"} if scene_dirs else {}
    if n == 2:
        return {scene_dirs[0].name: "train", scene_dirs[1].name: "test"}
    if n < 5:
        return {
            scene.name: ("train" if i == 0 else "val" if i == 1 else "test")
            for i, scene in enumerate(scene_dirs)
        }
    train_cut = max(1, int(round(0.60 * n)))
    val_cut = max(train_cut + 1, int(round(0.80 * n)))
    val_cut = min(val_cut, n - 1)
    return {
        scene.name: "train" if i < train_cut else "val" if i < val_cut else "test"
        for i, scene in enumerate(scene_dirs)
    }


def pairwise_baseline_stats(view_files):
    centers = []
    for path in view_files:
        pose = base.parse_pose(str(path).replace(".jpg", ".txt"))
        centers.append(pose[:3, 3].astype(np.float32))
    distances = [
        float(np.linalg.norm(centers[i] - centers[j]))
        for i in range(len(centers))
        for j in range(i + 1, len(centers))
    ]
    return {
        "mean_baseline_m": float(np.mean(distances)) if distances else 0.0,
        "max_baseline_m": float(np.max(distances)) if distances else 0.0,
    }


def select_unique_views(scene_dir, num_views, policy):
    selected = base.choose_views(
        scene_dir,
        num_views,
        policy,
        seed=RUN19_SEED + num_views,
    )
    unique = []
    seen = set()
    for path in selected:
        if path.name not in seen:
            unique.append(path)
            seen.add(path.name)
    if len(unique) < num_views:
        candidates = [path for path in sorted(scene_dir.glob("*.jpg")) if path.name not in seen]
        if candidates:
            fill_count = min(num_views - len(unique), len(candidates))
            fill_indices = np.linspace(0, len(candidates) - 1, fill_count, dtype=int)
            unique.extend(candidates[index] for index in fill_indices)
    if len(unique) < num_views:
        raise RuntimeError(
            f"Scene {scene_dir.name} has only {len(unique)} unique views for requested num_views={num_views}"
        )
    return unique[:num_views]


def build_group_manifest(scene_dirs, splits):
    rows = []
    for scene_dir in scene_dirs:
        for num_views in VIEW_COUNTS:
            for policy in POLICIES:
                group_key = f"{scene_dir.name}_{num_views}_{policy}"
                view_files = select_unique_views(scene_dir, num_views, policy)
                rows.append(
                    {
                        "run": RUN_NAME,
                        "split": splits[scene_dir.name],
                        "scene": scene_dir.name,
                        "num_views": num_views,
                        "view_policy": policy,
                        "group_key": group_key,
                        "selected_images": "|".join(path.name for path in view_files),
                        "group_classes": "self_contained_scene_group",
                        **pairwise_baseline_stats(view_files),
                    }
                )
    return rows


def balanced_group_subset(rows, limit):
    if len(rows) <= limit:
        return sorted(rows, key=lambda r: (r["scene"], as_int(r, "num_views"), r["view_policy"]))
    by_scene = {}
    for row in rows:
        by_scene.setdefault(row["scene"], []).append(row)
    for scene_rows in by_scene.values():
        scene_rows.sort(
            key=lambda r: (
                -as_float(r, "mean_baseline_m"),
                as_int(r, "num_views"),
                r["view_policy"],
            )
        )
    selected = []
    round_idx = 0
    scene_names = sorted(by_scene)
    while len(selected) < limit:
        added = False
        for scene_name in scene_names:
            scene_rows = by_scene[scene_name]
            if round_idx < len(scene_rows):
                selected.append(scene_rows[round_idx])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        round_idx += 1
    return selected


def split_internal_train_groups(train_groups):
    scenes = sorted({row["scene"] for row in train_groups})
    rng = np.random.default_rng(SEED)
    shuffled = [scenes[i] for i in rng.permutation(len(scenes))]
    num_val_scenes = max(2, int(round(0.25 * len(scenes))))
    internal_val_scenes = set(shuffled[:num_val_scenes])
    fit = [row for row in train_groups if row["scene"] not in internal_val_scenes]
    val = [row for row in train_groups if row["scene"] in internal_val_scenes]
    return fit, val, sorted(internal_val_scenes)


def output_to_candidates(output):
    pts = [output["pred1"]["pts3d"][0].detach().cpu()]
    pts += [x["pts3d_in_other_view"][0].detach().cpu() for x in output["pred2s"]]
    conf = [output["pred1"]["conf"][0].detach().cpu()]
    conf += [x["conf"][0].detach().cpu() for x in output["pred2s"]]

    pts = torch.stack(pts, dim=0).numpy().astype(np.float32)
    conf = torch.stack(conf, dim=0).numpy().astype(np.float32)
    n_views, h, w, _ = pts.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    xx = np.broadcast_to(xx[None, :, :], (n_views, h, w))
    yy = np.broadcast_to(yy[None, :, :], (n_views, h, w))
    view_idx = np.broadcast_to(np.arange(n_views, dtype=np.int32)[:, None, None], (n_views, h, w))

    flat_pts = pts.reshape(-1, 3)
    flat_conf = conf.reshape(-1)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    flat_view = view_idx.reshape(-1)
    finite = np.isfinite(flat_pts).all(axis=1) & np.isfinite(flat_conf)
    return flat_pts[finite], flat_conf[finite], flat_x[finite], flat_y[finite], flat_view[finite], h, w


def subsample_candidates(points, conf, xs, ys, view_ids, max_candidates, seed):
    if len(points) <= max_candidates or max_candidates <= 0:
        return points, conf, xs, ys, view_ids
    rng = np.random.default_rng(seed)
    conf_rank = rank_percentile(conf)
    high = np.where(conf_rank >= 0.50)[0]
    low = np.where(conf_rank < 0.50)[0]
    n_high = min(len(high), int(max_candidates * 0.80))
    n_low = min(len(low), max_candidates - n_high)
    selected = []
    if n_high:
        selected.append(rng.choice(high, n_high, replace=False))
    if n_low:
        selected.append(rng.choice(low, n_low, replace=False))
    idx = np.concatenate(selected) if selected else rng.choice(len(points), max_candidates, replace=False)
    if len(idx) < max_candidates:
        remaining = np.setdiff1d(np.arange(len(points)), idx, assume_unique=False)
        fill = rng.choice(remaining, min(len(remaining), max_candidates - len(idx)), replace=False)
        idx = np.concatenate([idx, fill])
    return points[idx], conf[idx], xs[idx], ys[idx], view_ids[idx]


def rank_percentile(values):
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def support_features(points, view_ids):
    n = len(points)
    nearest = np.full(n, 2.0, dtype=np.float32)
    mean3 = np.full(n, 2.0, dtype=np.float32)
    frac005 = np.zeros(n, dtype=np.float32)
    frac010 = np.zeros(n, dtype=np.float32)
    frac020 = np.zeros(n, dtype=np.float32)
    unique_views = sorted(set(int(v) for v in view_ids))
    for view in unique_views:
        idx = np.where(view_ids == view)[0]
        other = np.where(view_ids != view)[0]
        if len(idx) == 0 or len(other) == 0:
            continue
        tree = cKDTree(points[other])
        k = min(8, len(other))
        dists, _ = tree.query(points[idx], k=k, workers=-1)
        if k == 1:
            dists = dists[:, None]
        clipped = np.clip(dists.astype(np.float32), 0.0, 2.0)
        nearest[idx] = clipped[:, 0]
        mean3[idx] = clipped[:, : min(3, clipped.shape[1])].mean(axis=1)
        denom = float(clipped.shape[1])
        frac005[idx] = (clipped <= 0.05).sum(axis=1) / denom
        frac010[idx] = (clipped <= 0.10).sum(axis=1) / denom
        frac020[idx] = (clipped <= 0.20).sum(axis=1) / denom
    return nearest, mean3, frac005, frac010, frac020


def build_features(points, conf, xs, ys, view_ids, image_h, image_w, group_row):
    n = len(points)
    conf = conf.astype(np.float32)
    log_conf = np.log1p(np.maximum(conf, 0.0))
    q25, q75 = np.quantile(conf, [0.25, 0.75]) if n else (0.0, 1.0)
    conf_z = (conf - float(np.median(conf))) / (float(q75 - q25) + 1e-6)
    conf_rank = rank_percentile(conf)
    same_view_confile = np.zeros(n, dtype=np.float32)
    for view in sorted(set(int(v) for v in view_ids)):
        idx = np.where(view_ids == view)[0]
        same_view_confile[idx] = rank_percentile(conf[idx])

    med = np.median(points, axis=0, keepdims=True)
    centered = points - med
    radius = np.linalg.norm(centered, axis=1)
    scale = float(np.quantile(radius, 0.90)) + 1e-6 if n else 1.0
    scaled = np.clip(centered / scale, -4.0, 4.0)
    radius_scaled = np.clip(radius / scale, 0.0, 4.0)
    abs_z_scaled = np.clip(np.abs(centered[:, 2]) / scale, 0.0, 4.0)

    nearest, mean3, frac005, frac010, frac020 = support_features(points, view_ids)
    policy = group_row.get("view_policy", "")
    num_views = max(as_float(group_row, "num_views"), 1.0)
    features = np.column_stack(
        [
            log_conf,
            np.clip(conf_z, -6.0, 6.0),
            conf_rank,
            xs / max(image_w - 1, 1),
            ys / max(image_h - 1, 1),
            view_ids.astype(np.float32) / max(num_views - 1.0, 1.0),
            np.full(n, num_views / 5.0, dtype=np.float32),
            np.full(n, 1.0 if policy == "hybrid" else 0.0, dtype=np.float32),
            np.full(n, 1.0 if policy == "diversity_aware" else 0.0, dtype=np.float32),
            scaled[:, 0],
            scaled[:, 1],
            scaled[:, 2],
            radius_scaled,
            abs_z_scaled,
            nearest,
            mean3,
            frac005,
            frac010,
            frac020,
            same_view_confile,
        ]
    )
    return features.astype(np.float32)


def aggregate_image_pair_features(
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
):
    pair_features, candidate_indices, target_counts = rsdh_utils.build_pair_features(
        points, xs, ys, view_ids, conf, view_files, group_row, image_h, image_w
    )
    n = len(points)
    num_views = max(as_float(group_row, "num_views"), 1.0)
    selected_indices = [rsdh_utils.FEATURE_NAMES.index(name) for name in PAIR_SIGNAL_NAMES]
    selected = pair_features[:, selected_indices] if len(pair_features) else np.empty((0, len(selected_indices)), dtype=np.float32)
    count = np.maximum(target_counts.astype(np.float32), 1.0)
    aggregates = []
    for column in range(len(selected_indices)):
        values = selected[:, column] if len(selected) else np.empty((0,), dtype=np.float32)
        sums = np.zeros(n, dtype=np.float32)
        sums_sq = np.zeros(n, dtype=np.float32)
        mins = np.full(n, np.inf, dtype=np.float32)
        maxs = np.full(n, -np.inf, dtype=np.float32)
        if len(values):
            np.add.at(sums, candidate_indices, values)
            np.add.at(sums_sq, candidate_indices, values * values)
            np.minimum.at(mins, candidate_indices, values)
            np.maximum.at(maxs, candidate_indices, values)
        means = sums / count
        variances = np.maximum(sums_sq / count - means * means, 0.0)
        mins[~np.isfinite(mins)] = 0.0
        maxs[~np.isfinite(maxs)] = 0.0
        aggregates.extend([means, np.sqrt(variances), mins, maxs])

    def candidate_fraction(signal_name, predicate):
        out = np.zeros(n, dtype=np.float32)
        if len(pair_features):
            values = pair_features[:, rsdh_utils.FEATURE_NAMES.index(signal_name)]
            np.add.at(out, candidate_indices, predicate(values).astype(np.float32))
        return out / count

    photo_features = np.column_stack(
        [
            *aggregates,
            np.clip(target_counts / max(num_views - 1.0, 1.0), 0.0, 1.0),
            (target_counts > 0).astype(np.float32),
            candidate_fraction("gray_patch_corr", lambda value: value > 0.0),
            candidate_fraction("grad_patch_corr", lambda value: value > 0.0),
            candidate_fraction("gray_patch_l1", lambda value: value < 0.15),
        ]
    ).astype(np.float32)
    stats = {
        "num_pair_features": int(len(pair_features)),
        "mean_photo_gray_corr": float(
            photo_features[:, PHOTO_FEATURE_NAMES.index("photo_gray_patch_corr_mean")].mean()
        )
        if n
        else 0.0,
        "mean_photo_gray_l1": float(
            photo_features[:, PHOTO_FEATURE_NAMES.index("photo_gray_patch_l1_mean")].mean()
        )
        if n
        else 0.0,
        "mean_projected_targets": float(target_counts.mean()) if len(target_counts) else 0.0,
    }
    return np.column_stack([base_features, photo_features]).astype(np.float32), stats


def candidate_labels(points, gt):
    aligned = base.center_scale_align(points.astype(np.float32), gt)
    dists, _ = cKDTree(gt).query(aligned, k=1, workers=-1)
    dists = dists.astype(np.float32)
    labels = (dists <= LABEL_POS_M).astype(np.float32)
    hard_negative = (dists >= HARD_NEG_M).astype(np.float32)
    gt_dists, nearest_candidate = cKDTree(aligned).query(gt, k=1, workers=-1)
    covered = gt_dists < LABEL_POS_M
    coverage_mass = np.bincount(
        nearest_candidate[covered],
        minlength=len(points),
    ).astype(np.float32)
    return labels, dists, hard_negative, coverage_mass, int(len(gt))


def choose_group_views(scene_lookup, group_row):
    scene_dir = scene_lookup.get(group_row["scene"])
    if scene_dir is None:
        raise FileNotFoundError(f"Scene {group_row['scene']} not found in posed_images")
    num_views = as_int(group_row, "num_views")
    policy = group_row.get("view_policy")
    return select_unique_views(scene_dir, num_views, policy)


def run_group(backbone, root, scene_lookup, group_row, out_dir):
    view_files = choose_group_views(scene_lookup, group_row)
    print("Run 27 group views:", {"group": group_row.get("group_key"), "views": [p.name for p in view_files]})
    output, _glb, runtime = base.run_inference(backbone, root, view_files, out_dir)
    gt, _stats = base.build_gt_cloud(view_files)
    points, conf, xs, ys, view_ids, image_h, image_w = output_to_candidates(output)
    points, conf, xs, ys, view_ids = subsample_candidates(
        points,
        conf,
        xs,
        ys,
        view_ids,
        MAX_CANDIDATES_PER_GROUP,
        stable_seed(group_row.get("group_key", "")),
    )
    base_features = build_features(points, conf, xs, ys, view_ids, image_h, image_w, group_row)
    features, photo_stats = aggregate_image_pair_features(
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
    labels, distances, hard_negative, coverage_mass, num_gt_points = candidate_labels(points, gt)
    conf_rank = rank_percentile(conf)
    occluded_positive = (
        (labels > 0.5)
        & (features[:, SUPPORT_FRAC_010_INDEX] < 0.125)
        & (features[:, PHOTO_TARGET_COUNT_INDEX] > 0.0)
    ).astype(np.float32)
    ambiguity_negative = (
        (labels < 0.5)
        & ((hard_negative > 0.5) | (conf_rank >= 0.75))
    ).astype(np.float32)
    del output
    torch.cuda.empty_cache()
    return {
        "points": points,
        "conf": conf,
        "features": features,
        "labels": labels,
        "distances": distances,
        "hard_negative": hard_negative,
        "coverage_mass": coverage_mass,
        "num_gt_points": num_gt_points,
        "occluded_positive": occluded_positive,
        "ambiguity_negative": ambiguity_negative,
        "gt": gt,
        "runtime_seconds": runtime,
        "view_files": view_files,
        "photo_stats": photo_stats,
    }


def label_summary_row(group_row, pack, stage):
    labels = pack["labels"]
    distances = pack["distances"]
    hard = pack["hard_negative"]
    conf = pack["conf"]
    return {
        "run": RUN_NAME,
        "stage": stage,
        "split": group_row.get("split"),
        "scene": group_row.get("scene"),
        "num_views": as_int(group_row, "num_views"),
        "view_policy": group_row.get("view_policy"),
        "group_key": group_row.get("group_key"),
        "group_classes": group_row.get("group_classes"),
        "mean_baseline_m": as_float(group_row, "mean_baseline_m"),
        "max_baseline_m": as_float(group_row, "max_baseline_m"),
        "num_candidates": int(len(labels)),
        "candidate_positive_ratio": float(labels.mean()) if len(labels) else 0.0,
        "hard_negative_ratio": float(hard.mean()) if len(hard) else 0.0,
        "occluded_positive_ratio": float(pack["occluded_positive"].mean()) if len(labels) else 0.0,
        "ambiguity_negative_ratio": float(pack["ambiguity_negative"].mean()) if len(labels) else 0.0,
        "covered_gt_ratio_all_candidates": float(pack["coverage_mass"].sum() / max(pack["num_gt_points"], 1)),
        "mean_distance_to_gt_m": float(distances.mean()) if len(distances) else 0.0,
        "median_distance_to_gt_m": float(np.median(distances)) if len(distances) else 0.0,
        "mean_conf": float(conf.mean()) if len(conf) else 0.0,
        "runtime_seconds": float(pack["runtime_seconds"]),
        **pack.get("photo_stats", {}),
    }


def safe_device():
    if torch.cuda.is_available():
        try:
            major, _minor = torch.cuda.get_device_capability(0)
            if major >= 7:
                return "cuda"
        except Exception as exc:
            print("CUDA compatibility check failed for Run 27 MLP; using CPU:", repr(exc))
    return "cpu"


def candidate_label_metrics(prob, labels, threshold):
    pred = prob >= threshold
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"label_precision": float(precision), "label_recall": float(recall), "label_f1": float(f1)}


def feature_standardization(records):
    features = np.concatenate([record["features"] for record in records], axis=0)
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def score_logits(model, raw_features, mean_t, std_t):
    normalized = (raw_features - mean_t) / std_t
    residual = model(normalized)
    conf_rank = raw_features[:, CONF_RANK_INDEX].clamp(1e-3, 1.0 - 1e-3)
    confidence_logit = torch.log(conf_rank) - torch.log1p(-conf_rank)
    return confidence_logit + residual, residual


def reconstruction_aware_group_loss(model, record, mean_t, std_t):
    device = mean_t.device
    raw = torch.from_numpy(record["features"]).float().to(device)
    labels = torch.from_numpy(record["labels"]).float().to(device)
    coverage = torch.from_numpy(record["coverage_mass"]).float().to(device)
    occluded_positive = torch.from_numpy(record["occluded_positive"]).float().to(device)
    ambiguity_negative = torch.from_numpy(record["ambiguity_negative"]).float().to(device)
    logits, residual = score_logits(model, raw, mean_t, std_t)

    sample_weight = 1.0 + 2.0 * occluded_positive + 2.0 * ambiguity_negative
    bce = F.binary_cross_entropy_with_logits(logits, labels, weight=sample_weight)

    soft_f1_losses = []
    ratio_losses = []
    for ratio in TRAIN_RATIOS:
        threshold = torch.quantile(logits.detach(), 1.0 - ratio)
        keep = torch.sigmoid((logits - threshold) / SOFT_TOPK_TEMPERATURE)
        true_positive = torch.sum(keep * labels)
        soft_precision = true_positive / (torch.sum(keep) + 1e-6)
        soft_recall = torch.sum(keep * coverage) / max(float(record["num_gt_points"]), 1.0)
        soft_recall = soft_recall.clamp(0.0, 1.0)
        soft_f1 = 2.0 * soft_precision * soft_recall / (soft_precision + soft_recall + 1e-6)
        soft_f1_losses.append(1.0 - soft_f1)
        ratio_losses.append((keep.mean() - ratio) ** 2)

    positive_priority = coverage + 5.0 * occluded_positive
    negative_priority = (
        5.0 * ambiguity_negative
        + raw[:, CONF_RANK_INDEX] * (labels < 0.5).float()
    )
    positive_idx = torch.where(labels > 0.5)[0]
    negative_idx = torch.where(labels < 0.5)[0]
    pair_count = min(len(positive_idx), len(negative_idx), 1024)
    if pair_count:
        positive_idx = positive_idx[
            torch.topk(positive_priority[positive_idx], pair_count, largest=True).indices
        ]
        negative_idx = negative_idx[
            torch.topk(negative_priority[negative_idx], pair_count, largest=True).indices
        ]
        ranking = F.softplus(0.20 - logits[positive_idx] + logits[negative_idx]).mean()
    else:
        ranking = logits.new_tensor(0.0)

    reconstruction_f1 = torch.stack(soft_f1_losses).mean()
    ratio_loss = torch.stack(ratio_losses).mean()
    residual_loss = torch.mean(residual * residual)
    total = (
        LOSS_WEIGHTS["bce"] * bce
        + LOSS_WEIGHTS["reconstruction_f1"] * reconstruction_f1
        + LOSS_WEIGHTS["ranking"] * ranking
        + LOSS_WEIGHTS["ratio"] * ratio_loss
        + LOSS_WEIGHTS["residual"] * residual_loss
    )
    return total, {
        "bce_loss": float(bce.detach().cpu()),
        "reconstruction_f1_loss": float(reconstruction_f1.detach().cpu()),
        "ranking_loss": float(ranking.detach().cpu()),
        "ratio_loss": float(ratio_loss.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
    }


def predict_model_score(model, features, feature_mean, feature_std):
    device = next(model.parameters()).device
    mean_t = torch.from_numpy(feature_mean).float().to(device)
    std_t = torch.from_numpy(feature_std).float().to(device)
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            raw = torch.from_numpy(features[start : start + BATCH_SIZE]).float().to(device)
            logits, _residual = score_logits(model, raw, mean_t, std_t)
            chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def internal_reconstruction_validation(model, records, feature_mean, feature_std):
    rows = []
    for ratio in TRAIN_RATIOS:
        fscores = []
        for record in records:
            score = predict_model_score(model, record["features"], feature_mean, feature_std)
            mask, _threshold = exact_topk_mask(score, ratio, tie_breaker=record["conf"])
            selected = base.downsample(record["points"][mask].astype(np.float32), base.MAX_POINTS)
            fscores.append(base.compute_metrics(selected, record["gt"])["fscore"])
        rows.append(
            {
                "ratio": ratio,
                "mean_reconstruction_fscore": float(np.mean(fscores)) if fscores else 0.0,
            }
        )
    return max(rows, key=lambda row: row["mean_reconstruction_fscore"])


def train_model(fit_records, internal_val_records, feature_mean, feature_std, model_seed):
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    device = safe_device()
    model = JointCandidateAcceptanceHead(len(FEATURE_NAMES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    mean_t = torch.from_numpy(feature_mean).float().to(device)
    std_t = torch.from_numpy(feature_std).float().to(device)
    rng = np.random.default_rng(model_seed)
    history = []
    best_state = None
    best_validation = -1.0
    best_ratio = TRAIN_RATIOS[-1]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(fit_records))
        losses = []
        components = []
        for record_idx in order:
            loss, component = reconstruction_aware_group_loss(
                model,
                fit_records[int(record_idx)],
                mean_t,
                std_t,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            components.append(component)
        validation = internal_reconstruction_validation(
            model,
            internal_val_records,
            feature_mean,
            feature_std,
        )
        row = {
            "model_seed": model_seed,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{
                key: float(np.mean([component[key] for component in components]))
                for key in components[0]
            },
            "internal_val_best_ratio": validation["ratio"],
            "internal_val_reconstruction_fscore": validation["mean_reconstruction_fscore"],
        }
        history.append(row)
        print("Run 27 train row:", row)
        if validation["mean_reconstruction_fscore"] > best_validation:
            best_validation = validation["mean_reconstruction_fscore"]
            best_ratio = validation["ratio"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    return model, history, {
        "model_seed": model_seed,
        "best_internal_val_reconstruction_fscore": best_validation,
        "best_internal_val_ratio": best_ratio,
    }


def predict_ensemble(models, features, feature_mean, feature_std):
    scores = [
        predict_model_score(model, features, feature_mean, feature_std)
        for model in models
    ]
    mean_score = np.mean(np.stack(scores, axis=0), axis=0).astype(np.float32)
    probability = (1.0 / (1.0 + np.exp(-np.clip(mean_score, -20.0, 20.0)))).astype(np.float32)
    return mean_score, probability


def confidence_mask(conf, conf_percent):
    threshold = float(np.quantile(conf, conf_percent / 100.0))
    return conf >= threshold, threshold


def exact_topk_mask(score, ratio, tie_breaker=None):
    if len(score) == 0:
        return np.zeros(0, dtype=bool), 0.0
    ratio = float(np.clip(ratio, 0.0, 1.0))
    keep = max(MIN_SELECTED_POINTS, int(math.ceil(len(score) * ratio)))
    keep = min(keep, len(score))
    if keep >= len(score):
        return np.ones(len(score), dtype=bool), float(np.min(score)) if len(score) else 0.0
    if keep <= 0:
        return np.zeros(len(score), dtype=bool), float(np.max(score)) if len(score) else 0.0
    score = np.asarray(score, dtype=np.float32)
    if tie_breaker is None:
        tie_breaker = np.zeros(len(score), dtype=np.float32)
    else:
        tie_breaker = np.asarray(tie_breaker, dtype=np.float32)
    order = np.lexsort((np.arange(len(score)), -tie_breaker, -score))
    selected = order[:keep]
    mask = np.zeros(len(score), dtype=bool)
    mask[selected] = True
    threshold = float(score[order[keep - 1]])
    return mask, threshold


def method_family(method):
    if method in BASELINE_METHODS or method.startswith("confidence_"):
        return "baseline"
    if method == "rajah_internal_selected":
        return "learned_gate_candidate"
    return "learned_diagnostic"


def score_selection(points, mask, gt, method, group_row, extra):
    if int(mask.sum()) < MIN_SELECTED_POINTS:
        mask = np.ones(len(points), dtype=bool)
        fallback = 1
    else:
        fallback = 0
    selected = base.downsample(points[mask].astype(np.float32), base.MAX_POINTS)
    metrics = base.compute_metrics(selected, gt)
    return {
        "run": RUN_NAME,
        "method": method,
        "split": group_row.get("split"),
        "scene": group_row.get("scene"),
        "num_views": as_int(group_row, "num_views"),
        "view_policy": group_row.get("view_policy"),
        "group_key": group_row.get("group_key"),
        "group_classes": group_row.get("group_classes"),
        "mean_baseline_m": as_float(group_row, "mean_baseline_m"),
        "max_baseline_m": as_float(group_row, "max_baseline_m"),
        "method_family": method_family(method),
        "selected_ratio": float(mask.mean()) if len(mask) else 0.0,
        "fallback_all_points": fallback,
        **extra,
        **metrics,
    }


def evaluate_group(
    models,
    backbone,
    root,
    scene_lookup,
    group_row,
    out_dir,
    feature_mean,
    feature_std,
    selected_internal_ratio,
):
    pack = run_group(backbone, root, scene_lookup, group_row, out_dir)
    joint_score, prob = predict_ensemble(
        models,
        pack["features"],
        feature_mean,
        feature_std,
    )
    label_metrics = candidate_label_metrics(prob, pack["labels"], 0.5)
    num_views = as_int(group_row, "num_views")
    final_percent = base.FIXED_FINAL_CONF_BY_VIEW_COUNT.get(num_views, base.CONF_PERCENT)
    conf_final, conf_threshold = confidence_mask(pack["conf"], final_percent)
    conf_rank = rank_percentile(pack["conf"])
    combined_prob_conf = prob * conf_rank
    common_stats = {
        "runtime_seconds": pack["runtime_seconds"],
        "conf_percent": "",
        "conf_threshold": "",
        "jcah_ratio": "",
        "jcah_threshold": "",
        "jcah_score_threshold": "",
        "mean_keep_prob": float(prob.mean()) if len(prob) else 0.0,
        "selected_internal_ratio": selected_internal_ratio,
        "occluded_positive_ratio": float(pack["occluded_positive"].mean()),
        "ambiguity_negative_ratio": float(pack["ambiguity_negative"].mean()),
        "is_occlusion_challenging": int(
            float(pack["occluded_positive"].mean()) >= OCCLUSION_PROXY_THRESHOLD
        ),
        "is_ambiguity_challenging": int(
            float(pack["ambiguity_negative"].mean()) >= AMBIGUITY_PROXY_THRESHOLD
        ),
        **pack.get("photo_stats", {}),
        **label_metrics,
    }
    rows = [
        score_selection(
            pack["points"],
            conf_final,
            pack["gt"],
            "confidence_fixed_final",
            group_row,
            {
                **common_stats,
                "conf_percent": final_percent,
                "conf_threshold": conf_threshold,
            },
        ),
        score_selection(
            pack["points"],
            np.ones(len(pack["points"]), dtype=bool),
            pack["gt"],
            "all_candidates",
            group_row,
            common_stats,
        ),
    ]
    internal_mask, internal_threshold = exact_topk_mask(
        joint_score,
        selected_internal_ratio,
        tie_breaker=pack["conf"],
    )
    rows.append(
        score_selection(
            pack["points"],
            internal_mask,
            pack["gt"],
            "rajah_internal_selected",
            group_row,
            {
                **common_stats,
                "jcah_ratio": selected_internal_ratio,
                "jcah_score_threshold": internal_threshold,
                "target_selected_ratio": selected_internal_ratio,
            },
        )
    )
    for ratio in RANK_RATIOS:
        conf_mask, conf_score_threshold = exact_topk_mask(pack["conf"], ratio)
        rows.append(
            score_selection(
                pack["points"],
                conf_mask,
                pack["gt"],
                f"confidence_top_ratio_{ratio:.4f}",
                group_row,
                {
                    **common_stats,
                    "conf_threshold": conf_score_threshold,
                    "jcah_ratio": "",
                    "jcah_score_threshold": conf_score_threshold,
                    "target_selected_ratio": ratio,
                },
            )
        )
        mask, score_threshold = exact_topk_mask(joint_score, ratio, tie_breaker=pack["conf"])
        rows.append(
            score_selection(
                pack["points"],
                mask,
                pack["gt"],
                f"jcah_top_ratio_{ratio:.4f}",
                group_row,
                {
                    **common_stats,
                    "jcah_ratio": ratio,
                    "jcah_score_threshold": score_threshold,
                    "target_selected_ratio": ratio,
                },
            )
        )
        combined_mask, combined_score_threshold = exact_topk_mask(combined_prob_conf, ratio, tie_breaker=pack["conf"])
        rows.append(
            score_selection(
                pack["points"],
                combined_mask,
                pack["gt"],
                f"combined_jcah_conf_top_ratio_{ratio:.4f}",
                group_row,
                {
                    **common_stats,
                    "jcah_ratio": ratio,
                    "jcah_score_threshold": combined_score_threshold,
                    "target_selected_ratio": ratio,
                },
            )
        )
    matched_ratio = float(conf_final.mean())
    matched_mask, matched_threshold = exact_topk_mask(joint_score, matched_ratio, tie_breaker=pack["conf"])
    rows.append(
        score_selection(
            pack["points"],
            matched_mask,
            pack["gt"],
            "jcah_match_confidence_ratio",
            group_row,
            {
                **common_stats,
                "conf_percent": final_percent,
                "conf_threshold": conf_threshold,
                "jcah_ratio": matched_ratio,
                "jcah_score_threshold": matched_threshold,
                "target_selected_ratio": matched_ratio,
            },
        )
    )
    combined_matched_mask, combined_matched_threshold = exact_topk_mask(combined_prob_conf, matched_ratio, tie_breaker=pack["conf"])
    rows.append(
        score_selection(
            pack["points"],
            combined_matched_mask,
            pack["gt"],
            "combined_jcah_conf_match_confidence_ratio",
            group_row,
            {
                **common_stats,
                "conf_percent": final_percent,
                "conf_threshold": conf_threshold,
                "jcah_ratio": matched_ratio,
                "jcah_score_threshold": combined_matched_threshold,
                "target_selected_ratio": matched_ratio,
            },
        )
    )
    threshold_mask = prob >= 0.5
    rows.append(
        score_selection(
            pack["points"],
            threshold_mask,
            pack["gt"],
            "jcah_label_f1_threshold",
            group_row,
            {
                **common_stats,
                "jcah_threshold": 0.5,
                "jcah_score_threshold": 0.5,
            },
        )
    )
    guard_mask = threshold_mask & conf_final
    rows.append(
        score_selection(
            pack["points"],
            guard_mask,
            pack["gt"],
            "jcah_label_threshold_and_confidence_guard",
            group_row,
            {
                **common_stats,
                "conf_percent": final_percent,
                "conf_threshold": conf_threshold,
                "jcah_threshold": 0.5,
                "jcah_score_threshold": 0.5,
            },
        )
    )
    for row in rows:
        print("Run 27 metric row:", row)
    return rows, label_summary_row(group_row, pack, "eval")


def summarize(rows):
    out = []
    final_lookup = {}
    best_baseline_lookup = {}
    for split in sorted({r["split"] for r in rows}):
        vals = [r["fscore"] for r in rows if r["split"] == split and r["method"] == "confidence_fixed_final"]
        final_lookup[split] = float(np.mean(vals)) if vals else 0.0
        baseline_means = []
        for method in sorted({r["method"] for r in rows if r["split"] == split and r["method_family"] == "baseline"}):
            items = [r for r in rows if r["split"] == split and r["method"] == method]
            baseline_means.append(float(np.mean([r["fscore"] for r in items])) if items else 0.0)
        best_baseline_lookup[split] = max(baseline_means) if baseline_means else final_lookup[split]
    for split, method in sorted({(r["split"], r["method"]) for r in rows}):
        items = [r for r in rows if r["split"] == split and r["method"] == method]
        f = np.asarray([r["fscore"] for r in items], dtype=np.float32)
        family = method_family(method)
        out.append(
            {
                "run": RUN_NAME,
                "split": split,
                "method": method,
                "method_family": family,
                "num_groups": len(items),
                "mean_fscore": float(f.mean()) if len(f) else 0.0,
                "std_fscore": float(f.std()) if len(f) else 0.0,
                "delta_vs_confidence_fixed_final": float(f.mean() - final_lookup.get(split, 0.0)) if len(f) else 0.0,
                "delta_vs_best_baseline": float(f.mean() - best_baseline_lookup.get(split, 0.0)) if len(f) else 0.0,
                "mean_precision": float(np.mean([r["precision"] for r in items])) if items else 0.0,
                "mean_recall": float(np.mean([r["recall"] for r in items])) if items else 0.0,
                "mean_selected_ratio": float(np.mean([r["selected_ratio"] for r in items])) if items else 0.0,
                "mean_label_f1": float(np.mean([r["label_f1"] for r in items])) if items and "label_f1" in items[0] else 0.0,
                "mean_projected_targets": float(np.mean([r["mean_projected_targets"] for r in items])) if items and "mean_projected_targets" in items[0] else 0.0,
            }
        )
    return out


def summarize_limit_subsets(rows):
    output = []
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        group_diagnostics = {}
        for row in split_rows:
            group_diagnostics[row["group_key"]] = {
                "occlusion": float(row.get("occluded_positive_ratio", 0.0)),
                "ambiguity": float(row.get("ambiguity_negative_ratio", 0.0)),
            }
        challenging_count = max(1, int(math.ceil(len(group_diagnostics) / 3.0)))
        occlusion_keys = {
            key
            for key, _diag in sorted(
                group_diagnostics.items(),
                key=lambda item: (-item[1]["occlusion"], item[0]),
            )[:challenging_count]
        }
        ambiguity_keys = {
            key
            for key, _diag in sorted(
                group_diagnostics.items(),
                key=lambda item: (-item[1]["ambiguity"], item[0]),
            )[:challenging_count]
        }
        subset_specs = [
            ("overall", lambda row: True),
            ("occlusion_challenging", lambda row: row["group_key"] in occlusion_keys),
            ("ambiguity_challenging", lambda row: row["group_key"] in ambiguity_keys),
        ]
        for subset_name, predicate in subset_specs:
            subset_rows = [row for row in split_rows if predicate(row)]
            for method in sorted({row["method"] for row in subset_rows}):
                method_rows = [row for row in subset_rows if row["method"] == method]
                output.append(
                    {
                        "run": RUN_NAME,
                        "split": split,
                        "limit_subset": subset_name,
                        "method": method,
                        "method_family": method_family(method),
                        "num_groups": len(method_rows),
                        "mean_fscore": float(np.mean([row["fscore"] for row in method_rows])),
                        "mean_precision": float(np.mean([row["precision"] for row in method_rows])),
                        "mean_recall": float(np.mean([row["recall"] for row in method_rows])),
                        "mean_selected_ratio": float(np.mean([row["selected_ratio"] for row in method_rows])),
                    }
                )
    return output


def best_subset_comparison(limit_rows, subset_name):
    rows = [
        row
        for row in limit_rows
        if row["split"] == "val" and row["limit_subset"] == subset_name
    ]
    baselines = [row for row in rows if row["method_family"] == "baseline"]
    learned = [row for row in rows if row["method_family"] == "learned_gate_candidate"]
    if not baselines or not learned:
        return None
    best_baseline = max(baselines, key=lambda row: row["mean_fscore"])
    learned_row = learned[0]
    return {
        "subset": subset_name,
        "num_groups": learned_row["num_groups"],
        "best_baseline_method": best_baseline["method"],
        "best_baseline_fscore": best_baseline["mean_fscore"],
        "learned_fscore": learned_row["mean_fscore"],
        "delta": learned_row["mean_fscore"] - best_baseline["mean_fscore"],
    }


def make_gate_decision(summary_rows, limit_rows):
    val_rows = [row for row in summary_rows if row["split"] == "val"]
    final = next((row for row in val_rows if row["method"] == "confidence_fixed_final"), None)
    overall = best_subset_comparison(limit_rows, "overall")
    occlusion = best_subset_comparison(limit_rows, "occlusion_challenging")
    ambiguity = best_subset_comparison(limit_rows, "ambiguity_challenging")
    if not final or not overall or not occlusion or not ambiguity:
        return [
            {
                "run": RUN_NAME,
                "selected_method": "confidence_fixed_final",
                "reason": "Missing validation evidence for the overall, occlusion, or ambiguity gate.",
            }
        ]
    overall_pass = overall["delta"] >= GATE_MARGIN_F1
    occlusion_pass = occlusion["delta"] >= 0.0
    ambiguity_pass = ambiguity["delta"] >= 0.0
    pass_all_limits = overall_pass and occlusion_pass and ambiguity_pass
    selected = "rajah_internal_selected" if pass_all_limits else overall["best_baseline_method"]
    return [
        {
            "run": RUN_NAME,
            "selected_method": selected,
            "best_baseline_method": overall["best_baseline_method"],
            "best_learned_method": "rajah_internal_selected",
            "validation_confidence_fixed_fscore": final["mean_fscore"],
            "validation_best_baseline_fscore": overall["best_baseline_fscore"],
            "validation_best_learned_fscore": overall["learned_fscore"],
            "delta_vs_confidence_fixed": float(overall["learned_fscore"] - final["mean_fscore"]),
            "delta_vs_best_baseline": float(overall["delta"]),
            "occlusion_num_groups": occlusion["num_groups"],
            "occlusion_delta_vs_best_baseline": float(occlusion["delta"]),
            "ambiguity_num_groups": ambiguity["num_groups"],
            "ambiguity_delta_vs_best_baseline": float(ambiguity["delta"]),
            "overall_pass": int(overall_pass),
            "occlusion_non_regression_pass": int(occlusion_pass),
            "ambiguity_non_regression_pass": int(ambiguity_pass),
            "pass_all_limits": int(pass_all_limits),
            "gate_margin_f1": GATE_MARGIN_F1,
            "recommendation": (
                "Use the reconstruction-aware joint acceptance head on the untouched test split."
                if pass_all_limits
                else "Keep the best non-learned baseline; at least one overall/occlusion/ambiguity validation condition failed."
            ),
        }
    ]


def split_group_selection(group_manifest):
    train = balanced_group_subset(
        [row for row in group_manifest if row["split"] == "train"],
        MAX_TRAIN_GROUPS,
    )
    eval_groups = balanced_group_subset(
        [row for row in group_manifest if row["split"] in {"val", "test"}],
        MAX_EVAL_GROUPS,
    )
    return train, eval_groups


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_static_configuration()
    base.require_t4x2()
    root = base.clone_repo()
    base.install_deps(root)
    posed_root = base.find_posed_images_root()
    scene_dirs = discover_scene_dirs(posed_root)
    scene_lookup = {p.name: p for p in scene_dirs}
    splits = scene_splits(scene_dirs)
    group_manifest = build_group_manifest(scene_dirs, splits)
    train_groups, eval_groups = split_group_selection(group_manifest)
    fit_groups, internal_val_groups, internal_val_scenes = split_internal_train_groups(train_groups)
    if not fit_groups or not internal_val_groups or not eval_groups:
        raise RuntimeError(
            "Run 27 requires non-empty fit, internal-validation, and external-evaluation groups"
        )
    if not any(row["split"] == "val" for row in eval_groups):
        raise RuntimeError("Run 27 external evaluation must include validation groups")
    if not any(row["split"] == "test" for row in eval_groups):
        raise RuntimeError("Run 27 external evaluation must include test groups")
    print("POSED_IMAGES:", posed_root)
    print("SCENE_DIRS:", [str(p) for p in scene_dirs])
    print("SCENE_SPLITS:", splits)
    print("Run 27 fit groups:", len(fit_groups))
    print("Run 27 internal validation groups:", len(internal_val_groups))
    print("Run 27 internal validation scenes:", internal_val_scenes)
    print("Run 27 external evaluation groups:", len(eval_groups))
    for row in fit_groups:
        print("Run 27 selected fit group:", row)
    for row in internal_val_groups:
        print("Run 27 selected internal validation group:", row)
    for row in eval_groups:
        print("Run 27 selected external evaluation group:", row)

    ckpt_path = base.download_checkpoint(root)
    print("Checkpoint:", ckpt_path)
    backbone = base.load_model(root, ckpt_path)

    fit_records = []
    internal_val_records = []
    label_summary_rows = []
    for group in train_groups:
        pack = run_group(
            backbone,
            root,
            scene_lookup,
            group,
            out_dir / "train_groups" / group["group_key"],
        )
        record = {**pack, "group_row": group}
        stage = "internal_val" if group["scene"] in internal_val_scenes else "fit"
        if stage == "fit":
            fit_records.append(record)
        else:
            internal_val_records.append(record)
        label_summary_rows.append(label_summary_row(group, pack, stage))
        print(
            "Run 27 training group cache:",
            {
                "group": group.get("group_key"),
                "stage": stage,
                "rows": len(pack["labels"]),
                "positive_ratio": float(pack["labels"].mean()) if len(pack["labels"]) else 0.0,
                "covered_gt_ratio": float(
                    pack["coverage_mass"].sum() / max(pack["num_gt_points"], 1)
                ),
            },
        )
    num_fit_rows = sum(len(record["labels"]) for record in fit_records)
    fit_labels = np.concatenate([record["labels"] for record in fit_records])
    if num_fit_rows < 1000 or len(np.unique(fit_labels)) < 2:
        raise RuntimeError(
            f"Insufficient Run 27 fit rows: rows={num_fit_rows} positives={float(fit_labels.sum())}"
        )
    feature_mean, feature_std = feature_standardization(fit_records)
    print(
        "Run 27 fit matrix:",
        {
            "rows": num_fit_rows,
            "feature_dim": len(FEATURE_NAMES),
            "positive_ratio": float(fit_labels.mean()),
        },
    )

    models = []
    history = []
    model_selection_rows = []
    for model_seed in MODEL_SEEDS:
        model, model_history, model_selection = train_model(
            fit_records,
            internal_val_records,
            feature_mean,
            feature_std,
            model_seed,
        )
        models.append(model)
        history.extend(model_history)
        model_selection_rows.append(model_selection)
    ratio_votes = Counter(
        float(row["best_internal_val_ratio"])
        for row in model_selection_rows
    )
    selected_internal_ratio = ratio_votes.most_common(1)[0][0]
    print(
        "Run 27 ensemble selection:",
        {
            "model_rows": model_selection_rows,
            "selected_internal_ratio": selected_internal_ratio,
        },
    )

    metric_rows = []
    for group in eval_groups:
        rows, label_row = evaluate_group(
            models,
            backbone,
            root,
            scene_lookup,
            group,
            out_dir / "eval_groups" / group["group_key"],
            feature_mean,
            feature_std,
            selected_internal_ratio,
        )
        metric_rows.extend(rows)
        label_summary_rows.append(label_row)

    summary_rows = summarize(metric_rows)
    limit_summary_rows = summarize_limit_subsets(metric_rows)
    gate_rows = make_gate_decision(summary_rows, limit_summary_rows)
    write_csv_union(out_dir / "candidate_label_summary.csv", label_summary_rows)
    write_csv_union(out_dir / "training_history.csv", history)
    write_csv_union(out_dir / "metrics.csv", metric_rows)
    write_csv_union(out_dir / "summary.csv", summary_rows)
    write_csv_union(out_dir / "limit_summary.csv", limit_summary_rows)
    write_csv_union(out_dir / "gate_decision.csv", gate_rows)
    write_csv_union(out_dir / "selected_train_groups.csv", train_groups)
    write_csv_union(out_dir / "selected_eval_groups.csv", eval_groups)
    write_csv_union(out_dir / "model_selection.csv", model_selection_rows)
    write_csv_union(
        out_dir / "scene_split.csv",
        [{"scene": scene, "split": split} for scene, split in sorted(splits.items())],
    )
    write_csv_union(out_dir / "view_group_manifest.csv", group_manifest)
    torch.save(
        {
            "state_dicts": [model.state_dict() for model in models],
            "model_seeds": MODEL_SEEDS,
            "feature_dim": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "selected_internal_ratio": selected_internal_ratio,
            "rank_ratios": RANK_RATIOS,
        },
        out_dir / "joint_candidate_acceptance_head.pt",
    )
    config = {
        "run": RUN_NAME,
        "self_contained": True,
        "dataset_dependency": "tiantiansyrinx1102/scannet-data",
        "private_kernel_dependencies": [],
        "num_scenes": len(scene_dirs),
        "scene_splits": splits,
        "num_train_groups": len(train_groups),
        "num_fit_groups": len(fit_records),
        "num_internal_val_groups": len(internal_val_records),
        "num_eval_groups": len(eval_groups),
        "max_train_groups": MAX_TRAIN_GROUPS,
        "max_eval_groups": MAX_EVAL_GROUPS,
        "max_candidates_per_group": MAX_CANDIDATES_PER_GROUP,
        "num_fit_rows": num_fit_rows,
        "feature_dim": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "label_positive_threshold_m": LABEL_POS_M,
        "hard_negative_threshold_m": HARD_NEG_M,
        "model_seeds": MODEL_SEEDS,
        "train_ratios": TRAIN_RATIOS,
        "rank_ratios": RANK_RATIOS,
        "selected_internal_ratio": selected_internal_ratio,
        "loss_weights": LOSS_WEIGHTS,
        "residual_scale": RESIDUAL_SCALE,
        "soft_topk_temperature": SOFT_TOPK_TEMPERATURE,
        "occlusion_proxy_threshold": OCCLUSION_PROXY_THRESHOLD,
        "ambiguity_proxy_threshold": AMBIGUITY_PROXY_THRESHOLD,
        "gate_margin_f1": GATE_MARGIN_F1,
        "runtime_seconds": time.time() - started,
        "note": (
            "Run 27 is self-contained and removes the Run 20/24 private-kernel dependency. "
            "It learns a bounded residual over MV-DUSt3R confidence from actual reconstruction "
            "candidates using point layout, cross-view geometric support, and aggregated raw "
            "image-patch consistency. Its differentiable soft-top-k objective approximates "
            "reconstruction precision and GT-surface coverage recall, with extra ranking weight "
            "for low-support valid occluded points and high-confidence wrong-depth negatives. "
            "Scene-level internal validation selects the keep ratio before the external val/test "
            "gate, and a three-seed ensemble reduces training variance."
        ),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    print("Run 27 config:")
    print(config)
    print("Run 27 summary:")
    for row in summary_rows:
        print(row)
    print("Run 27 limit summary:")
    for row in limit_summary_rows:
        if row["method"] in {
            "confidence_fixed_final",
            "all_candidates",
            "rajah_internal_selected",
        }:
            print(row)
    print("Run 27 gate decision:")
    for row in gate_rows:
        print(row)


if __name__ == "__main__":
    main()
