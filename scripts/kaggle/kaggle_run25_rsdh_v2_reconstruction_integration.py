import csv
import json
import math
import os
import random
import sys
import time
import urllib.request
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree


RAW_BASE = "https://raw.githubusercontent.com/NguyenNgocMinh30012005/Computer_vision_2025.2/main/scripts/kaggle/kaggle_run11_final_validation_3seeds.py"


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


RUN_NAME = "run_25_rsdh_v2_reconstruction_integration"
RUN19_SEED = 1919
SEED = 2525
MAX_GROUPS = int(os.environ.get("RUN25_MAX_GROUPS", "32"))
MAX_CANDIDATES_PER_GROUP = int(os.environ.get("RUN25_MAX_CANDIDATES_PER_GROUP", "5000"))
MIN_SELECTED_POINTS = int(os.environ.get("RUN25_MIN_SELECTED_POINTS", "100"))
GATE_MARGIN_F1 = float(os.environ.get("RUN25_GATE_MARGIN_F1", "0.005"))
BATCH_SIZE = 65536
PATCH_RADIUS = 12
DOWNSAMPLE = 8
LABEL_POS_M = float(getattr(base, "F_SCORE_THRESHOLD_M", 0.05))
RANK_RATIOS = [0.85, 0.95, 0.97, 0.98, 0.995]


def feature_names():
    names = [
        "src_x_norm",
        "src_y_norm",
        "target_x_norm",
        "target_y_norm",
        "pixel_dx",
        "pixel_dy",
        "pixel_distance",
        "num_views_norm",
        "policy_hybrid",
        "policy_diversity",
    ]
    for prefix in ["src_rgb_mean", "target_rgb_mean", "abs_rgb_mean_diff", "src_rgb_std", "target_rgb_std", "abs_rgb_std_diff"]:
        names.extend([f"{prefix}_{c}" for c in ["r", "g", "b"]])
    names.extend(
        [
            "src_gray_mean",
            "target_gray_mean",
            "abs_gray_mean_diff",
            "src_gray_std",
            "target_gray_std",
            "abs_gray_std_diff",
            "gray_patch_corr",
            "gray_patch_l1",
            "gray_patch_l2",
            "src_grad_mean",
            "target_grad_mean",
            "abs_grad_mean_diff",
            "src_grad_std",
            "target_grad_std",
            "abs_grad_std_diff",
            "grad_patch_corr",
            "grad_patch_l1",
            "grad_patch_l2",
        ]
    )
    names.extend([f"gray_down_abs_{i:02d}" for i in range(DOWNSAMPLE * DOWNSAMPLE)])
    names.extend([f"gray_down_prod_{i:02d}" for i in range(DOWNSAMPLE * DOWNSAMPLE)])
    return names


FEATURE_NAMES = feature_names()


class RSDHv2(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(192, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(96, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", "nan", "NaN", None):
        return default
    try:
        out = float(value)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def as_int(row, key, default=0):
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def stable_seed(text):
    return SEED + (zlib.crc32(str(text).encode("utf-8")) % 100000)


def find_file_from_kernel_source(filename, preferred_tokens):
    root = Path("/kaggle/input")
    candidates = sorted(root.rglob(filename))
    for path in candidates:
        text = str(path).lower()
        if any(token in text for token in preferred_tokens):
            print(f"Using {filename}:", path)
            return path
    if candidates:
        print(f"Using first {filename}:", candidates[0])
        return candidates[0]
    raise FileNotFoundError(f"Cannot find {filename}. Add the required Kaggle kernel source.")


def discover_scene_dirs(posed_root):
    scenes = sorted([p for p in posed_root.glob("scene*") if p.is_dir() and list(p.glob("*.jpg"))])
    if not scenes:
        raise FileNotFoundError(f"No scene directories with JPG frames under {posed_root}")
    return scenes


def intrinsics(image_shape):
    h, w = image_shape[:2]
    fx = 577.870605 * (w / 640.0)
    fy = 577.870605 * (h / 480.0)
    cx = 319.5 * (w / 640.0)
    cy = 239.5 * (h / 480.0)
    return fx, fy, cx, cy


def project_cam_point(pt_cam, image_shape):
    z = float(pt_cam[2])
    if z <= 0.10 or not np.isfinite(z):
        return None
    fx, fy, cx, cy = intrinsics(image_shape)
    x = float(pt_cam[0] / z * fx + cx)
    y = float(pt_cam[1] / z * fy + cy)
    h, w = image_shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    return x, y, z


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


def rank_percentile(values):
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


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


def load_image_bundle(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    return {"rgb": rgb, "gray": gray, "grad": grad}


def crop_patch(arr, x, y, radius):
    h, w = arr.shape[:2]
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    pad_spec = ((radius, radius), (radius, radius))
    if arr.ndim == 3:
        pad_spec += ((0, 0),)
    padded = np.pad(arr, pad_spec, mode="reflect")
    xi += radius
    yi += radius
    return padded[yi - radius : yi + radius + 1, xi - radius : xi + radius + 1]


def corr(a, b):
    av = a.reshape(-1).astype(np.float32)
    bv = b.reshape(-1).astype(np.float32)
    av = av - float(av.mean())
    bv = bv - float(bv.mean())
    denom = float(np.sqrt(np.sum(av * av) * np.sum(bv * bv))) + 1e-8
    return float(np.sum(av * bv) / denom)


def patch_features(src_bundle, tgt_bundle, src_xy, tgt_xy):
    sx, sy = src_xy
    tx, ty = tgt_xy
    src_rgb = crop_patch(src_bundle["rgb"], sx, sy, PATCH_RADIUS)
    tgt_rgb = crop_patch(tgt_bundle["rgb"], tx, ty, PATCH_RADIUS)
    src_gray = crop_patch(src_bundle["gray"], sx, sy, PATCH_RADIUS)
    tgt_gray = crop_patch(tgt_bundle["gray"], tx, ty, PATCH_RADIUS)
    src_grad = crop_patch(src_bundle["grad"], sx, sy, PATCH_RADIUS)
    tgt_grad = crop_patch(tgt_bundle["grad"], tx, ty, PATCH_RADIUS)

    src_mean = src_rgb.reshape(-1, 3).mean(axis=0)
    tgt_mean = tgt_rgb.reshape(-1, 3).mean(axis=0)
    src_std = src_rgb.reshape(-1, 3).std(axis=0)
    tgt_std = tgt_rgb.reshape(-1, 3).std(axis=0)
    gray_corr = corr(src_gray, tgt_gray)
    grad_corr = corr(src_grad, tgt_grad)
    gray_l1 = float(np.mean(np.abs(src_gray - tgt_gray)))
    gray_l2 = float(np.sqrt(np.mean((src_gray - tgt_gray) ** 2)))
    grad_l1 = float(np.mean(np.abs(src_grad - tgt_grad)))
    grad_l2 = float(np.sqrt(np.mean((src_grad - tgt_grad) ** 2)))

    s8 = cv2.resize(src_gray, (DOWNSAMPLE, DOWNSAMPLE), interpolation=cv2.INTER_AREA)
    t8 = cv2.resize(tgt_gray, (DOWNSAMPLE, DOWNSAMPLE), interpolation=cv2.INTER_AREA)
    s8 = (s8 - float(s8.mean())) / (float(s8.std()) + 1e-6)
    t8 = (t8 - float(t8.mean())) / (float(t8.std()) + 1e-6)
    down_abs = np.abs(s8 - t8).reshape(-1)
    down_prod = (s8 * t8).reshape(-1)

    feats = []
    feats.extend(src_mean.tolist())
    feats.extend(tgt_mean.tolist())
    feats.extend(np.abs(src_mean - tgt_mean).tolist())
    feats.extend(src_std.tolist())
    feats.extend(tgt_std.tolist())
    feats.extend(np.abs(src_std - tgt_std).tolist())
    feats.extend(
        [
            float(src_gray.mean()),
            float(tgt_gray.mean()),
            float(abs(src_gray.mean() - tgt_gray.mean())),
            float(src_gray.std()),
            float(tgt_gray.std()),
            float(abs(src_gray.std() - tgt_gray.std())),
            gray_corr,
            gray_l1,
            gray_l2,
            float(src_grad.mean()),
            float(tgt_grad.mean()),
            float(abs(src_grad.mean() - tgt_grad.mean())),
            float(src_grad.std()),
            float(tgt_grad.std()),
            float(abs(src_grad.std() - tgt_grad.std())),
            grad_corr,
            grad_l1,
            grad_l2,
        ]
    )
    feats.extend(down_abs.astype(np.float32).tolist())
    feats.extend(down_prod.astype(np.float32).tolist())
    return feats


def load_rsdh_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_dim = len(ckpt.get("feature_names", FEATURE_NAMES))
    model = RSDHv2(feature_dim)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    threshold = float(ckpt.get("threshold", 0.3906255567446351))
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    print("Loaded Run 24 RSDH checkpoint:", checkpoint_path)
    print("Run 24 threshold:", threshold)
    return model, threshold, mean, std


def predict_prob(model, features, mean, std):
    if len(features) == 0:
        return np.zeros((0,), dtype=np.float32)
    x = (features.astype(np.float32) - mean) / std
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            xb = torch.from_numpy(x[start : start + BATCH_SIZE]).float()
            chunks.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def build_pair_features(points, xs, ys, view_ids, conf, view_files, group_row, image_h, image_w):
    poses = [base.parse_pose(str(p).replace(".jpg", ".txt")) for p in view_files]
    inv_poses = [np.linalg.inv(p) for p in poses]
    world_from_first = poses[0]
    bundles = [load_image_bundle(p) for p in view_files]
    policy = group_row.get("view_policy", "")
    num_views = max(as_float(group_row, "num_views"), 1.0)

    features = []
    candidate_indices = []
    target_counts = np.zeros(len(points), dtype=np.float32)
    for i, pt in enumerate(points):
        src_idx = int(np.clip(view_ids[i], 0, len(view_files) - 1))
        src_bundle = bundles[src_idx]
        src_h, src_w = src_bundle["gray"].shape
        sx_norm = float(xs[i] / max(image_w - 1, 1))
        sy_norm = float(ys[i] / max(image_h - 1, 1))
        sx = sx_norm * max(src_w - 1, 1)
        sy = sy_norm * max(src_h - 1, 1)

        pt_first = np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float32)
        pt_world = world_from_first @ pt_first
        for tgt_idx, tgt_bundle in enumerate(bundles):
            if tgt_idx == src_idx:
                continue
            pt_tgt = inv_poses[tgt_idx] @ pt_world
            projected = project_cam_point(pt_tgt[:3], tgt_bundle["gray"].shape)
            if projected is None:
                continue
            tx, ty, _tz = projected
            tgt_h, tgt_w = tgt_bundle["gray"].shape
            tx_norm = float(tx / max(tgt_w - 1, 1))
            ty_norm = float(ty / max(tgt_h - 1, 1))
            base_feats = [
                sx_norm,
                sy_norm,
                tx_norm,
                ty_norm,
                tx_norm - sx_norm,
                ty_norm - sy_norm,
                float(math.sqrt((tx_norm - sx_norm) ** 2 + (ty_norm - sy_norm) ** 2)),
                num_views / 5.0,
                1.0 if policy == "hybrid" else 0.0,
                1.0 if policy == "diversity_aware" else 0.0,
            ]
            features.append(base_feats + patch_features(src_bundle, tgt_bundle, (sx, sy), (tx, ty)))
            candidate_indices.append(i)
            target_counts[i] += 1.0
        if (i + 1) % 5000 == 0:
            print("Run 25 pair features:", {"candidates": i + 1, "pairs": len(features)})
    if not features:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.int64), target_counts
    return np.asarray(features, dtype=np.float32), np.asarray(candidate_indices, dtype=np.int64), target_counts


def aggregate_candidate_scores(pair_prob, candidate_indices, target_counts, n_candidates, threshold):
    max_prob = np.zeros(n_candidates, dtype=np.float32)
    sum_prob = np.zeros(n_candidates, dtype=np.float32)
    good_counts = np.zeros(n_candidates, dtype=np.float32)
    if len(pair_prob):
        np.maximum.at(max_prob, candidate_indices, pair_prob)
        np.add.at(sum_prob, candidate_indices, pair_prob)
        np.add.at(good_counts, candidate_indices, (pair_prob >= threshold).astype(np.float32))
    mean_prob = sum_prob / np.maximum(target_counts, 1.0)
    valid_frac = good_counts / np.maximum(target_counts, 1.0)
    return max_prob, mean_prob.astype(np.float32), valid_frac.astype(np.float32)


def candidate_labels(points, gt):
    aligned = base.center_scale_align(points.astype(np.float32), gt)
    dists, _ = cKDTree(gt).query(aligned, k=1, workers=-1)
    labels = (dists <= LABEL_POS_M).astype(np.float32)
    return labels


def label_metrics(mask, labels):
    pred = mask.astype(bool)
    y = labels > 0.5
    tp = float((pred & y).sum())
    fp = float((pred & ~y).sum())
    fn = float((~pred & y).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"label_precision": float(precision), "label_recall": float(recall), "label_f1": float(f1)}


def confidence_mask(conf, conf_percent):
    threshold = float(np.quantile(conf, conf_percent / 100.0))
    return conf >= threshold, threshold


def ratio_mask(score, ratio):
    if len(score) == 0:
        return np.zeros(0, dtype=bool), 0.0
    ratio = float(np.clip(ratio, 0.0, 1.0))
    keep = max(MIN_SELECTED_POINTS, int(math.ceil(len(score) * ratio)))
    keep = min(keep, len(score))
    kth = len(score) - keep
    threshold = float(np.partition(score, kth)[kth])
    return score >= threshold, threshold


def score_selection(points, mask, gt, labels, method, group_row, extra):
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
        "priority_score": as_float(group_row, "priority_score"),
        "selected_ratio": float(mask.mean()) if len(mask) else 0.0,
        "fallback_all_points": fallback,
        **extra,
        **label_metrics(mask, labels),
        **metrics,
    }


def choose_group_views(scene_lookup, group_row):
    scene_dir = scene_lookup.get(group_row["scene"])
    if scene_dir is None:
        raise FileNotFoundError(f"Scene {group_row['scene']} not found in posed_images")
    num_views = as_int(group_row, "num_views")
    policy = group_row.get("view_policy")
    return base.choose_views(scene_dir, num_views, policy, seed=RUN19_SEED + num_views)


def evaluate_group(rsdh, threshold, mean, std, backbone, root, scene_lookup, group_row, out_dir):
    view_files = choose_group_views(scene_lookup, group_row)
    print("Run 25 group views:", {"group": group_row.get("group_key"), "views": [p.name for p in view_files]})
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
    labels = candidate_labels(points, gt)
    pair_features, candidate_indices, target_counts = build_pair_features(
        points, xs, ys, view_ids, conf, view_files, group_row, image_h, image_w
    )
    pair_prob = predict_prob(rsdh, pair_features, mean, std)
    max_prob, mean_prob, valid_frac = aggregate_candidate_scores(
        pair_prob, candidate_indices, target_counts, len(points), threshold
    )
    conf_rank = rank_percentile(conf)
    combined_max = conf_rank * max_prob
    combined_mean = conf_rank * mean_prob

    num_views = as_int(group_row, "num_views")
    final_percent = base.FIXED_FINAL_CONF_BY_VIEW_COUNT.get(num_views, base.CONF_PERCENT)
    conf_final, conf_threshold = confidence_mask(conf, final_percent)
    rows = [
        score_selection(
            points,
            conf_final,
            gt,
            labels,
            "confidence_fixed_final",
            group_row,
            {
                "runtime_seconds": runtime,
                "conf_percent": final_percent,
                "conf_threshold": conf_threshold,
                "rsdh_threshold": "",
                "score_threshold": "",
                "mean_rsdh_max_prob": float(max_prob.mean()),
                "mean_rsdh_mean_prob": float(mean_prob.mean()),
                "mean_rsdh_valid_frac": float(valid_frac.mean()),
                "mean_projected_targets": float(target_counts.mean()),
                "num_pair_features": int(len(pair_features)),
            },
        )
    ]
    threshold_masks = {
        "rsdh_max_threshold_only": max_prob >= threshold,
        "rsdh_max_threshold_and_confidence": (max_prob >= threshold) & conf_final,
        "rsdh_valid_frac_and_confidence": (valid_frac > 0.0) & conf_final,
    }
    for method, mask in threshold_masks.items():
        rows.append(
            score_selection(
                points,
                mask,
                gt,
                labels,
                method,
                group_row,
                {
                    "runtime_seconds": runtime,
                    "conf_percent": final_percent if "confidence" in method else "",
                    "conf_threshold": conf_threshold if "confidence" in method else "",
                    "rsdh_threshold": threshold,
                    "score_threshold": threshold,
                    "mean_rsdh_max_prob": float(max_prob.mean()),
                    "mean_rsdh_mean_prob": float(mean_prob.mean()),
                    "mean_rsdh_valid_frac": float(valid_frac.mean()),
                    "mean_projected_targets": float(target_counts.mean()),
                    "num_pair_features": int(len(pair_features)),
                },
            )
        )

    matched_ratio = float(conf_final.mean())
    rank_scores = {
        "rsdh_max_match_confidence_ratio": max_prob,
        "rsdh_mean_match_confidence_ratio": mean_prob,
        "combined_max_match_confidence_ratio": combined_max,
        "combined_mean_match_confidence_ratio": combined_mean,
    }
    for method, score in rank_scores.items():
        mask, score_threshold = ratio_mask(score, matched_ratio)
        rows.append(
            score_selection(
                points,
                mask,
                gt,
                labels,
                method,
                group_row,
                {
                    "runtime_seconds": runtime,
                    "conf_percent": final_percent,
                    "conf_threshold": conf_threshold,
                    "rsdh_threshold": threshold,
                    "score_threshold": score_threshold,
                    "mean_rsdh_max_prob": float(max_prob.mean()),
                    "mean_rsdh_mean_prob": float(mean_prob.mean()),
                    "mean_rsdh_valid_frac": float(valid_frac.mean()),
                    "mean_projected_targets": float(target_counts.mean()),
                    "num_pair_features": int(len(pair_features)),
                },
            )
        )

    for ratio in RANK_RATIOS:
        for method, score in [("rsdh_max", max_prob), ("combined_max", combined_max)]:
            mask, score_threshold = ratio_mask(score, ratio)
            rows.append(
                score_selection(
                    points,
                    mask,
                    gt,
                    labels,
                    f"{method}_top_ratio_{ratio:.3f}",
                    group_row,
                    {
                        "runtime_seconds": runtime,
                        "conf_percent": "",
                        "conf_threshold": "",
                        "rsdh_threshold": threshold,
                        "score_threshold": score_threshold,
                        "mean_rsdh_max_prob": float(max_prob.mean()),
                        "mean_rsdh_mean_prob": float(mean_prob.mean()),
                        "mean_rsdh_valid_frac": float(valid_frac.mean()),
                        "mean_projected_targets": float(target_counts.mean()),
                        "num_pair_features": int(len(pair_features)),
                    },
                )
            )

    del output
    torch.cuda.empty_cache()
    for row in rows:
        print("Run 25 metric row:", row)
    return rows


def select_groups(manifest_rows):
    rows = [r for r in manifest_rows if str(r.get("is_final_eval_candidate", "0")) in {"1", "True", "true"}]
    rows = sorted(rows, key=lambda r: (r.get("split") != "val", -as_float(r, "priority_score"), r.get("scene", ""), as_int(r, "num_views")))
    if MAX_GROUPS > 0:
        rows = rows[:MAX_GROUPS]
    return rows


def summarize(rows):
    out = []
    final_lookup = {}
    for split in sorted({r["split"] for r in rows}):
        vals = [r["fscore"] for r in rows if r["split"] == split and r["method"] == "confidence_fixed_final"]
        final_lookup[split] = float(np.mean(vals)) if vals else 0.0
    for split, method in sorted({(r["split"], r["method"]) for r in rows}):
        items = [r for r in rows if r["split"] == split and r["method"] == method]
        f = np.asarray([r["fscore"] for r in items], dtype=np.float32)
        out.append(
            {
                "run": RUN_NAME,
                "split": split,
                "method": method,
                "num_groups": len(items),
                "mean_fscore": float(f.mean()) if len(f) else 0.0,
                "std_fscore": float(f.std()) if len(f) else 0.0,
                "delta_vs_confidence_fixed_final": float(f.mean() - final_lookup.get(split, 0.0)) if len(f) else 0.0,
                "mean_precision": float(np.mean([r["precision"] for r in items])) if items else 0.0,
                "mean_recall": float(np.mean([r["recall"] for r in items])) if items else 0.0,
                "mean_selected_ratio": float(np.mean([r["selected_ratio"] for r in items])) if items else 0.0,
                "mean_label_f1": float(np.mean([r["label_f1"] for r in items])) if items else 0.0,
                "mean_projected_targets": float(np.mean([r["mean_projected_targets"] for r in items])) if items else 0.0,
            }
        )
    return out


def make_gate_decision(summary_rows):
    val_rows = [r for r in summary_rows if r["split"] == "val"]
    final = next((r for r in val_rows if r["method"] == "confidence_fixed_final"), None)
    learned = [r for r in val_rows if r["method"] != "confidence_fixed_final"]
    if not final or not learned:
        return [
            {
                "run": RUN_NAME,
                "selected_method": "confidence_fixed_final",
                "reason": "No validation rows available for learned gate.",
            }
        ]
    best = max(learned, key=lambda r: r["mean_fscore"])
    delta = best["mean_fscore"] - final["mean_fscore"]
    selected = best["method"] if delta >= GATE_MARGIN_F1 else "confidence_fixed_final"
    return [
        {
            "run": RUN_NAME,
            "selected_method": selected,
            "best_learned_method": best["method"],
            "validation_confidence_fscore": final["mean_fscore"],
            "validation_best_learned_fscore": best["mean_fscore"],
            "delta_fscore": float(delta),
            "gate_margin_f1": GATE_MARGIN_F1,
            "recommendation": (
                "Use RSDH v2 reconstruction integration in the next joint learned run."
                if selected != "confidence_fixed_final"
                else "Keep RSDH v2 out of reconstruction until it wins validation F-score."
            ),
        }
    ]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    base.require_t4x2()
    run20_manifest = find_file_from_kernel_source(
        "final_eval_group_manifest.csv",
        ["run-20", "run_20", "subset-mining"],
    )
    run24_ckpt = find_file_from_kernel_source(
        "rsdh_v2_image_only_head.pt",
        ["run-24", "run_24", "rsdh-v2-image-only"],
    )
    manifest_rows = read_csv(run20_manifest)
    eval_groups = select_groups(manifest_rows)
    print("Run 25 eval groups:", len(eval_groups))
    for row in eval_groups:
        print("Run 25 selected group:", row)

    root = base.clone_repo()
    base.install_deps(root)
    posed_root = base.find_posed_images_root()
    scene_dirs = discover_scene_dirs(posed_root)
    scene_lookup = {p.name: p for p in scene_dirs}
    print("POSED_IMAGES:", posed_root)
    print("SCENE_DIRS:", [str(p) for p in scene_dirs])

    ckpt_path = base.download_checkpoint(root)
    print("Checkpoint:", ckpt_path)
    backbone = base.load_model(root, ckpt_path)
    rsdh, threshold, mean, std = load_rsdh_model(run24_ckpt)

    metric_rows = []
    for group in eval_groups:
        group_out = out_dir / "groups" / group["group_key"]
        metric_rows.extend(evaluate_group(rsdh, threshold, mean, std, backbone, root, scene_lookup, group, group_out))

    summary_rows = summarize(metric_rows)
    gate_rows = make_gate_decision(summary_rows)
    write_csv_union(out_dir / "metrics.csv", metric_rows)
    write_csv_union(out_dir / "summary.csv", summary_rows)
    write_csv_union(out_dir / "gate_decision.csv", gate_rows)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run": RUN_NAME,
                "source_run20_manifest": str(run20_manifest),
                "source_run24_checkpoint": str(run24_ckpt),
                "num_eval_groups": len(eval_groups),
                "max_groups": MAX_GROUPS,
                "max_candidates_per_group": MAX_CANDIDATES_PER_GROUP,
                "rank_ratios": RANK_RATIOS,
                "run24_threshold": threshold,
                "gate_margin_f1": GATE_MARGIN_F1,
                "runtime_seconds": time.time() - started,
                "note": (
                    "Run 25 integrates the Run 24 image-only RSDH v2 head into reconstruction. "
                    "For each MV-DUSt3R candidate point it projects the point into other selected views, "
                    "scores RGB patch match validity, aggregates candidate support, and compares RSDH-based "
                    "selection/ranking policies against fixed confidence on validation/test reconstruction F-score. "
                    "Absolute F-scores use a capped candidate pool for tractable image-patch scoring; the gate compares "
                    "all methods on the same pool."
                ),
            },
            indent=2,
        )
    )

    print("Run 25 summary:")
    for row in summary_rows:
        print(row)
    print("Run 25 gate decision:")
    for row in gate_rows:
        print(row)


if __name__ == "__main__":
    main()
