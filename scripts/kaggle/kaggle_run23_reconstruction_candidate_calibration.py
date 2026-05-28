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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


RUN_NAME = "run_23_reconstruction_candidate_calibration"
RUN19_SEED = 1919
SEED = 2323
MAX_TRAIN_GROUPS = int(os.environ.get("RUN23_MAX_TRAIN_GROUPS", "24"))
MAX_EVAL_GROUPS = int(os.environ.get("RUN23_MAX_EVAL_GROUPS", "32"))
MAX_CANDIDATES_PER_GROUP = int(os.environ.get("RUN23_MAX_CANDIDATES_PER_GROUP", "90000"))
MAX_TRAIN_ROWS_PER_GROUP = int(os.environ.get("RUN23_MAX_TRAIN_ROWS_PER_GROUP", "22000"))
MIN_SELECTED_POINTS = int(os.environ.get("RUN23_MIN_SELECTED_POINTS", "100"))
GATE_MARGIN_F1 = float(os.environ.get("RUN23_GATE_MARGIN_F1", "0.005"))
EPOCHS = int(os.environ.get("RUN23_EPOCHS", "18"))
BATCH_SIZE = int(os.environ.get("RUN23_BATCH_SIZE", "8192"))
LR = float(os.environ.get("RUN23_LR", "0.001"))
LABEL_POS_M = float(os.environ.get("RUN23_LABEL_POS_M", str(base.F_SCORE_THRESHOLD_M)))
HARD_NEG_M = float(os.environ.get("RUN23_HARD_NEG_M", "0.15"))
RANK_RATIOS = [0.30, 0.50, 0.70, 0.85, 0.95, 0.97, 0.98, 0.995]

FEATURE_NAMES = [
    "log_conf",
    "conf_z",
    "conf_rank",
    "src_x_norm",
    "src_y_norm",
    "src_view_norm",
    "num_views_norm",
    "policy_hybrid",
    "policy_diversity",
    "class_occlusion_core",
    "class_occlusion_borderline",
    "class_low_overlap_far",
    "class_wrong_depth_hard_negative",
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


class CandidateReliabilityHead(nn.Module):
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
        x = float(value)
    except ValueError:
        return default
    return x if math.isfinite(x) else default


def as_int(row, key, default=0):
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def row_classes(row):
    return str(row.get("group_classes", "")).split("|")


def stable_seed(text):
    return SEED + (zlib.crc32(str(text).encode("utf-8")) % 100000)


def find_run20_dir():
    root = Path("/kaggle/input")
    candidates = sorted(root.rglob("subset_group_manifest.csv"))
    for path in candidates:
        text = str(path).lower()
        if "run-20" in text or "run_20" in text or "subset-mining" in text:
            print("Using Run 20 output:", path.parent)
            return path.parent
    if candidates:
        print("Using first Run 20-like output:", candidates[0].parent)
        return candidates[0].parent
    raise FileNotFoundError("Cannot find Run 20 output. Add Run 20 as a Kaggle kernel source.")


def discover_scene_dirs(posed_root):
    scenes = sorted([p for p in posed_root.glob("scene*") if p.is_dir() and list(p.glob("*.jpg"))])
    if not scenes:
        raise FileNotFoundError(f"No scene directories with JPG frames under {posed_root}")
    return scenes


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
    idx = rng.choice(len(points), max_candidates, replace=False)
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
    classes = row_classes(group_row)
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
            np.full(n, 1.0 if "occlusion_core" in classes else 0.0, dtype=np.float32),
            np.full(n, 1.0 if "occlusion_borderline" in classes else 0.0, dtype=np.float32),
            np.full(n, 1.0 if "low_overlap_far" in classes else 0.0, dtype=np.float32),
            np.full(n, 1.0 if "wrong_depth_hard_negative" in classes else 0.0, dtype=np.float32),
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


def candidate_labels(points, gt):
    aligned = base.center_scale_align(points.astype(np.float32), gt)
    dists, _ = cKDTree(gt).query(aligned, k=1, workers=-1)
    dists = dists.astype(np.float32)
    labels = (dists <= LABEL_POS_M).astype(np.float32)
    hard_negative = (dists >= HARD_NEG_M).astype(np.float32)
    return labels, dists, hard_negative


def sample_training_rows(features, labels, hard_negative, conf, limit, seed):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels > 0.5)[0]
    neg_idx = np.where(labels <= 0.5)[0]
    hard_idx = np.where(hard_negative > 0.5)[0]
    high_conf_neg = neg_idx[np.argsort(conf[neg_idx])[-min(len(neg_idx), limit) :]] if len(neg_idx) else neg_idx

    buckets = []
    per_bucket = max(limit // 4, 1)
    if len(pos_idx):
        buckets.append(rng.choice(pos_idx, size=min(len(pos_idx), limit // 2), replace=False))
    if len(hard_idx):
        buckets.append(rng.choice(hard_idx, size=min(len(hard_idx), per_bucket), replace=False))
    if len(high_conf_neg):
        buckets.append(rng.choice(high_conf_neg, size=min(len(high_conf_neg), per_bucket), replace=False))
    if len(neg_idx):
        buckets.append(rng.choice(neg_idx, size=min(len(neg_idx), per_bucket), replace=False))
    if not buckets:
        return np.empty((0, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    idx = np.unique(np.concatenate(buckets))
    if len(idx) > limit:
        idx = rng.choice(idx, size=limit, replace=False)
    return features[idx], labels[idx]


def choose_group_views(scene_lookup, group_row):
    scene_dir = scene_lookup.get(group_row["scene"])
    if scene_dir is None:
        raise FileNotFoundError(f"Scene {group_row['scene']} not found in posed_images")
    num_views = as_int(group_row, "num_views")
    policy = group_row.get("view_policy")
    return base.choose_views(scene_dir, num_views, policy, seed=RUN19_SEED + num_views)


def run_group(backbone, root, scene_lookup, group_row, out_dir):
    view_files = choose_group_views(scene_lookup, group_row)
    print("Run 23 group views:", {"group": group_row.get("group_key"), "views": [p.name for p in view_files]})
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
    features = build_features(points, conf, xs, ys, view_ids, image_h, image_w, group_row)
    labels, distances, hard_negative = candidate_labels(points, gt)
    del output
    torch.cuda.empty_cache()
    return {
        "points": points,
        "conf": conf,
        "features": features,
        "labels": labels,
        "distances": distances,
        "hard_negative": hard_negative,
        "gt": gt,
        "runtime_seconds": runtime,
        "view_files": view_files,
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
        "priority_score": as_float(group_row, "priority_score"),
        "num_candidates": int(len(labels)),
        "candidate_positive_ratio": float(labels.mean()) if len(labels) else 0.0,
        "hard_negative_ratio": float(hard.mean()) if len(hard) else 0.0,
        "mean_distance_to_gt_m": float(distances.mean()) if len(distances) else 0.0,
        "median_distance_to_gt_m": float(np.median(distances)) if len(distances) else 0.0,
        "mean_conf": float(conf.mean()) if len(conf) else 0.0,
        "runtime_seconds": float(pack["runtime_seconds"]),
    }


def safe_device():
    if torch.cuda.is_available():
        try:
            major, _minor = torch.cuda.get_device_capability(0)
            if major >= 7:
                return "cuda"
        except Exception as exc:
            print("CUDA compatibility check failed for Run 23 MLP; using CPU:", repr(exc))
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


def best_label_threshold(prob, labels):
    best = {"threshold": 0.5, "label_f1": -1.0}
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = candidate_label_metrics(prob, labels, threshold)
        if metrics["label_f1"] > best["label_f1"]:
            best = {"threshold": float(threshold), **metrics}
    return best


def train_model(x_all, y_all):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y_all))
    split = int(0.80 * len(order))
    train_idx, val_idx = order[:split], order[split:]
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]
    device = safe_device()
    model = CandidateReliabilityHead(x_all.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    x_val_t = torch.from_numpy(x_val).to(device)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(y_train))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = torch.from_numpy(order[start : start + BATCH_SIZE]).long().to(device)
            logits = model(x_train_t[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_train_t[idx], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(x_val_t)).detach().cpu().numpy()
        best = best_label_threshold(prob, y_val)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **best}
        history.append(row)
        print("Run 23 train row:", row)
    return model, history


def predict_prob(model, features):
    device = next(model.parameters()).device
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE * 4):
            xb = torch.from_numpy(features[start : start + BATCH_SIZE * 4]).to(device)
            chunks.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(chunks)


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
        "priority_score": as_float(group_row, "priority_score"),
        "selected_ratio": float(mask.mean()) if len(mask) else 0.0,
        "fallback_all_points": fallback,
        **extra,
        **metrics,
    }


def evaluate_group(model, backbone, root, scene_lookup, group_row, out_dir, label_threshold):
    pack = run_group(backbone, root, scene_lookup, group_row, out_dir)
    prob = predict_prob(model, pack["features"])
    label_metrics = candidate_label_metrics(prob, pack["labels"], label_threshold)
    num_views = as_int(group_row, "num_views")
    final_percent = base.FIXED_FINAL_CONF_BY_VIEW_COUNT.get(num_views, base.CONF_PERCENT)
    conf_final, conf_threshold = confidence_mask(pack["conf"], final_percent)
    rows = [
        score_selection(
            pack["points"],
            conf_final,
            pack["gt"],
            "confidence_fixed_final",
            group_row,
            {
                "runtime_seconds": pack["runtime_seconds"],
                "conf_percent": final_percent,
                "conf_threshold": conf_threshold,
                "rcrh_ratio": "",
                "rcrh_threshold": "",
                "rcrh_score_threshold": "",
                "mean_keep_prob": float(prob.mean()),
                **label_metrics,
            },
        )
    ]
    for ratio in RANK_RATIOS:
        mask, score_threshold = ratio_mask(prob, ratio)
        rows.append(
            score_selection(
                pack["points"],
                mask,
                pack["gt"],
                f"rcrh_top_ratio_{ratio:.3f}",
                group_row,
                {
                    "runtime_seconds": pack["runtime_seconds"],
                    "conf_percent": "",
                    "conf_threshold": "",
                    "rcrh_ratio": ratio,
                    "rcrh_threshold": "",
                    "rcrh_score_threshold": score_threshold,
                    "mean_keep_prob": float(prob.mean()),
                    **label_metrics,
                },
            )
        )
    matched_ratio = float(conf_final.mean())
    matched_mask, matched_threshold = ratio_mask(prob, matched_ratio)
    rows.append(
        score_selection(
            pack["points"],
            matched_mask,
            pack["gt"],
            "rcrh_match_confidence_ratio",
            group_row,
            {
                "runtime_seconds": pack["runtime_seconds"],
                "conf_percent": final_percent,
                "conf_threshold": conf_threshold,
                "rcrh_ratio": matched_ratio,
                "rcrh_threshold": "",
                "rcrh_score_threshold": matched_threshold,
                "mean_keep_prob": float(prob.mean()),
                **label_metrics,
            },
        )
    )
    threshold_mask = prob >= label_threshold
    rows.append(
        score_selection(
            pack["points"],
            threshold_mask,
            pack["gt"],
            "rcrh_label_f1_threshold",
            group_row,
            {
                "runtime_seconds": pack["runtime_seconds"],
                "conf_percent": "",
                "conf_threshold": "",
                "rcrh_ratio": "",
                "rcrh_threshold": label_threshold,
                "rcrh_score_threshold": label_threshold,
                "mean_keep_prob": float(prob.mean()),
                **label_metrics,
            },
        )
    )
    guard_mask = threshold_mask & conf_final
    rows.append(
        score_selection(
            pack["points"],
            guard_mask,
            pack["gt"],
            "rcrh_label_threshold_and_confidence_guard",
            group_row,
            {
                "runtime_seconds": pack["runtime_seconds"],
                "conf_percent": final_percent,
                "conf_threshold": conf_threshold,
                "rcrh_ratio": "",
                "rcrh_threshold": label_threshold,
                "rcrh_score_threshold": label_threshold,
                "mean_keep_prob": float(prob.mean()),
                **label_metrics,
            },
        )
    )
    for row in rows:
        print("Run 23 metric row:", row)
    return rows, label_summary_row(group_row, pack, "eval")


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
                "mean_label_f1": float(np.mean([r["label_f1"] for r in items])) if items and "label_f1" in items[0] else 0.0,
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
                "Use reconstruction-candidate reliability calibration in the next full test run."
                if selected != "confidence_fixed_final"
                else "Keep fixed confidence until reconstruction-candidate calibration wins validation."
            ),
        }
    ]


def split_group_selection(group_manifest, final_manifest):
    train = [
        r
        for r in group_manifest
        if r.get("split") == "train"
        and any(c in row_classes(r) for c in ["occlusion_core", "occlusion_borderline", "low_overlap_far", "wrong_depth_hard_negative"])
    ]
    train = sorted(train, key=lambda r: (-as_float(r, "priority_score"), r.get("scene", ""), as_int(r, "num_views"), r.get("view_policy", "")))
    eval_groups = [r for r in final_manifest if str(r.get("is_final_eval_candidate", "1")) in {"1", "True", "true"}]
    eval_groups = sorted(eval_groups, key=lambda r: (r.get("split") != "val", -as_float(r, "priority_score"), r.get("scene", ""), as_int(r, "num_views")))
    return train[:MAX_TRAIN_GROUPS], eval_groups[:MAX_EVAL_GROUPS]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    base.require_t4x2()
    run20_dir = find_run20_dir()
    group_manifest = read_csv(run20_dir / "subset_group_manifest.csv")
    final_manifest = read_csv(run20_dir / "final_eval_group_manifest.csv")
    train_groups, eval_groups = split_group_selection(group_manifest, final_manifest)
    print("Run 23 train groups:", len(train_groups))
    for row in train_groups:
        print("Run 23 selected train group:", row)
    print("Run 23 eval groups:", len(eval_groups))
    for row in eval_groups:
        print("Run 23 selected eval group:", row)

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

    train_x_parts = []
    train_y_parts = []
    label_summary_rows = []
    for group in train_groups:
        pack = run_group(backbone, root, scene_lookup, group, out_dir / "train_groups" / group["group_key"])
        x_part, y_part = sample_training_rows(
            pack["features"],
            pack["labels"],
            pack["hard_negative"],
            pack["conf"],
            MAX_TRAIN_ROWS_PER_GROUP,
            stable_seed("train-" + group.get("group_key", "")),
        )
        train_x_parts.append(x_part)
        train_y_parts.append(y_part)
        label_summary_rows.append(label_summary_row(group, pack, "train"))
        print(
            "Run 23 train sample:",
            {
                "group": group.get("group_key"),
                "rows": len(y_part),
                "positive_ratio": float(y_part.mean()) if len(y_part) else 0.0,
            },
        )
    x_train = np.concatenate(train_x_parts, axis=0)
    y_train = np.concatenate(train_y_parts, axis=0)
    if len(y_train) < 1000 or len(np.unique(y_train)) < 2:
        raise RuntimeError(f"Insufficient Run 23 training rows: rows={len(y_train)} positives={float(y_train.sum())}")
    print("Run 23 training matrix:", {"rows": len(y_train), "feature_dim": x_train.shape[1], "positive_ratio": float(y_train.mean())})

    model, history = train_model(x_train, y_train)
    label_threshold = float(history[-1]["threshold"])
    metric_rows = []
    for group in eval_groups:
        rows, label_row = evaluate_group(
            model,
            backbone,
            root,
            scene_lookup,
            group,
            out_dir / "eval_groups" / group["group_key"],
            label_threshold,
        )
        metric_rows.extend(rows)
        label_summary_rows.append(label_row)

    summary_rows = summarize(metric_rows)
    gate_rows = make_gate_decision(summary_rows)
    write_csv_union(out_dir / "candidate_label_summary.csv", label_summary_rows)
    write_csv_union(out_dir / "training_history.csv", history)
    write_csv_union(out_dir / "metrics.csv", metric_rows)
    write_csv_union(out_dir / "summary.csv", summary_rows)
    write_csv_union(out_dir / "gate_decision.csv", gate_rows)
    write_csv_union(out_dir / "selected_train_groups.csv", train_groups)
    write_csv_union(out_dir / "selected_eval_groups.csv", eval_groups)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": int(x_train.shape[1]),
            "feature_names": FEATURE_NAMES,
            "label_threshold": label_threshold,
            "rank_ratios": RANK_RATIOS,
        },
        out_dir / "rcrh_candidate_head.pt",
    )
    config = {
        "run": RUN_NAME,
        "source_run20_dir": str(run20_dir),
        "num_train_groups": len(train_groups),
        "num_eval_groups": len(eval_groups),
        "max_train_groups": MAX_TRAIN_GROUPS,
        "max_eval_groups": MAX_EVAL_GROUPS,
        "max_candidates_per_group": MAX_CANDIDATES_PER_GROUP,
        "max_train_rows_per_group": MAX_TRAIN_ROWS_PER_GROUP,
        "num_train_rows": int(len(y_train)),
        "feature_dim": int(x_train.shape[1]),
        "feature_names": FEATURE_NAMES,
        "label_positive_threshold_m": LABEL_POS_M,
        "hard_negative_threshold_m": HARD_NEG_M,
        "rank_ratios": RANK_RATIOS,
        "label_threshold_from_internal_val": label_threshold,
        "gate_margin_f1": GATE_MARGIN_F1,
        "runtime_seconds": time.time() - started,
        "note": (
            "Run 23 responds to the Run 22 failure by training on actual MV-DUSt3R "
            "reconstruction candidates instead of the Run 20 proxy candidate rows. "
            "Training labels use GT geometry, but inference features use only prediction "
            "confidence, point coordinates, view metadata, and cross-view support among "
            "predicted candidates. Reconstruction policy is selected by validation F-score."
        ),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    print("Run 23 config:")
    print(config)
    print("Run 23 summary:")
    for row in summary_rows:
        print(row)
    print("Run 23 gate decision:")
    for row in gate_rows:
        print(row)


if __name__ == "__main__":
    main()
