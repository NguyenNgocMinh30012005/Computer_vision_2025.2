import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


RUN_NAME = "run_24_rsdh_v2_image_only"
SEED = 2424
EPOCHS = 22
BATCH_SIZE = 8192
LR = 1e-3
PATCH_RADIUS = 12
DOWNSAMPLE = 8
GATE_MARGIN_F1 = 0.01
MAX_ROWS_PER_SPLIT = {
    "train": 90000,
    "val": 90000,
    "test": 90000,
}


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


def find_run20_dir():
    root = Path("/kaggle/input")
    candidates = sorted(root.rglob("rsdh_v2_hard_negative_labels.csv"))
    for path in candidates:
        text = str(path).lower()
        if "run-20" in text or "run_20" in text or "subset-mining" in text:
            print("Using Run 20 output:", path.parent)
            return path.parent
    if candidates:
        print("Using first Run 20-like output:", candidates[0].parent)
        return candidates[0].parent
    raise FileNotFoundError(
        "Cannot find Run 20 rsdh_v2_hard_negative_labels.csv. Add the Run 20 kernel output as a Kaggle kernel source."
    )


def find_posed_images_root():
    root = Path("/kaggle/input")
    candidates = []
    for path in root.rglob("posed_images"):
        if path.is_dir() and any(path.glob("scene*")):
            candidates.append(path)
    for path in sorted(candidates):
        if any(p.is_dir() and list(p.glob("*.jpg")) for p in path.glob("scene*")):
            print("POSED_IMAGES:", path)
            return path
    raise FileNotFoundError("Cannot find ScanNet posed_images directory under /kaggle/input")


def as_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", "nan", "NaN", None):
        return default
    try:
        out = float(value)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def finite_float(row, key):
    value = row.get(key, "")
    if value in ("", "nan", "NaN", None):
        return math.nan
    try:
        out = float(value)
    except ValueError:
        return math.nan
    return out if math.isfinite(out) else math.nan


def row_is_usable(row):
    if row.get("split") not in {"train", "val", "test"}:
        return False
    if not math.isfinite(finite_float(row, "src_x_norm")):
        return False
    if not math.isfinite(finite_float(row, "src_y_norm")):
        return False
    if not math.isfinite(finite_float(row, "target_x_norm")):
        return False
    if not math.isfinite(finite_float(row, "target_y_norm")):
        return False
    return as_float(row, "target_in_bounds") > 0.5


def balanced_cap_rows(rows):
    rng = random.Random(SEED)
    buckets = defaultdict(list)
    for row in rows:
        label = "pos" if as_float(row, "match_label") > 0.5 else "neg"
        hard = "low_overlap_far" if "low_overlap_far" in row.get("sample_bucket", "") else "geometry_hard"
        buckets[(row["split"], label, hard)].append(row)

    out = []
    for split in ["train", "val", "test"]:
        split_rows = [r for key, vals in buckets.items() if key[0] == split for r in vals]
        cap = MAX_ROWS_PER_SPLIT.get(split, len(split_rows))
        if len(split_rows) <= cap:
            out.extend(split_rows)
            continue
        selected = []
        split_keys = [k for k in buckets if k[0] == split]
        per_bucket = max(1, cap // max(1, len(split_keys)))
        leftovers = []
        for key in split_keys:
            vals = buckets[key][:]
            rng.shuffle(vals)
            selected.extend(vals[:per_bucket])
            leftovers.extend(vals[per_bucket:])
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max(0, cap - len(selected))])
        out.extend(selected[:cap])
    return out


def load_image_cache(posed_root):
    cache = {}

    def load(scene, image_name):
        key = (scene, image_name)
        if key in cache:
            return cache[key]
        path = posed_root / scene / image_name
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        cache[key] = {"rgb": rgb, "gray": gray, "grad": grad}
        return cache[key]

    return load


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
    return feats, gray_corr


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
PATCH_CORR_INDEX = FEATURE_NAMES.index("gray_patch_corr")


def make_arrays(rows, posed_root):
    load = load_image_cache(posed_root)
    x = []
    y = []
    sim = []
    meta = []
    skipped = Counter()

    for idx, row in enumerate(rows, start=1):
        if idx % 50000 == 0:
            print("Run 24 featurized rows:", idx, "kept:", len(y))
        scene = row["scene"]
        try:
            src = load(scene, row["source_image"])
            tgt = load(scene, row["target_image"])
            sh, sw = src["gray"].shape
            th, tw = tgt["gray"].shape
            sxn = finite_float(row, "src_x_norm")
            syn = finite_float(row, "src_y_norm")
            txn = finite_float(row, "target_x_norm")
            tyn = finite_float(row, "target_y_norm")
            sx = sxn * max(sw - 1, 1)
            sy = syn * max(sh - 1, 1)
            tx = txn * max(tw - 1, 1)
            ty = tyn * max(th - 1, 1)
            patch_feats, patch_corr = patch_features(src, tgt, (sx, sy), (tx, ty))
        except Exception as exc:
            skipped[type(exc).__name__] += 1
            continue

        policy = row.get("view_policy", "")
        base_feats = [
            sxn,
            syn,
            txn,
            tyn,
            txn - sxn,
            tyn - syn,
            float(math.sqrt((txn - sxn) ** 2 + (tyn - syn) ** 2)),
            as_float(row, "num_views") / 5.0,
            1.0 if policy == "hybrid" else 0.0,
            1.0 if policy == "diversity_aware" else 0.0,
        ]
        x.append(base_feats + patch_feats)
        y.append(1.0 if as_float(row, "match_label") > 0.5 else 0.0)
        sim.append(patch_corr)
        meta.append(
            {
                "split": row.get("split"),
                "scene": row.get("scene"),
                "num_views": int(as_float(row, "num_views")),
                "view_policy": row.get("view_policy"),
                "group_key": row.get("group_key"),
                "source_image": row.get("source_image"),
                "target_image": row.get("target_image"),
                "sample_bucket": row.get("sample_bucket"),
                "group_classes": row.get("group_classes"),
                "hard_class": "low_overlap_far" if "low_overlap_far" in row.get("sample_bucket", "") else "geometry_hard",
            }
        )
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(sim, dtype=np.float32),
        meta,
        skipped,
    )


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


def binary_metrics(scores, labels, threshold, greater_is_positive=True):
    pred = scores >= threshold if greater_is_positive else scores <= threshold
    y = labels > 0.5
    tp = float((pred & y).sum())
    fp = float((pred & ~y).sum())
    fn = float((~pred & y).sum())
    tn = float((~pred & ~y).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / max(1.0, tp + fp + fn + tn)),
        "selected_ratio": float(pred.mean()) if len(pred) else 0.0,
        "positive_ratio": float(y.mean()) if len(y) else 0.0,
    }


def best_threshold(scores, labels):
    if len(scores) == 0:
        return {"threshold": 0.5, "f1": 0.0}
    lo = float(np.nanpercentile(scores, 1))
    hi = float(np.nanpercentile(scores, 99))
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        lo, hi = 0.0, 1.0
    best = {"threshold": lo, "f1": -1.0}
    for threshold in np.linspace(lo, hi, 41):
        metrics = binary_metrics(scores, labels, threshold)
        if metrics["f1"] > best["f1"]:
            best = metrics
    return best


def all_positive_metrics(labels):
    scores = np.ones_like(labels, dtype=np.float32)
    return binary_metrics(scores, labels, 0.0)


def standardize(train_x, *others):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    arrays = [(train_x - mean) / std]
    arrays.extend((arr - mean) / std for arr in others)
    return mean.astype(np.float32), std.astype(np.float32), arrays


def train_model(x_train, y_train, x_val, y_val):
    device = "cpu"
    model = RSDHv2(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    x_train_t = torch.from_numpy(x_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    x_val_t = torch.from_numpy(x_val).float().to(device)
    rng = np.random.default_rng(SEED)
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
            val_prob = torch.sigmoid(model(x_val_t)).detach().cpu().numpy()
        best = best_threshold(val_prob, y_val)
        row = {
            "run": RUN_NAME,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_threshold": best["threshold"],
            "val_precision": best["precision"],
            "val_recall": best["recall"],
            "val_f1": best["f1"],
            "val_selected_ratio": best["selected_ratio"],
        }
        history.append(row)
        print("Run 24 train row:", row)
    return model, history


def predict(model, x):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            batch = torch.from_numpy(x[start : start + BATCH_SIZE]).float()
            out.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.float32)


def split_summary(rows, labels):
    counts = Counter()
    positives = Counter()
    for row, label in zip(rows, labels):
        split = row["split"]
        counts[split] += 1
        positives[split] += int(label > 0.5)
    return [
        {
            "run": RUN_NAME,
            "split": split,
            "num_rows": counts[split],
            "positive_rows": positives[split],
            "positive_ratio": float(positives[split] / max(1, counts[split])),
        }
        for split in sorted(counts)
    ]


def make_metric_row(split, method, labels, scores, threshold):
    m = binary_metrics(scores, labels, threshold)
    return {
        "run": RUN_NAME,
        "split": split,
        "method": method,
        "num_rows": int(len(labels)),
        "threshold": m["threshold"],
        "match_precision": m["precision"],
        "match_recall": m["recall"],
        "match_f1": m["f1"],
        "accuracy": m["accuracy"],
        "selected_ratio": m["selected_ratio"],
        "positive_ratio": m["positive_ratio"],
    }


def grouped_metrics(method, labels, scores, meta, threshold):
    groups = defaultdict(list)
    for idx, row in enumerate(meta):
        if row["split"] not in {"val", "test"}:
            continue
        key = (row["split"], row["scene"], row["num_views"], row["view_policy"], row["group_key"], row["hard_class"])
        groups[key].append(idx)
    rows = []
    for key, idxs in sorted(groups.items()):
        split, scene, num_views, policy, group_key, hard_class = key
        idx = np.asarray(idxs, dtype=np.int64)
        m = binary_metrics(scores[idx], labels[idx], threshold)
        rows.append(
            {
                "run": RUN_NAME,
                "method": method,
                "split": split,
                "scene": scene,
                "num_views": num_views,
                "view_policy": policy,
                "group_key": group_key,
                "hard_class": hard_class,
                "num_rows": int(len(idx)),
                "threshold": m["threshold"],
                "match_precision": m["precision"],
                "match_recall": m["recall"],
                "match_f1": m["f1"],
                "accuracy": m["accuracy"],
                "selected_ratio": m["selected_ratio"],
                "positive_ratio": m["positive_ratio"],
            }
        )
    return rows


def main():
    started = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    run20_dir = find_run20_dir()
    posed_root = find_posed_images_root()
    raw_rows = [row for row in read_csv(run20_dir / "rsdh_v2_hard_negative_labels.csv") if row_is_usable(row)]
    rows = balanced_cap_rows(raw_rows)
    print("Run 24 loaded usable RSDH rows:", {"raw": len(raw_rows), "after_cap": len(rows)})

    x, y, sim, meta, skipped = make_arrays(rows, posed_root)
    if len(y) < 1000 or float(y.sum()) < 100.0:
        raise RuntimeError(f"Insufficient Run 24 rows after feature extraction: rows={len(y)} positives={float(y.sum())}")

    splits = np.asarray([m["split"] for m in meta])
    train_idx = np.where(splits == "train")[0]
    val_idx = np.where(splits == "val")[0]
    test_idx = np.where(splits == "test")[0]
    if len(train_idx) < 1000 or len(val_idx) < 1000 or len(test_idx) < 1000:
        raise RuntimeError(f"Bad split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    mean, std, scaled = standardize(x[train_idx], x[val_idx], x[test_idx])
    x_train, x_val, x_test = scaled
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    sim_train, sim_val, sim_test = sim[train_idx], sim[val_idx], sim[test_idx]
    meta_train = [meta[i] for i in train_idx]
    meta_val = [meta[i] for i in val_idx]
    meta_test = [meta[i] for i in test_idx]

    print(
        "Run 24 training matrix:",
        {
            "train_rows": len(y_train),
            "val_rows": len(y_val),
            "test_rows": len(y_test),
            "feature_dim": x.shape[1],
            "train_positive_ratio": float(y_train.mean()),
            "val_positive_ratio": float(y_val.mean()),
            "test_positive_ratio": float(y_test.mean()),
            "skipped": dict(skipped),
        },
    )

    model, history = train_model(x_train, y_train, x_val, y_val)
    train_prob = predict(model, x_train)
    val_prob = predict(model, x_val)
    test_prob = predict(model, x_test)
    mlp_threshold = best_threshold(val_prob, y_val)["threshold"]
    patch_threshold = best_threshold(sim_val, y_val)["threshold"]

    metric_rows = []
    for split, labels, probs, sims in [
        ("train", y_train, train_prob, sim_train),
        ("val", y_val, val_prob, sim_val),
        ("test", y_test, test_prob, sim_test),
    ]:
        metric_rows.append(make_metric_row(split, "all_positive", labels, np.ones_like(labels), 0.0))
        metric_rows.append(make_metric_row(split, "patch_similarity_threshold", labels, sims, patch_threshold))
        metric_rows.append(make_metric_row(split, "rsdh_v2_image_only_mlp", labels, probs, mlp_threshold))

    val_lookup = {(r["split"], r["method"]): r for r in metric_rows if r["split"] == "val"}
    learned_val = val_lookup[("val", "rsdh_v2_image_only_mlp")]["match_f1"]
    baseline_method = max(
        ["all_positive", "patch_similarity_threshold"],
        key=lambda method: val_lookup[("val", method)]["match_f1"],
    )
    baseline_val = val_lookup[("val", baseline_method)]["match_f1"]
    selected_method = "rsdh_v2_image_only_mlp" if learned_val >= baseline_val + GATE_MARGIN_F1 else baseline_method
    gate = {
        "run": RUN_NAME,
        "selected_method": selected_method,
        "best_baseline_method": baseline_method,
        "validation_baseline_f1": float(baseline_val),
        "validation_learned_f1": float(learned_val),
        "delta_f1": float(learned_val - baseline_val),
        "gate_margin_f1": GATE_MARGIN_F1,
        "recommendation": (
            "Use RSDH v2 image-only MLP for Run 25 integration."
            if selected_method == "rsdh_v2_image_only_mlp"
            else "Keep RSDH v2 out of reconstruction until image-only match validity beats validation baselines."
        ),
    }

    group_rows = []
    group_rows.extend(grouped_metrics("patch_similarity_threshold", y_val, sim_val, meta_val, patch_threshold))
    group_rows.extend(grouped_metrics("patch_similarity_threshold", y_test, sim_test, meta_test, patch_threshold))
    group_rows.extend(grouped_metrics("rsdh_v2_image_only_mlp", y_val, val_prob, meta_val, mlp_threshold))
    group_rows.extend(grouped_metrics("rsdh_v2_image_only_mlp", y_test, test_prob, meta_test, mlp_threshold))

    summary_rows = split_summary(meta, y)
    write_csv_union(out_dir / "split_metrics.csv", metric_rows)
    write_csv_union(out_dir / "group_metrics.csv", group_rows)
    write_csv_union(out_dir / "training_history.csv", history)
    write_csv_union(out_dir / "feature_summary.csv", summary_rows)
    write_csv_union(out_dir / "gate_decision.csv", [gate])
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "feature_mean": mean,
            "feature_std": std,
            "threshold": float(mlp_threshold),
            "patch_similarity_threshold": float(patch_threshold),
        },
        out_dir / "rsdh_v2_image_only_head.pt",
    )
    config = {
        "run": RUN_NAME,
        "source_run20_dir": str(run20_dir),
        "num_raw_rows": len(raw_rows),
        "num_rows": len(y),
        "num_train_rows": int(len(y_train)),
        "num_val_rows": int(len(y_val)),
        "num_test_rows": int(len(y_test)),
        "feature_dim": int(x.shape[1]),
        "feature_names": FEATURE_NAMES,
        "patch_radius": PATCH_RADIUS,
        "downsample": DOWNSAMPLE,
        "epochs": EPOCHS,
        "mlp_threshold_from_val": float(mlp_threshold),
        "patch_similarity_threshold_from_val": float(patch_threshold),
        "gate_margin_f1": GATE_MARGIN_F1,
        "runtime_seconds": time.time() - started,
        "note": (
            "Run 24 trains RSDH v2 from image-only patch and coordinate features using Run 20 hard-negative labels. "
            "No GT-depth residual, candidate_type, visibility_label, or group class is used as an inference feature. "
            "Run 25 should integrate this head only if the validation gate selects the learned MLP."
        ),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    print("Run 24 config:")
    print(config)
    print("Run 24 split metrics:")
    for row in metric_rows:
        print(row)
    print("Run 24 top group metrics:")
    for row in sorted(group_rows, key=lambda r: (r["split"], r["method"], r["match_f1"]))[:24]:
        print(row)
    print("Run 24 gate decision:")
    print(gate)


if __name__ == "__main__":
    main()
