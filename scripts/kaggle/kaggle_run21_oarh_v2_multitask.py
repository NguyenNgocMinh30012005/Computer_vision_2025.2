import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


RUN_NAME = "run_21_oarh_v2_multitask"
SEED = 2121
EPOCHS = 24
BATCH_SIZE = 8192
LR = 1e-3
WEIGHT_VIS = 0.50
WEIGHT_DEPTH = 0.20


VIS_LABELS = {
    "visible_consistent": 0,
    "occluded_behind_observed_surface": 1,
    "floating_in_front_of_observed_surface": 2,
}
VIS_NAMES = {v: k for k, v in VIS_LABELS.items()}


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
    candidates = sorted(root.rglob("oarh_v2_balanced_labels.csv"))
    for path in candidates:
        text = str(path).lower()
        if "run-20" in text or "run_20" in text or "subset-mining" in text:
            run_dir = path.parent
            print("Using Run 20 output:", run_dir)
            return run_dir
    if candidates:
        print("Using first Run 20-like output:", candidates[0].parent)
        return candidates[0].parent
    raise FileNotFoundError(
        "Cannot find Run 20 oarh_v2_balanced_labels.csv. Add the Run 20 kernel output as a Kaggle kernel source."
    )


def as_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", "nan", "NaN", None):
        return default
    try:
        x = float(value)
    except ValueError:
        return default
    return x if math.isfinite(x) else default


def raw_float(row, key):
    value = row.get(key, "")
    if value in ("", "nan", "NaN", None):
        return math.nan
    try:
        x = float(value)
    except ValueError:
        return math.nan
    return x if math.isfinite(x) else math.nan


def feature_row(row):
    group_classes = row.get("group_classes", "")
    policy = row.get("view_policy", "")
    num_views = as_float(row, "num_views", 0.0)
    depth_residual = as_float(row, "depth_residual_m", 0.0)
    observed = as_float(row, "observed_target_depth_m", 0.0)
    projected = as_float(row, "projected_target_depth_m", 0.0)
    candidate_depth = as_float(row, "candidate_depth_m", 0.0)
    src_depth = as_float(row, "src_depth_m", 0.0)
    baseline = as_float(row, "baseline_m", 0.0)

    feats = [
        as_float(row, "src_x_norm"),
        as_float(row, "src_y_norm"),
        as_float(row, "target_x_norm"),
        as_float(row, "target_y_norm"),
        math.log1p(max(src_depth, 0.0)),
        math.log1p(max(candidate_depth, 0.0)),
        as_float(row, "candidate_depth_delta_m"),
        math.log1p(max(projected, 0.0)),
        math.log1p(max(observed, 0.0)),
        depth_residual,
        abs(depth_residual),
        as_float(row, "target_in_bounds"),
        min(baseline, 5.0) / 5.0,
        num_views / 5.0,
        as_float(row, "group_priority_score"),
        1.0 if policy == "hybrid" else 0.0,
        1.0 if policy == "diversity_aware" else 0.0,
        1.0 if "occlusion_core" in group_classes else 0.0,
        1.0 if "occlusion_borderline" in group_classes else 0.0,
        1.0 if "low_overlap_far" in group_classes else 0.0,
        1.0 if "wrong_depth_hard_negative" in group_classes else 0.0,
    ]
    return feats


def make_arrays(rows):
    x, keep, vis, residual, residual_mask, meta = [], [], [], [], [], []
    for row in rows:
        visibility = row.get("visibility_label", "")
        if visibility not in VIS_LABELS:
            if as_float(row, "floating_label") > 0.5 or as_float(row, "keep_label") < 0.5:
                visibility = "floating_in_front_of_observed_surface"
            else:
                continue
        x.append(feature_row(row))
        keep.append(as_float(row, "keep_label"))
        vis.append(VIS_LABELS[visibility])
        r = raw_float(row, "depth_residual_m")
        residual.append(float(np.clip(r if math.isfinite(r) else 0.0, -1.0, 1.0)))
        residual_mask.append(1.0 if math.isfinite(r) else 0.0)
        meta.append(
            {
                "split": row.get("split"),
                "scene": row.get("scene"),
                "num_views": int(float(row.get("num_views", 0))),
                "view_policy": row.get("view_policy"),
                "group_key": row.get("group_key"),
                "sample_bucket": row.get("sample_bucket"),
                "group_classes": row.get("group_classes"),
            }
        )
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(keep, dtype=np.float32),
        np.asarray(vis, dtype=np.int64),
        np.asarray(residual, dtype=np.float32),
        np.asarray(residual_mask, dtype=np.float32),
        meta,
    )


class OARHv2(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(160, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.ReLU(inplace=True),
        )
        self.keep = nn.Linear(48, 1)
        self.visibility = nn.Linear(48, len(VIS_LABELS))
        self.depth = nn.Linear(48, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.keep(h).squeeze(-1), self.visibility(h), self.depth(h).squeeze(-1)


def metrics_from_predictions(keep_prob, keep_true, vis_pred, vis_true, residual_pred, residual_true, residual_mask, threshold):
    keep_pred = keep_prob >= threshold
    tp = float(((keep_pred == 1) & (keep_true == 1)).sum())
    fp = float(((keep_pred == 1) & (keep_true == 0)).sum())
    fn = float(((keep_pred == 0) & (keep_true == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    keep_f1 = 2 * precision * recall / (precision + recall + 1e-8)
    out = {
        "keep_threshold": float(threshold),
        "keep_precision": float(precision),
        "keep_recall": float(recall),
        "keep_f1": float(keep_f1),
        "visibility_accuracy": float((vis_pred == vis_true).mean()) if len(vis_true) else 0.0,
    }
    for cls, name in VIS_NAMES.items():
        pred = vis_pred == cls
        true = vis_true == cls
        ctp = float((pred & true).sum())
        cfp = float((pred & ~true).sum())
        cfn = float((~pred & true).sum())
        cp = ctp / (ctp + cfp + 1e-8)
        cr = ctp / (ctp + cfn + 1e-8)
        cf1 = 2 * cp * cr / (cp + cr + 1e-8)
        short = name.split("_")[0]
        out[f"{short}_precision"] = float(cp)
        out[f"{short}_recall"] = float(cr)
        out[f"{short}_f1"] = float(cf1)
    mask = residual_mask > 0.5
    out["depth_residual_mae"] = float(np.abs(residual_pred[mask] - residual_true[mask]).mean()) if mask.any() else 0.0
    return out


def best_threshold(keep_prob, keep_true):
    best = {"threshold": 0.5, "keep_f1": -1.0}
    for threshold in np.linspace(0.05, 0.95, 19):
        m = metrics_from_predictions(
            keep_prob,
            keep_true,
            np.zeros_like(keep_true, dtype=np.int64),
            np.zeros_like(keep_true, dtype=np.int64),
            np.zeros_like(keep_true),
            np.zeros_like(keep_true),
            np.zeros_like(keep_true),
            threshold,
        )
        if m["keep_f1"] > best["keep_f1"]:
            best = {"threshold": float(threshold), "keep_f1": m["keep_f1"]}
    return best


def safe_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        major, _minor = torch.cuda.get_device_capability(0)
        if major >= 7:
            return "cuda"
    except Exception as exc:
        print("CUDA compatibility check failed; using CPU:", repr(exc))
    print("CUDA device is not compatible with this PyTorch build; using CPU for Run 21 MLP training.")
    return "cpu"


def train_model(train, val):
    x_train, y_train, vis_train, res_train, res_mask_train, _ = train
    x_val, y_val, vis_val, res_val, res_mask_val, _ = val
    device = safe_device()
    model = OARHv2(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    class_counts = np.bincount(vis_train, minlength=len(VIS_LABELS)).astype(np.float32)
    class_weights = torch.from_numpy(class_counts.sum() / np.maximum(class_counts, 1.0)).float().to(device)

    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    vis_train_t = torch.from_numpy(vis_train).to(device)
    res_train_t = torch.from_numpy(res_train).to(device)
    res_mask_train_t = torch.from_numpy(res_mask_train).to(device)
    x_val_t = torch.from_numpy(x_val).to(device)

    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(y_train))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = torch.from_numpy(order[start : start + BATCH_SIZE]).long().to(device)
            keep_logits, vis_logits, res_pred = model(x_train_t[idx])
            keep_loss = F.binary_cross_entropy_with_logits(keep_logits, y_train_t[idx], pos_weight=pos_weight)
            vis_loss = F.cross_entropy(vis_logits, vis_train_t[idx], weight=class_weights)
            mask = res_mask_train_t[idx]
            depth_loss = (F.smooth_l1_loss(res_pred, res_train_t[idx], reduction="none") * mask).sum() / mask.sum().clamp_min(1.0)
            loss = keep_loss + WEIGHT_VIS * vis_loss + WEIGHT_DEPTH * depth_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            keep_logits, vis_logits, res_pred = model(x_val_t)
            keep_prob = torch.sigmoid(keep_logits).cpu().numpy()
            vis_pred = vis_logits.argmax(dim=1).cpu().numpy()
            res_np = res_pred.cpu().numpy()
        threshold = best_threshold(keep_prob, y_val)["threshold"]
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **metrics_from_predictions(keep_prob, y_val, vis_pred, vis_val, res_np, res_val, res_mask_val, threshold),
        }
        history.append(row)
        print("Run 21 train row:", row)
    return model, history


def predict(model, arrays):
    x, y, vis, res, res_mask, meta = arrays
    device = next(model.parameters()).device
    model.eval()
    keep_probs, vis_preds, res_preds = [], [], []
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE * 4):
            xb = torch.from_numpy(x[start : start + BATCH_SIZE * 4]).to(device)
            keep_logits, vis_logits, res_pred = model(xb)
            keep_probs.append(torch.sigmoid(keep_logits).cpu().numpy())
            vis_preds.append(vis_logits.argmax(dim=1).cpu().numpy())
            res_preds.append(res_pred.cpu().numpy())
    return np.concatenate(keep_probs), np.concatenate(vis_preds), np.concatenate(res_preds), y, vis, res, res_mask, meta


def grouped_metrics(pred_pack, threshold, prefix):
    keep_prob, vis_pred, res_pred, keep_true, vis_true, res_true, res_mask, meta = pred_pack
    rows = []
    keys = sorted({(m["split"], m["scene"], m["num_views"], m["view_policy"], m["group_key"]) for m in meta})
    for key in keys:
        idx = np.array(
            [
                i
                for i, m in enumerate(meta)
                if (m["split"], m["scene"], m["num_views"], m["view_policy"], m["group_key"]) == key
            ],
            dtype=int,
        )
        if len(idx) == 0:
            continue
        split, scene, num_views, policy, group_key = key
        rows.append(
            {
                "run": RUN_NAME,
                "table": prefix,
                "split": split,
                "scene": scene,
                "num_views": num_views,
                "view_policy": policy,
                "group_key": group_key,
                "num_rows": int(len(idx)),
                **metrics_from_predictions(
                    keep_prob[idx],
                    keep_true[idx],
                    vis_pred[idx],
                    vis_true[idx],
                    res_pred[idx],
                    res_true[idx],
                    res_mask[idx],
                    threshold,
                ),
            }
        )
    return rows


def split_metrics(pred_pack, threshold):
    keep_prob, vis_pred, res_pred, keep_true, vis_true, res_true, res_mask, meta = pred_pack
    rows = []
    for split in sorted({m["split"] for m in meta}):
        idx = np.array([i for i, m in enumerate(meta) if m["split"] == split], dtype=int)
        rows.append(
            {
                "run": RUN_NAME,
                "split": split,
                "num_rows": int(len(idx)),
                **metrics_from_predictions(
                    keep_prob[idx],
                    keep_true[idx],
                    vis_pred[idx],
                    vis_true[idx],
                    res_pred[idx],
                    res_true[idx],
                    res_mask[idx],
                    threshold,
                ),
            }
        )
    return rows


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    run20_dir = find_run20_dir()
    rows = read_csv(run20_dir / "oarh_v2_balanced_labels.csv")
    final_groups = read_csv(run20_dir / "final_eval_group_manifest.csv") if (run20_dir / "final_eval_group_manifest.csv").exists() else []
    print("Run 21 loaded OARH rows:", len(rows))
    print("Run 21 final eval groups:", len(final_groups))

    train_rows = [r for r in rows if r.get("split") == "train"]
    val_rows = [r for r in rows if r.get("split") == "val"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    if len(train_rows) < 1000 or len(val_rows) < 1000 or len(test_rows) < 1000:
        raise RuntimeError(f"Insufficient rows for Run 21: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    train = make_arrays(train_rows)
    val = make_arrays(val_rows)
    test = make_arrays(test_rows)
    model, history = train_model(train, val)

    val_pred = predict(model, val)
    test_pred = predict(model, test)
    threshold = best_threshold(val_pred[0], val[1])["threshold"]
    split_rows = split_metrics(val_pred, threshold) + split_metrics(test_pred, threshold)
    group_rows = grouped_metrics(val_pred, threshold, "val_groups") + grouped_metrics(test_pred, threshold, "test_groups")

    write_csv_union(out_dir / "training_history.csv", history)
    write_csv_union(out_dir / "split_metrics.csv", split_rows)
    write_csv_union(out_dir / "final_eval_group_metrics.csv", group_rows)
    write_csv_union(out_dir / "final_eval_group_manifest_copy.csv", final_groups)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": int(train[0].shape[1]),
            "keep_threshold": float(threshold),
            "visibility_labels": VIS_LABELS,
        },
        out_dir / "oarh_v2_multitask_head.pt",
    )
    config = {
        "run": RUN_NAME,
        "source_run20_dir": str(run20_dir),
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "num_test_rows": len(test_rows),
        "feature_dim": int(train[0].shape[1]),
        "keep_threshold_from_val": float(threshold),
        "loss_weights": {"visibility": WEIGHT_VIS, "depth": WEIGHT_DEPTH},
        "runtime_seconds": time.time() - started,
        "note": (
            "OARH v2 multitask classifier/regressor trained from Run 20 balanced labels. "
            "Direct target-label leakage features such as candidate_type and visibility_label "
            "are intentionally excluded from the inputs. "
            "This is still a label-cache/proxy training run; reconstruction-level proof comes in Run 22."
        ),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    print("Run 21 config:")
    print(config)
    print("Run 21 split metrics:")
    for row in split_rows:
        print(row)
    print("Run 21 top final group metrics:")
    for row in sorted(group_rows, key=lambda r: (r["split"], -r["occluded_f1"], -r["keep_f1"]))[:32]:
        print(row)


if __name__ == "__main__":
    main()
