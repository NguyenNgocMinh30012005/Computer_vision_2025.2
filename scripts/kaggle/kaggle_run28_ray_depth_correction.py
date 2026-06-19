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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


RAW_RUN27 = "https://raw.githubusercontent.com/NguyenNgocMinh30012005/Computer_vision_2025.2/main/scripts/kaggle/kaggle_run27_joint_candidate_acceptance.py"


def ensure_helper_module(module_name, raw_url):
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        helper_path = Path(f"{module_name}.py")
        print(f"Downloading helper module {module_name} from GitHub raw")
        urllib.request.urlretrieve(raw_url, helper_path)
        sys.path.insert(0, str(Path.cwd()))
        return __import__(module_name)


r27 = ensure_helper_module("kaggle_run27_joint_candidate_acceptance", RAW_RUN27)
base = r27.base

RUN_NAME = "run_28_ray_depth_correction"
SEED = 2828
MODEL_SEEDS = [2828, 2829, 2830]
MAX_SCENES = int(os.environ.get("RUN28_MAX_SCENES", "30"))
MAX_TRAIN_GROUPS = int(os.environ.get("RUN28_MAX_TRAIN_GROUPS", "48"))
MAX_EVAL_GROUPS = int(os.environ.get("RUN28_MAX_EVAL_GROUPS", "36"))
MAX_CANDIDATES_PER_GROUP = int(os.environ.get("RUN28_MAX_CANDIDATES_PER_GROUP", "3500"))
EPOCHS = int(os.environ.get("RUN28_EPOCHS", "20"))
LR = float(os.environ.get("RUN28_LR", "0.001"))
MAX_DELTA_NORM = float(os.environ.get("RUN28_MAX_DELTA_NORM", "1.5"))
MIN_DEPTH_M = 0.10
GATE_MARGIN_F1 = float(os.environ.get("RUN28_GATE_MARGIN_F1", "0.005"))
POLICY_ALPHAS = [0.25, 0.50, 0.75, 1.00]
POLICY_TRUST_THRESHOLDS = [0.0, 0.35, 0.50, 0.65]
LOSS_WEIGHTS = {
    "correction": 1.0,
    "trust": 0.20,
    "identity": 0.15,
    "delta": 0.01,
}


class RayDepthCorrectionHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )

    def forward(self, features):
        output = self.net(features)
        delta = MAX_DELTA_NORM * torch.tanh(output[:, :3])
        trust_logit = output[:, 3]
        return delta, trust_logit


def read_depth(path):
    depth = np.asarray(Image.open(path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if depth.size and float(depth.max()) > 100.0:
        depth /= 1000.0
    depth[~np.isfinite(depth)] = 0.0
    return depth


def intrinsics(depth_shape):
    height, width = depth_shape
    return (
        577.870605 * (width / 640.0),
        577.870605 * (height / 480.0),
        319.5 * (width / 640.0),
        239.5 * (height / 480.0),
    )


def source_ray_targets(view_files, xs, ys, view_ids, image_h, image_w):
    poses = [base.parse_pose(str(path).replace(".jpg", ".txt")).astype(np.float32) for path in view_files]
    first_pose_inv = np.linalg.inv(poses[0]).astype(np.float32)
    targets = np.zeros((len(xs), 3), dtype=np.float32)
    valid = np.zeros(len(xs), dtype=bool)
    source_depth = np.zeros(len(xs), dtype=np.float32)

    for view_index, view_file in enumerate(view_files):
        candidate_indices = np.where(view_ids == view_index)[0]
        if not len(candidate_indices):
            continue
        depth = read_depth(str(view_file).replace(".jpg", ".png"))
        height, width = depth.shape
        depth_x = np.clip(
            np.rint(xs[candidate_indices] / max(image_w - 1, 1) * max(width - 1, 1)).astype(np.int32),
            0,
            width - 1,
        )
        depth_y = np.clip(
            np.rint(ys[candidate_indices] / max(image_h - 1, 1) * max(height - 1, 1)).astype(np.int32),
            0,
            height - 1,
        )
        z = depth[depth_y, depth_x]
        usable = np.isfinite(z) & (z > MIN_DEPTH_M)
        if not usable.any():
            continue
        usable_indices = candidate_indices[usable]
        z = z[usable]
        x = depth_x[usable].astype(np.float32)
        y = depth_y[usable].astype(np.float32)
        fx, fy, cx, cy = intrinsics(depth.shape)
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
        source_depth[usable_indices] = z
        valid[usable_indices] = True
    return targets, valid, source_depth


def normalization(points):
    center = np.median(points, axis=0).astype(np.float32)
    radius = np.linalg.norm(points - center[None, :], axis=1)
    scale = float(np.quantile(radius, 0.90)) + 1e-6
    return center, scale


def paired_similarity_targets(points, metric_targets, valid):
    if int(valid.sum()) < 100:
        raise RuntimeError(f"Insufficient valid source-depth targets: {int(valid.sum())}")
    pred_valid = points[valid]
    target_valid = metric_targets[valid]
    pred_center = pred_valid.mean(axis=0, keepdims=True)
    target_center = target_valid.mean(axis=0, keepdims=True)
    pred_scale = float(np.sqrt(np.mean(np.sum((pred_valid - pred_center) ** 2, axis=1)))) + 1e-8
    target_scale = float(np.sqrt(np.mean(np.sum((target_valid - target_center) ** 2, axis=1)))) + 1e-8
    targets_in_prediction = (
        (metric_targets - target_center) / target_scale * pred_scale + pred_center
    ).astype(np.float32)
    predictions_in_metric = (
        (points - pred_center) / pred_scale * target_scale + target_center
    ).astype(np.float32)
    residual_m = np.linalg.norm(predictions_in_metric - metric_targets, axis=1).astype(np.float32)
    return targets_in_prediction, residual_m, {
        "prediction_rms_scale": pred_scale,
        "target_rms_scale_m": target_scale,
        "prediction_to_metric_scale": target_scale / pred_scale,
    }


def run_group(backbone, root, scene_lookup, group_row, out_dir):
    view_files = r27.choose_group_views(scene_lookup, group_row)
    print("Run 28 group views:", {"group": group_row["group_key"], "views": [path.name for path in view_files]})
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
        r27.stable_seed("run28-" + group_row["group_key"]),
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
    metric_targets, valid, source_depth = source_ray_targets(
        view_files,
        xs,
        ys,
        view_ids,
        image_h,
        image_w,
    )
    targets, residual_m, alignment_stats = paired_similarity_targets(
        points,
        metric_targets,
        valid,
    )
    center, scale = normalization(points)
    point_norm = (points - center[None, :]) / scale
    target_norm = (targets - center[None, :]) / scale
    residual_norm = target_norm - point_norm
    low_support = features[:, r27.SUPPORT_FRAC_010_INDEX] < 0.125
    has_projection = features[:, r27.PHOTO_TARGET_COUNT_INDEX] > 0.0
    occlusion_proxy = (valid & low_support & has_projection).astype(np.float32)
    ambiguity_proxy = (valid & (residual_m >= 0.15)).astype(np.float32)
    del output
    torch.cuda.empty_cache()
    return {
        "points": points,
        "conf": conf,
        "features": features,
        "targets": targets,
        "valid": valid,
        "source_depth": source_depth,
        "point_norm": point_norm.astype(np.float32),
        "target_norm": target_norm.astype(np.float32),
        "residual_norm": residual_norm.astype(np.float32),
        "residual_m": residual_m,
        "occlusion_proxy": occlusion_proxy,
        "ambiguity_proxy": ambiguity_proxy,
        "center": center,
        "scale": scale,
        "gt": gt,
        "runtime_seconds": runtime,
        "view_files": view_files,
        "photo_stats": photo_stats,
        "alignment_stats": alignment_stats,
        "group_row": group_row,
    }


def feature_standardization(records):
    features = np.concatenate([record["features"] for record in records], axis=0)
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def correction_loss(model, record, mean_t, std_t):
    device = mean_t.device
    raw = torch.from_numpy(record["features"]).float().to(device)
    valid = torch.from_numpy(record["valid"]).bool().to(device)
    point_norm = torch.from_numpy(record["point_norm"]).float().to(device)
    target_norm = torch.from_numpy(record["target_norm"]).float().to(device)
    residual_m = torch.from_numpy(record["residual_m"]).float().to(device)
    occlusion = torch.from_numpy(record["occlusion_proxy"]).float().to(device)
    ambiguity = torch.from_numpy(record["ambiguity_proxy"]).float().to(device)
    delta, trust_logit = model((raw - mean_t) / std_t)
    trust = torch.sigmoid(trust_logit)
    corrected = point_norm + trust[:, None] * delta
    weight = 1.0 + 2.0 * occlusion + 3.0 * ambiguity

    if valid.any():
        correction_per_point = F.smooth_l1_loss(
            corrected[valid],
            target_norm[valid],
            reduction="none",
            beta=0.05,
        ).sum(dim=1)
        correction = (correction_per_point * weight[valid]).sum() / (weight[valid].sum() + 1e-6)
        trust_target = (residual_m >= 0.05).float()
        trust_loss = F.binary_cross_entropy_with_logits(
            trust_logit[valid],
            trust_target[valid],
            weight=weight[valid],
        )
        accurate = valid & (residual_m < 0.05)
        identity = torch.mean((trust[accurate, None] * delta[accurate]) ** 2) if accurate.any() else delta.new_tensor(0.0)
    else:
        correction = delta.new_tensor(0.0)
        trust_loss = delta.new_tensor(0.0)
        identity = delta.new_tensor(0.0)
    delta_penalty = torch.mean(delta * delta)
    total = (
        LOSS_WEIGHTS["correction"] * correction
        + LOSS_WEIGHTS["trust"] * trust_loss
        + LOSS_WEIGHTS["identity"] * identity
        + LOSS_WEIGHTS["delta"] * delta_penalty
    )
    return total, {
        "correction_loss": float(correction.detach().cpu()),
        "trust_loss": float(trust_loss.detach().cpu()),
        "identity_loss": float(identity.detach().cpu()),
        "delta_loss": float(delta_penalty.detach().cpu()),
    }


def predict_model(model, features, feature_mean, feature_std):
    device = next(model.parameters()).device
    mean_t = torch.from_numpy(feature_mean).float().to(device)
    std_t = torch.from_numpy(feature_std).float().to(device)
    with torch.no_grad():
        raw = torch.from_numpy(features).float().to(device)
        delta, trust_logit = model((raw - mean_t) / std_t)
    return (
        delta.detach().cpu().numpy().astype(np.float32),
        torch.sigmoid(trust_logit).detach().cpu().numpy().astype(np.float32),
    )


def validation_mae(model, records, feature_mean, feature_std):
    errors = []
    for record in records:
        delta, trust = predict_model(model, record["features"], feature_mean, feature_std)
        corrected_norm = record["point_norm"] + trust[:, None] * delta
        valid = record["valid"]
        if valid.any():
            errors.append(
                float(np.linalg.norm(corrected_norm[valid] - record["target_norm"][valid], axis=1).mean())
            )
    return float(np.mean(errors)) if errors else math.inf


def train_model(fit_records, internal_val_records, feature_mean, feature_std, model_seed):
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    rng = np.random.default_rng(model_seed)
    device = r27.safe_device()
    model = RayDepthCorrectionHead(len(r27.FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    mean_t = torch.from_numpy(feature_mean).float().to(device)
    std_t = torch.from_numpy(feature_std).float().to(device)
    history = []
    best_mae = math.inf
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        components = []
        for record_index in rng.permutation(len(fit_records)):
            loss, component = correction_loss(
                model,
                fit_records[int(record_index)],
                mean_t,
                std_t,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            components.append(component)
        model.eval()
        mae = validation_mae(model, internal_val_records, feature_mean, feature_std)
        row = {
            "model_seed": model_seed,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "internal_val_normalized_mae": mae,
            **{
                key: float(np.mean([component[key] for component in components]))
                for key in components[0]
            },
        }
        history.append(row)
        print("Run 28 train row:", row)
        if mae < best_mae:
            best_mae = mae
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    return model, history, {"model_seed": model_seed, "best_internal_val_normalized_mae": best_mae}


def predict_ensemble(models, record, feature_mean, feature_std):
    deltas = []
    trusts = []
    for model in models:
        delta, trust = predict_model(model, record["features"], feature_mean, feature_std)
        deltas.append(delta)
        trusts.append(trust)
    return (
        np.mean(np.stack(deltas, axis=0), axis=0).astype(np.float32),
        np.mean(np.stack(trusts, axis=0), axis=0).astype(np.float32),
    )


def apply_policy(record, delta, trust, alpha, trust_threshold):
    apply = trust >= trust_threshold
    effective_delta = alpha * trust[:, None] * delta * apply[:, None]
    corrected_norm = record["point_norm"] + effective_delta
    return corrected_norm * record["scale"] + record["center"][None, :]


def select_policy(models, records, feature_mean, feature_std):
    rows = []
    predictions = {
        record["group_row"]["group_key"]: predict_ensemble(models, record, feature_mean, feature_std)
        for record in records
    }
    for alpha in POLICY_ALPHAS:
        for trust_threshold in POLICY_TRUST_THRESHOLDS:
            fscores = []
            for record in records:
                delta, trust = predictions[record["group_row"]["group_key"]]
                corrected = apply_policy(record, delta, trust, alpha, trust_threshold)
                fscores.append(base.compute_metrics(base.downsample(corrected, base.MAX_POINTS), record["gt"])["fscore"])
            rows.append(
                {
                    "alpha": alpha,
                    "trust_threshold": trust_threshold,
                    "mean_reconstruction_fscore": float(np.mean(fscores)),
                }
            )
    return max(rows, key=lambda row: row["mean_reconstruction_fscore"]), rows


def method_family(method):
    if method in {"all_candidates", "confidence_fixed_final"}:
        return "baseline"
    if method == "rdch_internal_selected":
        return "learned_gate_candidate"
    return "diagnostic_oracle"


def score_points(points, record, method, extra):
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
        "method_family": method_family(method),
        "occlusion_proxy_ratio": float(record["occlusion_proxy"].mean()),
        "ambiguity_proxy_ratio": float(record["ambiguity_proxy"].mean()),
        "runtime_seconds": record["runtime_seconds"],
        **record["photo_stats"],
        **record["alignment_stats"],
        **extra,
        **metrics,
    }


def confidence_mask(conf, num_views):
    percentile = base.FIXED_FINAL_CONF_BY_VIEW_COUNT.get(num_views, base.CONF_PERCENT)
    threshold = float(np.quantile(conf, percentile / 100.0))
    return conf >= threshold, percentile, threshold


def evaluate_group(models, record, feature_mean, feature_std, selected_policy):
    delta, trust = predict_ensemble(models, record, feature_mean, feature_std)
    corrected = apply_policy(
        record,
        delta,
        trust,
        selected_policy["alpha"],
        selected_policy["trust_threshold"],
    )
    conf_mask, conf_percent, conf_threshold = confidence_mask(
        record["conf"],
        int(record["group_row"]["num_views"]),
    )
    rows = [
        score_points(record["points"], record, "all_candidates", {"selected_ratio": 1.0}),
        score_points(
            record["points"][conf_mask],
            record,
            "confidence_fixed_final",
            {
                "selected_ratio": float(conf_mask.mean()),
                "conf_percent": conf_percent,
                "conf_threshold": conf_threshold,
            },
        ),
        score_points(
            corrected,
            record,
            "rdch_internal_selected",
            {
                "selected_ratio": 1.0,
                "alpha": selected_policy["alpha"],
                "trust_threshold": selected_policy["trust_threshold"],
                "mean_trust": float(trust.mean()),
                "corrected_ratio": float((trust >= selected_policy["trust_threshold"]).mean()),
                "mean_delta_m": float(np.linalg.norm(corrected - record["points"], axis=1).mean()),
            },
        ),
    ]
    oracle = record["points"].copy()
    oracle[record["valid"]] = record["targets"][record["valid"]]
    rows.append(
        score_points(
            oracle,
            record,
            "oracle_source_depth_correction",
            {
                "selected_ratio": 1.0,
                "valid_target_ratio": float(record["valid"].mean()),
            },
        )
    )
    for row in rows:
        print("Run 28 metric row:", row)
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
                "method_family": method_family(method),
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
                        "method_family": method_family(method),
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
    learned = next(row for row in candidates if row["method"] == "rdch_internal_selected")
    return baseline, learned


def gate_decision(summary_rows):
    comparisons = {
        subset: subset_comparison(summary_rows, subset)
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
            "selected_method": "rdch_internal_selected" if passed else overall_base["method"],
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
    valid = record["valid"]
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
        "mean_residual_m": float(record["residual_m"][valid].mean()) if valid.any() else 0.0,
        "median_residual_m": float(np.median(record["residual_m"][valid])) if valid.any() else 0.0,
        "wrong_depth_ratio": float(record["ambiguity_proxy"].mean()),
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
    fit_groups, internal_val_groups, internal_val_scenes = r27.split_internal_train_groups(train_groups)
    print("Run 28 splits:", splits)
    print("Run 28 group counts:", {"fit": len(fit_groups), "internal_val": len(internal_val_groups), "eval": len(eval_groups)})

    checkpoint = base.download_checkpoint(root)
    backbone = base.load_model(root, checkpoint)
    fit_records = []
    internal_val_records = []
    label_rows = []
    for group in train_groups:
        record = run_group(backbone, root, scene_lookup, group, out_dir / "train_groups" / group["group_key"])
        stage = "internal_val" if group["scene"] in internal_val_scenes else "fit"
        (internal_val_records if stage == "internal_val" else fit_records).append(record)
        label_rows.append(correction_summary(record, stage))

    feature_mean, feature_std = feature_standardization(fit_records)
    models = []
    history = []
    model_rows = []
    for model_seed in MODEL_SEEDS:
        model, model_history, model_row = train_model(
            fit_records,
            internal_val_records,
            feature_mean,
            feature_std,
            model_seed,
        )
        models.append(model)
        history.extend(model_history)
        model_rows.append(model_row)

    selected_policy, policy_rows = select_policy(models, internal_val_records, feature_mean, feature_std)
    print("Run 28 selected policy:", selected_policy)
    metric_rows = []
    for group in eval_groups:
        record = run_group(backbone, root, scene_lookup, group, out_dir / "eval_groups" / group["group_key"])
        label_rows.append(correction_summary(record, "external_eval"))
        metric_rows.extend(evaluate_group(models, record, feature_mean, feature_std, selected_policy))

    summary_rows = summarize(metric_rows)
    limit_rows = limit_summary(metric_rows)
    gate_rows = gate_decision(limit_rows)
    r27.write_csv_union(out_dir / "correction_label_summary.csv", label_rows)
    r27.write_csv_union(out_dir / "training_history.csv", history)
    r27.write_csv_union(out_dir / "model_selection.csv", model_rows)
    r27.write_csv_union(out_dir / "policy_selection.csv", policy_rows)
    r27.write_csv_union(out_dir / "metrics.csv", metric_rows)
    r27.write_csv_union(out_dir / "summary.csv", summary_rows)
    r27.write_csv_union(out_dir / "limit_summary.csv", limit_rows)
    r27.write_csv_union(out_dir / "gate_decision.csv", gate_rows)
    torch.save(
        {
            "state_dicts": [model.state_dict() for model in models],
            "model_seeds": MODEL_SEEDS,
            "feature_names": r27.FEATURE_NAMES,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "selected_policy": selected_policy,
        },
        out_dir / "ray_depth_correction_head.pt",
    )
    config = {
        "run": RUN_NAME,
        "self_contained": True,
        "num_scenes": len(scene_dirs),
        "scene_splits": splits,
        "num_fit_groups": len(fit_records),
        "num_internal_val_groups": len(internal_val_records),
        "num_eval_groups": len(eval_groups),
        "max_candidates_per_group": MAX_CANDIDATES_PER_GROUP,
        "model_seeds": MODEL_SEEDS,
        "epochs": EPOCHS,
        "loss_weights": LOSS_WEIGHTS,
        "selected_policy": selected_policy,
        "gate_margin_f1": GATE_MARGIN_F1,
        "runtime_seconds": time.time() - started,
        "inference_contract": "No depth or pose-derived target is used as an input feature at inference.",
        "oracle_contract": "Oracle source-depth correction is diagnostic only and excluded from policy selection and the gate.",
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print("Run 28 config:", config)
    print("Run 28 summary:")
    for row in summary_rows:
        print(row)
    print("Run 28 gate decision:")
    for row in gate_rows:
        print(row)


if __name__ == "__main__":
    main()
