import math

import numpy as np
from PIL import Image


MIN_DEPTH_M = 0.10


def resize_depth(depth, width, height):
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape={depth.shape}")
    image = Image.fromarray(depth, mode="F")
    return np.asarray(
        image.resize((int(width), int(height)), resample=Image.BICUBIC),
        dtype=np.float32,
    )


def read_metric_depth(path):
    depth = np.asarray(Image.open(path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if depth.size and float(np.nanmax(depth)) > 100.0:
        depth /= 1000.0
    depth[~np.isfinite(depth)] = 0.0
    return depth


def valid_depth_mask(source_depth, predicted_depth, min_depth_m=MIN_DEPTH_M):
    source = np.asarray(source_depth, dtype=np.float32)
    predicted = np.asarray(predicted_depth, dtype=np.float32)
    if source.shape != predicted.shape:
        raise ValueError(
            f"Depth shape mismatch: source={source.shape}, predicted={predicted.shape}"
        )
    return (
        np.isfinite(source)
        & np.isfinite(predicted)
        & (source > float(min_depth_m))
        & (predicted > 1e-6)
    )


def depth_scale_factors(predicted, source):
    predicted = np.asarray(predicted, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    ratio = source / np.maximum(predicted, 1e-8)
    median_scale = float(np.median(ratio))
    least_squares_scale = float(
        np.sum(predicted * source) / max(np.sum(predicted * predicted), 1e-12)
    )
    return median_scale, least_squares_scale


def depth_error_metrics(
    predicted_depth,
    source_depth,
    min_depth_m=MIN_DEPTH_M,
):
    predicted = np.asarray(predicted_depth, dtype=np.float32)
    source = np.asarray(source_depth, dtype=np.float32)
    valid = valid_depth_mask(source, predicted, min_depth_m=min_depth_m)
    if int(valid.sum()) < 16:
        raise ValueError(f"Insufficient valid depth pixels: {int(valid.sum())}")

    pred = predicted[valid].astype(np.float64)
    src = source[valid].astype(np.float64)
    error = pred - src
    abs_error = np.abs(error)
    ratio = np.maximum(
        pred / np.maximum(src, 1e-8),
        src / np.maximum(pred, 1e-8),
    )
    median_scale, least_squares_scale = depth_scale_factors(pred, src)
    aligned = pred * median_scale
    aligned_error = aligned - src

    return {
        "num_pixels": int(source.size),
        "num_valid_pixels": int(valid.sum()),
        "valid_pixel_ratio": float(valid.mean()),
        "abs_rel": float(np.mean(abs_error / np.maximum(src, 1e-8))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(abs_error)),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
        "scale_aligned_rmse": float(
            np.sqrt(np.mean(aligned_error * aligned_error))
        ),
        "scale_aligned_mae": float(np.mean(np.abs(aligned_error))),
        "median_scale_ratio": median_scale,
        "least_squares_scale_ratio": least_squares_scale,
        "source_depth_mean_m": float(np.mean(src)),
        "predicted_depth_mean_m": float(np.mean(pred)),
    }


def summarize_depth_rows(rows, split):
    selected = rows if split == "all" else [row for row in rows if row["split"] == split]
    if not selected:
        return None
    metric_names = [
        "valid_pixel_ratio",
        "abs_rel",
        "rmse",
        "mae",
        "delta1",
        "delta2",
        "delta3",
        "scale_aligned_rmse",
        "scale_aligned_mae",
        "median_scale_ratio",
        "least_squares_scale_ratio",
    ]
    output = {
        "split": split,
        "num_frames": len(selected),
        "num_scenes": len({row["scene"] for row in selected}),
    }
    for name in metric_names:
        values = [
            float(row[name])
            for row in selected
            if math.isfinite(float(row[name]))
        ]
        output[f"mean_{name}"] = float(np.mean(values))
        output[f"median_{name}"] = float(np.median(values))
    return output


def intrinsics(depth_shape):
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
    min_depth_m=MIN_DEPTH_M,
):
    poses = [
        parse_pose(str(path).replace(".jpg", ".txt")).astype(np.float32)
        for path in view_files
    ]
    first_pose_inv = np.linalg.inv(poses[0]).astype(np.float32)
    targets = np.zeros((len(xs), 3), dtype=np.float32)
    valid = np.zeros(len(xs), dtype=bool)
    sampled_depth = np.zeros(len(xs), dtype=np.float32)

    for view_index, depth in enumerate(depth_maps):
        candidate_indices = np.where(view_ids == view_index)[0]
        if not len(candidate_indices):
            continue
        depth = np.asarray(depth, dtype=np.float32) * float(scale)
        height, width = depth.shape
        depth_x = np.clip(
            np.rint(
                xs[candidate_indices]
                / max(candidate_width - 1, 1)
                * max(width - 1, 1)
            ).astype(np.int32),
            0,
            width - 1,
        )
        depth_y = np.clip(
            np.rint(
                ys[candidate_indices]
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
        sampled_depth[usable_indices] = z
        valid[usable_indices] = True
    return targets, valid, sampled_depth


def backproject_depth_cloud(
    view_files,
    depth_maps,
    parse_pose,
    scale=1.0,
    stride=4,
    min_depth_m=MIN_DEPTH_M,
):
    poses = [
        parse_pose(str(path).replace(".jpg", ".txt")).astype(np.float32)
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
        clouds.append(first_camera[:, :3].astype(np.float32))
    if not clouds:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(clouds, axis=0)


def apply_residual_correction(points, targets, valid, tau_pred, alpha):
    points = np.asarray(points, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    residual = np.linalg.norm(points - targets, axis=1).astype(np.float32)
    mask = valid & (residual >= float(tau_pred))
    corrected = points.copy()
    corrected[mask] = (
        (1.0 - float(alpha)) * points[mask] + float(alpha) * targets[mask]
    )
    return corrected, mask, residual
