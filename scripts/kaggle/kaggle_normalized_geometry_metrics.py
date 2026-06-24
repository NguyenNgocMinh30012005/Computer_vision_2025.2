"""Scale- and translation-invariant point-cloud metrics."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


NORMALIZED_DAC_THRESHOLD = 0.2
_EPSILON = 1e-8


def rms_normalize(points: np.ndarray) -> np.ndarray:
    """Zero-center a point cloud and divide it by its RMS radius."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Expected a non-empty point cloud with shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains non-finite coordinates")

    centered = points - points.mean(axis=0, keepdims=True)
    rms_scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    return centered / max(rms_scale, _EPSILON)


def compute_normalized_geometry_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    dac_threshold: float = NORMALIZED_DAC_THRESHOLD,
) -> dict[str, float]:
    """Compute symmetric normalized distance and prediction-side DAc."""
    pred_normalized = rms_normalize(pred)
    gt_normalized = rms_normalize(gt)

    pred_to_gt, _ = cKDTree(gt_normalized).query(
        pred_normalized, k=1, workers=-1
    )
    gt_to_pred, _ = cKDTree(pred_normalized).query(
        gt_normalized, k=1, workers=-1
    )
    normalized_distance = 0.5 * (
        float(pred_to_gt.mean()) + float(gt_to_pred.mean())
    )
    dac = float((pred_to_gt <= dac_threshold).mean())
    return {
        "normalized_distance": normalized_distance,
        "dac_at_0_2_normalized": dac,
        "normalized_dac_threshold": float(dac_threshold),
    }
