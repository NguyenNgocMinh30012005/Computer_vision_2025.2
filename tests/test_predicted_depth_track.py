import unittest

import numpy as np

from scripts.kaggle.kaggle_predicted_depth_utils import (
    apply_residual_correction,
    depth_error_metrics,
)


class PredictedDepthTrackTest(unittest.TestCase):
    def test_perfect_metric_depth(self):
        source = np.linspace(1.0, 4.0, 16, dtype=np.float32).reshape(4, 4)
        metrics = depth_error_metrics(source, source)
        self.assertAlmostEqual(metrics["abs_rel"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["delta1"], 1.0)
        self.assertAlmostEqual(metrics["scale_aligned_rmse"], 0.0)
        self.assertAlmostEqual(metrics["median_scale_ratio"], 1.0)

    def test_scale_alignment_removes_global_scale_error(self):
        source = np.linspace(0.5, 4.0, 64, dtype=np.float32).reshape(8, 8)
        predicted = source * 0.5
        metrics = depth_error_metrics(predicted, source)
        self.assertGreater(metrics["rmse"], 0.5)
        self.assertAlmostEqual(metrics["median_scale_ratio"], 2.0, places=6)
        self.assertLess(metrics["scale_aligned_rmse"], 1e-6)
        self.assertLess(metrics["scale_aligned_mae"], 1e-6)

    def test_residual_gate_corrects_only_large_residuals(self):
        points = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            dtype=np.float32,
        )
        targets = np.array(
            [[0.0, 0.0, 1.05], [0.0, 0.0, 3.0]],
            dtype=np.float32,
        )
        corrected, mask, residual = apply_residual_correction(
            points,
            targets,
            np.array([True, True]),
            tau_pred=0.2,
            alpha=0.5,
        )
        np.testing.assert_array_equal(mask, np.array([False, True]))
        np.testing.assert_allclose(corrected[0], points[0])
        np.testing.assert_allclose(
            corrected[1],
            np.array([0.0, 0.0, 2.5], dtype=np.float32),
        )
        np.testing.assert_allclose(residual, np.array([0.05, 1.0]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
