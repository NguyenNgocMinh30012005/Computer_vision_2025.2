import unittest

import numpy as np

from scripts.kaggle.kaggle_normalized_geometry_metrics import (
    compute_normalized_geometry_metrics,
)


class NormalizedGeometryMetricsTest(unittest.TestCase):
    def setUp(self):
        self.cloud = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def test_identical_cloud_is_perfect(self):
        metrics = compute_normalized_geometry_metrics(self.cloud, self.cloud)
        self.assertAlmostEqual(metrics["normalized_distance"], 0.0, places=12)
        self.assertAlmostEqual(metrics["dac_at_0_2_normalized"], 1.0, places=12)

    def test_translation_and_scale_are_removed(self):
        transformed = self.cloud * 4.25 + np.array([7.0, -3.0, 2.5])
        metrics = compute_normalized_geometry_metrics(transformed, self.cloud)
        self.assertAlmostEqual(metrics["normalized_distance"], 0.0, places=12)
        self.assertAlmostEqual(metrics["dac_at_0_2_normalized"], 1.0, places=12)

    def test_shape_error_is_detected(self):
        distorted = self.cloud.copy()
        distorted[-1] = [0.0, 0.0, 4.0]
        metrics = compute_normalized_geometry_metrics(distorted, self.cloud)
        self.assertGreater(metrics["normalized_distance"], 0.05)
        self.assertLess(metrics["dac_at_0_2_normalized"], 1.0)


if __name__ == "__main__":
    unittest.main()
