import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.kaggle.kaggle_run37_depth_finetune_utils import (
    assign_scene_splits,
    depth_error_metrics,
    discover_complete_rgbd_pose_frames,
    split_summary,
)


class Run37DepthFinetuneUtilsTest(unittest.TestCase):
    def test_scene_split_is_scene_level_and_balanced(self):
        scenes = [f"scene{i:04d}_00" for i in range(20)]
        splits = assign_scene_splits(scenes)
        self.assertEqual(set(splits), set(scenes))
        counts = {name: list(splits.values()).count(name) for name in {"train", "val", "test"}}
        self.assertEqual(counts["train"], 16)
        self.assertEqual(counts["val"], 2)
        self.assertEqual(counts["test"], 2)

    def test_complete_frame_inventory_requires_rgb_depth_and_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / "scene0000_00"
            scene.mkdir()
            Image.new("RGB", (4, 4), "red").save(scene / "00000.jpg")
            Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(
                scene / "00000.png"
            )
            (scene / "00000.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
            Image.new("RGB", (4, 4), "blue").save(scene / "00010.jpg")
            Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(
                scene / "00010.png"
            )

            rows = discover_complete_rgbd_pose_frames(
                root,
                scene_splits={"scene0000_00": "train"},
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["frame"], "00000")
            summary = split_summary(rows)
            self.assertEqual(summary[0]["num_frames"], 1)

    def test_depth_metrics_detect_scale_improvement_potential(self):
        source = np.linspace(0.5, 4.0, 64, dtype=np.float32).reshape(8, 8)
        predicted = source * 0.5
        metrics = depth_error_metrics(predicted, source)
        self.assertGreater(metrics["abs_rel"], 0.4)
        self.assertAlmostEqual(metrics["median_scale_ratio"], 2.0, places=6)
        self.assertLess(metrics["scale_aligned_mae"], 1e-6)


if __name__ == "__main__":
    unittest.main()
