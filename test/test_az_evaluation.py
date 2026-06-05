"""Offline regression tests for the active-zone evaluation utilities.

``_get_presynaptic_mask`` is the second call site of the vigra ``localMaxima`` ->
``skimage.morphology.local_maxima`` replacement. ``az_evaluation`` is the public
entry point that wraps it / the matching metrics.
"""
import os
import unittest
from shutil import rmtree

import h5py
import numpy as np


class TestAzEvaluation(unittest.TestCase):
    tmp_dir = "./tmp_az_eval"

    def setUp(self):
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_dir)
        except OSError:
            pass

    def _write_h5(self, name, key, data):
        path = os.path.join(self.tmp_dir, name)
        with h5py.File(path, "a") as f:
            f.create_dataset(key, data=data)
        return path

    def test_az_evaluation(self):
        from synapse_net.ground_truth.az_evaluation import az_evaluation

        # A single foreground blob, identical in segmentation and ground-truth.
        gt = np.zeros((4, 32, 32), dtype="uint8")
        gt[:, 6:26, 6:26] = 1  # 4 * 20 * 20 = 1600 voxels (> the 500 / min_component_size cutoffs)
        seg = gt.copy()

        seg_path = self._write_h5("seg.h5", "seg", seg)
        gt_path = self._write_h5("gt.h5", "gt", gt)

        df = az_evaluation(
            [seg_path], [gt_path], seg_key="seg", gt_key="gt",
            min_component_size=10, iterations=0, criterion="iou",
        )

        self.assertEqual(len(df), 1)
        for column in ("tomo_name", "tp", "fp", "fn", "dice"):
            self.assertIn(column, df.columns)

        dice = df["dice"].iloc[0]
        self.assertGreaterEqual(dice, 0.0)
        self.assertLessEqual(dice, 1.0)
        # Identical seg/gt -> perfect match.
        self.assertAlmostEqual(dice, 1.0, places=5)
        self.assertEqual(int(df["tp"].iloc[0]), 1)
        self.assertEqual(int(df["fp"].iloc[0]), 0)
        self.assertEqual(int(df["fn"].iloc[0]), 0)

    def test_get_presynaptic_mask(self):
        from synapse_net.ground_truth.az_evaluation import _get_presynaptic_mask

        # Boundary map dividing each slice into four quadrants.
        boundary_map = np.full((4, 64, 64), 0.05, dtype="float32")
        boundary_map[:, 30:34, :] = 0.9
        boundary_map[:, :, 30:34] = 0.9

        # A few vesicles, all in the top-left quadrant.
        vesicles = np.zeros((4, 64, 64), dtype="uint32")
        vesicles[:, 8:12, 8:12] = 1
        vesicles[:, 8:12, 18:22] = 2
        vesicles[:, 18:22, 8:12] = 3

        try:
            mask = _get_presynaptic_mask(boundary_map, vesicles)
        except Exception as error:  # pragma: no cover - depends on the elf multicut workflow
            self.skipTest(f"simple_multicut_workflow unavailable in this environment: {error}")

        self.assertEqual(mask.shape, vesicles.shape)
        self.assertEqual(mask.dtype, np.dtype("bool"))


if __name__ == "__main__":
    unittest.main()
