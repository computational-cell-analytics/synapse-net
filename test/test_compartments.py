"""Offline regression tests for the compartment segmentation helpers migrated
from vigra/nifty to bioimage-cpp.

- ``_segment_compartments_2d`` exercises the vigra ``localMaxima`` ->
  ``skimage.morphology.local_maxima`` seed replacement (no model needed).
- ``_merge_segmentation_3d`` exercises the nifty graph -> ``bic.graph`` and the
  ``nifty.tools.take`` -> numpy-indexing replacements.

The end-to-end ``segment_compartments`` (with a real model) stays covered by the
network-dependent ``test_inference`` suite; these tests give deterministic,
offline coverage of the migrated lines.
"""
import unittest

import numpy as np


class TestCompartments(unittest.TestCase):
    def test_segment_compartments_2d(self):
        from synapse_net.inference.compartments import _segment_compartments_2d

        # Boundary probability map: ~0 inside two regions separated by a high
        # boundary wall. The distance transform peaks inside each region, so the
        # seed extraction (local_maxima + connected components) must yield two
        # compartments.
        boundaries = np.zeros((120, 120), dtype="float32")
        boundaries[:, 58:62] = 1.0

        seg = _segment_compartments_2d(boundaries, large_seed_distance=10)

        self.assertEqual(seg.shape, boundaries.shape)
        labels = np.unique(seg)
        labels = labels[labels > 0]
        self.assertEqual(len(labels), 2)

    def test_merge_segmentation_3d(self):
        from synapse_net.inference.compartments import _merge_segmentation_3d

        # Build a 2d-per-slice segmentation with a left and right column object
        # per slice, each given a globally unique label. Consecutive slices fully
        # overlap, so the multicut should merge labels along a column.
        seg_2d = np.zeros((5, 16, 16), dtype="uint32")
        for z in range(seg_2d.shape[0]):
            seg_2d[z, :, :8] = 2 * z + 1
            seg_2d[z, :, 8:] = 2 * z + 2

        n_in = len(np.unique(seg_2d))  # 10 distinct labels, no background present

        # min_z_extent=0 disables the z-extent filtering so we only test the merge.
        merged = _merge_segmentation_3d(seg_2d, min_z_extent=0)

        # Shape is preserved.
        self.assertEqual(merged.shape, seg_2d.shape)
        # node_labels[seg_2d] must map every input label to a single output label.
        for input_label in np.unique(seg_2d):
            self.assertEqual(len(np.unique(merged[seg_2d == input_label])), 1)
        # The multicut (built via bic.graph + insert_edges) must merge at least
        # some labels, so fewer distinct objects than inputs remain.
        n_out = len(np.unique(merged[merged > 0]))
        self.assertGreater(n_out, 0)
        self.assertLess(n_out, n_in)


if __name__ == "__main__":
    unittest.main()
