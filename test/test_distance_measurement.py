import unittest

import numpy as np


class TestDistanceMeasurement(unittest.TestCase):
    def test_measure_pairwise_object_distances(self):
        from synapse_net.distance_measurements import measure_pairwise_object_distances

        shape = (4, 64, 64)
        seg = np.zeros(shape, dtype="uint32")

        seg[1, 16, 0] = 1
        seg[1, 16, 16] = 2
        seg[1, 16, 32] = 3
        seg[1, 16, 48] = 4
        seg[1, 16, 63] = 5

        for resolution in (None, 2.3, 4.4):
            distances, _, _, seg_ids = measure_pairwise_object_distances(seg, resolution=resolution, n_threads=1)

            factor = 1 if resolution is None else resolution

            self.assertTrue(np.isclose(distances[0, 1], factor * 16))  # distance between object 1 and 2
            self.assertTrue(np.isclose(distances[0, 2], factor * 32))  # distance between object 1 and 3
            self.assertTrue(np.isclose(distances[0, 3], factor * 48))  # distance between object 1 and 4
            self.assertTrue(np.isclose(distances[0, 4], factor * 63))  # distance between object 1 and 5

            self.assertTrue(np.isclose(distances[1, 2], factor * 16))  # distance between object 2 and 3
            self.assertTrue(np.isclose(distances[1, 3], factor * 32))  # distance between object 2 and 4
            self.assertTrue(np.isclose(distances[1, 4], factor * 47))  # distance between object 2 and 5

            self.assertTrue(np.isclose(distances[2, 3], factor * 16))  # distance between object 3 and 4
            self.assertTrue(np.isclose(distances[2, 4], factor * 31))  # distance between object 3 and 5

            self.assertTrue(np.isclose(distances[3, 4], factor * 15))  # distance between object 3 and 5

    def test_keep_direct_distances(self):
        from synapse_net.distance_measurements import measure_pairwise_object_distances, keep_direct_distances

        seg = np.zeros((4, 64, 64), dtype="uint32")
        seg[1, 16, 0] = 1
        seg[1, 16, 16:20] = 2  # Object 2 sits between object 1 and 3 and blocks their distance line.
        seg[1, 16, 32] = 3

        distances, endpoints1, endpoints2, seg_ids = measure_pairwise_object_distances(seg, n_threads=1)
        pairs = keep_direct_distances(seg, distances, endpoints1, endpoints2, seg_ids)
        self.assertEqual(sorted(map(tuple, pairs)), [(1, 2), (2, 3)])

    def test_filter_blocked_segmentation_to_object_distances(self):
        from synapse_net.distance_measurements import (
            measure_segmentation_to_object_distances, filter_blocked_segmentation_to_object_distances
        )

        seg = np.zeros((4, 64, 64), dtype="uint32")
        seg[1, 16, 8] = 1
        seg[1, 16, 24] = 2
        obj = np.zeros_like(seg)
        obj[1, 16, 40:48] = 1

        distances, endpoints1, endpoints2, seg_ids = measure_segmentation_to_object_distances(seg, obj)
        kept = filter_blocked_segmentation_to_object_distances(seg, distances, endpoints1, endpoints2, seg_ids)
        self.assertEqual(sorted(kept), [2])  # The distance line of object 1 is blocked by object 2.
        kept = filter_blocked_segmentation_to_object_distances(
            seg, distances, endpoints1, endpoints2, seg_ids, filter_seg_ids=[1]
        )
        self.assertEqual(kept, [])


if __name__ == "__main__":
    unittest.main()
