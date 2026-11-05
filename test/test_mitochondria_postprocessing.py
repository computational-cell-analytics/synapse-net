"""Offline tests for the post-processing of the mitochondria segmentation (_postprocess_seg_3d)."""
import unittest

import numpy as np

MITO_SETTINGS = dict(area_threshold=5000, iterations=2, iterations_3d=3, fill_holes=True, remove_tunnels=True)


def _curled_object(gap, length=16, width=3, overhang=4, offset=4):
    """A rod along six edges of a cube that almost closes into a ring: its end stops `gap` voxels before its start.

    The ring does not lie in a plane, so no hole filling per slice can close it. The rods overhang the corners,
    so that the gap is inside the bounding box of the object.
    """
    shift = offset + overhang
    shape = (2 * shift + length + width,) * 3
    seg = np.zeros(shape, dtype="uint32")
    corners = [(0, 0, 0), (length, 0, 0), (length, length, 0), (length, length, length),
               (0, length, length), (0, 0, length), (0, 0, 0)]
    for start, stop in zip(corners[:-1], corners[1:]):
        lower = [shift + min(a, b) - (overhang if a != b else 0) for a, b in zip(start, stop)]
        upper = [shift + max(a, b) + width + (overhang if a != b else 0) for a, b in zip(start, stop)]
        if stop == corners[0]:
            lower[2] = shift + width + gap
        seg[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] = 1
    return seg


class TestMitochondriaPostprocessing(unittest.TestCase):
    def test_count_tunnels(self):
        from synapse_net.inference.util import _count_tunnels

        z, y, x = np.mgrid[:30, :30, :30] - 14.5
        ball = z ** 2 + y ** 2 + x ** 2 < 100
        torus = (np.sqrt(y ** 2 + x ** 2) - 9) ** 2 + z ** 2 < 9
        self.assertEqual(_count_tunnels(ball), 0)
        self.assertEqual(_count_tunnels(torus), 1)
        self.assertEqual(_count_tunnels(np.concatenate([torus, torus])), 2)

    def test_curled_object(self):
        from synapse_net.inference.util import _count_tunnels, _postprocess_seg_3d

        seg = _curled_object(gap=2)
        self.assertEqual(_count_tunnels(seg == 1), 0)

        # The default closing joins the two ends and gives the object a tunnel.
        default = _postprocess_seg_3d(seg.copy(), area_threshold=5000)
        self.assertEqual(_count_tunnels(default == 1), 1)

        # The weaker closing joins them too, but the join is removed again.
        bridged = _postprocess_seg_3d(seg.copy(), **{**MITO_SETTINGS, "remove_tunnels": False})
        self.assertEqual(_count_tunnels(bridged == 1), 1)
        result = _postprocess_seg_3d(seg.copy(), **MITO_SETTINGS)
        self.assertEqual(_count_tunnels(result == 1), 0)

        # The other added voxels, which fill the corners, are kept.
        self.assertTrue(np.all(result[seg == 1] == 1))
        self.assertGreater(np.count_nonzero(result), np.count_nonzero(seg))

    def test_fill_holes(self):
        from synapse_net.inference.util import _count_tunnels, _postprocess_seg_3d

        # A tube along z with a second object inside it.
        seg = np.zeros((12, 40, 40), dtype="uint32")
        seg[1:11, 5:35, 5:35] = 1
        seg[1:11, 8:32, 8:32] = 0
        seg[4:8, 18:22, 18:22] = 2
        lumen = np.zeros(seg.shape, dtype=bool)
        lumen[1:11, 8:32, 8:32] = True
        lumen[seg == 2] = False

        result = _postprocess_seg_3d(seg.copy(), **MITO_SETTINGS)
        self.assertTrue(np.all(result[lumen] == 1))
        self.assertTrue(np.all(result[seg == 2] == 2))
        self.assertEqual(_count_tunnels(result == 1), 0)

        # Without the hole filling, the lumen stays open.
        result = _postprocess_seg_3d(seg.copy(), **{**MITO_SETTINGS, "fill_holes": False})
        self.assertEqual(_count_tunnels(result == 1), 1)


if __name__ == "__main__":
    unittest.main()
