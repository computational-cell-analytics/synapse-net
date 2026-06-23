"""Offline regression test for the size filter widget logic.

Exercises the ``nifty.tools.takeDict`` -> ``bioimage_cpp.utils.take_dict``
replacement in ``SizeFilterWidget._filter_segmentation``. The method does not use
``self``, so we invoke it on a bare instance (``object.__new__``) without building
the Qt widget / napari viewer.
"""
import unittest

import numpy as np


class TestSizeFilter(unittest.TestCase):
    def setUp(self):
        from synapse_net.tools.size_filter_widget import SizeFilterWidget

        # _filter_segmentation never touches self, so we call it as an unbound
        # function with self=None and avoid constructing the Qt widget / viewer.
        self.filter_segmentation = SizeFilterWidget._filter_segmentation

        # Label image with one large component (label 5, area 400) and one small
        # component (label 7, area 9).
        self.seg = np.zeros((30, 30), dtype="uint32")
        self.seg[2:22, 2:22] = 5   # large: 20 x 20 = 400 px
        self.seg[25:28, 25:28] = 7  # small: 3 x 3 = 9 px
        self.large_mask = self.seg == 5
        self.small_mask = self.seg == 7

    def test_filter_with_relabel(self):
        # Threshold between the two component sizes -> small one is removed, large
        # one is kept and remapped back to its original label (5) via take_dict.
        result = self.filter_segmentation(None, self.seg.copy(), size_threshold=100, apply_label=True)

        self.assertEqual(result.shape, self.seg.shape)
        self.assertEqual(result.dtype, self.seg.dtype)
        self.assertTrue(np.all(result[self.small_mask] == 0))
        self.assertTrue(np.all(result[self.large_mask] == 5))

    def test_filter_without_relabel(self):
        # apply_label=False does not go through take_dict; it filters on the
        # existing labels directly.
        result = self.filter_segmentation(None, self.seg.copy(), size_threshold=100, apply_label=False)

        self.assertEqual(result.shape, self.seg.shape)
        self.assertTrue(np.all(result[self.small_mask] == 0))
        self.assertTrue(np.all(result[self.large_mask] == 5))


if __name__ == "__main__":
    unittest.main()
