import unittest
from unittest import mock

import numpy as np


class TestCristaeInference(unittest.TestCase):
    """The post-processing of the cristae segmentation, with the prediction mocked."""

    def _segment(self, input_volume, **kwargs):
        from synapse_net.inference import cristae as cristae_module

        # Predict cristae everywhere, so that the segmentation is determined by the mitochondria mask.
        foreground = np.ones(input_volume.shape[1:], dtype="float32")
        predictions = np.stack([foreground, np.zeros_like(foreground)])
        with mock.patch.object(cristae_module, "get_prediction", return_value=predictions):
            return cristae_module.segment_cristae(
                input_volume, voxel_size=1.74, model=mock.Mock(), verbose=False, min_size=0, **kwargs
            )

    def _get_input(self, dtype):
        raw = np.random.default_rng(0).normal(size=(16, 32, 32))
        state = np.zeros((16, 32, 32))
        state[2:8, 4:14, 4:14] = 1
        state[9:15, 18:28, 18:28] = 2
        return np.stack([raw, state]).astype(dtype), state > 0

    def test_float_input(self):
        # The mitochondria state is taken from the input volume, which has the dtype of the tomogram.
        # This used to fail in regionprops, which does not accept float labels.
        input_volume, expected = self._get_input("float32")
        seg = self._segment(input_volume)
        np.testing.assert_array_equal(seg > 0, expected)

    def test_integer_input(self):
        input_volume, expected = self._get_input("uint8")
        seg = self._segment(input_volume)
        np.testing.assert_array_equal(seg > 0, expected)

    def test_float_input_with_erosion(self):
        from skimage.morphology import ball

        # Round mitochondria: the erosion runs within the bounding box of each mitochondrion, so it
        # does not shrink faces that are flat against the bounding box.
        state = np.zeros((32, 32, 32))
        state[2:15, 2:15, 2:15][ball(6).astype(bool)] = 1
        state[16:29, 16:29, 16:29][ball(6).astype(bool)] = 2
        raw = np.random.default_rng(0).normal(size=state.shape)
        input_volume, expected = np.stack([raw, state]).astype("float32"), state > 0
        seg = self._segment(input_volume, erosion_distance_nm=1.74)
        self.assertTrue((seg > 0).any())
        self.assertLess((seg > 0).sum(), expected.sum())
        self.assertFalse((seg[~expected] > 0).any())


if __name__ == "__main__":
    unittest.main()
