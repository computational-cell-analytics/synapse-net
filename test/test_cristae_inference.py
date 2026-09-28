import unittest
from unittest import mock

import numpy as np


class TestCristaeInference(unittest.TestCase):
    def test_segment_cristae(self):
        from synapse_net.inference import cristae as cristae_module

        state = np.zeros((16, 32, 32))
        state[2:8, 4:14, 4:14] = 1
        state[9:15, 18:28, 18:28] = 2
        # Predict cristae everywhere, so that the segmentation is the mitochondria mask.
        predictions = np.stack([np.ones(state.shape, dtype="float32"), np.zeros(state.shape, dtype="float32")])

        # The mitochondria are taken from the input, which has the dtype of the tomogram. A float input used to fail,
        # because regionprops does not accept float labels.
        patch = mock.patch.object(cristae_module, "get_prediction", return_value=predictions)
        for dtype in ("float32", "uint8"):
            with self.subTest(dtype=dtype), patch:
                input_volume = np.stack([np.random.default_rng(0).normal(size=state.shape), state]).astype(dtype)
                seg = cristae_module.segment_cristae(input_volume, voxel_size=1.74, model=mock.Mock(), verbose=False,
                                                     min_size=0)
                np.testing.assert_array_equal(seg > 0, state > 0)


if __name__ == "__main__":
    unittest.main()
