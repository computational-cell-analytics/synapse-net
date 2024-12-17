import os
import unittest
from functools import partial
from shutil import rmtree

import imageio.v3 as imageio
from synapse_net.file_utils import read_mrc
from synapse_net.sample_data import get_sample_data


class TestInference(unittest.TestCase):
    tmp_dir = "tmp"
    model_type = "vesicles_2d"
    tiling = {"tile": {"z": 1, "y": 512, "x": 512}, "halo": {"z": 0, "y": 32, "x": 32}}

    def setUp(self):
        self.data_path = get_sample_data("tem_2d")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_dir)
        except OSError:
            pass

    def test_run_segmentation(self):
        from synapse_net.inference import run_segmentation, get_model

        image, _ = read_mrc(self.data_path)
        model = get_model(self.model_type)
        seg = run_segmentation(image, model, model_type=self.model_type, tiling=self.tiling)
        self.assertEqual(image.shape, seg.shape)

    def test_segmentation_with_inference_helper(self):
        from synapse_net.inference import run_segmentation, get_model
        from synapse_net.inference.util import inference_helper

        model = get_model(self.model_type)
        segmentation_function = partial(
            run_segmentation, model=model, model_type=self.model_type, verbose=False, tiling=self.tiling,
        )
        inference_helper(self.data_path, self.tmp_dir, segmentation_function, data_ext=".mrc")
        expected_output_path = os.path.join(self.tmp_dir, "tem_2d_prediction.tif")
        self.assertTrue(os.path.exists(expected_output_path))
        seg = imageio.imread(expected_output_path)
        image, _ = read_mrc(self.data_path)
        self.assertEqual(image.shape, seg.shape)


if __name__ == "__main__":
    unittest.main()
