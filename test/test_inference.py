import os
import tempfile
import unittest
from functools import partial
from shutil import rmtree
from types import SimpleNamespace
from unittest import mock

import imageio.v3 as imageio
import requests
import torch
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

    # Test that all models can be accessed by pooch and have the necessary resolution information.
    def test_models(self):
        from synapse_net.inference.inference import (
            _get_model_registry, get_model_training_patch_shape, get_model_training_resolution
        )

        def check_url(url):
            try:
                # Make a HEAD request to the URL, which fetches HTTP headers but no content.
                response = requests.head(url, allow_redirects=True)
                # Check if the HTTP status code is one that indicates availability (200 <= code < 400).
                return response.status_code < 400
            except requests.RequestException:
                # Handle connection exceptions
                return False

        registry = _get_model_registry()
        for name in registry.registry.keys():
            url_exists = check_url(registry.get_url(name))
            self.assertTrue(url_exists)
            get_model_training_resolution(name)
            get_model_training_patch_shape(name)

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


class TestTiling(unittest.TestCase):
    patch_shape = {"z": 32, "y": 256, "x": 256}

    def _get_tiling(self, vram=None, **kwargs):
        from synapse_net.inference.util import get_default_tiling

        if vram is None:  # Simulate the CPU.
            with mock.patch("torch.cuda.is_available", return_value=False), \
                    mock.patch("torch.backends.mps.is_available", return_value=False):
                return get_default_tiling(**kwargs)
        properties = mock.Mock(total_memory=vram * 1e9)
        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.get_device_properties", return_value=properties):
            return get_default_tiling(**kwargs)

    def test_tiling_from_patch_shape(self):
        halo = {"z": 8, "y": 64, "x": 64}
        # The tile is at least the training patch shape, also on the CPU, where the VRAM tiling is smaller.
        self.assertEqual(self._get_tiling(patch_shape=self.patch_shape), {"tile": self.patch_shape, "halo": halo})
        self.assertEqual(self._get_tiling(12, patch_shape=self.patch_shape)["tile"], self.patch_shape)
        # With enough VRAM it grows by at most 20%, rounded down to the pooling factors (8 in z, 16 in y and x).
        expected = {"tile": {"z": 32, "y": 304, "x": 304}, "halo": halo}
        self.assertEqual(self._get_tiling(80, patch_shape=self.patch_shape), expected)
        expected = {"z": 72, "y": 608, "x": 608}
        self.assertEqual(self._get_tiling(80, patch_shape={"z": 64, "y": 512, "x": 512})["tile"], expected)
        # A UNETR must not grow beyond the training patch shape.
        self.assertEqual(self._get_tiling(80, patch_shape=self.patch_shape, max_growth=1.0)["tile"], self.patch_shape)
        # 2d models predict single slices.
        expected = {"tile": {"z": 1, "y": 304, "x": 304}, "halo": {"z": 0, "y": 64, "x": 64}}
        self.assertEqual(self._get_tiling(patch_shape={"y": 256, "x": 256}), expected)
        # Without a patch shape the tiling only depends on the VRAM.
        self.assertEqual(self._get_tiling(80)["tile"], {"z": 80, "y": 640, "x": 640})

    def test_tiling_from_checkpoint(self):
        from synapse_net.inference.util import _get_default_tiling_for_model, get_default_tiling
        from torch_em.model.unetr import UNETRBase

        # torch_em checkpoints store the (concatenated) training dataset, which knows the patch shape.
        dataset = SimpleNamespace(datasets=[SimpleNamespace(patch_shape=(32, 256, 256))])
        model_classes = {"torch_em.model.unet.AnisotropicUNet": 1.2, "torch_em.model.unetr.UNETR": 1.0}
        with tempfile.TemporaryDirectory() as checkpoint:
            for model_class, max_growth in model_classes.items():
                init = {"train_dataset": dataset, "model_class": model_class}
                torch.save({"init": init}, os.path.join(checkpoint, "best.pt"))
                expected = get_default_tiling(patch_shape=self.patch_shape, max_growth=max_growth)
                self.assertEqual(_get_default_tiling_for_model(None, checkpoint), expected)

        # The tiling for a UNETR without a checkpoint can't be derived.
        with self.assertRaises(ValueError):
            _get_default_tiling_for_model(UNETRBase.__new__(UNETRBase), None)


if __name__ == "__main__":
    unittest.main()
