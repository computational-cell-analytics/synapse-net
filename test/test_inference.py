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
                # Stream a GET request so that only the headers are fetched (some hosts reject HEAD requests).
                with requests.get(url, allow_redirects=True, stream=True) as response:
                    pass
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

    def test_compute_scale_from_voxel_size(self):
        from synapse_net.inference import compute_scale_from_voxel_size

        # The scale is in zyx order, which matters for a model with an anisotropic training resolution.
        scale = compute_scale_from_voxel_size({"x": 5.0, "y": 10.0, "z": 5.0}, "mitochondria_fibsem")
        self.assertEqual(scale, [0.2, 2.0, 1.0])
        scale = compute_scale_from_voxel_size({"x": 2.7, "y": 1.35}, "vesicles_2d")
        self.assertEqual(scale, [1.0, 2.0])

    # Test that bioimage.io models are converted correctly, by checking them against the models' test tensors.
    # Only one 2D and one 3D model are checked, to limit the download size.
    def test_bioimageio_models(self):
        import numpy as np
        import pooch
        from synapse_net.inference import get_model
        from synapse_net.inference.inference import get_model_path

        test_tensors = {
            "vesicles_2d": ("adaptable-microbe", {
                "test_input.npy": "0cd6f464ec8eaa2928b100e456bc0a53949f926e00a4c9e428c0a3db014a2d36",
                "test_output.npy": "7ba544cfa1ac3ea75dd0c546c485baa94de2e4f576449841678f651d99fe0653",
            }),
            "cristae5": ("gorgeous-eagle", {
                "test_raw.npy": "49bd5b93747875ae069bc13a5aa8e971f099a0180f06aa678890209feaf008fe",
                "test_mito_mask.npy": "391ebfa87a95f74d99a671701134c4fb437a879581dcf21743bd690f165b201d",
                "test_output.npy": "0e734e083895f7ee5abced5d20285a05e677be2405a8b1dfd9f7d89611f03042",
            }),
        }
        url = "https://hypha.aicell.io/bioimage-io/artifacts/{}/files/{}?version=v0"
        for model_type, (artifact, files) in test_tensors.items():
            with self.subTest(model_type=model_type):
                *inputs, expected = [np.load(pooch.retrieve(url.format(artifact, f), sha)) for f, sha in files.items()]
                # Only the raw data is standardized, the mitochondria mask of the cristae model is passed as is.
                raw = inputs[0]
                inputs[0] = (raw - raw.mean()) / (raw.std() + 1e-7)
                model = get_model(model_type, device="cpu")
                with torch.no_grad():
                    pred = model(torch.from_numpy(np.concatenate(inputs, axis=1).astype("float32"))).numpy()
                self.assertTrue(np.allclose(pred, expected, atol=1e-4))
                # Only the converted model stays in the cache, the downloaded state dict is removed.
                model_path = get_model_path(model_type)
                self.assertTrue(os.path.exists(model_path))
                self.assertFalse(os.path.exists(os.path.join(os.path.dirname(model_path), model_type)))

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

    def test_prediction_bfloat16(self):
        import numpy as np
        from synapse_net.inference import get_model
        from synapse_net.inference.util import get_prediction

        image, _ = read_mrc(self.data_path)

        # bf16 convolutions are slow on the CPU, so we predict a single small block.
        tiling = {"tile": {"z": 1, "y": 256, "x": 256}, "halo": {"z": 0, "y": 32, "x": 32}}
        image = image[:192, :192]
        model = get_model(self.model_type)
        kwargs = dict(model=model, tiling=tiling, verbose=False, devices=["cpu"])
        pred_fp32 = get_prediction(image, **kwargs)
        pred_bf16 = get_prediction(image, **kwargs, dtype=torch.bfloat16)
        diff = np.abs(pred_fp32 - pred_bf16)
        print(diff.max(), diff.mean(), (diff > 0.05).mean())

        # autocast runs in bfloat16, but returns float32
        self.assertEqual(pred_bf16.dtype, "float32")
        self.assertFalse(np.array_equal(pred_fp32, pred_bf16))
        self.assertTrue(np.allclose(pred_fp32, pred_bf16, atol=0.05))


class TestTiling(unittest.TestCase):
    patch_shape = {"z": 32, "y": 256, "x": 256}

    def _get_tiling(self, vram=None, **kwargs):
        from synapse_net.inference.util import get_default_tiling

        # The device always defaults to the CPU on CI, which would override the simulated hardware.
        no_ci = mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "false"})
        if vram is None:  # Simulate the CPU.
            with no_ci, mock.patch("torch.cuda.is_available", return_value=False), \
                    mock.patch("torch.backends.mps.is_available", return_value=False):
                return get_default_tiling(**kwargs)
        properties = mock.Mock(total_memory=vram * 1e9)
        with no_ci, mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.get_device_properties", return_value=properties):
            return get_default_tiling(**kwargs)

    def test_tiling_from_patch_shape(self):
        halo = {"z": 8, "y": 64, "x": 64}
        # The tile is at least the training patch shape, also on the CPU, where the VRAM tiling is smaller.
        # The CPU uses a smaller halo, because a large halo makes the prediction slow.
        expected = {"tile": self.patch_shape, "halo": {"z": 4, "y": 16, "x": 16}}
        self.assertEqual(self._get_tiling(patch_shape=self.patch_shape), expected)
        self.assertEqual(self._get_tiling(12, patch_shape=self.patch_shape)["halo"], halo)
        self.assertEqual(self._get_tiling(12, patch_shape=self.patch_shape)["tile"], self.patch_shape)
        # With enough VRAM it grows by at most 20%, rounded down to the pooling factors (8 in z, 16 in y and x).
        expected = {"tile": {"z": 32, "y": 304, "x": 304}, "halo": halo}
        self.assertEqual(self._get_tiling(80, patch_shape=self.patch_shape), expected)
        expected = {"z": 72, "y": 608, "x": 608}
        self.assertEqual(self._get_tiling(80, patch_shape={"z": 64, "y": 512, "x": 512})["tile"], expected)
        # A UNETR must not grow beyond the training patch shape.
        self.assertEqual(self._get_tiling(80, patch_shape=self.patch_shape, max_growth=1.0)["tile"], self.patch_shape)
        # 2d models predict single slices.
        expected = {"tile": {"z": 1, "y": 304, "x": 304}, "halo": {"z": 0, "y": 16, "x": 16}}
        self.assertEqual(self._get_tiling(patch_shape={"y": 256, "x": 256}), expected)
        self.assertEqual(self._get_tiling(12, patch_shape={"y": 256, "x": 256})["halo"], {"z": 0, "y": 64, "x": 64})
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


class TestMitochondriaSegmentation(unittest.TestCase):
    # Test that a mitochondrion does not flood along the boundary band of a touching mitochondrion.
    def test_touching_mitochondria(self):
        import numpy as np
        from synapse_net.inference.mitochondria import _run_segmentation

        # Two balls with noisy foreground and boundary predictions and overlapping boundary bands.
        shape, radius, band_width = (64, 80, 128), 18, 4
        coords = np.indices(shape)
        dists = [np.linalg.norm(coords - np.array([32, 40, x])[:, None, None, None], axis=0) for x in (40, 78)]
        foreground, boundaries = np.zeros(shape, "float32"), np.zeros(shape, "float32")
        for dist in dists:
            foreground[dist < radius] = 1.0
            band = (dist >= radius) & (dist < radius + band_width)
            foreground[band] = np.maximum(foreground[band], 0.5)
            boundaries[band] = 0.9
        rng = np.random.default_rng(0)
        foreground = np.clip(foreground + (foreground > 0) * 0.1 * rng.standard_normal(shape, dtype="float32"), 0, 1)
        boundaries = np.clip(boundaries + (boundaries > 0) * 0.1 * rng.standard_normal(shape, dtype="float32"), 0, 1)

        seg = _run_segmentation(foreground, boundaries, verbose=False, min_size=100, area_threshold=0)
        insides = [dist < radius + band_width for dist in dists]
        for this, other, dist in zip(insides, insides[::-1], dists[::-1]):
            other_id = np.bincount(seg[dist < 6]).argmax()
            # Without a graded boundary band, the other mitochondrion covered about 45% of this one.
            self.assertLess((seg[this & ~other] == other_id).sum(), 0.01 * this.sum())


if __name__ == "__main__":
    unittest.main()
