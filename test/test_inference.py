import os
import unittest
from functools import partial
from shutil import rmtree

import imageio.v3 as imageio
import requests
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
        from synapse_net.inference.inference import _get_model_registry, get_model_training_resolution

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
