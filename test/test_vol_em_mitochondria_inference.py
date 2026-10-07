"""The scripts are normally not covered by tests, but the preprocessing contract between the volume EM
mitochondria training and its inference script is the thing that is most likely to rot unnoticed: a
model applied with a different normalization than it was trained with does not predict anything useful,
and it fails silently rather than loudly.
"""

import argparse
import importlib.util
import os
import unittest
from unittest import mock

import torch_em

SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "volume_em", "inference", "run_mitochondria_vol_em_segmentation.py",
)


def _load_script():
    spec = importlib.util.spec_from_file_location("run_mitochondria_vol_em_segmentation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVolEmMitochondriaInference(unittest.TestCase):
    def setUp(self):
        self.script = _load_script()
        self.args = argparse.Namespace(
            input_path="in.h5", output_path="out", model="model.pt", segmentation_key="seg",
            data_ext=".h5", mask_path=None, force=False, scale=None,
            no_white_patch_fix=False, white_patch_min_size=20,
            tile_shape=None, halo=None,
            min_size=self.script.MIN_SIZE, seed_distance=self.script.SEED_DISTANCE,
            boundary_threshold=self.script.BOUNDARY_THRESHOLD, area_threshold=self.script.AREA_THRESHOLD,
            ws_block_shape=[128, 256, 256], ws_halo=[48, 48, 48], verbose=False,
        )

    def _run(self):
        with mock.patch.object(self.script, "inference_helper") as helper:
            self.script.run_mitochondria_segmentation(self.args)
        return helper.call_args

    def test_preprocessing_matches_the_training(self):
        keywords = self._run().args[2].keywords
        # The filler removal must not go into 'preprocess': the volume is standardized before that
        # runs, and the filler is identified by its literal value.
        self.assertIs(keywords["preprocess"], torch_em.transform.raw.normalize_percentile)
        self.assertEqual(keywords["white_patch_min_size"], 20)

    def test_preprocessing_without_the_white_patch_fix(self):
        self.args.no_white_patch_fix = True
        keywords = self._run().args[2].keywords
        self.assertIsNone(keywords["white_patch_min_size"])
        self.assertIs(keywords["preprocess"], torch_em.transform.raw.normalize_percentile)

    def test_the_filler_is_removed_before_the_segmentation(self):
        # The filler has to be gone before 'segment_mitochondria' rescales the volume, which would
        # interpolate it away from its literal value.
        import numpy as np

        raw = np.full((4, 16, 16), 100, dtype="uint8")
        raw[:, :, :4] = 255
        with mock.patch.object(self.script, "segment_mitochondria") as segment:
            self.script.segment_mitochondria_vol_em(raw, white_patch_min_size=20)
        passed = segment.call_args.args[0]
        self.assertTrue((passed[:, :, :4] == 0).all())

        with mock.patch.object(self.script, "segment_mitochondria") as segment:
            self.script.segment_mitochondria_vol_em(raw, white_patch_min_size=None)
        self.assertIs(segment.call_args.args[0], raw)

    def test_post_processing_defaults(self):
        keywords = self._run().args[2].keywords
        self.assertEqual(keywords["min_size"], 1000)
        self.assertEqual(keywords["seed_distance"], 1)
        self.assertEqual(keywords["boundary_threshold"], 0.12)
        self.assertEqual(keywords["area_threshold"], 200)
        self.assertEqual(keywords["model_path"], "model.pt")

    def test_the_data_is_not_rescaled_by_default(self):
        # The model is applied to data at its own training resolution, and the voxel size cannot be
        # derived from hdf5 files, so the scale has to stay unset unless it is passed explicitly.
        kwargs = self._run().kwargs
        self.assertIsNone(kwargs["scale"])
        self.assertEqual(kwargs["data_ext"], ".h5")
        self.assertEqual(kwargs["output_key"], "seg")

    def test_cli_defaults_match_the_script_constants(self):
        with mock.patch("sys.argv", ["script", "-i", "in.h5", "-o", "out", "-m", "model.pt"]), \
             mock.patch.object(self.script, "run_mitochondria_segmentation") as run_segmentation:
            self.script.main()
        args = run_segmentation.call_args.args[0]
        self.assertEqual(args.min_size, self.script.MIN_SIZE)
        self.assertEqual(args.seed_distance, self.script.SEED_DISTANCE)
        self.assertEqual(args.boundary_threshold, self.script.BOUNDARY_THRESHOLD)
        self.assertEqual(args.area_threshold, self.script.AREA_THRESHOLD)
        self.assertEqual(args.segmentation_key, "seg")
        self.assertIs(args.no_white_patch_fix, False)


if __name__ == "__main__":
    unittest.main()
