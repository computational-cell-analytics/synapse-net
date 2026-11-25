import json
import os
import pickle
import tempfile
import unittest
from shutil import rmtree
from unittest import mock

import numpy as np


class TestVolEmMitochondria(unittest.TestCase):
    """The data, transform and recipe of the volume EM mitochondria training, and its inference preprocessing."""

    def setUp(self):
        self.tmp_folder = tempfile.mkdtemp()

    def tearDown(self):
        rmtree(self.tmp_folder, ignore_errors=True)

    def test_split_file(self):
        from synapse_net.training.mitochondria_vol_em import get_vol_em_mitochondria_paths

        root = os.path.join(self.tmp_folder, "4007")
        os.makedirs(os.path.join(root, "sub"))
        for name in ("a.h5", "b.h5"):
            open(os.path.join(root, "sub", name), "w").close()
        split_file = os.path.join(self.tmp_folder, "split.json")
        with open(split_file, "w") as f:
            json.dump({"roots": {"4007": root}, "train": ["4007/sub/a.h5"], "val": ["4007/sub/b.h5"]}, f)

        train_paths, val_paths = get_vol_em_mitochondria_paths(split_file=split_file)
        self.assertEqual(train_paths, [os.path.join(root, "sub", "a.h5")])
        self.assertEqual(val_paths, [os.path.join(root, "sub", "b.h5")])
        self.assertEqual(sum(get_vol_em_mitochondria_paths({"4007": root}), []).__len__(), 2)
        with self.assertRaisesRegex(ValueError, "do not exist"):
            get_vol_em_mitochondria_paths({"4007": "elsewhere"}, split_file=split_file)

    def test_remove_white_patches(self):
        from synapse_net.inference.mitochondria import remove_white_patches

        raw = np.full((4, 16, 16), 100, dtype="uint8")
        raw[:, :, :4] = 255  # filler
        raw[0, 8, 8] = 255  # a small saturated structure that is kept
        fixed = remove_white_patches(raw)
        self.assertTrue((fixed[:, :, :4] == 0).all())
        self.assertEqual(fixed[0, 8, 8], 255)
        self.assertEqual(raw[0, 0, 0], 255)  # the input is not modified
        no_filler = np.full((4, 16, 16), 100, dtype="uint8")
        self.assertIs(remove_white_patches(no_filler), no_filler)
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            remove_white_patches(raw.astype("float32"))

    def test_raw_transform_is_picklable(self):
        from synapse_net.training.transform import RemoveWhitePatchesAndNormalize

        transform = pickle.loads(pickle.dumps(RemoveWhitePatchesAndNormalize(min_size=5)))
        self.assertEqual(transform.min_size, 5)
        raw = np.random.default_rng(0).integers(0, 200, size=(4, 16, 16)).astype("uint8")
        raw[:, :, :4] = 255
        self.assertLessEqual(transform(raw)[:, :, :4].max(), 0.0)

    def test_recipe(self):
        from synapse_net.training import mitochondria_vol_em
        from synapse_net.training.transform import RemoveWhitePatchesAndNormalize

        with mock.patch.object(mitochondria_vol_em, "supervised_training") as training:
            mitochondria_vol_em.vol_em_mitochondria_training("name", ["a.h5"], ["b.h5"])
        kwargs = training.call_args.kwargs
        self.assertIsNone(kwargs["norm"])
        self.assertEqual(kwargs["scale_factors"], [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]])
        self.assertIsInstance(kwargs["raw_transform"], RemoveWhitePatchesAndNormalize)
        self.assertEqual((kwargs["patch_shape"], kwargs["batch_size"]), ((32, 512, 512), 4))

    def test_model_architecture(self):
        from synapse_net.training.models import _get_model
        from synapse_net.training.mitochondria_vol_em import VOL_EM_SCALE_FACTORS

        model = _get_model((32, 512, 512), 2, norm=None, scale_factors=VOL_EM_SCALE_FACTORS)
        self.assertEqual(model.scale_factors, VOL_EM_SCALE_FACTORS)

    def test_inference_preprocessing(self):
        from torch_em.transform.raw import normalize_percentile
        from synapse_net.inference import mitochondria
        from synapse_net.inference.inference import get_segmentation_function

        self.assertIs(get_segmentation_function("mitochondria_fibsem"), mitochondria.segment_mitochondria_fibsem)
        raw = np.full((4, 16, 16), 100, dtype="uint8")
        raw[:, :, :4] = 255
        with mock.patch.object(mitochondria, "segment_mitochondria") as segment:
            mitochondria.segment_mitochondria_fibsem(raw, model="model")
        self.assertTrue((segment.call_args.args[0][:, :, :4] == 0).all())
        kwargs = segment.call_args.kwargs
        self.assertIs(kwargs["preprocess"], normalize_percentile)
        self.assertEqual((kwargs["min_size"], kwargs["seed_distance"]), (1000, 1))
        self.assertEqual((kwargs["boundary_threshold"], kwargs["area_threshold"]), (0.12, 200))


if __name__ == "__main__":
    unittest.main()
