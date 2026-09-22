import json
import os
import pickle
import platform
import unittest
from shutil import rmtree
from subprocess import run
from unittest import mock

import h5py
import numpy as np
import torch_em
from skimage.data import binary_blobs
from skimage.measure import label


class TestVolEmMitochondriaTraining(unittest.TestCase):
    tmp_folder = "./tmp_vol_em_mito_training"
    roots = {"4007": os.path.join(tmp_folder, "4007"), "4009": os.path.join(tmp_folder, "4009")}

    def setUp(self):
        for i in range(8):
            # The 4007 blocks sit in sub-folders and the 4009 blocks do not, as in the real data.
            if i % 2 == 0:
                path = os.path.join(self.roots["4007"], f"cutout_{i}", f"block-{i}.h5")
            else:
                path = os.path.join(self.roots["4009"], f"block-{i}.h5")
            os.makedirs(os.path.dirname(path), exist_ok=True)

            raw = (binary_blobs(length=64, n_dim=3, volume_fraction=0.15) * 200).astype("uint8")
            # A white filler border, so that the raw transform has something to remove.
            raw[:, :, :8] = 255
            labels = label(raw > 0).astype("uint16")

            with h5py.File(path, "a") as f:
                f.create_dataset("raw", data=raw)
                f.create_dataset("labels/mitochondria", data=labels)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    # Data discovery and splitting.

    def test_get_vol_em_mitochondria_paths(self):
        from synapse_net.training.mitochondria_vol_em import get_vol_em_mitochondria_paths

        train_paths, val_paths = get_vol_em_mitochondria_paths(self.roots)
        self.assertEqual(len(train_paths), 7)
        self.assertEqual(len(val_paths), 1)
        self.assertEqual(len(set(train_paths) & set(val_paths)), 0)
        for path in train_paths + val_paths:
            self.assertTrue(os.path.exists(path))

        # The split has to be deterministic for a fixed seed, and has to change with the seed.
        self.assertEqual(get_vol_em_mitochondria_paths(self.roots), (train_paths, val_paths))
        self.assertNotEqual(get_vol_em_mitochondria_paths(self.roots, seed=1), (train_paths, val_paths))

    def test_get_vol_em_mitochondria_paths_with_exclude(self):
        from synapse_net.training.mitochondria_vol_em import get_vol_em_mitochondria_paths

        train_paths, val_paths = get_vol_em_mitochondria_paths(self.roots, exclude=["block-0.", "block-1."])
        self.assertEqual(len(train_paths) + len(val_paths), 6)
        self.assertFalse(any("block-0." in path or "block-1." in path for path in train_paths + val_paths))

    def test_get_vol_em_mitochondria_paths_from_split_file(self):
        from synapse_net.training.mitochondria_vol_em import get_vol_em_mitochondria_paths

        split = {"train": ["4007/cutout_0/block-0.h5", "4009/block-1.h5"], "val": ["4007/cutout_2/block-2.h5"]}
        split_file = os.path.join(self.tmp_folder, "split.json")
        with open(split_file, "w") as f:
            json.dump(split, f)

        train_paths, val_paths = get_vol_em_mitochondria_paths(self.roots, split_file=split_file)
        self.assertEqual(train_paths, [os.path.join(self.tmp_folder, name) for name in split["train"]])
        self.assertEqual(val_paths, [os.path.join(self.tmp_folder, name) for name in split["val"]])

    def test_get_vol_em_mitochondria_paths_from_split_file_with_roots(self):
        from synapse_net.training.mitochondria_vol_em import get_vol_em_mitochondria_paths

        # A split file that carries its own roots can be used without passing them.
        split = {"roots": self.roots, "train": ["4007/cutout_0/block-0.h5"], "val": ["4009/block-1.h5"]}
        split_file = os.path.join(self.tmp_folder, "split.json")
        with open(split_file, "w") as f:
            json.dump(split, f)

        train_paths, val_paths = get_vol_em_mitochondria_paths(split_file=split_file)
        self.assertEqual(train_paths, [os.path.join(self.roots["4007"], "cutout_0", "block-0.h5")])
        self.assertEqual(val_paths, [os.path.join(self.roots["4009"], "block-1.h5")])

        # Roots passed explicitly take precedence, so that the data can be moved.
        moved = os.path.join(self.tmp_folder, "moved")
        os.makedirs(os.path.join(moved, "cutout_0"), exist_ok=True)
        open(os.path.join(moved, "cutout_0", "block-0.h5"), "w").close()
        train_paths, _ = get_vol_em_mitochondria_paths({"4007": moved}, split_file=split_file)
        self.assertEqual(train_paths, [os.path.join(moved, "cutout_0", "block-0.h5")])

    def test_get_vol_em_mitochondria_paths_without_roots(self):
        from synapse_net.training.mitochondria_vol_em import get_vol_em_mitochondria_paths

        with self.assertRaisesRegex(ValueError, "No data roots"):
            get_vol_em_mitochondria_paths()

    def test_get_vol_em_mitochondria_paths_from_invalid_split_file(self):
        from synapse_net.training.mitochondria_vol_em import get_vol_em_mitochondria_paths

        split_file = os.path.join(self.tmp_folder, "split.json")
        with open(split_file, "w") as f:
            json.dump({"train": ["4007/cutout_0/block-0.h5"], "val": ["4007/does-not-exist.h5"]}, f)
        with self.assertRaisesRegex(ValueError, "do not exist"):
            get_vol_em_mitochondria_paths(self.roots, split_file=split_file)

        with open(split_file, "w") as f:
            json.dump({"train": ["nope/block-0.h5"], "val": []}, f)
        with self.assertRaisesRegex(ValueError, "unknown data root"):
            get_vol_em_mitochondria_paths(self.roots, split_file=split_file)

    def test_get_vol_em_mitochondria_paths_without_data(self):
        from synapse_net.training.mitochondria_vol_em import get_vol_em_mitochondria_paths

        with self.assertRaisesRegex(ValueError, "Did not find any files"):
            get_vol_em_mitochondria_paths(self.roots, file_pattern="*.mrc")

    # The white patch removal.

    def test_remove_white_patches(self):
        from synapse_net.training.transform import remove_white_patches

        raw = np.full((8, 16, 16), 100, dtype="uint8")
        raw[:, :, :4] = 255  # The filler border, well above the minimal size.
        raw[0, 0, 10] = 255  # A single saturated voxel, which is sample structure.

        out = remove_white_patches(raw)
        self.assertTrue((out[:, :, :4] == 0).all())
        self.assertEqual(out[0, 0, 10], 255)
        self.assertTrue((out[:, :, 4:10] == 100).all())
        self.assertEqual(out.dtype, raw.dtype)
        # The input must not be modified in place.
        self.assertTrue((raw[:, :, :4] == 255).all())

    def test_remove_white_patches_size_threshold(self):
        from synapse_net.training.transform import remove_white_patches

        # The threshold is inclusive: a component of exactly 'min_size' voxels is removed.
        raw = np.zeros((4, 8, 8), dtype="uint8")
        raw[0, 0, :4] = 255
        self.assertTrue((remove_white_patches(raw, min_size=4) == 0).all())
        self.assertEqual((remove_white_patches(raw, min_size=5) == 255).sum(), 4)

    def test_remove_white_patches_connectivity(self):
        from synapse_net.training.transform import remove_white_patches

        # Two blocks that only touch diagonally form one component under full connectivity, so both
        # are removed. With the 6-connectivity of 'scipy.ndimage.label' neither would be.
        raw = np.zeros((4, 8, 8), dtype="uint8")
        raw[0, 0:2, 0:2] = 255
        raw[0, 2:4, 2:4] = 255
        self.assertEqual((remove_white_patches(raw, min_size=8) == 255).sum(), 0)
        self.assertEqual((remove_white_patches(raw, min_size=9) == 255).sum(), 8)

    def test_remove_white_patches_without_filler(self):
        from synapse_net.training.transform import remove_white_patches

        raw = np.full((4, 8, 8), 100, dtype="uint8")
        self.assertIs(remove_white_patches(raw), raw)

    def test_remove_white_patches_rejects_normalized_data(self):
        from synapse_net.training.transform import remove_white_patches

        raw = np.random.rand(4, 8, 8).astype("float32")
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            remove_white_patches(raw)

        with self.assertRaisesRegex(ValueError, "2d or 3d"):
            remove_white_patches(np.zeros((2, 4, 8, 8), dtype="uint8"))

    def test_remove_white_patches_and_normalize(self):
        from synapse_net.training.transform import RemoveWhitePatchesAndNormalize, remove_white_patches

        raw = (binary_blobs(length=32, n_dim=3, volume_fraction=0.2) * 200).astype("uint8")
        raw[:, :, :4] = 255

        transform = RemoveWhitePatchesAndNormalize()
        expected = torch_em.transform.raw.normalize_percentile(remove_white_patches(raw))
        self.assertTrue(np.allclose(transform(raw), expected))

    def test_remove_white_patches_and_normalize_is_picklable(self):
        from synapse_net.training.transform import RemoveWhitePatchesAndNormalize

        # The dataloader workers and the training checkpoint both pickle the raw transform.
        transform = pickle.loads(pickle.dumps(RemoveWhitePatchesAndNormalize(min_size=7)))
        self.assertEqual(transform.min_size, 7)

    # The recipe.

    def test_recipe(self):
        # Check that the hyperparameters of the published volume EM mitochondria model are passed on.
        from synapse_net.training.mitochondria_vol_em import vol_em_mitochondria_training
        from synapse_net.training.transform import RemoveWhitePatchesAndNormalize

        with mock.patch("synapse_net.training.mitochondria_vol_em.supervised_training") as training:
            vol_em_mitochondria_training(name="test-model", train_paths=["a.h5"], val_paths=["b.h5"])

        kwargs = training.call_args.kwargs
        self.assertEqual(kwargs["label_key"], "labels/mitochondria")
        self.assertEqual(kwargs["raw_key"], "raw")
        self.assertEqual(kwargs["patch_shape"], (32, 512, 512))
        self.assertEqual(kwargs["batch_size"], 4)
        self.assertEqual(kwargs["lr"], 1e-4)
        self.assertEqual(kwargs["n_iterations"], 50000)
        self.assertEqual(kwargs["n_samples_train"], 500)
        self.assertEqual(kwargs["n_samples_val"], 500)
        self.assertEqual(kwargs["initial_features"], 32)
        self.assertEqual(kwargs["scale_factors"], [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]])
        self.assertIsNone(kwargs["norm"])
        self.assertEqual(kwargs["num_workers"], 8)
        self.assertEqual(kwargs["early_stopping"], 20)
        self.assertEqual(kwargs["log_image_interval"], 50)
        self.assertIs(kwargs["mixed_precision"], False)
        self.assertIs(kwargs["shuffle"], False)
        self.assertIsInstance(kwargs["raw_transform"], RemoveWhitePatchesAndNormalize)
        self.assertEqual(kwargs["raw_transform"].min_size, 20)
        self.assertEqual(kwargs["sampler"].p_reject, 0.95)

    def test_recipe_without_white_patch_fix(self):
        from synapse_net.training.mitochondria_vol_em import vol_em_mitochondria_training

        with mock.patch("synapse_net.training.mitochondria_vol_em.supervised_training") as training:
            vol_em_mitochondria_training(name="test-model", train_paths=["a.h5"], val_paths=["b.h5"],
                                         fix_white_patches=False)
        raw_transform = training.call_args.kwargs["raw_transform"]
        self.assertIs(raw_transform, torch_em.transform.raw.normalize_percentile)

    @unittest.skipIf(platform.system() == "Windows", "CLI does not work on Windows")
    def test_vol_em_mitochondria_training_cli(self):
        name = "test-vol-em-mito-model"
        cmd = [
            "synapse_net.run_vol_em_mitochondria_training",
            "-n", name,
            "-i", self.roots["4007"], self.roots["4009"],
            "--patch_shape", "8", "64", "64",
            "--batch_size", "1",
            "--initial_features", "4",
            "--n_samples_train", "5",
            "--n_samples_val", "1",
            "--n_iterations", "6",
            "--num_workers", "1",
            "--save_root", self.tmp_folder,
        ]
        run(cmd, check=True)

        # Check that the checkpoint exists.
        ckpt_path = os.path.join(self.tmp_folder, "checkpoints", name, "latest.pt")
        self.assertTrue(os.path.exists(ckpt_path))


if __name__ == "__main__":
    unittest.main()
