import json
import os
import platform
import unittest
from shutil import rmtree
from subprocess import run
from unittest import mock

import h5py
import torch_em
from skimage.data import binary_blobs
from skimage.measure import label

from synapse_net.training.mitochondria import (
    _resolve_resume_checkpoint, get_mitochondria_paths, mitochondria_training
)


class TestMitochondriaTraining(unittest.TestCase):
    tmp_folder = "./tmp_mito_training"
    data_folder = os.path.join(tmp_folder, "tomograms")

    def setUp(self):
        os.makedirs(self.data_folder, exist_ok=True)
        # Write the tomograms into two sub-folders, to check that they are found recursively.
        for i in range(8):
            data = binary_blobs(length=64, n_dim=3, volume_fraction=0.15).astype("uint8")
            labels = label(data).astype("uint16")
            path = os.path.join(self.data_folder, f"ds{i % 2}", f"tomo-{i}.h5")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with h5py.File(path, "a") as f:
                f.create_dataset("raw", data=data)
                f.create_dataset("labels/mitochondria", data=labels)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def test_get_mitochondria_paths(self):
        train_paths, val_paths = get_mitochondria_paths(self.data_folder)

        self.assertEqual(len(train_paths), 7)
        self.assertEqual(len(val_paths), 1)
        self.assertEqual(len(set(train_paths) & set(val_paths)), 0)
        for path in train_paths + val_paths:
            self.assertTrue(os.path.exists(path))

        # The split has to be deterministic for a fixed seed, and has to change with the seed.
        self.assertEqual(get_mitochondria_paths(self.data_folder), (train_paths, val_paths))
        self.assertNotEqual(get_mitochondria_paths(self.data_folder, seed=1), (train_paths, val_paths))

    def test_get_mitochondria_paths_from_split_file(self):
        split = {"train": ["ds0/tomo-0.h5", "ds1/tomo-1.h5"], "val": ["ds0/tomo-2.h5"]}
        split_file = os.path.join(self.tmp_folder, "split.json")
        with open(split_file, "w") as f:
            json.dump(split, f)

        train_paths, val_paths = get_mitochondria_paths(self.data_folder, split_file=split_file)
        self.assertEqual(train_paths, [os.path.join(self.data_folder, name) for name in split["train"]])
        self.assertEqual(val_paths, [os.path.join(self.data_folder, name) for name in split["val"]])

    def test_get_mitochondria_paths_from_invalid_split_file(self):
        split_file = os.path.join(self.tmp_folder, "split.json")
        with open(split_file, "w") as f:
            json.dump({"train": ["ds0/tomo-0.h5"], "val": ["does-not-exist.h5"]}, f)

        with self.assertRaisesRegex(ValueError, "do not exist"):
            get_mitochondria_paths(self.data_folder, split_file=split_file)

    def test_get_mitochondria_paths_without_data(self):
        with self.assertRaisesRegex(ValueError, "Did not find any files"):
            get_mitochondria_paths(self.data_folder, file_pattern="*.mrc")

    def test_resolve_resume_checkpoint(self):
        name = "test-model"
        # An explicit checkpoint always wins.
        self.assertEqual(_resolve_resume_checkpoint(self.tmp_folder, name, "/some/checkpoint.pt"),
                         "/some/checkpoint.pt")
        # Without a previous run there is nothing to resume from.
        self.assertIsNone(_resolve_resume_checkpoint(self.tmp_folder, name, None))
        self.assertIsNone(_resolve_resume_checkpoint(None, name, None))
        # With a previous run we resume from its checkpoint folder.
        checkpoint_folder = os.path.join(self.tmp_folder, "checkpoints", name)
        os.makedirs(checkpoint_folder)
        open(os.path.join(checkpoint_folder, "best.pt"), "w").close()
        self.assertEqual(_resolve_resume_checkpoint(self.tmp_folder, name, None), checkpoint_folder)

    def test_recipe(self):
        # Check that the hyperparameters of the published 'mitochondria2' model are passed on.
        with mock.patch("synapse_net.training.mitochondria.supervised_training") as training:
            mitochondria_training(name="test-model", train_paths=["a.h5"], val_paths=["b.h5"])

        kwargs = training.call_args.kwargs
        self.assertEqual(kwargs["label_key"], "labels/mitochondria")
        self.assertEqual(kwargs["raw_key"], "raw")
        self.assertEqual(kwargs["patch_shape"], (32, 256, 256))
        self.assertEqual(kwargs["batch_size"], 8)
        self.assertEqual(kwargs["lr"], 1e-4)
        self.assertEqual(kwargs["n_samples_train"], 500)
        self.assertEqual(kwargs["n_samples_val"], 500)
        self.assertEqual(kwargs["initial_features"], 32)
        self.assertEqual(kwargs["num_workers"], 8)
        self.assertEqual(kwargs["early_stopping"], 20)
        self.assertEqual(kwargs["log_image_interval"], 50)
        self.assertIs(kwargs["mixed_precision"], False)
        self.assertIs(kwargs["shuffle"], False)
        self.assertIs(kwargs["raw_transform"], torch_em.transform.raw.normalize_percentile)
        self.assertEqual(kwargs["sampler"].p_reject, 0.95)

    @unittest.skipIf(platform.system() == "Windows", "CLI does not work on Windows")
    def test_mitochondria_training_cli(self):
        name = "test-mito-model"
        cmd = [
            "synapse_net.run_mitochondria_training",
            "-n", name,
            "-i", self.data_folder,
            "--patch_shape", "32", "64", "64",
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
