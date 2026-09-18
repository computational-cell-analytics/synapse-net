import json
import os
import platform
import unittest
from shutil import rmtree
from subprocess import run

import h5py
import numpy as np
import torch
from skimage.data import binary_blobs
from unittest import mock


class TestCristaeTraining(unittest.TestCase):
    tmp_folder = "./tmp_cristae_training"
    roots = {"ds0": os.path.join(tmp_folder, "ds0"), "ds1": os.path.join(tmp_folder, "ds1")}

    def setUp(self):
        rng = np.random.default_rng(42)
        for i in range(8):
            root = self.roots["ds0"] if i % 2 == 0 else self.roots["ds1"]
            path = os.path.join(root, "sub", f"tomo-{i}_combined.h5")
            os.makedirs(os.path.dirname(path), exist_ok=True)

            raw = rng.normal(size=(32, 64, 64)).astype("float32")
            # The mitochondria state: 0 = background, 1 = annotated mito, 2 = unannotated mito.
            state = np.zeros((32, 64, 64), dtype="float32")
            state[:, :32] = 1
            state[:, 32:48] = 2
            cristae = binary_blobs(length=64, n_dim=3, volume_fraction=0.2)[:32].astype("uint8")

            with h5py.File(path, "a") as f:
                f.create_dataset("raw_mitos_combined", data=np.stack([raw, state]))
                f.create_dataset("labels/cristae", data=cristae)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    # Data discovery and splitting.

    def test_get_cristae_paths(self):
        from synapse_net.training.cristae import get_cristae_paths

        train_paths, val_paths = get_cristae_paths(self.roots)
        self.assertEqual(len(train_paths), 7)
        self.assertEqual(len(val_paths), 1)
        self.assertEqual(len(set(train_paths) & set(val_paths)), 0)
        for path in train_paths + val_paths:
            self.assertTrue(os.path.exists(path))

        # The split has to be deterministic for a fixed seed.
        self.assertEqual(get_cristae_paths(self.roots), (train_paths, val_paths))

    def test_get_cristae_paths_with_exclude(self):
        from synapse_net.training.cristae import get_cristae_paths

        train_paths, val_paths = get_cristae_paths(self.roots, exclude=["tomo-0_", "tomo-1_"])
        self.assertEqual(len(train_paths) + len(val_paths), 6)
        self.assertFalse(any("tomo-0_" in path or "tomo-1_" in path for path in train_paths + val_paths))

    def test_get_cristae_paths_from_split_file(self):
        from synapse_net.training.cristae import get_cristae_paths

        split = {"train": ["ds0/sub/tomo-0_combined.h5", "ds1/sub/tomo-1_combined.h5"],
                 "val": ["ds0/sub/tomo-2_combined.h5"]}
        split_file = os.path.join(self.tmp_folder, "split.json")
        with open(split_file, "w") as f:
            json.dump(split, f)

        train_paths, val_paths = get_cristae_paths(self.roots, split_file=split_file)
        self.assertEqual(train_paths, [os.path.join(self.tmp_folder, name) for name in split["train"]])
        self.assertEqual(val_paths, [os.path.join(self.tmp_folder, name) for name in split["val"]])

    def test_get_cristae_paths_from_split_file_with_roots(self):
        from synapse_net.training.cristae import get_cristae_paths

        # A split file that carries its own roots can be used without passing them.
        split = {"roots": self.roots,
                 "train": ["ds0/sub/tomo-0_combined.h5"], "val": ["ds1/sub/tomo-1_combined.h5"]}
        split_file = os.path.join(self.tmp_folder, "split.json")
        with open(split_file, "w") as f:
            json.dump(split, f)

        train_paths, val_paths = get_cristae_paths(split_file=split_file)
        self.assertEqual(train_paths, [os.path.join(self.roots["ds0"], "sub", "tomo-0_combined.h5")])
        self.assertEqual(val_paths, [os.path.join(self.roots["ds1"], "sub", "tomo-1_combined.h5")])

        # Roots passed explicitly take precedence, so that the data can be moved.
        moved = os.path.join(self.tmp_folder, "moved")
        os.makedirs(os.path.join(moved, "sub"), exist_ok=True)
        open(os.path.join(moved, "sub", "tomo-0_combined.h5"), "w").close()
        train_paths, _ = get_cristae_paths({"ds0": moved}, split_file=split_file)
        self.assertEqual(train_paths, [os.path.join(moved, "sub", "tomo-0_combined.h5")])

    def test_get_cristae_paths_without_roots(self):
        from synapse_net.training.cristae import get_cristae_paths

        with self.assertRaisesRegex(ValueError, "No data roots"):
            get_cristae_paths()

    def test_get_cristae_paths_from_invalid_split_file(self):
        from synapse_net.training.cristae import get_cristae_paths

        split_file = os.path.join(self.tmp_folder, "split.json")
        with open(split_file, "w") as f:
            json.dump({"train": ["ds0/sub/tomo-0_combined.h5"], "val": ["ds0/does-not-exist.h5"]}, f)
        with self.assertRaisesRegex(ValueError, "do not exist"):
            get_cristae_paths(self.roots, split_file=split_file)

        with open(split_file, "w") as f:
            json.dump({"train": ["nope/tomo-0_combined.h5"], "val": []}, f)
        with self.assertRaisesRegex(ValueError, "unknown data root"):
            get_cristae_paths(self.roots, split_file=split_file)

    def test_get_cristae_paths_without_data(self):
        from synapse_net.training.cristae import get_cristae_paths

        with self.assertRaisesRegex(ValueError, "Did not find any files"):
            get_cristae_paths(self.roots, file_pattern="*.mrc")

    # The masked losses.

    def test_masked_dice_loss_per_sample(self):
        from synapse_net.training.loss import MaskedDiceLossPerSample

        loss = MaskedDiceLossPerSample()
        prediction = torch.rand(3, 2, 4, 8, 8)
        target = (torch.rand(3, 2, 4, 8, 8) > 0.5).float()

        # With an all-one mask this is the plain per-sample dice.
        full_mask = torch.cat([target, torch.ones_like(target)], dim=1)
        expected = 0.0
        for channel in range(2):
            dice = 0.0
            for sample in range(3):
                pred, tgt = prediction[sample, channel], target[sample, channel]
                dice += 2 * (pred * tgt).sum() / ((pred * pred).sum() + (tgt * tgt).sum())
            expected += 1.0 - dice / 3
        self.assertAlmostEqual(loss(prediction, full_mask).item(), expected.item(), places=5)

        # Masked-out voxels must not influence the loss: changing the prediction there is a no-op.
        mask = torch.ones_like(target)
        mask[..., :4] = 0
        masked_target = torch.cat([target, mask], dim=1)
        reference = loss(prediction, masked_target)
        perturbed = prediction.clone()
        perturbed[..., :4] = 1.0 - perturbed[..., :4]
        self.assertAlmostEqual(loss(perturbed, masked_target).item(), reference.item(), places=5)

        # A target with the wrong number of channels is rejected.
        with self.assertRaisesRegex(ValueError, "expects a target"):
            loss(prediction, target)

    def test_masked_dice_loss_weighting(self):
        from synapse_net.training.loss import MaskedDiceLossPerSample

        # A constant weight w has to cancel out of the dice, since it scales the numerator and the
        # denominator alike. This is what the sqrt of the mask is for.
        loss = MaskedDiceLossPerSample()
        prediction = torch.rand(2, 1, 4, 8, 8)
        target = (torch.rand(2, 1, 4, 8, 8) > 0.5).float()
        unweighted = loss(prediction, torch.cat([target, torch.ones_like(target)], dim=1))
        weighted = loss(prediction, torch.cat([target, 3.0 * torch.ones_like(target)], dim=1))
        self.assertAlmostEqual(unweighted.item(), weighted.item(), places=5)

    # The membrane weighting and the mask transforms.

    def test_membrane_proximity_weight(self):
        from synapse_net.training.transform import membrane_proximity_weight

        state = np.zeros((16, 32, 32), dtype="float32")
        state[4:12, 8:24, 8:24] = 1  # an annotated mitochondrion
        state[:2] = 2                # an unannotated mitochondrion
        foreground = np.zeros_like(state)
        foreground[4:12, 10:22, 10:22] = 1

        # Unit weights give exactly the binary mask, without computing a distance transform.
        unweighted = membrane_proximity_weight(state, foreground, w_pos=1.0, w_neg=1.0)
        np.testing.assert_array_equal(unweighted, (state != 2).astype("float32"))

        weighted = membrane_proximity_weight(state, foreground, w_pos=3.0, w_neg=2.0, band_nm=12.0)
        # Excluded voxels stay at zero whatever the weights are.
        self.assertTrue((weighted[state == 2] == 0).all())
        # Only the weights 1, w_pos and w_neg occur, and both weights are actually applied.
        self.assertEqual(set(np.unique(weighted).tolist()), {0.0, 1.0, 2.0, 3.0})

    def test_mito_state_mask_transform(self):
        from synapse_net.training.transform import MitoStateMaskTransform

        raw = np.zeros((2, 8, 16, 16), dtype="float32")
        raw[1, :, :8] = 1
        raw[1, :, 8:12] = 2
        labels = np.zeros((2, 8, 16, 16), dtype="float32")
        labels[0, :, 2:6] = 1

        transform = MitoStateMaskTransform()
        out_raw, out_labels = transform(raw, labels)
        # The raw data is unchanged and the labels gain one mask channel per label channel.
        np.testing.assert_array_equal(out_raw, raw)
        self.assertEqual(out_labels.shape, (4, 8, 16, 16))
        np.testing.assert_array_equal(out_labels[:2], labels)
        np.testing.assert_array_equal(out_labels[2], (raw[1] != 2).astype("float32"))
        np.testing.assert_array_equal(out_labels[2], out_labels[3])

    def test_augmented_mito_state_mask_transform(self):
        from synapse_net.training.transform import AugmentedMitoStateMaskTransform

        raw = np.zeros((2, 8, 16, 16), dtype="float32")
        raw[1, :, :8] = 1
        raw[1, :, 8:12] = 2
        labels = np.zeros((2, 8, 16, 16), dtype="float32")

        # Use the identity as augmentation, so that the output can be compared directly.
        transform = AugmentedMitoStateMaskTransform(lambda x, y: (x, y))
        out_raw, out_labels = transform(raw, labels)
        self.assertEqual(tuple(out_labels.shape), (4, 8, 16, 16))
        expected = torch.as_tensor((raw[1] != 2).astype("float32"))
        torch.testing.assert_close(out_labels[2], expected)
        torch.testing.assert_close(out_labels[3], expected)
        torch.testing.assert_close(out_raw, torch.as_tensor(raw))

    def test_standardize_channel(self):
        from synapse_net.training.transform import standardize_channel

        raw = np.stack([np.random.rand(4, 8, 8) * 100, np.full((4, 8, 8), 2.0)]).astype("float32")
        out = standardize_channel(raw)
        self.assertAlmostEqual(float(out[0].mean()), 0.0, places=4)
        self.assertAlmostEqual(float(out[0].std()), 1.0, places=3)
        # The state channel must keep its exact values, so that it can be compared against them.
        np.testing.assert_array_equal(out[1], raw[1])
        with self.assertRaisesRegex(ValueError, "4 dimensions"):
            standardize_channel(raw[0])

    # The recipe.

    def test_recipe(self):
        from synapse_net.training.cristae import cristae_training
        from synapse_net.training.loss import MaskedDiceLossPerSample
        from synapse_net.training.transform import AugmentedMitoStateMaskTransform, standardize_channel

        with mock.patch("synapse_net.training.cristae.supervised_training") as training:
            cristae_training(name="test-model", train_paths=["a.h5"], val_paths=["b.h5"])

        kwargs = training.call_args.kwargs
        self.assertEqual(kwargs["raw_key"], "raw_mitos_combined")
        self.assertEqual(kwargs["label_key"], "labels/cristae")
        self.assertEqual(kwargs["patch_shape"], (32, 256, 256))
        self.assertEqual(kwargs["batch_size"], 24)
        self.assertAlmostEqual(kwargs["lr"], 1.7e-4)
        self.assertEqual(kwargs["in_channels"], 2)
        self.assertEqual(kwargs["out_channels"], 2)
        self.assertEqual(kwargs["initial_features"], 32)
        self.assertIsNone(kwargs["norm"])
        self.assertIs(kwargs["with_channels"], True)
        self.assertEqual(kwargs["early_stopping"], 25)
        self.assertEqual(kwargs["log_image_interval"], 50)
        self.assertIs(kwargs["mixed_precision"], True)
        self.assertIs(kwargs["shuffle"], False)
        self.assertEqual(kwargs["num_workers"], 8)
        self.assertIs(kwargs["persistent_workers"], True)
        self.assertEqual(kwargs["prefetch_factor"], 4)
        self.assertIs(kwargs["raw_transform"], standardize_channel)
        self.assertEqual(kwargs["sampler"].p_reject, 0.95)
        self.assertIsInstance(kwargs["loss_fn"], MaskedDiceLossPerSample)

        transform = kwargs["transform"]
        self.assertIsInstance(transform, AugmentedMitoStateMaskTransform)
        self.assertEqual(transform.w_pos, 3.0)
        self.assertEqual(transform.w_neg, 2.0)
        self.assertEqual(transform.band_nm, 12.0)
        self.assertEqual(transform.mito_channel, 1)
        self.assertEqual(transform.exclude_state_value, 2.0)

    def test_recipe_without_augmentations(self):
        from synapse_net.training.cristae import cristae_training
        from synapse_net.training.transform import MitoStateMaskTransform

        with mock.patch("synapse_net.training.cristae.supervised_training") as training:
            cristae_training(name="test-model", train_paths=["a.h5"], val_paths=["b.h5"], augmentations=False)
        transform = training.call_args.kwargs["transform"]
        self.assertIsInstance(transform, MitoStateMaskTransform)
        self.assertNotIsInstance(transform, type(None))

    @unittest.skipIf(platform.system() == "Windows", "CLI does not work on Windows")
    def test_cristae_training_cli(self):
        name = "test-cristae-model"
        cmd = [
            "synapse_net.run_cristae_training",
            "-n", name,
            "-i", self.roots["ds0"], self.roots["ds1"],
            "--patch_shape", "16", "32", "32",
            "--batch_size", "1",
            "--initial_features", "4",
            "--n_samples_train", "4",
            "--n_samples_val", "1",
            "--n_iterations", "4",
            "--num_workers", "1",
            "--save_root", self.tmp_folder,
        ]
        run(cmd, check=True)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_folder, "checkpoints", name, "latest.pt")))


if __name__ == "__main__":
    unittest.main()
