import json
import os
import tempfile
import unittest
from shutil import rmtree

import h5py
import numpy as np
import torch
from skimage.data import binary_blobs


def _identity(raw, labels):
    return raw, labels


class TestTraining(unittest.TestCase):
    """The data, loss and transform of the mitochondria and cristae training, and a short cristae training."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_folder = tempfile.mkdtemp()
        cls.roots = {name: os.path.join(cls.tmp_folder, name) for name in ("ds0", "ds1")}
        rng = np.random.default_rng(42)
        for i in range(8):
            path = os.path.join(cls.roots[f"ds{i % 2}"], "sub", f"tomo-{i}_combined.h5")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # The mitochondria state: 0 = background, 1 = annotated mitochondria, 2 = unannotated mitochondria.
            state = np.zeros((32, 64, 64), dtype="float32")
            state[:, :32], state[:, 32:48] = 1, 2
            raw = rng.normal(size=state.shape).astype("float32")
            cristae = binary_blobs(length=64, n_dim=3, volume_fraction=0.2)[:32].astype("uint8")
            with h5py.File(path, "w") as f:
                f.create_dataset("raw_mitos_combined", data=np.stack([raw, state]))
                f.create_dataset("labels/cristae", data=cristae)

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmp_folder, ignore_errors=True)

    def _path(self, root, i):
        return os.path.join(self.roots[root], "sub", f"tomo-{i}_combined.h5")

    def _split_file(self, split):
        split_file = os.path.join(self.tmp_folder, "split.json")
        with open(split_file, "w") as f:
            json.dump(split, f)
        return split_file

    def test_random_split(self):
        from synapse_net.training.cristae import get_cristae_paths

        train_paths, val_paths = get_cristae_paths(self.roots)
        self.assertEqual((len(train_paths), len(val_paths)), (7, 1))
        self.assertFalse(set(train_paths) & set(val_paths))
        self.assertEqual(get_cristae_paths(self.roots), (train_paths, val_paths))

        train_paths, val_paths = get_cristae_paths(self.roots, exclude=["tomo-0_", "tomo-1_"])
        self.assertEqual(len(train_paths + val_paths), 6)
        with self.assertRaisesRegex(ValueError, "Did not find"):
            get_cristae_paths()

    def test_split_file(self):
        from synapse_net.training.cristae import get_cristae_paths, get_cristae_test_paths

        split_file = self._split_file({
            "roots": self.roots, "train": ["ds0/sub/tomo-0_combined.h5", "ds1/sub/tomo-1_combined.h5"],
            "val": ["ds0/sub/tomo-2_combined.h5"], "test": ["ds1/sub/tomo-3_combined.h5"],
        })
        train_paths, val_paths = get_cristae_paths(split_file=split_file)
        self.assertEqual(train_paths, [self._path("ds0", 0), self._path("ds1", 1)])
        self.assertEqual(val_paths, [self._path("ds0", 2)])

        # The test data is left out of a random split by passing it as exclude. This needs the same paths as the
        # ones found by glob, also on Windows, where the '/' of the entries is not the path separator.
        test_paths = get_cristae_test_paths(split_file)
        self.assertEqual(test_paths, [self._path("ds1", 3)])
        self.assertIn(test_paths[0], sum(get_cristae_paths(self.roots), []))
        self.assertNotIn(test_paths[0], sum(get_cristae_paths(self.roots, exclude=test_paths), []))
        # The roots stored in the split file can be overridden.
        moved = os.path.join("elsewhere", "ds1")
        moved_paths = get_cristae_test_paths(split_file, {"ds1": moved})
        self.assertEqual(moved_paths, [os.path.join(moved, "sub", "tomo-3_combined.h5")])

        for split, error in [({"train": ["ds0/missing.h5"], "val": []}, "do not exist"),
                             ({"train": ["ds2/sub/tomo-0_combined.h5"], "val": []}, "unknown data root")]:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                get_cristae_paths(self.roots, split_file=self._split_file(split))

    def test_mitochondria_split_file(self):
        from synapse_net.training.mitochondria import get_mitochondria_paths

        split_file = self._split_file({"train": ["ds0/sub/tomo-0_combined.h5"], "val": ["ds1/sub/tomo-1_combined.h5"]})
        train_paths, val_paths = get_mitochondria_paths(self.tmp_folder, split_file=split_file)
        self.assertEqual((train_paths, val_paths), ([self._path("ds0", 0)], [self._path("ds1", 1)]))
        self.assertTrue(set(train_paths + val_paths) <= set(sum(get_mitochondria_paths(self.tmp_folder), [])))
        with self.assertRaisesRegex(ValueError, "do not exist"):
            get_mitochondria_paths(self.tmp_folder, split_file=self._split_file({"train": ["missing.h5"], "val": []}))

    def test_masked_dice_loss(self):
        from synapse_net.training.loss import MaskedDiceLossPerSample

        loss = MaskedDiceLossPerSample()
        prediction = torch.rand(3, 2, 4, 8, 8)
        target = (torch.rand(3, 2, 4, 8, 8) > 0.5).float()

        # With a mask of ones this is the dice per sample, averaged over the samples and summed over the channels.
        expected = 0.0
        for c in range(2):
            dice = [2 * (p * t).sum() / ((p * p).sum() + (t * t).sum()) for p, t in zip(prediction[:, c], target[:, c])]
            expected += 1.0 - sum(dice) / 3
        self.assertAlmostEqual(loss(prediction, torch.cat([target, torch.ones_like(target)], dim=1)).item(),
                               expected.item(), places=5)
        # A constant weight cancels out, because of the square root of the mask.
        self.assertAlmostEqual(loss(prediction, torch.cat([target, 3 * torch.ones_like(target)], dim=1)).item(),
                               expected.item(), places=5)

        # Changing the prediction where the mask is zero does not change the loss.
        mask = torch.ones_like(target)
        mask[..., :4] = 0
        perturbed = prediction.clone()
        perturbed[..., :4] = 1.0 - perturbed[..., :4]
        masked_target = torch.cat([target, mask], dim=1)
        self.assertAlmostEqual(loss(prediction, masked_target).item(), loss(perturbed, masked_target).item(), places=5)

        with self.assertRaisesRegex(ValueError, "Expected a target"):
            loss(prediction, target)

    def test_mask_transform(self):
        from synapse_net.training.transform import AugmentedMitoStateMaskTransform

        # The data has a leading batch axis, as returned by the augmentations.
        raw = np.zeros((1, 2, 16, 32, 32), dtype="float32")
        raw[0, 1, 4:12, 8:24, 8:24] = 1
        raw[0, 1, :2] = 2
        labels = np.zeros((1, 2, 16, 32, 32), dtype="float32")
        labels[0, 0, 4:12, 10:22, 10:22] = 1

        for w_pos, w_neg, values in [(1.0, 1.0, {0.0, 1.0}), (3.0, 2.0, {0.0, 1.0, 2.0, 3.0})]:
            with self.subTest(w_pos=w_pos, w_neg=w_neg):
                transform = AugmentedMitoStateMaskTransform(_identity, w_pos=w_pos, w_neg=w_neg)
                out_raw, out_labels = transform(raw, labels)
                torch.testing.assert_close(out_raw, torch.as_tensor(raw))
                self.assertEqual(tuple(out_labels.shape), (1, 4, 16, 32, 32))
                torch.testing.assert_close(out_labels[:, :2], torch.as_tensor(labels))
                mask = out_labels[0, 2].numpy()
                np.testing.assert_array_equal(mask, out_labels[0, 3].numpy())
                # Zero in the unannotated mitochondria, and the weights in the membrane band.
                self.assertTrue((mask[raw[0, 1] == 2] == 0).all())
                self.assertEqual(set(np.unique(mask).tolist()), values)

    def test_standardize_channel(self):
        from synapse_net.training.transform import standardize_channel

        raw = np.stack([np.random.rand(4, 8, 8) * 100, np.full((4, 8, 8), 2.0)]).astype("float32")
        out = standardize_channel(raw)
        self.assertAlmostEqual(float(out[0].mean()), 0.0, places=4)
        self.assertAlmostEqual(float(out[0].std()), 1.0, places=3)
        # The state channel keeps its values, so that it can be compared against them.
        np.testing.assert_array_equal(out[1], raw[1])

    def test_cristae_training(self):
        from synapse_net.training.cristae import cristae_training, get_cristae_paths

        train_paths, val_paths = get_cristae_paths(self.roots)
        kwargs = dict(name="cristae", train_paths=train_paths, val_paths=val_paths, save_root=self.tmp_folder,
                      patch_shape=(16, 32, 32), batch_size=1, num_workers=0)
        latest = os.path.join(self.tmp_folder, "checkpoints", "cristae", "latest.pt")
        cristae_training(n_iterations=2, **kwargs)
        self.assertEqual(torch.load(latest, weights_only=False)["iteration"], 2)
        # Resuming continues the run, and `n_iterations` includes the iterations it already did.
        cristae_training(n_iterations=3, resume=True, **kwargs)
        self.assertEqual(torch.load(latest, weights_only=False)["iteration"], 3)


if __name__ == "__main__":
    unittest.main()
