import os
import unittest
from shutil import rmtree
from unittest import mock

import h5py
from skimage.data import binary_blobs
from skimage.measure import label

from synapse_net.training.models import microsam_models


class TestTraining(unittest.TestCase):
    tmp_folder = "./tmp_data"
    raw_key = "volume"
    label_key = "labels"

    # Create test folder and sample data.
    def setUp(self):
        n_train = 5
        n_val = 2

        self.train_data = [
            binary_blobs(length=128, n_dim=3, volume_fraction=0.15).astype("uint8") for _ in range(n_train)
        ]
        self.val_data = [
            binary_blobs(length=128, n_dim=3, volume_fraction=0.15).astype("uint8") for _ in range(n_val)
        ]

        self.train_labels = [label(data).astype("uint16") for data in self.train_data]
        self.val_labels = [label(data).astype("uint16") for data in self.val_data]

        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _write_h5_data(self, data, labels, prefix):
        paths = []
        for i, (datum, lab) in enumerate(zip(data, labels)):
            path = os.path.join(self.tmp_folder, f"{prefix}-{i}.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset(self.raw_key, data=datum)
                f.create_dataset(self.label_key, data=lab)
            paths.append(path)
        return paths

    def test_semisupervised_training(self):
        from synapse_net.training.semisupervised_training import semisupervised_training

        train_paths = self._write_h5_data(self.train_data, self.train_labels, "train")
        val_paths = self._write_h5_data(self.val_data, self.val_labels, "val")

        name = "test-semisupervised"
        kwargs = dict(
            name=name,
            unsupervised_train_paths=train_paths,
            unsupervised_val_paths=val_paths,
            patch_shape=(64, 64, 64),
            save_root=self.tmp_folder,
            supervised_train_paths=train_paths,
            supervised_val_paths=val_paths,
            label_key=self.label_key,
            raw_key=self.raw_key,
            batch_size=1,
            n_iterations=6,
            teacher_warmup_iterations=6,
            n_samples_train=5,
            n_samples_val=2,
        )
        semisupervised_training(**kwargs)

        warmup_ckpt = os.path.join(self.tmp_folder, "checkpoints", f"{name}-warmup", "best.pt")
        final_ckpt = os.path.join(self.tmp_folder, "checkpoints", name, "latest.pt")
        self.assertTrue(os.path.exists(warmup_ckpt))
        self.assertTrue(os.path.exists(final_ckpt))

        # The second run must reuse the warmup checkpoint instead of training the teacher again.
        with mock.patch("synapse_net.training.semisupervised_training.supervised_training") as warmup:
            semisupervised_training(**kwargs)
        warmup.assert_not_called()
        self.assertTrue(os.path.exists(final_ckpt))

    @unittest.skipIf(os.environ.get("CI"), "The SAM encoder weights are too large to download in CI.")
    @unittest.skipIf(microsam_models is None, "The 'sam' backbone requires micro_sam.")
    def test_supervised_training_backbone(self):
        from synapse_net.training.supervised_training import supervised_training

        train_paths = self._write_h5_data(self.train_data, self.train_labels, "train")
        val_paths = self._write_h5_data(self.val_data, self.val_labels, "val")

        name = "test-backbone"
        supervised_training(
            name=name,
            train_paths=train_paths,
            val_paths=val_paths,
            label_key=self.label_key,
            patch_shape=(1, 128, 128),
            save_root=self.tmp_folder,
            raw_key=self.raw_key,
            batch_size=1,
            n_iterations=4,
            n_samples_train=2,
            n_samples_val=1,
            backbone="sam",
            model_type="vit_b",
        )

        ckpt_path = os.path.join(self.tmp_folder, "checkpoints", name, "latest.pt")
        self.assertTrue(os.path.exists(ckpt_path))


if __name__ == "__main__":
    unittest.main()
