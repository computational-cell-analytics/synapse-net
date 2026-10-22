import os
import sys
import unittest
from shutil import rmtree
from subprocess import run
from unittest import mock

import torch
from skimage.data import binary_blobs
from skimage.measure import label

from synapse_net.training.domain_adaptation import main as domain_adaptation_main, _resolve_source_checkpoint
from synapse_net.training.supervised_training import get_3d_model


class TestTrainingCLI(unittest.TestCase):
    tmp_folder = "./tmp_data"

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

    def _test_supervised_training(
        self,
        train_image_folder,
        train_label_folder,
        file_pattern,
        val_image_folder=None,
        val_label_folder=None,
        initial_model=None,
    ):
        name = "test-model"
        cmd = [
            "synapse_net.run_supervised_training",
            "-n", name,
            "--train_folder", train_image_folder,
            "--image_file_pattern", file_pattern,
            "--label_folder", train_label_folder,
            "--label_file_pattern", file_pattern,
            "--patch_shape", "64", "64", "64",
            "--batch_size", "1",
            "--n_samples_train", "5",
            "--n_samples_val", "1",
            "--n_iterations", "6",
            "--save_root", self.tmp_folder,
        ]
        if val_image_folder is not None:
            assert val_label_folder is not None
            cmd.extend([
                "--val_folder", val_image_folder,
                "--val_label_folder", val_label_folder,
            ])
        if initial_model is not None:
            cmd.extend(["--initial_model", initial_model])
        run(cmd)

        # Check that the checkpoint exists.
        ckpt_path = os.path.join(self.tmp_folder, "checkpoints", name, "latest.pt")
        self.assertTrue(os.path.exists(ckpt_path))

    def _write_mrc_data(self, data, out_root, labels=None):
        import mrcfile

        data_out = os.path.join(out_root, "volumes")
        os.makedirs(data_out, exist_ok=True)

        if labels is None:
            labels = [None] * len(data)
            label_out = None
        else:
            label_out = os.path.join(out_root, "labels")
            os.makedirs(label_out, exist_ok=True)

        for i, (datum, lab) in enumerate(zip(data, labels)):
            fname = f"tomo-{i}.mrc"
            with mrcfile.new(os.path.join(data_out, fname), overwrite=True) as f:
                f.set_data(datum)

            if lab is None:
                continue
            with mrcfile.new(os.path.join(label_out, fname), overwrite=True) as f:
                f.set_data(lab)

        return data_out, label_out

    def test_supervised_training_with_val_data(self):
        train_image_folder, train_label_folder = self._write_mrc_data(
            self.train_data, os.path.join(self.tmp_folder, "train"), labels=self.train_labels,
        )
        val_image_folder, val_label_folder = self._write_mrc_data(
            self.val_data, os.path.join(self.tmp_folder, "val"), labels=self.val_labels
        )
        self._test_supervised_training(
            train_image_folder, train_label_folder, file_pattern="*.mrc",
            val_image_folder=val_image_folder, val_label_folder=val_label_folder,
        )

    def test_supervised_training_without_val_data(self):
        train_image_folder, train_label_folder = self._write_mrc_data(
            self.train_data, os.path.join(self.tmp_folder, "train"), labels=self.train_labels,
        )
        self._test_supervised_training(train_image_folder, train_label_folder, file_pattern="*.mrc")

    def test_supervised_training_with_initialization(self):
        train_image_folder, train_label_folder = self._write_mrc_data(
            self.train_data, os.path.join(self.tmp_folder, "train"), labels=self.train_labels,
        )
        self._test_supervised_training(
            train_image_folder, train_label_folder, file_pattern="*.mrc", initial_model="vesicles_3d"
        )

    def test_resolve_default_source_checkpoint(self):
        expected_path = "/cache/vesicles_3d.pt"
        with mock.patch(
            "synapse_net.training.domain_adaptation.get_model_path", return_value=expected_path,
        ) as get_model_path:
            checkpoint_path = _resolve_source_checkpoint("vesicles_3d", None)

        self.assertEqual(checkpoint_path, expected_path)
        get_model_path.assert_called_once_with("vesicles_3d")

    def test_resolve_custom_source_checkpoint(self):
        checkpoint_path = os.path.join(self.tmp_folder, "custom-source.pt")
        torch.save({}, checkpoint_path)
        with mock.patch("synapse_net.training.domain_adaptation.get_model_path") as get_model_path:
            resolved_path = _resolve_source_checkpoint("vesicles_3d", checkpoint_path)

        self.assertEqual(resolved_path, checkpoint_path)
        get_model_path.assert_not_called()

    def test_resolve_missing_source_checkpoint(self):
        checkpoint_path = os.path.join(self.tmp_folder, "missing-source.pt")
        with self.assertRaisesRegex(ValueError, "The source checkpoint does not exist"):
            _resolve_source_checkpoint("vesicles_3d", checkpoint_path)

    def test_domain_adaptation_cli_uses_custom_checkpoint(self):
        checkpoint_path = os.path.join(self.tmp_folder, "custom-source.pt")
        torch.save({}, checkpoint_path)
        args = [
            "synapse_net.run_domain_adaptation",
            "--name", "test-da-model",
            "--input_folder", self.tmp_folder,
            "--source_model", "vesicles_3d",
            "--source_checkpoint", checkpoint_path,
        ]
        with (
            mock.patch.object(sys, "argv", args),
            mock.patch(
                "synapse_net.training.domain_adaptation._get_paths",
                return_value=(["train.mrc"], ["val.mrc"]),
            ),
            mock.patch(
                "synapse_net.training.domain_adaptation._derive_key_from_files",
                return_value=(["train.mrc"], "data"),
            ),
            mock.patch("synapse_net.training.domain_adaptation.mean_teacher_adaptation") as adaptation,
        ):
            domain_adaptation_main()

        self.assertEqual(adaptation.call_args.kwargs["source_checkpoint"], checkpoint_path)

    def test_domain_adaptation(self):
        train_image_folder, _ = self._write_mrc_data(self.train_data, os.path.join(self.tmp_folder, "train"))
        source_checkpoint = os.path.join(self.tmp_folder, "custom-source.pt")
        torch.save(get_3d_model(out_channels=2, initial_features=4), source_checkpoint)
        name = "test-da-model"
        cmd = [
            "synapse_net.run_domain_adaptation",
            "-n", name,
            "--input_folder", train_image_folder,
            "--file_pattern", "*.mrc",
            "--source_model", "vesicles_3d",
            "--source_checkpoint", source_checkpoint,
            "--patch_shape", "64", "64", "64",
            "--batch_size", "1",
            "--n_samples_train", "5",
            "--n_samples_val", "1",
            "--n_iterations", "6",
            "--save_root", self.tmp_folder,
        ]
        run(cmd, check=True)

        # Check that the checkpoint exists.
        ckpt_path = os.path.join(self.tmp_folder, "checkpoints", name, "latest.pt")
        self.assertTrue(os.path.exists(ckpt_path))


if __name__ == "__main__":
    unittest.main()
