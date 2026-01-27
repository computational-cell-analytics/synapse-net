import os
import unittest
from shutil import rmtree
from subprocess import run

from skimage.data import binary_blobs
from skimage.measure import label


class TestTrainignCLI(unittest.TestCase):
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
        val_image_folder,
        val_label_folder,
        file_pattern,
    ):
        name = "test-model"
        cmd = [
            "synapse_net.run_supervised_training",
            "-n", name,
            "--train_folder", train_image_folder,
            "--image_file_pattern", file_pattern,
            "--label_folder", train_label_folder,
            "--label_file_pattern", file_pattern,
            "--val_folder", val_image_folder,
            "--val_label_folder", val_label_folder,
            "--patch_shape", "64", "64", "64",
            "--batch_size", "1",
            "--n_samples_train", "5",
            "--n_samples_val", "1",
            "--n_iterations", "6",
            "--save_root", self.tmp_folder,
        ]
        run(cmd)

        # Check that the checkpoint exists.
        ckpt_path = os.path.join(self.tmp_folder, "checkpoints", name, "latest.pt")
        self.assertTrue(os.path.exists(ckpt_path))

    def test_supervised_training_mrc(self):
        import mrcfile

        # Create MRC train and val data.
        def write_data(data, labels, out_root):
            data_out, label_out = os.path.join(out_root, "volumes"), os.path.join(out_root, "labels")
            os.makedirs(data_out, exist_ok=True)
            os.makedirs(label_out, exist_ok=True)
            for i, (data, labels) in enumerate(zip(data, labels)):
                fname = f"tomo-{i}.mrc"
                with mrcfile.new(os.path.join(data_out, fname), overwrite=True) as f:
                    f.set_data(data)
                with mrcfile.new(os.path.join(label_out, fname), overwrite=True) as f:
                    f.set_data(labels)
            return data_out, label_out

        train_image_folder, train_label_folder = write_data(
            self.train_data, self.train_labels, os.path.join(self.tmp_folder, "train")
        )
        val_image_folder, val_label_folder = write_data(
            self.val_data, self.val_labels, os.path.join(self.tmp_folder, "val")
        )

        self._test_supervised_training(
            train_image_folder, train_label_folder, val_image_folder, val_label_folder, file_pattern="*.mrc",
        )


if __name__ == "__main__":
    unittest.main()
