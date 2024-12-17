import os
import platform
import sys
import unittest

from subprocess import run
from shutil import rmtree

import imageio.v3 as imageio
import mrcfile
import pooch
from synapse_net.sample_data import get_sample_data


@unittest.skipIf(platform.system() == "Windows", "CLI does not work on Windows")
class TestCLI(unittest.TestCase):
    tmp_dir = "tmp"

    def setUp(self):
        self.data_path = get_sample_data("tem_2d")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_dir)
        except OSError:
            pass

    def check_segmentation_result(self):
        output_path = os.path.join(self.tmp_dir, "tem_2d_prediction.tif")
        self.assertTrue(os.path.exists(output_path))

        prediction = imageio.imread(output_path)
        with mrcfile.open(self.data_path, "r") as f:
            data = f.data[:]
        self.assertEqual(prediction.shape, data.shape)

        num_labels = prediction.max()
        self.assertGreater(num_labels, 1)

        # import napari
        # v = napari.Viewer()
        # v.add_image(data)
        # v.add_labels(prediction)
        # napari.run()

    def test_segmentation_cli(self):
        if platform.system() == "Windows":
            cmd = [
                sys.executable, "-m",  "synapse_net.run_segmentation",
                "-i", self.data_path, "-o", self.tmp_dir, "-m", "vesicles_2d"
            ]
        else:
            cmd = ["synapse_net.run_segmentation", "-i", self.data_path, "-o", self.tmp_dir, "-m", "vesicles_2d"]
        run(cmd)
        self.check_segmentation_result()

    def test_segmentation_cli_with_scale(self):
        if platform.system() == "Windows":
            cmd = [
                sys.executable, "-m", "synapse_net.run_segmentation",
                "-i", self.data_path, "-o", self.tmp_dir, "-m", "vesicles_2d",
                "--scale", "0.5"
            ]
        else:
            cmd = [
                "synapse_net.run_segmentation", "-i", self.data_path, "-o", self.tmp_dir, "-m", "vesicles_2d",
                "--scale", "0.5"
            ]
        run(cmd)
        self.check_segmentation_result()

    def test_segmentation_cli_with_checkpoint(self):
        cache_dir = os.path.expanduser(pooch.os_cache("synapse-net"))
        model_path = os.path.join(cache_dir, "models", "vesicles_2d")
        cmd = [
            "synapse_net.run_segmentation", "-i", self.data_path, "-o", self.tmp_dir, "-m", "vesicles_2d",
            "-c", model_path,
        ]
        run(cmd)
        self.check_segmentation_result()


if __name__ == "__main__":
    unittest.main()
