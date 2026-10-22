import os
import tempfile
import unittest
from unittest import mock

import numpy as np


os.environ.setdefault("XDG_CONFIG_HOME", os.path.join(tempfile.gettempdir(), "synapse-net-test-config"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "synapse-net-test-cache"))

from synapse_net.inference import vesicles as vesicles_module


class TestVesicleSegmentationModes(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((3, 4, 5), dtype="float32")
        self.predictions = np.zeros((2,) + self.image.shape, dtype="float32")
        self.expected = np.ones_like(self.image, dtype="uint64")

    def test_default_dispatches_to_distance_watershed(self):
        with (
            mock.patch.object(vesicles_module, "get_prediction", return_value=self.predictions),
            mock.patch.object(
                vesicles_module, "distance_based_vesicle_segmentation", return_value=self.expected
            ) as segment,
        ):
            result = vesicles_module.segment_vesicles(self.image, model=object(), verbose=False)

        self.assertIs(result, self.expected)
        np.testing.assert_array_equal(segment.call_args.args[0], self.predictions[0])
        np.testing.assert_array_equal(segment.call_args.args[1], self.predictions[1])
        self.assertEqual(segment.call_args.kwargs["min_size"], 500)
        self.assertEqual(segment.call_args.kwargs["boundary_threshold"], 0.5)

    def test_simple_watershed_dispatches_with_threshold(self):
        with (
            mock.patch.object(vesicles_module, "get_prediction", return_value=self.predictions),
            mock.patch.object(
                vesicles_module, "simple_vesicle_segmentation", return_value=self.expected
            ) as segment,
        ):
            result = vesicles_module.segment_vesicles(
                self.image,
                model=object(),
                min_size=25,
                verbose=False,
                mode="simple-watershed",
                threshold=0.65,
            )

        self.assertIs(result, self.expected)
        np.testing.assert_array_equal(segment.call_args.args[0], self.predictions[0])
        np.testing.assert_array_equal(segment.call_args.args[1], self.predictions[1])
        self.assertEqual(segment.call_args.kwargs["min_size"], 25)
        self.assertEqual(segment.call_args.kwargs["threshold"], 0.65)

    def test_label_dispatches_with_threshold_and_2d_block_shape(self):
        image = np.zeros((4, 5), dtype="float32")
        predictions = np.zeros((2,) + image.shape, dtype="float32")
        expected = np.ones_like(image, dtype="uint64")
        with (
            mock.patch.object(vesicles_module, "get_prediction", return_value=predictions),
            mock.patch.object(
                vesicles_module, "label_vesicle_segmentation", return_value=expected
            ) as segment,
        ):
            result = vesicles_module.segment_vesicles(
                image,
                model=object(),
                min_size=25,
                verbose=False,
                mode="label",
                threshold=0.6,
            )

        self.assertIs(result, expected)
        np.testing.assert_array_equal(segment.call_args.args[0], predictions[0])
        self.assertEqual(segment.call_args.kwargs["min_size"], 25)
        self.assertEqual(segment.call_args.kwargs["threshold"], 0.6)
        self.assertEqual(segment.call_args.kwargs["block_shape"], (256, 256))

    def test_invalid_mode_is_rejected_before_prediction(self):
        with mock.patch.object(vesicles_module, "get_prediction") as get_prediction:
            with self.assertRaisesRegex(ValueError, "Invalid vesicle segmentation mode"):
                vesicles_module.segment_vesicles(
                    self.image, model=object(), verbose=False, mode="invalid"
                )
        get_prediction.assert_not_called()

    def test_invalid_threshold_is_rejected_before_prediction(self):
        for threshold in (-0.01, 1.01):
            with self.subTest(threshold=threshold):
                with mock.patch.object(vesicles_module, "get_prediction") as get_prediction:
                    with self.assertRaisesRegex(ValueError, "threshold must be between 0 and 1"):
                        vesicles_module.segment_vesicles(
                            self.image,
                            model=object(),
                            verbose=False,
                            threshold=threshold,
                        )
                get_prediction.assert_not_called()

    def test_label_mode_thresholds_labels_and_filters(self):
        foreground = np.zeros((8, 8), dtype="float32")
        foreground[1:3, 1:3] = 0.7
        foreground[5, 5] = 0.9

        segmentation = vesicles_module.label_vesicle_segmentation(
            foreground,
            verbose=False,
            min_size=2,
            threshold=0.6,
            block_shape=(4, 4),
        )

        object_id = segmentation[1, 1]
        self.assertNotEqual(object_id, 0)
        np.testing.assert_array_equal(segmentation[1:3, 1:3], object_id)
        self.assertEqual(segmentation[5, 5], 0)

        segmentation = vesicles_module.label_vesicle_segmentation(
            foreground,
            verbose=False,
            min_size=2,
            threshold=0.8,
            block_shape=(4, 4),
        )
        self.assertFalse(segmentation.any())


if __name__ == "__main__":
    unittest.main()
