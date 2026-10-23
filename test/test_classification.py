"""Offline tests for instance classification, using a randomly initialized model."""
import json
import os
import types
import unittest
from shutil import rmtree

import numpy as np
import torch


class TestClassification(unittest.TestCase):
    tmp_folder = "./tmp_classification"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        rmtree(self.tmp_folder, ignore_errors=True)

    def _save_checkpoint(self, class_names=("type1", "type2", "type3", "type4"), n_outputs=None):
        from synapse_net.inference.classification import _build_classification_model

        model = _build_classification_model(n_outputs or len(class_names))
        torch.save({"model_state": model.state_dict(), "init": {}}, os.path.join(self.tmp_folder, "best.pt"))
        metadata = {"bbox_size": 8, "halo": 0, "idx_to_label": {str(i): name for i, name in enumerate(class_names)}}
        with open(os.path.join(self.tmp_folder, "classification_metadata.json"), "w") as f:
            json.dump(metadata, f)

    def test_classify_instances(self):
        from synapse_net.inference.classification import (
            classify_instances, get_class_masks, load_classification_model
        )

        self._save_checkpoint()
        model, metadata = load_classification_model(self.tmp_folder, device="cpu")
        self.assertEqual(metadata["bbox_size"], 8)
        self.assertEqual(metadata["idx_to_label"][0], "type1")

        image = np.random.rand(16, 32, 32).astype("float32")
        segmentation = np.zeros(image.shape, dtype="uint32")
        segmentation[2:6, 2:6, 2:6] = 3
        segmentation[8:12, 20:26, 20:26] = 7
        segmentation[13:16, 28:32, 0:3] = 9  # touches the border, needs padding

        calls = []
        seg_id_to_class = classify_instances(
            image, segmentation, model, bbox_size=metadata["bbox_size"], halo=metadata["halo"],
            idx_to_label=metadata["idx_to_label"], batch_size=2, device="cpu",
            progress_callback=lambda done, total: calls.append((done, total)),
        )
        self.assertEqual(set(seg_id_to_class), {3, 7, 9})
        self.assertEqual(calls, [(0, 3), (2, 3), (3, 3)])
        self.assertTrue(set(seg_id_to_class.values()) <= set(metadata["idx_to_label"].values()))

        class_names = list(metadata["idx_to_label"].values())
        masks = get_class_masks(segmentation, seg_id_to_class, class_names)
        self.assertEqual(list(masks), class_names)
        # Every instance ends up in exactly one mask, with its original id.
        np.testing.assert_array_equal(sum(masks.values()), segmentation)
        for seg_id, class_name in seg_id_to_class.items():
            self.assertTrue((masks[class_name][segmentation == seg_id] == seg_id).all())

    def test_combined_class_mask(self):
        from synapse_net.inference.classification import get_combined_class_mask

        segmentation = np.array([[0, 3, 3], [7, 7, 9]], dtype="uint32")
        mask = get_combined_class_mask(segmentation, {3: "b", 7: "a", 9: "b"}, ["a", "b", "c"])
        np.testing.assert_array_equal(mask, [[0, 2, 2], [1, 1, 2]])

    def test_classes_from_model(self):
        from synapse_net.inference.classification import load_classification_model

        self._save_checkpoint(class_names=("small", "medium", "large"))
        _, metadata = load_classification_model(self.tmp_folder, device="cpu")
        self.assertEqual(metadata["idx_to_label"], {0: "small", 1: "medium", 2: "large"})

    def _save_checkpoint_without_json(self, **dataset_attrs):
        from synapse_net.inference.classification import _build_classification_model

        model = _build_classification_model(len(dataset_attrs["label_to_index"]))
        dataset = types.SimpleNamespace(**dataset_attrs)
        torch.save({"model_state": model.state_dict(), "init": {"train_dataset": dataset}},
                   os.path.join(self.tmp_folder, "best.pt"))

    def test_metadata_from_train_dataset(self):
        from synapse_net.inference.classification import load_classification_model

        self._save_checkpoint_without_json(
            bbox_size=25, halo=2, label_to_index={"b": 1, "a": 0}, normalization=None,
        )
        _, metadata = load_classification_model(self.tmp_folder, device="cpu")
        self.assertEqual(metadata, {
            "bbox_size": 25, "halo": 2, "idx_to_label": {0: "a", 1: "b"}, "normalization": None,
        })

    def test_export(self):
        from synapse_net.inference.classification import export_classification_model, load_classification_model

        self._save_checkpoint_without_json(
            bbox_size=25, halo=2, label_to_index={"b": 1, "a": 0}, normalization=None,
        )
        model, metadata = load_classification_model(self.tmp_folder, device="cpu")
        export_path = os.path.join(self.tmp_folder, "exported.pt")
        export_classification_model(self.tmp_folder, export_path)
        os.remove(os.path.join(self.tmp_folder, "best.pt"))

        exported_model, exported_metadata = load_classification_model(export_path, device="cpu")
        self.assertEqual(exported_metadata, metadata)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(exported_model.state_dict()[key], value))

    def test_resized_training_is_rejected(self):
        from synapse_net.inference.classification import load_classification_model

        self._save_checkpoint_without_json(bbox_size=20, label_to_index={"a": 0, "b": 1}, image_shape=[20, 20, 20])
        with self.assertRaises(ValueError):
            load_classification_model(self.tmp_folder, device="cpu")

    def test_metadata_mismatch(self):
        from synapse_net.inference.classification import load_classification_model

        self._save_checkpoint(class_names=("a", "b"), n_outputs=3)
        with self.assertRaises(ValueError):
            load_classification_model(self.tmp_folder, device="cpu")

    def test_empty_segmentation(self):
        from synapse_net.inference.classification import classify_instances

        image = np.zeros((8, 8, 8), dtype="float32")
        segmentation = np.zeros(image.shape, dtype="uint32")
        self.assertEqual(classify_instances(image, segmentation, None, 8, 0, {}), {})


if __name__ == "__main__":
    unittest.main()
