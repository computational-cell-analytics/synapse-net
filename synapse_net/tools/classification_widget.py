import napari
import numpy as np

from napari.utils import progress
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget

from .base_widget import BaseWidget
from .segmentation_widget import _available_devices, _clean_filepath
from ..inference.classification import (
    classify_instances, get_class_masks, get_combined_class_mask, load_classification_model
)
from ..inference.util import get_device

# TODO: register pretrained classification models here once they are available.
_CLASSIFICATION_MODELS = []


class ClassificationWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()
        self._loaded_model = None  # (checkpoint, device, model, metadata), avoids reloading on every run.

        self.image_selector_name = "Image data"
        self.image_selector_widget = self._create_layer_selector(self.image_selector_name, layer_type="Image")

        self.segmentation_selector_name = "Segmentation mask"
        self.segmentation_selector_widget = self._create_layer_selector(
            self.segmentation_selector_name, layer_type="Labels"
        )

        self.model_selector_widget = self.load_model_widget()
        self.settings = self._create_settings_widget()

        self.classify_button = QPushButton("Run Classification")
        self.classify_button.clicked.connect(self.on_classify)

        layout.addWidget(self.image_selector_widget)
        layout.addWidget(self.segmentation_selector_widget)
        layout.addWidget(self.model_selector_widget)
        layout.addWidget(self.settings)
        layout.addWidget(self.classify_button)

        self.setLayout(layout)

    def load_model_widget(self):
        model_widget = QWidget()
        title_label = QLabel("Select Model:")

        self.model_selector = QComboBox()
        self.model_selector.addItems(["- choose -"] + _CLASSIFICATION_MODELS)

        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(self.model_selector)
        model_widget.setLayout(layout)
        return model_widget

    def _create_settings_widget(self):
        setting_values = QWidget()
        setting_values.setLayout(QVBoxLayout())

        self.device_dropdown, layout = self._add_choice_param("device", "auto", ["auto"] + _available_devices())
        setting_values.layout().addLayout(layout)

        self.batch_size_param, layout = self._add_int_param(
            "batch_size", 16, min_val=1, max_val=1024, tooltip="Number of instances classified at once.",
        )
        setting_values.layout().addLayout(layout)

        self.checkpoint_param, layout = self._add_string_param(
            name="checkpoint", value="", title="Load Custom Model", placeholder="path/to/checkpoint_dir",
            tooltip="Checkpoint folder with 'best.pt', or a .pt file.",
        )
        setting_values.layout().addLayout(layout)

        self.combine_masks = False
        setting_values.layout().addWidget(self._add_boolean_param(
            "combine_masks", self.combine_masks, title="Combine class masks",
            tooltip="Add a single layer in which each instance is labeled with its class id (1, 2, ...), "
                    "instead of one layer per class.",
        ))

        return self._make_collapsible(widget=setting_values, title="Advanced Settings")

    def _get_model(self, checkpoint, device):
        if self._loaded_model is None or self._loaded_model[:2] != (checkpoint, device):
            model, metadata = load_classification_model(checkpoint, device)
            self._loaded_model = (checkpoint, device, model, metadata)
        return self._loaded_model[2:]

    def on_classify(self):
        custom_model_path = _clean_filepath(self.checkpoint_param.text())
        if not custom_model_path:
            # TODO: download the selected model once pretrained models are registered.
            show_info("INFO: Please enter a custom model under 'Advanced Settings'.")
            return

        image = self._get_layer_selector_data(self.image_selector_name)
        if image is None:
            show_info("INFO: Please choose an image.")
            return

        seg_layer = self._get_layer_selector_layer(self.segmentation_selector_name)
        if seg_layer is None:
            show_info("INFO: Please choose a segmentation mask.")
            return
        segmentation = np.asarray(seg_layer.data)

        if image.ndim != 3 or image.shape != segmentation.shape:
            show_info(
                f"INFO: Image and segmentation mask must be 3D and have the same shape, "
                f"got {image.shape} and {segmentation.shape}."
            )
            return

        device = get_device(self.device_dropdown.currentText())
        with self._computing(
            self.classify_button, "Computing…", "Run Classification", "INFO: Running classification…"
        ):
            try:
                model, metadata = self._get_model(custom_model_path, device)
            except Exception as e:
                show_info(f"ERROR: Failed to load classification model from {custom_model_path}: {e}")
                return
            show_info(f"INFO: Using custom model from path: {custom_model_path}")

            # The total is only known inside classify_instances, so the activity-dock bar is created there.
            pbar = {}

            def _on_progress(done, total):
                if "bar" not in pbar:
                    pbar["bar"] = progress(total=total, desc="Classification")
                pbar["bar"].update(done - pbar["bar"].n)

            try:
                seg_id_to_class = classify_instances(
                    np.asarray(image), segmentation, model, **metadata,
                    batch_size=self.batch_size_param.value(), device=device, progress_callback=_on_progress,
                )
            finally:
                if "bar" in pbar:
                    pbar["bar"].close()

            class_names = [metadata["idx_to_label"][idx] for idx in sorted(metadata["idx_to_label"])]
            layer_kwargs = dict(
                scale=seg_layer.scale, translate=seg_layer.translate, metadata=dict(seg_layer.metadata)
            )
            if self.combine_masks:
                mask = get_combined_class_mask(segmentation, seg_id_to_class, class_names)
                # Shows the class name next to the label id when hovering.
                features = {"index": list(range(len(class_names) + 1)), "class": ["background"] + class_names}
                self.add_or_update_labels(f"{seg_layer.name}_classes", mask, features=features, **layer_kwargs)
            else:
                masks = get_class_masks(segmentation, seg_id_to_class, class_names)
                for class_name, mask in masks.items():
                    self.add_or_update_labels(f"{seg_layer.name}_{class_name}", mask, **layer_kwargs)

        counts = ", ".join(
            f"{name}: {sum(c == name for c in seg_id_to_class.values())}" for name in class_names
        )
        show_info(f"INFO: Classified {len(seg_id_to_class)} instances ({counts}).")
        if self.combine_masks:
            label_info = ", ".join(f"{label} = {name}" for label, name in enumerate(class_names, start=1))
            show_info(f"INFO: Label ids in {seg_layer.name}_classes: {label_info}.")
