import copy

import napari
import numpy as np

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox

from .base_widget import BaseWidget
from .util import (run_segmentation, get_model, get_model_registry, _available_devices, get_device,
                   get_current_tiling, compute_scale_from_voxel_size, load_custom_model)
from ..inference.util import get_default_tiling


class SegmentationWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()
        self.tiling = {}

        # Create the image selection dropdown.
        self.image_selector_name = "Image data"
        self.image_selector_widget = self._create_layer_selector(self.image_selector_name, layer_type="Image")

        # Create buttons and widgets.
        self.predict_button = QPushButton("Run Segmentation")
        self.predict_button.clicked.connect(self.on_predict)
        self.model_selector_widget = self.load_model_widget()
        self.settings = self._create_settings_widget()

        # Add the widgets to the layout.
        layout.addWidget(self.image_selector_widget)
        layout.addWidget(self.model_selector_widget)
        layout.addWidget(self.settings)
        layout.addWidget(self.predict_button)

        self.setLayout(layout)

    def load_model_widget(self):
        model_widget = QWidget()
        title_label = QLabel("Select Model:")

        models = ["- choose -"] + list(get_model_registry().urls.keys())
        self.model_selector = QComboBox()
        self.model_selector.addItems(models)
        # Create a layout and add the title label and combo box
        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(self.model_selector)

        # Set layout on the model widget
        model_widget.setLayout(layout)
        return model_widget

    def on_predict(self):
        # Get the model and postprocessing settings.
        model_type = self.model_selector.currentText()
        custom_model_path = self.checkpoint_param.text()
        if model_type == "- choose -" and custom_model_path is None:
            show_info("INFO: Please choose a model.")
            return

        device = get_device(self.device_dropdown.currentText())

        # Load the model. Override if user chose custom model
        if custom_model_path:
            model = load_custom_model(custom_model_path, device)
            if model:
                show_info(f"INFO: Using custom model from path: {custom_model_path}")
                model_type = "custom"
            else:
                show_info(f"ERROR: Failed to load custom model from path: {custom_model_path}")
                return
        else:
            model = get_model(model_type, device)

        # Get the image data.
        image = self._get_layer_selector_data(self.image_selector_name)
        if image is None:
            show_info("INFO: Please choose an image.")
            return

        # Get the current tiling.
        self.tiling = get_current_tiling(self.tiling, self.default_tiling, model_type)

        # Get the voxel size.
        metadata = self._get_layer_selector_data(self.image_selector_name, return_metadata=True)
        voxel_size = self._handle_resolution(metadata, self.voxel_size_param, image.ndim, return_as_list=False)

        # Determine the scaling based on the voxel size.
        scale = None
        if voxel_size:
            if model_type == "custom":
                show_info("INFO: The image is not rescaled for a custom model.")
            else:
                # calculate scale so voxel_size is the same as in training
                scale = compute_scale_from_voxel_size(voxel_size, model_type)
                scale_info = list(map(lambda x: np.round(x, 2), scale))
                show_info(f"INFO: Rescaled the image by {scale_info} to optimize for the selected model.")

        # Some models require an additional segmentation for inference or postprocessing.
        # For these models we read out the 'Extra Segmentation' widget.
        if model_type == "ribbon":  # Currently only the ribbon model needs the extra seg.
            extra_seg = self._get_layer_selector_data(self.extra_seg_selector_name)
            kwargs = {"extra_segmentation": extra_seg}
        else:
            kwargs = {}
        segmentation = run_segmentation(
            image, model=model, model_type=model_type, tiling=self.tiling, scale=scale, **kwargs
        )

        # Add the segmentation layer(s).
        if isinstance(segmentation, dict):
            for name, seg in segmentation.items():
                self.viewer.add_labels(seg, name=name, metadata=metadata)
        else:
            self.viewer.add_labels(segmentation, name=f"{model_type}-segmentation", metadata=metadata)
        show_info(f"INFO: Segmentation of {model_type} added to layers.")

    def _create_settings_widget(self):
        setting_values = QWidget()
        # setting_values.setToolTip(get_tooltip("embedding", "settings"))
        setting_values.setLayout(QVBoxLayout())

        # Create UI for the device.
        device = "auto"
        device_options = ["auto"] + _available_devices()

        self.device_dropdown, layout = self._add_choice_param("device", device, device_options)
        setting_values.layout().addLayout(layout)

        # Create UI for the tile shape.
        self.default_tiling = get_default_tiling()
        self.tiling = copy.deepcopy(self.default_tiling)
        self.tiling["tile"]["x"], self.tiling["tile"]["y"], self.tiling["tile"]["z"], layout = self._add_shape_param(
            ("tile_x", "tile_y", "tile_z"),
            (self.default_tiling["tile"]["x"], self.default_tiling["tile"]["y"], self.default_tiling["tile"]["z"]),
            min_val=0, max_val=2048, step=16,
            # tooltip=get_tooltip("embedding", "tiling")
        )
        setting_values.layout().addLayout(layout)

        # Create UI for the halo.
        self.tiling["halo"]["x"], self.tiling["halo"]["y"], self.tiling["halo"]["z"], layout = self._add_shape_param(
            ("halo_x", "halo_y", "halo_z"),
            (self.default_tiling["halo"]["x"], self.default_tiling["halo"]["y"], self.default_tiling["halo"]["z"]),
            min_val=0, max_val=512,
            # tooltip=get_tooltip("embedding", "halo")
        )
        setting_values.layout().addLayout(layout)

        # Read voxel size from layer metadata.
        self.voxel_size_param, layout = self._add_float_param(
            "voxel_size", 0.0, min_val=0.0, max_val=100.0,
        )
        setting_values.layout().addLayout(layout)

        self.checkpoint_param, layout = self._add_string_param(
            name="checkpoint", value="", title="Load Custom Model",
            placeholder="path/to/checkpoint.pt",
        )
        setting_values.layout().addLayout(layout)

        # Add selection UI for additional segmentation, which some models require for inference or postproc.
        self.extra_seg_selector_name = "Extra Segmentation"
        self.extra_selector_widget = self._create_layer_selector(self.extra_seg_selector_name, layer_type="Labels")
        setting_values.layout().addWidget(self.extra_selector_widget)

        settings = self._make_collapsible(widget=setting_values, title="Advanced Settings")
        return settings
