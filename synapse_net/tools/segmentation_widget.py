import copy
import re
from typing import Optional, Union

import napari
import numpy as np
import torch

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox

from .base_widget import BaseWidget
from ..inference.inference import _get_model_registry, get_model, run_segmentation, compute_scale_from_voxel_size
from ..inference.util import get_default_tiling, get_device


def _load_custom_model(model_path: str, device: Optional[Union[str, torch.device]] = None) -> torch.nn.Module:
    model_path = _clean_filepath(model_path)
    if device is None:
        device = get_device(device)
    try:
        model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    except Exception as e:
        print(e)
        print("model path", model_path)
        return None
    return model


def _available_devices():
    available_devices = []
    for i in ["cuda", "mps", "cpu"]:
        try:
            device = get_device(i)
        except RuntimeError:
            pass
        else:
            available_devices.append(device)
    return available_devices


def _get_current_tiling(tiling: dict, default_tiling: dict, model_type: str):
    # get tiling values from qt objects
    for k, v in tiling.items():
        for k2, v2 in v.items():
            if isinstance(v2, int):
                continue
            tiling[k][k2] = v2.value()
    # check if user inputs tiling/halo or not
    if default_tiling == tiling:
        if "2d" in model_type:
            # if its a 2d model expand x,y and set z to 1
            tiling = {
                "tile": {"x": 512, "y": 512, "z": 1},
                "halo": {"x": 64, "y": 64, "z": 1},
            }
    elif "2d" in model_type:
        # if its a 2d model set z to 1
        tiling["tile"]["z"] = 1
        tiling["halo"]["z"] = 1

    return tiling


def _clean_filepath(filepath):
    """Cleans a given filepath by:
    - Removing newline characters (\n)
    - Removing escape sequences
    - Stripping the 'file://' prefix if present

    Args:
        filepath (str): The original filepath

    Returns:
        str: The cleaned filepath
    """
    # Remove 'file://' prefix if present
    if filepath.startswith("file://"):
        filepath = filepath[7:]

    # Remove escape sequences and newlines
    filepath = re.sub(r'\\.', '', filepath)
    filepath = filepath.replace('\n', '').replace('\r', '')

    return filepath


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

        models = ["- choose -"] + list(_get_model_registry().urls.keys())
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
            model = _load_custom_model(custom_model_path, device)
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
        self.tiling = _get_current_tiling(self.tiling, self.default_tiling, model_type)

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
