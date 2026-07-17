import copy
import inspect
import re
from typing import Optional, Union

import napari
import numpy as np
import torch

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget

from .base_widget import BaseWidget
from ..inference.active_zone import segment_active_zone
from ..inference.compartments import segment_compartments
from ..inference.cristae import segment_cristae
from ..inference.inference import (
    _get_model_registry,
    _segment_ribbon_AZ,
    compute_scale_from_voxel_size,
    get_model,
    get_segmentation_function,
    run_segmentation,
)
from ..inference.mitochondria import segment_mitochondria
from ..inference.util import get_default_tiling, get_device
from ..inference.vesicles import VESICLE_SEGMENTATION_MODES, segment_vesicles


_MAX_MIN_SIZE = 100_000_000
_POSTPROCESSING_PARAMETER_SPECS = {
    segment_vesicles: {
        "min_size": {"type": "int", "min": 0, "max": _MAX_MIN_SIZE, "step": 1},
        "mode": {"type": "choice", "options": list(VESICLE_SEGMENTATION_MODES)},
        "threshold": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 2},
    },
    segment_mitochondria: {
        "min_size": {"type": "int", "min": 0, "max": _MAX_MIN_SIZE, "step": 1},
        "seed_distance": {"type": "int", "min": 0, "max": 10_000, "step": 1},
    },
    segment_active_zone: {
        "min_size": {"type": "int", "min": 0, "max": _MAX_MIN_SIZE, "step": 1},
        "foreground_threshold": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 2},
    },
    segment_compartments: {
        "boundary_threshold": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 2},
        "n_slices_exclude": {"type": "int", "min": 0, "max": 10_000, "step": 1},
        "min_z_extent": {"type": "int", "min": 0, "max": 10_000, "step": 1},
    },
    _segment_ribbon_AZ: {
        "threshold": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 2},
        "n_slices_exclude": {"type": "int", "min": 0, "max": 10_000, "step": 1},
        "min_membrane_size": {
            "type": "int", "min": 0, "max": _MAX_MIN_SIZE, "step": 1, "default": 50_000,
        },
        "n_ribbons": {"type": "int", "min": 1, "max": 1_000, "step": 1},
    },
    segment_cristae: {
        "min_size": {"type": "int", "min": 0, "max": _MAX_MIN_SIZE, "step": 1},
        "foreground_threshold": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "decimals": 2},
        "erosion_distance_nm": {"type": "float", "min": 0.0, "max": 1_000.0, "step": 0.1, "decimals": 1},
    },
}


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
            elif hasattr(v2, "value"):  # If it's a QSpinBox, extract the value
                tiling[k][k2] = v2.value()
            else:
                raise TypeError(f"Unexpected type for tiling value: {type(v2)} at {k}/{k2}")
    # check if user inputs tiling/halo or not
    if default_tiling == tiling:
        if "2d" in model_type:
            # if its a 2d model expand x,y and set z to 1
            tiling = {
                "tile": {"x": 512, "y": 512, "z": 1},
                "halo": {"x": 64, "y": 64, "z": 1},
            }
    else:
        show_info(f"Using custom tiling: {tiling}")
    if "2d" in model_type:
        # if its a 2d model set z to 1
        tiling["tile"]["z"] = 1
        tiling["halo"]["z"] = 0
        show_info(f"Using tiling: {tiling}")
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

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                SegmentationWidget._clear_layout(child_layout)
                child_layout.deleteLater()

    def _update_postprocessing_settings(self, model_type):
        self._clear_layout(self.postprocessing_settings_layout)
        self.postprocessing_parameter_widgets = {}
        if model_type == "- choose -":
            return

        segmentation_function = get_segmentation_function(model_type)
        parameter_specs = _POSTPROCESSING_PARAMETER_SPECS.get(segmentation_function, {})
        function_parameters = inspect.signature(segmentation_function).parameters

        for name, spec in parameter_specs.items():
            if name not in function_parameters or function_parameters[name].default is inspect.Parameter.empty:
                raise ValueError(
                    f"Configured post-processing parameter '{name}' is not an optional parameter "
                    f"of {segmentation_function.__name__}."
                )
            default = spec.get("default", function_parameters[name].default)
            if spec["type"] == "int":
                parameter_widget, parameter_layout = self._add_int_param(
                    name, default, min_val=spec["min"], max_val=spec["max"], step=spec["step"]
                )
                self.postprocessing_settings_layout.addLayout(parameter_layout)
            elif spec["type"] == "float":
                parameter_widget, parameter_layout = self._add_float_param(
                    name,
                    default,
                    min_val=spec["min"],
                    max_val=spec["max"],
                    step=spec["step"],
                    decimals=spec["decimals"],
                )
                self.postprocessing_settings_layout.addLayout(parameter_layout)
            elif spec["type"] == "bool":
                parameter_widget = self._add_boolean_param(name, default)
                self.postprocessing_settings_layout.addWidget(parameter_widget)
            elif spec["type"] == "choice":
                parameter_widget, parameter_layout = self._add_choice_param(name, default, spec["options"])
                self.postprocessing_settings_layout.addLayout(parameter_layout)
            else:
                raise ValueError(f"Unsupported post-processing parameter type: {spec['type']}")
            self.postprocessing_parameter_widgets[name] = parameter_widget

    def _get_postprocessing_kwargs(self):
        kwargs = {}
        for name, widget in self.postprocessing_parameter_widgets.items():
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value = widget.currentText()
            else:
                value = widget.value()
            kwargs[name] = value
        return kwargs

    def load_model_widget(self):
        model_widget = QWidget()
        title_label = QLabel("Select Model:")

        # Exclude the models that are only offered through the CLI and not in the plugin.
        model_list = set(_get_model_registry().urls.keys())
        # These are the models exlcuded due to their specificity and to keep the menu simple.
        # TODO: we should at some point update the logic here, to make it easier to support further models
        # without cluttering the UI.
        excluded_models = ["vesicles_2d_maus"]
        model_list = [name for name in model_list if name not in excluded_models]

        models = ["- choose -"] + model_list
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
        if model_type == "- choose -":
            show_info("INFO: Please choose a model.")
            return

        device = get_device(self.device_dropdown.currentText())

        # Load the model. Override if user chose custom model.
        rescale_input = True
        if custom_model_path:
            model = _load_custom_model(custom_model_path, device)
            rescale_input = False
            if model:
                show_info(f"INFO: Using custom model from path: {custom_model_path}")
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
        if voxel_size and rescale_input:
            # Calculate scale so voxel_size is the same as in training.
            scale = compute_scale_from_voxel_size(voxel_size, model_type)
            scale_info = list(map(lambda x: np.round(x, 2), scale))
            show_info(f"INFO: Rescaled the image by {scale_info} to optimize for the selected model.")

        # Some models require an additional segmentation for inference or postprocessing.
        # For these models we read out the 'Extra Segmentation' widget.
        if model_type == "ribbon":  # Currently only the ribbon model needs the extra seg.
            extra_seg = self._get_layer_selector_data(self.extra_seg_selector_name)
            resolution = tuple(voxel_size[ax] for ax in "zyx")
            kwargs = {"extra_segmentation": extra_seg, "resolution": resolution}
        elif "cristae" in model_type:  # Cristae model expects 2 3D volumes
            kwargs = {
                "extra_segmentation": self._get_layer_selector_data(self.extra_seg_selector_name),
                "with_channels": True,
                "channels_to_standardize": [0]
            }
        else:
            kwargs = {}
        kwargs.update(self._get_postprocessing_kwargs())
        segmentation = run_segmentation(
            image, model=model, model_type=model_type, tiling=self.tiling, scale=scale, **kwargs
        )

        # Add the segmentation layer(s).
        if isinstance(segmentation, dict):
            for name, seg in segmentation.items():
                self.viewer.add_labels(seg, name=name, metadata=metadata)
        else:
            self.viewer.add_labels(segmentation, name=f"{model_type}", metadata=metadata)
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

        # Add model-specific post-processing settings that are updated when the selected model changes.
        setting_values.layout().addWidget(QLabel("Post-processing:"))
        self.postprocessing_settings_widget = QWidget()
        self.postprocessing_settings_layout = QVBoxLayout()
        self.postprocessing_settings_layout.setContentsMargins(0, 0, 0, 0)
        self.postprocessing_settings_widget.setLayout(self.postprocessing_settings_layout)
        setting_values.layout().addWidget(self.postprocessing_settings_widget)
        self.postprocessing_parameter_widgets = {}
        self.model_selector.currentTextChanged.connect(self._update_postprocessing_settings)
        self._update_postprocessing_settings(self.model_selector.currentText())

        settings = self._make_collapsible(widget=setting_values, title="Advanced Settings")
        return settings
