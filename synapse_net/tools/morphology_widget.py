import napari
import pandas as pd
from skimage.measure import regionprops

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .base_widget import BaseWidget
from synapse_net.imod.to_imod import convert_segmentation_to_spheres
from synapse_net.morphology import compute_object_morphology


class MorphologyWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        self.image_selector_name = "Raw Image"
        self.image_selector_name1 = "Segmentation"
        # Create the image selection dropdown.
        self.image_selector_widget = self._create_layer_selector(self.image_selector_name, layer_type="Image")
        self.segmentation1_selector_widget = self._create_layer_selector(self.image_selector_name1, layer_type="Labels")

        # Create advanced settings.
        self.settings = self._create_settings_widget()

        # Create and connect buttons.
        self.measure_button1 = QPushButton("Measure Vesicle Morphology")
        self.measure_button1.clicked.connect(self.on_measure_vesicle_morphology)

        self.measure_button2 = QPushButton("Measure Structure Morphology")
        self.measure_button2.clicked.connect(self.on_measure_structure_morphology)

        # Add the widgets to the layout.
        layout.addWidget(self.image_selector_widget)
        layout.addWidget(self.segmentation1_selector_widget)
        layout.addWidget(self.settings)
        layout.addWidget(self.measure_button1)
        layout.addWidget(self.measure_button2)

        self.setLayout(layout)

    def _to_table_data(self, coords, radii, props):
        """
        Create a table of data including coordinates, radii, and intensity statistics.

        Args:
            coords (np.ndarray): Array of 2D or 3D coordinates.
            radii (np.ndarray): Array of radii corresponding to the coordinates.
            props (list): List of properties containing intensity statistics.

        Returns:
            pd.DataFrame: The formatted table data.
        """
        assert len(coords) == len(radii), f"Shape mismatch: {coords.shape}, {radii.shape}"

        # Define columns based on dimension (2D or 3D)
        col_names = ["x", "y"] if coords.shape[1] == 2 else ["x", "y", "z"]
        table_data = {
            "label": [prop.label for prop in props],
            **{col: coords[:, i] for i, col in enumerate(col_names)},
            "radius": radii,
            "intensity_max": [prop.intensity_max for prop in props],
            "intensity_mean": [prop.intensity_mean for prop in props],
            "intensity_min": [prop.intensity_min for prop in props],
            "intensity_std": [prop.intensity_std for prop in props],
        }

        return pd.DataFrame(table_data)

    def on_measure_vesicle_morphology(self):
        segmentation = self._get_layer_selector_data(self.image_selector_name1)
        image = self._get_layer_selector_data(self.image_selector_name)
        if segmentation is None:
            show_info("INFO: Please choose a segmentation.")
            return
        if image is None:
            show_info("INFO: Please choose an image.")
            return

        # Get the resolution / voxel size.
        metadata = self._get_layer_selector_data(self.image_selector_name, return_metadata=True)
        resolution = self._handle_resolution(metadata, self.voxel_size_param, segmentation.ndim)

        # Compute the mophology parameter.
        props = regionprops(label_image=segmentation, intensity_image=image)
        coords, radii = convert_segmentation_to_spheres(
            segmentation=segmentation,
            resolution=resolution,
            props=props,
        )

        # Create table data and add the properties and table to the layer.
        table_data = self._to_table_data(coords, radii, props)
        layer = self._get_layer_selector_layer(self.image_selector_name1)
        self._add_properties_and_table(layer, table_data, self.save_path.text())

    def on_measure_structure_morphology(self):
        """
        Add the structure measurements to the segmentation layer (via properties)
        and visualize the properties table
        """
        segmentation = self._get_layer_selector_data(self.image_selector_name1)
        if segmentation is None:
            show_info("INFO: Please choose a segmentation.")
            return
        metadata = self._get_layer_selector_data(self.image_selector_name, return_metadata=True) or {}
        resolution = self._handle_resolution(metadata, self.voxel_size_param, segmentation.ndim)
        layer = self._get_layer_selector_layer(self.image_selector_name1)
        morphology = compute_object_morphology(object_=segmentation, structure_name=layer.name, resolution=resolution)

        # Add the properties to the layer and add/save the table.
        self._add_properties_and_table(layer, morphology, self.save_path.text())

    def _create_settings_widget(self):
        setting_values = QWidget()
        setting_values.setLayout(QVBoxLayout())

        self.save_path, layout = self._add_path_param(name="Save Table", select_type="file", value="")
        setting_values.layout().addLayout(layout)

        self.voxel_size_param, layout = self._add_float_param(
            "voxel_size", 0.0, min_val=0.0, max_val=100.0,
        )
        setting_values.layout().addLayout(layout)

        settings = self._make_collapsible(widget=setting_values, title="Advanced Settings")
        return settings
