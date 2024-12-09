import napari
import napari.layers
import numpy as np
import pandas as pd

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .base_widget import BaseWidget
from .. import distance_measurements


class DistanceMeasureWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        self.image_selector_name1 = "Segmentation"
        self.image_selector_name2 = "Object"
        # Create the image selection dropdown.
        self.segmentation1_selector_widget = self._create_layer_selector(self.image_selector_name1, layer_type="Labels")
        self.segmentation2_selector_widget = self._create_layer_selector(self.image_selector_name2, layer_type="Labels")

        # Create advanced settings.
        self.settings = self._create_settings_widget()

        # Create and connect buttons.
        self.measure_button1 = QPushButton("Measure Distances")
        self.measure_button1.clicked.connect(self.on_measure_seg_to_object)

        self.measure_button2 = QPushButton("Measure Pairwise Distances")
        self.measure_button2.clicked.connect(self.on_measure_pairwise)

        # Add the widgets to the layout.
        layout.addWidget(self.segmentation1_selector_widget)
        layout.addWidget(self.segmentation2_selector_widget)
        layout.addWidget(self.settings)
        layout.addWidget(self.measure_button1)
        layout.addWidget(self.measure_button2)

        self.setLayout(layout)

    def _to_table_data(self, distances, seg_ids, endpoints1=None, endpoints2=None):
        assert len(distances) == len(seg_ids), f"{distances.shape}, {seg_ids.shape}"
        if seg_ids.ndim == 2:
            table_data = {"label1": seg_ids[:, 0], "label2": seg_ids[:, 1], "distance": distances}
        else:
            table_data = {"label": seg_ids, "distance": distances}
        if endpoints1 is not None:
            axis_names = "zyx" if endpoints1.shape[1] == 3 else "yx"
            table_data.update({f"begin-{ax}": endpoints1[:, i] for i, ax in enumerate(axis_names)})
            table_data.update({f"end-{ax}": endpoints2[:, i] for i, ax in enumerate(axis_names)})
        return pd.DataFrame(table_data)

    def _add_lines_and_table(self, lines, table_data, name):
        line_layer = self.viewer.add_shapes(
            lines,
            name=name,
            shape_type="line",
            edge_width=2,
            edge_color="red",
            blending="additive",
        )
        self._add_properties_and_table(line_layer, table_data, self.save_path.text())

    def on_measure_seg_to_object(self):
        segmentation = self._get_layer_selector_data(self.image_selector_name1)
        object_data = self._get_layer_selector_data(self.image_selector_name2)

        # Get the resolution / voxel size.
        metadata = self._get_layer_selector_data(self.image_selector_name1, return_metadata=True)
        resolution = self._handle_resolution(metadata, self.voxel_size_param, segmentation.ndim)

        (distances,
         endpoints1,
         endpoints2,
         seg_ids) = distance_measurements.measure_segmentation_to_object_distances(
            segmentation=segmentation, segmented_object=object_data, distance_type="boundary",
            resolution=resolution
        )
        lines, _ = distance_measurements.create_object_distance_lines(
            distances=distances,
            endpoints1=endpoints1,
            endpoints2=endpoints2,
            seg_ids=seg_ids,
        )
        table_data = self._to_table_data(distances, seg_ids, endpoints1, endpoints2)
        structure_layer_name = self._get_layer_selector_layer(self.image_selector_name2).name
        self._add_lines_and_table(lines, table_data, name="distances-to-" + structure_layer_name)

    def on_measure_pairwise(self):
        segmentation = self._get_layer_selector_data(self.image_selector_name1)
        if segmentation is None:
            show_info("Please choose a segmentation.")
            return

        metadata = self._get_layer_selector_data(self.image_selector_name1, return_metadata=True)
        resolution = self._handle_resolution(metadata, self.voxel_size_param, segmentation.ndim)

        (distances,
         endpoints1,
         endpoints2,
         seg_ids) = distance_measurements.measure_pairwise_object_distances(
            segmentation=segmentation, distance_type="boundary", resolution=resolution
        )
        lines, properties = distance_measurements.create_pairwise_distance_lines(
            distances=distances, endpoints1=endpoints1, endpoints2=endpoints2, seg_ids=seg_ids.tolist(),
        )
        table_data = self._to_table_data(
            distances=properties["distance"],
            seg_ids=np.concatenate([properties["id_a"][:, None], properties["id_b"][:, None]], axis=1)
        )
        self._add_lines_and_table(lines, table_data, name="pairwise-distances")

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
