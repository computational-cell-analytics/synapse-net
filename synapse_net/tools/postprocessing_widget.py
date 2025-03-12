import napari
import napari.layers
import napari.viewer

import numpy as np

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QVBoxLayout, QPushButton
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from .base_widget import BaseWidget


class PostprocessingWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        # Create the dropdown to select the segmentation to post-process.
        self.segmentation_selector_name = "Segmentation"
        self.segmentation_selector_widget = self._create_layer_selector(
            self.segmentation_selector_name, layer_type="Labels"
        )
        layout.addWidget(self.segmentation_selector_widget)

        # Create dropdown to select the mask for filtering / intersection.
        self.mask_selector_name = "Mask"
        self.mask_selector_widget = self._create_layer_selector(self.mask_selector_name, layer_type="Labels")
        layout.addWidget(self.mask_selector_widget)

        # Create input for label id in the mask.
        self.mask_id_param, _ = self._add_int_param(
            "mask_id", 0, min_val=0, max_val=1000, layout=layout, title="Mask ID"
        )

        # Create text field to choose the name of the output layer.
        self.output_layer_param, _ = self._add_string_param("output_layer", "", title="Output Layer", layout=layout)

        # First postprocessing option: Filter with mask.
        self.button1 = QPushButton("Filter")
        self.button1.clicked.connect(self.on_filter)
        layout.addWidget(self.button1)

        # Second postprocessing option: intersect with boundary of the mask.
        self.button2 = QPushButton("Intersect with Boundary")
        self.button2.clicked.connect(self.on_intersect_boundary)
        layout.addWidget(self.button2)

        # Third postprocessing option: intersect with the mask.
        self.button3 = QPushButton("Intersect")
        self.button3.clicked.connect(self.on_intersect)
        layout.addWidget(self.button3)

        # Add the widgets to the layout.
        self.setLayout(layout)

    def _write_pp(self, segmentation):
        layer_name = self.output_layer_param.text()
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = segmentation
        else:
            self.viewer.add_labels(segmentation, name=layer_name)

    def _conditions_met(self):
        if self.output_layer_param.text() == "":
            show_info("Please choose an output layer.")
            return False
        return True

    def _get_segmentation_and_mask(self):
        segmentation = self._get_layer_selector_data(self.segmentation_selector_name).copy()
        mask = self._get_layer_selector_data(self.mask_selector_name)
        mask_id = self.mask_id_param.value()
        if mask_id != 0:
            mask = (mask == mask_id).astype(mask.dtype)
        return segmentation, mask

    def on_filter(self):
        if not self._conditions_met():
            return
        segmentation, mask = self._get_segmentation_and_mask()
        props = regionprops(segmentation, mask)
        filter_ids = [prop.label for prop in props if prop.max_intensity == 0]
        segmentation[np.isin(segmentation, filter_ids)] = 0
        self._write_pp(segmentation)

    def on_intersect_boundary(self):
        if not self._conditions_met():
            return
        segmentation, mask = self._get_segmentation_and_mask()
        boundary = find_boundaries(mask)
        segmentation = np.logical_and(segmentation > 0, boundary).astype(segmentation.dtype)
        self._write_pp(segmentation)

    def on_intersect(self):
        if not self._conditions_met():
            return
        segmentation, mask = self._get_segmentation_and_mask()
        segmentation = np.logical_and(segmentation > 0, mask > 0).astype(segmentation.dtype)
        self._write_pp(segmentation)
