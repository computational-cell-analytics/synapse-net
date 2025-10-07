import napari
import napari.layers
import napari.viewer

import numpy as np
import nifty.tools as nt

from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSpinBox, QLabel
from skimage.measure import regionprops, label

from .base_widget import BaseWidget


class SizeFilterWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        # Create the dropdown to select the segmentation to filter.
        self.segmentation_selector_name = "Segmentation"
        self.segmentation_selector_widget = self._create_layer_selector(
            self.segmentation_selector_name, layer_type="Labels"
        )
        layout.addWidget(self.segmentation_selector_widget)

        # Add a field for entering the minimal size.
        self.size_threshold_widget = QSpinBox()
        self.size_threshold_widget.setRange(int(-1e9), int(1e9))
        self.size_threshold_widget.setSingleStep(1)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Minimal Size"))
        threshold_layout.addWidget(self.size_threshold_widget)
        layout.addLayout(threshold_layout)

        # Add a boolean field for applying a label operation before size filtering.
        self.apply_label = True
        layout.addWidget(self._add_boolean_param("apply_label", self.apply_label, title="Remove Disconnected Pieces"))

        # Add an optional output layer name. If not given the segmentation will be over-written.
        self.output_layer_param, _ = self._add_string_param("output_layer", "", title="Output Layer", layout=layout)

        self.button = QPushButton("Apply Size Filter")
        self.button.clicked.connect(self.on_size_filter)
        layout.addWidget(self.button)

        # Add the widgets to the layout.
        self.setLayout(layout)

    def _filter_mask(self, segmentation, size_threshold):
        segmentation = label(segmentation).astype(segmentation.dtype)
        props = regionprops(segmentation)
        filter_ids = [prop.label for prop in props if prop.area < size_threshold]
        segmentation[np.isin(segmentation, filter_ids)] = 0
        segmentation = (segmentation > 0).astype(segmentation.dtype)
        return segmentation

    def _filter_segmentation(self, segmentation, size_threshold, apply_label):
        dtype = segmentation.dtype
        if apply_label:
            original_segmentation = segmentation.copy()
            segmentation = label(segmentation)
            props = regionprops(segmentation, original_segmentation)
        else:
            props = regionprops(segmentation)
        filter_ids = [prop.label for prop in props if prop.area < size_threshold]
        segmentation[np.isin(segmentation, filter_ids)] = 0
        if apply_label:
            mapping = {prop.label: int(prop.max_intensity) for prop in props if prop.label not in filter_ids}
            mapping[0] = 0
            segmentation = nt.takeDict(mapping, segmentation)
        return segmentation.astype(dtype)

    def on_size_filter(self):
        size_threshold = self.size_threshold_widget.value()
        seg_layer = self._get_layer_selector_layer(self.segmentation_selector_name)
        segmentation = seg_layer.data.copy()

        segmentation = self._filter_segmentation(segmentation, size_threshold, self.apply_label)

        # Write or overwrite segmentation layer.
        layer_name = self.output_layer_param.text()
        if layer_name is None or layer_name == "":
            seg_layer.data = segmentation
        elif layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = segmentation
        else:
            self.viewer.add_labels(segmentation, name=layer_name)
