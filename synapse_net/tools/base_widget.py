import os
import sys
from pathlib import Path

import napari
import qtpy.QtWidgets as QtWidgets

from napari.utils.notifications import show_info
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QComboBox, QCheckBox
)
from superqt import QCollapsible

try:
    from napari_skimage_regionprops import add_table, get_table
except ImportError:
    add_table, get_table = None, None


class _SilencePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class BaseWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.viewer = napari.current_viewer()
        self.attribute_dict = {}

    def _create_layer_selector(self, selector_name, layer_type="Image"):
        """Create a layer selector for an image or labels and store it in a dictionary.

        Args:
            selector_name (str): The name of the selector, used as a key in the dictionary.
            layer_type (str): The type of layer to filter for ("Image" or "Labels").
        """
        if not hasattr(self, "layer_selectors"):
            self.layer_selectors = {}

        # Determine the annotation type for the widget
        if layer_type == "Image":
            layer_filter = napari.layers.Image
        elif layer_type == "Labels":
            layer_filter = napari.layers.Labels
        elif layer_type == "Shapes":
            layer_filter = napari.layers.Shapes
        else:
            raise ValueError("layer_type must be either 'Image' or 'Labels'.")

        selector_widget = QtWidgets.QWidget()
        image_selector = QtWidgets.QComboBox()
        layer_label = QtWidgets.QLabel(f"{selector_name}:")

        # Populate initial options
        self._update_selector(selector=image_selector, layer_filter=layer_filter)

        # Update selector on layer events
        self.viewer.layers.events.inserted.connect(lambda event: self._update_selector(image_selector, layer_filter))
        self.viewer.layers.events.removed.connect(lambda event: self._update_selector(image_selector, layer_filter))

        # Store the selector in the dictionary
        self.layer_selectors[selector_name] = selector_widget

        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(layer_label)
        layout.addWidget(image_selector)
        selector_widget.setLayout(layout)
        return selector_widget

    def _update_selector(self, selector, layer_filter):
        """Update a single selector with the current image layers in the viewer."""
        selector.clear()
        image_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, layer_filter)]
        selector.addItems(image_layers)

    def _get_layer_selector_layer(self, selector_name):
        """Return the layer currently selected in a given selector."""
        if selector_name in self.layer_selectors:
            selector_widget = self.layer_selectors[selector_name]

            # Retrieve the QComboBox from the QWidget's layout
            image_selector = selector_widget.layout().itemAt(1).widget()

            if isinstance(image_selector, QComboBox):
                selected_layer_name = image_selector.currentText()
                if selected_layer_name in self.viewer.layers:
                    return self.viewer.layers[selected_layer_name]
        return None  # Return None if layer not found

    def _get_layer_selector_data(self, selector_name, return_metadata=False):
        """Return the data for the layer currently selected in a given selector."""
        if selector_name in self.layer_selectors:
            selector_widget = self.layer_selectors[selector_name]

            # Retrieve the QComboBox from the QWidget's layout
            image_selector = selector_widget.layout().itemAt(1).widget()

            if isinstance(image_selector, QComboBox):
                selected_layer_name = image_selector.currentText()
                if selected_layer_name in self.viewer.layers:
                    if return_metadata:
                        return self.viewer.layers[selected_layer_name].metadata
                    else:
                        return self.viewer.layers[selected_layer_name].data
        return None  # Return None if layer not found

    def _add_string_param(self, name, value, title=None, placeholder=None, layout=None, tooltip=None):
        if layout is None:
            layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(title or name)
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)
        param = QtWidgets.QLineEdit()
        param.setText(value)
        if placeholder is not None:
            param.setPlaceholderText(placeholder)
        param.textChanged.connect(lambda val: setattr(self, name, val))
        if tooltip:
            param.setToolTip(tooltip)
        layout.addWidget(param)
        return param, layout

    def _add_float_param(self, name, value, title=None, min_val=0.0, max_val=1.0, decimals=2,
                         step=0.01, layout=None, tooltip=None):
        if layout is None:
            layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(title or name)
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)
        param = QtWidgets.QDoubleSpinBox()
        param.setRange(min_val, max_val)
        param.setDecimals(decimals)
        param.setValue(value)
        param.setSingleStep(step)
        param.valueChanged.connect(lambda val: setattr(self, name, val))
        if tooltip:
            param.setToolTip(tooltip)
        layout.addWidget(param)
        return param, layout

    def _add_int_param(self, name, value, min_val, max_val, title=None, step=1, layout=None, tooltip=None):
        if layout is None:
            layout = QHBoxLayout()
        label = QLabel(title or name)
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)
        param = QSpinBox()
        param.setRange(min_val, max_val)
        param.setValue(value)
        param.setSingleStep(step)
        param.valueChanged.connect(lambda val: setattr(self, name, val))
        if tooltip:
            param.setToolTip(tooltip)
        layout.addWidget(param)
        return param, layout

    def _add_choice_param(self, name, value, options, title=None, layout=None, update=None, tooltip=None):
        if layout is None:
            layout = QHBoxLayout()
        label = QLabel(title or name)
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)

        # Create the dropdown menu via QComboBox, set the available values.
        dropdown = QComboBox()
        dropdown.addItems(options)
        if update is None:
            dropdown.currentIndexChanged.connect(lambda index: setattr(self, name, options[index]))
        else:
            dropdown.currentIndexChanged.connect(update)

        # Set the correct value for the value.
        dropdown.setCurrentIndex(dropdown.findText(value))

        if tooltip:
            dropdown.setToolTip(tooltip)

        layout.addWidget(dropdown)
        return dropdown, layout

    def _add_shape_param(self, names, values, min_val, max_val, step=1, title=None, tooltip=None):
        layout = QHBoxLayout()

        x_layout = QVBoxLayout()
        x_param, _ = self._add_int_param(
            names[0], values[0], min_val=min_val, max_val=max_val, layout=x_layout, step=step,
            title=title[0] if title is not None else title, tooltip=tooltip
        )
        layout.addLayout(x_layout)

        y_layout = QVBoxLayout()
        y_param, _ = self._add_int_param(
            names[1], values[1], min_val=min_val, max_val=max_val, layout=y_layout, step=step,
            title=title[1] if title is not None else title, tooltip=tooltip
        )
        layout.addLayout(y_layout)

        if len(names) == 3:
            z_layout = QVBoxLayout()
            z_param, _ = self._add_int_param(
                names[2], values[2], min_val=min_val, max_val=max_val, layout=z_layout, step=step,
                title=title[2] if title is not None else title, tooltip=tooltip
            )
            layout.addLayout(z_layout)
            return x_param, y_param, z_param, layout

        return x_param, y_param, layout

    def _make_collapsible(self, widget, title):
        parent_widget = QWidget()
        parent_widget.setLayout(QVBoxLayout())
        collapsible = QCollapsible(title, parent_widget)
        collapsible.addWidget(widget)
        parent_widget.layout().addWidget(collapsible)
        return parent_widget

    def _add_boolean_param(self, name, value, title=None, tooltip=None):
        checkbox = QCheckBox(name if title is None else title)
        checkbox.setChecked(value)
        checkbox.stateChanged.connect(lambda val: setattr(self, name, val))
        if tooltip:
            checkbox.setToolTip(tooltip)
        return checkbox

    def _add_path_param(self, name, value, select_type, title=None, placeholder=None, tooltip=None):
        assert select_type in ("directory", "file", "both")

        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(title or name)
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)

        path_textbox = QtWidgets.QLineEdit()
        path_textbox.setText(str(value))
        if placeholder is not None:
            path_textbox.setPlaceholderText(placeholder)
        path_textbox.textChanged.connect(lambda val: setattr(self, name, val))
        if tooltip:
            path_textbox.setToolTip(tooltip)

        layout.addWidget(path_textbox)

        def add_path_button(select_type, tooltip=None):
            # Adjust button text.
            button_text = f"Select {select_type.capitalize()}"
            path_button = QtWidgets.QPushButton(button_text)

            # Call appropriate function based on select_type.
            path_button.clicked.connect(lambda: getattr(self, f"_get_{select_type}_path")(name, path_textbox))
            if tooltip:
                path_button.setToolTip(tooltip)
            layout.addWidget(path_button)

        if select_type == "both":
            add_path_button("file")
            add_path_button("directory")

        else:
            add_path_button(select_type)

        return path_textbox, layout

    def _get_directory_path(self, name, textbox, tooltip=None):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", "", QtWidgets.QFileDialog.ShowDirsOnly
        )
        if tooltip:
            directory.setToolTip(tooltip)
        if directory and Path(directory).is_dir():
            textbox.setText(str(directory))
        else:
            # Handle the case where the selected path is not a directory
            print("Invalid directory selected. Please try again.")

    def _get_file_path(self, name, textbox, tooltip=None):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select File", "", "All Files (*)"
        )
        if tooltip:
            file_path.setToolTip(tooltip)
        if file_path and Path(file_path).is_file():
            textbox.setText(str(file_path))
        else:
            # Handle the case where the selected path is not a file
            print("Invalid file selected. Please try again.")

    def _handle_resolution(self, metadata, voxel_size_param, ndim, return_as_list=True):
        # Get the resolution / voxel size from the layer metadata if available.
        resolution = metadata.get("voxel_size", None)

        # If user input was given then override resolution from metadata.
        axes = "zyx" if ndim == 3 else "yx"
        if voxel_size_param.value() != 0.0:  # Changed from default.
            resolution = {ax: voxel_size_param.value() for ax in axes}

        if resolution is not None and return_as_list:
            resolution = [resolution[ax] for ax in axes]
            assert len(resolution) == ndim

        return resolution

    def _save_table(self, save_path, data):
        ext = os.path.splitext(save_path)[1]
        if ext == "":  # No file extension given, By default we save to CSV.
            file_path = f"{save_path}.csv"
            data.to_csv(file_path, index=False)
        elif ext == ".csv":  # Extension was specified as csv
            file_path = save_path
            data.to_csv(file_path, index=False)
        elif ext == ".xlsx":  # We also support excel.
            file_path = save_path
            data.to_excel(file_path, index=False)
        else:
            raise ValueError("Invalid extension for table: {ext}. We support .csv or .xlsx.")
        return file_path

    def _add_properties_and_table(self, layer, table_data, save_path=""):
        layer.properties = table_data

        if add_table is not None:
            table = get_table(layer, self.viewer)
            if table is None:
                with _SilencePrint():
                    add_table(layer, self.viewer)
            else:
                # FIXME updating the table does not yet work
                with _SilencePrint():
                    table.update_content()
                # table_dict = table_data.to_dict()
                # table_dict["index"] = table_dict["label"]
                # table.set_content(table_dict)

        # Save table to file if save path is provided.
        if save_path != "":
            file_path = self._save_table(self.save_path.text(), table_data)
            show_info(f"INFO: Added table and saved file to {file_path}.")
