from pathlib import Path
import napari
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QSpinBox, QLineEdit, QGroupBox, QFormLayout, QFrame, QComboBox, QCheckBox
import qtpy.QtWidgets as QtWidgets
from superqt import QCollapsible


class BaseWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.viewer = napari.current_viewer()
        self.attribute_dict = {}

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
