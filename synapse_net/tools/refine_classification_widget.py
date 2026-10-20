import napari
import numpy as np

from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QLabel, QListWidget, QListWidgetItem, QPushButton, QShortcut, QVBoxLayout, QWidget
)
from scipy.ndimage import find_objects

from .base_widget import BaseWidget

_KEEP_KEY, _REMOVE_KEY, _UNDO_KEY = "K", "R", "U"
_MAX_TARGETS = 9  # Targets are bound to the digit keys 1-9.


def _present_ids(data):
    return {i + 1 for i, bbox in enumerate(find_objects(data)) if bbox is not None}


class RefineClassificationWidget(BaseWidget):
    """Step through the instances of one mask and keep, remove, or move each one to another mask."""

    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        self.source_selector_name = "Mask to refine"
        self.source_selector_widget = self._create_layer_selector(self.source_selector_name, layer_type="Labels")
        self.source_selector = self.source_selector_widget.layout().itemAt(1).widget()
        self.source_selector.currentTextChanged.connect(self._update_target_list)

        self.target_list = QListWidget()
        self.target_list.setToolTip("Masks the instances can be moved to. Bound to the keys 1-9 in this order.")
        self.target_list.setMaximumHeight(150)
        self.viewer.layers.events.inserted.connect(self._update_target_list)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self._update_target_list()

        self.settings = self._create_settings_widget()

        self.start_button = QPushButton("Start Review")
        self.start_button.clicked.connect(self.on_start)

        # Filled with one button per action when a review starts.
        self.review_widget = QWidget()
        self.review_widget.setLayout(QVBoxLayout())
        self.review_widget.layout().setContentsMargins(0, 0, 0, 0)
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)

        layout.addWidget(self.source_selector_widget)
        layout.addWidget(QLabel("Move instances to:"))
        layout.addWidget(self.target_list)
        layout.addWidget(self.settings)
        layout.addWidget(self.start_button)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.review_widget)
        self.setLayout(layout)

        self._source = None
        self._targets = []
        self._instances = []  # (seg_id, bbox) of the source mask, in id order.
        self._index = 0
        self._history = []
        self._target_ids = {}
        self._shortcuts = []
        self._action_buttons = []
        self._previous_display = None

    def _create_settings_widget(self):
        setting_values = QWidget()
        setting_values.setLayout(QVBoxLayout())

        self.zoom_to_instance = True
        setting_values.layout().addWidget(self._add_boolean_param(
            "zoom_to_instance", self.zoom_to_instance, title="Zoom to instance",
            tooltip="Zoom in on each instance. Otherwise only the view is centered on it.",
        ))

        self.context_param, layout = self._add_float_param(
            "context", 3.0, title="Field of view", min_val=1.0, max_val=50.0, decimals=1, step=0.5,
            tooltip="Size of the visible area, in multiples of the instance size.",
        )
        setting_values.layout().addLayout(layout)

        self.show_only_current = True
        setting_values.layout().addWidget(self._add_boolean_param(
            "show_only_current", self.show_only_current, title="Show only current instance",
            tooltip="Hide the other instances of the mask that is refined.",
        ))

        return self._make_collapsible(widget=setting_values, title="Advanced Settings")

    def _update_target_list(self, *args):
        checked = {
            self.target_list.item(i).text() for i in range(self.target_list.count())
            if self.target_list.item(i).checkState() == Qt.Checked
        }
        source_name = self.source_selector.currentText()
        self.target_list.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels) and layer.name != source_name:
                item = QListWidgetItem(layer.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if layer.name in checked else Qt.Unchecked)
                self.target_list.addItem(item)

    def _on_layer_removed(self, event):
        self._update_target_list()
        if self._source is not None and any(event.value is layer for layer in [self._source] + self._targets):
            show_info(f"INFO: Layer '{event.value.name}' was removed, the review was stopped.")
            self._stop_review()
            self.progress_label.setText("Review stopped.")

    def on_start(self):
        if self._source is not None:
            self._stop_review()

        source = self._get_layer_selector_layer(self.source_selector_name)
        if source is None:
            show_info("INFO: Please choose the mask to refine.")
            return
        targets = [
            self.viewer.layers[self.target_list.item(i).text()] for i in range(self.target_list.count())
            if self.target_list.item(i).checkState() == Qt.Checked
        ]
        if len(targets) > _MAX_TARGETS:
            show_info(f"INFO: Please choose at most {_MAX_TARGETS} masks to move instances to.")
            return
        for target in targets:
            if target.data.shape != source.data.shape:
                show_info(
                    f"INFO: '{target.name}' has shape {target.data.shape}, but '{source.name}' has "
                    f"{source.data.shape}. All masks must have the same shape."
                )
                return

        # Edits go through Labels.data_setitem, which needs in-memory data.
        for layer in [source] + targets:
            if not isinstance(layer.data, np.ndarray):
                layer.data = np.asarray(layer.data)

        instances = [(i + 1, bbox) for i, bbox in enumerate(find_objects(source.data)) if bbox is not None]
        if not instances:
            show_info(f"INFO: '{source.name}' has no instances.")
            return

        self._source, self._targets, self._instances = source, targets, instances
        self._index, self._history = 0, []
        self._target_ids = {target.name: _present_ids(target.data) for target in targets}
        self._previous_display = (source.show_selected_label, source.selected_label, source.visible)
        source.visible = True
        source.show_selected_label = self.show_only_current
        self.viewer.layers.selection.active = source

        self._build_actions()
        self.start_button.setText("Restart Review")
        self._show_current()

    def _build_actions(self):
        actions = [(f"Keep  [{_KEEP_KEY}]", _KEEP_KEY, lambda: self._apply("keep"))]
        for i, target in enumerate(self._targets, start=1):
            actions.append((f"Move to {target.name}  [{i}]", str(i), lambda t=target: self._apply(t)))
        actions.append((f"Remove  [{_REMOVE_KEY}]", _REMOVE_KEY, lambda: self._apply("remove")))

        layout = self.review_widget.layout()
        for text, key, callback in actions:
            button = QPushButton(text)
            # clicked passes a 'checked' bool, which must not reach the callback.
            button.clicked.connect(lambda checked=False, callback=callback: callback())
            layout.addWidget(button)
            self._action_buttons.append(button)
            self._add_shortcut(key, callback)

        undo_button = QPushButton(f"Undo  [{_UNDO_KEY}]")
        undo_button.clicked.connect(self._undo)
        self._add_shortcut(_UNDO_KEY, self._undo)
        finish_button = QPushButton("Finish Review")
        finish_button.clicked.connect(self._finish)
        layout.addWidget(undo_button)
        layout.addWidget(finish_button)

    def _add_shortcut(self, key, callback):
        # Application-wide, so the keys also work while the canvas has focus and beat napari's layer shortcuts.
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(callback)
        self._shortcuts.append(shortcut)

    def _stop_review(self):
        for shortcut in self._shortcuts:
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self._shortcuts, self._action_buttons = [], []
        layout = self.review_widget.layout()
        while layout.count():
            layout.takeAt(0).widget().deleteLater()

        if self._source is not None and self._source in self.viewer.layers:
            show_selected, selected, visible = self._previous_display
            self._source.show_selected_label = show_selected
            self._source.selected_label = selected
            self._source.visible = visible
        self._source, self._targets, self._instances, self._history = None, [], [], []
        self.start_button.setText("Start Review")

    def _finish(self):
        summary = self._summary()
        self._stop_review()
        self.progress_label.setText(f"Review finished. {summary}")

    def _summary(self):
        actions = [entry["action"] for entry in self._history]
        parts = [f"kept {actions.count('keep')}", f"removed {actions.count('remove')}"]
        parts += [f"moved {actions.count(target.name)} to {target.name}" for target in self._targets]
        return ", ".join(parts) + "."

    def _current_coords(self):
        seg_id, bbox = self._instances[self._index]
        coords = np.nonzero(self._source.data[bbox] == seg_id)
        return tuple(c + sl.start for c, sl in zip(coords, bbox))

    def _show_current(self):
        # Skip instances that no longer exist, e.g. because they were painted over.
        while self._index < len(self._instances) and self._current_coords()[0].size == 0:
            self._index += 1

        n = len(self._instances)
        at_end = self._index >= n
        for button in self._action_buttons:
            button.setEnabled(not at_end)
        if at_end:
            self.progress_label.setText(
                f"All {n} instances reviewed: {self._summary()}\n"
                "Press 'Undo' to go back or 'Finish Review' to release the shortcuts."
            )
            return

        seg_id, bbox = self._instances[self._index]
        self.progress_label.setText(f"Instance {self._index + 1} / {n} (id {seg_id})")
        self._source.selected_label = seg_id
        self._focus(bbox)

    def _focus(self, bbox):
        layer = self._source
        start = np.array([sl.start for sl in bbox])
        stop = np.array([sl.stop for sl in bbox])
        center = np.asarray(layer.data_to_world((start + stop - 1) / 2))

        # The layer occupies the last world axes if other layers have more dimensions.
        offset = self.viewer.dims.ndim - layer.ndim
        for axis, value in enumerate(center):
            self.viewer.dims.set_point(axis + offset, value)

        displayed = [axis - offset for axis in self.viewer.dims.displayed if axis >= offset]
        self.viewer.camera.center = tuple(center[displayed])
        if self.zoom_to_instance:
            extent = (stop - start) * np.abs(layer.scale)
            field_of_view = self.context_param.value() * np.max(extent[displayed])
            self.viewer.camera.zoom = np.min(self.viewer._canvas_size) / field_of_view

    def _apply(self, action):
        if self._source is None or self._index >= len(self._instances):
            return
        seg_id, _ = self._instances[self._index]
        entry = {"index": self._index, "action": "keep"}

        if action != "keep":
            coords = self._current_coords()
            self._source.data_setitem(coords, 0)
            entry.update(action="remove", coords=coords, seg_id=seg_id)

            if action != "remove":
                target = action
                ids = self._target_ids[target.name]
                # Keep the segmentation id unless the target already uses it.
                target_id = seg_id if seg_id not in ids else max(ids) + 1
                ids.add(target_id)
                entry.update(
                    action=target.name, target=target, target_id=target_id,
                    target_previous=np.array(target.data[coords], copy=True),
                )
                target.data_setitem(coords, target_id)

        self._history.append(entry)
        self._index += 1
        self._show_current()

    def _undo(self):
        if self._source is None or not self._history:
            return
        entry = self._history.pop()
        if "target" in entry:
            entry["target"].data_setitem(entry["coords"], entry["target_previous"])
            self._target_ids[entry["target"].name].discard(entry["target_id"])
        if "coords" in entry:
            self._source.data_setitem(entry["coords"], entry["seg_id"])
        self._index = entry["index"]
        self._show_current()
