"""Temporary widget for the LDCV split-A label check: step through pre-cropped vesicles and record a label for each."""
import csv
import json
import os
from datetime import datetime

import napari
import numpy as np

from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QShortcut, QVBoxLayout, QWidget
)

from .base_widget import BaseWidget

_KEEP_KEY, _ACCEPT_KEY = "K", "A"
_LABEL_CHOICES = ["type1", "type2", "type3", "type4", "other", "not a vesicle", "unsure"]
_RAW_LAYER, _MASK_LAYER = "review: raw", "review: vesicle"
_CSV_FIELDS = [
    "index", "tomogram", "instance_id", "z", "y", "x", "gt_label", "suggested_label", "suggested_prob",
    "decision", "final_label", "comment", "reviewer", "timestamp",
]
# Vesicles where all 6 models (split A and B seed ensembles) contradict the label; reviewed first.
_ALL_MODELS_AGREE = {
    "tomogram-009": {2, 75, 83},
    "tomogram-040": {463, 467},
    "tomogram-046": {303, 304},
    "tomogram-057": {2, 3, 24, 42, 106, 320, 373, 507, 511, 523, 526, 528, 535, 536, 537, 538, 539, 546, 551},
}


def _is_priority(item):
    return item["instance_id"] in _ALL_MODELS_AGREE.get(item["tomogram"], ())


class _ReviewData:
    """The review zip (a numpy .npz archive) or the folder it was extracted to."""

    def __init__(self, path):
        if os.path.isdir(path):
            self._get = lambda key: np.load(os.path.join(path, f"{key}.npy"))
        else:
            npz = np.load(path)
            self._get = npz.__getitem__
        meta = json.loads(str(self._get("manifest")))
        self.models, self.items = meta["models"], meta["items"]

    def crop(self, index):
        return self._get(f"raw_{index}"), self._get(f"mask_{index}")


class LabelReviewWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        self.data_path = ""
        self.data_path_param, data_layout = self._add_path_param(
            "data_path", "", select_type="both", title="Review data",
            tooltip="The zip file you received, or the folder it was extracted to.",
        )
        self.reviewer = ""
        self.reviewer_param, reviewer_layout = self._add_string_param(
            "reviewer", "", title="Your name", placeholder="e.g. initials",
            tooltip="Stored with every decision and used in the name of the result file.",
        )
        self.load_button = QPushButton("Load / Resume")
        self.load_button.clicked.connect(self.on_load)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setTextFormat(Qt.RichText)

        self.keep_button = QPushButton("")
        self.keep_button.clicked.connect(lambda: self._decide("keep_gt"))
        self.accept_button = QPushButton("")
        self.accept_button.clicked.connect(lambda: self._decide("accept_suggestion"))

        self.label_choice = QComboBox()
        self.label_choice.setEditable(True)
        self.label_choice.addItems(_LABEL_CHOICES)
        self.label_choice.setToolTip("Pick a label or type your own.")
        self.set_button = QPushButton("Use this label")
        self.set_button.clicked.connect(lambda: self._decide("new_label"))
        new_label_layout = QHBoxLayout()
        new_label_layout.addWidget(self.label_choice)
        new_label_layout.addWidget(self.set_button)

        self.comment = QLineEdit()
        self.comment.setPlaceholderText("Optional comment, saved with the decision")

        self.prev_button = QPushButton("< Previous")
        self.prev_button.clicked.connect(lambda: self._go(self._pos - 1))
        self.next_button = QPushButton("Next >")
        self.next_button.clicked.connect(lambda: self._go(self._pos + 1))
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.export_button = QPushButton("Export Results (CSV)")
        self.export_button.clicked.connect(self.on_export)

        self.review_widget = QWidget()
        review_layout = QVBoxLayout()
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_layout.addWidget(self.info_label)
        review_layout.addWidget(self.keep_button)
        review_layout.addWidget(self.accept_button)
        review_layout.addLayout(new_label_layout)
        review_layout.addWidget(self.comment)
        review_layout.addLayout(nav_layout)
        review_layout.addWidget(self.progress_label)
        review_layout.addWidget(self.export_button)
        self.review_widget.setLayout(review_layout)
        self.review_widget.setEnabled(False)

        layout.addLayout(data_layout)
        layout.addLayout(reviewer_layout)
        layout.addWidget(self.load_button)
        layout.addWidget(self.review_widget)
        layout.addStretch()
        self.setLayout(layout)

        # Application-wide, so the keys also work while the canvas has focus; text fields still get typed letters.
        self._shortcuts = []
        for key, decision in [(_KEEP_KEY, "keep_gt"), (_ACCEPT_KEY, "accept_suggestion")]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(lambda decision=decision: self._decide(decision))
            shortcut.setEnabled(False)
            self._shortcuts.append(shortcut)

        # _index is the item's position in the zip (and the CSV), _pos its position in the review order.
        self._data, self._decisions, self._index, self._pos, self._order, self._results_path = None, {}, 0, 0, [], None

    def on_load(self):
        path = self.data_path.strip()
        if not os.path.exists(path):
            show_info("INFO: Please select the review zip file or its extracted folder.")
            return
        reviewer = self.reviewer.strip()
        if not reviewer:
            show_info("INFO: Please enter your name first.")
            return
        try:
            data = _ReviewData(path)
        except Exception as e:
            show_info(f"INFO: Could not read the review data: {e}")
            return

        self._data = data
        folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
        safe_name = "".join(c if c.isalnum() else "_" for c in reviewer)
        self._results_path = os.path.join(folder, f"ldcv_label_review_{safe_name}.csv")
        self._decisions = self._read_results(self._results_path)
        self._order = sorted(range(len(data.items)), key=lambda i: not _is_priority(data.items[i]))

        self.review_widget.setEnabled(True)
        for shortcut in self._shortcuts:
            shortcut.setEnabled(True)
        undecided = [pos for pos, i in enumerate(self._order) if i not in self._decisions]
        self._go(undecided[0] if undecided else 0)
        if self._decisions:
            show_info(f"INFO: Resumed with {len(self._decisions)} saved decisions from {self._results_path}.")

    @staticmethod
    def _read_results(path):
        if not os.path.exists(path):
            return {}
        with open(path, newline="") as f:
            return {int(row["index"]): row for row in csv.DictReader(f)}

    def _priority_left(self):
        return sum(1 for i, item in enumerate(self._data.items) if _is_priority(item) and i not in self._decisions)

    def _write_results(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for index in sorted(self._decisions):
                writer.writerow(self._decisions[index])

    def _go(self, pos):
        if self._data is None or not 0 <= pos < len(self._order):
            return
        self._pos = pos
        self._index = index = self._order[pos]
        item = self._data.items[index]
        raw, mask = self._data.crop(index)

        lo, hi = np.percentile(raw, [1, 99])
        self._add_or_update_layer(
            self.viewer.add_image, _RAW_LAYER, raw, None, None, {"contrast_limits": (float(lo), float(hi))}
        )
        mask_layer = self.add_or_update_labels(_MASK_LAYER, mask, opacity=0.8)
        mask_layer.contour = 2  # not a constructor argument in napari 0.5
        self.viewer.layers.selection.active = self.viewer.layers[_RAW_LAYER]

        centre = np.array(item["centre_zyx"]) - np.array(item["crop_start_zyx"])
        self.viewer.dims.set_point(0, centre[0])
        self.viewer.reset_view()

        self._show_info(item)

    def _show_info(self, item):
        pos, n = self._pos, len(self._order)
        decision = self._decisions.get(self._index)
        current = "not decided yet" if decision is None else f"<b>{decision['final_label']}</b> ({decision['decision']})"
        self.info_label.setText(
            f"<b>Vesicle {pos + 1} / {n}</b> &nbsp; {item['tomogram']}, instance {item['instance_id']}<br>"
            + ("<b>High priority</b>: all 6 models agree<br>" if _is_priority(item) else "")
            + f"Ground-truth label: <b>{item['gt_label']}</b><br>"
            f"Models suggest: <b>{item['suggested_label']}</b> (p = {item['suggested_prob']:.2f})<br>"
            f"Your decision: {current}"
        )
        self.keep_button.setText(f"Keep ground truth: {item['gt_label']}  [{_KEEP_KEY}]")
        self.accept_button.setText(f"Accept suggestion: {item['suggested_label']}  [{_ACCEPT_KEY}]")
        self.comment.setText("" if decision is None else decision["comment"])
        self.prev_button.setEnabled(pos > 0)
        self.next_button.setEnabled(pos < n - 1)
        self.progress_label.setText(f"{len(self._decisions)} / {n} decided. Saved automatically to {self._results_path}")

    def _decide(self, decision):
        if self._data is None:
            return
        item = self._data.items[self._index]
        if decision == "keep_gt":
            final = item["gt_label"]
        elif decision == "accept_suggestion":
            final = item["suggested_label"]
        else:
            final = self.label_choice.currentText().strip()
            if not final:
                show_info("INFO: Please choose or type a label.")
                return

        priority_left_before = self._priority_left()
        z, y, x = item["centre_zyx"]
        self._decisions[self._index] = {
            "index": self._index, "tomogram": item["tomogram"], "instance_id": item["instance_id"],
            "z": z, "y": y, "x": x, "gt_label": item["gt_label"], "suggested_label": item["suggested_label"],
            "suggested_prob": item["suggested_prob"], "decision": decision, "final_label": final,
            "comment": self.comment.text().strip(), "reviewer": self.reviewer.strip(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            self._write_results(self._results_path)
        except OSError as e:
            show_info(f"INFO: Could not save to {self._results_path} ({e}). Use 'Export Results' to save elsewhere.")

        if priority_left_before and not self._priority_left():
            show_info("INFO: All high-priority vesicles (all 6 models agree) are done. The rest are optional.")

        if self._pos < len(self._order) - 1:
            self._go(self._pos + 1)
        else:
            self._show_info(item)
            undecided = len(self._data.items) - len(self._decisions)
            show_info("INFO: Last vesicle reached." + (f" {undecided} still undecided." if undecided else " All done."))

    def on_export(self):
        if self._data is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export results", self._results_path, "CSV files (*.csv)")
        if path:
            self._write_results(path)
            show_info(f"INFO: Saved {len(self._decisions)} decisions to {path}.")
