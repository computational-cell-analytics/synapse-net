import napari
import numpy as np

from napari.utils import progress
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .base_widget import BaseWidget
from ..cristae_analysis import approximate_membrane, compute_mito_crista_statistics, detect_contact_sites


class CristaeAnalysisWidget(BaseWidget):
    # Crista-orientation dropdown labels -> the `method` argument of compute_mito_crista_statistics.
    _ORIENTATION_FAST = "Fast (downsampled, approximate)"
    _ORIENTATION_TO_METHOD = {
        _ORIENTATION_FAST: "fast",
        "Exact (full resolution)": "exact",
        "Skip (no orientation)": "skip",
    }

    # Membrane-mode dropdown labels -> the `membrane_mode` argument of approximate_membrane.
    _MEMBRANE_SLICE_2D = "2D per-slice (z-parallel)"
    _MEMBRANE_TO_MODE = {
        _MEMBRANE_SLICE_2D: "slice_2d",
        "3D connected shell": "shell_3d",
    }

    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        self.crista_selector_name = "Crista Mask"
        self.mito_selector_name = "Mito Segmentation"

        # Auto-default each input to a layer whose name contains the matching keyword.
        self.crista_selector_widget = self._create_layer_selector(
            self.crista_selector_name, layer_type="Labels", prefer_substring="cristae")
        self.mito_selector_widget = self._create_layer_selector(
            self.mito_selector_name, layer_type="Labels", prefer_substring="mitochondria")

        self.settings = self._create_settings_widget()

        # Cheap preview of the membrane + junctions (the front-end of the pipeline) so the user can
        # tune Membrane Thickness / Border Gap and inspect them before the expensive per-mito run.
        self.preview_button = QPushButton("Preview Membrane && Junctions")
        self.preview_button.clicked.connect(self.on_preview)

        self.run_button = QPushButton("Run Cristae Analysis")
        self.run_button.clicked.connect(self.on_run)

        layout.addWidget(self.crista_selector_widget)
        layout.addWidget(self.mito_selector_widget)
        layout.addWidget(self.settings)
        layout.addWidget(self.preview_button)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    # Result layer names, shared by the preview and the full run so re-previewing / running updates
    # the same layers instead of duplicating them.
    _MEMBRANE_LAYER = "Membrane Mask"
    _JUNCTION_LAYER = "Crista-Membrane Junctions"

    def _create_settings_widget(self):
        setting_values = QWidget()
        setting_values.setLayout(QVBoxLayout())

        self.save_path, layout = self._add_path_param(
            name="save_path", select_type="file", value="",
            tooltip="Path to save the analysis results CSV file. An empty path will skip saving.",
        )
        setting_values.layout().addLayout(layout)

        self.voxel_size_param, layout = self._add_float_param(
            "voxel_size", 0.0, min_val=0.0, max_val=100.0,
            title="Voxel Size (nm, 0 = auto)", step=0.1,
            tooltip="Voxel size of the input volume in nanometers. Set to 0 (default) to auto-detect from layer metadata.",
        )
        setting_values.layout().addLayout(layout)

        self.mm_thickness_param, layout = self._add_float_param(
            "mm_thickness", 8.0, min_val=1.0, max_val=30.0,
            title="Membrane Thickness (nm)", decimals=1, step=0.5,
            tooltip="Thickness of the mitochondrial membrane shell in nanometers.",
        )
        setting_values.layout().addLayout(layout)

        self.border_gap_param, layout = self._add_float_param(
            "border_gap", 0.0, min_val=0.0, max_val=100.0,
            title="Border Gap (nm, 0 = same as membrane)", decimals=1, step=0.5,
            tooltip="Distance from each volume face within which membrane voxels are suppressed. "
                    "Set to 0 to use the same value as Membrane Thickness.",
        )
        setting_values.layout().addLayout(layout)

        self.show_membranes_param = self._add_boolean_param(
            "show_membranes", False,
            title="Show Membrane Mask",
            tooltip="Add the approximated mitochondrial membrane mask as a layer after running.",
        )
        setting_values.layout().addWidget(self.show_membranes_param)

        self.orientation_param, layout = self._add_choice_param(
            "orientation", self._ORIENTATION_FAST, list(self._ORIENTATION_TO_METHOD.keys()),
            title="Crista orientation",
            tooltip="How to compute the crista orientation anisotropy — the most expensive stage "
                    "(structure tensor). All other metrics (surface areas, junction distances, "
                    "thickness) are identical regardless of this choice.\n"
                    "- Fast (downsampled, approximate): ~8x faster; a relative indicator only, not "
                    "comparable in magnitude to the exact value.\n"
                    "- Exact (full resolution): the true anisotropy (slowest).\n"
                    "- Skip (no orientation): fastest; leaves the orientation column empty.",
        )
        setting_values.layout().addLayout(layout)

        self.membrane_mode_param, layout = self._add_choice_param(
            "membrane_mode", self._MEMBRANE_SLICE_2D, list(self._MEMBRANE_TO_MODE.keys()),
            title="Membrane mode",
            tooltip="How the membrane shell is approximated.\n"
                    "- 2D per-slice (z-parallel): erode each Z-slice independently in XY (no z-bleed); "
                    "the shell has no Z-caps and can fragment across slices (some junction pairs may "
                    "then have no along-membrane path).\n"
                    "- 3D connected shell: a single connected 3D shell including the Z-caps (no "
                    "fragmentation), somewhat slower; thickness acts in all axes.",
        )
        setting_values.layout().addLayout(layout)

        return self._make_collapsible(widget=setting_values, title="Advanced Settings")

    def _read_inputs(self):
        """Validate the selected layers/voxel size and read the shared run/preview parameters.

        Returns (crista_mask, mito_seg, voxel_size, layer_scale, layer_translate, mm_thickness,
        border_gap) or None (after showing a guidance message) if inputs are incomplete.
        """
        crista_mask = self._get_layer_selector_data(self.crista_selector_name)
        mito_seg = self._get_layer_selector_data(self.mito_selector_name)
        if crista_mask is None or mito_seg is None:
            show_info("Please select both a crista mask and a mito segmentation layer.")
            return None

        metadata = self._get_layer_selector_data(self.crista_selector_name, return_metadata=True)
        voxel_size = self._handle_resolution(metadata, self.voxel_size_param, crista_mask.ndim, return_as_list=False)
        if voxel_size is None:
            show_info("Please provide a voxel size (or ensure layer metadata contains voxel_size).")
            return None

        # Inherit the display scale/translate of the source layer so the result layers overlay
        # the input correctly (e.g. when the raw data was loaded with a physical voxel scale).
        ref_layer = self._get_layer_selector_layer(self.crista_selector_name)
        layer_scale = None if ref_layer is None else ref_layer.scale
        layer_translate = None if ref_layer is None else ref_layer.translate

        mm_thickness = self.mm_thickness_param.value()
        border_gap_val = self.border_gap_param.value()
        border_gap = border_gap_val if border_gap_val > 0.0 else None
        membrane_mode = self._MEMBRANE_TO_MODE[self.membrane_mode_param.currentText()]
        return (crista_mask, mito_seg, voxel_size, layer_scale, layer_translate,
                mm_thickness, border_gap, membrane_mode)

    def _compute_membrane_and_contacts(self, mito_seg, crista_mask, voxel_size, mm_thickness,
                                       border_gap, membrane_mode):
        """The cheap front-end shared by preview and run: membrane shell + crista-membrane junctions."""
        membrane_mask = approximate_membrane(
            mito_seg, voxel_size,
            membrane_thickness_nm=mm_thickness, border_gap_nm=border_gap,
            n_jobs=-1,  # parallelize across cores (per-Z-slice in 2D mode).
            membrane_mode=membrane_mode,
        )
        contact_labels, contact_summary = detect_contact_sites(
            crista_mask.astype(bool), membrane_mask, voxel_size
        )
        return membrane_mask, contact_labels, contact_summary

    def _add_or_update_labels(self, name, data, scale, translate, opacity=None):
        """Add a Labels layer, or refresh it in place if one with this name already exists."""
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            layer.data = data
            if opacity is not None:
                layer.opacity = opacity
        else:
            kwargs = {} if opacity is None else {"opacity": opacity}
            self.viewer.add_labels(data, name=name, scale=scale, translate=translate, **kwargs)

    def on_preview(self):
        """Compute and show ONLY the membrane + junctions (seconds) — the front-end of the pipeline —
        so the user can tune Membrane Thickness / Border Gap before the expensive per-mito run."""
        inputs = self._read_inputs()
        if inputs is None:
            return
        (crista_mask, mito_seg, voxel_size, layer_scale, layer_translate,
         mm_thickness, border_gap, membrane_mode) = inputs

        show_info("INFO: Previewing membrane & junctions...")
        membrane_mask, contact_labels, contact_summary = self._compute_membrane_and_contacts(
            mito_seg, crista_mask, voxel_size, mm_thickness, border_gap, membrane_mode
        )
        self._add_or_update_labels(
            self._MEMBRANE_LAYER, membrane_mask.astype(np.uint8), layer_scale, layer_translate, opacity=0.4
        )
        if contact_labels.max() > 0:
            self._add_or_update_labels(
                self._JUNCTION_LAYER, contact_labels.astype(np.uint32), layer_scale, layer_translate
            )
        else:
            show_info("INFO: No crista–membrane junctions detected at these settings.")
        show_info(
            f"INFO: Preview — {int(membrane_mask.sum())} membrane voxels, "
            f"{contact_summary['crista_junction_count']} junctions. "
            "Adjust Membrane Thickness / Border Gap and preview again, or Run."
        )

    def on_run(self):
        inputs = self._read_inputs()
        if inputs is None:
            return
        (crista_mask, mito_seg, voxel_size, layer_scale, layer_translate,
         mm_thickness, border_gap, membrane_mode) = inputs

        show_info("INFO: Approximating mitochondrial membrane & junctions...")
        membrane_mask, contact_labels, contact_summary = self._compute_membrane_and_contacts(
            mito_seg, crista_mask, voxel_size, mm_thickness, border_gap, membrane_mode
        )

        method = self._ORIENTATION_TO_METHOD[self.orientation_param.currentText()]

        show_info(f"INFO: Running cristae analysis per mitochondrion (orientation: {method})...")
        pbar = {"bar": None}

        def _on_progress(done, total):
            # Runs on the GUI thread (the joblib results generator is consumed by the caller),
            # so updating the napari progress bar here needs no cross-thread marshaling.
            if pbar["bar"] is None:
                pbar["bar"] = progress(total=total, desc="Cristae analysis")
            pbar["bar"].update(1)

        try:
            stats_df = compute_mito_crista_statistics(
                crista_mask, mito_seg, voxel_size,
                membrane_mask=membrane_mask,
                membrane_thickness_nm=mm_thickness,
                border_gap_nm=border_gap,
                method=method,
                membrane_mode=membrane_mode,
                n_jobs=-1,  # mitochondria are independent — use all cores.
                verbose=True,  # terminal tqdm bar.
                progress_callback=_on_progress,  # napari activity-dock bar.
            )
        finally:
            if pbar["bar"] is not None:
                pbar["bar"].close()

        if self.show_membranes_param.isChecked():
            self._add_or_update_labels(
                self._MEMBRANE_LAYER, membrane_mask.astype(np.uint8), layer_scale, layer_translate, opacity=0.4
            )

        # Crista-membrane junctions as a Labels layer (each junction has its own ID).
        if contact_labels.max() > 0:
            self._add_or_update_labels(
                self._JUNCTION_LAYER, contact_labels.astype(np.uint32), layer_scale, layer_translate
            )
        else:
            show_info("INFO: No crista–membrane junctions detected — junction layer not added.")

        # Attach per-mito stats table to the mito segmentation layer.
        mito_layer = self._get_layer_selector_layer(self.mito_selector_name)
        self._add_properties_and_table(mito_layer, stats_df, save_path=self.save_path.text())

        n_mito = len(stats_df)
        n_contacts = contact_summary["crista_junction_count"]
        show_info(
            f"INFO: Cristae analysis complete — {n_mito} mitochondria, "
            f"{n_contacts} crista junction sites detected."
        )
