import napari
import numpy as np

from napari.utils import progress
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .base_widget import BaseWidget
from ..cristae_analysis import (
    approximate_membrane, compute_crista_skeleton, compute_mito_crista_statistics, detect_junctions,
    _border_zone, _open_trimmed_mesh, _gap_radius, _to_sampling,
)


class CristaeAnalysisWidget(BaseWidget):
    """Napari widget for the cristae analysis (preview + full per-mitochondrion run).

    ``_ORIENTATION_TO_METHOD`` maps the orientation dropdown labels to the ``method`` argument of
    :func:`~synapse_net.cristae_analysis.compute_mito_crista_statistics`, ``_MEMBRANE_TO_MODE`` maps
    the membrane-mode labels to the ``membrane_mode`` argument of
    :func:`~synapse_net.cristae_analysis.approximate_membrane`, and ``_JUNCTION_TO_MODE`` maps the
    junction-mode labels to the ``junction_mode`` argument of
    :func:`~synapse_net.cristae_analysis.detect_junctions`. The ``_*_LAYER`` name constants are
    shared by the preview and the full run so re-previewing / running updates the same layers instead
    of duplicating them.
    """

    _ORIENTATION_FAST = "Fast (downsampled, approximate)"
    _ORIENTATION_SKIP = "Skip (no orientation)"
    _ORIENTATION_TO_METHOD = {
        _ORIENTATION_FAST: "fast",
        "Exact (full resolution)": "exact",
        _ORIENTATION_SKIP: "skip",
    }

    _MEMBRANE_SLICE_2D = "2D per-slice (z-parallel)"
    _MEMBRANE_TO_MODE = {
        _MEMBRANE_SLICE_2D: "slice_2d",
        "3D connected shell": "shell_3d",
    }

    _JUNCTION_OVERLAP = "Overlap (crista ∩ membrane)"
    _JUNCTION_TO_MODE = {
        _JUNCTION_OVERLAP: "overlap",
        "Skeleton (crista reaching the membrane)": "skeleton",
    }

    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        self.crista_selector_name = "Crista Mask"
        self.mito_selector_name = "Mito Segmentation"

        self.crista_selector_widget = self._create_layer_selector(
            self.crista_selector_name, layer_type="Labels", prefer_substring="cristae")
        self.mito_selector_widget = self._create_layer_selector(
            self.mito_selector_name, layer_type="Labels", prefer_substring="mitochondria")

        self.settings = self._create_settings_widget()

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

    _MEMBRANE_LAYER = "Membrane Mask"
    _MEMBRANE_MESH_LAYER = "Membrane Mesh"
    _JUNCTION_LAYER = "Crista-Membrane Junctions"
    _SKELETON_LAYER = "Crista Skeleton"
    _SKELETON_TERMINI_LAYER = "Crista Skeleton Termini"

    def _create_settings_widget(self):
        setting_values = QWidget()
        setting_values.setLayout(QVBoxLayout())

        self.save_path, layout = self._add_path_param(
            name="save_path", select_type="file", value="",
            tooltip="Path to save the analysis results CSV file. An empty path will skip saving. "
                    "See docs/cristae_analysis.md for how each column is computed.",
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
            title="Show Membrane Mesh",
            tooltip="Add the eroded-mito (lumen) inner surface — the single-wall surface the junction "
                    "geodesics run along — as a mesh (napari Surface layer) after running.",
        )
        setting_values.layout().addWidget(self.show_membranes_param)

        self.show_skeleton_param = self._add_boolean_param(
            "show_skeleton", False,
            title="Show Crista Skeleton",
            tooltip="Add the crista centerline skeleton as a napari Vectors layer (connected line "
                    "segments) plus a Points layer for the termini (the skeleton's free ends), both "
                    "restricted to the mito segmentation so they match what the analysis looks at. "
                    "Skeleton mode keys its terminus filter on exactly those termini, so this is how "
                    "you check whether a flagged junction sits at a real crista end or merely on a "
                    "flank. Available in both junction modes.",
        )
        setting_values.layout().addWidget(self.show_skeleton_param)

        self.orientation_param, layout = self._add_choice_param(
            "orientation", self._ORIENTATION_SKIP, list(self._ORIENTATION_TO_METHOD.keys()),
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

        self.junction_mode_param, layout = self._add_choice_param(
            "junction_mode", self._JUNCTION_OVERLAP, list(self._JUNCTION_TO_MODE.keys()),
            title="Junction detection",
            tooltip="How crista–membrane junctions are found.\n"
                    "- Overlap (crista ∩ membrane): counts the connected components where the crista "
                    "mask directly overlaps the membrane band. A crista that stops short of the "
                    "membrane scores no junction, and a lamella rim can fragment into many blobs.\n"
                    "- Skeleton (crista reaching the membrane): counts crista regions that come "
                    "within Max Extension of the membrane near a crista terminus (an end of the "
                    "crista skeleton). Tolerates a crista segmented short of the membrane, and never "
                    "merges two separate cristae into one junction. Requires 3D data and reports the "
                    "closest approach as mean_junction_extension_nm.",
        )
        setting_values.layout().addLayout(layout)

        self.max_extension_param, layout = self._add_float_param(
            "max_extension", 0.0, min_val=0.0, max_val=100.0,
            title="Max Extension (nm, 0 = same as membrane)", decimals=1, step=0.5,
            tooltip="How far a crista may fall short of the membrane and still count as a junction "
                    "(Skeleton mode only). Set to 0 to use the same value as Membrane Thickness. "
                    "Raising it well above the membrane thickness starts counting cristae that merely "
                    "pass near the boundary, and can fuse junctions whose near-membrane regions touch.",
        )
        setting_values.layout().addLayout(layout)

        self.min_junction_volume_param, layout = self._add_float_param(
            "min_junction_volume", 0.0, min_val=0.0, max_val=100000.0,
            title="Min Junction Volume (nm³, 0 = default 50)", decimals=1, step=10.0,
            tooltip="Junction regions smaller than this are discarded (Skeleton mode only). This only "
                    "removes specks — it does NOT fix the fact that skeleton mode over-counts on "
                    "densely packed cristae, where the spurious regions are full-sized. Set to 0 to "
                    "use the default of 50 nm³.",
        )
        setting_values.layout().addLayout(layout)

        self.terminus_param, layout = self._add_float_param(
            "terminus_distance", 0.0, min_val=0.0, max_val=200.0,
            title="Terminus Distance (nm, 0 = default 20)", decimals=1, step=0.5,
            tooltip="How close a near-membrane crista region must be to a crista terminus (an end of "
                    "the crista skeleton) to count as a junction (Skeleton mode only). This is what "
                    "separates a crista ENDING at the membrane from one merely running ALONGSIDE it. "
                    "Set to 0 to use the default of 20 nm; raise it to be more permissive.",
        )
        setting_values.layout().addLayout(layout)

        return self._make_collapsible(widget=setting_values, title="Advanced Settings")

    def _read_inputs(self):
        """Validate the selected layers/voxel size and read the shared run/preview parameters.

        ``layer_scale``/``layer_translate`` are inherited from the source (crista) layer so the result
        layers overlay the input correctly (e.g. when the raw data was loaded with a physical voxel
        scale).

        Returns (crista_mask, mito_seg, voxel_size, layer_scale, layer_translate, mm_thickness,
        border_gap, membrane_mode, junction_mode, max_extension, terminus, min_junction_volume)
        or None (after showing
        a guidance message) if inputs are incomplete.
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

        ref_layer = self._get_layer_selector_layer(self.crista_selector_name)
        layer_scale = None if ref_layer is None else ref_layer.scale
        layer_translate = None if ref_layer is None else ref_layer.translate

        mm_thickness = self.mm_thickness_param.value()
        border_gap_val = self.border_gap_param.value()
        border_gap = border_gap_val if border_gap_val > 0.0 else None
        membrane_mode = self._MEMBRANE_TO_MODE[self.membrane_mode_param.currentText()]
        junction_mode = self._JUNCTION_TO_MODE[self.junction_mode_param.currentText()]
        max_extension_val = self.max_extension_param.value()
        max_extension = max_extension_val if max_extension_val > 0.0 else mm_thickness
        terminus_val = self.terminus_param.value()
        terminus = terminus_val if terminus_val > 0.0 else None
        min_volume_val = self.min_junction_volume_param.value()
        min_junction_volume = min_volume_val if min_volume_val > 0.0 else None
        return (crista_mask, mito_seg, voxel_size, layer_scale, layer_translate,
                mm_thickness, border_gap, membrane_mode, junction_mode, max_extension, terminus,
                min_junction_volume)

    def _compute_membrane_and_contacts(self, mito_seg, crista_mask, voxel_size, mm_thickness,
                                       border_gap, membrane_mode, junction_mode, max_extension,
                                       terminus, min_junction_volume):
        """The cheap front-end shared by preview and run: membrane shell + crista-membrane junctions.

        Also returns the border-trimmed lumen (eroded-mito interior) so the run can both display it
        and feed it to the geodesic stage without recomputing the erosion, and the border-zone radius
        so callers can report how much of the volume it covers.

        ``border_radius`` is passed to the junction detector so skeleton mode does not claim junctions
        inside the zone where ``approximate_membrane`` removed the membrane as unknown. Here the arrays
        are the whole volume, so every face is a volume face and ``boundary`` is left at its default.
        """
        membrane_mask, lumen_mask = approximate_membrane(
            mito_seg, voxel_size,
            membrane_thickness_nm=mm_thickness, border_gap_nm=border_gap,
            n_jobs=-1,
            membrane_mode=membrane_mode,
            return_lumen=True,
        )
        border_radius = _gap_radius(voxel_size, mm_thickness, border_gap, mito_seg.ndim)
        contact_labels, contact_summary = detect_junctions(
            crista_mask.astype(bool), membrane_mask, voxel_size,
            junction_mode=junction_mode, max_extension_nm=max_extension,
            terminus_nm=terminus, min_junction_volume_nm3=min_junction_volume,
            border_radius=border_radius, n_jobs=-1, lumen_mask=lumen_mask,
        )
        return membrane_mask, lumen_mask, contact_labels, contact_summary, border_radius

    def _add_skeleton_layers(self, crista_mask, mito_seg, voxel_size, layer_scale, layer_translate):
        """Show the crista centerline skeleton as connected lines and, separately, its termini.

        The termini get their own layer because they are what the skeleton junction mode's terminus
        filter tests against — seeing them is how you tell a junction at a genuine crista end from one
        flagged on a flank.

        Two things make the display correspond to what the detector actually uses. The mask is
        restricted to ``mito_seg > 0``, because the analysis skeletonises each mitochondrion's own
        crista crop; without that the layer showed a skeleton over cristae the detector never looked at.
        (It is not an identity: the detector works per mito *instance*, so a crista spanning two touching
        instances is split there and not here.) And the skeleton is drawn as a **Vectors** layer built
        from the graph edges rather than one point per vertex, so a centerline reads as a curve instead
        of as scattered dots.

        ``compute_crista_skeleton`` returns nm coordinates, so they are divided by the voxel size to
        get array indices: every layer here is added in voxel coordinates with the physical placement
        left to ``scale``/``translate``, matching the membrane mesh.
        """
        crista = crista_mask.astype(bool) & (mito_seg > 0)
        vertices, is_terminus, edges = compute_crista_skeleton(
            crista, voxel_size, n_jobs=-1, return_edges=True
        )
        if len(vertices) == 0:
            show_info("INFO: No crista skeleton to display.")
            return
        vertices = vertices / _to_sampling(voxel_size, crista.ndim)
        if len(edges):
            starts = vertices[edges[:, 0]]
            self.add_or_update_vectors(
                self._SKELETON_LAYER, np.stack([starts, vertices[edges[:, 1]] - starts], axis=1),
                scale=layer_scale, translate=layer_translate,
                edge_width=0.4, edge_color="cyan", vector_style="line", out_of_slice_display=True,
                blending="translucent_no_depth",
            )
        self.add_or_update_points(
            self._SKELETON_TERMINI_LAYER, vertices[is_terminus],
            scale=layer_scale, translate=layer_translate,
            size=3.0, face_color="magenta", border_width=0.0, out_of_slice_display=True,
            blending="translucent_no_depth",
        )
        show_info(
            f"INFO: Skeleton — {len(vertices)} vertices, {len(edges)} segments, "
            f"{int(is_terminus.sum())} termini."
        )

    def on_preview(self):
        """Compute and show ONLY the membrane + junctions (seconds) — the front-end of the pipeline —
        so the user can tune Membrane Thickness / Border Gap before the expensive per-mito run.

        Runs synchronously; :meth:`_computing` provides the busy feedback while it blocks.
        """
        inputs = self._read_inputs()
        if inputs is None:
            return
        (crista_mask, mito_seg, voxel_size, layer_scale, layer_translate,
         mm_thickness, border_gap, membrane_mode, junction_mode, max_extension,
         terminus, min_junction_volume) = inputs

        with self._computing(
            self.preview_button, "Computing preview…", "Preview Membrane && Junctions",
            "INFO: Previewing membrane & junctions...",
        ):
            pbar = progress(total=2, desc="Preview: membrane & junctions")
            try:
                (membrane_mask, _lumen_mask, contact_labels, contact_summary,
                 border_radius) = self._compute_membrane_and_contacts(
                    mito_seg, crista_mask, voxel_size, mm_thickness, border_gap, membrane_mode,
                    junction_mode, max_extension, terminus, min_junction_volume,
                )
                pbar.update(1)
                self.add_or_update_labels(
                    self._MEMBRANE_LAYER, membrane_mask.astype(np.uint8),
                    scale=layer_scale, translate=layer_translate, opacity=0.4,
                )
                if contact_labels.max() > 0:
                    self.add_or_update_labels(
                        self._JUNCTION_LAYER, contact_labels.astype(np.uint32),
                        scale=layer_scale, translate=layer_translate,
                        blending="translucent_no_depth",
                    )
                else:
                    show_info("INFO: No crista–membrane junctions detected at these settings.")
                if self.show_skeleton_param.isChecked():
                    self._add_skeleton_layers(crista_mask, mito_seg, voxel_size, layer_scale, layer_translate)
                pbar.update(1)
                border_fraction = _border_zone(mito_seg.shape, border_radius).mean()
                show_info(
                    f"INFO: Preview — {int(membrane_mask.sum())} membrane voxels, "
                    f"{contact_summary['crista_junction_count']} junctions. "
                    f"Border zone (membrane unknown, junctions suppressed) covers "
                    f"{100 * border_fraction:.0f}% of the volume at Border Gap = "
                    f"{border_radius} voxels. "
                    "Adjust Membrane Thickness / Border Gap and preview again, or Run."
                )
            finally:
                pbar.close()

    def on_run(self):
        """Run the full per-mitochondrion cristae analysis and add the result layers + stats table.

        Runs synchronously; :meth:`_computing` provides the busy feedback while it blocks.
        """
        inputs = self._read_inputs()
        if inputs is None:
            return
        (crista_mask, mito_seg, voxel_size, layer_scale, layer_translate,
         mm_thickness, border_gap, membrane_mode, junction_mode, max_extension,
         terminus, min_junction_volume) = inputs

        with self._computing(
            self.run_button, "Computing analysis…", "Run Cristae Analysis",
            "INFO: Approximating mitochondrial membrane & junctions...",
        ):
            (membrane_mask, lumen_mask, contact_labels, contact_summary,
             border_radius) = self._compute_membrane_and_contacts(
                mito_seg, crista_mask, voxel_size, mm_thickness, border_gap, membrane_mode,
                junction_mode, max_extension, terminus, min_junction_volume,
            )

            method = self._ORIENTATION_TO_METHOD[self.orientation_param.currentText()]
            show_info(f"INFO: Running cristae analysis per mitochondrion (orientation: {method})...")

            # compute_mito_crista_statistics calls progress_callback once per mitochondrion, on this
            # (GUI) thread, so the activity-dock bar can be created/updated here directly.
            pbar = {"bar": None}

            def _on_progress(done, total):
                if pbar["bar"] is None:
                    pbar["bar"] = progress(total=total, desc="Cristae analysis")
                pbar["bar"].update(1)

            try:
                stats_df = compute_mito_crista_statistics(
                    crista_mask, mito_seg, voxel_size,
                    membrane_mask=membrane_mask,
                    lumen_mask=lumen_mask,
                    membrane_thickness_nm=mm_thickness,
                    border_gap_nm=border_gap,
                    method=method,
                    membrane_mode=membrane_mode,
                    junction_mode=junction_mode,
                    max_extension_nm=max_extension,
                    terminus_nm=terminus,
                    min_junction_volume_nm3=min_junction_volume,
                    n_jobs=-1,
                    verbose=True,
                    progress_callback=_on_progress,
                )
            finally:
                if pbar["bar"] is not None:
                    pbar["bar"].close()

            if self.show_membranes_param.isChecked():
                gap_radius = _gap_radius(voxel_size, mm_thickness, border_gap, mito_seg.ndim)
                mesh = _open_trimmed_mesh(
                    lumen_mask, np.ones(mito_seg.ndim), gap_radius, np.ones((mito_seg.ndim, 2), dtype=bool)
                )
                if mesh is not None:
                    verts, faces = mesh
                    self.add_or_update_surface(
                        self._MEMBRANE_MESH_LAYER, verts, faces,
                        scale=layer_scale, translate=layer_translate,
                        opacity=0.4, blending="translucent",
                    )
                else:
                    show_info("INFO: No membrane surface to display at these settings.")

            if self.show_skeleton_param.isChecked():
                self._add_skeleton_layers(crista_mask, mito_seg, voxel_size, layer_scale, layer_translate)

            if contact_labels.max() > 0:
                self.add_or_update_labels(
                    self._JUNCTION_LAYER, contact_labels.astype(np.uint32),
                    scale=layer_scale, translate=layer_translate,
                    blending="translucent_no_depth",
                )
            else:
                show_info("INFO: No crista–membrane junctions detected — junction layer not added.")

            mito_layer = self._get_layer_selector_layer(self.mito_selector_name)
            self._add_properties_and_table(mito_layer, stats_df, save_path=self.save_path.text())

            n_mito = len(stats_df)
            n_contacts = contact_summary["crista_junction_count"]
            show_info(
                f"INFO: Cristae analysis complete — {n_mito} mitochondria, "
                f"{n_contacts} crista junction sites detected."
            )
