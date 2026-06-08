import napari
import napari.layers
import numpy as np

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .base_widget import BaseWidget
from ..cristae_analysis import approximate_membrane, compute_mito_crista_statistics, detect_contact_sites


class CristaeAnalysisWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        self.crista_selector_name = "Crista Mask"
        self.mito_selector_name = "Mito Segmentation"

        self.crista_selector_widget = self._create_layer_selector(self.crista_selector_name, layer_type="Labels")
        self.mito_selector_widget = self._create_layer_selector(self.mito_selector_name, layer_type="Labels")

        self.settings = self._create_settings_widget()

        self.run_button = QPushButton("Run Cristae Analysis")
        self.run_button.clicked.connect(self.on_run)

        layout.addWidget(self.crista_selector_widget)
        layout.addWidget(self.mito_selector_widget)
        layout.addWidget(self.settings)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def _create_settings_widget(self):
        setting_values = QWidget()
        setting_values.setLayout(QVBoxLayout())

        self.save_path, layout = self._add_path_param(name="save_path", select_type="file", value="")
        setting_values.layout().addLayout(layout)

        self.voxel_size_param, layout = self._add_float_param(
            "voxel_size", 0.0, min_val=0.0, max_val=100.0,
            title="Voxel Size (nm, 0 = auto)", step=0.1,
        )
        setting_values.layout().addLayout(layout)

        self.ims_thickness_param, layout = self._add_float_param(
            "ims_thickness", 7.0, min_val=1.0, max_val=50.0,
            title="IMS Thickness (nm)", decimals=1, step=0.5,
        )
        setting_values.layout().addLayout(layout)

        self.mm_thickness_param, layout = self._add_float_param(
            "mm_thickness", 8.0, min_val=1.0, max_val=30.0,
            title="Membrane Thickness (nm)", decimals=1, step=0.5,
        )
        setting_values.layout().addLayout(layout)

        return self._make_collapsible(widget=setting_values, title="Advanced Settings")

    def on_run(self):
        crista_mask = self._get_layer_selector_data(self.crista_selector_name)
        mito_seg = self._get_layer_selector_data(self.mito_selector_name)

        if crista_mask is None or mito_seg is None:
            show_info("Please select both a crista mask and a mito segmentation layer.")
            return

        metadata = self._get_layer_selector_data(self.crista_selector_name, return_metadata=True)
        voxel_size = self._handle_resolution(metadata, self.voxel_size_param, crista_mask.ndim, return_as_list=False)

        if voxel_size is None:
            show_info("Please provide a voxel size (or ensure layer metadata contains voxel_size).")
            return

        ims_thickness = self.ims_thickness_param.value()
        mm_thickness = self.mm_thickness_param.value()

        show_info("INFO: Approximating mitochondrial membranes...")
        om_mask, imm_mask = approximate_membrane(
            mito_seg, voxel_size,
            outer_membrane_thickness_nm=mm_thickness,
            inner_membrane_thickness_nm=mm_thickness,
            ims_thickness_nm=ims_thickness,
        )

        show_info("INFO: Running cristae analysis per mitochondrion...")
        stats_df = compute_mito_crista_statistics(
            crista_mask, mito_seg, voxel_size,
            om_mask=om_mask, imm_mask=imm_mask,
        )

        # Add contact sites as a Points layer.
        contact_coords, contact_summary = detect_contact_sites(
            crista_mask.astype(bool), imm_mask, voxel_size
        )
        if contact_coords.shape[0] > 0:
            self.viewer.add_points(
                contact_coords,
                name="Crista-IMM Contacts",
                size=3,
                face_color="orange",
                blending="additive",
            )

        # Attach per-mito stats table to the mito segmentation layer.
        mito_layer = self._get_layer_selector_layer(self.mito_selector_name)
        self._add_properties_and_table(mito_layer, stats_df, save_path=self.save_path.text())

        n_mito = len(stats_df)
        n_contacts = contact_summary["contact_region_count"]
        show_info(
            f"INFO: Cristae analysis complete — {n_mito} mitochondria, "
            f"{n_contacts} contact regions detected."
        )
