from typing import Dict

import napari
import napari.layers
import napari.viewer
import numpy as np
import pandas as pd

from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .base_widget import BaseWidget

# This will fail if we have more than 8 pools.
COLORMAP = ["red", "blue", "yellow", "cyan", "purple", "magenta", "orange", "green"]


class VesiclePoolWidget(BaseWidget):
    def __init__(self):
        super().__init__()

        self.viewer = napari.current_viewer()
        layout = QVBoxLayout()

        # Create the selectors for the layers:
        # 1. Selector for the labels layer with vesicles.
        self.vesicle_selector_name = "Vesicle Segmentation"
        self.vesicle_selector_widget = self._create_layer_selector(self.vesicle_selector_name, layer_type="Labels")
        # 2. Selector for a distance layer.
        self.dist_selector_name1 = "Distances to Structure"
        self.dist_selector_widget1 = self._create_layer_selector(self.dist_selector_name1, layer_type="Shapes")
        # 3. Selector for a second distance layer (optional).
        self.dist_selector_name2 = "Distances to Structure 2"
        self.dist_selector_widget2 = self._create_layer_selector(self.dist_selector_name2, layer_type="Shapes")

        # Add the selector widgets to the layout.
        layout.addWidget(self.vesicle_selector_widget)
        layout.addWidget(self.dist_selector_widget1)
        layout.addWidget(self.dist_selector_widget2)

        # Create the UI elements for defining the vesicle pools:
        # The name of the output name, the name of the vesicle pool, and the criterion for the pool.
        self.pool_layer_name_param, pool_layer_name_layout = self._add_string_param(name="Layer Name", value="")
        self.pool_name_param, pool_name_layout = self._add_string_param(name="Vesicle Pool", value="")
        self.query_param, query_layout = self._add_string_param(
            name="Criterion", value="",
            tooltip="Enter a comma separated criterion (e.g., 'radius > 15, distance > 250') "
            "Possible filters: radius, distance, area, intensity_max, intensity_mean, intensity_min, intensity_std"
        )
        layout.addLayout(pool_layer_name_layout)
        layout.addLayout(pool_name_layout)
        layout.addLayout(query_layout)

        # Create the UI elements for advanced settings and the run button.
        self.settings = self._create_settings_widget()
        self.measure_button = QPushButton("Create Vesicle Pool")
        self.measure_button.clicked.connect(self.on_pool_vesicles)
        layout.addWidget(self.settings)
        layout.addWidget(self.measure_button)

        self.setLayout(layout)

        # The colormap for displaying the vesicle pools.
        self.pool_colors = {}

    def on_pool_vesicles(self):
        segmentation = self._get_layer_selector_data(self.vesicle_selector_name)
        morphology = self._get_layer_selector_layer(self.vesicle_selector_name).properties
        if not morphology:
            morphology = None

        distance_layer = self._get_layer_selector_layer(self.dist_selector_name1)
        distances = None if distance_layer is None else distance_layer.properties
        distance_layer2 = self._get_layer_selector_layer(self.dist_selector_name2)
        # Check if the second distance is the same as the first.
        if distance_layer2.name == distance_layer.name:
            distance_layer2 = None
        distances2 = None if distance_layer2 is None else distance_layer2.properties

        if segmentation is None:
            show_info("INFO: Please choose a segmentation.")
            return
        if self.query_param.text() == "":
            show_info("INFO: Please enter a query string.")
            return
        query = self.query_param.text()

        if self.pool_layer_name_param.text() == "":
            show_info("INFO: Please enter a name for the pool layer.")
            return
        pool_layer_name = self.pool_layer_name_param.text()
        if self.pool_name_param.text() == "":
            show_info("INFO: Please enter a name for the vesicle pool.")
            return
        pool_name = self.pool_name_param.text()

        pool_color = self.pool_color_param.text()
        self._compute_vesicle_pool(
            segmentation, distances, morphology, pool_layer_name, pool_name, query, pool_color, distances2
            )

    def _update_pool_colors(self, pool_name, pool_color):
        if pool_color == "":
            next_color_id = len(self.pool_colors)
            next_color = COLORMAP[next_color_id]
        else:
            # We could check here that this is a valid color.
            next_color = pool_color
        self.pool_colors[pool_name] = next_color

    def _compute_vesicle_pool(
        self,
        segmentation: np.ndarray,
        distances: Dict,
        morphology: Dict,
        pool_layer_name: str,
        pool_name: str,
        query: str,
        pool_color: str,
        distances2: Dict = None
    ):
        """Compute a vesicle pool based on the provided query parameters.

        Args:
            segmentation: Segmentation data (e.g., labeled regions).
            distances: Properties from the distances layer.
            morphology: Properties from the morphology layer.
            pool_layer_name: Name for the new layer to be created.
            pool_name: Name for the pooled group to be assigned.
            query: Query parameters.
            pool_color: Optional color for the vesicle pool.
            distances2: Properties from the second distances layer (optional).
        """
        # Check which of the properties are present and construct the combined properties based on this.
        if distances is None and morphology is None:  # No properties were given -> we can't do anything.
            show_info("ERROR: Neither distances nor vesicle morphology were found.")
            return
        elif distances is None and morphology is not None:  # Only morphology props were found.
            merged_df = pd.DataFrame(morphology).drop(columns=["index"])
        elif distances is not None and morphology is None:  # Only distances were found.
            merged_df = pd.DataFrame(distances).drop(columns=["index"])
        else:  # Both were found.
            distance_ids = distances.get("label", [])
            morphology_ids = morphology.get("label", [])

            # Ensure that IDs are identical.
            if set(distance_ids) != set(morphology_ids):
                show_info("ERROR: The IDs in distances and morphology are not identical.")
                return

            # Create a merged dataframe from the dataframes which are relevant for the criterion.
            distances = pd.DataFrame(distances).drop(columns=["index"])
            morphology = pd.DataFrame(morphology).drop(columns=["index"])
            merged_df = morphology.merge(distances, left_on="label", right_on="label", suffixes=("_morph", "_dist"))
        # Add distances2 if present.
        if distances2 is not None:
            distance_ids = distances2.get("label", [])
            if set(distance_ids) != set(merged_df.label):
                show_info("ERROR: The IDs in distances2 and morphology are not identical.")
                return
            distances2 = pd.DataFrame(distances2).drop(columns=["index"])
            merged_df = merged_df.merge(distances2, left_on="label", right_on="label", suffixes=("", "2"))
        # Assign the vesicles to the current pool by filtering the mergeddataframe based on the query.
        filtered_df = self._parse_query(query, merged_df)
        if len(filtered_df) == 0:
            show_info("No vesicles were found matching the condition.")
            return
        pool_vesicle_ids = filtered_df.label.values.tolist()
        vesicles_in_pool = len(pool_vesicle_ids)

        # Check if this layer was already created in a previous pool assignment.
        if pool_layer_name in self.viewer.layers:
            # If yes then load the previous pool assignments and merge them with the new pool assignments
            pool_layer = self.viewer.layers[pool_layer_name]
            pool_properties = pd.DataFrame.from_dict(pool_layer.properties)

            pool_names = pd.unique(pool_properties.pool)
            if pool_name in pool_names:
                show_info(f"Updating pool '{pool_name}' with {vesicles_in_pool} vesicles.")
                # This pool has already been assigned and we changed the criterion.
                # Its old assignment has to be over-written, remove the rows for this pool.
                pool_properties = pool_properties[pool_properties.pool != pool_name]
            else:
                show_info(f"Creating pool '{pool_name}' with {vesicles_in_pool} vesicles.")

            # Combine the vesicle ids corresponding to the previous assignment with the
            # assignment for the new / current pool.
            old_pool_ids = pool_properties.label.values.tolist()

            # Overwrite the intersection of the two pool assignments with the new pool.
            pool_intersections = np.intersect1d(pool_vesicle_ids, old_pool_ids)
            old_pool_ids = [item for item in old_pool_ids if item not in pool_intersections]
            pool_properties = pool_properties[~pool_properties['label'].isin(pool_intersections)]

            pool_assignments = sorted(pool_vesicle_ids + old_pool_ids)

            # Get a map for each vesicle id to its pool.
            id_to_pool_name = {ves_id: pool_name for ves_id in pool_vesicle_ids}
            id_to_pool_name.update({k: v for k, v in zip(old_pool_ids, pool_properties.pool.values)})

            # Get the pool values.
            # This is the list of pool names, corresponding to the selected ids in pool_assignments.
            pool_values = [id_to_pool_name[ves_id] for ves_id in pool_assignments]

        else:
            show_info(f"Creating pool '{pool_name}' with {vesicles_in_pool} vesicles.")
            # Otherwise, this is the first pool assignment.
            pool_assignments = pool_vesicle_ids
            pool_values = [pool_name] * len(pool_assignments)

        # Create the filtered segmentation.
        vesicle_pools = segmentation.copy()
        vesicle_pools[~np.isin(vesicle_pools, pool_assignments)] = 0

        # Create the pool properties.
        pool_properties = merged_df[merged_df.label.isin(pool_assignments)]
        # Remove columns that are not relevant for measurements.
        keep_columns = [
            col for col in pool_properties.columns
            if col not in ("x", "y", "z", "begin-x", "begin-y", "begin-z", "end-x", "end-y", "end-z")
        ]
        pool_properties = pool_properties[keep_columns]
        # Add a colun for the pool.
        pool_properties.insert(1, "pool", pool_values)

        # Update the colormap to display the pools.
        self._update_pool_colors(pool_name, pool_color)

        # Assign the vesicle ids to their pool color.
        vesicle_colors = {
            label_id: self.pool_colors[pname] for label_id, pname in zip(
                pool_properties.label.values, pool_properties.pool.values
            )
        }
        vesicle_colors[None] = "gray"

        # Add or replace the pool layer and properties.
        if pool_layer_name in self.viewer.layers:
            pool_layer = self.viewer.layers[pool_layer_name]
            pool_layer.data = vesicle_pools
            pool_layer.colormap = vesicle_colors
        else:
            pool_layer = self.viewer.add_labels(vesicle_pools, name=pool_layer_name, colormap=vesicle_colors)

        self._add_properties_and_table(pool_layer, pool_properties, save_path=self.save_path.text())
        pool_layer.refresh()

    def _parse_query(self, query: str, data: pd.DataFrame) -> pd.DataFrame:
        """Parse and apply a query string to filter data.

        Args:
            query: Comma-separated query string (e.g., "radius > 15, distance > 250").
            data: DataFrame containing the data to filter.

        Returns:
            Filtered DataFrame.
        """
        filters = query.split(",")  # Split the query into individual conditions
        filters = [f.strip() for f in filters]  # Remove extra spaces
        for condition in filters:
            try:
                # Apply each condition to filter the DataFrame
                data = data.query(condition)
            except Exception as e:
                print(f"Failed to apply condition '{condition}': {e}")
                continue
        return data

    def _create_settings_widget(self):
        setting_values = QWidget()
        setting_values.setLayout(QVBoxLayout())

        self.save_path, layout = self._add_path_param(name="Save Table", select_type="file", value="")
        setting_values.layout().addLayout(layout)

        self.pool_color_param, layout = self._add_string_param(name="Pool Color", value="")
        setting_values.layout().addLayout(layout)

        settings = self._make_collapsible(widget=setting_values, title="Advanced Settings")
        return settings
