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

        self.image_selector_name = "Distances to Structure"
        self.image_selector_name1 = "Vesicles Segmentation"
        # Create the image selection dropdown.
        self.image_selector_widget = self._create_layer_selector(self.image_selector_name, layer_type="Shapes")
        self.segmentation1_selector_widget = self._create_layer_selector(self.image_selector_name1, layer_type="Labels")

        # Create new layer name.
        self.pool_layer_name_param, pool_layer_name_layout = self._add_string_param(
            name="Output Layer Name", value="",
        )

        # Create pool name.
        self.pool_name_param, pool_name_layout = self._add_string_param(
            name="Vesicle Pool", value="",
        )

        # Create query string
        self.query_param, query_layout = self._add_string_param(
            name="Criterion", value="",
            tooltip="Enter a comma separated query string (e.g., 'radius > 15, distance > 250') "
            "Possible filters: radius, distance, area, intensity_max, intensity_mean, intensity_min, intensity_std"
        )

        # Create advanced settings.
        self.settings = self._create_settings_widget()

        # Create and connect buttons.
        self.measure_button1 = QPushButton("Create Vesicle Pool")
        self.measure_button1.clicked.connect(self.on_pool_vesicles)

        # Add the widgets to the layout.
        layout.addWidget(self.image_selector_widget)
        layout.addWidget(self.segmentation1_selector_widget)
        layout.addLayout(query_layout)
        layout.addLayout(pool_layer_name_layout)
        layout.addLayout(pool_name_layout)
        layout.addWidget(self.measure_button1)

        self.setLayout(layout)

    def on_pool_vesicles(self):
        distances_layer = self._get_layer_selector_layer(self.image_selector_name)
        distances = distances_layer.properties
        segmentation = self._get_layer_selector_data(self.image_selector_name1)
        morphology_layer = self._get_layer_selector_layer(self.image_selector_name1)
        morphology = morphology_layer.properties

        if segmentation is None:
            show_info("INFO: Please choose a segmentation.")
            return
        if self.query_param.text() == "":
            show_info("INFO: Please enter a query string.")
            return
        query = self.query_param.text()

        if self.pool_layer_name_param.text() == "":
            show_info("INFO: Please enter a new layer name.")
            return
        pool_layer_name = self.pool_layer_name_param.text()
        if self.pool_name_param.text() == "":
            show_info("INFO: Please enter a pooled group name.")
            return
        pool_name = self.pool_name_param.text()

        if distances is None:
            show_info("INFO: Distances layer could not be found or has no values.")
            return

        self._compute_vesicle_pool(segmentation, distances, morphology, pool_layer_name, pool_name, query)

    def _compute_vesicle_pool(self, segmentation, distances, morphology, pool_layer_name, pool_name, query):
        """
        Compute a vesicle pool based on the provided query parameters.

        Args:
            segmentation (array): Segmentation data (e.g., labeled regions).
            distances (dict): Properties from the distances layer.
            morphology (dict): Properties from the morphology layer.
            pool_layer_name (str): Name for the new layer to be created.
            pool_name (str): Name for the pooled group to be assigned.
            query (dict): Query parameters.
        """
        distance_ids = distances.get("label", [])
        morphology_ids = morphology.get("label", [])

        # Ensure that IDs are identical.
        if set(distance_ids) != set(morphology_ids):
            show_info("ERROR: The IDs in distances and morphology are not identical.")
            return

        # Create a merged dataframe from the dataframes which are relevant for the criterion.
        # TODO: select the dataframes more dynamically depending on the criterion defined by the user.
        distances = pd.DataFrame(distances)
        morphology = pd.DataFrame(morphology)
        merged_df = morphology.merge(distances, left_on="label", right_on="label", suffixes=("_morph", "_dist"))

        # Assign the vesicles to the current pool by filtering the mergeddataframe based on the query.
        filtered_df = self._parse_query(query, merged_df)
        pool_vesicle_ids = filtered_df.label.values.tolist()

        # Check if this layer was already created in a previous pool assignment.
        if pool_layer_name in self.viewer.layers:
            # If yes then load the previous pool assignments and merge them with the new pool assignments
            pool_layer = self.viewer.layers[pool_layer_name]
            pool_properties = pd.DataFrame.from_dict(pool_layer.properties)

            pool_names = pd.unique(pool_properties.pool).tolist()
            if pool_name in pool_names:
                # This pool has already been assigned and we changed the criterion.
                # Its old assignment has to be over-written, remove the rows for this pool.
                pool_properties = pool_properties[pool_properties.pool != pool_name]

            # Combine the vesicle ids corresponding to the previous assignment with the
            # assignment for the new / current pool.
            old_pool_ids = pool_properties.label.values.tolist()
            pool_assignments = sorted(pool_vesicle_ids + old_pool_ids)

            # Get a map for each vesicle id to its pool.
            id_to_pool_name = {ves_id: pool_name for ves_id in pool_vesicle_ids}
            id_to_pool_name.update({k: v for k, v in zip(old_pool_ids, pool_properties.pool.values)})

            # Get the pool values.
            # This is the list of pool names, corresponding to the selected ids in pool_assignments.
            pool_values = [id_to_pool_name[ves_id] for ves_id in pool_assignments]

        else:
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
        pool_properties = pool_properties[keep_columns].reset_index()
        # Add a colun for the pool.
        pool_properties.insert(1, "pool", pool_values)

        # Create the colormap to group the pools in the layer rendering.
        # This can lead to color switches: if a new pool gets added which starts with
        # a letter that's earlier in the alphabet the color will switch.
        # To avoid this the user has to specify the pool color (not yet implemented, see next todo).
        pool_names = np.unique(pool_values).tolist()
        # TODO: add setting so that users can over-ride the color for a pool.
        pool_colors = {pname: COLORMAP[pool_names.index(pname)] for pname in pool_names}
        vesicle_colors = {
            label_id: pool_colors[pname] for label_id, pname
            in zip(pool_properties.label.values, pool_properties.pool.values)
        }
        vesicle_colors[None] = "gray"

        # TODO print some messages
        # Add or replace the pool layer and properties.
        if pool_layer_name in self.viewer.layers:
            # message about added or over-ridden pool, including number of vesicles in pool
            pool_layer = self.viewer.layers[pool_layer_name]
            pool_layer.data = vesicle_pools
            pool_layer.colormap = vesicle_colors
        else:
            # message about new pool, including number of vesicles in pool
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

        settings = self._make_collapsible(widget=setting_values, title="Advanced Settings")
        return settings
