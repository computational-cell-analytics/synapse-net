from pathlib import Path

import imageio.v3 as imageio
import napari
import pandas as pd

from ..file_utils import read_mrc


def _create_pools(vesicles, table):
    label_ids, pool_colors = table.label.values, table.color.values

    pools = vesicles
    colormap = {label_id: color for label_id, color in zip(label_ids, pool_colors)}
    colormap[None] = [0, 0, 0, 0]

    return pools, colormap


def _parse_tables(table_paths):
    def load_table(path):
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".xlsx"):
            return pd.read_excel(path)
        else:
            raise RuntimeError("Unknown file ending.")

    if len(table_paths) == 1:
        table = load_table(table_paths[0])
    else:
        table = []
        for table_path in table_paths:
            this_table = load_table(table_path)
            pool_name = Path(table_path).stem
            this_table["pool"] = [pool_name] * len(this_table)
            table.append(this_table)
        table = pd.concat(table)
    return table


def _visualize_vesicle_pools(input_path, vesicle_path, table_paths, segmentation_paths):
    # Load the tomogram data, including scale information.
    data, voxel_size = read_mrc(input_path)
    axes = "zyx" if data.ndim == 3 else "yx"
    scale = tuple(float(voxel_size[ax]) for ax in axes)
    print("Loading data with scale", scale, "nanometer")

    # Load the vesicle layer.
    vesicles = imageio.imread(vesicle_path)

    # Load the tables with the pool assignments.
    # Create and add the pool layer.
    table = _parse_tables(table_paths)
    pools, colormap = _create_pools(vesicles, table)

    viewer = napari.Viewer()
    viewer.add_image(data, scale=scale)
    viewer.add_labels(vesicles, scale=scale)
    viewer.add_labels(pools, scale=scale, name="pools", colormap=colormap)

    # Add the additional segmentations.
    if segmentation_paths is not None:
        for seg_path in segmentation_paths:
            name = Path(seg_path).stem
            seg = imageio.imread(seg_path)
            viewer.add_labels(seg, name=name, scale=scale)

    # FIXME something is wrong here.
    # Add the scale bar.
    # @magicgui(call_button="Add Scale Bar")
    # def add_scale_bar(v: napari.Viewer):
    #     v.scale_bar.visible = True
    #     v.scale_bar.unit = "nm"
    # viewer.window.add_dock_widget(add_scale_bar)

    napari.run()
