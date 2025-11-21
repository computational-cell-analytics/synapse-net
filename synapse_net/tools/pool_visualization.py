from pathlib import Path

import imageio.v3 as imageio
import napari
import numpy as np
import pandas as pd

from elf.parallel import isin
from ..file_utils import read_mrc


def _create_pools(vesicles, table, split_pools):
    label_ids, pool_colors = table.label.values, table.color.values

    pools = vesicles
    colormap = {label_id: color for label_id, color in zip(label_ids, pool_colors)}
    colormap[None] = [0, 0, 0, 0]

    if split_pools:
        unique_colors = np.unique(pool_colors)
        pool_ret = {}
        for this_color in unique_colors:
            this_pool = pools.copy()
            this_ids = [label_id for label_id, color in colormap.items() if color == this_color]
            pool_mask = np.zeros(this_pool.shape, dtype="bool")
            pool_mask = isin(this_pool, this_ids, out=pool_mask, block_shape=(32, 128, 128))
            this_pool[~pool_mask] = 0
            pool_ret[this_color] = this_pool
    else:
        pool_ret = {"pools": pools}

    return pool_ret, colormap


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


def _visualize_vesicle_pools(input_path, vesicle_paths, table_paths, segmentation_paths, split_pools):
    # Load the tomogram data, including scale information.
    data, voxel_size = read_mrc(input_path)
    axes = "zyx" if data.ndim == 3 else "yx"
    scale = tuple(float(voxel_size[ax]) for ax in axes)
    print("Loading data with scale", scale, "nanometer")

    # Load the vesicle layer, either from a single file with
    if len(vesicle_paths) == 1:
        vesicles = imageio.imread(vesicle_paths)
    else:
        vesicles = None
        for path in vesicle_paths:
            this_vesicles = imageio.imread(path)
            if vesicles is None:
                vesicles = this_vesicles.copy()
            else:
                ves_mask = this_vesicles != 0
                vesicles[ves_mask] = this_vesicles[ves_mask]

    # Load the tables with the pool assignments.
    # Create and add the pool layer.
    table = _parse_tables(table_paths)
    pools, colormap = _create_pools(vesicles, table, split_pools)

    viewer = napari.Viewer()
    viewer.add_image(data, scale=scale)
    viewer.add_labels(vesicles, scale=scale)
    for pool_name, pool in pools.items():
        viewer.add_labels(pool, scale=scale, name=pool_name, colormap=colormap)

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
