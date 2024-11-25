import argparse
import os
import multiprocessing as mp
import warnings

import imageio.v3 as imageio
import napari

from elf.io import open_file
from magicgui import magicgui
try:
    from napari_skimage_regionprops import add_table
except ImportError:
    add_table = None

from skimage.transform import rescale, resize

from ..distance_measurements import (
    create_object_distance_lines,
    measure_pairwise_object_distances,
    keep_direct_distances,
)

DISTANCE_MEASUREMENT_PATH = None
VIEW_SCALE = None


def _downsample(data, scale, is_seg=False, target_shape=None):
    if target_shape is not None:
        if data.shape == target_shape:
            return data

        if is_seg:
            data = resize(data, target_shape, order=0, anti_aliasing=False, preserve_range=True).astype(data.dtype)
        else:
            data = resize(data, target_shape, preserve_range=True).astype(data.dtype)
        return data

    if scale is None:
        return data
    rescale_factor = 1.0 / scale
    if is_seg:
        data = rescale(data, rescale_factor, order=0, anti_aliasing=False, preserve_range=True).astype(data.dtype)
    else:
        data = rescale(data, rescale_factor, preserve_range=True).astype(data.dtype)

    return data


@magicgui(call_button="Visualize Distances")
def measurement_widget(
    viewer: napari.Viewer,
    compute_direct_distances: bool = True,
    compute_neighbor_distances: int = 0,
) -> None:
    if add_table is None:
        raise Exception("Please install 'napari_skimage_regionprops' to use the measurement widget.")

    if compute_direct_distances:
        n_neighbors = None
        segmentation = viewer.layers["segmentation"].data
        # TODO: we may want to "max-project" the segmentation before doing it (i.e. take the union of the mask
        # for an object across all slices and project it to all of z). This would make the exclusion
        # criterion a bit more stringent and avoid passing slighly above / below other objects.
        pairs = keep_direct_distances(segmentation, DISTANCE_MEASUREMENT_PATH, scale=VIEW_SCALE)
    else:
        assert compute_neighbor_distances > 0
        n_neighbors = compute_neighbor_distances
        pairs = None

    lines, properties = create_object_distance_lines(
        DISTANCE_MEASUREMENT_PATH, n_neighbors=n_neighbors, scale=VIEW_SCALE, pairs=pairs,
    )
    if "line" in viewer.layers:  # TODO update the line layer
        pass
    else:  # create a new line layer
        line_layer = viewer.add_shapes(lines, shape_type="line", properties=properties, name="distances")
        add_table(line_layer, viewer)


def measure_distances(tomogram, segmentation, distance_measurement_path, view_scale=2, resolution=None):
    global VIEW_SCALE, DISTANCE_MEASUREMENT_PATH
    VIEW_SCALE = view_scale
    DISTANCE_MEASUREMENT_PATH = distance_measurement_path

    if not os.path.exists(distance_measurement_path):
        warnings.warn(
            "Could not find measurement result at {distance_measurement_path}."
            "Will compute the distance measurement result, this will take some time!"
        )
        cpu_count = mp.cpu_count()
        measure_pairwise_object_distances(
            segmentation, "boundary", n_threads=cpu_count, resolution=resolution, save_path=distance_measurement_path,
        )

    if view_scale > 1:
        tomogram = _downsample(tomogram, scale=view_scale)
        segmentation = _downsample(segmentation, scale=view_scale, is_seg=True)
    assert tomogram.shape == segmentation.shape, f"{tomogram.shape}, {segmentation.shape}"

    viewer = napari.Viewer()
    viewer.add_image(tomogram, visible=False)
    viewer.add_labels(segmentation)

    viewer.window.add_dock_widget(measurement_widget)

    napari.run()


def main():
    parser = argparse.ArgumentParser("Measure and visualize distances.")
    parser.add_argument("-i", "--image_path", help="Filepath for the tomogram", required=True)
    parser.add_argument("-s", "--segmentation_path", help="Filepath for the segmentation", required=True)
    parser.add_argument("-m", "--measurement_path", help="Filepath for the distance measurement result", required=True)
    parser.add_argument("--scale", type=int, default=2)

    args = parser.parse_args()
    with open_file(args.image_path, "r") as f:
        tomogram = f["data"][:]
    segmentation = imageio.imread(args.segmentation_path)

    measure_distances(
        tomogram, segmentation, args.measurement_path, view_scale=args.scale,
    )
