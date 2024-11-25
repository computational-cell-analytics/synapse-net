import argparse

import imageio.v3 as imageio
import napari
import numpy as np

from elf.io import open_file
from magicgui import magicgui


@magicgui(call_button="Get new id")
def _next_id_widget(viewer: napari.Viewer):
    seg_layer = viewer.layers["segmentation"]
    curr_seg = seg_layer.data
    next_id = int(curr_seg.max()) + 1
    print("Selecting next label:", next_id)
    seg_layer.selected_label = next_id
    seg_layer.mode = "paint"


@magicgui(call_button="Selected object done")
def _mark_done_widget(viewer: napari.Viewer):
    seg_layer = viewer.layers["segmentation"]
    done_layer = viewer.layers["done"]

    selected_id = seg_layer.selected_label
    print("Marking label", selected_id, "as done")

    this_mask = seg_layer.data == selected_id
    done_layer.data[this_mask] = 1
    done_layer.refresh()


def correct_segmentation(tomogram, segmentation, done_mask=None):
    v = napari.Viewer()

    v.add_image(tomogram)
    v.add_labels(segmentation)

    if done_mask is None:
        done_mask = np.zeros_like(segmentation)
    v.add_labels(done_mask, name="done")

    # widget for next id
    v.window.add_dock_widget(_next_id_widget)

    # widget for marking object as done
    v.window.add_dock_widget(_mark_done_widget)

    napari.run()


def main():
    parser = argparse.ArgumentParser("Manually correct segmentation with napari")
    parser.add_argument("-i", "--image_path", help="Filepath for the tomogram", required=True)
    parser.add_argument("-s", "--segmentation_path", help="Filepath for the segmentation", required=True)
    parser.add_argument("-d", "--done_path", help="Filepath for the done mask (optional)")
    args = parser.parse_args()

    with open_file(args.image_path, "r") as f:
        tomogram = f["data"][:]
    segmentation = imageio.imread(args.segmentation_path)

    done_mask = None if args.done_path is None else imageio.imread(args.done_path)

    correct_segmentation(tomogram, segmentation, done_mask)
