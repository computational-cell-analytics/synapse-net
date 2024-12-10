from pathlib import Path

import napari

from synapse_net.file_utils import read_mrc
from common import get_all_tomograms, get_seg_path, load_segmentations, get_colormaps


def correct_structure_segmentation(mrc_path):
    seg_path = get_seg_path(mrc_path)

    data, _ = read_mrc(mrc_path)
    segmentations = load_segmentations(seg_path)
    color_maps = get_colormaps()

    v = napari.Viewer()
    v.add_image(data)
    for name, seg in segmentations.items():
        if name == "vesicles":
            name = "veiscles_postprocessed"
        v.add_labels(seg, name=name, colormap=color_maps.get(name, None))
    fname = Path(mrc_path).stem
    v.title = fname
    napari.run()


def main():
    tomograms = get_all_tomograms()
    for tomo in tomograms:
        correct_structure_segmentation(tomo)


if __name__ == "__main__":
    main()
