from common import get_all_tomograms, get_seg_path, load_segmentations
from tqdm import tqdm
from skimage.measure import label
import numpy as np


def ensure_labeled(vesicles):
    n_ids = len(np.unique(vesicles))
    n_ids_labeled = len(np.unique(label(vesicles)))
    assert n_ids == n_ids_labeled, f"{n_ids}, {n_ids_labeled}"


def main():
    tomograms = get_all_tomograms(restrict_to_good_tomos=True)
    for tomogram in tqdm(tomograms, desc="Process tomograms"):
        segmentations = load_segmentations(get_seg_path(tomogram))
        ensure_labeled(segmentations["vesicles"])


main()
