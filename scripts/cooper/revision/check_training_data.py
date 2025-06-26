import argparse
import os
from glob import glob

import napari
import h5py

ROOT = "/mnt/ceph-hdd/cold_store/projects/nim00007/new_AZ_train_data"
all_names = [
    "chemical_fixation",
    "tem",
    "stem",
    "stem_cropped",
    "endbulb_of_held",
    "endbulb_of_held_cropped",
]


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--names", nargs="+", default=all_names)
args = parser.parse_args()
names = args.names


for ds in names:
    paths = glob(os.path.join(ROOT, ds, "*.h5"))
    for p in paths:
        with h5py.File(p, "r") as f:
            raw = f["raw"][:]
            az = f["labels/az"][:]
            az_thin = f["labels/az_thin"][:]
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(az)
        v.add_labels(az_thin)
        v.title = os.path.basename(p)
        napari.run()
