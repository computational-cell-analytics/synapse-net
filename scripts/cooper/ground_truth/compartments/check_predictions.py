import os
from glob import glob

import h5py
import napari

from synaptic_reconstruction.inference.compartments import _segment_compartments_3d


def check_pred(path, pred_path, name):
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        # seg = f["labels/compartments"][:]

    with h5py.File(pred_path, "r") as f:
        pred = f["prediction"][:]

    print("Run segmentation ...")
    seg_new = _segment_compartments_3d(pred)
    print("done")

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(pred, visible=False)
    # v.add_labels(seg, visible=False)
    v.add_labels(seg_new)
    v.title = name
    napari.run()


def main():
    seg_paths = sorted(glob("./predictions/segmentation/**/*.h5", recursive=True))

    for seg_path in seg_paths:
        ds_name, fname = os.path.split(seg_path)
        ds_name = os.path.split(ds_name)[1]

        # if ds_name in ("20241019_Tomo-eval_MF_Synapse", "20241019_Tomo-eval_PS_Synapse"):
        #     continue

        name = f"{ds_name}/{fname}"
        pred_path = os.path.join("./predictions/prediction", ds_name, fname)
        assert os.path.exists(pred_path), pred_path
        check_pred(seg_path, pred_path, name)


main()
