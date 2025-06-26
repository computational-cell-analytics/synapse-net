import argparse
import os
from glob import glob
from tqdm import tqdm

import h5py
import napari
from synapse_net.ground_truth.az_evaluation import thin_az

ROOT = "/mnt/ceph-hdd/cold_store/projects/nim00007/AZ_data/training_data"
OUTPUT_ROOT = "/mnt/ceph-hdd/cold_store/projects/nim00007/AZ_predictions"


def run_az_thinning(folder=ROOT, out_path=OUTPUT_ROOT):
    files = sorted(glob(os.path.join(folder, "**/*.h5"), recursive=True))
    for ff in tqdm(files):
        ds_name = os.path.basename(os.path.split(ff)[0])
        '''if not ds_name.startswith(("04", "06")):
            continue
        if "rescaled" in ds_name:
            continue'''

        print(f"ff {ff}")
        ff_out = os.path.join(out_path, os.path.relpath(ff, folder))
        print(f"ff_out {ff_out}")
        with h5py.File(ff_out, "r") as f_out, h5py.File(ff, "r") as f_in:
            # if "labels/az_thin2" in f_out:
            #     continue

            boundary_pred = f_out["predictions/boundaries"]
            vesicles = f_out["predictions/vesicle_seg"]

            tomo = f_in["raw"]
            az = f_in["labels/az"][:]

            az_thin = thin_az(
                az, boundary_map=boundary_pred, vesicles=vesicles, tomo=tomo, presyn_dist=8, check=True,
                min_thinning_size=2500,
            )

        with h5py.File(ff_out, "a") as f:
            ds = f.require_dataset("labels/az_thin", shape=az_thin.shape, dtype=az_thin.dtype, compression="gzip")
            ds[:] = az_thin


def check_az_thinning(folder=ROOT, out_path=OUTPUT_ROOT):
    files = sorted(glob(os.path.join(folder, "**/*.h5"), recursive=True))
    for ff in files:

        f_out = os.path.join(out_path, os.path.relpath(ff, folder))
        with h5py.File(f_out, "r") as f:
            if "labels/az_thin" not in f:
                continue
            az_thin = f["labels/az_thin"][:]

        with h5py.File(ff, "r") as f:
            tomo = f["raw"][:]

        v = napari.Viewer()
        v.add_image(tomo)
        v.add_labels(az_thin)
        napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_folder", type=str)
    parser.add_argument("-o","--out_path", type=str)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        check_az_thinning(args.input_folder, args.out_path)
    else:
        run_az_thinning(args.input_folder, args.out_path)


if __name__ == "__main__":
    main()
