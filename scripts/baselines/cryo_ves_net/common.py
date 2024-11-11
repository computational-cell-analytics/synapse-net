import os
import tempfile
from glob import glob
from pathlib import Path
from shutil import copyfile

import h5py
import mrcfile
import numpy as np

import cryovesnet


# additional parameters?
def _segment_vesicles(directory):
    pl = cryovesnet.Pipeline(directory, pattern="*.mrc")
    pl.setup_cryovesnet_dir(make_masks=False)

    pl.run_deep()
    pl.rescale()
    pl.label_vesicles(within_segmentation_region=False)
    pl.label_vesicles_adaptive(separating=True)
    pl.make_spheres()
    pl.repair_spheres()


def _prepare_input(path, output_folder, input_key, resolution, rel_folder=None):
    fname = Path(path).stem
    if rel_folder is None:
        sub_folder = os.path.join(output_folder, fname)
    else:
        sub_folder = os.path.join(output_folder, rel_folder, fname)

    os.makedirs(sub_folder, exist_ok=True)
    out_path = os.path.join(sub_folder, f"{fname}.mrc")

    if path.endswith(".h5"):
        assert resolution is not None
        with h5py.File(path, "r") as f:
            vol = f[input_key][:]
        mrcfile.new(out_path, data=vol)

    # Copy the mrc file.
    elif path.endswith(".mrc"):
        copyfile(path, out_path)

    # Update the resolution if it was given.
    if resolution is not None:
        with mrcfile.open(out_path, mode="r+") as f:
            f.voxel_size = resolution
            f.update_header_from_data()

    return out_path, sub_folder


def _get_output_path(fname, output_folder, rel_folder=None):
    if rel_folder is None:
        this_output_folder = output_folder
    else:
        this_output_folder = os.path.join(output_folder, rel_folder)
        os.makedirs(this_output_folder, exist_ok=True)

    out_path = os.path.join(this_output_folder, f"{fname}.h5")
    return out_path


def _process_output(tmp, tmp_file, out_path, output_key, mask_file=None, mask_key=None):
    fname = Path(tmp_file).stem
    seg_path = os.path.join(tmp, "cryovesnet", f"{fname}_convex_labels.mrc")
    with mrcfile.open(seg_path, "r") as f:
        seg = f.data[:]

    if mask_file is not None:
        with h5py.File(mask_file, "r") as f:
            mask = f[mask_key][:].astype("bool")
        # We need to make this copy, otherwise seg is assignment only.
        seg = np.asarray(seg).copy()
        seg[~mask] = 0

    with h5py.File(out_path, "a") as f:
        f.create_dataset(output_key, data=seg, compression="gzip")


def apply_cryo_vesnet(
    input_folder, output_folder, pattern, input_key,
    resolution=None, output_key="prediction/vesicles/cryovesnet",
    mask_folder=None, mask_key=None, nested=False,
):
    os.makedirs(output_folder, exist_ok=True)
    if nested:
        files = sorted(glob(os.path.join(input_folder, "**", pattern), recursive=True))
    else:
        files = sorted(glob(os.path.join(input_folder, pattern)))

    if mask_folder is None:
        mask_files = None
    else:
        assert mask_key is not None
        if nested:
            mask_files = sorted(glob(os.path.join(mask_folder, "**", pattern), recursive=True))
        else:
            mask_files = sorted(glob(os.path.join(mask_folder, pattern)))
        assert len(mask_files) == len(files)

    with tempfile.TemporaryDirectory() as tmp:

        for i, file in enumerate(files):
            fname = Path(file).stem

            # Get the resolution info for this file.
            if resolution is None:
                res = None
            else:
                res = resolution[fname] if isinstance(resolution, dict) else resolution

            # Get the current output path, skip processing if it already exists
            rel_folder = os.path.split(os.path.relpath(file, input_folder))[0] if nested else None
            out_path = _get_output_path(fname, output_folder, rel_folder=rel_folder)
            if os.path.exists(out_path):
                print("Skipping processing of the", file, "because the output at", out_path, "already exists")
                continue

            # Prepare the input files by copying them over or resaving them (if h5).
            tmp_file, sub_folder = _prepare_input(file, tmp, input_key, res, rel_folder=rel_folder)

            # Segment the vesicles in the file.
            _segment_vesicles(sub_folder)

            # Write the output file.
            mask_file = None if mask_files is None else mask_files[i]
            _process_output(
                sub_folder, tmp_file, out_path, output_key, mask_file=mask_file, mask_key=mask_key
            )
