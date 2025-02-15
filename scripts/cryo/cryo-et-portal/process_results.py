import os
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import zarr

from synapse_net.file_utils import read_mrc
from tqdm import tqdm

from ome_zarr.writer import write_image
from ome_zarr.io import parse_url


IN_ROOT = "/scratch-grete/projects/nim00007/cryo-et/from_portal/for_eval"
OUT_ROOT = "/scratch-grete/projects/nim00007/cryo-et/from_portal/segmentations/DA_with_new_portalData_origDim"  # noqa

IN_ROOT = "/scratch-grete/projects/nim00007/cryo-et/from_portal/for_domain_adaptation"
OUT_ROOT = "/scratch-grete/projects/nim00007/cryo-et/from_portal/segmentations/DA_with_new_portalData_forDAdata"


def export_to_ome_zarr(export_file, seg, voxel_size):
    store = parse_url(export_file, mode="w").store
    root = zarr.group(store=store)

    scale = list(voxel_size.values())
    trafo = [
        [{"scale": scale, "type": "scale"}]
    ]
    write_image(seg, root, axes="zyx", coordinate_transformations=trafo, scaler=None)


def export_segmentation(export_folder, segmentation_file):
    fname = Path(segmentation_file).stem
    key = "/vesicles/segment_from_vesicle_DA_portal_v3"
    export_file = os.path.join(export_folder, f"{fname}.ome.zarr")

    if os.path.exists(export_file):
        return

    input_file = os.path.join(IN_ROOT, f"{fname}.mrc")
    raw, voxel_size = read_mrc(input_file)
    voxel_size = {k: v * 10 for k, v in voxel_size.items()}

    try:
        with h5py.File(segmentation_file, "r") as f:
            seg = f[key][:]
    except OSError as e:
        print(e)
        return

    seg = np.flip(seg, axis=1)
    assert seg.shape == raw.shape

    assert seg.max() < 128, f"{seg.max()}"
    seg = seg.astype("int8")
    export_to_ome_zarr(export_file, seg, voxel_size)


def main():
    export_folder = "./for_portal2"
    os.makedirs(export_folder, exist_ok=True)
    files = glob(os.path.join(OUT_ROOT, "*.h5"))
    for file in tqdm(files):
        export_segmentation(export_folder, file)


main()
