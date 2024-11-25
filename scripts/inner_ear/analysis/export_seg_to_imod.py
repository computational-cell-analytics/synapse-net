import os
from shutil import copyfile
from subprocess import run

import imageio.v3 as imageio
import mrcfile
import napari
import numpy as np
import pandas as pd
from elf.io import open_file
from skimage.transform import resize
from synaptic_reconstruction.imod.to_imod import write_segmentation_to_imod, write_segmentation_to_imod_as_points

out_folder = "./auto_seg_export"
os.makedirs(out_folder, exist_ok=True)


def _resize(seg, tomo_path):
    with open_file(tomo_path, "r") as f:
        shape = f["data"].shape

    if shape != seg.shape:
        seg = resize(seg, shape, order=0, anti_aliasing=False, preserve_range=True).astype(seg.dtype)
    assert seg.shape == shape
    return seg


def check_imod(tomo_path, mod_path):
    run(["imod", tomo_path, mod_path])


def export_pool(pool_name, pool_seg, tomo_path):
    seg_path = f"./auto_seg_export/{pool_name}.tif"
    pool_seg = _resize(pool_seg, tomo_path)
    imageio.imwrite(seg_path, pool_seg, compression="zlib")

    output_path = f"./auto_seg_export/{pool_name}.mod"
    write_segmentation_to_imod_as_points(tomo_path, seg_path, output_path, min_radius=5)

    check_imod(tomo_path, output_path)


def export_vesicles(folder, tomo_path):
    vesicle_pool_path = os.path.join(folder, "Korrektur", "vesicle_pools.tif")
    # pool_correction_path = os.path.join(folder, "Korrektur", "pool_correction.tif")
    # pool_correction = imageio.imread(pool_correction_path)

    assignment_path = os.path.join(folder, "Korrektur", "measurements.xlsx")
    assignments = pd.read_excel(assignment_path)

    vesicles = imageio.imread(vesicle_pool_path)

    pools = {}
    for pool_name in pd.unique(assignments.pool):
        pool_ids = assignments[assignments.pool == pool_name].id.values
        pool_seg = vesicles.copy()
        pool_seg[~np.isin(vesicles, pool_ids)] = 0
        pools[pool_name] = pool_seg

    view = False
    if view:
        v = napari.Viewer()
        v.add_labels(vesicles, visible=False)
        for pool_name, pool_seg in pools.items():
            v.add_labels(pool_seg, name=pool_name)
        napari.run()
    else:
        for pool_name, pool_seg in pools.items():
            export_pool(pool_name, pool_seg, tomo_path)


def export_structure(folder, tomo, name, view=False):
    path = os.path.join(folder, "Korrektur", f"{name}.tif")
    seg = imageio.imread(path)
    seg = _resize(seg, tomo)

    if view:
        with open_file(tomo, "r") as f:
            raw = f["data"][:]

        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(seg)
        napari.run()

        return

    seg_path = f"./auto_seg_export/{name}.tif"
    imageio.imwrite(seg_path, seg, compression="zlib")
    output_path = f"./auto_seg_export/{name}.mod"
    write_segmentation_to_imod(tomo, seg_path, output_path)
    check_imod(tomo, output_path)


def remove_scale(tomo):
    new_path = "./auto_seg_export/Emb71M1aGridA1sec1mod7.rec.rec"
    if os.path.exists(new_path):
        return new_path

    copyfile(tomo, new_path)

    with mrcfile.open(new_path, "r+") as f:
        # Set the origin to (0, 0, 0)
        f.header.nxstart = 0
        f.header.nystart = 0
        f.header.nzstart = 0
        f.header.origin = (0.0, 0.0, 0.0)

        # Save changes
        f.flush()

    return new_path


def main():
    folder = "/home/pape/Work/data/moser/em-synapses/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/modiolar/1"
    tomo = os.path.join(folder, "Emb71M1aGridA1sec1mod7.rec.rec")

    tomo = remove_scale(tomo)

    # export_vesicles(folder, tomo)
    # export_structure(folder, tomo, "ribbon", view=False)
    # export_structure(folder, tomo, "membrane", view=False)
    export_structure(folder, tomo, "PD", view=False)


if __name__ == "__main__":
    main()
