import os
import warnings

from glob import glob

import imageio.v3 as imageio
import mrcfile
import numpy as np
from bioimage_cpp.utils import take_dict
from elf.io import open_file
from skimage.transform import rescale
from tqdm import tqdm

from synapse_net.imod import export_segmentation, export_point_annotations
from synapse_net.file_utils import get_data_path

from parse_table import parse_table, get_data_root

# Files to skip because of issues in IMODMOP.
SKIP_FILES = []
# Non-point object
# data/moser/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/pillar/2/manuell


def process_labels(labels, label_names):
    if 4 in label_names and label_names[4] == " MP-V":
        label_names = {k - 2: v for k, v in label_names.items()}
        labels = {k: v - 2 for k, v in labels.items()}
    elif 4 in label_names and label_names[4] == " RA-V":
        label_names = {k - 3: v for k, v in label_names.items()}
        labels = {k: v - 3 for k, v in labels.items()}
    else:
        pass

    # Check correct mapping of the label names.
    assert "RA-V" in label_names[1], f"{label_names}"
    assert "MP-V" in label_names[2], f"{label_names}"
    if 3 in label_names:
        assert "Docked" in label_names[3], f"{label_names}"
    labels[0] = 0
    return labels, label_names


def export_vesicles(input_path, data_path, export_path):
    with open_file(data_path, "r") as f:
        shape = f["data"].shape
    with mrcfile.open(data_path, "r") as f:
        resolution = f.voxel_size.tolist()
    resolution = [res / 10 for res in resolution]

    try:
        segmentation, labels, label_names = export_point_annotations(
            input_path, shape, resolution=resolution[0],
        )
    except AssertionError:
        print("Contains non-point vesicle annotations:")
        print(input_path)
        return False

    labels, label_names = process_labels(labels, label_names)
    imageio.imwrite(export_path, segmentation, compression="zlib")

    segmentation = take_dict(labels, segmentation)
    export_pool_path = os.path.join(os.path.split(export_path)[0], "manual_pools.tif")
    imageio.imwrite(export_pool_path, segmentation, compression="zlib")

    return True


def process_folder(data_root, folder, have_pd, force):
    data_path = get_data_path(folder)
    annotation_folders = glob(os.path.join(folder, "manuell*"))
    assert len(annotation_folders) > 0, folder

    skip_files = [os.path.join(data_root, skip) for skip in SKIP_FILES]

    def process_annotations(file_, structure_name):
        if structure_name.lower() in file_.lower():
            export_path = os.path.join(os.path.split(file_)[0], f"{structure_name}.tif")

            if file_ in skip_files:
                print("Skipping", file_)
                return True

            if os.path.exists(export_path) and not force:
                return True

            print("Exporting", file_)
            if structure_name == "Vesikel":
                return export_vesicles(file_, data_path, export_path)

            export_segmentation(
                imod_path=file_, mrc_path=data_path, output_path=export_path,
            )
            return True
        else:
            return False

    structure_names = ("PD", "Ribbon", "Membrane", "Vesikel") if have_pd else\
        ("Ribbon", "Membrane", "Vesikel")

    for annotation_folder in annotation_folders:
        have_structures = {
            structure_name: False for structure_name in structure_names
        }
        annotation_files = glob(os.path.join(annotation_folder, "*.mod")) +\
            glob(os.path.join(annotation_folder, "*.3dmod"))

        for file_ in annotation_files:
            for structure_name in structure_names:
                is_structure = process_annotations(file_, structure_name)
                if is_structure:
                    have_structures[structure_name] = True

        for structure_name, have_structure in have_structures.items():
            # assert have_structure, f"{structure_name} is missing in {annotation_folder}"
            if not have_structure:
                print(f"{structure_name} is missing in {annotation_folder} !!!")
                # warnings.warn(f"{structure_name} is missing in {annotation_folder}")


def process_manual_segmentation(data_root, table, force):
    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue
        assert os.path.exists(folder), folder

        annotation = row["Manuelle Annotierung"].strip().lower()
        assert annotation in ("ja", "teilweise", "nein"), annotation
        have_manual = annotation in ("ja", "teilweise")
        have_pd = row["PD vorhanden? "] != "nein"

        if have_manual:
            process_folder(data_root, folder, have_pd, force)

    extra_folder = os.path.join(
        data_root,
        "Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/modiolar/1/Tomo neues EM"
    )
    process_folder(data_root, extra_folder, True, force)


def _rescale(data, scale, is_seg=True):
    if is_seg:
        data = rescale(
            data, scale, order=0, preserve_range=True, anti_aliasing=False
        ).astype(data.dtype)
    else:
        data = rescale(data, scale, preserve_range=True).astype(data.dtype)
    return data


def export_manual_segmentation_for_training(table, output_folder, root):
    os.makedirs(output_folder, exist_ok=True)

    def export_segmentation(folder, ii, is_new, name):
        pd_annotations = glob(os.path.join(folder, "**", "**PD**.tif"))
        ribbon_annotations = glob(os.path.join(folder, "**", "**ribbon**.tif")) +\
            glob(os.path.join(folder, "**", "**Ribbon**.tif"))
        membrane_annotations = glob(os.path.join(folder, "**", "**membrane**.tif")) +\
            glob(os.path.join(folder, "**", "**Membrane**.tif"))
        if any([
            len(pd_annotations) == 0,
            len(ribbon_annotations) == 0,
            len(membrane_annotations) == 0,
        ]):
            warnings.warn(f"Missing annotations for {folder}, skipping!")
            return ii

        output_file = os.path.join(output_folder, f"tomogram-{ii:03}.h5")
        if os.path.exists(output_file):
            return ii + 1

        print("Exporting segmentation to", output_file)
        data_path = get_data_path(folder)
        with open_file(data_path, "r") as f:
            tomo = f["data"][:]

        pd = np.zeros(tomo.shape, dtype="uint8")
        for pd_ann in pd_annotations:
            seg = imageio.imread(pd_ann)
            pd[seg > 0] = 1

        ribbon = np.zeros(tomo.shape, dtype="uint8")
        for ribbon_ann in ribbon_annotations:
            seg = imageio.imread(ribbon_ann)
            ribbon[seg > 0] = 1

        membrane = np.zeros(tomo.shape, dtype="uint8")
        for membrane_ann in membrane_annotations:
            seg = imageio.imread(membrane_ann)
            membrane[seg > 0] = 1

        if is_new:
            scale = 1.47
            tomo = _rescale(tomo, scale=scale, is_seg=False)
            pd = _rescale(pd, scale=scale)
            membrane = _rescale(membrane, scale=scale)
            ribbon = _rescale(ribbon, scale=scale)
            assert tomo.shape == pd.shape == ribbon.shape == membrane.shape, \
                f"{tomo.shape}, {pd.shape}, {membrane.shape}, {ribbon.shape}"

        with open_file(output_file, "a") as f:
            f.create_dataset("raw", data=tomo, compression="gzip")
            f.create_dataset("labels/presynapse", data=pd, compression="gzip")
            f.create_dataset("labels/membrane", data=membrane, compression="gzip")
            f.create_dataset("labels/ribbons", data=ribbon, compression="gzip")
            f.attrs["path"] = name

        return ii + 1

    i_export = 0
    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue
        assert os.path.exists(folder), folder

        annotation = row["Manuelle Annotierung"].strip().lower()
        assert annotation in ("ja", "teilweise", "nein"), annotation
        have_manual = annotation in ("ja", "teilweise")
        have_pd = row["PD vorhanden? "] == "ja"

        name = os.path.relpath(folder, root)

        is_new = row["EM alt vs. Neu"].lower() == "neu"
        if have_manual and have_pd:
            i_export = export_segmentation(folder, i_export, is_new, name)

    # export the extra annotation
    extra_folder = "/home/sophia/data/Electron-Microscopy-Susi/Analyse/WT strong stim/Mouse 1/modiolar/1/Tomo neues EM"
    name = os.path.relpath(extra_folder, root)
    export_segmentation(extra_folder, i_export, is_new=True, name=name)


def clear_manual_exports(data_root):
    for root, dirs, files in os.walk(data_root):
        if "manuell" in root:
            for ff in files:
                if ff.endswith(".tif"):
                    os.remove(os.path.join(root, ff))


def main():
    data_root = get_data_root()

    # clear_manual_exports(data_root)
    # return

    force = False
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)
    process_manual_segmentation(data_root, table, force=force)

    # output_folder = "/scratch-emmy/usr/nimcpape/data/moser/new-train-data"
    # export_manual_segmentation_for_training(table, output_folder, data_root)


if __name__ == "__main__":
    main()
