import os
import sys

from glob import glob

import mrcfile
import pandas as pd
from tqdm import tqdm

from synaptic_reconstruction.imod.export import load_points_from_imodinfo
from synaptic_reconstruction.file_utils import get_data_path

from common import get_finished_tomos

sys.path.append("../processing")


def aggregate_diameters(data_root, table, save_path, get_tab, include_names, sheet_name):
    radius_table = []
    for _, row in tqdm(table.iterrows(), total=len(table), desc="Collect tomo information"):
        folder = row["Local Path"]
        if folder == "":
            continue

        tomo_name = os.path.relpath(folder, os.path.join(data_root, "Electron-Microscopy-Susi/Analyse"))
        if (
            tomo_name in ("WT strong stim/Mouse 1/modiolar/1", "WT strong stim/Mouse 1/modiolar/2") and
            (row["EM alt vs. Neu"] == "neu")
        ):
            continue
        if tomo_name not in include_names:
            continue

        tab_path = get_tab(folder)
        if tab_path is None:
            continue

        tab = pd.read_excel(tab_path)
        this_tab = tab[["pool", "radius [nm]"]]
        this_tab.insert(0, "tomogram", [tomo_name] * len(this_tab))
        this_tab.insert(3, "diameter [nm]", this_tab["radius [nm]"] * 2)
        radius_table.append(this_tab)

    radius_table = pd.concat(radius_table)

    print("Saving table for", len(radius_table), "vesicles to", save_path, sheet_name)
    if os.path.exists(save_path):
        with pd.ExcelWriter(save_path, engine="openpyxl", mode="a") as writer:
            radius_table.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        radius_table.to_excel(save_path, sheet_name=sheet_name, index=False)


def aggregate_diameters_imod(data_root, table, save_path, include_names, sheet_name):
    radius_table = []
    for _, row in tqdm(table.iterrows(), total=len(table), desc="Collect tomo information"):
        folder = row["Local Path"]
        if folder == "":
            continue

        tomo_name = os.path.relpath(folder, os.path.join(data_root, "Electron-Microscopy-Susi/Analyse"))
        tomo_name = os.path.relpath(folder, os.path.join(data_root, "Electron-Microscopy-Susi/Analyse"))
        if (
            tomo_name in ("WT strong stim/Mouse 1/modiolar/1", "WT strong stim/Mouse 1/modiolar/2") and
            (row["EM alt vs. Neu"] == "neu")
        ):
            continue
        if tomo_name not in include_names:
            continue

        annotation_folder = os.path.join(folder, "manuell")
        if not os.path.exists(annotation_folder):
            annotation_folder = os.path.join(folder, "Manuell")
        if not os.path.exists(annotation_folder):
            continue

        annotations = glob(os.path.join(annotation_folder, "*.mod"))
        annotation_file = [ann for ann in annotations if ("vesikel" in ann.lower()) or ("vesicle" in ann.lower())]
        if len(annotation_file) != 1:
            continue
        annotation_file = annotation_file[0]

        tomo_file = get_data_path(folder)
        with mrcfile.open(tomo_file) as f:
            shape = f.data.shape
            resolution = list(f.voxel_size.item())
            resolution = [res / 10 for res in resolution][0]

        try:
            _, radii, labels, label_names = load_points_from_imodinfo(annotation_file, shape, resolution=resolution)
        except AssertionError:
            continue

        this_tab = pd.DataFrame({
            "tomogram": [tomo_name] * len(radii),
            "pool": [label_names[label_id] for label_id in labels],
            "radius [nm]": radii,
            "diameter [nm]": 2 * radii,
        })
        radius_table.append(this_tab)

    radius_table = pd.concat(radius_table)
    print("Saving table for", len(radius_table), "vesicles to", save_path, sheet_name)
    radius_table.to_excel(save_path, index=False, sheet_name=sheet_name)

    man_tomos = pd.unique(radius_table.tomogram)
    return man_tomos


def get_tab_semi_automatic(folder):
    tab_name = "measurements_uncorrected_assignments.xlsx"
    res_path = os.path.join(folder, "korrektur", tab_name)
    if not os.path.exists(res_path):
        res_path = os.path.join(folder, "Korrektur", tab_name)
    if not os.path.exists(res_path):
        res_path = None
    return res_path


def get_tab_proofread(folder):
    tab_name = "measurements.xlsx"
    res_path = os.path.join(folder, "korrektur", tab_name)
    if not os.path.exists(res_path):
        res_path = os.path.join(folder, "Korrektur", tab_name)
    if not os.path.exists(res_path):
        res_path = None
    return res_path


def get_tab_manual(folder):
    tab_name = "measurements.xlsx"
    res_path = os.path.join(folder, "manuell", tab_name)
    if not os.path.exists(res_path):
        res_path = os.path.join(folder, "Manuell", tab_name)
    if not os.path.exists(res_path):
        res_path = None
    return res_path


def main():
    from parse_table import parse_table, get_data_root

    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    all_tomos = get_finished_tomos()

    print("All tomograms")
    save_path = "./results/vesicle_diameters_all_tomos.xlsx"
    aggregate_diameters(
        data_root, table, save_path=save_path, get_tab=get_tab_semi_automatic, include_names=all_tomos,
        sheet_name="Semi-automatic",
    )
    aggregate_diameters(
        data_root, table, save_path=save_path, get_tab=get_tab_proofread, include_names=all_tomos,
        sheet_name="Proofread",
    )

    print()
    print("Tomograms with manual annotations")
    # aggregate_diameters(data_root, table, save_path="./results/vesicle_radii_manual.xlsx", get_tab=get_tab_manual)
    save_path = "./results/vesicle_diameters_tomos_with_manual_annotations.xlsx"
    man_tomos = aggregate_diameters_imod(
        data_root, table, save_path=save_path, include_names=all_tomos, sheet_name="Manual",
    )
    aggregate_diameters(
        data_root, table, save_path=save_path, get_tab=get_tab_semi_automatic, include_names=man_tomos,
        sheet_name="Semi-automatic",
    )
    aggregate_diameters(
        data_root, table, save_path=save_path, get_tab=get_tab_proofread, include_names=man_tomos,
        sheet_name="Proofread",
    )


if __name__ == "__main__":
    main()
