# import os
import sys

import numpy as np
import pandas as pd

sys.path.append("../processing")

from parse_table import get_data_root  # noqa


def get_finished_tomos():
    # data_root = get_data_root()
    # val_table = os.path.join(data_root, "Electron-Microscopy-Susi", "Validierungs-Tabelle-v3.xlsx")

    val_table = "/home/pape/Desktop/sfb1286/mboc_synapse/misc/Validierungs-Tabelle-v3-passt.xlsx"
    val_table = pd.read_excel(val_table)

    val_table = val_table[val_table["Kommentar 22.11.24"] == "passt"]
    n_tomos = len(val_table)
    assert n_tomos > 0

    tomo_names = []
    for _, row in val_table.iterrows():
        name = "/".join([
            row.Bedingung, f"Mouse {int(row.Maus)}",
            row["Ribbon-Orientierung"].lower().rstrip("?"),
            str(int(row["OwnCloud-Unterordner"]))]
        )
        tomo_names.append(name)

    return tomo_names


def get_manual_assignments():
    result_path = "../results/20241124_1/fully_manual_analysis_results.xlsx"
    results = pd.read_excel(result_path)
    return results


def get_proofread_assignments(tomograms):
    result_path = "../results/20241124_1/automatic_analysis_results.xlsx"
    results = pd.read_excel(result_path)
    results = results[results["tomogram"].isin(tomograms)]
    return results


def get_semi_automatic_assignments(tomograms):
    result_path = "../results/fully_automatic_analysis_results.xlsx"
    results = pd.read_excel(result_path)
    results = results[results["tomogram"].isin(tomograms)]
    return results


def get_measurements_with_annotation():
    manual_assignments = get_manual_assignments()

    # Get the tomos with manual annotations and the ones which are fully done in proofreading.
    manual_tomos = pd.unique(manual_assignments["tomogram"])
    finished_tomos = get_finished_tomos()
    # Intersect them to get the tomos we are using.
    tomos = np.intersect1d(manual_tomos, finished_tomos)

    manual_assignments = manual_assignments[manual_assignments["tomogram"].isin(tomos)]
    semi_automatic_assignments = get_semi_automatic_assignments(tomos)
    proofread_assignments = get_proofread_assignments(tomos)

    print("Tomograms with manual annotations:", len(tomos))
    return manual_assignments, semi_automatic_assignments, proofread_assignments


def get_all_measurements():
    tomos = get_finished_tomos()
    print("All tomograms:", len(tomos))

    semi_automatic_assignments = get_semi_automatic_assignments(tomos)
    proofread_assignments = get_proofread_assignments(tomos)

    return semi_automatic_assignments, proofread_assignments


def main():
    get_measurements_with_annotation()
    get_all_measurements()


if __name__ == "__main__":
    main()
