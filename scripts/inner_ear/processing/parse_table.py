import os
import warnings

import numpy as np
import pandas as pd


def get_data_root():
    root = os.path.join(os.path.expanduser("~"), "data")
    hostname = os.uname()[1]
    if hostname == "pc-kreshuk11":
        root = "/home/pape/Work/data/moser/em-synapses"
    elif hostname == "glogin9":
        root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/original_data_sophia"
    elif "ggpu" in hostname:
        root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/moser/original_data_sophia"
    return root


def parse_table(table_path, data_root, require_local_path=True):
    table = pd.read_excel(table_path)
    local_paths = []

    prefix = "Electron-Microscopy-Susi/Analyse"
    for i, row in table.iterrows():
        # Skip empty rows.
        if row.Bedingung == "":
            local_paths.append("")
            continue
        if isinstance(row.Bedingung, float) and np.isnan(row.Bedingung):
            local_paths.append("")
            continue
        if isinstance(row["OwnCloud-Unterordner"], float) and np.isnan(row["OwnCloud-Unterordner"]):
            local_paths.append("")
            continue

        try:
            path = os.path.join(
                data_root, prefix, row.Bedingung, f"Mouse {int(row.Maus)}",
                row["Ribbon-Orientierung"].lower().rstrip("?"), str(int(row["OwnCloud-Unterordner"]))
            )
        except ValueError as e:
            print(i, ":", row)
            raise e

        if not os.path.exists(path):
            if require_local_path:
                raise RuntimeError(f"Cannot find {path}")
            else:
                warnings.warn(f"Cannot find {path}")
        local_paths.append(path)

    assert len(local_paths) == len(table)
    table["Local Path"] = local_paths
    table = table.dropna(how="all")
    return table


def _match_correction_folder(folder):
    possible_names = ["Korrektur", "korrektur", "korektur", "Korektur"]
    for name in possible_names:
        correction_folder = os.path.join(folder, name)
        if os.path.exists(correction_folder):
            return correction_folder
    return folder


def _match_correction_file(correction_folder, seg_name):
    possible_names = [seg_name, seg_name.lower()]
    if seg_name == "vesicles":
        possible_names = ["vesicle_pools"] + possible_names
    for name in possible_names:
        correction_file = os.path.join(correction_folder, f"{name}.tif")
        if os.path.exists(correction_file):
            return correction_file
    return correction_file


def check_val_table(val_table, row):
    row_selection = (val_table.Bedingung == row.Bedingung) &\
        (val_table.Maus == row.Maus) &\
        (val_table["Ribbon-Orientierung"] == row["Ribbon-Orientierung"]) &\
        (val_table["OwnCloud-Unterordner"] == row["OwnCloud-Unterordner"])

    # We have different column names that mark the progress.
    # Latest: "Kommentar 16.09.24"
    # Previous: "Kommentar 08.09.24"
    # Fallback: "Fertig 3.0?"
    if "Kommentar 27-10-24" in val_table.columns:
        print("Using Kommentar 27.10.2024")
        complete_vals = val_table[row_selection]["Kommentar 27-10-24"].values
        is_complete = (
            (complete_vals == "passt") |
            (complete_vals == "skip")
        ).all()
    elif "Kommentar 08.10.2024" in val_table.columns:
        print("Using Kommentar 08.10.2024")
        complete_vals = val_table[row_selection]["Kommentar 08.10.2024"].values
        is_complete = (
            (complete_vals == "passt") |
            (complete_vals == "skip")
        ).all()
    elif "Fertig 3.0?" in val_table.columns:
        complete_vals = val_table[row_selection]["Fertig 3.0?"].values
        is_complete = (
            (complete_vals == "ja") |
            (complete_vals == "skip") |
            (complete_vals == "Anzeigefehler") |
            (complete_vals == "Ausschluss") |
            (complete_vals == "Keine PD")
        ).all()
    elif "Kommentar 16.09.24" in val_table.columns:
        complete_vals = val_table[row_selection]["Kommentar 16.09.24"].values
        is_complete = (
            (complete_vals == "passt") |
            (complete_vals == "Passt") |
            (complete_vals == "")
        ).all()
    elif "Kommentar 08.09.24" in val_table.columns:
        complete_vals = val_table[row_selection]["Kommentar 08.09.24"].values
        is_complete = (
            (complete_vals == "passt") |
            (complete_vals == "Passt") |
            (complete_vals == "")
        ).all()
    else:
        raise ValueError
    return is_complete


def main():
    table_path = "./Übersicht.xlsx"
    data_root = "/scratch-emmy/usr/nimcpape/data/moser"
    parse_table(table_path, data_root)


if __name__ == "__main__":
    main()
