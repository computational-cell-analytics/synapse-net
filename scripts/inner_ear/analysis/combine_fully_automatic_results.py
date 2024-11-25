import os
import sys

import pandas as pd

sys.path.append("..")
sys.path.append("../processing")


def combine_fully_auto_results(table, data_root, output_path):
    from combine_measurements import combine_results

    val_table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Validierungs-Tabelle-v3.xlsx")
    val_table = pd.read_excel(val_table_path)

    results = {}
    for _, row in table.iterrows():
        folder = row["Local Path"]
        if folder == "":
            continue

        row_selection = (val_table.Bedingung == row.Bedingung) &\
            (val_table.Maus == row.Maus) &\
            (val_table["Ribbon-Orientierung"] == row["Ribbon-Orientierung"]) &\
            (val_table["OwnCloud-Unterordner"] == row["OwnCloud-Unterordner"])
        complete_vals = val_table[row_selection]["Fertig!"].values
        is_complete = (complete_vals == "ja").all()
        if not is_complete:
            continue

        micro = row["EM alt vs. Neu"]

        tomo_name = os.path.relpath(folder, os.path.join(data_root, "Electron-Microscopy-Susi/Analyse"))
        tab_name = "measurements_uncorrected_assignments.xlsx"
        res_path = os.path.join(folder, "korrektur", tab_name)
        if not os.path.exists(res_path):
            res_path = os.path.join(folder, "Korrektur", tab_name)
        assert os.path.exists(res_path), res_path
        results[tomo_name] = (res_path, "alt" if micro == "beides" else micro)

        if micro == "beides":
            micro = "neu"

            new_root = os.path.join(folder, "neues EM")
            if not os.path.exists(new_root):
                new_root = os.path.join(folder, "Tomo neues EM")
            assert os.path.exists(new_root)

            res_path = os.path.join(new_root, "korrektur", "measurements.xlsx")
            if not os.path.exists(res_path):
                res_path = os.path.join(new_root, "Korrektur", "measurements.xlsx")
            assert os.path.exists(res_path), res_path
            results[tomo_name] = (res_path, "alt" if micro == "beides" else micro)

    combine_results(results, output_path, sheet_name="vesicles")


def main():
    from parse_table import parse_table, get_data_root

    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    res_path = "../results/fully_automatic_analysis_results.xlsx"
    combine_fully_auto_results(table, data_root, output_path=res_path)


main()
