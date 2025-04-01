import argparse
import os
import sys
from datetime import datetime

import pandas as pd

from add_summary_stats_to_table import add_summary_stats
from combine_measurements import combine_automatic_results, combine_manual_results
from compare_pool_assignments import create_manual_assignment, compare_assignments, update_measurements
from export_reformatted_results import export_reformatted_results

sys.path.append("processing")


def get_output_folder():
    output_root = "./results"
    date = datetime.now().strftime("%Y%m%d")

    version = 1
    output_folder = os.path.join(output_root, f"{date}_{version}")
    while os.path.exists(output_folder):
        version += 1
        output_folder = os.path.join(output_root, f"{date}_{version}")

    os.makedirs(output_folder)
    return output_folder


def main():
    """Run the steps to extract distance distributions
    for both the automatic and manual synapse annotations.
    """
    from parse_table import parse_table, get_data_root

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_folder = get_output_folder()

    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    # TODO exclude the tomograms not part of analysis
    # See ground_truth/extract_automatic_structures.py for how to do this
    # Step 1: Combine the measurements for the automatic analysis.
    automatic_res_path = os.path.join(output_folder, "automatic_analysis_results.xlsx")
    combine_automatic_results(table, data_root, output_path=automatic_res_path)

    # Step 2: Combine the measurements for the manual analysis.
    manual_res_path = os.path.join(output_folder, "manual_analysis_results.xlsx")
    combine_manual_results(table, data_root, output_path=manual_res_path)

    # Step 3: Compare the distance based and fully manual pool assignment for the
    # manually annotated tomograms.
    manual_result_table = pd.read_excel(manual_res_path)
    tomograms = pd.unique(manual_result_table["tomogram"])

    create_manual_assignment(data_root, tomograms, force=args.force)
    compare_assignments(
        data_root, tomograms, manual_result_table,
        output_path=os.path.join(output_folder, "assignment_comparison.png")
    )
    full_manual_res_path = os.path.join(output_folder, "fully_manual_analysis_results.xlsx")
    update_measurements(data_root, tomograms, manual_res_path, output_path=full_manual_res_path)

    # Step 4: Add summary stats to all the tables.
    result_tables = [automatic_res_path, manual_res_path, full_manual_res_path]
    for tab in result_tables:
        add_summary_stats(tab)

    # Step 5: Export the reformatted results.
    for tab in result_tables:
        name = os.path.basename(tab)
        out_tab = os.path.join(output_folder, f"summary_{name}")
        export_reformatted_results(tab, out_tab)


if __name__ == "__main__":
    main()
