import numpy as np
import pandas as pd


def main():
    man_path = "../results/20240917_1/fully_manual_analysis_results.xlsx"
    auto_path = "../results/20240917_1/automatic_analysis_results.xlsx"

    man_measurements = pd.read_excel(man_path, sheet_name="morphology")
    man_measurements = man_measurements[man_measurements.structure == "ribbon"][
        ["tomogram", "surface [nm^2]", "volume [nm^3]"]
    ]

    auto_measurements = pd.read_excel(auto_path, sheet_name="morphology")
    auto_measurements = auto_measurements[auto_measurements.structure == "ribbon"][
        ["tomogram", "surface [nm^2]", "volume [nm^3]"]
    ]

    # save all the automatic measurements
    auto_measurements.to_excel("./results/ribbon_morphology_auto.xlsx", index=False)

    man_tomograms = pd.unique(man_measurements["tomogram"])
    auto_tomograms = pd.unique(auto_measurements["tomogram"])
    tomos = np.intersect1d(man_tomograms, auto_tomograms)

    man_measurements = man_measurements[man_measurements.tomogram.isin(tomos)]
    auto_measurements = auto_measurements[auto_measurements.tomogram.isin(tomos)]

    save_path = "./results/ribbon_morphology_man-v-auto.xlsx"
    man_measurements.to_excel(save_path, sheet_name="manual", index=False)
    with pd.ExcelWriter(save_path, engine="openpyxl", mode="a") as writer:
        auto_measurements.to_excel(writer, sheet_name="auto", index=False)


if __name__ == "__main__":
    main()
