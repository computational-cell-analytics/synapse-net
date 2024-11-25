import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import get_all_measurements, get_measurements_with_annotation


def _plot_all(distances):
    pools = pd.unique(distances["pool"])
    dist_cols = ["ribbon_distance [nm]", "pd_distance [nm]", "boundary_distance [nm]"]

    fig, axes = plt.subplots(3, 3)

    # multiple = "stack"
    multiple = "layer"

    structures = ["Ribbon", "PD", "Boundary"]
    for i, pool in enumerate(pools):
        pool_distances = distances[distances["pool"] == pool]
        for j, dist_col in enumerate(dist_cols):
            ax = axes[i, j]
            ax.set_title(f"{pool} to {structures[j]}")
            sns.histplot(
                data=pool_distances, x=dist_col, hue="approach", multiple=multiple, kde=False, ax=ax
            )
            ax.set_xlabel("distance [nm]")

    fig.tight_layout()
    plt.show()


# We only care about the following distances:
# - MP-V -> PD, AZ (Boundary)
# - Docked-V -> PD, AZ
# - RA-V -> Ribbon
def _plot_selected(distances, save_path=None):
    fig, axes = plt.subplots(2, 2)
    multiple = "layer"

    if save_path is not None and os.path.exists(save_path):
        os.remove(save_path)

    def _plot(pool_name, distance_col, structure_name, ax):

        this_distances = distances[distances["pool"] == pool_name][["tomogram", "approach", distance_col]]

        ax.set_title(f"{pool_name} to {structure_name}")
        sns.histplot(
            data=this_distances, x=distance_col, hue="approach", multiple=multiple, kde=False, ax=ax
        )
        ax.set_xlabel("distance [nm]")

        if save_path is not None:
            approaches = pd.unique(this_distances["approach"])
            tomo_names = pd.unique(this_distances["tomogram"])

            tomograms = []
            distance_values = {approach: [] for approach in approaches}

            for tomo in tomo_names:
                tomo_dists = this_distances[this_distances["tomogram"] == tomo]
                max_vesicles = 0
                for approach in approaches:
                    n_vesicles = len(tomo_dists[tomo_dists["approach"] == approach].values)
                    if n_vesicles > max_vesicles:
                        max_vesicles = n_vesicles

                for approach in approaches:
                    app_dists = tomo_dists[tomo_dists["approach"] == approach][distance_col].values.tolist()
                    app_dists = app_dists + [np.nan] * (max_vesicles - len(app_dists))
                    distance_values[approach].extend(app_dists)
                tomograms.extend([tomo] * max_vesicles)

            save_distances = {"tomograms": tomograms}
            save_distances.update(distance_values)
            save_distances = pd.DataFrame(save_distances)

            sheet_name = f"{pool_name}_{structure_name}"
            if os.path.exists(save_path):
                with pd.ExcelWriter(save_path, engine="openpyxl", mode="a") as writer:
                    save_distances.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                save_distances.to_excel(save_path, index=False, sheet_name=sheet_name)

    # NOTE: we over-ride a plot here, should not do this in the actual version
    _plot("MP-V", "pd_distance [nm]", "PD", axes[0, 0])
    _plot("MP-V", "boundary_distance [nm]", "AZ Membrane", axes[0, 1])
    _plot("Docked-V", "pd_distance [nm]", "PD", axes[1, 0])
    _plot("Docked-V", "boundary_distance [nm]", "AZ Membrane", axes[1, 0])
    _plot("RA-V", "ribbon_distance [nm]", "Ribbon", axes[1, 1])

    fig.tight_layout()
    plt.show()


def for_tomos_with_annotation(plot_all=True):
    manual_assignments, semi_automatic_assignments, proofread_assignments = get_measurements_with_annotation()

    manual_distances = manual_assignments[
        ["tomogram", "pool", "ribbon_distance [nm]", "pd_distance [nm]", "boundary_distance [nm]"]
    ]
    manual_distances["approach"] = ["manual"] * len(manual_distances)

    semi_automatic_distances = semi_automatic_assignments[
        ["tomogram", "pool", "ribbon_distance [nm]", "pd_distance [nm]", "boundary_distance [nm]"]
    ]
    semi_automatic_distances["approach"] = ["semi_automatic"] * len(semi_automatic_distances)

    proofread_distances = proofread_assignments[
        ["tomogram", "pool", "ribbon_distance [nm]", "pd_distance [nm]", "boundary_distance [nm]"]
    ]
    proofread_distances["approach"] = ["proofread"] * len(proofread_distances)

    distances = pd.concat([manual_distances, semi_automatic_distances, proofread_distances])
    if plot_all:
        distances.to_excel("./results/distances_tomos_with_manual_annotations.xlsx", index=False)
        _plot_all(distances)
    else:
        _plot_selected(distances, save_path="./results/selected_distances_tomos_with_manual_annotations.xlsx")


def for_all_tomos(plot_all=True):
    semi_automatic_assignments, proofread_assignments = get_all_measurements()

    semi_automatic_distances = semi_automatic_assignments[
        ["tomogram", "pool", "ribbon_distance [nm]", "pd_distance [nm]", "boundary_distance [nm]"]
    ]
    semi_automatic_distances["approach"] = ["semi_automatic"] * len(semi_automatic_distances)

    proofread_distances = proofread_assignments[
        ["tomogram", "pool", "ribbon_distance [nm]", "pd_distance [nm]", "boundary_distance [nm]"]
    ]
    proofread_distances["approach"] = ["proofread"] * len(proofread_distances)

    distances = pd.concat([semi_automatic_distances, proofread_distances])
    if plot_all:
        distances.to_excel("./results/distances_all_tomos.xlsx", index=False)
        _plot_all(distances)
    else:
        _plot_selected(distances, save_path="./results/selected_distances_all_tomos.xlsx")


def main():
    plot_all = False
    for_tomos_with_annotation(plot_all=plot_all)
    for_all_tomos(plot_all=plot_all)


if __name__ == "__main__":
    main()
