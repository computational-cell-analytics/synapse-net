import os
import argparse
import pandas as pd


def aggregate_statistics(vesicle_table, morpho_table):
    boundary_dists = {"All": [], "RA-V": [], "MP-V": [], "Docked-V": []}
    pd_dists = {"All": [], "RA-V": [], "MP-V": [], "Docked-V": []}
    ribbon_dists = {"All": [], "RA-V": [], "MP-V": [], "Docked-V": []}
    radii = {"All": [], "RA-V": [], "MP-V": [], "Docked-V": []}

    tomo_names = []
    n_ravs, n_mpvs, n_dockeds = [], [], []
    n_vesicles, ves_per_surfs, ribbon_ids = [], [], []

    tomograms = pd.unique(vesicle_table.tomogram)
    for tomo in tomograms:
        tomo_table = vesicle_table[vesicle_table.tomogram == tomo]
        this_ribbon_ids = pd.unique(tomo_table.ribbon_id)

        for ribbon_id in this_ribbon_ids:

            ribbon_table = tomo_table[tomo_table.ribbon_id == ribbon_id]
            # FIXME we need the ribbon_id for the morpho table
            this_morpho_table = morpho_table[morpho_table.tomogram == tomo]

            rav_mask = ribbon_table.pool == "RA-V"
            mpv_mask = ribbon_table.pool == "MP-V"
            docked_mask = ribbon_table.pool == "Docked-V"

            masks = {"All": ribbon_table.pool != "", "RA-V": rav_mask, "MP-V": mpv_mask, "Docked-V": docked_mask}

            for pool, mask in masks.items():
                pool_table = ribbon_table[mask]
                radii[pool].append(pool_table["radius [nm]"].mean())
                ribbon_dists[pool].append(pool_table["ribbon_distance [nm]"].mean())
                pd_dists[pool].append(pool_table["pd_distance [nm]"].mean())
                boundary_dists[pool].append(pool_table["boundary_distance [nm]"].mean())

            tomo_names.append(tomo)
            ribbon_ids.append(ribbon_id)
            n_rav = rav_mask.sum()
            n_mpv = mpv_mask.sum()
            n_docked = docked_mask.sum()

            n_ves = n_rav + n_mpv + n_docked
            ribbon_surface = this_morpho_table[this_morpho_table.structure == "ribbon"]["surface [nm^2]"].values[0]
            ves_per_surface = n_ves / ribbon_surface

            n_ravs.append(n_rav)
            n_mpvs.append(n_mpv)
            n_dockeds.append(n_docked)
            n_vesicles.append(n_ves)
            ves_per_surfs.append(ves_per_surface)

    summary = {
        "tomogram": tomo_names,
        "ribbon_id": ribbon_ids,
        "N_RA-V": n_ravs,
        "N_MP-V": n_mpvs,
        "N_Docked-V": n_dockeds,
        "N_Vesicles": n_vesicles,
        "Vesicles / Surface [1 / nm^2]": ves_per_surfs,
    }
    summary.update({f"{pool}: radius [nm]": dists for pool, dists in radii.items()})
    summary.update({f"{pool}: ribbon_distance [nm]": dists for pool, dists in ribbon_dists.items()})
    summary.update({f"{pool}: pd_distance [nm]": dists for pool, dists in pd_dists.items()})
    summary.update({f"{pool}: boundary_distance [nm]": dists for pool, dists in boundary_dists.items()})
    summary = pd.DataFrame(summary)
    return summary


# TODO
# - add ribbon id to the morphology table!
def export_reformatted_results(input_path, output_path):
    vesicle_table = pd.read_excel(input_path, sheet_name="vesicles")
    morpho_table = pd.read_excel(input_path, sheet_name="morphology")

    vesicle_table["stimulation"] = vesicle_table["tomogram"].apply(lambda x: x.split("/")[0])
    # Separating by mouse is currently not required, but we leave in the column for now.
    vesicle_table["mouse"] = vesicle_table["tomogram"].apply(lambda x: x.split("/")[-3])
    vesicle_table["pil_v_mod"] = vesicle_table["tomogram"].apply(lambda x: x.split("/")[-2])

    # For now: export only the vesicle pools per tomogram.
    for stim in ("WT strong stim", "WT control"):
        for condition in ("pillar", "modiolar"):
            condition_table = vesicle_table[
                (vesicle_table.stimulation == stim) & (vesicle_table.pil_v_mod == condition)
            ]

            this_tomograms = pd.unique(condition_table.tomogram)
            this_morpho_table = morpho_table[morpho_table.tomogram.isin(this_tomograms)]
            condition_table = aggregate_statistics(condition_table, this_morpho_table)

            # Simpler aggregation for just the number of vesicles.
            # condition_table = condition_table.pivot_table(
            #     index=["tomogram", "ribbon_id"], columns="pool", aggfunc="size", fill_value=0
            # ).reset_index()

            sheet_name = f"{stim}-{condition}"

            if os.path.exists(output_path):
                with pd.ExcelWriter(output_path, engine="openpyxl", mode="a") as writer:
                    condition_table.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                condition_table.to_excel(output_path, sheet_name=sheet_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", "-i",
        default="results/20250221_1/automatic_analysis_results.xlsx"
    )
    parser.add_argument(
        "--output_path", "-o",
        default="results/vesicle_pools_automatic.xlsx"
    )
    args = parser.parse_args()
    export_reformatted_results(args.input_path, args.output_path)
