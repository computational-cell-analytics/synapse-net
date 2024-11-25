import os
from glob import glob

import h5py
import napari
import numpy as np
import pandas as pd

from skimage.filters import gaussian

ROOT = "./04_full_reconstruction"
TABLE = "/home/pape/Desktop/sfb1286/mboc_synapse/draft_figures/full_reconstruction.xlsx"

# Skip datasets for which all figures were already done.
SKIP_DS = ["20241019_Tomo-eval_MF_Synapse", "20241019_Tomo-eval_PS_Synapse"]
# SKIP_DS = []


def _get_name_and_row(path, table):
    ds_name, name = os.path.split(path)
    ds_name = os.path.split(ds_name)[1]
    row = table[(table["dataset"] == ds_name) & (table["tomogram"] == name)]
    return ds_name, name, row


def _get_compartment_ids(row):
    compartment_ids = []
    for comp in ("Compartment 1", "Compartment 2", "Compartment 3", "Compartment 4"):
        comp_ids = row[comp].values[0]
        try:
            comp_ids = list(map(int, comp_ids.split(", ")))
        except AttributeError:
            pass

        if np.isnan(comp_ids).all():
            compartment_ids.append(None)
            continue

        if isinstance(comp_ids, int):
            comp_ids = [comp_ids]
        compartment_ids.append(comp_ids)

    return compartment_ids


def visualize_result(path, table):
    ds_name, name, row = _get_name_and_row(path, table)

    if ds_name in SKIP_DS:
        return

    if row["Use for Vis"].values[0] == "no":
        return
    compartment_ids = _get_compartment_ids(row)

    # access = np.s_[:]
    access = np.s_[::3, ::3, ::3]

    with h5py.File(path, "r") as f:
        raw = f["raw"][access]
        vesicles = f["labels/vesicles"][access]
        active_zone = f["labels/active_zone"][access]
        mitos = f["labels/mitochondria"][access]
        compartments = f["labels/compartments"][access]
    print("Loading done")

    raw = gaussian(raw)
    print("Gaussian done")

    if any(comp_ids is not None for comp_ids in compartment_ids):
        mask = np.zeros(raw.shape, dtype="bool")
        compartments_new = np.zeros_like(compartments)

        print("Filtering compartments:")
        for i, comp_ids in enumerate(compartment_ids, 1):
            if comp_ids is None:
                continue
            print(i, comp_ids)
            this_mask = np.isin(compartments, comp_ids)
            mask[this_mask] = 1
            compartments_new[this_mask] = i

        vesicles[~mask] = 0
        mitos[~mask] = 0
        compartments = compartments_new

    vesicle_ids = np.unique(vesicles)[1:]

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(mitos)
    v.add_labels(vesicles, colormap={ves_id: "orange" for ves_id in vesicle_ids})
    v.add_labels(compartments, colormap={1: "red", 2: "green", 3: "orange"})
    v.add_labels(active_zone, colormap={1: "blue"})
    v.title = f"{ds_name}/{name}"
    napari.run()


def visualize_only_compartment(path, table):
    ds_name, name, row = _get_name_and_row(path, table)
    compartment_ids = _get_compartment_ids(row)

    # Skip if we already have annotated the presynapse compartment(s)
    if any(comp_id is not None for comp_id in compartment_ids):
        print("Compartments already annotated for", ds_name, name)
        return

    # access = np.s_[:]
    access = np.s_[::2, ::2, ::2]

    with h5py.File(path, "r") as f:
        raw = f["raw"][access]
        compartments = f["labels/compartments"][access]

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(compartments)
    v.title = f"{ds_name}/{name}"
    napari.run()


def main():
    paths = sorted(glob(os.path.join(ROOT, "**/*.h5"), recursive=True))
    table = pd.read_excel(TABLE)
    for path in paths:
        print(path)
        visualize_result(path, table)
        # visualize_only_compartment(path, table)


if __name__ == "__main__":
    main()
