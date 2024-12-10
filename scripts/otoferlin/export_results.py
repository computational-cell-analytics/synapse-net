import os
from datetime import datetime

import numpy as np
import pandas as pd
from common import get_all_tomograms, get_seg_path, to_condition

from synapse_net.distance_measurements import load_distances


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


def _export_results(tomograms, result_path, result_extraction):
    results = {}
    for tomo in tomograms:
        condition = to_condition(tomo)
        res = result_extraction(tomo)
        if condition in results:
            results[condition].append(res)
        else:
            results[condition] = [res]

    for condition, res in results.items():
        res = pd.concat(res)
        if os.path.exists(result_path):
            with pd.ExcelWriter(result_path, engine="openpyxl", mode="a") as writer:
                res.to_excel(writer, sheet_name=condition, index=False)
        else:
            res.to_excel(result_path, sheet_name=condition, index=False)


def export_vesicle_pools(tomograms, result_path):

    def result_extraction(tomo):
        folder = os.path.split(get_seg_path(tomo))[0]
        measure_path = os.path.join(folder, "vesicle_pools.csv")
        measures = pd.read_csv(measure_path).dropna()
        pool_names, counts = np.unique(measures.pool.values, return_counts=True)
        res = {"tomogram": [os.path.basename(tomo)]}
        res.update({k: v for k, v in zip(pool_names, counts)})
        res = pd.DataFrame(res)
        return res

    _export_results(tomograms, result_path, result_extraction)


def export_distances(tomograms, result_path):
    def result_extraction(tomo):
        folder = os.path.split(get_seg_path(tomo))[0]
        measure_path = os.path.join(folder, "vesicle_pools.csv")
        measures = pd.read_csv(measure_path).dropna()

        measures = measures[measures.pool.isin(["MP-V", "Docked-V"])][["vesicle_id", "pool"]]

        # Load the distances to PD.
        pd_distances, _, _, seg_ids = load_distances(os.path.join(folder, "distances", "PD.npz"))
        pd_distances = {sid: dist for sid, dist in zip(seg_ids, pd_distances)}
        measures["distance-to-pd"] = [pd_distances[vid] for vid in measures.vesicle_id.values]

        # Load the distances to membrane.
        mem_distances, _, _, seg_ids = load_distances(os.path.join(folder, "distances", "membrane.npz"))
        mem_distances = {sid: dist for sid, dist in zip(seg_ids, mem_distances)}
        measures["distance-to-membrane"] = [mem_distances[vid] for vid in measures.vesicle_id.values]

        measures = measures.drop(columns=["vesicle_id"])
        measures.insert(0, "tomogram", len(measures) * [os.path.basename(tomo)])

        return measures

    _export_results(tomograms, result_path, result_extraction)


def export_diameter(tomograms, result_path):
    def result_extraction(tomo):
        folder = os.path.split(get_seg_path(tomo))[0]
        measure_path = os.path.join(folder, "vesicle_pools.csv")
        measures = pd.read_csv(measure_path).dropna()

        measures = measures[measures.pool.isin(["MP-V", "Docked-V"])][["pool", "diameter"]]
        measures.insert(0, "tomogram", len(measures) * [os.path.basename(tomo)])

        return measures

    _export_results(tomograms, result_path, result_extraction)


def main():
    tomograms = get_all_tomograms()
    result_folder = get_output_folder()

    result_path = os.path.join(result_folder, "vesicle_pools.xlsx")
    export_vesicle_pools(tomograms, result_path)

    result_path = os.path.join(result_folder, "distances.xlsx")
    export_distances(tomograms, result_path)

    result_path = os.path.join(result_folder, "diameter.xlsx")
    export_diameter(tomograms, result_path)


if __name__ == "__main__":
    main()
