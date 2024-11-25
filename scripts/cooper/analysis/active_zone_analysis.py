import os
from glob import glob

import h5py
import numpy as np
import napari
import pandas as pd

from scipy.ndimage import binary_closing
from skimage.measure import label
from synaptic_reconstruction.ground_truth.shape_refinement import edge_filter
from synaptic_reconstruction.morphology import skeletonize_object
from synaptic_reconstruction.distance_measurements import measure_segmentation_to_object_distances
from tqdm import tqdm

from compute_skeleton_area import calculate_surface_area

ROOT = "./imig_data"  # noqa
OUTPUT_AZ = "./az_segmentation"

RESOLUTION = (1.554,) * 3


def filter_az(path):
    # Check if we have the output already.
    ds, fname = os.path.split(path)
    ds = os.path.basename(ds)
    out_path = os.path.join(OUTPUT_AZ, ds, fname)
    os.makedirs(os.path.join(OUTPUT_AZ, ds), exist_ok=True)

    if os.path.exists(out_path):
        return

    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        az = f["AZ/segment_from_AZmodel_v3"][:]
        vesicles = f["/vesicles/segment_from_combined_vesicles"][:]

    # Compute the sato filter of the raw data, smooth it afterwards.
    # This will highlight dark ridge-like structures, and so
    # will yield high values for the plasma membrane.
    hmap = edge_filter(raw, sigma=1.0, method="sato", per_slice=True, n_threads=8)

    # Filter the active zone by combining a bunch of things:
    # 1. Find a mask with high values in the ridge filter.
    threshold_hmap = 0.5
    az_filtered = hmap > threshold_hmap
    # 2. Intersect it with the active zone predictions.
    az_filtered = np.logical_and(az_filtered, az)
    # 3. Intersect it with the negative vesicle mask.
    az_filtered = np.logical_and(az_filtered, vesicles == 0)

    # Postprocessing of the filtered active zone:
    # 1. Apply connected components and only keep the largest component.
    az_filtered = label(az_filtered)
    ids, sizes = np.unique(az_filtered, return_counts=True)
    ids, sizes = ids[1:], sizes[1:]
    az_filtered = (az_filtered == ids[np.argmax(sizes)]).astype("uint8")
    # 2. Apply binary closing.
    az_filtered = np.logical_or(az_filtered, binary_closing(az_filtered, iterations=4)).astype("uint8")

    # Save the result.
    with h5py.File(out_path, "a") as f:
        f.create_dataset("filtered_az", data=az_filtered, compression="gzip")


def filter_all_azs():
    files = sorted(glob(os.path.join(ROOT, "**/*.h5"), recursive=True))
    for ff in tqdm(files, desc="Filter AZ segmentations."):
        filter_az(ff)


def process_az(path, view=True):
    key = "thin_az"

    with h5py.File(path, "r") as f:
        if key in f and not view:
            return
        az_seg = f["filtered_az"][:]

    az_thin = skeletonize_object(az_seg)

    if view:
        ds, fname = os.path.split(path)
        ds = os.path.basename(ds)
        raw_path = os.path.join(ROOT, ds, fname)
        with h5py.File(raw_path, "r") as f:
            raw = f["raw"][:]
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(az_seg)
        v.add_labels(az_thin)
        napari.run()
    else:
        with h5py.File(path, "a") as f:
            f.create_dataset(key, data=az_thin, compression="gzip")


# Apply thinning to all active zones to obtain 1d surface.
def process_all_azs():
    files = sorted(glob(os.path.join(OUTPUT_AZ, "**/*.h5"), recursive=True))
    for ff in tqdm(files, desc="Thin AZ segmentations."):
        process_az(ff, view=False)


def measure_az_area(path):
    from skimage import measure

    with h5py.File(path, "r") as f:
        seg = f["thin_az"][:]

    # Try via surface mesh.
    verts, faces, normals, values = measure.marching_cubes(seg, spacing=RESOLUTION)
    surface_area1 = measure.mesh_surface_area(verts, faces)

    # Try via custom function.
    surface_area2 = calculate_surface_area(seg, voxel_size=RESOLUTION)

    ds, fname = os.path.split(path)
    ds = os.path.basename(ds)

    return pd.DataFrame({
        "Dataset": [ds],
        "Tomogram": [fname],
        "surface_mesh [nm^2]": [surface_area1],
        "surface_custom [nm^2]": [surface_area2],
    })


# Measure the AZ surface areas.
def measure_all_areas():
    save_path = "./results/area_measurements.xlsx"
    if os.path.exists(save_path):
        return

    files = sorted(glob(os.path.join(OUTPUT_AZ, "**/*.h5"), recursive=True))
    area_table = []
    for ff in tqdm(files, desc="Measure AZ areas."):
        area = measure_az_area(ff)
        area_table.append(area)
    area_table = pd.concat(area_table)
    area_table.to_excel(save_path, index=False)

    manual_results = "/home/pape/Work/my_projects/synaptic-reconstruction/scripts/cooper/debug/surface/manualAZ_surface_area.xlsx"  # noqa
    manual_results = pd.read_excel(manual_results)[["Dataset", "Tomogram", "manual"]]
    comparison_table = pd.merge(area_table, manual_results, on=["Dataset", "Tomogram"], how="inner")
    comparison_table.to_excel("./results/area_comparison.xlsx", index=False)


def analyze_areas():
    import seaborn as sns
    import matplotlib.pyplot as plt

    table = pd.read_excel("./results/area_comparison.xlsx")

    fig, axes = plt.subplots(2)
    sns.scatterplot(data=table, x="manual", y="surface_mesh [nm^2]", ax=axes[0])
    sns.scatterplot(data=table, x="manual", y="surface_custom [nm^2]", ax=axes[1])
    plt.show()


def measure_distances(ves_path, az_path):
    with h5py.File(az_path, "r") as f:
        az = f["thin_az"][:]

    with h5py.File(ves_path, "r") as f:
        vesicles = f["vesicles/segment_from_combined_vesicles"][:]

    distances, _, _, _ = measure_segmentation_to_object_distances(vesicles, az, resolution=RESOLUTION)

    ds, fname = os.path.split(az_path)
    ds = os.path.basename(ds)

    return pd.DataFrame({
        "Dataset": [ds] * len(distances),
        "Tomogram": [fname] * len(distances),
        "Distance": distances,
    })


# Measure the AZ vesicle distances for all vesicles.
def measure_all_distances():
    save_path = "./results/vesicle_az_distances.xlsx"
    if os.path.exists(save_path):
        return

    ves_files = sorted(glob(os.path.join(ROOT, "**/*.h5"), recursive=True))
    az_files = sorted(glob(os.path.join(OUTPUT_AZ, "**/*.h5"), recursive=True))
    assert len(ves_files) == len(az_files)

    dist_table = []
    for ves_file, az_file in tqdm(zip(ves_files, az_files), total=len(az_files), desc="Measure distances."):
        dist = measure_distances(ves_file, az_file)
        dist_table.append(dist)
    dist_table = pd.concat(dist_table)

    dist_table.to_excel(save_path, index=False)


def reformat_distances():
    tab = pd.read_excel("./results/vesicle_az_distances.xlsx")

    munc_ko = {}
    munc_ctrl = {}

    snap_ko = {}
    snap_ctrl = {}

    for _, row in tab.iterrows():
        ds = row.Dataset
        tomo = row.Tomogram

        if ds == "Munc13DKO":
            if "CTRL" in tomo:
                group = munc_ctrl
            else:
                group = munc_ko
        else:
            assert ds == "SNAP25"
            if "CTRL" in tomo:
                group = snap_ctrl
            else:
                group = snap_ko

        name = os.path.splitext(tomo)[0]
        val = row["Distance [nm]"]
        if name in group:
            group[name].append(val)
        else:
            group[name] = [val]

    def save_tab(group, path):
        n_ves_max = max(len(v) for v in group.values())
        group = {k: v + [np.nan] * (n_ves_max - len(v)) for k, v in group.items()}
        group_tab = pd.DataFrame(group)
        group_tab.to_excel(path, index=False)

    os.makedirs("./results/distances_formatted", exist_ok=True)
    save_tab(munc_ko, "./results/distances_formatted/munc_ko.xlsx")
    save_tab(munc_ctrl, "./results/distances_formatted/munc_ctrl.xlsx")
    save_tab(snap_ko, "./results/distances_formatted/snap_ko.xlsx")
    save_tab(snap_ctrl, "./results/distances_formatted/snap_ctrl.xlsx")


def main():
    # filter_all_azs()
    # process_all_azs()
    # measure_all_areas()
    # analyze_areas()
    # measure_all_distances()
    reformat_distances()


if __name__ == "__main__":
    main()
