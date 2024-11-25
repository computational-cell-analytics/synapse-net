import os
import warnings
from pathlib import Path

import imageio.v3 as imageio
import mrcfile
import numpy as np
import pandas
import vigra

from synaptic_reconstruction.file_utils import get_data_path
from synaptic_reconstruction.distance_measurements import (
    measure_segmentation_to_object_distances,
    filter_blocked_segmentation_to_object_distances,
)

from synaptic_reconstruction.morphology import compute_radii, compute_object_morphology
from synaptic_reconstruction.inference.postprocessing import filter_border_vesicles
from skimage.transform import resize
from skimage.measure import regionprops, label

from elf.io import open_file
from tqdm import tqdm
from parse_table import (
    parse_table, get_data_root, _match_correction_folder, _match_correction_file, check_val_table
)


def _to_pool_name(correction_val):
    if correction_val == 1:
        pool_name = "RA-V"
    elif correction_val == 2:
        pool_name = "MP-V"
    elif correction_val == 3:
        pool_name = "Docked-V"
    else:
        raise ValueError
    return pool_name


def _load_segmentation(seg_path, tomo_shape):
    if seg_path.endswith(".tif"):
        seg = imageio.imread(seg_path)
    else:
        with open_file(seg_path, "r") as f:
            if "segmentation" not in f:
                return None
            seg = f["segmentation"][:]

    if tomo_shape is not None and seg.shape != tomo_shape:
        seg = resize(seg, tomo_shape, order=0, anti_aliasing=False, preserve_range=True).astype(seg.dtype)
    return seg


def compute_distances(segmentation_paths, save_folder, resolution, force, tomo_shape, use_corrected_vesicles=True):
    os.makedirs(save_folder, exist_ok=True)

    vesicles = None

    def _require_vesicles():
        vesicle_path = segmentation_paths["vesicles"]

        if vesicles is None:
            if use_corrected_vesicles:
                vesicle_pool_path = os.path.join(os.path.split(save_folder)[0], "vesicle_pools.tif")
                if os.path.exists(vesicle_pool_path):
                    vesicle_path = vesicle_pool_path
            return _load_segmentation(vesicle_path, tomo_shape)

        else:
            return vesicles

    # Compute the distance of the vesicles to the ribbon.
    ribbon_save = os.path.join(save_folder, "ribbon.npz")
    if force or not os.path.exists(ribbon_save):
        vesicles = _require_vesicles()
        ribbon_path = segmentation_paths["ribbon"]
        ribbon = _load_segmentation(ribbon_path, tomo_shape)

        if ribbon is None or ribbon.sum() == 0:
            print("The ribbon segmentation at", segmentation_paths["ribbon"], "is empty. Skipping analysis.")
            return None, True
        measure_segmentation_to_object_distances(vesicles, ribbon, save_path=ribbon_save, resolution=resolution)

    # Compute the distance of the vesicles to the PD.
    pd_save = os.path.join(save_folder, "PD.npz")
    if force or not os.path.exists(pd_save):
        vesicles = _require_vesicles()
        pd_path = segmentation_paths["PD"]
        pd = _load_segmentation(pd_path, tomo_shape)

        if pd.sum() == 0:
            print("The PD segmentation at", segmentation_paths["PD"], "is empty. Skipping analysis.")
            return None, True
        measure_segmentation_to_object_distances(vesicles, pd, save_path=pd_save, resolution=resolution)

    # Compute the distance of the vesicle to the membrane.
    membrane_save = os.path.join(save_folder, "membrane.npz")
    if force or not os.path.exists(membrane_save):
        vesicles = _require_vesicles()
        mem_path = segmentation_paths["membrane"]
        membrane = _load_segmentation(mem_path, tomo_shape)

        try:
            measure_segmentation_to_object_distances(
                vesicles, membrane, save_path=membrane_save, resolution=resolution
            )
        except AssertionError:
            return None, True

    distance_paths = {"ribbon": ribbon_save, "PD": pd_save, "membrane": membrane_save}
    return distance_paths, False


def _overwrite_pool_assignments(
    vesicles, vesicle_ids, pool_assignments, pool_correction_path
):

    correction = _load_segmentation(pool_correction_path, vesicles.shape)
    assert correction.shape == vesicles.shape

    def uniques(ves, corr):
        un = np.unique(corr[ves])
        return un

    # Map the correction volume to vesicle ids.
    props = regionprops(vesicles, correction, extra_properties=[uniques])
    for prop in props:
        vals = prop.uniques
        if 0 in vals:
            vals = vals[1:]

        if len(vals) == 1:
            label_id = prop.label
            correction_val = vals[0]
            pool_name = _to_pool_name(correction_val)

            pool_assignments[label_id] = pool_name
            if label_id not in vesicle_ids:
                vesicle_ids = np.concatenate([vesicle_ids, [label_id]])

        elif len(vals) == 2:
            warnings.warn("Multiple correction values found for a vesicle!")

    vesicle_ids = np.sort(vesicle_ids)

    return vesicle_ids, pool_assignments


def assign_vesicles_to_pools(
    vesicles, distance_paths, keep_unassigned=False, pool_correction_path=None, apply_extra_filters=True,
):

    def load_dist(measurement_path, seg_ids=None):
        auto_dists = np.load(measurement_path)
        distances, this_seg_ids = auto_dists["distances"], auto_dists["seg_ids"]
        object_ids = auto_dists.get("object_ids", np.zeros_like(seg_ids))
        if seg_ids is not None:
            assert np.array_equal(seg_ids, this_seg_ids)
        return distances, this_seg_ids, object_ids

    ribbon_distances, seg_ids, ribbon_ids = load_dist(distance_paths["ribbon"])
    pd_distances, _, pd_ids = load_dist(distance_paths["PD"], seg_ids=seg_ids)
    bd_distances, _, _ = load_dist(distance_paths["membrane"], seg_ids=seg_ids)
    assert len(seg_ids) == len(ribbon_distances) == len(pd_distances) == len(bd_distances)

    # Find the vesicles that are ribbon associated (RA-V).
    # Criterion: vesicles are closer than 80 nm to ribbon and they are in the first row
    # (i.e. not blocked by another vesicle).
    rav_ribbon_distance = 80  # nm
    rav_ids = seg_ids[ribbon_distances < rav_ribbon_distance]

    # Filter out the blocked vesicles.
    if apply_extra_filters:
        rav_ids = filter_blocked_segmentation_to_object_distances(
            vesicles, distance_paths["ribbon"], seg_ids=rav_ids, line_dilation=4, verbose=True,
        )
        rav_ids = filter_border_vesicles(vesicles, seg_ids=rav_ids)

    # Find the vesicles that are membrane proximal (MP-V).
    # Criterion: vesicles are closer than 50 nm to the membrane and closer than 100 nm to the PD.
    mpv_pd_distance = 100  # nm
    mpv_bd_distance = 50  # nm
    mpv_ids = seg_ids[np.logical_and(pd_distances < mpv_pd_distance, bd_distances < mpv_bd_distance)]
    if apply_extra_filters:
        mpv_ids = filter_border_vesicles(vesicles, seg_ids=mpv_ids)

    # Find the vesicles that are membrane docked (Docked-V).
    # Criterion: vesicles are closer than 2 nm to the membrane and closer than 100 nm to the PD.
    docked_pd_distance = 100  # nm
    docked_bd_distance = 2  # nm
    docked_ids = seg_ids[np.logical_and(pd_distances < docked_pd_distance, bd_distances < docked_bd_distance)]
    if apply_extra_filters:
        docked_ids = filter_border_vesicles(vesicles, seg_ids=docked_ids)

    # Keep only the vesicle ids that are in one of the three categories.
    vesicle_ids = np.unique(np.concatenate([rav_ids, mpv_ids, docked_ids]))

    # Create a dictionary to map vesicle ids to their corresponding pool.
    # (RA-V get's over-written by MP-V, which is correct).
    pool_assignments = {vid: "RA-V" for vid in rav_ids}
    pool_assignments.update({vid: "MP-V" for vid in mpv_ids})
    pool_assignments.update({vid: "Docked-V" for vid in docked_ids})
    if keep_unassigned:
        unassigned_vesicles = np.setdiff1d(seg_ids, vesicle_ids)
        pool_assignments.update({vid: "unassigned" for vid in unassigned_vesicles})
        vesicle_ids = seg_ids

    if pool_correction_path is not None:
        vesicle_ids, pool_assignments = _overwrite_pool_assignments(
            vesicles, vesicle_ids, pool_assignments, pool_correction_path,
        )

    id_mask = np.isin(seg_ids, vesicle_ids)
    assert id_mask.sum() == len(vesicle_ids)
    distances = {
        "ribbon": pandas.DataFrame({
            "id": vesicle_ids,
            "distance": ribbon_distances[id_mask],
            "ribbon_id": ribbon_ids[id_mask],
        }),
        "PD": pandas.DataFrame({
            "id": vesicle_ids,
            "distance": pd_distances[id_mask],
            "pd_id": pd_ids[id_mask],
        }),
        "membrane": pandas.DataFrame({
            "id": vesicle_ids,
            "distance": bd_distances[id_mask],
        })
    }

    return vesicle_ids, pool_assignments, distances


def to_excel(
    ves_assignments,
    ribbon_distances,
    pd_distances,
    boundary_distances,
    morphology_measurements,
    result_path
):
    # Merge all vesicle features into one table.
    table = ves_assignments.merge(ribbon_distances, on="id")
    table = table.rename(columns={"distance": "ribbon_distance [nm]"})

    table = table.merge(pd_distances, on="id")
    table = table.rename(columns={"distance": "pd_distance [nm]"})

    table = table.merge(boundary_distances, on="id")
    table = table.rename(columns={"distance": "boundary_distance [nm]"})

    table = table.sort_values(by="pool")

    table.to_excel(result_path, sheet_name="vesicles", index=False)
    with pandas.ExcelWriter(result_path, engine="openpyxl", mode="a") as writer:
        morphology_measurements.to_excel(writer, sheet_name="morphology", index=False)


def compute_morphology(ribbon, pd, resolution):
    # Compute ribbon and PD morphology.
    ribbon_morph = compute_object_morphology(ribbon, "ribbon", resolution=resolution)
    pd_morph = compute_object_morphology(pd, "presynaptic-density", resolution=resolution)

    measurements = pandas.concat([ribbon_morph, pd_morph])
    return measurements


def analyze_distances(
    segmentation_paths, distance_paths, resolution, result_path, tomo_shape,
    keep_unassigned=False, pool_correction_path=None, apply_extra_filters=True
):
    vesicles = _load_segmentation(segmentation_paths["vesicles"], tomo_shape)
    ribbon = _load_segmentation(segmentation_paths["ribbon"], tomo_shape)
    pd = _load_segmentation(segmentation_paths["PD"], tomo_shape)

    vesicle_ids, pool_assignments, distances = assign_vesicles_to_pools(
        vesicles, distance_paths, keep_unassigned=keep_unassigned,
        pool_correction_path=pool_correction_path, apply_extra_filters=apply_extra_filters,
    )
    vesicle_ids, vesicle_radii = compute_radii(vesicles, resolution, ids=vesicle_ids)
    morphology_measurements = compute_morphology(ribbon, pd, resolution=resolution)

    ves_assignments = pandas.DataFrame.from_dict(
        {
            "id": vesicle_ids,
            "pool": [pool_assignments[vid] for vid in vesicle_ids],
            "radius [nm]": [vesicle_radii[vid] for vid in vesicle_ids],
        }
    )

    to_excel(
        ves_assignments, distances["ribbon"], distances["PD"], distances["membrane"],
        morphology_measurements=morphology_measurements,
        result_path=result_path,
    )


def _relabel_vesicles(path):
    print("Relabel vesicles at", path)
    seg = _load_segmentation(path, None)
    seg = vigra.analysis.labelVolumeWithBackground(seg.astype("uint32"))
    seg, _, _ = vigra.analysis.relabelConsecutive(seg, start_label=1, keep_zeros=True)
    imageio.imwrite(path, seg, compression="zlib")


def _insert_missing_vesicles(vesicle_path, original_vesicle_path, pool_correction_path):
    print("Inserting missing vesicles due to correction at:", pool_correction_path)
    vesicles = imageio.imread(vesicle_path)
    original_vesicles = _load_segmentation(original_vesicle_path, vesicles.shape)
    correction = _load_segmentation(pool_correction_path, vesicles.shape)

    correction_labels = label(correction)
    correction_ids = np.unique(correction_labels)[1:]

    for corr_id in correction_ids:
        area = np.where(correction_labels == corr_id)
        vesicle_ids = np.unique(vesicles[area])
        is_missing = len(vesicle_ids) == 1 and vesicle_ids[0] == 0
        if is_missing:
            og_vesicle_id = np.unique(original_vesicles[area])
            print("Inserting", og_vesicle_id)
            if 0 in og_vesicle_id:
                og_vesicle_id = og_vesicle_id[1:]
            if len(og_vesicle_id) == 0:
                print("No vesicles found for", corr_id)
                continue
            assert len(og_vesicle_id) == 1

            new_vesicle_id = int(vesicles.max() + 1)
            vesicle_mask = original_vesicles == og_vesicle_id
            vesicles[vesicle_mask] = new_vesicle_id

    imageio.imwrite(vesicle_path, vesicles)


def analyze_folder(folder, version, n_ribbons, force, use_corrected_vesicles):
    data_path = get_data_path(folder)
    output_folder = os.path.join(folder, "automatisch", f"v{version}")

    fname = Path(data_path).stem

    segmentation_names = ["vesicles", "ribbon", "PD", "membrane"]
    segmentation_paths = {name: os.path.join(output_folder, f"{fname}_{name}.h5") for name in segmentation_names}

    if not all(os.path.exists(path) for path in segmentation_paths.values()):
        print("Not all required segmentations were found")
        return

    pool_correction_path = None
    correction_folder = _match_correction_folder(folder)
    if os.path.exists(correction_folder):
        output_folder = correction_folder

        if use_corrected_vesicles:
            result_path = os.path.join(output_folder, "measurements.xlsx")
        else:
            result_path = os.path.join(output_folder, "measurements_uncorrected_assignments.xlsx")

        if os.path.exists(result_path) and not force:
            return

        print("Analyse the corrected segmentations from", correction_folder)
        for seg_name in segmentation_names:
            if seg_name == "vesicles" and not use_corrected_vesicles:
                continue

            seg_path = _match_correction_file(correction_folder, seg_name)
            if os.path.exists(seg_path):

                if seg_name == "vesicles":
                    pool_correction_path = os.path.join(correction_folder, "pool_correction.tif")
                    if os.path.exists(pool_correction_path):
                        original_vesicle_path = segmentation_paths["vesicles"]
                        _insert_missing_vesicles(seg_path, original_vesicle_path, pool_correction_path)
                    else:
                        pool_correction_path = None

                segmentation_paths[seg_name] = seg_path

    if use_corrected_vesicles:
        result_path = os.path.join(output_folder, "measurements.xlsx")
    else:
        result_path = os.path.join(output_folder, "measurements_uncorrected_assignments.xlsx")
    if os.path.exists(result_path) and not force:
        return

    # Get the resolution (in Angstrom) and convert it to nanometer
    with mrcfile.open(data_path, "r") as f:
        resolution = f.voxel_size.tolist()
    resolution = [res / 10 for res in resolution]

    # Get the tomogram shape.
    with open_file(data_path, "r") as f:
        tomo_shape = f["data"].shape

    if use_corrected_vesicles:
        out_distance_folder = os.path.join(output_folder, "distances")
    else:
        out_distance_folder = os.path.join(output_folder, "distances_uncorrected")
    distance_paths, skip = compute_distances(
        segmentation_paths, out_distance_folder, resolution, force=force, tomo_shape=tomo_shape,
        use_corrected_vesicles=use_corrected_vesicles
    )
    if skip:
        return

    if force or not os.path.exists(result_path):

        if not use_corrected_vesicles:
            pool_correction_path = None

        analyze_distances(
            segmentation_paths, distance_paths, resolution, result_path, tomo_shape,
            pool_correction_path=pool_correction_path
        )


def run_analysis(table, version, force=False, val_table=None, use_corrected_vesicles=True):
    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        # We have to handle the segmentation without ribbon separately.
        if row["PD vorhanden? "] == "nein":
            continue

        n_pds = row["Anzahl PDs"]
        if n_pds == "unklar":
            n_pds = 1
            # continue
        n_pds = int(n_pds)
        n_ribbons = int(row["Anzahl Ribbons"])
        if (n_ribbons == 2 and n_pds == 1):
            print(f"The tomogram {folder} has {n_ribbons} ribbons and {n_pds} PDs.")
            print("The structure post-processing for this case is not yet implemented and will be skipped.")
            continue

        if val_table is not None:
            is_complete = check_val_table(val_table, row)
            if is_complete:
                continue

        micro = row["EM alt vs. Neu"]
        if micro == "beides":
            analyze_folder(folder, version, n_ribbons, force=force, use_corrected_vesicles=use_corrected_vesicles)

            folder_new = os.path.join(folder, "Tomo neues EM")
            if not os.path.exists(folder_new):
                folder_new = os.path.join(folder, "neues EM")
            assert os.path.exists(folder_new), folder_new
            analyze_folder(folder_new, version, n_ribbons, force=force, use_corrected_vesicles=use_corrected_vesicles)

        elif micro == "alt":
            analyze_folder(folder, version, n_ribbons, force=force, use_corrected_vesicles=use_corrected_vesicles)

        elif micro == "neu":
            analyze_folder(folder, version, n_ribbons, force=force, use_corrected_vesicles=use_corrected_vesicles)


def main():
    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    version = 2
    force = False
    use_corrected_vesicles = False

    # val_table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Validierungs-Tabelle-v3.xlsx")
    # val_table = pandas.read_excel(val_table_path)
    val_table = None

    run_analysis(
        table, version, force=force, val_table=val_table, use_corrected_vesicles=use_corrected_vesicles
    )


if __name__ == "__main__":
    main()
