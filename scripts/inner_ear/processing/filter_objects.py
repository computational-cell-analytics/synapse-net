import os
from pathlib import Path
from tqdm import tqdm

import h5py
import imageio.v3 as imageio
import numpy as np
from skimage.measure import label
from skimage.segmentation import relabel_sequential

from synapse_net.file_utils import get_data_path
from parse_table import parse_table, get_data_root, _match_correction_folder, _match_correction_file


def _load_segmentation(seg_path):
    ext = Path(seg_path).suffix
    assert ext in (".h5", ".tif"), ext
    if ext == ".tif":
        seg = imageio.imread(seg_path)
    else:
        with h5py.File(seg_path, "r") as f:
            seg = f["segmentation"][:]
    return seg


def _save_segmentation(seg_path, seg):
    ext = Path(seg_path).suffix
    assert ext in (".h5", ".tif"), ext
    if ext == ".tif":
        imageio.imwrite(seg_path, seg, compression="zlib")
    else:
        with h5py.File(seg_path, "a") as f:
            f.create_dataset("segmentation", data=seg, compression="gzip")
    return seg


def _filter_n_objects(segmentation, num_objects):
    # Create individual objects for all disconnected pieces.
    segmentation = label(segmentation)
    # Find object ids and sizes, excluding background.
    ids, sizes = np.unique(segmentation, return_counts=True)
    ids, sizes = ids[1:], sizes[1:]
    # Only keep the biggest 'num_objects' objects.
    keep_ids = ids[np.argsort(sizes)[::-1]][:num_objects]
    segmentation[~np.isin(segmentation, keep_ids)] = 0
    # Relabel the segmentation sequentially.
    segmentation, _, _ = relabel_sequential(segmentation)
    # Ensure that we have the correct number of objects.
    n_ids = int(segmentation.max())
    assert n_ids == num_objects
    return segmentation


def process_tomogram(folder, num_ribbon, num_pd):
    data_path = get_data_path(folder)
    output_folder = os.path.join(folder, "automatisch", "v2")
    fname = Path(data_path).stem

    correction_folder = _match_correction_folder(folder)

    ribbon_path = _match_correction_file(correction_folder, "ribbon")
    if not os.path.exists(ribbon_path):
        ribbon_path = os.path.join(output_folder, f"{fname}_ribbon.h5")
    assert os.path.exists(ribbon_path), ribbon_path
    ribbon = _load_segmentation(ribbon_path)

    pd_path = _match_correction_file(correction_folder, "PD")
    if not os.path.exists(pd_path):
        pd_path = os.path.join(output_folder, f"{fname}_pd.h5")
    assert os.path.exists(pd_path), pd_path
    PD = _load_segmentation(pd_path)

    # Filter the ribbon and the PD.
    print("Filtering number of ribbons:", num_ribbon)
    ribbon = _filter_n_objects(ribbon, num_ribbon)
    bkp_path_ribbon = ribbon_path + ".bkp"
    os.rename(ribbon_path, bkp_path_ribbon)
    _save_segmentation(ribbon_path, ribbon)

    print("Filtering number of PDs:", num_pd)
    PD = _filter_n_objects(PD, num_pd)
    bkp_path_pd = pd_path + ".bkp"
    os.rename(pd_path, bkp_path_pd)
    _save_segmentation(pd_path, PD)


def filter_objects(table, version):
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

        n_pds = int(n_pds)
        n_ribbons = int(row["Anzahl Ribbons"])
        if (n_ribbons == 2 and n_pds == 1):
            print(f"The tomogram {folder} has {n_ribbons} ribbons and {n_pds} PDs.")
            print("The structure post-processing for this case is not yet implemented and will be skipped.")
            continue

        micro = row["EM alt vs. Neu"]
        if micro == "beides":
            process_tomogram(folder, n_ribbons, n_pds)

            folder_new = os.path.join(folder, "Tomo neues EM")
            if not os.path.exists(folder_new):
                folder_new = os.path.join(folder, "neues EM")
            assert os.path.exists(folder_new), folder_new
            process_tomogram(folder_new, n_ribbons, n_pds)

        elif micro == "alt":
            process_tomogram(folder, n_ribbons, n_pds)

        elif micro == "neu":
            process_tomogram(folder, n_ribbons, n_pds)


def main():
    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)
    version = 2
    filter_objects(table, version)


if __name__ == "__main__":
    main()
