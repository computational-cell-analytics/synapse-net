import os
from pathlib import Path

from tqdm import tqdm
from elf.io import open_file

from synapse_net.file_utils import get_data_path
from synapse_net.inference import segment_vesicles
from synapse_net.inference.postprocessing import close_holes
from parse_table import parse_table, get_data_root

VERSIONS = {
    1: {
        "model": "/scratch/projects/nim00007/data/synapse_net/models/moser/vesicles/mean-teacher-v3.zip",
        "mode": "distance-watershed",
    },
    2: {
        "model": "/scratch/projects/nim00007/data/synapse_net/models/moser/vesicles/mean-teacher-v5.zip",
        "mode": "distance-watershed",
        "closing_iterations": 4,
    }
}


def segment_folder(model_path, folder, version, is_new):
    if is_new:
        # This is the difference in scale between the new and old tomogram.
        scale = 1.47
    else:
        scale = None

    output_folder = os.path.join(folder, "automatisch", f"v{version}")
    os.makedirs(output_folder, exist_ok=True)

    data_path = get_data_path(folder)

    output_path = os.path.join(
        output_folder, Path(data_path).stem + "_vesicles.h5"
    )
    if os.path.exists(output_path):
        return

    print("Segmenting vesicles for", data_path)
    with open_file(data_path, "r") as f:
        data = f["data"][:]

    segmentation = segment_vesicles(
        data, model_path, verbose=False,
        mode=VERSIONS[version]["mode"],
        scale=scale,
    )
    closing_iterations = VERSIONS[version].get("closing_iterations", None)
    if closing_iterations is not None:
        segmentation = close_holes(segmentation, closing_iterations=closing_iterations)

    with open_file(output_path, "a") as f:
        f.create_dataset("segmentation", data=segmentation, compression="gzip")


def run_vesicle_segmentation(table, version, process_new_microscope):
    model_path = VERSIONS[version]["model"]

    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue

        micro = row["EM alt vs. Neu"]
        if micro == "beides":
            segment_folder(model_path, folder, version, is_new=False)
            if process_new_microscope:
                folder_new = os.path.join(folder, "Tomo neues EM")
                if not os.path.exists(folder_new):
                    folder_new = os.path.join(folder, "neues EM")
                assert os.path.exists(folder_new), folder_new
                segment_folder(model_path, folder_new, version, is_new=True)
        elif micro == "alt":
            segment_folder(model_path, folder, version, is_new=False)
        elif micro == "neu" and process_new_microscope:
            segment_folder(model_path, folder, version, is_new=True)


def main():
    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    version = 2
    process_new_microscope = True

    run_vesicle_segmentation(table, version, process_new_microscope)


if __name__ == "__main__":
    main()
