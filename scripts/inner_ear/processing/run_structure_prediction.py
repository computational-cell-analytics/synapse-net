import os
import warnings
from pathlib import Path

from tqdm import tqdm
from elf.io import open_file

from synapse_net.file_utils import get_data_path
from synapse_net.inference import segment_structures
from parse_table import parse_table, get_data_root

VERSIONS = {
    1: {
        "model": "/scratch/projects/nim00007/data/synapse_net/models/moser/structures/supervised-v4.zip",
    },
    2: {
        "model": "/scratch/projects/nim00007/data/synapse_net/models/moser/structures/supervised-v4.zip",
        # The new model doesn't actually work better. it predicts too much for PD and for Ribbon.
        # "model": "/scratch/projects/nim00007/data/synapse_net/models/moser/structures/supervised-v5.zip",
    },
}


def segment_folder(model_path, folder, version, is_new, force):
    if is_new:
        # This is the difference in scale between the new and old tomogram.
        scale = 1.47
    else:
        scale = None

    output_folder = os.path.join(folder, "automatisch", f"v{version}")
    os.makedirs(output_folder, exist_ok=True)

    data_path = get_data_path(folder)
    structure_names = ["ribbon", "PD", "membrane"]

    output_paths = {
        name: os.path.join(output_folder, Path(data_path).stem + f"_{name}.h5")
        for name in structure_names
    }
    if all(os.path.exists(path) for path in output_paths.values()) and not force:
        return

    print("Segmenting structures for", data_path)
    with open_file(data_path, "r") as f:
        data = f["data"][:]

    # Suppress warnings from bioimageio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = segment_structures(
            data, model_path, structure_names,
            threshold=0.5, verbose=False, scale=scale
        )

    assert list(prediction.keys()) == structure_names
    for name, pred in prediction.items():
        with open_file(output_paths[name], "a") as f:
            ds = f.require_dataset("prediction", shape=pred.shape, compression="gzip", dtype="uint8")
            ds[:] = pred.astype("uint8")


def run_structure_prediction(table, version, process_new_microscope, force=False):
    model_path = VERSIONS[version]["model"]

    for i, row in tqdm(table.iterrows(), total=len(table)):
        folder = row["Local Path"]
        if folder == "":
            continue
        micro = row["EM alt vs. Neu"]
        if micro == "beides":
            segment_folder(
                model_path, folder, version, is_new=False, force=force
            )
            if process_new_microscope:
                folder_new = os.path.join(folder, "Tomo neues EM")
                if not os.path.exists(folder_new):
                    folder_new = os.path.join(folder, "neues EM")
                assert os.path.exists(folder_new), folder_new
                segment_folder(
                    model_path, folder_new, version, is_new=True, force=force
                )

        elif micro == "alt":
            segment_folder(
                model_path, folder, version, is_new=False, force=force
            )

        elif micro == "neu" and process_new_microscope:
            segment_folder(
                model_path, folder, version, is_new=True, force=force
            )


def main():
    data_root = get_data_root()
    table_path = os.path.join(data_root, "Electron-Microscopy-Susi", "Übersicht.xlsx")
    table = parse_table(table_path, data_root)

    version = 2
    process_new_microscope = True
    force = False

    run_structure_prediction(table, version, process_new_microscope, force=force)


if __name__ == "__main__":
    main()
