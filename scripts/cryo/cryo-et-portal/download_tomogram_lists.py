import json
import os

from synapse_net.file_utils import read_data_from_cryo_et_portal_run
from tqdm import tqdm


def download_tomogram_list(run_ids, output_root):
    print("Downloading", len(run_ids), "tomograms")
    os.makedirs(output_root, exist_ok=True)
    for run_id in tqdm(run_ids):
        output_path = os.path.join(output_root, f"{run_id}.mrc")
        data, voxel_size = read_data_from_cryo_et_portal_run(
            run_id, use_zarr_format=False, output_path=output_path, id_field="id",
        )
        if data is None:
            print("Did not find a tomogram for", run_id)


def download_tomograms_for_da():
    with open("./list_for_da.json") as f:
        run_ids = json.load(f)
    output_root = "/scratch-grete/projects/nim00007/cryo-et/from_portal/for_domain_adaptation"
    download_tomogram_list(run_ids, output_root)


def download_tomograms_for_eval():
    with open("./list_for_eval.json") as f:
        run_ids = json.load(f)
    output_root = "/scratch-grete/projects/nim00007/cryo-et/from_portal/for_eval"
    download_tomogram_list(run_ids, output_root)


def main():
    download_tomograms_for_eval()
    # download_tomograms_for_da()


if __name__ == "__main__":
    main()
