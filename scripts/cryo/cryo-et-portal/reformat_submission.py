import os
from glob import glob
from pathlib import Path
from shutil import move

import cryoet_data_portal as cdp
from tqdm import tqdm


def main():
    input_folder = "segmentations"
    output_root = "upload_CZCDP-10330"

    client = cdp.Client()

    tomograms = sorted(glob(os.path.join("segmentations/*.ome.zarr")))
    for input_file in tqdm(tomograms, desc="Formatting submission"):
        tomo_id = Path(input_file).stem
        tomo_id = int(Path(tomo_id).stem)

        tomo = cdp.Tomogram.get_by_id(client, tomo_id)
        output_folder = os.path.join(output_root, str(tomo.run.dataset_id))
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{tomo.run.name}.zarr")
        move(input_file, output_file)


if __name__ == "__main__":
    main()
