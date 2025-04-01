import argparse
import os
import cryoet_data_portal as cdp
import zarr

from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from synapse_net.file_utils import read_data_from_cryo_et_portal_run
from synapse_net.inference.vesicles import segment_vesicles
from tqdm import tqdm


def get_tomograms(deposition_id, processing_type):
    client = cdp.Client()
    tomograms = cdp.Tomogram.find(
        client, [cdp.Tomogram.deposition_id == deposition_id, cdp.Tomogram.processing == processing_type]
    )
    return tomograms


def write_ome_zarr(output_file, segmentation, voxel_size):
    store = parse_url(output_file, mode="w").store
    root = zarr.group(store=store)

    scale = list(voxel_size.values())
    trafo = [
        [{"scale": scale, "type": "scale"}]
    ]
    write_image(segmentation, root, axes="zyx", coordinate_transformations=trafo, scaler=None)


def run_prediction(tomogram, deposition_id, processing_type):
    output_folder = os.path.join(f"upload_CZCDP-{deposition_id}", str(tomogram.run.dataset_id))
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, f"{tomogram.run.name}.zarr")
    # We don't need to do anything if this file is already processed.
    if os.path.exists(output_file):
        return

    # Read tomogram data on the fly.
    data, voxel_size = read_data_from_cryo_et_portal_run(
        tomogram.run_id, processing_type=processing_type
    )

    # Segment vesicles.
    model_path = "/mnt/lustre-emmy-hdd/projects/nim00007/models/synaptic-reconstruction/vesicle-DA-portal-v3"
    segmentation = segment_vesicles(data, model_path=model_path)

    # Save the segmentation.
    write_ome_zarr(output_file, segmentation, voxel_size)


# TODO download on lower scale
def check_result(tomogram, deposition_id, processing_type):
    import napari

    # Read tomogram data on the fly.
    print("Download data ...")
    data, voxel_size = read_data_from_cryo_et_portal_run(
        tomogram.run_id, processing_type=processing_type
    )

    # Read the output file if it exists.
    output_folder = os.path.join(f"upload_CZCDP-{deposition_id}", str(tomogram.run.dataset_id))
    output_file = os.path.join(output_folder, f"{tomogram.run.name}.zarr")
    if os.path.exists(output_file):
        with zarr.open(output_file, "r") as f:
            segmentation = f["0"][:]
    else:
        segmentation = None

    v = napari.Viewer()
    v.add_image(data)
    if segmentation is not None:
        v.add_labels(segmentation)
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    # Whether to check the result with napari instead of running the prediction.
    parser.add_argument("-c", "--check", action="store_true")
    args = parser.parse_args()

    deposition_id = 10313
    processing_type = "denoised"

    # Get all the (processed) tomogram ids in the deposition.
    tomograms = get_tomograms(deposition_id, processing_type)

    # Process each tomogram.
    print("Start processing", len(tomograms), "tomograms")
    for tomogram in tqdm(tomograms, desc="Run prediction for tomograms on-the-fly"):
        if args.check:
            check_result(tomogram, deposition_id, processing_type)
        else:
            run_prediction(tomogram, deposition_id, processing_type)


# TODO segmented at wrong size, check voxel size!
if __name__ == "__main__":
    main()
