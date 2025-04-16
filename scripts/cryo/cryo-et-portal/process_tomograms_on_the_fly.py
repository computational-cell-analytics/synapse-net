import argparse
import os
import subprocess

import cryoet_data_portal as cdp
import numpy as np
import zarr

from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from synapse_net.file_utils import read_data_from_cryo_et_portal_run
from synapse_net.inference.vesicles import segment_vesicles
from tqdm import tqdm

# OUTPUT_ROOT = ""
OUTPUT_ROOT = "/mnt/vast-nhr/projects/nim00007/data/synaptic-reconstruction/portal"


def get_tomograms(deposition_id, processing_type, number_of_tomograms=None):
    client = cdp.Client()
    tomograms = cdp.Tomogram.find(
        client, [cdp.Tomogram.deposition_id == deposition_id, cdp.Tomogram.processing == processing_type]
    )
    if number_of_tomograms is not None:
        tomograms = tomograms[:number_of_tomograms]
    return tomograms


def write_ome_zarr(output_file, segmentation, voxel_size, unit="nanometer"):
    store = parse_url(output_file, mode="w").store
    root = zarr.group(store=store)

    scale = list(voxel_size.values())
    trafo = [
        [{"scale": scale, "type": "scale"}]
    ]
    axes = [
        {"name": "z", "type": "space", "unit": unit},
        {"name": "y", "type": "space", "unit": unit},
        {"name": "x", "type": "space", "unit": unit},
    ]
    write_image(segmentation, root, axes=axes, coordinate_transformations=trafo, scaler=None)


def run_prediction(tomogram, deposition_id, processing_type):
    output_folder = os.path.join(OUTPUT_ROOT, f"upload_CZCDP-{deposition_id}", str(tomogram.run.dataset_id))
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
    scale = (1.0 / 2.7,) * 3
    segmentation = segment_vesicles(data, model_path=model_path, scale=scale)

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


def _get_task_tomograms(tomograms, slurm_tasks, task_id):
    # TODO we could also filter already done tomos.
    inputs_to_tasks = np.array_split(tomograms, slurm_tasks)
    assert len(inputs_to_tasks) == slurm_tasks
    return inputs_to_tasks[task_id]


def process_slurm(args, tomograms, deposition_id, processing_type):
    assert not args.check
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if task_id is None:  # We are not in the slurm task and submit the job.
        # Assemble the command for submitting a slurm array job.
        script_path = "process_tomograms_on_the_fly.sbatch"
        cmd = ["sbatch", "-a", f"0-{args.slurm_tasks-1}", script_path, "-s", str(args.slurm_tasks)]
        print("Submitting to slurm:")
        print(cmd)
        subprocess.run(cmd)
    else:  # We are in the task.
        task_id = int(task_id)
        this_tomograms = _get_task_tomograms(tomograms, args.slurm_tasks, task_id)
        for tomogram in tqdm(this_tomograms, desc="Run prediction for tomograms on-the-fly"):
            run_prediction(tomogram, deposition_id, processing_type)


def process_local(args, tomograms, deposition_id, processing_type):
    # Process each tomogram.
    print("Start processing", len(tomograms), "tomograms")
    for tomogram in tqdm(tomograms, desc="Run prediction for tomograms on-the-fly"):
        if args.check:
            check_result(tomogram, deposition_id, processing_type)
        else:
            run_prediction(tomogram, deposition_id, processing_type)


def main():
    parser = argparse.ArgumentParser()
    # Whether to check the result with napari instead of running the prediction.
    parser.add_argument("-c", "--check", action="store_true")
    parser.add_argument("-n", "--number_of_tomograms", type=int, default=None)
    parser.add_argument("-s", "--slurm_tasks", type=int, default=None)
    args = parser.parse_args()

    deposition_id = 10313
    processing_type = "denoised"

    # Get all the (processed) tomogram ids in the deposition.
    tomograms = get_tomograms(deposition_id, processing_type, args.number_of_tomograms)

    if args.slurm_tasks is None:
        process_local(args, tomograms, deposition_id, processing_type)
    else:
        process_slurm(args, tomograms, deposition_id, processing_type)


if __name__ == "__main__":
    main()
