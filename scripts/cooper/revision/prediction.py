import os
from glob import glob
import argparse

import h5py
from synapse_net.inference.inference import get_model, compute_scale_from_voxel_size
from synapse_net.inference.compartments import segment_compartments
from synapse_net.inference.vesicles import segment_vesicles
from tqdm import tqdm

ROOT = "/mnt/ceph-hdd/cold_store/projects/nim00007/AZ_data/training_data"
OUTPUT_ROOT = "/mnt/ceph-hdd/cold_store/projects/nim00007/AZ_predictions"
RESOLUTIONS = {
    "01data_withoutInvertedFiles_minusSVseg_corrected": {"x": 1.554, "y": 1.554, "z": 1.554},
    "04_hoi_stem_examples_fidi_and_sarah_corrected": {"x": 0.8681, "y": 0.8681, "z": 0.8681},
    "04_hoi_stem_examples_fidi_and_sarah_corrected_rescaled_tomograms": {"x": 1.554, "y": 1.554, "z": 1.554},
    "04_hoi_stem_examples_minusSVseg_cropped_corrected": {"x": 0.8681, "y": 0.8681, "z": 0.8681},
    "04_hoi_stem_examples_minusSVseg_cropped_corrected_rescaled_tomograms": {"x": 1.554, "y": 1.554, "z": 1.554},
    "06_hoi_wt_stem750_fm_minusSVseg_cropped_corrected": {"x": 0.8681, "y": 0.8681, "z": 0.8681},
    "06_hoi_wt_stem750_fm_minusSVseg_cropped_corrected_rescaled_tomograms": {"x": 1.554, "y": 1.554, "z": 1.554},
    "12_chemical_fix_cryopreparation_minusSVseg_corrected": {"x": 1.554, "y": 1.554, "z": 1.554},
    "wichmann_withAZ": {"x": 1.748, "y": 1.748, "z": 1.748},
    "wichmann_withAZ_rescaled_tomograms": {"x": 1.554, "y": 1.554, "z": 1.554},
    "stem_cropped2_rescaled": {"x": 1.554, "y": 1.554, "z": 1.554},
}


def predict_boundaries(model, path, output_path, visualize=False):
    output_key = "predictions/boundaries"
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            if output_key in f:
                return

    dataset = os.path.basename(os.path.split(path)[0])

    with h5py.File(path, "r") as f:
        data = f["raw"][:]
    scale = compute_scale_from_voxel_size(RESOLUTIONS[dataset], "compartments")
    _, pred = segment_compartments(data, model=model, scale=scale, verbose=False, return_predictions=True)

    if visualize:
        import napari
        v = napari.Viewer()
        v.add_image(data)
        v.add_labels(pred)
        napari.run()

    with h5py.File(output_path, "a") as f:
        f.create_dataset(output_key, data=pred, compression="lzf")


def predict_all_boundaries(folder=ROOT, out_path=OUTPUT_ROOT, visualize=False):
    model = get_model("compartments")
    files = sorted(glob(os.path.join(folder, "**/*.h5"), recursive=True))
    for path in tqdm(files):
        folder_name = os.path.basename(os.path.split(path)[0])
        output_folder = os.path.join(out_path, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.basename(path))
        predict_boundaries(model, path, output_path, visualize)


def predict_vesicles(model, path, output_path, visualize=False):
    output_key = "predictions/vesicle_seg"
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            if output_key in f:
                return

    dataset = os.path.basename(os.path.split(path)[0])
    #if "rescaled" in dataset:
    #    return

    with h5py.File(path, "r") as f:
        data = f["raw"][:]
    scale = compute_scale_from_voxel_size(RESOLUTIONS[dataset], "vesicles_3d")
    seg = segment_vesicles(data, model=model, scale=scale, verbose=False)

    if visualize:
        import napari
        v = napari.Viewer()
        v.add_image(data)
        v.add_labels(seg)
        napari.run()

    with h5py.File(output_path, "a") as f:
        f.create_dataset(output_key, data=seg, compression="lzf")


def predict_all_vesicles(folder=ROOT, out_path=OUTPUT_ROOT, visualize=False):
    model = get_model("vesicles_3d")
    files = sorted(glob(os.path.join(folder, "**/*.h5"), recursive=True))
    for path in tqdm(files):
        folder_name = os.path.basename(os.path.split(path)[0])
        output_folder = os.path.join(out_path, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.basename(path))
        predict_vesicles(model, path, output_path, visualize)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i","--input_folder", type=str)
    parser.add_argument("-o","--out_path", type=str)
    parser.add_argument("--vesicles", action="store_true")
    parser.add_argument("--boundaries", action="store_true")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()
    if args.boundaries:
        predict_all_boundaries(args.input_folder, args.out_path, args.visualize)
    elif args.vesicles:
        predict_all_vesicles(args.input_folder, args.out_path, args.visualize)
    else:
        print("Choose which structure to predict: --vesicles or --boundaries")


if __name__ == "__main__":
    main()
