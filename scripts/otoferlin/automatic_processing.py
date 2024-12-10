import os

import h5py
import numpy as np

from skimage.measure import label
from skimage.segmentation import relabel_sequential

from synapse_net.distance_measurements import measure_segmentation_to_object_distances
from synapse_net.file_utils import read_mrc
from synapse_net.inference.vesicles import segment_vesicles
from synapse_net.tools.util import get_model, compute_scale_from_voxel_size, _segment_ribbon_AZ
from tqdm import tqdm

from common import get_all_tomograms, get_seg_path, get_adapted_model

# These are tomograms for which the sophisticated membrane processing fails.
# In this case, we just select the largest boundary piece.
SIMPLE_MEM_POSTPROCESSING = [
    "Otof_TDAKO1blockA_GridN5_2_rec.mrc", "Otof_TDAKO2blockC_GridF5_1_rec.mrc", "Otof_TDAKO2blockC_GridF5_2_rec.mrc"
]


def _get_center_crop(input_):
    halo_xy = (600, 600)
    bb_xy = tuple(
        slice(max(sh // 2 - ha, 0), min(sh // 2 + ha, sh)) for sh, ha in zip(input_.shape[1:], halo_xy)
    )
    bb = (np.s_[:],) + bb_xy
    return bb, input_.shape


def _get_tiling():
    tile = {"x": 768, "y": 768, "z": 48}
    halo = {"x": 128, "y": 128, "z": 8}
    return {"tile": tile, "halo": halo}


def process_vesicles(mrc_path, output_path, process_center_crop):
    key = "segmentation/vesicles"
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            if key in f:
                return

    input_, voxel_size = read_mrc(mrc_path)
    if process_center_crop:
        bb, full_shape = _get_center_crop(input_)
        input_ = input_[bb]

    model = get_adapted_model()
    scale = compute_scale_from_voxel_size(voxel_size, "ribbon")
    print("Rescaling volume for vesicle segmentation with factor:", scale)
    tiling = _get_tiling()
    segmentation = segment_vesicles(input_, model=model, scale=scale, tiling=tiling)

    if process_center_crop:
        full_seg = np.zeros(full_shape, dtype=segmentation.dtype)
        full_seg[bb] = segmentation
        segmentation = full_seg

    with h5py.File(output_path, "a") as f:
        f.create_dataset(key, data=segmentation, compression="gzip")


def _simple_membrane_postprocessing(membrane_prediction):
    seg = label(membrane_prediction)
    ids, sizes = np.unique(seg, return_counts=True)
    ids, sizes = ids[1:], sizes[1:]
    return (seg == ids[np.argmax(sizes)]).astype("uint8")


def process_ribbon_structures(mrc_path, output_path, process_center_crop):
    key = "segmentation/ribbon"
    with h5py.File(output_path, "r") as f:
        if key in f:
            return
        vesicles = f["segmentation/vesicles"][:]

    input_, voxel_size = read_mrc(mrc_path)
    if process_center_crop:
        bb, full_shape = _get_center_crop(input_)
        input_, vesicles = input_[bb], vesicles[bb]
        assert input_.shape == vesicles.shape

    model_name = "ribbon"
    model = get_model(model_name)
    scale = compute_scale_from_voxel_size(voxel_size, model_name)
    tiling = _get_tiling()

    segmentations, predictions = _segment_ribbon_AZ(
        input_, model, tiling=tiling, scale=scale, verbose=True, extra_segmentation=vesicles,
        return_predictions=True, n_slices_exclude=5,
    )

    # The distance based post-processing for membranes fails for some tomograms.
    # In these cases, just choose the largest membrane piece.
    fname = os.path.basename(mrc_path)
    if fname in SIMPLE_MEM_POSTPROCESSING:
        segmentations["membrane"] = _simple_membrane_postprocessing(predictions["membrane"])

    if process_center_crop:
        for name, seg in segmentations.items():
            full_seg = np.zeros(full_shape, dtype=seg.dtype)
            full_seg[bb] = seg
            segmentations[name] = full_seg
        for name, pred in predictions.items():
            full_pred = np.zeros(full_shape, dtype=seg.dtype)
            full_pred[bb] = pred
            predictions[name] = full_pred

    with h5py.File(output_path, "a") as f:
        for name, seg in segmentations.items():
            f.create_dataset(f"segmentation/{name}", data=seg, compression="gzip")
            f.create_dataset(f"prediction/{name}", data=predictions[name], compression="gzip")


def postprocess_vesicles(mrc_path, output_path, process_center_crop):
    key = "segmentation/veiscles_postprocessed"
    with h5py.File(output_path, "r") as f:
        if key in f:
            return
        vesicles = f["segmentation/vesicles"][:]
        if process_center_crop:
            bb, full_shape = _get_center_crop(vesicles)
            vesicles = vesicles[bb]
        else:
            bb = np.s_[:]

        ribbon = f["segmentation/ribbon"][bb]
        membrane = f["segmentation/membrane"][bb]

    # Filter out small vesicle fragments.
    min_size = 5000
    ids, sizes = np.unique(vesicles, return_counts=True)
    ids, sizes = ids[1:], sizes[1:]
    filter_ids = ids[sizes < min_size]
    vesicles[np.isin(vesicles, filter_ids)] = 0

    input_, voxel_size = read_mrc(mrc_path)
    voxel_size = tuple(voxel_size[ax] for ax in "zyx")
    input_ = input_[bb]

    # Filter out all vesicles farther than 120 nm from the membrane or ribbon.
    max_dist = 120
    seg = (ribbon + membrane) > 0
    distances, _, _, seg_ids = measure_segmentation_to_object_distances(vesicles, seg, resolution=voxel_size)
    filter_ids = seg_ids[distances > max_dist]
    vesicles[np.isin(vesicles, filter_ids)] = 0

    vesicles, _, _ = relabel_sequential(vesicles)

    if process_center_crop:
        full_seg = np.zeros(full_shape, dtype=vesicles.dtype)
        full_seg[bb] = vesicles
        vesicles = full_seg
    with h5py.File(output_path, "a") as f:
        f.create_dataset(key, data=vesicles, compression="gzip")


def process_tomogram(mrc_path):
    output_path = get_seg_path(mrc_path)
    output_folder = os.path.split(output_path)[0]
    os.makedirs(output_folder, exist_ok=True)

    process_center_crop = True

    process_vesicles(mrc_path, output_path, process_center_crop)
    process_ribbon_structures(mrc_path, output_path, process_center_crop)
    postprocess_vesicles(mrc_path, output_path, process_center_crop)


def main():
    tomograms = get_all_tomograms()
    for tomogram in tqdm(tomograms, desc="Process tomograms"):
        process_tomogram(tomogram)

    # Update the membrane postprocessing for the tomograms where this went wrong.
    # for tomo in tqdm(tomograms, desc="Fix membrame postprocesing"):
    #     if os.path.basename(tomo) not in SIMPLE_MEM_POSTPROCESSING:
    #         continue
    #     seg_path = get_seg_path(tomo)
    #     with h5py.File(seg_path, "r") as f:
    #         pred = f["prediction/membrane"][:]
    #     seg = _simple_membrane_postprocessing(pred)
    #     with h5py.File(seg_path, "a") as f:
    #         f["segmentation/membrane"][:] = seg


if __name__:
    main()
