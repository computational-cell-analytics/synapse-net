import os
import numpy as np
import h5py

from skimage.measure import regionprops
from skimage.morphology import remove_small_holes
from skimage.segmentation import relabel_sequential

from synapse_net.inference.vesicles import segment_vesicles
from synapse_net.inference.compartments import segment_compartments
from synapse_net.inference.active_zone import segment_active_zone
from synapse_net.inference.inference import get_model_path
from synapse_net.ground_truth.az_evaluation import _get_presynaptic_mask


def fill_and_filter_vesicles(vesicles: np.ndarray) -> np.ndarray:
    """
    Apply a size filter and fill small holes in vesicle segments.

    Args:
        vesicles (np.ndarray): 3D volume with vesicle segment labels.

    Returns:
        np.ndarray: Processed vesicle segmentation volume.
    """
    ids, sizes = np.unique(vesicles, return_counts=True)
    ids, sizes = ids[1:], sizes[1:]  # remove background

    min_size = 2500
    vesicles_pp = vesicles.copy()
    filter_ids = ids[sizes < min_size]
    vesicles_pp[np.isin(vesicles, filter_ids)] = 0

    props = regionprops(vesicles_pp)
    for prop in props:
        bb = prop.bbox
        bb = np.s_[
            bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]
        ]
        mask = vesicles_pp[bb] == prop.label
        mask = remove_small_holes(mask, area_threshold=1000)
        vesicles_pp[bb][mask] = prop.label

    return vesicles_pp


def SV_pred(raw: np.ndarray, SV_model: str, output_path: str = None, store: bool = False) -> np.ndarray:
    """
    Run synaptic vesicle segmentation and optionally store the output.

    Args:
        raw (np.ndarray): Raw EM image volume.
        SV_model (str): Path to vesicle model.
        output_path (str): HDF5 file to store predictions.
        store (bool): Whether to store predictions.

    Returns:
        np.ndarray: Segmentation result.
    """
    pred_key = f"predictions/SV/pred"
    seg_key = f"predictions/SV/seg"

    use_existing_seg = False
    #checking if segmentation is already in output path and if so, use it
    if output_path and os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            if seg_key in f:
                seg = f[seg_key][:]
                use_existing_seg = True
                print(f"Using existing SV seg in {output_path}")

    if not use_existing_seg:
        #Excluding boundary SV, because they would also not be used in the manual annotation
        seg, pred = segment_vesicles(input_volume=raw, model_path=SV_model, exclude_boundary=True, verbose=False, return_predictions=True)

        if store and output_path:
            with h5py.File(output_path, "a") as f:
                if pred_key in f:
                    print(f"{pred_key} already saved")
                else:
                    f.create_dataset(pred_key, data=pred, compression="lzf")
                
                f.create_dataset(seg_key, data=seg, compression="lzf")
        elif store and not output_path:
            print("Output path is missing, not storing SV predictions")
        else:
            print("Not storing SV predictions")
    
    return seg


def compartment_pred(raw: np.ndarray, compartment_model: str, output_path: str = None, store: bool = False) -> np.ndarray:
    """
    Run compartment segmentation and optionally store the output.

    Args:
        raw (np.ndarray): Raw EM image volume.
        compartment_model (str): Path to compartment model.
        output_path (str): HDF5 file to store predictions.
        store (bool): Whether to store predictions.

    Returns:
        np.ndarray: Segmentation result.
    """

    pred_key = f"predictions/compartment/pred"
    seg_key = f"predictions/compartment/seg"

    use_existing_seg = False
    #checking if segmentation is already in output path and if so, use it
    if output_path and os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            if seg_key in f and pred_key in f:
                seg = f[seg_key][:]
                pred = f[pred_key][:]
                use_existing_seg = True
                print(f"Using existing compartment seg in {output_path}")

    if not use_existing_seg:
        seg, pred = segment_compartments(input_volume=raw, model_path=compartment_model, verbose=False, return_predictions=True, boundary_threshold=0.9)

        if store and output_path:
            with h5py.File(output_path, "a") as f:
                if pred_key in f:
                    print(f"{pred_key} already saved")
                else:
                    f.create_dataset(pred_key, data=pred, compression="lzf")

                f.create_dataset(seg_key, data=seg, compression="lzf")
        elif store and not output_path:
            print("Output path is missing, not storing compartment predictions")
        else:
            print("Not storing compartment predictions")

    return seg, pred


def AZ_pred(raw: np.ndarray, AZ_model: str, output_path: str = None, store: bool = False) -> np.ndarray:
    """
    Run active zone segmentation and optionally store the output.

    Args:
        raw (np.ndarray): Raw EM image volume.
        AZ_model (str): Path to AZ model.
        output_path (str): HDF5 file to store predictions.
        store (bool): Whether to store predictions.

    Returns:
        np.ndarray: Segmentation result.
    """
    pred_key = f"predictions/az/pred"
    seg_key = f"predictions/az/seg"

    use_existing_seg = False
    #checking if segmentation is already in output path and if so, use it
    if output_path and os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            if seg_key in f:
                seg = f[seg_key][:]
                use_existing_seg = True
                print(f"Using existing AZ seg in {output_path}")

    if not use_existing_seg:

        seg, pred = segment_active_zone(raw, model_path=AZ_model, verbose=False, return_predictions=True)

        if store and output_path:
            
            with h5py.File(output_path, "a") as f:
                if pred_key in f:
                    print(f"{pred_key} already saved")
                else:
                    f.create_dataset(pred_key, data=pred, compression="lzf")

                f.create_dataset(seg_key, data=seg, compression="lzf")
        elif store and not output_path:
            print("Output path is missing, not storing AZ predictions")
        else:
            print("Not storing AZ predictions")
    
    return seg


def filter_presynaptic_SV(sv_seg: np.ndarray, compartment_seg: np.ndarray, compartment_pred: np.ndarray, output_path: str = None,
                          store: bool = False, input_path: str = None) -> np.ndarray:
    """
    Filters synaptic vesicle segmentation to retain only vesicles in the presynaptic region.

    Args:
        sv_seg (np.ndarray): Vesicle segmentation.
        compartment_seg (np.ndarray): Compartment segmentation.
        output_path (str): Optional HDF5 file to store outputs.
        store (bool): Whether to store outputs.
        input_path (str): Path to input file (for filename-based filtering).

    Returns:
        np.ndarray: Filtered presynaptic vesicle segmentation.
    """
    # Fill out small holes in vesicles and then apply a size filter.
    vesicles_pp = fill_and_filter_vesicles(sv_seg)

    def n_vesicles(mask, ves):
        return len(np.unique(ves[mask])) - 1

    '''# Find the segment with most vesicles.
    props = regionprops(compartment_seg, intensity_image=vesicles_pp, extra_properties=[n_vesicles])
    compartment_ids = [prop.label for prop in props]
    vesicle_counts = [prop.n_vesicles for prop in props]
    if len(compartment_ids) == 0:
        mask = np.ones(compartment_seg.shape, dtype="bool")
    else:
        mask = (compartment_seg == compartment_ids[np.argmax(vesicle_counts)]).astype("uint8")'''

    mask = _get_presynaptic_mask(compartment_pred, vesicles_pp)

    # Filter all vesicles that are not in the mask.
    props = regionprops(vesicles_pp, mask)
    filter_ids = [prop.label for prop in props if prop.max_intensity == 0]

    name = os.path.basename(input_path) if input_path else "unknown"
    print(name)

    no_filter = ["C_M13DKO_080212_CTRL6.7B_crop.h5", "E_M13DKO_080212_DKO1.2_crop.h5",
                 "G_M13DKO_080212_CTRL6.7B_crop.h5", "A_SNAP25_120812_CTRL2.3_14_crop.h5",
                 "A_SNAP25_12082_KO2.1_6_crop.h5", "B_SNAP25_120812_CTRL2.3_14_crop.h5",
                 "B_SNAP25_12082_CTRL2.3_5_crop.h5", "D_SNAP25_120812_CTRL2.3_14_crop.h5",
                 "G_SNAP25_12.08.12_KO1.1_3_crop.h5"]
    # Don't filter for wrong masks (visual inspection)
    if name not in no_filter:
        vesicles_pp[np.isin(vesicles_pp, filter_ids)] = 0

    if store and output_path:
        seg_presynapse = f"predictions/compartment/presynapse"
        seg_presynaptic_SV = f"predictions/SV/presynaptic"

        with h5py.File(output_path, "a") as f:
            if seg_presynapse in f:
                print(f"{seg_presynapse} already saved")
            else:
                f.create_dataset(seg_presynapse, data=mask, compression="lzf")
            if seg_presynaptic_SV in f:
                print(f"{seg_presynaptic_SV} already saved")
            else:
                f.create_dataset(seg_presynaptic_SV, data=vesicles_pp, compression="lzf")
    elif store and not output_path:
        print("Output path is missing, not storing presynapse seg and presynaptic SV seg")
    else:
        print("Not storing presynapse seg and presynaptic SV seg")

    #All non-zero labels are relabeled starting from 1.Labels are sequential (1, 2, 3, ..., n).
    #We do this to make the analysis part easier -> can match distances and diameters better
    vesicles_pp, _, _ = relabel_sequential(vesicles_pp)

    return vesicles_pp


def run_predictions(input_path: str, output_path: str = None, store: bool = False):
    """
    Run full inference pipeline: vesicles, compartments, active zone, and presynaptic SV filtering.

    Args:
        input_path (str): Path to input HDF5 file with 'raw' dataset.
        output_path (str): Path to output HDF5 file to store predictions.
        store (bool): Whether to store intermediate and final results.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Filtered vesicle segmentation, AZ segmentation)
    """
    with h5py.File(input_path, "r") as f:
        raw = f["raw"][:]
    
    SV_model = get_model_path("vesicles_3d")
    compartment_model = get_model_path("compartments")
    # TODO upload better AZ model
    AZ_model = "/mnt/lustre-emmy-hdd/usr/u12095/synapse_net/models/ConstantinAZ/checkpoints/v7/"

    print("Running SV prediction")
    sv_seg = SV_pred(raw, SV_model, output_path, store)

    print("Running compartment prediction")
    comp_seg, comp_pred = compartment_pred(raw, compartment_model, output_path, store)

    print("Running AZ prediction")
    az_seg = AZ_pred(raw, AZ_model, output_path, store)

    print("Filtering the presynaptic SV")
    presyn_SV_seg = filter_presynaptic_SV(sv_seg, comp_seg, comp_pred, output_path, store, input_path)

    print("Done with predictions")

    return presyn_SV_seg, az_seg
