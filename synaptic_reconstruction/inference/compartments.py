import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import vigra
import torch

import elf.segmentation as eseg
import nifty
from elf.tracking.tracking_utils import compute_edges_from_overlap
from scipy.ndimage import distance_transform_edt, binary_closing
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.morphology import remove_small_holes

from synaptic_reconstruction.inference.util import get_prediction, _Scaler, _postprocess_seg_3d


def _segment_compartments_2d(
    boundaries,
    boundary_threshold=0.4,  # Threshold for the boundary distance computation.
    large_seed_distance=30,  # The distance threshold for computing large seeds (= components).
    distances=None,  # Pre-computed distances to take into account z-context.
):
    # Compoute distances if already not precomputed.
    if distances is None:
        distances = distance_transform_edt(boundaries < boundary_threshold).astype("float32")
        distances_z = distances
    else:
        # If the distances were pre-computed then compute them again in 2d.
        # This is needed for inserting small seeds from maxima, otherwise we will get spurious maxima.
        distances_z = distance_transform_edt(boundaries < boundary_threshold).astype("float32")

    # Find the large seeds as connected components in the distances > large_seed_distance.
    seeds = label(distances > large_seed_distance)

    # Remove to small large seeds.
    min_seed_area = 50
    ids, sizes = np.unique(seeds, return_counts=True)
    remove_ids = ids[sizes < min_seed_area]
    seeds[np.isin(seeds, remove_ids)] = 0

    # Compute the small seeds = local maxima of the in-plane distance map
    small_seeds = vigra.analysis.localMaxima(distances_z, marker=np.nan, allowAtBorder=True, allowPlateaus=True)
    small_seeds = label(np.isnan(small_seeds))

    # We only keep small seeds that don't intersect with a large seed.
    props = regionprops(small_seeds, seeds)
    keep_seeds = [prop.label for prop in props if prop.max_intensity == 0]
    keep_mask = np.isin(small_seeds, keep_seeds)

    # Add up the small seeds we keep with the large seeds.
    all_seeds = seeds.copy()
    seed_offset = seeds.max()
    all_seeds[keep_mask] = (small_seeds[keep_mask] + seed_offset)

    # Run watershed to get the segmentation.
    hmap = boundaries + (distances.max() - distances) / distances.max()
    raw_segmentation = watershed(hmap, markers=all_seeds)

    # Thee are the large seed ids that we will keep.
    keep_ids = list(range(1, seed_offset + 1))

    # Iterate over the ids, only keep large seeds and remove holes in their respective masks.
    props = regionprops(raw_segmentation)
    segmentation = np.zeros_like(raw_segmentation)
    for prop in props:
        if prop.label not in keep_ids:
            continue

        # Get bounding box and mask.
        bb = tuple(slice(start, stop) for start, stop in zip(prop.bbox[:2], prop.bbox[2:]))
        mask = raw_segmentation[bb] == prop.label

        # Fill small holes and apply closing.
        mask = remove_small_holes(mask, area_threshold=500)
        mask = np.logical_or(binary_closing(mask, iterations=4), mask)
        segmentation[bb][mask] = prop.label

    return segmentation


def _merge_segmentation_3d(seg_2d, beta=0.5, min_z_extent=10):
    edges = compute_edges_from_overlap(seg_2d, verbose=False)

    uv_ids = np.array([[edge["source"], edge["target"]] for edge in edges])
    overlaps = np.array([edge["score"] for edge in edges])

    n_nodes = int(seg_2d.max() + 1)
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)

    costs = eseg.multicut.compute_edge_costs(1.0 - overlaps)
    # set background weights to be maximally repulsive
    bg_edges = (uv_ids == 0).any(axis=1)
    costs[bg_edges] = -8.0

    node_labels = eseg.multicut.multicut_decomposition(graph, costs, beta=beta)
    segmentation = nifty.tools.take(node_labels, seg_2d)

    if min_z_extent is not None and min_z_extent > 0:
        props = regionprops(segmentation)
        filter_ids = []
        for prop in props:
            box = prop.bbox
            z_extent = box[3] - box[0]
            if z_extent < min_z_extent:
                filter_ids.append(prop.label)
        if filter_ids:
            segmentation[np.isin(segmentation, filter_ids)] = 0

    return segmentation


def _segment_compartments_3d(
    prediction,
    boundary_threshold=0.4,
    n_slices_exclude=0,
    min_z_extent=10,
):
    distances = distance_transform_edt(prediction < boundary_threshold).astype("float32")
    seg_2d = np.zeros(prediction.shape, dtype="uint32")

    offset = 0
    # Parallelize?
    for z in range(seg_2d.shape[0]):
        if z < n_slices_exclude or z >= seg_2d.shape[0] - n_slices_exclude:
            continue
        seg_z = _segment_compartments_2d(prediction[z], distances=distances[z])
        seg_z[seg_z != 0] += offset
        offset = int(seg_z.max())
        seg_2d[z] = seg_z

    seg = _merge_segmentation_3d(seg_2d, min_z_extent)
    seg = _postprocess_seg_3d(seg)

    # import napari
    # v = napari.Viewer()
    # v.add_image(prediction)
    # v.add_image(distances)
    # v.add_labels(seg_2d)
    # v.add_labels(seg)
    # napari.run()

    return seg


def segment_compartments(
    input_volume: np.ndarray,
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    tiling: Optional[Dict[str, Dict[str, int]]] = None,
    verbose: bool = True,
    return_predictions: bool = False,
    scale: Optional[List[float]] = None,
    mask: Optional[np.ndarray] = None,
    n_slices_exclude: int = 0,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Segment synaptic compartments in an input volume.

    Args:
        input_volume: The input volume to segment.
        model_path: The path to the model checkpoint if `model` is not provided.
        model: Pre-loaded model. Either `model_path` or `model` is required.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        return_predictions: Whether to return the predictions (foreground, boundaries) alongside the segmentation.
        scale: The scale factor to use for rescaling the input volume before prediction.
        n_slices_exclude:

    Returns:
        The segmentation mask as a numpy array, or a tuple containing the segmentation mask
        and the predictions if return_predictions is True.
    """
    if verbose:
        print("Segmenting compartments in volume of shape", input_volume.shape)

    # Create the scaler to handle prediction with a different scaling factor.
    scaler = _Scaler(scale, verbose)
    input_volume = scaler.scale_input(input_volume)

    # Run prediction. Support models with a single or multiple channels,
    # assuming that the first channel is the boundary prediction.
    pred = get_prediction(input_volume, tiling=tiling, model_path=model_path, model=model, verbose=verbose)

    # Remove channel axis if necessary.
    if pred.ndim != input_volume.ndim:
        assert pred.ndim == input_volume.ndim + 1
        pred = pred[0]

    # Run the compartment segmentation.
    # We may want to expose some of the parameters here.
    t0 = time.time()
    if input_volume.ndim == 2:
        seg = _segment_compartments_2d(pred)
    else:
        seg = _segment_compartments_3d(pred, n_slices_exclude=n_slices_exclude)
    if verbose:
        print("Run segmentation in", time.time() - t0, "s")

    seg = scaler.rescale_output(seg, is_segmentation=True)

    if return_predictions:
        pred = scaler.rescale_output(pred, is_segmentation=False)
        return seg, pred
    return seg
