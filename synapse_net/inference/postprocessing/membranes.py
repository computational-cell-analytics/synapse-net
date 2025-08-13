from typing import Optional
import numpy as np

from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops

from elf.parallel import label
from synapse_net.distance_measurements import compute_geodesic_distances


def segment_membrane_next_to_object(
    boundary_prediction: np.array,
    object_segmentation: np.array,
    n_slices_exclude: int,
    radius: int = 25,
    n_fragments: int = 1,
):
    """Derive boundary segmentation from boundary predictions by
    selecting large boundary fragment closest to the object.

    Args:
        boundary_prediction: Binary prediction for boundaries in the tomogram.
        object_segmentation: The object segmentation.
        n_slices_exclude: The number of slices to exclude on the top / bottom
            in order to avoid segmentation errors due to imaging artifacts in top and bottom.
        radius: The radius for membrane fragments that are considered.
        n_fragments: The number of boundary fragments to keep.
    """
    assert boundary_prediction.shape == object_segmentation.shape

    original_shape = boundary_prediction.shape

    # Cut away the exclude mask.
    slice_mask = np.s_[n_slices_exclude:-n_slices_exclude]
    boundary_prediction = boundary_prediction[slice_mask]
    object_segmentation = object_segmentation[slice_mask]

    # Label the boundary predictions.
    boundary_segmentation = label(boundary_prediction, block_shape=(32, 256, 256))

    # Compute the distance to object and the corresponding index.
    object_dist = distance_transform_edt(object_segmentation == 0)

    # Find the distances to the object and fragment size.
    ids = []
    distances = []
    sizes = []

    props = regionprops(boundary_segmentation)
    for prop in props:
        bb = prop.bbox
        bb = np.s_[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]

        label_id = prop.label
        boundary_mask = boundary_segmentation[bb] == label_id
        dist = object_dist[bb].copy()
        dist[~boundary_mask] = np.inf

        min_dist = np.min(dist)
        size = prop.area

        ids.append(prop.label)
        distances.append(min_dist)
        sizes.append(size)

    ids, distances, sizes = np.array(ids), np.array(distances), np.array(sizes)

    mask = distances < radius
    if mask.sum() > 0:
        ids, sizes = ids[mask], sizes[mask]

    keep_ids = ids[np.argsort(sizes)[::-1][:n_fragments]]

    # Create the output segmentation for the full output shape,
    # keeping only the boundary fragment closest to the PD.
    full_boundary_segmentation = np.zeros(original_shape, dtype="uint8")
    full_boundary_segmentation[slice_mask][np.isin(boundary_segmentation, keep_ids)] = 1

    return full_boundary_segmentation


def segment_membrane_distance_based(
    boundary_prediction: np.array,
    reference_segmentation: np.array,
    n_slices_exclude: int,
    max_distance: float,
    resolution: Optional[float] = None,
    min_size: int = 0,
):
    """Derive boundary segmentation from boundary predictions by selecting the fragment closest to the ribbon.

    Args:
        boundary_prediction: Binary prediction for boundaries in the tomogram.
        reference_segmentation: The reference segmentation, typically of the ribbon.
        n_slices_exclude: The number of slices to exclude on the top / bottom
            in order to avoid segmentation errors due to imaging artifacts in top and bottom.
        max_distance: The maximal distance from the ribbon to consider.
        resolution: The resolution / voxel size of the data.
        min_size: The minimal size of a boundary fragment to be included.
    """
    assert boundary_prediction.shape == reference_segmentation.shape

    original_shape = boundary_prediction.shape

    # Cut away the exclude mask.
    slice_mask = np.s_[n_slices_exclude:-n_slices_exclude]
    boundary_prediction = boundary_prediction[slice_mask]
    reference_segmentation = reference_segmentation[slice_mask]

    if min_size > 0:
        boundary_prediction = label(boundary_prediction, block_shape=(32, 256, 256))
        ids, sizes = np.unique(boundary_prediction, return_counts=True)
        ids, sizes = ids[1:], sizes[1:]
        keep_ids = ids[sizes > min_size]
        boundary_prediction = np.isin(boundary_prediction, keep_ids)

    # Get the unique objects in the reference segmentation.
    reference_ids = np.unique(reference_segmentation)
    assert reference_ids[0] == 0
    reference_ids = reference_ids[1:]

    # Compute the boundary fragments close to the unique objects in the reference.
    full_boundary_segmentation = np.zeros(original_shape, dtype="uint8")
    for seg_id in reference_ids:

        # First, we find the closest point on the membrane surface.
        ref_dist = distance_transform_edt(reference_segmentation != seg_id)
        ref_dist[boundary_prediction == 0] = np.inf
        closest_membrane = np.argmin(ref_dist)
        closest_point = np.unravel_index(closest_membrane, ref_dist.shape)

        # Then we compute the geodesic distance to this point on the distance and threshold it.
        boundary_segmentation = compute_geodesic_distances(
            boundary_prediction, closest_point, resolution
        ) < max_distance

        # boundary_segmentation = np.logical_and(boundary_prediction > 0, pd_dist < max_distance)
        full_boundary_segmentation[slice_mask][boundary_segmentation] = 1

    return full_boundary_segmentation
