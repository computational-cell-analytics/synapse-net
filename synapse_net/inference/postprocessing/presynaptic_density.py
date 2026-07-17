import numpy as np

from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops

from elf.parallel import label


# TODO update for multiple ribbons and pds
def segment_presynaptic_density(
    presyn_prediction: np.array,
    ribbon_segmentation: np.array,
    n_slices_exclude: int,
    n_pds_per_ribbon: int = 1,
    max_distance_to_ribbon: int = 15,
):
    """Derive presynaptic density segmentation from predictions by
    only keeping a PD prediction close to the ribbon.

    Args:
        presyn_prediction: Binary prediction for presynaptic densities in the tomogram.
        ribbon_segmentation: The ribbon segmentation.
        n_slices_exclude: The number of slices to exclude on the top / bottom
            in order to avoid segmentation errors due to imaging artifacts in top and bottom.
        max_distance_to_ribbon: The minimal distance to associate a PD with a ribbon.
    """
    assert presyn_prediction.shape == ribbon_segmentation.shape

    original_shape = ribbon_segmentation.shape

    # Cut away the exclude mask.
    slice_mask = np.s_[:] if n_slices_exclude == 0 else np.s_[n_slices_exclude:-n_slices_exclude]
    presyn_prediction = presyn_prediction[slice_mask]
    ribbon_segmentation = ribbon_segmentation[slice_mask]

    # Label the presyn predictions.
    presyn_segmentation = label(presyn_prediction, block_shape=(32, 256, 256))

    # Compute the distance to a ribbon.
    ribbon_dist, ribbon_idx = distance_transform_edt(ribbon_segmentation == 0, return_indices=True)

    # Associate presynaptic densities with ribbons.
    ribbon_matches = {}
    props = regionprops(presyn_segmentation)
    for prop in props:
        bb = prop.bbox
        bb = np.s_[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]

        presyn_mask = presyn_segmentation[bb] == prop.label
        dist, idx = ribbon_dist[bb].copy(), ribbon_idx[(slice(None),) + bb]
        dist[~presyn_mask] = np.inf

        min_dist_point = np.argmin(dist)
        min_dist_point = np.unravel_index(min_dist_point, presyn_mask.shape)

        this_distance = dist[min_dist_point]
        if this_distance > max_distance_to_ribbon:
            continue

        ribbon_coord = tuple(idx_[min_dist_point] for idx_ in idx)
        ribbon_id = ribbon_segmentation[ribbon_coord]
        assert ribbon_id != 0

        if ribbon_id in ribbon_matches:
            ribbon_matches[ribbon_id].append([prop.label, this_distance])
        else:
            ribbon_matches[ribbon_id] = [[prop.label, this_distance]]

    # Create the output segmentation for the full output shape,
    # keeping only the presyns that are associated with a ribbon.
    full_presyn_segmentation = np.zeros(original_shape, dtype="uint8")

    for ribbon_id, matches in ribbon_matches.items():
        if len(matches) == 0:  # no associated PD was found
            continue
        elif len(matches) == 1:  # exactly one associated PD was found
            presyn_ids = [matches[0][0]]
        else:  # multiple matches were found, assign all of them to the ribbon
            matched_ids, matched_distances = [match[0] for match in matches], [match[1] for match in matches]
            presyn_ids = np.array(matched_ids)[np.argsort(matched_distances)[:n_pds_per_ribbon]]

        full_presyn_segmentation[slice_mask][np.isin(presyn_segmentation, presyn_ids)] = ribbon_id

    if full_presyn_segmentation.sum() == 0:
        print("No presynapse was found")
    return full_presyn_segmentation
