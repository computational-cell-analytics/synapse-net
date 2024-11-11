import numpy as np

import elf.parallel as parallel
from skimage.measure import regionprops


def segment_ribbon(
    ribbon_prediction: np.array,
    vesicle_segmentation: np.array,
    n_slices_exclude: int,
    n_ribbons: int,
    max_vesicle_distance: int = 20,
):
    """Derive ribbon segmentation from ribbon predictions by
    filtering out ribbons that don't have sufficient associated vesicles.

    Args:
        ribbon_prediction: Binary prediction for ribbons in the tomogram.
        vesicle_segmentation: The vesicle segmentation.
        n_slices_exclude: The number of slices to exclude on the top / bottom
            in order to avoid segmentation errors due to imaging artifacts in top and bottom.
        n_ribbons: The number of ribbons in the tomogram.
        max_vesicle_distance: The maximal distance to associate a vesicle with a ribbon.
    """
    assert ribbon_prediction.shape == vesicle_segmentation.shape

    original_shape = ribbon_prediction.shape

    # Cut away the exclude mask.
    slice_mask = np.s_[n_slices_exclude:-n_slices_exclude]
    ribbon_prediction = ribbon_prediction[slice_mask]
    vesicle_segmentation = vesicle_segmentation[slice_mask]

    block_shape = (48, 384, 384)
    # Label the ribbon predictions.
    ribbon_segmentation = parallel.label(ribbon_prediction, block_shape=block_shape)

    # Compute the distance to ribbon and the corresponding index.
    halo = 3 * (max_vesicle_distance + 1,)
    ribbon_dist, ribbon_idx = parallel.distance_transform(
        ribbon_prediction == 0, return_indices=True, halo=halo, block_shape=block_shape
    )

    # Count the number of vesicles associated with each foreground object in the ribbon prediction.
    vesicle_counts = {}
    props = regionprops(vesicle_segmentation)
    for prop in props:
        bb = prop.bbox
        bb = np.s_[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]

        vesicle_mask = vesicle_segmentation[bb] == prop.label
        dist, idx = ribbon_dist[bb].copy(), ribbon_idx[(slice(None),) + bb]
        dist[~vesicle_mask] = np.inf

        min_dist_point = np.argmin(dist)
        min_dist_point = np.unravel_index(min_dist_point, vesicle_mask.shape)
        min_dist = dist[min_dist_point]

        if min_dist > max_vesicle_distance:
            continue

        ribbon_coord = tuple(idx_[min_dist_point] for idx_ in idx)
        ribbon_id = ribbon_segmentation[ribbon_coord]
        if ribbon_id == 0:
            continue

        if ribbon_id in vesicle_counts:
            vesicle_counts[ribbon_id] += 1
        else:
            vesicle_counts[ribbon_id] = 1

    # Create the output segmentation for the full output shape,
    # keeping only the ribbons with sufficient number of associated vesicles.
    full_ribbon_segmentation = np.zeros(original_shape, dtype="uint8")

    if vesicle_counts:
        ids = np.array(list(vesicle_counts.keys()))
        counts = np.array(list(vesicle_counts.values()))
    else:
        print("No vesicles were matched to a ribbon")
        print("Skipping postprocessing and returning the initial input")
        full_ribbon_segmentation[slice_mask] = ribbon_prediction
        return full_ribbon_segmentation

    ids = ids[np.argsort(counts)[::-1]]

    for output_id, ribbon_id in enumerate(ids[:n_ribbons], 1):
        full_ribbon_segmentation[slice_mask][ribbon_segmentation == ribbon_id] = output_id

    return full_ribbon_segmentation
