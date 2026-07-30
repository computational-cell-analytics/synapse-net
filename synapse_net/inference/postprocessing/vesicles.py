import numpy as np

from scipy.ndimage import binary_closing, distance_transform_edt, find_objects, gaussian_filter
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential, watershed
from tqdm import tqdm


def close_holes(vesicle_segmentation, closing_iterations=4, min_size=0, verbose=False):
    assert vesicle_segmentation.ndim == 3
    props = regionprops(vesicle_segmentation)
    closed_segmentation = np.zeros_like(vesicle_segmentation)

    for prop in tqdm(props, desc="Close holes in segmentation", disable=not verbose):
        if prop.area < min_size:
            continue
        bb = prop.bbox
        bb = tuple(slice(beg, end) for beg, end in zip(bb[:3], bb[3:]))
        mask = vesicle_segmentation[bb] == prop.label
        closed_mask = np.logical_or(binary_closing(mask, iterations=closing_iterations), mask)
        closed_segmentation[bb][closed_mask] = prop.label

    return closed_segmentation


def split_touching_vesicles(
    segmentation: np.ndarray,
    min_distance: int = 12,
    smooth_sigma: float = 2.0,
    z_open: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Split vesicles that were segmented as a single instance because they touch.

    Each instance is re-partitioned by a watershed over its own 3D distance transform, seeded on
    the distance maxima. The cut therefore follows the narrowest 3D neck between the blobs and
    curves around each vesicle, rather than slicing straight through z.

    Args:
        segmentation: The input vesicle segmentation.
        min_distance: The minimum distance in voxels between two vesicle centers for a split.
            Set it to roughly the vesicle radius; lower values over-split single vesicles.
        smooth_sigma: Gaussian sigma applied to the distance transform before locating centers.
            Suppresses spurious maxima that would over-split a single vesicle. 0 disables smoothing.
        z_open: Whether to treat the upper / lower z faces as open instead of as membrane. This is
            the right choice for tomograms thinner than a vesicle diameter, where most vesicles are
            truncated: counting the slab faces as surface would flatten the distance transform and
            pull the cut back towards a vertical plane.
        verbose: Whether to report how many instances were split.

    Returns:
        The segmentation with touching vesicles split into separate instances.
    """
    assert segmentation.ndim == 3
    seg = segmentation.astype(np.int64, copy=True)  # headroom for the ids created while splitting
    next_id = int(seg.max()) + 1
    n_split = 0

    for label, bb in enumerate(find_objects(seg), start=1):
        if bb is None:
            continue
        sub = seg[bb] == label

        # Pad so the distance transform sees the lateral membrane as surface. See `z_open`.
        sub_padded = np.pad(sub, ((1, 1), (0, 0), (0, 0)), mode="edge" if z_open else "constant")
        sub_padded = np.pad(sub_padded, ((0, 0), (1, 1), (1, 1)), mode="constant")

        distances = distance_transform_edt(sub_padded)
        centers = gaussian_filter(distances, smooth_sigma) if smooth_sigma else distances
        peaks = peak_local_max(
            centers, min_distance=min_distance, threshold_abs=0.4 * min_distance,
            labels=sub_padded, exclude_border=False,
        )
        if peaks.shape[0] < 2:
            continue

        markers = np.zeros(sub_padded.shape, dtype=np.int32)
        markers[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
        basins = watershed(-distances, markers, mask=sub_padded)[1:-1, 1:-1, 1:-1]

        # The first basin keeps the original id, every further one becomes a new instance.
        for basin_id in range(2, peaks.shape[0] + 1):
            region = np.logical_and(basins == basin_id, sub)
            if not region.any():
                continue
            seg[bb][region] = next_id
            next_id += 1
        n_split += 1

    if verbose:
        print(f"Split {n_split} touching vesicle instance(s)")

    seg = relabel_sequential(seg)[0]
    dtype = segmentation.dtype
    if np.issubdtype(dtype, np.integer) and seg.max() <= np.iinfo(dtype).max:
        seg = seg.astype(dtype)
    return seg


def filter_border_objects(segmentation: np.ndarray, z_border_only: bool = False) -> np.ndarray:
    """Filter any object that touches one of the volume borders.

    Args:
        segmentation: The input segmentation.
        z_border_only: Whether to only filter the objects that touch the depth axis border (True)
            or to filter all objects touching an image borhder (False).

    Returns:
        The filtered segmentation.
    """
    props = regionprops(segmentation)

    filter_ids = []
    for prop in props:
        bbox = np.array(prop.bbox)
        if z_border_only:
            z_start, z_stop = bbox[0], bbox[3]
            if z_start == 0 or z_stop == segmentation.shape[0]:
                filter_ids.append(prop.label)
        else:
            start, stop = bbox[:3], bbox[3:]
            if (start == 0).any() or (stop == np.array(segmentation.shape)).any():
                filter_ids.append(prop.label)

    segmentation[np.isin(segmentation, filter_ids)] = 0
    return segmentation


def filter_border_vesicles(vesicle_segmentation, seg_ids=None, border_slices=4):
    props = regionprops(vesicle_segmentation)

    filtered_ids = []
    for prop in tqdm(props, desc="Filter vesicles at the tomogram border"):
        seg_id = prop.label
        if (seg_ids is not None) and (seg_id not in seg_ids):
            continue

        bb = prop.bbox
        bb = tuple(slice(beg, end) for beg, end in zip(bb[:3], bb[3:]))
        mask = vesicle_segmentation[bb] == seg_id

        # Compute the mass per slice. Only keep the vesicle if the maximum of the mass is central.
        mass_per_slice = [m.sum() for m in mask]
        max_slice = np.argmax(mass_per_slice)
        if (max_slice >= border_slices) and (max_slice < mask.shape[0] - border_slices):
            filtered_ids.append(seg_id)

    # print(len(filtered_ids), "/", len(seg_ids))
    return filtered_ids
