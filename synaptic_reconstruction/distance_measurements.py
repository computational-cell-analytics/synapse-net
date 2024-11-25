import os
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

import numpy as np

from concurrent import futures
from scipy.ndimage import distance_transform_edt, binary_dilation
from sklearn.metrics import pairwise_distances

from skimage.measure import regionprops
from skimage.draw import line_nd
from tqdm import tqdm

try:
    import skfmm
except ImportError:
    skfmm = None


def compute_geodesic_distances(segmentation, distance_to, resolution=None, unsigned=True):
    assert skfmm is not None, "Please install scikit-fmm to use compute_geodesic_distance."

    invalid = segmentation == 0
    input_ = np.ma.array(segmentation.copy(), mask=invalid)
    input_[distance_to] = 0

    if resolution is None:
        dx = 1.0
    elif isinstance(resolution, (int, float)):
        dx = float(resolution)
    else:
        assert len(resolution) == segmentation.ndim
        dx = resolution

    distances = skfmm.distance(input_, dx=dx).data
    distances[distances == 0] = np.inf
    distances[distance_to] = 0

    if unsigned:
        distances = np.abs(distances)

    return distances


# TODO update this
def _compute_centroid_distances(segmentation, resolution, n_neighbors):
    # TODO enable eccentricity centers instead
    props = regionprops(segmentation)
    centroids = np.array([prop.centroid for prop in props])
    if resolution is not None:
        pass  # TODO scale the centroids

    pair_distances = pairwise_distances(centroids)
    return pair_distances


def _compute_boundary_distances(segmentation, resolution, n_threads):

    seg_ids = np.unique(segmentation)[1:]
    n = len(seg_ids)

    pairwise_distances = np.zeros((n, n))
    ndim = segmentation.ndim
    end_points1 = np.zeros((n, n, ndim), dtype="int")
    end_points2 = np.zeros((n, n, ndim), dtype="int")

    properties = regionprops(segmentation)
    properties = {prop.label: prop for prop in properties}

    def compute_distances_for_object(i):

        seg_id = seg_ids[i]
        distances, indices = distance_transform_edt(segmentation != seg_id, return_indices=True, sampling=resolution)

        for j in range(len(seg_ids)):
            if i >= j:
                continue

            ngb_id = seg_ids[j]
            prop = properties[ngb_id]

            bb = prop.bbox
            offset = np.array(bb[:ndim])
            if ndim == 2:
                bb = np.s_[bb[0]:bb[2], bb[1]:bb[3]]
            else:
                bb = np.s_[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]

            mask = segmentation[bb] == ngb_id
            ngb_dist, ngb_index = distances[bb].copy(), indices[(slice(None),) + bb]
            ngb_dist[~mask] = np.inf
            min_point_ngb = np.unravel_index(np.argmin(ngb_dist), shape=mask.shape)

            min_dist = ngb_dist[min_point_ngb]

            min_point = tuple(ind[min_point_ngb] for ind in ngb_index)
            pairwise_distances[i, j] = min_dist

            end_points1[i, j] = min_point
            min_point_ngb = [off + minp for off, minp in zip(offset, min_point_ngb)]
            end_points2[i, j] = min_point_ngb

    n_threads = mp.cpu_count() if n_threads is None else n_threads
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(compute_distances_for_object, range(n)), total=n, desc="Compute boundary distances"
        ))

    return pairwise_distances, end_points1, end_points2, seg_ids


def measure_pairwise_object_distances(
    segmentation: np.ndarray,
    distance_type: str = "boundary",
    resolution: Optional[Tuple[int, int, int]] = None,
    n_threads: Optional[int] = None,
    save_path: Optional[os.PathLike] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the pairwise distances between all objects within a segmentation.

    Args:
        segmentation: The input segmentation.
        distance_type: The type of distance to compute, can either be 'boundary' to
            compute the distance between the boundary / surface of the objects or 'centroid'
            to compute the distance between centroids.
        resolution: The resolution / pixel size of the data.
        n_threads: The number of threads for parallelizing the distance computation.
        save_path: Path for saving the measurement results in numpy zipped format.

    Returns:
        The pairwise object distances.
        The 'left' endpoint coordinates of the distances.
        The 'right' endpoint coordinates of the distances.
        The segmentation id pairs of the distances.
    """
    supported_distances = ("boundary", "centroid")
    assert distance_type in supported_distances
    if distance_type == "boundary":
        distances, endpoints1, endpoints2, seg_ids = _compute_boundary_distances(segmentation, resolution, n_threads)
    elif distance_type == "centroid":
        raise NotImplementedError
        # TODO has to be adapted
        # distances, neighbors = _compute_centroid_distances(segmentation, resolution)

    if save_path is not None:
        np.savez(
            save_path,
            distances=distances,
            endpoints1=endpoints1,
            endpoints2=endpoints2,
            seg_ids=seg_ids,
        )

    return distances, endpoints1, endpoints2, seg_ids


def _compute_seg_object_distances(segmentation, segmented_object, resolution, verbose):
    distance_map, indices = distance_transform_edt(segmented_object == 0, return_indices=True, sampling=resolution)

    seg_ids = np.unique(segmentation)[1:].tolist()
    n = len(seg_ids)

    distances = np.zeros(n)
    ndim = segmentation.ndim
    endpoints1 = np.zeros((n, ndim), dtype="int")
    endpoints2 = np.zeros((n, ndim), dtype="int")

    object_ids = []
    # We use this so often, it should be refactored.
    props = regionprops(segmentation)
    for prop in tqdm(props, disable=not verbose):
        bb = prop.bbox
        offset = np.array(bb[:ndim])
        if ndim == 2:
            bb = np.s_[bb[0]:bb[2], bb[1]:bb[3]]
        else:
            bb = np.s_[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]

        label = prop.label
        mask = segmentation[bb] == label

        dist, idx = distance_map[bb].copy(), indices[(slice(None),) + bb]
        dist[~mask] = np.inf

        min_dist_coord = np.argmin(dist)
        min_dist_coord = np.unravel_index(min_dist_coord, mask.shape)
        distance = dist[min_dist_coord]

        object_coord = tuple(idx_[min_dist_coord] for idx_ in idx)
        object_id = segmented_object[object_coord]
        assert object_id != 0

        seg_idx = seg_ids.index(label)
        distances[seg_idx] = distance
        endpoints1[seg_idx] = object_coord

        min_dist_coord = [off + minc for off, minc in zip(offset, min_dist_coord)]
        endpoints2[seg_idx] = min_dist_coord

        object_ids.append(object_id)

    return distances, endpoints1, endpoints2, np.array(seg_ids), np.array(object_ids)


def measure_segmentation_to_object_distances(
    segmentation: np.ndarray,
    segmented_object: np.ndarray,
    distance_type: str = "boundary",
    resolution: Optional[Tuple[int, int, int]] = None,
    save_path: Optional[os.PathLike] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the distance betwen all objects in a segmentation and another object.

    Args:
        segmentation: The input segmentation.
        segmented_object: The segmented object.
        distance_type: The type of distance to compute, can either be 'boundary' to
            compute the distance between the boundary / surface of the objects or 'centroid'
            to compute the distance between centroids.
        resolution: The resolution / pixel size of the data.
        save_path: Path for saving the measurement results in numpy zipped format.
        verbose: Whether to print the progress of the distance computation.

    Returns:
        The segmentation to object distances.
        The 'left' endpoint coordinates of the distances.
        The 'right' endpoint coordinates of the distances.
        The segmentation ids corresponding to the distances.
    """
    if distance_type == "boundary":
        distances, endpoints1, endpoints2, seg_ids, object_ids = _compute_seg_object_distances(
            segmentation, segmented_object, resolution, verbose
        )
        assert len(distances) == len(endpoints1) == len(endpoints2) == len(seg_ids) == len(object_ids)
    else:
        raise NotImplementedError

    if save_path is not None:
        np.savez(
            save_path,
            distances=distances,
            endpoints1=endpoints1,
            endpoints2=endpoints2,
            seg_ids=seg_ids,
            object_ids=object_ids,
        )
    return distances, endpoints1, endpoints2, seg_ids


def _extract_nearest_neighbors(pairwise_distances, seg_ids, n_neighbors, remove_duplicates=True):
    distance_matrix = pairwise_distances.copy()

    # Set the diagonal (distance to self) to infinity.
    distance_matrix[np.diag_indices(len(distance_matrix))] = np.inf
    # Mirror the distances.
    # (We only compute upper triangle, but need to take all distances into account here)
    tril_indices = np.tril_indices_from(distance_matrix)
    distance_matrix[tril_indices] = distance_matrix.T[tril_indices]

    neighbor_distances = np.sort(distance_matrix, axis=1)[:, :n_neighbors]
    neighbor_indices = np.argsort(distance_matrix, axis=1)[:, :n_neighbors]

    pairs = []
    for i, (dists, inds) in enumerate(zip(neighbor_distances, neighbor_indices)):
        seg_id = seg_ids[i]
        ngb_ids = [seg_ids[j] for j, dist in zip(inds, dists) if np.isfinite(dist)]
        pairs.extend([[min(seg_id, ngb_id), max(seg_id, ngb_id)] for ngb_id in ngb_ids])

    pairs = np.array(pairs)
    pairs = np.sort(pairs, axis=1)
    if remove_duplicates:
        pairs = np.unique(pairs, axis=0)
    return pairs


def load_distances(
    measurement_path: os.PathLike
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the saved distacnes from a zipped numpy file.

    Args:
        measurement_path: The path where the distances where saved.

    Returns:
        The segmentation to object distances.
        The 'left' endpoint coordinates of the distances.
        The 'right' endpoint coordinates of the distances.
        The segmentation ids corresponding to the distances.
    """
    auto_dists = np.load(measurement_path)
    distances, seg_ids = auto_dists["distances"], list(auto_dists["seg_ids"])
    endpoints1, endpoints2 = auto_dists["endpoints1"], auto_dists["endpoints2"]
    return distances, endpoints1, endpoints2, seg_ids


def create_pairwise_distance_lines(
    distances: np.ndarray,
    endpoints1: np.ndarray,
    endpoints2: np.ndarray,
    seg_ids: List[List[int]],
    n_neighbors: Optional[int] = None,
    pairs: Optional[np.ndarray] = None,
    bb: Optional[Tuple[slice]] = None,
    scale: Optional[float] = None,
    remove_duplicates: bool = True
) -> Tuple[np.ndarray, Dict]:
    """Create a line representation of pair-wise object distances for display in napari.

    Args:
        distances: The pairwise distances.
        endpoints1: One set of distance end points.
        endpoints2: The other set of distance end points.
        seg_ids: The segmentation pair corresponding to each distance.
        n_neighbors: ...
        pairs: ...
        bb: ....
        scale: ...
        remove_duplicates: ...

    Returns:
        The lines for plotting in napari.
        Additional attributes for the line layer in napari.
    """
    if pairs is None and n_neighbors is not None:
        pairs = _extract_nearest_neighbors(distances, seg_ids, n_neighbors, remove_duplicates=remove_duplicates)
    elif pairs is None:
        pairs = [[id1, id2] for id1 in seg_ids for id2 in seg_ids if id1 < id2]

    assert pairs is not None
    pair_indices = (
        np.array([seg_ids.index(pair[0]) for pair in pairs]),
        np.array([seg_ids.index(pair[1]) for pair in pairs])
    )

    pairs = np.array(pairs)
    distances = distances[pair_indices]
    endpoints1 = endpoints1[pair_indices]
    endpoints2 = endpoints2[pair_indices]

    if bb is not None:
        in_bb = np.where(
            (endpoints1[:, 0] > bb[0].start) & (endpoints1[:, 0] < bb[0].stop) &
            (endpoints1[:, 1] > bb[1].start) & (endpoints1[:, 1] < bb[1].stop) &
            (endpoints1[:, 2] > bb[2].start) & (endpoints1[:, 2] < bb[2].stop) &
            (endpoints2[:, 0] > bb[0].start) & (endpoints2[:, 0] < bb[0].stop) &
            (endpoints2[:, 1] > bb[1].start) & (endpoints2[:, 1] < bb[1].stop) &
            (endpoints2[:, 2] > bb[2].start) & (endpoints2[:, 2] < bb[2].stop)
        )

        pairs = pairs[in_bb]
        distances, endpoints1, endpoints2 = distances[in_bb], endpoints1[in_bb], endpoints2[in_bb]

        offset = np.array([b.start for b in bb])[None]
        endpoints1 -= offset
        endpoints2 -= offset

    lines = np.array([[start, end] for start, end in zip(endpoints1, endpoints2)])

    if scale is not None:
        scale_factor = np.array(3 * [scale])[None, None]
        lines //= scale_factor

    properties = {
        "id_a": pairs[:, 0],
        "id_b": pairs[:, 1],
        "distance": np.round(distances, 2),
    }
    return lines, properties


def create_object_distance_lines(
    distances: np.ndarray,
    endpoints1: np.ndarray,
    endpoints2: np.ndarray,
    seg_ids: np.ndarray,
    max_distance: Optional[float] = None,
    filter_seg_ids: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """Create a line representation of object distances for display in napari.

    Args:
        distances: The measurd distances.
        endpoints1: One set of distance end points.
        endpoints2: The other set of distance end points.
        seg_ids: The segmentation ids corresponding to each distance.
        max_distance: ...
        scale: ...

    Returns:
        The lines for plotting in napari.
        Additional attributes for the line layer in napari.
    """

    if filter_seg_ids is not None:
        id_mask = np.isin(seg_ids, filter_seg_ids)
        distances = distances[id_mask]
        endpoints1, endpoints2 = endpoints1[id_mask], endpoints2[id_mask]
        seg_ids = filter_seg_ids

    if max_distance is not None:
        distance_mask = distances <= max_distance
        distances, seg_ids = distances[distance_mask], seg_ids[distance_mask]
        endpoints1, endpoints2 = endpoints1[distance_mask], endpoints2[distance_mask]

    assert len(distances) == len(seg_ids) == len(endpoints1) == len(endpoints2)
    lines = np.array([[start, end] for start, end in zip(endpoints1, endpoints2)])

    if scale is not None and len(lines > 0):
        scale_factor = np.array(3 * [scale])[None, None]
        lines //= scale_factor

    properties = {"id": seg_ids, "distance": np.round(distances, 2)}
    return lines, properties


def keep_direct_distances(segmentation, measurement_path, line_dilation=0, scale=None):
    """Filter out all distances that are not direct.
    I.e. distances that cross another segmented object.
    """

    distances, ep1, ep2, seg_ids = load_distances(measurement_path)
    distance_lines, properties = create_object_distance_lines(distances, ep1, ep2, seg_ids, scale=scale)

    ids_a, ids_b = properties["id_a"], properties["id_b"]
    filtered_ids_a, filtered_ids_b = [], []

    for i, line in tqdm(enumerate(distance_lines), total=len(distance_lines)):
        id_a, id_b = ids_a[i], ids_b[i]

        start, stop = line
        line = line_nd(start, stop, endpoint=True)

        if line_dilation > 0:
            # TODO make this more efficient, ideally by dilating the mask coordinates
            # instead of dilating the actual mask.
            # We turn the line into a binary mask and dilate it to have some tolerance.
            line_vol = np.zeros_like(segmentation)
            line_vol[line] = 1
            line_vol = binary_dilation(line_vol, iterations=line_dilation)
        else:
            line_vol = line

        # Check if we cross any other segments:
        # Extract the unique ids in the segmentation that overlap with the segmentation.
        # We count this as a direct distance if no other object overlaps with the line.
        line_seg_ids = np.unique(segmentation[line_vol])
        line_seg_ids = np.setdiff1d(line_seg_ids, [0, id_a, id_b])

        if len(line_seg_ids) == 0:  # No other objet is overlapping, we keep the line.
            filtered_ids_a.append(id_a)
            filtered_ids_b.append(id_b)

    print("Keeping", len(filtered_ids_a), "/", len(ids_a), "distance pairs")
    filtered_pairs = [[ida, idb] for ida, idb in zip(filtered_ids_a, filtered_ids_b)]
    return filtered_pairs


def filter_blocked_segmentation_to_object_distances(
    segmentation, measurement_path, line_dilation=0, scale=None, seg_ids=None, verbose=False,
):
    distances, ep1, ep2, seg_ids = load_distances(measurement_path)
    distance_lines, properties = create_object_distance_lines(distances, ep1, ep2, seg_ids, scale=scale)
    all_seg_ids = properties["id"]

    filtered_ids = []
    for seg_id, line in tqdm(zip(all_seg_ids, distance_lines), total=len(distance_lines), disable=not verbose):
        if (seg_ids is not None) and (seg_id not in seg_ids):
            continue

        start, stop = line
        line = line_nd(start, stop, endpoint=True)

        if line_dilation > 0:
            # TODO make this more efficient, ideally by dilating the mask coordinates
            # instead of dilating the actual mask.
            # We turn the line into a binary mask and dilate it to have some tolerance.
            line_vol = np.zeros_like(segmentation)
            line_vol[line] = 1
            line_vol = binary_dilation(line_vol, iterations=line_dilation)
        else:
            line_vol = line

        # Check if we cross any other segments:
        # Extract the unique ids in the segmentation that overlap with the segmentation.
        # We count this as a direct distance if no other object overlaps with the line.
        line_seg_ids = np.unique(segmentation[line_vol])
        line_seg_ids = np.setdiff1d(line_seg_ids, [0, seg_id])

        if len(line_seg_ids) == 0:  # No other objet is overlapping, we keep the line.
            filtered_ids.append(seg_id)

    return filtered_ids
