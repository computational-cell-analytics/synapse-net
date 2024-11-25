import multiprocessing as mp
from concurrent import futures
from functools import partial
from typing import List, Dict, Optional

import numpy as np

from scipy.ndimage import binary_erosion, binary_dilation
from skimage import img_as_ubyte
from skimage.filters import gaussian, rank, sato, sobel
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.segmentation import watershed
from tqdm import tqdm

try:
    import vigra
except ImportError:
    vigra = None

FILTERS = ("sobel", "laplace", "ggm", "structure-tensor", "sato")


def _sato_filter(raw, sigma, max_window=16):
    if raw.ndim != 2:
        raise NotImplementedError("The sato filter is only implemented for 2D data.")
    hmap = sato(raw)
    hmap = gaussian(hmap, sigma=sigma)
    hmap -= hmap.min()
    hmap /= hmap.max()
    hmap = rank.autolevel(img_as_ubyte(hmap), disk(max_window)).astype("float") / 255
    return hmap


def edge_filter(
    data: np.ndarray,
    sigma: float,
    method: str = "sato",
    per_slice: bool = True,
    n_threads: Optional[int] = None,
) -> np.ndarray:
    """Find edges in the image data.

    Args:
        data: The input data.
        sigma: The smoothing factor applied before the edge filter.
        method: The method for finding edges. The following methods are supported:
            - "sobel": Edges are found by smoothing the data and then applying a sobel filter.
            - "laplace": Edges are found with a laplacian of gaussian filter.
            - "ggm": Edges are found with a gaussian gradient magnitude filter.
            - "structure-tensor": Edges are found based on the 2nd eigenvalue of the structure tensor.
            - "sato": Edges are found with a sato-filter, followed by smoothing and leveling.
        per_slice: Compute the filter per slice instead of for the whole volume.
        n_threads: Number of threads for parallel computation over the slices.
    Returns:
        Volume with edge strength.
    """
    if method not in FILTERS:
        raise ValueError(f"Invalid edge filter method: {method}. Expect one of {FILTERS}.")
    if method in FILTERS[1:] and vigra is None:
        raise ValueError(f"Filter {method} requires vigra.")

    if per_slice and data.ndim == 3:
        n_threads = mp.cpu_count() if n_threads is None else n_threads
        filter_func = partial(edge_filter, sigma=sigma, method=method, per_slice=False)
        with futures.ThreadPoolExecutor(n_threads) as tp:
            edge_map = list(tp.map(filter_func, data))
        edge_map = np.stack(edge_map)
        return edge_map

    if method == "sobel":
        edge_map = gaussian(data, sigma=sigma)
        edge_map = sobel(edge_map)
    elif method == "laplace":
        edge_map = vigra.filters.laplacianOfGaussian(data.astype("float32"), sigma)
    elif method == "ggm":
        edge_map = vigra.filters.gaussianGradientMagnitude(data.astype("float32"), sigma)
    elif method == "structure-tensor":
        inner_scale, outer_scale = sigma, sigma * 0.5
        edge_map = vigra.filters.structureTensorEigenvalues(
            data.astype("float32"), innerScale=inner_scale, outerScale=outer_scale
        )[..., 1]
    elif method == "sato":
        edge_map = _sato_filter(data, sigma)

    return edge_map


def check_filters(
    data: np.ndarray,
    filters: List[str] = FILTERS,
    sigmas: List[float] = [2.0, 4.0],
    show: bool = True,
) -> Dict[str, np.ndarray]:
    """Apply different edge filters to the input data.

    Args:
        data: The input data volume.
        filters: The names of edge filters to apply.
            The filter names must match `method` in `edge_filter`.
        sigmas: The sigma values to use for the filters.
        show: Whether to show the filter responses in napari.
    Returns:
        Dictionary with the filter responses.
    """

    n_filters = len(filters) * len(sigmas)
    pbar = tqdm(total=n_filters, desc="Compute filters")

    responses = {}
    for filter_ in filters:
        for sigma in sigmas:
            name = f"{filter_}_{sigma}"
            responses[name] = edge_filter(data, sigma, method=filter_)
            pbar.update(1)

    if show:
        import napari

        v = napari.Viewer()
        v.add_image(data)
        for name, response in responses.items():
            v.add_image(response, name=name)
        napari.run()

    return responses


def refine_vesicle_shapes(
    vesicles: np.ndarray,
    edge_map: np.ndarray,
    foreground_erosion: int = 2,
    background_erosion: int = 6,
    fit_to_outer_boundary: bool = False,
    return_seeds: bool = False,
    compactness: float = 1.0,
) -> np.ndarray:
    """Refine vesicle shapes by fitting vesicles to a boundary map.

    This function erodes the segmented vesicles, and then fits them
    to a bonudary using a seeded watershed. This is done with two watersheds,
    one two separate foreground from background and one to separate vesicles within
    the foreground.

    Args:
        vesicles: The vesicle segmentation.
        edge_map: Volume with high intensities for vesicle membrane.
            You can use `edge_filter` to compute this based on the tomogram.
        foreground_erosion: By how many pixels the foreground should be eroded in the seeds.
        background_erosion: By how many pixels the background should be eroded in the seeds.
        fit_to_outer_boundary: Whether to fit the seeds to the outer membrane by
            applying a second edge filter to `edge_map`.
        return_seeds: Whether to return the seeds used for the watershed.
        compactness: The compactness parameter passed to the watershed function.
            Higher compactness leads to more regular sized vesicles.
    Returns:
        The refined vesicles.
    """

    fg = vesicles != 0
    if foreground_erosion > 0:
        fg_seeds = binary_erosion(fg, iterations=foreground_erosion).astype("uint8")
    else:
        fg_seeds = fg.astype("uint8")
    bg = vesicles == 0
    bg_seeds = binary_erosion(bg, iterations=background_erosion).astype("uint8")
    # Create 1 pixel wide mask and set to 1 and add to bg seed
    # Create a 1-pixel wide boundary at the edges of the tomogram
    boundary_mask = np.zeros_like(bg, dtype="uint8")

    # Set the boundary to 1 along the edges of each dimension
    boundary_mask[0, :, :] = 1
    boundary_mask[-1, :, :] = 1
    boundary_mask[:, 0, :] = 1
    boundary_mask[:, -1, :] = 1
    boundary_mask[:, :, 0] = 1
    boundary_mask[:, :, -1] = 1

    # Add the boundary to the background seeds without affecting existing seeds
    bg_seeds = np.clip(bg_seeds + boundary_mask, 0, 1)  # Ensure values are either 0 or 1

    if fit_to_outer_boundary:
        outer_edge_map = edge_filter(edge_map, sigma=2)
    else:
        outer_edge_map = edge_map

    seeds = fg_seeds + 2 * bg_seeds
    refined_mask = watershed(outer_edge_map, seeds, compactness=compactness)
    refined_mask[refined_mask == 2] = 0

    refined_vesicles = watershed(edge_map, vesicles, mask=refined_mask, compactness=compactness)

    if return_seeds:
        return refined_vesicles, seeds
    return refined_vesicles


def refine_individual_vesicle_shapes(
    vesicles: np.ndarray,
    edge_map: np.ndarray,
    foreground_erosion: int = 4,
    background_erosion: int = 8,
) -> np.ndarray:
    """Refine vesicle shapes by fitting vesicles to a boundary map.

    This function erodes the segmented vesicles, and then fits them
    to a bonudary using a seeded watershed. This is done individually for each vesicle.

    Args:
        vesicles: The vesicle segmentation.
        edge_map: Volume with high intensities for vesicle membrane.
            You can use `edge_filter` to compute this based on the tomogram.
        foreground_erosion: By how many pixels the foreground should be eroded in the seeds.
        background_erosion: By how many pixels the background should be eroded in the seeds.
    Returns:
        The refined vesicles.
    """

    refined_vesicles = np.zeros_like(vesicles)
    halo = [0, 12, 12]

    def fit_vesicle(prop):
        label_id = prop.label

        bb = prop.bbox
        bb = tuple(
            slice(max(start - ha, 0), min(stop + ha, sh)) for start, stop, ha, sh in
            zip(bb[:3], bb[3:], halo, vesicles.shape)
        )
        vesicle_sub = vesicles[bb]

        vesicle_mask = vesicle_sub == label_id
        hmap = edge_map[bb]

        # Do refinement in 2d to avoid effects caused by anisotropy.
        seg = np.zeros_like(vesicle_mask)
        for z in range(seg.shape[0]):
            m = vesicle_mask[z]
            fg_seed = binary_erosion(m, iterations=foreground_erosion).astype("uint8")
            if fg_seed.sum() == 0:
                seg[z][m] = 1
                continue
            bg_seed = (~binary_dilation(m, iterations=background_erosion)).astype("uint8")
            # Make sure all other vesicles in the local bbox are part of the bg seed,
            # to avoid leaking into other vesicles.
            bg_seed[(vesicle_sub[z] != 0) & (vesicle_sub[z] != label_id)] = 1

            # Run seeded watershed to fit the shapes.
            seeds = fg_seed + 2 * bg_seed
            seg[z] = watershed(hmap[z], seeds) == 1

        # import napari
        # v = napari.Viewer()
        # v.add_image(hmap)
        # # v.add_labels(seeds)
        # v.add_labels(seg)
        # v.title = label_id
        # napari.run()

        refined_vesicles[bb][seg] = label_id

    props = regionprops(vesicles)
    # fit_vesicle(props[1])
    n_threads = 8
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(fit_vesicle, props), total=len(props), disable=False, desc="refine vesicles"))

    return refined_vesicles
