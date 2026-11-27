import multiprocessing as mp
from concurrent import futures
from functools import partial
from typing import Optional

import numpy as np

import bioimage_cpp as bic
from scipy.ndimage import binary_erosion
from skimage import img_as_ubyte
from skimage.filters import gaussian, rank, sato, sobel
from skimage.morphology import disk
from skimage.segmentation import watershed

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
            - "structure-tensor": Edges are found based on the 1st eigenvalue of the structure tensor.
            - "sato": Edges are found with a sato-filter, followed by smoothing and leveling.
        per_slice: Compute the filter per slice instead of for the whole volume.
        n_threads: Number of threads for parallel computation over the slices.

    Returns:
        Edge filter response.
    """
    if method not in FILTERS:
        raise ValueError(f"Invalid edge filter method: {method}. Expect one of {FILTERS}.")

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
        edge_map = bic.filters.laplacian_of_gaussian(data.astype("float32"), sigma)
    elif method == "ggm":
        edge_map = bic.filters.gaussian_gradient_magnitude(data.astype("float32"), sigma)
    elif method == "structure-tensor":
        inner_scale, outer_scale = sigma, sigma * 0.5
        edge_map = bic.filters.structure_tensor_eigenvalues(
            data.astype("float32"), inner_scale, outer_scale
        )[..., 0]
    elif method == "sato":
        edge_map = _sato_filter(data, sigma)

    return edge_map


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


