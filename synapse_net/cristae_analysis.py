import os
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from scipy.ndimage import label as ndimage_label
from skimage.graph import MCP_Geometric
from skimage.measure import marching_cubes, mesh_surface_area, regionprops
from skimage.morphology import disk, local_maxima
from skimage.segmentation import find_boundaries
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_sampling(voxel_size: Union[float, Dict[str, float]], ndim: int) -> np.ndarray:
    axes = ("z", "y", "x") if ndim == 3 else ("y", "x")
    if isinstance(voxel_size, dict):
        return np.array([voxel_size[ax] for ax in axes[:ndim]], dtype=float)
    return np.full(ndim, float(voxel_size))


def _voxel_radius(thickness_nm: float, voxel_size: Union[float, Dict[str, float]], ndim: int) -> int:
    return max(1, int(round(thickness_nm / float(np.mean(_to_sampling(voxel_size, ndim))))))


def _voxel_radius_xy(thickness_nm: float, voxel_size: Union[float, Dict[str, float]]) -> int:
    """Membrane radius in XY pixels — uses only the Y and X voxel sizes."""
    if isinstance(voxel_size, dict):
        xy_nm = (voxel_size["y"] + voxel_size["x"]) / 2.0
    else:
        xy_nm = float(voxel_size)
    return max(1, int(round(thickness_nm / xy_nm)))


def _border_zone(shape: tuple, radius: int) -> np.ndarray:
    """Boolean mask that is True within `radius` voxels of any face of the volume."""
    mask = np.zeros(shape, dtype=bool)
    for ax in range(len(shape)):
        idx_lo = [slice(None)] * len(shape)
        idx_hi = [slice(None)] * len(shape)
        idx_lo[ax] = slice(0, radius)
        idx_hi[ax] = slice(shape[ax] - radius, None)
        mask[tuple(idx_lo)] = True
        mask[tuple(idx_hi)] = True
    return mask


def _surface_area(mask: np.ndarray, sampling: np.ndarray) -> float:
    """Closed-surface area (nm^2) of a binary mask via marching cubes.

    The mask is padded by one background voxel so objects touching the array edge
    (e.g. an instance cropped to its bounding box) yield a closed surface rather than an
    open mesh with the edge-touching faces missing.

    Args:
        mask: Binary segmentation.
        sampling: Voxel size per axis (nm), in array (z, y, x) order.

    Returns:
        Surface area in nm^2, or NaN if the mask is empty.
    """
    binary = mask.astype(bool)
    if not binary.any():
        return np.nan
    padded = np.pad(binary.astype(np.float32), 1)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=tuple(float(s) for s in sampling))
    return float(mesh_surface_area(verts, faces))


# ---------------------------------------------------------------------------
# Membrane approximation
# ---------------------------------------------------------------------------

def approximate_membrane(
    mito_segmentation: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    membrane_thickness_nm: float = 8.0,
    border_gap_nm: Optional[float] = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Approximate the mitochondrial membrane as an inner shell of the segmentation.

    For 3D inputs the erosion is applied independently per Z-slice using a 2D disk,
    so a mitochondrion that changes shape rapidly along Z does not cause the membrane
    shell to bleed into neighbouring slices in XY. For 2D inputs a single erosion is
    applied to the whole array.

    Membrane voxels within border_gap_nm of any volume face are suppressed to avoid
    treating clipped mito edges as membrane.

    Args:
        mito_segmentation: Instance label array (background = 0).
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        membrane_thickness_nm: Thickness of the membrane shell in nm.
        border_gap_nm: Distance from each volume face within which membrane voxels are
            suppressed. Defaults to membrane_thickness_nm when None.
        n_jobs: Number of workers for the per-Z-slice erosion (3D only). 1 = serial;
            -1 = all cores. Results are identical regardless of n_jobs.

    Returns:
        membrane_mask: Binary mask of the mitochondrial membrane (outermost shell),
            with border-adjacent voxels zeroed out.
    """
    ndim = mito_segmentation.ndim
    mito_binary = mito_segmentation > 0

    if ndim == 3:
        membrane_radius = _voxel_radius_xy(membrane_thickness_nm, voxel_size)
        struct = disk(membrane_radius)
        membrane_mask = np.zeros_like(mito_binary)

        # Only the mito bounding box needs eroding; a `membrane_radius` margin in XY makes the
        # cropped erosion (with border_value=1) byte-identical to eroding the full slice, and
        # empty slices are skipped. This removes wasted work on the (often large) background.
        coords = np.argwhere(mito_binary)
        if coords.size:
            zmin, ymin, xmin = coords.min(axis=0)
            zmax, ymax, xmax = coords.max(axis=0) + 1
            m = membrane_radius
            y0, y1 = max(0, ymin - m), min(mito_binary.shape[1], ymax + m)
            x0, x1 = max(0, xmin - m), min(mito_binary.shape[2], xmax + m)

            def _erode_slice(z):
                sl = mito_binary[z, y0:y1, x0:x1]
                if not sl.any():
                    return z, None
                return z, sl & ~binary_erosion(sl, structure=struct, border_value=1)

            z_range = range(int(zmin), int(zmax))
            if n_jobs == 1:
                results = [_erode_slice(z) for z in z_range]
            else:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_erode_slice)(z) for z in z_range
                )
            for z, res in results:
                if res is not None:
                    membrane_mask[z, y0:y1, x0:x1] = res
    else:
        membrane_radius = _voxel_radius(membrane_thickness_nm, voxel_size, ndim)
        membrane_mask = mito_binary & ~binary_erosion(mito_binary, structure=disk(membrane_radius), border_value=1)

    gap_nm = border_gap_nm if border_gap_nm is not None else membrane_thickness_nm
    gap_radius = _voxel_radius(gap_nm, voxel_size, ndim)
    membrane_mask &= ~_border_zone(mito_segmentation.shape, gap_radius)
    return membrane_mask.astype(bool)


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def compute_crista_orientation(
    crista_mask: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    neighborhood_size_nm: float = 30.0,
    need_eigenvectors: bool = True,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Compute dominant crista orientation via structure tensor.

    Args:
        crista_mask: Binary crista segmentation.
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        neighborhood_size_nm: Gaussian smoothing radius in nm for tensor averaging.
        need_eigenvectors: If False, the eigenvectors are not computed and a low-memory path
            is used: the full (..., ndim, ndim) structure tensor is never materialised (only
            the unique smoothed components are kept) and eigenvalues are evaluated in chunks.
            In that case the returned eigenvalues and eigenvectors are both None — only the
            anisotropy is produced. Use this when only the anisotropy scalar is needed, e.g.
            in :func:`compute_mito_crista_statistics`.
        n_jobs: Number of threads for the structure-tensor Gaussian smoothing (low-memory path
            only). The unique tensor components are independent and ``gaussian_filter`` releases
            the GIL, so threads give a real speedup. 1 = serial; -1 = all cores.

    Returns:
        eigenvalues: (..., ndim) sorted ascending, or None when ``need_eigenvectors`` is False.
        eigenvectors: (..., ndim, ndim) — columns are principal directions, or None when
            ``need_eigenvectors`` is False.
        anisotropy: (...) — λ_max / (λ_min + ε) per voxel. High values indicate a strongly
            directional crista (e.g. parallel lamellae); low values indicate isotropic or
            tubular/disordered morphology.
    """
    ndim = crista_mask.ndim
    sampling = _to_sampling(voxel_size, ndim)
    sigma = neighborhood_size_nm / sampling

    grads = np.gradient(crista_mask.astype(np.float32), *sampling.tolist())

    if need_eigenvectors:
        J = np.zeros(crista_mask.shape + (ndim, ndim), dtype=np.float32)
        for i in range(ndim):
            for j in range(i, ndim):
                s = gaussian_filter(grads[i] * grads[j], sigma=sigma)
                J[..., i, j] = s
                J[..., j, i] = s
        eigenvalues, eigenvectors = np.linalg.eigh(J)
        anisotropy = eigenvalues[..., -1] / (eigenvalues[..., 0] + 1e-10)
        return eigenvalues, eigenvectors, anisotropy

    # Low-memory path: keep only the ndim*(ndim+1)/2 unique smoothed components instead of
    # the full (..., ndim, ndim) tensor, and evaluate eigenvalues chunk-wise along axis 0 so
    # only a small block is stacked at any time. Numerics match the full eigvalsh exactly.
    # The unique components are independent Gaussian smoothings (GIL-releasing) → thread them.
    pairs = [(i, j) for i in range(ndim) for j in range(i, ndim)]

    def _component(i, j):
        return (i, j), gaussian_filter(grads[i] * grads[j], sigma=sigma)

    workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    if workers == 1 or len(pairs) == 1:
        components = dict(_component(i, j) for i, j in pairs)
    else:
        from joblib import Parallel, delayed
        components = dict(
            Parallel(n_jobs=min(workers, len(pairs)), prefer="threads")(
                delayed(_component)(i, j) for i, j in pairs
            )
        )
    del grads

    shape = crista_mask.shape
    anisotropy = np.empty(shape, dtype=np.float32)
    plane_voxels = int(np.prod(shape[1:])) if ndim > 1 else 1
    chunk = max(1, 1_000_000 // max(1, plane_voxels))
    for z0 in range(0, shape[0], chunk):
        z1 = min(z0 + chunk, shape[0])
        block = np.empty((z1 - z0,) + shape[1:] + (ndim, ndim), dtype=np.float32)
        for (i, j), comp in components.items():
            block[..., i, j] = comp[z0:z1]
            block[..., j, i] = comp[z0:z1]
        evals = np.linalg.eigvalsh(block)
        anisotropy[z0:z1] = evals[..., -1] / (evals[..., 0] + 1e-10)
        del block, evals
    return None, None, anisotropy


# ---------------------------------------------------------------------------
# Proximity
# ---------------------------------------------------------------------------

def compute_crista_proximity(
    crista_mask: np.ndarray,
    membrane_mask: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    membrane_distance: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Distance from each crista voxel to the nearest membrane voxel (nm).

    Args:
        crista_mask: Binary crista segmentation.
        membrane_mask: Binary membrane mask (OM or IMM).
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        membrane_distance: Optional precomputed per-voxel distance to the nearest membrane
            voxel (nm), i.e. ``distance_transform_edt(~membrane_mask, sampling=...)``. When
            given, the distance transform is not recomputed (used to avoid redundant work in
            :func:`compute_mito_crista_statistics`).

    Returns:
        distance_map: Per-voxel distance to membrane (nm); zero outside crista.
        summary_stats: min_nm, median_nm, max_nm.
    """
    sampling = _to_sampling(voxel_size, crista_mask.ndim)
    if membrane_distance is None:
        dist = distance_transform_edt(~membrane_mask.astype(bool), sampling=sampling.tolist())
    else:
        dist = membrane_distance
    crista_dists = dist[crista_mask.astype(bool)]

    if crista_dists.size == 0:
        summary: Dict[str, float] = {"min_nm": np.nan, "median_nm": np.nan, "max_nm": np.nan}
    else:
        summary = {
            "min_nm": float(crista_dists.min()),
            "median_nm": float(np.median(crista_dists)),
            "max_nm": float(crista_dists.max()),
        }

    distance_map = np.zeros(crista_mask.shape, dtype=np.float32)
    distance_map[crista_mask.astype(bool)] = crista_dists
    return distance_map, summary


def compute_crista_density(
    crista_mask: np.ndarray,
    mito_mask: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
) -> Dict[str, float]:
    """Volume fraction of crista within a mitochondrion.

    Args:
        crista_mask: Binary crista segmentation.
        mito_mask: Binary or instance mito mask (> 0 = inside mito).
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.

    Returns:
        Dict with crista_volume_nm3, mito_volume_nm3, crista_fraction.
    """
    sampling = _to_sampling(voxel_size, crista_mask.ndim)
    voxel_vol = float(np.prod(sampling))
    mito_binary = mito_mask > 0

    mito_vol = float(mito_binary.sum()) * voxel_vol
    crista_vol = float((crista_mask.astype(bool) & mito_binary).sum()) * voxel_vol
    return {
        "crista_volume_nm3": crista_vol,
        "mito_volume_nm3": mito_vol,
        "crista_fraction": crista_vol / mito_vol if mito_vol > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# Contact sites
# ---------------------------------------------------------------------------

def detect_contact_sites(
    crista_mask: np.ndarray,
    membrane_mask: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Detect crista-membrane contact sites as the direct overlap of the two masks.

    Contact = crista voxels that are also membrane voxels (the pure intersection of the crista
    mask and the mitochondrial membrane band). No dilation/erosion is applied here, so the
    detected junctions correspond exactly to the visible overlap of the two layers; connected
    overlaps are grouped into junctions with 26-connectivity in 3D.

    Args:
        crista_mask: Binary crista segmentation.
        membrane_mask: Binary mitochondrial membrane mask (as displayed).
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.

    Returns:
        contact_labels: Integer array (same shape as the input) where each connected
            junction has a unique ID (0 = background, 1..n = junctions). Contact voxel
            coordinates are recoverable via ``np.argwhere(contact_labels > 0)``.
        summary: contact_voxel_count, crista_junction_count, contact_volume_nm3.
    """
    ndim = crista_mask.ndim
    sampling = _to_sampling(voxel_size, ndim)
    voxel_vol = float(np.prod(sampling))

    contact_mask = crista_mask.astype(bool) & membrane_mask.astype(bool)

    connectivity_struct = np.ones(ndim * (3,), dtype=bool)
    contact_labels, n_regions = ndimage_label(contact_mask, structure=connectivity_struct)
    contact_voxel_count = int(np.count_nonzero(contact_labels))

    return contact_labels, {
        "contact_voxel_count": contact_voxel_count,
        "crista_junction_count": int(n_regions),
        "contact_volume_nm3": float(contact_voxel_count) * voxel_vol,
    }


_JUNCTION_DISTANCE_NAN = {
    "junction_count": 0,
    "mean_nn_junction_distance_nm": np.nan,
    "median_nn_junction_distance_nm": np.nan,
    "junction_clustering_index": np.nan,
}


def _geodesic_rows(costs: np.ndarray, sampling_t: tuple, seeds: list, seed_idxs: list) -> list:
    """Geodesic distance rows for a chunk of source seeds (one MCP build per chunk).

    Module-level so it is picklable for a loky process pool. Returns (row_index, distances)
    for each source seed in ``seed_idxs``, where distances are to every seed.
    """
    mcp = MCP_Geometric(costs, sampling=sampling_t)
    out = []
    for i in seed_idxs:
        cum, _ = mcp.find_costs([seeds[i]])
        out.append((i, [float(cum[s]) for s in seeds]))
    return out


def compute_junction_distances(
    contact_labels: np.ndarray,
    membrane_mask: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    surface_area_nm2: Optional[float] = None,
    membrane_indices: Optional[np.ndarray] = None,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Geodesic distances between crista-membrane junctions measured along the membrane.

    Each junction (a connected component in ``contact_labels``) is reduced to its centroid,
    snapped to the nearest membrane voxel, and pairwise geodesic distances are computed over
    the membrane shell with ``skimage.graph.MCP_Geometric`` (so paths follow the surface
    instead of cutting through the lumen). A Clark-Evans nearest-neighbour index summarises
    whether the junctions are clustered.

    Args:
        contact_labels: Integer junction label array (0 = background, 1..n = junctions),
            e.g. the first return value of :func:`detect_contact_sites`.
        membrane_mask: Binary mitochondrial membrane mask the junctions sit on.
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        surface_area_nm2: Membrane/mito surface area used as the reference area for the
            Clark-Evans expectation. If None or non-positive, the clustering index is NaN.
        membrane_indices: Optional precomputed nearest-membrane-voxel index array, i.e. the
            second output of ``distance_transform_edt(~membrane_mask, return_indices=True,
            sampling=...)``. When given, the distance transform is not recomputed.

    Returns:
        distance_matrix: (n, n) geodesic distances in nm between junctions; the diagonal is
            0 and unreachable pairs (disconnected membrane fragments) are NaN. Empty for
            fewer than two junctions.
        summary: junction_count, mean_nn_junction_distance_nm, median_nn_junction_distance_nm,
            junction_clustering_index (Clark-Evans R: < 1 clustered, ~ 1 random, > 1 dispersed).

    Notes:
        The Clark-Evans expected nearest-neighbour distance uses the standard 2-D planar
        approximation ``0.5 * sqrt(A / n)`` with ``A = surface_area_nm2``.
    """
    ndim = contact_labels.ndim
    sampling = _to_sampling(voxel_size, ndim)
    membrane = membrane_mask.astype(bool)

    labels = [lbl for lbl in np.unique(contact_labels) if lbl != 0]
    n = len(labels)
    if n < 2 or not membrane.any():
        summary = dict(_JUNCTION_DISTANCE_NAN)
        summary["junction_count"] = n
        return np.zeros((n, n), dtype=float), summary

    # Snap each junction centroid to the nearest membrane voxel so the geodesic runs on the membrane.
    if membrane_indices is None:
        _, nearest_idx = distance_transform_edt(~membrane, return_indices=True, sampling=sampling.tolist())
    else:
        nearest_idx = membrane_indices
    seeds = []
    for lbl in labels:
        coords = np.argwhere(contact_labels == lbl)
        centroid = np.round(coords.mean(axis=0)).astype(int)
        centroid = np.clip(centroid, 0, np.array(contact_labels.shape) - 1)
        seed = tuple(int(nearest_idx[d][tuple(centroid)]) for d in range(ndim))
        seeds.append(seed)

    # Pairwise geodesic distances over the membrane (cost 1 on membrane, impassable elsewhere).
    # Each source seed is an independent MCP run; parallelize them across processes (MCP holds
    # the GIL, so a process pool is needed for real speedup). Order-independent → identical result.
    costs = np.where(membrane, 1.0, np.inf)
    sampling_t = tuple(float(s) for s in sampling)
    distance_matrix = np.full((n, n), np.nan, dtype=float)

    workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    if workers == 1 or n <= 2:
        rows = _geodesic_rows(costs, sampling_t, seeds, list(range(n)))
    else:
        from joblib import Parallel, delayed
        # Round-robin chunks so each worker builds MCP once and processes several sources.
        chunks = [list(range(k, n, workers)) for k in range(min(workers, n))]
        results = Parallel(n_jobs=workers, prefer="processes")(
            delayed(_geodesic_rows)(costs, sampling_t, seeds, ch) for ch in chunks
        )
        rows = [item for chunk in results for item in chunk]
    for i, row in rows:
        distance_matrix[i] = row
    distance_matrix[~np.isfinite(distance_matrix)] = np.nan
    np.fill_diagonal(distance_matrix, 0.0)

    # Nearest-neighbour distance per junction (nearest reachable other junction).
    nn_distances = []
    for i in range(n):
        others = np.delete(distance_matrix[i], i)
        finite = others[np.isfinite(others)]
        if finite.size:
            nn_distances.append(float(finite.min()))
    nn_distances = np.array(nn_distances, dtype=float)

    mean_nn = float(np.mean(nn_distances)) if nn_distances.size else np.nan
    median_nn = float(np.median(nn_distances)) if nn_distances.size else np.nan

    clustering_index = np.nan
    if surface_area_nm2 is not None and surface_area_nm2 > 0 and np.isfinite(mean_nn):
        expected_nn = 0.5 * np.sqrt(float(surface_area_nm2) / n)
        if expected_nn > 0:
            clustering_index = mean_nn / expected_nn

    return distance_matrix, {
        "junction_count": n,
        "mean_nn_junction_distance_nm": mean_nn,
        "median_nn_junction_distance_nm": median_nn,
        "junction_clustering_index": clustering_index,
    }


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------

def compute_crista_morphology(
    crista_mask: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    method: str = "both",
) -> Dict[str, float]:
    """Compute crista shape metrics from binary mask.

    Args:
        crista_mask: Binary crista segmentation.
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        method: "area" | "medial_axis" | "both".

    Returns:
        Dict with cristae_surface_area_nm2 (area/both) and avg_thickness_nm (medial_axis/both).
        avg_thickness_nm is 2 × mean distance-transform value at skeleton voxels.
    """
    if method not in ("area", "medial_axis", "both"):
        raise ValueError(f"method must be 'area', 'medial_axis', or 'both', got {method!r}")

    sampling = _to_sampling(voxel_size, crista_mask.ndim)
    spacing = tuple(float(s) for s in sampling)
    result: Dict[str, float] = {}

    if method in ("area", "both"):
        result["cristae_surface_area_nm2"] = _surface_area(crista_mask, sampling)

    if method in ("medial_axis", "both"):
        dist = distance_transform_edt(crista_mask.astype(bool), sampling=spacing)
        # Local maxima of the EDT form the medial axis; 2 × distance there = local thickness.
        ridges = local_maxima(dist) & crista_mask.astype(bool)
        ridge_dists = dist[ridges]
        result["avg_thickness_nm"] = float(2.0 * np.mean(ridge_dists)) if ridge_dists.size > 0 else np.nan

    return result


# ---------------------------------------------------------------------------
# Per-mitochondrion statistics
# ---------------------------------------------------------------------------

def _single_mito_row(
    label: int,
    bbox: tuple,
    mito_crop: np.ndarray,
    crista_crop: np.ndarray,
    membrane_crop: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    sampling: np.ndarray,
    voxel_vol: float,
    vol_shape: tuple,
    border_radius: int,
    inner_n_jobs: int = 1,
) -> Dict[str, float]:
    """Compute the statistics row for a single mitochondrion instance.

    Factored out of :func:`compute_mito_crista_statistics` so the per-mito work (which is
    independent between instances) can be parallelised. Takes the bbox-cropped label arrays
    (``mito_crop`` = label array cropped to ``bbox``; ``crista_crop``/``membrane_crop`` the
    matching binary crops), so it can be dispatched to a process worker without pickling the
    whole volume. ``inner_n_jobs`` is forwarded to the (parallelisable) junction-distance stage.
    """
    ndim = mito_crop.ndim
    touches_border = any(
        bbox[i] < border_radius or bbox[i + ndim] > vol_shape[i] - border_radius
        for i in range(ndim)
    )

    mito_local = mito_crop == label
    crista_local = crista_crop & mito_local
    membrane_local = membrane_crop & mito_local

    mito_vol = float(mito_local.sum()) * voxel_vol
    crista_vol = float(crista_local.sum()) * voxel_vol

    has_crista = crista_local.any()
    has_membrane = membrane_local.any()
    mito_surface = _surface_area(mito_local, sampling)

    if has_crista and has_membrane:
        contact_labels_local, contact_summary = detect_contact_sites(crista_local, membrane_local, voxel_size)
        # One distance transform of the membrane, reused for proximity (distances) and for
        # junction snapping (indices) — return_indices gives both at ~no extra cost.
        membrane_distance, membrane_indices = distance_transform_edt(
            ~membrane_local, sampling=sampling.tolist(), return_indices=True
        )
        _, proximity = compute_crista_proximity(
            crista_local, membrane_local, voxel_size, membrane_distance=membrane_distance
        )
        _, junction_dist = compute_junction_distances(
            contact_labels_local, membrane_local, voxel_size,
            surface_area_nm2=mito_surface, membrane_indices=membrane_indices, n_jobs=inner_n_jobs,
        )
        # Free the (potentially large) distance-transform arrays before the orientation
        # computation so they don't co-reside with its structure-tensor components.
        del membrane_distance, membrane_indices, contact_labels_local
    else:
        contact_summary = {"contact_voxel_count": 0, "crista_junction_count": 0, "contact_volume_nm3": 0.0}
        proximity = {"median_nm": np.nan}
        junction_dist = dict(_JUNCTION_DISTANCE_NAN)

    if has_crista:
        _, _, anisotropy = compute_crista_orientation(
            crista_local, voxel_size, need_eigenvectors=False, n_jobs=inner_n_jobs
        )
        crista_orientation_anisotropy = float(np.mean(anisotropy[crista_local]))
        morph = compute_crista_morphology(crista_local, voxel_size)
    else:
        crista_orientation_anisotropy = np.nan
        morph = {"cristae_surface_area_nm2": np.nan, "avg_thickness_nm": np.nan}

    crista_surface = morph.get("cristae_surface_area_nm2", np.nan)
    if mito_surface and mito_surface > 0 and np.isfinite(crista_surface):
        crista_to_mito_surface_ratio = crista_surface / mito_surface
    else:
        crista_to_mito_surface_ratio = np.nan

    return {
        "mito_label_id": int(label),
        "mito_touches_border": touches_border,
        "mito_volume_nm3": mito_vol,
        "crista_volume_nm3": crista_vol,
        "crista_fraction": crista_vol / mito_vol if mito_vol > 0 else np.nan,
        "contact_voxel_count": contact_summary["contact_voxel_count"],
        "crista_junction_count": contact_summary["crista_junction_count"],
        "contact_volume_nm3": contact_summary["contact_volume_nm3"],
        "avg_crista_to_membrane_nm": proximity["median_nm"],
        "mean_nn_junction_distance_nm": junction_dist["mean_nn_junction_distance_nm"],
        "median_nn_junction_distance_nm": junction_dist["median_nn_junction_distance_nm"],
        "junction_clustering_index": junction_dist["junction_clustering_index"],
        "crista_orientation_anisotropy": crista_orientation_anisotropy,
        "cristae_surface_area_nm2": crista_surface,
        "mito_surface_area_nm2": mito_surface,
        "crista_to_mito_surface_ratio": crista_to_mito_surface_ratio,
        "avg_thickness_nm": morph.get("avg_thickness_nm", np.nan),
    }


def compute_mito_crista_statistics(
    crista_mask: np.ndarray,
    mito_segmentation: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    membrane_mask: Optional[np.ndarray] = None,
    membrane_thickness_nm: float = 8.0,
    border_gap_nm: Optional[float] = None,
    n_jobs: int = 1,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Compute all crista metrics organised by mitochondrial instance.

    Args:
        crista_mask: Binary crista segmentation (global volume).
        mito_segmentation: Instance label array (background = 0).
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        membrane_mask: Precomputed membrane mask; recomputed if None.
        membrane_thickness_nm: Membrane shell thickness used if membrane_mask is None.
        border_gap_nm: Border suppression distance passed to approximate_membrane;
            defaults to membrane_thickness_nm when None.
        n_jobs: Number of workers for processing mitochondria in parallel (they are
            independent). 1 (default) runs serially; other values use a joblib thread pool
            (-1 = all cores). Results are identical regardless of n_jobs.
        verbose: If True, show a terminal tqdm progress bar over mitochondria.
        progress_callback: Optional callable invoked once per completed mitochondrion with
            (completed_count, total_count) — e.g. to drive a napari progress bar. It is
            always called from the calling thread (the joblib results generator is consumed
            here), so GUI updates from it need no cross-thread marshaling.

    Returns:
        DataFrame with one row per mito instance:
        label | mito_volume_nm3 | crista_volume_nm3 | crista_fraction |
        contact_voxel_count | crista_junction_count | contact_volume_nm3 |
        avg_crista_to_membrane_nm | mean_nn_junction_distance_nm | median_nn_junction_distance_nm |
        junction_clustering_index | crista_orientation_anisotropy | cristae_surface_area_nm2 |
        mito_surface_area_nm2 | crista_to_mito_surface_ratio | avg_thickness_nm.
        cristae_surface_area_nm2 is the crista surface area; crista_to_mito_surface_ratio is
        crista surface / mitochondrial outer-membrane surface (can exceed 1 for folded cristae).
        The *_nn_junction_distance_nm columns are geodesic nearest-neighbour distances between
        crista-membrane junctions along the membrane; junction_clustering_index is a Clark-Evans
        index (< 1 clustered, ~ 1 random, > 1 dispersed).
    """
    if membrane_mask is None:
        membrane_mask = approximate_membrane(
            mito_segmentation, voxel_size, membrane_thickness_nm, border_gap_nm, n_jobs=n_jobs
        )

    ndim = mito_segmentation.ndim
    sampling = _to_sampling(voxel_size, ndim)
    voxel_vol = float(np.prod(sampling))
    crista_binary = crista_mask.astype(bool)
    vol_shape = mito_segmentation.shape
    effective_gap_nm = border_gap_nm if border_gap_nm is not None else membrane_thickness_nm
    border_radius = _voxel_radius(effective_gap_nm, voxel_size, ndim)

    # Pre-crop each mito to its bounding box (basic slicing → views, so this is memory-free;
    # for the process path only the bbox region gets pickled, not the whole volume).
    tasks = []
    for prop in regionprops(mito_segmentation):
        bbox = prop.bbox
        slices = tuple(slice(bbox[i], bbox[i + ndim]) for i in range(ndim))
        tasks.append((
            int(prop.label), bbox,
            mito_segmentation[slices], crista_binary[slices], membrane_mask[slices],
        ))
    total = len(tasks)

    def _run(task, inner_n_jobs):
        label, bbox, mito_crop, crista_crop, membrane_crop = task
        return _single_mito_row(
            label, bbox, mito_crop, crista_crop, membrane_crop,
            voxel_size, sampling, voxel_vol, vol_shape, border_radius, inner_n_jobs=inner_n_jobs,
        )

    n_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    # Adaptive: many mitochondria → parallelize ACROSS them (processes, so the GIL-bound stages
    # actually scale), BLAS capped per worker, inner stages serial. Few mitochondria (e.g. one
    # dominant mito) → run them serially but give each mito's junction-distance stage all the
    # cores and let BLAS multithread the orientation. Exactly one level of parallelism is ever
    # active, so there is no process/thread oversubscription.
    across = n_workers > 1 and total >= n_workers

    rows = []

    def _consume(results):
        for i, row in enumerate(
            tqdm(results, total=total, desc="Cristae analysis", disable=not verbose), start=1
        ):
            rows.append(row)
            if progress_callback is not None:
                progress_callback(i, total)

    if across:
        from joblib import Parallel, delayed, parallel_config
        # loky processes bypass the GIL; inner_max_num_threads=1 stops each worker's BLAS from
        # oversubscribing against the pool.
        with parallel_config(backend="loky", inner_max_num_threads=1):
            _consume(
                Parallel(n_jobs=n_workers, return_as="generator_unordered")(
                    delayed(_run)(task, 1) for task in tasks
                )
            )
    else:
        _consume(_run(task, n_workers) for task in tasks)

    # Stable, n_jobs-independent ordering.
    rows.sort(key=lambda row: row["mito_label_id"])
    return pd.DataFrame(rows)
