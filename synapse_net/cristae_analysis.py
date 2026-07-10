import os
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, center_of_mass, distance_transform_edt, gaussian_filter
from scipy.ndimage import label as ndimage_label
from skimage.measure import marching_cubes, mesh_surface_area, regionprops
from skimage.morphology import disk, local_maxima
from tqdm import tqdm

try:  # Optional: surface-mesh geodesic backend (postdates bioimage-cpp 0.5.0).
    from bioimage_cpp.distance import geodesic_distances_mesh
except Exception:  # pragma: no cover - depends on installed bioimage-cpp version
    geodesic_distances_mesh = None


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


def _surface_mesh(mask: np.ndarray, sampling: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Closed triangle-mesh surface of a binary mask via marching cubes.

    The mask is padded by one background voxel so objects touching the array edge
    (e.g. an instance cropped to its bounding box) yield a closed surface rather than an
    open mesh with the edge-touching faces missing. Because of that pad, a mask voxel at
    array index ``(z, y, x)`` maps to physical mesh coordinates ``(index + 1) * sampling`` —
    callers that snap voxel coordinates onto the mesh must apply the same ``+1`` offset.

    Args:
        mask: Binary segmentation.
        sampling: Voxel size per axis (nm), in array (z, y, x) order.

    Returns:
        (vertices, faces) with vertices in nm (padded frame), or None if the mask is empty.
    """
    binary = mask.astype(bool)
    if not binary.any():
        return None
    padded = np.pad(binary.astype(np.float32), 1)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=tuple(float(s) for s in sampling))
    return verts, faces


def _surface_area(mask: np.ndarray, sampling: np.ndarray) -> float:
    """Closed-surface area (nm^2) of a binary mask via marching cubes.

    Args:
        mask: Binary segmentation.
        sampling: Voxel size per axis (nm), in array (z, y, x) order.

    Returns:
        Surface area in nm^2, or NaN if the mask is empty.
    """
    mesh = _surface_mesh(mask, sampling)
    if mesh is None:
        return np.nan
    return float(mesh_surface_area(*mesh))


def _medial_axis_thickness_nm(mask: np.ndarray, sampling: np.ndarray) -> float:
    """Local thickness (nm) of a mask via the distance transform, with no mesh generation.

    The medial axis is approximated by the local maxima of the interior EDT; the thickness is
    ``2 × mean(EDT)`` there (the EDT at the medial axis is the half-thickness). This is the same
    estimator used by :func:`compute_crista_morphology`'s ``medial_axis`` branch, factored out so
    the distance-based (``method="fast"``) surface-area estimates can reuse it.

    Args:
        mask: Binary segmentation.
        sampling: Voxel size per axis (nm), in array (z, y, x) order.

    Returns:
        Mean local thickness in nm, or NaN if the mask is empty.
    """
    binary = mask.astype(bool)
    if not binary.any():
        return np.nan
    dist = distance_transform_edt(binary, sampling=tuple(float(s) for s in sampling))
    ridges = local_maxima(dist) & binary
    ridge_dists = dist[ridges]
    return float(2.0 * np.mean(ridge_dists)) if ridge_dists.size > 0 else np.nan


def _available_memory_bytes() -> int:
    """Best-effort available RAM in bytes (used to keep parallel working sets from OOMing)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except Exception:
            return 4 * 1024 ** 3  # conservative fallback


def _bounded_workers(n_jobs: int, per_worker_bytes: int, fraction: float = 0.5) -> int:
    """Resolve n_jobs to a worker count whose combined working set fits in memory.

    n_jobs: 1 = serial, -1 = all cores, else that many. The result is additionally capped so
    ``workers * per_worker_bytes <= fraction * available_RAM`` (at least 1).
    """
    workers = os.cpu_count() if n_jobs == -1 else max(1, int(n_jobs))
    if per_worker_bytes > 0:
        budget = int(_available_memory_bytes() * fraction)
        workers = min(workers, max(1, budget // int(per_worker_bytes)))
    return int(max(1, workers))


# ---------------------------------------------------------------------------
# Membrane approximation
# ---------------------------------------------------------------------------

def approximate_membrane(
    mito_segmentation: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    membrane_thickness_nm: float = 8.0,
    border_gap_nm: Optional[float] = None,
    n_jobs: int = 1,
    membrane_mode: str = "slice_2d",
) -> np.ndarray:
    """Approximate the mitochondrial membrane as the outer shell of the segmentation.

    Two shell constructions are available via ``membrane_mode``:

    - ``"slice_2d"`` (default): erode each Z-slice **independently in 2D** by an XY disk of radius
      ``round(thickness / xy_voxel)`` and keep ``slice & ~eroded``. A mitochondrion that changes shape
      rapidly along Z does not bleed into neighbouring slices, and the per-slice erosions are
      parallelised over Z (``n_jobs``). The shell has no Z-caps and can fragment into disconnected
      pieces across slices. (2D inputs get a single 2D erosion.)
    - ``"shell_3d"``: a full 3D morphological erosion, ``mito & ~erode3d(mito, k)`` with
      ``k = round(thickness / mean_voxel)`` iterations of a 3×3×3 structuring element, per instance on
      its padded bounding box. A single **connected** shell including the Z-caps (no per-slice
      fragmentation), at a higher cost; thickness acts in all axes.

    The complementary interior ``mito & ~membrane`` is the eroded-mito "lumen"; its surface is the
    single-wall mesh used by the mesh geodesic backend, so the mesh follows the chosen mode.

    Membrane voxels within border_gap_nm of any volume face are suppressed to avoid treating clipped
    mito edges as membrane.

    Args:
        mito_segmentation: Instance label array (background = 0).
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        membrane_thickness_nm: Thickness of the membrane shell in nm.
        border_gap_nm: Distance from each volume face within which membrane voxels are
            suppressed. Defaults to membrane_thickness_nm when None.
        n_jobs: Workers for the per-Z-slice erosion (``"slice_2d"`` only): 1 = serial, -1 = all cores.
            Results are identical regardless of n_jobs.
        membrane_mode: ``"slice_2d"`` (default, per-slice 2D, z-parallel) or ``"shell_3d"``
            (connected 3D shell).

    Returns:
        membrane_mask: Binary mask of the mitochondrial membrane (outer shell),
            with border-adjacent voxels zeroed out.
    """
    if membrane_mode not in ("slice_2d", "shell_3d"):
        raise ValueError(f"membrane_mode must be 'slice_2d' or 'shell_3d', got {membrane_mode!r}")
    ndim = mito_segmentation.ndim
    mito_binary = mito_segmentation > 0

    if membrane_mode == "shell_3d":
        # Full 3D erosion → single connected shell (incl. Z-caps), per instance on its padded bbox.
        sampling = _to_sampling(voxel_size, ndim)
        k = max(1, int(round(float(membrane_thickness_nm) / float(np.mean(sampling)))))
        struct = np.ones((3,) * ndim, dtype=bool)
        membrane_mask = np.zeros(mito_segmentation.shape, dtype=bool)
        for prop in regionprops(mito_segmentation):
            bbox = prop.bbox
            sl = tuple(
                slice(max(0, bbox[i] - k), min(mito_segmentation.shape[i], bbox[i + ndim] + k))
                for i in range(ndim)
            )
            sub = mito_binary[sl]  # merged mito (all instances in the crop) → shared boundaries
            eroded = binary_erosion(sub, structure=struct, iterations=k, border_value=1)
            cur = mito_segmentation[sl] == prop.label
            membrane_mask[sl] |= cur & ~eroded
    elif ndim == 3:
        # Per-Z-slice 2D erosion (no z-bleed), parallelised over Z. Only the mito XY bounding box
        # needs eroding; a `membrane_radius` margin makes the cropped erosion (border_value=1)
        # identical to eroding the full slice, and empty slices are skipped.
        membrane_radius = _voxel_radius_xy(membrane_thickness_nm, voxel_size)
        struct = disk(membrane_radius)
        membrane_mask = np.zeros_like(mito_binary)
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

    # Each concurrent component holds a product + its smoothed output (~2 float32 arrays); cap the
    # thread count by available memory so the smoothing peak can't OOM on a large mitochondrion.
    workers = _bounded_workers(n_jobs, per_worker_bytes=crista_mask.size * 4 * 2)
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


def _scale_voxel_size(
    voxel_size: Union[float, Dict[str, float]], factor: float
) -> Union[float, Dict[str, float]]:
    """Multiply a voxel size (scalar or z/y/x dict) by ``factor``, preserving its type."""
    if isinstance(voxel_size, dict):
        return {ax: voxel_size[ax] * factor for ax in voxel_size}
    return float(voxel_size) * factor


def _downsampled_orientation_anisotropy(
    crista_mask: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    factor: int = 2,
    n_jobs: int = 1,
) -> float:
    """Mean crista orientation anisotropy computed on a downsampled crop (fast, approximate).

    The crista mask is block-mean downsampled by ``factor`` per axis to an anti-aliased float field,
    and :func:`compute_crista_orientation` is run on that coarser grid at the correspondingly scaled
    voxel size (so the physical structure-tensor neighbourhood is unchanged). This is ~``factor**ndim``
    times cheaper than the full-resolution structure tensor — the dominant cost of the analysis.

    Because thin cristae (only a few voxels across) lose structure when downsampled, the returned
    anisotropy is a *relative* indicator only: it preserves the ordering between mitochondria but is
    systematically smaller in magnitude than the full-resolution value (measured 0.24–0.75× on real
    data). It is therefore NOT comparable to the ``method="exact"`` anisotropy.

    Args:
        crista_mask: Binary crista segmentation.
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        factor: Integer downsampling factor per axis.
        n_jobs: Threads forwarded to :func:`compute_crista_orientation`.

    Returns:
        Mean anisotropy over the downsampled crista region, or NaN if it vanishes when downsampled.
    """
    from skimage.transform import downscale_local_mean

    ndim = crista_mask.ndim
    field = downscale_local_mean(crista_mask.astype(np.float32), (factor,) * ndim)
    region = field > 0.5
    if not region.any():
        return np.nan
    _, _, anisotropy = compute_crista_orientation(
        field, _scale_voxel_size(voxel_size, factor), need_eigenvectors=False, n_jobs=n_jobs
    )
    return float(np.mean(anisotropy[region]))


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


def _junction_matrix_mesh(
    centroids: np.ndarray,
    sampling: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    n_jobs: int = 1,
) -> Optional[np.ndarray]:
    """Pairwise junction geodesic distances (nm) along a triangle-mesh surface.

    Snaps each junction centroid to its nearest mesh vertex and calls
    ``bioimage_cpp.distance.geodesic_distances_mesh``. The mesh comes from marching cubes on the
    padded eroded-mito (lumen) surface (see :func:`_surface_mesh`), so voxel centroids are mapped to
    the mesh frame with the matching ``+1`` pad offset. Returns None (junction distances become NaN)
    when the bioimage-cpp geodesic API is unavailable or the mesh is empty.

    Args:
        centroids: (n, ndim) junction centroids in voxel coordinates.
        sampling: Voxel size per axis (nm), array (z, y, x) order.
        vertices: Mesh vertices (n_vertices, 3) in nm (padded frame).
        faces: Mesh triangle indices (n_faces, 3).
        n_jobs: 1 = serial, -1 = all cores (forwarded to the C++ solver's thread count).

    Returns:
        (n, n) geodesic distance matrix in nm (0 diagonal, NaN for disconnected pairs), or None.
    """
    if geodesic_distances_mesh is None:
        return None
    verts = np.ascontiguousarray(vertices, dtype=np.float64)
    tris = np.ascontiguousarray(faces, dtype=np.int64)
    if verts.shape[0] == 0 or tris.shape[0] == 0:
        return None
    from scipy.spatial import cKDTree

    # Mask voxel (z, y, x) sits at (index + 1) * sampling in the padded mesh frame (_surface_mesh).
    points = (np.asarray(centroids, dtype=float) + 1.0) * sampling
    _, vertex_ids = cKDTree(verts).query(points)
    vertex_ids = np.atleast_1d(np.asarray(vertex_ids, dtype=np.int64))
    n_threads = 0 if n_jobs in (-1, 0) else max(1, int(n_jobs))  # bic: 0 = hardware_concurrency
    dm = np.asarray(
        geodesic_distances_mesh(verts, tris, vertex_ids, number_of_threads=n_threads), dtype=float
    )
    dm[~np.isfinite(dm)] = np.nan  # disconnected components come back as +inf
    np.fill_diagonal(dm, 0.0)
    return dm


def compute_junction_distances(
    contact_labels: np.ndarray,
    membrane_mask: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    surface_area_nm2: Optional[float] = None,
    n_jobs: int = 1,
    mesh_vertices: Optional[np.ndarray] = None,
    mesh_faces: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Geodesic distances between crista-membrane junctions along the eroded-mito surface mesh.

    Each junction (a connected component in ``contact_labels``) is reduced to its centroid, snapped to
    the nearest vertex of a triangle mesh, and pairwise surface geodesics are computed with
    ``bioimage_cpp.distance.geodesic_distances_mesh``. The mesh is the **eroded-mito (lumen) surface**
    passed in as ``mesh_vertices``/``mesh_faces`` by :func:`_single_mito_row` (a clean, single-wall
    surface at the membrane's inner edge); if none is supplied a mesh is built from ``membrane_mask``
    as a convenience. When the bioimage-cpp geodesic API is unavailable or the mesh is empty, the
    junction distances are NaN (:func:`compute_mito_crista_statistics` emits a single warning).

    A Clark-Evans nearest-neighbour index summarises whether the junctions are clustered.

    Args:
        contact_labels: Integer junction label array (0 = background, 1..n = junctions),
            e.g. the first return value of :func:`detect_contact_sites`.
        membrane_mask: Binary mitochondrial membrane mask the junctions sit on (used to build a
            fallback mesh when ``mesh_vertices``/``mesh_faces`` are not supplied).
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        surface_area_nm2: Membrane/mito surface area used as the reference area for the
            Clark-Evans expectation. If None or non-positive, the clustering index is NaN.
        n_jobs: 1 = serial, -1 = all cores (forwarded to the mesh solver's thread count).
        mesh_vertices: Optional (n_vertices, 3) mesh vertices in nm (padded frame; see
            :func:`_surface_mesh`) — the eroded-mito (lumen) surface. If omitted, a mesh is built
            from ``membrane_mask``.
        mesh_faces: Optional (n_faces, 3) triangle indices matching ``mesh_vertices``.

    Returns:
        distance_matrix: (n, n) geodesic distances in nm between junctions; the diagonal is
            0 and unreachable pairs (disconnected fragments) are NaN. Empty for fewer than two
            junctions.
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

    # Junction centroids in one labelled reduction (uniform weights → geometric centroid).
    centroids = np.atleast_2d(
        np.asarray(center_of_mass(contact_labels > 0, labels=contact_labels, index=labels), dtype=float)
    )

    # Surface geodesic along the supplied eroded-mito mesh (or a mesh built from the membrane).
    distance_matrix = None
    if geodesic_distances_mesh is not None:
        if mesh_vertices is not None and mesh_faces is not None and len(mesh_faces) > 0:
            mesh = (mesh_vertices, mesh_faces)
        else:
            mesh = _surface_mesh(membrane, sampling)
        if mesh is not None:
            distance_matrix = _junction_matrix_mesh(centroids, sampling, mesh[0], mesh[1], n_jobs)

    if distance_matrix is None:
        # bioimage-cpp geodesic API unavailable or no usable mesh → distances are NaN (no fallback).
        summary = dict(_JUNCTION_DISTANCE_NAN)
        summary["junction_count"] = n
        return np.full((n, n), np.nan, dtype=float), summary

    # Nearest-neighbour distance per junction (nearest reachable other junction). Vectorised:
    # exclude self (diagonal) and unreachable pairs (NaN) by setting them to +inf, take the row
    # minimum, and drop rows with no reachable neighbour (min stays +inf).
    dm = distance_matrix.copy()
    np.fill_diagonal(dm, np.inf)
    dm[~np.isfinite(dm)] = np.inf
    row_min = dm.min(axis=1)
    nn_distances = row_min[np.isfinite(row_min)]

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
    result: Dict[str, float] = {}

    if method in ("area", "both"):
        result["cristae_surface_area_nm2"] = _surface_area(crista_mask, sampling)

    if method in ("medial_axis", "both"):
        # Local maxima of the EDT form the medial axis; 2 × distance there = local thickness.
        result["avg_thickness_nm"] = _medial_axis_thickness_nm(crista_mask, sampling)

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
    method: str = "fast",
    inner_n_jobs: int = 1,
) -> Dict[str, float]:
    """Compute the statistics row for a single mitochondrion instance.

    Factored out of :func:`compute_mito_crista_statistics` so the per-mito work (which is
    independent between instances) can be parallelised. Takes the bbox-cropped label arrays
    (``mito_crop`` = label array cropped to ``bbox``; ``crista_crop``/``membrane_crop`` the
    matching binary crops), so it can be dispatched to a process worker without pickling the
    whole volume. ``inner_n_jobs`` is forwarded to the (parallelisable) junction-distance stage.

    ``method`` controls only the crista orientation anisotropy — every other metric (marching-cubes
    surface areas, geodesic junction distances, EDT proximity/thickness) is computed identically for
    all modes. ``"exact"`` computes the anisotropy from the full-resolution structure tensor;
    ``"fast"`` (the default) computes it on a 2× downsampled crop (~8× cheaper — the structure tensor
    is the dominant cost), which is a *relative* indicator only and not comparable to the exact value;
    ``"skip"`` does not compute it at all (left NaN), which is the fastest.
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
    # Mito outer surface area via marching cubes (this metric correctly uses the outer OM surface).
    mito_surface = _surface_area(mito_local, sampling)

    if has_crista and has_membrane:
        contact_labels_local, contact_summary = detect_contact_sites(crista_local, membrane_local, voxel_size)
        # Distance transform of the membrane, for the crista→membrane proximity metric.
        membrane_distance = distance_transform_edt(~membrane_local, sampling=sampling.tolist())
        _, proximity = compute_crista_proximity(
            crista_local, membrane_local, voxel_size, membrane_distance=membrane_distance
        )
        # Measure junction geodesics on the eroded-mito (lumen) surface — the mito interior inside the
        # membrane band (mito & ~membrane): a clean, single-wall surface at the membrane's inner edge.
        lumen_mesh = _surface_mesh(mito_local & ~membrane_local, sampling)
        mesh_verts, mesh_faces = lumen_mesh if lumen_mesh is not None else (None, None)
        _, junction_dist = compute_junction_distances(
            contact_labels_local, membrane_local, voxel_size,
            surface_area_nm2=mito_surface, n_jobs=inner_n_jobs,
            mesh_vertices=mesh_verts, mesh_faces=mesh_faces,
        )
        # Free the (potentially large) distance-transform array before the orientation computation.
        del membrane_distance, contact_labels_local
    else:
        contact_summary = {"contact_voxel_count": 0, "crista_junction_count": 0, "contact_volume_nm3": 0.0}
        proximity = {"median_nm": np.nan}
        junction_dist = dict(_JUNCTION_DISTANCE_NAN)

    if has_crista:
        morph = compute_crista_morphology(crista_local, voxel_size)
        crista_surface = morph.get("cristae_surface_area_nm2", np.nan)
        avg_thickness_nm = morph.get("avg_thickness_nm", np.nan)
        # Orientation is the dominant cost: skip it (NaN), a downsampled relative-only approximation
        # (fast), or the full structure tensor (exact).
        if method == "skip":
            crista_orientation_anisotropy = np.nan
        elif method == "fast":
            crista_orientation_anisotropy = _downsampled_orientation_anisotropy(
                crista_local, voxel_size, factor=2, n_jobs=inner_n_jobs
            )
        else:  # exact
            _, _, anisotropy = compute_crista_orientation(
                crista_local, voxel_size, need_eigenvectors=False, n_jobs=inner_n_jobs
            )
            crista_orientation_anisotropy = float(np.mean(anisotropy[crista_local]))
    else:
        crista_orientation_anisotropy = np.nan
        crista_surface = np.nan
        avg_thickness_nm = np.nan

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
        "avg_thickness_nm": avg_thickness_nm,
    }


def compute_mito_crista_statistics(
    crista_mask: np.ndarray,
    mito_segmentation: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    membrane_mask: Optional[np.ndarray] = None,
    membrane_thickness_nm: float = 8.0,
    border_gap_nm: Optional[float] = None,
    method: str = "fast",
    n_jobs: int = 1,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    membrane_mode: str = "slice_2d",
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
        method: How the crista orientation anisotropy is computed — this is the ONLY metric that
            differs between modes; surface areas (marching cubes), junction distances (geodesic
            along the membrane) and thickness/proximity (EDT) are computed identically for all of
            them. ``"fast"`` (default) computes the anisotropy on a 2× downsampled crista crop
            (~8× cheaper — the structure tensor is by far the dominant cost); the resulting value is
            a *relative* indicator that preserves the ordering between mitochondria but is
            systematically different in magnitude from the full-resolution value and is NOT
            comparable to ``method="exact"``. ``"exact"`` computes the anisotropy from the
            full-resolution structure tensor (use it when the magnitude must be precise). ``"skip"``
            does not compute orientation at all (``crista_orientation_anisotropy`` is NaN) and is the
            fastest — use it when only the other metrics are needed.
        n_jobs: Number of workers for processing mitochondria in parallel (they are
            independent). 1 (default) runs serially; other values use a joblib thread pool
            (-1 = all cores). Results are identical regardless of n_jobs.
        verbose: If True, show a terminal tqdm progress bar over mitochondria.
        progress_callback: Optional callable invoked once per completed mitochondrion with
            (completed_count, total_count) — e.g. to drive a napari progress bar. It is
            always called from the calling thread (the joblib results generator is consumed
            here), so GUI updates from it need no cross-thread marshaling.
        membrane_mode: How the membrane shell is built when ``membrane_mask`` is None —
            ``"slice_2d"`` (default, per-Z-slice 2D erosion, z-parallel) or ``"shell_3d"`` (connected
            3D shell). See :func:`approximate_membrane`.

    The junction nearest-neighbour distances are geodesics along the eroded-mito surface mesh
    (``bioimage_cpp.distance.geodesic_distances_mesh``, needs bioimage-cpp>=0.6.0); when that API is
    unavailable those columns are NaN (a single warning is emitted).

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
        index (< 1 clustered, ~ 1 random, > 1 dispersed). ``crista_orientation_anisotropy`` is
        computed at full resolution for ``method="exact"``, on a downsampled crop (relative-only,
        not comparable) for ``method="fast"``, and left NaN for ``method="skip"``.
    """
    if method not in ("fast", "exact", "skip"):
        raise ValueError(f"method must be 'fast', 'exact', or 'skip', got {method!r}")
    # Warn once (not per mito) if the mesh geodesic API is unavailable → junction distances are NaN.
    if geodesic_distances_mesh is None:
        warnings.warn(
            "The bioimage-cpp geodesic API is unavailable (needs bioimage-cpp>=0.6.0); "
            "junction nearest-neighbour distance columns will be NaN.",
            RuntimeWarning, stacklevel=2,
        )
    if membrane_mask is None:
        membrane_mask = approximate_membrane(
            mito_segmentation, voxel_size, membrane_thickness_nm, border_gap_nm,
            n_jobs=n_jobs, membrane_mode=membrane_mode,
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
            voxel_size, sampling, voxel_vol, vol_shape, border_radius,
            method=method, inner_n_jobs=inner_n_jobs,
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
        # Cap concurrent workers so their combined per-mito working set (graph + tensor
        # components + label crops, ~40 bytes/voxel of the largest mito) fits in RAM.
        max_voxels = max(int(task[2].size) for task in tasks)
        across_workers = _bounded_workers(n_jobs, per_worker_bytes=max_voxels * 40)
        # loky processes bypass the GIL; inner_max_num_threads=1 stops each worker's BLAS from
        # oversubscribing against the pool.
        with parallel_config(backend="loky", inner_max_num_threads=1):
            _consume(
                Parallel(n_jobs=across_workers, return_as="generator_unordered")(
                    delayed(_run)(task, 1) for task in tasks
                )
            )
    else:
        _consume(_run(task, n_workers) for task in tasks)

    # Stable, n_jobs-independent ordering.
    rows.sort(key=lambda row: row["mito_label_id"])
    return pd.DataFrame(rows)
