import multiprocessing as mp
import os
from concurrent import futures
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, center_of_mass
from scipy.ndimage import label as ndimage_label
from skimage.measure import mesh_surface_area, regionprops
from skimage.morphology import disk, local_maxima
from tqdm import tqdm

from bioimage_cpp.distance import distance_transform, geodesic_distances_mesh
from bioimage_cpp.filters import structure_tensor_eigenvalues
from bioimage_cpp.mesh import marching_cubes


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


def _gap_radius(
    voxel_size: Union[float, Dict[str, float]],
    membrane_thickness_nm: float,
    border_gap_nm: Optional[float],
    ndim: int,
) -> int:
    """Border-zone / mesh-trim radius in voxels; ``border_gap_nm`` defaults to ``membrane_thickness_nm``."""
    gap_nm = border_gap_nm if border_gap_nm is not None else membrane_thickness_nm
    return _voxel_radius(gap_nm, voxel_size, ndim)


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


def _surface_mesh(
    mask: np.ndarray,
    sampling: np.ndarray,
    closed_faces: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Triangle-mesh surface of a binary mask via marching cubes.

    Each side of the array is padded by one background voxel before meshing so that objects touching
    the array edge yield a closed surface there. ``marching_cubes`` only emits triangles for cube
    cells that exist inside the array, so leaving a side *unpadded* omits that boundary face — an open
    mesh at that face — while interior surfaces still close. ``closed_faces`` chooses this per side:
    pad+close a face where there is genuine background beyond it, leave open a face where the object
    is clipped by the volume boundary (membrane presence unknown there).

    Vertices are returned in the mask's own **unpadded** index frame (nm): a mask voxel at array index
    ``(z, y, x)`` maps to physical coordinates ``index * sampling`` regardless of the padding, so
    callers snapping voxel coordinates onto the mesh use ``index * sampling`` with no offset.

    Args:
        mask: Binary segmentation.
        sampling: Voxel size per axis (nm), in array (z, y, x) order.
        closed_faces: Optional ``(ndim, 2)`` boolean array; ``closed_faces[a, s]`` True closes
            (pads) side ``s`` (0 = low, 1 = high) of axis ``a``, False leaves it open. Defaults to
            all True (fully closed watertight surface).

    Returns:
        (vertices, faces) with vertices in nm (unpadded mask frame), or None if the mask is empty.
    """
    binary = mask.astype(bool)
    if not binary.any():
        return None
    if closed_faces is None:
        closed_faces = np.ones((binary.ndim, 2), dtype=bool)
    else:
        closed_faces = np.asarray(closed_faces, dtype=bool)
    pad_width = [(int(closed_faces[a, 0]), int(closed_faces[a, 1])) for a in range(binary.ndim)]
    padded = np.pad(binary.astype(np.float32), pad_width)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=tuple(float(s) for s in sampling))
    pad_before = np.array([pw[0] for pw in pad_width], dtype=float)
    verts = verts - pad_before * np.asarray(sampling, dtype=float)
    return verts, faces


def _surface_area(
    mask: np.ndarray,
    sampling: np.ndarray,
    closed_faces: Optional[np.ndarray] = None,
) -> float:
    """Surface area (nm^2) of a binary mask via marching cubes.

    Args:
        mask: Binary segmentation.
        sampling: Voxel size per axis (nm), in array (z, y, x) order.
        closed_faces: Optional per-side padding spec forwarded to :func:`_surface_mesh` — leave a
            clipped volume-boundary face open so its fabricated cap is not counted as surface area.
            Defaults to a fully closed (watertight) surface.

    Returns:
        Surface area in nm^2, or NaN if the mask is empty.
    """
    mesh = _surface_mesh(mask, sampling, closed_faces=closed_faces)
    if mesh is None:
        return np.nan
    return float(mesh_surface_area(*mesh))


def _open_trimmed_mesh(
    mask: np.ndarray,
    sampling: np.ndarray,
    gap_radius: int,
    boundary: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Surface mesh of ``mask`` trimmed to the certain region and left OPEN at volume-boundary faces.

    Near a clipped volume face the segmentation is cut off and membrane presence is unknown, so the
    surface must neither flare into that region nor be capped there — a cap is a fabricated flat disk
    that lets geodesics shortcut straight across instead of wrapping around the tube wall. For each
    ``(axis, side)`` flagged True in ``boundary`` (an ``(ndim, 2)`` bool of volume-boundary faces),
    ``gap_radius`` voxels are cropped off that face so the trim plane becomes an array boundary, and
    that face is then left open (marching cubes omits it). Interior faces stay closed.

    Args:
        mask: Binary segmentation.
        sampling: Voxel size per axis (nm), in array (z, y, x) order.
        gap_radius: Border-zone width in voxels cropped off each flagged face (matches the membrane's
            border-gap trim).
        boundary: ``(ndim, 2)`` bool; True where the face is at the volume boundary (crop + open).

    Returns:
        (vertices, faces) with vertices in nm in the original (uncropped) ``mask`` index frame, or None
        if the trimmed mask is empty.
    """
    ndim = mask.ndim
    boundary = np.asarray(boundary, dtype=bool)
    lo = [gap_radius if boundary[a, 0] else 0 for a in range(ndim)]
    hi = [mask.shape[a] - (gap_radius if boundary[a, 1] else 0) for a in range(ndim)]
    mesh = _surface_mesh(
        mask[tuple(slice(lo[a], hi[a]) for a in range(ndim))], sampling, closed_faces=~boundary
    )
    if mesh is None:
        return None
    verts, faces = mesh
    verts = verts + np.array(lo, dtype=float) * np.asarray(sampling, dtype=float)
    return verts, faces


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
    dist = distance_transform(binary, sampling=tuple(float(s) for s in sampling), number_of_threads=1)
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
            return 4 * 1024 ** 3


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
    return_lumen: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Approximate the mitochondrial membrane as the outer shell of the segmentation.

    Two shell constructions are available via ``membrane_mode``:

    - ``"slice_2d"`` (default): erode each Z-slice **independently in 2D** by an XY disk of radius
      ``round(thickness / xy_voxel)`` and keep ``slice & ~eroded``. A mitochondrion that changes shape
      rapidly along Z does not bleed into neighbouring slices, and the per-slice erosions are
      parallelised over Z (``n_jobs``). A separable Z-only erosion (radius ``round(thickness /
      z_voxel)``, no XY coupling) then adds **Z-caps** where a mito column truly ends in Z; ends
      clipped by a volume Z-face are left uncapped (``border_value=1`` + the ``border_gap`` trim). The
      XY shell can still fragment across slices. (2D inputs get a single 2D erosion.)
    - ``"shell_3d"``: a full 3D morphological erosion, ``mito & ~erode3d(mito, k)`` with
      ``k = round(thickness / mean_voxel)`` iterations of a 3×3×3 structuring element, per instance on
      its padded bounding box. A single **connected** shell including the Z-caps (no per-slice
      fragmentation), at a higher cost; thickness acts in all axes.

    The eroded interior is the "lumen"; its surface is the single-wall mesh used by the geodesic
    backend and the display, so the mesh follows the chosen mode.

    Membrane voxels within ``border_gap_nm`` of any volume face are removed so clipped mito edges are
    not treated as membrane. The lumen is NOT trimmed here — the mesh is trimmed to the certain region
    (and left open there) at mesh time by :func:`_open_trimmed_mesh`, which requires the untrimmed
    interior to produce an open cut rather than a fabricated cap.

    Implementation notes: ``"slice_2d"`` erodes each Z-slice on the mito XY bbox with a
    ``membrane_radius`` margin, so the cropped ``border_value=1`` erosion matches eroding the full
    slice (empty slices are skipped), then adds Z-caps via a separable Z-only line erosion (which
    inspects only the same column, so no XY-shape bleed); ``border_value=1`` leaves ends clipped by a
    volume Z-face uncapped, and the ``border_gap`` removal clears anything near a face, so only true
    ends are capped. ``"shell_3d"`` erodes the *merged* binary in each instance's padded bbox so
    instances that share a boundary are handled together.

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
        return_lumen: If True, also return the eroded-mito interior ("lumen") mask — the single-wall
            surface source for the geodesic/display mesh. It is the plain eroded interior (not
            ``mito & ~membrane``, which would re-include the outer shell where the membrane is
            border-trimmed); trimming to the certain region happens at mesh time.

    Returns:
        membrane_mask: Binary mask of the mitochondrial membrane (outer shell), with border-adjacent
            voxels zeroed out. If ``return_lumen`` is True, returns ``(membrane_mask, lumen_mask)``
            where ``lumen_mask`` is the (untrimmed) eroded interior described above.
    """
    if membrane_mode not in ("slice_2d", "shell_3d"):
        raise ValueError(f"membrane_mode must be 'slice_2d' or 'shell_3d', got {membrane_mode!r}")
    ndim = mito_segmentation.ndim
    mito_binary = mito_segmentation > 0

    # NOTE (possible simplification, deferred): the "shell_3d" branch below builds the shell with an
    # iterated 3x3x3 erosion per instance. It could likely be a single anisotropic distance transform
    # instead — membrane = mito & (distance_transform(mito, sampling) <= thickness), lumen = the rest —
    # which is simpler and handles anisotropy directly. This would NOT replace "slice_2d": a distance
    # transform couples all axes, so it cannot reproduce slice_2d's per-Z-slice-independent erosion,
    # whose whole purpose is to stop the shell bleeding across slices in XY. Worth investigating.
    if membrane_mode == "shell_3d":
        sampling = _to_sampling(voxel_size, ndim)
        k = max(1, int(round(float(membrane_thickness_nm) / float(np.mean(sampling)))))
        struct = np.ones((3,) * ndim, dtype=bool)
        membrane_mask = np.zeros(mito_segmentation.shape, dtype=bool)
        lumen_mask = np.zeros(mito_segmentation.shape, dtype=bool)
        for prop in regionprops(mito_segmentation):
            bbox = prop.bbox
            sl = tuple(
                slice(max(0, bbox[i] - k), min(mito_segmentation.shape[i], bbox[i + ndim] + k))
                for i in range(ndim)
            )
            sub = mito_binary[sl]
            eroded = binary_erosion(sub, structure=struct, iterations=k, border_value=1)
            cur = mito_segmentation[sl] == prop.label
            membrane_mask[sl] |= cur & ~eroded
            lumen_mask[sl] |= cur & eroded
    elif ndim == 3:
        membrane_radius = _voxel_radius_xy(membrane_thickness_nm, voxel_size)
        struct = disk(membrane_radius)
        membrane_mask = np.zeros_like(mito_binary)
        lumen_mask = np.zeros_like(mito_binary)
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
                eroded = binary_erosion(sl, structure=struct, border_value=1)
                return z, (sl & ~eroded, eroded)

            z_range = range(int(zmin), int(zmax))
            if n_jobs == 1:
                results = [_erode_slice(z) for z in z_range]
            else:
                n_workers = mp.cpu_count() if n_jobs == -1 else n_jobs
                with futures.ThreadPoolExecutor(n_workers) as tp:
                    results = list(tp.map(_erode_slice, z_range))
            for z, res in results:
                if res is not None:
                    mem_sl, lum_sl = res
                    membrane_mask[z, y0:y1, x0:x1] = mem_sl
                    lumen_mask[z, y0:y1, x0:x1] = lum_sl

            k_z = max(1, int(round(float(membrane_thickness_nm) / float(_to_sampling(voxel_size, ndim)[0]))))
            z0m, z1m = max(0, int(zmin) - k_z), min(mito_binary.shape[0], int(zmax) + k_z)
            sub = mito_binary[z0m:z1m, y0:y1, x0:x1]
            z_eroded = binary_erosion(sub, structure=np.ones((2 * k_z + 1, 1, 1), dtype=bool), border_value=1)
            membrane_mask[z0m:z1m, y0:y1, x0:x1] |= sub & ~z_eroded
            lumen_mask[z0m:z1m, y0:y1, x0:x1] &= z_eroded
    else:
        membrane_radius = _voxel_radius(membrane_thickness_nm, voxel_size, ndim)
        eroded = binary_erosion(mito_binary, structure=disk(membrane_radius), border_value=1)
        membrane_mask = mito_binary & ~eroded
        lumen_mask = mito_binary & eroded

    gap_radius = _gap_radius(voxel_size, membrane_thickness_nm, border_gap_nm, ndim)
    membrane_mask &= ~_border_zone(mito_segmentation.shape, gap_radius)
    if return_lumen:
        return membrane_mask.astype(bool), lumen_mask.astype(bool)
    return membrane_mask.astype(bool)


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def compute_crista_orientation(
    crista_mask: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    neighborhood_size_nm: float = 30.0,
) -> np.ndarray:
    """Compute the per-voxel crista orientation anisotropy via the structure tensor.

    Uses ``bioimage_cpp.filters.structure_tensor_eigenvalues`` (a fast C++ routine). Only the
    anisotropy is produced (the principal directions / eigenvectors are not computed).

    The structure tensor's outer/integration sigma is ``neighborhood_size_nm`` per axis (in voxels);
    the inner (derivative) sigma must be > 0, so a minimal 1-voxel scale is used. Eigenvalues are
    non-negative in theory, but the solver emits tiny negatives for near-rank-deficient tensors
    (degenerate sheets/tubes), so they are clamped to 0 and the ratio is taken as
    ``max/min`` over the trailing axis — order-agnostic and sign-safe, so a tiny negative minor
    eigenvalue cannot flip the denominator and blow the ratio up.

    Args:
        crista_mask: Binary crista segmentation.
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        neighborhood_size_nm: Gaussian integration radius in nm for tensor averaging (the
            structure tensor's outer/integration scale).

    Returns:
        anisotropy: (...) — λ_max / (λ_min + ε) per voxel. High values indicate a strongly
            directional crista (e.g. parallel lamellae); low values indicate isotropic or
            tubular/disordered morphology. Magnitude only (rotation-invariant).
    """
    ndim = crista_mask.ndim
    sampling = _to_sampling(voxel_size, ndim)

    outer_sigma = [float(s) for s in (neighborhood_size_nm / sampling)]
    inner_sigma = 1.0
    evals = structure_tensor_eigenvalues(crista_mask.astype(np.float32), inner_sigma, outer_sigma)
    evals = np.clip(evals, 0.0, None)
    anisotropy = evals.max(axis=-1) / (evals.min(axis=-1) + 1e-10)
    return anisotropy.astype(np.float32)


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
) -> float:
    """Mean crista orientation anisotropy computed on a downsampled crop (fast, approximate).

    The crista mask is **nearest-neighbour** downsampled by ``factor`` per axis (strided subsampling,
    i.e. every ``factor``-th voxel), and :func:`compute_crista_orientation` is run on that coarser grid
    at the correspondingly scaled voxel size (so the physical structure-tensor neighbourhood is
    unchanged). This is ~``factor**ndim`` times cheaper than the full-resolution structure tensor — the
    dominant cost of the analysis. Nearest-neighbour keeps the segmentation binary; block-mean
    (``downscale_local_mean``) would blur it into meaningless partial-occupancy values.

    Because thin cristae (only a few voxels across) lose structure when downsampled, the returned
    anisotropy is a *relative* indicator only: it preserves the ordering between mitochondria but is
    not comparable in magnitude to the full-resolution ``method="exact"`` value.

    Args:
        crista_mask: Binary crista segmentation.
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        factor: Integer downsampling factor per axis.

    Returns:
        Mean anisotropy over the downsampled crista region, or NaN if it vanishes when downsampled.
    """
    ndim = crista_mask.ndim
    sub = crista_mask[(slice(None, None, factor),) * ndim]
    region = sub.astype(bool)
    if not region.any():
        return np.nan
    anisotropy = compute_crista_orientation(sub, _scale_voxel_size(voxel_size, factor))
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
            voxel (nm), i.e. ``distance_transform(~membrane_mask, sampling=...)``. When
            given, the distance transform is not recomputed (used to avoid redundant work in
            :func:`compute_mito_crista_statistics`).

    Returns:
        distance_map: Per-voxel distance to membrane (nm); zero outside crista.
        summary_stats: min_nm, median_nm, max_nm.
    """
    sampling = _to_sampling(voxel_size, crista_mask.ndim)
    if membrane_distance is None:
        dist = distance_transform(~membrane_mask.astype(bool), sampling=sampling.tolist(), number_of_threads=1)
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
    ``bioimage_cpp.distance.geodesic_distances_mesh`` (passing all junction vertices as sources, so it
    returns the full pairwise matrix directly). The mesh (see :func:`_surface_mesh`) returns vertices
    in the unpadded mask index frame, so voxel centroids map to it as ``index * sampling`` with no
    offset. Returns None (junction distances become NaN) when the mesh is empty.

    Args:
        centroids: (n, ndim) junction centroids in voxel coordinates.
        sampling: Voxel size per axis (nm), array (z, y, x) order.
        vertices: Mesh vertices (n_vertices, 3) in nm (unpadded mask frame).
        faces: Mesh triangle indices (n_faces, 3).
        n_jobs: Forwarded to the C++ solver's ``number_of_threads``: -1/0 map to 0 (the solver's
            "use hardware_concurrency"), otherwise that many threads.

    Returns:
        (n, n) geodesic distance matrix in nm (0 diagonal, NaN for disconnected pairs), or None.
        Disconnected pairs come back from the solver as ``+inf`` and are converted to NaN.
    """
    verts = np.ascontiguousarray(vertices, dtype=np.float64)
    tris = np.ascontiguousarray(faces, dtype=np.int64)
    if verts.shape[0] == 0 or tris.shape[0] == 0:
        return None
    from scipy.spatial import cKDTree

    points = np.asarray(centroids, dtype=float) * sampling
    _, vertex_ids = cKDTree(verts).query(points)
    vertex_ids = np.atleast_1d(np.asarray(vertex_ids, dtype=np.int64))
    n_threads = 0 if n_jobs in (-1, 0) else max(1, int(n_jobs))
    dm = np.asarray(
        geodesic_distances_mesh(verts, tris, vertex_ids, number_of_threads=n_threads), dtype=float
    )
    dm[~np.isfinite(dm)] = np.nan
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
    surface at the membrane's inner edge). If no mesh is supplied — or no usable surface mesh exists
    (empty membrane / degenerate mesh) — the junction distances are NaN. (There is no membrane-band
    fallback mesh: the metric is defined on the lumen surface, and meshing the thick membrane band
    would give a different, capped double-wall surface.)

    A Clark-Evans nearest-neighbour index summarises whether the junctions are clustered.

    Args:
        contact_labels: Integer junction label array (0 = background, 1..n = junctions),
            e.g. the first return value of :func:`detect_contact_sites`.
        membrane_mask: Binary mitochondrial membrane mask the junctions sit on. Only used for the
            empty-membrane early-out (no membrane → NaN); it is not meshed.
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        surface_area_nm2: Membrane/mito surface area used as the reference area for the
            Clark-Evans expectation. If None or non-positive, the clustering index is NaN.
        n_jobs: 1 = serial, -1 = all cores (forwarded to the mesh solver's thread count).
        mesh_vertices: Optional (n_vertices, 3) mesh vertices in nm (unpadded mask frame; see
            :func:`_surface_mesh`) — the eroded-mito (lumen) surface. If omitted, the junction
            distances are NaN.
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

        Each junction's nearest-neighbour distance is the smallest distance to a *reachable* other
        junction: self (diagonal) and unreachable (NaN) pairs are set to +inf before the per-row
        minimum, and rows with no reachable neighbour (min stays +inf) are dropped.
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

    centroids = np.atleast_2d(
        np.asarray(center_of_mass(contact_labels > 0, labels=contact_labels, index=labels), dtype=float)
    )

    if mesh_vertices is not None and mesh_faces is not None and len(mesh_faces) > 0:
        distance_matrix = _junction_matrix_mesh(centroids, sampling, mesh_vertices, mesh_faces, n_jobs)
    else:
        distance_matrix = None

    if distance_matrix is None:
        summary = dict(_JUNCTION_DISTANCE_NAN)
        summary["junction_count"] = n
        return np.full((n, n), np.nan, dtype=float), summary

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
    method: str = "skip",
    inner_n_jobs: int = 1,
    lumen_crop: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute the statistics row for a single mitochondrion instance.

    Factored out of :func:`compute_mito_crista_statistics` so the per-mito work (which is
    independent between instances) can be parallelised. Takes the bbox-cropped label arrays
    (``mito_crop`` = label array cropped to ``bbox``; ``crista_crop``/``membrane_crop`` the
    matching binary crops), so a worker only touches its own mito's bbox region rather than the
    whole volume. ``inner_n_jobs`` is forwarded to the (parallelisable) junction-distance stage.

    ``method`` controls only the crista orientation anisotropy — every other metric (marching-cubes
    surface areas, geodesic junction distances, EDT proximity/thickness) is computed identically for
    all modes. ``"skip"`` (the default) leaves the anisotropy NaN and is the fastest; ``"fast"``
    computes it on a 2× downsampled crop (~8× cheaper, a *relative* indicator only, not comparable to
    exact); ``"exact"`` uses the full-resolution structure tensor (the dominant cost).

    ``lumen_crop`` is the (optional) bbox-cropped eroded lumen from :func:`approximate_membrane`
    (``return_lumen=True``) used for the junction geodesic mesh; without it the mesh falls back to
    ``mito_local & ~membrane_local`` (used only when a caller supplies their own membrane, and
    contaminated near clipped faces). Both the lumen geodesic mesh and the mito outer-surface-area mesh
    are trimmed/opened at faces where the mito is clipped by the volume boundary: the lumen via
    :func:`_open_trimmed_mesh` (trimmed to the certain region and left open, so geodesics do not
    shortcut across a cap), the mito surface via ``closed_faces`` (open, so a fabricated cap is not
    counted as membrane area). The membrane distance transform is freed before the (memory-heavy)
    orientation stage to cap peak memory.
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

    boundary = np.array(
        [[bbox[a] == 0, bbox[a + ndim] == vol_shape[a]] for a in range(ndim)], dtype=bool
    )

    has_crista = crista_local.any()
    has_membrane = membrane_local.any()
    mito_surface = _surface_area(mito_local, sampling, closed_faces=~boundary)

    if has_crista and has_membrane:
        contact_labels_local, contact_summary = detect_contact_sites(crista_local, membrane_local, voxel_size)
        membrane_distance = distance_transform(~membrane_local, sampling=sampling.tolist(), number_of_threads=1)
        _, proximity = compute_crista_proximity(
            crista_local, membrane_local, voxel_size, membrane_distance=membrane_distance
        )
        lumen_local = (lumen_crop & mito_local) if lumen_crop is not None else (mito_local & ~membrane_local)
        lumen_mesh = _open_trimmed_mesh(lumen_local, sampling, border_radius, boundary)
        mesh_verts, mesh_faces = lumen_mesh if lumen_mesh is not None else (None, None)
        _, junction_dist = compute_junction_distances(
            contact_labels_local, membrane_local, voxel_size,
            surface_area_nm2=mito_surface, n_jobs=inner_n_jobs,
            mesh_vertices=mesh_verts, mesh_faces=mesh_faces,
        )
        del membrane_distance, contact_labels_local
    else:
        contact_summary = {"contact_voxel_count": 0, "crista_junction_count": 0, "contact_volume_nm3": 0.0}
        proximity = {"median_nm": np.nan}
        junction_dist = dict(_JUNCTION_DISTANCE_NAN)

    if has_crista:
        morph = compute_crista_morphology(crista_local, voxel_size)
        crista_surface = morph.get("cristae_surface_area_nm2", np.nan)
        avg_thickness_nm = morph.get("avg_thickness_nm", np.nan)
        if method == "skip":
            crista_orientation_anisotropy = np.nan
        elif method == "fast":
            crista_orientation_anisotropy = _downsampled_orientation_anisotropy(
                crista_local, voxel_size, factor=2
            )
        else:
            anisotropy = compute_crista_orientation(crista_local, voxel_size)
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
    method: str = "skip",
    n_jobs: int = 1,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    membrane_mode: str = "slice_2d",
    lumen_mask: Optional[np.ndarray] = None,
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
            them. ``"skip"`` (default) does not compute orientation at all
            (``crista_orientation_anisotropy`` is NaN) and is the fastest — use it when only the other
            metrics are needed. ``"fast"`` computes the anisotropy on a 2× downsampled crista crop
            (~8× cheaper — the structure tensor is by far the dominant cost); the resulting value is a
            *relative* indicator that preserves the ordering between mitochondria but is systematically
            different in magnitude from the full-resolution value and is NOT comparable to
            ``method="exact"``. ``"exact"`` computes the anisotropy from the full-resolution structure
            tensor (use it when the magnitude must be precise).
        n_jobs: Number of workers for processing mitochondria in parallel (they are
            independent). 1 (default) runs serially; other values use a ``concurrent.futures``
            thread pool (-1 = all cores). Results are identical regardless of n_jobs.
        verbose: If True, show a terminal tqdm progress bar over mitochondria.
        progress_callback: Optional callable invoked once per completed mitochondrion with
            (completed_count, total_count) — e.g. to drive a napari progress bar. It is
            always called from the calling thread (the futures are consumed here as they
            complete), so GUI updates from it need no cross-thread marshaling.
        membrane_mode: How the membrane shell is built when ``membrane_mask`` is None —
            ``"slice_2d"`` (default, per-Z-slice 2D erosion, z-parallel) or ``"shell_3d"`` (connected
            3D shell). See :func:`approximate_membrane`.
        lumen_mask: Optional eroded-mito interior matching ``membrane_mask``, i.e. the second return
            value of ``approximate_membrane(..., return_lumen=True)``. It is the clean single-wall
            surface the junction geodesics run along. Only used when ``membrane_mask`` is also
            supplied (when the membrane is built here, the matching lumen is derived automatically);
            when neither is available the geodesic mesh falls back to ``mito & ~membrane``, which is
            contaminated by the membrane's border-gap suppression near clipped volume faces.

    The junction nearest-neighbour distances are geodesics along the eroded-mito surface mesh
    (``bioimage_cpp.distance.geodesic_distances_mesh``); for a mito with no usable mesh (empty
    membrane / degenerate mesh) those columns are NaN.

    Implementation notes: each mito is pre-cropped to its bounding box by basic slicing (views, so
    cropping is memory-free). Parallelism is adaptive and single-level (never oversubscribed): with
    many mitochondria the work is parallelised *across* them on a ``concurrent.futures``
    ``ThreadPoolExecutor`` — the heavy per-mito stages (structure tensor, EDT, geodesics) are
    GIL-releasing C++, so threads scale them — with each worker's inner stages kept single-threaded
    (the EDT/geodesic solvers are called with ``number_of_threads=1``); with few mitochondria they run
    serially and each mito's junction-distance stage gets all cores. The concurrent worker count is
    additionally capped so the combined per-mito working set (tensor components + label crops,
    ~40 bytes/voxel of the largest mito) fits in RAM. Results stream in as they complete and are
    finally sorted by label for an n_jobs-independent ordering.

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
    if membrane_mask is None:
        membrane_mask, lumen_mask = approximate_membrane(
            mito_segmentation, voxel_size, membrane_thickness_nm, border_gap_nm,
            n_jobs=n_jobs, membrane_mode=membrane_mode, return_lumen=True,
        )

    ndim = mito_segmentation.ndim
    sampling = _to_sampling(voxel_size, ndim)
    voxel_vol = float(np.prod(sampling))
    crista_binary = crista_mask.astype(bool)
    vol_shape = mito_segmentation.shape
    border_radius = _gap_radius(voxel_size, membrane_thickness_nm, border_gap_nm, ndim)

    tasks = []
    for prop in regionprops(mito_segmentation):
        bbox = prop.bbox
        slices = tuple(slice(bbox[i], bbox[i + ndim]) for i in range(ndim))
        lumen_crop = None if lumen_mask is None else lumen_mask[slices]
        tasks.append((
            int(prop.label), bbox,
            mito_segmentation[slices], crista_binary[slices], membrane_mask[slices],
            lumen_crop,
        ))
    total = len(tasks)

    def _run(task, inner_n_jobs):
        label, bbox, mito_crop, crista_crop, membrane_crop, lumen_crop = task
        return _single_mito_row(
            label, bbox, mito_crop, crista_crop, membrane_crop,
            voxel_size, sampling, voxel_vol, vol_shape, border_radius,
            method=method, inner_n_jobs=inner_n_jobs, lumen_crop=lumen_crop,
        )

    n_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
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
        max_voxels = max(int(task[2].size) for task in tasks)
        across_workers = _bounded_workers(n_jobs, per_worker_bytes=max_voxels * 40)
        # Parallelise across mitochondria with a thread pool (the heavy per-mito stages — structure
        # tensor, EDT, geodesics — are GIL-releasing C++). Each worker's inner stages run
        # single-threaded (``_run(task, 1)`` passes ``number_of_threads=1`` down to the EDT/geodesic
        # solvers) so the across-mito threads do not oversubscribe the cores. Results stream in as they
        # complete (``as_completed``) to drive the progress bar; rows are label-sorted below.
        with futures.ThreadPoolExecutor(across_workers) as tp:
            submitted = [tp.submit(_run, task, 1) for task in tasks]
            _consume(future.result() for future in futures.as_completed(submitted))
    else:
        _consume(_run(task, n_workers) for task in tasks)

    rows.sort(key=lambda row: row["mito_label_id"])
    return pd.DataFrame(rows)
