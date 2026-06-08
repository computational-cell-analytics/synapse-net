from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, gaussian_filter
from scipy.ndimage import label as ndimage_label
from skimage.measure import marching_cubes, mesh_surface_area, regionprops
from skimage.morphology import ball, disk, local_maxima
from skimage.segmentation import find_boundaries


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


def _struct_elem(radius: int, ndim: int) -> np.ndarray:
    return ball(radius) if ndim == 3 else disk(radius)


# ---------------------------------------------------------------------------
# Membrane approximation
# ---------------------------------------------------------------------------

def approximate_membrane(
    mito_segmentation: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    outer_membrane_thickness_nm: float = 8.0,
    inner_membrane_thickness_nm: float = 8.0,
    ims_thickness_nm: float = 7.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate OM and IMM from mito instance labels.

    Args:
        mito_segmentation: Instance label array (background = 0).
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        outer_membrane_thickness_nm: OM surface dilation thickness in nm.
        inner_membrane_thickness_nm: IMM surface dilation thickness in nm.
        ims_thickness_nm: Intermembrane space thickness in nm used to erode the
            mito binary before detecting the IMM boundary.

    Returns:
        om_mask: Binary mask of the outer mitochondrial membrane.
        imm_mask: Binary mask of the inner mitochondrial membrane (clipped to mito interior).
    """
    ndim = mito_segmentation.ndim
    mito_binary = mito_segmentation > 0

    om_radius = _voxel_radius(outer_membrane_thickness_nm, voxel_size, ndim)
    om_mask = binary_dilation(find_boundaries(mito_binary, mode="thick"), structure=_struct_elem(om_radius, ndim))

    ims_radius = _voxel_radius(ims_thickness_nm, voxel_size, ndim)
    eroded = binary_erosion(mito_binary, structure=_struct_elem(ims_radius, ndim))
    imm_radius = _voxel_radius(inner_membrane_thickness_nm, voxel_size, ndim)
    imm_mask = binary_dilation(find_boundaries(eroded, mode="thick"), structure=_struct_elem(imm_radius, ndim))
    imm_mask &= mito_binary

    return om_mask.astype(bool), imm_mask.astype(bool)


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def compute_crista_orientation(
    crista_mask: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    neighborhood_size_nm: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute dominant crista orientation via structure tensor.

    Args:
        crista_mask: Binary crista segmentation.
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        neighborhood_size_nm: Gaussian smoothing radius in nm for tensor averaging.

    Returns:
        eigenvalues: (..., ndim) sorted ascending.
        eigenvectors: (..., ndim, ndim) — columns are principal directions.
        anisotropy: (...) — λ_max / (λ_min + ε), measures degree of alignment.
    """
    ndim = crista_mask.ndim
    sampling = _to_sampling(voxel_size, ndim)
    sigma = neighborhood_size_nm / sampling

    grads = np.gradient(crista_mask.astype(np.float32), *sampling.tolist())

    J = np.zeros(crista_mask.shape + (ndim, ndim), dtype=np.float32)
    for i in range(ndim):
        for j in range(i, ndim):
            s = gaussian_filter(grads[i] * grads[j], sigma=sigma)
            J[..., i, j] = s
            J[..., j, i] = s

    eigenvalues, eigenvectors = np.linalg.eigh(J)
    anisotropy = eigenvalues[..., -1] / (eigenvalues[..., 0] + 1e-10)
    return eigenvalues, eigenvectors, anisotropy


# ---------------------------------------------------------------------------
# Proximity
# ---------------------------------------------------------------------------

def compute_crista_proximity(
    crista_mask: np.ndarray,
    membrane_mask: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Distance from each crista voxel to the nearest membrane voxel (nm).

    Args:
        crista_mask: Binary crista segmentation.
        membrane_mask: Binary membrane mask (OM or IMM).
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.

    Returns:
        distance_map: Per-voxel distance to membrane (nm); zero outside crista.
        summary_stats: min_nm, median_nm, max_nm.
    """
    sampling = _to_sampling(voxel_size, crista_mask.ndim)
    dist = distance_transform_edt(~membrane_mask.astype(bool), sampling=sampling.tolist())
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
    imm_mask: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Detect crista-IMM contact sites by voxel adjacency (26-connectivity in 3D).

    Contact = crista voxels that are face-, edge-, or corner-adjacent to any IMM voxel.

    Args:
        crista_mask: Binary crista segmentation.
        imm_mask: Binary inner mitochondrial membrane mask.
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.

    Returns:
        contact_coords: (N, ndim) integer array of contact voxel coordinates.
        summary: contact_voxel_count, contact_region_count, contact_volume_nm3.
    """
    ndim = crista_mask.ndim
    sampling = _to_sampling(voxel_size, ndim)
    voxel_vol = float(np.prod(sampling))

    dilated_imm = binary_dilation(imm_mask.astype(bool), structure=_struct_elem(1, ndim))
    contact_mask = crista_mask.astype(bool) & dilated_imm

    _, n_regions = ndimage_label(contact_mask)
    contact_coords = np.argwhere(contact_mask)

    return contact_coords, {
        "contact_voxel_count": int(contact_coords.shape[0]),
        "contact_region_count": int(n_regions),
        "contact_volume_nm3": float(contact_coords.shape[0]) * voxel_vol,
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
        Dict with total_surface_area_nm2 (area/both) and avg_thickness_nm (medial_axis/both).
        avg_thickness_nm is 2 × mean distance-transform value at skeleton voxels.
    """
    if method not in ("area", "medial_axis", "both"):
        raise ValueError(f"method must be 'area', 'medial_axis', or 'both', got {method!r}")

    sampling = _to_sampling(voxel_size, crista_mask.ndim)
    spacing = tuple(float(s) for s in sampling)
    result: Dict[str, float] = {}

    if method in ("area", "both"):
        verts, faces, _, _ = marching_cubes(crista_mask.astype(np.float32), level=0.5, spacing=spacing)
        result["total_surface_area_nm2"] = float(mesh_surface_area(verts, faces))

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

def compute_mito_crista_statistics(
    crista_mask: np.ndarray,
    mito_segmentation: np.ndarray,
    voxel_size: Union[float, Dict[str, float]],
    om_mask: Optional[np.ndarray] = None,
    imm_mask: Optional[np.ndarray] = None,
    ims_thickness_nm: float = 7.0,
    outer_membrane_thickness_nm: float = 8.0,
    inner_membrane_thickness_nm: float = 8.0,
) -> pd.DataFrame:
    """Compute all crista metrics organised by mitochondrial instance.

    Args:
        crista_mask: Binary crista segmentation (global volume).
        mito_segmentation: Instance label array (background = 0).
        voxel_size: Voxel size in nm — scalar or dict with "z"/"y"/"x" keys.
        om_mask: Precomputed OM mask; recomputed if None.
        imm_mask: Precomputed IMM mask; recomputed if None.
        ims_thickness_nm: Intermembrane space thickness for IMM approximation.
        outer_membrane_thickness_nm: OM dilation thickness.
        inner_membrane_thickness_nm: IMM dilation thickness.

    Returns:
        DataFrame with one row per mito instance:
        label | mito_volume_nm3 | crista_volume_nm3 | crista_fraction |
        contact_voxel_count | contact_region_count | contact_volume_nm3 |
        avg_crista_to_imm_nm | anisotropy_mean | total_surface_area_nm2 | avg_thickness_nm
    """
    if om_mask is None or imm_mask is None:
        om_mask, imm_mask = approximate_membrane(
            mito_segmentation, voxel_size,
            outer_membrane_thickness_nm=outer_membrane_thickness_nm,
            inner_membrane_thickness_nm=inner_membrane_thickness_nm,
            ims_thickness_nm=ims_thickness_nm,
        )

    ndim = mito_segmentation.ndim
    sampling = _to_sampling(voxel_size, ndim)
    voxel_vol = float(np.prod(sampling))
    crista_binary = crista_mask.astype(bool)

    rows = []
    for prop in regionprops(mito_segmentation):
        mito_id = prop.label
        bbox = prop.bbox
        slices = tuple(slice(bbox[i], bbox[i + ndim]) for i in range(ndim))

        mito_local = mito_segmentation[slices] == mito_id
        crista_local = crista_binary[slices] & mito_local
        imm_local = imm_mask[slices] & mito_local

        mito_vol = float(mito_local.sum()) * voxel_vol
        crista_vol = float(crista_local.sum()) * voxel_vol

        has_crista = crista_local.any()
        has_imm = imm_local.any()

        if has_crista and has_imm:
            _, contact_summary = detect_contact_sites(crista_local, imm_local, voxel_size)
            _, proximity = compute_crista_proximity(crista_local, imm_local, voxel_size)
        else:
            contact_summary = {"contact_voxel_count": 0, "contact_region_count": 0, "contact_volume_nm3": 0.0}
            proximity = {"median_nm": np.nan}

        if has_crista:
            _, _, anisotropy = compute_crista_orientation(crista_local, voxel_size)
            anisotropy_mean = float(np.mean(anisotropy[crista_local]))
            morph = compute_crista_morphology(crista_local, voxel_size)
        else:
            anisotropy_mean = np.nan
            morph = {"total_surface_area_nm2": np.nan, "avg_thickness_nm": np.nan}

        rows.append({
            "label": int(mito_id),
            "mito_volume_nm3": mito_vol,
            "crista_volume_nm3": crista_vol,
            "crista_fraction": crista_vol / mito_vol if mito_vol > 0 else np.nan,
            "contact_voxel_count": contact_summary["contact_voxel_count"],
            "contact_region_count": contact_summary["contact_region_count"],
            "contact_volume_nm3": contact_summary["contact_volume_nm3"],
            "avg_crista_to_imm_nm": proximity["median_nm"],
            "anisotropy_mean": anisotropy_mean,
            "total_surface_area_nm2": morph.get("total_surface_area_nm2", np.nan),
            "avg_thickness_nm": morph.get("avg_thickness_nm", np.nan),
        })

    return pd.DataFrame(rows)
