"""Benchmark compute_mito_crista_statistics: fast vs exact, serial vs parallel.

Builds a synthetic volume with a configurable number of mitochondria (each filled with
parallel cristae) and times the analysis. ``method="fast"`` and ``method="exact"`` differ ONLY in
the crista orientation anisotropy: fast computes it on a 2x-downsampled crop (the structure tensor
is the dominant cost), everything else (marching-cubes surface areas, geodesic junction distances,
EDT thickness/proximity) is identical. The script therefore expects ~0% relative error on the
surface / junction / thickness columns and reports the orientation separately (fast is a
downsampled, relative-only approximation, so its magnitude differs from exact by design).

    python scripts/cooper/benchmark_cristae_analysis.py
    python scripts/cooper/benchmark_cristae_analysis.py --mito 4 --size 160 --voxel_size 1.5

``--backend`` instead runs the geodesic junction-distance backend comparison: the exact membrane
sparse-graph Dijkstra (default) vs the bioimage-cpp mesh surface geodesic (``mesh``, the shipped
optional backend) vs a scikit-fmm eikonal solver masked to the membrane shell or the mitochondrion
interior. It reports per-backend wall-clock and agreement (max |Δ mean-NN|) with Dijkstra. The
``mesh`` backend needs a bioimage-cpp build with the geodesic API and ``fmm-*`` need scikit-fmm; each
is skipped with a note when its dependency is unavailable.

    python scripts/cooper/benchmark_cristae_analysis.py --backend all
    python scripts/cooper/benchmark_cristae_analysis.py --backend mesh --mito 2 --size 120

To benchmark on a real tomogram, replace ``build_volume`` with loaders for your mito instance
segmentation and binary cristae mask (e.g. read the .h5/.mrc as in the other cooper scripts) and
pass the matching ``voxel_size``.
"""

import argparse
import time
import tracemalloc

import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt
from skimage.measure import regionprops

from synapse_net.cristae_analysis import (
    approximate_membrane,
    compute_junction_distances,
    compute_mito_crista_statistics,
    detect_contact_sites,
    geodesic_distances_mesh,  # None when the installed bioimage-cpp lacks the geodesic API
    _surface_mesh,
    _to_sampling,
)

try:
    import skfmm
except ImportError:
    skfmm = None


def build_volume(n_mito, size, gap=4):
    """A row of `n_mito` cube mitochondria, each packed with parallel crista sheets."""
    shape = (size, size, size * n_mito)
    mito = np.zeros(shape, dtype="uint32")
    crista = np.zeros(shape, dtype=bool)
    for m in range(n_mito):
        x0 = m * size
        mito[gap:size - gap, gap:size - gap, x0 + gap:x0 + size - gap] = m + 1
        for x in range(x0 + gap + 4, x0 + size - gap - 2, 6):
            crista[gap + 2:size - gap - 2, gap + 2:size - gap - 2, x:x + 2] = True
    return mito, crista


# ---------------------------------------------------------------------------
# Geodesic junction-distance backend comparison (--backend)
#
# The shipped junction geodesic uses an exact membrane-only sparse-graph Dijkstra
# (compute_junction_distances). This section lets you reproduce the evaluation of alternatives:
#   - mesh          : bioimage-cpp surface geodesic on the mito marching-cubes mesh (the shipped
#                     optional backend). Requires a bioimage-cpp build with the geodesic API.
#   - fmm-shell     : scikit-fmm eikonal masked to the membrane shell (same "along the surface"
#                     meaning as Dijkstra).
#   - fmm-interior  : scikit-fmm masked to the whole mitochondrion interior (paths may shortcut
#                     through the lumen — a different metric).
# It reports, per backend, wall-clock and agreement with Dijkstra. Optional backends are skipped
# with a note when their dependency (bioimage-cpp geodesic API / scikit-fmm) is unavailable.
# ---------------------------------------------------------------------------

BACKENDS = ("dijkstra", "mesh", "fmm-shell", "fmm-interior")


def _seed_positions(contact_labels, membrane, sampling):
    """Junction seeds exactly as compute_junction_distances places them (centroid snapped to membrane)."""
    labels = [lbl for lbl in np.unique(contact_labels) if lbl != 0]
    if len(labels) < 2 or not membrane.any():
        return []
    _, nearest = distance_transform_edt(~membrane, return_indices=True, sampling=sampling.tolist())
    centroids = center_of_mass(contact_labels > 0, labels=contact_labels, index=labels)
    shape_max = np.array(contact_labels.shape) - 1
    seeds = []
    for c in np.atleast_2d(np.asarray(centroids, dtype=float)):
        c = np.clip(np.round(c).astype(int), 0, shape_max)
        seeds.append(tuple(int(nearest[d][tuple(c)]) for d in range(contact_labels.ndim)))
    return seeds


def _fmm_distance_matrix(seeds, domain, sampling):
    """Pairwise geodesic matrix via one skfmm.distance solve per seed, masked to `domain` (bool)."""
    n = len(seeds)
    dm = np.full((n, n), np.nan)
    inv = ~domain
    dx = [float(s) for s in sampling]
    half = 0.5 * float(np.mean(sampling))  # the zero level set sits ~half a voxel off the seed centre
    phi_base = np.ones(domain.shape, dtype=float)
    for i, s in enumerate(seeds):
        phi = phi_base.copy()
        phi[s] = -1.0
        d = np.abs(skfmm.distance(np.ma.MaskedArray(phi, inv), dx=dx))
        for j, t in enumerate(seeds):
            v = d[t]
            dm[i, j] = np.nan if v is np.ma.masked else float(v) + half
    np.fill_diagonal(dm, 0.0)
    return dm


def _mean_nn(dm):
    """Mean nearest-neighbour distance from a junction distance matrix (matches the library)."""
    d = dm.copy()
    np.fill_diagonal(d, np.inf)
    d[~np.isfinite(d)] = np.inf
    row_min = d.min(axis=1)
    row_min = row_min[np.isfinite(row_min)]
    return float(np.mean(row_min)) if row_min.size else np.nan


def _mito_junction_tasks(mito, crista, voxel_size):
    """Per-mito (contact_labels, membrane_shell, mito_interior) crops with >= 2 junctions."""
    membrane = approximate_membrane(mito, voxel_size)
    crista_b = crista.astype(bool)
    ndim = mito.ndim
    tasks = []
    for prop in regionprops(mito):
        sl = tuple(slice(prop.bbox[i], prop.bbox[i + ndim]) for i in range(ndim))
        mito_local = mito[sl] == prop.label
        crista_local = crista_b[sl] & mito_local
        membrane_local = membrane[sl] & mito_local
        if not (crista_local.any() and membrane_local.any()):
            continue
        contact_labels, _ = detect_contact_sites(crista_local, membrane_local, voxel_size)
        if contact_labels.max() < 2:
            continue
        tasks.append((contact_labels, membrane_local, mito_local))
    return tasks


def run_backend_comparison(mito, crista, voxel_size, backends):
    """Time each geodesic backend over every mitochondrion and report agreement vs Dijkstra."""
    sampling = _to_sampling(voxel_size, mito.ndim)
    tasks = _mito_junction_tasks(mito, crista, voxel_size)
    print(f"\ngeodesic backend comparison — {len(tasks)} mitochondria with >= 2 junctions")
    if not tasks:
        print("  (no mitochondria with >= 2 junctions; nothing to compare)")
        return

    def matrix(backend, contact_labels, membrane_local, mito_local):
        if backend == "dijkstra":
            dm, _ = compute_junction_distances(contact_labels, membrane_local, voxel_size)
            return dm
        if backend == "mesh":
            # Mirror the pipeline: mesh the eroded-mito (lumen) surface = mito interior inside the
            # membrane band; pass it to the mesh geodesic backend.
            lumen = mito_local & ~membrane_local
            lmesh = _surface_mesh(lumen, sampling)
            mv, mf = (lmesh if lmesh is not None else (None, None))
            dm, _ = compute_junction_distances(
                contact_labels, membrane_local, voxel_size, geodesic_backend="mesh",
                mesh_vertices=mv, mesh_faces=mf,
            )
            return dm
        domain = membrane_local if backend == "fmm-shell" else mito_local
        return _fmm_distance_matrix(_seed_positions(contact_labels, membrane_local, sampling), domain, sampling)

    # Always time Dijkstra first as the reference (speed + per-mito mean-NN for agreement).
    ordered = ["dijkstra"] + [b for b in backends if b != "dijkstra"]
    ref_mean_nns = None
    ref_time = None
    for backend in ordered:
        if backend.startswith("fmm") and skfmm is None:
            print(f"  {backend:<13}: skipped (scikit-fmm not installed)")
            continue
        if backend == "mesh" and geodesic_distances_mesh is None:
            print(f"  {backend:<13}: skipped (bioimage-cpp geodesic API not available)")
            continue
        t = time.time()
        mean_nns = [_mean_nn(matrix(backend, *task)) for task in tasks]
        dt = time.time() - t
        avg = np.nanmean(mean_nns) if mean_nns else float("nan")
        if backend == "dijkstra":
            ref_mean_nns, ref_time = mean_nns, dt
            print(f"  {backend:<13}: {dt * 1000:8.1f} ms   mean-NN over mitos = {avg:.2f} nm   (reference)")
        else:
            diffs = [abs(a - b) for a, b in zip(mean_nns, ref_mean_nns) if np.isfinite(a) and np.isfinite(b)]
            worst = max(diffs) if diffs else float("nan")
            speedup = f"{ref_time / dt:.2f}x" if (ref_time and dt > 0) else "n/a"
            print(f"  {backend:<13}: {dt * 1000:8.1f} ms   mean-NN over mitos = {avg:.2f} nm"
                  f"   max |Δ mean-NN| vs dijkstra = {worst:.2f} nm   speed vs dijkstra = {speedup}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark cristae analysis (serial vs parallel).")
    parser.add_argument("--mito", type=int, default=4, help="Number of mitochondria.")
    parser.add_argument("--size", type=int, default=140, help="Cube side length (voxels) per mito.")
    parser.add_argument("--voxel_size", type=float, default=1.5, help="Voxel size (nm, isotropic).")
    parser.add_argument(
        "--backend", choices=("off", "all") + BACKENDS, default="off",
        help="Also run the geodesic junction-distance backend comparison. 'off' (default) runs only "
             "the orientation-method benchmark; 'all' compares dijkstra vs mesh vs fmm-shell vs "
             "fmm-interior; or name a single backend. 'mesh' requires a bioimage-cpp build with the "
             "geodesic API; fmm-* require scikit-fmm (both optional).",
    )
    args = parser.parse_args()

    mito, crista = build_volume(args.mito, args.size)
    print(f"volume {mito.shape} ({mito.size / 1e6:.1f} M voxels), {args.mito} mitochondria")

    if args.backend != "off":
        backends = BACKENDS if args.backend == "all" else (args.backend,)
        run_backend_comparison(mito, crista, args.voxel_size, backends)
        return

    membrane = approximate_membrane(mito, args.voxel_size)

    def run(method, n_jobs, track_ram=False, verbose=False):
        if track_ram:
            tracemalloc.start()
        t = time.time()
        df = compute_mito_crista_statistics(
            crista, mito, args.voxel_size, membrane_mask=membrane,
            method=method, n_jobs=n_jobs, verbose=verbose,
        )
        dt = time.time() - t
        peak = None
        if track_ram:
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        return dt, df, peak

    def rel_err(col, df_fast, df_exact):
        a = df_fast[col].to_numpy(dtype=float)
        b = df_exact[col].to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
        if not mask.any():
            return float("nan")
        return float(np.mean(np.abs(a[mask] - b[mask]) / np.abs(b[mask])))

    # --- Timing: exact vs fast (both serial), fast also reports peak RAM and the progress bar. ---
    t_exact, df_exact, _ = run("exact", 1)
    print(f"exact  n_jobs=1 : {t_exact:6.2f} s  ({len(df_exact)} rows, "
          f"{int(df_exact['crista_junction_count'].sum())} junctions total)")

    t_fast, df_fast, peak = run("fast", 1, track_ram=True, verbose=True)
    print(f"fast   n_jobs=1 : {t_fast:6.2f} s  ({len(df_fast)} rows, "
          f"{int(df_fast['crista_junction_count'].sum())} junctions total), "
          f"peak RAM {peak / 1e6:.0f} MB")
    if t_fast > 0:
        print(f"fast vs exact speedup : {t_exact / t_fast:.2f}x")

    # --- Skip orientation entirely: the fastest mode (structure tensor not run at all). ---
    t_skip, df_skip, _ = run("skip", 1)
    print(f"skip   n_jobs=1 : {t_skip:6.2f} s  (orientation column empty)")
    if t_skip > 0:
        print(f"skip vs exact speedup : {t_exact / t_skip:.2f}x")

    # --- Per-mito parallelism on the fast path. ---
    t_par, _, _ = run("fast", -1)
    print(f"fast   n_jobs=-1: {t_par:6.2f} s")
    if t_par > 0:
        print(f"fast serial vs parallel speedup : {t_fast / t_par:.2f}x")

    # --- Accuracy vs exact. Surface / junction / thickness are shared code -> expect ~0%.
    #     Orientation is the one approximated stage (downsampled) -> reported separately. ---
    print("\nrelative error (fast vs exact), mean over mitochondria:")
    for col in (
        "cristae_surface_area_nm2", "mito_surface_area_nm2",
        "avg_thickness_nm", "mean_nn_junction_distance_nm",
    ):
        print(f"  {col:<32}: {rel_err(col, df_fast, df_exact):.1%}  (expect ~0%)")
    print(f"  {'crista_orientation_anisotropy':<32}: "
          f"{rel_err('crista_orientation_anisotropy', df_fast, df_exact):.1%}  "
          f"(downsampled approximation, differs by design)")


if __name__ == "__main__":
    main()
