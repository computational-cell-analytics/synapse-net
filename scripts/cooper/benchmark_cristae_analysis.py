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

To benchmark on a real tomogram, replace ``build_volume`` with loaders for your mito instance
segmentation and binary cristae mask (e.g. read the .h5/.mrc as in the other cooper scripts) and
pass the matching ``voxel_size``.
"""

import argparse
import time
import tracemalloc

import numpy as np

from synapse_net.cristae_analysis import approximate_membrane, compute_mito_crista_statistics


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


def main():
    parser = argparse.ArgumentParser(description="Benchmark cristae analysis (serial vs parallel).")
    parser.add_argument("--mito", type=int, default=4, help="Number of mitochondria.")
    parser.add_argument("--size", type=int, default=140, help="Cube side length (voxels) per mito.")
    parser.add_argument("--voxel_size", type=float, default=1.5, help="Voxel size (nm, isotropic).")
    args = parser.parse_args()

    mito, crista = build_volume(args.mito, args.size)
    print(f"volume {mito.shape} ({mito.size / 1e6:.1f} M voxels), {args.mito} mitochondria")

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
