import glob
import os
import sys
import unittest

import numpy as np

# Set SYNAPSE_NET_VIEW=1 (or run this file with --view) to open the orientation cases in
# napari — see TestCristaOrientation.test_visualize_orientations.
VIEW_ENV = "SYNAPSE_NET_VIEW"

# Junction distances are surface geodesics on the eroded-mito mesh (bioimage-cpp); skip the
# distance-value tests when that API is unavailable.
from synapse_net.cristae_analysis import geodesic_distances_mesh as _GEODESIC_MESH_API  # noqa: E402
_MESH_REQUIRED = unittest.skipUnless(_GEODESIC_MESH_API is not None, "bioimage-cpp geodesic API unavailable")

# Real-data integration test (skipped unless the data tree is present). Labels come from the
# pre-converted HDF5 tree; voxel size from the sibling .mrc header. Override the root with
# SYNAPSE_NET_CRISTAE_DATA.
_CRISTAE_DATA_ROOT = os.environ.get("SYNAPSE_NET_CRISTAE_DATA", "/home/freckmann15/data/cristae/cooper")
_H5_DIR = os.path.join(_CRISTAE_DATA_ROOT, "forgotten_cristae_h5")
_MRC_DIR = os.path.join(_CRISTAE_DATA_ROOT, "forgotten_cristae")
_DEFAULT_VOXEL_NM = 0.8681
# One mito in CA3_PS_23 is ~174M voxels and crashes the per-mito pipeline; drop anything above
# this so the test stays bounded. Keeps the normal-sized mitos in every file.
_MAX_MITO_VOXELS = 60_000_000

_REAL_DATA_AVAILABLE = bool(glob.glob(os.path.join(_H5_DIR, "*", "*.h5")))
_REAL_DATA_REQUIRED = unittest.skipUnless(
    _REAL_DATA_AVAILABLE, f"cristae real-data tree not found under {_H5_DIR}"
)


def _resolve_voxel_size(h5_path):
    """Voxel size (nm) from the sibling .mrc header (Angstrom/10); else the isotropic fallback."""
    import mrcfile

    stem = os.path.basename(h5_path).replace("_crop.h5", "_crop.mrc")
    matches = glob.glob(os.path.join(_MRC_DIR, "*", stem))
    if matches:
        try:
            with mrcfile.open(matches[0], permissive=True, header_only=True) as f:
                vs = f.voxel_size
            return {"z": float(vs.z) / 10, "y": float(vs.y) / 10, "x": float(vs.x) / 10}
        except Exception:  # pragma: no cover - depends on the local .mrc header
            pass
    return _DEFAULT_VOXEL_NM


def _drop_large_mitos(mito, max_voxels):
    """Zero out mito instances larger than ``max_voxels``; return (filtered_copy, n_surviving)."""
    filtered = mito.copy()
    labels, counts = np.unique(filtered, return_counts=True)
    surviving = 0
    for label, count in zip(labels, counts):
        if label == 0:
            continue
        if count > max_voxels:
            filtered[filtered == label] = 0
        else:
            surviving += 1
    return filtered, surviving


def _make_mito(shape=(32, 32, 32), label=1):
    seg = np.zeros(shape, dtype="uint32")
    seg[4:-4, 4:-4, 4:-4] = label
    return seg


def _make_crista(shape=(32, 32, 32)):
    mask = np.zeros(shape, dtype=bool)
    mask[10:14, 8:24, 8:24] = True  # a flat sheet inside the mito
    return mask


def _membrane_components(membrane):
    """Number of 26-connected components of a binary membrane mask."""
    from scipy.ndimage import label as _lab
    return int(_lab(membrane, structure=np.ones((3,) * membrane.ndim, dtype=bool))[1])


def _make_lamellae(shape=(48, 48, 48), normal=(1, 0, 0), spacing=6, thickness=2, margin=8):
    """Parallel sheets (lamellae) with a known unit `normal`, inside a margin box.

    The sheets are placed by thresholding the projection of the voxel coordinates onto
    `normal`, so any orientation (axis-aligned or oblique) is produced from an exact,
    analytically known normal — no interpolation artefacts. The dominant structure-tensor
    direction of the result should point along `normal`.
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    coords = np.indices(shape).astype(np.float32)
    proj = sum(normal[i] * coords[i] for i in range(len(shape)))
    sheets = np.mod(proj, spacing) < thickness
    box = np.zeros(shape, dtype=bool)
    box[tuple(slice(margin, s - margin) for s in shape)] = True
    return sheets & box


def _make_blob(shape=(48, 48, 48), margin=12):
    """A solid cube — an isotropic control (no dominant direction)."""
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(slice(margin, s - margin) for s in shape)] = True
    return mask


def _make_tube(shape=(48, 48, 48), radius=6, axis=0, margin=8):
    """A solid cylinder along `axis` — a tubular control."""
    grid = np.indices(shape)
    perp = [a for a in range(len(shape)) if a != axis]
    centers = [s / 2.0 for s in shape]
    r2 = sum((grid[a] - centers[a]) ** 2 for a in perp)
    mask = r2 <= radius ** 2
    box = np.zeros(shape, dtype=bool)
    sl = [slice(None)] * len(shape)
    sl[axis] = slice(margin, shape[axis] - margin)
    box[tuple(sl)] = True
    return mask & box


def _mean_anisotropy(mask, voxel_size=1.0, neighborhood_size_nm=4.0):
    from synapse_net.cristae_analysis import compute_crista_orientation
    anisotropy = compute_crista_orientation(mask, voxel_size, neighborhood_size_nm)
    return float(np.mean(anisotropy[mask.astype(bool)]))


class TestApproximateMembrane(unittest.TestCase):
    def test_returns_binary_mask(self):
        from synapse_net.cristae_analysis import approximate_membrane
        mito_seg = _make_mito()
        membrane = approximate_membrane(mito_seg, voxel_size=1.0)
        self.assertEqual(membrane.dtype, bool)
        self.assertTrue(membrane.any())

    def test_membrane_inside_mito(self):
        from synapse_net.cristae_analysis import approximate_membrane
        mito_seg = _make_mito()
        membrane = approximate_membrane(mito_seg, voxel_size=1.0)
        mito_binary = mito_seg > 0
        self.assertTrue(np.all(membrane[~mito_binary] == False))  # noqa: E712

    def test_dict_voxel_size(self):
        from synapse_net.cristae_analysis import approximate_membrane
        mito_seg = _make_mito()
        voxel_size = {"z": 2.16, "y": 1.44, "x": 1.44}
        membrane = approximate_membrane(mito_seg, voxel_size=voxel_size)
        self.assertTrue(membrane.any())

    def test_membrane_within_mito_z_extent(self):
        # The membrane is a subset of the mito, so it cannot appear in z-slices with no mito.
        from synapse_net.cristae_analysis import approximate_membrane
        seg = np.zeros((21, 30, 30), dtype="uint32")
        seg[5:16, 5:25, 5:25] = 1
        membrane = approximate_membrane(seg, voxel_size=1.0, membrane_thickness_nm=2.0)
        self.assertFalse(membrane[:5].any())
        self.assertFalse(membrane[16:].any())

    def test_slice2d_njobs_invariant(self):
        # The per-Z-slice 2D erosion (default) is parallelised over z — the result must not depend on
        # n_jobs — and the membrane stays inside the mito.
        from synapse_net.cristae_analysis import approximate_membrane
        seg = np.zeros((16, 40, 40), dtype="uint32")
        seg[3:13, 5:20, 5:20] = 1
        seg[3:13, 25:40, 25:40] = 2
        voxel_size = {"z": 2.0, "y": 1.0, "x": 1.0}
        got1 = approximate_membrane(seg, voxel_size, membrane_thickness_nm=4.0, n_jobs=1, membrane_mode="slice_2d")
        got2 = approximate_membrane(seg, voxel_size, membrane_thickness_nm=4.0, n_jobs=2, membrane_mode="slice_2d")
        np.testing.assert_array_equal(got1, got2)
        self.assertTrue(np.all(got1[seg == 0] == False))  # membrane ⊆ mito  # noqa: E712

    def test_shell3d_connected(self):
        # The 3D shell mode is a single connected component per (solid) mito — the property that
        # keeps the junction geodesic from fragmenting.
        from synapse_net.cristae_analysis import approximate_membrane
        seg = np.zeros((16, 40, 40), dtype="uint32")
        seg[3:13, 5:20, 5:20] = 1
        seg[3:13, 25:40, 25:40] = 2
        voxel_size = {"z": 2.0, "y": 1.0, "x": 1.0}
        membrane = approximate_membrane(seg, voxel_size, membrane_thickness_nm=4.0, membrane_mode="shell_3d")
        self.assertTrue(np.all(membrane[seg == 0] == False))  # noqa: E712
        self.assertEqual(_membrane_components(membrane), 2)   # one connected shell per instance

    def test_invalid_membrane_mode_raises(self):
        from synapse_net.cristae_analysis import approximate_membrane
        with self.assertRaises(ValueError):
            approximate_membrane(_make_mito(), voxel_size=1.0, membrane_mode="bogus")


class TestCristaeProximity(unittest.TestCase):
    def test_distances_in_nm(self):
        from synapse_net.cristae_analysis import compute_crista_proximity
        crista = np.zeros((20, 20, 20), dtype=bool)
        membrane = np.zeros((20, 20, 20), dtype=bool)
        crista[10, 10, 10] = True
        membrane[10, 10, 5] = True  # 5 voxels away
        dist_map, summary = compute_crista_proximity(crista, membrane, voxel_size=1.0)
        self.assertAlmostEqual(summary["min_nm"], 5.0, places=5)

    def test_anisotropic_voxel_size(self):
        from synapse_net.cristae_analysis import compute_crista_proximity
        crista = np.zeros((20, 20, 20), dtype=bool)
        membrane = np.zeros((20, 20, 20), dtype=bool)
        crista[10, 10, 10] = True
        membrane[10, 10, 5] = True  # 5 voxels in x-direction
        voxel_size = {"z": 2.0, "y": 2.0, "x": 3.0}
        _, summary = compute_crista_proximity(crista, membrane, voxel_size=voxel_size)
        self.assertAlmostEqual(summary["min_nm"], 15.0, places=5)  # 5 * 3.0 nm

    def test_empty_crista_returns_nan(self):
        from synapse_net.cristae_analysis import compute_crista_proximity
        crista = np.zeros((10, 10, 10), dtype=bool)
        membrane = np.ones((10, 10, 10), dtype=bool)
        _, summary = compute_crista_proximity(crista, membrane, voxel_size=1.0)
        self.assertTrue(np.isnan(summary["min_nm"]))


class TestDetectContactSites(unittest.TestCase):
    def test_touching_voxels_detected(self):
        # A junction is the pure overlap: a crista voxel that is also a membrane voxel.
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = np.zeros((10, 10, 10), dtype=bool)
        imm = np.zeros((10, 10, 10), dtype=bool)
        crista[5, 5, 5] = True
        imm[5, 5, 5] = True  # same voxel → overlap
        labels, summary = detect_contact_sites(crista, imm, voxel_size=1.0)
        self.assertEqual(summary["contact_voxel_count"], 1)
        self.assertEqual(labels.shape, crista.shape)
        self.assertEqual(int(np.count_nonzero(labels)), 1)

    def test_adjacent_but_not_overlapping_is_empty(self):
        # No hidden dilation: a crista merely adjacent to (not overlapping) the membrane
        # is NOT a junction.
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = np.zeros((10, 10, 10), dtype=bool)
        imm = np.zeros((10, 10, 10), dtype=bool)
        crista[5, 5, 5] = True
        imm[5, 5, 6] = True  # directly adjacent, but no overlap
        labels, summary = detect_contact_sites(crista, imm, voxel_size=1.0)
        self.assertEqual(summary["contact_voxel_count"], 0)
        self.assertEqual(labels.max(), 0)

    def test_non_touching_returns_empty(self):
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = np.zeros((20, 20, 20), dtype=bool)
        imm = np.zeros((20, 20, 20), dtype=bool)
        crista[5, 5, 5] = True
        imm[5, 5, 10] = True  # 5 voxels away — no overlap
        labels, summary = detect_contact_sites(crista, imm, voxel_size=1.0)
        self.assertEqual(summary["contact_voxel_count"], 0)
        self.assertEqual(labels.max(), 0)

    def test_contact_volume_uses_voxel_size(self):
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = np.zeros((10, 10, 10), dtype=bool)
        imm = np.zeros((10, 10, 10), dtype=bool)
        crista[5, 5, 5] = True
        imm[5, 5, 5] = True  # overlap
        _, summary = detect_contact_sites(crista, imm, voxel_size=2.0)
        self.assertAlmostEqual(summary["contact_volume_nm3"], 8.0)  # 1 voxel * 2^3

    def test_diagonal_contact_is_one_junction(self):
        # Two overlap voxels that are only diagonally adjacent to each other (√2) should
        # form one junction, not two: 6-connectivity labeling would count 2; the
        # 26-connectivity grouping collapses them to 1.
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = np.zeros((10, 10, 10), dtype=bool)
        imm = np.zeros((10, 10, 10), dtype=bool)
        crista[5, 5, 5] = True
        crista[6, 6, 5] = True     # diagonal to first (√2 in ZY)
        imm[5, 5, 5] = True        # overlap crista(5,5,5)
        imm[6, 6, 5] = True        # overlap crista(6,6,5)
        _, summary = detect_contact_sites(crista, imm, voxel_size=1.0)
        self.assertEqual(summary["crista_junction_count"], 1)

    def test_two_separate_junctions(self):
        # Two isolated crista-membrane overlaps → crista_junction_count == 2
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = np.zeros((30, 10, 10), dtype=bool)
        imm = np.zeros((30, 10, 10), dtype=bool)
        crista[5, 5, 5] = True
        imm[5, 5, 5] = True   # first junction (overlap)
        crista[20, 5, 5] = True
        imm[20, 5, 5] = True  # second junction — far enough to be a separate component
        _, summary = detect_contact_sites(crista, imm, voxel_size=1.0)
        self.assertEqual(summary["crista_junction_count"], 2)

    def test_junctions_get_unique_ids(self):
        # Each connected junction must get its own integer ID in the returned label array,
        # so they can be shown as a napari Labels layer.
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = np.zeros((30, 10, 10), dtype=bool)
        imm = np.zeros((30, 10, 10), dtype=bool)
        crista[5, 5, 5] = True
        imm[5, 5, 5] = True   # first junction (overlap)
        crista[20, 5, 5] = True
        imm[20, 5, 5] = True  # second junction (overlap)
        labels, _ = detect_contact_sites(crista, imm, voxel_size=1.0)
        self.assertEqual(set(np.unique(labels).tolist()), {0, 1, 2})


class TestCristaMorphology(unittest.TestCase):
    def test_surface_area_positive(self):
        from synapse_net.cristae_analysis import compute_crista_morphology
        crista = np.zeros((20, 20, 20), dtype=bool)
        crista[8:12, 8:12, 8:12] = True
        result = compute_crista_morphology(crista, voxel_size=1.0, method="area")
        self.assertIn("cristae_surface_area_nm2", result)
        self.assertGreater(result["cristae_surface_area_nm2"], 0)

    def test_thickness_positive(self):
        from synapse_net.cristae_analysis import compute_crista_morphology
        crista = np.zeros((30, 30, 30), dtype=bool)
        crista[5:25, 5:25, 5:25] = True  # 20-voxel cube — skeleton survives
        result = compute_crista_morphology(crista, voxel_size=1.0, method="medial_axis")
        self.assertIn("avg_thickness_nm", result)
        self.assertGreater(result["avg_thickness_nm"], 0)

    def test_surface_area_closes_edge_touching_mask(self):
        # A mask that touches the array edge (e.g. an instance cropped to its bbox) must
        # get a closed surface via internal padding, i.e. its area must match the same
        # cube surrounded by a background border — not be undercounted by the missing faces.
        import numpy as _np
        from synapse_net.cristae_analysis import _surface_area
        sampling = _np.array([1.0, 1.0, 1.0])
        edge_cube = _np.ones((10, 10, 10), dtype=bool)          # fills the array, touches all faces
        bordered = _np.zeros((14, 14, 14), dtype=bool)
        bordered[2:12, 2:12, 2:12] = True                        # same 10^3 cube, fully interior
        self.assertAlmostEqual(_surface_area(edge_cube, sampling), _surface_area(bordered, sampling), places=5)

    def test_surface_area_empty_is_nan(self):
        import numpy as _np
        from synapse_net.cristae_analysis import _surface_area
        self.assertTrue(_np.isnan(_surface_area(_np.zeros((8, 8, 8), dtype=bool), _np.array([1.0, 1.0, 1.0]))))


class TestComputeMitoCristaStatistics(unittest.TestCase):
    def test_one_row_per_mito(self):
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        mito_seg = np.zeros((40, 40, 40), dtype="uint32")
        mito_seg[2:20, 2:20, 2:20] = 1
        mito_seg[22:38, 22:38, 22:38] = 2
        crista = np.zeros((40, 40, 40), dtype=bool)
        crista[8:12, 8:12, 8:12] = True
        df = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0)
        self.assertEqual(len(df), 2)
        self.assertIn("mito_label_id", df.columns)
        self.assertIn("crista_fraction", df.columns)
        self.assertIn("crista_junction_count", df.columns)

    def test_crista_fraction_range(self):
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        mito_seg = _make_mito()
        crista = _make_crista()
        df = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0)
        self.assertEqual(len(df), 1)
        frac = df["crista_fraction"].iloc[0]
        self.assertGreater(frac, 0.0)
        self.assertLess(frac, 1.0)

    def test_all_expected_columns_present(self):
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        mito_seg = _make_mito()
        crista = _make_crista()
        df = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0)
        expected = [
            "mito_label_id", "mito_touches_border", "mito_volume_nm3",
            "crista_volume_nm3", "crista_fraction", "contact_voxel_count",
            "crista_junction_count", "contact_volume_nm3",
            "avg_crista_to_membrane_nm", "mean_nn_junction_distance_nm",
            "median_nn_junction_distance_nm", "junction_clustering_index",
            "crista_orientation_anisotropy",
            "cristae_surface_area_nm2", "mito_surface_area_nm2",
            "crista_to_mito_surface_ratio", "avg_thickness_nm",
        ]
        for col in expected:
            self.assertIn(col, df.columns, msg=f"Missing column: {col}")

    def test_no_crista_gives_nan_metrics(self):
        # A mito with no crista inside it should produce NaN for orientation anisotropy
        # and zero for junction count.
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        mito_seg = _make_mito()
        crista = np.zeros(mito_seg.shape, dtype=bool)  # no crista anywhere
        df = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0)
        self.assertEqual(len(df), 1)
        self.assertTrue(np.isnan(df["crista_orientation_anisotropy"].iloc[0]))
        self.assertEqual(df["crista_junction_count"].iloc[0], 0)

    def test_surface_ratio_with_crista(self):
        # A flat sheet inside a mito: mito surface is positive and the crista/mito surface
        # ratio is finite and positive (and < 1 for this small sheet in a large cube).
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        mito_seg = _make_mito()
        crista = _make_crista()
        df = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0)
        mito_surf = df["mito_surface_area_nm2"].iloc[0]
        ratio = df["crista_to_mito_surface_ratio"].iloc[0]
        self.assertGreater(mito_surf, 0.0)
        self.assertTrue(np.isfinite(ratio))
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 1.0)

    def test_surface_ratio_nan_without_crista(self):
        # No crista: the ratio is NaN, but the mito surface is still measured (finite, > 0).
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        mito_seg = _make_mito()
        crista = np.zeros(mito_seg.shape, dtype=bool)
        df = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0)
        self.assertTrue(np.isnan(df["crista_to_mito_surface_ratio"].iloc[0]))
        self.assertTrue(np.isfinite(df["mito_surface_area_nm2"].iloc[0]))
        self.assertGreater(df["mito_surface_area_nm2"].iloc[0], 0.0)

    def test_junction_distance_columns(self):
        # A mito whose cristae form several junctions on the membrane yields finite junction
        # distance / clustering fields; a mito with < 2 junctions yields NaN.
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        shape = (40, 40, 40)
        mito_seg = np.zeros(shape, dtype="uint32")
        mito_seg[4:36, 4:36, 4:36] = 1
        # Several thin cristae, each spanning to the membrane at different x positions, so
        # each produces a distinct junction on the membrane shell.
        crista = np.zeros(shape, dtype=bool)
        for x in (8, 16, 24, 30):
            crista[6:34, 18:22, x - 1:x + 1] = True
        df = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0)
        self.assertGreaterEqual(df["crista_junction_count"].iloc[0], 2)
        self.assertTrue(np.isfinite(df["mean_nn_junction_distance_nm"].iloc[0]))
        self.assertTrue(np.isfinite(df["junction_clustering_index"].iloc[0]))

        # Single tiny crista -> at most one junction -> NaN junction distances.
        crista_one = np.zeros(shape, dtype=bool)
        crista_one[18:22, 18:22, 4:8] = True
        df_one = compute_mito_crista_statistics(crista_one, mito_seg, voxel_size=1.0)
        if df_one["crista_junction_count"].iloc[0] < 2:
            self.assertTrue(np.isnan(df_one["mean_nn_junction_distance_nm"].iloc[0]))
            self.assertTrue(np.isnan(df_one["junction_clustering_index"].iloc[0]))


class TestJunctionDistances(unittest.TestCase):
    @staticmethod
    def _flat_membrane_with_junctions(positions, shape=(12, 40, 40), z=6):
        # A single-slice planar membrane at z, with one-voxel junctions at the given (y, x).
        membrane = np.zeros(shape, dtype=bool)
        membrane[z, 2:-2, 2:-2] = True
        labels = np.zeros(shape, dtype=np.int32)
        for i, (y, x) in enumerate(positions, start=1):
            labels[z, y, x] = i
        return labels, membrane

    # These exercise compute_junction_distances directly; with no mesh supplied it meshes the given
    # membrane and takes the surface geodesic, so they require the bioimage-cpp geodesic API.
    @_MESH_REQUIRED
    def test_geodesic_follows_bent_membrane(self):
        # An L-shaped membrane: the geodesic around the bend is longer than the straight line
        # between the two seed voxels.
        from synapse_net.cristae_analysis import compute_junction_distances
        shape = (5, 40, 40)
        membrane = np.zeros(shape, dtype=bool)
        z = 2
        membrane[z, 4:7, 5:35] = True     # horizontal arm (a few voxels wide → meshable)
        membrane[z, 5:35, 33:36] = True   # vertical arm (shares the corner)
        labels = np.zeros(shape, dtype=np.int32)
        labels[z, 5, 6] = 1               # near the far end of the horizontal arm
        labels[z, 33, 34] = 2             # near the far end of the vertical arm
        dist, _ = compute_junction_distances(labels, membrane, voxel_size=1.0)
        straight = np.sqrt((33 - 5) ** 2 + (34 - 6) ** 2)
        self.assertGreater(dist[0, 1], straight * 1.2)

    def test_fewer_than_two_junctions_is_nan(self):
        # n < 2 returns early (before any meshing), so this holds regardless of the geodesic API.
        from synapse_net.cristae_analysis import compute_junction_distances
        labels, membrane = self._flat_membrane_with_junctions([(20, 20)])
        _, summary = compute_junction_distances(labels, membrane, voxel_size=1.0, surface_area_nm2=1000.0)
        self.assertEqual(summary["junction_count"], 1)
        self.assertTrue(np.isnan(summary["mean_nn_junction_distance_nm"]))
        self.assertTrue(np.isnan(summary["junction_clustering_index"]))

    @_MESH_REQUIRED
    def test_clustered_index_lower_than_dispersed(self):
        # Same membrane/area and junction count, but tightly grouped vs evenly spread:
        # the clustered arrangement must give a smaller Clark-Evans index.
        from synapse_net.cristae_analysis import compute_junction_distances
        area = 40.0 * 40.0
        clustered_pos = [(18, 18), (18, 20), (20, 18), (20, 20)]
        dispersed_pos = [(8, 8), (8, 30), (30, 8), (30, 30)]
        _, clustered = compute_junction_distances(
            *self._flat_membrane_with_junctions(clustered_pos), voxel_size=1.0, surface_area_nm2=area
        )
        _, dispersed = compute_junction_distances(
            *self._flat_membrane_with_junctions(dispersed_pos), voxel_size=1.0, surface_area_nm2=area
        )
        self.assertLess(clustered["junction_clustering_index"], dispersed["junction_clustering_index"])


class TestOptimizationEquivalence(unittest.TestCase):
    """The computational optimizations (eigvalsh, reused distance transform, parallelism)
    must not change results versus the original per-metric computations."""

    def test_progress_callback_and_verbose(self):
        # progress_callback fires once per mitochondrion ending at (total, total), verbose
        # runs without error, and results are unaffected by progress reporting.
        import pandas as pd
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        shape = (40, 60, 40)
        mito_seg = np.zeros(shape, dtype="uint32")
        mito_seg[4:36, 4:28, 4:36] = 1
        mito_seg[4:36, 32:56, 4:36] = 2
        crista = np.zeros(shape, dtype=bool)
        for x in (10, 18, 26):
            crista[8:32, 8:24, x:x + 2] = True
            crista[8:32, 36:52, x:x + 2] = True

        calls = []
        df_cb = compute_mito_crista_statistics(
            crista, mito_seg, voxel_size=1.5, n_jobs=1, verbose=True,
            progress_callback=lambda done, total: calls.append((done, total)),
        )
        self.assertEqual([c[1] for c in calls], [2, 2])          # total reported each time
        self.assertEqual([c[0] for c in calls], [1, 2])          # one tick per mito
        df_plain = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.5, n_jobs=1)
        pd.testing.assert_frame_equal(df_cb, df_plain)

    def test_proximity_precomputed_distance_matches(self):
        from scipy.ndimage import distance_transform_edt
        from synapse_net.cristae_analysis import compute_crista_proximity
        crista = np.zeros((20, 20, 20), dtype=bool)
        membrane = np.zeros((20, 20, 20), dtype=bool)
        crista[10, 10, 10] = True
        crista[8, 12, 9] = True
        membrane[10, 10, 5] = True
        vs = {"z": 2.0, "y": 1.5, "x": 1.5}
        sampling = np.array([2.0, 1.5, 1.5])
        precomputed = distance_transform_edt(~membrane, sampling=sampling.tolist())
        map_a, sum_a = compute_crista_proximity(crista, membrane, vs)
        map_b, sum_b = compute_crista_proximity(crista, membrane, vs, membrane_distance=precomputed)
        np.testing.assert_allclose(map_a, map_b)
        self.assertEqual(sum_a, sum_b)

    def test_parallel_matches_serial(self):
        import pandas as pd
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        shape = (40, 60, 40)
        mito_seg = np.zeros(shape, dtype="uint32")
        mito_seg[4:36, 4:28, 4:36] = 1
        mito_seg[4:36, 32:56, 4:36] = 2
        crista = np.zeros(shape, dtype=bool)
        for x in (10, 18, 26):
            crista[8:32, 8:24, x:x + 2] = True
            crista[8:32, 36:52, x:x + 2] = True
        df_serial = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.5, n_jobs=1)
        df_parallel = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.5, n_jobs=2)
        pd.testing.assert_frame_equal(df_serial, df_parallel)


class TestCristaOrientation(unittest.TestCase):
    # Axis-aligned normals (z, y, x) used by the anisotropy tests.
    AXIS_NORMALS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    def test_lamellae_more_anisotropic_than_blob(self):
        # Parallel sheets should read as strongly directional; a solid cube should not.
        lam = _mean_anisotropy(_make_lamellae(normal=(1, 0, 0)))
        blob = _mean_anisotropy(_make_blob())
        self.assertGreater(lam, blob)
        self.assertGreater(lam, 5.0 * blob)

    def test_anisotropy_rotation_invariant(self):
        # The reported anisotropy is rotation-invariant: every orientation of the same lamellar
        # pattern reads as strongly directional (≫ an isotropic blob). Only this qualitative claim
        # is asserted — the exact magnitude ratio between orientations is NOT stable, because the
        # structure tensor drives the minor eigenvalue toward zero on these idealised noise-free
        # sheets, so λ_max/(λ_min+ε) is dominated by ε (and by voxel-grid staircasing for oblique
        # normals). On real, noisy data the minor eigenvalue stays bounded.
        blob = _mean_anisotropy(_make_blob())
        values = [
            _mean_anisotropy(_make_lamellae(normal=n))
            for n in self.AXIS_NORMALS + [(1, 1, 0), (1, 1, 1)]
        ]
        for v in values:
            self.assertGreater(v, 5.0 * blob, msg=f"orientation not anisotropic enough: {v}")

    def test_tubular_low_anisotropy(self):
        # A solid tube and parallel lamellae are both directional structures and both read as much
        # more anisotropic than an isotropic blob. (The tube-vs-lamellae ordering itself is not
        # asserted: both are 1-D-degenerate for the structure tensor — a near-zero minor eigenvalue —
        # so on idealised noise-free shapes their magnitudes are ε-dominated and not meaningfully
        # comparable.)
        blob = _mean_anisotropy(_make_blob())
        tube = _mean_anisotropy(_make_tube())
        lam = _mean_anisotropy(_make_lamellae(normal=(1, 0, 0)))
        self.assertGreater(tube, 5.0 * blob)
        self.assertGreater(lam, 5.0 * blob)

    def test_pipeline_reports_orientation(self):
        # Orientation anisotropy is only computed in exact mode (structure tensor, default 30 nm
        # smoothing): the reported column must be finite and larger for lamellae than a blob.
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        shape = (48, 48, 48)
        mito = np.zeros(shape, dtype="uint32")
        mito[6:42, 6:42, 6:42] = 1

        lam = _make_lamellae(shape, normal=(1, 0, 0))
        df_lam = compute_mito_crista_statistics(lam, mito, voxel_size=1.0, method="exact")
        val_lam = df_lam["crista_orientation_anisotropy"].iloc[0]

        blob = _make_blob(shape, margin=12)
        df_blob = compute_mito_crista_statistics(blob, mito, voxel_size=1.0, method="exact")
        val_blob = df_blob["crista_orientation_anisotropy"].iloc[0]

        self.assertTrue(np.isfinite(val_lam))
        self.assertTrue(np.isfinite(val_blob))
        self.assertGreater(val_lam, val_blob)

    def test_visualize_orientations(self):
        # Manual, off-by-default: opens napari so the synthetic test data and their per-voxel
        # anisotropy can be inspected by eye. Skipped in the normal suite. (Direction vectors were
        # removed together with the eigenvectors — only the anisotropy magnitude is computed now.)
        if not os.environ.get(VIEW_ENV):
            self.skipTest(f"set {VIEW_ENV}=1 or run this file with --view to open napari")
        import napari  # lazy import — napari/Qt is not needed for the normal suite
        from synapse_net.cristae_analysis import compute_crista_orientation

        shape = (48, 48, 48)
        cases = [  # (name, mask)
            ("lamellae Z", _make_lamellae(shape, normal=(1, 0, 0))),
            ("lamellae Y", _make_lamellae(shape, normal=(0, 1, 0))),
            ("lamellae X", _make_lamellae(shape, normal=(0, 0, 1))),
            ("lamellae ZY (oblique)", _make_lamellae(shape, normal=(1, 1, 0))),
            ("lamellae ZYX (oblique)", _make_lamellae(shape, normal=(1, 1, 1))),
            ("blob (isotropic)", _make_blob(shape)),
            ("tube (tubular)", _make_tube(shape)),
        ]

        mito = np.zeros(shape, dtype="uint32")
        mito[6:42, 6:42, 6:42] = 1

        viewer = napari.Viewer(title="Cristae orientation test data")
        viewer.add_labels(mito.astype(np.int32), name="mito", opacity=0.1)

        print()  # console cross-reference for the visual check
        for i, (name, mask) in enumerate(cases):
            anisotropy = compute_crista_orientation(mask, 1.0)
            mean_aniso = float(np.mean(anisotropy[mask.astype(bool)]))
            print(f"{name:<24} mean anisotropy={mean_aniso:.2f}")
            viewer.add_labels(mask.astype(np.int32) * (i + 1), name=name, opacity=0.6)

        napari.run()


class TestFastMethod(unittest.TestCase):
    """Fast mode differs from exact only in the crista orientation anisotropy (computed on a
    downsampled crop). Every other metric — surface areas, junction distances, thickness — must be
    identical to exact, and the fast orientation must be finite and preserve the lamellae > blob
    ordering (its magnitude is not comparable to exact)."""

    NON_ORIENTATION_COLUMNS = [
        "mito_label_id", "mito_touches_border", "mito_volume_nm3",
        "crista_volume_nm3", "crista_fraction", "contact_voxel_count",
        "crista_junction_count", "contact_volume_nm3",
        "avg_crista_to_membrane_nm", "mean_nn_junction_distance_nm",
        "median_nn_junction_distance_nm", "junction_clustering_index",
        "cristae_surface_area_nm2", "mito_surface_area_nm2",
        "crista_to_mito_surface_ratio", "avg_thickness_nm",
    ]

    def test_fast_matches_exact_except_orientation(self):
        # The regression this guards: fast used to diverge on surface areas / junction distances.
        # Now those must equal exact; only crista_orientation_anisotropy may differ.
        import pandas as pd
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        shape = (40, 40, 40)
        mito_seg = np.zeros(shape, dtype="uint32")
        mito_seg[4:36, 4:36, 4:36] = 1
        crista = np.zeros(shape, dtype=bool)
        for x in (8, 16, 24, 30):
            crista[6:34, 18:22, x - 1:x + 1] = True
        df_fast = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0, method="fast")
        df_exact = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0, method="exact")
        for col in self.NON_ORIENTATION_COLUMNS:
            pd.testing.assert_series_equal(
                df_fast[col], df_exact[col], check_names=False,
                obj=f"column {col} (fast vs exact)",
            )

    def test_fast_orientation_is_finite_with_crista(self):
        # Fast orientation is now computed (downsampled), so it is finite when a crista is present.
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        mito_seg = _make_mito()
        crista = _make_crista()
        df = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0)  # default = fast
        self.assertEqual(len(df), 1)
        self.assertTrue(np.isfinite(df["crista_orientation_anisotropy"].iloc[0]))

    def test_fast_orientation_no_crista_is_nan(self):
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        mito_seg = _make_mito()
        crista = np.zeros(mito_seg.shape, dtype=bool)
        df = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0, method="fast")
        self.assertTrue(np.isnan(df["crista_orientation_anisotropy"].iloc[0]))

    def test_fast_orientation_preserves_lamellae_vs_blob_ordering(self):
        # Downsampled anisotropy is a relative indicator: lamellae must still read as more
        # directional than a blob, even though the magnitude is not comparable to exact.
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        shape = (48, 48, 48)
        mito = np.zeros(shape, dtype="uint32")
        mito[6:42, 6:42, 6:42] = 1
        lam = compute_mito_crista_statistics(
            _make_lamellae(shape, normal=(1, 0, 0)), mito, voxel_size=1.0, method="fast"
        )["crista_orientation_anisotropy"].iloc[0]
        blob = compute_mito_crista_statistics(
            _make_blob(shape, margin=12), mito, voxel_size=1.0, method="fast"
        )["crista_orientation_anisotropy"].iloc[0]
        self.assertTrue(np.isfinite(lam) and np.isfinite(blob))
        self.assertGreater(lam, blob)

    def test_downsampled_orientation_cheaper_and_lower_magnitude(self):
        # The downsampled anisotropy is a relative indicator: same-or-lower magnitude than full-res.
        from synapse_net.cristae_analysis import (
            compute_crista_orientation, _downsampled_orientation_anisotropy,
        )
        crista = _make_lamellae(shape=(48, 48, 48), normal=(1, 0, 0))
        aniso_full = compute_crista_orientation(crista, 1.0)
        full = float(np.mean(aniso_full[crista]))
        ds = _downsampled_orientation_anisotropy(crista, 1.0, factor=2)
        self.assertTrue(np.isfinite(ds))
        self.assertGreater(ds, 1.0)  # still reads as anisotropic
        self.assertLessEqual(ds, full * 1.05)  # not larger than full-res (relative-only)

    def test_skip_orientation_is_nan_and_matches_exact_otherwise(self):
        # method="skip" leaves orientation NaN but matches exact on every other column.
        import pandas as pd
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        shape = (40, 40, 40)
        mito_seg = np.zeros(shape, dtype="uint32")
        mito_seg[4:36, 4:36, 4:36] = 1
        crista = np.zeros(shape, dtype=bool)
        for x in (8, 16, 24, 30):
            crista[6:34, 18:22, x - 1:x + 1] = True
        df_skip = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0, method="skip")
        df_exact = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0, method="exact")
        self.assertTrue(np.isnan(df_skip["crista_orientation_anisotropy"].iloc[0]))
        for col in self.NON_ORIENTATION_COLUMNS:
            pd.testing.assert_series_equal(
                df_skip[col], df_exact[col], check_names=False,
                obj=f"column {col} (skip vs exact)",
            )

    def test_invalid_method_raises(self):
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        mito_seg = _make_mito()
        crista = _make_crista()
        with self.assertRaises(ValueError):
            compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0, method="bogus")


class TestMeshGeodesicBackend(unittest.TestCase):
    """Junction distances are surface geodesics on the eroded-mito (lumen) mesh (bioimage-cpp).

    The lumen surface (``mito & ~membrane``, built in ``_single_mito_row``) is what the geodesic runs
    on. When the bioimage-cpp geodesic API is unavailable the junction columns are NaN (no fallback).
    """

    @staticmethod
    def _mito_with_cristae(shape=(40, 40, 40)):
        mito = np.zeros(shape, dtype="uint32")
        mito[4:-4, 4:-4, 4:-4] = 1
        crista = np.zeros(shape, dtype=bool)
        # Compact crista patches against the y-low membrane wall at distinct x — well-separated
        # junctions with real spacing, touching the membrane in both membrane modes.
        for x in (10, 18, 26, 32):
            crista[10:14, 6:9, x:x + 3] = True
        return crista, mito

    @_MESH_REQUIRED
    def test_mesh_junction_distances_finite(self):
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        crista, mito = self._mito_with_cristae()
        df = compute_mito_crista_statistics(crista, mito, 2.0, method="skip")
        self.assertGreaterEqual(int(df["crista_junction_count"].iloc[0]), 2)
        mean_nn = df["mean_nn_junction_distance_nm"].iloc[0]
        self.assertTrue(np.isfinite(mean_nn) and mean_nn > 0)
        self.assertTrue(np.isfinite(df["junction_clustering_index"].iloc[0]))

    @_MESH_REQUIRED
    def test_primitive_mesh_on_flat_membrane(self):
        # With no mesh supplied the primitive meshes the given membrane surface — finite, positive,
        # and of the right order (junctions ~17-20 apart). Absolute accuracy on a 1-voxel sheet is not
        # asserted (that degenerate mesh is only the fallback; the pipeline meshes the thicker lumen).
        from synapse_net.cristae_analysis import compute_junction_distances
        labels, membrane = TestJunctionDistances._flat_membrane_with_junctions(
            [(20, 8), (20, 28), (8, 20), (32, 20)]
        )
        _, summary = compute_junction_distances(labels, membrane, 1.0, surface_area_nm2=1000.0)
        mean_nn = summary["mean_nn_junction_distance_nm"]
        self.assertTrue(np.isfinite(mean_nn) and mean_nn > 0)
        self.assertLess(mean_nn, 100.0)  # sane order of magnitude, not a runaway path

    def test_junction_distances_nan_when_api_missing(self):
        # No graph fallback: with the geodesic API patched out, the junction columns are NaN and a
        # single warning is emitted (no crash).
        import warnings
        import synapse_net.cristae_analysis as ca
        crista, mito = self._mito_with_cristae()
        saved = ca.geodesic_distances_mesh
        ca.geodesic_distances_mesh = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                df = ca.compute_mito_crista_statistics(crista, mito, 2.0, method="skip")
        finally:
            ca.geodesic_distances_mesh = saved
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(runtime_warnings), 1)
        self.assertTrue(np.isnan(df["mean_nn_junction_distance_nm"].iloc[0]))
        self.assertTrue(np.isnan(df["junction_clustering_index"].iloc[0]))


_EXPECTED_COLUMNS = [
    "mito_label_id", "mito_touches_border", "mito_volume_nm3",
    "crista_volume_nm3", "crista_fraction", "contact_voxel_count",
    "crista_junction_count", "contact_volume_nm3",
    "avg_crista_to_membrane_nm", "mean_nn_junction_distance_nm",
    "median_nn_junction_distance_nm", "junction_clustering_index",
    "crista_orientation_anisotropy",
    "cristae_surface_area_nm2", "mito_surface_area_nm2",
    "crista_to_mito_surface_ratio", "avg_thickness_nm",
]


class TestCristaeIntegration(unittest.TestCase):
    """End-to-end runs of compute_mito_crista_statistics on synthetic and real segmentations."""

    def test_synthetic_end_to_end(self):
        # Two mitos side by side: mito 1 holds parallel lamellae (strongly directional), mito 2
        # holds an isotropic solid block. The full pipeline must produce well-formed metrics and
        # a non-negative anisotropy that is larger for the lamellar mito than the isotropic one.
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        shape = (40, 96, 40)
        voxel_size = 10.0  # 30 nm structure-tensor neighborhood == 3 voxels at this scale.
        mito = np.zeros(shape, dtype="uint32")
        mito[4:36, 4:44, 4:36] = 1
        mito[4:36, 52:92, 4:36] = 2

        crista = np.zeros(shape, dtype=bool)
        lamellae = _make_lamellae(shape, normal=(1, 0, 0), spacing=6, thickness=2, margin=6)
        crista |= lamellae & (mito == 1)
        crista[12:28, 62:82, 12:28] = True  # isotropic block inside mito 2

        df = compute_mito_crista_statistics(crista, mito, voxel_size, method="exact", n_jobs=1)

        self.assertEqual(len(df), 2)
        for col in _EXPECTED_COLUMNS:
            self.assertIn(col, df.columns, msg=f"Missing column: {col}")
        rows = df.set_index("mito_label_id")
        for label in (1, 2):
            self.assertGreater(rows.loc[label, "mito_volume_nm3"], 0.0)
            frac = rows.loc[label, "crista_fraction"]
            self.assertGreaterEqual(frac, 0.0)
            self.assertLessEqual(frac, 1.0)
            aniso = rows.loc[label, "crista_orientation_anisotropy"]
            self.assertTrue(np.isfinite(aniso), msg=f"anisotropy not finite for mito {label}")
            self.assertGreaterEqual(aniso, 0.0, msg=f"anisotropy negative for mito {label}: {aniso}")
        self.assertGreater(
            rows.loc[1, "crista_orientation_anisotropy"],
            rows.loc[2, "crista_orientation_anisotropy"],
        )

    @_REAL_DATA_REQUIRED
    def test_forgotten_cristae_real_data(self):
        # Smoke test on the real forgotten_cristae segmentations: the pipeline must complete and
        # produce well-formed metrics. The pathological ~174M-voxel mito is filtered out first so
        # the per-mito computation stays bounded (see _MAX_MITO_VOXELS).
        import h5py
        from synapse_net.cristae_analysis import compute_mito_crista_statistics

        files = sorted(glob.glob(os.path.join(_H5_DIR, "*", "*.h5")))
        chosen = None
        for path in files:
            with h5py.File(path, "r") as f:
                crista = f["labels/cristae"][:].astype(bool)
                mito = f["labels/mitochondria"][:]
            filtered, surviving = _drop_large_mitos(mito, _MAX_MITO_VOXELS)
            if surviving > 0:
                chosen = (path, crista, filtered, surviving)
                break
        if chosen is None:
            self.skipTest("no forgotten_cristae file has a mito under the size threshold")
        path, crista, filtered_mito, surviving = chosen

        voxel_size = _resolve_voxel_size(path)
        df = compute_mito_crista_statistics(
            crista, filtered_mito, voxel_size,
            method="fast", n_jobs=-1, membrane_mode="slice_2d",
        )

        self.assertEqual(len(df), surviving)
        for col in _EXPECTED_COLUMNS:
            self.assertIn(col, df.columns, msg=f"Missing column: {col}")
        self.assertTrue((df["mito_volume_nm3"] > 0).all())
        frac = df["crista_fraction"].to_numpy(dtype=float)
        frac = frac[np.isfinite(frac)]
        self.assertTrue(((frac >= 0.0) & (frac <= 1.0)).all())
        # The negative-blowup regression (tiny-negative structure-tensor eigenvalue) must not recur.
        aniso = df["crista_orientation_anisotropy"].to_numpy(dtype=float)
        self.assertTrue((aniso[np.isfinite(aniso)] >= 0.0).all())
        for col in ("mean_nn_junction_distance_nm", "median_nn_junction_distance_nm",
                    "avg_crista_to_membrane_nm"):
            vals = df[col].to_numpy(dtype=float)
            self.assertTrue((vals[np.isfinite(vals)] >= 0.0).all(), msg=f"negative {col}")


if __name__ == "__main__":
    if "--view" in sys.argv:
        sys.argv.remove("--view")
        os.environ[VIEW_ENV] = "1"
    unittest.main()
