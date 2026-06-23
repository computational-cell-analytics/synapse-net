import unittest

import numpy as np


def _make_mito(shape=(32, 32, 32), label=1):
    seg = np.zeros(shape, dtype="uint32")
    seg[4:-4, 4:-4, 4:-4] = label
    return seg


def _make_crista(shape=(32, 32, 32)):
    mask = np.zeros(shape, dtype=bool)
    mask[10:14, 8:24, 8:24] = True  # a flat sheet inside the mito
    return mask


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

    def test_no_z_bleed(self):
        # Mito exists only at z=5..15; slices 0..4 and 16..20 are empty.
        # The 2D per-slice erosion must not produce membrane voxels outside the
        # z-range where the mito is present (a 3D ball would bleed into neighbours).
        from synapse_net.cristae_analysis import approximate_membrane
        seg = np.zeros((21, 30, 30), dtype="uint32")
        seg[5:16, 5:25, 5:25] = 1
        membrane = approximate_membrane(seg, voxel_size=1.0, membrane_thickness_nm=2.0)
        self.assertFalse(membrane[:5].any(),  "membrane bled into z < 5")
        self.assertFalse(membrane[16:].any(), "membrane bled into z > 15")


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
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = np.zeros((10, 10, 10), dtype=bool)
        imm = np.zeros((10, 10, 10), dtype=bool)
        crista[5, 5, 5] = True
        imm[5, 5, 6] = True  # directly adjacent
        coords, summary = detect_contact_sites(crista, imm, voxel_size=1.0)
        self.assertEqual(summary["contact_voxel_count"], 1)
        self.assertEqual(coords.shape, (1, 3))

    def test_non_touching_returns_empty(self):
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = np.zeros((20, 20, 20), dtype=bool)
        imm = np.zeros((20, 20, 20), dtype=bool)
        crista[5, 5, 5] = True
        imm[5, 5, 10] = True  # 4 voxels away — not touching
        coords, summary = detect_contact_sites(crista, imm, voxel_size=1.0)
        self.assertEqual(summary["contact_voxel_count"], 0)

    def test_contact_volume_uses_voxel_size(self):
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = np.zeros((10, 10, 10), dtype=bool)
        imm = np.zeros((10, 10, 10), dtype=bool)
        crista[5, 5, 5] = True
        imm[5, 5, 6] = True
        _, summary = detect_contact_sites(crista, imm, voxel_size=2.0)
        self.assertAlmostEqual(summary["contact_volume_nm3"], 8.0)  # 1 voxel * 2^3

    def test_diagonal_contact_is_one_junction(self):
        # Two crista voxels that are each face-adjacent to the membrane but only
        # diagonally adjacent to each other (√2) should form one junction, not two.
        # With 6-connectivity labeling they would be counted as 2; the 26-connectivity
        # fix collapses them to 1.
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = np.zeros((10, 10, 10), dtype=bool)
        imm = np.zeros((10, 10, 10), dtype=bool)
        crista[5, 5, 5] = True
        crista[6, 6, 5] = True     # diagonal to first (√2 in ZY), not face-adjacent
        imm[5, 5, 4] = True     # face-adjacent to crista(5,5,5) via X
        imm[6, 6, 4] = True     # face-adjacent to crista(6,6,5) via X
        _, summary = detect_contact_sites(crista, imm, voxel_size=1.0)
        self.assertEqual(summary["crista_junction_count"], 1)

    def test_two_separate_junctions(self):
        # Two isolated crista blobs each touching the membrane → crista_junction_count == 2
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = np.zeros((30, 10, 10), dtype=bool)
        imm = np.zeros((30, 10, 10), dtype=bool)
        crista[5, 5, 5] = True
        imm[5, 5, 6] = True   # first junction
        crista[20, 5, 5] = True
        imm[20, 5, 6] = True  # second junction — far enough to be a separate component
        _, summary = detect_contact_sites(crista, imm, voxel_size=1.0)
        self.assertEqual(summary["crista_junction_count"], 2)


class TestCristaMorphology(unittest.TestCase):
    def test_surface_area_positive(self):
        from synapse_net.cristae_analysis import compute_crista_morphology
        crista = np.zeros((20, 20, 20), dtype=bool)
        crista[8:12, 8:12, 8:12] = True
        result = compute_crista_morphology(crista, voxel_size=1.0, method="area")
        self.assertIn("total_surface_area_nm2", result)
        self.assertGreater(result["total_surface_area_nm2"], 0)

    def test_thickness_positive(self):
        from synapse_net.cristae_analysis import compute_crista_morphology
        crista = np.zeros((30, 30, 30), dtype=bool)
        crista[5:25, 5:25, 5:25] = True  # 20-voxel cube — skeleton survives
        result = compute_crista_morphology(crista, voxel_size=1.0, method="medial_axis")
        self.assertIn("avg_thickness_nm", result)
        self.assertGreater(result["avg_thickness_nm"], 0)


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
            "avg_crista_to_membrane_nm", "crista_orientation_anisotropy",
            "total_surface_area_nm2", "avg_thickness_nm",
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


if __name__ == "__main__":
    unittest.main()
