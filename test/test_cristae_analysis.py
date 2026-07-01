import os
import sys
import unittest

import numpy as np

# Set SYNAPSE_NET_VIEW=1 (or run this file with --view) to open the orientation cases in
# napari — see TestCristaOrientation.test_visualize_orientations.
VIEW_ENV = "SYNAPSE_NET_VIEW"


def _make_mito(shape=(32, 32, 32), label=1):
    seg = np.zeros(shape, dtype="uint32")
    seg[4:-4, 4:-4, 4:-4] = label
    return seg


def _make_crista(shape=(32, 32, 32)):
    mask = np.zeros(shape, dtype=bool)
    mask[10:14, 8:24, 8:24] = True  # a flat sheet inside the mito
    return mask


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
    _, _, anisotropy = compute_crista_orientation(mask, voxel_size, neighborhood_size_nm)
    return float(np.mean(anisotropy[mask.astype(bool)]))


def _dominant_direction(mask, voxel_size=1.0, neighborhood_size_nm=4.0):
    """Robust aggregate of the per-voxel major eigenvector over the mask.

    Eigenvectors carry an arbitrary sign, so we average the outer products v·vᵀ (which are
    sign-invariant) and return the top eigenvector of that mean tensor.
    """
    from synapse_net.cristae_analysis import compute_crista_orientation
    _, eigenvectors, _ = compute_crista_orientation(mask, voxel_size, neighborhood_size_nm)
    major = eigenvectors[..., :, -1]           # eigenvector for the largest eigenvalue
    vecs = major[mask.astype(bool)]            # (N, ndim)
    tensor = np.einsum("ni,nj->ij", vecs, vecs) / len(vecs)
    _, agg = np.linalg.eigh(tensor)
    return agg[:, -1]


def _abs_cos(u, v):
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    return abs(float(np.dot(u, v)))


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
            "avg_crista_to_membrane_nm", "crista_orientation_anisotropy",
            "total_surface_area_nm2", "mito_surface_area_nm2",
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


class TestCristaOrientation(unittest.TestCase):
    # Axis-aligned normals (z, y, x) and oblique normals used across the direction tests.
    AXIS_NORMALS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    OBLIQUE_NORMALS = [(1, 1, 0), (1, 1, 1), (2, 1, 0)]

    def test_lamellae_more_anisotropic_than_blob(self):
        # Parallel sheets should read as strongly directional; a solid cube should not.
        lam = _mean_anisotropy(_make_lamellae(normal=(1, 0, 0)))
        blob = _mean_anisotropy(_make_blob())
        self.assertGreater(lam, blob)
        self.assertGreater(lam, 5.0 * blob)

    def test_anisotropy_rotation_invariant(self):
        # The reported anisotropy scalar is rotation-invariant: every orientation of the
        # same lamellar pattern should read as strongly directional, within a loose factor.
        blob = _mean_anisotropy(_make_blob())
        values = [
            _mean_anisotropy(_make_lamellae(normal=n))
            for n in self.AXIS_NORMALS + [(1, 1, 0), (1, 1, 1)]
        ]
        for v in values:
            self.assertGreater(v, 5.0 * blob, msg=f"orientation not anisotropic enough: {v}")
        self.assertLess(max(values) / min(values), 20.0, msg=f"anisotropy varies too much: {values}")

    def test_dominant_eigenvector_axis_aligned(self):
        # For sheets stacked along a coordinate axis, the dominant direction (the lamellae
        # normal) must align with that axis.
        for normal in self.AXIS_NORMALS:
            direction = _dominant_direction(_make_lamellae(normal=normal))
            self.assertGreater(
                _abs_cos(direction, normal), 0.9,
                msg=f"dominant direction {direction} not aligned with {normal}",
            )

    def test_dominant_eigenvector_oblique(self):
        # For obliquely oriented sheets the dominant direction must still track the known
        # normal (looser threshold, since oblique sheets are staircased on the voxel grid).
        for normal in self.OBLIQUE_NORMALS:
            direction = _dominant_direction(_make_lamellae(normal=normal))
            self.assertGreater(
                _abs_cos(direction, normal), 0.8,
                msg=f"dominant direction {direction} not aligned with oblique {normal}",
            )

    def test_tubular_low_anisotropy(self):
        # A solid tube is less directional than parallel lamellae.
        lam = _mean_anisotropy(_make_lamellae(normal=(1, 0, 0)))
        tube = _mean_anisotropy(_make_tube())
        self.assertLess(tube, lam)

    def test_pipeline_reports_orientation(self):
        # Exercise the exact widget code path (compute_mito_crista_statistics, default 30 nm
        # smoothing): the reported column must be finite and larger for lamellae than a blob.
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        shape = (48, 48, 48)
        mito = np.zeros(shape, dtype="uint32")
        mito[6:42, 6:42, 6:42] = 1

        lam = _make_lamellae(shape, normal=(1, 0, 0))
        df_lam = compute_mito_crista_statistics(lam, mito, voxel_size=1.0)
        val_lam = df_lam["crista_orientation_anisotropy"].iloc[0]

        blob = _make_blob(shape, margin=12)
        df_blob = compute_mito_crista_statistics(blob, mito, voxel_size=1.0)
        val_blob = df_blob["crista_orientation_anisotropy"].iloc[0]

        self.assertTrue(np.isfinite(val_lam))
        self.assertTrue(np.isfinite(val_blob))
        self.assertGreater(val_lam, val_blob)

    def test_visualize_orientations(self):
        # Manual, off-by-default: opens napari so the synthetic test data and the computed
        # dominant-orientation vectors can be inspected by eye. Skipped in the normal suite.
        if not os.environ.get(VIEW_ENV):
            self.skipTest(f"set {VIEW_ENV}=1 or run this file with --view to open napari")
        import napari  # lazy import — napari/Qt is not needed for the normal suite

        shape = (48, 48, 48)
        cases = [  # (name, mask, expected_normal)
            ("lamellae Z", _make_lamellae(shape, normal=(1, 0, 0)), (1, 0, 0)),
            ("lamellae Y", _make_lamellae(shape, normal=(0, 1, 0)), (0, 1, 0)),
            ("lamellae X", _make_lamellae(shape, normal=(0, 0, 1)), (0, 0, 1)),
            ("lamellae ZY (oblique)", _make_lamellae(shape, normal=(1, 1, 0)), (1, 1, 0)),
            ("lamellae ZYX (oblique)", _make_lamellae(shape, normal=(1, 1, 1)), (1, 1, 1)),
            ("blob (isotropic)", _make_blob(shape), None),
            ("tube (tubular)", _make_tube(shape), None),
        ]

        mito = np.zeros(shape, dtype="uint32")
        mito[6:42, 6:42, 6:42] = 1

        viewer = napari.Viewer(title="Cristae orientation test data")
        viewer.add_labels(mito.astype(np.int32), name="mito", opacity=0.1)

        print()  # console cross-reference for the visual check
        for i, (name, mask, normal) in enumerate(cases):
            direction = _dominant_direction(mask)
            cos = _abs_cos(direction, normal) if normal is not None else None
            cos_str = f"{cos:.3f}" if cos is not None else "n/a"
            print(f"{name:<24} dominant={np.round(direction, 2)}  |cos| vs normal={cos_str}")

            viewer.add_labels(mask.astype(np.int32) * (i + 1), name=name, opacity=0.6)
            center = np.array(np.nonzero(mask)).mean(axis=1)
            length = 0.35 * min(shape)
            vec = np.stack([center, direction * length])[None]  # (1, 2, 3): origin + direction
            viewer.add_vectors(vec, name=f"{name} dir", edge_color="red", edge_width=1.5)

        napari.run()


if __name__ == "__main__":
    if "--view" in sys.argv:
        sys.argv.remove("--view")
        os.environ[VIEW_ENV] = "1"
    unittest.main()
