import glob
import os
import sys
import unittest

import numpy as np

# Set SYNAPSE_NET_VIEW=1 (or run this file with --view) to open the orientation cases in
# napari — see TestCristaOrientation.test_visualize_orientations.
VIEW_ENV = "SYNAPSE_NET_VIEW"

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

# Single-mitochondrion cutout used by the skeleton-junction regression below. It carries its own
# voxel_size attribute (8.681 Angstrom -> 0.8681 nm), so no .mrc lookup is needed. Override with
# SYNAPSE_NET_CRISTAE_CUTOUT.
_CUTOUT_H5 = os.environ.get(
    "SYNAPSE_NET_CRISTAE_CUTOUT",
    os.path.join(_CRISTAE_DATA_ROOT, "cristae_cutout_test", "cutout_mito2.h5"),
)
_CUTOUT_REQUIRED = unittest.skipUnless(
    os.path.exists(_CUTOUT_H5), f"cristae cutout not found at {_CUTOUT_H5}"
)

# A densely packed mitochondrion, where skeleton mode's proximity premise over-detects. Lives in the
# same tree as the other real data; mito instance 1, cropped to its bounding box.
_DENSE_H5 = os.path.join(
    _H5_DIR, "2_20230817_TOMO_HOI_WT_36859_J1_STEM750",
    "36859_J1_66K_TS_PS_01_rec_2kb1dawbp_crop.h5",
)
_DENSE_BBOX = (slice(0, 401), slice(285, 542), slice(162, 452))
_DENSE_REQUIRED = unittest.skipUnless(
    os.path.exists(_DENSE_H5), f"dense-cristae tomogram not found at {_DENSE_H5}"
)
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


# Fixture for the skeleton-extension junction mode (TestSkeletonJunctions). A box mito at 1.5 nm
# voxels with an 8 nm shell leaves a lumen spanning z 11..28 and y 13..46, so a lamella's distance
# to the membrane is set purely by how far its Y ends fall short of y 13 / y 46.
_SHEET_SHAPE = (40, 60, 60)
_SHEET_VOXEL_NM = 1.5
_SHEET_MEMBRANE_NM = 8.0


def _make_hollow_mito(shape=_SHEET_SHAPE, margin=(6, 8, 8),
                      voxel_size=_SHEET_VOXEL_NM, membrane_thickness_nm=_SHEET_MEMBRANE_NM):
    """A box mitochondrion with the membrane shell and lumen that approximate_membrane derives."""
    from synapse_net.cristae_analysis import approximate_membrane
    mito = np.zeros(shape, dtype=np.uint32)
    mito[margin[0]:shape[0] - margin[0],
         margin[1]:shape[1] - margin[1],
         margin[2]:shape[2] - margin[2]] = 1
    membrane, lumen = approximate_membrane(
        mito, voxel_size, membrane_thickness_nm=membrane_thickness_nm, return_lumen=True
    )
    return mito, membrane, lumen


def _make_crista_sheet(lumen, y_range, z_range=(18, 24), x_range=(29, 31), clip_to_lumen=True):
    """A flat lamella, by default clipped to the lumen so it cannot overlap the membrane band.

    ``z_range`` sits well inside the lumen in Z, so only the sheet's two Y ends can ever be in reach
    of the membrane and the expected junction count is just the number of Y ends within range.
    Clipping to the lumen is what makes these cases interesting: the overlap detector scores zero on
    every one of them, because there is no overlap to find by construction.
    """
    crista = np.zeros(lumen.shape, dtype=bool)
    crista[z_range[0]:z_range[1], y_range[0]:y_range[1], x_range[0]:x_range[1]] = True
    return (crista & lumen) if clip_to_lumen else crista


class TestSurfaceMesh(unittest.TestCase):
    """The marching-cubes helper: unpadded return frame and selective (open-face) padding."""

    def test_unpadded_return_frame(self):
        # A block not touching any array edge: verts are returned in the mask's own index frame
        # (surface at index ± 0.5), not shifted by the old +1 pad.
        from synapse_net.cristae_analysis import _surface_mesh
        block = np.zeros((16, 16, 16), dtype=bool)
        block[4:8, 4:8, 4:8] = True  # occupies indices 4..7 on each axis
        verts, _ = _surface_mesh(block, np.ones(3))
        self.assertAlmostEqual(float(verts[:, 0].min()), 3.5, places=5)
        self.assertAlmostEqual(float(verts[:, 0].max()), 7.5, places=5)

    def test_open_faces_omit_cap(self):
        # A block flush against z=0 and z=max: opening those faces omits the caps, leaving an open
        # surface (fewer faces, smaller area, no vertices beyond the clipped planes) while the closed
        # mesh caps them (vertices at z ≈ -0.5 and z ≈ 9.5).
        from skimage.measure import mesh_surface_area
        from synapse_net.cristae_analysis import _surface_mesh
        block = np.zeros((10, 20, 20), dtype=bool)
        block[:, 6:14, 6:14] = True  # spans the full z extent → touches z=0 and z=9
        closed_v, closed_f = _surface_mesh(block, np.ones(3))
        open_faces = np.array([[False, False], [True, True], [True, True]])  # z open, y/x closed
        open_v, open_f = _surface_mesh(block, np.ones(3), closed_faces=open_faces)

        self.assertLess(len(open_f), len(closed_f))
        self.assertLess(mesh_surface_area(open_v, open_f), mesh_surface_area(closed_v, closed_f))
        # Open mesh stays within the clipped z-planes; closed mesh extends beyond them (the caps).
        self.assertGreaterEqual(float(open_v[:, 0].min()), -1e-6)
        self.assertLessEqual(float(open_v[:, 0].max()), 9.0 + 1e-6)
        self.assertLess(float(closed_v[:, 0].min()), 0.0)
        self.assertGreater(float(closed_v[:, 0].max()), 9.0)

    def test_closed_default_unchanged_topology(self):
        # Default (closed_faces=None) is watertight: every boundary face capped.
        from synapse_net.cristae_analysis import _surface_mesh
        block = np.zeros((10, 20, 20), dtype=bool)
        block[:, 6:14, 6:14] = True
        _, faces_default = _surface_mesh(block, np.ones(3))
        all_closed = np.ones((3, 2), dtype=bool)
        _, faces_explicit = _surface_mesh(block, np.ones(3), closed_faces=all_closed)
        self.assertEqual(len(faces_default), len(faces_explicit))

    @staticmethod
    def _has_boundary_edge(faces):
        # An open (non-watertight) mesh has an edge used by only one triangle; a closed manifold uses
        # every edge exactly twice.
        from collections import Counter
        edges = Counter()
        for tri in faces:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edges[(a, b) if a < b else (b, a)] += 1
        return any(count == 1 for count in edges.values())

    def test_open_trimmed_mesh(self):
        # A block spanning the full z-extent (clipped at both z faces): _open_trimmed_mesh crops the
        # border zone off and leaves the cut OPEN, so the mesh is trimmed to [gap, shape-gap] and is
        # not watertight (no cap for geodesics to shortcut across). An interior block (no clipped face)
        # stays closed and untrimmed.
        from synapse_net.cristae_analysis import _open_trimmed_mesh
        gap = 2
        clipped = np.zeros((12, 20, 20), dtype=bool)
        clipped[:, 6:14, 6:14] = True  # spans z=0..11 → clipped at both z faces
        verts, faces = _open_trimmed_mesh(clipped, np.ones(3), gap, np.ones((3, 2), dtype=bool))
        self.assertGreaterEqual(float(verts[:, 0].min()), gap - 1e-6)          # trimmed low
        self.assertLessEqual(float(verts[:, 0].max()), clipped.shape[0] - gap + 1e-6)  # trimmed high
        self.assertTrue(self._has_boundary_edge(faces))                        # open at the cut

        interior = np.zeros((20, 20, 20), dtype=bool)
        interior[6:14, 6:14, 6:14] = True  # touches no volume face
        v2, f2 = _open_trimmed_mesh(interior, np.ones(3), gap, np.zeros((3, 2), dtype=bool))
        self.assertFalse(self._has_boundary_edge(f2))                          # closed, watertight
        self.assertAlmostEqual(float(v2[:, 0].min()), 5.5, places=5)           # untrimmed (index frame)



class TestBorderZone(unittest.TestCase):
    """_border_zone's per-face gating — the piece that must not blank interior bbox faces."""

    def test_all_faces_by_default(self):
        from synapse_net.cristae_analysis import _border_zone
        zone = _border_zone((10, 10, 10), 2)
        self.assertTrue(zone[0].all() and zone[-1].all())
        self.assertTrue(zone[:, 0].all() and zone[:, -1].all())
        self.assertTrue(zone[:, :, 0].all() and zone[:, :, -1].all())
        self.assertFalse(zone[5, 5, 5])

    def test_no_faces_gives_empty_zone(self):
        # A bbox crop entirely in the volume interior: nothing may be excluded, or real junctions in
        # the middle of the volume would be deleted.
        from synapse_net.cristae_analysis import _border_zone
        zone = _border_zone((10, 10, 10), 3, boundary=np.zeros((3, 2), dtype=bool))
        self.assertEqual(int(zone.sum()), 0)

    def test_single_face_marks_only_that_face(self):
        from synapse_net.cristae_analysis import _border_zone
        boundary = np.zeros((3, 2), dtype=bool)
        boundary[0, 0] = True  # low z only
        zone = _border_zone((10, 8, 8), 2, boundary=boundary)
        self.assertTrue(zone[:2].all())
        self.assertEqual(int(zone[2:].sum()), 0)

    def test_matches_open_trimmed_mesh_gating(self):
        # The zone must be the complement of the region _open_trimmed_mesh keeps, for the same
        # (radius, boundary) — they encode the same "certain region" and must not drift apart.
        from synapse_net.cristae_analysis import _border_zone
        radius = 3
        for boundary in (np.ones((3, 2), dtype=bool), np.zeros((3, 2), dtype=bool),
                         np.array([[True, False], [False, True], [True, True]])):
            with self.subTest(boundary=boundary.tolist()):
                shape = (14, 15, 16)
                zone = _border_zone(shape, radius, boundary=boundary)
                lo = [radius if boundary[a, 0] else 0 for a in range(3)]
                hi = [shape[a] - (radius if boundary[a, 1] else 0) for a in range(3)]
                kept = np.zeros(shape, dtype=bool)
                kept[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
                np.testing.assert_array_equal(zone, ~kept)


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

    def test_slice2d_caps_true_z_end(self):
        # A mito that ends within the volume in Z: slice_2d must cap the true Z-ends (membrane fills the
        # end slices' interior, like shell_3d), while a middle slice's interior stays lumen.
        from synapse_net.cristae_analysis import approximate_membrane
        seg = np.zeros((20, 40, 40), dtype="uint32")
        seg[6:14, 8:32, 8:32] = 1  # z-extent 6..13, away from z=0 and z=19
        mem = approximate_membrane(seg, voxel_size=1.0, membrane_thickness_nm=3.0, membrane_mode="slice_2d")
        self.assertFalse(mem[10, 20, 20])  # middle-slice interior → lumen, not membrane
        self.assertTrue(mem[6, 20, 20])    # true Z-start cap → membrane
        self.assertTrue(mem[13, 20, 20])   # true Z-end cap → membrane

    def test_slice2d_no_cap_at_clipped_z_end(self):
        # A mito clipped by the z=0 volume face: that end must NOT be capped (unknown / near-face),
        # while the true interior end at the other side is capped.
        from synapse_net.cristae_analysis import approximate_membrane
        seg = np.zeros((20, 40, 40), dtype="uint32")
        seg[0:14, 8:32, 8:32] = 1  # touches the z=0 face
        mem = approximate_membrane(seg, voxel_size=1.0, membrane_thickness_nm=3.0, membrane_mode="slice_2d")
        self.assertFalse(mem[0, 20, 20])   # clipped end interior → not membrane
        self.assertFalse(mem[2, 20, 20])
        self.assertTrue(mem[13, 20, 20])   # true interior Z-end → capped

    def test_return_lumen_partitions_mito(self):
        # return_lumen=True gives the eroded interior: a boolean mask inside the mito and disjoint from
        # the membrane shell.
        from synapse_net.cristae_analysis import approximate_membrane
        mito_seg = _make_mito()
        membrane, lumen = approximate_membrane(mito_seg, voxel_size=1.0, return_lumen=True)
        mito_binary = mito_seg > 0
        self.assertEqual(lumen.dtype, bool)
        self.assertTrue(lumen.any())
        self.assertTrue(np.all(lumen[~mito_binary] == False))  # lumen ⊆ mito  # noqa: E712
        self.assertFalse((membrane & lumen).any())             # membrane ∩ lumen == ∅

    def test_lumen_does_not_reinclude_outer_shell(self):
        # A mito flush against a volume face: `mito & ~membrane` re-includes the outer shell wherever
        # border-gap suppression zeroed the membrane, but the returned lumen does not — it is a strict
        # subset. This is the bug the display/geodesic mesh hit when a mito touched the crop edge.
        from synapse_net.cristae_analysis import approximate_membrane
        seg = np.zeros((24, 30, 30), dtype="uint32")
        seg[0:18, 6:24, 6:24] = 1  # touches the z=0 face
        for mode in ("slice_2d", "shell_3d"):
            membrane, lumen = approximate_membrane(
                seg, voxel_size=1.0, membrane_thickness_nm=4.0, membrane_mode=mode, return_lumen=True
            )
            contaminated = (seg > 0) & ~membrane
            self.assertFalse((lumen & ~contaminated).any(), msg=f"lumen ⊄ mito & ~membrane for {mode}")
            self.assertLess(
                int(lumen.sum()), int(contaminated.sum()),
                msg=f"lumen not strictly smaller than mito & ~membrane for {mode}",
            )

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


class TestSkeletonJunctions(unittest.TestCase):
    """The skeleton junction mode: crista regions reaching close to the membrane near a terminus.

    Every sheet here is clipped to the lumen, so it never overlaps the membrane band and the overlap
    detector reports zero junctions throughout. That is the point of the mode: the junctions are real
    but invisible to an intersection test.
    """

    @classmethod
    def setUpClass(cls):
        cls.mito, cls.membrane, cls.lumen = _make_hollow_mito()
        cls.voxel_size = _SHEET_VOXEL_NM
        cls.max_extension = _SHEET_MEMBRANE_NM

    def _detect(self, crista, **kwargs):
        from synapse_net.cristae_analysis import detect_junctions_skeleton
        kwargs.setdefault("max_extension_nm", self.max_extension)
        return detect_junctions_skeleton(crista, self.membrane, self.voxel_size, **kwargs)

    def _sheet_with_speck(self, length=4):
        """Two real junctions plus one speck on the membrane, small in both volume and skeleton length.

        ``length`` voxels is 13.5 nm^3 (below the 50 nm^3 volume default) and ~4.5 nm of skeleton
        (below the 10 nm min_skeleton_nm default), so either filter alone removes it.
        """
        crista = _make_crista_sheet(self.lumen, y_range=(16, 44))
        z, y, x = np.argwhere(self.membrane)[0]
        crista[z, y, x:x + length] = True
        return crista

    def test_sheet_touching_membrane_gives_one_junction_per_end(self):
        # Lumen-spanning sheet: both Y ends sit right against the membrane band. Being clipped to the
        # lumen it does not overlap the band, so its closest approach is one voxel, not zero — the
        # reported extension is the region's real distance to the membrane.
        crista = _make_crista_sheet(self.lumen, y_range=(8, 52))
        labels, summary = self._detect(crista)
        self.assertEqual(summary["crista_junction_count"], 2)
        self.assertLessEqual(summary["mean_junction_extension_nm"], self.voxel_size)
        self.assertEqual(labels.shape, crista.shape)
        self.assertEqual(set(np.unique(labels).tolist()), {0, 1, 2})

    def test_overlapping_crista_reports_zero_extension(self):
        # A crista that genuinely runs through the band has a closest approach of exactly zero, which
        # is what distinguishes the extension metric from a mere proximity score.
        crista = _make_crista_sheet(self.lumen, y_range=(8, 52), clip_to_lumen=False)
        _, summary = self._detect(crista)
        self.assertEqual(summary["crista_junction_count"], 2)
        self.assertEqual(summary["mean_junction_extension_nm"], 0.0)

    def test_sub_threshold_gap_detected_where_overlap_finds_nothing(self):
        # The headline case for this mode. The sheet stops ~4.5 nm short of the membrane on both
        # sides — a real junction that the overlap detector cannot see at all.
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = _make_crista_sheet(self.lumen, y_range=(16, 44))
        _, overlap_summary = detect_contact_sites(crista, self.membrane, self.voxel_size)
        self.assertEqual(overlap_summary["crista_junction_count"], 0)
        self.assertEqual(overlap_summary["contact_voxel_count"], 0)

        _, summary = self._detect(crista)
        self.assertEqual(summary["crista_junction_count"], 2)
        extension = summary["mean_junction_extension_nm"]
        self.assertGreater(extension, 0.0)
        self.assertLessEqual(extension, self.max_extension)

    def test_super_threshold_gap_rejected(self):
        # Well beyond one membrane thickness the ends must not be credited as junctions.
        for y_range in [(21, 39), (27, 33)]:
            with self.subTest(y_range=y_range):
                crista = _make_crista_sheet(self.lumen, y_range=y_range)
                labels, summary = self._detect(crista)
                self.assertEqual(summary["crista_junction_count"], 0)
                self.assertEqual(labels.max(), 0)
                self.assertTrue(np.isnan(summary["mean_junction_extension_nm"]))

    def test_asymmetric_sheet_gives_one_junction(self):
        # Reaches the membrane at low Y only; the high-Y end is far out of range.
        crista = _make_crista_sheet(self.lumen, y_range=(8, 39))
        _, summary = self._detect(crista)
        self.assertEqual(summary["crista_junction_count"], 1)

    def test_labels_confined_to_membrane_and_mitochondrion(self):
        # An earlier implementation painted a ball(merge_radius) around each hit as the junction's
        # extent, which put 39% of the labelled voxels outside the mitochondrion — up to merge_radius
        # past it. A junction is now the contact region itself: on the membrane band where the crista
        # overlaps it, on the crista where it falls short. Either way it stays inside the mito.
        mito_binary = self.mito > 0
        cases = [
            # Runs through the band to the mito surface — the geometry that produced the spill.
            ("through the band", _make_crista_sheet(self.lumen, (8, 52), clip_to_lumen=False), 2),
            ("touching the band", _make_crista_sheet(self.lumen, (8, 52)), 2),
            ("gap under threshold", _make_crista_sheet(self.lumen, (16, 44)), 2),
            ("one end in reach", _make_crista_sheet(self.lumen, (8, 39)), 1),
        ]
        for case, crista, expected_count in cases:
            with self.subTest(case=case):
                labels, summary = self._detect(crista)
                self.assertEqual(summary["crista_junction_count"], expected_count)
                junction = labels > 0
                self.assertTrue(junction.any())
                self.assertEqual(int(np.count_nonzero(junction & ~(self.membrane | crista))), 0)
                self.assertEqual(int(np.count_nonzero(junction & ~mito_binary)), 0)

    def test_neighbouring_cristae_do_not_fuse(self):
        # A junction is a connected component of the near-membrane crista mask, so it is a subset of
        # one crista and two disconnected cristae can never be fused — no parameter governs this. An
        # earlier radius-based merge did fuse them: 11 lamellae 15 nm apart reported 7 junctions,
        # fewer than there were cristae. Two sheets a few nm apart must stay two junctions.
        from scipy.ndimage import label as ndimage_label
        for x_gap in (3, 5, 8):
            with self.subTest(x_gap_voxels=x_gap):
                crista = np.zeros(_SHEET_SHAPE, dtype=bool)
                crista[18:24, 8:26, 29:31] = True
                crista[18:24, 8:26, 29 + x_gap:31 + x_gap] = True
                crista &= self.lumen
                n_components = ndimage_label(crista, structure=np.ones((3, 3, 3), dtype=bool))[1]
                self.assertEqual(n_components, 2)  # guard the fixture: two distinct cristae
                _, summary = self._detect(crista)
                self.assertEqual(summary["crista_junction_count"], 2)

    def test_min_extension_excludes_touching_ends(self):
        # min_extension_nm isolates junctions that genuinely span a gap: a sheet already against the
        # membrane is dropped, while one stopping short of it survives.
        touching = _make_crista_sheet(self.lumen, y_range=(8, 52))
        _, summary = self._detect(touching, min_extension_nm=3.0)
        self.assertEqual(summary["crista_junction_count"], 0)

        with_gap = _make_crista_sheet(self.lumen, y_range=(16, 44))
        _, summary = self._detect(with_gap, min_extension_nm=3.0)
        self.assertEqual(summary["crista_junction_count"], 2)

    def test_contact_voxel_columns_stay_overlap_based(self):
        # crista_junction_count is the only mode-specific number: contact_voxel_count and
        # contact_volume_nm3 must keep describing the true overlap so they mean the same thing in
        # both modes and stay comparable across runs.
        from synapse_net.cristae_analysis import detect_contact_sites
        crista = _make_crista_sheet(self.lumen, y_range=(8, 52), clip_to_lumen=False)
        _, overlap_summary = detect_contact_sites(crista, self.membrane, self.voxel_size)
        _, summary = self._detect(crista)
        self.assertGreater(overlap_summary["contact_voxel_count"], 0)
        self.assertEqual(summary["contact_voxel_count"], overlap_summary["contact_voxel_count"])
        self.assertAlmostEqual(summary["contact_volume_nm3"], overlap_summary["contact_volume_nm3"])

    def test_empty_inputs_give_no_junctions(self):
        empty = np.zeros(_SHEET_SHAPE, dtype=bool)
        for crista, membrane, case in [
            (empty, self.membrane, "empty crista"),
            (_make_crista_sheet(self.lumen, y_range=(8, 52)), empty, "empty membrane"),
        ]:
            with self.subTest(case=case):
                from synapse_net.cristae_analysis import detect_junctions_skeleton
                labels, summary = detect_junctions_skeleton(
                    crista, membrane, self.voxel_size, max_extension_nm=self.max_extension
                )
                self.assertEqual(summary["crista_junction_count"], 0)
                self.assertEqual(summary["contact_voxel_count"], 0)
                self.assertEqual(labels.max(), 0)
                self.assertTrue(np.isnan(summary["mean_junction_extension_nm"]))

    def test_terminus_filter_rejects_a_crista_running_alongside_the_membrane(self):
        # The terminus filter is the one place the skeleton still matters: a crista that ENDS at the
        # membrane is a junction, one that merely runs ALONGSIDE it is not. A sheet laid parallel to
        # the membrane band has its long flank in range but its skeleton ends far away, so a tight
        # terminus distance must reject it while a permissive one accepts it.
        crista = np.zeros(_SHEET_SHAPE, dtype=bool)
        crista[18:24, 15:45, 15:17] = True          # parallel to the low-x wall, along its length
        crista &= self.lumen
        self.assertTrue(crista.any())
        _, permissive = self._detect(crista, terminus_nm=float("inf"))
        _, strict = self._detect(crista, terminus_nm=1.0)
        self.assertGreater(permissive["crista_junction_count"], 0)
        self.assertLessEqual(strict["crista_junction_count"], permissive["crista_junction_count"])

    def test_junction_footprint_is_a_small_patch_at_the_closest_approach(self):
        # The complaint this fixes: the label used to be the whole candidate region, a slab of crista
        # anywhere within max_extension of the membrane, so it was fat and its centre of mass sat in the
        # middle of the crista rather than at the contact. The label must now be a small patch, and it
        # must contain the region's closest-approach voxel.
        from bioimage_cpp.distance import distance_transform
        from synapse_net.cristae_analysis import _to_sampling
        sampling = _to_sampling(self.voxel_size, 3)
        reference = distance_transform(~self.membrane, sampling=sampling.tolist(), number_of_threads=1)

        # A sheet meeting the membrane end-on and running well away from it: the candidate region spans
        # the full max_extension reach while the contact itself is at one end of that span. A crista
        # lying exactly parallel is deliberately not used here — being equidistant along its whole
        # flank, its closest approach genuinely *is* the whole flank, so nothing should shrink.
        crista = np.zeros(_SHEET_SHAPE, dtype=bool)
        crista[18:24, 13:40, 29:31] = True
        crista &= self.lumen
        labels, summary = self._detect(crista, terminus_nm=float("inf"))
        self.assertGreater(summary["crista_junction_count"], 0)

        near = crista & (reference <= self.max_extension)
        self.assertLess(int(np.count_nonzero(labels)), int(np.count_nonzero(near)))
        # The region reaches much further from the membrane than the label does.
        self.assertGreater(float(reference[near].max()), self.max_extension / 2)

        tolerance = float(np.linalg.norm(sampling))
        for junction in range(1, summary["crista_junction_count"] + 1):
            mask = labels == junction
            self.assertTrue(mask.any())
            # Every labelled voxel sits at the closest approach, not spread along the crista.
            self.assertLessEqual(float(reference[mask].max()) - float(reference[mask].min()),
                                 tolerance + 1e-6)
            self.assertAlmostEqual(float(reference[mask].min()), float(reference[near].min()), delta=2.0)

    def test_lumen_reference_localises_inside_the_membrane_band(self):
        # Why the lumen is passed at all. The membrane band's own distance transform is 0 on every voxel
        # of the band, so it cannot say where within the band a crista sits and the closest approach
        # collapses to 0. Measuring from the inner boundary membrane surface restores that information.
        from bioimage_cpp.distance import distance_transform
        from synapse_net.cristae_analysis import _inner_surface_distance, _to_sampling
        sampling = _to_sampling(self.voxel_size, 3)
        band = distance_transform(~self.membrane, sampling=sampling.tolist(), number_of_threads=1)
        inner = _inner_surface_distance(self.lumen, sampling)

        in_band = self.membrane
        self.assertEqual(float(band[in_band].max()), 0.0)
        self.assertGreater(float(inner[in_band].max()), self.voxel_size)

    def test_infinite_terminus_disables_the_filter(self):
        # inf must be a pure no-op relative to the generous default on ordinary geometry, so the
        # filter cannot silently be dropping real junctions.
        crista = _make_crista_sheet(self.lumen, y_range=(16, 44))
        _, default = self._detect(crista)
        _, disabled = self._detect(crista, terminus_nm=float("inf"))
        self.assertEqual(default["crista_junction_count"], disabled["crista_junction_count"])

    def test_min_junction_volume_drops_specks(self):
        # Real data produces 1-, 3- and 12-voxel "junctions", which are not junctions on any reading.
        # A speck placed right on the membrane must be dropped while the real junctions survive.
        # min_skeleton_nm is disabled here so the assertion is about the volume knob alone. The speck is
        # a short line rather than a single voxel because an isolated voxel has skeleton degree 0 and is
        # therefore not a terminus at all — it would be rejected before the volume filter is reached.
        crista = self._sheet_with_speck()
        _, permissive = self._detect(crista, min_junction_volume_nm3=0.0, min_skeleton_nm=0.0)
        _, default = self._detect(crista, min_skeleton_nm=0.0)
        self.assertEqual(permissive["crista_junction_count"], 3)  # 2 real + the speck
        self.assertEqual(default["crista_junction_count"], 2)     # speck filtered out

    def test_speck_is_also_dropped_by_the_skeleton_length_filter(self):
        # The second, independent route by which a speck is rejected: its skeleton is shorter than
        # min_skeleton_nm, so the component is dropped, it contributes no terminus, and the region can
        # never pass the terminus gate — with the volume filter switched off entirely.
        crista = self._sheet_with_speck()
        _, kept = self._detect(crista, min_junction_volume_nm3=0.0, min_skeleton_nm=0.0)
        _, dropped = self._detect(crista, min_junction_volume_nm3=0.0)
        self.assertEqual(kept["crista_junction_count"], 3)
        self.assertEqual(dropped["crista_junction_count"], 2)

    def test_min_junction_volume_is_measured_on_the_region_not_the_footprint(self):
        # The painted label is only the closest-approach footprint, a small fraction of the candidate
        # region. If the threshold were ever moved onto the footprint it would silently become far
        # stricter; this pins it to the region. A threshold between the two sizes must keep the junction.
        crista = _make_crista_sheet(self.lumen, y_range=(8, 52))
        labels, summary = self._detect(crista)
        self.assertEqual(summary["crista_junction_count"], 2)
        footprint_voxels = int(np.count_nonzero(labels == 1))
        voxel_vol = self.voxel_size ** 3
        threshold = (footprint_voxels + 1) * voxel_vol  # above the footprint, below the region
        _, still_there = self._detect(crista, min_junction_volume_nm3=threshold)
        self.assertEqual(still_there["crista_junction_count"], 2)

    def test_min_junction_volume_can_remove_everything(self):
        # Monotonic in the threshold, and a large enough value leaves nothing — the knob is doing
        # what it claims rather than silently capping.
        crista = _make_crista_sheet(self.lumen, y_range=(16, 44))
        _, summary = self._detect(crista, min_junction_volume_nm3=1e9)
        self.assertEqual(summary["crista_junction_count"], 0)
        self.assertTrue(np.isnan(summary["mean_junction_extension_nm"]))

    def test_junction_in_the_border_zone_is_suppressed(self):
        # The regression for the reported bug. approximate_membrane deletes the membrane within
        # border_gap of a clipped volume face because its presence is unknown there. Overlap mode
        # therefore cannot report a junction in that zone; skeleton mode must not either. Without the
        # exclusion the distance transform measures straight across the deleted region to the nearest
        # surviving membrane voxel and flags a junction against a membrane it was told nothing about.
        from synapse_net.cristae_analysis import _border_zone, _gap_radius
        border_radius = _gap_radius(self.voxel_size, _SHEET_MEMBRANE_NM, None, 3)
        zone = _border_zone(_SHEET_SHAPE, border_radius)
        # A crista sitting inside the border zone, hugging the low-x volume face.
        crista = np.zeros(_SHEET_SHAPE, dtype=bool)
        crista[18:24, 20:40, 0:border_radius] = True
        self.assertEqual(int(np.count_nonzero(crista & ~zone)), 0)  # guard: wholly inside the zone
        self.assertGreater(int(crista.sum()), 0)

        _, unguarded = self._detect(crista, border_radius=0)
        _, guarded = self._detect(crista, border_radius=border_radius)
        self.assertGreater(unguarded["crista_junction_count"], 0)   # the bug
        self.assertEqual(guarded["crista_junction_count"], 0)       # the fix

    def test_border_exclusion_keeps_junctions_away_from_the_faces(self):
        # The exclusion must be surgical: ordinary junctions well inside the volume are untouched.
        from synapse_net.cristae_analysis import _gap_radius
        border_radius = _gap_radius(self.voxel_size, _SHEET_MEMBRANE_NM, None, 3)
        for name, y_range, expected in [("touching", (8, 52), 2), ("sub-threshold gap", (16, 44), 2)]:
            with self.subTest(case=name):
                crista = _make_crista_sheet(self.lumen, y_range=y_range)
                _, summary = self._detect(crista, border_radius=border_radius)
                self.assertEqual(summary["crista_junction_count"], expected)

    def test_no_junction_voxel_lands_in_the_border_zone(self):
        # The alignment invariant, stated positively: overlap mode has this property by construction
        # (crista & membrane is empty where membrane is), and skeleton mode must match it.
        from synapse_net.cristae_analysis import _border_zone, _gap_radius
        border_radius = _gap_radius(self.voxel_size, _SHEET_MEMBRANE_NM, None, 3)
        zone = _border_zone(_SHEET_SHAPE, border_radius)
        crista = np.zeros(_SHEET_SHAPE, dtype=bool)
        crista[18:24, 8:52, 29:31] = True            # spans the mito, unclipped to the lumen
        crista[18:24, 20:40, 0:border_radius] = True  # plus a chunk inside the border zone
        labels, _ = self._detect(crista, border_radius=border_radius)
        self.assertEqual(int(np.count_nonzero((labels > 0) & zone)), 0)

    def test_interior_crop_faces_are_not_excluded(self):
        # The trap: in the per-mito path the arrays are bbox crops, so gating on `boundary` is what
        # stops the exclusion from eating junctions at interior bbox faces. With every face marked
        # interior, the exclusion must be a no-op even at a large radius.
        crista = _make_crista_sheet(self.lumen, y_range=(8, 52))
        _, unguarded = self._detect(crista, border_radius=0)
        _, interior = self._detect(
            crista, border_radius=12, boundary=np.zeros((3, 2), dtype=bool)
        )
        self.assertEqual(
            interior["crista_junction_count"], unguarded["crista_junction_count"]
        )

    def test_two_dimensional_input_raises(self):
        # TEASAR has no 2D implementation, and cristae_analysis otherwise accepts 2D — so this has
        # to fail loudly rather than silently mis-handle a 2D volume.
        from synapse_net.cristae_analysis import detect_junctions_skeleton
        flat = np.zeros((20, 20), dtype=bool)
        with self.assertRaises(ValueError):
            detect_junctions_skeleton(flat, flat, voxel_size=1.0)

    def test_non_positive_voxel_size_raises(self):
        # Every physical threshold is scaled by the voxel size, so a zero silently makes the
        # near-membrane test meaningless. Fail loudly instead.
        from synapse_net.cristae_analysis import detect_junctions_skeleton
        crista = _make_crista_sheet(self.lumen, y_range=(8, 52))
        with self.assertRaises(ValueError):
            detect_junctions_skeleton(crista, self.membrane, voxel_size=0.0)

    def test_inverted_extension_range_raises(self):
        from synapse_net.cristae_analysis import detect_junctions_skeleton
        crista = _make_crista_sheet(self.lumen, y_range=(8, 52))
        with self.assertRaises(ValueError):
            detect_junctions_skeleton(
                crista, self.membrane, self.voxel_size,
                min_extension_nm=10.0, max_extension_nm=2.0,
            )




class TestCristaSkeleton(unittest.TestCase):
    """compute_crista_skeleton — the centerline exposed for the widget's Show Crista Skeleton layers."""

    def test_tube_has_two_termini(self):
        from synapse_net.cristae_analysis import compute_crista_skeleton
        tube = np.zeros((40, 40, 40), dtype=bool)
        tube[18:22, 18:22, 5:35] = True
        vertices, is_terminus = compute_crista_skeleton(tube, voxel_size=1.0)
        self.assertGreater(len(vertices), 0)
        self.assertEqual(len(vertices), len(is_terminus))
        self.assertEqual(int(is_terminus.sum()), 2)

    def test_sheet_termini_populate_the_rim(self):
        # A lamella is a sheet, not a tube: TEASAR spans it with a forest whose ends line the rim.
        # This is why the terminus filter is not simply "the two ends of a line".
        from synapse_net.cristae_analysis import compute_crista_skeleton
        sheet = np.zeros((40, 40, 40), dtype=bool)
        sheet[5:35, 8:32, 19:21] = True
        _, is_terminus = compute_crista_skeleton(sheet, voxel_size=1.0)
        self.assertGreater(int(is_terminus.sum()), 2)

    def test_terminus_merging_thins_the_rim_fan(self):
        # The cleanup that makes the termini usable. Raw TEASAR puts a fan of free ends along a sheet's
        # rim; merging collapses each fan to one representative, which is what takes a real tomogram
        # from ~10000 termini to a few hundred.
        from synapse_net.cristae_analysis import compute_crista_skeleton
        sheet = np.zeros((40, 40, 40), dtype=bool)
        sheet[5:35, 8:32, 19:21] = True
        _, raw = compute_crista_skeleton(sheet, voxel_size=1.0, terminus_merge_nm=0.0)
        _, merged = compute_crista_skeleton(sheet, voxel_size=1.0)
        self.assertLess(int(merged.sum()), int(raw.sum()))

    def test_termini_do_not_merge_across_separate_cristae(self):
        # The hazard that makes per-component clustering mandatory: densely packed cristae sit a few nm
        # apart, so plain single-linkage clustering chains their termini together and collapses the
        # whole field. Two parallel lamellae must each keep their own termini.
        #
        # The radius is set explicitly to 8 nm rather than left at the 4 nm default because that is
        # where the hazard bites for this 6 nm spacing: at 8 nm, clustering restricted to one component
        # keeps one terminus per lamella, while unrestricted clustering fuses them and leaves one
        # lamella with none at all.
        from synapse_net.cristae_analysis import compute_crista_skeleton
        pair = np.zeros((40, 40, 60), dtype=bool)
        pair[10:30, 18:20, 5:55] = True
        pair[10:30, 24:26, 5:55] = True
        vertices, is_terminus = compute_crista_skeleton(pair, voxel_size=1.0, terminus_merge_nm=8.0)
        y = vertices[is_terminus][:, 1]
        self.assertTrue((y < 22).any(), "the first lamella lost all of its termini")
        self.assertTrue((y > 22).any(), "the second lamella lost all of its termini")

    def test_speck_components_are_dropped(self):
        # Every segmentation speck contributes its own miniature skeleton and hence its own termini.
        # Filtering whole components by length removes them at the root.
        from synapse_net.cristae_analysis import compute_crista_skeleton
        rng = np.random.default_rng(0)
        specks = np.zeros((40, 60, 60), dtype=bool)
        for _ in range(60):
            z, y, x = rng.integers(2, 38), rng.integers(2, 58), rng.integers(2, 58)
            specks[z:z + 2, y:y + 2, x:x + 2] = True
        vertices, is_terminus = compute_crista_skeleton(specks, voxel_size=1.5)
        self.assertEqual(len(vertices), 0)
        self.assertEqual(int(is_terminus.sum()), 0)
        kept, _ = compute_crista_skeleton(specks, voxel_size=1.5, min_skeleton_nm=0.0)
        self.assertGreater(len(kept), 0)

    def test_isolated_vertex_is_not_a_terminus(self):
        # degree == 1, not <= 1. A lone vertex is not the free end of anything, and counting it was part
        # of why the raw terminus set was unusable.
        from synapse_net.cristae_analysis import compute_crista_skeleton
        blob = np.zeros((16, 16, 16), dtype=bool)
        blob[8, 8, 8] = True
        vertices, is_terminus = compute_crista_skeleton(blob, voxel_size=1.0, min_skeleton_nm=0.0)
        if len(vertices):
            self.assertEqual(int(is_terminus.sum()), 0)

    def test_edges_index_the_returned_vertices(self):
        # The reindex after dropping components is the easiest thing here to get wrong: stale indices
        # would silently draw the skeleton as garbage in the widget.
        from synapse_net.cristae_analysis import compute_crista_skeleton
        sheet = np.zeros((40, 40, 40), dtype=bool)
        sheet[5:35, 8:32, 19:21] = True
        vertices, is_terminus, edges = compute_crista_skeleton(sheet, voxel_size=1.0, return_edges=True)
        self.assertEqual(edges.shape[1], 2)
        self.assertGreater(len(edges), 0)
        self.assertGreaterEqual(int(edges.min()), 0)
        self.assertLess(int(edges.max()), len(vertices))
        self.assertEqual(len(vertices), len(is_terminus))

    def test_vertices_are_nm_and_land_inside_the_crista(self):
        # The widget divides by the voxel size to get array indices; if the returned frame were voxels
        # rather than nm that conversion would scatter the skeleton outside the mask.
        from synapse_net.cristae_analysis import compute_crista_skeleton
        voxel_size = 2.5
        tube = np.zeros((40, 40, 40), dtype=bool)
        tube[18:22, 18:22, 5:35] = True
        vertices, _ = compute_crista_skeleton(tube, voxel_size=voxel_size)
        indices = np.round(vertices / voxel_size).astype(int)
        self.assertTrue((indices >= 0).all() and (indices < np.array(tube.shape)).all())
        self.assertTrue(tube[tuple(indices.T)].all())

    def test_empty_and_2d(self):
        from synapse_net.cristae_analysis import compute_crista_skeleton
        vertices, is_terminus = compute_crista_skeleton(np.zeros((8, 8, 8), dtype=bool), voxel_size=1.0)
        self.assertEqual(vertices.shape, (0, 3))
        self.assertEqual(is_terminus.shape, (0,))
        _, _, edges = compute_crista_skeleton(
            np.zeros((8, 8, 8), dtype=bool), voxel_size=1.0, return_edges=True
        )
        self.assertEqual(edges.shape, (0, 2))
        with self.assertRaises(ValueError):
            compute_crista_skeleton(np.zeros((8, 8), dtype=bool), voxel_size=1.0)


@_DENSE_REQUIRED
class TestSkeletonJunctionsDenseCristae(unittest.TestCase):
    """The known over-detection, pinned as tested behaviour rather than left as a surprise.

    On a mitochondrion with densely packed cristae, "crista within max_extension of the membrane" is a
    common condition that mostly does not mean a junction is there. Skeleton mode therefore reports
    far more junctions than have any literal membrane contact. That is a property of the premise, not
    a defect these tests should hide: they assert the invariant that must hold (skeleton is a superset
    of overlap) and a loose ceiling that would catch a runaway regression, without pinning the exact
    count.
    """

    MEMBRANE_NM = 8.0
    VOXEL_NM = 0.8681

    @classmethod
    def setUpClass(cls):
        import h5py
        from synapse_net.cristae_analysis import approximate_membrane
        with h5py.File(_DENSE_H5, "r") as f:
            mito = (f["labels/mitochondria"][_DENSE_BBOX] == 1).astype(np.uint8)
            crista = f["labels/cristae"][_DENSE_BBOX].astype(bool)
        cls.mito = mito
        cls.crista = crista & (mito > 0)
        cls.membrane, _ = approximate_membrane(
            mito, cls.VOXEL_NM, membrane_thickness_nm=cls.MEMBRANE_NM, n_jobs=-1, return_lumen=True
        )

    def _skeleton(self, **kwargs):
        from synapse_net.cristae_analysis import detect_junctions_skeleton
        kwargs.setdefault("max_extension_nm", self.MEMBRANE_NM)
        return detect_junctions_skeleton(self.crista, self.membrane, self.VOXEL_NM, **kwargs)[1]

    def test_skeleton_is_a_superset_of_overlap(self):
        # The invariant. A region overlapping the membrane has a closest approach of 0, so it is
        # always within max_extension: skeleton mode can never report fewer junctions than overlap.
        from synapse_net.cristae_analysis import detect_contact_sites
        _, overlap = detect_contact_sites(self.crista, self.membrane, self.VOXEL_NM)
        skeleton = self._skeleton()
        self.assertGreater(overlap["crista_junction_count"], 0)
        self.assertGreaterEqual(
            skeleton["crista_junction_count"], overlap["crista_junction_count"]
        )

    def test_over_detection_is_real_but_bounded(self):
        # Documents the limitation: on this mito the count is several times the number of literal
        # contacts. The ceiling is deliberately loose — it catches a runaway regression, not a
        # legitimate change in the count.
        from synapse_net.cristae_analysis import detect_contact_sites
        _, overlap = detect_contact_sites(self.crista, self.membrane, self.VOXEL_NM)
        count = self._skeleton()["crista_junction_count"]
        self.assertGreater(count, overlap["crista_junction_count"])
        self.assertLess(count, 60)

    def test_no_junction_lands_in_the_border_zone(self):
        # The alignment invariant on real data. approximate_membrane has 0 voxels in the border zone,
        # so overlap mode structurally cannot report a junction there; skeleton mode must match.
        from synapse_net.cristae_analysis import _border_zone, _gap_radius, detect_junctions_skeleton
        border_radius = _gap_radius(self.VOXEL_NM, self.MEMBRANE_NM, None, 3)
        zone = _border_zone(self.crista.shape, border_radius)
        self.assertEqual(int(np.count_nonzero(self.membrane & zone)), 0)  # premise
        labels, _ = detect_junctions_skeleton(
            self.crista, self.membrane, self.VOXEL_NM,
            max_extension_nm=self.MEMBRANE_NM, border_radius=border_radius,
        )
        self.assertEqual(int(np.count_nonzero((labels > 0) & zone)), 0)

    def test_min_junction_volume_is_monotonic(self):
        counts = [
            self._skeleton(min_junction_volume_nm3=v)["crista_junction_count"]
            for v in (0.0, 50.0, 200.0, 500.0)
        ]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertGreater(counts[0], counts[-1])


class TestBorderZoneAlignmentEndToEnd(unittest.TestCase):
    """The border exclusion through the full per-mito path, where the arrays are bbox CROPS.

    This is the plumbing that matters: ``_single_mito_row`` must pass both ``border_radius`` and
    ``boundary`` down. Passing neither reproduces the reported bug; passing ``border_radius`` without
    ``boundary`` would over-correct and delete junctions at interior bbox faces instead.

    The fixture is a mitochondrion spanning Z completely, so its Z faces are genuine volume faces
    (``boundary = [[True, True], [False, False], [False, False]]``) while Y and X are interior. That
    gives one fixture that exercises both directions at once.
    """

    VOXEL_NM = 1.5
    MEMBRANE_NM = 8.0
    SHAPE = (40, 60, 60)

    @classmethod
    def setUpClass(cls):
        from synapse_net.cristae_analysis import _gap_radius, approximate_membrane
        cls.gap = _gap_radius(cls.VOXEL_NM, cls.MEMBRANE_NM, None, 3)
        mito = np.zeros(cls.SHAPE, dtype=np.uint32)
        mito[:, 8:52, 8:52] = 1          # spans Z -> both Z faces are volume faces
        cls.mito = mito
        cls.membrane, cls.lumen = approximate_membrane(
            mito, cls.VOXEL_NM, membrane_thickness_nm=cls.MEMBRANE_NM, return_lumen=True
        )

    def _crista_hugging_low_y_shell(self, z_start):
        """A slab against the low-Y membrane shell, starting at ``z_start``."""
        crista = np.zeros(self.SHAPE, dtype=bool)
        crista[z_start:z_start + self.gap, 8:16, 20:40] = True
        return crista & (self.mito > 0)

    def _stats_count(self, crista):
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        table = compute_mito_crista_statistics(
            crista, self.mito, self.VOXEL_NM,
            membrane_mask=self.membrane, lumen_mask=self.lumen,
            membrane_thickness_nm=self.MEMBRANE_NM, junction_mode="skeleton",
        )
        return int(table["crista_junction_count"][0])

    def test_membrane_is_absent_from_the_border_zone(self):
        # The premise of the whole fix, pinned: approximate_membrane leaves nothing in the zone.
        from synapse_net.cristae_analysis import _border_zone
        zone = _border_zone(self.SHAPE, self.gap)
        self.assertEqual(int(np.count_nonzero(self.membrane & zone)), 0)
        self.assertGreater(int(self.membrane.sum()), 0)

    def test_junction_in_the_z_border_zone_is_not_counted(self):
        # The reported bug, end to end. The crista lies wholly inside the low-Z border zone, where the
        # membrane was removed as unknown. Overlap mode reports nothing there; skeleton mode must agree.
        from synapse_net.cristae_analysis import _border_zone, detect_contact_sites, detect_junctions_skeleton
        crista = self._crista_hugging_low_y_shell(z_start=0)
        zone = _border_zone(self.SHAPE, self.gap)
        self.assertEqual(int(np.count_nonzero(crista & ~zone)), 0)  # guard: wholly inside the zone

        _, overlap = detect_contact_sites(crista, self.membrane, self.VOXEL_NM)
        self.assertEqual(overlap["crista_junction_count"], 0)       # the alignment target

        _, unguarded = detect_junctions_skeleton(
            crista, self.membrane, self.VOXEL_NM,
            max_extension_nm=self.MEMBRANE_NM, border_radius=0,
        )
        self.assertEqual(unguarded["crista_junction_count"], 1)     # the bug, without the exclusion
        self.assertEqual(self._stats_count(crista), 0)              # fixed, through the stats path

    def test_junction_away_from_the_z_faces_is_still_counted(self):
        # The other half: Y and X bbox faces are interior, so the exclusion must not touch them. The
        # same crista shape at mid-Z must survive — if `boundary` were dropped this would go to 0.
        from synapse_net.cristae_analysis import _border_zone
        crista = self._crista_hugging_low_y_shell(z_start=18)
        zone = _border_zone(self.SHAPE, self.gap)
        self.assertEqual(int(np.count_nonzero(crista & zone)), 0)   # guard: clear of the zone
        self.assertEqual(self._stats_count(crista), 1)

@_CUTOUT_REQUIRED
class TestSkeletonJunctionsRealData(unittest.TestCase):
    """Regression on a real mitochondrion — the case every synthetic fixture missed.

    An earlier tangent-extension implementation reported 2 junctions here while the mitochondrion has
    3, silently dropping a 36-voxel crista-membrane contact. Every synthetic test passed at the time,
    which is why this one reads the real cutout: the failure mode only appears on real sheet-like
    cristae, where a skeleton end's tangent points 142-146 degrees away from the nearby contact.
    """

    MEMBRANE_NM = 8.0

    @classmethod
    def setUpClass(cls):
        import h5py
        from synapse_net.cristae_analysis import approximate_membrane
        with h5py.File(_CUTOUT_H5, "r") as f:
            cls.mito = f["labels/mitochondria"][:]
            cls.crista = f["labels/cristae"][:].astype(bool) & (f["labels/mitochondria"][:] > 0)
        cls.voxel_size = 0.8681
        cls.membrane, cls.lumen = approximate_membrane(
            cls.mito, cls.voxel_size, membrane_thickness_nm=cls.MEMBRANE_NM,
            n_jobs=-1, return_lumen=True,
        )

    def test_ground_truth_is_three_contacts(self):
        # Pin the fixture itself: the mito has exactly 3 regions where the crista reaches the
        # membrane, and the overlap detector agrees. If this changes the expectations below are stale.
        from scipy.ndimage import label as ndimage_label
        from synapse_net.cristae_analysis import detect_contact_sites
        connectivity = np.ones((3, 3, 3), dtype=bool)
        n_overlap = ndimage_label(self.crista & self.membrane, structure=connectivity)[1]
        self.assertEqual(n_overlap, 3)
        _, summary = detect_contact_sites(self.crista, self.membrane, self.voxel_size)
        self.assertEqual(summary["crista_junction_count"], 3)

    def test_skeleton_mode_finds_every_contact(self):
        # The actual regression. Skeleton mode must not be worse than overlap mode: it has to label
        # all three literal-overlap components and count three junctions.
        from scipy.ndimage import label as ndimage_label
        from synapse_net.cristae_analysis import detect_junctions_skeleton
        connectivity = np.ones((3, 3, 3), dtype=bool)
        overlap_labels, n_overlap = ndimage_label(self.crista & self.membrane, structure=connectivity)
        labels, summary = detect_junctions_skeleton(
            self.crista, self.membrane, self.voxel_size, max_extension_nm=self.MEMBRANE_NM,
        )
        for component in range(1, n_overlap + 1):
            with self.subTest(overlap_component=component):
                self.assertTrue((labels[overlap_labels == component] > 0).any())
        self.assertEqual(summary["crista_junction_count"], 3)

    def test_labels_stay_inside_the_mitochondrion(self):
        from synapse_net.cristae_analysis import detect_junctions_skeleton
        labels, _ = detect_junctions_skeleton(
            self.crista, self.membrane, self.voxel_size, max_extension_nm=self.MEMBRANE_NM,
        )
        junction = labels > 0
        self.assertTrue(junction.any())
        self.assertEqual(int(np.count_nonzero(junction & ~(self.mito > 0))), 0)
        self.assertEqual(int(np.count_nonzero(junction & ~(self.membrane | self.crista))), 0)


class TestJunctionModeDispatch(unittest.TestCase):
    """The detect_junctions dispatcher and the junction_mode wiring into the stats table."""

    @classmethod
    def setUpClass(cls):
        cls.mito, cls.membrane, cls.lumen = _make_hollow_mito()
        # Unclipped, so this sheet overlaps the membrane band and BOTH modes find junctions.
        cls.crista = _make_crista_sheet(cls.lumen, y_range=(8, 52), clip_to_lumen=False)
        cls.voxel_size = _SHEET_VOXEL_NM

    def test_overlap_mode_matches_detect_contact_sites(self):
        # The default path must be a pure pass-through, apart from the added extension key.
        from synapse_net.cristae_analysis import detect_contact_sites, detect_junctions
        expected_labels, expected = detect_contact_sites(self.crista, self.membrane, self.voxel_size)
        labels, summary = detect_junctions(self.crista, self.membrane, self.voxel_size)
        np.testing.assert_array_equal(labels, expected_labels)
        for key, value in expected.items():
            self.assertEqual(summary[key], value)
        self.assertTrue(np.isnan(summary["mean_junction_extension_nm"]))

    def test_invalid_junction_mode_raises(self):
        from synapse_net.cristae_analysis import detect_junctions
        with self.assertRaises(ValueError):
            detect_junctions(self.crista, self.membrane, self.voxel_size, junction_mode="nope")

    def test_stats_columns_identical_between_modes(self):
        # The CSV schema must not depend on the mode — only the values may.
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        frames = {}
        for mode in ("overlap", "skeleton"):
            frames[mode] = compute_mito_crista_statistics(
                self.crista, self.mito, self.voxel_size,
                membrane_mask=self.membrane, lumen_mask=self.lumen,
                membrane_thickness_nm=_SHEET_MEMBRANE_NM, junction_mode=mode,
            )
        self.assertEqual(list(frames["overlap"].columns), list(frames["skeleton"].columns))
        self.assertIn("mean_junction_extension_nm", frames["overlap"].columns)
        self.assertTrue(np.isnan(frames["overlap"]["mean_junction_extension_nm"][0]))
        self.assertFalse(np.isnan(frames["skeleton"]["mean_junction_extension_nm"][0]))
        for mode in ("overlap", "skeleton"):
            self.assertGreater(int(frames[mode]["crista_junction_count"][0]), 0)

    def test_max_extension_defaults_to_membrane_thickness(self):
        # A None max_extension_nm must inherit membrane_thickness_nm, like border_gap_nm does.
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        crista = _make_crista_sheet(self.lumen, y_range=(16, 44))
        kwargs = dict(
            membrane_mask=self.membrane, lumen_mask=self.lumen,
            membrane_thickness_nm=_SHEET_MEMBRANE_NM, junction_mode="skeleton",
        )
        default = compute_mito_crista_statistics(crista, self.mito, self.voxel_size, **kwargs)
        explicit = compute_mito_crista_statistics(
            crista, self.mito, self.voxel_size, max_extension_nm=_SHEET_MEMBRANE_NM, **kwargs
        )
        self.assertEqual(
            int(default["crista_junction_count"][0]), int(explicit["crista_junction_count"][0])
        )
        # A tiny tolerance cannot bridge the ~4.5 nm gap, so the count must collapse.
        tight = compute_mito_crista_statistics(
            crista, self.mito, self.voxel_size, max_extension_nm=0.5, **kwargs
        )
        self.assertEqual(int(tight["crista_junction_count"][0]), 0)

    def test_invalid_junction_mode_raises_in_stats(self):
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        with self.assertRaises(ValueError):
            compute_mito_crista_statistics(
                self.crista, self.mito, self.voxel_size, junction_mode="nope"
            )


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

    # These exercise compute_junction_distances directly. The metric is defined on the lumen surface,
    # so the mesh must be supplied explicitly (there is no membrane-band fallback); here the membrane's
    # own surface is meshed as the test fixture. They require the bioimage-cpp geodesic API.
    def test_geodesic_follows_bent_membrane(self):
        # An L-shaped membrane: the geodesic around the bend is longer than the straight line
        # between the two seed voxels.
        from synapse_net.cristae_analysis import compute_junction_distances, _surface_mesh
        shape = (5, 40, 40)
        membrane = np.zeros(shape, dtype=bool)
        z = 2
        membrane[z, 4:7, 5:35] = True     # horizontal arm (a few voxels wide → meshable)
        membrane[z, 5:35, 33:36] = True   # vertical arm (shares the corner)
        labels = np.zeros(shape, dtype=np.int32)
        labels[z, 5, 6] = 1               # near the far end of the horizontal arm
        labels[z, 33, 34] = 2             # near the far end of the vertical arm
        verts, faces = _surface_mesh(membrane, np.ones(3))
        dist, _ = compute_junction_distances(
            labels, membrane, voxel_size=1.0, mesh_vertices=verts, mesh_faces=faces
        )
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

    def test_clustered_index_lower_than_dispersed(self):
        # Same membrane/area and junction count, but tightly grouped vs evenly spread:
        # the clustered arrangement must give a smaller Clark-Evans index.
        from synapse_net.cristae_analysis import compute_junction_distances, _surface_mesh
        area = 40.0 * 40.0
        clustered_pos = [(18, 18), (18, 20), (20, 18), (20, 20)]
        dispersed_pos = [(8, 8), (8, 30), (30, 8), (30, 30)]

        def _cjd(positions):
            labels, membrane = self._flat_membrane_with_junctions(positions)
            verts, faces = _surface_mesh(membrane, np.ones(3))
            return compute_junction_distances(
                labels, membrane, voxel_size=1.0, surface_area_nm2=area,
                mesh_vertices=verts, mesh_faces=faces,
            )

        _, clustered = _cjd(clustered_pos)
        _, dispersed = _cjd(dispersed_pos)
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
        # The precomputed distance is built with the same backend the internal path uses
        # (bioimage_cpp.distance.distance_transform), so passing it in must reproduce the internally
        # computed result bit-for-bit — this checks the reuse plumbing, not cross-backend agreement.
        from bioimage_cpp.distance import distance_transform
        from synapse_net.cristae_analysis import compute_crista_proximity
        crista = np.zeros((20, 20, 20), dtype=bool)
        membrane = np.zeros((20, 20, 20), dtype=bool)
        crista[10, 10, 10] = True
        crista[8, 12, 9] = True
        membrane[10, 10, 5] = True
        vs = {"z": 2.0, "y": 1.5, "x": 1.5}
        sampling = np.array([2.0, 1.5, 1.5])
        precomputed = distance_transform(~membrane, sampling=sampling.tolist(), number_of_threads=1)
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

    def test_default_orientation_is_skip(self):
        # The library default method is "skip": orientation is NaN even with a crista present.
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        df = compute_mito_crista_statistics(_make_crista(), _make_mito(), voxel_size=1.0)
        self.assertTrue(np.isnan(df["crista_orientation_anisotropy"].iloc[0]))

    def test_fast_orientation_is_finite_with_crista(self):
        # Fast orientation is computed (downsampled), so it is finite when a crista is present.
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        mito_seg = _make_mito()
        crista = _make_crista()
        df = compute_mito_crista_statistics(crista, mito_seg, voxel_size=1.0, method="fast")
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

    The lumen surface is the border-suppression-free eroded interior from
    ``approximate_membrane(..., return_lumen=True)``, threaded through ``_single_mito_row`` (it falls
    back to ``mito & ~membrane`` only when a caller supplies their own membrane and no lumen). When
    the bioimage-cpp geodesic API is unavailable the junction columns are NaN (no fallback).
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

    def test_mesh_junction_distances_finite(self):
        from synapse_net.cristae_analysis import compute_mito_crista_statistics
        crista, mito = self._mito_with_cristae()
        df = compute_mito_crista_statistics(crista, mito, 2.0, method="skip")
        self.assertGreaterEqual(int(df["crista_junction_count"].iloc[0]), 2)
        mean_nn = df["mean_nn_junction_distance_nm"].iloc[0]
        self.assertTrue(np.isfinite(mean_nn) and mean_nn > 0)
        self.assertTrue(np.isfinite(df["junction_clustering_index"].iloc[0]))

    def test_clipped_mito_open_mesh_finite(self):
        # A mito clipped by the volume boundary (spans the full z extent) meshes its lumen OPEN at the
        # clipped z-caps. Guards that the bioimage-cpp geodesic solver accepts the resulting
        # non-watertight (open) mesh and still returns finite junction distances (no spurious NaN), and
        # that the clipped mito's outer surface area is smaller than the same mito meshed closed.
        from synapse_net.cristae_analysis import compute_mito_crista_statistics, _surface_area
        shape = (24, 40, 40)
        mito = np.zeros(shape, dtype="uint32")
        mito[:, 6:34, 6:34] = 1  # spans z=0..23 → clipped at both z faces
        crista = np.zeros(shape, dtype=bool)
        for x in (12, 20, 28):
            crista[8:16, 8:11, x:x + 3] = True
        df = compute_mito_crista_statistics(crista, mito, 1.0, method="skip")
        self.assertGreaterEqual(int(df["crista_junction_count"].iloc[0]), 2)
        mean_nn = df["mean_nn_junction_distance_nm"].iloc[0]
        self.assertTrue(np.isfinite(mean_nn) and mean_nn > 0)  # open mesh accepted by the solver

        mito_binary = mito > 0
        ndim = mito.ndim
        open_faces = np.zeros((ndim, 2), dtype=bool)  # z (and all) faces clipped here
        area_open = _surface_area(mito_binary, np.ones(ndim), closed_faces=open_faces)
        area_closed = _surface_area(mito_binary, np.ones(ndim))
        self.assertLess(area_open, area_closed)  # the fabricated z-caps are no longer counted

    def test_no_mesh_supplied_is_nan(self):
        # With no lumen mesh supplied the junction distances are NaN — there is no membrane-band
        # fallback mesh (the metric is defined on the eroded-mito lumen surface).
        from synapse_net.cristae_analysis import compute_junction_distances
        labels, membrane = TestJunctionDistances._flat_membrane_with_junctions(
            [(20, 8), (20, 28), (8, 20), (32, 20)]
        )
        _, summary = compute_junction_distances(labels, membrane, 1.0, surface_area_nm2=1000.0)
        self.assertEqual(summary["junction_count"], 4)
        self.assertTrue(np.isnan(summary["mean_nn_junction_distance_nm"]))
        self.assertTrue(np.isnan(summary["median_nn_junction_distance_nm"]))
        self.assertTrue(np.isnan(summary["junction_clustering_index"]))


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


class TestCristaeAnalysisCLI(unittest.TestCase):
    """The headless CLI helper: file discovery, loading, and CSV output (no network)."""

    @staticmethod
    def _write_tif(path, array):
        import imageio.v3 as imageio
        imageio.imwrite(path, array)

    def test_single_file_pair(self):
        import tempfile
        import pandas as pd
        from synapse_net.tools.cli import cristae_analysis_helper

        mito_seg = _make_mito()
        crista = _make_crista().astype("uint8")
        with tempfile.TemporaryDirectory() as tmp:
            crista_path = os.path.join(tmp, "crista.tif")
            mito_path = os.path.join(tmp, "mito.tif")
            out_dir = os.path.join(tmp, "out")
            self._write_tif(crista_path, crista)
            self._write_tif(mito_path, mito_seg)

            cristae_analysis_helper(
                crista_path, mito_path, out_dir, voxel_size=1.0, method="skip",
            )

            csv_path = os.path.join(out_dir, "crista_cristae_analysis.csv")
            self.assertTrue(os.path.exists(csv_path), msg=f"missing output: {csv_path}")
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 1)
            for col in _EXPECTED_COLUMNS:
                self.assertIn(col, df.columns, msg=f"Missing column: {col}")

    def test_directory_batch(self):
        import tempfile
        import pandas as pd
        from synapse_net.tools.cli import cristae_analysis_helper

        with tempfile.TemporaryDirectory() as tmp:
            crista_dir = os.path.join(tmp, "crista")
            mito_dir = os.path.join(tmp, "mito")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(crista_dir)
            os.makedirs(mito_dir)
            for name in ("sample_1", "sample_2"):
                self._write_tif(os.path.join(crista_dir, f"{name}.tif"), _make_crista().astype("uint8"))
                self._write_tif(os.path.join(mito_dir, f"{name}.tif"), _make_mito())

            cristae_analysis_helper(crista_dir, mito_dir, out_dir, voxel_size=1.0, method="skip")

            for name in ("sample_1", "sample_2"):
                csv_path = os.path.join(out_dir, f"{name}_cristae_analysis.csv")
                self.assertTrue(os.path.exists(csv_path), msg=f"missing output: {csv_path}")
                self.assertEqual(len(pd.read_csv(csv_path)), 1)

    def test_requires_voxel_size_or_tomogram(self):
        import tempfile
        from synapse_net.tools.cli import cristae_analysis_helper

        with tempfile.TemporaryDirectory() as tmp:
            crista_path = os.path.join(tmp, "crista.tif")
            mito_path = os.path.join(tmp, "mito.tif")
            self._write_tif(crista_path, _make_crista().astype("uint8"))
            self._write_tif(mito_path, _make_mito())
            with self.assertRaises(ValueError):
                cristae_analysis_helper(crista_path, mito_path, os.path.join(tmp, "out"))

    def test_mismatched_input_counts(self):
        import tempfile
        from synapse_net.tools.cli import cristae_analysis_helper

        with tempfile.TemporaryDirectory() as tmp:
            crista_dir = os.path.join(tmp, "crista")
            mito_dir = os.path.join(tmp, "mito")
            os.makedirs(crista_dir)
            os.makedirs(mito_dir)
            self._write_tif(os.path.join(crista_dir, "a.tif"), _make_crista().astype("uint8"))
            self._write_tif(os.path.join(crista_dir, "b.tif"), _make_crista().astype("uint8"))
            self._write_tif(os.path.join(mito_dir, "a.tif"), _make_mito())
            with self.assertRaises(ValueError):
                cristae_analysis_helper(crista_dir, mito_dir, os.path.join(tmp, "out"), voxel_size=1.0)


if __name__ == "__main__":
    if "--view" in sys.argv:
        sys.argv.remove("--view")
        os.environ[VIEW_ENV] = "1"
    unittest.main()
