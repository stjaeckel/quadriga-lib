# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


def m2p(M):
    # Convert a per-face (n_face, 9) material matrix with columns
    # (a, b, c, d, att, attB, alpha, alphaB, fRef) into the (mtl_ind, dict) pair
    # the API expects. Identical rows are deduplicated; mtl_ind is 1-based
    # (index 0 is reserved for "no material").
    names = ['a', 'b', 'c', 'd', 'att', 'attB', 'alpha', 'alphaB', 'fRef']
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    uniq = []
    mtl_ind = np.zeros(n, dtype=np.uint64)
    for f in range(n):
        hit = 0
        for m in range(len(uniq)):
            if np.all(M[f, :] == uniq[m]):
                hit = m + 1  # 1-based
                break
        if hit == 0:
            uniq.append(M[f, :].copy())
            hit = len(uniq)  # 1-based index of the row just added
        mtl_ind[f] = hit
    uniq = np.array(uniq, dtype=float) if uniq else np.zeros((0, M.shape[1]))
    prop = {names[c]: uniq[:, c].copy() for c in range(9)}
    return mtl_ind, prop


class test_case(unittest.TestCase):

    def setUp(self):
        # 2x2x2 box at origin (+/-1), 12 triangles.
        self.cube = quadriga_lib.RTtools.cube()

        # Base material: att = 3 dB at fRef = 1 GHz, eps_r = 1 (no Fresnel), no
        # other loss and no frequency scaling.
        base = np.tile([1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        self.mtl_ind, self.mtl_prop = m2p(base)

        # Two straight-through paths (enter west wall, exit east wall). att is
        # charged once per body on the entry face -> a single 3 dB loss each.
        self.orig = np.array([[-10.0, 0.0, 0.5], [10.0, 0.0, -0.5]])
        self.dest = np.array([[10.0, 0.0, 0.5], [-10.0, 0.0, -0.5]])

        # Reused single-path geometries.
        self.orig_in = np.array([[-10.0, 0.0, 0.5]])   # enters west wall, ends inside
        self.dest_in = np.array([[0.5, 0.0, 0.5]])
        self.orig_clear = np.array([[-10.0, 0.0, 0.5]])  # entirely left of the cube
        self.dest_clear = np.array([[-5.0, 0.0, 0.5]])

    # --- Output plumbing ---

    def test_minimal_args(self):
        # Six args (the required minimum) must run; all three outputs are always
        # returned regardless of how the caller unpacks them.
        result = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9)
        self.assertEqual(len(result), 3)

    def test_basic_gain_lod0(self):
        # Per-body att -> 3 dB per path.
        gain, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 0, 0)
        npt.assert_allclose(gain, [10 ** -0.3, 10 ** -0.3], atol=1e-10, rtol=0)

    def test_output_shapes(self):
        # gain (n_pos,), xprmat EM (8, n_pos).
        gain, xprmat, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 2)
        self.assertEqual(gain.shape, (2,))
        self.assertEqual(xprmat.shape, (8, 2))

        # lod = 5 -> n_seg = 1 (path midpoints).
        gain5, xpr5, coord5 = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 5)
        npt.assert_allclose(gain5, [10 ** -0.3, 10 ** -0.3], atol=1e-6, rtol=0)
        self.assertEqual(xpr5.shape, (8, 2))

        expected = np.zeros((3, 1, 2))
        expected[:, 0, 0] = [0.0, 0.0, 0.5]
        expected[:, 0, 1] = [0.0, 0.0, -0.5]
        npt.assert_allclose(coord5, expected, atol=1e-10, rtol=0)

    def test_coord_dimensions(self):
        # coord is (3, n_seg, n_pos); n_seg = 2 (lod 1,2), 3 (lod 3), 4 (lod 4),
        # 1 (lod 5,6).
        lods = [1, 2, 3, 4, 5, 6]
        n_segs = [2, 2, 3, 4, 1, 1]
        for lod, n_seg in zip(lods, n_segs):
            _, _, coord = quadriga_lib.RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, lod)
            self.assertEqual(coord.shape, (3, n_seg, 2),
                             f'coord dimension mismatch at lod {lod}')

    # --- Input casting & acceleration passthrough ---

    def test_input_casting_single(self):
        # Single-precision inputs are cast to double internally.
        gain_ref, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 2)
        gain_single, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig.astype(np.float32), self.dest.astype(np.float32),
            self.cube.astype(np.float32), self.mtl_ind, self.mtl_prop, 1e9, 2)
        npt.assert_allclose(gain_ref, gain_single, atol=1e-5, rtol=0)

    def test_sub_mesh_passthrough(self):
        gain_ref, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 0)

        # None (no sub-mesh) runs without error.
        quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 0,
            sub_mesh_index=None)

        # A single whole-mesh sub-mesh at offset 0 matches the no-sub-mesh result.
        gain_smi, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 0,
            sub_mesh_index=np.array([0], dtype=np.uint32))
        npt.assert_allclose(gain_ref, gain_smi, atol=1e-10, rtol=0)

    def test_sub_mesh_dtype_cast(self):
        # Non-uint32 sub_mesh_index (e.g. numpy's default int64) is cast and
        # gives an identical result.
        gain_u32, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 0,
            sub_mesh_index=np.array([0], dtype=np.uint32))
        gain_i64, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 0,
            sub_mesh_index=np.array([0], dtype=np.int64))
        npt.assert_allclose(gain_u32, gain_i64, atol=1e-10, rtol=0)

    def test_kernel_selection(self):
        gain_ref, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 2)

        # use_kernel = 1 (GENERIC) matches the default (auto) kernel.
        gain_generic, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 2,
            use_kernel=1)
        npt.assert_allclose(gain_ref, gain_generic, atol=1e-10, rtol=0)

        # Full acceleration args (use_kernel + gpu_id) match the default.
        gain_full, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 2,
            use_kernel=1, gpu_id=0)
        npt.assert_allclose(gain_ref, gain_full, atol=1e-10, rtol=0)

    # --- xprmat sanity ---

    def test_xprmat_clear_path(self):
        # Clear path (entirely outside) -> gain 1 and identity Jones matrix. Row
        # layout is col-major 2x2: [ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH].
        g_clr, xpr_clr, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig_clear, self.dest_clear, self.cube, self.mtl_ind, self.mtl_prop, 10e9, 0)
        npt.assert_allclose(g_clr, 1.0, atol=1e-12, rtol=0)
        self.assertEqual(xpr_clr.shape, (8, 1))
        npt.assert_allclose(xpr_clr[:, 0], [1, 0, 0, 0, 0, 0, 1, 0], atol=1e-12, rtol=0)

    def test_xprmat_dielectric(self):
        # Normal-incidence transmission into a lossless dielectric (eps_r = 2):
        # real Fresnel loss (gain < 1) but TE == TM, so the normalized Jones
        # matrix is still identity and 0.5*sum|xpr|^2 == 1 holds.
        eps2 = np.tile([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        eps2_ind, eps2_prop = m2p(eps2)
        g_e2, xpr_e2, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig_in, self.dest_in, self.cube, eps2_ind, eps2_prop, 10e9, 0)

        xpr = xpr_e2[:, 0]
        R0 = (1 - np.sqrt(2)) / (1 + np.sqrt(2))
        npt.assert_allclose(g_e2, 1 - R0 ** 2, atol=1e-9, rtol=0)
        npt.assert_allclose(abs(xpr[0] + 1j * xpr[1]), 1.0, atol=1e-9, rtol=0)  # |VV|
        npt.assert_allclose(abs(xpr[6] + 1j * xpr[7]), 1.0, atol=1e-9, rtol=0)  # |HH|
        npt.assert_allclose(xpr[2:6], np.zeros(4), atol=1e-9, rtol=0)           # off-diagonals
        npt.assert_allclose(0.5 * np.sum(xpr ** 2), 1.0, atol=1e-9, rtol=0)     # normalization

    # --- scalar_mode ---

    def test_scalar_mode(self):
        eps2 = np.tile([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        eps2_ind, eps2_prop = m2p(eps2)

        # EM reference at normal incidence (TE == TM).
        g_e2, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig_in, self.dest_in, self.cube, eps2_ind, eps2_prop, 10e9, 0)

        # Scalar transmission collapses xprmat to (2, n_pos) and equals EM here.
        g_sc, xpr_sc, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig_in, self.dest_in, self.cube, eps2_ind, eps2_prop, 10e9, 0,
            scalar_mode=True)
        npt.assert_allclose(g_sc, g_e2, atol=1e-9, rtol=0)
        self.assertEqual(xpr_sc.shape, (2, 1))

        # Scalar clear path -> gain 1, coefficient (1, 0).
        g_sc_c, xpr_sc_c, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig_clear, self.dest_clear, self.cube, eps2_ind, eps2_prop, 10e9, 0,
            scalar_mode=True)
        npt.assert_allclose(g_sc_c, 1.0, atol=1e-12, rtol=0)
        self.assertEqual(xpr_sc_c.shape, (2, 1))
        npt.assert_allclose(xpr_sc_c[:, 0], [1.0, 0.0], atol=1e-12, rtol=0)

    # --- Material index 0 (no material) ---

    def test_material_index_zero(self):
        # Index 0 means "no material": the face is intersected but applies no
        # transition. All-zero indices -> fully transparent (gain 1). Exercises
        # the wrapper passing mtl_ind through 1-based with no decrement.
        mtl_ind_zero = np.zeros(12, dtype=np.uint64)
        gain_zero, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, mtl_ind_zero, self.mtl_prop, 10e9, 0)
        npt.assert_allclose(gain_zero, [1.0, 1.0], atol=1e-12, rtol=0)

        # Same geometry, real lossy material (att = 6 dB) clearly attenuates.
        lossy = np.tile([1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        lossy_ind, lossy_prop = m2p(lossy)
        gain_lossy, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, lossy_ind, lossy_prop, 10e9, 0)
        self.assertTrue(np.all(gain_lossy < 0.5))

    # --- Physics sanity ---

    def test_physics_los(self):
        # LOS (unobstructed) path above the cube -> gain ~ 1.
        gain_los, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            np.array([[0.0, 0.0, 5.0]]), np.array([[0.0, 0.0, 10.0]]),
            self.cube, self.mtl_ind, self.mtl_prop, 1e9, 2)
        npt.assert_allclose(gain_los, 1.0, atol=1e-6, rtol=0)

    def test_physics_alpha(self):
        # In-medium distance absorption: eps_r = 1, alpha = 4 dB/m. Path enters at
        # x = -1 and ends at x = 0.5 -> 1.5 m in medium -> 6 dB -> 10^(-0.6).
        alpha = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0], (12, 1))
        alpha_ind, alpha_prop = m2p(alpha)
        gain_alpha, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig_in, self.dest_in, self.cube, alpha_ind, alpha_prop, 10e9, 0)
        npt.assert_allclose(gain_alpha, 10 ** -0.6, atol=1e-7, rtol=0)

    def test_physics_freq_scaling(self):
        # Penetration loss frequency scaling: att = 3 dB at fRef = 2 GHz, attB = 1.
        # At 10 GHz -> 3*(10/2)^1 = 15 dB -> 10^(-1.5).
        attB = np.tile([1.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0], (12, 1))
        attB_ind, attB_prop = m2p(attB)
        gain_attB, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig_in, self.dest_in, self.cube, attB_ind, attB_prop, 10e9, 0)
        npt.assert_allclose(gain_attB, 10 ** -1.5, atol=1e-10, rtol=0)

    def test_physics_fref_equivalence(self):
        # Two materials specified at different reference frequencies but
        # numerically identical at every frequency must give identical gain.
        # lod = 3 exercises the multi-arc / multi-hit ray-state machine.
        mat_A = np.tile([1.5, 1.0, 0.001, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0], (12, 1))  # fRef = 1 GHz
        mat_B = np.tile([3.0, 1.0, 0.002, 1.0, 4.0, 1.0, 1.0, 1.0, 2.0], (12, 1))  # fRef = 2 GHz
        matA_ind, matA_prop = m2p(mat_A)
        matB_ind, matB_prop = m2p(mat_B)
        gain_A, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig_in, self.dest_in, self.cube, matA_ind, matA_prop, 10e9, 3)
        gain_B, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            self.orig_in, self.dest_in, self.cube, matB_ind, matB_prop, 10e9, 3)
        npt.assert_allclose(gain_A, gain_B, atol=1e-12, rtol=0)

    # --- thin_slab_threshold (Fabry-Perot plumbing) ---

    def test_thin_slab_threshold(self):
        # Thin lossless dielectric slab tuned to half-wave optical thickness at
        # 10 GHz. Resolution on (threshold = 0) keeps the internal interference;
        # resolution off (threshold = 1) discards it. The two must differ, and the
        # half-wave resonance transmits more than the incoherent result. Exact
        # Airy transmittance values are covered in the Catch2 suite.
        f = 10e9
        n = 1.5                                # eps_r = 2.25
        t_half = 299792458.0 / f / (2 * n)     # half-wave thickness (~1 cm)
        slab = quadriga_lib.RTtools.cube(
            np.array([t_half / 2, 5.0, 5.0]), None, np.array([t_half / 2, 0.0, 2.001]))
        slab_mtl = np.tile([n ** 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        slab_ind, slab_prop = m2p(slab_mtl)

        orig_fp = np.array([[-10.0, 0.0, 0.0]])
        dest_fp = np.array([[10.0, 0.0, 0.0]])  # normal incidence through both faces
        g_on, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            orig_fp, dest_fp, slab, slab_ind, slab_prop, f, 0, thin_slab_threshold=0.0)
        g_off, _, _ = quadriga_lib.RTtools.calc_diffraction_gain(
            orig_fp, dest_fp, slab, slab_ind, slab_prop, f, 0, thin_slab_threshold=1.0)

        self.assertTrue(0.0 < g_on[0] <= 1.0 + 1e-9)
        self.assertTrue(0.0 < g_off[0] <= 1.0 + 1e-9)
        self.assertTrue(g_on[0] > g_off[0] + 1e-3)  # resonance enhances transmission

    # --- Error handling ---
    # pybind11 re-raises std::exception as RuntimeError; the reference tests match
    # only that an exception is raised, not the message text (library-layer
    # strings are being refactored). MEX-only guards (nlhs / nrhs counts, the
    # "1-based sub_mesh cannot be 0" check) do not apply to the Python binding.

    def test_error_missing_required_arg(self):
        # center_frequency is required (no default).
        with self.assertRaises(Exception):
            quadriga_lib.RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop)

    def test_error_orig_dest_row_mismatch(self):
        with self.assertRaises(Exception):
            quadriga_lib.RTtools.calc_diffraction_gain(
                self.orig, self.dest[0:1, :], self.cube, self.mtl_ind, self.mtl_prop, 1e9, 0)

    def test_error_mtl_ind_length(self):
        with self.assertRaises(Exception):
            quadriga_lib.RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_ind[0:1], self.mtl_prop, 1e9, 0)

    def test_error_sub_mesh_first_nonzero(self):
        # First sub-mesh must start at index 0 (0-based in Python).
        with self.assertRaises(Exception):
            quadriga_lib.RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 0,
                sub_mesh_index=np.array([1], dtype=np.uint32))

    def test_error_sub_mesh_exceeds_faces(self):
        with self.assertRaises(Exception):
            quadriga_lib.RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 0,
                sub_mesh_index=np.array([0, 32], dtype=np.uint32))

    def test_error_orig_not_three_cols(self):
        with self.assertRaises(Exception):
            quadriga_lib.RTtools.calc_diffraction_gain(
                self.orig[:, 0:2], self.dest, self.cube, self.mtl_ind, self.mtl_prop, 1e9, 0)

    def test_error_mesh_not_nine_cols(self):
        with self.assertRaises(Exception):
            quadriga_lib.RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube[:, 0:8], self.mtl_ind, self.mtl_prop, 1e9, 0)

    # Best-effort guards (mirroring the MEX "(guess)" cases) — remove or adjust
    # if the refactored library does not enforce them.

    def test_error_nonpositive_frequency(self):
        with self.assertRaises(Exception):
            quadriga_lib.RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_prop, 0.0, 0)

    def test_error_material_index_too_large(self):
        mtl_ind_big = self.mtl_ind.copy()
        mtl_ind_big[0] = 40000  # exceeds the int16 material-index cap (32767)
        with self.assertRaises(Exception):
            quadriga_lib.RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, mtl_ind_big, self.mtl_prop, 1e9, 0)


if __name__ == '__main__':
    unittest.main()