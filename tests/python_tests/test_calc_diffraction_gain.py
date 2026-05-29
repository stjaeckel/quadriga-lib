# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing your package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

from quadriga_lib import RTtools


def build_cube_mesh():
    """
    Build a [-1,1]^3 cube as 12 triangles matching the MEX test geometry.
    """
    cube = np.array(
        [
            [-1, 1, 1, 1, -1, 1, 1, 1, 1],  #  1 Top NorthEast
            [1, -1, 1, -1, -1, -1, 1, -1, -1],  #  2 South Lower
            [-1, -1, 1, -1, 1, -1, -1, -1, -1],  #  3 West Lower
            [1, 1, -1, -1, -1, -1, -1, 1, -1],  #  4 Bottom NorthWest
            [1, 1, 1, 1, -1, -1, 1, 1, -1],  #  5 East Lower
            [-1, 1, 1, 1, 1, -1, -1, 1, -1],  #  6 North Lower
            [-1, 1, 1, -1, -1, 1, 1, -1, 1],  #  7 Top SouthWest
            [1, -1, 1, -1, -1, 1, -1, -1, -1],  #  8 South Upper
            [-1, -1, 1, -1, 1, 1, -1, 1, -1],  #  9 West Upper
            [1, 1, -1, 1, -1, -1, -1, -1, -1],  # 10 Bottom SouthEast
            [1, 1, 1, 1, -1, 1, 1, -1, -1],  # 11 East Upper
            [-1, 1, 1, 1, 1, 1, 1, 1, -1],  # 12 North Upper
        ],
        dtype=np.float64,
    )
    return cube


class TestCalcDiffractionGain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cube = build_cube_mesh()
        cls.mtl_prop = np.tile([1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0], (12, 1))

        cls.orig = np.array(
            [
                [-10.0, 0.0, 0.5],  # path 0: FBS West Upper (9), SBS East Upper (11)
                [10.0, 0.0, -0.5],  # path 1: FBS East Lower (5), SBS West Lower (3)
            ]
        )
        cls.dest = np.array(
            [
                [10.0, 0.0, 0.5],
                [-10.0, 0.0, -0.5],
            ]
        )

        cls.expected_gain = np.array([10 ** (-0.3), 10 ** (-0.3)])

    # Basic diffraction gain, lod = 0
    def test_basic_lod0(self):
        gain, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0
        )
        npt.assert_allclose(gain, self.expected_gain, atol=1e-14)

    # 2 outputs, lod = 5
    def test_lod5_with_coord(self):
        gain, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=5
        )
        npt.assert_allclose(gain, self.expected_gain, atol=1e-14)

        expected_coord = np.array([[[0.0, 0.0]], [[0.0, 0.0]], [[0.5, -0.5]]])
        # lod=5 -> n_seg=1, coord shape: (3, 1, 2)
        npt.assert_allclose(coord, expected_coord, atol=1e-14)

    # LOS (unobstructed) path: TX and RX above cube, gain ~ 1.0
    def test_los_unobstructed(self):
        orig_los = np.array([[0.0, 0.0, 5.0]])
        dest_los = np.array([[0.0, 0.0, 10.0]])
        gain, _ = RTtools.calc_diffraction_gain(
            orig_los, dest_los, self.cube, self.mtl_prop, 1e9, lod=2
        )
        npt.assert_allclose(gain, [1.0], atol=1e-6)

    # Single-precision inputs (adapter should cast to double)
    def test_single_precision_inputs(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2
        )

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig.astype(np.float32),
            self.dest.astype(np.float32),
            self.cube.astype(np.float32),
            self.mtl_prop.astype(np.float32),
            1e9,
            lod=2,
        )
        npt.assert_allclose(gain, gain_ref, atol=1e-5)

    # Output shapes
    def test_output_shapes(self):
        n_pos = self.orig.shape[0]

        gain, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2
        )
        self.assertEqual(gain.shape, (n_pos,))
        self.assertEqual(coord.shape, (3, 2, n_pos))  # lod 2 -> n_seg=2

    #  Coord dimensions for each lod value
    def test_coord_shape_lod1(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=1
        )
        self.assertEqual(coord.shape, (3, 2, 2))  # n_seg=2

    def test_coord_shape_lod2(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2
        )
        self.assertEqual(coord.shape, (3, 2, 2))  # n_seg=2

    def test_coord_shape_lod3(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=3
        )
        self.assertEqual(coord.shape, (3, 3, 2))  # n_seg=3

    def test_coord_shape_lod4(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=4
        )
        self.assertEqual(coord.shape, (3, 4, 2))  # n_seg=4

    def test_coord_shape_lod6(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=6
        )
        self.assertEqual(coord.shape, (3, 1, 2))  # n_seg=1

    # Empty sub_mesh_index
    def test_empty_sub_mesh_index(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0
        )

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig,
            self.dest,
            self.cube,
            self.mtl_prop,
            1e9,
            lod=0,
            sub_mesh_index=np.array([], dtype=np.uint32),
        )
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    #  Valid sub_mesh_index (single sub-mesh covering all triangles)
    def test_single_submesh(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0
        )

        smi = np.zeros(1, dtype=np.uint32)  # one sub-mesh starting at 0
        gain, _ = RTtools.calc_diffraction_gain(
            self.orig,
            self.dest,
            self.cube,
            self.mtl_prop,
            1e9,
            lod=0,
            sub_mesh_index=smi,
        )
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    # sub_mesh_index dtype variants
    def test_submesh_index_int32(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig,
            self.dest,
            self.cube,
            self.mtl_prop,
            1e9,
            lod=0,
            sub_mesh_index=np.zeros(1, dtype=np.uint32),
        )

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig,
            self.dest,
            self.cube,
            self.mtl_prop,
            1e9,
            lod=0,
            sub_mesh_index=np.zeros(1, dtype=np.int32),
        )
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    def test_submesh_index_int64(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig,
            self.dest,
            self.cube,
            self.mtl_prop,
            1e9,
            lod=0,
            sub_mesh_index=np.zeros(1, dtype=np.uint32),
        )

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig,
            self.dest,
            self.cube,
            self.mtl_prop,
            1e9,
            lod=0,
            sub_mesh_index=np.zeros(1, dtype=np.int64),
        )
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    #  Kernel selector: GENERIC (1)
    def test_kernel_generic(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2
        )

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2, use_kernel=1
        )
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    #  Kernel selector: AVX2 (2) — may not be available
    def test_kernel_avx2(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2
        )

        try:
            gain, _ = RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2, use_kernel=2
            )
        except ValueError:
            return

        npt.assert_allclose(gain, gain_ref, atol=1e-5)

    #  Kernel selector: CUDA (3) — may not be available
    def test_kernel_cuda(self):
        try:
            RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2, use_kernel=3
            )
        except ValueError:
            pass

    #  All 10 args with explicit gpu_id
    def test_all_args_explicit(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2
        )

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig,
            self.dest,
            self.cube,
            self.mtl_prop,
            1e9,
            lod=2,
            verbose=0,
            sub_mesh_index=np.array([], dtype=np.uint32),
            use_kernel=1,
            gpu_id=0,
        )
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    #  Single TX-RX pair — minimal batch
    def test_single_pair(self):
        gain, coord = RTtools.calc_diffraction_gain(
            self.orig[:1], self.dest[:1], self.cube, self.mtl_prop, 1e9, lod=2
        )
        self.assertEqual(gain.shape, (1,))
        self.assertEqual(coord.shape, (3, 2, 1))

    #  F-contiguous vs C-contiguous inputs
    def test_f_vs_c_contiguous(self):
        gain_c, _ = RTtools.calc_diffraction_gain(
            np.ascontiguousarray(self.orig),
            np.ascontiguousarray(self.dest),
            np.ascontiguousarray(self.cube),
            np.ascontiguousarray(self.mtl_prop),
            1e9,
            lod=2,
        )

        gain_f, _ = RTtools.calc_diffraction_gain(
            np.asfortranarray(self.orig),
            np.asfortranarray(self.dest),
            np.asfortranarray(self.cube),
            np.asfortranarray(self.mtl_prop),
            1e9,
            lod=2,
        )

        npt.assert_allclose(gain_c, gain_f, atol=1e-14)

    #  Error: wrong dest size
    def test_error_wrong_dest_size(self):
        with self.assertRaises(ValueError) as ctx:
            RTtools.calc_diffraction_gain(
                self.orig, self.dest[:1], self.cube, self.mtl_prop, 1e9, lod=0
            )
        self.assertIn("orig", str(ctx.exception).lower())

    #  Error: wrong mtl_prop rows
    def test_error_wrong_mtl_prop_rows(self):
        with self.assertRaises(ValueError) as ctx:
            RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_prop[:1], 1e9, lod=0
            )
        self.assertIn("mesh", str(ctx.exception).lower())

    #  Error: sub_mesh_index first element not 0
    def test_error_submesh_not_starting_at_zero(self):
        with self.assertRaises(ValueError) as ctx:
            RTtools.calc_diffraction_gain(
                self.orig,
                self.dest,
                self.cube,
                self.mtl_prop,
                1e9,
                lod=0,
                sub_mesh_index=np.array([1], dtype=np.uint32),
            )
        self.assertIn("sub-mesh", str(ctx.exception).lower())

    #  Error: sub_mesh_index exceeds mesh count
    def test_error_submesh_exceeds_mesh(self):
        with self.assertRaises(ValueError) as ctx:
            RTtools.calc_diffraction_gain(
                self.orig,
                self.dest,
                self.cube,
                self.mtl_prop,
                1e9,
                lod=0,
                sub_mesh_index=np.array([0, 32], dtype=np.uint32),
            )
        self.assertIn("sub-mesh", str(ctx.exception).lower())

    #  Different center frequency should produce different gain
    def test_different_frequency(self):
        # attB = 1 → linear frequency scaling of penetration loss
        mtl_freq = np.tile([1.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0], (12, 1))
        gain_1g, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, mtl_freq, 1e9, lod=2
        )
        gain_10g, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, mtl_freq, 10e9, lod=2
        )
        self.assertFalse(np.allclose(gain_1g, gain_10g))

    # In-medium distance absorption
    def test_alpha_in_medium_absorption(self):
        # eps_r=1 (no Fresnel), sigma=0, att=0, alpha=4 dB/m, exponents=0, fRef=1
        mtl_alpha = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0], (12, 1))
        orig_in = np.array([[-10.0, 0.0, 0.5]])
        dest_in = np.array([[0.5, 0.0, 0.5]])  # ends inside cube
        gain, _ = RTtools.calc_diffraction_gain(
            orig_in, dest_in, self.cube, mtl_alpha, 10e9, lod=0
        )
        npt.assert_allclose(gain, [10 ** (-0.6)], atol=1e-10)

    # Penetration-loss frequency scaling
    def test_attB_frequency_scaling(self):
        mtl_attB = np.tile([1.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0], (12, 1))
        orig_in = np.array([[-10.0, 0.0, 0.5]])
        dest_in = np.array([[0.5, 0.0, 0.5]])
        gain, _ = RTtools.calc_diffraction_gain(
            orig_in, dest_in, self.cube, mtl_attB, 10e9, lod=0
        )
        npt.assert_allclose(gain, [10 ** (-1.5)], atol=1e-10)

    # fRef parameterization equivalence
    def test_fref_equivalence(self):
        # At every f: eps_r=1.5*f, sigma=0.001*f, att=2*f dB, alpha=0.5*f dB/m
        mat_A = np.tile(
            [1.5, 1.0, 0.001, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0], (12, 1)
        )  # fRef=1
        mat_B = np.tile(
            [3.0, 1.0, 0.002, 1.0, 4.0, 1.0, 1.0, 1.0, 2.0], (12, 1)
        )  # fRef=2
        orig_in = np.array([[-10.0, 0.0, 0.5]])
        dest_in = np.array([[0.5, 0.0, 0.5]])
        gain_A, _ = RTtools.calc_diffraction_gain(
            orig_in, dest_in, self.cube, mat_A, 10e9, lod=3
        )
        gain_B, _ = RTtools.calc_diffraction_gain(
            orig_in, dest_in, self.cube, mat_B, 10e9, lod=3
        )
        npt.assert_allclose(gain_A, gain_B, atol=1e-12)

    # Normal incidence: scalar ≡ EM. At normal incidence R_TE = R_TM, so ½(|R_TE|² + |R_TM|²) = |R_TE|².
    # The two modes must be bit-identical.
    def test_scalar_normal_incidence_equivalence(self):
        mtl = np.tile([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        orig = np.array([[-10.0, 0.0, 0.5]])
        dest = np.array([[0.5, 0.0, 0.5]])  # ray along +x → normal hit on west wall
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=False
        )
        gain_sc, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=True
        )
        npt.assert_allclose(gain_em, gain_sc, atol=1e-12)

    # Oblique incidence: scalar ≠ EM, both physical
    def test_scalar_oblique_incidence_diverges(self):
        mtl = np.tile([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        orig = np.array([[-3.0, 0.0, 0.5]])
        dest = np.array([[0.5, 1.5, 0.5]])  # ~23° off-normal on west wall
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=False
        )
        gain_sc, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=True
        )
        self.assertFalse(np.allclose(gain_em, gain_sc, atol=1e-4))
        self.assertTrue(0.0 < gain_em[0] < 1.0)
        self.assertTrue(0.0 < gain_sc[0] < 1.0)

    # Inequality direction at oblique incidence
    # For a non-magnetic dielectric |R_TM|² ≤ |R_TE|² always, so EM transmission ≥ scalar transmission,
    # with strict inequality off-normal. This is a much tighter constraint than “they differ”.
    def test_scalar_em_inequality_direction(self):
        # High contrast → clear margin between TE and TM transmission
        mtl = np.tile([4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        orig = np.array([[-3.0, 0.0, 0.5]])
        dest = np.array([[0.5, 1.5, 0.5]])
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=False
        )
        gain_sc, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=True
        )
        self.assertGreater(gain_em[0], gain_sc[0])

    # No Fresnel contrast (eps_r = 1): scalar ≡ EM at any angle. Both R_TE and R_TM vanish when there
    # is no permittivity contrast — the polarization branch becomes irrelevant.
    def test_scalar_no_fresnel_contrast_invariance(self):
        # eps_r = 1, only penetration loss
        mtl = np.tile([1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        orig = np.array([[-3.0, 0.0, 0.5], [-3.0, 0.0, 0.0]])
        dest = np.array([[0.5, 1.5, 0.5], [0.5, 0.5, 0.0]])
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=False
        )
        gain_sc, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=True
        )
        npt.assert_allclose(gain_em, gain_sc, atol=1e-14)

    # Pure in-medium absorption (alpha): scalar ≡ EM.
    # Distance absorption is polarization-independent — a path that ends inside the medium isolates the alpha branch.
    def test_scalar_alpha_only_invariance(self):
        mtl = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0], (12, 1))
        orig = np.array([[-10.0, 0.0, 0.5]])
        dest = np.array([[0.5, 0.0, 0.5]])  # ends inside cube
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=False
        )
        gain_sc, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=True
        )
        npt.assert_allclose(gain_em, gain_sc, atol=1e-14)

    # Lossy material at normal incidence: still equal.
    # Conductivity makes eta complex but still keeps R_TE = R_TM at normal incidence — guards against
    # a regression where complex-arithmetic paths split the polarizations spuriously.
    def test_scalar_lossy_normal_incidence(self):
        mtl = np.tile(
            [4.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1)
        )  # sigma = 0.5
        orig = np.array([[-10.0, 0.0, 0.5]])
        dest = np.array([[0.5, 0.0, 0.5]])
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=False
        )
        gain_sc, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=True
        )
        npt.assert_allclose(gain_em, gain_sc, atol=1e-12)

    # Default flag value is locked. Picks an oblique high-contrast geometry where the two modes are clearly
    # distinguishable, so this test will fail loudly if the default ever flips.
    def test_scalar_default_value(self):
        mtl = np.tile([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        orig = np.array([[-3.0, 0.0, 0.5]])
        dest = np.array([[0.5, 1.5, 0.5]])
        gain_default, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0
        )
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=False
        )
        npt.assert_allclose(gain_default, gain_em, atol=1e-14)  # default = EM

    # Multi-segment paths (lod > 0): coords match, gains finite.
    # The scalar/EM split only affects the field amplitude, never the geometry. This catches regressions 
    # where the scalar code path takes a different ray-state machine branch.
    def test_scalar_with_lod_multipath(self):
        mtl = np.tile([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        gain_em, coord_em = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, mtl, 10e9, lod=3, scalar_mode=False)
        gain_sc, coord_sc = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, mtl, 10e9, lod=3, scalar_mode=True)
        npt.assert_allclose(coord_em, coord_sc, atol=1e-6)       # geometry identical
        self.assertTrue(np.all(np.isfinite(gain_em)))
        self.assertTrue(np.all(np.isfinite(gain_sc)))
        self.assertTrue(np.all(gain_em > 0.0))
        self.assertTrue(np.all(gain_sc > 0.0))

    #  LOS path: scalar mode irrelevant.
    # Pretty much pure regression — guards against the scalar branch accidentally affecting unobstructed paths.
    def test_scalar_los_unaffected(self):
        mtl = np.tile([4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        orig = np.array([[0.0, 0.0,  5.0]])
        dest = np.array([[0.0, 0.0, 10.0]])
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=False)
        gain_sc, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl, 10e9, lod=0, scalar_mode=True)
        npt.assert_allclose(gain_em, [1.0], atol=1e-12)
        npt.assert_allclose(gain_sc, [1.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
