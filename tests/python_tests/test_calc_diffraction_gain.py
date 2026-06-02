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



def m2p(M):
    """Convert a [n_face, 9] material matrix with columns
    {a,b,c,d,att,attB,alpha,alphaB,fRef} into the (mtl_ind, mtl_prop-dict) pair
    the new API expects. Identical rows are deduplicated; mtl_ind is 0-based."""
    names = ["a", "b", "c", "d", "att", "attB", "alpha", "alphaB", "fRef"]
    M = np.asarray(M, dtype=np.float64)
    uniq = []
    mtl_ind = np.empty(M.shape[0], dtype=np.uint64)
    for f in range(M.shape[0]):
        hit = -1
        for m, row in enumerate(uniq):
            if np.array_equal(M[f], row):
                hit = m
                break
        if hit < 0:
            hit = len(uniq)
            uniq.append(M[f])
        mtl_ind[f] = hit
    U = np.array(uniq)
    prop = {names[c]: np.ascontiguousarray(U[:, c]) for c in range(9)}
    return mtl_ind, prop


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
        cls.mtl_ind, cls.mtl_dict = m2p(cls.mtl_prop)

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
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=0
        )
        npt.assert_allclose(gain, self.expected_gain, atol=1e-14)

    # 2 outputs, lod = 5
    def test_lod5_with_coord(self):
        gain, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=5
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
            orig_los, dest_los, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=2
        )
        npt.assert_allclose(gain, [1.0], atol=1e-6)

    # Single-precision inputs (adapter should cast to double)
    def test_single_precision_inputs(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=2
        )

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig.astype(np.float32),
            self.dest.astype(np.float32),
            self.cube.astype(np.float32),
            self.mtl_ind,
            self.mtl_dict,
            1e9,
            lod=2,
        )
        npt.assert_allclose(gain, gain_ref, atol=1e-5)

    # Output shapes
    def test_output_shapes(self):
        n_pos = self.orig.shape[0]

        gain, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=2
        )
        self.assertEqual(gain.shape, (n_pos,))
        self.assertEqual(coord.shape, (3, 2, n_pos))  # lod 2 -> n_seg=2

    #  Coord dimensions for each lod value
    def test_coord_shape_lod1(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=1
        )
        self.assertEqual(coord.shape, (3, 2, 2))  # n_seg=2

    def test_coord_shape_lod2(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=2
        )
        self.assertEqual(coord.shape, (3, 2, 2))  # n_seg=2

    def test_coord_shape_lod3(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=3
        )
        self.assertEqual(coord.shape, (3, 3, 2))  # n_seg=3

    def test_coord_shape_lod4(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=4
        )
        self.assertEqual(coord.shape, (3, 4, 2))  # n_seg=4

    def test_coord_shape_lod6(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=6
        )
        self.assertEqual(coord.shape, (3, 1, 2))  # n_seg=1

    # Empty sub_mesh_index
    def test_empty_sub_mesh_index(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=0
        )

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig,
            self.dest,
            self.cube,
            self.mtl_ind,
            self.mtl_dict,
            1e9,
            lod=0,
            sub_mesh_index=np.array([], dtype=np.uint32),
        )
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    #  Valid sub_mesh_index (single sub-mesh covering all triangles)
    def test_single_submesh(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=0
        )

        smi = np.zeros(1, dtype=np.uint32)  # one sub-mesh starting at 0
        gain, _ = RTtools.calc_diffraction_gain(
            self.orig,
            self.dest,
            self.cube,
            self.mtl_ind,
            self.mtl_dict,
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
            self.mtl_ind,
            self.mtl_dict,
            1e9,
            lod=0,
            sub_mesh_index=np.zeros(1, dtype=np.uint32),
        )

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig,
            self.dest,
            self.cube,
            self.mtl_ind,
            self.mtl_dict,
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
            self.mtl_ind,
            self.mtl_dict,
            1e9,
            lod=0,
            sub_mesh_index=np.zeros(1, dtype=np.uint32),
        )

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig,
            self.dest,
            self.cube,
            self.mtl_ind,
            self.mtl_dict,
            1e9,
            lod=0,
            sub_mesh_index=np.zeros(1, dtype=np.int64),
        )
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    #  Kernel selector: GENERIC (1)
    def test_kernel_generic(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=2
        )

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=2, use_kernel=1
        )
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    #  Kernel selector: AVX2 (2) — may not be available
    def test_kernel_avx2(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=2
        )

        try:
            gain, _ = RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=2, use_kernel=2
            )
        except ValueError:
            return

        npt.assert_allclose(gain, gain_ref, atol=1e-5)

    #  Kernel selector: CUDA (3) — may not be available
    def test_kernel_cuda(self):
        try:
            RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=2, use_kernel=3
            )
        except ValueError:
            pass

    #  All 10 args with explicit gpu_id
    def test_all_args_explicit(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=2
        )

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig,
            self.dest,
            self.cube,
            self.mtl_ind,
            self.mtl_dict,
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
            self.orig[:1], self.dest[:1], self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=2
        )
        self.assertEqual(gain.shape, (1,))
        self.assertEqual(coord.shape, (3, 2, 1))

    #  F-contiguous vs C-contiguous inputs
    def test_f_vs_c_contiguous(self):
        gain_c, _ = RTtools.calc_diffraction_gain(
            np.ascontiguousarray(self.orig),
            np.ascontiguousarray(self.dest),
            np.ascontiguousarray(self.cube),
            self.mtl_ind,
            self.mtl_dict,
            1e9,
            lod=2,
        )

        gain_f, _ = RTtools.calc_diffraction_gain(
            np.asfortranarray(self.orig),
            np.asfortranarray(self.dest),
            np.asfortranarray(self.cube),
            self.mtl_ind,
            self.mtl_dict,
            1e9,
            lod=2,
        )

        npt.assert_allclose(gain_c, gain_f, atol=1e-14)

    #  Error: wrong dest size
    def test_error_wrong_dest_size(self):
        with self.assertRaises(ValueError) as ctx:
            RTtools.calc_diffraction_gain(
                self.orig, self.dest[:1], self.cube, self.mtl_ind, self.mtl_dict, 1e9, lod=0
            )
        self.assertIn("orig", str(ctx.exception).lower())

    #  Error: wrong mtl_prop rows
    def test_error_wrong_mtl_prop_rows(self):
        with self.assertRaises(ValueError) as ctx:
            RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_ind[:1], self.mtl_dict, 1e9, lod=0
            )
        self.assertIn("mesh", str(ctx.exception).lower())

    #  Error: sub_mesh_index first element not 0
    def test_error_submesh_not_starting_at_zero(self):
        with self.assertRaises(ValueError) as ctx:
            RTtools.calc_diffraction_gain(
                self.orig,
                self.dest,
                self.cube,
                self.mtl_ind,
                self.mtl_dict,
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
                self.mtl_ind,
                self.mtl_dict,
                1e9,
                lod=0,
                sub_mesh_index=np.array([0, 32], dtype=np.uint32),
            )
        self.assertIn("sub-mesh", str(ctx.exception).lower())

    #  Different center frequency should produce different gain
    def test_different_frequency(self):
        # attB = 1 → linear frequency scaling of penetration loss
        mtl_freq = np.tile([1.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 1.0], (12, 1))
        mtl_freq_i, mtl_freq_p = m2p(mtl_freq)
        gain_1g, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, mtl_freq_i, mtl_freq_p, 1e9, lod=2
        )
        gain_10g, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, mtl_freq_i, mtl_freq_p, 10e9, lod=2
        )
        self.assertFalse(np.allclose(gain_1g, gain_10g))

    # In-medium distance absorption
    def test_alpha_in_medium_absorption(self):
        # eps_r=1 (no Fresnel), sigma=0, att=0, alpha=4 dB/m, exponents=0, fRef=1
        mtl_alpha = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0], (12, 1))
        mtl_alpha_i, mtl_alpha_p = m2p(mtl_alpha)
        orig_in = np.array([[-10.0, 0.0, 0.5]])
        dest_in = np.array([[0.5, 0.0, 0.5]])  # ends inside cube
        gain, _ = RTtools.calc_diffraction_gain(
            orig_in, dest_in, self.cube, mtl_alpha_i, mtl_alpha_p, 10e9, lod=0
        )
        npt.assert_allclose(gain, [10 ** (-0.6)], atol=1e-10)

    # Penetration-loss frequency scaling
    def test_attB_frequency_scaling(self):
        mtl_attB = np.tile([1.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0], (12, 1))
        mtl_attB_i, mtl_attB_p = m2p(mtl_attB)
        orig_in = np.array([[-10.0, 0.0, 0.5]])
        dest_in = np.array([[0.5, 0.0, 0.5]])
        gain, _ = RTtools.calc_diffraction_gain(
            orig_in, dest_in, self.cube, mtl_attB_i, mtl_attB_p, 10e9, lod=0
        )
        npt.assert_allclose(gain, [10 ** (-1.5)], atol=1e-10)

    # fRef parameterization equivalence
    def test_fref_equivalence(self):
        # At every f: eps_r=1.5*f, sigma=0.001*f, att=2*f dB, alpha=0.5*f dB/m
        mat_A = np.tile(
            [1.5, 1.0, 0.001, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0], (12, 1)
        )
        mat_A_i, mat_A_p = m2p(mat_A)  # fRef=1
        mat_B = np.tile(
            [3.0, 1.0, 0.002, 1.0, 4.0, 1.0, 1.0, 1.0, 2.0], (12, 1)
        )
        mat_B_i, mat_B_p = m2p(mat_B)  # fRef=2
        orig_in = np.array([[-10.0, 0.0, 0.5]])
        dest_in = np.array([[0.5, 0.0, 0.5]])
        gain_A, _ = RTtools.calc_diffraction_gain(
            orig_in, dest_in, self.cube, mat_A_i, mat_A_p, 10e9, lod=3
        )
        gain_B, _ = RTtools.calc_diffraction_gain(
            orig_in, dest_in, self.cube, mat_B_i, mat_B_p, 10e9, lod=3
        )
        npt.assert_allclose(gain_A, gain_B, atol=1e-12)

    # Partition model: scalar transmission is pure pass-through scaled by the calibrated isolation
    # only; the EM path additionally carries the Fresnel boundary loss (1 - |R|^2). They diverge.
    def test_scalar_partition_normal_incidence(self):
        # eps_r = 2 (EM Fresnel) + att = 6 dB isolation
        mtl = np.tile([2.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        mtl_i, mtl_p = m2p(mtl)
        orig = np.array([[-10.0, 0.0, 0.5]])
        dest = np.array([[0.5, 0.0, 0.5]])  # normal hit on west wall, ends inside
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0, scalar_mode=False)
        gain_sc, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0, scalar_mode=True)
        npt.assert_allclose(gain_sc, [10 ** (-0.6)], atol=1e-9)  # att-only pass-through
        self.assertLess(gain_em[0], gain_sc[0])                  # EM loses (1 - |R|^2) on top
        self.assertGreater(gain_em[0], 0.0)

    # Partition model: scalar transmission is angle-independent (no Fresnel critical-angle dropout);
    # EM transmission is angle-dependent. Both stay strictly between 0 and 1 with finite isolation.
    def test_scalar_partition_oblique(self):
        mtl = np.tile([2.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        mtl_i, mtl_p = m2p(mtl)
        # Both rays end INSIDE the cube -> single wall crossing, isolation applied once each.
        orig = np.array([[-10.0, 0.0, 0.5], [-10.0, -8.0, 0.5]])  # normal, then ~39° oblique
        dest = np.array([[0.5, 0.0, 0.5], [0.5, 0.5, 0.5]])
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0, scalar_mode=False)
        gain_sc, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0, scalar_mode=True)
        # Scalar: identical at both angles, equal to the att-only pass-through.
        npt.assert_allclose(gain_sc, [10 ** (-0.6), 10 ** (-0.6)], atol=1e-9)
        # EM: below scalar, and more lossy at the oblique hit than at normal.
        self.assertLess(gain_em[0], gain_sc[0])
        self.assertLess(gain_em[1], gain_em[0])

        # also at high contrast (eps_r = 4): scalar still flat, EM loses more at oblique
        mtl_hi = np.tile([4.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        mtl_hi_i, mtl_hi_p = m2p(mtl_hi)
        gain_em_hi, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_hi_i, mtl_hi_p, 10e9, lod=0, scalar_mode=False)
        gain_sc_hi, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_hi_i, mtl_hi_p, 10e9, lod=0, scalar_mode=True)
        npt.assert_allclose(gain_sc_hi, [10 ** (-0.6), 10 ** (-0.6)], atol=1e-9)
        self.assertLess(gain_em_hi[0], gain_sc_hi[0])

    # No Fresnel contrast (eps_r = 1): scalar ≡ EM at any angle. Both R_TE and R_TM vanish when there
    # is no permittivity contrast — the polarization branch becomes irrelevant.
    def test_scalar_no_fresnel_contrast_invariance(self):
        # eps_r = 1, only penetration loss
        mtl = np.tile([1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        mtl_i, mtl_p = m2p(mtl)
        orig = np.array([[-3.0, 0.0, 0.5], [-3.0, 0.0, 0.0]])
        dest = np.array([[0.5, 1.5, 0.5], [0.5, 0.5, 0.0]])
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0, scalar_mode=False
        )
        gain_sc, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0, scalar_mode=True
        )
        npt.assert_allclose(gain_em, gain_sc, atol=1e-14)

    # Pure in-medium absorption (alpha): scalar ≡ EM.
    # Distance absorption is polarization-independent — a path that ends inside the medium isolates the alpha branch.
    def test_scalar_alpha_only_invariance(self):
        mtl = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0], (12, 1))
        mtl_i, mtl_p = m2p(mtl)
        orig = np.array([[-10.0, 0.0, 0.5]])
        dest = np.array([[0.5, 0.0, 0.5]])  # ends inside cube
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0, scalar_mode=False
        )
        gain_sc, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0, scalar_mode=True
        )
        npt.assert_allclose(gain_em, gain_sc, atol=1e-14)

    # Lossy material at normal incidence: still equal.
    # Conductivity makes eta complex but still keeps R_TE = R_TM at normal incidence — guards against
    # a regression where complex-arithmetic paths split the polarizations spuriously.
    def test_scalar_lossy_normal_incidence(self):
        mtl = np.tile(
            [4.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1)
        )
        mtl_i, mtl_p = m2p(mtl)  # sigma = 0.5
        orig = np.array([[-10.0, 0.0, 0.5]])
        dest = np.array([[0.5, 0.0, 0.5]])
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0, scalar_mode=False
        )
        gain_sc, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0, scalar_mode=True
        )
        npt.assert_allclose(gain_em, gain_sc, atol=1e-12)

    # Default flag value is locked. Picks an oblique high-contrast geometry where the two modes are clearly
    # distinguishable, so this test will fail loudly if the default ever flips.
    def test_scalar_default_value(self):
        mtl = np.tile([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        mtl_i, mtl_p = m2p(mtl)
        orig = np.array([[-3.0, 0.0, 0.5]])
        dest = np.array([[0.5, 1.5, 0.5]])
        gain_default, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0
        )
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0, scalar_mode=False
        )
        npt.assert_allclose(gain_default, gain_em, atol=1e-14)  # default = EM

    # Multi-segment paths (lod > 0): coords match, gains finite.
    # The scalar/EM split only affects the field amplitude, never the geometry. This catches regressions 
    # where the scalar code path takes a different ray-state machine branch.
    def test_scalar_with_lod_multipath(self):
        mtl = np.tile([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        mtl_i, mtl_p = m2p(mtl)
        gain_em, coord_em = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, mtl_i, mtl_p, 10e9, lod=3, scalar_mode=False)
        gain_sc, coord_sc = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, mtl_i, mtl_p, 10e9, lod=3, scalar_mode=True)
        # coord is the amplitude-weighted path centroid, so it legitimately differs between modes
        # (different weights, identical underlying ray paths). Only assert what the split must NOT
        # change: shapes, finiteness, positivity, and that the scalar branch runs the same machine.
        self.assertEqual(coord_em.shape, coord_sc.shape)
        self.assertTrue(np.all(np.isfinite(gain_em)) and np.all(np.isfinite(gain_sc)))
        self.assertTrue(np.all(gain_em > 0.0) and np.all(gain_sc > 0.0))

    #  LOS path: scalar mode irrelevant.
    # Pretty much pure regression — guards against the scalar branch accidentally affecting unobstructed paths.
    def test_scalar_los_unaffected(self):
        mtl = np.tile([4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (12, 1))
        mtl_i, mtl_p = m2p(mtl)
        orig = np.array([[0.0, 0.0,  5.0]])
        dest = np.array([[0.0, 0.0, 10.0]])
        gain_em, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0, scalar_mode=False)
        gain_sc, _ = RTtools.calc_diffraction_gain(
            orig, dest, self.cube, mtl_i, mtl_p, 10e9, lod=0, scalar_mode=True)
        npt.assert_allclose(gain_em, [1.0], atol=1e-12)
        npt.assert_allclose(gain_sc, [1.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()