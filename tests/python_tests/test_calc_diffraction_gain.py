import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing your package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

from quadriga_lib import RTtools


def build_cube_mesh():
    """
    Build a [-1,1]^3 cube as 12 triangles matching the MEX test geometry.
    """
    cube = np.array([
        [-1, 1, 1,   1,-1, 1,   1, 1, 1],   #  1 Top NorthEast
        [ 1,-1, 1,  -1,-1,-1,   1,-1,-1],    #  2 South Lower
        [-1,-1, 1,  -1, 1,-1,  -1,-1,-1],    #  3 West Lower
        [ 1, 1,-1,  -1,-1,-1,  -1, 1,-1],    #  4 Bottom NorthWest
        [ 1, 1, 1,   1,-1,-1,   1, 1,-1],    #  5 East Lower
        [-1, 1, 1,   1, 1,-1,  -1, 1,-1],    #  6 North Lower
        [-1, 1, 1,  -1,-1, 1,   1,-1, 1],    #  7 Top SouthWest
        [ 1,-1, 1,  -1,-1, 1,  -1,-1,-1],    #  8 South Upper
        [-1,-1, 1,  -1, 1, 1,  -1, 1,-1],    #  9 West Upper
        [ 1, 1,-1,   1,-1,-1,  -1,-1,-1],    # 10 Bottom SouthEast
        [ 1, 1, 1,   1,-1, 1,   1,-1,-1],    # 11 East Upper
        [-1, 1, 1,   1, 1, 1,   1, 1,-1],    # 12 North Upper
    ], dtype=np.float64)
    return cube


class TestCalcDiffractionGain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cube = build_cube_mesh()
        cls.mtl_prop = np.tile([1.0, 0.0, 0.0, 0.0, 3.0], (12, 1))

        cls.orig = np.array([
            [-10.0,  0.0,  0.5],   # path 0: FBS West Upper (9), SBS East Upper (11)
            [ 10.0,  0.0, -0.5],   # path 1: FBS East Lower (5), SBS West Lower (3)
        ])
        cls.dest = np.array([
            [ 10.0,  0.0,  0.5],
            [-10.0,  0.0, -0.5],
        ])

        cls.expected_gain = np.array([10**(-0.3), 10**(-0.3)])

    # ------------------------------------------------------------------ #
    #  Basic diffraction gain, lod = 0                                    #
    # ------------------------------------------------------------------ #
    def test_basic_lod0(self):
        gain, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0)
        npt.assert_allclose(gain, self.expected_gain, atol=1e-14)

    # ------------------------------------------------------------------ #
    #  2 outputs, lod = 5                                                 #
    # ------------------------------------------------------------------ #
    def test_lod5_with_coord(self):
        gain, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=5)
        npt.assert_allclose(gain, self.expected_gain, atol=1e-14)

        expected_coord = np.array([[[0.0, 0.0]], [[0.0, 0.0]], [[0.5, -0.5]]])
        # lod=5 -> n_seg=1, coord shape: (3, 1, 2)
        npt.assert_allclose(coord, expected_coord, atol=1e-14)

    # ------------------------------------------------------------------ #
    #  LOS (unobstructed) path: TX and RX above cube, gain ~ 1.0         #
    # ------------------------------------------------------------------ #
    def test_los_unobstructed(self):
        orig_los = np.array([[0.0, 0.0, 5.0]])
        dest_los = np.array([[0.0, 0.0, 10.0]])
        gain, _ = RTtools.calc_diffraction_gain(
            orig_los, dest_los, self.cube, self.mtl_prop, 1e9, lod=2)
        npt.assert_allclose(gain, [1.0], atol=1e-6)

    # ------------------------------------------------------------------ #
    #  Single-precision inputs (adapter should cast to double)            #
    # ------------------------------------------------------------------ #
    def test_single_precision_inputs(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2)

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig.astype(np.float32), self.dest.astype(np.float32),
            self.cube.astype(np.float32), self.mtl_prop.astype(np.float32),
            1e9, lod=2)
        npt.assert_allclose(gain, gain_ref, atol=1e-5)

    # ------------------------------------------------------------------ #
    #  Output shapes                                                      #
    # ------------------------------------------------------------------ #
    def test_output_shapes(self):
        n_pos = self.orig.shape[0]

        gain, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2)
        self.assertEqual(gain.shape, (n_pos,))
        self.assertEqual(coord.shape, (3, 2, n_pos))   # lod 2 -> n_seg=2

    # ------------------------------------------------------------------ #
    #  Coord dimensions for each lod value                                #
    # ------------------------------------------------------------------ #
    def test_coord_shape_lod1(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=1)
        self.assertEqual(coord.shape, (3, 2, 2))   # n_seg=2

    def test_coord_shape_lod2(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2)
        self.assertEqual(coord.shape, (3, 2, 2))   # n_seg=2

    def test_coord_shape_lod3(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=3)
        self.assertEqual(coord.shape, (3, 3, 2))   # n_seg=3

    def test_coord_shape_lod4(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=4)
        self.assertEqual(coord.shape, (3, 4, 2))   # n_seg=4

    def test_coord_shape_lod6(self):
        _, coord = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=6)
        self.assertEqual(coord.shape, (3, 1, 2))   # n_seg=1

    # ------------------------------------------------------------------ #
    #  Empty sub_mesh_index                                               #
    # ------------------------------------------------------------------ #
    def test_empty_sub_mesh_index(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0)

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0,
            sub_mesh_index=np.array([], dtype=np.uint32))
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    # ------------------------------------------------------------------ #
    #  Valid sub_mesh_index (single sub-mesh covering all triangles)       #
    # ------------------------------------------------------------------ #
    def test_single_submesh(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0)

        smi = np.zeros(1, dtype=np.uint32)  # one sub-mesh starting at 0
        gain, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0,
            sub_mesh_index=smi)
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    # ------------------------------------------------------------------ #
    #  sub_mesh_index dtype variants                                      #
    # ------------------------------------------------------------------ #
    def test_submesh_index_int32(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0,
            sub_mesh_index=np.zeros(1, dtype=np.uint32))

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0,
            sub_mesh_index=np.zeros(1, dtype=np.int32))
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    def test_submesh_index_int64(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0,
            sub_mesh_index=np.zeros(1, dtype=np.uint32))

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0,
            sub_mesh_index=np.zeros(1, dtype=np.int64))
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    # ------------------------------------------------------------------ #
    #  Kernel selector: GENERIC (1)                                       #
    # ------------------------------------------------------------------ #
    def test_kernel_generic(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2)

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2,
            use_kernel=1)
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    # ------------------------------------------------------------------ #
    #  Kernel selector: AVX2 (2) — may not be available                   #
    # ------------------------------------------------------------------ #
    def test_kernel_avx2(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2)

        try:
            gain, _ = RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2,
                use_kernel=2)
        except ValueError:
            return

        npt.assert_allclose(gain, gain_ref, atol=1e-5)

    # ------------------------------------------------------------------ #
    #  Kernel selector: CUDA (3) — may not be available                   #
    # ------------------------------------------------------------------ #
    def test_kernel_cuda(self):
        try:
            RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2,
                use_kernel=3)
        except ValueError:
            pass

    # ------------------------------------------------------------------ #
    #  All 10 args with explicit gpu_id                                   #
    # ------------------------------------------------------------------ #
    def test_all_args_explicit(self):
        gain_ref, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2)

        gain, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2,
            verbose=0, sub_mesh_index=np.array([], dtype=np.uint32),
            use_kernel=1, gpu_id=0)
        npt.assert_allclose(gain, gain_ref, atol=1e-14)

    # ------------------------------------------------------------------ #
    #  Single TX-RX pair — minimal batch                                  #
    # ------------------------------------------------------------------ #
    def test_single_pair(self):
        gain, coord = RTtools.calc_diffraction_gain(
            self.orig[:1], self.dest[:1], self.cube, self.mtl_prop, 1e9, lod=2)
        self.assertEqual(gain.shape, (1,))
        self.assertEqual(coord.shape, (3, 2, 1))

    # ------------------------------------------------------------------ #
    #  F-contiguous vs C-contiguous inputs                                #
    # ------------------------------------------------------------------ #
    def test_f_vs_c_contiguous(self):
        gain_c, _ = RTtools.calc_diffraction_gain(
            np.ascontiguousarray(self.orig), np.ascontiguousarray(self.dest),
            np.ascontiguousarray(self.cube), np.ascontiguousarray(self.mtl_prop),
            1e9, lod=2)

        gain_f, _ = RTtools.calc_diffraction_gain(
            np.asfortranarray(self.orig), np.asfortranarray(self.dest),
            np.asfortranarray(self.cube), np.asfortranarray(self.mtl_prop),
            1e9, lod=2)

        npt.assert_allclose(gain_c, gain_f, atol=1e-14)

    # ------------------------------------------------------------------ #
    #  Error: wrong dest size                                             #
    # ------------------------------------------------------------------ #
    def test_error_wrong_dest_size(self):
        with self.assertRaises(ValueError) as ctx:
            RTtools.calc_diffraction_gain(
                self.orig, self.dest[:1], self.cube, self.mtl_prop, 1e9, lod=0)
        self.assertIn("orig", str(ctx.exception).lower())

    # ------------------------------------------------------------------ #
    #  Error: wrong mtl_prop columns                                      #
    # ------------------------------------------------------------------ #
    def test_error_wrong_mtl_prop_columns(self):
        with self.assertRaises(ValueError) as ctx:
            RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_prop[:, :1],
                1e9, lod=0)
        self.assertIn("mtl_prop", str(ctx.exception).lower())

    # ------------------------------------------------------------------ #
    #  Error: wrong mtl_prop rows                                         #
    # ------------------------------------------------------------------ #
    def test_error_wrong_mtl_prop_rows(self):
        with self.assertRaises(ValueError) as ctx:
            RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_prop[:1],
                1e9, lod=0)
        self.assertIn("mesh", str(ctx.exception).lower())

    # ------------------------------------------------------------------ #
    #  Error: sub_mesh_index first element not 0                          #
    # ------------------------------------------------------------------ #
    def test_error_submesh_not_starting_at_zero(self):
        with self.assertRaises(ValueError) as ctx:
            RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0,
                sub_mesh_index=np.array([1], dtype=np.uint32))
        self.assertIn("sub-mesh", str(ctx.exception).lower())

    # ------------------------------------------------------------------ #
    #  Error: sub_mesh_index exceeds mesh count                           #
    # ------------------------------------------------------------------ #
    def test_error_submesh_exceeds_mesh(self):
        with self.assertRaises(ValueError) as ctx:
            RTtools.calc_diffraction_gain(
                self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=0,
                sub_mesh_index=np.array([0, 32], dtype=np.uint32))
        self.assertIn("sub-mesh", str(ctx.exception).lower())

    # ------------------------------------------------------------------ #
    #  Different center frequency should produce different gain           #
    # ------------------------------------------------------------------ #
    def test_different_frequency(self):
        gain_1g, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 1e9, lod=2)
        gain_10g, _ = RTtools.calc_diffraction_gain(
            self.orig, self.dest, self.cube, self.mtl_prop, 10e9, lod=2)
        # Higher frequency → more attenuation around obstacle
        self.assertFalse(np.allclose(gain_1g, gain_10g))


if __name__ == "__main__":
    unittest.main()
