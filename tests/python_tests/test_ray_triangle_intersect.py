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
    Triangle ordering defines the expected 1-based hit indices.
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


class TestRayTriangleIntersect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cube = build_cube_mesh()

        cls.orig = np.array([
            [-10.0,  0.0,  0.5],  # ray 0: FBS West Upper (9), SBS East Upper (11)
            [ 10.0,  0.0, -0.5],  # ray 1: FBS East Lower (5), SBS West Lower (3)
            [-10.0,  0.0,  2.0],  # ray 2: miss
            [  0.5,  0.0, 10.0],  # ray 3: FBS Top NE (1), SBS Bottom SE (10)
            [  0.0, 10.0, -0.5],  # ray 4: FBS North Lower (6), SBS South Lower (2)
            [  0.0,  0.0,  0.0],  # ray 5: co-located origin==dest
        ])
        cls.dest = np.array([
            [ 10.0,  0.0,  0.5],
            [-10.0,  0.0, -0.5],
            [ 10.0,  0.0,  2.0],
            [  0.5,  0.0,-10.0],
            [  0.0,-10.0, -0.5],
            [  0.0,  0.0,  0.0],
        ])

    # ------------------------------------------------------------------ #
    #  Basic intersection correctness (matches MEX test)                  #
    # ------------------------------------------------------------------ #
    def test_basic_intersections(self):
        atol = 1e-6
        fbs, sbs, no_hit, ifbs, isbs = RTtools.ray_triangle_intersect(
            self.orig, self.dest, self.cube)

        # ray 0: two hits, west/east
        npt.assert_allclose(fbs[0], [-1, 0, 0.5], atol=atol)
        npt.assert_allclose(sbs[0], [ 1, 0, 0.5], atol=atol)
        self.assertEqual(int(no_hit[0]), 2)
        self.assertEqual(int(ifbs[0]), 9)
        self.assertEqual(int(isbs[0]), 11)

        # ray 1: two hits, east/west reversed
        npt.assert_allclose(fbs[1], [ 1, 0, -0.5], atol=atol)
        npt.assert_allclose(sbs[1], [-1, 0, -0.5], atol=atol)
        self.assertEqual(int(no_hit[1]), 2)
        self.assertEqual(int(ifbs[1]), 5)
        self.assertEqual(int(isbs[1]), 3)

        # ray 2: miss
        npt.assert_allclose(fbs[2], self.dest[2], atol=atol)
        npt.assert_allclose(sbs[2], self.dest[2], atol=atol)
        self.assertEqual(int(no_hit[2]), 0)
        self.assertEqual(int(ifbs[2]), 0)
        self.assertEqual(int(isbs[2]), 0)

        # ray 3: top/bottom
        npt.assert_allclose(fbs[3], [0.5, 0, 1], atol=atol)
        npt.assert_allclose(sbs[3], [0.5, 0, -1], atol=atol)
        self.assertEqual(int(no_hit[3]), 2)
        self.assertEqual(int(ifbs[3]), 1)
        self.assertEqual(int(isbs[3]), 10)

        # ray 4: north/south
        npt.assert_allclose(fbs[4], [0, 1, -0.5], atol=atol)
        npt.assert_allclose(sbs[4], [0, -1, -0.5], atol=atol)
        self.assertEqual(int(no_hit[4]), 2)
        self.assertEqual(int(ifbs[4]), 6)
        self.assertEqual(int(isbs[4]), 2)

        # ray 5: co-located, no hit
        npt.assert_allclose(fbs[5], [0, 0, 0], atol=atol)
        npt.assert_allclose(sbs[5], [0, 0, 0], atol=atol)
        self.assertEqual(int(no_hit[5]), 0)
        self.assertEqual(int(ifbs[5]), 0)
        self.assertEqual(int(isbs[5]), 0)

    # ------------------------------------------------------------------ #
    #  Output shapes                                                      #
    # ------------------------------------------------------------------ #
    def test_output_shapes(self):
        fbs, sbs, no_hit, ifbs, isbs = RTtools.ray_triangle_intersect(
            self.orig, self.dest, self.cube)
        n = self.orig.shape[0]
        self.assertEqual(fbs.shape, (n, 3))
        self.assertEqual(sbs.shape, (n, 3))
        self.assertEqual(no_hit.shape, (n,))
        self.assertEqual(ifbs.shape, (n,))
        self.assertEqual(isbs.shape, (n,))

    # ------------------------------------------------------------------ #
    #  NaN handling in orig/dest                                          #
    # ------------------------------------------------------------------ #
    def test_nan_in_orig_dest(self):
        nan_row = np.array([[np.nan, np.nan, np.nan]])
        orig2 = np.vstack([nan_row, self.orig[1:], nan_row])
        dest2 = np.vstack([self.dest, nan_row])

        fbs, sbs, no_hit, ifbs, isbs = RTtools.ray_triangle_intersect(
            orig2, dest2, self.cube)

        # First and last rows should be NaN / zero-hit
        for i in [0, orig2.shape[0] - 1]:
            self.assertTrue(np.all(np.isnan(fbs[i])))
            self.assertTrue(np.all(np.isnan(sbs[i])))
            self.assertEqual(int(no_hit[i]), 0)
            self.assertEqual(int(ifbs[i]), 0)
            self.assertEqual(int(isbs[i]), 0)

        # Row 1 should still match ray 1 from the basic test
        npt.assert_allclose(fbs[1], [1, 0, -0.5], atol=1e-6)
        npt.assert_allclose(sbs[1], [-1, 0, -0.5], atol=1e-6)
        self.assertEqual(int(no_hit[1]), 2)
        self.assertEqual(int(ifbs[1]), 5)
        self.assertEqual(int(isbs[1]), 3)

    # ------------------------------------------------------------------ #
    #  NaN in mesh (last row, no effect on hits)                          #
    # ------------------------------------------------------------------ #
    def test_nan_in_mesh_last_row(self):
        mesh2 = self.cube.copy()
        mesh2[-1] = np.nan

        fbs_ref, sbs_ref, no_hit_ref, ifbs_ref, isbs_ref = RTtools.ray_triangle_intersect(
            self.orig, self.dest, self.cube)
        fbs, sbs, no_hit, ifbs, isbs = RTtools.ray_triangle_intersect(
            self.orig, self.dest, mesh2)

        npt.assert_allclose(fbs, fbs_ref, atol=1e-7)
        npt.assert_allclose(sbs, sbs_ref, atol=1e-7)
        npt.assert_array_equal(no_hit, no_hit_ref)
        npt.assert_array_equal(ifbs, ifbs_ref)
        npt.assert_array_equal(isbs, isbs_ref)

    # ------------------------------------------------------------------ #
    #  NaN in mesh (first row, affects ray 3 FBS which hits tri 1)        #
    # ------------------------------------------------------------------ #
    def test_nan_in_mesh_first_row(self):
        mesh2 = self.cube.copy()
        mesh2[0] = np.nan

        fbs, sbs, no_hit, ifbs, isbs = RTtools.ray_triangle_intersect(
            self.orig, self.dest, mesh2)

        # ray 3 originally hit tri 1 at FBS; now tri 1 is NaN → only SBS remains
        npt.assert_allclose(fbs[3], [0.5, 0, -1], atol=1e-6)
        npt.assert_allclose(sbs[3], self.dest[3], atol=1e-6)
        self.assertEqual(int(no_hit[3]), 1)
        self.assertEqual(int(ifbs[3]), 10)
        self.assertEqual(int(isbs[3]), 0)

    # ------------------------------------------------------------------ #
    #  Input validation: column mismatch                                  #
    # ------------------------------------------------------------------ #
    def test_error_orig_too_few_cols(self):
        with self.assertRaises(Exception) as ctx:
            RTtools.ray_triangle_intersect(self.orig[:, :2], self.dest, self.cube)
        self.assertIn("orig", str(ctx.exception).lower())

    def test_error_dest_too_few_cols(self):
        with self.assertRaises(Exception) as ctx:
            RTtools.ray_triangle_intersect(self.orig, self.dest[:, :2], self.cube)
        self.assertIn("dest", str(ctx.exception).lower())

    def test_error_mesh_too_few_cols(self):
        with self.assertRaises(Exception) as ctx:
            RTtools.ray_triangle_intersect(self.orig, self.dest, self.cube[:, :8])
        self.assertIn("mesh", str(ctx.exception).lower())

    # ------------------------------------------------------------------ #
    #  Input validation: row mismatch orig/dest                           #
    # ------------------------------------------------------------------ #
    def test_error_row_mismatch(self):
        with self.assertRaises(Exception) as ctx:
            RTtools.ray_triangle_intersect(self.orig, self.dest[1:], self.cube)
        self.assertIn("match", str(ctx.exception).lower())

    # ------------------------------------------------------------------ #
    #  Input validation: empty inputs                                     #
    # ------------------------------------------------------------------ #
    def test_error_empty_inputs(self):
        empty = np.empty((0, 3))
        with self.assertRaises(Exception):
            RTtools.ray_triangle_intersect(empty, empty, self.cube)

    # ------------------------------------------------------------------ #
    #  Sub-mesh index validation                                          #
    # ------------------------------------------------------------------ #
    def test_error_submesh_not_starting_at_zero(self):
        smi = np.array([1, 6], dtype=np.uint32)
        with self.assertRaises(Exception) as ctx:
            RTtools.ray_triangle_intersect(self.orig, self.dest, self.cube, smi)
        self.assertIn("start at index 0", str(ctx.exception))

    def test_error_submesh_not_sorted(self):
        smi = np.array([0, 6, 3], dtype=np.uint32)
        with self.assertRaises(Exception) as ctx:
            RTtools.ray_triangle_intersect(self.orig, self.dest, self.cube, smi)
        self.assertIn("ascending order", str(ctx.exception))

    def test_error_submesh_exceeds_mesh(self):
        smi = np.array([0, 99], dtype=np.uint32)
        with self.assertRaises(Exception) as ctx:
            RTtools.ray_triangle_intersect(self.orig, self.dest, self.cube, smi)
        self.assertIn("exceed", str(ctx.exception).lower())

    # ------------------------------------------------------------------ #
    #  Segmented mesh with sub-mesh index                                 #
    # ------------------------------------------------------------------ #
    def test_segmented_mesh(self):
        cube_seg, smi, _, _ = RTtools.triangle_mesh_segmentation(self.cube, 4)
        smi = smi.astype(np.uint32)

        fbs_ref, sbs_ref, no_hit_ref, _, _ = RTtools.ray_triangle_intersect(
            self.orig, self.dest, self.cube)

        fbs, sbs, no_hit, _, _ = RTtools.ray_triangle_intersect(
            self.orig, self.dest, cube_seg, smi)

        # Segmentation reorders the mesh → coordinates and hit counts must match,
        # but triangle indices may differ.
        npt.assert_allclose(fbs, fbs_ref, atol=1e-6)
        npt.assert_allclose(sbs, sbs_ref, atol=1e-6)
        npt.assert_array_equal(no_hit, no_hit_ref)

    # ------------------------------------------------------------------ #
    #  Pre-computed AABB                                                  #
    # ------------------------------------------------------------------ #
    def test_precomputed_aabb(self):
        cube_seg, smi, _, _ = RTtools.triangle_mesh_segmentation(self.cube, 4)
        smi = smi.astype(np.uint32)
        aabb = RTtools.triangle_mesh_aabb(cube_seg, smi)

        fbs_ref, sbs_ref, no_hit_ref, ifbs_ref, isbs_ref = RTtools.ray_triangle_intersect(
            self.orig, self.dest, cube_seg, smi)

        fbs, sbs, no_hit, ifbs, isbs = RTtools.ray_triangle_intersect(
            self.orig, self.dest, cube_seg, smi, aabb)

        npt.assert_allclose(fbs, fbs_ref, atol=1e-7)
        npt.assert_allclose(sbs, sbs_ref, atol=1e-7)
        npt.assert_array_equal(no_hit, no_hit_ref)
        npt.assert_array_equal(ifbs, ifbs_ref)
        npt.assert_array_equal(isbs, isbs_ref)

    # ------------------------------------------------------------------ #
    #  AABB validation errors                                             #
    # ------------------------------------------------------------------ #
    def test_error_aabb_row_mismatch(self):
        cube_seg, smi, _, _ = RTtools.triangle_mesh_segmentation(self.cube, 4)
        smi = smi.astype(np.uint32)
        aabb = RTtools.triangle_mesh_aabb(cube_seg, smi)

        with self.assertRaises(Exception) as ctx:
            RTtools.ray_triangle_intersect(self.orig, self.dest, cube_seg, smi, aabb[:1])
        self.assertIn("aabb", str(ctx.exception).lower())

    def test_error_aabb_col_mismatch(self):
        cube_seg, smi, _, _ = RTtools.triangle_mesh_segmentation(self.cube, 4)
        smi = smi.astype(np.uint32)
        aabb = RTtools.triangle_mesh_aabb(cube_seg, smi)

        with self.assertRaises(Exception) as ctx:
            RTtools.ray_triangle_intersect(self.orig, self.dest, cube_seg, smi, aabb[:, :3])
        self.assertIn("aabb", str(ctx.exception).lower())

    # ------------------------------------------------------------------ #
    #  Kernel selector: GENERIC (1)                                       #
    # ------------------------------------------------------------------ #
    def test_kernel_generic(self):
        fbs_ref, sbs_ref, no_hit_ref, ifbs_ref, isbs_ref = RTtools.ray_triangle_intersect(
            self.orig, self.dest, self.cube)

        fbs, sbs, no_hit, ifbs, isbs = RTtools.ray_triangle_intersect(
            self.orig, self.dest, self.cube, use_kernel=1)

        npt.assert_allclose(fbs, fbs_ref, atol=1e-6)
        npt.assert_allclose(sbs, sbs_ref, atol=1e-6)
        npt.assert_array_equal(no_hit, no_hit_ref)
        npt.assert_array_equal(ifbs, ifbs_ref)
        npt.assert_array_equal(isbs, isbs_ref)

    # ------------------------------------------------------------------ #
    #  Kernel selector: AVX2 (2) — may not be available                   #
    # ------------------------------------------------------------------ #
    def test_kernel_avx2(self):
        fbs_ref, sbs_ref, no_hit_ref, ifbs_ref, isbs_ref = RTtools.ray_triangle_intersect(
            self.orig, self.dest, self.cube)

        try:
            fbs, sbs, no_hit, ifbs, isbs = RTtools.ray_triangle_intersect(
                self.orig, self.dest, self.cube, use_kernel=2)
        except ValueError as e:
            self.assertIn("AVX2 kernel requested but not available", str(e))
            return

        npt.assert_allclose(fbs, fbs_ref, atol=1e-5)
        npt.assert_allclose(sbs, sbs_ref, atol=1e-5)
        npt.assert_array_equal(no_hit, no_hit_ref)
        npt.assert_array_equal(ifbs, ifbs_ref)
        npt.assert_array_equal(isbs, isbs_ref)

    # ------------------------------------------------------------------ #
    #  Kernel selector: CUDA (3) — may not be available                   #
    # ------------------------------------------------------------------ #
    def test_kernel_cuda(self):
        try:
            RTtools.ray_triangle_intersect(
                self.orig, self.dest, self.cube, use_kernel=3)
        except ValueError as e:
            self.assertIn("CUDA kernel requested but not available", str(e))

    # ------------------------------------------------------------------ #
    #  AVX2 kernel with segmented mesh + pre-computed AABB                #
    # ------------------------------------------------------------------ #
    def test_kernel_avx2_with_submesh_and_aabb(self):
        cube_seg, smi, _, _ = RTtools.triangle_mesh_segmentation(self.cube, 4)
        smi = smi.astype(np.uint32)
        aabb = RTtools.triangle_mesh_aabb(cube_seg, smi)

        fbs_ref, sbs_ref, no_hit_ref, _, _ = RTtools.ray_triangle_intersect(
            self.orig, self.dest, cube_seg, smi, aabb)

        try:
            fbs, sbs, no_hit, _, _ = RTtools.ray_triangle_intersect(
                self.orig, self.dest, cube_seg, smi, aabb, use_kernel=2)
        except ValueError as e:
            self.assertIn("AVX2 kernel requested but not available", str(e))
            return

        npt.assert_allclose(fbs, fbs_ref, atol=1e-5)
        npt.assert_allclose(sbs, sbs_ref, atol=1e-5)
        npt.assert_array_equal(no_hit, no_hit_ref)

    # ------------------------------------------------------------------ #
    #  GENERIC kernel with explicit skip of optional args                 #
    # ------------------------------------------------------------------ #
    def test_generic_kernel_skip_optional_args(self):
        fbs_ref, _, _, _, _ = RTtools.ray_triangle_intersect(
            self.orig, self.dest, self.cube)

        # Pass no sub_mesh_index, no aabb, explicit kernel=1, gpu_id=0
        fbs, _, _, _, _ = RTtools.ray_triangle_intersect(
            self.orig, self.dest, self.cube, use_kernel=1, gpu_id=0)

        npt.assert_allclose(fbs, fbs_ref, atol=1e-6)

    # ------------------------------------------------------------------ #
    #  Single-precision inputs (adapter should cast to double internally) #
    # ------------------------------------------------------------------ #
    def test_single_precision_inputs(self):
        fbs_ref, sbs_ref, no_hit_ref, ifbs_ref, isbs_ref = RTtools.ray_triangle_intersect(
            self.orig, self.dest, self.cube)

        fbs, sbs, no_hit, ifbs, isbs = RTtools.ray_triangle_intersect(
            self.orig.astype(np.float32), self.dest.astype(np.float32),
            self.cube.astype(np.float32))

        npt.assert_allclose(fbs, fbs_ref, atol=1e-5)
        npt.assert_allclose(sbs, sbs_ref, atol=1e-5)
        npt.assert_array_equal(no_hit, no_hit_ref)
        npt.assert_array_equal(ifbs, ifbs_ref)
        npt.assert_array_equal(isbs, isbs_ref)

    # ------------------------------------------------------------------ #
    #  Sub-mesh index dtype variants (int64, int32)                       #
    # ------------------------------------------------------------------ #
    def test_submesh_index_int64(self):
        cube_seg, smi, _, _ = RTtools.triangle_mesh_segmentation(self.cube, 4)

        fbs_ref, _, _, _, _ = RTtools.ray_triangle_intersect(
            self.orig, self.dest, cube_seg, smi.astype(np.uint32))

        fbs, _, _, _, _ = RTtools.ray_triangle_intersect(
            self.orig, self.dest, cube_seg, smi.astype(np.int64))

        npt.assert_allclose(fbs, fbs_ref, atol=1e-6)

    def test_submesh_index_int32(self):
        cube_seg, smi, _, _ = RTtools.triangle_mesh_segmentation(self.cube, 4)

        fbs_ref, _, _, _, _ = RTtools.ray_triangle_intersect(
            self.orig, self.dest, cube_seg, smi.astype(np.uint32))

        fbs, _, _, _, _ = RTtools.ray_triangle_intersect(
            self.orig, self.dest, cube_seg, smi.astype(np.int32))

        npt.assert_allclose(fbs, fbs_ref, atol=1e-6)

    # ------------------------------------------------------------------ #
    #  F-contiguous vs C-contiguous inputs produce identical results       #
    # ------------------------------------------------------------------ #
    def test_f_vs_c_contiguous(self):
        # Default numpy arrays are C-contiguous (row-major)
        orig_c = np.ascontiguousarray(self.orig)
        dest_c = np.ascontiguousarray(self.dest)
        mesh_c = np.ascontiguousarray(self.cube)

        orig_f = np.asfortranarray(self.orig)
        dest_f = np.asfortranarray(self.dest)
        mesh_f = np.asfortranarray(self.cube)

        # Verify memory layout
        self.assertTrue(orig_c.flags['C_CONTIGUOUS'])
        self.assertTrue(orig_f.flags['F_CONTIGUOUS'])

        fbs_c, sbs_c, no_hit_c, ifbs_c, isbs_c = RTtools.ray_triangle_intersect(
            orig_c, dest_c, mesh_c)
        fbs_f, sbs_f, no_hit_f, ifbs_f, isbs_f = RTtools.ray_triangle_intersect(
            orig_f, dest_f, mesh_f)

        npt.assert_allclose(fbs_c, fbs_f, atol=1e-7)
        npt.assert_allclose(sbs_c, sbs_f, atol=1e-7)
        npt.assert_array_equal(no_hit_c, no_hit_f)
        npt.assert_array_equal(ifbs_c, ifbs_f)
        npt.assert_array_equal(isbs_c, isbs_f)

    # ------------------------------------------------------------------ #
    #  Mixed contiguity: F-contiguous mesh, C-contiguous rays             #
    # ------------------------------------------------------------------ #
    def test_mixed_contiguity(self):
        orig_c = np.ascontiguousarray(self.orig)
        dest_c = np.ascontiguousarray(self.dest)
        mesh_f = np.asfortranarray(self.cube)

        fbs_ref, sbs_ref, no_hit_ref, ifbs_ref, isbs_ref = RTtools.ray_triangle_intersect(
            self.orig, self.dest, self.cube)

        fbs, sbs, no_hit, ifbs, isbs = RTtools.ray_triangle_intersect(
            orig_c, dest_c, mesh_f)

        npt.assert_allclose(fbs, fbs_ref, atol=1e-7)
        npt.assert_allclose(sbs, sbs_ref, atol=1e-7)
        npt.assert_array_equal(no_hit, no_hit_ref)
        npt.assert_array_equal(ifbs, ifbs_ref)
        npt.assert_array_equal(isbs, isbs_ref)

    # ------------------------------------------------------------------ #
    #  F-contiguous with segmented mesh + AABB                            #
    # ------------------------------------------------------------------ #
    def test_f_contiguous_with_submesh_and_aabb(self):
        cube_seg, smi, _, _ = RTtools.triangle_mesh_segmentation(self.cube, 4)
        smi = smi.astype(np.uint32)
        aabb = RTtools.triangle_mesh_aabb(cube_seg, smi)

        fbs_ref, sbs_ref, no_hit_ref, _, _ = RTtools.ray_triangle_intersect(
            self.orig, self.dest, cube_seg, smi, aabb)

        fbs, sbs, no_hit, _, _ = RTtools.ray_triangle_intersect(
            np.asfortranarray(self.orig), np.asfortranarray(self.dest),
            np.asfortranarray(cube_seg), smi, np.asfortranarray(aabb))

        npt.assert_allclose(fbs, fbs_ref, atol=1e-7)
        npt.assert_allclose(sbs, sbs_ref, atol=1e-7)
        npt.assert_array_equal(no_hit, no_hit_ref)

    # ------------------------------------------------------------------ #
    #  Single ray (1, 3) — minimal batch                                  #
    # ------------------------------------------------------------------ #
    def test_single_ray(self):
        fbs, sbs, no_hit, ifbs, isbs = RTtools.ray_triangle_intersect(
            self.orig[:1], self.dest[:1], self.cube)

        self.assertEqual(fbs.shape, (1, 3))
        npt.assert_allclose(fbs[0], [-1, 0, 0.5], atol=1e-6)
        npt.assert_allclose(sbs[0], [1, 0, 0.5], atol=1e-6)
        self.assertEqual(int(no_hit[0]), 2)
        self.assertEqual(int(ifbs[0]), 9)
        self.assertEqual(int(isbs[0]), 11)


if __name__ == "__main__":
    unittest.main()