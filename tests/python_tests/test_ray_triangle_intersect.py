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


def build_unit_cube_triangles(dtype=np.float32):
    """
    Build an axis-aligned unit cube [0,1]^3 as 12 triangles.
    Face triangulation is chosen so that the ray passing through
    (y,z) = (0.3, 0.6) hits known triangles unambiguously.
    Triangle order defines the expected 1-based indices used below.
    """
    # Left face (x=0): split along diagonal (y=z)
    T0 = [0,0,0,  0,1,0,  0,1,1]  # (0,0,0)-(0,1,0)-(0,1,1)
    T1 = [0,0,0,  0,1,1,  0,0,1]  # (0,0,0)-(0,1,1)-(0,0,1)

    # Right face (x=1): split along same diagonal (y=z)
    T2 = [1,0,0,  1,1,1,  1,1,0]
    T3 = [1,0,0,  1,0,1,  1,1,1]

    # Front face (y=0)
    T4 = [0,0,0,  1,0,1,  1,0,0]
    T5 = [0,0,0,  0,0,1,  1,0,1]

    # Back face (y=1)
    T6 = [0,1,0,  1,1,0,  1,1,1]
    T7 = [0,1,0,  1,1,1,  0,1,1]

    # Bottom face (z=0)
    T8  = [0,0,0,  1,1,0,  1,0,0]
    T9  = [0,0,0,  0,1,0,  1,1,0]

    # Top face (z=1)
    T10 = [0,0,1,  1,0,1,  1,1,1]
    T11 = [0,0,1,  1,1,1,  0,1,1]

    tris = np.array(
        [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11],
        dtype=dtype
    )
    return tris


class test_version(unittest.TestCase):

    def test(self):
        atol = 1e-7

        # Build mesh (float32 on purpose to exercise internal conversion to double)
        mesh = build_unit_cube_triangles(dtype=np.float32)

        # A ray that crosses the closed mesh: expect two intersections (enter at x=0, exit at x=1)
        orig_A = np.array([-1.0, 0.3, 0.6], dtype=np.float32)
        dest_A = np.array([ 2.0, 0.3, 0.6], dtype=np.float32)
        exp_fbs_A = np.array([0.0, 0.3, 0.6])
        exp_sbs_A = np.array([1.0, 0.3, 0.6])
        # From the chosen triangulation, these points fall into triangles:
        # left face -> T1 (index 1 zero-based => 2 one-based), right face -> T3 (index 3 => 4 one-based)
        exp_fbs_ind_A = 2
        exp_sbs_ind_A = 4

        # A ray starting inside heading out: expect one intersection (exit at x=1), SBS == dest
        orig_B = np.array([0.5, 0.3, 0.6], dtype=np.float32)
        dest_B = np.array([2.0, 0.3, 0.6], dtype=np.float32)
        exp_fbs_B = np.array([1.0, 0.3, 0.6])
        # sbs equals dest when there's only one interaction
        exp_sbs_B = dest_B.astype(np.float64)  # compare in double

        # A ray that misses entirely: FBS == dest, SBS == dest, no_interact == 0
        orig_C = np.array([-1.0, 2.0, 2.0], dtype=np.float32)
        dest_C = np.array([ 2.0, 2.0, 2.0], dtype=np.float32)

        # ---------- Case 1: two hits (no sub-mesh index) ----------
        with self.subTest("two_hits_without_submesh"):
            fbs, sbs, no_interact, fbs_ind, sbs_ind = RTtools.ray_triangle_intersect(
                orig_A[None, :], dest_A[None, :], mesh
            )
            self.assertEqual(fbs.shape, (1, 3))
            self.assertEqual(sbs.shape, (1, 3))
            self.assertEqual(no_interact.shape, (1,))
            self.assertEqual(fbs_ind.shape, (1,))
            self.assertEqual(sbs_ind.shape, (1,))

            npt.assert_allclose(fbs[0], exp_fbs_A, atol=atol)
            npt.assert_allclose(sbs[0], exp_sbs_A, atol=atol)
            self.assertEqual(int(no_interact[0]), 2)
            # 1-based indices expected
            self.assertEqual(int(fbs_ind[0]), exp_fbs_ind_A)
            self.assertEqual(int(sbs_ind[0]), exp_sbs_ind_A)
            self.assertGreaterEqual(int(fbs_ind[0]), 1)
            self.assertGreaterEqual(int(sbs_ind[0]), 1)

        # ---------- Case 2: two hits with sub-mesh indices ----------
        # Use a single sub-mesh starting at 0 (aligned to vec_size=8).
        sub_mesh_index = np.array([0], dtype=np.int64)

        with self.subTest("two_hits_with_submesh"):
            fbs, sbs, no_interact, fbs_ind, sbs_ind = RTtools.ray_triangle_intersect(
                orig_A[None, :], dest_A[None, :], mesh, sub_mesh_index
            )
            npt.assert_allclose(fbs[0], exp_fbs_A, atol=atol)
            npt.assert_allclose(sbs[0], exp_sbs_A, atol=atol)
            self.assertEqual(int(no_interact[0]), 2)
            self.assertEqual(int(fbs_ind[0]), exp_fbs_ind_A)  # 2
            self.assertEqual(int(sbs_ind[0]), exp_sbs_ind_A)  # 4

        # ---------- Case 3: one hit (start inside) ----------
        with self.subTest("one_hit_from_inside"):
            fbs, sbs, no_interact, fbs_ind, sbs_ind = RTtools.ray_triangle_intersect(
                orig_B[None, :], dest_B[None, :], mesh
            )
            npt.assert_allclose(fbs[0], exp_fbs_B, atol=atol)
            npt.assert_allclose(sbs[0], exp_sbs_B, atol=atol)
            self.assertEqual(int(no_interact[0]), 1)
            # First hit is on the x=1 face; by our triangulation that's T3 -> 4 (1-based)
            self.assertEqual(int(fbs_ind[0]), 4)
            # For the second-hit index when there's only one interaction the API doesn't
            # document a specific value; just check it's a valid uint32.
            self.assertGreaterEqual(int(sbs_ind[0]), 0)

        # ---------- Case 4: no hit ----------
        with self.subTest("no_hit"):
            fbs, sbs, no_interact, fbs_ind, sbs_ind = RTtools.ray_triangle_intersect(
                orig_C[None, :], dest_C[None, :], mesh
            )
            npt.assert_allclose(fbs[0], dest_C.astype(np.float64), atol=atol)
            npt.assert_allclose(sbs[0], dest_C.astype(np.float64), atol=atol)
            self.assertEqual(int(no_interact[0]), 0)

        # ---------- Case 5: batched rays ----------
        with self.subTest("batched_rays"):
            orig = np.vstack([orig_A, orig_B, orig_C]).astype(np.float32)
            dest = np.vstack([dest_A, dest_B, dest_C]).astype(np.float32)
            fbs, sbs, no_interact, fbs_ind, sbs_ind = RTtools.ray_triangle_intersect(orig, dest, mesh)

            # Row 0: two hits
            npt.assert_allclose(fbs[0], exp_fbs_A, atol=atol)
            npt.assert_allclose(sbs[0], exp_sbs_A, atol=atol)
            self.assertEqual(int(no_interact[0]), 2)
            self.assertEqual(int(fbs_ind[0]), exp_fbs_ind_A)
            self.assertEqual(int(sbs_ind[0]), exp_sbs_ind_A)

            # Row 1: one hit
            npt.assert_allclose(fbs[1], exp_fbs_B, atol=atol)
            npt.assert_allclose(sbs[1], exp_sbs_B, atol=atol)
            self.assertEqual(int(no_interact[1]), 1)
            self.assertEqual(int(fbs_ind[1]), 4)

            # Row 2: no hit
            npt.assert_allclose(fbs[2], dest_C.astype(np.float64), atol=atol)
            npt.assert_allclose(sbs[2], dest_C.astype(np.float64), atol=atol)
            self.assertEqual(int(no_interact[2]), 0)


if __name__ == "__main__":
    unittest.main()
