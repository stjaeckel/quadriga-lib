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

import quadriga_lib


class test_case(unittest.TestCase):
    """Tests for quadriga_lib.RTtools.ray_point_intersect

    API: ray_point_intersect(orig, trivec, tridir, points,
                             sub_cloud_ind=<empty>, use_kernel=0, gpu_id=0)

    Outputs:
      hit_count : uint32 ndarray of shape (n_points,)
      ray_ind   : list of length n_points; each entry is a 1-D uint32 array
                  of 0-based ray indices that hit that point.
    """

    # ---------------- helpers ----------------

    def _ray_ind_to_sets(self, ray_ind, n_points):
        """Convert ray_ind (list of 1-D arrays) to list[set[int]] for set-equality checks."""
        self.assertTrue(hasattr(ray_ind, "__len__"), "ray_ind must be list-like.")
        self.assertEqual(len(ray_ind), n_points, "ray_ind length must equal n_points.")
        out = []
        for entry in ray_ind:
            arr = np.asarray(entry, dtype=np.int64)
            self.assertEqual(arr.ndim, 1, "each ray_ind entry must be 1-D.")
            out.append(set(arr.tolist()))
        return out

    @staticmethod
    def _two_beam_setup():
        """Two beams along +z with overlapping equilateral-triangle cross-sections."""
        L = 0.12
        r = L / np.sqrt(3.0)  # centroid-to-vertex distance
        v1 = np.array([r, 0.0, 0.0])
        v2 = np.array([-r / 2.0, +L / 2.0, 0.0])
        v3 = np.array([-r / 2.0, -L / 2.0, 0.0])
        orig = np.array(
            [
                [0.00, 0.00, 0.00],  # ray 0
                [0.06, 0.00, 0.00],  # ray 1 (shifted in +x; triangles overlap)
            ]
        )
        trivec = np.vstack(
            [
                np.hstack([v1, v2, v3]),
                np.hstack([v1, v2, v3]),
            ]
        )
        d = np.array([0.0, 0.0, 1.0])
        tridir = np.vstack(
            [
                np.hstack([d, d, d]),
                np.hstack([d, d, d]),
            ]
        )
        return orig, trivec, tridir, L, r

    @staticmethod
    def _icosphere_setup(no_div=20):
        """Full icosphere of beams shifted by (-10,-20,-30); 16 test points around origin.

        Mirrors the MATLAB test: each point is hit by exactly one beam.
        """
        orig, _, trivec, tridir = quadriga_lib.RTtools.icosphere(no_div, 1.0, 1)
        orig = orig - np.array([10.0, 20.0, 30.0])

        pts = np.zeros((4, 3))
        pts[:, 0] = np.array([-0.1, 0.0, 0.1, 0.2])
        pts = np.tile(pts, (2, 1))  # 8 rows
        pts[4:8, 0] += 40.0
        pts = np.tile(pts, (2, 1))  # 16 rows
        pts[0:8, 1] -= 5.0
        pts[8:16, 1] += 5.0
        pts[:, 2] += 1.0
        return orig, trivec, tridir, pts

    # ---------------- tests ----------------

    def test_two_beams_specific_geometry(self):
        """Two overlapping beams + 16 points; verify per-point hit membership."""
        orig, trivec, tridir, L, r = self._two_beam_setup()
        eps = 1e-4

        P_both = np.array([0.035, 0.0, 1.0])  # inside both
        P_only_r1 = np.array([0.080, 0.0, 1.0])  # outside r0, inside r1
        P_none = np.array([0.200, 0.0, 1.0])  # outside both
        P_inside_r0 = np.array([-r / 2.0 + eps, -L / 2.0 + eps, 1.0])  # near v3 of r0
        base = np.vstack([P_both, P_only_r1, P_none, P_inside_r0])

        # Pad to 16 so two halves of 8 are SIMD-aligned for AVX2.
        extra = np.array([[0.5, 0.0, 0.5 + 0.1 * i] for i in range(12)])
        points = np.vstack([base, extra])
        n_points = points.shape[0]
        self.assertEqual(n_points, 16)

        hit_count, ray_ind = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points
        )
        sets = self._ray_ind_to_sets(ray_ind, n_points)
        expected = [{0, 1}, {1}, set(), {0}] + [set()] * 12

        for i, (got, exp) in enumerate(zip(sets, expected)):
            self.assertSetEqual(got, exp, f"Point {i}: expected {exp}, got {got}")

        npt.assert_array_equal(
            hit_count.astype(np.uint32),
            np.array([len(s) for s in expected], dtype=np.uint32),
        )

    def test_icosphere_each_point_hit_once(self):
        """A full icosphere of beams must hit every test point exactly once."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_count, ray_ind = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points
        )
        npt.assert_array_equal(
            hit_count.astype(np.uint32),
            np.ones(points.shape[0], dtype=np.uint32),
        )
        sets = self._ray_ind_to_sets(ray_ind, points.shape[0])
        for i, s in enumerate(sets):
            self.assertEqual(
                len(s), 1, f"Point {i}: expected exactly 1 hit, got {len(s)}"
            )
            (idx,) = tuple(s)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, orig.shape[0])

    def test_output_types_and_shapes(self):
        """Verify dtype, shape, and structure of both outputs."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_count, ray_ind = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points
        )
        n_points = points.shape[0]

        # hit_count
        self.assertIsInstance(hit_count, np.ndarray)
        self.assertEqual(hit_count.shape, (n_points,))
        self.assertEqual(hit_count.dtype, np.uint32)

        # ray_ind
        self.assertEqual(len(ray_ind), n_points)
        for i, entry in enumerate(ray_ind):
            arr = np.asarray(entry)
            self.assertEqual(arr.ndim, 1, f"ray_ind[{i}] must be 1-D")
            self.assertTrue(
                np.issubdtype(arr.dtype, np.integer),
                f"ray_ind[{i}] dtype must be integer, got {arr.dtype}",
            )
            for v in arr:
                self.assertGreaterEqual(int(v), 0)
                self.assertLess(int(v), orig.shape[0])
            # hit_count[i] equals number of indices for point i
            self.assertEqual(
                int(hit_count[i]),
                arr.size,
                f"hit_count[{i}] disagrees with len(ray_ind[{i}])",
            )

    def test_sub_cloud_ind_omitted_vs_empty_vs_manual(self):
        """Omitted, empty, and explicit segmentation must yield identical results."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)

        hit_a, ind_a = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points
        )
        empty_idx = np.array([], dtype=np.uint32)
        hit_b, ind_b = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, sub_cloud_ind=empty_idx
        )
        # Manual SIMD-aligned segmentation: split into halves of 8.
        manual_idx = np.array([0, 8], dtype=np.uint32)
        hit_c, ind_c = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, sub_cloud_ind=manual_idx
        )

        npt.assert_array_equal(hit_a, hit_b)
        npt.assert_array_equal(hit_a, hit_c)
        n = points.shape[0]
        self.assertEqual(
            self._ray_ind_to_sets(ind_a, n), self._ray_ind_to_sets(ind_b, n)
        )
        self.assertEqual(
            self._ray_ind_to_sets(ind_a, n), self._ray_ind_to_sets(ind_c, n)
        )

    def test_integration_with_point_cloud_segmentation(self):
        """Permuted point cloud + auto-generated sub_cloud_index must match unsegmented after un-permuting."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_ref, _ = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points
        )
        pointsR, sub_cloud_index, _, reverse_index = (
            quadriga_lib.RTtools.point_cloud_segmentation(points, 4, 8)
        )
        hit_R, _ = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, pointsR, sub_cloud_ind=sub_cloud_index
        )
        npt.assert_array_equal(hit_ref, hit_R[reverse_index])

    def test_kernel_auto_vs_generic(self):
        """auto and GENERIC kernel must produce identical results."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_auto, ind_auto = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, use_kernel=0
        )
        hit_gen, ind_gen = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, use_kernel=1
        )
        npt.assert_array_equal(hit_auto, hit_gen)
        n = points.shape[0]
        self.assertEqual(
            self._ray_ind_to_sets(ind_auto, n), self._ray_ind_to_sets(ind_gen, n)
        )

    def test_kernel_avx2(self):
        """AVX2 kernel must agree with GENERIC if available; skip otherwise."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_gen, ind_gen = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, use_kernel=1
        )
        try:
            hit_avx, ind_avx = quadriga_lib.RTtools.ray_point_intersect(
                orig, trivec, tridir, points, use_kernel=2
            )
        except ValueError as e:
            self.skipTest(f"AVX2 kernel not available: {e}")
            return
        npt.assert_array_equal(hit_gen, hit_avx)
        n = points.shape[0]
        self.assertEqual(
            self._ray_ind_to_sets(ind_gen, n), self._ray_ind_to_sets(ind_avx, n)
        )

    def test_kernel_cuda(self):
        """CUDA kernel must agree with GENERIC if available; skip otherwise."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_gen, ind_gen = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, use_kernel=1
        )
        try:
            hit_cuda, ind_cuda = quadriga_lib.RTtools.ray_point_intersect(
                orig, trivec, tridir, points, use_kernel=3, gpu_id=0
            )
        except ValueError as e:
            self.skipTest(f"CUDA kernel not available: {e}")
            return
        npt.assert_array_equal(hit_gen, hit_cuda)
        n = points.shape[0]
        self.assertEqual(
            self._ray_ind_to_sets(ind_gen, n), self._ray_ind_to_sets(ind_cuda, n)
        )

    def test_input_validation(self):
        """Invalid inputs must raise."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)

        cases = [
            (
                "orig row count mismatch with trivec",
                lambda: quadriga_lib.RTtools.ray_point_intersect(
                    orig[:2, :], trivec, tridir, points
                ),
            ),
            (
                "orig must have 3 columns",
                lambda: quadriga_lib.RTtools.ray_point_intersect(
                    orig[:, :2], trivec, tridir, points
                ),
            ),
            (
                "trivec must have 9 columns",
                lambda: quadriga_lib.RTtools.ray_point_intersect(
                    orig, trivec[:, :2], tridir, points
                ),
            ),
            (
                "tridir row count mismatch with orig",
                lambda: quadriga_lib.RTtools.ray_point_intersect(
                    orig, trivec, tridir[:2, :], points
                ),
            ),
            (
                "tridir must have 9 columns",
                lambda: quadriga_lib.RTtools.ray_point_intersect(
                    orig, trivec, tridir[:, :2], points
                ),
            ),
            (
                "points must have 3 columns",
                lambda: quadriga_lib.RTtools.ray_point_intersect(
                    orig, trivec, tridir, points[:, :2]
                ),
            ),
            (
                "sub_cloud_ind must start at 0",
                lambda: quadriga_lib.RTtools.ray_point_intersect(
                    orig,
                    trivec,
                    tridir,
                    points,
                    sub_cloud_ind=np.array([2, 8], dtype=np.uint32),
                ),
            ),
            (
                "sub_cloud_ind exceeds number of points",
                lambda: quadriga_lib.RTtools.ray_point_intersect(
                    orig,
                    trivec,
                    tridir,
                    points,
                    sub_cloud_ind=np.array([0, 33], dtype=np.uint32),
                ),
            ),
        ]
        for name, fn in cases:
            with self.subTest(case=name):
                with self.assertRaises(Exception):
                    fn()


if __name__ == "__main__":
    unittest.main()
