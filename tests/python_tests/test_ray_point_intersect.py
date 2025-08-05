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

import quadriga_lib


class test_case(unittest.TestCase):

    def _normalize_ray_ind(self, ray_ind, n_points):
        """
        Normalize ray_ind to a list[set[int]] of 0-based indices, regardless of backend format:
        - Pythonic format: list of variable-length 0-based arrays/lists
        - MATLAB-compat format: ndarray shape (n_points, max_no_hit) with 1-based indices and 0 padding
        """
        result = []
        if isinstance(ray_ind, np.ndarray):
            # Expect MATLAB-like fixed-width matrix: (n_points, max_no_hit)
            self.assertEqual(ray_ind.ndim, 2, "Unexpected ray_ind ndarray rank.")
            self.assertEqual(ray_ind.shape[0], n_points, "ray_ind row count must equal n_points.")
            for row in ray_ind:
                row = np.asarray(row)
                vals = row[row != 0] - 1  # convert 1-based to 0-based, drop zeros
                result.append(set(vals.astype(np.int64).tolist()))
        else:
            # Expect list/tuple of length n_points with variable-length arrays
            self.assertTrue(hasattr(ray_ind, '__len__'), "ray_ind must be list-like.")
            self.assertEqual(len(ray_ind), n_points, "ray_ind length must equal n_points.")
            for entry in ray_ind:
                arr = np.asarray(entry, dtype=np.int64)
                result.append(set(arr.tolist()))
        return result

    def test(self):
        # ---- Construct two simple beams extruding along +z with identical triangular apertures ----
        # Equilateral triangle centered at origin with side length L
        L = 0.12
        r = L / np.sqrt(3.0)  # distance from centroid to each vertex

        v1 = np.array([ r,       0.0,    0.0])
        v2 = np.array([-r/2.0,  +L/2.0,  0.0])
        v3 = np.array([-r/2.0,  -L/2.0,  0.0])

        # Ray 0 at origin, Ray 1 shifted in +x so triangles overlap but are not identical
        orig = np.array([
            [0.00, 0.00, 0.00],   # ray 0
            [0.06, 0.00, 0.00],   # ray 1
        ])

        trivec = np.vstack([
            np.hstack([v1, v2, v3]),
            np.hstack([v1, v2, v3]),
        ])  # shape (n_ray, 9)

        # Both beams point along +z; cross-section translates without changing shape
        d = np.array([0.0, 0.0, 1.0])
        tridir = np.vstack([
            np.hstack([d, d, d]),
            np.hstack([d, d, d]),
        ])  # shape (n_ray, 9)

        # ---- Points: one hit by both, one only by ray 1, one by none, one slightly inside near a vertex of ray 0 ----
        P_both      = np.array([0.035,        0.0,   1.0])   # inside both
        P_only_r1   = np.array([0.080,        0.0,   1.0])   # outside r0, inside r1
        P_none      = np.array([0.200,        0.0,   1.0])   # outside both

        # Avoid exact boundary: move a tiny epsilon inside the triangle near v3 at z=1
        eps = 1e-4
        P_inside_r0 = np.array([-r/2.0 + eps, -L/2.0 + eps, 1.0])

        points_base = np.vstack([P_both, P_only_r1, P_none, P_inside_r0])

        # ---- Add extra points clearly outside to reach a SIMD-aligned total of 16 points ----
        # AVX2 requires sub-cloud sizes to be multiples of 8. We'll make two sub-clouds of size 8.
        extra_outside = np.array([[0.5, 0.0, 0.5 + 0.1*i] for i in range(12)])
        points = np.vstack([points_base, extra_outside])
        n_points = points.shape[0]
        self.assertEqual(n_points, 16)

        # Valid optional-arg usages for this API:
        sub_cloud_ind_empty = np.array([], dtype=np.uint32)  # "no segmentation" sentinel
        sub_cloud_ind_seg = np.array([0, 8], dtype=np.uint32)  # two SIMD-aligned sub-clouds (8 each)

        # ---- Call under test ----
        # A) Omitted optional arguments
        hit_count_a, ray_ind_a = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points
        )

        # B) Explicit empty/zero form (None is NOT used)
        hit_count_b, ray_ind_b = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points,
            sub_cloud_ind=sub_cloud_ind_empty, target_size=0
        )

        # C) With SIMD-aligned segmentation
        hit_count_c, ray_ind_c = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points,
            sub_cloud_ind=sub_cloud_ind_seg, target_size=0
        )

        # ---- Basic type/shape checks ----
        for hc in (hit_count_a, hit_count_b, hit_count_c):
            self.assertIsInstance(hc, np.ndarray)
            self.assertEqual(hc.shape, (n_points,))
            self.assertTrue(np.issubdtype(hc.dtype, np.unsignedinteger) or
                            np.issubdtype(hc.dtype, np.integer))

        # ---- Normalize ray indices to 0-based sets per point and compare all calls ----
        sets_a = self._normalize_ray_ind(ray_ind_a, n_points)
        sets_b = self._normalize_ray_ind(ray_ind_b, n_points)
        sets_c = self._normalize_ray_ind(ray_ind_c, n_points)

        self.assertEqual(sets_a, sets_b,
                         "Omitting optional args vs. passing empty/0 should yield identical results.")
        self.assertEqual(sets_a, sets_c,
                         "Providing a valid SIMD-aligned segmentation should not change hit membership.")

        # ---- Expected hits per point (0-based ray indices) ----
        expected = [
            {0, 1},   # P_both
            {1},      # P_only_r1
            set(),    # P_none
            {0},      # P_inside_r0 (slightly inside near a vertex -> counts as hit for ray 0 only)
        ] + [set() for _ in range(12)]  # the 12 extra points are outside

        # ---- Value checks ----
        for i, (got, exp) in enumerate(zip(sets_a, expected)):
            self.assertSetEqual(got, exp, f"Point {i} expected hits {exp}, got {got}.")

        # hit_count should match the number of unique indices per point
        expected_counts = np.array([len(s) for s in expected], dtype=np.uint32)
        npt.assert_array_equal(hit_count_a.astype(np.uint32), expected_counts)
        npt.assert_array_equal(hit_count_b.astype(np.uint32), expected_counts)
        npt.assert_array_equal(hit_count_c.astype(np.uint32), expected_counts)

        # ---- Idempotence / dtype sanity ----
        for s in sets_a:
            for idx in s:
                self.assertIsInstance(idx, int)
                self.assertGreaterEqual(idx, 0)


if __name__ == "__main__":
    unittest.main()
