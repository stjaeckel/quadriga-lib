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

import quadriga_lib


class test_case(unittest.TestCase):
    """Tests for quadriga_lib.RTtools.ray_point_intersect

    API: ray_point_intersect(orig, trivec, tridir, points,
                             sub_cloud_ind=<empty>, use_kernel=0, gpu_id=0)

    Outputs (compressed sparse row layout):
      hit_count  : uint32 ndarray of shape (n_points,)
      hit_index  : 1-D ndarray of 0-based ray indices, grouped by point, shape (n_hit,)
      hit_offset : uint32 ndarray of shape (n_points + 1,); the rays hitting point i
                   are hit_index[hit_offset[i]:hit_offset[i+1]]
    """

    # ---------------- helpers ----------------

    def _csr_to_sets(self, hit_index, hit_offset, n_points):
        """Convert the CSR outputs to list[set[int]] for set-equality checks."""
        hit_index = np.asarray(hit_index)
        hit_offset = np.asarray(hit_offset)

        self.assertEqual(hit_index.ndim, 1, "hit_index must be 1-D.")
        self.assertEqual(hit_offset.ndim, 1, "hit_offset must be 1-D.")
        self.assertEqual(
            hit_offset.shape[0], n_points + 1, "hit_offset length must equal n_points + 1."
        )
        self.assertEqual(
            int(hit_offset[-1]), hit_index.shape[0], "hit_offset[-1] must equal len(hit_index)."
        )

        out = []
        for i in range(n_points):
            block = hit_index[hit_offset[i] : hit_offset[i + 1]]
            as_set = set(int(v) for v in block)
            self.assertEqual(
                len(as_set), block.size, f"point {i}: duplicate ray index in its block"
            )
            out.append(as_set)
        return out

    @staticmethod
    def _csr_to_list(hit_index, hit_offset):
        """Per-point list of arrays, the form the wrapper returned before the CSR change."""
        return np.split(np.asarray(hit_index), np.asarray(hit_offset)[1:-1])

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

        hit_count, hit_index, hit_offset = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points
        )
        sets = self._csr_to_sets(hit_index, hit_offset, n_points)
        expected = [{0, 1}, {1}, set(), {0}] + [set()] * 12

        for i, (got, exp) in enumerate(zip(sets, expected)):
            self.assertSetEqual(got, exp, f"Point {i}: expected {exp}, got {got}")

        npt.assert_array_equal(
            hit_count.astype(np.uint32),
            np.array([len(s) for s in expected], dtype=np.uint32),
        )

        # The block boundaries are independent of the order within a block
        npt.assert_array_equal(
            np.asarray(hit_offset).astype(np.int64),
            np.concatenate(([0], np.cumsum([len(s) for s in expected]))).astype(np.int64),
        )

    def test_icosphere_each_point_hit_once(self):
        """A full icosphere of beams must hit every test point exactly once."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_count, hit_index, hit_offset = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points
        )
        npt.assert_array_equal(
            hit_count.astype(np.uint32),
            np.ones(points.shape[0], dtype=np.uint32),
        )
        sets = self._csr_to_sets(hit_index, hit_offset, points.shape[0])
        for i, s in enumerate(sets):
            self.assertEqual(
                len(s), 1, f"Point {i}: expected exactly 1 hit, got {len(s)}"
            )
            (idx,) = tuple(s)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, orig.shape[0])

    def test_output_types_and_shapes(self):
        """Verify dtype, shape, and structure of all three outputs."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_count, hit_index, hit_offset = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points
        )
        n_points = points.shape[0]

        # hit_count
        self.assertIsInstance(hit_count, np.ndarray)
        self.assertEqual(hit_count.shape, (n_points,))
        self.assertEqual(hit_count.dtype, np.uint32)

        # hit_offset
        self.assertIsInstance(hit_offset, np.ndarray)
        self.assertEqual(hit_offset.shape, (n_points + 1,))
        self.assertEqual(hit_offset.dtype, np.uint32)

        # hit_index
        self.assertIsInstance(hit_index, np.ndarray)
        self.assertEqual(hit_index.ndim, 1)
        self.assertTrue(
            np.issubdtype(hit_index.dtype, np.integer),
            f"hit_index dtype must be integer, got {hit_index.dtype}",
        )
        self.assertEqual(hit_index.shape, (int(hit_offset[-1]),))

        for i in range(n_points):
            block = hit_index[hit_offset[i] : hit_offset[i + 1]]
            for v in block:
                self.assertGreaterEqual(int(v), 0)
                self.assertLess(int(v), orig.shape[0])
            self.assertEqual(
                int(hit_count[i]),
                block.size,
                f"hit_count[{i}] disagrees with the size of its hit_index block",
            )

    def test_csr_structure_invariants(self):
        """hit_offset must start at 0, be monotone, and end at len(hit_index)."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_count, hit_index, hit_offset = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points
        )
        offs = np.asarray(hit_offset).astype(np.int64)

        self.assertEqual(int(offs[0]), 0, "hit_offset[0] must be 0")
        self.assertTrue(np.all(np.diff(offs) >= 0), "hit_offset must be non-decreasing")
        self.assertEqual(
            int(offs[-1]), np.asarray(hit_index).size, "hit_offset[-1] must equal len(hit_index)"
        )
        npt.assert_array_equal(
            np.diff(offs), np.asarray(hit_count).astype(np.int64),
            err_msg="hit_count must equal numpy.diff(hit_offset)",
        )

    def test_split_form_matches_csr(self):
        """numpy.split reproduces the per-point list form documented for migration."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_count, hit_index, hit_offset = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points
        )
        n_points = points.shape[0]

        parts = self._csr_to_list(hit_index, hit_offset)
        self.assertEqual(len(parts), n_points)
        for i, part in enumerate(parts):
            self.assertEqual(part.ndim, 1, f"part {i} must be 1-D")
            self.assertEqual(part.size, int(hit_count[i]))
            npt.assert_array_equal(
                part, np.asarray(hit_index)[hit_offset[i] : hit_offset[i + 1]]
            )

    def test_no_hits_returns_empty_csr(self):
        """Points outside every beam give an empty flat list and all-zero offsets."""
        orig, trivec, tridir, L, r = self._two_beam_setup()
        # 5 points, deliberately not a multiple of the AVX2 vector size, placed
        # well outside both beams
        points = np.array([[10.0, 10.0, 5.0 + 0.1 * i] for i in range(5)])
        n_points = points.shape[0]

        for kernel in (0, 1):
            with self.subTest(use_kernel=kernel):
                hit_count, hit_index, hit_offset = quadriga_lib.RTtools.ray_point_intersect(
                    orig, trivec, tridir, points, use_kernel=kernel
                )
                self.assertEqual(np.asarray(hit_index).size, 0)
                npt.assert_array_equal(
                    np.asarray(hit_count).astype(np.int64), np.zeros(n_points, dtype=np.int64)
                )
                npt.assert_array_equal(
                    np.asarray(hit_offset).astype(np.int64),
                    np.zeros(n_points + 1, dtype=np.int64),
                )
                parts = self._csr_to_list(hit_index, hit_offset)
                self.assertEqual(len(parts), n_points)
                self.assertTrue(all(p.size == 0 for p in parts))

    def test_padding_is_not_leaked(self):
        """A point count that is not a multiple of the SIMD width must give the
        same result on every kernel; the internal padding sits at the origin,
        which lies inside beam 0, so a leak would show up as extra hits."""
        orig, trivec, tridir, L, r = self._two_beam_setup()
        points = np.array([[0.0, 0.0, 1.0 + 0.1 * i] for i in range(5)])
        n_points = points.shape[0]
        self.assertNotEqual(n_points % 8, 0, "test needs an unaligned point count")

        hc_gen, hi_gen, ho_gen = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, use_kernel=1
        )
        self.assertGreater(
            int(np.asarray(hc_gen).sum()), 0, "test would be vacuous with no hits"
        )
        self.assertEqual(np.asarray(ho_gen).shape, (n_points + 1,))

        try:
            hc_avx, hi_avx, ho_avx = quadriga_lib.RTtools.ray_point_intersect(
                orig, trivec, tridir, points, use_kernel=2
            )
        except Exception as e:
            self.skipTest(f"AVX2 kernel not available: {e}")
            return

        self.assertEqual(np.asarray(ho_avx).shape, (n_points + 1,))
        npt.assert_array_equal(np.asarray(hc_avx), np.asarray(hc_gen))
        npt.assert_array_equal(np.asarray(ho_avx), np.asarray(ho_gen))
        self.assertEqual(
            self._csr_to_sets(hi_avx, ho_avx, n_points),
            self._csr_to_sets(hi_gen, ho_gen, n_points),
        )

    def test_ray_indices_ascending_generic(self):
        """The GENERIC kernel emits each point's ray list in ascending order."""
        orig, trivec, tridir, L, r = self._two_beam_setup()
        # x = 0.035 lies inside both beams, so every block holds two indices
        points = np.array([[0.035, 0.0, 1.0 + 0.1 * i] for i in range(16)])
        n_points = points.shape[0]

        hit_count, hit_index, hit_offset = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, use_kernel=1
        )
        self.assertGreater(
            int(np.asarray(hit_count).max()), 1, "test needs a point with several hits"
        )

        idx = np.asarray(hit_index)
        for i in range(n_points):
            block = idx[hit_offset[i] : hit_offset[i + 1]]
            if block.size > 1:
                self.assertTrue(
                    np.all(np.diff(block.astype(np.int64)) > 0),
                    f"point {i}: ray indices not strictly ascending: {block}",
                )

    def test_sub_cloud_ind_omitted_vs_empty_vs_manual(self):
        """Omitted, empty, and explicit segmentation must yield identical results."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)

        hit_a, idx_a, off_a = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points
        )
        empty_idx = np.array([], dtype=np.uint32)
        hit_b, idx_b, off_b = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, sub_cloud_ind=empty_idx
        )
        # Manual SIMD-aligned segmentation: split into halves of 8.
        manual_idx = np.array([0, 8], dtype=np.uint32)
        hit_c, idx_c, off_c = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, sub_cloud_ind=manual_idx
        )

        npt.assert_array_equal(hit_a, hit_b)
        npt.assert_array_equal(hit_a, hit_c)
        npt.assert_array_equal(off_a, off_b)
        npt.assert_array_equal(off_a, off_c)
        n = points.shape[0]
        self.assertEqual(
            self._csr_to_sets(idx_a, off_a, n), self._csr_to_sets(idx_b, off_b, n)
        )
        self.assertEqual(
            self._csr_to_sets(idx_a, off_a, n), self._csr_to_sets(idx_c, off_c, n)
        )

    def test_integration_with_point_cloud_segmentation(self):
        """Permuted point cloud + auto-generated sub_cloud_index must match unsegmented after un-permuting."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_ref, _, _ = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points
        )
        pointsR, sub_cloud_index, _, reverse_index = (
            quadriga_lib.RTtools.point_cloud_segmentation(points, 4, 8)
        )
        hit_R, _, _ = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, pointsR, sub_cloud_ind=sub_cloud_index
        )
        npt.assert_array_equal(hit_ref, hit_R[reverse_index])

    def test_kernel_auto_vs_generic(self):
        """auto and GENERIC kernel must produce identical results."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_auto, idx_auto, off_auto = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, use_kernel=0
        )
        hit_gen, idx_gen, off_gen = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, use_kernel=1
        )
        npt.assert_array_equal(hit_auto, hit_gen)
        npt.assert_array_equal(off_auto, off_gen)
        n = points.shape[0]
        self.assertEqual(
            self._csr_to_sets(idx_auto, off_auto, n), self._csr_to_sets(idx_gen, off_gen, n)
        )

    def test_kernel_avx2(self):
        """AVX2 kernel must agree with GENERIC if available; skip otherwise."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_gen, idx_gen, off_gen = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, use_kernel=1
        )
        try:
            hit_avx, idx_avx, off_avx = quadriga_lib.RTtools.ray_point_intersect(
                orig, trivec, tridir, points, use_kernel=2
            )
        except ValueError as e:
            self.skipTest(f"AVX2 kernel not available: {e}")
            return
        npt.assert_array_equal(hit_gen, hit_avx)
        npt.assert_array_equal(off_gen, off_avx)
        n = points.shape[0]
        self.assertEqual(
            self._csr_to_sets(idx_gen, off_gen, n), self._csr_to_sets(idx_avx, off_avx, n)
        )

    def test_kernel_cuda(self):
        """CUDA kernel must agree with GENERIC if available; skip otherwise."""
        orig, trivec, tridir, points = self._icosphere_setup(no_div=20)
        hit_gen, idx_gen, off_gen = quadriga_lib.RTtools.ray_point_intersect(
            orig, trivec, tridir, points, use_kernel=1
        )
        try:
            hit_cuda, idx_cuda, off_cuda = quadriga_lib.RTtools.ray_point_intersect(
                orig, trivec, tridir, points, use_kernel=3, gpu_id=0
            )
        except ValueError as e:
            self.skipTest(f"CUDA kernel not available: {e}")
            return
        npt.assert_array_equal(hit_gen, hit_cuda)
        npt.assert_array_equal(off_gen, off_cuda)
        n = points.shape[0]
        # CUDA groups a point's ray list by batch, so compare as sets
        self.assertEqual(
            self._csr_to_sets(idx_gen, off_gen, n), self._csr_to_sets(idx_cuda, off_cuda, n)
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