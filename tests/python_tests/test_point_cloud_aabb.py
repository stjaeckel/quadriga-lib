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


class test_case(unittest.TestCase):

    def _aabb_numpy(self, pts: np.ndarray) -> np.ndarray:
        """Compute AABB for a set of points: [xmin, xmax, ymin, ymax, zmin, zmax]."""
        return np.array([
            np.min(pts[:, 0]), np.max(pts[:, 0]),
            np.min(pts[:, 1]), np.max(pts[:, 1]),
            np.min(pts[:, 2]), np.max(pts[:, 2]),
        ], dtype=pts.dtype)

    def test(self):
        # ---------- 1) Whole-cloud AABB (deterministic small set) ----------
        with self.subTest("single sub-cloud / whole cloud"):
            points = np.array([[1.0, 2.0, 3.0],
                               [4.0, -5.0, 6.0],
                               [-7.0, 8.0, -9.0]], dtype=np.float64)
            expected = self._aabb_numpy(points)[None, :]
            got = RTtools.point_cloud_aabb(points)
            npt.assert_allclose(got, expected, rtol=0, atol=0)

        # ---------- 2) Multiple sub-clouds via explicit sub_cloud_ind ----------
        with self.subTest("multiple sub-clouds explicit indices"):
            g1 = np.array([[0., 0., 0.], [2., 3., -1.], [1., -1., 5.]], dtype=np.float64)
            g2 = np.array([[-4., 2., 9.], [10., 10., -2.]], dtype=np.float64)
            points = np.vstack([g1, g2])
            sub_cloud_ind = np.array([0, len(g1)], dtype=np.uint32)
            expected = np.vstack([self._aabb_numpy(g1), self._aabb_numpy(g2)])
            got = RTtools.point_cloud_aabb(points, sub_cloud_ind)
            npt.assert_allclose(got, expected, rtol=0, atol=0)

        # ---------- 3) vec_size padding for output rows ----------
        with self.subTest("vec_size padding on output rows"):
            vec_size = 4
            got = RTtools.point_cloud_aabb(points, sub_cloud_ind, vec_size)
            # Expect rows padded to a multiple of vec_size (2 -> 4)
            self.assertEqual(got.shape[0], vec_size)
            # First two rows match expected; padded rows are zeros
            npt.assert_allclose(got[:2], expected, rtol=0, atol=0)
            npt.assert_allclose(got[2:], 0.0, rtol=0, atol=0)

        # ---------- 4) Invariance to within-subcloud shuffling ----------
        with self.subTest("order invariance within each sub-cloud"):
            # Shuffle points within each block defined by sub_cloud_ind
            p = points.copy()
            n1 = len(g1)
            perm1 = np.random.default_rng(0).permutation(n1)
            perm2 = np.random.default_rng(1).permutation(len(g2))
            p[:n1] = p[:n1][perm1]
            p[n1:] = p[n1:][perm2]
            got = RTtools.point_cloud_aabb(p, sub_cloud_ind)
            npt.assert_allclose(got[:2], expected, rtol=0, atol=0)

        # ---------- 5) Float32 inputs ----------
        with self.subTest("float32 input support"):
            pts32 = points.astype(np.float32)
            got = RTtools.point_cloud_aabb(pts32, sub_cloud_ind)
            npt.assert_allclose(got[:2], expected.astype(np.float32), rtol=0, atol=0)

        # ---------- 6) Integration with point_cloud_segmentation ----------
        with self.subTest("integration with segmentation (padding-aware)"):
            rng = np.random.default_rng(123)
            N = 12345
            pts = rng.standard_normal((N, 3)).astype(np.float64) * 10.0 + 5.0
            target_size = 1000
            vec_size = 8

            points_out, sub_ind, fwd_ind, _ = RTtools.point_cloud_segmentation(
                pts, target_size, vec_size
            )
            aabb_lib = RTtools.point_cloud_aabb(points_out, sub_ind, vec_size)

            # Manually compute expected per sub-cloud using only "real" elements (fwd_ind > 0)
            n_sub = len(sub_ind)
            expected_list = []
            for i in range(n_sub):
                start = int(sub_ind[i])
                end = int(sub_ind[i + 1]) if i + 1 < n_sub else points_out.shape[0]
                real_mask = (fwd_ind[start:end] > 0)
                self.assertGreater(np.count_nonzero(real_mask), 0, "Empty real-mask in sub-cloud")
                block_real = points_out[start:end][real_mask]
                expected_list.append(self._aabb_numpy(block_real))

            expected_stack = np.vstack(expected_list)

            # First n_sub rows must match expected; extra rows (padding to multiple of vec_size) must be zeros
            npt.assert_allclose(aabb_lib[:n_sub], expected_stack, rtol=1e-12, atol=1e-12)
            # Rows padded to multiple of vec_size
            pad_rows = aabb_lib.shape[0] - n_sub
            self.assertEqual(aabb_lib.shape[0] % vec_size, 0)
            if pad_rows > 0:
                npt.assert_allclose(aabb_lib[n_sub:], 0.0, rtol=0, atol=0)

        # ---------- 7) Single point edge-case ----------
        with self.subTest("single point edge-case"):
            p = np.array([[3.14, -2.0, 7.0]], dtype=np.float64)
            got = RTtools.point_cloud_aabb(p)
            expected = np.array([[3.14, 3.14, -2.0, -2.0, 7.0, 7.0]], dtype=np.float64)
            npt.assert_allclose(got, expected, rtol=0, atol=0)

        # ---------- 8) vec_size padding when sub_cloud_ind is omitted ----------
        with self.subTest("vec_size padding without sub_cloud_ind"):
            p = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float64)
            vec_size = 8
            got = RTtools.point_cloud_aabb(p, vec_size=vec_size)
            # one sub-cloud -> rows padded to vec_size
            self.assertEqual(got.shape[0], vec_size)
            expected = self._aabb_numpy(p)[None, :]
            npt.assert_allclose(got[0:1], expected, rtol=0, atol=0)
            npt.assert_allclose(got[1:], 0.0, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
