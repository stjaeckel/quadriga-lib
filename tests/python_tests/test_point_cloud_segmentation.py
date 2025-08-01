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


class test_version(unittest.TestCase):
    """
    Contract tests for RTtools.point_cloud_segmentation

    Verifies:
      - Output tuple length and array shapes match the spec
      - points_out has shape [n_points_out, 3] and float dtype
      - sub_cloud_ind is 0-based, strictly increasing, within bounds
      - For vec_size>1 each sub-cloud length is a multiple of vec_size
      - forward_ind is 1-based with zeros indicating padding; positive entries cover 1..n_points exactly once
      - reverse_ind is 0-based, unique, and inverses forward_ind (for non-padding entries)
      - Padding rows (if any) are placed at the AABB center of their sub-cloud
      - Determinism across repeated calls
    """

    def _run_case_and_assert(self, n_points, target_size, vec_size, seed=7):
        rng = np.random.default_rng(seed)
        points_in = rng.normal(size=(n_points, 3))

        # Run once
        data = RTtools.point_cloud_segmentation(points_in, target_size, vec_size)
        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 4)

        points_out, sub_cloud_ind, forward_ind, reverse_ind = data

        # --- Basic shape & dtype checks ---
        self.assertEqual(points_out.ndim, 2)
        self.assertEqual(points_out.shape[1], 3, msg="points_out must have 3 columns (x,y,z)")
        n_points_out = points_out.shape[0]
        self.assertGreater(n_points_out, 0)
        self.assertIn(points_out.dtype.kind, ('f',), msg="points_out must be floating point (float32/float64)")

        # sub_cloud_ind / indices types
        self.assertGreaterEqual(sub_cloud_ind.ndim, 1)
        self.assertGreaterEqual(sub_cloud_ind.size, 1)
        self.assertIn(sub_cloud_ind.dtype.kind, ('u', 'i'), msg="sub_cloud_ind must be integer")
        self.assertEqual(forward_ind.ndim, 1)
        self.assertEqual(forward_ind.size, n_points_out)
        self.assertIn(forward_ind.dtype.kind, ('u', 'i'), msg="forward_ind must be integer")
        self.assertEqual(reverse_ind.ndim, 1)
        self.assertEqual(reverse_ind.size, n_points)
        self.assertIn(reverse_ind.dtype.kind, ('u', 'i'), msg="reverse_ind must be integer")

        # --- sub_cloud_ind properties (0-based, increasing, within range) ---
        sci = np.asarray(sub_cloud_ind, dtype=np.int64)
        self.assertEqual(sci[0], 0, msg="First sub-cloud must start at index 0 (0-based)")
        self.assertTrue(np.all(np.diff(sci) > 0), msg="sub_cloud_ind must be strictly increasing")
        self.assertTrue(0 <= sci.min() <= sci.max() < n_points_out, msg="sub_cloud_ind out of bounds")

        # Sub-cloud sizes
        boundaries = np.r_[sci, n_points_out]
        sizes = np.diff(boundaries)
        self.assertTrue(np.all(sizes > 0), msg="Each sub-cloud must have positive length")
        if vec_size > 1:
            self.assertTrue(np.all(sizes % vec_size == 0),
                            msg="For vec_size>1, each sub-cloud size must be a multiple of vec_size")

        # --- forward_ind properties (1-based, zeros for padding) ---
        fwd = np.asarray(forward_ind, dtype=np.int64)
        self.assertTrue(np.all((fwd >= 0) & (fwd <= n_points)),
                        msg="forward_ind entries must be 0 or in [1..n_points]")
        positive_fwd = fwd[fwd > 0]
        counts = np.bincount(positive_fwd, minlength=n_points + 1)
        self.assertTrue(np.all(counts[1:] == 1),
                        msg="Each input index must appear exactly once in forward_ind")
        n_padding = int((fwd == 0).sum())
        self.assertEqual(n_padding, n_points_out - n_points,
                         msg="Padding count must equal n_points_out - n_points")

        # --- reverse_ind properties (0-based, unique, inverse of forward_ind) ---
        rev = np.asarray(reverse_ind, dtype=np.int64)
        self.assertTrue(np.all((rev >= 0) & (rev < n_points_out)),
                        msg="reverse_ind entries must be in [0..n_points_out-1]")
        self.assertEqual(np.unique(rev).size, rev.size, msg="reverse_ind entries must be unique")
        # Inverse mapping: forward_ind[reverse_ind[i]] == i+1
        i_indices = np.arange(1, n_points + 1, dtype=np.int64)
        mapped = fwd[rev]
        npt.assert_array_equal(mapped, i_indices,
                               err_msg="reverse_ind must invert forward_ind for non-padding entries")

        # --- Padding-point sanity: padding rows sit at AABB center of their sub-cloud ---
        if n_padding > 0:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                sl = slice(start, end)
                fwd_sl = fwd[sl]
                pts_sl = points_out[sl]
                pad_mask = (fwd_sl == 0)
                real_mask = ~pad_mask
                if not pad_mask.any() or not real_mask.any():
                    continue
                real_pts = pts_sl[real_mask]
                aabb_min = real_pts.min(axis=0)
                aabb_max = real_pts.max(axis=0)
                center = (aabb_min + aabb_max) / 2.0
                pads = pts_sl[pad_mask]
                # Broadcast center to pads.shape for a proper elementwise comparison
                center_tiled = np.broadcast_to(center, pads.shape)
                npt.assert_allclose(
                    pads, center_tiled, rtol=1e-7, atol=1e-6,
                    err_msg="Padding points must be at sub-cloud AABB center"
                )

        # --- Determinism: running twice gives identical outputs ---
        data2 = RTtools.point_cloud_segmentation(points_in, target_size, vec_size)
        for a, b, name in zip(data, data2, ["points_out", "sub_cloud_ind", "forward_ind", "reverse_ind"]):
            self.assertEqual(a.shape, b.shape, msg=f"{name} shape changed between runs")
            if a.dtype.kind in ('f',):
                npt.assert_allclose(a, b, rtol=0, atol=0, err_msg=f"{name} not deterministic")
            else:
                npt.assert_array_equal(a, b, err_msg=f"{name} not deterministic")

    def test(self):
        cases = [
            (1234, 200, 1),
            (1234, 200, 8),
            (50,   64, 8),
            (4097, 512, 32),
        ]
        for n_points, target_size, vec_size in cases:
            with self.subTest(n_points=n_points, target_size=target_size, vec_size=vec_size):
                self._run_case_and_assert(n_points, target_size, vec_size)

    def test_invalid_points_shape_raises(self):
        rng = np.random.default_rng(0)

        bad_points_list = [
            rng.normal(size=(100,)),      # 1D array
            rng.normal(size=(100, 2)),    # wrong second dimension
            rng.normal(size=(100, 4)),    # wrong second dimension
            np.array([[1, 2, 3], [4, 5]], dtype=object),  # ragged -> dtype=object, shape (2,)
            np.empty((1, 0)),             # 2D but zero columns
        ]

        for pts in bad_points_list:
            with self.subTest(points_shape=getattr(pts, "shape", None), dtype=getattr(pts, "dtype", None)):
                with self.assertRaises(Exception):
                    RTtools.point_cloud_segmentation(pts, target_size=64, vec_size=8)


if __name__ == "__main__":
    unittest.main()
