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

def aabb_numpy(triangles, sub_mesh_index=None):
    """
    Reference AABB in NumPy (always float64).
    triangles: (N, 9)
    sub_mesh_index: 1-D start indices (0-based). If empty or None => whole mesh.
    """
    t = np.asarray(triangles, dtype=np.float64)
    if sub_mesh_index is None or len(sub_mesh_index) == 0:
        xs = t[:, [0, 3, 6]].ravel()
        ys = t[:, [1, 4, 7]].ravel()
        zs = t[:, [2, 5, 8]].ravel()
        return np.array([[xs.min(), xs.max(), ys.min(), ys.max(), zs.min(), zs.max()]], dtype=np.float64)

    idx = np.asarray(sub_mesh_index).astype(np.int64, copy=False)
    ends = np.r_[idx[1:], t.shape[0]]
    out = []
    for s, e in zip(idx, ends):
        part = t[s:e]
        if part.size == 0:
            out.append(np.array([0, 0, 0, 0, 0, 0], dtype=np.float64))
        else:
            xs = part[:, [0, 3, 6]].ravel()
            ys = part[:, [1, 4, 7]].ravel()
            zs = part[:, [2, 5, 8]].ravel()
            out.append(np.array([xs.min(), xs.max(), ys.min(), ys.max(), zs.min(), zs.max()], dtype=np.float64))
    return np.vstack(out)


class test_case(unittest.TestCase):

    def setUp(self):
        # Two simple sub-mesh clusters, separated spatially
        tri1 = np.array([
            [0, 0, 0, 1, 0, -1, 0, 2, -0.5],
            [0.5, 0.5, -1, 1, 2, 0, 0, 1, -0.2],
        ], dtype=np.float64)

        tri2 = np.array([
            [10, -2, 5, 11, -1, 5.5, 10.2, -1.5, 6],
            [10.8, -1.2, 5.1, 10.1, -1.9, 5.9, 11, -1.1, 5.2],
        ], dtype=np.float64)

        self.triangles = np.vstack([tri1, tri2])  # (4, 9)
        self.sub_mesh_index = np.array([0, 2], dtype=int)  # dtype flexible

    def test_entire_mesh_aabb_single_triangle(self):
        tri = np.array([[0, 0, 0, 2, 0, 0, 1, 3, -1]], dtype=np.float64)
        expected = aabb_numpy(tri)
        # Use empty index array (None not supported)
        got = RTtools.triangle_mesh_aabb(tri, np.array([], dtype=int), 1)
        self.assertEqual(got.dtype, np.float64)
        npt.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_entire_mesh_multiple_triangles(self):
        expected = aabb_numpy(self.triangles)
        got = RTtools.triangle_mesh_aabb(self.triangles, np.array([], dtype=int), 1)
        self.assertEqual(got.dtype, np.float64)
        npt.assert_allclose(got, expected, rtol=0, atol=1e-12)
        self.assertEqual(got.shape, (1, 6))

    def test_submesh_indices(self):
        expected = aabb_numpy(self.triangles, self.sub_mesh_index)
        got = RTtools.triangle_mesh_aabb(self.triangles, self.sub_mesh_index, 1)
        self.assertEqual(got.dtype, np.float64)
        npt.assert_allclose(got, expected, rtol=0, atol=1e-12)
        self.assertEqual(got.shape, (2, 6))

    def test_vec_size_padding(self):
        vec_size = 8
        got = RTtools.triangle_mesh_aabb(self.triangles, self.sub_mesh_index, vec_size)
        self.assertEqual(got.dtype, np.float64)
        self.assertEqual(got.shape[0] % vec_size, 0)
        self.assertEqual(got.shape, (vec_size, 6))
        expected = aabb_numpy(self.triangles, self.sub_mesh_index)
        npt.assert_allclose(got[:2], expected, rtol=0, atol=1e-12)
        npt.assert_allclose(got[2:], 0.0, rtol=0, atol=0.0)

    def test_output_always_float64(self):
        # Inputs as float32 should still yield float64 output
        got = RTtools.triangle_mesh_aabb(self.triangles.astype(np.float32), self.sub_mesh_index, 1)
        self.assertEqual(got.dtype, np.float64)

    def test_submesh_index_dtype_flexibility(self):
        """
        The API converts sub_mesh_index to uint32 internally; accept different integer dtypes.
        """
        variants = [
            np.array([0, 2], dtype=int),
            np.array([0, 2], dtype=np.int64),
            np.array([0, 2], dtype=np.int32),
            np.array([0, 2], dtype=np.uint16),
            [0, 2],  # Python list
        ]
        expected = aabb_numpy(self.triangles, [0, 2])
        for v in variants:
            with self.subTest(dtype=str(np.asarray(v).dtype) if not isinstance(v, list) else "list"):
                got = RTtools.triangle_mesh_aabb(self.triangles, v, 1)
                self.assertEqual(got.dtype, np.float64)
                npt.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_randomized_against_numpy(self):
        rng = np.random.default_rng(42)
        n = 17
        tris = rng.normal(size=(n, 9)).astype(np.float64)
        starts = np.unique(rng.integers(0, n, size=5)).astype(int)
        if 0 not in starts:
            starts = np.r_[0, starts]
        starts.sort()
        expected = aabb_numpy(tris, starts)
        got = RTtools.triangle_mesh_aabb(tris, starts, 1)
        self.assertEqual(got.dtype, np.float64)
        npt.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_integration_with_segmentation(self):
        triangles_out, sub_idx, mesh_idx, mtl_prop_out = RTtools.triangle_mesh_segmentation(
            self.triangles.copy(),
            2,   # target_size
            1    # vec_size
        )
        expected = aabb_numpy(triangles_out, sub_idx)
        # Pass sub_idx as-is (dtype may vary); API should accept and cast internally
        got = RTtools.triangle_mesh_aabb(triangles_out.astype(np.float32), sub_idx, 1)
        self.assertEqual(got.dtype, np.float64)
        npt.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_no_submesh_vec_size_padding(self):
        vec_size = 4
        got = RTtools.triangle_mesh_aabb(self.triangles, np.array([], dtype=int), vec_size)
        self.assertEqual(got.dtype, np.float64)
        self.assertEqual(got.shape, (vec_size, 6))
        expected = aabb_numpy(self.triangles)
        npt.assert_allclose(got[0:1], expected, rtol=0, atol=1e-12)
        npt.assert_allclose(got[1:], 0.0, rtol=0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
