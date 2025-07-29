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


def make_grid_triangles(nx: int, ny: int, spacing: float = 1.0, z: float = 0.0, dtype=np.float64):
    """
    Create a regular (nx x ny) grid of points in the XY plane and split each cell into 2 triangles.
    Returns an array of shape [n_triangles, 9] with dtype=float64 by default.
    """
    xs = np.arange(nx, dtype=dtype) * dtype(spacing)
    ys = np.arange(ny, dtype=dtype) * dtype(spacing)
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]
            v00 = [x0, y0, dtype(z)]
            v10 = [x1, y0, dtype(z)]
            v01 = [x0, y1, dtype(z)]
            v11 = [x1, y1, dtype(z)]
            # two triangles per quad
            tris.append(v00 + v10 + v11)
            tris.append(v00 + v11 + v01)
    return np.asarray(tris, dtype=dtype)


def triangles_to_vertices(triangles: np.ndarray):
    """Reshape [N, 9] -> [N, 3, 3] for convenience."""
    return triangles.reshape((-1, 3, 3))


class test_version(unittest.TestCase):

    def test(self):
        rng = np.random.default_rng(0)

        # --- Prepare a reproducible input mesh (float64) ---
        triangles_f64 = make_grid_triangles(nx=20, ny=15, spacing=1.0, z=0.0, dtype=np.float64)
        n_in = triangles_f64.shape[0]
        self.assertEqual(triangles_f64.dtype, np.float64)

        # Random permutation to ensure order is not accidentally preserved
        perm = rng.permutation(n_in)
        triangles_f64 = triangles_f64[perm]

        # Create material properties with identifiable mapping (shape [n, 5], float64)
        # Col0 is a unique id encoded as float64 to keep everything in double precision.
        mtl_prop_in = np.column_stack([
            np.arange(n_in, dtype=np.float64),                 # unique id (as float64)
            rng.integers(0, 4, size=n_in).astype(np.float64),  # category (as float64)
            rng.random(n_in, dtype=np.float64),                # value1
            rng.random(n_in, dtype=np.float64),                # value2
            rng.random(n_in, dtype=np.float64)                 # value3
        ])
        self.assertEqual(mtl_prop_in.dtype, np.float64)

        # Helper to compute per-submesh start/end indices and sizes
        def submesh_ranges(sub_idx: np.ndarray, n_out: int):
            sub_idx = np.asarray(sub_idx, dtype=np.int64)
            self.assertGreaterEqual(sub_idx.min(initial=0), 0)
            self.assertTrue(np.all(np.diff(sub_idx) > 0), "sub_mesh_index must be strictly increasing")
            starts = sub_idx
            ends = np.append(sub_idx[1:], n_out)
            sizes = ends - starts
            self.assertTrue(np.all(sizes > 0), "All submeshes must be non-empty")
            return starts, ends, sizes

        # ---------- Case A: Default usage with materials (vec_size=1 to avoid padding) ----------
        with self.subTest("default_args_with_materials"):
            tri_out, sub_idx, mesh_idx, mtl_out = RTtools.triangle_mesh_segmentation(
                triangles_f64, mtl_prop=mtl_prop_in
            )

            # Dtypes and shapes (all floats must be double precision)
            self.assertEqual(tri_out.dtype, np.float64)
            self.assertEqual(mtl_out.dtype, np.float64)
            self.assertEqual(tri_out.shape[1], 9)
            self.assertEqual(len(mesh_idx), tri_out.shape[0])
            self.assertEqual(mtl_out.shape, (tri_out.shape[0], 5))
            self.assertEqual(np.asarray(sub_idx).dtype.kind, 'i')  # integer start indices
            self.assertEqual(np.asarray(mesh_idx).dtype.kind, 'i') # integer mapping indices

            # No padding expected when vec_size == 1
            self.assertTrue(np.all(mesh_idx > 0), "No zeros expected in mesh_index for vec_size == 1")

            # Mapping correctness: triangles_out must be a reordering of triangles_in
            npt.assert_allclose(tri_out, triangles_f64[mesh_idx - 1], rtol=0, atol=0)

            # Materials mapping: output materials at mapped rows equal input at (mesh_idx-1)
            npt.assert_array_equal(mtl_out, mtl_prop_in[mesh_idx - 1])

            # Sub-mesh index sanity
            starts, ends, sizes = submesh_ranges(sub_idx, tri_out.shape[0])
            self.assertEqual(starts[0], 0)

        # ---------- Case B: Explicit target_size (force multiple submeshes, vec_size=1) ----------
        with self.subTest("target_size_only"):
            target_size = 50
            tri_out, sub_idx, mesh_idx, mtl_out = RTtools.triangle_mesh_segmentation(
                triangles_f64, target_size=target_size, vec_size=1, mtl_prop=mtl_prop_in
            )

            self.assertEqual(tri_out.dtype, np.float64)
            self.assertEqual(mtl_out.dtype, np.float64)

            # Mapping and materials consistency (no padding here)
            npt.assert_allclose(tri_out, triangles_f64[mesh_idx - 1], rtol=0, atol=0)
            npt.assert_array_equal(mtl_out, mtl_prop_in[mesh_idx - 1])

            # Submesh partitioning is valid
            starts, ends, sizes = submesh_ranges(sub_idx, tri_out.shape[0])
            self.assertGreater(len(starts), 1)

        # ---------- Case C: vec_size > 1 triggers padding to multiples per submesh ----------
        with self.subTest("vec_size_padding_and_aabb_center"):
            vec_size = 8
            target_size = 33  # deliberately not a multiple of vec_size
            tri_out, sub_idx, mesh_idx, mtl_out = RTtools.triangle_mesh_segmentation(
                triangles_f64, target_size=target_size, vec_size=vec_size, mtl_prop=mtl_prop_in
            )

            self.assertEqual(tri_out.dtype, np.float64)
            self.assertEqual(mtl_out.dtype, np.float64)

            # Submesh sizes must be multiples of vec_size
            starts, ends, sizes = submesh_ranges(sub_idx, tri_out.shape[0])
            self.assertTrue(np.all(sizes % vec_size == 0), "Each submesh size must be a multiple of vec_size")

            # Identify padded rows via mesh_index == 0
            mesh_idx = np.asarray(mesh_idx)
            pad_mask = (mesh_idx == 0)
            self.assertTrue(np.any(pad_mask), "Expected some padding rows when vec_size > 1")

            # For non-padded rows: geometry/material mapping correctness
            nonpad_idx = np.where(~pad_mask)[0]
            npt.assert_allclose(tri_out[nonpad_idx], triangles_f64[mesh_idx[nonpad_idx] - 1], rtol=0, atol=0)
            npt.assert_array_equal(mtl_out[nonpad_idx], mtl_prop_in[mesh_idx[nonpad_idx] - 1])

            # For padded rows: triangles must be zero-sized, placed at submesh AABB center
            tri_out_v = triangles_to_vertices(tri_out)

            # Precompute submesh AABB centers from NON-padded triangles only
            centers = []
            for s, e in zip(starts, ends):
                sel = np.arange(s, e)
                sel_np = sel[~pad_mask[sel]]
                self.assertGreater(len(sel_np), 0, "Submesh should contain at least one non-padded triangle")
                verts = tri_out_v[sel_np].reshape((-1, 3))  # stack all vertices
                vmin = verts.min(axis=0)
                vmax = verts.max(axis=0)
                centers.append((vmin + vmax) / 2.0)
            centers = np.asarray(centers, dtype=np.float64)

            # Check each padded triangle row
            for i in np.where(pad_mask)[0]:
                # Determine which submesh the row i belongs to
                k = np.searchsorted(starts, i, side='right') - 1
                c = centers[k]
                v = tri_out_v[i]  # shape (3,3)
                # Zero-sized: all three vertices identical
                npt.assert_allclose(v[0], v[1], rtol=0, atol=1e-12)
                npt.assert_allclose(v[1], v[2], rtol=0, atol=1e-12)
                # Located at AABB center (tight tolerance in double precision)
                npt.assert_allclose(v[0], c, rtol=0, atol=1e-12)

        # ---------- Case D: Without materials (mtl_prop_out should be empty) ----------
        with self.subTest("no_materials_provided"):
            tri_out, sub_idx, mesh_idx, mtl_out = RTtools.triangle_mesh_segmentation(
                triangles_f64, target_size=64, vec_size=1
            )
            self.assertEqual(tri_out.dtype, np.float64)
            empty = (mtl_out is None) or (isinstance(mtl_out, np.ndarray) and mtl_out.size == 0)
            self.assertTrue(empty, "mtl_prop_out should be empty when mtl_prop is not provided")

        # ---------- Case E: Determinism (same inputs -> same outputs) ----------
        with self.subTest("deterministic_results"):
            args = dict(target_size=64, vec_size=8, mtl_prop=mtl_prop_in)
            out1 = RTtools.triangle_mesh_segmentation(triangles_f64, **args)
            out2 = RTtools.triangle_mesh_segmentation(triangles_f64, **args)

            for a, b in zip(out1, out2):
                if isinstance(a, np.ndarray) and a.dtype.kind in ('f',):
                    self.assertEqual(a.dtype, np.float64)
                    npt.assert_allclose(a, b, rtol=0, atol=0)
                else:
                    npt.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
