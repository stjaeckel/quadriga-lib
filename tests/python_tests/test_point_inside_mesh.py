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

    def test(self):

        # Define the mesh as a (12 × 9) array. Each row is one triangular face:
        # [x₁, y₁, z₁,   x₂, y₂, z₂,   x₃, y₃, z₃]
        mesh = np.array([
            [-1.0,  1.0,  1.0,   1.0, -1.0,  1.0,   1.0,  1.0,  1.0],   #  0 Top NorthEast
            [ 1.0, -1.0,  1.0,  -1.0, -1.0, -1.0,   1.0, -1.0, -1.0],   #  1 South Lower
            [-1.0, -1.0,  1.0,  -1.0,  1.0, -1.0,  -1.0, -1.0, -1.0],   #  2 West Lower
            [ 1.0,  1.0, -1.0,  -1.0, -1.0, -1.0,  -1.0,  1.0, -1.0],   #  3 Bottom NorthWest
            [ 1.0,  1.0,  1.0,   1.0, -1.0, -1.0,   1.0,  1.0, -1.0],   #  4 East Lower
            [-1.0,  1.0,  1.0,   1.0,  1.0, -1.0,  -1.0,  1.0, -1.0],   #  5 North Lower
            [-1.0,  1.0,  1.0,  -1.0, -1.0,  1.0,   1.0, -1.0,  1.0],   #  6 Top SouthWest
            [ 1.0, -1.0,  1.0,  -1.0, -1.0,  1.0,  -1.0, -1.0, -1.0],   #  7 South Upper
            [-1.0, -1.0,  1.0,  -1.0,  1.0,  1.0,  -1.0,  1.0, -1.0],   #  8 West Upper
            [ 1.0,  1.0, -1.0,   1.0, -1.0, -1.0,  -1.0, -1.0, -1.0],   #  9 Bottom SouthEast
            [ 1.0,  1.0,  1.0,   1.0, -1.0,  1.0,   1.0, -1.0, -1.0],   # 10 East Upper
            [-1.0,  1.0,  1.0,   1.0,  1.0,  1.0,   1.0,  1.0, -1.0]    # 11 North Upper
        ])

        # Define two query points [(0, 0, 0.5), (-1.1, 0, 0)]
        points = np.array([
            [ 0.0,  0.0,  0.5],
            [-1.1,  0.0,  0.0]
        ])

        # Define obj_ind = [2, 2, 2, ..., 2]
        obj_ind = np.full(12, 2)

        # ──────────── Case 1: Provide (points, mesh, obj_ind) ────────────
        # C++ expected:  res.n_elem == 2; res[0] == 2; res[1] == 0
        res = RTtools.point_inside_mesh(points, mesh, obj_ind)
        # Check that two results are returned
        self.assertEqual(res.shape[0], 2)
        # First point should lie in object index 2
        self.assertEqual(int(res[0]), 2)
        # Second point is outside all triangles → returns 0
        self.assertEqual(int(res[1]), 0)

        # ──────────── Case 2: Provide only (points, mesh) ────────────
        # C++ expected:  res.n_elem == 2; res[0] == 1; res[1] == 0
        res = RTtools.point_inside_mesh(points, mesh)
        self.assertEqual(res.shape[0], 2)
        # Without obj_ind, it should default to "1"
        self.assertEqual(int(res[0]), 1)
        self.assertEqual(int(res[1]), 0)

        # ──────────── Case 3: (points, mesh, None, distance = 0.12) ────────────
        # C++ expected:  res.n_elem == 2; res[0] == 1; res[1] == 1
        res = RTtools.point_inside_mesh(points, mesh, distance=0.12)
        self.assertEqual(res.shape[0], 2)
        self.assertEqual(int(res[0]), 1)
        # The second point is still just outside → but within 0.12 distance of face 0, so index=1
        self.assertEqual(int(res[1]), 1)

        # ──────────── Case 4: (points, mesh, obj_ind, distance = 0.09) ────────────
        # C++ expected:  res.n_elem == 2; res[0] == 2; res[1] == 0
        res = RTtools.point_inside_mesh(points, mesh, obj_ind, 0.09)
        self.assertEqual(res.shape[0], 2)
        # Tolerance too small → same as Case 1
        self.assertEqual(int(res[0]), 2)
        self.assertEqual(int(res[1]), 0)

        # ──────────── Case 5: (points, mesh, obj_ind, distance = 2.0) ────────────
        # C++ expected:  res.n_elem == 2; res[0] == 2; res[1] == 2
        res = RTtools.point_inside_mesh(points, mesh, obj_ind, 2.0)
        self.assertEqual(res.shape[0], 2)
        # With a huge tolerance, both points fall within the distance of some face whose obj_ind=2
        self.assertEqual(int(res[0]), 2)
        self.assertEqual(int(res[1]), 2)
        