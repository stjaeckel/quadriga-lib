# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.RTtools.point_inside_mesh.

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

        points = np.array([
            [ 0.0,  0.0,  0.5],
            [-1.1,  0.0,  0.0]
        ])

        # All faces belong to object 1 (0-based), reported as 2 (1-based)
        obj_ind = np.full(12, 1)

        # Case 1: (points, mesh, obj_ind)
        res = RTtools.point_inside_mesh(points, mesh, obj_ind)
        self.assertEqual(res.shape[0], 2)
        # First point should lie in object index 2
        self.assertEqual(int(res[0]), 2)
        # Second point is outside all triangles → returns 0
        self.assertEqual(int(res[1]), 0)

        # Case 2: (points, mesh)
        res = RTtools.point_inside_mesh(points, mesh)
        self.assertEqual(res.shape[0], 2)
        # Without obj_ind, it should default to "1"
        self.assertEqual(int(res[0]), 1)
        self.assertEqual(int(res[1]), 0)

        # Case 3: (points, mesh, None, distance = 0.12)
        res = RTtools.point_inside_mesh(points, mesh, distance=0.12)
        self.assertEqual(res.shape[0], 2)
        self.assertEqual(int(res[0]), 1)
        # The second point is 0.1 m outside the west face (x = -1), within distance 0.12
        self.assertEqual(int(res[1]), 1)

        # Case 4: (points, mesh, obj_ind, distance = 0.09)
        res = RTtools.point_inside_mesh(points, mesh, obj_ind, distance=0.09)
        self.assertEqual(res.shape[0], 2)
        # Tolerance too small → same as Case 1
        self.assertEqual(int(res[0]), 2)
        self.assertEqual(int(res[1]), 0)

        # Case 5: (points, mesh, obj_ind, distance = 2.0)
        res = RTtools.point_inside_mesh(points, mesh, obj_ind, distance=2.0)
        self.assertEqual(res.shape[0], 2)
        # With distance 2.0 both points count as inside object 2 (1-based)
        self.assertEqual(int(res[0]), 2)
        self.assertEqual(int(res[1]), 2)
        

    def test_output_dtype(self):
        """The result is int64 of shape (n_points,)."""
        res = RTtools.point_inside_mesh(np.zeros((2, 3)), RTtools.cube())
        self.assertEqual(res.dtype, np.int64)
        self.assertEqual(res.shape, (2,))
        res = RTtools.point_inside_mesh(np.zeros((1, 3)), RTtools.cube(), np.ones(12, dtype=np.uint64))
        self.assertEqual(res.dtype, np.int64)

    def test_none_inputs(self):
        """The mandatory arrays refuse None."""
        with self.assertRaises(TypeError):
            RTtools.point_inside_mesh(None, RTtools.cube())
        with self.assertRaises(TypeError):
            RTtools.point_inside_mesh(np.zeros((1, 3)), None)
