# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for the integer array inputs of all wrappers (indices, counts, labels).
# Scope: values of another dtype are checked, not wrapped or truncated; the shared input helper is exercised through
# a few wrappers with uint64 and uint32 inputs.

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


class test_case(unittest.TestCase):

    def setUp(self):
        ant = quadriga_lib.arrayant.generate("omni")
        self.ant = quadriga_lib.arrayant.copy_element(ant, 0, [1, 2])
        self.tri = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

    def test_integer_values_of_any_dtype(self):
        ref = quadriga_lib.arrayant.copy_element(self.ant, np.array([1], dtype=np.uint64), [0])["e_theta_re"]
        for i_source in ([1], 1, np.array([1], dtype=np.int32), np.array([1.0]), np.array([1.0], dtype=np.float32)):
            out = quadriga_lib.arrayant.copy_element(self.ant, i_source, [0])["e_theta_re"]
            npt.assert_array_equal(out, ref)
        _, mtl = quadriga_lib.RTtools.subdivide_triangles(1, self.tri, np.array([2**40], dtype=np.int64))
        npt.assert_array_equal(mtl, [2**40])

    def test_invalid_values(self):
        # uint64 input (copy_element) and uint32 input (sub_mesh_ind of triangle_mesh_aabb)
        for bad in (np.array([-1]), np.array([1.9]), [1.9], [-1], np.array([np.nan]), np.array([np.inf])):
            with self.assertRaises(ValueError) as ctx:
                quadriga_lib.arrayant.copy_element(self.ant, bad, [0])
            self.assertIn("i_source", str(ctx.exception))
        mesh = quadriga_lib.RTtools.cube()[0]
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.RTtools.triangle_mesh_aabb(mesh, np.array([0, 2**32]))  # above the uint32 range
        self.assertIn("sub_mesh_ind", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
