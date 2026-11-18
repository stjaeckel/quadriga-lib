# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.tools.calc_rotation_matrix.
# Scope: argument and output handling, ranks, flag, error paths.
# Physics validation lives in the Catch2 suite (test_quadriga_tools.cpp).

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


def rot(o):
    # R = Rz(heading) Ry(tilt) Rx(bank), column-major in 9 elements
    cb, sb, ct, st, ch, sh = np.cos(o[0]), np.sin(o[0]), np.cos(o[1]), np.sin(o[1]), np.cos(o[2]), np.sin(o[2])
    rx = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
    ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    rz = np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1]])
    return (rz @ ry @ rx).flatten(order="F")


class test_case(unittest.TestCase):

    def setUp(self):
        # 3 x 2 x 4 distinct orientations
        self.ori = np.random.default_rng(0).uniform(-np.pi, np.pi, (3, 2, 4))

    def test_basic_call(self):
        out = quadriga_lib.tools.calc_rotation_matrix(np.zeros(3))
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (9,))
        self.assertEqual(out.dtype, np.float64)
        npt.assert_allclose(out, np.eye(3).flatten(), atol=1e-14, rtol=0)
        out = quadriga_lib.tools.calc_rotation_matrix(np.array([0.0, 0.0, np.pi / 2]))  # 90° heading: x -> y
        npt.assert_allclose(out, [0, 1, 0, -1, 0, 0, 0, 0, 1], atol=1e-14, rtol=0)

    def test_ranks(self):
        out3 = quadriga_lib.tools.calc_rotation_matrix(self.ori)
        self.assertEqual(out3.shape, (9, 2, 4))
        for i in range(2):
            for j in range(4):
                npt.assert_allclose(out3[:, i, j], rot(self.ori[:, i, j]), atol=1e-14, rtol=0)
        out2 = quadriga_lib.tools.calc_rotation_matrix(self.ori[:, 1, :])
        self.assertEqual(out2.shape, (9, 4))
        npt.assert_array_equal(out2, out3[:, 1, :])

    def test_transposed(self):
        out = quadriga_lib.tools.calc_rotation_matrix(self.ori)
        npt.assert_array_equal(quadriga_lib.tools.calc_rotation_matrix(self.ori, transposed=None), out)
        npt.assert_array_equal(quadriga_lib.tools.calc_rotation_matrix(self.ori, transposed=False), out)
        out_t = quadriga_lib.tools.calc_rotation_matrix(self.ori, transposed=True)
        npt.assert_allclose(out_t[:, 0, 0].reshape(3, 3), out[:, 0, 0].reshape(3, 3).T, atol=1e-14, rtol=0)
        out_k = quadriga_lib.tools.calc_rotation_matrix(orientation=self.ori, transposed=True)
        npt.assert_array_equal(out_k, out_t)

    def test_array_layout_and_dtype(self):
        ori0 = self.ori.copy()
        out = quadriga_lib.tools.calc_rotation_matrix(self.ori)
        npt.assert_array_equal(self.ori, ori0)
        npt.assert_array_equal(quadriga_lib.tools.calc_rotation_matrix(np.asfortranarray(self.ori)), out)
        big = np.zeros((3, 4, 4))
        big[:, ::2, :] = self.ori
        npt.assert_array_equal(quadriga_lib.tools.calc_rotation_matrix(big[:, ::2, :]), out)
        o32 = self.ori.astype(np.float32)
        npt.assert_array_equal(quadriga_lib.tools.calc_rotation_matrix(o32),
                               quadriga_lib.tools.calc_rotation_matrix(o32.astype(np.float64)))
        npt.assert_array_equal(quadriga_lib.tools.calc_rotation_matrix([0, 0, 0]), np.eye(3).flatten())

    def test_errors(self):
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.calc_rotation_matrix(np.zeros((4, 2)))
        self.assertIn("3 rows", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.calc_rotation_matrix(np.zeros((3, 0)))
        self.assertIn("empty", str(ctx.exception))
        with self.assertRaises(TypeError) as ctx:
            quadriga_lib.tools.calc_rotation_matrix(self.ori, transposed=2.5)
        self.assertIn("transposed", str(ctx.exception))
        with self.assertRaises(TypeError):
            quadriga_lib.tools.calc_rotation_matrix()
        with self.assertRaises(TypeError):
            quadriga_lib.tools.calc_rotation_matrix(self.ori, invert=True)


if __name__ == "__main__":
    unittest.main()
