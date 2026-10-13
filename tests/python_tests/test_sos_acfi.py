# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.sos.acfi.
# Scope: argument and output handling, defaults, array handling, error paths.
# Physics validation lives in the Catch2 suite (test_sos_acfi.cpp).

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
        # Table with unequal spacing
        self.acf = np.array([1.0, 0.5, 0.2])
        self.dist = np.array([0.0, 1.0, 3.0])
        self.d = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 10.0])
        self.ref = np.array([1.0, 0.75, 0.5, 0.35, 0.2, 0.2])

    def test_basic_call(self):
        val = quadriga_lib.sos.acfi(self.acf, self.dist, self.d)
        self.assertIsInstance(val, np.ndarray)
        self.assertEqual(val.shape, (6,))
        self.assertEqual(val.dtype, np.float64)
        npt.assert_allclose(val, self.ref, atol=1e-14, rtol=0)

    def test_keyword_arguments(self):
        val = quadriga_lib.sos.acfi(acf=self.acf, dist=self.dist, d=self.d)
        npt.assert_allclose(val, self.ref, atol=1e-14, rtol=0)

    def test_array_layout_and_dtype(self):
        variants = [
            lambda a: np.repeat(a, 2)[::2],     # non-contiguous view
            lambda a: a.reshape(-1, 1),         # column
            lambda a: a.reshape(1, -1),         # row
            lambda a: a.astype(np.float32),     # other dtype
            lambda a: a.tolist(),               # plain list
        ]
        for fn in variants:
            val = quadriga_lib.sos.acfi(fn(self.acf), fn(self.dist), fn(self.d))
            self.assertEqual(val.shape, (6,))
            self.assertEqual(val.dtype, np.float64)
            npt.assert_allclose(val, self.ref, atol=1e-6, rtol=0)  # float32 input

        # Integer distances; an empty d gives an empty result
        val = quadriga_lib.sos.acfi(self.acf, self.dist, np.array([0, 1, 2, 3]))
        npt.assert_allclose(val, [1.0, 0.5, 0.35, 0.2], atol=1e-14, rtol=0)
        self.assertEqual(quadriga_lib.sos.acfi(self.acf, self.dist, []).shape, (0,))

    def test_inputs_not_modified(self):
        args = [self.acf, self.dist, self.d]
        copies = [a.copy() for a in args]
        quadriga_lib.sos.acfi(*args)
        for a, c in zip(args, copies):
            npt.assert_array_equal(a, c)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.acfi(self.acf, self.dist[:2], self.d)
        self.assertIn("dist", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.acfi(self.acf, self.dist, [-1.0])
        self.assertIn("'d'", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.acfi(self.acf, self.dist)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.acfi(self.acf, self.dist, self.d, no_such_argument=1)

    def test_none_for_mandatory_arrays(self):
        # None for a mandatory array is rejected by pybind11; it must not be read as one value of NaN
        args = [self.acf, self.dist, self.d]
        quadriga_lib.sos.acfi(*args)
        for i in range(3):
            bad = list(args)
            bad[i] = None
            with self.assertRaises(TypeError):
                quadriga_lib.sos.acfi(*bad)


if __name__ == "__main__":
    unittest.main()
