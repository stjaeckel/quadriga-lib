# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.sos.init_phases.
# Scope: argument and output handling, defaults, array handling, error paths.
# Physics validation lives in the Catch2 suite (test_sos_init_phases.cpp).

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

    def test_basic_call(self):
        phase = quadriga_lib.sos.init_phases(7, 3, 42)
        self.assertIsInstance(phase, np.ndarray)
        self.assertEqual(phase.shape, (7, 3))  # (n_coeff, n_gen)
        self.assertEqual(phase.dtype, np.float64)
        self.assertTrue(np.all(phase >= -np.pi) and np.all(phase < np.pi))
        self.assertGreater(np.max(np.abs(phase)), 1.0)
        self.assertEqual(np.unique(phase).size, 21)  # every element is written

    def test_keyword_arguments(self):
        ref = quadriga_lib.sos.init_phases(7, 3, 42)
        phase = quadriga_lib.sos.init_phases(n_coeff=7, n_gen=3, seed=42)
        npt.assert_array_equal(phase, ref)

    def test_seed_and_fill_order(self):
        ref = quadriga_lib.sos.init_phases(7, 3, 42)

        # A seed is reproducible; the array is filled column by column from one stream
        npt.assert_array_equal(quadriga_lib.sos.init_phases(7, 3, 42), ref)
        npt.assert_array_equal(quadriga_lib.sos.init_phases(7, 5, 42)[:, :3], ref)

        # n_coeff and n_gen are not swapped
        phase = quadriga_lib.sos.init_phases(3, 7, 42)
        self.assertEqual(phase.shape, (3, 7))
        npt.assert_array_equal(phase.flatten(order="F"), ref.flatten(order="F"))

        phase = quadriga_lib.sos.init_phases(7, 3, 43)
        self.assertGreater(np.max(np.abs(phase - ref)), 0.1)

        # The random state of numpy has no effect
        np.random.seed(1)
        a = quadriga_lib.sos.init_phases(7, 3, 42)
        np.random.seed(2)
        npt.assert_array_equal(quadriga_lib.sos.init_phases(7, 3, 42), a)

    def test_optional_inputs(self):
        ref = quadriga_lib.sos.init_phases(7, 3, 42)

        # n_gen = 1 and seed = -1 (system random device: new values on each call)
        p1 = quadriga_lib.sos.init_phases(9)
        p2 = quadriga_lib.sos.init_phases(9)
        self.assertEqual(p1.shape, (9, 1))
        self.assertGreater(np.max(np.abs(p1 - p2)), 1e-6)

        npt.assert_array_equal(quadriga_lib.sos.init_phases(7, seed=42), ref[:, :1])

        p3 = quadriga_lib.sos.init_phases(7, 3, -1)
        self.assertGreater(np.max(np.abs(p3 - ref)), 1e-6)

        # numpy integers
        npt.assert_array_equal(quadriga_lib.sos.init_phases(np.uint32(7), np.int64(3), np.int32(42)), ref)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.init_phases(0)
        self.assertIn("n_coeff", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.init_phases(7, 0)
        self.assertIn("n_gen", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.init_phases(7, 3, -2)
        self.assertIn("seed", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.init_phases()
        with self.assertRaises(TypeError):
            quadriga_lib.sos.init_phases(7, no_such_argument=1)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.init_phases(-7)   # unsigned
        with self.assertRaises(TypeError):
            quadriga_lib.sos.init_phases(7.0)  # no implicit conversion from float


if __name__ == "__main__":
    unittest.main()
