# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.sos.preset.
# Scope: argument and output handling, defaults, array handling, error paths.
# Physics validation lives in the Catch2 suite (test_sos_preset.cpp).

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
        out = quadriga_lib.sos.preset("Comb256")
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        freq, acf, dist = out

        self.assertEqual(freq.shape, (256, 3))
        self.assertEqual(acf.shape, (200,))
        self.assertEqual(dist.shape, (200,))
        for x in out:
            self.assertIsInstance(x, np.ndarray)
            self.assertEqual(x.dtype, np.float64)

        npt.assert_allclose(dist[[0, 1, 199]], [0.0, 0.25, 49.75], atol=1e-13, rtol=0)
        npt.assert_allclose(acf[[0, 40, 80]], [1.0, np.exp(-1.0), np.exp(-2.0)], atol=1e-14, rtol=0)  # Comb at 0, 10, 20 m
        self.assertGreater(np.max(np.abs(freq)), 0.01)

    def test_name_selects_type_and_size(self):
        freq, acf, _ = quadriga_lib.sos.preset("Exp128")
        self.assertEqual(freq.shape, (128, 3))
        npt.assert_allclose(acf[80], np.exp(-2.0), atol=1e-14, rtol=0)  # Exp at 20 m

        freq, acf, _ = quadriga_lib.sos.preset("Gauss1024")
        self.assertEqual(freq.shape, (1024, 3))
        npt.assert_allclose(acf[80], np.exp(-4.0), atol=1e-14, rtol=0)  # Gauss at 20 m

    def test_keyword_arguments(self):
        ref = quadriga_lib.sos.preset("Comb256", 25.0)
        out = quadriga_lib.sos.preset(name="Comb256", dist_decorr=25.0)
        for a, b in zip(out, ref):
            npt.assert_array_equal(a, b)

    def test_optional_inputs(self):
        freq, acf, dist = quadriga_lib.sos.preset("Comb256")

        # The default is 10 m
        out = quadriga_lib.sos.preset("Comb256", 10.0)
        for a, b in zip(out, (freq, acf, dist)):
            npt.assert_array_equal(a, b)

        # Another decorrelation distance scales the distances and the frequencies
        freq2, acf2, dist2 = quadriga_lib.sos.preset("Comb256", 25.0)
        npt.assert_allclose(dist2, dist * 2.5, atol=1e-12, rtol=0)
        npt.assert_allclose(freq2 * 2.5, freq, atol=1e-14, rtol=0)
        npt.assert_allclose(acf2, acf, atol=1e-14, rtol=0)

        # Integer and numpy scalars
        for d in (25, np.float32(25.0), np.int64(25)):
            freq3, _, _ = quadriga_lib.sos.preset("Comb256", d)
            npt.assert_array_equal(freq3, freq2)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.preset("comb256")  # the name is matched exactly
        self.assertIn("name", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.preset("Comb256", 0.0)
        self.assertIn("dist_decorr", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.preset()
        with self.assertRaises(TypeError):
            quadriga_lib.sos.preset("Comb256", no_such_argument=1)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.preset(256)


if __name__ == "__main__":
    unittest.main()
