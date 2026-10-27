# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.o2i_loss.
# Scope: argument and output handling, model index, defaults, error paths.
# Physics validation lives in the Catch2 suite (test_o2i_loss.cpp).

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
        # 3 UTs, 3 models, 3 frequencies; UT 0: low, UT 1: car, UT 2: high
        self.fc = np.array([1e9, 10e9, 30e9])
        self.rnd = np.array([[0.2, 0.5, 0.3, 0.1], [0.7, 0.4, 0.9, -1.2], [0.95, 0.2, 0.6, 0.5]])
        self.models = ["low", "car", "high"]
        self.weights = np.array([0.5, 0.3, 0.2])
        self.out = quadriga_lib.gbsm.o2i_loss(self.fc, self.rnd, self.models, self.weights)

    def test_basic_call(self):
        out = self.out
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        loss, d2d_in, model_index = out
        self.assertEqual(loss.shape, (3, 3))
        self.assertEqual(d2d_in.shape, (3,))
        self.assertEqual(model_index.shape, (3,))
        self.assertEqual(loss.dtype, np.float64)
        self.assertEqual(d2d_in.dtype, np.float64)
        self.assertEqual(model_index.dtype, np.uint64)
        npt.assert_array_equal(model_index, [0, 1, 2])  # 0-based
        npt.assert_allclose(d2d_in, [0.3 * 25, 0.0, 0.2 * 25], atol=1e-14, rtol=0)
        npt.assert_allclose(loss[1], [3.0, 3.0, 3.0], atol=1e-14, rtol=0)  # car: 9 + 5·N, N = -1.2
        f = self.fc / 1e9  # low-loss model of UT 0
        tw = 5 - 10 * np.log10(0.3 * 10 ** (-(2 + 0.2 * f) / 10) + 0.7 * 10 ** (-(5 + 4 * f) / 10))
        npt.assert_allclose(loss[0], tw + 0.5 * 7.5 + 4.4 * 0.1, atol=1e-12, rtol=0)

    def test_keyword_arguments(self):
        ok = quadriga_lib.gbsm.o2i_loss(fc=self.fc, rnd=self.rnd, models=self.models, weights=self.weights,
                                        d2d_in_max=25.0)
        op = quadriga_lib.gbsm.o2i_loss(self.fc, self.rnd, self.models, self.weights, 25.0)
        for a, b, c in zip(ok, op, self.out):
            npt.assert_array_equal(a, c)
            npt.assert_array_equal(b, c)

    def test_d2d_in_max(self):
        loss3, d3, _ = quadriga_lib.gbsm.o2i_loss(self.fc, self.rnd, self.models, self.weights, 10.0)
        npt.assert_allclose(d3, [3.0, 0.0, 2.0], atol=1e-14, rtol=0)
        npt.assert_allclose(loss3[0] - self.out[0][0], np.full(3, 0.5 * (3 - 7.5)), atol=1e-12, rtol=0)
        npt.assert_array_equal(quadriga_lib.gbsm.o2i_loss(self.fc, self.rnd, self.models, self.weights, 25)[1],
                               self.out[1])  # int value

    def test_models(self):
        # One str or a list (or tuple) of str; the model index points into 'models' (0-based)
        l_c, _, i_c = quadriga_lib.gbsm.o2i_loss(self.fc, self.rnd, "car", [1.0])
        l_cc, _, i_cc = quadriga_lib.gbsm.o2i_loss(self.fc, self.rnd, ["car"], [1.0])
        npt.assert_array_equal(l_c, l_cc)
        npt.assert_array_equal(i_c, [0, 0, 0])
        npt.assert_array_equal(i_cc, i_c)
        _, _, i_r = quadriga_lib.gbsm.o2i_loss(self.fc, self.rnd, ("high", "car", "low"), [0.5, 0.3, 0.2])
        npt.assert_array_equal(i_r, [0, 1, 2])
        _, _, i_s = quadriga_lib.gbsm.o2i_loss(self.fc, self.rnd, ["car", "low", "high"], [0.0, 0.5, 0.5])
        npt.assert_array_equal(i_s, [1, 2, 2])  # weight 0 never chosen

    def test_array_layout_and_dtype(self):
        f = quadriga_lib.gbsm.o2i_loss
        npt.assert_array_equal(f(self.fc, np.asfortranarray(self.rnd), self.models, self.weights)[0], self.out[0])
        npt.assert_array_equal(f(self.fc, np.ascontiguousarray(self.rnd), self.models, self.weights)[0], self.out[0])
        big = np.zeros((6, 8))
        big[::2, ::2] = self.rnd
        npt.assert_array_equal(f(self.fc, big[::2, ::2], self.models, self.weights)[0], self.out[0])
        lt, dt, it = f(self.fc.astype(np.float32), self.rnd.astype(np.float32), self.models,
                       self.weights.astype(np.float32))
        npt.assert_allclose(lt, self.out[0], atol=1e-5, rtol=0)  # rnd rounded to float32
        npt.assert_allclose(dt, self.out[1], atol=1e-5, rtol=0)
        npt.assert_array_equal(it, self.out[2])
        npt.assert_array_equal(f(self.fc.tolist(), self.rnd.tolist(), self.models, self.weights.tolist())[0],
                               self.out[0])

    def test_zero_uts(self):
        loss, d2d_in, idx = quadriga_lib.gbsm.o2i_loss(self.fc, np.zeros((0, 4)), self.models, self.weights)
        self.assertEqual(loss.shape, (0, 3))
        self.assertEqual(d2d_in.shape, (0,))
        self.assertEqual(idx.shape, (0,))

    def test_inputs_not_modified(self):
        rnd, w, m = self.rnd.copy(), self.weights.copy(), list(self.models)
        quadriga_lib.gbsm.o2i_loss(self.fc, self.rnd, self.models, self.weights)
        npt.assert_array_equal(self.rnd, rnd)
        npt.assert_array_equal(self.weights, w)
        self.assertEqual(self.models, m)

    def test_errors(self):
        f = quadriga_lib.gbsm.o2i_loss
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.fc, np.zeros((3, 6)), self.models, self.weights)
        self.assertIn("Input 'rnd' must have 4 columns.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.fc, self.rnd, self.models, [0.5, 0.3, 0.3])
        self.assertIn("Input 'weights' must sum to 1.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.fc, self.rnd, ["low", "foo"], [0.5, 0.5])
        self.assertIn("Model 'foo' is not supported.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.fc, self.rnd, None, self.weights)
        self.assertIn("Input 'models' cannot be empty.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # string-list helper
            f(self.fc, self.rnd, ["low", 3], [0.5, 0.5])
        self.assertIn("Expected a string, a list of strings, or None.", str(ctx.exception))
        with self.assertRaises(TypeError):
            f(self.fc, self.rnd, self.models)
        with self.assertRaises(TypeError):
            f(self.fc, self.rnd, self.models, self.weights, foo=1)


if __name__ == "__main__":
    unittest.main()
