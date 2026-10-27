# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.los_probability.
# Scope: argument and output handling, defaults, error paths.
# Physics validation lives in the Catch2 suite (test_los_probability.cpp).

import sys
import os
import math
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


def umi(x):  # TR UMi for d > 18 m
    return 18.0 / x + np.exp(-x / 36.0) * (1.0 - 18.0 / x)


class test_case(unittest.TestCase):

    def setUp(self):
        self.d = np.array([5.0, 10.0, 50.0, 200.0])
        self.p = quadriga_lib.gbsm.los_probability("38.901_UMi", self.d, 10.0, 1.5)[0]
        self.rnd_g = np.array([0.1, -0.2, 0.3, 0.0])
        self.fc = np.array([1e9, 30e9])

    def test_basic_call(self):
        out = quadriga_lib.gbsm.los_probability("38.901_UMi", self.d, 10.0, 1.5)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        p, s = out
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(p.shape, (4,))
        self.assertEqual(p.dtype, np.float64)
        self.assertIsNone(s)
        npt.assert_array_equal(p[:2], [1.0, 1.0])  # d <= 18 m
        npt.assert_allclose(p[2:], umi(self.d[2:]), atol=1e-14, rtol=0)
        npt.assert_array_equal(quadriga_lib.gbsm.los_probability("NLOS", self.d, 10.0, 1.5)[0], np.zeros(4))
        npt.assert_array_equal(quadriga_lib.gbsm.los_probability("LOS", self.d, 10.0, 1.5)[0], np.ones(4))

    def test_keyword_arguments(self):
        pk, sk = quadriga_lib.gbsm.los_probability(model="38.901_UMi", d2d_out=self.d, h_bs=10.0, h_ut=1.5,
                                                   d_clutter=math.nan, r_clutter=math.nan, h_clutter=math.nan,
                                                   rnd_g=self.rnd_g, fc=self.fc)
        pp, sp = quadriga_lib.gbsm.los_probability("38.901_UMi", self.d, 10.0, 1.5, math.nan, math.nan, math.nan,
                                                   self.rnd_g, self.fc)
        npt.assert_array_equal(pk, self.p)
        npt.assert_array_equal(pk, pp)
        npt.assert_array_equal(sk, sp)

    def test_heights(self):
        # h_bs and h_ut: scalar or one value per link (QuaDRiGa_Industrial uses hB, UMa uses hU)
        f = quadriga_lib.gbsm.los_probability
        h_bs = np.array([2.0, 4.0, 6.0, 10.0])
        p_ind = f("QuaDRiGa_Industrial", self.d, h_bs, 1.5)[0]
        ref = np.minimum(1, np.exp(-(self.d + 20) / 80) * np.exp(np.minimum(h_bs, 8) / 5 - 1))
        npt.assert_allclose(p_ind, ref, atol=1e-14, rtol=0)
        self.assertEqual(p_ind[2], f("QuaDRiGa_Industrial", self.d, 6.0, 1.5)[0][2])
        self.assertEqual(p_ind[2], f("QuaDRiGa_Industrial", self.d, [6.0], 1.5)[0][2])
        h_ut = np.array([1.5, 1.5, 22.5, 22.5])
        p_uma = f("38.901_UMa", self.d, 25.0, h_ut)[0]
        p1 = f("38.901_UMa", self.d, 25.0, 1.5)[0]
        p2 = f("38.901_UMa", self.d, 25.0, 22.5)[0]
        npt.assert_array_equal(p_uma, np.concatenate([p1[:2], p2[2:]]))
        self.assertGreater(p_uma[3], p1[3] + 1e-3)

    def test_clutter_arguments(self):
        # Omitted, NaN and the TR defaults give the same result
        f = quadriga_lib.gbsm.los_probability
        p_sl = f("38.901_InF_SL", self.d, 8.0, 1.5)[0]
        npt.assert_allclose(p_sl, np.exp(-self.d * (-np.log(1 - 0.2) / 10)), atol=1e-14, rtol=0)
        npt.assert_array_equal(f("38.901_InF_SL", self.d, 8.0, 1.5, math.nan, math.nan, math.nan)[0], p_sl)
        npt.assert_array_equal(f("38.901_InF_SL", self.d, 8.0, 1.5, None, None, None)[0], p_sl)  # None = default
        npt.assert_array_equal(f("38.901_InF_SL", self.d, 8.0, 1.5, d_clutter=np.array([10.0]))[0], p_sl)
        npt.assert_allclose(f("38.901_InF_SL", self.d, 8.0, 1.5, 10.0, 0.2)[0], p_sl, atol=1e-15, rtol=0)
        p_d = f("38.901_InF_SL", self.d, 8.0, 1.5, d_clutter=20.0)[0]
        npt.assert_allclose(p_d, np.exp(-self.d * (-np.log(1 - 0.2) / 20)), atol=1e-14, rtol=0)
        p_r = f("38.901_InF_SL", self.d, 8.0, 1.5, r_clutter=0.4)[0]
        npt.assert_allclose(p_r, np.exp(-self.d * (-np.log(1 - 0.4) / 10)), atol=1e-14, rtol=0)
        npt.assert_array_equal(f("38.901_InF_SL", self.d, 8.0, 1.5, d_clutter=20)[0], p_d)  # int scalar
        p_sh = f("38.901_InF_SH", self.d, 8.0, 1.5)[0]  # h_c = 2 m above the UT
        self.assertTrue(np.all(p_sh[1:] < 1))
        npt.assert_array_equal(f("38.901_InF_SH", self.d, 8.0, 1.5, h_clutter=1.0)[0], np.ones(4))  # UT above

    def test_soft_los(self):
        f = quadriga_lib.gbsm.los_probability
        self.assertIsNone(f("38.901_UMi", self.d, 10.0, 1.5, rnd_g=self.rnd_g)[1])  # no fc
        self.assertIsNone(f("38.901_UMi", self.d, 10.0, 1.5, fc=self.fc)[1])  # no rnd_g
        self.assertIsNone(f("38.901_UMi", self.d, 10.0, 1.5, rnd_g=None, fc=None)[1])
        p4, s4 = f("38.901_UMi", self.d, 10.0, 1.5, rnd_g=self.rnd_g, fc=self.fc)
        npt.assert_array_equal(p4, self.p)
        self.assertEqual(s4.shape, (4, 2))
        self.assertEqual(s4.dtype, np.float64)
        npt.assert_array_equal(s4[:2], np.ones((2, 2)))  # p = 1 gives 1
        lam = 299792458.0 / self.fc
        # Inverse normal CDF of p by bisection on 0.5·erfc(-x/sqrt(2))
        F = []
        for pv in self.p[2:]:
            lo, hi = -10.0, 10.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if 0.5 * math.erfc(-mid / math.sqrt(2)) < pv:
                    lo = mid
                else:
                    hi = mid
            F.append(0.5 * (lo + hi))
        F = np.array(F)[:, None]
        ref = 0.5 + np.arctan(np.sqrt(20 / lam) * (self.rnd_g[2:, None] + F)) / np.pi
        npt.assert_allclose(s4[2:], ref, atol=1e-8, rtol=0)
        s5 = f("38.901_UMi", self.d, 10.0, 1.5, rnd_g=self.rnd_g.tolist(), fc=1e9)[1]  # list and scalar
        npt.assert_array_equal(s5, s4[:, :1])

    def test_array_layout_and_dtype(self):
        f = quadriga_lib.gbsm.los_probability
        big = np.zeros(8)
        big[::2] = self.d
        npt.assert_array_equal(f("38.901_UMi", big[::2], 10.0, 1.5)[0], self.p)  # strided view
        npt.assert_array_equal(f("38.901_UMi", self.d.reshape(1, 4), 10.0, 1.5)[0], self.p)
        npt.assert_array_equal(f("38.901_UMi", np.asfortranarray(self.d.reshape(4, 1)), 10.0, 1.5)[0], self.p)
        npt.assert_array_equal(f("38.901_UMi", self.d.astype(np.float32), 10, np.float32(1.5))[0], self.p)
        npt.assert_array_equal(f("38.901_UMi", [5, 10, 50, 200], 10, 1.5)[0], self.p)

    def test_zero_links(self):
        p, s = quadriga_lib.gbsm.los_probability("38.901_UMi", np.zeros(0), 10.0, 1.5)
        self.assertEqual(p.shape, (0,))
        self.assertIsNone(s)
        p, s = quadriga_lib.gbsm.los_probability("38.901_UMi", np.zeros(0), 10.0, 1.5, rnd_g=np.zeros(0), fc=self.fc)
        self.assertEqual(s.shape, (0, 2))

    def test_inputs_not_modified(self):
        d, g = self.d.copy(), self.rnd_g.copy()
        quadriga_lib.gbsm.los_probability("38.901_UMi", self.d, 10.0, 1.5, rnd_g=self.rnd_g, fc=self.fc)
        npt.assert_array_equal(self.d, d)
        npt.assert_array_equal(self.rnd_g, g)

    def test_errors(self):
        f = quadriga_lib.gbsm.los_probability
        with self.assertRaises(ValueError) as ctx:  # C++
            f("foo", self.d, 10.0, 1.5)
        self.assertIn("Model 'foo' is not supported.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f("38.901_UMi", self.d, [10.0, 20.0], 1.5)
        self.assertIn("Number of elements in 'h_bs' must be 1 or match 'd2d_out'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f("38.901_UMi", self.d, 10.0, 1.5, r_clutter=1.0)
        self.assertIn("Input 'r_clutter' must be NaN or in the range from 0 to 1, 1 excluded.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f("38.901_UMi", self.d, 10.0, 1.5, rnd_g=[0.0, 1.0], fc=self.fc)
        self.assertIn("Number of elements in 'rnd_g' does not match number of elements in 'd2d_out'.",
                      str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f("38.901_UMi", self.d, 10.0, 1.5, rnd_g=self.rnd_g, fc=np.zeros(0))
        self.assertIn("Input 'fc' cannot be empty.", str(ctx.exception))
        with self.assertRaises(TypeError):
            f("38.901_UMi", self.d, 10.0)
        with self.assertRaises(TypeError):
            f("38.901_UMi", self.d, 10.0, 1.5, foo=1)
        for name in ("d_clutter", "r_clutter", "h_clutter"):
            for bad in ("a", [1.0, 2.0], 1j):  # wrapper
                with self.assertRaises(ValueError) as ctx:
                    f("38.901_UMi", self.d, 10.0, 1.5, **{name: bad})
                self.assertIn(f"Input '{name}' must be a real scalar or None.", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
