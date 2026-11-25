# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.drop_terminals.
# Scope: argument and output handling, defaults, seed, in/out split, error paths.
# The drop rules are tested in the Catch2 suite (test_generate_layout.cpp).
# Physics validation lives in the Catch2 suite.

import sys
import os
import pathlib
import tempfile
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


def same(a, b):
    # Deep equality of dicts, lists and arrays; NaN equals NaN
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a.keys()) == list(b.keys()) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return (isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and
                np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    return a == b


class test_case(unittest.TestCase):

    def setUp(self):
        # UMa, one ring (7 sites, 3 sectors) and an indoor hall, both without UTs
        self.L = quadriga_lib.layout.generate_layout("38.901_UMa", 0, grid=[1, np.nan, 3], seed=9)
        self.H = quadriga_lib.layout.generate_layout("38.901_InH_Open", seed=5)
        self.bounds = [-60, 60, -25, 25]

    def test_basic_call(self):
        L = self.L
        L1 = quadriga_lib.layout.drop_terminals(L, 2, seed=9)
        self.assertEqual(L1["rx"]["pos"].shape, (3, 42))  # 2 UTs in each of the 21 cells
        self.assertEqual(L1["rx"]["name"][:2], ["UT1", "UT2"])
        npt.assert_array_equal(L1["rx"]["offset"][:3], [0, 1, 2])  # 0-based
        self.assertTrue(same(L1["tx"], L["tx"]))
        self.assertEqual(L["rx"]["pos"].size, 0)  # the input stays unchanged
        Lg = quadriga_lib.layout.generate_layout("38.901_UMa", 2, grid=[1, np.nan, 3], seed=9)  # the same drop
        self.assertTrue(same(L1, Lg))

    def test_optional_inputs(self):
        H, bounds = self.H, self.bounds
        H1 = quadriga_lib.layout.drop_terminals(H, 4, bounds=bounds, per_cell=False, seed=2)
        self.assertEqual(H1["rx"]["pos"].shape, (3, 4))  # per_cell = False: 4 UTs in total
        self.assertTrue(np.all(np.abs(H1["rx"]["pos"][0]) <= 60) and np.all(np.abs(H1["rx"]["pos"][1]) <= 25))
        npt.assert_array_equal(H1["rx"]["pos"][2], np.ones(4))  # UT height in an indoor hall
        H2 = quadriga_lib.layout.drop_terminals(H, 4, bounds=bounds, ut=[1.5, 1, 0.8, 0, 90, 0], per_cell=False, seed=2)
        npt.assert_array_equal(H2["rx"]["pos"][2], 1.5 * np.ones(4))
        npt.assert_allclose(H2["rx"]["speed"], 0.8, atol=1e-14, rtol=0)
        npt.assert_allclose(H2["rx"]["orientation"], [np.zeros(4), np.full(4, np.pi / 2), np.zeros(4)], atol=1e-14, rtol=0)
        H3 = quadriga_lib.layout.drop_terminals(H, 4, bounds=bounds, ut=None, indoor=None, options=None, per_cell=None, seed=2)
        self.assertTrue(same(H3, quadriga_lib.layout.drop_terminals(H, 4, bounds=bounds, seed=2)))  # None = the defaults
        H4 = quadriga_lib.layout.drop_terminals(H, 4, bounds=bounds, per_cell=False, seed=3)
        self.assertFalse(np.array_equal(H4["rx"]["pos"], H1["rx"]["pos"]))
        H5 = quadriga_lib.layout.drop_terminals(H, 4, bounds=bounds)  # seed = layout seed
        self.assertEqual(H5["rx"]["pos"].shape, (3, 48))
        self.assertEqual(H5["seed"], H["seed"])
        H6 = quadriga_lib.layout.drop_terminals(H, np.int32(4), bounds=np.array(bounds, dtype=np.float32), per_cell=False, seed=np.int8(2))
        self.assertTrue(same(H6, H1))
        H7 = quadriga_lib.layout.drop_terminals(layout_in=H, n_ut=4, bounds=bounds, per_cell=False, seed=2)  # by keyword
        self.assertTrue(same(H7, H1))

    def test_cell(self):
        # The micro sites of M.2412 Dense Urban config 3 have no cells of their own (uint32, 0)
        D = quadriga_lib.layout.generate_layout("M.2412_DenseUrban", 0, options={"config": 3}, seed=4)
        npt.assert_array_equal(D["tx"]["cell"], np.r_[np.ones(57), np.zeros(171)])
        self.assertEqual(D["tx"]["cell"].dtype, np.uint32)
        D1 = quadriga_lib.layout.drop_terminals(D, 1, seed=4)
        self.assertEqual(D1["rx"]["pos"].shape, (3, 228))  # 4 UTs in each of the 57 macro cells
        self.assertTrue(same(D1["tx"], D["tx"]))
        D["tx"]["cell"] = np.zeros(228, dtype=np.uint32)
        with self.assertRaisesRegex(ValueError, "Input 'layout' has no site with 'cell' = 1."):
            quadriga_lib.layout.drop_terminals(D, 1)
        D["tx"]["cell"] = None  # empty: all sites have cells
        self.assertEqual(quadriga_lib.layout.drop_terminals(D, 1, seed=4)["rx"]["pos"].shape, (3, 228))

    def test_errors(self):
        H = self.H
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # bounds is keyword-only
            quadriga_lib.layout.drop_terminals(H, 4, self.bounds)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.drop_terminals(H)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.drop_terminals(1, 4)
        with self.assertRaisesRegex(TypeError, "Input 'per_cell' must be a bool or None"):
            quadriga_lib.layout.drop_terminals(H, 4, bounds=self.bounds, per_cell=0)
        with self.assertRaisesRegex(ValueError, "Input 'bounds' is required"):  # C++
            quadriga_lib.layout.drop_terminals(H, 4)
        with self.assertRaisesRegex(ValueError, "bounds"):  # C++
            quadriga_lib.layout.drop_terminals(H, 4, bounds=[60, -60, -25, 25])
        with self.assertRaisesRegex(ValueError, "foo"):  # C++
            quadriga_lib.layout.drop_terminals(H, 4, bounds=self.bounds, options={"foo": 1})


if __name__ == "__main__":
    unittest.main()
