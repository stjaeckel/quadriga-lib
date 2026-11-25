# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.generate_layout.
# Scope: argument and output handling, defaults, seed, units, error paths.
# The presets are tested in the Catch2 suite (test_generate_layout.cpp).
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


LAYOUT_KEYS = ["name", "tx", "rx", "links", "seed", "center_freq", "antenna", "antenna_cols", "antenna_args", "wrap", "pairing"]


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
        # UMa, one ring of sites (7 sites with 3 sectors = 21 Tx nodes), 2 UTs per cell
        self.L = quadriga_lib.layout.generate_layout("38.901_UMa", 2, grid=[1, np.nan, 3], seed=42)

    def test_basic_call(self):
        L, d2r = self.L, np.pi / 180
        self.assertEqual(list(L.keys()), LAYOUT_KEYS)
        self.assertEqual(L["tx"]["pos"].shape, (3, 21))
        self.assertEqual(L["rx"]["pos"].shape, (3, 42))  # 2 UTs in each of the 21 cells
        self.assertEqual(L["tx"]["name"][:4], ["BS1", "BS1", "BS1", "BS2"])
        self.assertEqual(L["rx"]["name"][:2], ["UT1", "UT2"])
        npt.assert_array_equal(L["tx"]["sector"][:4], [0, 1, 2, 0])
        npt.assert_array_equal(L["tx"]["offset"][:3], [0, 1, 2])  # 0-based
        npt.assert_array_equal(L["tx"]["pos"][:, 0], [0, 0, 25])  # site 1 at the origin, hBS = 25 m
        npt.assert_allclose(np.linalg.norm(L["tx"]["pos"][:2, 3] - L["tx"]["pos"][:2, 0]), 500, atol=1e-9, rtol=0)  # ISD
        npt.assert_allclose(L["tx"]["orientation"][2, :3], np.array([30, 150, -90]) * d2r, atol=1e-14, rtol=0)  # in rad
        self.assertEqual(L["tx"]["scenario"][:2], ["UMa", "UMa"])
        self.assertEqual(L["wrap"].shape, (2, 2))
        self.assertEqual(L["seed"], 42)
        self.assertEqual(L["rx"]["indoor"].dtype, np.int32)
        self.assertTrue(np.all(L["rx"]["pos"][2] >= 1.5))
        npt.assert_allclose(L["rx"]["speed"], 3 / 3.6, atol=1e-14, rtol=0)  # 3 km/h
        self.assertEqual(L["links"]["i_tx"].size, 0)
        self.assertEqual(L["center_freq"].size, 0)
        self.assertEqual(L["antenna"], [])

    def test_seed(self):
        L2 = quadriga_lib.layout.generate_layout("38.901_UMa", 2, grid=[1, np.nan, 3], seed=42)
        self.assertTrue(same(self.L, L2))
        L3 = quadriga_lib.layout.generate_layout("38.901_UMa", 2, grid=[1, np.nan, 3], seed=43)
        npt.assert_array_equal(L3["tx"]["pos"], self.L["tx"]["pos"])
        self.assertFalse(np.array_equal(L3["rx"]["pos"], self.L["rx"]["pos"]))
        Ln = quadriga_lib.layout.generate_layout("38.901_UMa", grid=[0, np.nan, 1])  # no seed: one is drawn; n_ut = 0
        self.assertGreaterEqual(Ln["seed"], 0)
        self.assertEqual(Ln["rx"]["pos"].size, 0)

    def test_optional_inputs(self):
        d2r = np.pi / 180
        kw = dict(spacing=[1000, np.nan], grid=[0, np.nan, 1], bs=[10, np.nan, 15], ut=[np.nan, np.nan, np.nan, 0, 90],
                  per_cell=False, seed=7)
        L4 = quadriga_lib.layout.generate_layout("38.901_UMa", 5, **kw)  # tuples in m and degrees, NaN = preset value
        self.assertEqual(L4["tx"]["pos"].shape, (3, 1))
        self.assertEqual(L4["rx"]["pos"].shape, (3, 5))  # per_cell = False: 5 UTs in total
        self.assertEqual(L4["tx"]["pos"][2, 0], 10)
        npt.assert_allclose(L4["tx"]["orientation"][:, 0], np.array([0, 0, 45]) * d2r, atol=1e-14, rtol=0)  # 30 deg + 15 deg
        npt.assert_allclose(L4["rx"]["orientation"][:2], [np.zeros(5), np.full(5, 90 * d2r)], atol=1e-14, rtol=0)
        Ls = quadriga_lib.layout.generate_layout("38.901_SMa", 1, grid=[0, np.nan, 1], options={"p_residential": 0.5}, seed=3)
        self.assertEqual(Ls["tx"]["pos"].shape, (3, 1))
        Lh = quadriga_lib.layout.generate_layout("38.901_InH_Open", seed=5)  # indoor hall: 6 x 2 BSs at 20 m, no wrap
        self.assertEqual(Lh["tx"]["pos"].shape, (3, 12))
        npt.assert_array_equal(Lh["tx"]["pos"][:, [0, 11]], [[-50, 50], [-10, 10], [3, 3]])
        self.assertEqual(Lh["wrap"].size, 0)
        # None for every optional argument is the default; other types
        Ln = quadriga_lib.layout.generate_layout("38.901_InH_Open", 0, spacing=None, grid=None, area=None, bs=None, ut=None,
                                                 indoor=None, options=None, per_cell=None, seed=5)
        self.assertTrue(same(Ln, Lh))
        L5 = quadriga_lib.layout.generate_layout("38.901_UMa", np.int32(5), spacing=np.array([1000, np.nan], dtype=np.float32),
                                                 grid=(0, np.nan, 1), bs=[10, np.nan, 15], ut=[np.nan] * 3 + [0, 90],
                                                 per_cell=False, seed=np.int8(7))
        self.assertTrue(same(L5, L4))
        L6 = quadriga_lib.layout.generate_layout("38.901_UMa", n_ut=5, **kw)  # n_ut by keyword
        self.assertTrue(same(L6, L4))
        Lf = quadriga_lib.layout.generate_layout("38.901_UMa", 0, [3.5e9, 28e9], grid=[0, np.nan, 1])  # by position
        npt.assert_array_equal(Lf["center_freq"], [3.5e9, 28e9])
        Lk = quadriga_lib.layout.generate_layout("38.901_UMa", center_freq=np.array([3.5e9]), grid=[0, np.nan, 1])
        npt.assert_array_equal(Lk["center_freq"], [3.5e9])

    def test_errors(self):
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # the tuples are keyword-only
            quadriga_lib.layout.generate_layout("38.901_UMa", 1, None, [1000, np.nan])
        with self.assertRaisesRegex(ValueError, "Input 'center_freq' must be finite, positive and strictly ascending"):
            quadriga_lib.layout.generate_layout("38.901_UMa", 1, [2e9, 1e9])
        with self.assertRaises(TypeError):
            quadriga_lib.layout.generate_layout(1)
        with self.assertRaises(TypeError):  # n_ut is an unsigned integer
            quadriga_lib.layout.generate_layout("38.901_UMa", -1)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.generate_layout("38.901_UMa", 1.5)
        with self.assertRaisesRegex(TypeError, "Input 'per_cell' must be a bool or None"):
            quadriga_lib.layout.generate_layout("38.901_UMa", 1, per_cell=1)
        with self.assertRaisesRegex(ValueError, "Input 'options' must be a dict or None"):
            quadriga_lib.layout.generate_layout("38.901_InH_Open", 3, options=5)
        with self.assertRaisesRegex(ValueError, "Keys of 'options' must be str"):
            quadriga_lib.layout.generate_layout("38.901_InH_Open", 3, options={1: 2})
        with self.assertRaisesRegex(ValueError, "foo"):  # C++: unknown option
            quadriga_lib.layout.generate_layout("38.901_InH_Open", 3, options={"foo": 1})
        with self.assertRaisesRegex(ValueError, "no_such_preset"):  # C++
            quadriga_lib.layout.generate_layout("no_such_preset")


if __name__ == "__main__":
    unittest.main()
