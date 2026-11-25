# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.attach_terminals.
# Scope: argument and output handling, index base, defaults, error paths.
# The rules are tested in the Catch2 suite (test_attach_terminals.cpp).
# Physics validation lives in the Catch2 suite.

import sys
import os
import copy
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


def write_text(fn, text):
    with open(fn, "w") as f:
        f.write(text)


def same(a, b):
    # Deep equality of dicts, lists and arrays; NaN equals NaN, dtypes and shapes must match
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a.keys()) == list(b.keys()) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return (isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and
                np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return a == b


class test_case(unittest.TestCase):

    def setUp(self):
        # Site A (100 m from U1) with two sectors, site B (100 m from U2); UMa
        self.tmp = tempfile.TemporaryDirectory()
        fn_tx, fn_rx = os.path.join(self.tmp.name, "tx.csv"), os.path.join(self.tmp.name, "rx.csv")
        write_text(fn_tx, "name,sector,x,y,z,scenario,heading\nA,0,0,0,25,UMa,0\nA,1,,,,,180\nB,0,1000,0,25,UMa,0\n")
        write_text(fn_rx, "name,x,y,z,indoor\nU1,100,0,1.5,0\nU2,900,0,1.5,0\n")
        self.L = quadriga_lib.layout.csv_read(fn_tx, fn_rx, center_freq=[2e9, 3.5e9])[0]
        self.L2 = quadriga_lib.layout.init_link_states(self.L, seed=5)

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_call(self):
        out = quadriga_lib.layout.attach_terminals(self.L2)
        self.assertIsInstance(out, tuple)
        node, site, gain_dB = out
        npt.assert_array_equal(node, [0, 2])  # 0-based; a tie of the sectors of A goes to the lowest node
        npt.assert_array_equal(site, [0, 1])
        self.assertEqual((node.dtype, site.dtype, node.shape), (np.int64, np.int64, (2,)))
        self.assertEqual(gain_dB.shape, (2, 3))  # (n_rx_terminal, n_tx_node)
        npt.assert_array_equal(gain_dB[:, 0], gain_dB[:, 1])  # same antenna at the same position
        self.assertTrue(gain_dB[0, 0] > gain_dB[0, 2] and gain_dB[1, 2] > gain_dB[1, 0])  # the nearer site has the higher gain

    def test_optional_inputs(self):
        L2 = self.L2
        node, site, gain_dB = quadriga_lib.layout.attach_terminals(L2)
        rays = quadriga_lib.layout.generate_rays(L2)
        node2, site2, rsrp = quadriga_lib.layout.attach_terminals(L2, rule="rsrp", rays=rays)
        npt.assert_array_equal(node2, node)
        npt.assert_array_equal(site2, site)
        self.assertEqual(rsrp.shape, (2, 3))
        self.assertFalse(np.array_equal(rsrp, gain_dB))
        node3, _, gain3 = quadriga_lib.layout.attach_terminals(L2, i_freq=1, tx_port=1)  # another band and port, 0-based
        npt.assert_array_equal(node3, node)
        self.assertFalse(np.array_equal(gain3, gain_dB))
        _, _, gain4 = quadriga_lib.layout.attach_terminals(L2, rule="large_scale", rays=None, i_freq=0, tx_port=0)  # the defaults
        npt.assert_array_equal(gain4, gain_dB)
        _, _, gain5 = quadriga_lib.layout.attach_terminals(L2, rule="rsrp", rays=rays, i_freq=1)
        self.assertFalse(np.array_equal(gain5, rsrp))
        node6, _, _ = quadriga_lib.layout.attach_terminals(L2, i_freq=np.int8(1), tx_port=np.uint8(1))
        npt.assert_array_equal(node6, node)
        Lp = dict(L2, pairing=np.array([[1], [0]]))  # the pairing is ignored
        npt.assert_array_equal(quadriga_lib.layout.attach_terminals(Lp)[0], node)

    def test_input_is_not_modified(self):
        L2 = self.L2
        rays = quadriga_lib.layout.generate_rays(L2)
        before_layout, before_rays = copy.deepcopy(L2), copy.deepcopy(rays)
        quadriga_lib.layout.attach_terminals(L2, rule="rsrp", rays=rays)
        quadriga_lib.layout.attach_terminals(L2)
        self.assertTrue(same(L2, before_layout))
        self.assertTrue(same(rays, before_rays))

    def test_errors(self):
        L2 = self.L2
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # rule is keyword-only
            quadriga_lib.layout.attach_terminals(L2, "rsrp")
        with self.assertRaises(TypeError):
            quadriga_lib.layout.attach_terminals(1)
        with self.assertRaises(TypeError):  # an index is an unsigned integer
            quadriga_lib.layout.attach_terminals(L2, i_freq=-1)
        with self.assertRaisesRegex(ValueError, "Input 'rays' must be a list"):
            quadriga_lib.layout.attach_terminals(L2, rays=5)
        with self.assertRaisesRegex(ValueError, "i_freq"):  # C++: out of range
            quadriga_lib.layout.attach_terminals(L2, i_freq=2)
        with self.assertRaisesRegex(ValueError, "tx_port"):  # C++
            quadriga_lib.layout.attach_terminals(L2, tx_port=3)
        with self.assertRaisesRegex(ValueError, "rule"):  # C++
            quadriga_lib.layout.attach_terminals(L2, rule="foo")
        with self.assertRaisesRegex(ValueError, "rays"):  # C++: rsrp needs the rays of all links
            quadriga_lib.layout.attach_terminals(L2, rule="rsrp")


if __name__ == "__main__":
    unittest.main()
