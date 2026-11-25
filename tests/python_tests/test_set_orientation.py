# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.set_orientation.
# Scope: argument and output handling, index base, defaults, in/out split, error paths.
# The orientation rules are tested in the Catch2 suite (test_generate_track.cpp).
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
        # A track T1 along +x and then +y (3 rows) and a point drop P with heading 30 degrees
        self.tmp = tempfile.TemporaryDirectory()
        fn_rx = os.path.join(self.tmp.name, "rx.csv")
        write_text(fn_rx, "name,x,y,z,heading\nT1,0,0,0,0\nT1,10,0,0,\nT1,10,10,0,\nP,0,0,0,30\n")
        self.nodes = quadriga_lib.layout.csv_read("", fn_rx)[0]["rx"]
        self.h30 = 30 * np.pi / 180
        self.follow = [0, np.pi / 2, np.pi / 2, self.h30]  # heading of the rows when following the motion

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_call(self):
        nodes = self.nodes
        copy = nodes["orientation"].copy()
        out = quadriga_lib.layout.set_orientation(nodes)  # the heading follows the motion, the last step is repeated
        npt.assert_allclose(out["orientation"], [[0] * 4, [0] * 4, self.follow], atol=1e-14, rtol=0)
        npt.assert_array_equal(nodes["orientation"], copy)  # the input stays unchanged
        npt.assert_array_equal(out["pos"], nodes["pos"])
        npt.assert_array_equal(out["offset"], [0, 3])  # 0-based
        self.assertEqual(out["orientation"].dtype, np.float64)

    def test_optional_inputs(self):
        nodes, h30 = self.nodes, self.h30
        out = quadriga_lib.layout.set_orientation(nodes, orientation=[0, 0, 0.5])  # an offset
        npt.assert_allclose(out["orientation"][2], np.array(self.follow) + 0.5, atol=1e-14, rtol=0)
        out = quadriga_lib.layout.set_orientation(nodes, i_node=0)  # T1 only
        npt.assert_allclose(out["orientation"][2], self.follow, atol=1e-14, rtol=0)
        out = quadriga_lib.layout.set_orientation(nodes, orientation=[0.1, 0.2, 0.3], i_node=1, force=True)  # P: the given value
        npt.assert_array_equal(out["orientation"][:, 3], [0.1, 0.2, 0.3])
        npt.assert_array_equal(out["orientation"][:, :3], nodes["orientation"][:, :3])
        out = quadriga_lib.layout.set_orientation(nodes, orientation=[0.1, 0.2, 0.3], force=True)
        npt.assert_array_equal(out["orientation"], np.tile([[0.1], [0.2], [0.3]], (1, 4)))
        out = quadriga_lib.layout.set_orientation(nodes, orientation=None, i_node=None, force=None)  # None = the defaults
        npt.assert_allclose(out["orientation"][2], self.follow, atol=1e-14, rtol=0)
        out = quadriga_lib.layout.set_orientation(nodes, orientation=np.array([0, 0, 0.5], dtype=np.float32),
                                                  i_node=np.array([0, 1], dtype=np.uint8))
        npt.assert_allclose(out["orientation"][2], np.array(self.follow) + 0.5, atol=1e-6, rtol=0)
        out = quadriga_lib.layout.set_orientation(nodes_in=nodes, orientation=(0, 0, 0.5), i_node=[0, 1], force=False)
        npt.assert_allclose(out["orientation"][2], np.array(self.follow) + 0.5, atol=1e-14, rtol=0)

    def test_errors(self):
        nodes = self.nodes
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # orientation is keyword-only
            quadriga_lib.layout.set_orientation(nodes, [0, 0, 0.5])
        with self.assertRaises(TypeError):
            quadriga_lib.layout.set_orientation(1)
        with self.assertRaisesRegex(TypeError, "Input 'force' must be a bool or None"):
            quadriga_lib.layout.set_orientation(nodes, force=1)
        with self.assertRaisesRegex(ValueError, "Input 'i_node'"):  # negative index
            quadriga_lib.layout.set_orientation(nodes, i_node=-1)
        with self.assertRaisesRegex(ValueError, "Unknown key 'foo' in 'nodes_in'"):
            quadriga_lib.layout.set_orientation(dict(nodes, foo=1))
        with self.assertRaisesRegex(ValueError, "i_node"):  # C++: out of range
            quadriga_lib.layout.set_orientation(nodes, i_node=2)
        with self.assertRaisesRegex(ValueError, "orientation"):  # C++: 3 values
            quadriga_lib.layout.set_orientation(nodes, orientation=[0, 1])


if __name__ == "__main__":
    unittest.main()
