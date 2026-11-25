# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.sample_track.
# Scope: argument and output handling, index base, empty times, error paths.
# The sampling rules are tested in the Catch2 suite (test_generate_track.cpp).
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
        # A track A (3 rows, 1 m/s), a point drop B (heading 90 degrees, 2 m/s) and a track C without time and speed
        self.tmp = tempfile.TemporaryDirectory()
        fn_rx = os.path.join(self.tmp.name, "rx.csv")
        write_text(fn_rx, "name,x,y,z,heading,speed\nA,0,0,0,0,1\nA,10,0,0,,\nA,10,10,0,,\nB,5,5,1.5,90,2\nC,0,0,0,,\nC,1,0,0,,\n")
        self.nodes = quadriga_lib.layout.csv_read("", fn_rx)[0]["rx"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_track(self):
        # Speed 1 m/s: the rows are at 0, 10 and 20 s
        out = quadriga_lib.layout.sample_track(self.nodes, 0, [0, 5, 10, 15, 20])
        self.assertIsInstance(out, tuple)
        pos, ori, i_row = out
        npt.assert_allclose(pos, [[0, 5, 10, 10, 10], [0, 0, 0, 5, 10], [0, 0, 0, 0, 0]], atol=1e-12, rtol=0)
        npt.assert_array_equal(ori, np.zeros((3, 5)))
        npt.assert_array_equal(i_row, [0, 0, 1, 1, 2])  # 0-based row at or before the snapshot
        self.assertEqual((pos.dtype, ori.dtype, i_row.dtype), (np.float64, np.float64, np.int64))
        self.assertEqual(i_row.shape, (5,))

    def test_point_drop(self):
        # A point drop moves along its heading with its speed (90 deg = +y)
        pos, ori, i_row = quadriga_lib.layout.sample_track(self.nodes, 1, np.array([0, 1, 2]))
        npt.assert_allclose(pos, [[5, 5, 5], [5, 7, 9], [1.5, 1.5, 1.5]], atol=1e-12, rtol=0)
        npt.assert_allclose(ori, [[0, 0, 0], [0, 0, 0], [np.pi / 2] * 3], atol=1e-14, rtol=0)  # in rad
        npt.assert_array_equal(i_row, [3, 3, 3])  # index into the arrays of the nodes, not into the rows of the node

    def test_empty_times_and_types(self):
        pos, _, i_row = quadriga_lib.layout.sample_track(self.nodes, 0, [])  # one snapshot per row
        npt.assert_array_equal(pos, [[0, 10, 10], [0, 0, 10], [0, 0, 0]])
        npt.assert_array_equal(i_row, [0, 1, 2])
        pos, _, i_row = quadriga_lib.layout.sample_track(nodes=self.nodes, i_node=np.uint8(0),
                                                         times=np.array([0, 5, 10, 15, 20], dtype=np.float32))
        npt.assert_array_equal(pos[0], [0, 5, 10, 10, 10])
        npt.assert_array_equal(i_row, [0, 0, 1, 1, 2])
        pos, _, _ = quadriga_lib.layout.sample_track(self.nodes, 0, (0, 5))  # a tuple
        npt.assert_array_equal(pos[0], [0, 5])

    def test_input_is_not_modified(self):
        nodes = self.nodes
        copy = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in nodes.items()}
        quadriga_lib.layout.sample_track(nodes, 0, [0, 5])
        self.assertTrue(same(nodes, copy))

    def test_errors(self):
        nodes = self.nodes
        with self.assertRaises(TypeError):  # times is mandatory
            quadriga_lib.layout.sample_track(nodes, 0)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.sample_track(nodes, 0, None)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.sample_track(1, 0, [])
        with self.assertRaises(TypeError):  # a node index is an unsigned integer
            quadriga_lib.layout.sample_track(nodes, -1, [])
        with self.assertRaisesRegex(ValueError, "Unknown key 'foo' in 'nodes'"):
            quadriga_lib.layout.sample_track(dict(nodes, foo=1), 0, [])
        with self.assertRaisesRegex(ValueError, "i_node"):  # C++: out of range
            quadriga_lib.layout.sample_track(nodes, 4, [])
        with self.assertRaisesRegex(ValueError, "time"):  # C++: no time and no speed
            quadriga_lib.layout.sample_track(nodes, 2, [0, 1])
        with self.assertRaisesRegex(ValueError, "times"):  # C++
            quadriga_lib.layout.sample_track(nodes, 0, [0, np.nan])


if __name__ == "__main__":
    unittest.main()
