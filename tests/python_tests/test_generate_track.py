# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.generate_track.
# Scope: argument and output handling, index base, defaults, seed, in/out split, error paths.
# The tracks are tested in the Catch2 suite (test_generate_track.cpp).
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
        # Terminal UT1 (one node) and UT2 (two sectors, one position)
        self.tmp = tempfile.TemporaryDirectory()
        fn_rx = os.path.join(self.tmp.name, "rx.csv")
        write_text(fn_rx, "name,sector,x,y,z,speed\nUT1,0,0,0,1.5,3\nUT2,0,100,50,1.5,2\nUT2,1,,,,\n")
        self.nodes = quadriga_lib.layout.csv_read("", fn_rx)[0]["rx"]
        npt.assert_array_equal(self.nodes["offset"], [0, 1, 2])

    def tearDown(self):
        self.tmp.cleanup()

    def test_linear_track(self):
        nodes = self.nodes
        copy = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in nodes.items()}
        # A track of 10 m in the direction 0.5 rad for node 1 (0-based); its sector shares the name and the track
        out = quadriga_lib.layout.generate_track(nodes, 1, "linear", length=10, speed=2, direction=0.5)
        npt.assert_array_equal(out["offset"], [0, 1, 3])  # 0-based
        self.assertEqual(out["offset"].dtype, np.int64)
        self.assertEqual(out["pos"].shape, (3, 5))
        end = [100 + 10 * np.cos(0.5), 50 + 10 * np.sin(0.5), 1.5]
        npt.assert_allclose(out["pos"][:, 1:], np.array([[100, 50, 1.5], end, [100, 50, 1.5], end]).T, atol=1e-12, rtol=0)
        npt.assert_array_equal(out["pos"][:, 0], nodes["pos"][:, 0])  # UT1 is not changed
        npt.assert_allclose(out["orientation"][2, 1:], 0.5, atol=1e-12, rtol=0)  # the orientation follows the motion
        npt.assert_array_equal(out["speed"][1:], 2)
        self.assertTrue(same(nodes, copy))  # the input stays unchanged

    def test_point_circular_street(self):
        nodes = self.nodes
        pt = quadriga_lib.layout.generate_track(nodes, 0, "point", speed=4, direction=1.0)  # explicit direction in rad
        self.assertEqual(pt["pos"].shape, (3, 3))
        self.assertEqual(pt["speed"][0], 4)
        npt.assert_allclose(pt["direction"][:, 0], [1.0, 0], atol=1e-14, rtol=0)
        ci = quadriga_lib.layout.generate_track(nodes, 0, "circular", length=2 * np.pi * 5, speed=1, direction=0)
        self.assertEqual(ci["pos"].shape, (3, 129 + 2))
        npt.assert_allclose(ci["pos"][:, 0], ci["pos"][:, 128], atol=1e-12, rtol=0)  # closed
        dist = np.hypot(ci["pos"][0, :129] + 5, ci["pos"][1, :129])  # the centre is at (-5, 0)
        npt.assert_allclose(dist, 5, atol=1e-12, rtol=0)
        s1 = quadriga_lib.layout.generate_track(nodes, 0, "street", length=200, speed=5, seed=3)
        s2 = quadriga_lib.layout.generate_track(nodes, 0, "street", length=200, speed=5, spacing=1, options={}, seed=3)
        npt.assert_array_equal(s1["pos"], s2["pos"])  # spacing 1 m is the default
        s3 = quadriga_lib.layout.generate_track(nodes, 0, "street", length=200, speed=5, seed=4)
        self.assertFalse(np.array_equal(s1["pos"], s3["pos"]))
        st = quadriga_lib.layout.generate_track(nodes, 0, "street", length=200, speed=5, direction=0.3,
                                                options={"turn_probability": 0}, seed=3)  # no turns: a straight route
        d = st["pos"][:2, :201] - st["pos"][:2, :1]
        npt.assert_allclose(d[0] * np.sin(0.3) - d[1] * np.cos(0.3), 0, atol=1e-9, rtol=0)
        npt.assert_allclose(np.hypot(d[0, -1], d[1, -1]), 200, atol=1e-9, rtol=0)

    def test_indices_and_types(self):
        nodes = self.nodes
        two = quadriga_lib.layout.generate_track(nodes, np.array([0, 2], dtype=np.uint8), "linear", length=10, direction=[0, 0.5])
        npt.assert_array_equal(two["offset"], [0, 2, 4])  # node 2 = sector of UT2: both UT2 nodes
        npt.assert_allclose(two["pos"][:, 1], [10, 0, 1.5], atol=1e-12, rtol=0)
        npt.assert_array_equal(two["speed"][:2], [3, 3])  # speed = NaN: the speed of the first row
        out = quadriga_lib.layout.generate_track(nodes, [1], "linear", length=10, speed=2, direction=np.array([0.5], dtype=np.float32))
        self.assertEqual(out["pos"].shape, (3, 5))
        out = quadriga_lib.layout.generate_track(nodes_in=nodes, i_node=1, type="linear", length=10, speed=2, direction=0.5)
        self.assertEqual(out["pos"].shape, (3, 5))

    def test_errors(self):
        nodes = self.nodes
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # length is keyword-only
            quadriga_lib.layout.generate_track(nodes, 0, "linear", 10)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.generate_track(nodes, None, "linear")
        with self.assertRaises(TypeError):
            quadriga_lib.layout.generate_track(1, 0, "linear")
        with self.assertRaisesRegex(ValueError, "Input 'i_node'"):  # negative index
            quadriga_lib.layout.generate_track(nodes, -1, "linear")
        with self.assertRaisesRegex(ValueError, "Unknown key 'foo' in 'nodes_in'"):
            quadriga_lib.layout.generate_track(dict(nodes, foo=1), 0, "linear", speed=1)
        with self.assertRaisesRegex(ValueError, "i_node"):  # C++: out of range
            quadriga_lib.layout.generate_track(nodes, 3, "linear", length=10, speed=1, direction=0)
        with self.assertRaisesRegex(ValueError, "spiral"):  # C++
            quadriga_lib.layout.generate_track(nodes, 0, "spiral", length=10, speed=1, direction=0)
        with self.assertRaisesRegex(ValueError, "foo"):  # C++
            quadriga_lib.layout.generate_track(nodes, 0, "linear", length=10, speed=1, direction=0, options={"foo": 1})


if __name__ == "__main__":
    unittest.main()
