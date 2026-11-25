# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.channel.crossfade_segments.
# Scope: argument and output handling, defaults, par, stack, error paths.
# The cross-fade rules are tested in the Catch2 suite (test_crossfade_segments.cpp).
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


def segment(amp, n_snap):
    # One path with amplitude amp per snapshot (rays form)
    return {"name": "seg",
            "tx_pos": np.array([[0.0], [0.0], [10.0]]),
            "rx_pos": np.tile([[0.0], [0.0], [1.5]], (1, n_snap)),
            "coeff": [np.full((1, 1, 1), amp + 0j) for _ in range(n_snap)],
            "delay": [np.full((1, 1, 1), 1e-7) for _ in range(n_snap)],
            "path_gain": [np.full(1, amp ** 2) for _ in range(n_snap)],
            "center_freq": np.array([3.5e9])}


class test_case(unittest.TestCase):

    def setUp(self):
        # Two segments of 4 snapshots with one path each (amplitude 1 and 2); the last 2 snapshots of segment 0 are the first 2 of segment 1
        self.seg = [segment(1, 4), segment(2, 4)]
        self.n_overlap = [0, 2]
        self.par = [{"tap": np.array([0], dtype=np.uint32)}, {"tap": np.array([0], dtype=np.uint32)}]  # both paths are LOS paths

    def amp(self, out):
        return np.array([np.abs(c).max() for c in out["coeff"]])

    def test_los_ramp(self):
        # LOS path with |h|^2 = (1 - w) |h_old|^2 + w |h_new|^2 in the ramp (w = 0.25, 0.75 for 2 ramp snapshots)
        out = quadriga_lib.channel.crossfade_segments(self.seg, self.n_overlap, overlap=1, par=self.par)
        self.assertIsInstance(out, dict)
        self.assertEqual(len(out["coeff"]), 6)  # 4 + (4 - 2) snapshots
        npt.assert_allclose(self.amp(out), [1, 1, np.sqrt(1.75), np.sqrt(3.25), 2, 2], atol=1e-12, rtol=0)
        self.assertEqual(out["rx_pos"].shape, (3, 6))
        self.assertEqual(out["name"], "seg")
        npt.assert_array_equal(out["center_freq"], [3.5e9])

    def test_optional_inputs(self):
        seg, n_overlap, par = self.seg, self.n_overlap, self.par
        out = quadriga_lib.channel.crossfade_segments(seg, n_overlap, par=par)  # overlap 0.5: one ramp snapshot, w = 0.5
        npt.assert_allclose(self.amp(out), [1, 1, 1, np.sqrt(2.5), 2, 2], atol=1e-12, rtol=0)
        self.assertTrue(same(quadriga_lib.channel.crossfade_segments(seg, n_overlap, overlap=0.5, par=par), out))
        # Without par all paths are NLOS and stay side by side in the ramp, with the same total power as the LOS rule
        outs = quadriga_lib.channel.crossfade_segments(seg, n_overlap, overlap=1, stack=True)
        self.assertEqual(outs["coeff"].shape, (1, 1, 2, 6))
        power = (np.abs(outs["coeff"][0, 0]) ** 2).sum(axis=0)
        npt.assert_allclose(power, [1, 1, 1.75, 3.25, 4, 4], atol=1e-12, rtol=0)
        out2 = quadriga_lib.channel.crossfade_segments(seg, n_overlap, overlap=1, par=None, stack=None)
        self.assertTrue(same(out2, quadriga_lib.channel.crossfade_segments(seg, n_overlap, overlap=1)))
        self.assertTrue(same(quadriga_lib.channel.crossfade_segments(seg, n_overlap, overlap=1, par=[]), quadriga_lib.channel.crossfade_segments(seg, n_overlap, overlap=1)))
        # Other types: uint8 n_overlap, tap indices as lists, a single par dict for one segment
        out3 = quadriga_lib.channel.crossfade_segments(seg, np.array(n_overlap, dtype=np.uint8), overlap=np.float32(1),
                                                       par=[{"tap": [0]}, {"tap": (0,), "n_ray_tap": [1]}])
        npt.assert_allclose(self.amp(out3), [1, 1, np.sqrt(1.75), np.sqrt(3.25), 2, 2], atol=1e-6, rtol=0)
        one = quadriga_lib.channel.crossfade_segments([seg[0]], [0], par={"tap": [0]})
        self.assertEqual(len(one["coeff"]), 4)
        outk = quadriga_lib.channel.crossfade_segments(segments=seg, n_overlap=n_overlap, overlap=1, par=self.par, stack=False)
        self.assertTrue(same(outk, quadriga_lib.channel.crossfade_segments(seg, n_overlap, overlap=1, par=self.par)))

    def test_path_type(self):
        # A single path of tap index 3 is a GR path for the tap rule (merged); path_type 2 makes it NLOS (side by side)
        seg, n_overlap = self.seg, self.n_overlap
        out = quadriga_lib.channel.crossfade_segments(seg, n_overlap, overlap=1, par=[{"tap": [3]}, {"tap": [3]}], stack=True)
        self.assertEqual(out["coeff"].shape[2], 1)
        par = [{"tap": [3], "path_type": [2]}, {"tap": [3], "path_type": np.array([2], dtype=np.uint32)}]
        out = quadriga_lib.channel.crossfade_segments(seg, n_overlap, overlap=1, par=par, stack=True)
        self.assertEqual(out["coeff"].shape[2], 2)
        out = quadriga_lib.channel.crossfade_segments(seg, n_overlap, overlap=1, par=[{"path_type": [0]}, {"path_type": [0]}], stack=True)
        self.assertEqual(out["coeff"].shape[2], 1)  # path_type alone: both LOS
        with self.assertRaisesRegex(ValueError, "path_type"):
            quadriga_lib.channel.crossfade_segments(seg, n_overlap, par=[{"tap": [0], "path_type": [5]}, {"tap": [0]}])

    def test_input_is_not_modified(self):
        seg = self.seg
        before = [dict(s, coeff=[c.copy() for c in s["coeff"]]) for s in seg]
        quadriga_lib.channel.crossfade_segments(seg, self.n_overlap, overlap=1, par=self.par)
        self.assertTrue(same(seg, before))
        self.assertEqual(set(self.par[0].keys()), {"tap"})

    def test_errors(self):
        seg, n_overlap, par = self.seg, self.n_overlap, self.par
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # overlap is keyword-only
            quadriga_lib.channel.crossfade_segments(seg, n_overlap, 1)
        with self.assertRaises(TypeError):
            quadriga_lib.channel.crossfade_segments(seg)
        with self.assertRaises(TypeError):
            quadriga_lib.channel.crossfade_segments(None, n_overlap)
        with self.assertRaisesRegex(ValueError, "Channel input must be a dict or a list of dicts"):
            quadriga_lib.channel.crossfade_segments(1, n_overlap)
        with self.assertRaisesRegex(ValueError, "Input 'par' must be a dict, a list of dicts, or None"):
            quadriga_lib.channel.crossfade_segments(seg, n_overlap, par=5)
        with self.assertRaisesRegex(ValueError, "Input 'par' has no key 'tap' or 'path_type'"):
            quadriga_lib.channel.crossfade_segments(seg, n_overlap, par=[{"tapp": [0]}, {"tapp": [0]}])
        with self.assertRaisesRegex(ValueError, "Inputs 'par' and 'segments' must have the same number of entries"):
            quadriga_lib.channel.crossfade_segments(seg, n_overlap, par=par[:1])
        with self.assertRaisesRegex(ValueError, "Input 'n_overlap'"):  # a negative count
            quadriga_lib.channel.crossfade_segments(seg, [0, -1])
        with self.assertRaisesRegex(TypeError, "Input 'stack' must be a bool or None"):
            quadriga_lib.channel.crossfade_segments(seg, n_overlap, stack=1)
        with self.assertRaisesRegex(ValueError, "n_overlap"):  # C++: one entry per segment
            quadriga_lib.channel.crossfade_segments(seg, [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "overlap"):  # C++
            quadriga_lib.channel.crossfade_segments(seg, n_overlap, overlap=2)


if __name__ == "__main__":
    unittest.main()
