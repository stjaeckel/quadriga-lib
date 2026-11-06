# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.rays_from_paths.
# Scope: argument and output handling, ray dict keys, codes, error paths.
# Physics validation lives in the Catch2 suite (test_gbsm_rays_from_paths.cpp).

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

gbsm = quadriga_lib.gbsm
KEYS = ["method", "tx_pos", "rx_pos", "center_frequency", "delay", "power", "aod", "eod", "aoa", "eoa", "M",
        "path_type", "tap", "gr_eps", "fbs_pos", "lbs_pos", "fallback"]
C0 = 299792458.0


class test_case(unittest.TestCase):

    def setUp(self):
        # LOS, GR and NLOS path of one link at two frequencies
        tx, rx = np.array([0.0, 0.0, 10.0]), np.array([40.0, 30.0, 1.5])
        gr = np.append(tx[:2] + (rx[:2] - tx[:2]) * tx[2] / (tx[2] + rx[2]), 0.0)  # specular point
        mid = (tx + rx) / 2
        self.fb = np.stack([mid, gr, [10.0, 20.0, 5.0]], 1)
        self.lb = np.stack([mid, gr, [30.0, 35.0, 3.0]], 1)
        self.d = np.linalg.norm(rx - tx)
        f3, l3 = self.fb[:, 2], self.lb[:, 2]
        self.pl = np.array([self.d, np.linalg.norm(gr - tx) + np.linalg.norm(rx - gr),
                            np.linalg.norm(f3 - tx) + np.linalg.norm(l3 - f3) + np.linalg.norm(rx - l3)])
        self.pg = np.array([[1e-6, 2e-6], [3e-7, 4e-7], [5e-8, 6e-8]])
        self.M = np.reshape(np.arange(1.0, 49.0), (8, 3, 2), order="F") / 48
        self.fc, self.tx, self.rx = np.array([3.5e9, 28e9]), tx, rx
        self.args = (self.fc, tx, rx, self.fb, self.lb, self.pg, self.pl, self.M)
        self.pt = np.array([0, 1, 2])
        self.rays = gbsm.rays_from_paths(*self.args, "quadriga", self.pt)

    def test_basic_call(self):
        r = self.rays
        self.assertIsInstance(r, dict)
        self.assertEqual(list(r.keys()), KEYS)
        self.assertEqual(r["method"], "quadriga")
        npt.assert_array_equal(r["tx_pos"], self.tx)
        npt.assert_array_equal(r["center_frequency"], self.fc)
        npt.assert_allclose(r["delay"], (self.pl - self.d) / C0, atol=1e-18, rtol=0)
        npt.assert_array_equal(r["power"], self.pg)
        npt.assert_array_equal(r["M"], self.M)
        npt.assert_array_equal(r["fbs_pos"], self.fb)
        npt.assert_array_equal(r["lbs_pos"], self.lb)
        for k in ["path_type", "tap", "fallback"]:
            self.assertEqual(r[k].dtype, np.uint64)
        npt.assert_array_equal(r["path_type"], self.pt)  # codes
        npt.assert_array_equal(r["tap"], [0, 3, 6])
        self.assertEqual(r["fallback"].size, 0)
        self.assertEqual(r["gr_eps"].size, 0)
        self.assertEqual(r["gr_eps"].dtype, np.complex128)
        v = self.fb[:, 2] - self.tx
        npt.assert_allclose([r["aod"][2], r["eod"][2]], [np.arctan2(v[1], v[0]), np.arctan2(v[2], np.hypot(v[0], v[1]))],
                            atol=1e-14, rtol=0)
        # Round trip: the paths of "drift" at the same positions are the input paths
        out = gbsm.ssf_paths(r, np.zeros(2), "drift")
        npt.assert_allclose(out[0], self.pg, atol=1e-20, rtol=0)
        npt.assert_allclose(out[1], self.pl, atol=1e-9, rtol=0)
        npt.assert_allclose(out[2], self.M, atol=1e-14, rtol=0)

    def test_keywords_and_optional_inputs(self):
        r = gbsm.rays_from_paths(center_freq=self.fc, tx_pos=self.tx, rx_pos=self.rx, fbs_pos=self.fb,
                                 lbs_pos=self.lb, path_gain=self.pg, path_length=self.pl, M=self.M, method="quadriga",
                                 path_type=self.pt, tap=None, gr_eps=None)
        for k in KEYS[1:]:
            npt.assert_array_equal(r[k], self.rays[k])
        i = [0, 2]
        r2 = gbsm.rays_from_paths(self.fc, self.tx, self.rx, self.fb[:, i], self.lb[:, i], self.pg[i], self.pl[i],
                                  self.M[:, i])
        npt.assert_array_equal(r2["path_type"], [0, 2])  # LOS by geometry
        npt.assert_array_equal(r2["tap"], [0, 6])
        r3 = gbsm.rays_from_paths(*self.args, "38.901", [0.0, 1.0, 2.0], np.array([5, 3, 0], dtype=np.int32),
                                  np.array([5 + 0.1j, 7 + 0.2j]))
        self.assertEqual(r3["method"], "38.901")
        npt.assert_array_equal(r3["tap"], [5, 3, 0])  # codes
        npt.assert_array_equal(r3["gr_eps"], [5 + 0.1j, 7 + 0.2j])
        self.assertEqual(r3["gr_eps"].dtype, np.complex128)
        r4 = gbsm.rays_from_paths(*self.args, path_type=self.pt, gr_eps=[5.0, 7.0])  # real: imaginary part 0
        npt.assert_array_equal(r4["gr_eps"], [5.0, 7.0])

    def test_array_handling(self):
        args = (list(self.fc), self.tx.astype(np.float32), self.rx, np.asfortranarray(self.fb),
                np.ascontiguousarray(self.lb), self.pg[::1], self.pl, np.ascontiguousarray(self.M))
        r = gbsm.rays_from_paths(*args, "quadriga", self.pt)
        npt.assert_array_equal(r["delay"], self.rays["delay"])
        npt.assert_array_equal(r["M"], self.M)
        fb0 = self.fb.copy()
        gbsm.rays_from_paths(*self.args, "quadriga", self.pt)
        npt.assert_array_equal(self.fb, fb0)
        r0 = gbsm.rays_from_paths(self.fc, self.tx, self.rx, np.zeros((3, 0)), np.zeros((3, 0)), np.zeros((0, 2)),
                                  np.zeros(0), np.zeros((8, 0, 2)))
        self.assertEqual(r0["delay"].shape, (0,))  # no paths
        self.assertEqual(r0["M"].shape, (8, 0, 2))

    def test_errors(self):
        with self.assertRaises(ValueError) as ctx:
            i = [0, 0]
            gbsm.rays_from_paths(self.fc, self.tx, self.rx, self.fb[:, i], self.lb[:, i], self.pg[i],
                                 np.array([self.d, self.d]), self.M[:, i])
        self.assertIn("More than one path matches the LOS geometry; pass 'path_type'.", str(ctx.exception))
        lz = self.lb.copy()
        lz[2, 1] = 0.01
        with self.assertRaises(ValueError) as ctx:
            gbsm.rays_from_paths(self.fc, self.tx, self.rx, lz, lz, self.pg, self.pl, self.M, path_type=self.pt)
        self.assertIn("The interaction point of the GR path must be on the ground", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.rays_from_paths(*self.args, tap=[-1, 0, 1])
        self.assertIn("Entries in 'tap' must be non-negative integers", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            gbsm.rays_from_paths(self.fc * 0, *self.args[1:])
        self.assertIn("Input 'center_frequency' must be finite and > 0.", str(ctx.exception))
        with self.assertRaises(TypeError):
            gbsm.rays_from_paths(*self.args[:7])
        with self.assertRaises(TypeError):
            gbsm.rays_from_paths(*self.args[:7], None)
        with self.assertRaises(TypeError):
            gbsm.rays_from_paths(*self.args, foo=1)


if __name__ == "__main__":
    unittest.main()
