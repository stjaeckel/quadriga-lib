# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.path_loss.
# Scope: argument and output handling, dict inputs, defaults, error paths.
# Physics validation lives in the Catch2 suite (test_path_loss.cpp).

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

config_dir = os.path.join(current_dir, "../../data/config")


class test_case(unittest.TestCase):

    def setUp(self):
        # A hand-made logdist model, 3 links with d3D = 100, 1000 and 50 m, 2 frequencies
        self.plpar = {"model": "logdist", "A": 20.0, "B": 30.0, "C": 20.0}
        self.scenpar = {"SF_sigma": 3.0}
        self.tx_pos = np.array([[0.0, 0.0, 10.0]])
        self.rx_pos = np.array([[100.0, 0.0, 10.0], [0.0, 1000.0, 10.0], [30.0, 40.0, 10.0]])
        self.fc = np.array([1e9, 10e9])
        self.ref = 20 * np.log10([[100.0], [1000.0], [50.0]]) + 30 + 20 * np.log10(self.fc / 1e9)
        self.pl, self.sf = quadriga_lib.gbsm.path_loss(self.plpar, self.scenpar, self.fc, self.tx_pos, self.rx_pos)

        # A v19 file read with conf_file_read: dual_slope (LOS) and nlos (NLOS) with the random hE
        out = quadriga_lib.gbsm.conf_file_read(os.path.join(config_dir, "3GPP_38.901_v19_UMa_LOS.conf"))
        self.sp_los, self.pl_los = out[0], out[1]
        out = quadriga_lib.gbsm.conf_file_read(os.path.join(config_dir, "3GPP_38.901_v19_UMa_NLOS.conf"))
        self.sp_nlos, self.pl_nlos = out[0], out[1]
        self.tx_u = np.array([[0.0, 0.0, 25.0]])
        self.rx_u = np.array([[300.0, 0.0, 20.0]])

    def test_basic_call(self):
        out = quadriga_lib.gbsm.path_loss(self.plpar, self.scenpar, self.fc, self.tx_pos, self.rx_pos)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        pl, sf = out
        for x in (pl, sf):
            self.assertIsInstance(x, np.ndarray)
            self.assertEqual(x.shape, (3, 2))
            self.assertEqual(x.dtype, np.float64)
        self.assertAlmostEqual(pl[0, 0], 70.0, delta=1e-12)  # 100 m, 1 GHz
        npt.assert_allclose(pl, self.ref, atol=1e-12, rtol=0)
        npt.assert_array_equal(sf, np.full((3, 2), 3.0))  # SF_sigma of scenpar

    def test_keyword_arguments(self):
        pl, sf = quadriga_lib.gbsm.path_loss(plpar=self.plpar, scenpar=self.scenpar, fc=self.fc, tx_pos=self.tx_pos,
                                             rx_pos=self.rx_pos, d2d_in=None, rnd_he=None)
        npt.assert_array_equal(pl, self.pl)
        npt.assert_array_equal(sf, self.sf)

    def test_dict_inputs(self):
        f = quadriga_lib.gbsm.path_loss
        plpar_sf = dict(self.plpar, SF=5.0)  # sf from the model
        pl2, sf2 = f(plpar_sf, self.scenpar, self.fc, self.tx_pos, self.rx_pos)
        npt.assert_array_equal(pl2, self.pl)
        npt.assert_array_equal(sf2, np.full((3, 2), 5.0))
        _, sf3 = f(self.plpar, None, self.fc, self.tx_pos, self.rx_pos)  # scenpar None: default SF_sigma
        npt.assert_array_equal(sf3, np.full((3, 2), 8.0))
        npt.assert_array_equal(f(None, self.scenpar, self.fc, self.tx_pos, self.rx_pos)[0], np.zeros((3, 2)))
        npt.assert_array_equal(f({}, self.scenpar, self.fc, self.tx_pos, self.rx_pos)[0], np.zeros((3, 2)))
        npt.assert_array_equal(f({"A": 20.0}, self.scenpar, self.fc, self.tx_pos, self.rx_pos)[0], np.zeros((3, 2)))
        plpar_t = {"model": "logdist", "A": 20, "B": np.float32(30), "C": np.array([20])}  # int, float32, array
        npt.assert_array_equal(f(plpar_t, {"SF_sigma": 3}, self.fc, self.tx_pos, self.rx_pos)[0], self.pl)

    def test_conf_file(self):
        f = quadriga_lib.gbsm.path_loss
        pa, sfa = f(self.pl_los, self.sp_los, 1e9, self.tx_u, self.rx_u)
        self.assertEqual(pa.shape, (1, 1))
        self.assertTrue(80 < pa[0, 0] < 120)
        self.assertEqual(sfa[0, 0], 4.0)  # SF_sigma of the file
        pn = f(self.pl_nlos, self.sp_nlos, 1e9, self.tx_u, self.rx_u)[0]
        self.assertTrue(pn[0, 0] >= pa[0, 0])  # NLOS >= LOS term
        pr = f(self.pl_nlos, self.sp_nlos, 1e9, self.rx_u, self.tx_u)[0]
        self.assertGreater(abs(pr[0, 0] - pn[0, 0]), 0.1)  # Tx and Rx differ

    def test_d2d_in(self):
        f = quadriga_lib.gbsm.path_loss
        args = (self.plpar, self.scenpar, self.fc, self.tx_pos, self.rx_pos)
        npt.assert_array_equal(f(*args, None)[0], self.pl)
        npt.assert_array_equal(f(*args, np.zeros(0))[0], self.pl)  # empty = geometric distance
        npt.assert_array_equal(f(*args, np.zeros(3))[0], self.pl)  # 0 = outdoor link
        pl_in = f(*args, [0.0, 20.0, 0.0])[0]  # legacy outdoor distance
        npt.assert_array_equal(pl_in[[0, 2]], self.pl[[0, 2]])
        npt.assert_allclose(pl_in[1], 20 * np.log10(980.0) + 30 + 20 * np.log10(self.fc / 1e9), atol=1e-12, rtol=0)
        npt.assert_array_equal(f(*args, np.array([0, 20, 0], dtype=np.int32))[0], pl_in)
        npt.assert_array_equal(f(*args, np.array([0, 20, 0], dtype=np.float32))[0], pl_in)

    def test_rnd_he(self):
        # UMa LOS: hE = 1 m for u < p1, hE = 18 m for u near 1 at hUT = 20 m
        f = quadriga_lib.gbsm.path_loss
        args = (self.pl_los, self.sp_los, 1e9, self.tx_u, self.rx_u)
        pa = f(*args)[0]
        npt.assert_array_equal(f(*args, None, None)[0], pa)
        npt.assert_allclose(f(*args, rnd_he=[0.0])[0], pa, atol=1e-12, rtol=0)
        pr = f(*args, rnd_he=[0.999])[0]
        self.assertGreater(pr[0, 0] - pa[0, 0], 1.0)  # beyond the breakpoint
        npt.assert_array_equal(f(*args, [0.0], [0.999])[0], pr)
        npt.assert_array_equal(f(*args, rnd_he=0.999)[0], pr)  # scalar for one link

    def test_array_layout_and_dtype(self):
        f = quadriga_lib.gbsm.path_loss
        args = (self.plpar, self.scenpar, self.fc, self.tx_pos)
        npt.assert_array_equal(f(*args, np.asfortranarray(self.rx_pos))[0], self.pl)
        npt.assert_array_equal(f(*args, np.ascontiguousarray(self.rx_pos))[0], self.pl)
        big = np.zeros((6, 6))
        big[::2, ::2] = self.rx_pos
        npt.assert_array_equal(f(*args, big[::2, ::2])[0], self.pl)
        pl_t = f(self.plpar, self.scenpar, self.fc.astype(np.float32), self.tx_pos.astype(np.float32),
                 self.rx_pos.astype(np.float32))[0]
        npt.assert_allclose(pl_t, self.pl, atol=1e-12, rtol=0)  # values exact in float32
        npt.assert_array_equal(f(self.plpar, self.scenpar, self.fc.tolist(), self.tx_pos, self.rx_pos)[0], self.pl)
        npt.assert_array_equal(f(self.plpar, self.scenpar, self.fc, np.tile(self.tx_pos, (3, 1)), self.rx_pos)[0],
                               self.pl)  # (n_link, 3)

    def test_zero_links(self):
        pl, sf = quadriga_lib.gbsm.path_loss(self.plpar, self.scenpar, self.fc, self.tx_pos, np.zeros((0, 3)))
        self.assertEqual(pl.shape, (0, 2))
        self.assertEqual(sf.shape, (0, 2))

    def test_inputs_not_modified(self):
        rx, fc, pp = self.rx_pos.copy(), self.fc.copy(), dict(self.plpar)
        d2d = np.array([0.0, 20.0, 0.0])
        quadriga_lib.gbsm.path_loss(self.plpar, self.scenpar, self.fc, self.tx_pos, self.rx_pos, d2d)
        npt.assert_array_equal(self.rx_pos, rx)
        npt.assert_array_equal(self.fc, fc)
        npt.assert_array_equal(d2d, [0.0, 20.0, 0.0])
        self.assertEqual(self.plpar, pp)

    def test_errors(self):
        f = quadriga_lib.gbsm.path_loss
        args = (self.scenpar, self.fc, self.tx_pos, self.rx_pos)
        with self.assertRaises(ValueError) as ctx:  # C++
            f({"model": "foo"}, *args)
        self.assertIn("Path-loss model 'foo' is not supported.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f({"model": "logdist", "A": 20.0}, *args)
        self.assertIn("Path-loss model 'logdist' requires the parameter 'B'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.pl_los, self.sp_los, 1e9, self.tx_u, self.rx_u, None, [1.5])
        self.assertIn("Input 'rnd_he' must be in the range from 0 to 1.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.plpar, *args, [0.0, 1.0])
        self.assertIn("Number of elements in 'd2d_in' does not match number of rows in 'rx_pos'.",
                      str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f(self.plpar, {"Foo": 1.0}, self.fc, self.tx_pos, self.rx_pos)
        self.assertIn("Unknown key 'Foo' in 'scenpar'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # plpar helper
            f(5, *args)
        self.assertIn("Input 'plpar' must be a dict or None.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # plpar helper
            f({"model": 3}, *args)
        self.assertIn("Entry 'model' of 'plpar' must be a str.", str(ctx.exception))
        with self.assertRaises(TypeError):
            f(self.plpar, self.scenpar, self.fc, self.tx_pos)
        with self.assertRaises(TypeError):
            f(self.plpar, *args, foo=1)


if __name__ == "__main__":
    unittest.main()
