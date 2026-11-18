# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.channel.irs.
# Scope: argument and output handling, defaults, flags, error paths.
# Physics validation lives in the Catch2 suite (test_get_channels_irs.cpp).

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

C0 = 299792458.0
MANDATORY_KEYS = ("e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im", "azimuth_grid", "elevation_grid")


def irs_two_ports():
    # Two IRS elements, two codebook entries with different weights
    ant = quadriga_lib.arrayant.generate("omni")
    out = {k: (np.repeat(v, 2, axis=2) if k.startswith("e_") else v) for k, v in ant.items()}
    out["element_pos"] = np.array([[0.0, 0.0], [0.0, 0.1], [0.0, 0.0]])
    out["coupling_re"] = np.array([[1.0, 0.3], [0.0, 0.4]])
    out["coupling_im"] = np.zeros((2, 2))
    return out


def distinct_args():
    # Different antennas, orientations, FBS / LBS, path lengths and polarization per segment, so that a swapped
    # argument changes the result
    tx, rx = quadriga_lib.arrayant.generate("dipole"), quadriga_lib.arrayant.generate("xpol")
    geo = [np.array([0.0, 0.0, 1.0]), np.array([0.1, 0.2, 0.3]), np.array([40.0, 0.0, 1.5]),
           np.array([-0.2, 0.1, 2.0]), np.array([20.0, 1.0, 3.0]), np.array([0.3, -0.1, 1.2])]
    M1, M2 = np.zeros((8, 2)), np.zeros((8, 2))
    M1[0], M1[6] = 1.0, -1.0
    M2[0], M2[2], M2[5], M2[6] = 0.8, 0.3, -0.2, 0.9
    seg1 = [np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 11.0]]).T, np.array([[12.0, 0.5, 2.0], [5.0, 12.0, 9.0]]).T,
            np.array([0.2, 0.1]), np.array([0.0, 60.0]), M1]
    seg2 = [np.array([[30.0, 0.0, 1.0], [40.0, -20.0, 1.0]]).T, np.array([[32.0, 0.5, 1.2], [42.0, -15.0, 2.0]]).T,
            np.array([0.5, 0.3]), np.zeros(2), M2]
    return [tx, rx, irs_two_ports()] + geo + seg1 + seg2


class test_case(unittest.TestCase):

    def setUp(self):
        # TX -> IRS -> RX on the x-axis at 1 m height; segment 1: LOS + one path, segment 2: LOS + one path
        self.omni = quadriga_lib.arrayant.generate("omni")
        self.irs = irs_two_ports()
        z = np.zeros(3)
        self.geo = [np.array([0.0, 0.0, 1.0]), z, np.array([40.0, 0.0, 1.0]), z, np.array([20.0, 0.0, 1.0]), z]
        M = np.zeros((8, 2))
        M[0], M[6] = 1.0, -1.0
        f1 = np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 11.0]]).T
        f2 = np.array([[30.0, 0.0, 1.0], [40.0, -20.0, 1.0]]).T
        self.seg1 = [f1, f1.copy(), np.array([0.2, 0.1]), np.zeros(2), M]
        self.seg2 = [f2, f2.copy(), np.array([0.5, 0.3]), np.zeros(2), M.copy()]
        self.names = ["tx_array", "rx_array", "irs_array", "tx_pos", "tx_orientation", "rx_pos", "rx_orientation",
                      "irs_pos", "irs_orientation", "fbs_pos_1", "lbs_pos_1", "path_gain_1", "path_length_1", "M_1",
                      "fbs_pos_2", "lbs_pos_2", "path_gain_2", "path_length_2", "M_2"]

    def args(self, irs=None):
        return [self.omni, self.omni, self.irs if irs is None else irs] + self.geo + self.seg1 + self.seg2

    def call(self, *extra, **kw):
        return quadriga_lib.channel.irs(*self.args(), *extra, **kw)

    def test_basic_call(self):
        out = self.call(use_absolute_delays=True)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        coeff, delay, active_path_out = out
        self.assertEqual(coeff.shape, (1, 1, 4))
        self.assertEqual(coeff.dtype, np.complex128)
        self.assertEqual(delay.dtype, np.float64)
        self.assertEqual(active_path_out.dtype, np.bool_)
        npt.assert_array_equal(active_path_out, [True, True, True, True])
        # Path pair (i_1, i_2) at index i_1 * n_path_2 + i_2; first codebook entry uses element 0 only
        g = np.sqrt(np.outer([0.2, 0.1], [0.5, 0.3])).ravel()
        npt.assert_allclose(coeff.ravel(), g, atol=1e-14, rtol=0)
        d1 = np.array([20.0, np.sqrt(200.0) + np.sqrt(600.0)])
        d2 = np.array([20.0, np.sqrt(800.0) + 20.0])
        npt.assert_allclose(delay.ravel() * C0, np.add.outer(d1, d2).ravel(), atol=1e-9, rtol=0)

    def test_segments_reach_their_arguments(self):
        # With a single unit IRS weight on element 0, a pair is the product of its two segment channels; each segment
        # is computed by channel.spherical, so every input and the four angle outputs are checked against it
        args = distinct_args()
        tx, rx, irs = args[:3]
        tp, to, rp, ro, ip, io = args[3:9]
        seg1, seg2 = args[9:14], args[14:19]
        irs_1 = dict(irs, coupling_re=irs["coupling_re"][:, :1], coupling_im=irs["coupling_im"][:, :1])
        c1, d1, aod, eod, _, _ = quadriga_lib.channel.spherical(tx, irs_1, tp, to, ip, io, *seg1, 3.5e9, True, angles=True)
        c2, d2, _, _, aoa, eoa = quadriga_lib.channel.spherical(irs_1, rx, ip, io, rp, ro, *seg2, 3.5e9, True, angles=True)
        shape = (2, 1, 4)
        pair_1 = lambda x: np.broadcast_to(x[:, :, :, None], (2, 1, 2, 2)).reshape(shape)
        pair_2 = lambda x: np.broadcast_to(x[:, :, None, :], (2, 1, 2, 2)).reshape(shape)
        out = quadriga_lib.channel.irs(*args, center_freq=3.5e9, use_absolute_delays=True, angles=True)
        self.assertEqual(out[0].shape, shape)
        npt.assert_allclose(out[0], pair_1(c1) * pair_2(c2), atol=1e-14, rtol=0)
        npt.assert_allclose(out[1], pair_1(d1) + pair_2(d2), atol=1e-17, rtol=0)
        for a, b in zip(out[3:], (pair_1(aod), pair_1(eod), pair_2(aoa), pair_2(eoa))):
            npt.assert_allclose(a, b, atol=1e-14, rtol=0)

    def test_keyword_arguments(self):
        args = distinct_args()
        opts = dict(i_irs=1, threshold_dB=-100.0, center_freq=1.0e9, use_absolute_delays=True, angles=True,
                    irs_array_2=irs_two_ports(), active_path=[1, 1, 0, 1], use_kernel=1, complex=False)
        ref = quadriga_lib.channel.irs(*args, *list(opts.values())[:-1], complex=False)
        out = quadriga_lib.channel.irs(**dict(zip(self.names, args)), **opts)
        self.assertEqual(len(out), 8)
        for a, b in zip(ref, out):
            npt.assert_array_equal(a, b)

    def test_optional_inputs(self):
        ref = self.call()
        explicit = self.call(0, -140.0, 0.0, False, False, None, None, 0, complex=True)
        nones = self.call(use_absolute_delays=None, angles=None, irs_array_2=None, active_path=None, complex=None)
        empty_mask = self.call(active_path=[])
        for out in (explicit, nones, empty_mask):
            for a, b in zip(ref, out):
                npt.assert_array_equal(a, b)

        # i_irs selects the codebook column: entry 1 equals entry 0 of the array with swapped columns
        swapped = dict(self.irs, coupling_re=self.irs["coupling_re"][:, ::-1], coupling_im=self.irs["coupling_im"][:, ::-1])
        out = self.call(i_irs=1)[0]
        npt.assert_allclose(out, quadriga_lib.channel.irs(*self.args(irs=swapped))[0], atol=1e-14, rtol=0)
        self.assertGreater(np.abs(out - ref[0]).max(), 1e-3)
        # threshold_dB drops the weak pairs (pair powers -10.0, -12.2, -13.0, -15.2 dB)
        coeff, _, act = self.call(threshold_dB=-14.0)
        npt.assert_array_equal(act, [True, True, True, False])
        self.assertEqual(coeff.shape, (1, 1, 3))
        # center_freq adds the delay phase
        self.assertGreater(np.abs(self.call(center_freq=3.5e9)[0].imag).max(), 1e-3)
        # use_absolute_delays adds the TX-RX LOS delay (40 m)
        npt.assert_allclose(self.call(use_absolute_delays=True)[1] - ref[1], 40.0 / C0, atol=1e-15, rtol=0)
        # irs_array_2 replaces the RX-facing IRS pattern (vertically polarized omni: scaling e_theta scales it)
        irs_2 = dict(self.irs, e_theta_re=0.5 * self.irs["e_theta_re"])
        npt.assert_allclose(self.call(irs_array_2=irs_2)[0], ref[0] * 0.5, atol=1e-14, rtol=0)
        # active_path overrides threshold_dB; bool, numeric and list input
        for mask in ([True, False, False, True], np.array([1.0, 0.0, 0.0, 2.0]), np.array([1, 0, 0, 1])):
            coeff, _, act = self.call(threshold_dB=100.0, active_path=mask)
            npt.assert_array_equal(act, [True, False, False, True])
            npt.assert_array_equal(coeff.ravel(), ref[0].ravel()[[0, 3]])
        # use_kernel reaches C++: GENERIC equals auto, 3 is rejected
        npt.assert_array_equal(self.call(use_kernel=1)[0], ref[0])
        with self.assertRaises(ValueError) as ctx:
            self.call(use_kernel=3)
        self.assertIn("use_kernel", str(ctx.exception))
        # TX and RX dicts with only the mandatory keys
        minimal = {k: self.omni[k] for k in MANDATORY_KEYS}
        out = quadriga_lib.channel.irs(minimal, minimal, *self.args()[2:])
        npt.assert_allclose(out[0], ref[0], atol=1e-14, rtol=0)

    def test_angles_flag(self):
        ref = self.call()
        out = self.call(angles=True)
        self.assertEqual(len(out), 7)
        for a, b in zip(ref, out[:3]):
            npt.assert_array_equal(a, b)
        for a in out[3:]:
            self.assertEqual(a.shape, (1, 1, 4))
            self.assertEqual(a.dtype, np.float64)
        self.assertEqual(len(self.call(angles=True, complex=False)), 8)

    def test_complex_flag(self):
        coeff, delay, act = self.call(center_freq=3.5e9)
        out = self.call(center_freq=3.5e9, complex=False)
        self.assertEqual(len(out), 4)
        coeff_re, coeff_im, delay_s, act_s = out
        self.assertEqual(coeff_re.dtype, np.float64)
        npt.assert_array_equal(coeff_re, coeff.real)
        npt.assert_array_equal(coeff_im, coeff.imag)
        npt.assert_array_equal(delay_s, delay)
        npt.assert_array_equal(act_s, act)

    def test_memory_layout_and_dtype(self):
        ref = self.call(center_freq=3.5e9)
        args = self.args()
        f = [np.asfortranarray(a) if isinstance(a, np.ndarray) else a for a in args]
        wide = np.zeros((3, 4))
        wide[:, ::2] = self.seg1[0]
        strided = list(args)
        strided[9] = wide[:, ::2]
        single = [a.astype(np.float32) if isinstance(a, np.ndarray) else a for a in args]
        single[3] = [0, 0, 1]
        # float32 inputs: the gains are rounded (0.2, 0.1, ...), the integer geometry is exact
        for a, tol in ((f, 1e-14), (strided, 1e-14), (single, 1e-6)):
            out = quadriga_lib.channel.irs(*a, center_freq=3.5e9)
            npt.assert_allclose(out[0], ref[0], atol=tol, rtol=0)
            npt.assert_allclose(out[1], ref[1], atol=1e-15, rtol=0)

    def test_inputs_not_modified(self):
        args = self.args()
        copies = [{k: v.copy() for k, v in a.items() if isinstance(v, np.ndarray)} if isinstance(a, dict) else a.copy()
                  for a in args]
        self.call(1, -100.0, 3.5e9, True, True, irs_two_ports(), [1, 1, 0, 1])
        for a, c in zip(args, copies):
            if isinstance(a, dict):
                for k in c:
                    npt.assert_array_equal(a[k], c[k])
            else:
                npt.assert_array_equal(a, c)

    def test_empty_segment(self):
        args = self.args()
        args[14:19] = [np.zeros((3, 0)), np.zeros((3, 0)), np.zeros(0), np.zeros(0), np.zeros((8, 0))]
        out = quadriga_lib.channel.irs(*args, angles=True)
        for k in (0, 1, 3, 4, 5, 6):
            self.assertEqual(out[k].shape, (1, 1, 0))
        self.assertEqual(out[2].shape, (0,))

    def test_errors(self):
        # From C++
        with self.assertRaises(ValueError) as ctx:
            self.call(i_irs=2)
        self.assertIn("IRS codebook index out of bound", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.call(active_path=[True, False])
        self.assertIn("active_path", str(ctx.exception))
        # IRS dicts without coupling (weights) are rejected, also for irs_array_2
        no_coupling = {k: v for k, v in self.irs.items() if not k.startswith("coupling")}
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.channel.irs(*self.args(irs=no_coupling))
        self.assertIn("Coupling matrix", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            self.call(irs_array_2=no_coupling)
        self.assertIn("Input 'irs_array_2' must have a coupling matrix (the IRS weights)", str(ctx.exception))
        # From the wrapper and the input conversion
        with self.assertRaises(TypeError) as ctx:
            self.call(irs_array_2=5)
        self.assertIn("irs_array_2", str(ctx.exception))
        args = self.args()
        args[7] = np.zeros(2)
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.channel.irs(*args)
        self.assertIn("irs_pos", str(ctx.exception))
        # Flags
        for flag in ("use_absolute_delays", "angles", "complex"):
            with self.assertRaises(TypeError) as ctx:
                self.call(**{flag: 2.5})
            self.assertIn(flag, str(ctx.exception))
        # From pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.channel.irs(*self.args()[:-1])
        with self.assertRaises(TypeError):
            self.call(bad_name=1)
        with self.assertRaises(TypeError):  # complex is keyword-only
            self.call(0, -140.0, 0.0, False, False, None, None, 0, True)
        args = self.args()
        args[9] = None
        with self.assertRaises(TypeError):
            quadriga_lib.channel.irs(*args)


if __name__ == "__main__":
    unittest.main()
