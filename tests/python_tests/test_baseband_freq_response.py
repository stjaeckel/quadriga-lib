# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.channel.baseband_freq_response.
# Scope: argument and output handling, defaults, snapshot lists, error paths.
# Physics validation lives in the Catch2 suite (test_baseband_freq_response.cpp).

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib
from quadriga_lib import channel

PG = np.linspace(0, 1, 4)  # pilot grid of the error tests


class test_case(unittest.TestCase):

    # ---- Reference values ----

    def test_reference_values(self):
        coeff = np.zeros((4, 3, 2), dtype=complex)
        coeff[:, 0, 0] = np.arange(0.25, 1.25, 0.25)
        coeff[:, 1, 0] = np.arange(1, 5)
        coeff[:, 2, 0] = 1j * np.arange(1, 5)
        coeff[:, :, 1] = -coeff[:, :, 0]

        coeff_list = [coeff, 2 * coeff, 3 * coeff]
        fc = 299792458.0
        delay = np.zeros((1, 1, 2), dtype=float)
        delay[0, 0, 0] = 1 / fc
        delay[0, 0, 1] = 1.5 / fc
        delay_list = [delay, delay, delay]
        pilots = np.arange(0, 2.1, 0.1)

        hmat = channel.baseband_freq_response(coeff_list, delay_list, np.linspace(0, 1, 11), fc)
        hmat_re = np.real(hmat)
        hmat_im = np.imag(hmat)

        T = np.zeros((4, 3))
        npt.assert_almost_equal(hmat_re[:, :, 0, 0], T, decimal=5)
        npt.assert_almost_equal(hmat_im[:, :, 0, 0], T, decimal=5)
        npt.assert_almost_equal(hmat_re[:, 0, :, :] * 4.0, hmat_re[:, 1, :, :], decimal=6)

        T = np.array([[0.5, 2, 0], [1, 4, 0], [1.5, 6, 0], [2, 8, 0]])
        npt.assert_almost_equal(hmat_re[:, :, 10, 0], T, decimal=6)

        npt.assert_almost_equal(hmat_re[:, :, :, 0] * 2.0, hmat_re[:, :, :, 1], decimal=6)
        npt.assert_almost_equal(hmat_re[:, :, :, 0] * 3.0, hmat_re[:, :, :, 2], decimal=5)

        hmat = channel.baseband_freq_response(coeff_list, delay_list, pilots, fc, i_snap=[1, 2, 1, 0])
        hmat_re = np.real(hmat)
        hmat_im = np.imag(hmat)

        T = np.zeros((4, 3))
        npt.assert_almost_equal(hmat_re[:, :, 20, 3], T, decimal=5)
        npt.assert_almost_equal(hmat_im[:, :, 20, 3], T, decimal=5)

        T = np.array([[0.25, 1, -1], [0.5, 2, -2], [0.75, 3, -3], [1, 4, -4]])
        npt.assert_almost_equal(hmat_im[:, :, 15, 3], T, decimal=5)

        npt.assert_almost_equal(hmat_re[:, :, :, 3] * 2.0, hmat_re[:, :, :, 0], decimal=5)
        npt.assert_almost_equal(hmat_re[:, :, :, 3] * 3.0, hmat_re[:, :, :, 1], decimal=5)
        npt.assert_almost_equal(hmat_re[:, :, :, 3] * 2.0, hmat_re[:, :, :, 2], decimal=5)

    # ---- Error handling ----

    def test_error_missing_argument(self):
        c, d = [np.zeros((1, 1, 1), dtype=complex)], [np.zeros((1, 1, 1))]
        for kw in ({'delay': d, 'pilot_grid': PG, 'bandwidth': 1e6}, {'coeff': c, 'pilot_grid': PG, 'bandwidth': 1e6},
                   {'coeff': c, 'delay': d, 'bandwidth': 1e6}, {'coeff': c, 'delay': d, 'pilot_grid': PG}):
            with self.assertRaises(TypeError):
                channel.baseband_freq_response(**kw)

    def test_error_delay_not_list(self):
        with self.assertRaisesRegex(ValueError, "'delay' must be a list"):
            channel.baseband_freq_response([np.zeros((1, 1, 1), dtype=complex)], np.zeros((1, 1, 1)), PG, 1e6)

    def test_error_coeff_not_list(self):
        with self.assertRaisesRegex(ValueError, "'coeff' must be a list"):
            channel.baseband_freq_response(np.zeros((1, 1, 1), dtype=complex), [np.zeros((1, 1, 1))], PG, 1e6)

    def test_error_delay_length_mismatch(self):
        c = [np.zeros((1, 1, 1), dtype=complex)] * 2
        with self.assertRaises(ValueError):
            channel.baseband_freq_response(c, [np.zeros((1, 1, 1))], PG, 1e6)

    def test_error_2d_coeff(self):
        with self.assertRaisesRegex(ValueError, 'must be 3D, got 2D'):
            channel.baseband_freq_response([np.zeros((2, 3), dtype=complex)], [np.zeros((1, 1, 1))], PG, 1e6)

    def test_error_4d_coeff(self):
        # A multi-frequency item is an error, not a batch
        with self.assertRaisesRegex(ValueError, 'must be 3D, got 4D'):
            channel.baseband_freq_response([np.zeros((1, 1, 1, 2), dtype=complex)], [np.zeros((1, 1, 1))], PG, 1e6)
        c = [np.zeros((1, 1, 1), dtype=complex), np.zeros((1, 1, 1, 2), dtype=complex)]
        with self.assertRaisesRegex(ValueError, 'Snapshot 1: .*must be 3D, got 4D'):
            channel.baseband_freq_response(c, [np.zeros((1, 1, 1))] * 2, PG, 1e6)

    def test_error_4d_delay(self):
        with self.assertRaisesRegex(ValueError, 'Delay arrays must have at most 3 dimensions'):
            channel.baseband_freq_response([np.zeros((1, 1, 1), dtype=complex)], [np.zeros((1, 1, 1, 2))], PG, 1e6)

    def test_error_empty_pilot_grid(self):
        with self.assertRaises(ValueError):
            channel.baseband_freq_response([np.zeros((1, 1, 1), dtype=complex)], [np.zeros((1, 1, 1))], np.array([]), 1e6)

    def test_error_n_rx_mismatch(self):
        c = [np.zeros((2, 1, 1), dtype=complex), np.zeros((3, 1, 1), dtype=complex)]
        with self.assertRaises(ValueError):
            channel.baseband_freq_response(c, [np.zeros((1, 1, 1))] * 2, PG, 1e6)

    def test_error_delay_shape_mismatch(self):
        with self.assertRaises(ValueError):
            channel.baseband_freq_response([np.zeros((1, 1, 3), dtype=complex)], [np.zeros((1, 1, 2))], PG, 1e6)

    def test_error_snap_out_of_range(self):
        with self.assertRaises(ValueError):
            channel.baseband_freq_response([np.zeros((1, 1, 1), dtype=complex)], [np.zeros((1, 1, 1))], PG, 1e6,
                                           i_snap=np.array([5]))

    def test_empty_list(self):
        with self.assertRaisesRegex(ValueError, "'coeff_re' must contain at least one snapshot"):
            channel.baseband_freq_response([], [], PG, 1e6)

    # ---- i_snap and varying n_path ----

    def test_snap(self):
        c = [i * np.ones((1, 1, 1), dtype=complex) for i in [1, 2, 3]]
        dl = [np.zeros((1, 1, 1))] * 3
        hmat = channel.baseband_freq_response(c, dl, PG, 1e6, i_snap=np.array([2, 0]))
        self.assertEqual(hmat.shape, (1, 1, 4, 2))
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 0]), 3.0, atol=1e-6, rtol=0)
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 1]), 1.0, atol=1e-6, rtol=0)

    def test_varying_n_path(self):
        c0 = np.ones((1, 1, 2), dtype=complex)
        c1 = np.ones((1, 1, 5), dtype=complex)
        hmat = channel.baseband_freq_response([c0, c1], [np.zeros((1, 1, 2)), np.zeros((1, 1, 5))], PG, 1e6)
        self.assertEqual(hmat.shape, (1, 1, 4, 2))
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 0]), 2.0, atol=1e-6, rtol=0)
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 1]), 5.0, atol=1e-6, rtol=0)

    def test_output_shape(self):
        hmat = channel.baseband_freq_response([np.ones((3, 2, 4), dtype=complex)], [np.zeros((1, 1, 4))],
                                              np.linspace(0, 1, 16), 1e6)
        self.assertEqual(hmat.shape, (3, 2, 16, 1))


class test_array_handling(unittest.TestCase):
    # Keyword call, defaults as None, memory layout and dtype of the inputs; inputs are not modified
    def test(self):
        rng = np.random.default_rng(3)
        c = rng.normal(size=(3, 2, 5)) + 1j * rng.normal(size=(3, 2, 5))
        d = rng.random((3, 2, 5)) * 1e-7
        pg = np.array([0.0, 0.3, -0.2, 1.0])
        c0, d0, pg0 = c.copy(), d.copy(), pg.copy()
        ref = channel.baseband_freq_response([c, 2 * c], [d, d], pg, 1e8)
        npt.assert_array_equal(channel.baseband_freq_response(coeff=[c, 2 * c], delay=[d, d], pilot_grid=pg,
                                                              bandwidth=1e8, i_snap=None, use_kernel=0), ref)
        big_c, big_d = np.zeros((3, 2, 10), dtype=complex), np.zeros((3, 2, 10))
        big_c[:, :, ::2], big_d[:, :, ::2] = c, d
        variants = {"fortran": ([np.asfortranarray(c), np.asfortranarray(2 * c)], [np.asfortranarray(d)] * 2),
                    "sliced": ([big_c[:, :, ::2], 2 * big_c[:, :, ::2]], [big_d[:, :, ::2]] * 2)}
        for name, (cv, dv) in variants.items():
            npt.assert_array_equal(channel.baseband_freq_response(cv, dv, pg, 1e8), ref, err_msg=name)
        h32 = channel.baseband_freq_response([c.astype(np.complex64), (2 * c).astype(np.complex64)],
                                             [d.astype(np.float32)] * 2, pg.astype(np.float32), 1e8)
        npt.assert_allclose(h32, ref, atol=1e-5 * np.abs(ref).max(), rtol=0)
        npt.assert_array_equal(c, c0)
        npt.assert_array_equal(d, d0)
        npt.assert_array_equal(pg, pg0)


class test_use_kernel(unittest.TestCase):
    # use_kernel (last argument) reaches C++: 1 is accepted, 3 raises, also when all snapshots are empty
    def test(self):
        coeff, delay = [np.ones((2, 3, 4), dtype=complex)], [np.zeros((1, 1, 4))]
        pg = np.linspace(0, 1, 5)
        channel.baseband_freq_response(coeff, delay, pg, 1e8, use_kernel=1)
        calls = (lambda: channel.baseband_freq_response(coeff, delay, pg, 1e8, use_kernel=3),
                 lambda: channel.baseband_freq_response([np.ones((2, 3, 0), dtype=complex)], [np.zeros((1, 1, 0))],
                                                        pg, 1e8, use_kernel=3),
                 lambda: channel.baseband_freq_response([], [], pg, 1e8, use_kernel=3))
        for call in calls:
            with self.assertRaises(ValueError) as ctx:
                call()
            self.assertIn("'use_kernel' must be 0, 1 or 2", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
