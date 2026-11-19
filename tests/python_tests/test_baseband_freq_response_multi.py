# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.channel.baseband_freq_response_multi.
# Scope: argument and output handling, defaults, snapshot lists, error paths.
# Physics validation lives in the Catch2 suite (test_baseband_freq_response_multi.cpp).

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

import subprocess
import quadriga_lib
from quadriga_lib import channel


def module_dir():
    """Directory to put on PYTHONPATH so that a subprocess imports the same quadriga_lib module."""
    path = os.path.abspath(quadriga_lib.__file__)
    if os.path.basename(path).startswith('__init__.'):
        path = os.path.dirname(path)
    return os.path.dirname(path)

class test_case(unittest.TestCase):

    # ---- Error handling ----

    def test_3d_coeff(self):
        # 3D items are one input frequency; with more input frequencies they are an error
        c = np.array([1.0, 0.5j]).reshape(1, 1, 2)
        d = np.array([10e-9, 35e-9]).reshape(1, 1, 2)
        f, f_new = np.array([3e9]), np.array([3.01e9, 2.99e9])
        npt.assert_array_equal(channel.baseband_freq_response_multi([c], [d], f, f_new),
                               channel.baseband_freq_response_multi([c[..., None]], [d], f, f_new))
        with self.assertRaisesRegex(ValueError, 'must be 4D, got 3D'):
            channel.baseband_freq_response_multi([c], [d], np.array([1e9, 2e9]), f_new)

    def test_error_freq_dim_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'does not match the length of'):
            channel.baseband_freq_response_multi([np.zeros((1, 1, 1, 3), dtype=complex)], [np.zeros((1, 1, 1))],
                                                 np.array([1e9, 2e9]), np.array([1e9]))

    def test_error_delay_freq_dim_mismatch(self):
        with self.assertRaisesRegex(ValueError, "4th dimension of 'delay'"):
            channel.baseband_freq_response_multi([np.zeros((1, 1, 1, 3), dtype=complex)], [np.zeros((1, 1, 1, 2))],
                                                 np.array([1e9, 2e9, 3e9]), np.array([1e9]))

    def test_error_empty_freq(self):
        with self.assertRaisesRegex(ValueError, "'freq' must have at least one element"):
            channel.baseband_freq_response_multi([np.zeros((1, 1, 1, 1), dtype=complex)], [np.zeros((1, 1, 1))],
                                                 np.array([]), np.array([1e9]))

    def test_error_snap_out_of_range(self):
        with self.assertRaises(ValueError):
            channel.baseband_freq_response_multi([np.zeros((1, 1, 1, 1), dtype=complex)], [np.zeros((1, 1, 1))],
                                                 np.array([1e9]), np.array([1e9]), i_snap=np.array([5]))

    def test_empty_list(self):
        with self.assertRaisesRegex(ValueError, "'coeff_re' must contain at least one snapshot"):
            channel.baseband_freq_response_multi([], [], np.array([1e9]), np.array([1e9]))

    # ---- Snapshot lists, delay layouts, remove_delay_phase ----

    def test_remove_delay_phase_false(self):
        # remove_delay_phase = False reaches C++: a constant envelope gets the delay phase of freq_new only
        tau = 10e-9; fout = np.array([1e9, 1.5e9, 2e9])
        dl = np.zeros((1, 1, 1)); dl[0, 0, 0] = tau
        hmat = channel.baseband_freq_response_multi(coeff=[np.ones((1, 1, 1, 1), dtype=complex)],
                                                     delay=[dl], freq=np.array([1e9]), freq_new=fout,
                                                     remove_delay_phase=False)
        for k in range(3):
            npt.assert_allclose(hmat[0, 0, k, 0], np.exp(-1j * 2 * np.pi * fout[k] * tau), atol=1e-6, rtol=0)

    def test_varying_n_path(self):
        fin = np.array([1e9, 2e9]); fout = np.array([1.5e9])
        c0 = np.ones((1, 1, 1, 2), dtype=complex)
        c1 = np.ones((1, 1, 3, 2), dtype=complex)
        hmat = channel.baseband_freq_response_multi(coeff=[c0, c1],
                                                     delay=[np.zeros((1, 1, 1)), np.zeros((1, 1, 3))],
                                                     freq=fin, freq_new=fout, remove_delay_phase=False)
        self.assertEqual(hmat.shape, (1, 1, 1, 2))
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 0]), 1.0, atol=1e-6, rtol=0)
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 1]), 3.0, atol=1e-6, rtol=0)

    def test_snap_indices(self):
        fin = np.array([1e9, 2e9]); fout = np.array([1.5e9])
        cs = [np.ones((1, 1, 1, 2), dtype=complex) * v for v in [1, 2, 3]]
        dl = [np.zeros((1, 1, 1))] * 3
        hmat = channel.baseband_freq_response_multi(coeff=cs, delay=dl, freq=fin, freq_new=fout,
                                                     i_snap=np.array([2, 0]), remove_delay_phase=False)
        self.assertEqual(hmat.shape, (1, 1, 1, 2))
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 0]), 3.0, atol=1e-6, rtol=0)
        npt.assert_allclose(np.abs(hmat[0, 0, 0, 1]), 1.0, atol=1e-6, rtol=0)

    def test_3d_and_4d_delay(self):
        fin = np.array([1e9, 2e9]); fout = np.array([1e9]); tau = 10e-9
        c = np.ones((1, 1, 1, 2), dtype=complex)
        dl3 = np.zeros((1, 1, 1)); dl3[0, 0, 0] = tau
        dl4 = np.zeros((1, 1, 1, 2)); dl4[0, 0, 0, :] = tau
        h3 = channel.baseband_freq_response_multi(coeff=[c], delay=[dl3], freq=fin, freq_new=fout,
                                                   remove_delay_phase=False)
        h4 = channel.baseband_freq_response_multi(coeff=[c], delay=[dl4], freq=fin, freq_new=fout,
                                                   remove_delay_phase=False)
        npt.assert_allclose(h3, h4, atol=1e-6, rtol=0)

    def test_output_shape(self):
        hmat = channel.baseband_freq_response_multi(coeff=[np.ones((2, 3, 5, 2), dtype=complex)],
                                                    delay=[np.zeros((1, 1, 5))],
                                                    freq=np.array([1e9, 2e9]),
                                                    freq_new=np.linspace(1e9, 2e9, 64))
        self.assertEqual(hmat.shape, (2, 3, 64, 1))

    # An invalid multi-frequency input raises ValueError with one and with several threads. It runs in a subprocess,
    # since an exception leaving an OpenMP region aborts the interpreter (std::terminate).
    def test_exception_in_parallel_region_raises(self):
        code = (
            "import numpy as np, quadriga_lib as q\n"
            "rng = np.random.default_rng(0)\n"
            "coeff = [rng.normal(size=(2, 2, 3, 2)) + 1j * rng.normal(size=(2, 2, 3, 2)) for _ in range(5)]\n"
            "delay = [rng.random((2, 2, 3)) * 1e-7 for _ in range(5)]\n"
            "try:\n"
            "    q.channel.baseband_freq_response_multi(coeff, delay, freq=np.array([2e9, 1e9]), freq_new=np.array([1.5e9]))\n"
            "    print('no exception')\n"
            "except ValueError as e:\n"
            "    print('ValueError:', e)\n"
            "print('alive')\n")
        env = dict(os.environ)
        env['PYTHONPATH'] = module_dir()
        for n_threads in ('1', '4'):
            env['OMP_NUM_THREADS'] = n_threads
            r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=300)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            lines = r.stdout.splitlines()
            self.assertEqual(len(lines), 2, r.stdout)
            self.assertIn("ValueError: Input 'freq' must be sorted in strictly ascending order.", lines[0])
            self.assertEqual(lines[1], 'alive')


class test_carrier_frequencies(unittest.TestCase):

    # Absolute carriers for single-frequency coefficients: one input frequency f0 gives
    # H(f) = sum_p c_p * exp(-j 2 pi (f - f0) tau_p), the result of baseband_freq_response with
    # pilot_grid * bandwidth = f - f0; carriers in any order, also below f0. Same closed form as in the MATLAB test.
    def test(self):
        cre = np.array([1.0, 0.5]); tau = np.array([10e-9, 35e-9])
        c = [cre.reshape(1, 1, 2).astype(complex)]; d = [tau.reshape(1, 1, 2)]
        f0 = 3e9; fa = f0 + np.array([0, 10e6, 20e6])
        def H(f):
            return (cre[:, None] * np.exp(-2j * np.pi * (np.atleast_1d(f)[None, :] - f0) * tau[:, None])).sum(0)
        def run(fo):
            return channel.baseband_freq_response_multi(c, d, np.array([f0]), np.atleast_1d(fo)).ravel()
        npt.assert_allclose(run(fa), H(fa), atol=1e-5)
        npt.assert_allclose(run(fa[::-1]), H(fa[::-1]), atol=1e-5)
        fu = f0 + np.array([10e6, 0, 20e6, -10e6])
        npt.assert_allclose(run(fu), H(fu), atol=1e-5)
        h_ref = channel.baseband_freq_response(c, d, (fu - f0) / 10e6, 10e6).ravel()
        npt.assert_allclose(run(fu), h_ref, atol=1e-5)


class test_keyword_call(unittest.TestCase):
    # All arguments by keyword and the optional ones as None give the positional result; inputs are not modified
    def test(self):
        rng = np.random.default_rng(4)
        c = rng.normal(size=(3, 2, 4, 2)) + 1j * rng.normal(size=(3, 2, 4, 2))
        d = rng.random((3, 2, 4)) * 1e-7
        f, f_new = np.array([3e9, 3.2e9]), np.array([3.05e9, 2.9e9, 3.1e9])
        c0, d0 = c.copy(), d.copy()
        ref = channel.baseband_freq_response_multi([c, 2 * c], [d, d], f, f_new)
        h = channel.baseband_freq_response_multi(coeff=[c, 2 * c], delay=[d, d], freq=f, freq_new=f_new,
                                                 remove_delay_phase=None, i_snap=None, use_kernel=0)
        npt.assert_array_equal(h, ref)
        npt.assert_array_equal(channel.baseband_freq_response_multi([np.asfortranarray(c)], [d], f, f_new),
                               ref[..., :1])
        npt.assert_array_equal(c, c0)
        npt.assert_array_equal(d, d0)


class test_use_kernel(unittest.TestCase):
    # use_kernel (last argument) reaches C++: 1 is accepted, 3 raises, also when all snapshots are empty
    def test(self):
        coeff_mf, delay_mf = [np.ones((2, 3, 4, 2), dtype=complex)], [np.zeros((1, 1, 4, 2))]
        f, f_new = np.array([3e9, 3.2e9]), np.array([3e9, 3.1e9])
        channel.baseband_freq_response_multi(coeff_mf, delay_mf, f, f_new, use_kernel=1)
        calls = (lambda: channel.baseband_freq_response_multi(coeff_mf, delay_mf, f, f_new, use_kernel=3),
                 lambda: channel.baseband_freq_response_multi([], [], f, f_new, use_kernel=3))
        for call in calls:
            with self.assertRaises(ValueError) as ctx:
                call()
            self.assertIn("'use_kernel' must be 0, 1 or 2", str(ctx.exception))


class test_remove_delay_phase_flag(unittest.TestCase):
    # remove_delay_phase: None equals the default (True), a non-bool raises TypeError naming the argument
    def test(self):
        rng = np.random.default_rng(0)
        coeff = [rng.normal(size=(1, 1, 2, 2)) + 1j * rng.normal(size=(1, 1, 2, 2))]
        delay = [rng.random((1, 1, 2)) * 1e-7]
        f, f_new = np.array([3e9, 3.2e9]), np.array([3e9, 3.1e9])
        h_def = channel.baseband_freq_response_multi(coeff, delay, freq=f, freq_new=f_new)
        h_none = channel.baseband_freq_response_multi(coeff, delay, freq=f, freq_new=f_new, remove_delay_phase=None)
        h_true = channel.baseband_freq_response_multi(coeff, delay, freq=f, freq_new=f_new, remove_delay_phase=True)
        npt.assert_array_equal(h_none, h_def)
        npt.assert_array_equal(h_true, h_def)
        with self.assertRaisesRegex(TypeError, 'remove_delay_phase'):
            channel.baseband_freq_response_multi(coeff, delay, freq=f, freq_new=f_new, remove_delay_phase=2.5)

if __name__ == '__main__':
    unittest.main()
