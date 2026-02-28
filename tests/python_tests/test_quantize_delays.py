# SPDX-License-Identifier: Apache-2.0
#
# quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
# Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

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


class test_quantize_delays(unittest.TestCase):

    def test_input_validation(self):
        """Test that invalid inputs raise errors"""
        cre = [np.ones((2, 2, 3))]
        cim = [np.zeros((2, 2, 3))]
        dl = [np.random.rand(2, 2, 3) * 100e-9]

        # Invalid tap_spacing
        with self.assertRaises(Exception):
            quadriga_lib.channel.quantize_delays(cre, cim, dl, tap_spacing=0.0)

        with self.assertRaises(Exception):
            quadriga_lib.channel.quantize_delays(cre, cim, dl, tap_spacing=-1e-9)

        # Invalid power_exponent
        with self.assertRaises(Exception):
            quadriga_lib.channel.quantize_delays(cre, cim, dl, power_exponent=0.0)

        # Invalid fix_taps
        with self.assertRaises(Exception):
            quadriga_lib.channel.quantize_delays(cre, cim, dl, fix_taps=-1)

        with self.assertRaises(Exception):
            quadriga_lib.channel.quantize_delays(cre, cim, dl, fix_taps=4)

        # Mismatched n_rx across snapshots
        cre2 = [np.ones((2, 2, 3)), np.ones((3, 2, 3))]
        cim2 = [np.zeros((2, 2, 3)), np.zeros((3, 2, 3))]
        dl2 = [np.zeros((2, 2, 3)), np.zeros((3, 2, 3))]
        with self.assertRaises(Exception):
            quadriga_lib.channel.quantize_delays(cre2, cim2, dl2)

    def test_exact_tap_boundary(self):
        """Single path at exact tap boundary passes through unchanged"""
        cre = [np.array([[[2.0]]])]
        cim = [np.array([[[3.0]]])]
        dl = [np.array([[[15.0e-9]]])]

        cre_q, cim_q, dl_q = quadriga_lib.channel.quantize_delays(cre, cim, dl, tap_spacing=5e-9)

        self.assertIsInstance(cre_q, list)
        self.assertEqual(len(cre_q), 1)

        # Find the non-zero tap
        found = False
        for k in range(cre_q[0].shape[2]):
            if abs(cre_q[0][0, 0, k]) > 1e-10:
                npt.assert_allclose(cre_q[0][0, 0, k], 2.0, atol=1e-12, rtol=0)
                npt.assert_allclose(cim_q[0][0, 0, k], 3.0, atol=1e-12, rtol=0)
                npt.assert_allclose(dl_q[0][0, 0, k], 15.0e-9, atol=1e-20, rtol=0)
                found = True
        self.assertTrue(found)

    def test_half_tap_linear(self):
        """Single path at half-tap offset with linear exponent (alpha=1.0)"""
        cre = [np.array([[[4.0]]])]
        cim = [np.array([[[0.0]]])]
        dl = [np.array([[[12.5e-9]]])]

        cre_q, cim_q, dl_q = quadriga_lib.channel.quantize_delays(
            cre, cim, dl, tap_spacing=5e-9, power_exponent=1.0)

        n_nonzero = 0
        sum_re = 0.0
        for k in range(cre_q[0].shape[2]):
            v = cre_q[0][0, 0, k]
            if abs(v) > 1e-10:
                npt.assert_allclose(v, 2.0, atol=1e-12, rtol=0)
                sum_re += v
                n_nonzero += 1

        self.assertEqual(n_nonzero, 2)
        npt.assert_allclose(sum_re, 4.0, atol=1e-12, rtol=0)

    def test_half_tap_sqrt(self):
        """Single path at half-tap offset with sqrt exponent (alpha=0.5)"""
        cre = [np.array([[[4.0]]])]
        cim = [np.array([[[0.0]]])]
        dl = [np.array([[[12.5e-9]]])]

        cre_q, cim_q, dl_q = quadriga_lib.channel.quantize_delays(
            cre, cim, dl, tap_spacing=5e-9, power_exponent=0.5)

        expected_coeff = np.sqrt(0.5) * 4.0
        sum_power = 0.0
        n_nonzero = 0
        for k in range(cre_q[0].shape[2]):
            v = cre_q[0][0, 0, k]
            if abs(v) > 1e-10:
                npt.assert_allclose(v, expected_coeff, atol=1e-12, rtol=0)
                sum_power += v * v
                n_nonzero += 1

        self.assertEqual(n_nonzero, 2)
        npt.assert_allclose(sum_power, 16.0, atol=1e-10, rtol=0)

    def test_already_quantized(self):
        """Already quantized input passes through with correct coefficients"""
        cre = [np.zeros((1, 1, 3))]
        cim = [np.zeros((1, 1, 3))]
        dl = [np.zeros((1, 1, 3))]

        cre[0][0, 0, 0] = 1.0; cim[0][0, 0, 0] = 0.5; dl[0][0, 0, 0] = 0.0
        cre[0][0, 0, 1] = 2.0; cim[0][0, 0, 1] = 1.0; dl[0][0, 0, 1] = 5.0e-9
        cre[0][0, 0, 2] = 0.5; cim[0][0, 0, 2] = 0.3; dl[0][0, 0, 2] = 20.0e-9

        cre_q, cim_q, dl_q = quadriga_lib.channel.quantize_delays(
            cre, cim, dl, tap_spacing=5e-9)

        for k in range(cre_q[0].shape[2]):
            d = dl_q[0][0, 0, k]
            re = cre_q[0][0, 0, k]
            im = cim_q[0][0, 0, k]

            if abs(d) < 1e-20:
                npt.assert_allclose(re, 1.0, atol=1e-12, rtol=0)
                npt.assert_allclose(im, 0.5, atol=1e-12, rtol=0)
            elif abs(d - 5e-9) < 1e-20:
                npt.assert_allclose(re, 2.0, atol=1e-12, rtol=0)
                npt.assert_allclose(im, 1.0, atol=1e-12, rtol=0)
            elif abs(d - 20e-9) < 1e-20:
                npt.assert_allclose(re, 0.5, atol=1e-12, rtol=0)
                npt.assert_allclose(im, 0.3, atol=1e-12, rtol=0)

    def test_max_no_taps(self):
        """max_no_taps limits the number of output taps"""
        cre = [np.zeros((1, 1, 5))]
        cim = [np.zeros((1, 1, 5))]
        dl = [np.zeros((1, 1, 5))]

        cre[0][0, 0, 0] = 5.0; dl[0][0, 0, 0] = 0.0
        cre[0][0, 0, 1] = 1.0; dl[0][0, 0, 1] = 5.0e-9
        cre[0][0, 0, 2] = 3.0; dl[0][0, 0, 2] = 10.0e-9
        cre[0][0, 0, 3] = 0.5; dl[0][0, 0, 3] = 15.0e-9
        cre[0][0, 0, 4] = 4.0; dl[0][0, 0, 4] = 20.0e-9

        cre_q, cim_q, dl_q = quadriga_lib.channel.quantize_delays(
            cre, cim, dl, tap_spacing=5e-9, max_no_taps=3)

        self.assertLessEqual(cre_q[0].shape[2], 3)

        n_nonzero = sum(1 for k in range(cre_q[0].shape[2]) if abs(cre_q[0][0, 0, k]) > 1e-10)
        self.assertEqual(n_nonzero, 3)

    def test_shared_delays_fix2(self):
        """Shared delays with fix_taps=2 produce shared output delays"""
        cre = [np.random.randn(2, 2, 3), np.random.randn(2, 2, 3)]
        cim = [np.random.randn(2, 2, 3), np.random.randn(2, 2, 3)]
        dl = [np.zeros((1, 1, 3)), np.zeros((1, 1, 3))]

        for s in range(2):
            dl[s][0, 0, 0] = 0.0
            dl[s][0, 0, 1] = 12.5e-9
            dl[s][0, 0, 2] = 30.0e-9

        cre_q, cim_q, dl_q = quadriga_lib.channel.quantize_delays(
            cre, cim, dl, tap_spacing=5e-9, fix_taps=2)

        self.assertEqual(len(dl_q), 2)
        self.assertEqual(dl_q[0].shape[0], 1)  # shared
        self.assertEqual(dl_q[0].shape[1], 1)
        self.assertEqual(dl_q[1].shape[0], 1)
        self.assertEqual(dl_q[1].shape[1], 1)

    def test_shared_delays_fix0(self):
        """Shared delays with fix_taps=0 produce per-antenna output"""
        cre = [np.random.randn(2, 2, 2)]
        cim = [np.random.randn(2, 2, 2)]
        dl = [np.zeros((1, 1, 2))]
        dl[0][0, 0, 0] = 0.0
        dl[0][0, 0, 1] = 17.5e-9

        cre_q, cim_q, dl_q = quadriga_lib.channel.quantize_delays(
            cre, cim, dl, tap_spacing=5e-9, fix_taps=0)

        self.assertEqual(dl_q[0].shape[0], 2)
        self.assertEqual(dl_q[0].shape[1], 2)

    def test_multiple_snapshots(self):
        """Multiple snapshots with same n_path produce correct output"""
        n_snap = 3
        cre = [np.ones((1, 1, 2)) for _ in range(n_snap)]
        cim = [np.zeros((1, 1, 2)) for _ in range(n_snap)]
        dl = [np.zeros((1, 1, 2)) for _ in range(n_snap)]

        for s in range(n_snap):
            dl[s][0, 0, 1] = (s + 1) * 7.5e-9

        cre_q, cim_q, dl_q = quadriga_lib.channel.quantize_delays(
            cre, cim, dl, tap_spacing=5e-9)

        self.assertEqual(len(cre_q), n_snap)

    def test_variable_n_path(self):
        """Variable number of paths across snapshots"""
        # Snap 0: 3 paths, Snap 1: 1 path, Snap 2: 2 paths
        cre = [np.zeros((1, 1, 3)), np.zeros((1, 1, 1)), np.zeros((1, 1, 2))]
        cim = [np.zeros((1, 1, 3)), np.zeros((1, 1, 1)), np.zeros((1, 1, 2))]
        dl = [np.zeros((1, 1, 3)), np.zeros((1, 1, 1)), np.zeros((1, 1, 2))]

        cre[0][0, 0, 0] = 1.0; dl[0][0, 0, 0] = 0.0
        cre[0][0, 0, 1] = 2.0; dl[0][0, 0, 1] = 12.5e-9
        cre[0][0, 0, 2] = 0.5; dl[0][0, 0, 2] = 20.0e-9

        cre[1][0, 0, 0] = 3.0; dl[1][0, 0, 0] = 5.0e-9

        cre[2][0, 0, 0] = 1.5; dl[2][0, 0, 0] = 7.5e-9
        cre[2][0, 0, 1] = 0.8; dl[2][0, 0, 1] = 17.5e-9

        cre_q, cim_q, dl_q = quadriga_lib.channel.quantize_delays(
            cre, cim, dl, tap_spacing=5e-9)

        self.assertEqual(len(cre_q), 3)
        # All snapshots share same n_taps
        n_taps = cre_q[0].shape[2]
        self.assertEqual(cre_q[1].shape[2], n_taps)
        self.assertEqual(cre_q[2].shape[2], n_taps)
        self.assertGreater(n_taps, 0)

        # Snapshot 1: only 1 on-grid path at 5 ns
        nz_snap1 = sum(1 for k in range(n_taps) if abs(cre_q[1][0, 0, k]) > 1e-10)
        self.assertEqual(nz_snap1, 1)

        # Verify snap 1 coefficient
        found = False
        for k in range(n_taps):
            if abs(dl_q[1][0, 0, k] - 5e-9) < 1e-20:
                npt.assert_allclose(cre_q[1][0, 0, k], 3.0, atol=1e-12, rtol=0)
                found = True
        self.assertTrue(found)

    def test_variable_n_path_fix1(self):
        """Variable n_path with fix_taps=1 produces uniform grid"""
        cre = [np.zeros((1, 1, 3)), np.zeros((1, 1, 1))]
        cim = [np.zeros((1, 1, 3)), np.zeros((1, 1, 1))]
        dl = [np.zeros((1, 1, 3)), np.zeros((1, 1, 1))]

        cre[0][0, 0, 0] = 1.0; dl[0][0, 0, 0] = 0.0
        cre[0][0, 0, 1] = 1.0; dl[0][0, 0, 1] = 10.0e-9
        cre[0][0, 0, 2] = 1.0; dl[0][0, 0, 2] = 25.0e-9

        cre[1][0, 0, 0] = 2.0; dl[1][0, 0, 0] = 15.0e-9

        cre_q, cim_q, dl_q = quadriga_lib.channel.quantize_delays(
            cre, cim, dl, tap_spacing=5e-9, fix_taps=1)

        n_taps = dl_q[0].shape[2]
        # Both snapshots should have identical delay grids
        for k in range(n_taps):
            npt.assert_allclose(dl_q[0][0, 0, k], dl_q[1][0, 0, k], atol=1e-20, rtol=0)

    def test_fix_taps_1_uniform_grid(self):
        """fix_taps=1 produces uniform delay grid across all snapshots"""
        cre = [np.ones((2, 1, 2)), np.ones((2, 1, 2))]
        cim = [np.zeros((2, 1, 2)), np.zeros((2, 1, 2))]
        dl = [np.zeros((2, 1, 2)), np.zeros((2, 1, 2))]

        for s in range(2):
            dl[s][0, 0, 0] = 0.0
            dl[s][0, 0, 1] = 12.5e-9
            dl[s][1, 0, 0] = 5.0e-9
            dl[s][1, 0, 1] = 27.5e-9

        cre_q, cim_q, dl_q = quadriga_lib.channel.quantize_delays(
            cre, cim, dl, tap_spacing=5e-9, fix_taps=1)

        n_taps = dl_q[0].shape[2]
        for k in range(n_taps):
            npt.assert_allclose(dl_q[0][0, 0, k], dl_q[1][0, 0, k], atol=1e-20, rtol=0)

    def test_paths_combining(self):
        """Two paths at same delay have coefficients summed"""
        cre = [np.zeros((1, 1, 2))]
        cim = [np.zeros((1, 1, 2))]
        dl = [np.zeros((1, 1, 2))]

        cre[0][0, 0, 0] = 1.0; dl[0][0, 0, 0] = 10.0e-9
        cre[0][0, 0, 1] = 2.0; dl[0][0, 0, 1] = 10.0e-9

        cre_q, cim_q, dl_q = quadriga_lib.channel.quantize_delays(
            cre, cim, dl, tap_spacing=5e-9)

        found = False
        for k in range(cre_q[0].shape[2]):
            if abs(dl_q[0][0, 0, k] - 10e-9) < 1e-20:
                npt.assert_allclose(cre_q[0][0, 0, k], 3.0, atol=1e-12, rtol=0)
                found = True
        self.assertTrue(found)

    def test_zero_delay(self):
        """Zero delay maps to tap 0"""
        cre = [np.array([[[1.0]]])]
        cim = [np.array([[[-1.0]]])]
        dl = [np.array([[[0.0]]])]

        cre_q, cim_q, dl_q = quadriga_lib.channel.quantize_delays(
            cre, cim, dl, tap_spacing=5e-9)

        npt.assert_allclose(cre_q[0][0, 0, 0], 1.0, atol=1e-12, rtol=0)
        npt.assert_allclose(cim_q[0][0, 0, 0], -1.0, atol=1e-12, rtol=0)
        npt.assert_allclose(dl_q[0][0, 0, 0], 0.0, atol=1e-20, rtol=0)

    def test_round_trip_already_quantized(self):
        """Feeding quantized output back in triggers already-quantized path and returns identical results"""
        # Create non-trivial multi-snapshot input with mixed on/off-grid delays
        cre = [np.zeros((2, 1, 3)), np.zeros((2, 1, 2))]
        cim = [np.zeros((2, 1, 3)), np.zeros((2, 1, 2))]
        dl = [np.zeros((2, 1, 3)), np.zeros((2, 1, 2))]

        cre[0][0, 0, 0] = 1.0; cim[0][0, 0, 0] = 0.5; dl[0][0, 0, 0] = 0.0
        cre[0][0, 0, 1] = 2.0; cim[0][0, 0, 1] = -1.0; dl[0][0, 0, 1] = 12.5e-9
        cre[0][0, 0, 2] = 0.5; cim[0][0, 0, 2] = 0.3; dl[0][0, 0, 2] = 25.0e-9
        cre[0][1, 0, 0] = 0.8; cim[0][1, 0, 0] = 0.2; dl[0][1, 0, 0] = 5.0e-9
        cre[0][1, 0, 1] = 1.5; cim[0][1, 0, 1] = -0.5; dl[0][1, 0, 1] = 17.5e-9
        cre[0][1, 0, 2] = 0.3; cim[0][1, 0, 2] = 0.1; dl[0][1, 0, 2] = 30.0e-9

        cre[1][0, 0, 0] = 3.0; cim[1][0, 0, 0] = 1.0; dl[1][0, 0, 0] = 10.0e-9
        cre[1][0, 0, 1] = 0.7; cim[1][0, 0, 1] = -0.3; dl[1][0, 0, 1] = 22.5e-9
        cre[1][1, 0, 0] = 1.2; cim[1][1, 0, 0] = 0.4; dl[1][1, 0, 0] = 0.0
        cre[1][1, 0, 1] = 0.9; cim[1][1, 0, 1] = -0.6; dl[1][1, 0, 1] = 35.0e-9

        # First pass: quantize
        cre_q1, cim_q1, dl_q1 = quadriga_lib.channel.quantize_delays(
            cre, cim, dl, tap_spacing=5e-9, max_no_taps=48, power_exponent=1.0, fix_taps=0)

        # Verify output delays are on-grid (multiples of tap_spacing)
        for s in range(len(dl_q1)):
            for k in range(dl_q1[s].shape[2]):
                for r in range(dl_q1[s].shape[0]):
                    for t in range(dl_q1[s].shape[1]):
                        d = dl_q1[s][r, t, k]
                        if d > 0:
                            tap_idx = d / 5e-9
                            npt.assert_allclose(tap_idx, round(tap_idx), atol=1e-6, rtol=0)

        # Second pass: feed quantized output back in
        cre_q2, cim_q2, dl_q2 = quadriga_lib.channel.quantize_delays(
            cre_q1, cim_q1, dl_q1, tap_spacing=5e-9, max_no_taps=48, power_exponent=1.0, fix_taps=0)

        # Output must be identical
        self.assertEqual(len(cre_q2), len(cre_q1))
        for s in range(len(cre_q1)):
            npt.assert_allclose(cre_q2[s], cre_q1[s], atol=1e-12, rtol=0)
            npt.assert_allclose(cim_q2[s], cim_q1[s], atol=1e-12, rtol=0)
            npt.assert_allclose(dl_q2[s], dl_q1[s], atol=1e-20, rtol=0)


if __name__ == '__main__':
    unittest.main()
