## SPDX-License-Identifier: Apache-2.0
##
## quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
## Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
## ------------------------------------------------------------------------

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


class test_case(unittest.TestCase):

    def test_basic_functionality(self):
        """Single snapshot: 3 paths, TX at origin, RX at (10, 0, 0)"""
        powers = [np.array([1.0, 0.5, 0.25])]
        path_length = [np.array([10.0, 11.0, 12.0])]
        tx_pos = np.array([[0.0], [0.0], [0.0]])
        rx_pos = np.array([[10.0], [0.0], [0.0]])

        kf, pg = quadriga_lib.tools.calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, 0.01)

        self.assertEqual(kf.shape, (1,))
        self.assertEqual(pg.shape, (1,))
        npt.assert_allclose(kf[0], 1.0 / 0.75, atol=1e-10, rtol=0)
        npt.assert_allclose(pg[0], 1.75, atol=1e-10, rtol=0)

    def test_multiple_los_paths(self):
        """Two paths within LOS window"""
        powers = [np.array([2.0, 1.0, 0.5])]
        path_length = [np.array([10.0, 10.005, 15.0])]
        tx_pos = np.array([[0.0], [0.0], [0.0]])
        rx_pos = np.array([[10.0], [0.0], [0.0]])

        kf, pg = quadriga_lib.tools.calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, 0.01)

        npt.assert_allclose(kf[0], 3.0 / 0.5, atol=1e-10, rtol=0)
        npt.assert_allclose(pg[0], 3.5, atol=1e-10, rtol=0)

    def test_no_nlos_paths(self):
        """All paths within LOS window => infinite K-Factor"""
        powers = [np.array([1.0, 0.5])]
        path_length = [np.array([10.0, 10.005])]
        tx_pos = np.array([[0.0], [0.0], [0.0]])
        rx_pos = np.array([[10.0], [0.0], [0.0]])

        kf, pg = quadriga_lib.tools.calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, 0.01)

        self.assertTrue(np.isinf(kf[0]))
        self.assertGreater(kf[0], 0.0)

    def test_no_los_paths(self):
        """All paths beyond LOS window => zero K-Factor"""
        powers = [np.array([1.0, 0.5])]
        path_length = [np.array([11.0, 12.0])]
        tx_pos = np.array([[0.0], [0.0], [0.0]])
        rx_pos = np.array([[10.0], [0.0], [0.0]])

        kf, pg = quadriga_lib.tools.calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, 0.01)

        self.assertEqual(kf[0], 0.0)
        npt.assert_allclose(pg[0], 1.5, atol=1e-10, rtol=0)

    def test_multiple_snapshots_mobile_rx(self):
        """2 snapshots, fixed TX, mobile RX"""
        powers = [np.array([1.0, 0.5]), np.array([2.0, 1.0])]
        path_length = [np.array([10.0, 12.0]), np.array([20.0, 25.0])]
        tx_pos = np.array([[0.0], [0.0], [0.0]])
        rx_pos = np.array([[10.0, 20.0], [0.0, 0.0], [0.0, 0.0]])

        kf, pg = quadriga_lib.tools.calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, 0.01)

        self.assertEqual(kf.shape, (2,))
        self.assertEqual(pg.shape, (2,))
        npt.assert_allclose(kf[0], 1.0 / 0.5, atol=1e-10, rtol=0)
        npt.assert_allclose(kf[1], 2.0 / 1.0, atol=1e-10, rtol=0)
        npt.assert_allclose(pg[0], 1.5, atol=1e-10, rtol=0)
        npt.assert_allclose(pg[1], 3.0, atol=1e-10, rtol=0)

    def test_3d_positions(self):
        """TX and RX in 3D space"""
        powers = [np.array([1.0, 0.5])]
        path_length = [np.array([5.0, 8.0])]
        tx_pos = np.array([[0.0], [0.0], [0.0]])
        rx_pos = np.array([[3.0], [4.0], [0.0]])  # dTR = 5.0

        kf, pg = quadriga_lib.tools.calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, 0.01)

        npt.assert_allclose(kf[0], 1.0 / 0.5, atol=1e-10, rtol=0)

    def test_custom_window_size(self):
        """Different window sizes change LOS classification"""
        powers = [np.array([1.0, 0.5, 0.25])]
        path_length = [np.array([10.0, 10.5, 12.0])]
        tx_pos = np.array([[0.0], [0.0], [0.0]])
        rx_pos = np.array([[10.0], [0.0], [0.0]])

        # Small window: only first path is LOS
        kf, _ = quadriga_lib.tools.calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, 0.01)
        npt.assert_allclose(kf[0], 1.0 / 0.75, atol=1e-10, rtol=0)

        # Large window: first two paths are LOS
        kf, _ = quadriga_lib.tools.calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, 1.0)
        npt.assert_allclose(kf[0], 1.5 / 0.25, atol=1e-10, rtol=0)

    def test_empty_snapshot(self):
        """Snapshot with no paths"""
        powers = [np.array([])]
        path_length = [np.array([])]
        tx_pos = np.array([[0.0], [0.0], [0.0]])
        rx_pos = np.array([[10.0], [0.0], [0.0]])

        kf, pg = quadriga_lib.tools.calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, 0.01)

        self.assertEqual(kf[0], 0.0)
        self.assertEqual(pg[0], 0.0)

    def test_error_empty_powers(self):
        """Empty powers list should raise error"""
        with self.assertRaises(Exception):
            quadriga_lib.tools.calc_rician_k_factor([], [], np.zeros((3, 1)), np.zeros((3, 1)), 0.01)

    def test_error_mismatched_sizes(self):
        """Mismatched powers and path_length list sizes"""
        powers = [np.array([1.0]), np.array([2.0])]
        path_length = [np.array([10.0])]
        tx_pos = np.array([[0.0], [0.0], [0.0]])
        rx_pos = np.array([[10.0], [0.0], [0.0]])

        with self.assertRaises(Exception):
            quadriga_lib.tools.calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, 0.01)

    def test_error_wrong_tx_pos_shape(self):
        """Wrong tx_pos shape should raise error"""
        powers = [np.array([1.0])]
        path_length = [np.array([10.0])]
        tx_pos = np.array([[0.0], [0.0]])  # 2 rows instead of 3
        rx_pos = np.array([[10.0], [0.0], [0.0]])

        with self.assertRaises(Exception):
            quadriga_lib.tools.calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, 0.01)


if __name__ == '__main__':
    unittest.main()
