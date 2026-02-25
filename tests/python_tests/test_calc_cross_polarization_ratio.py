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

    def test_basic_nlos_xpr(self):
        """Basic NLOS XPR with 1 LOS + 2 NLOS paths"""
        tx = np.array([[0.0], [0.0], [0.0]])
        rx = np.array([[10.0], [0.0], [0.0]])

        pw = [np.array([1.0, 0.5, 0.5])]
        M_los = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]).reshape(8, 1)
        M_nlos1 = np.array([0.9, 0.0, 0.1, 0.0, 0.1, 0.0, 0.8, 0.0]).reshape(8, 1)
        M_nlos2 = np.array([0.7, 0.0, 0.3, 0.0, 0.2, 0.0, 0.6, 0.0]).reshape(8, 1)
        M = [np.hstack([M_los, M_nlos1, M_nlos2])]
        pl = [np.array([10.0, 12.0, 15.0])]

        xpr, pg = quadriga_lib.tools.calc_cross_polarization_ratio(pw, M, pl, tx, rx)

        self.assertEqual(xpr.shape, (1, 6))
        self.assertEqual(pg.shape, (1,))

        # pg includes all paths
        npt.assert_allclose(pg[0], 3.225, atol=1e-10, rtol=0)

        # V-XPR = 0.65 / 0.05 = 13.0
        npt.assert_allclose(xpr[0, 1], 13.0, atol=1e-10, rtol=0)

        # H-XPR = 0.50 / 0.025 = 20.0
        npt.assert_allclose(xpr[0, 2], 20.0, atol=1e-10, rtol=0)

        # Aggregate linear = 1.15 / 0.075
        npt.assert_allclose(xpr[0, 0], 1.15 / 0.075, atol=1e-10, rtol=0)

    def test_include_los(self):
        """Including LOS path changes XPR"""
        tx = np.array([[0.0], [0.0], [0.0]])
        rx = np.array([[10.0], [0.0], [0.0]])

        pw = [np.array([1.0, 0.5])]
        M_los = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]).reshape(8, 1)
        M_nlos = np.array([0.8, 0.0, 0.2, 0.0, 0.2, 0.0, 0.7, 0.0]).reshape(8, 1)
        M = [np.hstack([M_los, M_nlos])]
        pl = [np.array([10.0, 15.0])]

        # Without LOS
        xpr_no, _ = quadriga_lib.tools.calc_cross_polarization_ratio(pw, M, pl, tx, rx, False)

        # With LOS
        xpr_yes, _ = quadriga_lib.tools.calc_cross_polarization_ratio(pw, M, pl, tx, rx, True)

        # V-XPR without LOS: 0.5*0.64 / (0.5*0.04) = 16
        npt.assert_allclose(xpr_no[0, 1], 16.0, atol=1e-10, rtol=0)

        # V-XPR with LOS: (1.0 + 0.32) / 0.02 = 66
        npt.assert_allclose(xpr_yes[0, 1], 66.0, atol=1e-10, rtol=0)

    def test_complex_m_elements(self):
        """Complex-valued polarization matrix"""
        tx = np.array([[0.0], [0.0], [0.0]])
        rx = np.array([[10.0], [0.0], [0.0]])

        pw = [np.array([1.0])]
        M = [np.array([0.8, 0.2, 0.1, -0.1, 0.05, 0.05, 0.7, -0.3]).reshape(8, 1)]
        pl = [np.array([20.0])]

        xpr, pg = quadriga_lib.tools.calc_cross_polarization_ratio(pw, M, pl, tx, rx)

        abs2_vv = 0.8**2 + 0.2**2   # 0.68
        abs2_hv = 0.1**2 + 0.1**2   # 0.02
        abs2_vh = 0.05**2 + 0.05**2  # 0.005
        abs2_hh = 0.7**2 + 0.3**2   # 0.58

        npt.assert_allclose(pg[0], abs2_vv + abs2_hv + abs2_vh + abs2_hh, atol=1e-14, rtol=0)
        npt.assert_allclose(xpr[0, 1], abs2_vv / abs2_hv, atol=1e-10, rtol=0)
        npt.assert_allclose(xpr[0, 2], abs2_hh / abs2_vh, atol=1e-10, rtol=0)
        npt.assert_allclose(xpr[0, 0], (abs2_vv + abs2_hh) / (abs2_hv + abs2_vh), atol=1e-10, rtol=0)

    def test_multiple_cirs(self):
        """Multiple CIRs with mobile TX/RX"""
        tx = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        rx = np.array([[10.0, 11.0], [0.0, 0.0], [0.0, 0.0]])

        pw0 = np.array([1.0, 0.5])
        pw1 = np.array([0.8, 0.4])

        M0_c0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]).reshape(8, 1)
        M0_c1 = np.array([0.8, 0.0, 0.2, 0.0, 0.15, 0.0, 0.7, 0.0]).reshape(8, 1)
        M1_c0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]).reshape(8, 1)
        M1_c1 = np.array([0.6, 0.0, 0.3, 0.0, 0.25, 0.0, 0.5, 0.0]).reshape(8, 1)

        M = [np.hstack([M0_c0, M0_c1]), np.hstack([M1_c0, M1_c1])]
        pl = [np.array([10.0, 14.0]), np.array([10.0, 13.0])]

        xpr, pg = quadriga_lib.tools.calc_cross_polarization_ratio([pw0, pw1], M, pl, tx, rx)

        self.assertEqual(xpr.shape, (2, 6))
        self.assertEqual(pg.shape, (2,))

        # CIR 0 V-XPR: 0.5*0.64 / (0.5*0.04) = 16
        npt.assert_allclose(xpr[0, 1], 0.32 / 0.02, atol=1e-10, rtol=0)

        # CIR 1 V-XPR: 0.4*0.36 / (0.4*0.09) = 4.0
        npt.assert_allclose(xpr[1, 1], 0.144 / 0.036, atol=1e-10, rtol=0)

    def test_circular_xpr_diagonal_m(self):
        """Diagonal M with M_hh=-1 should have zero circular co-pol"""
        tx = np.array([[0.0], [0.0], [0.0]])
        rx = np.array([[5.0], [0.0], [0.0]])

        pw = [np.array([1.0])]
        M = [np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]).reshape(8, 1)]
        pl = [np.array([20.0])]

        xpr, _ = quadriga_lib.tools.calc_cross_polarization_ratio(pw, M, pl, tx, rx)

        # All XPR values should be 0 (undefined: either zero numerator or zero denominator)
        npt.assert_allclose(xpr[0, :], 0.0, atol=1e-14, rtol=0)

    def test_window_size_effect(self):
        """Window size controls which near-LOS paths are excluded"""
        tx = np.array([[0.0], [0.0], [0.0]])
        rx = np.array([[10.0], [0.0], [0.0]])

        pw = [np.array([1.0, 0.8, 0.5])]
        M_los = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]).reshape(8, 1)
        M_near = np.array([0.9, 0.0, 0.15, 0.0, 0.1, 0.0, 0.85, 0.0]).reshape(8, 1)
        M_far = np.array([0.7, 0.0, 0.3, 0.0, 0.2, 0.0, 0.6, 0.0]).reshape(8, 1)
        M = [np.hstack([M_los, M_near, M_far])]
        pl = [np.array([10.0, 10.005, 12.0])]

        # Large window: excludes paths 0 and 1
        xpr_large, _ = quadriga_lib.tools.calc_cross_polarization_ratio(
            pw, M, pl, tx, rx, False, 0.01)

        # Only path 2: V-XPR = 0.49/0.09
        npt.assert_allclose(xpr_large[0, 1], 0.49 / 0.09, atol=1e-10, rtol=0)

    def test_error_empty_powers(self):
        """Empty powers list should raise error"""
        tx = np.array([[0.0], [0.0], [0.0]])
        rx = np.array([[10.0], [0.0], [0.0]])
        with self.assertRaises(Exception):
            quadriga_lib.tools.calc_cross_polarization_ratio([], [], [], tx, rx)

    def test_equal_linear_xpr(self):
        """Equal M elements give XPR = 1 in linear basis"""
        tx = np.array([[0.0], [0.0], [0.0]])
        rx = np.array([[10.0], [0.0], [0.0]])

        pw = [np.array([1.0])]
        M = [np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0]).reshape(8, 1)]
        pl = [np.array([20.0])]

        xpr, _ = quadriga_lib.tools.calc_cross_polarization_ratio(pw, M, pl, tx, rx)

        npt.assert_allclose(xpr[0, 0], 1.0, atol=1e-10, rtol=0)
        npt.assert_allclose(xpr[0, 1], 1.0, atol=1e-10, rtol=0)
        npt.assert_allclose(xpr[0, 2], 1.0, atol=1e-10, rtol=0)


if __name__ == '__main__':
    unittest.main()
