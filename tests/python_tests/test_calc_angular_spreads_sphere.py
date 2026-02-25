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


class test_case(unittest.TestCase):

    def test_single_path_zero_spread(self):
        az = [np.array([0.5])]
        el = [np.array([0.3])]
        powers = [np.array([1.0])]

        a_s, e_s, orient, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(az, el, powers)

        self.assertEqual(a_s.shape, (1,))
        npt.assert_allclose(a_s[0], 0.0, atol=1e-10, rtol=0)
        npt.assert_allclose(e_s[0], 0.0, atol=1e-10, rtol=0)
        npt.assert_allclose(orient[2, 0], 0.5, atol=1e-10, rtol=0)
        npt.assert_allclose(orient[1, 0], 0.3, atol=1e-10, rtol=0)

    def test_symmetric_azimuth_paths(self):
        az = [np.array([0.1, -0.1])]
        el = [np.array([0.0, 0.0])]
        powers = [np.array([1.0, 1.0])]

        a_s, e_s, orient, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(az, el, powers)

        npt.assert_allclose(a_s[0], 0.1, atol=1e-6, rtol=0)
        npt.assert_allclose(e_s[0], 0.0, atol=1e-6, rtol=0)
        npt.assert_allclose(orient[2, 0], 0.0, atol=1e-6, rtol=0)

    def test_symmetric_elevation_paths(self):
        az = [np.array([0.0, 0.0])]
        el = [np.array([0.2, -0.2])]
        powers = [np.array([1.0, 1.0])]

        a_s, e_s, orient, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(az, el, powers)

        self.assertGreaterEqual(a_s[0] + 1e-6, e_s[0])
        total = np.sqrt(a_s[0]**2 + e_s[0]**2)
        npt.assert_allclose(total, 0.2, atol=1e-4, rtol=0)

    def test_pole_paths(self):
        az = [np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])]
        el = [np.array([1.4, 1.4, 1.4, 1.4])]
        powers = [np.array([1.0, 1.0, 1.0, 1.0])]

        a_s, e_s, orient, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(az, el, powers)

        self.assertLess(a_s[0], 0.5)
        self.assertLess(e_s[0], 0.5)

    def test_variable_path_counts(self):
        az = [np.array([0.1, -0.1]),
              np.array([0.2, -0.2, 0.0]),
              np.array([1.0, -1.0, 0.5, -0.5, 0.0])]
        el = [np.array([0.0, 0.0]),
              np.array([0.0, 0.0, 0.0]),
              np.array([0.0, 0.0, 0.0, 0.0, 0.0])]
        powers = [np.array([1.0, 1.0]),
                  np.array([1.0, 1.0, 1.0]),
                  np.array([1.0, 1.0, 1.0, 1.0, 1.0])]

        a_s, e_s, orient, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(az, el, powers)

        self.assertEqual(a_s.shape, (3,))
        self.assertEqual(len(phi), 3)
        self.assertEqual(phi[0].shape, (2,))
        self.assertEqual(phi[1].shape, (3,))
        self.assertEqual(phi[2].shape, (5,))
        self.assertLess(a_s[0], a_s[1])
        self.assertLess(a_s[1], a_s[2])

    def test_no_bank_angle(self):
        az = [np.array([0.0, 0.0])]
        el = [np.array([0.2, -0.2])]
        powers = [np.array([1.0, 1.0])]

        a_bank, e_bank, o_bank, _, _ = quadriga_lib.tools.calc_angular_spreads_sphere(
            az, el, powers, calc_bank_angle=True)
        a_nobank, e_nobank, o_nobank, _, _ = quadriga_lib.tools.calc_angular_spreads_sphere(
            az, el, powers, calc_bank_angle=False)

        npt.assert_allclose(o_nobank[0, 0], 0.0, atol=1e-10, rtol=0)
        self.assertGreaterEqual(a_bank[0] + 1e-6, a_nobank[0])

    def test_disable_wrapping(self):
        az = [np.array([0.1, -0.1, 0.05])]
        el = [np.array([0.2, -0.2, 0.0])]
        powers = [np.array([1.0, 1.0, 1.0])]

        a_s, e_s, orient, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(
            az, el, powers, disable_wrapping=True)

        # Orientation should be zero
        npt.assert_allclose(orient[:, 0], [0.0, 0.0, 0.0], atol=1e-10, rtol=0)
        # phi/theta should equal input
        npt.assert_allclose(np.asarray(phi[0]), az[0], atol=1e-14, rtol=0)
        npt.assert_allclose(np.asarray(theta[0]), el[0], atol=1e-14, rtol=0)

    def test_quantization(self):
        az = [np.array([0.0, 0.01])]
        el = [np.array([0.0, 0.0])]
        powers = [np.array([1.0, 1.0])]

        a_raw, _, _, _, _ = quadriga_lib.tools.calc_angular_spreads_sphere(
            az, el, powers, quantize=0.0)
        a_quant, _, _, _, _ = quadriga_lib.tools.calc_angular_spreads_sphere(
            az, el, powers, quantize=3.0)

        self.assertLessEqual(a_quant[0], a_raw[0] + 1e-8)

    def test_wrap_around_pi(self):
        az = [np.array([3.0, -3.0])]
        el = [np.array([0.0, 0.0])]
        powers = [np.array([1.0, 1.0])]

        a_s, e_s, orient, _, _ = quadriga_lib.tools.calc_angular_spreads_sphere(az, el, powers)

        gap = 2.0 * (np.pi - 3.0)
        self.assertLess(a_s[0], gap)
        self.assertGreater(a_s[0], 0.0)

    def test_output_shapes(self):
        az = [np.array([0.3, -0.2, 0.1, -0.4, 0.0])]
        el = [np.array([0.1, -0.1, 0.05, -0.05, 0.0])]
        powers = [np.array([1.0, 1.0, 1.0, 1.0, 1.0])]

        a_s, e_s, orient, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(az, el, powers)

        self.assertEqual(a_s.shape, (1,))
        self.assertEqual(e_s.shape, (1,))
        self.assertEqual(orient.shape, (3, 1))
        self.assertEqual(len(phi), 1)
        self.assertEqual(phi[0].shape, (5,))
        self.assertEqual(len(theta), 1)
        self.assertEqual(theta[0].shape, (5,))

    def test_rotated_angles_valid_range(self):
        az = [np.array([0.3, -0.2, 0.1, -0.4, 0.0])]
        el = [np.array([0.1, -0.1, 0.05, -0.05, 0.0])]
        powers = [np.array([1.0, 1.0, 1.0, 1.0, 1.0])]

        _, _, _, phi, theta = quadriga_lib.tools.calc_angular_spreads_sphere(az, el, powers)

        phi_arr = np.asarray(phi[0])
        theta_arr = np.asarray(theta[0])
        self.assertTrue(np.all(phi_arr >= -np.pi - 0.01))
        self.assertTrue(np.all(phi_arr <= np.pi + 0.01))
        self.assertTrue(np.all(theta_arr >= -np.pi / 2 - 0.01))
        self.assertTrue(np.all(theta_arr <= np.pi / 2 + 0.01))


if __name__ == '__main__':
    unittest.main()
