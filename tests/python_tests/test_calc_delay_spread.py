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


class TestCalcDelaySpread(unittest.TestCase):

    def test_basic_single_cir(self):
        """Basic single CIR with 3 paths"""
        delays = [np.array([0.0, 1e-6, 2e-6])]
        powers = [np.array([1.0, 0.5, 0.25])]

        ds, mean_delay = quadriga_lib.tools.calc_delay_spread(delays, powers)

        self.assertEqual(ds.shape, (1,))
        self.assertEqual(mean_delay.shape, (1,))

        # Expected mean delay: (1.0*0 + 0.5*1e-6 + 0.25*2e-6) / 1.75
        expected_mean = 1.0e-6 / 1.75
        npt.assert_allclose(mean_delay[0], expected_mean, atol=1e-14, rtol=0)

        # Delay spread must be positive and less than max delay
        self.assertGreater(ds[0], 0.0)
        self.assertLess(ds[0], 2e-6)

    def test_single_path_zero_spread(self):
        """Single path yields zero delay spread"""
        delays = [np.array([5e-6])]
        powers = [np.array([1.0])]

        ds, mean_delay = quadriga_lib.tools.calc_delay_spread(delays, powers)

        npt.assert_allclose(ds[0], 0.0, atol=1e-20, rtol=0)
        npt.assert_allclose(mean_delay[0], 5e-6, atol=1e-20, rtol=0)

    def test_equal_power_two_paths(self):
        """Two equal-power paths: DS = half the delay difference"""
        delays = [np.array([0.0, 2e-6])]
        powers = [np.array([1.0, 1.0])]

        ds, mean_delay = quadriga_lib.tools.calc_delay_spread(delays, powers)

        npt.assert_allclose(mean_delay[0], 1e-6, atol=1e-14, rtol=0)
        npt.assert_allclose(ds[0], 1e-6, atol=1e-14, rtol=0)

    def test_multiple_cirs(self):
        """Multiple CIRs with different path counts"""
        delays = [np.array([0.0, 1e-6]), np.array([0.0, 1e-6, 2e-6])]
        powers = [np.array([1.0, 1.0]), np.array([1.0, 1.0, 1.0])]

        ds, mean_delay = quadriga_lib.tools.calc_delay_spread(delays, powers)

        self.assertEqual(ds.shape, (2,))
        self.assertEqual(mean_delay.shape, (2,))

        # CIR 0: mean = 0.5e-6, DS = 0.5e-6
        npt.assert_allclose(mean_delay[0], 0.5e-6, atol=1e-14, rtol=0)
        npt.assert_allclose(ds[0], 0.5e-6, atol=1e-14, rtol=0)

        # CIR 1: mean = 1e-6
        npt.assert_allclose(mean_delay[1], 1e-6, atol=1e-14, rtol=0)
        self.assertGreater(ds[1], 0.0)

    def test_threshold_filters_weak_paths(self):
        """Threshold excludes weak paths"""
        delays = [np.array([0.0, 10e-6, 1e-6])]
        powers = [np.array([1.0, 0.001, 0.5])]

        # 20 dB threshold -> path at -30 dB excluded
        ds_20, _ = quadriga_lib.tools.calc_delay_spread(delays, powers, threshold=20.0)

        # 100 dB threshold -> all paths included
        ds_all, _ = quadriga_lib.tools.calc_delay_spread(delays, powers, threshold=100.0)

        self.assertGreater(ds_all[0], ds_20[0])

    def test_granularity_bins_paths(self):
        """Granularity groups nearby paths"""
        delays = [np.array([100e-9, 110e-9, 1000e-9])]
        powers = [np.array([1.0, 1.0, 1.0])]

        ds_no_gran, _ = quadriga_lib.tools.calc_delay_spread(delays, powers, granularity=0.0)
        ds_gran, _ = quadriga_lib.tools.calc_delay_spread(delays, powers, granularity=50e-9)

        self.assertGreater(ds_no_gran[0], 0.0)
        self.assertGreater(ds_gran[0], 0.0)

    def test_input_validation_empty(self):
        """Empty input raises error"""
        with self.assertRaises(Exception):
            quadriga_lib.tools.calc_delay_spread([], [np.array([1.0])])

    def test_input_validation_mismatched_sizes(self):
        """Mismatched delays/powers sizes raises error"""
        with self.assertRaises(Exception):
            quadriga_lib.tools.calc_delay_spread(
                [np.array([0.0]), np.array([1e-6])],
                [np.array([1.0])]
            )

    def test_input_validation_mismatched_path_counts(self):
        """Mismatched path counts within a CIR raises error"""
        with self.assertRaises(Exception):
            quadriga_lib.tools.calc_delay_spread(
                [np.array([0.0, 1e-6])],
                [np.array([1.0])]
            )

    def test_granularity_with_mean_delay(self):
        """Granularity produces correct mean delay"""
        delays = [np.array([0.0, 50e-9, 500e-9, 550e-9])]
        powers = [np.array([1.0, 1.0, 1.0, 1.0])]

        ds, mean_delay = quadriga_lib.tools.calc_delay_spread(delays, powers, granularity=100e-9)

        self.assertGreater(mean_delay[0], 0.0)
        self.assertGreater(ds[0], 0.0)


if __name__ == '__main__':
    unittest.main()
