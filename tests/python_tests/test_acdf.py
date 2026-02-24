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


class test_acdf(unittest.TestCase):

    def test_basic_single_column(self):
        data = np.arange(10, dtype=np.float64).reshape(10, 1)
        result = quadriga_lib.tools.acdf(data)

        bins = result["bins"]
        Sh = result["Sh"]
        Sc = result["Sc"]
        mu = result["mu"]
        sig = result["sig"]

        # Bins should be auto-generated with 201 points from 0 to 9
        self.assertEqual(bins.shape[0], 201)
        npt.assert_allclose(bins[0], 0.0, atol=1e-10, rtol=0)
        npt.assert_allclose(bins[-1], 9.0, atol=1e-10, rtol=0)

        # Sh should be (201, 1)
        self.assertEqual(Sh.shape, (201, 1))

        # CDF should end at 1.0
        npt.assert_allclose(Sh[-1, 0], 1.0, atol=1e-10, rtol=0)

        # CDF should be non-decreasing
        self.assertTrue(np.all(np.diff(Sh[:, 0]) >= 0))

        # Sc should equal Sh for single column
        self.assertEqual(Sc.shape[0], 201)
        npt.assert_allclose(Sc, Sh[:, 0], atol=1e-10, rtol=0)

        # mu and sig shapes
        self.assertEqual(mu.shape[0], 9)
        self.assertEqual(sig.shape[0], 9)

        # sig should be all zeros for single column
        npt.assert_allclose(sig, np.zeros(9), atol=1e-10, rtol=0)

    def test_multiple_columns(self):
        data = np.column_stack([np.arange(100, dtype=np.float64),
                                np.arange(100, dtype=np.float64)])
        result = quadriga_lib.tools.acdf(data)

        Sh = result["Sh"]
        Sc = result["Sc"]
        sig = result["sig"]

        self.assertEqual(Sh.shape, (201, 2))

        # Both columns should have the same CDF
        npt.assert_allclose(Sh[:, 0], Sh[:, 1], atol=1e-10, rtol=0)

        # Sc should be close to 1.0 at last bin (quantile grid tops at 0.999)
        npt.assert_allclose(Sc[-1], 1.0, atol=0.02, rtol=0)

        # sig reflects variation within quantile window, not across sets
        for i in range(9):
            self.assertLess(sig[i], 1.0)

    def test_custom_bins(self):
        data = np.arange(100, dtype=np.float64).reshape(100, 1)
        bins_in = np.array([0.0, 25.0, 50.0, 75.0, 99.0])
        result = quadriga_lib.tools.acdf(data, bins=bins_in)

        bins = result["bins"]
        Sh = result["Sh"]

        self.assertEqual(bins.shape[0], 5)
        self.assertEqual(Sh.shape, (5, 1))

        # CDF at last bin should be 1.0
        npt.assert_allclose(Sh[-1, 0], 1.0, atol=1e-10, rtol=0)

    def test_handles_inf_nan(self):
        data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.inf, np.nan],
                        dtype=np.float64).reshape(12, 1)
        result = quadriga_lib.tools.acdf(data)

        bins = result["bins"]
        Sh = result["Sh"]

        # Bins should span 0 to 9
        npt.assert_allclose(bins[0], 0.0, atol=1e-10, rtol=0)
        npt.assert_allclose(bins[-1], 9.0, atol=1e-10, rtol=0)

        # CDF should reach 1.0
        npt.assert_allclose(Sh[-1, 0], 1.0, atol=1e-10, rtol=0)

    def test_custom_n_bins(self):
        data = np.arange(100, dtype=np.float64).reshape(100, 1)
        result = quadriga_lib.tools.acdf(data, n_bins=51)

        self.assertEqual(result["bins"].shape[0], 51)
        self.assertEqual(result["Sh"].shape[0], 51)

    def test_quantile_correctness(self):
        data = np.arange(1000, dtype=np.float64).reshape(1000, 1)
        result = quadriga_lib.tools.acdf(data)

        mu = result["mu"]
        self.assertEqual(mu.shape[0], 9)

        # For uniform [0, 999], the p-th quantile ~ p * 999
        for q in range(9):
            expected = (q + 1) * 0.1 * 999.0
            self.assertLess(abs(mu[q] - expected), 10.0)

    def test_error_empty_data(self):
        with self.assertRaises(Exception):
            quadriga_lib.tools.acdf(np.empty((0, 0), dtype=np.float64))

    def test_error_n_bins_too_small(self):
        data = np.arange(10, dtype=np.float64).reshape(10, 1)
        with self.assertRaises(Exception):
            quadriga_lib.tools.acdf(data, n_bins=1)


if __name__ == '__main__':
    unittest.main()
