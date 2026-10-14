# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.tools.scenpar_validate.
# Scope: argument and output handling, defaults, dict conversion, error paths.
# Physics validation lives in the Catch2 suite (test_scenpar_validate.cpp).

import sys
import os
import copy
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


class test_case(unittest.TestCase):

    def setUp(self):
        # Distinct values; asD_ds is the (DS, ASD) entry, esA_ds the (DS, ESA) entry of lsp_xcorr
        self.scenpar = {"DS_mu": -7.5, "NumClusters": 12.0, "asD_ds": 0.25, "esA_ds": -0.5,
                        "SubpathMethod": "Laplacian"}

    def test_basic_call(self):
        out = quadriga_lib.tools.scenpar_validate(self.scenpar)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        scenpar_out, lsp_xcorr, lsp_xcorr_chk = out

        # 166 entries, float values, 'SubpathMethod' (str) at position 106
        self.assertIsInstance(scenpar_out, dict)
        self.assertEqual(len(scenpar_out), 166)
        keys = list(scenpar_out)
        self.assertEqual(keys[105:108], ["absTOA_lambda", "SubpathMethod", "LOS_scatter_radius"])
        for key, value in scenpar_out.items():
            self.assertIsInstance(value, str if key == "SubpathMethod" else float, key)

        # Given values are kept, missing keys have their default
        for key, value in self.scenpar.items():
            self.assertEqual(scenpar_out[key], value, key)
        self.assertEqual(scenpar_out["r_DS"], 2.0)
        self.assertEqual(scenpar_out["ES_D_mu_min"], -np.inf)

        # lsp_xcorr: symmetric, unit diagonal, the two correlations at their places
        self.assertIsInstance(lsp_xcorr, np.ndarray)
        self.assertEqual(lsp_xcorr.shape, (8, 8))
        self.assertEqual(lsp_xcorr.dtype, np.float64)
        expected = np.eye(8)
        expected[0, 3] = expected[3, 0] = 0.25
        expected[0, 6] = expected[6, 0] = -0.5
        npt.assert_array_equal(lsp_xcorr, expected)

        self.assertIsInstance(lsp_xcorr_chk, bool)
        self.assertTrue(lsp_xcorr_chk)

    def test_default_set(self):
        # Omitted, None and {} give the same complete default set
        ref = quadriga_lib.tools.scenpar_validate()
        for scenpar_in in [None, {}]:
            out = quadriga_lib.tools.scenpar_validate(scenpar_in)
            self.assertEqual(out[0], ref[0])
            npt.assert_array_equal(out[1], ref[1])
            self.assertEqual(out[2], ref[2])
        self.assertEqual(len(ref[0]), 166)
        self.assertEqual(ref[0]["SubpathMethod"], "legacy")
        self.assertEqual(ref[0]["NumClusters"], 6.0)
        npt.assert_array_equal(ref[1], np.eye(8))
        self.assertTrue(ref[2])

    def test_keyword_arguments(self):
        ref = quadriga_lib.tools.scenpar_validate(self.scenpar, True)
        out = quadriga_lib.tools.scenpar_validate(scenpar_in=self.scenpar, validate=True)
        self.assertEqual(out[0], ref[0])
        npt.assert_array_equal(out[1], ref[1])
        self.assertEqual(out[2], ref[2])

    def test_validate_flag_types(self):
        # None selects the default (True); bool and numpy.bool_ are accepted; anything else is a TypeError
        bad = {"r_DS": -1.0}
        for args, kwargs in [((bad, None), {}), ((bad,), {"validate": None}), ((bad, np.bool_(True)), {})]:
            with self.assertRaises(ValueError) as ctx:
                quadriga_lib.tools.scenpar_validate(*args, **kwargs)
            self.assertIn("r_DS", str(ctx.exception))
        self.assertEqual(quadriga_lib.tools.scenpar_validate(bad, np.bool_(False))[0]["r_DS"], -1.0)

        ref = quadriga_lib.tools.scenpar_validate(self.scenpar)
        out = quadriga_lib.tools.scenpar_validate(self.scenpar, None)
        self.assertEqual(out[0], ref[0])
        npt.assert_array_equal(out[1], ref[1])
        self.assertEqual(out[2], ref[2])

        for value in ["yes", 1, 0.0]:
            with self.assertRaises(TypeError) as ctx:
                quadriga_lib.tools.scenpar_validate(bad, value)
            self.assertIn("Input 'validate' must be a bool or None.", str(ctx.exception))

    def test_validate(self):
        # Default is validate = True
        bad = {"DS_mu": np.nan, "SubpathMethod": "foo"}
        with self.assertRaises(ValueError):
            quadriga_lib.tools.scenpar_validate(bad)
        with self.assertRaises(ValueError):
            quadriga_lib.tools.scenpar_validate(bad, True)

        # validate = False accepts NaN and any SubpathMethod; NaN in a correlation gives lsp_xcorr_chk = False
        scenpar_out, _, lsp_xcorr_chk = quadriga_lib.tools.scenpar_validate(bad, False)
        self.assertTrue(np.isnan(scenpar_out["DS_mu"]))
        self.assertEqual(scenpar_out["SubpathMethod"], "foo")
        self.assertTrue(lsp_xcorr_chk)
        _, _, lsp_xcorr_chk = quadriga_lib.tools.scenpar_validate({"asD_ds": np.nan}, False)
        self.assertFalse(lsp_xcorr_chk)

    def test_not_positive_definite(self):
        # DS-ASD and DS-ASA strongly positive, ASD-ASA strongly negative: valid values, no valid matrix
        scenpar_out, lsp_xcorr, lsp_xcorr_chk = quadriga_lib.tools.scenpar_validate(
            {"asD_ds": 0.9, "asA_ds": 0.9, "asD_asA": -0.9})
        self.assertIsInstance(lsp_xcorr_chk, bool)
        self.assertFalse(lsp_xcorr_chk)
        self.assertEqual(lsp_xcorr[3, 4], -0.9)

    def test_value_types(self):
        # Every real scalar form gives the same result as float
        ref = quadriga_lib.tools.scenpar_validate({"DS_mu": -7.5, "NumClusters": 12.0, "GR_enabled": 1.0})[0]
        variants = [
            {"DS_mu": np.float32(-7.5), "NumClusters": 12, "GR_enabled": True},
            {"DS_mu": np.array(-7.5), "NumClusters": np.int64(12), "GR_enabled": np.bool_(True)},
            {"DS_mu": np.array([[-7.5]]), "NumClusters": np.uint32(12), "GR_enabled": [1]},
        ]
        for scenpar_in in variants:
            out = quadriga_lib.tools.scenpar_validate(scenpar_in)[0]
            self.assertEqual(out, ref)
            for key in ["DS_mu", "NumClusters", "GR_enabled"]:
                self.assertIsInstance(out[key], float)

    def test_inputs_not_modified(self):
        scenpar = copy.deepcopy(self.scenpar)
        quadriga_lib.tools.scenpar_validate(scenpar)
        self.assertEqual(scenpar, self.scenpar)
        self.assertEqual(list(scenpar), list(self.scenpar))

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.scenpar_validate({"SNS_mu": 2.0})
        self.assertIn("SNS_mu", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.scenpar_validate({"SubpathMethod": "foo"})
        self.assertIn("SubpathMethod", str(ctx.exception))

        # Unknown keys are an error also with validate = False, GR_epsilon included
        for validate in [True, False]:
            with self.assertRaises(ValueError) as ctx:
                quadriga_lib.tools.scenpar_validate({"GR_epsilon": 1.0}, validate)
            self.assertIn("Unknown key 'GR_epsilon'", str(ctx.exception))

        # Thrown by the wrapper (dict conversion)
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.scenpar_validate([("DS_mu", 1.0)])
        self.assertIn("Input 'scenpar_in' must be a dict or None", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.scenpar_validate({1: 1.0})
        self.assertIn("Keys of 'scenpar_in' must be str", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.scenpar_validate({"SubpathMethod": 1})
        self.assertIn("Entry 'SubpathMethod' of 'scenpar_in' must be a str", str(ctx.exception))
        for value in ["1.5", 1 + 2j, None, np.array([1.0, 2.0]), np.zeros(0)]:
            with self.assertRaises(ValueError) as ctx:
                quadriga_lib.tools.scenpar_validate({"DS_mu": value})
            self.assertIn("Entry 'DS_mu' of 'scenpar_in' must be a real scalar", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.tools.scenpar_validate(self.scenpar, no_such_argument=1)


if __name__ == "__main__":
    unittest.main()
