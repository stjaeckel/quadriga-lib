# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.tools.conf_file_write.
# Scope: argument and output handling, defaults, dict conversion, round trip, error paths.
# Physics validation lives in the Catch2 suite (test_conf_file_write.cpp).

import sys
import os
import copy
import glob
import tempfile
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib

config_dir = os.path.join(current_dir, "../../data/config")


def scenpar_bits(scenpar):
    # Numeric values as raw 64-bit patterns, so that -0.0 and 0.0 differ
    return np.array([v for k, v in scenpar.items() if k != "SubpathMethod"], dtype=np.float64).view(np.uint64)


class test_case(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fn = os.path.join(self.tmp.name, "conf_file_write_test.conf")

        # Distinct values: one that needs 17 digits, -0, Inf, a correlation, a non-default SubpathMethod
        self.scenpar = {"DS_mu": 0.1 + 0.2, "NumClusters": 12.0, "asD_ds": 0.25, "DS_gamma": -0.0,
                        "absTOA_mu": np.inf, "SubpathMethod": "mmMAGIC"}
        # PL keys not in sorted order, one with three values
        self.plpar = {"model": "logdist", "B": np.array([1.0, 2.5, -3.0]), "A": np.array([4.0])}
        self.short_names = ["Ta", "Tb"]
        self.ref = "Test reference"
        self.ref_sec = ["Sec0", "", "Sec2"]
        self.valid_range = np.array([[0.5, 100.0], [10.0, 1000.0], [25.0, 25.0], [1.5, np.nan]])

    def tearDown(self):
        self.tmp.cleanup()

    def write_text(self, *args, **kwargs):
        # Write to self.fn (overwriting) and return the file content
        quadriga_lib.tools.conf_file_write(self.fn, *args, **kwargs)
        with open(self.fn) as f:
            return f.read()

    def test_basic_call(self):
        out = quadriga_lib.tools.conf_file_write(self.fn, self.scenpar, self.plpar, self.short_names,
                                                 self.ref, self.ref_sec, self.valid_range)
        self.assertIsNone(out)

        scenpar, plpar, lsp_xcorr, lsp_xcorr_chk, short_names, ignored_keys = \
            quadriga_lib.tools.conf_file_read(self.fn)
        for key, value in self.scenpar.items():
            self.assertEqual(scenpar[key], value, key)
        self.assertTrue(np.signbit(scenpar["DS_gamma"]))
        self.assertEqual(list(plpar), ["model", "B", "A"])
        self.assertEqual(plpar["model"], "logdist")
        npt.assert_array_equal(plpar["B"], self.plpar["B"])
        npt.assert_array_equal(plpar["A"], self.plpar["A"])
        self.assertEqual(lsp_xcorr[0, 3], 0.25)
        self.assertEqual(short_names, self.short_names)
        self.assertEqual(ignored_keys, [])

        # ref, ref_sec (0-based section index) and valid_range reach the file
        with open(self.fn) as f:
            text = f.read()
        self.assertIn('% Config file for scenario "conf_file_write_test"', text)
        self.assertIn("% See: Test reference\n", text)
        self.assertIn("Large scale distributions\n% See: Test reference; Sec0\n", text)
        self.assertIn("Inter-parameter correlations\n% See: Test reference; Sec2\n", text)
        self.assertIn("0.5 to 100 GHz", text)
        self.assertIn("10 to 1000 m", text)
        self.assertNotIn("MT antenna height", text)  # row with NaN is not written

    def test_round_trip_data_config(self):
        files = sorted(glob.glob(os.path.join(config_dir, "*.conf")))
        self.assertEqual(len(files), 23)
        for fn in files:
            ref = quadriga_lib.tools.conf_file_read(fn)
            scenpar, plpar, _, _, short_names, _ = ref

            fn_out = os.path.join(self.tmp.name, "round_trip_" + os.path.basename(fn))
            quadriga_lib.tools.conf_file_write(fn_out, scenpar, plpar, short_names)
            out = quadriga_lib.tools.conf_file_read(fn_out)
            os.remove(fn_out)

            # scenpar: same keys in the same order, same values bit for bit
            self.assertEqual(list(out[0]), list(scenpar), fn)
            self.assertEqual(out[0]["SubpathMethod"], scenpar["SubpathMethod"], fn)
            npt.assert_array_equal(scenpar_bits(out[0]), scenpar_bits(scenpar), fn)
            self.assertEqual(out[0], scenpar, fn)

            # plpar: same keys in the same order, same values bit for bit
            self.assertEqual(list(out[1]), list(plpar), fn)
            for key, value in plpar.items():
                if key == "model":
                    self.assertEqual(out[1][key], value, fn)
                else:
                    self.assertEqual(out[1][key].dtype, np.float64)
                    npt.assert_array_equal(out[1][key].view(np.uint64), value.view(np.uint64), fn)

            npt.assert_array_equal(out[2], ref[2], fn)
            self.assertEqual(out[3:], ref[3:], fn)

        self.assertEqual(sorted(os.listdir(self.tmp.name)), [])

    def test_keyword_arguments(self):
        ref = self.write_text(self.scenpar, self.plpar, self.short_names, self.ref, self.ref_sec,
                              self.valid_range, True)
        quadriga_lib.tools.conf_file_write(fn=self.fn, scenpar=self.scenpar, plpar=self.plpar,
                                           short_names=self.short_names, ref=self.ref, ref_sec=self.ref_sec,
                                           valid_range=self.valid_range, write_defaults=True)
        with open(self.fn) as f:
            self.assertEqual(f.read(), ref)

    def test_optional_inputs(self):
        # Omitted and None give the same file; scenpar None and {} write the default set
        ref = self.write_text({})
        self.assertEqual(ref, '% Config file for scenario "conf_file_write_test"\n')
        self.assertEqual(self.write_text(None), ref)
        self.assertEqual(self.write_text({}, None, None, None, None, None, False), ref)
        self.assertEqual(self.write_text({}, {}, [], "", []), ref)

        # Each optional input changes the file
        texts = [self.write_text({}, plpar=self.plpar),
                 self.write_text({}, short_names=self.short_names),
                 self.write_text({}, ref=self.ref),
                 self.write_text(self.scenpar, ref=self.ref, ref_sec=self.ref_sec),
                 self.write_text({}, valid_range=self.valid_range),
                 self.write_text({}, write_defaults=True)]
        for text in texts:
            self.assertNotEqual(text, ref)
        self.assertIn("PL_model", texts[0])
        self.assertIn("ShortName =            Ta", texts[1])
        self.assertIn("% See: Test reference", texts[2])
        self.assertIn("% See: Test reference; Sec0", texts[3])
        self.assertIn("Valid frequency range", texts[4])
        self.assertIn("SubpathMethod =        legacy", texts[5])

        # write_defaults = True writes all 165 numeric keys and SubpathMethod, the values do not change
        quadriga_lib.tools.conf_file_write(self.fn, self.scenpar, write_defaults=True)
        default_set = quadriga_lib.tools.scenpar_validate(self.scenpar)[0]
        self.assertEqual(quadriga_lib.tools.conf_file_read(self.fn)[0], default_set)

        # A single str is one short name
        self.assertEqual(self.write_text({}, short_names="Ta"), self.write_text({}, short_names=["Ta"]))

    def test_plpar_order(self):
        # The dict order is the write order; 'model' can be at any position of the input dict
        plpar = {"A": [4.0], "model": "logdist", "B": [1.0, 2.5, -3.0]}
        quadriga_lib.tools.conf_file_write(self.fn, {}, plpar)
        out = quadriga_lib.tools.conf_file_read(self.fn)[1]
        self.assertEqual(list(out), ["model", "A", "B"])

        # Without 'model'
        quadriga_lib.tools.conf_file_write(self.fn, {}, {"B": 1.0, "A": 2.0})
        out = quadriga_lib.tools.conf_file_read(self.fn)[1]
        self.assertEqual(list(out), ["B", "A"])

    def test_value_types_and_layout(self):
        ref = self.write_text(self.scenpar, self.plpar, valid_range=self.valid_range)

        # Real scalars of other types and shapes in scenpar
        scenpar = dict(self.scenpar)
        scenpar.update({"DS_mu": np.array([[0.1 + 0.2]]), "NumClusters": np.int32(12),
                        "asD_ds": np.float32(0.25), "DS_gamma": np.array(-0.0)})
        self.assertEqual(self.write_text(scenpar, self.plpar, valid_range=self.valid_range), ref)

        # plpar values as list, tuple, integer array, float32, scalar and non-contiguous view
        variants = [{"model": "logdist", "B": [1, 2.5, -3], "A": (4,)},
                    {"model": "logdist", "B": np.array([1.0, 2.5, -3.0], dtype=np.float32), "A": np.array([4])},
                    {"model": "logdist", "B": np.array([1.0, 0.0, 2.5, 0.0, -3.0])[::2], "A": 4.0}]
        for plpar in variants:
            self.assertEqual(self.write_text(self.scenpar, plpar, valid_range=self.valid_range), ref)

        # valid_range in Fortran order, as non-contiguous view and as nested list
        for valid_range in [np.asfortranarray(self.valid_range),
                            np.repeat(self.valid_range, 2, axis=1)[:, ::2],
                            self.valid_range.tolist()]:
            self.assertEqual(self.write_text(self.scenpar, self.plpar, valid_range=valid_range), ref)

    def test_inputs_not_modified(self):
        scenpar = copy.deepcopy(self.scenpar)
        plpar = copy.deepcopy(self.plpar)
        valid_range = self.valid_range.copy()
        quadriga_lib.tools.conf_file_write(self.fn, scenpar, plpar, self.short_names, self.ref, self.ref_sec,
                                           valid_range)
        self.assertEqual(list(scenpar), list(self.scenpar))
        npt.assert_array_equal(scenpar_bits(scenpar), scenpar_bits(self.scenpar))
        self.assertEqual(list(plpar), list(self.plpar))
        for key in ["B", "A"]:
            npt.assert_array_equal(plpar[key], self.plpar[key])
        npt.assert_array_equal(valid_range, self.valid_range)

    def test_errors(self):
        bad_fn = os.path.join(self.tmp.name, "conf_file_write_error.conf")

        def check(exc_type, text, *args, **kwargs):
            with self.assertRaises(exc_type) as ctx:
                quadriga_lib.tools.conf_file_write(bad_fn, *args, **kwargs)
            self.assertIn(text, str(ctx.exception))
            self.assertFalse(os.path.exists(bad_fn))  # a failed call does not create a file

        # Thrown by C++: std::invalid_argument -> ValueError
        check(ValueError, "Unknown key 'GR_epsilon'", {"GR_epsilon": 1.0})
        check(ValueError, "SNS_mu", {"SNS_mu": 2.0})
        check(ValueError, "'A' in 'plpar' must not be NaN", {}, {"A": [np.nan]})
        check(ValueError, "'A' in 'plpar' has no value", {}, {"A": []})
        check(ValueError, "short_names", {}, short_names=["a b"])
        check(ValueError, "line break", {}, ref="a\nb")
        check(ValueError, "at most 10 entries", {}, ref_sec=["a"] * 11)
        check(ValueError, "valid_range", {}, valid_range=np.ones((2, 4)))

        # Thrown by C++: std::runtime_error -> RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            quadriga_lib.tools.conf_file_write(os.path.join(self.tmp.name, "no_such_dir", "x.conf"), {})
        self.assertIn("for writing", str(ctx.exception))

        # Thrown by the wrapper and its dict conversion
        check(ValueError, "Input 'scenpar' must be a dict or None", [("DS_mu", 1.0)])
        check(ValueError, "Keys of 'scenpar' must be str", {1: 1.0})
        check(ValueError, "Entry 'SubpathMethod' of 'scenpar' must be a str", {"SubpathMethod": None})
        for value in ["1.5", 1 + 2j, None, [1.0, 2.0]]:
            check(ValueError, "Entry 'DS_mu' of 'scenpar' must be a real scalar", {"DS_mu": value})
        check(ValueError, "Input 'plpar' must be a dict or None", {}, [1.0])
        check(ValueError, "Keys of 'plpar' must be str", {}, {1: 1.0})
        check(ValueError, "Entry 'model' of 'plpar' must be a str", {}, {"model": 3})
        for value in ["1.5", np.array([1 + 2j]), np.ones((2, 2)), [1.0, [2.0, 3.0]]]:
            check(ValueError, "Entry 'A' of 'plpar' must be a real scalar or 1D array", {}, {"A": value})
        check(ValueError, "Input 'ref' must be a str or None", {}, ref=3)
        check(ValueError, "Input 'ref_sec' must be a list of str or None", {}, ref_sec="Sec0")
        check(ValueError, "Expected a string", {}, short_names=[1])

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.tools.conf_file_write(bad_fn)
        with self.assertRaises(TypeError):
            quadriga_lib.tools.conf_file_write(bad_fn, {}, no_such_argument=1)


if __name__ == "__main__":
    unittest.main()
