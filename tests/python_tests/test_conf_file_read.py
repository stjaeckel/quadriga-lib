# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.conf_file_read.
# Scope: argument and output handling, defaults, dict order, error paths.
# Physics validation lives in the Catch2 suite (test_conf_file_read.cpp).

import sys
import os
import glob
import pathlib
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

# A file with every kind of key: PL_ keys not in sorted order, one with three values, two short names,
# an unknown key and the unsupported key GR_epsilon
test_conf = """% Test file
DS_mu = -7.5
NumClusters = 12
asD_ds = 0.25
SubpathMethod = mmMAGIC
PL_model = logdist
PL_B = 1 2.5 -3
PL_A = 4
ShortName = Ta
ShortName = Tb
FooKey = 3
GR_epsilon = 5
"""


class test_case(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fn = os.path.join(self.tmp.name, "conf_file_read_test.conf")
        with open(self.fn, "w") as f:
            f.write(test_conf)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        fn = os.path.join(self.tmp.name, name)
        with open(fn, "w") as f:
            f.write(text)
        return fn

    def test_basic_call(self):
        out = quadriga_lib.gbsm.conf_file_read(self.fn)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 6)
        scenpar, plpar, lsp_xcorr, lsp_xcorr_chk, short_names, ignored_keys = out

        # scenpar: 172 entries, float values, 'SubpathMethod' (str) at position 106, the newest keys at the end
        self.assertIsInstance(scenpar, dict)
        self.assertEqual(len(scenpar), 172)
        keys = list(scenpar)
        self.assertEqual(keys[105:108], ["absTOA_lambda", "SubpathMethod", "LOS_scatter_radius"])
        self.assertEqual(keys[-6:], ["LOS_state_lambda", "IO_state_lambda", "O2I_d2D_in_max", "DS_VS", "DS_VS_a", "DS_VS_b"])
        self.assertEqual(keys[0], "NumClusters")
        for key, value in scenpar.items():
            self.assertIsInstance(value, str if key == "SubpathMethod" else float, key)
        self.assertEqual(scenpar["SubpathMethod"], "mmMAGIC")
        self.assertEqual(scenpar["DS_mu"], -7.5)
        self.assertEqual(scenpar["NumClusters"], 12.0)

        # The order is the canonical one, the same as for the default set
        default_set, _, _ = quadriga_lib.gbsm.scenpar_validate()
        self.assertEqual(keys, list(default_set))

        # plpar: 'model' first, then the PL_ keys in file order (B before A), 1D float64 arrays
        self.assertIsInstance(plpar, dict)
        self.assertEqual(list(plpar), ["model", "B", "A"])
        self.assertEqual(plpar["model"], "logdist")
        for key in ["B", "A"]:
            self.assertIsInstance(plpar[key], np.ndarray)
            self.assertEqual(plpar[key].dtype, np.float64)
            self.assertEqual(plpar[key].ndim, 1)
        npt.assert_array_equal(plpar["B"], [1.0, 2.5, -3.0])
        npt.assert_array_equal(plpar["A"], [4.0])

        # lsp_xcorr: symmetric, unit diagonal; asD_ds is the (DS, ASD) entry
        self.assertIsInstance(lsp_xcorr, np.ndarray)
        self.assertEqual(lsp_xcorr.shape, (8, 8))
        self.assertEqual(lsp_xcorr.dtype, np.float64)
        npt.assert_array_equal(np.diag(lsp_xcorr), np.ones(8))
        npt.assert_array_equal(lsp_xcorr, lsp_xcorr.T)
        self.assertEqual(lsp_xcorr[0, 3], 0.25)
        self.assertEqual(np.count_nonzero(lsp_xcorr - np.eye(8)), 2)

        self.assertIsInstance(lsp_xcorr_chk, bool)
        self.assertTrue(lsp_xcorr_chk)

        self.assertEqual(short_names, ["Ta", "Tb"])
        self.assertEqual(ignored_keys, ["FooKey", "GR_epsilon"])

    def test_data_config_files(self):
        files = sorted(glob.glob(os.path.join(config_dir, "*.conf")))
        self.assertEqual(len(files), 23)
        for fn in files:
            scenpar, plpar, lsp_xcorr, lsp_xcorr_chk, short_names, ignored_keys = \
                quadriga_lib.gbsm.conf_file_read(fn)
            self.assertEqual(len(scenpar), 172, fn)
            self.assertEqual(list(scenpar)[106], "SubpathMethod", fn)
            self.assertEqual(list(plpar)[0], "model", fn)
            self.assertEqual(lsp_xcorr.shape, (8, 8), fn)
            self.assertTrue(lsp_xcorr_chk, fn)
            self.assertEqual(ignored_keys, [], fn)

    def test_keyword_arguments(self):
        ref = quadriga_lib.gbsm.conf_file_read(self.fn, True)
        out = quadriga_lib.gbsm.conf_file_read(fn=self.fn, validate=True)
        self.assertEqual(out[0], ref[0])
        self.assertEqual(list(out[1]), list(ref[1]))
        npt.assert_array_equal(out[2], ref[2])
        self.assertEqual(out[3:], ref[3:])

    def test_validate(self):
        fn = self.write("conf_file_read_invalid.conf", "r_DS = -1\n")

        # Default is validate = True
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.gbsm.conf_file_read(fn)
        self.assertIn("r_DS", str(ctx.exception))
        with self.assertRaises(ValueError):
            quadriga_lib.gbsm.conf_file_read(fn, True)

        # validate = False only checks the grammar
        scenpar = quadriga_lib.gbsm.conf_file_read(fn, False)[0]
        self.assertEqual(scenpar["r_DS"], -1.0)

    def test_validate_flag_types(self):
        # None selects the default (True); bool and numpy.bool_ are accepted; anything else is a TypeError
        fn = self.write("conf_file_read_invalid.conf", "r_DS = -1\n")
        for args, kwargs in [((fn, None), {}), ((fn,), {"validate": None}), ((fn, np.bool_(True)), {})]:
            with self.assertRaises(ValueError) as ctx:
                quadriga_lib.gbsm.conf_file_read(*args, **kwargs)
            self.assertIn("r_DS", str(ctx.exception))
        self.assertEqual(quadriga_lib.gbsm.conf_file_read(fn, np.bool_(False))[0]["r_DS"], -1.0)

        ref = quadriga_lib.gbsm.conf_file_read(self.fn)
        out = quadriga_lib.gbsm.conf_file_read(self.fn, None)
        self.assertEqual(out[0], ref[0])
        npt.assert_array_equal(out[2], ref[2])
        self.assertEqual(out[3:], ref[3:])

        for value in ["yes", 1, 0, 1.0]:
            with self.assertRaises(TypeError) as ctx:
                quadriga_lib.gbsm.conf_file_read(fn, value)
            self.assertIn("Input 'validate' must be a bool or None.", str(ctx.exception))

    def test_path_like(self):
        # str, bytes and os.PathLike give the same result
        class fspath_only:
            def __init__(self, path):
                self.path = path

            def __fspath__(self):
                return self.path

        ref = quadriga_lib.gbsm.conf_file_read(self.fn)
        for fn in [pathlib.Path(self.fn), os.fsencode(self.fn), fspath_only(self.fn), fspath_only(os.fsencode(self.fn))]:
            out = quadriga_lib.gbsm.conf_file_read(fn)
            self.assertEqual(out[0], ref[0])
            self.assertEqual(list(out[1]), list(ref[1]))
            npt.assert_array_equal(out[2], ref[2])
            self.assertEqual(out[3:], ref[3:])
        self.assertEqual(quadriga_lib.gbsm.conf_file_read(fn=pathlib.Path(self.fn))[0], ref[0])

        for fn in [5, None, [self.fn]]:
            with self.assertRaises(TypeError) as ctx:
                quadriga_lib.gbsm.conf_file_read(fn)
            self.assertIn("Input 'fn' must be a str or path-like object.", str(ctx.exception))

    def test_byte_order_mark_and_range(self):
        # A UTF-8 byte order mark at the start of the file is skipped
        fn = os.path.join(self.tmp.name, "conf_file_read_bom.conf")
        with open(fn, "wb") as f:
            f.write(b"\xef\xbb\xbfDS_mu = -7.25\nShortName = Bm\n")
        scenpar, _, _, _, short_names, ignored_keys = quadriga_lib.gbsm.conf_file_read(fn)
        self.assertEqual(scenpar["DS_mu"], -7.25)
        self.assertEqual(short_names, ["Bm"])
        self.assertEqual(ignored_keys, [])

        # A number beyond the range of a double is an error, below it reads as 0
        fn = self.write("conf_file_read_range.conf", "DS_sigma = 0.3\nDS_mu = 1e400\n")
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.gbsm.conf_file_read(fn)
        self.assertIn("line 2: Value of 'DS_mu' is out of range.", str(ctx.exception))
        fn = self.write("conf_file_read_underflow.conf", "DS_mu = -1e-400\n")
        value = quadriga_lib.gbsm.conf_file_read(fn)[0]["DS_mu"]
        self.assertEqual(value, 0.0)
        self.assertTrue(np.signbit(value))

    def test_empty_outputs(self):
        # No PL_ key, no short name, no unknown key: empty dict and empty lists
        fn = self.write("conf_file_read_empty.conf", "% only a comment\n")
        scenpar, plpar, _, lsp_xcorr_chk, short_names, ignored_keys = quadriga_lib.gbsm.conf_file_read(fn)
        self.assertEqual(plpar, {})
        self.assertIsInstance(plpar, dict)
        self.assertEqual(short_names, [])
        self.assertEqual(ignored_keys, [])
        self.assertEqual(scenpar["SubpathMethod"], "legacy")
        self.assertTrue(lsp_xcorr_chk)

        # PL_ keys without PL_model: no 'model' entry
        fn = self.write("conf_file_read_nomodel.conf", "PL_A = 1\n")
        plpar = quadriga_lib.gbsm.conf_file_read(fn)[1]
        self.assertEqual(list(plpar), ["A"])

    def test_errors(self):
        # Thrown by C++: std::runtime_error -> RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            quadriga_lib.gbsm.conf_file_read(os.path.join(self.tmp.name, "no_such_file.conf"))
        self.assertIn("Cannot open file", str(ctx.exception))

        # Thrown by C++: std::invalid_argument -> ValueError, with file name and line number
        fn = self.write("conf_file_read_grammar.conf", "% comment\nDS_mu = abc\n")
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.gbsm.conf_file_read(fn, False)
        self.assertIn("line 2", str(ctx.exception))
        self.assertIn("DS_mu", str(ctx.exception))

        # Cross-key check: DS_VS > 0 needs DS_VS_a or DS_VS_b > 0; the message names the line of DS_VS
        fn = self.write("conf_file_read_ds_vs.conf", "DS_mu = -9.44\nDS_VS = 4\n")
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.gbsm.conf_file_read(fn)
        self.assertIn("line 2", str(ctx.exception))
        self.assertIn("Values of 'DS_VS_a' and 'DS_VS_b' must not both be 0 if 'DS_VS' is > 0.", str(ctx.exception))
        self.assertEqual(quadriga_lib.gbsm.conf_file_read(fn, False)[0]["DS_VS"], 4.0)

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.gbsm.conf_file_read()
        with self.assertRaises(TypeError):
            quadriga_lib.gbsm.conf_file_read(self.fn, no_such_argument=1)

    def test_submodule(self):
        # The conf functions live in quadriga_lib.gbsm, no longer in quadriga_lib.tools
        for name in ["conf_file_read", "conf_file_write", "conf_file_list", "scenpar_validate"]:
            self.assertTrue(hasattr(quadriga_lib.gbsm, name), name)
            self.assertFalse(hasattr(quadriga_lib.tools, name), name)


if __name__ == "__main__":
    unittest.main()
