# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.gbsm.conf_file_list.
# Scope: argument and output handling, defaults, error paths, the shipped folder.
# Physics validation lives in the Catch2 suite (test_conf_file_list.cpp). The calls that compare whole lists leave
# out the shipped folder (include_shipped=False).

import sys
import os
import pathlib
import tempfile
import unittest
import numpy as np

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib

config_dir = os.path.join(current_dir, "../../data/config")


class test_case(unittest.TestCase):

    def setUp(self):
        # d1: B (short name Bs), a; d2: B (shadowed by d1), C (short name Cs)
        self.tmp = tempfile.TemporaryDirectory()
        self.d1 = os.path.join(self.tmp.name, "d1")
        self.d2 = os.path.join(self.tmp.name, "d2")
        files = {self.d1: {"B.conf": "ShortName = Bs\n", "a.conf": "% no short name\n"},
                 self.d2: {"B.conf": "ShortName = Other\n", "C.conf": "ShortName = Cs\n"}}
        for d, content in files.items():
            os.mkdir(d)
            for name, text in content.items():
                with open(os.path.join(d, name), "w") as f:
                    f.write(text)

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_call(self):
        out = quadriga_lib.gbsm.conf_file_list([self.d1, self.d2], include_shipped=False)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        scenarios, file_names, file_dirs = out
        for x in out:
            self.assertIsInstance(x, list)
            for s in x:
                self.assertIsInstance(s, str)

        # Byte-wise sorted per directory, short names right after their file, d2/B.conf skipped
        self.assertEqual(scenarios, ["B", "Bs", "a", "C", "Cs"])
        self.assertEqual(file_names, ["B.conf", "B.conf", "a.conf", "C.conf", "C.conf"])
        self.assertEqual(file_dirs, [self.d1, self.d1, self.d1, self.d2, self.d2])

    def test_keyword_arguments(self):
        ref = quadriga_lib.gbsm.conf_file_list([self.d1, self.d2], parse_shortnames=False, include_shipped=False)
        out = quadriga_lib.gbsm.conf_file_list(dirs=[self.d1, self.d2], parse_shortnames=False, include_shipped=False)
        self.assertEqual(out, ref)
        self.assertEqual(out[0], ["B", "a", "C"])

    def test_parse_shortnames(self):
        ref = quadriga_lib.gbsm.conf_file_list([self.d1, self.d2])
        self.assertEqual(quadriga_lib.gbsm.conf_file_list([self.d1, self.d2], parse_shortnames=True), ref)

        scenarios, file_names, file_dirs = quadriga_lib.gbsm.conf_file_list([self.d1, self.d2], parse_shortnames=False,
                                                                            include_shipped=False)
        self.assertEqual(scenarios, ["B", "a", "C"])
        self.assertEqual(file_names, ["B.conf", "a.conf", "C.conf"])
        self.assertEqual(file_dirs, [self.d1, self.d1, self.d2])

    def test_parse_shortnames_flag_types(self):
        # None selects the default (True); bool and numpy.bool_ are accepted; anything else is a TypeError
        ref = quadriga_lib.gbsm.conf_file_list([self.d1, self.d2])
        self.assertEqual(quadriga_lib.gbsm.conf_file_list([self.d1, self.d2], parse_shortnames=None), ref)
        self.assertEqual(quadriga_lib.gbsm.conf_file_list([self.d1, self.d2], parse_shortnames=None), ref)
        self.assertEqual(quadriga_lib.gbsm.conf_file_list([self.d1, self.d2], parse_shortnames=np.bool_(True)), ref)
        self.assertEqual(quadriga_lib.gbsm.conf_file_list([self.d1, self.d2], parse_shortnames=np.bool_(False)),
                         quadriga_lib.gbsm.conf_file_list([self.d1, self.d2], parse_shortnames=False))
        for value in ["yes", 1, 0]:
            with self.assertRaises(TypeError) as ctx:
                quadriga_lib.gbsm.conf_file_list([self.d1, self.d2], parse_shortnames=value)
            self.assertIn("'parse_shortnames'", str(ctx.exception))

    def test_dirs_path_like(self):
        # pathlib.Path and bytes, alone and in a list, give the same result; file_dirs holds the str form
        ref = quadriga_lib.gbsm.conf_file_list([self.d1, self.d2])
        p1, p2 = pathlib.Path(self.d1), pathlib.Path(self.d2)
        for dirs in [[p1, p2], (p1, self.d2), [os.fsencode(self.d1), p2]]:
            out = quadriga_lib.gbsm.conf_file_list(dirs)
            self.assertEqual(out[0], ref[0])
            self.assertEqual(out[1], ref[1])
            self.assertEqual(out[2], ref[2])
            for d in out[2]:
                self.assertIsInstance(d, str)
        out = quadriga_lib.gbsm.conf_file_list(p1, include_shipped=False)
        self.assertEqual(out, quadriga_lib.gbsm.conf_file_list(self.d1, include_shipped=False))
        self.assertEqual(out[2], [str(p1)] * len(out[2]))
        self.assertEqual(quadriga_lib.gbsm.conf_file_list(os.fsencode(self.d1), include_shipped=False), out)

    def test_dirs_forms(self):
        # One str is one directory; a tuple works like a list
        ref = quadriga_lib.gbsm.conf_file_list([self.d1])
        self.assertEqual(quadriga_lib.gbsm.conf_file_list(self.d1), ref)
        self.assertEqual(quadriga_lib.gbsm.conf_file_list((self.d1,)), ref)

        # The order of the directories decides which B.conf wins
        scenarios, _, file_dirs = quadriga_lib.gbsm.conf_file_list([self.d2, self.d1], include_shipped=False)
        self.assertEqual(scenarios, ["B", "Other", "C", "Cs", "a"])
        self.assertEqual(file_dirs[0], self.d2)

        # An empty list lists nothing; a missing directory is skipped
        self.assertEqual(quadriga_lib.gbsm.conf_file_list([], include_shipped=False), ([], [], []))
        missing = os.path.join(self.tmp.name, "no_such_dir")
        self.assertEqual(quadriga_lib.gbsm.conf_file_list([missing, self.d1])[0], ref[0])

        # "" is the current directory and is returned as given
        cwd = os.getcwd()
        try:
            os.chdir(self.d1)
            scenarios, _, file_dirs = quadriga_lib.gbsm.conf_file_list("", include_shipped=False)
        finally:
            os.chdir(cwd)
        self.assertEqual(scenarios, quadriga_lib.gbsm.conf_file_list(self.d1, include_shipped=False)[0])
        self.assertEqual(file_dirs, ["", "", ""])

    def test_data_config(self):
        scenarios, file_names, file_dirs = quadriga_lib.gbsm.conf_file_list(config_dir)
        self.assertEqual(len(scenarios), 23)
        self.assertEqual(scenarios[0], "3GPP_38.901_v19_InF_LOS")
        self.assertEqual(file_names[0], "3GPP_38.901_v19_InF_LOS.conf")
        self.assertEqual(file_dirs, [config_dir] * 23)

    def test_include_shipped(self):
        # Default: the shipped conf files after dirs; oracle: the listing of the shipped folder given explicitly
        scenarios, file_names, file_dirs = quadriga_lib.gbsm.conf_file_list([])
        self.assertTrue(scenarios)
        self.assertTrue(os.path.isabs(file_dirs[0]))
        self.assertEqual(quadriga_lib.gbsm.conf_file_list(file_dirs[0], include_shipped=False)[0], scenarios)
        own = quadriga_lib.gbsm.conf_file_list([self.d1, self.d2], include_shipped=False)
        out = quadriga_lib.gbsm.conf_file_list([self.d1, self.d2], include_shipped=np.bool_(True))
        self.assertEqual(out[0], own[0] + scenarios)
        with self.assertRaises(TypeError) as ctx:
            quadriga_lib.gbsm.conf_file_list([], include_shipped="yes")
        self.assertIn("'include_shipped'", str(ctx.exception))

    def test_errors(self):
        # Thrown by C++: std::runtime_error -> RuntimeError
        with open(os.path.join(self.d2, "D.conf"), "w") as f:
            f.write("ShortName = Bs\n")
        with self.assertRaises(RuntimeError) as ctx:
            quadriga_lib.gbsm.conf_file_list([self.d1, self.d2])  # also with the shipped folder
        self.assertIn("ShortName 'Bs' is ambiguous", str(ctx.exception))

        # Thrown by the string list conversion
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.gbsm.conf_file_list(3)
        self.assertIn("Input 'dirs' must be a path", str(ctx.exception))
        with self.assertRaises(ValueError):
            quadriga_lib.gbsm.conf_file_list([self.d1, 3])

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.gbsm.conf_file_list()
        with self.assertRaises(TypeError):
            quadriga_lib.gbsm.conf_file_list(self.d1, no_such_argument=1)


if __name__ == "__main__":
    unittest.main()
