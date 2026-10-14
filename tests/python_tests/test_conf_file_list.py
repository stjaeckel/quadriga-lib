# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.tools.conf_file_list.
# Scope: argument and output handling, defaults, error paths.
# Physics validation lives in the Catch2 suite (test_conf_file_list.cpp).

import sys
import os
import tempfile
import unittest

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
        out = quadriga_lib.tools.conf_file_list([self.d1, self.d2])
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
        ref = quadriga_lib.tools.conf_file_list([self.d1, self.d2], False)
        out = quadriga_lib.tools.conf_file_list(dirs=[self.d1, self.d2], parse_shortnames=False)
        self.assertEqual(out, ref)

    def test_parse_shortnames(self):
        ref = quadriga_lib.tools.conf_file_list([self.d1, self.d2])
        self.assertEqual(quadriga_lib.tools.conf_file_list([self.d1, self.d2], True), ref)

        scenarios, file_names, file_dirs = quadriga_lib.tools.conf_file_list([self.d1, self.d2], False)
        self.assertEqual(scenarios, ["B", "a", "C"])
        self.assertEqual(file_names, ["B.conf", "a.conf", "C.conf"])
        self.assertEqual(file_dirs, [self.d1, self.d1, self.d2])

    def test_dirs_forms(self):
        # One str is one directory; a tuple works like a list
        ref = quadriga_lib.tools.conf_file_list([self.d1])
        self.assertEqual(quadriga_lib.tools.conf_file_list(self.d1), ref)
        self.assertEqual(quadriga_lib.tools.conf_file_list((self.d1,)), ref)

        # The order of the directories decides which B.conf wins
        scenarios, _, file_dirs = quadriga_lib.tools.conf_file_list([self.d2, self.d1])
        self.assertEqual(scenarios, ["B", "Other", "C", "Cs", "a"])
        self.assertEqual(file_dirs[0], self.d2)

        # An empty list lists nothing; a missing directory is skipped
        self.assertEqual(quadriga_lib.tools.conf_file_list([]), ([], [], []))
        missing = os.path.join(self.tmp.name, "no_such_dir")
        self.assertEqual(quadriga_lib.tools.conf_file_list([missing, self.d1])[0], ref[0])

        # "" is the current directory and is returned as given
        cwd = os.getcwd()
        try:
            os.chdir(self.d1)
            scenarios, _, file_dirs = quadriga_lib.tools.conf_file_list("")
        finally:
            os.chdir(cwd)
        self.assertEqual(scenarios, ref[0])
        self.assertEqual(file_dirs, ["", "", ""])

    def test_data_config(self):
        scenarios, file_names, file_dirs = quadriga_lib.tools.conf_file_list(config_dir)
        self.assertEqual(len(scenarios), 23)
        self.assertEqual(scenarios[0], "3GPP_38.901_v19_InF_LOS")
        self.assertEqual(file_names[0], "3GPP_38.901_v19_InF_LOS.conf")
        self.assertEqual(file_dirs, [config_dir] * 23)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with open(os.path.join(self.d2, "D.conf"), "w") as f:
            f.write("ShortName = Bs\n")
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.conf_file_list([self.d1, self.d2])
        self.assertIn("ShortName 'Bs' is ambiguous", str(ctx.exception))

        # Thrown by the string list conversion
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.tools.conf_file_list(3)
        self.assertIn("Expected a string", str(ctx.exception))
        with self.assertRaises(ValueError):
            quadriga_lib.tools.conf_file_list([self.d1, 3])

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.tools.conf_file_list()
        with self.assertRaises(TypeError):
            quadriga_lib.tools.conf_file_list(self.d1, no_such_argument=1)


if __name__ == "__main__":
    unittest.main()
