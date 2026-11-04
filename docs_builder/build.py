#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Entry point of the docs builder: python3 docs_builder/build.py docs [--check] [--root DIR]

No syntax newer than Python 3.6 here, and qdocs is imported only after the checks, so an
older Python gets a plain message instead of a traceback.
"""

import os
import sys


def missing_tools():
    """One line per missing requirement; empty if Python and a TOML reader are there."""
    found = "Python in use: %s (%s)" % (sys.executable, sys.version.split()[0])
    if sys.version_info < (3, 10):
        return ["Python 3.10 or newer is needed. " + found]
    import importlib.util
    if importlib.util.find_spec("tomllib") is None and importlib.util.find_spec("tomli") is None:
        return ["No TOML reader: use Python 3.11 or newer, or on 3.10 install python3-tomli (Ubuntu) "
                "or pip install tomli. " + found]
    return []


def main():
    problems = missing_tools()
    if problems:
        for line in problems:
            print("build.py: " + line)
        return 2
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="backslashreplace")  # a diagnostic must not stop a run on a Windows console
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qdocs import cli
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())
