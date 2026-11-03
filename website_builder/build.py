#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Entry point of the website builder: python3 website_builder/build.py website [--release] [--root DIR]

No syntax newer than Python 3.6 here, and qsite is imported only after the checks, so an
older Python or a missing tool gets a plain message instead of a traceback.
"""

import os
import shutil
import sys

SETUP = """To set up a build machine (see "Set up a build machine" in website_builder/README.md):
  conda create -y -n quadriga-web -c conda-forge "python>=3.11" "markdown-it-py>=3.0" "mdit-py-plugins>=0.4.2" pygments "jinja2>=3.1" pytest "pandoc>=3.7"
  conda activate quadriga-web
or on Ubuntu 26.04 or newer:
  sudo apt install pandoc python3-markdown-it python3-mdit-py-plugins python3-pygments python3-jinja2 python3-pytest
With conda, activate the environment quadriga-web before each build session."""


def missing_tools():
    """One line per missing requirement; empty if everything is there."""
    if sys.version_info < (3, 10):
        return ["Python 3.10 or newer is needed"]
    import importlib.util
    out = []
    if importlib.util.find_spec("tomllib") is None and importlib.util.find_spec("tomli") is None:
        out.append("No TOML reader: use Python 3.11 or newer, or on 3.10 install python3-tomli (Ubuntu) or pip install tomli")
    for module, package in (("markdown_it", "markdown-it-py"), ("mdit_py_plugins", "mdit-py-plugins"),
                            ("pygments", "Pygments"), ("jinja2", "Jinja2")):
        if importlib.util.find_spec(module) is None:
            out.append("Python package %s missing (module %s)" % (package, module))
    if shutil.which("pandoc") is None:
        out.append("pandoc not found on PATH")
    return out


def main():
    problems = missing_tools()
    if problems:
        for line in problems:
            print("build.py: " + line)
        print("build.py: Python in use: %s (%s)" % (sys.executable, sys.version.split()[0]))
        print(SETUP)
        return 2
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="backslashreplace")  # a diagnostic must not stop a run on a Windows console
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "docs_builder"))
    sys.path.insert(0, here)
    from qsite import cli
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())
