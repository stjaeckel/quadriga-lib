# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Run every notebook that is newer than its executed copy: python -m qnb.runall [--matlab-kernel NAME]

The Windows Makefile uses this instead of the pattern rules of tutorials/GNUmakefile, which nmake cannot express.
- Runs tutorials/*.ipynb into executed/<stem>.ipynb and matlab/*.ipynb into executed/matlab/<stem>.ipynb, one
  after another, from the tutorials folder; each run as qnb.run does it
- --matlab-kernel runs the MATLAB variants with that kernel instead of the one in the notebook (on Windows the
  MATLAB kernel of jupyter-matlab-proxy, as Octave is not available there)
- A failing notebook does not stop the others; the exit code is the number of failed notebooks"""

import glob
import os
import sys

from . import run


def pending():
    """(notebook, executed copy) of the notebooks in the current folder whose executed copy is missing or older."""
    out = []
    for src in sorted(glob.glob("*.ipynb")) + sorted(glob.glob(os.path.join("matlab", "*.ipynb"))):
        dst = os.path.join("executed", src)
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
            out.append((src, dst))
    return out


def main(argv):
    kernel = None
    if len(argv) == 2 and argv[0] == "--matlab-kernel":
        kernel = argv[1]
    elif argv:
        print("usage: python -m qnb.runall [--matlab-kernel NAME]", file=sys.stderr)
        return 2
    failed = 0
    for src, dst in pending():
        opts = ["--kernel", kernel] if kernel and src.startswith("matlab" + os.sep) else []
        failed += run.main(opts + [src, dst]) != 0
    if failed:
        print("qnb.runall: %d notebook(s) failed" % failed, file=sys.stderr)
    return failed


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
