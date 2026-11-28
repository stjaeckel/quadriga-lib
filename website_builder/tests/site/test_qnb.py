# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Helpers of the tutorial build (tutorials/qnb): failing Octave cells, the script of the MATLAB check."""

import json
import os
import sys

from conftest import REPO

sys.path.insert(0, str(REPO / "tutorials"))
from qnb import run, runall, to_m  # noqa: E402


def notebook(language, *outputs):
    cell = {"cell_type": "code", "source": "x", "metadata": {}, "outputs": list(outputs), "execution_count": 1}
    return {"metadata": {"kernelspec": {"language": language}},
            "cells": [{"cell_type": "markdown", "source": "## A", "metadata": {}}, cell]}


def stream(name, text):
    return {"output_type": "stream", "name": name, "text": text}


def test_octave_errors_are_found_in_the_printed_text():
    failing = stream("stdout", "1\r\nerror: 'y' undefined near line 1, column 5\r\n")  # as the Octave kernel prints
    assert run.kernel_errors(notebook("octave", stream("stdout", "1\n"), failing)) == [
        (2, "error: 'y' undefined near line 1, column 5")]
    assert run.kernel_errors(notebook("octave", stream("stderr", "\x1b[0;31mx\nerror: z\n"))) == [(2, "error: z")]
    assert run.kernel_errors(notebook("octave", stream("stdout", "warning: shadows a function\n"))) == []
    traceback = stream("stdout", "error: 'y' undefined\nerror: called from\n    f at line 2 column 1\n")
    assert run.kernel_errors(notebook("octave", traceback)) == [(2, "error: 'y' undefined")]
    # text that the code prints like an error fails the run too (decided: loud rather than silent)
    assert run.kernel_errors(notebook("octave", stream("stdout", "error: printed\n"))) == [(2, "error: printed")]
    assert run.kernel_errors(notebook("python", failing)) == []  # Python errors are error outputs


def test_matlab_kernel_errors_are_found_in_stderr():
    # as the MATLAB kernel (jupyter-matlab-proxy) prints them: errors and warnings on stderr
    err = stream("stderr", "Error using *\nIncorrect dimensions for matrix multiplication.")
    assert run.kernel_errors(notebook("matlab", stream("stdout", "     2\n"), err)) == [(2, "Error using *")]
    assert run.kernel_errors(notebook("matlab", stream("stderr", "boom 3"))) == [(2, "boom 3")]  # error('id', ...)
    assert run.kernel_errors(notebook("matlab", stream("stderr", "Warning: my warning"))) == []


def test_runall_picks_notebooks_newer_than_their_executed_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for p in ("a.ipynb", "b.ipynb", "matlab/a.ipynb", "executed/a.ipynb", "executed/b.ipynb"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text("{}", encoding="utf-8")
    os.utime(tmp_path / "executed/b.ipynb", (1, 1))  # older than b.ipynb
    assert runall.pending() == [("b.ipynb", os.path.join("executed", "b.ipynb")),
                                (os.path.join("matlab", "a.ipynb"), os.path.join("executed", "matlab", "a.ipynb"))]


def test_matlab_check_script(tmp_path):
    nb = notebook("octave")
    nb["cells"].append({"cell_type": "code", "source": ["a = 1;\n", "disp(a)"], "metadata": {}, "outputs": []})
    src, dst = tmp_path / "x.ipynb", tmp_path / "x.m"
    src.write_text(json.dumps(nb), encoding="utf-8")
    assert to_m.main([str(src), str(dst)]) == 0
    assert dst.read_text(encoding="utf-8") == ("set(0, 'DefaultFigureVisible', 'off');\n\n%% cell 2\nx\n\n"
                                               "%% cell 3\na = 1;\ndisp(a)\n")
