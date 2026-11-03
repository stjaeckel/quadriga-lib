# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Shared fixtures of the builder tests: a copy of the fixture repository and runners of both builders."""

import contextlib
import io
import shutil
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parents[1]
MINI_REPO = TESTS / "fixtures" / "mini_repo"

# No pythonpath setting in pytest.ini, so that pytest 6.2 (Ubuntu 22.04) works too
for folder in (REPO / "docs_builder", REPO / "website_builder"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


@pytest.fixture
def mini_repo(tmp_path):
    """A fresh copy of tests/fixtures/mini_repo below tmp_path."""
    root = tmp_path / "mini_repo"
    shutil.copytree(str(MINI_REPO), str(root))
    return root


def _run(main, root, args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = main([args[0], "--root", str(root)] + list(args[1:]))
    return code, out.getvalue()


@pytest.fixture
def run_docs():
    """run_docs(root, *args) -> (exit code, output) of 'build.py docs' (args default to none)."""
    def run(root, *args):
        from qdocs import cli
        return _run(cli.main, root, ("docs",) + args)
    return run


@pytest.fixture
def run_site():
    """run_site(root, *args) -> (exit code, output) of 'build.py website'; imports qsite only when called."""
    def run(root, *args):
        from qsite import cli
        return _run(cli.main, root, ("website",) + args)
    return run
