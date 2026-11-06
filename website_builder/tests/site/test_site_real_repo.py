# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""'website' on the real sources, built into a temporary root: every notebook converts without warnings."""

import contextlib
import io
import shutil
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[1]
REPO = TESTS.parents[1]

pytestmark = pytest.mark.repo


def tree(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """One 'website' run on a copy of the real inputs: (root, output, site tree)."""
    from qsite import cli
    root = tmp_path_factory.mktemp("real_repo")
    for folder in ("src", "api_mex", "api_python"):
        (root / folder).mkdir()
        for p in sorted((REPO / folder).glob("*.cpp")):
            shutil.copy2(str(p), str(root / folder / p.name))
    (root / "include").mkdir()
    shutil.copy2(str(REPO / "include" / "quadriga_lib.hpp"), str(root / "include"))
    shutil.copy2(str(REPO / "CMakeLists.txt"), str(root))
    for folder in ("docs_builder/content", "website_builder/content", "website_builder/skill", "tutorials"):
        shutil.copytree(str(REPO / folder), str(root / folder))
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = cli.main(["website", "--root", str(root)])
    assert code == 0, out.getvalue()
    return root, out.getvalue(), tree(root / "website")


def test_tutorials_without_warnings(site):
    _, out, files = site
    warnings = [ln for ln in out.splitlines() if ln.startswith("tutorials/") and ": warning:" in ln]
    notebooks = sorted((REPO / "tutorials").glob("*.ipynb"))
    if notebooks and not (REPO / "tutorials/executed").is_dir():  # tutorials not run in this checkout
        assert warnings == ["tutorials/executed: warning: missing: the tutorials have not been run (make tutorials); "
                            "the tutorials page lists none [not-executed]"]
        return
    assert warnings == []
    for nb in notebooks:
        for ext in (".html", ".md", ".ipynb"):
            assert "tutorials/" + nb.stem + ext in files
