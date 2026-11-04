# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""'website' on the real sources, built into a temporary root: old anchors still exist, and every
notebook converts without warnings."""

import contextlib
import html.parser
import io
import json
import shutil
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[1]
REPO = TESTS.parents[1]
OLD_ANCHORS = TESTS / "fixtures" / "old_anchors.json"

pytestmark = pytest.mark.repo


class _Ids(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        self.ids.update(v for k, v in attrs if k == "id")


def page_ids(path):
    p = _Ids()
    p.feed(path.read_text(encoding="utf-8"))
    return p.ids


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


def test_old_anchors_exist(site):
    root = site[0]
    old = json.loads(OLD_ANCHORS.read_text(encoding="utf-8"))
    missing = []
    for page, ids in sorted(old.items()):
        have = page_ids(root / "website" / page)
        missing += ["%s#%s" % (page, i) for i in ids if i not in have]
    assert missing == []
    assert "quickstart" in page_ids(root / "website" / "index.html")


def test_tutorials_without_warnings(site):
    _, out, files = site
    assert [ln for ln in out.splitlines() if ln.startswith("tutorials/") and ": warning:" in ln] == []
    for nb in sorted((REPO / "tutorials").glob("*.ipynb")):
        for ext in (".html", ".md", ".ipynb"):
            assert "tutorials/" + nb.stem + ext in files
