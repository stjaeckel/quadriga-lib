# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""'docs' on the real sources, built into a temporary root: no errors, all functions and sections, check passes."""

import shutil
from pathlib import Path

import pytest

from qdocs import cli

REPO = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.repo


@pytest.fixture(scope="module")
def real_root(tmp_path_factory):
    """The inputs of the docs builder copied from the real repository; docs/ is written there."""
    root = tmp_path_factory.mktemp("real_repo")
    for folder in ("src", "api_mex", "api_python"):
        (root / folder).mkdir()
        for p in sorted((REPO / folder).glob("*.cpp")):
            shutil.copy2(str(p), str(root / folder / p.name))
    (root / "include").mkdir()
    shutil.copy2(str(REPO / "include" / "quadriga_lib.hpp"), str(root / "include"))
    shutil.copy2(str(REPO / "CMakeLists.txt"), str(root))
    shutil.copytree(str(REPO / "docs_builder" / "content"), str(root / "docs_builder" / "content"))
    return root


def test_docs_and_check(real_root, run_docs):
    code, out = run_docs(real_root)
    assert code == 0, out
    assert out.splitlines()[-1].startswith("docs: 0 errors,")
    assert (real_root / "docs" / "faq.md").is_file()
    code, out = run_docs(real_root, "--check")
    assert code == 0, out


def test_function_and_section_counts(real_root):
    diag = cli.Diagnostics()
    model, _ = cli.build_docs(real_root, diag)
    assert diag.errors == 0
    api = model["api"]
    assert [len(api[lang].sections) for lang in ("cpp", "mex", "python")] == [10, 9, 9]
    counts = [sum(len(s.functions) for s in api[lang].sections) for lang in ("cpp", "mex", "python")]
    assert counts == [137, 85, 80]
