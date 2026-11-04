# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""The docs command: exit codes, nothing written on errors, standard library only."""

import ast
import os
import subprocess
import sys
from pathlib import Path

DOCS_BUILDER = Path(__file__).resolve().parents[3] / "docs_builder"


def test_structural_error_writes_nothing(mini_repo, run_docs):
    assert run_docs(mini_repo)[0] == 0
    before = {p: p.read_bytes() for p in (mini_repo / "docs").rglob("*") if p.is_file()}
    src = mini_repo / "src/qd_tools.cpp"
    src.write_text(src.read_text(encoding="utf-8").replace("# quadriga_lib_version", "#quadriga_lib_version")
                   .replace("Helper functions", "Changed helper functions"), encoding="utf-8")
    code, out = run_docs(mini_repo)
    assert code == 1 and "src/qd_tools.cpp:13: error: E-name:" in out
    assert {p: p.read_bytes() for p in (mini_repo / "docs").rglob("*") if p.is_file()} == before


def test_entry_point_and_usage_errors(mini_repo):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    run = lambda *a: subprocess.run([sys.executable, str(DOCS_BUILDER / "build.py")] + list(a), env=env,  # noqa: E731
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert run("docs", "--root", str(mini_repo)).returncode == 0
    assert run("website").returncode == 2
    assert run("docs", "--root", str(mini_repo / "nosuch")).returncode == 2


def test_imports_are_standard_library():
    allowed = set(sys.stdlib_module_names) | {"qdocs", "tomllib", "tomli"}
    files = sorted(DOCS_BUILDER.rglob("*.py"))
    assert DOCS_BUILDER / "build.py" in files and DOCS_BUILDER / "qdocs" / "cli.py" in files
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "qdocs"] if node.level == 0 else ["qdocs"]
            else:
                continue
            for name in names:
                assert name.split(".")[0] in allowed, "%s imports %s" % (path, name)
                assert not name.startswith("qsite"), path
