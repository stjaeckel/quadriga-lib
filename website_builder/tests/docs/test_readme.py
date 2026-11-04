# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""The table "Errors and warnings" of docs_builder/README.md lists exactly the error ids and warning kinds of the code.

The ids and kinds are read from the source text of both builders (no import of the website builder):
an error id is a string that starts a message with "E-<id>:", a warning kind is the string literal
passed last to a call of warning() or of env["warn"]().
"""

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
README = REPO / "docs_builder" / "README.md"
CODE = [REPO / "docs_builder" / "qdocs", REPO / "website_builder" / "qsite"]

ERROR_ID_RE = re.compile(r"^(E-[a-z]+(?:-[a-z]+)*):")
ROW_RE = re.compile(r"^\| `([^`]+)` \| (error|warning) \|")


def _is_warning_call(func):
    if isinstance(func, ast.Attribute):
        return func.attr == "warning"
    return isinstance(func, ast.Subscript) and "warn" in ast.dump(func.slice)


def code_ids():
    """({error ids}, {warning kinds}) found in the builder modules."""
    errors, kinds = set(), set()
    for folder in CODE:
        for path in sorted(folder.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    m = ERROR_ID_RE.match(node.value)
                    if m:
                        errors.add(m.group(1))
                elif isinstance(node, ast.Call) and _is_warning_call(node.func) and node.args:
                    last = node.args[-1]
                    if isinstance(last, ast.Constant) and isinstance(last.value, str):
                        kinds.add(last.value)
    return errors, kinds


def readme_ids():
    """({error ids}, {warning kinds}) of the first table below '## Errors and warnings'."""
    lines = README.read_text(encoding="utf-8").split("\n")
    start = lines.index("## Errors and warnings")
    errors, kinds, in_table = set(), set(), False
    for ln in lines[start + 1:]:
        if ln.startswith("## "):
            break
        if not ln.startswith("|"):
            if in_table:
                break
            continue
        in_table = True
        m = ROW_RE.match(ln)
        if m:
            (errors if m.group(2) == "error" else kinds).add(m.group(1))
    return errors, kinds


def test_code_defines_ids_and_kinds():
    errors, kinds = code_ids()
    assert {"E-link", "E-table", "E-heading"} <= errors
    assert {"param-item", "code-span", "fence-language", "size"} <= kinds


def test_readme_table_matches_code():
    code_errors, code_kinds = code_ids()
    doc_errors, doc_kinds = readme_ids()
    assert sorted(code_errors - doc_errors) == [], "error ids missing in the README table"
    assert sorted(doc_errors - code_errors) == [], "error ids in the README table that the code does not define"
    assert sorted(code_kinds - doc_kinds) == [], "warning kinds missing in the README table"
    assert sorted(doc_kinds - code_kinds) == [], "warning kinds in the README table that the code does not define"
