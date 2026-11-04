# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Markdown output: the twin writer, API files, section files, prose twins, llms.txt and docs/README.md."""

import re

import pytest

from qdocs import cli, prose, render_md

BASE = "http://quadriga-lib.org"


@pytest.fixture
def docs(mini_repo):
    diag = cli.Diagnostics()
    _, mapping = cli.build_docs(mini_repo, diag)
    assert diag.errors == 0
    return {k: v.decode("utf-8") for k, v in mapping.items()}


def test_twin_lines_docblock_rewrites(docs):
    lines = ["## Inputs:<br>",
             "- See [[arrayant_interpolate_multi]] and .[[interpolate]], not `[[arrayant]]`",
             '- <a target="_blank" rel="noopener noreferrer" href="formats.html#6cab4884">QDANT</a> and '
             '`<a href="x.html">raw</a>`',
             "- [site](http://quadriga-lib.org/faq.html#cite), `[code](http://quadriga-lib.org/faq.html)`",
             "- First line<br>",
             "  continues<br><br>",
             "",
             "```",
             "f([[arrayant]]); // http://quadriga-lib.org/faq.html <br>",
             "```",
             "  ```python",
             "  x = 1",
             "  ```"]
    out = render_md.twin_lines(lines, "api/cpp/x.md", "cpp", fence_lang="cpp", docblock=True, html=True)
    assert out == ["### Inputs:",
                   "- See [arrayant_interpolate_multi](miscellaneous-tools.md#arrayant_interpolate_multi) and "
                   "[.interpolate](array-antenna-class.md#.interpolate), not `[[arrayant]]`",
                   '- [QDANT](../../formats.md#6cab4884) and `<a href="x.html">raw</a>`',
                   "- [site](../../faq.md#cite), `[code](http://quadriga-lib.org/faq.html)`",
                   "- First line\\",
                   "  continues",
                   "",
                   "```cpp",
                   "f([[arrayant]]); // http://quadriga-lib.org/faq.html <br>",
                   "```",
                   "  ```python",
                   "  x = 1",
                   "  ```"]


def test_twin_lines_code_span_across_lines_and_warning():
    lines = ["Text `code with [link](a.html)", "still code` and [link](faq.html).", "", "Open `run [x](faq.html)"]
    diag = cli.Diagnostics()
    out = render_md.twin_lines(lines, "formats.md", diag=diag, path="p.md", first_line=10)
    assert out == ["Text `code with [link](a.html)", "still code` and [link](faq.md).", "", "Open `run [x](faq.html)"]
    assert diag.lines == ["p.md:13: warning: backtick run without a closing run in its paragraph; "
                          "the rest of the paragraph is left unrewritten [code-span]"]


def test_twin_lines_prose_headings():
    lines = ["## Material Properties {#material-properties}", "", "### Cite {#cite}", "",
             "#### Detail {#ignored}"]
    out = render_md.twin_lines(lines, "formats.md", headings=prose.heading_ids(lines))
    assert out == ['<a id="material-properties"></a>', "## Material Properties", "",
                   '<a id="cite"></a>', "### Cite", "", "#### Detail"]
    cell = ["## Step", "### Deeper", "#### Deepest"]
    out = render_md.twin_lines(cell, "tutorials/x.md", headings=prose.heading_ids(cell))
    assert out == ['<a id="step"></a>', "## Step", '<a id="deeper"></a>', "### Deeper", "#### Deepest"]


def test_twin_header_topics_column():
    rows = render_md.index_rows(prose.heading_ids(["## A", "## B", "### B1"]))
    assert render_md.twin_header("T", "u", rows)[-4:] == [
        "| Section | Topics |", "| --- | --- |", "| [A](#a) |  |", "| [B](#b) | [B1](#b1) |"]


def test_expected_files(docs):
    expected = {"README.md", "llms.txt", "download.md", "faq.md", "ai_agents.md", "publications.md", "formats.md",
                "formats/qdant.md", "formats/materials.md"}
    for lang in ("cpp", "mex", "python"):
        expected |= {"quadriga_lib_api_%s.md" % lang, "api/%s/general-usage-notes.md" % lang,
                     "api/%s/miscellaneous-tools.md" % lang}
    assert expected <= set(docs)


def test_no_source_syntax_outside_code(docs):
    for path, text in docs.items():
        masked = "\n".join(render_md.code_mask(text.split("\n")))
        for syntax in ("{#", "[[", "%%GAP", "legacy="):
            assert syntax not in masked, (path, syntax)


def test_cross_language_lines(docs):
    """Every function has its counterpart line; links go to the full file or the section file of that language."""
    for path, text in docs.items():
        if not (path.startswith("quadriga_lib_api_") or path.startswith("api/")):
            continue
        functions = text.count('\n## ')
        lines = [ln for ln in text.split("\n") if ln.startswith(("Also in: ", "Not in: "))]
        assert len(lines) == functions, path
        for target in re.findall(r"\[(?:C\+\+|MATLAB|Python)\]\(([^)#]+)#", "\n".join(lines)):
            if path.startswith("api/"):
                assert re.fullmatch(r"\.\./(cpp|mex|python)/[a-z0-9-]+\.md", target), (path, target)
            else:
                assert re.fullmatch(r"quadriga_lib_api_(cpp|mex|python)\.md", target), (path, target)
    assert "Not in: MATLAB, Python" in docs["quadriga_lib_api_cpp.md"]


def test_llms_absolute_and_readme_relative(docs):
    llms, readme = docs["llms.txt"], docs["README.md"]
    assert re.findall(r"\]\((?!http)[^)]*\)", llms) == []
    assert "](faq.md)" in readme and "](api/cpp/general-usage-notes.md)" in readme


def test_no_date_in_any_output(docs):
    for path, text in docs.items():
        assert not re.search(r"\b(19|20)\d\d-\d\d-\d\d\b|\b\d\d\.\d\d\.(19|20)\d\d\b", text), path


def test_bare_tilde_fence_gets_the_page_language():
    out = render_md.twin_lines(["~~~", "x = 1;", "~~~", "~~~python", "y", "~~~"], "api/cpp/x.md", fence_lang="cpp",
                               docblock=True, html=True)
    assert out == ["~~~cpp", "x = 1;", "~~~", "~~~python", "y", "~~~"]
