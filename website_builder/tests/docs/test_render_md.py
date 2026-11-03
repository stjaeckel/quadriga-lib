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
    out = render_md.twin_lines(lines, "api/cpp/x.md", "cpp", fence_lang="cpp", docblock=True)
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
    lines = ["## Material Properties {#material-properties legacy=\"9f6922f9\"}", "", "### Cite {#cite}", "",
             "#### Detail {#ignored}"]
    out = render_md.twin_lines(lines, "formats.md", headings=prose.heading_ids(lines, 0))
    assert out == ['<a id="material-properties"></a><a id="9f6922f9"></a>', "## Material Properties", "",
                   '<a id="cite"></a>', "### Cite", "", "#### Detail"]
    cell = ["# Set-up", "## Step", "### Deeper", "#### Deepest"]
    out = render_md.twin_lines(cell, "tutorials/x.md", headings=prose.heading_ids(cell, 1), level_shift=1)
    assert out == ['<a id="set-up"></a>', "## Set-up", '<a id="step"></a>', "### Step", "#### Deeper", "#### Deepest"]


def test_full_api_file(docs):
    text = docs["quadriga_lib_api_cpp.md"]
    assert text.startswith('---\ntitle: "C++ API Documentation for Quadriga-Lib v9.9.9"\nauthor: "Stephan Jaeckel"\n'
                           'lang: en-US\n---\n\n<a id="sec-general-usage-notes"></a>\n# General usage notes\n')
    assert "date:" not in text and "Line" not in text
    assert "| Function | Section | Summary |\n| --- | --- | --- |\n| [arrayant](#arrayant) | Array antenna class |" in text
    assert '---\n\n<a id="sec-miscellaneous-tools"></a><a id="b11e2d16"></a>\n# Miscellaneous / Tools\n' in text
    assert ('---\n<a id=".interpolate"></a>\n## .interpolate\nInterpolate the field patterns\n\n'
            "Also in: [MATLAB](quadriga_lib_api_mex.md#arrayant_interpolate) · "
            "[Python](quadriga_lib_api_python.md#interpolate)\n") in text
    assert "## quadriga_lib_version\nReturns the library version\n\nNot in: MATLAB, Python\n" in text
    assert "### Declaration:\n```cpp\n" in text
    mex = docs["quadriga_lib_api_mex.md"]
    assert "## version\nReturns the quadriga-lib version number\n\nNot in: C++, Python\n" in mex
    assert "### Usage:\n```matlab\n" in mex
    py = docs["quadriga_lib_api_python.md"]
    assert "## components\nReturns the compiled components\n\nNot in: C++, MATLAB\n" in py


def test_section_file(docs):
    text = docs["api/cpp/array-antenna-class.md"]
    assert text.startswith("# Array antenna class\n\nQuadriga-Lib 9.9.9 · Web page: " + BASE +
                           "/cpp_api.html#sec-array-antenna-class\n\n| Function | Summary |\n| --- | --- |\n"
                           "| [arrayant](#arrayant) | Class for storing array antenna models |\n")
    assert ("Also in: [MATLAB](../mex/array-antenna-functions.md#arrayant_interpolate) · "
            "[Python](../python/array-antenna-functions.md#interpolate)") in text
    assert "[QDANT](../../formats.md#6cab4884)" in text
    assert "[material model](%s/quadriga_lib_material_model.md)" % BASE in text
    notes = docs["api/python/general-usage-notes.md"]
    assert notes == "# General usage notes\n\nQuadriga-Lib 9.9.9 · Web page: %s/python_api.html#sec-general-usage-notes\n" % BASE


def test_prose_twin(docs):
    text = docs["faq.md"]
    assert text.startswith("# Questions\n\nQuadriga-Lib 9.9.9 · Web page: %s/faq.html\n\n"
                           "| Section | Topics |\n| --- | --- |\n"
                           "| [About the Library](#about-the-library) | [How do I cite Quadriga-Lib?](#cite) · " % BASE)
    assert '<a id="cite"></a>\n### How do I cite Quadriga-Lib?\n' in text
    assert "{#" not in text
    formats = docs["formats.md"]
    assert "> [!NOTE]\n> **Units**\n> Angles are in degrees.\n" in formats
    assert "| [Entry `<arrayant>`](#entry-arrayant) |" in formats
    download = docs["download.md"]
    assert "\n\nQuadriga-Lib 9.9.9 for C++, MATLAB / Octave and Python.\n\n" in download  # lead, gap removed
    assert "g++ -I armadillo-1.0.0/include main.cpp" in download
    assert "[release archive](http://quadriga-lib.org/releases/)" in download
    pubs = docs["publications.md"]
    assert '<a id="quadriga-lib"></a>\n## Quadriga-Lib\n\nPapers about the library.\n\n- 2025 · **A modern C++' in pubs
    assert "[PDF](%s/publications/2018_sum_of_sinusoids.pdf)" % BASE in pubs and "[FAQ](faq.md#cite)" in pubs


def test_llms_and_readme(docs):
    llms, readme = docs["llms.txt"], docs["README.md"]
    assert llms.startswith("# Quadriga-Lib\n\n> A radio channel modelling library for C++, MATLAB / Octave and Python.\n"
                           "\nVersion 9.9.9. Every page")
    assert "- [Questions](%s/faq.md): FAQ, how to cite\n" % BASE in llms
    assert "- [C++: Miscellaneous / Tools](%s/api/cpp/miscellaneous-tools.md): 2 functions\n" % BASE in llms
    assert "- [MATLAB / Octave API, complete](%s/quadriga_lib_api_mex.md)\n" % BASE in llms
    assert "- [Agent skill (placeholder)](%s/skill/SKILL.md)\n" % BASE in llms
    assert "Generated by `make docs`" in readme and "Generated" not in llms
    assert "- [Questions](faq.md): FAQ, how to cite\n" in readme
    assert "- [Tutorials](%s/tutorials.md): " % BASE in readme
    assert "- [Python: General usage notes](api/python/general-usage-notes.md)\n" in readme
    assert re.findall(r"\]\((?!http)[^)]*\)", llms) == []


def test_no_date_in_any_output(docs):
    for path, text in docs.items():
        assert not re.search(r"\b(19|20)\d\d-\d\d-\d\d\b|\b\d\d\.\d\d\.(19|20)\d\d\b", text), path
