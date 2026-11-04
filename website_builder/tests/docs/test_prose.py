# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Prose pages: front matter, publications data, build variables, heading ids, gap markers."""

import pytest

from qdocs import cli, config, prose, render_md

CONTENT = "docs_builder/content"


def page(tmp_path, text, name="page.md"):
    (tmp_path / name).write_text(text, encoding="utf-8")
    diag = cli.Diagnostics()
    return prose.read_page(tmp_path / name, diag, tmp_path), diag


def errors(diag):
    return [ln for ln in diag.lines if ": error: " in ln]


FRONT = '+++\ntitle = "T"\ndescription = "D"\n+++\n'


def test_front_matter(tmp_path):
    p, diag = page(tmp_path, '+++\ntitle = "T"\ndescription = "D"\nlead = "L"\n+++\n\nBody\n')
    assert errors(diag) == [] and (p.title, p.description, p.lead, p.body, p.body_line) == ("T", "D", "L", ["", "Body"], 6)
    assert p.page_id == "page"


@pytest.mark.parametrize("text, message", [
    ("title = 1\n", "page.md: error: front matter: title: not a string"),
    ('title = "T"\n', "page.md: error: front matter: description: missing"),
    ('title = "T"\ndescription = "D"\ncolour = "red"\n', "page.md: error: front matter: colour: unknown key"),
    ('title = "T"\ndescription = "D"\ndummy_note = "x"\n', "page.md: error: front matter: dummy_note: unknown key"),
    ('title = = "T"\n', "page.md:2: error: TOML: "),
])
def test_front_matter_errors(tmp_path, text, message):
    _, diag = page(tmp_path, "+++\n" + text + "+++\nBody\n")
    assert any(e.startswith(message) for e in errors(diag)), diag.lines


def test_front_matter_extra_keys_and_missing_delimiters(tmp_path):
    p, diag = page(tmp_path, '+++\ntitle = "T"\ndescription = "D"\nllms_summary = "S"\n+++\n', "ai_agents.md")
    assert errors(diag) == [] and p.extra == {"llms_summary": "S"}
    _, diag = page(tmp_path, "# No front matter\n")
    assert errors(diag) == ["page.md:1: error: the file must start with TOML front matter between two lines '+++'"]


def test_e_table_and_e_wrap_on_a_prose_page(tmp_path):
    _, diag = page(tmp_path, FRONT + "\n| a | b |\n| - | - |\nText\n\nWrapped and\n> 0\n")
    assert errors(diag) == ["page.md:8: error: E-table: a table and the text next to it need a blank line between them",
                            "page.md:11: error: E-wrap: a wrapped line must not start with '>' or '+ '; move the line break"]


def test_e_heading_on_a_prose_page(tmp_path):
    _, diag = page(tmp_path, FRONT + "\n# Second title\n\nA heading\n=========\n\n   ### Indented\n\n## Fine\n")
    assert [e.split(": E-heading: ")[0] for e in errors(diag)] == ["page.md:6: error", "page.md:8: error",
                                                                   "page.md:11: error"]
    assert "the title comes from the front matter" in errors(diag)[0]


def test_build_variables(tmp_path):
    config.VARIABLES["version"] = "9.9.9"
    p, diag = page(tmp_path, FRONT.replace('"T"', '"T {{ version }}"') +
                   "Version {{version}}, {{ github_url }}.\n```\nv{{ version }}\n```\n{{ nosuch }}\n")
    assert p.title == "T 9.9.9"
    assert p.body[:3] == ["Version 9.9.9, https://github.com/stjaeckel/quadriga-lib.", "```", "v9.9.9"]
    assert errors(diag) == ["page.md:9: error: unknown build variable '{{ nosuch }}'"]


def test_publications_key_check(tmp_path):
    good = ('[page]\ntitle = "P"\ndescription = "D"\nlead = "L"\n[[group]]\nid = "g"\ntitle = "G"\n'
            '[[group.entry]]\nyear = 2020\ntitle = "T"\nauthors = "A"\nvenue = "V"\nlink = [{label = "PDF", url = "x.pdf"}]\n')
    (tmp_path / "p.toml").write_text(good, encoding="utf-8")
    diag = cli.Diagnostics()
    assert prose.read_publications(tmp_path / "p.toml", diag, tmp_path)["group"][0]["id"] == "g"
    bad = (good.replace("year = 2020\n", "").replace('venue = "V"', 'venue = "V"\npages = "1-2"')
           .replace('[{label = "PDF", url = "x.pdf"}]', "[]") + '[[group]]\nid = "h"\ntitle = 3\n[[group.entry]]\n'
           'year = "2021"\ntitle = "T"\nauthors = "A"\nvenue = "V"\nlink = [{label = "L"}]\n')
    (tmp_path / "p.toml").write_text(bad, encoding="utf-8")
    diag = cli.Diagnostics()
    assert prose.read_publications(tmp_path / "p.toml", diag, tmp_path) is None
    assert [e.split("error: ")[1] for e in errors(diag)] == [
        "group[1].entry[1].pages: unknown key", "group[1].entry[1].year: missing", "group[1].entry[1].link: empty list",
        "group[2].title: not a string", "group[2].entry[1].year: not an integer", "group[2].entry[1].link[1].url: missing"]


def test_heading_ids_plain_text():
    lines = ["## Using `arrayant_generate`", "### A [link](faq.html#cite) and [[name]] and .[[member]]",
             "## Escaped \\*stars\\* and *emphasis*", "## Entities &amp; &lt;tags&gt;",
             "## How do I cite Quadriga-Lib? {#cite}", '## Material Properties {#material-properties}',
             "## Using `arrayant_generate`", "#### Not an h3", "```", "## In a fence", "```", " ## Indented"]
    heads = prose.heading_ids(lines)
    assert [(h.index, h.id) for h in heads] == [
        (0, "using-arrayant-generate"), (1, "a-link-and-name-and-member"), (2, "escaped-stars-and-emphasis"),
        (3, "entities-tags"), (4, "cite"), (5, "material-properties"), (6, "using-arrayant-generate-2"), (7, None)]
    assert heads[2].plain == "Escaped *stars* and emphasis" and heads[3].plain == "Entities & <tags>"
    assert heads[5].text == "Material Properties"
    # no level shift: only ## and ### get ids
    assert [h.id for h in prose.heading_ids(["# Title", "## Detail", "### Deeper"])] == [None, "detail", "deeper"]


def test_heading_ids_explicit_first_and_duplicates():
    heads = prose.heading_ids(["## Cite", "## Other {#cite}"])
    assert [h.id for h in heads] == ["cite-2", "cite"]
    diag = cli.Diagnostics()
    prose.heading_ids(["## A {#x}", "## B {#x}"], diag, "p.md", 5)
    assert errors(diag) == ["p.md:6: error: E-anchor: id 'x' appears twice on the page"]


def test_gap_markers_left_out_of_twins(tmp_path):
    text = ('+++\ntitle = "%%GAP: title%%"\ndescription = "D"\nlead = "%%GAP: lead draft%%"\n+++\n\n'
            "%%GAP: whole page is a draft%%\n\n## Section\n\nKept text %%GAP: more text%% and more.\n\n"
            "- one\n- %%GAP: second item%%\n- three\n\n> [!NOTE]\n> %%GAP: note text%%\n")
    p, diag = page(tmp_path, text, "faq.md")
    twin = render_md.prose_twin(p)
    assert "GAP" not in twin and "draft" not in twin
    assert twin.startswith("Quadriga-Lib ")  # the empty title is omitted, so is the empty lead
    assert "Kept text and more.\n\n- one\n- three\n\n> [!NOTE]\n" in twin
    assert "\n\n\n" not in twin


def test_gap_scan_and_e_gap(mini_repo, run_docs):
    code, out = run_docs(mini_repo)
    assert code == 0
    assert "docs_builder/content/formats.md:6: an introduction to the formats\n" in out
    assert "docs: 0 errors, 2 warnings, 7 gaps" in out
    f = mini_repo / CONTENT / "faq.md"
    text = f.read_text(encoding="utf-8") + "\n%%GAP: not closed\n"
    f.write_text(text, encoding="utf-8")
    code, out = run_docs(mini_repo)
    line = text.split("\n").index("%%GAP: not closed") + 1
    assert code == 1 and "docs_builder/content/faq.md:%d: error: E-gap: " % line in out


@pytest.mark.parametrize("name", ["faq.md", "api_intro_mex.md", "publications.toml", "api_matching.toml"])
def test_missing_content_file(mini_repo, run_docs, name):
    (mini_repo / CONTENT / name).unlink()
    code, out = run_docs(mini_repo)
    assert code == 1 and "%s/%s: error: content file missing" % (CONTENT, name) in out


def test_format_files(mini_repo, run_docs):
    folder = mini_repo / CONTENT / "formats"
    (folder / "zz.md").write_text('+++\ntitle = "Z"\nsummary = "S"\norder = 0\n+++\n\n## Notes {#qdant}\n', encoding="utf-8")
    code, out = run_docs(mini_repo)
    assert code == 1 and "docs_builder/content/formats/zz.md:7: error: E-anchor: id 'qdant' appears twice" in out
    (folder / "zz.md").write_text('+++\ntitle = "Z"\norder = "1"\n+++\n', encoding="utf-8")
    code, out = run_docs(mini_repo)
    assert "docs_builder/content/formats/zz.md: error: front matter: summary: missing" in out
    assert "docs_builder/content/formats/zz.md: error: front matter: order: not an integer" in out
    for p in folder.iterdir():
        p.unlink()
    code, out = run_docs(mini_repo)
    assert code == 1 and "docs_builder/content/formats: error: no format files (*.md) found" in out


def test_heading_ids_avoid_template_ids():
    heads = prose.heading_ids(["## Content", "### Quick start", "## Why"])
    assert [h.id for h in heads] == ["content-2", "quick-start", "why-2"]
    diag = cli.Diagnostics()
    prose.heading_ids(["## Main {#content}"], diag, "p.md", 3)
    assert errors(diag) == ["p.md:3: error: E-anchor: id 'content' is used by the page template; choose another"]
