# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Rendered pages of the fixture repository, checked by substrings."""

import shutil

import pytest


@pytest.fixture
def site(mini_repo, run_site):
    """Build the fixture site; site(path) -> text of a file in website/."""
    code, out = run_site(mini_repo)
    assert code == 0, out
    return lambda rel: (mini_repo / "website" / rel).read_text(encoding="utf-8")


@pytest.fixture
def static_copy(tmp_path, monkeypatch):
    """A copy of static/ that the build reads instead of the code folder."""
    from qsite import assets
    copy = tmp_path / "static"
    shutil.copytree(str(assets.STATIC_DIR), str(copy))
    monkeypatch.setattr(assets, "STATIC_DIR", copy)
    return copy


def test_head_tags(site):
    faq, index = site("faq.html"), site("index.html")
    assert '<link rel="alternate" type="text/markdown" href="faq.md">' in faq
    assert 'type="text/markdown"' not in index
    assert "og:image" not in faq


def test_og_image_only_with_the_image(mini_repo, run_site, static_copy):
    (static_copy / "img/og-card.png").write_bytes(b"\x89PNG")
    code, out = run_site(mini_repo)
    assert code == 0 and "og-card.png" not in out
    faq = (mini_repo / "website/faq.html").read_text(encoding="utf-8")
    assert '<meta property="og:image" content="http://quadriga-lib.org/img/og-card.png">' in faq
    assert '<meta name="twitter:card" content="summary_large_image">' in faq


def test_tabs_and_menu(site):
    assert site("faq.html").count('aria-current="page"') == 2  # tab row and menu
    tut = site("tutorials/mini_demo.html")
    assert tut.count('<a href="../tutorials.html" aria-current="page">') == 2


def test_api_page(site):
    cpp = site("cpp_api.html")
    assert 'id="sec-array-antenna-class"' in cpp and 'id="sec-general-usage-notes"' in cpp
    assert 'id=".interpolate"' in cpp
    assert 'href="mex_api.html#arrayant_interpolate">MATLAB</a>' in cpp
    assert 'href="python_api.html#interpolate">Python</a>' in cpp
    assert 'title="No MATLAB counterpart">MATLAB –</span>' in cpp
    assert '<dt>elevation<span class="opt">optional</span></dt>' in cpp
    mex = site("mex_api.html")
    assert "<dt>V_re, V_im</dt>" in mex and 'id="arrayant_interpolate"' in mex


def test_sub_block_label_as_inline_markdown(mini_repo, run_site):
    src = mini_repo / "src/qd_tools.cpp"
    src.write_text(src.read_text(encoding="utf-8").replace("## Inputs:", "## Inputs (required when `ant` is `[]`):"),
                   encoding="utf-8")
    code, out = run_site(mini_repo)
    assert code == 0, out
    cpp = (mini_repo / "website/cpp_api.html").read_text(encoding="utf-8")
    assert "<h4>Inputs (required when <code>ant</code> is <code>[]</code>)</h4>" in cpp
    assert '<dl class="params"><div><dt>ant</dt>' in cpp  # still a parameter list


def test_text_page_index_and_note_boxes(site):
    assert "doc-side" in site("faq.html") and "doc-side" not in site("impressum.html")
    assert '<div class="note-box"><b>Note</b>' in site("privacy.html")
    assert '<div class="note-box warn"><b>Placeholder</b>' in site("impressum.html")


def test_formats_page(site):
    formats = site("formats.html")
    assert 'id="qdant"' in formats and 'id="materials"' in formats
    assert '<h3 id="entry-arrayant">' in formats and '<h3 id="notes-2">' in formats  # ## moves down to h3
    assert formats.count('href="formats/qdant.md"') == 1 and formats.count('href="formats/materials.md"') == 1


def test_code_colouring(site):
    cpp = site("cpp_api.html")
    assert '<span class="kt">void</span>' in cpp
    assert '<span class="cp">' not in cpp and '<span class="w">' not in cpp
    assert "// see [[not_a_link]]</span>" in cpp  # no link inside a code comment


def test_404_root_absolute_and_tutorial_relative(site):
    nf = site("404.html")
    assert 'href="/css/site.css"' in nf and 'href="/cpp_api.html">C++ API</a>' in nf and 'href="/mex_api.html"' in nf
    tut = site("tutorials/mini_demo.html")
    assert 'href="../css/site.css"' in tut and '<a href="../tutorials.html">Tutorials</a>' in tut
    assert '<link rel="alternate" type="text/markdown" href="../tutorials/mini_demo.md">' in tut


def test_gap_markers_kept_as_placeholders(site):
    assert '<span class="ph">an introduction to the formats</span>' in site("formats.html")
    assert '<p class="lead">Worked examples. <span class="ph">lead text</span></p>' in site("tutorials.html")
    assert '<meta name="description" content="page description">' in site("publications.html")
    for page in ("index.html", "formats.html", "publications.html", "tutorials.html"):
        assert "%%GAP" not in site(page)
    # The Markdown leaves the markers out
    assert "GAP" not in site("formats.md") and "lead text" not in site("tutorials.md")


def test_title_and_attribute_gaps(mini_repo, run_site):
    page = mini_repo / "website_builder/content/privacy.md"
    page.write_text(page.read_text(encoding="utf-8").replace('title = "Privacy"', 'title = "%%GAP: Privacy%%"'),
                    encoding="utf-8")
    assert run_site(mini_repo)[0] == 0
    html = (mini_repo / "website/privacy.html").read_text(encoding="utf-8")
    assert "<title>Privacy · Quadriga-Lib</title>" in html and '<meta property="og:title" content="Privacy">' in html
    assert '<h1><span class="ph">Privacy</span></h1>' in html


def test_release_fails_on_gaps(mini_repo, run_site):
    code, out = run_site(mini_repo, "--release")
    assert code == 1 and "website --release: error:" in out
    assert not (mini_repo / "website").exists()


def test_release_fails_on_replaced_markers(mini_repo, run_site, monkeypatch):
    from qsite import cli
    monkeypatch.setattr(cli.docs_cli.Diagnostics, "gap", lambda self, *a: None)  # an empty gap list
    monkeypatch.setattr(cli.prose, "gap_scan", lambda *a: [])
    code, out = run_site(mini_repo, "--release")
    assert code == 1 and "0 gaps on the list and " in out and "gap markers in the HTML" in out


def test_landing_figures_and_size_warning(mini_repo, run_site, static_copy):
    from qsite import assets
    base = assets.figure(assets.LANDING_FIGURES[0])
    (static_copy / base).parent.mkdir(parents=True, exist_ok=True)
    for v in ("light", "dark"):
        (static_copy / ("%s-%s.webp" % (base, v))).write_bytes(b"\0" * 400_000)
    code, out = run_site(mini_repo)
    assert code == 0, out
    assert "website/%s-light.webp: warning: " % base in out and "[size]" in out
    index = (mini_repo / "website/index.html").read_text(encoding="utf-8")
    assert '<source media="(prefers-color-scheme: dark)" srcset="%s-dark.webp"><img src="%s-light.webp"' % (base, base) in index
