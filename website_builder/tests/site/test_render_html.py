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
    assert faq.startswith('<!doctype html>\n<html lang="en" class="no-js">')
    assert "<title>Questions · Quadriga-Lib</title>" in faq and "<title>Quadriga-Lib</title>" in index
    assert '<meta property="og:url" content="http://quadriga-lib.org/faq.html">' in faq
    assert '<link rel="alternate" type="text/markdown" href="faq.md">' in faq
    assert '<a href="faq.md">This page as Markdown</a>' in faq
    assert 'type="text/markdown"' not in index and "This page as Markdown" not in index
    assert "og:image" not in faq and "twitter:card" not in faq
    assert '<meta name="theme-color" content="#24603a" media="(prefers-color-scheme: dark)">' in faq
    assert '<script src="js/site.js" defer></script>' in faq


def test_og_image_only_with_the_image(mini_repo, run_site, static_copy):
    (static_copy / "img/og-card.png").write_bytes(b"\x89PNG")
    code, out = run_site(mini_repo)
    assert code == 0 and "og-card.png" not in out
    faq = (mini_repo / "website/faq.html").read_text(encoding="utf-8")
    assert '<meta property="og:image" content="http://quadriga-lib.org/img/og-card.png">' in faq
    assert '<meta name="twitter:card" content="summary_large_image">' in faq


def test_tabs_and_menu(site):
    faq = site("faq.html")
    assert '<li><a href="faq.html" aria-current="page">FAQ</a></li>' in faq
    assert '<p class="menu-group">More</p><a href="formats.html">Formats</a>' in faq
    assert faq.count('aria-current="page"') == 2  # tab row and menu
    assert '<a href="tutorials.html" aria-current="page">Tutorials</a>' in site("tutorials/mini_demo.html").replace("../", "")


def test_api_page(site):
    cpp = site("cpp_api.html")
    assert '<section class="api-section" id="sec-array-antenna-class"><h2 id="3a0c780f">Array antenna class</h2>' in cpp
    assert '<section class="api-section" id="sec-general-usage-notes"><h2>General usage notes</h2>' in cpp
    assert '<article class="fn-entry" id=".interpolate">' in cpp
    assert ('<a class="pill" aria-current="page" href="#.interpolate">C++</a><a class="pill" '
            'href="mex_api.html#arrayant_interpolate">MATLAB</a><a class="pill" href="python_api.html#interpolate">'
            'Python</a><a class="md" href="api/cpp/array-antenna-class.md">Markdown</a>') in cpp
    assert '<span class="pill none" title="No MATLAB counterpart">MATLAB –</span>' in cpp
    assert '<p class="fn-brief">Interpolate the field patterns</p>' in cpp
    assert '<div><dt>elevation<span class="opt">optional</span></dt><dd>Elevation angles in rad</dd></div>' in cpp
    assert '<div class="pnote"><dt></dt><dd>A note about both inputs:<br' in cpp
    assert '<span class="count">2</span>' in cpp and '<li class="fn"><a href="#.interpolate">.interpolate</a></li>' in cpp
    assert "Home</a> › <a href=\"index.html#apis\">API reference</a> › C++</p>" in cpp
    mex = site("mex_api.html")
    assert '<dt>V_re, V_im</dt>' in mex and '<article class="fn-entry" id="arrayant_interpolate">' in mex
    assert "<h1>MATLAB / Octave API</h1>" in mex


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
    formats = site("formats.html")
    side = ('<ul><li><details open><summary><span>QuaDRiGa Array Antenna Exchange Format (QDANT)</span></summary><ul>'
            '<li><a href="#qdant">Overview</a></li><li class="fn"><a href="#entry-arrayant">Entry <code>&lt;arrayant&gt;</code></a>'
            '</li></ul></details></li><li><a href="#material-properties">Material Properties</a></li></ul>')
    assert '<nav class="doc-side idx" aria-label="Page sections">' + side in formats
    assert '<details class="doc-index"><summary>On this page</summary>' in formats
    assert '<div class="note-box"><b>Units</b>' in formats
    assert '<div class="note-box"><b>Note</b>' in site("privacy.html")
    assert '<div class="note-box warn"><b>Placeholder</b>' in site("impressum.html")
    assert "doc-side" not in site("impressum.html")


def test_code_colouring(site):
    index = site("index.html")
    assert '<code class="language-python"><span class="k">import</span> <span class="kt">quadriga_lib</span>' in index
    assert '<code class="language-bash">pip install quadriga-lib</code>' in index
    assert '<span class="cp">' not in index and '<span class="w">' not in index
    cpp = site("cpp_api.html")
    assert '<span class="kt">void</span> <span class="nf">interpolate</span>(<span class="k">const</span>' in cpp
    assert '<span class="c">// see [[not_a_link]]</span></code>' in cpp


def test_highlight_filter():
    from qsite.render_html import highlight
    assert str(highlight("x = 1  # n\n", "python")) == 'x = <span class="m">1</span>  <span class="c"># n</span>'
    assert str(highlight("$ ls <a>", "console")) == '<span class="gp">$ </span>ls &lt;a&gt;'
    assert str(highlight("a < b", "text")) == "a &lt; b"


def test_404_root_absolute_and_tutorial_relative(site):
    nf = site("404.html")
    assert 'href="/css/site.css"' in nf and 'href="/cpp_api.html">C++ API</a>' in nf and 'href="/mex_api.html"' in nf
    tut = site("tutorials/mini_demo.html")
    assert 'href="../css/site.css"' in tut and '<a href="../tutorials.html">Tutorials</a>' in tut
    assert '<link rel="alternate" type="text/markdown" href="../tutorials/mini_demo.md">' in tut


def test_gap_markers_kept_as_placeholders(site):
    formats = site("formats.html")
    assert '<p><span class="ph">an introduction to the formats</span></p>' in formats
    index = site("index.html")
    assert '<span class="ph">Realistic radio channels for mobile and Wi-Fi network simulation</span>' in index
    assert '<meta name="description" content="Open-source radio channel modelling' in index
    assert "%%GAP" not in index and "%%GAP" not in site("publications.html")
    assert '<meta name="description" content="page description">' in site("publications.html")
    tuts = site("tutorials.html")
    assert '<p class="lead">Worked examples. <span class="ph">lead text</span></p>' in tuts
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


def test_static_files(site, mini_repo):
    assert site(".htaccess").startswith("Options -Indexes\nErrorDocument 404 /404.html\n")
    assert not (mini_repo / "website/img/landing/README.md").exists()
    assert (mini_repo / "website/fonts/LICENSE.txt").is_file()


def test_landing_figures_and_size_warning(mini_repo, run_site, static_copy):
    big = b"\0" * 400_000
    for v in ("light", "dark"):
        (static_copy / ("img/landing/coverage-%s.webp" % v)).write_bytes(big)
        (static_copy / ("img/landing/antenna-%s.webp" % v)).write_bytes(b"\0" * 1000)
    code, out = run_site(mini_repo)
    assert code == 0, out
    assert "website/img/landing/coverage-light.webp: warning: landing figure is 400 kB, budget 150 kB [size]" in out
    assert "website/index.html: warning: landing page with styles, script, fonts and figures is" in out
    assert "landing figure missing" not in out
    index = (mini_repo / "website/index.html").read_text(encoding="utf-8")
    assert ('<picture><source media="(prefers-color-scheme: dark)" srcset="img/landing/coverage-dark.webp">'
            '<img src="img/landing/coverage-light.webp" width="1600" height="1200" alt="Placeholder for a coverage map"'
            ' decoding="async"></picture>') in index
    assert 'alt="Placeholder for an antenna pattern" decoding="async" loading="lazy">' in index


def test_empty_figure_slots(site):
    index = site("index.html")
    assert '<div class="frame frame-empty"><span>Placeholder for a coverage map</span></div>' in index
    assert '<section class="section" aria-labelledby="quickstart">' in index and 'id="quickstart"' in index
