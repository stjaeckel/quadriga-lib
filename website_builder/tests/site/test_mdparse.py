# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""The markdown-it set-up and the builder's rules, on fragments rendered for one output location."""

import pytest

from qdocs import cli, prose


@pytest.fixture
def ctx(mini_repo):
    """A render context for cpp_api.html, with the link index of the fixture repository."""
    from qsite import render_html
    diag = cli.Diagnostics()
    cli.build_docs(mini_repo, diag)
    assert not diag.errors
    c = render_html.Context(cli.Diagnostics())
    c.location = "cpp_api.html"
    return c


def doc(ctx, text, lang="cpp"):
    return str(ctx.fragment(text.split("\n"), "docblock", "src/x.cpp", 10, lang))


def prose_html(ctx, text):
    lines = text.split("\n")
    return str(ctx.fragment(lines, "prose", "p.md", 1, headings=prose.heading_ids(lines)))


def test_wikilink_with_member_form(ctx):
    html = doc(ctx, "See [[arrayant_interpolate_multi]] and .[[interpolate]].")
    assert '<a href="#arrayant_interpolate_multi">arrayant_interpolate_multi</a>' in html
    assert 'and <a href="#.interpolate">.interpolate</a>.' in html
    ctx.location = "mex_api.html"
    assert '<a href="#version">VERSION</a>' in doc(ctx, "[[VERSION]]", "mex")  # text as written, as in the twin
    ctx.location = "tutorials/x.html"
    assert 'href="../cpp_api.html#.interpolate"' in doc(ctx, ".[[interpolate]]")


def test_raw_a_drops_target_and_rel(ctx):
    html = doc(ctx, 'A <a target="_blank" rel="noopener" href="formats.html#qdant">QDANT</a> file')
    assert 'A <a href="formats.html#qdant">QDANT</a> file' in html


def test_raw_br_trailing_dropped(ctx):
    html = doc(ctx, "first<br>\nsecond<br>")
    assert "first<br>\nsecond</p>" in html


def test_raw_html_in_prose_is_text(ctx):
    assert "&lt;b&gt;bold&lt;/b&gt;" in prose_html(ctx, "Some <b>bold</b> text")


def test_heading_attrs(ctx):
    html = prose_html(ctx, '## Material Properties {#materials}\n\n### Plain `code` [[x]]\n\n#### Deep {#d}')
    assert '<h2 id="materials">Material Properties</h2>' in html
    assert '<h3 id="plain-code-x">Plain <code>code</code> [[x]]</h3>' in html
    assert "<h4>Deep</h4>" in html


def test_alert_boxes(ctx):
    html = prose_html(ctx, "> [!NOTE]\n> Body text.\n\n> [!WARNING]\n> **Placeholder**\n> Draft.\n\n> [!NOTE] not alone\n")
    assert '<div class="note-box"><b>Note</b>\n<p>Body text.</p>\n</div>' in html
    assert '<div class="note-box warn"><b>Placeholder</b>\n<p>Draft.</p>\n</div>' in html
    assert "<blockquote>\n<p>[!NOTE] not alone</p>" in html


def test_site_url(ctx):
    html = prose_html(ctx, "[a](http://www.quadriga-lib.org/faq.html#cite) [b](https://quadriga-lib.org/) "
                           "[c](http://quadriga-lib.org/download/x.zip) [d](http://quadriga-lib.org/releases/)")
    assert 'href="faq.html#cite"' in html and 'href="index.html"' in html
    assert 'href="http://quadriga-lib.org/download/x.zip"' in html
    assert 'href="http://quadriga-lib.org/releases/"' in html


def test_dollar_amounts_are_not_math(ctx):
    html = prose_html(ctx, "It costs $5 and $6.")
    assert "It costs $5 and $6." in html and not ctx.math


def test_code_untouched(ctx):
    html = doc(ctx, "`[[interpolate]]` and `<a href=\"x\">y</a>`\n\n```\n[[interpolate]] <br>\n```")
    assert "<code>[[interpolate]]</code>" in html and "<code>&lt;a href=&quot;x&quot;&gt;y&lt;/a&gt;</code>" in html
    assert "[[interpolate]] &lt;br&gt;" in html


def test_setext_heading_is_an_error(ctx):
    prose_html(ctx, "Title\n-----\n\nText")
    assert ctx.diag.errors == 1 and "p.md:1: error: E-heading" in ctx.diag.lines[0]
