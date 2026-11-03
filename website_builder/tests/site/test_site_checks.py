# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Checks of the built site: links and anchors, well-formedness, size budgets."""

from qdocs import cli
from qsite import checks

PAGE = '<!doctype html><html><body><main id="content"><h1>T</h1>%s</main></body></html>'


def run(files, markdown=()):
    diag = cli.Diagnostics()
    mapping = {p: (PAGE % v if p.endswith(".html") and not v.startswith("<!") else v).encode("utf-8")
               for p, v in files.items()}
    checks.check_site(mapping, diag, list(markdown))
    return diag


def test_links_and_anchors():
    d = run({"a.html": '<a href="b.html">b</a><a href="b.html#x">x</a><a href="c.html">c</a><a href="#y">y</a>'
                       '<img src="/img/p.png" alt=""><a href="https://example.org/">e</a><a href="mailto:a@b">m</a>'
                       '<a href="t.md#t1">t</a><a href="t.md#t2">t2</a><a href="h.md#any">h</a>',
             "b.html": '<p id="z">z</p>', "img/p.png": "", "t.md": '<a id="t1"></a>\n', "h.md": "hand-made"},
            markdown=["t.md"])
    assert d.lines == ["website/a.html:1: error: anchor missing: b.html#x (no id 'x' in b.html)",
                       "website/a.html:1: error: link target missing: c.html",
                       "website/a.html:1: error: anchor missing: #y (no id 'y' in a.html)",
                       "website/a.html:1: error: anchor missing: t.md#t2 (no <a id> 't2' in t.md)"]


def test_absolute_site_link_in_a_twin():
    d = run({"a.md": "[x](http://quadriga-lib.org/b.md#y)\n", "b.md": '<a id="z"></a>\n'}, markdown=["a.md", "b.md"])
    assert d.errors == 1 and "website/a.md:1: error: anchor missing" in d.lines[0]


def test_faq_without_cite(mini_repo, run_site):
    # In HTML: the footer link of every page
    d = run({"index.html": '<a href="faq.html#cite">How to cite</a>', "faq.html": '<h2 id="about">About</h2>'})
    assert d.lines == ["website/index.html:1: error: anchor missing: faq.html#cite (no id 'cite' in faq.html)"]
    # In the Markdown: the twin links stop the build already in the docs phase
    faq = mini_repo / "docs_builder/content/faq.md"
    faq.write_text(faq.read_text(encoding="utf-8").replace(" {#cite}", ""), encoding="utf-8")
    code, out = run_site(mini_repo)
    assert code == 1 and "docs/publications.md:10: error: anchor missing: faq.md#cite (no id 'cite' in faq.md)" in out
    assert not (mini_repo / "website").exists()


def test_well_formedness():
    d = run({"a.html": "<div><p>x</div>", "b.html": '<p id="q"></p><p id="q"></p><img src="b.html">',
             "c.html": "<h1>again</h1>", "e.html": "<section>open", "d.html": "<!doctype html><html><body><h1>x</h1></body></html>"})
    text = "\n".join(d.lines)
    assert "website/a.html:1: error: </div> closes <p> of line 1" in text
    assert "website/b.html:1: error: <img> without alt" in text
    assert "website/e.html:1: error: </main> closes <section> of line 1" in text
    assert "website/b.html:1: error: id 'q' appears twice" in text
    assert "website/c.html: error: 2 <h1> elements, expected exactly one" in text
    assert 'website/d.html: error: 0 <main id="content"> elements, expected exactly one' in text


def test_size_warnings():
    d = run({"x_api.html": "a" * 810_000, "faq.html": "a" * 160_000, "tutorials/x_api.html": "a" * 160_000})
    assert d.errors == 0
    assert d.lines == ["website/faq.html: warning: page is 160 kB, budget 150 kB [size]",
                       "website/tutorials/x_api.html: warning: page is 160 kB, budget 150 kB [size]",
                       "website/x_api.html: warning: page is 810 kB, budget 800 kB [size]"]
