# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Tutorial notebooks: executed copies, metadata checks, outputs, headings, pages and twins."""

import json
import re
import shutil

NB = "tutorials/mini_demo.ipynb"            # the notebook (cleared); messages name it
RUN = "tutorials/executed/mini_demo.ipynb"  # its executed copy, which the builder reads


def edit(root, change, name=NB):
    """Change the executed copy; with another name, rename the notebook and its copy."""
    path = root / RUN
    nb = json.loads(path.read_text(encoding="utf-8"))
    change(nb)
    path.unlink()
    (root / "tutorials/executed" / name.split("/")[-1]).write_text(json.dumps(nb), encoding="utf-8")
    if name != NB:
        (root / NB).rename(root / name)


def site(root, rel):
    return (root / "website" / rel).read_text(encoding="utf-8")


def both(root, change, stem="mini_demo"):
    """Change the tutorial metadata of a notebook and of its executed copy alike (a re-run notebook)."""
    for path in (root / "tutorials" / (stem + ".ipynb"), root / "tutorials/executed" / (stem + ".ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        change(nb["metadata"]["quadriga_tutorial"])
        path.write_text(json.dumps(nb), encoding="utf-8")


def second(root, **meta):
    """A second tutorial mini_zero, a copy of mini_demo with other metadata."""
    for folder in ("tutorials", "tutorials/executed"):
        shutil.copy(root / folder / "mini_demo.ipynb", root / folder / "mini_zero.ipynb")
    both(root, lambda m: m.update(meta), "mini_zero")


def test_metadata_errors(mini_repo, run_site):
    def bad(nb):
        meta = nb["metadata"]["quadriga_tutorial"]
        del meta["title"]
        meta["level"] = "expert"
        meta["order"] = "1"
        meta["dummy"] = True
        del nb["metadata"]["kernelspec"]["language"]
    edit(mini_repo, bad, "tutorials/Bad Name.ipynb")
    code, out = run_site(mini_repo)
    assert code == 1
    for msg in ("notebook file name must match", "metadata.quadriga_tutorial.title: missing",
                "metadata.quadriga_tutorial.order: not an integer", "metadata.quadriga_tutorial.level: one of",
                "metadata.quadriga_tutorial.dummy: unknown key",
                "metadata.kernelspec.language: missing"):
        assert "tutorials/Bad Name.ipynb: error: " + msg in out


def test_outputs(mini_repo, run_site):
    code, out = run_site(mini_repo)
    assert code == 0, out
    page = site(mini_repo, "tutorials/mini_demo.html")
    assert '<div class="out"><pre>9.9.9</pre></div>' in page  # ANSI codes removed
    assert '<img src="mini_demo/fig-4-1.svg" alt="A black square" loading="lazy">' in page  # SVG before PNG
    assert '<img src="mini_demo/fig-5-1.png" alt="A pixel" loading="lazy"></div><div class="out"><pre>1</pre>' in page
    assert (mini_repo / "website/tutorials/mini_demo/fig-5-1.png").read_bytes().startswith(b"\x89PNG")
    assert site(mini_repo, "tutorials/mini_demo/fig-4-1.svg").startswith("<svg")
    assert '<div class="cell-label">In [2]</div>' in page
    assert "tutorials/mini_demo.ipynb: warning" not in out


def test_output_warnings_and_error(mini_repo, run_site):
    def change(nb):
        cells = nb["cells"]
        cells[1]["execution_count"] = None
        cells[3]["metadata"] = {}
        cells[4]["outputs"].append({"output_type": "display_data", "metadata": {}, "data": {"application/json": {}}})
    edit(mini_repo, change)
    code, out = run_site(mini_repo)
    assert code == 0, out
    assert "tutorials/mini_demo.ipynb: warning: cell 2: code cell without an execution count (not run) [not-run]" in out
    assert "tutorials/mini_demo.ipynb: warning: cell 4: image without metadata.quadriga_tutorial.alt [figure-alt]" in out
    assert "tutorials/mini_demo.ipynb: warning: cell 5: output 3 has no SVG, PNG or plain text; left out [output-type]" in out
    page = site(mini_repo, "tutorials/mini_demo.html")
    assert 'alt="Figure output of cell 4"' in page and "In [ ]" in page

    edit(mini_repo, lambda nb: nb["cells"][1]["outputs"].append(
        {"output_type": "error", "ename": "ValueError", "evalue": "x", "traceback": []}))
    code, out = run_site(mini_repo)
    assert code == 1 and "cell 2: the executed copy has a failing cell (ValueError)" in out


def test_headings(mini_repo, run_site):
    def deep(nb):
        nb["cells"][2]["source"].append("\n##### Deep\n")
    edit(mini_repo, deep)
    assert run_site(mini_repo)[0] == 0
    page = site(mini_repo, "tutorials/mini_demo.html")
    assert '<h1>Mini demo</h1>' in page and '<h2 id="set-up">Set-up</h2>' in page
    assert '<h3 id="a-figure">A figure</h3>' in page and "<h4>Deep</h4>" in page
    twin = site(mini_repo, "tutorials/mini_demo.md")
    assert '<a id="set-up"></a>\n## Set-up' in twin and '<a id="a-figure"></a>\n### A figure' in twin
    assert "\n##### Deep\n" in twin  # the twin keeps the source level
    assert "| [Set-up](#set-up) |" in twin  # index of the two h2 headings


def test_heading_errors_name_the_cell(mini_repo, run_site):
    def twice(nb):
        nb["cells"][0]["source"][0] = "## Set-up {#setup}\n"
        nb["cells"][5]["source"].insert(0, "## Again {#setup}\n\n")
    edit(mini_repo, twice)
    code, out = run_site(mini_repo)
    assert code == 1
    assert "tutorials/mini_demo.ipynb: error: cell 6: E-anchor: id 'setup' appears twice on the page" in out


def test_level_one_heading_is_an_error(mini_repo, run_site):
    edit(mini_repo, lambda nb: nb["cells"][0]["source"].__setitem__(0, "# Set-up\n"))
    code, out = run_site(mini_repo)
    assert code == 1
    assert ("tutorials/mini_demo.ipynb: error: cell 1: a notebook starts its sections with ##; the page title "
            "comes from the notebook metadata") in out


def test_pages_twins_and_index(mini_repo, run_site):
    assert run_site(mini_repo)[0] == 0
    twin = site(mini_repo, "tutorials/mini_demo.md")
    assert twin.startswith("# Mini demo\n\nQuadriga-Lib 9.9.9 · Web page: http://quadriga-lib.org/tutorials/mini_demo.html")
    assert "[Download notebook .ipynb](mini_demo.ipynb) · [Markdown .md](mini_demo.md)" in twin
    assert "```python\nimport quadriga_lib\nprint(quadriga_lib.version())\n```\n\n```text\n9.9.9\n```" in twin
    assert "![A black square](mini_demo/fig-4-1.svg)" in twin
    assert "the [FAQ](../faq.md#cite)" in twin
    assert (mini_repo / "website" / NB).read_bytes() == (mini_repo / RUN).read_bytes()  # the executed copy
    page = site(mini_repo, "tutorials/mini_demo.html")
    assert '<a class="dl" href="mini_demo.ipynb">Download notebook <small>.ipynb</small></a>' in page
    index = site(mini_repo, "tutorials.html")
    assert 'href="tutorials/mini_demo.html"' in index and ">Python</span>" in index and ">Beginner</span>" in index
    assert "<h3>Mini demo</h3>" in index and ">about 5 min</span>" in index
    md = site(mini_repo, "tutorials.md")
    assert ("1. [Mini demo](tutorials/mini_demo.md) · Python · beginner · about 5 min · A *small* notebook for the "
            "tests.") in md


SERIES = "website_builder/content/tutorials/demo.md"


def test_series(mini_repo, run_site):
    code, out = run_site(mini_repo)
    assert code == 0, out
    assert "series-empty" not in out
    index = site(mini_repo, "tutorials.html")
    assert '<section class="tut-series" id="demo"><h2>Demo series</h2>' in index
    assert "What the <em>demo</em> series covers." in index
    assert '<h3 id="before-you-start">Before you start</h3>' in index  # series headings one level down
    assert '<span class="tut-part">Part 1</span><h3>Mini demo</h3>' in index
    assert '<summary><span>Demo series</span><span class="count">1</span></summary>' in index  # sidebar, closed
    assert '<li class="sub"><a href="tutorials/mini_demo.html">Mini demo</a></li>' in index
    md = site(mini_repo, "tutorials.md")
    assert '<a id="demo"></a>\n## Demo series\n\nWhat the *demo* series covers.' in md
    assert '<a id="before-you-start"></a>\n### Before you start' in md
    page = site(mini_repo, "tutorials/mini_demo.html")
    assert '<a href="../tutorials.html#demo">Demo series</a></p>' in page  # breadcrumb
    assert '<span class="tut-part">Part 1 of 1</span>' in page
    assert '<a class="current" href="mini_demo.html" aria-current="page">Mini demo</a>' in page
    assert "tut-pager" not in page  # a series of one has no previous or next
    assert "\n\nPart 1 of 1 of [Demo series](../tutorials.md#demo)\n" in site(mini_repo, "tutorials/mini_demo.md")


def test_series_order_and_pager(mini_repo, run_site):
    second(mini_repo, order=0, title="Zero")
    code, out = run_site(mini_repo)
    assert code == 0, out
    index = site(mini_repo, "tutorials.html")
    assert index.index("<h3>Zero</h3>") < index.index("<h3>Mini demo</h3>")
    page = site(mini_repo, "tutorials/mini_demo.html")
    assert '<span class="tut-part">Part 2 of 2</span>' in page
    assert '<a class="prev" href="mini_zero.html"><small>← Previous</small><span>Zero</span></a>' in page
    assert 'class="next"' not in page
    zero = site(mini_repo, "tutorials/mini_zero.html")
    assert '<a class="next" href="mini_demo.html"><small>Next →</small><span>Mini demo</span></a>' in zero
    assert 'class="prev"' not in zero and '<ol class="tut-parts numbered">' in zero
    assert ("Part 2 of 2 of [Demo series](../tutorials.md#demo) · Previous: [Zero](mini_zero.md)\n"
            in site(mini_repo, "tutorials/mini_demo.md"))
    assert ("Part 1 of 2 of [Demo series](../tutorials.md#demo) · Next: [Mini demo](mini_demo.md)\n"
            in site(mini_repo, "tutorials/mini_zero.md"))


def test_series_order_and_repeated_headings(mini_repo, run_site):
    content = mini_repo / "website_builder/content/tutorials"
    (content / "early.md").write_text('+++\ntitle = "Early"\norder = 0\n+++\n\n## Before you start\n\nText.\n',
                                      encoding="utf-8")
    second(mini_repo, series="early")
    code, out = run_site(mini_repo)
    assert code == 0, out
    index = site(mini_repo, "tutorials.html")
    assert index.index('id="early"') < index.index('id="demo"')  # series by order
    assert '<h3 id="before-you-start">' in index and '<h3 id="before-you-start-2">' in index  # one set of ids


def test_unknown_series(mini_repo, run_site):
    both(mini_repo, lambda m: m.update(series="nope"))
    code, out = run_site(mini_repo)
    assert code == 1
    assert ("tutorials/mini_demo.ipynb: error: metadata.quadriga_tutorial.series: no series file "
            "website_builder/content/tutorials/nope.md") in out
    shutil.rmtree(mini_repo / "tutorials/executed")  # checked in the committed notebook, run or not
    code, out = run_site(mini_repo)
    assert code == 1 and "tutorials/mini_demo.ipynb: error: metadata.quadriga_tutorial.series: no series" in out


def test_executed_copy_with_other_metadata(mini_repo, run_site):
    edit(mini_repo, lambda nb: nb["metadata"]["quadriga_tutorial"].pop("series"))  # run before the key was set
    code, out = run_site(mini_repo)
    assert code == 0, out
    assert ("tutorials/mini_demo.ipynb: warning: the executed copy tutorials/executed/mini_demo.ipynb has other "
            "tutorial metadata than the notebook (make tutorials) [not-executed]") in out
    assert "series-empty" not in out  # the notebook names the series
    code, out = run_site(mini_repo, "--release")
    assert code == 1 and "tutorials/mini_demo.ipynb: error: the executed copy" in out


def test_tutorials_without_series(mini_repo, run_site):
    both(mini_repo, lambda m: m.pop("series"))
    second(mini_repo, title="Two", order=2)
    code, out = run_site(mini_repo)
    assert code == 0, out
    assert ("%s: warning: no notebook names this series (metadata.quadriga_tutorial.series) [series-empty]"
            % SERIES) in out
    index = site(mini_repo, "tutorials.html")
    assert '<section class="tut-series" id="demo"><h2>Demo series</h2>' in index  # kept, without tutorials
    assert '<section class="tut-series" id="other"><h2>Other tutorials</h2>' in index
    assert "<h3>Mini demo</h3>" in index and 'class="tut-part"' not in index
    page = site(mini_repo, "tutorials/mini_demo.html")
    assert 'class="tut-part"' not in page and "tut-pager" not in page  # no parts, no previous and next
    assert '<ol class="tut-parts">' in page and 'href="mini_zero.html">Two</a>' in page
    md = site(mini_repo, "tutorials.md")
    assert "## Other tutorials\n\n- [Mini demo](tutorials/mini_demo.md) · " in md
    assert "\n- [Two](tutorials/mini_zero.md) · " in md
    assert "Part " not in site(mini_repo, "tutorials/mini_demo.md")


def test_links_in_titles_and_headings(mini_repo, run_site):
    path = mini_repo / SERIES
    path.write_text(path.read_text(encoding="utf-8").replace('"Demo series"', '"Demo [FAQ](faq.html)"'),
                    encoding="utf-8")
    edit(mini_repo, lambda nb: nb["cells"].append(
        {"cell_type": "markdown", "metadata": {}, "source": ["## About [the FAQ](faq.html)"]}))
    code, out = run_site(mini_repo)
    assert code == 0, out
    page = site(mini_repo, "tutorials/mini_demo.html")
    assert '<a href="../tutorials.html#demo">Demo FAQ</a></p>' in page  # inside links as plain text
    assert "Part 1 of 1 of [Demo FAQ](../tutorials.md#demo)" in site(mini_repo, "tutorials/mini_demo.md")
    assert '<li class="sub"><a href="#about-the-faq">About the FAQ</a></li>' in page


def test_series_id_clash(mini_repo, run_site):
    (mini_repo / SERIES).rename(mini_repo / "website_builder/content/tutorials/content.md")
    both(mini_repo, lambda m: m.update(series="content"))
    code, out = run_site(mini_repo)
    assert code == 1
    assert ("website_builder/content/tutorials/content.md: error: E-anchor: the series id 'content' is already "
            "an id on tutorials.html") in out


def test_series_id_clash_with_index_heading(mini_repo, run_site):
    with open(mini_repo / "website_builder/content/tutorials.md", "a", encoding="utf-8") as f:
        f.write("\n## Demo\n\nText.\n\n## Other\n\nText.\n")
    code, out = run_site(mini_repo)
    assert code == 1
    assert ("%s: error: E-anchor: the series id 'demo' is already an id on tutorials.html" % SERIES) in out
    assert re.search(r"website_builder/content/tutorials\.md:11: error: E-anchor: the id 'other' is the section "
                     r"of the tutorials without a series", out), out


def test_no_tutorials(mini_repo, run_site):
    (mini_repo / NB).unlink()
    code, out = run_site(mini_repo)
    assert code == 0 and "not-executed" not in out
    assert "tut-row" not in site(mini_repo, "tutorials.html")
    assert "tutorials/mini_demo" not in site(mini_repo, "tutorials.md")
    assert not (mini_repo / "website/tutorials/mini_demo.html").exists()


def test_not_executed(mini_repo, run_site):
    shutil.rmtree(mini_repo / "tutorials/executed")
    code, out = run_site(mini_repo)
    assert code == 0, out
    assert ("tutorials/executed: warning: missing: the tutorials have not been run (make tutorials); the tutorials "
            "page lists none [not-executed]") in out
    index = site(mini_repo, "tutorials.html")
    assert "<h1>Tutorials</h1>" in index and "tut-row" not in index
    assert '<section class="tut-series" id="demo"><h2>Demo series</h2>' in index  # the series stays
    assert "series-empty" not in out
    assert not (mini_repo / "website/tutorials/mini_demo.html").exists()
    code, out = run_site(mini_repo, "--release")
    assert code == 1 and "tutorials/executed: error: missing: the tutorials have not been run" in out


def test_executed_copy_missing(mini_repo, run_site):
    (mini_repo / RUN).unlink()
    code, out = run_site(mini_repo)
    assert code == 0, out
    assert ("tutorials/mini_demo.ipynb: warning: no executed copy tutorials/executed/mini_demo.ipynb "
            "(make tutorials); left out [not-executed]") in out
    assert "tut-row" not in site(mini_repo, "tutorials.html")
    code, out = run_site(mini_repo, "--release")
    assert code == 1 and "tutorials/mini_demo.ipynb: error: no executed copy" in out


def test_executed_copy_without_notebook_is_ignored(mini_repo, run_site):
    (mini_repo / NB).unlink()  # a deleted tutorial whose executed copy is still there
    code, out = run_site(mini_repo)
    assert code == 0 and "mini_demo" not in out
    assert not (mini_repo / "website/tutorials/mini_demo.html").exists()


def link_to_tutorials(root, *stems):
    """Append a link to each tutorials/<stem>.html to a content page that becomes faq.html and faq.md."""
    with open(root / "docs_builder/content/faq.md", "a", encoding="utf-8") as f:
        f.write("\n" + " ".join("[%s](tutorials/%s.html)" % (s, s) for s in stems) + "\n")


def test_link_to_unrun_tutorial(mini_repo, run_site):
    link_to_tutorials(mini_repo, "mini_demo")
    code, out = run_site(mini_repo)
    assert code == 0 and "link target" not in out, out  # with its executed copy the link resolves
    for remove in (lambda: (mini_repo / RUN).unlink(), lambda: shutil.rmtree(mini_repo / "tutorials/executed")):
        remove()
        code, out = run_site(mini_repo)
        assert code == 0, out
        assert re.search(r"website/faq\.html:\d+: warning: link target left out: tutorials/mini_demo\.html "
                         r"\(tutorial not run, make tutorials\) \[not-executed\]", out)
        assert re.search(r"website/faq\.md:\d+: warning: link target left out: "
                         r"http://quadriga-lib\.org/tutorials/mini_demo\.md \(tutorial not run", out)
        assert "link target missing" not in out
        code, out = run_site(mini_repo, "--release")
        assert code == 1 and re.search(r"tutorials/(mini_demo\.ipynb|executed): error: ", out), out


def test_link_to_missing_tutorial(mini_repo, run_site):
    link_to_tutorials(mini_repo, "mini_dmeo")  # no notebook of that name: an error with or without executed copies
    for remove in (lambda: None, lambda: shutil.rmtree(mini_repo / "tutorials/executed")):
        remove()
        code, out = run_site(mini_repo)
        assert code == 1
        assert re.search(r"website/faq\.html:\d+: error: link target missing: tutorials/mini_dmeo\.html", out)
        assert re.search(r"website/faq\.md:\d+: error: link target missing: "
                         r"http://quadriga-lib\.org/tutorials/mini_dmeo\.md", out)
