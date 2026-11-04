# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Tutorial notebooks: metadata checks, outputs, headings, pages and twins."""

import json

NB = "tutorials/mini_demo.ipynb"


def edit(root, change, name=NB):
    path = root / NB
    nb = json.loads(path.read_text(encoding="utf-8"))
    change(nb)
    path.unlink()
    (root / name).write_text(json.dumps(nb), encoding="utf-8")


def site(root, rel):
    return (root / "website" / rel).read_text(encoding="utf-8")


def test_metadata_errors(mini_repo, run_site):
    def bad(nb):
        meta = nb["metadata"]["quadriga_tutorial"]
        del meta["title"]
        meta["level"] = "expert"
        meta["order"] = "1"
        del nb["metadata"]["kernelspec"]["language"]
    edit(mini_repo, bad, "tutorials/Bad Name.ipynb")
    code, out = run_site(mini_repo)
    assert code == 1
    for msg in ("notebook file name must match", "metadata.quadriga_tutorial.title: missing",
                "metadata.quadriga_tutorial.order: not an integer", "metadata.quadriga_tutorial.level: one of",
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
    assert code == 1 and "cell 2: the notebook was committed with a failing cell (ValueError)" in out


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
    assert (mini_repo / "website" / NB).read_bytes() == (mini_repo / NB).read_bytes()
    page = site(mini_repo, "tutorials/mini_demo.html")
    assert '<a class="dl" href="mini_demo.ipynb">Download notebook <small>.ipynb</small></a>' in page
    index = site(mini_repo, "tutorials.html")
    assert 'href="tutorials/mini_demo.html"' in index and ">Python</span>" in index and ">Beginner</span>" in index
    assert "<h3>Mini demo</h3>" in index
    assert '<div class="note-box warn"><b>Placeholder</b>These cards are examples.</div>' in index
    md = site(mini_repo, "tutorials.md")
    assert "- [Mini demo](tutorials/mini_demo.md) · Python · beginner · A *small* notebook for the tests." in md


def test_no_tutorials(mini_repo, run_site):
    (mini_repo / NB).unlink()
    assert run_site(mini_repo)[0] == 0
    assert "No tutorials yet." in site(mini_repo, "tutorials.html") and "No tutorials yet." in site(mini_repo, "tutorials.md")
    assert "note-box" not in site(mini_repo, "tutorials.html")
