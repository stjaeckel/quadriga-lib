# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Link and anchor check of the generated Markdown."""

from qdocs import checks, cli


def check(files, scope="docs"):
    diag = cli.Diagnostics()
    checks.check_markdown({k: v.encode("utf-8") for k, v in files.items()}, scope, diag)
    return [ln.split(": error: ", 1)[0] + ": " + ln.split(": error: ", 1)[1] for ln in diag.lines]


def test_missing_file_and_anchor():
    files = {"a.md": "# A\n\n[b](b.md) [c](c.md) [b2](b.md#here) [b3](b.md#gone) [self](#top)\n",
             "b.md": '<a id="here"></a>\n## B\n', "api/x.md": "[up](../a.md#top) [ok](../b.md#here)\n"}
    assert check(files) == ["docs/a.md:3: link target missing: c.md",
                            "docs/a.md:3: anchor missing: b.md#gone (no id 'gone' in b.md)",
                            "docs/a.md:3: anchor missing: #top (no id 'top' in a.md)",
                            "docs/api/x.md:1: anchor missing: ../a.md#top (no id 'top' in a.md)"]


def test_code_is_not_checked():
    assert check({"a.md": "`[x](nowhere.md)`\n\n```\n[y](nowhere.md)\n```\n"}) == []


def test_links_outside_docs_skipped_in_docs_phase():
    files = {"a.md": "[pdf](http://quadriga-lib.org/publications/x.pdf) [up](../outside.md) "
                     "[ext](https://example.org/x.md) [zip](http://quadriga-lib.org/download/x.zip)\n"}
    assert check(files) == []
    # the website phase checks absolute site links and links that leave the folder
    assert check(files, "website") == ["website/a.md:1: link target missing: http://quadriga-lib.org/publications/x.pdf",
                                       "website/a.md:1: link leaves the site: ../outside.md"]
    site = dict(files, **{"publications/x.pdf": "%PDF", "index.html": '<p id="quickstart">'})
    site["b.md"] = "[q](http://quadriga-lib.org/index.html#quickstart) [r](http://quadriga-lib.org/index.html#nope)\n"
    assert check(site, "website")[1:] == ["website/b.md:1: anchor missing: http://quadriga-lib.org/index.html#nope "
                                          "(no id 'nope' in index.html)"]


def test_hand_made_targets_checked_for_existence_only():
    site = {"a.md": "[m](quadriga_lib_material_model.md#any-heading) [p](publications/gone.md#x)\n",
            "quadriga_lib_material_model.md": "# Material model\n"}
    diag = cli.Diagnostics()
    checks.check_markdown({k: v.encode("utf-8") for k, v in site.items()}, "website", diag, paths=["a.md"])
    assert [ln.split(": error: ")[1] for ln in diag.lines] == ["link target missing: publications/gone.md#x"]


def test_faq_without_cite(mini_repo, run_docs):
    faq = mini_repo / "docs_builder/content/faq.md"
    faq.write_text(faq.read_text(encoding="utf-8").replace(" {#cite}", ""), encoding="utf-8")
    code, out = run_docs(mini_repo)
    assert code == 1
    assert "docs/formats.md:" in out and ": error: anchor missing: faq.md#cite (no id 'cite' in faq.md)" in out
    assert "docs/publications.md:" in out
    assert not (mini_repo / "docs").exists()
