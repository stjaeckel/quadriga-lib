# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""The real content files: they parse, keep the anchors that links use, and the publications pass their check."""

from pathlib import Path

import pytest

from qdocs import cli, prose

REPO = Path(__file__).resolve().parents[3]
CONTENT = REPO / "docs_builder" / "content"

# Ids that links point to: the doc blocks link formats.html#qdant, the footer faq.html#cite
PROSE_IDS = {"faq": ["cite", "about-the-library"], "download": ["download", "quick-start"]}

pytestmark = pytest.mark.repo


@pytest.fixture(scope="module")
def model():
    diag = cli.Diagnostics()
    model, mapping = cli.build_docs(REPO, diag)
    assert diag.errors == 0, "\n".join(diag.lines)
    assert mapping
    return model


def ids(page):
    return [h.id for h in page.headings if h.id]


def test_prose_pages_parse(model):
    for pid in ("download", "formats", "faq", "ai_agents"):
        page = model["pages"][pid]
        assert page.title and page.description
        assert page.body or pid == "formats"


def test_format_files(model):
    formats = model["formats"]
    assert [f.page_id for f in formats][:2] == ["qdant", "materials"]
    for f in formats:
        assert f.title and f.summary and f.body
    every = [i for f in formats for i in [f.page_id] + ids(f)]
    assert len(every) == len(set(every))  # one formats page: no id twice


def test_ids_that_links_use(model):
    for pid, wanted in PROSE_IDS.items():
        assert [a for a in wanted if a not in ids(model["pages"][pid])] == []
    for api in model["api"].values():
        for sec in api.sections:
            assert sec.slug and not hasattr(sec, "legacy")


def test_publications_key_check():
    diag = cli.Diagnostics()
    data = prose.read_publications(CONTENT / "publications.toml", diag, REPO)
    assert diag.errors == 0, "\n".join(diag.lines)
    assert [g["id"] for g in data["group"]] == ["quadriga-lib", "quadriga-background"]
    for g in data["group"]:
        years = [e["year"] for e in g["entry"]]
        assert years == sorted(years, reverse=True)  # newest first
