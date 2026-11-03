# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""The real content files: they parse, keep the anchors that old links use, and the publications pass their check."""

from pathlib import Path

import pytest

from qdocs import cli, prose

REPO = Path(__file__).resolve().parents[3]
CONTENT = REPO / "docs_builder" / "content"

# Old hash ids of the prose sections and the two QDANT entry ids; old links into these pages use them
PROSE_LEGACY = {
    "formats": ["6cab4884", "9f6922f9", "<arrayant>", "<layout>"],
    "faq": ["e63d3e38", "59d38498", "89373af1", "d8094d72"],
    "download": ["7c9ba8cc", "7fc9f392", "904e3979", "fdc96a45"],
}

# Old hash ids of the API sections, the same on every page that has the section
API_LEGACY = {
    "Array antenna class": "3a0c780f", "Array antenna functions": "e091e809", "Channel class": "4f5f509b",
    "Channel functions": "eab50272", "Channel generation functions": "6f9553d7", "Channel statistics": "c2d4d113",
    "Geometry-based stochastic model": "85ec7f8f", "Math functions": "40522dfe", "Miscellaneous / Tools": "b11e2d16",
    "Site-specific simulation tools": "386370db", "Sum-of-sinusoids random number generator": "2d15ed9b",
}

pytestmark = pytest.mark.repo


@pytest.fixture(scope="module")
def model():
    diag = cli.Diagnostics()
    model, mapping = cli.build_docs(REPO, diag)
    assert diag.errors == 0, "\n".join(diag.lines)
    assert mapping
    return model


def ids(page):
    return [a for h in page.headings for a in ([h.id] if h.id else []) + h.legacy]


def test_prose_pages_parse(model):
    for pid in ("download", "formats", "faq", "ai_agents"):
        page = model["pages"][pid]
        assert page.title and page.description and page.body


def test_cite_and_prose_legacy_ids(model):
    assert "cite" in ids(model["pages"]["faq"])
    for pid, legacy in PROSE_LEGACY.items():
        have = ids(model["pages"][pid])
        assert [a for a in legacy if a not in have] == []


def test_api_section_legacy_ids(model):
    for lang, api in model["api"].items():
        for sec in api.sections:
            assert sec.legacy == API_LEGACY[sec.title], (lang, sec.title)


def test_publications_key_check():
    diag = cli.Diagnostics()
    data = prose.read_publications(CONTENT / "publications.toml", diag, REPO)
    assert diag.errors == 0, "\n".join(diag.lines)
    assert [g["id"] for g in data["group"]] == ["quadriga-lib", "quadriga-background"]
    for g in data["group"]:
        years = [e["year"] for e in g["entry"]]
        assert years == sorted(years, reverse=True)  # newest first
