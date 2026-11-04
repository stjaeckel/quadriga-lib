# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Ids, old hash ids, [[name]] resolution, cross-language matching and URLs per output location."""

import re

import pytest

from qdocs import cli, links

BASE = "http://quadriga-lib.org"


def build(root):
    diag = cli.Diagnostics()
    model, mapping = cli.build_docs(root, diag)
    return model, mapping, diag


def fn(model, lang, name):
    return next(f for s in model["api"][lang].sections for f in s.functions if f.name == name)


def test_slug():
    assert links.slug("Miscellaneous / Tools") == "miscellaneous-tools"
    assert links.slug("  How do I cite Quadriga-Lib? ") == "how-do-i-cite-quadriga-lib"
    assert links.slug("C++") == "c"



def test_wikilinks_resolve(mini_repo):
    model, mapping, diag = build(mini_repo)
    assert diag.errors == 0
    assert links.resolve("cpp", "interpolate", True) == ".interpolate"
    assert links.resolve("cpp", "interpolate", False) is None
    assert links.resolve("mex", "VERSION", False) == "version"  # MATLAB compares in lower case
    assert links.resolve("python", "version", False) is None
    assert b"[.interpolate](#.interpolate)" in mapping["quadriga_lib_api_cpp.md"]


def test_e_link(mini_repo, run_docs):
    f = mini_repo / "src" / "qd_tools.cpp"
    f.write_text(f.read_text(encoding="utf-8").replace("Second line of", "See [[nowhere]] and .[[quadriga_lib_version]]. Second line of"),
                 encoding="utf-8")
    code, out = run_docs(mini_repo)
    assert code == 1
    assert out.count("src/qd_tools.cpp:9: error: E-link:") == 2  # [[nowhere]] and the member form
    assert not (mini_repo / "docs").exists()


def test_e_anchor(mini_repo, run_docs):
    f = mini_repo / "src" / "qd_tools.cpp"
    f.write_text(f.read_text(encoding="utf-8").replace("# arrayant_interpolate_multi", "# quadriga_lib_version"),
                 encoding="utf-8")
    code, out = run_docs(mini_repo)
    assert code == 1 and re.search(r"src/qd_tools.cpp:\d+: error: E-anchor:", out)


def test_families_and_mapping_order(mini_repo):
    model, _, diag = build(mini_repo)
    assert diag.errors == 0
    assert fn(model, "mex", "arrayant_interpolate").counterparts == {"python": "interpolate", "cpp": ".interpolate"}
    assert fn(model, "python", "interpolate").counterparts == {"mex": "arrayant_interpolate", "cpp": ".interpolate"}
    assert fn(model, "cpp", "arrayant_interpolate_multi").counterparts == {"mex": "arrayant_interpolate",
                                                                            "python": "interpolate"}
    assert fn(model, "cpp", "quadriga_lib_version").counterparts == {}
    assert fn(model, "python", "components").counterparts == {}
    # the first listed C++ function is the pill target
    (mini_repo / "docs_builder/content/api_matching.toml").write_text(
        '[families]\narrayant_interpolate = ["arrayant_interpolate_multi", ".interpolate"]\n', encoding="utf-8")
    model, _, _ = build(mini_repo)
    assert fn(model, "mex", "arrayant_interpolate").counterparts["cpp"] == "arrayant_interpolate_multi"


def test_family_by_name(mini_repo):
    (mini_repo / "docs_builder/content/api_matching.toml").write_text("[families]\n", encoding="utf-8")
    f = mini_repo / "src" / "qd_tools.cpp"
    f.write_text(f.read_text(encoding="utf-8").replace("# quadriga_lib_version", "# Version"), encoding="utf-8")
    model, _, diag = build(mini_repo)
    assert fn(model, "mex", "version").counterparts == {"cpp": "Version"}  # lower case, '.' as '_'
    assert fn(model, "cpp", ".interpolate").counterparts == {}


@pytest.mark.parametrize("toml, message", [
    ('nosuch = ["quadriga_lib_version"]', "unknown stem 'nosuch'"),
    ('version = ["nosuch"]', "unknown C++ function 'nosuch' in family 'version'"),
    ('version = ["quadriga_lib_version"]\ncomponents = ["quadriga_lib_version"]',
     "C++ function 'quadriga_lib_version' is in two families"),
])
def test_mapping_errors(mini_repo, run_docs, toml, message):
    (mini_repo / "docs_builder/content/api_matching.toml").write_text("[families]\n" + toml + "\n", encoding="utf-8")
    code, out = run_docs(mini_repo)
    assert code == 1 and "docs_builder/content/api_matching.toml:2: error: " + message in out


@pytest.mark.parametrize("target, location, expected", [
    # HTML pages: same page, other page, folder depth, the 404 page (root-absolute)
    ("cpp_api.html#.interpolate", "cpp_api.html", "#.interpolate"),
    ("cpp_api.html#.interpolate", "mex_api.html", "cpp_api.html#.interpolate"),
    ("cpp_api.html#.interpolate", "tutorials/x.html", "../cpp_api.html#.interpolate"),
    ("faq.html#cite", "404.html", "/faq.html#cite"),
    # API targets in twins: full file, section file, prose twin
    ("cpp_api.html#.interpolate", "quadriga_lib_api_cpp.md", "#.interpolate"),
    ("cpp_api.html#.interpolate", "quadriga_lib_api_mex.md", "quadriga_lib_api_cpp.md#.interpolate"),
    ("cpp_api.html#.interpolate", "api/cpp/miscellaneous-tools.md", "array-antenna-class.md#.interpolate"),
    ("cpp_api.html#.interpolate", "api/mex/miscellaneous-tools.md", "../cpp/array-antenna-class.md#.interpolate"),
    ("cpp_api.html#.interpolate", "faq.md", "api/cpp/array-antenna-class.md#.interpolate"),
    ("cpp_api.html#sec-miscellaneous-tools", "faq.md", "api/cpp/miscellaneous-tools.md"),
    ("cpp_api.html", "faq.md", "quadriga_lib_api_cpp.md"),
    # page with twin, page without twin
    ("faq.html#cite", "api/cpp/x.md", "../../faq.md#cite"),
    ("index.html#quickstart", "faq.md", BASE + "/index.html#quickstart"),
    # the formats page: one file per format
    ("formats.html#qdant", "faq.md", "formats/qdant.md"),
    ("formats.html#notes-2", "faq.md", "formats/materials.md#notes-2"),
    ("formats.html#notes", "formats/qdant.md", "#notes"),
    # llms.txt: everything absolute
    ("faq.html#cite", "llms.txt", BASE + "/faq.md#cite"),
    # absolute site URLs become site-relative; download/ and releases/ stay absolute
    ("http://quadriga-lib.org/formats.html#qdant", "cpp_api.html", "formats.html#qdant"),
    ("https://www.quadriga-lib.org/formats.html", "faq.md", "formats.md"),
    ("http://quadriga-lib.org/download/x.zip", "faq.md", BASE + "/download/x.zip"),
    ("http://quadriga-lib.org/releases/", "faq.html", BASE + "/releases/"),
    # website-only twins: absolute from docs/, relative among themselves
    ("tutorials.html", "faq.md", BASE + "/tutorials.md"),
    ("tutorials.html", "tutorials/x.md", "../tutorials.md"),
    # hand-made site files: absolute from docs/, relative from a website-only twin; external unchanged
    ("publications/x.pdf", "faq.md", BASE + "/publications/x.pdf"),
    ("quadriga_lib_material_model.md", "tutorials/x.md", "../quadriga_lib_material_model.md"),
    ("https://example.org/a#b", "faq.md", "https://example.org/a#b"),
])
def test_url(mini_repo, target, location, expected):
    _, _, diag = build(mini_repo)
    assert diag.errors == 0
    assert links.url(target, location) == expected
