# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Site constants, page list, source paths and the versions read once per run."""

import re

try:
    import tomllib
except ImportError:  # Python 3.10: the same API from the tomli package
    import tomli as tomllib

SITE_BASE_URL = "http://quadriga-lib.org"  # http until HTTPS works at the hoster
DOWNLOAD_BASE_URL = "http://quadriga-lib.org/download"
RELEASES_URL = "http://quadriga-lib.org/releases/"
GITHUB_URL = "https://github.com/stjaeckel/quadriga-lib"

# (id, tab label or None, twin path or None), in tab order; the file of a page is <id>.html
PAGES = (
    ("index", "Home", None),
    ("download", "Get started", "download.md"),
    ("cpp_api", "C++ API", "quadriga_lib_api_cpp.md"),
    ("mex_api", "MATLAB API", "quadriga_lib_api_mex.md"),
    ("python_api", "Python API", "quadriga_lib_api_python.md"),
    ("tutorials", "Tutorials", "tutorials.md"),
    ("formats", "Formats", "formats.md"),
    ("publications", "Publications", "publications.md"),
    ("ai_agents", "AI agents", "ai_agents.md"),
    ("faq", "FAQ", "faq.md"),
    ("impressum", None, None),
    ("privacy", None, None),
    ("404", None, None),
)
TWINS = {pid: twin for pid, _, twin in PAGES if twin}

# Twins written only into the website, never into docs/: the tutorial index and the tutorial pages
WEBSITE_ONLY = ("tutorials.md", "tutorials/")

# Prose pages of docs/ whose source is docs_builder/content/<id>.md; formats.md is the formats page's intro
TEXT_PAGES = ("download", "formats", "faq", "ai_agents")
FORMATS_DIR = "docs_builder/content/formats"  # one file per data format, a section of formats.html each

# Ids the website templates write themselves (main content, landing sections, quick-start tabs, API intro);
# heading ids avoid them, because an id may appear only once on a page
RESERVED_IDS = ("apis", "content", "covers", "open", "paths", "qs-cpp", "qs-matlab", "qs-python", "quickstart",
                "sec-general-usage-notes", "why")

LANGS = ("cpp", "mex", "python")
LANG_TITLE = {"cpp": "C++", "mex": "MATLAB / Octave", "python": "Python"}  # page titles, llms.txt
LANG_PILL = {"cpp": "C++", "mex": "MATLAB", "python": "Python"}  # pills and counterpart lines
SOURCE_DIRS = {"cpp": "src", "mex": "api_mex", "python": "api_python"}
FENCE_LANG = {"cpp": "cpp", "mex": "matlab", "python": "python"}

DOCS_CONTENT = "docs_builder/content"
WEBSITE_CONTENT = "website_builder/content"
HEADER_FILE = "include/quadriga_lib.hpp"
CMAKE_FILE = "CMakeLists.txt"
DOCS_DIR = "docs"

# Build variables; read_versions() fills in the versions
VARIABLES = {
    "version": None,
    "armadillo_version": None,
    "site_base_url": SITE_BASE_URL,
    "download_base_url": DOWNLOAD_BASE_URL,
    "releases_url": RELEASES_URL,
    "github_url": GITHUB_URL,
}


def read_versions(root, diag):
    """Read the library and Armadillo versions into VARIABLES; a missing file or match is an error."""
    for key, rel, pattern in (
        ("version", HEADER_FILE, r'#define\s+QUADRIGA_LIB_VERSION_STR\s+"([^"]+)"'),
        ("armadillo_version", CMAKE_FILE, r'set\(armadillo_version\s+"([^"]+)"\)'),
    ):
        VARIABLES[key] = None
        path = root / rel
        if not path.is_file():
            diag.error(rel, None, "file not found (needed for the %s)" % key.replace("_", " "))
            continue
        m = re.search(pattern, path.read_text(encoding="utf-8"))
        if not m:
            diag.error(rel, None, "no match for %s" % pattern)
            continue
        VARIABLES[key] = m.group(1)
    return VARIABLES["version"], VARIABLES["armadillo_version"]


def version():
    return VARIABLES["version"] or ""
