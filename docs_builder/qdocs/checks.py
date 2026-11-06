# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Link and anchor check of the generated Markdown files."""

import posixpath
import re

from . import links, render_md

# The explicit anchors that the twins carry, one per id
ANCHOR_RE = re.compile(r'<a id="([^"]*)"></a>')
_HTML_ID_RE = re.compile(r'\sid="([^"]*)"')


def collect_ids(mapping, path, cache):
    """The ids of the file mapping[path]: <a id> anchors in a .md file, id attributes otherwise.
    cache is a dict kept by the caller, so each file is read once."""
    if path not in cache:
        text = mapping[path].decode("utf-8", "replace")
        cache[path] = set((ANCHOR_RE if path.endswith(".md") else _HTML_ID_RE).findall(text))
    return cache[path]


def check_markdown(mapping, scope, diag, paths=None, left_out=None):
    """Check every link and image URL of the generated Markdown files; each problem is an error.

    mapping: {path: bytes}; scope "docs": mapping is docs/ (paths relative to docs/), and only links
    to files in it are checked; scope "website": mapping is the whole site, absolute links to
    SITE_BASE_URL are checked too, and fragments on HTML targets must be ids there.
    paths: the Markdown files to check (default: every .md and .txt file of mapping).
    A fragment on a generated .md target (one of paths) must be an <a id> there; hand-made
    files (publications/**, quadriga_lib_material_model.md) are checked for existence only.
    left_out: {path: (warning kind, note)} of files left out of this build on purpose; a link to one of
    them is a warning instead of an error.
    """
    left_out = left_out or {}
    prefix = "docs/" if scope == "docs" else "website/"
    cache = {}
    if paths is None:
        paths = [p for p in mapping if p.endswith((".md", ".txt"))]
    paths = set(paths)
    for path in sorted(paths):
        lines = mapping[path].decode("utf-8").split("\n")
        for i, url in render_md.scan_links(lines):
            where = (prefix + path, i + 1)
            site = links.site_path(url)
            if url.startswith("#"):
                target = path + url
            elif links.is_external(url):
                continue  # external, download/ and releases/ are not fetched
            elif site != url:  # an absolute link into the site
                if scope == "docs":
                    continue
                target = site
            else:
                target = posixpath.normpath(posixpath.join(posixpath.dirname(path), url))
            file_part, _, frag = target.partition("#")
            if file_part.startswith("../"):
                if scope == "docs":
                    continue  # outside docs/: checked by the website build
                diag.error(*where, "link leaves the site: %s" % url)
            elif file_part in left_out:
                kind, note = left_out[file_part]
                diag.warning(*where, "link target left out: %s (%s)" % (url, note), kind)
            elif file_part not in mapping:
                diag.error(*where, "link target missing: %s" % url)
            elif frag and (file_part.endswith(".html") or file_part in paths) and frag not in collect_ids(mapping, file_part, cache):
                diag.error(*where, "anchor missing: %s (no id '%s' in %s)" % (url, frag, file_part))
