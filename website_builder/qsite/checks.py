# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Checks of the built site in memory: links and anchors, HTML well-formedness, size budgets."""

import posixpath
from html.parser import HTMLParser

from qdocs import checks as md_checks
from qdocs import links

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

API_BUDGET, PAGE_BUDGET = 800_000, 150_000
LANDING_BUDGET, FIGURE_BUDGET = 480_000, 150_000
# The fonts the landing page uses: Sans 400, 600 and 700, Mono 400 for the quick start
LANDING_FONTS = ("fonts/ibm-plex-sans-latin-400-normal.woff2", "fonts/ibm-plex-sans-latin-600-normal.woff2",
                 "fonts/ibm-plex-sans-latin-700-normal.woff2", "fonts/ibm-plex-mono-latin-400-normal.woff2")


class _Page(HTMLParser):
    """One pass over a page: tag stack, ids, headings, main, img alt and every URL."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.ids, self.urls, self.problems = [], [], [], []
        self.h1 = self.main = 0

    def _tag(self, tag, attrs, closed):
        a = dict(attrs)
        line = self.getpos()[0]
        if "id" in a:
            self.ids.append((a["id"], line))
        if tag == "h1":
            self.h1 += 1
        if tag == "main" and a.get("id") == "content":
            self.main += 1
        if tag == "img" and "alt" not in a:
            self.problems.append((line, "<img> without alt"))
        for key in ("href", "src"):
            if a.get(key) is not None:
                self.urls.append((line, a[key]))
        if a.get("srcset"):
            self.urls += [(line, s.split()[0]) for s in a["srcset"].split(",") if s.strip()]
        if not closed and tag not in VOID:
            self.stack.append((tag, line))

    def handle_starttag(self, tag, attrs):
        self._tag(tag, attrs, False)

    def handle_startendtag(self, tag, attrs):
        self._tag(tag, attrs, True)

    def handle_endtag(self, tag):
        line = self.getpos()[0]
        if tag in VOID:
            return
        if not self.stack or self.stack[-1][0] != tag:
            open_tag = "<%s> of line %d" % self.stack[-1] if self.stack else "nothing"
            self.problems.append((line, "</%s> closes %s" % (tag, open_tag)))
            if any(t == tag for t, _ in self.stack):
                while self.stack and self.stack.pop()[0] != tag:
                    pass
            return
        self.stack.pop()


def _target(path, url):
    """The site path and fragment a relative or root-absolute URL points to."""
    file_part, _, frag = url.partition("#")
    if not file_part:
        return path, frag
    if file_part.startswith("/"):
        return posixpath.normpath(file_part.lstrip("/")), frag
    return posixpath.normpath(posixpath.join(posixpath.dirname(path), file_part)), frag


def check_site(mapping, diag, markdown, left_out=None):
    """All checks of the built site; markdown lists the generated Markdown files to check inside. left_out:
    {site path: (warning kind, note)} of files left out of this build on purpose; links to them are warnings."""
    pages = {}
    for path in sorted(p for p in mapping if p.endswith(".html")):
        parser = _Page()
        parser.feed(mapping[path].decode("utf-8"))
        parser.close()
        pages[path] = parser
        where = "website/" + path
        for line, msg in parser.problems:
            diag.error(where, line, msg)
        for tag, line in parser.stack:
            diag.error(where, line, "<%s> not closed" % tag)
        seen = set()
        for i, line in parser.ids:
            if i in seen:
                diag.error(where, line, "id '%s' appears twice" % i)
            seen.add(i)
        if parser.h1 != 1:
            diag.error(where, None, "%d <h1> elements, expected exactly one" % parser.h1)
        if parser.main != 1:
            diag.error(where, None, '%d <main id="content"> elements, expected exactly one' % parser.main)

    md_ids, generated = {}, set(markdown)
    for path, parser in pages.items():
        for line, url in parser.urls:
            if links.ABSOLUTE_URL_RE.match(url) or url.startswith("//"):
                continue  # external, mailto:, download/ and releases/ are not fetched
            target, frag = _target(path, url)
            where = ("website/" + path, line)
            if target in (left_out or {}):
                kind, note = left_out[target]
                diag.warning(*where, "link target left out: %s (%s)" % (url, note), kind)
            elif target.startswith("../") or target not in mapping:
                diag.error(*where, "link target missing: %s" % url)
            elif frag and target.endswith(".html") and frag not in {i for i, _ in pages[target].ids}:
                diag.error(*where, "anchor missing: %s (no id '%s' in %s)" % (url, frag, target))
            elif frag and target in generated and frag not in md_checks.collect_ids(mapping, target, md_ids):
                diag.error(*where, "anchor missing: %s (no <a id> '%s' in %s)" % (url, frag, target))
    md_checks.check_markdown(mapping, "website", diag, markdown, left_out)
    sizes(mapping, diag)


def sizes(mapping, diag):
    """Size budgets in raw bytes; each exceeded budget is a warning."""
    def warn(path, size, budget, what):
        diag.warning("website/" + path, None, "%s is %d kB, budget %d kB" % (what, size // 1000, budget // 1000), "size")

    for path in sorted(p for p in mapping if p.endswith(".html")):
        budget = API_BUDGET if path.endswith("_api.html") and "/" not in path else PAGE_BUDGET
        if len(mapping[path]) > budget:
            warn(path, len(mapping[path]), budget, "page")
    figures = sorted(p for p in mapping if p.startswith("img/landing/") and p.endswith(".webp"))
    for path in figures:
        if len(mapping[path]) > FIGURE_BUDGET:
            warn(path, len(mapping[path]), FIGURE_BUDGET, "landing figure")
    parts = ["index.html", "css/site.css", "js/site.js"] + list(LANDING_FONTS) + [p for p in figures if p.endswith("-light.webp")]
    total = sum(len(mapping.get(p, b"")) for p in parts)
    if total > LANDING_BUDGET:
        warn("index.html", total, LANDING_BUDGET, "landing page with styles, script, fonts and figures")
