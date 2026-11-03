# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Prose pages (TOML front matter and Markdown body), heading ids, build variables,
publications data and the gap scan."""

import html
import re
from dataclasses import dataclass, field

from . import config, docblocks, links

VARIABLE_RE = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")
# A gap marker %%GAP: text%% (one line): missing content or a draft awaiting approval; group 1 is the text
GAP_RE = re.compile(r"%%GAP:(.*?)%%")
HEADING_RE = re.compile(r"^(#{1,6}) (.+)$")
_ATTRS_RE = re.compile(r'\s*\{\s*(#[^\s}"]+)?((?:\s*legacy="[^"]*")*)\s*\}\s*$')
_ESCAPE_RE = re.compile(r"(\\[!-/:-@\[-`{-~])")

PAGE_KEYS = {"title": (str, True), "description": (str, True), "lead": (str, False)}
EXTRA_KEYS = {"tutorials.md": "dummy_note", "ai_agents.md": "llms_summary"}  # allowed in one file each

_LINK = {"label": (str, True), "url": (str, True)}
_ENTRY = {"year": (int, True), "title": (str, True), "authors": (str, True), "venue": (str, True),
          "link": ([_LINK], True)}
_GROUP = {"id": (str, True), "title": (str, True), "intro": (str, False), "outro": (str, False),
          "entry": ([_ENTRY], True)}
PUBLICATION_KEYS = {"page": ({"title": (str, True), "description": (str, True), "lead": (str, True)}, True),
                    "group": ([_GROUP], True)}


@dataclass
class Heading:
    index: int          # line index in the list of lines passed to heading_ids()
    level: int          # number of '#' in the source
    text: str           # heading text without the '#'s and without a trailing {...}
    plain: str          # text without Markdown: links as their text, no '*' or backticks, escapes and entities resolved
    id: str = None      # set for headings that become h2 or h3
    legacy: list = field(default_factory=list)


@dataclass
class Page:
    path: str           # relative to the repository root
    page_id: str        # file stem
    title: str = ""
    description: str = ""
    lead: str = ""
    extra: dict = field(default_factory=dict)  # dummy_note, llms_summary
    body: list = field(default_factory=list)   # source lines after the front matter
    body_line: int = 1  # 1-based line of body[0]
    headings: list = field(default_factory=list)


def substitute(text, diag=None, path=None, first_line=1):
    """Replace build variables {{ name }}; an unknown name is a structural error."""
    def repl(m):
        name = m.group(1)
        if name in config.VARIABLES:
            return config.VARIABLES[name] or ""
        if diag:
            line = first_line + text.count("\n", 0, m.start())
            diag.error(path, line, "unknown build variable '{{ %s }}'" % name)
        return m.group(0)
    return VARIABLE_RE.sub(repl, text)


def strip_gaps(text):
    """Remove every gap marker with its text (for docs/, which leaves the markers out)."""
    return GAP_RE.sub("", text).strip() if "%%GAP:" in text else text


def check_keys(data, spec, prefix, report):
    """Plain key check: report(key_path_message) for missing, unknown, wrongly typed keys and empty lists."""
    for key in data:
        if key not in spec:
            report("%s%s: unknown key" % (prefix, key))
    for key, (kind, required) in spec.items():
        p = prefix + key
        if key not in data:
            if required:
                report(p + ": missing")
            continue
        v = data[key]
        if isinstance(kind, list):
            if not isinstance(v, list) or not all(isinstance(x, dict) for x in v):
                report(p + ": not a list of tables")
            elif not v:
                report(p + ": empty list")
            else:
                for n, x in enumerate(v, 1):
                    check_keys(x, kind[0], "%s[%d]." % (p, n), report)
        elif isinstance(kind, dict):
            if isinstance(v, dict):
                check_keys(v, kind, p + ".", report)
            else:
                report(p + ": not a table")
        elif kind is int and (not isinstance(v, int) or isinstance(v, bool)):
            report(p + ": not an integer")
        elif kind is str and not isinstance(v, str):
            report(p + ": not a string")


def _toml(text, diag, rel, first_line):
    try:
        return config.tomllib.loads(text)
    except config.tomllib.TOMLDecodeError as e:
        m = re.search(r"line (\d+)", str(e))
        diag.error(rel, first_line + int(m.group(1)) - 1 if m else None, "TOML: %s" % e)
        return None


def plain_text(text):
    """Plain text of a heading: links to their text, no '*' or backticks, escapes and entities resolved."""
    text = re.sub(r"\.?\[\[([^\[\]]+)\]\]", r"\1", text)
    text = re.sub(r"!?\[((?:\\.|[^\]\\])*)\]\([^)]*\)", r"\1", text)
    parts = _ESCAPE_RE.split(text)
    text = "".join(p[1] if n % 2 else p.replace("*", "").replace("`", "") for n, p in enumerate(parts))
    return html.unescape(text)


def heading_ids(lines, diag=None, path=None, first_line=1):
    """Headings of a list of Markdown lines (column 0, outside fences), in order.

    ## and ### headings get an id: the explicit {#id}, else the slug of the plain text, made
    unique with -2, -3. Explicit ids or legacy ids used twice are E-anchor.
    """
    states, _ = docblocks.fence_states(lines)
    heads = []
    for i, ln in enumerate(lines):
        m = None if states[i] else HEADING_RE.match(ln)
        if not m:
            continue
        text, explicit, legacy = m.group(2).rstrip(), None, []
        a = _ATTRS_RE.search(text)
        if a:
            explicit = a.group(1)[1:] if a.group(1) else None
            legacy = re.findall(r'legacy="([^"]*)"', a.group(2))
            text = text[:a.start()].rstrip()
        heads.append(Heading(i, len(m.group(1)), text, plain_text(text), explicit, legacy))
    with_id = [h for h in heads if h.level in (2, 3)]
    used = set()
    for h in with_id:
        for anchor in ([h.id] if h.id else []) + h.legacy:
            if anchor in used and diag:
                diag.error(path, first_line + h.index, "E-anchor: id '%s' appears twice on the page" % anchor)
            used.add(anchor)
    for h in heads:
        if h not in with_id:
            h.id = None
        elif h.id is None:
            base = links.slug(h.plain) or "section"
            h.id, n = base, 2
            while h.id in used:
                h.id, n = "%s-%d" % (base, n), n + 1
            used.add(h.id)
    return heads


def read_page(path, diag, root):
    """Read a prose page: front matter between '+++' lines, build variables substituted, heading ids."""
    rel = path.relative_to(root).as_posix()
    page = Page(rel, path.stem)
    lines = substitute(path.read_text(encoding="utf-8"), diag, rel).split("\n")
    end = lines.index("+++", 1) if lines[0] == "+++" and "+++" in lines[1:] else None
    if end is None:
        diag.error(rel, 1, "the file must start with TOML front matter between two lines '+++'")
        return page
    data = _toml("\n".join(lines[1:end]), diag, rel, 2)
    if data is None:
        return page
    spec = dict(PAGE_KEYS)
    if path.name in EXTRA_KEYS:
        spec[EXTRA_KEYS[path.name]] = (str, False)
    check_keys(data, spec, "", lambda msg: diag.error(rel, None, "front matter: " + msg))
    for key in ("title", "description", "lead"):
        if isinstance(data.get(key), str):
            setattr(page, key, data[key])
    page.extra = {k: v for k, v in data.items() if k not in PAGE_KEYS and k in spec and isinstance(v, str)}
    body = lines[end + 1:]
    while body and not body[-1].strip():
        body.pop()
    page.body, page.body_line = body, end + 2
    docblocks.block_errors(body, rel, page.body_line, diag)
    page.headings = heading_ids(body, diag, rel, page.body_line)
    return page


def read_publications(path, diag, root):
    """Read and check publications.toml; returns its data (dict) or None after an error."""
    rel = path.relative_to(root).as_posix()
    data = _toml(substitute(path.read_text(encoding="utf-8"), diag, rel), diag, rel, 1)
    if data is None:
        return None
    errors = []
    check_keys(data, PUBLICATION_KEYS, "", errors.append)
    for msg in errors:
        diag.error(rel, None, msg)
    if errors:
        return None
    seen = set()
    for g in data["group"]:
        if g["id"] in seen:
            diag.error(rel, None, "E-anchor: group id '%s' appears twice on the page" % g["id"])
        seen.add(g["id"])
    return data


def gap_scan(root, folders, diag):
    """All gap markers in the files below folders (relative to root): sorted [(path, line, text)].
    A marker opened and not closed on its line is E-gap."""
    gaps = []
    for folder in folders:
        base = root / folder
        files = sorted((p for p in base.rglob("*") if p.is_file()), key=lambda p: p.as_posix()) if base.is_dir() else []
        for p in files:
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue  # images and other binary files hold no markers
            rel = p.relative_to(root).as_posix()
            for n, ln in enumerate(text.split("\n"), 1):
                if "%%GAP:" not in ln:
                    continue
                for m in GAP_RE.finditer(ln):
                    gaps.append((rel, n, m.group(1).strip()))
                if "%%GAP:" in GAP_RE.sub("", ln):
                    diag.error(rel, n, "E-gap: gap marker %%GAP: ... not closed with %% on the same line")
    return sorted(gaps, key=lambda g: (g[0], g[1]))
