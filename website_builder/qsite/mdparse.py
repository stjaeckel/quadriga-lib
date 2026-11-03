# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""markdown-it set-up for the three kinds of source (docblock, prose, notebook) and the builder's rules.

The rules read from the parse environment `env`:
  location  output path of the page (relative to the site root), for link URLs
  headings  {line index in the parsed text: qdocs Heading} (prose and notebook)
  kind      docblock, prose or notebook; notebook headings deeper than h4 become h4
  error     error(line index, message): reports a structural error at a source line
"""

import re

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

from qdocs import links

_RAW_A_RE = re.compile(r'^<a\s[^>]*?\bhref="([^"]*)"[^>]*>$', re.I)
_RAW_A_CLOSE_RE = re.compile(r"^</a\s*>$", re.I)
_BR_RE = re.compile(r"^<br\s*/?>$", re.I)
_ALERTS = {"[!NOTE]": "note", "[!WARNING]": "warn"}
MATH = ("math_inline", "math_inline_double")


def make_parser(kind, lang=None):
    """A MarkdownIt parser for one kind; lang adds the wikilink rule for the API page of that language."""
    md = MarkdownIt("commonmark", {"html": kind == "docblock"}).enable("table")
    md.use(dollarmath_plugin, allow_space=False, allow_digits=False, double_inline=True)
    if lang:
        md.inline.ruler.before("link", "wikilink", _wikilink_rule(lang))
    if kind == "docblock":
        md.core.ruler.push("raw_a", _raw_a)
        md.core.ruler.push("raw_br", _raw_br)
    else:
        md.core.ruler.before("inline", "heading_attrs", _heading_attrs)
        md.core.ruler.push("alert", _alert)
    md.core.ruler.push("site_url", _site_url)
    return md


def _wikilink_rule(lang):
    """[[name]] and .[[name]] outside code: a link to the function on the API page of lang."""
    def rule(state, silent):
        src, pos = state.src, state.pos
        if not src.startswith("[[", pos):
            return False
        end = src.find("]]", pos + 2)
        name = src[pos + 2:end] if end > 0 else ""
        if not name or "[" in name or "]" in name:
            return False
        member = state.pending.endswith(".")  # the text rule has already taken the '.'
        anchor = links.resolve(lang, name, member)
        if anchor is None:
            return False  # the docs phase reports E-link
        if not silent:
            if member:
                state.pending = state.pending[:-1]
            state.push("link_open", "a", 1).attrs = {"href": "%s_api.html#%s" % (lang, anchor)}
            state.push("text", "", 0).content = ("." if member else "") + name
            state.push("link_close", "a", -1)
        state.pos = end + 2
        return True
    return rule


def _inlines(state):
    return [t for t in state.tokens if t.type == "inline" and t.children]


def _raw_a(state):
    """<a href="U" ...>text</a> becomes a Markdown link; target and rel are dropped."""
    for tok in _inlines(state):
        depth = 0
        for c in tok.children:
            if c.type != "html_inline":
                continue
            m = _RAW_A_RE.match(c.content)
            if m:
                c.type, c.tag, c.nesting, c.content, c.attrs = "link_open", "a", 1, "", {"href": m.group(1)}
                depth += 1
            elif depth and _RAW_A_CLOSE_RE.match(c.content):
                c.type, c.tag, c.nesting, c.content = "link_close", "a", -1, ""
                depth -= 1


def _raw_br(state):
    """Trailing <br> at the end of a paragraph or heading is dropped (the old pages used it as spacing)."""
    for tok in _inlines(state):
        ch = tok.children
        k = len(ch)
        while k and (ch[k - 1].type == "softbreak" or (ch[k - 1].type == "text" and not ch[k - 1].content.strip())
                     or (ch[k - 1].type == "html_inline" and _BR_RE.match(ch[k - 1].content))):
            k -= 1
        if any(c.type == "html_inline" for c in ch[k:]):
            tok.children = ch[:k]


def _heading_attrs(state):
    """Heading ids from the content model by source line; the trailing {...} is removed from the text."""
    env, toks = state.env, state.tokens
    for i, tok in enumerate(toks):
        if tok.type != "heading_open":
            continue
        level = int(tok.tag[1:])
        level = min(level, 4) if env["kind"] == "notebook" else level
        tok.tag = toks[i + 2].tag = "h%d" % level
        h = env["headings"].get(tok.map[0]) if tok.map else None
        if h is None:
            if level in (2, 3):
                env["error"](tok.map[0] if tok.map else 0, "E-heading: heading not written as '## text' in "
                             "column 0 (setext or indented headings get no id)")
            continue
        toks[i + 1].content = h.text
        if h.id:
            tok.attrs["id"] = h.id
        tok.meta["legacy"] = list(h.legacy)


def _alert(state):
    """A blockquote whose first line is [!NOTE] or [!WARNING] becomes a note box; a first body
    line that is entirely bold is its title."""
    toks = state.tokens
    for i, tok in enumerate(toks):
        if tok.type != "blockquote_open" or i + 2 >= len(toks) or toks[i + 1].type != "paragraph_open":
            continue
        inl = toks[i + 2]
        kind = _ALERTS.get(inl.content.split("\n")[0])
        if not kind:
            continue
        ch = [c for c in inl.children if not (c.type == "text" and not c.content)]
        ch = ch[1:]
        if ch and ch[0].type == "softbreak":
            ch = ch[1:]
        title = None
        if ch and ch[0].type == "strong_open":
            k = next(n for n, c in enumerate(ch) if c.type == "strong_close")
            if k + 1 == len(ch) or ch[k + 1].type in ("softbreak", "hardbreak"):
                title, ch = ch[1:k], ch[k + 2:]
        inl.children = ch
        if not ch:
            toks[i + 1].hidden = toks[i + 3].hidden = True
        depth = 0
        for close in toks[i:]:
            depth += close.nesting if close.type.startswith("blockquote") else 0
            if depth == 0:
                break
        tok.meta = close.meta = {"alert": kind, "title": title}


def _site_url(state):
    """Link and image URLs for the output location; absolute quadriga-lib.org URLs become site-relative.
    Formula tokens get their line, for the diagnostics of the formula step."""
    loc = state.env["location"]
    for tok in _inlines(state):
        for c in tok.children:
            if c.type == "link_open":
                c.attrs["href"] = links.url(c.attrs["href"], loc)
            elif c.type == "image":
                c.attrs["src"] = links.url(c.attrs["src"], loc)
            elif c.type in MATH:
                c.meta["line"] = tok.map[0] if tok.map else 0
