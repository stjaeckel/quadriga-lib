# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""HTML output: fragments from source lines, span-free code colouring, the pages from the templates,
the formula step (one pandoc call) and the gap pass."""

import html
import re
import subprocess

import pygments
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup
from pygments.lexers import get_lexer_by_name
from pygments.token import Comment, Generic, Keyword, Name, Number, String

from qdocs import config, links, prose, render_md
from . import assets, mdparse

# Span-free colouring: the first matching entry gives the class; other tokens are plain text
TOKEN_CLASSES = ((Comment, "c"), (Keyword.Type, "kt"), (Name.Class, "kt"), (Name.Namespace, "kt"),
                 (Keyword, "k"), (Name.Function, "nf"), (Name.Builtin, "nf"), (String, "s"),
                 (Number, "m"), (Generic.Prompt, "gp"))
LEXERS = {"cpp": "cpp", "c++": "cpp", "matlab": "matlab", "octave": "matlab", "python": "python",
          "py": "python", "bash": "bash", "sh": "bash", "shell": "bash", "console": "console"}
PLAIN = ("text", "none")
MATH_RE = re.compile("\x02M(\\d+)\x03")


def highlight(code, lang):
    """Code as HTML with one <span class> per run of coloured tokens and no span for the rest."""
    code = code.rstrip("\n")
    name = LEXERS.get(lang)
    if not name:
        return Markup(html.escape(code, quote=False))
    tokens = list(pygments.lex(code, get_lexer_by_name(name)))
    while tokens and tokens[-1][1].endswith("\n"):  # the lexer adds a final newline
        ttype, text = tokens.pop()
        if text.rstrip("\n"):
            tokens.append((ttype, text.rstrip("\n")))
    out, run, cls = [], [], None
    for ttype, text in tokens:
        c = next((k for t, k in TOKEN_CLASSES if ttype in t), None)
        if c != cls:
            if run:
                out.append(_span(cls, "".join(run)))
            run, cls = [], c
        run.append(text)
    if run:
        out.append(_span(cls, "".join(run)))
    return Markup("".join(out))


def _span(cls, text):
    text = html.escape(text, quote=False)
    return '<span class="%s">%s</span>' % (cls, text) if cls else text


class Context:
    """State of one build: diagnostics, the formulas found so far, the output location of the page."""

    def __init__(self, diag):
        self.diag = diag
        self.math = []      # (tex, token type, path, line, cell note) per formula, in order
        self.location = ""
        self._parsers = {}

    def parser(self, kind, lang):
        key = (kind, lang)
        if key not in self._parsers:
            md = mdparse.make_parser(kind, lang)
            for name, rule in (("fence", _fence), ("table_open", _table_open), ("table_close", _table_close),
                               ("blockquote_open", _quote_open),
                               ("blockquote_close", _quote_close), ("math_inline", _math),
                               ("math_inline_double", _math), ("math_block", _math),
                               ("math_block_label", _math_label)):
                md.add_render_rule(name, rule)
            self._parsers[key] = md
        return self._parsers[key]

    def fragment(self, lines, kind, path, line, lang=None, headings=None, tight=False, note="", shift=0):
        """Render source lines to HTML (Markup).

        kind: docblock, prose or notebook. path, line: source file and 1-based line of lines[0]
        (line None for notebook cells, with note naming the cell). lang: the API page language (wikilinks,
        fence default in doc blocks). headings: qdocs Headings of these lines. tight: a single paragraph
        is rendered without its <p>. shift: headings move down by this many levels.
        """
        def where(i):
            return (path, line + i) if line else (path, None)

        def error(i, msg):
            self.diag.error(*where(i), note + msg)

        def warn(i, msg, k):
            self.diag.warning(*where(i), note + msg, k)

        md = self.parser(kind, lang)
        env = {"location": self.location, "headings": {h.index: h for h in headings or []}, "shift": shift,
               "error": error, "warn": warn, "ctx": self, "kind": kind, "lang": lang, "where": where, "note": note}
        tokens = md.parse("\n".join(lines), env)
        if tight and [t.type for t in tokens] == ["paragraph_open", "inline", "paragraph_close"]:
            return Markup(md.renderer.render(tokens[1].children, md.options, env))
        return Markup(md.renderer.render(tokens, md.options, env))

    def inline(self, text, path, lang=None):
        """Inline Markdown of a front-matter or data string."""
        return self.fragment(text.split("\n"), "prose", path, None, lang, tight=True) if text else Markup("")


def _fence(self, tokens, idx, options, env):
    tok = tokens[idx]
    info = tok.info.strip().split()[0] if tok.info.strip() else ""
    if not info and env["kind"] == "docblock":
        info = config.FENCE_LANG[env["lang"]]
    if info and info not in LEXERS and info not in PLAIN:
        env["warn"](tok.map[0], "unknown fence language '%s'; no colouring" % info, "fence-language")
    cls = ' class="language-%s"' % html.escape(info) if info else ""
    return "<pre><code%s>%s</code></pre>\n" % (cls, highlight(tok.content, info))


def _table_open(self, tokens, idx, options, env):
    return '<div class="table-wrap"><table>\n'


def _table_close(self, tokens, idx, options, env):
    return "</table></div>\n"


def _quote_open(self, tokens, idx, options, env):
    meta = tokens[idx].meta
    if not meta.get("alert"):
        return self.renderToken(tokens, idx, options, env)
    title = self.renderInline(meta["title"], options, env) if meta["title"] else \
        ("Note" if meta["alert"] == "note" else "Warning")
    return '<div class="note-box%s"><b>%s</b>\n' % (" warn" if meta["alert"] == "warn" else "", title)


def _quote_close(self, tokens, idx, options, env):
    return "</div>\n" if tokens[idx].meta.get("alert") else self.renderToken(tokens, idx, options, env)


def _math(self, tokens, idx, options, env):
    """A placeholder; the formula step replaces it with MathML after one pandoc call for the whole build."""
    tok = tokens[idx]
    line = tok.map[0] if tok.map else tok.meta.get("line", 0)
    path, ln = env["where"](line)
    math = env["ctx"].math
    math.append((tok.content.strip(), tok.type, path, ln, env["note"]))
    return "\x02M%d\x03" % (len(math) - 1)


def _math_label(self, tokens, idx, options, env):
    env["error"](tokens[idx].map[0], "equation labels ($$...$$ (label)) are not supported")
    return ""


def formulas(ctx, pages):
    """Convert every collected formula with one pandoc call and put the MathML into the pages."""
    if not ctx.math:
        return pages
    doc = []
    for n, (tex, kind, *_) in enumerate(ctx.math):
        doc += ["QDOCSMATH%d" % n, "", ("$%s$" if kind == "math_inline" else "$$%s$$") % tex, ""]
    res = subprocess.run(["pandoc", "-d", str(assets.PANDOC_DEFAULTS), "-f", "markdown", "-t", "html"],
                         input="\n".join(doc), capture_output=True, text=True, encoding="utf-8")
    if res.returncode or res.stderr.strip():
        ctx.diag.error("pandoc", None, "formula conversion failed:\n" + res.stderr.rstrip())
    chunks = re.split(r"^<p>QDOCSMATH(\d+)</p>$", res.stdout, flags=re.M)
    found = {int(chunks[k]): chunks[k + 1].strip() for k in range(1, len(chunks) - 1, 2)}
    mathml = []
    for n, (tex, kind, path, line, note) in enumerate(ctx.math):
        m = re.fullmatch(r"<p>(<math\b.*</math>)</p>", found.get(n, ""), re.S)
        if not m or m.group(1).count("<math") != 1:
            ctx.diag.error(path, line, "%sformula not converted to MathML (unknown command or unbalanced "
                           "braces?): %s" % (note, tex))
            mathml.append("")
        else:
            mathml.append('<div class="math-block">%s</div>' % m.group(1) if kind == "math_block" else m.group(1))
    return {p: MATH_RE.sub(lambda m: mathml[int(m.group(1))], text) for p, text in pages.items()}


def gap_pass(text):
    """Gap markers: plain text inside tags and <title>, a <span class="ph"> in element text.
    Returns (text, number of markers replaced)."""
    count = [0]

    def plain(m):
        count[0] += 1
        return m.group(1).strip()

    text = re.sub(r"<[^>]*>", lambda t: prose.GAP_RE.sub(plain, t.group(0)), text)
    text = re.sub(r"<title>.*?</title>", lambda t: prose.GAP_RE.sub(plain, t.group(0)), text, flags=re.S)

    def span(m):
        count[0] += 1
        return '<span class="ph">%s</span>' % m.group(1).strip()

    return prose.GAP_RE.sub(span, text), count[0]


# ---------- Pages ----------

def environment():
    env = Environment(loader=FileSystemLoader(str(assets.TEMPLATES_DIR)), autoescape=True, trim_blocks=True,
                      lstrip_blocks=True, keep_trailing_newline=True, undefined=StrictUndefined)
    env.filters["highlight"] = highlight
    env.globals.update(config.VARIABLES)
    env.globals.update(tabs=[(pid, tab) for pid, tab, _ in config.PAGES if tab],
                       labels={pid: tab for pid, tab, _ in config.PAGES if tab}, lang_display=config.LANG_TITLE,
                       figure_names=assets.LANDING_FIGURES, figure_base=assets.figure)
    return env


def _index(ctx, page, least=2):
    """Text-page index (at least `least` h2 headings): [(text, id, [(text, id)])], texts as inline HTML."""
    rows = render_md.index_rows(page.headings)
    return [(ctx.inline(t, page.path), i, [(ctx.inline(st, page.path), si) for st, si in sub])
            for t, i, sub in rows] if len(rows) >= least else []


def formats_page(ctx, formats):
    """One section per format file: its title, id, Markdown twin, body (headings one level down) and
    the index entries of its ## headings."""
    return [{"id": f.page_id, "title": ctx.inline(f.title, f.path), "twin": "formats/%s.md" % f.page_id,
             "body": ctx.fragment(f.body, "prose", f.path, f.body_line, headings=f.headings, shift=1),
             "index": [(ctx.inline(t, f.path), i) for t, i, _ in render_md.index_rows(f.headings)]}
            for f in formats]


def _pills(f, page_lang, location):
    out = []
    for lang in config.LANGS:
        label = config.LANG_PILL[lang]
        if lang == page_lang:
            out.append((label, "#" + f.name, True))
        elif f.counterparts.get(lang):
            out.append((label, links.url("%s_api.html#%s" % (lang, f.counterparts[lang]), location), False))
        else:
            out.append((label, None, False))
    return out


def _function(ctx, f, sec):
    def frag(part, tight=False):
        return ctx.fragment(part.lines, "docblock", part.path, part.line, f.lang, tight=tight)

    def label(sub):  # labels may hold inline Markdown, such as code spans
        return ctx.inline(sub.label, sub.heading.path, f.lang)

    subs = []
    for sub in f.subblocks:
        if sub.kind != "params":
            subs.append({"label": label(sub), "parts": [{"rows": None, "html": frag(sub.content)}]})
            continue
        items = []
        for it in sub.items:
            if hasattr(it, "note"):  # a parameter row
                row = {"names": it.names.replace("`", ""), "qual": it.qual, "note": it.note,
                       "desc": frag(it.desc, tight=True)}
                if items and items[-1]["rows"] is not None:
                    items[-1]["rows"].append(row)
                else:
                    items.append({"rows": [row], "html": None})
            else:
                items.append({"rows": None, "html": frag(it)})
        subs.append({"label": label(sub), "parts": items})
    return {"name": f.name, "pills": _pills(f, f.lang, ctx.location), "md": "api/%s/%s.md" % (f.lang, sec.slug),
            "brief": frag(f.brief, tight=True), "intro": frag(f.intro), "subs": subs}


def api_page(ctx, page):
    lang, intro = page.lang, page.intro
    sections = []
    for sec in page.sections:
        desc = Markup("").join(ctx.fragment(d.lines, "docblock", d.path, d.line, lang) for d in sec.descs)
        sections.append({"title": sec.title, "slug": sec.slug, "desc": desc,
                         "functions": [_function(ctx, f, sec) for f in sec.functions]})
    return {"lang": lang, "display": config.LANG_TITLE[lang], "title": intro.title, "description": intro.description,
            "lead": ctx.inline(intro.lead, intro.path, lang), "others": [x for x in config.LANGS if x != lang],
            "notes": ctx.fragment(intro.body, "prose", intro.path, intro.body_line, lang, intro.headings),
            "sections": sections}


def _prose(ctx, page, lang=None):
    return ctx.fragment(page.body, "prose", page.path, page.body_line, lang, page.headings)


def publications(ctx, data, path):
    def text(s):
        return ctx.inline(s, path)

    groups = []
    for g in data["group"]:
        entries = [{"year": e["year"], "title": e["title"], "meta": "%s · %s" % (e["authors"], e["venue"]),
                    "links": [(k["label"], links.url(k["url"], ctx.location)) for k in e["link"]]}
                   for e in g["entry"]]
        groups.append({"id": g["id"], "title": g["title"], "intro": text(g.get("intro", "")),
                       "outro": text(g.get("outro", "")), "entries": entries})
    p = data["page"]
    return {"title": p["title"], "description": p["description"], "lead": text(p["lead"]), "groups": groups}


def _plain(html_text):
    return html.unescape(re.sub(r"<[^>]*>", "", str(html_text)))


def tutorial_page(ctx, t):
    cells = []
    for c in t.cells:
        if c.kind == "markdown":
            cells.append({"md": ctx.fragment(c.lines, "notebook", t.path, None, headings=c.headings,
                                             note="cell %d: " % c.number)})
        else:
            outs = []
            for o in c.outputs:
                if o.kind == "image":
                    outs.append(Markup('<img src="%s" alt="%s" loading="lazy">') % (o.src, o.alt))
                else:
                    outs.append(Markup("<pre>%s</pre>") % o.text)
            cells.append({"md": None, "label": c.count, "lang": t.lexer,
                          "code": highlight("\n".join(c.lines), t.lexer), "outs": outs})
    summary = ctx.inline(t.summary, t.path)
    return {"stem": t.stem, "title": t.title, "summary": summary, "description": _plain(summary),
            "language": t.language, "level": t.level.capitalize(), "duration": t.duration, "cells": cells}


def render_site(model, diag):
    """The site: {path relative to website/: bytes}, and the number of gap markers the HTML pass replaced.

    model: the qdocs model ("api", "pages", "publications") plus "docs" (the docs/ mapping),
    "website" ({page id: prose.Page} of website_builder/content/), "tutorials" (tutorials.Tutorial list),
    "files" (further files: static parts, skill, tutorial twins, notebooks and figures).
    """
    ctx, env = Context(diag), environment()
    figures, og = assets.landing_figures(diag), assets.og_image(diag)
    pages = {}

    def page(path, template, root="", current=None, twin=None, **values):
        ctx.location = path
        base = {"root": root, "path": path, "current": current, "twin": twin, "og_image": og,
                "url": lambda pid: root + pid + ".html"}
        pages[path] = env.get_template(template).render(**base, **values)

    page("index.html", "landing.html", current="index", figures=figures, title="Quadriga-Lib", description="")
    for lang in config.LANGS:
        pid = lang + "_api"
        ctx.location = pid + ".html"
        api = api_page(ctx, model["api"][lang])
        page(pid + ".html", "api.html", current=pid, twin=config.TWINS[pid], api=api, title=api["title"],
             description=api["description"])
    for pid, p in sorted(model["pages"].items()):
        ctx.location = pid + ".html"
        if pid == "formats":
            page("formats.html", "formats.html", current=pid, twin=config.TWINS[pid], page=p,
                 lead=ctx.inline(p.lead, p.path), body=_prose(ctx, p), formats=formats_page(ctx, model["formats"]),
                 intro_index=_index(ctx, p, 1), title=p.title, description=p.description)
            continue
        page(pid + ".html", "text.html", current=pid, twin=config.TWINS[pid], page=p, lead=ctx.inline(p.lead, p.path),
             body=_prose(ctx, p), index=_index(ctx, p), title=p.title, description=p.description)
    ctx.location = "publications.html"
    pubs = publications(ctx, model["publications"], config.DOCS_CONTENT + "/publications.toml")
    page("publications.html", "publications.html", current="publications", twin="publications.md", pubs=pubs,
         title=pubs["title"], description=pubs["description"])
    # Website-only pages: the tutorial index and the not-found page have their own templates, every
    # other one (Impressum, privacy, any page added later) is a plain text page without index
    web = model["website"]
    for pid, p in web.items():
        if pid == "tutorials":
            continue
        ctx.location = pid + ".html"
        page(pid + ".html", "notfound.html" if pid == "404" else "legal.html", root="/" if pid == "404" else "",
             page=p, body=_prose(ctx, p), title=p.title, description=p.description)

    tuts = model["tutorials"]
    ctx.location = "tutorials.html"
    idx = web["tutorials"]
    cards = [{"stem": t.stem, "title": t.title, "summary": ctx.inline(t.summary, t.path), "language": t.language,
              "level": t.level.capitalize()} for t in tuts]
    page("tutorials.html", "tutorials_index.html", current="tutorials", twin="tutorials.md", page=idx,
         lead=ctx.inline(idx.lead, idx.path), body=_prose(ctx, idx), cards=cards,
         title=idx.title, description=idx.description)
    for t in tuts:
        path = "tutorials/%s.html" % t.stem
        ctx.location = path
        tut = tutorial_page(ctx, t)
        page(path, "tutorial.html", root="../", current="tutorials", twin="tutorials/%s.md" % t.stem, tut=tut,
             title=t.title, description=tut["description"])

    pages = formulas(ctx, pages)
    files, markers = {}, 0
    for path, text in pages.items():
        text, n = gap_pass(text)
        markers += n
        files[path] = (text.rstrip("\n") + "\n").encode("utf-8")
    files.update({p: b for p, b in model["docs"].items() if p != "README.md"})
    files.update(model["files"])
    return {p: files[p] for p in sorted(files)}, markers
