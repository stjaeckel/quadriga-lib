# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Markdown output of docs/: twins written from the source text, the full API files, the
per-section files, publications, llms.txt and docs/README.md."""

import re

from . import config, docblocks, links, prose

MASK = "\x00"  # stands for code in masked lines, which have the length of the source lines
_TICKS_RE = re.compile(r"`+")
# One pass finds [[name]], raw <a href> and Markdown links and images; code is masked out first
_INLINE_RE = re.compile(
    r"(?P<wiki>(?P<dot>\.?)\[\[(?P<wname>[^\[\]\x00]+)\]\])"
    r"|(?P<raw><a\s[^>\x00]*?\bhref=\"(?P<href>[^\"\x00]*)\"[^>\x00]*>(?P<atext>.*?)</a>)"
    r"|(?P<md>(?<!\\)(?P<bang>!?)\[(?P<text>(?:\\.|[^\]\\])*)\]\((?P<url>[^)\s\x00]*)(?P<title>\s+\"[^\"\x00]*\")?\))")
_TRAILING_BR_RE = re.compile(r"(?:\s*<br\s*/?>)+\s*$")
_BLOCK_START_RE = re.compile(r"^\s*(?:[-*+] |\d+[.)] |\||#|>|```|~~~)")
_MARKER_ONLY_RE = re.compile(r"^[\s>]*(?:[-*+]|\d+[.)])?\s*$")
_GAP_SUB_RE = re.compile(r" ?%%GAP:.*?%%")

API_TITLES = {"cpp": "C++ API Documentation for Quadriga-Lib",
              "mex": "MATLAB / Octave API Documentation for Quadriga-Lib",
              "python": "Python API Documentation for Quadriga-Lib"}


def code_mask(lines, diag=None, path=None, first_line=1):
    """Return the lines with code replaced by MASK: fence lines, and code spans, which may run across
    the lines of a paragraph (a blank line or a fence ends it). A backtick run left open at the end of
    a paragraph is literal text; then the rest of that paragraph is masked as well (left unrewritten)
    and, with diag, the warning code-span is given."""
    states, _ = docblocks.fence_states(lines)
    out = [MASK * len(ln) if states[i] else ln for i, ln in enumerate(lines)]

    def mask(i0, c0, i1, c1):
        for i in range(i0, i1 + 1):
            a = c0 if i == i0 else 0
            b = c1 if i == i1 else len(lines[i])
            out[i] = out[i][:a] + MASK * (b - a) + out[i][b:]

    para = []
    for i in range(len(lines) + 1):
        if i < len(lines) and not states[i] and lines[i].strip():
            para.append(i)
            continue
        opened = None  # (line, column, run length)
        for k in para:
            ln = lines[k]
            for m in _TICKS_RE.finditer(ln):
                if opened is None:
                    if m.start() and ln[m.start() - 1] == "\\":
                        continue  # an escaped backtick is text
                    opened = (k, m.start(), len(m.group()))
                elif len(m.group()) == opened[2]:
                    mask(opened[0], opened[1], k, m.end())
                    opened = None
        if opened:
            if diag:
                diag.warning(path, first_line + opened[0], "backtick run without a closing run in its paragraph; "
                             "the rest of the paragraph is left unrewritten", "code-span")
            mask(opened[0], opened[1], para[-1], len(lines[para[-1]]))
        para = []
    return out


def scan_links(lines):
    """Yield (line index, URL) of every Markdown link and image outside code, and of raw <a href>."""
    for i, ln in enumerate(code_mask(lines)):
        for m in _INLINE_RE.finditer(ln):
            if m.group("md"):
                yield i, m.group("url")
            elif m.group("raw"):
                yield i, m.group("href")


def wikilinks(lines):
    """Yield (line index, name, member) of every [[name]] outside code."""
    for i, ln in enumerate(code_mask(lines)):
        for m in _INLINE_RE.finditer(ln):
            if m.group("wiki"):
                yield i, m.group("wname"), bool(m.group("dot"))


def _rewrite(line, masked, location, lang, html):
    """Rewrite the links of one line outside code for the output location; raw <a href> only with html."""
    out, pos = [], 0
    for m in _INLINE_RE.finditer(masked):
        out.append(line[pos:m.start()])
        src = line[m.start():m.end()]
        if m.group("wiki"):
            anchor = links.resolve(lang, m.group("wname"), bool(m.group("dot"))) if lang else None
            if anchor:
                text = line[m.start("dot"):m.end("dot")] + line[m.start("wname"):m.end("wname")]
                src = "[%s](%s)" % (text, links.url("%s_api.html#%s" % (lang, anchor), location))
        elif m.group("raw"):
            if html:
                src = "[%s](%s)" % (line[m.start("atext"):m.end("atext")], links.url(m.group("href"), location))
        else:
            u = links.url(m.group("url"), location)
            src = line[m.start():m.start("url")] + u + line[m.end("url"):m.end()]
        out.append(src)
        pos = m.end()
    out.append(line[pos:])
    return "".join(out)


def twin_lines(lines, location, lang=None, headings=None, fence_lang=None, docblock=False, html=False,
               diag=None, path=None, first_line=1):
    """The twin writer for one part: copy the source lines and change only headings, bare fences,
    [[name]] links, raw <a href> links, trailing <br>, link URLs (all outside code) and gap markers.

    location: output path relative to the site root (see links.url).
    lang: resolve [[name]] and .[[name]] on the API page of this language (doc blocks, API intros).
    headings: prose.heading_ids(lines) for prose and notebook cells; each heading loses its
      trailing {...} and gets <a id> anchors on the line before it. None for doc-block parts.
    fence_lang, docblock: doc-block parts; fences without info string get fence_lang, and the
      sub-block heading "## X:" becomes "### X:".
    html: the part is from a doc block, where raw HTML is interpreted: raw <a href> links become
      Markdown links and a trailing <br> a hard break. In prose and notebooks raw HTML is text,
      so it stays as written.
    diag, path, first_line: give the code-span warning for this part (once per part).
    Gap markers are removed; a line or list item holding only a marker is dropped.
    """
    work, dropped = [], set()
    for i, ln in enumerate(lines):
        if "%%GAP:" in ln:
            new = _GAP_SUB_RE.sub("", ln)
            if _MARKER_ONLY_RE.match(new):
                dropped.add(len(work))
                work.append(None)
                continue
            ln = new.rstrip()
        work.append(ln)
    heads = {h.index: h for h in headings or []}
    states, _ = docblocks.fence_states([ln or "" for ln in work])
    anchors = {}
    for i, ln in enumerate(work):
        if ln is None or states[i] == "in":
            continue
        if states[i] == "open":
            s = ln.strip()
            if fence_lang and docblocks.FENCE_RE.match(s) and s == s[0] * len(s):  # no info string
                work[i] = ln.rstrip() + fence_lang
        elif docblock and ln.startswith("## "):
            work[i] = "#" + ln
        elif i in heads:
            h = heads[i]
            work[i] = "#" * h.level + " " + h.text
            if h.id:
                anchors[i] = '<a id="%s"></a>' % h.id
    keep = [i for i, ln in enumerate(work) if ln is not None]
    texts = [work[i] for i in keep]
    masked = code_mask(texts, diag, path, first_line)
    out = []
    for n, i in enumerate(keep):
        ln, mk = texts[n], masked[n]
        if i - 1 in dropped and out and not out[-1].strip() and not ln.strip():
            continue  # a dropped marker line leaves no double blank line
        if i in anchors:
            out.append(anchors[i])
        new = _rewrite(ln, mk, location, lang, html)
        br = _TRAILING_BR_RE.search(mk) if html else None
        if br:  # a hard break if the next line continues the paragraph, else nothing
            nxt = texts[n + 1] if n + 1 < len(texts) else ""
            hard = nxt.strip() and not _BLOCK_START_RE.match(nxt) and not ln.lstrip().startswith(("|", "#"))
            new = new[:len(new) - (len(ln) - br.start())] + ("\\" if hard else "")
        out.append(new)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def _cell(text):
    return text.replace("|", "\\|")


def twin_header(title, web_url, headings):
    """The lines every twin starts with: '# title', the version and web page line, and an index table.

    headings: [(text, id, topics)] of the page's ## headings; topics is None or a list of (text, id)
    of the ### headings below. The table is written for two or more headings; it has a Topics
    column if at least one row has topics.
    """
    out = ["# " + title, ""] if title else []
    out += ["Quadriga-Lib %s · Web page: %s" % (config.version(), web_url)]
    if len(headings) >= 2:
        topics = any(t for _, _, t in headings)  # a Topics column only if some row has topics
        out += [""] + (["| Section | Topics |", "| --- | --- |"] if topics else ["| Section |", "| --- |"])
        for text, hid, sub in headings:
            row = "| [%s](#%s) |" % (_cell(text), hid)
            if topics:
                row += " %s |" % " · ".join("[%s](#%s)" % (_cell(t), i) for t, i in sub or [])
            out.append(row)
    return out


def _link_text(text):
    """Heading text for an index table: links replaced by their text (a link cannot hold a link)."""
    text = re.sub(r"(\.?)\[\[([^\[\]]+)\]\]", r"\1\2", text)
    return re.sub(r"!?\[((?:\\.|[^\]\\])*)\]\([^)]*\)", r"\1", text)


def index_rows(headings):
    """Rows of a page index: [(text, id, topics)] of the ## headings (from prose.heading_ids) with
    the (text, id) of the ### headings below each; text is the heading text with links as their text."""
    out = []
    for h in headings:
        if h.id and h.level == 2:
            out.append((_link_text(h.text), h.id, []))
        elif h.id and h.level == 3 and out:
            out[-1][2].append((_link_text(h.text), h.id))
    return out


def _join(*blocks):
    """Join blocks of lines with one blank line between non-empty blocks."""
    out = []
    for b in blocks:
        if b:
            if out:
                out.append("")
            out += b
    return out


def _text(lines):
    return "\n".join(lines).rstrip("\n") + "\n"


def prose_twin(page, location=None, lang=None):
    """The whole twin of a prose page (gap markers removed, also from title and lead).
    location defaults to the twin path of the page id (config.TWINS); lang: wikilinks of an API intro."""
    location = location or config.TWINS[page.page_id]
    web = "%s/%s.html" % (config.SITE_BASE_URL, page.page_id)
    head = twin_header(prose.strip_gaps(page.title), web, index_rows(page.headings))
    lead = prose.strip_gaps(page.lead)
    return _text(_join(head,
                       twin_lines(lead.split("\n"), location, lang) if lead else [],
                       twin_lines(page.body, location, lang, page.headings)))


def formats_index(intro, formats):
    """docs/formats.md: the formats page's title, lead and intro, and a table linking each format file."""
    web = config.SITE_BASE_URL + "/formats.html"
    head = twin_header(prose.strip_gaps(intro.title), web, [])
    lead = prose.strip_gaps(intro.lead)
    table = ["| Format | Summary |", "| --- | --- |"]
    table += ["| [%s](formats/%s.md) | %s |" % (_cell(prose.strip_gaps(f.title)), f.page_id,
                                                _cell(" ".join(twin_lines([prose.strip_gaps(f.summary)], "formats.md"))))
              for f in formats]
    return _text(_join(head, twin_lines(lead.split("\n"), "formats.md") if lead else [],
                       twin_lines(intro.body, "formats.md", headings=intro.headings), table))


def format_file(f):
    """docs/formats/<id>.md: twin header with the index of the format's headings, then its text."""
    loc = "formats/%s.md" % f.page_id
    web = "%s/formats.html#%s" % (config.SITE_BASE_URL, f.page_id)
    head = twin_header(prose.strip_gaps(f.title), web, index_rows(f.headings))
    return _text(_join(head, twin_lines(f.body, loc, headings=f.headings)))


def counterpart_line(f, location):
    """'Also in: [C++](...) · Not in: Python': both other languages, as the pills of the HTML page."""
    also, absent = [], []
    for lang in config.LANGS:
        if lang == f.lang:
            continue
        anchor = f.counterparts.get(lang)
        if anchor:
            also.append("[%s](%s)" % (config.LANG_PILL[lang], links.url("%s_api.html#%s" % (lang, anchor), location)))
        else:
            absent.append(config.LANG_PILL[lang])
    return " · ".join(([("Also in: " + " · ".join(also))] if also else []) +
                      ([("Not in: " + ", ".join(absent))] if absent else []))


def _part(part, location, lang, docblock=False):
    return twin_lines(part.lines, location, lang, fence_lang=config.FENCE_LANG[lang], docblock=docblock, html=True)


def function_lines(f, location):
    out = ["---", '<a id="%s"></a>' % f.name, "## " + f.name] + _part(f.brief, location, f.lang)
    out = _join(out, [counterpart_line(f, location)], _part(f.intro, location, f.lang))
    for sub in f.subblocks:
        out = _join(out, twin_lines(sub.heading.lines + sub.content.lines, location, f.lang,
                                    fence_lang=config.FENCE_LANG[f.lang], docblock=True, html=True))
    return out


def _brief(f, location):
    return _cell(" ".join(_part(f.brief, location, f.lang)))


def full_api(page):
    """docs/quadriga_lib_api_<lang>.md: front matter, general usage notes, function index, sections."""
    lang = page.lang
    loc = config.TWINS["%s_api" % lang]
    out = ["---", 'title: "%s v%s"' % (API_TITLES[lang], config.version()), 'author: "Stephan Jaeckel"',
           "lang: en-US", "---", "", '<a id="sec-general-usage-notes"></a>', "# General usage notes"]
    out = _join(out, twin_lines(page.intro.body, loc, lang, page.intro.headings))
    index = ["# Function Index", "", "| Function | Section | Summary |", "| --- | --- | --- |"]
    for sec in page.sections:
        for f in sec.functions:
            index.append("| [%s](#%s) | %s | %s |" % (f.name, f.name, _cell(sec.title), _brief(f, loc)))
    out = _join(out, index)
    for sec in page.sections:
        out = _join(out, ["---"], ['<a id="sec-%s"></a>' % sec.slug, "# " + sec.title],
                    *[_part(d, loc, lang) for d in sec.descs])
        for f in sec.functions:
            out = _join(out, function_lines(f, loc))
    return _text(out)


def section_file(page, sec):
    """docs/api/<lang>/<slug>.md: twin header, function table, description, functions."""
    loc = "api/%s/%s.md" % (page.lang, sec.slug)
    web = "%s/%s_api.html#sec-%s" % (config.SITE_BASE_URL, page.lang, sec.slug)
    index = ["| Function | Summary |", "| --- | --- |"]
    index += ["| [%s](#%s) | %s |" % (f.name, f.name, _brief(f, loc)) for f in sec.functions]
    out = _join(twin_header(sec.title, web, []), index, *[_part(d, loc, page.lang) for d in sec.descs])
    for f in sec.functions:
        out = _join(out, function_lines(f, loc))
    return _text(out)


def intro_file(page):
    """docs/api/<lang>/general-usage-notes.md."""
    loc = "api/%s/general-usage-notes.md" % page.lang
    web = "%s/%s_api.html#sec-general-usage-notes" % (config.SITE_BASE_URL, page.lang)
    head = twin_header("General usage notes", web, index_rows(page.intro.headings))
    return _text(_join(head, twin_lines(page.intro.body, loc, page.lang, page.intro.headings)))


def publications_twin(data):
    """docs/publications.md from publications.toml: per group its heading, intro, entries, outro."""
    loc = "publications.md"
    g = prose.strip_gaps
    page = data["page"]
    head = twin_header(g(page["title"]), config.SITE_BASE_URL + "/publications.html",
                       [(g(x["title"]), x["id"], None) for x in data["group"]])
    out = _join(head, twin_lines([g(page["lead"])], loc) if g(page["lead"]) else [])
    for grp in data["group"]:
        entries = []
        for e in grp["entry"]:
            refs = " · ".join("[%s](%s)" % (g(k["label"]), links.url(k["url"], loc)) for k in e["link"])
            entries.append("- %d · **%s** · %s · %s · %s" % (e["year"], g(e["title"]), g(e["authors"]),
                                                              g(e["venue"]), refs))
        out = _join(out, ['<a id="%s"></a>' % grp["id"], "## " + g(grp["title"])],
                    twin_lines([g(grp["intro"])], loc) if g(grp.get("intro", "")) else [], entries,
                    twin_lines([g(grp["outro"])], loc) if g(grp.get("outro", "")) else [])
    return _text(out)


DOC_LIST = (("Get started", "download.html", "installation and quick start"),
            ("Data formats", "formats.html", "index of the file formats, one Markdown file per format"),
            ("Questions", "faq.html", "FAQ, how to cite"),
            ("Publications", "publications.html", "papers and IEEE 802.11 contributions"),
            ("AI agents", "ai_agents.html", "how to use these files and the agent skill"),
            ("Tutorials", "tutorials.html", "worked examples as Jupyter notebooks"))


def index_file(model, location):
    """llms.txt (location 'llms.txt', absolute links) or docs/README.md (location 'README.md', relative)."""
    u = lambda target: links.url(target, location)  # noqa: E731
    out = ["# Quadriga-Lib", ""]
    if location == "README.md":
        out += ["Generated by `make docs` from the documentation comments and `docs_builder/content/`, do not edit; "
                "quadriga-lib.org shows the latest release, these files this branch as of the last `make docs`.", ""]
    summary = prose.strip_gaps(model["pages"]["ai_agents"].extra.get("llms_summary", ""))
    if summary:
        out += ["> " + summary, ""]
    out += ["Version %s. Every page of quadriga-lib.org has a Markdown version; the API reference exists per "
            "language as one full file and as one file per section." % config.version(), "", "## Documentation"]
    out += ["- [%s](%s): %s" % (t, u(target), d) for t, target, d in DOC_LIST]
    out += ["", "## Data formats"]
    out += ["- [%s](%s): %s" % (prose.strip_gaps(f.title), u("formats.html#" + f.page_id), prose.strip_gaps(f.summary))
            for f in model["formats"]]
    out += ["", "## API reference"]
    out += ["- [%s API, complete](%s)" % (config.LANG_TITLE[lang], u(lang + "_api.html")) for lang in config.LANGS]
    out += ["", "## API sections"]
    for lang in config.LANGS:
        name = config.LANG_TITLE[lang]
        out.append("- [%s: General usage notes](%s)" % (name, u(lang + "_api.html#sec-general-usage-notes")))
        for sec in model["api"][lang].sections:
            n = len(sec.functions)
            out.append("- [%s: %s](%s): %d function%s" % (name, sec.title, u("%s_api.html#sec-%s" % (lang, sec.slug)),
                                                          n, "" if n == 1 else "s"))
    out += ["", "## Optional", "- [Agent skill (placeholder)](%s)" % u("skill/SKILL.md")]
    return _text(out)


def check_parts(model, diag):
    """One pass over every source part of docs/: E-link for unresolved [[name]] and the code-span warning."""
    for lang, page in model["api"].items():
        parts = [(page.intro.path, page.intro.body_line, page.intro.body)]
        for sec in page.sections:
            parts += [(d.path, d.line, d.lines) for d in sec.descs]
            for f in sec.functions:
                parts += [(p.path, p.line, p.lines) for p in [f.brief, f.intro]]
                parts += [(s.heading.path, s.heading.line, s.heading.lines + s.content.lines) for s in f.subblocks]
        for path, first, lines in parts:
            code_mask(lines, diag, path, first)
            for i, name, member in wikilinks(lines):
                if not links.resolve(lang, name, member):
                    diag.error(path, first + i, "E-link: [[%s]] matches no function on the %s API page"
                               % (("." if member else "") + name, lang))
    for page in list(model["pages"].values()) + model["formats"]:
        code_mask(page.body, diag, page.path, page.body_line)


def render_docs(model, diag):
    """The docs/ mapping: {path relative to docs/: bytes}, sorted by path."""
    files = {}
    for lang, page in model["api"].items():
        files[config.TWINS[lang + "_api"]] = full_api(page)
        files["api/%s/general-usage-notes.md" % lang] = intro_file(page)
        for sec in page.sections:
            files["api/%s/%s.md" % (lang, sec.slug)] = section_file(page, sec)
    for pid, page in model["pages"].items():
        files[config.TWINS[pid]] = prose_twin(page) if pid != "formats" else formats_index(page, model["formats"])
    for f in model["formats"]:
        files["formats/%s.md" % f.page_id] = format_file(f)
    files["publications.md"] = publications_twin(model["publications"])
    files["llms.txt"] = index_file(model, "llms.txt")
    files["README.md"] = index_file(model, "README.md")
    return {k: files[k].encode("utf-8") for k in sorted(files)}
