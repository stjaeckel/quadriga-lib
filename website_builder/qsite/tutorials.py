# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Tutorial notebooks: reading and checking their executed copies (never running or rewriting anything), their
outputs and figures, and the Markdown twins of the tutorial pages and of the tutorial index."""

import base64
import dataclasses
import json
import re
from dataclasses import dataclass, field

from qdocs import config, prose, render_md

STEM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LANGUAGES = {"python": ("Python", "python"), "matlab": ("MATLAB", "matlab"), "octave": ("Octave", "matlab")}
META_KEYS = {"title": (str, True), "summary": (str, True), "level": (str, True), "duration_min": (int, True),
             "order": (int, True), "series": (str, False)}
LEVELS = ("beginner", "intermediate", "advanced")
FOLDER = "tutorials"
EXECUTED = "tutorials/executed"  # written by the tutorial build (tutorials/GNUmakefile), not by this builder
SERIES_DIR = "website_builder/content/tutorials"  # one file per series: title, order and its introduction
SERIES_KEYS = {"title": (str, True), "order": (int, True)}
OTHER = ("other", "Other tutorials")  # id and title of the group of tutorials without a series


@dataclass
class Output:
    kind: str           # "text" or "image"
    text: str = ""      # text outputs
    src: str = ""       # image path relative to the tutorial page: <stem>/fig-<cell>-<output>.<ext>
    alt: str = ""


@dataclass
class Cell:
    kind: str           # "markdown" or "code"
    number: int         # 1-based position among all cells
    lines: list
    headings: list = field(default_factory=list)  # markdown: qdocs Headings, index relative to lines
    count: str = " "    # code: execution count as shown in "In [n]"
    outputs: list = field(default_factory=list)


@dataclass
class Tutorial:
    stem: str
    path: str           # tutorials/<stem>.ipynb
    title: str
    summary: str
    level: str
    duration: int
    order: int          # position in its series (or among the tutorials without a series)
    series: str         # series id (file stem in SERIES_DIR), "" without a series
    language: str       # display name
    lexer: str          # fence language of the code cells
    cells: list
    files: dict         # site path -> bytes: notebook copy, figures, twin


def _text(value):
    return "".join(value) if isinstance(value, list) else (value or "")


def read_all(root, diag, release=False):
    """The tutorials in index order (order, then title): one per notebook tutorials/<stem>.ipynb, read from its
    executed copy tutorials/executed/<stem>.ipynb; diagnostics name the notebook. A notebook with errors is left
    out. Missing executed copies leave their tutorials out with the warning not-executed (an error with
    release); without tutorials/executed/ the index lists no tutorial."""
    folder = root / FOLDER
    sources = sorted(folder.glob("*.ipynb")) if folder.is_dir() else []
    if not sources:
        return []
    if release:
        report = diag.error
    else:
        def report(path, line, message):
            diag.warning(path, line, message, "not-executed")
    executed = root / EXECUTED
    if not executed.is_dir():
        report(EXECUTED, None, "missing: the tutorials have not been run (make tutorials); the tutorials page "
               "lists none")
        return []
    out = []
    for p in sources:
        rel = "%s/%s" % (FOLDER, p.name)
        run = executed / p.name
        if not run.is_file():
            report(rel, None, "no executed copy %s/%s (make tutorials); left out" % (EXECUTED, p.name))
            continue
        t = read(run, rel, diag)
        if t:
            if _meta(p) not in (None, _meta(run)):
                report(rel, None, "the executed copy %s/%s has other tutorial metadata than the notebook "
                       "(make tutorials)" % (EXECUTED, p.name))
            out.append(t)
    return sorted(out, key=lambda t: (t.order, t.title))


def _meta(p):
    """metadata.quadriga_tutorial of a notebook file, None if the file is not a JSON notebook."""
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("metadata", {}).get("quadriga_tutorial")
    except (ValueError, AttributeError):
        return None


def left_out(root):
    """The site files of the tutorials that read_all leaves out because they have not been run: {site path:
    (warning kind, note)} for tutorials/<stem>.html, .md and .ipynb of each notebook without an executed copy.
    The link checks report links to them as warnings; links to tutorials without a notebook stay errors."""
    folder = root / FOLDER
    sources = sorted(folder.glob("*.ipynb")) if folder.is_dir() else []
    out = {}
    for p in sources:
        if not (root / EXECUTED / p.name).is_file():
            for ext in ("html", "md", "ipynb"):
                out["%s/%s.%s" % (FOLDER, p.stem, ext)] = ("not-executed", "tutorial not run, make tutorials")
    return out


def read(p, rel, diag):
    errors = diag.errors
    stem = p.stem
    if not STEM_RE.match(stem):
        diag.error(rel, None, "notebook file name must match [a-z0-9][a-z0-9_-]*.ipynb")
    raw = p.read_bytes()
    try:
        nb = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        diag.error(rel, None, "not a JSON notebook: %s" % e)
        return None
    meta = nb.get("metadata", {})
    info = meta.get("quadriga_tutorial")
    if not isinstance(info, dict):
        diag.error(rel, None, "metadata.quadriga_tutorial: missing")
        return None
    prose.check_keys(info, META_KEYS, "metadata.quadriga_tutorial.", lambda msg: diag.error(rel, None, msg))
    if "level" in info and info["level"] not in LEVELS:  # a missing level is reported by check_keys
        diag.error(rel, None, "metadata.quadriga_tutorial.level: one of %s" % ", ".join(LEVELS))
    kernel = meta.get("kernelspec", {}).get("language")
    if kernel not in LANGUAGES:
        diag.error(rel, None, "metadata.kernelspec.language: missing or not one of %s" % ", ".join(LANGUAGES))
    if diag.errors > errors:
        return None
    language, lexer = LANGUAGES[kernel]
    t = Tutorial(stem, rel, info["title"], info["summary"], info["level"], info["duration_min"], info["order"],
                 info.get("series", ""), language, lexer, [], {"%s/%s.ipynb" % (FOLDER, stem): raw})
    md_lines, offsets = [], []
    for n, c in enumerate(nb.get("cells", []), 1):
        lines = _text(c.get("source")).split("\n")
        if c.get("cell_type") == "markdown":
            cell = Cell("markdown", n, lines)
            offsets.append((cell, len(md_lines)))
            md_lines += lines + [""]
        elif c.get("cell_type") == "code":
            cell = Cell("code", n, lines)
            if c.get("execution_count") is None:
                diag.warning(rel, None, "cell %d: code cell without an execution count (not run)" % n, "not-run")
            else:
                cell.count = str(c["execution_count"])
            cell.outputs = _outputs(t, c, n, rel, diag)
        else:
            continue
        t.cells.append(cell)
    # One heading scan over all Markdown cells keeps the ids unique on the page
    heads = prose.heading_ids(md_lines, _CellDiag(diag, offsets), rel)
    for cell, off in offsets:
        cell.headings = [dataclasses.replace(h, index=h.index - off) for h in heads
                         if off <= h.index < off + len(cell.lines)]
        if any(h.level == 1 for h in cell.headings):
            diag.error(rel, None, "cell %d: a notebook starts its sections with ##; the page title comes from "
                       "the notebook metadata" % cell.number)
    if diag.errors > errors:
        return None
    return t


@dataclass
class Series:
    id: str             # file stem; the section id on tutorials.html
    title: str
    page: object        # prose.Page of the series file, None for the group without a series
    tutorials: list     # in series order (order, then title)

    @property
    def numbered(self):
        """A series has parts (Part n of m, previous and next); the group without a series has none."""
        return self.page is not None


def read_series(root, index, diag):
    """The series files in page order (order, then id); their heading ids share one set with the index page's
    headings, the series ids and the id of the group without a series, since all are on tutorials.html."""
    folder = root / SERIES_DIR
    files = sorted(folder.glob("*.md"), key=lambda p: p.name) if folder.is_dir() else []
    pages = sorted((prose.read_page(p, diag, root, SERIES_KEYS, ids=False) for p in files),
                   key=lambda s: (s.order, s.page_id))
    for h in index.headings:
        if h.id == OTHER[0]:
            diag.error(index.path, index.body_line + h.index, "E-anchor: the id '%s' is the section of the "
                       "tutorials without a series" % OTHER[0])
    used = set(config.RESERVED_IDS) | {h.id for h in index.headings if h.id} | {OTHER[0]}
    for s in pages:
        if not STEM_RE.match(s.page_id):
            diag.error(s.path, None, "series file name must match [a-z0-9][a-z0-9_-]*.md")
        if s.page_id in used:
            diag.error(s.path, None, "E-anchor: the series id '%s' is already an id on tutorials.html" % s.page_id)
        used.add(s.page_id)
    for s in pages:  # in page order, so that a repeated heading gets -2 in the later section
        s.headings = prose.heading_ids(s.body, diag, s.path, s.body_line, used)
    return pages


def _declared(root):
    """{series id: [notebook path]} from the committed notebooks, executed or not (unreadable ones are
    reported by read_all)."""
    out = {}
    for p in sorted((root / FOLDER).glob("*.ipynb")) if (root / FOLDER).is_dir() else []:
        info = _meta(p)
        if isinstance(info, dict) and isinstance(info.get("series"), str) and info["series"]:
            out.setdefault(info["series"], []).append("%s/%s" % (FOLDER, p.name))
    return out


def group(root, pages, tutorials, diag):
    """The groups of tutorials.html: one Series per series file (also without tutorials, e.g. before the
    tutorials are run), then the tutorials without a series; also writes each tutorial's twin, which names its
    place in the series. The series ids are checked in the committed
    notebooks, run or not: an id without a file is an error; a series file that no notebook names gives the
    warning series-empty."""
    by_id = {s.page_id: Series(s.page_id, s.title, s, []) for s in pages}
    other = Series(OTHER[0], OTHER[1], None, [])
    declared = _declared(root)
    for sid, paths in sorted(declared.items()):
        if sid not in by_id:
            for rel in paths:
                diag.error(rel, None, "metadata.quadriga_tutorial.series: no series file %s/%s.md"
                           % (SERIES_DIR, sid))
    for t in tutorials:  # an executed copy that differs from its notebook was reported by read_all
        by_id.get(t.series, other).tutorials.append(t)
    for s in pages:
        if s.page_id not in declared:
            diag.warning(s.path, None, "no notebook names this series (metadata.quadriga_tutorial.series)",
                         "series-empty")
    out = list(by_id.values()) + ([other] if other.tutorials else [])
    for g in out:
        g.tutorials.sort(key=lambda t: (t.order, t.title))
        for t in g.tutorials:
            t.files["%s/%s.md" % (FOLDER, t.stem)] = twin(t, g).encode("utf-8")
    return out


class _CellDiag:
    """Reports errors at a line of the joined Markdown cells as errors of that cell."""

    def __init__(self, diag, offsets):
        self.diag, self.offsets = diag, offsets

    def error(self, path, line, message):
        cell = [c for c, off in self.offsets if off <= line - 1][-1]
        self.diag.error(path, None, "cell %d: %s" % (cell.number, message))


def _outputs(t, c, n, rel, diag):
    alt = (c.get("metadata", {}).get("quadriga_tutorial") or {}).get("alt")
    out = []
    for k, o in enumerate(c.get("outputs", []), 1):
        kind = o.get("output_type")
        if kind == "stream":
            out.append(Output("text", ANSI_RE.sub("", _text(o.get("text"))).rstrip("\n")))
        elif kind == "error":
            diag.error(rel, None, "cell %d: the executed copy has a failing cell (%s)" % (n, o.get("ename")))
        elif kind in ("execute_result", "display_data"):
            data = o.get("data", {})
            src = "%s/fig-%d-%d" % (t.stem, n, k)
            if "image/svg+xml" in data:
                src += ".svg"
                t.files["%s/%s" % (FOLDER, src)] = _text(data["image/svg+xml"]).encode("utf-8")
            elif "image/png" in data:
                src += ".png"
                t.files["%s/%s" % (FOLDER, src)] = base64.b64decode(_text(data["image/png"]))
            elif "text/plain" in data:
                out.append(Output("text", ANSI_RE.sub("", _text(data["text/plain"])).rstrip("\n")))
                continue
            else:
                diag.warning(rel, None, "cell %d: output %d has no SVG, PNG or plain text; left out" % (n, k),
                             "output-type")
                continue
            if not alt:
                diag.warning(rel, None, "cell %d: image without metadata.quadriga_tutorial.alt" % n, "figure-alt")
            out.append(Output("image", src=src, alt=alt or "Figure output of cell %d" % n))
    return out


def _fence(lines, lang):
    ticks = "`" * max([3] + [len(m) + 1 for ln in lines for m in re.findall(r"`{3,}", ln)])
    return [ticks + lang] + lines + [ticks]


def twin(t, g):
    """tutorials/<stem>.md: twin header, the downloads, the place in the series g (part n of m, previous and
    next; not for the tutorials without a series), Markdown cells through the twin writer, code and outputs."""
    loc = "%s/%s.md" % (FOLDER, t.stem)
    rows = render_md.index_rows([h for c in t.cells for h in c.headings])  # links as their text
    h2 = [(text, hid, None) for text, hid, _ in rows]
    out = render_md.twin_header(t.title, "%s/%s/%s.html" % (config.SITE_BASE_URL, FOLDER, t.stem), h2)
    out += ["", "[Download notebook .ipynb](%s.ipynb) · [Markdown .md](%s.md)" % (t.stem, t.stem)]
    if g.numbered:
        n = g.tutorials.index(t)
        line = ["Part %d of %d of [%s](../tutorials.md#%s)" % (n + 1, len(g.tutorials),
                                                             prose.plain_text(prose.strip_gaps(g.title)), g.id)]
        if n > 0:
            line.append("Previous: [%s](%s.md)" % (g.tutorials[n - 1].title, g.tutorials[n - 1].stem))
        if n + 1 < len(g.tutorials):
            line.append("Next: [%s](%s.md)" % (g.tutorials[n + 1].title, g.tutorials[n + 1].stem))
        out += ["", " · ".join(line)]
    for c in t.cells:
        if c.kind == "markdown":
            block = render_md.twin_lines(c.lines, loc, headings=c.headings)
        else:
            block = _fence(c.lines, t.lexer)
            for o in c.outputs:
                block += [""] + (["![%s](%s)" % (o.alt, o.src)] if o.kind == "image" else _fence(o.text.split("\n"), "text"))
        if block:
            out += [""] + block
    return "\n".join(out) + "\n"


def index_twin(page, groups):
    """tutorials.md: the twin of the index page; per series a ## heading, its introduction (headings one level
    down) and one list item per tutorial."""
    loc = "tutorials.md"
    text = render_md.prose_twin(page, loc).rstrip("\n").split("\n")
    for g in groups:
        text += ["", '<a id="%s"></a>' % g.id, "## " + prose.strip_gaps(g.title)]
        if g.page is not None and g.page.body:
            heads = [dataclasses.replace(h, level=h.level + 1) for h in g.page.headings]
            body = render_md.twin_lines(g.page.body, loc, headings=heads)
            text += [""] + body if body else []
        items = ["%s [%s](%s/%s.md) · %s · %s · about %d min · %s" % (
                     "%d." % n if g.numbered else "-", t.title, FOLDER, t.stem, t.language, t.level, t.duration,
                     " ".join(render_md.twin_lines([t.summary], loc)))
                 for n, t in enumerate(g.tutorials, 1)]
        text += [""] + items if items else []
    return "\n".join(text) + "\n"
