# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Tutorial notebooks: reading and checking them (never running or rewriting), their outputs and figures,
and the Markdown twins of the tutorial pages and of the tutorial index."""

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
             "order": (int, True), "dummy": (bool, False)}
LEVELS = ("beginner", "intermediate", "advanced")
FOLDER = "tutorials"


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
    order: int
    dummy: bool
    language: str       # display name
    lexer: str          # fence language of the code cells
    cells: list
    files: dict         # site path -> bytes: notebook copy, figures, twin


def _text(value):
    return "".join(value) if isinstance(value, list) else (value or "")


def read_all(root, diag):
    """All notebooks of tutorials/, sorted by file name; a notebook with errors is left out."""
    folder = root / FOLDER
    out = []
    for p in sorted(folder.glob("*.ipynb")) if folder.is_dir() else []:
        t = read(p, "%s/%s" % (FOLDER, p.name), diag)
        if t:
            out.append(t)
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
    if info.get("level") not in LEVELS:
        diag.error(rel, None, "metadata.quadriga_tutorial.level: one of %s" % ", ".join(LEVELS))
    kernel = meta.get("kernelspec", {}).get("language")
    if kernel not in LANGUAGES:
        diag.error(rel, None, "metadata.kernelspec.language: missing or not one of %s" % ", ".join(LANGUAGES))
    if diag.errors > errors:
        return None
    language, lexer = LANGUAGES[kernel]
    t = Tutorial(stem, rel, info["title"], info["summary"], info["level"], info["duration_min"], info["order"],
                 bool(info.get("dummy")), language, lexer, [], {"%s/%s.ipynb" % (FOLDER, stem): raw})
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
    # One heading scan over all Markdown cells keeps the ids unique on the page; headings deeper
    # than #### are written as #### (h4), as on the page
    heads = prose.heading_ids(md_lines, diag, rel)
    for cell, off in offsets:
        cell.headings = [dataclasses.replace(h, index=h.index - off, level=min(h.level, 4)) for h in heads
                         if off <= h.index < off + len(cell.lines)]
        if any(h.level == 1 for h in cell.headings):
            diag.error(rel, None, "cell %d: a notebook starts its sections with ##; the page title comes from "
                       "the notebook metadata" % cell.number)
    if diag.errors > errors:
        return None
    t.files["%s/%s.md" % (FOLDER, stem)] = twin(t).encode("utf-8")
    return t


def _outputs(t, c, n, rel, diag):
    alt = (c.get("metadata", {}).get("quadriga_tutorial") or {}).get("alt")
    out = []
    for k, o in enumerate(c.get("outputs", []), 1):
        kind = o.get("output_type")
        if kind == "stream":
            out.append(Output("text", ANSI_RE.sub("", _text(o.get("text"))).rstrip("\n")))
        elif kind == "error":
            diag.error(rel, None, "cell %d: the notebook was committed with a failing cell (%s)" % (n, o.get("ename")))
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


def twin(t):
    """tutorials/<stem>.md: twin header, the downloads, Markdown cells through the twin writer, code and outputs."""
    loc = "%s/%s.md" % (FOLDER, t.stem)
    h2 = [(h.text, h.id, None) for c in t.cells for h in c.headings if h.id and h.level == 2]
    out = render_md.twin_header(t.title, "%s/%s/%s.html" % (config.SITE_BASE_URL, FOLDER, t.stem), h2)
    out += ["", "[Download notebook .ipynb](%s.ipynb) · [Markdown .md](%s.md)" % (t.stem, t.stem)]
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


def index_twin(page, tutorials):
    """tutorials.md: the twin of the index page with one list item per tutorial."""
    loc = "tutorials.md"
    text = render_md.prose_twin(page, loc).rstrip("\n").split("\n")
    items = ["- [%s](%s/%s.md) · %s · %s · %s" % (t.title, FOLDER, t.stem, t.language, t.level,
                                                    " ".join(render_md.twin_lines([t.summary], loc)))
             for t in tutorials] or ["No tutorials yet."]
    return "\n".join(text + [""] + items) + "\n"
