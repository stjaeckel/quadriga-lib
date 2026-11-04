# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Doc-block parser: /*!SECTION, /*!SECTION_DESC and /*!MD blocks of the C++, MATLAB and Python sources.

Line based. Each part of a block keeps its source lines with file and first line, so the
Markdown writer can copy them and the website builder can parse them into HTML.
"""

import re
from dataclasses import dataclass, field

from . import config, links

OPEN = {"/*!SECTION": "SECTION", "/*!SECTION_DESC": "SECTION_DESC", "/*!MD": "MD"}
CLOSE = {"SECTION": "SECTION!*/", "SECTION_DESC": "SECTION_DESC!*/", "MD": "MD!*/"}

NAME_RE = re.compile(r"^# (\.?[A-Za-z_][A-Za-z0-9_.]*)(<\+\+>)?$")

CODE_LABELS = {"Declaration", "Declarations", "Usage", "Example", "Examples"}
PARAM_LABELS = {"Input", "Inputs", "Output", "Outputs", "Attributes", "Arguments", "Input Arguments",
                "Output Argument", "Output Arguments", "Optional outputs"}
PARAM_PREFIXES = ("Inputs (", "Outputs (", "In/out")

_NAME = r"`[^`]+`(?:, (?:\.\.\.|`[^`]+`))*"
PARAM_RE = re.compile(
    r"^- (?P<names>\*\*" + _NAME + r"\*\*(?:(?:, | / )\*\*" + _NAME + r"\*\*)*)"
    r"(?:\s+(?P<qual>\*\([^)]*\)\*|\([^)]*\)))?"
    r"(?P<sep>\s+[—–-]\s+|<br>\s*$|\s*$)")
_BOLD_NAME_RE = re.compile(r"\*\*(" + _NAME + r")\*\*")
_CLOSING_FENCE_RE = re.compile(r"`{3,}\s*")


@dataclass
class Part:
    """Source lines of one part of a block; line is the 1-based line of lines[0] in path."""
    path: str
    line: int
    lines: list


@dataclass
class ParamRow:
    names: str          # as written without the bold markers, e.g. "`V_re` / `V_im`"; "" for a note row
    qual: str           # text in the parentheses, e.g. "optional"; "" if absent
    desc: Part          # description lines (the whole item for a note row)
    note: bool          # True: an item that does not match the parameter pattern
    line: int


@dataclass
class SubBlock:
    label: str          # heading text without "## ", trailing <br>, ":" or ";"
    kind: str           # "code", "params" or "plain"
    heading: Part       # the "## X:" line
    content: Part       # the lines below the heading
    items: list = None  # params only: ParamRow and Part (other blocks, in place) in source order


@dataclass
class Function:
    lang: str
    name: str           # display name = anchor
    path: str
    line: int
    brief: Part
    intro: Part
    subblocks: list
    section: str = ""
    stem: str = ""      # family stem of MATLAB and Python files
    counterparts: dict = field(default_factory=dict)  # lang -> anchor of the counterpart there


@dataclass
class Section:
    lang: str
    title: str
    slug: str
    descs: list         # Parts of the SECTION_DESC blocks, in file-name order
    functions: list
    path: str           # where the section first appears
    line: int


@dataclass
class ApiPage:
    lang: str
    intro: object       # prose.Page of docs_builder/content/api_intro_<lang>.md
    sections: list      # sorted by title


def fence_states(lines):
    """Per line: "open" for a fence opener, "in" for a line inside a fence or its closer, "" outside.
    A fence closes on a line of backticks only, at least as many as the opener has.
    Returns (states, line index of an unclosed opener or None)."""
    states, opener, run = [], None, 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if opener is None:
            if s.startswith("```"):
                opener, run = i, len(s) - len(s.lstrip("`"))
                states.append("open")
            else:
                states.append("")
        else:
            states.append("in")
            if _CLOSING_FENCE_RE.fullmatch(s) and len(s) >= run:
                opener = None
    return states, opener


_TABLE_ROW_RE = re.compile(r"^\s*\|")
_DELIMITER_ROW_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
# Lines that end a table: lists, quotes, headings, fences and rules start their own block
_ENDS_TABLE_RE = re.compile(r"^(?:[-*+](?:\s|$)|\d{1,9}[.)](?:\s|$)|>|#{1,6}(?:\s|$)|```|~~~|(?:[-*_]\s*){3,}$)")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d{1,9}[.)])\s+")


WRAP_MESSAGE = "E-wrap: a wrapped line must not start with '>' or '+ '; move the line break"
_WRAP_START_RE = re.compile(r"^(>|\+ )")
_BLOCK_LINE_RE = re.compile(r"^(#{1,6}(\s|$)|```|~~~)")


def wrap_errors(lines):
    """Line indices of wrapped lines that Markdown would read as a block quote or a '+' list item:
    outside fences, a line starting with '>' or '+ ' directly after a non-blank line, unless that
    line is a heading, a fence line or itself starts with the same marker."""
    states, _ = fence_states(lines)
    out = []
    for i in range(1, len(lines)):
        cur, prev = lines[i].strip(), lines[i - 1].strip()
        m = _WRAP_START_RE.match(cur)
        if not m or states[i] or not prev or states[i - 1] or _BLOCK_LINE_RE.match(prev) or \
                prev.startswith(m.group(1)):
            continue
        out.append(i)
    return out


_UNDERLINE_RE = re.compile(r"^ {0,3}(=+|-+)\s*$")
_INDENTED_HEADING_RE = re.compile(r"^\s+#{1,6}(\s|$)")
_NOT_PARAGRAPH_RE = re.compile(r"^(#{1,6}(\s|$)|>|\||```|~~~|<)")
SETEXT_MESSAGE = ("E-heading: the line below this text underlines it with '=' or '-', which makes it a heading; "
                  "write headings as '## text', or add a blank line before a rule")
INDENTED_MESSAGE = "E-heading: a heading must start in column 0 ('## text'); remove the indentation"
H1_MESSAGE = "E-heading: a '#' heading is not allowed in the page body (the title comes from the front matter); use '##'"


def heading_errors(lines, page=False):
    """(line index, message) of headings that the line scan for heading ids would not see:
    a text line underlined with '=' or '-' (a setext heading) and an indented '#' heading,
    outside fences. With page=True (prose pages) also a '#' heading in column 0."""
    states, _ = fence_states(lines)
    out = []
    for i, ln in enumerate(lines):
        if states[i]:
            continue
        if page and re.match(r"^#(\s|$)", ln):
            out.append((i, H1_MESSAGE))
        elif _INDENTED_HEADING_RE.match(ln):
            out.append((i, INDENTED_MESSAGE))
        elif i and _UNDERLINE_RE.match(ln) and not states[i - 1]:
            prev = lines[i - 1]
            text = prev.strip()
            if not text or _NOT_PARAGRAPH_RE.match(text) or _DELIMITER_ROW_RE.match(prev):
                continue
            m = _LIST_MARKER_RE.match(prev)
            if m and (prev[m.end():].strip() == "" or _indent(ln) < m.end()):
                continue  # inside a list item the underline must reach the item's text
            if _UNDERLINE_RE.match(prev):
                continue
            out.append((i - 1, SETEXT_MESSAGE))
    return out


def block_errors(lines, path, first, diag, page=False):
    """Report E-table, E-wrap and E-heading for the lines of a block part or page body;
    first is the line of lines[0]; page=True for prose pages (a '#' heading is an error there)."""
    for k in table_errors(lines):
        diag.error(path, first + k, TABLE_MESSAGE)
    for k in wrap_errors(lines):
        diag.error(path, first + k, WRAP_MESSAGE)
    for k, message in heading_errors(lines, page):
        diag.error(path, first + k, message)


TABLE_MESSAGE = "E-table: a table and the text next to it need a blank line between them"


def _indent(ln):
    return len(ln) - len(ln.lstrip(" "))


def table_errors(lines):
    """Line indices where a Markdown table and its neighbours run together (outside fences):
    a non-blank line directly after a table that does not start a block of its own (it would
    become a table row, at any indent: a parameter item is dedented before it is parsed), and a
    table that directly follows a text
    line whose content starts further right (the table would be read as part of that text)."""
    states, _ = fence_states(lines)
    out, i = [], 0
    while i + 1 < len(lines):
        if states[i] or not _TABLE_ROW_RE.match(lines[i]) or states[i + 1] or not _DELIMITER_ROW_RE.match(lines[i + 1]):
            i += 1
            continue
        indent = _indent(lines[i])
        prev = lines[i - 1] if i else ""
        if prev.strip() and not states[i - 1] and not _TABLE_ROW_RE.match(prev):
            m = _LIST_MARKER_RE.match(prev)
            if (m.end() if m else _indent(prev)) > indent:
                out.append(i)
        j = i + 2
        while j < len(lines) and not states[j] and _TABLE_ROW_RE.match(lines[j]):
            j += 1
        if j < len(lines) and lines[j].strip() and not _ENDS_TABLE_RE.match(lines[j].strip()):
            out.append(j)
        i = j
    return out


def sub_label(heading):
    label = heading[3:].strip()
    if label.endswith("<br>"):
        label = label[:-4].rstrip()
    if label.endswith((":", ";")):
        label = label[:-1].rstrip()
    return label


def sub_kind(label):
    if label in CODE_LABELS:
        return "code"
    if label in PARAM_LABELS or label.startswith(PARAM_PREFIXES):
        return "params"
    return "plain"


def _dedent2(ln):
    return ln[2:] if ln.startswith("  ") else ln.lstrip(" ")


def parse_params(part, diag):
    """Split a params sub-block into parameter rows and other blocks (Parts), in source order."""
    lines, items = part.lines, []
    # A line inside a fence belongs to the item; the opener itself follows the item rules
    in_fence = [st == "in" for st in fence_states(lines)[0]]
    i = other = 0  # other: first line of the pending run of non-item lines
    while i <= len(lines):
        if i < len(lines) and (in_fence[i] or not lines[i].startswith("- ")):
            i += 1
            continue
        if any(ln.strip() for ln in lines[other:i]):
            items.append(Part(part.path, part.line + other, lines[other:i]))
        if i == len(lines):
            break
        j, prev_blank = i + 1, False
        while j < len(lines):
            ln = lines[j]
            if in_fence[j]:
                prev_blank = False
            elif ln.startswith("- "):
                break
            elif not ln.strip():
                prev_blank = True
            elif ln[0] in " \t":
                prev_blank = False
            elif prev_blank:
                break
            j += 1
        item = lines[i:j]
        while item and not item[-1].strip():
            item.pop()
        items.append(_param_row(part, i, item, diag))
        i = other = j
    return items


def _param_row(part, i, item, diag):
    line = part.line + i
    m = PARAM_RE.match(item[0])
    rest = [_dedent2(ln) for ln in item[1:]]
    if not m:
        diag.warning(part.path, line, "list item in a parameter list is not of the form - **`name`** — text",
                     "param-item")
        return ParamRow("", "", Part(part.path, line, [item[0][2:]] + rest), True, line)
    sep = m.group("sep").strip()
    if sep in ("-", "–"):
        diag.warning(part.path, line, "parameter separator '%s' instead of ' — '" % sep, "param-separator")
    names = _BOLD_NAME_RE.sub(r"\1", m.group("names"))
    qual = (m.group("qual") or "").strip("*")[1:-1] if m.group("qual") else ""
    first = item[0][m.end():]
    if first.strip():
        desc = Part(part.path, line, [first] + rest)
    else:
        desc = Part(part.path, line + 1, rest)
    return ParamRow(names, qual, desc, False, line)


def split_blocks(text, path, diag):
    """Return [(kind, first content line, lines)] of the blocks of one file."""
    lines = text.split("\n")
    blocks, i = [], 0
    while i < len(lines):
        kind = OPEN.get(lines[i])
        if not kind:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and lines[j] != CLOSE[kind] and lines[j] not in OPEN:
            j += 1
        if j == len(lines) or lines[j] != CLOSE[kind]:
            diag.error(path, i + 1, "E-delim-open: %s without %s before the next block or the end of the file"
                       % (lines[i], CLOSE[kind]))
            i = j
            continue
        blocks.append((kind, i + 2, lines[i + 1:j]))
        i = j + 1
    return blocks


def parse_function(lang, path, first, lines, diag):
    """Parse one /*!MD block; return a Function or None after an error."""
    m = NAME_RE.match(lines[0]) if lines else None
    if not m:
        diag.error(path, first, "E-name: the first line of a /*!MD block must be '# name'")
        return None
    name = m.group(1).lower() if lang == "mex" else m.group(1)
    brief = Part(path, first + 1, lines[1:2])
    if brief.lines and brief.lines[0].strip().startswith("```"):
        diag.error(path, first + 1, "E-fence: code fence without a closing fence in the same part")
        return None
    body = lines[2:]
    states, opener = fence_states(body)
    if opener is not None:
        diag.error(path, first + 2 + opener, "E-fence: code fence without a closing fence in the same part")
        return None
    block_errors(body, path, first + 2, diag)
    starts = [k for k, ln in enumerate(body) if ln.startswith("## ") and not states[k]]
    intro_end = starts[0] if starts else len(body)
    intro = Part(path, first + 2, body[:intro_end])
    subs = []
    for n, k in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(body)
        label = sub_label(body[k])
        sub = SubBlock(label, sub_kind(label), Part(path, first + 2 + k, [body[k]]),
                       Part(path, first + 3 + k, body[k + 1:end]))
        if sub.kind == "params":
            sub.items = parse_params(sub.content, diag)
        subs.append(sub)
    labels = {s.label for s in subs}
    if lang == "cpp":
        if "Attributes" not in labels and not labels & {"Declaration", "Declarations", "Usage"}:
            diag.error(path, first, "E-required: C++ block '%s' has no ## Declaration: (or Declarations, Usage)" % name)
    elif "Usage" not in labels:
        diag.error(path, first, "E-required: block '%s' has no ## Usage:" % name)
    return Function(lang, name, path, first, brief, intro, subs)


def read_api(root, lang, intro, diag):
    """Parse the doc blocks of one language into an ApiPage (sections sorted, functions sorted)."""
    folder = config.SOURCE_DIRS[lang]
    sections = {}
    for p in sorted((root / folder).glob("*.cpp"), key=lambda p: p.name):
        path = "%s/%s" % (folder, p.name)
        blocks = split_blocks(p.read_text(encoding="utf-8"), path, diag)
        if not blocks:
            continue
        secs = [b for b in blocks if b[0] == "SECTION"]
        others = [b for b in blocks if b[0] != "SECTION"]
        if len(secs) != 1:
            if others or len(secs) > 1:
                where = secs[1][1] - 1 if len(secs) > 1 else others[0][1] - 1
                diag.error(path, where, "E-section-count: a file with blocks needs exactly one /*!SECTION block, found %d"
                           % len(secs))
            continue
        sec_lines = secs[0][2]
        if len(sec_lines) != 1 or not sec_lines[0].strip():
            diag.error(path, secs[0][1] - 1, "E-section-count: a /*!SECTION block holds exactly one non-empty line")
            continue
        title = sec_lines[0].strip()
        sec = sections.get(title)
        if sec is None:
            sec = sections[title] = Section(lang, title, links.slug(title), [], [],
                                             path, secs[0][1] - 1)
        stem = p.stem
        if lang == "python" and stem.startswith("qpy_"):
            stem = stem[4:]
        for kind, first, lines in others:
            if kind == "SECTION_DESC":
                _, opener = fence_states(lines)
                if opener is not None:
                    diag.error(path, first + opener, "E-fence: code fence without a closing fence in the same part")
                    continue
                block_errors(lines, path, first, diag)
                sec.descs.append(Part(path, first, lines))
            else:
                fn = parse_function(lang, path, first, lines, diag)
                if fn:
                    fn.section, fn.stem = title, (stem if lang != "cpp" else "")
                    sec.functions.append(fn)
    for sec in sections.values():
        sec.functions.sort(key=lambda f: (f.name.startswith("."), f.name))
    return ApiPage(lang, intro, [sections[t] for t in sorted(sections)])
