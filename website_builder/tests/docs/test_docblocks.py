# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Doc-block parser: grammar, structural errors, sub-blocks and parameter lists."""

import re

from qdocs import cli, config, docblocks


def parse(tmp_path, lang, text, name=None):
    folder = tmp_path / config.SOURCE_DIRS[lang]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (name or {"cpp": "a.cpp", "mex": "a.cpp", "python": "qpy_a.cpp"}[lang])).write_text(text, encoding="utf-8")
    diag = cli.Diagnostics()
    return docblocks.read_api(tmp_path, lang, None, diag), diag


def errors(diag):
    return [ln for ln in diag.lines if ": error: " in ln]


def error_ids(diag):
    """[(line, error id)] of the errors, without the message text."""
    out = []
    for ln in diag.lines:
        m = re.match(r"^[^:]+:(\d+): error: (E-[a-z-]+):", ln)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


def warnings(diag, kind):
    return [ln for ln in diag.lines if ln.endswith("[%s]" % kind)]


SECTION = "/*!SECTION\nTools\nSECTION!*/\n"


def md(*lines):
    return "/*!MD\n" + "\n".join(lines) + "\nMD!*/\n"


CPP_FN = md("# f", "Brief", "", "- intro", "", "## Declaration:", "```", "void f();", "```",
            "", "## Inputs:", "- **`x`** — The x", "", "## See also:", "- [[g]]")


def test_line_numbers_and_parts(tmp_path):
    page, diag = parse(tmp_path, "cpp", "// header\n" + SECTION + "\n" + CPP_FN)
    assert errors(diag) == []
    f = page.sections[0].functions[0]
    assert (f.path, f.line) == ("src/a.cpp", 7)  # the "# f" line
    assert (f.brief.line, f.brief.lines) == (8, ["Brief"])
    assert f.intro.line == 9 and f.intro.lines == ["", "- intro", ""]
    decl, inputs, see = f.subblocks
    assert (decl.label, decl.kind, decl.heading.line, decl.content.line) == ("Declaration", "code", 12, 13)
    assert (inputs.label, inputs.kind) == ("Inputs", "params")
    row = inputs.items[0]
    assert (row.names, row.qual, row.desc.lines, row.desc.line, row.note) == ("`x`", "", ["The x"], 18, False)
    assert (see.label, see.kind) == ("See also", "plain")


def test_section_and_function_order(tmp_path):
    text = SECTION + md("# .b", "B", "## Declaration:") + md("# z", "Z", "## Declaration:") + md("# a", "A", "## Usage:")
    page, diag = parse(tmp_path, "cpp", text)
    assert [f.name for f in page.sections[0].functions] == ["a", "z", ".b"]
    sec = page.sections[0]
    assert (sec.title, sec.slug) == ("Tools", "tools")


def test_display_names(tmp_path):
    page, _ = parse(tmp_path, "mex", SECTION + md("# ARRAYANT_GENERATE", "B", "## Usage:"))
    assert page.sections[0].functions[0].name == "arrayant_generate"
    page, _ = parse(tmp_path / "p", "python", SECTION + md("# sos.acf_2d", "B", "## Usage:"))
    assert page.sections[0].functions[0].name == "sos.acf_2d"
    page, diag = parse(tmp_path / "c", "cpp", SECTION + md("# arrayant<++>", "B", "## Attributes:<br>", "| a | b |"))
    assert page.sections[0].functions[0].name == "arrayant" and errors(diag) == []


def test_section_descriptions_in_file_order(tmp_path):
    desc = "/*!SECTION_DESC\n%s\nSECTION_DESC!*/\n"
    parse(tmp_path, "cpp", SECTION + desc % "second", name="b.cpp")
    page, diag = parse(tmp_path, "cpp", SECTION + desc % "first", name="a.cpp")
    assert [d.lines for d in page.sections[0].descs] == [["first"], ["second"]]


def test_e_delim_open(tmp_path):
    _, diag = parse(tmp_path, "cpp", SECTION + "/*!MD\n# f\nBrief\n/*!MD\n# g\nB\n## Declaration:\nMD!*/\n")
    assert error_ids(diag) == [(4, "E-delim-open")]
    _, diag = parse(tmp_path / "x", "cpp", SECTION + "/*!SECTION_DESC\ntext\n")
    assert [i for _, i in error_ids(diag)] == ["E-delim-open"]


def test_e_section_count(tmp_path):
    _, diag = parse(tmp_path / "none", "cpp", md("# f", "B", "## Declaration:"))
    assert [i for _, i in error_ids(diag)] == ["E-section-count"]
    _, diag = parse(tmp_path / "two", "cpp", SECTION + SECTION + md("# f", "B", "## Declaration:"))
    assert [i for _, i in error_ids(diag)] == ["E-section-count"]
    _, diag = parse(tmp_path / "lines", "cpp", "/*!SECTION\nTools\nMore\nSECTION!*/\n")
    assert [i for _, i in error_ids(diag)] == ["E-section-count"]
    page, diag = parse(tmp_path / "only", "cpp", SECTION)  # a section without functions is fine
    assert errors(diag) == [] and page.sections[0].functions == []


def test_e_name(tmp_path):
    _, diag = parse(tmp_path, "cpp", SECTION + md("#f", "B", "## Declaration:"))
    assert error_ids(diag) == [(5, "E-name")]
    _, diag = parse(tmp_path / "b", "cpp", SECTION + md("# 3f", "B", "## Declaration:"))
    assert [i for _, i in error_ids(diag)] == ["E-name"]


def test_e_fence(tmp_path):
    _, diag = parse(tmp_path, "cpp", SECTION + md("# f", "B", "## Declaration:", "```", "void f();"))
    assert error_ids(diag) == [(8, "E-fence")]
    # a "## " line inside a fence does not start a sub-block
    page, diag = parse(tmp_path / "b", "cpp", SECTION + md("# f", "B", "## Declaration:", "```", "## x", "```"))
    assert errors(diag) == [] and len(page.sections[0].functions[0].subblocks) == 1


def test_fence_closes_on_a_run_as_long_as_the_opener(tmp_path):
    states, opener = docblocks.fence_states(["````", "```", "## x", "````"])
    assert states == ["open", "in", "in", "in"] and opener is None
    assert docblocks.fence_states(["````", "```"])[1] == 0
    _, diag = parse(tmp_path, "cpp", SECTION + md("# f", "B", "## Declaration:", "````", "```", "x"))
    assert [i for _, i in error_ids(diag)] == ["E-fence"]


def test_e_table_text_after_table(tmp_path):
    _, diag = parse(tmp_path, "cpp", SECTION + md("# f", "B", "## Declaration:", "## Inputs:",
                                                  "- **`x`** — Codes:", "  | a | b |", "  | - | - |", "  | 1 | 2 |",
                                                  "  Text read as a row", "- **`y`** — Next item ends the table",
                                                  "  | a | b |", "  | - | - |", "- **`z`** — Fine"))
    assert error_ids(diag) == [(13, "E-table")]
    page, diag = parse(tmp_path / "ok", "cpp", SECTION + md("# f", "B", "## Declaration:", "Text before",
                                                            "| a | b |", "| - | - |", "", "Text after", "| a |", "| - |",
                                                            "```", "| not a table |", "text", "```"))
    assert errors(diag) == []


def test_e_table_column_0_text_after_an_indented_table(tmp_path):
    _, diag = parse(tmp_path, "cpp", SECTION + md("# f", "B", "## Declaration:", "## Inputs:",
                                                  "- **`x`** — Codes:", "  | a | b |", "  | - | - |",
                                                  "lazy text in column 0"))
    assert error_ids(diag) == [(12, "E-table")]


def test_e_wrap(tmp_path):
    _, diag = parse(tmp_path, "cpp", SECTION + md("# f", "B", "## Declaration:", "## Inputs:",
                                                  "- **`x`** — Distance, finite and", "  >= 0; `[n]`",
                                                  "- **`y`** — Sum `a + b", "  + c`"))
    assert error_ids(diag) == [(10, "E-wrap"), (12, "E-wrap")]
    _, diag = parse(tmp_path / "desc", "cpp", SECTION + "/*!SECTION_DESC\nText\n> quote\nSECTION_DESC!*/\n")
    assert error_ids(diag) == [(6, "E-wrap")]
    # fine: fenced code, a quote after a fence, a blank line or a heading, a quote or '+' list continued;
    # a wrapped code span is not: the line still becomes a block quote
    _, diag = parse(tmp_path / "ok", "cpp", SECTION + md("# f", "B", "## Declaration:", "```", "a", "+ b", "```",
                                                         "> quote", "> more", "", "> [!NOTE]", "> text", "",
                                                         "#### Heading", "> quote", "", "+ one", "+ two",
                                                         "Text with `x", ">= 0` in code"))
    assert error_ids(diag) == [(24, "E-wrap")]


def test_e_heading_setext_and_indented(tmp_path):
    _, diag = parse(tmp_path, "cpp", SECTION + md("# f", "B", "", "Intro text", "----------", "",
                                                  "  ## Indented heading", "## Declaration:", "```", "Code",
                                                  "---", "```"))
    assert error_ids(diag) == [(8, "E-heading"), (11, "E-heading")]
    _, diag = parse(tmp_path / "desc", "cpp", SECTION + "/*!SECTION_DESC\nTitle\n=====\nSECTION_DESC!*/\n")
    assert error_ids(diag) == [(5, "E-heading")]
    # rules after a blank line, table delimiter rows, a list item's rule and '#' name lines are fine
    _, diag = parse(tmp_path / "ok", "cpp", SECTION + md("# f", "B", "", "Text", "", "---", "| a |", "| - |", "",
                                                         "- item", "---", "## Declaration:"))
    assert errors(diag) == []


def test_e_table_after_deeper_text(tmp_path):
    _, diag = parse(tmp_path, "cpp", SECTION + md("# f", "B", "## Declaration:", "## Inputs:",
                                                  "- **`x`** — Modes", "  - v1: text", "  | a | b |", "  | - | - |"))
    assert error_ids(diag) == [(11, "E-table")]
    _, diag = parse(tmp_path / "ok", "cpp", SECTION + md("# f", "B", "## Declaration:", "## Inputs:",
                                                         "- **`x`** — Modes", "  - v1: text", "    | a | b |",
                                                         "    | - | - |"))
    assert errors(diag) == []


def test_e_required_and_class_exemption(tmp_path):
    _, diag = parse(tmp_path / "c", "cpp", SECTION + md("# f", "B", "## Inputs:", "- **`x`** — X"))
    assert [i for _, i in error_ids(diag)] == ["E-required"]
    _, diag = parse(tmp_path / "k", "cpp", SECTION + md("# cls", "B", "## Attributes:<br>", "| a |"))
    assert errors(diag) == []
    _, diag = parse(tmp_path / "u", "cpp", SECTION + md("# f", "B", "## Usage:"))
    assert errors(diag) == []
    _, diag = parse(tmp_path / "m", "mex", SECTION + md("# F", "B", "## Declaration:"))
    assert [i for _, i in error_ids(diag)] == ["E-required"]
    _, diag = parse(tmp_path / "p", "python", SECTION + md("# f", "B", "## Example:"))
    assert [i for _, i in error_ids(diag)] == ["E-required"]


def test_sub_block_labels_and_kinds():
    assert docblocks.sub_label("## Attributes:<br>") == "Attributes"
    assert docblocks.sub_label("## Function 'duplicate';") == "Function 'duplicate'"
    assert [docblocks.sub_kind(lab) for lab in ("Usage", "Inputs (split mode)", "See also")] == ["code", "params", "plain"]


def params(tmp_path, *lines):
    page, diag = parse(tmp_path, "cpp", SECTION + md("# f", "B", "## Declaration:", "## Inputs:", *lines))
    return page.sections[0].functions[0].subblocks[1].items, diag


def test_param_items_continuation_and_end(tmp_path):
    items, diag = params(tmp_path,
                         "- **`a`** — First line",
                         "lazy continuation",
                         "  indented continuation",
                         "",
                         "  - nested item",
                         "",
                         "  | t | u |",
                         "  | - | - |",
                         "  ```",
                         "- not an item inside a fence",
                         "  ```",
                         "- **`b`** — Second",
                         "",
                         "A paragraph after a blank line ends the item.",
                         "",
                         "| x | y |")
    assert errors(diag) == [] and warnings(diag, "param-item") == []
    a, b, other = items
    assert a.desc.lines == ["First line", "lazy continuation", "indented continuation", "", "- nested item", "",
                            "| t | u |", "| - | - |", "```", "- not an item inside a fence", "```"]
    assert b.desc.lines == ["Second"]
    assert isinstance(other, docblocks.Part) and other.lines[0] == "A paragraph after a blank line ends the item."
    assert other.line == a.line + 13


def test_param_pattern_variants(tmp_path):
    items, diag = params(tmp_path,
                         "- **`x`** *(optional)* — Opt",
                         "- **`y`** (optional, in/out) — Opt",
                         "- **`Dx`, `Dy`, `Dz`, `*D`** *(in/out)* — Several",
                         "- **`V_re`** / **`V_im`** — Slash",
                         "- **`a`**, **`b`** — Comma",
                         "- **`p`**, ... — no: dots belong inside the bold part",
                         "- **`c`**<br>",
                         "  Next line",
                         "- **`d`**",
                         "  Also next line",
                         "- **`e`** - Hyphen",
                         "- **`f`** – En dash",
                         "- Free text item")
    rows = [(r.names, r.qual, r.desc.lines, r.note) for r in items]
    assert rows[:5] == [("`x`", "optional", ["Opt"], False), ("`y`", "optional, in/out", ["Opt"], False),
                        ("`Dx`, `Dy`, `Dz`, `*D`", "in/out", ["Several"], False),
                        ("`V_re` / `V_im`", "", ["Slash"], False), ("`a`, `b`", "", ["Comma"], False)]
    assert rows[5][3] is True  # a note row
    assert rows[6] == ("`c`", "", ["Next line"], False) and rows[7] == ("`d`", "", ["Also next line"], False)
    assert rows[8][:3] == ("`e`", "", ["Hyphen"]) and rows[9][:3] == ("`f`", "", ["En dash"])
    assert rows[10] == ("", "", ["Free text item"], True)
    assert len(warnings(diag, "param-separator")) == 2 and len(warnings(diag, "param-item")) == 2
    assert warnings(diag, "param-item")[1].startswith("src/a.cpp:%d: warning:" % items[10].line)


def test_tilde_fences(tmp_path):
    states, opener = docblocks.fence_states(["~~~~", "```", "~~~", "## x", "~~~~~", "after"])
    assert states == ["open", "in", "in", "in", "in", ""] and opener is None
    assert docblocks.fence_states(["~~~", "```"])[1] == 0
    page, diag = parse(tmp_path, "cpp", SECTION + md("# f", "B", "## Declaration:", "~~~", "## not a sub-block", "~~~"))
    assert errors(diag) == [] and len(page.sections[0].functions[0].subblocks) == 1
    _, diag = parse(tmp_path / "open", "cpp", SECTION + md("# f", "B", "## Declaration:", "~~~", "x", "```"))
    assert error_ids(diag) == [(8, "E-fence")]
