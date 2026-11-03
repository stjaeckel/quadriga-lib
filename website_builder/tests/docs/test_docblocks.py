# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Doc-block parser: grammar, structural errors, sub-blocks and parameter lists."""

from qdocs import cli, config, docblocks


def parse(tmp_path, lang, text, name=None):
    folder = tmp_path / config.SOURCE_DIRS[lang]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (name or {"cpp": "a.cpp", "mex": "a.cpp", "python": "qpy_a.cpp"}[lang])).write_text(text, encoding="utf-8")
    diag = cli.Diagnostics()
    return docblocks.read_api(tmp_path, lang, None, diag), diag


def errors(diag):
    return [ln for ln in diag.lines if ": error: " in ln]


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
    assert (sec.title, sec.slug, sec.legacy) == ("Tools", "tools", docblocks.links.legacy_hash("Tools"))


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
    assert errors(diag) == ["src/a.cpp:4: error: E-delim-open: /*!MD without MD!*/ before the next block or the end of the file"]
    _, diag = parse(tmp_path / "x", "cpp", SECTION + "/*!SECTION_DESC\ntext\n")
    assert len(errors(diag)) == 1 and "E-delim-open" in errors(diag)[0]


def test_e_section_count(tmp_path):
    _, diag = parse(tmp_path / "none", "cpp", md("# f", "B", "## Declaration:"))
    assert "E-section-count" in errors(diag)[0]
    _, diag = parse(tmp_path / "two", "cpp", SECTION + SECTION + md("# f", "B", "## Declaration:"))
    assert "E-section-count" in errors(diag)[0]
    _, diag = parse(tmp_path / "lines", "cpp", "/*!SECTION\nTools\nMore\nSECTION!*/\n")
    assert "E-section-count" in errors(diag)[0]
    page, diag = parse(tmp_path / "only", "cpp", SECTION)  # a section without functions is fine
    assert errors(diag) == [] and page.sections[0].functions == []


def test_e_name(tmp_path):
    _, diag = parse(tmp_path, "cpp", SECTION + md("#f", "B", "## Declaration:"))
    assert errors(diag) == ["src/a.cpp:5: error: E-name: the first line of a /*!MD block must be '# name'"]
    _, diag = parse(tmp_path / "b", "cpp", SECTION + md("# 3f", "B", "## Declaration:"))
    assert "E-name" in errors(diag)[0]


def test_e_fence(tmp_path):
    _, diag = parse(tmp_path, "cpp", SECTION + md("# f", "B", "## Declaration:", "```", "void f();"))
    assert errors(diag) == ["src/a.cpp:8: error: E-fence: code fence without a closing fence in the same part"]
    # a "## " line inside a fence does not start a sub-block
    page, diag = parse(tmp_path / "b", "cpp", SECTION + md("# f", "B", "## Declaration:", "```", "## x", "```"))
    assert errors(diag) == [] and len(page.sections[0].functions[0].subblocks) == 1


def test_e_required_and_class_exemption(tmp_path):
    _, diag = parse(tmp_path / "c", "cpp", SECTION + md("# f", "B", "## Inputs:", "- **`x`** — X"))
    assert "E-required" in errors(diag)[0]
    _, diag = parse(tmp_path / "k", "cpp", SECTION + md("# cls", "B", "## Attributes:<br>", "| a |"))
    assert errors(diag) == []
    _, diag = parse(tmp_path / "u", "cpp", SECTION + md("# f", "B", "## Usage:"))
    assert errors(diag) == []
    _, diag = parse(tmp_path / "m", "mex", SECTION + md("# F", "B", "## Declaration:"))
    assert "E-required" in errors(diag)[0]
    _, diag = parse(tmp_path / "p", "python", SECTION + md("# f", "B", "## Example:"))
    assert "E-required" in errors(diag)[0]


def test_sub_block_labels_and_kinds():
    assert docblocks.sub_label("## Attributes:<br>") == "Attributes"
    assert docblocks.sub_label("## Function 'duplicate';") == "Function 'duplicate'"
    assert docblocks.sub_label("## Inputs:") == "Inputs"
    kinds = {lab: docblocks.sub_kind(lab) for lab in
             ["Declaration", "Declarations", "Usage", "Example", "Examples", "Input", "Outputs", "Attributes",
              "Input Arguments", "Optional outputs", "Inputs (split mode)", "Outputs (x)", "In/out (launch)",
              "Returns", "See also", "Simple member functions"]}
    assert [k for k, v in kinds.items() if v == "code"] == ["Declaration", "Declarations", "Usage", "Example", "Examples"]
    assert [k for k, v in kinds.items() if v == "plain"] == ["Returns", "See also", "Simple member functions"]


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
    assert a.desc.lines == ["First line", "lazy continuation", "indented continuation", "", "- nested item",
                            "| t | u |", "| - | - |", "```", "- not an item inside a fence", "```"]
    assert b.desc.lines == ["Second"]
    assert isinstance(other, docblocks.Part) and other.lines[0] == "A paragraph after a blank line ends the item."
    assert other.line == a.line + 12


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
