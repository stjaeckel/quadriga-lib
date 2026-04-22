#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import os
import re
import sys
import argparse
from datetime import date


# ---------------------------------------------------------------------------
#  Version detection (from extract_version.py logic)
# ---------------------------------------------------------------------------


def get_quadriga_lib_version(hpp_path=None):
    """Read the version string from quadriga_lib.hpp."""
    search_paths = [hpp_path] if hpp_path else ["include/quadriga_lib.hpp"]
    for path in search_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    m = re.search(r'#define\s+QUADRIGA_LIB_VERSION_STR\s+"([^"]+)"', line)
                    if m:
                        return m.group(1)
            print(
                f"Error: QUADRIGA_LIB_VERSION_STR not found in '{path}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        except FileNotFoundError:
            continue
    print("Error: quadriga_lib.hpp not found in any expected path.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
#  Preambles (placeholder text — edit as needed)
# ---------------------------------------------------------------------------

API_TITLES = {
    "cpp": "C++ API Documentation for Quadriga-Lib",
    "python": "Python API Documentation for Quadriga-Lib",
    "mex": "MATLAB / Octave API Documentation for Quadriga-Lib",
}

API_DEFAULTS = {
    "cpp": "src/",
    "python": "api_python/",
    "mex": "api_mex/",
}


def get_preamble(api_type):
    """Return placeholder preamble Markdown for the given API type."""
    if api_type == "cpp":
        return (
            "# General usage notes\n"
            "- Each function has a 1-line short description, optional detailed notes, a Declaration block, and Inputs/Outputs/Returns sections.\n"
            "- Array sizes follow in backticks, e.g. `[n_rx, n_tx, n_path]`.\n"
            "- All functions and classes live in the `quadriga_lib` namespace.\n"
            '- Default include: `#include "quadriga_lib.hpp"`.\n'
            "- Template parameter `dtype` is `float` or `double` unless stated.\n"
            "- Armadillo types are column-major. Shape notation `[a, b, c]` means `[rows, cols, slices]` for `arma::Cube`; `[rows, cols]` for `arma::Mat`; `[n]` for `arma::Col`/`arma::Row`.\n"
            "- Pointer arguments: `nullptr` skips optional outputs; required inputs throw on `nullptr`.\n"
            "- Output containers are resized automatically unless they already have the correct shape; this invalidates any prior pointers into their memory.\n"
            "- Invalid inputs (shape/domain) cause a `std::invalid_argument`; I/O failures a `std::runtime_error`.\n"
            '- Index conventions: 0-based unless the field is explicitly called "1-based" (which applies to `obj_ind`, `mtl_ind`, `fbs_ind`, `sbs_ind`, and QDANT `id`).\n'
            "- Units: angles in radians (degrees only where stated, e.g. `*_deg`, `*_3dB`); distances in meters; frequencies in Hz; time in seconds; powers linear unless `_dB`.\n"
            "- Coordinate system: GCS = right-handed Cartesian, meters. Euler angles are intrinsic Tait-Bryan in the order (bank=x, tilt=y, heading=z), applied as Rz·Ry·Rx.\n"
            "- Polarization transfer matrix `M`: 8 rows per path, interleaved real/imaginary, order `[ReVV, ImVV, ReVH, ImVH, ReHV, ImHV, ReHH, ImHH]`. A 2-row form `[ReVV, ImVV]` is used for scalar (acoustic) fields.\n"
            "- Speed of light/sound defaults: `299792458.0` m/s (EM), `343.0` m/s (acoustic).\n"
            "- Kernel-selection parameters (`use_kernel`): `0` = auto (CUDA if available and problem large enough, else AVX2 if available, else GENERIC), `1` = GENERIC, `2` = AVX2, `3` = CUDA. Throws if the requested kernel is unavailable.\n"
            "- `gpu_id` is only read when `use_kernel` resolves to CUDA.\n"
        )
    elif api_type == "python":
        return (
            "<!-- PLACEHOLDER: Python API preamble -->\n"
            "<!-- Edit this section to add pip install instructions, "
            "import conventions, and general usage notes for the Python API. -->\n"
        )
    elif api_type == "mex":
        return (
            "<!-- PLACEHOLDER: MEX API preamble -->\n"
            "<!-- Edit this section to add MEX setup instructions, "
            "path configuration, and general usage notes for the MATLAB / Octave API. -->\n"
        )
    return ""


# ---------------------------------------------------------------------------
#  Block extraction (same regexes as extract_html.py)
# ---------------------------------------------------------------------------


def extract_sections(content):
    return re.findall(r"/\*!SECTION\n(.*?)\nSECTION!\*/", content, re.DOTALL)


def extract_md_sections(content):
    return re.findall(r"/\*!MD\n(.*?)\nMD!\*/", content, re.DOTALL)


def extract_sections_desc(content):
    return re.findall(r"/\*!SECTION_DESC\n(.*?)\nSECTION_DESC!\*/", content, re.DOTALL)


# ---------------------------------------------------------------------------
#  Markdown cleanup helpers
# ---------------------------------------------------------------------------


def clean_md(text):
    """
    Light cleanup of the quasi-Markdown found inside /*!MD blocks so that
    the output is valid, self-contained Markdown.
    """
    # Convert <a href="#target">text</a>  →  [text](#target)
    text = re.sub(r'<a\s+href="(#[^"]*)">(.*?)</a>', r"[\2](\1)", text)

    # Convert [[target]]  →  [target](#target)
    text = re.sub(r"\[\[(.*?)\]\]", r"[\1](#\1)", text)

    # Strip <br> / <br/> / <br />  →  just a newline (or blank if already at EOL)
    text = re.sub(r"<br\s*/?>", "", text)

    # Pointer-bold pattern:  ****name**  →  **\*\*name**
    # The source uses this to show a bold C-pointer like ***points**
    text = re.sub(r"\*\*\*\*(\w+)\*\*", r"**\*\*\1**", text)

    return text


def slugify(text):
    """Produce a GitHub-style anchor slug from a heading string."""
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s)
    return s


# ---------------------------------------------------------------------------
#  Main generation
# ---------------------------------------------------------------------------


def generate_markdown(folder_name, api_type, version_path):
    version = get_quadriga_lib_version(version_path)
    title = f"{API_TITLES[api_type]} v{version}"
    today = date.today().strftime("%d.%m.%Y")

    # YAML front matter
    md = "---\n"
    md += f'title: "{title}"\n'
    md += f'author: "Stephan Jaeckel"\n'
    md += f'date: "{today}"\n'
    md += "lang: en-US\n"
    md += "---\n\n"

    # Preamble
    md += get_preamble(api_type) + "\n"

    # ------------------------------------------------------------------
    #  Scan source files
    # ------------------------------------------------------------------
    if not os.path.isdir(folder_name):
        print(f"Error: directory '{folder_name}' not found.", file=sys.stderr)
        sys.exit(1)

    files = sorted(os.listdir(folder_name))
    section_dict = {}  # section  →  [(func_name, md_block, add_space), ...]
    section_desc_dict = {}  # section  →  [desc_block, ...]

    for file_name in files:
        if not (file_name.endswith(".cpp") or file_name.endswith(".md")):
            continue
        filepath = os.path.join(folder_name, file_name)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if "/*!SECTION" not in content:
            continue

        file_sections = extract_sections(content)
        file_md_sections = extract_md_sections(content)
        file_sections_desc = extract_sections_desc(content)

        for section in file_sections:
            if section not in section_dict:
                section_dict[section] = []
                section_desc_dict[section] = []

            for md_block in file_md_sections:
                lines = md_block.split("\n")
                func_name = lines[0].replace("# ", "")
                add_space = "<++>" in func_name
                func_name = func_name.replace("<++>", "")
                if func_name not in [f[0] for f in section_dict[section]]:
                    section_dict[section].append((func_name, md_block, add_space))

            for desc_block in file_sections_desc:
                section_desc_dict[section].append(desc_block)

    # Sort sections alphabetically; within each section: regular first, then
    # member functions (starting with '.'), both groups alphabetically.
    section_dict = dict(sorted(section_dict.items()))
    section_desc_dict = dict(sorted(section_desc_dict.items()))
    for section in section_dict:
        section_dict[section].sort(key=lambda x: (x[0].startswith("."), x[0]))

    sections = list(section_dict.keys())

    # ------------------------------------------------------------------
    #  Function index table (line numbers patched in after body gen)
    # ------------------------------------------------------------------
    md += "# Function Index\n\n"
    md += "| Function | Section | Line |\n"
    md += "| --- | --- | --- |\n"
    line_tokens = {}  # (section, func_name) -> placeholder token
    for section in sections:
        for func_name, _, _ in section_dict[section]:
            token = f"__LINE_{len(line_tokens):04d}__"
            line_tokens[(section, func_name)] = token
            md += f"| [{func_name}](#{slugify(func_name)}) | {section} | {token} |\n"
    md += "\n"
    func_lines = {}  # (section, func_name) -> actual line number

    # ------------------------------------------------------------------
    #  Per-section content
    # ------------------------------------------------------------------
    for section in sections:
        md += "---\n\n"
        md += f"# {section}\n\n"

        # Section descriptions
        for desc_block in section_desc_dict[section]:
            md += clean_md(desc_block).strip() + "\n\n"

        # Full function documentation
        for func_name, md_block, _ in section_dict[section]:
            lines = md_block.split("\n")
            short_desc = lines[1] if len(lines) > 1 else ""

            # Record line number of the upcoming "## func_name" heading
            func_lines[(section, func_name)] = md.count("\n") + 2

            if short_desc:
                md += f"---\n## {func_name}\n{short_desc}\n\n"
            else:
                md += f"---\n## {func_name}\n\n"

            # Emit any intro lines before the first ## subsection
            i = 2  # skip function name line and short description
            intro_lines = []
            while i < len(lines) and not lines[i].startswith("## "):
                intro_lines.append(lines[i])
                i += 1
            intro_text = clean_md("\n".join(intro_lines)).strip()
            if intro_text:
                md += intro_text + "\n\n"

            # Parse ## subsections and emit them as ### headings
            while i < len(lines):
                line = lines[i]
                if line.startswith("## "):
                    subsection_name = clean_md(line[3:]).strip()
                    i += 1
                    block_lines = []
                    while i < len(lines) and not lines[i].startswith("## "):
                        block_lines.append(lines[i])
                        i += 1
                    block_text = clean_md("\n".join(block_lines)).strip()
                    md += f"### {subsection_name}\n{block_text}\n\n"
                else:
                    i += 1

    # Fill in line-number placeholders in the Function Index table
    for key, token in line_tokens.items():
        md = md.replace(token, str(func_lines[key]))

    return md


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def get_parser():
    parser = argparse.ArgumentParser(description="Generate Markdown API documentation from quadriga-lib source files.")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the output Markdown file.",
    )
    parser.add_argument(
        "-a",
        "--api",
        required=True,
        choices=["cpp", "python", "mex"],
        help="API type (determines preamble and default source directory).",
    )
    parser.add_argument(
        "-d",
        "--directory",
        default=None,
        help="Path to folder containing source files. " "Defaults to src/ (cpp), api_mex/ (mex), or api_python/ (python).",
    )
    parser.add_argument(
        "--version-path",
        default=None,
        help="Path to quadriga_lib.hpp (default: include/quadriga_lib.hpp).",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    folder = args.directory if args.directory else API_DEFAULTS[args.api]

    md_output = generate_markdown(folder, args.api, args.version_path)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md_output)


if __name__ == "__main__":
    main()
