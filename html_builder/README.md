# HTML documentation builder

This folder holds the scripts and text that make the Quadriga-Lib website in `html_docu/`. The API pages come from documentation comments in the C++, MATLAB/Octave and Python sources. The other pages come from the Markdown and HTML files in this folder. The same comments also give the MATLAB `help` text.

## Quick start

From the repository root:

```bash
make documentation
```

This writes the HTML pages and the Markdown API files to `html_docu/`. Open `html_docu/index.html` in a browser to check the result.

The MATLAB help files (`+quadriga_lib/*.m`) are made by CMake as part of the normal MEX build (target `mex_documentation`), not by `make documentation`.

## Requirements

- Python 3, standard library only
- GNU make, `sed`, and `grep` with `-P` support (used to read the version from `include/quadriga_lib.hpp`)
- CMake, only for the MATLAB help files

Run every command from the repository root. The scripts read `include/quadriga_lib.hpp` and the source folders by paths relative to the root.

## Layout

| Path | Contents |
|------|----------|
| `extract_html.py` | Builds one HTML page from a page start, an optional source folder, and the shared header and footer |
| `extract_md.py` | Builds one Markdown file with the full API reference of one language |
| `extract_version.py` | Writes the page heading with the current version into line 1 of a page start |
| `extract_matlab_comments.py` | Turns the first documentation block of a MEX source file into a MATLAB `.m` help file |
| `parts/` | `html_head.html.part` and `html_foot.html.part` (shared by all pages, including the menu) and one `<page>.html.part` with the start of each page |
| `formats/` | Text of the Data Formats page |
| `faq/` | Text of the Questions page |
| `download/` | Text of the Download page; `download.md.in` is the template for `download.md` |
| `img/` | Logo source files (SVG and PNG) |

## How the pages are built

`make documentation` (see `GNUmakefile`) runs these steps:

1. Make `download/download.md` from `download/download.md.in`, with `__VERSION__` replaced by the library version.
2. Write the heading with the current version into line 1 of `parts/cpp_api.html.part`, `parts/mex_api.html.part` and `parts/python_api.html.part`.
3. Build each HTML page with `extract_html.py`:

   | Output in `html_docu/` | Page start in `parts/` | Source folder | Options |
   |------------------------|------------------------|---------------|---------|
   | `index.html` | `index.html.part` | none | |
   | `cpp_api.html` | `cpp_api.html.part` | `src/` | |
   | `mex_api.html` | `mex_api.html.part` | `api_mex/` | `-l` |
   | `python_api.html` | `python_api.html.part` | `api_python/` | |
   | `formats.html` | `formats.html.part` | `html_builder/formats/` | |
   | `faq.html` | `faq.html.part` | `html_builder/faq/` | `-c` |
   | `publications.html` | `publications.html.part` | none | |
   | `download.html` | `download.html.part` | `html_builder/download/` | `-c` |

4. Build `quadriga_lib_api_cpp.md`, `quadriga_lib_api_mex.md` and `quadriga_lib_api_python.md` with `extract_md.py`.

Each HTML page is the header, then the page start, then the content from the source folder, then the footer. The header marks the current page in the menu.

Steps 1 and 2 rewrite files in this folder that are under version control. After a version change, commit `download/download.md` and the three page starts together with the pages.

## Documentation comments

The format and content of the blocks are set by `md/SPEC_CPP_API.md` (sections "Section statement" and "Documentation block"), `md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md`. This section covers only what the scripts read.

`extract_html.py` and `extract_md.py` read all `.cpp` and `.md` files in the source folder and skip files without a `/*!SECTION` block. Each delimiter must sit alone on its own line. A minimal C++ block:

````cpp
/*!SECTION
Math functions
SECTION!*/

/*!MD
# my_function
One-line summary

- Description as a bullet list

## Declaration:
```
double quadriga_lib::my_function(double x);
```

## Input:
- **`x`** — Input value

## Returns:
- Result
MD!*/
````

What the scripts take from the blocks:

- `SECTION` names the section. All `MD` blocks of a file go into the sections named in that file. A page lists its sections alphabetically, and the functions alphabetically within each section. Names that start with `.` (member functions) come after the others.
- A `/*!SECTION_DESC ... SECTION_DESC!*/` block adds text under the section heading, before its functions.
- In an `MD` block, line 1 (`# NAME`) is the function name, which is also its link anchor. Line 2 is the short description, used in the overview table, and `## ` starts a subsection. With `-l`, names are shown in lower case (used for MATLAB).
- `<++>` in the name line adds space under that row of the overview table. It is not shown.
- The section anchor is made from the section title. Renaming a section changes its link.
- `extract_matlab_comments.py` uses only the first `MD` block of a file.

The converter turns `` `code` ``, `**bold**`, `*italic*`, code blocks between lines of three backticks, `-` lists (nested by indentation), Markdown pipe tables, `[[name]]` links to a function and `.[[name]]` links to a member function on the same page into HTML, and passes plain HTML through. The SPEC files limit what a block may use; for example, no `*` or `|` in running text.

## Options of `extract_html.py`

| Option | Meaning |
|--------|---------|
| `-o FILE` | Output HTML file (required) |
| `-p FILE` | Page start to insert after the header (required) |
| `-d DIR` | Folder with the source files; without it, the page has only the page start |
| `-c` | Compact layout: no overview table of functions |
| `-l` | Show function names in lower case |

`extract_md.py -o FILE -a cpp|mex|python [-d DIR]` takes the source folder from the API type unless `-d` is given. The preamble of each Markdown file is written in the script.

## Common tasks

- **Document a new function:** add the `SECTION` and `MD` blocks to its source file and run `make documentation`. For a MEX function, the help file comes with the next CMake build.
- **Add a FAQ entry or a Data Formats topic:** edit or add a `.md` file in `faq/` or `formats/`. Files are read in name order.
- **Change the Download page:** edit `download/download.md.in`, not `download/download.md`.
- **Add a page:** add `parts/<page>.html.part`, add the menu entry in `parts/html_head.html.part`, and add a line to the `documentation` target in `GNUmakefile`.

## Files in `html_docu/` that are not generated

`style2.css`, `.htaccess`, the logo PNGs, `publications/` and `quadriga_lib_material_model.md` are maintained by hand in `html_docu/`. The logo PNGs there are copies of the files in `img/`; copy them again after changing a logo.
