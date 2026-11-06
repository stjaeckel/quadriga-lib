# Tutorials

This folder holds the tutorials of Quadriga-Lib: Jupyter notebooks that teach a topic with working code and are published on the website, [quadriga-lib.org/tutorials.html](http://quadriga-lib.org/tutorials.html). This README describes how a tutorial is written, run, checked and submitted. It applies to all tutorials.

Contributions are welcome. For anything beyond a small fix, please open an issue first so that the topic and scope can be agreed; how to contribute is described in [CONTRIBUTING.md](../CONTRIBUTING.md). The build described here runs with GNU make on Linux; a route for Windows does not exist yet.

- [Adding a tutorial](#adding-a-tutorial)
- [Environment](#environment)
- [Starting a notebook](#starting-a-notebook)
- [Naming and metadata](#naming-and-metadata)
- [Editing notebooks](#editing-notebooks)
- [Notebook structure](#notebook-structure)
- [Writing](#writing)
- [Figures](#figures)
- [Results and sources](#results-and-sources)
- [Running the notebooks](#running-the-notebooks)
- [Checking the page on the website](#checking-the-page-on-the-website)
- [Before submitting](#before-submitting)
- [Folder layout](#folder-layout)

## Adding a tutorial

1. Set up an environment with Quadriga-Lib and Jupyter ([Environment](#environment)).
2. Create the notebook under its final name, with the Python kernel and the tutorial metadata ([Starting a notebook](#starting-a-notebook)).
3. Write it ([Editing notebooks](#editing-notebooks) to [Results and sources](#results-and-sources)).
4. Run it: `make -C tutorials executed/<stem>.ipynb`, or `make tutorials` for all notebooks ([Running the notebooks](#running-the-notebooks)).
5. Build the website and look at the page ([Checking the page on the website](#checking-the-page-on-the-website)).
6. Clear the notebook, go through [Before submitting](#before-submitting), commit it on a branch and open a pull request.

## Environment

A tutorial runs in an ordinary Python environment with Quadriga-Lib, numpy, matplotlib and Jupyter; the notebooks use the Jupyter kernel `python3` of that environment. Use a Python version for which Quadriga-Lib publishes wheels; they are listed on its [PyPI page](https://pypi.org/project/quadriga-lib/).

With conda (the Miniforge installer uses the conda-forge channel only):

```bash
conda create -n quadriga-tutorials -c conda-forge python=<version> numpy matplotlib jupyter
conda activate quadriga-tutorials
pip install quadriga-lib
```

With a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install quadriga-lib numpy matplotlib jupyter
```

Tutorials are written against the next release, so they may use functions that are not on PyPI yet; an existing tutorial can then fail with a released package (e.g. `AttributeError: module 'quadriga_lib' has no attribute ...`). In that case, build the library from the source tree and install its Python module into the active environment with `make python_install` from the repository root. This needs a C++ compiler and CMake; the steps and requirements are in [Building from source](../README.md#building-from-source) in the repository README.

Building the website needs further tools: pandoc and a few Python packages, listed in [Set up a build machine](../website_builder/README.md#set-up-a-build-machine) in the website README. `make website` uses the `python3` found first on the `PATH`, so activate the environment that has these packages before building; it can be the same environment as for the tutorials or another one.

Readers set up their environment from the "Setup" section of the tutorial index page of the website. Its text is maintained in `website_builder/content/tutorials.md` and applies to all tutorials; packages that only one tutorial or series needs are named in that tutorial.

## Starting a notebook

A new tutorial is created under its final name, `tutorials/<stem>.ipynb` ([Naming and metadata](#naming-and-metadata)). Files with other names, such as JupyterLab's `Untitled.ipynb`, are picked up by the build as well and stop the website build, because the name does not follow the rules; rename them before running the build.

1. In JupyterLab, create a notebook with the Python 3 kernel and rename it to `<stem>.ipynb` in `tutorials/`. Saving it with a kernel writes `metadata.kernelspec`, which the website builder needs.
2. Add the tutorial metadata `quadriga_tutorial` to the notebook metadata ([Naming and metadata](#naming-and-metadata)).
3. Save.

A notebook can also start as a copy of an existing tutorial: copy the file, rename it, replace the metadata and the cells. A copy keeps the cell ids of the original, which is harmless for a new file.

## Naming and metadata

The file name `<stem>.ipynb` uses lower-case letters, digits, `_` and `-`, and starts with a letter or digit; the stem is also the name of the web page. Tutorials of a series share a prefix and a number, `<series>_<nn>_<topic>`, so that they sort together and keep their order when topics are renamed.

The notebook metadata `metadata.quadriga_tutorial` describes the tutorial:

| Key | Type | Use |
|-----|------|-----|
| `title` | string | Page title and card title |
| `summary` | string, inline Markdown | Card text; as plain text the page description |
| `level` | `beginner`, `intermediate` or `advanced` | Tag on card and page |
| `duration_min` | integer | Reading time in minutes |
| `order` | integer | Position on the index (then by title); leave gaps between series (e.g. 10, 11, 12 for one series, 20, 21 for the next) so that tutorials can be inserted |
| `dummy` | boolean, optional | Placeholder notebook; if every notebook is a placeholder, the index shows the `dummy_note` of `content/tutorials.md` as a warning box |

Example, as it appears in the notebook metadata:

```json
"quadriga_tutorial": {
  "title": "Random fields and the sum-of-sinusoids method",
  "summary": "What random fields are, and how the sum-of-sinusoids generator of Quadriga-Lib makes them.",
  "level": "beginner",
  "duration_min": 25,
  "order": 10
}
```

The language of the code comes from `metadata.kernelspec.language`, which Jupyter writes when the notebook is saved with a kernel. The website builder accepts `python`, `matlab` and `octave`; this README describes Python notebooks.

Each code cell that produces a figure carries an alt text in its cell metadata, `"quadriga_tutorial": {"alt": "..."}`. The alt text describes what the figure shows (axes, curves, the visible result) for readers who cannot see it. In JupyterLab, cell metadata is edited in the property inspector ([Editing notebooks](#editing-notebooks)).

## Editing notebooks

The committed notebook `<stem>.ipynb`, without outputs, is the source of a tutorial. JupyterLab is the expected editor; VS Code or any other editor that keeps the notebook format works as well.

- Metadata is edited in the property inspector of JupyterLab (the gear icon in the right sidebar, which shows the notebook and cell metadata as JSON) or in the raw JSON of the file.
- The final result is seen in the executed copy `executed/<stem>.ipynb` made by the build ([Running the notebooks](#running-the-notebooks)) or on the website build. A notebook that was run in place while writing stores its outputs and the Python version of the kernel; it has to be cleared before it is committed ([Running the notebooks](#running-the-notebooks)).
- Cell ids stay stable. Jupyter assigns an id once, when a cell is created, and keeps it; copying cells within Jupyter is fine. A tool that creates new ids on every save makes every cell show up as changed in git.
- In Markdown cells, each paragraph is one line, as in the other Markdown files of the repository, so that a change of wording changes one line of the diff.
- A notebook can also be generated with a script (e.g. with `nbformat`). The committed notebook stays the source; such a script is not committed, it keeps the cell ids from one run to the next, and it writes `metadata.kernelspec` and the tutorial metadata.

## Notebook structure

The page title comes from `title` in the metadata, so the notebook has no `#` heading; its sections start with `##`, subsections with `###`. A notebook typically has:

1. A first Markdown cell, without heading: what the tutorial teaches and for whom, and what it needs: the oldest Quadriga-Lib release, the packages, and a link to the setup (`tutorials.html#setup`).
2. A first code cell with the imports and `print(qd.version())`, so that the published page shows the version that made the outputs.
3. A result early: a short cell that shows what the tutorial builds, before the explanation.
4. The sections, each idea with a short code cell.
5. A summary and, where the text cites literature, a references section.

Until the release that a tutorial needs is out, the executed copy is made with a build from the source tree, so the printed version (e.g. `0.12.5`) is older than the release named in the text. The executed copies are made again with the release build before the website is published.

Markdown cells follow the format of the content pages:

- Links are site paths from the site root: `faq.html#cite`, `tutorials.html#setup`, `publications/<file>.pdf`, `python_api.html#<anchor>`. The builder makes them relative to the page. A path relative to the notebook (`../tutorials.html`) does not resolve. Other pages link to a tutorial as `tutorials/<stem>.html`.
- Formulas are written in LaTeX between `$...$` and `$$...$$` and become MathML; pandoc converts them, and the commands known to work are listed in [Formulas](../website_builder/README.md#formulas) in the website README.
- Of each output, the page shows the first of SVG, PNG and plain text; printed text appears as text. An output with none of these is left out with a warning.

## Writing

- The library is called Quadriga-Lib in prose. `quadriga-lib` is the name of the package, `quadriga_lib` the name of the module, and the module is imported as `import quadriga_lib as qd`.
- The text uses American English (realization, modeled, color) and the terms of the API documentation.
- Tutorials are written for engineers and scientists who know the basics of their field but not necessarily the topic. Theory is short and serves the code: enough to understand what the code does, with references for further reading.
- Code cells are short, typically a handful of lines, and each is introduced by the text before it. They use numpy and matplotlib; other packages only where the topic needs them, named at the start. Variable names say what they hold; comments give units and array shapes.
- Terms are introduced before they are used. A term that appears early, e.g. in the first result, is explained in the same sentence or announced for later.
- Outputs are published. They contain nothing private, such as local paths, user names or machine names.
- A tutorial runs in seconds, at most about a minute, so that readers can run it and the build stays practical. Larger computations use smaller sizes and say how a full study would differ.

## Figures

A figure should be readable without the surrounding text.

- Axes have labels with units; maps have a colorbar. Panels that are compared share their scales and colorbar.
- Curves that overlap are drawn so that each stays visible (line styles, widths); a logarithmic axis can show differences in the tails of a distribution.
- Each figure has an alt text ([Naming and metadata](#naming-and-metadata)).
- Figures are PNG, matplotlib's default in Jupyter. SVG output changes at every run (it contains a date and random element ids), so executed copies would differ between runs.
- The website builder warns when an HTML page exceeds 150 kB. Figures are separate files and do not count against this budget; still, they should not be larger than they need to be.

## Results and sources

- Every number in the text is either printed by the notebook or taken from a cited source. After a change of the code or of the library, the printed values are compared with the text again.
- Random results use fixed seeds, so that the page and a reader's run agree. The seeds are chosen so that the shown result is typical, not an extreme case; where a result scatters between realizations, the text says by how much, and quantities defined over the ensemble are estimated over several realizations.
- Statements from the literature cite the publication they come from, with the section, equation or page where the statement can be found. A reference names the version that was read (e.g. an arXiv preprint) and, where it differs, the published version.
- A statement is cited from a source that was read in full, not from its abstract or from another paper that cites it. Where a claim could not be checked in the original, the text says so or leaves the claim out. Conventions (definitions, parameters, normalizations) are attributed only to sources that use them.
- Bibliographic details and DOIs can be checked with the Crossref API, `https://api.crossref.org/works/<doi>`.

## Running the notebooks

```bash
make tutorials                                   # from the repository root: all notebooks
make -C tutorials executed/<stem>.ipynb          # one notebook
make -C tutorials clean                          # delete the executed copies, clear all notebooks
```

`make tutorials` runs every notebook that is newer than its executed copy into `executed/<stem>.ipynb`, four notebooks at a time; `JOBS=8` or `JOBS=1` changes that. It uses the `python3` of the active environment; `PYTHON=/path/to/env/bin/python` selects another one without activating it. An executed copy depends only on its notebook: after a change of Quadriga-Lib or of the environment, `make -C tutorials clean` forces a new run of all notebooks.

- The notebook itself is not changed. The run records no timestamps, so two runs in the same environment give byte-identical copies, and clearing an executed copy gives back the committed notebook.
- A cell that raises an error stops the run of its notebook with the traceback and leaves no executed copy. The other notebooks still run; `make` then exits with an error.
- `make -C tutorials clean` deletes `executed/` and clears every notebook in the folder in place: it removes outputs, execution counts and the metadata a run adds, and keeps cell ids and the tutorial metadata. A notebook that is already clear is not rewritten. To clear only one notebook, run `python3 -m qnb.clear <stem>.ipynb` in this folder.
- From the repository root, `make clean` keeps the executed copies; `make tidy` runs the tutorial clean.

Troubleshooting:

- The kernel prints a warning that it runs "over TCP without encryption". It refers to the connection between the build and the kernel on the local machine (127.0.0.1) and can be ignored.
- `ModuleNotFoundError: No module named 'quadriga_lib'` (or numpy, matplotlib) in a cell: the kernel `python3` belongs to another environment. `jupyter kernelspec list` shows which one; run `make` in the environment that has the packages, or select it with `PYTHON=`.
- `AttributeError: module 'quadriga_lib' has no attribute ...`: the installed Quadriga-Lib is older than the tutorial needs ([Environment](#environment)).
- `qnb.run: No module named 'nbclient'`: Jupyter is not installed in the environment that runs `make`.

## Checking the page on the website

The website builder publishes one tutorial for each notebook `tutorials/<stem>.ipynb` and reads its content from the executed copy `tutorials/executed/<stem>.ipynb`; it never runs or changes the notebooks. Each tutorial gives a page `tutorials/<stem>.html`, a Markdown version `tutorials/<stem>.md`, its figures, and the executed notebook as a download. The tutorial index `tutorials.html` lists the tutorials as cards below a common part.

```bash
make website                                     # in an environment with the website tools
python3 -m http.server --directory website       # then open http://localhost:8000/tutorials.html
```

- The build prints warnings and errors for the whole site, including many that have nothing to do with tutorials. Those of a tutorial name its notebook, `tutorials/<stem>.ipynb`, or its page, `website/tutorials/<stem>.html` (e.g. a broken link); a tutorial is ready when none of them name it.
- Without `tutorials/executed/` the index lists no tutorial, and a notebook without an executed copy is left out; both give the warning `not-executed`. The release build, `python3 website_builder/build.py website --release`, treats them as errors, so a published site has every tutorial.
- `make website` also regenerates `docs/` from the sources of the documentation. Changes there do not come from the tutorial and are not part of its pull request.
- The common part of the index (title, lead, the setup that all tutorials need) is the content page `website_builder/content/tutorials.md`; its format is described in [The tutorial index page](../website_builder/README.md#the-tutorial-index-page) in the website README.

How the builder renders notebooks and the full list of its error messages are in [Tutorials](../website_builder/README.md#tutorials) in the website README.

## Before submitting

- `make tutorials` runs without errors, and `make website` gives no warning or error that names the tutorial.
- The page looks right in the local preview: figures, formulas, links, and the text around each output.
- The text matches the printed values.
- The notebook is cleared, and `git diff` shows only the intended changes (no outputs, no changes in `docs/`).
- The change is committed on a branch and submitted as a pull request, as described in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Folder layout

| Path | Content |
|------|---------|
| `<stem>.ipynb` | The tutorials, committed without outputs |
| `executed/` | The executed copies with their outputs, made by `make tutorials`; not in git |
| `GNUmakefile` | Targets to run and clear the notebooks |
| `qnb/` | Python helpers of the build: `qnb/run.py` runs one notebook into its executed copy, `qnb/clear.py` clears notebooks (standard library only) |
| `README.md` | This file |

The helpers are called as `python3 -m qnb.run` and `python3 -m qnb.clear` from this folder, so that a makefile for another platform can use the same commands.
