+++
title = "Questions"
description = "%%GAP: meta description of the FAQ page%%"
llms_summary = "%%GAP: Open-source library built on the 3GPP TR 38.901 and IEEE 802.11 channel models. A fast C++ core with MATLAB, Octave and Python interfaces, from the developer of QuaDRiGa.%%"
+++

%%GAP: text converted from the old page; check it against the new layout (headings for the index)%%

## About the Library {#about-the-library}

### How is Quadriga-Lib related to QuaDRiGa?

[QuaDRiGa](https://quadriga-channel-model.de) is a geometry-based stochastic channel model written in MATLAB. Quadriga-Lib is its C++ successor by the same lead developer: the model components are being rewritten in C++ for speed, with thin interfaces to MATLAB, Octave and Python. It is not a drop-in replacement for the MATLAB toolbox; functions, names and file formats follow their own API. [QuaDRiGa-NG](https://github.com/stjaeckel/QuaDRiGa-NG) is a fork of the MATLAB QuaDRiGa toolbox that uses Quadriga-Lib in its backend.

### How do I cite Quadriga-Lib? {#cite}

If you use Quadriga-Lib in published work, please cite the software and the version you used. The installed version is returned by `quadriga_lib.version()` in Python, `quadriga_lib.version` in MATLAB / Octave and `quadriga_lib::quadriga_lib_version()` in C++. Publications on the models are listed on the [Publications](publications.html) page.

```
@software{quadriga_lib,
  author  = {Jaeckel, Stephan},
  title   = {Quadriga-Lib: C++/MATLAB/Python library for radio channel modelling and simulation},
  url     = {https://github.com/stjaeckel/quadriga-lib},
  version = {<version>},
  year    = {<year>}
}
```

### Is the API stable?

Quadriga-Lib is under active development. Breaking changes are avoided where possible, but they do happen between releases. For reproducible results, use a release version and note it, or check out a release tag of the [GitHub repository](https://github.com/stjaeckel/quadriga-lib). The `main` branch may be ahead of the latest release. The API documentation on this site is generated from the source code of the latest release.

### Which compilers are supported?

Quadriga-Lib needs C++17. It is built and tested with GCC 11 or newer (GCC 12 by default on Linux) and with MSVC 2022 on Windows.

### MATLAB reports a "GLIBCXX ... not found" error when loading the MEX files. What can I do?

MATLAB ships its own, older C++ runtime (`libstdc++`). MEX files built with a newer GCC than your MATLAB release supports fail to load with this error. The build uses GCC 12 by default. If your MATLAB release needs a different version, install it (e.g. `sudo apt install g++-11`) and build with `make gcc_version=11`. The compilers supported by each MATLAB release are listed by MathWorks.

### Can Quadriga-Lib be used together with other C++ libraries?

Yes. Quadriga-Lib uses [Armadillo](https://arma.sourceforge.net) for its matrices and cubes. To exchange data with other libraries, create Armadillo objects that use auxiliary memory, use their STL-style iterators, or get a pointer to the matrix memory with `.memptr()`.

## Using AI Agents {#ai-agents}

%%GAP: section text is a draft, moved from the former AI agents page%%

### Which files can an AI agent read?

| File | Content |
| --- | --- |
| [`llms.txt`](llms.txt) | Short summary of the library and links to all Markdown files |
| `<page>.md` | Markdown version next to every content page, linked in its footer |
| `api/cpp/<section>.md` | One API section per file, small enough to fetch on its own; also for `mex` and `python` |
| [`quadriga_lib_api_cpp.md`](quadriga_lib_api_cpp.md) | Full API per language in one file (also `_mex` and `_python`) |

### How do I point an agent at the docs?

Give your agent the address of `llms.txt` and ask it to fetch only the sections it needs:

```text
Read {{ site_base_url }}/llms.txt, then use the Python API section files to write a script that generates a 3GPP antenna and prints its directivity.
```

The same Markdown files are in the `docs/` folder of the GitHub repository, so an agent working in a clone can read them without network access.

### Is there an agent skill?

> [!WARNING]
> **Placeholder**
> An agent skill with recipes and conventions for Quadriga-Lib is in preparation. Until it exists, the [skill file](skill/SKILL.md) is a placeholder that points to `llms.txt`.

## Bug Reports and Contact {#bug-reports-and-contact}

### How do I report a bug?

Open an issue with the [bug report form](https://github.com/stjaeckel/quadriga-lib/issues/new?template=bug_report.yml) on GitHub. The most useful part is a small, self-contained example that shows the problem, together with the library version and the interface (C++, Python, MATLAB or Octave). Please search the [existing issues](https://github.com/stjaeckel/quadriga-lib/issues?q=is%3Aissue) first; if your problem is already there, add your case as a comment.

### How do I request a feature?

Use the [feature request form](https://github.com/stjaeckel/quadriga-lib/issues/new?template=feature_request.yml). Describe the task you want to solve, not only the function you have in mind.

### Where can I ask how to use the library?

Ask in [GitHub Discussions](https://github.com/stjaeckel/quadriga-lib/discussions). You can also share your results there.

### How do I report a security problem?

Please do not open a public issue. Use GitHub's [private vulnerability reporting](https://github.com/stjaeckel/quadriga-lib/security/advisories/new) instead; only the maintainer sees the report.

### How can I contribute code?

Small fixes are welcome as pull requests on [GitHub](https://github.com/stjaeckel/quadriga-lib). For anything larger, please open an issue first so the approach can be agreed before you write code. Contributions need tests and documentation; see [CONTRIBUTING.md](https://github.com/stjaeckel/quadriga-lib/blob/main/CONTRIBUTING.md).

### Who develops Quadriga-Lib?

Lead development is done by [Stephan Jaeckel](https://sjc-wireless.com).

### How can I contact the developers?

By email to "info \[ÄT] quadriga-lib \[DÖT] org". For bugs, feature requests and usage questions, GitHub is preferred, so that other users can find the answers.

## Distribution License {#distribution-license}

### What is the distribution license for Quadriga-Lib?

Quadriga-Lib is licensed under the [Apache License 2.0](https://opensource.org/license/apache-2-0), a permissive license. See also a [quick summary](https://www.tldrlegal.com/license/apache-license-2-0-apache-2-0) of the license.

### Can I use Quadriga-Lib in commercial products without releasing the source code?

Yes. Quadriga-Lib can be used in open-source and proprietary software. If you distribute it, in source or binary form, keep its copyright and license notice (the `LICENSE` file) with your software or its documentation.

### How does the Apache License compare to the GPL, MIT and BSD licenses?

Unlike the GPL, the Apache License does not require you to publish the source code of software that uses Quadriga-Lib. It is a permissive license like MIT and BSD, and it also grants users the patent rights of the contributors.

### Is Quadriga-Lib a product of the Apache Software Foundation?

No. It only uses the Apache License.

## Supporting the Project {#supporting-the-project}

### How can I support the development?

If Quadriga-Lib is useful to you, you can support its development with a donation. It lets me spend more time on new features, bug fixes and documentation.

- [GitHub Sponsors](https://github.com/sponsors/stjaeckel)
- [PayPal](https://www.paypal.com/donate/?hosted_button_id=B2N5HS5F4HHFE)

### Other ways to help

Report bugs with a small example, answer questions in [Discussions](https://github.com/stjaeckel/quadriga-lib/discussions), and cite the library in your publications.
