+++
title = "AI agents"
description = "%%GAP: How to use Quadriga-Lib with AI coding agents: llms.txt, Markdown versions of every page, per-section API files.%%"
lead = "%%GAP: Every page of this site also exists as plain Markdown, so coding agents can read the reference without parsing HTML.%%"
llms_summary = "%%GAP: Open-source library built on the 3GPP TR 38.901 and IEEE 802.11 channel models. A fast C++ core with MATLAB, Octave and Python interfaces, from the developer of QuaDRiGa.%%"
+++

%%GAP: whole page text is a draft%%

## Files for agents

| File | Content |
| --- | --- |
| [`llms.txt`](llms.txt) | Short summary of the library and links to all Markdown files |
| `<page>.md` | Markdown version next to every content page, linked in its footer |
| `api/cpp/<section>.md` | One API section per file, small enough to fetch on its own; also for `mex` and `python` |
| [`quadriga_lib_api_cpp.md`](quadriga_lib_api_cpp.md) | Full API per language in one file (also `_mex` and `_python`) |

## Pointing an agent at the docs

Give your agent the address of `llms.txt` and ask it to fetch only the sections it needs:

```text
Read {{ site_base_url }}/llms.txt, then use the Python API section files to write a script that generates a 3GPP antenna and prints its directivity.
```

The same Markdown files are in the `docs/` folder of the GitHub repository, so an agent working in a clone can read them without network access.

## Agent skill

> [!WARNING]
> **Placeholder**
> An agent skill with recipes and conventions for Quadriga-Lib is in preparation. Until it exists, the [skill file](skill/SKILL.md) is a placeholder that points to `llms.txt`.
