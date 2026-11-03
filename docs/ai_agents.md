# AI agents

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/ai_agents.html

| Section | Topics |
| --- | --- |
| [Files for agents](#files-for-agents) |  |
| [Pointing an agent at the docs](#pointing-an-agent-at-the-docs) |  |
| [Agent skill](#agent-skill) |  |

<a id="files-for-agents"></a>
## Files for agents

| File | Content |
| --- | --- |
| [`llms.txt`](llms.txt) | Short summary of the library and links to all Markdown files |
| `<page>.md` | Markdown version next to every content page, linked in its footer |
| `api/cpp/<section>.md` | One API section per file, small enough to fetch on its own; also for `mex` and `python` |
| [`quadriga_lib_api_cpp.md`](quadriga_lib_api_cpp.md) | Full API per language in one file (also `_mex` and `_python`) |

<a id="pointing-an-agent-at-the-docs"></a>
## Pointing an agent at the docs

Give your agent the address of `llms.txt` and ask it to fetch only the sections it needs:

```text
Read http://quadriga-lib.org/llms.txt, then use the Python API section files to write a script that generates a 3GPP antenna and prints its directivity.
```

The same Markdown files are in the `docs/` folder of the GitHub repository, so an agent working in a clone can read them without network access.

<a id="agent-skill"></a>
## Agent skill

> [!WARNING]
> **Placeholder**
> An agent skill with recipes and conventions for Quadriga-Lib is in preparation. Until it exists, the [skill file](http://quadriga-lib.org/skill/SKILL.md) is a placeholder that points to `llms.txt`.
