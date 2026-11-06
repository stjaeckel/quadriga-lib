# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Clear notebooks in place: python -m qnb.clear NOTEBOOK [NOTEBOOK ...]

Removes what a run adds and keeps what the author wrote:
- code cells: outputs and execution count; cell metadata written by Jupyter during a run ("execution",
  "ExecuteTime", "scrolled", "collapsed"); all other cell metadata stays, e.g. "quadriga_tutorial" (alt texts)
- notebook metadata: "language_info" is reduced to its "name" (the rest records the Python version of the
  kernel that last ran it), "widgets" is removed; all other metadata stays, e.g. "kernelspec" and
  "quadriga_tutorial"
- cell ids, sources and Markdown cells stay unchanged

A file is only rewritten if its content changes. Standard library only, so it runs without Jupyter."""

import json
import sys

from . import dumps

RUN_CELL_KEYS = ("execution", "ExecuteTime", "scrolled", "collapsed")


def clear(nb):
    """Clear a notebook dict in place and return it."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        meta = cell.get("metadata", {})
        for key in RUN_CELL_KEYS:
            meta.pop(key, None)
    meta = nb.get("metadata", {})
    if "language_info" in meta:
        name = meta["language_info"].get("name")
        meta["language_info"] = {"name": name} if name else {}
    meta.pop("widgets", None)
    return nb


def main(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            old = f.read()
        new = dumps(clear(json.loads(old)))
        if new != old:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(new)
            print("cleared %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
