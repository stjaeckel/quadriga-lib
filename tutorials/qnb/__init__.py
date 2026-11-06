# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Helpers of the tutorial build (tutorials/GNUmakefile): run a notebook into an executed copy, clear notebooks.

Called as modules from the tutorials folder, e.g. `python -m qnb.run sos_01_random_fields.ipynb
executed/sos_01_random_fields.ipynb`, so that make on Linux and nmake on Windows can use the same commands."""

import json

# Notebook JSON as Jupyter writes it (nbformat): one-space indent, sorted keys, UTF-8, final newline.
# Writing with the same settings keeps a cleared notebook byte-identical to one saved by Jupyter.
JSON_OPTIONS = dict(indent=1, sort_keys=True, ensure_ascii=False)


def dumps(nb):
    """Notebook dict as JSON text in the format of Jupyter."""
    return json.dumps(nb, **JSON_OPTIONS) + "\n"
