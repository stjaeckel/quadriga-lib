# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Run a notebook into an executed copy: python -m qnb.run NOTEBOOK EXECUTED

- Executes all cells in a Jupyter kernel (nbclient) of the kernel named in the notebook, normally "python3" of
  the active environment; the working directory of the kernel is the folder of NOTEBOOK
- Records no execution timestamps, so two runs with the same environment give byte-identical files
- Writes to a temporary file next to EXECUTED and renames it on success; a failing cell stops with its error
  and leaves neither a partial nor an old copy
- NOTEBOOK itself is not changed

Needs nbclient, nbformat and ipykernel (all part of the "jupyter" package)."""

import os
import sys

TIMEOUT = 600  # seconds per cell


def main(argv):
    if len(argv) != 2:
        print("usage: python -m qnb.run NOTEBOOK EXECUTED", file=sys.stderr)
        return 2
    src, dst = argv
    try:
        import nbclient
        import nbformat
    except ImportError as e:
        print("qnb.run: %s; install jupyter in the environment that runs the tutorials" % e, file=sys.stderr)
        return 2

    nb = nbformat.read(src, as_version=4)
    folder = os.path.dirname(os.path.abspath(src))
    # Default TCP transport on 127.0.0.1 with a free port per kernel: safe for parallel runs (make -j) and
    # available on Windows. The kernel's warning about unencrypted TCP refers to this local connection.
    client = nbclient.NotebookClient(nb, timeout=TIMEOUT, record_timing=False,
                                     resources={"metadata": {"path": folder}})
    tmp = dst + ".tmp"
    if os.path.exists(dst):
        os.remove(dst)  # no old copy survives a failed run
    try:
        client.execute()
    except Exception as e:  # nbclient.exceptions.CellExecutionError and kernel errors
        print("qnb.run: %s failed:\n%s" % (src, e), file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    nbformat.write(nb, tmp)
    os.replace(tmp, dst)
    print("ran %s -> %s" % (src, dst))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
