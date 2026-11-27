# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Run a notebook into an executed copy: python -m qnb.run NOTEBOOK EXECUTED

- Executes all cells in a Jupyter kernel (nbclient) of the kernel named in the notebook, normally "python3" of
  the active environment; the working directory of the kernel is the folder of NOTEBOOK
- Records no execution timestamps, so two runs with the same environment give byte-identical files
- Writes to a temporary file next to EXECUTED and renames it on success; a failing cell stops with its error
  and leaves neither a partial nor an old copy. The Octave kernel reports a failing cell only as printed text
  ("error: ..."), and the following cells still run; such a cell fails the run as well
- The kernel must be installed in the environment: "python3" comes with Jupyter, "octave" with octave_kernel
- NOTEBOOK itself is not changed

Needs nbclient, nbformat and ipykernel (all part of the "jupyter" package)."""

import os
import re
import sys

TIMEOUT = 600  # seconds per cell
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
KERNEL_PACKAGES = {"python3": "jupyter", "octave": "octave_kernel"}  # hint for a kernel that is not installed


def _text(value):
    return "".join(value) if isinstance(value, list) else (value or "")


def kernel_errors(nb):
    """[(cell number, message)] of the errors that a kernel printed as text instead of an error output, as the
    Octave kernel does (on stdout): lines starting with "error: " in the streams of MATLAB and Octave notebooks."""
    if nb.get("metadata", {}).get("kernelspec", {}).get("language") not in ("matlab", "octave"):
        return []
    out = []
    for n, c in enumerate(nb.get("cells", []), 1):
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                for line in ANSI_RE.sub("", _text(o.get("text"))).splitlines():
                    if line.startswith("error: ") and line.strip() != "error: called from":  # traceback line
                        out.append((n, line.strip()))
    return out


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
        name = nb.metadata.get("kernelspec", {}).get("name", "")
        hint = ""
        if type(e).__name__ == "NoSuchKernel" and name in KERNEL_PACKAGES:
            hint = "\ninstall the package %s in the environment that runs the tutorials" % KERNEL_PACKAGES[name]
        print("qnb.run: %s failed:\n%s%s" % (src, e, hint), file=sys.stderr)
        return 1
    errors = kernel_errors(nb)
    if errors:
        print("qnb.run: %s failed:\n%s" % (src, "\n".join("cell %d: %s" % e for e in errors)), file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    nbformat.write(nb, tmp)
    os.replace(tmp, dst)
    print("ran %s -> %s" % (src, dst))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
