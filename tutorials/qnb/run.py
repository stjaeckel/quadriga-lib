# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Run a notebook into an executed copy: python -m qnb.run [--kernel NAME] NOTEBOOK EXECUTED

- Executes all cells in a Jupyter kernel (nbclient) of the kernel named in the notebook, normally "python3" of
  the active environment; the working directory of the kernel is the folder of NOTEBOOK
- Records no execution timestamps, so two runs with the same environment give byte-identical files
- Writes to a temporary file next to EXECUTED and renames it on success; a failing cell stops with its error
  and leaves neither a partial nor an old copy. The Octave kernel reports a failing cell only as printed text
  ("error: ..."), and the following cells still run; such a cell fails the run as well
- The kernel must be installed in the environment: "python3" comes with Jupyter, "octave" with octave_kernel
- --kernel runs the notebook with another kernel than the one named in it, and the executed copy names that
  kernel; used on Windows, where the MATLAB variants run with the MATLAB kernel (jupyter-matlab-proxy)
  instead of Octave. The MATLAB kernel, too, reports errors only as text (on stderr, without "Warning:")
- NOTEBOOK itself is not changed

Needs nbclient, nbformat and ipykernel (all part of the "jupyter" package)."""

import os
import re
import sys

TIMEOUT = 600  # seconds per cell
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
KERNEL_PACKAGES = {"python3": "jupyter", "octave": "octave_kernel",  # hint for a kernel that is not installed
                   "jupyter_matlab_kernel": "jupyter-matlab-proxy"}


def _text(value):
    return "".join(value) if isinstance(value, list) else (value or "")


def kernel_errors(nb):
    """[(cell number, message)] of the errors that a kernel printed as text instead of an error output: the Octave
    kernel prints lines starting with "error: " (on stdout); the MATLAB kernel prints the error message on stderr,
    as it does with warnings, which start with "Warning:". The message is the first line of the error."""
    language = nb.get("metadata", {}).get("kernelspec", {}).get("language")
    out = []
    for n, c in enumerate(nb.get("cells", []), 1):
        for o in c.get("outputs", []):
            if o.get("output_type") != "stream":
                continue
            lines = [ln.strip() for ln in ANSI_RE.sub("", _text(o.get("text"))).splitlines() if ln.strip()]
            if language == "octave":
                out += [(n, ln) for ln in lines if ln.startswith("error: ") and ln != "error: called from"]
            elif language == "matlab" and o.get("name") == "stderr" and lines and not lines[0].startswith("Warning:"):
                out.append((n, lines[0]))
    return out


def _hint(kernel):
    if kernel in KERNEL_PACKAGES:
        return "\ninstall the package %s in the environment that runs the tutorials" % KERNEL_PACKAGES[kernel]
    return ""


def main(argv):
    kernel = None
    if len(argv) == 4 and argv[0] == "--kernel":
        kernel, argv = argv[1], argv[2:]
    if len(argv) != 2:
        print("usage: python -m qnb.run [--kernel NAME] NOTEBOOK EXECUTED", file=sys.stderr)
        return 2
    src, dst = argv
    try:
        import nbclient
        import nbformat
    except ImportError as e:
        print("qnb.run: %s; install jupyter in the environment that runs the tutorials" % e, file=sys.stderr)
        return 2

    nb = nbformat.read(src, as_version=4)
    if kernel:
        try:
            from jupyter_client.kernelspec import get_kernel_spec
            spec = get_kernel_spec(kernel)
        except Exception as e:  # NoSuchKernel
            print("qnb.run: %s failed:\n%s%s" % (src, e, _hint(kernel)), file=sys.stderr)
            return 1
        nb.metadata["kernelspec"] = {"name": kernel, "display_name": spec.display_name, "language": spec.language}
    folder = os.path.dirname(os.path.abspath(src))
    # Default TCP transport on 127.0.0.1 with a free port per kernel: safe for parallel runs (make -j) and
    # available on Windows. The kernel's warning about unencrypted TCP refers to this local connection.
    client = nbclient.NotebookClient(nb, timeout=TIMEOUT, startup_timeout=300, record_timing=False,
                                     resources={"metadata": {"path": folder}})  # MATLAB starts in ~30 s
    tmp = dst + ".tmp"
    if os.path.exists(dst):
        os.remove(dst)  # no old copy survives a failed run
    try:
        client.execute()
    except Exception as e:  # nbclient.exceptions.CellExecutionError and kernel errors
        hint = _hint(nb.metadata.get("kernelspec", {}).get("name", "")) if type(e).__name__ == "NoSuchKernel" else ""
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
