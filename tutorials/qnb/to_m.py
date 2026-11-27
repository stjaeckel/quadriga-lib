# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Write the code cells of a notebook as a MATLAB script: python -m qnb.to_m NOTEBOOK SCRIPT

Used by "make check-matlab", which runs the MATLAB variants of the tutorials in MATLAB without keeping any
output: the published outputs come from Octave, and this check shows that the same code runs in MATLAB.
- The script starts by hiding figures; each code cell follows under a line "%% cell <n>" (n counts all cells)
- Standard library only"""

import json
import sys


def main(argv):
    if len(argv) != 2:
        print("usage: python -m qnb.to_m NOTEBOOK SCRIPT", file=sys.stderr)
        return 2
    src, dst = argv
    with open(src, encoding="utf-8") as f:
        nb = json.load(f)
    blocks = ["set(0, 'DefaultFigureVisible', 'off');"]
    for n, c in enumerate(nb.get("cells", []), 1):
        if c.get("cell_type") == "code":
            source = c.get("source")
            blocks.append("%%%% cell %d\n%s" % (n, "".join(source) if isinstance(source, list) else source))
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
