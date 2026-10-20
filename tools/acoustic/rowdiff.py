#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Part of quadriga-lib — see LICENSE for terms.
"""Compare two Catch2 test outputs (the printed diagnostic rows) number by number.

    python3 -I tools/acoustic/rowdiff.py BASE NEW [--src tests/catch2_tests] [--max N]

Rule: the same lines in the same order, the text between numbers identical, and every number equal to the
baseline or within one unit of its last printed digit (the unit of the baseline token: "0.1234" -> 1e-4,
"1.234e-05" -> 1e-8, "17" -> 1). Summary lines (Filters, Randomness seed, All tests passed / test cases /
assertions, the ==== rule) are set aside and printed for both runs.

Prints every differing value with its line, the table header above it (the nearest line starting with '#')
and the test case that prints that header (found by searching the test sources for the header's text).
Exit status: 0 if every number is within the rule and the structure is equal, 1 otherwise.
"""
import argparse, bisect, difflib, glob, math, os, re, sys

NUM = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)\b', re.IGNORECASE)
SUMMARY = re.compile(r'^(Filters:|Randomness seeded to:|All tests passed|test cases:|assertions:|={10,}$)')


def split(line):
    """Text skeleton (numbers replaced by \\x00) and the number tokens."""
    toks = NUM.findall(line)
    return NUM.sub('\x00', line), toks


def unit(tok):
    m = re.fullmatch(r'[-+]?(\d*)(?:\.(\d*))?(?:[eE]([-+]?\d+))?', tok)
    if not m:
        return 0.0
    dec = len(m.group(2) or '')
    exp = int(m.group(3) or 0)
    return 10.0 ** (exp - dec)


def value(tok):
    try:
        return float(tok)
    except ValueError:
        return float('nan')


class CaseIndex:
    """Finds the TEST_CASE that prints a header text, by a literal chunk of the header."""

    def __init__(self, src):
        self.files = []
        for fn in sorted(glob.glob(os.path.join(src, '*.cpp')) + glob.glob(os.path.join(src, '*.hpp'))):
            with open(fn, encoding='utf-8', errors='replace') as f:
                text = f.read()
            starts = [m.start() for m in re.finditer(r'TEST_CASE\s*\(', text)]
            names = [re.match(r'TEST_CASE\s*\(\s*"((?:[^"\\]|\\.)*)"', text[s:]) for s in starts]
            names = [n.group(1) if n else '?' for n in names]
            self.files.append((os.path.basename(fn), text, starts, names))
        self.cache = {}

    def find(self, header):
        if header in self.cache:
            return self.cache[header]
        chunks = sorted((c.strip() for c in re.split(r'[-+]?\d[\d.eE+-]*|%[-+ #0-9.]*[a-zA-Z]', header.lstrip('# '))),
                        key=len, reverse=True)
        res = '?'
        for chunk in chunks[:3]:
            if len(chunk) < 10:
                break
            hits = []
            for fn, text, starts, names in self.files:
                p = text.find(chunk)
                while p >= 0:
                    i = bisect.bisect_right(starts, p) - 1
                    line = text.count('\n', 0, p) + 1
                    hits.append(f'{names[i] if i >= 0 else "(helper)"} [{fn}:{line}]')
                    p = text.find(chunk, p + 1)
            if hits:
                res = hits[0] + (f' (+{len(hits) - 1} more)' if len(hits) > 1 else '')
                break
        self.cache[header] = res
        return res


def load(fn):
    with open(fn, encoding='utf-8', errors='replace') as f:
        lines = f.read().split('\n')
    body, summ = [], []
    for i, l in enumerate(lines):
        (summ if SUMMARY.match(l) else body).append((i + 1, l))
    return body, summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('base')
    ap.add_argument('new')
    ap.add_argument('--src', default='tests/catch2_tests', help='test sources, to name the case of a header')
    ap.add_argument('--max', type=int, default=1000, help='stop listing after this many differences')
    a = ap.parse_args()

    bb, bs = load(a.base)
    nb, ns = load(a.new)
    idx = CaseIndex(a.src) if os.path.isdir(a.src) else None

    bsk = [split(l) for _, l in bb]
    nsk = [split(l) for _, l in nb]
    # align on whole lines (a dropped or added row stays local), then pair the rows of a changed block
    # position by position when the block keeps its length and every pair keeps its text skeleton
    sm = difflib.SequenceMatcher(None, [l for _, l in bb], [l for _, l in nb], autojunk=False)

    header_at = []  # nearest header line for each base body line
    h = ''
    for _, l in bb:
        if l.startswith('#'):
            h = l
        header_at.append(h)

    n_same = n_within = 0
    beyond, struct = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            n_same += sum(len(bsk[i][1]) for i in range(i1, i2))
            continue
        if op == 'replace' and i2 - i1 == j2 - j1 and all(
                bsk[i1 + k][0] == nsk[j1 + k][0] and len(bsk[i1 + k][1]) == len(nsk[j1 + k][1])
                for k in range(i2 - i1)):
            for k in range(i2 - i1):
                bt, nt = bsk[i1 + k][1], nsk[j1 + k][1]
                for c, (x, y) in enumerate(zip(bt, nt)):
                    if x == y:
                        n_same += 1
                        continue
                    vx, vy = value(x), value(y)
                    ok = not (math.isnan(vx) or math.isnan(vy)) and abs(vx - vy) <= unit(x) * (1 + 1e-9)
                    n_within += ok
                    beyond.append((i1 + k, j1 + k, c, x, y, ok))
        else:
            struct.append((op, i1, i2, j1, j2))

    n_out = sum(1 for r in beyond if not r[5])
    print(f'base {a.base}: {len(bb)} lines, new {a.new}: {len(nb)} lines')
    print(f'numbers identical: {n_same}, within one last digit: {n_within}, beyond: {n_out}, '
          f'structural blocks: {len(struct)}')
    for s1, s2 in zip(bs, ns):
        if s1[1] != s2[1] and not s1[1].startswith(('Randomness', 'Filters')):
            print(f'  summary differs: "{s1[1]}" -> "{s2[1]}"')
    if len(bs) != len(ns):
        print(f'  summary line count differs: {len(bs)} -> {len(ns)}')

    shown = 0
    for i, j, c, x, y, ok in beyond:
        if shown >= a.max:
            print(f'... (more, --max {a.max})')
            break
        hdr = header_at[i]
        case = idx.find(hdr) if (idx and hdr) else '?'
        tag = 'within' if ok else 'BEYOND'
        print(f'{tag} base:{bb[i][0]} new:{nb[j][0]} col {c}: {x} -> {y}  (unit {unit(x):.0e})')
        print(f'    case: {case}')
        print(f'    table: {hdr[:150]}')
        print(f'    row:  {bb[i][1][:200]}')
        shown += 1
    for op, i1, i2, j1, j2 in struct[:a.max]:
        print(f'STRUCT {op} base lines {bb[i1][0] if i1 < len(bb) else "end"}..+{i2 - i1} '
              f'new lines {nb[j1][0] if j1 < len(nb) else "end"}..+{j2 - j1}')
        for _, l in bb[i1:min(i2, i1 + 5)]:
            print(f'  - {l[:200]}')
        for _, l in nb[j1:min(j2, j1 + 5)]:
            print(f'  + {l[:200]}')
    return 0 if (n_out == 0 and not struct and len(bs) == len(ns)) else 1


if __name__ == '__main__':
    sys.exit(main())
