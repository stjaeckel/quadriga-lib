# SPDX-License-Identifier: Apache-2.0
# Part of quadriga-lib — see LICENSE for terms.
#
# Per-case comparison of two `./build_linux/test_bin --durations yes` runs: the printed table rows that differ per test
# case (the block of output that ends with the case's duration line; failure messages are not counted as rows) and the
# failed assertions per case. Cases are matched by name, so a renamed case shows as new.
# usage: python3 tools/acoustic/cases.py <base run> <new run> [rows to show per case]
import difflib, re, sys
DUR = re.compile(r'^\d+\.\d+ s: (.*)$')
def blocks(fn):
    out, cur, order = {}, [], []
    for l in open(fn, errors='replace'):
        l = l.rstrip('\n')
        m = DUR.match(l)
        if m:
            out[m.group(1)] = cur; order.append(m.group(1)); cur = []
        elif not l.startswith('Randomness seeded') and not l.startswith('Filters:'):
            cur.append(l)
    return out, order
def fails(fn):
    f, cur, L = {}, None, open(fn, errors='replace').read().split('\n')
    i = 0
    while i < len(L):
        if L[i] == '-' * 79:
            j = i + 1
            while j < len(L) and L[j] != '-' * 79 and j - i < 6: j += 1
            if j < len(L) and L[j] == '-' * 79 and j > i + 1:
                cur = ' '.join(x.strip() for x in L[i + 1:j])
                i = j + 1
                continue
        if ': FAILED:' in L[i] and cur:
            f[cur] = f.get(cur, 0) + 1
        i += 1
    return f
def norm(n): return n.replace(' ', '')
if __name__ == '__main__':
    A, oa = blocks(sys.argv[1]); B, ob = blocks(sys.argv[2])
    show = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    fa, fb = fails(sys.argv[1]), fails(sys.argv[2])
    tot = 0
    for n in ob:
        a, b = A.get(n, []), B.get(n, [])
        d = [x for x in difflib.unified_diff(a, b, lineterm='', n=0) if x[:1] in '+-' and not x.startswith(('+++', '---'))]
        isrow = lambda x: ',' in x and not x[1:].startswith((' ', '/', '-', '.', '~')) and 'FAILED' not in x
        nd = sum(1 for x in d if x.startswith('+') and isrow(x))
        nf = sum(v for k, v in fb.items() if norm(k) == norm(n))
        if nd or nf:
            tot += nd
            print('%5d lines  %4d failed  %s' % (nd, nf, n))
            for x in d[:show]: print('      ' + x[:220])
    print('total %d differing lines; failed assertions %d (base %d)' % (tot, sum(fb.values()), sum(fa.values())))
