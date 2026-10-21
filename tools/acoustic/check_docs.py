#!/usr/bin/env python3
"""
Mechanical checks of the acoustic documents (md/acoustic.md, md/wave.md), run from the repository root:

  python3 tools/acoustic/check_docs.py

1. Tables: every row of a markdown table has the column count of its header row (an unescaped `|` inside a cell, e.g.
   in `|R|` or `4|16|4`, splits the cell; write `\\|`).
2. Source keys: every [Key_YYYY_Name] cited in md/acoustic.md resolves in the literature review's references/index.csv
   (path from $QD_REVIEW_DIR, default /sjc/accoustic_research).
3. Test cases: every TEST_CASE of tests/catch2_tests/test_acoustic_*.cpp is named in md/acoustic.md (whitespace
   normalized, `\\|` and `\\*` unescaped).
4. Counts: the "All tests passed (N assertions in M test cases)" lines quoted in md/acoustic.md and md/wave.md
   against the last line of the baseline files test_wave.txt, test_acoustic.txt, test.txt (if present; regenerate
   them first, md/wave.md section 0).
5. Spelling: British spellings that the documents avoid (American English); italic test-case names are skipped.

Exit code 0 when every check passes, 1 otherwise. It reports; it changes nothing.
"""
import csv, glob, os, re, sys

ROOT = os.getcwd()
DOCS = ['md/acoustic.md', 'md/wave.md']
REVIEW = os.environ.get('QD_REVIEW_DIR', '/sjc/accoustic_research')
BRITISH = ['neighbour', 'centre', 'behaviour', 'modelled', 'modelling', 'colour', 'millimetre', 'metre ', 'favour',
           'labelled', 'travelled', 'grey ']

def read(p):
    with open(os.path.join(ROOT, p), encoding='utf-8') as f:
        return f.read()

def check_tables(fn, text):
    bad, hdr = [], None
    for i, line in enumerate(text.split('\n'), 1):
        if line.lstrip().startswith('|'):
            n = len(re.findall(r'(?<!\\)\|', line))
            if hdr is None:
                hdr = n
            elif n != hdr:
                bad.append(i)
        else:
            hdr = None
    return bad

def cited_keys(text):
    keys = set()
    for b in re.findall(r'\[([^\]]+)\]', text):
        for k in re.split(r';\s*', b):
            k = k.split(',')[0].strip()
            if re.match(r'^[A-Z][A-Za-z]+_[A-Za-z0-9]+_', k):
                keys.add(k)
    return keys

def main():
    ok = True
    texts = {fn: read(fn) for fn in DOCS if os.path.exists(os.path.join(ROOT, fn))}
    if 'md/acoustic.md' not in texts:
        print('run from the repository root (md/acoustic.md not found)'); return 1
    # 1. tables
    for fn, t in texts.items():
        bad = check_tables(fn, t)
        print('tables     %-16s %s' % (fn, 'ok' if not bad else 'rows off at lines %s' % bad))
        ok &= not bad
    # 2. source keys
    idx = os.path.join(REVIEW, 'references', 'index.csv')
    if os.path.exists(idx):
        with open(idx, encoding='utf-8') as f:
            index = {r['key'] for r in csv.DictReader(f)}
        keys = cited_keys(texts['md/acoustic.md'])
        unknown = sorted(keys - index)
        print('keys       %d cited, %s' % (len(keys), 'all resolve' if not unknown else 'UNRESOLVED: %s' % unknown))
        ok &= not unknown
    else:
        print('keys       skipped (%s not found; set QD_REVIEW_DIR)' % idx)
    # 3. test cases
    src = ''.join(read(p) for p in sorted(glob.glob(os.path.join(ROOT, 'tests/catch2_tests/test_acoustic_*.cpp'))))
    names = re.findall(r'TEST_CASE\("([^"]*)"', src)
    flat = re.sub(r'\s+', ' ', texts['md/acoustic.md']).replace('\\|', '|').replace('\\*', '*')
    missing = [n for n in names if n not in flat]
    print('cases      %d [acoustic] cases, %s' % (len(names), 'all named in md/acoustic.md' if not missing else 'MISSING: %s' % missing))
    ok &= not missing
    # 4. counts
    base = {}
    for b in ('test_wave.txt', 'test_acoustic.txt', 'test.txt'):
        p = os.path.join(ROOT, b)
        if os.path.exists(p):
            with open(p, encoding='utf-8', errors='replace') as f:
                m = re.findall(r'All tests passed \((\d+) assertions in (\d+) test cases\)', f.read())
            if m:
                base[b] = m[-1]
    for fn, t in texts.items():
        quoted = re.findall(r'(test_wave\.txt|test_acoustic\.txt|test\.txt)[^\n]*?All tests passed \((\d+) assertions in (\d+) test cases\)', t)
        for b, a, c in quoted:
            if b in base:
                same = base[b] == (a, c)
                print('counts     %-16s %-18s quoted %s / %s, baseline %s / %s %s' % (fn, b, a, c, base[b][0], base[b][1], 'ok' if same else 'DIFFER'))
                ok &= same
    if not base:
        print('counts     skipped (no baseline files in the repository root)')
    # 5. spelling
    for fn, t in texts.items():
        plain = re.sub(r'\*[^*]+\*', '', t)            # italic test-case names are quoted as written
        hits = sorted({w.strip() for w in BRITISH if re.search(r'\b' + re.escape(w.strip()), plain, re.I)})
        print('spelling   %-16s %s' % (fn, 'ok' if not hits else 'British spellings: %s' % hits))
        ok &= not hits
    print('RESULT     %s' % ('all checks pass' if ok else 'checks failed'))
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
