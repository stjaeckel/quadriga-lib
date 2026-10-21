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
6. The material-model page html_docu/quadriga_lib_material_model.md: tables and spelling as above; every column that
   MaterialCols reads (resolve("...") in src/ray_mesh_interact.cpp) is a row of the page's column table; no internal
   labels (item ids, "round 1", md/ files, briefs, rigs, scratch paths) outside math and code; `$` balanced per
   paragraph and `$$` per file; no \newcommand; every in-page anchor resolves to a heading.

Exit code 0 when every check passes, 1 otherwise. It reports; it changes nothing.
"""
import csv, glob, os, re, sys

ROOT = os.getcwd()
DOCS = ['md/acoustic.md', 'md/wave.md']
PAGE = 'html_docu/quadriga_lib_material_model.md'
SOURCE = 'src/ray_mesh_interact.cpp'
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

def strip_math_code(text):
    """the text without $$ blocks, $`...`$ and $...$ math, code fences and code spans"""
    t = re.sub(r'```.*?```', '', text, flags=re.S)
    t = re.sub(r'\$\$.*?\$\$', '', t, flags=re.S)
    t = re.sub(r'\$`.*?`\$', '', t)
    t = re.sub(r'`[^`\n]*`', '', t)
    return re.sub(r'(?<!\\)\$[^$\n]*\$', '', t)

def check_math(text):
    """paragraphs (blank-line separated, code fences skipped) whose unescaped $ outside $$ is odd; $$ count of the file"""
    t = re.sub(r'```.*?```', '', text, flags=re.S)
    bad = []
    for i, para in enumerate(re.split(r'\n\s*\n', t)):
        q = re.sub(r'\$`.*?`\$', '', para.replace('$$', ''))
        q = re.sub(r'`[^`\n]*`', '', q)
        if len(re.findall(r'(?<!\\)\$', q)) % 2:
            bad.append(para.strip().split('\n')[0][:60])
    return bad, t.count('$$') % 2 == 0, '\\newcommand' not in text

def slug(h):
    h = re.sub(r'`', '', h.strip().lower())
    h = re.sub(r'[^\w\- ]', '', h)
    return h.replace(' ', '-')

def check_page(text):
    """(missing columns, label hits, bad math paragraphs, $$ ok, no newcommand, unresolved anchors)"""
    cols = re.findall(r'resolve\("([A-Za-z]+)"\)', read(SOURCE))
    rows, in_tab = set(), False
    for line in text.split('\n'):
        if line.startswith('| column |'):
            in_tab = True
        elif in_tab and line.startswith('|'):
            m = re.match(r'\|\s*`([A-Za-z]+)`\s*\|', line)
            if m:
                rows.add(m.group(1))
        else:
            in_tab = False
    missing = [c for c in cols if c not in rows]
    plain = strip_math_code(text)
    labels = []
    for pat in (r'\b[AMNPDW]\d+[a-z\u2032]?\b', r'\bD2-b\b', r'\bround 1\b', r'\bmd/', r'_report\.md', r'_brief\.md',
                r'\bbriefs?\b', r'\brigs?\b', r'build_linux/scratch'):
        labels += re.findall(pat, plain)
    bad_math, dd_ok, nc_ok = check_math(text)
    heads = {slug(m) for m in re.findall(r'^#+\s+(.*)$', text, flags=re.M)}
    anchors = [a for a in re.findall(r'\]\(#([^)]+)\)', text) if a not in heads]
    return missing, labels, bad_math, dd_ok, nc_ok, anchors, len(cols)

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
    # 6. the material-model page
    if os.path.exists(os.path.join(ROOT, PAGE)):
        page = read(PAGE)
        bad = check_tables(PAGE, page)
        print('tables     %-16s %s' % ('page', 'ok' if not bad else 'rows off at lines %s' % bad))
        plain = re.sub(r'\*[^*]+\*', '', page)
        hits = sorted({w.strip() for w in BRITISH if re.search(r'\b' + re.escape(w.strip()), plain, re.I)})
        print('spelling   %-16s %s' % ('page', 'ok' if not hits else 'British spellings: %s' % hits))
        missing, labels, bad_math, dd_ok, nc_ok, anchors, ncol = check_page(page)
        print('columns    %-16s %s' % ('page', '%d of %d in the column table' % (ncol - len(missing), ncol) + ('' if not missing else ', MISSING: %s' % missing)))
        print('labels     %-16s %s' % ('page', 'none' if not labels else 'FOUND: %s' % sorted(set(labels))))
        print('math       %-16s %s' % ('page', 'ok' if not bad_math and dd_ok and nc_ok else 'UNBALANCED: %s%s%s' % (bad_math, '' if dd_ok else ' $$ odd', '' if nc_ok else ' newcommand')))
        print('anchors    %-16s %s' % ('page', 'all resolve' if not anchors else 'UNRESOLVED: %s' % anchors))
        ok &= not bad and not hits and not missing and not labels and not bad_math and dd_ok and nc_ok and not anchors
    else:
        print('page       skipped (%s not found)' % PAGE)
    print('RESULT     %s' % ('all checks pass' if ok else 'checks failed'))
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
