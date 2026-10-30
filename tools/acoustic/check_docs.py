#!/usr/bin/env python3
"""
Mechanical checks of the material-model page and of the material part of the Data Formats page, run from the
repository root:

  python3 tools/acoustic/check_docs.py

1. The material-model page html_docu/quadriga_lib_material_model.md: every row of a markdown table has the column
   count of its header row (an unescaped `|` inside a cell, e.g. in `|R|` or `4|16|4`, splits the cell; write `\\|`);
   no British spellings (American English; italic test-case names are skipped); every column that MaterialCols reads
   (resolve("...") in src/ray_mesh_interact.cpp) is a row of the page's column table; no internal labels (item ids,
   "round 1", md/ files, briefs, rigs, scratch paths) outside math and code; `$` balanced per paragraph and `$$` per
   file; no \newcommand; every in-page anchor resolves to a heading.
2. The material part of the Data Formats page, html_builder/formats/format_materials.md: tables and spelling as
   above; every column that MaterialCols reads is a row of its column table (header "| Key |"); no internal labels.

Exit code 0 when every check passes, 1 otherwise. It reports; it changes nothing.
"""
import os, re, sys

ROOT = os.getcwd()
PAGE = 'html_docu/quadriga_lib_material_model.md'
FORMATS = 'html_builder/formats/format_materials.md'
SOURCE = 'src/ray_mesh_interact.cpp'
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

def missing_columns(text, header):
    """(columns MaterialCols reads that are no row of the table whose header line starts with `header`, count)"""
    cols = re.findall(r'resolve\("([A-Za-z]+)"\)', read(SOURCE))
    rows, in_tab = set(), False
    for line in text.split('\n'):
        if re.match(header, line):
            in_tab = True
        elif in_tab and line.startswith('|'):
            m = re.match(r'\|\s*`?([A-Za-z]+)`?\s*\|', line)
            if m:
                rows.add(m.group(1))
        else:
            in_tab = False
    return [c for c in cols if c not in rows], len(cols)

def find_labels(text):
    """internal labels outside math and code"""
    plain = strip_math_code(text)
    labels = []
    for pat in (r'\b[AMNPDW]\d+[a-z\u2032]?\b', r'\bD2-b\b', r'\bround 1\b', r'\bmd/', r'_report\.md', r'_brief\.md',
                r'\bbriefs?\b', r'\brigs?\b', r'build_linux/scratch'):
        labels += re.findall(pat, plain)
    return labels

def check_page(text):
    """(missing columns, label hits, bad math paragraphs, $$ ok, no newcommand, unresolved anchors)"""
    missing, ncol = missing_columns(text, r'\| column \|')
    labels = find_labels(text)
    bad_math, dd_ok, nc_ok = check_math(text)
    heads = {slug(m) for m in re.findall(r'^#+\s+(.*)$', text, flags=re.M)}
    anchors = [a for a in re.findall(r'\]\(#([^)]+)\)', text) if a not in heads]
    return missing, labels, bad_math, dd_ok, nc_ok, anchors, ncol

def main():
    ok = True
    if not any(os.path.exists(os.path.join(ROOT, p)) for p in (PAGE, FORMATS)):
        print('run from the repository root (%s and %s not found)' % (PAGE, FORMATS)); return 1
    # 1. the material-model page
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
    # 2. the material part of the Data Formats page
    if os.path.exists(os.path.join(ROOT, FORMATS)):
        fmt = read(FORMATS)
        bad = check_tables(FORMATS, fmt)
        print('tables     %-16s %s' % ('formats', 'ok' if not bad else 'rows off at lines %s' % bad))
        hits = sorted({w.strip() for w in BRITISH if re.search(r'\b' + re.escape(w.strip()), fmt, re.I)})
        print('spelling   %-16s %s' % ('formats', 'ok' if not hits else 'British spellings: %s' % hits))
        missing, ncol = missing_columns(fmt, r'\|\s*Key\s*\|')
        print('columns    %-16s %s' % ('formats', '%d of %d in the column table' % (ncol - len(missing), ncol) + ('' if not missing else ', MISSING: %s' % missing)))
        labels = find_labels(fmt)
        print('labels     %-16s %s' % ('formats', 'none' if not labels else 'FOUND: %s' % sorted(set(labels))))
        ok &= not bad and not hits and not missing and not labels
    else:
        print('formats    skipped (%s not found)' % FORMATS)
    print('RESULT     %s' % ('all checks pass' if ok else 'checks failed'))
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
