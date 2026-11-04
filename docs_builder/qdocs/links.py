# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Ids, slugs, old hash ids, cross-language matching, [[name]] resolution and URLs per output location."""

import posixpath
import re

from . import config

# An absolute URL: it starts with a scheme such as http: or mailto:
ABSOLUTE_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_SITE_RE = re.compile(r"^https?://(?:www\.)?quadriga-lib\.org(?=/|#|$)/?")

API_PAGES = {"%s_api" % lang: lang for lang in config.LANGS}
FULL_FILES = {config.TWINS["%s_api" % lang] for lang in config.LANGS}

# lang -> {anchor: (section slug, True for a section id)}; filled by set_index()
_API = {lang: {} for lang in config.LANGS}
# anchor on formats.html -> (format file stem, True for the section id of the format); filled by set_formats()
_FORMATS = {}


def slug(text):
    """Lower case; every run of characters outside [a-z0-9] becomes one '-'; no '-' at the ends."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _norm(name):
    return name.lower().replace(".", "_")


def _toml_line(text, key):
    for n, ln in enumerate(text.split("\n"), 1):
        if re.match(r'\s*"?%s"?\s*=' % re.escape(key), ln):
            return n
    return None


def match_families(api, toml_path, diag, root):
    """Set Function.counterparts of every function from the file stems, the names and api_matching.toml."""
    rel = toml_path.relative_to(root).as_posix()
    text = toml_path.read_text(encoding="utf-8")
    try:
        data = config.tomllib.loads(text)
    except config.tomllib.TOMLDecodeError as e:
        diag.error(rel, None, "TOML: %s" % e)
        return
    families = data.get("families", {})
    if set(data) - {"families"} or not isinstance(families, dict) or not all(
            isinstance(v, list) and all(isinstance(n, str) for n in v) for v in families.values()):
        diag.error(rel, None, "expected only a [families] table of stem = [C++ names]")
        return

    def by_stem(lang):
        out = {}
        for sec in api[lang].sections:
            for f in sec.functions:
                out.setdefault(f.stem, f)
        return out

    mex, py = by_stem("mex"), by_stem("python")
    cpp = {f.name: f for sec in api["cpp"].sections for f in sec.functions}
    cpp_norm = {}
    for name in cpp:
        cpp_norm.setdefault(_norm(name), name)
    for stem in sorted(set(families) - set(mex) - set(py)):
        diag.error(rel, _toml_line(text, stem), "unknown stem '%s' (no api_mex/%s.cpp or api_python/qpy_%s.cpp)"
                   % (stem, stem, stem))
    owner = {}
    for stem in sorted(set(mex) | set(py)):
        fam = {lang: f for lang, f in (("mex", mex.get(stem)), ("python", py.get(stem))) if f}
        if stem in families:
            names = []
            for n in families[stem]:
                if n in cpp:
                    names.append(n)
                else:
                    diag.error(rel, _toml_line(text, stem), "unknown C++ function '%s' in family '%s'" % (n, stem))
        else:
            names = [cpp_norm[_norm(f.name)] for f in fam.values() if _norm(f.name) in cpp_norm][:1]
        for n in names:
            if n in owner:
                diag.error(rel, _toml_line(text, stem), "C++ function '%s' is in two families: '%s' and '%s'"
                           % (n, owner[n], stem))
            owner[n] = stem
        for lang, f in fam.items():
            f.counterparts = {k: g.name for k, g in fam.items() if k != lang}
            if names:
                f.counterparts["cpp"] = names[0]
        for n in names:
            cpp[n].counterparts = {k: g.name for k, g in fam.items()}


def set_index(api, diag):
    """Index the anchors of each API page for url(); two equal anchors on one page are E-anchor."""
    for lang, page in api.items():
        index = _API[lang] = {}

        def add(anchor, value, path, line):
            if anchor in index:
                diag.error(path, line, "E-anchor: id '%s' appears twice on the %s API page" % (anchor, lang))
            index[anchor] = value

        intro = page.intro
        add("sec-general-usage-notes", ("general-usage-notes", True), intro.path, None)
        for h in intro.headings:
            if h.id:
                add(h.id, ("general-usage-notes", False), intro.path, intro.body_line + h.index)
        for sec in page.sections:
            add("sec-" + sec.slug, (sec.slug, True), sec.path, sec.line)
            for f in sec.functions:
                add(f.name, (sec.slug, False), f.path, f.line)


def set_formats(formats):
    """Index the anchors of formats.html for url(): each format's section id and its heading ids."""
    _FORMATS.clear()
    for f in formats:
        _FORMATS[f.page_id] = (f.page_id, True)
        for h in f.headings:
            if h.id:
                _FORMATS[h.id] = (f.page_id, False)


def resolve(lang, name, member):
    """Anchor of [[name]] (member: .[[name]]) on the API page of lang, or None."""
    anchor = ("." if member else "") + name
    if lang == "mex":
        anchor = anchor.lower()
    entry = _API[lang].get(anchor)
    return anchor if entry and not entry[1] else None


def site_path(target):
    """Make an absolute quadriga-lib.org URL site-relative; download/ and releases/ stay absolute."""
    m = _SITE_RE.match(target)
    if not m:
        return target
    rest = target[m.end():]
    if rest.startswith(("download/", "releases/")):
        return target
    return "index.html" + rest if rest == "" or rest.startswith("#") else rest


def is_external(target):
    return bool(ABSOLUTE_URL_RE.match(site_path(target)))


def website_only(path):
    """True for the twins that exist only in website/ (the tutorial index and tutorials)."""
    return path == config.WEBSITE_ONLY[0] or path.startswith(config.WEBSITE_ONLY[1])


def in_docs(path):
    """True for a file that docs/ holds (and the website copies at the same path)."""
    return path.startswith(("api/", "formats/")) or path == "llms.txt" or (
        path in config.TWINS.values() and not website_only(path))


def _relative(path, frag, location):
    if path == location:
        return frag or posixpath.basename(path)
    return posixpath.relpath(path, posixpath.dirname(location) or ".") + frag


def url(target, location):
    """URL of a link target for a file at an output location.

    target: a URL as written in a source: relative to the site root (for example
    'cpp_api.html#x', 'faq.html#cite', 'publications/x.pdf', 'quadriga_lib_api_cpp.md'),
    a same-page anchor '#x', or absolute ('https://...', also quadriga-lib.org URLs).
    location: path of the output file relative to the site root, with '/': an HTML page
    ('cpp_api.html', 'tutorials/x.html', '404.html'), a twin ('faq.md', 'api/cpp/x.md',
    'tutorials.md', 'tutorials/x.md'), 'README.md' for docs/README.md, or 'llms.txt'.
    """
    t = site_path(target)
    if ABSOLUTE_URL_RE.match(t) or t.startswith("#"):
        return t
    path, hash_, frag = t.lstrip("/").partition("#")
    frag = hash_ + frag
    if location.endswith(".html"):
        if location == "404.html":  # served for any missing URL, so its links are root-absolute
            return "/" + path + frag
        return _relative(path, frag, location)

    must_absolute = False
    if path.endswith(".html"):
        pid = path[:-5]
        if pid in API_PAGES:
            lang, full = API_PAGES[pid], config.TWINS[pid]
            entry = _API[lang].get(frag[1:]) if frag else None
            if location in FULL_FILES or entry is None:
                path = full
            else:
                path = "api/%s/%s.md" % (lang, entry[0])
                frag = "" if entry[1] else frag
        elif pid == "formats" and frag[1:] in _FORMATS:
            stem, section = _FORMATS[frag[1:]]
            path, frag = "formats/%s.md" % stem, "" if section else frag
        elif pid in config.TWINS:
            path = config.TWINS[pid]
        elif website_only(path):
            path = pid + ".md"  # a tutorial page and its twin sit side by side
        else:
            must_absolute = True  # a page without twin
    if location == "llms.txt" or must_absolute or (not website_only(location) and not in_docs(path)):
        return "%s/%s%s" % (config.SITE_BASE_URL, path, frag)
    return _relative(path, frag, location)
