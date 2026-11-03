# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""The website command: the docs phase in memory, then the site rendered, checked and written."""

import argparse
from pathlib import Path

from qdocs import cli as docs_cli
from qdocs import config, links, prose
from . import assets, checks, render_html, tutorials

# Pages whose source is website_builder/content/<id>.md: no twin in docs/ (the landing is its template)
WEBSITE_PAGES = [pid for pid, _, twin in config.PAGES if pid != "index" and (twin is None or links.website_only(twin))]


def build_site(root, diag, release=False):
    """Build docs/ and the site in memory: returns (docs mapping, site mapping); both empty after an error."""
    model, docs = docs_cli.build_docs(root, diag)
    web = {}
    for pid in WEBSITE_PAGES:
        path = root / config.WEBSITE_CONTENT / (pid + ".md")
        if path.is_file():
            web[pid] = prose.read_page(path, diag, root)
        else:
            diag.error("%s/%s.md" % (config.WEBSITE_CONTENT, pid), None, "content file missing")
    tuts = tutorials.read_all(root, diag)
    files = assets.static_files()
    files.update(assets.skill(root, diag))
    for t in tuts:
        files.update(t.files)
    diag.gaps.extend(prose.gap_scan(root, [config.WEBSITE_CONTENT], diag))
    # Named from the folder above the builder, like the other paths: website_builder/templates/<file>
    code_root = assets.TEMPLATES_DIR.parents[1]
    diag.gaps.extend(prose.gap_scan(code_root, [assets.TEMPLATES_DIR.relative_to(code_root).as_posix()], diag))
    if diag.errors:
        return {}, {}
    order = sorted(tuts, key=lambda t: (t.order, t.title))
    files["tutorials.md"] = tutorials.index_twin(web["tutorials"], order).encode("utf-8")
    model.update(docs=docs, website=web, tutorials=tuts, files=files)
    site, markers = render_html.render_site(model, diag)
    if diag.errors:
        return {}, {}
    # Generated Markdown; the hand-made static files are only checked as link targets
    generated = [p for p in site if p.endswith((".md", ".txt")) and (p in docs or p.startswith("tutorials"))]
    checks.check_site(site, diag, generated)
    if release:
        gaps = len(set(diag.gaps))
        if gaps or markers:
            diag.error("website --release", None, "%d gaps on the list and %d gap markers in the HTML; "
                       "the release build needs none" % (gaps, markers))
    if diag.errors:
        return {}, {}
    return docs, site


def main(argv=None):
    parser = argparse.ArgumentParser(prog="build.py", description="Build the website into website/.")
    parser.add_argument("command", choices=["website"])
    parser.add_argument("--release", action="store_true", help="also fail on any remaining gap")
    parser.add_argument("--root", help="repository root (default: the parent of website_builder/)")
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code
    root = Path(args.root) if args.root else Path(__file__).resolve().parents[2]
    if not root.is_dir():
        print("build.py: --root %s is not a folder" % args.root)
        return 2
    diag = docs_cli.Diagnostics()
    docs, site = build_site(root, diag, args.release)
    command = "website --release" if args.release else "website"
    if not diag.errors:
        written, deleted = docs_cli.write_tree(root / config.DOCS_DIR, docs)
        for rel in written:
            print("wrote docs/" + rel)
        for rel in deleted:
            print("deleted docs/" + rel)
        written, deleted = docs_cli.write_tree(root / "website", site)
        print("website: %d files written, %d deleted, %d unchanged" % (len(written), len(deleted),
                                                                      len(site) - len(written)))
    return diag.report(command)
