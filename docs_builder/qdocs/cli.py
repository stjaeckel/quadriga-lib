# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""The docs command: build docs/ in memory, then write it or compare it with the files on disk."""

import argparse
import sys
from pathlib import Path

from . import checks, config, docblocks, links, prose, render_md


class Diagnostics:
    """Collects errors, warnings and gaps; paths are relative to the repository root, with '/'."""

    def __init__(self):
        self.lines = []  # errors and warnings in the order they were found
        self.errors = 0
        self.warnings = 0
        self.gaps = []  # (path, line, text)

    @staticmethod
    def _where(path, line):
        return "%s:%d" % (path, line) if line else str(path)

    def error(self, path, line, message):
        self.errors += 1
        self.lines.append("%s: error: %s" % (self._where(path, line), message))

    def warning(self, path, line, message, kind):
        self.warnings += 1
        self.lines.append("%s: warning: %s [%s]" % (self._where(path, line), message, kind))

    def gap(self, path, line, text):
        self.gaps.append((path, line, text))

    def report(self, command, out=None):
        """Print all diagnostics, the sorted gap list and the summary line; return the exit code."""
        out = out or sys.stdout
        for line in self.lines:
            print(line, file=out)
        gaps = sorted(set(self.gaps), key=lambda g: (g[0], g[1] or 0, g[2]))
        for path, line, text in gaps:
            print("%s: %s" % (self._where(path, line), text), file=out)
        print("%s: %d errors, %d warnings, %d gaps" % (command, self.errors, self.warnings, len(gaps)), file=out)
        return 1 if self.errors else 0


def _same(data, path):
    """True if the file at path holds data; text files are compared as read in text mode."""
    old = path.read_bytes()
    if old == data:
        return True
    try:
        old_text = old.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        return old_text == data.decode("utf-8")
    except UnicodeDecodeError:
        return False


def compare_tree(folder, mapping):
    """Return (differing, missing, extra) relative paths of folder against mapping, sorted."""
    on_disk = set()
    if folder.is_dir():
        on_disk = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    differing = sorted(p for p in mapping if p in on_disk and not _same(mapping[p], folder / p))
    missing = sorted(p for p in mapping if p not in on_disk)
    extra = sorted(on_disk - set(mapping))
    return differing, missing, extra


def write_tree(folder, mapping):
    """Write mapping (relative path -> bytes) into folder: changed files only, delete the rest, drop empty folders."""
    differing, missing, extra = compare_tree(folder, mapping)
    for rel in differing + missing:
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(mapping[rel])
    for rel in extra:
        (folder / rel).unlink()
    for path in sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    return differing + missing, extra


def build_docs(root, diag):
    """Run the whole docs phase in memory.

    root: repository root (Path). diag: a Diagnostics. Returns (model, mapping):
    model is a dict with
      "api":          {lang: docblocks.ApiPage}, intro included (lang in config.LANGS),
      "pages":        {page_id: prose.Page} for config.TEXT_PAGES,
      "publications": prose.Publications or None,
    and mapping is {path relative to docs/: bytes}. The mapping is empty if a structural
    error stopped the build before rendering; the caller checks diag.errors.
    Also sets the link index of links.py and adds the gaps of docs_builder/content/ to diag.
    """
    root = Path(root)
    config.read_versions(root, diag)
    content = root / config.DOCS_CONTENT
    needed = ["%s.md" % p for p in config.TEXT_PAGES]
    needed += ["api_intro_%s.md" % lang for lang in config.LANGS]
    needed += ["publications.toml", "api_matching.toml"]
    for name in needed:
        if not (content / name).is_file():
            diag.error("%s/%s" % (config.DOCS_CONTENT, name), None, "content file missing")
    model = {"api": {}, "pages": {}, "publications": None}
    if diag.errors:
        return model, {}

    for lang in config.LANGS:
        intro = prose.read_page(content / ("api_intro_%s.md" % lang), diag, root)
        model["api"][lang] = docblocks.read_api(root, lang, intro, diag)
    for pid in config.TEXT_PAGES:
        model["pages"][pid] = prose.read_page(content / (pid + ".md"), diag, root)
    model["publications"] = prose.read_publications(content / "publications.toml", diag, root)
    links.match_families(model["api"], content / "api_matching.toml", diag, root)
    links.set_index(model["api"], diag)
    render_md.check_parts(model, diag)
    diag.gaps.extend(prose.gap_scan(root, [config.DOCS_CONTENT], diag))
    if diag.errors:
        return model, {}

    mapping = render_md.render_docs(model, diag)
    checks.check_markdown(mapping, "docs", diag)
    return model, mapping


def main(argv=None):
    parser = argparse.ArgumentParser(prog="build.py", description="Build the tracked Markdown in docs/.")
    parser.add_argument("command", choices=["docs"])
    parser.add_argument("--check", action="store_true", help="compare with docs/ and write nothing")
    parser.add_argument("--root", help="repository root (default: the parent of docs_builder/)")
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code
    root = Path(args.root) if args.root else Path(__file__).resolve().parents[2]
    if not root.is_dir():
        print("build.py: --root %s is not a folder" % args.root)
        return 2

    diag = Diagnostics()
    _, mapping = build_docs(root, diag)
    command = "docs --check" if args.check else "docs"
    if not diag.errors:
        docs = root / config.DOCS_DIR
        if args.check:
            differing, missing, extra = compare_tree(docs, mapping)
            for rel in differing:
                diag.error("docs/" + rel, None, "differs from the build output (run make docs)")
            for rel in missing:
                diag.error("docs/" + rel, None, "missing, the build writes it (run make docs)")
            for rel in extra:
                diag.error("docs/" + rel, None, "extra, the build does not write it (run make docs)")
        else:
            written, deleted = write_tree(docs, mapping)
            for rel in written:
                print("wrote docs/" + rel)
            for rel in deleted:
                print("deleted docs/" + rel)
    return diag.report(command)
