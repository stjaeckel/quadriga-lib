#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

"""Update the pybind11 headers bundled in external/ to the latest official release.

Usage:  python tools/update_pybind11.py                    (check only, changes nothing)
        python tools/update_pybind11.py --update           (install the latest release)
        python tools/update_pybind11.py --version 3.0.0    (install a specific release)

quadriga-lib uses only pybind11's headers for the Python interface (include directory of the build). This script
downloads the source archive of the release from PyPI, verifies its SHA-256, and writes external/pybind11-<ver>.zip
containing only include/, LICENSE and README.rst. CMake files, docs, tests and tooling are left out. The zip is
reproducible: running the script twice gives a byte-identical file. When the version changes, the old zip is deleted
and the version is replaced in the files listed in VERSION_FILES. Nothing is committed; the script prints the git
commands to run.

The latest release is taken from PyPI and must match the latest GitHub release (pre-releases excluded). The PyPI
source archive holds the same headers as the GitHub tag, and PyPI lists its SHA-256 (GitHub publishes no release
assets for pybind11). The version macros in include/pybind11/detail/common.h must match the release. Only the Python
standard library is used.
"""

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PYPI = "https://pypi.org/pypi/pybind11/"  # + "json" or "<ver>/json"
GITHUB_LATEST = "https://api.github.com/repos/pybind/pybind11/releases/latest"

# Files that name the bundled version.
VERSION_FILES = ("CMakeLists.txt",)

KEEP_FILES = ("LICENSE", "README.rst")  # BSD-3-Clause requires the license text
KEEP_DIRS = ("include/",)
VERSION_HEADER = "include/pybind11/detail/common.h"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)  # fixed timestamp for a reproducible zip
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "quadriga-lib-update-pybind11"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def release_info(version=None):
    """Return (version, url, sha256) of the source archive of the latest or the given release on PyPI."""
    try:
        info = json.loads(fetch(PYPI + (f"{version}/json" if version else "json")))
    except urllib.error.HTTPError as e:
        sys.exit(f"PyPI: {e.code} {e.reason}" + (f" (no release {version})" if e.code == 404 else ""))
    version = info["info"]["version"]
    if not VERSION_RE.match(version):
        sys.exit(f"Unexpected release on PyPI: {version}")
    sdists = [u for u in info["urls"] if u["packagetype"] == "sdist" and not u.get("yanked")]
    if len(sdists) != 1 or sdists[0]["filename"] != f"pybind11-{version}.tar.gz":
        sys.exit(f"Expected one source archive pybind11-{version}.tar.gz on PyPI, found "
                 f"{[u['filename'] for u in sdists]}")
    return version, sdists[0]["url"], sdists[0]["digests"]["sha256"]


def github_latest():
    rel = json.loads(fetch(GITHUB_LATEST))
    m = re.fullmatch(r"v(\d+\.\d+\.\d+)", rel["tag_name"])
    if not m or rel["draft"] or rel["prerelease"]:
        sys.exit(f"Unexpected GitHub release: {rel['tag_name']} (draft={rel['draft']}, prerelease={rel['prerelease']})")
    return m.group(1)


def bundled_version(root):
    m = re.search(r'set\(pybind11_version "([^"]+)"\)', (root / "CMakeLists.txt").read_text(encoding="utf-8"))
    if not m:
        sys.exit("pybind11_version not found in CMakeLists.txt")
    return m.group(1)


def header_version(text):
    parts = [re.search(rf"#define PYBIND11_VERSION_{k}\s+(\d+)", text) for k in ("MAJOR", "MINOR", "MICRO")]
    return ".".join(p.group(1) for p in parts) if all(parts) else "?"


def headers_zip(tar_gz, version):
    """Repack the source archive into a reproducible headers-only zip. Nothing is extracted to disk."""
    top = f"pybind11-{version}/"
    files = {}
    with tarfile.open(fileobj=io.BytesIO(tar_gz), mode="r:gz") as tar:
        for m in tar:
            if m.isdir() and m.name == top[:-1]:
                continue
            if not m.name.startswith(top) or ".." in m.name.split("/"):
                sys.exit(f"Unexpected entry in the release archive: {m.name}")
            rel = m.name[len(top):]
            if m.isdir() or not (rel in KEEP_FILES or rel.startswith(KEEP_DIRS)):
                continue
            if not m.isfile():
                sys.exit(f"Unexpected entry in the release archive: {m.name}")
            files[m.name] = tar.extractfile(m).read()

    missing = [f for f in KEEP_FILES + (VERSION_HEADER, "include/pybind11/pybind11.h") if top + f not in files]
    if missing:
        sys.exit(f"Release archive is missing expected files: {missing}")
    got = header_version(files[top + VERSION_HEADER].decode("utf-8", "replace"))
    if got != version:
        sys.exit(f"{VERSION_HEADER} says version {got}, expected {version}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3  # Unix, so the permissions above are honoured
            z.writestr(info, files[name], compresslevel=9)
    return buf.getvalue(), len(files)


def update_references(root, old, new):
    """Replace the old version in VERSION_FILES. Returns the list of changed files."""
    patterns = [
        (f"pybind11-{old}".encode(), f"pybind11-{new}".encode()),
        (f'set(pybind11_version "{old}")'.encode(), f'set(pybind11_version "{new}")'.encode()),
    ]
    changed = []
    for rel in VERSION_FILES:
        path = root / rel
        data = path.read_bytes()
        new_data = data
        for a, b in patterns:
            new_data = new_data.replace(a, b)
        if new_data != data:
            path.write_bytes(new_data)  # bytes in, bytes out: line endings and encoding are kept
            changed.append(rel)
        else:
            print(f"  {rel}: version {old} not found")
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--update", action="store_true", help="install the latest release")
    ap.add_argument("--version", help="install this release instead (e.g. 3.0.0)")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="repository root (default: the checkout this script is in)")
    args = ap.parse_args()
    root = args.root.resolve()

    old = bundled_version(root)
    print(f"Bundled:        {old}")

    if args.version:
        if not VERSION_RE.match(args.version):
            sys.exit(f"Not a release number: {args.version}")
        version, url, sha256 = release_info(args.version)
    else:
        version, url, sha256 = release_info()
        gh = github_latest()
        if gh != version:
            sys.exit(f"Sources disagree: PyPI says {version}, GitHub says {gh}. Use --version to choose one.")
        print(f"Latest release: {version}")
        if not args.update:
            print("Up to date." if version == old else "Run with --update to install it.")
            return

    print(f"Downloading pybind11-{version}.tar.gz ...")
    tar_gz = fetch(url, timeout=300)
    got = hashlib.sha256(tar_gz).hexdigest()
    if got != sha256:
        sys.exit(f"sha256 mismatch: expected {sha256}, got {got}")
    print(f"  {len(tar_gz) / 1e6:.1f} MB, sha256 verified")

    data, n = headers_zip(tar_gz, version)
    out = root / "external" / f"pybind11-{version}.zip"
    out.write_bytes(data)
    print(f"Wrote {out.relative_to(root)} ({n} files, {len(data) / 1e6:.2f} MB)")

    git_cmds = [f"git add {out.relative_to(root).as_posix()}"]
    if version != old:
        old_zip = root / "external" / f"pybind11-{old}.zip"
        if old_zip.exists():
            old_zip.unlink()
            print(f"Deleted {old_zip.relative_to(root)}")
            git_cmds.append(f"git rm --quiet external/pybind11-{old}.zip")
        changed = update_references(root, old, version)
        print(f"Updated {old} -> {version} in {len(changed)} files:")
        for rel in changed:
            print(f"  {rel}")
        if changed:
            git_cmds.append("git add " + " ".join(changed))
    print("\nReview with 'git diff', then stage with:")
    for c in git_cmds:
        print(f"  {c}")


if __name__ == "__main__":
    main()
