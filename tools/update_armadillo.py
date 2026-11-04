#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

"""Update the Armadillo headers bundled in external/ to the latest official stable release.

Usage:  python tools/update_armadillo.py                    (check only, changes nothing)
        python tools/update_armadillo.py --update           (install the latest stable release)
        python tools/update_armadillo.py --version 14.2.2   (install a specific release)

quadriga-lib only needs Armadillo's headers when no system Armadillo is found (always on Windows). This script
downloads the official release from SourceForge, verifies its checksum, and writes external/armadillo-<ver>.zip
containing only include/, LICENSE.txt, NOTICE.txt and README.md, with the local edits in PATCHES applied to the
headers. The zip is reproducible: running the script twice gives a byte-identical file. When the version changes,
the old zip is deleted and the version is replaced in the files listed in VERSION_FILES. Nothing is committed; the
script prints the git commands to run.

The latest stable release is taken from SourceForge's best_release.json (which also gives the SHA-256) and must match
the "Stable Version" on the official download page. Older releases are checked against the MD5 from SourceForge's
file feed. Only the Python standard library is used.
"""

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

DOWNLOAD_PAGE = "https://arma.sourceforge.net/download.html"
BEST_RELEASE = "https://sourceforge.net/projects/arma/best_release.json"
FILE_FEED = "https://sourceforge.net/projects/arma/rss?path=/"  # lists the most recent ~100 files
FILE_URL = "https://sourceforge.net/projects/arma/files/armadillo-{v}.tar.xz/download"

# Files that name the bundled version.
VERSION_FILES = ("CMakeLists.txt", "Makefile", "README.md")

KEEP_FILES = ("LICENSE.txt", "NOTICE.txt", "README.md")  # Apache-2.0 requires LICENSE and NOTICE

# Local edits applied to every release: explicit integer-to-float casts in linspace and regspace, which otherwise
# warn under -Wconversion (GCC, Clang). The results are unchanged. Remove an entry once upstream has the same fix;
# the script reports entries that no longer match.
PATCHES = [
    ("include/armadillo_bits/fn_misc.hpp", "eT(start + i*delta)", "eT(start + T(i)*delta)"),
    ("include/armadillo_bits/fn_misc.hpp", "eT(float(start) + i*delta)", "eT(float(start) + float(i)*delta)"),
    ("include/armadillo_bits/fn_misc.hpp", "eT(double(start) + i*delta)", "eT(double(start) + double(i)*delta)"),
    ("include/armadillo_bits/fn_regspace.hpp", "T(i*inc)", "T(i)*inc"),
]
ZIP_DATE = (1980, 1, 1, 0, 0, 0)  # fixed timestamp for a reproducible zip
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "quadriga-lib-update-armadillo"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def latest_stable():
    """Return (version, sha256) of the latest stable release; both sources must agree."""
    best = json.loads(fetch(BEST_RELEASE))["release"]
    m = re.search(r"armadillo-(\d+\.\d+\.\d+)\.tar\.xz$", best["filename"])
    if not m:
        sys.exit(f"Unexpected file in best_release.json: {best['filename']}")
    version, sha256 = m.group(1), best["sha256sum"]

    page = fetch(DOWNLOAD_PAGE).decode("utf-8", "replace")
    pos = page.find("Stable Version")
    m = re.search(r"armadillo-(\d+\.\d+\.\d+)\.tar\.xz", page[pos:]) if pos >= 0 else None
    if not m:
        sys.exit(f"Could not find the 'Stable Version' entry on {DOWNLOAD_PAGE}")
    if m.group(1) != version:
        sys.exit(f"Sources disagree: download page says {m.group(1)}, best_release.json says {version}. "
                 f"Use --version to choose one.")
    return version, sha256


def release_md5(version):
    """Return the MD5 of a release from SourceForge's file feed."""
    feed = fetch(FILE_FEED).decode("utf-8", "replace")
    name = re.escape(f"armadillo-{version}.tar.xz")
    m = re.search(name + r"/download\".*?<media:hash algo=\"md5\">([0-9a-f]{32})<", feed, re.S)
    if not m:
        sys.exit(f"armadillo-{version}.tar.xz is not in the SourceForge file feed (unknown or too old)")
    return m.group(1)


def bundled_version(root):
    m = re.search(r'set\(armadillo_version "([^"]+)"\)', (root / "CMakeLists.txt").read_text(encoding="utf-8"))
    if not m:
        sys.exit("armadillo_version not found in CMakeLists.txt")
    return m.group(1)


def headers_zip(tar_xz, version):
    """Repack the release tarball into a reproducible headers-only zip. Nothing is extracted to disk."""
    top = f"armadillo-{version}/"
    keep = {top + f for f in KEEP_FILES}
    files = {}
    with tarfile.open(fileobj=io.BytesIO(tar_xz), mode="r:xz") as tar:
        for m in tar:
            if not (m.name in keep or m.name.startswith(top + "include/")):
                continue
            if m.isdir():
                continue
            if not m.isfile() or ".." in m.name.split("/") or m.name.startswith("/"):
                sys.exit(f"Unexpected entry in the release archive: {m.name}")
            files[m.name] = tar.extractfile(m).read()

    missing = [k for k in keep if k not in files]
    if missing or top + "include/armadillo" not in files:
        sys.exit(f"Release archive is missing expected files: {missing or ['include/armadillo']}")

    for rel, old, new in PATCHES:
        name = top + rel
        n = files[name].count(old.encode()) if name in files else 0
        if n:
            files[name] = files[name].replace(old.encode(), new.encode())
        print(f"  patch {rel.split('/')[-1]}: '{old}' -> '{new}' ({n}x)" + ("" if n else "  NOT FOUND"))

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
        (f"armadillo-{old}".encode(), f"armadillo-{new}".encode()),
        (f'set(armadillo_version "{old}")'.encode(), f'set(armadillo_version "{new}")'.encode()),
        (f"armadillo_version = {old}".encode(), f"armadillo_version = {new}".encode()),
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
    ap.add_argument("--update", action="store_true", help="install the latest stable release")
    ap.add_argument("--version", help="install this release instead (e.g. 14.2.2)")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="repository root (default: the checkout this script is in)")
    args = ap.parse_args()
    root = args.root.resolve()

    old = bundled_version(root)
    print(f"Bundled:       {old}")

    if args.version:
        if not VERSION_RE.match(args.version):
            sys.exit(f"Not a release number: {args.version}")
        version = args.version
        algo, digest = "md5", release_md5(version)
    else:
        version, digest = latest_stable()
        algo = "sha256"
        print(f"Latest stable: {version}")
        if not args.update:
            print("Up to date." if version == old else "Run with --update to install it.")
            return

    print(f"Downloading armadillo-{version}.tar.xz ...")
    tar_xz = fetch(FILE_URL.format(v=version), timeout=300)
    got = hashlib.new(algo, tar_xz).hexdigest()
    if got != digest:
        sys.exit(f"{algo} mismatch: expected {digest}, got {got}")
    print(f"  {len(tar_xz) / 1e6:.1f} MB, {algo} verified")

    data, n = headers_zip(tar_xz, version)
    out = root / "external" / f"armadillo-{version}.zip"
    out.write_bytes(data)
    print(f"Wrote {out.relative_to(root)} ({n} files, {len(data) / 1e6:.2f} MB)")

    git_cmds = [f"git add {out.relative_to(root).as_posix()}"]
    if version != old:
        old_zip = root / "external" / f"armadillo-{old}.zip"
        if old_zip.exists():
            old_zip.unlink()
            print(f"Deleted {old_zip.relative_to(root)}")
            git_cmds.append(f"git rm --quiet external/armadillo-{old}.zip")
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
