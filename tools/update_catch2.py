#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

"""Update the Catch2 sources bundled in external/ to the latest official release.

Usage:  python tools/update_catch2.py                    (check only, changes nothing)
        python tools/update_catch2.py --update           (install the latest release)
        python tools/update_catch2.py --version 3.8.1    (install a specific release)

quadriga-lib builds Catch2 from source for the C++ tests (CMake ExternalProject, unless CATCH2_PATH is set). This
script downloads the source archive of the release tag from GitHub and writes external/Catch2-<ver>.zip containing
only what that build needs: CMakeLists.txt, CMake/ (without the coverage helpers), src/, LICENSE.txt and README.md.
Docs, tests, examples, extras and tooling are left out; CMakeLists.txt therefore builds Catch2 with
CATCH_INSTALL_DOCS=OFF and CATCH_INSTALL_EXTRAS=OFF. The zip is reproducible: running the script twice gives a
byte-identical file. When the version changes, the old zip is deleted and the version is replaced in the files listed
in VERSION_FILES. Nothing is committed; the script prints the git commands to run.

The latest release is taken from the GitHub API (pre-releases excluded). GitHub publishes no checksum for tag
archives, so the archive is checked against the release assets instead: its extras/catch_amalgamated.hpp and .cpp
must match the SHA-256 digests GitHub lists for those assets (older releases without digests: the downloaded assets
themselves). The version in Catch2's CMakeLists.txt must match the tag. Only the Python standard library is used.
"""

import argparse
import hashlib
import io
import json
import re
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.github.com/repos/catchorg/Catch2/releases/"  # + "latest" or "tags/v<ver>"
ARCHIVE_URL = "https://github.com/catchorg/Catch2/archive/refs/tags/v{v}.zip"

# Files that name the bundled version.
VERSION_FILES = ("CMakeLists.txt",)

KEEP_FILES = ("CMakeLists.txt", "LICENSE.txt", "README.md")  # BSL-1.0 requires the license text
KEEP_DIRS = ("CMake/", "src/")
DROP_FILES = re.compile(r"^CMake/(Find\w+\.cmake|llvm-cov-wrapper)$")  # coverage helpers, only for development builds
CHECK_FILES = ("catch_amalgamated.hpp", "catch_amalgamated.cpp")  # compared with the release assets
ZIP_DATE = (1980, 1, 1, 0, 0, 0)  # fixed timestamp for a reproducible zip
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "quadriga-lib-update-catch2"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def release_info(version=None):
    """Return (version, assets) of the latest or the given release; assets maps name -> (url, sha256 or None)."""
    try:
        rel = json.loads(fetch(API + (f"tags/v{version}" if version else "latest")))
    except urllib.error.HTTPError as e:
        sys.exit(f"GitHub API: {e.code} {e.reason}" + (f" (no release v{version})" if e.code == 404 else ""))
    m = re.fullmatch(r"v(\d+\.\d+\.\d+)", rel["tag_name"])
    if not m or rel["draft"] or rel["prerelease"]:
        sys.exit(f"Unexpected release: {rel['tag_name']} (draft={rel['draft']}, prerelease={rel['prerelease']})")
    assets = {}
    for a in rel["assets"]:
        digest = a.get("digest") or ""
        assets[a["name"]] = (a["browser_download_url"], digest[7:] if digest.startswith("sha256:") else None)
    return m.group(1), assets


def bundled_version(root):
    m = re.search(r'set\(catch2_version "([^"]+)"\)', (root / "CMakeLists.txt").read_text(encoding="utf-8"))
    if not m:
        sys.exit("catch2_version not found in CMakeLists.txt")
    return m.group(1)


def verify(archive, version, assets):
    """Check the archive against the release assets and its own version number."""
    top = f"Catch2-{version}/"
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for name in CHECK_FILES:
            if name not in assets:
                sys.exit(f"Release v{version} has no asset {name}")
            url, digest = assets[name]
            how = "GitHub digest"
            if digest is None:
                digest, how = hashlib.sha256(fetch(url, timeout=120)).hexdigest(), "downloaded asset"
            got = hashlib.sha256(z.read(top + "extras/" + name)).hexdigest()
            if got != digest:
                sys.exit(f"extras/{name} does not match the release asset ({how}): expected {digest}, got {got}")
            print(f"  extras/{name}: sha256 matches the release asset ({how})")
        cml = z.read(top + "CMakeLists.txt").decode("utf-8", "replace")
    m = re.search(r"project\(Catch2\s+VERSION\s+(\d+\.\d+\.\d+)", cml)
    if not m or m.group(1) != version:
        sys.exit(f"Catch2's CMakeLists.txt says version {m.group(1) if m else '?'}, expected {version}")


def stripped_zip(archive, version):
    """Repack the tag archive into a reproducible zip with only the build files. Nothing is extracted to disk."""
    top = f"Catch2-{version}/"
    files = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for info in z.infolist():
            name = info.filename
            if not name.startswith(top) or ".." in name.split("/"):
                sys.exit(f"Unexpected entry in the release archive: {name}")
            rel = name[len(top):]
            if info.is_dir() or not (rel in KEEP_FILES or rel.startswith(KEEP_DIRS)) or DROP_FILES.match(rel):
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                sys.exit(f"Unexpected symbolic link in the release archive: {name}")
            files[name] = z.read(info)

    missing = [f for f in KEEP_FILES + ("src/CMakeLists.txt", "src/catch2/catch_all.hpp") if top + f not in files]
    if missing:
        sys.exit(f"Release archive is missing expected files: {missing}")

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
        (f"Catch2-{old}".encode(), f"Catch2-{new}".encode()),
        (f'set(catch2_version "{old}")'.encode(), f'set(catch2_version "{new}")'.encode()),
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
    ap.add_argument("--version", help="install this release instead (e.g. 3.8.1)")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="repository root (default: the checkout this script is in)")
    args = ap.parse_args()
    root = args.root.resolve()

    old = bundled_version(root)
    print(f"Bundled:        {old}")

    if args.version:
        if not VERSION_RE.match(args.version):
            sys.exit(f"Not a release number: {args.version}")
        version, assets = release_info(args.version)
    else:
        version, assets = release_info()
        print(f"Latest release: {version}")
        if not args.update:
            print("Up to date." if version == old else "Run with --update to install it.")
            return

    print(f"Downloading Catch2 v{version} source archive ...")
    archive = fetch(ARCHIVE_URL.format(v=version), timeout=300)
    print(f"  {len(archive) / 1e6:.1f} MB")
    verify(archive, version, assets)

    data, n = stripped_zip(archive, version)
    out = root / "external" / f"Catch2-{version}.zip"
    out.write_bytes(data)
    print(f"Wrote {out.relative_to(root)} ({n} files, {len(data) / 1e6:.2f} MB)")

    git_cmds = [f"git add {out.relative_to(root).as_posix()}"]
    if version != old:
        old_zip = root / "external" / f"Catch2-{old}.zip"
        if old_zip.exists():
            old_zip.unlink()
            print(f"Deleted {old_zip.relative_to(root)}")
            git_cmds.append(f"git rm --quiet external/Catch2-{old}.zip")
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
