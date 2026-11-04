# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Static files, landing figures and the skill copy."""

from pathlib import Path

from qdocs import prose

CODE_DIR = Path(__file__).resolve().parents[1]
# Module constants, read at call time, so tests may point them to a copy
STATIC_DIR = CODE_DIR / "static"
TEMPLATES_DIR = CODE_DIR / "templates"
PANDOC_DEFAULTS = CODE_DIR / "pandoc_math.yaml"
LANDING_FIGURES = ("coverage", "antenna")
OG_IMAGE = "img/og-card.png"
SKILL = "website_builder/skill/quadriga-lib/SKILL.md"


def static_files():
    """{site path: bytes} of every file below STATIC_DIR, except files named README.md."""
    base = STATIC_DIR
    return {p.relative_to(base).as_posix(): p.read_bytes() for p in sorted(base.rglob("*"))
            if p.is_file() and p.name != "README.md"}


def figure(name):
    """Site path base of a landing figure: img/landing/<name>."""
    return "img/landing/" + name


def landing_figures(diag):
    """{name: True if both variants exist}; a missing variant goes on the gap list."""
    out = {}
    for name in LANDING_FIGURES:
        missing = [v for v in ("light", "dark") if not (STATIC_DIR / ("%s-%s.webp" % (figure(name), v))).is_file()]
        for v in missing:
            diag.gap("website_builder/static/%s-%s.webp" % (figure(name), v), None, "landing figure missing")
        out[name] = not missing
    return out


def og_image(diag):
    """True if the preview image exists; otherwise it goes on the gap list."""
    if (STATIC_DIR / OG_IMAGE).is_file():
        return True
    diag.gap("website_builder/static/" + OG_IMAGE, None, "preview image missing")
    return False


def skill(root, diag):
    """skill/SKILL.md of the site: the stub with build variables substituted."""
    path = root / SKILL
    if not path.is_file():
        diag.error(SKILL, None, "content file missing")
        return {}
    return {"skill/SKILL.md": prose.substitute(path.read_text(encoding="utf-8"), diag, SKILL).encode("utf-8")}
