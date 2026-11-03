# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""The website command: tool check, outputs written, the gap list."""

import os
import subprocess
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parents[2] / "build.py"


def test_missing_tools_in_one_run(mini_repo):
    # -S hides site-packages (the Python packages), the empty PATH hides pandoc
    res = subprocess.run([sys.executable, "-S", str(BUILD), "website", "--root", str(mini_repo)],
                         capture_output=True, text=True, env={"PATH": ""})
    assert res.returncode == 2
    out = res.stdout
    for name in ("markdown-it-py", "mdit-py-plugins", "Pygments", "Jinja2", "pandoc not found on PATH"):
        assert name in out
    assert sys.executable in out and "quadriga-web" in out and "website_builder/README.md" in out
    assert "sudo apt install pandoc" in out and "Traceback" not in res.stderr
    assert not (mini_repo / "docs").exists() and not (mini_repo / "website").exists()


def test_website_writes_docs_and_site(mini_repo, run_site):
    code, out = run_site(mini_repo)
    assert code == 0, out
    assert out.rstrip("\n").split("\n")[-1].startswith("website: 0 errors, ")
    assert (mini_repo / "docs/quadriga_lib_api_cpp.md").is_file()
    assert not (mini_repo / "website/README.md").exists()
    for rel in ("index.html", "cpp_api.html", "quadriga_lib_api_cpp.md", "api/cpp/miscellaneous-tools.md",
                "llms.txt", "skill/SKILL.md", "tutorials.md", "tutorials/mini_demo.html", ".htaccess",
                "css/site.css", "js/site.js", "favicon.ico", "img/favicon.svg"):
        assert (mini_repo / "website" / rel).is_file(), rel
    skill = (mini_repo / "website/skill/SKILL.md").read_text(encoding="utf-8")
    assert "http://quadriga-lib.org/llms.txt" in skill and "{{" not in skill
    # A second run writes nothing
    code, out = run_site(mini_repo)
    assert code == 0 and "website: 0 files written, 0 deleted" in out


def test_gap_list_of_all_three_folders(mini_repo, run_site):
    code, out = run_site(mini_repo)
    lines = out.split("\n")
    assert "docs_builder/content/formats.md:6: an introduction to the formats" in lines
    assert "website_builder/content/404.md:6: This address does not exist." in lines
    assert any(ln.startswith("website_builder/templates/landing.html:") for ln in lines)
    assert "website_builder/static/img/landing/coverage-light.webp: landing figure missing" in lines
    assert "website_builder/static/img/og-card.png: preview image missing" in lines


def test_structural_error_writes_nothing(mini_repo, run_site):
    (mini_repo / "website_builder/content/privacy.md").unlink()
    code, out = run_site(mini_repo)
    assert code == 1 and "website_builder/content/privacy.md: error: content file missing" in out
    assert not (mini_repo / "website").exists() and not (mini_repo / "docs").exists()
