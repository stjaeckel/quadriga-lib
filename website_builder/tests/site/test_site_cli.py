# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""The website command: tool check, outputs written, the gap list."""

import shutil
import subprocess
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parents[2] / "build.py"


def test_missing_tools_in_one_run(mini_repo):
    # -S hides site-packages (the Python packages), the empty PATH hides pandoc
    res = subprocess.run([sys.executable, "-S", str(BUILD), "website", "--root", str(mini_repo)],
                         capture_output=True, text=True, env={"PATH": ""})
    assert res.returncode == 2 and "Traceback" not in res.stderr
    for name in ("markdown-it-py", "mdit-py-plugins", "Pygments", "Jinja2", "pandoc"):
        assert name in res.stdout
    assert sys.executable in res.stdout
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


def code_copy(tmp_path, monkeypatch):
    """Copies of templates/ and static/ that the build reads instead of the code folder; static/ without
    landing figures and preview image."""
    from qsite import assets
    code = tmp_path / "code" / "website_builder"
    shutil.copytree(str(assets.TEMPLATES_DIR), str(code / "templates"))
    shutil.copytree(str(assets.STATIC_DIR), str(code / "static"))
    shutil.rmtree(str(code / "static/img/landing"), ignore_errors=True)
    (code / "static" / assets.OG_IMAGE).unlink(missing_ok=True)
    monkeypatch.setattr(assets, "TEMPLATES_DIR", code / "templates")
    monkeypatch.setattr(assets, "STATIC_DIR", code / "static")
    return code


def add_template_line(code, text):
    """Append a line to templates/legal.html; return its line number."""
    legal = code / "templates/legal.html"
    legal.write_text(legal.read_text(encoding="utf-8") + text + "\n", encoding="utf-8")
    return len(legal.read_text(encoding="utf-8").split("\n")) - 1


def test_gap_list_of_all_three_folders(mini_repo, run_site, tmp_path, monkeypatch):
    from qsite import assets
    code = code_copy(tmp_path, monkeypatch)
    line = add_template_line(code, "%%GAP: template draft%%")
    out = run_site(mini_repo)[1]
    lines = out.split("\n")
    assert "docs_builder/content/formats.md:6: an introduction to the formats" in lines
    assert "website_builder/content/404.md:6: This address does not exist." in lines
    assert "website_builder/templates/legal.html:%d: template draft" % line in lines
    base = assets.figure(assets.LANDING_FIGURES[0])
    assert "website_builder/static/%s-light.webp: landing figure missing" % base in lines
    assert "website_builder/static/%s: preview image missing" % assets.OG_IMAGE in lines


def test_structural_error_writes_nothing(mini_repo, run_site):
    (mini_repo / "website_builder/content/privacy.md").unlink()
    code, out = run_site(mini_repo)
    assert code == 1 and "website_builder/content/privacy.md: error: content file missing" in out
    assert not (mini_repo / "website").exists() and not (mini_repo / "docs").exists()


def test_template_gap_errors_name_the_template(mini_repo, run_site, tmp_path, monkeypatch):
    line = add_template_line(code_copy(tmp_path, monkeypatch), "%%GAP: not closed")
    code, out = run_site(mini_repo)
    assert code == 1 and "website_builder/templates/legal.html:%d: error: E-gap" % line in out


def test_website_only_page_from_pages(mini_repo, run_site, monkeypatch):
    from qdocs import config
    monkeypatch.setattr(config, "PAGES", config.PAGES + (("terms", None, None),))
    (mini_repo / "website_builder/content/terms.md").write_text(
        '+++\ntitle = "Terms"\ndescription = "Terms of use"\n+++\n\nThe terms.\n', encoding="utf-8")
    code, out = run_site(mini_repo)
    assert code == 0, out
    html = (mini_repo / "website/terms.html").read_text(encoding="utf-8")
    assert "<title>Terms · Quadriga-Lib</title>" in html and "<h1>Terms</h1>" in html and "<p>The terms.</p>" in html
    assert "doc-side" not in html and 'type="text/markdown"' not in html
    (mini_repo / "website_builder/content/terms.md").unlink()
    code, out = run_site(mini_repo)
    assert code == 1 and "website_builder/content/terms.md: error: content file missing" in out
