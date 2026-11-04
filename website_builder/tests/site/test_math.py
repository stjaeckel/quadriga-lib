# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Formulas: MathML from one pandoc call for the whole build, and the errors of the formula step."""

import subprocess

import pytest


@pytest.fixture
def pandoc_calls(monkeypatch):
    """Count the pandoc calls; the real pandoc still runs."""
    calls, run = [], subprocess.run

    def counting(cmd, *a, **k):
        calls.append(cmd)
        return run(cmd, *a, **k)
    monkeypatch.setattr(subprocess, "run", counting)
    return calls


def add_to_faq(root, text):
    faq = root / "docs_builder/content/faq.md"
    faq.write_text(faq.read_text(encoding="utf-8") + "\n" + text + "\n", encoding="utf-8")
    return len(faq.read_text(encoding="utf-8").split("\n")) - 1  # line of the last added line


def test_inline_and_display_formulas(mini_repo, run_site, pandoc_calls):
    add_to_faq(mini_repo, "Inline $a^2$, double $$b$$ in a paragraph.\n\n$$\nc = \\frac{1}{2}\n$$")
    code, out = run_site(mini_repo)
    assert code == 0, out
    faq = (mini_repo / "website/faq.html").read_text(encoding="utf-8")
    assert '<p>Inline <math display="inline"' in faq
    assert ', double <math display="block"' in faq  # math_inline_double: no wrapper inside the <p>
    assert '<div class="math-block"><math display="block"' in faq
    assert len(pandoc_calls) == 1 and pandoc_calls[0][:2] == ["pandoc", "-d"]
    assert "Inline $a^2$, double $$b$$" in (mini_repo / "website/faq.md").read_text(encoding="utf-8")
    assert "$$\\sigma^2" in (mini_repo / "website/tutorials/mini_demo.md").read_text(encoding="utf-8")


def test_equation_label_is_an_error(mini_repo, run_site):
    line = add_to_faq(mini_repo, "$$x$$ (eq1)")
    code, out = run_site(mini_repo)
    assert code == 1
    assert "docs_builder/content/faq.md:%d: error: equation labels" % line in out


def test_bad_formulas_at_their_lines(mini_repo, run_site):
    first = add_to_faq(mini_repo, "Unknown $\\foo{x}$ here.")
    second = add_to_faq(mini_repo, "Unbalanced $a{b$ here.")
    code, out = run_site(mini_repo)
    assert code == 1
    assert "docs_builder/content/faq.md:%d: error: formula not converted to MathML" % first in out
    assert "docs_builder/content/faq.md:%d: error: formula not converted to MathML" % second in out
    assert "pandoc: error: formula conversion failed:" in out
    assert not (mini_repo / "website").exists()
