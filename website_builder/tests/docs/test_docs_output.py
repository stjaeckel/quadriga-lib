# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""Writing docs/: only changed files, stale files removed, check mode, \\r\\n line ends."""

import os


def tree(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_only_changed_files_written_and_stale_removed(mini_repo, run_docs):
    docs = mini_repo / "docs"
    assert run_docs(mini_repo)[0] == 0
    for p in docs.rglob("*"):
        if p.is_file():
            os.utime(p, ns=(0, 0))
    (docs / "old").mkdir()
    (docs / "old" / "stale.md").write_text("stale\n", encoding="utf-8")
    faq = mini_repo / "docs_builder/content/faq.md"
    faq.write_text(faq.read_text(encoding="utf-8").replace("Open an issue", "Please open an issue"), encoding="utf-8")
    code, out = run_docs(mini_repo)
    assert code == 0 and "wrote docs/faq.md\ndeleted docs/old/stale.md\n" in out
    assert not (docs / "old").exists()
    changed = sorted(p.relative_to(docs).as_posix() for p in docs.rglob("*") if p.is_file() and p.stat().st_mtime_ns)
    assert changed == ["faq.md"]


def test_check_mode(mini_repo, run_docs):
    docs = mini_repo / "docs"
    code, out = run_docs(mini_repo, "--check")
    assert code == 1 and not docs.exists()
    assert run_docs(mini_repo)[0] == 0
    code, out = run_docs(mini_repo, "--check")
    assert code == 0 and out.rstrip("\n").split("\n")[-1].startswith("docs --check: 0 errors,")
    (docs / "faq.md").write_text("changed\n", encoding="utf-8")
    (docs / "llms.txt").unlink()
    (docs / "api" / "extra.md").write_text("extra\n", encoding="utf-8")
    before = tree(docs)
    code, out = run_docs(mini_repo, "--check")
    assert code == 1
    assert "docs/faq.md: error: differs" in out and "docs/llms.txt: error: missing" in out
    assert "docs/api/extra.md: error: extra" in out
    assert tree(docs) == before


def test_crlf_line_ends_give_the_same_output(mini_repo, tmp_path, run_docs):
    assert run_docs(mini_repo)[0] == 0
    reference = tree(mini_repo / "docs")
    for p in mini_repo.rglob("*"):
        if p.is_file():
            p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    code, out = run_docs(mini_repo, "--check")
    assert code == 0, out
    code, out = run_docs(mini_repo)
    assert code == 0 and "wrote" not in out  # files that differ only in line ends are not rewritten
    (mini_repo / "docs" / "faq.md").unlink()
    assert run_docs(mini_repo)[0] == 0
    assert tree(mini_repo / "docs")["faq.md"] == reference["faq.md"]
