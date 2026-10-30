# Contributing to Quadriga-Lib

## For users

**Bugs.** Open an issue with the [bug report form](https://github.com/stjaeckel/quadriga-lib/issues/new?template=bug_report.yml). The most useful part is a small, self-contained example that shows the problem, together with the library version and the interface (C++, Python, MATLAB or Octave). Please search the [existing issues](https://github.com/stjaeckel/quadriga-lib/issues?q=is%3Aissue) first; if your problem is already there, add your case as a comment.

**Feature requests.** Use the [feature request form](https://github.com/stjaeckel/quadriga-lib/issues/new?template=feature_request.yml). Describe the task you want to solve, not only the function you have in mind.

**Questions.** Ask in [Discussions](https://github.com/stjaeckel/quadriga-lib/discussions). The API reference is on [quadriga-lib.org](http://quadriga-lib.org).

**Security problems.** Report them privately, see [SECURITY.md](SECURITY.md).

**Pull requests.** Small fixes are welcome. For anything larger, please open an issue first so the approach can be agreed before you write code. New or changed API functions follow `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md`.

## For the maintainer and coding sessions

This part describes how issues are filed, fixed and closed. It is written so that a coding session can follow it on its own. Everything in an issue is public: no private details, no local paths, no local branch names, no references to temporary `md/` documents. Refer to release tags (e.g. `v0.12.4`) or commits on `main`.

All commands use the GitHub CLI (`gh`), run from the repository root.

### Labels

| Group | Labels | Use |
| :---- | :----- | :-- |
| Type (one) | `bug`, `enhancement`, `documentation`, `question` | What kind of issue it is |
| Area (one or more) | `area:cpp`, `area:python`, `area:matlab`, `area:acoustic`, `area:build` | Which part of the library |
| Status | `needs-triage` | Set by the issue forms; removed once the issue is checked |

`duplicate`, `invalid`, `wontfix`, `good first issue` and `help wanted` are GitHub's defaults and may be used where they fit.

### 1. Filing an issue

**When.** A problem found during work that is not fixed in the same change gets an issue, so it is not lost.

- When the maintainer says "make this an issue" (or similar), that is the go-ahead: file it and report the link.
- When a session finds a problem on its own, it writes the draft (title, labels, body) and shows it to the maintainer. It files the issue only after a yes.

**Steps.**

1. Search for an existing issue: `gh issue list --state all --search "<keywords>"`. If one exists, add the new information as a comment (`gh issue comment <N> --body-file <file>`) instead of opening a second issue.
2. Write the body to a file outside the repository, with these sections:
   - **What happens.** The wrong behaviour in plain terms, physical meaning first.
   - **Expected.** The correct behaviour and the reference it comes from (measurement, published data, analytic case, another tool).
   - **How to reproduce.** The smallest example, the function and inputs, the commit on `main` or the release it was seen on.
   - **Size and impact.** How large the error is and which typical use cases meet it.
   - **Notes.** Suspected cause (marked as not proven), workaround, related code (`src/<file>.cpp`, function name).

   Only what a program showed counts as observed. Say so where something is an estimate.
3. Title: `<area>: <what is wrong>`, e.g. `python: <function> returns NaN for a zero-length path`.
4. Create it with one type label and the area labels. Issues filed this way are triaged already, so no `needs-triage`:

   ```bash
   gh issue create --title "<title>" --body-file <file> --label bug --label area:cpp
   ```
5. Report the issue number and link.

### 2. Fixing an issue

1. Read the issue and its comments: `gh issue view <N> --comments`.
2. Work through the usual steps: plan, research (optional), specification (optional, for larger issues), review, code with tests, review. Plans, specifications and reports are working files; they stay outside the repository and are not committed.
3. Code and test comments do not mention the issue number. A regression test describes the case in its own words; the commit links it to the issue.
4. If the work changes what the issue says (different cause, smaller scope), add a short comment to the issue so its record stays right.
5. The commit message follows the usual rules (title `<area>: <what changed>`, body in plain terms with measured numbers). Its last line before the `Co-Authored-By` trailer names the issue:
   - `Fixes #<N>` when the commit fixes it completely. GitHub closes the issue when the commit reaches `main` and links the two.
   - `Part of #<N>` when it is one step of several. This links the commit without closing the issue.
   - One line per issue when a commit fixes several.
6. The maintainer reviews, commits and pushes. Sessions do not commit or push.
7. After the push, check that the issue is closed: `gh issue view <N>`. If the fix was done on a branch, the issue closes when the branch is merged into `main`.

### 3. Closing an issue without a fix

Always give the reason in the closing comment, so the reporter and later readers know why.

| Case | Command |
| :--- | :------ |
| Dropped (not worth fixing, out of scope, works as intended) | `gh issue close <N> --reason "not planned" --comment "<why>"` |
| Duplicate | `gh issue close <N> --reason "not planned" --comment "Duplicate of #<M>."` |
| Fixed, but the commit had no `Fixes` line | `gh issue close <N> --reason completed --comment "Fixed in <commit hash>."` |
| No answer to a request for information after four weeks | `gh issue close <N> --reason "not planned" --comment "Closed for lack of information; please reopen with an example."` |

A closed issue can be reopened with `gh issue reopen <N>`.

### 4. Triage of new issues from users

New issues from the forms carry `needs-triage`. List them with `gh issue list --label needs-triage`. For each:

1. Try to reproduce it. If information is missing, ask in a comment and keep `needs-triage`.
2. If it is confirmed or accepted, set the area labels and remove `needs-triage`: `gh issue edit <N> --add-label area:python --remove-label needs-triage`.
3. If it is a usage question, answer it, or move it to Discussions with **Convert to discussion** on the issue page.
4. If it is declined, close it as in section 3, with the reason.

Reply to each new issue, even when the answer is "not planned".
