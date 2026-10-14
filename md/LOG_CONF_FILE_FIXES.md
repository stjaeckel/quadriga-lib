# Log: fixes after the review of the conf file parser

Session of 2026-10-14 in `/sjc/quadriga-lib-conf`, branch `conf-file-parser`, on top of 145f163, following
`md/PLAN_CONF_FILE_FIXES.md` (the plan) and `md/REVIEW_CONF_FILE.md` (the review). Nothing is committed. One
decision was asked during the session (section 3): the number conversion keeps a classic-locale stream instead
of `std::from_chars`, because `from_chars` made the MEX files unloadable in MATLAB R2024a.

Machine: 32 cores; gcc 12.5.0 (library, MEX, Python), gcc 15.2.0 (sanitizers, coverage), g++ 16 (one warning
check), Octave 11.1.0, MATLAB 24.1.0.3050227 (R2024a) Update 8, Python 3.14.4, numpy 2.3.5, pytest 9.0.2.

Marks: **[M]** measured in this session, **[R]** reproduced with a scratch program or script, **[C]** read from
the code. Scratch programs and logs are in the session scratch folder
(`/tmp/claude-1005/-sjc-quadriga-lib-conf/4335c5be-63e1-4191-b897-e61093a330db/scratchpad/`), not in the
worktree.

## Summary

- All 13 work items are done, and so are all 18 tests and 12 documentation items. Every new test failed for the
  intended reason on the unchanged code before the source was changed. Tests that only pin existing behaviour
  (T2, T3, T6, T7) passed on the unchanged code, as expected.
- Final state [M]: Catch2 complete `All tests passed (1301077 assertions in 1117 test cases)` (1108 + 9 new
  cases); the 48 conf cases are clean under ASan + UBSan with the leak check on; Python `760 passed, 3 skipped`
  (752 + 8); Octave and MATLAB 65 of 66, only `test_ray_mesh_interact` line 113 failing as before; no compiler
  warning.
- Deviations from the plan, each explained below: F4 took the plan's fallback (section 3, decided by Stephan
  in the session). F1 additionally rejects an existing target that is not a regular file. T1 measured
  18,364 bytes for the "16 KB" text. The T8 BOM literal had to be split.
- `qd_scenpar.cpp`: 1406 lines before, 1546 after; 99.18 % of 485 lines run in the conf cases. The four lines
  that do not run are explained in section 4.3.


## 1. Baseline before the change

Measured at 145f163 before any edit; it equals the plan's table. Build as in plan section 0 (gcc 12, no AVX2,
MEX, MEX docs, Python, tests).

| Suite | Result [M] |
|---|---|
| Catch2, new cases | `All tests passed (7949 assertions in 39 test cases)` |
| Catch2, complete | `All tests passed (1300843 assertions in 1108 test cases)`, 1 min 50 s |
| MEX, Octave 11.1.0, complete | `FAILED (passed=65, failure=1)`; `test_ray_mesh_interact:113`, 0.98485 vs 0.69639 |
| MEX, MATLAB R2024a, complete | `FAILED (passed=65, failure=1)`; same test, same numbers |
| Python, complete | `752 passed, 3 skipped in 43.02s` |
| Warnings (library, MEX, Python build log) | 0 lines with `warning:` |

Build note: the first `cmake --build` failed because the Catch2 sub-build in `build_linux/catch2-3.8.1_build`
had a stale cache with `CMAKE_INSTALL_PREFIX=/usr/local` (permission denied on install). I deleted that
sub-build folder and its stamp folder; it then reconfigured with `build_linux/catch2-3.8.1_bin`. These are
build artefacts only and no source changed.


## 2. Decisions, tests and documentation items

### 2.1 Decisions of plan section 2

| # | What was changed | Pinned by | State |
|---|---|---|---|
| B1 | `qd_scenpar.cpp`, `conf_file_write`: write to `<target>.tmp<hex time><hex thread>` in the same directory, `close()`, check `fail()`, copy the permissions of an existing target, `rename`; the temporary file is removed on every failure; a symbolic link is resolved with `weakly_canonical` and written through; an existing target must be a regular file that opens with `app` (F1) | T1 | done [M] |
| B2 | `conf_file_write`: the scenario name is computed right after the `fn` check; `\n` or `\r` in it is `invalid_argument` "Input 'fn' must not contain a line break in the file name." (F2) | T4, T17 | done [M] |
| B3 | `python_quadriga_adapter.hpp`: `qd_python_flag`; the flags of the four wrappers are `py::handle`; `None` gives the default, `bool` and `numpy.bool_` their value, anything else `TypeError` "Input '<name>' must be a bool or None." (F11) | T15 | done [M] |
| Q1 | `qd_conf_read_lines`: `EF BB BF` at the start of line 1 is removed; reader and lister (F3) | T8, T18 | done [M] |
| Q2 | `qd_conf_parse_number` returns `QD_NUM_OK` / `QD_NUM_INVALID` / `QD_NUM_RANGE`; the reader reports "is out of range."; underflow keeps the value of `strtod` (F4). The conversion stays a classic-locale stream (section 3) | T5, T18 | done, fallback [M] |
| Q3 | Reader: a repeated `ShortName` value is "ShortName '<v>' is stated twice." in both modes; writer: "Entries of 'short_names' must be distinct." (F5) | T9, T17 | done [M] |
| Q4 | `qd_python_try_path`, `qd_python_path`: `str`, `bytes`, `os.PathLike` (via `os.fspath`); `fn` of read and write is `py::object`; `dirs` of the lister: one path, or a list or tuple of paths; other types keep the `ValueError` "Expected a string, a list of strings, or None." (F12) | T16 | done [M] |
| Q5 | `src/mex_helper_functions.hpp`: 31 identifiers `MATLAB:unexpectedCPPexception` replaced by `quadriga_lib:CPPerror`; the old identifier remains only in `md/PLAN_CONF_FILE_FIXES.md` and `md/REVIEW_CONF_FILE.md` (grep of the repository without `build_*`) (F9) | T14 | done [M] |
| Q6 | `qd_mex_struct2plpar`: more than 2 dimensions, or more than one row and more than one column, is "Field '<name>' must be a real numeric vector."; helper comment updated (F10) | T13 | done [M] |
| Q7 | Kept; documented in the three `conf_file_list` blocks ("whatever characters it has") | doc lint | done [C] |
| Q8 | Kept; documented (D5) | doc lint, lister probe | done [C, R] |
| Q10 | `conf_file_write` overview: frequency `max( fGHz, <fc_min> )` if `LSP_fc_min > 0`; the ESD mean gets the terms of `ES_D_mu_A` to `ES_D_mu_D` and `max( <mu_min>, <mean> )` if `ES_D_mu_min` is not `-Inf` (F7) | T11 | done [M] |
| Q11 | `QD_INT_POS`, `QD_INT_NONNEG`: upper bound 1000, messages "must be an integer from 1 to 1000" and "... from 0 to 1000"; enum comments updated. Only `NumClusters`, `NumSubPaths`, `NumClusters_min`, `NF_k1` use the classes [C] (F6) | T10 | done [M] |
| Q12 | Test file names per case (T12) | T12 | done [M] |
| D12 | The four tables are `static const char *const`; the lambda `where` of `qd_scenpar_check` is `fn.empty() ? "" : fn + ", line " ...`, after checking that every reader call with a failing key passes a line > 0 (defaults pass every check; the cross-key checks use the line of a key that was set) (F8) | full suites | done [C, M] |

Additions to the plan, each a deviation to review:
- F1 rejects every existing target that is not a regular file, not only a directory. Without this a write to a
  device as root would rename a regular file over the device node (`/dev/null`). As a result `/dev/full` now
  fails at this check ("Cannot open file '/dev/full' for writing.", line 1511 ran) and not at the creation of
  the temporary file. The error type and message are the ones the plan expects [M].
- Behaviour change of F1, not tested and documented as "which must be writable": an existing writable file in a
  read-only directory can no longer be written, because the temporary file needs the directory.

### 2.2 Tests T1 to T18

Red = run against the unchanged code, before the source change [M].

| # | Where | What | Red run | Green |
|---|---|---|---|---|
| T1 | `test_conf_file_write.cpp`, new case "a failed write leaves an existing file unchanged" (Linux) | file size limit 2048 B, `SIGXFSZ` ignored, both restored before any assertion: 3432 B text (60 path-loss keys, values 1/3) and 18,364 B text (`write_defaults`); `/dev/full` (skipped as root); read-only file; read-only directory; permission bits kept (0600); new file has the permissions of `std::ofstream`; a symbolic link is written through; a link to itself throws; no `*.tmp*` left | 3432 B: no exception, content lost; 18,364 B: exception, content lost; `/dev/full` without `write_defaults`: no exception. Read-only, permission and link parts passed (they pin the old behaviour that the rename must keep); the check of a link to itself was added after the change, for coverage | pass |
| T2 | same file, new case "numbers do not depend on the global locale" | `setlocale` + `std::locale::global`; ran with `en_DK.UTF-8` (no `de_DE` installed), checked that `1.5` prints as `1,5`; bits and text equal the classic run; restored by an RAII guard | passed (pins existing behaviour) | pass |
| T3 | `test_conf_file_read.cpp`, "input validation / file cannot be opened" | directory `test_conf_file_read_dir.conf` as `fn` throws `runtime_error` | passed (pin) | pass |
| T4 | `test_conf_file_write.cpp`, "input validation / fn" | `"test_conf_file_write_x\nKF_mu = 5 %.conf"` and the `\r` form throw `invalid_argument`, no file | no exception, file created (removed by the guard) | pass |
| T5 | `test_conf_file_read.cpp`, new case "a number beyond the range of a double is an error, below it reads as 0" | overflow: `1e400`, `-1e400`, `1e99999999999999999999`, `1.8e308`, `1.7976931348623159e308`, `PL_A = 1 1e400`, both modes, message with "line 2" and the key; underflow and range ends compared bit for bit, also as a path-loss value: `1e-400` +0, `-1e-400` -0, `4.9e-324` and `2.4703282292062328e-324` the smallest subnormal, `2.4703282292062327e-324` +0 (and -0 with a sign), `1e-310`, a 1e-331 text, `±1.7976931348623157e308`, `0e999999999999` | 12 failures: "is not a number" instead of "is out of range"; the underflow section passed | pass |
| T6 | `test_conf_file_list.cpp`, new case "a listed file that cannot be read is an error only when its short names are read" | mode 000; skipped if the file still opens (root); `runtime_error`, outputs untouched; listed with `false` | passed (pin) | pass |
| T7 | same file, new case "a symbolic link to a regular file is listed, other links are skipped" | link to a file listed with its short name; dangling link and link to a directory skipped; skipped if links cannot be created | passed (pin) | pass |
| T8 | read and list files, two new cases | BOM before a comment and before a key, BOM alone, BOM in line 2 ("line 2", both modes), incomplete BOM ("line 1"); lister reads `ShortName` after a BOM. The plan's literal `"\xEF\xBB\xBFDS_mu"` would make `\xBFD` one hex escape in C++, so the literal is split | reader: error in line 1; lister: `Bs` missing | pass |
| T9 | read file, new case "a short name stated twice in one file is an error"; write file, "input validation / short_names" | "line 2" / "line 4" and the message, both modes; `Ab`, `AB`, `Cd` pass; writer `{"Ab", "Cd", "Ab"}` throws, no file | no error | pass |
| T10 | validate, read and write files | 1000 accepted, 1001 rejected for the four keys; message of `NumClusters_min = 1001` with `NumClusters = 1000` names the bound (class check comes first); through a file with the line; writer rejects 1001, accepts 1000 | accepted; old messages "integer > 0" | pass |
| T11 | write file, layout test and new case "the ESD overview line shows the distance and height terms and the minimum" | DS and ESD lines of the layout typed by hand; three sections: C and D terms without minimum; minimum, A term, `max( fGHz, 2.5 )` in mean and STD; minimum `-0` with a B term | old lines | pass |
| T12 | read, list files | `read_text` uses `test_<case name>.conf` (`Catch::getResultCapture().getCurrentTestName()`, non-alphanumerics as `_`); the lister directories `test_conf_file_list_a` and `_b` were shared by 5 cases and are now distinct per case; `test_conf_file_read_missing.conf` is named by two cases but never created, so it was kept; the write and validate files had no shared names | n/a | pass |
| T13 | `tests/mex_tests/test_conf_file_write.m` | `[1 2; 3 4]` and `ones(1, 1, 2)` raise "must be a real numeric vector."; row `[1 2 3]` and column `[4; 5; 6]` read back in element order | the matrix was accepted and written as `1 3 2 4` | pass |
| T14 | `test_conf_file_read.m`, `test_scenpar_validate.m` | local `assert_identifier`: `conf_file_read(5)` and a C++ error carry `quadriga_lib:CPPerror`; `scenpar_validate(struct('NumClusters', int8(7)))` as well | `MATLAB:unexpectedCPPexception` | pass (Octave, MATLAB) |
| T15 | the four Python test files, `test_*_flag_types` | `None` = omitted (positional and keyword), `numpy.bool_` both values, `"yes"`, `1`, `0`, `1.0` raise `TypeError` with the message. No existing Python test passed an integer flag (grep), so none was adapted | `None` was `False`; `1` was accepted | pass |
| T16 | read, write, list Python tests, `test_path_like`, `test_dirs_path_like` | `pathlib.Path`, `bytes`, an object with `__fspath__` (str and bytes) give the same result as `str`; `dirs` alone and in a list or tuple, mixed with bytes; `file_dirs` are `str`; `int`, `None`, list as `fn` raise `TypeError` | `TypeError` / `ValueError` | pass |
| T17 | MEX and Python write tests | line break in the file name (MEX `char(10)`, Python `\n` and `\r`), duplicate short names; no file | no error | pass |
| T18 | MEX and Python read tests | BOM file reads (value, short name, no ignored key); `1e400` raises "line 2: Value of 'DS_mu' is out of range."; Python also `-1e-400` gives -0 | BOM error in line 1; "is not a number" | pass |

### 2.3 Documentation D1 to D12

| # | Blocks | Change | State |
|---|---|---|---|
| D1 | `conf_file_read` C++, MEX, Python | "A number beyond the range of a double (`1e400`) is an error "out of range"; a number below the smallest subnormal reads as `0` (or `-0`)"; "a number out of range" added to the error list | done |
| D2 | `conf_file_write` ×3 | "up to 15 significant digits, or with 16 or 17 where 15 do not read back ..." | done |
| D3 | `conf_file_write` ×3 | "A failed call does not create a file and leaves an existing file unchanged; an existing file is replaced and keeps its permissions"; temporary file, writable directory, link written through; line break in the file name and duplicate short names listed as errors. Header `quadriga_tools.hpp`: "an existing file is replaced; a failed call leaves it unchanged" | done |
| D4 | `conf_file_read` ×3 | "... cannot be opened or is a directory" | done |
| D5 | `conf_file_list` ×3 | regular files and links to regular files, longer than `.conf`; "cannot be listed is skipped" | done |
| D6 | `CONF_FILE_READ` | See also `[[conf_file_write]]` | done |
| D7 | `scenpar_validate` ×3 | See also "(applies the same checks before writing)" | done |
| D8 | MEX blocks | lines rewrapped; all four MEX blocks at most 100 characters | done |
| D9 | `conf_file_read` C++ (also MEX, Python) | "Numbers are read independent of the global locale (the decimal point is always `.`)" | done |
| D10 | `conf_file_read` C++, MEX | `[n_short]`, `[n_ignored]`; MEX `[n_short, 1]`, `[n_ignored, 1]` | done |
| D11 | `md/conf_file_parser.md` | new section "Record of later changes" after section 3: `GR_epsilon` (3690107), `conf_file_write` (e525c7b), wrappers (3690107, ad7a9a4, 145f163), parity test, decisions B1 to Q12, including the F4 decision of this session | done |
| D12 | code | see F8 above | done |

New behaviour documented in all three layers: BOM, repeated `ShortName` ("can occur several times with
different values" and the error list), counts up to 1000 (class list of `scenpar_validate`), the overview
formulas of `conf_file_write`; Python flags ("bool or `numpy.bool_`; default: ...; `None` selects the default")
and path-like `fn` and `dirs`; MEX `plpar` "row or column; a matrix is an error". Doc lint of the twelve blocks
with the review's `doclint.py` [M]: no `*` or `|` in text; longest lines C++ 117, Python 112, MEX 100.
`make documentation` [M]: the content hunks of `html_docu/` are all in the four conf entries; the other hunks
are line numbers in the function index tables, which moved because the entries grew. A second run changes
nothing.


## 3. F4: `from_chars` against `strtod`, and the path taken

Measured before the change with `f4/fc_probe.cpp` (scratch), libstdc++ 12.5.0 and 15.2.0 give identical
results [M]:

| Text | `from_chars` | value bits | `strtod` (C locale) | bits | equal |
|---|---|---|---|---|---|
| `4.9e-324` | ok | `0000000000000001` | ERANGE | `0000000000000001` | yes |
| `2.4703282292062328e-324` | ok | `0000000000000001` | ERANGE | `0000000000000001` | yes |
| `2.4703282292062327e-324` | out of range | (unchanged) | ERANGE | `0000000000000000` | no |
| `1e-310` | ok | `000012688b70e62b` | ERANGE | `000012688b70e62b` | yes |
| `1e-400`, `-1e-400` | out of range | (unchanged) | ERANGE | +0, -0 | no |
| `1e400`, `-1e400`, `1.8e308`, `1e99999999999999999999` | out of range | (unchanged) | ERANGE | ±Inf | no |
| `1.7976931348623157e308` | ok | `7fefffffffffffff` | 0 | `7fefffffffffffff` | yes |

`from_chars` reports `result_out_of_range` only where `strtod` gives 0 or ±Inf, never for a non-zero
subnormal, so the plan's main path applied and was implemented first. All 48 Catch2 cases, the Python suite
(760) and Octave passed with it [M]. MATLAB R2024a then failed the four conf tests with "Invalid MEX-file ...
libstdc++.so.6: version `GLIBCXX_3.4.29' not found" (61 of 66) [M]. `objdump -T` showed one symbol at that
version: `std::from_chars(char const*, char const*, double&, std::chars_format)`. In libstdc++ the
floating-point `from_chars` is in the shared library, and MATLAB's bundled libstdc++ ends at
`GLIBCXX_3.4.28` [M].

Stephan chose the plan's fallback in the session. The conversion stays `std::istringstream` with
`std::locale::classic()`, and only the classification is added: a text that passed the grammar and makes the
stream fail is `QD_NUM_RANGE` if the decimal order of its first non-zero digit (digits before the point
counted from it, minus the zeros after the point before it, plus the saturated exponent) is > 0. Otherwise it
stays `QD_NUM_INVALID`. libstdc++ sets `failbit` on overflow (LWG 23) and returns the `strtod` value on
underflow, so the decided behaviour holds. The four conf MEX files now need at most `GLIBCXX_3.4.26` [M]. A
comment at `qd_conf_parse_number` gives the reason. The writer's read-back uses the same function.


## 4. Final results

### 4.1 Suites [M]

| Build | Suite | Result |
|---|---|---|
| gcc 12, no AVX2 | Catch2, conf cases | `All tests passed (8183 assertions in 48 test cases)`, 0.19 s |
| gcc 12, no AVX2 | Catch2, complete | `All tests passed (1301077 assertions in 1117 test cases)`, 9 min 49 s wall time under a load of 39 to 57 from other processes on the machine (1 min 50 s at the baseline) |
| gcc 15, ASan + UBSan, `detect_leaks=1` | Catch2, conf cases | `All tests passed (8183 assertions in 48 test cases)`, no sanitizer report |
| gcc 15, `--coverage -O0` | Catch2, conf cases | `All tests passed (8183 assertions in 48 test cases)` |
| gcc 12, Octave 11.1.0 | MEX, complete | `FAILED (passed=65, failure=1)`; `test_ray_mesh_interact:113`, 0.98485 vs 0.69639 |
| gcc 12, MATLAB R2024a, no `LD_PRELOAD` | MEX, complete | `FAILED (passed=65, failure=1)`; same test, same numbers |
| gcc 12 | Python, conf files | `36 passed` |
| gcc 12 | Python, complete | `760 passed, 3 skipped in 48.69s` |

- The MEX test count stays 66: the new MEX checks are in the existing files `test_conf_file_read.m`,
  `test_conf_file_write.m` and `test_scenpar_validate.m`.
- Warnings: 0 lines with `warning:` in the final gcc 12 build log (library, Octave MEX, Python module) and in
  the gcc 15 ASan build. The four conf MEX files and `icosphere.cpp` compiled by hand with `g++-12 -std=c++17
  -Wall -Wextra -Wpedantic -DMATLAB_MEX_FILE`: 0 warnings. `qd_scenpar.cpp` with `-Wall -Wextra -Wpedantic
  -Wshadow -Wconversion -Wsign-conversion` under g++ 12, 15 and 16: 0 warnings.
- What-if check of T2 [M]: a scratch test binary linked with a `qd_scenpar.cpp` whose stream is not imbued
  with the classic locale fails T2 under `en_DK.UTF-8` ("Value of 'DS_mu' is out of range." for `-6.28`).
  The classic-locale grammar case passes with the same binary. So T2 detects a locale-dependent parser.

### 4.2 Review probes [R]

Copied from the review's scratch folder. `build.sh` was pointed at this worktree and `build_linux/libquadriga.a`.

| Probe | Result |
|---|---|
| `grammar2.cpp` (= `grammar.cpp` with the Q2 oracle: an overflowing text, or the first bad path-loss token overflowing, must give "is out of range"; no other text may) | `cases 400102, accepted 42944, accept mismatches 0, value mismatches 0`; `overflowing texts 132, of which rejected with 'is out of range' 132, other outcome 0; 'out of range' for other texts 0; underflow-to-zero accepted 59` |
| `writer.cpp 30000` | `values 30021, round-trip failures 0, not minimal within 15..17 0`; digits 1837 / 14618 / 13564 (as in the review); `/dev/full` both rows "Cannot open file '/dev/full' for writing."; read-only file kept; read-only directory and directory as `fn` throw; duplicate short names and a line break in the file name throw |
| `lines.cpp` | as in the review, except the decided changes: BOM before a comment or a key accepted; `ShortName` twice "line 2: ShortName 'Ab' is stated twice."; exponent overflow "is out of range"; outputs untouched after 4 of 4 failed calls |
| `lister2.cpp` (= `lister.cpp` with a `try` around its `conf_file_read` of a file with a repeated short name, which now throws by Q3) | as in the review; that file: lister "ShortName 'As' is ambiguous.", reader "ShortName 'As' is stated twice." |
| `locale_probe.cpp`, `LOCPATH` with the `localedef`-built `de_DE.UTF-8` | `read and written text identical in both locales: yes (16240 bytes)` |
| `partial_run.sh` (B1) | the two review rows under a 2 KiB limit (3407 B and the `write_defaults` text) now throw "Cannot write file", old content kept (44 bytes), no temporary file; texts that fit (665 B) are written |
| `xlang/dump_m.m` (MATLAB, Octave), `dump_py.py` | 24 files (23 v19 files and `zz_tricky.conf`), the three dumps byte-identical (md5 `1b448090c08dcb3f8b89546c355aee36`), and identical to the review's dumps |

### 4.3 Coverage of `qd_scenpar.cpp` [M]

gcov 15, `-O0 --coverage`, the 48 conf cases.

| | Review (145f163) | Now |
|---|---|---|
| Lines executed | 99.31 % of 433 | 99.18 % of 485 |
| Branches executed | 91.4 % | 91.22 % of 1458 |
| Branches taken at least once | 67.6 % | 67.22 % of 1458 |

| Line | Code | Reason not executed |
|---|---|---|
| 439 | closing brace of `qd_conf_tokens` (`=====`) | exception landing pad, only on `bad_alloc` (as in the review) |
| 1488 | `const std::string rule = ...` | gcov attribution artefact: the next line uses `rule` and runs (as in the review) |
| 1543, 1544 | `remove(tmp)`, `throw "Cannot write file"` after a failed `rename` | the temporary file is in the directory of the target and the target was checked to be a regular file, so `rename` fails only if the target is replaced by a directory between the check and the rename (a race) or on a file system that refuses a rename; not reachable in a test |

Old line 1405 (the write error) is now lines 1533–1534 and runs twice (T1). The symbolic-link error (line
1503) runs through a link to itself (added to T1 after the first coverage run).


## 5. Changed files

`git diff --numstat` against 145f163 [M]. The log itself (`md/LOG_CONF_FILE_FIXES.md`) is new.
`md/PLAN_CONF_FILE_FIXES.md` and `md/REVIEW_CONF_FILE.md` are untracked and unchanged, and
`data/config` is unchanged.

| File | Added | Removed |
|---|---|---|
| `src/qd_scenpar.cpp` | 209 | 69 |
| `src/mex_helper_functions.hpp` | 31 | 31 |
| `src/mex_quadriga_lib_functions.hpp` | 4 | 3 |
| `include/quadriga_tools.hpp` | 2 | 1 |
| `api_mex/conf_file_read.cpp` | 20 | 12 |
| `api_mex/conf_file_write.cpp` | 18 | 10 |
| `api_mex/conf_file_list.cpp` | 10 | 7 |
| `api_mex/scenpar_validate.cpp` | 9 | 7 |
| `api_python/python_quadriga_adapter.hpp` | 41 | 0 |
| `api_python/qpy_conf_file_read.cpp` | 18 | 10 |
| `api_python/qpy_conf_file_write.cpp` | 23 | 14 |
| `api_python/qpy_conf_file_list.cpp` | 29 | 15 |
| `api_python/qpy_scenpar_validate.cpp` | 8 | 5 |
| `tests/catch2_tests/test_conf_file_read.cpp` | 174 | 2 |
| `tests/catch2_tests/test_conf_file_write.cpp` | 282 | 5 |
| `tests/catch2_tests/test_conf_file_list.cpp` | 77 | 8 |
| `tests/catch2_tests/test_scenpar_validate.cpp` | 30 | 0 |
| `tests/mex_tests/test_conf_file_read.m` | 27 | 0 |
| `tests/mex_tests/test_conf_file_write.m` | 15 | 0 |
| `tests/mex_tests/test_scenpar_validate.m` | 16 | 0 |
| `tests/python_tests/test_conf_file_read.py` | 64 | 0 |
| `tests/python_tests/test_conf_file_write.py` | 35 | 0 |
| `tests/python_tests/test_conf_file_list.py` | 31 | 0 |
| `tests/python_tests/test_scenpar_validate.py` | 20 | 0 |
| `md/conf_file_parser.md` | 40 | 0 |
| `html_docu/` (6 generated files) | 370 | 302 |
| Total (31 files) | 1603 | 501 |

`src/qd_scenpar.cpp`: 1406 lines before, 1546 after. The `python_tools.cpp` registration is unchanged: names
and defaults stay as they were (`py::arg("validate") = true` and so on); only the C++ parameter types became
`py::handle` / `py::object`.


## 6. Known limitations kept, and questions

Kept as stated in plan section 1:
- The sanitizer reports in other functions (review section 0). Not re-run here; the complete suite under ASan
  still stops in `baseband_freq_response`.
- Q9: on Windows, narrow `std::string` paths use the ANSI code page. This also applies to the new temporary
  file name (`target.string()`).
- `qd_mex_get_scalar` semantics (first element of `[1 0]`, `NaN` read as `true`): unchanged; only the
  identifier changed.
- A short name equal to the scenario name is rejected by `conf_file_list` only.

New, from this work:
- F1 replaces the file, so the inode changes: hard links to the old file keep the old content, owner and group
  become the writer's, ACLs and extended attributes are not copied (only the permission bits). The directory
  must be writable, even for an existing writable file. Writing to a device or FIFO (`/dev/stdout`) is an
  error (section 2.1).
- MSVC is still not checked (review Q2): with the stream conversion kept, a `num_get` that fails on the
  `ERANGE` of an underflow would report a non-zero subnormal as "is not a number". On libstdc++ it reads as
  `strtod` gives it [M].

Questions for Stephan, answered on 2026-10-14:
1. Keep the F1 extension that rejects an existing target that is not a regular file? The plan only names
   directories. The extension protects device nodes from being replaced by a privileged writer.
   **Answer: keep it.**
2. With the stream conversion kept, should Q2 count as closed? The alternative that keeps `from_chars`
   (`-static-libstdc++` for the MATLAB MEX link in `CMakeLists.txt`) would affect every MEX file.
   **Answer: Q2 is closed as is.**
