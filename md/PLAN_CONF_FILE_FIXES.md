# Plan: fixes after the review of the conf file parser

Written 2026-10-14 from `md/REVIEW_CONF_FILE.md` (review of commits 338eaa0..145f163, branch `conf-file-parser`, HEAD
145f163). All open points were decided by Stephan on 2026-10-14 (section 2). This plan is the complete brief for one
fixing session that runs on its own. Nothing in it is implemented yet.

## 0. How to work

- Worktree: `/sjc/quadriga-lib-conf`, branch `conf-file-parser`. Run every command from there.
- Read first: this plan; `md/REVIEW_CONF_FILE.md` (the findings B, Q, T, D referred to below); `md/SPEC_CPP_API.md`,
  `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md` (style, docs, test rules); `md/conf_file_parser.md` (the
  original plan; its decisions stay binding where this plan does not change them).
- Order: tests first. For each work item write the failing test, run it against the unchanged code, confirm it fails
  for the right reason, then change the source, then run it green. Keep a short note per item for the log.
- The review points A, B and C of the specs are waived for this work: do not stop to report between phases. Stop
  and ask only if a decision in this plan cannot be implemented as written, two decisions conflict, or a measurement
  contradicts this plan (for example a baseline count that differs before any change).
- Do not commit, do not push. Stephan commits himself. Do not edit `md/REVIEW_CONF_FILE.md`. Do not change
  `data/config/*.conf`. QuaDRiGa-NG (`/sjc/QuaDRiGa-NG`) is read-only and no test may depend on it.
- Scratch programs and throw-away builds go into the session scratch folder, not into the worktree. Builds in
  `build_*` folders of the worktree are fine (ignored by git).
- Measure, do not compute by hand: every number in the log comes from a run in this session.
- At the end write `md/LOG_CONF_FILE_FIXES.md` (section 6) and stop.

### Build and run (gcc 12, as the GNUmakefile; no AVX2)

```
cmake -B build_linux -D CMAKE_INSTALL_PREFIX=. -D CMAKE_C_COMPILER=gcc-12 -D CMAKE_CXX_COMPILER=g++-12 \
  -D ENABLE_MATLAB=ON -D ENABLE_OCTAVE=ON -D ENABLE_MEX_DOC=ON -D ENABLE_PYTHON=ON -D ENABLE_TESTS=ON \
  -D ENABLE_AVX2=OFF -D ENABLE_CUDA=OFF -D ENABLE_STATIC_LIB=ON -D ENABLE_SHARED_LIB=OFF
cmake --build build_linux --parallel 24 && cmake --install build_linux
```

- Catch2, new cases: `build_linux/test_bin "conf_file_read*,scenpar_validate*,conf_file_list*,conf_file_write*,conf files v19*"`
- Catch2, complete: `build_linux/test_bin`
- MEX: once `unzip -q -o external/MOxUnit.zip -d external/`; then
  `octave --no-gui --quiet --eval "cd tests; quadriga_lib_mex_tests;"` and
  `matlab24 -batch "disp(version); run('tests/quadriga_lib_mex_tests.m');"`. MATLAB uses the gcc 12 MEX files,
  without `LD_PRELOAD`. Do not run Octave and MATLAB at the same time (the tests share file names).
- Python: `python3 -m pytest tests/python_tests -x -s`
- Documentation: `make documentation` (regenerates `html_docu/`; the changed files are part of the result).
- Sanitizers (new cases only): a second build folder `build_asan` with `g++-15`,
  `-D "CMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined"`,
  `-D "CMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined"`, MEX and Python OFF, tests ON; run the new cases with
  `ASAN_OPTIONS=detect_leaks=1`. The complete suite under ASan aborts in `baseband_freq_response` (known, out of
  scope, review section 0), so only the new cases are required to be clean.
- Coverage: a third folder `build_cov` with `g++-15`, `-D "CMAKE_CXX_FLAGS=--coverage -O0"`,
  `-D "CMAKE_EXE_LINKER_FLAGS=--coverage"`; run the new cases, then
  `gcov-15 build_cov/CMakeFiles/quadriga_static.dir/src/qd_scenpar.cpp.o` in a scratch folder.

### Baseline at 145f163 (measured in the review; confirm before changing anything)

| Suite | Result |
|---|---|
| Catch2, new cases | `All tests passed (7949 assertions in 39 test cases)` |
| Catch2, complete | `All tests passed (1300843 assertions in 1108 test cases)` |
| MEX, Octave 11.1.0 and MATLAB R2024a, complete | 65 of 66; `test_ray_mesh_interact` line 113 fails (known, not part of this work) |
| Python, complete | `752 passed, 3 skipped` |
| Warnings (library, MEX, Python) | none |

If the baseline differs, stop and report.

### Scratch probes of the review

The review's probe programs are in
`/tmp/claude-1005/-sjc-quadriga-lib-conf/1e0f27ea-e81e-4c0b-920b-87eae9205036/scratchpad/` (`probe/` and `xlang/`).
Copy what you use into your own scratch folder. If the folder is gone, recreate the probes from the descriptions
in review section 1 ("Checked and found correct"). `probe/build.sh` links a probe against
`build_*/libquadriga.a` (`-DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK`, hdf5 serial, `-fopenmp`). Re-run at the end:
`grammar.cpp` (number grammar fuzz against a regex and `strtod`), `writer.cpp` (round trip of 30,000 doubles),
`lines.cpp`, `lister.cpp`, `locale_probe.cpp` (needs `LOCPATH` with a `localedef`-built `de_DE.UTF-8`, or use the
installed `en_DK.utf8`, decimal comma), `partial.cpp` with `partial_run.sh` (B1), and `xlang/dump_m.m` plus
`xlang/dump_py.py` (MATLAB, Octave and Python must give byte-identical dumps).


## 1. Scope

Everything that belongs to the conf file parser: `src/qd_scenpar.cpp`, the declarations in
`include/quadriga_tools.hpp` (only comments may change), `api_mex/conf_file_*.cpp`, `api_mex/scenpar_validate.cpp`,
the scenpar/plpar helpers in `src/mex_quadriga_lib_functions.hpp`, the error identifiers in
`src/mex_helper_functions.hpp` (decision Q5), `api_python/qpy_conf_file_*.cpp`, `api_python/qpy_scenpar_validate.cpp`,
`api_python/python_tools.cpp` (registration), the converters in `api_python/python_quadriga_adapter.hpp`, the test
files `tests/catch2_tests/test_conf_file_*.cpp`, `test_scenpar_validate.cpp`, `tests/mex_tests/test_conf_file_*.m`,
`test_scenpar_validate.m`, `tests/python_tests/test_conf_file_*.py`, `test_scenpar_validate.py`, the generated
`html_docu/`, and `md/conf_file_parser.md` (record of changes).

Not in scope (recorded as known limitations in the log, no change):
- The sanitizer reports in other functions (review section 0; a separate task exists).
- Q9, Windows: narrow `std::string` paths are read in the ANSI code page by `std::ifstream` and `std::filesystem`;
  non-ASCII paths from Python or MATLAB (UTF-8) would fail on Windows. Library-wide, not testable here.
- The semantics of `qd_mex_get_scalar` (first element of `[1 0]`, `NaN` read as `true`) stay as they are; only the
  error identifier changes (decision of 2026-10-14).
- A short name equal to the scenario (file) name: `conf_file_list` rejects it; reader and writer are not changed for
  it.


## 2. Decisions (Stephan, 2026-10-14)

| # | Finding | Decision |
|---|---|---|
| B1 | Failed writes undetected or leaving a truncated file | Write to a temporary file in the same directory, close and check it, then rename it over the target. A failed call never changes an existing file and leaves no file behind. |
| B2 | Line break in the file name breaks the header | `std::invalid_argument` before anything is written. |
| B3 | Python `None` for a bool flag is `False` | `None` selects the documented default; `bool` and `numpy.bool_` are accepted; anything else is a `TypeError`. For the four conf wrappers. |
| Q1 | UTF-8 byte order mark is an error | Skip `EF BB BF` at the start of the file (reader and lister); document; test. |
| Q2 | `1e400` "is not a number", `1e-400` silently 0 | Overflow is an error "is out of range"; underflow gives what `strtod` gives in the C locale (a subnormal or a signed 0), so the writer's subnormals keep reading back on every compiler. Parse with `std::from_chars`. Documented and tested. |
| Q3 | A short name stated twice in one file | `conf_file_read` rejects a repeated `ShortName` value (grammar level, both `validate` modes); `conf_file_write` rejects duplicate entries in `short_names`. The lister stays as it is (it already throws). |
| Q4 | Python rejects `pathlib.Path` | `fn` and each entry of `dirs` may be `str`, `bytes` or `os.PathLike` (converted with `os.fspath`). For the four conf wrappers. |
| Q5 | Shared MEX helpers raise `MATLAB:unexpectedCPPexception` | Change the identifier to `quadriga_lib:CPPerror` in `src/mex_helper_functions.hpp` (31 places; the only file that uses it; no test matches on it). Library-wide effect; the complete MEX suite must pass in Octave and MATLAB. |
| Q6 | MEX accepts a matrix for a `plpar` field | `qd_mex_struct2plpar` rejects anything that is not empty, a scalar, a row or a column vector. |
| Q7 | Lister accepts any file name as scenario name | Keep; document that the scenario name is the file name without `.conf`, whatever characters it has. |
| Q8 | Unlistable directory skipped silently | Keep; document (D5). |
| Q10 | ESD overview comment of a written file is incomplete | Complete it (section 3, F7). |
| Q11 | Counts without upper bound | `NumClusters`, `NumSubPaths`: integer from 1 to 1000. `NumClusters_min`, `NF_k1`: integer from 0 to 1000. The shipped files stay valid (largest values: QuaDRiGa 26, v19 26 / 16 / 4). |
| Q12 | Tests share a temp file name | Use a name per test case. |
| D1-D12 | Documentation | As in section 4. |


## 3. Work items, C++ (`src/qd_scenpar.cpp`)

Line numbers refer to 145f163.

### F1 (B1) Safe write in `conf_file_write`

Replace lines 1399-1405 (open with `trunc`, `<<`, `good()`). After the text is composed:

1. `target = fn`. If `fn` is a symbolic link, `target = std::filesystem::weakly_canonical(fn)` (write through the
   link, as today).
2. If `target` exists: if it is a directory, or opening it with `std::ios::out | std::ios::app` fails (read-only
   file), throw `std::runtime_error("Cannot open file '" + fn + "' for writing.")`. The `app` open does not change
   the file. Do not create `target` here when it does not exist.
3. Temporary name in the same directory: `target` + `".tmp"` + a hex token built from
   `std::chrono::steady_clock::now().time_since_epoch().count()` and `std::hash<std::thread::id>` of the current
   thread. No global state.
4. Write the text with `std::ofstream(tmp, binary | trunc)`. If it does not open: throw the "Cannot open file ...
   for writing." error (no temp file exists then). Then `file << out; file.close();` and if `file.fail()`:
   remove the temp file (`std::filesystem::remove(tmp, ec)`), throw `std::runtime_error("Cannot write file '" + fn
   + "'.")`.
5. If `target` existed, copy its permissions to the temp file (`std::filesystem::permissions(tmp,
   status(target).permissions(), ec)`; ignore `ec`).
6. `std::filesystem::rename(tmp, target, ec)`; on error remove the temp file and throw "Cannot write file".

Expected results: the four rows of review B1 now all throw `runtime_error`, the old content stays, no `*.tmp*`
file is left; `/dev/full` throws (the temp file cannot be created in `/dev`); a read-only existing file throws and
keeps its content; a read-only directory throws; an existing file is replaced (its permissions kept).

### F2 (B2) Line break in the file name

In the validation block (before any text is composed), compute the scenario name as today (lines 1216-1218) and
throw `std::invalid_argument("Input 'fn' must not contain a line break in the file name.")` if it contains `'\n'`
or `'\r'`.

### F3 (Q1) Byte order mark

In `qd_conf_read_lines` (line 377): if the first line starts with `"\xEF\xBB\xBF"`, remove these three bytes. Only
at the start of the file; a BOM later in the file stays an error. This covers the reader and the lister.

### F4 (Q2) Number conversion

`qd_conf_parse_number` (line 439): keep the hand-written grammar scanner unchanged. Replace the `istringstream`
conversion by `std::from_chars` (`<charconv>`) on the text without a leading `+` (`from_chars` does not accept
`+`). `Inf` stays handled before. Change the return type so that the caller can tell "not a number" from
"out of range" (for example an enum or an `int` with three states).

- `ec == std::errc{}` and the whole text consumed: accept.
- `ec == std::errc::result_out_of_range`: classify by the decimal order of the first non-zero digit (its position
  relative to the decimal point plus the exponent). Order > 0: overflow, error. Otherwise: underflow, the value is
  `+0.0` or `-0.0` by the sign (that is what `strtod` returns when the result rounds to 0).
- First measure what libstdc++ 12 and 15 return for `4.9e-324`, `2.4703282292062328e-324` (rounds up to the
  smallest subnormal), `2.4703282292062327e-324` (rounds to 0), `1e-310`, `1e-400`, `-1e-400`, `1e400`, and compare
  the bits with `strtod` in the C locale. If `from_chars` reports `result_out_of_range` for a text whose `strtod`
  result is a non-zero subnormal, use this fallback instead and note it in the log: keep `istringstream` with the
  classic locale for the conversion and add only the overflow classification (`failbit` after a text that passed
  the grammar is overflow when the order is > 0).
- Messages: reader `"<fn>, line <n>: Value of '<key>' is out of range."` (scenario and path-loss keys);
  `"... is not a number."` stays for grammar errors.
- The writer's read-back check (`qd_conf_number`, line 617) uses the same function; every finite double must still
  round-trip (re-run `writer.cpp`).
- Re-run `grammar.cpp` with the new oracle: overflow texts expected as "out of range", all other texts as before.

### F5 (Q3) Repeated short names

- `conf_file_read` (line 772): keep a set of the short names of the file; a value seen before throws
  `std::invalid_argument(where + "ShortName '" + value + "' is stated twice.")`, in both `validate` modes.
- `conf_file_write` (line 1196): duplicate entries in `short_names` throw
  `std::invalid_argument("Entries of 'short_names' must be distinct.")`.

### F6 (Q11) Bound 1000 for the counts

Classes `QD_INT_POS` and `QD_INT_NONNEG` (lines 246-249) get the upper bound 1000; their messages become
`"must be an integer from 1 to 1000"` and `"must be an integer from 0 to 1000"`. Only `NumClusters`,
`NumSubPaths` (`QD_INT_POS`) and `NumClusters_min`, `NF_k1` (`QD_INT_NONNEG`) use these classes; check that this is
still true. Update the comments of the enum.

### F7 (Q10) Overview formula of a written file

In the overview block (lines 1255-1267 and 1347-1358):

- If `LSP_fc_min > 0`, the frequency variable is written as `max( fGHz, <fc_min> )` in every overview line, in the
  mean and in the STD term (e.g. `log10( 0.5 + max( fGHz, 6 ) )`).
- ESD line: after the existing terms of the mean, append with the same sign style as `add`:
  `± <A> * d2D / 1000` (`ES_D_mu_A`), `± <B> * hUT` (`ES_D_mu_B`), `± <C> * abs( hUT - hBS )` (`ES_D_mu_C`),
  `± <D> * max( hUT - hBS, 0 )` (`ES_D_mu_D`), each only if non-zero. If `ES_D_mu_min` is not `-Inf`, the mean is
  written as `max( <mu_min>, <mean> )`. The STD term stays as it is.
- Example from the layout test (`ES_D_mu = 0.9`, `ES_D_mu_A = -2.1`, `ES_D_mu_min = -0.5`, `ES_D_mu_B = -0.01`,
  `LSP_fc_min = 6`): `% ESD = max( -0.5, 0.9 - 2.1 * d2D / 1000 - 0.01 * hUT ) + Xesd * ( 0.2 )`. Type the
  expected lines of the layout test by hand; do not take them from the output.
- These are comments; `*` is fine in the conf file text (the `*`/`|` rule applies to documentation blocks only).

### F8 (D12, cleanup) Constant tables and a dead branch

- `qd_conf_sections`, `qd_conf_lsp`, `qd_conf_pl_comments`, `qd_conf_pl_formulas`: declare as
  `static const char *const ...` so that no static state is mutable.
- `qd_scenpar_check`, lambda `where` (line 264): the branch `ln == 0 ? fn + ": "` is unreachable from the reader
  (review 3.1). Simplify to `fn.empty() ? "" : fn + ", line " + std::to_string(ln) + ": "` only after checking that
  every call with a non-empty `fn` passes a line > 0; if not, keep it.


## 4. Work items, wrappers and documentation

### F9 (Q5) MEX error identifier

Replace every `"MATLAB:unexpectedCPPexception"` in `src/mex_helper_functions.hpp` by `"quadriga_lib:CPPerror"`.
Grep the whole repository for the old identifier afterwards (tests, api_mex, tools); nothing may remain except
in `md/` texts. Run the complete MEX suite in Octave and MATLAB.

### F10 (Q6) MEX `plpar` fields are vectors

`qd_mex_struct2plpar` in `src/mex_quadriga_lib_functions.hpp`: reject a field with more than 2 dimensions or with
more than one row and more than one column, message unchanged `"Field '<name>' must be a real numeric vector."`.
Update the helper comment ("read as a vector in memory order").

### F11 (B3) Python flags

Add a helper to `api_python/python_quadriga_adapter.hpp`, e.g.
`static bool qd_python_flag(const py::handle &value, bool default_value, const std::string &name)`: `None` gives
the default; Python `bool` and `numpy.bool_` give their value; anything else throws `py::type_error("Input '<name>'
must be a bool or None.")`. The wrappers take the flags as `py::object` (or `py::handle`): `validate`
(`conf_file_read`, `scenpar_validate`, default `True`), `parse_shortnames` (`conf_file_list`, default `True`),
`write_defaults` (`conf_file_write`, default `False`). Registration in `python_tools.cpp` and the comment block in
each wrapper keep the same names and defaults (`py::arg("validate") = true`). Note: integers `0`/`1` were accepted
before and are now a `TypeError`; check that no existing Python test passes an integer flag (adapt only tests of
the four conf wrappers if one does, and say so in the log).

### F12 (Q4) Python path-like inputs

Add a helper, e.g. `static std::string qd_python_path(const py::handle &value, const std::string &name)`: `str`
and `bytes` as today; an object with `__fspath__` is converted with `os.fspath` and then as `str`/`bytes`; anything
else throws `py::type_error("Input '<name>' must be a str or path-like object.")`. Use it for `fn` of
`conf_file_read` and `conf_file_write` (signature `py::object fn`). For `dirs` of `conf_file_list`: `None` gives an
empty list (as today), a `str`, `bytes` or path-like is one directory, a `list` or `tuple` is converted entry by
entry; other types keep the current `ValueError` "Expected a string, a list of strings, or None." (an existing test
asserts it). The returned `file_dirs` stay `str` exactly as given (path-like converted with `os.fspath`).

### F13 Documentation

Apply D1 to D12 of review section 4, in all three layers where the review says so, and document the new
behaviour. Keep the spec rules: no `*` or `|` in text, C++ and Python lines up to 120 characters, MEX lines up to
100 (D8 fixes the six long ones; check all four MEX blocks again with a script).

- D1 (all three `conf_file_read` blocks): numbers beyond the range of a double (`1e400`) are an error "out of
  range"; a number below the smallest subnormal reads as `0` (or `-0`).
- D2 (all three `conf_file_write`): "Numbers are written with up to 15 significant digits, or with 16 or 17 where 15
  do not read back to the same double in [[conf_file_read]]".
- D3 (all three `conf_file_write`): "A failed call does not create a file and leaves an existing file unchanged; an
  existing file is replaced and keeps its permissions" (true after F1). Also say that a line break in the file
  name is an error (F2) and that duplicate short names are an error (F5).
- D4 (all three `conf_file_read`): "... if the file cannot be opened or is a directory".
- D5 (all three `conf_file_list`): "Lists the regular files and symbolic links to regular files whose name ends in
  `.conf` (case-sensitive) and is longer than `.conf`, not recursive. The scenario name is the file name without
  this extension." and "A directory that does not exist, is not a directory or cannot be listed is skipped".
- D6 (`CONF_FILE_READ`, MEX): add "- [[conf_file_write]] (write a conf file)" to See also.
- D7 (all three `scenpar_validate`): add "- [[conf_file_write]] (applies the same checks before writing)".
- D8 (MEX): rewrap `conf_file_read.cpp` block lines 16, 45, 52; `scenpar_validate.cpp` 48; `conf_file_list.cpp` 13,
  31 to at most 100 characters.
- D9 (C++ `conf_file_read`): "- Numbers are read independent of the global locale (the decimal point is always
  `.`)"; MEX and Python may say the same.
- D10: C++ `conf_file_read` outputs `short_names` "...; `[n_short]`", `ignored_keys` "...; `[n_ignored]`";
  MEX `[n_short, 1]`, `[n_ignored, 1]`.
- D11 (`md/conf_file_parser.md`): add a section "Record of later changes" after section 3 (do not rewrite the old
  sections): `GR_epsilon` dropped in 3690107 (138 + 27 = 165 numeric keys, 166 struct fields; in a file the key is
  ignored and reported, in `scenpar_validate` it is an unknown name); `conf_file_write` added in e525c7b; MEX
  wrappers (3690107, ad7a9a4) and Python wrappers in `quadriga_lib.tools` (145f163); the parity test removed
  (already noted in WI-5); and the decisions of this plan (B1 to Q12 of section 2).
- D12: see F8.
- New behaviour to document: BOM skipped (reader and lister); repeated `ShortName` is an error (reader: "can occur
  several times with different values"); counts limited to 1000 (`scenpar_validate` class list in all three
  layers); Python flags accept `None` for the default ("default: True; `None` selects the default"); Python `fn`
  and `dirs` accept path-like objects; MEX `plpar` fields must be vectors; the ESD overview line of a written file.
- Header comments in `quadriga_tools.hpp`: update only if a fact there changes (e.g. the `conf_file_write` comment
  "an existing file is overwritten" → "an existing file is replaced; a failed call leaves it unchanged").
- Run `make documentation` and check that `html_docu/` changed only in the conf entries.


## 5. Tests (failing first, then green)

Use distinctive inputs, one behaviour per case or section, as in the existing files. Every Catch2 case name is
unique across the suite. New temp file names contain the case (Q12).

| # | File | Test | Covers |
|---|---|---|---|
| T1 | `test_conf_file_write.cpp` | New case "conf_file_write - a failed write leaves an existing file unchanged": guarded by `#if defined(__linux__)`: an existing file with known content; `signal(SIGXFSZ, SIG_IGN)` and `setrlimit(RLIMIT_FSIZE)` with a soft limit of 2048 bytes (save and restore both, and restore before any `CHECK`, `INFO` or `QD_DIAG`, since the limit also applies to a redirected stdout); write a 3.4 KB text (60 path-loss keys) and a 16 KB text (`write_defaults = true`); expect `std::runtime_error` both times, the old content unchanged and no file matching `*.tmp*` in the directory. Also `/dev/full` (if it exists) throws `std::runtime_error`. A read-only existing file (`std::filesystem::permissions`, skipped when running as root) throws and keeps its content. An existing file keeps its permission bits after a successful write. | B1, line 1405 |
| T2 | `test_conf_file_write.cpp` | New case "conf_file_write - numbers do not depend on the global locale": try `std::locale` names `de_DE.UTF-8`, `de_DE.utf8`, `en_DK.UTF-8`, `en_DK.utf8`, `German_Germany.1252`; if none exists, `SKIP`. Set `std::setlocale(LC_ALL, ...)` and `std::locale::global(...)`, read a file with `-6.28`, `1.5e3`, `.25`, write it back, compare bits and the written text with a run in the classic locale; restore both locales (also on failure). `en_DK.utf8` (decimal comma) is installed on this machine, so the case must run here, not skip. | T2 |
| T3 | `test_conf_file_read.cpp` | In "input validation / file cannot be opened": a temporary directory as `fn` throws `std::runtime_error`. | T3 |
| T4 | `test_conf_file_write.cpp` | In "input validation / fn": `"test_conf_file_write_x\nKF_mu = 5 %.conf"` and the variant with `\r` throw `std::invalid_argument` and create no file. | B2 |
| T5 | `test_conf_file_read.cpp` | Numbers: `1e400`, `-1e400`, `1e99999999999999999999`, `PL_A = 1 1e400` throw `std::invalid_argument` with "out of range" in the message (both `validate` modes); `1e-400` reads as `+0` and `-1e-400` as `-0` (check the sign bit); `4.9e-324` and `2.4703282292062328e-324` read as the smallest subnormal; `2.4703282292062327e-324` reads as 0; `1.7976931348623157e308` reads as the largest double; `1.8e308` is out of range. | Q2 |
| T6 | `test_conf_file_list.cpp` | New section: a listed file without read permission (POSIX only, skipped as root): `std::runtime_error` with `parse_shortnames = true`, listed without error with `false`; outputs untouched on the error. | T6 |
| T7 | `test_conf_file_list.cpp` | New section: a symlink to a regular `.conf` file is listed and its short names are read; a dangling symlink and a symlink to a directory are skipped. Skip the section if `create_symlink` fails (Windows without privilege). | T7 |
| T8 | `test_conf_file_read.cpp`, `test_conf_file_list.cpp` | BOM: `"\xEF\xBB\xBF% c\nDS_mu = -7\n"` and `"\xEF\xBB\xBFDS_mu = -7\n"` read `DS_mu = -7` with no ignored key; a BOM at the start of line 2 is an error with "line 2"; the lister reads a `ShortName` in line 1 after a BOM. | Q1 |
| T9 | `test_conf_file_read.cpp`, `test_conf_file_write.cpp` | Repeated short name: `"ShortName = Ab\nShortName = Ab\n"` throws `std::invalid_argument` with "line 2" in both modes; distinct names still pass; `conf_file_write` with `{"Ab", "Ab"}` throws and creates no file. | Q3 |
| T10 | `test_scenpar_validate.cpp`, `test_conf_file_read.cpp`, `test_conf_file_write.cpp` | Bound: 1000 accepted, 1001 rejected for `NumClusters`, `NumSubPaths`, `NumClusters_min` (with `NumClusters = 1000`), `NF_k1`; also through a file (with line number) and in the writer. Extend the class sections. | Q11 |
| T11 | `test_conf_file_write.cpp` | Layout test: the expected text gets the new ESD line and the `max( fGHz, 6 )` terms, typed by hand. One more small case: `ES_D_mu_C`, `ES_D_mu_D` non-zero and `ES_D_mu_min = -Inf` give no `max( ..., )` wrapper and the two abs/max terms. | Q10 |
| T12 | `test_conf_file_read.cpp` | Replace the shared name `test_conf_file_read_tmp.conf` of `read_text` by a name that includes a per-case tag (e.g. a second parameter or `Catch::getResultCapture().getCurrentTestName()`); check the other three files for shared names too. | Q12 |
| T13 | `tests/mex_tests/test_conf_file_write.m` | `struct('A', [1 2; 3 4])` as `plpar` raises "must be a real numeric vector."; a row and a column vector still work. | Q6 |
| T14 | `tests/mex_tests/test_conf_file_read.m` (and one more conf test) | A helper-raised error carries the identifier: `quadriga_lib.conf_file_read(5)` fails with `ME.identifier` equal to `'quadriga_lib:CPPerror'` (works in MATLAB and Octave). Write a small local function for the identifier check; keep `assert_throws`. | Q5 |
| T15 | `tests/python_tests/test_conf_file_read.py`, `test_scenpar_validate.py`, `test_conf_file_list.py`, `test_conf_file_write.py` | `None` for each flag gives the same result as omitting it (`conf_file_read(fn_bad, None)` raises the `ValueError` of `r_DS`); `numpy.bool_(False)` works; `"yes"` and `1` raise `TypeError`. | B3 |
| T16 | Python, same files | `pathlib.Path` for `fn` (read and write) and for `dirs` (alone and in a list) gives the same result as `str`; `file_dirs` returns the `str` form; an `int` for `fn` raises `TypeError`. | Q4 |
| T17 | MEX and Python write tests | Line break in the file name raises (C++ error passed through); duplicate short names raise. | B2, Q3 |
| T18 | MEX and Python read tests | A file with a BOM reads; an out-of-range number raises with "out of range". | Q1, Q2 |

After the C++ changes, re-run the review probes (section 0) and record: grammar fuzz mismatches (expected 0 with
the new oracle for overflow), writer round-trip failures (expected 0), `partial_run.sh` (all rows throw, content
kept, no temp file), locale probe identical, MATLAB/Octave/Python dumps byte-identical (regenerate the dumps; the
tricky file of `xlang/` must still read, it has no repeated short name).


## 6. Finish and log

All of this must hold before the log is written:

- New Catch2 cases green; complete Catch2 suite green (expected 1108 + the new cases); new cases clean under
  ASan + UBSan; coverage of `qd_scenpar.cpp` re-measured, every unexecuted line listed with a reason (line 1405 must
  now be executed, or its successor in the new code).
- MEX complete suite in Octave and MATLAB: 65 + new of 66 + new, only `test_ray_mesh_interact` line 113 failing.
- Python complete suite: 752 + new passed, 3 skipped.
- No compiler warning in the library, the MEX files (compile the four conf MEX files and one other by hand with
  `g++-12 -std=c++17 -Wall -Wextra -Wpedantic`, as in review section 0) and the Python module.
- `make documentation` run; doc lint of the twelve blocks: no `*`/`|` in text, line limits met.
- `git status` shows only files of section 1 (plus the log and regenerated `html_docu/`).

`md/LOG_CONF_FILE_FIXES.md`, in the form of `md/REVIEW_QD_SOS_V2.md` (marks [M], [R], [C]):

1. Baseline before the change (the table of section 0, measured).
2. One row per decision of section 2 and per T and D item: what was changed (file and function), the test that
   pins it, and its state; anything not done, with the reason.
3. The F4 measurement of `from_chars` against `strtod` and which path was taken.
4. Final results with the actual summary lines of every run, the coverage table, the probe results.
5. Changed files with added and removed line counts (`git diff --stat`), and the lines of `qd_scenpar.cpp` before
   and after.
6. Known limitations kept (section 1, "Not in scope") and any new question for Stephan.

Then stop. Stephan reviews the log in the review session and commits.
