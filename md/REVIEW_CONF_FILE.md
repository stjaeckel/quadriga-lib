# Review of the scenario configuration file implementation

Reviewed on 2026-10-14, branch `conf-file-parser` at commit 145f163 ("Python wrappers for the conf file parser, and
documentation regenerated"), the five commits 338eaa0..145f163 (`git diff 114e9eb..HEAD`). Scope: the C++ core in
`src/qd_scenpar.cpp` and `include/quadriga_tools.hpp`, the four MEX and four Python wrappers with their helpers, and
the Catch2, MEX and Python tests. The values in `data/config/*.conf` were not reviewed; the files were used as input.
No source, test or data file was changed; all builds, probes and runs were made in a scratch copy of HEAD
(`git archive`), outside the worktree.

## Summary

- **Bugs (question 1).** Three, all reproduced; no bug in the parser, the validator or the lister:
  - B1 (medium): `conf_file_write` does not detect a failed write when the text fits the stream buffer, and on a
    detected failure it leaves a truncated file. Both times the previous content is gone and `conf_file_read`
    accepts the truncated file with missing values (28 of 60 path-loss keys; `XPR_mu` falls back to its default).
  - B2 (low): a line break in the file name goes into the header comment of the written file; the file then reads
    back a different value (`KF_mu = 5` instead of 0) or is rejected.
  - B3 (low): Python, `validate=None` (and `parse_shortnames=None`) is taken as `False`; validation is switched off
    without a message.
- **What holds** [R, M]: 200,102 fuzzed and table-driven number texts are accepted exactly when the documented
  grammar accepts them, with the value of `strtod`; 30,021 doubles (random bit patterns, subnormals, `-0`, max,
  `Inf`, 15/16/17-digit boundaries) round-trip bit-exact, each with the fewest digits in 15..17; MATLAB, Octave and
  Python return byte-identical results for the 23 v19 files and a hard file; parsing and writing give the same
  bits and text under `de_DE.UTF-8`; every failed call of the four functions leaves the outputs untouched; no
  static mutable state; the legacy field order equals QuaDRiGa's with `SubpathMethod` at field 107.
- **Coverage (question 2).** Good. The 39 Catch2 cases pin every documented C++ behaviour except two (the
  lister's `runtime_error` for an unreadable file, the writer's locale independence); four undocumented behaviours
  are untested as well. 99.3 % of the lines of `qd_scenpar.cpp` run; the only unexecuted reachable line is the
  write-error branch of B1. Gaps that
  matter: the write-failure path (T1), locale independence (T2), a directory as `fn` (T3). The wrapper tests meet
  their checklists item by item.
- **Documentation (question 3).** Correct in substance and consistent across the three layers; `html_docu/` is
  current. Improvements: overflow and BOM behaviour are undocumented, "A failed call does not create or change a
  file" is not true (B1), six MEX lines exceed 100 characters, two "See also" entries are missing, and the plan does
  not record the `GR_epsilon` removal and the writer (D1 to D12).
- **Questions** worth a decision: a UTF-8 byte order mark is an error although QuaDRiGa read such files (Q1);
  `1e400` is "not a number" while `1e-400` silently becomes 0 (Q2); a short name stated twice passes the reader and
  the writer but fails the lister (Q3).

Marks: **[M]** measured in this session, **[R]** reproduced with a scratch program or script, **[C]** read from the
code. Scratch programs are in the session scratch folder under `probe/` and `xlang/`; they are not part of the
repository.


## 0. Baseline

Machine: 32 cores. Compilers gcc 12.5.0 (library, MEX, Python, as in the GNUmakefile), gcc 15.2.0 (sanitizers,
coverage), Octave 11.1.0, MATLAB 24.1.0.3050227 (R2024a) Update 8, Python 3.14.4, numpy 2.3.5, pytest 9.0.2.
MATLAB was run with the MEX files built by gcc 12, without `LD_PRELOAD`.

| Build | Suite | Result [M] |
|---|---|---|
| gcc 12, no AVX2 | Catch2, the 39 new cases (`conf_file_*`, `scenpar_validate*`, `conf files v19*`) | `All tests passed (7949 assertions in 39 test cases)`, 0.12 s |
| gcc 12, no AVX2 | Catch2, complete | `All tests passed (1300843 assertions in 1108 test cases)`, 2 min 50 s |
| gcc 15, ASan + UBSan, leak check on | Catch2, the 39 new cases | all passed, no sanitizer report |
| gcc 15, ASan + UBSan | Catch2, complete | aborts at the first report, in `baseband_freq_response` (`invalid-aligned-alloc-alignment`); see below |
| gcc 15, `--coverage -O0` | Catch2, the 39 new cases | all passed; `qd_scenpar.cpp` 99.31 % of 433 lines, see section 3.1 |
| gcc 12, Octave 11.1.0 | MEX, the 4 new tests | `OK (passed=4)` |
| gcc 12, Octave 11.1.0 | MEX, complete | 65 of 66; `test_ray_mesh_interact` line 113 fails (0.98485 vs 0.69639) |
| gcc 12, MATLAB R2024a | MEX, the 4 new tests | `OK (passed=4)` |
| gcc 12, MATLAB R2024a | MEX, complete | 65 of 66; `test_ray_mesh_interact` line 113 fails, same numbers |
| gcc 12 | Python, the 4 new files | `28 passed` |
| gcc 12 | Python, complete (`pytest tests/python_tests -x -s`) | `752 passed, 3 skipped` |

- `test_ray_mesh_interact.m` line 113: unchanged state, it fails as before this work, in Octave and MATLAB alike.
- Complete Catch2 suite under ASan + UBSan: because the run aborts at the first report, every case was run in its
  own process [M]. 1085 of 1108 cases exit cleanly. 21 cases have sanitizer reports, all outside the scope and in
  code these commits do not touch: `Baseband Frequency Response` (2) and `Generate Arrayant - Multi-Beam` (1),
  `aligned_alloc` with a size that is not a multiple of the alignment (`src/baseband_freq_response.cpp:137`);
  `IEEE Chan` (5); `HDF - Minimal Test` (heap-buffer-overflow); `Get Channels IRS - Minimal test` (null pointer to
  a non-null argument, `get_channels_irs.cpp:342`); `ray_progress` (11, "reference binding to null pointer" in
  Armadillo `Mat_meat.hpp:7490`). The two remaining non-zero exits were `conf_file_read - value checks per
  validation class` and `- input validation`; they failed only because the per-case runner started them in
  parallel and both write `test_conf_file_read_tmp.conf` (see T9). Run one after the other they pass under ASan.
- Compiler warnings [M]: none. The library, the Octave MEX files and the Python module are compiled with
  `-Wall -Wextra -Wpedantic` by the build (0 lines with `warning:` in the gcc 12 build log, 0 in the gcc 15 sanitizer
  build). The MATLAB MEX files are built without warning flags (`CMakeLists.txt:531`); the four new ones compiled by
  hand with `g++-12 -std=c++17 -Wall -Wextra -Wpedantic` give no warning. `qd_scenpar.cpp` with
  `-std=c++17 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion` under g++ 12, 15 and 16: no warning.
  No clang and no MSVC were available (section 5).


## 1. Bugs

| # | Severity | Function | Finding |
|---|---|---|---|
| B1 | medium | `conf_file_write` | A failed write is not detected when the text fits the stream buffer, and a detected failure leaves a truncated file that `conf_file_read` accepts |
| B2 | low | `conf_file_write` | A line break in the file name breaks the header comment |
| B3 | low | Python wrappers | `None` for a bool flag is `False` |

### B1 A failed write is lost or leaves a truncated file

`conf_file_write` opens the file with `std::ios::trunc`, writes the text with `file << out` and checks
`file.good()` (`qd_scenpar.cpp:1400-1405`). Text shorter than the stream buffer (8 KiB in libstdc++) is still in the
buffer at the check; it is written by the destructor, whose error is lost. Longer text fails during `<<`, but the
file has been truncated and partly written by then.

Reproduction [R] (`probe/partial.cpp`, run by `probe/partial_run.sh` with `ulimit -f 2` and `SIGXFSZ` ignored, so
that `write` returns `EFBIG` after 2 KiB; the file holds 46 bytes of older content before each call):

| Call | Expected | Actual |
|---|---|---|
| `scenpar` with 3 keys, 60 path-loss keys, `write_defaults = false` (3407 bytes) | exception, old content kept | no exception; file truncated to 2048 bytes; `conf_file_read` accepts it and returns 28 of the 60 path-loss keys |
| same `scenpar`, `write_defaults = true` (about 16 KB) | exception, old content kept | `runtime_error "Cannot write file"`; file truncated to 2048 bytes; `conf_file_read` accepts it, `XPR_mu` = 0 instead of 8 |
| 3 keys to `/dev/full` (no space left on every write) | exception | no exception |
| `write_defaults = true` to `/dev/full` | exception | `runtime_error "Cannot write file '/dev/full'."` |

The C++ documentation promises only that a failed *check* changes no file, but the MEX and Python blocks state
"A failed call does not create or change a file", and all three say "Every file written passes conf_file_read":
here a file passes with wrong values. The cause is a full disk, a quota or a size limit; nothing in the input.

Fix: compose as now, write to a temporary file in the same directory, `close()` it and test `fail()`, then
`std::filesystem::rename` it over `fn`; remove the temporary file on any failure. The minimal fix (`file.close()`
before the check) reports the error but still leaves a truncated file.

### B2 A line break in the file name breaks the header

The scenario name in the header comment is the file name (`qd_scenpar.cpp:1216-1220`). Only `ref` and `ref_sec`
are checked for line breaks. Reproduction [R] (`xlang`, Python, the same C++ path):

| `fn` (file name part) | Expected | Actual |
|---|---|---|
| `"x\nKF_mu = 5 %.conf"`, `scenpar = {}` | file whose read-back equals the input (`KF_mu` = 0) | line 2 of the file is `KF_mu = 5 %"`; `conf_file_read` returns `KF_mu = 5.0` |
| `"x\nKF_mu = 5.conf"`, `scenpar = {}` | as above | `conf_file_read` throws `line 2: Value of 'KF_mu' is not a number.` |

Such names are legal on Linux and macOS. Fix: reject a file name with `\n` or `\r` (`std::invalid_argument`
before anything is written), or replace those characters in the header line.

### B3 Python: `None` for a bool flag means `False`

pybind11 converts `None` to `false` for a `bool` argument. Reproduction [R] (`xlang/probe_py.py`):

| Call | Expected | Actual |
|---|---|---|
| `conf_file_read(fn, None)`, file `r_DS = -1` | the default (`True`): `ValueError`, or a `TypeError` | returns `r_DS = -1.0`, no check |
| `scenpar_validate({"DS_mu": nan}, None)` | as above | returns `nan` |
| `conf_file_list(d, None)` | default `True` (short names listed) or `TypeError` | short names are not listed |

The Python spec lets `None` stand for the default of array inputs, so a user may expect the same for flags. The
behaviour is library-wide (every wrapper with a `bool` argument). Fix: declare the flags with
`py::arg("validate").noconvert() = true` (accepts `True`, `False`, `numpy.bool_`; rejects `None` and `0` with a
`TypeError`), or take a `py::object` and map `None` to the default.

### Checked and found correct

- Parser [R, `probe/grammar.cpp`]: 200,102 values (random strings of length 0 to 9 over digits, `.eE+-`, `iInNfF`,
  space, tab, `x`, `%`, `=`, plus 52 hand-picked texts: leading `+`, `5.`, `.5`, `1e-0`, `-0`, 400-digit integers,
  5000-digit mantissas, halfway subnormals `2.4703282292062327e-324`, `0x1p3`, `1_000`) for a scenario key and for
  a path-loss key. Accepted set equal to a `std::regex` of the documented grammar, values bit-equal to `strtod`:
  0 mismatches. The only texts outside the documentation are out-of-range numbers (Q2).
- Line level [R, `probe/lines.cpp`]: NUL in a comment is accepted, NUL in a key or value is an error; a 10 MB
  comment line, a value with 10^6 digits and a key with 10^6 characters work; no line end at EOF works; `=` twice,
  `==`, `=` after the value and `= 1` are errors; `%` directly after the value or the key behaves as a comment;
  `DS_m`, `DS_mu_`, `DS_mux` are unknown keys (no prefix match); duplicate `PL_` keys keep the last value and the
  first position; a directory, a missing, an empty and an unreadable (mode 000) `fn` throw `runtime_error`;
  `/dev/null` gives the defaults. CR-only and `\r\r\n` line ends, form feed, vertical tab and non-breaking space are
  errors (not white space by the documentation).
- Writer [R, `probe/writer.cpp`]: 30,021 values, 1837 written with at most 15 digits, 14,618 with 16, 13,564 with
  17; no round-trip failure; in no case would one digit fewer have read back; `std::to_chars` is never shorter where
  16 or 17 digits were used. `-0` stays `-0`, `5e-324` is `4.94065645841247e-324`, `DBL_MAX` is
  `1.7976931348623157e+308`. A `ref` with `%`, NUL or non-ASCII text, `ref_sec` with `%` and an empty entry, 301
  path-loss keys with 200,000 values (4.4 MB, exact round trip), an empty `plpar`, an existing file (overwritten), a
  read-only file (kept), a read-only directory and a directory as `fn` (`runtime_error`) all behave as documented.
- Validation [C, M]: every class boundary is pinned by the suite (section 3); the two cross-key checks are skipped
  with `validate = false`; NaN and `Inf` correlations give `false` from the Cholesky test (a NaN in any column
  reaches a later pivot) [C].
- Outputs untouched on failure [R, M]: `conf_file_read` for a cross-key, `NF_*`, `SubpathMethod` and grammar
  failure in the last line (4 of 4, all ten outputs including a 2 x 3 `lsp_xcorr`); `scenpar_validate`,
  `conf_file_list` and `conf_file_write` by the suite. All four functions compose into locals and write at the end
  [C].
- Lister [R, `probe/lister.cpp`]: a symlink to a regular file is listed (and its short names read); dangling
  symlinks, directories named `x.conf` and symlinks to directories are skipped; a directory `X.conf` in the first
  directory does not shadow a file `X.conf` in the second; a file named `.conf` is not listed; the order is byte-wise
  (` ` < `Z` < `_` < `b` < `z` < `~` < UTF-8 `été`); the same directory under two names (a symlink) adds nothing; an
  unreadable listed file throws `runtime_error` (with `parse_shortnames`) or is listed (without); aliasing `dirs`
  with `scenarios` or `file_dirs` gives the correct result.
- Wrappers: MATLAB, Octave and Python give byte-identical dumps (every value as hex, field order, `plpar`,
  `lsp_xcorr`, flag, short names, ignored keys) for the 23 files of `data/config` and a file with `-0`, `-Inf`,
  `INF`, `+.5E-3`, a subnormal, `DBL_MAX`, a 17-digit value and a four-value path-loss key [R, `xlang/`]. The field
  order equals QuaDRiGa's `struct(...)` call without `GR_epsilon`, `SubpathMethod` at field 107 [R]. Non-ASCII file
  names round-trip through write, read and list in MATLAB (UTF-8 on disk), Octave and Python [R]. Python outputs are
  new arrays (no shared memory between calls) and the inputs are not modified [R, M].
- Robustness [C, R]: no static mutable state (all tables `static const`; see D12 for the pointer arrays); no output
  is written before the checks; numbers are parsed and written with the classic locale: under
  `setlocale(LC_ALL, "de_DE.UTF-8")` plus `std::locale::global(de_DE)`, read values and the written 16 KB text are
  bit-identical to the C locale [R, `probe/locale_probe.cpp`]; the same in Python with `LC_ALL=de_DE.UTF-8` [R]. No
  GCC-only construct: no VLA, attribute, builtin or POSIX header; the code compiles with `-std=c++17 -Wpedantic`
  [M]. MSVC itself was not available (section 5).


## 2. Questions

Suspicious behaviour that is not a defect by the documentation, and decisions I would revisit.

| # | Topic | Observation | Proposal |
|---|---|---|---|
| Q1 | UTF-8 byte order mark | A file that starts with `EF BB BF` is rejected in line 1, before a comment ("Line is not a comment and has no '='") or a key ("Text before the '=' must be one key name") [R]. QuaDRiGa's `set_scenario_table.m` finds the key with a regular expression and skips a comment line without `=`, so it read such files [C]. Windows editors write a BOM. Not among the changes Q5 to Q14 of the plan. | Skip a leading BOM in line 1 and document it; or document that it is an error. |
| Q2 | Out-of-range numbers | `1e400` and `1e99999999999999999999` are errors with the message "is not a number"; `1e-400` and a value with 10^6 leading zeros read as 0 (or `-0`) without a message [R]. The cause is `std::istream >> double`, which sets `failbit` on overflow (LWG 23) and accepts underflow in libstdc++ [C, libstdc++ `__convert_to_v`]. If MSVC's `num_get` also fails on the `ERANGE` of an underflow, it would reject subnormals such as `4.94065645841247e-324` that the writer produces, and the writer's round trip would break there; not verified (no MSVC). | Parse with `std::from_chars` (locale-free, reports `result_out_of_range`; available for `double` in GCC 11 and current MSVC); reject overflow with "is out of range", decide whether underflow to a subnormal or 0 is accepted, and document it (D1). |
| Q3 | A short name stated twice in one file | `conf_file_read` returns it twice, `conf_file_write` writes `short_names = {"Ab", "Ab"}`, but `conf_file_list` throws "ShortName 'Ab' is ambiguous" for that file [R]. QuaDRiGa's lister does the same (it compares with all earlier entries) [C]. | Reject duplicates in `conf_file_write` (and in the reader with `validate = true`), or let the lister ignore a repeat within one file. |
| Q4 | Python path-like objects | `conf_file_read(pathlib.Path(...))` and `conf_file_write(pathlib.Path(...), ...)` raise `TypeError`; `conf_file_list(pathlib.Path(...))` raises `ValueError "Expected a string..."`; `bytes` are accepted for `fn` [R]. | Accept `os.PathLike` (convert with `os.fspath`); library-wide decision. |
| Q5 | MEX error identifiers | Errors raised by the shared helpers carry the identifier `MATLAB:unexpectedCPPexception`, not `quadriga_lib:CPPerror` as `SPEC_MATLAB_API.md` requires: `fn` not a char array, `validate = 'abc'`, a field of class `int8`, `uint16` [R]. The helpers also take the first element of `validate = [1 0]` and read `NaN` as `true` [R]. Pre-existing in `mex_helper_functions.hpp`. | Change the identifier in the helpers (library-wide). |
| Q6 | `plpar` field shape | MEX accepts a matrix for a path-loss field and reads it in memory order (`struct('A', [1 2; 3 4])` writes `1 3 2 4`) [R]; Python rejects a 2-D array; the MEX documentation says "one real vector" [C]. | Reject non-vectors in `qd_mex_struct2plpar`, as Python does. |
| Q7 | Scenario names of the lister | Only short names must be names; a file `a b.conf` or ` .conf` gives the scenario `a b` or ` ` [R]. Harmless for reading by file name, but such a scenario name could not be stated as a short name. | Skip such files or document it; low priority. |
| Q8 | Unlistable directory | A directory without read permission (mode 000 or 0311) is skipped silently, like a missing one [R]. Not documented. | Keep, document (D5). |
| Q9 | Windows paths | `std::ifstream(std::string)`, `std::filesystem::path(std::string)` and `path::string()` use the ANSI code page on Windows; Python and MATLAB (`mxArrayToString`) pass UTF-8, so non-ASCII paths would fail there, and `path::string()` can throw for names that the code page cannot represent [C, not reproduced]. | Library-wide: build paths with `std::filesystem::u8path` (or `char8_t` in C++20) on Windows. |
| Q10 | Overview formula in a written file | The comment `% ESD = ...` lists `mu`, `gamma`, `epsilon`, `zeta`, `alpha` terms only; `ES_D_mu_A`, `ES_D_mu_min`, `ES_D_mu_B/C/D` and `LSP_fc_min` are not shown although they change the ESD mean (layout test, `test_conf_file_write.cpp:340`) [C]. Comments only; the values are written. | Add the extra terms to the ESD line, or a note "plus the terms of section EoD offset / ESD". |
| Q11 | Counts without upper bound | `NumClusters = 1e300` passes the class "integer > 0" [C], as in QuaDRiGa. The consumer allocates per cluster. | Keep (deliberate, MATLAB parity) or add a bound such as 1000. |
| Q12 | Tests share a temp file name | `read_text` writes `test_conf_file_read_tmp.conf` in two cases; running test cases in parallel processes makes them fail (42 failed assertions in the per-case ASan run) [M]. The spec asks only for names distinct between suites. | Use the test case name in the file name (T9). |


## 3. Test coverage

### 3.1 C++ lines (gcov, gcc 15, `-O0 --coverage`, the 39 new cases) [M]

`src/qd_scenpar.cpp`: 430 of 433 lines (99.31 %), 91.4 % of the branches executed, 67.6 % taken at least once (most
untaken branches are exception clean-up paths of string temporaries). The three unexecuted lines:

| Line | Code | Reason |
|---|---|---|
| 435 | closing brace of `qd_conf_tokens` (`=====`) | exception landing pad, only on `bad_alloc` |
| 1392 | `const std::string rule = ...` (`#####`) | gcov attribution artefact: the next line uses `rule` and ran 888 times |
| 1405 | `throw std::runtime_error("Cannot write file ...")` | not tested; reachable (B1, `/dev/full`), see T1 |

One reachable-looking branch is dead: `ln == 0 ? fn + ": "` in the message lambda of `qd_scenpar_check`
(line 268). From the reader a failing key always has a line: defaults pass every check, and both cross-key
messages use the line of the key that was set [C]. Cosmetic.

### 3.2 Documented C++ behaviour and the test that pins it

Test files: `R` = `test_conf_file_read.cpp`, `V` = `test_scenpar_validate.cpp`, `L` = `test_conf_file_list.cpp`,
`W` = `test_conf_file_write.cpp`, `19` = `test_conf_files_v19.cpp`. Case names shortened.

| Behaviour (bullet of the documentation) | Pinned by |
|---|---|
| **conf_file_read** | |
| `Key = value` lines, `%` comments, blank lines, LF/CRLF, space and tab | R "line grammar" |
| Keys case-sensitive, characters `A-Za-z0-9_` | R "unknown keys" (`ds_mu`), R "input validation / key side" |
| 165 numeric keys, defaults for missing keys, `key_order` | R "an empty file gives the 165 default values" (typed table) |
| `SubpathMethod` word, default `legacy` | R "line grammar", R "value checks / Q14", R "input validation / Q6" |
| `PL_` keys: model one name, others one or more numbers, prefix removed | R "path-loss keys" |
| `ShortName` one name, several times, not in comments | R "ShortName lines" |
| Other keys ignored, reported once in order, value not examined | R "unknown keys" (`Alpha_unknown = some words ...`) |
| Last value of a repeated key wins | R "line grammar" (`SC_lambda`), R "path-loss keys" (`PL_A`, `PL_model`) |
| Number grammar, `Inf` any case | R "line grammar", R "input validation / Q5" |
| Out-of-range numbers (undocumented, Q2) | **none** (T5) |
| `GR_epsilon` ignored and reported | R "unknown keys" |
| 28 correlation keys define `lsp_xcorr` | R "each correlation key lands in two symmetric positions" |
| `runtime_error` if the file cannot be opened | R "input validation / file cannot be opened" (missing file only) |
| ... or `fn` is a directory | **none** (T3) |
| `invalid_argument` with file and line for each listed grammar error | R "input validation" (7 sections), R "error messages name the file and the line" |
| `validate = true` applies the checks of `scenpar_validate`; `false` only the grammar | R "value checks per validation class" (`rejects_value` runs both modes) |
| Not positive definite is not an error | R "the return value tells if lsp_xcorr is positive definite" |
| Failed call leaves all outputs untouched | R "a failed call leaves all outputs untouched" (3 sections) |
| `scenpar` previous content discarded; optional outputs alone and together; pre-sized `lsp_xcorr` | R "optional outputs alone and together" |
| `pl_order` first occurrence; `ignored_keys` each once | R "path-loss keys", R "unknown keys" |
| **scenpar_validate** | |
| Missing keys added; empty map gives the defaults | V "an empty map returns the 165 default values" |
| 138-key legacy set grows to 165, values kept | V "a legacy set with 138 numeric keys" |
| Unknown name is an error in both modes, `GR_epsilon` included, `SubpathMethod` not numeric | V "input validation" |
| NaN is an error for every key | V "value checks / NaN" (all 165 keys) |
| Each class and its boundary; other keys accept `Inf` | V "value checks" (8 sections, 98 unbounded keys counted) |
| Cross-key checks; `subpath_method` names | V "value checks / cross-key", "/ subpath_method" |
| `validate = false`: no check, NaN and any `subpath_method` accepted | V `rejects_value`, V "subpath_method" |
| `lsp_xcorr` from the 28 keys; flag by Cholesky; NaN gives `false` | V "each correlation key", V "the return value tells ..." (343 grid cases against the determinant) |
| Failed call leaves `scenpar` and the outputs untouched | V "input validation / failed value check" |
| Pre-sized output in place; repeatable | V "a pre-sized lsp_xcorr ..." |
| **conf_file_list** | |
| Files ending in `.conf` (case-sensitive), not recursive | L "files ending in .conf ..." |
| Byte-wise sorted per directory, directories in order | L "files ending ...", L "a file name defined by an earlier directory ..." |
| Earlier directory wins, shadowed file skipped completely (with its short names) | L "a file name defined by an earlier directory ..." |
| Short names after their file, only non-comment lines, other lines not examined | L "short names follow their file ..." |
| Ambiguity errors (4 forms) and malformed `ShortName` | L "input validation" |
| `runtime_error` if a listed file cannot be opened | **none** (T6) |
| Missing or non-directory entries skipped; `""` is the current directory; directory returned as given | L "directories that do not exist ...", L "an empty string ..." |
| Failed call leaves the outputs untouched | L "input validation" (marker checks) |
| Symlinks, unlistable directories (undocumented) | **none** (T7) |
| **conf_file_write** | |
| Layout: header, reference, validity range, short names, 10 sections, overview formulas, matrix, paragraphs, columns, comments with units | W "layout of a written file" (complete file typed by hand) |
| Path-loss formula blocks and comments per model | W "path-loss comments and formulas ..." (8 models) |
| Fewest digits 15..17, `Inf`, `-Inf`, `-0` | W "numbers are written with the fewest digits ..." |
| Locale independence ("independent of the global locale") | **none** (T2) |
| Without `write_defaults` only differing keys (bit-wise), empty sections left out, `SubpathMethod` only if not `legacy` | W "without write_defaults ..." (4 sections) |
| With `write_defaults` all 165 keys and `SubpathMethod` once | W "with write_defaults ..." |
| `pl_order` given, else sorted | W "path-loss keys are written in the given order ..." |
| Each validation rule before the file is opened; no file created or changed | W "input validation" (10 sections, each asserts that no file exists) |
| Existing file overwritten; failed check changes no file | W "an existing file is overwritten ..." |
| `runtime_error` if the file cannot be opened | W "input validation / fn" |
| `runtime_error` if the write fails | **none** (T1, B1) |
| Every written file passes `conf_file_read` without ignored keys | W "round trip of the conf files in data/config" (23 files, both modes, idempotent) |
| **v19 files** | 19 (5 cases; values not reviewed) |

### 3.3 Wrapper checklists

`SPEC_MATLAB_API.md`, "What every test covers" [C, M]:

| Item | read | validate | list | write |
|---|---|---|---|---|
| Basic call, size, class, values of every output | yes | yes | yes | yes (read back) |
| Each optional input omitted, `[]`, non-default | yes | yes | yes | yes |
| 0 outputs, fewer outputs give the same values | yes | yes | yes | n/a (no outputs) |
| Argument counts (inputs too many / too few, outputs too many) | yes | yes (no "too few": 0 inputs is valid) | yes | yes |
| Non-double input classes | yes (`1`, `uint32(0)`) | yes (`uint32`, `single`, logical, `int64`) | yes | yes |
| Orientation and shape | column cell outputs | n/a | row and column cell input | column `plpar` vector returned as row |
| Structs: partial accepted, documented fields, round trip, non-struct error | yes | yes | n/a | yes |
| Every `mexErrMsgIdAndTxt` of the wrapper and its helpers reached | yes | yes | yes | yes |
| At least one C++ error passed through | yes | yes | yes | yes |

`SPEC_PYTHON_API.md`, "What every test covers" [C, M]:

| Item | read | validate | list | write |
|---|---|---|---|---|
| Basic call: tuple length, types, dtypes, values | yes | yes | yes | yes (returns `None`) |
| Keyword arguments equal positional | yes | yes | yes | yes |
| Each optional input omitted, `None`, non-default | yes | yes | yes | yes |
| Array memory layout (C, Fortran, strided view) | n/a | 0-d and 2-D scalars | n/a | yes (`valid_range`, `plpar`) |
| Array dtype (`float32`, int, list) | n/a | yes | n/a | yes |
| Inputs not modified | n/a (str) | yes | n/a (str) | yes |
| Output dtypes | yes | yes | yes (list of str) | n/a |
| Dicts: partial accepted, keys and order, round trip | yes | yes | n/a | yes (23 files, bit patterns) |
| C++ errors (type and message) | yes | yes | yes | yes |
| Every wrapper `throw` reached | n/a | yes | yes | yes |
| pybind11 `TypeError` (missing argument, unknown keyword) | yes | unknown keyword only (no mandatory argument) | yes | yes |

### 3.4 Gaps

| # | Severity | Gap | Could a plausible change break it unnoticed? | Proposed test (one line) |
|---|---|---|---|---|
| T1 | medium | The write-error path of `conf_file_write` (line 1405, B1) | Yes: it is broken today | After the B1 fix: on Linux, if `/dev/full` exists, a write there throws `runtime_error`; and a write that fails leaves an existing file unchanged (temp-file path) |
| T2 | medium | Locale independence of reading and writing | Yes: replacing `istringstream`/`ostringstream` by `strtod`/`snprintf` passes every test today | Set `std::locale::global` and `setlocale` to a decimal-comma locale if one exists (skip otherwise), round-trip a file, restore the locale |
| T3 | low | A directory as `fn` of `conf_file_read` | Yes: without the `is_directory` check, Linux opens the directory and returns the defaults without an error | `conf_file_read` on a temporary directory throws `runtime_error` |
| T4 | low | B2 (line break in `fn`) | It is broken today | After the fix: `conf_file_write("a\nb.conf", ...)` throws `invalid_argument` or the read-back equals the input |
| T5 | low | Out-of-range numbers (Q2) | Yes: a parser change would flip the behaviour silently | Pin the decision: `1e400` gives an error with "out of range", `1e-400` gives the decided value |
| T6 | low | `runtime_error` of `conf_file_list` for an unreadable listed file | Moderately: a refactor to "skip unreadable" would pass | POSIX only, not as root: `chmod 000` on a listed file, expect `runtime_error` with `parse_shortnames`, a listing without |
| T7 | low | Symlinks and directories named `*.conf` in links | Moderately (`is_regular_file` vs `symlink_status`) | A symlink to a file is listed, a dangling link is skipped (skip where symlinks cannot be created) |
| T8 | low | Python `None` for `validate` (B3) | It is broken today | After the fix: `conf_file_read(fn, None)` raises `TypeError` or validates |
| T9 | low | One temp file name for several cases (Q12) | Only under parallel test runners | Use the case name in the temp file name |
| T10 | low | MEX `plpar` matrix input (Q6) | Yes: either behaviour passes today | After the decision: `struct('A', [1 2; 3 4])` errors (or is documented) |


## 4. Documentation

Checked [C, M]: all twelve blocks against the code and the spec rules (layout and section names, singular and
plural headings, `*(optional)*` and "default:" marks, argument order, sizes, units, error behaviour, "See also"),
the three layers against each other, and the header comments against the declarations.

- No `*` or `|` outside code spans in any block; all C++ and Python lines are within 120 characters [M,
  `probe/doclint.py`].
- Header comments of `quadriga_tools.hpp` match the declarations: names, defaults, "Optional input/output", sizes;
  the group line `// ---- Scenario parameters ----` is present [C].
- `html_docu/` is current: `make documentation` on HEAD reproduces all generated files byte for byte [M].
- MATLAB help files: generated by the build from the MEX blocks and current by construction (`ENABLE_MEX_DOC=ON`);
  the worktree has no `+quadriga_lib/` folder, so there is no stale local copy [M]. The generator indents by four
  characters, so the six long MEX lines of D8 become 102 to 104 characters in `help`.
- The three layers agree on every behaviour; the differences are the intended ones (0-based `ref_sec` in C++ and
  Python, 1-based in MATLAB; struct versus dict; `[]` versus `{}` for an empty `plpar`).

Improvements, each with the current and the proposed text:

| # | Block(s) | Current | Proposed |
|---|---|---|---|
| D1 | `conf_file_read`, C++/MEX/Python | "Numbers are decimal with an optional sign, an optional decimal point and an optional exponent (`e` or `E`), for example ...; `Inf`, `+Inf` and `-Inf` are accepted in any case" | add: "; a number beyond the range of a double (`1e400`) is an error; a number below the smallest subnormal reads as 0" (or the outcome of Q2) |
| D2 | `conf_file_write`, C++/MEX/Python | "Numbers are written with the fewest significant digits (15 to 17) that [[conf_file_read]] reads back to the same double" | "Numbers are written with up to 15 significant digits, or with 16 or 17 where 15 do not read back to the same double in [[conf_file_read]]" (`-6.28` has 3 digits; "fewest (15 to 17)" reads as at least 15) |
| D3 | `conf_file_write`, MEX and Python | "A failed call does not create or change a file; an existing file is overwritten" | until B1 is fixed: "A failed input check does not create or change a file; an existing file is overwritten" (the C++ wording); after the fix: "A failed call does not create or change a file" is true |
| D4 | `conf_file_read`, C++ | "Throws `std::runtime_error` if the file cannot be opened" | "Throws `std::runtime_error` if the file cannot be opened or is a directory" (MEX, Python the same) |
| D5 | `conf_file_list`, C++/MEX/Python | "Lists the files whose name ends in `.conf` (case-sensitive) in each directory, not recursive." / "A directory that does not exist or is not a directory is skipped" | "Lists the regular files and symbolic links to regular files whose name ends in `.conf` (case-sensitive) and is longer than `.conf`, not recursive." / "A directory that does not exist, is not a directory or cannot be listed is skipped" |
| D6 | `CONF_FILE_READ` (MEX) | "## See also:" lists `scenpar_validate`, `conf_file_list` | add "- [[conf_file_write]] (write a conf file)", as in C++ and Python |
| D7 | `scenpar_validate`, C++/MEX/Python | "## See also: - [[conf_file_read]] (read the parameters from a conf file)" | add "- [[conf_file_write]] (applies the same checks before writing)" |
| D8 | MEX blocks, 100-character limit | `conf_file_read.cpp` lines 16 (103), 45 (101), 52 (101) of the block; `scenpar_validate.cpp` 48 (102); `conf_file_list.cpp` 13 (102), 31 (101) | rewrap, e.g. line 16: "  (`e` or `E`), for example `-6.28`, `.5`, `6.2e-07`; `Inf`, `+Inf` and `-Inf` are accepted" / "  in any case" |
| D9 | `conf_file_read`, C++ | (nothing on the locale; the writer says "independent of the global locale") | add "- Numbers are read independent of the global locale (the decimal point is always `.`)" |
| D10 | `conf_file_read`, C++ and MEX | C++: "**`short_names`** *(optional)* — Values of all `ShortName` lines in file order" (no size); MEX: "`[n, 1]`" with `n` undefined | C++: "...; `[n_short]`", "...; `[n_ignored]`"; MEX: "`[n_short, 1]`", "`[n_ignored, 1]`" |
| D11 | `md/conf_file_parser.md` | Sections 4.2, 6, 7.1, WI-3 and 9 still count `GR_epsilon` (`n_key = 166`, "grows from 140 to 167 fields"); `conf_file_write` and the wrappers are not mentioned; only WI-5 has a status note | Add a record of the later changes: `GR_epsilon` dropped (138 + 27 = 165 numeric keys, 166 struct fields), `conf_file_write` added (with the decisions of its review point A), wrappers in `tools` |
| D12 | `qd_scenpar.cpp` (code comment level) | `static const char *qd_conf_sections[]`, `qd_conf_lsp`, `qd_conf_pl_comments`, `qd_conf_pl_formulas`: the arrays of pointers are not `const` | `static const char *const ...` so that the tables are immutable, as the spec's "no static mutable state" asks; never written today [C] |

Smaller points, no table entry: `conf_file_write` C++ `pl_order` "omitted: sorted by name" could say "sorted
byte-wise by name"; Python `conf_file_list` accepts a tuple and `None` (lists nothing), which "one str or a list of
str" does not mention; MEX `CONF_FILE_WRITE` `plpar` says "one real vector" while a matrix is accepted (Q6).


## 5. Not checked

- MSVC and clang builds (no compiler available); the MSVC questions Q2 (underflow in `num_get`) and Q9 (code page)
  are from reading.
- Windows and macOS behaviour of paths, symlinks and line ends.
- A FIFO or a device other than `/dev/null` and `/dev/full` as `fn` (`ifstream` blocks on a FIFO without writer).
- Read errors in the middle of a file (`EIO`); only open and write errors were provoked.
- The values in `data/config/*.conf` against TR 38.901 (out of scope), and the physics of the new keys (no consumer
  exists yet).
- AVX2 builds (`avx2 = OFF` in the GNUmakefile; the conf code has no kernel).
- Performance; the 39 new cases run in 0.12 s.
- The sanitizer reports in other functions (section 0) beyond naming them.
