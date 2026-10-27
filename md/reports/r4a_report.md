# Run 4a report — MEX wrappers, documentation sweep

## Summary

- Six MEX wrappers written as in spec 6.1 (`lsf_init_phases`, `lsf_draw`, `path_loss`, `lsp_generate`,
  `los_probability`, `o2i_loss`), each followed by its MOxUnit test (SPEC_MATLAB_API order: wrapper, then test).
- `SECTION_DESC` of "Geometry-based stochastic model" added in C++ (`src/qd_scenpar.cpp`) and MEX
  (`api_mex/conf_file_read.cpp`), 20 lines each; `See also` lines extended so that each function names its
  neighbours in the data flow of 1.1.
- Gates: Catch2 full suite (AVX2 off and on), MEX suite in Octave 11.1, MATLAB R2024a and R2026a, objdump gate,
  Python suite; no compiler warning in any build. Only failure: the known `test_ray_mesh_interact` (since before Run 1).
- One spec problem: the test point "`uint8` floor IDs accepted" (6.1) cannot be met with the existing MEX helpers
  (Open points 1); not implemented, everything else done.

## Files

Created (12):
- `api_mex/lsf_init_phases.cpp`, `api_mex/lsf_draw.cpp`, `api_mex/path_loss.cpp`, `api_mex/lsp_generate.cpp`,
  `api_mex/los_probability.cpp`, `api_mex/o2i_loss.cpp`
- `tests/mex_tests/test_lsf_init_phases.m`, `test_lsf_draw.m`, `test_path_loss.m`, `test_lsp_generate.m`,
  `test_los_probability.m`, `test_o2i_loss.m`

Changed (5):
- `src/qd_scenpar.cpp`: `SECTION_DESC` after the `SECTION` block; `See also` of `conf_file_read` (+ `lsf_init_phases`,
  `path_loss`)
- `api_mex/conf_file_read.cpp`: `SECTION_DESC` (MEX wording, 1-based columns); the same two `See also` lines
- `src/lsf_init_phases.cpp`: `See also` + `conf_file_read`
- `src/lsf_draw.cpp`: `See also` + `path_loss`, `lsp_generate`
- `src/lsp_generate.cpp`: `See also` + `path_loss`

Plus this report. Not from this run: `md/prompts/impl_run4a.md` (untracked, the coordinator's prompt).

## Decisions and deviations

Review points of SPEC_MATLAB_API (not stops): A — the interface of all six functions is fixed by spec 6.1; the only
open question (uint8, Open points 1) is not a question of the interface. B — this report.

1. Helpers and headers. The four wrappers with `scenpar` / `plpar` include `mex_quadriga_lib_functions.hpp` and use
   `qd_mex_struct2scenpar` (the returned `SubpathMethod` is discarded, so a struct from `conf_file_read` /
   `scenpar_validate` is accepted) and `qd_mex_struct2plpar` (its `model` field is `pl_model`; `[]` or no `model`
   field = empty model). `los_probability` and `o2i_loss` need only `mex_helper_functions.hpp`. The `seed` of
   `lsf_init_phases` is read with the existing `qd_mex_get_seed` (as `sos_init_phases`, `sos_randn`), so a seed
   below -1 or a fractional seed is rejected by the wrapper's message before the C++ check.
2. Optional outputs: allocated with `qd_mex_init_output` and passed with the `(nlhs > X)` guard instead of the
   `.empty()` variant that SPEC_MATLAB_API prefers. Reason: spec 6.1 states "the C++ outputs beyond nlhs passed as
   `nullptr`", which the guard expresses directly; both are identical for `n_link >= 1`. `phase` (size from the
   preset) is copied after the call, as 6.1 says. Mandatory outputs (`phase`, `rnd`, `pl`, `lsp`, `p_los`, `loss`)
   are computed also for `nlhs = 0` (local object), so the C++ validation runs on a call without outputs; no
   wrapper returns early for `nlhs = 0`.
3. `los_probability`: `los_soft` is allocated and passed only if `nlhs > 1` and both `rnd_g` and `fc` are non-empty;
   otherwise the second output is `[]` (0x0 double) and C++ gets `nullptr` (so the C++ error "Output 'los_soft'
   requires 'rnd_g' and 'fc'." cannot be reached from MATLAB, as 6.1 intends). Clutter arguments: `[]` -> NaN.
4. `o2i_loss`: `model_index` pre-allocated as uint64 and shifted +1 in place after the call; `models` read with
   `qd_mex_get_strings` (cell array or one char array).
5. `lsf_draw` `floor_id`: read with `qd_mex_get_Col<double>` (var name `floor_id`), each entry checked with
   `v >= 0 && v < 2^64 && v == floor(v)` (NaN and Inf fail) -> `Entries in 'floor_id' must be non-negative
   integers.`, converted to `arma::uvec`; no index shift (labels, as 6.1).
6. Documentation adaptations (one-line summaries identical to C++): every column, slice and index is 1-based in the
   MEX blocks (generator columns 1 to 11, phase slices 1 / 2, LSP slices 1 to 8, `o2i_loss` `rnd` columns 1 to 4,
   `rnd_he` = column 11 of `lsf_draw`); "key" -> "field" for struct entries. Where a C++ message counts columns
   from 0 (`Column 9 of 'rnd' ...` of `lsp_generate`, `Column 3 of 'rnd' ...` of `o2i_loss`), the MEX block says
   so; `lsp_generate` states that column 10 is checked only with 3 or more outputs (6.1).
7. `path_loss` MEX block: the C++ notation `[X|v]` is replaced by "`[X]` the value of field `X`, or its default if
   the field is missing (`X = v` in the lists below)" and `[LOSmax]`, `[hE_rand]`, `[hE]`, because SPEC_MATLAB_API
   forbids `|` in documentation text. The C++ block keeps `[X|v]` (not in the Run 4a edit list; Open points 3).
8. `n_link = 0`: the MEX helpers read an empty array (`zeros(0, 3)`) as 0x0, so position-based wrappers raise the
   C++ message `Input 'rx_pos' must have 3 columns.` for zero links. The MEX blocks therefore do not state the
   `n_link = 0` rule of the C++ blocks (`lsf_draw`, `lsp_generate`, `o2i_loss` with `n_ut = 0`).
   `los_probability` accepts an empty `d2d_out` (tested). No wrapper code was added for this (Open points 2).
9. `See also` (C++ and MEX): added where a data-flow neighbour of 1.1 was missing: `conf_file_read` -> `lsf_init_phases`,
   `path_loss`; `lsf_init_phases` -> `conf_file_read`; `lsf_draw` -> `path_loss`, `lsp_generate`; `lsp_generate` ->
   `path_loss`. `path_loss`, `los_probability` and `o2i_loss` already named their neighbours. `scenpar_validate`,
   `conf_file_list`, `conf_file_write` are not in the data flow and were left unchanged. The `conf_file_read` MEX
   block got the same two lines (file in the edit list); the MEX `scenpar_validate`, `conf_file_list`,
   `conf_file_write` blocks are unchanged.
10. `SECTION_DESC`: placed after the `SECTION` block of `src/qd_scenpar.cpp` and `api_mex/conf_file_read.cpp` (the
    documentation tools pair a description with the section of its file); content as 7 Run 4a: GBSM and TR 38.901
    clause 7.5 steps 1 to 4, the conf files (`data/config`, 23 v19 files), the data flow with `[[links]]` incl. the
    layout order `o2i_loss` before `los_probability`, the generator index (columns, Normal/Uniform, own
    decorrelation distance), the units, and the LOS-state / O2I draws by the caller with `[[sos_rand]]` /
    `[[sos_randn]]` at `LOS_state_lambda`, `IO_state_lambda`. 20 text lines each. `tools/extract_md.py` was run to
    a scratch file for both APIs to check that the blocks are picked up (no file in `html_docu/` was written).
11. Tests: each covers the SPEC_MATLAB_API checklist and the points of 6.1, except `uint8` floor IDs (uint32,
    int64 and single are tested instead). Value checks that involve `lsf_draw` use 1e-5 / 1e-4 tolerances with a
    comment (AVX2 kernel with single-precision sin/cos when the library is built with AVX2; single-precision phases
    in the type test); everything else is exact or 1e-12. Additions beyond 6.1, cheap and wrapper-related:
    `lsf_init_phases` slice 1 equals `sos_init_phases(256, 11, 7)` and the bias-removed fields have a zero mean in
    `lsf_draw`; `lsf_draw` column 1 equals `sos_val` of the DS field, floor 1 equals the phases plus `2·pi·n/256`;
    `lsp_generate` columns 9 and 10 land in `abs_toa` and the permittivity; the `sf_sigma` replacement and the
    InF V/S difference `lg(332/152)` are checked by value; `path_loss` legacy distance by value (`d3 = 980` m).
12. The spec's `lsp_generate` check "column 10 of `rnd` = 1.5 runs with `nlhs = 1` and raises the C++ error with
    `nlhs = 3`" is tested with 1 and 2 outputs (runs) and a local function requesting 3 outputs (raises).

## Test results

Load average (1-min) before the heavy runs: 3.8 before the MEX build, 2.3 before the test build, 8.9 before the
Catch2 run, 7.8 before Octave, 7.4 before R2024a, 6.2 before R2026a, 5.0 before the Python build, 5.2 before the
final AVX2 build (other sessions). No timing results are reported, so the load does not affect any result.
All builds with `--parallel 12`, test runs with `OMP_NUM_THREADS=12`; builds and suites one at a time.

Builds: the `make` configuration (gcc-12, MATLAB and Octave on, AVX2 off, Python off) configured and built by hand
with `cmake --build build_linux --parallel 12` and `cmake --install build_linux` (the Makefile passes `--parallel`
without a number); then the same with `ENABLE_TESTS=ON` (the `make test` configuration).

Catch2 (`make test` configuration, AVX2 off), `OMP_NUM_THREADS=12 build_linux/test_bin`:
```
All tests passed (1353575 assertions in 1373 test cases)
```
Catch2 with AVX2 on (build tree restored to its start configuration: tests on, AVX2 on, MATLAB/Octave/Python off):
```
All tests passed (1355498 assertions in 1374 test cases)
```

MEX, new tests first one by one in Octave (all `PASS` at the first run), then the complete suite
(`tests/quadriga_lib_mex_tests.m`, 74 tests = 68 + 6 new):
- Octave 11.1.0 (`octave --eval "cd tests; quadriga_lib_mex_tests;"`): `FAILED (passed=73, failure=1)`
- MATLAB R2024a (`matlab24` = `/usr/local/MATLAB/R2024a/bin/matlab -batch "run('tests/quadriga_lib_mex_tests.m');"`):
  `FAILED (passed=73, failure=1)`
- MATLAB R2026a (`/usr/local/MATLAB/R2026a/bin/matlab -batch ...`): `FAILED (passed=73, failure=1)`
- The one failure in all three is `test_ray_mesh_interact.m` line 113 (0.98485 vs. 0.69639), the pre-existing
  failure reported since Run 1; all six new tests and the conf / `scenpar_validate` tests pass in all three.

objdump gate (`objdump -T +quadriga_lib/*.mexa64 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -u`, 83 MEX files incl. the
6 new): 3.4.9, 3.4.11, 3.4.18, 3.4.19, 3.4.20, 3.4.21, 3.4.26 — nothing above 3.4.28.

Python (the `make python` configuration, built by hand with `--parallel 12`; no Python file changed in this run):
`python3 -m pytest tests/python_tests -x -q` -> `816 passed, 3 skipped, 52 subtests passed in 2.56s`.

## Warnings

None. `grep -ci warning` on the logs of the MEX build (library, 83 MATLAB and 83 Octave MEX files with
`-Wall -Wpedantic -Wextra`), the test build, the Python build and the AVX2 rebuild: 0 each.

## Open points

1. **Spec problem (6.1, `lsf_draw` test point "`uint8` floor IDs accepted").** `floor_id` is "read as double"
   (6, general rules); the MEX conversion helpers (`qd_mex_copy_convert` behind `qd_mex_get_Col`) accept only
   double, single, int32, uint32, int64 and uint64 and raise `Input 'floor_id' has an unsupported data type.` for
   uint8 (checked in Octave). Meeting the test point needs either a helper change (`src/mex_helper_functions.hpp`,
   not in the Run 4a file list, and it would change every wrapper) or raw `mx*` code in the wrapper (against
   SPEC_MATLAB_API). Not implemented. Proposal: change the test point to "`uint32` floor IDs accepted" (done in the
   test), or, as a separate change, add int8/uint8/int16/uint16 to `qd_mex_copy_convert` and `qd_mex_read_scalar`.
2. `n_link = 0` through the MEX wrappers (decision 8): a `[0, 3]` position array becomes 0x0 in the helpers. If the
   C++ rule should hold in MATLAB too, the helpers would have to keep the column count of empty arrays; a
   library-wide change, outside this run.
3. The C++ `path_loss` block uses `[X|v]`, which SPEC_CPP_API forbids (`|` is markdown syntax); proposal for a later
   documentation pass: the wording of the MEX block (decision 7).
4. Not tested: `uint8` floor IDs (point 1); the non-PD error with the symmetric root (Catch2 covers it); R2026a was
   run in addition to the required Octave and R2024a.
5. The pre-existing MEX failure of `test_ray_mesh_interact` persists (see the Run 1 report).

## Suggested commit cut

One commit (wrappers, tests and the section documentation belong together; the `See also` lines are one line each):
- Files: `api_mex/lsf_init_phases.cpp`, `api_mex/lsf_draw.cpp`, `api_mex/path_loss.cpp`, `api_mex/lsp_generate.cpp`,
  `api_mex/los_probability.cpp`, `api_mex/o2i_loss.cpp`, `api_mex/conf_file_read.cpp`,
  `tests/mex_tests/test_lsf_init_phases.m`, `tests/mex_tests/test_lsf_draw.m`, `tests/mex_tests/test_path_loss.m`,
  `tests/mex_tests/test_lsp_generate.m`, `tests/mex_tests/test_los_probability.m`, `tests/mex_tests/test_o2i_loss.m`,
  `src/qd_scenpar.cpp`, `src/lsf_init_phases.cpp`, `src/lsf_draw.cpp`, `src/lsp_generate.cpp`
- Message:
  ```
  MATLAB/Octave wrappers of the large-scale model and section overview

  Adds the MEX wrappers lsf_init_phases, lsf_draw, path_loss, lsp_generate, los_probability and o2i_loss with
  their MOxUnit tests. Scenario and path-loss parameters are the structs returned by conf_file_read or
  scenpar_validate; [] gives the defaults. Columns, slices and the model index of o2i_loss are 1-based in
  MATLAB; floor IDs are labels and pass through unchanged. Optional outputs that are not requested are not
  computed, so the ground permittivity check of lsp_generate applies only when it is requested.

  The section "Geometry-based stochastic model" gets an overview for C++ and MATLAB: the scope (TR 38.901
  clause 7.5 steps 1 to 4), the conf files, the data flow per drop and per BS site, the generator index and
  the units. The See also lists name the neighbouring functions of the data flow.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```
