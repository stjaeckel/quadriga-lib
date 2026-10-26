# Run 1 report — header move, new keys, Python submodule

## Summary

- New public header `include/quadriga_gbsm.hpp` (guard, includes, `static_assert`s, the four conf declarations moved
  verbatim from `quadriga_tools.hpp`, the comment block of the "Large-scale fading" group, no new declarations);
  `quadriga_lib.hpp` includes it after `quadriga_sos.hpp`. Section "Geometry-based stochastic model" in C++, MEX, Python.
- Six new keys (`LOS_state_lambda`, `IO_state_lambda`, `O2I_d2D_in_max`, `DS_VS`, `DS_VS_a`, `DS_VS_b`), the cross-key
  check of `DS_VS`, the V/S term in the DS overview line of the writer, all counts 165/27/166 -> 171/33/172.
- 18 v19 conf files edited (state distances, O2I maximum indoor distance, InF delay spread with V/S keys, `absTOA_Lmax`).
- Python submodule `quadriga_lib.gbsm` (dispatcher `python_gbsm.cpp`); the four conf functions are gone from `tools`.
- All gates pass except one MEX test (`test_ray_mesh_interact`) that fails identically on the unchanged base commit.

## Files

Created (2):
- `include/quadriga_gbsm.hpp`
- `api_python/python_gbsm.cpp`

Changed (47):
- Headers: `include/quadriga_tools.hpp`, `include/quadriga_lib.hpp`
- Library: `src/qd_scenpar.cpp`
- MEX: `api_mex/conf_file_read.cpp`, `api_mex/conf_file_write.cpp`, `api_mex/conf_file_list.cpp`,
  `api_mex/scenpar_validate.cpp`
- Python: `api_python/python_tools.cpp`, `api_python/python_main.cpp`, `api_python/qpy_conf_file_read.cpp`,
  `api_python/qpy_conf_file_write.cpp`, `api_python/qpy_conf_file_list.cpp`, `api_python/qpy_scenpar_validate.cpp`,
  `CMakeLists.txt`
- Conf files (18): `data/config/3GPP_38.901_v19_` + `RMa_NLOS`, `UMi_NLOS`, `UMa_NLOS`, `SMa_NLOS`, `Indoor_NLOS`,
  `InF_LOS`, `InF_NLOS_SL`, `InF_NLOS_SH`, `InF_NLOS_DL`, `InF_NLOS_DH`, `RMa_LOS_O2I`, `RMa_NLOS_O2I`, `UMi_LOS_O2I`,
  `UMi_NLOS_O2I`, `UMa_LOS_O2I`, `UMa_NLOS_O2I`, `SMa_LOS_O2I`, `SMa_NLOS_O2I` (`.conf`)
- Catch2: `tests/catch2_tests/test_conf_file_read.cpp`, `test_conf_file_write.cpp`, `test_conf_file_list.cpp`,
  `test_conf_files_v19.cpp`, `test_scenpar_validate.cpp`
- MEX tests: `tests/mex_tests/test_conf_file_read.m`, `test_conf_file_write.m`, `test_scenpar_validate.m`
- Python tests: `tests/python_tests/test_conf_file_read.py`, `test_conf_file_write.py`, `test_conf_file_list.py`,
  `test_scenpar_validate.py`
- SPEC files: `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`

Plus this report. Not from this run: `md/reports/calibration_research.md` and `md/prompts/impl_run1.md` (untracked,
appeared from other sessions / the coordinator; not to be committed with this run unless intended).

## Decisions and deviations

1. Order of work: the Run 1 steps were followed in their order (code of step 3 before the test updates of step 5,
   as the step list prescribes). The SPEC review points A/B/C were not stops; nothing was open at A (the contract is
   the moved declarations plus the key table of 4.1).
2. Header text: built exactly from 2.1 (guard, includes, `static_assert`, the "Scenario parameters" group with the
   block copied byte for byte from `quadriga_tools.hpp`, the "Large-scale fading" comment block). `quadriga_tools.hpp`
   keeps its includes (`<unordered_map>` is still used there by the material functions); the blank line before the
   closing brace of its namespace is kept so that the diff is a pure removal.
3. Conf Catch2 tests include `quadriga_gbsm.hpp` instead of `quadriga_tools.hpp` (they use only the conf functions;
   `conf_file_list` is declared there too). `src/qd_scenpar.cpp` includes `quadriga_gbsm.hpp` likewise.
4. Writer DS overview line (4.1): `log10( a * V/S + b )` followed by the mean with its own sign (`std::signbit`, so
   `-0` gives ` - 0`) and the remaining terms of the usual mean expression. Implemented by prefixing the existing
   term string, so `DS_gamma`, `DS_epsilon`, ... follow unchanged.
5. Documentation of the overview item (C++, MEX, Python): written as "the DS line has the term
   `log10( a · V/S + b )` when `DS_VS > 0` (a = `DS_VS_a`, b = `DS_VS_b`; `V/S` stays symbolic)". Deviation from the
   literal text of 4.1 (`a * V/S`): SPEC_CPP_API forbids `*` in documentation text, and the MATLAB help extractor
   strips asterisks. The written file itself uses `*` as specified.
6. Planning labels: 9.1 requires every planning label in a file that the run edits to be reworded. Besides the
   labels named in 5.8 (O10, Q1, Q2, Q3, B1, "review point A"), the edited conf tests contained further labels
   (Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, T3, B2, "section 7.2 of the plan", "Key formula of the plan",
   "Examples of the plan", "measured in MATLAB at review point A"); all were reworded to state the rule, including
   the `SECTION` names that carried them (e.g. `"... (Q13)"` -> without the label). The meaning of Q9/Q10/Q11-Q14
   was taken from the retired working document in git history.
7. `ref_default` oracles in `test_conf_file_read.cpp` and `test_scenpar_validate.cpp`: the rule "every `*_lambda`
   defaults to 20" would also catch `LOS_state_lambda` and `IO_state_lambda`; an explicit default-0 rule was added
   before it.
8. The class tests of `DS_VS` in the "non-negative" sections pair a positive `DS_VS` with `DS_VS_a` (the cross-key
   rule would otherwise reject the valid case); negative and `Inf` values still fail on the class check first.
9. Writer test "the new keys land in their section and paragraph": the small-scale section is the last section of a
   file without path loss, so no blank line follows the `IO_state_lambda` line; the check asserts the paragraph start
   (blank line after `SC_lambda`) and the order, not a trailing blank line.
10. Conf files: the new value lines use the column layout of the writer (name padded to 19, value to 13). Placement:
    `LOS_state_lambda` / `IO_state_lambda` as a new paragraph after `SC_lambda` (the writer's paragraph 2);
    `O2I_d2D_in_max` as a new last paragraph of "Model parameters"; the V/S keys directly after `DS_sigma`, before the
    replaced comment block; `absTOA_Lmax` directly after `absTOA_lambda`. Values are written with full double
    precision (`4.2727272727272725`, `4.545454545454546`). The comment of the `absTOA_Lmax` value line is the one of
    4.4.1 also in Indoor_NLOS ("of the hall").
11. SMa O2I files: one extra comment line after `O2I_d2D_in_max = 25`: "SMa: 25 m for UTs in commercial buildings,
    10 m for UTs in residential buildings." (the note of 4.4.1, checked against TR clause 7.4.3.1).
12. Python: `python_gbsm.cpp` holds the four includes and the four `m.def` blocks moved unchanged from
    `python_tools.cpp`; the submodule is registered after `sos` in `python_main.cpp`; the forward declaration is
    sorted with the others. Usage lines and test headers say `quadriga_lib.gbsm.<name>`.
13. Python test additions: the last 6 keys of the dict are the new keys (read and validate), the `DS_VS` cross-key
    `ValueError` through all three functions, and a test `test_submodule` (in `test_conf_file_read.py`) that checks
    all four names exist in `gbsm` and none in `tools`.
14. MEX tests: field indices 167..172 of the new keys (read and validate), `rmfield(names(140:172))` for the legacy
    struct, the `DS_VS` cross-key error through `scenpar_validate` and `conf_file_write`, and the 172 assignments of a
    `write_defaults` file (`regexp ... 'lineanchors'`, works in MATLAB and Octave).
15. SPEC files: besides the required entries, `SPEC_CPP_API.md` now lists `quadriga_sos.hpp` too and says
    "includes the headers below" (it said "the four headers" and already missed `quadriga_sos.hpp`); the section-name
    lists of SPEC_CPP_API and SPEC_MATLAB_API gained "Sum-of-sinusoids random number generator", which both APIs
    already use. `SPEC_PYTHON_API.md`: 7 dispatcher files incl. `python_gbsm.cpp`, submodule `gbsm` in the submodule
    list, table row `Geometry-based stochastic model | gbsm`, `python_gbsm.cpp` in the registration list.
16. The build tree `build_linux` was reconfigured by `make python` / `make` / `make test`; at the end it was restored
    to Catch2 with AVX2 on and MATLAB/Octave/Python off (the configuration it had at the start, judging by the
    1307 test cases of the first run).

## Test results

Machine load (1-min) before the heavy runs: 0.6 to 2 at the start, up to 59 during the MEX build (other sessions);
no timing results are reported, so the load does not affect any result.

Catch2 (`cmake --build build_linux --parallel 12`, `build_linux/test_bin`):
- conf tests: `build_linux/test_bin "conf*"` -> `All tests passed (4618 assertions in 45 test cases)`;
  `build_linux/test_bin "scenpar*"` -> `All tests passed (3936 assertions in 7 test cases)`
- full suite, AVX2 on (final state): `All tests passed (1349151 assertions in 1307 test cases)`
- `make test` (AVX2 off, as configured by the Makefile targets): `All tests passed (1347228 assertions in 1306 test
  cases)`; the one case less is `quadriga_lib_has_AVX2`, which exists only with AVX2.

MEX (built with `make`, MATLAB R2024a = `/usr/local/MATLAB/R2024a/bin/matlab` via `matlab24`, Octave = `/usr/bin/octave`;
MOxUnit unpacked with `make moxunit-lib`):
- Octave: `FAILED (passed=67, failure=1)`
- MATLAB R2024a: `FAILED (passed=67, failure=1)`
- The one failure is `test_ray_mesh_interact` line 113 (transmission gain 0.98485 vs. oracle 0.69639). It is not
  related to this run: the same test fails identically with `ray_mesh_interact.mexa64` built from the unchanged base
  commit 46dd28d in a separate worktree (MATLAB R2024a: `FAILED (failure=1)`, same numbers). All conf and
  `scenpar_validate` MEX tests pass. R2026a was not run (not required).
- `make test` ran Octave and R2024a again with the same result.

objdump gate (`objdump -T +quadriga_lib/*.mexa64 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -u`): 3.4.9, 3.4.11,
3.4.18, 3.4.19, 3.4.20, 3.4.21, 3.4.26 — nothing above 3.4.28.

Python (`make python`, then `make python_test`): the four conf test files `37 passed`; full suite
`816 passed, 3 skipped in 2.39s`.

## Warnings

No compiler warnings in the library (`-Wall -Wextra -Wpedantic`), the MEX files or the Python module. The only
"warning" in the `make test` log is the run-time Catch2 `WARN` of `test_qdant_write.cpp:386` ("No locale with a
decimal comma installed"), unrelated and pre-existing.

## Open points

- Pre-existing MEX failure `test_ray_mesh_interact` (line 113, gain of mode 2 = transmission): the MEX oracle
  `0.5·T_te² + 0.5·T_tm²` disagrees with the library (0.98485); probably the oracle lacks the power-transmission
  factor `n2·cos(th2) / (n1·cos(th1))`, or the library changed. Outside the scope of this series; worth a separate fix.
- Proposal for the spec: 5.8 could say "all planning labels in the edited files" instead of listing a few; 4.1 could
  give the documentation text with `·` (SPEC_CPP_API rule).
- The new Catch2 tests were written after the implementation of step 3 (the step order of Run 1); they pass, but
  were not run against the old code.
- `html_docu/` still shows the old section and the `tools` submodule (rebuilt after the merge, out of scope).

## Suggested commit cut

1. Message: "Scenario parameters: header quadriga_gbsm.hpp, keys for LOS/indoor state distances, indoor distance
   and InF delay spread"
   Body: "The conf-file functions move unchanged from quadriga_tools.hpp to the new public header
   quadriga_gbsm.hpp (documentation section 'Geometry-based stochastic model'). Six numeric keys for TR 38.901
   v19.5.0: LOS_state_lambda, IO_state_lambda, O2I_d2D_in_max, DS_VS, DS_VS_a, DS_VS_b (171 keys); DS_VS > 0 needs
   DS_VS_a or DS_VS_b; the writer shows the V/S term in the DS overview line. The v19 files state the
   correlation distances of the LOS and indoor states, the maximum indoor distance of O2I UTs, the InF delay spread
   as a function of V/S and the excess-delay bound of InF and InH. Planning labels in the conf tests are replaced by
   the rules they stood for."
   Files: `include/quadriga_gbsm.hpp`, `include/quadriga_tools.hpp`, `include/quadriga_lib.hpp`,
   `src/qd_scenpar.cpp`, the 18 conf files, the 5 Catch2 test files, the 4 `api_mex` files, the 3 MEX test files.
2. Message: "Python: submodule quadriga_lib.gbsm with the conf-file functions"
   Body: "conf_file_read, conf_file_write, conf_file_list and scenpar_validate move from quadriga_lib.tools to
   quadriga_lib.gbsm (breaking for Python callers). Counts 172 and the DS_VS check in the documentation and tests."
   Files: `api_python/python_gbsm.cpp`, `api_python/python_tools.cpp`, `api_python/python_main.cpp`,
   `api_python/qpy_conf_file_read.cpp`, `api_python/qpy_conf_file_write.cpp`, `api_python/qpy_conf_file_list.cpp`,
   `api_python/qpy_scenpar_validate.cpp`, `CMakeLists.txt`, the 4 Python test files.
3. Message: "API conventions: header quadriga_gbsm.hpp, section 'Geometry-based stochastic model', submodule gbsm"
   Files: `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`.
   (The run report itself stays out of the commits or goes with the working documents, as the maintainer prefers.)

## Precision fix

Rule (spec 1.2, binding): conf values with the precision of their source as the TR prints them; derived values with
at most 4 significant digits (fewer where they suffice), more only where 4 digits change a result by more than
0.01 dB (path loss, SF) or 0.1 % (LSP means, distances); tolerances of TR-oracle tests come from the rounding bound.
The statements that called for full double precision are removed (1.2, 3.0.1, 4.4.1, 4.4.2, 5.3, 5.4, 5.8, Run 3a,
Run 4b, 10.2).

Conf files (5 InF files): `DS_VS = 4.27` (InF_LOS, was 47/11 at full precision) and `4.55` (InF_NLOS_DL, _SH, was
50/11); comments `50/11 = 4.55`, `47/11 = 4.27`. Effect on mu_lgDS: -2.46e-4 (LOS, -0.06 % in DS) and +3.52e-4 (DL,
SH, +0.08 % in DS). No other long number was added in Run 1.

Spec values planned for Run 3a (4.4.2), bounds against the exact constants over the TR ranges (computed with a
script of the `dual_slope` form, d2D up to 10 km for RMa/SMa):
- `PL_E = 13.33` (13 files; was 13.34): <= 2.1e-3 dB (UMa 1.95e-3, UMi 2.06e-3); InH/InF unaffected (A1 = A2).
- RMa: `PL_A1 = 20.48`, `PL_B = 31.74`, `PL_D = 0.001398`, `PL_E = 20.94`: <= 8.2e-3 dB; `PL_Bn = 31.75`: 5e-3 dB.
- SMa LOS: `PL_A1 = 21.574` (5 digits: 21.57 would give 0.023 dB at 10 km), `PL_B = 30.13`, `PL_E = 20.94`:
  <= 6.6e-3 dB.
- Change against today's files: up to 0.014 dB (RMa, mostly `PL_D` 0.0014 -> 0.001398); release-note text adjusted.
- Test tolerances: 5.3 per-file `tol_r` (UMa/UMi 2.1e-3, RMa 8.2e-3, SMa 6.6e-3, others 1e-9 dB), RMa/SMa beyond
  dBP 0.035 dB + `tol_r`; random-hE matching 2.1e-3 dB; 5.4 file-default DS means 3.6e-4 / 2.5e-4 in lg; 5.8 Run 3a
  constants equal the exact values rounded as in 4.4.2 (half a unit of the last digit).

Tests: `test_conf_files_v19.cpp` case "the InF delay spread follows the V/S rule": `DS_VS` equals the hall V/S
rounded to 2 decimals; mean within `log10(1.001)` = 4.34e-4 of the TR formula (the 0.1 % limit; the rounding gives
at most 3.52e-4), with a comment. No MEX or Python test compares these file values (no change needed).

Results (load 1-min about 3): Catch2 `test_bin "conf*"` 4618 assertions / 45 cases passed, `"scenpar*"` 3936 / 7
passed, full suite `All tests passed (1349151 assertions in 1307 test cases)`; Python (4 conf test files, module of
Run 1, wrappers unchanged) `37 passed`; Octave MEX `test_conf_file_read`, `_write`, `_list`, `test_scenpar_validate`
`OK (passed=4)`.
