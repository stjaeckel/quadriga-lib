# Code review round 2: fixes

Branch `fixes`, base HEAD 83301c0, nothing committed. All 8 findings of `md/gbsm_lsf_code_findings.md` (C2-01 to
C2-08) have their resolution cell filled: **7 fixed, 1 rejected** (C2-08, by the user's decision B). C2-05 is fixed
with the user's option B. Resources: at most 12 threads, one build or suite at a time; load average 1.6 to 5.5
during the work (other sessions). No timing in this report is a benchmark.

## 1. Changes per finding

| ID | Change |
|---|---|
| C2-01 | `src/lsp_generate.cpp` `qd_sqrtm8` now returns `bool`. It stores the iterate `Y` with the smallest error `e` (the `Y` that `e` measures), stops when `e` is not finite, and on convergence (`e <= 1e-10`) keeps the converged update as before. It returns false if the symmetrized root `S` is not finite; `lsp_generate` then throws the existing "LSP cross-correlation matrix in 'scenpar' is not positive definite." before any output is written. Matrices that converge give bit-identical results (300 random correlation matrices each with smallest eigenvalue 1e-2 and 1e-6: all identical to the old iteration). Catch2 adds the reviewer's DS/KF/SF block, which is singular to rounding (`-0.6000000000000001, 0.8, -2.2e-16`), to "the symmetric square root is the principal root of R": the root is finite and symmetric, with residual 1.9e-12 (tolerance 1e-10). Spec 3.4.1 item 4 is extended and the wrong claim in spec 3.4.3 is corrected (new pseudo-code and measured residuals). |
| C2-02 | `qd_gbsm_lsp_stats` constructor (private header): for every evaluated LSP these coefficients must be finite: `P_sigma`, `P_gamma`, `P_delta`, `P_epsilon`, `P_zeta`, `P_alpha`, `P_kappa`, `P_tau`, `P_beta`, plus `ES_D_mu_A` to `_D` for ESD and `P_omega` of an active frequency term. Otherwise it throws "Scenario parameter '<key>' must be finite." `P_mu` and `ES_D_mu_min` stay unchecked because an infinite value gives no NaN (`ES_D_mu_min = -Inf` is the default). **Extended beyond the proposal:** `lsp_generate` reads other coefficients of the same class (they can also be ±Inf and then give `Inf·0 = NaN`). The `EoD_offset_e`, `_e_gamma`, `_a`, `_a_gamma`, `_c`, `_c_gamma`, `_c_hUT` coefficients must be finite when `eod_offset_deg` is requested, and `absTOA_sigma` when `abs_toa` is requested and active. `path_loss` now builds the stats with an empty LSP mask when `sf_sigma` is not requested, so the new SF checks also run only with `sf_sigma`, as the spec requires. Docs: C++, MEX and Python blocks of `lsp_generate` and `path_loss` (formula bullet, TOA and EoD bullets, validation order); spec 3.0.4, 3.0.5, 3.3.3, 3.4.1 items 6 and 7. Catch2: section "infinite coefficients" in the `lsp_generate` validation (5 keys; outputs untouched; EoD and TOA only with their outputs; inactive TOA not checked; `XPR_mu = -Inf` allowed). The `path_loss` SF section adds `SF_kappa = Inf` with and without `sf_sigma`. |
| C2-03 | `path_loss` `opt()` (also used by `req()`) throws "Path-loss parameter '<key>' must be finite." for every key a model reads, in all 10 models. `G2n` of `nlos` must be > 0 ("Path-loss parameter 'G2n' must be > 0."); it is rejected even when `G1n = 0`, because `0·(lg 0)²` is NaN and the v19 files all have `G2n = 11.75`. Docs: the validation bullet of all three blocks; spec 3.3.1 items 4 and 5. Catch2: new section "finite parameters": every read key of every model is set to NaN or −Inf, plus the optional `SF` and `D`, and a NaN in a key the model does not read is ignored. The "breakpoint factors" section adds `G2n` 0 and −1. |
| C2-04 | The SF message names only the existing key: "... must be > 0 where 'SF_delta' is not 0." Expectations updated in `test_lsp_generate.cpp` and `test_path_loss.cpp`; spec 3.0.4. |
| C2-05 | Option B. `qd_sqrtm8` stays where it is (the only helper of `lsp_generate.cpp`, directly above the documentation block). `qd_los_probit` is moved to the end of the anonymous namespace of `los_probability.cpp`, so it sits directly above the documentation block of its single caller (the code is unchanged). `md/SPEC_CPP_API.md`, "File-local helpers": a helper used once is a lambda, except a self-contained algorithm of more than about 20 lines, which may be a named file-local function placed directly above its single caller (for a public function: above its documentation block); a helper used several times within one file is `static inline` or in the anonymous namespace. Spec 3.0.5 states the rule for the GBSM helpers. |
| C2-06 | `path_loss`: the 25-line if/else chain is replaced by a name table and a loop, as in `o2i_loss` (11 names; the empty name is "no model"). The enum stays for the `switch` labels, because they read better than numbers. `qd_pl_dual dual = {}` and `qd_pl_triple tri = {}` replace the positional zero lists. A sentinel enum value would trigger `-Wswitch`, so the table size is `constexpr int n_model = m_satellite + 1`. |
| C2-07 | Section overview in `src/qd_scenpar.cpp`, `api_mex/conf_file_read.cpp` and `api_python/qpy_conf_file_read.cpp`: "draws the LOS state and the indoor state ... at the decorrelation distances of the keys `LOS_state_lambda` and `IO_state_lambda`, and the O2I variables at the distances of TR 38.901 clause 7.6.3.3 (see [[o2i_loss]])". The SECTION_DESC requirement in the Run 4a part of the spec is updated in the same way. |
| C2-08 | Rejected (option B, the user's decision): the file layout is kept. |

Overall diff (code, tests, wrappers): 13 files, +261 / −112 lines. Docs in `md/`: `SPEC_CPP_API.md`, `gbsm_lsf_spec.md` and `gbsm_lsf_code_findings.md`.

## 2. File layout (unchanged)

`include/quadriga_gbsm.hpp`; `src/quadriga_gbsm_helpers.hpp` (`qd_gbsm_check_links`, `qd_gbsm_check_fc`,
`qd_gbsm_lsp_stats`, `qd_gbsm_chol8`); `src/qd_gbsm_phases.cpp` (`lsf_init_phases`, `lsf_draw` and the file-local
phase-set machinery); `src/path_loss.cpp`, `src/lsp_generate.cpp` (`qd_sqrtm8` file-local),
`src/los_probability.cpp` (`qd_los_probit` file-local), `src/o2i_loss.cpp`; one wrapper file per function for MEX
and Python.

## 3. Tests and sanitizers

| Run | Result |
|---|---|
| Builds `build_linux` (g++-12, AVX2 on, `test_bin`), `build_mex`, `build_py` (`--parallel 12`), each installed | 0 warnings, 0 errors |
| `-fsyntax-only -Wall -Wextra -Wpedantic -Wshadow` with g++-12 and clang++-21 (clang also `-Wmicrosoft -Wlanguage-extension-token`): `path_loss.cpp`, `lsp_generate.cpp`, `los_probability.cpp`, `qd_scenpar.cpp`, `o2i_loss.cpp`, `qd_gbsm_phases.cpp`; both changed test files with g++ `-Wall -Wextra -Wpedantic` | no diagnostics |
| Catch2 GBSM filter (`lsf_*`, `lsp_generate*`, `path_loss*`, `los_probability*`, `o2i_loss*`, `conf_file_*`, `scenpar_validate*`, `conf files v19*`, `sos_*`) | All tests passed (54637 assertions in 232 test cases; 54559 before) |
| `make test` (`CMAKE_BUILD_PARALLEL_LEVEL=12`, OMP 12): full Catch2 suite, AVX2 build | All tests passed (1355667 assertions in 1374 test cases); the only message is the existing qdant locale warning ("No locale with a decimal comma installed") |
| `make test`: MEX suite Octave | 73 of 74 passed. The one failure is `test_ray_mesh_interact.m:113` (0.98485 vs 0.69639); it is older than this work and unchanged (see the round-1 report) |
| `make test`: MEX suite MATLAB R2024a (`matlab24`) | 73 of 74, the same failure that was already there |
| MEX suite MATLAB R2026a (26.1.0.3203278) | 73 of 74, the same failure that was already there |
| pytest `tests/python_tests` (OMP 12) | 876 passed, 3 skipped, 52 subtests passed |
| objdump gate `+quadriga_lib/*.mexa64` | highest GLIBCXX_3.4.26 (limit 3.4.28) |
| ASan/UBSan: `build_asan` `test_bin` rebuilt (g++-12, `-fsanitize=address,undefined`, AVX2; 8 changed objects, 1 min 10 s), same filter as above, `detect_leaks=1`, `halt_on_error=0`, OMP 8 | All tests passed (54637 assertions in 232 test cases); **0 runtime errors, 0 AddressSanitizer / LeakSanitizer reports**. No allocation probe ran under ASan. |

Load average during `make test` 5.5 → 4.5, ASan run 2.2 to 2.4.

## 4. Probes, rerun on the fixed code

The review's sources were copied to `scratchpad/r2fix/` and built with the review's `build.sh` against the new
`build_linux/libquadriga.a`. The C++ comparison of old and new iteration is `scratchpad/r2fix/sq.cpp`; the Python
replay through the library is `scratchpad/r2fix/pystat.py`.

| Probe | Before (review) | Now |
|---|---|---|
| p2 D4 `ds_kf -0.6000000000000001, ds_sf 0.8, sf_kf -2.2e-16`, `cholesky = false` | all 8 LSPs NaN | finite, `max abs(X·X - R)` 1.9e-12; Cholesky unchanged (4.4e-16) |
| p2 grid of singular 3x3 blocks (361) | 110 accepted, 26 NaN with `cholesky = false` (16 of them although both Cholesky orders succeed) | 110 accepted, **0 NaN**; 24 fail the TR-order Cholesky (documented error, unchanged) |
| p2 near-singular correlations (C4 to C6, 1 − 1e-9 … 1 − 2.2e-16) | residual 2e-12 … 1.2e-9 | unchanged (2.3e-12, 2.8e-10, 1.2e-9) |
| p1 B random rank-7 matrices | 11 of 20 accepted, 3 fail TR-order Cholesky; `cholesky = false` NaN for 2 of 3 | Cholesky: documented error (unchanged); `cholesky = false` finite for all 3 |
| Python statistics, 200 random 8x8 correlation matrices per smallest eigenvalue, `cholesky = False`, accepted by `scenpar_validate` | 1e-16: 20 of 178 NaN; 0: 29 of 94 NaN | 1e-6: 0 NaN (residual 2.7e-13); 1e-15: 0 NaN (2.7e-8); 1e-16: 0 of 173 NaN (3.7e-7); 0: 0 of 104 NaN (1.1e-7); no root throws |
| C++ old vs new iteration (`sq.cpp`, 300 matrices per eigenvalue) | — | 1e-2, 1e-6: bit-identical; 1e-12: residual ≤ 7.1e-10 (old 1.1e-9); 1e-15: ≤ 3.0e-8 (old 4.6e-8); 1e-16 / 0: old 28 / 14 NaN, new 0 NaN, residual ≤ 4e-7 |
| p1 A `nlos`, `An = Bn = 30`, `LOSmax = 0`, `G2n = 0` / `-1` | `pl` NaN, no error | "Path-loss parameter 'G2n' must be > 0." (both) |
| p3 `DS_omega = Inf` with `DS_delta = 1` / `DS_gamma = 1` | DS NaN (sigma Inf) / Inf | "Scenario parameter 'DS_omega' must be finite." (both) |
| p3 `SF_kappa = Inf` | `lsp` SF NaN; `path_loss` `sf_sigma` Inf | "Scenario parameter 'SF_kappa' must be finite." in `lsp_generate` and in `path_loss` with `sf_sigma` |
| p3 SF frequency-term message | names `SF_gamma` (no such key) | "... must be > 0 where 'SF_delta' is not 0." |
| p4 `logdist` `B = NaN`; `dual_slope` `E = Inf`; conf file `PL_B = NaN` | NaN loss; finite loss; conf rejects | "Path-loss parameter 'B' must be finite."; "Path-loss parameter 'E' must be finite."; conf unchanged ("Value of 'PL_B' is not a number.") |
| p5 floors, soft LOS (no bug) | identical to the bit; `los_soft` 5.6e-4, 5.5e-4, 0 | unchanged |
| Round-1 aliasing probe (`probe_fix1.cpp`) | A1, A2 `std::logic_error`; A3 0 dB | unchanged (aliasing documented as not supported); B1, B2, C1, D1 (sqrtm residual 1.3e-15 … 1.1e-9), E1 unchanged; C1 SF message now names only `SF_delta` |
| `path_loss` memory (round-1 `probe2.cpp`, 1e6 links × 4 frequencies, `ulimit -v 3000000`, plain build) | 58 MB / 89 MB, 0.11 / 0.16 s | 58 MB without / 89 MB with `sf_sigma` (32 MB of that is the output itself), 0.10 / 0.15 s (load 1.7, indicative only) |

## 5. Proposed commit split

The `md/` working files go into a separate commit. The untracked `md/prompts/*_r3.md` files were not created by
this work and are left out.

1. **Large-scale model: robust symmetric root, finite parameters, path-loss model table**
   `src/lsp_generate.cpp`, `src/quadriga_gbsm_helpers.hpp`, `src/path_loss.cpp`, `src/los_probability.cpp`,
   `src/qd_scenpar.cpp`, `api_mex/lsp_generate.cpp`, `api_mex/path_loss.cpp`, `api_mex/conf_file_read.cpp`,
   `api_python/qpy_lsp_generate.cpp`, `api_python/qpy_path_loss.cpp`, `api_python/qpy_conf_file_read.cpp`,
   `tests/catch2_tests/test_lsp_generate.cpp`, `tests/catch2_tests/test_path_loss.cpp`.
   Message: "The symmetric square root of lsp_generate (cholesky = false) keeps its best iterate and stops when the
   iteration diverges, so correlation matrices that are singular to rounding give a finite root instead of NaN;
   a root that is still not finite is reported as not positive definite. Regular matrices give bit-identical
   results. Infinite LSP coefficients (spread, frequency, distance, height, elevation and ESD terms; the EoD
   offset and absolute TOA coefficients when those outputs are requested) and non-finite path-loss parameters
   are rejected instead of producing NaN; G2n of the nlos model must be > 0. The SF frequency-term message names
   only SF_delta. path_loss finds the model in a name table. The section overview states that the O2I variables
   use the distances of TR 38.901 clause 7.6.3.3. Tests cover the singular-to-rounding root and the new checks."
2. **Planning notes: code review round 2 and its fixes**
   `md/SPEC_CPP_API.md`, `md/gbsm_lsf_spec.md`, `md/gbsm_lsf_code_findings.md`, `md/prompts/code_review_r2.md`,
   `md/prompts/code_fix_r2.md`, `md/reports/code_r2_review.md`, `md/reports/code_r2_fix.md`.
   `SPEC_CPP_API.md` is persistent, so its helper rule can go into a commit of its own, for example "C++ API guidelines: single-use helpers".

## 6. Notes

- `html_docu/` was not regenerated (done by `make documentation` after the merge).
- Logs: `scratchpad/r2fix/` (`make_test.log`, `matlab26.log`, `pytest.log`, `asan_build.log`, `asan_run.log`,
  `mex_build.log`, `py_build.log`) in the session scratchpad
  `/tmp/claude-1005/-home-stephan-sjc-quadriga-lib-fixes/240c04ce-057b-420a-a310-dd24cc94ffbc/scratchpad/`.
