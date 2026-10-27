# Code review round 1: fixes

Branch `fixes`, base HEAD 00e19aa, nothing committed. All 20 findings of `md/gbsm_lsf_code_findings.md` are
resolved (resolution cells filled): **20 fixed, 0 rejected**. One sub-item kept with a reason (C1-16, `conf()`).
Resources: at most 12 threads, one build or suite at a time; load average 3 to 11 during the work (other sessions).

## 1. Changes per finding

| ID | Change |
|---|---|
| C1-01 | Option A. `qpy_lsf_init_phases.cpp`, `qpy_lsf_draw.cpp`, `qpy_lsp_generate.cpp`: the six flags (`dual_mobility`, `remove_bias`, `horizontal`, `cholesky`, `clip_spreads`, `complex`) are `py::handle`, read with `qd_python_flag(handle, default, name)`; `m.def` defaults unchanged. One doc bullet per block. SPEC_PYTHON_API: new subsection "Flags" (and the scalar list no longer names `bool`); spec 6.2 updated. Python tests: `test_flags` in the three files (None = default, `numpy.bool_` accepted, 2.5 / 1 / str / array raise `TypeError` with the message). |
| C1-02 | `qd_gbsm_lsp_mu_sigma` (two `[n_link, n_freq, 8]` cubes) replaced by the class `qd_gbsm_lsp_stats` in the private header: the constructor evaluates the frequency terms per LSP and frequency for an LSP mask, `set_link()` takes one link, `get(i, k, mu, sigma)` returns one value pair. `path_loss` builds it with the SF bit only and writes the default SF std per link and frequency inside its link loop (no intermediate). The summation order of the mean is kept, so all exact-formula tests pass unchanged. |
| C1-03 | `los_probability.cpp`: the soft-LOS loop runs only for `n_link > 0` (no `colptr` on an empty matrix). |
| C1-04 | `lsp_generate.cpp`: column 9 of `rnd` is checked with a raw-pointer loop over `n_link` elements (no `col(9)` view). |
| C1-05 | `o2i_loss.cpp`: columns 0 to 2 and column 3 of `rnd` are checked with raw-pointer loops (no `col(3)` view). |
| C1-06 | Documented (the library's practice, cf. `qd_channel_quantize_delays.cpp`; SPEC_CPP_API only requires saying so): `path_loss`, `lsp_generate`, `los_probability`, `o2i_loss` "Outputs must not share memory with the inputs or with each other"; `lsf_init_phases`: `phase` must not share memory with the positions; `lsf_draw`: `rnd` may share memory with the positions (they are copied first). A runtime check would need up to 8 x 5 pointer-range comparisons per call (e.g. `lsp_generate`) and was not added. Side effect of C1-02: the reviewer's silent 28.4 dB error (`sf_sigma` aliasing `rx_pos`) no longer occurs (probe A3: 0 dB). |
| C1-07 | `path_loss.cpp`: random hE is a lambda (C1-13) with `p1 >= 1` -> hE = 1; C++/MEX/Python blocks and spec 3.3.6 state the rule; Catch2: UMa LOS, hUT 20 m, u = 1 at 4 km (largest hE, differs from u = 0) and 10 km (equals u = 0, finite). |
| C1-08 | `qd_gbsm_lsp_stats` throws `std::invalid_argument` for an active frequency term (`P_gamma` or `P_delta` != 0) with `P_omega + f' <= 0` (NaN included): "Scenario parameter 'DS_omega' plus the frequency in GHz must be > 0 where 'DS_gamma' or 'DS_delta' is not 0." `lsp_generate` checks all 8 LSPs, `path_loss` SF only and only with `sf_sigma`; both before any output. Docs (C++/MEX/Python, validation order), spec 3.0.4 and 3.3.3; Catch2 sections in both tests (incl. "outputs untouched" and an inactive term with `P_omega = -10` that passes). |
| C1-09 | `los_probability.cpp`: clutter attenuation lambda `exp(-d·k)` = 1 at `d = 0`; SMa clutter types with `h_x <= hU` add no term (bit-identical to the former `max(., 0)` for finite `k`, no `0·Inf`). Spec 3.5.1; Catch2: InF SL/DL/SH/DH and SMa with `d_clutter = 1e-310`. |
| C1-10 | Option A. `lsf_draw` C++ block (bullet + `floor_id` input), public header comment, MEX and Python blocks: in dual mobility `floor_id` is the floor of both devices, with the consequence for a Tx on another floor. `lsf_init_phases` takes no floors (its block already treats all positions as one floor). `qd_gbsm_eval` comment; spec 2.1, 3.2 step 3, 6.0. |
| C1-11 | Option A. RMa_NLOS, RMa_LOS_O2I, RMa_NLOS_O2I: the comment line "identical to the TR at the default heights" is replaced by "This generalizes the TR row, which uses the fixed heights for every link: it equals the TR at the default heights hBS = 35 m, hUT = 1.5 m and is 0 for UTs at or above the clutter height of 3.5 m." Spec 3.4.2 step 8 states the generalization (the spec never claimed literal equality). |
| C1-12 | Private header: `qd_gbsm_check_links(const arma::mat *tx, const arma::mat *rx)` (users: both functions of `qd_gbsm_phases.cpp`, `path_loss`, `lsp_generate`) and `qd_gbsm_check_fc` (users: `path_loss`, `lsp_generate`, `los_probability`, `o2i_loss`). One preset lookup per call: file-local `qd_gbsm_n_coeff(preset)`; `qd_gbsm_freq` no longer looks up and returns `void` (its return value was unused). Messages and validation orders unchanged. |
| C1-13 | `path_loss.cpp`: `read_dual` returns nothing; "A1 or A" / "E1 or E" throw directly through a `missing(key)` lambda that `req` also uses; `qd_pl_random_hE` -> lambda `draw_hE`; the `tripple_slope` extras in a struct `qd_pl_triple` and the `qd_pl_dual` comment says that `E` holds E1 there. `lsp_generate.cpp`: `qd_mul8` -> lambda inside `qd_sqrtm8`. `qd_sos_acf_interp` renamed `qd_interp_linear(y, x, n, x0)` with a neutral comment (all 7 call sites, incl. `sos_acfi.cpp`, `sos_remove_bias.cpp`, `sos_val.cpp`). Two `-Wshadow` hits introduced by the lambdas were renamed away. |
| C1-14 | Merge, see section 2. |
| C1-15 | Option A. `test_gbsm_phases.cpp`: `acf_at(tab, D)` calls `quadriga_lib::sos_acfi`; `test_path_loss.cpp` `zenith_att` and `test_los_probability.cpp` `ref_811` call `quadriga_lib::interp_1D` (constant extrapolation, as the library); the tables stay typed. Header comments and spec 5.0 updated. |
| C1-16 | All `err_msg` helpers removed from the five GBSM test files; the 131 converted checks use `CHECK_THROWS_MATCHES(expr, std::invalid_argument, Catch::Matchers::Message("..."))`, "<no exception>" checks `CHECK_NOTHROW`, "!= <no exception>" `CHECK_THROWS`. The `call` lambdas of the validation cases now just invoke. LSF fixtures exist once (merged file). Kept: `conf()` in `test_path_loss.cpp` and `test_lsp_generate.cpp` (a 5-line wrapper of `conf_file_read` with different signatures; a shared test header would add more than it removes). |
| C1-17 | `lsf_draw - dual mobility values follow the two-device formulas`: floors {3, 259, 0, 259, 3, 0} on the 6 links, oracle with the typed offsets added on both sides. |
| C1-18 | `lsp_generate.cpp`: one link loop writes `lsp`, `lsp_mu`, `lsp_sigma` and `eod_offset_deg` directly (no local cubes); `sf_sigma` replaces the SF std inline. |
| C1-19 | Indoor_NLOS: value comment "largest dimension L of the room [m]". |
| C1-20 | Python `d_clutter`, `r_clutter`, `h_clutter`: `py::handle`, default `None` = NaN = TR default, else a real scalar (`ValueError` "Input '<name>' must be a real scalar or None."). `floor_id` (Python and MEX): values from 2^53 on are rejected ("Entries in 'floor_id' must be non-negative integers below 2^53."), so an int64 ID that a double cannot hold no longer changes silently; docs, spec 6.0/6.2, Python and MEX tests. |

## 2. New file layout (C1-14)

- `src/qd_gbsm_phases.cpp` (585 lines) replaces `src/lsf_init_phases.cpp` and `src/lsf_draw.cpp`. Name: the `qd_`
  prefix marks a source with several public functions (as `qd_scenpar.cpp`); "gbsm_phases" because the file holds
  the phase sets of the whole model, so the later small-scale sets fit without renaming (an `lsf_` name would be
  wrong then). Content: file comment with the LSF generator index, slice convention and units; anonymous namespace
  with `qd_gbsm_gen`, `qd_gbsm_n_coeff`, `qd_gbsm_lsf_generators`, `qd_gbsm_freq`, `qd_gbsm_positions`,
  `qd_gbsm_floor_offset`, `qd_gbsm_eval`, `qd_gbsm_remove_bias` (signatures kept, `qd_gbsm_freq` returns `void`);
  then one SECTION and the two documentation blocks and definitions.
- `src/quadriga_gbsm_helpers.hpp`: 254 -> 183 lines; `qd_gbsm_check_links`, `qd_gbsm_check_fc`,
  `qd_gbsm_lsp_stats`, `qd_gbsm_chol8`; its documentation block lists the users and points to the phase file.
- `tests/catch2_tests/test_gbsm_phases.cpp` (1009 lines, 25 cases with unchanged names `lsf_init_phases - ...` /
  `lsf_draw - ...`) replaces `test_lsf_init_phases.cpp` (423) and `test_lsf_draw.cpp` (683).
- Unchanged split: `path_loss.cpp`, `lsp_generate.cpp`, `los_probability.cpp`, `o2i_loss.cpp`; wrappers one per
  function. Public header comment and `qd_test_diag.hpp` point to the new files; spec 1.4, 2.1, 3.0.5 (rewritten),
  3.2, 3.2.1, 5.0 to 5.2, 5.7, 8 and 9.1 updated (the run descriptions of section 7 are left as history).
- Overall diff of the fixes: 41 tracked files, +941 / -2450 lines, plus the two new files.

## 3. Tests and sanitizers

| Run | Result |
|---|---|
| Builds `build_linux` (AVX2 on, `test_bin`), `build_mex`, `build_py` (`--parallel 12`) | 0 warnings, 0 errors; changed sources also clean with `-Wall -Wextra -Wpedantic -Wshadow`, changed test files with `-Wall -Wextra -Wpedantic` (the CMake test flags contain `-w`) |
| Catch2 GBSM filter (`lsf_*`, `lsp_generate*`, `path_loss*`, `los_probability*`, `o2i_loss*`, `conf_file_*`, `scenpar_validate*`, `conf files v19*`, `sos_*`) | All tests passed (54559 assertions in 232 test cases) |
| `make test` (OMP 12): full Catch2 suite, AVX2 build | All tests passed (1355589 assertions in 1374 test cases) |
| `make test`: MEX suite Octave | 73 of 74 passed; the failure is `test_ray_mesh_interact.m:113` (0.98485 vs 0.69639), pre-existing since before Run 1 (r1 to r4b reports) |
| `make test`: MEX suite MATLAB R2024a | same: 73 of 74, the same pre-existing failure |
| MEX suite MATLAB R2026a | same: 73 of 74, the same pre-existing failure |
| pytest `tests/python_tests` | 876 passed, 3 skipped, 52 subtests passed |
| objdump gate `+quadriga_lib/*.mexa64` | highest GLIBCXX_3.4.26 (limit 3.4.28) |
| ASan/UBSan: `build_asan` rebuilt (g++-12, `-fsanitize=address,undefined`, AVX2, 7 min 44 s), same filter as above, `detect_leaks=1`, `halt_on_error=0`, OMP 8 | All tests passed (54559 assertions in 232 test cases); 0 runtime errors, 0 AddressSanitizer / LeakSanitizer reports (the 8 UBSan reports of the review are gone). No allocation probe under ASan. |

Load average 3.3 to 11 during the runs (other sessions); no timing in this report is a benchmark.

## 4. Probes of the review, rerun on the fixed code

Rebuilt from the review's sources against `build_linux/libquadriga.a` (C1 probe wrapped in try/catch, one SMa case
added: `probes/probe_fix1.cpp`; Python: `probes/py_flags_fix.py`, both in the session scratchpad).

| Probe | Before | Now |
|---|---|---|
| A1 `o2i_loss`, `loss` aliasing `rnd` | `std::logic_error` index out of bounds | unchanged; documented as not supported (C1-06) |
| A2 `lsp_generate`, `gr_eps_re` aliasing `rnd` | `std::logic_error` | unchanged; documented as not supported |
| A3 `path_loss`, `sf_sigma` aliasing `rx_pos` | pl off by 28.4 dB, no error | 0 dB difference (per-link SF std), still documented as not supported |
| B1 InF_SL, `d_clutter = 1e-310`, d = {0, 5} | p_los = {NaN, 0} | {1, 0}; SMa hUT {21, 10}: {1, 0} |
| C1 `DS_omega -10, DS_gamma 1` / `SF_omega -10, SF_delta 1` | DS NaN, mu NaN, sigma 0; `sf_sigma` 0 | `std::invalid_argument` with the message of C1-08 in `lsp_generate` and in `path_loss` with `sf_sigma`; `path_loss` without `sf_sigma`: no error |
| probe2 `path_loss` 1e6 links x 4 frequencies, `ulimit -v 3000000` | peak RSS 59 MB vs 559 MB, 0.11 s vs 0.73 s | 58 MB vs 89 MB (the extra 32 MB is the `sf_sigma` output itself), 0.11 s vs 0.16 s (load 8, indicative only) |
| probe3 UMa LOS, hUT 20 m, `rnd_he = 1`, d2D 4 / 6 / 10 km | 132.29 / 139.34 dB / NaN | 132.29 / 139.34 / 126.88 dB (10 km = the hE = 1 value of u = 0.999) |
| py_flags | `remove_bias=None` = False; `horizontal=None` = 3D; `complex=None` -> 7-tuple; 2.5 accepted; `d_clutter=None` TypeError | None = default for all three; 2.5, 1, 'yes' -> TypeError "Input '<name>' must be a bool or None."; `d_clutter=None` equals NaN; int64 floor ID 2^53 + 1 -> ValueError |
| D1 (sqrtm), E1 (floor 2^64 - 1) | correct | unchanged: max abs(X·X - R) 1.3e-15 ... 1.1e-9, X symmetric; finite |

## 5. Proposed commit split

The `md/` working files go into a separate last commit.

1. **Large-scale model: phase sets in one source file, shared validation, robust LSP statistics**
   `src/qd_gbsm_phases.cpp` (new), `src/lsf_init_phases.cpp`, `src/lsf_draw.cpp` (deleted),
   `src/quadriga_gbsm_helpers.hpp`, `src/path_loss.cpp`, `src/lsp_generate.cpp`, `src/los_probability.cpp`,
   `src/o2i_loss.cpp`, `src/quadriga_lib_helper_functions.hpp`, `src/sos_acfi.cpp`, `src/sos_remove_bias.cpp`,
   `src/sos_val.cpp`, `include/quadriga_gbsm.hpp`, `tests/catch2_tests/*` (new `test_gbsm_phases.cpp`, two deleted,
   four changed, `qd_test_diag.hpp`).
   Message: "lsf_init_phases and lsf_draw share one source file with the phase-set machinery; shared position and
   frequency checks; the default sf_sigma of path_loss and lsp_generate compute the LSP statistics per link
   without intermediate cubes. An undefined frequency term of the LSP formulas is an error, random hE is 1 where
   1/(1 + C) rounds to 1, a subnormal clutter size no longer gives NaN, and zero-link calls no longer bind
   references to empty matrices. Documentation states the aliasing rules and that both devices of a dual-mobility
   link are on the link's floor. Tests check messages with Catch2 matchers and use sos_acfi / interp_1D for table
   interpolation; dual-mobility floors get an oracle."
2. **Python and MATLAB wrappers: flags with None, clutter arguments with None, floor IDs below 2^53**
   `api_python/python_gbsm.cpp`, `qpy_lsf_init_phases.cpp`, `qpy_lsf_draw.cpp`, `qpy_lsp_generate.cpp`,
   `qpy_los_probability.cpp`, `qpy_path_loss.cpp`, `api_mex/lsf_draw.cpp`, `lsp_generate.cpp`, `path_loss.cpp`,
   `tests/python_tests/test_lsf_init_phases.py`, `test_lsf_draw.py`, `test_lsp_generate.py`,
   `test_los_probability.py`, `tests/mex_tests/test_lsf_draw.m`, `test_lsf_init_phases.m`.
   Message: "The flags of the gbsm submodule accept None for the default and reject non-bool values; the clutter
   arguments of los_probability accept None; floor IDs from 2^53 on are rejected instead of being rounded;
   documentation of the dual-mobility floor rule and the new error cases."
3. **Conf files: RMa EoD offset comment, office dimension**
   `data/config/3GPP_38.901_v19_RMa_NLOS.conf`, `RMa_LOS_O2I.conf`, `RMa_NLOS_O2I.conf`, `Indoor_NLOS.conf`.
   Message: "RMa: the geometric EoD offset generalizes TR 38.901 Table 7.5-9 to the actual heights; Indoor NLOS:
   absTOA_Lmax refers to the room."
4. **Planning notes** `md/gbsm_lsf_spec.md`, `md/SPEC_PYTHON_API.md`, `md/gbsm_lsf_code_findings.md`,
   `md/prompts/code_review_r1.md`, `md/prompts/code_fix_r1.md`, `md/reports/code_r1_review.md`,
   `md/reports/code_r1_fix.md`. Message: "Planning notes: code review round 1 and its fixes."
   (SPEC_PYTHON_API is persistent; it can go with commit 2 if preferred.)

Commit 1 builds and passes alone only together with the wrapper changes of commit 2 for the MEX/Python suites (the
MEX floor-ID message test and the Python flag tests belong to commit 2); the C++ library and Catch2 suite are
self-contained in commit 1.

## 6. Notes

- `html_docu/` was not regenerated (not part of the port; done by `make documentation` after the merge).
- Two scratchpad logs of earlier runs (`make_test.log`, `mex_build.log`) were overwritten by the first runs of this
  work; the final logs are `fix_*.log` in the session scratchpad.
