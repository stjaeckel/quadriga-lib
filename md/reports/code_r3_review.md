# Code review round 3 (final): large-scale model port

Reviewer: independent agent (did not write the code and did not do rounds 1 and 2). Scope: `git diff 46dd28d..HEAD
-- . ':!md'` at HEAD b10f332 (branch `fixes`), 90 files, +11979 / -530 lines; no uncommitted code changes. Findings
table: `md/gbsm_lsf_code_findings.md` (C3-01 to C3-04).

## 1. Method

1. Read the round-1 and round-2 findings with their resolutions, `code_r1_review.md`, `code_r2_review.md`,
   `code_r1_fix.md`, `code_r2_fix.md`, and the parts of the spec needed for the checks (2.1, 3.0.4/3.0.5, 4.4.1,
   8, plan decision D8).
2. Read every line of the public header, the private header, `qd_gbsm_phases.cpp`, `path_loss.cpp`,
   `lsp_generate.cpp`, `los_probability.cpp`, `o2i_loss.cpp`, the `qd_scenpar.cpp` diff, the six MEX and six
   Python wrappers with `python_gbsm.cpp`, the documentation blocks of `lsp_generate`, `path_loss`, `lsf_draw` and
   `los_probability` in all three languages, the helpers and selected cases of the five GBSM Catch2 files, the
   diff of `test_conf_files_v19.cpp`, and the conf-file diffs of UMa NLOS, InF NLOS DL, RMa NLOS O2I, SMa NLOS O2I
   plus a grep of the new keys over all 23 files.
3. Re-checked against TR 38.901 v19.5.0: Table 7.4.2-1 (all rows), Tables 7.4.3-1 to -3 and 7.4.3.2, eq. (7.6-18),
   Table 7.6.3.1-2, the UMa random hE of Table 7.4.1-1 note 1, the ZSD rows of Tables 7.5-7 to 7.5-9 (ES_D_mu_A to
   _D of the files), the EoD-offset sign and constants, the V/S rule of Table 7.5-6 Part-3; QuaDRiGa-NG `get_pl.m`
   (`winner_nlos` frequency bands) and `set_scenario.m` (LOS-state distances).
4. Builds, sanitizer run and tests (section 2); probes (section 3); generated the C++, MEX and Python Markdown with
   `tools/extract_md.py` into the scratchpad and checked the links of the ten GBSM blocks; grep of all added lines
   for internal references; `-fsyntax-only` of the five new sources and `qd_scenpar.cpp`.

Resources: at most 12 threads, one build or suite at a time; load average 0.5 to 3.6 during the work. No timing in
this report is a benchmark.

## 2. Builds, sanitizers and tests

| Run | Result |
|---|---|
| `cmake --build build_asan --parallel 12 --target test_bin` (g++-12, `-fsanitize=address,undefined -fno-omit-frame-pointer -g`, AVX2 on) | up to date (binary built from the current sources at 12:52; no source newer), 1.1 s |
| `build_asan/test_bin "lsf_*,lsp_generate*,path_loss*,los_probability*,o2i_loss*,conf_file_*,scenpar_validate*,conf files v19*,sos_*"`, OMP 8, `ASAN_OPTIONS=detect_leaks=1:halt_on_error=0`, `UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0` | All tests passed (54637 assertions in 232 test cases), 35 s; **0 UBSan runtime errors, 0 AddressSanitizer / LeakSanitizer reports**; load 1.0 before, 3.6 after |
| `build_linux/test_bin` same filter without `sos_*` (plain, AVX2) | All tests passed (15070 assertions in 119 test cases) |
| pytest of the ten gbsm files (OMP 8) | 97 passed |
| g++-12 `-fsyntax-only -Wall -Wextra -Wpedantic -Wshadow -Wconversion` (six sources); clang++-21 with `-Wmicrosoft -Wlanguage-extension-token` (five new sources) | no diagnostic |

No allocation probe was needed (all allocations scale with the inputs); nothing but the suite ran under ASan. The
MEX suites were not run (no MEX source changed since the round-2 fix run, which shows them green).

## 3. Probes and results

Sources in `/tmp/claude-1005/-home-stephan-sjc-quadriga-lib-fixes/240c04ce-057b-420a-a310-dd24cc94ffbc/scratchpad/r3/probes/`
(C++ against `build_linux/libquadriga.a`, `-DARMA_DONT_USE_BLAS`; Python against `lib/quadriga_lib...so`).

| Probe | What | Result |
|---|---|---|
| `q1.cpp` | `path_loss`, UMa LOS v19 file, hBS 25 m, 3.5 GHz, hUT 0.5 to 13.2 m, d2D 20 to 300 m, with and without `rnd_he` (u = 0.3; C' = 0 below 13 m, so hE = 1 in both) | hUT 0.5 m: 71.88 / 77.29 / 83.16 / 100.97 dB without, 49.41 / 59.24 / 69.92 / 88.55 dB with `rnd_he`; hUT 1.0 m: +2.1 to +12.2 dB; hUT 1.05 m: equal up to 50 m, +4.1 / +4.9 dB at 100 / 300 m; equal from 1.1 m on (C3-01) |
| `py1.py` | Python argument handling of the six wrappers: clutter scalars as int / float32 / 1-element array / bool, `h_bs` as float, `floor_id` float / negative / int64 above 2^53 / list, flags `None` / `1` / `numpy.bool_`, `preset=None`, `seed` None / -2 / float / 2^63, `sf_sigma` `[]` / 1-D, `fc` scalar, 1-D `rx_pos`, `plpar=None`, `models` as str, `d2d_in_max=None` | all as documented: floor IDs and flags rejected with the documented messages, `None` = default for flags and clutter arguments, arrays accepted as lists; `d2d_in_max=None` and `seed=None` are TypeErrors (plain scalars, as in `sos.init_phases`); `d_clutter=True` is accepted as 1.0 (pybind11 real-scalar conversion, not worth a finding) |
| round-2 probes `p1`, `p2`, `p3`, `p4` rerun | `G2n` 0 / -1, random rank-7 matrices, singular 3x3 blocks, infinite LSP coefficients, non-finite path-loss parameters | all fixed: "Path-loss parameter 'G2n' must be > 0.", "... 'DS_omega' / 'SF_kappa' must be finite." (also in `path_loss`), SF message names only `SF_delta`, "... 'B' / 'E' must be finite."; singular grid: 110 accepted, 0 non-finite with `cholesky = false` (was 26) |
| docs | `extract_md.py` for cpp, mex, python; links of the ten GBSM blocks | no broken link, no unresolved `[[...]]` |
| grep | added lines of the 90 files for `md/`, spec sections, plan / review IDs, local paths, TODO | none (the `D1` hits are model keys and TR cluster counts) |
| conf keys | `LOS_state_lambda`, `IO_state_lambda`, `O2I_d2D_in_max`, `absTOA_Lmax`, `DS_VS` over the 23 files | values as in TR Table 7.6.3.1-2, clause 7.4.3.1, clause 7.6.9 and Table 7.8-7; the two state distances only in NLOS resp. O2I files (C3-02) |

## 4. Round-2 and round-1 fixes verified

| ID | Check | Status |
|---|---|---|
| C2-01 | `qd_sqrtm8` keeps the best iterate, stops on a non-finite error, returns false for a non-finite root, caller throws before any output; probe p2 0 NaN; Catch2 check of the singular-to-rounding block present | verified |
| C2-02 | constructor of `qd_gbsm_lsp_stats` checks `_sigma` ... `_beta`, `ES_D_mu_A` to `_D`, `_omega` of active terms; `lsp_generate` checks the EoD coefficients with `eod_offset_deg`, `absTOA_sigma` with active `abs_toa`; `path_loss` builds the stats with an empty mask without `sf_sigma`. The non-negative keys (`LSP_fc_min`, `EoD_offset_b`, `EoD_offset_hcl`, `absTOA_Lmax`, `DS_VS*`) are finite by their class, so no Inf·0 path is left | verified |
| C2-03 | `opt()` throws for non-finite values of every read key; `G2n > 0` | verified |
| C2-04 | SF message | verified |
| C2-05 | `qd_sqrtm8`, `qd_los_probit` named file-local, directly above their caller's block | verified (user decision) |
| C2-06 | name table and loop in `path_loss`, `qd_pl_dual dual = {}` | verified |
| C2-07 | section overview in the three languages | verified |
| C1-01 to C1-20 | spot re-check of each resolution in the current code (flags, `qd_gbsm_lsp_stats` per link, the three `n_link = 0` guards, aliasing sentences, `draw_hE` with `p1 >= 1`, clutter lambda, dual-mobility floor sentence, RMa EoD comment, shared validation helpers, merged phase file, Catch2 matchers, `sos_acfi` / `interp_1D` in the tests, floor oracle in dual mobility, direct writes in `lsp_generate`, Indoor `absTOA_Lmax` comment, Python `None` clutter arguments and the 2^53 floor rule) | no regression; the sanitizer run stays clean |

## 5. Findings rationale

One major, one minor, two nits. No blocker.

- **C3-01 (major), random hE below 1.1 m.** The random environment height of note 1 drops the `1.1·hE` clamp of
  the UT height because `hUT - hE >= 1.5` holds for the drawn heights 12 ... hUT - 1.5. But the draw also returns
  hE = 1 (always for hUT <= 13 m), and then the missing clamp makes `hUT - hE` zero or negative for hUT <= 1 m:
  the breakpoint distance is <= 0 and the loss is evaluated "beyond the breakpoint" at every distance. The UMa v19
  files enable `hE_rand`, and NG will pass column 10 of `lsf_draw` as `rnd_he`, so every UMa link with a UT below
  1.1 m changes its loss silently by -22 to +12 dB compared with the same call without `rnd_he` (hE = 1 in both),
  down to 20 dB below free space. The TR range of UMa starts at hUT = 1.5 m, which keeps this from being a
  blocker, but the library otherwise evaluates its models "as written also outside the validity range" and the
  fixed-hE path handles these heights. The fix is one expression in `qd_pl_dual_slope`
  (`hm = max(hUT, hE + 0.1, 0.1)` in random mode), which equals the fixed mode for hE = 1 and leaves the TR set
  unchanged, plus the doc sentence and a Catch2 check. The existing random-hE test covers hUT 10, 13.2, 19.5, 20,
  22.5 m only.
- **C3-02 (minor), state distances only in some files.** Plan decision D8 stores each value once (LOS state in the
  NLOS file, indoor state in the O2I files). The produced documentation does not say so, and the key descriptions
  define 0 as "uncorrelated", so a caller who takes the key from the file of the link state draws uncorrelated LOS
  states for LOS links of UMa (TR: 50 m for all links of the scenario). Documentation fix proposed; filling all
  files would change the decision and is the user's call.
- **C3-03 (nit)**: conf comments (`>= 13` vs. note 1, 0-based column in a language-neutral file, duplicated
  "constants evaluated" sentences, a missing rounding sentence in the SMa NLOS files).
- **C3-04 (nit)**: one 15-line single-use helper outside the lambda rule; optional.

### Checked and found correct (fresh eyes)

- `qd_gbsm_eval`: grouping by floor and table, Tx side with `tab[m(g)]` and slice 1, dual-mobility combination,
  exact swap reciprocity (sums commute), Uniform map after the combination; `qd_gbsm_floor_offset` for IDs up to
  2^64 - 1; bias removal over `[R; T]`; validation orders of both functions as documented.
- `path_loss`: all ten models, the `winner_nlos` bands (= `get_pl.m`), `nlos` maximum with the random-hE LOS term
  and the InF-SL term, `tripple_slope`, satellite, legacy outdoor distance, `draw_hE` set size and probability,
  default `sf_sigma` with the geometric distance; validation order as documented.
- `lsp_generate`: TR-order Cholesky and its back-permutation, `x_SF = s_SF`, ESD terms against Tables 7.5-7 to
  7.5-9 (UMa/RMa `ES_D_mu_B = -0.01`, UMi LOS `_C = 0.01`, UMi NLOS `_D = 0.01`, floors -0.5 / -1 / -0.21), EoD
  sign and the geometric RMa/SMa term, TOA bound, permittivity interpolation, clipping; the v19 NLOS files have no
  KF correlations, so the K row in the TR order does not change the other factors.
- `los_probability`: Table 7.4.2-1 rows incl. the InF and SMa edge rules and the limit to [0, 1]; eq. (7.6-18)
  with `sqrt(20 fc / c)`. `o2i_loss`: Tables 7.4.3-1 to -3, car losses, cumulative model choice with zero weights.
- Wrappers: argument counts, optional outputs passed as nullptr (MEX checks of column 9 and the EoD/TOA
  coefficients only with their outputs, Python always), 1-based columns and `model_index` in MEX, flags and `None`
  rules in Python; documentation blocks consistent across the three languages.
- MSVC / rules: no GCC extensions, VLAs, OpenMP, `M_PI`, BLAS/LAPACK calls or floating-point `std::to_chars`;
  clean under `-Wconversion` and clang's `-Wmicrosoft`.

## 6. File structure

Current layout: `include/quadriga_gbsm.hpp` 182 lines; `src/quadriga_gbsm_helpers.hpp` 204 (`qd_gbsm_check_links`,
`qd_gbsm_check_fc`, `qd_gbsm_lsp_stats`, `qd_gbsm_chol8`); `src/qd_gbsm_phases.cpp` 585 (`lsf_init_phases`,
`lsf_draw` and the file-local machinery); `src/path_loss.cpp` 585; `src/lsp_generate.cpp` 405;
`src/los_probability.cpp` 393; `src/o2i_loss.cpp` 216. The remaining merge candidates were evaluated in rounds 1
and 2: `path_loss` + `lsp_generate` would only make `qd_gbsm_lsp_stats` file-local (rejected, C2-08),
`los_probability` + `o2i_loss` share nothing but `qd_gbsm_check_fc` (no header entry would go). The user's
decision "no further file merging" is final and I see no reason to reopen it: each file holds one public function
(or the two phase-set functions with their machinery), its documentation block and only its own helpers; the
private header holds exactly the four cross-file helpers with their users.

## 7. SSF readiness

Unchanged since round 2 and still met: the machinery in `qd_gbsm_phases.cpp` is generic over the generator table
(`qd_gbsm_gen`), groups `sos_val` / `sos_remove_bias` calls per table, side and floor, and handles mirrors, floors,
2D/3D and dual mobility per generator; SSF phase-set functions can be added to that file without signature
changes. The private-header helpers are reusable unchanged; `lsp_generate` and `los_probability` provide every
quantity of spec section 8. C3-04 does not affect this (a lambda inside `qd_gbsm_eval` is still used by every
phase set through `qd_gbsm_eval`).

## 8. Untested areas

- `path_loss` with `rnd_he` and hUT below 1.1 m (C3-01).
- The `return false` path of `qd_sqrtm8` (a root that stays non-finite): no matrix accepted by `scenpar_validate`
  reaches it in the probes; the message is shared with the tested Cholesky path.
- MEX with zero links (accepted restriction); `lsf_init_phases` dual-mobility bias removal with a one-row `tx_pos`
  (documented as `2·n_link` positions, not pinned by a test).

## 9. Overall judgement

The port is correct and robust in the TR ranges, the round-1 and round-2 fixes are complete and did not regress,
the sanitizer run is clean, the wrappers follow the flag, `None` and index rules, and no produced file carries an
internal reference. The one substantive issue is C3-01, a silent path-loss error of up to 22 dB for UMa UTs below
1.1 m when `rnd_he` is passed; it has a one-line fix. **Ready to merge into `main` after C3-01 is fixed** (with
its Catch2 check); C3-02 should go in with it (documentation only), C3-03 and C3-04 are optional.
