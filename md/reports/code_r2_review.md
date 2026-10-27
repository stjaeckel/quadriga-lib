# Code review round 2: large-scale model port

Reviewer: independent agent (did not write the code and did not do round 1). Scope: `git diff 46dd28d..HEAD -- .
':!md'` at HEAD 83301c0 (branch `fixes`), 90 files, +11830 / -530 lines; no uncommitted code changes (only the
untracked prompt `md/prompts/code_review_r2.md`). Findings table: `md/gbsm_lsf_code_findings.md` (C2-01 to C2-08).

## 1. Method

1. Read the round-1 findings with their resolutions, `md/reports/code_r1_review.md` and `md/reports/code_r1_fix.md`,
   then the spec (sections 1 to 6, 8, 9 in full; 7 and 10 skimmed), SPEC_CPP_API (outputs, tests, BLAS rule) and the
   Python flag rule of SPEC_PYTHON_API.
2. Read every line of the six new sources, the public header, the private header, the `qd_scenpar.cpp` diff, the
   renamed `qd_interp_linear` and its six call sites, the six MEX and six Python wrappers with `python_gbsm.cpp`, the
   helpers of the five Catch2 test files plus selected cases, one MEX test, and the conf-file diffs.
3. Checked formulas against TR 38.901 v19.5.0 again where round 1 was light or the code changed: Table 7.4.1-1 with
   notes 1 and 5 (UMa PL2 with `-9 lg(dBP'^2 + (hBS - hUT)^2)` equals the `dual_slope` form with the 3D breakpoint
   distance exactly; RMa/SMa constants), Table 7.4.2-1, 7.4.3 (Tables 7.4.3-1 to -3, 7.4.3.2), clause 7.5 step 4
   (Cholesky order, clipping), Tables 7.5-7 to 7.5-12 with their notes (NOTE 4: the 6 GHz clamp of the UMa and InH
   ZOD terms, i.e. `LSP_fc_min` for the EoD offset is TR), Table 7.6.3.1-2, clauses 7.6.3.3, 7.6.9 and Table 7.8-7
   (halls of SL/DH 120 x 60 m, DL/SH 300 x 150 m: V/S 4 and 50/11, L 120 and 300 m); QuaDRiGa-NG `set_scenario.m`
   for the 38.811 tables and `QuaDRiGa_Industrial`.
4. ASan/UBSan run of the new and changed test cases (section 2), pytest of the ten gbsm Python test files.
5. Probes in the scratchpad (section 3), C++ against `build_linux/libquadriga.a` (up to date with the sources),
   Python against `lib/quadriga_lib...so` (up to date with the wrappers).
6. Generated the C++, MEX and Python Markdown documentation with `tools/extract_md.py` into the scratchpad and read
   the new section as a user would (overview, the six blocks, link check); `html_docu/` itself was not touched.
7. Grep of all 90 produced files for internal references; `-fsyntax-only` of the five new sources with g++-12
   (`-Wall -Wextra -Wpedantic -Wshadow`) and of these plus `qd_scenpar.cpp` with clang 21 (same set plus
   `-Wmicrosoft -Wlanguage-extension-token`).

Resource use: at most 12 threads, one build or suite at a time; load average 13 to 15 during the work (other
sessions). No timing in this report is a benchmark.

## 2. Builds and test runs

| Run | Result |
|---|---|
| `cmake --build build_asan --parallel 12 --target test_bin` (g++-12, `-fsanitize=address,undefined -fno-omit-frame-pointer -g`, AVX2 on) | up to date (built from the current sources by the round-1 fix session; make found nothing to rebuild), 1.2 s |
| `build_asan/test_bin "lsf_*,lsp_generate*,path_loss*,los_probability*,o2i_loss*,conf_file_*,scenpar_validate*,conf files v19*,sos_*"`, `OMP_NUM_THREADS=8`, `ASAN_OPTIONS=detect_leaks=1:halt_on_error=0`, `UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0` | All tests passed (54559 assertions in 232 test cases), 38 s, load 14.6; **0 UBSan reports, 0 AddressSanitizer / LeakSanitizer reports**: the three round-1 UBSan sites (C1-03 to C1-05) stay clean |
| pytest of `test_lsf_init_phases`, `test_lsf_draw`, `test_path_loss`, `test_lsp_generate`, `test_los_probability`, `test_o2i_loss`, `test_conf_file_read/write/list`, `test_scenpar_validate` (`OMP_NUM_THREADS=8`) | 97 passed |
| g++-12 and clang 21 `-fsyntax-only` with the warning sets above (five new sources; clang also `qd_scenpar.cpp`) | no warning |

No allocation probe was needed (all allocations scale with the inputs); nothing ran under ASan except the suite.
The plain Catch2 suite, the MEX suites and the full pytest were not rerun: the fix report shows them green on the
same code, and only `md/` changed since.

## 3. Probes and results

Sources in `/tmp/claude-1005/-home-stephan-sjc-quadriga-lib-fixes/240c04ce-057b-420a-a310-dd24cc94ffbc/scratchpad/r2/probes/`
(build script `build.sh`, `-DARMA_DONT_USE_BLAS`).

| Probe | What | Result |
|---|---|---|
| `p1.cpp` A | `nlos`, `An = Bn = 30`, `LOSmax = 0`, `G2n = 0` and `-1` (G1n default 0) | `pl` NaN, no error (C2-03) |
| `p1.cpp` B | random rank-7 correlation matrices: `scenpar_validate` result vs `qd_gbsm_chol8` in TR order | 11 of 20 accepted, 3 of them fail the TR-order Cholesky: `lsp_generate` throws "LSP cross-correlation matrix in 'scenpar' is not positive definite." as documented (the round-1 untested path works); `cholesky = false` gives NaN for 2 of the 3 (C2-01) |
| `p2.cpp` | singular 3x3 DS/KF/SF blocks on a grid (361), explicit matrices | 110 accepted by `scenpar_validate`, 24 fail the TR-order Cholesky (documented error), 26 give NaN with `cholesky = false`, 16 of these although both Cholesky orders succeed; e.g. `ds_kf = -0.6000000000000001, ds_sf = 0.8, sf_kf = -2.220446049250313e-16`: Cholesky residual 4.4e-16, symmetric root NaN. Round values (`-0.6, 0.8, 0`; `0.6, 0.8, 0.96`) are rejected correctly; correlations 1 - 1e-9 ... 1 - 2.2e-16 give a finite root (residual 2e-12 ... 1.2e-9) (C2-01) |
| Python, sqrtm statistics | 200 random 8x8 correlation matrices per smallest eigenvalue, `cholesky = False` | 1e-6 ... 1e-15: 0 NaN; 1e-16: 20 NaN of 178 accepted; 0: 29 of 94 (C2-01) |
| Python, real conf files | smallest eigenvalue of R and both roots for the 91 QuaDRiGa-NG and 23 v19 files | smallest eigenvalue >= 2.5e-4 (WINNER_UMa_C2_LOS), both roots finite for all 114 files: C2-01 needs synthetic or fitted matrices |
| numpy replay of `qd_sqrtm8` | the two NaN matrices above, iteration with "keep the iterate with the smallest e, stop on a non-finite e" | finite, residual 1.0e-11 and 4.5e-8 (fix of C2-01) |
| `p3.cpp` | `DS_omega = Inf` with `DS_delta = 1` / `DS_gamma = 1`; `SF_kappa = Inf`; SF frequency-term message | `lsp` DS NaN (sigma Inf) / Inf; `lsp` SF NaN, `path_loss` `sf_sigma` Inf; the message names `SF_gamma`, which `scenpar_validate` rejects as unknown (C2-02, C2-04) |
| `p4.cpp` | `logdist` with `B = NaN`; `dual_slope` with `E = Inf`; a conf file with `PL_B = NaN` | NaN loss without an error; `E = Inf` gives a finite loss (breakpoint at infinity); `conf_file_read` rejects the file ("Value of 'PL_B' is not a number.") (C2-03) |
| `p5.cpp` | `lsf_draw`: all links on floor 3 (one group) vs. mixed floors vs. one link alone, single and dual mobility; soft LOS at RMa 713 to 800 km (p 2e-310, 8e-311, 0) | identical to the bit; `los_soft` 5.6e-4, 5.5e-4, 0 (subnormal p handled, Halley step skipped where it would overflow) |
| docs | `extract_md.py` for cpp, mex, python; anchors vs links | the new section and blocks render; no broken link in the new section (the MEX/Python link misses found are pre-existing, outside the section) |
| grep | `md/`, spec sections, planning/finding IDs, local paths, NG line numbers in the 90 files | none (only parameter names such as `D1` and published QuaDRiGa file names in older test comments) |

## 4. Round-1 fixes verified

| ID | Check | Status |
|---|---|---|
| C1-01 | `qd_python_flag` for the six flags (and the conf flags); `None` = default, `2.5` TypeError; pytest passes | verified |
| C1-02 | `qd_gbsm_lsp_stats` with an LSP mask; `path_loss` evaluates only the SF std per link, no cube | verified (C2-02 extends its checks) |
| C1-03 to C1-05 | raw-pointer loops, `n_link > 0` guard; ASan/UBSan clean | verified |
| C1-06 | aliasing sentences in all six blocks; `lsf_draw` copies the positions before `rnd` is written (R, T in `qd_gbsm_positions`) | verified |
| C1-07 | `draw_hE` lambda with `p1 >= 1` -> 1; `u = 1`, `p1 < 1` gives `j = n - 1` | verified |
| C1-08 | constructor check `!(arg > 0)` for active terms before any output | verified for `<= 0` and NaN; `+Inf` and infinite coefficients still pass: C2-02 |
| C1-09 | clutter lambda `d > 0 ? exp(-d k) : 1`, SMa terms only for `h_x > hU` | verified (also with `r_veg = 0` and a subnormal `d_clutter`) |
| C1-10 | dual-mobility floor rule in the C++, header, MEX and Python blocks | verified |
| C1-11 | comment in RMa_NLOS, RMa_LOS_O2I, RMa_NLOS_O2I | verified |
| C1-12 | `qd_gbsm_check_links` (3 files), `qd_gbsm_check_fc` (4 files), one preset lookup per call | verified |
| C1-13 | `read_dual` without return, `missing` lambda, `draw_hE` lambda, `qd_pl_triple`, `mul8` lambda, `qd_interp_linear` at all 6 call sites (sos_acfi, sos_remove_bias, sos_val, qd_gbsm_phases, path_loss, los_probability); no `qd_sos_acf_interp` left outside `md/` | verified |
| C1-14 | `src/qd_gbsm_phases.cpp` with the file-local machinery, header 183 lines, `test_gbsm_phases.cpp` | verified |
| C1-15 | `sos_acfi` / `interp_1D` in the tests | verified (user decision) |
| C1-16 | no try/catch helper left; `CHECK_THROWS_MATCHES` (20 to 30 per file) | verified |
| C1-17 | dual-mobility oracle with floors {3, 259, 0} on both sides | verified |
| C1-18 | `lsp_generate` writes `lsp`, `lsp_mu`, `lsp_sigma`, EoD offset per link directly | verified |
| C1-19 | "largest dimension L of the room" | verified |
| C1-20 | clutter arguments `py::handle`, floor IDs below 2^53 in MEX and Python | verified |

## 5. Findings rationale

No blocker, no major. Three minors, five nits (two of them need a decision).

### Minors

- **C2-01 symmetric root NaN.** The coupled Newton-Schulz iteration needs `Z·Y` to have positive eigenvalues; for a
  smallest eigenvalue of R at the rounding level the scaled eigenvalue is ~1e-17 and may be slightly negative in the
  library's rounding, so `m <- m(3 - m)²/4` drifts away from 1 after about 45 iterations and overflows (numpy shows
  the stall at `e` about 1e-8 with `Z` about 2e8 for the positive case). The Cholesky test of `scenpar_validate`
  accepts such matrices (a pivot of 1e-17 is > 0), so `cholesky = false` returns NaN silently while
  `cholesky = true` either works or throws the documented message. The real conf files are far away from this
  (smallest eigenvalue >= 2.5e-4), so the impact is limited to synthetic or fitted matrices, but it is a silent NaN
  from a validated input and contradicts the documentation and spec 3.4.3. The fix is small (best iterate, stop on a
  non-finite error, throw if the result is still not finite) and keeps the results of every regular matrix
  unchanged (the stopping rule `e <= 1e-10` is reached long before).
- **C2-02 infinite LSP coefficients.** Same class as C1-08: values that `scenpar_validate` accepts give NaN LSPs.
  The fix belongs into the constructor of `qd_gbsm_lsp_stats` (one place for both callers, before any output).
- **C2-03 non-finite path-loss parameters.** `path_loss` validates presence, the one-value rule, `E > 0` and
  `A1 != A2`, but not finiteness; `G2n <= 0` is a NaN even with the term switched off (`G1n = 0`), because
  `0 · (lg 0)^2` is `0 · Inf`. One check in `opt()` closes it for all ten models.

### Nits

- **C2-04** message names `SF_gamma`, which is no key.
- **C2-05 [escalate]** single-use helpers `qd_sqrtm8`, `qd_los_probit` as anonymous-namespace functions: allowed by
  the spec, not by the user's literal rule. My recommendation is to keep them (they are long numerical kernels and
  the probit is recursive) and to record the exception; the user decides.
- **C2-06** `path_loss` uses a 25-line if/else chain for the model name where the two sibling files use a table.
- **C2-07** the section overview attributes the O2I variables to `IO_state_lambda`.
- **C2-08 [escalate]** file structure, see section 6.

### Checked and found correct (beyond round 1)

- `qd_gbsm_eval` indexing (column-major `colptr`, group scatter `L.at(j)`, Tx side with `tab[m(g)]` and slice 1),
  the one-group fast path with a non-zero floor (identical to the grouped path, probe p5), the floor offsets with
  `(k·n) mod N`, bias removal over `[R; T]`, the dual-mobility normalisation with the ACF of the generator's own table.
- `qd_gbsm_chol8` index mapping and the TR-order permutation `Q(pi[a], pi[b]) = L(a, b)`; `x = Q·s` reads
  `Q[j·8 + i]`.
- `qd_gbsm_lsp_stats`: frequency terms per LSP and frequency, V/S term, ESD terms with the clamped `h` and raw `hUT`,
  the floor, the outer limit at 0; the `lsps` mask is respected by `path_loss` (only SF queried).
- `path_loss`: `dual_slope` is exactly the TR UMa/UMi PL2 (the `(A1 - A2)·lg dBP3` identity), random hE set size
  `n = floor((hUT - 13.5)/3 + 1e-9) + 1` matches {12, ..., hUT - 1.5} at hUT 13.5, 16.5, 22.5, 23; legacy O2I
  distance; `nlos` maximum of up to three terms; satellite table and `sa` rule.
- `los_probability`: all rows of Table 7.4.2-1 (again), InF defaults of Table 7.8-7, SMa clutter terms, 38.811
  tables and `QuaDRiGa_Industrial` equal `set_scenario.m` (MATLAB uses `atand(h/d)`, the library `atan2(h, d)`,
  which differs only at `h = d = 0`), soft LOS with `sqrt(20 fc / c)`.
- `o2i_loss`: model choice (a zero weight is never chosen, the fallback is the last non-zero weight), indoor
  distance rules, Tables 7.4.3-1 to -3, car losses.
- Conf files: new keys and values of all 23 files against Table 7.6.3.1-2, clause 7.4.3.1, clause 7.6.9 with
  Table 7.8-7 and Table 7.2-2, Table 7.5-6 Part-3 (V/S 4, 50/11 = 4.55, 47/11 = 4.27 per hall), PL constants
  (RMa 20.48/31.74/0.001398/31.75, SMa 21.574/30.13/36.15, `E` 13.33 and 20.94), `PL_hE_rand` (UMa only),
  `PL_LOSmax = 0` (SMa NLOS), `PL_Ax/Bx/Cx` (InF-DL); the optional NLOS formula comments are usable as written.
- Wrappers: argument counts, defaults, `[]`/`None`, outputs beyond `nlhs` passed as `nullptr`, MEX `model_index`
  1-based uint64, `los_soft` `[]`/`None` without `rnd_g` or `fc`, floor IDs below 2^53, flags with `None`; the
  0-based messages of `lsp_generate` / `o2i_loss` in MATLAB are documented in the MEX blocks.
- MSVC: no GCC extensions, no VLAs, no OpenMP in the new code, no `M_PI`, `NAN` defaults with precedent; no
  BLAS/LAPACK calls (8x8 products by hand, `arma::any`/`vectorise` only); no floating-point `std::to_chars`.

### Spec points

- Spec 3.4.3: "the coupled iteration is stable, so the remaining iterations do not degrade S (... about 1e-8 for a
  smallest eigenvalue near 1e-16)" holds only when the eigenvalue stays positive in floating point (C2-01).
- Spec 3.0.4 / C1-08 rationale covers `P_omega + f' <= 0` only; infinite values need the same treatment (C2-02).

## 6. File structure proposal

Current layout (lines): `include/quadriga_gbsm.hpp` 182; `src/quadriga_gbsm_helpers.hpp` 183 (`qd_gbsm_check_links`
13, `qd_gbsm_check_fc` 10, `qd_gbsm_lsp_stats` 92, `qd_gbsm_chol8` 27, documentation block 20);
`src/qd_gbsm_phases.cpp` 585 (`lsf_init_phases`, `lsf_draw`); `src/path_loss.cpp` 596; `src/lsp_generate.cpp` 368;
`src/los_probability.cpp` 393; `src/o2i_loss.cpp` 216; `src/qd_scenpar.cpp` 1583 (four conf functions, user of
`qd_gbsm_chol8`).

| Candidate | Moves | Private header afterwards | SSF effect | Judgement |
|---|---|---|---|---|
| A: `path_loss` + `lsp_generate` -> `src/qd_gbsm_lsp.cpp` (~960 lines) | both blocks and definitions; `qd_gbsm_lsp_stats` into the anonymous namespace of the new file; `qd_pl_*`, zenith tables, `qd_sqrtm8` stay file-local | `qd_gbsm_check_links` (phases, lsp file), `qd_gbsm_check_fc` (lsp file, los_probability, o2i_loss), `qd_gbsm_chol8` (lsp file, qd_scenpar); ~95 lines | none: the phase machinery stays in `qd_gbsm_phases.cpp`; the SSF uses the outputs of `lsp_generate`, not the stats class | possible but not recommended: saves one header entry and its "users" line, but one file then holds the 10 path-loss models with their 93-point table and the LSP generation, two unrelated 110-line blocks; both functions read well alone |
| B: `los_probability` + `o2i_loss` -> `src/qd_gbsm_layout.cpp` (~600 lines) | both blocks; probit, model tables | unchanged | none | not worth it: no helper becomes file-local, the two functions share nothing but `qd_gbsm_check_fc` |
| C: everything of the builder level in one file | phases + path loss + LSPs (~1550 lines) | `check_fc`, `chol8` | the SSF functions would join a 1550-line file | no |

Recommendation: keep the current layout. The round-1 merge removed the real split (one mechanism over two sources
and a header); the remaining files each hold one public function with a long documentation block and only their own
file-local helpers, and the private header holds exactly the four cross-file helpers with their users. C2-08 records
option A for the user's decision.

## 7. SSF readiness

The machinery (`qd_gbsm_gen`, `qd_gbsm_n_coeff`, `qd_gbsm_freq`, `qd_gbsm_positions`, `qd_gbsm_floor_offset`,
`qd_gbsm_eval`, `qd_gbsm_remove_bias`) is generic over the generator table, has no LSF constants outside
`qd_gbsm_lsf_generators`, groups the `sos_val` / `sos_remove_bias` calls per frequency table and side, and handles
floors, 2D/3D and dual mobility per generator; an SSF set with hundreds of generators at `SC_lambda` costs one call
per table and side. Adding SSF phase-set functions to `qd_gbsm_phases.cpp` needs no change of these signatures. The
private header's helpers (`qd_gbsm_check_links`, `qd_gbsm_check_fc`, `qd_gbsm_lsp_stats`, `qd_gbsm_chol8`) are
reusable unchanged; the SSF quantities of spec section 8 are outputs of `lsp_generate` (`lsp_mu` slice 5 after the ESD
floor for `c_ZSD`, `eod_offset_deg`, `abs_toa`, permittivity) or `los_probability` (`los_soft`). Open as before: a
separate Tx floor in dual mobility (C1-10, documented).

## 8. Untested areas

- Symmetric root for a smallest eigenvalue below 1e-15 (C2-01): the tests stop at 1e-6.
- Non-finite LSP coefficients and path-loss parameters, `G2n <= 0` (C2-02, C2-03).
- `path_loss` random hE above hUT = 23 m (the `C' = 1` branch, outside the TR range).
- `lsf_init_phases` bias removal in dual mobility with a one-row `tx_pos` (the Tx position enters `n_link` times);
  consistent with the documented "2·n_link positions", but not pinned by a test.
- MEX zero links (accepted restriction); the release-note text in `md/reports/r4b_report.md` predates the round-1
  behaviour changes (flags with `None`, floor IDs below 2^53, the new frequency-term and hE rules) and should be
  extended when it is used.

## 9. Overall judgement

The port is correct and the round-1 fixes are complete and clean: all 20 resolutions verified in code, tests and
documentation, the sanitizer run is free of reports, the wrappers follow the flag and `None` rules, and no internal
reference is left in a produced file. I found no wrong formula, constant, unit, index, sign or conf value. The
remaining issues are robustness at the edge of the accepted input domain: a NaN of the legacy symmetric root for
correlation matrices that are singular to rounding (C2-01) and NaN results from infinite or non-finite parameters
that the validation lets through (C2-02, C2-03); each has a local fix of a few lines. The nits are wording, one
naming idiom and two decisions on scoping and layout. Recommended before the merge: C2-01 to C2-04 and C2-07; C2-05,
C2-06 and C2-08 as the user decides.
