# Run 3a report — `path_loss`, path-loss keys and constants

## Summary

- `path_loss` (`src/path_loss.cpp`) implemented as in 3.3: validation (3.0.2, 3.3.1), legacy O2I distance rule,
  the 10 models of 3.3.4 with their defaults, the random UMa environment height of 3.3.6, the default SF std from
  `qd_gbsm_lsp_mu_sigma`; final documentation block (3.3.7). Declaration added to `include/quadriga_gbsm.hpp`.
- Tests of 5.3 written first (9 cases), run against the stub, then green against the implementation.
- 21 v19 files edited as in 4.4.2 (rounded constants, `PL_hE_rand`, `PL_LOSmax`, `PL_Ax/Bx/Cx`, comments);
  writer comment rows of 4.3 in `qd_conf_pl_comments`; `test_conf_files_v19.cpp`, `test_conf_file_write.cpp` updated (5.8).
- Full Catch2 suite, `make test` (Catch2 + MEX in Octave and R2024a), objdump gate, Python suite run; no new
  failure, no compiler warning.
- One spec problem: the point list of the TR test (5.3) contains points outside the d2D ranges for which the
  stated rounding tolerances hold; restricted to those ranges (decision 1).

## Files

Created (2):
- `src/path_loss.cpp`
- `tests/catch2_tests/test_path_loss.cpp` (9 test cases)

Changed (25):
- `include/quadriga_gbsm.hpp` (declaration of `path_loss`, text of 2.1, between `lsf_draw` and `lsp_generate`)
- `src/qd_scenpar.cpp` (5 rows in `qd_conf_pl_comments` before the `usePLa` row, text of 4.3)
- `tests/catch2_tests/test_conf_files_v19.cpp` (case "path-loss constants of RMa and SMa follow Table 7.4.1-1"
  rewritten, header comment)
- `tests/catch2_tests/test_conf_file_write.cpp` (case "path-loss comments and formulas depend on the model": 5 keys
  and checks; comment with QuaDRiGa line numbers reworded)
- 21 files `data/config/3GPP_38.901_v19_*.conf`: Indoor_NLOS, InF_NLOS_DH/DL/SH/SL, RMa_LOS/_LOS_O2I/_NLOS/
  _NLOS_O2I, SMa_LOS/_LOS_O2I/_NLOS/_NLOS_O2I, UMa_LOS/_LOS_O2I/_NLOS/_NLOS_O2I, UMi_LOS/_LOS_O2I/_NLOS/_NLOS_O2I
  (InF_LOS and Indoor_LOS unchanged: `logdist`, no derived constant)

Plus this report. Not from this run: `md/prompts/impl_run3a.md` (untracked, the coordinator's prompt).
`src/quadriga_gbsm_helpers.hpp` is unchanged (the helpers of 3.0.5 are used with their fixed signatures;
`qd_gbsm_lsp_mu_sigma` already listed `path_loss.cpp` as user).

## Decisions and deviations

Review points of SPEC_CPP_API (not stops): A — no open contract question apart from decision 1. B — tests and
stub run below. C — this report.

1. **TR test points restricted to the TR d2D ranges (spec problem, 5.3 vs. 4.4.2).** 5.3 lists `0.5 dBP` without a
   range limit and, for SMa, `2 dBP` without one; its tolerances are "the bound of that rounding over the TR
   ranges (4.4.2)", which assumes d2D up to 5 km (UMa, UMi) or 10 km (RMa, SMa). Outside, the bound does not hold:
   with all listed points the run gives RMa LOS 8.26e-3 dB > 8.2e-3 (hBS 35, hUT 5, 6 GHz, `0.5 dBP` = 11.0 km)
   and SMa LOS 0.078 dB > 0.0416 (`0.5 dBP` / `2 dBP` up to 396 km at hBS 35, hUT 13.5, 20 GHz: the shift of the
   breakpoint by E = 20.94 times D = 0.002 dB/m). Points with d2D above the range are skipped: UMa, UMi 5 km;
   RMa LOS 10 km; RMa NLOS 5 km (as 5.3 states for `2 dBP`); SMa 10 km (the range of the 4.4.2 bound; the TR
   itself ends SMa at 5 km). Stated in a comment of the test. Proposal for the spec: write the range limit into
   the point list of 5.3. With the restriction the maxima are within the bounds (table under Test results).
2. Validation order: 3.0.2 (rx columns, tx columns, tx rows, rx finite, tx finite, `fc` empty, `fc` values,
   length of `d2d_in`, length of `rnd_he`), then 3.3.1: `d2d_in` values, `rnd_he` values (NaN fails the range
   check), model name, the parameters of the model in the order the model reads them, `scenpar`. All before any
   output is written (tested with pre-filled outputs).
3. Parameters are read in the order of the model (required and optional interleaved); each key is checked for
   "present" (if required) and "exactly one value" when it is reached. The breakpoint factor check (`E`, `E1` > 0)
   runs right after the dual-slope base keys are read, `E2` when read, `A1 != A2` after the 14 `winner_los` keys.
   With two errors in one call, the first in this order is reported (documented in the block).
4. Missing "A1 or A" names `'A1'`, missing "E1 or E" names `'E1'` in `requires the parameter '<key>'`; the `> 0`
   message names the key actually read (`'E'` when `tripple_slope` takes `E` for `E1`). Documented.
5. `nlos` with `LOSmax = 0` reads none of the dual-slope keys (`A1`, `B`, `E`, `hE`, `hE_rand` are "not read",
   so they are neither required nor checked; tested). `Ax/Bx/Cx`: the term is active if any is present, missing
   ones are 0.
6. `sf_sigma`: the default (slice 2 of `qd_gbsm_lsp_mu_sigma` with the raw geometry) is computed only when
   `sf_sigma` is requested; a model value (`SF`, `sig*`) replaces it only where the model sets it. The model values
   are not range-checked (3.3.1 has no such check; a negative `sig` is passed through).
7. Random hE: `d` is the `d2` of 3.3.2 after the O2I rule and the clamp (as written in 3.3.6); `n` is computed as a
   double, `floor((hUT - 13.5)/3 + 1e-9) + 1`. `hUT > 23` gives `C' = 1` as written.
8. Satellite "below the horizon" is the case `hBS <= 0` (absolute Tx height, `sa <= 0`), tested with a Tx at
   -1000 m. The zenith table is interpolated with `qd_sos_acf_interp` (93 points).
9. No OpenMP in `path_loss` (optional per 9.1; the cost is a few `log10` per link and frequency).
10. File-local helpers (anonymous namespace): `qd_pl_dual_slope` (used by `dual_slope` and `nlos`),
    `qd_pl_random_hE`, the zenith table; the per-model formulas are cases of one switch in the link loop.
11. Conf files: new lines `PL_hE_rand`, `PL_LOSmax`, `PL_Ax`, `PL_Bx`, `PL_Cx` carry the writer comments of 4.3,
    aligned with the neighbouring lines. Placement of the 4.4.2 comment lines (the spec names the text, not the
    exact place): "The constants are evaluated ... rounded to 4 significant digits ..." right after the existing
    line "The constants below are evaluated for ..."; the 3D-breakpoint note right after the dual-slope formula
    block (in the RMa NLOS files before the `nlos` block); the optional NLOS formulas right after the `% NLOS:` TR
    line. Indoor_NLOS: the optional formula in two lines ("..., SF std 8.29 dB, used in max( PL_LOS, PL' ). It is
    used with PL_model = nlos, PL_An = 31.9, PL_Bn = 32.4, PL_Cn = 20, PL_sig = 8.29."). The SMa NLOS files keep
    the `nlos` formula block (it still states `PL = max( PL_LOS, PLn )`; the next comment says `PL_LOSmax = 0`
    switches it off).
12. TR test, SF: exact equality with the TR value (LOS files of RMa/SMa by the side of `dBP`; NLOS files
    `SF_sigma`; O2I files 7 / 8 dB). InF-DL: the InF-SL term wins at 12 of 24 points (asserted > 0).
13. O2I distance test: the oracle is the `nlos` formula with the constants of the UMa NLOS file (not the TR
    formula, whose `E = 40/3` differs from the file's 13.33 by more than 1e-12), evaluated at the distances of the
    typed legacy rule; tolerance 1e-12 relative (dB values of order 100, 5.0 "relative for others").
14. Random-hE test: hE identified by the nearest TR candidate; the candidates differ by >= 0.53 dB (asserted
    > 0.01). Counts: 2350 / 2550 / 2550 / 2550 against 2349.9 and 2550.03. The `nlos` copy with `Bn = -1000` and
    the unmodified UMa NLOS file are checked link by link with the identified hE. The `rnd_he` omitted case uses
    the UMa LOS file (fixed `PL_hE = 1`).
15. Legacy-models case: one section per model, 4 links with per-link Tx (heights below 1.5 m, hBS > hUT and
    hBS < hUT) x 3 frequencies; region counters assert that both `winner_los` and `dual_slope` branches, all three
    `tripple_slope` regions, and the PLn, LOS and extra terms of `nlos` occur; `winner_los` continuity at the
    breakpoint (`F1 = F2`, `C1 != C2`): difference 2.8e-8 dB for a distance change of 2e-9 relative.
16. `test_conf_files_v19.cpp`: the 5.8 checks are in the case "path-loss constants of RMa and SMa follow Table
    7.4.1-1" (also for the O2I variants): "rounded" = the value has at most that many decimals and lies within half
    a unit of the last digit plus 1e-12 of the exact constant; RMa NLOS also checks its LOS constants; SMa NLOS
    `LOSmax == 0` and no `A1, A2, B, C, E, hE`; UMa, UMi, InH NLOS, InF NLOS `E == 13.33`; `hE_rand` exactly in the
    4 UMa files, `Ax/Bx/Cx` exactly in InF-DL, `LOSmax` exactly in the SMa NLOS files. The comment "needed by the
    QuaDRiGa model ..." is gone; the stale header sentence "No library code evaluates them yet." is reworded.
17. 9.1 rewording: `test_conf_file_write.cpp` had "comments typed from write_conf_file.m, lines 568 to 692"
    (QuaDRiGa line numbers); now "typed from the path-loss section of QuaDRiGa's write_conf_file.m".

## Test results

Load average (1-min) before the Catch2 runs 1.0 to 2.6; before `make` 3.1; during `make test` (MEX) about 59 and
during `make python` 26 (other sessions). No timing results are reported, so the load does not affect any result.

Run against the stub (outputs sized and zeroed, no validation), `build_linux/test_bin "path_loss*"`, before the
conf edits:
```
test cases:    9 |   1 passed |    8 failed
assertions: 2017 | 273 passed | 1744 failed
```
- Failing for the right reason (zeros, no exception): the TR case (1154 of 1334; passing are the size REQUIREs),
  O2I distances (29 of 39; passing are the typed checks of the rule's special cases), legacy models (456 of 501),
  clamps (7 of 7), default `sf_sigma` (8 of 11), empty model (`sf` 0 instead of 6), validation (88 of 107; passing
  are the baseline sections). The random-hE case stopped at `REQUIRE(hE_rand == 1)`: the files had no
  `PL_hE_rand` yet (the 4.4.2 edits came after the stub run, as the run order says).
- Passing against the stub, as expected: "optional outputs and pre-sized outputs" (zeros equal, sizes right).

Final, new cases: `build_linux/test_bin "path_loss*"`
```
All tests passed (2079 assertions in 9 test cases)
```
TR case per file (`QD_DIAG_GBSM=1`, maximum deviation from the 3D-breakpoint form; for the RMa/SMa points beyond
the breakpoint also from the TR form):

| Files | points | beyond dBP (RMa/SMa) | max dev. | tol_r |
|---|---|---|---|---|
| InF (5), Indoor (2) | 24 / 12 | 0 | 0 | 1e-9 |
| RMa_LOS, _LOS_O2I | 42 | 13 | 7.77e-3 | 8.2e-3 |
| RMa_NLOS, _NLOS_O2I | 36 | 9 | 4.96e-3 | 8.2e-3 |
| SMa_LOS, _LOS_O2I | 31 | 3 | 4.27e-3 | 6.6e-3 |
| SMa_NLOS, _NLOS_O2I | 31 | 0 | 5.7e-14 | 1e-9 |
| UMa_LOS, _LOS_O2I / UMi_LOS, _LOS_O2I | 15 / 17 | 0 | 1.96e-3 / 2.06e-3 | 2.1e-3 |
| UMa_NLOS, UMi_NLOS (+O2I) | 15 / 17 | 0 | 2.8e-14 | 2.1e-3 |

Conf and writer tests: `build_linux/test_bin "conf*,scenpar_validate*,path_loss*"` ->
`All tests passed (10894 assertions in 61 test cases)`.

Full Catch2 suite (build tree with AVX2 on, `OMP_NUM_THREADS=12 build_linux/test_bin`):
```
All tests passed (1353325 assertions in 1357 test cases)
```
`make test` (`CMAKE_BUILD_PARALLEL_LEVEL=12 OMP_NUM_THREADS=12`, after `make`; Makefile configuration: AVX2 off,
gcc-12, MATLAB and Octave on):
```
All tests passed (1351402 assertions in 1356 test cases)
```
MEX (run by `make test`): Octave `/usr/bin/octave`, MATLAB R2024a `matlab24`:
- Octave: `FAILED (passed=67, failure=1)`
- MATLAB R2024a: `FAILED (passed=67, failure=1)`
- The one failure is `test_ray_mesh_interact.m` line 113 (0.98485 vs 0.69639), the pre-existing failure of the
  earlier runs (ray-tracing area). All conf MEX tests pass with the rebuilt MEX files (they link the changed
  `qd_scenpar.cpp` and read the changed v19 files). R2026a was not run.

objdump gate: `objdump -T +quadriga_lib/*.mexa64 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -u` -> 3.4.9, 3.4.11,
3.4.18, 3.4.19, 3.4.20, 3.4.21, 3.4.26 — nothing above 3.4.28.

Python: `make python` (`CMAKE_BUILD_PARALLEL_LEVEL=12`), `make python_test`: `816 passed, 3 skipped in 2.53s`
(no Python wrapper in this run; the conf functions read the changed files and link the changed `qd_scenpar.cpp`).

The build tree `build_linux` was restored afterwards to its configuration at the start (tests on, AVX2 on,
MATLAB/Octave/Python off) and rebuilt; `path_loss*, lsp_generate*, lsf_*, conf*, scenpar*` ->
`All tests passed (12728 assertions in 102 test cases)`.

## Warnings

None. `src/path_loss.cpp` and `src/qd_scenpar.cpp` were recompiled with `-O3 -Wall -Wextra -Wpedantic` (checked with
`VERBOSE=1`) without a warning; the `make`, `make test` and `make python` logs have no compiler warning. The only
"warning" in the `make test` log is the run-time Catch2 `WARN` of `test_qdant_write.cpp:386` (no decimal-comma
locale installed), pre-existing.

## Open points

- Spec 5.3 point list vs. tolerance (decision 1): proposal to add "points with d2D beyond the range of the bound
  (UMa, UMi 5 km; RMa, SMa 10 km; RMa NLOS 5 km) are skipped". 4.4.2 quotes 8.1e-3 dB at the RMa test points; with
  the 11 km point removed the maximum is 7.77e-3.
- Random hE below the TR UT heights: without the 1.1·hE clamp, `hUT < 2.5 m` with `hE = 1` gives `h'UT < 1.5`, and
  `hUT <= 1 m` gives `dBP <= 0`, so every link is beyond the breakpoint (finite; `dBP3 >= hb - hm > 0` since
  `hb >= 1.1`). Outside the TR range (hUT >= 1.5 m); not tested, no change proposed.
- Integration (section 10): QuaDRiGa-NG's MATLAB `get_pl` cannot evaluate the changed SMa NLOS files (no `A1`, `B`,
  `E`; `PL_LOSmax` unknown) nor the new keys; NG must use `path_loss` for the v19 files.
- Not tested: a negative `sig*` / `SF` value (passed through, decision 6); non-finite `plpar` values (no check in
  3.3.1).
- The MEX failure of `test_ray_mesh_interact` persists from before Run 1.

## Suggested commit cut

1. Conf files and writer comments:
   - Files: the 21 `data/config/3GPP_38.901_v19_*.conf` files above, `src/qd_scenpar.cpp`,
     `tests/catch2_tests/test_conf_files_v19.cpp`, `tests/catch2_tests/test_conf_file_write.cpp`
   - Message:
     ```
     Conf files v19: path-loss constants with 4 significant digits, new path-loss keys

     The derived path-loss constants of the TR 38.901 v19 files are written with 4 significant digits (SMa A1
     with 5): E = 13.33 for 40/3 (UMa, UMi, InH, InF), E = 20.94 for 20 pi / 3 and the RMa and SMa constants
     A1, B, D, Bn evaluated for the stated building height and street width; the loss changes by less than
     0.01 dB against the exact constants. New path-loss keys: PL_hE_rand (random environment height of UMa, TR
     Table 7.4.1-1 note 1), PL_LOSmax (SMa NLOS without the maximum with the LOS term, replacing the zeroed LOS
     constants), PL_Ax, PL_Bx, PL_Cx (the InF-SL term in the maximum of InF-DL). The NLOS files of UMa, UMi and
     InH document the optional NLOS formulas of the TR. conf_file_write has comments for the new keys.

     Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
     ```
2. Path loss:
   - Files: `include/quadriga_gbsm.hpp`, `src/path_loss.cpp`, `tests/catch2_tests/test_path_loss.cpp`
   - Message:
     ```
     Path loss: path_loss

     path_loss evaluates the parametric path-loss models of the conf files (logdist, logdist_simple, constant,
     winner_los, winner_nlos, winner_pathloss, dual_slope, tripple_slope, nlos, satellite) and the shadow-fading
     standard deviation for a set of links and carrier frequencies. The v19 files reproduce TR 38.901 Table
     7.4.1-1, including the random environment height of UMa from a spatially consistent uniform value, the
     maximum of three terms for InF-DL and SMa NLOS without the LOS term. winner_los takes the breakpoint from
     the heights of each link and uses C2 beyond it. An optional indoor distance per link selects the legacy
     outdoor-distance mode for O2I links. The default shadow-fading standard deviation follows the LSP formula,
     so that passing it to lsp_generate is neutral.

     Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
     ```
   Commit 2 needs commit 1 (the tests read `PL_hE_rand` and `PL_Ax/Bx/Cx` from the files).
