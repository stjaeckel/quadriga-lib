# Final review round (round 3): fixes

Branch `fixes`, base HEAD b10f332, nothing committed. All 30 findings of round 3 have their resolution cell in
`md/gbsm_lsf_code_findings.md`: C3-01 to C3-04 (code review), T3-01 to T3-06 (TR compliance review, new rows) and
N3-01 to N3-20 (QuaDRiGa-NG integration review, new rows). **10 fixed in the library (C3-01 to C3-04, T3-01 to
T3-06), 20 handed over to spec section 10 (N3), none rejected**; N3-05 also brings the new library rule "no floor IDs
in 3D mode". Resources: at most 12 threads, one build or suite at a time; load average 1.5 to 24 during the work
(other sessions). No timing in this report is a benchmark.

## 1. Changes per finding

| ID | Change |
|---|---|
| C3-01 | `src/path_loss.cpp` `qd_pl_dual_slope`: in random-hE mode the UT height is clamped with `hm = max(hUT, hE + 0.1, 0.1)` (fixed mode unchanged: `max(hUT, 1.1·hE, 0.1)`). For hE = 1 this is the 1.1 m of the fixed mode, so `rnd_he` no longer changes the loss of UTs below 1.1 m; the drawn heights 12, 15, ... are unaffected (`hUT - hE >= 1.5`), so the TR results and the floor-7/8 checks are unchanged. Docs: the random-hE bullet of the C++, MEX and Python blocks. Catch2 (`test_path_loss.cpp`, random-hE case): hUT 0.5 and 1.0 m, d2D 20 / 50 / 100 / 300 m, `rnd_he` 0.3 / 0.99, UMa LOS file and the `Bn = -1000` NLOS copy give the same loss with and without `rnd_he` (1e-12 dB) and equal the reference with hE = 1 (1e-9 dB, three links before and one beyond the 112 m breakpoint); the test's reference `ref_dual_slope` follows the new clamp. Probe `q1` rerun: every difference 0.000 dB (was -22 to +12 dB). |
| C3-02 | Documentation only (user decision). Key descriptions in `src/qd_scenpar.cpp`: "..., stated in the NLOS file of a scenario; 0 = not stated here" and "..., stated in the O2I files of a scenario; 0 = not stated here" (these are also the comments `conf_file_write` writes). Section overview (C++, MEX, Python) and the `los_probability` blocks (C++, MEX, Python) say that the two distances apply to all links of a scenario, where they are stated and that 0 elsewhere means "not stated here". All 23 v19 files: a three-line comment after `SC_lambda`. Test expectation of `test_conf_file_write.cpp` and two test comments updated. |
| C3-03 | UMa files (4): "hE = 1 m, or random for hUT > 13 m (... drawn from the hE column of lsf_draw, index 10 counted from 0)". RMa (4) and SMa LOS / LOS_O2I: the two sentences merged into one ("evaluated for ... and rounded to 4 significant digits ..."). SMa NLOS / NLOS_O2I: the review's premise was checked and does not hold: with h = W = 10 m the constants are exact (Bn = 161.04 - 7.1 + 7.5 - 130.26 + 4.97 = 36.15, Dn = -24.37 + 9.3 = -15.07, D2n = 370), so the files now say "with these values they are exact (no rounding)". |
| C3-04 | Lambda (15-line body, one caller, below the 20-line rule): `qd_gbsm_floor_offset` became the lambda `floor_offset` inside `qd_gbsm_eval`, with its comment. Every phase set, also the later SSF sets, reaches it through `qd_gbsm_eval`. Its block variable is named `blk` (avoids `-Wshadow` with the Tx-side matrix `b`). Values bit-identical (the typed-offset Catch2 cases pass unchanged). |
| T3-01 | The four LOS_O2I files (UMa, UMi, RMa, SMa) get the section "Absolute time of arrival offset" (Clause 7.6.9, Table 7.6.9-1) before "Optional modelling components", with the three `absTOA_*` lines of the matching NLOS_O2I file and the comment that an O2I link has no LOS ray (K-factor N/A, `KF_mu = -100`), so its response is the NLOS form of TR eq. (7.6-43). `test_conf_files_v19.cpp`: the absolute TOA is active in every non-LOS file (was: every NLOS file), and typed values of the four files are checked (UMa_LOS_O2I `absTOA_mu` was -100 in the oracle table, now -7.4). No MEX or Python test lists the files with absolute TOA. |
| T3-02 | `lsp_generate` blocks (C++, MEX, Python): the three ground types are QuaDRiGa's fits (as in `gen_lsf_parameters`), not Table 7.6.8-1; very dry agrees, medium and wet differ (3.5 GHz numbers); the caller computes the table values with eq. (7.6-40) to (7.6-42). |
| T3-03 | `path_loss` blocks (C++, MEX, Python): the TR draws hE per link and gives no decorrelation distance; the library makes it spatially consistent in the sense of clause 7.6.3 with `SF_lambda`, so nearby UTs of a site share hE. |
| T3-04 | Both SMa O2I files (comment) and the `o2i_loss` blocks (C++, MEX, Python): the TR SMa layout has 90 % residential and 10 % commercial buildings (Table 7.2-5); split the UTs by building type, one `o2i_loss` call per type (10 m / 25 m). |
| T3-05 | RMa_LOS, SMa_LOS: comment after `SF_sigma`: 5 dB is a legacy midpoint of the TR values 4 / 6 dB (`PL_sig1`, `PL_sig2`); TR conformance needs the `sf_sigma` output of `path_loss` passed to `lsp_generate`. |
| T3-06 | `lsf_init_phases` blocks (C++, MEX, Python) and the header comment: TR clause 7.4.4 literally is `preset = "Exp256"`, `remove_bias = false`; the defaults are QuaDRiGa choices. |
| 3D mode and floors (new rule, user decision; with N3-05) | `lsf_draw` throws `Input 'floor_id' cannot be used with 'horizontal = false': in 3D mode the height separates the floors.` when `floor_id` is given with `horizontal = false` (also all zeros; checked last, before any output is written; MEX and Python pass an empty `floor_id` as `nullptr`, so `[]` / `None` stay valid). `lsf_draw` is the only function that accepts floor IDs. Docs (C++, MEX, Python of `lsf_draw` and `lsf_init_phases`, header comments): per-link statistics are identical in 2D and 3D mode; 3D correlates the LSPs of UTs on different floors by the ACF of their height difference, while TR Table 7.5-6 note 3 assumes no correlation between floors; in 3D mode the bias removal covers all given positions as one 3D field (2D: one floor). Validation-order sentences extended. Tests: Catch2 (`lsf_draw - input validation`: IDs 0/1/2 single mobility and all zeros dual mobility throw, output untouched, `nullptr` with 3D runs), MEX (`test_lsf_draw.m`: two error cases), Python (`test_lsf_draw.py`: `self.fl` and zeros raise `ValueError`, `None` with `horizontal=False` runs). |
| N3-01 to N3-20 | Spec section 10 (integration hand-over), see section 2. |

Code and test diff (without `md/`): 52 files, about +385 / -115 lines (23 conf files, 14 wrapper files, 7 sources and the
header, 8 test files).

## 2. Spec updates (`md/gbsm_lsf_spec.md`)

- 1.2, 2.1 (header text), 3.1, 3.2 (validation, properties), 3.2.1, 3.0.5 and the places that named
  `qd_gbsm_floor_offset` (5.7, Run 2a, section 8, 9.1): 3D/floor rule, lambda.
- 3.3.4, 3.3.6, 3.3.7: random-hE clamp `hE + 0.1` with its reason; spatial consistency of hE.
- 3.4.4: ground-permittivity statement. 4.1: key descriptions and the "Meaning" column. New 4.4.3: absolute TOA of the
  LOS_O2I files and the comment changes. 5.2, 5.3, 5.8, 6.1, 6.2: the new tests.
- 10.1: the three `interp( x, 0, ... )` call sites (`get_pl.m` 469, `@qd_track/interpolate.m` 241 and 266) that
  must pass `[]`, the 29 failing NG tests and the two other failures (N3-01).
- 10.2: the complete list of sites that store, copy or clear the LSF state or read `lsp_vals`, with the `loadobj`
  migration (N3-06); floors: NG has no multi-floor model, builders pass no floor IDs and remove the bias over all
  positions; the rule for a future extension (2D: floor IDs, bias removal over floor-0 positions; 3D: no floor IDs,
  all positions) and its data path (`floor_id`, `o2i_d2din` track fields, builder property, `max(0, round(...))`;
  `o2i_d2din` needed now for legacy `d2d_in`) (N3-04, N3-05); the `get_pl` O2I rule and the stored hE value (N3-07);
  further notes: positions as columns (N3-03), empty builders (N3-02), `dual_mobility = -1` (N3-09), legacy ACF names
  and `'Disable'` (N3-10), stored `preset` / `horizontal` and draw-time lambdas (N3-11), baseline-mode draws (N3-15),
  `get_lsp_map` Tx (N3-18), stacked UTs in 2D (N3-19), the `power_map` SF bug (N3-17), the performance notes
  (N3-16); the builder-state list; the two "not switchable" items reworded as reachable by the caller (N3-12); InF-DH
  `r = 0.4` and the absolute TOA of the LOS_O2I files in the list of behaviour changes (N3-14, T3-01).
- 10.3: state distances read from the NLOS / O2I file of the group (C3-02); `3GPP_3D_UMi` -> `38.901_UMi`,
  `3GPP_3D_UMa` keeps its 36.873 formula in `set_scenario.m`, the 3GPP_3D tutorials stay unchanged (N3-08); O2I
  variable distances decided in the integration of `gen_o2i_loss` (N3-20); floor index as future rule.
- 10.4: reference-script rule `d2d_in = d2D·d3din/d3D` (N3-13). 10.5: the 36.873 formula is not retired; the four
  unit tests that inspect internals (N3-06).

## 3. Tests and sanitizers

| Run | Result |
|---|---|
| Builds `build_linux` (g++-12, AVX2, `test_bin`), `build_mex`, `build_py` (`--parallel 12`), each installed | 0 warnings, 0 errors |
| g++-12 `-fsyntax-only -Wall -Wextra -Wpedantic -Wshadow -Wconversion` and clang++-21 `-Wall -Wextra -Wmicrosoft -Wlanguage-extension-token`: `path_loss.cpp`, `qd_gbsm_phases.cpp`, `lsp_generate.cpp`, `o2i_loss.cpp`, `los_probability.cpp`, `qd_scenpar.cpp`; g++ `-Wall -Wextra -Wpedantic` on the four changed Catch2 files | no diagnostics (after renaming the lambda's `b` to `blk`; the first check found a `-Wshadow` warning) |
| Catch2 GBSM filter (`lsf_*`, `lsp_generate*`, `path_loss*`, `los_probability*`, `o2i_loss*`, `conf_file_*`, `scenpar_validate*`, `conf files v19*`, `sos_*`) | All tests passed (54668 assertions in 232 test cases; 54637 before) |
| `make test` (`CMAKE_BUILD_PARALLEL_LEVEL=12`, OMP 12): full Catch2 suite, AVX2 build | All tests passed (1355698 assertions in 1374 test cases); the only warning is the existing qdant locale message |
| `make test`: MEX suite Octave | 73 of 74 passed; the one failure is the known `test_ray_mesh_interact.m:113` (0.98485 vs 0.69639), older than this work |
| `make test`: MEX suite MATLAB R2024a (`matlab24`) | 73 of 74, the same known failure |
| MEX suite MATLAB R2026a (26.1.0.3203278) | 73 of 74 (same known failure) in every run. **Open observation:** in 3 of the first 5 full runs (load 18 to 24) MATLAB aborted *after* the suite had finished, at exit, with a glibc heap message ("free(): chunks in smallbin corrupted", "corrupted double-linked list"; once it then hung until the timeout). Bisection: GBSM subset with the new tests 1 of 4 at high load, 0 of 8 at load 3; the HEAD (b10f332) MEX files with the HEAD tests 0 of 6; the new MEX files with the HEAD tests 0 of 6; `test_lsf_draw` alone 0 of 8; the other 64 tests 0 of 4; three more full runs at load 3: 0 of 3. Not reproducible at low load and not attributable to a change of this round (ASan/UBSan of the C++ code is clean, Octave and R2024a never show it); it looks like the load-dependent R2026a exit problem that the `-z nodelete` link option in `CMakeLists.txt` addresses (HDF5 thread-key destructors at exit). Recommendation: watch it in the next R2026a runs; R2024a stays the release target. |
| pytest `tests/python_tests` (OMP 12) | 876 passed, 3 skipped, 52 subtests passed |
| objdump gate `+quadriga_lib/*.mexa64` | highest GLIBCXX_3.4.26 (limit 3.4.28); no `to_chars` symbol |
| ASan/UBSan: `build_asan` `test_bin` rebuilt (g++-12, `-fsanitize=address,undefined`, AVX2; 6 min 28 s), GBSM filter as above, `detect_leaks=1`, `halt_on_error=0`, OMP 8 | All tests passed (54668 assertions in 232 test cases); **0 runtime errors, 0 AddressSanitizer / LeakSanitizer reports**. No allocation probe ran under ASan. |

Load average: `make test` 10.1 -> 8.0, ASan run 4.0 -> 3.4, the first Python and MEX runs 18 to 24 (other sessions).

## 4. Probe

`scratchpad/r3fix/q1.cpp` (the review's `q1.cpp`, built with its `build.sh` against the new `build_linux/libquadriga.a`):
UMa LOS v19 file, hBS 25 m, 3.5 GHz, `u = 0.3`, hUT 0.5 / 0.9 / 1.0 / 1.05 / 1.5 / 13 / 13.2 m, d2D 20 / 50 / 100 /
300 m: the loss with `rnd_he` equals the loss without it at every point (difference 0.000 dB; before: hUT 0.5 m
49.41 vs 71.88 dB at 20 m, hUT 1.0 m +2.1 to +12.2 dB, hUT 1.05 m +4.1 dB at 100 m).

## 5. Proposed commit split

The `md/` files go into the last commit. The untracked `md/prompts/code_fix_r3.md` and the three round-3 review
reports were not created by this work; they belong to the planning-notes commit.

1. **Large-scale model: random environment height below 1.1 m, floor IDs only in 2D mode**
   `src/path_loss.cpp`, `src/qd_gbsm_phases.cpp`, `include/quadriga_gbsm.hpp`, `api_mex/path_loss.cpp`,
   `api_mex/lsf_draw.cpp`, `api_mex/lsf_init_phases.cpp`, `api_python/qpy_path_loss.cpp`,
   `api_python/qpy_lsf_draw.cpp`, `api_python/qpy_lsf_init_phases.cpp`, `tests/catch2_tests/test_path_loss.cpp`,
   `tests/catch2_tests/test_gbsm_phases.cpp`, `tests/mex_tests/test_lsf_draw.m`, `tests/python_tests/test_lsf_draw.py`.
   Message: "With the random UMa environment height, path_loss clamps the UT height to hE + 0.1 m instead of
   dropping the clamp. For hE = 1 m this is the 1.1 m of the fixed height, so UTs below 1.1 m get the same loss with
   and without rnd_he (before, every distance was beyond the breakpoint there, up to 22 dB too low); the drawn heights
   of 12 m and more are not affected. lsf_draw rejects floor IDs with horizontal = false: in 3D mode the height
   separates the floors. The documentation states that 2D and 3D mode have the same per-link statistics, that 3D
   correlates UTs on different floors by the ACF of their height difference while TR 38.901 assumes no correlation
   between floors, that the 3D bias removal covers all positions, and the TR-literal settings Exp256 and
   remove_bias = false. The floor phase offsets are a lambda of the evaluation function. Tests cover UTs at 0.5 and
   1.0 m and the new error in C++, MATLAB/Octave and Python."
2. **Conf files v19: absolute TOA of the O2I-LOS files, state distances, comments**
   `data/config/3GPP_38.901_v19_*.conf` (23), `src/qd_scenpar.cpp`, `api_mex/conf_file_read.cpp`,
   `api_python/qpy_conf_file_read.cpp`, `src/los_probability.cpp`, `api_mex/los_probability.cpp`,
   `api_python/qpy_los_probability.cpp`, `src/lsp_generate.cpp`, `api_mex/lsp_generate.cpp`,
   `api_python/qpy_lsp_generate.cpp`, `src/o2i_loss.cpp`, `api_mex/o2i_loss.cpp`, `api_python/qpy_o2i_loss.cpp`,
   `tests/catch2_tests/test_conf_files_v19.cpp`, `test_conf_file_write.cpp`, `test_conf_file_read.cpp`,
   `test_scenpar_validate.cpp`.
   Message: "The UMa, UMi, RMa and SMa LOS_O2I files carry the absolute time of arrival of their NLOS_O2I files: an
   O2I link has no LOS ray, so its impulse response is the NLOS form of TR 38.901 eq. (7.6-43). The files and the key
   descriptions state that LOS_state_lambda is given in the NLOS file and IO_state_lambda in the O2I files of a
   scenario, 0 elsewhere meaning not stated. Comments: random hE only above 13 m, one sentence on the evaluation and
   rounding of the RMa and SMa constants (exact for SMa NLOS), the SMa building mix for o2i_loss, the legacy
   SF_sigma of RMa and SMa LOS. The documentation states that the ground permittivity uses QuaDRiGa's fits, not TR
   Table 7.6.8-1, and how the TR building mix of SMa is modelled."
3. **Planning notes: final review round and its fixes**
   `md/gbsm_lsf_spec.md`, `md/gbsm_lsf_code_findings.md`, `md/prompts/code_fix_r3.md`,
   `md/reports/code_r3_review.md`, `md/reports/tr_r3_review.md`, `md/reports/ng_r3_review.md`,
   `md/reports/code_r3_fix.md`.

## 6. Notes

- `html_docu/` was not regenerated (done by `make documentation` after the merge); `tools/extract_md.py` for cpp,
  mex and python runs without unresolved `[[...]]` links.
- Logs in `/tmp/claude-1005/-home-stephan-sjc-quadriga-lib-fixes/240c04ce-057b-420a-a310-dd24cc94ffbc/scratchpad/r3fix/`.
