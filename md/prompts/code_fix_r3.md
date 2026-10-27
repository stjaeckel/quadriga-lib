# Task: fix the findings of the final review round (round 3)

Working directory `/home/stephan/sjc/quadriga-lib-fixes`, branch `fixes` (clean tree at HEAD). Resolve:
- C3-01 to C3-04 in `md/gbsm_lsf_code_findings.md` (report `md/reports/code_r3_review.md`),
- T3-01 to T3-06 from `md/reports/tr_r3_review.md`,
- N3-01 to N3-20 from `md/reports/ng_r3_review.md` (most are integration notes: they go into the spec's
  integration hand-over, section 10; library changes only where stated below or where the finding proposes one).

Append rows for the T3 and N3 findings to the findings table (same format, round 3) and fill the resolution cell
of every C3, T3 and N3 row (`fixed: ...`, `hand-over: spec 10.x`, or `rejected: <reason>`).

## User decisions (binding)

- **C3-01**: fix as proposed (random hE keeps the UT-height clamp `hm = max(hUT, hE + 0.1, 0.1)`), docs in all
  three blocks, Catch2 check for hUT 0.5 and 1.0 m.
- **C3-02: documentation only.** Key descriptions and v19 file comments state that `LOS_state_lambda` is read from
  the NLOS file and `IO_state_lambda` from an O2I file of the group, and that 0 in other files means "not stated
  here".
- **C3-03, C3-04**: fix (C3-04: lambda, or keep under the >20-line rule if it is a self-contained algorithm; say
  which).
- **T3-01: add** the absolute-TOA keys to the four LOS_O2I files (UMa, UMi, RMa, SMa), copied from the matching
  NLOS_O2I file, with a comment: an O2I link has no LOS ray, its response is the NLOS form of TR 38.901 eq. 7.6-43.
  Update tests that list the files with absolute TOA.
- **T3-02 to T3-06: documentation** in code/conf comments as proposed (ground permittivity is QuaDRiGa's fit, not
  TR Table 7.6.8-1; spatially consistent random hE; SMa O2I indoor distance; RMa/SMa LOS `SF_sigma` and passing
  `sf_sigma`; the TR-literal settings `Exp256` and `remove_bias = false`).
- **3D mode and floors (new rule):** with `horizontal = false` (3D correlation) the height separates floors, so
  `floor_id` together with 3D mode is an error (`std::invalid_argument`, clear message) in `lsf_draw` (and wherever
  floor IDs are accepted); bias removal in 3D mode covers all given positions as one 3D field. Document in C++, MEX
  and Python blocks: per-link statistics are identical in 2D and 3D mode; 3D correlates the LSPs of UTs on different
  floors by the ACF of their height difference, while TR 38.901 (Table 7.5-6 note 3) assumes no correlation between
  floors. Tests in Catch2, MEX, Python for the new error. Update the spec.
- **N3-05 (hand-over):** QuaDRiGa-NG has no multi-floor model today. Integration: builders pass no floor IDs (all
  UTs on one floor), bias removal over all positions; per-floor handling (floor-0 positions for bias removal in 2D
  mode, all positions in 3D mode) is noted as the rule for a future multi-floor extension.
- **N3-08 (hand-over):** keep the 36.873 UMa LOS formula in NG `set_scenario.m` (the only LOS formula left there);
  the 3GPP_3D tutorials stay unchanged.
- **N3-01 (hand-over):** list the three `quadriga_lib.interp(x, 0, data, xi)` call sites in NG (`get_pl.m`,
  `@qd_track/interpolate.m`) that must pass `[]` instead of `0`; add the other missing call sites and NG-side notes
  of the review to section 10.

## Rules (binding)

- Produced source, header, test, wrapper and conf files never reference temporary `md/` documents or tracking IDs;
  they may cite published material; `md/SPEC_*.md` may be referenced.
- Lean code, reuse helpers, helper scoping rules of `md/SPEC_CPP_API.md`; MSVC-compatible; no BLAS/LAPACK; no
  `std::to_chars` for floating point.
- Shared machine: `cat /proc/loadavg`, at most 12 threads, one build or suite at a time. Do not commit.

## Verification

Full Catch2 suite, MEX suite in Octave and R2024a (R2026a if time allows), Python suite, objdump gate, no warnings,
ASan/UBSan run of the gbsm cases in `build_asan` (zero reports). Rerun the C3-01 probe.

## Report

`md/reports/code_r3_fix.md`: per finding what changed, rejected ones with reason, results, and a proposed commit
split with messages (no internal IDs or `md/` references in messages; `md/` files in a last planning-notes commit).
Final answer: at most 12 lines.
