# SSF fix round 1 — report

Applies `md/ssf_code_review_r1.md` (B1, M1–M3, m1–m8) with the maintainer's decisions. Nothing committed.

## Per finding

- **B1** `src/ssf_clusters.cpp` (quadriga delay scale): fallback 0 instead of 1 when the cluster delays have no
  spread (comment at the code). Spec 3.6 step 6 corrected. Regression test
  `ssf_clusters - quadriga gives delay 0 to clusters without delay spread` (RMa NLOS and RMa LOS with K = 0,
  `n_cluster = 1`, GR off/on, 2 frequencies: active delay 0, NLOS with abs TOA: `abs_toa`; legacy rule: 0).
- **M1** (maintainer variant): `quadriga` in an NLOS scenario with `absTOA_mu > -30` and N > 0 without `abs_toa` →
  `std::invalid_argument` ("... the NLOS clusters start at the excess delay of TR 38.901 clause 7.6.9."). With
  `abs_toa` the first NLOS cluster is at `abs_toa(l)` always, also for 0 (exact: `d - min + toa`). Legacy NLOS
  scenarios (`absTOA_mu <= -30`): unchanged (shift only where `abs_toa(l) > 0`). LOS and TR methods unchanged. The
  shift moved out of the kernel next to the TR addition (kernel lost its `toa` argument). Tests: error case (Catch2
  table, quadriga file, MEX, Python), shift case incl. Δτ = 0, legacy case (omitted / given), LOS ignored. All NLOS
  quadriga Catch2 cases now pass `abs_toa`; MEX/Python tests had no NLOS quadriga call, one each was added.
- **M2** docs rewritten by the binding rules (appendix texts; the 7-line method × property table without numbers and
  the LOS-slot consumer rule in the C++ `ssf_clusters` block only, MEX/Python refer to it; abs-TOA rule per M1;
  `absTOA_mu` listed as used key). Moved into source comments: K-limit reason and removal note (`qd_ssf_tr`),
  TR-literal points of 7.6.5.1 (`qd_ssf_mf`), algorithm/changes/properties of `quadriga` (`qd_ssf_quadriga`), GR
  geometry details (GR block), SplitMix64 rule (file comment of `qd_gbsm_phases.cpp`, machinery paragraph dropped).
  SECTION_DESC first paragraph without the output list, new paragraph 9 → 5 lines (MEX 6 at its 100-column width).
  Header SSF comments 26 → 8. `html_docu` regenerated (`make documentation`).
- **M3** tests: `tests/catch2_tests/qd_test_sos_oracle.hpp` (sum-of-sinusoids oracle, preset/ACF, floor offsets,
  grid, corr, equal) shared by `test_gbsm_phases.cpp` and `test_ssf_phases.cpp` (variance via `arma::var/stddev`).
  `tests/catch2_tests/qd_test_ssf.hpp`: one fixture (`make_fix` with F frequencies, anchor, all optional inputs),
  `run(sp, f, method)`, geometry/GR oracle, swap mirror, library-input drop (`lsf_*`, `lsp_generate`, `ssf_*`),
  `spread_ratios` (`calc_delay_spread`, `calc_angular_spreads_sphere`), track events, inactive-slot check. Generic
  cases loop over the three methods: sizes/slot layout/GR slot, LOS condition, link independence, LOS-only,
  extreme values, ignored inputs; swap test shared by `38.901`/`38.901_MF`; quadriga 1-/multi-frequency oracle
  cases merged; validation table-driven (one pass-through each for `qd_gbsm_check_links`, `scenpar_validate`).
  Every assertion class kept; coverage grew (GR slot and link independence now for all methods).
- **m1** `n_link > 0 &&` guard before the z check; UBSan clean.
- **m2** "comparison study" wording removed; the windows and their justification stated in the test.
- **m3** `centre` computed only for the TR methods. **m4** ρ once per cluster, threshold constant.
  **m5** GR weight/offset written only with GR. **m6** `qd_pi` → `arma::datum::pi`; `wrap180` kept (degree domain
  of step 5', commented).
- **m7** Python `gr_eps` with `ndim > 2` rejected ("Input 'gr_eps' must have size (n_link, n_freq)."), test added.
- **m8** `qd_mex_uint_entries` (`src/mex_quadriga_lib_functions.hpp`) and `qd_python_uint_entries`
  (`api_python/python_quadriga_adapter.hpp`) replace the six loops (lsf_draw, ssf_draw, ssf_clusters); the
  `n_cluster` message now ends "below 2^53." like `floor_id` (MEX/Python tests updated).
- Spec `md/ssf_spec.md`: W13, 3.3.1, 3.3.3, 3.3.5 (superseded note), 3.6 steps 6 and 10, 3.7 (qualitative table,
  consumer rule), 5.4 test rows, 10.3 (Δτ hand-off to NG, DS unchanged, legacy files) and the minimum-distance guard
  for the later scatterer series.

Nothing rejected.

## Documentation sizes (lines between `/*!MD` and `MD!*/`, before → after)

| Block | C++ | MEX | Python |
|---|---|---|---|
| `ssf_init_phases` | 65 → 39 | 70 → 34 | 61 → 34 |
| `ssf_draw` | 84 → 51 | 86 → 43 | 81 → 43 |
| `ssf_clusters` | 272 → 106 | 283 → 79 | 246 → 80 |

Sum 1248 → 509. `ssf_clusters` C++ is 6 above the ≈100 target (table + consumer rule + M1 rule).

## Test line counts (before → after)

- Cluster tests: 3119 → 2445 (`test_ssf_clusters.cpp` 1367 → 1007, `_mf` 709 → 350, `_quadriga` 1043 → 755,
  new `qd_test_ssf.hpp` 333). About 150 above the ≈2300 target; the header holds the shared helpers that replaced
  per-file copies, and the generic cases now cover all three methods.
- Phase tests: `test_ssf_phases.cpp` 844 → 762, `test_gbsm_phases.cpp` 1021 → 940, new `qd_test_sos_oracle.hpp` 100.

## Gates (load average 1-min in brackets; no timings reported)

- Catch2 full suite (`build_ssf`, OMP 10): All tests passed, 1,385,712 assertions in 1,439 test cases (5.3 → 4.8).
- ASan + UBSan (`build_ssf_asan`, library + SSF, phases and v19 conf test files, OMP 8): All tests passed, 33,161
  assertions in 86 test cases; 0 sanitizer reports, 0 leaks (3.9 → 4.2).
- MEX (g++-12, `scratch/ssf_r3/build`): Octave and MATLAB R2024a — `test_ssf_init_phases`, `test_ssf_draw`,
  `test_ssf_clusters`, `test_lsf_draw` PASS (1.4–7.3).
- objdump: all 86 MEX files ≤ GLIBCXX_3.4.26 (gate 3.4.28).
- Python (in-tree `build_linux`, installed to `lib/`): 35 passed (SSF + lsf_draw tests).
- Warnings: none (library `-Wall -Wextra -Wpedantic`, MEX, Python; SSF test files checked with `-Wall -Wextra`).

## Suggested commit message

```
Small-scale fading: review fixes for the cluster level

- quadriga: clusters without delay spread get delay 0 (was -ln(u) seconds)
- quadriga: abs_toa is required in NLOS scenarios that define the absolute TOA
  (TR 38.901 clause 7.6.9); the NLOS delays then start at abs_toa, also at 0
- Documentation of ssf_init_phases, ssf_draw and ssf_clusters reduced to the API
  (1248 -> 509 lines); implementation notes moved to source comments; method
  property table and the LOS-slot rule in ssf_clusters
- Wrappers: shared integer-entry check; Python rejects a 3-D gr_eps
- Tests: shared fixtures and oracles in test headers, method-generic cases and
  table-driven validation (cluster tests 3119 -> 2445 lines)
- Zero links with ground reflection no longer bind a null reference; cleanups

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```
