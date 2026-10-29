# SSF fixes, round 2 (on e266efc, not committed)

## Changes per finding

- M1: MEX `ssf_clusters`: spatial-consistency rule now "blocks 1 to 8 of `rnd` (columns N + 1 to 9·N)".
- M2: C++, MEX, Python argument lists: `eod_offset_deg` "used by 38.901 and quadriga in NLOS scenarios", `abs_toa`
  "required by quadriga in NLOS scenarios with absTOA_mu > -30", `o2i` "used by 38.901 in NLOS scenarios"; abs-TOA
  bullet without the requirement sentence; C++ table row "TR ZOD offset, O2I arrival elevation" deleted.
- m1: NLOS bullet "(`NumClusters > 1`)"; `NumClusters = 1` bullet: K/(K+1) = 1 also in the GR power; `rnd`,
  `rnd_freq`, `lsp_anchor` may be empty (all three APIs).
- m2: C++ table: quadriga spreads "at one frequency without ground reflection"; 38.901 cell "one frequency" → "n/a".
- m3: `ref_qd` fallback for `den = 0` is 0.0 (`test_ssf_clusters_quadriga.cpp`).
- m4: `quadriga_gbsm.hpp` SSF comment points to the `ssf_draw` documentation, not a source path.
- m5: `qd_gbsm_phases.cpp` file comment: SSF part 33 → 12 lines (purpose, column layout b·N + n, pointer to the
  `ssf_draw` doc and `qd_gbsm_gen_ssf`, SC_lambda/no bias, N = 0, seed rule in one sentence with `derive_seed`,
  append-only rule); generator table and SplitMix64 formula removed; stale "(index table above)" in
  `qd_gbsm_gen_ssf` → "(column layout above)".
- m6: `N_UMi_LOS`, `N_UMi_NLOS` in `qd_test_ssf.hpp` (removed from the three files); the two finiteness-only
  Diagnostic cases deleted. Cluster tests 2048 lines in the three files (was 2107) plus the header.
- `make documentation`: `html_docu/*` regenerated (SSF blocks and the line-number index).

## Tests (load average 1-min before → after)

- Catch2 (`build_ssf`, OMP 10): SSF/LSF/v19 filter: all passed, 34634 assertions in 86 cases; full suite: all
  passed, 1385681 assertions in 1437 cases (1439 before; the two deleted cases) (7.1 → 6.1). Build without warnings.
- MEX (g++-12, Octave): `test_ssf_init_phases`, `test_ssf_draw`, `test_ssf_clusters` PASS (8.5 → 8.5).
- Python (`build_linux`, installed to `lib/`): SSF tests 24 passed (6.7 → 6.7).
- No timings reported.

## Suggested commit message

```
Small-scale fading: documentation fixes for the cluster level

- ssf_clusters: the argument lists say which method uses eod_offset_deg and
  o2i and when quadriga requires abs_toa; MEX block numbers of the spatial-
  consistency rule corrected (1 to 8); NumClusters = 1 with ground reflection
  and empty random inputs documented; method table corrected
- Shorter source comment for the SSF phase sets; header refers to the API docs
- Tests: shared cluster counts, oracle fallback 0 for zero delay spread,
  two finiteness-only diagnostic cases removed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```
