# R1 report — random fields (`ssf_init_phases`, `ssf_draw`)

## Summary
- Two public functions added to `src/qd_gbsm_phases.cpp`, declared in `include/quadriga_gbsm.hpp`, with doc blocks.
- One file-local generator table `qd_gbsm_gen_ssf`; evaluation reuses `qd_gbsm_freq` / `qd_gbsm_positions` /
  `qd_gbsm_eval` unchanged (no new machinery). Seed rule as a lambda inside `ssf_init_phases`.
- New test file `tests/catch2_tests/test_ssf_phases.cpp`, 18 cases of spec 5.1, 1111 assertions, all pass.
- Full Catch2 suite passes (1403 cases, 1 357 259 assertions); no compiler warnings.
- Nothing committed.

## Files
| File | Change |
|---|---|
| `include/quadriga_gbsm.hpp` | +38: "Small-scale fading" group comment, two declarations |
| `src/qd_gbsm_phases.cpp` | +328 / −19: header comment (SSF index, seed rule, append-only rule replace the "Design for further phase sets" paragraph), `qd_gbsm_gen_ssf`, `ssf_init_phases`, `ssf_draw` with doc blocks (~95 code lines, ~190 doc/comment lines), `<cstdint>` |
| `tests/catch2_tests/test_ssf_phases.cpp` | new, 844 lines |
| `tests/catch2_tests/qd_test_diag.hpp` | `QD_DIAG_GBSM` line lists `test_ssf_phases.cpp` |

(`md/ssf_plan.md` was already modified before this run; not touched.)

## Decisions and deviations
1. Header group comment (2.1): the last bullet names only `src/qd_gbsm_phases.cpp (ssf_init_phases, ssf_draw)`;
   the spec text also lists `src/ssf_clusters.cpp`, which does not exist yet. **R2a adds ", src/ssf_clusters.cpp"**.
2. `ssf_init_phases`, N = 0 (3.0.9): output is sized `[n_coeff, 0, 2]` after the validation; `set_id` and `seed` are
   validated also then.
3. `ssf_draw`, n_gen = 0: any cube with `n_elem == 0` is accepted; a non-empty cube throws the size message with
   `[n_coeff, 0, 2]` (spec: "any empty phase accepted"; the reverse is my reading).
4. `ssf_draw` validation order: positions, `set_id`, preset, scenpar, phase size, finite, floor_id length,
   floor_id with 3D — as 3.2 ("3.0.9" adds no check beyond `scenpar_validate`'s NumClusters ≥ 1).
5. Doc block of `ssf_draw`: the "caller's rule for 38.901 under SC" is written as "Clusters by TR 38.901 clause
   7.5 under spatial consistency …" (no reference to a method/function that does not exist yet). The paper is cited
   by authors, title, year and eqs. (3)–(5).
6. `qd_gbsm_eval`'s comment now names `qd_gbsm_gen_lsf or qd_gbsm_gen_ssf` (comment only, code unchanged).
7. Tests, beyond the table of 5.1: the ACF case compares Uniform columns with the Pearson correlation
   (6/π)·asin(ρ/2) of the Φ-mapped values (spec: "all columns" against the ACF; the Uniform map lowers the
   correlation by 0.015 at λ'); the SC_lambda case also checks that the large-scale distances do not enter; the
   column-independence case also checks dual mobility (keeping column c and its mirror, both of which enter c).
8. Test file 844 lines vs. "about 600": the extra lines are the oracle helpers copied from `test_gbsm_phases.cpp`
   (test files share no helpers) and the dual-mobility/floor combinations.

SPEC_CPP_API review points A/B/C: no new API beyond the spec's declarations; argument order and defaults as 2.1.

## Test results
- Build: `cmake -S /home/stephan/quadriga-lib-ssf -B /home/stephan/quadriga-lib-agents/build_ssf -D ENABLE_TESTS=ON
  -D ENABLE_MATLAB=OFF -D ENABLE_OCTAVE=OFF -D ENABLE_MEX_DOC=OFF -D ENABLE_PYTHON=OFF -D CMAKE_BUILD_TYPE=Release`,
  `cmake --build … --parallel 12` (AVX2 on). Library compiled with `-Wall -Wextra -Wpedantic`: no warnings; the
  test file compiled separately with the same flags: no warnings.
- Against the stubs (sizes right, zeros): `test cases: 18 | 4 passed | 14 failed`, `assertions: 1111 | 447 passed |
  664 failed`. The 4 passing were the structural cases a zero stub satisfies (output sizes, SC_lambda 0 = 0.1,
  swap rule, sizes/empty input); all value, seed, mirror, statistics and validation checks failed.
- Final: `OMP_NUM_THREADS=12 build_ssf/test_bin "ssf_*"`: `All tests passed (1111 assertions in 18 test cases)`,
  1.1 s wall.
- Full suite, from the repository root: `All tests passed (1357259 assertions in 1403 test cases)`, 47.6 s wall;
  load before 2.02, after 21.85 (other sessions).
- Margins (QD_DIAG_GBSM): SOS oracle max diff 3.2e-7 (single) / 3.5e-7 (dual) vs. 2e-5; floor correlation
  |r| ≤ 0.029 vs. 0.05; ACF at λ': Normal 0.374–0.416 vs. 0.368, Uniform 0.361–0.391 vs. 0.353 (bound 0.1);
  phase correlation of the frequency sets 0.0066 vs. 0.090.

## Warnings
None from the compiler. CMake prints its usual Catch2 URL_HASH / unused CMAKE_C_COMPILER notes (pre-existing).

## For the next runs
- R2a: add `src/ssf_clusters.cpp` to the header group comment (decision 1).
- The `ssf_draw` output of the shared set has the delay block already mapped to Uniform; blocks 5–8 are Normal and
  need Φ(x) = 0.5·erfc(−x/√2) where a method wants Uniform (2.2.1).
- `qd_gbsm_gen_ssf` is in the anonymous namespace of `qd_gbsm_phases.cpp`; `ssf_clusters` (other file) does not
  need it — it validates `rnd` sizes from `NumClusters` only.
- `html_docu/` is not regenerated (R3).
- Test diagnostic lines use `QD_DIAG_GBSM`.

## Open points / risks
- The full suite's OpenMP code uses all hardware threads unless `OMP_NUM_THREADS` is set (the first `ssf_*` run
  used 31 s CPU in 1.1 s); runs here were limited to 12 threads.
- The swap and column-independence tests are bitwise; they rely on `sos_val` evaluating columns independently
  (true for the current scalar and AVX2 kernels).

## Suggested commit cut
One commit, files: `include/quadriga_gbsm.hpp`, `src/qd_gbsm_phases.cpp`, `tests/catch2_tests/test_ssf_phases.cpp`,
`tests/catch2_tests/qd_test_diag.hpp`.

```
Small-scale fading random fields: ssf_init_phases and ssf_draw

Spatially correlated cluster variables of TR 38.901 clause 7.6.3.1 as sum-of-sinusoids sets:
a shared, frequency-independent set of 9·N generators (cluster delay Uniform; signs and angle
offsets of AOD, AOA, ZOD, ZOA Normal, departure and arrival mirrored for dual mobility) and one
set of N cluster-shadowing generators per frequency, N = NumClusters - 1. All use SC_lambda
(0 used as 0.1 m), the floors and 2D/3D modes of the large-scale set, and no bias removal.
Seeds derived from the large-scale seed by SplitMix64 and a set tag, so the sets are independent
streams and never repeat the large-scale phases. Columns are append-only: later blocks keep the
existing values bit for bit. NumClusters = 1 gives empty sets.
```
