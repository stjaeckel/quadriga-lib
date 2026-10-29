# R2b report — `ssf_clusters` method `38.901_MF`

## Summary
- Kernel `qd_ssf_mf` (TR 38.901 clause 7.6.5.1 steps 5'–7', TR-literal, wrap in 5' then scale by √(1+K) in 7') in
  `src/ssf_clusters.cpp`; the dispatch replaces the throw for `38.901_MF` (`quadriga` still throws, for R2c).
- Documentation block: compact formulas of 3.5 and the TR-literal points; swap-reciprocity bullet names `lsp_anchor`.
- New `tests/catch2_tests/test_ssf_clusters_mf.cpp`: the 9 cases of 5.3 plus the MF extreme-value case asked by 5.2;
  10 cases, 7585 assertions, all pass. Full suite passes; no warnings. Nothing committed.
- **No spec error found.**

## Files
| File | Change |
|---|---|
| `src/ssf_clusters.cpp` | +122/−10: `qd_ssf_mf` (~70 lines), dispatch (~22), doc block (+30) |
| `tests/catch2_tests/test_ssf_clusters_mf.cpp` | new, 709 lines |
| `tests/catch2_tests/qd_test_diag.hpp` | `QD_DIAG_GBSM` line lists `test_ssf_clusters*.cpp` (5.0) |

## Decisions and deviations
1. Spec errors: none. 3.5 checked against TR 7.6.5.1 text (eqs. 7.6.30a–m): the erfc form of the Laplacian, the
   signs of arg(exp(−j·…)), g factors, 7.6.30h split (LOS slot + cluster 0 = TR P_1, tested), √(1+K/2) delays.
2. Range of the principal value: `wrap180` gives [−180, 180) (3.0.1), the TR says "e.g. (−180, 180)"; differs only at
   exactly ±180° (measure zero; in LOS it would change the scaled angle by √(1+K)·360°). Kept as 3.0.1.
3. Kernel interface: arrays as `qd_ssf_tr` (shared-set row, z[k·N+n], anchor spreads, per-k spreads, per-k K with 0
   in NLOS, LOS centre in degrees); output layout of the R2a report. Delays and angles are computed per k also in
   NLOS (√(1+0) = 1 exactly), so the slices are bitwise equal without a copy branch (tested).
4. The degree wrap is a lambda in the kernel (single user); `qd_wrap`/`qd_elev` reused for the final wrap/fold.
5. The centre (LOS angles in degrees) is now computed once before the dispatch; the ZOD offset / O2I modifications
   stay inside the `38.901` branch (MF ignores both, W6). `38.901` outputs unchanged (its tests pass bitwise).
6. Tests beyond 5.3: abs TOA for MF (all n_fd slices, LOS slot stays 0), per-link `n_cluster` (normalisation over
   active clusters, inactive slots at every k), extreme values for MF (5.2 asks R2b to add its method), LOS with
   equal K at all frequencies gives equal slices, the order check "scale then wrap" differs by ≥ 0.25 rad.
7. Oracle `ref_mf` uses `std::arg(std::polar(1, −x))` for the TR's principal value and the TR's V-form of the
   Laplacian, linear powers; fixture spreads at k are 0.5–1.2 × anchor (g > 0) and anchor AS ≥ 20°, so no linear
   power underflows (relative 1e-12 holds).
8. Size: test file 709 lines vs. "about 350": typed oracle, fixture and checks for 10 cases; code + docs ~130.

SPEC_CPP_API review points A/B/C: no API change.

## Test results
- Build `/home/stephan/quadriga-lib-agents/build_ssf` (cmake re-run for the new file), `--parallel 12`,
  `-Wall -Wextra -Wpedantic`: 0 warnings (forced recompile of both files).
- Stub run (R2a throw in place): `test cases: 10 | 10 failed`, `assertions: 11 | 11 failed`, all 11 with
  "Method '38.901_MF' is not implemented yet." (right reason).
- Final: `"*38.901_MF*"`: `All tests passed (7585 assertions in 10 test cases)`, each < 2 ms;
  `"ssf_clusters*"`: 14244 assertions in 28 cases.
- Measured errors (QD_DIAG_GBSM): NLOS delay 0, power 6.7e-15, angle 1.8e-15 rad; LOS delay 0, power 1.4e-14,
  angle 6.9e-15 rad; swap 1.8e-15 rad; GR power 8.9e-16.
- Mutation probe: removing the step-5' wrap (scale-then-wrap) fails the LOS and NLOS cases (300 assertions); restored.
- Full suite from the repository root, `OMP_NUM_THREADS=12`: `All tests passed (1371530 assertions in 1432 test
  cases)`, 26.9 s wall; load 2.24 before, 3.01 after.

## Warnings
None.

## For R2c
- Dispatch: `if (m_tr) {...} else if (m_mf) {...}` inside `if (N_l > 0)`; add `else` for `qd_ssf_quadriga` and
  replace `if (m_qd) throw ...` after the validation.
- `centre` (LOS angles in degrees) is computed before the dispatch; `quadriga`'s NLOS EoD offset (mean over k) must
  be applied in its own branch.
- The `z` copy from `rnd_freq` still runs for every method; it must be skipped for `quadriga` (`rnd_freq` may be
  nullptr).
- `qd_test_diag.hpp` already lists `test_ssf_clusters*.cpp`.

## Open points / risks
- The LOS jumps of the TR order are by design (documented); not measured on a track here.
- `html_docu/` not regenerated (R3).

## Suggested commit cut
One commit: `src/ssf_clusters.cpp`, `tests/catch2_tests/test_ssf_clusters_mf.cpp`, `tests/catch2_tests/qd_test_diag.hpp`.

```
Small-scale fading: multi-frequency cluster method of TR 38.901 clause 7.6.5.1

ssf_clusters gains the method "38.901_MF", the alternative channel generation of TR 38.901
clause 7.6.5.1 as written: nominal delays and angles from the anchor spreads (r = 1.5,
wrapped Gaussian azimuths, wrapped Laplacian zeniths through erfc for full precision in the
tails), cluster powers per frequency with the g factors of eqs. (7.6.30b)-(7.6.30f) and
per-cluster shadowing drawn per frequency, normalised in the log domain, and the LOS scaling
of delays by sqrt(1 + K/2) and of the wrapped nominal angles by sqrt(1 + K) at each
frequency. Delays and angles have one slice per frequency; in NLOS they are equal at all
frequencies. The LOS power stays in its own slot (eq. (7.6.30h) split into the LOS path and
the first cluster); absolute time of arrival, ground reflection and per-link cluster numbers
follow the shared rules. EoD offset and O2I flag are ignored, as the clause has neither.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```
