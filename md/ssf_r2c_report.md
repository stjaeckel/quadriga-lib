# R2c report — `ssf_clusters` method `quadriga`

## Summary
- Kernel `qd_ssf_quadriga` (spec 3.6) in `src/ssf_clusters.cpp`; the dispatch replaces the last
  "not implemented" throw; the `rnd_freq` copy is skipped for `quadriga` (may be nullptr).
- Documentation block: compact formulas of 3.6, the changes against the paper/QuaDRiGa, the properties.
- New `tests/catch2_tests/test_ssf_clusters_quadriga.cpp`: the 16 cases of 5.4 plus the extreme-value case asked by
  5.2; 17 cases, all pass. Full suite passes; no warnings. Nothing committed.
- Two spec points do not hold as written (fixture of the cap case; "AS exact" for elevation with GR in LOS), see 1, 2.

## Files
| File | Change |
|---|---|
| `src/ssf_clusters.cpp` | +204/−4: kernel `qd_ssf_quadriga` (~140), dispatch (~20), doc block (+45) |
| `tests/catch2_tests/test_ssf_clusters_quadriga.cpp` | new, 1043 lines |

`qd_test_diag.hpp` already lists `test_ssf_clusters*.cpp` (R2b).

## Decisions and deviations
Spec errors / spec statements that do not hold:
1. **5.4 cap case fixture**: ASA 170° and ESA 100° do not always exceed the caps. The per-link linear initial spread
   is often well above the quoted 42°/44° (power weighting); on the 4 fixture links the unlimited azimuth scale was below 3 in 7
   of 8 cases (e.g. 2.72–2.92) and one elevation scale below 1.5. The test uses 300° and 200° (all unlimited scales above the caps,
   checked). Proposal for the spec: "ASA 300°, ESA 200°".
2. **3.6 "AS exact below the caps" fails for elevation with GR in LOS.** The fixed LOS and GR elevations alone often
   have a larger linear spread than the target ES; the iteration then cannot converge (9 updates drive the scale
   down, the output ES stays above the input). 5.4 fixture (UMi LOS, GR, ES ≤ 10°): 310 of 400 elevation cases
   unreachable. Same behaviour as NG (D3), so no code change; the test checks the converged cases (77 ESD, 13 ESA
   links with GR in LOS; all 200 in the other variants and for the azimuths) and the doc block states the limit.
   Proposal for the spec 3.6 / 3.7: "AS exact below the caps, except elevation with GR when the LOS–GR pair alone
   exceeds the target".
3. GR drop measurement (diagnostic, UMi LOS 6 GHz, ε = 13.5 − 1.18j): DS from the slot powers 0.823 (study 0.845,
   p10–p90 0.81–0.90): consistent. New: ASD 0.80, ASA 0.70, ESD 0.94, ESA 0.92 from slot powers (the LOS slot holds
   K/(K+1), the scaling used (1 − R²)·K/(K+1)); not in the study. The doc says "about 0.85" for DS only.

Choices within the spec:
4. Initial angles as `erf(y/√2)·π/2`, identical to `(Φ(y) − 0.5)·π` and exact near 0 (oracle uses the erfc form).
5. Delay scale: the GR closed form whenever `GR_enabled` (in NLOS scenarios S_GR = 0, same value as the no-GR
   form); `den = c − e²`; the averages over k accumulate `s_k/F`; δ accumulates `offset/r2d/F` (NLOS only).
6. Abs TOA: the dispatch passes `abs_toa(l)` only in NLOS scenarios, the kernel shifts when it is > 0.
7. The split weights: `S_LOS = K/(1+K)·(1 − G·R²)`, as written; output powers coefficient powers via the shared slot
   writer (R2a's factor 1 for `quadriga`).
8. Kernel: file-local named function directly above the doc block (3.3); `efold` a lambda (single user).
9. Tests beyond 5.4: rotation about the z-axis shifts only the azimuths (additive LOS rule, independent of the
   oracle); per-link `n_cluster` and GR in the single-frequency case; GR on and F = 2 in the swap case; the
   extreme-value case of 5.2 for `quadriga`; a GR row in the drop diagnostic. Oracle `ref_qd` uses linear power
   normalisation, the erfc map, the DS scale as the larger root of the quadratic, its own iteration.
10. DS case asserts `n_clamped < 20` (measured 0 of 400) besides printing it.
11. Size: test file 1043 lines vs "about 600" (typed oracle with iteration, 17 cases, drop and track set-ups);
    code + docs ~205 vs "about 350".

SPEC_CPP_API review points A/B/C: no API change.

## Test results
- Build `/home/stephan/quadriga-lib-agents/build_ssf` (cmake re-run for the new file), `--parallel 12`,
  `-Wall -Wextra -Wpedantic`: 0 warnings (forced recompile of both files).
- Stub run (throw in place): `test cases: 17 | 17 failed`, `assertions: 18 | 18 failed`, all with
  "Method 'quadriga' is not implemented yet." (right reason).
- Final `"ssf_clusters - quadriga*"` + diagnostic: 17 cases pass, 0.23 s wall; `"*ssf_clusters*"`:
  `All tests passed (27692 assertions in 46 test cases)`.
- Measured errors against `ref_qd`: one frequency delay 1.0e-14, power 2.2e-15 (relative), angle 1.8e-15 rad;
  several frequencies 5.9e-14, 6.6e-15, 1.8e-15 rad. DS from the output 1e-15 relative (0 clamped links); linear
  AS 9e-16 (no GR), 2.3e-5 (GR, iteration threshold). Swap 8.9e-16 rad. Tracks: 0 events in UMi LOS (105 526
  cluster samples) and NLOS (182 134).
- Full suite from the repository root, `OMP_NUM_THREADS=12`: `All tests passed (1384974 assertions in 1449 test
  cases)`, 26.9 s wall; load 1.64 before, 3.17 after.

## Spread medians of the diagnostic case (6 GHz, 2000 links each, library definitions) vs study (3.6, 3.7)
| Scenario | DS | ASD | ASA | ESD | ESA | Study |
|---|---|---|---|---|---|---|
| UMi NLOS | 1.0000 | 1.0000 | 1.0001 | 1.0000 | 1.0000 | 1.00 all |
| UMi LOS | 1.0000 | 1.0000 | 0.8483 | 1.0000 | 1.0000 | ASA 0.86, others 1.00 |
| UMa NLOS | 1.0000 | 1.0000 | 1.0007 | 1.0000 | 1.0000 | 1.00 all |
| UMa LOS | 1.0000 | 1.0000 | 0.6038 | 1.0000 | 1.0000 | ASA 0.59, others 1.00 |
| UMi LOS + GR | 0.8227 | 0.8047 | 0.6962 | 0.9449 | 0.9182 | DS 0.845 (p10–p90 0.81–0.90) |

## For R3
- `quadriga`: `rnd_freq`, `lsp_anchor`, `o2i` ignored and not validated; n_fd = 1; `eod_offset_deg` [n_link, n_freq]
  (mean over frequencies, NLOS only); wrappers need no method-specific code.
- The C++ doc block now has all three methods; the MEX/Python blocks adapt it (the `quadriga` formulas are in rad).

## Open points / risks
- GR elevation non-convergence (item 2) is NG's behaviour; a maintainer decision whether the spec text changes.
- Oracle and function could take different iteration branches if a fixture value lands within rounding of the
  1e-4 dB threshold (not seen; the diagnostic prints the errors).
- `html_docu/` not regenerated (R3).

## Suggested commit message
```
Small-scale fading: spatially consistent multi-frequency cluster method

ssf_clusters gains the method "quadriga", the cluster generation of Jaeckel et al.,
"A Spatially Consistent Geometric D2D Small-Scale Fading Model for Multiple Frequencies"
(2019): unitless exponential initial delays and uniform initial angles from the
spatially consistent random values, cluster powers as the product of the paper's
delay and angle terms with fitted coefficients per frequency, the K-factor split
with the ground reflection as a fixed path, a closed-form delay scale and an
iterative angle scale (limited to 3 in azimuth and 1.5 in elevation), both averaged
over the frequencies. Delays and angles are the same at all frequencies; at one
frequency the delay spread of the output equals the input and the linear angular
spreads equal the input below the limits. The LOS direction is added to the scaled
offsets (azimuth wrapped, elevation mirrored at the poles), the TR 38.901 EoD offset
is applied to the NLOS departures of NLOS scenarios, and the absolute time of arrival
shifts the NLOS delays in NLOS scenarios. Outputs are coefficient powers (LOS K/(K+1),
ground reflection K/(K+1)·R²). The method is continuous along tracks and
swap-reciprocal with dual-mobility random values.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```
