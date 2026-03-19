# AVX2 Interpolation Refactoring — Results

**File:** `qd_arrayant_interpolate_avx2.cpp`
**Date:** 2026-03-17

---

## Summary

Refactored the AVX2 antenna pattern interpolation function from a nested
angle/element loop with per-iteration branching into a collapsed flat loop
with full AVX2 vectorization across all pipeline stages, including the grid
search. The original AVX2 implementation was *slower* than scalar for most
scenarios (0.46x–1.03x); the final version achieves 3.76x–6.17x speedup.

---

## Architecture

**Collapsed flat loop** over `N_total = n_ang × n_out` items, processed 8 at
a time in AVX2 registers. Each lane independently computes its own (angle,
element) pair from the flat index.

**Flat SoA precomputation** moves all branching (`per_element_angles`,
`per_element_rotation`, `per_angle_rotation`) into an O(N) scalar pass before
the hot loop. The AVX2 loop is completely branch-free.

**Pipeline stages per 8-wide vector:**

| Stage | What                           | Method                                                                                      |
| ----- | ------------------------------ | ------------------------------------------------------------------------------------------- |
| A     | Input angles → Cartesian       | AVX2 sincos, pole guard, mul                                                                |
| B     | Rotation → local geographic    | 9-wide SoA loads, 3×3 FMA, cart2geo                                                         |
| C     | Basis vectors + gamma          | AVX2 mul/FMA, dot products                                                                  |
| D     | Distance                       | FMA chain, signum, sqrt, negate                                                             |
| E     | Grid search                    | Uniform: O(1) AVX2 index + gather weight; non-uniform: vectorized binary search with gather |
| F     | Pattern gather + 6× SLERP      | AVX2 gather + `_fm256_slerp_complex_ps`                                                     |
| G     | Polarization rotation + stores | FMA, tail-masked stores                                                                     |

**Tail masking:** Last vector uses `_mm256_maskstore_ps` (float) or scalar
loop (double). Padding slots duplicate last valid item so all AVX2 computation
is safe.

---

## Key Assumptions & Design Decisions

- **All computation in float.** Both `float` and `double` template
  instantiations run the same float compute path. Inputs cast to float during
  precomputation; outputs promoted back to `dtype` on store. For true
  double-precision results, use the scalar reference.

- **Double gather via cvt.** For `dtype=double`, pattern data is gathered as
  `__m256d` (two 4-wide gathers per 8-wide vector) and converted to float via
  `_mm256_cvtpd_ps` + `_mm256_set_m128`. This allows the double path to share
  the same AVX2 SLERP and rotation code as the float path.

- **Double stores via cvt.** Results are converted back with
  `_mm256_cvtps_pd` and stored as `__m256d` pairs. Tail vectors use a scalar
  loop to avoid overwriting past `N_total`.

- **Constants use float precision.** `R0`, `R1`, `tL`, `tS`, `dT`,
  `az_diff`, `el_diff` all use `arma::Datum<float>::eps` regardless of
  `dtype`.

- **Stage E (grid search) is fully vectorized.** Three paths per axis,
  selected by loop-invariant branches (predicted perfectly after iteration 1):
  (1) n==1: trivial zero-index; (2) uniform grid: O(1) arithmetic index
  lookup via `floor((angle - grid_min) * rinv)` with gathered grid values for
  weight computation (avoids catastrophic cancellation from `fidx - floor(fidx)`
  at large index values); (3) non-uniform grid: vectorized binary search with
  fixed `ceil(log2(n+1))` iterations using `_mm256_i32gather_ps` + `blendv`.
  Azimuth wraps circularly; elevation clamps at boundaries. Uniformity is
  detected once at function entry (relative tolerance 1e-4).

- **`_fm256_slerp_complex_ps` handles all edge cases** (zero amplitude,
  near-antipodal, cPhase clamping). No reimplementation of scalar SLERP logic.

- **SLERP weight convention:** `w=0→A, w=1→B`. Pass `w=up` for azimuth pair,
  `w=vp` for elevation pair.

- **OpenMP threshold at 4096 vectors.** Primary case (n_out=64, n_ang=1 →
  n_vec=8) never engages OMP.

---

## Refactoring Steps

| Step | Change                                               | Key effect                        |
| ---- | ---------------------------------------------------- | --------------------------------- |
| 1    | 8-wide block stride + remainder loop                 | Structural prep, no perf change   |
| 2    | SoA data layout for R and pos                        | Eliminated AoS stride overhead    |
| 3    | Collapsed flat loop + precomputation                 | Branch-free hot loop, 1.4–2.9x    |
| 4    | AVX2 Stages A–D (sincos, rotation, gamma, dist)      | +15–35% over Step 3               |
| 5    | AVX2 Stages F–G (gather, SLERP, rotation, stores)    | Full pipeline vectorized          |
| 6    | AVX2 Stage E (uniform O(1) + binary search fallback) | Eliminated grid search bottleneck |

---

## Final Performance

| Scenario | n_out | Grid | dtype  | Scalar (ms) | AVX2 (ms) | Speedup | Baseline |
| -------- | ----: | ---- | ------ | ----------: | --------: | :-----: | :------: |
| S1       |     1 | 5°   | float  |      0.0037 |    0.0010 |  3.76x  |  0.96x   |
| S2       |     8 | 5°   | float  |      0.0061 |    0.0013 |  4.75x  |  0.46x   |
| S3       |    64 | 5°   | float  |      0.0267 |    0.0058 |  4.57x  |  1.03x   |
| S4       |     8 | 1°   | float  |      0.0077 |    0.0020 |  3.85x  |  0.76x   |
| S5       |    64 | 1°   | float  |      0.0389 |    0.0065 |  6.00x  |  0.90x   |
| S6       |     8 | 5°   | double |      0.0076 |    0.0012 |  6.17x  |  0.47x   |

"Baseline" = original AVX2 speedup before refactoring.

**Speedup progression across steps:**

| Scenario | Baseline | Step 3 | Step 4 | Step 5 | Step 6 |
| -------- | :------: | :----: | :----: | :----: | :----: |
| S1       |  0.96x   | 2.50x  | 2.72x  | 3.60x  | 3.76x  |
| S2       |  0.46x   | 2.21x  | 2.96x  | 4.81x  | 4.75x  |
| S3       |  1.03x   | 1.43x  | 1.90x  | 3.84x  | 4.57x  |
| S4       |  0.76x   | 2.16x  | 2.74x  | 3.45x  | 3.85x  |
| S5       |  0.90x   | 1.58x  | 1.83x  | 3.25x  | 6.00x  |
| S6       |  0.47x   | 2.89x  | 3.15x  | 5.68x  | 6.17x  |

Step 6 impact is largest on S5 (1° grid, n_out=64): 3.25x → 6.00x, consistent
with the grid search being the dominant bottleneck for large grids and many
elements. S1/S2/S6 are within run-to-run variance (Stage E was already a
small fraction there).

---

## Final Accuracy

| Scenario | max ULP V_re | avg ULP V_re | max ULP gamma | max phase err |
| -------- | -----------: | -----------: | ------------: | ------------: |
| S1       |            1 |          1.0 |             4 |   5.7e-09 rad |
| S2       |            5 |          2.0 |             4 |   1.2e-07 rad |
| S3       |         1389 |         46.7 |             4 |   1.4e-06 rad |
| S4       |           15 |          4.6 |             4 |   2.0e-07 rad |
| S5       |          405 |         19.0 |             4 |   1.4e-06 rad |
| S6       |            6 |          1.2 |             3 |   6.4e-08 rad |

S3/S5 max ULP outliers are inherent to the SLERP interpolation chain and
exist in the scalar reference too. Step 6 dramatically improved S5
(max ULP 1533 → 405) by using gather-based weight computation instead of
the fidx fractional-part approach that suffered catastrophic cancellation
on large grids. S4 max ULP increased slightly (7 → 15) due to minor weight
differences between the O(1) index path and the scalar linear search, but
remains excellent. Average ULP and phase errors are small across all scenarios.
No meaningful accuracy regression relative to baseline.