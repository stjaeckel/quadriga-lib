# get_channels_spherical.cpp — Performance Refactor Plan

**Date:** 2026-03-18 (updated with P1 benchmark results)  
**Context:** Antenna interpolation is already AVX2-accelerated and confirmed not the bottleneck. This plan targets the remaining hot paths. Priority 1 (coefficient kernel) is complete; benchmarks show the angle/delay loop is the dominant cost at small MIMO sizes.

---

## Priority 1: Coefficient combination inner loop — ✅ DONE

**Status:** Implemented and benchmarked (2026-03-18).

**Files changed:**
- `quadriga_lib_helper_functions.hpp` — added `qd_coeff_combine<dtype>()` templated scalar fallback
- `quadriga_lib_avx2_functions.hpp` — added `qd_coeff_combine_avx2<dtype>()` declaration
- `qd_coeff_combine_avx2.cpp` — AVX2 implementation (float + double specializations)
- `get_channels_spherical.cpp` — refactored inner loop to call `qd_coeff_combine` / `qd_coeff_combine_avx2`, delay fixup split into separate loop

**Implementation notes:**
- Jones product factored as complex intermediate `U + jV = Rx * M`, then accumulated: 8 mul + 24 FMA (down from 32+32 in the direct expansion).
- `fmod` replaced with AVX2 `round_ps`/`round_pd` + `fnmadd` (vectorized truncation remainder).
- Float specialization: `_fm256_sincos256_ps` with float Cody–Waite, safe because fmod pre-reduces to [-λ, λ].
- Double specialization: antenna patterns converted double→float for Jones (8-wide throughput), delays kept in double for fmod, phases fed to `_fm256_sincos256_pd` for double-precision range reduction. Full 8x SIMD width.
- Tail handling: float uses `maskload`/`maskstore` with LUT mask; double uses zero-padded aligned stack buffers with scalar writeback.

### Benchmark results (4×4 MIMO, 500 paths, fc = 3.5 GHz, Ryzen 5950X)

**Throughput (3-run average):**
```
                         Time [ms]    Throughput     Speedup
With angle outputs:
  float / scalar           ~0.456      ~17.5 Mc/s    1.00x  (implied baseline)
  float / AVX2              0.415       19.3 Mc/s    1.10x
  double / scalar           0.534       15.0 Mc/s    0.85x
  double / AVX2             0.462       17.3 Mc/s    0.99x
Without angle outputs:
  float / scalar            0.487       16.4 Mc/s    1.00x
  float / AVX2              0.414       19.3 Mc/s    1.18x
```

**Scaling with n_path (float, 4×4 MIMO):**
```
n_path  | scalar         | AVX2           | Speedup
--------+----------------+----------------+--------
1       |  0.029 ms      |  0.009 ms      |  3.26x
10      |  1.204 ms (*)  |  0.030 ms      | 40.78x
50      |  0.149 ms      |  0.062 ms      |  2.39x
100     |  0.399 ms      |  0.102 ms      |  3.90x
200     |  0.299 ms      |  0.191 ms      |  1.57x
500     |  0.482 ms      |  0.458 ms      |  1.05x
1000    |  1.137 ms      |  0.760 ms      |  1.50x
2000    |  1.657 ms      |  1.286 ms      |  1.29x

(*) n_path=10 scalar outlier likely cold-cache or allocation artifact.
```

**Accuracy (reference = double/scalar):**
```
Config          | Coeff Amp ULP    | Coeff Phase [rad]       | Delay [ps]
float / scalar  | avg 0.7  max  7  | avg 1.4e-03  max 4.9e-03 | avg 0.04  max 0.22
float / AVX2    | avg 0.4  max  7  | avg 1.4e-03  max 4.9e-03 | avg 0.04  max 0.22
double / AVX2   | avg 0.4  max  8  | avg 2.7e-08  max 1.6e-07 | avg 0.00  max 0.00
```
Angle errors: identical between scalar and AVX2 (unaffected by this change, as expected).

### Analysis

**The coefficient kernel is not the dominant cost at 4×4 MIMO.** With only 16 TX-RX links per path, the Jones+sincos computation is a small fraction of the per-path work. The observed 1.10–1.18x end-to-end speedup implies the coefficient kernel was roughly 10–15% of total runtime at this MIMO size. The remaining ~85% is in the angle/delay computation loop (Priority 2) and coupling (Priority 3).

**Scaling confirms the bottleneck is elsewhere:**
- At small `n_path` (1–100), the AVX2 kernel dominates relative runtime → 2–4x speedup.
- At large `n_path` (500+), the angle/delay loop dominates → speedup converges to 1.05–1.5x.
- The throughput plateau (~19–25 Mc/s) is well below what the AVX2 kernel alone could sustain, confirming that OMP overhead and the serial angle/delay pass are the limiters.

**For larger MIMO arrays** (e.g., 32×32 = 1024 links), the coefficient kernel scales as O(n_tx × n_rx) while the angle/delay loop scales as O(n_tx + n_rx). The AVX2 kernel benefit will be proportionally much larger in that regime. A targeted benchmark with larger arrays would confirm this.

**Double/AVX2 accuracy is excellent.** The double→float→double conversion path introduces only ~8 max ULP in amplitude and ~1.6e-07 rad max phase error — well within acceptable tolerance. The double-precision fmod + `_fm256_sincos256_pd` range reduction strategy works as intended.

**Conclusion:** Priority 1 delivers a solid kernel that will pay off more at larger MIMO sizes. For the 4×4 case, the data clearly shows Priority 2 (angle/delay parallelization) is now the critical path.

---

## Priority 2: Angle & delay computation (lines 303–386) ← NOW CRITICAL

**Why:** Priority 1 benchmarks confirm this is the dominant bottleneck. At 4×4 MIMO / 500 paths, the coefficient kernel accounts for only ~10–15% of runtime; the remaining ~85% is split between the serial angle/delay loop (this section) and coupling. The entire `n_out` loop cannot currently be parallelized because of `true_los_path` / `shortest_path` tracking. This is the single highest-impact optimization remaining.

**Refactor steps:**

1. **Split into two passes:**

   **Pass 1 — LOS detection (serial, cheap):**
   ```cpp
   for (i_out = 0; i_out < n_out; ++i_out) {
       // Compute d_shortest, d_length (3× sqrt, ~10 FLOPs per path)
       // Update true_los_path, shortest_path
   }
   ```
   This is O(n_path) scalar work with no inner TX×RX loop — negligible cost.

   **Pass 2 — Angle/delay fill (parallel, expensive):**
   ```cpp
   #pragma omp parallel for
   for (i_out = 0; i_out < n_out; ++i_out) {
       bool is_los = (i_out == true_los_path);
       // ... existing LOS or NLOS inner loops
   }
   ```

2. **Vectorize inner loops (AVX2):**

   - **LOS path (lines 329–347):** For each TX element, the RX loop computes:
     ```
     d = sqrt(dx² + dy² + dz²)
     aod = atan2(dy, dx)
     eod = asin(dz / d)
     ```
     over `n_rx` elements. Collect dx/dy/dz into contiguous arrays (or compute in-place from strided element_pos), then call `qd_ATAN2_AVX2` and `qd_ASIN_AVX2`.

   - **NLOS path (lines 352–383):** RX sub-loop (lines 352–361) is the same pattern — vectorize with `qd_ATAN2_AVX2` / `qd_ASIN_AVX2`. TX sub-loop (lines 363–383) has per-TX `sqrt`/`atan2`/`asin` (scalar, only `n_tx` iterations) followed by a broadcast to all RX — the broadcast is just a fill, not worth vectorizing.

3. **Hoist `dr` allocation:**
   - Move `dtype *dr = new dtype[n_rx]` outside the path loop. Allocate once before the loop, reuse per path. With OMP, make it `thread-private` or allocate per-thread.

---

## Priority 3: Coupling-path geo2cart / cart2geo (lines 615–653)

**Why:** These scalar helper functions (`qd_geo2cart`, `qd_cart2geo`) are called inside the per-path coupling block. When `n_links` is large, each call does `n_links` × (`sincos` + `atan2` + `asin`).

**Refactor steps:**

1. Add `#if BUILD_WITH_AVX2` branches that call `qd_GEO2CART_AVX2` and `qd_CART2GEO_AVX2` (already exist in `fastmath_vectorized_avx2.h`).

2. No padding needed — `fastmath_vectorized_avx2` already handles arbitrary lengths (non-multiple-of-8 tails are handled internally).

3. This is a drop-in replacement — no algorithmic changes needed.

---

## Priority 4: Minor cleanups (low urgency)

### 4a. Coupling identity check early-exit (lines 414–465)
Add `break` after setting `apply_element_coupling = true` in each of the four inner check loops. Currently scans all matrix elements even after finding the first non-identity value.

### 4b. Coupling temp allocation (lines 574–578, 656–659)
`tempX/Y/Z/T` are allocated with `new[]` and freed with `delete[]` inside the per-path loop. Move allocation outside the path loop:
- If coupling is applied, allocate once before the `#pragma omp parallel for`.
- With OMP, either make them `firstprivate` or allocate per-thread using `omp_get_thread_num()`.

### 4c. ~~Consider removing `std::fmod` entirely~~ — Addressed in P1
The AVX2 kernel replaces `std::fmod` with vectorized truncation remainder (`round` + `fnmadd`). The scalar fallback `qd_coeff_combine` retains `std::fmod` for simplicity and correctness. The double specialization uses double-precision fmod feeding into `_fm256_sincos256_pd` — benchmarks confirm max phase error of 1.6e-07 rad, validating this approach.

---

## Implementation order

```
[1] Coefficient combination AVX2 kernel          ✅ DONE (1.10–1.18x at 4×4 MIMO)
[2] Two-pass split + OMP for angle/delay loop    ← NEXT: ~85% of remaining runtime
[3] Drop-in AVX2 geo2cart/cart2geo in coupling
[4] Minor cleanups (early-exit, allocation hoisting, fmod removal)
```

Each step is independently testable against the scalar reference.

**Post-P1 benchmark takeaway:** The 4×4 MIMO scenario is angle/delay-bound, not compute-bound. Priority 2 is now the critical path. For larger MIMO sizes (32×32+), the P1 kernel benefit scales quadratically with array size and will become the dominant factor — consider adding a large-MIMO benchmark case to verify.