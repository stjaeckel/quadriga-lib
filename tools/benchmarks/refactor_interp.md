# Refactoring Plan: `qd_arrayant_interpolate_avx2.cpp`

## Problem Statement

The current implementation uses a **flatten-then-process** two-pass design:

1. **Pass 1 (scalar):** A nested `for a in n_ang, for o in n_out` loop materializes *all* per-sample data (angles, 9 rotation-matrix entries, 3 positions, element index = 15 values) into giant SoA flat buffers of size `N_total = n_ang × n_out`, padded to multiples of 8.

2. **Pass 2 (AVX2):** A flat `for iv in 0..N_total/8` loop processes 8 samples at a time from those buffers.

This causes three compounding performance problems when `n_ang > 1`:

| Issue | Detail |
|-------|--------|
| **Scalar bottleneck** | The precompute loop (lines 284–320) is O(N_total) sequential scalar work that runs *before* any SIMD. For S10 (N_total=128k), this alone costs ~1.3 ms vs the scalar reference's 3.6 ms total. |
| **Strided writes** | The SoA layout writes `R_flat[k * N_padded + i]` for k=0..8, touching 9 cache lines per iteration spaced 512 KB apart (for N_padded=128k). This destroys store-buffer efficiency. |
| **Zero-init waste** | `std::make_unique<float[]>(flat_floats)` value-initializes the entire buffer. For S10 that's ~7 MB of zeroing immediately overwritten. |

**Benchmark evidence:**

| Scenario | n_out | n_ang | N_total | Speedup |
|----------|-------|-------|---------|---------|
| S2 | 8 | 1 | 8 | 30.1× |
| S6 | 8 | 1 | 8 | 34.8× |
| S7 | 1 | 500 | 500 | **0.66×** |
| S10 | 64 | 2000 | 128k | **0.75×** |
| S13 | 64 | 2000 | 128k | **0.67×** |

All `n_ang=1` scenarios are excellent (8–35×). All large `n_ang` scenarios are degraded or slower than scalar.

---

## Key Observation: Data Reuse Structure

The benchmark (and typical calling pattern) uses **`per_element_rotation=YES`**, meaning `orientation` is `[3, n_out, 1]`. Examining what varies across the two loop dimensions:

| Data item | Varies with element `o`? | Varies with angle `a`? | Size |
|-----------|--------------------------|------------------------|------|
| Rotation matrix R (9 floats) | YES (`per_element_rotation`) | NO (single slice) | 9 × n_out |
| Element position (3 floats) | YES | NO (constant) | 3 × n_out |
| Element index ie | YES | NO (constant) | n_out |
| Azimuth angle | NO (`per_element_angles=false`) | YES | n_ang |
| Elevation angle | NO | YES | n_ang |
| Pattern gather offset | YES (depends on ie) | NO | n_out |

**The rotation matrix, positions, and element indices are constant across all angles.** They only need to be stored once per element (n_out entries), not replicated n_ang times.

The angles are constant across all elements (in the `per_element_angles=false` case) and only need one sincos evaluation per angle, not per sample.

---

## Refactoring Strategy: Two-Level Loop

Replace the flatten-then-process design with:

```
Precompute per-element data:  O(n_out) — tiny, fits in L1
Outer loop:  for a in 0..n_ang        (parallelizable with OMP)
  Compute angle-dependent data once: sincos(az), sincos(el)  — scalar or broadcast
  Inner loop:  for ov in 0..ceil(n_out/8)   (AVX2, 8 elements at a time)
    Load per-element R, pos, ie from small buffers
    Execute Stages A–G as before
    Store to output[ov*8 .. ov*8+7, a]
```

### Output Memory Layout

Outputs `V_re, V_im, H_re, H_im` etc. are `arma::Mat<dtype>` of size `[n_out, n_ang]`, column-major. Column `a` occupies `p_v_re + a * n_out` through `p_v_re + (a+1) * n_out - 1`. The inner loop over elements writes to **contiguous memory** within each column. This is optimal for cache and store coalescing.

---

## Detailed Changes

### 1. Per-Element Precompute Buffer (replaces the giant flat buffer)

Allocate once, size proportional to `n_out` only (not `n_ang × n_out`):

```cpp
// SoA layout, all float, padded to multiple of 8
const size_t n_out_padded = ((n_out + 7) / 8) * 8;
const size_t n_out_vec = n_out_padded / 8;

// Per-element buffers (constant across angles):
//   R_elem[9 * n_out_padded]    — rotation matrices, SoA
//   pos_elem[3 * n_out_padded]  — element positions, SoA
//   offset_elem[n_out_padded]   — precomputed (ie-1)*n_pattern_samples  (as int32)
const size_t elem_floats = 9 * n_out_padded + 3 * n_out_padded;
auto elem_buf_ptr = std::make_unique<float[]>(elem_floats);
float *elem_buf = elem_buf_ptr.get();
float *R_elem   = elem_buf;
float *pos_elem = elem_buf + 9 * n_out_padded;

auto offset_elem_ptr = std::make_unique<int[]>(n_out_padded);
int *offset_elem = offset_elem_ptr.get();
```

**Portability note:** The new buffer is O(n_out)-sized (~3 KB for n_out=64), so
`std::make_unique<float[]>` (which value-initializes/zeroes) is perfectly fine.
The zero-init overhead was only a problem for the old O(n_ang × n_out) buffer
which we're eliminating entirely. No need for `_mm_malloc`, `std::aligned_alloc`
(unavailable on MSVC), or other platform-specific allocators.
`_mm256_loadu_ps` handles unaligned addresses, so 32-byte alignment is not required.

Fill with a simple O(n_out) loop:

```cpp
for (size_t o = 0; o < n_out; ++o) {
    const dtype *Rp = R_typed.slice_colptr(0, per_element_rotation ? o : 0);
    for (size_t k = 0; k < 9; ++k)
        R_elem[k * n_out_padded + o] = (float)Rp[k];

    pos_elem[0 * n_out_padded + o] = (float)p_element_pos[3 * o + 0];
    pos_elem[1 * n_out_padded + o] = (float)p_element_pos[3 * o + 1];
    pos_elem[2 * n_out_padded + o] = (float)p_element_pos[3 * o + 2];

    offset_elem[o] = (int)(p_i_element[o] - 1) * (int)n_pattern_samples;
}
// Pad [n_out..n_out_padded) with last valid entry (same pattern as before)
```

For S10 (n_out=64): this buffer is `(9+3) × 64 × 4 = 3 KB`. Fits entirely in L1.

### 2. Grid Arrays — Keep As-Is

The `az_grid_f`, `el_grid_f`, `az_step_inv`, `el_step_inv` arrays and the uniform-grid detection logic are all fine. They're small (proportional to grid size, not N_total) and only computed once. Keep them.

### 3. Outer Angle Loop

```cpp
#pragma omp parallel for schedule(static) if (n_ang >= OMP_ANG_THRESHOLD)
for (size_t a = 0; a < n_ang; ++a)
{
    // --- Angle-dependent data (computed once per angle) ---
    // Output base pointers for this angle's column
    dtype *out_vr = p_v_re + a * n_out;
    dtype *out_vi = p_v_im + a * n_out;
    dtype *out_hr = p_h_re + a * n_out;
    dtype *out_hi = p_h_im + a * n_out;
    dtype *out_dist = p_dist ? p_dist + a * n_out : nullptr;
    dtype *out_az   = p_azimuth_loc ? p_azimuth_loc + a * n_out : nullptr;
    dtype *out_el   = p_elevation_loc ? p_elevation_loc + a * n_out : nullptr;
    dtype *out_gam  = p_gamma ? p_gamma + a * n_out : nullptr;

    // Load input angles for this angle index
    // (handling per_element_angles case — see Section 5)
    float az_a, el_a;  // scalar for broadcast case
    if (!per_element_angles) {
        az_a = (float)p_az_global[a];
        el_a = (float)p_el_global[a];
    }

    // For per_angle_rotation: load R for this angle
    // (see Section 5 for the rare per_angle_rotation case)

    // --- Inner element loop (AVX2) ---
    for (size_t ov = 0; ov < n_out_vec; ++ov)
    {
        const size_t o_base = ov * 8;
        // ... Stages A–G, see Section 4 ...
    }
}
```

**OMP threshold:** Parallelize the outer angle loop. Suggested threshold: `n_ang >= 4` or `n_ang * n_out >= 512` (tune to taste). This replaces the current flat-loop OMP.

### 4. Inner Element Loop — Stages A–G (modifications per stage)

The AVX2 kernel body (Stages A–G) stays almost identical. The key differences:

#### Stage A: Input Angles → Cartesian

**Before:** Loaded from `az_flat[i_base]`, `el_flat[i_base]`.

**After (common case, `!per_element_angles`):** Broadcast scalar angle:
```cpp
__m256 az_in8 = _mm256_set1_ps(az_a);
__m256 el_in8 = _mm256_set1_ps(el_a);
```

The sincos can even be hoisted out of the inner loop entirely (compute once per angle as scalar, broadcast results):
```cpp
// Before inner loop:
// NOTE: sincosf() is a GNU extension, not available on MSVC.
// Use std::sin / std::cos for portability.
float sAZi_s = std::sin(az_a), cAZi_s = std::cos(az_a);
float sELi_s = std::sin(el_a), cELi_s = std::cos(el_a);
cELi_s += eps;  // pole guard

__m256 sAZi8 = _mm256_set1_ps(sAZi_s);
__m256 cAZi8 = _mm256_set1_ps(cAZi_s);
__m256 sELi8 = _mm256_set1_ps(sELi_s);
__m256 cELi8 = _mm256_set1_ps(cELi_s);
__m256 Cx8 = _mm256_set1_ps(cELi_s * cAZi_s);
__m256 Cy8 = _mm256_set1_ps(cELi_s * sAZi_s);
// Also precompute the input basis vectors:
__m256 eTHi_x8 = _mm256_set1_ps(sELi_s * cAZi_s);
__m256 eTHi_y8 = _mm256_set1_ps(sELi_s * sAZi_s);
__m256 eTHi_z8 = _mm256_set1_ps(-cELi_s);
__m256 ePHi_x8 = _mm256_set1_ps(-sAZi_s);
__m256 ePHi_y8 = _mm256_set1_ps(cAZi_s);
```

This eliminates 2 `_fm256_sincos256_ps` calls **per 8-element vector iteration** in the inner loop.

#### Stage B: Rotation + cart2geo

**Before:** Loaded R from `R_flat[k * N_padded + i_base]`.

**After:** Load from `R_elem[k * n_out_padded + o_base]`:
```cpp
__m256 Rm0_8 = _mm256_loadu_ps(&R_elem[0 * n_out_padded + o_base]);
// ... same pattern, just different base pointer and stride
```

The matrix-vector multiply, cart2geo, rsqrt+NR, and basis vector computation remain identical.

#### Stage C: Basis Vectors + Gamma

Input basis vectors (`eTHi`, `ePHi`) are now precomputed broadcasts (see Stage A above). The output basis vectors and gamma dot products remain identical.

#### Stage D: Distance

**Before:** Loaded positions from `pos_flat[k * N_padded + i_base]`.

**After:** Load from `pos_elem[k * n_out_padded + o_base]`:
```cpp
__m256 px8 = _mm256_loadu_ps(&pos_elem[0 * n_out_padded + o_base]);
```

Rest unchanged.

#### Stage E: Grid Search

Completely unchanged. The grid search operates on `az8`/`el8` which come from Stage B output (the rotated local angles). These vary per element even when input angles are broadcast, because R differs per element.

#### Stage F: Gather Index Computation

**Before:** Loaded `ie_flat[i_base]` and computed offset.

**After:** Load from `offset_elem` (precomputed `(ie-1)*n_pattern_samples`):
```cpp
__m256i offset8 = _mm256_loadu_si256((__m256i *)&offset_elem[o_base]);
```

This saves the per-iteration `mullo_epi32(sub(ie,1), n_pat)`.

The rest of Stage F (row_up, row_un, iA–iD) and the pattern gather remain identical.

#### Stage G: SLERP + Polarization Rotation + Stores

Unchanged, except the store target pointers are now `out_vr + o_base` etc.:
```cpp
_mm256_storeu_ps((float *)out_vr + o_base, v_re8);   // float path
// or for double:
STORE_F2D(out_vr, v_re8, o_base);
```

#### Tail Masking

The tail mask is now based on `n_out % 8` (not `N_total % 8`). Compute once before the angle loop:
```cpp
const int tail = (int)(n_out % 8);
// ... same mask construction as before
```

The last iteration of the inner loop (`ov == n_out_vec - 1 && tail != 0`) uses masked stores.

### 5. Handling All Orientation/Angle Modes

The function supports 4 orientation modes × 2 angle modes = 8 combinations. The benchmark (and dominant real-world case) is `per_element_rotation=YES, per_angle_rotation=NO, per_element_angles=NO`. The refactoring should be optimized for this case, with correct handling of the others.

#### Mode Matrix

| `per_element_rotation` | `per_angle_rotation` | `per_element_angles` | R source | Angle source |
|:-:|:-:|:-:|---|---|
| NO | NO | NO | Broadcast single R | Broadcast per angle |
| YES | NO | NO | **Load from R_elem (per-element, angle-invariant)** | **Broadcast per angle** |
| NO | YES | NO | Load/compute R per angle (broadcast across elements) | Broadcast per angle |
| YES | YES | NO | Full [n_out, n_ang] — need per-sample R | Broadcast per angle |
| NO | NO | YES | Broadcast single R | Load per-element-angle from matrix |
| YES | NO | YES | Load from R_elem | Load per-element-angle |
| NO | YES | YES | Per-angle R broadcast | Load per-element-angle |
| YES | YES | YES | Full per-sample R | Load per-element-angle |

**Recommended approach:**

- **Primary fast path** (row 2 in table above, the benchmark case): Use the two-level loop with hoisted angle broadcasts as described. This is the performance-critical path.

- **`per_angle_rotation=YES`**: In the outer angle loop, index `R_typed.slice_colptr(a, ...)` to get the R for this angle. If also `per_element_rotation`, each element has its own R that changes every angle — fill a small temporary `R_this_angle[9 * n_out_padded]` per angle, or just read from R_typed directly. This case is rare and N_total-dominated anyway, so moderate overhead is fine.

- **`per_element_angles=YES`**: Instead of broadcasting, load from `p_az_global[a * n_out + o_base]` in the inner loop. The sincos can't be hoisted, so use `_fm256_sincos256_ps` as before. Still much better than the flat precompute because no giant buffer.

**Implementation suggestion:** Use `if constexpr`-like branching or separate code paths for the dominant fast path vs. the general path. A practical approach:

```cpp
if (!per_angle_rotation && !per_element_angles) {
    // FAST PATH: two-level loop with angle hoisting
    // ... (bulk of the new code)
} else {
    // GENERAL PATH: either keep the current flat-loop approach
    // (acceptable — these modes are rare) or do a two-level loop
    // without angle hoisting
}
```

### 6. OMP Strategy

**Current:** Single flat loop, parallelized when `n_vec >= 64`.

**New:** Parallelize the **outer angle loop**:

```cpp
#pragma omp parallel for schedule(dynamic, 1) if (n_ang >= 4 && n_ang * n_out >= 512)
for (size_t a = 0; a < n_ang; ++a)
```

Use `dynamic` scheduling because per-angle work is uniform but we want good load balancing with small n_ang. If `n_ang` is very small but `n_out` is huge, consider alternatively parallelizing the inner loop — but this is a rare edge case (n_ang=1 is already fast).

**Edge case n_ang=1:** Falls back to sequential, which is fine — these cases are already 8–35× speedup.

### 7. Memory Allocation — Portability Note

The old code uses `std::make_unique<float[]>(flat_floats)` for the giant flat buffer.
The new per-element buffer is tiny (O(n_out)), so `std::make_unique<float[]>` is the
right choice — portable across GCC and MSVC, RAII-safe, and the zero-init cost on
~3 KB is negligible.

**Do NOT use** `std::aligned_alloc` (not available on MSVC) or `_mm_malloc`/`_mm_free`
(would work on both via `<immintrin.h>`, but adds unnecessary complexity and a custom
deleter). Alignment is not needed since all SIMD loads use `_mm256_loadu_ps` (unaligned).

The grid arrays (`az_grid_f`, `el_grid_f`, `az_step_inv`, `el_step_inv`) are also small.
Use `std::make_unique<float[]>` for those too, matching the existing code style.

---

## What NOT to Change

- **`qd_rotation_matrix()`** — keep as-is, it's called once.
- **Stages B–G AVX2 kernel logic** — the intrinsics stay the same, only the load/store addresses change.
- **Grid search logic** (uniform, binary search, wrap handling) — keep identical.
- **Pattern gather + SLERP** — keep identical.
- **Float-only internal computation** with dtype-dependent load/store — keep this design.
- **Function signature** — unchanged.
- **Tail masking approach** — same pattern, just based on `n_out % 8` instead of `N_total % 8`.

---

## Summary of Expected Performance Impact

| Scenario | Current | Expected | Why |
|----------|---------|----------|-----|
| S1/S2/S6 (n_ang=1) | 8–35× | ~same | Inner loop identical, tiny precompute either way |
| S7 (n_out=1, n_ang=500) | 0.66× | **5–8×** | No precompute overhead; sincos hoisted; n_out=1 means 1 vector per angle |
| S10 (n_out=64, n_ang=2000) | 0.75× | **6–10×** | No 7MB buffer; angle hoisting saves 2 sincos/vector; OMP on angle loop |
| S12 (n_out=64, n_ang=500, 1deg) | 1.92× | **5–8×** | Same benefits, pattern gather still dominates |
| S13 (n_out=64, n_ang=2000, 1deg) | 0.67× | **5–8×** | Precompute was the dominant cost |

---

## File Inventory (what to edit)

Only one file changes: **`qd_arrayant_interpolate_avx2.cpp`**

No changes needed to:
- `qd_arrayant_interpolate_avx2.hpp` (function signature unchanged)
- `fastmath_avx2.h` (all helpers reused as-is)
- Benchmark harness (same interface)
- Tests (same interface, same expected results)

---

## Suggested Implementation Order

1. **Build the per-element precompute** (R_elem, pos_elem, offset_elem) — small O(n_out) loop, using `std::make_unique<float[]>` and `std::make_unique<int[]>`.
2. **Rewrite the grid arrays** to use standalone `std::make_unique<float[]>` allocations instead of the big flat buffer.
3. **Write the fast-path two-level loop** for `!per_angle_rotation && !per_element_angles`:
   - Outer angle loop with OMP.
   - Angle hoisting (scalar `std::sin`/`std::cos` + broadcast — do NOT use `sincosf`, it's a GNU extension unavailable on MSVC).
   - Inner element loop: paste Stages B–G with updated load addresses.
   - Tail masking on `n_out % 8`.
4. **Wire the general path** — for `per_angle_rotation || per_element_angles`, either keep a variant of the old flat approach (with the zero-init fix) or implement a two-level loop without angle hoisting. Both are acceptable since these modes are rare.
5. **Remove the old flat buffer** allocation and precompute loop.
6. **Test:** Run the existing benchmark. Verify accuracy unchanged, speedup improved for S7/S10/S13.