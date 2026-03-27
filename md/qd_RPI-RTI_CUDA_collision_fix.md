# CUDA ODR Collision Fix Plan
## `ray_triangle_intersect_cuda.cu` and `ray_point_intersect_cuda.cu`

---

## 1. Root Cause

When both `.cu` files are compiled and linked into the same binary, **four categories of symbol collision** occur. The first causes the runtime segfault; the others are latent hazards.

### 1.1 `struct CudaBuffers` — C++ ODR violation (active segfault)

Both files define `struct CudaBuffers` at **global namespace scope** with completely different member layouts:

| | `ray_triangle_intersect_cuda.cu` (RTI) | `ray_point_intersect_cuda.cu` (RPI) |
|---|---|---|
| Scene pointers | 16 individual `float*` fields | 10 individual `float*` fields |
| Ray input | 6 per-stream `float*` arrays | `float* d_ray[2][24]` (2-D array) |
| Key difference | `d_fbs_packed`, `d_sbs_packed`, `d_wq_had_hit`, `d_compact_indices` etc. | `d_ray_ptrs`, `d_seg_*` table fields, `d_hit_ray_idx`, `d_overflow_flag` etc. |

Because the struct is in the global namespace, its destructor is compiled as an **external C++ symbol** (`CudaBuffers::~CudaBuffers()`) in both object files. The C++ One Definition Rule (ODR) requires that all definitions of the same symbol be identical; these are not. The linker cannot detect this — it silently picks one destructor and discards the other.

**Effect:** when `qd_RTI_CUDA()` returns, RAII destroys `CudaBuffers buf` using the **RPI destructor** applied to the **RTI struct memory layout**. The destructor reads member pointer fields from byte offsets that correspond to the RPI layout — getting garbage values — then calls `cudaFree()` and `cudaFreeHost()` on those garbage pointers → **SIGSEGV**.

This is why `qd_RPI_CUDA` does not need to be called to trigger the crash. Merely linking it into the binary installs the wrong destructor.

### 1.2 `aabb_test_and_enqueue` — `__global__` kernel name collision

Both files define `__global__ void aabb_test_and_enqueue(...)` with different parameter lists. In the current non-`-rdc` build this is silently tolerated (each `.cu` file gets its own fatbinary and the kernels never meet). However:
- It makes profiler output (Nsight Compute, `nvprof`) ambiguous — both appear as `aabb_test_and_enqueue`.
- It will cause a **hard `nvlink` error** the moment `-rdc=true` (separable compilation) is enabled.

### 1.3 `enqueue_warp` — `__device__` helper name collision

Both files define `__device__ __forceinline__ void enqueue_warp(...)` with different signatures. Same risk profile as 1.2.

### 1.4 `CUDA_CHECK` macro — duplicate `#define`

Both files define `#define CUDA_CHECK(call) ...` with slightly different error message formats. Harmless while they remain separate translation units, but violates the DRY principle and is one accidental `#include` away from a redefinition conflict.

---

## 2. Fix Overview

Two mechanical changes, applied consistently to both files now and to every future CUDA file added to the project:

1. **Wrap all internal code in an anonymous namespace** — gives internal linkage to the struct, all kernels, and all device helpers.
2. **Prefix all `__global__` and `__device__` symbols with a per-file tag** — eliminates name collisions in profilers and under `-rdc=true`.

A shared header handles the macro duplication.

---

## 3. New File: `cuda_common.hpp`

Create this file in the same directory as the `.cu` files. It contains only the `CUDA_CHECK` macro in its canonical form (use RTI's current format, which includes `__FILE__` and `__LINE__`):

```cpp
// cuda_common.hpp
// Internal header shared by all quadriga-lib CUDA translation units.
// Do NOT include in public headers.
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                        \
    do                                                                          \
    {                                                                           \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess)                                                 \
            throw std::runtime_error(std::string("CUDA error in ") + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + " — " +   \
                                     cudaGetErrorString(err));                  \
    } while (0)
```

Both `.cu` files replace their local `#define CUDA_CHECK` block with `#include "cuda_common.hpp"` and remove the `#include <cuda_runtime.h>` line (it is now covered by the header). All future CUDA files do the same.

---

## 4. Changes to `ray_triangle_intersect_cuda.cu`

### 4.1 Replace the local `CUDA_CHECK` macro

Remove lines 49–57 (`#define CUDA_CHECK ...`) and replace with:
```cpp
#include "cuda_common.hpp"
```
Also remove the now-redundant `#include <cuda_runtime.h>` from the include block (line 25 approximately) since `cuda_common.hpp` includes it.

### 4.2 Open an anonymous namespace after the includes

After all `#include` directives and before the constants block, insert:
```cpp
namespace { // ---------------------------------------------------------------
```

### 4.3 Convert constants from `static constexpr` to plain `constexpr` inside the namespace

The three constants currently at global scope with `static` (lines 36, 40, 43) remain unchanged in value but the `static` keyword becomes redundant inside an anonymous namespace. Leave them as-is or drop `static` — either is correct.

```cpp
// Inside the anonymous namespace — no change needed to values
constexpr int BLOCK_SIZE = 256;
static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32 (warp size)");
constexpr int EST_AVG_HITS = 10;
constexpr uint64_t PACKED_SENTINEL = (uint64_t(0x3F800000u) << 32) | uint64_t(0xFFFFFFFFu);
```

### 4.4 Rename all `__device__` helpers with the `RTI_` prefix

| Current name | New name |
|---|---|
| `pack_hit` | `RTI_pack_hit` |
| `unpack_t` | `RTI_unpack_t` |
| `unpack_face` | `RTI_unpack_face` |
| `enqueue_warp` | `RTI_enqueue_warp` |

Update all call sites inside the kernels accordingly.

### 4.5 Rename all `__global__` kernels with the `RTI_` prefix

| Current name | New name |
|---|---|
| `init_per_ray_state` | `RTI_init_per_ray_state` |
| `aabb_test_and_enqueue` | `RTI_aabb_test_and_enqueue` |
| `moller_trumbore_fbs` | `RTI_moller_trumbore_fbs` |
| `moller_trumbore_sbs` | `RTI_moller_trumbore_sbs` |
| `finalize_outputs` | `RTI_finalize_outputs` |

Update all launch sites (`<<<...>>>` calls) inside `qd_RTI_CUDA` accordingly.

### 4.6 Rename `static inline` host helpers with the `RTI_` prefix

| Current name | New name |
|---|---|
| `ceil_log2` | `RTI_ceil_log2` |
| `upload_float` | `RTI_upload_float` |
| `stage_ray_input` | `RTI_stage_ray_input` |
| `unstage_ray_output` | `RTI_unstage_ray_output` |

Update all call sites inside `qd_RTI_CUDA` accordingly.

### 4.7 Close the anonymous namespace before the public entry point

```cpp
} // end anonymous namespace --------------------------------------------------

// Public entry point — external linkage
void qd_RTI_CUDA(
    const float *Tx, ...
```

`struct CudaBuffers`, all kernels, all device helpers, and all host helpers are now inside the anonymous namespace. Only `qd_RTI_CUDA` remains at external linkage.

---

## 5. Changes to `ray_point_intersect_cuda.cu`

### 5.1 Replace the local `CUDA_CHECK` macro

Remove lines 49–57 (`#define CUDA_CHECK ...`) and replace with:
```cpp
#include "cuda_common.hpp"
```
Remove the now-redundant `#include <cuda_runtime.h>`.

### 5.2 Open an anonymous namespace after the includes

After all `#include` directives and before the constants block, insert:
```cpp
namespace { // ---------------------------------------------------------------
```

### 5.3 Convert `#define` constants to `constexpr` inside the namespace

Remove lines 35–41:
```cpp
// REMOVE these:
#define K1_BLOCK_SIZE 256
#define K1B_BLOCK_SIZE 256
#define K2_BLOCK_SIZE 256
#define RAYS_PER_BLOCK 32
#define EST_AVG_HITS 10
#define EST_HIT_RATE 2
#define N_RAY_ATTRS 24
```

Replace with `constexpr` inside the anonymous namespace:
```cpp
constexpr int K1_BLOCK_SIZE  = 256;
constexpr int K1B_BLOCK_SIZE = 256;
constexpr int K2_BLOCK_SIZE  = 256;
constexpr int RAYS_PER_BLOCK = 32;
constexpr int EST_AVG_HITS   = 10;
constexpr int EST_HIT_RATE   = 2;
constexpr int N_RAY_ATTRS    = 24;
```

The `static_assert` lines that reference these constants remain unchanged.

> **Note:** `N_RAY_ATTRS` is also used as a template-like compile-time size in `__shared__ float s_ray[N_RAY_ATTRS]` inside `point_intersect`. `constexpr` variables are valid in this context — no change needed at those use sites.

### 5.4 Rename all `__device__` helpers with the `RPI_` prefix

| Current name | New name |
|---|---|
| `enqueue_warp` | `RPI_enqueue_warp` |

Update the call site inside `aabb_test_and_enqueue` accordingly.

### 5.5 Rename all `__global__` kernels with the `RPI_` prefix

| Current name | New name |
|---|---|
| `aabb_test_and_enqueue` | `RPI_aabb_test_and_enqueue` |
| `compute_chunks_per_segment` | `RPI_compute_chunks_per_segment` |
| `point_intersect` | `RPI_point_intersect` |

Update all launch sites (`<<<...>>>` calls) inside `qd_RPI_CUDA` accordingly.

### 5.6 Rename the `static` host helper with the `RPI_` prefix

| Current name | New name |
|---|---|
| `stage_ray_batch` | `RPI_stage_ray_batch` |

Update all call sites inside `qd_RPI_CUDA` accordingly.

### 5.7 Close the anonymous namespace before the public entry point

```cpp
} // end anonymous namespace --------------------------------------------------

// Public entry point — external linkage
void qd_RPI_CUDA(
    const float *Px, ...
```

---

## 6. Resulting File Structure (both files)

```
// === includes ===
#include <cstdint>
// ... other system headers ...
#include "cuda_common.hpp"          // ← replaces local CUDA_CHECK + cuda_runtime.h
#include "quadriga_lib_cuda_functions.hpp"

namespace { // ----------------------------------------------------------------

// === constants ===
constexpr int BLOCK_SIZE = 256;     // (RTI) or K1_BLOCK_SIZE etc. (RPI)
// ...

// === struct CudaBuffers ===
struct CudaBuffers { ... ~CudaBuffers() { ... } };

// === __device__ helpers ===
__device__ __forceinline__ void RTI_enqueue_warp(...) { ... }   // RTI
__device__ __forceinline__ void RPI_enqueue_warp(...) { ... }   // RPI

// === __global__ kernels ===
__global__ void RTI_init_per_ray_state(...) { ... }             // RTI
__global__ void RPI_aabb_test_and_enqueue(...) { ... }          // RPI
// ...

// === host helpers ===
static inline void RTI_upload_float(...) { ... }                // RTI
static void RPI_stage_ray_batch(...) { ... }                    // RPI

} // end anonymous namespace --------------------------------------------------

// === public entry point ===
void qd_RTI_CUDA(...) { ... }   // RTI
void qd_RPI_CUDA(...) { ... }   // RPI
```

---

## 7. Convention for Future CUDA Files

Every new `*_cuda.cu` file added to the project must follow these four rules:

1. **`#include "cuda_common.hpp"`** — never define `CUDA_CHECK` locally.
2. **All internal code inside `namespace { }`** — only the public `qd_XXX_CUDA` entry point is outside.
3. **`constexpr` for constants, never `#define`** (except macros that genuinely need to be macros).
4. **All `__global__` and `__device__` symbols prefixed with the file-specific tag** (e.g. `RDI_` for a future `ray_diffraction_cuda.cu`).

These four rules together guarantee that adding a new CUDA function can never silently corrupt an existing one.

---

## 8. Verification

After implementing the fix, confirm with:

```bash
# No duplicate external symbols from the two .cu object files
nm -C ray_triangle_intersect_cuda.o ray_point_intersect_cuda.o \
  | grep " T " | sort | uniq -d
# Expected output: empty (only qd_RTI_CUDA and qd_RPI_CUDA should be T-symbols,
# and they are distinct)

# Valgrind / ASAN smoke test with the 11488-RX case that previously segfaulted
valgrind --error-exitcode=1 qrt_ray_trace ... -cl 1 ...
```

The 11488-RX (CUDA path) case must complete without segfault and produce the same LOS ray count as the 9408-RX (AVX2 path) would for equivalent geometry.
