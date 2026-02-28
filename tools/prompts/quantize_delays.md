Please implement a C++ function `quantize_delays` for the quadriga-lib project. Attached are:

1. **quantize_delays.m** — The reference MATLAB implementation from QuaDRiGa. This is the algorithm to port.
2. **quadriga_lib_integration_manual.md** — The coding conventions, file structure, documentation format, and all project rules you must follow exactly.

The section is "Channel functions", the header is `quadriga_channel.hpp`.

## C++ Header Declaration

```cpp
// Fixes the path delays to a grid of delay bins
// fix_taps options:
//   0 - Use different delays for each tx-rx pair and for each snapshot (default)
//   1 - Use same delays for all antenna pairs and snapshots (least accurate)
//   2 - Use same delays for all antenna pairs, but different delays for the snapshots
//   3 - Use same delays for all snapshots, but different delays for each tx-rx pair
template <typename dtype>
void quantize_delays(const std::vector<arma::Cube<dtype>> *coeff_re,        // Channel coefficients, real part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path]
                     const std::vector<arma::Cube<dtype>> *coeff_im,        // Channel coefficients, imaginary part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path]
                     const std::vector<arma::Cube<dtype>> *delay,           // Path delays in seconds, vector (n_snap) of Cubes of size [n_rx, n_tx, n_path] or [1, 1, n_path]
                     std::vector<arma::Cube<dtype>> *coeff_re_quant,        // Output coefficients, real part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_taps]
                     std::vector<arma::Cube<dtype>> *coeff_im_quant,        // Output coefficients, imaginary part, vector (n_snap) of Cubes of size [n_rx, n_tx, n_taps]
                     std::vector<arma::Cube<dtype>> *delay_quant,           // Output delays in seconds, vector (n_snap) of Cubes of size [n_rx, n_tx, n_taps] or [1, 1, n_taps]
                     dtype tap_spacing = (dtype)5.0e-9,                     // Tap spacing in seconds
                     arma::uword max_no_taps = 48,                          // 0 = unlimited
                     dtype power_exponent = (dtype)1.0,                     // Interpolation exponent (0.5=wideband, 1.0=narrowband)
                     int fix_taps = 0);                                     // 0-3, delay sharing mode
```

## Key Design Decisions

**Power exponent fix (the main reason for this port):**
The MATLAB code hardcodes exponent 0.5, which preserves wideband (incoherent) power but introduces up to +3 dB error in the narrowband (coherent) regime. The new function defaults to 1.0 (linear interpolation), which is correct for the stated narrowband use case. The exponent is parameterized so users can pass 0.5 for wideband scenarios.

With exponent α, each path at fractional offset δ produces two taps with weights `(1-δ)^α` and `δ^α` applied to the complex coefficient.

**No separate phase correction:**
With 2 taps, linear interpolation (α=1.0, same phase on both taps) is provably the optimal narrowband approximation. There are no remaining degrees of freedom for a phase correction. The residual error scales as (bandwidth/sample_rate)² and is inherent to the 2-tap structure. Do not attempt any phase rotation.

**`unsigned` tap indices instead of `uint16_t`:**
The MATLAB code uses uint16, which silently overflows at ~328 μs. Use `unsigned` (32-bit) internally for tap indices. This also eliminates the +1/−1 index offset hack the MATLAB code uses (MATLAB's accumarray can't handle index 0; C++ zero-based arrays can).

**Unified algorithm — eliminate the separate already-quantized code path:**
Detect already-quantized input by checking if all fractional offsets δ are below a threshold (e.g., 0.01). If detected, skip the interpolation weight computation but route through the same `find_optimal_delays` logic as the general case. This eliminates the MATLAB code's separate sort-based branch and the subtle bug where sort operates on complex values directly.

**`find_optimal_delays` as internal helper:**
Port the MATLAB helper function as a file-local (non-exported) C++ function. It accumulates PDP (power delay profile) into a temporary array indexed by tap number, sorts by power, and returns the strongest taps. Same algorithm as MATLAB, same behavior for all fix_taps modes.

**Delay input flexibility:**
The delay input allows `[1, 1, n_path]` cubes (shared delays across antennas, corresponding to `individual_delays = false` in QuaDRiGa). When delays are shared and fix_taps is 0 or 3, expand internally. For fix_taps 1 or 2, work on the compact form directly. If input delays are per-antenna `[n_rx, n_tx, n_path]`, the output delays should also be per-antenna. If input delays are shared `[1, 1, n_path]`, output delays should be shared `[1, 1, n_taps]`.

**fix_taps modes — preserve exact MATLAB behavior:**
- Mode 0: each (rx, tx, snap) combination gets its own optimal delay grid
- Mode 1: single delay grid computed across all data, applied everywhere
- Mode 2: one delay grid per snapshot, shared across antenna pairs
- Mode 3: one delay grid per antenna pair, shared across snapshots

**Output sizing:**
`n_taps` is not known a priori (depends on delay spread, max_no_taps, and the data). Outputs must be allocated inside the function.

**Number of paths (input):**
Unlike in the MATLAB code, do not assume the same number of paths for each snapshot. Each snapshot can have a different number of paths, as indicated by the size of the input cubes. The number of antennas (n_rx, n_tx) is consistent across snapshots (validate this in the function).
In the MATLAB wrapper, coefficients and delays are passed as 4D arrays, in which case the number of paths is determined by the size of the 3th dimension and the number of snapshots by the 4th dimension. In the C++ version, we use vectors of cubes, so each snapshot can have its own number of paths.
Use these wrappers for conversion:
std::vector<arma::cube> vec_of_cubes = qd_mex_matlab2vector_Cube<double>(prhs[0], 3);
plhs[0] = qd_mex_vector2matlab(&out_vec_of_cubes);
Python uses list of 3D arrays, so the same flexibility applies as for C++. Use converters:
`qd_python_list2vector_Cube<T>(pylist)` In: `py::list` of `np.ndarray` Out: `std::vector<arma::Cube<T>>` 
`qd_python_copy2numpy_4d(vecCubes)` Out: `std::vector<arma::Cube<T>>` In: `np.ndarray`


## Implementation Constraints

- **No AVX2 or SIMD intrinsics.** Use plain C++ loops.
- **Use `.memptr()` for raw pointer access** in performance-critical loops instead of Armadillo element access. This is the project convention for tight loops.
- Follow all conventions in the integration manual exactly: namespace, error handling, template instantiation, documentation block format, coding style.
- Use `throw std::invalid_argument(...)` for input validation errors.
- Validate: matching dimensions across coeff_re/coeff_im/delay vectors, tap_spacing > 0, fix_taps in {0,1,2,3}, power_exponent > 0, output pointers not null.
