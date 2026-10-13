# Review of the sum-of-sinusoids implementation (qd_sos)

Reviewed on 2026-10-12, branch `qd-sos` at commit 38d871a. No repository file was changed; this report is
the only file added. Scratch programs, builds, logs and benchmark data are in the session scratch space
(`/tmp/claude-1005/-sjc-quadriga-lib-sos/7c27bc47-99d4-47d2-8a63-1e03cfe2efc8/scratchpad/`).

## Summary

- Formulas: no error found in any of the 14 functions, the three device modes or the Uniform mapping.
- Ten defects reproduced, all at the edges: MEX `sos_randi` returns garbage for a negative `imax`, Python
  turns `None` for `pos` into NaN; the rest is validation, aliasing and allocation failure.
- Trap, not a code defect: separate `sos_randn` calls with one seed give fields that are identical at the
  origin and correlated around it, whatever the decorrelation distance or preset.
- Tests: all but 3 statements run. Not pinned: non-finite inputs, unequal spacing, the shipped table values.
- Performance: the AVX2 kernel scales to 16 cores (13.6 times); the serial code around it in `sos_val.cpp`
  is the limit. Removing it gives 2.3 to 9 times at 16 threads, a reworked accumulation another 1.55 to 2.0
  times, both bit-identical. `sos_remove_bias` is serial and 190 times slower than `sos_val`.
- Documentation: rules met. Three statements fail at the edges; the introductions use undefined variables.

Marks used below: **[M]** measured in this session, **[R]** reproduced with a scratch program or script,
**[C]** read from the code and not executed.


## 0. Baseline

Builds, all out of tree in the scratch space. `build_linux/` was in use by another process
(`build_linux/scratch/i39/suite_C`) and was not touched.

| Build | Compiler | AVX2 | Suite | Result |
|---|---|---|---|---|
| `b_avx2` | gcc 15.2 | on | Catch2, complete | 1008 test cases, 1,192,905 assertions, all passed [M] |
| `b_avx2` | gcc 15.2 | on | Catch2, `sos_*` | 78 test cases, 35,608 assertions, all passed [M] |
| `b_gcc12` | gcc 12 (the compiler `make` selects) | on | Catch2, `sos_*` | 78 test cases, all passed; 10.0 s on an idle machine [M] |
| `b_noavx` | gcc 15.2 | off | Catch2, `sos_*` | 78 test cases, 35,528 assertions, all passed [M] |
| `b_asan` | gcc 15.2, ASan + UBSan | on | Catch2, `sos_*` | all passed, no sanitizer report [M] |
| `b_cov` | gcc 15.2, `--coverage` | on | Catch2, `sos_*` | all passed; coverage in section 2 [M] |
| `b_gcc12` | gcc 12 | on | MEX, complete, MATLAB R2024a | 61 of 62 passed; `test_ray_mesh_interact` line 113 fails as known [M] |
| `b_gcc12_noavx` | gcc 12 | off | MEX, `test_sos_*` | 14 of 14 passed [M] |
| `b_avx2` | gcc 15.2 | on | Python, complete | 710 passed, 1 skipped; the 87 SOS tests passed [M] |
| `b_noavx` | gcc 15.2 | off | Python, SOS | 87 passed [M] |

- The gcc 15 MEX files cannot be loaded by MATLAB R2024a (`GLIBCXX_3.4.32` not found), hence the gcc 12 builds.
- Both gcc 12 and gcc 15 compile the library without warnings.
- Octave was not run.


## 1. Bugs

### 1.1 Confirmed defects

Severity: **medium** = silently wrong result for an input a user can plausibly give; **low** = needs an
unusual or invalid input, or the effect is a wrong exception type.

| # | Severity | Where | Defect |
|---|---|---|---|
| B1 | medium | `api_mex/sos_randi.cpp:76` | Negative or NaN `imax` returns values around 1e18 instead of an error |
| B2 | medium | `api_python/qpy_sos_*.cpp`, mandatory position arrays | `None` is accepted and evaluated as one position at NaN |
| B3 | low | `src/quadriga_lib_helper_functions.hpp:600` | `qd_sos_equally_spaced` rejects equally spaced `float` tables with more than about 2000 samples |
| B4 | low | `src/sos_acfi.cpp:57` | Output that is the same object as `acf` or `dist`: out-of-bounds read |
| B5 | low | `src/sos_val_avx2.cpp:39`, `src/sos_val_generic.cpp:25` | Allocation failure inside the parallel region: segmentation fault (AVX2) or `std::terminate` (GENERIC) |
| B6 | low | `src/quadriga_lib_helper_functions.hpp:581`, `src/sos_acf_approx.cpp:63` | Non-finite sample points are accepted and give NaN or a `std::length_error` |
| B7 | low | `src/sos_val.cpp:382`, `src/sos_remove_bias.cpp:211` | An ACF value of -1 or below gives Inf or NaN in the same-map mode |
| B8 | low | `src/sos_generate.cpp:179` | Memory and run time grow without limit with `freq_max · max(dist)` |
| B9 | low | `src/sos_remove_bias.cpp:221` | `phase_b` being the same object as `phase` leaves the first device with a non-zero mean |
| B10 | low | `api_mex/sos_*.cpp`, scalar arguments | Negative counts end in allocator errors instead of a clear message |

**B1. MEX `sos_randi`: negative or NaN `imax`** [R]

- Input: `quadriga_lib.sos_randi(10, pos, -3, [], [], [], 1)` with 7 positions.
- Expected: an error ("Input 'imax' ..."), as for `imax = 0`.
- Actual: no error; values `2.034e+18, 3.494e+18, 6.838e+18, ...`. `imax = NaN` does the same.
  `imax = 2.9` is truncated to 2 without notice.
- Cause: `qd_mex_get_scalar<arma::uword>` casts the double to an unsigned type; the C++ check `imax == 0`
  cannot see the wrap-around.
- Fix: read `imax` as `double` in the wrapper and reject values that are not finite or below 1 before the
  cast. Better, do this once in `qd_mex_get_scalar` for unsigned target types, which also fixes B10.

**B2. Python: `None` for a mandatory array** [R]

- Input: `quadriga_lib.sos.val(freq, phase, None)`.
- Expected: `TypeError` or `ValueError`.
- Actual: returns `[[nan nan nan]]`. Same for `sos.randn([10], None)` (returns `[[nan]]`),
  `sos.map(freq, phase, None)` and `sos.remove_bias(freq, phase, None)` (offsets 0, `mean_before` NaN).
- Cause: `const py::array_t<double> &` converts `None` to a 0-dimensional array holding NaN. The other
  mandatory arrays happen to be caught by later size checks (`freq`, `phase`, `acf`, `dist_decorr`).
- Fix: `py::arg("pos").none(false)` (and for `x`, `d`) in `python_sos.cpp`, or an explicit check in the
  wrapper. The pattern comes from `SPEC_PYTHON_API.md`, so other wrappers of the library probably share it;
  that was not checked.

**B3. Equal-spacing test fails for long `float` tables** [R]

- Input: `arma::fvec dist(5001)` with `dist(i) = (float)(0.01 · i)`, any `acf`;
  `sos_acf_2d<float>(acf, dist, out)` or `sos_calc_mse<float>(freq, acf, dist)`.
- Expected: accepted. The same table in `double` is accepted.
- Actual: `std::invalid_argument("Input 'dist' must be equally spaced.")`. 2001 samples pass, 5001 and
  20001 fail.
- Cause: the tolerance is `1e-4 · step`, but a `float` near 50 has a spacing of 3.8e-6, which is more than
  `1e-4 · 0.01`. In `sos_generate<float>` the same test silently selects the slow cosine path instead of
  failing [C].
- Fix: `tol = max(1e-4 · step, 4 · eps(dtype) · dist.back())`. The threshold is sound for `double`.

**B4. `sos_acfi`: output aliasing an input** [R, AddressSanitizer]

- Input: `acf`, `dist` with 40 elements, `d` with 4000 elements, call `sos_acfi<double>(acf, dist, d, acf)`.
- Expected: a correct result, or a documented restriction.
- Actual: heap-buffer-overflow, READ of size 8 in `qd_sos_acf_interp`
  (`quadriga_lib_helper_functions.hpp:562`), called from `sos_acfi.cpp:61`. Without the sanitizer the
  call returns garbage.
- Cause: `val.set_size()` at line 57 runs before the inputs are copied at line 59, so the table is resized
  while `dist` keeps its length.
- Fix: move the two `conv_to` lines above the `set_size`. Only `sos_val` documents "must not be the same
  object as an input"; the other functions should either say the same or be safe.

**B5. Allocation failure inside the kernels** [R]

- Input: `freq` with 4,000,000 rows and 1 column, 64 positions, 32 threads, address space limited with
  `ulimit -v 3000000`.
- Expected: `std::bad_alloc` that the caller can catch.
- Actual: `use_kernel = 2`: segmentation fault (exit 139). `use_kernel = 1`: "terminate called after
  throwing an instance of 'std::bad_alloc'" (exit 134). Without the limit the call succeeds.
- Cause: `_mm_malloc` is not checked; `std::vector<double> c(n_coeff), s(n_coeff)` is constructed inside
  `#pragma omp parallel`. `SPEC_CPP_API.md`, "OpenMP": exceptions must not leave a parallel region.
- Fix: allocate the per-thread buffers before the region (one block of `n_threads` slices), or check and
  record the failure with the `OMP_SAFE_CALL` pattern and rethrow after the region.

**B6. Non-finite sample points** [R]

- `sos_acf_approx(freq, {0, NaN, 2}, out, &axis)`: accepted, axis `-4 -3 -2 ...`, i.e. evaluated as if
  `dist = {0, 1, 2}`. Expected: rejected. The NaN passes because `abs(NaN - x) > tol` is false.
- `dist = {0, 1, Inf}`: accepted by `sos_acf_2d`, `sos_acf_approx`, `sos_calc_mse` and `sos_acfi`.
  Results: axis `-inf ... nan`, `mse = nan`. `sos_generate` with this table and `freq_max = 1` throws
  `std::length_error` ("cannot create std::vector larger than max_size()") instead of
  `std::invalid_argument`.
- `sos_acf_estimate` with `bins = {1, 2, Inf}`: accepted, all outputs NaN.
- Fix: require finite values in `qd_sos_check_acf` (for `dist`), in `sos_acf_approx` and for `bins`.

**B7. ACF values of -1 or below** [R]

- Input: `acf = {1, -1, -1}`, `dist = {0, 1, 2}`, two devices 1 m and 1.5 m apart on the same map.
- Expected: rejected, or documented.
- Actual: `val = inf` (division by `sqrt(2 + 2·A) = 0`); for `A < -1` NaN.
- Fix: require `acf > -1` in the same-map mode, or clamp the denominator.

**B8. `sos_generate`: unbounded search grid** [R]

- Input: 200-sample table to 49.75 m, 4 sinusoids, 1-D, `freq_max` = 0.3, 3, 30, 300.
- Actual: 0.004 s / 0.2 MB, 0.03 s / 2.4 MB, 0.32 s / 24 MB, 3.3 s / 239 MB. `freq_max = 1e6` would ask
  for 800 GB.
- Cause: the coarse grid has `(freq_max - freq_min) · max(dist) / 0.1` points and its cosines are stored
  for all sample points.
- Fix: evaluate the grid cosines on the fly (the recurrence used in `cos_series` already exists), or limit
  the number of grid points and throw `std::invalid_argument`.

**B9. `sos_remove_bias`: `phase_b` is the same object as `phase`** [R]

- Input: `sos_remove_bias(freq, ph, pos_a, &pos_b, &ph, ...)`, 50 positions per device.
- Expected: both devices zero-mean, or an error. `sos_val` documents that the same matrix for both
  arguments selects the two-map formula.
- Actual: mean of device a = -0.557, mean of device b = -1.3e-16. The second correction overwrites the first.
- Fix: throw if `phase_b == &phase`, or document it.

**B10. MEX: negative counts** [R]

- `sos_init_phases(-1)`: "Requested array exceeds the maximum possible variable size."
- `sos_generate(acf, dist, -1, 1)`, `sos_calc_mse(freq, acf, dist, -1)`: "cannot create std::vector larger
  than max_size()". `sos_acf_estimate(..., -1)`: "Mat::init(): requested size is too large".
- No crash, but the message does not name the argument. Same cause and fix as B1.

### 1.2 Behaviour that follows the documentation but is a trap

- **Same seed in separate calls gives related fields.** [R] `sos.randn([10], pos, seed=1)` and
  `sos.randn([50], pos, seed=1)` return the same value at the origin (2.2031 both) and are correlated around
  it: correlation over 400 seeds 1.000 at 0 m, 0.998 at 0.54 m, 0.938 at 3.2 m. The 50 m field is the 10 m
  field stretched by 5 (difference 0.0). The same holds between `Comb256` and `Exp256`. Cause: generator `g`
  gets the same phases for a given seed whatever its frequencies. Two entries of one call are independent
  (correlation -0.007). A user who draws shadow fading and delay spread in two calls with one seed gets maps
  that coincide near the coordinate origin. The documentation does not warn about this.
- **Two-map values depend on the number of generators.** [R] `sos_randn` with `dist_decorr_b` different
  from `dist_decorr`: generator 0 of a call with 1 entry and of a call with 2 entries differs by 1.54,
  although the first entry is the same. One device: difference 0. The second device uses phase column
  `n_gen + g`, as documented, so appending a generator changes all two-map fields
  (`src/sos_rand.cpp:57` and `:91`). Drawing the second device's phases from a second stream (for example
  seeded from the first) would make column `g` independent of `n_gen` for both devices.
- **Huge coordinates.** [R] Both kernels lose the argument as `f·x` grows (see D1 in section 4). Beyond
  2^53 periods along one axis every position returns the value at the origin: `pos = {1e300, 0, 0}` gives
  2.20306, which is `s(0)` of that generator. A decorrelation distance of 1e-300 m gives one
  constant per generator at all positions; 1e-320 m gives NaN.
- **MEX type handling** (library helpers, not specific to this port) [R]: complex input arrays are
  accepted and the imaginary part is dropped without a warning; non-integer scalars are truncated
  (`use_kernel = 1.7`, seed 1.9); a vector for a scalar argument uses its first element; `uint8` and
  `logical` arrays and MATLAB strings (`"Comb256"`) are rejected.
- **Python array handling** [R]: a 1-D array for `pos` is a column, so `[x, y, z]` is three positions on the
  x-axis and not one position; a 3-D array is flattened to 2-D; complex input raises numpy's
  `ComplexWarning` and continues.

### 1.3 Checked and found correct

Formulas
- One device, both kernels, against a scalar transcription; `n_coeff = 1` and `n_pos = 1` [R].
- Two devices on one map and on two maps: read against the header, and the existing brute-force tests
  cover both kernels. Positions of the two devices with different numbers of columns give exactly the result
  of zero-padded positions in `sos_val` and `sos_remove_bias` (difference 0.0) [R].
- Uniform mapping, `sos_randi` mapping and clamping [C].
- `sos_remove_bias`: derivation of `Z`, both roots, the choice of the smaller offset, the weights of the
  same-map mode [C]; mean after the call 3.7e-16 [R].
- `sos_acf_2d`, `sos_acf_approx` (plane and row/column assignment), `sos_calc_mse` (core mask, 1-D branch,
  tail), `sos_acf_estimate` (bin limits, pair selection) [C].
- `sos_generate`: the residual, the build-up pass (same rule as the MATLAB original), the acceptance test,
  the unit conversion [C]. Reported error against an independent evaluation: equal to 12 digits for unequal
  sample spacing, relative difference 1.1e-16 after a 128-sinusoid run (no drift of the incremental sums)
  [R]. Build-up start reaches 30.99 dB where the random start reaches 31.12 dB [M].

Kernels
- Lane independence, padded last block, chunk boundaries: read; the existing 210-placement test is
  bit-exact. Arguments beyond 2^31 periods: no cliff, the reduction uses `_mm256_round_pd`, which has no
  integer range limit; the error grows smoothly with the rounding of the double product [R].
- NaN and Inf positions or frequencies give NaN in both kernels, no crash, neighbours unaffected [R].
- Alignment: the buffers are 32-byte aligned and accessed with aligned loads; inputs use unaligned loads [C].
- The AVX2 object file exports no weak template symbols with AVX code (the `std::vector<float>` is inlined),
  so nothing leaks into the non-AVX2 code path with gcc [M, `nm`/`objdump`]. This depends on inlining; other
  compilers were not checked.
- Requesting the AVX2 kernel in a build without it throws `std::invalid_argument` from `sos_val` and
  `sos_map`; the Python wrapper turns it into `ValueError` [R].

OpenMP
- No shared writes: each thread writes disjoint output elements and owns its buffers [C].
- Loop variables are `long long`; only OpenMP 2.0 constructs are used [C]. MSVC was not compiled.
- Results are independent of the thread count by construction (no reduction across positions) [C].

Random stream
- `seed >= 0`, `seed = -1`, `seed < -1` rejected in every function that takes a seed [C].
- The phases depend only on `std::mt19937_64` and IEEE arithmetic, so the claim "same phases on all
  platforms" holds for `sos_init_phases` [C].

Outputs
- Every output is sized after validation; a failed validation leaves outputs untouched in all 14
  functions [C].
- `sos_val` with `val` aliasing an input works today because all inputs are copied first [C].

Thresholds
- `sos_remove_bias`, `abs(Z) < 1e-14 · n_coeff`: reached with one sinusoid and two positions half a
  wavelength apart; offsets 0, phases untouched [R]. The threshold is of the size of the rounding error
  of `Z` (about `sqrt(n_coeff · n_pos) · 1e-16`); a generator just above it gets an offset computed from
  noise, which changes the realisation but not the mean. Harmless.

Wrappers
- Argument order and defaults of all 14 MEX and Python wrappers match the "MATLAB:" lines of the header
  [C]. `n_gen` has the default 1 in both wrappers and no default in C++.
- `[]` and `None` for optional arrays, empty mandatory arrays (rejected by the library), in/out phases of
  `sos_remove_bias` (the caller's array is unchanged in MATLAB for `double` and `single`, in Python for C
  and Fortran order) [R].


## 2. Test coverage

### 2.1 Measured coverage

gcov, gcc 15.2, `-O3 --coverage`, Catch2 `sos_*` cases only [M].

| File | Lines executed | Branches taken at least once |
|---|---|---|
| `sos_acf_2d.cpp` | 98.6 % | 73.1 % |
| `sos_acf_approx.cpp` | 93.2 % | 62.5 % |
| `sos_acf_estimate.cpp` | 88.8 % | 53.7 % |
| `sos_acfi.cpp` | 97.0 % | 67.9 % |
| `sos_calc_mse.cpp` | 92.5 % | 61.3 % |
| `sos_generate.cpp` | 97.3 % | 59.3 % |
| `sos_init_phases.cpp` | 92.3 % | 55.0 % |
| `sos_map.cpp` | 91.9 % | 63.4 % |
| `sos_preset.cpp` | 92.8 % | 55.3 % |
| `sos_rand.cpp` | 89.8 % | 53.9 % |
| `sos_remove_bias.cpp` | 85.1 % | 47.7 % |
| `sos_val.cpp` | 91.9 % | 61.4 % |

- The percentages count the `float` and the `double` instantiation separately. Branch numbers include the
  exception edges the compiler adds and are optimised code; they are an indication, not a measure.
- Statements never executed in either instantiation: three.
  - `sos_generate.cpp:157-158`, the cosine path for unequal sample spacing.
  - `sos_remove_bias.cpp:119`, "pos_b cannot have more columns than freq".
  - `sos_val.cpp:339`, "AVX2 kernel requested but not available" (and line 71 in a build without AVX2).
- Executed only in `double`: all validation errors of `sos_acf_approx`, `sos_acf_estimate`, `sos_calc_mse`,
  `sos_generate`, `sos_init_phases`, `sos_map`, `sos_preset`; `acf_est_dual`; the 1-D branches of
  `sos_acf_approx` and `sos_calc_mse`; `test_dir`; the z-axis of `sos_map`. The code is shared, so this is
  acceptable.
- Decision outcomes never taken:
  - `sos_remove_bias.cpp:182`: the "left unchanged" side (0 of 4051 evaluations).
  - `!std::isfinite(...)` in `sos_val.cpp:321`, `sos_remove_bias.cpp:127`, `sos_generate.cpp:117/119`,
    `sos_rand.cpp:25/45`, `sos_preset.cpp:1563`, and the finite check of `acf` in
    `quadriga_lib_helper_functions.hpp:594`.
  - `sos_rand.cpp:115`: both clamps of `sos_randi` (they cannot be reached with `u` in (0, 1]).
  - `sos_acf_estimate.cpp:131`: a pair closer than 1e-5 m.
  - `sos_generate.cpp:196`: `freq_min` at or above the random start range.
  - `sos_remove_bias.cpp:207`: a device with fewer position columns in the same-map mode.

### 2.2 Gaps per function, with the test to add

| Function | No test for | Test to add |
|---|---|---|
| `sos_val` | NaN, Inf and huge coordinates | One case: NaN/Inf rows give NaN in that row only; values at 1e9 periods still within the AVX2 bound |
| | AVX2 bound for the largest preset | Kernel parity with `Comb1024` (measured here: 5.6e-7 against a bound of 1.8e-5) |
| | `pos` and `pos_b` with different column counts in the two-device modes | Compare with zero-padded positions, both modes |
| | Unavailable kernel | `if (!has_avx2()) CHECK_THROWS_AS(..., 2)`; runs in the build without AVX2 |
| | `freq_scale_b` NaN or Inf; ACF value <= -1 (B7) | Sections in the validation case |
| | Thread independence in the two-device modes | Extend the existing case by one two-device call |
| | `float` computed in double | Compare `float` output with the `double` call on the rounded inputs, tolerance 2e-6 instead of 1e-3 |
| `sos_map` | Nothing significant | - |
| `sos_randn`, `sos_rand`, `sos_randi` | Two-map value of generator `g` against `n_gen` | Decide the intended behaviour first (1.2), then pin it |
| | Non-finite `dist_decorr` | Validation section |
| | Kernel dependence of `sos_randi` | Not testable across machines; document instead |
| `sos_remove_bias` | "left unchanged" below the threshold | One sinusoid, positions 0 and half a wavelength: offset 0, phases bit-identical |
| | Phase range when the input is outside [-pi, pi) | Input shifted by 20 rad: output inside the interval (measured: -3.086 to 3.123) |
| | `pos_b` with too many columns; `pos_b` with fewer columns, same map | Validation section; padded comparison |
| | `phase_b` aliasing `phase` (B9) | After the fix |
| `sos_init_phases` | Nothing significant | - |
| `sos_acfi` | NaN in `d`; output aliasing an input (B4) | Validation section; aliasing case after the fix |
| `sos_acf_2d` | Long `float` table (B3) | 5001 samples in `float` |
| `sos_acf_approx` | NaN or Inf in `dist` (B6) | Validation section |
| `sos_calc_mse` | A perfect fit (returns +Inf dB in 2-D and 3-D) | Decide and pin: Inf or a cap |
| `sos_acf_estimate` | The 1e-5 m lower limit of the first bin; non-finite `bins` | Positions cannot be injected, so only the validation part is testable |
| `sos_generate` | Unequal sample spacing | Reported error against the brute-force oracle on an unequal table (measured here: equal to 12 digits) |
| | Termination at 1000 passes; "gain below 1e-6" | The 1-D case already runs into the limit; assert `ase.n_elem <= 1000` and that one configuration stops earlier by the gain rule |
| | `n_coeff = 1`, `n_acf = 2`, `n_dir = 1` | Smoke case (runs: 2 passes, error 1.9e-11) |
| | Large `freq_max` (B8), non-finite `dist` (B6) | After the fixes |
| `sos_preset` | The shipped table values | A checksum per table, for example the sum and the sum of squares of the 128 to 1024 stored numbers to 1e-15 relative. Today a changed digit passes as long as the fit stays above the threshold |
| MEX wrappers | Negative, NaN, non-integer scalars (B1, B10); complex arrays; `uint8` | One `assert_throws` per case after the helper fix |
| Python wrappers | `None` for mandatory arrays (B2); 1-D `pos`; C-ordered against Fortran-ordered input | `assertRaises` after the fix; one orientation case per wrapper that takes `pos` |

Documented statements without a test, in addition to the table:
- `sos_val`: "the value does not depend on ... the number of threads" is tested for one device only.
- `sos_rand`: "nearby positions give similar values in separate calls with the same seed" (a subset test
  exists; similarity is not asserted).
- `sos_generate`: "with few test directions the result fits those directions and is worse in others".
- `sos_preset`: "magnitude below 0.015 at 10, 20, 50 and 100 decorrelation distances" is tested; the seeds
  listed in the documentation are tested for the three 128 tables only.

### 2.3 Oracles and tolerances

- The oracles are independent of the implementation in every suite read in full (`test_sos_val.cpp`) or in
  part (`generate`, `preset`, `remove_bias`, `calc_mse` headers and main cases): scalar transcriptions,
  hand values, own interpolation, own random stream. `test_sos_rand.cpp` and `test_sos_preset.cpp` use
  `sos_preset`, `sos_init_phases`, `sos_val` and `sos_calc_mse` as building blocks, which the spec allows
  because they have their own suites.
- `test_sos_remove_bias.cpp` takes the mean from `sos_val` (GENERIC). A common error of `sos_val` and
  `sos_remove_bias` in the phase convention would cancel; the closed-form oracle in scalar complex
  arithmetic covers that.
- Tolerances looser than the arithmetic needs:
  - `sos_val`, float against double: 1e-3. Justified in the comment by the input rounding; measured
    difference 1.1e-5. The stricter test proposed above separates input rounding from the computation.
  - `sos_randi`, counts within 150 of 1000: five standard deviations of independent samples would be 144,
    so the limit is honest, but it would not notice a mapping that shifts one face by 10 %. The exact
    mapping test covers that.
  - `sos_preset`, regenerated tables within 1.5 dB: needed across platforms. On this machine the three
    128 tables are regenerated to 3e-17 by gcc 12 and gcc 15, with and without AVX2, and under the
    sanitizers [M]. Other platforms were not tested.
  - `sos_preset`, correlation within 0.08 of the target ACF: measured deviation 0.044.
- No tolerance was found that hides a defect.

### 2.4 Wrong implementations that would pass

- A changed number in a preset table that keeps the fit within the thresholds (see the checksum above).
- Swapped roles of `pos` and `pos_b` in the same-map mode (the result is symmetric by design).
- A transposed `sos_acf_2d` (symmetric by design).
- A different rule for which entries of `sos_randn` share one kernel call: results are identical either
  way, so only performance changes.
- A threshold in `sos_remove_bias` of any size between about 1e-16 and 1e-3 times `n_coeff`.
- An `sos_generate` that uses the cosine path for all tables, or the recurrence for unequal spacing (the
  second would be wrong and is not caught).

### 2.5 What the build without AVX2 does not exercise

- `qd_SOS_AVX2` itself and every `kernel == 2` branch of the tests (80 assertions fewer).
- The kernel parity case and the "auto kernel is AVX2" case reduce to GENERIC against GENERIC.
- It does not assert that requesting kernel 2 throws; the call does throw [R].
- MEX and Python suites pass without AVX2; their value tolerances of 1e-6 are then met by GENERIC.


## 3. Performance

### 3.1 Conditions

- AMD Ryzen 9 5950X, 16 cores, 32 threads, L1d 32 kB, L2 512 kB per core. Library built with gcc 12, `-O3`,
  as `make avx2=ON` does.
- Each number is the fastest of at least 3 calls after a warm-up call, one process per thread count.
- The machine was shared. The tables of 3.2 to 3.4 were measured after the other job had ended (load below 1
  at the start). The kernel variants of 3.6 were measured later with an editor and an idle MATLAB session
  in the background, and their 16-thread numbers are less reliable. Values marked "(busy)" were taken
  while the other job ran. Preset `Comb128` to `Comb1024`, positions uniform in a cube of 1 km.
- Programs: `scratchpad/bench/bench.cpp` (kernels and `sos_val`), `bench2.cpp` (other functions),
  `bench3.cpp` with `var_*.cpp` (kernel variants), `bench4.cpp` (parts of `sos_val`).
- `perf` is not available to the session (`perf_event_paranoid = 4`), so cache behaviour is inferred from
  timings and from the disassembly.

### 3.2 Kernels

Time per (position × generator) in ns, one thread [M]. `n_pos` = 100,000 (1,000,000 for one generator);
for 1000 and 10,000 positions the values are within 3 % for 1 and for 400 or more generators and up to
16 % lower for 20 generators.

| n_gen | N | GENERIC | AVX2 | AVX2, ms per call |
|---|---|---|---|---|
| 1 | 128 | 3272 | 170 | 170 |
| 1 | 256 | 6569 | 337 | 337 |
| 1 | 512 | 13145 | 673 | 673 |
| 1 | 1024 | 26362 | 1343 | 1343 |
| 20 | 128 | 235 | 15.9 | 31.8 |
| 20 | 256 | 488 | 35.2 | 70.5 |
| 20 | 512 | 1017 | 70.3 | 140 |
| 20 | 1024 | 1986 | 156 | 312 |
| 400 | 128 | 84.8 | 9.13 | 365 |
| 400 | 256 | 178 (10,000 pos.) | 17.0 | 681 |
| 400 | 512 | 368 (10,000 pos.) | 34.1 | 1365 |
| 400 | 1024 | 736 (10,000 pos.) | 82.3 | 3290 |
| 2288 | 128 | 78.2 (10,000 pos.) | 9.32 | 2131 |
| 2288 | 256 | 164 (1000 pos.) | 17.4 | 3972 |
| 2288 | 512 | 340 (1000 pos.) | 33.9 | 7747 |
| 2288 | 1024 | 766 (1000 pos.) | 77.5 | 17734 |

- Sine and cosine cost 1.3 ns per (sinusoid × position) in AVX2 and 25.6 ns in GENERIC. They dominate up to
  about 20 generators.
- The accumulation costs 0.066 ns per (sinusoid × position × generator) in AVX2, 0.076 ns for 1024
  sinusoids, and 0.69 ns in GENERIC. It dominates from about 50 generators.
- `n_pos = 1,000,000` with 400 or 2288 generators was not run for memory: `sos_val` would need 2 × 18 GB.

Thread scaling of the AVX2 kernel, threads not bound, ns per element and speed-up [M]:

| n_gen, N, n_pos | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 1, 256, 1e6 | 327 | 167 (2.0) | 84.7 (3.9) | 43.1 (7.6) | 22.1 (14.8) | 18.4 (17.8) |
| 20, 256, 1e5 | 30.1 | 15.0 (2.0) | 7.60 (4.0) | 3.89 (7.8) | 2.02 (14.9) | 1.82 (16.6) |
| 400, 256, 1e5 | 16.9 | 8.60 (2.0) | 4.47 (3.8) | 2.28 (7.4) | 1.25 (13.6) | 1.31 (12.9) |
| 400, 1024, 1e5 | 77.6 | 38.3 (2.0) | 20.8 (3.7) | 10.9 (7.1) | 6.36 (12.2) | 6.93 (11.2) |
| 2288, 256, 1e5 | 16.5 | 8.87 (1.9) | 4.35 (3.8) | 2.37 (7.0) | 1.33 (12.4) | 1.30 (12.7) |

- The kernel scales to 16 cores for 400 generators (13.6 times). The stop at 8 threads reported in
  `LOG_QD_SOS.md` is not a property of the kernel; it comes from the serial code in `sos_val.cpp`
  (section 3.3), which that benchmark included, and from its small problem size (5000 positions).
- With `OMP_PROC_BIND=true` and 16 threads the same kernel took 2.01 ns instead of 1.25 ns [M]. A later
  check gave 2.03 ns (bound), 1.81 ns (`OMP_PLACES=cores`) and 1.83 ns (not bound), all slower than the
  first run, so the machine was probably not idle then. The cause was not established and the bound numbers
  are not used here.

### 3.3 `sos_val` around the kernel

Parts of one call, ms [M]. `sos_val` computes into a temporary matrix and copies it.

| Part | 400 gen, N 256, 1e5 pos, 1 thread | same, 16 threads | 1 gen, 1e6 pos, 16 threads | 20 gen, 1e5 pos, 16 threads |
|---|---|---|---|---|
| Kernel into an existing matrix | 685 | 51.7 | 22.7 | 4.01 |
| Kernel into a new temporary | 835 | 96.5 | 23.1 | 4.22 |
| Copy temporary to `val` | 26.8 | 27.7 | 0.14 | 0.55 |
| Weights (cos, sin of the phases) | 1.98 | 1.96 | 0.004 | 0.08 |
| Positions to x/y/z vectors | 0.28 | 0.28 | 3.16 | 0.28 |
| `erfc` mapping as written (serial) | 1147 | 1142 | 28.8 | 57.9 |
| `erfc` mapping, parallel loop | 1147 | 83.6 | 1.85 | 3.88 |
| Normalisation weights, same map | 0.51 | 0.52 | 5.16 | 0.51 |
| Combination of two devices | 29.0 | 28.6 | 0.24 | 0.71 |
| **`sos_val`, `val` already sized** | 839 | 121 | 36.7 | 5.55 |
| **`sos_val`, `val` empty** | 987 | 264 | 49.3 | 16.0 |
| **`sos_val`, `uniform = true`** | 1974 | 1250 | 74.2 | 68.6 |
| **`sos_val`, two devices, same map** | 1710 | 270 | 98.1 | 16.4 |

- With 16 threads `sos_val` takes 2.3 times the kernel when `val` is pre-sized and 5.1 times when it is
  empty (400 generators). The temporary of 320 MB is allocated and page-faulted on every call, then copied
  by one thread; an empty `val` is page-faulted by that same thread.
- `uniform = true` takes 24 times the kernel with 16 threads: 4e7 calls of `erfc` on one thread.
- The `float` instantiation costs the same as `double` (855 against 861 ms on one thread).
- Peak memory: 2 times the output for one device, 3 times for two devices.

### 3.4 Other functions

All times in ms [M].

| Call | 1 thread | 16 threads |
|---|---|---|
| `sos_preset` Comb128 / Comb256 / Comb1024 | 0.006 / 0.010 / 0.034 | same |
| `sos_init_phases` 256 × 2 / 256 × 800 | 0.002 / 0.93 | same |
| `sos_val`, 1 position, 1 generator | 0.0076 | 0.0094 |
| `sos_randn`, 1 position, 1 generator | 0.020 | 0.023 |
| `sos_val`, 1e5 pos, 8 generators | 41.7 | 3.40 |
| `sos_randn`, 1e5 pos, 8 equal distances | 48.3 | 7.70 |
| `sos_randn`, 1e5 pos, 8 different distances | 268 | 21.0 |
| `sos_randn`, 1e5 pos, 8 equal, two devices | 93.4 | 11.8 |
| `sos_rand`, 1e5 pos, 8 equal distances | 70.9 | 33.2 |
| `sos_randi`, same | 71.3 | 33.7 |
| `sos_val`, 1e5 pos, 400 generators | 863 | 134 |
| `sos_randn`, 1e5 pos, 400 equal distances | 1022 | 296 |
| `sos_randn`, 1e5 pos, 400 different distances | 13465 | 1142 |
| `sos_map`, 501 × 201 grid, 20 generators | 77.6 | 18.7 |
| `sos_val` on the same 100,701 positions | 62.3 | 5.53 |
| `sos_remove_bias`, 1e5 pos, 8 or 400 generators, one device | 643 / 645 | 633 / 635 |
| `sos_remove_bias`, same, same map | 1284 / 1288 | 1265 / 1267 |
| `sos_remove_bias`, 1000 pos, 8 generators | 6.42 | 6.35 |
| `sos_calc_mse` Comb1024, 10 / 300 directions, return value only | 17.9 / 539 | 17.7 / 534 |
| `sos_calc_mse` Comb1024, with `mse_core`, `mse_all` | 1363 | 168 |
| `sos_acf_approx` Comb256 / Comb1024 | 317 / 1337 | 37.0 / 146 |
| `sos_acf_2d`, 797 × 797 | 1.3 | 1.4 |
| `sos_acf_estimate`, `n_pos` 1000 / 5000 / 20000 | - | 57.9 / 1420 / 22475 |
| `sos_generate`, 128 / 256 sinusoids, preset settings | 298 (7 passes) / 857 (9 passes) | serial |
| `sos_generate`, 24 sinusoids, 1-D, as in the tests | 159 (1000 passes) | serial |

- `sos_remove_bias` does not use threads and costs 25 ns per (position × sinusoid). For 8 generators and
  16 threads it is 190 times slower than evaluating them.
- `sos_randn`: preset and phases cost 12 µs per call. Drawing phases for `2 · n_gen` generators and loading
  the preset on every call are not a cost worth changing. The overhead over `sos_val` (7.7 against 3.4 ms)
  is copying: positions, phase columns for both devices, the temporary inside `sos_val`, and the result.
- `sos_acf_estimate` costs 28 ns per pair on one thread.
- Catch2 `sos_*` cases: 10.0 s in total. The slowest are `sos_preset` fit (2.1 s), `sos_preset` regeneration
  (1.5 s), `sos_val` position independence (1.2 s), `sos_remove_bias` scan (1.1 s). The 1-D optimiser case
  with four runs of 1000 passes takes 0.64 s, so it is not what makes the suite slow.

### 3.5 Wrappers

- MEX [C]: `double` inputs are used in place (`qd_mex_get_Mat` reinterprets); other classes are converted.
  `val`, `map`, `phase`, `acf`, `dist`, `acf_2d`, `acf_approx`, `dist_2d`, `acf_est` are written directly
  into MATLAB memory. `freq`, `ase`, `test_dir`, the scalars of `sos_calc_mse` and all outputs of
  `sos_remove_bias` are copied once (their size is known only after the call, or the input must stay intact).
- Python [C, R]: Fortran-ordered arrays and 1-D arrays are used in place; C-ordered 2-D arrays, which is
  numpy's default, and lists are copied. Outputs as in MEX.
- Cost of the copies: not measurable for large inputs. `sos.val` with 1e5 positions and 400 generators takes
  132 ms (Fortran-ordered `pos`) and 130 ms (C-ordered) against 121 to 134 ms for the C++ call [M].
- Per call [M]: Python `sos.val` for 1 position 76 µs, `sos.randn` 42 µs, `sos.preset` 10 µs. MEX `sos_val`
  15 µs, `sos_randn` 31 µs, `sos_preset` 13 µs (busy). The C++ calls take 8 to 20 µs.
- The Python `calc_mse` always computes `mse_core` and `mse_all` (1.36 s on one thread for Comb1024, 0.17 s
  on 16), because a Python wrapper returns all outputs. The MEX wrapper computes them only when requested.

### 3.6 Proposals, ranked by gain against effort

| # | Proposal | Cost today | Gain | Basis | Effort and guarantees |
|---|---|---|---|---|---|
| P1 | `sos_val`: let the kernel write into `val` when `dtype` is `double`; run the copy, the `erfc` mapping, the two-device combination and the normalisation as parallel loops | 121 to 264 ms against a kernel of 52 ms; Uniform 1250 ms (400 gen, 1e5 pos, 16 threads) | 2.3 to 4.8 times; Uniform about 9 times; 1.2 to 1.4 times on one thread; memory 1 instead of 2 outputs | Sum of the measured parts in 3.3 | About 40 lines in `sos_val.cpp`. All loops are element-wise, results bit-identical. `sos_rand`, `sos_randi`, `sos_map` gain with it |
| P2 | `sos_remove_bias`: parallel loop over the sinusoids in `accumulate` | 633 ms for 1e5 positions, any thread count | About 14 times on 16 cores | Kernel scaling in 3.2; each `z_n` is computed by one thread in the same order | A pragma and a signed loop variable. Bit-identical |
| P3 | AVX2 kernel: weights packed per group of 8 generators, 8 named accumulators, positions in tiles so that the weights of a group stay in the first-level cache | 17.0 ns (400 gen, N 256), 76 ns (N 1024), 1 thread | 1 thread: 1.55 (N 256), 1.72 (N 1024), 1.64 (2288 gen), 1.62 (N 128) times. 16 threads: 2.0 (400 gen), 1.34 (2288 gen) times. 20 generators: 1.1 times. 1 generator: 10 % slower unless small counts keep a narrow path | Prototype `var_t.cpp`, output bit-identical to `qd_SOS_AVX2` in all runs | Kernel grows from about 100 to 170 lines. Same arithmetic per (position, generator), so every existing test keeps its meaning |
| P3a | Smaller step of P3: only replace the `part[j]` array with a run-time loop bound by named accumulators | as P3 | 1.15 times | Prototype `var_h.cpp`, bit-identical | 30 lines. gcc 12 keeps `part[]` on the stack and branches on the group size inside the innermost loop (disassembly) |
| P4 | GENERIC kernel: accumulate 4 generators per pass over the sinusoids | 185 ns (400 gen, N 256, 1 thread) | 1.9 times at 400 generators, 1.3 times at 20, none at 1 | Prototype `var_generic.cpp`, bit-identical. `sincos()` instead of `cos` and `sin` gave no gain | 20 lines. Matters where GENERIC is the only kernel: default `make`, ARM |
| P5 | `sos_calc_mse`: parallel loop over the test directions, per-direction sums added in order | 539 ms (Comb1024, 300 directions) | About 14 times | Loop is independent per direction | Small. Bit-identical if the partial sums are added serially |
| P6 | `sos_randn` and friends, `sos_map`: no conversion copy for `double`, no copy of unused second-device phases, evaluate directly into the output columns when all entries form one group | 7.7 against 3.4 ms; 18.7 against 5.5 ms | Most of it comes with P1; the rest is below 20 % | 3.4 | Small, but more branches in the helper |
| P7 | `sos_acf_estimate`: parallel pair loop over fixed blocks of rows, merged in order | 22.5 s for 20,000 positions | About 14 times | 28 ns per pair, independent rows | Medium: the block size must not depend on the thread count, otherwise the sums change in the last bit. Offline tool |
| P8 | `sos_acf_approx`: use `R(-dx) = R(dx)` and compute half of each plane | 1337 ms (Comb1024, 1 thread) | 2 times | Symmetry | Small. Offline tool |
| P9 | `sos_map`: sines and cosines per axis and angle addition instead of one evaluation per grid point | 77.6 ms (1 thread, 20 generators), about half of it sine and cosine | Up to 2 times for 20 generators, more for fewer | Estimate from the share of sine and cosine, not prototyped | A third kernel. Values would differ from `sos_val` in the last bits, which the documentation of `sos_map` excludes today |

Not proposed:
- More generators per group in the present kernel (`QD_SOS_GEN` 6, 8, 12): no gain or slower, up to 3.5
  times slower for 1024 sinusoids, where the weight columns are 4096 bytes apart and collide in the cache
  [M, `var_g*.cpp`].
- A tiled kernel with 4 blocks per generator: slower for 256 sinusoids and more [M, `proto_avx2.cpp`].
- A looser stop criterion in `sos_generate`: it would change the shipped tables' regeneration for a saving
  of 0.6 s in the tests.

### 3.7 The leads on performance

| Lead | Verdict |
|---|---|
| Copies and a temporary in `sos_val.cpp` for every call | Confirmed; the largest cost at 16 threads (3.3) |
| AVX2 kernel stops scaling beyond 8 threads for 400 generators | Not reproduced for the kernel (13.6 times at 16 threads). True for `sos_val` as a whole: 839 ms on 1 thread, 121 ms on 16, a factor of 6.9. Walking all weights per block is real and is what P3 removes |
| GENERIC on one thread slower than MATLAB for one generator | GENERIC: 6.6 µs per position for 256 sinusoids. MATLAB was not re-measured. The time is `cos` and `sin` of the math library; P4 does not change it |
| `sos_randn` draws `2 · n_gen` phase columns and loads the preset on every call | Measured as negligible: 12 µs per call, 0.93 ms for 800 columns |
| 1-D fits run 1000 passes and make the suite slow | They do run 1000 passes (4 of 12 configurations), each 0.16 s. The suite time is in the `sos_preset` cases |
| `sos_acf_estimate` runs a serial pair loop | Confirmed, 28 ns per pair |
| 2-D tables for the exponential ACF stop improving | See below |

The 2-D exponential tables [M], 200-sample table, `n_dir = 100`, `freq_min = 0.005`, error on 199 other
directions:

| N | seed | start | `freq_max` | ASE [dB] | passes |
|---|---|---|---|---|---|
| 250 | 1 | random | default | 32.24 | 11 |
| 250 | 1 | random | 4 × default | 32.24 | 11 |
| 500 | 1 | random | default | 36.29 | 26 |
| 1000 | 1 | random | default | 34.43 | 24 |
| 1000 | 1 | random | 2 × and 4 × default | 33.61 | 22 |
| 1000 | 1 | build-up | default | 34.97 | 21 |
| 1000 | 2 | random | default | 37.62 | 28 |
| 1000 | 3 | random | default | 34.31 | 30 |
| 1000, 3-D | 1 | random | default | 37.08 | 32 |
| 1000, Gauss, 2-D | 1 | random | default | 52.11 | 77 |

- The upper frequency limit is not the cause: no sinusoid lies within 5 % of it (largest magnitude 0.285
  against 0.302 per metre), and raising it does not help.
- The start mode is not the cause.
- The result depends strongly on the seed (34.3 to 37.6 dB for 1000 sinusoids), and 500 sinusoids with seed
  1 beat 1000 sinusoids with seed 1. Every run ends with a pass that accepts nothing. This points to the
  search ending in local optima for this ACF in 2-D; the Gaussian ACF does not show it. The cause is still
  not established.


## 4. Documentation

Read: the header, all C++ blocks, the MEX and Python blocks of the introduction, `sos_val`,
`sos_remove_bias`, `sos_randn`, `sos_map`, `sos_init_phases`, `sos_preset`, `sos_generate`, `sos_calc_mse`,
the ACF tools, and the generated Markdown files. The MEX and Python blocks of `sos_rand` and `sos_randi`
were compared with `sos_randn` by structure only.

### 4.1 Statements that do not match the code

| # | Where | Statement | What the code does |
|---|---|---|---|
| D1 | `sos_val`, all three APIs; header line 85 | AVX2 values differ from GENERIC by less than `4e-7 · sqrt(2·n_coeff)` "at any distance from the origin" | Holds up to about 1e9 periods. Measured for Comb256: 3.2e-7 at 1.5e8 periods, 2.5e-6 at 4.5e9, 6.7e-5 at 1.5e11 (bound 9.1e-6), 0.07 at 1.5e14 [R]. Both kernels lose the argument there, because the product `f·x` is rounded to about 1e-16 of its size |
| D2 | Header line 86 against the `sos_val` block | "absolute error of val below 1e-5" | The block gives `4e-7 · sqrt(2·n_coeff)`, which is 1.8e-5 for 1024 sinusoids. Measured maximum over 6 presets, 10 m and 1 m, 160,000 values each: 6.8e-7 [M] |
| D3 | Header lines 84, 100, 115, 141; "Uniform between 0 and 1" | Uniform in (0, 1), open interval | The value can be exactly 1: 64 sinusoids with aligned phases give 1.0 in `double` [R]. In `float` every value above 5.4 maps to 1.0f. Not found in 8e7 random `float` values [M] |
| D4 | Header line 34; `sos_randn`, `sos_rand`, `sos_randi`: "with a seed >= 0 the same positions always give the same values" | Reproducible values | True for the phases. The values come from the automatic kernel: AVX2 and GENERIC differ by up to 6.8e-7, and 259 of 1e7 integers of `sos_randi` with `imax = 1000` differ (2 of 1e7 for `imax = 6`) [R]. The three functions have no `use_kernel` argument |
| D5 | Introductions: "With the same seed, a later call returns the same field, also at other positions" | Same field | Only for the same number and order of entries when a second device has its own map, and the same seed also ties together fields of different calls (1.2) |
| D6 | `sos_remove_bias`, MEX and Python: "Phases are returned in the interval from -pi to pi" | Always | Not for a generator below the threshold, which is returned as given (5.0 stays 5.0) [R]. The C++ block says so; the wrapper blocks dropped that sentence |
| D7 | Header line 217: "For n_dim = 1, mse_core and mse_all are evaluated on the 1-D axis" | Both on the axis | `mse_core` is evaluated on the sample points and equals the return value; only `mse_all` uses the axis. The blocks say it correctly |
| D8 | Introductions: "evaluating many generators in one call costs little more than evaluating one" | Little more | 20 generators cost 1.8 to 2.1 times one generator, 400 cost 20 times [M]. Per generator the cost falls from 337 to 17 ns |
| D9 | `sos_val` C++: "`val` must not be the same object as an input" | Restriction | Harmless today. The functions where aliasing does fail (`sos_acfi`, B4) do not carry the remark |

Statements verified as true: the preset formulas and scaling, "no wavelength above 20 decorrelation
distances" (smallest table value 0.005 per metre), the phase stream, the bin rules of `sos_acf_estimate`,
the stop rule and grid spacing of `sos_generate`, "more sinusoids take proportionally longer" (9.1, 17.0,
34.1, 82 ns), "about 35 dB" in the example (34.85 dB).

### 4.2 Consistency between header, C++, MEX and Python

- Sizes, optionality and defaults agree. `n_gen` of `sos_init_phases` has the default 1 in MEX and Python
  and none in C++.
- Units: the C++ blocks state `[m]` throughout. MEX and Python drop it, as their specs ask for SI units, but
  keep `[1/m]` and `[rad]`. Radians are in the specs' own list of SI units, so this is applied unevenly.
  Positions and decorrelation distances without a unit next to frequencies with one read oddly.
- Error conditions are stated as "must be" clauses in all three. The wrapper-level errors (argument counts)
  are not documented anywhere, which matches the rest of the library.
- The Python entries are headed `sos_val` and so on, while the functions are called `quadriga_lib.sos.val`.
  The introduction explains this; `SPEC_PYTHON_API.md` asks for the name as exposed to Python.

### 4.3 Examples of the three introductions

Run as written [R]:

| API | Block 1 (quick start) | Block 2 (keeping a generator) | Block 3 (two devices) | Block 4 (own ACF) |
|---|---|---|---|---|
| C++ | runs | `pos_new` is not declared | `pos_tx`, `pos_rx` are not declared | runs, 34.85 dB |
| MATLAB | runs | "Unrecognized function or variable 'pos_new'" | runs only because block 2 was repaired | runs, 34.85 dB |
| Python | runs | "name 'pos_new' is not defined" | `pos_tx`, `pos_rx` undefined | runs, 34.85 dB |

- With `pos_new`, `pos_tx` and `pos_rx` defined, all blocks run in all three APIs and give the documented
  sizes. The Python example writes `map.png`.
- C++: pasted into one scope the blocks redeclare `val`, `uni`, `freq`, `acf` and `dist`. Each block
  compiles in its own scope.
- Python: the example assigns to `map`, which hides the built-in.

### 4.4 What a user needs and does not find

- **Seeds.** That a seed identifies a set of phases, not a field: the same seed with another decorrelation
  distance or preset gives a related field (1.2). Advice: one call with several entries, or one seed per
  quantity.
- **Decorrelation distance.** The introductions use the term without defining it. One sentence: the
  distance at which the correlation has fallen to exp(-1), about 0.37. It appears only in `sos_preset`.
- **Choosing a preset.** `Comb256` is named as default, but not what the others are for: the fit improves
  by about 3 dB per doubling (28, 31, 34, 38 dB for Exp), the cost doubles. A two-line table would do.
- **`sos_remove_bias`.** The statistics are explained well. Missing is a rule of thumb for when to call it;
  `LOG_QD_SOS.md` has the numbers (loss of variance 3 to 5 % for an area of 20 decorrelation distances and
  more, 43 % at 2, 69 % at 1). Also missing: `phase_b` must not be `phase` (B9).
- **Coordinates and precision.** The `float` instantiation rounds positions to single precision (1 mm at
  10 km); the accuracy of both kernels falls with the distance from the origin in units of the wavelength
  (D1). Neither is in the blocks.
- **Memory and threads.** `sos_val` needs two to three times the output size (until P1), and all functions
  use the OpenMP thread count. `sos_remove_bias`, `sos_generate`, `sos_acf_estimate` are serial.
- **Python.** A 1-D array for `pos` is a column. `[x, y, z]` is three positions.
- **Tolerance of "equally spaced".** `sos_acf_2d`, `sos_acf_approx` and `sos_calc_mse` do not say how equal
  (1e-4 of the step).
- **`sos_calc_mse`.** A perfect fit returns +Inf.

What is there and a user does not need: the internal description of the kernels in `sos_val` (partial sums,
chunking) is absent from the blocks, which is right. The derivation in `sos_remove_bias` is long but is what
makes the caveats credible. Nothing to remove.

### 4.5 Conformance with the documentation rules

Checked by script over all `/*!MD` and `/*!SECTION_DESC` blocks [M]:

- No `*` or `|` in running text in any of the 42 blocks.
- All `[[links]]` resolve to an entry of the same API.
- Line length, C++ (120): three lines exceed it, the `val` output lines of `sos_randn`, `sos_rand` and
  `sos_randi` (153, 137, 157 characters, `src/sos_rand.cpp:169, 234, 301`), plus one link line.
- Line length, MEX (100): the function blocks comply except one usage line of 101 characters
  (`api_mex/sos_calc_mse.cpp:39`). The section introduction in `api_mex/sos_val.cpp` has 23 lines of 101 to
  132 characters. Whether the limit applies to
  an introduction, which is not turned into a `.m` help text, is for you to decide.
- Line length, Python (120): complies except the same link line.
- Layout (summary line, sections, argument format, `*(optional)*` against `default:`): complies in the
  blocks read.


## 5. Not checked, and why

- **MSVC, macOS, ARM.** Only gcc 12 and gcc 15 on Linux x86-64 were available. OpenMP 2.0 conformance and
  the reproducibility of the tables across math libraries are read from the code, not tested.
- **Octave.** Not run.
- **MATLAB reference values.** The implementation was not compared value by value with `@qd_sos`;
  decision D2 excludes that. The optimiser was compared with `generate.m` and `suosi_iteration_step.m` by
  reading.
- **MATLAB timing.** The comparison of GENERIC with MATLAB's `val` in the log was not repeated.
- **Cache behaviour.** No hardware counters. The explanations in 3.2 and 3.6 rest on timings, the
  disassembly and cache sizes.
- **Thread binding.** Why `OMP_PROC_BIND=true` cost 60 % in one run was not established.
- **Machine load.** Another job used the machine during parts of the session. The benchmark tables were
  taken while it was idle; the MEX per-call times and the first full Catch2 run (17 minutes wall time) were
  not.
- **Every sentence of every block.** The statements with a number or a testable claim were verified. The
  MEX and Python blocks of `sos_rand`, `sos_randi` and the ACF tools were read for structure, not word by
  word against the code.
- **Tests read in full:** `test_sos_val.cpp`, three MEX tests. The other Catch2 files were read in their
  headers, test case lists and selected cases; their coverage was judged from gcov. The Python tests were
  run but not read.
- **The 2018 paper.** The formulas of `sos_val` were not re-derived from the paper;
  `tools/sos_port/SOS_PAPER_REPRODUCTION.md` was skimmed for contradictions and none was found.
- **Other wrappers of the library** for B2 and B10. The causes are in shared helpers and in the wrapper
  pattern of the specs, so they probably reach beyond this port.
