# Arrayant interpolation refactor

This document plans and specifies the refactor of the antenna pattern interpolation API before the 1.0 release. It records what changes, why, and how the result is checked, so that an implementer can act on it without further context and a later review can trace each change to its reason. It is a working document: it is deleted when the refactor is complete and remains available in the git history. Section 4 lists the decisions and their reasons; all are settled. Sections 5 to 9 are written for the settled state.

Rules that apply throughout: `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`, `CONTRIBUTING.md`. Source code, tests and commit messages do not refer to this document (`md/SPEC_CPP_API.md`, "GENERAL CONVENTIONS").

# 1. Purpose and scope

- Replace the member function `arrayant<dtype>::interpolate` and the internal function `qd_arrayant_interpolate` by one public, documented free function `quadriga_lib::arrayant_interpolate`, placed directly before `arrayant_interpolate_multi`.
- Move the interpolation kernels to the layout used by the other accelerated functions (public function validates and dispatches; kernels `_GENERIC` / `_AVX2` without own checks), with a `use_kernel` argument.
- Migrate `arrayant_interpolate_multi` to the same argument conventions, and replace `bool use_avx2` in `get_channels_spherical` by `int use_kernel`.
- Move all library callers to the new function, delete the internal headers and sources, and bring the MEX and Python wrappers of these functions and their tests in line with their SPECs.
- API changes are intended: the interpolation API is changed once now so that it stays stable after 1.0.

Not in scope: the interpolation algorithm itself (results of the GENERIC kernel stay as they are), the remaining SPEC deviations of the `get_channels_spherical` wrappers other than the kernel argument (section 10), symbol visibility of the shared libraries (section 10).

# 2. Current state

Line numbers refer to commit `34e675e`.

## 2.1 Functions and files

| Item | Where | Notes |
| :-- | :-- | :-- |
| `arrayant<dtype>::interpolate` | declared `include/quadriga_arrayant.hpp:77`, defined `src/qd_arrayant.cpp:1218` | public member; mandatory arguments by pointer; validates, resizes outputs, converts `i_element` to 1-based `unsigned`, copies element positions, calls `qd_arrayant_interpolate` |
| `qd_arrayant_interpolate` | `src/qd_arrayant_interpolate.hpp`, `src/qd_arrayant_interpolate.cpp` | internal scalar kernel; validates again with its own messages; requires pre-sized outputs; treats a non-null but empty optional output as "not requested" |
| `qd_arrayant_interpolate_avx2` | `src/qd_arrayant_interpolate_avx2.hpp`, `src/qd_arrayant_interpolate_avx2.cpp` | internal AVX2 kernel; computes in single precision for `float` and `double`; repeats the checks; calls the scalar kernel for patterns with more than `INT_MAX` samples (int32 gather offsets); own copy of `qd_rotation_matrix` |
| `arrayant_interpolate_multi` | `include/quadriga_arrayant.hpp:160`, `src/qd_arrayant_xinterpol_multi.cpp:88` | mandatory arguments by pointer, `i_element` by value, `bool validate_input`; calls the scalar kernel per frequency entry |
| `get_channels_spherical` | `include/quadriga_arrayant.hpp:356`, `src/get_channels_spherical.cpp` | `bool use_avx2 = false` selects the AVX2 interpolation kernel (`:523`) and the AVX2 coefficient kernel `qd_coeff_combine_avx2` (`:578`) |

The single-precision AVX2 design is deliberate: the write-up of its last refactor states that both instantiations run the same float path and that double-precision results come from the scalar reference (`tests/benchmarks_cpp/REFACTORING_RESULTS_ARRAY_INTERP.md` at commit `ebe677b`, removed in `67d09b3`). `qd_coeff_combine_avx2<double>` computes the Jones product in float and the phase in double (`src/qd_coeff_combine_avx2.cpp:242-251`).

## 2.2 Callers of the internal kernels and the member

- `combine_pattern` (`src/qd_arrayant.cpp:581`), `rotate_pattern` (`:1944`, `:1952`). Both are used by `generate_arrayant_*` (`src/qd_arrayant_generate.cpp:605-611`, `657`, `1014-1024`, `1071`) and by `arrayant_combine_pattern_multi` / `arrayant_rotate_pattern_multi`. `rotate_pattern` relies on "empty optional output = not requested".
- `arrayant_interpolate_multi` (`src/qd_arrayant_xinterpol_multi.cpp:231`, `242`).
- `get_channels_planar` (`src/get_channels_planar.cpp:294`, `308`); itself called by `gbsm_single_link_channel` and `get_channels_ieee_indoor`.
- `get_channels_spherical` (`src/get_channels_spherical.cpp:521-554`); itself called by `get_channels_irs` (`src/get_channels_irs.cpp:196`, `201`, `247`, `252`) and `gbsm_single_link_channel` (`src/gbsm_single_link_channel.cpp:376`), both with the default `use_avx2 = false`.
- `get_channels_multifreq` (`src/get_channels_multifreq.cpp:788-857`): six calls per path, inside the OpenMP frequency loop (`:601-995`), with `n_links x 1` angles each.
- Member callers: Python wrapper (`api_python/qpy_arrayant_interpolate.cpp:195-200`), MEX wrapper (`api_mex/arrayant_interpolate.cpp:197-202`), fuzz target (`tests/fuzzing/targets/fz_arrayant_interpolate.cpp:79-82`), 33 calls in `test_arrayant_interpolate.cpp`, `test_arrayant.cpp`, `test_arrayant_interpolate_multi.cpp`.
- Explicit `use_avx2` arguments: `tests/catch2_tests/test_get_channels_spherical.cpp:1943`, `1946`, `1982`, `1985`; `tests/fuzzing/targets/fz_get_channels.cpp:78-82`; `api_mex/get_channels_spherical.cpp:142`. The Python wrapper of `get_channels_spherical` does not pass it.

## 2.3 Index convention

The public functions take 0-based `arma::uvec`. The member and `_multi` copy it into a 1-based `arma::Col<unsigned>`; `get_channels_*` build 1-based `Col<unsigned>` per link. Both kernels reject 0 and subtract 1 immediately (`qd_arrayant_interpolate.cpp:308`, `qd_arrayant_interpolate_avx2.cpp:867`). The 1-based form has no function: 0 is never a sentinel.

## 2.4 Symbol exposure

`qd_arrayant_interpolate<float|double>` and `qd_arrayant_interpolate_avx2<float|double>` are exported (weak symbols) from the Python module and the static library, without a public declaration. This is not a privilege boundary (code that links the library can call any exported symbol), but it is an undocumented interface with its own, second set of checks. All internal `qd_*` symbols are exported the same way (section 10).

# 3. Measurements

Scratch programs linked against the library at `34e675e`, single thread, idle machine (load average about 2 on 32 hardware threads). Pattern: `generate_arrayant_3GPP(2, 2, 3 GHz, pol 3)`, 8 elements, 1 degree grid; 20 000 random angles per element (per-element angles), random per-element orientation, fixed seed. One pattern and one seed: the numbers show orders of magnitude, not bounds.

Error against the double-precision GENERIC kernel (maximum absolute value; peak field magnitude 2.4):

| Output | AVX2, double | GENERIC, float | AVX2, float |
| :-- | --: | --: | --: |
| `V_re`, `H_re` | 2.1e-6, 2.5e-6 | 3.5e-6, 5.7e-6 | 5.4e-5, 5.4e-5 |
| `dist` in m | 2.4e-8 | 1.9e-8 | 2.3e-8 |
| `azimuth_loc`, `gamma` in rad | 7.4e-5 | 3.9e-5 | 7.4e-5 |
| `elevation_loc` in rad | 2.1e-5 | 2.9e-5 | 2.1e-5 |

(`V_im`, `H_im` are zero for this pattern.)

CPU time per call: GENERIC 92 ms (double) and 67 ms (float); AVX2 11 ms and 4.4 ms. AVX2 is about 8x faster for double, 15x for float. The four optional outputs add 6 % (GENERIC) and 12 % (AVX2).

Per-call overhead of validation, measured for the smallest calls in the hot loop (32 elements, one angle, double): kernel alone 9.9 us; member (validation, index conversion, position copy) 10.0 us when the arrayant passes the quick check, 10.8 us when it falls back to the deep check. `is_valid()`: quick check 8.5 ns, deep check 0.9 us. The quick check passes only after `validate()` stored the data pointers; `is_valid` itself is `const` and writes nothing, so it is safe inside OpenMP regions.

Grids changed in place after `validate()` (the quick check still passes): reversed, NaN entries, values far out of range, constant grids. The two kernel sources were compiled with AddressSanitizer and UndefinedBehaviorSanitizer (the rest of the library without) and called with random angles including NaN and infinity, `float` and `double`: no report, both returned.

# 4. Decisions

## 4.1 Settled with the scope

| ID | Decision | Choice | Reason |
| :-- | :-- | :-- | :-- |
| D1 | Where the plan lives | This document, tracked in `md/`, deleted at the end | reference beyond the commit messages, kept in the history |
| D2 | Input of the public function | `const arrayant<dtype> &` | every caller holds an `arrayant`; `_multi` takes `std::vector<arrayant>`; raw cubes would move arrayant validation to the caller |
| D3 | Member `arrayant::interpolate` | removed | its signature changes anyway, so keeping it saves no caller an edit; one entry point, one doc block, one test suite |
| D4 | Kernel layout | as `ray_triangle_intersect` (`md/SPEC_CPP_API.md`, "AVX2 and CUDA kernels") | one validation path; the overhead of always going through the public function is at most 9 % on the smallest calls (section 3) |
| D5 | Default kernel | `use_kernel = 0` (auto) selects AVX2 when available, also for `double`; `use_kernel = 1` gives double precision and is documented as such | patterns are stored at about float precision (QDANT), and AVX2 is about 8x faster for double; bit-identical results are not required. Tests that break are handled per case: either the tolerance is relaxed or the test selects kernel 1 (step 5, a stop point) |
| D7 | Passing `i_element` | `const arma::uvec *i_element = nullptr` | SPEC rule for optional inputs |
| D8 | Empty optional inputs | `nullptr` and an empty object both mean "not provided" | matches `combine_pattern` and the wrappers, which pass empty objects |
| D10 | `arrayant_interpolate_multi` | migrated in this refactor, moved into `src/arrayant_interpolate.cpp` | one change to the interpolation API; the SPEC lets the variants of one function share a file |
| D11 | Kernel argument of `get_channels_spherical` | `int use_kernel = 0` replaces `bool use_avx2 = false` | the same selector as everywhere else |
| D12 | Wrappers | MEX and Python wrappers of the changed functions follow their SPECs and are updated in this refactor | the whole interpolation API changes in one go |
| D13 | Tests | rewritten to `md/SPEC_CPP_API.md`, "TESTS", with a GENERIC-versus-AVX2 comparison | the existing suites predate the SPEC; existing cases serve as sources of oracles |

## 4.2 Settled in review

These points had options; the recommended option was accepted in review. The analysis is kept as the reason.

**D6 Width of the element index.** Options: (a) 64-bit `arma::uvec` in the public functions and the kernels; (b) 32-bit (`arma::Col<unsigned>`) in the kernels. With (a), no caller converts anything: all public arrayant functions take `uvec` (`copy_element`, `export_obj_file`, `arrayant_interpolate_multi`, `arrayant_rotate_pattern_multi`), the MEX wrapper already produces `uvec` (`qd_mex_typecast_Col<arma::uword>`), and `get_channels_*` build their per-link index directly as `uvec`. With (b), the public functions and `_multi` keep a copy-and-narrow step, as today. The AVX2 kernel needs int32 offsets either way; it computes one per output element (`n_out` values, not per angle), so (a) costs nothing measurable there. Memory: the index has `n_out` entries (per link in `get_channels_*`, e.g. 4096 for 64 x 64), a few tens of kB at most. Safety: a 64-bit index narrowed to 32 bits before the bounds check wraps into the valid range (`2^32` becomes `0`); (a) has no narrowing step, (b) must keep the check before the conversion. **Decision: (a).**

**D9 A `validate_input` flag.** Options: (a) no flag; every call validates (shapes, indices, aliasing, `is_valid()` of the arrayant); `_multi` loses its existing flag. (b) a flag as `_multi` has today. Implications:
- Cost: section 3 shows at most 9 % on the smallest calls in the hot loop, nothing measurable on large calls.
- Memory safety: under D4 the kernels have no checks of their own. With (b), `validate_input = false` with inconsistent data (e.g. `e_theta_im` smaller than `e_theta_re`, an element index out of range, `element_pos` of the wrong size) reads out of bounds. That is undefined behaviour, reachable whenever pattern data comes from a file (QDANT, HDF5) or from MATLAB/Python. Today `_multi` with the flag off is still protected by the kernel's own checks; after D4 it would not be. The `arrayant` members are public and mutable, so data checked once can change before the call.
- Remaining gap with (a): the quick check accepts an arrayant whose grid values were changed in place after `validate()`. Section 3 found no out-of-bounds access for that case in either kernel; it stays covered by the fuzz target, which already does such in-place changes (`tests/fuzzing/fuzz_target.hpp:241-249`).
**Decision: (a).**

**D14 Kernel used by library functions that have no `use_kernel` argument.** Concerns `combine_pattern`, `rotate_pattern` (and through them all `generate_arrayant_*` and the `_multi` variants), `get_channels_planar`, `get_channels_multifreq`, and the calls of `get_channels_spherical` inside `get_channels_irs` and `gbsm_single_link_channel`. Options: (a) they pass `use_kernel = 1`; their results stay as they are; a `use_kernel` argument for them is a later, per-function API decision. (b) they use auto; they become faster but give single-precision results with no way back to double precision, and every generated pattern changes (`generate_arrayant_*` interpolates through `rotate_pattern` and `combine_pattern`). (c) they all get a `use_kernel` argument now, which adds eight public functions and their wrappers to this refactor. D5 says double precision must be reachable by choosing kernel 1; (b) breaks that for these functions. **Decision: (a).** Note for `get_channels_multifreq`: its test compares against `get_channels_spherical` at 5e-6 (`test_get_channels_multifreq.cpp:256-257`); with (a) that test must call `get_channels_spherical` with `use_kernel = 1`.

**D15 Small contract points.** Written into section 5:
- (a) Parameter name of the arrayant: `ant`. A parameter named `arrayant` hides the class template inside the function, which turns any later use of `arrayant<dtype>` in the body into an error.
- (b) `use_kernel = 2` when AVX2 is unavailable, or the pattern has more than `INT_MAX` samples: throws `std::invalid_argument`. Auto selects GENERIC in both cases.
- (c) Empty angle input: `azimuth` with `n_ang = 0` columns gives outputs of size `[n_out, 0]`; zero rows is an error.
- (d) The alias check covers the arrayant's own data members besides the other inputs.

**W1 to W7 Wrapper points.** Section 7.

## 4.3 Single document

This document combines plan and spec. A separate spec would pay off if the contract were reviewed by someone other than the implementer before the plan, or if the contract were large; here it is three declarations and two internal kernels, and most decisions tie a contract point to an implementation step. Section 5 is the contract and is not changed by the implementer (`md/SPEC_CPP_API.md`, "CONTRACT").

# 5. Contract

Normative. Written for the decisions of section 4 and 7.2.

## 5.1 `arrayant_interpolate`

Header `include/quadriga_arrayant.hpp`, directly before `arrayant_interpolate_multi`. Source `src/arrayant_interpolate.cpp`, section "Array antenna functions".

```
// Interpolate the polarimetric field pattern of an array antenna at given azimuth and elevation angles
template <typename dtype>
void arrayant_interpolate(const arrayant<dtype> &ant,                     // Array antenna with n_elements elements
                          const arma::Mat<dtype> &azimuth,                // Azimuth angles in GCS in rad; [1, n_ang] or [n_out, n_ang]
                          const arma::Mat<dtype> &elevation,              // Elevation angles in GCS in rad; same size as azimuth
                          arma::Mat<dtype> &V_re,                         // Output: e-theta (vertical) field, real part; [n_out, n_ang]
                          arma::Mat<dtype> &V_im,                         // Output: e-theta (vertical) field, imaginary part; [n_out, n_ang]
                          arma::Mat<dtype> &H_re,                         // Output: e-phi (horizontal) field, real part; [n_out, n_ang]
                          arma::Mat<dtype> &H_im,                         // Output: e-phi (horizontal) field, imaginary part; [n_out, n_ang]
                          const arma::uvec *i_element = nullptr,          // Optional: 0-based element indices, duplicates allowed; [n_out]
                          const arma::Cube<dtype> *orientation = nullptr, // Optional: Euler angles (bank, tilt, heading) in rad; [3, 1, 1], [3, n_out, 1], [3, 1, n_ang] or [3, n_out, n_ang]
                          const arma::Mat<dtype> *element_pos = nullptr,  // Optional: element positions in m, replacing ant.element_pos; [3, n_out]
                          arma::Mat<dtype> *dist = nullptr,               // Optional output: distance of each element from the wavefront plane through the origin in m; [n_out, n_ang]
                          arma::Mat<dtype> *azimuth_loc = nullptr,        // Optional output: azimuth in the local antenna frame in rad; [n_out, n_ang]
                          arma::Mat<dtype> *elevation_loc = nullptr,      // Optional output: elevation in the local antenna frame in rad; [n_out, n_ang]
                          arma::Mat<dtype> *gamma = nullptr,              // Optional output: polarization rotation angle in rad; [n_out, n_ang]
                          int use_kernel = 0);                            // Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2
```

Behaviour:
- `n_out` is the length of `i_element`; when `i_element` is `nullptr` or empty, all elements are used in order and `n_out = n_elements`.
- `azimuth` with one row gives the same angles for all elements (planar wave); `n_out` rows give one angle per element (spherical wave). `elevation` has the size of `azimuth`. `n_ang = 0` gives outputs of size `[n_out, 0]`.
- `orientation` `nullptr` or empty: no rotation. `element_pos` `nullptr` or empty: the columns `i_element` of `ant.element_pos`, or zeros when `ant.element_pos` is empty.
- NaN or infinite angles give NaN in all outputs of that angle and element; no exception. Angles are not range-checked; values outside [-pi, pi] / [-pi/2, pi/2] behave as today (pinned by a test, section 8).
- Mandatory outputs are resized only when their size differs and are completely written. A non-null optional output is computed and resized in the same way; `nullptr` skips it.
- `use_kernel`: 0 selects AVX2 when `quadriga_lib_has_AVX2()` is true and the pattern has at most `INT_MAX` samples, else GENERIC; 1 selects GENERIC; 2 selects AVX2 and throws when it is unavailable or the pattern is too large; other values throw. The AVX2 kernel computes in single precision also for `double`; the documentation states this with the order of the errors in section 3 and names `use_kernel = 1` as the way to double precision.
- Validation, all before any output is written, all `std::invalid_argument` with messages in the SPEC style: `ant.is_valid()`; `azimuth` / `elevation` sizes; `i_element` entries `< n_elements`; `orientation` shape; `element_pos` shape; `use_kernel` value and availability; aliasing of any output with an input, with a data member of `ant`, or with another output (`qd_check_alias_list`).

## 5.2 `arrayant_interpolate_multi`

Same header, directly after `arrayant_interpolate`; moved to `src/arrayant_interpolate.cpp`.

```
// Interpolate a multi-frequency array antenna at given angles and frequencies
template <typename dtype>
void arrayant_interpolate_multi(const std::vector<arrayant<dtype>> &arrayant_vec, // Multi-frequency arrayant vector; entries need not be sorted by frequency
                                const arma::Mat<dtype> &azimuth,                  // Azimuth angles in rad; [1, n_ang] or [n_out, n_ang]
                                const arma::Mat<dtype> &elevation,                // Elevation angles in rad; same size as azimuth
                                arma::Cube<dtype> &V_re,                          // Output: e-theta field, real part; [n_out, n_ang, n_freq]
                                arma::Cube<dtype> &V_im,                          // Output: e-theta field, imaginary part; [n_out, n_ang, n_freq]
                                arma::Cube<dtype> &H_re,                          // Output: e-phi field, real part; [n_out, n_ang, n_freq]
                                arma::Cube<dtype> &H_im,                          // Output: e-phi field, imaginary part; [n_out, n_ang, n_freq]
                                const arma::Col<dtype> *frequency = nullptr,      // Optional: target frequencies in Hz; default: the center frequencies of the entries; [n_freq]
                                const arma::uvec *i_element = nullptr,            // Optional: 0-based element indices; [n_out]
                                const arma::Cube<dtype> *orientation = nullptr,   // Optional: Euler angles in rad, applied at all frequencies; shapes as in arrayant_interpolate
                                const arma::Mat<dtype> *element_pos = nullptr,    // Optional: element positions in m; default: those of entry 0; [3, n_out]
                                int use_kernel = 0);                              // Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2
```

- The frequency blending (bracketing entries, SLERP with linear fallback, clamping, caching of consecutive brackets) is unchanged.
- `validate_input` is removed (D9); `arrayant_is_valid_multi(arrayant_vec, true)` runs on every call.
- `frequency` becomes optional (W2 (a)): `nullptr` or empty means the center frequencies of the entries, in vector order, so `n_freq` equals the number of entries. This default moves from both wrappers into C++. - It calls the kernels through the same file-local dispatch as `arrayant_interpolate`, after validating once, not the public function per entry.

## 5.3 `get_channels_spherical`

- The last argument `bool use_avx2 = false` becomes `int use_kernel = 0` with the values of section 5.1. It selects the kernel for the antenna interpolation and for the coefficient combination (`qd_coeff_combine` / `qd_coeff_combine_avx2`).
- Hazard: `false` and `true` convert silently to 0 (auto) and 1 (GENERIC), the opposite of their meaning today, and the compiler does not warn. All six call sites in section 2.2 are edited explicitly to 1 (former `false`) or 2 (former `true`), and a search for `use_avx2` in the tree must come back empty.
- Behaviour change: `use_avx2 = true` was ignored where AVX2 is unavailable; `use_kernel = 2` throws there. Tests that request kernel 2 run only where `quadriga_lib_has_AVX2()` is true.
- `get_channels_irs` and `gbsm_single_link_channel` pass `use_kernel = 1` (D14 (a)).

## 5.4 Internal kernels

- `qd_arrayant_interpolate_GENERIC<dtype>` in `src/arrayant_interpolate_generic.cpp`, declared in `src/quadriga_lib_generic_functions.hpp`; `qd_arrayant_interpolate_AVX2<dtype>` in `src/arrayant_interpolate_avx2.cpp`, declared in `src/quadriga_lib_avx2_functions.hpp`, compiled only with AVX2 (file suffix).
- Arguments as the current kernels (Armadillo references for the pattern data, grids, angles, orientation, positions and mandatory outputs; pointers for the optional outputs), with `const arma::uvec &i_element` 0-based (D6 (a)).
- Preconditions, stated in the header comment and not checked: all inputs validated by the caller; `orientation` non-empty with a valid shape; `element_pos` of size `[3, n_out]`; outputs pre-sized; optional outputs `nullptr` or pre-sized; for AVX2, at most `INT_MAX` pattern samples.
- The kernels use `qd_rotation_matrix` from `src/quadriga_lib_helper_functions.hpp` instead of their local copies, after a diff shows that the copies compute the same matrix (`invert_y_axis = true`, `transposeR = true` as called today).
- The computation and the OpenMP regions are moved unchanged (GENERIC `if (n_ang > 1 && n_ang * n_out >= 256)`, AVX2 regions as they are); `md/SPEC_CPP_API.md` asks for measured limits only for new or changed regions.

## 5.5 Removed

`arrayant<dtype>::interpolate` (declaration, definition, documentation block), `src/qd_arrayant_interpolate.hpp`, `src/qd_arrayant_interpolate.cpp`, `src/qd_arrayant_interpolate_avx2.hpp`, `src/qd_arrayant_interpolate_avx2.cpp` (replaced by `src/arrayant_interpolate_avx2.cpp`), `src/qd_arrayant_xinterpol_multi.cpp` (it holds only `arrayant_interpolate_multi`, which moves).

# 6. Implementation steps

Test-first as in `md/SPEC_CPP_API.md`, "IMPLEMENTATION ORDER". Build out of tree; the in-repository `build_linux` is not used. Run the C++ tests with `OMP_WAIT_POLICY=passive`. Each step ends with a build without warnings.

**Step 0 — Baseline.** Build at the starting commit and run the complete C++ suite; record the result. Diff the two local `qd_rotation_matrix` copies against the shared helper.

**Step 1 — Stubs.** Declarations of section 5.1 and 5.2 in the header (old declarations still present), `src/arrayant_interpolate.cpp` with license, includes, section statement, first documentation blocks, stub bodies that size all outputs and fill zeros, and explicit instantiations.

**Step 2 — Tests against the stubs.** New `tests/catch2_tests/test_arrayant_interpolate.cpp` and `test_arrayant_interpolate_multi.cpp` (section 8). Run them against the stubs, evaluate the oracles, write the test record. **Stop point 1: the maintainer reviews the test record.**

**Step 3 — Kernels and implementation.** Move the kernels (section 5.4), write the file-local dispatch, implement both public functions, iterate until the new tests pass.

**Step 4 — Callers and removal.** Move every caller of section 2.2 to `arrayant_interpolate` (D14 (a): `use_kernel = 1` where the caller has no selector); change `get_channels_spherical` (section 5.3) and edit its six call sites; remove what section 5.5 lists; update the fuzz targets `fz_arrayant_interpolate.cpp` (new functions, random `use_kernel` among the available kernels) and `fz_get_channels.cpp`, and the table in `tests/fuzzing/README.md`.

**Step 5 — Full suite with the new default.** Run the complete suite. For every failure, record test, quantity, observed difference and tolerance, and propose (a) a tolerance derived from the kernel accuracy or (b) `use_kernel = 1` in that test, with the reason. Expected candidates (tolerances below about 1e-5 on interpolated values): `test_get_channels_spherical.cpp` (1e-6 on coefficients in many cases, `coeff_im < 1e-14` at `:660`), `test_get_channels_multifreq.cpp:256-257`, and the arrayant suites that compare against hand values. **Stop point 2: the maintainer decides per test.**

**Step 6 — Wrappers.** MEX and Python wrappers and their tests (section 7). Run `make test` with the MEX suites, and the Python suite (`python -m pytest tests/python_tests`, with the built module in `lib/`).

**Step 7 — Finish.** `make docs`; update the `arrayant_interpolate` row of `docs_builder/content/api_matching.toml` to `["arrayant_interpolate", "arrayant_interpolate_multi"]`; complete suite; AddressSanitizer run (`make test-asan`) of the changed test cases; a short fuzz run of the two fuzz targets (`make test-fuzz FUZZ_TIME=2m FUZZ_ARGS='--targets fz_arrayant_interpolate'`, same for `fz_get_channels`); MSVC check by reading (no GCC-only extensions); hand-over report (`md/SPEC_CPP_API.md`, "HAND-OVER REPORT"). **Stop point 3: review of the hand-over.**

Commit messages follow `CONTRIBUTING.md`; the maintainer commits.

# 7. Wrappers

Facts from the review of the current wrappers (line numbers at `34e675e`) and the points to decide. Mechanical SPEC fixes are listed as settled.

## 7.1 Settled

- MEX `arrayant_interpolate` (`api_mex/arrayant_interpolate.cpp`): calls `quadriga_lib::arrayant_interpolate` / `arrayant_interpolate_multi`; `nullptr` for `orientation`, `element_pos` and optional outputs that are absent or not requested (`md/SPEC_MATLAB_API.md`:352-355); `var_name` in all argument readers; arguments are parsed and C++ is called also when `nlhs == 0`, so that errors are raised; documentation to the SPEC (arrayant phrase, `default:` on every optional argument, standard headings).
- Python `arrayant_interpolate` (`api_python/qpy_arrayant_interpolate.cpp`, registered in `api_python/python_arrayant.cpp:104-115`): `azimuth` and `elevation` as `py::array_t` with `.none(false)`; flags read with `qd_python_flag`; `complex` last; the comment block at the end of the file identical to the registration and naming the right file.
- Both: a `use_kernel` argument, default 0, after the existing arguments and, in Python, before `complex`; documented as in the `sos_val` wrappers, plus the precision note of section 5.1.
- `get_channels_spherical`: MEX input 15 `use_avx2` becomes `use_kernel` (int, default 0, read with `qd_mex_get_scalar<int>(..., "use_kernel", 0)`); Python gets `use_kernel` and passes it.
- Tests: `tests/mex_tests/test_arrayant_interpolate.m` and `tests/python_tests/test_arrayant_interpolate.py` rewritten to the SPEC test rules (MATLAB: header comment, `%%` sections, checklist; Python: one `test_case` class, keyword-argument, memory layout, dtype, inputs-not-modified, exception types), including `use_kernel`; the `get_channels_spherical` tests in both languages gain `use_kernel` cases. The Python test file imports the module as in the SPEC template (`import quadriga_lib`, `md/SPEC_PYTHON_API.md`:536) instead of `from quadriga_lib import arrayant` (`:17`). Python tests that pass `dist=1` / `local_angles=1` as integers are changed to booleans, since `qd_python_flag` rejects integers.

## 7.2 Settled in review

These points had options; the recommended option was accepted in review.

**W1 MEX separate-field input mode** (13 inputs, `api_mex/arrayant_interpolate.cpp:162-170`). `md/SPEC_MATLAB_API.md`:261 calls it legacy. Options: remove (MATLAB users of this form change to the struct) or keep. **Decision: remove.**

**W2 Frequency default.** Both wrappers fill the target frequencies from the entries' center frequencies when none are given (MEX `:130-135`, Python `:131-139`), a default that `md/SPEC_MATLAB_API.md`:56 places in C++. Options: (a) `frequency` optional in C++ as in section 5.2; (b) keep it in the wrappers as a documented exception. **Decision: (a).**

**W3 Single- versus multi-frequency dispatch in the wrappers.** The wrappers choose between the two C++ functions from the input (struct array, frequency given). `md/SPEC_MATLAB_API.md`:55 allows one MATLAB function dispatching to several C++ functions when the plan settles it. **Decision: keep the dispatch.** The selection changes no numbers; the computation stays in C++.

**W4 Python flags `dist` and `local_angles`.** They change the returned tuple, which `md/SPEC_PYTHON_API.md`:330 and :451 allow only by decision. Options: (a) keep them as a decided exception; (b) always return all eight outputs in the single-frequency mode. The optional outputs cost 6 % (GENERIC) to 12 % (AVX2) (section 3), so cost is not the argument; but the multi-frequency mode cannot produce them, so with (b) the tuple length would depend on the mode, and every current Python caller that unpacks four values breaks. **Decision: (a).**

**W5 Argument names.** Python uses `frequency`, MEX `freq`; Python documents the outputs as `vr, vi, hr, hi`, MEX as `V_re, ...`. `md/SPEC_PYTHON_API.md`:489 takes MEX as the reference. Options: rename the MEX argument in its documentation to `frequency` (the C++ name; MEX arguments are positional, so nothing breaks) and document the Python outputs as `V_re, V_im, H_re, H_im`; or rename the Python keyword to `freq` (breaks keyword callers). **Decision: the first.**

**W6 Python `fast_access`.** Python-only, maps to the strict flag of the dictionary conversion (`:70-72`, `:119`). Not part of the interpolation contract. **Decision: keep unchanged.**

**W7 Remaining deviations of the `get_channels_spherical` wrappers** (MEX `var_name`, documentation wording; Python plain `bool` flags, the non-standard `angles` flag, the batch mode with OpenMP and released GIL). Not interpolation API. **Decision: out of scope; listed in section 10.**

# 8. C++ tests

Rules: `md/SPEC_CPP_API.md`, "TESTS". Case names `"arrayant_interpolate - ..."` and `"arrayant_interpolate_multi - ..."`. Oracles independent of the library's interpolation code; supporting tasks (pattern generation, rotation matrices) use library calls.

`test_arrayant_interpolate.cpp` covers at least:
- Values against closed-form oracles, for GENERIC (`use_kernel = 1`, tolerances near double precision) and for AVX2 (tolerances from single-precision arithmetic, stated in a comment): grid-point reproduction; interpolation along azimuth and elevation on small grids (the hand values of the current cases at `test_arrayant_interpolate.cpp:14-257`); z- and y-rotation shifting the local angles; x-rotation turning polarization; projected distance (`:292`).
- Every argument: `i_element` omitted, permuted, with duplicates; the four `orientation` shapes; `element_pos` given, omitted with `ant.element_pos` set, omitted with `ant.element_pos` empty; planar and per-element angles; `n_ang = 0`.
- Optional outputs: each alone and all together give the same values; omitted outputs are not touched.
- Pre-sized outputs are written in place (data pointer unchanged); outputs of wrong size or with stale content are resized and fully overwritten.
- NaN and infinite angles give NaN, for both kernels; angles outside the nominal ranges behave as with GENERIC today.
- GENERIC versus AVX2 on a realistic pattern (a 3GPP panel, per-element angles and orientation, a size that crosses the AVX2 vector width and the OpenMP limits): all outputs agree within tolerances derived from single-precision arithmetic, with a diagnostic table (`QD_DIAG`) of the differences.
- `float` and `double`: both instantiations, and a parity case.
- Determinism: repeated calls identical; a large call (OpenMP active) equals the concatenation of single-angle calls, per kernel.
- `use_kernel`: 0 equals 2 when AVX2 is available, else 1; 2 throws when unavailable (case runs only where AVX2 is missing); values outside 0 to 2 throw.
- `"arrayant_interpolate - input validation"`: valid baseline with `CHECK_NOTHROW`, one section per invalid argument, including aliasing with an input, with a member of `ant`, and between outputs.

`test_arrayant_interpolate_multi.cpp`: the existing 48 cases (`test_arrayant_interpolate_multi.cpp`) are the source of the oracles (frequency bracketing, SLERP and linear fallback, clamping, caching, independence of the entry order); they are rewritten to the new signature and the SPEC structure, with comparisons against `arrayant_interpolate` instead of the member, the separate validation cases merged into one validation case with a baseline, the `validate_input` cases removed, and cases for omitted `frequency` and for `use_kernel`.

Other suites: calls of the member in `test_arrayant.cpp` move to `arrayant_interpolate`; `use_avx2` in `test_get_channels_spherical.cpp` becomes `use_kernel` with the meaning kept (1 for the former `false`, 2 for the former `true`); step 5 decides on the tolerance failures.

# 9. Documentation

- Documentation blocks of `arrayant_interpolate` and `arrayant_interpolate_multi` to `md/SPEC_CPP_API.md`, "Documentation block": what the function does and is for, data layout, sizes, units, defaults, the kernel choice with its precision, the errors a caller can trigger; no implementation details. The member's block is the starting point for content; its Example is kept in the new call form.
- Links: the class table in `src/qd_arrayant.cpp:67`, the "See also" of `combine_pattern` (`:525`) and of `_multi` point to `[[arrayant_interpolate]]`; the header comment of `_multi` no longer names `qd_arrayant_interpolate`.
- `get_channels_spherical`: argument entry for `use_kernel`, naming both kernels it selects.
- Generated files in `docs/` are regenerated with `make docs` and committed with the change. The `website_builder` test fixtures use `.interpolate` as sample data of a mini repository and are not changed.

# 10. Follow-ups, not in this refactor

- `use_kernel` for `combine_pattern`, `rotate_pattern`, `get_channels_planar`, `get_channels_multifreq` and the functions built on them (D14), as part of the 1.0 API cleanup.
- The remaining SPEC deviations of the `get_channels_spherical` wrappers (W7).
- Hidden symbol visibility for the shared library and the Python module, so that internal `qd_*` symbols are not exported.
