# Arrayant interpolation refactor

This document plans and specifies the refactor of the antenna pattern interpolation API before the 1.0 release. It records what changes, why, and how the result is checked, so that an implementer can act on it without further context and a later review can trace each change to its reason. It is a working document: it is deleted when the refactor is complete and remains available in the git history. Section 4 lists the decisions and their reasons, including those taken after an independent review of the first version (section 4.3); all are settled. Sections 5 to 9 are written for the settled state.

Rules that apply throughout: `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`, `CONTRIBUTING.md`. Source code, tests and commit messages do not refer to this document (`md/SPEC_CPP_API.md`, "GENERAL CONVENTIONS").

# 1. Purpose and scope

- Replace the member function `arrayant<dtype>::interpolate` and the internal function `qd_arrayant_interpolate` by one public, documented free function `quadriga_lib::arrayant_interpolate`, placed directly before `arrayant_interpolate_multi`.
- Move the interpolation kernels to the layout used by the other accelerated functions (public function validates and dispatches; kernels `_GENERIC` / `_AVX2` without own checks), with a `use_kernel` argument.
- Migrate `arrayant_interpolate_multi` to the same argument conventions, and replace `bool use_avx2` in `get_channels_spherical` by `int use_kernel`.
- Move all library callers to the new function, delete the internal headers and sources, and bring the MEX and Python wrappers of these functions and their tests in line with their SPECs.
- Results do not change: the default kernel is GENERIC for every caller (D5). The AVX2 kernel stays available on explicit request.
- API changes are intended: the interpolation API is changed once now so that it stays stable after 1.0.

Not in scope (section 10): fixing the AVX2 kernel (it has a known error at the poles, section 2.1); deciding when auto selects AVX2; the remaining SPEC deviations of the `get_channels_spherical` wrappers other than the kernel argument; symbol visibility of the shared libraries; the weak-symbol hazard of the AVX2 objects (section 2.6).

# 2. Current state

Line numbers refer to commit `34e675e`.

## 2.1 Functions and files

| Item | Where | Notes |
| :-- | :-- | :-- |
| `arrayant<dtype>::interpolate` | declared `include/quadriga_arrayant.hpp:77`, defined `src/qd_arrayant.cpp:1218` | public member; mandatory arguments by pointer; validates, resizes outputs, converts `i_element` to 1-based `unsigned`, copies element positions, calls `qd_arrayant_interpolate` |
| `qd_arrayant_interpolate` | `src/qd_arrayant_interpolate.hpp`, `src/qd_arrayant_interpolate.cpp` | internal scalar kernel; validates again with its own messages; requires pre-sized outputs; treats a non-null but empty optional output as "not requested" |
| `qd_arrayant_interpolate_avx2` | `src/qd_arrayant_interpolate_avx2.hpp`, `src/qd_arrayant_interpolate_avx2.cpp` | internal AVX2 kernel; computes in single precision for `float` and `double`; repeats the checks; calls the scalar kernel when `e_theta_re.n_elem > INT_MAX` (int32 gather offsets, `:668`); own copy of `qd_rotation_matrix` |
| `arrayant_interpolate_multi` | `include/quadriga_arrayant.hpp:160`, `src/qd_arrayant_xinterpol_multi.cpp:82` | mandatory arguments by pointer, `i_element` by value, `bool validate_input`; calls the scalar kernel per frequency entry |
| `get_channels_spherical` | `include/quadriga_arrayant.hpp:356`, `src/get_channels_spherical.cpp` | `bool use_avx2 = false` selects the AVX2 interpolation kernel (`:523`) and the AVX2 coefficient kernel `qd_coeff_combine_avx2` (`:578`) |

The single-precision AVX2 design is deliberate: the write-up of its last refactor states that both instantiations run the same float path and that double-precision results come from the scalar reference (`tests/benchmarks_cpp/REFACTORING_RESULTS_ARRAY_INTERP.md` at commit `ebe677b`, removed in `67d09b3`). `qd_coeff_combine_avx2<double>` computes the Jones product in float and the phase in double (`src/qd_coeff_combine_avx2.cpp:242-251`).

Known defect of the AVX2 interpolation kernel: at exactly ±90° elevation without antenna rotation, it returns about 0.40 times the power of the GENERIC kernel (crossed dipoles: 0.401 against 1.000; at 90° − 1e-7 rad both agree; with a tilt of 0.4 rad both agree). Through `get_channels_spherical` with `use_avx2 = true` and the receiver directly above the transmitter, the received power is 0.161 instead of 1.000. The cause is not analysed. This refactor does not fix or test it; it is the subject of a follow-up issue (section 10).

## 2.2 Callers of the internal kernels and the member

- `combine_pattern` (`src/qd_arrayant.cpp:581`), `rotate_pattern` (`:1944`, `:1952`). Both are used by `generate_arrayant_*` (`src/qd_arrayant_generate.cpp:605-611`, `657`, `1014-1024`, `1071`) and by `arrayant_combine_pattern_multi` / `arrayant_rotate_pattern_multi`. `rotate_pattern` relies on "empty optional output = not requested".
- `calc_beamwidth_deg` (`src/qd_arrayant.cpp:310`, `this->interpolate`) and `export_obj_file` (`:959`, unqualified `interpolate`): member callers inside the class.
- `arrayant_interpolate_multi` (`src/qd_arrayant_xinterpol_multi.cpp:231`, `242`).
- `get_channels_planar` (`src/get_channels_planar.cpp:294`, `308`); itself called by `gbsm_single_link_channel` and `get_channels_ieee_indoor`. It passes empty `AOA_loc` / `EOA_loc` when no Doppler output is requested (`:273`, `:308`), relying on "empty optional output = not requested".
- `get_channels_spherical` (`src/get_channels_spherical.cpp:521-554`); itself called by `get_channels_irs` (`src/get_channels_irs.cpp:196`, `201`, `247`, `252`) and `gbsm_single_link_channel` (`src/gbsm_single_link_channel.cpp:376`), both with the default `use_avx2 = false`.
- `get_channels_multifreq` (`src/get_channels_multifreq.cpp:788-857`): up to six calls per path, inside the OpenMP frequency loop (`:601-995`), with `n_links x 1` angles each.
- Member callers outside the library: Python wrapper (`api_python/qpy_arrayant_interpolate.cpp:195-200`), MEX wrapper (`api_mex/arrayant_interpolate.cpp:197-202`), fuzz target (`tests/fuzzing/targets/fz_arrayant_interpolate.cpp:79-82`), 33 calls in `test_arrayant_interpolate.cpp`, `test_arrayant.cpp`, `test_arrayant_interpolate_multi.cpp`.
- Explicit `use_avx2` arguments: `tests/catch2_tests/test_get_channels_spherical.cpp:1943`, `1946`, `1982`, `1985`; `tests/fuzzing/targets/fz_get_channels.cpp:78-82`; `api_mex/get_channels_spherical.cpp:142`. The Python wrapper of `get_channels_spherical` does not pass it.
- A comment naming `qd_arrayant_interpolate`: `tests/catch2_tests/test_arrayant_combine_pattern_multi.cpp:745`.

## 2.3 Index convention

The public functions take 0-based `arma::uvec`. The member and `_multi` copy it into a 1-based `arma::Col<unsigned>`; `get_channels_*` build 1-based `Col<unsigned>` per link. Both kernels reject 0 and subtract 1 immediately (`qd_arrayant_interpolate.cpp:308`, `qd_arrayant_interpolate_avx2.cpp:867`). The 1-based form has no function: 0 is never a sentinel.

## 2.4 Symbol exposure

`qd_arrayant_interpolate<float|double>` and `qd_arrayant_interpolate_avx2<float|double>` are exported (weak symbols) from the Python module and the static library, without a public declaration. This is not a privilege boundary (code that links the library can call any exported symbol), but it is an undocumented interface with its own, second set of checks. All internal `qd_*` symbols are exported the same way (section 10).

## 2.5 The validation cache

`is_valid()` (quick check) compares the sizes of the members and the data pointers stored by `validate()` (`src/qd_arrayant.cpp:2172-2197`); if they match, it returns at once, otherwise it runs the deep check. `is_valid` is `const` and writes nothing, so it is safe inside OpenMP regions. The cache is stale after `copy()` (`:659`, member-wise assignment), after the implicit copy constructor (`push_back`, `std::vector<arrayant> v = {a, a}`, passing by value), and for arrayants that never went through `validate()`. Of the wrappers, only the Python adapter calls `validate()` (`api_python/python_quadriga_adapter.hpp:167`); the MEX conversion does not.

## 2.6 AVX2 code at runtime

`runtime_AVX2_Check()` (`src/runtime_AVX2_Check_avx2.cpp`) tests AVX, FMA and AVX2 with CPUID and the OS support for the YMM state with XGETBV, and caches the result in a function-local static. Its machine code contains no AVX instructions, although the file is compiled with `-mavx2`. All AVX2 kernels are called behind this check or `quadriga_lib_has_AVX2()`, which calls a test kernel on every call (`src/quadriga_lib.cpp:64-99`).

Hazard not covered by the runtime check: template and inline functions that an AVX2 translation unit uses are emitted there as weak symbols with AVX instructions, and the linker keeps one copy for all callers. In the current build, one such function exists: `std::vector<std::vector<unsigned long long>>::_M_realloc_insert` in `ray_point_intersect_avx2.cpp.o` (contains `vmovdqu`, `vpxor`); the same symbol is defined in `ray_point_intersect_generic.cpp.o` and `obj_file_write.cpp.o`. The Python module and `test_bin` contain the copy without AVX, because the generic object comes earlier in the link order. This is safe by link order, not by design: a program that links `libquadriga.a` in another order could crash on a CPU without AVX2. The weak AVX2 kernel instantiations themselves (`qd_DFT_AVX2`, the interpolation kernel) are called only behind the runtime check. Recorded as an issue at the end of the refactor (step 8).

# 3. Measurements

Scratch programs linked against the library at `34e675e`, single thread, idle machine (load average 1 to 2.3 on 32 hardware threads), unless noted. The accuracy and the large-call times use `generate_arrayant_3GPP(2, 2, 3 GHz, pol 3)` (8 elements, 1° grid), 20 000 random angles per element (per-element angles), random per-element orientation, fixed seed. Random angles never hit a pole exactly, so the error table holds away from the poles only (section 2.1). One pattern and one seed: the numbers show orders of magnitude, not bounds.

Error against the double-precision GENERIC kernel (maximum absolute value; peak field magnitude 2.4):

| Output | AVX2, double | GENERIC, float | AVX2, float |
| :-- | --: | --: | --: |
| `V_re`, `H_re` | 2.1e-6, 2.5e-6 | 3.5e-6, 5.7e-6 | 5.4e-5, 5.4e-5 |
| `dist` in m | 2.4e-8 | 1.9e-8 | 2.3e-8 |
| `azimuth_loc`, `gamma` in rad | 7.4e-5 | 3.9e-5 | 7.4e-5 |
| `elevation_loc` in rad | 2.1e-5 | 2.9e-5 | 2.1e-5 |

(`V_im`, `H_im` are zero for this pattern.)

CPU time per call, large call: GENERIC 92 ms (double) and 67 ms (float); AVX2 11 ms and 4.4 ms. The four optional outputs add 6 % (GENERIC) and 12 % (AVX2). For small arrays AVX2 gains little or nothing (crossed dipoles, kernel alone, measured in the review):

| n_out x n_ang | GENERIC | AVX2 |
| :-- | --: | --: |
| 1 x 1 | 1.11 us | 1.26 us |
| 2 x 10 | 5.2 us | 5.0 us |
| 2 x 100 | 48.8 us | 40.2 us |
| 8 x 10 | 20.1 us | 5.0 us |
| 16 x 100 | 413 us | 82 us |

Validation cost: quick check 8.5 ns; deep check 0.7 to 0.9 us. For 32 elements and one angle, the member (validation, index conversion, position copy) adds 1 % (quick check) to 9 % (deep check) to the kernel's 9.9 us. In `get_channels_multifreq` (200 paths, 64 output frequencies, three entries copied into the vectors, so the cache is stale), one deep check per interpolation call would add an estimated 44 % (one element), 30 % (two elements) and 3 % (eight elements) to the end-to-end time (measured run time and check cost, counted calls). Hence the internal dispatch without `is_valid` (R4).

Grids changed in place after `validate()` (the quick check still passes): reversed, NaN entries, values far out of range, constant grids. The two kernel sources were compiled with AddressSanitizer and UndefinedBehaviorSanitizer (the rest of the library without) and called with random angles including NaN and infinity, `float` and `double`: no report, both returned. The review confirmed by reading that the AVX2 grid searches stay in bounds for reordered and NaN grids, and that a batch call of the AVX2 kernel equals single-angle calls bit for bit.

# 4. Decisions

## 4.1 Settled with the scope

| ID | Decision | Choice | Reason |
| :-- | :-- | :-- | :-- |
| D1 | Where the plan lives | This document, tracked in `md/`, deleted at the end | reference beyond the commit messages, kept in the history |
| D2 | Input of the public function | `const arrayant<dtype> &` | every caller holds an `arrayant`; `_multi` takes `std::vector<arrayant>`; raw cubes would move arrayant validation to the caller |
| D3 | Member `arrayant::interpolate` | removed | its signature changes anyway, so keeping it saves no caller an edit; one entry point, one doc block, one test suite |
| D4 | Kernel layout | as `ray_triangle_intersect` (`md/SPEC_CPP_API.md`, "AVX2 and CUDA kernels"), plus an internal dispatch for library callers that validated the arrayant themselves (R4) | one validation path for users; no deep check per call in the hot loops |
| D5 | Default kernel | `use_kernel = 0` (auto) everywhere; auto selects GENERIC in this refactor; `use_kernel = 2` selects AVX2 on request | results stay as they are; the AVX2 kernel has a known error at the poles (section 2.1), and when auto may select AVX2 is decided after its fix, from benchmarks (section 10) |
| D7 | Passing `i_element` | `const arma::uvec *i_element = nullptr` | SPEC rule for optional inputs |
| D8 | Empty optional inputs | `nullptr` and an empty object both mean "not provided" | matches `combine_pattern` and the wrappers, which pass empty objects |
| D10 | `arrayant_interpolate_multi` | migrated in this refactor, moved into `src/arrayant_interpolate.cpp` | one change to the interpolation API; the SPEC lets the variants of one function share a file |
| D11 | Kernel argument of `get_channels_spherical` | `int use_kernel = 0` replaces `bool use_avx2 = false` | the same selector as everywhere else |
| D12 | Wrappers | MEX and Python wrappers of the changed functions follow their SPECs and are updated in this refactor | the whole interpolation API changes in one go |
| D13 | Tests | rewritten to `md/SPEC_CPP_API.md`, "TESTS", with a GENERIC-versus-AVX2 comparison away from the poles | the existing suites predate the SPEC; existing cases serve as sources of oracles |

## 4.2 Settled in the first review

These points had options; the recommended option was accepted. The analysis is kept as the reason.

**D6 Width of the element index.** Options: (a) 64-bit `arma::uvec` in the public functions and the kernels; (b) 32-bit (`arma::Col<unsigned>`) in the kernels. With (a), no caller converts anything: all public arrayant functions take `uvec` (`copy_element`, `export_obj_file`, `arrayant_interpolate_multi`, `arrayant_rotate_pattern_multi`), the MEX wrapper already produces `uvec` (`qd_mex_typecast_Col<arma::uword>`), and `get_channels_*` build their per-link index directly as `uvec`. With (b), the public functions and `_multi` keep a copy-and-narrow step, as today. The AVX2 kernel needs int32 offsets either way; it computes one per output element (`n_out` values, not per angle), so (a) costs nothing measurable there. Memory: the index has `n_out` entries (per link in `get_channels_*`, e.g. 4096 for 64 x 64), a few tens of kB at most. Safety: a 64-bit index narrowed to 32 bits before the bounds check wraps into the valid range (`2^32` becomes `0`); (a) has no narrowing step, (b) must keep the check before the conversion. **Decision: (a).**

**D9 A public `validate_input` flag.** Options: (a) no flag; every call of a public function validates (shapes, indices, aliasing, `is_valid()` of the arrayant); `_multi` loses its existing flag. (b) a flag as `_multi` has today. Implications:
- Cost: see section 3. The deep check matters only for small arrays in hot loops inside the library; those callers use the internal dispatch (R4), which is not a public flag.
- Memory safety: under D4 the kernels have no checks of their own. With (b), `validate_input = false` with inconsistent data (e.g. `e_theta_im` smaller than `e_theta_re`, an element index out of range, `element_pos` of the wrong size) reads out of bounds. That is undefined behaviour, reachable whenever pattern data comes from a file (QDANT, HDF5) or from MATLAB/Python. Today `_multi` with the flag off is still protected by the kernel's own checks; after D4 it would not be. The `arrayant` members are public and mutable, so data checked once can change before the call.
- Remaining gap with (a): the quick check accepts an arrayant whose grid values were changed in place after `validate()`. Section 3 found no out-of-bounds access for that case in either kernel; it stays covered by the fuzz target, which already does such in-place changes (`tests/fuzzing/fuzz_target.hpp:241-249`).
**Decision: (a).**

**D14 Kernel used by library functions that have no `use_kernel` argument.** Concerns `combine_pattern`, `rotate_pattern` (and through them all `generate_arrayant_*` and the `_multi` variants), `calc_beamwidth_deg`, `export_obj_file`, `get_channels_planar`, `get_channels_multifreq`, and the calls of `get_channels_spherical` inside `get_channels_irs` and `gbsm_single_link_channel`. Options: (a) they pass `use_kernel = 1`; their results stay as they are also when auto changes later; a `use_kernel` argument for them is a later, per-function API decision. (b) they use auto. (c) they all get a `use_kernel` argument now. With (b), a later change of auto would change every generated pattern (`generate_arrayant_*` interpolates through `rotate_pattern` and `combine_pattern`) without a way back to double precision; (c) adds ten public functions and their wrappers to this refactor. **Decision: (a).**

**D15 Small contract points.** Written into section 5:
- (a) Parameter name of the arrayant: `ant`. A parameter named `arrayant` hides the class template inside the function, which turns any later use of `arrayant<dtype>` in the body into an error.
- (b) `use_kernel = 2` when AVX2 is unavailable, or when `ant.e_theta_re.n_elem > INT_MAX` (the limit applies to all elements together, because the int32 offset is `index * n_pattern_samples`): throws `std::invalid_argument`.
- (c) Empty angle input: `azimuth` with `n_ang = 0` columns gives outputs of size `[n_out, 0]`; zero rows is an error.
- (d) The alias check covers the arrayant's own data members besides the other inputs.

**W1 to W7 Wrapper points.** Section 7.2.

## 4.3 Settled after the independent review

An independent review of the first version checked the claims against the code, simulated the planned changes on a scratch copy and measured. Its findings are worked into sections 2, 3, 5, 6, 7 and 8. These points needed a decision:

**R1 How the GENERIC default is expressed.** Options: (a) default argument `use_kernel = 1`, changed to 0 later; (b) default `use_kernel = 0`, with auto selecting GENERIC for now. (a) changes the declared default later, a second API change; (b) keeps the declarations stable and uses the SPEC meaning of 0, at the price that callers who rely on auto get other results when auto changes; that change is announced in the release notes when it happens. **Decision: (b).**

**R2 Documenting the pole error until the issue exists.** Options: (a) a warning line in the `use_kernel` documentation now; (b) no mention; the note comes with the issue. The cause is not analysed yet, and the documentation cites only published material. **Decision: (b).**

**R3 When auto selects AVX2.** AVX2 gains nothing for one or two elements and about 4x from eight elements (section 3). Not part of this refactor, and not part of the AVX2 fix: after the fix, a dedicated session benchmarks typical usage scenarios, and the results decide what auto does (section 10). **Decision: deferred.**

**R4 Validation cost for internal callers.** Options: (a) an internal dispatch that checks the shapes it is given and skips `is_valid`, used by library functions that validated the arrayant themselves; (b) all callers use the public function and pay the deep check where the cache is stale; (c) both (a) and a `copy()` that refreshes the cache of the copy. A refreshed cache after `copy()` is safe: the copy holds the same data as a source that passed `is_valid()`. It helps user code that copies and then interpolates in a loop; it does not help the implicit copy constructor (vector copies) or arrayants from MATLAB, so (a) is still needed for the library's hot loops. A user-defined copy constructor would have to list every member, a trap for later members, and is not part of this refactor. **Decision: (c).**

**R5 Logical input for `use_kernel` in MEX.** All MEX kernel selectors (`ray_triangle_intersect`, `ray_point_intersect`, `calc_diffraction_gain`, `sos_val`, `sos_map`, `cart2geo`, `geo2cart`) read `use_kernel` with `qd_mex_get_scalar<int>`, which accepts logical input (`src/mex_helper_functions.hpp:114`), so `true` becomes 1. Options: (a) accept logical as everywhere; (b) reject logical for `get_channels_spherical` only; (c) reject it for every selector. With (a), an old MATLAB call that passes `true` as argument 15 of `get_channels_spherical` (formerly `use_avx2`) now selects GENERIC. **Decision: (a)**, consistent with the other selectors; the change of meaning of argument 15 is stated in the commit message of the wrapper step.

**R6 Form of the kernel arguments.** Options: (a) raw pointers and sizes, as the other kernels in `quadriga_lib_generic_functions.hpp` / `quadriga_lib_avx2_functions.hpp`; (b) Armadillo references with a symbol check. Armadillo types in the AVX2 header would put Armadillo template code into every AVX2 object, the hazard of section 2.6. **Decision: (a), plus a check in step 7 that no weak function in the new AVX2 object contains AVX instructions.** The hazard in the existing AVX2 objects is recorded as an issue at the end of the refactor (step 8).

**R7 Python argument order.** Options: (a) the SPEC order with `use_kernel` before `complex` and `complex` last; (b) `use_kernel` appended, `complex` unchanged. (a) breaks callers that pass `complex` by position; keyword calls keep working. **Decision: (a)**, as part of the one planned API break; `complex` keeps its default `False` (`md/SPEC_PYTHON_API.md` keeps the default of existing wrappers).

**R8 The wrappers' own "angles cannot be empty" check.** Options: (a) remove it, so that empty angles give empty outputs as in C++; (b) keep it. **Decision: (a).**

**R9 Import style of the rewritten Python test.** **Decision: `import quadriga_lib`**, as in most Python test files (53 of the 76 files; 26 use `from quadriga_lib import ...`, some both).

## 4.4 Single document

This document combines plan and spec. A separate spec would pay off if the contract were reviewed by someone other than the implementer before the plan, or if the contract were large; here it is three declarations, one internal dispatch and two kernels, and most decisions tie a contract point to an implementation step. Section 5 is the contract and is not changed by the implementer (`md/SPEC_CPP_API.md`, "CONTRACT").

# 5. Contract

Normative.

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
- `use_kernel`: 0 = auto, which selects GENERIC in this version; 1 = GENERIC; 2 = AVX2, which throws when AVX2 is unavailable (`BUILD_WITH_AVX2` and the cached `runtime_AVX2_Check()`, not `quadriga_lib_has_AVX2()`, which runs a test kernel per call) or when `ant.e_theta_re.n_elem > INT_MAX`; other values throw. The documentation states that AVX2 computes in single precision also for `double`, with the order of the errors of section 3, that GENERIC computes in the precision of `dtype`, and that the kernel chosen by auto may change in a later version. It does not mention the pole error (R2).
- Validation, all before any output is written, all `std::invalid_argument` with messages in the SPEC style: `ant.is_valid()`; `azimuth` / `elevation` sizes; `i_element` entries `< n_elements`; `orientation` shape; `element_pos` shape; `use_kernel` value and availability; aliasing of any output with an input, with a data member of `ant`, or with another output (`qd_check_alias_list`).
- After validation and output sizing it calls the internal dispatch (section 5.4).

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

Behaviour:
- The frequency blending (bracketing entries, SLERP with linear fallback, clamping, caching of consecutive brackets) is unchanged.
- `frequency` `nullptr` or empty: the center frequencies of the entries, in vector order, so `n_freq` equals the number of entries. This default moves from both wrappers into C++ (W2).
- `n_out`, `i_element`, `azimuth` / `elevation` shapes, `orientation` and `element_pos` as in section 5.1, with `n_elements` and the default positions taken from entry 0. `n_ang = 0` gives outputs of size `[n_out, 0, n_freq]`.
- `use_kernel` as in section 5.1; kernel 2 is checked against the largest entry.
- Outputs are resized only when their size differs and are completely written.
- Validation, all before any output is written, all `std::invalid_argument`: `arrayant_vec` not empty; `arrayant_is_valid_multi(arrayant_vec, true)` on every call (`validate_input` is removed, D9); the shape checks of section 5.1; `frequency` values finite; `use_kernel`; aliasing of any output with an input, with a data member of any entry, or with another output.
- After validating once, it calls the internal dispatch per entry and bracket, not the public function.

## 5.3 `get_channels_spherical`

- The last argument `bool use_avx2 = false` becomes `int use_kernel = 0` with the values of section 5.1. It selects the kernel for the antenna interpolation and for the coefficient combination (`qd_coeff_combine` / `qd_coeff_combine_avx2`); auto selects GENERIC for both.
- Hazard: `false` and `true` convert silently to 0 (auto) and 1 (GENERIC); `true` thus selects the opposite of its meaning today, and the compiler does not warn. All six call sites of section 2.2 are edited explicitly to 1 (former `false`) or 2 (former `true`); a search for `use_avx2` outside `md/` and `docs/` must come back empty after step 4 (and in `docs/` after `make docs`).
- Behaviour change: `use_avx2 = true` was ignored where AVX2 is unavailable; `use_kernel = 2` throws there. Tests that request kernel 2 run only where AVX2 is available.
- It validates the two arrayants itself and calls the internal dispatch.
- `get_channels_irs` and `gbsm_single_link_channel` pass `use_kernel = 1` (D14). This means spelling out every trailing optional output pointer with its type, e.g. `(arma::Cube<dtype> *)nullptr`, because a bare `nullptr` fails template argument deduction.

## 5.4 Internal dispatch and kernels

- Internal dispatch `qd_arrayant_interpolate_dispatch<dtype>`, defined in `src/arrayant_interpolate.cpp`, declared in `src/qd_arrayant_functions.hpp` with a comment that states its preconditions. Arguments: the arrayant, angles, a 0-based `i_element` (`uvec`, never empty), a non-empty orientation, positions `[3, n_out]`, the four mandatory outputs and four optional output pointers, all pre-sized by the caller, and the resolved kernel (1 or 2). It checks, cheaply and before any kernel runs: index range, the shapes of angles, orientation, positions and outputs, and for kernel 2 the `INT_MAX` limit; it throws `std::invalid_argument` on a violation. It does not call `is_valid()` and does not check aliasing: the caller has validated the arrayant and owns the outputs. Callers: the two public functions, `get_channels_planar`, `get_channels_spherical`, `get_channels_multifreq`, `combine_pattern`, `rotate_pattern`, `calc_beamwidth_deg`, `export_obj_file`; each of the library callers validates its arrayants before the first call (as they do today, e.g. `get_channels_spherical.cpp:189-195`, `get_channels_multifreq.cpp:239-242`).
- Kernels `qd_arrayant_interpolate_GENERIC<dtype>` in `src/arrayant_interpolate_generic.cpp`, declared in `src/quadriga_lib_generic_functions.hpp`; `qd_arrayant_interpolate_AVX2<dtype>` in `src/arrayant_interpolate_avx2.cpp`, declared in `src/quadriga_lib_avx2_functions.hpp`, compiled only with AVX2 (file suffix), called only under `#if BUILD_WITH_AVX2`.
- Kernel arguments: raw pointers and sizes, as the other kernels in these headers (R6): pattern data pointers with `n_elevation`, `n_azimuth`, `n_elements`; grid pointers; angle pointers with `n_ang` and a flag or row count for per-element angles; orientation pointer with its column and slice counts; position pointer; 0-based index pointer with `n_out`; output pointers (optional ones may be `nullptr`). No Armadillo types in either header declaration.
- Preconditions, stated in the header comment and not checked by the kernels: everything the dispatch checks.
- The kernels use `qd_rotation_matrix` from `src/quadriga_lib_helper_functions.hpp` instead of their local copies (the review confirmed that the AVX2 copy computes the same matrix; step 0 confirms it for the scalar copy, with `invert_y_axis = true`, `transposeR = true` as called today).
- The computation and the OpenMP regions are taken over unchanged (GENERIC `if (n_ang > 1 && n_ang * n_out >= 256)`, AVX2 regions as they are); `md/SPEC_CPP_API.md` asks for measured limits only for new or changed regions. File-local helper types of the AVX2 file (`AzGridParams`, `ElGridParams`, ...) go into an anonymous namespace.

## 5.5 Class change: `copy()`

`arrayant<dtype>::copy()` (`src/qd_arrayant.cpp:659`) stores the data pointers of the copy in its validation cache when `this->is_valid()` returns an empty string, so that the copy passes the quick check (R4). Nothing else in the class changes. Its documentation gets one line saying that a copy of a valid arrayant is marked as validated.

## 5.6 Removed

`arrayant<dtype>::interpolate` (declaration, definition, documentation block), `src/qd_arrayant_interpolate.hpp`, `src/qd_arrayant_interpolate.cpp`, `src/qd_arrayant_interpolate_avx2.hpp`, `src/qd_arrayant_interpolate_avx2.cpp` (replaced by `src/arrayant_interpolate_avx2.cpp`), `src/qd_arrayant_xinterpol_multi.cpp` (it holds only `arrayant_interpolate_multi`, which moves).

# 6. Implementation steps

Test-first as in `md/SPEC_CPP_API.md`, "IMPLEMENTATION ORDER". Build out of tree; the in-repository `build_linux` is not used. Run the C++ tests with `OMP_WAIT_POLICY=passive`. Each step ends with a build without warnings.

**Step 0 — Baseline.** Build at the starting commit and run the complete C++ suite; record the result (the review measured 1657 cases passed, 1 skipped). Diff the scalar kernel's local `qd_rotation_matrix` against the shared helper.

**Step 1 — Stubs.** Declarations of sections 5.1 and 5.2 in the header (old declarations still present), `src/arrayant_interpolate.cpp` with license, includes, section statement, first documentation blocks, stub bodies that size all outputs and fill zeros, and explicit instantiations.

**Step 2 — Tests against the stubs.** Replace `tests/catch2_tests/test_arrayant_interpolate.cpp` and `test_arrayant_interpolate_multi.cpp` by new files (section 8); the old cases go, their oracles are carried over. Until step 4 the old member and the old `_multi` signature still exist, so the new files call only the new functions. Run them against the stubs, evaluate the oracles, write the test record. **Stop point 1: the maintainer reviews the test record.**

**Step 3 — Kernels and implementation.** Copy the kernels into the new files (section 5.4) with the new arguments; the old files stay until step 4, so that the old callers still build. Write the dispatch and implement both public functions; change `copy()` (section 5.5). Iterate until the new tests pass.

**Step 4 — Callers and removal.** Move every caller of section 2.2 to the dispatch or the public function (D14: `use_kernel = 1` where the caller has no selector); pass `nullptr` instead of empty objects for optional outputs that are not needed (`rotate_pattern`, `get_channels_planar`); change `get_channels_spherical` (section 5.3) and edit its six call sites; remove what section 5.6 lists; update the fuzz targets `fz_arrayant_interpolate.cpp` (new functions, `use_kernel` drawn from the available kernels) and `fz_get_channels.cpp`, the table in `tests/fuzzing/README.md`, and the comment at `test_arrayant_combine_pattern_multi.cpp:745`.

**Step 5 — Full suite.** Run the complete suite. With auto selecting GENERIC and the library callers on kernel 1, every suite except the two rewritten ones must pass unchanged; a failure there is a defect of the refactor and is fixed, not tolerated. The tests that requested AVX2 explicitly (`test_get_channels_spherical.cpp:1943-1985`) keep their tolerances. **Stop point 2: report the result; if a test outside the rewritten suites needed a change, the maintainer decides.**

**Step 6 — Wrappers.** MEX and Python wrappers and their tests (section 7). Run `make test` with the MEX suites, and the Python suite (`python -m pytest tests/python_tests`, with the built module in `lib/`).

**Step 7 — Finish.** `make docs`; update the `arrayant_interpolate` row of `docs_builder/content/api_matching.toml` to `["arrayant_interpolate", "arrayant_interpolate_multi"]`; complete suite; AddressSanitizer run (`make test-asan`) of the changed test cases; a short fuzz run of the two fuzz targets (`make test-fuzz FUZZ_TIME=2m FUZZ_ARGS='--targets fz_arrayant_interpolate'`, same for `fz_get_channels`); check with `nm` and `objdump` that no weak function in the new AVX2 object contains AVX instructions (R6); MSVC check by reading (no GCC-only extensions); hand-over report (`md/SPEC_CPP_API.md`, "HAND-OVER REPORT"). **Stop point 3: review of the hand-over.**

**Step 8 — Issues.** Draft two issues for the maintainer, each shown before it is filed (`CONTRIBUTING.md`, "Filing an issue"), based on the refactored code: the AVX2 pole error (section 2.1; an agent measures it, finds the cause and writes the reproduction; the issue contains the test that pins it), and the weak-symbol hazard of the AVX2 objects (section 2.6). Draft also the enhancement issue for deciding auto by benchmarks (R3), if the maintainer wants it filed now.

Commit messages follow `CONTRIBUTING.md`; the maintainer commits.

# 7. Wrappers

Facts from the review of the current wrappers (line numbers at `34e675e`) and the decisions on them.

## 7.1 Settled

- MEX `arrayant_interpolate` (`api_mex/arrayant_interpolate.cpp`): calls `quadriga_lib::arrayant_interpolate` / `arrayant_interpolate_multi`; `nullptr` for `orientation`, `element_pos` and optional outputs that are absent or not requested (`md/SPEC_MATLAB_API.md`:352-355); `var_name` in all argument readers; arguments are parsed and C++ is called also when `nlhs == 0`, so that errors are raised; no own check for empty angles (R8); documentation to the SPEC (arrayant phrase, `default:` on every optional argument, standard headings).
- Python `arrayant_interpolate` (`api_python/qpy_arrayant_interpolate.cpp`, registered in `api_python/python_arrayant.cpp:104-115`): `azimuth` and `elevation` as `py::array_t` with `.none(false)`; flags read with `qd_python_flag`; no own check for empty angles (R8); with `local_angles` but without `dist`, `nullptr` is passed for `dist` (no scratch output); the comment block at the end of the file identical to the registration and naming the right file. Argument list (R7): `arrayant, azimuth, elevation, element=None, orientation=None, element_pos=None, frequency=None, dist=False, local_angles=False, fast_access=False, use_kernel=0, complex=False`.
- MEX `arrayant_interpolate`: `use_kernel` as the last input, default 0, read with `qd_mex_get_scalar<int>(..., "use_kernel", 0)`, which accepts logical input like the other selectors (R5).
- Both: `use_kernel` documented as in the `sos_val` wrappers, with the precision note of section 5.1.
- `get_channels_spherical`: MEX input 15 `use_avx2` becomes `use_kernel` (int, default 0, read as above); the commit message of this step states that a logical `true` there now selects GENERIC. Python gets `use_kernel` after its existing arguments and before the standard flags `complex` / `stack` where present, and passes it.
- Tests: `tests/mex_tests/test_arrayant_interpolate.m` and `tests/python_tests/test_arrayant_interpolate.py` rewritten to the SPEC test rules (MATLAB: header comment, `%%` sections, checklist; Python: one `test_case` class, keyword-argument, memory layout, dtype, inputs-not-modified, exception types), including `use_kernel` and empty angles; `import quadriga_lib` in the Python test (R9). The `get_channels_spherical` tests in both languages gain `use_kernel` cases. Python tests that pass `dist=1` / `local_angles=1` as integers are changed to booleans, since `qd_python_flag` rejects integers. Positional uses of `complex` in Python tests and examples move to keywords.

## 7.2 Settled in the first review

**W1 MEX separate-field input mode** (13 inputs, `api_mex/arrayant_interpolate.cpp:162-170`). `md/SPEC_MATLAB_API.md`:261 calls it legacy. **Decision: remove**; MATLAB users of this form change to the struct.

**W2 Frequency default.** Both wrappers fill the target frequencies from the entries' center frequencies when none are given (MEX `:130-135`, Python `:131-139`), a default that `md/SPEC_MATLAB_API.md`:56 places in C++. **Decision: `frequency` optional in C++ (section 5.2)**; the wrappers pass `nullptr` when it is absent.

**W3 Single- versus multi-frequency dispatch in the wrappers.** The wrappers choose between the two C++ functions from the input (struct array, frequency given). `md/SPEC_MATLAB_API.md`:55 allows one MATLAB function dispatching to several C++ functions when the plan settles it. **Decision: keep the dispatch.** The selection changes no numbers; the computation stays in C++.

**W4 Python flags `dist` and `local_angles`.** They change the returned tuple, which `md/SPEC_PYTHON_API.md`:330 and :451 allow only by decision. The optional outputs cost 6 % to 12 % (section 3), so cost is not the argument; but the multi-frequency mode cannot produce them, so always returning all eight outputs would make the tuple length depend on the mode, and every current caller that unpacks four values would break. **Decision: keep them as a decided exception.**

**W5 Argument names.** Python uses `frequency`, MEX `freq`; Python documents the outputs as `vr, vi, hr, hi`, MEX as `V_re, ...`. `md/SPEC_PYTHON_API.md`:489 takes MEX as the reference. **Decision: the MEX documentation names the argument `frequency`** (the C++ name; MEX arguments are positional, so nothing breaks), **and the Python documentation names the outputs `V_re, V_im, H_re, H_im`.**

**W6 Python `fast_access`.** Python-only, maps to the strict flag of the dictionary conversion (`:70-72`, `:119`). Not part of the interpolation contract. **Decision: keep unchanged.**

**W7 Remaining deviations of the `get_channels_spherical` wrappers** (MEX `var_name`, documentation wording; Python plain `bool` flags, the non-standard `angles` flag, the batch mode with OpenMP and released GIL). Not interpolation API. **Decision: out of scope (section 10).**

# 8. C++ tests

Rules: `md/SPEC_CPP_API.md`, "TESTS". Case names `"arrayant_interpolate - ..."` and `"arrayant_interpolate_multi - ..."`. Oracles independent of the library's interpolation code; supporting tasks (pattern generation, rotation matrices) use library calls. No test evaluates the AVX2 kernel at exactly ±90° elevation; that test belongs to the pole issue (step 8).

`test_arrayant_interpolate.cpp` covers at least:
- Values against closed-form oracles, for GENERIC (`use_kernel = 1`, tolerances near double precision) and, where AVX2 is available, for AVX2 (tolerances from single-precision arithmetic, stated in a comment; oracle angles away from the poles): grid-point reproduction; interpolation along azimuth and elevation on small grids (the hand values of the current cases at `test_arrayant_interpolate.cpp:14-257`); z- and y-rotation shifting the local angles; x-rotation turning polarization; projected distance (`:292`).
- Every argument: `i_element` omitted, permuted, with duplicates; the four `orientation` shapes; `element_pos` given, omitted with `ant.element_pos` set, omitted with `ant.element_pos` empty; planar and per-element angles; `n_ang = 0`.
- Optional outputs: each alone and all together give the same values; omitted outputs are not touched.
- Pre-sized outputs are written in place (data pointer unchanged); outputs of wrong size or with stale content are resized and fully overwritten.
- NaN and infinite angles give NaN, for both kernels; angles outside the nominal ranges behave as with GENERIC today.
- GENERIC versus AVX2 on a realistic pattern (a 3GPP panel, per-element angles and orientation, angles drawn with elevation inside ±(90° − 0.01°), a size that crosses the AVX2 vector width and the OpenMP limits): all outputs agree within tolerances derived from single-precision arithmetic, with a diagnostic table (`QD_DIAG`) of the differences.
- `float` and `double`: both instantiations, and a parity case.
- Determinism: repeated calls identical; a large call (OpenMP active) equals the concatenation of single-angle calls, per kernel.
- `use_kernel`: 0 gives the same result as 1; 2 throws where AVX2 is unavailable (case runs only there); values outside 0 to 2 throw.
- `copy()`: the copy of a validated arrayant passes the quick check (its cache holds its own data pointers); the copy of an invalid arrayant does not get a cache.
- `"arrayant_interpolate - input validation"`: valid baseline with `CHECK_NOTHROW`, one section per invalid argument, including aliasing with an input, with a member of `ant`, and between outputs.

`test_arrayant_interpolate_multi.cpp`: the existing 48 cases are the source of the oracles (frequency bracketing, SLERP and linear fallback, clamping, caching, independence of the entry order); the new file covers them with the new signature and the SPEC structure, with comparisons against `arrayant_interpolate` instead of the member, one validation case with a baseline instead of the separate validation cases, no `validate_input` cases, and cases for omitted `frequency`, `n_ang = 0` and `use_kernel`.

Other suites: calls of the member in `test_arrayant.cpp` move to `arrayant_interpolate`; `use_avx2` in `test_get_channels_spherical.cpp` becomes `use_kernel` with the meaning kept (1 for the former `false`, 2 for the former `true`, the latter only where AVX2 is available).

# 9. Documentation

- Documentation blocks of `arrayant_interpolate` and `arrayant_interpolate_multi` to `md/SPEC_CPP_API.md`, "Documentation block": what the function does and is for, data layout, sizes, units, defaults, the kernel choice with its precision, the errors a caller can trigger; no implementation details. The member's block is the starting point for content; its Example is kept in the new call form.
- Links: the class table in `src/qd_arrayant.cpp:67`, the "See also" of `combine_pattern` (`:525`) and of `_multi` point to `[[arrayant_interpolate]]`; the header comment of `_multi` no longer names `qd_arrayant_interpolate`.
- `get_channels_spherical`: argument entry for `use_kernel`, naming both kernels it selects.
- `copy()`: one line on the validated state of the copy (section 5.5).
- Generated files in `docs/` are regenerated with `make docs` and committed with the change. The `website_builder` test fixtures use `.interpolate` as sample data of a mini repository and are not changed.

# 10. Follow-ups, not in this refactor

- The AVX2 pole error (section 2.1): after the refactor, an agent measures it on the refactored code, finds the cause and writes the reproduction; the issue contains the test (step 8). Fixed in its own change.
- When auto selects AVX2 (R3): after the pole fix, a dedicated session benchmarks typical usage scenarios (array sizes, angle counts, channel generation); the results decide the auto rule. Possibly an enhancement issue of its own.
- The weak-symbol hazard of the AVX2 objects (section 2.6): recorded as an issue at the end of the refactor (step 8).
- `use_kernel` for `combine_pattern`, `rotate_pattern`, `calc_beamwidth_deg`, `export_obj_file`, `get_channels_planar`, `get_channels_multifreq` and the functions built on them (D14), as part of the 1.0 API cleanup.
- A validation cache that survives the implicit copy constructor (R4).
- The remaining SPEC deviations of the `get_channels_spherical` wrappers (W7).
- Hidden symbol visibility for the shared library and the Python module, so that internal `qd_*` symbols are not exported.
