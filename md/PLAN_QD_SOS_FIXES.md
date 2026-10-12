# Plan: fixes and speed-ups for the sum-of-sinusoids functions

Written 2026-10-12 from `md/REVIEW_QD_SOS.md` (the review). This is the brief for a session that implements
the findings. The review holds the evidence, reproductions and measurements; this plan holds the decisions,
the order of work and what "done" means. Identifiers (B1 to B10, P1 to P9, D1 to D9) are those of the review.

Read first: this plan, then `md/REVIEW_QD_SOS.md`, then the rules in `md/SPEC_CPP_API.md`,
`md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md`. `md/LOG_QD_SOS.md` records how the code came about.


## 1. Scope

In scope: all defects B1 to B10, all test gaps of review section 2.2, the speed-ups P1 to P8, and the
documentation points of review section 4.

Out of scope:
- P9 (`sos_map` with per-axis sines and cosines). It would make `sos_map` differ from `sos_val` in the last
  bits, which the documentation excludes.
- The error floor of the 2-D exponential tables. Unexplained; no change to `sos_generate`'s search.
- Any change to the declarations in `include/quadriga_sos.hpp` (names, order, types, defaults). Comments
  in the header may be corrected.
- The shipped preset tables. They must stay bit-identical.


## 2. Boundary to the rest of the library

Checked on 2026-10-12: no code outside the SOS files calls an SOS function, a `qd_sos_*` helper or a
`qd_SOS_*` kernel. Sources and tests are collected by glob, so `CMakeLists.txt` needs no change. All work
therefore stays inside this list:

- `src/sos_*.cpp`, `include/quadriga_sos.hpp` (comments only)
- the `qd_sos_*` block at the end of `src/quadriga_lib_helper_functions.hpp`
- the `qd_SOS_*` declarations in `src/quadriga_lib_generic_functions.hpp` and
  `src/quadriga_lib_avx2_functions.hpp`
- `api_mex/sos_*.cpp`, `api_python/qpy_sos_*.cpp`, `api_python/python_sos.cpp`
- `tests/catch2_tests/test_sos_*.cpp`, `tests/mex_tests/test_sos_*.m`, `tests/python_tests/test_sos_*.py`
- `md/LOG_QD_SOS.md` (append), `html_docu/` only as described in section 7

Do not edit, although a fix there looks natural:

- `src/mex_helper_functions.hpp` (`qd_mex_get_scalar`). 19 wrappers outside this port read unsigned scalars
  through it, and existing MEX tests of other functions pass negative numbers. B1 and B10 are fixed inside
  the SOS wrappers.
- `api_python/python_arma_adapter.hpp`. B2 is fixed in the SOS wrappers or their registration.
- `src/fastmath_avx2.h`, any other kernel, any other test.

If a task seems to need a file outside the first list, stop and ask.


## 3. Decisions

Answered by Stephan on 2026-10-12, one by one. They are binding for the session.

| # | Question | Decision |
|---|---|---|
| F1 | B1, B10: where to validate MEX scalars | In each SOS wrapper, not in the shared helper: read counts and `imax` as `double`, reject NaN, Inf, negative and (where the library forbids it) zero with a message that names the argument, then cast. Non-integer values stay truncated, as in the rest of the library |
| F2 | B2: `None` for mandatory arrays in Python | `py::arg("name").none(false)` in `python_sos.cpp` for every mandatory array of the 14 functions |
| F3 | B4, D9: outputs that alias inputs | Make safe where cheap: read or copy inputs before any output is resized, so that aliasing works. Where that would cost a copy of a large array, keep a documented restriction for that function ("must not be the same object as ...") and list these functions in the report. `sos_val` is such a case once P1 writes into `val` directly: its restriction stays. A function documented as safe gets a test |
| F4 | B7: ACF value of -1 or below | `std::invalid_argument` in the same-map mode of `sos_val`, `sos_remove_bias` and in `sos_acf_estimate` with `acf_est_dual` |
| F5 | B8: large `freq_max` | Keep the stored table and throw `std::invalid_argument` when the coarse grid exceeds a fixed number of points (100,000; the presets use 150). No change to the arithmetic of the search. State the limit in the documentation |
| F6 | B9: `phase_b` is the same object as `phase` | `std::invalid_argument` in `sos_remove_bias` |
| F7 | Review 1.2: two-map phases depend on `n_gen` | Change: the second device's phases come from a second stream, `sos_init_phases(n_coeff, n_gen, seed_b)`, with `seed_b` derived from `seed` by a fixed, documented rule; for `seed = -1` both streams are random. Values of the two-map case change; one device and same map stay bit-identical. Update the documentation of the three functions and the MEX and Python tests that pin column `n_gen + g` |
| F8 | Review 1.2: same seed in separate calls gives related fields | Documentation only: a warning in the three introductions and in `sos_randn`, `sos_rand`, `sos_randi`, `sos_init_phases` |
| F9 | Perfect fit in `sos_calc_mse` | Keep +Inf, document it, pin it with a test |
| F10 | MEX introduction longer than 100 characters per line | Leave as is. Only the three C++ lines above 120 characters and the MEX usage line of 101 characters are wrapped |
| F11 | Speed-ups | P1 to P8, not P9; each only if the existing tests pass unchanged and the bit-identity checks of section 5 hold |


## 4. Work items

Work through them in this order. Each item ends with the complete SOS suites passing (section 6).

### W0 Baseline
- Build as in section 6 and run the three SOS suites. Record the result in `md/LOG_QD_SOS.md` under a new
  heading "Fixes after the review".
- Save reference outputs for the bit-identity checks of section 5 before any code changes.

### W1 Defects in the C++ library
B3, B4, B5, B6, B7, B8, B9 with decisions F3 to F6 and F9. Test first: add the failing test, see it fail for the
right reason, then fix.

- B3: tolerance `max(1e-4 · step, 4 · eps(dtype) · dist.back())` in `qd_sos_equally_spaced`.
- B5: no allocation that can throw, and no unchecked `_mm_malloc`, inside a parallel region. Allocate
  per-thread buffers before the region or use `OMP_SAFE_CALL` and rethrow after it.
- B6: finite `dist` in `qd_sos_check_acf`; the same checks in `sos_acf_approx`; finite `bins`.

### W2 Defects in the wrappers
B1, B10 (F1) and B2 (F2), with MEX and Python tests. Also add the wrapper tests listed in review 2.2:
complex and `uint8` input in MEX (assert the present behaviour, whatever it is, so that a later change of
the helper is noticed), 1-D `pos` and C against Fortran order in Python.

### W3 Remaining test gaps
Every row of the table in review 2.2 that W1 and W2 did not cover, and the four statements listed below
that table. In particular:

- a checksum test per preset table (sum and sum of squares of the stored values, relative 1e-15), computed
  once from the present tables;
- the "left unchanged" branch of `sos_remove_bias`;
- unequal sample spacing in `sos_generate` against the brute-force oracle;
- `float` against `double` on rounded inputs with a tolerance of 2e-6;
- the unavailable-kernel error, guarded by `if (!has_avx2())`.

Oracles follow "Fixtures and oracles" of `SPEC_CPP_API.md`. Do not loosen an existing tolerance.

### W4 Speed-ups without a change of results
P1, P2, P4, P5, P6, P8, then P3, then P7.

- P1 (`sos_val.cpp`): the kernel writes into `val` for `dtype = double`; for `float` one temporary. Copy,
  `erfc` mapping, two-device combination and normalisation as parallel loops with a signed loop variable.
  No more than 2 output-sized arrays alive for two devices, 1 for one device in `double`.
- P2: parallel loop over the sinusoids in `accumulate` of `sos_remove_bias`.
- P4: GENERIC kernel accumulates 4 generators per pass over the sinusoids; the order of additions per
  (position, generator) stays as it is.
- P5: `sos_calc_mse`, parallel over the test directions, per-direction sums added serially in order.
- P6: `sos_randn` and friends and `sos_map`: drop copies that P1 makes unnecessary. Only if it stays simple.
- P8: `sos_acf_approx` computes half of each plane and mirrors it.
- P3 (AVX2 kernel): weights packed per group of 8 generators, 8 named accumulators, positions in tiles.
  Keep the arithmetic per (position, generator): single-precision partial sums over 16 sinusoids, cosine
  term then sine term in order of the sinusoids, added in double. Generator counts that are not a multiple
  of 8 must not be slower than today (review: padding one generator to 8 cost 10 %); use narrower paths
  for the remainder. If the full rework does not reach the gains of the review, fall back to P3a.
- P7: `sos_acf_estimate`, parallel over fixed-size blocks of rows whose size does not depend on the thread
  count, merged in order.

Prototypes of P3, P3a and P4 from the review session, bit-identical to the present kernels, and the
benchmark programs of the review are in `tools/sos_port/review_prototypes/` (see its README). They are
reference material: the tiled prototype pads the generators to a multiple of 8, which the implementation
must not do. Use `bench_variants.cpp` as the pattern for the bit-identity checks and the other programs for
the timings of section 5.

### W5 Decision F7
Second phase stream for the two-map case. This is the only item that changes values on purpose. Do it after
W4 so that the bit-identity checks of W4 are not mixed with it.

### W6 Documentation
- D1 to D8 of review 4.1, the consistency points of 4.2, the gaps of 4.4, the line lengths of 4.5 as far as F10 says, F8, F9.
- Define `pos_new`, `pos_tx` and `pos_rx` in the examples of all three introductions; the C++ blocks must
  compile when pasted into one scope. Run all examples again.
- Header comments: lines 34, 84 to 86, 100, 115, 141, 217.
- MEX and Python blocks follow the C++ blocks; units as their specs say, applied the same way to `[m]`,
  `[1/m]` and `[rad]`.

### W7 Report
Append to `md/LOG_QD_SOS.md`: per item what was changed, test results with the actual output, the benchmark
table of section 5, every deviation from this plan, and what was not done and why.


## 5. Acceptance

Bit-identity. With reference outputs saved in W0 (seeded inputs, both kernels, 1 and all threads,
`n_gen` = 1, 3, 7, 8, 9, 20, 400; `n_coeff` = 1, 16, 17, 128, 1024; `n_pos` = 1, 7, 8, 9, 1003):

- `sos_val`, all three device modes, Normal and Uniform, `float` and `double`: identical after W4.
- `sos_map`, `sos_randn`, `sos_rand`, `sos_randi` (one device and same map): identical after W4 and after W5.
- `sos_remove_bias`, `sos_calc_mse` return value, `sos_acf_estimate`: identical after W4.
- `sos_acf_approx`: identical in the half that is computed, mirrored exactly in the other.
- `sos_generate` with the preset settings: the three 128 tables are regenerated to the same values as
  before (the diagnostic of the existing test prints the difference; it is 3e-17 today).
- The checksum tests of W3 pass unchanged at the end.

If an item cannot keep bit-identity, stop and report the difference before going on.

Speed, measured with one process per thread count, fastest of at least 3 calls, on an idle machine (check
`uptime`; the machine is shared). Report before and after for 400 generators, 256 sinusoids, 100,000
positions, 1 and 16 threads:

| Call | Review, before | Expected after |
|---|---|---|
| `sos_val`, `val` pre-sized, 16 threads | 121 ms | about 55 ms with P1, about 30 ms with P3 |
| `sos_val`, `val` empty, 16 threads | 264 ms | as above |
| `sos_val`, `uniform = true`, 16 threads | 1250 ms | about 140 ms |
| `sos_val`, 1 thread | 839 ms | about 690 ms with P1, about 440 ms with P3 |
| GENERIC kernel, 1 thread, 5000 positions | 369 ms | about 195 ms |
| `sos_remove_bias`, 16 threads | 633 ms | about 50 ms |
| `sos_val`, 1 generator, 1,000,000 positions, 1 thread | 346 ms | not slower |

A speed-up that does not show is reported with its numbers and then either kept, if harmless, or removed.

Other criteria:
- No compiler warnings with gcc 12.
- The complete Catch2, MEX and Python suites pass at the end, in a build with AVX2 and in one without.
  `test_ray_mesh_interact` (MEX, line 113) fails on the committed code and is unrelated.
- The SOS Catch2 cases also pass in a build with `-fsanitize=address,undefined`.
- No file outside the list of section 2 is modified (`git status` at the end).


## 6. How to build and test

- `make` uses gcc 12; MEX files built with the system compiler (gcc 15) do not load in MATLAB R2024a. With
  `cmake` directly, pass `-D CMAKE_C_COMPILER=gcc-12 -D CMAKE_CXX_COMPILER=g++-12`.
- With AVX2 and MEX: `make avx2=ON matlab=ON`, then
  `cmake -B build_linux -D ENABLE_TESTS=ON && cmake --build build_linux --parallel`. `make test` alone
  leaves AVX2 off. For Python see `SPEC_PYTHON_API.md`.
- `build_linux/` may be in use by another session. If it is, build out of tree
  (`cmake -S . -B <scratch>/b_x ...`) and run `test_bin` from a directory that has a link `tests` to the
  repository's `tests`.
- SOS cases only: `test_bin "sos_*"`; MEX: the `test_sos_*.m` files; Python: `pytest -k sos`.
- The Catch2 tests are linked without BLAS: no `arma::dot` and the like in test code.
- `arma::mat x = {{0.7}};` does not compile; size the matrix and assign.
- `qd_mex_get_scalar<bool>` accepts logical and floating-point values only.


## 7. Rules for the session

- Do not commit. Leave the changes in the working tree for review.
- `make documentation` regenerates all of `html_docu/`, including pending changes of other work. Run it to
  check that the blocks compile, look at the three API files, then restore `html_docu/` with
  `git checkout -- html_docu` unless told otherwise.
- `/sjc/QuaDRiGa-NG` is read-only.
- Scratch programs and benchmarks go into the session's scratch space, not into the repository.
- Never change an expectation or a tolerance to make a test pass. The tests named in F7 are the exception,
  and only for the two-map values.
- Stop and ask only when: a file outside section 2 seems necessary, bit-identity cannot be kept, a decision
  of section 3 turns out to be impossible, or a test exposes a contradiction this plan and the review do
  not resolve. Otherwise work through to W7.
