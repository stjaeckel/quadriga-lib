# Run 3b report — `los_probability`, `o2i_loss`

## Summary

- `los_probability` (`src/los_probability.cpp`) implemented as in 3.5: validation (3.5.1), the 18 models of 3.5.2
  with the InF/SMa defaults and edge rules, the [0, 1] limit, the soft LOS state (3.5.3) with the file-local
  probit (Acklam + one Halley step, symmetric form, 3.5.4); final documentation block (3.5.5).
- `o2i_loss` (`src/o2i_loss.cpp`) implemented as in 3.6 (validation 3.6.1, algorithm 3.6.2); documentation block
  (3.6.3). Both declarations added to `include/quadriga_gbsm.hpp` (text of 2.1, after `lsp_generate`).
- Tests of 5.5 (10 cases) and 5.6 (7 cases) written first, run against stubs, then green against the implementation.
- Full Catch2 suite, `make test` (Catch2 + MEX in Octave and R2024a), objdump gate, `make python` / `make python_test`
  run; no new failure, no compiler warning.
- No contradiction in the spec found; three small gaps decided within the spec (decisions 2, 3, 6).

## Files

Created (4):
- `src/los_probability.cpp`
- `src/o2i_loss.cpp`
- `tests/catch2_tests/test_los_probability.cpp` (10 test cases)
- `tests/catch2_tests/test_o2i_loss.cpp` (7 test cases)

Changed (1):
- `include/quadriga_gbsm.hpp` (the two declarations of 2.1, verbatim; checked with `diff` against the spec text)

Plus this report. Not from this run: `md/prompts/impl_run3b.md` (untracked, the coordinator's prompt).
`src/quadriga_gbsm_helpers.hpp` is unchanged: neither function uses a shared helper (the probit is file-local as
3.0.5 requires; the 38.811 interpolation uses the existing `qd_sos_acf_interp` of
`src/quadriga_lib_helper_functions.hpp`, as 3.5.2 states).

## Decisions and deviations

Review points of SPEC_CPP_API (not stops): A — contract complete apart from decisions 2 and 3; B — tests and stub
run below; C — this report.

1. Validation order of `los_probability` (3.5.1): model; `d2d_out`; `h_bs` size, `h_bs` finite, `h_ut` size,
   `h_ut` finite (per argument, shape before value, SPEC_CPP_API); `d_clutter`, `r_clutter`, `h_clutter`; then the
   soft-LOS block: presence of `rnd_g` and `fc`, `rnd_g` length, `rnd_g` finite, `fc` empty, `fc` values. All before
   any output is written.
2. Messages that 3.5.1 does not spell out (item 5): `Number of elements in 'rnd_g' does not match number of
   elements in 'd2d_out'.` (the 3.0.2 pattern, with `d2d_out` in place of `rx_pos`, which this function does not
   have), `Input 'rnd_g' must be finite.`, and for `fc` the 3.0.2 messages `Input 'fc' cannot be empty.` /
   `Input 'fc' must be finite and > 0.`. `d_clutter` = +Inf is rejected ("finite > 0").
3. Probit (3.5.4): the Halley step is skipped when `u = e·sqrt(2·pi)·exp(x²/2)` is not finite. This happens only
   for `p` below about 1e-308 (`|x| > 37.7`, where `exp(x²/2)` overflows and `0·Inf` would give NaN); Acklam's
   value is then kept. Not in the spec text; it only adds robustness for subnormal probabilities (e.g. RMa at
   d2D > 700 km), which the accuracy target of 3.5.4 does not cover.
4. The probit is evaluated once per link (not per link and frequency); `sqrt(20/lambda)` is computed as
   `sqrt(20·fc/c)` with `c = 299792458`.
5. Model identifiers: a file-local enum with the prefix `QD_LOS_` and the 18-name table (no clash with macros such
   as `LOS`); the 38.811 tables and the elevation grid are file-local constants (3.0.5: "the tables of section 11
   file-local"). Elevation `atan2(hB, d)·180/pi`, interpolated by `qd_sos_acf_interp` (constant beyond the ends).
6. `o2i_loss`: the cumulative weights are divided by their sum (3.6.2), so weights that pass the 1e-6 sum check
   are normalized (tested: `{0.5000004, 0.5000004}` splits at u = 0.5). The fallback "last j with w_j > 0" covers
   `u` at or above the last cumulative value (u = 1). `loss = PL_tw + (0.5·d2din + sigma_P·N)` (wall loss
   precomputed per model type and frequency). No OpenMP (optional per 9.1; the loop is a few flops per UT).
7. Validation order of `o2i_loss` exactly as 3.6.1; the uniform-column check uses `!(v >= 0 && v <= 1)`, so NaN
   fails it; `rnd` with a wrong column count is rejected also for 0 rows.
8. Tests: the table of 5.5 has 9 cases; one case was added, "los_probability - pre-sized outputs, empty input and
   repeated calls" (SPEC_CPP_API "What every suite covers": pre-sized outputs, `n_link = 0`, and the determinism
   check of 5.0). In `test_o2i_loss.cpp` the determinism check is in "optional outputs and sizes". The validation
   cases have one `SECTION` per argument of the validation list, each with all checks of that argument and the exact
   message (as the earlier gbsm tests).
9. Test oracles (5.5): the soft-LOS quantile-grid check compares against the typed TR probability of RMa (not the
   returned `p_los`), so that it fails against the stub; the inverse-CDF case asserts on the typed TR values that
   the grid reaches 1e-12 on both tails. UMa above hUT = 23 m: the oracle uses `C' = 1` as the spec states (outside
   the TR range; stated in the test header).
10. Test "InF edge rules": hU >= h_c (equal and above, default and explicit h_c), hB < hU < h_c and hB = hU < h_c
    (SH with explicit h_c 3 m, DH with the default 6 m), r = 0 for all four InF models with hB above and below hU.
    "SMa edge rules": also checks that the literal TR form (no `max(., 0)` on the vegetation term) gives a larger
    value at hU = 18 m, i.e. that the rule is active, and an explicit `d_clutter`.
11. Spec vs. QuaDRiGa (information, no deviation): QuaDRiGa's `set_scenario` uses r = 0.4 for InF-DH; the library
    uses the TR value 0.6 (Table 7.2-4 / 7.8-7) as 3.5.2 states.

## Test results

Load average (1-min): 4.2 to 7.8 during the Catch2 runs; 7.1 before `make`; 73 at the start of `make test`
(other sessions; MATLAB/Octave), 32 before `make python`. No timing results are reported, so the load does not
affect any result.

Run against the stubs (outputs sized and zeroed, no validation), `build_linux/test_bin "los_probability*"`,
`"o2i_loss*"`:
```
los_probability: test cases:  10 |  1 passed |  9 failed;  assertions: 1816 | 1459 passed | 357 failed
o2i_loss:        test cases:   7 |  1 passed |  6 failed;  assertions:  357 |  130 passed | 227 failed
```
- Failing for the right reason (zeros, no exception): TR models (127 of 127), 38.811/QuaDRiGa/fixed (40 of 56;
  passing are the NLOS == 0 and below-horizon == 0 checks), [0, 1] limit (3 of 1303: the UMa values; the grid
  check passes with zeros), InF edge rules (18 of 18), SMa edge rules (9 of 12; passing are the checks on the
  oracle side), soft LOS, inverse CDF, the two throw checks of "optional inputs", validation (53 of 110; passing
  are the baseline sections); o2i: building models (134 of 174), legacy (27 of 34), car (10 of 54; d2d_in = 0 and
  the f-independence pass with zeros, as expected), model choice (11 of 15), indoor distance (6 of 9), validation
  (39 of 45).
- Passing against the stubs, as expected: "pre-sized outputs, empty input and repeated calls" and "optional outputs
  and sizes" (zeros are equal, sizes right).
- Two oracle weaknesses found by the stub run and fixed before the implementation: the quantile-grid fraction and
  the tail-coverage check compared with the returned `p_los` (0 = 0 against the stub); now the typed TR value
  (decision 9). After the fix: soft LOS 40 | 10 passed | 30 failed, inverse CDF 87 | 3 passed | 84 failed.

Final, new cases: `build_linux/test_bin "los_probability*,o2i_loss*"`
```
All tests passed (2173 assertions in 17 test cases)
```
Diagnostics (`QD_DIAG_GBSM=1`): TR models maximum difference below 1e-15; soft LOS at `G = -F(p)` within 1.2e-14
of 0.5 (tolerance 4.9e-10 to 8.1e-9); inverse normal CDF through `los_soft` maximum relative error 1.6e-14 over
p in [1e-12, 1 - 1e-12] (target 1e-10; the residual is the rounding of `los_soft` in the recovery); quantile
fractions equal p within 1e-4; o2i model counts 200 / 0 / 500 / 300; d2D-in CDF on the 100 x 100 grid equal to
`1 - (1 - x)²` to 4 decimals.

Full Catch2 suite (build tree with AVX2 on, `OMP_NUM_THREADS=12 build_linux/test_bin`):
```
All tests passed (1355498 assertions in 1374 test cases)
```
`make test` (`CMAKE_BUILD_PARALLEL_LEVEL=12 OMP_NUM_THREADS=12`, after `make`; Makefile configuration: AVX2 off,
gcc-12, MATLAB and Octave on):
```
All tests passed (1353575 assertions in 1373 test cases)
```
MEX (run by `make test`): Octave `/usr/bin/octave`, MATLAB R2024a `matlab24`:
- Octave: `FAILED (passed=67, failure=1)`
- MATLAB R2024a: `FAILED (passed=67, failure=1)`
- The one failure is `test_ray_mesh_interact.m` line 113, the pre-existing failure of the earlier runs
  (ray-tracing area). No MEX file changed in this run. R2026a was not run.

objdump gate: `objdump -T +quadriga_lib/*.mexa64 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -u` -> 3.4.9, 3.4.11,
3.4.18, 3.4.19, 3.4.20, 3.4.21, 3.4.26 — nothing above 3.4.28 (77 MEX files).

Python: `make python` (`CMAKE_BUILD_PARALLEL_LEVEL=12`), `make python_test`: `816 passed, 3 skipped in 30.94s`
(no Python wrapper in this run).

The build tree `build_linux` was restored afterwards to its configuration at the start (tests on, AVX2 on,
MATLAB/Octave/Python off) and rebuilt; `los_probability*, o2i_loss*, path_loss*, lsp_generate*, lsf_*, conf*,
scenpar*` -> `All tests passed (14901 assertions in 119 test cases)`.

## Warnings

None. `src/los_probability.cpp` and `src/o2i_loss.cpp` were recompiled with `-O3 -Wall -Wextra -Wpedantic`
(checked with `VERBOSE=1`) without a warning; the `make`, `make test` and `make python` logs have no compiler
warning. The only "warning" in the `make test` log is the run-time Catch2 `WARN` of `test_qdant_write.cpp:386` (no
decimal-comma locale installed), pre-existing.

## Open points

- Spec 3.5.1 item 5: proposal to write the `rnd_g` length and finiteness messages into the spec (decision 2).
- Spec 3.5.4: proposal to mention the skipped Halley step for subnormal `p` (decision 3).
- Spec 5.5: the table has no case for pre-sized outputs, `n_link = 0` and determinism; proposal to add the case of
  decision 8.
- Not tested: subnormal probabilities (decision 3; the soft LOS is then 0 to double precision for any reasonable
  `rnd_g`); OpenMP (none used).
- The MEX failure of `test_ray_mesh_interact` persists from before Run 1.

## Suggested commit cut

One commit (the two functions share the header edit and form the layout-level part of the model):
- Files: `include/quadriga_gbsm.hpp`, `src/los_probability.cpp`, `src/o2i_loss.cpp`,
  `tests/catch2_tests/test_los_probability.cpp`, `tests/catch2_tests/test_o2i_loss.cpp`
- Message:
  ```
  LOS probability and O2I penetration loss: los_probability, o2i_loss

  los_probability evaluates the LOS probability of TR 38.901 Table 7.4.2-1 (RMa, UMi, UMa, InH mixed and open
  office, the four InF sub-scenarios and InF-HH, SMa with vegetation), the TR 38.811 elevation tables, the
  QuaDRiGa industrial model and the fixed models LOS and NLOS, limited to [0, 1]. The InF and SMa clutter
  geometry are arguments with the TR values as defaults; for a high-BS InF deployment with the BS not above the
  UT, the rule of the low-BS sub-scenarios applies. Optionally it returns the soft LOS state of eq. (7.6-18) per
  link and frequency, with an inverse normal CDF accurate to 1e-10 on both tails.

  o2i_loss draws per UT one of the O2I penetration-loss models of TR 38.901 clause 7.4.3 (low-loss, high-loss
  with the Release 19 IRR glass, low-loss A, the backward compatible model for single-frequency simulations
  below 6 GHz, car and metallized car) by cumulative weights and returns the loss per frequency, the indoor 2D
  distance (minimum of two uniforms) and the chosen model.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```
