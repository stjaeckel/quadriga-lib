# Run 2b report — `lsp_generate`, shared LSP mean/std and Cholesky

## Summary

- `lsp_generate` (`src/lsp_generate.cpp`) implemented as in 3.0.4 and 3.4: validation, mean/std per link and
  frequency, Cholesky root in TR order or the symmetric square root (coupled Newton-Schulz, file-local), linear
  LSPs with clipping, absolute TOA, ground permittivity, EoD offset; final documentation block (3.4.4).
- Helper header extended by `qd_gbsm_lsp_mu_sigma` and `qd_gbsm_chol8` with the signatures of 3.0.5;
  `qd_scenpar_xcorr` now calls `qd_gbsm_chol8` (own loop removed; the library has one 8x8 Cholesky).
- Tests of 5.4 written first (16 cases), run against the stub, then green against the implementation.
- Full Catch2 suite, MEX suite (Octave, R2024a) and Python suite run; no new failure, no compiler warnings.

## Files

Created (2):
- `src/lsp_generate.cpp`
- `tests/catch2_tests/test_lsp_generate.cpp` (16 test cases)

Changed (3):
- `include/quadriga_gbsm.hpp` (declaration of `lsp_generate`, text of 2.1, after `lsf_draw`)
- `src/quadriga_gbsm_helpers.hpp` (`qd_gbsm_lsp_mu_sigma`, `qd_gbsm_chol8`, documentation block: users list, LSP
  order and units of the mean/std cubes, unit of carrier frequencies; includes `<algorithm>`, `<cmath>`)
- `src/qd_scenpar.cpp` (`qd_scenpar_xcorr` calls `qd_gbsm_chol8`; includes `quadriga_gbsm_helpers.hpp`)

Plus this report. Not from this run: `md/prompts/impl_run2b.md` (untracked, the coordinator's prompt).

## Decisions and deviations

Review points of SPEC_CPP_API (not stops): A — no open contract question, the choices below settle everything.
B — tests and the stub run below. C — this report.

Final form of the helpers (3.0.5): signatures exactly as given.
- `static inline void qd_gbsm_lsp_mu_sigma(const std::unordered_map<std::string, double> &sp, const arma::vec &fc,
  const arma::mat &tx_pos, const arma::mat &rx_pos, arma::cube &mu, arma::cube &sigma);` — precondition: completed
  scenpar, validated positions and `fc`; outputs resized only if their size differs. The SF keys that do not exist
  (`SF_mu`, `SF_gamma`, `SF_epsilon`, `SF_zeta`, `SF_alpha`) are looked up with `find` and count as 0; all other keys
  are present in a completed scenpar.
- `static inline bool qd_gbsm_chol8(const double *A, double *L);` — L is zeroed first (zeros above the diagonal),
  only the lower triangle of A is read. The loop is the former loop of `qd_scenpar_xcorr` (column by column, pivots
  in ascending order, the same expression and summation order per element as Cholesky-Banachiewicz), so the
  positive-definite flag is bit-identical to before; the comment says "Cholesky factorization" instead of
  "Cholesky-Banachiewicz" for that reason.

Choices within the spec:
1. Validation order: 3.0.2 (rx columns, tx columns, tx rows, rx finite, tx finite, `fc` empty, `fc` values), then
   3.4.1 in its order (`rnd` size, `rnd` finite, `sf_sigma` size, `sf_sigma` values, scenpar, positive definite,
   column 9). The tx-row rule "1 or n_link" covers `n_link = 0` with 0 rows. For `n_link = 0`, `rnd` must be
   `[0, 11]` (what `lsf_draw` returns). Documented in the block.
2. The positive-definite check covers both roots (return value of `scenpar_validate`); with `cholesky = true` the
   same message is thrown when `qd_gbsm_chol8` fails on the permuted matrix. All of this runs before any output is
   written (tested: a pre-filled `lsp` stays untouched for both roots).
3. A frequency term whose coefficient is 0 (`P_gamma`, `P_delta`) is skipped, so `lg(P_omega + f')` that is not
   finite (only possible for `P_omega + f' <= 0`, `P_omega` has class "any value") does not turn `0·(-Inf)` into
   NaN. Identical to the formula for every finite value; stated in the block and in the helper comment.
4. Symmetric square root: written as in 3.4.3 on `double[64]` with hand-written 8x8 products (no Armadillo
   products, so nothing depends on the BLAS setting). The update computed with the final `e` is kept; then
   `S = sqrt(s)·(Y + Y^T)/2`.
5. `lsp_mu` / `lsp_sigma`: the mean and std are computed once into local cubes (needed for `lsp` anyway) and copied
   into the requested outputs; the `sf_sigma` replacement is applied before, so `lsp_sigma` slice 2 holds it.
6. EoD offset: the geometric `hcl` term is evaluated per link (frequency-independent), the power term per frequency
   with `f'`; `d`, `h` with the same clamps as the mean/std formula.
7. `abs_toa` uses `c = 299792458` m/s; `absTOA_Lmax = 0` gives no bound (`+Inf`).
8. No OpenMP: the loops are short (the 20000-link statistical case runs in milliseconds).
9. `See also`: only functions that exist now (`lsf_draw`, `lsf_init_phases`, `scenpar_validate`), as in Run 2a;
   Run 4a adds `path_loss`.

Test-side choices (within 5.0 and 5.4):
10. Matrix products in the test are written out (`mm`): the test binary links no BLAS, and Armadillo calls the BLAS
    wrappers for products above 4x4 (the first build failed to link with `wrapper2_dgemm_`). Proposal for
    SPEC_CPP_API: state that tests must not use Armadillo matrix products of larger matrices either.
11. `arma::max` skips NaN: the cases that compare a maximum (`sample covariance`, `symmetric square root`, Cholesky
    root) first `REQUIRE` finite values; without this the stub passed the covariance case.
12. "mean and std follow the formula with every term": every coefficient non-zero and distinct (table in the test);
    links: hUT above hBS (with a separate Tx row), hUT below hBS, d2D 1000 m and 5000 m (the ESD floor acts for both;
    at 5000 m the ASD std is negative before the outer limit at all frequencies, so `lsp` slice 3 equals `10^mu`);
    the inner limit of the KF std acts only at 28 GHz.
    The fixture checks these conditions itself (sanity `CHECK`s).
13. TR spot checks: InF NLOS DL has no frequency in 5.4; 3.5 and 28 GHz are used. SMa NLOS: Table 7.5-12 has no
    distance or height term for the ZSD (0.14 for all heights); the spec's "the ESD height terms" is therefore
    checked as "no height dependence" at hUT 1.5 and 4.5 m (and the EoD case covers the SMa `hcl` term).
    SF std from Table 7.4.1-1 (UMa NLOS 6, UMi LOS 4, SMa NLOS 8, InF-DL 7.2 dB).
14. "InF delay spread uses V/S": InF LOS with `DS_VS` 4 and 50/11 as the spec table says (the LOS hall is 47/11;
    any V/S works for the formula check), file default 4.27 against 47/11 within 2.5e-4 (actual 2.46e-4).
15. EoD formula case: `b` 20 and 0, `hcl` 6 m with hUT 1.5 and 9 m, d2D {5, 50, 500, 2000}, `LSP_fc_min` 4 with
    fc 2 and 10 GHz (so the clamp acts); TR checks UMa NLOS (d2D 10 and 200 m, hUT 1.5 and 12 m, 3.5 and 28 GHz),
    SMa NLOS (d2D 50 and 200 m, hUT 1.5 and 4.5 m), UMa LOS (0).
16. "the same random values apply to all frequencies": UMa LOS at 7, 15, 28 GHz (above its 6 GHz clamp, so the
    means differ), all slices with std > 0, `clip_spreads = false`.
17. "LSP_fc_min clamps the frequency" also checks the EoD offset (2 GHz equals 6 GHz, 10 GHz differs).
18. "ground permittivity": `LSP_fc_min = 20` in the scenpar, to show that `f` is not clamped.
19. Determinism (one per file) is in the optional-outputs case; it also checks that pre-sized outputs keep their
    memory and that wrongly sized ones are resized.

Deviations from SPEC_CPP_API (approved in spec 1.4): no templates / float instantiation / parity case; diagnostics
via `qd_test_diag.hpp` (area `QD_DIAG_GBSM`); internal helper header; argument order of 2.1. No other deviation.
No deviation from the spec.

## Test results

Load average (1-min) before the Catch2 runs 1.2 to 3.9; before the MEX build 3.6, during the MEX and Python runs
10 to 21 (other sessions). No timing results are reported, so the load does not affect any result.

Run against the stub (outputs sized and zeroed, no validation), `build_linux/test_bin "lsp_generate*"`:
```
test cases:  16 |   1 passed |  15 failed
assertions: 984 | 406 passed | 578 failed
```
- Failing for the right reason (zeros or no exception): every value check (mean/std, TR, EoD, 10^(...), roots,
  covariance, clipping, absolute TOA, permittivity, `sf_sigma`, `LSP_fc_min` differences, V/S), every validation
  section.
- Passing against the stub, as expected: "optional outputs alone and together give the same values" (zeros are
  equal, sizes are right), the validation baseline, the size `REQUIRE`s and the fixture sanity checks.
- One test fix before the implementation (decision 11): the covariance case passed against the stub because
  `arma::max` skips NaN; with the added `REQUIRE(C.is_finite())` it fails as it should (the counts above are after
  the fix).

Final, new cases: `build_linux/test_bin "lsp_generate*"`
```
All tests passed (1057 assertions in 16 test cases)
```
Key diagnostic numbers (`QD_DIAG_GBSM=1`): Cholesky root against the test-side root 4.4e-16, `Q·Q^T - R` 4.4e-16;
symmetric root `X·X - R`: 7.8e-16 (custom), 8.9e-16 (0.999999), 4.4e-16 (identity), `X - X^T` <= 4.7e-16; sample
covariance of 20000 links within 0.0154 (Cholesky) / 0.0152 (symmetric) of R; InF NLOS DL DS mean with the file
default 3.5e-4 from the TR value for V/S = 50/11 (bound 3.52e-4); InF LOS 2.46e-4 (bound 2.46e-4).

Conf tests after the `qd_scenpar_xcorr` refactor: `build_linux/test_bin "conf_*,scenpar_validate*"` ->
`All tests passed (6033 assertions in 45 test cases)`.

Full Catch2 suite (AVX2 on, `OMP_NUM_THREADS=12 build_linux/test_bin`):
```
All tests passed (1350985 assertions in 1348 test cases)
```
`make test` (`CMAKE_BUILD_PARALLEL_LEVEL=12 OMP_NUM_THREADS=12`; Makefile configuration: AVX2 off, gcc-12, MATLAB
and Octave on):
```
All tests passed (1349062 assertions in 1347 test cases)
```
(one case less: `quadriga_lib_has_AVX2` exists only with AVX2).

MEX: built with the configuration of `make` (gcc-12, `ENABLE_MATLAB=ON`, `ENABLE_OCTAVE=ON`, AVX2 off,
`cmake --build build_linux --parallel 12`, `cmake --install`), run by `make test`: Octave `/usr/bin/octave`,
MATLAB R2024a `matlab24` = `/usr/local/MATLAB/R2024a/bin/matlab`:
- Octave: `FAILED (passed=67, failure=1)`
- MATLAB R2024a: `FAILED (passed=67, failure=1)`
- The one failure is `test_ray_mesh_interact.m` line 113 (0.98485 vs 0.69639), the pre-existing failure recorded
  in the Run 1 and Run 2a reports (ray-tracing area, fails on the unchanged base commit). All conf and
  `scenpar_validate` MEX tests pass with the rebuilt MEX files (they link the refactored `qd_scenpar.cpp`).
  R2026a was not run.

objdump gate: `objdump -T +quadriga_lib/*.mexa64 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -u` -> 3.4.9, 3.4.11,
3.4.18, 3.4.19, 3.4.20, 3.4.21, 3.4.26 — nothing above 3.4.28.

Python: `make python` (`CMAKE_BUILD_PARALLEL_LEVEL=12`), `make python_test`:
`816 passed, 3 skipped in 2.55s` (no Python change in this run; the conf functions link the refactored
`qd_scenpar.cpp`).

The build tree `build_linux` was restored afterwards to the configuration it had at the start (tests on, AVX2 on,
MATLAB/Octave/Python off) and rebuilt; `lsp_generate*, lsf_*, conf*, scenpar*` -> `All tests passed (10388
assertions in 93 test cases)`.

## Warnings

None. `src/lsp_generate.cpp` and `src/qd_scenpar.cpp` were recompiled with `-O3 -Wall -Wextra -Wpedantic`
(checked with `VERBOSE=1`) without a warning; the library, MEX and Python build logs have no compiler warning. The
only "warning" in the `make test` log is the run-time Catch2 `WARN` of `test_qdant_write.cpp:386` (no decimal-comma
locale installed), pre-existing.

## Open points

- Spec 5.4, TR spot checks: "SMa NLOS ... the ESD height terms" — TR Table 7.5-12 has no height (or distance) term
  for the SMa ZSD, and the SMa NLOS file has none; checked as "no height dependence" (decision 13). Proposal: say
  "the ESD without height terms" there, or move the height point to the EoD case where `hcl` acts.
- Spec 5.4 does not name the frequencies of the InF NLOS DL spot check (3.5 and 28 GHz used) and names 50/11 as a
  `DS_VS` value of the InF LOS case, whose hall has 47/11 (50/11 used as written; any value checks the formula).
- Not tested: the second positive-definite failure path (`scenpar_validate` accepts `R`, the Cholesky of the
  permuted matrix fails); a nearly singular matrix with this property was not constructed. The code path is one
  `if` with the same message.
- Not tested: `P_omega + f' <= 0` with a zero coefficient (decision 3); no conf file has a negative `_omega`.
- SPEC_CPP_API proposal: tests must not use Armadillo matrix products above 4x4 (no BLAS in the test binary).
- The MEX failure of `test_ray_mesh_interact` persists from before Run 1.

## Suggested commit cut

1. Library and tests:
   - Files: `include/quadriga_gbsm.hpp`, `src/quadriga_gbsm_helpers.hpp`, `src/lsp_generate.cpp`,
     `src/qd_scenpar.cpp`, `tests/catch2_tests/test_lsp_generate.cpp`
   - Message:
     ```
     Large-scale parameters: lsp_generate

     lsp_generate turns the normalized random values of lsf_draw into the eight large-scale parameters of a set
     of links (DS, KF, SF, ASD, ASA, ESD, ESA, XPR) per link and frequency. Mean and standard deviation follow
     the scenario parameters with the frequency, distance, height and elevation terms, the lower frequency limit
     LSP_fc_min, the ESD terms and floor of TR 38.901 v19 and the V/S term of the InF delay spread; the standard
     deviation is limited at 0. The cross-correlation uses the Cholesky root in the TR order (default) or the
     symmetric square root (legacy), with the same random vector for all frequencies; spreads are clipped at 104
     and 52 degrees. Optional outputs: absolute TOA offset with its room bound, ground permittivity, EoD offset,
     and the mean and standard deviation in the log domain. An SF standard deviation from the path-loss model can
     replace the scenario value.

     The LSP mean/std and an 8x8 Cholesky factorization move into the internal helper header; the
     positive-definite check of the scenario parameters uses the same Cholesky.

     Catch2: 16 new test cases with the mean/std formula typed in the test, TR 38.901 spot checks of four v19
     files, a test-side Cholesky, the defining properties of the symmetric root and a statistical check of the
     cross-correlation; full suite passes.

     Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
     ```
2. Planning notes (last, separate):
   - Files: `md/reports/r2b_report.md` (and `md/prompts/impl_run2b.md` if the maintainer keeps prompts in git)
   - Message:
     ```
     Planning notes: report of the large-scale parameter implementation

     Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
     ```
