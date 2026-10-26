# Run 2a report — internal header, `lsf_init_phases`, `lsf_draw`

## Summary

- New internal header `src/quadriga_gbsm_helpers.hpp` (documentation block, `qd_gbsm_gen`, `qd_gbsm_lsf_generators`,
  `qd_gbsm_freq`, `qd_gbsm_positions`) with the signatures of spec 3.0.5 unchanged.
- `lsf_init_phases` (`src/lsf_init_phases.cpp`, with the generic `qd_gbsm_remove_bias`) and `lsf_draw`
  (`src/lsf_draw.cpp`, with the generic `qd_gbsm_eval` and `qd_gbsm_floor_offset`) implemented as in 3.1, 3.2, 3.2.1;
  the two declarations added to `include/quadriga_gbsm.hpp` verbatim from 2.1; final documentation blocks.
- Tests of 5.1 and 5.2 written first (25 cases), run against the stubs, then green against the implementation.
- Full Catch2 suite passes, no compiler warnings. No MEX or Python code touched (none needed in this run).

## Files

Created (5):
- `src/quadriga_gbsm_helpers.hpp`
- `src/lsf_init_phases.cpp`
- `src/lsf_draw.cpp`
- `tests/catch2_tests/test_lsf_init_phases.cpp` (10 test cases)
- `tests/catch2_tests/test_lsf_draw.cpp` (15 test cases)

Changed (2):
- `include/quadriga_gbsm.hpp` (declarations of `lsf_init_phases` and `lsf_draw`, text of 2.1)
- `tests/catch2_tests/qd_test_diag.hpp` (area line `QD_DIAG_GBSM`)

Plus this report. Not from this run: `md/prompts/impl_run2a.md` (untracked, the coordinator's prompt).

## Decisions and deviations

Review points of SPEC_CPP_API (not stops): A — no open contract question; the choices below settle everything.
B — tests and stub run below ("Run against the stubs"). C — this report.

Final form of the helpers (3.0.5): exactly the given signatures, no adaptation.
- `struct qd_gbsm_gen { double lambda; bool uniform; arma::uword mirror; bool bias; };`
- `static inline std::vector<qd_gbsm_gen> qd_gbsm_lsf_generators(const std::unordered_map<std::string, double> &sp);`
  reads the keys with `sp.at` (precondition: completed scenpar; never violated by the two callers).
- `static inline arma::uword qd_gbsm_freq(const std::string &preset, const std::vector<qd_gbsm_gen> &gen,
  std::vector<arma::mat> &freq, std::vector<arma::vec> &acf, std::vector<arma::vec> &dist, std::vector<arma::uword> &tab);`
  groups the distances by exact equality of the (already 0 -> 0.1 replaced) input values: equal values give identical
  tables, so this is a grouping of inputs, not a comparison of results.
- `static inline void qd_gbsm_positions(const arma::mat &rx_pos, const arma::mat &tx_pos, bool horizontal, arma::mat &R, arma::mat &T);`
- Anonymous namespace, `src/lsf_draw.cpp`: `arma::vec qd_gbsm_floor_offset(arma::uword n_coeff, arma::uword q);` and
  `void qd_gbsm_eval(gen, freq, acf, dist, tab, phase, R, T, floor_id, dual_mobility, val);` (full parameter list of 3.0.5).
- Anonymous namespace, `src/lsf_init_phases.cpp`: `void qd_gbsm_remove_bias(gen, freq, tab, P0, U);`.
- Generic as required by section 8: everything is driven by the generator table (`n_gen = gen.size()`, mirror,
  uniform and bias flags); one `sos_val` / `sos_remove_bias` call per frequency table (and per floor and side in
  `qd_gbsm_eval`) with all phase columns that use the table. The Tx side of generator g is grouped by the table of
  its mirror `m(g)`. No LSF-specific constant inside the three functions.

Choices within the spec:
1. Validation order of `lsf_init_phases` (3.1): the numbered list is followed (seed, preset, scenpar, positions,
   `tx_pos` requirement for dual-mobility bias removal); within the position checks the order of 3.0.2 (rx columns,
   tx columns, tx rows, rx finite, tx finite). The introductory phrase "after 3.0.2 for the inputs that are given"
   would put the position checks first, item 1 says the seed comes "before any other work"; item 1 is explicit, so
   it wins. Documented in the block.
2. Validation order of `lsf_draw` (3.2): 3.0.2 position checks, preset, scenpar, `phase` size, `phase` finite,
   `floor_id` length (the list of 3.2; 3.0.2 would place the length check of an optional vector inside the common
   block, the explicit list of 3.2 puts it last). Documented in the block.
3. The tx-row rule of `lsf_init_phases` applies only when `rx_pos` is given (3.1 item 4); in single mobility
   `tx_pos` is validated (columns, finite) but otherwise ignored.
4. `lsf_draw` passes `rnd` itself as `val` to `qd_gbsm_eval` instead of copying `val` into `rnd` (3.0.5 says
   "lsf_draw copies val into rnd"): same result, written in place, so a pre-sized output keeps its memory
   (tested). `qd_gbsm_eval` writes the Rx-side fields directly into `val`, the Tx side into a temporary.
5. Floor offsets: `2·pi·((k·n) mod N)/N` instead of `2·pi·k·n/N` — equal modulo 2·pi, keeps the argument below 2·pi.
   The test types the literal rule `2·pi·k·n/N + psi_b(n)` with its own `std::mt19937_64 gen(b)` and agrees within the
   kernel tolerance. `psi_b` uses `qd_sos_rng((arma::sword)b)` and `qd_sos_uniform` (b <= 2^57, no overflow).
6. Floor groups are processed in increasing ID order (`std::map`); with one group (no `floor_id`, or all IDs equal)
   the positions are used without copies. The values do not depend on the grouping (tested: one link at a time
   equals the joint call, with mixed floors, single and dual mobility).
7. Dual-mobility combination: pairs `(a + b)·sqrt(0.5)`, self-mirrored `(a + b)/sqrt(2 + 2·A(D))` with
   `qd_sos_acf_interp`; `D` from the projected positions in 2D mode, 3D otherwise. Swap reciprocity is bit-exact
   (addition is commutative, same `sos_val` evaluations in swapped roles) — tested with and without bias removal.
8. Bias removal in single mobility calls `qd_gbsm_positions(rx, tx or rx, ...)` (T unused); dual mobility uses
   `U = [R; T]` with the broadcast Tx rows (a single Tx row then counts `n_link` times, matching the link mean).
9. Phase-size message built with `std::to_string` of an integer (header-only integer `to_chars`, no
   floating-point `to_chars`; `nm` shows only `__to_chars_10_impl<unsigned long long>`).
10. `See also` lines: only functions that exist now (`lsf_draw` <-> `lsf_init_phases`, `sos_init_phases`,
    `sos_remove_bias`, `sos_val`); Run 4a adds the data-flow neighbours (`path_loss`, `lsp_generate`, ...) when they
    exist, so that no link points to an undocumented function in between.
11. Documentation: the TR is cited without a clause number for the 2D correlation ("TR 38.901 behaviour") and with
    Table 7.5-6 note 3 for the floor rule and Table 7.2-1 for the UT heights, as the spec does. No section numbers,
    no `md/` paths, no NG line numbers in any produced file (checked by grep).

Test-side choices (within 5.0 to 5.2):
12. Error messages are checked by exact text with a helper that catches `std::invalid_argument` (type and text in one
    check), stricter than `CHECK_THROWS_AS` alone.
13. "a decorrelation distance of 0 is used as 0.1 m": the 200 positions lie in a 10 m square (100 decorrelation
    distances) instead of 100 m, so that the arguments of the 0.1 m sinusoids stay small enough for 1e-12 (stated
    in a comment).
14. "fields of different floors ..." / "marginal distributions" / "the correlation follows the preset ACF": all 10
    decorrelation distances 10 m ("lambda 10 m"); the 4 directions are azimuth 0, 45, 90, 135 deg; all 9 Normal
    columns are checked (the table prints column 0).
15. Additions beyond the tables of 5.1/5.2 (from the SPEC_CPP_API checklist): pre-sized outputs keep their memory
    (`lsf_init_phases` cube; `lsf_draw` single mobility with floors and dual mobility); Tx broadcast (one Tx row) in
    the dual-mobility formula case; link independence also in dual mobility and with mixed floors; z independence also
    in dual mobility; an empty `scenpar` in the validation baselines. Determinism: repeated call in
    "the same seed gives the same phases" and in the first `lsf_draw` case.

Deviations from SPEC_CPP_API (approved in spec 1.4): no templates / no float instantiation / no float-double
parity case; diagnostics via `qd_test_diag.hpp` (area `QD_DIAG_GBSM`) instead of `QD_TEST_VERBOSE`; new internal
header; section 1.4 argument order. No other deviation.

## Test results

Build: `cmake --build build_linux --parallel 12` (configured with `ENABLE_TESTS=ON`); load average before the runs
0.7 to 2.8 (1-min), never above 3.1.

Run against the stubs (stub: validation-free, outputs sized and zeroed), `build_linux/test_bin "lsf_*"`:
```
test cases:  25 |   7 passed |  18 failed
assertions: 773 | 124 passed | 649 failed
```
- Failing for the right reason (zero output or no exception): all value checks (`ref_sos`, two-device formulas,
  floors, marginals, ACF, variances, zero means after bias removal), "slice 0 equals sos_init_phases", "the same seed"
  (`a != c` false for zeros), "no bias removal ..." (zeros are not the raw phases), every validation section (no
  exception thrown).
- Passing against the stub, as expected for zero outputs: the pure invariance / size cases ("output size follows the
  preset", "slice 1 is the mirror of slice 0" — zeros mirror themselves; "2D mode values do not depend on z",
  "single mobility ignores tx_pos", "a link does not depend on the other links", "swapping Tx and Rx ...", "sizes,
  empty input and pre-sized output") and the two validation baselines. They become meaningful with real values;
  nothing passed or failed unexpectedly.

Final, new cases: `build_linux/test_bin "lsf_*"`
```
All tests passed (777 assertions in 25 test cases)
```
Key diagnostic numbers (`QD_DIAG_GBSM=1`): `lsf_draw` against the double-precision `ref_sos` max 3.1e-7 (Normal),
7.3e-8 (Uniform), i.e. the AVX2 kernel is in use and well inside 2e-5 / 1e-5; zero drop means after bias removal
below 3.5e-16 (2D, 3D, dual, 0.1 m); max phase difference Tx/Rx swap 1.8e-15; dual mobility variances 0.98 to 1.014,
pair correlation 0.835 vs 0.837 (D 5 m), 0.533 vs 0.505 (10 m), 0.221 vs 0.200 (20 m); floor correlations within a
block below 0.024, floors 0/256 about -0.08 to -0.12 (one random sum, 2.3 standard deviations of 0.044); ACF at 10 m
along x 0.418 vs 0.368 (Comb256), as the spec predicts.

Full suite: `make test` with `CMAKE_BUILD_PARALLEL_LEVEL=12 OMP_NUM_THREADS=12` (load 2.78 before, 2.92 after):
```
All tests passed (1349928 assertions in 1332 test cases)
```
`make test` also ran the MEX suite with the existing MEX binaries of Run 1 (no MEX build in this run, the build
directory has `ENABLE_MATLAB=OFF`, `ENABLE_OCTAVE=OFF`):
- Octave: `FAILED (passed=67, failure=1)`; MATLAB R2024a (`matlab24 -batch`): `FAILED (passed=67, failure=1)`.
- The one failure is `test_ray_mesh_interact.m` line 113 (gain 0.98485 vs 0.69639), the same pre-existing failure
  recorded in the Run 1 report on the unchanged base commit; ray-tracing area, unrelated to this run (no MEX file
  links the new code). Not investigated (other session's files).

objdump gate (MEX files unchanged by this run):
`objdump -T +quadriga_lib/*.mexa64 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -u` -> highest `GLIBCXX_3.4.26` (nothing
above 3.4.28). The new object files contain no floating-point `to_chars`.

Python: not built or run (no Python change in Run 2a; spec Run 2a acceptance: "No MEX or Python build needed").

## Warnings

None. The two new sources were recompiled with `-O3 -Wall -Wextra -Wpedantic` (checked with `VERBOSE=1`) without a
warning; the full build log has no compiler warning. (The only "warning" in the `make test` log is the runtime
message of `test_qdant_write.cpp` that no decimal-comma locale is installed.)

## Open points

- Spec 3.1 "Validation": the phrase "after 3.0.2 for the inputs that are given" and item 1 ("checked here so that it
  comes before any other work") disagree on whether the position checks or the seed come first; implemented as the
  numbered list (decision 1). Proposal: drop "after 3.0.2" there or say "in this order".
- Spec 3.0.5 says "lsf_draw copies val into rnd"; passing `rnd` as `val` is equivalent and avoids a copy
  (decision 4). Proposal: "lsf_draw passes rnd as val".
- Spec 5.2 "the correlation follows the preset ACF" does not name the columns; all 9 Normal columns are checked.
- Not tested: GENERIC-kernel machines (the tolerances cover both kernels by construction); a phase cube whose slice 1
  is not the mirror of slice 0 (the function uses slice 1 as given, as specified); very large floor IDs (only the
  rule for b = 1 is typed in the test).
- The MEX failure of `test_ray_mesh_interact` persists from before Run 1 (see Test results).

## Suggested commit cut

1. Library and tests (one theme):
   - Files: `include/quadriga_gbsm.hpp`, `src/quadriga_gbsm_helpers.hpp`, `src/lsf_init_phases.cpp`,
     `src/lsf_draw.cpp`, `tests/catch2_tests/test_lsf_init_phases.cpp`, `tests/catch2_tests/test_lsf_draw.cpp`,
     `tests/catch2_tests/qd_test_diag.hpp`
   - Message:
     ```
     Large-scale fading generators: lsf_init_phases and lsf_draw

     lsf_init_phases draws the phases of the 11 large-scale generators of one site (DS, KF, SF, ASD, ASA, ESD,
     ESA, XPR, absolute TOA as Normal fields, ground permittivity and UMa environment height as Uniform fields)
     with the Tx side mirrored from the Rx side (ASD <-> ASA, ESD <-> ESA). The optional bias removal zeroes
     the sample mean of every Normal field over all positions where it is evaluated (Rx positions, or Rx and Tx
     positions together for dual mobility) and mirrors afterwards, so mirror relations and swap reciprocity hold
     exactly.

     lsf_draw evaluates the generators at a set of links: 2D correlation in the horizontal plane by default
     (3D as legacy option), dual mobility with one map for the self-mirrored generators and two maps without
     further mixing for ASD/ASA and ESD/ESA, and per-floor phase offsets that make the fields of different
     floors uncorrelated at equal horizontal positions.

     The evaluation, the floor offsets and the bias removal are written generically over a generator table so
     that the small-scale generators can reuse them. New internal header quadriga_gbsm_helpers.hpp with the
     helpers shared by the source files of the model; diagnostic area QD_DIAG_GBSM for the tests.

     Catch2: 25 new test cases against a sum-of-sinusoids oracle typed in the tests, the two-device formulas,
     typed floor offsets and statistical checks; full suite passes.

     Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
     ```
2. Planning notes (last, separate):
   - Files: `md/reports/r2a_report.md` (and `md/prompts/impl_run2a.md` if the maintainer keeps prompts in git)
   - Message:
     ```
     Planning notes: report of the large-scale fading generator implementation

     Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
     ```
