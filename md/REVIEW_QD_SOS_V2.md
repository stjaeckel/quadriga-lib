# Review of the sum-of-sinusoids implementation, second pass

Reviewed on 2026-10-13, branch `qd-sos` at commit 99f8ff5 ("fixes and speed-ups after the review"). This
pass checks that the findings of `md/REVIEW_QD_SOS.md` (v1) landed as decided in `md/PLAN_QD_SOS_FIXES.md`,
and looks at the changed code for new problems. The findings are stated as of that commit; what was done
about them afterwards is in section 8 and marked "resolved" where it applies. Identifiers B, P, D and F are
those of v1 and of the plan.

## Summary

- All ten defects are fixed: every v1 reproduction now gives the expected error or the correct result.
- Bit-identity holds: 8853 output hashes of the library before and after agree for 1 and 13 threads,
  except the 6 two-map results of the random functions that decision F7 changed on purpose.
- All suites pass with and without AVX2 and under ASan/UBSan; only 4 statements stay unexecuted.
- Speed against a build of the old commit: `sos_val` 836 to 430 ms on one thread, 116 to 33 ms on 16
  (400 generators); no generator count is slower. The numbers of `md/LOG_QD_SOS.md` are confirmed.
- Found in the commit and resolved since (section 8): a stray `test.txt`, stale `html_docu/`, a
  use-after-free in `sos_val` when `val` is `pos`. Still open: the MEX helpers misread complex arrays.
- Documentation rules are met and all 12 example blocks run as written.

Marks: **[M]** measured in this session, **[R]** reproduced with a scratch program or script, **[C]** read
from the code.


## 0. Baseline

The scratch space of the first review was lost with a reboot, so all builds, reproductions and benchmarks
were made again, out of tree. For comparisons the library of commit 38d871a (before the fixes) was built
from `git archive` with the same compiler and flags.

| Build | Suite | Result [M] |
|---|---|---|
| gcc 12, AVX2 | Catch2, complete | `All tests passed (1196758 assertions in 1038 test cases)` |
| gcc 12, AVX2 | Catch2, `sos_*` | `All tests passed (39461 assertions in 108 test cases)`, 8.8 s on the idle machine (10.0 s before) |
| gcc 15, AVX2 | Catch2, `sos_*` | same counts, all passed |
| gcc 15, no AVX2 | Catch2, `sos_*` | `All tests passed (39230 assertions in 108 test cases)` |
| gcc 15, AVX2, ASan + UBSan, leak check on | Catch2, `sos_*` | all passed, no sanitizer report |
| gcc 15, `--coverage` | Catch2, `sos_*` | all passed |
| gcc 12, AVX2, MATLAB R2024a | MEX, complete | 61 of 62; `test_ray_mesh_interact` line 113 fails as before |
| gcc 12, no AVX2 | MEX, `test_sos_*` | 14 of 14 |
| gcc 15, AVX2 | Python, complete | 726 passed, 1 skipped |
| gcc 15, no AVX2 | Python, `-k sos` | 103 passed |

- gcc 12 and gcc 15 compile without warnings.
- These are the numbers `md/LOG_QD_SOS.md` reports. Not repeated here: the complete Catch2, MEX and Python
  suites without AVX2, and Octave.


## 1. Defects of v1

| # | Defect | State | Evidence |
|---|---|---|---|
| B1 | MEX `sos_randi`, negative or NaN `imax` | fixed | `imax` = -3, NaN, Inf, `[]`, 0, `int32(-2)`, `uint64(2^63)`: "Input 'imax' must be a finite number >= 1."; 2.9 gives values 1 and 2 as decided [R] |
| B2 | Python, `None` for a mandatory array | fixed | `TypeError` for `val`, `randn`, `map`, `remove_bias`, `acfi`, `calc_mse`, `acf_estimate`; `None` for optional arrays still works [R] |
| B3 | Long `float` tables rejected | fixed | 2001, 5001 and 20001 samples accepted; a point off by 0.3 of a step (`float`) or 2e-4 of a step (`double`) is still rejected [R] |
| B4 | `sos_acfi`, output aliasing an input | fixed | `val` as `acf`, `dist` or `d`: difference 0.0 to a separate output [R]; ASan run of the suite clean |
| B5 | Allocation failure inside the kernels | fixed | AVX2 under `ulimit -v` of 3 GB and 1.5 GB: `std::bad_alloc` caught, exit 0; GENERIC at 3 GB now completes [R] |
| B6 | Non-finite sample points and bins | fixed | `std::invalid_argument` from `sos_acf_approx`, `sos_acf_2d`, `sos_calc_mse`, `sos_acfi`, `sos_generate`, `sos_acf_estimate` [R] |
| B7 | ACF of -1 or below on the same map | fixed | rejected by `sos_val` and `sos_remove_bias`; -0.999 accepted and finite; not checked on two maps, as decided [R] |
| B8 | Unbounded search grid | fixed as decided | `freq_max` = 200 per metre accepted (2.1 s, the cap leaves up to 160 MB), 202, 1e6 and 1e300 rejected at once [R] |
| B9 | `phase_b` is `phase` in `sos_remove_bias` | fixed | rejected, phases untouched; equal values in another object accepted [R] |
| B10 | MEX, negative counts | fixed | `n_coeff`, `n_gen`, `n_dim`, `n_dir`, `n_pos`: "must be a finite number >= 1"; seeds NaN and Inf rejected; `[]` still selects the default [R] |

Traps of v1 section 1.2:

- Two-map values against the number of generators (F7): fixed. Generator 0 of a call with 1 and with 2
  entries: difference 0.0 for one device, same map and two maps. The result equals the documented
  composition with the second stream `seed + 2^62` in C++ and in MATLAB (difference 0.0) [R].
- Same seed in separate calls (F8): unchanged behaviour (2.2031 at the origin for 10 m and 50 m), now
  explained in all three introductions and in the function blocks [R, C].
- Huge coordinates, Python 1-D positions: documented [C].
- MEX type handling: see N4 below.


## 2. New findings

| # | Severity | Finding |
|---|---|---|
| N1 | resolved | `test.txt` at the repository root is part of commit 99f8ff5: 7332 lines of Catch2 output of a run without AVX2 ("1196509 assertions in 1038 test cases"). It is outside the file list of the plan and not mentioned in the log, which counts 76 files; the commit has 77 [M] |
| N2 | resolved | `html_docu/` was not regenerated, as the plan asked. The published API files still describe the old behaviour: `quadriga_lib_api_cpp.md` has "column `n_gen + g`" three times and the example with the undefined `pos_new` [M]. They need `make documentation` before a release |
| N3 | resolved | `sos_val` with `val` being the same object as `pos`: heap-use-after-free in `qd_SOS_GENERIC` (`sos_val_generic.cpp:47`) when the output has another size than `pos`, reported by ASan; without the sanitizer the call returns values that are off by up to 3.3 [R]. In v1 this call worked. The restriction is documented and was decided (F3), but the misuse is now undefined behaviour without a message. A comparison of addresses (`&val == &pos`, `pos_b`, `phase`, `phase_b`, `freq`) that throws `std::invalid_argument` costs nothing and turns it into an error |
| N4 | low, outside this port | Correction of v1: complex input arrays are not read with the imaginary part dropped. In MATLAB R2024a the shared helpers read the interleaved data as consecutive real numbers: `sos_val` with `complex(pos, pos+1)` returns exactly the values of the positions (0,2,0), (1,0,1), (1,1,0) [R]. The log found this; v1 had only checked that the call returns. The cause is in `src/mex_helper_functions.hpp` and concerns every MEX wrapper that takes an array |
| N5 | cosmetic | The file-local helpers `qd_mex_sos_count` and `qd_mex_sos_seed` are repeated in 7 wrapper files. Unavoidable under "do not edit the shared helper"; one more reason to move the check there |
| N6 | cosmetic | `src/quadriga_lib_helper_functions.hpp` now has `#include <algorithm>` and `<limits>` in the middle of the file, above the `qd_sos_*` block |

Checked in the new code and found correct:

- `sos_val.cpp` [C]: validation is complete before `val` is touched; weights and frequencies are copied
  before the output is resized; all parallel loops are element-wise over fixed ranges, with signed loop
  variables and nothing that can throw; at most one further array of output size for two devices.
- AVX2 kernel [C, R]: the packed weight layout matches the accumulation for every group size; the last
  group has exactly the remaining 1 to 7 generators; per-thread buffers are indexed by the thread number
  of a team whose size is fixed to the count the buffer was sized for; both buffers are freed on the
  failure path. Brute-force comparison over 475 combinations (1 to 19 generators, 1 to 33 sinusoids, 1 to
  131 positions): GENERIC within 6.4e-14, AVX2 within 1.9e-7 · sqrt(2·n_coeff) against a bound of 4e-7.
- The AVX2 object file still exports no weak symbol with AVX code [M].
- `sos_map` with cubes in one block of memory gives the values of separate cubes (difference 0.0) [R].
- NaN and Inf coordinates give NaN at that position only, in both kernels [R].
- Small calls are not slower: 1 position and 1 generator 9.3 to 11 µs before and after, `sos_randn` 22 µs
  both, measured alternately on the busy machine [M].
- `sos_acf_estimate` [C]: the pair buffer is bounded (at most 1.5e6 pairs per block), the block size does
  not depend on the thread count, and the sums are added in the old order.


## 3. Bit-identity

A scratch program (`repro/hash.cpp`, written for this pass and independent of the one used by the
implementation session) prints one FNV-1a hash of the output bytes per call and was linked once with the
old and once with the new library [M]:

- 8853 lines: `sos_val` in the three device modes, Normal and Uniform, both kernels, `double` and `float`,
  for 12 generator counts from 1 to 400, 5 sinusoid counts from 1 to 1024 and 6 position counts from 1 to
  1003; `sos_remove_bias` in three modes; `sos_map`; the three random functions; `sos_preset`;
  `sos_init_phases`; `sos_calc_mse` with all outputs; `sos_acf_2d`; `sos_acf_approx` in 1, 2 and 3
  dimensions; `sos_acfi`; `sos_acf_estimate` with the two-device estimate; `sos_generate` in 13
  configurations including unequal spacing.
- Old against new: 6 lines differ. They are the two-map columns of `sos_randn`, `sos_rand` and `sos_randi`
  for two presets; the same-map columns of the same calls are identical.
- 1 thread against 13 threads: no difference, for the old and for the new library.
- `sos_generate` with the documented settings regenerates `Comb128` to 2.1e-17 of the shipped table.


## 4. Test coverage

gcov on the `sos_*` cases, gcc 15, `-O3 --coverage` [M]:

- Statements never executed in either instantiation: `sos_val.cpp:412` and `sos_map.cpp:95` (AVX2
  requested but not available; both are executed in the build without AVX2, where the new test asserts the
  error), `sos_val_avx2.cpp:158` (early return for empty input, unreachable through `sos_val`) and
  `sos_val_avx2.cpp:183` (`throw std::bad_alloc`).
- The three statements that v1 listed are now executed: the cosine path for unequal spacing (403,271
  evaluations), the `pos_b` column check of `sos_remove_bias`, and the "left unchanged" side of the
  threshold (62 evaluations).
- AVX2 kernel: 99.4 % of the lines, 96 % of the branches taken, i.e. every group size from 1 to 8.
- Catch2 cases: 78 to 108. MEX: 14 files, extended. Python: 87 to 103 tests.

Every row of the table in v1 section 2.2 has its test, except:

- The lower limit of 1e-5 m of the first bin of `sos_acf_estimate` (positions cannot be injected).
- The dependence of `sos_randi` on the kernel (documented, not testable on one machine).
- B5 as a suite case (needs a memory limit on the process; reproduced by hand, see section 1).
- N3: no test at commit 99f8ff5, because the call was declared invalid; added with the fix (section 8).

Remaining "wrong implementations that would pass" from v1 section 2.4: the size of the threshold in
`sos_remove_bias` (only the exact-zero case is pinned) and the two symmetric cases, which are inherent. The
shipped tables are now pinned by checksums.


## 5. Performance

The machine was shared with another job (`build_linux/scratch/i43/...`, up to 29 threads) for parts of the
session. Old and new library were always measured back to back; each number is the fastest of at least 3
calls. Rows marked "busy" were taken while the other job ran.

One thread, ms per call, before to after (idle machine) [M]:

| Call | 1 gen, 1e6 pos | 4 gen, 2e5 pos | 8 gen | 20 gen | 400 gen | 400 gen, N 1024 | 2288 gen |
|---|---|---|---|---|---|---|---|
| AVX2 kernel | 328 to 320 | 70.9 to 70.6 | 40.5 to 36.9 | 59.4 to 50.0 | 683 to 438 | 2971 to 1712 | 4347 to 2900 |
| `sos_val` | 341 to 317 | 71.5 to 70.6 | 40.8 to 36.7 | 59.5 to 50.4 | 836 to 430 | 3117 to 1693 | 4955 to 2844 |
| `sos_val<float>` | 343 to 320 | 70.6 to 69.9 | 41.2 to 37.4 | 60.5 to 51.1 | 834 to 589 | 3114 to 1867 | 4921 to 3342 |
| two devices, same map | 693 to 657 | 150 to 143 | 91.6 to 75.3 | 139 to 103 | 1681 to 1020 | 6265 to 3567 | 11700 to 5781 |

256 sinusoids and 100,000 positions unless stated. No generator count from 1 to 2288 is slower.

16 threads, ms per call, before to after (idle at the start of the run) [M]:

| Call | 400 gen | 1 gen, 1e6 pos | 20 gen | 4 gen, 2e5 pos | 2288 gen | 400 gen, N 1024 |
|---|---|---|---|---|---|---|
| AVX2 kernel | 49.0 to 31.4 | 22.8 to 22.1 | 4.15 to 3.51 | 4.70 to 4.96 | 295 to 187 | 268 to 126 |
| `sos_val` | 116 to 33.2 | 36.3 to 22.8 | 5.42 to 3.59 | 5.51 to 4.93 | 646 to 191 | 330 to 131 |
| `sos_val<float>` | 111 to 89.9 | 35.8 to 24.5 | 5.11 to 4.05 | 5.44 to 5.44 | 588 to 518 | 314 to 180 |
| two devices, same map | 238 to 127 | 79.7 to 51.5 | 16.9 to 7.78 | 15.4 to 10.5 | 1315 to 757 | 658 to 328 |
| two devices, two maps | 249 to 129 | 75.9 to 45.1 | 16.5 to 7.76 | 14.4 to 10.1 | 1330 to 774 | 624 to 324 |

Other functions, one thread (idle) [M]: GENERIC kernel 4 to 20 generators 1.05 to 1.26 times; `sos_map`
77.5 to 51.9 ms; `sos_randn` with 400 equal distances 1001 to 431 ms; `sos_acf_approx` Comb1024 1306 to
688 ms; `sos_remove_bias`, `sos_calc_mse` (return value only) and `sos_acf_estimate` unchanged, as expected
for loops that only became parallel.

16 threads, busy, so only the direction counts [M]: `sos_remove_bias` 627 to 93 ms, `sos_calc_mse` with 300
directions 683 to 108 ms, `sos_acf_estimate` 1335 to 545 ms, `sos_val` Uniform 1241 to 306 ms. The log
gives 14.9, 14.7, 6.7 and 10.8 times for these on the quieter machine; they could not be reproduced to that
precision here.

Against the proposals of v1:

| # | State |
|---|---|
| P1 | Done. 3.5 times at 16 threads for 400 generators, 1.9 times on one thread, as estimated. `float` gains less (1.2 times at 16 threads), because the kernel computes in double and the `float` call keeps one temporary |
| P2 | Done |
| P3 | Done, 1.5 to 1.7 times on one thread and 1.6 to 2.1 times at 16 for 400 generators and more. One remainder group of the exact size instead of padding; no count is slower |
| P4 | Done: 1.26 times at 20 generators. The 1.9 times at 400 generators of the log could not be measured cleanly here (busy); the 16-thread GENERIC rows show 1.4 to 1.9 times |
| P5 | Done, with the squared errors added in the old order |
| P6 | Done |
| P7 | Done at 6.7 times (log) instead of the 14 times estimated in v1: the pairs are added serially to keep the result bit-identical. The estimate of v1 did not account for that |
| P8 | Done, 1.9 times on one thread |
| P9 | Not done, as decided |

Unchanged from v1 and not addressed, by decision: the error floor of the 2-D exponential tables; the
sensitivity of small calls to an oversubscribed machine (single calls of 6 ms instead of 10 µs occur with
the old and the new library when another job holds the cores).


## 6. Documentation

- Conformance script over all blocks [M]: no `*` or `|` in running text, all links resolve, no line of a
  function block above the limit (120, 100, 120). The MEX introduction has 37 lines above 100 characters
  (23 in v1), accepted by decision F10.
- Examples [R]: the four C++ blocks pasted into one `main` compile with `-Wall -Wextra` without a warning
  and run; the four MATLAB blocks and the four Python blocks run as written. All print 34.85 dB. The Python
  example no longer assigns to `map`.
- D1 to D9 of v1 [C]: the bound between the kernels now carries its range; "Uniform in [0, 1]" with the
  limits included; what a seed fixes and what depends on the kernel; the unchanged generator in all three
  APIs; "20 generators take less than twice as long as one". The header comments were shortened on request
  and no longer contain the statements D2, D3, D4 and D7 referred to; the comments that remain are correct.
- New limits are documented in all three APIs: ACF above -1, the grid of 100000 points, `phase_b` not
  `phase`, the second stream, +Inf for a perfect fit, finite sample points, the meaning of "equally spaced".
- The gaps of v1 section 4.4 are closed in the introductions: decorrelation distance defined, guide to the
  presets, rule of thumb for `sos_remove_bias`, coordinates and precision, threads and memory, 1-D positions
  in Python.
- Correction of v1: the rule of thumb now quotes losses of variance measured with the new `Comb256` table
  (2.5 % at 20 decorrelation distances, 14 % at 5, 44 % at 2). The 3 to 5 % that v1 quoted from the log
  belonged to the old tables.
- The Python entries were headed `sos_val` and so on, while the functions are called
  `quadriga_lib.sos.val`. Resolved, see section 8.
- Not verified word by word: the MEX and Python blocks of the functions other than `sos_val`,
  `sos_remove_bias` and the random functions.


## 7. Not checked

- MSVC, macOS, ARM, Octave.
- The complete Catch2, MEX and Python suites in the build without AVX2 (the log reports them as passing;
  here only the SOS parts were run without AVX2).
- 16-thread timings of the functions other than `sos_val` on an idle machine.
- The regeneration of the tables with 512 and 1024 sinusoids (62 s, reported in the log).
- The generated files in `html_docu/` at commit 99f8ff5, because they had not been regenerated. After the
  regeneration only the SOS links were checked (section 8).
- `md/REVIEW_QD_SOS.md` and `md/PLAN_QD_SOS_FIXES.md` changed on disk since they were written; they were
  not re-read for this pass.


## 8. Follow-up after this review (2026-10-13)

Changes made in the working tree on top of commit 99f8ff5, on request.

| Finding | What was done |
|---|---|
| N1 | `test.txt` removed (by Stephan) |
| N2 | `html_docu/` regenerated with `make documentation` (by Stephan, and again after the changes below) |
| N3 | `sos_val` compares the output with all seven inputs after validation and throws `std::invalid_argument("Output 'val' cannot be the same object as an input.")`. A second matrix object on the memory of an input is caught as well. `val` is left unchanged. The documentation bullet says that the call throws |
| Python headings | The 14 Python entries are headed `sos.val`, `sos.randn` and so on, and cross-references are written `[[sos.val]]`. The sentence about the `sos_` prefix in the introduction is replaced. `md/SPEC_PYTHON_API.md` has the rule |
| Markdown links | `tools/extract_md.py` builds the target of a cross-link with the slug it already uses for the index (one line). Needed for names with a dot |

Verification [M, R]:

- N3: new section "the output cannot be an input" in the validation case of `test_sos_val.cpp`: each input in
  turn as output, the ACF table and the sample points, and a view on the memory of `pos`; every call throws
  and the matrix keeps its values; a copy with the same values is accepted. Catch2 `sos_*`: `All tests passed
  (39480 assertions in 108 test cases)`. The `sos_val` cases under ASan and UBSan: no report. The
  reproduction of N3 now ends with the exception. The 8853 output hashes of section 3 are unchanged, so valid
  calls return the same values.
- Not re-run after the N3 change: the MEX and Python suites. Their wrappers allocate the output themselves
  and cannot reach the check.
- Dotted headings: `python_api.html` has 14 entries with the ids `sos.*` and 100 links to them, none
  unresolved; `quadriga_lib_api_python.md` has 14 headings and 100 links, none unresolved. Only documentation
  blocks of `api_python/qpy_sos_*.cpp` changed, no code.
- Tool change: the generated MEX files are unchanged. In `quadriga_lib_api_cpp.md` two lines changed, the
  links to the member function `.calc_directivity_dBi`, whose target is now the lower-case anchor of its
  heading; that link did not resolve before.

Still open:

- N4, complex arrays in the shared MEX helpers (`src/mex_helper_functions.hpp`), outside this port.
- N5 and N6 (cosmetic).
- The points of section 7.
