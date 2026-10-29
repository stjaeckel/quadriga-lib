# SSF series, run R3: wrappers, documentation, gates

## Summary
MEX and Python wrappers for `ssf_init_phases`, `ssf_draw` and `ssf_clusters`, with tests. `SECTION_DESC` (C++, MEX,
Python) extended to steps 1 to 7, See-also lines added, `html_docu` regenerated with `make documentation`. All the
new tests pass in Octave, MATLAB R2024a and R2026a and in pytest. The full Catch2 suite and the full Python suite
pass. The full MEX suite passes except `test_ray_mesh_interact`, which this run did not touch (Risks). No warnings in
any build. Nothing committed.

## Files
| File | Lines |
|---|---|
| `api_mex/ssf_init_phases.cpp` (new) | 110 |
| `api_mex/ssf_draw.cpp` (new) | 146 |
| `api_mex/ssf_clusters.cpp` (new) | 364 |
| `api_python/qpy_ssf_init_phases.cpp` (new) | 94 |
| `api_python/qpy_ssf_draw.cpp` (new) | 144 |
| `api_python/qpy_ssf_clusters.cpp` (new) | 347 |
| `api_python/python_gbsm.cpp` (3 includes, 3 `m.def`) | +35 |
| `tests/mex_tests/test_ssf_{init_phases,draw,clusters}.m` (new) | 93 / 128 / 201 |
| `tests/python_tests/test_ssf_{init_phases,draw,clusters}.py` (new) | 104 / 164 / 226 |
| `SECTION_DESC`: `src/qd_scenpar.cpp`, `api_mex/conf_file_read.cpp`, `api_python/qpy_conf_file_read.cpp` | +13/−3, +14/−4, +13/−4 |
| See also: `src/qd_gbsm_phases.cpp` (3), `src/lsp_generate.cpp`, `api_mex/lsf_init_phases.cpp`, `api_mex/lsp_generate.cpp`, `api_python/qpy_lsf_init_phases.cpp`, `api_python/qpy_lsp_generate.cpp` | +1 each |
| `html_docu/{cpp,mex,python}_api.html`, `html_docu/quadriga_lib_api_{cpp,mex,python}.md` (regenerated) | +2601/−320 |

Size: the wrappers have about 250 lines of code and 950 lines of documentation, and the tests have 916 lines. The
spec estimated about 600 and 400. The documentation is long because the MEX and Python blocks carry the same facts
as the C++ `ssf_clusters` block (spec 7). The tests follow the SPEC_MATLAB_API and SPEC_PYTHON_API checklists.

## Decisions and deviations
- **MEX `set_id`.** Read as a double. A negative, non-integer or NaN value raises `Input 'set_id' must be 0 (shared
  set) or a frequency index >= 1.` The wrapper then passes `set_id − 1` to C++ (spec 6.1). The check is three lines
  in each of the two wrappers that use it; no shared helper.
- **MEX empty sets.** `ssf_init_phases` and `ssf_draw` copy the output after the call with `qd_mex_copy2matlab`
  (spec 6). With `N = 0`, MATLAB receives an empty array, not `[n_coeff, 0, 2]` or `[n_link, 0]`. The MEX
  documentation says "empty", and the tests check `isempty`. Python keeps the shapes (`(256, 0, 2)` and `(5, 0)`
  are tested).
- **MEX defaults.** An empty optional array (`[]`) selects the default, the usual MEX pattern. In Python only `None`
  selects the default. An empty array is passed to C++ as given, so that zero-link calls with correctly sized empty
  arrays work (`rnd_freq` of shape `(0, N, 1)`).
- **`n_cluster`.** Read as double. The wrapper rejects non-integer and negative values with `Entries in 'n_cluster'
  must be non-negative integers.`; the range check stays in C++.
- **`o2i`.** MEX uses `qd_mex_matlab2vector_Bool`. Python reads a double array, and nonzero means true.
- **Python `gr_eps`.** Cast to `complex128` with `py::cast<py::array_t<std::complex<double>>>` and split by
  `qd_python_copy2arma` into real and imaginary cubes, then viewed as `(n_link, n_freq)` matrices. A real array is
  accepted with imaginary part 0.
- **Python outputs.** Always a 7-tuple of copies. `n_active` is uint64 `(n_link,)`.
- **Python flags.** `dual_mobility` and `horizontal` are read with `qd_python_flag`: `None` gives the default and a
  non-bool raises `TypeError`. The rule is stated in one bullet of the `ssf_draw` block.
- **Link in the wrapper docs.** The C++ block links `[[calc_angular_spreads_sphere]]`. The MEX and Python blocks link
  `[[calc_angular_spread]]`, which is the wrapper of that function. MEX method names are written `'38.901'`.
- **See also.** Changed in 8 places instead of the 7 the spec names. The spec's 7: `lsf_init_phases` →
  `ssf_init_phases` in C++, MEX and Python; `lsp_generate` → `ssf_clusters` in all three; `ssf_init_phases` →
  `ssf_clusters`. The 8th: `ssf_draw` → `ssf_clusters` in C++, so that the three new functions name each other in
  every API. `ssf_clusters` names `lsp_generate`, `lsf_draw` and the spread functions; the other two new functions
  do not (they do not produce clusters).
- **`make python`.** Run as its two cmake commands, with `--parallel 12` instead of the target's unbounded
  `--parallel`. The options, compiler (gcc-12) and build directory (`build_linux`) are the same as the target's.
- **MEX build.** Scratch build directory `/home/stephan/quadriga-lib-agents/scratch/ssf_r3/build`: g++-12, MATLAB +
  Octave + MEX documentation on, AVX2 off (as `make`). MOxUnit was unpacked into the scratch directory.

## Gate results
| Gate | Result | Load (1-min) |
|---|---|---|
| MEX build (g++-12) | 0 warnings | 1.4 |
| objdump `GLIBCXX` maximum, all `.mexa64` | `GLIBCXX_3.4.26` (the three `ssf_*` files: 3.4.26) | – |
| Octave, the three new tests | 3/3 PASS | 2.1 |
| Octave, full MEX suite | 76/77; fails only `test_ray_mesh_interact` line 113 | 8.6 |
| MATLAB R2024a, full suite | 76/77, same single failure | 6.3 |
| MATLAB R2026a, full suite | 76/77, same single failure | 4.6 |
| `make python` build | 0 warnings | 2.1 / 3.7 |
| pytest `tests/python_tests` | 900 passed, 3 skipped, 52 subtests passed | 3.7 |
| pytest `test_ssf_*.py` | 6 + 9 + 9 passed | 5.1 |
| Catch2 (`build_ssf`, 12 threads) | All tests passed (1384974 assertions in 1449 test cases) | 3.9 |
| `make documentation` | rc 0; no unresolved `[[...]]` link to the new pages (the existing `.calc_diffraction_gain` link of the MEX page is unresolved, as before) | – |

MATLAB help: `tools/extract_matlab_comments.py` converts the three new blocks (82, 98 and 295 lines).

## Release note
- New functions in the geometry-based stochastic model (C++, MATLAB/Octave, Python):
  - `ssf_init_phases`: random phases of the small-scale fading sets. There is one shared cluster set, and one set
    per frequency. Their seeds are derived from the seed of the large-scale set.
  - `ssf_draw`: spatially consistent cluster random values at the links (delays, signs, angle offsets, cluster
    shadowing), with the same floors and mobility rules as `lsf_draw`.
  - `ssf_clusters`: cluster delays, powers and angles of TR 38.901 clause 7.5 steps 5 to 7, with ground reflection,
    absolute time of arrival and per-link cluster numbers. Three methods: `38.901` (clause 7.5, one frequency, the
    default), `38.901_MF` (clause 7.6.5.1) and `quadriga` (spatially consistent, the same delays and angles at all
    frequencies).
- The six TR 38.901 v19 LOS scenario files (UMi, UMa, RMa, SMa, InF, Indoor) count one more cluster.
  `NumClusters` and `NumClusters_min` now include one slot for the LOS path, as in the NLOS and O2I files, so the
  number of NLOS clusters is `NumClusters − 1` in every file. Code that read the LOS cluster number from these
  files gets one more.

## Risks
- `test_ray_mesh_interact` (MEX, line 113, transmission gain 0.985 against 0.696) fails in Octave, R2024a and
  R2026a. This run does not change that code. The test and `src/ray_mesh_interact.cpp` are unchanged since base
  26da9f93, and no commit on `main` after the base touches them. Not checked: whether the failure also occurs on
  `main`.
- The Python wrappers copy every output once (sizes depend on `scenpar` and `method`). This is not significant for
  cluster-level sizes.
- MSVC is not compiled here. The wrappers use only `std::floor`, `std::complex` and the existing helpers.

## Suggested commit message
```
Small-scale fading: MATLAB/Octave and Python interfaces

ssf_init_phases, ssf_draw and ssf_clusters are available in MATLAB/Octave
(quadriga_lib.*) and in Python (quadriga_lib.gbsm.*). In MATLAB the set index is
0 for the shared cluster set and k >= 1 for the k-th frequency. In Python the
ground permittivity is one complex array, as lsp_generate returns it. The
section overview now covers TR 38.901 clause 7.5 steps 1 to 7, the three new
functions are cross-linked with the large-scale functions, and the HTML and
Markdown API documentation is regenerated.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```
