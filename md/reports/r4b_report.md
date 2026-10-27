# Run 4b report — Python wrappers, release notes

## Summary

- Six Python wrappers in the submodule `quadriga_lib.gbsm` as in spec 6.2 (`lsf_init_phases`, `lsf_draw`, `path_loss`,
  `lsp_generate`, `los_probability`, `o2i_loss`), registered in `api_python/python_gbsm.cpp`, each with its pytest
  file (SPEC_PYTHON_API order: wrapper, then test); `SECTION_DESC` (Python) in `api_python/qpy_conf_file_read.cpp`.
- User decisions of Run 4a applied: spec 6.1 edited (MEX floor-ID types, zero links), the C++ `path_loss` block
  reworded without `[X|v]`, and the rule on matrix products above 4x4 added to the tests section of `SPEC_CPP_API.md`.
- Zero links work in Python (the helpers keep the `(0, 3)` shape); tested for all six wrappers.
- Gates: Python suite, Catch2 full suite, MEX suite in Octave 11.1, R2024a, R2026a, objdump gate; no warning in any
  build. Only failure: the known `test_ray_mesh_interact` MEX case (since before Run 1). No spec problem found.

## Files

Created (12):
- `api_python/qpy_lsf_init_phases.cpp`, `qpy_lsf_draw.cpp`, `qpy_path_loss.cpp`, `qpy_lsp_generate.cpp`,
  `qpy_los_probability.cpp`, `qpy_o2i_loss.cpp`
- `tests/python_tests/test_lsf_init_phases.py`, `test_lsf_draw.py`, `test_path_loss.py`, `test_lsp_generate.py`,
  `test_los_probability.py`, `test_o2i_loss.py`

Changed (5):
- `api_python/python_gbsm.cpp`: six includes and registrations, `#include <limits>` (NaN defaults), the
  `.none(false)` comment as in `python_sos.cpp`
- `api_python/qpy_conf_file_read.cpp`: `SECTION_DESC` after the `SECTION` block
- `src/path_loss.cpp`: documentation block only (`[X|v]` -> `[X]` "or its default if the key is missing (`X = v`
  in the lists below)"; `[LOSmax]`, `[hE_rand]`, `[hE]`), the MEX wording
- `md/gbsm_lsf_spec.md`: 6.1 only (user decisions 1 and 2, see Decisions 13)
- `md/SPEC_CPP_API.md`: tests section, "Fixtures and oracles", one rule on BLAS-backed Armadillo calls

Plus this report. Not from this run: `md/prompts/impl_run4b.md`, `md/prompts/code_review_r1.md` (untracked). Build
directories `build_py/`, `build_mex/` (gitignored, see Test results).

## Decisions and deviations

Review points of SPEC_PYTHON_API (not stops): A — the interfaces are fixed by spec 6.2; the choices below are within
it. B — this report.

1. Usage lines and registration exactly as spec 6.2; kwarg names = wrapper parameter names = Usage names = comment
   block at the end of each wrapper file (copied verbatim into the dispatcher by script). `scenpar` and `plpar` are
   mandatory (no default) `py::handle` arguments that accept `None` (= defaults / no model). Mandatory arrays use
   `.none(false)`. `models` of `o2i_loss` is a plain `py::handle` (str, list or tuple of str); `None` gives an empty
   list and the C++ error `Input 'models' cannot be empty.` (tested).
2. Helpers: `qd_python_dict2scenpar` / `qd_python_dict2plpar` (`SubpathMethod` read and discarded, `model` -> `pl_model`),
   so the dicts of `conf_file_read` and `scenpar_validate` feed the wrappers directly (tested with v19 files).
   `los_probability` and `o2i_loss` include `python_arma_adapter.hpp` + `quadriga_lib.hpp` (no dict helpers needed),
   the four others `python_quadriga_adapter.hpp`.
3. Outputs: `phase` copied after the call (size from the preset); all other outputs pre-allocated with
   `qd_python_init_output` (zero-copy), `model_index` as uint64 (0-based, C++ type kept), `abs_toa` 1D `(n_link,)`.
4. `lsp_generate` `complex` flag, default `True` (spec 6.2; SPEC_PYTHON_API standard default). `complex=True`:
   C++ writes `gr_eps_re` / `gr_eps_im` into Armadillo-owned matrices, merged once with
   `qd_python_copy2numpy<double, std::complex<double>>`; `complex=False`: both pre-allocated zero-copy. All outputs
   are always computed, so column 9 of `rnd` is always checked; the documentation block states this (6.2).
5. `los_probability`: `los_soft` is computed iff `rnd_g` and `fc` are both not `None` (a `None` check, not an
   emptiness check). Reason: spec 6.2 words it with `None`, and an emptiness check would return `None` for zero
   links with both given. Consequence: an empty `fc` with `rnd_g` given reaches C++ and raises `Input 'fc' cannot be
   empty.` (tested); `rnd_g=np.zeros(0)` with `n_link = 0` gives `los_soft` of shape `(0, n_freq)` (tested). The
   MEX wrapper treats empty as omitted (MATLAB has no `None`).
6. Other optional arrays (`tx_pos`, `rx_pos` of `lsf_init_phases`, `floor_id`, `d2d_in`, `rnd_he`, `sf_sigma`):
   `None` or empty = not given (`nullptr`), the SPEC_PYTHON_API `is_empty() ? nullptr : &x` pattern, as the MEX
   wrappers.
7. `floor_id`: read as double with the `py::handle` helper (any numeric dtype or list, so `uint8` is accepted in
   Python, tested), checked in the wrapper (`v >= 0`, `v < 2^64`, integer; NaN and Inf fail) with the spec message
   `Entries in 'floor_id' must be non-negative integers.` (`ValueError`), converted to `arma::uvec`, no index shift.
   The check runs before the C++ validation; the documentation block's validation order says so.
8. Zero links (user decision 2, Python part): the Python helpers keep the shape of `(0, 3)` / `(0, 11)` / `(0, 4)`
   arrays, so zero links are supported and tested in all six wrappers (outputs with 0 rows). The C++ statements
   `n_link = 0` / `n_ut = 0 allowed` are kept in the Python blocks.
9. Flags (`dual_mobility`, `remove_bias`, `horizontal`, `cholesky`, `clip_spreads`, `complex`) are plain `bool`
   as spec 6.2 says (SPEC_PYTHON_API scalars), not `qd_python_flag` as in `conf_file_read`. Open point 2.
10. `seed` is `long long` (spec 6.2), cast to `arma::sword`; `-2` reaches the C++ check (`Input 'seed' must be >= -1.`).
    Clutter arguments are `double` with default `std::numeric_limits<double>::quiet_NaN()`, documented as `nan`.
11. Documentation blocks: built from the C++ blocks (0-based columns, slices and indices; "key" for dict entries),
    shapes as tuples, units without brackets, `[[sos.init_phases]]`, `[[sos.remove_bias]]`, `[[sos.val]]`,
    `[[sos.preset]]`, `[[sos.rand]]`, `[[sos.randn]]` (Python names of the SOS functions), "throws" -> "raises",
    "All checks run before any output is written" -> "All errors are raised as `ValueError`". One-line summaries are
    identical to the C++ / MEX blocks. No `*` or `|` in any Python block (checked by script). `tools/extract_md.py`
    was run for Python and C++ to a scratch file: all six functions and the section description are picked up, links
    resolve; no file in `html_docu/` was written.
12. `SECTION_DESC` (Python): the C++ text with `[[sos.rand]]` / `[[sos.randn]]` and two additions (the functions are
    in `quadriga_lib.gbsm`; `conf_file_read` returns the two dicts `scenpar` and `plpar`); 20 text lines.
13. User decisions of Run 4a, applied as instructed (the run prompt overrides "do not edit the spec" for 6.1 only):
    - Spec 6.1, after the sentence on `nullptr` outputs: a paragraph that the MEX numeric inputs, `floor_id`
      included, accept the types of the MEX helpers (double, single, int32, uint32, int64, uint64), `uint8` is
      rejected; that the MEX wrappers reject zero links (the helpers turn `[0, 3]` into 0x0) and their blocks do not
      state the `n_link = 0` rule; C++ and the Python wrappers support zero links. In the `lsf_draw` test points
      "`uint8` floor IDs accepted" became "`uint32` and `int64` floor IDs accepted" (what the Run 4a MEX test checks).
    - `src/path_loss.cpp` block: `[X|v]` notation reworded as in the MEX block (decision 7 of Run 4a). No `|` is left
      in the block.
    - `md/SPEC_CPP_API.md`, tests section: "The Catch2 test binary links no BLAS (by design, like the library), so
      tests must not use Armadillo matrix products above 4x4 ...; write small loops instead", with the reason (on
      Linux `test_bin` is not compiled with `ARMA_DONT_USE_BLAS`, so Armadillo calls BLAS above 4x4, which does not
      link) and the related calls `arma::norm`, `arma::dot`, `arma::accu(a % b)`.
14. Builds: `make python` configures `build_linux` with tests off, which would have dropped the Catch2 configuration
    of `build_linux`; the Python and MEX configurations were therefore built in their own (gitignored) directories
    `build_py` (the `make python` options, gcc-12, AVX2 off) and `build_mex` (the `make` options: gcc-12, MATLAB and
    Octave on, AVX2 off, Python off), each with `--parallel 12` (the GNUmakefile passes `--parallel` without a
    number), installed with `cmake --install`. `build_linux` keeps its configuration (tests on, AVX2 on) for `make test`.

## Test results

Load average (1-min) before the heavy runs: 2.1 before the Python build, 2.1 before the Python suite, 2.0 before the
Catch2 build, 2.1 before the Catch2 run, 2.9 before the MEX build, 11.4 before Octave, 10.1 before R2024a, 8.7 before
R2026a (other sessions). No timing results are reported. Builds and suites one at a time, `OMP_NUM_THREADS=12`.

New Python tests one by one first (all passed at the first run): `test_lsf_init_phases.py` 9 passed,
`test_lsf_draw.py` 10, `test_path_loss.py` 10, `test_lsp_generate.py` 11, `test_los_probability.py` +
`test_o2i_loss.py` 17 (57 new test methods).

Python, full suite (`python3 -m pytest tests/python_tests -x -q`, module from `build_py`):
```
873 passed, 3 skipped, 52 subtests passed in 2.72s
```
(816 before this run + 57 new.)

Catch2 (`build_linux`, the `make test` configuration of the tree: tests on, AVX2 on; `OMP_NUM_THREADS=12
build_linux/test_bin`):
```
All tests passed (1355498 assertions in 1374 test cases)
```

MEX (`build_mex`, 83 MATLAB and 83 Octave MEX files; `tests/quadriga_lib_mex_tests.m`, 74 tests):
- Octave 11.1.0 (`octave --no-gui --eval "cd tests; quadriga_lib_mex_tests;"`): `FAILED (passed=73, failure=1)`
- MATLAB R2024a (`/usr/local/MATLAB/R2024a/bin/matlab -batch "run('tests/quadriga_lib_mex_tests.m');"`):
  `FAILED (passed=73, failure=1)`
- MATLAB R2026a (`/usr/local/MATLAB/R2026a/bin/matlab -batch ...`): `FAILED (passed=73, failure=1)`
- The one failure in all three is `test_ray_mesh_interact.m` line 113 (0.98485 vs. 0.69639), the pre-existing
  failure reported since Run 1. All GBSM, conf and `scenpar_validate` tests pass.

objdump gate (`objdump -T +quadriga_lib/*.mexa64 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -u`, 83 files):
3.4.9, 3.4.11, 3.4.18, 3.4.19, 3.4.20, 3.4.21, 3.4.26 — nothing above 3.4.28.

## Warnings

None. `grep -ci warning` on the logs of the Python build (module compiled with `-Wall -Wextra -Wpedantic`), the
Catch2 build and the MEX build: 0 each. (`test_bin` itself is compiled with `-w` by `CMakeLists.txt`.)

## Open points

1. No spec problem found in Run 4b.
2. Plain `bool` flags (decision 9): pybind11's bool caster converts `None` to `False` and accepts `int` in convert
   mode, so `remove_bias=None` gives `False`, not the default `True` (checked). `conf_file_read` /
   `conf_file_write` use `qd_python_flag` (None = default, int rejected). Proposal for a later pass: use
   `qd_python_flag` for the GBSM flags too (a change of spec 6.2), or document that `None` is not a default.
3. MEX vs Python difference for `los_probability` (decision 5): an empty `rnd_g` or `fc` gives `[]` in MATLAB but
   reaches C++ in Python. Intended by the wording of 6.1 / 6.2; stated in both blocks.
4. Not tested: `qd_python_dict2scenpar` errors beyond `Input 'scenpar' must be a dict or None.` (covered by the conf
   tests); the uniform-range check of `o2i_loss` (Catch2 and MEX cover it).
5. The pre-existing MEX failure of `test_ray_mesh_interact` persists.

## Release notes (text proposal; `README.md` has no release-notes section)

```
Large-scale model of 3GPP TR 38.901 (geometry-based stochastic model)

New functions (C++, MATLAB/Octave, Python submodule quadriga_lib.gbsm), section "Geometry-based stochastic model":
- lsf_init_phases, lsf_draw: spatially consistent large-scale fading generators of a BS site (11 generators:
  8 LSPs, absolute TOA, ground permittivity, UMa environment height), 2D correlation by default (3D as option),
  floor separation, dual mobility, bias removal over the drop.
- path_loss: the 10 parametric path-loss models of the QuaDRiGa conf files and the shadow-fading standard
  deviation per link and frequency, incl. the random UMa environment height and the legacy outdoor distance.
- lsp_generate: DS, KF, SF, ASD, ASA, ESD, ESA, XPR per link and frequency (TR 38.901 clause 7.5 step 4,
  Cholesky in TR order or the symmetric root, spread clipping), absolute TOA, ground permittivity, EoD offset,
  log-domain mean and standard deviation.
- los_probability: TR 38.901 Table 7.4.2-1 incl. InF and SMa (v19), TR 38.811 tables, soft LOS state (7.6-18).
- o2i_loss: O2I penetration loss of TR 38.901 clause 7.4.3 (low, high, low_A, legacy_6GHz, car, car_metal) and
  the indoor distance.

Interface changes:
- conf_file_read, scenpar_validate, conf_file_list and conf_file_write moved from quadriga_tools.hpp to the new
  header quadriga_gbsm.hpp (included by quadriga_lib.hpp, so code that includes quadriga_lib.hpp is not affected).
- Python (breaking): the four conf functions moved from quadriga_lib.tools to quadriga_lib.gbsm.
- New scenario keys: LOS_state_lambda, IO_state_lambda, O2I_d2D_in_max, DS_VS, DS_VS_a, DS_VS_b (171 numeric
  keys); new path-loss keys PL_hE_rand, PL_LOSmax, PL_Ax, PL_Bx, PL_Cx. Defaults keep the previous behaviour.

Changed conf files (data/config, TR 38.901 v19):
- Path-loss constants rounded to 4 significant digits; the path loss changes by up to 0.014 dB (RMa), 0.006 dB
  (UMa, UMi), 0.005 dB (SMa). UMa uses the random environment height (PL_hE_rand = 1); SMa NLOS without the LOS
  term (PL_LOSmax = 0); InF-DL with the InF-SL term (PL_Ax, PL_Bx, PL_Cx).
- The InF and InH NLOS files bound the absolute TOA by 2·L/c (absTOA_Lmax: 120 m or 300 m for the InF calibration
  halls, 120 m for the InH office).
- InF delay spread with the V/S term of Table 7.5-6 Part-3 (DS_VS, DS_VS_a, DS_VS_b).
- LOS-state and indoor-state decorrelation distances and the maximum indoor distance in the NLOS and O2I files.

Differences to QuaDRiGa's large-scale model:
- O2I loss: only the models of TR 38.901 clause 7.4.3 and the TR indoor distance (minimum of two uniforms); the
  mmMAGIC model, the TR v14 wall values and QuaDRiGa's indoor-distance distribution are not provided.
- Bias removal: each Normal field is corrected once over all positions where it is evaluated (Rx and, in dual
  mobility, Tx positions), and the Tx side is mirrored from the corrected phases.
- Dual mobility: ASD/ASA and ESD/ESA keep unit variance and are correlated only by the ACF at the Tx-Rx distance;
  QuaDRiGa's extra 2x2 coupling is not applied.
- An LSP standard deviation that the distance, height or elevation terms make negative is limited at 0.
- winner_los uses C2 beyond the breakpoint and computes the breakpoint from the heights of each link.
```

## Suggested commit cut

Commit 1 — the Python wrappers:
- Files: `api_python/qpy_lsf_init_phases.cpp`, `api_python/qpy_lsf_draw.cpp`, `api_python/qpy_path_loss.cpp`,
  `api_python/qpy_lsp_generate.cpp`, `api_python/qpy_los_probability.cpp`, `api_python/qpy_o2i_loss.cpp`,
  `api_python/python_gbsm.cpp`, `api_python/qpy_conf_file_read.cpp`, `tests/python_tests/test_lsf_init_phases.py`,
  `tests/python_tests/test_lsf_draw.py`, `tests/python_tests/test_path_loss.py`,
  `tests/python_tests/test_lsp_generate.py`, `tests/python_tests/test_los_probability.py`,
  `tests/python_tests/test_o2i_loss.py` (with the maintainer's usual run files `md/prompts/impl_run4b.md`,
  `md/reports/r4b_report.md`)
- Message:
  ```
  Python wrappers of the large-scale model

  Adds lsf_init_phases, lsf_draw, path_loss, lsp_generate, los_probability and o2i_loss to the submodule
  quadriga_lib.gbsm, with their pytest files. Scenario and path-loss parameters are the dicts returned by
  conf_file_read or scenpar_validate; None gives the defaults. Indices stay 0-based, the model index of
  o2i_loss is uint64. lsp_generate returns the ground permittivity as one complex array by default
  (complex=False: real and imaginary part). los_probability returns the soft LOS state only when rnd_g and fc
  are given. Zero links give outputs with 0 rows.

  The section "Geometry-based stochastic model" gets its overview for Python.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```

Commit 2 — documentation and coding rules:
- Files: `src/path_loss.cpp`, `md/SPEC_CPP_API.md`, `md/gbsm_lsf_spec.md`
- Message:
  ```
  path_loss documentation without the [X|v] notation; test rule on matrix products

  The documentation of path_loss writes [X] for a parameter or its default, as the MATLAB documentation does;
  the bar is reserved for Markdown tables. The C++ coding rules state that the test binary links no BLAS, so
  tests use loops instead of Armadillo matrix products above 4x4. The wrapper specification records the
  accepted MATLAB input types of floor IDs and that the MATLAB wrappers do not accept zero links.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```
