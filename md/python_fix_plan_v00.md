# Python API fix plan v00

Issues found in `api_python/` while comparing all `qpy_*.cpp` wrappers, the dispatcher files
(`python_main.cpp`, `python_arrayant.cpp`, `python_channel.cpp`, `python_tools.cpp`,
`python_RTtools.cpp`) and both adapter headers against `md/SPEC_PYTHON_API.md`.

- Date of the review: 2026-10-05
- All findings come from reading the source. Nothing was compiled, run or benchmarked.
  Items marked **(verify)** are conclusions that should be confirmed with a test before fixing.
- Line numbers refer to the state of the repository at the review date.
- Each item is independent unless stated otherwise. Items in sections 1 and 2 change behavior or fix
  bugs; sections 3 to 6 are consistency work against the revised spec; section 7 is informational.
- Any change of a kwarg name, default or return type is an API change for Python users. Decide per item
  whether the code or the documentation is the side to fix.

## 1. Functional defects

### 1.1 `view = true` copies instead of aliasing (verify)
- File: `api_python/python_arma_adapter.hpp`, lines 1055, 1070, 1085 (`qd_python_numpy2arma_Col/Mat/Cube`)
  and 1182 (`qd_python_numpy2arma_vecCube`)
- The "view" path builds the Armadillo object as `arma::Col<dtype>(ptr, n, true, false)`. In Armadillo
  the third argument is `copy_aux_mem`; `true` copies the data. Other places in the same header that
  really alias use `(ptr, n, false, true)` (e.g. line 1531).
- Effect if confirmed: array inputs are never zero-copy; `view = true` only selects a faster contiguous
  copy. Comments in the header ("direct memory views") and the spec describe it as zero-copy.
- Consequences to check before changing it:
  - Wrappers that modify an input read with `view = true` would start mutating the user's numpy array.
    The spec requires `view = false` for such inputs; audit all wrappers for writes into `*_a` objects.
  - The returned objects are non-const; aliasing read-only numpy arrays needs care.
  - Lifetime: the numpy array must outlive the Armadillo object (true inside a wrapper call).
- Suggested steps: (1) confirm with a small test (modify the Armadillo object, check the numpy array) or
  a benchmark with a large input, (2) decide whether to switch to real aliasing or to rename / document
  the flag as "fast copy".

### 1.2 `hdf5_write_dset` bounds check is off by one
- File: `api_python/qpy_hdf5_write_dset.cpp`, line 65
- `if (ix > storage_space.at(0) || iy > ... )` — indices are 0-based, so `ix == nx` passes the wrapper
  check. Should be `>=`.
- The C++ function may catch it anyway; the wrapper check is then misleading rather than unsafe. **(verify)**

### 1.3 `hdf5_create_file` return type undoes the integer cast (verify)
- File: `api_python/qpy_hdf5_create_file.cpp`, lines 42 and 54
- Declared return type is `py::array_t<unsigned>`, but the function returns
  `qd_python_copy2numpy<unsigned, py::ssize_t>(...)`. The implicit conversion to the declared type
  converts the array back to uint32 (one extra copy, and the intended int64 result is lost).
- Fix: make the declared return type match what is produced. See also item 4.1 for which type is wanted.

### 1.4 `qd_python_dict2arrayant(dict, true)` does not validate
- Files: `qpy_arrayant_generate.cpp` lines 154, 179; `qpy_get_channels_planar.cpp` line 97;
  `qpy_get_channels_spherical.cpp` line 162; `qpy_get_channels_ieee_indoor.cpp` lines 118, 119;
  `qpy_get_channels_multifreq.cpp` line 89 (`_multi`)
- The helper signature is `(dict, view, strict, validate)`. A single `true` sets `view`, not `validate`.
- In `qpy_arrayant_generate.cpp` the MEX counterpart validates the custom `pattern`
  (`qd_mex_struct2arrayant(prhs[15], true)` where `true` = validate). The Python call looks like a
  port that kept the argument but lost the meaning. **(verify intent)**
- For the `get_channels_*` wrappers the C++ functions may validate themselves; check, then decide
  whether `validate = true` is needed.

### 1.5 `sub_cloud_ind` / `sub_mesh_index` passed as pointer to an empty object (verify)
- Files: `qpy_point_cloud_aabb.cpp` line 48, `qpy_triangle_mesh_aabb.cpp` line 46,
  `qpy_ray_point_intersect.cpp` line 85
- These pass `&index_arma` even when the user gave no index, while `qpy_ray_triangle_intersect.cpp`
  and the MEX wrappers pass `nullptr` for an empty index.
- Check whether the C++ functions treat an empty vector the same as `nullptr`. If not, convert to the
  `x.empty() ? nullptr : &x` pattern.

### 1.6 `version()` does not report CUDA
- File: `api_python/qpy_version.cpp`
- The MEX `version` appends `_AVX2` and `_CUDA`; the Python one appends only `_AVX2`.
- Decide whether Python should match.

### 1.7 `cart2geo` / `geo2cart`: `use_kernel = 2` does not throw in a build without AVX2
- Files: `api_python/qpy_cart2geo.cpp`, `api_python/qpy_geo2cart.cpp` (documentation of `use_kernel`);
  cause in `src/fastmath_armadillo.cpp`, `fast_geo2cart` (lines 405, 406) and `fast_cart2geo`
  (lines 524, 525)
- Found on 2026-10-12 by running the wrappers, not by reading: in a module built with `make python`
  (`ENABLE_AVX2=OFF`), `quadriga_lib.tools.cart2geo( np.ones((3,2,2)), use_kernel=2 )` and the same
  call of `geo2cart` return a result. The documentation of both wrappers and of the C++ functions says
  "2 = AVX2 (throws if unavailable)".
- Cause: without `BUILD_WITH_AVX2` both C++ functions start with `(void)use_kernel;` and always run
  the GENERIC kernel. The argument is not looked at, so an invalid value such as 3 is not rejected
  either **(verify)**. With `BUILD_WITH_AVX2` on a CPU without AVX2 the throw is in place.
- The defect is in the C++ library; the wrappers pass the value through unchanged, and the MEX wrappers
  of the same functions are affected in the same way **(verify)**.
- For comparison, `sos_val` and `sos_map` throw `std::invalid_argument` ("AVX2 kernel requested but not
  available ...") in the same build.
- Decide which side to fix: make the C++ functions throw in a build without AVX2 (matches the
  documentation, but callers that pass 2 unconditionally start to fail), or change the documentation
  to "falls back to GENERIC".
- No test covers it: Python has no function to query AVX2 support, so a test has to accept either the
  result or the exception, as `tests/python_tests/test_sos_val.py` does.

## 2. Documentation that does not match the code (users will get errors)

### 2.1 `qrt_file_read`: kwarg names differ between docs and registration
- File: `api_python/qpy_qrt_file_read.cpp` lines 31, 36, 37 (docs: `cir`, `orig`), lines 90, 91
  (C++ parameters: `cir`, `orig`), lines 234, 235 and `python_channel.cpp` (registered kwargs:
  `i_cir`, `i_orig`)
- A user following the docs and calling `qrt_file_read(fn, cir=...)` gets a `TypeError`.
- Decide which name is the API (the registered `i_cir` / `i_orig` is what currently works), then align
  docs, C++ parameter names and the comment block.

### 2.2 `icosphere`: usage line uses `no_div`
- File: `api_python/qpy_icosphere.cpp` lines 23, 26
- The kwarg is `n_div` (line 45, and in `python_RTtools.cpp`). Fix the usage lines.

### 2.3 `qdant_write`: malformed usage line
- File: `api_python/qpy_arrayant_qdant_write.cpp` line 30
- `arrayant.quadriga_lib.qdant_write( fn, arrayant )` should be
  `quadriga_lib.arrayant.qdant_write( fn, arrayant )`; the frequency-dependent variant returns 0, so the
  line should also show the return value or state that it is 0.

### 2.4 `ray_triangle_intersect`: usage lines lack the `quadriga_lib.` prefix
- File: `api_python/qpy_ray_triangle_intersect.cpp` lines 23, 26

### 2.5 `calc_diffraction_gain`: parameter name differs from kwarg
- File: `api_python/qpy_calc_diffraction_gain.cpp` line 88
- The C++ parameter is `center_freq`, the registered kwarg and the docs say `center_frequency`.
  Works for the user; rename the C++ parameter so that the three names agree (spec rule).

### 2.6 `arrayant_interpolate`: comment block differs from the dispatcher
- Files: `api_python/qpy_arrayant_interpolate.cpp` lines 226, 227 (comment: `azimuth`, `elevation`
  mandatory) vs. `api_python/python_arrayant.cpp` lines 105, 106 (`= py::none()`)
- The wrapper throws on empty azimuth, so mandatory is the consistent choice. Decide and align.

### 2.7 `ray_point_intersect`: usage continuation line lacks the trailing ` \`
- File: `api_python/qpy_ray_point_intersect.cpp` lines 27, 28 (cosmetic)

## 3. Registration comment blocks

The revised spec requires an identical copy of the dispatcher `m.def(...)` entry as a comment block at
the end of each wrapper file.

### 3.1 Files without a `// pybind11 declaration:` comment block (18)
`qpy_arrayant_copy_element.cpp`, `qpy_channel_export_obj_file.cpp`, `qpy_components.cpp`,
`qpy_get_channels_ieee_indoor.cpp`, `qpy_hdf5_read_channel.cpp` (has the `m.def` lines but no heading
line), `qpy_hdf5_read_dset_names.cpp`, `qpy_hdf5_reshape_layout.cpp`, `qpy_hdf5_write_dset.cpp`,
`qpy_icosphere.cpp`, `qpy_mitsuba_xml_file_write.cpp`, `qpy_point_cloud_aabb.cpp`,
`qpy_point_cloud_segmentation.cpp`, `qpy_point_inside_mesh.cpp` (same as `hdf5_read_channel`),
`qpy_ray_point_intersect.cpp`, `qpy_ray_triangle_intersect.cpp`, `qpy_triangle_mesh_aabb.cpp`,
`qpy_version.cpp`, `qpy_write_png.cpp`

### 3.2 Check all existing comment blocks against the dispatchers
- Known drift: items 2.1 and 2.6. A full diff of every comment block against its `m.def` entry has not
  been done; do it as one pass (script or by hand) when working on 3.1.
- Mandatory inputs registered with a default (dispatcher), e.g. `py::arg("orig") = py::array_t<double>()`
  in `ray_triangle_intersect`, `ray_point_intersect`, `triangle_mesh_aabb`, `point_cloud_aabb`,
  `point_cloud_segmentation`, `py::arg("fn") = ""` in `obj_file_read` / `obj_file_write`,
  `py::arg("arrayant") ...` etc. Under the spec, mandatory parameters have no default. Changing this is
  an API change only for callers that relied on omitting them.

## 4. Conformance with the revised spec (existing wrappers)

None of these are bugs. They are places where existing wrappers follow older conventions. Decide per
group whether to migrate or to leave as legacy.

### 4.1 Integer output types
Spec rule: pre-allocated outputs keep the C++ integer type; copied outputs are cast to signed 64-bit;
the declared return type must match.
- Copied without the cast: `qpy_point_cloud_segmentation.cpp` (`sub_cloud_ind`, `forward_ind`,
  `reverse_ind`, uint32), `qpy_qrt_file_parse.cpp` (`cir_offset`, uint64), `qpy_qrt_file_read.cpp`
  (`no_int`, uint32; `interact_type`, uint8), `qpy_hdf5_read_layout.cpp` line 51 (`storage_dims`,
  uint32), `qpy_hdf5_write_channel.cpp` line 169 (uint32, declared `py::array_t<unsigned>`)
- Inconsistent within one function: `qpy_ray_point_intersect.cpp` returns `hit_count` / `hit_offset`
  as uint32 (pre-allocated, conforms) and `hit_index` as int64 (copied, conforms) — fine under the
  rule, but document the types in the Outputs section.
- `qpy_hdf5_create_file.cpp`: see item 1.3.
- Changing a numpy dtype is visible to users; check the tests in `tests/python_tests/` for dtype
  assertions before changing.

### 4.2 Array input style
Spec rule: `const py::array_t<dtype> &` for mandatory arrays, `py::handle` with `py::none()` for
optional ones.
- Optional arrays as `py::array_t<...>` with an empty-array default: `qpy_ray_triangle_intersect.cpp`
  (`sub_mesh_index`, `aabb`), `qpy_ray_point_intersect.cpp` (`sub_cloud_ind`), `qpy_point_cloud_aabb.cpp`
  (`sub_cloud_ind`), `qpy_triangle_mesh_aabb.cpp` (`sub_mesh_index`), `qpy_get_channels_ieee_indoor.cpp`
  (`Dist_m`, `n_floors`, `offset_angles`, `n_walls`), `qpy_mitsuba_xml_file_write.cpp` (all arrays),
  `qpy_write_png.cpp` (`data`)
- Inputs as `py::object`: `qpy_baseband_freq_response.cpp`, `qpy_quantize_delays.cpp`
- Mandatory arrays as `py::handle`: `qpy_get_channels_spherical.cpp` (needed for its batch mode),
  `qpy_arrayant_interpolate.cpp` (`azimuth`, `elevation`), `qpy_arrayant_copy_element.cpp`,
  `qpy_calc_diffraction_gain.cpp` (`mtl_ind`, `mtl_prop`), `qpy_channel_export_obj_file.cpp`
- `qpy_channel_export_obj_file.cpp` lines 106, 107, 110 take `py::handle &` (non-const reference);
  should be by value or `const &`.
- `qpy_hdf5_write_dset.cpp` lines 45, 46 and `qpy_hdf5_reshape_layout.cpp` have defaults in the C++
  signature and take `std::string` by value; spec: defaults only in the registration, strings as
  `const std::string &`.

### 4.3 `complex` and `stack` flags
Spec rule: `complex` defaults to `true`, `stack` to `false`, both are the last arguments in this order.
- `complex` default is `false` in: `arrayant.interpolate`, `arrayant.get_channels_planar`,
  `arrayant.get_channels_spherical`, `arrayant.get_channels_multifreq`, `channel.quantize_delays`
- Argument order differs in `channel.quantize_delays` (`stack`, `complex`, then `coeff`).
- `channel.baseband_freq_response` always returns complex and has no flag.
- Changing a default changes the returned tuple for every existing caller. Recommendation: leave the
  existing wrappers as they are unless a breaking release is planned; the spec already says so.

### 4.4 Other flags that change the returned tuple
- `angles` (`get_channels_spherical`), `dist`, `local_angles` (`interpolate`), `combine`
  (`cart2geo`, `geo2cart`). Not standard under the spec. Keep as legacy or redesign; no action needed
  unless wanted.

### 4.5 Naming
- Wrapper function name, Python name and file name differ for the arrayant group
  (`arrayant_<name>` exposed as `arrayant.<name>`), for `generate_speaker`
  (C++ function `arrayant_generate_speaker`, file `qpy_generate_speaker.cpp`) and for
  `get_channels_ieee_indoor` (exposed as `channel.get_ieee_indoor`).
- `get_channels_planar`, `get_channels_spherical`, `get_channels_multifreq` are registered in the
  `arrayant` submodule, `get_ieee_indoor` in `channel`; all four have the section
  "Channel generation functions".
- Decided case by case under the spec; listed here only so that the choice is made once and recorded.

### 4.6 Wrapper-side optimizations
- `qpy_qrt_file_read.cpp` (reader thread, double buffer, GIL release), `qpy_get_channels_spherical.cpp`
  (batch mode, OpenMP), `qpy_baseband_freq_response.cpp` (635 lines, manual list parsing, OpenMP).
- Allowed as requested optimizations. Worth a check that each releases / reacquires the GIL correctly
  and touches no Python object without it; `qpy_qrt_file_read.cpp` documents this, the other two were
  not reviewed in that depth.

## 5. Documentation block format

### 5.1 Headings that differ from the spec
- `qpy_write_png.cpp` lines 29 to 41: old layout (`## Description:`, `## Declaration:`,
  `## Arguments:`, no `## Usage:`). Rewrite the block.
- `## Input Arguments:` / `## Output Arguments:` / `## Output Argument:` in
  `qpy_mitsuba_xml_file_write.cpp`, `qpy_point_inside_mesh.cpp`, `qpy_hdf5_write_dset.cpp`,
  `qpy_point_cloud_aabb.cpp`
- Singular / plural: apply "singular only for exactly one argument" in one pass over all files
  (e.g. `qpy_hdf5_read_layout.cpp` has one input under `## Inputs:`).
- `qpy_version.cpp` line 22: `## Caveat:` is a non-standard heading; fold it into the description.

### 5.2 Usage blocks
- "Output as tuple / Unpacked outputs" double usage lines in `qpy_icosphere.cpp`,
  `qpy_calc_diffraction_gain.cpp`, `qpy_point_cloud_segmentation.cpp`,
  `qpy_ray_triangle_intersect.cpp`, `qpy_triangle_mesh_segmentation.cpp`. The spec shows one usage line
  per variant; the tuple form is redundant. Remove or keep, but consistently.
- `qpy_version.cpp`: usage line ends with `;`.

### 5.3 Line length
- 109 lines inside `/*!MD` blocks are longer than 120 characters (soft limit). Reflow when touching a
  file; no dedicated pass needed.

### 5.4 Output types
- Integer and complex outputs should state their type in the Outputs section (spec). Check together
  with item 4.1.

## 6. File headers
- `qpy_components.cpp` and `qpy_write_png.cpp` carry the old long Apache header instead of the 3-line
  SPDX header.

## 7. Coverage (informational)

### 7.1 MEX functions without a Python wrapper
`calc_rotation_matrix`, `fast_sincos`, `generate_diffraction_paths`, `get_channels_irs`,
`hdf5_version` (available through `components()`), `interp`, `ray_mesh_interact`,
`subdivide_triangles`

### 7.2 Python functions without a MEX wrapper
`components`, `mitsuba_xml_file_write`

### 7.3 Wrappers without a test file of the same name in `tests/python_tests/`
`components`, `cube`, `plane`, `qrt_file_parse`, and all eight `hdf5_*` wrappers. Some may be covered
by tests with other names (`test_qrt_file_reader.py`) or by the files in `tests/python_tests_disable/`;
check before writing new tests.

### 7.4 API differences between Python and MEX for the same function
Not defects; listed because the spec makes the MEX wrapper the reference for new wrappers only.
- HDF5 functions: Python takes `ix, iy, iz, iw` scalars, MEX takes a `location` vector; Python
  `hdf5_read_dset` / `hdf5_write_dset` / `hdf5_read_dset_names` have no `prefix` argument;
  `hdf5_write_dset` and `hdf5_reshape_layout` return nothing in Python, the storage dims in MEX.
- `ray_point_intersect`: Python returns the flat hit list plus offsets (`hit_index`, `hit_offset`),
  MEX a zero-padded `[max_hits, n_points]` matrix.
- `arrayant.generate`: different argument list (`beam_az`, `beam_el`, `beam_weight`, `separate_beams`,
  `apply_weights` instead of `dir` and the `multibeam_sep` type; `rear_gain_lin` instead of `bfr`).
- `cart2geo` / `geo2cart`: `combine` flag in Python, `split` flag and extra sin / cos outputs in MEX.
- `qdant_read`: Python returns `layout` as a key of the arrayant dict, MEX as a separate output;
  Python `id=0` reads all entries, MEX reads all when `id` is omitted.

## Suggested order
1. Section 2 (docs that mislead users) and item 1.2 — small, safe, no API change except 2.1.
2. Items 1.3 to 1.7 — each needs a short check, then a small fix (1.7 is a change in the C++ library).
3. Section 3 — one mechanical pass, including the full comment-versus-dispatcher diff.
4. Item 1.1 — needs a test or benchmark first and an audit of all wrappers; highest impact.
5. Sections 5 and 6 — cosmetic, can be done file by file.
6. Section 4 — only where an API change is wanted.
