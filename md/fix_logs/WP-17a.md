# WP-17a log

- Date 2026-10-22. The work started from commit `83ed6a0` (branch `fixes`, clean tree; `build_linux`, `build_generic`,
  `build_asan` current). Load (1 min) during the suite runs 0.9 to 10.0 (per run below; other sessions were active).
- Scratch: `build_linux/scratch/fix/WP-17a/` (`suites.sh`, `baseline.txt`, `final.txt`, `asan.txt`, `before.sh` and
  `before_*.txt` (test files built against the libraries of `83ed6a0`, saved as `libquadriga_old.a` /
  `libquadriga_gen_old.a`), `mex_before.m` / `mex_before.txt`, `a6/`, `b4/`, `bench/`).
- Not done here, as instructed: `make documentation`, `html_docu/`, MATLAB, AFL, GPU. No file of the acoustic session was
  touched (`src/ray_mesh_interact.cpp`, `src/ray_progress.cpp`, `test_wave_*`, `test_acoustic_*`, `ray_test_helpers.hpp`,
  `tests/data/wave_twins.tsv`, `md/acoustic*.md`, `md/wave.md`).

| Item | Result | Test added (file, case name) | Failed before the fix? | Files changed |
|---|---|---|---|---|
| B-060 | fixed (doc) | none (comments) | - | `include/quadriga_tools.hpp`, `include/quadriga_channel.hpp` |
| B-102 | fixed | `test_arrayant_obj_export.cpp` "Arrayant OBJ Export - every vertex is written once per element" (point 7); point 6 not testable (needs > 2^31 pattern samples, > 34 GB) | point 7 yes: 92 unique of 96 written vertices per element (`before_objexp.txt`) | `include/quadriga_arrayant.hpp`, `src/qd_arrayant.cpp`, `src/qd_arrayant_generate.cpp`, `src/qd_arrayant_interpolate_avx2.cpp`, `api_python/qpy_arrayant_generate.cpp` |
| B-104 | fixed (points 1-4 and 6-9; 5 was fixed in WP-14) | `test_mitsuba_xml_file_write.cpp` "Test Mitsuba XML File Write - PLY paths are relative to the XML file" (6); `test_qrt_write.cpp` "QRT Write - init leaves an existing file unchanged when an input is invalid" (8); 9: none (the static builds cannot show it) | 6: yes, 4 of 16 assertions (`before_mitsuba.txt`); 8: yes, 3 of 325 (`before_qrtw.txt`) | `include/quadriga_channel.hpp`, `include/quadriga_tools.hpp`, `src/qd_channel_hdf5.cpp`, `src/obj_file_read.cpp`, `src/qd_channel_qrt.cpp`, `src/mitsuba_xml_file_write.cpp`, `src/any_type_id.cpp`, `api_python/qpy_mitsuba_xml_file_write.cpp` |
| B-106 | fixed | `test_calc_diffraction_gain.cpp` "Calc Diffraction Gain - verbose debug output with one position" (point 3) | yes: 7 of 515 assertions, the printed destination of a ray changed between the iterations of one segment (`before_cdg.txt`); the ASan build of the probe `a6/probe.cpp` reported heap- and stack-buffer-overflow at `calc_diffraction_gain.cpp:500-502` | `src/calc_diffraction_gain.cpp`, `src/path_to_tube.cpp`, `src/qd_path.cpp`, `include/quadriga_channel.hpp` |
| B-108 | fixed (points 1, 2, 4, 5, 6; 3 was fixed in WP-12) | `test_ray_triangle_intersect.cpp` "Ray-Triangle Intersect - Mesh with extra columns and row count of dest" (2); `test_runtime_avx2_check.cpp` "quadriga_lib_has_AVX2 - agrees with the CPU features" (6) | 2: yes, exception "Input 'mesh' must have 9 columns" (`before_rti.txt`); 6: no, this CPU has AVX2, FMA and YMM state, so the old check gives the same answer | `src/ray_triangle_intersect.cpp`, `src/triangle_mesh_aabb.cpp`, `src/runtime_AVX2_Check_avx2.cpp`, `src/obj_overlap_test.cpp`, `src/fastmath_avx2.h`, `src/avx_mathfun.h` (deleted), `src/point_cloud_segmentation.cpp`, `src/triangle_mesh_segmentation.cpp`, `src/icosphere.cpp`, `src/subdivide_triangles.cpp`, `include/quadriga_tools.hpp`, MEX / Python blocks |
| B-115 | fixed | MOxUnit: `test_version.m` (new, a), `test_qdant_read.m` (c), `test_interp.m` (e), `test_calc_diffraction_gain.m` (empty `center_freq`), `test_generate_diffraction_paths.m` (empty `lod`) | yes, all five (`mex_before.txt`) | `api_mex/*` (see A10 / B6), `src/mex_helper_functions.hpp`, `CMakeLists.txt` |
| WP-16 notes | fixed | `test_sos_map.cpp` "sos_map - a map cube that shares memory with 'freq' or 'phase' throws"; `test_sos_acf_approx.cpp` "sos_acf_approx - an output that shares memory with an input throws"; `test_sos_preset.cpp` "sos_preset - outputs that share memory with each other throw" | yes: 5 / 5 / 7 assertions (`before_sos_*.txt`) | `src/sos_map.cpp`, `src/sos_acf_approx.cpp`, `src/sos_preset.cpp`, `src/quadriga_lib_helper_functions.hpp` |
| WP-13 notes | fixed | `test_interp.cpp` "interp - non-finite grid values in 'xi' or 'yi' throw"; `test_acdf.m` checks rewritten | Catch2 yes, 27 of 131 (`before_interp.txt`) | `src/interp.cpp`, `api_mex/interp.cpp`, `tests/mex_tests/test_acdf.m` |
| WP-11 note | fixed | `test_qd_ieee_chan.cpp` "IEEE Chan - Tap spacing 10 ns / 2^k and non-finite carrier frequency, distance and speeds" | yes, 15 of 36 (`before_ieee.txt`) | `src/ieee_channel_model_functions.hpp`, `src/get_channels_ieee_indoor.cpp`, MEX / Python blocks |
| WP-10 note | fixed | `test_qdant_write.cpp` "Arrayant writing to QDANT file - append to a file with a namespace prefix" | yes: "Could not find an arrayant object with ID 2 in the file." (`before_qdw.txt`) | `src/qd_arrayant_qdant.cpp` |
| WP-09 note | fixed | `test_arrayant.cpp` "Arrayant remove_zeros - name and the azimuth spacing at the wrap-around" | yes, 4 of 8 (`before_arr.txt`) | `src/qd_arrayant.cpp` |
| WP-08 notes | fixed | `test_fast_geo2cart.cpp` "fast_cart2geo - NaN input gives NaN elevation" (float, double; both kernels) | yes, 36 of 260 in both kernels (`before_geo.txt`) | `src/fastmath_avx2.h`, `src/fastmath_vectorized_generic.h`, `src/fastmath_armadillo.cpp`, MEX / Python blocks |
| WP-06 note | fixed (doc) | - | - | `include/quadriga_tools.hpp` |
| WP-03 notes | fixed | `test_quantize_delays.cpp` (one assertion added to the validation case) | yes (`before_qd.txt`) | `src/qd_channel_quantize_delays.cpp`, `include/quadriga_channel.hpp` |
| WP-02 note | fixed (doc, message) | - | - | `include/quadriga_channel.hpp`, `src/qd_channel_hdf5.cpp` |

## Checklist

### A. Code fixes

- **A1** (B-102.6) done. `qd_arrayant_interpolate_avx2` calls the generic `qd_arrayant_interpolate` with the same
  arguments when `e_theta_re.n_elem > INT32_MAX` (D3, no error); the generic kernel has the same input contract
  (pre-allocated outputs). All gather offsets are below the cube size, so this bound covers every int32 index. Not
  testable (four cubes of 2^31 values, > 34 GB).
- **A2** (B-102.7) done. `export_obj_file` uses a `found` flag; vertex 0 is no longer written again. Test counts the
  unique `v` lines per object and checks that the largest face index of each object is its last vertex.
- **A3** (B-104.6) done. The XML stores `<stem>_meshes/<object>.ply` (`generic_string`, `/` separator), relative to
  the XML file as Mitsuba resolves it (D4). Note on the manager's remark: the WP-14 byte-identity test writes
  `mref.xml` into the working directory, so its expected path `mref_meshes/wall.ply` was already relative and stays
  valid; that expectation was **not** changed. The new test writes into a subdirectory, by relative and by absolute
  path. Documentation (C++, Python; no MEX wrapper exists) says the path is relative to the XML file.
- **A4** (B-104.8) done. `qrt_file_init` checks everything before the file is opened: the RX name length (moved up),
  `cir_pos` / `dest_names` sizes that do not fit `unsigned` (they were cast silently), and `cir_offset[0] == 0` (the
  reader rejects every other value with "Invalid CIR index", so such a call wrote an unreadable file). The new
  `cir_offset` check is a small addition beyond the listed name check; it is listed under behaviour changes. Ascending
  order and range of the further `cir_offset` entries are not checked (the reader does not check them either).
  Documentation of `fn`, `dest_names`, `cir_offset` updated.
- **A5** (B-104.9) done. `any_type_id` compares `par_data->type() == typeid(T)` (31 places) instead of the name
  pointers. No test can show the shared-library case in the static builds; the existing `any_type_id` / HDF5 tests
  pass.
- **A6** (B-106.3) done. The debug print indexes `s_orig`, `s_dest`, `fbsN`, `sbsN` with `iR`. The single-cube scene of
  the bug report never put a ray with `iG >= R` into a second iteration (the inner rays hit, the outer ones finish
  first), so the test uses two cubes beside the direct path (`yoff` 0.6 and 1.0, lod 4, 0.3 GHz), captures `std::cout`,
  and checks that the print does not throw, that the gain equals the quiet run, and that the destination printed for a
  ray is the same in every iteration of a segment. `[wave]` / `[acoustic]` unchanged (see Suites).
- **A7** (B-108.2) done (D5). `triangle_mesh_aabb` accepts `>= 9` columns and uses the first 9 (the column-major
  layout makes them the first `9 * n_face` values); its message says "at least 9 columns". `ray_triangle_intersect`
  throws when `dest` has a different row count (was: only fewer rows); the message text "Number of elements in 'orig'
  and 'dest' dont match." is kept, since the MEX test pins it. Two existing tests needed an update because of D5:
  - `tests/mex_tests/test_triangle_mesh_aabb.m:74` expected "must have 9 columns" (now "at least 9 columns");
  - `test_ray_triangle_intersect.cpp` "Outputs that alias the inputs" (WP-12) passed a `dest` with an extra row to
    force a reallocation when the output is written into `dest`; it now uses an extra column instead (same intent).
  Documentation (C++, MEX, Python) says further mesh columns are ignored. `[wave]` / `[acoustic]` unchanged.
- **A8** (B-108.5) done. `src/avx_mathfun.h` deleted (included nowhere, also not by CMake or packaging; third-party
  zlib code, not listed in any notice file), `arma::mt19937_64_instance` removed from `src/obj_overlap_test.cpp`,
  `_fm256_polar_ps` removed from `src/fastmath_avx2.h`. The deletion of `avx_mathfun.h` is unstaged (`git status`: ` D`).
- **A9** (B-108.6) done. `runtime_AVX2_Check` tests CPUID leaf 1 ECX bits 12 (FMA), 27 (OSXSAVE), 28 (AVX), XGETBV
  XCR0 bits 1-2 (SSE and YMM state saved by the OS) and leaf 7 EBX bit 5 (AVX2), and returns false before any AVX
  instruction; the leaf count is checked first. The result is computed once (function-local static), so the many
  callers no longer run CPUID on every call. The `std::vector` is gone (the file is compiled with `-mavx2`, so its
  zero-initialisation could have used VEX instructions before the check). The MSVC branch uses `__cpuid`, `__cpuidex`
  and `_xgetbv`; it was not compiled here. The new test compares with `__builtin_cpu_supports("avx2") &&
  __builtin_cpu_supports("fma")` (GCC/Clang) and with `quadriga_lib_has_AVX2()`.
- **A10** (B-115) done:
  - (a) `version` uses `quadriga_lib:CPPerror` for both errors.
  - (c) `arrayant_qdant_read`: `nargout` 0 is treated as 1 (struct to `ans`), 1-2 give the struct (and layout), 3-12
    separate fields (3 was assigned nothing); multi-entry files: 0 behaves as 1, more than 2 is the existing
    "Separate-fields" error. Documentation: "Separate-fields output (3 to 12 outputs)".
  - (e) `interp` with a non-empty `y` and no `yI` throws `Input 'yI' is required when 'y' is given (2D case).` before
    the `nlhs == 0` shortcut (D6). `yI = []` stays a valid empty query.
  - `calc_diffraction_gain` `center_freq = []` throws `Input 'center_freq' cannot be empty.`;
    `generate_diffraction_paths` `lod = []` throws `Input 'lod' cannot be empty.` (no documented default; the MEX
    function requires all 4 inputs).
  - The string-literal `copy` flags were already corrected in fd4cfe0 (`true, "i_element"` at all five places); only
    the `i_freq` part was left: `arrayant_export_obj_file` message `Input 'i_freq' is out of bound.`, the variable name
    of `qd_mex_get_scalar`, and two documentation lines that said `freq`.
  - `src/mex_helper_functions.hpp`: `qd_mex_copy2matlab` (Col / Mat / Cube) copies at most `ns` elements / columns /
    slices without `is` (further output stays zero); `qd_mex_vector2matlab` (Col / Mat / Cube) without `is` no longer
    reads an index array past `input->size()` (the heap array `js` is gone; snapshots after the last input stay zero
    or `padding`); the `int` branch added to the Cube version; `qd_mex_matlab2vector_Bool`, `qd_mex_set_field`,
    `qd_mex_has_field` are `inline`. No current caller reaches the changed paths; checked by reading and by the MEX
    suite.
  - MOxUnit tests for (a), (c), (e), empty `center_freq` and empty `lod`, all flag style with `[~] = ...`.
- **A11** (WP-16 notes) done. `qd_sos_share_memory` got a pointer/length form and a Cube-vs-Mat form (the Mat form now
  calls the pointer form). `sos_map` checks each existing cube of `map` (up to `n_gen`) against `freq` and `phase`
  before anything is resized or written: `Output 'map' cannot share memory with 'freq' or 'phase'.` (was the message of
  `sos_val` naming `val`, and only in the one-block path; separate cubes overwrote `phase` after use).
  `sos_acf_approx`: `Output 'acf_approx' cannot share memory with an input.` (against `freq` and `dist`; `dist_2d` may
  still be `dist`). `sos_preset`: each pair of `freq`, `acf`, `dist` must be different objects without shared memory:
  `Output '<later>' cannot share memory with '<earlier>'.` C++ documentation blocks updated; the wrappers allocate
  their outputs, so MEX / Python cannot reach the checks.
- **A12** (WP-13 notes) done. `interp_2D` (and through it `interp_1D`) throws `Input 'xi' must be finite.` /
  `Input 'yi' must be finite.` after the size check. `tests/mex_tests/test_acdf.m`: the two `try ...; error(...);
  catch; end` checks are flag checks with `[~] = ...` and the exact messages; they pass. (The same broken pattern is in
  `test_calc_angular_spreads_sphere.m:107`, see "Found on the way".)
- **A13** (WP-11 note) done. `qd_ieee_indoor_param`: carrier frequency must be finite; distances must be finite; tap
  spacing: `10 ns / tap` rounded to an integer `bw` in 1 ... 2^53, `bw` a power of 2 and `|tap * bw - 10 ns| <= 1e-9 *
  10 ns` (NaN, Inf, 0, negative, 7 ns, 4 ns, 3 ns, 12 ns, 5.000001 ns and 1e-300 s throw; `10 ns / 2^k` for k = 0 ... 6,
  also with a relative offset of 5e-10, are accepted). The old code rejected `10 ns * (1 + 5e-10)` (strict `> 10 ns`
  check), which the relative tolerance now accepts. `get_channels_ieee_indoor`: station and environment speed must be finite. Documentation
  (C++, MEX, Python) adds "(finite)" to carrier frequency, distance and both speeds.
- **A14** (WP-10 note) done. When the writer adds to an existing file with a namespace prefix, all nodes it creates
  (`arrayant`, `name`, `CenterFrequency`, `NoElements`, `ElementPosition`, grids, coupling, `E*Mag` / `E*Phase`) get
  the prefix, also when an entry is overwritten in place. A new file is unchanged (no prefix). Test: append ID 2 to a
  prefixed file, read back ID 1 and 2, overwrite ID 1, check that no unprefixed node is in the file.
- **A15** (WP-09 note) done. `remove_zeros(&output)` copies `name`; the azimuth spacing of the first point to its
  previous neighbour is `az(0) - az(n-1) + 2 pi`, and of the last point to its next neighbour `az(0) + 2 pi - az(n-1)`
  (both were `az(0) + az(n-1)`; the second formula had the same error). For grids from -180 to 180 deg (both ends
  present) both formulas give 0 and nothing changes; grids that do not reach 180 deg now keep the right neighbours
  (test: a -30 ... 30 deg sector).
- **A16** (WP-08 note) done. Generic kernel: a NaN length gives `zn = NaN`, so `el = asin(NaN) = NaN`. AVX2 float
  kernel: `el |= unordered(r, r)`; AVX2 double kernel: the same with the double mask of each half narrowed to float
  (an all-ones double converts to a float NaN). Azimuth is unchanged (NaN for NaN in x or y; for NaN in z the AVX2
  double kernel computes it from the unscaled x, y, which differs from a finite-z result only in the last float digit).
  Documentation (C++, MEX, Python, kernel comment) says NaN in `az`, `el` and `len`.
- **A17** (WP-03 note) done. `quantize_delays` rejects a NaN `power_exponent` (`!(power_exponent > 0)`, same message).

### B. Documentation

- **B1** done. `include/quadriga_tools.hpp`: Mitsuba writer `obj_ind` 0-based, `obj_names` length `max(obj_ind)+1`,
  `mtl_names` length `max(mtl_ind)`; `obj_file_read` `obj_names` `[max(obj_ind)+1]`, `mtl_ind` and `csv_ind` 1-based
  (0 = no material), `mtl_names` `[no_mtl]`, `csv_names` the full table `[n_csv]`, `csv_strict` "false maps it to
  index 0 (no material)" (was "defaults to first scv materia"); `obj_file_write` `obj_ind` 0-based (WP-06 note),
  `mtl_ind` 1-based (0 = no material), `obj_names` length `> max(obj_ind)`, `mtl_names` length `>= max(mtl_ind)`.
  `include/quadriga_channel.hpp`: `calc_effective_path_gain` sums over all antenna pairs (decision B-060 (a), code
  unchanged); `qrt_file_read` `normalize_M = 1`: maximum column power of 1.
- **B2** done. `generate_arrayant_omni` in the usage example; "0-based"; `az_point_ang` in the MD declaration of
  `calc_beamwidth_deg`; `generate_speaker` header: one element per arrayant, multi-driver speakers via `append` and
  `element_pos` (the two lines about driver positions / orientations replaced), horn coverage 0 defaults to 90 / 60 deg
  and only `horn_control_freq` is derived from the radius; multibeam `weight` "not normalized" (C++, Python; the MEX
  block has no such statement).
- **B3** done. 1: `nx = 65536` in the `hdf5_write` comment; message "Error creating file. It may be already opened."
  2: `obj_file_read` (C++): the table lookup uses the name before the first dot (D2), as the MEX and Python blocks and
  the code comment already said. 3: `qrt_file_read_raw` no longer mentions `iR`. 4: `map_to_itu` comment "air".
  7: `conf_file_write` header comment uses the wording of the sources (up to 15 digits, 16 or 17 where 15 do not read
  back, Inf written as such).
- **B4** done, **with a correction of the checklist**: `path_to_tube` splits where the path turns by more than **160
  deg**, not 170 deg. The code compares the cosine of the angle between the two segments with `0.9397 = cos(20 deg)`
  (its comment said `cosd(10)`, and the bug report took the 10 deg from that comment). Measured (`b4/out.txt`):
  deviations 150, 158, 159.5 deg give 3 vertex rings, 160.5, 162, 165, 169, 171 deg give 4 (the bug report's own repro
  also showed a split at 165 deg). The documentation follows the code (160 deg; the segments enclose less than 20
  deg), and the code comment says `cosd(20)`; the threshold itself is unchanged. `qd_path`: 56-byte header, 2 inline
  type codes, `(nSEG+1)/4` floats for `nSEG > 2` (MD tables and both box comments), no `iR` row; `extend` comment:
  "nREF and nTRA are not changed" (header and the copy in `qd_path.cpp`).
- **B5** done. `forward_index(i)`: 1-based row in `points` of row i of `pointsR` (0 = padding), `reverse_index(j)`: row
  in `pointsR` of row j of `points` (0-based in C++ / Python, 1-based in MEX); `mesh_index`: 1-based row in `mesh` of
  each row of `meshR` (C++, MEX, Python, header). Header typos: `point_cloud_split` `[n_pointsA, 3]` / `[n_pointsB, 3]`;
  `plane` scale (x and y, z ignored); `icosphere` `vert` `[x1 y1 z1 x2 y2 z2 x3 y3 z3]`; "Directions of the vertex
  rays" (C++, MEX, Python); `subdivide_triangles` column comments of the upper triangle.
- **B6** done. (b) "Separate outputs (2 to 11 outputs)"; (d) "Row vectors are stored as given (`[1, n]`)";
  `point_inside_mesh` `obj_ind` already said 1-based (no change); typos "Variable", "split mode" (`arrayant_combine_pattern`;
  `arrayant_generate` has no "slit-mode"), "exists" / "existing" / "throw" and "a multi-frequency array" in
  `arrayant_qdant_write` (message and comment), "Row" (`point_cloud_aabb`), "multiple" (`interp`), "existing"
  (`hdf5_reshape_layout`); "Forsingle" and "gnerate" were no longer present. `CMakeLists.txt:196` already said that
  the different soname separates the libraries (WP-01), so no change there; the MATLAB API note (was `:498`) now says
  the static link spares a `libquadriga.so` at run time and the sonames keep the HDF5 libraries apart. Configure run:
  `cmake --build build_linux` re-ran the configure step ("Configuring done", "Generating done", 0 warnings).
- **B7** done. SLERP comment: the weighted direction sum is computed in single precision with one FMA per component
  (the double-precision bullet and its ULP claim removed).
- **B8** done. `quantize_delays` header: `delay` finite, `> -tap_spacing/2` (values below 0 count as 0), `< 2^31 *
  tap_spacing`; `delay_quant` `[1, 1, n_taps]` only for delay input `[1, 1, n_path]` with `fix_taps` 1 or 2.

## Performance

Harness `build_linux/scratch/fix/WP-17a/bench/` (`run_bench.sh`, `bench_wp17a.cpp`): `src/fastmath_armadillo.cpp`,
`src/interp.cpp`, `src/fastmath_vectorized_avx2.cpp` and `src/runtime_AVX2_Check_avx2.cpp` (the last two with
`-mavx2 -mfma`) and their private headers of `83ed6a0` ("old") against the work tree ("new"), same flags as
`build_linux`, linked with the same benchmark and `build_linux/libquadriga.a`; symbol check passed, 4 of 4 objects
differ. Workloads C1, C2 (`fast_geo2cart`, as WP-08), K1-K6 (`fast_cart2geo`, 2^20 values x 20 calls; K5, K6 with 1 %
NaN inputs), I1-I3 (interp, as WP-13). 1 thread, pinned to the idlest core of the less busy CCD (core 15), 3
alternating runs of 7 repetitions, median, under `flock /tmp/bench.lock`. Noise floor (old vs old,
`timing_noise_floor.txt`, load 1.5 / 2.2): 0.984 to 1.000 (minimum 0.990 to 1.011).

Final run (`timing_final.txt`), load (1 min) 3.11 before / 2.82 after (below 4):

| Workload | old ms | new ms | new/old | min new/min old | checksum |
|---|---|---|---|---|---|
| C1 geo2cart float AVX2 | 20.584 | 20.887 | 1.015 | 1.022 | identical |
| C2 geo2cart double AVX2 | 45.360 | 45.360 | 1.000 | 1.001 | identical |
| K1 cart2geo float AVX2 | 23.175 | 23.687 | 1.022 | 1.025 | identical |
| K2 cart2geo double AVX2 | 54.835 | 55.937 | 1.020 | 1.018 | identical |
| K3 cart2geo float generic | 317.318 | 319.814 | 1.008 | 1.008 | identical |
| K4 cart2geo double generic | 736.326 | 727.630 | 0.988 | 0.990 | identical |
| K5 cart2geo float AVX2, 1 % NaN | 23.133 | 23.688 | 1.024 | 1.020 | sum identical, hash differs (NaN elevations) |
| K6 cart2geo double generic, 1 % NaN | 734.947 | 728.272 | 0.991 | 0.989 | sum identical, hash differs (NaN elevations) |
| I1 interp_1D 160k x 160k | 6.218 | 6.276 | 1.009 | 0.999 | identical |
| I2 interp_2D 361x181, 100x100, x200 | 2.813 | 2.803 | 0.996 | 1.008 | identical |
| I3 interp_1D 10 x 100k, x10 | 5.088 | 5.083 | 0.999 | 0.997 | identical |

All workloads below the limit of 1.05. The AVX2 cart2geo kernels pay 2 % for the NaN mask (one compare and one OR per
8 values, plus two conversions in the double kernel); C1 (`fast_geo2cart`, code unchanged) shows 1.5 %, which is
noise or code layout. The K5 / K6 sums cover the finite outputs and are identical; only the hash of the output bytes
changes, because the elevations of the NaN inputs are now NaN.

## Suites

| Run | Result | Time | Load (1 min) before / after |
|---|---|---|---|
| Baseline Catch2 AVX2 (`83ed6a0`) | `All tests passed (1346786 assertions in 1285 test cases)` | 19 s | 0.9 / 1.2 |
| Baseline Catch2 generic | `All tests passed (1345004 assertions in 1284 test cases)` | 20 s | 1.2 / 1.5 |
| Baseline `[wave]` / `[acoustic]`, AVX2 and generic | `163474 / 178` and `42741 / 59` in both builds | 6 s each | 1.5 / 1.6 |
| Baseline MEX, Octave 11.1 | `FAILED (passed=66, failure=1)`, only the excluded `test_ray_mesh_interact:113` | 1.2 s | 1.6 |
| Baseline pytest | `817 passed, 1 skipped, 52 subtests passed` | 2.5 s | 1.7 |
| Catch2 AVX2 (final) | `All tests passed (1348756 assertions in 1300 test cases)` (+15 cases, +1970 assertions) | 19 s | 10.0 / 8.5 |
| Catch2 generic (final) | `All tests passed (1346833 assertions in 1299 test cases)` (+15 cases, +1829 assertions) | 19 s | 8.5 / 6.7 |
| `[wave]` / `[acoustic]` AVX2 and generic (after A6 / A7 and on the final code) | `163474 / 178` and `42741 / 59` in both builds, same as baseline | 6 s each | 6.7 / 6.0 |
| ASan + UBSan `build_asan/test_bin`, `halt_on_error=0:detect_leaks=1:suppress_equal_pcs=0`, 12 threads (final) | `All tests passed (1348756 assertions in 1300 test cases)`; no ASan report, no leak; UBSan only the 6 excluded `ray_progress` reports (`access.hpp:26` / `Mat_meat.hpp:7490`, from `src/ray_progress.cpp:829-831` via `test_wave_progress.cpp:299`) | 126 s | 3.5 / 2.1 |
| MEX, Octave 11.1 (final, `ulimit -v 16000000`) | `FAILED (passed=67, failure=1)`, only the excluded `test_ray_mesh_interact:113`; 68 test files (new `test_version.m`) | 1.3 s | 6.0 |
| pytest (final, `ulimit -v 16000000`) | `817 passed, 1 skipped, 52 subtests passed` (no Python test changed) | 2.5 s | 6.0 |

All `test_bin` runs through `flock /tmp/quadriga-fixes-suite.lock` from the repository root, 12 threads; builds one
after the other with `--parallel 12`, 0 warnings (`build_linux` 2 min 4 s including the configure step after the
`CMakeLists.txt` edit, installed; `build_generic` 1 min 15 s; `build_asan` 9 min 43 s, only GCC's
"variable tracking size limit exceeded" notes in `test_wave_*`). The load was raised by other sessions during the
final Catch2 runs (6 to 10); the results are pass/fail and not affected. No GPU, no AFL, no MATLAB (WP-17b).

Before-the-fix evidence: every new Catch2 test was compiled against the saved libraries of `83ed6a0`
(`before.sh`, outputs `before_*.txt`, numbers in the item table); the MEX tests ran against the MEX files of `83ed6a0`
before they were rebuilt (`mex_before.txt`: `test_version`, `test_qdant_read`, `test_interp`,
`test_calc_diffraction_gain`, `test_generate_diffraction_paths`, `test_triangle_mesh_aabb` all failed at the new
checks). Not shown to fail before: A1 (needs > 34 GB), A5 (static build), A9 (this CPU has AVX2, FMA and YMM state),
the helper changes of A10 (no caller).

## Behaviour changes for users

- `mitsuba_xml_file_write` (C++, Python): the XML refers to the PLY files as `<stem>_meshes/<object>.ply`, relative to
  the XML file; before, the path as given to the writer was stored (absolute when an absolute file name was given,
  relative to the working directory otherwise). XML and mesh folder can now be moved together. The bytes of an XML
  written into the working directory are unchanged.
- `ray_triangle_intersect` (C++, MATLAB, Python): `dest` with more rows than `orig` throws (was accepted, extra rows
  ignored). Meshes with more than 9 columns are accepted with and without `aabb` (extra columns ignored); before, they
  threw without `aabb`. `triangle_mesh_aabb` accepts more than 9 columns; message "must have at least 9 columns".
- MATLAB / Octave: `version` errors have the identifier `quadriga_lib:CPPerror`. `arrayant_qdant_read` works with every
  number of outputs from 0 to 12 (3 gave no outputs; 0 on a multi-entry file was an error). `interp` with `y` and
  without `yI` throws (was an interpolation at y = 0). `calc_diffraction_gain` with `center_freq = []` throws (was
  1 GHz); `generate_diffraction_paths` with `lod = []` throws (was 1). `arrayant_export_obj_file` message names `i_freq`;
  `arrayant_qdant_write` message spelling.
- `interp_1D` / `interp_2D` (C++, MATLAB): NaN or Inf in `xi` / `yi` throws (before: NaN results at some queries, or an
  exception from the sort).
- `get_channels_ieee_indoor` (C++, MATLAB, Python): a tap spacing that is not `10 ns / 2^k` within a relative 1e-9
  throws (7 ns was treated as 10 ns); NaN / Inf carrier frequency, distance or speeds throw.
- `fast_cart2geo` (C++; MATLAB / Python `cart2geo`): NaN input gives NaN elevation (was 0) in both kernels.
- `quantize_delays` (C++, MATLAB, Python): NaN `power_exponent` throws (gave NaN coefficients).
- `qrt_file_init` (C++): an invalid input leaves an existing file unchanged (an RX name over 255 characters used to
  leave a partly written file); `cir_offset` must start with 0 (such files could not be read).
- `sos_map`, `sos_acf_approx`, `sos_preset` (C++ only): outputs on hand-built foreign memory that overlaps an input
  (or, for `sos_preset`, another output) throw before anything is written.
- `arrayant::remove_zeros(&output)` (C++): `output.name` is set; azimuth grids that do not reach +/-180 deg keep the
  correct neighbours of the first and last grid point. `export_obj_file`: vertex 0 of each element is written once.
- qdant writer (C++, MATLAB, Python): an entry appended to a file with a namespace prefix uses that prefix and reads back.
- `quadriga_lib_has_AVX2()` returns false on CPUs / VMs without FMA or OS YMM support (instead of faulting).

## Found on the way

- `tests/mex_tests/test_calc_angular_spreads_sphere.m:107-112`: the same `try ...; error('Expected an error'); catch;
  end` pattern as in `test_acdf.m`; it passes whether or not an error is raised. Not changed (not in the checklist).
- `src/path_to_tube.cpp:392`: the projection comment derives the distance factor 5.7 from a minimum angle of 10 deg;
  with the split at 20 deg the minimum is 20 deg, so the factor is conservative. Not changed.
- `src/mex_helper_functions.hpp` still selects the MATLAB class with `typeid(dtype).name() == typeid(T).name()`
  (compile-time types; harmless, `std::is_same` would be the clean form). Not changed.
