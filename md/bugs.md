# Bug hunt on branch `fixes`

- Date: 2026-10-15
- Branch: `fixes`, commit 4720a92 ("Acoustic round 1, phase 3: the plate rule and the decided machine, built")
- Worktree: `/sjc/quadriga-lib-fixes`; task description: `md/bug_hunt_brief.md`
- Marks: **[R]** reproduced by a program or script run in this session; **[M]** measured (timings, counts, memory);
  **[C]** read from the code, not reproduced.
- Scope: the whole library except the ray-interaction machinery and CUDA (see the brief). Severity scale S1 (critical)
  to S4 (cosmetic) as defined in the brief; each item gives its reason in one sentence.

How to read the items:

- Items are numbered in the order of the summary table (severity, then area). Area letters: A arrayant, B channel
  object and file I/O, C channel generation and tools, D geometry and math, E sum of sinusoids, F MATLAB/Octave MEX
  layer, G Python layer. Kinds: 1 defect (wrong result, crash, UB, memory, race, leak), 2 documentation, 3 API
  inconsistency, 4 performance.
- Every [R] item contains its reproduction in full: the source files, a `run.sh` that changes to the worktree root,
  builds against the libraries of the Baseline section and runs, and the output of the phase-2 re-run of exactly that
  script. The folder names under `build_linux/scratch/hunt/final/` are the IDs used during the hunt (e.g. `B-006`,
  `C-1`); they are not the item numbers of this document. If the scratch folder is lost, save the shown files into a
  folder of that name and run `run.sh`.
- All reproductions were re-run once from the scripts shown here on 2026-10-15 after the items were written
  (sequentially, one script at a time; see "Re-run" at the end). Sanitizer and timing output contains addresses and
  times that differ from run to run. Timing items state the load; measurements at a load above about 4 are marked
  as contaminated or provisional.
- Several items call for a decision that belongs to the user (which side, code or documentation, is the intended
  behaviour). They say "the user decides". They are collected here so that they can be answered in one pass:
  B-001, B-002, B-007, B-012, B-019, B-023, B-041, B-045, B-047, B-052, B-060,
  B-069, B-070, B-074, B-077, B-082, B-087, B-088, B-091, B-092, B-095, B-098,
  B-099, B-113.
- Running a MATLAB reproduction that crashes (B-029, B-032, B-033, B-034) writes a crash dump
  `~/matlab_crash_dump.<pid>-1` into the home directory.

## Summary table

Items: 123 (S1: 5, S2: 31, S3: 68, S4: 19); B-119 to B-123 were found during the fixes (WP-02, WP-04, WP-07).
Status: open: 4, fixed: 109, partly fixed: 2, won't fix: 1, documented: 7 (after WP-16).

| ID | Title | Area | Kind | Severity | Mark | Status |
|---|---|---|---|---|---|---|
| [B-001](#b-001) | `hdf5_write` silently drops the `par_*` data when the slot already holds a `Data` group | B channel, file I/O | 1 defect | S1 | [R] | fixed (WP-02) |
| [B-002](#b-002) | `quantize_delays` segfaults on a negative delay, including the slightly negative relative delays that `get_channels_spherical` returns | B channel, file I/O | 1 defect | S1 | [R] | fixed (WP-03) |
| [B-003](#b-003) | `get_channels_spherical`, `_multifreq` and `_irs` ignore `path_length` for NLOS paths: delays and phases too short | C channel gen., tools | 1 defect | S1 | [R] | fixed (WP-04) |
| [B-004](#b-004) | Python `tools.geo2cart` / `tools.cart2geo` with separate inputs read freed memory: wrong values, segfault | G Python layer | 1 defect | S1 | [R] | fixed (WP-05) |
| [B-005](#b-005) | Python HDF5 writers ignore numpy strides for 1-D and (n,1) arrays: wrong data written, heap over-read | G Python layer | 1 defect | S1 | [R] | fixed (WP-05) |
| [B-006](#b-006) | AVX2 antenna interpolation reads up to 56 bytes past the angle matrices (per-element angles, n_out not a multiple of 8) | A arrayant | 1 defect | S2 | [R] | fixed (WP-09) |
| [B-007](#b-007) | Generic antenna interpolation reads before the pattern buffer for NaN or infinite angles | A arrayant | 1 defect | S2 | [R] | fixed (WP-09) |
| [B-008](#b-008) | `arrayant::append` builds a wrong coupling matrix, and segfaults when `coupling_im` is empty | A arrayant | 1 defect | S2 | [R] | fixed (WP-09) |
| [B-009](#b-009) | `arrayant::rotate_pattern` loses coupling, centre frequency and name | A arrayant | 1 defect | S2 | [R] | fixed (WP-09) |
| [B-010](#b-010) | `arrayant::calc_beamwidth_deg` is wrong when the main lobe wraps ±180° or points towards a pole | A arrayant | 1 defect | S2 | [R] | fixed (WP-09) |
| [B-011](#b-011) | `generate_arrayant_multibeam` with a beam towards el = ±90° returns wrong ports (pol 1) or segfaults (pol 2, 3) | A arrayant | 1 defect | S2 | [R] | fixed (WP-09) |
| [B-012](#b-012) | qdant writer drops every pattern sample below -100 dB absolute power; a whole low-level element reads back as zeros | A arrayant | 1 defect | S2 | [R] | fixed (WP-10) |
| [B-013](#b-013) | `hdf5_read_channel`: a malformed HDF5 file overflows heap and stack buffers (sizes taken from attributes, not from the datasets) | B channel, file I/O | 1 defect | S2 | [R] | fixed (WP-02) |
| [B-014](#b-014) | HDF5 storage layout: `nx*ny*nz*nw` wraps in 32 bits; `hdf5_write` then reads and writes past the index buffer | B channel, file I/O | 1 defect | S2 | [R] | fixed (WP-02) |
| [B-015](#b-015) | `hdf5_write` copies 256 bytes from the channel name: heap memory ends up in the file; names of 256+ characters over-read on reading | B channel, file I/O | 1 defect | S2 | [R] | fixed (WP-02) |
| [B-016](#b-016) | `qrt_file_read` (C++) fills `aod`, `eod`, `aoa`, `eoa` only when `path_gain` or `path_length` is also requested | B channel, file I/O | 1 defect | S2 | [R] | fixed (WP-01) |
| [B-017](#b-017) | `qrt_file_append` corrupts the file when a CIR orientation is non-zero in double but rounds to 0 in float | B channel, file I/O | 1 defect | S2 | [R] | fixed (WP-07) |
| [B-018](#b-018) | OBJ round trip loses "no material": faces with `mtl_ind = 0` inherit the previous material | B channel, file I/O | 1 defect | S2 | [R] | fixed (WP-14) |
| [B-019](#b-019) | `baseband_freq_response_multi` gives wrong phases for `get_channels_*` output with the default relative delays | C channel gen., tools | 1 defect | S2 | [R] | documented (WP-04) |
| [B-020](#b-020) | `baseband_freq_response`: phase computed in single precision, error grows with delay × bandwidth (both builds) | C channel gen., tools | 1 defect | S2 | [R] | fixed (WP-04) |
| [B-021](#b-021) | IEEE indoor model: `n_floors > 0` shifts all tap delays (3-D path length minus 2-D distance) | C channel gen., tools | 1 defect | S2 | [R] | fixed (WP-11) |
| [B-022](#b-022) | IEEE indoor model prunes taps with an absolute power threshold after path loss (taps vanish with distance, floors, walls) | C channel gen., tools | 1 defect | S2 | [R] | fixed (WP-11) |
| [B-023](#b-023) | IEEE indoor model uplink conjugates the channel (Doppler sign reversed), unlike the reciprocity of `get_channels_spherical` (user decision) | C channel gen., tools | 3 cross-API | S2 | [R] | fixed (WP-11) |
| [B-024](#b-024) | `xpr_update`: data race when `ray_index` contains duplicates (wrong, non-deterministic result with threads) | C channel gen., tools | 1 defect | S2 | [R] | fixed (WP-13) |
| [B-025](#b-025) | `path_to_tube`: NaN vertices when interior path points are collinear | C channel gen., tools | 1 defect | S2 | [R] | fixed (WP-13) |
| [B-026](#b-026) | AVX2 sin/cos returns values far outside [-1, 1], ±inf or NaN for large arguments (fast_sincos, fast_geo2cart, baseband_freq_response) | D geometry, math | 1 defect | S2 | [R] | fixed (WP-08) |
| [B-027](#b-027) | `point_inside_mesh` misclassifies points 5 cm from a face at UTM-like coordinates in AVX2 builds (float ray–triangle kernel) | D geometry, math | 1 defect | S2 | [R] | fixed (WP-12) |
| [B-028](#b-028) | `obj_overlap_test` misses a clear overlap when vertices of the penetrating object lie on extensions of the other object's face planes | D geometry, math | 1 defect | S2 | [R] | fixed (WP-12) |
| [B-029](#b-029) | MATLAB R2026a segfaults at exit after any HDF5 MEX call (the unloaded thread-safe libhdf5 leaves a dangling pthread-key destructor) | F MEX layer | 1 defect | S2 | [R] | fixed (WP-01) |
| [B-030](#b-030) | MEX: sparse inputs are read as dense arrays (wrong values, heap over-read) in every wrapper | F MEX layer | 1 defect | S2 | [R] | fixed (WP-06) |
| [B-031](#b-031) | MEX `qrt_file_read` (`i_cir`) and `hdf5_read_channel` (`snap`) modify the caller's uint64 variable in place | F MEX layer | 1 defect | S2 | [R] | fixed (WP-01) |
| [B-032](#b-032) | MEX `channel_export_obj_file` segfaults when `no_interact` has fewer snapshots than the coefficients | F MEX layer | 1 defect | S2 | [R] | fixed (WP-06) |
| [B-033](#b-033) | MEX `qd_mex_get_strings` dereferences empty cell elements: `cell(1,1)` as a name list crashes MATLAB R2026a | F MEX layer | 1 defect | S2 | [R] | fixed (WP-06) |
| [B-034](#b-034) | MEX helpers convert NaN, negative, fractional and out-of-range numbers to integers without checks: crashes, bad_alloc, silent truncation, wrapped seeds | F MEX layer | 1 defect | S2 | [R] | fixed (WP-06) |
| [B-035](#b-035) | Python `arrayant.interpolate(local_angles=True)` without `dist=True` returns uninitialised memory | G Python layer | 1 defect | S2 | [R] | fixed (WP-05) |
| [B-036](#b-036) | Python `channel.baseband_freq_response`: an exception inside the OpenMP region aborts the interpreter | G Python layer | 1 defect | S2 | [R] | fixed (WP-05) |
| [B-037](#b-037) | AVX2 antenna interpolation returns 0 where all four corner amplitudes are below 1.19e-7 (generic interpolates) | A arrayant | 1 defect | S3 | [R] | fixed (WP-08) |
| [B-038](#b-038) | qdant writer stores element positions, coupling and pattern values with only 6 significant digits | A arrayant | 1 defect | S3 | [R] | fixed (WP-10) |
| [B-039](#b-039) | `qdant_write_multi` deletes an existing file before it validates its input | A arrayant | 1 defect | S3 | [R] | fixed (WP-01) |
| [B-040](#b-040) | `qdant_read(fn, 1)` returns another entry without error when the file has no ID 1 | A arrayant | 1 defect | S3 | [R] | fixed (WP-10) |
| [B-041](#b-041) | `arrayant::qdant_write` without a layout replaces an existing custom layout in the file (user decision) | A arrayant | 2 doc | S3 | [R] | documented (WP-10) |
| [B-042](#b-042) | A 138-byte qdant file makes the reader allocate 4.7 GB and fail with `std::bad_alloc` | A arrayant | 1 defect | S3 | [R] | fixed (WP-10) |
| [B-043](#b-043) | `arrayant` default constructor leaves `check_ptr` uninitialised; the quick `is_valid()` can pass an invalid object | A arrayant | 1 defect | S3 | [R] | fixed (WP-01) |
| [B-044](#b-044) | ULA and 3GPP generators keep the custom pattern's element positions and coupling, contrary to the docs | A arrayant | 2 doc | S3 | [R] | fixed (WP-09) |
| [B-045](#b-045) | `combine_pattern` and `arrayant_combine_pattern_multi` hard-code the speed of light (acoustic arrays combined without position phase) | A arrayant | 1 defect | S3 | [R] | fixed (WP-10) |
| [B-046](#b-046) | `remove_zeros` and `calc_directivity_dBi` use absolute power thresholds (low-level patterns lose data or report 0 dBi) | A arrayant | 1 defect | S3 | [R] | fixed (WP-09) |
| [B-047](#b-047) | `generate_arrayant_multibeam` defaults M = N = 2 in the header, 1 in the docs and both wrappers | A arrayant | 3 cross-API | S3 | [C] | fixed (WP-10) |
| [B-048](#b-048) | QRT readers trust header and path counts: a 136-byte file returns 20 million zero paths or allocates GBs | B channel, file I/O | 1 defect | S3 | [R, M] | fixed (WP-07) |
| [B-049](#b-049) | `qrt_file_read` truncates `i_cir` / `i_orig` to 32 bits before the range check | B channel, file I/O | 1 defect | S3 | [R] | fixed (WP-07) |
| [B-050](#b-050) | HDF5 write/read round trip changes data (empty snapshots, missing frequency, 1-element vectors, empty strings, long names) | B channel, file I/O | 1 defect | S3 | [R] | fixed (WP-02) |
| [B-051](#b-051) | HDF5 functions call `H5close()`, switch off error printing process-wide, and leak handles and memory on error paths | B channel, file I/O | 1 defect | S3 | [R] | fixed (WP-02) |
| [B-052](#b-052) | `hdf5_write` "overwrites" a slot by adding a new group; the file grows with every rewrite | B channel, file I/O | 2 doc | S3 | [R] | fixed (WP-02) |
| [B-053](#b-053) | `hdf5_read_dset` returns a variable-length string as 8 bytes of a heap pointer, and leaks it | B channel, file I/O | 1 defect | S3 | [R] | fixed (WP-02) |
| [B-054](#b-054) | `quantize_delays` builds dense vectors over the absolute delay range: a 20 ms offset makes it 4000 times slower | B channel, file I/O | 4 performance | S3 | [M] | fixed (WP-03) |
| [B-055](#b-055) | `quantize_delays` gives wrong results when the outputs alias the inputs (undocumented) | B channel, file I/O | 1 defect | S3 | [R] | fixed (WP-03) |
| [B-056](#b-056) | `mitsuba_xml_file_write` is quadratic in the number of vertices per object | B channel, file I/O | 4 performance | S3 | [M] | fixed (WP-14) |
| [B-057](#b-057) | OBJ reader: CRLF `.mtl` files lose their materials; face indices unchecked when only `face_ind` is requested; no-output call skips validation | B channel, file I/O | 1 defect | S3 | [R] | fixed (WP-14) |
| [B-058](#b-058) | `write_png`: constant, NaN or inverted-range data trigger a NaN-to-integer cast and all pixels get the last colour; an empty matrix writes an invalid PNG | B channel, file I/O | 1 defect | S3 | [R] | fixed (WP-14) |
| [B-059](#b-059) | `channel::add_paths` accepts inconsistent counts and leaves an invalid object; `qrt_file_read` accepts any `normalize_M` | B channel, file I/O | 1 defect | S3 | [R] | fixed (WP-07) |
| [B-060](#b-060) | Channel and file-I/O header comments contradict the code (index bases, `calc_effective_path_gain` "average", `normalize_M` column power) | B channel, file I/O | 2 doc | S3 | [R, C] | open |
| [B-061](#b-061) | `combine_irs_coord`: no shape check on `interact_coord_1/2`, heap over-read | C channel gen., tools | 1 defect | S3 | [R] | fixed (WP-13) |
| [B-062](#b-062) | `baseband_freq_response`: `aligned_alloc` with sizes that are not a multiple of 32 (generic build) and unchecked NULL return | C channel gen., tools | 1 defect | S3 | [R] | fixed (WP-13) |
| [B-063](#b-063) | Test defect: `test_qd_ieee_chan.cpp:318` reads past the delay cube (hidden in the full ASan suite run) | C channel gen., tools | 1 defect | S3 | [R] | fixed (WP-01) |
| [B-064](#b-064) | IEEE indoor model: `XPR_NLOS_linear = 0` gives NaN coefficients | C channel gen., tools | 1 defect | S3 | [R] | fixed (WP-11) |
| [B-065](#b-065) | Unchecked float-to-integer casts (UB) for non-finite `observation_time` and tiny `quantize` | C channel gen., tools | 1 defect | S3 | [R] | fixed (WP-13) |
| [B-066](#b-066) | `calc_delay_spread` with `granularity`: memory n_cir × global delay span / granularity (800 MB for 100 paths) | C channel gen., tools | 4 performance | S3 | [M] | fixed (WP-13) |
| [B-067](#b-067) | `acdf`: all-NaN data sets bias the averaged CDF; unsorted user bins are accepted | C channel gen., tools | 1 defect | S3 | [R] | fixed (WP-13) |
| [B-068](#b-068) | Docs of `add_fake_los_path` do not match the code: always one extra path; a present LOS is moved to slot 0 and a zero path is left behind | C channel gen., tools | 2 doc | S3 | [R] | documented (WP-04) |
| [B-069](#b-069) | Naming of the cross-polarisation rows of `M` is inconsistent; `calc_cross_polarization_ratio` V/H columns are ambiguous (user decision) | C channel gen., tools | 2 doc | S3 | [R] | documented (WP-04) |
| [B-070](#b-070) | `calc_diffraction_gain` (C++) scalar mode expects an electromagnetic-equivalent frequency, the docs say the speed of sound is used | C channel gen., tools | 2 doc | S3 | [R] | documented (WP-13) |
| [B-071](#b-071) | `ray_triangle_intersect`: AVX2 and generic kernels swap `fbs_ind`/`sbs_ind` for hits at equal distance (shared edge or vertex) | D geometry, math | 1 defect | S3 | [R] | fixed (WP-12) |
| [B-072](#b-072) | `ray_triangle_intersect`: the AABB slab test rejects rays with dz = 0 in the plane z = Zmin/Zmax (x and y are accepted) | D geometry, math | 1 defect | S3 | [R] | fixed (WP-12) |
| [B-073](#b-073) | `ray_point_intersect` requires sub-clouds aligned to 8 only in AVX2 builds; the default `point_cloud_segmentation` output makes it throw | D geometry, math | 3 cross-API | S3 | [R] | fixed (WP-12) |
| [B-074](#b-074) | Invalid `use_kernel` values are accepted silently; `use_kernel = 2` does not throw in builds without AVX2 (confirms item 1.7 of the Python fix plan) | D geometry, math | 2 doc | S3 | [R] | fixed (WP-08) |
| [B-075](#b-075) | AVX2 `fast_atan2` / `fast_cart2geo`: NaN becomes π/4, (inf, inf) gives NaN, tiny or huge double vectors give elevation 0; two doc claims are wrong | D geometry, math | 1 defect | S3 | [R] | fixed (WP-08) |
| [B-076](#b-076) | AVX2 `fast_slerp` fast path: error 3e-6 (about 50 ULP, documented ~5) and each lane's result depends on its neighbours | D geometry, math | 1 defect | S3 | [R] | fixed (WP-08) |
| [B-077](#b-077) | `obj_overlap_test` crashes on an empty mesh | D geometry, math | 1 defect | S3 | [R] | fixed (WP-12) |
| [B-078](#b-078) | `subdivide_triangles` (and `icosphere`, `cube`, `plane`): unchecked overflow of the face count leads to writes through a null pointer | D geometry, math | 1 defect | S3 | [R] | fixed (WP-12) |
| [B-079](#b-079) | An output that aliases an input gives wrong results (`ray_triangle_intersect` with `fbs = &orig`, `subdivide_triangles` in place) | D geometry, math | 1 defect | S3 | [R] | fixed (WP-12) |
| [B-080](#b-080) | `point_cloud_split` leaks its work buffer on every failed split | D geometry, math | 1 defect | S3 | [R] | fixed (WP-01) |
| [B-081](#b-081) | `obj_overlap_test` docs call `obj_ind` 1-based and "the output of `obj_file_read`", which is 0-based; IDs are returned as given | D geometry, math | 2 doc | S3 | [R] | fixed (WP-12) |
| [B-082](#b-082) | `icosphere(radius = 0)` returns NaN; `cube` / `plane` with a negative scale return inward / -z normals, contrary to the docs | D geometry, math | 1 defect | S3 | [R] | fixed (WP-12) |
| [B-083](#b-083) | `interp` 1-D/2-D index search is O(n·m): 4 s for 160 000 samples | D geometry, math | 4 performance | S3 | [M] | fixed (WP-13) |
| [B-084](#b-084) | `obj_overlap_test`: reason strings are non-deterministic (unseeded shuffle, thread count), and a flag array is read outside its critical section | D geometry, math | 1 defect | S3 | [R, C] | fixed (WP-12) |
| [B-085](#b-085) | MEX `fast_sincos(x, [])` returns cos instead of sin; the single-precision output is undocumented | F MEX layer | 1 defect | S3 | [R] | fixed (WP-01) |
| [B-086](#b-086) | MEX `arrayant_qdant_write` (separate inputs) writes `center_freq = NaN` for `[]` instead of the documented 299792458 | F MEX layer | 1 defect | S3 | [R] | fixed (WP-01) |
| [B-087](#b-087) | `arrayant_generate('multibeam')` uses a 90° element beamwidth by default in MATLAB and Python; docs (and the C++ default) say 120° | F MEX layer | 3 cross-API | S3 | [R] | fixed (WP-15) |
| [B-088](#b-088) | `calc_angular_spread` defaults differ: MATLAB and Python disable wrapping and the bank angle, C++ enables both | F MEX layer | 3 cross-API | S3 | [R] | fixed (WP-15) |
| [B-089](#b-089) | `triangle_mesh_segmentation` pads `mtl_ind_out` with material 1 in MATLAB, with 0 ("no material") in C++ and Python | F MEX layer | 3 cross-API | S3 | [R] | fixed (WP-15) |
| [B-090](#b-090) | MEX `baseband_freq_response` with `carrier_freq`: descending carriers are rejected, duplicate carriers give silent NaN (Python handles both) | F MEX layer | 3 cross-API | S3 | [R] | fixed (WP-15) |
| [B-091](#b-091) | MEX `get_channels_irs` `i_irs` is 0-based, against the 1-based rule of the MATLAB spec; the doc does not say which | F MEX layer | 2 doc | S3 | [R] | fixed (WP-15) |
| [B-092](#b-092) | `arrayant_qdant_read(fn)` returns every entry as a struct array in MATLAB, Python's default reads ID 1 | F MEX layer | 3 cross-API | S3 | [R] | fixed (WP-15) |
| [B-093](#b-093) | MEX: arrays with more than 4 dimensions are silently truncated; a 4-D dataset written with `hdf5_write_dset` comes back 3-D | F MEX layer | 1 defect | S3 | [R] | fixed (WP-06) |
| [B-094](#b-094) | MEX: index value 0 is not rejected in several wrappers (silent wrong result or a misleading error) | F MEX layer | 1 defect | S3 | [R] | fixed (WP-06) |
| [B-095](#b-095) | Diffraction level of detail 7: accepted by C++, empty `coord` in MATLAB, internal error in Python, rejected by MATLAB `generate_diffraction_paths`, documented as 0-6 | F MEX layer | 3 cross-API | S3 | [R] | fixed (WP-15) |
| [B-096](#b-096) | Python `qrt_file_read`: an exception in the main loop while the reader thread runs calls `std::terminate` | G Python layer | 1 defect | S3 | [R] | fixed (WP-05) |
| [B-097](#b-097) | Python HDF5 writers reject numpy scalars, 0-d arrays and non-native dtypes that the docs list as supported | G Python layer | 3 cross-API | S3 | [R] | fixed (WP-05) |
| [B-098](#b-098) | Python adapter `view = true` always copies (confirms item 1.1 of `md/python_fix_plan_v00.md`) | G Python layer | 2 doc | S3 | [R] | fixed (WP-05) |
| [B-099](#b-099) | Python: documented kwargs `sub_cloud_index` and `map_to_itu_materials` raise `TypeError` | G Python layer | 2 doc | S3 | [R] | documented (WP-05) |
| [B-100](#b-100) | Python: invalid input accepted silently (2-D positions, short `interact_coord`, 4-D arrayant without `center_freq`) | G Python layer | 1 defect | S3 | [R] | fixed (WP-05) |
| [B-101](#b-101) | `arrayant::copy_element` ignores `read_only`; an empty destination gives an Armadillo-internal error | A arrayant | 1 defect | S4 | [R] | fixed (WP-09) |
| [B-102](#b-102) | Arrayant documentation typos and misleading statements; latent int32 offsets and a vertex sentinel (grouped) | A arrayant | 2 doc | S4 | [C] | open |
| [B-103](#b-103) | `quantize_delays` docs: `fix_taps` 1 and 2 give `[1, 1, n_taps]` delays only for shared input | B channel, file I/O | 2 doc | S4 | [R] | fixed (WP-03) |
| [B-104](#b-104) | Channel and file-I/O documentation and message nits (grouped) | B channel, file I/O | 2 doc | S4 | [R, C] | partly fixed (WP-14) |
| [B-105](#b-105) | `memcpy` with a null pointer and size 0 (formal UB) in `get_channels_irs` and in the Python list adapter | C channel gen., tools | 1 defect | S4 | [R] | fixed (WP-01) |
| [B-106](#b-106) | Docs and code of `path_to_tube`, `qd_path` and `calc_diffraction_gain` debug output disagree (grouped) | C channel gen., tools | 2 doc | S4 | [R, C] | open |
| [B-107](#b-107) | `interp`: a NaN query returns the first data value; clamped extrapolation is undocumented | D geometry, math | 2 doc | S4 | [R] | fixed (WP-13) |
| [B-108](#b-108) | Geometry documentation and validation nits (index-map directions, 9-column check, precision claim, header typos; dead code; CPU check) | D geometry, math | 2 doc | S4 | [R, C] | partly fixed (WP-12) |
| [B-109](#b-109) | `sos_val` / `sos_remove_bias`: an output that partially overlaps an input is accepted and gives wrong values (docs promise an error) | E SOS | 2 doc | S4 | [R] | fixed (WP-16) |
| [B-110](#b-110) | `sos_randn` / `sos_rand` / `sos_randi` report an unknown preset as "Input 'name' ...", an argument they do not have | E SOS | 2 doc | S4 | [R] | fixed (WP-16) |
| [B-111](#b-111) | `md/SPEC_PYTHON_API.md` gives two contradicting rules for the headings of the `sos` entries | E SOS | 2 doc | S4 | [C] | fixed (WP-16) |
| [B-112](#b-112) | MEX `qrt_file_read` returns `interact_type` as uint8; the doc says uint32 | F MEX layer | 2 doc | S4 | [R] | fixed (WP-01) |
| [B-113](#b-113) | MEX `get_channels_*` with 0 paths and `add_fake_los_path` return a zero path with delay 0, where C++ throws | F MEX layer | 3 cross-API | S4 | [R] | fixed (WP-15) |
| [B-114](#b-114) | MEX `arrayant_rotate_pattern`: struct arrays reject usage 3 and 4 (Python maps them); a call without output argument errors | F MEX layer | 3 cross-API | S4 | [R] | fixed (WP-15) |
| [B-115](#b-115) | MEX documentation, error IDs and latent helper defects (grouped) | F MEX layer | 2 doc | S4 | [R, C] | open |
| [B-116](#b-116) | Python `hdf5_create_file` returns uint32 although the wrapper builds int64 (confirms item 1.3 of the Python fix plan) | G Python layer | 2 doc | S4 | [R] | fixed (WP-05) |
| [B-117](#b-117) | Python output dtypes and shapes differ from the docs (`qdant_read`, `point_inside_mesh`, `generate_speaker`, `get_ieee_indoor`) | G Python layer | 2 doc | S4 | [R] | documented (WP-05) |
| [B-118](#b-118) | Python single-frequency arrayant dict: a 1-element `center_freq` array gives a numpy DeprecationWarning, 2 elements an unclear error | G Python layer | 3 cross-API | S4 | [R] | fixed (WP-05) |
| [B-119](#b-119) | Crafted HDF5 files make HDF5 1.14.6 itself zero-fill about 4 GB in `H5Fopen` or request a 2^64-byte allocation in `H5Gopen2` (upstream, not quadriga-lib code) | B channel, file I/O | 1 defect | S4 | [R] | won't fix (upstream HDF5) |
| [B-120](#b-120) | `calc_cross_polarization_ratio`: the circular XPR columns report 0 (fully cross-polar) for the LOS matrix that `get_channels_*` treats as co-polar, and are defined per transmit polarisation while the linear columns are per receive polarisation | C channel gen., tools | 1 defect | S3 | [R] | fixed (WP-13) |
| [B-121](#b-121) | Test defect: `test_bin` is compiled without `BUILD_WITH_AVX2`, so the AVX2-vs-generic cases of the ray-triangle and ray-point kernel tests never run | D geometry, math | 1 defect | S3 | [R] | fixed (WP-12) |
| [B-122](#b-122) | `qrt_file_read_raw` is quadratic in the number of paths per origin (`reserve` per CIR): 240 000 paths take 7.5 s | B channel, file I/O | 4 performance | S3 | [M] | fixed (WP-14) |
| [B-123](#b-123) | MEX and Python `qrt_file_read` build a 2^32-entry index vector (34 GB) for a valid file with 0 CIRs | F MEX layer | 1 defect | S3 | [R] | fixed (WP-15) |

## Baseline

Machine: Ryzen 9 5950X, 32 threads, shared with other sessions. Tools: g++ 12.5.0 (library, suite, MEX,
Python), g++ 15.2.0 (second sanitizer build), afl-clang-fast / clang 21.1.8 with AFL++ 4.33c, cmake 4.2.3,
system Armadillo 15.2.1 (`/usr/include`), system HDF5 1.14.6 (serial), Octave 11.1.0, MATLAB R2026a
(26.1.0.3203278) and R2024a, Python 3.14.4 with numpy 2.3.5 and pytest 9.0.2.

All build directories are inside the worktree and gitignored (`build_*/*`). Logs and outputs are in
`build_linux/scratch/hunt/`.

### Builds

| Config | Directory | Command (from the worktree root) | Result |
|---|---|---|---|
| base: gcc-12, AVX2, Catch2, MATLAB + Octave MEX, Python | `build_linux` (installed to `lib/`, `+quadriga_lib/`) | `cmake -B build_linux -D CMAKE_INSTALL_PREFIX=. -D CMAKE_C_COMPILER=gcc-12 -D CMAKE_CXX_COMPILER=g++-12 -D ENABLE_MATLAB=ON -D ENABLE_OCTAVE=ON -D ENABLE_MEX_DOC=ON -D ENABLE_PYTHON=ON -D ENABLE_TESTS=ON -D ENABLE_AVX2=ON -D ENABLE_CUDA=OFF -D ENABLE_STATIC_LIB=ON -D ENABLE_SHARED_LIB=OFF && cmake --build build_linux --parallel 32 && cmake --install build_linux` | 1 min 38 s, 0 warnings |
| generic: gcc-12, no AVX2, Catch2 | `build_generic` | `cmake -B build_generic -D CMAKE_C_COMPILER=gcc-12 -D CMAKE_CXX_COMPILER=g++-12 -D ENABLE_MATLAB=OFF -D ENABLE_OCTAVE=OFF -D ENABLE_MEX_DOC=OFF -D ENABLE_PYTHON=OFF -D ENABLE_TESTS=ON -D ENABLE_CUDA=OFF -D ENABLE_STATIC_LIB=ON -D ENABLE_SHARED_LIB=OFF -D ENABLE_AVX2=OFF && cmake --build build_generic --parallel 16` | 1 min 52 s, 0 warnings |
| ASan + UBSan, gcc-12, AVX2 | `build_asan` | as generic, with `-D ENABLE_AVX2=ON -D "CMAKE_CXX_FLAGS=$SAN" -D "CMAKE_C_FLAGS=$SAN" -D "CMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined"`, where `SAN="-fsanitize=address,undefined -fsanitize-recover=address -fno-omit-frame-pointer -g"` | 8 min 48 s, 0 warnings |
| ASan + UBSan, gcc-15, AVX2 | `build_asan15` | as `build_asan` with `gcc-15` / `g++-15` | 4 min 42 s, 0 warnings |
| AFL, ASan, no AVX2, no OpenMP | `build_afl_plain` | `AFL_USE_ASAN=1 cmake -B build_afl_plain -D CMAKE_C_COMPILER=afl-clang-fast -D CMAKE_CXX_COMPILER=afl-clang-fast++ -D ENABLE_MATLAB=OFF -D ENABLE_OCTAVE=OFF -D ENABLE_MEX_DOC=OFF -D ENABLE_PYTHON=OFF -D ENABLE_TESTS=OFF -D ENABLE_AVX2=OFF -D ENABLE_CUDA=OFF && AFL_USE_ASAN=1 cmake --build build_afl_plain --parallel 8` | 1 min 20 s, 0 warnings. The whole library compiles with clang 21 |
| AFL CMPLOG, no AVX2 | `build_afl_cmplog` | as `build_afl_plain` with `AFL_LLVM_CMPLOG=1` instead of `AFL_USE_ASAN=1` | 1 min 1 s |
| AFL with LAF (discarded) | `build_afl` | as `build_afl_plain` with `AFL_LLVM_LAF_ALL=1` added | built, but not usable: see the note below |

Note on LAF: with `AFL_LLVM_LAF_ALL=1` every harness that calls `obj_file_read` crashed on the valid
seed `tests/benchmarks_mex/cube.obj` (SEGV in `arma::Mat::init_warm` called from `obj_file_read.cpp:580`).
The same harness with `src/obj_file_read.cpp` compiled by plain `clang++ -O0`, `-O2` and `-O3` with ASan,
and by `afl-clang-fast++` without LAF, runs clean; with LAF it crashes again [R]. This is a
miscompilation by the LAF split-compare passes, not a library defect. The campaign used CMPLOG instead.

### Catch2 suite

| Build | Result | Wall time | Load (1 min) before / after |
|---|---|---|---|
| base (`build_linux/test_bin`) | `All tests passed (1307605 assertions in 1132 test cases)` | 44.5 s | 18.3 / 12.4 (the base build had just finished) |
| generic (`build_generic/test_bin`) | `All tests passed (1307356 assertions in 1132 test cases)` | 245 s | 3.2 / 31.7 (the AFL build ran at the same time) |
| ASan + UBSan gcc-12 | `All tests passed (1307605 assertions in 1132 test cases)`, sanitizer reports below | 346 s | 24.8 / 2.3 |
| ASan + UBSan gcc-15 | `All tests passed (1307605 assertions in 1132 test cases)`, same reports | 203 s | 5.3 / 6.8 |
| after WP-01: base | `All tests passed (1307644 assertions in 1137 test cases)` | 49 s | 1.6 / 6.5 |
| after WP-01: generic | `All tests passed (1307395 assertions in 1137 test cases)` | 39 s | 6.5 / 8.3 |
| after WP-01: ASan + UBSan gcc-12, `suppress_equal_pcs=0` | `All tests passed (1307644 assertions in 1137 test cases)`; reports: B-015 (`test_hdf_functions.cpp:114`, `:350`) and the excluded `ray_progress` UBSan reports (`test_wave_progress.cpp:301`), nothing else, no leak | 323 s | 7.1 / 5.8 |
| after WP-02 round 1: base | `All tests passed (1307922 assertions in 1144 test cases)` | 40 s | 1.7 / 1.4 |
| after WP-02 round 1: generic | `All tests passed (1307673 assertions in 1144 test cases)` | 46 s | 1.4 / 6.0 |
| after WP-02 round 1: ASan + UBSan gcc-12, `suppress_equal_pcs=0` | `All tests passed (1307922 assertions in 1144 test cases)`; reports: only the excluded `ray_progress` UBSan reports (`test_wave_progress.cpp:301`), no ASan report, no leak | 317 s | 5.1 / 2.5 |
| after WP-02 round 2: base / generic | `All tests passed (1307922 assertions in 1144 test cases)` / `(1307673 assertions in 1144 test cases)` | 92 s / 40 s | 3.0 / 18.3, 18.3 / 9.9 |
| after WP-02 round 2: ASan + UBSan gcc-12, `suppress_equal_pcs=0` | `All tests passed (1307922 assertions in 1144 test cases)`; only the excluded `ray_progress` reports, no leak | 322 s | 8.5 / 2.6 |
| after WP-03: base | `All tests passed (1332501 assertions in 1151 test cases)` (round 2; round 1: 1332491 / 1150) | 148 s (8 threads) | 26.1 / 31.8 |
| after WP-03 round 1: generic | `All tests passed (1332242 assertions in 1150 test cases)` (not re-run for round 2, which changed one message and documentation) | 40 s | 3.5 / 4.5 |
| after WP-03 round 1: ASan + UBSan gcc-12, `suppress_equal_pcs=0` | `All tests passed (1332491 assertions in 1150 test cases)`; only the excluded `ray_progress` reports, no leak (not re-run for round 2) | 317 s | 3.8 / 1.6 |
| after WP-04: base | `All tests passed (1332667 assertions in 1162 test cases)` | 113 s (12 threads) | 13.3 / 36.3 |
| after WP-04: generic | `All tests passed (1332418 assertions in 1162 test cases)` | 88 s (12 threads) | 36.3 / 25.1 |
| after WP-04: ASan + UBSan gcc-12, `suppress_equal_pcs=0` | `All tests passed (1332667 assertions in 1162 test cases)`; only the excluded `ray_progress` reports, no leak | 428 s (8 threads) | 21.4 / 1.5 |
| after WP-07: base / generic | `All tests passed (1333935 assertions in 1169 test cases)` / `(1333686 assertions in 1169 test cases)` | 38 s / 37 s (12 threads) | 5.3 / 4.1 |
| after WP-07: ASan + UBSan gcc-12, `suppress_equal_pcs=0` | `All tests passed (1333935 assertions in 1169 test cases)`; only the excluded `ray_progress` reports, no leak | 550 s (12 threads) | 3.5 / 35.7 |
| after WP-08: base | `All tests passed (1334516 assertions in 1184 test cases)` (manager) | 38 s (12 threads) | 1.7 / 2.6 |
| after WP-08: generic, ASan + UBSan | generic `1334185 / 1184`; ASan + UBSan `1334516 / 1184`, only the excluded `ray_progress` reports (worker's runs; Octave MEX 65 of 66, pytest 792 passed, 1 skipped) | - | - |
| after WP-09: base | `All tests passed (1334881 assertions in 1195 test cases)` (manager) | 37 s (12 threads) | 1.3 / 2.7 |
| after WP-09: generic, ASan + UBSan | generic `1334550 / 1195`; ASan + UBSan all passed, only the excluded `ray_progress` reports (worker's runs; Octave MEX 65 of 66, pytest 792 passed, 1 skipped) | - | - |
| after WP-10: base | `All tests passed (1334981 assertions in 1203 test cases)` (manager, rounds 2 and 3; round 1: 1334988) | 38 s (12 threads) | 0.6 / 2.4 |
| after WP-10: generic, ASan + UBSan | generic `1334650 / 1203` (round 2); ASan + UBSan of round 1 (the reader did not change in round 2); ASan + UBSan all passed, only the excluded `ray_progress` reports (worker's runs; Octave MEX 65 of 66, pytest 793 passed, 1 skipped; qdant AFL 15 min: no crash, 1637 inputs replayed, slowest 0.035 s) | - | - |
| after WP-11: base | `All tests passed (1335771 assertions in 1222 test cases)` (manager); publication tests 14 cases in 1.2 s | 39 s (12 threads) | 1.2 / 2.7 |
| after WP-11: generic, ASan + UBSan | generic `1335440 / 1222`; ASan + UBSan all passed, only the excluded `ray_progress` reports (worker's runs; Octave MEX 65 of 66, pytest 793 passed, 1 skipped) | - | - |
| after the merge with main (2026-10-20, 71b894b) | `All tests passed (1346299 assertions in 1238 test cases)` (manager; includes the acoustic session's tests) | 39 s (12 threads) | 10.2 / 7.2 |
| after WP-12: base | `All tests passed (1347918 assertions in 1255 test cases)` (manager; generic `1346136 / 1254`, also manager) | 40 s (12 threads) | 3.9 / 4.5 |
| after WP-12: ASan + UBSan, gate | ASan + UBSan all passed, only the excluded `ray_progress` reports; `[wave]` 165594 / 181 and `[acoustic]` 42741 / 60 in both builds (worker's runs; Octave MEX 66 of 67, pytest 797 passed, 1 skipped) | - | - |
| after WP-13: base | `All tests passed (1348300 assertions in 1269 test cases)` (manager; generic `1346518 / 1268`, also manager) | 40 s (12 threads) | 1.3 / 2.8 |
| after WP-13: ASan + UBSan, gate | ASan + UBSan all passed, only the excluded `ray_progress` reports; `[wave]` 165594 / 181 and `[acoustic]` 42741 / 60 in both builds (worker's runs; Octave MEX 66 of 67, pytest 799 passed, 1 skipped) | - | - |
| after the merge with main (2026-10-21, f0d093d) | `All tests passed (1346180 assertions in 1265 test cases)`, generic `1344398 / 1264`; `[wave]` 163474 / 178, `[acoustic]` 42741 / 59 in both builds (manager, temporary worktree; includes the acoustic session's test cleanup; Octave MEX 66 of 67, pytest 799 passed, 1 skipped) | - (12 threads) | 2.2 / 6.2 |
| after WP-14: base | `All tests passed (1346416 assertions in 1273 test cases)` (manager, round 2; round 1: 1346411 / 1272) | 40 s (12 threads) | 3.3 / 2.6 |
| after WP-14: generic, ASan + UBSan, gate | generic `1344629 / 1271`; ASan + UBSan all passed, only the excluded `ray_progress` reports; `[wave]` 163474 / 178 and `[acoustic]` 42741 / 59 in both builds (worker's runs; Octave MEX 66 of 67, pytest 801 passed, 1 skipped; OBJ / MTL / CSV AFL 15 min each: no crash, no hang, 2613 inputs replayed, slowest 0.034 s) | - | - |
| after WP-15: base | `All tests passed (1346724 assertions in 1283 test cases)` (manager, round 2; round 1: 1346567 / 1278) | 40 s (12 threads) | 1.7 / 3.4 |
| after WP-15: generic, ASan + UBSan, gate | generic `1344942 / 1282`; ASan + UBSan all passed, only the excluded `ray_progress` reports; `[wave]` 163474 / 178 and `[acoustic]` 42741 / 59 in both builds (worker's runs, round 2; Octave MEX and MATLAB R2026a 66 of 67, pytest 817 passed) | - | - |
| after WP-16: base | `All tests passed (1346786 assertions in 1285 test cases)` (manager) | 40 s (12 threads) | 0.7 / 2.0 |
| after WP-16: generic, ASan + UBSan | generic `1345004 / 1284`; ASan + UBSan all passed, only the excluded `ray_progress` reports (worker's runs; Octave MEX 66 of 67, pytest 817 passed, 1 skipped; SOS hashes of valid calls identical old vs new at 1 and 12 threads, 90 AVX2 / 77 generic) | - | - |

Run commands: `flock /tmp/quadriga-fixes-suite.lock build_linux/test_bin` (output
`build_linux/scratch/hunt/base_test.txt`); the sanitizer runs with
`ASAN_OPTIONS=halt_on_error=0:detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0`.
The suite passes at HEAD in every build.

Sanitizer reports (identical in the gcc-12 and gcc-15 builds; leak check on, no leak reported):

1. `heap-buffer-overflow` READ of 256 bytes in `H5Awrite`, called from `quadriga_lib::hdf5_write<double>`
   for the channel name; the 50-byte `std::string` allocated at `test_hdf_functions.cpp:27` is read 206
   bytes past its end (case "HDF - Minimal Test", `test_hdf_functions.cpp:114`): B-015.
2. UBSan "null pointer passed as argument 1, which is declared to never be null" at
   `src/get_channels_irs.cpp:342, 343, 344, 349, 355, 361, 367` (`test_get_channels_irs.cpp:154`): B-105.
3. UBSan "reference binding to null pointer" in `arma::Mat<float>::colptr` from `quadriga_lib::ray_progress`
   (`test_wave_progress.cpp:301`): excluded code, see the appendix.

Correction found in phase 2: ASan's default `suppress_equal_pcs=1` drops a report at the same program counter as an
earlier one. Two `memcpy` over-reads are caught in the same ASan interceptor, so the IEEE test over-read was hidden
behind the HDF5 one. The suite was re-run (gcc-12 build) with
`ASAN_OPTIONS=halt_on_error=0:detect_leaks=1:suppress_equal_pcs=0`: `All tests passed (1307605 assertions in 1132 test
cases)`, 691 s, load 2.8 before / 4.2 after. Reports: the HDF5 writer over-read at two call sites
(`test_hdf_functions.cpp:114` and `:350`, B-015), the over-read in the test itself at `test_qd_ieee_chan.cpp:318`
(B-063), and the same UBSan reports as above; nothing else.

The conf-file review (`md/REVIEW_CONF_FILE.md`, section 0) had also seen an `aligned_alloc` report in
`baseband_freq_response` and five reports in "IEEE Chan" under gcc 15; neither build reports them now,
also not when the baseband cases run alone with `halt_on_error=1`. The `aligned_alloc` report needs the generic build (B-062); the IEEE report is B-063.

### MEX suite (`tests/quadriga_lib_mex_tests.m`)

Command: `cd tests && OMP_NUM_THREADS=8 octave --no-gui --eval quadriga_lib_mex_tests` (Octave), and
`OMP_NUM_THREADS=8 matlab -batch quadriga_lib_mex_tests` / `matlab24 -batch ...`. MOxUnit was unpacked
with `unzip -q external/MOxUnit.zip -d external/` (gitignored). MEX files: gcc-12, AVX2 (note: the
GNUmakefile default is `avx2 = OFF`, so a default `make` gives MEX files without AVX2).

| Runtime | Result | Wall | Load before |
|---|---|---|---|
| Octave 11.1.0 | `FAILED (passed=65, failure=1)`: `test_ray_mesh_interact:113`, 0.98485 vs 0.69639 (excluded code) | 1.6 s | 21.5 |
| MATLAB R2024a | same, 65 of 66, same failure | 6.4 s | 17.6 |
| MATLAB R2026a | same, 65 of 66, then **MATLAB crashes with a segmentation violation at exit**, exit code 137, crash dump `~/matlab_crash_dump.878185-1` | 10.9 s | 21.5 |
| after WP-01: Octave 11.1, R2024a, R2026a | 65 of 66 in each, same excluded failure; R2026a exits with 0, no crash dump (B-029 fixed) | - | 2 |
| after WP-02 round 1: Octave 11.1, R2024a, R2026a | 65 of 66 in each, same excluded failure; R2026a exits with 0, no crash dump | - | 1-2 |
| after WP-02 round 2: Octave 11.1, R2026a | 65 of 66 in each, same excluded failure; R2026a exits with 0, no crash dump | - | 2-3 |
| after WP-03 round 1: Octave 11.1, R2026a | 65 of 66 in each, same excluded failure; R2026a exits with 0, no crash dump (round 2: only `test_quantize_delays` in Octave, OK) | - | 2-4 |
| after WP-04: Octave 11.1, R2026a | 65 of 66 in each, same excluded failure; R2026a exits with 0, no crash dump | - | 15-36 |
| after WP-06: Octave 11.1, R2024a, R2026a | 65 of 66 in each, same excluded failure; R2026a exits with 0, no crash dump. Malformed-input sweep (77 wrappers x 16 argument lists x nout 0/1/3, Octave): no crash. WP-06 changed only the MEX layer; Catch2 and pytest were not re-run | - | 12 |

The R2026a crash is B-029. A single `quadriga_lib.version` call does not crash R2026a at
exit (2 of 2 runs).

### Python suite

`PYTHONPATH=/sjc/quadriga-lib-fixes/lib OMP_NUM_THREADS=8 python3 -m pytest tests/python_tests -q -p no:cacheprovider`:
`762 passed, 1 skipped, 37 subtests passed in 2.55s` (load 9.5 before, 11.7 after). The skip is
`test_ray_point_intersect.py:403` (CUDA kernel not available). Module version string `0.12.2_AVX2`.

After WP-01: `762 passed, 1 skipped, 37 subtests passed in 1.85s`. After WP-02 round 1: `762 passed, 1 skipped, 37 subtests passed in 1.80s` (the Python suite has no HDF5 tests). After WP-02 round 2: the same. After WP-03 round 1: `763 passed, 1 skipped, 37 subtests passed` (round 2: `test_quantize_delays.py`, 23 passed). After WP-04: `764 passed, 1 skipped, 37 subtests passed`. After WP-05: `789 passed, 1 skipped, 37 subtests passed` (25 new cases, new file `test_hdf5_functions.py`); against the ASan + UBSan module (`build_linux/scratch/fix/build_asan_mod.sh`): `788 passed, 2 skipped`, no report. WP-05 changed only `api_python/` and `tests/python_tests/`; Catch2 and MEX were not re-run.

### Fuzzing (AFL++ 4.33c)

Harnesses in `build_linux/scratch/hunt/fuzz/fz_<reader>.cpp`, built with
`AFL_USE_ASAN=1 afl-clang-fast++ -std=c++17 -O2 -g -I include -I src fz_X.cpp build_afl_plain/libquadriga.a $(pkg-config --cflags --libs hdf5)`
and a CMPLOG twin linked against `build_afl_cmplog`. Every harness calls `prctl(PR_SET_DUMPABLE, 0)`
first, copies the input to a file with the extension the reader requires, and catches
`std::exception`. Per reader one main instance (ASan) and one CMPLOG secondary, 900 s each, all 14
instances at once, with `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 AFL_SKIP_CPUFREQ=1 AFL_NO_UI=1`,
`-m none -t 2000`. Load 5.8 before, 20.8 after (other sessions started meanwhile). There is no PNG
reader in the library (only `write_png`), so no PNG harness.

| Reader (harness) | What the harness calls | Seeds | Executions (both instances) | Edges found (main) | Crashes | Hangs |
|---|---|---|---|---|---|---|
| qdant XML (`fz_qdant`) | `qdant_read<double>(fn,1,&layout)`, `qdant_read_multi<float>`, `qdant_read<float>(fn,2)` | `tests/afl/qdant_samples/minimal.qdant`, two files written by the library | 476,507 | 2493 | 0 | 0 |
| OBJ (`fz_obj`) | `obj_file_read<double>` with all mesh, vertex, face, object and material outputs | `tests/benchmarks_mex/cube.obj`, `tests/data/test_scene.obj` | 950,154 | 511 | 0 | 0 |
| MTL (`fz_mtl`) | `obj_file_read<double>` of a fixed OBJ whose `mtllib` is the fuzzed file | `tests/data/test_scene1.mtl`, first 3000 bytes of `test_scene_pbr.mtl` | 873,548 | 575 | 0 | 0 |
| material CSV (`fz_csv`) | `obj_file_read` with `fn_csv` = the fuzzed file, `csv_strict` false and true | first 6 lines of `data/materials_rf.csv` and `materials_ac.csv` | 857,836 | 712 | 0 | 0 |
| conf (`fz_conf`) | `conf_file_read` with all outputs, `validate` false and true | `data/config/3GPP_38.901_v19_UMi_LOS.conf`, a 40-line excerpt of `UMa_NLOS`, a 5-line file | 709,932 | 936 | 0 | 0 |
| QRT (`fz_qrt`) | `qrt_file_parse`, then `qrt_file_read<double>` with all outputs for up to 2 origins × 4 CIRs × both directions, `qrt_file_read_raw` | `tests/data/test_v5.qrt`, a 131-byte file written by the library | 5,989 | 1064 | 0 | 28 |
| HDF5 (`fz_hdf5`) | `hdf5_read_layout`, `hdf5_read_channel<double>` and `hdf5_read_dset_names` + `hdf5_read_dset` for 3 indices | a 10 kB file written by the library (one channel, two datasets) | 38,980 | 1001 | 11 | 20 |

Edge counts are of the whole instrumented binary (the map includes the unused parts of the library), so
they compare runs, not files. The QRT throughput is low because inputs make the reader allocate and
clear very large arrays (see the item). The HDF5 crashes all reduce to heap-buffer-overflow writes by
`H5Dread` into buffers that `hdf5_read_channel` sized from attributes (`qd_channel_hdf5.cpp:1838/1839`
and `1880/1881`); the HDF5 library itself is not instrumented and was not the target. HDF5: B-013; QRT: B-048.

Note after WP-14: with the WP-14 OBJ reader, the `fz_mtl` and `fz_csv` harnesses built at `-O2` crash on every valid
seed in AFL's own coverage-counter increment (a garbage guard index, inside `arma::Mat::init_warm` called from
`obj_file_read`, the site of the LAF note above); gcc ASan + UBSan, plain clang -O3 ASan and clang with
`trace-pc-guard` are clean. Build the harness TUs with `-O1` (the library keeps `-O3`); evidence in
`build_linux/scratch/fix/WP-14/afl/dbg/`.

## Coverage

One row per in-scope file: who looked at it (area), whether it was reviewed by reading, probed by a program, and run
under a sanitizer, and the items that name the file. "Suite" in the sanitizer column means the file is part of the
ASan + UBSan Catch2 suite runs of the Baseline section (whether a given function is exercised depends on the tests);
"probe" means at least one dedicated probe ran under ASan + UBSan; "fuzz" means the AFL campaign with an ASan harness.
"ASan module" means the Python bindings compiled by area G with ASan + UBSan against `build_asan/libquadriga.a`
(probes and the Python suite without the SOS tests: 659 passed, only the reports of B-004).

What was not covered, or only partly:

- Excluded by the brief: the ray-interaction machinery (`ray_mesh_interact`, `ray_progress`, `ray_commit`, `ray_init`,
  `ray_subdivide_flag`, `subdivide_rays`, the material helpers and physics) and all CUDA code and paths.
- No ThreadSanitizer, MemorySanitizer or valgrind (not installed / would need a rebuild of the library): races were
  found by comparing thread counts (B-024) or by reading (B-084); uninitialised reads by their effects.
- MEX files were not sanitizer-built, except one what-if (`calc_rotation_matrix` for Octave, B-030). Most MEX
  checks ran in Octave 11.1; MATLAB R2024a and R2026a only where an item says so.
- The MEX and Python modules exist only in the AVX2 build; the generic kernels were compared through the C++
  libraries.
- Fuzzing: 15 minutes per reader; the HDF5 library itself is not instrumented; there is no PNG reader. The qdant,
  OBJ, MTL, material-CSV and conf readers had no crash.
- Not probed (read only, or not at all): the MEX wrappers marked "sweep only" below; `get_channels_irs` second IRS
  array and angle outputs; numeric edge cases of `calc_rician_k_factor` and `calc_cross_polarization_ratio`; the
  optimiser maths of `sos_generate` and the SOS preset tables; the sub-sampled sphere branch of `rotate_pattern`
  (step below 0.1°); `generate_*` with NaN resolution or angles; the Mitsuba writer under ASan.
- Platforms: Linux / gcc 12 and 15 / clang 21 only; MSVC, macOS and ARM were not checked.
- Timings were taken on a shared machine; numbers marked contaminated or provisional are not baselines.

| File | Area | Reviewed by reading | Probed | Sanitizer | IDs |
|---|---|---|---|---|---|
| `api_mex/acdf.cpp` | F | yes | yes | no | B-030 |
| `api_mex/arrayant_calc_beamwidth.cpp` | F | yes | yes | no | B-030, B-115 |
| `api_mex/arrayant_calc_directivity.cpp` | F | yes | yes | no | B-030, B-115 |
| `api_mex/arrayant_combine_pattern.cpp` | F | yes | yes | no | B-030 |
| `api_mex/arrayant_concat.cpp` | F | yes | malformed-input sweep only | no | B-030 |
| `api_mex/arrayant_copy_element.cpp` | F | yes | malformed-input sweep only | no | B-030 |
| `api_mex/arrayant_export_obj_file.cpp` | F | yes | malformed-input sweep only | no | B-030, B-115 |
| `api_mex/arrayant_generate.cpp` | F | yes | yes | no | B-030, B-047, B-087 |
| `api_mex/arrayant_interpolate.cpp` | F | yes | yes | no | B-030, B-094 |
| `api_mex/arrayant_qdant_read.cpp` | F | yes | yes | no | B-030, B-092 |
| `api_mex/arrayant_qdant_write.cpp` | F | yes | yes | no | B-030, B-086, B-115 |
| `api_mex/arrayant_rotate_pattern.cpp` | F | yes | yes | no | B-030, B-114 |
| `api_mex/baseband_freq_response.cpp` | F | yes | yes | no | B-030, B-090 |
| `api_mex/calc_angular_spread.cpp` | F | yes | yes | no | B-030, B-088 |
| `api_mex/calc_cross_polarization_ratio.cpp` | F | yes | yes | no | B-030 |
| `api_mex/calc_delay_spread.cpp` | F | yes | yes | no | B-030 |
| `api_mex/calc_diffraction_gain.cpp` | F | yes | yes | no | B-030, B-095 |
| `api_mex/calc_rician_k_factor.cpp` | F | yes | yes | no | B-030 |
| `api_mex/calc_rotation_matrix.cpp` | F | yes | yes | ASan what-if | B-030 |
| `api_mex/cart2geo.cpp` | F | yes | yes | no | B-030 |
| `api_mex/channel_export_obj_file.cpp` | F | yes | yes | no | B-030, B-032, B-094 |
| `api_mex/conf_file_list.cpp` | B | light pass (reviewed twice before) | no | no | B-030 |
| `api_mex/conf_file_read.cpp` | B | light pass (reviewed twice before) | no | no | B-030 |
| `api_mex/conf_file_write.cpp` | B | light pass (reviewed twice before) | no | no | B-030 |
| `api_mex/cube.cpp` | F | yes | yes | no | B-030 |
| `api_mex/fast_sincos.cpp` | F | yes | yes | no | B-030, B-085 |
| `api_mex/generate_diffraction_paths.cpp` | F | yes | malformed-input sweep only | no | B-030, B-095 |
| `api_mex/generate_speaker.cpp` | F | yes | yes | no | B-030 |
| `api_mex/geo2cart.cpp` | F | yes | malformed-input sweep only | no | B-030 |
| `api_mex/get_channels_ieee_indoor.cpp` | F | yes | yes | no | B-030 |
| `api_mex/get_channels_irs.cpp` | F | yes | yes | no | B-030, B-091 |
| `api_mex/get_channels_multifreq.cpp` | F | yes | malformed-input sweep only | no | B-030, B-113 |
| `api_mex/get_channels_planar.cpp` | F | yes | yes | no | B-030, B-113 |
| `api_mex/get_channels_spherical.cpp` | F | yes | yes | no | B-006, B-030, B-113 |
| `api_mex/hdf5_create_file.cpp` | F | yes | yes | no | B-030 |
| `api_mex/hdf5_read_channel.cpp` | F | yes | yes | no | B-030, B-031 |
| `api_mex/hdf5_read_dset.cpp` | F | yes | yes | no | B-030 |
| `api_mex/hdf5_read_dset_names.cpp` | F | yes | malformed-input sweep only | no | B-030 |
| `api_mex/hdf5_read_layout.cpp` | F | yes | malformed-input sweep only | no | B-030 |
| `api_mex/hdf5_reshape_layout.cpp` | F | yes | malformed-input sweep only | no | B-030 |
| `api_mex/hdf5_version.cpp` | F | yes | yes | no | B-030 |
| `api_mex/hdf5_write_channel.cpp` | F | yes | yes | no | B-030 |
| `api_mex/hdf5_write_dset.cpp` | F | yes | yes | no | B-030 |
| `api_mex/icosphere.cpp` | F | yes | yes | no | B-030 |
| `api_mex/interp.cpp` | F | yes | yes | no | B-030 |
| `api_mex/obj_file_read.cpp` | F | yes | yes | no | B-030 |
| `api_mex/obj_file_write.cpp` | F | yes | yes | no | B-030, B-094 |
| `api_mex/plane.cpp` | F | yes | malformed-input sweep only | no | B-030 |
| `api_mex/point_cloud_aabb.cpp` | F | yes | yes | no | B-030 |
| `api_mex/point_cloud_segmentation.cpp` | F | yes | yes | no | B-030 |
| `api_mex/point_inside_mesh.cpp` | F | yes | yes | no | B-030, B-094 |
| `api_mex/qrt_file_parse.cpp` | F | yes | malformed-input sweep only | no | B-030 |
| `api_mex/qrt_file_read.cpp` | F | yes | yes | no | B-030, B-031, B-112 |
| `api_mex/quantize_delays.cpp` | F | yes | yes | no | B-030 |
| `api_mex/ray_point_intersect.cpp` | F | yes | malformed-input sweep only | no | B-030 |
| `api_mex/ray_triangle_intersect.cpp` | F | yes | yes | no | B-030 |
| `api_mex/scenpar_validate.cpp` | B | light pass (reviewed twice before) | no | no | B-030 |
| `api_mex/sos_acf_2d.cpp` | E | yes | yes (Octave, R2026a) | no | B-030 |
| `api_mex/sos_acf_approx.cpp` | E | yes | yes (Octave, R2026a) | no | B-030 |
| `api_mex/sos_acf_estimate.cpp` | E | yes | yes (Octave, R2026a) | no | B-030 |
| `api_mex/sos_acfi.cpp` | E | yes | yes (Octave, R2026a) | no | B-030 |
| `api_mex/sos_calc_mse.cpp` | E | yes | yes (Octave, R2026a) | no | B-030 |
| `api_mex/sos_generate.cpp` | E | yes | yes (Octave, R2026a) | no | B-030 |
| `api_mex/sos_init_phases.cpp` | E | yes | yes (Octave, R2026a) | no | B-030, B-034 |
| `api_mex/sos_map.cpp` | E | yes | yes (Octave, R2026a) | no | B-030 |
| `api_mex/sos_preset.cpp` | E | yes | yes (Octave, R2026a) | no | B-030 |
| `api_mex/sos_rand.cpp` | E | yes | yes (Octave, R2026a) | no | B-030 |
| `api_mex/sos_randi.cpp` | E | yes | yes (Octave, R2026a) | no | B-030 |
| `api_mex/sos_randn.cpp` | E | yes | yes (Octave, R2026a) | no | B-030 |
| `api_mex/sos_remove_bias.cpp` | E | yes | yes (Octave, R2026a) | no | B-030 |
| `api_mex/sos_val.cpp` | E | yes | yes (Octave, R2026a) | no | B-030 |
| `api_mex/subdivide_triangles.cpp` | F | yes | yes | no | B-030, B-089 |
| `api_mex/triangle_mesh_aabb.cpp` | F | yes | yes | no | B-030 |
| `api_mex/triangle_mesh_segmentation.cpp` | F | yes | yes | no | B-030, B-089 |
| `api_mex/version.cpp` | F | yes | malformed-input sweep only | no | B-030, B-115 |
| `api_mex/write_png.cpp` | F | yes | malformed-input sweep only | no | B-030 |
| `api_python/__init__.py` | G | yes (empty) | n/a | n/a | - |
| `api_python/python_RTtools.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/python_arma_adapter.hpp` | G | yes | yes | ASan module (probes + Python suite) | B-005, B-097, B-098, B-100, B-105 |
| `api_python/python_arrayant.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/python_channel.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/python_main.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/python_quadriga_adapter.hpp` | G | yes | yes | ASan module (probes + Python suite) | B-100, B-117, B-118 |
| `api_python/python_sos.cpp` | E | yes | yes | no | - |
| `api_python/python_tools.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_acdf.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_arrayant_calc_beamwidth.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_arrayant_calc_directivity.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_arrayant_combine_pattern.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_arrayant_concat.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_arrayant_copy_element.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_arrayant_export_obj_file.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_arrayant_generate.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-087 |
| `api_python/qpy_arrayant_interpolate.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-035 |
| `api_python/qpy_arrayant_qdant_read.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-117 |
| `api_python/qpy_arrayant_qdant_write.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_arrayant_rotate_pattern.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_baseband_freq_response.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-036, B-090 |
| `api_python/qpy_calc_angular_spread.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-088 |
| `api_python/qpy_calc_cross_polarization_ratio.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_calc_delay_spread.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_calc_diffraction_gain.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-095 |
| `api_python/qpy_calc_rician_k_factor.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_cart2geo.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-004 |
| `api_python/qpy_channel_export_obj_file.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-100 |
| `api_python/qpy_components.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_conf_file_list.cpp` | B | light pass (reviewed twice before) | adapter only (G) | Python suite with ASan module | - |
| `api_python/qpy_conf_file_read.cpp` | B | light pass (reviewed twice before) | adapter only (G) | Python suite with ASan module | - |
| `api_python/qpy_conf_file_write.cpp` | B | light pass (reviewed twice before) | adapter only (G) | Python suite with ASan module | - |
| `api_python/qpy_cube.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_generate_speaker.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-117 |
| `api_python/qpy_geo2cart.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-004 |
| `api_python/qpy_get_channels_ieee_indoor.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-117 |
| `api_python/qpy_get_channels_multifreq.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_get_channels_planar.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_get_channels_spherical.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_hdf5_create_file.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-116 |
| `api_python/qpy_hdf5_read_channel.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_hdf5_read_dset.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_hdf5_read_dset_names.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_hdf5_read_layout.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_hdf5_reshape_layout.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_hdf5_write_channel.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-005 |
| `api_python/qpy_hdf5_write_dset.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-005 |
| `api_python/qpy_icosphere.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_mitsuba_xml_file_write.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-099 |
| `api_python/qpy_obj_file_read.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_obj_file_write.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_plane.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_point_cloud_aabb.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-099 |
| `api_python/qpy_point_cloud_segmentation.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_point_inside_mesh.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-117 |
| `api_python/qpy_qrt_file_parse.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_qrt_file_read.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-096 |
| `api_python/qpy_quantize_delays.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_ray_point_intersect.cpp` | G | yes | yes | ASan module (probes + Python suite) | B-099 |
| `api_python/qpy_ray_triangle_intersect.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_scenpar_validate.cpp` | B | light pass (reviewed twice before) | adapter only (G) | Python suite with ASan module | - |
| `api_python/qpy_sos_acf_2d.cpp` | E | yes | yes | no | - |
| `api_python/qpy_sos_acf_approx.cpp` | E | yes | yes | no | - |
| `api_python/qpy_sos_acf_estimate.cpp` | E | yes | yes | no | - |
| `api_python/qpy_sos_acfi.cpp` | E | yes | yes | no | - |
| `api_python/qpy_sos_calc_mse.cpp` | E | yes | yes | no | - |
| `api_python/qpy_sos_generate.cpp` | E | yes | yes | no | - |
| `api_python/qpy_sos_init_phases.cpp` | E | yes | yes | no | - |
| `api_python/qpy_sos_map.cpp` | E | yes | yes | no | - |
| `api_python/qpy_sos_preset.cpp` | E | yes | yes | no | - |
| `api_python/qpy_sos_rand.cpp` | E | yes | yes | no | - |
| `api_python/qpy_sos_randi.cpp` | E | yes | yes | no | - |
| `api_python/qpy_sos_randn.cpp` | E | yes | yes | no | - |
| `api_python/qpy_sos_remove_bias.cpp` | E | yes | yes | no | - |
| `api_python/qpy_sos_val.cpp` | E | yes | yes | no | - |
| `api_python/qpy_triangle_mesh_aabb.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_triangle_mesh_segmentation.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_version.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `api_python/qpy_write_png.cpp` | G | yes | yes | ASan module (probes + Python suite) | - |
| `include/quadriga_arrayant.hpp` | A | yes | n/a | n/a | B-043, B-044, B-045, B-047, B-102 |
| `include/quadriga_channel.hpp` | B, C | yes | n/a | n/a | B-060, B-104, B-106 |
| `include/quadriga_lib.hpp` | C | partly | n/a | n/a | - |
| `include/quadriga_math.hpp` | D | yes | yes | n/a | - |
| `include/quadriga_sos.hpp` | E | yes | n/a | n/a | - |
| `include/quadriga_tools.hpp` | B, C, D | yes (by part) | n/a | n/a | B-060, B-081, B-088, B-104, B-108 |
| `src/any_type_id.cpp` | B | partly | indirectly | suite | - |
| `src/avx_mathfun.h` | D | partly (unused) | no | n/a | B-108 |
| `src/baseband_freq_response.cpp` | C | yes | yes | probe + suite | B-019, B-020, B-036, B-062 |
| `src/baseband_freq_response_avx2.cpp` | C | yes | yes | probe + suite | B-020, B-026 |
| `src/bits.hpp` | D | yes | yes | probe + suite | - |
| `src/calc_diffraction_gain.cpp` | C | partly | yes | probe + suite | B-070, B-095, B-106 |
| `src/calc_rotation_matrix.cpp` | D | yes | yes | probe + suite | - |
| `src/colormap.cpp` | B | yes | yes | UBSan what-if + suite | - |
| `src/combine_irs_coord.cpp` | C | yes | yes | probe + suite | B-061 |
| `src/coord2path.cpp` | C | yes | yes | probe + suite | - |
| `src/cube.cpp` | D | yes | yes | suite | B-082 |
| `src/fastmath_armadillo.cpp` | D | yes | yes | probe + suite | B-026, B-074, B-075 |
| `src/fastmath_avx2.h` | D | yes | yes | probe + suite | B-026, B-037, B-075, B-076 |
| `src/fastmath_vectorized_avx2.cpp` | D | yes | yes | probe + suite | B-026, B-075 |
| `src/fastmath_vectorized_avx2.h` | D | yes | n/a | n/a | - |
| `src/fastmath_vectorized_generic.h` | D | yes | yes | probe + suite | - |
| `src/generate_diffraction_paths.cpp` | C | yes | via calc_diffraction_gain | probe + suite | B-095 |
| `src/get_channels_ieee_indoor.cpp` | C | yes | yes | probe + suite | B-021, B-023, B-064, B-065 |
| `src/get_channels_irs.cpp` | C | yes | no (suite) | suite | B-003, B-105 |
| `src/get_channels_multifreq.cpp` | C | yes | yes | suite only | B-003, B-019, B-068 |
| `src/get_channels_planar.cpp` | C | yes | yes | probe + suite | B-068 |
| `src/get_channels_spherical.cpp` | C | yes | yes | probe + suite | B-002, B-003, B-006, B-068 |
| `src/icosphere.cpp` | D | yes | yes | probe + suite | B-078, B-082, B-108 |
| `src/ieee_channel_model_functions.hpp` | C | yes | yes | probe + suite | B-022, B-064 |
| `src/interp.cpp` | D | yes | yes | probe + suite | B-083, B-107 |
| `src/lodepng.cpp` | B | partly (third-party; only the calls) | indirectly | UBSan what-if | - |
| `src/lodepng.h` | B | partly (third-party) | n/a | n/a | - |
| `src/mex_compat_shim.cpp` | F | yes | linked in what-if | no | B-029 |
| `src/mex_helper_functions.hpp` | F | yes | yes | ASan what-if (one MEX) | B-030, B-033, B-034, B-093 |
| `src/mex_quadriga_lib_functions.hpp` | F | yes | yes | no | - |
| `src/mitsuba_xml_file_write.cpp` | B | yes | yes | suite only | B-056, B-104 |
| `src/obj_file_read.cpp` | B | yes | yes, fuzzed (AFL: OBJ, MTL, CSV) | probe + suite + fuzz | B-057, B-081 |
| `src/obj_file_write.cpp` | B | yes | yes | probe + suite | B-018 |
| `src/obj_overlap_test.cpp` | D | yes | yes | probe + suite | B-027, B-028, B-077, B-081, B-084, B-108 |
| `src/path_to_tube.cpp` | C | yes | yes | probe + suite | B-025, B-106 |
| `src/plane.cpp` | D | yes | yes | suite | B-082 |
| `src/point_cloud_aabb.cpp` | D | yes | indirectly | probe + suite | - |
| `src/point_cloud_segmentation.cpp` | D | yes | yes | probe + suite | B-108 |
| `src/point_cloud_split.cpp` | D | yes | yes | probe + suite | B-080 |
| `src/point_inside_mesh.cpp` | D | yes | yes | probe + suite | B-027 |
| `src/qd_arrayant.cpp` | A | yes | yes | probe + suite | B-008, B-009, B-010, B-043, B-045, B-046, B-101, B-102 |
| `src/qd_arrayant_functions.hpp` | A | yes | no (declarations) | n/a | - |
| `src/qd_arrayant_generate.cpp` | A | yes | yes | probe + suite | B-011, B-044, B-047, B-087 |
| `src/qd_arrayant_interpolate.cpp` | A | yes | yes | probe + suite | B-007 |
| `src/qd_arrayant_interpolate.hpp` | A | yes | no (declarations) | n/a | - |
| `src/qd_arrayant_interpolate_avx2.cpp` | A | yes | yes | probe + suite | B-006, B-102 |
| `src/qd_arrayant_interpolate_avx2.hpp` | A | yes | no (declarations) | n/a | - |
| `src/qd_arrayant_qdant.cpp` | A | yes | yes, fuzzed (AFL) | probe + suite + fuzz | B-012, B-038, B-040, B-041, B-042 |
| `src/qd_arrayant_speaker.cpp` | A | yes | partly | probe + suite | - |
| `src/qd_arrayant_xinterpol_multi.cpp` | A | yes | yes | probe + suite | - |
| `src/qd_arrayant_xio.cpp` | A | yes | yes | probe + suite | B-008, B-039, B-045 |
| `src/qd_atmospheric_absorption.hpp` | C | yes | no | suite only | - |
| `src/qd_channel.cpp` | B | yes | yes | probe + suite | B-059 |
| `src/qd_channel_hdf5.cpp` | B | yes | yes, fuzzed (AFL) | probe + suite + fuzz | B-001, B-013, B-014, B-015, B-050, B-051, B-052, B-053 |
| `src/qd_channel_qrt.cpp` | B | yes | yes, fuzzed (AFL) | probe + suite + fuzz | B-016, B-017, B-048, B-049, B-059, B-104 |
| `src/qd_channel_quantize_delays.cpp` | B | yes | yes | probe + suite | B-002, B-054, B-055, B-103 |
| `src/qd_coeff_combine_avx2.cpp` | C | yes | yes | probe + suite | - |
| `src/qd_path.cpp` | C | yes | yes | probe + suite | - |
| `src/qd_scenpar.cpp` | B | light pass (reviewed twice before) | regression probe, fuzzed (AFL) | probe + suite + fuzz | - |
| `src/qd_tools_acdf.cpp` | C | yes | yes | probe + suite | B-067 |
| `src/qd_tools_calc_angular_spread.cpp` | C | yes | yes | what-if + suite | B-065, B-088 |
| `src/qd_tools_calc_delay_spread.cpp` | C | yes | yes | probe + suite | B-065, B-066 |
| `src/qd_tools_calc_rician_k_factor.cpp` | C | yes | no | suite only | - |
| `src/qd_tools_calc_xpr.cpp` | C | yes | no (via C-18 probe) | suite only | B-069 |
| `src/quadriga_lib.cpp` | C | yes | no | suite only | B-062, B-108 |
| `src/quadriga_lib_avx2_functions.hpp` | D | partly | n/a | n/a | - |
| `src/quadriga_lib_generic_functions.hpp` | D | partly | n/a | n/a | B-020 |
| `src/quadriga_lib_helper_functions.hpp` | D | partly (geometry, rotation, cart/geo) | yes | probe + suite | - |
| `src/quadriga_lib_test_avx2.cpp` | D | yes | indirectly | probe + suite | - |
| `src/ray_point_intersect.cpp` | D | yes | yes | probe + suite | B-073, B-074 |
| `src/ray_point_intersect_avx2.cpp` | D | yes | yes | probe + suite | B-073 |
| `src/ray_point_intersect_generic.cpp` | D | yes | yes | probe + suite | - |
| `src/ray_triangle_intersect.cpp` | D | yes | yes | probe + suite | B-027, B-074, B-079, B-108 |
| `src/ray_triangle_intersect_avx2.cpp` | D | yes | yes | probe + suite | B-027, B-071, B-072 |
| `src/ray_triangle_intersect_generic.cpp` | D | yes | yes | probe + suite | B-071, B-072 |
| `src/runtime_AVX2_Check_avx2.cpp` | D | yes | no | suite | B-108 |
| `src/slerp.h` | D | yes | yes | probe + suite | - |
| `src/sos_acf_2d.cpp` | E | yes | yes | probe + suite | - |
| `src/sos_acf_approx.cpp` | E | yes | yes | probe + suite | - |
| `src/sos_acf_estimate.cpp` | E | yes | yes | probe + suite | - |
| `src/sos_acfi.cpp` | E | yes | yes | probe + suite | - |
| `src/sos_calc_mse.cpp` | E | yes | yes | probe + suite | - |
| `src/sos_generate.cpp` | E | partly (not the optimiser maths) | yes | probe + suite | - |
| `src/sos_init_phases.cpp` | E | yes | yes | probe + suite | - |
| `src/sos_map.cpp` | E | yes | yes | probe + suite | - |
| `src/sos_preset.cpp` | E | partly (not the tables) | yes | probe + suite | B-110 |
| `src/sos_rand.cpp` | E | yes | yes | probe + suite | B-110 |
| `src/sos_remove_bias.cpp` | E | yes | yes | probe + suite | B-109 |
| `src/sos_val.cpp` | E | yes | yes | probe + suite | B-109 |
| `src/sos_val_avx2.cpp` | E | yes | yes | probe + suite | - |
| `src/sos_val_generic.cpp` | E | yes | yes | probe + suite | - |
| `src/subdivide_triangles.cpp` | D | yes | yes | probe + suite | B-078, B-079, B-108 |
| `src/triangle_mesh_aabb.cpp` | D | yes | yes | probe + suite | - |
| `src/triangle_mesh_segmentation.cpp` | D | yes | yes | probe + suite | B-089, B-108 |
| `src/triangle_mesh_split.cpp` | D | yes | yes | probe + suite | B-080 |
| `src/update_xpr.cpp` | C | yes | yes | probe + suite | B-024 |
| `src/write_png.cpp` | B | yes | yes | UBSan what-if + suite | B-058 |

## B-001: `hdf5_write` silently drops the `par_*` data when the slot already holds a `Data` group

- Status: fixed (WP-02)
- Severity: S1, because it is data loss in a writer in normal use: unstructured parameters written with `hdf5_write` are lost (or old values survive) without an error.
- Kind: wrong result (writer data loss)
- Mark: [R]
- Where: `src/qd_channel_hdf5.cpp:1497-1512` (`hdf5_write`, `H5Gcreate2(..., "Data", ...)`) together with `qHDF_get_channel_ID` (usage 1) at 197-219, which reuses the group of the slot.

**What happens.** Case 1: `hdf5_write_dset(fn, "a", 1)` into a slot, then `hdf5_write` of a channel with `par` "b" = 2 into the same slot: reading back gives only `a = 1` (b expected). Case 2: rewriting a parameter-only slot with `x = 99` reads back the old `x = 1`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-003/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B-003: hdf5_write silently drops unstructured data when the slot holds only unstructured data
#include "quadriga_lib.hpp"
#include <cstdio>
static void show(const std::string &fn, const char *label)
{
    auto r = quadriga_lib::hdf5_read_channel<double>(fn);
    std::printf("%s: par fields read back:", label);
    for (size_t i = 0; i < r.par_names.size(); ++i)
        std::printf(" %s=%g", r.par_names[i].c_str(), std::any_cast<double>(r.par_data[i]));
    std::printf("\n");
}
int main(int argc, char **argv)
{
    std::string fn = std::string(argv[1]) + "/par.hdf5";
    // Case 1: hdf5_write_dset first, then hdf5_write of a channel with par "b"
    std::remove(fn.c_str());
    std::any a = 1.0;
    quadriga_lib::hdf5_write_dset(fn, "a", &a);
    quadriga_lib::channel<double> c;
    c.tx_pos = arma::mat(3, 1, arma::fill::ones);
    c.rx_pos = arma::mat(3, 1, arma::fill::zeros);
    c.path_gain.push_back(arma::vec(3, arma::fill::ones));
    c.par_names = {"b"};
    c.par_data = {std::any(2.0)};
    std::printf("case 1: hdf5_write returned %d, no exception\n", quadriga_lib::hdf5_write(&c, fn));
    show(fn, "case 1 (expected a=1 b=2)");
    // Case 2: the same parameter-only channel written twice to slot 0 ("overwrites slot content")
    std::remove(fn.c_str());
    quadriga_lib::channel<double> p;
    p.par_names = {"x"};
    p.par_data = {std::any(1.0)};
    quadriga_lib::hdf5_write(&p, fn);
    p.par_data = {std::any(99.0)};
    std::printf("case 2: second hdf5_write returned %d, no exception\n", quadriga_lib::hdf5_write(&p, fn));
    show(fn, "case 2 (expected x=99)");
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-003/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-003
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
# keep sanitizer reports short: error line, access line, frames reduced to file:line
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|^    #[0-9]+ .*(src/|repro)|$1" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${2:-8}; }
$REL "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $H5
"$D/repro" "$D" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
case 1: hdf5_write returned 1, no exception
case 1 (expected a=1 b=2): par fields read back: a=1
case 2: second hdf5_write returned 1, no exception
case 2 (expected x=99): par fields read back: x=1
```

**Cause.** The reused group already contains `Data`; `H5Gcreate2("Data")` fails, and every later `H5Dcreate2` fails silently because HDF5 error printing has been switched off (B-051) and the return codes are not checked.

**Proposed fix.** Open `Data` if it exists; to overwrite, delete an existing `par_<name>` link before writing it (or always create a fresh group); check the results of `H5Gcreate2`/`H5Dcreate2` and throw. File: `src/qd_channel_hdf5.cpp`. Decide (and document) whether the old `par` fields of a slot are kept when the channel is rewritten.

**Test for the fix session.** Catch2: `hdf5_write_dset("a")` then `hdf5_write` with par "b" reads back both a and b; writing a parameter-only channel twice reads back `x = 99`. (Superseded by decision B-001 (a): after the rewrite only b is there; the repro's "expected a=1 b=2" is the old expectation. WP-02 tests b only.)

## B-002: `quantize_delays` segfaults on a negative delay, including the slightly negative relative delays that `get_channels_spherical` returns

- Status: fixed (WP-03); delays at or below `-tap_spacing / 2` still throw by decision (2026-10-17): tapped delay line emulators need non-negative delays, and relative delays of acoustic arrays and of radio arrays of about 1.5 m and more go below that limit (use `use_absolute_delays = true`)
- Severity: S1, because it crashes on valid input in normal use: `get_channels_spherical` with its default relative delays returns negative delays for some element pairs of a rotated array (here 16 of 64, down to -6.3e-11 s), and passing that output to `quantize_delays` segfaults, also from Python.
- Kind: crash (memory, UB)
- Mark: [R]
- Where: `src/qd_channel_quantize_delays.cpp:301-307` (`(unsigned)fl` of a negative or NaN tap value, then `fl_u + 1`), 36-38 and 76 (`pdp(max_idx + 1)` with `max_idx` = `UINT_MAX`, write through a null pointer at line 38), 545 (`tap2pos(max_grid_val + 2)`, same wrap risk).

**What happens.** Delays `{-1e-20, 12.5e-9, 33.4e-9}` or `{-5e-9, ...}`: segfault (exit 139); ASan: "load of null pointer" and SEGV at `qd_channel_quantize_delays.cpp:38`; the same call from Python kills the interpreter. A NaN delay silently drops the path (4 taps returned, no error). The output of `get_channels_spherical` for an 8-element TX ULA and a 4-element RX ULA turned by 0.3 rad has 16 negative delays and crashes `quantize_delays` the same way. Delays of 21.47 s or more (2^32 taps of 5 ns) give `bad_alloc` or a multi-GB allocation.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-006/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 13: quantize_delays edge cases.
// Usage: p13 <case>
//   negtiny : one delay of -1e-20 s (rounding noise below zero)
//   neg     : one delay of -5e-9 s
//   nan     : one delay NaN
//   big     : one delay of 21.4748 s (= (2^32 - 1) taps of 5 ns)
//   alias   : outputs alias the inputs (same std::vector objects)
//   shape   : per-antenna delays with fix_taps = 1 and 2 (doc says output delays [1,1,n_taps])
#include "quadriga_lib.hpp"
#include <cstdio>
#include <cmath>
int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string cs = argc > 1 ? argv[1] : "negtiny";
    std::vector<arma::cube> cre(1), cim(1), dl(1), qre, qim, qdl;
    cre[0] = arma::cube(1, 1, 3, arma::fill::ones);
    cim[0] = arma::cube(1, 1, 3, arma::fill::zeros);
    dl[0] = arma::cube(1, 1, 3);
    dl[0](0, 0, 0) = 0.0, dl[0](0, 0, 1) = 12.5e-9, dl[0](0, 0, 2) = 33.4e-9;
    if (cs == "negtiny") dl[0](0, 0, 0) = -1e-20;
    if (cs == "neg") dl[0](0, 0, 0) = -5e-9;
    if (cs == "nan") dl[0](0, 0, 0) = NAN;
    if (cs == "big") dl[0](0, 0, 0) = 21.4748;
    if (cs == "shape")
    {
        cre[0] = arma::cube(2, 2, 3, arma::fill::ones), cim[0] = arma::cube(2, 2, 3, arma::fill::zeros);
        dl[0] = arma::cube(2, 2, 3);
        for (int p = 0; p < 3; ++p) dl[0].slice(p).fill(10e-9 * p + 2e-9);
        for (int ft : {1, 2})
        {
            quadriga_lib::quantize_delays(&cre, &cim, &dl, &qre, &qim, &qdl, 5e-9, 48, 1.0, ft);
            std::printf("per-antenna input, fix_taps=%d: delay_quant size %llux%llux%llu\n", ft,
                        (unsigned long long)qdl[0].n_rows, (unsigned long long)qdl[0].n_cols, (unsigned long long)qdl[0].n_slices);
        }
        return 0;
    }
    if (cs == "alias")
    {
        std::vector<arma::cube> r0, i0, d0;
        quadriga_lib::quantize_delays(&cre, &cim, &dl, &r0, &i0, &d0, 5e-9, 48, 1.0, 0);
        quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre, &cim, &dl, 5e-9, 48, 1.0, 0);
        std::printf("separate outputs: coeff_re taps = ");
        for (auto v : r0[0]) std::printf("%g ", v);
        std::printf("\naliased outputs:  coeff_re taps = ");
        for (auto v : cre[0]) std::printf("%g ", v);
        std::printf("\n");
        return 0;
    }
    std::printf("case %s: delays = %g %g %g s\n", cs.c_str(), dl[0](0), dl[0](1), dl[0](2));
    try
    {
        quadriga_lib::quantize_delays(&cre, &cim, &dl, &qre, &qim, &qdl, 5e-9, 48, 1.0, 0);
        std::printf("returned: %llu taps, delays:", (unsigned long long)qdl[0].n_slices);
        for (auto v : qdl[0]) std::printf(" %g", v);
        std::printf("\n");
    }
    catch (const std::exception &e) { std::printf("exception: %s\n", e.what()); }
    return 0;
}
```

`repro.py`:
```python
# B probe 13 (Python): a delay that is negative by rounding noise crashes the interpreter
import numpy as np, quadriga_lib, sys
cre = np.ones((1, 1, 3)); cim = np.zeros((1, 1, 3))
dl = np.array([-1e-20, 12.5e-9, 33.4e-9]).reshape(1, 1, 3)
print("calling quantize_delays with delays", dl.ravel()); sys.stdout.flush()
out = quadriga_lib.channel.quantize_delays([cre], [cim], [dl])
print("returned", [o.shape for o in out])
```

`repro_normal.cpp`:
```cpp
// B-006 (normal use): get_channels_spherical with relative delays (the default) returns negative delays for a
// rotated receive array; quantize_delays on that output then crashes.
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    auto tx = quadriga_lib::generate_arrayant_ula<double>(8, 3.5e9, 0.5);
    auto rx = quadriga_lib::generate_arrayant_ula<double>(4, 3.5e9, 0.5);
    arma::mat fbs = {{50.0, 50.0}, {0.0, 30.0}, {10.0, 2.0}}, lbs = fbs; // path 0: LOS (on the TX-RX line), path 1: NLOS
    arma::vec gain = {1.0, 0.5}, len = {0.0, 0.0};
    arma::mat M(8, 2, arma::fill::zeros);
    M.row(0).fill(1.0), M.row(6).fill(-1.0);
    std::vector<arma::cube> cre(1), cim(1), dl(1), qre, qim, qdl;
    quadriga_lib::get_channels_spherical<double>(&tx, &rx, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 100.0, 0.0, 10.0, 0.0, 0.0, 0.3,
                                                 &fbs, &lbs, &gain, &len, &M, &cre[0], &cim[0], &dl[0], 3.5e9);
    std::printf("get_channels_spherical, RX heading 0.3 rad: min delay = %.3e s, negative delays: %llu of %llu\n",
                dl[0].min(), (unsigned long long)arma::accu(dl[0] < 0.0), (unsigned long long)dl[0].n_elem);
    std::printf("calling quantize_delays on this output ...\n");
    std::fflush(stdout);
    quadriga_lib::quantize_delays(&cre, &cim, &dl, &qre, &qim, &qdl);
    std::printf("returned %llu taps\n", (unsigned long long)qdl[0].n_slices);
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-006/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-006
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$REL "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $H5
$ASAN "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $H5
for c in negtiny neg nan; do echo "== release, case $c"; "$D/repro" $c || echo "exit code $?"; done
echo "== ASan, case negtiny"; ASAN_OPTIONS=halt_on_error=1 "$D/repro_asan" negtiny 2>&1 | filt 6
echo "== Python"; PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 -u "$D/repro.py" || echo "exit code $?"
echo "== normal use: output of get_channels_spherical (relative delays, rotated RX array) passed to quantize_delays"
$REL "$D/repro_normal.cpp" -o "$D/repro_normal" build_linux/libquadriga.a $H5
"$D/repro_normal" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== release, case negtiny
case negtiny: delays = -1e-20 1.25e-08 3.34e-08 s
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/B-006//run.sh: line 11: 712783 Segmentation fault         (core dumped) "$D/repro" $c
exit code 139
== release, case neg
case neg: delays = -5e-09 1.25e-08 3.34e-08 s
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/B-006//run.sh: line 11: 712809 Segmentation fault         (core dumped) "$D/repro" $c
exit code 139
== release, case nan
case nan: delays = nan 1.25e-08 3.34e-08 s
returned: 4 taps, delays: 1e-08 1.5e-08 3e-08 3.5e-08
== ASan, case negtiny
/usr/include/c++/12/bits/stl_vector.h:1129:34: runtime error: reference binding to null pointer of type 'value_type'
/sjc/quadriga-lib-fixes/src/qd_channel_quantize_delays.cpp:38:24: runtime error: load of null pointer of type 'value_type'
==712827==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000 (pc 0x63c0292dcb01 bp 0x7fffd3074df0 sp 0x7fffd3073d80 T0)
    #0 /sjc/quadriga-lib-fixes/src/qd_channel_quantize_delays.cpp:38
    #1 /sjc/quadriga-lib-fixes/src/qd_channel_quantize_delays.cpp:426
    #2 /sjc/quadriga-lib-fixes/src/qd_channel_quantize_delays.cpp:478
== Python
calling quantize_delays with delays [-1.00e-20  1.25e-08  3.34e-08]
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/B-006//run.sh: line 13: 712834 Segmentation fault         (core dumped) PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 -u "$D/repro.py"
exit code 139
== normal use: output of get_channels_spherical (relative delays, rotated RX array) passed to quantize_delays
get_channels_spherical, RX heading 0.3 rad: min delay = -6.333e-11 s, negative delays: 16 of 64
calling quantize_delays on this output ...
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/B-006//run.sh: line 16: 712867 Segmentation fault         (core dumped) "$D/repro_normal"
exit code 139
```

Added after WP-03: the negative delays are not always small. They reach about (half the TX array size + half the RX array size) / propagation speed (both arrays endfire): -0.9 taps for a 2.7 m radio array at 5 ns, -1 to -24 taps for 5-35 cm acoustic arrays at 48 kHz (manager's check, `build_linux/scratch/fix/WP-03/manager/neg_delay_use.cpp`). Such delays throw after WP-03 (decision of 2026-10-17).

**Cause.** The conversion of a negative (or NaN, or too large) tap value to `unsigned` is undefined; it wraps to `UINT_MAX`, `max_idx + 1` becomes 0, the vector of length 0 is indexed. Separately, `get_channels_spherical` documents that "the minimum delay (LOS delay) is subtracted", but with per-element spherical delays the subtracted LOS delay is that of the phase centres, so elements closer than the phase centre get negative delays.

**Proposed fix.** In `quantize_delays`: validate the delays (finite, `delay / tap_spacing` below a bound such as 2^31) and throw `std::invalid_argument` otherwise; clamp small negative delays (above `-tap_spacing / 2`) to tap 0 as a documented rule, since negative relative delays are normal output of `get_channels_spherical`. Also correct the `get_channels_spherical` sentence on the subtracted delay. Files: `src/qd_channel_quantize_delays.cpp`, documentation in `src/get_channels_spherical.cpp`.

**Test for the fix session.** Catch2: the `get_channels_spherical` output of `repro_normal.cpp` quantizes without crash and every negative delay lands in tap 0; delays `{-1e-20, 12.5e-9, 33.4e-9}` quantize with tap 0; a NaN delay throws `std::invalid_argument`. pytest: the Python call returns.

## B-003: `get_channels_spherical`, `_multifreq` and `_irs` ignore `path_length` for NLOS paths: delays and phases too short

- Status: fixed (WP-04)
- Severity: S1, because paths with more than two interactions (the normal output of `qrt_file_read` / `coord2path`: FBS = first, LBS = last interaction, `path_length` = full length) silently get the delay and phase of the shortest TX-FBS-LBS-RX polyline, which is wrong on valid input in normal use.
- Kind: wrong result
- Mark: [R]
- Where: `src/get_channels_spherical.cpp:271-283` (`d_fbs_lbs` is the geometric FBS-LBS distance; `d_length = max(path_length, d_shortest)` is used only in the LOS test at 286) and 343 (`p_delays[io] = dt + d_fbs_lbs + dr[r]`); `src/get_channels_multifreq.cpp:348-365, 420` (same code); `src/get_channels_irs.cpp:180-237` (both segments, through the spherical code). The documentation of all three says "`path_length` — Total path lengths from TX to RX phase center".
- Since: 808cb0d (2023-08-24, "Added get_channels_planar"), which replaced the earlier `d_fbs_lbs = d_length - d_shortest` (1ab452e).

**What happens.** TX (0,0,0), RX (100,0,0), FBS (30,40,0), LBS (70,40,0), omni antennas, absolute delays: `coord2path` gives `path_length = 156.568542 m`, but `get_channels_spherical` returns `delay·c = 140.000 m` for `path_length` 140, 156.57 and 300 alike (the same coefficient each time); `get_channels_multifreq` the same. Expected 156.569 m and 300 m.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-1/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-1: get_channels_spherical / _multifreq ignore path_length for NLOS paths
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>(10.0);
    const double C = 299792458.0;
    // TX (0,0,0), RX (100,0,0); 3-bounce path via (30,40,0), (50,60,0), (70,40,0) as coord2path / qrt_file_read return it
    arma::u32_vec no_int = {3};
    arma::mat coord = {{30.0, 50.0, 70.0}, {40.0, 60.0, 40.0}, {0.0, 0.0, 0.0}};
    arma::vec len; arma::mat fbs, lbs;
    quadriga_lib::coord2path<double>(0, 0, 0, 100, 0, 0, &no_int, &coord, &len, &fbs, &lbs);
    printf("coord2path: path_length = %.6f m (geometric TX-FBS-LBS-RX = 140 m)\n", len(0));
    arma::vec gain = {1.0};
    arma::mat M = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0});
    arma::cube cr, ci, dl;
    quadriga_lib::get_channels_spherical<double>(&ant, &ant, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0,
                                                 &fbs, &lbs, &gain, &len, &M, &cr, &ci, &dl, 3.0e9, true, false);
    printf("get_channels_spherical: delay*c = %.6f m (expected %.6f)\n", dl(0) * C, len(0));
    std::vector<quadriga_lib::arrayant<double>> va = {ant};
    arma::mat pg(1, 1, arma::fill::ones);
    arma::cube Mc(8, 1, 1); Mc.slice(0) = M;
    arma::vec f = {3.0e9};
    std::vector<arma::cube> vr, vi, vd;
    quadriga_lib::get_channels_multifreq<double>(va, va, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0,
                                                 fbs, lbs, pg, len, Mc, f, f, vr, vi, vd, true, false);
    printf("get_channels_multifreq: delay*c = %.6f m (expected %.6f)\n", vd[0](0) * C, len(0));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-1/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-1
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
coord2path: path_length = 156.568542 m (geometric TX-FBS-LBS-RX = 140 m)
get_channels_spherical: delay*c = 140.000000 m (expected 156.568542)
get_channels_multifreq: delay*c = 140.000000 m (expected 156.568542)
```

**Cause.** The NLOS branch uses only the geometric segment lengths TX-FBS, FBS-LBS and LBS-RX; the given total path length enters only the LOS detection.

**Proposed fix.** In the NLOS branch use `d_fbs_lbs = d_length - |TX-FBS| - |LBS-RX|` (with phase-centre distances, lines 275-281), so that the phase-centre delay equals `max(path_length, d_shortest)` and the per-element spherical offsets stay as they are. Files: `src/get_channels_spherical.cpp`, `src/get_channels_multifreq.cpp` (IRS follows).

**Test for the fix session.** Catch2 with the repro geometry: `delay·c == 156.568542` (1e-6) for spherical and multifreq; `path_length < 140` still gives 140; the IRS chain gives the sum of both segment lengths.

## B-004: Python `tools.geo2cart` / `tools.cart2geo` with separate inputs read freed memory: wrong values, segfault

- Status: fixed (WP-05)
- Severity: S1, because an ordinary call with separate coordinate arrays of 17 or more elements silently returns wrong values, and large inputs crash the interpreter.
- Kind: memory (use after free) / wrong result
- Mark: [R]
- Where: `api_python/qpy_geo2cart.cpp:75-92` (`geo2cart`, separate-input branch), `api_python/qpy_cart2geo.cpp:69-80` (`cart2geo`, same pattern). No other wrapper uses the pattern (grep for `= arma::vec(..., false, true)`).
- Since: a00507e (2026-06-18)

**What happens.** `geo2cart(az, el, len)` with three separate arrays must equal `geo2cart(np.stack([az, el, len]))`. For n = 1000 the results differ by up to 7.9 (geo2cart) and 9.7 (cart2geo); for n = 100000 the process dies with SIGSEGV. Under the ASan build of the module: heap-use-after-free, read in `fast_geo2cart` called from `qpy_geo2cart.cpp:124`, memory freed by `~Mat` at `qpy_geo2cart.cpp:92`; for n <= 16 (Armadillo's in-object storage) it is a stack-use-after-scope instead. The Python test suite does hit this path (15 stack-use-after-scope reports when it runs against an ASan build of the module) but passes, because its inputs have at most 16 elements.

**Reproduce.** The second half of `run.sh` uses an ASan+UBSan build of the Python module made by area G in `build_linux/scratch/hunt/G/asan_mod/`: the unchanged `api_python/python_*.cpp` compiled with `-O1 -g -fsanitize=address,undefined` and linked against `build_asan/libquadriga.a`. The first half needs only the normal module in `lib/`.

Folder `build_linux/scratch/hunt/final/G-1/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-1: geo2cart / cart2geo with separate inputs read freed memory: wrong results, segfault
import numpy as np, sys, subprocess, quadriga_lib as q
T = q.tools
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rng = np.random.default_rng(0)
a = rng.uniform(-3, 3, n); b = rng.uniform(-1.5, 1.5, n); c = rng.uniform(1, 10, n)
for name, f in (('geo2cart', T.geo2cart), ('cart2geo', T.cart2geo)):
    got = np.stack([x.ravel() for x in f(a, b, c, combine=False)])            # separate inputs
    ref = f(np.stack([a, b, c]).reshape(3, n, 1), combine=True).reshape(3, n)  # combined input (copies)
    print(f"{name} n={n}: max |separate - combined| = {np.max(np.abs(got - ref)):.3e} (expected 0)")
if len(sys.argv) == 1:
    r = subprocess.run([sys.executable, __file__, '100000'], capture_output=True, text=True)
    print("n=100000 in a subprocess: return code", r.returncode, "(-11 = SIGSEGV)", r.stdout.strip())
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-1/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-1
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
echo "--- same repro against the ASan/UBSan probe module ---"
# ASan module: build_linux/scratch/hunt/G/asan_mod (api_python/python_*.cpp unchanged, -O1 -g -fsanitize=address,undefined,
# linked against build_asan/libquadriga.a; build commands in G area report)
LD_PRELOAD="$(g++-12 -print-file-name=libasan.so) $(g++-12 -print-file-name=libubsan.so)" ASAN_OPTIONS=detect_leaks=0 \
  OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/G/asan_mod python3 "$D/repro.py" 2>&1 \
  | grep -E "ERROR: |SUMMARY|runtime error|^    #[0-4] |^[a-zA-Z].*->|max" | cut -c1-200 | head -12 || true
```

Output of the phase-2 re-run (`bash run.sh`):
```text
geo2cart n=1000: max |separate - combined| = 7.897e+00 (expected 0)
cart2geo n=1000: max |separate - combined| = 9.651e+00 (expected 0)
n=100000 in a subprocess: return code -11 (-11 = SIGSEGV) 
--- same repro against the ASan/UBSan probe module ---
==721861==ERROR: AddressSanitizer: heap-use-after-free on address 0x7e25e0cdc900 at pc 0x7bd5c449c7d7 bp 0x7ffc3690f990 sp 0x7ffc3690f980
    #0 0x7bd5c449c7d6 in void qd_GEO2CART_GENERIC<double>(double const*, double const*, double const*, double*, double*, double*, double*, double*, double*, double*, unsigned long) [clone ._omp_fn.0] 
    #1 0x7bd5c9b76708 in GOMP_parallel (/usr/lib/x86_64-linux-gnu/libgomp.so.1+0x19708) (BuildId: 2cc7443742f4addce7754a3559f228a07b467315)
    #2 0x7bd5c450b79a in void qd_GEO2CART_GENERIC<double>(double const*, double const*, double const*, double*, double*, double*, double*, double*, double*, double*, unsigned long) /sjc/quadriga-lib-f
    #3 0x7bd5c450b79a in void quadriga_lib::fast_geo2cart<double>(arma::Col<double> const&, arma::Col<double> const&, arma::Col<double>&, arma::Col<double>&, arma::Col<double>&, arma::Col<double>*, ar
    #4 0x7bd5c42ea429 in geo2cart(pybind11::array_t<double, 16> const&, pybind11::handle, pybind11::handle, bool, int) /sjc/quadriga-lib-fixes/api_python/qpy_geo2cart.cpp:124
    #0 0x7fd5e292a3ff in free ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:51
    #1 0x7bd5c42e8956 in void arma::memory::release<double const>(double const*) /usr/include/armadillo_bits/memory.hpp:125
    #2 0x7bd5c42e8956 in arma::Mat<double>::~Mat() /usr/include/armadillo_bits/Mat_meat.hpp:32
    #3 0x7bd5c42e8956 in geo2cart(pybind11::array_t<double, 16> const&, pybind11::handle, pybind11::handle, bool, int) /sjc/quadriga-lib-fixes/api_python/qpy_geo2cart.cpp:92
    #4 0x7bd5c434d655 in pybind11::object pybind11::detail::argument_loader<pybind11::array_t<double, 16> const&, pybind11::handle, pybind11::handle, bool, int>::call_impl<pybind11::object, pybind11::
    #0 0x7fd5e292ae8f in posix_memalign ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:139
```

**Cause.** In the separate-input branch the wrapper copies each numpy input into a local `arma::vec az_a` (and `el_a`, `len_a`) declared inside the `else` block, then sets `az_v = arma::vec(az_a.memptr(), n, false, true)`. That is a move assignment from a temporary that aliases memory; Armadillo takes over the alias, so `az_v` points into `az_a`. At the end of the `else` block `az_a` is destroyed, and the later `fast_geo2cart` / `fast_cart2geo` call reads freed heap memory (or out-of-scope stack memory for n <= 16).

**Proposed fix.** Declare `az_a`, `el_a`, `len_a` (and in `cart2geo` `x_a`, `y_a`, `z_a`) in function scope before the `if`, so that they outlive the views. Alternative: plain copies (`az_v = az_a`), one extra copy. Files: `api_python/qpy_geo2cart.cpp`, `api_python/qpy_cart2geo.cpp`.

**Test for the fix session.** pytest: for n in (5, 100, 1000, 100000) and `combine` in (True, False), `geo2cart(az, el, len)` equals `geo2cart(np.stack([az, el, len]))` exactly, and the same for `cart2geo`. Run the test once against an ASan build of the module.

## B-005: Python HDF5 writers ignore numpy strides for 1-D and (n,1) arrays: wrong data written, heap over-read

- Status: fixed (WP-05)
- Severity: S1, because ordinary numpy slices (`x[::2]`, `y[:, 1:2]`, `x[::-1]`) are written to the file as the wrong values without any error, which is data loss in a writer; the reversed case also reads past the heap buffer.
- Kind: wrong result (writer data loss) / memory
- Mark: [R]
- Where: `api_python/python_arma_adapter.hpp:1434-1437` (`qd_python_anycast`, ndim 1 and (n,1) branches), the same pattern at 1447-1450, 1460-1463, 1473-1476, 1486-1489, 1499-1502. Callers: `api_python/qpy_hdf5_write_dset.cpp:57` (`hdf5_write_dset`), `api_python/qpy_hdf5_write_channel.cpp:148` (`par` data of `hdf5_write_channel`).

**What happens.** `hdf5_write_dset(fn, 'x', x[::2])` with `x = arange(8)` reads back `[0 1 2 3]` instead of `[0 2 4 6]`; a column slice `y[:, 1:2]` reads back the first column; a structured-field view reads back zeros; `x[::-1]` of 1000 elements reads back garbage and ASan reports a heap-buffer-overflow READ. The `par` dictionary of `hdf5_write_channel` behaves the same.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/G-3/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-3: qd_python_anycast ignores strides: hdf5_write_dset / hdf5_write_channel par store wrong data
import numpy as np, os, quadriga_lib as q
C = q.channel
D = os.path.dirname(os.path.abspath(__file__)); fn = os.path.join(D, 'g3.hdf5')
if os.path.exists(fn): os.remove(fn)
C.hdf5_create_file(fn, 1)
x = np.arange(1000.0); y = np.arange(12.0).reshape(6, 2)
for name, v in (('step2', x[::2]), ('column', y[:, 1:2]), ('reversed', x[::-1])):
    C.hdf5_write_dset(fn, 0, 0, 0, 0, name, v)
    r = np.ravel(C.hdf5_read_dset(fn, name=name))
    print(f"hdf5_write_dset {name:8s} -> read back {r[:4]} expected {np.ravel(v)[:4]}")
fn2 = os.path.join(D, 'g3c.hdf5')
if os.path.exists(fn2): os.remove(fn2)
chan = {'name': 'c', 'tx_position': np.zeros((3, 1)), 'rx_position': np.ones((3, 1)), 'center_frequency': np.array([1e9]),
        'coeff': [np.ones((1, 1, 1), complex)], 'delay': [np.zeros((1, 1, 1))]}
C.hdf5_write_channel(fn2, chan, {'p': x[:10:2]})
print("hdf5_write_channel par x[:10:2] -> read back", C.hdf5_read_channel(fn2)[1]['p'], "expected", x[:10:2])
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-3/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-3
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
echo "--- same repro against the ASan/UBSan probe module ---"
# ASan module: build_linux/scratch/hunt/G/asan_mod (api_python/python_*.cpp unchanged, -O1 -g -fsanitize=address,undefined,
# linked against build_asan/libquadriga.a; build commands in G area report)
LD_PRELOAD="$(g++-12 -print-file-name=libasan.so) $(g++-12 -print-file-name=libubsan.so)" ASAN_OPTIONS=detect_leaks=0 \
  OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/G/asan_mod python3 "$D/repro.py" 2>&1 \
  | grep -E "ERROR: |SUMMARY|runtime error|^    #[0-4] |^[a-zA-Z].*->|max" | cut -c1-200 | head -12 || true
```

Output of the phase-2 re-run (`bash run.sh`):
```text
hdf5_write_dset step2    -> read back [0. 1. 2. 3.] expected [0. 2. 4. 6.]
hdf5_write_dset column   -> read back [1. 2. 3. 4.] expected [1. 3. 5. 7.]
hdf5_write_dset reversed -> read back [9.99000000e+002 0.00000000e+000 1.63041663e-322 6.85701098e-310] expected [999. 998. 997. 996.]
hdf5_write_channel par x[:10:2] -> read back [0. 1. 2. 3. 4.] expected [0. 2. 4. 6. 8.]
--- same repro against the ASan/UBSan probe module ---
==721964==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6eb27e6cd040 at pc 0x70628032901f bp 0x7ffe0908be40 sp 0x7ffe0908b5e8
    #0 0x70628032901e in memcpy ../../../../src/libsanitizer/sanitizer_common/sanitizer_common_interceptors_memintrinsics.inc:117
    #1 0x6c625deb5bb3  (/usr/lib/x86_64-linux-gnu/libhdf5_serial.so.310+0xb5bb3) (BuildId: 130cc2538103f875a0562cce53c427061070a0f3)
    #2 0x6c625e112550 in H5VM_opvv (/usr/lib/x86_64-linux-gnu/libhdf5_serial.so.310+0x312550) (BuildId: 130cc2538103f875a0562cce53c427061070a0f3)
    #3 0x6c625deb370f  (/usr/lib/x86_64-linux-gnu/libhdf5_serial.so.310+0xb370f) (BuildId: 130cc2538103f875a0562cce53c427061070a0f3)
    #4 0x6c625ded15b1  (/usr/lib/x86_64-linux-gnu/libhdf5_serial.so.310+0xd15b1) (BuildId: 130cc2538103f875a0562cce53c427061070a0f3)
    #0 0x70628032b60f in malloc ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:67
    #1 0x6c627b939173  (/usr/lib/python3/dist-packages/numpy/_core/_multiarray_umath.cpython-314-x86_64-linux-gnu.so+0x139173) (BuildId: 0da58c889ef4e9f278c5b5fff9e9a89c2ceb7179)
    #2 0x6c627b93ccea  (/usr/lib/python3/dist-packages/numpy/_core/_multiarray_umath.cpython-314-x86_64-linux-gnu.so+0x13ccea) (BuildId: 0da58c889ef4e9f278c5b5fff9e9a89c2ceb7179)
    #3 0x6c627b9609ba  (/usr/lib/python3/dist-packages/numpy/_core/_multiarray_umath.cpython-314-x86_64-linux-gnu.so+0x1609ba) (BuildId: 0da58c889ef4e9f278c5b5fff9e9a89c2ceb7179)
    #4 0x6c627b96edd7  (/usr/lib/python3/dist-packages/numpy/_core/_multiarray_umath.cpython-314-x86_64-linux-gnu.so+0x16edd7) (BuildId: 0da58c889ef4e9f278c5b5fff9e9a89c2ceb7179)
SUMMARY: AddressSanitizer: heap-buffer-overflow (/usr/lib/x86_64-linux-gnu/libhdf5_serial.so.310+0xb5bb3) (BuildId: 130cc2538103f875a0562cce53c427061070a0f3) 
```

**Cause.** For `ndim == 1` and for shape `(n, 1)`, `qd_python_anycast` builds `arma::Col/Row<T>(buf.ptr, shape[0], false, true)` directly on the numpy buffer and ignores `buf.strides`. The 2-D and 3-D branches go through `qd_python_numpy2arma_Mat/Cube`, which handle strides.

**Proposed fix.** Route the 1-D and (n,1) cases through `qd_python_numpy2arma_Col<T>` (honours strides, copies), converting to `Row` where needed; or first call `py::array_t<T, py::array::f_style | py::array::forcecast>::ensure(arr)`. File: `api_python/python_arma_adapter.hpp`. Combine with B-097, which touches the same function.

**Test for the fix session.** pytest round trip through `hdf5_write_dset` / `hdf5_read_dset` and through `hdf5_write_channel(par=...)` / `hdf5_read_channel` of `x[::2]`, `x[::-1]`, `y[:, 1:2]` and a structured-array field: the values read back equal the input.

## B-006: AVX2 antenna interpolation reads up to 56 bytes past the angle matrices (per-element angles, n_out not a multiple of 8)

- Status: fixed (WP-09)
- Severity: S2, because every `get_channels_spherical(..., use_avx2 = true)` call (and the MEX `use_avx2` flag) with a number of element pairs that is not a multiple of 8 reads past a heap buffer; the values are discarded, so results are right, but the read can fault at a page boundary. `use_avx2` is off by default.
- Kind: memory (heap over-read)
- Mark: [R]
- Where: `src/qd_arrayant_interpolate_avx2.cpp:1364-1377` (general path with `per_element_angles`; float loads 1368/1369, double loads 1374-1377). Reached from `src/get_channels_spherical.cpp:454/459` (angles of size `[n_links, n_out]`) and from `api_mex/get_channels_spherical.cpp` (`use_avx2`).

**What happens.** ASan: `unknown-crash ... READ of size 32` in `_mm256_loadu_pd` / `qd_arrayant_interpolate_avx2`, called from `get_channels_spherical`, "0 bytes after 1600-byte region". A sweep of all kernel paths under ASan reports only lines 1368, 1369 and 1374-1377; the element-vectorised and angle-vectorised paths are clean.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-01/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-01: AVX2 interpolation kernel reads past the end of the angle matrices (per-element angles, n_out % 8 != 0)
// Public path: get_channels_spherical(..., use_avx2 = true) with 2 TX elements, 1 RX element (n_links = 2)
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    auto rx = quadriga_lib::generate_arrayant_omni<double>(10.0);
    auto tx = quadriga_lib::generate_arrayant_xpol<double>(10.0); // 2 elements
    arma::uword n_path = 100;
    arma::mat fbs(3, n_path, arma::fill::randu);
    fbs.row(0) += 5.0;
    arma::mat lbs = fbs;
    arma::vec gain(n_path, arma::fill::ones), len(n_path);
    for (arma::uword p = 0; p < n_path; ++p)
        len(p) = arma::norm(fbs.col(p)) + arma::norm(fbs.col(p) - arma::vec{10.0, 0.0, 0.0}) + 0.1;
    arma::mat M(8, n_path, arma::fill::zeros);
    M.row(0).ones(), M.row(6).fill(-1.0);
    arma::cube cr, ci, dl;
    quadriga_lib::get_channels_spherical<double>(&tx, &rx, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0,
                                                 &fbs, &lbs, &gain, &len, &M, &cr, &ci, &dl, 3e9,
                                                 false, false, nullptr, nullptr, nullptr, nullptr, true);
    std::printf("get_channels_spherical(use_avx2=true) returned, %llu coefficients\n", (unsigned long long)cr.n_elem);
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-01/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-01
g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_asan/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(ASAN_OPTIONS=halt_on_error=1 OMP_NUM_THREADS=1 "$D/repro" 2>&1 | grep -E "ERROR: |SUMMARY|runtime error|^    #[0-4] |^[^ =]" | grep -v "^Shadow\|^  0x\|^=>\|^Address\|^Hint\|^HINT\|^AddressSanitizer can" | cut -c1-200 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
==710910==ERROR: AddressSanitizer: unknown-crash on address 0x78a74ebe1bb0 at pc 0x560c82b22aef bp 0x7fffda737cd0 sp 0x7fffda737cc0
READ of size 32 at 0x78a74ebe1bb0 thread T0
    #0 0x560c82b22aee in _mm256_loadu_pd(double const*) /usr/lib/gcc/x86_64-linux-gnu/12/include/avxintrin.h:893
    #1 0x560c82b22aee in void qd_arrayant_interpolate_avx2<double>(arma::Cube<double> const&, arma::Cube<double> const&, arma::Cube<double> const&, arma::Cube<double> const&, arma::Col<double> const&,
    #2 0x7af752182708 in GOMP_parallel (/usr/lib/x86_64-linux-gnu/libgomp.so.1+0x19708) (BuildId: 2cc7443742f4addce7754a3559f228a07b467315)
    #3 0x560c82b5ab5d in void qd_arrayant_interpolate_avx2<double>(arma::Cube<double> const&, arma::Cube<double> const&, arma::Cube<double> const&, arma::Cube<double> const&, arma::Col<double> const&,
    #4 0x560c8268efc1 in void quadriga_lib::get_channels_spherical<double>(quadriga_lib::arrayant<double> const*, quadriga_lib::arrayant<double> const*, double, double, double, double, double, double,
0x78a74ebe1bc0 is located 0 bytes after 1600-byte region [0x78a74ebe1580,0x78a74ebe1bc0)
allocated by thread T0 here:
    #0 0x7af751b2ae8f in posix_memalign ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:139
    #1 0x560c82660c97 in double* arma::memory::acquire<double>(unsigned long long) /usr/include/armadillo_bits/memory.hpp:72
    #2 0x560c82660c97 in arma::Mat<double>::init_warm(unsigned long long, unsigned long long) /usr/include/armadillo_bits/Mat_meat.hpp:444
    #3 0x560c826929e2 in arma::Mat<double>::set_size(unsigned long long, unsigned long long) /usr/include/armadillo_bits/Mat_meat.hpp:7554
    #4 0x560c826929e2 in void quadriga_lib::get_channels_spherical<double>(quadriga_lib::arrayant<double> const*, quadriga_lib::arrayant<double> const*, double, double, double, double, double, double,
SUMMARY: AddressSanitizer: unknown-crash /usr/lib/gcc/x86_64-linux-gnu/12/include/avxintrin.h:893 in _mm256_loadu_pd(double const*)
exit code 1
```

**Cause.** In the last, partial block of `n_out`, `_mm256_loadu_ps(az_col + o_base)` (and the double variant) loads 8 values. The tail mask (`tail_mask`) covers the stores but not these per-element angle loads, so for the last angle column the load runs past the end of the buffer.

**Proposed fix.** In the general path, when `is_tail_iter && per_element_angles`, load with `_mm256_maskload_ps`; for double copy the `n_valid` values into a zeroed `alignas(32)` temporary, as the angle-vectorised path already does (about lines 1120-1131). File: `src/qd_arrayant_interpolate_avx2.cpp`.

**Test for the fix session.** Catch2: `get_channels_spherical<double>` and `<float>` with `use_avx2 = true`, n_tx = 2, n_rx = 1, 100 paths; assert equality with `use_avx2 = false` within 1e-4. The test detects the bug only in the ASan suite, so it must run there.

## B-007: Generic antenna interpolation reads before the pattern buffer for NaN or infinite angles

- Status: fixed (WP-09)
- Severity: S2, because a NaN or infinite angle (unchecked input) makes the public `arrayant::interpolate` and everything built on it read out of bounds; the output is NaN, so no wrong value is visible, but it is undefined behaviour.
- Kind: UB / memory (heap under-read)
- Mark: [R]
- Where: `src/qd_arrayant_interpolate.cpp:252-273` (bracket search; `i_un = i_up--` at line 270), reached through `arrayant::interpolate`, `rotate_pattern`, `combine_pattern` and the `get_channels_*` functions.

**What happens.** A 5° omni pattern interpolated at azimuth NaN: ASan `heap-buffer-overflow ... READ of size 8` in `qd_arrayant_interpolate` (`qd_arrayant_interpolate.cpp:317`), "296 bytes before 21608-byte region" (296 = 37 elevation samples × 8 bytes); the returned value is NaN. Azimuth ±Inf and elevation ±Inf behave the same. The AVX2 kernel clamps its indices and stays in bounds.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-02/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-02: Generic interpolation kernel reads before the pattern buffer for NaN/Inf angles
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>(5.0); // 37 x 73 grid
    arma::mat az(1, 1), el(1, 1, arma::fill::zeros), vr, vi, hr, hi;
    az(0) = arma::datum::nan;
    ant.interpolate(&az, &el, &vr, &vi, &hr, &hi);
    std::printf("V_re = %g\n", vr(0));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-02/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-02
g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_asan/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(ASAN_OPTIONS=halt_on_error=1 OMP_NUM_THREADS=1 "$D/repro" 2>&1 | grep -E "ERROR: |SUMMARY|runtime error|^    #[0-4] |^[^ =]" | grep -v "^Shadow\|^  0x\|^=>\|^Address\|^Hint\|^HINT\|^AddressSanitizer can" | cut -c1-200 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
==710966==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x725f36de00d8 at pc 0x64e7be71ab5d bp 0x7ffdf643ee30 sp 0x7ffdf643ee20
READ of size 8 at 0x725f36de00d8 thread T0
    #0 0x64e7be71ab5c in void qd_arrayant_interpolate<double>(arma::Cube<double> const&, arma::Cube<double> const&, arma::Cube<double> const&, arma::Cube<double> const&, arma::Col<double> const&, arma
    #1 0x73bf3a39e708 in GOMP_parallel (/usr/lib/x86_64-linux-gnu/libgomp.so.1+0x19708) (BuildId: 2cc7443742f4addce7754a3559f228a07b467315)
    #2 0x64e7be71e0c7 in void qd_arrayant_interpolate<double>(arma::Cube<double> const&, arma::Cube<double> const&, arma::Cube<double> const&, arma::Cube<double> const&, arma::Col<double> const&, arma
    #3 0x64e7be4bedde in quadriga_lib::arrayant<double>::interpolate(arma::Mat<double> const*, arma::Mat<double> const*, arma::Mat<double>*, arma::Mat<double>*, arma::Mat<double>*, arma::Mat<double>*,
    #4 0x64e7be3f3105 in main build_linux/scratch/hunt/final/A-02/repro.cpp:9
0x725f36de00d8 is located 296 bytes before 21608-byte region [0x725f36de0200,0x725f36de5668)
allocated by thread T0 here:
    #0 0x73bf39d2ae8f in posix_memalign ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:139
    #1 0x64e7be49ed15 in double* arma::memory::acquire<double>(unsigned long long) /usr/include/armadillo_bits/memory.hpp:72
    #2 0x64e7be49ed15 in arma::Cube<double>::init_warm(unsigned long long, unsigned long long, unsigned long long) /usr/include/armadillo_bits/Cube_meat.hpp:440
    #3 0x64e7be614bf2 in arma::Cube<double>::set_size(unsigned long long, unsigned long long, unsigned long long) /usr/include/armadillo_bits/Cube_meat.hpp:3936
    #4 0x64e7be614bf2 in arma::Cube<double>::ones(unsigned long long, unsigned long long, unsigned long long) /usr/include/armadillo_bits/Cube_meat.hpp:4289
SUMMARY: AddressSanitizer: heap-buffer-overflow /sjc/quadriga-lib-fixes/src/qd_arrayant_interpolate.cpp:317 in void qd_arrayant_interpolate<double>(arma::Cube<double> const&, arma::Cube<double> const&
exit code 1
```

**Cause.** For a NaN azimuth the comparison `*p_azimuth_grid > az` is false, the search loop does not advance, `i_up` stays 0, and `i_un = i_up--` wraps `i_up` to `SIZE_MAX`; the index `i_up * n_elevation + ...` then points before the buffer. Infinite elevations produce NaN through `dz` and take the same branch.

**Proposed fix.** Guard non-finite angles before the search (set both indices to 0 and write NaN), or replace the decrement by `if (i_up == 0) i_un = 0;`. Alternatively `arrayant::interpolate` rejects non-finite angles with `std::invalid_argument` (an API change; the user decides). File: `src/qd_arrayant_interpolate.cpp`.

**Test for the fix session.** Catch2, run in the ASan suite: `interpolate` with az = {NaN}, {+Inf} and el = {-Inf} gives NaN output (or a clean `invalid_argument`, whichever is decided) and no sanitizer report.

## B-008: `arrayant::append` builds a wrong coupling matrix, and segfaults when `coupling_im` is empty

- Status: fixed (WP-09)
- Severity: S2, because appending any array whose number of ports differs from its number of elements (multibeam without weights applied, a 2-element/1-port array) silently produces a wrong coupling matrix and so wrong channels; a state that `is_valid` accepts (empty `coupling_im`) crashes.
- Kind: wrong result / crash
- Mark: [R]
- Where: `src/qd_arrayant.cpp:156` (`output.set_size(...)` initialises `coupling_re` to `eye(n_el_tot, n_ports_tot)`), 167-172 and 185-190 (only the block columns are copied; off-block identity entries remain; an empty coupling is skipped instead of treated as identity), 171 and 189 (`memcpy` from `coupling_im.colptr(n)` of an empty matrix).

**What happens.** Appending a 2-element/1-port array (coupling [1;1]) to itself gives `coupling_re = [1 0; 1 1; 0 1; 0 1]` instead of the block-diagonal `[1 0; 1 0; 0 1; 0 1]`, and differs from `arrayant_concat_multi` of the same inputs (which is correct) by 1. With `coupling_re = 1` and an empty `coupling_im`, `is_valid(false)` returns "" and `append` then fails with UBSan "null pointer passed as argument 2" at `qd_arrayant.cpp:171` and a SEGV.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-04/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-04: arrayant::append builds a wrong coupling matrix (A-05: segfault when coupling_im is empty)
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    // (A-04) a: 2 elements -> 1 port; append to itself; expected block diagonal [1 0; 1 0; 0 1; 0 1]
    auto a = quadriga_lib::generate_arrayant_omni<double>(10.0);
    a.copy_element(0, 1);
    a.coupling_re = arma::mat(2, 1, arma::fill::ones);
    a.coupling_im = arma::mat(2, 1, arma::fill::zeros);
    auto c = a.append(&a);
    std::printf("A-04 append coupling_re (expected [1 0; 1 0; 0 1; 0 1]):");
    for (arma::uword r = 0; r < c.coupling_re.n_rows; ++r)
        std::printf(" [%g %g]", c.coupling_re(r, 0), c.coupling_re(r, 1));
    std::printf("\n");
    auto cm = quadriga_lib::arrayant_concat_multi(std::vector<quadriga_lib::arrayant<double>>{a}, std::vector<quadriga_lib::arrayant<double>>{a});
    std::printf("A-04 max |append - concat_multi| coupling_re = %g\n", arma::abs(c.coupling_re - cm[0].coupling_re).max());

    // (A-05) coupling_re set, coupling_im empty (accepted by is_valid) -> append crashes
    auto b = quadriga_lib::generate_arrayant_omni<double>(10.0);
    b.coupling_re = arma::mat(1, 1, arma::fill::ones);
    b.coupling_im.reset();
    std::printf("A-05 is_valid(false) = '%s'\n", b.is_valid(false).c_str());
    std::fflush(stdout);
    auto d = b.append(&b);
    std::printf("A-05 append returned\n");
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-04/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-04
g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_asan/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(ASAN_OPTIONS=halt_on_error=1 OMP_NUM_THREADS=1 "$D/repro" 2>&1 | grep -E "ERROR: |SUMMARY|runtime error|^    #[0-4] |^[^ =]" | grep -v "^Shadow\|^  0x\|^=>\|^Address\|^Hint\|^HINT\|^AddressSanitizer can" | cut -c1-200 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
A-04 append coupling_re (expected [1 0; 1 0; 0 1; 0 1]): [1 0] [1 1] [0 1] [0 1]
A-04 max |append - concat_multi| coupling_re = 1
A-05 is_valid(false) = ''
/sjc/quadriga-lib-fixes/src/qd_arrayant.cpp:171:24: runtime error: null pointer passed as argument 2, which is declared to never be null
==711130==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000 (pc 0x795387d97a80 bp 0x7ffe7c466440 sp 0x7ffe7c466068 T0)
    #0 0x795387d97a80 in __memmove_avx_unaligned_erms ../sysdeps/x86_64/multiarch/memmove-vec-unaligned-erms.S:364
    #1 0x598ac477a149 in memcpy /usr/include/x86_64-linux-gnu/bits/string_fortified.h:29
    #2 0x598ac477a149 in quadriga_lib::arrayant<double>::append(quadriga_lib::arrayant<double> const*) const /sjc/quadriga-lib-fixes/src/qd_arrayant.cpp:171
    #3 0x598ac467310b in main build_linux/scratch/hunt/final/A-04/repro.cpp:25
    #4 0x795387c2a600 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:59
rax = 0x00007553853f5fa0  rbx = 0x00007553853f51c0  rcx = 0x00000eaa70a7ebf4  rdx = 0x0000000000000008  
rdi = 0x00007553853f5fa0  rsi = 0x0000000000000000  rbp = 0x00007ffe7c466440  rsp = 0x00007ffe7c466068  
r12 = 0x00007553853f5ea0  r13 = 0x00007553853f5f60  r14 = 0x00007553853f4280  r15 = 0x00007553853f5020  
SUMMARY: AddressSanitizer: SEGV ../sysdeps/x86_64/multiarch/memmove-vec-unaligned-erms.S:364 in __memmove_avx_unaligned_erms
exit code 1
```

**Cause.** `set_size` leaves an identity matrix in place that is not cleared, empty input couplings are not expanded to identity blocks, and `coupling_im` is assumed to exist whenever `coupling_re` does.

**Proposed fix.** In `append`: zero `coupling_re/coupling_im` after `set_size`; fill each block from the input's coupling, or from an identity block when it is empty; treat an empty `coupling_im` as zeros. `arrayant_concat_multi` (`src/qd_arrayant_xio.cpp:795-807`) already does this. File: `src/qd_arrayant.cpp`.

**Test for the fix session.** Catch2: (a) append of two 2-element/1-port arrays gives the block-diagonal coupling and equals `arrayant_concat_multi`; (b) append with an empty `coupling_im` does not crash and gives zero imaginary coupling.

## B-009: `arrayant::rotate_pattern` loses coupling, centre frequency and name

- Status: fixed (WP-09)
- Severity: S2, because a rotated copy of an array with port coupling or a non-default frequency is silently wrong (ports become elements, frequency resets to the default).
- Kind: wrong result
- Mark: [R]
- Where: `src/qd_arrayant.cpp:1875-1876` (`output->set_size(n_el, n_az, n_out, n_out)`; coupling, `center_frequency` and name are never copied to `*output`), `src/qd_arrayant.cpp:1931-1932` (in place with a grid update: `set_size(..., n_out, n_out)` resets the coupling to identity when ports differ from elements).

**What happens.** Input: 2 elements, 1 port, fc = 3.5 GHz, name "myant". `rotate_pattern(0, 0, 45, usage, -1, &out)` for usage 0, 2 and 3 gives `name = "" fc = 2.99792e+08 n_ports = 2`. In place with a non-uniform grid (grid update) gives `n_ports = 2`; in place on a uniform grid keeps 1 port.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-06/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-06: rotate_pattern loses coupling, center frequency and name (output mode, and in place with grid update)
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    auto a = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0, 10.0);
    a.copy_element(0, 1);                              // 2 elements
    a.coupling_re = arma::mat(2, 1, arma::fill::ones); // -> 1 port
    a.coupling_im = arma::mat(2, 1, arma::fill::zeros);
    a.center_frequency = 3.5e9;
    a.name = "myant";
    std::printf("input            : name=%s fc=%g n_ports=%llu\n", a.name.c_str(), a.center_frequency, (unsigned long long)a.n_ports());
    quadriga_lib::arrayant<double> out;
    a.rotate_pattern(0.0, 0.0, 45.0, 0, -1, &out);
    std::printf("output (usage 0) : name=%s fc=%g n_ports=%llu\n", out.name.c_str(), out.center_frequency, (unsigned long long)out.n_ports());
    auto c = a.copy();
    c.azimuth_grid(5) += 0.01; // non-uniform grid -> grid update in rotate_pattern
    c.rotate_pattern(0.0, 0.0, 45.0, 0);
    std::printf("in place + grid update: n_ports=%llu\n", (unsigned long long)c.n_ports());
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-06/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-06
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(OMP_NUM_THREADS=1 "$D/repro" 2>&1 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
input            : name=myant fc=3.5e+09 n_ports=1
output (usage 0) : name=empty fc=2.99792e+08 n_ports=2
in place + grid update: n_ports=2
```

**Cause.** `set_size` with `n_ports = n_out`, and no copy of the metadata.

**Proposed fix.** When writing to `output` with all elements, copy `coupling_re/im`, `center_frequency` and `name`. For in-place grid updates call `set_size(n_el, n_az, n_out, n_ports())` and keep the coupling. If only a subset of elements is written to `output`, document that the coupling becomes identity. File: `src/qd_arrayant.cpp`.

**Test for the fix session.** Catch2: a 2-element/1-port array at 3.5 GHz, rotated into an output and in place with a non-uniform grid; assert that coupling, centre frequency and name are preserved.

## B-010: `arrayant::calc_beamwidth_deg` is wrong when the main lobe wraps ±180° or points towards a pole

- Status: fixed (WP-09)
- Severity: S2, because any element pointing backwards or towards the zenith gets a meaningless beamwidth on valid input.
- Kind: wrong result
- Mark: [R]
- Where: `src/qd_arrayant.cpp:276-327` (azimuth sweep), 329-381 (elevation sweep), `arrayant::calc_beamwidth_deg`.

**What happens.** `generate_arrayant_custom(90, 90, 0, 1)` rotated with `rotate_pattern` (usage 3): boresight gives `bw_az = 89.83, bw_el = 89.86`; heading 150° or 180° gives `bw_az = 360.00`; tilted to the zenith gives `bw_az = 360.00, bw_el = 44.62` (about 90 expected).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-07/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-07: calc_beamwidth_deg is wrong when the main lobe wraps +-180 deg or points to the zenith
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    auto a = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0, 1.0); // 90 deg FWHM in az and el
    double bw_az, bw_el, paz, pel;
    a.calc_beamwidth_deg(0, 3.0, &bw_az, &bw_el, &paz, &pel);
    std::printf("boresight         : bw_az=%.2f bw_el=%.2f point=(%.1f, %.1f)\n", bw_az, bw_el, paz, pel);
    auto b = a.copy();
    b.rotate_pattern(0.0, 0.0, 180.0, 3); // heading 180
    b.calc_beamwidth_deg(0, 3.0, &bw_az, &bw_el, &paz, &pel);
    std::printf("heading 180       : bw_az=%.2f bw_el=%.2f point=(%.1f, %.1f)\n", bw_az, bw_el, paz, pel);
    auto u = a.copy();
    u.rotate_pattern(0.0, -90.0, 0.0, 3); // pointing to zenith
    u.calc_beamwidth_deg(0, 3.0, &bw_az, &bw_el, &paz, &pel);
    std::printf("pointing to zenith: bw_az=%.2f bw_el=%.2f point=(%.1f, %.1f)\n", bw_az, bw_el, paz, pel);
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-07/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-07
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(OMP_NUM_THREADS=1 "$D/repro" 2>&1 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
boresight         : bw_az=89.83 bw_el=89.86 point=(0.0, 0.0)
heading 180       : bw_az=360.00 bw_el=89.86 point=(-180.0, 0.0)
pointing to zenith: bw_az=360.00 bw_el=44.62 point=(-108.0, 90.0)
```

**Cause.** The width is taken as the distance between the first and the last sample above the -3 dB threshold, scanned linearly from the first to the last grid value; there is no circular handling in azimuth and no continuation across the pole in elevation, and it does not restrict itself to the lobe that contains the maximum.

**Proposed fix.** Search outwards from the maximum index until the power drops below the threshold, circularly in azimuth and across the pole in elevation. Alternatively document the limitation. File: `src/qd_arrayant.cpp`.

**Test for the fix session.** Catch2: the custom 90/90 pattern rotated by heading 180°: `bw_az` within 1° of 90; tilted -90°: `bw_el` within 2° of 90.

## B-011: `generate_arrayant_multibeam` with a beam towards el = ±90° returns wrong ports (pol 1) or segfaults (pol 2, 3)

- Status: fixed (WP-09)
- Severity: S2, because a valid beam direction gives a silently wrong array for pol = 1 and a crash for pol = 2 and 3.
- Kind: crash / wrong result
- Mark: [R]
- Where: `src/qd_arrayant_generate.cpp:964` (`gain_new > gain[i_beam]` never holds when the gain is exactly 0, so `cpl_re` stays empty), 1040 (`memcpy(ant.coupling_re.memptr(), cpl_re.memptr(), ...)` from a null pointer), 1048 (`ant.coupling_re = cpl_re` with an empty matrix, which becomes identity coupling).

**What happens.** M = N = 2 with one beam at (0°, 90°): pol = 1 returns `n_elements = 4, n_ports = 4` (one port expected); pol = 2 fails with UBSan "null pointer passed as argument 2" at `qd_arrayant_generate.cpp:1040` and an ASan SEGV.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-08/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-08: generate_arrayant_multibeam with a beam at el = 90 deg: pol=1 returns M*N ports, pol=2 segfaults
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    for (unsigned pol : {1u, 2u})
    {
        std::printf("multibeam M=N=2, one beam at (az=0, el=90), pol=%u\n", pol);
        std::fflush(stdout);
        auto a = quadriga_lib::generate_arrayant_multibeam<double>(2, 2, arma::vec{0.0}, arma::vec{90.0}, arma::vec{1.0}, 3e9, pol);
        std::printf("-> n_elements=%llu n_ports=%llu (expected n_ports=%u)\n", (unsigned long long)a.n_elements(), (unsigned long long)a.n_ports(), pol);
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-08/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-08
g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_asan/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(ASAN_OPTIONS=halt_on_error=1 OMP_NUM_THREADS=1 "$D/repro" 2>&1 | grep -E "ERROR: |SUMMARY|runtime error|^    #[0-4] |^[^ =]" | grep -v "^Shadow\|^  0x\|^=>\|^Address\|^Hint\|^HINT\|^AddressSanitizer can" | cut -c1-200 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
multibeam M=N=2, one beam at (az=0, el=90), pol=1
-> n_elements=4 n_ports=4 (expected n_ports=1)
multibeam M=N=2, one beam at (az=0, el=90), pol=2
/sjc/quadriga-lib-fixes/src/qd_arrayant_generate.cpp:1040:24: runtime error: null pointer passed as argument 2, which is declared to never be null
==711833==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000 (pc 0x7ff69a797a0d bp 0x7ffc29c82230 sp 0x7ffc29c7ab58 T0)
    #0 0x7ff69a797a0d in __memmove_avx_unaligned_erms ../sysdeps/x86_64/multiarch/memmove-vec-unaligned-erms.S:265
    #1 0x5a29be6d06bf in memcpy /usr/include/x86_64-linux-gnu/bits/string_fortified.h:29
    #2 0x5a29be6d06bf in quadriga_lib::arrayant<double> quadriga_lib::generate_arrayant_multibeam<double>(unsigned long long, unsigned long long, arma::Col<double>, arma::Col<double>, arma::Col<double
    #3 0x5a29be3b952b in main build_linux/scratch/hunt/final/A-08/repro.cpp:10
    #4 0x7ff69a62a600 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:59
rax = 0x00007bf697df1140  rbx = 0x00007bf697df1140  rcx = 0x00000f7ed2fbe22b  rdx = 0x0000000000000020  
rdi = 0x00007bf697df1140  rsi = 0x0000000000000000  rbp = 0x00007ffc29c82230  rsp = 0x00007ffc29c7ab58  
r12 = 0x0000000000000008  r13 = 0x0000000000000000  r14 = 0x00007bf697ff93b0  r15 = 0x0000000000000020  
SUMMARY: AddressSanitizer: SEGV ../sysdeps/x86_64/multiarch/memmove-vec-unaligned-erms.S:265 in __memmove_avx_unaligned_erms
exit code 1
```

**Cause.** The element pattern is exactly 0 at the poles (`cos_theta` is forced to 0); the per-beam gain evaluated there never improves on its initial 0, so the beam weights are never stored.

**Proposed fix.** Store the weights of the first iteration unconditionally (or compare with `>=` on the first iteration). File: `src/qd_arrayant_generate.cpp`.

**Test for the fix session.** Catch2: multibeam M = N = 2 with a beam at (0, 90) for pol 1, 2, 3; assert n_ports 1, 2, 2 and no crash (run in the ASan suite as well).

## B-012: qdant writer drops every pattern sample below -100 dB absolute power; a whole low-level element reads back as zeros

- Status: fixed (WP-10)
- Severity: S2: data loss in a writer (S1 by the letter of the scale), but only for samples below an absolute field amplitude of 1e-5, so whole elements are lost only when a pattern is scaled to a very low absolute level; the floor is undocumented.
- Kind: wrong result (writer data loss)
- Mark: [R]
- Where: `src/qd_arrayant_qdant.cpp:512` (`calc_power(..., -100.0, 1.0e-4)`), 477 (`p < th` becomes `-INFINITY`), 514 (an element with nothing above -100 dB is not written), `qd_arrayant_qdant_write`.

**What happens.** A random pattern scaled to a maximum |e_theta| of 2.0e-6 is written and read back as 0 everywhere; a single sample of 3e-6 in an otherwise normal pattern reads back as 0.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-09/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-09: qdant writer silently drops pattern data below -100 dB absolute power
#include "quadriga_lib.hpp"
#include <cstdio>
#include <filesystem>
int main()
{
    std::string fn = "build_linux/scratch/hunt/final/A-09/a09.qdant";
    std::filesystem::remove(fn);
    auto a = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.1, 10.0);
    a.e_theta_re *= 1e-6; // whole pattern about -120 dB
    a.qdant_write(fn);
    auto b = quadriga_lib::qdant_read<double>(fn, 1);
    std::printf("max |e_theta| written %.3g, read back %.3g\n", arma::abs(arma::vectorise(a.e_theta_re)).max(), arma::abs(arma::vectorise(b.e_theta_re)).max());
    std::filesystem::remove(fn);
    auto c = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.1, 10.0);
    c.e_theta_re(1, 0, 0) = 3e-6; // one sample at about -110 dB
    c.qdant_write(fn);
    auto d = quadriga_lib::qdant_read<double>(fn, 1);
    std::printf("sample 3e-6 read back as %g\n", d.e_theta_re(1, 0, 0));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-09/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-09
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(OMP_NUM_THREADS=1 "$D/repro" 2>&1 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
max |e_theta| written 2.01e-06, read back 0
sample 3e-6 read back as 0
```

**Cause.** The file stores magnitudes in dB with a hard floor at -100 dB absolute.

**Proposed fix.** Make the floor relative to the element's peak (e.g. peak - 200 dB), or write all finite dB values and keep -Inf only for exact zeros; document the format either way. File: `src/qd_arrayant_qdant.cpp`. This changes the file content, so the user decides.

**Test for the fix session.** Catch2: write a pattern scaled by 1e-6 and read it back; relative error below 1e-4.

## B-013: `hdf5_read_channel`: a malformed HDF5 file overflows heap and stack buffers (sizes taken from attributes, not from the datasets)

- Status: fixed (WP-02)
- Severity: S2, because reading a malformed or foreign HDF5 file corrupts memory (heap and stack buffer overflows, segfault in the release build); valid files written by the library are not affected.
- Kind: memory (heap and stack buffer overflow, uninitialised data)
- Mark: [R] (crafted files; also found by the phase-0 AFL campaign: 11 crash inputs, all at lines 1838/1839 and 1880/1881)
- Where: `src/qd_channel_hdf5.cpp`, `hdf5_read_channel`: `H5Dread(..., H5S_ALL, H5S_ALL, ...)` into buffers sized from the attributes `NumSnap`, `NumRx`, `NumTx`, `NumPath` at 1663/1664, 1688/1689, 1716/1717, 1744/1745, 1772/1773, 1838/1839, 1853/1854, 1880/1881, 1903, 1925, 1947, 1969, 1991, 2013, 2032/2033, 2048/2049; `char buffer_name[256]` read at 1617-1623; scalar attribute reads at 1628-1646, 1676, 1701, 1729, 1757, 1785, 1821; `hsize_t dims[3]` filled without a rank check at 1869-1871 and in `qHDF_read_data` at 496-499.

**What happens.** A file written by the library and then patched: `NumPath` reduced to 1 gives an ASan heap-buffer-overflow WRITE of size 1600 at `qd_channel_hdf5.cpp:1839`; a 300-byte `Name` attribute gives a stack-buffer-overflow WRITE of size 300 at 1620; a missing `coeff_im` dataset leaves the `fill::none` buffer uninitialised (`coeff_im[0](0) = -0.372549`, written 1) and the channel passes `is_valid`. The release build exits with code 139.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-001/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B-001: HDF5 reader writes file data into buffers sized from attributes (heap/stack overflow on malformed files)
// A valid file is written by hdf5_write, then one item is changed with the HDF5 C API and read back.
#include "quadriga_lib.hpp"
#include <hdf5.h>
#include <cstdio>
static std::string make(const std::string &dir, const char *tag)
{
    std::string fn = dir + "/" + tag + ".hdf5";
    std::remove(fn.c_str());
    quadriga_lib::channel<double> c;
    c.center_frequency = {3e9};
    c.tx_pos = arma::mat(3, 1, arma::fill::ones);
    c.rx_pos = arma::mat(3, 1, arma::fill::zeros);
    c.coeff_re.push_back(arma::cube(4, 2, 50, arma::fill::ones));
    c.coeff_im.push_back(arma::cube(4, 2, 50, arma::fill::ones));
    c.delay.push_back(arma::cube(4, 2, 50, arma::fill::ones));
    quadriga_lib::hdf5_write(&c, fn); // 1 snapshot: data sits in /channel_1
    return fn;
}
int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string dir = argv[1], what = argv[2];
    std::string fn = make(dir, what.c_str());
    hid_t f = H5Fopen(fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t g = H5Gopen2(f, "/channel_1", H5P_DEFAULT);
    if (what == "numpath") // NumPath 50 -> 1: buffer of 8 floats, dataset of 400
    {
        H5Adelete(g, "NumPath");
        hid_t s = H5Screate(H5S_SCALAR), a = H5Acreate2(g, "NumPath", H5T_NATIVE_UINT, s, H5P_DEFAULT, H5P_DEFAULT);
        unsigned v = 1;
        H5Awrite(a, H5T_NATIVE_UINT, &v), H5Aclose(a), H5Sclose(s);
    }
    else if (what == "missing_im") // coeff_im deleted: buffer returned uninitialised, no error
        H5Ldelete(g, "coeff_im", H5P_DEFAULT);
    else if (what == "name300") // Name as 300-byte string: read into char[256]
    {
        H5Adelete(g, "Name");
        std::string nm(300, 'x');
        hid_t t = H5Tcopy(H5T_C_S1);
        H5Tset_size(t, 300);
        hid_t s = H5Screate(H5S_SCALAR), a = H5Acreate2(g, "Name", t, s, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(a, t, nm.c_str()), H5Aclose(a), H5Sclose(s), H5Tclose(t);
    }
    H5Gclose(g), H5Fclose(f);
    auto r = quadriga_lib::hdf5_read_channel<double>(fn);
    std::printf("%s: read without exception, is_valid()='%s', coeff_im[0](0)=%g (written: 1)\n", what.c_str(), r.is_valid().c_str(), r.coeff_im[0](0));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-001/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-001
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
# keep sanitizer reports short: error line, access line, frames reduced to file:line
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|^    #[0-9]+ .*(src/|repro)|$1" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${2:-8}; }
$ASAN "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $H5
$REL "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $H5
for w in numpath name300 missing_im; do
  echo "== ASan: $w"
  ASAN_OPTIONS=halt_on_error=1 "$D/repro_asan" "$D" $w 2>&1 | filt "read without" 5
done
echo "== release build: numpath"
"$D/repro" "$D" numpath > /dev/null 2>&1 || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== ASan: numpath
==712397==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x7a571a5e52b0 at pc 0x7e271d728d2d bp 0x7fff12392af0 sp 0x7fff12392298
WRITE of size 1600 at 0x7a571a5e52b0 thread T0
    #13 /sjc/quadriga-lib-fixes/src/qd_channel_hdf5.cpp:1839
    #14 build_linux/scratch/hunt/final/B-001/repro.cpp:46
    #1 /sjc/quadriga-lib-fixes/src/qd_channel_hdf5.cpp:1838
== ASan: name300
==712404==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x75e3aa1f42c0 at pc 0x79e3adf28d2d bp 0x7ffe006e1740 sp 0x7ffe006e0ee8
WRITE of size 300 at 0x75e3aa1f42c0 thread T0
    #7 /sjc/quadriga-lib-fixes/src/qd_channel_hdf5.cpp:1620
    #8 build_linux/scratch/hunt/final/B-001/repro.cpp:46
    #0 /sjc/quadriga-lib-fixes/src/qd_channel_hdf5.cpp:1564
== ASan: missing_im
missing_im: read without exception, is_valid()='', coeff_im[0](0)=-0.372549 (written: 1)
== release build: numpath
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/B-001//run.sh: line 17: 712420 Segmentation fault         (core dumped) "$D/repro" "$D" numpath > /dev/null 2>&1
exit code 139
```

**Cause.** Buffer sizes come from the attributes, but `H5Dread` with `H5S_ALL` transfers the dataset's own extent; there is no rank or extent check, HDF5 return codes are not checked (a missing dataset leaves the buffer as allocated), and some size products are 32-bit `unsigned`.

**Proposed fix.** A checked read helper: open the dataset, compare rank and every extent (computed in 64 bits) with the expected shape, read, check the return code, close; throw `std::invalid_argument` on any mismatch. A checked scalar-attribute helper. For `Name`, check that the type is a fixed-length string of at most 256 bytes. In `qHDF_read_data` check `ndims <= 3`. File: `src/qd_channel_hdf5.cpp`; no API change.

**Test for the fix session.** Catch2: write a channel, then in turn patch `NumPath` to 1, delete `coeff_im`, replace `delay` by a rank-1 and a rank-4 dataset, set a 300-byte `Name`; each `hdf5_read_channel` throws `std::invalid_argument`, and the ASan suite shows no report. The AFL harness `build_linux/scratch/hunt/fuzz/fz_hdf5.cpp` can be re-run against the fix.

## B-014: HDF5 storage layout: `nx*ny*nz*nw` wraps in 32 bits; `hdf5_write` then reads and writes past the index buffer

- Status: fixed (WP-02)
- Severity: S2, because a layout with more than 2^32 slots is accepted and a later write corrupts memory (segfault in the release build); such layouts are unusual but nothing rejects them.
- Kind: memory (integer overflow)
- Mark: [R]
- Where: `src/qd_channel_hdf5.cpp:848` (`hdf5_create`), 2177-2178 (`hdf5_reshape_layout`), 176/190/194/232 (`qHDF_get_channel_ID`), 939 (`hdf5_read_layout`).

**What happens.** `hdf5_create(fn, 65536, 65537)` and `hdf5_reshape_layout(fn, 65536, 65537)` are accepted, but the `Order` index has only 65536 entries; `hdf5_write` at `iy >= 1` gives ASan READ of size 4 at line 194 and WRITE of size 4 at line 232; the release build exits with code 139.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-002/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B-002: slot count nx*ny*nz*nw wraps in 32 bits; hdf5_reshape_layout accepts it and hdf5_write indexes past 'Order'
#include "quadriga_lib.hpp"
#include <cstdio>
int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string fn = std::string(argv[1]) + "/layout.hdf5";
    std::remove(fn.c_str());
    quadriga_lib::hdf5_create(fn);                       // default layout: 65536 slots
    quadriga_lib::hdf5_reshape_layout(fn, 65536, 65537); // 65536*65537 = 2^32 + 65536 -> accepted
    arma::u32_vec id;
    arma::u32_vec lay = quadriga_lib::hdf5_read_layout(fn, &id);
    std::printf("layout %u x %u, index has %llu entries\n", lay(0), lay(1), (unsigned long long)id.n_elem);
    quadriga_lib::channel<double> c;
    c.tx_pos = arma::mat(3, 1, arma::fill::ones);
    c.rx_pos = arma::mat(3, 1, arma::fill::zeros);
    c.path_gain.push_back(arma::vec(3, arma::fill::ones));
    for (unsigned iy = 1; iy < 3; ++iy)
        std::printf("hdf5_write(ix=0, iy=%u) returned %d\n", iy, quadriga_lib::hdf5_write(&c, fn, 0, iy));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-002/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-002
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
# keep sanitizer reports short: error line, access line, frames reduced to file:line
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|^    #[0-9]+ .*(src/|repro)|$1" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${2:-8}; }
$ASAN -fsanitize-recover=address "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $H5
$REL "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $H5
echo "== release"
"$D/repro" "$D" || echo "exit code $?"
echo "== ASan (halt_on_error=0: first READ and first WRITE report)"
ASAN_OPTIONS=halt_on_error=0 "$D/repro_asan" "$D" 2>&1 | filt "layout|returned" 40 | grep -E "layout|returned|READ of|WRITE of|#0 " | head -8
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== release
layout 65536 x 65537, index has 65536 entries
hdf5_write(ix=0, iy=1) returned 0
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/B-002//run.sh: line 13: 712502 Segmentation fault         (core dumped) "$D/repro" "$D"
exit code 139
== ASan (halt_on_error=0: first READ and first WRITE report)
layout 65536 x 65537, index has 65536 entries
READ of size 4 at 0x7847b517c800 thread T0
    #0 /sjc/quadriga-lib-fixes/src/qd_channel_hdf5.cpp:194
WRITE of size 4 at 0x7447af6d2800 thread T0
    #0 /sjc/quadriga-lib-fixes/src/qd_channel_hdf5.cpp:232
hdf5_write(ix=0, iy=1) returned 0
hdf5_write(ix=0, iy=2) returned 0
```

**Cause.** The slot count is an `unsigned` product that wraps; the storage location is never compared with the index length.

**Proposed fix.** Compute the product in `uint64_t`; throw in `hdf5_create` and `hdf5_reshape_layout` when it exceeds `UINT32_MAX`; check `storage_location < n_order` before indexing. File: `src/qd_channel_hdf5.cpp`. A new error, no change of documented behaviour.

**Test for the fix session.** Catch2: `hdf5_create(fn, 65536, 65537)` and `hdf5_reshape_layout(fn, 65536, 65537)` throw `std::invalid_argument`.

## B-015: `hdf5_write` copies 256 bytes from the channel name: heap memory ends up in the file; names of 256+ characters over-read on reading

- Status: fixed (WP-02)
- Severity: S2, because every HDF5 write reads past the end of the name string (undefined behaviour) and stores up to 255 bytes of unrelated process memory (heap pointers, other data) in the file.
- Kind: memory (heap over-read, information leak into the file)
- Mark: [R] (also in the phase-0 sanitizer suite: `H5Awrite` READ of size 256, "HDF - Minimal Test", `tests/catch2_tests/test_hdf_functions.cpp:114`)
- Where: `src/qd_channel_hdf5.cpp:1073-1077` (`hdf5_write`: `H5Awrite` of `name.c_str()` with a 256-byte fixed-string type), reader at 1617-1623.

**What happens.** A 20-character name: 32 non-zero bytes are stored after the terminating NUL (heap content). With `assume_valid = true` and a 300-character name (otherwise rejected by `is_valid`), the stored name has no NUL and the reader over-reads its stack buffer (ASan READ of size 257 at 1623); the name reads back with length 262. ASan on the writer: READ of size 256 at 1075.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-004/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B-004: hdf5_write copies 256 bytes from the channel name buffer into the file (over-read, memory written to disk);
// a name > 255 characters with assume_valid=true is stored without NUL and the reader over-reads its stack buffer
#include "quadriga_lib.hpp"
#include <hdf5.h>
#include <cstdio>
static void write_and_dump(const std::string &fn, size_t len, bool assume_valid)
{
    std::remove(fn.c_str());
    auto *c = new quadriga_lib::channel<double>();
    c->name = std::string(len, 'N');
    c->tx_pos = arma::mat(3, 1, arma::fill::ones);
    c->rx_pos = arma::mat(3, 1, arma::fill::zeros);
    c->path_gain.push_back(arma::vec(3, arma::fill::ones));
    quadriga_lib::hdf5_write(c, fn, 0, 0, 0, 0, assume_valid);
    delete c;
    hid_t f = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t a = H5Aopen_by_name(f, "/channel_1", "Name", H5P_DEFAULT, H5P_DEFAULT), t = H5Aget_type(a);
    unsigned char raw[256] = {0};
    H5Aread(a, t, raw);
    int nz = 0;
    for (size_t i = len + 1; i < 256; ++i) nz += raw[i] != 0;
    std::printf("name length %zu: %d non-zero bytes stored after the terminating NUL\n", len, nz);
    H5Tclose(t), H5Aclose(a), H5Fclose(f);
    auto r = quadriga_lib::hdf5_read_channel<double>(fn);
    std::printf("name length %zu: read back with length %zu\n", len, r.name.size());
}
int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string fn = std::string(argv[1]) + "/name.hdf5";
    write_and_dump(fn, 20, false);
    write_and_dump(fn, 300, true);
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-004/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-004
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
# keep sanitizer reports short: error line, access line, frames reduced to file:line
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|^    #[0-9]+ .*(src/|repro)|$1" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${2:-8}; }
$REL "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $H5
$ASAN -fsanitize-recover=address "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $H5
echo "== release"
"$D/repro" "$D" || echo "exit code $?"
echo "== ASan (halt_on_error=0)"
ASAN_OPTIONS=halt_on_error=0 "$D/repro_asan" "$D" 2>&1 | filt "name length" 40 | grep -E "name length|ERROR: |READ of|#[0-9]+ /sjc" | sed -E 's/ on address.*//' | head -10
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== release
name length 20: 32 non-zero bytes stored after the terminating NUL
name length 20: read back with length 20
name length 300: 0 non-zero bytes stored after the terminating NUL
name length 300: read back with length 262
== ASan (halt_on_error=0)
==712669==ERROR: AddressSanitizer: heap-buffer-overflow
READ of size 256 at 0x73e1517e1195 thread T0
    #7 /sjc/quadriga-lib-fixes/src/qd_channel_hdf5.cpp:1075
name length 20: 84 non-zero bytes stored after the terminating NUL
name length 20: read back with length 20
name length 300: 0 non-zero bytes stored after the terminating NUL
==712669==ERROR: AddressSanitizer: stack-buffer-overflow
READ of size 257 at 0x73b150bfc2c0 thread T0
    #3 /sjc/quadriga-lib-fixes/src/qd_channel_hdf5.cpp:1623
    #0 /sjc/quadriga-lib-fixes/src/qd_channel_hdf5.cpp:1564
```

**Cause.** `H5Awrite` with a 256-byte file type copies 256 bytes from the pointer it is given, regardless of the string's length.

**Proposed fix.** Copy the name into a zeroed `char[256]`, truncated to 255 characters, and write that buffer; in the reader, read into a zeroed 257-byte buffer (or use the type check of B-013). Optionally reject names over 255 characters even with `assume_valid`. File: `src/qd_channel_hdf5.cpp`.

**Test for the fix session.** Catch2 under ASan: write names of 20 and 255 characters without a report; read the raw attribute and assert all bytes after the NUL are 0; with `assume_valid` and 300 characters the name reads back with at most 255 characters.

## B-016: `qrt_file_read` (C++) fills `aod`, `eod`, `aoa`, `eoa` only when `path_gain` or `path_length` is also requested

- Status: fixed (WP-01)
- Severity: S2, because a C++ caller who requests only the angles silently keeps the previous contents of the output vectors; the MEX and Python wrappers always request `path_length` and are not affected.
- Kind: wrong result (stale output), UB for 0 paths
- Mark: [R]
- Where: `src/qd_channel_qrt.cpp:1515-1564` (`qrt_file_read`): the four copy blocks for the angles sit inside `if (want_length)`; `memcpy` from an empty `path_angles` at 1544, 1550, 1556, 1562 when there are 0 paths.

**What happens.** `tests/data/test_v5.qrt`, `aod` pre-filled with `{12345}`: requesting `aod` alone returns 1 element, `aod(0) = 12345` (stale); with `path_length` also requested it returns 2 elements. With 0 paths UBSan reports "null pointer passed as argument 1/2" at the four `memcpy` lines.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-005/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B-005: qrt_file_read returns aod/eod/aoa/eoa only if path_gain or path_length is also requested
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    const std::string fn = "tests/data/test_v5.qrt";
    arma::vec aod1 = {12345.0}, aod2, pl;
    // angles requested alone
    quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, &aod1);
    // angles requested together with path_length
    quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, &pl, nullptr, &aod2);
    std::printf("aod alone: %llu element(s), aod(0) = %g (stale input value 12345 kept)\n", (unsigned long long)aod1.n_elem, aod1(0));
    std::printf("aod with path_length: %llu element(s)\n", (unsigned long long)aod2.n_elem);
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-005/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-005
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
# keep sanitizer reports short: error line, access line, frames reduced to file:line
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|^    #[0-9]+ .*(src/|repro)|$1" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${2:-8}; }
$REL "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $H5
"$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
aod alone: 1 element(s), aod(0) = 12345 (stale input value 12345 kept)
aod with path_length: 2 element(s)
```

**Cause.** A misplaced closing brace: the angle copies are nested in the path-length block.

**Proposed fix.** Move the four copy blocks after the `want_length` block (inside the angle request), and skip `memcpy` when `no_path == 0`. File: `src/qd_channel_qrt.cpp`.

**Test for the fix session.** Catch2 on `tests/data/test_v5.qrt`: `aod` requested alone equals `aod` requested together with `path_length`.

## B-017: `qrt_file_append` corrupts the file when a CIR orientation is non-zero in double but rounds to 0 in float

- Status: fixed (WP-07)
- Severity: S2, because a valid input (an orientation column of tiny values such as 1e-50 rad) leads to a corrupted QRT file: shifted TX positions, lost paths, zero CIR positions, without an error.
- Kind: wrong result (writer data corruption)
- Mark: [R]
- Where: `src/qd_channel_qrt.cpp:438-450` (`qrt_file_append` re-derives the CIR format `cir_fmt` from the float cache), with `derive_fmt` in `qrt_file_init` at 281-295 (decides on the double values).

**What happens.** A v5 and a v6 file with 3 TX, one CIR orientation column of 1e-50: after appending, TX0 reads back at position [101 200 30] (TX1's), TX2 at [0 0 0]; "TX0: 0 paths read, 3 written"; CIR positions read back as 0; `qrt_file_read_raw` has 2 mismatches.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-007/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 11: qrt_file_init + qrt_file_append + qrt_file_read_raw / qrt_file_read round trips.
// Usage: p11 [tiny]   tiny = CIR orientation column of 1e-50 rad (non-zero in double, 0.0f in float)
#include "quadriga_lib.hpp"
#include <cstdio>
using quadriga_lib::path;
static path mk(unsigned iC, size_t ns, size_t nf, bool sc, float base)
{
    path p(ns, nf, sc);
    p.iC = iC;
    for (size_t s = 0; s < ns; ++s)
        for (int k = 0; k < 3; ++k) p.coord(s)[k] = base + 10.0f * (float)s + (float)k;
    for (size_t f = 0; f < nf; ++f)
        for (size_t k = 0; k < (sc ? 2u : 8u); ++k) p.xpr_coeff(f)[k] = base * 0.01f + (float)(f * 8 + k) * 0.001f;
    std::vector<uint8_t> t(ns);
    for (size_t s = 0; s < ns; ++s) t[s] = (uint8_t)(s + 1);
    p.set_interaction_type_codes(t);
    return p;
}
int main(int argc, char **argv)
{
    bool tiny = argc > 1;
    for (bool sc : {false, true})
    {
        std::string fn = std::string("build_linux/scratch/hunt/final/B-007/p11_") + (sc ? "v6" : "v5") + ".qrt";
        arma::vec freq = sc ? arma::vec{1000.0, 2000.0, 4000.0, 8000.0, 16000.0} : arma::vec{3.5, 28.0};
        size_t nf = freq.n_elem;
        arma::mat cir_pos = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        arma::mat cir_ori(3, 3, arma::fill::zeros);
        if (tiny) cir_ori.col(0).fill(1e-50); else cir_ori.col(2) = arma::vec{0.1, 0.2, 0.3};
        quadriga_lib::qrt_file_init<double>(fn, freq, cir_pos, cir_ori, {"RX"}, {0}, 3, sc);

        std::vector<std::vector<path>> tx(3);
        tx[0].push_back(mk(0, 0, nf, sc, 1)), tx[0].push_back(mk(2, 3, nf, sc, 2)), tx[0].push_back(mk(0, 1, nf, sc, 3));
        // tx[1] empty
        tx[2].push_back(mk(1, 9, nf, sc, 4));
        for (int t = 0; t < 3; ++t)
            quadriga_lib::qrt_file_append<double>(fn, tx[t], arma::vec{100.0 + t, 200.0, 30.0}, arma::vec{0.0, 0.0, 0.5 * t}, "TX" + std::to_string(t));

        auto cache = quadriga_lib::qrt_read_cache_init(fn);
        std::printf("%s: version %d, no_orig %u, no_cir %u, no_freq %u\n", fn.c_str(), cache.version, cache.no_orig, cache.no_cir, cache.no_freq);
        for (int t = 0; t < 3; ++t)
            std::printf("  TX%d read back position [%g %g %g] orientation [%g %g %g]\n", t, cache.orig_pos_all(t, 0), cache.orig_pos_all(t, 1),
                        cache.orig_pos_all(t, 2), cache.orig_orientation(t, 0), cache.orig_orientation(t, 1), cache.orig_orientation(t, 2));
        int bad = 0;
        for (int t = 0; t < 3; ++t)
        {
            auto raw = quadriga_lib::qrt_file_read_raw(fn, t);
            if (raw.size() != tx[t].size()) { std::printf("  TX%d: %zu paths read, %zu written\n", t, raw.size(), tx[t].size()); ++bad; continue; }
            // read_raw returns paths grouped by CIR; written order within a CIR is kept
            std::vector<const path *> exp;
            for (unsigned c = 0; c < cache.no_cir; ++c) for (auto &p : tx[t]) if (p.iC == c) exp.push_back(&p);
            for (size_t i = 0; i < raw.size(); ++i)
            {
                const path &a = *exp[i], &b = raw[i];
                bool ok = a.iC == b.iC && a.n_seg() == b.n_seg() && a.n_freq() == b.n_freq() && a.is_scalar() == b.is_scalar() && a.interaction_type_codes() == b.interaction_type_codes();
                for (size_t s = 0; ok && s < a.n_seg(); ++s) for (int k = 0; k < 3; ++k) ok = ok && a.coord(s)[k] == b.coord(s)[k];
                for (size_t f = 0; ok && f < a.n_freq(); ++f) for (size_t k = 0; k < (sc ? 2u : 8u); ++k) ok = ok && a.xpr_coeff(f)[k] == b.xpr_coeff(f)[k];
                if (!ok) { std::printf("  TX%d path %zu differs\n", t, i); ++bad; }
            }
        }
        std::printf("  read_raw round trip: %d mismatches\n", bad);
        for (unsigned c = 0; c < 3; ++c)
        {
            arma::vec rp, ro; arma::cube M; arma::u32_vec ni;
            quadriga_lib::qrt_file_read<double>(fn, c, 0, true, nullptr, nullptr, nullptr, &rp, &ro, nullptr, nullptr, nullptr, nullptr, &M,
                                                nullptr, nullptr, nullptr, nullptr, nullptr, 0, &ni);
            std::printf("  qrt_file_read TX0 CIR%u: rx_pos [%g %g %g] rx_ori [%g %g %g], %llu paths\n", c, rp(0), rp(1), rp(2), ro(0), ro(1), ro(2), (unsigned long long)ni.n_elem);
        }
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-007/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-007
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$ASAN "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $H5
"$D/repro_asan" tiny 2>&1 | head -20 || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
build_linux/scratch/hunt/final/B-007/p11_v5.qrt: version 5, no_orig 3, no_cir 3, no_freq 2
  TX0 read back position [101 200 30] orientation [0 0 0]
  TX1 read back position [102 200 30] orientation [0 0 0]
  TX2 read back position [0 0 0] orientation [0 0 0]
  TX0: 0 paths read, 3 written
  TX2: 0 paths read, 1 written
  read_raw round trip: 2 mismatches
  qrt_file_read TX0 CIR0: rx_pos [0 0 0] rx_ori [0 0 0], 0 paths
  qrt_file_read TX0 CIR1: rx_pos [0 0 0] rx_ori [0 0 0], 0 paths
  qrt_file_read TX0 CIR2: rx_pos [0 0 0] rx_ori [0 0 0], 0 paths
build_linux/scratch/hunt/final/B-007/p11_v6.qrt: version 6, no_orig 3, no_cir 3, no_freq 5
  TX0 read back position [101 200 30] orientation [0 0 0]
  TX1 read back position [102 200 30] orientation [0 0 0]
  TX2 read back position [0 0 0] orientation [0 0 0]
  TX0: 0 paths read, 3 written
  TX2: 0 paths read, 1 written
  read_raw round trip: 2 mismatches
  qrt_file_read TX0 CIR0: rx_pos [0 0 0] rx_ori [0 0 0], 0 paths
  qrt_file_read TX0 CIR1: rx_pos [0 0 0] rx_ori [0 0 0], 0 paths
  qrt_file_read TX0 CIR2: rx_pos [0 0 0] rx_ori [0 0 0], 0 paths
```

**Cause.** `qrt_file_init` writes the format byte based on the double data (orientation present), `qrt_file_append` derives it again from float data (1e-50 becomes 0.0f, orientation absent); every later offset is off by `no_cir * 4` bytes.

**Proposed fix.** In `qrt_file_append` read the `cir_fmt` byte from the file (offset `11 + 16 + 4*no_freq`); in `derive_fmt` test `(float)p[i] != 0.0f` so that both decide alike. File: `src/qd_channel_qrt.cpp`.

**Test for the fix session.** Catch2: `qrt_file_init` with an orientation column of 1e-50, append paths for 3 TX, then `qrt_file_read_raw` returns the written paths and the TX positions read back as written.

## B-018: OBJ round trip loses "no material": faces with `mtl_ind = 0` inherit the previous material

- Status: fixed (WP-14): a no-material face after a material face in the same object is written after `usemtl (null)`, which the reader maps back to 0; face order and objects unchanged; a used material named `(null)` is rejected by the writer (round 2)
- Severity: S2, because writing and reading a mesh changes its materials without an error whenever a face without material follows one with material.
- Kind: wrong result (writer data loss)
- Mark: [R]
- Where: `src/obj_file_write.cpp:~829-845` (`usemtl` is written only for faces with a material).

**What happens.** `mtl_ind` written `{1 0 2}` reads back `{1 1 2}`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-008/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 15: OBJ writer/reader round trips and parser edge cases.
#include "quadriga_lib.hpp"
#include <cstdio>
#include <fstream>
int main()
{
    std::string dir = "build_linux/scratch/hunt/final/B-008/";
    // (a) faces without material (mtl_ind = 0) following a face with material, same object
    {
        arma::mat mesh = {{0, 0, 0, 1, 0, 0, 0, 1, 0}, {5, 0, 0, 6, 0, 0, 5, 1, 0}, {9, 0, 0, 10, 0, 0, 9, 1, 0}};
        arma::uvec obj_ind = {0, 0, 0}, mtl_ind = {1, 0, 2};
        std::vector<std::string> obj_names = {"box"}, mtl_names = {"itu_wood", "itu_glass"};
        quadriga_lib::obj_file_write<double>(dir + "p15a.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names);
        arma::uvec mi;
        quadriga_lib::obj_file_read<double>(dir + "p15a.obj", nullptr, nullptr, nullptr, nullptr, nullptr, &mi);
        std::printf("(a) mtl_ind written {1 0 2}, read back {%llu %llu %llu}\n", (unsigned long long)mi(0), (unsigned long long)mi(1), (unsigned long long)mi(2));
    }
    // (b) .mtl with Windows line endings: BSDF of the matching material
    {
        std::ofstream o(dir + "p15b.obj", std::ios::binary);
        o << "mtllib p15b.mtl\r\no a\r\nv 0 0 0\r\nv 1 0 0\r\nv 0 1 0\r\nusemtl red\r\nf 1 2 3\r\n";
        std::ofstream m(dir + "p15b.mtl", std::ios::binary);
        m << "newmtl red\r\nKd 1 0 0\r\n";
        o.close(), m.close();
        arma::mat bsdf; std::vector<std::string> mn;
        quadriga_lib::obj_file_read<double>(dir + "p15b.obj", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &mn, &bsdf);
        std::printf("(b) CRLF: material '%s', bsdf %llux%llu", mn[0].c_str(), (unsigned long long)bsdf.n_rows, (unsigned long long)bsdf.n_cols);
        if (bsdf.n_rows) std::printf(", Kd = %g %g %g", bsdf(0, 0), bsdf(0, 1), bsdf(0, 2));
        std::printf("  (expected 1x17, Kd = 1 0 0)\n");
    }
    // (c) relative (negative) vertex indices, valid OBJ syntax
    {
        std::ofstream o(dir + "p15c.obj");
        o << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n";
        o.close();
        arma::umat fi; arma::mat mesh;
        try { quadriga_lib::obj_file_read<double>(dir + "p15c.obj", nullptr, nullptr, &fi); std::printf("(c) face_ind only: [%llu %llu %llu]\n", (unsigned long long)fi(0, 0), (unsigned long long)fi(0, 1), (unsigned long long)fi(0, 2)); }
        catch (const std::exception &e) { std::printf("(c) face_ind only: exception %s\n", e.what()); }
        try { quadriga_lib::obj_file_read<double>(dir + "p15c.obj", &mesh); std::printf("(c) mesh: ok\n"); }
        catch (const std::exception &e) { std::printf("(c) mesh: exception %s\n", e.what()); }
    }
    // (d) out-of-range vertex index, face_ind requested without mesh
    {
        std::ofstream o(dir + "p15d.obj");
        o << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 99\n";
        o.close();
        arma::umat fi;
        try { quadriga_lib::obj_file_read<double>(dir + "p15d.obj", nullptr, nullptr, &fi); std::printf("(d) face_ind only: accepted, [%llu %llu %llu] with 3 vertices\n", (unsigned long long)fi(0, 0), (unsigned long long)fi(0, 1), (unsigned long long)fi(0, 2)); }
        catch (const std::exception &e) { std::printf("(d) face_ind only: exception %s\n", e.what()); }
        try { arma::mat mesh; quadriga_lib::obj_file_read<double>(dir + "p15d.obj", &mesh); std::printf("(d) mesh: accepted\n"); }
        catch (const std::exception &e) { std::printf("(d) mesh: exception %s\n", e.what()); }
    }
    // (e) count-only call on a quad file
    {
        std::ofstream o(dir + "p15e.obj");
        o << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
        o.close();
        try { auto n = quadriga_lib::obj_file_read<double>(dir + "p15e.obj"); std::printf("(e) quad file, no outputs: returns %llu\n", (unsigned long long)n); }
        catch (const std::exception &e) { std::printf("(e) no outputs: exception %s\n", e.what()); }
        try { arma::mat mesh; quadriga_lib::obj_file_read<double>(dir + "p15e.obj", &mesh); std::printf("(e) mesh: ok\n"); }
        catch (const std::exception &e) { std::printf("(e) quad file, mesh requested: exception %s\n", e.what()); }
    }
    // (f) material-name matching: "my.material" (no numeric suffix) in default table lookup
    {
        std::ofstream o(dir + "p15f.obj");
        o << "v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl itu_glass.blue\nf 1 2 3\n";
        o.close();
        arma::uvec ci; std::vector<std::string> cn;
        quadriga_lib::obj_file_read<double>(dir + "p15f.obj", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "", &ci, &cn);
        std::printf("(f) 'itu_glass.blue' -> csv_ind %llu (%s)\n", (unsigned long long)ci(0), ci(0) ? cn[ci(0) - 1].c_str() : "none");
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-008/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-008
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$ASAN "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $H5
"$D/repro_asan" 2>&1 | grep -E "^\(a\)" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
(a) mtl_ind written {1 0 2}, read back {1 1 2}
```

**Cause.** The OBJ format has no "no material" statement; a face without `usemtl` keeps the previous material in the reader.

**Proposed fix.** Within each object write the faces with `mtl_ind == 0` before the first `usemtl` (stable reordering), or start a new `o` block for them; the face order then changes, which must be documented. File: `src/obj_file_write.cpp`.

**Test for the fix session.** Catch2: write `mtl_ind {1, 0, 2}` and read back `{1, 0, 2}` (up to a documented reordering).

## B-019: `baseband_freq_response_multi` gives wrong phases for `get_channels_*` output with the default relative delays

- Status: documented (WP-04)
- Severity: S2, because the documented combination ("`remove_delay_phase` must be true for output from `get_channels_multifreq`") returns phases off by up to π when the channels were generated with the default `use_absolute_delays = false`; magnitudes are right.
- Kind: wrong result
- Mark: [R]
- Where: `src/baseband_freq_response.cpp:400-403` (documentation), `:653-667` (undo of the delay phase), together with `src/get_channels_multifreq.cpp:824-833` (coefficients carry the absolute phase, `delay` is relative).

**What happens.** LOS plus one NLOS path, LOS distance d0 = 30 m, `freq_in` 2-3 GHz in 100 MHz steps, `freq_out` in 1 MHz steps, truth = `get_channels_multifreq` evaluated directly at `freq_out`: with absolute delays max |H - H_true| = 9.8e-8; with relative delays 2.000 (magnitude error only 3e-7).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-2/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-2: baseband_freq_response_multi gives a wrong phase for get_channels_multifreq output with relative delays
#include "quadriga_lib.hpp"
#include <cstdio>
#include <complex>
int main()
{
    auto omni = quadriga_lib::generate_arrayant_omni<double>(5.0);
    std::vector<quadriga_lib::arrayant<double>> va = {omni};
    arma::mat fbs(3, 2); // LOS + one single-bounce path, TX (0,0,2), RX (30,0,1.5)
    fbs.col(0) = arma::vec({15.0, 0.0, 1.75}); fbs.col(1) = arma::vec({10.0, 12.0, 2.0});
    arma::vec len = {0.0, 0.0};
    arma::vec fin = arma::regspace<arma::vec>(2.0e9, 0.1e9, 3.0e9), fout = arma::regspace<arma::vec>(2.0e9, 1.0e6, 3.0e9);
    auto mk = [](arma::uword nf, arma::mat &pg, arma::cube &M) {
        pg.set_size(2, nf); pg.row(0).fill(1.0); pg.row(1).fill(0.5);
        M.zeros(8, 2, nf); M.tube(0, 0).fill(1.0); M.tube(6, 0).fill(-1.0); };
    arma::mat pg, pgo; arma::cube M, Mo;
    mk(fin.n_elem, pg, M); mk(fout.n_elem, pgo, Mo);
    for (int absd = 1; absd >= 0; --absd)
    {
        std::vector<arma::cube> cr, ci, dl, tr, ti, td;
        quadriga_lib::get_channels_multifreq<double>(va, va, 0, 0, 2, 0, 0, 0, 30, 0, 1.5, 0, 0, 0, fbs, fbs, pg, len, M, fin, fin, cr, ci, dl, absd == 1);
        quadriga_lib::get_channels_multifreq<double>(va, va, 0, 0, 2, 0, 0, 0, 30, 0, 1.5, 0, 0, 0, fbs, fbs, pgo, len, Mo, fout, fout, tr, ti, td, absd == 1);
        arma::cx_cube h;
        quadriga_lib::baseband_freq_response_multi<double>(cr, ci, dl, fin, fout, nullptr, nullptr, &h, true);
        double err = 0.0, err_mag = 0.0;
        for (arma::uword k = 0; k < fout.n_elem; ++k)
        {
            std::complex<double> ht = std::complex<double>(tr[k](0), ti[k](0)) + std::complex<double>(tr[k](1), ti[k](1));
            err = std::max(err, std::abs(h(0, 0, k) - ht));
            err_mag = std::max(err_mag, std::abs(std::abs(h(0, 0, k)) - std::abs(ht)));
        }
        printf("use_absolute_delays = %d: max |H_multi - H_true| = %.3e, max ||H_multi| - |H_true|| = %.3e\n", absd, err, err_mag);
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-2/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-2
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
use_absolute_delays = 1: max |H_multi - H_true| = 9.813e-08, max ||H_multi| - |H_true|| = 9.357e-08
use_absolute_delays = 0: max |H_multi - H_true| = 2.000e+00, max ||H_multi| - |H_true|| = 2.980e-07
```

**Cause.** The coefficients carry `exp(-j2πf·τ_abs)` but `delay` holds `τ_abs - d0/c`. `remove_delay_phase` removes only the relative part; the residual `exp(-j2πf·d0/c)` is then SLERP-interpolated across the coarse `freq_in` grid, which aliases.

**Proposed fix.** User decision: (a, recommended) require absolute delays and say so in the docs, detecting the relative case (or adding a `los_delay` argument so that the undo uses `τ + d0/c`); (b) let `get_channels_*` use the same delay convention in the phase as in `delay`, which changes a convention used everywhere. Files: `src/baseband_freq_response.cpp` (and docs of the `get_channels_*` functions).

**Test for the fix session.** Catch2 from the repro: max |H_multi - H_true| < 1e-5 for both delay modes, or a clean error in the relative case.

## B-020: `baseband_freq_response`: phase computed in single precision, error grows with delay × bandwidth (both builds)

- Status: fixed (WP-04)
- Severity: S2, because valid long delays (absolute delays, km links, satellites) give visibly wrong frequency responses: 0.064 at 2 ms, 0.31 at 120 ms for 100 MHz bandwidth; the documentation mentions single precision but not this.
- Kind: wrong result
- Mark: [R]
- Where: `src/baseband_freq_response.cpp:146-147` (phasor in float), `src/quadriga_lib_generic_functions.hpp:107` and `src/baseband_freq_response_avx2.cpp:57` (`theta = phasor * delay` in float).

**What happens.** One path, coefficient 1, bandwidth 100 MHz, pilot 1.0, exact reference `exp(-j2πBpτ)`: |H - H_exact| = 1.2e-5 at τ = 1 µs, 1.5e-3 at 100 µs, 0.064 at 2 ms, 0.313 at 0.12 s, identical in the AVX2 and generic builds. `baseband_freq_response_multi` already computes the phase in double.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-3/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-3: baseband_freq_response: single-precision phase, error grows with tau*bandwidth (both builds)
#include "quadriga_lib.hpp"
#include <cstdio>
#include <complex>
int main()
{
    const double B = 100e6; // one path, coefficient 1, pilot 1.0: exact H = exp(-j*2*pi*B*tau), |H| = 1
    for (double tau : {1e-6, 1e-4, 2e-3, 0.12})
    {
        arma::cube cr(1, 1, 1, arma::fill::ones), ci(1, 1, 1, arma::fill::zeros), dl(1, 1, 1);
        dl(0) = tau;
        arma::vec pg = {1.0};
        arma::cx_cube h;
        quadriga_lib::baseband_freq_response<double>(&cr, &ci, &dl, &pg, B, nullptr, nullptr, &h);
        std::complex<double> ref = std::polar(1.0, -2.0 * M_PI * B * tau);
        printf("tau = %-6g s: |H| = %-12.6g |H - H_exact| = %.3g\n", tau, std::abs(h(0)), std::abs(h(0) - ref));
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-3/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-3
for L in build_linux build_generic; do
  g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro_${L}" $L/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
  echo "== library $L"; OMP_NUM_THREADS=1 "$D/repro_${L}" || echo "exit code $?"
done
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== library build_linux
tau = 1e-06  s: |H| = 1            |H - H_exact| = 1.18e-05
tau = 0.0001 s: |H| = 1            |H - H_exact| = 0.00151
tau = 0.002  s: |H| = 1            |H - H_exact| = 0.0636
tau = 0.12   s: |H| = 1            |H - H_exact| = 0.313
== library build_generic
tau = 1e-06  s: |H| = 1            |H - H_exact| = 1.18e-05
tau = 0.0001 s: |H| = 1            |H - H_exact| = 0.00151
tau = 0.002  s: |H| = 1            |H - H_exact| = 0.0636
tau = 0.12   s: |H| = 1            |H - H_exact| = 0.313
```

**Cause.** The product `B·p·τ` is formed in float; its absolute error grows with its magnitude, and the phase is that product times 2π.

**Proposed fix.** Compute `x = B·p·τ` in double, reduce it to `x - round(x)` and only then convert to float for the sin/cos kernel, in both kernels. This also removes the AVX2 overflow in B-026. Files: `src/baseband_freq_response.cpp`, `src/baseband_freq_response_avx2.cpp`, `src/quadriga_lib_generic_functions.hpp`.

**Test for the fix session.** Catch2: τ = 0.12 s, B = 1e8, pilots {0.5, 1}: error below 1e-4 in both builds.

## B-021: IEEE indoor model: `n_floors > 0` shifts all tap delays (3-D path length minus 2-D distance)

- Status: fixed (WP-11)
- Severity: S2, because any multi-floor scenario gets all taps delayed by a few to tens of ns, so the first tap is not at 0.
- Kind: wrong result
- Mark: [R]
- Where: `src/get_channels_ieee_indoor.cpp:462` (`Dist` includes `Rz = 3·n_floors`) against `:476, :567` (`get_channels_planar` is called with z = 0, so its `dist_rx_tx` is 2-D and is subtracted from a 3-D path length).

**What happens.** Model B, 5 m, seed 7: delay of the first tap 0.000 ns (0 floors), 2.772 ns (1 floor), 9.374 ns (2 floors), 26.685 ns (4 floors); all later taps are offset by the same amount (same at 5.25 GHz and 0.9 GHz).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-5/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-5: get_channels_ieee_indoor: n_floors > 0 shifts all tap delays
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);
    for (arma::uword nf : {0, 1, 2, 4})
    {
        auto ch = quadriga_lib::get_channels_ieee_indoor(omni, omni, "B", 5.25e9, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2, {5.0}, {nf},
                                                         false, {}, 20, 50.0, 7);
        printf("model B, 5 m, n_floors = %llu: delay of tap 0 = %6.3f ns, tap 1 = %6.3f ns (expected 0 and 10)\n",
               (unsigned long long)nf, ch[0].delay[0](0) * 1e9, ch[0].delay[0](1) * 1e9);
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-5/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-5
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
model B, 5 m, n_floors = 0: delay of tap 0 =  0.000 ns, tap 1 = 10.000 ns (expected 0 and 10)
model B, 5 m, n_floors = 1: delay of tap 0 =  2.772 ns, tap 1 = 12.772 ns (expected 0 and 10)
model B, 5 m, n_floors = 2: delay of tap 0 =  9.374 ns, tap 1 = 19.374 ns (expected 0 and 10)
model B, 5 m, n_floors = 4: delay of tap 0 = 26.685 ns, tap 1 = 36.685 ns (expected 0 and 10)
```

**Proposed fix.** Pass `Rz` to both `get_channels_planar` calls, or build `path_length` from the same 2-D distance. Whether floors add delay at all is a model question for the user. File: `src/get_channels_ieee_indoor.cpp`.

**Test for the fix session.** Catch2: `delay[0](0) == 0` for `n_floors` 1..4.

## B-022: IEEE indoor model prunes taps with an absolute power threshold after path loss (taps vanish with distance, floors, walls)

- Status: fixed (WP-11)
- Severity: S2, because the number of taps (and so the delay spread) shrinks with distance and wall/floor loss, and in strong-loss cases all taps vanish and the call fails with a misleading error.
- Kind: wrong result
- Mark: [R]
- Where: `src/ieee_channel_model_functions.hpp:750` (`p_path > 1.0e-14`, applied after the path-loss and shadow-fading scaling at line 745).

**What happens.** Model F, no shadow fading: without floors 41-42 paths at every distance; with 2 floors 41 paths at 30 m, 36 at 40 m, 24 at 60 m, 2 at 100 m; with 4 floors at 40 m all paths are removed and the call throws "Inputs 'aod', 'eod', ... must have the same number of columns (n_paths)."

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-6/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-6: get_channels_ieee_indoor prunes taps with an absolute power threshold applied after path loss
#include "quadriga_lib.hpp"
#include <cstdio>
#include <cmath>
int main()
{
    auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);
    for (double d : {10.0, 30.0, 40.0, 60.0, 100.0})
        for (arma::uword nf : {0, 2, 4})
            try
            {
                auto ch = quadriga_lib::get_channels_ieee_indoor(omni, omni, "F", 5.25e9, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2, {d}, {nf},
                                                                 false, {}, 20, 0.0, 7, NAN, NAN, 0.0, 0.0);
                printf("model F, SF = 0, d = %5.0f m, %llu floors: n_path = %llu\n", d, (unsigned long long)nf,
                       (unsigned long long)ch[0].path_gain[0].n_elem);
            }
            catch (const std::exception &e)
            {
                printf("model F, SF = 0, d = %5.0f m, %llu floors: exception: %.80s\n", d, (unsigned long long)nf, e.what());
            }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-6/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-6
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
model F, SF = 0, d =    10 m, 0 floors: n_path = 42
model F, SF = 0, d =    10 m, 2 floors: n_path = 41
model F, SF = 0, d =    10 m, 4 floors: n_path = 29
model F, SF = 0, d =    30 m, 0 floors: n_path = 41
model F, SF = 0, d =    30 m, 2 floors: n_path = 41
model F, SF = 0, d =    30 m, 4 floors: n_path = 6
model F, SF = 0, d =    40 m, 0 floors: n_path = 41
model F, SF = 0, d =    40 m, 2 floors: n_path = 36
model F, SF = 0, d =    40 m, 4 floors: exception: Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length', and 'M' must have
model F, SF = 0, d =    60 m, 0 floors: n_path = 41
model F, SF = 0, d =    60 m, 2 floors: n_path = 24
model F, SF = 0, d =    60 m, 4 floors: exception: Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length', and 'M' must have
model F, SF = 0, d =   100 m, 0 floors: n_path = 41
model F, SF = 0, d =   100 m, 2 floors: n_path = 2
model F, SF = 0, d =   100 m, 4 floors: exception: Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length', and 'M' must have
```

**Proposed fix.** Apply the threshold to the normalised power-delay profile before path loss and shadow fading (a relative threshold). File: `src/ieee_channel_model_functions.hpp`.

**Test for the fix session.** Catch2: the number of paths is independent of distance, floors and walls; no exception at 100 m with 4 floors and 6 walls.

## B-023: IEEE indoor model uplink conjugates the channel (Doppler sign reversed), unlike the reciprocity of `get_channels_spherical` (user decision)

- Status: fixed (WP-11): the uplink is the plain transpose; the downlink Doppler sign was also wrong (opposite to get_channels_spherical) and was flipped to exp(+j2 pi f t), decided with the user during WP-11 (2026-10-20). Sections 3.8 and 3.9 of the published paper 11-25-2318 now differ from the code; no published result used uplink channels
- Severity: S2, because with a moving station the uplink channel has the opposite Doppler sign of a reciprocal channel; it is consistent with its own test, so the user must decide which convention is intended.
- Kind: wrong result / cross-function inconsistency
- Mark: [R]
- Where: `src/get_channels_ieee_indoor.cpp:711-716` (`p_cf_im = -imag`); `tests/catch2_tests/test_qd_ieee_chan.cpp:781-900` encodes `H_up(f) = H_down(-f)^H`.

**What happens.** `max |H_ul^T - H_dl| = 1.5e-3`, `max |H_ul^T - conj(H_dl)| = 0`; phase step per snapshot of path 1: downlink -0.1480 rad, uplink +0.1480 rad. `get_channels_spherical` with TX and RX swapped (and Mᵀ) gives `H_ul = H_dlᵀ` to 1.6e-16.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-7/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-7: get_channels_ieee_indoor uplink = conj(downlink)^T instead of downlink^T (Doppler sign reversed)
#include "quadriga_lib.hpp"
#include <cstdio>
#include <complex>
int main()
{
    auto ap = quadriga_lib::generate_arrayant_ula<double>(2, 5.25e9, 0.5);
    auto sta = quadriga_lib::generate_arrayant_ula<double>(1, 5.25e9, 0.5);
    auto dl = quadriga_lib::get_channels_ieee_indoor(ap, sta, "B", 5.25e9, 10e-9, 1, 0.01, 1e-3, 5.0, 0.0, {3.0}, {0}, false, {}, 20, 0.0, 5);
    auto ul = quadriga_lib::get_channels_ieee_indoor(ap, sta, "B", 5.25e9, 10e-9, 1, 0.01, 1e-3, 5.0, 0.0, {3.0}, {0}, true, {}, 20, 0.0, 5);
    auto c = [](const quadriga_lib::channel<double> &ch, size_t s, arma::uword r, arma::uword t, arma::uword p)
    { return std::complex<double>(ch.coeff_re[s](r, t, p), ch.coeff_im[s](r, t, p)); };
    double eT = 0.0, eH = 0.0;
    for (size_t s = 0; s < dl[0].coeff_re.size(); ++s)
        for (arma::uword p = 0; p < dl[0].coeff_re[s].n_slices; ++p)
            for (arma::uword t = 0; t < 2; ++t)
            {
                eT = std::max(eT, std::abs(c(ul[0], s, t, 0, p) - c(dl[0], s, 0, t, p)));
                eH = std::max(eH, std::abs(c(ul[0], s, t, 0, p) - std::conj(c(dl[0], s, 0, t, p))));
            }
    printf("max |H_ul^T - H_dl| = %.3e (reciprocity, as get_channels_spherical obeys)\n", eT);
    printf("max |H_ul^T - conj(H_dl)| = %.3e\n", eH);
    printf("phase step per snapshot of path 1: downlink %+.4f rad, uplink %+.4f rad\n",
           std::arg(c(dl[0], 1, 0, 0, 1) / c(dl[0], 0, 0, 0, 1)), std::arg(c(ul[0], 1, 0, 0, 1) / c(ul[0], 0, 0, 0, 1)));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-7/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-7
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
max |H_ul^T - H_dl| = 1.533e-03 (reciprocity, as get_channels_spherical obeys)
max |H_ul^T - conj(H_dl)| = 0.000e+00
phase step per snapshot of path 1: downlink -0.1480 rad, uplink +0.1480 rad
```

**Proposed fix.** If reciprocity is the intent: transpose without conjugating in the uplink branch and update the test. File: `src/get_channels_ieee_indoor.cpp`, `tests/catch2_tests/test_qd_ieee_chan.cpp`.

**Test for the fix session.** Catch2: `H_ul == H_dl.t()` (per snapshot) for a moving station.

## B-024: `xpr_update`: data race when `ray_index` contains duplicates (wrong, non-deterministic result with threads)

- Status: fixed (WP-13): duplicate `ray_index` entries are applied in update order (single-thread result); unique indices keep the parallel loop
- Severity: S2, because duplicate indices are allowed by the documentation (only `ray_index < n_ray` is required) and give a wrong, run-to-run different result as soon as the update count reaches the OpenMP threshold.
- Kind: race
- Mark: [R]
- Where: `src/update_xpr.cpp:112` (`QD_XPR_OMP_MIN` = 2048) and the parallel loop at the end of `xpr_update`.

**What happens.** 200 000 updates, all to column 0, each a rotation by 1e-5 rad: 1 thread gives X(0,0) = -0.416146837 (= cos 2, correct); 16 threads give values around 0.98 that differ from run to run (0.9804 and 0.9819 in phase 1, 0.9830 and 0.9885 in the re-run below).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-8/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-8: xpr_update: data race when ray_index contains duplicate entries (>= 2048 updates)
#include "quadriga_lib.hpp"
#include <cstdio>
#include <cmath>
#include <omp.h>
int main()
{
    arma::uword N = 200000;
    arma::uvec idx(N, arma::fill::zeros); // all updates target column 0
    arma::mat U(8, N, arma::fill::zeros);  // each update: rotation by 1e-5 rad, product = rotation by 2 rad
    for (arma::uword i = 0; i < N; ++i)
        U(0, i) = std::cos(1e-5), U(2, i) = std::sin(1e-5), U(4, i) = -std::sin(1e-5), U(6, i) = std::cos(1e-5);
    arma::mat X(8, 4, arma::fill::zeros);
    quadriga_lib::xpr_update<double>(X, nullptr, nullptr, true); // identity
    quadriga_lib::xpr_update<double>(X, &U, nullptr, false, false, false, &idx);
    printf("%2d threads: X(0,0) = %.9f, X(2,0) = %.9f (expected cos(2) = %.9f, sin(2) = %.9f)\n", omp_get_max_threads(),
           X(0, 0), X(2, 0), std::cos(2.0), std::sin(2.0));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-8/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-8
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
for T in 1 16 16; do OMP_NUM_THREADS=$T "$D/repro" || echo "exit code $?"; done
```

Output of the phase-2 re-run (`bash run.sh`):
```text
 1 threads: X(0,0) = -0.416146837, X(2,0) = 0.909297427 (expected cos(2) = -0.416146837, sin(2) = 0.909297427)
16 threads: X(0,0) = 0.982990049, X(2,0) = 0.188768110 (expected cos(2) = -0.416146837, sin(2) = 0.909297427)
16 threads: X(0,0) = 0.988495421, X(2,0) = 0.159726642 (expected cos(2) = -0.416146837, sin(2) = 0.909297427)
```

**Cause.** The parallel loop updates `X(:, ray_index(i))` in place without synchronisation; duplicate indices make different threads read-modify-write the same column.

**Proposed fix.** Either document that indices must be unique and reject duplicates (O(n) check), or run serially (or group by index) when duplicates are present. `calc_diffraction_gain` passes unique indices and is not affected. File: `src/update_xpr.cpp`.

**Test for the fix session.** Catch2 with the repro case: the result equals the single-thread result for any thread count (or duplicates throw).

## B-025: `path_to_tube`: NaN vertices when interior path points are collinear

- Status: fixed (WP-13): also for an exact reversal
- Severity: S2, because straight multi-point paths (e.g. transmission through a wall) are common, and the NaN mesh breaks the OBJ export of channels.
- Kind: wrong result
- Mark: [R]
- Where: `src/path_to_tube.cpp:323` (`crossp(..., true)` of antiparallel vectors normalises a zero vector), `:327, :410` (`acos` of a value slightly above 1).

**What happens.** 2 points: no NaN; 3 collinear points: 12 of 36 vertex coordinates are NaN; a 5.7° bend: no NaN.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-9/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-9: path_to_tube: NaN vertices for collinear interior points
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    for (int n : {2, 3})
    {
        arma::mat pc(3, n, arma::fill::zeros); // straight path along x
        for (int i = 0; i < n; ++i) pc(0, i) = 10.0 * i;
        arma::mat vert; arma::umat faces;
        quadriga_lib::path_to_tube<double>(&pc, &vert, &faces, 0.5, 4);
        printf("%d collinear points: %llu vertex coordinates, %llu are NaN\n", n, (unsigned long long)vert.n_elem,
               (unsigned long long)arma::accu(arma::conv_to<arma::umat>::from(vert != vert)));
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-9/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-9
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
2 collinear points: 24 vertex coordinates, 0 are NaN
3 collinear points: 36 vertex coordinates, 12 are NaN
```

**Proposed fix.** When the cross product is (near) zero, skip the projection and reuse the previous ring shifted to the new vertex; clamp the `acos` arguments to [-1, 1]. File: `src/path_to_tube.cpp`.

**Test for the fix session.** Catch2: 3 collinear points give a finite mesh.

## B-026: AVX2 sin/cos returns values far outside [-1, 1], ±inf or NaN for large arguments (fast_sincos, fast_geo2cart, baseband_freq_response)

- Status: fixed (WP-08)
- Severity: S2, because the AVX2 kernel differs from the generic one far beyond rounding on valid finite input; `baseband_freq_response` reaches it with long delays (e.g. 2 s gives |H| = 1.6e11 instead of 1, 5 s gives NaN).
- Kind: wrong result (AVX2 differs from generic)
- Mark: [R]
- Where: `src/fastmath_avx2.h:85-89` (`_fm256_sincos256_ps`: Cody-Waite reduction in float plus `_mm256_cvttps_epi32`), `:159-174` (`_fm256_sincos256_pd`: `_mm256_cvttpd_epi32` overflows for |x|·4/π > 2^31). Reached through `fast_sincos` (`src/fastmath_armadillo.cpp:70-77`), `fast_geo2cart` (`src/fastmath_vectorized_avx2.cpp:523` float, `:611-612` double), `_fm256_polar_ps`, and `src/baseband_freq_response_avx2.cpp:59` (float phase).

**What happens.** Float input: max |sin error| 1.6e-7 up to 8.2e6 rad, 3.3e-5 up to 1.6e7, 0.027 up to 3.3e7; above that |s| exceeds 1 (13 near 5e7, 6e5 near 2e8, 1e10 near 1e9) and becomes inf above 1.7e9. Double input: accurate up to 1.686e9, then sin = -inf / cos = NaN (x = 3e9). `fast_geo2cart(az = 2e9, el = 0.25)` gives (inf, -inf, 0.247). The generic kernel is accurate to 1.3e-7 over [-1e10, 1e10]. `baseband_freq_response` (one path, coefficient 1): AVX2 |H| = 1.6e11 at τ = 2 s and NaN at 5 s, generic |H| = 1 (area C). The documentation states error bounds only for [-π, π] and [-500, 500].

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-1/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-1: AVX2 fast_sincos / fast_geo2cart return garbage (|sin| >> 1, inf, NaN) for large arguments; GENERIC correct
#include "quadriga_math.hpp"
#include <cstdio>
#include <cmath>
int main()
{
    // float input, one decade sample each: max |err| and max |s|
    for (double a : {4.1e6, 1.6e7, 3.3e7, 1.3e8, 1.0e9})
    {
        arma::fvec x = arma::linspace<arma::fvec>(float(a), float(2 * a), 20001), s, c;
        quadriga_lib::fast_sincos(x, &s, &c);
        double e = 0, smax = 0;
        for (arma::uword i = 0; i < x.n_elem; ++i)
            e = std::fmax(e, std::fabs(std::sin((double)x(i)) - s(i))), smax = std::fmax(smax, std::fabs(s(i)));
        printf("float  x in [%.2g, %.2g]: max|err sin| = %.3g, max|s| = %.4g\n", a, 2 * a, e, smax);
    }
    // double input, single values
    arma::vec xd = {1.0e9, 1.6866e9, 3.0e9, INFINITY};
    arma::fvec s, c;
    quadriga_lib::fast_sincos(xd, &s, &c);
    for (arma::uword i = 0; i < xd.n_elem; ++i)
        printf("double x = %-10g sin = %-12.7g (ref %-12.7g) cos = %-12.7g (ref %.7g)\n", xd(i), s(i), std::sin(xd(i)), c(i), std::cos(xd(i)));
    // fast_geo2cart with large azimuth
    arma::vec az = {2.0e9}, el = {0.25}, X, Y, Z;
    quadriga_lib::fast_geo2cart<double>(az, el, X, Y, Z);
    printf("fast_geo2cart(az=2e9, el=0.25) = (%g, %g, %g), ref (%g, %g, %g)\n", X(0), Y(0), Z(0),
           std::cos(0.25) * std::cos(2e9), std::cos(0.25) * std::sin(2e9), std::sin(0.25));
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-1/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro_gen" build_generic/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
echo "== generic library (build_generic)"
"$D/repro_gen"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
float  x in [4.1e+06, 8.2e+06]: max|err sin| = 1.39e-07, max|s| = 1
float  x in [1.6e+07, 3.2e+07]: max|err sin| = 0.0267, max|s| = 1
float  x in [3.3e+07, 6.6e+07]: max|err sin| = 14.2, max|s| = 14.68
float  x in [1.3e+08, 2.6e+08]: max|err sin| = 6.01e+05, max|s| = 6.011e+05
float  x in [1e+09, 2e+09]: max|err sin| = inf, max|s| = inf
double x = 1e+09      sin = 0.5458435    (ref 0.5458434   ) cos = 0.8378872    (ref 0.8378872)
double x = 1.6866e+09 sin = 0.1177912    (ref 0.1177912   ) cos = 0.9930384    (ref 0.9930384)
double x = 3e+09      sin = -inf         (ref 0.9870049   ) cos = -nan         (ref -0.1606902)
double x = inf        sin = -inf         (ref -nan        ) cos = -nan         (ref -nan)
fast_geo2cart(az=2e9, el=0.25) = (inf, -inf, 0.247404), ref (0.391547, 0.886274, 0.247404)
== generic library (build_generic)
float  x in [4.1e+06, 8.2e+06]: max|err sin| = 1.27e-07, max|s| = 1
float  x in [1.6e+07, 3.2e+07]: max|err sin| = 1.3e-07, max|s| = 1
float  x in [3.3e+07, 6.6e+07]: max|err sin| = 1.3e-07, max|s| = 1
float  x in [1.3e+08, 2.6e+08]: max|err sin| = 1.31e-07, max|s| = 1
float  x in [1e+09, 2e+09]: max|err sin| = 1.28e-07, max|s| = 1
double x = 1e+09      sin = 0.5458435    (ref 0.5458434   ) cos = 0.8378872    (ref 0.8378872)
double x = 1.6866e+09 sin = 0.1177912    (ref 0.1177912   ) cos = 0.9930384    (ref 0.9930384)
double x = 3e+09      sin = 0.9870049    (ref 0.9870049   ) cos = -0.1606903   (ref -0.1606902)
double x = inf        sin = -nan         (ref -nan        ) cos = -nan         (ref -nan)
fast_geo2cart(az=2e9, el=0.25) = (0.391547, 0.886274, 0.247404), ref (0.391547, 0.886274, 0.247404)
```

Folder `build_linux/scratch/hunt/final/C-4/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-4: AVX2 baseband_freq_response returns unbounded values / NaN for |theta| > ~1e9 rad; generic stays bounded
#include "quadriga_lib.hpp"
#include <cstdio>
#include <complex>
int main()
{
    const double B = 100e6; // one path, coefficient 1, pilot 1.0: exact H = exp(-j*2*pi*B*tau), |H| = 1
    for (double tau : {0.12, 2.0, 5.0})
    {
        arma::cube cr(1, 1, 1, arma::fill::ones), ci(1, 1, 1, arma::fill::zeros), dl(1, 1, 1);
        dl(0) = tau;
        arma::vec pg = {1.0};
        arma::cx_cube h;
        quadriga_lib::baseband_freq_response<double>(&cr, &ci, &dl, &pg, B, nullptr, nullptr, &h);
        std::complex<double> ref = std::polar(1.0, -2.0 * M_PI * B * tau);
        printf("tau = %-6g s: |H| = %-12.6g |H - H_exact| = %.3g\n", tau, std::abs(h(0)), std::abs(h(0) - ref));
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-4/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-4
for L in build_linux build_generic; do
  g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro_${L}" $L/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
  echo "== library $L"; OMP_NUM_THREADS=1 "$D/repro_${L}" || echo "exit code $?"
done
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== library build_linux
tau = 0.12   s: |H| = 1            |H - H_exact| = 0.313
tau = 2      s: |H| = 1.62946e+11  |H - H_exact| = 1.63e+11
tau = 5      s: |H| = -nan         |H - H_exact| = -nan
== library build_generic
tau = 0.12   s: |H| = 1            |H - H_exact| = 0.313
tau = 2      s: |H| = 1            |H - H_exact| = 0.822
tau = 5      s: |H| = 1            |H - H_exact| = 1.78
```

**Cause.** Float path: the reduction `x - y·DP1 - y·DP2 - y·DP3` in float with `y` up to 2^24 and beyond is no longer exact; beyond 2^31·π/4 the conversion to int32 yields `INT_MIN`. Double path: the reduction is exact, but the octant index is int32 and overflows at |x| > 1.686e9.

**Proposed fix.** Use the strategy of the generic kernel in `src/fastmath_avx2.h`: first reduce in double by multiples of 2π (n = round(x / 2π) as double, two-constant subtraction as in `qd_range_reduce_2pi`), then do the octant reduction on |x| <= π so the index stays in 0..4 (inf then gives NaN naturally); for float input convert to double and use the `_pd` variant. Small cost for every user of `_fm256_sincos256_ps` (antenna interpolation, `qd_coeff_combine_avx2`, `baseband_freq_response_avx2`). Fixing B-020 also keeps the baseband phase small.

**Test for the fix session.** Catch2: for x in {1e7, 3.3e7, 1e8, 1.7e9, 3e9, -5e9} as `fvec` and `vec`: |s - sin(double(x))| < 1e-6 and |s| <= 1; x = inf gives NaN; `fast_geo2cart` with `use_kernel` 1 and 2 agree within 1e-6 for az = 2e9.

## B-027: `point_inside_mesh` misclassifies points 5 cm from a face at UTM-like coordinates in AVX2 builds (float ray–triangle kernel)

- Status: fixed (WP-12): the AVX2 kernel forms origin-to-vertex, edge and box-to-origin vectors in double before rounding them to float, so its precision for double input depends on the scene size, not on the distance from the origin; the CUDA kernel still rounds the coordinates (not built here, noted)
- Severity: S2, because the result is wrong for valid double input in a common setting (scenes in UTM coordinates) and depends on the build; the documentation says nothing about single precision.
- Kind: wrong result (AVX2 differs from generic)
- Mark: [R]
- Where: `src/point_inside_mesh.cpp:148` calls `ray_triangle_intersect` with the default `use_kernel = 0`, which selects AVX2; `src/ray_triangle_intersect_avx2.cpp:491-497, 518-580` narrows double coordinates to float. Same pattern in `src/obj_overlap_test.cpp:610` (no wrong result found there up to offsets of 5e6, thanks to its double-precision 2-D logic).

**What happens.** A 10 m cube and 12 points each 5 cm inside or outside one face: at offset (0, 0) all correct in both builds; at (5e5, 5e6) 2 of 12 misclassified, at (5e6, 5e6) 4 of 12 (AVX2 library); the generic library misclassifies none.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-2/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-2: point_inside_mesh misclassifies points near faces at UTM-like coordinates in AVX2 builds (generic correct)
#include "quadriga_tools.hpp"
#include <cstdio>
static void run(double ox, double oy)
{
    arma::mat mesh = quadriga_lib::cube<double>({5.0}, {}, {ox, oy, 5.0}); // 10 m cube
    arma::mat pts;
    arma::uvec expect;
    for (int ax = 0; ax < 3; ++ax)
        for (int s = -1; s <= 1; s += 2)
            for (double dd : {-0.05, 0.05}) // 5 cm inside (-) / outside (+) each face
            {
                arma::rowvec p = {ox, oy, 5.0};
                p(ax) += s * (5.0 + dd);
                pts = arma::join_cols(pts, p);
                expect = arma::join_cols(expect, arma::uvec({dd < 0 ? 1ULL : 0ULL}));
            }
    arma::uvec r = quadriga_lib::point_inside_mesh(&pts, &mesh);
    printf("offset (%g, %g): %llu of %llu points misclassified\n", ox, oy,
           (unsigned long long)arma::accu(r != expect), (unsigned long long)r.n_elem);
}
int main()
{
    run(0, 0);
    run(5e5, 5e6);
    run(5e6, 5e6);
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-2/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro_gen" build_generic/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
echo "== generic library (build_generic)"
"$D/repro_gen"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
offset (0, 0): 0 of 12 points misclassified
offset (500000, 5e+06): 2 of 12 points misclassified
offset (5e+06, 5e+06): 4 of 12 points misclassified
== generic library (build_generic)
offset (0, 0): 0 of 12 points misclassified
offset (500000, 5e+06): 0 of 12 points misclassified
offset (5e+06, 5e+06): 0 of 12 points misclassified
```

**Cause.** The AVX2 ray–triangle kernel computes in float, whose spacing is 0.5 m at 5e6. `ray_mesh_interact` (excluded) has no kernel dispatch, so the kernel choice is the only difference between the builds.

**Proposed fix.** Recommended: in `ray_triangle_intersect`, when the type is double and the AVX2 kernel is chosen, subtract a reference point (the centre of the mesh AABB) from origins, destinations, mesh and AABB before narrowing to float; this keeps the speed and fixes every caller. Minimal: `point_inside_mesh` shifts its inputs to a local origin itself or passes `use_kernel = 1` for double. Files: `src/ray_triangle_intersect.cpp` (or the double path in `src/ray_triangle_intersect_avx2.cpp`), `src/point_inside_mesh.cpp`.

**Test for the fix session.** Catch2 from the repro: a cube at offset (5e6, 5e6), points ±5 cm from each face, expected 1/0; must pass in the AVX2 build.

## B-028: `obj_overlap_test` misses a clear overlap when vertices of the penetrating object lie on extensions of the other object's face planes

- Status: fixed (WP-12): each 3-D hit is judged against the face that produced it, a rejected first hit falls back to the second, the 2-D test only adds hits; plus a graze rule (decided 2026-10-21): a hit on a face is a touch when the edge lies in the plane of another face of the same object and the hit face lies on one side of that plane
- Severity: S2, because the overlap checker silently reports "no overlap" for a clear volumetric overlap on valid input; vertices on face-plane extensions are common in axis-aligned building scenes.
- Kind: wrong result
- Mark: [R]
- Where: `src/obj_overlap_test.cpp:690-727` (loop over all faces of object 1 inside the edge loop).

**What happens.** Object 1 is the cube [-1, 1]^3; object 2 is a tetrahedron with its apex at the cube centre and its base vertices at x = 3, each on a cube face-plane extension (y = 1, y = -1, z = 1). `obj_overlap_test` returns 0 overlapping objects although `ray_triangle_intersect` confirms that the apex edge hits the cube. With the base vertices moved 0.05 off the planes it returns 2. Same in both builds.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-3/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-3: obj_overlap_test misses an overlap when vertices of the penetrating object lie on extensions of the other object's face planes
// Probe: obj_overlap_test false negative. Object 0 = cube [-1,1]^3. Object 1 = tetrahedron with its apex at the
// cube centre (deep inside the cube) and its three base vertices outside the cube at x = 3, each base vertex lying
// on the extension of one cube face plane (y = 1, y = -1, z = 1). The tetrahedron clearly penetrates the cube;
// its cross-section with the face x = 1 stays clear of that face's diagonal, so no cube edge pierces it.
// Control: the same tetrahedron with base vertices moved off the planes by 0.05.
#include "quadriga_tools.hpp"
#include <cstdio>
static void test(const char *tag, double off)
{
    arma::mat A = quadriga_lib::cube<double>();
    arma::rowvec P0 = {0, 0, 0}, P1 = {3, 1 + off, 0.2}, P2 = {3, -1 - off, -1.5}, P3 = {3, 2.5, 1 + off};
    arma::mat B(4, 9);
    B.row(0) = arma::join_rows(P0, P1, P2);
    B.row(1) = arma::join_rows(P0, P2, P3);
    B.row(2) = arma::join_rows(P0, P3, P1);
    B.row(3) = arma::join_rows(P1, P3, P2);
    arma::mat mesh = arma::join_cols(A, B);
    arma::uvec obj = arma::join_cols(arma::uvec(A.n_rows, arma::fill::value(1)), arma::uvec(B.n_rows, arma::fill::value(2)));
    std::vector<std::string> why;
    arma::uvec r = quadriga_lib::obj_overlap_test(&mesh, &obj, &why);
    printf("%-45s -> %llu overlapping objects", tag, (unsigned long long)r.n_elem);
    if (!why.empty()) printf(" [%s]", why[0].c_str());
    printf("\n");
    // independent check: does the apex edge P0->P1 hit a cube face?
    arma::mat o = P0, d = P1;
    arma::u32_vec n_hit;
    quadriga_lib::ray_triangle_intersect<double>(&o, &d, &A, nullptr, nullptr, &n_hit);
    printf("%-45s    ray_triangle_intersect(P0->P1, cube) hits: %u\n", "", n_hit(0));
}
int main()
{
    test("base vertices on cube face planes (off=0)", 0.0);
    test("control: base vertices off the planes (0.05)", 0.05);
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-3/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-3
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro_gen" build_generic/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
echo "== generic library (build_generic)"
"$D/repro_gen"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
base vertices on cube face planes (off=0)     -> 0 overlapping objects
                                                 ray_triangle_intersect(P0->P1, cube) hits: 1
control: base vertices off the planes (0.05)  -> 2 overlapping objects [3D Intersect: OBJ-IDs (1,2) @ F:4 {(1,1,1),(1,-1,-1),(1,1,-1)} vs. E:0(0) {(0,0,0),(3,1.05,0.2)}]
                                                 ray_triangle_intersect(P0->P1, cube) hits: 1
== generic library (build_generic)
base vertices on cube face planes (off=0)     -> 0 overlapping objects
                                                 ray_triangle_intersect(P0->P1, cube) hits: 1
control: base vertices off the planes (0.05)  -> 2 overlapping objects [3D Intersect: OBJ-IDs (1,2) @ F:4 {(1,1,1),(1,-1,-1),(1,1,-1)} vs. E:0(0) {(0,0,0),(3,1.05,0.2)}]
                                                 ray_triangle_intersect(P0->P1, cube) hits: 1
```

**Cause.** `p_hit[i_edge]` holds the first-hit face from `ray_triangle_intersect`. The following loop over all faces of object 1 clears it when one endpoint of the edge lies on the plane of any face (line 698), and overwrites it in the coplanar 2-D branch (714, 725, 727) for any coplanar face; neither test is restricted to the face that produced the hit. Corrected after WP-12: the same blanket clear also hid false hits of touching objects (the diagonal of a touching face crosses the other object's edge, which the edge-inclusive triangle test reports as a hit on a side face), so restricting it to the hit face alone reports touching boxes with an offset as overlapping; the fix adds a graze rule (see Status).

**Proposed fix.** Apply the endpoint-on-plane rejection only for the face that produced the hit (`i_face == p_hit[i_edge] - 2`), and let the 2-D branch only add hits, never clear a 3-D hit from another face. File: `src/obj_overlap_test.cpp`.

**Test for the fix session.** Catch2: the tetrahedron case returns {1, 2}; the existing touching-cube cases still return {}.

## B-029: MATLAB R2026a segfaults at exit after any HDF5 MEX call (the unloaded thread-safe libhdf5 leaves a dangling pthread-key destructor)

- Status: fixed (WP-01)
- Severity: S2, because every R2026a session that read or wrote an HDF5 file through quadriga-lib ends with a segmentation violation, a crash dump and exit code 137 (results already computed are fine, but `-batch` jobs report failure); R2024a and Octave are not affected.
- Kind: crash
- Mark: [R] (found in phase 0: the MEX suite in R2026a ended with a segfault after all tests had finished)
- Where: the Linux MATLAB MEX link line, `CMakeLists.txt:527-535`; it affects the `.mexa64` files that load `libhdf5_serial.so.310`: `hdf5_create_file`, `hdf5_read_channel`, `hdf5_read_dset`, `hdf5_read_dset_names`, `hdf5_read_layout`, `hdf5_reshape_layout`, `hdf5_write_channel`, `hdf5_write_dset` (`hdf5_version` links it too, but its call does not crash). Not related to OpenMP or `src/mex_compat_shim.cpp`.

**What happens.** One `quadriga_lib.hdf5_create_file(...)` in R2026a `-batch`, then normal exit: "Segmentation violation detected", crash dump, exit code 137 (with `OMP_NUM_THREADS` 8 and 1). R2024a: exit code 0. The crash stack is the faulting address in an unmapped module called from `libc.so.6+0xa0df2` and `+0x1375cc`, which `gdb` resolves to `__nptl_deallocate_tsd + 178` and `__clone3 + 44`: the exiting interpreter thread runs a pthread-key destructor in unloaded code. A what-if MEX built exactly as CMake does crashes the same way; the same build with `-Wl,-z,nodelete` exits with 0; preloading `libhdf5_serial.so.310` also avoids the crash. Each run of the repro writes new `~/matlab_crash_dump.*` files.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-1/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-1: MATLAB R2026a segfaults at exit after an HDF5 MEX call (dangling pthread-key destructor of libhdf5)
% Called by run.sh through wrun.sh: creates and deletes an HDF5 file, then MATLAB exits normally.
fn = fullfile(pwd, 'x.hdf5');
if exist(fn, 'file'), delete(fn); end
disp(['which: ' which('quadriga_lib.hdf5_create_file')]);
quadriga_lib.hdf5_create_file(fn);
delete(fn);
disp('created');
```

`wrun.sh`:
```bash
#!/bin/bash
# F-1 helper: wrun.sh <matlab-binary> <path-to-addpath> <code>
# Runs MATLAB -batch in this folder, prints exit code, new crash dumps and the lines "which:"/"created".
D=$(dirname "$(realpath "$0")")
before=$(ls ~/matlab_crash_dump.* 2>/dev/null | sort)
cd "$D"
OMP_NUM_THREADS=${OMP:-8} timeout 600 $1 -batch "addpath('$2'); $3" > "$D/wrun_last.log" 2>&1
rc=$?
after=$(ls ~/matlab_crash_dump.* 2>/dev/null | sort)
echo "rc=$rc newdumps=[$(comm -13 <(echo "$before") <(echo "$after") | tr '\n' ' ')] out=$(grep -E '^(created|which:)' "$D/wrun_last.log" | tr '\n' ' ')"
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-1/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-1: MATLAB R2026a crash at exit after hdf5_create_file; R2024a clean; what-if: MEX linked with -z nodelete is clean
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-1)
C="run('$D/repro.m')"
echo "== R2026a (matlab), OMP=8, worktree MEX (expect rc=137 + dump)";   OMP=8 "$D/wrun.sh" matlab   /sjc/quadriga-lib-fixes "$C"
echo "== R2026a, OMP_NUM_THREADS=1 (expect rc=137 + dump)";              OMP=1 "$D/wrun.sh" matlab   /sjc/quadriga-lib-fixes "$C"
echo "== R2024a (matlab24) (expect rc=0)";                               OMP=8 "$D/wrun.sh" matlab24 /sjc/quadriga-lib-fixes "$C"
echo "== R2026a, single call quadriga_lib.hdf5_version (expect rc=0)";  OMP=8 "$D/wrun.sh" matlab   /sjc/quadriga-lib-fixes "disp(quadriga_lib.hdf5_version); disp('created')"
# What-if (NOT a fix): same MEX built as in CMakeLists.txt, once plain and once with -Wl,-z,nodelete, into this folder
H="/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.so"
for v in plain nodelete; do
  X=""; [ $v = nodelete ] && X="-Wl,-z,nodelete"
  mkdir -p "$D/whatif_$v/+quadriga_lib"
  g++-12 -std=c++17 -O3 -shared -fPIC -DMATLAB_MEX_FILE -Ibuild_linux/matlab_mex/include -Iinclude -Isrc -I/usr/include \
    api_mex/hdf5_create_file.cpp src/mex_compat_shim.cpp build_linux/libquadriga.a $H -lgomp \
    -Wl,--version-script,build_linux/matlab_mex/mexFunction.map -Wl,--no-undefined-version -Wl,--unresolved-symbols=ignore-all \
    $X -o "$D/whatif_$v/+quadriga_lib/hdf5_create_file.mexa64" || echo "build failed $?"
  echo "== R2026a, what-if MEX '$v' (plain: expect 137; nodelete: expect 0)"; OMP=8 "$D/wrun.sh" matlab "$D/whatif_$v" "$C"
done
echo "== crash stack of the last dump (libc offsets)"
last=$(ls -t ~/matlab_crash_dump.* | head -1); echo "$last"; grep -A3 "Stack Trace" "$last"
gdb -batch -ex "info symbol 0xa0df2" -ex "info symbol 0x1375cc" /usr/lib/x86_64-linux-gnu/libc.so.6 2>/dev/null | tail -2
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== R2026a (matlab), OMP=8, worktree MEX (expect rc=137 + dump)
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/F-1/wrun.sh: line 7: 716407 Killed                     OMP_NUM_THREADS=${OMP:-8} timeout 600 $1 -batch "addpath('$2'); $3" > "$D/wrun_last.log" 2>&1
rc=137 newdumps=[/home/stephan/matlab_crash_dump.716408-1 ] out=which: /sjc/quadriga-lib-fixes/+quadriga_lib/hdf5_create_file.mexa64 created 
== R2026a, OMP_NUM_THREADS=1 (expect rc=137 + dump)
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/F-1/wrun.sh: line 7: 716733 Killed                     OMP_NUM_THREADS=${OMP:-8} timeout 600 $1 -batch "addpath('$2'); $3" > "$D/wrun_last.log" 2>&1
rc=137 newdumps=[/home/stephan/matlab_crash_dump.716734-1 ] out=which: /sjc/quadriga-lib-fixes/+quadriga_lib/hdf5_create_file.mexa64 created 
== R2024a (matlab24) (expect rc=0)
rc=0 newdumps=[] out=which: /sjc/quadriga-lib-fixes/+quadriga_lib/hdf5_create_file.mexa64 created 
== R2026a, single call quadriga_lib.hdf5_version (expect rc=0)
rc=0 newdumps=[] out=created 
== R2026a, what-if MEX 'plain' (plain: expect 137; nodelete: expect 0)
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/F-1/wrun.sh: line 7: 717815 Killed                     OMP_NUM_THREADS=${OMP:-8} timeout 600 $1 -batch "addpath('$2'); $3" > "$D/wrun_last.log" 2>&1
rc=137 newdumps=[/home/stephan/matlab_crash_dump.717816-1 ] out=which: /sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/F-1/whatif_plain/+quadriga_lib/hdf5_create_file.mexa64 created 
== R2026a, what-if MEX 'nodelete' (plain: expect 137; nodelete: expect 0)
rc=0 newdumps=[] out=which: /sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/F-1/whatif_nodelete/+quadriga_lib/hdf5_create_file.mexa64 created 
== crash stack of the last dump (libc offsets)
/home/stephan/matlab_crash_dump.717816-1
Stack Trace (from fault):
[  0] 0x00007815d7ce8770                                   <unknown-module>+00000000
[  1] 0x00007816ca8a0df2                /usr/lib/x86_64-linux-gnu/libc.so.6+00658930
[  2] 0x00007816ca9375cc                /usr/lib/x86_64-linux-gnu/libc.so.6+01275340
__nptl_deallocate_tsd + 178 in section .text of /usr/lib/x86_64-linux-gnu/libc.so.6
__clone3 + 44 in section .text of /usr/lib/x86_64-linux-gnu/libc.so.6
```

**Cause.** The MEX files link the shared, thread-safe system HDF5 (`HDF5_LINK_LIBS` = `.../hdf5/serial/libhdf5.so`, soname `libhdf5_serial.so.310`; the comment at `CMakeLists.txt:196` says MATLAB requires static linking). This HDF5 creates pthread keys with destructors (`H5TS_apictx_key_g`, `H5TS_errstk_key_g`) on its first API call and never deletes them (it imports `pthread_key_create` but not `pthread_key_delete`). R2026a unloads the MEX files at shutdown (and on `clear mex`); nothing else holds libhdf5, so it is unmapped with them; when the interpreter thread then terminates, glibc calls the dangling destructor. Static linking would not help (the destructor would sit in the unloaded MEX file).

**Proposed fix.** Keep HDF5 mapped for the life of the process: add `-Wl,-z,nodelete` to the Linux MATLAB MEX link line, at least for the `hdf5_*` files (confirmed by the what-if; a rebuilt MEX then needs a MATLAB restart). Alternatives: `mexLock()` in the 8 HDF5 wrappers, or a one-time `dlopen("libhdf5_serial.so.310", RTLD_NOW | RTLD_NODELETE | RTLD_NOLOAD)` in a shared helper. File: `CMakeLists.txt` (and the comment at line 196).

**Test for the fix session.** A shell or CTest step: R2026a `-batch "addpath(root); quadriga_lib.hdf5_create_file(tempname)"` exits with 0 and writes no new crash dump.

## B-030: MEX: sparse inputs are read as dense arrays (wrong values, heap over-read) in every wrapper

- Status: fixed (WP-06)
- Severity: S2, because a sparse argument (common in MATLAB code) silently gives wrong numbers and reads past the heap buffer in every wrapper.
- Kind: memory / wrong result
- Mark: [R] (heap over-read shown with an ASan what-if build of one MEX file for Octave)
- Where: `src/mex_helper_functions.hpp`: `qd_mex_get_dims` (75), `qd_mex_get_scalar` (90), `qd_mex_reinterpret_*` (161-219), `qd_mex_anycast` (229), `qd_mex_copy_convert` (318), `qd_mex_matlab2vector_*` (938 ff.), `qd_mex_matlab2vector_Bool` (1429). No helper checks `mxIsSparse`, and `mxIsDouble` is true for sparse arrays.

**What happens.** `fast_sincos(sparse([0 0 0 0 0 0 0 1.5]))` returns sin `[0.9975 0 0 ...]` with garbage (0, ±Inf, tiny values that change from run to run) in the rest, expected `[0 ... 0 0.9975]`; `calc_rotation_matrix(sparse([0;0;1]))` returns `[1 <garbage> ... 0.5403 0.8415 ...]` against `[0.5403 0.8415 0 ...]`; `cart2geo(sparse([1 0; 0 0; 0 3]))` gives lengths `[3.162 <garbage>]` against `[1 3]`. The ASan what-if (Octave MEX of `calc_rotation_matrix` against `build_asan`): heap-buffer-overflow READ of size 8 in `qd_rotation_matrix`, from `mexFunction` at `api_mex/calc_rotation_matrix.cpp:48`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-2/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-2: Sparse inputs are read as dense arrays (wrong results, heap over-read)
addpath('/sjc/quadriga-lib-fixes');
x = sparse([0 0 0 0 0 0 0 1.5]);
s = quadriga_lib.fast_sincos(x);
fprintf('fast_sincos(sparse) sin: %s\n', mat2str(double(s), 4));
fprintf('expected            sin: %s\n', mat2str(sin(full(x)), 4));
r = quadriga_lib.calc_rotation_matrix(sparse([0;0;1]));
fprintf('calc_rotation_matrix(sparse [0;0;1]) : %s\n', mat2str(r(:).', 4));
fprintf('calc_rotation_matrix(full   [0;0;1]) : %s\n', mat2str(reshape(quadriga_lib.calc_rotation_matrix([0;0;1]),1,[]), 4));
[az, el, len] = quadriga_lib.cart2geo(sparse([1 0; 0 0; 0 3]));
fprintf('cart2geo(sparse) len: %s   full: %s\n', mat2str(len.', 4), mat2str(sqrt(sum([1 0; 0 0; 0 3].^2)), 4));
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-2/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-2: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-2)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== MATLAB (matlab24)"
before=$(ls ~/matlab_crash_dump.* 2>/dev/null | sort)
(cd "$D" && OMP_NUM_THREADS=1 timeout 600 matlab24 -batch "run('$D/repro.m')" 2>&1 | grep -v -E '^\s*$|Warning:|vscode|License|academic|^\s+/home' | grep -E -v '^ {2,}[A-Z][A-Za-z ]+: ' | head -12) || echo "exit code $?"
echo "new crash dumps: $(comm -13 <(echo "$before") <(ls ~/matlab_crash_dump.* 2>/dev/null | sort) | tr "\n" " ")"
echo "== ASan what-if: calc_rotation_matrix.mex for Octave built with -fsanitize against build_asan (into this folder)"
mkdir -p "$D/asan/+quadriga_lib"
printf 'extern "C" { extern const int __octave_mex_soversion__; const int __octave_mex_soversion__ = 1; }\n' > "$D/asan/soversion.cpp"
g++-12 -std=c++17 -O1 -g -fPIC -shared -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DMEX_DEBUG $(mkoctfile -p INCFLAGS) -Iinclude -Isrc \
  api_mex/calc_rotation_matrix.cpp "$D/asan/soversion.cpp" -o "$D/asan/+quadriga_lib/calc_rotation_matrix.mex" build_asan/libquadriga.a \
  -L$(mkoctfile -p OCTLIBDIR) -loctmex -Wl,-rpath,$(mkoctfile -p OCTLIBDIR) || echo "build failed"
(cd "$D/asan" && ASAN_OPTIONS=detect_leaks=0:halt_on_error=0:verify_asan_link_order=0 LD_PRELOAD="$(g++-12 -print-file-name=libasan.so) $(g++-12 -print-file-name=libubsan.so)" \
  OMP_NUM_THREADS=1 octave --no-gui --norc --eval "addpath('$D/asan'); r = quadriga_lib.calc_rotation_matrix(sparse([0;0;1]));" 2>&1 \
  | grep -E "ERROR: |READ of|^    #[0-2] " | head -5 | cut -c1-200) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
fast_sincos(sparse) sin: [0.9975 0 0 0 -Inf -Inf 0 0]
expected            sin: [0 0 0 0 0 0 0 0.9975]
calc_rotation_matrix(sparse [0;0;1]) : [1 1.581e-322 -0 -8.399e-323 0.5403 0.8415 1.334e-322 -0.8415 0.5403]
calc_rotation_matrix(full   [0;0;1]) : [0.5403 0.8415 -0 -0.8415 0.5403 0 0 0 1]
cart2geo(sparse) len: [3.162 6.086e-147]   full: [1 3]
== MATLAB (matlab24)
fast_sincos(sparse) sin: [0.9975 0 0 0 0 0 0 0]
expected            sin: [0 0 0 0 0 0 0 0.9975]
calc_rotation_matrix(sparse [0;0;1]) : [1 2.437e-320 -6.591e-310 5.546e-310 0.5403 0.8415 3.561e-310 -0.8415 0.5403]
calc_rotation_matrix(full   [0;0;1]) : [0.5403 0.8415 0 -0.8415 0.5403 0 0 0 1]
cart2geo(sparse) len: [3.162 0]   full: [1 3]
new crash dumps: 
== ASan what-if: calc_rotation_matrix.mex for Octave built with -fsanitize against build_asan (into this folder)
==719305==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x796954f2a0b8 at pc 0x794938c9d988 bp 0x79493bde5960 sp 0x79493bde5950
READ of size 8 at 0x796954f2a0b8 thread T10 (QThread)
    #0 0x794938c9d987 in qd_rotation_matrix<double> /sjc/quadriga-lib-fixes/src/quadriga_lib_helper_functions.hpp:118
    #1 0x794938c9d987 in arma::Cube<double> quadriga_lib::calc_rotation_matrix<double>(arma::Cube<double> const&, bool, bool) /sjc/quadriga-lib-fixes/src/calc_rotation_matrix.cpp:56
    #2 0x794938c83a5f in mexFunction api_mex/calc_rotation_matrix.cpp:48
```

**Cause.** `mxGetData` of a sparse array returns only the `nzmax` non-zero values; the helpers index it with the dense size.

**Proposed fix.** Reject sparse input centrally: `if (mxIsSparse(input)) mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Sparse inputs are not supported.")` in `qd_mex_get_dims`, `qd_mex_get_scalar`, `qd_mex_anycast`, `qd_mex_matlab2vector_*` and `qd_mex_matlab2vector_Bool`. Alternative: convert with `full` semantics (costlier). File: `src/mex_helper_functions.hpp`.

**Test for the fix session.** MOxUnit: `calc_rotation_matrix(sparse([0;0;1]))` and `fast_sincos(sparse(1:3))` raise `quadriga_lib:CPPerror`.

## B-031: MEX `qrt_file_read` (`i_cir`) and `hdf5_read_channel` (`snap`) modify the caller's uint64 variable in place

- Status: fixed (WP-01)
- Severity: S2, because the wrapper silently changes a user variable (and every MATLAB copy sharing its data), so the next call with the same variable reads wrong data.
- Kind: wrong result / UB (write into an input)
- Mark: [R]
- Where: `api_mex/qrt_file_read.cpp:83` (`qd_mex_get_Col<arma::uword>(prhs[1])` without copy), then `:104` `i_cir -= 1`; `api_mex/hdf5_read_channel.cpp:119` (`snap`, no copy; corrected in WP-01, the item first said 149), then `:166-170` `--s`. Both break the rule of `md/SPEC_MATLAB_API.md` "Index inputs that get converted MUST be read with copy = true".

**What happens.** `i_cir = uint64([1 2])` is `[0 1]` after `qrt_file_read`, and so is its copy `keep`; `snap = uint64([2 3])` becomes `[1 2]` after `hdf5_read_channel`, and a second call with the same variable returns snapshots 1-2 instead of 2-3. Same in Octave and R2024a. (For `double` input the helpers convert into a new buffer, so ordinary calls are not affected.)

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-3/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-3: qrt_file_read (i_cir) and hdf5_read_channel (snap) modify the caller's uint64 variable in place
addpath('/sjc/quadriga-lib-fixes');
i_cir = uint64([1 2]); keep = i_cir;
quadriga_lib.qrt_file_read('/sjc/quadriga-lib-fixes/tests/data/test.qrt', i_cir, 1, true, 1);
fprintf('qrt_file_read: i_cir after call = %s, copy "keep" = %s (expected [1 2])\n', mat2str(double(i_cir)), mat2str(double(keep)));
h5 = fullfile(pwd, 'f3.hdf5'); if exist(h5, 'file'), delete(h5); end
cre = reshape(1:6, 1, 1, 2, 3);
chan = struct('name','x','tx_position',[0;0;0],'rx_position',[1;2;3],'tx_orientation',[0;0;0],'rx_orientation',[0;0;0], ...
  'coeff_re', cre, 'coeff_im', zeros(1,1,2,3), 'delay', ones(1,1,2,3)*1e-6, 'center_frequency', 1e9);
quadriga_lib.hdf5_write_channel(h5, chan);
snap = uint64([2 3]); keep = snap;
c = quadriga_lib.hdf5_read_channel(h5, 1, 1, 1, 1, snap);
fprintf('hdf5_read_channel: snap after call = %s, copy "keep" = %s (expected [2 3])\n', mat2str(double(snap)), mat2str(double(keep)));
fprintf('1st call coeff_re = %s (snapshots 2,3 = [3 4 5 6])\n', mat2str(c.coeff_re(:).'));
c = quadriga_lib.hdf5_read_channel(h5, 1, 1, 1, 1, snap);
fprintf('2nd call, same variable: coeff_re = %s (expected [3 4 5 6])\n', mat2str(c.coeff_re(:).'));
delete(h5);
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-3/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-3: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-3)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== MATLAB (matlab24)"
before=$(ls ~/matlab_crash_dump.* 2>/dev/null | sort)
(cd "$D" && OMP_NUM_THREADS=1 timeout 600 matlab24 -batch "run('$D/repro.m')" 2>&1 | grep -v -E '^\s*$|Warning:|vscode|License|academic|^\s+/home' | grep -E -v '^ {2,}[A-Z][A-Za-z ]+: ' | head -12) || echo "exit code $?"
echo "new crash dumps: $(comm -13 <(echo "$before") <(ls ~/matlab_crash_dump.* 2>/dev/null | sort) | tr "\n" " ")"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
qrt_file_read: i_cir after call = [0 1], copy "keep" = [0 1] (expected [1 2])
hdf5_read_channel: snap after call = [1 2], copy "keep" = [1 2] (expected [2 3])
1st call coeff_re = [3 4 5 6] (snapshots 2,3 = [3 4 5 6])
2nd call, same variable: coeff_re = [1 2 3 4] (expected [3 4 5 6])
== MATLAB (matlab24)
qrt_file_read: i_cir after call = [0 1], copy "keep" = [0 1] (expected [1 2])
hdf5_read_channel: snap after call = [1 2], copy "keep" = [1 2] (expected [2 3])
1st call coeff_re = [3 4 5 6] (snapshots 2,3 = [3 4 5 6])
2nd call, same variable: coeff_re = [1 2 3 4] (expected [3 4 5 6])
new crash dumps: 
```

**Proposed fix.** `qd_mex_get_Col<arma::uword>(prhs[1], true)` in `api_mex/qrt_file_read.cpp:83` and `qd_mex_get_Col<arma::uword>(prhs[5], true)` in `api_mex/hdf5_read_channel.cpp:119`.

**Test for the fix session.** MOxUnit: pass `uint64` indices, keep a copy, assert both unchanged after the call, and two calls return the same data.

## B-032: MEX `channel_export_obj_file` segfaults when `no_interact` has fewer snapshots than the coefficients

- Status: fixed (WP-06)
- Severity: S2, because an easy input mistake (e.g. `[]` for `no_interact`) kills the MATLAB or Octave session.
- Kind: crash (out-of-bounds `std::vector` access)
- Mark: [R]
- Where: `api_mex/channel_export_obj_file.cpp:100-112` (the loop over `c.coeff_re.size()` reads `c.no_interact[i_snap_a]` without checking `c.no_interact.size()`; only `interact_coord` is checked at :97).

**What happens.** A valid call works; the same call with `no_interact = []` gives "fatal: caught signal Segmentation fault" in Octave and "Segmentation violation detected" (crash dump) in R2024a.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-4/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-4: channel_export_obj_file segfaults when no_interact has fewer snapshots than coeff_re
addpath('/sjc/quadriga-lib-fixes');
fn = fullfile(pwd, 'f4.obj');
cre = ones(1,1,2,1); cim = zeros(1,1,2,1);
ni = uint32([1;1]); ic = [1 2; 0 0; 0 0];
quadriga_lib.channel_export_obj_file(fn, 0, -60, -140, 'jet', 0.05, 0.01, 5, [10;0;0], [0;0;0], ni, ic, 1e9, cre, cim);
disp('valid call OK');
disp('calling with no_interact = []');
quadriga_lib.channel_export_obj_file(fn, 0, -60, -140, 'jet', 0.05, 0.01, 5, [10;0;0], [0;0;0], [], ic, 1e9, cre, cim);
disp('no_interact empty: returned (not expected)');
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-4/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-4: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-4)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== MATLAB (matlab24)"
before=$(ls ~/matlab_crash_dump.* 2>/dev/null | sort)
(cd "$D" && OMP_NUM_THREADS=1 timeout 600 matlab24 -batch "run('$D/repro.m')" 2>&1 | grep -v -E '^\s*$|Warning:|vscode|License|academic|^\s+/home' | grep -E -v '^ {2,}[A-Z][A-Za-z ]+: ' | head -12) || echo "exit code $?"
echo "new crash dumps: $(comm -13 <(echo "$before") <(ls ~/matlab_crash_dump.* 2>/dev/null | sort) | tr "\n" " ")"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
valid call OK
calling with no_interact = []
fatal: caught signal Segmentation fault -- stopping myself...
== MATLAB (matlab24)
valid call OK
calling with no_interact = []
--------------------------------------------------------------------------------
          Segmentation violation detected at 2026-10-15 18:49:44 +0200
--------------------------------------------------------------------------------
Configuration:
  Graphics card 1          : 0x10de ( 0x10de ) 0x1e84 Version 595.91.7.0 (0-0-0)
Fault Count: 1
Abnormal termination:
Segmentation violation
Current Thread: 'MCR 0 interpret' id 129994281965248
Register State (from fault):
new crash dumps: /home/stephan/matlab_crash_dump.719873-1 
```

**Proposed fix.** Before the loop, require `c.no_interact.size() == c.coeff_re.size()` (and the same for `interact_coord`), else `mexErrMsgIdAndTxt`. File: `api_mex/channel_export_obj_file.cpp`.

**Test for the fix session.** MOxUnit: the second call raises `quadriga_lib:CPPerror`.

## B-033: MEX `qd_mex_get_strings` dereferences empty cell elements: `cell(1,1)` as a name list crashes MATLAB R2026a

- Status: fixed (WP-06); unset cell elements and `[]` elements are rejected (Octave stores `cell(1,1)` as `[]`), `''` is allowed
- Severity: S2, because a valid MATLAB value (a cell array with unset elements) crashes MATLAB.
- Kind: crash (NULL dereference)
- Mark: [R] in R2026a; Octave does not crash (its `mxGetCell` returns an empty array)
- Where: `src/mex_helper_functions.hpp:501` (`qd_mex_get_string(mxGetCell(input, i))`; `mxGetCell` returns NULL for unset elements, and `qd_mex_get_string` calls `mxGetNumberOfElements(NULL)`). Users: `obj_file_write` (`obj_names`, `mtl_names`, `csv_names`), and in area B `conf_file_list`, `conf_file_write` (not tested).

**What happens.** `obj_file_write(fn, cube(), ones(12,1), [], cell(1,1))`: R2026a "Segmentation violation detected ... MATLAB is exiting because of fatal error" (crash dump); Octave returns without error.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-5/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-5: qd_mex_get_strings passes NULL cell elements on: cell(1,1) as obj_names crashes MATLAB (Octave does not crash)
addpath('/sjc/quadriga-lib-fixes');
mesh = quadriga_lib.cube();
fn = fullfile(pwd, 'f5.obj');
disp('calling obj_file_write with obj_names = cell(1,1)');
try
    quadriga_lib.obj_file_write(fn, mesh, ones(12,1), [], cell(1,1));
    disp('returned without error');
catch e
    disp(['error: ' e.message]);
end
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-5/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-5: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-5)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== MATLAB (matlab)"
before=$(ls ~/matlab_crash_dump.* 2>/dev/null | sort)
(cd "$D" && OMP_NUM_THREADS=1 timeout 600 matlab -batch "run('$D/repro.m')" 2>&1 | grep -v -E '^\s*$|Warning:|vscode|License|academic|^\s+/home' | grep -E -v '^ {2,}[A-Z][A-Za-z ]+: ' | head -12) || echo "exit code $?"
echo "new crash dumps: $(comm -13 <(echo "$before") <(ls ~/matlab_crash_dump.* 2>/dev/null | sort) | tr "\n" " ")"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
calling obj_file_write with obj_names = cell(1,1)
returned without error
== MATLAB (matlab)
calling obj_file_write with obj_names = cell(1,1)
--------------------------------------------------------------------------------
          Segmentation violation detected at 2026-10-15 18:49:48 +0200
--------------------------------------------------------------------------------
Configuration:
  Graphics card 1          : 0x10de ( 0x10de ) 0x1e84 Version 595.91.7.0 (0-0-0)
  Interpreter 0            : Executing request: 64657461696C2F44656661756C744D564D2E637070
Fault Count: 1
Abnormal termination:
Segmentation violation
Current Thread: 'MCR 0 interpret' id 130554691319488
Register State (from fault):
new crash dumps: /home/stephan/matlab_crash_dump.720262-1 
```

**Proposed fix.** In `qd_mex_get_strings` treat a NULL element as an error ("cell element i is not a string") or as an empty string, and let `qd_mex_get_string` reject `nullptr`. File: `src/mex_helper_functions.hpp`.

**Test for the fix session.** MOxUnit: the call above raises `quadriga_lib:CPPerror` (in Octave the test passes trivially; run it in MATLAB).

## B-034: MEX helpers convert NaN, negative, fractional and out-of-range numbers to integers without checks: crashes, bad_alloc, silent truncation, wrapped seeds

- Status: fixed (WP-06) on the MEX side; very large `n_div` still overflows in the C++ `icosphere` (B-078, WP-12)
- Severity: S2, because invalid but easy-to-type input (`-1`, `NaN`) crashes MATLAB and Octave; the other effects (silent truncation, a uint64 seed that silently becomes "random") are S3 on their own.
- Kind: UB / crash / invalid input accepted
- Mark: [R]
- Where: `src/mex_helper_functions.hpp:105-133` (`qd_mex_get_scalar` returns `dtype(*tmp)` for unsigned/uword types with a negative or NaN double, which is undefined; for a `uint64` input to `arma::sword` it wraps), `qd_mex_copy_convert:323` and `qd_mex_matlab2vector_*` (`(dtype)src[m]`); the default argument `dtype(NAN)` is UB for integer types when no default is given. Affected, for example: `icosphere` (`n_div`), `arrayant_generate` (`M`, `N`, `Mg`, `Ng`, `pol`), `quantize_delays` (`max_no_taps`), `cube` / `plane` (`n_div`), `point_cloud_*` and `triangle_mesh_*` (`target_size`, `vec_size`), every 1-based index input (`element`, `i_element`, `location`, `ix` ... `iw`), `get_channels_ieee_indoor` (`n_users`, `n_floors`, `n_walls`), `arrayant_qdant_write` (`id`), and the seeds of the SOS wrappers (copies of `qd_mex_sos_seed` in `api_mex/sos_init_phases.cpp:28-38`, `sos_randn.cpp:15`, `sos_rand.cpp:15`, `sos_randi.cpp:28`, `sos_generate.cpp:28`, `sos_acf_estimate.cpp:28`).

**What happens.** `icosphere(-1)` and `icosphere(NaN)` crash Octave ("malloc(): unaligned tcache chunk" or SIGSEGV) and MATLAB (exit code 139): -1 becomes `n_div = 2^64-1` (20·n_div² wraps to 20), NaN becomes 2^63 (wraps to 0), and the C++ `icosphere` then writes out of bounds (B-078). `arrayant_generate('3GPP', [], [], [], [], [], -1)` gives `std::bad_alloc`; `M = 2.7` silently builds an M = 2 array; `arrayant_interpolate(ant, 0, 0, 1.7)` silently uses element 1. SOS seeds (area E): `intmax('uint64')` is accepted and silently means "random" (two calls differ), `uint64(2)^63` is rejected with "must be >= -1"; Python raises `TypeError` for both.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-6/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// F-6: what the C++ icosphere does with the n_div values that MATLAB -1 / NaN become (same cast as qd_mex_get_scalar)
#include "quadriga_lib.hpp"
#include <cstdio>
#include <cmath>
int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    for (double in : {2.9, -1.0, (double)NAN})
    {
        volatile double v = in;
        arma::uword n_div = (arma::uword)v; // UB for -1 and NaN; this is what the wrapper does
        printf("input %g -> n_div = %llu, 20*n_div*n_div = %llu\n", in, (unsigned long long)n_div, (unsigned long long)(20 * n_div * n_div));
        arma::mat center;
        quadriga_lib::icosphere<double>(n_div, 1.0, &center);
        printf("  returned, center %llu x %llu\n", (unsigned long long)center.n_rows, (unsigned long long)center.n_cols);
    }
    return 0;
}
```

`repro.m`:
```matlab
% F-6: qd_mex_get_scalar converts NaN/negative/fractional values to unsigned without checks (UB): truncation, bad_alloc, crash
addpath('/sjc/quadriga-lib-fixes');
a = quadriga_lib.arrayant_generate('3GPP', [], [], [], [], [], 2.7);
fprintf('3GPP M=2.7 -> size(e_theta_re) = %s (silently M=2)\n', mat2str(size(a.e_theta_re)));
try
    quadriga_lib.arrayant_generate('3GPP', [], [], [], [], [], -1); disp('3GPP M=-1 returned');
catch e, fprintf('3GPP M=-1 -> error: %s\n', e.message); end
ant = quadriga_lib.arrayant_generate('omni');
v = quadriga_lib.arrayant_interpolate(ant, 0, 0, 1.7);
fprintf('arrayant_interpolate element=1.7 -> accepted, V_re = %g (element 1)\n', v);
disp('calling quadriga_lib.icosphere(-1)');
quadriga_lib.icosphere(-1);
disp('icosphere(-1) returned (not expected)');
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-6/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-6: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-6)
echo "== C++ (build_linux/libquadriga.a)"
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(cd "$D" && OMP_NUM_THREADS=1 ./repro) || echo "exit code $?"
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== MATLAB (matlab24)"
before=$(ls ~/matlab_crash_dump.* 2>/dev/null | sort)
(cd "$D" && OMP_NUM_THREADS=1 timeout 600 matlab24 -batch "run('$D/repro.m')" > "$D/matlab_matlab24.log" 2>&1; echo "matlab exit code $?")
grep -E -v '^\s*$|Warning:|vscode|License|academic|^\s+/home|^-+$|Configuration:|^ {2,}[A-Z][A-Za-z0-9 ]+: |Fault Count|Register State|^\s*\[' "$D/matlab_matlab24.log" | head -10
echo "new crash dumps: $(comm -13 <(echo "$before") <(ls ~/matlab_crash_dump.* 2>/dev/null | sort) | tr "\n" " ")"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== C++ (build_linux/libquadriga.a)
input 2.9 -> n_div = 2, 20*n_div*n_div = 80
  returned, center 80 x 3
input -1 -> n_div = 18446744073709551615, 20*n_div*n_div = 20
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/F-6//run.sh: line 7: 720592 Segmentation fault         (core dumped) ( cd "$D" && OMP_NUM_THREADS=1 ./repro )
exit code 139
== Octave 11.1
3GPP M=2.7 -> size(e_theta_re) = [181 361 2] (silently M=2)
3GPP M=-1 -> error: arrayant_generate: std::bad_alloc
arrayant_interpolate element=1.7 -> accepted, V_re = 1 (element 1)
calling quadriga_lib.icosphere(-1)
fatal: caught signal malloc(): unaligned tcache chunk detected
fatal: caught signal malloc(): unaligned tcache chunk detected
== MATLAB (matlab24)
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/F-6//run.sh: line 12: 720994 Segmentation fault         (core dumped) OMP_NUM_THREADS=1 timeout 600 matlab24 -batch "run('$D/repro.m')" > "$D/matlab_matlab24.log" 2>&1
matlab exit code 139
3GPP M=2.7 -> size(e_theta_re) = [181 361 2] (silently M=2)
3GPP M=-1 -> error: std::bad_alloc
arrayant_interpolate element=1.7 -> accepted, V_re = 1 (element 1)
calling quadriga_lib.icosphere(-1)
malloc(): invalid size (unsorted)
new crash dumps: 
```

Folder `build_linux/scratch/hunt/final/E-2/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% E-2: MEX uint64 seeds >= 2^63 wrap to negative: intmax('uint64') silently means random, 2^63 rejected as < -1
addpath('/sjc/quadriga-lib-fixes');
pos = [0 0 0; 1 2 3; 4 5 6];
a = quadriga_lib.sos_randn(10, pos, [], [], [], intmax('uint64'));
b = quadriga_lib.sos_randn(10, pos, [], [], [], intmax('uint64'));
fprintf('sos_randn seed intmax(''uint64''): accepted, two calls equal = %d\n', isequal(a, b));
p1 = quadriga_lib.sos_init_phases(4, 1, intmax('uint64')); p2 = quadriga_lib.sos_init_phases(4, 1, intmax('uint64'));
fprintf('sos_init_phases seed intmax(''uint64''): two calls equal = %d\n', isequal(p1, p2));
try, quadriga_lib.sos_randn(10, pos, [], [], [], uint64(2)^63); catch e, fprintf('seed uint64(2^63): %s\n', e.message); end
a = quadriga_lib.sos_randn(10, pos, [], [], [], uint64(2)^62);
b = quadriga_lib.sos_randn(10, pos, [], [], [], uint64(2)^62);
fprintf('seed uint64(2^62): two calls equal = %d\n', isequal(a, b));
```

`repro.py`:
```python
# E-2: Python comparison - seeds >= 2^63 are rejected with TypeError
import numpy as np
import quadriga_lib
for s in (2**63, 2**64 - 1):
    try:
        quadriga_lib.sos.randn(np.array([10.0]), np.zeros((1, 3)), seed=s)
        print("python seed", s, "accepted")
    except Exception as e:
        print("python seed", s, type(e).__name__)
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/E-2/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# E-2: MEX uint64 seeds >= 2^63 wrap around (Octave), Python for comparison
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/E-2
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')") 2>&1 | grep -v "qt.qpa" || echo "exit code $?"
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
sos_randn seed intmax('uint64'): accepted, two calls equal = 0
sos_init_phases seed intmax('uint64'): two calls equal = 0
seed uint64(2^63): sos_randn: Input 'seed' must be >= -1.
seed uint64(2^62): two calls equal = 1
python seed 9223372036854775808 TypeError
python seed 18446744073709551615 TypeError
```

**Proposed fix.** In `qd_mex_get_scalar` and `qd_mex_copy_convert`, for integer target types reject NaN, values below the type minimum (negative for unsigned), values above the maximum, and non-integers (`std::floor(x) != x`) with "Input 'name' must be a non-negative integer" (or similar); read `uint64` inputs as `unsigned long long` and reject values above the target maximum; make the default-less overload a compile error for integer types. Separately, the C++ `icosphere` and `subdivide_triangles` reject overflowing counts (B-078). Files: `src/mex_helper_functions.hpp`, the six `qd_mex_sos_seed` copies (or one shared helper).

**Test for the fix session.** MOxUnit: `icosphere(-1)`, `icosphere(NaN)`, `arrayant_generate('3GPP', [], [], [], [], [], -1)` and `sos_init_phases(4, 1, intmax('uint64'))` raise `quadriga_lib:CPPerror`; `icosphere(2.5)` errors if integer-ness is enforced; `uint64(2)^62` remains a valid, reproducible seed.

## B-035: Python `arrayant.interpolate(local_angles=True)` without `dist=True` returns uninitialised memory

- Status: fixed (WP-05)
- Severity: S2, because a documented option combination returns garbage for the local angles and `gamma` without an error.
- Kind: wrong result (uninitialised output)
- Mark: [R]
- Where: `api_python/qpy_arrayant_interpolate.cpp:191-196`. Since e8ece583.

**What happens.** `interpolate(arrayant, az, el, orientation=ori, local_angles=True)` returns `azimuth_loc`, `elevation_loc` and `gamma` filled with whatever `np.empty` contained (e.g. `[6.69e-310 0 0 6.69e-310 0 0]`). With `dist=True` added, the same call returns the correct local azimuth `[-0.6 -0.2 -1.7 ...]` (azimuth minus heading 0.7).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/G-2/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-2: arrayant.interpolate(local_angles=True) without dist=True returns uninitialized memory
import numpy as np, quadriga_lib as q
A = q.arrayant
ant = A.generate('3gpp', M=1, N=2, pol=1, res=10.0)
az = np.array([[0.1, 0.5, -1.0]]); el = np.array([[0.0, 0.2, -0.3]]); ori = np.array([[0.0], [0.0], [0.7]])
for i in range(3):
    r = A.interpolate(ant, az, el, orientation=ori, local_angles=True)   # documented usage line
    print("local_angles only : azimuth_loc =", r[4].ravel())
r = A.interpolate(ant, az, el, orientation=ori, dist=True, local_angles=True)
print("dist + local_angles: azimuth_loc =", r[5].ravel(), "(expected)")
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-2/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-2
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
local_angles only : azimuth_loc = [6.50280875e-310 0.00000000e+000 0.00000000e+000 6.50280875e-310
 0.00000000e+000 0.00000000e+000]
local_angles only : azimuth_loc = [6.50280875e-310 0.00000000e+000 4.29564862e-316 6.50280875e-310
 0.00000000e+000 0.00000000e+000]
local_angles only : azimuth_loc = [6.50280875e-310 0.00000000e+000 0.00000000e+000 6.50280875e-310
 0.00000000e+000 0.00000000e+000]
dist + local_angles: azimuth_loc = [-0.6 -0.2 -1.7 -0.6 -0.2 -1.7] (expected)
```

**Cause.** The wrapper allocates the three output arrays when `local_angles` is set, but the call chain `if (dist && local_angles) {...} else if (dist) {...} else {...}` has no branch for `local_angles && !dist`: the overload without local-angle outputs is called and the arrays are returned unwritten.

**Proposed fix.** When `local_angles` is set, always call the full overload of `arrayant::interpolate` and pass a scratch matrix for `dist` when `dist` is false. File: `api_python/qpy_arrayant_interpolate.cpp`.

**Test for the fix session.** pytest: with `local_angles=True` alone, `azimuth_loc`, `elevation_loc` and `gamma` equal the values returned with `dist=True, local_angles=True`.

## B-036: Python `channel.baseband_freq_response`: an exception inside the OpenMP region aborts the interpreter

- Status: fixed (WP-05)
- Severity: S2, because one invalid argument (an unsorted `freq_in`) kills the Python process with `std::terminate` instead of raising an exception.
- Kind: crash (exception safety)
- Mark: [R]
- Where: `api_python/qpy_baseband_freq_response.cpp:539-616` (multi-frequency branch); the single-frequency branch at 493-532 has the same structure. Since e0af5b12. The throw comes from `quadriga_lib::baseband_freq_response_multi` (`src/baseband_freq_response.cpp:449`), called inside `#pragma omp parallel`.

**What happens.** 4-D coefficients with `freq_in = [2e9, 1e9]`: the process prints `terminate called after throwing an instance of 'std::invalid_argument' ... Input 'freq_in' must be sorted in strictly ascending order.` and exits with code 134 (SIGABRT), also with `OMP_NUM_THREADS=1`/`2`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/G-4/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-4: baseband_freq_response: exception inside OpenMP region aborts the Python process
import numpy as np, quadriga_lib as q
rng = np.random.default_rng(0)
coeff = [rng.normal(size=(2, 2, 3, 2)) + 1j * rng.normal(size=(2, 2, 3, 2))]; delay = [rng.random((2, 2, 3)) * 1e-7]
print("calling with unsorted freq_in (expected: ValueError) ...", flush=True)
try:
    q.channel.baseband_freq_response(coeff=coeff, delay=delay, freq_in=np.array([2e9, 1e9]), freq_out=np.array([1.5e9]))
except Exception as e:
    print("clean exception:", type(e).__name__, e)
print("still alive")
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-4/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-4
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
calling with unsorted freq_in (expected: ValueError) ...
terminate called after throwing an instance of 'std::invalid_argument'
  what():  Input 'freq_in' must be sorted in strictly ascending order.
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/G-4//run.sh: line 4: 721989 Aborted                    (core dumped) OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py"
exit code 134
```

**Cause.** An exception thrown inside an OpenMP parallel region cannot propagate out of it; the runtime calls `std::terminate`. The wrapper has no try/catch inside the region. `qpy_get_channels_spherical.cpp` (batch mode) already uses the right pattern (catch inside, record under `omp critical`, rethrow after the region).

**Proposed fix.** Wrap each loop body in try/catch, store the first exception (`std::exception_ptr`) under `omp critical`, rethrow after the region; additionally validate `freq_in` (sorted, finite) once before the region. File: `api_python/qpy_baseband_freq_response.cpp` (both branches).

**Test for the fix session.** pytest: the call above raises `ValueError` and the interpreter survives (run it in a subprocess before the fix to avoid killing the test run).

## B-037: AVX2 antenna interpolation returns 0 where all four corner amplitudes are below 1.19e-7 (generic interpolates)

- Status: fixed (WP-08)
- Severity: S3: the AVX2 kernel differs from the generic one beyond rounding (S2 by the letter of the scale), but only for patterns whose absolute field amplitude is below 1.2e-7 (-138 dB) around the query; in deep nulls of normalised patterns the difference is a tiny value against 0.
- Kind: wrong result (AVX2 differs from generic)
- Mark: [R]
- Where: `src/fastmath_avx2.h:543, 559-561, 599-600, 679-680` (`_fm256_slerp_complex_ps`, mask `both_tiny`), used by `qd_arrayant_interpolate_avx2`. Related, different cause: B-076 (same function).

**What happens.** A custom pattern scaled by s and interpolated at 3 angles (values shown divided by s): s = 1e-7: generic 2.354853 2.019394 1.292355, AVX2 the same; s = 1e-8: generic unchanged, AVX2 0 0 0.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-03/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-03: AVX2 kernel returns zero for low-amplitude patterns (absolute float-eps threshold); generic kernel is correct
#include "quadriga_lib.hpp"
#include "qd_arrayant_interpolate.hpp"
#include "qd_arrayant_interpolate_avx2.hpp"
#include <cstdio>
int main()
{
    auto ant = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0, 5.0);
    arma::mat az = {{0.1, 0.5, 1.0}}, el = {{0.05, 0.2, -0.3}};
    arma::Col<unsigned> ie = {1};
    arma::cube ori(3, 1, 1, arma::fill::zeros);
    arma::mat pos(3, 1, arma::fill::zeros);
    for (double s : {1e-7, 1e-8})
    {
        auto b = ant.copy();
        b.e_theta_re *= s;
        arma::mat g[4], v[4];
        for (int i = 0; i < 4; ++i)
            g[i].zeros(1, 3), v[i].zeros(1, 3);
        qd_arrayant_interpolate(b.e_theta_re, b.e_theta_im, b.e_phi_re, b.e_phi_im, b.azimuth_grid, b.elevation_grid,
                                az, el, ie, ori, pos, g[0], g[1], g[2], g[3]);
        qd_arrayant_interpolate_avx2(b.e_theta_re, b.e_theta_im, b.e_phi_re, b.e_phi_im, b.azimuth_grid, b.elevation_grid,
                                     az, el, ie, ori, pos, v[0], v[1], v[2], v[3]);
        std::printf("scale %.0e: generic V_re/s = %.6f %.6f %.6f   avx2 V_re/s = %.6f %.6f %.6f\n", s,
                    g[0](0) / s, g[0](1) / s, g[0](2) / s, v[0](0) / s, v[0](1) / s, v[0](2) / s);
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-03/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-03
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(OMP_NUM_THREADS=1 "$D/repro" 2>&1 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
scale 1e-07: generic V_re/s = 2.354853 2.019394 1.292355   avx2 V_re/s = 2.354853 2.019394 1.292356
scale 1e-08: generic V_re/s = 2.354853 2.019394 1.292355   avx2 V_re/s = 0.000000 0.000000 0.000000
```

**Cause.** Lanes where both SLERP end points have an amplitude below float epsilon are set to zero. The generic kernel uses epsilon only to switch from SLERP to linear interpolation; it never zeroes.

**Proposed fix.** Return the linear result for `both_tiny` lanes (as the generic kernel does), or use a threshold relative to the larger amplitude. File: `src/fastmath_avx2.h`.

**Test for the fix session.** Catch2: AVX2 and generic interpolation of a pattern scaled by 1e-9 agree to a relative 1e-4.

## B-038: qdant writer stores element positions, coupling and pattern values with only 6 significant digits

- Status: fixed (WP-10) with single-equivalent precision relative to the element peak (user decisions of 2026-10-20: files need only single-precision accuracy and must stay as small as possible): dB magnitudes and phases with up to 6 decimals, one fewer per 20 dB below the element peak, so every sample stays within 6e-8 of the element peak; positions and coupling as the shortest text that reads back as the same float
- Severity: S3, because the precision loss is silent but small: 0.2 mm on a 123 m element position, about 1e-6 relative on coupling and magnitudes, far coarser phases than the 1e-5° the code intends.
- Kind: wrong result (precision)
- Mark: [R]
- Where: `src/qd_arrayant_qdant.cpp:401, 444, 456, 517, 532` (`raw_print` into an `std::ostringstream` with the default precision of 6), `qd_arrayant_qdant_write`.

**What happens.** `element_pos` x = 123.456789 reads back as 123.457 (error 0.000211 m); `coupling_re` 0.123456789 reads back as 0.123457. The centre frequency (written by pugixml with 17 digits) and the grids (rounded to 0.001°) round-trip as intended.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-10/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-10: qdant writer stores numbers with 6 significant digits (element position 123.456789 m -> 123.457 m)
#include "quadriga_lib.hpp"
#include <cstdio>
#include <filesystem>
int main()
{
    std::string fn = "build_linux/scratch/hunt/final/A-10/a10.qdant";
    std::filesystem::remove(fn);
    auto a = quadriga_lib::generate_arrayant_omni<double>(10.0);
    a.element_pos = arma::mat(3, 1, arma::fill::zeros);
    a.element_pos(0, 0) = 123.456789, a.element_pos(1, 0) = 0.000123456789;
    a.coupling_re(0, 0) = 0.123456789;
    a.qdant_write(fn);
    auto b = quadriga_lib::qdant_read<double>(fn, 1);
    std::printf("element_pos x: %.9f -> %.9f (error %.3g m)\n", a.element_pos(0, 0), b.element_pos(0, 0), std::abs(a.element_pos(0, 0) - b.element_pos(0, 0)));
    std::printf("element_pos y: %.12f -> %.12f\n", a.element_pos(1, 0), b.element_pos(1, 0));
    std::printf("coupling_re  : %.9f -> %.9f\n", a.coupling_re(0, 0), b.coupling_re(0, 0));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-10/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-10
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(OMP_NUM_THREADS=1 "$D/repro" 2>&1 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
element_pos x: 123.456789000 -> 123.457000000 (error 0.000211 m)
element_pos y: 0.000123456789 -> 0.000123457000
coupling_re  : 0.123456789 -> 0.123457000
```

**Cause.** The stream is never given a precision.

**Proposed fix.** `strm << std::setprecision(std::numeric_limits<dtype>::max_digits10)` before the `raw_print` calls (or a fixed number of digits for dB and phase values). Files get larger. File: `src/qd_arrayant_qdant.cpp`.

**Test for the fix session.** Catch2: round trip of `element_pos(0, 1) = 123.456789`; error below 1e-9 for double.

## B-039: `qdant_write_multi` deletes an existing file before it validates its input

- Status: fixed (WP-01)
- Severity: S3, because invalid input destroys the user's existing file, while the documentation says all entries are validated first.
- Kind: wrong result (data loss on invalid input) / doc
- Mark: [R]
- Where: `src/qd_arrayant_xio.cpp:54-55` (`std::remove(fn)`) runs before 58-63 (validation of each entry), `qdant_write_multi`.

**What happens.** An existing file, then `qdant_write_multi` with an invalid second entry: the call throws "Entry 1 is invalid: Missing data ..." and the file no longer exists.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-11/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-11: qdant_write_multi deletes the existing file before validating its input
#include "quadriga_lib.hpp"
#include <cstdio>
#include <filesystem>
int main()
{
    std::string fn = "build_linux/scratch/hunt/final/A-11/a11.qdant";
    auto a = quadriga_lib::generate_arrayant_omni<double>(10.0);
    a.qdant_write(fn);
    std::printf("before: file exists = %d\n", (int)std::filesystem::exists(fn));
    std::vector<quadriga_lib::arrayant<double>> vec(2, a);
    vec[1].e_theta_im.reset(); // invalid entry
    try
    {
        quadriga_lib::qdant_write_multi(fn, vec);
    }
    catch (const std::exception &e)
    {
        std::printf("qdant_write_multi threw: %s\n", e.what());
    }
    std::printf("after: file exists = %d\n", (int)std::filesystem::exists(fn));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-11/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-11
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(OMP_NUM_THREADS=1 "$D/repro" 2>&1 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
before: file exists = 1
qdant_write_multi threw: qdant_write_multi: Entry 1 is invalid: Missing data for any of: e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid
after: file exists = 0
```

**Proposed fix.** Move the removal after the validation loop. File: `src/qd_arrayant_xio.cpp`.

**Test for the fix session.** Catch2: create a file, call `qdant_write_multi` with an invalid entry, expect the exception, assert the file still exists unchanged.

## B-040: `qdant_read(fn, 1)` returns another entry without error when the file has no ID 1

- Status: fixed (WP-10)
- Severity: S3, because the request is accepted and a different antenna is returned.
- Kind: wrong result
- Mark: [R]
- Where: `src/qd_arrayant_qdant.cpp:85-86` (`qd_arrayant_qdant_read`: fallback to the first `<arrayant>` node, meant for files without an `id` attribute).

**What happens.** A file with entries ID 2 ("two") and ID 3 ("three"): `qdant_read(fn, 1)` returns "two" without an error.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-12/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-12: qdant_read(fn, 1) silently returns a different entry when ID 1 does not exist
#include "quadriga_lib.hpp"
#include <cstdio>
#include <filesystem>
int main()
{
    std::string fn = "build_linux/scratch/hunt/final/A-12/a12.qdant";
    std::filesystem::remove(fn);
    auto a = quadriga_lib::generate_arrayant_omni<double>(10.0);
    a.name = "two";
    a.qdant_write(fn, 2);
    a.name = "three";
    a.qdant_write(fn, 3);
    try
    {
        auto c = quadriga_lib::qdant_read<double>(fn, 1);
        std::printf("read id=1 from a file with ids {2,3}: returned '%s' (no error)\n", c.name.c_str());
    }
    catch (const std::exception &e)
    {
        std::printf("read id=1: error '%s'\n", e.what());
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-12/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-12
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(OMP_NUM_THREADS=1 "$D/repro" 2>&1 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
read id=1 from a file with ids {2,3}: returned 'two' (no error)
```

**Proposed fix.** Use the fallback only when the first `<arrayant>` node has no `id` attribute; otherwise throw "ID not found". File: `src/qd_arrayant_qdant.cpp`.

**Test for the fix session.** Catch2: a file with IDs 2 and 3; `qdant_read(fn, 1)` throws `std::invalid_argument`.

## B-041: `arrayant::qdant_write` without a layout replaces an existing custom layout in the file (user decision)

- Status: documented (WP-10): qdant_write without a layout resets the layout (decision (b))
- Severity: S3, because appending an antenna silently discards a layout the user wrote before; this may be intended, but the documentation does not say so.
- Kind: doc / wrong result
- Mark: [R]
- Where: `src/qd_arrayant_qdant.cpp:554-560` (`qd_arrayant_qdant_write`, layout handling when no layout is given).

**What happens.** A file with the 2×2 layout `[1 2; 3 3]`; appending ID 4 without a layout leaves the layout `1 2 3 4`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-13/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-13: qdant_write without a layout replaces an existing custom layout in the file
#include "quadriga_lib.hpp"
#include <cstdio>
#include <filesystem>
int main()
{
    std::string fn = "build_linux/scratch/hunt/final/A-13/a13.qdant";
    std::filesystem::remove(fn);
    auto a = quadriga_lib::generate_arrayant_omni<double>(30.0);
    a.qdant_write(fn, 1), a.qdant_write(fn, 2), a.qdant_write(fn, 3);
    arma::u32_mat L = {{1, 2}, {3, 3}};
    a.qdant_write(fn, 3, L); // custom 2x2 layout
    arma::u32_mat L1, L2;
    quadriga_lib::qdant_read<double>(fn, 1, &L1);
    L1.print("layout after writing it:");
    a.qdant_write(fn); // append id 4, no layout given
    quadriga_lib::qdant_read<double>(fn, 1, &L2);
    L2.print("layout after appending id 4 without layout:");
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-13/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-13
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(OMP_NUM_THREADS=1 "$D/repro" 2>&1 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
layout after writing it:
        1        2
        3        3
layout after appending id 4 without layout:
        1        2        3        4
```

**Proposed fix.** Keep an existing layout node when no layout is given, or document the reset in the `qdant_write` block. File: `src/qd_arrayant_qdant.cpp`. The user decides which side.

**Test for the fix session.** Catch2 (if the behaviour changes): write a custom layout, append a fourth entry without a layout, assert the layout is unchanged.

## B-042: A 138-byte qdant file makes the reader allocate 4.7 GB and fail with `std::bad_alloc`

- Status: fixed (WP-10)
- Severity: S3, because unchecked file input leads to an unbounded allocation and an unclear exception type instead of a clean error.
- Kind: resource (unbounded allocation)
- Mark: [R]
- Where: `src/qd_arrayant_qdant.cpp:103` (`NoElements` read as an unsigned long long, unchecked), 109 (`zeros(3, n)`), 136 (`eye(n, n)` coupling), `qd_arrayant_qdant_read`.

**What happens.** The file `<qdant><arrayant id="1"><NoElements>200000000</NoElements><ElevationGrid>0</ElevationGrid><AzimuthGrid>0</AzimuthGrid></arrayant></qdant>` under `ulimit -v 8000000`: `std::bad_alloc` after a maximum RSS of 4.69 GB. (The 15-minute AFL campaign on the qdant reader did not find this, because the harness ran with ASan's allocator limits; it is an allocation, not a memory error.)

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-14/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-14: A 138-byte qdant file with NoElements = 2e8 makes the reader allocate gigabytes, then throw std::bad_alloc
#include "quadriga_lib.hpp"
#include <cstdio>
#include <fstream>
#include <typeinfo>
int main()
{
    std::string fn = "build_linux/scratch/hunt/final/A-14/huge.qdant";
    std::ofstream(fn) << "<qdant><arrayant id=\"1\"><NoElements>200000000</NoElements><ElevationGrid>0</ElevationGrid>"
                         "<AzimuthGrid>0</AzimuthGrid></arrayant></qdant>\n";
    try
    {
        auto a = quadriga_lib::qdant_read<double>(fn);
        std::printf("read OK, n_elements = %llu\n", (unsigned long long)a.n_elements());
    }
    catch (const std::exception &e)
    {
        std::printf("exception type %s: %s\n", typeid(e).name(), e.what());
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-14/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-14
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(ulimit -v 8000000; OMP_NUM_THREADS=1 /usr/bin/time -f "max RSS %M kB" "$D/repro" 2>&1 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
exception type St9bad_alloc: std::bad_alloc
max RSS 4690260 kB
```

**Proposed fix.** Before allocating, check `NoElements` against the number of `EthetaMag`/`EphiMag` nodes or `ElementPosition` columns actually present, and cap it; throw `std::invalid_argument`. File: `src/qd_arrayant_qdant.cpp`.

**Test for the fix session.** Catch2: read the file above; expect `std::invalid_argument` quickly.

## B-043: `arrayant` default constructor leaves `check_ptr` uninitialised; the quick `is_valid()` can pass an invalid object

- Status: fixed (WP-01)
- Severity: S3, because reading indeterminate pointers is undefined behaviour, and a stale value lets a never-validated, invalid arrayant pass the quick check; the full check (`is_valid(false)`) is not affected.
- Kind: UB
- Mark: [R]
- Where: `include/quadriga_arrayant.hpp:32-33` (`dtype *check_ptr[9];`, `arrayant() {};`), read in `src/qd_arrayant.cpp:2077-2088` (`is_valid` quick path).

**What happens.** The same function is called twice so that the arrayant lands in the same stack slot (small matrices live inside the object, so their pointers repeat). Call 1 validates a valid object; in call 2 an invalid object (unsorted azimuth grid) gives `is_valid() = ''` but `is_valid(false) = 'Values of 'azimuth_grid' must be sorted in ascending order.'`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-15/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-15: check_ptr is not initialised by the default constructor; stale values make is_valid() pass an invalid object
// The function is called twice; the arrayant lands in the same stack slot. Matrices <= 16 elements live inside the
// Armadillo objects, so the second object's data pointers equal the stale check_ptr values of the first one.
#include "quadriga_lib.hpp"
#include <cstdio>
__attribute__((noinline)) static void run(bool first)
{
    quadriga_lib::arrayant<double> a; // check_ptr[] indeterminate
    a.e_theta_re.ones(2, 2, 1), a.e_theta_im.zeros(2, 2, 1), a.e_phi_re.zeros(2, 2, 1), a.e_phi_im.zeros(2, 2, 1);
    a.elevation_grid = {-0.5, 0.5};
    a.element_pos.zeros(3, 1), a.coupling_re.ones(1, 1), a.coupling_im.zeros(1, 1);
    if (first)
    {
        a.azimuth_grid = {-1.0, 1.0};
        std::printf("call 1: validate() = '%s'\n", a.validate().c_str());
    }
    else
    {
        a.azimuth_grid = {1.0, -1.0}; // invalid: unsorted; never validated
        std::printf("call 2: is_valid() = '%s'   is_valid(false) = '%s'\n", a.is_valid().c_str(), a.is_valid(false).c_str());
    }
}
int main()
{
    run(true);
    run(false);
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-15/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-15
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(OMP_NUM_THREADS=1 "$D/repro" 2>&1 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
call 1: validate() = ''
call 2: is_valid() = ''   is_valid(false) = 'Values of 'azimuth_grid' must be sorted in ascending order.'
```

**Proposed fix.** `dtype *check_ptr[9] = {};` in the header. File: `include/quadriga_arrayant.hpp`.

**Test for the fix session.** Catch2 with the repro scenario: `is_valid()` returns a non-empty message in call 2.

## B-044: ULA and 3GPP generators keep the custom pattern's element positions and coupling, contrary to the docs

- Status: fixed (WP-09); the generators reset positions and coupling of a custom pattern, as documented (behaviour change)
- Severity: S3, because the documentation says these are ignored; the code keeps them, and in the ULA case the coupling becomes inconsistent (neither identity nor replicated).
- Kind: doc / wrong result
- Mark: [R]
- Where: `src/qd_arrayant_generate.cpp:435` (`generate_arrayant_ula`) and `549` (`generate_arrayant_3GPP`): `pattern->copy()` keeps `element_pos` and the coupling, `copy_element` then extends the coupling. Documentation: `include/quadriga_arrayant.hpp` ("Values for coupling, element positions and center frequency of the custom pattern are ignored").

**What happens.** A 2-element pattern with x = 0.3, z = ±0.7 and coupling [1; 1]: the ULA with N = 2 has `n_elements = 4, n_ports = 3`, keeps x = 0.3 and z = ±0.7, and coupling `[1 0 0; 1 0 0; 0 1 0; 0 0 1]`; 3GPP with M = N = 1 has 1 port and the same positions.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-16/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-16: ULA / 3GPP generators keep the custom pattern's element positions and coupling (documented as ignored)
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    auto p = quadriga_lib::generate_arrayant_omni<double>(10.0);
    p.copy_element(0, 1);
    p.element_pos = arma::mat({{0.3, 0.3}, {0.0, 0.0}, {0.7, -0.7}});
    p.coupling_re = arma::mat(2, 1, arma::fill::ones); // 2 elements -> 1 port
    p.coupling_im = arma::mat(2, 1, arma::fill::zeros);
    auto u = quadriga_lib::generate_arrayant_ula<double>(2, 3e9, 0.5, &p);
    std::printf("ULA N=2: n_elements=%llu n_ports=%llu (doc: coupling ignored -> 4 ports)\n", (unsigned long long)u.n_elements(), (unsigned long long)u.n_ports());
    u.element_pos.raw_print(std::cout, "ULA element_pos (doc: pattern positions ignored):");
    u.coupling_re.raw_print(std::cout, "ULA coupling_re:");
    auto g = quadriga_lib::generate_arrayant_3GPP<double>(1, 1, 3e9, 1, 0.0, 0.5, 1, 1, 0.5, 0.5, &p);
    std::printf("3GPP M=N=1: n_ports=%llu, element_pos x = %g %g, z = %g %g\n", (unsigned long long)g.n_ports(),
                g.element_pos(0, 0), g.element_pos(0, 1), g.element_pos(2, 0), g.element_pos(2, 1));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-16/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-16
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(OMP_NUM_THREADS=1 "$D/repro" 2>&1 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
ULA N=2: n_elements=4 n_ports=3 (doc: coupling ignored -> 4 ports)
ULA element_pos (doc: pattern positions ignored):
0.3 0.3 0.3 0.3
-0.0249827 -0.0249827 0.0249827 0.0249827
0.7 -0.7 0.7 -0.7
ULA coupling_re:
1 0 0
1 0 0
0 1 0
0 0 1
3GPP M=N=1: n_ports=1, element_pos x = 0.3 0.3, z = 0.7 -0.7
```

**Proposed fix.** After `pattern->copy()`, reset `element_pos` to zeros and the coupling to identity, as documented. Alternative: change the documentation and replicate the coupling block-diagonally. File: `src/qd_arrayant_generate.cpp`.

**Test for the fix session.** Catch2: with the pattern above, assert `n_ports == n_elements` and zero x/z positions before the array geometry is applied.

## B-045: `combine_pattern` and `arrayant_combine_pattern_multi` hard-code the speed of light (acoustic arrays combined without position phase)

- Status: fixed (WP-10)
- Severity: S3, because the acoustic multi-driver workflow (`generate_speaker`, `set_element_pos_multi`, `concat_multi`, `combine_pattern_multi`) silently ignores the driver spacing; there is no `propagation_speed` argument, unlike `get_channels_multifreq`.
- Kind: wrong result / doc
- Mark: [R]
- Where: `src/qd_arrayant_xio.cpp:979` (`C0` in `arrayant_combine_pattern_multi`), `src/qd_arrayant.cpp:553` (`combine_pattern`).

**What happens.** Two omni speakers 0.2 m apart at 1 kHz with coupling [1; 1]: |p| = 2.000 at az = 0 and at az = 90°; with the speed of sound (344 m/s) the expected values are 2.000 and 0.506.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-17/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-17: arrayant_combine_pattern_multi assumes the speed of light; acoustic arrays combine without position phase
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    // two omni "speakers" 0.2 m apart at 1 kHz (lambda = 0.344 m in air), summed into one port
    auto s = quadriga_lib::generate_speaker<double>("omni", 0.05, 80.0, 12000.0, 12.0, 12.0, 85.0, "monopole",
                                                    0, 0, 0, 0.15, 0.25, arma::vec{1000.0}, 10.0);
    auto two = quadriga_lib::arrayant_concat_multi(s, s);
    quadriga_lib::arrayant_set_element_pos_multi(two, arma::mat({{0.0, 0.0}, {-0.1, 0.1}, {0.0, 0.0}}));
    two[0].coupling_re = arma::mat(2, 1, arma::fill::ones);
    two[0].coupling_im = arma::mat(2, 1, arma::fill::zeros);
    auto cm = quadriga_lib::arrayant_combine_pattern_multi(two);
    arma::uword ie = 9, ia0 = 18, ia90 = 27; // el = 0; az = 0 and 90 deg on the 10 deg grid
    std::printf("|p(az=0)| = %.3f   |p(az=90)| = %.3f   expected with c=344 m/s: 2.000 and %.3f\n",
                std::hypot(cm[0].e_theta_re(ie, ia0, 0), cm[0].e_theta_im(ie, ia0, 0)),
                std::hypot(cm[0].e_theta_re(ie, ia90, 0), cm[0].e_theta_im(ie, ia90, 0)),
                2.0 * std::abs(std::cos(arma::datum::pi * 1000.0 * 0.2 / 344.0)));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-17/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-17
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(OMP_NUM_THREADS=1 "$D/repro" 2>&1 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
|p(az=0)| = 2.000   |p(az=90)| = 2.000   expected with c=344 m/s: 2.000 and 0.506
```

**Proposed fix.** Add an optional `propagation_speed` argument (default 299792458) to `arrayant_combine_pattern_multi` and `combine_pattern` (an API change), or at least document that the functions assume electromagnetic waves. Files: `src/qd_arrayant_xio.cpp`, `src/qd_arrayant.cpp`, `include/quadriga_arrayant.hpp`, wrappers if extended.

**Test for the fix session.** Catch2: the case above with speed 344: |p(az = 90°)| ≈ 0.506.

## B-046: `remove_zeros` and `calc_directivity_dBi` use absolute power thresholds (low-level patterns lose data or report 0 dBi)

- Status: fixed (WP-09)
- Severity: S3, because valid but low-level patterns lose non-zero samples or become empty (`rotate_pattern` with a grid update calls `remove_zeros`), and directivity changes with the pattern scale.
- Kind: wrong result
- Mark: [R]
- Where: `src/qd_arrayant.cpp:1370` (`remove_zeros`, `limit = 1e-12` on summed power), `src/qd_arrayant.cpp:469` (`calc_directivity_dBi`, `p_max < 1e-14` returns 0 dBi); the all-zero case then passes empty matrices to `memcpy` at 1509 and 1524 (UBSan).

**What happens.** A 37×73 custom pattern scaled by s: s = 1: 7.512 dBi, `remove_zeros` keeps 37×73; s = 1e-6: keeps 35×49 (non-zero samples removed); s = 1e-7: 9×5; s = 1e-8: 0.000 dBi and 0×0, with UBSan "null pointer passed as argument 1" at `qd_arrayant.cpp:1509` and `1524`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-18/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-18: remove_zeros and calc_directivity_dBi use absolute thresholds; low-level patterns lose data
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    auto a = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0, 5.0); // 37 x 73
    for (double s : {1.0, 1e-6, 1e-7, 1e-8})
    {
        auto b = a.copy();
        b.e_theta_re *= s;
        double d = b.calc_directivity_dBi(0);
        quadriga_lib::arrayant<double> o;
        b.remove_zeros(&o);
        std::printf("scale %.0e: directivity = %.3f dBi, remove_zeros -> %llu x %llu (input 37 x 73)\n", s, d,
                    (unsigned long long)o.n_elevation(), (unsigned long long)o.n_azimuth());
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-18/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-18
g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_asan/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(ASAN_OPTIONS=halt_on_error=1 OMP_NUM_THREADS=1 "$D/repro" 2>&1 | grep -E "ERROR: |SUMMARY|runtime error|^    #[0-4] |^[^ =]" | grep -v "^Shadow\|^  0x\|^=>\|^Address\|^Hint\|^HINT\|^AddressSanitizer can" | cut -c1-200 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
/sjc/quadriga-lib-fixes/src/qd_arrayant.cpp:1509:16: runtime error: null pointer passed as argument 1, which is declared to never be null
/sjc/quadriga-lib-fixes/src/qd_arrayant.cpp:1524:16: runtime error: null pointer passed as argument 1, which is declared to never be null
scale 1e+00: directivity = 7.512 dBi, remove_zeros -> 37 x 73 (input 37 x 73)
scale 1e-06: directivity = 7.512 dBi, remove_zeros -> 35 x 49 (input 37 x 73)
scale 1e-07: directivity = 7.512 dBi, remove_zeros -> 9 x 5 (input 37 x 73)
scale 1e-08: directivity = 0.000 dBi, remove_zeros -> 0 x 0 (input 37 x 73)
```

**Proposed fix.** Make both thresholds relative to the peak power; if everything is zero, leave the pattern unchanged instead of emptying it. File: `src/qd_arrayant.cpp`.

**Test for the fix session.** Catch2: `remove_zeros` and `calc_directivity_dBi` give the same grid size and directivity for s = 1 and s = 1e-8.

## B-047: `generate_arrayant_multibeam` defaults M = N = 2 in the header, 1 in the docs and both wrappers

- Status: fixed (WP-10)
- Severity: S3, because a C++ call that relies on the defaults gets a 2×2 array while the documentation and the MATLAB/Python wrappers say 1×1.
- Kind: doc / cross-API
- Mark: [C]
- Where: `include/quadriga_arrayant.hpp:294-295` (`M = 2`, `N = 2`); the MD block in `src/qd_arrayant_generate.cpp:737-738` says 1 and 1; `api_mex/arrayant_generate.cpp:150-151` uses 1 and 1; the Python registration uses 1 and 1.

**What happens / Reproduce.** Read from the code: compare the four places above.

**Proposed fix.** Change the header defaults to 1 (behaviour change for C++ callers that rely on the default) or the documentation to 2. The user decides. File: `include/quadriga_arrayant.hpp` or the MD block.

**Test for the fix session.** Catch2: `generate_arrayant_multibeam<double>()` with defaults has the documented number of elements.

## B-048: QRT readers trust header and path counts: a 136-byte file returns 20 million zero paths or allocates GBs

- Status: fixed (WP-07)
- Severity: S3, because unchecked counts in a malformed file lead to accepted-but-wrong results and multi-GB allocations instead of an error; no memory corruption was found.
- Kind: resource / invalid input accepted
- Mark: [R] results, [M] memory and time; [C] for the uninitialised values at the end
- Where: `src/qd_channel_qrt.cpp`: `qrt_read_cache_init` 95-157, `qrt_file_parse` 713-794 (`zeros()`, `dest_names.resize`), `qrt_file_read` 1288-1327, `qrt_file_read_raw` 1674-1706.

**What happens.** From a valid 136-byte v5 file with one count patched: `no_path` = 2e7 makes `qrt_file_read` return 20 000 000 all-zero paths without an exception (2.5 GB RSS, 2.9 s; 1e8 gave 12.5 GB and 11.4 s in phase 1); `no_orig` = 1e8 makes `qrt_file_parse` return `no_orig = 100000000` (4.3 GB, 3.6 s); `no_dest` = 1e7 returns 1e7 empty RX names. The phase-0 AFL campaign found 28 hang inputs of 96 and 371 bytes, which run 30 to over 65 s each for the same reason. By reading: `unsigned no_path` stays uninitialised when the seek lands past the end of the file (1288-1289), and `l_orig_index` is read from memory that was never filled when the file is short.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-009/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 09: QRT readers size their buffers from 32-bit header counts without comparing them
// with the file size. A tiny valid v5 file is written with qrt_file_init + qrt_file_append, then one
// count is patched in place. Usage: p09 <field> <value> [reader]
//   field  = no_orig | no_cir | no_dest | no_freq  (header, byte offsets 11/15/19/23)
//          | no_path                               (path count of CIR 0 in the data block)
//   reader = parse (qrt_file_parse with all outputs, default) | read (qrt_file_read) | raw (qrt_file_read_raw)
#include "quadriga_lib.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string field = argc > 1 ? argv[1] : "no_cir";
    unsigned value = argc > 2 ? (unsigned)std::strtoul(argv[2], nullptr, 10) : 100000000u;
    std::string reader = argc > 3 ? argv[3] : "parse";
    const std::string fn = "build_linux/scratch/hunt/final/B-009/p09_" + field + ".qrt";

    arma::vec freq = {3.0};
    arma::mat cir_pos = {{1.0, 2.0, 3.0}};
    quadriga_lib::qrt_file_init<double>(fn, freq, cir_pos, arma::mat());
    std::vector<quadriga_lib::path> pd(1);
    quadriga_lib::qrt_file_append<double>(fn, pd, arma::vec{0.0, 0.0, 10.0});

    size_t pos = field == "no_orig" ? 11 : field == "no_cir" ? 15 : field == "no_dest" ? 19 : 23;
    if (field == "no_path")
    {
        auto cache = quadriga_lib::qrt_read_cache_init(fn);
        std::ifstream r(fn, std::ios::binary);
        r.seekg((std::streamoff)cache.path_data_offset(0));
        r.read((char *)&pos, sizeof(size_t)); // data_offset of CIR 0 = position of its path count
    }
    {
        std::fstream f(fn, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp((std::streamoff)pos);
        f.write((char *)&value, sizeof(unsigned));
    }
    std::ifstream sz(fn, std::ios::binary | std::ios::ate);
    std::printf("file size %lld bytes, %s = %u at byte %zu, reader = %s\n", (long long)sz.tellg(), field.c_str(), value, pos, reader.c_str());

    auto t0 = std::chrono::steady_clock::now();
    try
    {
        if (reader == "parse")
        {
            arma::uword a, b, c, d; arma::uvec co; std::vector<std::string> on, dn; int v;
            arma::fvec f; arma::fmat cp, cr, op, oo;
            quadriga_lib::qrt_file_parse(fn, &a, &b, &c, &d, &co, &on, &dn, &v, &f, &cp, &cr, &op, &oo);
            std::printf("returned without exception: no_cir=%llu no_orig=%llu no_dest=%llu no_freq=%llu\n",
                        (unsigned long long)a, (unsigned long long)b, (unsigned long long)c, (unsigned long long)d);
        }
        else if (reader == "read")
        {
            arma::vec fc, pl; arma::mat pg; arma::cube M; arma::u32_vec ni;
            quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, &fc, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                &pg, &pl, &M, nullptr, nullptr, nullptr, nullptr, nullptr, 1, &ni);
            std::printf("returned without exception: n_path=%llu, n_freq=%llu\n", (unsigned long long)pl.n_elem, (unsigned long long)fc.n_elem);
        }
        else
        {
            auto p = quadriga_lib::qrt_file_read_raw(fn, 0);
            std::printf("returned without exception: %zu paths\n", p.size());
        }
    }
    catch (const std::exception &e) { std::printf("exception: %s\n", e.what()); }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("reader time %.2f s\n", s);
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-009/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-009
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$REL "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $H5
echo "load before: $(cat /proc/loadavg)"
(ulimit -v 30000000; /usr/bin/time -f 'max RSS %M KB, wall %e s' "$D/repro" no_path 20000000 read) 2>&1 || echo "exit code $?"
(ulimit -v 30000000; /usr/bin/time -f 'max RSS %M KB, wall %e s' "$D/repro" no_orig 100000000 parse) 2>&1 || echo "exit code $?"
(ulimit -v 30000000; /usr/bin/time -f 'max RSS %M KB, wall %e s' "$D/repro" no_dest 10000000 parse) 2>&1 || echo "exit code $?"
echo "load after: $(cat /proc/loadavg)"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
load before: 1.66 5.76 8.35 1/2793 713039
file size 136 bytes, no_path = 20000000 at byte 99, reader = read
returned without exception: n_path=20000000, n_freq=1
reader time 2.32 s
max RSS 2503484 KB, wall 2.32 s
file size 136 bytes, no_orig = 100000000 at byte 11, reader = parse
returned without exception: no_cir=1 no_orig=100000000 no_dest=1 no_freq=1
reader time 3.00 s
max RSS 4300412 KB, wall 3.00 s
file size 136 bytes, no_dest = 10000000 at byte 19, reader = parse
returned without exception: no_cir=1 no_orig=1 no_dest=10000000 no_freq=1
reader time 0.40 s
max RSS 394660 KB, wall 0.40 s
load after: 1.61 5.68 8.31 1/2786 713088
```

**Cause.** Counts are never compared with the file size, and the stream state is not checked after reads.

**Proposed fix.** Get the file size once; before each allocation check `count * element_size <= remaining bytes`; check the stream after every read and throw "file truncated or corrupt"; initialise `no_path = 0`. File: `src/qd_channel_qrt.cpp`.

**Test for the fix session.** Catch2: the 136-byte file with `no_path` or `no_orig` patched to 1e8 throws `std::invalid_argument` within 0.1 s (run under `ulimit -v` 1 GB).

## B-049: `qrt_file_read` truncates `i_cir` / `i_orig` to 32 bits before the range check

- Status: fixed (WP-07)
- Severity: S3, because an out-of-range index is accepted (returns another CIR without paths) or rejected with Armadillo's internal message, depending on the cache.
- Kind: invalid input accepted
- Mark: [R]
- Where: `src/qd_channel_qrt.cpp:1016/1018, 1069/1071` (`qrt_file_read`) and 1655 (`qrt_file_read_raw`): `(unsigned)i_cir`, `(unsigned)i_orig`.

**What happens.** `i_cir = 2^32` without a cache: accepted, returns the position of CIR 0 and 0 paths; with a cache: "Mat::operator(): index out of bounds". (The output of `run.sh` also contains the lines of B-016, which uses the same probe.)

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-010/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 10: qrt_file_read behaviour on the library's own test files.
//  (a) cache path vs. non-cache path, every origin/CIR/direction, all outputs
//  (b) angles requested without path_gain/path_length
//  (c) i_cir / i_orig >= 2^32 (truncated to unsigned before the range check)
#include "quadriga_lib.hpp"
#include <cstdio>
struct Out { arma::vec fc, tp, to, rp, ro, pl, aod, eod, aoa, eoa; arma::mat fbs, lbs, pg; arma::cube M; std::vector<arma::mat> pc; arma::u32_vec ni; arma::fmat co; std::vector<uint8_t> it; };
static void rd(const std::string &fn, arma::uword c, arma::uword o, bool dl, Out &r, std::ifstream *f, const quadriga_lib::qrt_read_cache *cache)
{
    quadriga_lib::qrt_file_read<double>(fn, c, o, dl, &r.fc, &r.tp, &r.to, &r.rp, &r.ro, &r.fbs, &r.lbs, &r.pg, &r.pl, &r.M,
                                        &r.aod, &r.eod, &r.aoa, &r.eoa, &r.pc, 1, &r.ni, &r.co, &r.it, f, cache);
}
template <typename T> static double d(const T &a, const T &b)
{ if (arma::size(a) != arma::size(b)) return 1e99; if (a.n_elem == 0) return 0; return arma::abs(arma::vectorise(arma::conv_to<arma::mat>::from(arma::vectorise(a)) - arma::conv_to<arma::mat>::from(arma::vectorise(b)))).max(); }
int main()
{
    for (std::string fn : {"tests/data/test.qrt", "tests/data/test_v5.qrt"})
    {
        std::ifstream f(fn, std::ios::binary);
        auto cache = quadriga_lib::qrt_read_cache_init(fn, &f);
        double worst = 0; std::string where;
        for (arma::uword o = 0; o < cache.no_orig; ++o)
            for (arma::uword c = 0; c < cache.no_cir; ++c)
                for (bool dl : {true, false})
                {
                    Out a, b;
                    rd(fn, c, o, dl, a, nullptr, nullptr);
                    rd(fn, c, o, dl, b, &f, &cache);
                    double m = std::max({d(a.fc, b.fc), d(a.tp, b.tp), d(a.to, b.to), d(a.rp, b.rp), d(a.ro, b.ro), d(a.pl, b.pl), d(a.pg, b.pg), d(a.M, b.M), d(a.aod, b.aod), d(a.eoa, b.eoa), d(a.fbs, b.fbs), d(a.co, b.co)});
                    if (m > worst) { worst = m; where = "orig " + std::to_string(o) + " cir " + std::to_string(c) + (dl ? " DL" : " UL"); }
                    if (o == cache.no_orig - 1 && c == 0 && dl)
                        std::printf("%s v%d: orig %llu cir 0: center_frequency no-cache %.6g Hz, cache %.6g Hz\n", fn.c_str(), cache.version, (unsigned long long)o, a.fc(0), b.fc(0));
                }
        std::printf("%s: %u orig x %u cir, worst cache/no-cache difference %g at %s\n", fn.c_str(), cache.no_orig, cache.no_cir, worst, where.c_str());

        // (b) angles only
        arma::vec aod1, aod2, pl;
        aod1 = {12345.0};
        quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &aod1);
        quadriga_lib::qrt_file_read<double>(fn, 0, 0, true, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &pl, nullptr, &aod2);
        std::printf("%s: aod requested alone -> %llu elements (first %g); aod with path_length -> %llu elements\n", fn.c_str(),
                    (unsigned long long)aod1.n_elem, aod1.n_elem ? aod1(0) : 0.0, (unsigned long long)aod2.n_elem);

        // (c) index >= 2^32
        for (int use_cache = 0; use_cache < 2; ++use_cache)
            try
            {
                Out r;
                rd(fn, (1ULL << 32), 0, true, r, use_cache ? &f : nullptr, use_cache ? &cache : nullptr);
                std::printf("%s: i_cir = 2^32 (%s) accepted: rx_pos = [%g %g %g], %llu paths\n", fn.c_str(), use_cache ? "cache" : "no cache",
                            r.rp(0), r.rp(1), r.rp(2), (unsigned long long)r.pl.n_elem);
            }
            catch (const std::exception &e) { std::printf("%s: i_cir = 2^32 (%s): exception: %s\n", fn.c_str(), use_cache ? "cache" : "no cache", e.what()); }
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-010/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-010
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$ASAN "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $H5
"$D/repro_asan" 2>&1 | sort | uniq -c | sort -rn | head -20
```

Output of the phase-2 re-run (`bash run.sh`):
```text
      4 /usr/include/armadillo_bits/access.hpp:26:100: runtime error: reference binding to null pointer of type 'double'
      4 /usr/include/armadillo_bits/Mat_meat.hpp:7490:26: runtime error: reference binding to null pointer of type 'const double'
      1 tests/data/test_v5.qrt: i_cir = 2^32 (no cache) accepted: rx_pos = [-2.37388 -3.83962 0], 0 paths
      1 tests/data/test_v5.qrt: i_cir = 2^32 (cache): exception: Mat::operator(): index out of bounds
      1 tests/data/test_v5.qrt: aod requested alone -> 1 elements (first 12345); aod with path_length -> 2 elements
      1 tests/data/test_v5.qrt: 1 orig x 2 cir, worst cache/no-cache difference 0 at 
      1 tests/data/test_v5.qrt v5: orig 0 cir 0: center_frequency no-cache 1e+09 Hz, cache 1e+09 Hz
      1 tests/data/test.qrt: i_cir = 2^32 (no cache) accepted: rx_pos = [-8.83498 57.1893 1], 0 paths
      1 tests/data/test.qrt: i_cir = 2^32 (cache): exception: Mat::operator(): index out of bounds
      1 tests/data/test.qrt: aod requested alone -> 1 elements (first 12345); aod with path_length -> 19 elements
      1 tests/data/test.qrt: 3 orig x 7 cir, worst cache/no-cache difference 0 at 
      1 tests/data/test.qrt v4: orig 2 cir 0: center_frequency no-cache 3.75e+09 Hz, cache 3.75e+09 Hz
      1 /sjc/quadriga-lib-fixes/src/qd_channel_qrt.cpp:1562:28: runtime error: null pointer passed as argument 2, which is declared to never be null
      1 /sjc/quadriga-lib-fixes/src/qd_channel_qrt.cpp:1562:28: runtime error: null pointer passed as argument 1, which is declared to never be null
      1 /sjc/quadriga-lib-fixes/src/qd_channel_qrt.cpp:1556:28: runtime error: null pointer passed as argument 2, which is declared to never be null
      1 /sjc/quadriga-lib-fixes/src/qd_channel_qrt.cpp:1556:28: runtime error: null pointer passed as argument 1, which is declared to never be null
      1 /sjc/quadriga-lib-fixes/src/qd_channel_qrt.cpp:1550:28: runtime error: null pointer passed as argument 2, which is declared to never be null
      1 /sjc/quadriga-lib-fixes/src/qd_channel_qrt.cpp:1550:28: runtime error: null pointer passed as argument 1, which is declared to never be null
      1 /sjc/quadriga-lib-fixes/src/qd_channel_qrt.cpp:1544:28: runtime error: null pointer passed as argument 2, which is declared to never be null
      1 /sjc/quadriga-lib-fixes/src/qd_channel_qrt.cpp:1544:28: runtime error: null pointer passed as argument 1, which is declared to never be null
```

**Proposed fix.** Compare as `arma::uword` without the casts. File: `src/qd_channel_qrt.cpp`.

**Test for the fix session.** Catch2: `i_cir = 2^32` throws `std::invalid_argument` with and without a cache.

## B-050: HDF5 write/read round trip changes data (empty snapshots, missing frequency, 1-element vectors, empty strings, long names)

- Status: fixed (WP-02); empty `tx_orientation` / `rx_orientation` still read back as 3x1 zeros, kept and documented by decision (2026-10-16)
- Severity: S3, because the round trip alters sizes and types in edge cases without an error (the numeric values of normal channels survive).
- Kind: wrong result (round trip)
- Mark: [R]
- Where: `src/qd_channel_hdf5.cpp`: the writer skips empty snapshots (1254-1267); padding at 2064-2091 uses `Cube(n_rx, n_tx, 0)` for `delay`; an empty `center_frequency` is not written and the attribute default is NaN (261-264); `qHDF_read_data` treats rank 1 with extent 1 as a scalar (504); an empty string is written with `H5Tset_size(0)` (329); `obj_name[256]` (436) truncates long parameter names.

**What happens.** A 1-snapshot, 0-path channel loses every per-snapshot vector (length 1 becomes 0); the padding of a trailing empty snapshot changes the `delay` shape; an empty `center_frequency` reads back as `[NaN]` (size 0 becomes 1); `Col`/`Row`/`u32_vec` of length 1 come back as scalars (type IDs 41/51/44 become 11/11/14); `""` comes back as a 1-character string `"\0"`; parameter names over 251 characters are listed truncated and dropped by `hdf5_read_channel`. The orientation fields come back as 3×1 zeros when they were empty (shown in the output, arguably harmless).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-011/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 06: HDF5 write -> read round trips of valid channel objects and unstructured data.
#include "quadriga_lib.hpp"
#include <cstdio>
template <typename T>
static double maxdiff(const T &a, const T &b) { return (a.n_elem == b.n_elem && a.n_elem) ? (double)arma::max(arma::abs(arma::vectorise(a) - arma::vectorise(b))) : (a.n_elem == b.n_elem ? 0.0 : 1e99); }
template <typename D>
static void roundtrip(const char *label, const quadriga_lib::channel<D> &c, const char *fn)
{
    std::remove(fn);
    quadriga_lib::hdf5_write(&c, fn);
    auto r = quadriga_lib::hdf5_read_channel<D>(fn);
    std::printf("[%s] is_valid(read)='%s' n_snap %llu->%llu\n", label, r.is_valid().c_str(), (unsigned long long)c.n_snap(), (unsigned long long)r.n_snap());
    auto cmpv = [&](const char *f, size_t a, size_t b) { if (a != b) std::printf("   %s: vector length %zu -> %zu\n", f, a, b); };
    cmpv("coeff_re", c.coeff_re.size(), r.coeff_re.size());
    cmpv("delay", c.delay.size(), r.delay.size());
    cmpv("path_gain", c.path_gain.size(), r.path_gain.size());
    cmpv("path_angles", c.path_angles.size(), r.path_angles.size());
    cmpv("no_interact", c.no_interact.size(), r.no_interact.size());
    cmpv("interact_coord", c.interact_coord.size(), r.interact_coord.size());
    auto cm = [&](const char *f, const arma::Mat<D> &a, const arma::Mat<D> &b) {
        if (a.n_rows != b.n_rows || a.n_cols != b.n_cols) std::printf("   %s: size %llux%llu -> %llux%llu\n", f, (unsigned long long)a.n_rows, (unsigned long long)a.n_cols, (unsigned long long)b.n_rows, (unsigned long long)b.n_cols);
        else if (a.n_elem && maxdiff(a, b) > 1e-3 * (1 + arma::abs(a).max())) std::printf("   %s: values differ by %g\n", f, maxdiff(a, b)); };
    cm("center_frequency", arma::Mat<D>(c.center_frequency), arma::Mat<D>(r.center_frequency));
    cm("tx_pos", c.tx_pos, r.tx_pos);
    cm("rx_pos", c.rx_pos, r.rx_pos);
    cm("tx_orientation", c.tx_orientation, r.tx_orientation);
    cm("rx_orientation", c.rx_orientation, r.rx_orientation);
    for (size_t s = 0; s < std::min(c.coeff_re.size(), r.coeff_re.size()); ++s)
    {
        if (arma::size(c.coeff_re[s]) != arma::size(r.coeff_re[s])) std::printf("   coeff_re[%zu]: size %llux%llux%llu -> %llux%llux%llu\n", s,
            (unsigned long long)c.coeff_re[s].n_rows, (unsigned long long)c.coeff_re[s].n_cols, (unsigned long long)c.coeff_re[s].n_slices,
            (unsigned long long)r.coeff_re[s].n_rows, (unsigned long long)r.coeff_re[s].n_cols, (unsigned long long)r.coeff_re[s].n_slices);
        else if (maxdiff(c.coeff_re[s], r.coeff_re[s]) > 1e-6) std::printf("   coeff_re[%zu] differs\n", s);
        if (arma::size(c.delay[s]) != arma::size(r.delay[s])) std::printf("   delay[%zu]: size differs\n", s);
        else if (maxdiff(c.delay[s], r.delay[s]) > 1e-12) std::printf("   delay[%zu] differs\n", s);
    }
    for (size_t s = 0; s < std::min(c.path_angles.size(), r.path_angles.size()); ++s)
        cm("path_angles[s]", c.path_angles[s], r.path_angles[s]);
    for (size_t s = 0; s < std::min(c.interact_coord.size(), r.interact_coord.size()); ++s)
        cm("interact_coord[s]", c.interact_coord[s], r.interact_coord[s]);
    if (c.name != r.name) std::printf("   name '%s' -> '%s'\n", c.name.c_str(), r.name.c_str());
    if (c.initial_position != r.initial_position) std::printf("   initial_position %d -> %d\n", c.initial_position, r.initial_position);
}
template <typename D>
static quadriga_lib::channel<D> make(arma::uword n_snap, arma::uvec n_path, bool pervalue)
{
    quadriga_lib::channel<D> c;
    c.name = "rt";
    c.center_frequency = pervalue ? arma::Col<D>(n_snap, arma::fill::randu) * 1e9 : arma::Col<D>{(D)3e9};
    c.tx_pos = arma::Mat<D>(3, pervalue ? n_snap : 1, arma::fill::randu);
    c.rx_pos = arma::Mat<D>(3, n_snap, arma::fill::randu);
    c.tx_orientation = arma::Mat<D>(3, pervalue ? n_snap : 1, arma::fill::randu);
    c.initial_position = (int)n_snap - 1;
    for (arma::uword s = 0; s < n_snap; ++s)
    {
        arma::uword np = n_path(s);
        c.coeff_re.push_back(arma::Cube<D>(2, 3, np, arma::fill::randu));
        c.coeff_im.push_back(arma::Cube<D>(2, 3, np, arma::fill::randu));
        c.delay.push_back(arma::Cube<D>(1, 1, np, arma::fill::randu) * 1e-6);
        c.path_gain.push_back(arma::Col<D>(np, arma::fill::randu));
        c.path_angles.push_back(arma::Mat<D>(np, 4, arma::fill::randu));
        arma::u32_vec ni(np, arma::fill::ones);
        c.no_interact.push_back(ni);
        c.interact_coord.push_back(arma::Mat<D>(3, np, arma::fill::randu));
    }
    return c;
}
int main()
{
    const char *fn = "build_linux/scratch/hunt/final/B-011/p06.hdf5";
    roundtrip("double, 3 snaps, paths 4/0/5, per-snap values", make<double>(3, {4, 0, 5}, true), fn);
    roundtrip("float, 3 snaps, paths 0/0/2", make<float>(3, {0, 0, 2}, false), fn);
    roundtrip("double, 1 snap, 0 paths", make<double>(1, {0}, false), fn);
    roundtrip("double, 2 snaps, paths 3/0 (last empty)", make<double>(2, {3, 0}, false), fn);
    {
        auto c = make<double>(1, {2}, false);
        c.center_frequency.reset();
        c.tx_orientation.reset();
        roundtrip("double, 1 snap, no center_frequency, no orientation", c, fn);
    }

    // Unstructured data
    std::remove(fn);
    struct T { std::string n; std::any v; };
    std::vector<T> par = {
        {"vec1", arma::vec{7.0}}, {"vec0", arma::vec()}, {"row1", arma::rowvec{7.0}}, {"mat11", arma::mat(1, 1, arma::fill::ones)},
        {"cube111", arma::cube(1, 1, 1, arma::fill::ones)}, {"str_empty", std::string("")}, {"str", std::string("abc")},
        {"mat0x3", arma::mat(0, 3)}, {"cube230", arma::cube(2, 3, 0)}, {"u32vec1", arma::Col<unsigned>{5u}},
        {std::string(260, 'L'), 1.0}};
    for (auto &p : par)
    {
        try { quadriga_lib::hdf5_write_dset(fn, p.n, &p.v); }
        catch (const std::exception &e) { std::printf("write %.20s: exception %s\n", p.n.c_str(), e.what()); }
    }
    std::vector<std::string> names;
    quadriga_lib::hdf5_read_dset_names(fn, &names);
    std::printf("dset names read back: %zu of %zu\n", names.size(), par.size());
    for (auto &p : par)
    {
        unsigned long long d0[3], d1[3];
        std::any r = quadriga_lib::hdf5_read_dset(fn, p.n);
        int t0 = quadriga_lib::any_type_id(&p.v, d0), t1 = quadriga_lib::any_type_id(&r, d1);
        if (t0 != t1 || (t1 > 0 && (d0[0] != d1[0] || d0[1] != d1[1] || d0[2] != d1[2])))
            std::printf("   par %.20s: type %d [%llu %llu %llu] -> type %d [%llu %llu %llu]\n", p.n.c_str(), t0, d0[0], d0[1], d0[2], t1,
                        t1 > 0 ? d1[0] : 0ULL, t1 > 0 ? d1[1] : 0ULL, t1 > 0 ? d1[2] : 0ULL);
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-011/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-011
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$ASAN "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $H5
ASAN_OPTIONS=halt_on_error=1 "$D/repro_asan" 2>&1 | grep -v -E '^ +#|HDF5-DIAG|major:|minor:' | head -30
```

Output of the phase-2 re-run (`bash run.sh`):
```text
[double, 3 snaps, paths 4/0/5, per-snap values] is_valid(read)='' n_snap 3->3
   rx_orientation: size 0x0 -> 3x1
[float, 3 snaps, paths 0/0/2] is_valid(read)='' n_snap 3->3
   rx_orientation: size 0x0 -> 3x1
[double, 1 snap, 0 paths] is_valid(read)='' n_snap 1->1
   coeff_re: vector length 1 -> 0
   delay: vector length 1 -> 0
   path_gain: vector length 1 -> 0
   path_angles: vector length 1 -> 0
   no_interact: vector length 1 -> 0
   interact_coord: vector length 1 -> 0
   rx_orientation: size 0x0 -> 3x1
[double, 2 snaps, paths 3/0 (last empty)] is_valid(read)='' n_snap 2->2
   rx_orientation: size 0x0 -> 3x1
   delay[1]: size differs
[double, 1 snap, no center_frequency, no orientation] is_valid(read)='' n_snap 1->1
   center_frequency: size 0x1 -> 1x1
   tx_orientation: size 0x0 -> 3x1
   rx_orientation: size 0x0 -> 3x1
dset names read back: 11 of 11
   par vec1: type 41 [1 1 1] -> type 11 [1 1 1]
   par row1: type 51 [1 1 1] -> type 11 [1 1 1]
   par str_empty: type 9 [0 0 0] -> type 9 [1 0 0]
   par u32vec1: type 44 [1 1 1] -> type 14 [1 1 1]
```

**Proposed fix.** Write explicit markers for empty snapshots and restore vectors of the right length; store a missing `center_frequency` as absent and read it back empty; distinguish scalar dataspaces (`H5S_SCALAR`) from 1-element vectors; store an empty string with size 1 and read it back as `""`; reject parameter names over 251 characters in the writer (or query the name length). Document the round-trip rules. File: `src/qd_channel_hdf5.cpp`.

**Test for the fix session.** Catch2: each listed case round-trips with identical sizes and type IDs.

## B-051: HDF5 functions call `H5close()`, switch off error printing process-wide, and leak handles and memory on error paths

- Status: fixed (WP-02)
- Severity: S3, because a program that uses HDF5 itself loses its open handles after any quadriga-lib HDF5 call, and error paths leak; no library-internal data is corrupted.
- Kind: resource / side effects
- Mark: [R]
- Where: `src/qd_channel_hdf5.cpp`: `H5close()` at 870 (`hdf5_create`), 961 (`hdf5_read_layout`), 1519 (`hdf5_write`); `H5Eset_auto2` at 1042 (error printing off, never restored); the throw in `qHDF_close_file` at 59-61; `p_order` allocated at 177 and leaked on the throw paths of `qHDF_get_channel_ID` (177-187).

**What happens.** A caller's own file and dataset IDs are valid before `hdf5_create` and invalid after it (`H5Iis_valid` 0), and the caller's `H5Dwrite` then returns -1. If the caller has the file open, the call throws "Error closing file. File is opened multiple times." and leaves the library's own handle open. Reading a file with a corrupted storage index three times leaks 36 bytes (LeakSanitizer, `qd_channel_hdf5.cpp:177`).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-012/` (create it and save the files below into it if the scratch folder is lost).


`repro_h5close.cpp`:
```cpp
// B probe 07: hdf5_create, hdf5_read_layout and hdf5_write end with H5close(), which shuts the HDF5
// library down process-wide and closes every open HDF5 identifier, including those of the caller.
// hdf5_write also switches the HDF5 error printing off process-wide (H5Eset_auto2(H5E_DEFAULT, NULL, NULL)).
#include "quadriga_lib.hpp"
#include <hdf5.h>
#include <cstdio>
int main()
{
    const char *user_fn = "build_linux/scratch/hunt/final/B-012/p07_user.h5";
    const char *qd_fn = "build_linux/scratch/hunt/final/B-012/p07_qd.hdf5";
    std::remove(qd_fn);
    // The caller has its own HDF5 file open
    hid_t uf = H5Fcreate(user_fn, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t n = 4;
    hid_t sp = H5Screate_simple(1, &n, NULL);
    hid_t ds = H5Dcreate2(uf, "x", H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    std::printf("before: H5Iis_valid(file)=%d H5Iis_valid(dset)=%d\n", (int)H5Iis_valid(uf), (int)H5Iis_valid(ds));

    quadriga_lib::hdf5_create(qd_fn, 4); // another file

    std::printf("after hdf5_create: H5Iis_valid(file)=%d H5Iis_valid(dset)=%d\n", (int)H5Iis_valid(uf), (int)H5Iis_valid(ds));
    double v[4] = {1, 2, 3, 4};
    herr_t e = H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v);
    std::printf("caller's H5Dwrite after hdf5_create returned %d\n", (int)e);
    return 0;
}
```

`repro_leak.cpp`:
```cpp
// B probe 08: throw paths of the HDF5 reader leak memory and identifiers.
// ChannelDims is changed to {3,1,1,1} while Order keeps 4 entries; every reader then throws
// "Storage index in HDF file is corrupted." from qHDF_get_channel_ID after 'new unsigned[n_order]'.
// A second case: the caller has the same file open -> qHDF_close_file throws "File is opened multiple times"
// without closing the library's own file handle.
#include "quadriga_lib.hpp"
#include <hdf5.h>
#include <cstdio>
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char *fn = "build_linux/scratch/hunt/final/B-012/p08.hdf5";
    std::remove(fn);
    quadriga_lib::hdf5_create(fn, 4);
    hid_t f = H5Fopen(fn, H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t d = H5Dopen2(f, "ChannelDims", H5P_DEFAULT);
    unsigned dims[4] = {3, 1, 1, 1};
    H5Dwrite(d, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, dims);
    H5Dclose(d), H5Fclose(f);
    for (int i = 0; i < 3; ++i)
        try { quadriga_lib::hdf5_read_channel<double>(fn, 0); }
        catch (const std::exception &e) { std::printf("case 1, call %d: exception: %s\n", i, e.what()); }

    // case 2
    std::remove(fn);
    quadriga_lib::hdf5_create(fn, 4);
    hid_t uf = H5Fopen(fn, H5F_ACC_RDONLY, H5P_DEFAULT);
    try { quadriga_lib::hdf5_read_layout(fn); }
    catch (const std::exception &e) { std::printf("case 2: exception: %s\n", e.what()); }
    std::printf("case 2: open file ids in process after the call: %lld (caller holds 1)\n", (long long)H5Fget_obj_count(H5F_OBJ_ALL, H5F_OBJ_FILE));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-012/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-012
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$REL "$D/repro_h5close.cpp" -o "$D/repro_h5close" build_linux/libquadriga.a $H5
"$D/repro_h5close" 2>&1 | grep -v -E '^ +#|HDF5-DIAG|major:|minor:' | head -15
$ASAN "$D/repro_leak.cpp" -o "$D/repro_leak_asan" build_asan/libquadriga.a $H5
ASAN_OPTIONS=detect_leaks=1 "$D/repro_leak_asan" 2>&1 | grep -E '^case|Direct leak|#[0-9] .*qd_channel_hdf5|SUMMARY' | head -10
```

Output of the phase-2 re-run (`bash run.sh`):
```text
before: H5Iis_valid(file)=1 H5Iis_valid(dset)=1
after hdf5_create: H5Iis_valid(file)=0 H5Iis_valid(dset)=0
caller's H5Dwrite after hdf5_create returned -1
case 1, call 0: exception: Storage index in HDF file is corrupted.
case 1, call 1: exception: Storage index in HDF file is corrupted.
case 1, call 2: exception: Storage index in HDF file is corrupted.
case 2: exception: Error closing file. File is opened multiple times.
case 2: open file ids in process after the call: 2 (caller holds 1)
Direct leak of 36 byte(s) in 3 object(s) allocated from:
    #1 0x615ab939e3ed in qHDF_get_channel_ID /sjc/quadriga-lib-fixes/src/qd_channel_hdf5.cpp:177
    #2 0x615ab9538cc5 in quadriga_lib::channel<double> quadriga_lib::hdf5_read_channel<double>(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, unsigned int, unsigned int, unsigned int, unsigned int) /sjc/quadriga-lib-fixes/src/qd_channel_hdf5.cpp:1582
SUMMARY: AddressSanitizer: 36 byte(s) leaked in 3 allocation(s).
```

**Proposed fix.** Remove the `H5close()` calls; save and restore the error handler around the library's calls; in `qHDF_close_file` close only the library's own handle and do not throw on an open count above 1; use `std::vector` for `p_order`. File: `src/qd_channel_hdf5.cpp`.

**Test for the fix session.** Catch2: a handle opened by the test with `H5Fcreate` stays valid after `hdf5_create`; reading a corrupted index leaks nothing under LSan.

## B-052: `hdf5_write` "overwrites" a slot by adding a new group; the file grows with every rewrite

- Status: fixed (WP-02)
- Severity: S3, because the documentation says the slot is overwritten, while the old data stays in the file and the file grows linearly (160 kB to 792 kB over 5 rewrites of one slot); the read result is correct.
- Kind: doc / resource
- Mark: [R]
- Where: `src/qd_channel_hdf5.cpp:197-233` (`qHDF_get_channel_ID` allocates a new `channel_N` group on every write; the old group is never removed).

**What happens.** Five writes to slot 0: file size 160 108, 318 132, 476 156, 634 300, 792 324 bytes; the slot maps to `channel_1` ... `channel_5`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-013/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 12: hdf5_write "overwrites" an occupied slot by creating a new group /channel_<N+1> and
// re-pointing the index; the old group stays in the file, so the file grows with every rewrite.
#include "quadriga_lib.hpp"
#include <cstdio>
#include <fstream>
int main()
{
    const char *fn = "build_linux/scratch/hunt/final/B-013/p12.hdf5";
    std::remove(fn);
    quadriga_lib::hdf5_create(fn, 4);
    quadriga_lib::channel<double> c;
    c.center_frequency = {3e9};
    c.tx_pos = arma::mat(3, 1, arma::fill::ones);
    c.rx_pos = arma::mat(3, 1, arma::fill::zeros);
    c.coeff_re.push_back(arma::cube(8, 8, 200, arma::fill::randu));
    c.coeff_im.push_back(arma::cube(8, 8, 200, arma::fill::randu));
    c.delay.push_back(arma::cube(8, 8, 200, arma::fill::randu));
    for (int i = 0; i < 5; ++i)
    {
        int rc = quadriga_lib::hdf5_write(&c, fn, 0);
        std::ifstream f(fn, std::ios::binary | std::ios::ate);
        arma::u32_vec id;
        quadriga_lib::hdf5_read_layout(fn, &id);
        std::printf("write %d to slot 0: rc=%d, file size %lld bytes, slot 0 -> channel_%u\n", i, rc, (long long)f.tellg(), id(0));
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-013/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-013
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$REL "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $H5
"$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
write 0 to slot 0: rc=0, file size 160108 bytes, slot 0 -> channel_1
write 1 to slot 0: rc=1, file size 318132 bytes, slot 0 -> channel_2
write 2 to slot 0: rc=1, file size 476156 bytes, slot 0 -> channel_3
write 3 to slot 0: rc=1, file size 634300 bytes, slot 0 -> channel_4
write 4 to slot 0: rc=1, file size 792324 bytes, slot 0 -> channel_5
```

**Proposed fix.** `H5Ldelete` the old group before reusing the slot (space is still not reclaimed without `h5repack`; document it), or change the documentation to say that the old data stays in the file. File: `src/qd_channel_hdf5.cpp`.

**Test for the fix session.** Catch2: after 5 rewrites of one slot only one `channel_*` group is reachable (or the documented behaviour is pinned).

## B-053: `hdf5_read_dset` returns a variable-length string as 8 bytes of a heap pointer, and leaks it

- Status: fixed (WP-02)
- Severity: S3, because strings written by other tools (variable-length is the default of h5py and MATLAB) are returned as garbage and the buffer leaks.
- Kind: wrong result / leak
- Mark: [R]
- Where: `src/qd_channel_hdf5.cpp:506-515` (`qHDF_read_data`, string branch).

**What happens.** A `par_note` dataset written with a variable-length string type ("written by another HDF5 tool") is read as a string of length 8 with bytes `b0 3b 7e ...` (a pointer); LeakSanitizer reports 29 bytes.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-014/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 20: a variable-length string dataset (the default of h5py and of MATLAB's h5create for strings)
// in the Data group is read by hdf5_read_dset as a std::string holding the bytes of a heap pointer.
#include "quadriga_lib.hpp"
#include <hdf5.h>
#include <cstdio>
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char *fn = "build_linux/scratch/hunt/final/B-014/p20.hdf5";
    std::remove(fn);
    std::any a = 1.0;
    quadriga_lib::hdf5_write_dset(fn, "a", &a); // creates /channel_1/Data
    hid_t f = H5Fopen(fn, H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t g = H5Gopen2(f, "/channel_1/Data", H5P_DEFAULT);
    hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, H5T_VARIABLE);
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t d = H5Dcreate2(g, "par_note", t, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const char *txt = "written by another HDF5 tool";
    H5Dwrite(d, t, H5S_ALL, H5S_ALL, H5P_DEFAULT, &txt);
    H5Dclose(d), H5Sclose(sp), H5Tclose(t), H5Gclose(g), H5Fclose(f);

    std::any r = quadriga_lib::hdf5_read_dset(fn, "note");
    int id = quadriga_lib::any_type_id(&r);
    std::printf("type id %d\n", id);
    if (id == 9)
    {
        std::string s = std::any_cast<std::string>(r);
        std::printf("string of length %zu, bytes:", s.size());
        for (unsigned char ch : s) std::printf(" %02x", ch);
        std::printf("\n(expected: \"%s\")\n", txt);
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-014/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-014
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$ASAN "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $H5
ASAN_OPTIONS=detect_leaks=1 "$D/repro_asan" 2>&1 | grep -v -E '^ +#[1-9]|^$' | head -12
```

Output of the phase-2 re-run (`bash run.sh`):
```text
type id 9
string of length 8, bytes: b0 3b be 30 43 77 00 00
(expected: "written by another HDF5 tool")
=================================================================
==713494==ERROR: LeakSanitizer: detected memory leaks
Direct leak of 29 byte(s) in 1 object(s) allocated from:
    #0 0x7b1333d2b60f in malloc ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:67
SUMMARY: AddressSanitizer: 29 byte(s) leaked in 1 allocation(s).
```

**Cause.** The data is read with the file type; for a variable-length string `H5Dread` stores a `char*`, which is then interpreted as characters and never freed.

**Proposed fix.** If `H5Tis_variable_str`, read into a `char*`, copy the string, then free it with `H5free_memory` / `H5Dvlen_reclaim`. File: `src/qd_channel_hdf5.cpp`.

**Test for the fix session.** Catch2: a VL-string dataset written with the HDF5 C API reads back with the right text and no leak.

## B-054: `quantize_delays` builds dense vectors over the absolute delay range: a 20 ms offset makes it 4000 times slower

- Status: fixed (WP-03)
- Severity: S3, because it is a clear performance defect: the same channel shifted by 20 ms takes seconds instead of a millisecond (absolute delays are a documented option of the channel functions).
- Kind: performance
- Mark: [M]
- Where: `src/qd_channel_quantize_delays.cpp:25-37, 76-80` (dense vectors of length `max_idx + 1`), 545 (`tap2pos`).

**What happens.** 16×16 antennas, `fix_taps = 0`, 36 taps: 0.0009 s without offset and 3.88 s with a 20 ms offset (re-run below, under `flock /tmp/bench.lock`, 1-minute load 1.45 before and 1.50 after; phase 1 measured 0.0009 s and 3.90 s at a load of 2.8).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-015/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 14: quantize_delays cost grows with the absolute delay value, not with the number of paths:
// find_optimal_delays allocates and scans dense PDP vectors of length max_delay / tap_spacing
// for every grid (fix_taps = 0: one grid per antenna pair and snapshot).
// The same channel is quantized twice: delays in [0, 2 us], and the same delays + 20 ms.
#include "quadriga_lib.hpp"
#include <chrono>
#include <cstdio>
int main()
{
    arma::arma_rng::set_seed(1);
    std::vector<arma::cube> cre(1), cim(1), dl0(1);
    cre[0] = arma::cube(16, 16, 20, arma::fill::randn);
    cim[0] = arma::cube(16, 16, 20, arma::fill::randn);
    dl0[0] = arma::cube(1, 1, 20, arma::fill::randu) * 2e-6;
    for (double off : {0.0, 20e-3})
    {
        std::vector<arma::cube> dl(1), qre, qim, qdl;
        dl[0] = dl0[0] + off;
        auto t0 = std::chrono::steady_clock::now();
        quadriga_lib::quantize_delays(&cre, &cim, &dl, &qre, &qim, &qdl, 5e-9, 48, 1.0, 0);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("delay offset %g s: %.4f s, %llu taps\n", off, s, (unsigned long long)qdl[0].n_slices);
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-015/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-015
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$REL "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $H5
echo "load before: $(cat /proc/loadavg)"
flock /tmp/bench.lock "$D/repro"
echo "load after: $(cat /proc/loadavg)"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
load before: 1.45 5.19 8.04 2/2763 713544
delay offset 0 s: 0.0009 s, 36 taps
delay offset 0.02 s: 3.8801 s, 36 taps
load after: 1.50 5.14 8.01 1/2762 713575
```

**Proposed fix.** Sparse accumulation (sort (tap, power) pairs or use a hash map), or subtract the minimum tap index before building the vectors. File: `src/qd_channel_quantize_delays.cpp`.

**Test for the fix session.** Catch2: quantization with a 20 ms offset gives the same taps (after removing the offset) as without; a timing guard is optional.

## B-055: `quantize_delays` gives wrong results when the outputs alias the inputs (undocumented)

- Status: fixed (WP-03)
- Severity: S3, because an in-place call silently returns a wrong first tap; aliasing is neither supported nor documented as excluded.
- Kind: wrong result (aliasing)
- Mark: [R]
- Where: `src/qd_channel_quantize_delays.cpp:495-517` (the outputs are resized and zeroed before the inputs are read).

**What happens.** Separate outputs give coefficient taps `1 0.5 0.5 0.32 0.68`; with the outputs aliasing the inputs the first tap is `0`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-016/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 13: quantize_delays edge cases.
// Usage: p13 <case>
//   negtiny : one delay of -1e-20 s (rounding noise below zero)
//   neg     : one delay of -5e-9 s
//   nan     : one delay NaN
//   big     : one delay of 21.4748 s (= (2^32 - 1) taps of 5 ns)
//   alias   : outputs alias the inputs (same std::vector objects)
//   shape   : per-antenna delays with fix_taps = 1 and 2 (doc says output delays [1,1,n_taps])
#include "quadriga_lib.hpp"
#include <cstdio>
#include <cmath>
int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string cs = argc > 1 ? argv[1] : "negtiny";
    std::vector<arma::cube> cre(1), cim(1), dl(1), qre, qim, qdl;
    cre[0] = arma::cube(1, 1, 3, arma::fill::ones);
    cim[0] = arma::cube(1, 1, 3, arma::fill::zeros);
    dl[0] = arma::cube(1, 1, 3);
    dl[0](0, 0, 0) = 0.0, dl[0](0, 0, 1) = 12.5e-9, dl[0](0, 0, 2) = 33.4e-9;
    if (cs == "negtiny") dl[0](0, 0, 0) = -1e-20;
    if (cs == "neg") dl[0](0, 0, 0) = -5e-9;
    if (cs == "nan") dl[0](0, 0, 0) = NAN;
    if (cs == "big") dl[0](0, 0, 0) = 21.4748;
    if (cs == "shape")
    {
        cre[0] = arma::cube(2, 2, 3, arma::fill::ones), cim[0] = arma::cube(2, 2, 3, arma::fill::zeros);
        dl[0] = arma::cube(2, 2, 3);
        for (int p = 0; p < 3; ++p) dl[0].slice(p).fill(10e-9 * p + 2e-9);
        for (int ft : {1, 2})
        {
            quadriga_lib::quantize_delays(&cre, &cim, &dl, &qre, &qim, &qdl, 5e-9, 48, 1.0, ft);
            std::printf("per-antenna input, fix_taps=%d: delay_quant size %llux%llux%llu\n", ft,
                        (unsigned long long)qdl[0].n_rows, (unsigned long long)qdl[0].n_cols, (unsigned long long)qdl[0].n_slices);
        }
        return 0;
    }
    if (cs == "alias")
    {
        std::vector<arma::cube> r0, i0, d0;
        quadriga_lib::quantize_delays(&cre, &cim, &dl, &r0, &i0, &d0, 5e-9, 48, 1.0, 0);
        quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre, &cim, &dl, 5e-9, 48, 1.0, 0);
        std::printf("separate outputs: coeff_re taps = ");
        for (auto v : r0[0]) std::printf("%g ", v);
        std::printf("\naliased outputs:  coeff_re taps = ");
        for (auto v : cre[0]) std::printf("%g ", v);
        std::printf("\n");
        return 0;
    }
    std::printf("case %s: delays = %g %g %g s\n", cs.c_str(), dl[0](0), dl[0](1), dl[0](2));
    try
    {
        quadriga_lib::quantize_delays(&cre, &cim, &dl, &qre, &qim, &qdl, 5e-9, 48, 1.0, 0);
        std::printf("returned: %llu taps, delays:", (unsigned long long)qdl[0].n_slices);
        for (auto v : qdl[0]) std::printf(" %g", v);
        std::printf("\n");
    }
    catch (const std::exception &e) { std::printf("exception: %s\n", e.what()); }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-016/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-016
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$REL "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $H5
"$D/repro" alias || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
separate outputs: coeff_re taps = 1 0.5 0.5 0.32 0.68 
aliased outputs:  coeff_re taps = 0 0.5 0.5 0.32 0.68 
```

**Proposed fix.** Detect aliasing (`coeff_re_quant == coeff_re` and so on), compute into locals and swap; or document that aliasing is not allowed and throw. File: `src/qd_channel_quantize_delays.cpp`.

**Test for the fix session.** Catch2: the aliased call equals the separate call.

## B-056: `mitsuba_xml_file_write` is quadratic in the number of vertices per object

- Status: fixed (WP-14): one-pass vertex map and per-object face buckets, output byte-identical; 80 000 faces 0.02 s instead of 5.9 s
- Severity: S3, because it is a clear performance defect: 80 000 faces take 5.9 s (and 4× the faces take about 15× the time).
- Kind: performance
- Mark: [M]
- Where: `src/mitsuba_xml_file_write.cpp:~395-410` (`arma::find(vert_idx == i_vert)` once per vertex), `~374` and `~466` (`find(obj_ind_local == i_obj)` once per object).

**What happens.** 5 000 / 20 000 / 80 000 faces: 0.026 / 0.376 / 5.870 s (re-run below, under `flock /tmp/bench.lock`, 1-minute load 1.54 before and 1.57 after; phase 1 gave the same at a load of 2.6). The output also shows the absolute PLY path of B-104.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-017/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 16: mitsuba_xml_file_write builds the per-object PLY vertex remap with one arma::find over all
// face corners per vertex, O(n_vert * n_face) per object. Time for a single-object grid mesh of growing size.
// Also prints the PLY path stored in the XML for an absolute output path.
#include "quadriga_lib.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <filesystem>
int main()
{
    std::string dir = std::filesystem::absolute("build_linux/scratch/hunt/final/B-017").string();
    for (arma::uword n : {50, 100, 200})
    {
        // n x n grid of quads -> 2 n^2 triangles, (n+1)^2 vertices, one object
        arma::mat vert((n + 1) * (n + 1), 3);
        for (arma::uword i = 0; i <= n; ++i)
            for (arma::uword j = 0; j <= n; ++j)
                vert(i * (n + 1) + j, 0) = (double)i, vert(i * (n + 1) + j, 1) = (double)j, vert(i * (n + 1) + j, 2) = 0.0;
        arma::umat face(2 * n * n, 3);
        arma::uword f = 0;
        for (arma::uword i = 0; i < n; ++i)
            for (arma::uword j = 0; j < n; ++j)
            {
                arma::uword a = i * (n + 1) + j, b = a + 1, c = a + n + 1, d = c + 1;
                face(f, 0) = a, face(f, 1) = b, face(f, 2) = d, ++f;
                face(f, 0) = a, face(f, 1) = d, face(f, 2) = c, ++f;
            }
        arma::uvec obj(face.n_rows, arma::fill::zeros), mtl(face.n_rows, arma::fill::ones);
        auto t0 = std::chrono::steady_clock::now();
        quadriga_lib::mitsuba_xml_file_write<double>(dir + "/p16.xml", vert, face, obj, mtl, {"plane"}, {"itu_concrete"});
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%llu faces, %llu vertices: %.3f s\n", (unsigned long long)face.n_rows, (unsigned long long)vert.n_rows, s);
    }
    std::ifstream x(dir + "/p16.xml");
    std::string line;
    while (std::getline(x, line))
        if (line.find("filename") != std::string::npos)
            std::printf("XML: %s\n", line.c_str());
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-017/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-017
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$REL "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $H5
echo "load before: $(cat /proc/loadavg)"
flock /tmp/bench.lock "$D/repro"
echo "load after: $(cat /proc/loadavg)"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
load before: 1.54 5.09 7.98 2/2771 713647
5000 faces, 2601 vertices: 0.026 s
20000 faces, 10201 vertices: 0.376 s
80000 faces, 40401 vertices: 5.870 s
XML: <string name="filename" value="/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/B-017/p16_meshes/plane.ply" />
load after: 1.57 5.03 7.95 1/2779 713700
```

**Proposed fix.** Build a global-to-local index map in one pass and bucket the faces per object once. File: `src/mitsuba_xml_file_write.cpp`.

**Test for the fix session.** Catch2: the PLY output is byte-identical to the current one for a small scene; optionally a timing guard (80 000 faces in under 0.5 s).

## B-057: OBJ reader: CRLF `.mtl` files lose their materials; face indices unchecked when only `face_ind` is requested; no-output call skips validation

- Status: fixed (WP-14): CR stripped from every OBJ / MTL line; own face parser with relative (negative) indices; every index checked for every output combination (also against the vertices defined before the face); the no-output call validates the whole file
- Severity: S3, because invalid or Windows-style input is accepted with wrong results (materials missing, out-of-range or wrapped indices) instead of being read correctly or rejected.
- Kind: wrong result / invalid input accepted
- Mark: [R]
- Where: `src/obj_file_read.cpp:233-235` (`newmtl` compared without trimming the line end), 715-722 (face indices checked only when `mesh` is requested), 652-662 (`%llu` accepts negative relative indices), 556-559 (the call without outputs skips the validation).

**What happens.** (b) An `.mtl` with CRLF line endings: material "red" is found in the OBJ, but `bsdf` is 0×17 (expected 1×17 with Kd 1 0 0). (c) `f -3 -2 -1` with only `face_ind` requested gives `[18446744073709551612 ...]` (2^64-4); with `mesh` it throws. (d) `f 1 2 99` with 3 vertices: accepted as `[0 1 98]` when only `face_ind` is requested. (e) A quad file returns 1 face when no output is requested, while requesting `mesh` throws "not triangulated". (f) belongs to B-104.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-018/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 15: OBJ writer/reader round trips and parser edge cases.
#include "quadriga_lib.hpp"
#include <cstdio>
#include <fstream>
int main()
{
    std::string dir = "build_linux/scratch/hunt/final/B-018/";
    // (a) faces without material (mtl_ind = 0) following a face with material, same object
    {
        arma::mat mesh = {{0, 0, 0, 1, 0, 0, 0, 1, 0}, {5, 0, 0, 6, 0, 0, 5, 1, 0}, {9, 0, 0, 10, 0, 0, 9, 1, 0}};
        arma::uvec obj_ind = {0, 0, 0}, mtl_ind = {1, 0, 2};
        std::vector<std::string> obj_names = {"box"}, mtl_names = {"itu_wood", "itu_glass"};
        quadriga_lib::obj_file_write<double>(dir + "p15a.obj", &mesh, &obj_ind, &mtl_ind, &obj_names, &mtl_names);
        arma::uvec mi;
        quadriga_lib::obj_file_read<double>(dir + "p15a.obj", nullptr, nullptr, nullptr, nullptr, nullptr, &mi);
        std::printf("(a) mtl_ind written {1 0 2}, read back {%llu %llu %llu}\n", (unsigned long long)mi(0), (unsigned long long)mi(1), (unsigned long long)mi(2));
    }
    // (b) .mtl with Windows line endings: BSDF of the matching material
    {
        std::ofstream o(dir + "p15b.obj", std::ios::binary);
        o << "mtllib p15b.mtl\r\no a\r\nv 0 0 0\r\nv 1 0 0\r\nv 0 1 0\r\nusemtl red\r\nf 1 2 3\r\n";
        std::ofstream m(dir + "p15b.mtl", std::ios::binary);
        m << "newmtl red\r\nKd 1 0 0\r\n";
        o.close(), m.close();
        arma::mat bsdf; std::vector<std::string> mn;
        quadriga_lib::obj_file_read<double>(dir + "p15b.obj", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &mn, &bsdf);
        std::printf("(b) CRLF: material '%s', bsdf %llux%llu", mn[0].c_str(), (unsigned long long)bsdf.n_rows, (unsigned long long)bsdf.n_cols);
        if (bsdf.n_rows) std::printf(", Kd = %g %g %g", bsdf(0, 0), bsdf(0, 1), bsdf(0, 2));
        std::printf("  (expected 1x17, Kd = 1 0 0)\n");
    }
    // (c) relative (negative) vertex indices, valid OBJ syntax
    {
        std::ofstream o(dir + "p15c.obj");
        o << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n";
        o.close();
        arma::umat fi; arma::mat mesh;
        try { quadriga_lib::obj_file_read<double>(dir + "p15c.obj", nullptr, nullptr, &fi); std::printf("(c) face_ind only: [%llu %llu %llu]\n", (unsigned long long)fi(0, 0), (unsigned long long)fi(0, 1), (unsigned long long)fi(0, 2)); }
        catch (const std::exception &e) { std::printf("(c) face_ind only: exception %s\n", e.what()); }
        try { quadriga_lib::obj_file_read<double>(dir + "p15c.obj", &mesh); std::printf("(c) mesh: ok\n"); }
        catch (const std::exception &e) { std::printf("(c) mesh: exception %s\n", e.what()); }
    }
    // (d) out-of-range vertex index, face_ind requested without mesh
    {
        std::ofstream o(dir + "p15d.obj");
        o << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 99\n";
        o.close();
        arma::umat fi;
        try { quadriga_lib::obj_file_read<double>(dir + "p15d.obj", nullptr, nullptr, &fi); std::printf("(d) face_ind only: accepted, [%llu %llu %llu] with 3 vertices\n", (unsigned long long)fi(0, 0), (unsigned long long)fi(0, 1), (unsigned long long)fi(0, 2)); }
        catch (const std::exception &e) { std::printf("(d) face_ind only: exception %s\n", e.what()); }
        try { arma::mat mesh; quadriga_lib::obj_file_read<double>(dir + "p15d.obj", &mesh); std::printf("(d) mesh: accepted\n"); }
        catch (const std::exception &e) { std::printf("(d) mesh: exception %s\n", e.what()); }
    }
    // (e) count-only call on a quad file
    {
        std::ofstream o(dir + "p15e.obj");
        o << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
        o.close();
        try { auto n = quadriga_lib::obj_file_read<double>(dir + "p15e.obj"); std::printf("(e) quad file, no outputs: returns %llu\n", (unsigned long long)n); }
        catch (const std::exception &e) { std::printf("(e) no outputs: exception %s\n", e.what()); }
        try { arma::mat mesh; quadriga_lib::obj_file_read<double>(dir + "p15e.obj", &mesh); std::printf("(e) mesh: ok\n"); }
        catch (const std::exception &e) { std::printf("(e) quad file, mesh requested: exception %s\n", e.what()); }
    }
    // (f) material-name matching: "my.material" (no numeric suffix) in default table lookup
    {
        std::ofstream o(dir + "p15f.obj");
        o << "v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl itu_glass.blue\nf 1 2 3\n";
        o.close();
        arma::uvec ci; std::vector<std::string> cn;
        quadriga_lib::obj_file_read<double>(dir + "p15f.obj", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "", &ci, &cn);
        std::printf("(f) 'itu_glass.blue' -> csv_ind %llu (%s)\n", (unsigned long long)ci(0), ci(0) ? cn[ci(0) - 1].c_str() : "none");
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-018/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-018
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$ASAN "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $H5
"$D/repro_asan" 2>&1 | grep -E "^\((b|c|d|e|f)\)|ERROR|SUMMARY" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
(b) CRLF: material 'red', bsdf 0x17  (expected 1x17, Kd = 1 0 0)
(c) face_ind only: [18446744073709551612 18446744073709551613 18446744073709551614]
(c) mesh: exception Error assembling triangle mesh from 'build_linux/scratch/hunt/final/B-018/p15c.obj'.
(d) face_ind only: accepted, [0 1 98] with 3 vertices
(d) mesh: exception Error assembling triangle mesh from 'build_linux/scratch/hunt/final/B-018/p15d.obj'.
(e) quad file, no outputs: returns 1
(e) quad file, mesh requested: exception Mesh in 'build_linux/scratch/hunt/final/B-018/p15e.obj' is not triangulated (quads/n-gons unsupported).
(f) 'itu_glass.blue' -> csv_ind 14 (itu_glass)
```

**Proposed fix.** `trim_ws(line.substr(7))` in the `newmtl` comparison; check `a, b, c <= n_vert` in the face loop for every output; support relative indices or reject them explicitly; run the same validation when no output is requested. File: `src/obj_file_read.cpp`.

**Test for the fix session.** Catch2: a CRLF `.mtl` gives Kd 1 0 0; `f 1 2 99` throws with only `face_ind` requested; `f -3 -2 -1` gives `[0 1 2]` or a clean error; a quad file throws also without outputs.

## B-058: `write_png`: constant, NaN or inverted-range data trigger a NaN-to-integer cast and all pixels get the last colour; an empty matrix writes an invalid PNG

- Status: fixed (WP-14): degenerate ranges and NaN give the first colour, +-Inf clipped, inverted user range and empty matrix throw
- Severity: S3, because edge input gives a wrong image (all maximum colour, also for log input that the docs map to the minimum colour), undefined behaviour, or an undecodable file, without an error.
- Kind: wrong result / UB
- Mark: [R] (UBSan as a what-if: the unchanged `src/write_png.cpp` compiled into the probe with `-fsanitize=undefined,float-cast-overflow`)
- Where: `src/write_png.cpp:78-99` (`scl = (n-1)/(ma-mi)` with `ma == mi`, then `(size_t)NaN` at line 96); an empty matrix reaches the encoder.

**What happens.** 1×1 input, a constant 4×4 matrix, all-NaN input, `min_val > max_val`, and log scale with all values <= 0 all give pixel RGB `128 0 0`, the last colour of "jet" (first colour `0 0 131`); UBSan: "-nan is outside the range of representable values of type 'long unsigned int'" at `write_png.cpp:96`. A 0×0 matrix writes a file that cannot be decoded ("zero width or height is invalid").

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-019/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 17: write_png on constant / 1x1 / all-NaN / min>max / all non-positive (log) input.
// The colour index is computed as (size_t)round((x - mi) * scl) with scl = 255 / (ma - mi); for ma == mi
// this is (size_t)NaN, a float-to-integer conversion with undefined behaviour (the code relies on the
// x86 result 0x8000000000000000 being clipped to the last colour).
// Build A (library):   links build_linux/libquadriga.a, prints the decoded colour of pixel (0,0).
// Build B (what-if):   compiles src/write_png.cpp, src/colormap.cpp, src/lodepng.cpp into the probe with
//                      -fsanitize=float-cast-overflow so UBSan can see the conversion (sources unchanged).
#include "quadriga_lib.hpp"
#include "lodepng.h"
#include <cstdio>
#include <cmath>
static void run(const char *label, const arma::mat &d, double mn = NAN, double mx = NAN, bool lg = false)
{
    std::string fn = "build_linux/scratch/hunt/final/B-019/p17.png";
    try
    {
        quadriga_lib::write_png<double>(d, fn, "jet", mn, mx, lg);
        std::vector<unsigned char> img; unsigned w, h;
        unsigned err = lodepng::decode(img, w, h, fn);
        if (err) { std::printf("%-34s -> no exception from write_png; decoding the file fails: %s\n", label, lodepng_error_text(err)); return; }
        std::printf("%-34s -> %ux%u, pixel(0,0) RGB = %d %d %d\n", label, w, h, img[0], img[1], img[2]);
    }
    catch (const std::exception &e) { std::printf("%-34s -> exception: %s\n", label, e.what()); }
}
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    auto jet = quadriga_lib::colormap("jet", true);
    std::printf("jet first colour %d %d %d, last colour %d %d %d\n", jet(0, 0), jet(0, 1), jet(0, 2), jet(255, 0), jet(255, 1), jet(255, 2));
    run("1x1 value 5", arma::mat(1, 1, arma::fill::value(5.0)));
    run("4x4 constant 0", arma::mat(4, 4, arma::fill::zeros));
    run("4x4 all NaN", arma::mat(4, 4, arma::fill::value(NAN)));
    run("min_val=2 > max_val=1", arma::mat(4, 4, arma::fill::randu), 2.0, 1.0);
    run("log, all values <= 0", arma::mat(4, 4, arma::fill::zeros), NAN, NAN, true);
    run("empty 0x0", arma::mat());
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-019/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-019
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$REL "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $H5
"$D/repro" || echo "exit code $?"
echo "== what-if: the unchanged src/write_png.cpp compiled into the probe with -fsanitize=undefined,float-cast-overflow"
g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=undefined,float-cast-overflow -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" src/write_png.cpp src/colormap.cpp src/lodepng.cpp -o "$D/repro_ub"
"$D/repro_ub" 2>&1 | grep 'runtime error' | sort | uniq -c
```

Output of the phase-2 re-run (`bash run.sh`):
```text
jet first colour 0 0 131, last colour 128 0 0
1x1 value 5                        -> 1x1, pixel(0,0) RGB = 128 0 0
4x4 constant 0                     -> 4x4, pixel(0,0) RGB = 128 0 0
4x4 all NaN                        -> 4x4, pixel(0,0) RGB = 128 0 0
min_val=2 > max_val=1              -> 4x4, pixel(0,0) RGB = 128 0 0
log, all values <= 0               -> 4x4, pixel(0,0) RGB = 128 0 0
empty 0x0                          -> no exception from write_png; decoding the file fails: zero width or height is invalid
== what-if: the unchanged src/write_png.cpp compiled into the probe with -fsanitize=undefined,float-cast-overflow
      1 src/write_png.cpp:96:33: runtime error: -nan is outside the range of representable values of type 'long unsigned int'
```

**Proposed fix.** If `!(ma > mi)` map all values to one documented colour (the first, or the middle); map NaN to colour 0 explicitly; throw on an empty matrix; also close the quote in the error message (B-104). File: `src/write_png.cpp`.

**Test for the fix session.** Catch2: a constant matrix gives the documented colour; all non-positive log input gives the minimum colour; an empty matrix throws.

## B-059: `channel::add_paths` accepts inconsistent counts and leaves an invalid object; `qrt_file_read` accepts any `normalize_M`

- Status: fixed (WP-07)
- Severity: S3, because invalid input is accepted without an error (and in the first case leaves the object in a state that `is_valid` rejects).
- Kind: invalid input accepted
- Mark: [R]
- Where: `src/qd_channel.cpp:445-520` (`add_paths`: when the coefficients add 0 paths, later fields set the count), `src/qd_channel_qrt.cpp:1429` (`normalize_M` not validated).

**What happens.** (b) `add_paths` with 0 coefficient paths and 5 `path_gain` entries is accepted; `is_valid()` then reports "Size mismatch in 'path_gain[0]'". (c) `normalize_M = 5` is accepted and behaves like 0. Line (a) and the column powers belong to B-060.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-020/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 19: channel class methods and QRT normalize_M against their documentation.
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    // (a) calc_effective_path_gain: header comment says "sum up the power of all paths and average over the
    //     transmit and receive antennas"; MD block says "Sums power over all paths and TX/RX antenna pairs"
    quadriga_lib::channel<double> c;
    c.tx_pos = arma::mat(3, 1, arma::fill::zeros);
    c.rx_pos = arma::mat(3, 1, arma::fill::ones);
    c.coeff_re.push_back(arma::cube(2, 2, 1, arma::fill::ones));
    c.coeff_im.push_back(arma::cube(2, 2, 1, arma::fill::zeros));
    c.delay.push_back(arma::cube(1, 1, 1, arma::fill::zeros));
    arma::vec pg = c.calc_effective_path_gain();
    std::printf("(a) 2x2 MIMO, one path, |h|=1 on every antenna pair: calc_effective_path_gain = %g (average would be 1, sum 4)\n", pg(0));

    // (b) add_paths: coefficient cubes with 0 new paths and a path_gain vector with 5 new paths
    quadriga_lib::channel<double> d = c;
    d.path_gain.push_back(arma::vec(1, arma::fill::ones));
    arma::cube re0(2, 2, 0), im0(2, 2, 0), dl0(1, 1, 0);
    arma::vec g5(5, arma::fill::ones);
    try
    {
        d.add_paths(0, &re0, &im0, &dl0, nullptr, nullptr, &g5);
        std::printf("(b) add_paths accepted 0 coefficient paths + 5 path_gain entries; is_valid() now: '%s'\n", d.is_valid().c_str());
    }
    catch (const std::exception &e) { std::printf("(b) add_paths: exception %s\n", e.what()); }

    // (c) qrt_file_read with normalize_M = 5 (documented values: 0, 1)
    arma::cube M0, M1, M5; arma::mat g0, g1, g5m;
    quadriga_lib::qrt_file_read<double>("tests/data/test_v5.qrt", 0, 0, true, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &g0, nullptr, &M0, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
    quadriga_lib::qrt_file_read<double>("tests/data/test_v5.qrt", 0, 0, true, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &g1, nullptr, &M1, nullptr, nullptr, nullptr, nullptr, nullptr, 1);
    try
    {
        quadriga_lib::qrt_file_read<double>("tests/data/test_v5.qrt", 0, 0, true, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &g5m, nullptr, &M5, nullptr, nullptr, nullptr, nullptr, nullptr, 5);
        std::printf("(c) normalize_M = 5 accepted; equal to normalize_M = 0: %d, equal to 1: %d\n", (int)arma::approx_equal(M5, M0, "absdiff", 0.0), (int)arma::approx_equal(M5, M1, "absdiff", 0.0));
    }
    catch (const std::exception &e) { std::printf("(c) normalize_M = 5: exception %s\n", e.what()); }
    // max column power of M with normalize_M = 1 (header comment says "sum-column power is 2")
    for (arma::uword p = 0; p < M1.n_cols; ++p)
    {
        const double *m = M1.slice(0).colptr(p);
        double c1 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2] + m[3] * m[3], c2 = m[4] * m[4] + m[5] * m[5] + m[6] * m[6] + m[7] * m[7];
        std::printf("    normalize_M = 1, path %llu: column powers %.4f %.4f, sum %.4f\n", (unsigned long long)p, c1, c2, c1 + c2);
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-020/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-020
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$ASAN "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $H5
"$D/repro_asan" 2>&1 | head -20 || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
(a) 2x2 MIMO, one path, |h|=1 on every antenna pair: calc_effective_path_gain = 4 (average would be 1, sum 4)
(b) add_paths accepted 0 coefficient paths + 5 path_gain entries; is_valid() now: 'Size mismatch in 'path_gain[0]'.'
(c) normalize_M = 5 accepted; equal to normalize_M = 0: 1, equal to 1: 0
    normalize_M = 1, path 0: column powers 1.0000 0.9586, sum 1.9586
    normalize_M = 1, path 1: column powers 1.0000 1.0000, sum 2.0000
```

**Proposed fix.** In `add_paths` take the path count from the first non-null field even when it is 0 and require every other field to match; in `qrt_file_read` throw if `normalize_M` is not 0 or 1. Files: `src/qd_channel.cpp`, `src/qd_channel_qrt.cpp`.

**Test for the fix session.** Catch2: both calls throw `std::invalid_argument`.

## B-060: Channel and file-I/O header comments contradict the code (index bases, `calc_effective_path_gain` "average", `normalize_M` column power)

- Status: open
- Severity: S3, because these comments would lead a user to a wrong call or a wrong interpretation (index bases off by one, a sum read as an average).
- Kind: doc
- Mark: [R] for the two measured items, [C] for the index-base items
- Where:
  - `include/quadriga_tools.hpp:210-213` (Mitsuba writer: `obj_ind` "1-based", name lengths `max(...)-1`), 223-231 (`obj_file_read`: `mtl_ind`/`csv_ind` "0-based", sizes `max(...)-1`, `csv_strict` "defaults to first materia"), 238-239 (`obj_file_write`: `obj_ind` "1-based"); all contradict the MD blocks in the sources.
  - `include/quadriga_channel.hpp:76`: `calc_effective_path_gain` "average"; the code sums.
  - `normalize_M = 1` documented as "sum-column power is 2" (header comment of `qrt_file_read`); the code normalises the maximum column power to 1.

**What happens.** (a) A 2×2 MIMO channel with one path and |h| = 1 on every antenna pair: `calc_effective_path_gain` returns 4 (an average would be 1). With `normalize_M = 1` the column powers of path 0 are 1.0000 and 0.9586 (sum 1.9586), not a sum of 2.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-020/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 19: channel class methods and QRT normalize_M against their documentation.
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    // (a) calc_effective_path_gain: header comment says "sum up the power of all paths and average over the
    //     transmit and receive antennas"; MD block says "Sums power over all paths and TX/RX antenna pairs"
    quadriga_lib::channel<double> c;
    c.tx_pos = arma::mat(3, 1, arma::fill::zeros);
    c.rx_pos = arma::mat(3, 1, arma::fill::ones);
    c.coeff_re.push_back(arma::cube(2, 2, 1, arma::fill::ones));
    c.coeff_im.push_back(arma::cube(2, 2, 1, arma::fill::zeros));
    c.delay.push_back(arma::cube(1, 1, 1, arma::fill::zeros));
    arma::vec pg = c.calc_effective_path_gain();
    std::printf("(a) 2x2 MIMO, one path, |h|=1 on every antenna pair: calc_effective_path_gain = %g (average would be 1, sum 4)\n", pg(0));

    // (b) add_paths: coefficient cubes with 0 new paths and a path_gain vector with 5 new paths
    quadriga_lib::channel<double> d = c;
    d.path_gain.push_back(arma::vec(1, arma::fill::ones));
    arma::cube re0(2, 2, 0), im0(2, 2, 0), dl0(1, 1, 0);
    arma::vec g5(5, arma::fill::ones);
    try
    {
        d.add_paths(0, &re0, &im0, &dl0, nullptr, nullptr, &g5);
        std::printf("(b) add_paths accepted 0 coefficient paths + 5 path_gain entries; is_valid() now: '%s'\n", d.is_valid().c_str());
    }
    catch (const std::exception &e) { std::printf("(b) add_paths: exception %s\n", e.what()); }

    // (c) qrt_file_read with normalize_M = 5 (documented values: 0, 1)
    arma::cube M0, M1, M5; arma::mat g0, g1, g5m;
    quadriga_lib::qrt_file_read<double>("tests/data/test_v5.qrt", 0, 0, true, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &g0, nullptr, &M0, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
    quadriga_lib::qrt_file_read<double>("tests/data/test_v5.qrt", 0, 0, true, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &g1, nullptr, &M1, nullptr, nullptr, nullptr, nullptr, nullptr, 1);
    try
    {
        quadriga_lib::qrt_file_read<double>("tests/data/test_v5.qrt", 0, 0, true, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &g5m, nullptr, &M5, nullptr, nullptr, nullptr, nullptr, nullptr, 5);
        std::printf("(c) normalize_M = 5 accepted; equal to normalize_M = 0: %d, equal to 1: %d\n", (int)arma::approx_equal(M5, M0, "absdiff", 0.0), (int)arma::approx_equal(M5, M1, "absdiff", 0.0));
    }
    catch (const std::exception &e) { std::printf("(c) normalize_M = 5: exception %s\n", e.what()); }
    // max column power of M with normalize_M = 1 (header comment says "sum-column power is 2")
    for (arma::uword p = 0; p < M1.n_cols; ++p)
    {
        const double *m = M1.slice(0).colptr(p);
        double c1 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2] + m[3] * m[3], c2 = m[4] * m[4] + m[5] * m[5] + m[6] * m[6] + m[7] * m[7];
        std::printf("    normalize_M = 1, path %llu: column powers %.4f %.4f, sum %.4f\n", (unsigned long long)p, c1, c2, c1 + c2);
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-020/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-020
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$ASAN "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $H5
"$D/repro_asan" 2>&1 | head -20 || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
(a) 2x2 MIMO, one path, |h|=1 on every antenna pair: calc_effective_path_gain = 4 (average would be 1, sum 4)
(b) add_paths accepted 0 coefficient paths + 5 path_gain entries; is_valid() now: 'Size mismatch in 'path_gain[0]'.'
(c) normalize_M = 5 accepted; equal to normalize_M = 0: 1, equal to 1: 0
    normalize_M = 1, path 0: column powers 1.0000 0.9586, sum 1.9586
    normalize_M = 1, path 1: column powers 1.0000 1.0000, sum 2.0000
```

**Proposed fix.** Align the header comments with the MD blocks. For `calc_effective_path_gain` the user decides: fix the comment ("sum") or divide by `n_rx * n_tx`. Files: `include/quadriga_tools.hpp`, `include/quadriga_channel.hpp`.

**Test for the fix session.** None for the comments; for `calc_effective_path_gain` a Catch2 case once the behaviour is decided.

## B-061: `combine_irs_coord`: no shape check on `interact_coord_1/2`, heap over-read

- Status: fixed (WP-13)
- Severity: S3, because invalid input (coordinates with fewer rows or columns than `no_interact` implies) is accepted and read out of bounds instead of rejected.
- Kind: memory
- Mark: [R]
- Where: `src/combine_irs_coord.cpp:111-170` (`memcpy` of `3·n_int` values from `seg_1`/`seg_2`, line 159).

**What happens.** `no_interact_1 = {10}` with `interact_coord_1` of size 2×10: ASan heap-buffer-overflow READ of size 240, "0 bytes after 160-byte region" at `combine_irs_coord.cpp:159`. For 16 elements or fewer (Armadillo in-object storage) it silently reads garbage; a column mismatch gives only Armadillo's generic bounds error.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-10/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-10: combine_irs_coord: no shape check on interact_coord_1 -> heap over-read
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    arma::u32_vec n1 = {10}, n2 = {1}, no;
    arma::mat c1(2, 10, arma::fill::ones), c2(3, 1, arma::fill::ones), out; // c1 has 2 rows instead of 3
    quadriga_lib::combine_irs_coord<double>(0, 0, 0, &n1, &c1, &n2, &c2, &no, &out);
    printf("combine_irs_coord accepted a [2,10] interact_coord_1, output %llux%llu\n", (unsigned long long)out.n_rows, (unsigned long long)out.n_cols);
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-10/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-10
g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_asan/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" 2>&1 | grep -E "ERROR: |SUMMARY|runtime error|^    #[0-4] |accepted" | cut -c1-200 || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
==714251==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6f55f2a04710 at pc 0x7265f5b2901f bp 0x7ffe9cbb0b00 sp 0x7ffe9cbb02a8
    #0 0x7265f5b2901e in memcpy ../../../../src/libsanitizer/sanitizer_common/sanitizer_common_interceptors_memintrinsics.inc:117
    #1 0x5d406e8c992d in memcpy /usr/include/x86_64-linux-gnu/bits/string_fortified.h:29
    #2 0x5d406e8c992d in void quadriga_lib::combine_irs_coord<double>(double, double, double, arma::Col<unsigned int> const*, arma::Mat<double> const*, arma::Col<unsigned int> const*, arma::Mat<double
    #3 0x5d406e8937a8 in main build_linux/scratch/hunt/final/C-10/repro.cpp:8
    #4 0x7265f462a600 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:59
    #0 0x7265f5b2ae8f in posix_memalign ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:139
    #1 0x5d406e8cd80d in double* arma::memory::acquire<double>(unsigned long long) /usr/include/armadillo_bits/memory.hpp:72
    #2 0x5d406e8cd80d in arma::Mat<double>::init_warm(unsigned long long, unsigned long long) /usr/include/armadillo_bits/Mat_meat.hpp:444
    #3 0x5d406e8cd80d in arma::Mat<double>::operator=(arma::subview<double> const&) /usr/include/armadillo_bits/Mat_meat.hpp:2102
    #4 0x5d406e8cd80d in void quadriga_lib::combine_irs_coord<double>(double, double, double, arma::Col<unsigned int> const*, arma::Mat<double> const*, arma::Col<unsigned int> const*, arma::Mat<double
SUMMARY: AddressSanitizer: heap-buffer-overflow /usr/include/x86_64-linux-gnu/bits/string_fortified.h:29 in memcpy
```

**Proposed fix.** Check `n_rows == 3` and `n_cols == sum(no_interact)` for both inputs and throw `std::invalid_argument`. File: `src/combine_irs_coord.cpp`.

**Test for the fix session.** Catch2: the case above throws `std::invalid_argument`.

## B-062: `baseband_freq_response`: `aligned_alloc` with sizes that are not a multiple of 32 (generic build) and unchecked NULL return

- Status: fixed (WP-13): the AVX2 DFT kernel uses unaligned loads and stores, the buffers are ordinary Armadillo vectors (no `aligned_alloc`); a call without outputs returns after validation
- Severity: S3, because (a) is undefined behaviour under C11 before DR 460 and fails on some C libraries, and (b) turns an allocation failure into a segfault; both need unusual conditions on glibc.
- Kind: UB / crash
- Mark: [R]
- Where: `src/baseband_freq_response.cpp:137-139` (`aligned_alloc(32, n_carrier_s * sizeof(float))` and two more); also `src/quadriga_lib.cpp:57` (64 bytes, harmless).

**What happens.** (a) In the generic build (`VEC_SIZE` 1) the size is 4·n_carrier bytes. A what-if build that compiles `src/baseband_freq_response.cpp` with `BUILD_WITH_AVX2=0` and ASan into the probe (sources unchanged) reports `invalid-aligned-alloc-alignment` for 7 carriers. In the AVX2 build the sizes are always multiples of 32. (b) With `ulimit -v` 8 GB, 64×64 antennas and 4e6 carriers (no outputs requested) the AVX2 library segfaults (exit 139) instead of throwing `std::bad_alloc`. The conf-file review (`md/REVIEW_CONF_FILE.md`, section 0) saw report (a) in the gcc-15 suite; neither phase-0 suite run reported it.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-11/` (create it and save the files below into it if the scratch folder is lost).


`repro_align.cpp`:
```cpp
// C-11: baseband_freq_response: aligned_alloc size not a multiple of 32 in a generic build (BUILD_WITH_AVX2=0)
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    arma::cube cr(2, 2, 1, arma::fill::ones), ci(2, 2, 1, arma::fill::zeros), dl(1, 1, 1, arma::fill::zeros);
    arma::vec pg = arma::linspace(0.0, 1.0, 7); // 7 carriers -> aligned_alloc(32, 28)
    arma::cube hr;
    quadriga_lib::baseband_freq_response<double>(&cr, &ci, &dl, &pg, 100e6, &hr, nullptr);
    printf("returned, hr(0,0,0) = %g\n", hr(0, 0, 0));
    return 0;
}
```

`repro_null.cpp`:
```cpp
// C-11: baseband_freq_response: return value of aligned_alloc is not checked (segfault instead of std::bad_alloc)
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    arma::cube cr(64, 64, 1, arma::fill::ones), ci(64, 64, 1, arma::fill::zeros), dl(1, 1, 1, arma::fill::zeros);
    arma::vec pg = arma::linspace(0.0, 1.0, 4000000); // internal buffers: 64*64*4e6*4 B = 65.5 GB each
    try
    {
        quadriga_lib::baseband_freq_response<double>(&cr, &ci, &dl, &pg, 100e6, nullptr, nullptr); // no output requested
        printf("returned\n");
    }
    catch (const std::exception &e) { printf("exception: %s\n", e.what()); }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-11/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-11
echo "== (a) what-if: src/baseband_freq_response.cpp compiled with BUILD_WITH_AVX2=0 + ASan (the generic configuration)"
g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -DBUILD_WITH_AVX2=0 -DBUILD_WITH_CUDA=0 -I include -I src "$D/repro_align.cpp" src/baseband_freq_response.cpp -o "$D/repro_align" build_generic/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro_align" 2>&1 | grep -E "ERROR: |SUMMARY|^    #[0-1] |returned" | cut -c1-200
echo "== (b) AVX2 library, allocation fails (ulimit -v 8 GB)"
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro_null.cpp" -o "$D/repro_null" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
(ulimit -c 0; ulimit -v 8000000; OMP_NUM_THREADS=1 "$D/repro_null" || echo "exit code $?") 2>&1 | grep -v "^/bin/bash\|^bash: line"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== (a) what-if: src/baseband_freq_response.cpp compiled with BUILD_WITH_AVX2=0 + ASan (the generic configuration)
==714365==ERROR: AddressSanitizer: invalid alignment requested in aligned_alloc: 32, alignment must be a power of two and the requested size 0x1c must be a multiple of alignment (thread T0)
    #0 0x7bd58ed2ac9f in aligned_alloc ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:107
    #1 0x5ae0b9fe1433 in void quadriga_lib::baseband_freq_response<double>(arma::Cube<double> const*, arma::Cube<double> const*, arma::Cube<double> const*, arma::Col<double> const*, double, arma::Cube
SUMMARY: AddressSanitizer: invalid-aligned-alloc-alignment src/baseband_freq_response.cpp:137 in void quadriga_lib::baseband_freq_response<double>(arma::Cube<double> const*, arma::Cube<double> const*,
== (b) AVX2 library, allocation fails (ulimit -v 8 GB)
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/C-11//run.sh: line 9: 714407 Segmentation fault         (core dumped) OMP_NUM_THREADS=1 "$D/repro_null"
exit code 139
```

**Proposed fix.** Replace the three buffers with `std::vector<float>` or an Armadillo vector (or round the size up to a multiple of 32 and check for NULL); return early when no outputs are requested. File: `src/baseband_freq_response.cpp`.

**Test for the fix session.** The ASan suite with the generic build covers (a) once the baseband cases use a carrier count that is not a multiple of 8; (b) is not practical as a unit test.

## B-063: Test defect: `test_qd_ieee_chan.cpp:318` reads past the delay cube (hidden in the full ASan suite run)

- Status: fixed (WP-01)
- Severity: S3, because it is an out-of-bounds read in the test, not in the library; no check reads the extra column, so the case still passes.
- Kind: memory (test code)
- Mark: [R]
- Where: `tests/catch2_tests/test_qd_ieee_chan.cpp:318`: `chan[1]` has 12 paths (the test itself checks 12 at line 248), but line 318 maps `delay[0]` as 16×13. Corrected in WP-01: `d` is declared at line 301 as a strict alias (`false, true`) of `chan[0].delay[0]`, so line 318 copy-assigns 16×13 values of `chan[1]` into `chan[0]`'s delay memory (the over-read), and `d *= 1e9` scaled `chan[0].delay` in place.

**What happens.** `build_asan/test_bin "IEEE Chan*"` reports a heap-buffer-overflow, "READ of size 1664 ... 0 bytes after 1536-byte region", at test line 318, and the cases still pass. The full-suite sanitizer run of phase 0 did not show it: with ASan's default `suppress_equal_pcs=1`, a report at the same program counter as an earlier one is dropped, and the HDF5 over-read (B-015) is caught in the same ASan `memcpy` interceptor and runs first. `run.sh` shows the case alone, the HDF5 case followed by the IEEE case (only the HDF5 report), and the same pair with `suppress_equal_pcs=0` (both).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-12/` (create it and save the files below into it if the scratch folder is lost).


`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-12/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# C-12: test_qd_ieee_chan.cpp:318 maps chan[1].delay[0] as 16 x 13 although user 1 has 12 paths -> heap over-read
cd /sjc/quadriga-lib-fixes
O="ASAN_OPTIONS=halt_on_error=0:detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0 OMP_NUM_THREADS=4"
echo "== (1) the case alone, suite sanitizer options"
flock /tmp/quadriga-fixes-suite.lock env $O build_asan/test_bin "IEEE Chan - B + 3user + Floors and Distances" 2>&1 | grep -E "ERROR: |READ of|test_qd_ieee_chan.cpp|located|passed" | cut -c1-200
echo "== (2) HDF case before it: the second report is suppressed (same PC: memcpy interceptor, suppress_equal_pcs=1)"
flock /tmp/quadriga-fixes-suite.lock env $O build_asan/test_bin "HDF - Minimal Test","IEEE Chan - B + 3user + Floors and Distances" 2>&1 | grep -E "ERROR: |test_qd_ieee_chan.cpp:318|test_hdf_functions.cpp:114|passed" | cut -c1-200
echo "== (3) same with suppress_equal_pcs=0"
flock /tmp/quadriga-fixes-suite.lock env ASAN_OPTIONS=halt_on_error=0:detect_leaks=1:suppress_equal_pcs=0 OMP_NUM_THREADS=4 build_asan/test_bin "HDF - Minimal Test","IEEE Chan - B + 3user + Floors and Distances" 2>&1 | grep -E "ERROR: |test_qd_ieee_chan.cpp:318|test_hdf_functions.cpp:114|passed" | sort -u | cut -c1-200
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== (1) the case alone, suite sanitizer options
==714426==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x765e75be3e80 at pc 0x78ae78d2901f bp 0x7ffd08a54df0 sp 0x7ffd08a54598
READ of size 1664 at 0x765e75be3e80 thread T0
    #6 0x599bac450c81 in CATCH2_INTERNAL_TEST_4 /sjc/quadriga-lib-fixes/tests/catch2_tests/test_qd_ieee_chan.cpp:318
0x765e75be3e80 is located 0 bytes after 1536-byte region [0x765e75be3880,0x765e75be3e80)
allocated by thread T0 here:
    #15 0x599bac43dc86 in CATCH2_INTERNAL_TEST_4 /sjc/quadriga-lib-fixes/tests/catch2_tests/test_qd_ieee_chan.cpp:234
All tests passed (60 assertions in 1 test case)
== (2) HDF case before it: the second report is suppressed (same PC: memcpy interceptor, suppress_equal_pcs=1)
==714435==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x73d38220f392 at pc 0x77738512901f bp 0x7fffb907cc10 sp 0x7fffb907c3b8
    #8 0x5dbc58a40f49 in CATCH2_INTERNAL_TEST_0 /sjc/quadriga-lib-fixes/tests/catch2_tests/test_hdf_functions.cpp:114
All tests passed (313 assertions in 2 test cases)
== (3) same with suppress_equal_pcs=0
==714456==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x73bffb20f392 at pc 0x775ffe32901f bp 0x7ffc9250bfd0 sp 0x7ffc9250b778
==714456==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x750ffb212480 at pc 0x775ffe32901f bp 0x7ffc92517ff0 sp 0x7ffc92517798
    #6 0x5648c18f4c81 in CATCH2_INTERNAL_TEST_4 /sjc/quadriga-lib-fixes/tests/catch2_tests/test_qd_ieee_chan.cpp:318
    #8 0x5648c14d8f49 in CATCH2_INTERNAL_TEST_0 /sjc/quadriga-lib-fixes/tests/catch2_tests/test_hdf_functions.cpp:114
All tests passed (313 assertions in 2 test cases)
```

**Proposed fix.** Change 13 to 12 at `tests/catch2_tests/test_qd_ieee_chan.cpp:318`. Run sanitizer suites with `suppress_equal_pcs=0`. (WP-01: changing only the 13 makes Armadillo throw, because the strict alias cannot be resized; lines 301 and 318 now build copies.)

**Test for the fix session.** The ASan suite (with `suppress_equal_pcs=0`) is clean for "IEEE Chan*".

## B-064: IEEE indoor model: `XPR_NLOS_linear = 0` gives NaN coefficients

- Status: fixed (WP-11)
- Severity: S3, because an edge input (pure cross-polarisation) is accepted and gives NaN instead of an error or a valid channel.
- Kind: wrong result
- Mark: [R]
- Where: `src/ieee_channel_model_functions.hpp:531` (`get_Jones_matrix` returns with M = 0) and `src/get_channels_ieee_indoor.cpp:531` (`sqrt(x / 0)`).

**What happens.** `XPR_NLOS_linear = 0`: NaN in 12 of 13 `coeff_re` entries, total path gain NaN. The documentation restores the default only for NaN or negative values.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-13/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-13: get_channels_ieee_indoor: XPR_NLOS_linear = 0 is accepted and gives NaN coefficients
#include "quadriga_lib.hpp"
#include <cstdio>
#include <cmath>
int main()
{
    auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);
    auto ch = quadriga_lib::get_channels_ieee_indoor(omni, omni, "B", 5.25e9, 10e-9, 1, 0.0, 1e-3, 0.0, 1.2, {3.0}, {0}, false, {},
                                                     20, 50.0, 7, NAN, 0.0);
    arma::uword n_nan = 0;
    for (double v : ch[0].coeff_re[0]) n_nan += std::isnan(v) ? 1 : 0;
    printf("XPR_NLOS_linear = 0: NaN in coeff_re = %llu of %llu, sum(path_gain) = %g\n", (unsigned long long)n_nan,
           (unsigned long long)ch[0].coeff_re[0].n_elem, arma::accu(ch[0].path_gain[0]));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-13/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-13
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
XPR_NLOS_linear = 0: NaN in coeff_re = 12 of 13, sum(path_gain) = -nan
```

**Proposed fix.** Reject `XPR <= 0` with `std::invalid_argument`, or handle 0 explicitly. File: `src/get_channels_ieee_indoor.cpp`.

**Test for the fix session.** Catch2: `XPR_NLOS_linear = 0` either throws or gives finite coefficients (as decided).

## B-065: Unchecked float-to-integer casts (UB) for non-finite `observation_time` and tiny `quantize`

- Status: fixed (WP-11, WP-13): observation_time and update_rate validated in WP-11; quantize (0 or >= 1e-3 deg) and granularity (finite, finite delays, bounded bin index) in WP-13
- Severity: S3, because invalid input reaches undefined behaviour (the results are returned without an error or fail with an unrelated message).
- Kind: UB
- Mark: [R] (what-if builds with `-fsanitize=float-cast-overflow`, sources unchanged), [C] for `calc_delay_spread`
- Where: `src/get_channels_ieee_indoor.cpp:142` (`observation_time` Inf/NaN), `src/qd_tools_calc_angular_spread.cpp:42-43` (`quantize` of 1e-8°), and by reading `src/qd_tools_calc_delay_spread.cpp:95/117` (NaN/Inf delays with `granularity > 0`).

**What happens.** `observation_time = Inf`: "inf is outside the range of representable values of type 'unsigned long long'", and 1 snapshot is returned; NaN: `arma::memory::acquire(): requested size is too large`. `quantize = 1e-8`: "-1.8e+10 is outside the range of representable values of type 'int'"; a result is still returned.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-14/` (create it and save the files below into it if the scratch folder is lost).


`repro_as.cpp`:
```cpp
// C-14: calc_angular_spreads_sphere: a tiny quantize step overflows an int cast (UB)
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    std::vector<arma::vec> az = {{0.1, 0.2}}, el = {{0.0, 0.05}}, pw = {{1.0, 1.0}};
    arma::vec as;
    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, pw, &as, nullptr, nullptr, nullptr, nullptr, false, true, 1e-8);
    printf("azimuth spread with quantize = 1e-8 deg: %g (no error)\n", as(0));
    return 0;
}
```

`repro_ieee.cpp`:
```cpp
// C-14: get_channels_ieee_indoor: observation_time = Inf / NaN is cast to an integer (UB)
#include "quadriga_lib.hpp"
#include <cstdio>
#include <cmath>
int main()
{
    auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);
    for (double ot : {INFINITY, NAN})
        try
        {
            auto ch = quadriga_lib::get_channels_ieee_indoor(omni, omni, "A", 5.25e9, 10e-9, 1, ot, 1e-3);
            printf("observation_time = %g: returned %zu snapshot(s)\n", ot, ch[0].coeff_re.size());
        }
        catch (const std::exception &e) { printf("observation_time = %g: exception: %s\n", ot, e.what()); }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-14/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-14
# what-if: the affected source file is compiled into the probe with -fsanitize=float-cast-overflow
# (not part of -fsanitize=undefined in GCC, so the build_asan library does not detect it)
F="-std=c++17 -O1 -g -fopenmp -fsanitize=float-cast-overflow -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -DBUILD_WITH_AVX2=1 -DBUILD_WITH_CUDA=0 -I include -I src"
g++-12 $F "$D/repro_ieee.cpp" src/get_channels_ieee_indoor.cpp -o "$D/repro_ieee" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro_ieee" 2>&1 | cut -c1-200 || echo "exit code $?"
g++-12 $F "$D/repro_as.cpp" src/qd_tools_calc_angular_spread.cpp -o "$D/repro_as" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro_as" 2>&1 | cut -c1-200 || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
src/get_channels_ieee_indoor.cpp:142:53: runtime error: inf is outside the range of representable values of type 'long long unsigned int'
observation_time = inf: returned 1 snapshot(s)
observation_time = nan: exception: arma::memory::acquire(): requested size is too large
src/qd_tools_calc_angular_spread.cpp:42:33: runtime error: -1.8e+10 is outside the range of representable values of type 'int'
src/qd_tools_calc_angular_spread.cpp:43:32: runtime error: 1.8e+10 is outside the range of representable values of type 'int'
azimuth spread with quantize = 1e-8 deg: 0.374807 (no error)
```

**Proposed fix.** Validate with `std::isfinite` and give `quantize`/`granularity` a sensible lower bound; throw `std::invalid_argument`. Files: `src/get_channels_ieee_indoor.cpp`, `src/qd_tools_calc_angular_spread.cpp`, `src/qd_tools_calc_delay_spread.cpp`.

**Test for the fix session.** Catch2: the three inputs throw `std::invalid_argument`.

## B-066: `calc_delay_spread` with `granularity`: memory n_cir × global delay span / granularity (800 MB for 100 paths)

- Status: fixed (WP-13): sparse binning per CIR, results bit-identical; C-15 needs 4 MB instead of 808 MB
- Severity: S3, because a clear performance (memory) defect: 50 CIRs with 2 paths each and a 1 ms delay span at 1 ns granularity need 0.8 GB.
- Kind: performance (memory)
- Mark: [M] (maximum RSS; not a timing)
- Where: `src/qd_tools_calc_delay_spread.cpp:95-123` (one dense bin axis over the global min-max delay range, copied per CIR, plus a dense PDP per CIR).

**What happens.** `/usr/bin/time` of the repro: maximum RSS 808 MB.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-15/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-15: calc_delay_spread with granularity allocates n_cir x (global delay span / granularity) bins
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    std::vector<arma::vec> d(50), p(50); // 50 CIRs with 2 paths each, delay span 1 ms, granularity 1 ns
    for (int i = 0; i < 50; ++i) d[i] = {0.0, 1e-3}, p[i] = {1.0, 0.1};
    arma::vec ds = quadriga_lib::calc_delay_spread<double>(d, p, 100.0, 1e-9);
    printf("calc_delay_spread: 50 CIRs x 2 paths, granularity 1 ns: ds[0] = %g s\n", ds(0));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-15/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-15
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
cat /proc/loadavg
OMP_NUM_THREADS=1 /usr/bin/time -f "max RSS = %M kB, wall = %e s" "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
1.61 4.15 7.36 2/2770 714710
calc_delay_spread: 50 CIRs x 2 paths, granularity 1 ns: ds[0] = 0.00028748 s
max RSS = 807456 kB, wall = 0.73 s
```

**Proposed fix.** Bin each CIR sparsely (sort its delays and accumulate by `round(τ / g)` keys). File: `src/qd_tools_calc_delay_spread.cpp`.

**Test for the fix session.** Catch2: results with granularity equal those of a reference implementation for random delays; optionally a memory-light case with a 1 s span.

## B-067: `acdf`: all-NaN data sets bias the averaged CDF; unsorted user bins are accepted

- Status: fixed (WP-13): a set without finite values gets a NaN column and is left out of the averages
- Severity: S3, because an empty (all-NaN) data set distorts the average silently, and invalid bins give a wrong CDF without an error.
- Kind: wrong result / invalid input accepted
- Mark: [R]
- Where: `src/qd_tools_acdf.cpp:209-223` (an empty CDF falls through to the top bin), `:75-79` (no ordering check of user bins), binary search at `:121-131`.

**What happens.** Median of 0..99 alone: 49.005; together with an all-NaN second set: 74.2275. Bins {50, 0, 99, 25} give CDF `0.51 0.51 0.51 1` without an error.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-16/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-16: acdf: an all-NaN data set biases the averaged quantiles; unsorted user bins are accepted
#include "quadriga_lib.hpp"
#include <cstdio>
#include <cmath>
int main()
{
    arma::mat data(100, 2);
    for (int i = 0; i < 100; ++i) data(i, 0) = i, data(i, 1) = NAN; // second set has no finite value
    arma::vec bins, mu, mu1;
    quadriga_lib::acdf<double>(data, &bins, nullptr, nullptr, &mu);
    arma::mat d1 = data.col(0);
    arma::vec b1 = bins;
    quadriga_lib::acdf<double>(d1, &b1, nullptr, nullptr, &mu1);
    printf("0.5-quantile of 0..99 alone = %g, with an all-NaN second set = %g\n", mu1(4), mu(4));
    arma::vec ub = {50.0, 0.0, 99.0, 25.0}; // unsorted bin centres
    arma::mat cdf;
    quadriga_lib::acdf<double>(d1, &ub, &cdf);
    printf("unsorted bins {50,0,99,25}: cdf = %g %g %g %g (no error)\n", cdf(0), cdf(1), cdf(2), cdf(3));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-16/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-16
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
0.5-quantile of 0..99 alone = 49.005, with an all-NaN second set = 74.2275
unsorted bins {50,0,99,25}: cdf = 0.51 0.51 0.51 1 (no error)
```

**Proposed fix.** Skip data sets with no valid value in the averaging (or throw); validate that user bins are strictly ascending. File: `src/qd_tools_acdf.cpp`.

**Test for the fix session.** Catch2: the median with an added all-NaN set equals the median alone; unsorted bins throw `std::invalid_argument`.

## B-068: Docs of `add_fake_los_path` do not match the code: always one extra path; a present LOS is moved to slot 0 and a zero path is left behind

- Status: documented (WP-04)
- Severity: S3, because a user reading "added when no LOS is present" expects n_path outputs when a LOS exists, but always gets n_path + 1 with a zero path in the middle.
- Kind: doc
- Mark: [R]
- Where: documentation in `src/get_channels_spherical.cpp:33, 76` ("prepends a zero-power LOS path when none is present"), `src/get_channels_planar.cpp:27, 72` ("appended ... when no LOS is present"), `src/get_channels_multifreq.cpp:164`; the same wording in the MEX and Python blocks.

**What happens.** Three input paths with the LOS at index 1: output has 4 paths with |c|² = `1.000 0.500 0.000 0.250` (LOS moved to slot 0, a zero path left at its old index); without a LOS: `0.000 0.500 1.000 0.250`. Planar and spherical behave the same; MATLAB gives the same result as C++ (area F).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-17/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-17: add_fake_los_path always adds a slot; a present LOS path is moved to slot 0 (docs say otherwise)
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);
    arma::mat F(3, 3); // NLOS, LOS (index 1, FBS on the TX-RX line), NLOS; TX (0,0,0), RX (50,0,0)
    F.col(0) = arma::vec({20, 20, 0}); F.col(1) = arma::vec({25, 0, 0}); F.col(2) = arma::vec({30, -25, 0});
    arma::vec G = {0.5, 1.0, 0.25}, PL = {0.0, 50.0, 0.0};
    arma::mat M(8, 3, arma::fill::zeros); M.row(0).fill(1.0); M.row(6).fill(-1.0);
    arma::cube cr, ci, dl;
    quadriga_lib::get_channels_spherical<double>(&omni, &omni, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0, &F, &F, &G, &PL, &M,
                                                 &cr, &ci, &dl, 1e9, true, true);
    printf("3 input paths, LOS at index 1, add_fake_los_path = true: n_out = %llu, |c|^2 =", (unsigned long long)cr.n_slices);
    for (arma::uword p = 0; p < cr.n_slices; ++p) printf(" %.3f", cr(0, 0, p) * cr(0, 0, p) + ci(0, 0, p) * ci(0, 0, p));
    printf("\n");
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-17/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-17
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
3 input paths, LOS at index 1, add_fake_los_path = true: n_out = 4, |c|^2 = 1.000 0.500 0.000 0.250
```

**Proposed fix.** Documentation: "always adds one path: slot 0 holds the LOS path (zero power if none was found); if a LOS path was present, its original slot holds a zero-power path". Files: the three source blocks plus the MEX and Python blocks.

**Test for the fix session.** Catch2 pinning the documented layout for both cases.

## B-069: Naming of the cross-polarisation rows of `M` is inconsistent; `calc_cross_polarization_ratio` V/H columns are ambiguous (user decision)

- Status: documented (WP-04); circular XPR columns split off as B-120
- Severity: S3, because the same rows are called "VH" in one place and "HV" in another, so users can build `M` or read the XPR columns the wrong way round.
- Kind: doc
- Mark: [R]
- Where: the `get_channels_*` documentation and `calc_cross_polarization_ratio` call rows 2-3 of `M` "VH" and rows 4-5 "HV"; `xpr_update`, `calc_diffraction_gain`, `qd_path` and the `qd_coeff_combine` comment call rows 2-3 "HV" (H response to a V input). `src/qd_tools_calc_xpr.cpp` (column 1 "V-XPR" = rows 0-1 / rows 4-5). Minor: the doc says `< dTR + window`, the code uses `<=`.

**What happens.** In `get_channels_spherical`, a V-polarised TX and an H-polarised RX give |coeff| = 1 with only row 2 of `M` set and 0 with only row 4 set, so rows 2-3 are the V→H coupling. `calc_cross_polarization_ratio` with only V→H leakage gives `V-XPR = inf, H-XPR = 4`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-18/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-18: M row naming: row 2 couples TX V-pol into RX H-pol in get_channels_*, docs call it "VH" there and "HV" elsewhere;
//       calc_cross_polarization_ratio V-XPR divides by rows 4-5
#include "quadriga_lib.hpp"
#include <cstdio>
#include <cmath>
int main()
{
    auto v = quadriga_lib::generate_arrayant_omni<double>(10.0); // V-pol
    auto h = v; std::swap(h.e_theta_re, h.e_phi_re); std::swap(h.e_theta_im, h.e_phi_im); // H-pol
    arma::mat F = arma::vec({50.0, 20.0, 0.0});
    arma::vec g = {1.0}, L = {0.0};
    for (int k : {2, 4})
    {
        arma::mat M(8, 1, arma::fill::zeros); M(k) = 1.0;
        arma::cube cr, ci, dl;
        quadriga_lib::get_channels_spherical<double>(&v, &h, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, &F, &F, &g, &L, &M, &cr, &ci, &dl, 0.0);
        printf("TX V-pol -> RX H-pol, only M row %d = 1: |coeff| = %g\n", k, std::hypot(cr(0), ci(0)));
    }
    // A path that leaks V-transmitted power into H (row 2), nothing from H into V (row 4)
    std::vector<arma::vec> pw = {{1.0}}, pl = {{200.0}};
    std::vector<arma::mat> Mv = {arma::mat(arma::vec({1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0}))};
    arma::mat tx = arma::vec({0.0, 0.0, 0.0}), rx = arma::vec({100.0, 0.0, 0.0}), xpr;
    quadriga_lib::calc_cross_polarization_ratio<double>(pw, Mv, pl, tx, rx, &xpr);
    printf("calc_cross_polarization_ratio, M rows 0,2,6 = 1,0.5,1: V-XPR = %g, H-XPR = %g\n", xpr(0, 1), xpr(0, 2));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-18/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-18
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
TX V-pol -> RX H-pol, only M row 2 = 1: |coeff| = 1
TX V-pol -> RX H-pol, only M row 4 = 1: |coeff| = 0
calc_cross_polarization_ratio, M rows 0,2,6 = 1,0.5,1: V-XPR = inf, H-XPR = 4
```

**Proposed fix.** The user decides the convention; then unify the documentation (and, if needed, the columns 1/2/4/5 of `calc_cross_polarization_ratio`). Files: documentation blocks of the functions listed, `src/qd_tools_calc_xpr.cpp`.

**Test for the fix session.** Catch2 with V→H leakage only: the documented XPR column is the finite one.

## B-070: `calc_diffraction_gain` (C++) scalar mode expects an electromagnetic-equivalent frequency, the docs say the speed of sound is used

- Status: documented (WP-13): decision changed to (a) on 2026-10-21; the C++ block states that scalar mode takes the EM-equivalent frequency, as the whole C++ core does; wrappers unchanged
- Severity: S3, because a C++ user passing the acoustic frequency gets a wavelength 870 000 times too long and wrong diffraction phases; the MEX and Python wrappers convert internally, so only C++ callers are affected.
- Kind: doc / cross-API
- Mark: [R]
- Where: documentation of `calc_diffraction_gain` in `src/calc_diffraction_gain.cpp` (claims the wavelength follows from the speed of sound) against `:629-659` (uses `c_light / f`); the wrappers pass `f · 3e8 / 343`.

**What happens.** Scalar mode with a detour of half an acoustic wavelength: `center_frequency = 1000` gives phase -0.0000 rad; the EM-equivalent 8.74636e8 gives -3.1416 rad (expected -π).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-19/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-19: calc_diffraction_gain (C++) scalar mode expects an EM-equivalent frequency; the doc does not say so
#include "quadriga_lib.hpp"
#include <cstdio>
#include <cmath>
int main()
{
    arma::mat orig = {{0.0, 0.0, 0.0}}, dest = {{0.17138, 0.0, 0.0}}; // d = half the acoustic wavelength at 1 kHz
    arma::mat mesh = {{100.0, 100.0, 100.0, 101.0, 100.0, 100.0, 100.0, 101.0, 100.0}}; // far-away triangle
    arma::uvec mtl = {0};
    std::unordered_map<std::string, std::vector<double>> prop;
    for (double f : {1000.0, 1000.0 * 299792458.0 / 342.76271})
    {
        arma::vec gain; arma::mat xpr;
        quadriga_lib::calc_diffraction_gain<double>(orig, dest, mesh, mtl, prop, f, 0, &gain, &xpr, nullptr, 0, nullptr, 1, 0,
                                                    true, 0.0, false, true);
        printf("scalar mode, center_frequency = %.6g: phase of xprmat = %.4f rad (expected -pi for 1 kHz sound)\n", f,
               std::atan2(xpr(1, 0), xpr(0, 0)));
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-19/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-19
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
scalar mode, center_frequency = 1000: phase of xprmat = -0.0000 rad (expected -pi for 1 kHz sound)
scalar mode, center_frequency = 8.74636e+08: phase of xprmat = -3.1416 rad (expected -pi for 1 kHz sound)
```

**Proposed fix.** Document the EM-equivalent frequency in the C++ block, or convert inside the C++ function and drop the conversion in the wrappers (an API change for C++). File: `src/calc_diffraction_gain.cpp`.

**Test for the fix session.** Catch2: scalar mode at the documented frequency unit gives phase -π for a half-wavelength detour.

## B-071: `ray_triangle_intersect`: AVX2 and generic kernels swap `fbs_ind`/`sbs_ind` for hits at equal distance (shared edge or vertex)

- Status: fixed (WP-12)
- Severity: S3, because the hit positions agree, but the reported first/second triangle (and so its material or object) differs between builds for rays through shared edges or vertices, which are common in synthetic grids.
- Kind: wrong result (kernel inconsistency)
- Mark: [R]
- Where: `src/ray_triangle_intersect_avx2.cpp:194-212, 245-262` (min-reductions with `_CMP_LT_OQ`, the partner lane wins on ties) against `src/ray_triangle_intersect_generic.cpp:166` (strict `<`, the first index wins).

**What happens.** A unit square of two triangles sharing the diagonal, vertical ray through (0.5, 0.5): generic `fbs_ind = 1, sbs_ind = 2`; AVX2 `fbs_ind = 2, sbs_ind = 1`. Same through the shared vertex, for float and double.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-4/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-4: ray_triangle_intersect: AVX2 and GENERIC report swapped fbs_ind/sbs_ind for hits at equal distance
#include "quadriga_tools.hpp"
#include <cstdio>
int main()
{
    // unit square of two triangles sharing the diagonal (0,0)-(1,1); vertical ray through the diagonal
    arma::mat mesh = {{0, 0, 0, 1, 0, 0, 1, 1, 0}, {0, 0, 0, 1, 1, 0, 0, 1, 0}};
    arma::mat orig = {{0.5, 0.5, 1}}, dest = {{0.5, 0.5, -1}};
    for (int k = 1; k <= 2; ++k)
    {
        arma::u32_vec no, fi, si;
        quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh, nullptr, nullptr, &no, &fi, &si, nullptr, nullptr, k);
        printf("use_kernel=%d (%s): no_interact=%u fbs_ind=%u sbs_ind=%u\n", k, k == 1 ? "GENERIC" : "AVX2", no(0), fi(0), si(0));
    }
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-4/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-4
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
use_kernel=1 (GENERIC): no_interact=2 fbs_ind=1 sbs_ind=2
use_kernel=2 (AVX2): no_interact=2 fbs_ind=2 sbs_ind=1
```

**Proposed fix.** Compare (distance, index) lexicographically in all three reduction steps of the FBS and SBS reductions: keep the own lane if `W < Wp || (W == Wp && I < Ip)`. File: `src/ray_triangle_intersect_avx2.cpp`.

**Test for the fix session.** Catch2: for the diagonal ray, `fbs_ind == 1` and `sbs_ind == 2` for `use_kernel` 1 and 2, float and double.

## B-072: `ray_triangle_intersect`: the AABB slab test rejects rays with dz = 0 in the plane z = Zmin/Zmax (x and y are accepted)

- Status: fixed (WP-12): also for a direction component of -0; the CUDA kernel has the same fold (not changed, noted)
- Severity: S3, because only boundary hits are affected, but the result depends on the axis and contradicts the edge-inclusive triangle test of the same function.
- Kind: wrong result
- Mark: [R]
- Where: `src/ray_triangle_intersect_generic.cpp:88-98` and the same order in `src/ray_triangle_intersect_avx2.cpp:80-97` (`rti_aabb_slab_test`).

**What happens.** A wall triangle in the plane x = 1 with its bottom edge on z = 0, ray from (0,0,0) to (2,0,0) through the edge midpoint: `fbs_ind = 0` with the computed AABB, `1` with an enlarged user AABB, for both kernels. The rotated cases (dy = 0 at y = Ymin, dx = 0 at x = Xmin) are hits.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-5/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-5: ray_triangle_intersect AABB slab test rejects a ray with dz=0 lying in the AABB plane z=Zmin (x/y analogues are hits)
#include "quadriga_tools.hpp"
#include <cstdio>
static void run(const char *tag, arma::mat orig, arma::mat dest, arma::mat mesh)
{
    arma::mat big = {{-5, 5, -5, 5, -5, 5}}; // enlarged user AABB {xmin xmax ymin ymax zmin zmax}
    for (int k = 1; k <= 2; ++k)
    {
        arma::u32_vec f1, f2;
        quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh, nullptr, nullptr, nullptr, &f1, nullptr, nullptr, nullptr, k);
        quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh, nullptr, nullptr, nullptr, &f2, nullptr, nullptr, &big, k);
        printf("%-34s kernel %d: fbs_ind computed AABB = %u, enlarged AABB = %u\n", tag, k, f1(0), f2(0));
    }
}
int main()
{
    // wall in plane x=1; the ray hits the midpoint of its bottom edge (z=0 = Zmin), ray has dz=0
    run("dz=0, ray in plane z=Zmin", {{0, 0, 0}}, {{2, 0, 0}}, {{1, -1, 0, 1, 1, 0, 1, 0, 2}});
    // same geometry rotated: ray has dy=0 and lies in plane y=Ymin
    run("dy=0, ray in plane y=Ymin", {{0, 0, 0}}, {{2, 0, 0}}, {{1, 0, -1, 1, 0, 1, 1, 2, 0}});
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-5/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-5
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
dz=0, ray in plane z=Zmin          kernel 1: fbs_ind computed AABB = 0, enlarged AABB = 1
dz=0, ray in plane z=Zmin          kernel 2: fbs_ind computed AABB = 0, enlarged AABB = 1
dy=0, ray in plane y=Ymin          kernel 1: fbs_ind computed AABB = 1, enlarged AABB = 1
dy=0, ray in plane y=Ymin          kernel 2: fbs_ind computed AABB = 1, enlarged AABB = 1
```

**Cause.** `(Zmin - oz) · (1 / dz)` is 0·inf = NaN; the fold `t0_low = (t0_low >= t2_low) ? t0_low : t2_low` adopts NaN from the third slab and keeps it, so the final test fails. A NaN in the x or y slab is replaced by the later comparisons. Found in WP-12: a direction component of -0 gives the unsorted slab (NaN, -inf) and is rejected as well.

**Proposed fix.** Initialise the interval with the segment [0, 1] and fold each slab with NaN-ignoring max/min (AVX2: `_mm256_max_ps(slab_low, t_low)`, which returns the second operand when the first is NaN; generic: `(slab_low > t_low) ? slab_low : t_low`), then test `t_low <= t_high`. Files: `src/ray_triangle_intersect_generic.cpp`, `src/ray_triangle_intersect_avx2.cpp` (and the CUDA kernel, out of scope here).

**Test for the fix session.** Catch2: the wall case gives `fbs_ind == 1` with and without a user AABB, kernels 1 and 2.

## B-073: `ray_point_intersect` requires sub-clouds aligned to 8 only in AVX2 builds; the default `point_cloud_segmentation` output makes it throw

- Status: fixed (WP-12): the AVX2 kernel masks the lanes past the end of each sub-cloud; `vec_size` padding is optional
- Severity: S3, because the same call works with the generic library and throws with the AVX2 library, and the alignment rule is not documented.
- Kind: cross-build inconsistency / doc
- Mark: [R]
- Where: `src/ray_point_intersect.cpp:165-166`; documentation of `ray_point_intersect` (`sub_cloud_index`) and `point_cloud_segmentation` (default `vec_size = 1`); the AVX2 kernel masks only the global padding (`src/ray_point_intersect_avx2.cpp:401`).

**What happens.** `point_cloud_segmentation` with defaults gives `sub_cloud_index = 0 707 1456 ...`; `ray_point_intersect(..., &sci)` with `use_kernel = 0` throws "Sub-clouds must be aligned with the SIMD vector size (8 for AVX2)." with the AVX2 library and returns 2999 hits with the generic library.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-6/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-6: ray_point_intersect (auto kernel) throws for the default point_cloud_segmentation output in AVX2 builds only
#include "quadriga_tools.hpp"
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    printf("has AVX2: %d\n", (int)quadriga_lib::quadriga_lib_has_AVX2());
    arma::mat center, vert, tridir;
    quadriga_lib::icosphere<double>(2, 1.0, &center, nullptr, &vert, &tridir, true);
    arma::mat trivec = vert;
    for (arma::uword c = 0; c < 9; ++c) trivec.col(c) += center.col(c % 3);
    arma::mat orig(center.n_rows, 3, arma::fill::zeros);
    arma::mat points = arma::randu<arma::mat>(3000, 3) * 20.0 - 10.0;
    arma::mat pR;
    arma::u32_vec sci, cnt;
    quadriga_lib::point_cloud_segmentation(&points, &pR, &sci); // defaults: target_size 1024, vec_size 1
    printf("n_sub = %llu, sub_cloud_index = %u %u %u ...\n", (unsigned long long)sci.n_elem, sci(0), sci(1), sci(2));
    try
    {
        quadriga_lib::ray_point_intersect<double>(pR, orig, trivec, tridir, nullptr, nullptr, &cnt, nullptr, &sci);
        printf("use_kernel=0: OK, total hits %llu\n", (unsigned long long)arma::accu(cnt));
    }
    catch (const std::exception &e)
    {
        printf("use_kernel=0: EXCEPTION %s\n", e.what());
    }
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-6/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-6
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro_gen" build_generic/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
echo "== generic library (build_generic)"
"$D/repro_gen"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
has AVX2: 1
n_sub = 4, sub_cloud_index = 0 707 1456 ...
use_kernel=0: EXCEPTION Sub-clouds must be aligned with the SIMD vector size (8 for AVX2).
== generic library (build_generic)
has AVX2: 0
n_sub = 4, sub_cloud_index = 0 707 1456 ...
use_kernel=0: OK, total hits 2999
```

**Proposed fix.** Recommended: in the AVX2 kernel mask lanes with `i_point + i >= i_point_end` (the mechanism of the padding mask) and drop the requirement. Alternatives: fall back to generic in auto mode when the index is not aligned; or at least document it in both functions. Files: `src/ray_point_intersect_avx2.cpp`, `src/ray_point_intersect.cpp`.

**Test for the fix session.** Catch2: unaligned `sub_cloud_index`, `use_kernel` 1 and 2 give equal hit lists and no exception.

## B-074: Invalid `use_kernel` values are accepted silently; `use_kernel = 2` does not throw in builds without AVX2 (confirms item 1.7 of the Python fix plan)

- Status: fixed (WP-08)
- Severity: S3, because invalid input is accepted without an error; `md/SPEC_CPP_API.md` requires a throw for an unavailable kernel and for out-of-range values, and the documentation says "2 = AVX2 (throws if unavailable)".
- Kind: doc / validation
- Mark: [R]
- Where: `src/fastmath_armadillo.cpp:405-407, 524-526` (`(void)use_kernel` in builds without AVX2) and `:420-430, 536-546` (any value other than 1 or 2 means auto) in `fast_geo2cart` / `fast_cart2geo`; `src/ray_triangle_intersect.cpp:126-134` (other values mean auto; 7 and -1 measured); `src/ray_point_intersect.cpp:136-144` (same pattern, by reading). The MEX and Python wrappers pass the value through. Item 1.7 of `md/python_fix_plan_v00.md` (found by running the Python wrappers; its "(verify)" part, value 3, is confirmed here).

**What happens.** Generic library: `fast_cart2geo` / `fast_geo2cart` with `use_kernel` = 2, 3, -1, 99 all run without an exception. AVX2 library: 3, -1 and 99 run as auto. `ray_triangle_intersect` with `use_kernel` 7 or -1 runs and returns hits. Python (AVX2 module): `cart2geo(..., use_kernel=3)`, 99 and -1 are accepted.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-7/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-7: use_kernel not validated: 2 accepted without AVX2 (plan item 1.7); 3, -1, 99 (fastmath) and 7, -1 (ray_triangle_intersect) accepted
#include "quadriga_math.hpp"
#include "quadriga_tools.hpp"
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    printf("quadriga_lib_has_AVX2() = %d\n", (int)quadriga_lib::quadriga_lib_has_AVX2());
    arma::vec x = {1, 0}, y = {0, 1}, z = {0, 0}, az, el, X, Y, Z;
    for (int k : {2, 3, -1, 99})
    {
        try { quadriga_lib::fast_cart2geo<double>(x, y, z, az, el, nullptr, k); printf("fast_cart2geo use_kernel=%d: no exception\n", k); }
        catch (const std::exception &e) { printf("fast_cart2geo use_kernel=%d: EXCEPTION %s\n", k, e.what()); }
        try { quadriga_lib::fast_geo2cart<double>(az, el, X, Y, Z, nullptr, nullptr, nullptr, nullptr, nullptr, k); printf("fast_geo2cart use_kernel=%d: no exception\n", k); }
        catch (const std::exception &e) { printf("fast_geo2cart use_kernel=%d: EXCEPTION %s\n", k, e.what()); }
    }
    arma::mat o = {{0, 0, 1}}, d = {{0, 0, -1}}, m = {{-1, -1, 0, 1, -1, 0, 0, 1, 0}};
    arma::u32_vec fi;
    for (int k : {7, -1})
    {
        try { quadriga_lib::ray_triangle_intersect<double>(&o, &d, &m, nullptr, nullptr, nullptr, &fi, nullptr, nullptr, nullptr, k);
              printf("ray_triangle_intersect use_kernel=%d: no exception, fbs_ind=%u\n", k, fi(0)); }
        catch (const std::exception &e) { printf("ray_triangle_intersect use_kernel=%d: EXCEPTION %s\n", k, e.what()); }
    }
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-7/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-7
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro_gen" build_generic/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
echo "== generic library (build_generic)"
"$D/repro_gen"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
quadriga_lib_has_AVX2() = 1
fast_cart2geo use_kernel=2: no exception
fast_geo2cart use_kernel=2: no exception
fast_cart2geo use_kernel=3: no exception
fast_geo2cart use_kernel=3: no exception
fast_cart2geo use_kernel=-1: no exception
fast_geo2cart use_kernel=-1: no exception
fast_cart2geo use_kernel=99: no exception
fast_geo2cart use_kernel=99: no exception
ray_triangle_intersect use_kernel=7: no exception, fbs_ind=1
ray_triangle_intersect use_kernel=-1: no exception, fbs_ind=1
== generic library (build_generic)
quadriga_lib_has_AVX2() = 0
fast_cart2geo use_kernel=2: no exception
fast_geo2cart use_kernel=2: no exception
fast_cart2geo use_kernel=3: no exception
fast_geo2cart use_kernel=3: no exception
fast_cart2geo use_kernel=-1: no exception
fast_geo2cart use_kernel=-1: no exception
fast_cart2geo use_kernel=99: no exception
fast_geo2cart use_kernel=99: no exception
ray_triangle_intersect use_kernel=7: no exception, fbs_ind=1
ray_triangle_intersect use_kernel=-1: no exception, fbs_ind=1
```

Folder `build_linux/scratch/hunt/final/G-10/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-10: invalid use_kernel values accepted by Python cart2geo / geo2cart (plan item 1.7; merged with D-7)
import numpy as np, quadriga_lib as q
T = q.tools
print("version", q.version(), "(documented use_kernel: 0, 1, 2)")
for k in (3, 99, -1):
    for name, f in (('cart2geo', T.cart2geo), ('geo2cart', T.geo2cart)):
        try: print(f"{name} use_kernel={k}: accepted, result shape {f(np.ones((3, 2, 2)), use_kernel=k).shape}")
        except Exception as e: print(f"{name} use_kernel={k}: {type(e).__name__}: {e}")
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-10/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-10
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
version 0.12.2_AVX2 (documented use_kernel: 0, 1, 2)
cart2geo use_kernel=3: accepted, result shape (3, 2, 2)
geo2cart use_kernel=3: accepted, result shape (3, 2, 2)
cart2geo use_kernel=99: accepted, result shape (3, 2, 2)
geo2cart use_kernel=99: accepted, result shape (3, 2, 2)
cart2geo use_kernel=-1: accepted, result shape (3, 2, 2)
geo2cart use_kernel=-1: accepted, result shape (3, 2, 2)
```

**Proposed fix.** Throw `std::invalid_argument` for values outside {0, 1, 2} (fastmath) and outside the documented range of the ray kernels; in builds without AVX2 throw for 2, as `sos_val` does. This is a behaviour change: callers that pass 2 unconditionally start to fail; the user decides. Files: `src/fastmath_armadillo.cpp`, `src/ray_triangle_intersect.cpp`, `src/ray_point_intersect.cpp`.

**Test for the fix session.** Catch2: `fast_cart2geo(..., 3)` and -1 throw; `ray_triangle_intersect` with 7 throws; with `quadriga_lib_has_AVX2() == false`, `use_kernel = 2` throws. pytest: `cart2geo(..., use_kernel=3)` raises.

## B-075: AVX2 `fast_atan2` / `fast_cart2geo`: NaN becomes π/4, (inf, inf) gives NaN, tiny or huge double vectors give elevation 0; two doc claims are wrong

- Status: fixed (WP-08)
- Severity: S3, because it needs non-finite or extreme input, but NaN silently turns into a plausible angle in the AVX2 build.
- Kind: wrong result (AVX2 differs from generic) / doc
- Mark: [R]
- Where: `src/fastmath_avx2.h:394-395` (`_mm256_min_ps`/`_mm256_max_ps` drop a NaN), `:401-411` (inf/inf), `:480-487` (`_fm256_cart2geo_ps` squares in float); documentation in `src/fastmath_armadillo.cpp:214-215`.

**What happens.** AVX2: `atan2(1, NaN) = 0.785398185` (reference NaN), `atan2(inf, inf)` and `atan2(inf, -inf)` = NaN (references 0.785, 2.356); `fast_cart2geo<double>` of (1e-23, 1e-23, 1e-23) gives el = 0, len = 0 (generic: 0.6155, 1.73e-23), of (1e20, 1e20, 1e20) el = 0, len = inf. Generic gives the reference values. The documented "atan2(±0, -0) returns ±0 (not ±π)" holds only for AVX2 (generic returns ±π); the documented range "(-π, π]" is wrong (-π is returned for y = -0, x < 0).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-8/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-8: AVX2 fast_atan2 / fast_cart2geo special values differ from GENERIC (NaN x -> pi/4, inf/inf -> NaN, double under/overflow)
#include "quadriga_math.hpp"
#include <cstdio>
#include <cmath>
int main()
{
    arma::vec ys = {1.0, INFINITY, INFINITY, 0.0, -0.0}, xs = {NAN, INFINITY, -INFINITY, -0.0, -1.0};
    arma::fvec a;
    quadriga_lib::fast_atan2(ys, xs, a);
    for (arma::uword i = 0; i < ys.n_elem; ++i)
        printf("fast_atan2(%g, %g) = %-12.9g (ref %.9g)\n", ys(i), xs(i), a(i), std::atan2(ys(i), xs(i)));
    arma::vec x = {NAN, INFINITY, 1e-23, 1e20}, y = {1, INFINITY, 1e-23, 1e20}, z = {1, 0, 1e-23, 1e20}, az, el, len;
    quadriga_lib::fast_cart2geo<double>(x, y, z, az, el, &len); // auto kernel
    for (arma::uword i = 0; i < x.n_elem; ++i)
        printf("fast_cart2geo(%g,%g,%g): az=%-10.7g el=%-10.7g len=%.7g\n", x(i), y(i), z(i), az(i), el(i), len(i));
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-8/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro_gen" build_generic/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
echo "== generic library (build_generic)"
"$D/repro_gen"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
fast_atan2(1, nan) = 0.785398185  (ref nan)
fast_atan2(inf, inf) = -nan         (ref 0.785398163)
fast_atan2(inf, -inf) = -nan         (ref 2.35619449)
fast_atan2(0, -0) = 0            (ref 3.14159265)
fast_atan2(-0, -1) = -3.14159274  (ref -3.14159265)
fast_cart2geo(nan,1,1): az=0.7853982  el=0          len=nan
fast_cart2geo(inf,inf,0): az=-nan       el=0          len=inf
fast_cart2geo(1e-23,1e-23,1e-23): az=0.7853982  el=0          len=0
fast_cart2geo(1e+20,1e+20,1e+20): az=0.7853982  el=0          len=inf
== generic library (build_generic)
fast_atan2(1, nan) = nan          (ref nan)
fast_atan2(inf, inf) = 0.785398185  (ref 0.785398163)
fast_atan2(inf, -inf) = 2.3561945    (ref 2.35619449)
fast_atan2(0, -0) = 3.14159274   (ref 3.14159265)
fast_atan2(-0, -1) = -3.14159274  (ref -3.14159265)
fast_cart2geo(nan,1,1): az=nan        el=0          len=nan
fast_cart2geo(inf,inf,0): az=0.7853982  el=0          len=inf
fast_cart2geo(1e-23,1e-23,1e-23): az=0.7853982  el=0.6154797  len=1.732051e-23
fast_cart2geo(1e+20,1e+20,1e+20): az=0.7853982  el=0.6154797  len=1.732051e+20
```

**Proposed fix.** In `_fm256_atan2256_ps` blend NaN where `_CMP_UNORD_Q` holds and special-case both-infinite input; in the double `cart2geo` path scale by the largest component before narrowing, or compute the length in double; correct the documentation (range [-π, π] and the ±0 behaviour per kernel, or make the generic kernel match). Files: `src/fastmath_avx2.h`, `src/fastmath_vectorized_avx2.cpp`, `src/fastmath_armadillo.cpp`.

**Test for the fix session.** Catch2: `fast_atan2({1, inf}, {NaN, inf})` gives {NaN, π/4} in both builds; `fast_cart2geo` of (1e-23, 1e-23, 1e-23) with `use_kernel = 2` gives el = 0.6155 within 1e-6.

## B-076: AVX2 `fast_slerp` fast path: error 3e-6 (about 50 ULP, documented ~5) and each lane's result depends on its neighbours

- Status: fixed (WP-08)
- Severity: S3, because the error is small but ten times the documented bound and non-local: identical inputs give different outputs depending on the other 7 lanes.
- Kind: wrong result / doc
- Mark: [R]
- Where: `src/fastmath_avx2.h:597-602` (fast path when all 8 lanes have cos Φ > 0.99998). Related, different cause: B-037 (same function).

**What happens.** A = 1, B = exp(i·0.005), w = 0.5, exact |X| = 1: AVX2 with all 8 lanes equal gives |X| - 1 = -3.13e-6; with lane 7 changed to Φ = 1.0, lane 0 gives -3.4e-8; generic gives 2.6e-8 in both cases.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-9/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-9: fast_slerp AVX2 fast path: ~3e-6 error (doc ~5 ULP) and result depends on neighbouring lanes
// A = 1, B = exp(i*phi), w = 0.5; the exact SLERP result is exp(i*phi/2) with magnitude 1.
#include "quadriga_math.hpp"
#include <cstdio>
#include <cmath>

static void run(const char *tag, double phi_other)
{
    const double phi = 0.005; // cos(phi) = 0.9999875 > 0.99998 -> AVX2 fast path when all lanes agree
    arma::vec Ar(8, arma::fill::ones), Ai(8, arma::fill::zeros), Br(8), Bi(8), w(8, arma::fill::value(0.5));
    for (int i = 0; i < 8; ++i)
    {
        double p = (i == 7) ? phi_other : phi;
        Br(i) = std::cos(p), Bi(i) = std::sin(p);
    }
    arma::fvec Xr, Xi;
    quadriga_lib::fast_slerp(Ar, Ai, Br, Bi, w, Xr, Xi);
    double mag = std::hypot((double)Xr(0), (double)Xi(0));
    printf("%-36s lane 0: Xr=%.9f Xi=%.9f |X|=%.9f  (exact |X| = 1, Xr = %.9f, Xi = %.9f), |X|-1 = %.3g\n", tag,
           Xr(0), Xi(0), mag, std::cos(phi / 2), std::sin(phi / 2), mag - 1.0);
}

int main()
{
    run("all 8 lanes phi=0.005", 0.005);
    run("lane 7 phi=1.0, lanes 0-6 phi=0.005", 1.0);
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-9/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-9
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro_gen" build_generic/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
echo "== generic library (build_generic)"
"$D/repro_gen"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
all 8 lanes phi=0.005                lane 0: Xr=0.999993742 Xi=0.002499990 |X|=0.999996867  (exact |X| = 1, Xr = 0.999996875, Xi = 0.002499997), |X|-1 = -3.13e-06
lane 7 phi=1.0, lanes 0-6 phi=0.005  lane 0: Xr=0.999996841 Xi=0.002499997 |X|=0.999999966  (exact |X| = 1, Xr = 0.999996875, Xi = 0.002499997), |X|-1 = -3.4e-08
== generic library (build_generic)
all 8 lanes phi=0.005                lane 0: Xr=0.999996901 Xi=0.002499998 |X|=1.000000026  (exact |X| = 1, Xr = 0.999996875, Xi = 0.002499997), |X|-1 = 2.56e-08
lane 7 phi=1.0, lanes 0-6 phi=0.005  lane 0: Xr=0.999996901 Xi=0.002499998 |X|=1.000000026  (exact |X| = 1, Xr = 0.999996875, Xi = 0.002499997), |X|-1 = 2.56e-08
```

**Cause.** The fast path returns plain linear interpolation of the complex values when all lanes are nearly parallel; its amplitude error is about Φ²/8 (3e-6 at Φ = 0.005), not the 1e-8 claimed in the comment.

**Proposed fix.** Remove the fast path, or raise the threshold so that Φ²/8 < 3e-7 (cos Φ > 0.9999988). File: `src/fastmath_avx2.h`.

**Test for the fix session.** Catch2: ||X| - 1| < 5e-7 with 8 identical lanes, and lane 0 is exactly the same whatever lane 7 holds.

## B-077: `obj_overlap_test` crashes on an empty mesh

- Status: fixed (WP-12)
- Severity: S3, because degenerate but plausible input (an OBJ without faces) is accepted and crashes instead of returning or throwing.
- Kind: crash
- Mark: [R]
- Where: `src/obj_overlap_test.cpp:347` (read of `box[0]` on an empty AABB), 401 (write), 522 (`arma::regspace<arma::uvec>(0, n_obj - 1)` underflows for `n_obj = 0`).

**What happens.** A 0×9 mesh with an empty `obj_ind` passes the checks; UBSan "load of null pointer", ASan SEGV at `obj_overlap_test.cpp:347`. (`point_inside_mesh` with an empty mesh throws cleanly.)

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-10/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-10: obj_overlap_test crashes (null access) on an empty mesh instead of throwing / returning empty
#include "quadriga_tools.hpp"
#include <cstdio>
int main()
{
    arma::mat mesh(0, 9);
    arma::uvec obj;
    try
    {
        arma::uvec r = quadriga_lib::obj_overlap_test(&mesh, &obj);
        printf("obj_overlap_test empty mesh: returned %llu ids\n", (unsigned long long)r.n_elem);
    }
    catch (const std::exception &e)
    {
        printf("obj_overlap_test empty mesh: EXCEPTION %s\n", e.what());
    }
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-10/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-10
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== ASan+UBSan build (build_asan)"
ASAN_OPTIONS=detect_leaks=1 "$D/repro_asan" 2>&1 | grep -vE '^\s*$' | grep -E '^[^ =#]|ERROR: |SUMMARY|runtime error|^    #[0-4] ' | cut -c1-200 || true
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== ASan+UBSan build (build_asan)
/sjc/quadriga-lib-fixes/src/obj_overlap_test.cpp:347:15: runtime error: load of null pointer of type 'double'
AddressSanitizer:DEADLYSIGNAL
==715470==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000 (pc 0x5a5cd562f0ce bp 0x7ffd21457e30 sp 0x7ffd214570c0 T0)
    #0 0x5a5cd562f0ce in arma::Col<unsigned long long> quadriga_lib::obj_overlap_test<double>(arma::Mat<double> const*, arma::Col<unsigned long long> const*, std::vector<std::__cxx11::basic_string<cha
    #1 0x5a5cd5583cf8 in main build_linux/scratch/hunt/final/D-10/repro.cpp:10
    #2 0x73002b82a600 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:59
    #3 0x73002b82a717 in __libc_start_main_impl ../csu/libc-start.c:360
    #4 0x5a5cd5583374 in _start (/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/D-10/repro_asan+0xf2374) (BuildId: 5074bd0771223ad0d41dbc045b2c41bc484fbf93)
rax = 0x0000000000000000  rbx = 0x0000000000000000  rcx = 0x000073002bc1d92b  rdx = 0x0000000000000000  
rdi = 0x00005a5cd563901d  rsi = 0x00007ffd214570b0  rbp = 0x00007ffd21457e30  rsp = 0x00007ffd214570c0  
r12 = 0x0000000000000000  r13 = 0x0000000000000000  r14 = 0x00006f0028cf0be0  r15 = 0x0000000000000000  
AddressSanitizer can not provide additional info.
SUMMARY: AddressSanitizer: SEGV /sjc/quadriga-lib-fixes/src/obj_overlap_test.cpp:347 in arma::Col<unsigned long long> quadriga_lib::obj_overlap_test<double>(arma::Mat<double> const*, arma::Col<unsigne
```

**Proposed fix.** Return an empty result early when the mesh is empty (or throw "Input 'mesh' cannot be empty."; the user decides). File: `src/obj_overlap_test.cpp`.

**Test for the fix session.** Catch2: an empty mesh gives an empty result (or the decided exception).

## B-078: `subdivide_triangles` (and `icosphere`, `cube`, `plane`): unchecked overflow of the face count leads to writes through a null pointer

- Status: fixed (WP-12)
- Severity: S3, because invalid input (a subdivision count around 2^32, or a negative value converted by the MEX helpers, see B-034) is accepted and crashes.
- Kind: UB / memory
- Mark: [R]
- Where: `src/subdivide_triangles.cpp:63` (product `n_in * n_div * n_div`), 66 (`set_size`), writes from 116; the same unchecked product in `src/icosphere.cpp:57` and, through `subdivide_triangles`, in `cube` and `plane`.

**What happens.** `n_div = 4294967296`: the output count wraps to 0, the output is 0×9 and the loop writes anyway: UBSan "store to null pointer", ASan SEGV at `subdivide_triangles.cpp:116`. (Area F measured the same for `icosphere(-1)` from MATLAB: `n_div` becomes 2^64-1, `20·n_div²` wraps to 20, segfault.)

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-11/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-11: subdivide_triangles: unchecked overflow of n_triangles_in*n_div*n_div leads to heap writes (SEGV)
#include "quadriga_tools.hpp"
#include <cstdio>
int main()
{
    arma::mat t = {{0, 0, 0, 1, 0, 0, 0, 1, 0}}, out;
    try
    {
        arma::uword n = quadriga_lib::subdivide_triangles(4294967296ULL, &t, &out); // n_div = 2^32 -> n_div^2 wraps to 0
        printf("returned %llu, out rows %llu\n", (unsigned long long)n, (unsigned long long)out.n_rows);
    }
    catch (const std::exception &e)
    {
        printf("EXCEPTION %s\n", e.what());
    }
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-11/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-11
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== ASan+UBSan build (build_asan)"
ASAN_OPTIONS=detect_leaks=1 "$D/repro_asan" 2>&1 | grep -vE '^\s*$' | grep -E '^[^ =#]|ERROR: |SUMMARY|runtime error|^    #[0-4] ' | cut -c1-200 || true
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== ASan+UBSan build (build_asan)
/sjc/quadriga-lib-fixes/src/subdivide_triangles.cpp:116:38: runtime error: store to null pointer of type 'double'
AddressSanitizer:DEADLYSIGNAL
==715500==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000 (pc 0x5f7aa611e891 bp 0x7fff98985cf0 sp 0x7fff989859b0 T0)
    #0 0x5f7aa611e891 in unsigned long long quadriga_lib::subdivide_triangles<double>(unsigned long long, arma::Mat<double> const*, arma::Mat<double>*, arma::Col<unsigned long long> const*, arma::Col<
    #1 0x5f7aa610f152 in main build_linux/scratch/hunt/final/D-11/repro.cpp:9
    #2 0x7a8420e2a600 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:59
    #3 0x7a8420e2a717 in __libc_start_main_impl ../csu/libc-start.c:360
    #4 0x5f7aa610e004 in _start (/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/D-11/repro_asan+0x1b004) (BuildId: 5d8dffae1092d1a7da93b353a089a9ad4d87a552)
rax = 0x0000000000000000  rbx = 0x0000000000000000  rcx = 0x00007a842121d92b  rdx = 0x0000000000000000  
rdi = 0x00005f7aa611f1a6  rsi = 0x00007fff989859a0  rbp = 0x00007fff98985cf0  rsp = 0x00007fff989859b0  
r12 = 0x0000000000000000  r13 = 0x0000000000000000  r14 = 0x0000000000000000  r15 = 0x0000000000000000  
AddressSanitizer can not provide additional info.
SUMMARY: AddressSanitizer: SEGV /sjc/quadriga-lib-fixes/src/subdivide_triangles.cpp:116 in unsigned long long quadriga_lib::subdivide_triangles<double>(unsigned long long, arma::Mat<double> const*, ar
```

**Proposed fix.** Throw `std::invalid_argument` when `n_div > UINT32_MAX` or when `n_in * n_div * n_div` overflows (check with `__builtin_mul_overflow` or by division); the same in `icosphere` before computing the face count. Files: `src/subdivide_triangles.cpp`, `src/icosphere.cpp`.

**Test for the fix session.** Catch2: `subdivide_triangles(1ULL << 32, ...)` and `icosphere(1ULL << 32)` throw `std::invalid_argument`.

## B-079: An output that aliases an input gives wrong results (`ray_triangle_intersect` with `fbs = &orig`, `subdivide_triangles` in place)

- Status: fixed (WP-12): aliasing is supported, not excluded
- Severity: S3, because advancing rays in place (`fbs = &orig`) is a natural call and silently gives a wrong second hit; `md/SPEC_CPP_API.md` ("Outputs") requires aliasing support or a documented exclusion.
- Kind: wrong result (aliasing) / doc
- Mark: [R]
- Where: `src/ray_triangle_intersect.cpp:224-255` (fbs is written before sbs reads `ox/oy/oz`); `src/subdivide_triangles.cpp:65-66, 82` (`set_size` on the aliased input, then reads of the new buffer).

**What happens.** Ray (0,0,1) to (0,0,-2) through the planes z = 0 and z = -1: separate outputs give sbs = (0, 0, -1); `fbs = &orig` gives sbs = (0, 0, -1.33333). `subdivide_triangles(2, &t, &t)` differs from the out-of-place result by up to 1.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-12/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-12: output aliasing an input gives wrong results: ray_triangle_intersect(fbs=&orig) and subdivide_triangles in place
#include "quadriga_tools.hpp"
#include <cstdio>
int main()
{
    arma::mat mesh = {{-1, -1, 0, 1, -1, 0, 0, 1, 0}, {-1, -1, -1, 1, -1, -1, 0, 1, -1}}; // planes z=0 and z=-1
    arma::mat orig = {{0, 0, 1}}, dest = {{0, 0, -2}}, fbs, sbs;
    quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh, &fbs, &sbs);
    printf("separate outputs : sbs = (%g %g %g)\n", sbs(0, 0), sbs(0, 1), sbs(0, 2));
    arma::mat o2 = orig;
    quadriga_lib::ray_triangle_intersect<double>(&o2, &dest, &mesh, &o2, &sbs); // fbs aliases orig
    printf("fbs aliases orig : sbs = (%g %g %g)   (expected 0 0 -1)\n", sbs(0, 0), sbs(0, 1), sbs(0, 2));

    arma::mat t = {{0, 0, 0, 1, 0, 0, 0, 1, 0}}, ref;
    quadriga_lib::subdivide_triangles(2, &t, &ref);
    arma::mat t2 = t;
    quadriga_lib::subdivide_triangles(2, &t2, &t2); // output aliases input
    printf("subdivide_triangles in place: max |diff to out-of-place| = %g (expected 0)\n", arma::abs(t2 - ref).max());
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-12/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-12
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== ASan+UBSan build (build_asan)"
ASAN_OPTIONS=detect_leaks=1 "$D/repro_asan" 2>&1 | grep -vE '^\s*$' | grep -E '^[^ =#]|ERROR: |SUMMARY|runtime error|^    #[0-4] ' | cut -c1-200 || true
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== ASan+UBSan build (build_asan)
separate outputs : sbs = (0 0 -1)
fbs aliases orig : sbs = (0 0 -1.33333)   (expected 0 0 -1)
subdivide_triangles in place: max |diff to out-of-place| = 1 (expected 0)
```

**Proposed fix.** `ray_triangle_intersect`: compute sbs before fbs, or into a local matrix when an output equals `orig` or `dest`. `subdivide_triangles`: work from a copy when output and input are the same object. Alternatively document "outputs must not alias inputs" and throw when they do. Files: `src/ray_triangle_intersect.cpp`, `src/subdivide_triangles.cpp`.

**Test for the fix session.** Catch2: the aliased calls equal the separate ones (sbs z = -1; identical subdivided mesh).

## B-080: `point_cloud_split` leaks its work buffer on every failed split

- Status: fixed (WP-01)
- Severity: S3, because it leaks n_points bytes per failure, and `point_cloud_segmentation` hits it repeatedly for clusters that cannot be split (e.g. identical points).
- Kind: leak
- Mark: [R]
- Where: `src/point_cloud_split.cpp:93` (`new bool[]`), `:118-119` (`return -axis` without `delete[]`); `src/triangle_mesh_split.cpp:116-119` does it correctly.

**What happens.** 1000 identical points: `point_cloud_split` returns -3 and LeakSanitizer reports a direct leak of 1000 bytes allocated at `point_cloud_split.cpp:93`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-13/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-13: point_cloud_split leaks its work buffer (new bool[n_points]) on every failed split
#include "quadriga_tools.hpp"
#include <cstdio>
int main()
{
    arma::mat pts(1000, 3, arma::fill::ones); // all points identical -> split must fail
    arma::mat A, B;
    int r = quadriga_lib::point_cloud_split(&pts, &A, &B);
    printf("point_cloud_split on 1000 identical points returned %d (failure)\n", r);
    fflush(stdout);
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-13/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-13
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer "$D/repro.cpp" -o "$D/repro_asan" build_asan/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== ASan+UBSan build (build_asan)"
ASAN_OPTIONS=detect_leaks=1 "$D/repro_asan" 2>&1 | grep -vE '^\s*$' | grep -E '^[^ =#]|ERROR: |SUMMARY|runtime error|^    #[0-4] ' | cut -c1-200 || true
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== ASan+UBSan build (build_asan)
point_cloud_split on 1000 identical points returned -3 (failure)
==715591==ERROR: LeakSanitizer: detected memory leaks
Direct leak of 1000 byte(s) in 1 object(s) allocated from:
    #0 0x7af98672c8cf in operator new[](unsigned long) ../../../../src/libsanitizer/asan/asan_new_delete.cpp:111
    #1 0x5c5dda49914e in int quadriga_lib::point_cloud_split<double>(arma::Mat<double> const*, arma::Mat<double>*, arma::Mat<double>*, int, arma::Col<int>*) /sjc/quadriga-lib-fixes/src/point_cloud_spl
    #2 0x5c5dda48a79b in main build_linux/scratch/hunt/final/D-13/repro.cpp:8
    #3 0x7af98522a600 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:59
    #4 0x7af98522a717 in __libc_start_main_impl ../csu/libc-start.c:360
SUMMARY: AddressSanitizer: 1000 byte(s) leaked in 1 allocation(s).
```

**Proposed fix.** `delete[] isB;` before `return -axis;`, or better a `std::vector<char>`. File: `src/point_cloud_split.cpp`.

**Test for the fix session.** The ASan suite (leak check on) with a Catch2 case: identical points give a negative return value and unchanged outputs.

## B-081: `obj_overlap_test` docs call `obj_ind` 1-based and "the output of `obj_file_read`", which is 0-based; IDs are returned as given

- Status: fixed (WP-12): documentation
- Severity: S3, because a user following the documentation misreads the returned IDs by one.
- Kind: doc
- Mark: [R]
- Where: the documentation block of `src/obj_overlap_test.cpp` (inputs `obj_ind`, return value "1-based") and `include/quadriga_tools.hpp:258` ("Object index, 1-based"); `src/obj_file_read.cpp:403, 439` documents `obj_ind` as 0-based.

**What happens.** The same geometry with object IDs {0, 1} returns `0 1`; IDs are returned exactly as given.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-14/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-14: obj_overlap_test documents obj_ind as 1-based ("output of obj_file_read"), but obj_file_read is 0-based and ids are returned as given
#include "quadriga_tools.hpp"
#include <cstdio>
static void test(arma::uword id0, arma::uword id1)
{
    arma::mat A = quadriga_lib::cube<double>(), B = quadriga_lib::cube<double>({1, 0.5, 0.5}, {}, {1, 0, 0}); // B overlaps A
    arma::mat mesh = arma::join_cols(A, B);
    arma::uvec obj = arma::join_cols(arma::uvec(12, arma::fill::value(id0)), arma::uvec(12, arma::fill::value(id1)));
    arma::uvec r = quadriga_lib::obj_overlap_test(&mesh, &obj);
    printf("obj_ind ids {%llu, %llu} -> returned:", (unsigned long long)id0, (unsigned long long)id1);
    for (auto v : r) printf(" %llu", (unsigned long long)v);
    printf("\n");
}
int main()
{
    test(0, 1); // 0-based ids as produced by obj_file_read
    test(1, 2);
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-14/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-14
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
obj_ind ids {0, 1} -> returned: 0 1
obj_ind ids {1, 2} -> returned: 1 2
```

**Proposed fix.** Documentation: "`obj_ind`: object ID per triangle, any integers (e.g. the 0-based `obj_ind` of `obj_file_read`); returns the IDs of the overlapping objects as given". Also the header comment; check the MEX/Python wrapper texts (the MEX wrapper converts 1-based to 0-based). Files: `src/obj_overlap_test.cpp`, `include/quadriga_tools.hpp`.

**Test for the fix session.** Optional Catch2: the returned IDs equal the input IDs ({0, 1}).

## B-082: `icosphere(radius = 0)` returns NaN; `cube` / `plane` with a negative scale return inward / -z normals, contrary to the docs

- Status: fixed (WP-12)
- Severity: S3, because degenerate input is accepted without an error and the result contradicts the documentation ("outward-facing normals", "normals point in +z").
- Kind: wrong result / validation
- Mark: [R]
- Where: `src/icosphere.cpp:54` (rejects only `radius < 0`; NaN also passes), `src/cube.cpp:56-61`, `src/plane.cpp:59-62` (negative scale accepted, winding not swapped).

**What happens.** `icosphere(1, 0.0)`: centres (NaN, NaN, NaN), length NaN. `cube({-1})` and `cube({1, 1, -2})`: 0 % of the faces point outward (100 % for the default). `plane({-1, 1})`: normal z of face 0 is -4 (+4 for the default).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-15/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-15: icosphere(radius=0) returns NaN; cube/plane with negative scale flip the winding (doc: outward / +z normals)
#include "quadriga_tools.hpp"
#include <cstdio>
static double outward(const arma::mat &m) // fraction of faces whose normal points away from the mesh centroid
{
    arma::rowvec c = arma::mean(arma::join_cols(arma::join_cols(m.cols(0, 2), m.cols(3, 5)), m.cols(6, 8)), 0);
    arma::uword n = 0;
    for (arma::uword i = 0; i < m.n_rows; ++i)
    {
        arma::vec3 a = m.row(i).cols(0, 2).t(), b = m.row(i).cols(3, 5).t(), d = m.row(i).cols(6, 8).t();
        arma::vec3 nv = arma::cross(b - a, d - a), ctr = (a + b + d) / 3.0 - c.t();
        if (arma::dot(nv, ctr) > 0) ++n;
    }
    return (double)n / (double)m.n_rows;
}
static double plane_nz(const arma::mat &p)
{
    arma::vec3 a = p.row(0).cols(0, 2).t(), b = p.row(0).cols(3, 5).t(), d = p.row(0).cols(6, 8).t();
    arma::vec3 nv = arma::cross(b - a, d - a);
    return nv(2);
}
int main()
{
    arma::mat c; arma::vec l;
    quadriga_lib::icosphere<double>(1, 0.0, &c, &l);
    printf("icosphere radius 0: center(0,:) = %g %g %g, length(0) = %g\n", c(0, 0), c(0, 1), c(0, 2), l(0));
    printf("cube default: outward fraction %.2f\n", outward(quadriga_lib::cube<double>()));
    printf("cube scale {-1}: outward fraction %.2f\n", outward(quadriga_lib::cube<double>({-1.0})));
    printf("cube scale {1,1,-2}: outward fraction %.2f\n", outward(quadriga_lib::cube<double>({1.0, 1.0, -2.0})));
    printf("plane default: normal z of face 0 = %g\n", plane_nz(quadriga_lib::plane<double>()));
    printf("plane scale {-1,1}: normal z of face 0 = %g\n", plane_nz(quadriga_lib::plane<double>({-1.0, 1.0})));
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-15/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-15
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
icosphere radius 0: center(0,:) = -nan -nan -nan, length(0) = nan
cube default: outward fraction 1.00
cube scale {-1}: outward fraction 0.00
cube scale {1,1,-2}: outward fraction 0.00
plane default: normal z of face 0 = 4
plane scale {-1,1}: normal z of face 0 = -4
```

**Proposed fix.** `icosphere`: require a finite `radius > 0`. `cube` / `plane`: if the product of the scales is negative, swap two vertices of every triangle after the transform, or reject negative scales (the user decides). Files: `src/icosphere.cpp`, `src/cube.cpp`, `src/plane.cpp`.

**Test for the fix session.** Catch2: `icosphere(1, 0.0)` throws; `cube({-1})` has all face normals pointing outward.

## B-083: `interp` 1-D/2-D index search is O(n·m): 4 s for 160 000 samples

- Status: fixed (WP-13): binary search for grids of more than 16 values, results bit-identical
- Severity: S3, because it is a clear performance defect for long series, which a binary search or a merge walk makes negligible.
- Kind: performance
- Mark: [M] (the re-run below ran under `flock /tmp/bench.lock` at a 1-minute load of 1.59; the 160 000 point is from phase 1 at a load of 10.5 and is provisional)
- Where: `src/interp.cpp:121-138` (linear scan from 0 for every output sample).

**What happens.** n_in = n_out = 10 000: 0.016 s; 40 000: 0.243 s (re-run below); 160 000: 4.09 s (phase 1, provisional). The time grows 15-16× per 4× n, i.e. O(n·m).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-16/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-16: interp_1D index search is O(n_in * n_out) (timing, provisional: depends on load)
#include "quadriga_math.hpp"
#include <chrono>
#include <cstdio>
int main()
{
    for (arma::uword n : {10000, 40000})
    {
        arma::vec g = arma::linspace(0, 1, n), d = arma::randu(n), o = arma::linspace(0, 1, n);
        auto t0 = std::chrono::steady_clock::now();
        arma::vec r = quadriga_lib::interp_1D<double>(d, g, o);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("interp_1D n_in = n_out = %llu: %.3f s\n", (unsigned long long)n, s);
    }
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-16/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-16
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
echo "load before: $(cat /proc/loadavg)"
flock /tmp/bench.lock "$D/repro" || echo "exit code $?"
echo "load after: $(cat /proc/loadavg)"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
load before: 1.59 3.61 6.94 1/2770 715719
interp_1D n_in = n_out = 10000: 0.016 s
interp_1D n_in = n_out = 40000: 0.243 s
load after: 1.59 3.61 6.94 1/2770 715722
```

**Proposed fix.** `std::upper_bound` on the sorted grid per output sample (O(m log n)), or a merge walk when the output is sorted. File: `src/interp.cpp`. Re-measure on an idle machine under `flock /tmp/bench.lock`.

**Test for the fix session.** A regression test of the results only (duplicates, unsorted grid, extrapolation, NaN; see B-107); a timing guard is optional.

## B-084: `obj_overlap_test`: reason strings are non-deterministic (unseeded shuffle, thread count), and a flag array is read outside its critical section

- Status: fixed (WP-12): the reason of an object is the finding of its first overlapping pair in ID order
- Severity: S3, because the returned IDs are deterministic, but the reasons vary between calls and thread counts (`md/SPEC_CPP_API.md`: results must not depend on the thread count; random numbers only through an explicit seed); the flag race is formal UB that only affects an early skip.
- Kind: determinism / race
- Mark: [R] non-determinism; [C] race (a TSan build of the library was not allowed)
- Where: `src/obj_overlap_test.cpp:522-523` (`arma::shuffle` with Armadillo's global RNG), `:546` (`intersecting_objects[...]` read outside the `omp critical` that writes it at 567-569 and 829-831); `why[]` is written by whichever pair finishes last.

**What happens.** Four mutually overlapping cubes, six calls (3 with 1 thread, 3 with 4 threads): IDs 1..4 every time, but 5 distinct reason vectors, also between consecutive single-thread calls.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-17/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-17: obj_overlap_test reason strings differ between repeated calls / thread counts (4 mutually overlapping cubes)
#include "quadriga_tools.hpp"
#include <omp.h>
#include <cstdio>
#include <set>
int main()
{
    arma::mat mesh; arma::uvec obj;
    for (int i = 0; i < 4; ++i)
    {
        arma::mat c = quadriga_lib::cube<double>({1.0}, {}, {0.5 * i, 0.3 * i, 0.1 * i});
        mesh = arma::join_cols(mesh, c);
        obj = arma::join_cols(obj, arma::uvec(12, arma::fill::value(i + 1)));
    }
    std::set<std::string> seen;
    for (int nt : {1, 1, 1, 4, 4, 4})
    {
        omp_set_num_threads(nt);
        std::vector<std::string> why;
        arma::uvec r = quadriga_lib::obj_overlap_test(&mesh, &obj, &why);
        std::string all;
        for (auto &s : why) all += s.substr(0, 40) + " | ";
        seen.insert(all);
        printf("threads %d: ids %llu..%llu, reason[0] = %s\n", nt, (unsigned long long)r.min(), (unsigned long long)r.max(), why[0].substr(0, 70).c_str());
    }
    printf("distinct reason vectors over 6 calls: %zu\n", seen.size());
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-17/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-17
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
threads 1: ids 1..4, reason[0] = 3D Intersect: OBJ-IDs (3,1) @ F:8 {(0,-0.4,1.2),(0,1.6,1.2),(0,1.6,-0.
threads 1: ids 1..4, reason[0] = 3D Intersect: OBJ-IDs (1,4) @ F:4 {(1,1,1),(1,-1,-1),(1,1,-1)} vs. E:1
threads 1: ids 1..4, reason[0] = 3D Intersect: OBJ-IDs (3,1) @ F:8 {(0,-0.4,1.2),(0,1.6,1.2),(0,1.6,-0.
threads 4: ids 1..4, reason[0] = 3D Intersect: OBJ-IDs (4,1) @ F:7 {(2.5,-0.1,1.3),(0.5,-0.1,1.3),(0.5,
threads 4: ids 1..4, reason[0] = 3D Intersect: OBJ-IDs (1,2) @ F:10 {(1,1,1),(1,-1,1),(1,-1,-1)} vs. E:
threads 4: ids 1..4, reason[0] = 3D Intersect: OBJ-IDs (1,3) @ F:10 {(1,1,1),(1,-1,1),(1,-1,-1)} vs. E:
distinct reason vectors over 6 calls: 5
```

**Proposed fix.** Drop the shuffle or use a local RNG with a fixed seed; read and write the flags atomically (or drop the early skip); keep the reason of the lowest (object 1, object 2) pair. File: `src/obj_overlap_test.cpp`.

**Test for the fix session.** Catch2 from the repro: identical reason vectors over repeated calls with 1 and 4 threads.

## B-085: MEX `fast_sincos(x, [])` returns cos instead of sin; the single-precision output is undocumented

- Status: fixed (WP-01)
- Severity: S3, because `[]`, documented as "use the default", flips the result to cosine.
- Kind: wrong result / doc
- Mark: [R]
- Where: `api_mex/fast_sincos.cpp:77` (`qd_mex_get_scalar<bool>(prhs[1], "cosineOnly")` without a default, so `dtype(NAN)` gives `true`); the documentation block does not say that the outputs are `single` for `double` input.

**What happens.** `fast_sincos([0.1 0.5 1], [])` returns `[0.995 0.8776 0.5403]` (= cos); class `single`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-7/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-7: fast_sincos(x, []) returns cos instead of sin; output is single for double input (undocumented)
addpath('/sjc/quadriga-lib-fixes');
x = [0.1 0.5 1];
s = quadriga_lib.fast_sincos(x, []);
fprintf('fast_sincos(x, []) = %s\n', mat2str(double(s), 4));
fprintf('sin(x)             = %s\n', mat2str(sin(x), 4));
fprintf('cos(x)             = %s\n', mat2str(cos(x), 4));
fprintf('class(fast_sincos(double x)) = %s\n', class(quadriga_lib.fast_sincos(x)));
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-7/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-7: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-7)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
fast_sincos(x, []) = [0.995 0.8776 0.5403]
sin(x)             = [0.09983 0.4794 0.8415]
cos(x)             = [0.995 0.8776 0.5403]
class(fast_sincos(double x)) = single
```

**Proposed fix.** Pass `false` as the default; state "single" in the outputs section. File: `api_mex/fast_sincos.cpp`.

**Test for the fix session.** MOxUnit: `isequal(fast_sincos(x, []), fast_sincos(x))`.

## B-086: MEX `arrayant_qdant_write` (separate inputs) writes `center_freq = NaN` for `[]` instead of the documented 299792458

- Status: fixed (WP-01)
- Severity: S3, because the documented default is not applied and NaN is written to the file.
- Kind: wrong result / doc
- Mark: [R]
- Where: `api_mex/arrayant_qdant_write.cpp:118` (`qd_mex_get_scalar<double>(prhs[13])` without a default).

**What happens.** Writing an omni pattern with `center_freq = []` and reading it back gives `center_freq = NaN`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-8/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-8: arrayant_qdant_write (separate inputs) writes center_freq = NaN for [] (doc default 299792458)
addpath('/sjc/quadriga-lib-fixes');
ant = quadriga_lib.arrayant_generate('omni');
fn = fullfile(pwd, 'f8.qdant'); if exist(fn, 'file'), delete(fn); end
quadriga_lib.arrayant_qdant_write(fn, [], [], [], ant.e_theta_re, ant.e_theta_im, ant.e_phi_re, ant.e_phi_im, ...
    ant.azimuth_grid, ant.elevation_grid, [], [], [], [], 'x');
a2 = quadriga_lib.arrayant_qdant_read(fn, 1);
fprintf('center_freq read back = %g (documented default 299792458)\n', a2.center_freq);
delete(fn);
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-8/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-8: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-8)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
center_freq read back = NaN (documented default 299792458)
```

**Proposed fix.** `qd_mex_get_scalar<double>(prhs[13], "center_freq", 299792458.0)`. File: `api_mex/arrayant_qdant_write.cpp`.

**Test for the fix session.** MOxUnit: write with `[]`, read back, assert 299792458.

## B-087: `arrayant_generate('multibeam')` uses a 90° element beamwidth by default in MATLAB and Python; docs (and the C++ default) say 120°

- Status: fixed (WP-15): the MATLAB and Python wrappers pass 120 deg for multibeam; round 2: the C++ generator maps a beamwidth <= 0 to 120 deg (was 90), breaking for C++ callers that pass 0
- Severity: S3, because the documented default is not what is computed (max difference 0.48 in the pattern), in both wrappers.
- Kind: doc / cross-API
- Mark: [R]
- Where: `api_mex/arrayant_generate.cpp:174-177` passes `az_3dB = 0` and `el_3dB = 0`; the C++ function (`src/qd_arrayant_generate.cpp:805-806`) maps `<= 0` to 90; the MEX documentation table says "multibeam ... 120 | 120"; `api_python/qpy_arrayant_generate.cpp:55, 204-221` has the same ("120 for multibeam"). Also: MATLAB cannot set `apply_weights`, Python can.

**What happens.** `arrayant_generate('multibeam')` equals the call with beamwidth 90 (max difference 0) and differs from the call with 120 by up to 0.479; the same in Python.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-9/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-9: arrayant_generate('multibeam') uses 90 deg element beamwidth, documented 120 (C++ default 120)
addpath('/sjc/quadriga-lib-fixes');
a0 = quadriga_lib.arrayant_generate('multibeam');
a120 = quadriga_lib.arrayant_generate('multibeam', [], [], 120, 120);
a90 = quadriga_lib.arrayant_generate('multibeam', [], [], 90, 90);
fprintf('MATLAB default vs az/el_3dB=120: max diff %g\n', max(abs(a0.e_theta_re(:) - a120.e_theta_re(:))));
fprintf('MATLAB default vs az/el_3dB=90 : max diff %g\n', max(abs(a0.e_theta_re(:) - a90.e_theta_re(:))));
```

`repro.py`:
```python
# F-9: same check in Python (doc also says 120 for multibeam)
import numpy as np, quadriga_lib as q
a0 = q.arrayant.generate('multibeam')
a120 = q.arrayant.generate('multibeam', az_3dB=120, el_3dB=120)
a90 = q.arrayant.generate('multibeam', az_3dB=90, el_3dB=90)
print('Python default vs 120: max diff %g' % np.abs(a0['e_theta_re'] - a120['e_theta_re']).max())
print('Python default vs 90 : max diff %g' % np.abs(a0['e_theta_re'] - a90['e_theta_re']).max())
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-9/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-9: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-9)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== Python"
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
MATLAB default vs az/el_3dB=120: max diff 0.479469
MATLAB default vs az/el_3dB=90 : max diff 0
== Python
Python default vs 120: max diff 0.479469
Python default vs 90 : max diff 0
```

**Proposed fix.** Pass 120 (the C++ default) when the user gives 0 or `[]` for multibeam, or change both documentation blocks to 90. The user decides. Files: `api_mex/arrayant_generate.cpp`, `api_python/qpy_arrayant_generate.cpp`.

**Test for the fix session.** MOxUnit and pytest: the default multibeam call equals the call with the documented beamwidth.

## B-088: `calc_angular_spread` defaults differ: MATLAB and Python disable wrapping and the bank angle, C++ enables both

- Status: fixed (WP-15): C++ calc_angular_spreads_sphere defaults to wrapping off and bank angle off, as the wrappers; breaking for C++ callers
- Severity: S3, because the same default call gives different spreads in C++ (1.678) and MATLAB/Python (1.627).
- Kind: cross-API
- Mark: [R]
- Where: `api_mex/calc_angular_spread.cpp:56-57` (defaults `wrapping = false`, `calc_bank_angle = false`); `include/quadriga_tools.hpp:62-63` (`disable_wrapping = false`, i.e. wrapping on, `calc_bank_angle = true`); `api_python/qpy_calc_angular_spread.cpp` matches MATLAB.

**What happens.** az = [0.1; 2.9; -3.0], el = [0.1; -0.2; 0.3], p = [1; 0.8; 0.5]: MATLAB and Python default `as = 1.627282, es = 0.190512`; with wrapping and bank angle on `as = 1.677888, es = 0.157755`, which is the C++ default result.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-10/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// F-10: C++ calc_angular_spreads_sphere with all defaults on the same input
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    std::vector<arma::vec> az = {{0.1, 2.9, -3.0}}, el = {{0.1, -0.2, 0.3}}, p = {{1, 0.8, 0.5}};
    arma::vec as, es;
    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, p, &as, &es);
    printf("C++ default                    as=%.6f es=%.6f\n", as(0), es(0));
    return 0;
}
```

`repro.m`:
```matlab
% F-10: calc_angular_spread defaults (wrapping off, bank off) differ from C++ defaults (wrapping on, bank on)
addpath('/sjc/quadriga-lib-fixes');
az = [0.1; 2.9; -3.0]; el = [0.1; -0.2; 0.3]; p = [1; 0.8; 0.5];
[as, es] = quadriga_lib.calc_angular_spread(az, el, p);
fprintf('MATLAB default                 as=%.6f es=%.6f\n', as, es);
[as, es] = quadriga_lib.calc_angular_spread(az, el, p, true, true);
fprintf('MATLAB wrapping=true,bank=true as=%.6f es=%.6f\n', as, es);
```

`repro.py`:
```python
# F-10: Python default on the same input (same as MATLAB, differs from C++)
import numpy as np, quadriga_lib as q
r = q.tools.calc_angular_spread([np.array([0.1, 2.9, -3.0])], [np.array([0.1, -0.2, 0.3])], [np.array([1, 0.8, 0.5])])
print('Python default                 as=%.6f es=%.6f' % (np.ravel(r[0])[0], np.ravel(r[1])[0]))
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-10/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-10: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-10)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== C++ (build_linux/libquadriga.a)"
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(cd "$D" && OMP_NUM_THREADS=1 ./repro) || echo "exit code $?"
echo "== Python"
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
MATLAB default                 as=1.627282 es=0.190512
MATLAB wrapping=true,bank=true as=1.677888 es=0.157755
== C++ (build_linux/libquadriga.a)
C++ default                    as=1.677888 es=0.157755
== Python
Python default                 as=1.627282 es=0.190512
```

**Proposed fix.** Decide which defaults are intended and align C++ or both wrappers (a change of documented behaviour on one side). Files: `include/quadriga_tools.hpp` / `src/qd_tools_calc_angular_spread.cpp`, or `api_mex/calc_angular_spread.cpp` and `api_python/qpy_calc_angular_spread.cpp`.

**Test for the fix session.** MOxUnit and pytest: the default call equals the C++ default result on the input above.

## B-089: `triangle_mesh_segmentation` pads `mtl_ind_out` with material 1 in MATLAB, with 0 ("no material") in C++ and Python

- Status: fixed (WP-15): the MEX wrappers pass mtl_ind through; MATLAB padding entries are 0 (breaking for MATLAB)
- Severity: S3, because padding faces get a real material in MATLAB only, and an input 0 ("no material") underflows on the way.
- Kind: cross-API
- Mark: [R] (also seen by area G)
- Where: `api_mex/triangle_mesh_segmentation.cpp:60-62, 82` (applies -1 then +1 to indices that C++ already treats as 1-based with 0 = none, `src/triangle_mesh_segmentation.cpp:18, 42`: "padding uses index 0"); the MATLAB documentation says "padding entries get index 1". `api_mex/subdivide_triangles.cpp:51-53, 84` does the same shift, harmless there because the unsigned wrap restores 0.

**What happens.** `cube()` with `mtl = (1:12)'` and `vec_size = 8`: MATLAB `mtl_ind_out = [1 ... 12 1 1 1 1]`, Python `[1, ..., 12, 0, 0, 0, 0]`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-11/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-11: triangle_mesh_segmentation pads mtl_ind_out with 1 in MATLAB, 0 in C++/Python
addpath('/sjc/quadriga-lib-fixes');
[~, ~, mi, mtlR] = quadriga_lib.triangle_mesh_segmentation(quadriga_lib.cube(), 1024, 8, (1:12)');
fprintf('MATLAB mesh_index  = %s (0 = padding)\n', mat2str(double(mi.')));
fprintf('MATLAB mtl_ind_out = %s\n', mat2str(double(mtlR.')));
```

`repro.py`:
```python
# F-11: same call in Python
import numpy as np, quadriga_lib as q
r = q.RTtools.triangle_mesh_segmentation(q.RTtools.cube(), 1024, 8, np.arange(1, 13, dtype=np.uint64))
print('Python mtl_ind_out =', np.asarray(r[3]).ravel().tolist())
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-11/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-11: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-11)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== Python"
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
MATLAB mesh_index  = [1 2 3 4 5 6 7 8 9 10 11 12 0 0 0 0] (0 = padding)
MATLAB mtl_ind_out = [1 2 3 4 5 6 7 8 9 10 11 12 1 1 1 1]
== Python
Python mtl_ind_out = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0]
```

**Proposed fix.** Pass `mtl_ind` through unchanged in both MEX wrappers (no shift) and change the MATLAB documentation to "padding entries get 0". Files: `api_mex/triangle_mesh_segmentation.cpp`, `api_mex/subdivide_triangles.cpp`.

**Test for the fix session.** MOxUnit: the padding entries of `mtl_ind_out` are 0.

## B-090: MEX `baseband_freq_response` with `carrier_freq`: descending carriers are rejected, duplicate carriers give silent NaN (Python handles both)

- Status: fixed (WP-15): MEX follows Python (span from min / max, zero span with several carriers throws, reference = input carrier or min, input order kept)
- Severity: S3, because valid carrier lists are rejected or give silent NaN in MATLAB, while Python handles them.
- Kind: cross-API / wrong result
- Mark: [R]
- Where: `api_mex/baseband_freq_response.cpp:178-189, 207-219` (bandwidth = last - first carrier, negative when descending, 0 for duplicates; `pilot_grid = (f - fc) / bw`). Python (`api_python/qpy_baseband_freq_response.cpp`) uses min/max and rejects a zero span. The mode logic sits in the wrapper, against the "thin wrapper" rule of `md/SPEC_MATLAB_API.md`.

**What happens.** Two paths, `fc = 3e9`, carriers `fc + [0 10e6 20e6]'`: ascending works; descending gives "Bandwidth cannot be negative"; two equal carriers give `[NaN NaN]` without an error. Python returns the reversed order for descending and a clean error for equal carriers.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-12/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-12: baseband_freq_response with center_freq + carrier_freq: descending -> error, duplicates -> NaN
addpath('/sjc/quadriga-lib-fixes');
cre = reshape([1 0.5], 1, 1, 2); cim = zeros(1, 1, 2); dl = reshape([10e-9 35e-9], 1, 1, 2);
fc = 3e9; fa = fc + [0 10e6 20e6]';
[hr, hi] = quadriga_lib.baseband_freq_response(cre, cim, dl, [], [], fc, fa);
fprintf('ascending : %s\n', mat2str(squeeze(hr + 1i*hi).', 5));
try
    [hr, hi] = quadriga_lib.baseband_freq_response(cre, cim, dl, [], [], fc, flipud(fa));
    fprintf('descending: %s\n', mat2str(squeeze(hr + 1i*hi).', 5));
catch e, fprintf('descending: error: %s\n', e.message); end
[hr, hi] = quadriga_lib.baseband_freq_response(cre, cim, dl, [], [], fc, [fc; fc]);
fprintf('two equal carriers: %s\n', mat2str(squeeze(hr + 1i*hi).', 5));
```

`repro.py`:
```python
# F-12: same inputs in Python
import numpy as np, quadriga_lib as q
c = [np.array([1, 0.5]).reshape(1, 1, 2).astype(complex)]; d = [np.array([10e-9, 35e-9]).reshape(1, 1, 2)]
fc = 3e9; fa = fc + np.array([0, 10e6, 20e6])
for lab, fo in [('ascending ', fa), ('descending', fa[::-1]), ('two equal carriers', np.array([fc, fc]))]:
    try: print(lab + ':', np.round(q.channel.baseband_freq_response(coeff=c, delay=d, freq_in=np.array([fc]), freq_out=fo).ravel(), 5))
    except Exception as e: print(lab + ': error:', e)
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-12/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-12: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-12)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== Python"
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
ascending : [1.5+0i 0.51512-0.99229i 0.15451-0.47553i]
descending: error: baseband_freq_response: Bandwidth cannot be negative
two equal carriers: [complex(NaN,NaN) complex(NaN,NaN)]
== Python
ascending : [1.5    +0.j      0.51512-0.99229j 0.15451-0.47553j]
descending: [0.15451-0.47553j 0.51512-0.99229j 1.5    +0.j     ]
two equal carriers: error: 'freq_out' values must span a positive range.
```

**Proposed fix.** Compute the span from min/max, reject a zero span for more than one carrier, keep the output order; better, move the carrier-to-pilot conversion into C++ and share it with Python. File: `api_mex/baseband_freq_response.cpp` (or a C++ helper).

**Test for the fix session.** MOxUnit: descending carriers give the reversed ascending result; duplicate carriers give a clean error.

## B-091: MEX `get_channels_irs` `i_irs` is 0-based, against the 1-based rule of the MATLAB spec; the doc does not say which

- Status: fixed (WP-15): MEX i_irs is 1-based (default 1, 0 rejected); breaking for MATLAB
- Severity: S3, because a MATLAB user who passes 1 for the first codebook port silently gets the second.
- Kind: doc / spec
- Mark: [R]
- Where: `api_mex/get_channels_irs.cpp:102` (passes the value through); the documentation says only "IRS codebook port index; default: `0`"; `md/SPEC_MATLAB_API.md` requires 1-based indices.

**What happens.** An IRS with 2 ports: `i_irs = 0` and `1` work (different results), `i_irs = 2` gives "IRS codebook index out of bound".

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-13/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-13: get_channels_irs i_irs is 0-based in MATLAB (SPEC: 1-based); IRS below has 2 codebook ports
addpath('/sjc/quadriga-lib-fixes');
ant = quadriga_lib.arrayant_generate('omni');
irs = quadriga_lib.arrayant_copy_element(ant, 1, 2);          % 2 elements
irs.coupling_re = [1 0; 0 0.5]; irs.coupling_im = zeros(2);   % 2 ports (codebook entries)
irs.element_pos = [0 0.5; 0 0; 0 0];
tx = [0;0;1]; rx = [20;10;1]; ip = [10;0;5]; o = [0;0;0];
f1 = [2;0;3]; l1 = [6;0;4]; f2 = [12;1;5]; l2 = [16;5;4];
pl1 = norm(tx-f1) + norm(f1-l1) + norm(l1-ip); pl2 = norm(ip-f2) + norm(f2-l2) + norm(l2-rx);
M = [1;0;0;0;0;0;1;0];
for i_irs = [0 1 2]
    try
        cr = quadriga_lib.get_channels_irs(ant, ant, irs, f1, l1, 1, pl1, M, f2, l2, 1, pl2, M, ...
            tx, o, rx, o, ip, o, i_irs, -300, 0, false, [], []);
        fprintf('i_irs=%d: OK, abs(coeff_re) = %.6f\n', i_irs, abs(cr(:)));
    catch e
        fprintf('i_irs=%d: error: %s\n', i_irs, e.message);
    end
end
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-13/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-13: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-13)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
i_irs=0: OK, abs(coeff_re) = 1.000000
i_irs=1: OK, abs(coeff_re) = 0.250000
i_irs=2: error: get_channels_irs: IRS codebook index out of bound.
```

**Proposed fix.** The user decides: make it 1-based (default 1, reject 0, subtract 1; an API change) as the spec requires, or keep it 0-based and say so explicitly in the documentation. File: `api_mex/get_channels_irs.cpp`.

**Test for the fix session.** Depends on the decision.

## B-092: `arrayant_qdant_read(fn)` returns every entry as a struct array in MATLAB, Python's default reads ID 1

- Status: fixed (WP-15): MATLAB keeps reading all entries and throws when they do not form a frequency set (arrayant_is_valid_multi); Python qdant_read(fn, id=0) checks the same
- Severity: S3, because the same default call differs between the APIs, and MATLAB returns a struct array of unrelated antennas, which its own convention treats as a frequency-dependent model.
- Kind: cross-API
- Mark: [R]
- Where: `api_mex/arrayant_qdant_read.cpp:73-88` (reads all entries via `qdant_read_multi` when `id` is omitted); Python `qdant_read(fn, id=1)`.

**What happens.** A file with ID 1 = omni and ID 2 = xpol: MATLAB/Octave returns a 1×2 struct array (1 and 2 elements); Python returns one dict with shape (19, 37, 1).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-14/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-14: arrayant_qdant_read(fn) returns all entries as a struct array (even unrelated antennas); Python default reads id 1
addpath('/sjc/quadriga-lib-fixes');
fn = fullfile(pwd, 'two.qdant'); if exist(fn, 'file'), delete(fn); end
quadriga_lib.arrayant_qdant_write(fn, quadriga_lib.arrayant_generate('omni'), 1);   % 1 element, 10 deg grid
quadriga_lib.arrayant_qdant_write(fn, quadriga_lib.arrayant_generate('xpol'), 2);   % 2 elements
r = quadriga_lib.arrayant_qdant_read(fn);
fprintf('MATLAB arrayant_qdant_read(fn): size %s, n_elements per entry %s\n', mat2str(size(r)), ...
    mat2str(arrayfun(@(x) size(x.e_theta_re, 3), r)));
```

`repro.py`:
```python
# F-14: same file (written by repro.m) read with the Python default
import os, quadriga_lib as q
r = q.arrayant.qdant_read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'two.qdant'))
print('Python qdant_read(fn): single dict, e_theta_re shape', r['e_theta_re'].shape)
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-14/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-14: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-14)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== Python"
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
MATLAB arrayant_qdant_read(fn): size [1 2], n_elements per entry [1 2]
== Python
Python qdant_read(fn): single dict, e_theta_re shape (19, 37, 1)
```

**Proposed fix.** The user decides: align the defaults; if MATLAB keeps reading all entries, validate that they form a consistent frequency set (same grids and element count) or fall back to ID 1. File: `api_mex/arrayant_qdant_read.cpp` (or the Python default).

**Test for the fix session.** MOxUnit and pytest pin the decided default.

## B-093: MEX: arrays with more than 4 dimensions are silently truncated; a 4-D dataset written with `hdf5_write_dset` comes back 3-D

- Status: fixed (WP-06)
- Severity: S3, because invalid input is accepted with data loss and without an error.
- Kind: wrong result (invalid input accepted)
- Mark: [R]
- Where: `src/mex_helper_functions.hpp:75-86` (`qd_mex_get_dims` multiplies only dimensions 1-4), the same pattern in `qd_mex_anycast:231-236` (which also merges dimensions 3 and 4) and `qd_mex_matlab2vector_*`.

**What happens.** `fast_sincos(ones(2,2,2,2,2))` returns size `[2 2 4]` (16 of 32 values); `hdf5_write_dset` of `rand(2,3,4,2)` reads back as `[2 3 8]`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-15/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-15: arrays with more than 4 dims are silently truncated; a 4-D HDF5 dataset comes back 3-D
addpath('/sjc/quadriga-lib-fixes');
s = quadriga_lib.fast_sincos(ones(2,2,2,2,2));
fprintf('fast_sincos(ones(2,2,2,2,2)): output size %s, numel %d (input numel 32)\n', mat2str(size(s)), numel(s));
fn = fullfile(pwd, 'f15.hdf5'); if exist(fn, 'file'), delete(fn); end
quadriga_lib.hdf5_create_file(fn, [2 2]);
quadriga_lib.hdf5_write_dset(fn, [1 1], 'd4', rand(2,3,4,2));
r = quadriga_lib.hdf5_read_dset(fn, [1 1], 'd4');
fprintf('hdf5_write_dset rand(2,3,4,2) -> read back size %s\n', mat2str(size(r)));
delete(fn);
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-15/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-15: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-15)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
fast_sincos(ones(2,2,2,2,2)): output size [2 2 4], numel 16 (input numel 32)
hdf5_write_dset rand(2,3,4,2) -> read back size [2 3 8]
```

**Proposed fix.** Error when `mxGetNumberOfDimensions` exceeds what the helper supports (4, or 3 where only cubes are allowed); in `hdf5_write_dset` reject 4-D input or document the reshape. File: `src/mex_helper_functions.hpp`.

**Test for the fix session.** MOxUnit: a 5-D input raises `quadriga_lib:CPPerror`.

## B-094: MEX: index value 0 is not rejected in several wrappers (silent wrong result or a misleading error)

- Status: fixed (WP-06); zero checks only where the wrapper converts to a 0-based C++ index (`obj_ind` is 0-based in C++)
- Severity: S3, because an invalid 1-based index is accepted with a wrong answer or rejected with a misleading message.
- Kind: wrong result / unclear error
- Mark: [R]
- Where: `api_mex/point_inside_mesh.cpp:56-58` (`obj_ind -= 1` without a zero check), `api_mex/obj_file_write.cpp:79-81, 89-91`, `api_mex/arrayant_interpolate.cpp:116` (`typecast - 1`), `api_mex/channel_export_obj_file.cpp:89-94` (`i_snap - 1`).

**What happens.** `point_inside_mesh` with `obj_ind = 0` returns `[0 0]` (point 1 is inside, `obj_ind = 1` gives `[1 0]`); `obj_file_write` with `obj_ind = 0` gives "'obj_names' is missing or too short ..."; `arrayant_interpolate` with element 0 gives "Element indices 'i_element' cannot exceed the array antenna size."

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-16/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-16: index value 0 not checked: silent wrong result or misleading error
addpath('/sjc/quadriga-lib-fixes');
mesh = quadriga_lib.cube();       % unit cube at origin; point 1 is inside, point 2 outside
r1 = quadriga_lib.point_inside_mesh([0 0 0; 5 5 5], mesh, ones(12,1));
r0 = quadriga_lib.point_inside_mesh([0 0 0; 5 5 5], mesh, zeros(12,1));
fprintf('point_inside_mesh obj_ind=1: %s ; obj_ind=0: %s (no error)\n', mat2str(double(r1.')), mat2str(double(r0.')));
try, quadriga_lib.obj_file_write(fullfile(pwd, 'f16.obj'), mesh, zeros(12,1), [], {'a'});
catch e, fprintf('obj_file_write obj_ind=0 -> %s\n', e.message); end
try, v = quadriga_lib.arrayant_interpolate(quadriga_lib.arrayant_generate('omni'), 0, 0, 0);
catch e, fprintf('arrayant_interpolate element=0 -> %s\n', e.message); end
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-16/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-16: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-16)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
point_inside_mesh obj_ind=1: [1 0] ; obj_ind=0: [0 0] (no error)
obj_file_write obj_ind=0 -> obj_file_write: 'obj_names' is missing or too short for the given 'obj_ind'.
arrayant_interpolate element=0 -> arrayant_interpolate: Element indices 'i_element' cannot exceed the array antenna size.
```

**Proposed fix.** Add the spec's zero check ("Entries in 'x' cannot be 0 (1-based index)") before each subtraction. Files: the four wrappers.

**Test for the fix session.** MOxUnit: each call with index 0 raises `quadriga_lib:CPPerror` with the zero-index message.

## B-095: Diffraction level of detail 7: accepted by C++, empty `coord` in MATLAB, internal error in Python, rejected by MATLAB `generate_diffraction_paths`, documented as 0-6

- Status: fixed (WP-15): lod documented as 1-7 (generate_diffraction_paths) and 0-7 (calc_diffraction_gain); the wrappers size coord from C++; empty orig / dest throw
- Severity: S3, because the same value gives four different behaviours across functions and APIs, including an Armadillo-internal error in Python, and the documented range does not match any of them.
- Kind: cross-API / doc
- Mark: [R]
- Where: `src/generate_diffraction_paths.cpp:117` accepts lod 1-7 (lod 7: 1 path, 10 segments); its documentation table lists 1-6 and the `calc_diffraction_gain` documentation says 0-6; `api_mex/calc_diffraction_gain.cpp:117-126` and `api_python/qpy_calc_diffraction_gain.cpp:106-115` duplicate the lod-to-segment table without lod 7; the MATLAB `generate_diffraction_paths` wrapper rejects 7 ("must be in range 1..6"). Also (area G) the `coord` documentation "[3, n_seg-1, n_pos]" (C++, MEX, Python) does not match the actual shape for lod 2 (3 × 2 × n_pos).

**What happens.** C++ `calc_diffraction_gain` with lod 7 returns `coord` of 3×9×n_pos; MATLAB returns `coord` of size `[3 0 2]`; Python raises `RuntimeError: Cube::init(): mismatch between size of auxiliary memory and requested size`; lod -1 and 8 give "Input 'lod' must be 1-7." (documentation: 0-6); empty `orig`/`dest` give "At least one of 'xprmat' or 'update' must be non-empty to infer the mode."

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-18/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// F-18: C++ calc_diffraction_gain coord size for lod 2 and 7 on the same input
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    arma::mat orig = {{0, 0, 1}, {0, 0, 1}}, dest = {{10, 0, 1}, {10, 0, 2}};
    arma::mat mesh = quadriga_lib::cube<double>(arma::vec({1, 1, 1}), arma::vec(), arma::vec({5, 0, 5}));
    arma::uvec mtl(12, arma::fill::ones);
    std::unordered_map<std::string, std::vector<double>> mst = {{"a", {1}}, {"b", {0}}, {"c", {0}}, {"d", {0}}};
    for (int lod : {2, 7})
    {
        arma::vec g; arma::mat x; arma::cube c;
        quadriga_lib::calc_diffraction_gain<double>(orig, dest, mesh, mtl, mst, 1e9, lod, &g, &x, &c);
        printf("C++ calc_diffraction_gain lod=%d: coord %llux%llux%llu\n", lod, (unsigned long long)c.n_rows, (unsigned long long)c.n_cols, (unsigned long long)c.n_slices);
    }
    return 0;
}
```

`repro.m`:
```matlab
% F-18: lod 7 is accepted by C++ (doc: 0-6), MATLAB calc_diffraction_gain returns an empty coord for it,
%       MATLAB generate_diffraction_paths rejects it
addpath('/sjc/quadriga-lib-fixes');
orig = [0 0 1; 0 0 1]; dest = [10 0 1; 10 0 2];
mesh = quadriga_lib.cube([1 1 1], [], [5 0 5]);
mst = struct('a', 1, 'b', 0, 'c', 0, 'd', 0);
for lod = [2 7]
    [g, x, c] = quadriga_lib.calc_diffraction_gain(orig, dest, mesh, ones(12,1), mst, 1e9, lod);
    fprintf('MATLAB calc_diffraction_gain lod=%d: size(coord) = %s\n', lod, mat2str(size(c)));
end
try, [r, w] = quadriga_lib.generate_diffraction_paths(orig, dest, 1e9, 7); fprintf('generate_diffraction_paths lod=7 OK\n');
catch e, fprintf('MATLAB generate_diffraction_paths lod=7 -> %s\n', e.message); end
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-18/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-18: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-18)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== C++ (build_linux/libquadriga.a)"
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(cd "$D" && OMP_NUM_THREADS=1 ./repro) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
MATLAB calc_diffraction_gain lod=2: size(coord) = [3 2 2]
MATLAB calc_diffraction_gain lod=7: size(coord) = [3 0 2]
MATLAB generate_diffraction_paths lod=7 -> generate_diffraction_paths: Input 'lod' must be in range 1..6.
== C++ (build_linux/libquadriga.a)
C++ calc_diffraction_gain lod=2: coord 3x2x2
C++ calc_diffraction_gain lod=7: coord 3x9x2
```

Folder `build_linux/scratch/hunt/final/G-11/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-11: calc_diffraction_gain: lod 7 fails with Armadillo-internal error; unclear lod / empty-input errors
import numpy as np, quadriga_lib as q
R = q.RTtools
cube = R.cube(); o = np.array([[-3.0, 0, 0.5]]); d = np.array([[3.0, 0, 0.5]])
for lod in (2, 7, 8, -1):
    try: g, x, c = R.calc_diffraction_gain(o, d, cube, np.zeros(12), {}, 3.5e9, lod); print(f"lod {lod}: ok, coord {c.shape}")
    except Exception as e: print(f"lod {lod}: {type(e).__name__}: {e}  (docs: lod 0-6)")
try: R.calc_diffraction_gain(np.zeros((0, 3)), np.zeros((0, 3)), cube, np.zeros(12), {}, 3.5e9)
except Exception as e: print("empty orig/dest:", type(e).__name__, e)
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-11/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-11
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
lod 2: ok, coord (3, 2, 1)
lod 7: RuntimeError: Cube::init(): mismatch between size of auxiliary memory and requested size  (docs: lod 0-6)
lod 8: ValueError: Input 'lod' must be 1-7.  (docs: lod 0-6)
lod -1: ValueError: Input 'lod' must be 1-7.  (docs: lod 0-6)
empty orig/dest: ValueError At least one of 'xprmat' or 'update' must be non-empty to infer the mode.
```

**Proposed fix.** Decide whether lod 7 is public. Let the C++ function size `coord` (the wrappers copy it out after the call) instead of duplicating the table in both wrappers; document one lod range consistently in C++, MEX and Python; validate empty input with a clear message. Files: `src/generate_diffraction_paths.cpp`, `src/calc_diffraction_gain.cpp` (docs), `api_mex/calc_diffraction_gain.cpp`, `api_mex/generate_diffraction_paths.cpp`, `api_python/qpy_calc_diffraction_gain.cpp`.

**Test for the fix session.** MOxUnit and pytest: lod 7 gives the same `coord` shape as C++ (or a clean error in all APIs, if 7 is not public); empty input gives a clear error.

## B-096: Python `qrt_file_read`: an exception in the main loop while the reader thread runs calls `std::terminate`

- Status: fixed (WP-05)
- Severity: S3, because it needs an exception in the converting loop (in practice an allocation failure for a very large request), but the result is a process abort instead of `MemoryError`.
- Kind: crash (exception safety)
- Mark: [R]
- Where: `api_python/qpy_qrt_file_read.cpp:135` (`std::thread reader`), 168-208 (main loop, can throw: numpy allocations, `py::list::append`), 216 (`join`).

**What happens.** With the address space capped at current + 300 MiB (`resource.setrlimit(RLIMIT_AS)`), `qrt_file_read('tests/data/test.qrt', i_cir=np.zeros(400000))` prints `terminate called without an active exception` and aborts (exit code 134). Expected: `MemoryError`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/G-5/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-5: qrt_file_read: exception in the main loop while the reader thread runs -> std::terminate
import resource, numpy as np, quadriga_lib as q
fn = '/sjc/quadriga-lib-fixes/tests/data/test.qrt'
def vm():
    for l in open('/proc/self/status'):
        if l.startswith('VmSize'): return int(l.split()[1]) * 1024
cir = np.zeros(400000, dtype=np.uint64)          # snapshot 0, 400000 times -> millions of small arrays
lim = vm() + 300 * 2**20
resource.setrlimit(resource.RLIMIT_AS, (lim, lim))  # make an allocation in the converting loop fail
print("address space limited to current + 300 MiB; calling qrt_file_read (expected: MemoryError) ...", flush=True)
try:
    q.channel.qrt_file_read(fn, i_cir=cir)
    print("returned")
except BaseException as e:
    print("clean exception:", type(e).__name__)
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-5/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-5
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
address space limited to current + 300 MiB; calling qrt_file_read (expected: MemoryError) ...
terminate called without an active exception
/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/final/G-5//run.sh: line 4: 722013 Aborted                    (core dumped) OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py"
exit code 134
```

**Cause.** When the main loop throws, stack unwinding destroys the still-joinable `std::thread`, and the destructor of a joinable thread calls `std::terminate`.

**Proposed fix.** An RAII guard around the reader thread: on unwind it sets a stop flag, notifies the condition variable and joins the reader (with the GIL released). Or catch around the loop, stop and join, then rethrow. File: `api_python/qpy_qrt_file_read.cpp`.

**Test for the fix session.** pytest in a subprocess with the same RLIMIT_AS cap: the call ends with `MemoryError` and the subprocess exit code is not 134.

## B-097: Python HDF5 writers reject numpy scalars, 0-d arrays and non-native dtypes that the docs list as supported

- Status: fixed (WP-05); Python bool is stored as an integer, documented
- Severity: S3, because valid data is rejected with "unsupported type" (and `bool` is silently stored as int64); nothing is corrupted.
- Kind: cross-API / invalid rejection
- Mark: [R]
- Where: `api_python/python_arma_adapter.hpp:1414-1514` (`qd_python_anycast`); used by `hdf5_write_dset` and the `par` data of `hdf5_write_channel`. The docs list "scalar ... single, double, int32, uint32, int64, uint64".

**What happens.** `ValueError: Input '<name>' has an unsupported type.` for `np.float32(1.5)`, `np.int64(3)`, a 0-d `np.array(2.0)`, a float64 array after a pickle round trip, an `np.longlong` array, a big-endian `>f8` array and an int16 array. A Python `True` is stored as int64 and reads back as `1`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/G-6/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-6: qd_python_anycast rejects valid numpy types documented as supported
import numpy as np, os, pickle, quadriga_lib as q
C = q.channel
D = os.path.dirname(os.path.abspath(__file__)); fn = os.path.join(D, 'g6.hdf5')
if os.path.exists(fn): os.remove(fn)
C.hdf5_create_file(fn, 1)
vals = {'float32_scalar': np.float32(1.5), 'int64_scalar': np.int64(3), 'zero_d': np.array(2.0),
        'pickled_f64': pickle.loads(pickle.dumps(np.arange(3.0))), 'longlong': np.arange(3, dtype=np.longlong),
        'big_endian': np.arange(3.0).astype('>f8'), 'python_bool': True}
for k, v in vals.items():
    try:
        C.hdf5_write_dset(fn, 0, 0, 0, 0, k, v); print(f"{k:15s} written, read back {C.hdf5_read_dset(fn, name=k)!r}")
    except Exception as e: print(f"{k:15s} -> {type(e).__name__}: {e}")
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-6/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-6
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
float32_scalar  -> ValueError: Input 'float32_scalar' has an unsupported type.
int64_scalar    -> ValueError: Input 'int64_scalar' has an unsupported type.
zero_d          -> ValueError: Input 'zero_d' has an unsupported type.
pickled_f64     -> ValueError: Input 'pickled_f64' has an unsupported type.
longlong        -> ValueError: Input 'longlong' has an unsupported type.
big_endian      -> ValueError: Input 'big_endian' has an unsupported type.
python_bool     written, read back 1
```

**Cause.** Scalars are recognised only as Python `str`, `float` or `int`; numpy scalars and 0-d arrays fall through because no array branch handles `ndim == 0`. Array dtypes are compared by object identity (`arr.dtype().is(py::dtype::of<T>())`), so an equal but distinct descriptor (pickled, byte-swapped, `NPY_LONGLONG` vs `NPY_LONG`) does not match.

**Proposed fix.** Compare with `dtype().equal(...)` or by `num()`/itemsize; convert numpy scalars and 0-d arrays to the scalar branch; accept non-native byte order through `py::array_t<T>::ensure`; decide whether `bool` is accepted (as uint8/int) or rejected. File: `api_python/python_arma_adapter.hpp` (same function as B-005).

**Test for the fix session.** pytest: each listed value round-trips through `hdf5_write_dset` / `hdf5_read_dset` with its documented type.

## B-098: Python adapter `view = true` always copies (confirms item 1.1 of `md/python_fix_plan_v00.md`)

- Status: fixed (WP-05)
- Severity: S3, because docs and spec promise zero-copy input where the code copies (a clear performance and documentation defect); the values are correct.
- Kind: doc / performance
- Mark: [R]
- Where: `api_python/python_arma_adapter.hpp:1055, 1070, 1085, 1184` (`qd_python_numpy2arma_Col/Mat/Cube`, `qd_python_numpy2arma_vecCube`): the third constructor argument `copy_aux_mem = true` copies. First reported as item 1.1 "(verify)" of `md/python_fix_plan_v00.md` (2026-10-05, from reading); this is the first measurement.

**What happens.** A small pybind11 module (`gprobe7.cpp`, built by `run.sh`) calls the adapters with `view = true, strict = true` on Fortran-ordered arrays and compares `memptr()` with the numpy data pointer: `(False, False, False, False)` for Col, Mat, Cube and vecCube. Documentation that is therefore wrong: `get_channels_spherical` batch mode ("list entries are aliased, not copied"), `arrayant.interpolate` `fast_access` ("read without a copy"), `md/SPEC_PYTHON_API.md` ("view = true requests a view").

**Reproduce.**

Folder `build_linux/scratch/hunt/final/G-7/` (create it and save the files below into it if the scratch folder is lost).


`gprobe7.cpp`:
```cpp
// G-7: view=true in qd_python_numpy2arma_* copies instead of aliasing (plan item 1.1)
#include "python_arma_adapter.hpp"
namespace py = pybind11;
static py::tuple view_alias(const py::array_t<double> &a)   // view = true, strict = true
{
    auto c = qd_python_numpy2arma_Col<double>(a, true, true);
    auto m = qd_python_numpy2arma_Mat<double>(a, true, true);
    auto q = qd_python_numpy2arma_Cube<double>(a, true, true);
    auto v = qd_python_numpy2arma_vecCube<double>(a, true, true);
    const double *p = a.data();
    return py::make_tuple(c.memptr() == p, m.memptr() == p, q.memptr() == p, v[0].memptr() == p);
}
PYBIND11_MODULE(gprobe7, m) { m.def("view_alias", &view_alias); }
```

`repro.py`:
```python
# G-7: view=true in qd_python_numpy2arma_* copies instead of aliasing (plan item 1.1)
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gprobe7
a = np.asfortranarray(np.arange(24.0).reshape(2, 3, 4))   # Fortran-contiguous float64: should be aliased
print("memptr == numpy data (Col, Mat, Cube, vecCube) with view=true, strict=true:", gprobe7.view_alias(a), "(expected all True)")
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-7/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-7
# scratch pybind11 module, includes api_python/python_arma_adapter.hpp unchanged
g++-12 -std=c++17 -O1 -shared -fPIC -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -I api_python \
  -I build_linux/pybind11-3.0.0/include -I /usr/include/python3.14 "$D/gprobe7.cpp" \
  -o "$D/gprobe7$(python3-config --extension-suffix)" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
memptr == numpy data (Col, Mat, Cube, vecCube) with view=true, strict=true: (False, False, False, False) (expected all True)
```

**Cause.** `arma::Col<dtype>(ptr, n, true, false)`: in Armadillo the third argument is `copy_aux_mem`, so `true` copies. Real aliasing needs `(ptr, n, false, strict)`.

**Proposed fix.** As item 1.1 of the plan: switch to `(ptr, n, false, true)` or rename/document the flag as "fast contiguous copy". Notes from the audit: no wrapper writes into a `view = true` input (rotate_pattern, copy_element and concat already use `view = false`); with real aliasing, `get_channels_spherical` batch mode releases the GIL while threads read list entries, so the lists must stay referenced and must not be mutated meanwhile; read-only numpy arrays would become `const_cast` targets. The user decides between aliasing and a doc change.

**Test for the fix session.** A C++-side test helper (as in `gprobe7.cpp`) or a timing-free check: with `view = true` the Armadillo `memptr()` equals the numpy data pointer for F-contiguous input.

## B-099: Python: documented kwargs `sub_cloud_index` and `map_to_itu_materials` raise `TypeError`

- Status: documented (WP-05): the docs use the registered names `sub_cloud_ind` and `map_to_itu`
- Severity: S3, because a user who follows the documentation gets `TypeError`.
- Kind: doc
- Mark: [R]
- Where: `api_python/qpy_point_cloud_aabb.cpp:17,28` and `api_python/qpy_ray_point_intersect.cpp:38` document `sub_cloud_index`, registered as `sub_cloud_ind` (`python_RTtools.cpp`); `api_python/qpy_mitsuba_xml_file_write.cpp:40` documents `map_to_itu_materials`, the usage line and the registration use `map_to_itu`. Not in `md/python_fix_plan_v00.md` (its item 2.1 covers `qrt_file_read`).

**What happens.** `point_cloud_aabb(pts, sub_cloud_index=...)`, `ray_point_intersect(..., sub_cloud_index=...)` and `mitsuba_xml_file_write(..., map_to_itu_materials=True)` raise `TypeError: ... incompatible function arguments`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/G-8/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-8: documented kwargs raise TypeError: sub_cloud_index, map_to_itu_materials
import numpy as np, os, quadriga_lib as q
R = q.RTtools
D = os.path.dirname(os.path.abspath(__file__))
pts = np.random.default_rng(0).normal(size=(20, 3)); idx = np.array([0], dtype=np.uint32)
cube = R.cube(); n = cube.shape[0]
vl = cube.reshape(-1, 3); fi = np.arange(3 * n, dtype=np.uint64).reshape(n, 3)
calls = {
 'point_cloud_aabb(sub_cloud_index=)': lambda: R.point_cloud_aabb(pts, sub_cloud_index=idx),
 'ray_point_intersect(sub_cloud_index=)': lambda: R.ray_point_intersect(np.zeros((1, 3)), np.array([[1.0, 0, 0, 1, 1, 0, 1, 0, 1]]),
                                                    np.array([[1.0, 0, 0, 1, 0, 0, 1, 0, 0]]), pts, sub_cloud_index=idx),
 'mitsuba_xml_file_write(map_to_itu_materials=)': lambda: R.mitsuba_xml_file_write(os.path.join(D, 's.xml'), vert_list=vl, face_ind=fi,
        obj_ind=np.zeros(n, np.uint64), mtl_ind=np.zeros(n, np.uint64), obj_names=['o'], mtl_names=['m'], map_to_itu_materials=True)}
for k, f in calls.items():
    try: f(); print(k, "accepted")
    except TypeError as e: print(k, "-> TypeError:", str(e).split('\n')[0])
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-8/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-8
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
point_cloud_aabb(sub_cloud_index=) -> TypeError: point_cloud_aabb(): incompatible function arguments. The following argument types are supported:
ray_point_intersect(sub_cloud_index=) -> TypeError: ray_point_intersect(): incompatible function arguments. The following argument types are supported:
mitsuba_xml_file_write(map_to_itu_materials=) -> TypeError: mitsuba_xml_file_write(): incompatible function arguments. The following argument types are supported:
```

**Cause.** Names in the comment blocks differ from the `py::arg` names in the registration.

**Proposed fix.** Align the names; the registered names are what works today, so changing the docs is the non-breaking choice. The user decides. Files: the three `qpy_*.cpp` comment blocks (or the registrations).

**Test for the fix session.** pytest calls each function with the documented kwarg name.

## B-100: Python: invalid input accepted silently (2-D positions, short `interact_coord`, 4-D arrayant without `center_freq`)

- Status: fixed (WP-05); `center_freq` is required for 4-D arrayant dicts only where the frequency selects data (`interpolate`, `combine_pattern`, `get_channels_multifreq`, `qdant_write`)
- Severity: S3, because invalid input is accepted without an error and gives a result that looks valid.
- Kind: wrong result (invalid input accepted)
- Mark: [R]
- Where:
  - (a) `api_python/python_arma_adapter.hpp:1047` (`qd_python_numpy2arma_Col(..., var_name, n_elem)` checks only `shape[0]`); used for `tx_pos`, `rx_pos` and orientations in `qpy_get_channels_planar.cpp`, `qpy_get_channels_spherical.cpp` (single mode) and `qpy_get_channels_multifreq.cpp`.
  - (b) `api_python/qpy_channel_export_obj_file.cpp:142` (`arma::resize(interact_coord, 3, sum_no_int)`).
  - (c) `api_python/python_quadriga_adapter.hpp:231-232` (a 4-D arrayant dict without `center_freq` gets 299792458 for every entry).

**What happens.** (a) A `(3, 2)` `tx_pos` (documented `(3,)`) is accepted and its first column is used. (b) `interact_coord` with 0 columns while `no_interact` sums to 3 is zero-padded and exported. (c) `interpolate` of a 4-D speaker dict without `center_freq` returns entry 0's pattern for every frequency (`[0.9915 0.9915 0.9915]`, against `[0.9915 0.9824 0.9616]` with `center_freq`).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/G-9/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-9: invalid input accepted silently: (3,k) positions, interact_coord columns, 4D arrayant without center_freq
import numpy as np, os, quadriga_lib as q
A, C = q.arrayant, q.channel
D = os.path.dirname(os.path.abspath(__file__))
o = A.generate('omni'); M = np.array([[1.0, 0, 0, 0, 0, 0, 1, 0]]).T
pos2 = np.array([[0.0, 50.0], [0.0, 0.0], [10.0, 10.0]])          # documented shape: (3,)
r = A.get_channels_planar(o, o, np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1), np.ones(1) * 10, M,
                          pos2, np.zeros(3), np.array([100.0, 0, 10]), np.zeros(3))
print("(a) planar with tx_pos shape (3,2): accepted, delay =", r[2].ravel())
ni = [np.array([1, 0, 2], dtype=np.uint32)]                        # needs 3 interaction coordinates
C.channel_export_obj_file(os.path.join(D, 'g9.obj'), rx_pos=np.array([[1.0], [0], [0]]), tx_pos=np.zeros((3, 1)), no_interact=ni,
                          interact_coord=[np.zeros((3, 0))], center_freq=np.array([1e9]), coeff=[np.ones((1, 1, 3), complex) * 1e-4], gain_min=-300)
print("(b) channel_export_obj_file with 0 interaction coordinates for sum(no_interact)=3: accepted")
s = A.generate_speaker(frequencies=np.array([500.0, 1000.0, 2000.0]), angular_resolution=10)
az, el = np.array([[0.1]]), np.array([[0.2]])
print("(c) interpolate 4D with center_freq   :", A.interpolate(s, az, el)[0].ravel())
del s['center_freq']
print("(c) interpolate 4D without center_freq:", A.interpolate(s, az, el)[0].ravel(), "(accepted, entry 0 repeated)")
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-9/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-9
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
(a) planar with tx_pos shape (3,2): accepted, delay = [-3.00207686e-07]
(b) channel_export_obj_file with 0 interaction coordinates for sum(no_interact)=3: accepted
(c) interpolate 4D with center_freq   : [0.99149546 0.9824022  0.96156958]
(c) interpolate 4D without center_freq: [0.99149546 0.99149546 0.99149546] (accepted, entry 0 repeated)
```

**Cause.** Missing size checks in the adapter and the wrapper; a silent default where the frequency is required to pick entries.

**Proposed fix.** (a) Check `n_elem` against the total element count (`arr.size()`); (b) require `interact_coord[i].n_cols >= sum(no_interact[i])`; (c) require `center_freq` (with one value per entry) for 4-D input. Files: `api_python/python_arma_adapter.hpp`, `api_python/qpy_channel_export_obj_file.cpp`, `api_python/python_quadriga_adapter.hpp`.

**Test for the fix session.** pytest: each of the three cases raises `ValueError`.

## B-101: `arrayant::copy_element` ignores `read_only`; an empty destination gives an Armadillo-internal error

- Status: fixed (WP-09)
- Severity: S4, because the effects are an unenforced flag and an unclear error message.
- Kind: doc / error message
- Mark: [R]
- Where: `src/qd_arrayant.cpp:668-679` (`copy_element`: no `read_only` check; `destination.max()` on an empty vector).

**What happens.** `copy_element` on a `read_only` object succeeds (`n_elements` 4 afterwards); an empty destination gives `Mat::max(): object has no elements`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/A-20/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// A-20: copy_element ignores read_only; empty destination gives an Armadillo-internal error
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    auto a = quadriga_lib::generate_arrayant_omni<double>(10.0);
    a.read_only = true;
    try
    {
        a.copy_element(0, 3);
        std::printf("copy_element on read_only object: no error, n_elements now %llu\n", (unsigned long long)a.n_elements());
    }
    catch (const std::exception &e)
    {
        std::printf("copy_element on read_only: error '%s'\n", e.what());
    }
    try
    {
        a.copy_element(0, arma::uvec());
        std::printf("copy_element with empty destination: no error\n");
    }
    catch (const std::exception &e)
    {
        std::printf("copy_element with empty destination: '%s'\n", e.what());
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/A-20/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
set -o pipefail
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/A-20
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(OMP_NUM_THREADS=1 "$D/repro" 2>&1 | head -25) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
copy_element on read_only object: no error, n_elements now 4
copy_element with empty destination: 'Mat::max(): object has no elements'
```

**Proposed fix.** Throw when `read_only` is set (as other modifying members should); throw `std::invalid_argument("destination is empty")`. File: `src/qd_arrayant.cpp`.

**Test for the fix session.** Catch2: both calls throw `std::invalid_argument` with a clear message.

## B-102: Arrayant documentation typos and misleading statements; latent int32 offsets and a vertex sentinel (grouped)

- Status: open
- Severity: S4, because these are typos, misleading comments or code paths that are not reachable in practice.
- Kind: doc / dead code
- Mark: [C]
- Where and what:
  1. `include/quadriga_arrayant.hpp:229`: usage example `generate_omni<float>()` should be `generate_arrayant_omni`.
  2. `include/quadriga_arrayant.hpp:107`: "0-bases".
  3. `src/qd_arrayant.cpp:210`: the MD declaration of `calc_beamwidth_deg` shows `z_point_ang` (should be `az_point_ang`).
  4. `include/quadriga_arrayant.hpp:312-316`: the `generate_speaker` comments mention driver positions and orientations that are not inputs, and a horn coverage "auto-derived from radius" (only `horn_control_freq` is; coverage defaults to 90°/60°).
  5. The multibeam MD says weights are "normalized to sum = 1"; the code does not normalise them (harmless, only ratios matter).
  6. `src/qd_arrayant_interpolate_avx2.cpp:856` and 543-548: the gather offsets are `int32` without a size check; they overflow only for more than 2^31 pattern samples (over 34 GB), so not practical. Fix: throw or fall back to generic above `INT_MAX`.
  7. `src/qd_arrayant.cpp:1046` (`export_obj_file`): `if (i_vert_final == 0)` uses 0 as "not found" although 0 is a valid match, so vertex 0 is duplicated in the OBJ; geometry stays correct. Fix: a separate `found` flag.

**Reproduce.** Read from the code.

**Proposed fix.** As listed per item.

**Test for the fix session.** None needed for 1-5; for 7 an OBJ export test can count unique vertices.

## B-103: `quantize_delays` docs: `fix_taps` 1 and 2 give `[1, 1, n_taps]` delays only for shared input

- Status: fixed (WP-03)
- Severity: S4, because only the documented output shape is incomplete.
- Kind: doc
- Mark: [R]
- Where: the `quantize_delays` MD table in `src/qd_channel_quantize_delays.cpp:145-150`.

**What happens.** Per-antenna delay input with `fix_taps = 1` and `2` gives `delay_quant` of size 2×2×6, not `[1, 1, n_taps]`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-022/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// B probe 13: quantize_delays edge cases.
// Usage: p13 <case>
//   negtiny : one delay of -1e-20 s (rounding noise below zero)
//   neg     : one delay of -5e-9 s
//   nan     : one delay NaN
//   big     : one delay of 21.4748 s (= (2^32 - 1) taps of 5 ns)
//   alias   : outputs alias the inputs (same std::vector objects)
//   shape   : per-antenna delays with fix_taps = 1 and 2 (doc says output delays [1,1,n_taps])
#include "quadriga_lib.hpp"
#include <cstdio>
#include <cmath>
int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string cs = argc > 1 ? argv[1] : "negtiny";
    std::vector<arma::cube> cre(1), cim(1), dl(1), qre, qim, qdl;
    cre[0] = arma::cube(1, 1, 3, arma::fill::ones);
    cim[0] = arma::cube(1, 1, 3, arma::fill::zeros);
    dl[0] = arma::cube(1, 1, 3);
    dl[0](0, 0, 0) = 0.0, dl[0](0, 0, 1) = 12.5e-9, dl[0](0, 0, 2) = 33.4e-9;
    if (cs == "negtiny") dl[0](0, 0, 0) = -1e-20;
    if (cs == "neg") dl[0](0, 0, 0) = -5e-9;
    if (cs == "nan") dl[0](0, 0, 0) = NAN;
    if (cs == "big") dl[0](0, 0, 0) = 21.4748;
    if (cs == "shape")
    {
        cre[0] = arma::cube(2, 2, 3, arma::fill::ones), cim[0] = arma::cube(2, 2, 3, arma::fill::zeros);
        dl[0] = arma::cube(2, 2, 3);
        for (int p = 0; p < 3; ++p) dl[0].slice(p).fill(10e-9 * p + 2e-9);
        for (int ft : {1, 2})
        {
            quadriga_lib::quantize_delays(&cre, &cim, &dl, &qre, &qim, &qdl, 5e-9, 48, 1.0, ft);
            std::printf("per-antenna input, fix_taps=%d: delay_quant size %llux%llux%llu\n", ft,
                        (unsigned long long)qdl[0].n_rows, (unsigned long long)qdl[0].n_cols, (unsigned long long)qdl[0].n_slices);
        }
        return 0;
    }
    if (cs == "alias")
    {
        std::vector<arma::cube> r0, i0, d0;
        quadriga_lib::quantize_delays(&cre, &cim, &dl, &r0, &i0, &d0, 5e-9, 48, 1.0, 0);
        quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre, &cim, &dl, 5e-9, 48, 1.0, 0);
        std::printf("separate outputs: coeff_re taps = ");
        for (auto v : r0[0]) std::printf("%g ", v);
        std::printf("\naliased outputs:  coeff_re taps = ");
        for (auto v : cre[0]) std::printf("%g ", v);
        std::printf("\n");
        return 0;
    }
    std::printf("case %s: delays = %g %g %g s\n", cs.c_str(), dl[0](0), dl[0](1), dl[0](2));
    try
    {
        quadriga_lib::quantize_delays(&cre, &cim, &dl, &qre, &qim, &qdl, 5e-9, 48, 1.0, 0);
        std::printf("returned: %llu taps, delays:", (unsigned long long)qdl[0].n_slices);
        for (auto v : qdl[0]) std::printf(" %g", v);
        std::printf("\n");
    }
    catch (const std::exception &e) { std::printf("exception: %s\n", e.what()); }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-022/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-022
export OMP_NUM_THREADS=1
H5="$(pkg-config --cflags --libs hdf5)"
REL="g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
ASAN="g++-12 -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src"
filt() { grep -E "ERROR: |WRITE of|READ of|SUMMARY|runtime error|leak of|^    #[0-9]+ .*(src/|repro)" | grep -v -E "libsanitizer| in _start" | sed -E 's/ 0x[0-9a-f]+ in .* ([^ ]+:[0-9]+)$/ \1/' | cut -c1-200 | head -${1:-8}; }
$REL "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $H5
"$D/repro" shape || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
per-antenna input, fix_taps=1: delay_quant size 2x2x6
per-antenna input, fix_taps=2: delay_quant size 2x2x6
```

**Proposed fix.** State in the table that the shape is `[1, 1, n_taps]` only for shared (`[1, 1, n_path]`) input. File: `src/qd_channel_quantize_delays.cpp`.

**Test for the fix session.** Catch2 asserting the output shape for both input kinds.

## B-104: Channel and file-I/O documentation and message nits (grouped)

- Status: partly fixed (WP-14): point 5 (closing quote of the write_png message) fixed; the other points remain (WP-17)
- Severity: S4, because these are cosmetic or minor documentation mismatches.
- Kind: doc
- Mark: [R] for items 2 and 6 (shown in the outputs of B-057 line (f) and B-056), [C] for the rest
- Where and what:
  1. `include/quadriga_channel.hpp`: the `hdf5_write` comment says the default file has 65535 slots; the code uses 65536.
  2. `obj_file_read` documentation says a trailing ".NNN" is removed from material names; the code strips everything after the first dot (`'itu_glass.blue'` maps to `itu_glass`).
  3. `qrt_file_read_raw` documentation mentions an `iR` member that does not exist.
  4. The `map_to_itu` comment says "vacuum"; the code returns "air".
  5. The `write_png` error message is missing its closing quote.
  6. The Mitsuba XML stores absolute PLY paths (`<string name="filename" value="/sjc/.../plane.ply" />`), against "distributable together"; store the path relative to the XML (`src/mitsuba_xml_file_write.cpp:~548`).
  7. `include/quadriga_tools.hpp:552` still says "fewest digits (15 to 17)" for `conf_file_write`; the sources, MEX and Python were updated to the D2 wording of the conf-file review.
  8. `qrt_file_init` throws on an RX name over 255 characters after the file has been partly written (`src/qd_channel_qrt.cpp:329-330`); validate all inputs before opening the file.
  9. `any_type_id` compares type-name pointers, which is fragile across shared-library boundaries (fine in the static builds used here).

**Reproduce.** Items 2 and 6: see B-057 and B-056. The others are read from the code.

**Proposed fix.** Documentation edits; item 6 a relative path; item 8 validation before opening the file; item 9 compare `std::type_index` or names.

**Test for the fix session.** None needed except for items 6 and 8 (a Catch2 check of the written path and of the unchanged file after the throw).

## B-105: `memcpy` with a null pointer and size 0 (formal UB) in `get_channels_irs` and in the Python list adapter

- Status: fixed (WP-01)
- Severity: S4, because the length is 0, so no memory is touched; it is undefined behaviour only by the letter of the standard and shows up as UBSan noise.
- Kind: UB
- Mark: [R]
- Where: `src/get_channels_irs.cpp:342-367` (seven `memcpy` calls; when no combination is kept, `set_size(., ., 0)` leaves `memptr() == nullptr`), seen in the phase-0 sanitizer suite from `tests/catch2_tests/test_get_channels_irs.cpp:154`; `api_python/python_arma_adapter.hpp:979` (`qd_python_copy2arma`, reached from `qd_python_list2vector_Col` at line 1294, e.g. `calc_delay_spread([np.zeros(0)], ...)`).

**What happens.** UBSan "null pointer passed as argument 1, which is declared to never be null" at the listed lines.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-20/` (create it and save the files below into it if the scratch folder is lost).


`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-20/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# C-20: get_channels_irs: memcpy(nullptr, src, 0) when no path combination is kept (test_get_channels_irs.cpp:154)
cd /sjc/quadriga-lib-fixes
flock /tmp/quadriga-fixes-suite.lock env UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0 ASAN_OPTIONS=halt_on_error=0 OMP_NUM_THREADS=1 build_asan/test_bin "Get Channels IRS - Minimal test" 2>&1 | grep -E "runtime error|test_get_channels_irs.cpp:154|passed" | cut -c1-160 | sort | uniq -c
```

Output of the phase-2 re-run (`bash run.sh`):
```text
      7     #1 0x585084a34d3e in CATCH2_INTERNAL_TEST_2 /sjc/quadriga-lib-fixes/tests/catch2_tests/test_get_channels_irs.cpp:154
      1 All tests passed (22 assertions in 1 test case)
      1 /sjc/quadriga-lib-fixes/src/get_channels_irs.cpp:342:16: runtime error: null pointer passed as argument 1, which is declared to never be null
      1 /sjc/quadriga-lib-fixes/src/get_channels_irs.cpp:343:16: runtime error: null pointer passed as argument 1, which is declared to never be null
      1 /sjc/quadriga-lib-fixes/src/get_channels_irs.cpp:344:16: runtime error: null pointer passed as argument 1, which is declared to never be null
      1 /sjc/quadriga-lib-fixes/src/get_channels_irs.cpp:349:20: runtime error: null pointer passed as argument 1, which is declared to never be null
      1 /sjc/quadriga-lib-fixes/src/get_channels_irs.cpp:355:20: runtime error: null pointer passed as argument 1, which is declared to never be null
      1 /sjc/quadriga-lib-fixes/src/get_channels_irs.cpp:361:20: runtime error: null pointer passed as argument 1, which is declared to never be null
      1 /sjc/quadriga-lib-fixes/src/get_channels_irs.cpp:367:20: runtime error: null pointer passed as argument 1, which is declared to never be null
```

Folder `build_linux/scratch/hunt/final/G-14/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-14: UB: memcpy(nullptr, src, 0) in qd_python_copy2arma for empty inputs
import numpy as np, quadriga_lib as q
try:   # an empty list item is converted with memcpy(nullptr, ..., 0) (python_arma_adapter.hpp:979) before C++ rejects it
    q.tools.calc_delay_spread([np.zeros(0)], [np.zeros(0)])
except Exception as e: print("calc_delay_spread([empty]) ->", type(e).__name__, e)
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-14/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-14
echo "--- same repro against the ASan/UBSan probe module ---"
# ASan module: build_linux/scratch/hunt/G/asan_mod (api_python/python_*.cpp unchanged, -O1 -g -fsanitize=address,undefined,
# linked against build_asan/libquadriga.a; build commands in G area report)
LD_PRELOAD="$(g++-12 -print-file-name=libasan.so) $(g++-12 -print-file-name=libubsan.so)" ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 \
  OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/build_linux/scratch/hunt/G/asan_mod python3 "$D/repro.py" 2>&1 \
  | grep -E "ERROR: |SUMMARY|runtime error|^    #[0-4] |^[a-zA-Z].*->|max" | cut -c1-200 | head -12 || true
```

Output of the phase-2 re-run (`bash run.sh`):
```text
--- same repro against the ASan/UBSan probe module ---
/sjc/quadriga-lib-fixes/api_python/python_arma_adapter.hpp:979:20: runtime error: null pointer passed as argument 1, which is declared to never be null
    #0 0x7b51fe8d1e9c in qd_python_copy2arma<double> /sjc/quadriga-lib-fixes/api_python/python_arma_adapter.hpp:979
    #1 0x7b51fe9051ab in qd_python_list2vector_Col<double> /sjc/quadriga-lib-fixes/api_python/python_arma_adapter.hpp:1294
    #2 0x7b51fe905f88 in calc_delay_spread(pybind11::list, pybind11::list, double, double) /sjc/quadriga-lib-fixes/api_python/qpy_calc_delay_spread.cpp:47
    #3 0x7b51fe93ed2a in pybind11::tuple pybind11::detail::argument_loader<pybind11::list, pybind11::list, double, double>::call_impl<pybind11::tuple, pybind11::tuple (*&)(pybind11::list, pybind11::li
    #4 0x7b51fe93ed2a in std::enable_if<!std::is_void<pybind11::tuple>::value, pybind11::tuple>::type pybind11::detail::argument_loader<pybind11::list, pybind11::list, double, double>::call<pybind11::
calc_delay_spread([empty]) -> ValueError Delay and power vectors must not be empty.
```

**Proposed fix.** Guard the copies with `if (n_bytes)` (or copy through Armadillo subviews). Files: `src/get_channels_irs.cpp`, `api_python/python_arma_adapter.hpp`.

**Test for the fix session.** The UBSan suite and an ASan build of the Python module are clean for these calls.

## B-106: Docs and code of `path_to_tube`, `qd_path` and `calc_diffraction_gain` debug output disagree (grouped)

- Status: open
- Severity: S4, because these are stale or misleading comments and a debug-only code path.
- Kind: doc
- Mark: [R] for items 1 and 2, [C] for item 3
- Where and what:
  1. `src/path_to_tube.cpp` documentation says the tube is split "at bends steeper than 10°"; the code splits only when the angle between segments is below 10°, i.e. at a deviation above 170°. Measured: deviations of 5°, 30° and 90° give no split; 165° and 175° split.
  2. `qd_path` documentation describes a "64-byte header ... first 6 interaction type codes", "(nSEG-3)/4 floats for nSEG >= 7" and an attribute `unsigned iR`; the code has `sizeof(path) == 56`, 2 inline type codes, and `(nSEG+1)/4` floats for nSEG > 2. The comment at `include/quadriga_channel.hpp:177` says `extend` increments nTRA/nREF; it does not (nREF = 0, nTRA = 0 after `extend` with types 130 and 5), which matches the MD block.
  3. `src/calc_diffraction_gain.cpp`, debug block (`verbose == 2`, `n_pos == 1`): it indexes the R-set matrices `s_orig`, `s_dest`, `fbsN`, `sbsN` with the G index `iG`; once rays have been compacted this throws an out-of-bounds exception.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/C-21/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-21: path_to_tube splits only at deviations > 170 deg, the doc says "bends steeper than 10 deg"
#include "quadriga_lib.hpp"
#include <cstdio>
#include <cmath>
int main()
{
    for (double dev : {5.0, 30.0, 90.0, 165.0, 175.0})
    {
        double a = dev * M_PI / 180.0;
        arma::mat pc(3, 3, arma::fill::zeros);
        pc(0, 1) = 10.0, pc(0, 2) = 10.0 + 10.0 * std::cos(a), pc(1, 2) = 10.0 * std::sin(a);
        arma::mat vert; arma::umat faces;
        quadriga_lib::path_to_tube<double>(&pc, &vert, &faces, 0.5, 4);
        printf("deviation %5.1f deg: %llu vertex rings (3 = no split, 4 = split)\n", dev, (unsigned long long)vert.n_cols / 4);
    }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-21/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-21
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
deviation   5.0 deg: 3 vertex rings (3 = no split, 4 = split)
deviation  30.0 deg: 3 vertex rings (3 = no split, 4 = split)
deviation  90.0 deg: 3 vertex rings (3 = no split, 4 = split)
deviation 165.0 deg: 4 vertex rings (3 = no split, 4 = split)
deviation 175.0 deg: 4 vertex rings (3 = no split, 4 = split)
```

Folder `build_linux/scratch/hunt/final/C-22/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// C-22: qd_path docs stale: header size, inline type codes; extend does not update nREF/nTRA (header comment says it does)
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    quadriga_lib::path p(0, 1), q;
    p.extend(q, 1, 0, 0, 130); // reflection type code (128-255)
    q.extend(p, 2, 0, 0, 5);   // transmission type code (1-127)
    printf("after extend with types 130 and 5: nREF = %d, nTRA = %d\n", (int)p.nREF, (int)p.nTRA);
    printf("sizeof(path) = %zu (doc: 64-byte header); interaction codes = %zu\n", sizeof(quadriga_lib::path), p.interaction_type_codes().size());
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/C-22/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/C-22
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) 2>&1 | grep -E "error"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
after extend with types 130 and 5: nREF = 0, nTRA = 0
sizeof(path) = 56 (doc: 64-byte header); interaction codes = 2
```

**Proposed fix.** Correct the comments (items 1, 2); index with the R index in the debug print (item 3).

**Test for the fix session.** None needed for 1-2; for 3 a run with `verbose = 2` and one position does not throw.

## B-107: `interp`: a NaN query returns the first data value; clamped extrapolation is undocumented

- Status: fixed (WP-13)
- Severity: S4, because a NaN query becomes a valid-looking number and the documentation does not mention clamping.
- Kind: doc / wrong result on NaN
- Mark: [R]
- Where: `src/interp.cpp:124-137` (NaN leaves index 0 with weight 1); documentation block at `:11-70`.

**What happens.** `interp_1D({0, 10, 20, 30}, {0, 1, 2, 3}, {-1, 0.5, 3.5, NaN})` gives `0 5 30 0`: NaN maps to 0, -1 and 3.5 are clamped.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-18/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-18: interp_1D returns the first data value for a NaN output sample; extrapolation is clamped (undocumented)
#include "quadriga_math.hpp"
#include <cstdio>
int main()
{
    arma::vec xi = {0, 1, 2, 3}, in = {0, 10, 20, 30}, xo = {-1, 0.5, 3.5, NAN};
    arma::vec r = quadriga_lib::interp_1D<double>(in, xi, xo);
    for (arma::uword i = 0; i < xo.n_elem; ++i)
        printf("xo = %-4g -> %g\n", xo(i), r(i));
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-18/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-18
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
xo = -1   -> 0
xo = 0.5  -> 5
xo = 3.5  -> 30
xo = nan  -> 0
```

**Proposed fix.** Document "outside the grid, the nearest end value is returned" and return NaN for a NaN query. File: `src/interp.cpp`.

**Test for the fix session.** Catch2: a NaN query gives NaN; -1 gives the first value.

## B-108: Geometry documentation and validation nits (index-map directions, 9-column check, precision claim, header typos; dead code; CPU check)

- Status: partly fixed (WP-12): point 3 (precision claim in the header) fixed; points 1, 2, 4, 5 and 6 remain (WP-17)
- Severity: S4, because these are misleading wording, a confusing error message, dead code or a robustness gap on unusual CPUs.
- Kind: doc / dead code / robustness
- Mark: [R] for items 1 and 2, [C] for the rest
- Where and what:
  1. `point_cloud_segmentation` (`src/point_cloud_segmentation.cpp:39-40`, `include/quadriga_tools.hpp:290-291, 298-299`): `forward_index` is described as a map "from points to pointsR"; measured, it gives for each `pointsR` row the original row (1-based), and `reverse_index` the opposite. The same wording for `triangle_mesh_segmentation` `mesh_index` (`src/triangle_mesh_segmentation.cpp:43`, `include/quadriga_tools.hpp:483`). The round trips themselves are correct.
  2. `ray_triangle_intersect` accepts meshes with more than 9 columns only when `aabb` is given (`src/ray_triangle_intersect.cpp:102` against the internal `triangle_mesh_aabb` at `:163`, which requires exactly 9): "10 columns, no aabb: EXCEPTION Input 'mesh' must have 9 columns", "10 columns with aabb: fbs_ind = 1". `dest` with more rows than `orig` is accepted silently.
  3. `include/quadriga_tools.hpp:427` says for `ray_triangle_intersect` "All internal computations are done using single precision"; the generic kernel supports double (the `.cpp` documentation is right).
  4. Header typos: `include/quadriga_tools.hpp:308-310` (`point_cloud_split` "Size: [n_pointsA, 9]", should be 3), `:170` (`plane` comment: z of the scale is ignored), `:195` (`icosphere` `vert` "[x1 y1 z1 x2 y2 z3 ...]"), `src/icosphere.cpp:40` ("Edge directions" should be vertex-ray directions), `src/subdivide_triangles.cpp:143, 148` (comments for the v2/v3 columns swapped; the winding itself is correct).
  5. Dead code: `src/avx_mathfun.h` (725 lines) is not included anywhere; `src/obj_overlap_test.cpp:8-11` defines `arma::mt19937_64_instance`, which neither Armadillo 14 nor 15 uses, inside a third-party namespace.
  6. `src/runtime_AVX2_Check_avx2.cpp:13-23` tests only the CPUID AVX2 bit, not FMA (CPUID.1.ECX bit 12), OSXSAVE and the XGETBV YMM state; `quadriga_lib_has_AVX2()` (`src/quadriga_lib.cpp:50-87`) then executes an FMA instruction, which would fault instead of returning false on a CPU or VM without FMA/YMM support.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/D-19/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// D-19: ray_triangle_intersect accepts a 10-column mesh only when 'aabb' is given (internal triangle_mesh_aabb needs exactly 9);
//       point_cloud_segmentation: forward_index maps pointsR->points, reverse_index maps points->pointsR (doc says the opposite)
#include "quadriga_tools.hpp"
#include <cstdio>
int main()
{
    arma::mat o = {{0, 0, 1}}, d = {{0, 0, -1}}, m(1, 10, arma::fill::zeros);
    m.row(0).cols(0, 8) = arma::rowvec({-1, -1, 0, 1, -1, 0, 0, 1, 0});
    arma::u32_vec fi;
    try { quadriga_lib::ray_triangle_intersect<double>(&o, &d, &m, nullptr, nullptr, nullptr, &fi); printf("10 columns: fbs_ind = %u\n", fi(0)); }
    catch (const std::exception &e) { printf("10 columns, no aabb: EXCEPTION %s\n", e.what()); }
    arma::mat aabb = {{-1, 1, -1, 1, 0, 0}};
    try { quadriga_lib::ray_triangle_intersect<double>(&o, &d, &m, nullptr, nullptr, nullptr, &fi, nullptr, nullptr, &aabb); printf("10 columns with aabb: fbs_ind = %u\n", fi(0)); }
    catch (const std::exception &e) { printf("10 columns with aabb: EXCEPTION %s\n", e.what()); }

    arma::mat p = {{3, 0, 0}, {0, 0, 0}, {2, 0, 0}, {1, 0, 0}}, pR;
    arma::u32_vec sci, fwd, rev;
    quadriga_lib::point_cloud_segmentation(&p, &pR, &sci, 1, 1, &fwd, &rev);
    printf("points x  :"); for (arma::uword i = 0; i < p.n_rows; ++i) printf(" %g", p(i, 0)); printf("\n");
    printf("pointsR x :"); for (arma::uword i = 0; i < pR.n_rows; ++i) printf(" %g", pR(i, 0)); printf("\n");
    printf("forward_index (len %llu):", (unsigned long long)fwd.n_elem); for (auto v : fwd) printf(" %u", v);
    printf("   -> pointsR row i = points row forward_index(i)-1\n");
    printf("reverse_index (len %llu):", (unsigned long long)rev.n_elem); for (auto v : rev) printf(" %u", v);
    printf("   -> points row j = pointsR row reverse_index(j)\n");
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/D-19/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/D-19
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
g++-12 -std=c++17 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -O2 "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
echo "== AVX2 library (build_linux)"
"$D/repro"  || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== AVX2 library (build_linux)
10 columns, no aabb: EXCEPTION Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.
10 columns with aabb: fbs_ind = 1
points x  : 3 0 2 1
pointsR x : 0 1 2 3
forward_index (len 4): 2 4 3 1   -> pointsR row i = points row forward_index(i)-1
reverse_index (len 4): 3 0 2 1   -> points row j = pointsR row reverse_index(j)
```

**Proposed fix.** Rewrite the wording (e.g. "`forward_index(i)`: 1-based row in `points` of `pointsR` row i (0 = padding)"); make the column check in `ray_triangle_intersect` consistent (require exactly 9, or pass a 9-column view to `triangle_mesh_aabb`); fix the typos; delete the dead code; add the FMA/OSXSAVE/XGETBV checks (or `__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")`).

**Test for the fix session.** For item 2: a 10-column mesh without `aabb` either throws clearly or gives the correct result, consistently.

## B-109: `sos_val` / `sos_remove_bias`: an output that partially overlaps an input is accepted and gives wrong values (docs promise an error)

- Status: fixed (WP-16): sos_val rejects a val that shares any part of an input's memory; sos_remove_bias rejects any written argument that shares memory with another argument, and offset == mean_before; checked before anything is written
- Severity: S4, because it needs a hand-built Armadillo matrix on foreign memory that starts inside an input, which no wrapper can produce; the documentation promises an exception, the call silently returns values off by up to 0.92.
- Kind: doc / invalid input accepted
- Mark: [R]
- Where: `src/sos_val.cpp:390-395` (the check compares start addresses only), documentation `src/sos_val.cpp:304-305` ("`val` cannot be the same object as an input, nor use the memory of one ... Such a call throws"); same gap in `src/sos_remove_bias.cpp:113` (`phase_b == &phase`, object identity only; there the documentation only says "not the same object"). Follow-up of N3 and B9 of `md/REVIEW_QD_SOS_V2.md` / `md/REVIEW_QD_SOS.md`, which are fixed for the identical-object case.

**What happens.** `val` is a strict matrix on column 1 (y) of `pos`, two devices on the same map: no exception, max |val - correct| = 0.92 (the first kernel overwrites y, which the normalisation then reads). With one device the result is correct. `sos_remove_bias` with `phase_b` a matrix on the memory of `phase` (two maps): device a keeps a mean of -4.3e-2, device b -1.1e-16.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/E-1/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// E-1: sos_val / sos_remove_bias accept an output that partially shares the memory of an input (doc says it throws)
#include "quadriga_sos.hpp"
#include <cstdio>
#include <stdexcept>
int main()
{
    using namespace quadriga_lib;
    arma::mat freq, phase;
    arma::vec acf, dist;
    sos_preset<double>("Comb128", freq, &acf, &dist, 10.0);
    sos_init_phases<double>(freq.n_rows, 1, phase, 1);
    arma::arma_rng::set_seed(3);
    arma::mat pos = 100.0 * arma::randu<arma::mat>(40, 3), pos_b = pos + 1.0;

    // (1) sos_val, two devices on the same map, val is a strict matrix on column 1 (y) of pos
    arma::mat ref, p = pos;
    sos_val<double>(freq, phase, pos, ref, &pos_b, nullptr, &acf, &dist, 1.0, false, 1);
    try
    {
        arma::mat v(p.colptr(1), 40, 1, false, true);
        sos_val<double>(freq, phase, p, v, &pos_b, nullptr, &acf, &dist, 1.0, false, 1);
        std::printf("(1) sos_val: no exception, max |val - correct| = %.3e\n", arma::abs(p.col(1) - ref.col(0)).max());
    }
    catch (const std::exception &e) { std::printf("(1) sos_val: %s\n", e.what()); }

    // (2) sos_remove_bias, phase_b is a second matrix on the memory of phase (two maps)
    arma::mat ph = phase;
    arma::mat ph_b(ph.memptr(), ph.n_rows, ph.n_cols, false, true);
    try
    {
        sos_remove_bias<double>(freq, ph, pos, &pos_b, &ph_b);
        arma::mat va, vb;
        sos_val<double>(freq, ph, pos, va, nullptr, nullptr, nullptr, nullptr, 1.0, false, 1);
        sos_val<double>(freq, ph, pos_b, vb, nullptr, nullptr, nullptr, nullptr, 1.0, false, 1);
        std::printf("(2) sos_remove_bias: no exception, mean after: device a %.3e, device b %.3e\n", arma::mean(va.col(0)), arma::mean(vb.col(0)));
    }
    catch (const std::exception &e) { std::printf("(2) sos_remove_bias: %s\n", e.what()); }
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/E-1/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# E-1: partial memory overlap of an output with an input is not detected by sos_val / sos_remove_bias
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/E-1
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5)
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
(1) sos_val: no exception, max |val - correct| = 9.227e-01
(2) sos_remove_bias: no exception, mean after: device a -4.304e-02, device b -1.110e-16
```

**Proposed fix.** Replace the start-address test with a range-overlap test of `[memptr, memptr + n_elem)` of each input against `val`; the same for `phase_b` against `phase`. Smaller alternative: change the doc sentence to "nor start at the memory of one". Files: `src/sos_val.cpp`, `src/sos_remove_bias.cpp`.

**Test for the fix session.** Catch2 in `test_sos_val.cpp`: the strict matrix on `pos.colptr(1)` as `val` with two devices throws `std::invalid_argument` and leaves `pos` unchanged; in `test_sos_remove_bias.cpp`: `phase_b` on `phase.memptr()` throws, phases unchanged.

## B-110: `sos_randn` / `sos_rand` / `sos_randi` report an unknown preset as "Input 'name' ...", an argument they do not have

- Status: fixed (WP-16): the random functions check the preset name with the shared lookup of sos_preset and report "Input 'preset' ..."
- Severity: S4, because only the error message names the wrong argument; the call is rejected correctly in all three APIs.
- Kind: doc (error message)
- Mark: [R]
- Where: the throw "Input 'name' is not a known preset." in `src/sos_preset.cpp`, reached from `qd_sos_rand_impl` in `src/sos_rand.cpp`.

**What happens.** MEX: "sos_randn: Input 'name' is not a known preset." (also `sos_rand`, `sos_randi`); Python: `ValueError: Input 'name' is not a known preset.`; expected "Input 'preset' ...".

**Reproduce.**

Folder `build_linux/scratch/hunt/final/E-3/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% E-3: sos_randn/rand/randi report an unknown preset as "Input 'name' ...", an argument they do not have
addpath('/sjc/quadriga-lib-fixes');
try, quadriga_lib.sos_randn(10, [0 0 0], [], [], 'Comb100', 1); catch e, disp(e.message); end
try, quadriga_lib.sos_rand(10, [0 0 0], [], [], 'Comb100', 1); catch e, disp(e.message); end
try, quadriga_lib.sos_randi(10, [0 0 0], 6, [], [], 'Comb100', 1); catch e, disp(e.message); end
```

`repro.py`:
```python
# E-3: same message in Python
import numpy as np
import quadriga_lib
try:
    quadriga_lib.sos.randn(np.array([10.0]), np.zeros((1, 3)), preset="Comb100", seed=1)
except Exception as e:
    print("python sos.randn:", type(e).__name__, e)
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/E-3/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# E-3: unknown preset in the random functions is reported as argument 'name' (Octave and Python)
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/E-3
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')") 2>&1 | grep -v "qt.qpa" || echo "exit code $?"
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
sos_randn: Input 'name' is not a known preset.
sos_rand: Input 'name' is not a known preset.
sos_randi: Input 'name' is not a known preset.
python sos.randn: ValueError Input 'name' is not a known preset.
```

**Proposed fix.** In `qd_sos_rand_impl` catch the exception of `sos_preset` and rethrow "Input 'preset' is not a known preset." (or check the name before the call). File: `src/sos_rand.cpp`.

**Test for the fix session.** Catch2 in `test_sos_rand.cpp`: `CHECK_THROWS_WITH(sos_randn<double>(..., "Comb100", 1), ContainsSubstring("'preset'"))`.

## B-111: `md/SPEC_PYTHON_API.md` gives two contradicting rules for the headings of the `sos` entries

- Status: fixed (WP-16)
- Severity: S4, because only the spec is inconsistent; the code follows the later rule.
- Kind: doc
- Mark: [C] (checked with `sed -n 104p md/SPEC_PYTHON_API.md; sed -n 348p md/SPEC_PYTHON_API.md` and `grep -c "^# sos\." api_python/qpy_sos_*.cpp`, one per file)
- Where: `md/SPEC_PYTHON_API.md:104` (added in 38d871a: the heading "keeps the prefix (`# sos_<name>`, linked as `[[sos_<name>]]`)") against `:348` (added in 450bbdc: `# sos.val`, links `[[sos.val]]`). All 14 `api_python/qpy_sos_*.cpp` use `# sos.<name>`.

**What happens.** A wrapper written from line 104 would get the old heading and broken links.

**Proposed fix.** Replace the second sentence of line 104 by a pointer to the rule at line 348. File: `md/SPEC_PYTHON_API.md`.

**Test for the fix session.** None (documentation).

## B-112: MEX `qrt_file_read` returns `interact_type` as uint8; the doc says uint32

- Status: fixed (WP-01)
- Severity: S4, because only the documented type is wrong.
- Kind: doc
- Mark: [R]
- Where: documentation in `api_mex/qrt_file_read.cpp`; code at `:243` uses `mxUINT8_CLASS`.

**What happens.** `class(interact_type{1})` is `uint8`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-17/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-17: qrt_file_read interact_type is uint8; the doc says uint32
addpath('/sjc/quadriga-lib-fixes');
out = cell(1, 18);
[out{:}] = quadriga_lib.qrt_file_read('/sjc/quadriga-lib-fixes/tests/data/test_v5.qrt', 1);
fprintf('class(interact_type{1}) = %s (doc: uint32)\n', class(out{18}{1}));
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-17/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-17: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-17)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
class(interact_type{1}) = uint8 (doc: uint32)
```

**Proposed fix.** Change "uint32" to "uint8" in the documentation. File: `api_mex/qrt_file_read.cpp`.

**Test for the fix session.** MOxUnit asserting `uint8`.

## B-113: MEX `get_channels_*` with 0 paths and `add_fake_los_path` return a zero path with delay 0, where C++ throws

- Status: fixed (WP-15): C++ returns the zero-power LOS path with its LOS delays for 0 paths with add_fake_los_path, the wrappers call C++; round 2: 0 paths without it gives empty outputs [n_rx, n_tx, 0] in all APIs, and Python spherical batch mode adds the fake LOS path too
- Severity: S4, because it is an error-behaviour difference on an edge case (and the returned delay ignores `use_absolute_delays`).
- Kind: cross-API
- Mark: [R]
- Where: `api_mex/get_channels_planar.cpp:101-102`, `api_mex/get_channels_spherical.cpp:111-112`, `api_mex/get_channels_multifreq.cpp:105-106` skip the C++ call when only the fake path would exist.

**What happens.** MATLAB returns `coeff = 0, delay = 0` (with `use_absolute_delays = true` the LOS delay would be 3.34e-8 s); C++ `get_channels_planar` throws "Inputs 'aod', ... must have the same number of columns (n_paths)." and `get_channels_spherical` "Missing data for ... fbs_pos ...".

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-19/` (create it and save the files below into it if the scratch folder is lost).


`repro.cpp`:
```cpp
// F-19: the same calls in C++
#include "quadriga_lib.hpp"
#include <cstdio>
int main()
{
    auto ant = quadriga_lib::generate_arrayant_omni<double>(10.0);
    arma::vec e; arma::mat M0(8, 0), f0(3, 0); arma::cube cr, ci, dl;
    try { quadriga_lib::get_channels_planar<double>(&ant, &ant, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, &e, &e, &e, &e, &e, &e, &M0, &cr, &ci, &dl, 1e9, true, true); printf("C++ planar returned\n"); }
    catch (const std::exception &ex) { printf("C++ planar    -> %s\n", ex.what()); }
    try { quadriga_lib::get_channels_spherical<double>(&ant, &ant, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, &f0, &f0, &e, &e, &M0, &cr, &ci, &dl, 1e9, true, true); printf("C++ spherical returned\n"); }
    catch (const std::exception &ex) { printf("C++ spherical -> %s\n", ex.what()); }
    return 0;
}
```

`repro.m`:
```matlab
% F-19: 0 paths + add_fake_los_path: MATLAB returns a zero path (delay 0 even with absolute delays); C++ throws
addpath('/sjc/quadriga-lib-fixes');
ant = quadriga_lib.arrayant_generate('omni');
e = zeros(0, 1);
[cr, ci, dl] = quadriga_lib.get_channels_planar(ant, ant, e, e, e, e, e, e, zeros(8,0), [0;0;0], [0;0;0], [10;0;0], [0;0;0], 1e9, true, true);
fprintf('MATLAB planar   0 paths + fake LOS, abs delays: coeff=%g delay=%g (LOS delay would be %g)\n', cr, dl, 10/299792458);
[cr, ci, dl] = quadriga_lib.get_channels_spherical(ant, ant, zeros(3,0), zeros(3,0), e, e, zeros(8,0), [0;0;0], [0;0;0], [10;0;0], [0;0;0], 1e9, true, true);
fprintf('MATLAB spherical 0 paths + fake LOS, abs delays: coeff=%g delay=%g\n', cr, dl);
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-19/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-19: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-19)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== C++ (build_linux/libquadriga.a)"
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
(cd "$D" && OMP_NUM_THREADS=1 ./repro) || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
MATLAB planar   0 paths + fake LOS, abs delays: coeff=0 delay=0 (LOS delay would be 3.33564e-08)
MATLAB spherical 0 paths + fake LOS, abs delays: coeff=0 delay=0
== C++ (build_linux/libquadriga.a)
C++ planar    -> Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length', and 'M' must have the same number of columns (n_paths).
C++ spherical -> Missing data for any of: fbs_pos, lbs_pos, path_gain, path_length, M
```

**Proposed fix.** Call C++ unconditionally and let it decide, or make C++ return the fake LOS path for 0 paths (then all APIs agree); document the behaviour. Files: the three MEX wrappers (and the C++ functions if they change).

**Test for the fix session.** MOxUnit and Catch2 pin the same behaviour for 0 paths with `add_fake_los_path`.

## B-114: MEX `arrayant_rotate_pattern`: struct arrays reject usage 3 and 4 (Python maps them); a call without output argument errors

- Status: fixed (WP-15)
- Severity: S4, because it is an API inconsistency on a less used path.
- Kind: cross-API
- Mark: [R]
- Where: `api_mex/arrayant_rotate_pattern.cpp:106` (passes `usage` unmapped to `arrayant_rotate_pattern_multi`, while Python maps 3 to 0 and 4 to 1), `:165` (errors when `nlhs == 0`).

**What happens.** MATLAB, multi-frequency struct array with `usage = 3`: "arrayant_rotate_pattern_multi: Usage must be 0, 1, or 2."; Python accepts it. A call without an output: "Wrong number of output arguments."

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-20/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-20: arrayant_rotate_pattern: struct array rejects usage 3 (Python maps 3->0); call without output errors
addpath('/sjc/quadriga-lib-fixes');
xp = quadriga_lib.arrayant_generate('xpol');
xm = [xp, xp]; xm(2).center_freq = 2e9;
try, r = quadriga_lib.arrayant_rotate_pattern(xm, 10, 0, 0, 3); disp('MATLAB multi usage=3: OK');
catch e, fprintf('MATLAB multi usage=3 -> %s\n', e.message); end
try, quadriga_lib.arrayant_rotate_pattern(xp, 10); disp('MATLAB nlhs=0: OK');
catch e, fprintf('MATLAB nlhs=0 -> %s\n', e.message); end
```

`repro.py`:
```python
# F-20: Python accepts usage=3 for a frequency-dependent (multi-entry) arrayant
import numpy as np, quadriga_lib as q
xp = q.arrayant.generate('xpol')
xm = {k: (np.stack([v, v], axis=-1) if isinstance(v, np.ndarray) and k.startswith('e_') else v) for k, v in xp.items()}
xm['center_freq'] = np.array([299792458.0, 2e9])
try:
    r = q.arrayant.rotate_pattern(xm, x_deg=10, usage=3); print('Python multi usage=3: OK, e_theta_re shape', np.shape(r['e_theta_re']))
except Exception as e: print('Python multi usage=3 ->', e)
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-20/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-20: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-20)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== Python"
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
MATLAB multi usage=3 -> arrayant_rotate_pattern: arrayant_rotate_pattern_multi: Usage must be 0, 1, or 2.
MATLAB nlhs=0 -> arrayant_rotate_pattern: Wrong number of output arguments.
== Python
Python multi usage=3: OK, e_theta_re shape (19, 37, 2, 2)
```

**Proposed fix.** Map usage 3/4 as Python does (or reject them in Python as well), and allow `nlhs == 0` (return the struct as `ans`). File: `api_mex/arrayant_rotate_pattern.cpp`.

**Test for the fix session.** MOxUnit: usage 3 on a struct array equals the Python result.

## B-115: MEX documentation, error IDs and latent helper defects (grouped)

- Status: open
- Severity: S4, because these are documentation errors, a non-standard error ID, typos and latent defects without a current caller.
- Kind: doc / dead code
- Mark: [R] for items a-e (in the repro), [C] for the rest
- Where and what:
  - (a) `api_mex/version.cpp:34-40` uses the error IDs `quadriga_lib:version:no_input` / `no_output` instead of `quadriga_lib:CPPerror`.
  - (b) `arrayant_generate` documentation says "Separate outputs (must request exactly 11)", but 2 to 11 outputs work.
  - (c) `arrayant_qdant_read` with 3 outputs assigns nothing ("element number 1 undefined in return list"); neither branch covers `nlhs == 3`.
  - (d) `hdf5_write_dset` documentation says "Row vectors are stored as column vectors", but `[1 2 3]` is stored as 1×3 (Python reads shape (1, 3)).
  - (e) `interp` 2-D call without `yI` silently interpolates one row at y = 0 (`interp([0 1], [0 1], [0 1; 2 3], 0.5)` returns 0.5).
  - `point_inside_mesh` documentation for `obj_ind` lacks "1-based".
  - `calc_diffraction_gain` mandatory `center_freq = []` silently becomes 1e9; `generate_diffraction_paths` `lod = []` becomes 1.
  - `qd_mex_get_Col<...>(prhs[i], "i_element")` passes a string literal as the `bool copy` flag (works by accident): `api_mex/arrayant_calc_beamwidth.cpp:110, 112`, `api_mex/arrayant_calc_directivity.cpp:96, 98`, `api_mex/arrayant_export_obj_file.cpp:60`; the latter's error message calls the index `freq` where the documentation says `i_freq`.
  - Latent helper defects (no current caller): `qd_mex_copy2matlab` (Col/Mat/Cube) with `ns < n_elem` and `is == nullptr` would overflow the output; `qd_mex_vector2matlab` with `ns > input->size()` and `is == nullptr` would read past the input; `qd_mex_vector2matlab(std::vector<Cube>)` lacks the `int` branch; non-inline functions in a header (`qd_mex_matlab2vector_Bool`, `qd_mex_set_field`, `qd_mex_has_field`).
  - `CMakeLists.txt:196` says MATLAB requires static HDF5, but the shared library is linked (see B-029).
  - Typos: "Varialbe", "slit-mode" (`arrayant_generate` / `arrayant_combine_pattern` docs), "Forsingle" (`fast_sincos`), "exisits" / "aeisting" / "trow" (`api_mex/arrayant_qdant_write.cpp:123-124`), "Rrow" (`point_cloud_aabb`), "gnerate" (`ray_point_intersect`), "mutiple" (`interp`), "exisiting" (`hdf5_reshape_layout`).

**Reproduce.**

Folder `build_linux/scratch/hunt/final/F-21/` (create it and save the files below into it if the scratch folder is lost).


`repro.m`:
```matlab
% F-21: small doc/SPEC items ([R] parts only)
addpath('/sjc/quadriga-lib-fixes');
try, quadriga_lib.version(1); catch e, fprintf('(a) version(1) error identifier = %s (SPEC: quadriga_lib:CPPerror)\n', e.identifier); end
[a, b] = quadriga_lib.arrayant_generate('omni');
fprintf('(b) arrayant_generate with 2 outputs works: %s %s (doc: "must request exactly 11")\n', mat2str(size(a)), mat2str(size(b)));
fn = fullfile(pwd, 'f21.qdant'); if exist(fn, 'file'), delete(fn); end
quadriga_lib.arrayant_qdant_write(fn, quadriga_lib.arrayant_generate('omni'), 1);
try, [o1, o2, o3] = quadriga_lib.arrayant_qdant_read(fn, 1); disp('(c) qdant_read 3 outputs: OK');
catch e, fprintf('(c) arrayant_qdant_read with 3 outputs -> %s\n', e.message); end
delete(fn);
h5 = fullfile(pwd, 'row.hdf5'); if exist(h5, 'file'), delete(h5); end
quadriga_lib.hdf5_create_file(h5); quadriga_lib.hdf5_write_dset(h5, 1, 'row', [1 2 3]);
disp('(d) wrote [1 2 3] to row.hdf5 (doc: "Row vectors are stored as column vectors"); see repro.py');
v = quadriga_lib.interp([0 1], [0 1], [0 1; 2 3], 0.5);
fprintf('(e) interp 2-D call without yI accepted: result %s (one row at y = 0)\n', mat2str(v));
```

`repro.py`:
```python
# F-21 (d): shape of the dataset written by repro.m from a MATLAB row vector
import os, numpy as np, quadriga_lib as q
r = q.channel.hdf5_read_dset(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'row.hdf5'), name='row')
print('(d) Python reads the MATLAB row vector dataset as shape', np.shape(r), '(doc says stored as column)')
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/F-21/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# F-21: run from anywhere; bash run.sh > out.txt 2>&1
cd /sjc/quadriga-lib-fixes
D=$(realpath build_linux/scratch/hunt/final/F-21)
echo "== Octave 11.1"
(cd "$D" && OMP_NUM_THREADS=1 octave --no-gui --eval "run('repro.m')" 2>&1 | grep -v qt.qpa) || echo "exit code $?"
echo "== Python"
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
== Octave 11.1
(a) version(1) error identifier = quadriga_lib:version:no_input (SPEC: quadriga_lib:CPPerror)
(b) arrayant_generate with 2 outputs works: [19 37] [19 37] (doc: "must request exactly 11")
(c) arrayant_qdant_read with 3 outputs -> element number 1 undefined in return list
(d) wrote [1 2 3] to row.hdf5 (doc: "Row vectors are stored as column vectors"); see repro.py
(e) interp 2-D call without yI accepted: result 0.5 (one row at y = 0)
== Python
(d) Python reads the MATLAB row vector dataset as shape (1, 3) (doc says stored as column)
```

**Proposed fix.** As listed per item.

**Test for the fix session.** MOxUnit for (a) (error ID) and (c) (3 outputs); the rest are documentation.

## B-116: Python `hdf5_create_file` returns uint32 although the wrapper builds int64 (confirms item 1.3 of the Python fix plan)

- Status: fixed (WP-05)
- Severity: S4, because only the dtype of a correct result is affected.
- Kind: doc / cross-API (dtype)
- Mark: [R]
- Where: `api_python/qpy_hdf5_create_file.cpp:42` (declared `py::array_t<unsigned>`, returns `qd_python_copy2numpy<unsigned, py::ssize_t>(...)`). Item 1.3 "(verify)" of `md/python_fix_plan_v00.md`; this is the first measurement.

**What happens.** `hdf5_create_file(fn, 2)` returns `[2 1 1 1]` with dtype `uint32`; pybind11 converts the int64 array back to the declared type.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/G-12/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-12: hdf5_create_file returns uint32 instead of the built int64 (plan item 1.3)
import os, quadriga_lib as q
D = os.path.dirname(os.path.abspath(__file__)); fn = os.path.join(D, 'g12.hdf5')
if os.path.exists(fn): os.remove(fn)
s = q.channel.hdf5_create_file(fn, 2, 1, 1, 1)
print("hdf5_create_file returns", s, s.dtype, "(wrapper builds int64; declared return type py::array_t<unsigned>)")
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-12/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-12
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
hdf5_create_file returns [2 1 1 1] uint32 (wrapper builds int64; declared return type py::array_t<unsigned>)
```

**Cause / fix.** Declared return type differs from the produced one; make them match (int64 per the spec rule for copied integer outputs, see plan items 1.3 and 4.1). File: `api_python/qpy_hdf5_create_file.cpp`.

**Test for the fix session.** pytest asserts the documented dtype of the return value.

## B-117: Python output dtypes and shapes differ from the docs (`qdant_read`, `point_inside_mesh`, `generate_speaker`, `get_ieee_indoor`)

- Status: documented (WP-05): the docs give the produced dtypes and shapes; `generate_speaker` with one frequency keeps 3-D fields
- Severity: S4, because only documentation is wrong; the values are right.
- Kind: doc
- Mark: [R]
- Where: `api_python/qpy_arrayant_qdant_read.cpp:73` (layout), `api_python/qpy_point_inside_mesh.cpp` (docs), `api_python/qpy_generate_speaker.cpp` and `api_python/python_quadriga_adapter.hpp` (`arrayant2dict_multi` collapses a single frequency), `api_python/qpy_get_channels_ieee_indoor.cpp` (docs).

**What happens.** `qdant_read` `layout` is int64 (docs: uint32); `point_inside_mesh` returns int64 (docs: uint32); `generate_speaker(frequencies=[1000.0])` returns 3-D fields and a scalar `center_freq` (docs: 4-D and `(n_freq,)`); `get_ieee_indoor` returns `center_frequency` as an array of shape `(1,)` (docs: scalar) and adds a key `initial_position` (docs: "only keys listed").

**Reproduce.**

Folder `build_linux/scratch/hunt/final/G-13/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-13: output type / shape docs disagree with the code
import numpy as np, os, quadriga_lib as q
A, C, R = q.arrayant, q.channel, q.RTtools
D = os.path.dirname(os.path.abspath(__file__)); fn = os.path.join(D, 'g13.qdant')
if os.path.exists(fn): os.remove(fn)
A.qdant_write(fn, A.generate('omni'))
print("qdant_read layout dtype:", A.qdant_read(fn)['layout'].dtype, "(doc: uint32)")
print("point_inside_mesh dtype:", R.point_inside_mesh(np.zeros((1, 3)), R.cube()).dtype, "(doc: uint32)")
s = A.generate_speaker(frequencies=np.array([1000.0]))
print("generate_speaker 1 freq: e_theta_re", s['e_theta_re'].shape, "center_freq", repr(s['center_freq']), "(doc: 4D, (n_freq,))")
c = C.get_ieee_indoor(A.generate('omni'), A.generate('omni'), 'B', seed=1)[0]
print("get_ieee_indoor center_frequency:", repr(c['center_frequency']), "(doc: scalar); extra key initial_position:", 'initial_position' in c)
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-13/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-13
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
qdant_read layout dtype: int64 (doc: uint32)
point_inside_mesh dtype: int64 (doc: uint32)
generate_speaker 1 freq: e_theta_re (37, 73, 1) center_freq 1000.0 (doc: 4D, (n_freq,))
get_ieee_indoor center_frequency: array([5.25e+09]) (doc: scalar); extra key initial_position: True
```

**Proposed fix.** Correct the docs; for `generate_speaker` decide whether a single frequency keeps the 4-D form (consistency for callers that loop over frequencies). Files: the four `qpy_*.cpp` comment blocks.

**Test for the fix session.** pytest asserting the documented dtypes and shapes (after the docs are corrected, they pin the behaviour).

## B-118: Python single-frequency arrayant dict: a 1-element `center_freq` array gives a numpy DeprecationWarning, 2 elements an unclear error

- Status: fixed (WP-05)
- Severity: S4, because it is a warning today (an error in a future numpy) and an unclear message for wrong input.
- Kind: cross-API
- Mark: [R]
- Where: `api_python/python_quadriga_adapter.hpp:153` (`arrayant["center_freq"].cast<double>()`); used by `get_channels_planar`, `get_channels_spherical`, `get_ieee_indoor` and `arrayant.generate` (custom pattern).

**What happens.** `center_freq=np.array([2e9])` emits "DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated"; a 2-element array gives `RuntimeError: Unable to cast Python instance of type <class 'numpy.ndarray'> to C++ type 'double'`. The multi-frequency path accepts arrays.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/G-15/` (create it and save the files below into it if the scratch folder is lost).


`repro.py`:
```python
# G-15: single-frequency dict path: 1-element center_freq array -> numpy DeprecationWarning
import numpy as np, warnings, quadriga_lib as q
A = q.arrayant
o = A.generate('omni'); o['center_freq'] = np.array([2e9])
args = (np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1), np.ones(1) * 10, np.array([[1.0, 0, 0, 0, 0, 0, 1, 0]]).T,
        np.zeros(3), np.zeros(3), np.array([10.0, 0, 0]), np.zeros(3))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    A.get_channels_planar(o, o, *args)
    print("planar, center_freq=array([2e9]):", [f"{x.category.__name__}: {str(x.message)[:70]}" for x in w][:1])
print("interpolate (multi path) accepts it:", A.interpolate(o, np.array([[0.0]]), np.array([[0.0]]))[0].ravel())
o['center_freq'] = np.array([2e9, 3e9])
try: A.get_channels_planar(o, o, *args)
except Exception as e: print("planar, 2-element center_freq:", type(e).__name__, e)
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/G-15/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/G-15
OMP_NUM_THREADS=1 PYTHONPATH=/sjc/quadriga-lib-fixes/lib python3 "$D/repro.py" || echo "exit code $?"
```

Output of the phase-2 re-run (`bash run.sh`):
```text
planar, center_freq=array([2e9]): ['DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and wi']
interpolate (multi path) accepts it: [1.]
planar, 2-element center_freq: RuntimeError Unable to cast Python instance of type <class 'numpy.ndarray'> to C++ type 'double'
```

**Proposed fix.** Read `center_freq` through `qd_python_numpy2arma_Col` and require exactly one element (clear `ValueError` otherwise). File: `api_python/python_quadriga_adapter.hpp`.

**Test for the fix session.** pytest with `warnings` as errors: a 1-element array is accepted without a warning; a 2-element array raises `ValueError`.

## Appendix: Seen in excluded code

- `ray_progress`: UBSan "reference binding to null pointer" in `arma::Mat<float>::colptr` called from
  `quadriga_lib::ray_progress` (5 reports, `tests/catch2_tests/test_wave_progress.cpp:301`), in every sanitizer suite run.
- `tests/mex_tests/test_ray_mesh_interact.m:113` fails in Octave 11.1, MATLAB R2024a and R2026a (0.98485 vs 0.69639),
  as recorded by the earlier reviews.
- `point_inside_mesh` decides inside/outside from `fbs_angle` and `thickness` of `ray_mesh_interact`; not analysed
  (B-027 is traced to the ray–triangle kernel only).
- The physics of `ray_mesh_interact` / `ray_state_update` used by `calc_diffraction_gain` was not analysed.

## Appendix: Checked and found correct

Arrayant (A):
- AVX2 against generic interpolation over 7 pattern types, float/double, n_out {1, 2, 3, 7, 8, 9, 15, 17}, n_ang
  {1, 3, 8, 9, 33, 257}, all combinations of per-element angles and orientations, azimuths ±π and ±3π: max field
  difference 7e-5 (float, coarse random grid) and 2e-7 (double); local angles differ up to 3e-4 rad only next to the
  poles. Non-uniform grids 6e-7.
- 1 against 8 threads bitwise identical for both kernels and all three AVX2 code paths; OpenMP clauses correct.
- Angles far outside ±π: generic exact; AVX2 drifts only beyond about 6000 rad (documented range is [-π, π]).
- qdant round trip of names with XML special characters and newlines, 17-digit centre frequency, grids to 1e-14 rad,
  layouts, overwriting one ID, multi-entry files; clean under ASan (apart from B-012, B-038).
- `combine_pattern` against the analytic array factor: 1.9e-15. Multi-frequency functions equal their single-frequency
  counterparts (bitwise or to 8e-15). 3GPP down-tilt and multibeam steering point where documented.

Channel and file I/O (B):
- QRT cache path and non-cache path identical on every origin, CIR and direction of `tests/data/test.qrt` (v4) and
  `test_v5.qrt`; `qrt_file_init` + `append` + `read_raw` round trip exact for v5 and v6 with several TX, including an
  empty one; `read_raw` rejects `no_freq > 127`.
- HDF5 round trips of valid multi-snapshot channels (float, double, an empty middle snapshot, every `par` type
  including 0×3 and 2×3×0) correct apart from B-050; `is_valid` rejects names over 255 characters.
- Conf-file items B1, B2, Q1 and Q3 of `md/REVIEW_CONF_FILE.md` still resolved (no regression).
- `html_docu/` is current: regenerating the HTML and MD API documents into a scratch folder gives identical files
  except the date line of the three MD files.

Channel generation and tools (C):
- `get_channels_spherical` LOS against a hand-built `exp(-jk·d)` per element pair: 3.0e-12; delays 4e-22 s; spherical
  against planar in the far field (850 m): 5.0e-4 (LOS, aperture effect) and 6.5e-6 (NLOS); reciprocity 1.6e-16.
- `use_avx2` true against false in `get_channels_spherical`: at most 7.5e-7 relative for 4-80 links, float and double,
  up to 100 km. `baseband_freq_response` AVX2 against generic for delays up to 10 ms: at most 1.2e-6 (carriers
  {1, 7, 8, 13, 64}, paths {0, 1, 5, 9}); NaN/Inf propagate identically; 0 paths give zeros.
- 1 against 8 threads bitwise identical (spherical, IEEE indoor with a fixed seed over 51 snapshots).
- `coord2path` angle conventions with `reverse_path`; diffraction weights sum to 1 (lod 1-4); the acoustic scale of
  the wrappers (3e8/343) equals the library's to 9e-10.

Geometry and math (D):
- `ray_triangle_intersect` AVX2 against generic on random scenes (n_ray, n_mesh in {1, 7, 8, 9, 15, 17, 63, 1023},
  sub-meshes, float/double): no disagreement in indices or counts, FBS within 1e-4; stacks of 1-40 triangles;
  degenerate triangles, zero-length and parallel rays, rays starting or ending on a face, NaN/Inf: consistent.
- `ray_point_intersect` AVX2 against generic (n_points up to 5000, aligned segmentation): identical; ray order
  deterministic for 1-64 threads.
- Accuracy against long-double references: `fast_sincos` on [-1e6, 1e6] within 1.3e-7, `fast_asin`/`fast_acos`
  1.7e-7 / 3.0e-7, `fast_atan2` 2.5e-7; tails for n = 0, 1, 7, 8, 9, 15, 17, 1023 clean under ASan.
- `calc_rotation_matrix` orthonormal to 4.4e-16 with det 1 and the documented conventions; `icosphere` face count,
  radius and normals; segmentation round trips (both functions) and padding with 0 in C++; `bits.hpp` under UBSan.
- `obj_overlap_test` with touching and overlapping axis-aligned cubes at offsets up to 5e6: correct in both builds.

Sum of sinusoids (E):
- Every resolved item of `md/REVIEW_QD_SOS.md` and `md/REVIEW_QD_SOS_V2.md` (B1-B10, N3, N4) behaves as fixed at HEAD
  in C++, Octave 11.1, MATLAB R2026a and Python; no SOS source changed after the second review.
- AVX2 against generic against a long-double oracle (2090 cases): generic within 4.6e-14, AVX2 within the documented
  bound in every case; the generic output is bit-identical between the AVX2 and the generic library.
- 37 output hashes identical for 1 and 16 threads in both libraries and under ASan; nested OpenMP equals serial.
- C++, Python, Octave and MATLAB R2026a agree in 34 sizes and values.

MEX layer (F):
- All shared helpers use the error ID `quadriga_lib:CPPerror` (conf-file review Q5 not regressed), except
  `version.cpp` (B-115); complex input is rejected; int8, int16, uint16 and logical inputs give a clean error.
- Robustness sweep: 58 wrappers × 16 malformed argument lists × 0, 1, 3 outputs in Octave: only `icosphere(NaN)`
  crashed (B-034); all else gave a clean error or a sensible result.
- Octave against Python: identical sums and shapes for `acdf`, delay spread, Rician K, XPR, `icosphere`,
  point-cloud segmentation and AABB, `arrayant_interpolate` (with distances), beamwidth, directivity,
  `combine_pattern`, `get_channels_spherical` (with angles), `ray_triangle_intersect`, `triangle_mesh_aabb`,
  `point_inside_mesh`, `quantize_delays`, `baseband_freq_response` (ascending carriers and pilot grid).
- HDF5 dataset round trip from MATLAB for double, single, uint32, int64 scalars, vectors, matrices, 3-D and char.

Python layer (G):
- Items 1.2, 1.4 and 1.5 of `md/python_fix_plan_v00.md` are not defects in effect: `hdf5_write_dset` with `ix == nx`
  is rejected by C++ ("Index out of bound."); an invalid arrayant dict is rejected by the C++ validation in every
  wrapper that uses it; an empty `sub_cloud_ind` / `sub_mesh_index` behaves like no index.
- Python against Octave MEX: more than 80 outputs bit-identical over arrayant, channel generation, channel tools,
  statistics, geometry and file functions; the only value difference is B-089.
- numpy layouts (reversed, strided, column slices, transposed, C and F order), dtype conversion by forcecast, no RSS
  growth over 2000 calls of 10 wrappers; batch mode of `get_channels_spherical` equals single snapshots with 4
  threads; the GIL is released only around Armadillo-only code; the QRT reader thread touches no Python object.

## Re-run

Command: `bash build_linux/scratch/hunt/doc/rerun_all.sh`, which runs every `final/<folder>/run.sh` once, one after the other, and stores its output as `rerun.txt` (the outputs shown in the items). Start: Thu Oct 15 06:42:10 PM CEST 2026 load 2.06 7.75 9.26 4/2816 710821; end: Thu Oct 15 06:51:41 PM CEST 2026 load 1.60 2.57 5.76 2/2782 722191.

| Folder | Exit code of run.sh | Time | Load (1 min) after |
|---|---|---|---|
| A-01 | 0 | 4s | 2.06 |
| A-02 | 0 | 4s | 1.97 |
| A-03 | 0 | 3s | 1.97 |
| A-04 | 0 | 4s | 1.97 |
| A-06 | 0 | 3s | 2.22 |
| A-07 | 0 | 3s | 2.22 |
| A-08 | 0 | 3s | 2.20 |
| A-09 | 0 | 3s | 2.18 |
| A-10 | 0 | 3s | 2.18 |
| A-11 | 0 | 3s | 2.09 |
| A-12 | 0 | 3s | 2.08 |
| A-13 | 0 | 3s | 2.08 |
| A-14 | 0 | 5s | 1.99 |
| A-15 | 0 | 3s | 1.99 |
| A-16 | 0 | 3s | 1.99 |
| A-17 | 0 | 3s | 2.00 |
| A-18 | 0 | 4s | 2.00 |
| A-20 | 0 | 2s | 2.00 |
| B-001 | 0 | 10s | 2.00 |
| B-002 | 0 | 9s | 2.07 |
| B-003 | 0 | 3s | 1.98 |
| B-004 | 0 | 9s | 1.98 |
| B-005 | 0 | 2s | 1.91 |
| B-006 | 0 | 14s | 1.78 |
| B-007 | 0 | 5s | 1.72 |
| B-008 | 0 | 4s | 1.66 |
| B-009 | 0 | 8s | 1.61 |
| B-010 | 0 | 5s | 1.56 |
| B-011 | 0 | 13s | 1.58 |
| B-012 | 0 | 6s | 1.54 |
| B-013 | 0 | 4s | 1.49 |
| B-014 | 0 | 3s | 1.49 |
| B-015 | 0 | 7s | 1.50 |
| B-016 | 0 | 3s | 1.50 |
| B-017 | 0 | 9s | 1.57 |
| B-018 | 0 | 4s | 1.53 |
| B-019 | 0 | 29s | 1.77 |
| B-020 | 0 | 7s | 1.71 |
| B-022 | 0 | 3s | 1.65 |
| C-1 | 0 | 3s | 1.65 |
| C-10 | 0 | 4s | 1.60 |
| C-11 | 0 | 14s | 1.54 |
| C-12 | 0 | 2s | 1.54 |
| C-13 | 0 | 3s | 1.58 |
| C-14 | 0 | 16s | 1.66 |
| C-15 | 0 | 3s | 1.61 |
| C-16 | 0 | 3s | 1.61 |
| C-17 | 0 | 3s | 1.56 |
| C-18 | 0 | 3s | 1.60 |
| C-19 | 0 | 3s | 1.60 |
| C-2 | 0 | 3s | 1.55 |
| C-20 | 0 | 0s | 1.55 |
| C-21 | 0 | 3s | 1.55 |
| C-22 | 0 | 3s | 1.58 |
| C-3 | 0 | 5s | 1.54 |
| C-4 | 0 | 6s | 1.73 |
| C-5 | 0 | 3s | 1.68 |
| C-6 | 0 | 3s | 1.68 |
| C-7 | 0 | 2s | 1.62 |
| C-8 | 0 | 3s | 1.62 |
| C-9 | 0 | 3s | 1.57 |
| D-1 | 0 | 5s | 1.53 |
| D-10 | 0 | 3s | 1.56 |
| D-11 | 0 | 4s | 1.56 |
| D-12 | 0 | 3s | 1.60 |
| D-13 | 0 | 3s | 1.55 |
| D-14 | 0 | 3s | 1.55 |
| D-15 | 0 | 3s | 1.59 |
| D-16 | 0 | 3s | 1.59 |
| D-17 | 0 | 3s | 1.54 |
| D-18 | 0 | 2s | 1.54 |
| D-19 | 0 | 3s | 1.50 |
| D-2 | 0 | 6s | 1.46 |
| D-3 | 0 | 6s | 1.42 |
| D-4 | 0 | 2s | 1.47 |
| D-5 | 0 | 3s | 1.47 |
| D-6 | 0 | 6s | 1.47 |
| D-7 | 0 | 5s | 1.44 |
| D-8 | 0 | 5s | 1.40 |
| D-9 | 0 | 5s | 1.37 |
| E-1 | 0 | 3s | 1.37 |
| E-2 | 0 | 0s | 1.37 |
| E-3 | 0 | 1s | 1.37 |
| F-1 | 0 | 31s | 1.75 |
| F-10 | 0 | 3s | 1.75 |
| F-11 | 0 | 0s | 1.75 |
| F-12 | 0 | 1s | 1.75 |
| F-13 | 0 | 0s | 1.75 |
| F-14 | 0 | 0s | 1.75 |
| F-15 | 0 | 1s | 1.69 |
| F-16 | 0 | 0s | 1.69 |
| F-17 | 0 | 0s | 1.69 |
| F-18 | 0 | 4s | 1.69 |
| F-19 | 0 | 3s | 1.63 |
| F-2 | 0 | 12s | 2.13 |
| F-20 | 0 | 0s | 2.13 |
| F-21 | 0 | 1s | 2.12 |
| F-3 | 0 | 7s | 2.11 |
| F-4 | 0 | 5s | 2.02 |
| F-5 | 0 | 4s | 2.18 |
| F-6 | 0 | 99s | 1.44 |
| F-7 | 0 | 0s | 1.44 |
| F-8 | 0 | 1s | 1.44 |
| F-9 | 0 | 0s | 1.44 |
| G-1 | 0 | 2s | 1.44 |
| G-10 | 0 | 0s | 1.44 |
| G-11 | 0 | 0s | 1.44 |
| G-12 | 0 | 0s | 1.44 |
| G-13 | 0 | 0s | 1.44 |
| G-14 | 0 | 1s | 1.44 |
| G-15 | 0 | 0s | 1.44 |
| G-2 | 0 | 0s | 1.44 |
| G-3 | 0 | 1s | 1.57 |
| G-4 | 0 | 1s | 1.57 |
| G-5 | 0 | 1s | 1.57 |
| G-6 | 0 | 1s | 1.57 |
| G-7 | 0 | 6s | 1.60 |
| G-8 | 0 | 0s | 1.60 |
| G-9 | 0 | 0s | 1.60 |

The exit code is that of the script; expected crashes are caught inside it (`|| echo "exit code ..."`). MATLAB crash dumps written by the re-run: `/home/stephan/matlab_crash_dump.716408-1`, `/home/stephan/matlab_crash_dump.716734-1`, `/home/stephan/matlab_crash_dump.717816-1`, `/home/stephan/matlab_crash_dump.719873-1`, `/home/stephan/matlab_crash_dump.720262-1`.

## Items found during the fixes

## B-119: Crafted HDF5 files make HDF5 1.14.6 itself zero-fill about 4 GB in `H5Fopen` or request a 2^64-byte allocation in `H5Gopen2` (upstream, not quadriga-lib code)

- Status: won't fix (upstream HDF5 1.14.6; the library cannot check a file before HDF5 parses it, decided 2026-10-16)
- Severity: S4, because only crafted or corrupted files trigger it, quadriga-lib code is not involved, and after WP-02 the
  library ends both cases with a clean `std::invalid_argument`; the cost is time and memory (12.7 s and 4.2 GB peak for a
  10 kB file).
- Kind: resource (in a dependency)
- Mark: [R] (found by the WP-02 AFL campaign and the manager's corpus replay, 2026-10-16)
- Where: HDF5 1.14.6 (system `libhdf5_serial.so.310`): `H5Fopen` → `H5F__super_read` → `H5FD_read` → `memset` (worker's
  gdb trace in `md/fix_logs/WP-02.md`); `H5Gopen2` → `H5G__open_name` → `H5O_obj_type` → `H5O_protect` → `H5C_protect`.
  quadriga-lib calls: every HDF5 reader opens the file; `qHDF_open_channel_group` (`src/qd_channel_hdf5.cpp:456` after
  WP-02) opens the channel group named in the storage index.

**What happens.** (1) A 10 kB file with a crafted superblock: each `H5Fopen` reads and zero-fills about 4 GB; the
harness opens the file 7 times, 12.7 s, 4.2 GB peak RSS, then every call ends in "Storage index in HDF file is
corrupted.". The pre-WP-02 library behaves the same (14.0 s, 4.2 GB). (2) Two files with a crafted object header: HDF5
requests an allocation of `0xffffffff00000430` bytes while opening the channel group. With glibc `malloc` the request
fails and HDF5 returns an error; the library throws "Storage index in HDF file is corrupted." (before WP-02 it returned
silently). Under ASan with the default `allocator_may_return_null=0` the process aborts; AFL runs ASan with
`allocator_may_return_null=1`, so the campaign did not report these inputs as crashes.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-119/`. The three inputs are binary AFL outputs (10 kB each) and cannot be
shown here; they are identified by the SHA-256 sums in the output below and are also kept in
`build_linux/scratch/fix/WP-02/afl/out/hdf5/main/hangs/id:000000,...`, `.../main/queue/id:000006,...` and
`.../cmp/queue/id:000003,...`.

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-119/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# B-119: crafted HDF5 files make HDF5 1.14.6 itself zero-fill ~4 GB in H5Fopen, or request a ~2^64-byte allocation in
# H5Gopen2. Inputs found by the WP-02 AFL campaign. Uses the HDF5 fuzz harness (hunt/fuzz/fz_hdf5.cpp) twice:
# a release build against build_linux/libquadriga.a, and the existing ASan harness hunt/fuzz/bin/fz_hdf5.
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-119
F=build_linux/scratch/hunt/fuzz
mkdir -p $F/work
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src -I $F $F/fz_hdf5.cpp -o $D/fz_release build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
sha256sum $D/*.bin | sed 's#  .*/#  #'
echo "== (1) release harness, H5Fopen input: wall time and peak memory"
/usr/bin/time -f "   %e s wall, %M kB peak RSS" $D/fz_release $D/h5fopen_zero_fill.bin 2>&1 | grep -E 'EXC|wall' | sort | uniq -c
echo "== (2) release harness, H5Gopen2 inputs (glibc malloc fails, the library throws)"
for f in $D/h5gopen_huge_alloc_*.bin; do echo "   $(basename $f):"; $D/fz_release $f 2>&1 | sort | uniq -c; done
echo "== (3) ASan harness, first H5Gopen2 input: allocation request and frames (unsymbolized, resolved below)"
out=$(ASAN_OPTIONS=symbolize=0:detect_leaks=0 timeout 60 $F/bin/fz_hdf5 $D/h5gopen_huge_alloc_1.bin 2>&1)
echo "$out" | grep -m1 'requested allocation size' | cut -c1-150
for a in $(echo "$out" | grep -oE 'libhdf5_serial.so.310\+0x[0-9a-f]+' | sed 's/.*+//' | head -10); do
  s=$(gdb -batch -ex "info symbol $a" /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.310 2>/dev/null | tail -1); echo "   libhdf5 $a: $s"; done
c=$(echo "$out" | grep -oE 'fz_hdf5\+0x[0-9a-f]+' | sed 's/.*+//' | head -2 | tail -1)
echo "   first caller in quadriga-lib: $(addr2line -f -C -e $F/bin/fz_hdf5 $c | paste -s -d' ' | cut -c1-140)"
```

Output (`bash run.sh`, 2026-10-16, library of the WP-02 work tree):
```text
061c4a41545e5be8e00877322f6520de6e25d2d8df96b7b68445197acc9de79a  h5fopen_zero_fill.bin
3f0c8e82de8d28a56d86b77a346efd866600632653926447a9bc3d610c0a1b79  h5gopen_huge_alloc_1.bin
a8e3bf5444eed09616f04a34dad0edc265805f49f6c1ef1715d072c2d268024b  h5gopen_huge_alloc_2.bin
== (1) release harness, H5Fopen input: wall time and peak memory
      1    12.67 s wall, 4210140 kB peak RSS
      1 EXC0: Storage index in HDF file is corrupted.
      3 EXC1: Storage index in HDF file is corrupted.
      3 EXC2: Storage index in HDF file is corrupted.
== (2) release harness, H5Gopen2 inputs (glibc malloc fails, the library throws)
   h5gopen_huge_alloc_1.bin:
      1 EXC1: Storage index in HDF file is corrupted.
      1 EXC2: Storage index in HDF file is corrupted.
   h5gopen_huge_alloc_2.bin:
      1 EXC1: Storage index in HDF file is corrupted.
      1 EXC2: Storage index in HDF file is corrupted.
== (3) ASan harness, first H5Gopen2 input: allocation request and frames (unsymbolized, resolved below)
==570133==ERROR: AddressSanitizer: requested allocation size 0xffffffff00000430 (0xffffffff00001430 after adjustments for alignment, red zones etc.) e
   libhdf5 0x87062: H5C_protect + 242 in section .text
   libhdf5 0x65bf0: H5AC_protect + 80 in section .text
   libhdf5 0x1c3100: H5O_protect + 480 in section .text
   libhdf5 0x1c5377: H5O_obj_type + 71 in section .text
   libhdf5 0x1502a3: H5G__open_name + 115 in section .text
   libhdf5 0x30a12a: H5VL__native_group_open + 58 in section .text
   libhdf5 0x2f6dde: No symbol matches 0x2f6dde.
   libhdf5 0x30060a: H5VL_group_open + 90 in section .text
   libhdf5 0x1423ae: No symbol matches 0x1423ae.
   libhdf5 0x1425f3: H5Gopen2 + 115 in section .text
   first caller in quadriga-lib: qHDF_open_channel_group(long, unsigned int) /sjc/quadriga-lib-fixes/src/qd_channel_hdf5.cpp:456
```

**Cause.** HDF5 trusts sizes in the superblock and in object headers when it reads them; quadriga-lib only passes the
file name and a group name.

**Proposed fix.** None in quadriga-lib. Options if it ever matters: report upstream; open files through a custom
file driver that rejects reads beyond the file size; run untrusted files in a separate process with a memory limit.

**Test for the fix session.** None (won't fix).

## B-120: `calc_cross_polarization_ratio`: the circular XPR columns report 0 (fully cross-polar) for the LOS matrix that `get_channels_*` treats as co-polar, and are defined per transmit polarisation while the linear columns are per receive polarisation

- Status: fixed (WP-13): circular entries are the coupling r^T M t of the same circular pattern at both ends, LHCP = (V, H) = (1, +j)/sqrt 2 (derived from the library's frames); columns 4 and 5 are per receive polarisation
- Severity: S3, because the circular XPR of every LOS-dominated channel from the ray tracer or `get_channels_*` is reported inverted (co and cross swapped), so circular-polarisation studies read the wrong ratio; the linear columns are not affected.
- Kind: wrong result
- Mark: [R] (found by the WP-04 worker while documenting B-069; repro re-run by the manager, 2026-10-17)
- Where: `src/qd_tools_calc_xpr.cpp`, circular columns 3-5 (transform of `M` to the L/R basis); the Catch2 case "calc_cross_polarization_ratio - Circular polarization identity check" (`tests/catch2_tests/test_calc_cross_polarization_ratio.cpp:124`) pins the 0 and explains it as "a single reflection reverses handedness".

**What happens.** The library's LOS matrix is `M = diag(1, -1)` (rows 0-1 = 1, rows 6-7 = -1: the receive-side mirror of `ray_commit.cpp`, used in the `get_channels_*` tests). With an LHCP pattern (V, H) = (1, j)/sqrt 2 at both ends, `get_channels_spherical` gives |c| = 1 (opposite patterns: 0), i.e. a co-polar link. `calc_cross_polarization_ratio` with the same `M` and `include_los = true` returns circular XPR 0 (LHCP 0, RHCP 0). In addition, the code comment calls column 4 "LHCP XPR (transmit L, co = LL, cross = RL)", an XPR per transmit polarisation, while columns 1-2 are per receive polarisation (documented in WP-04); the handedness convention of the L/R transform (`E_L = (E_V - jE_H)/sqrt 2` or `(E_V + jE_H)/sqrt 2`) is not documented.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-120/` (from the WP-04 worker's `build_linux/scratch/fix/WP-04/circular/circ.cpp`).

`repro.cpp`:
```cpp
// WP-04 B-069 check: circular polarisation with get_channels_spherical vs the circular columns of calc_cross_polarization_ratio
#include "quadriga_lib.hpp"
#include <cstdio>
#include <cmath>
int main()
{
    auto a = quadriga_lib::generate_arrayant_omni<double>(10.0); // (V, H) = (1, j) / sqrt 2
    a.e_theta_re *= std::sqrt(0.5);
    a.e_phi_im = a.e_theta_re;
    a.e_phi_re.zeros();
    auto b = a; // (V, H) = (1, -j) / sqrt 2
    b.e_phi_im *= -1.0;
    arma::mat F = arma::vec({50.0, 0.0, 0.0}); // LOS
    arma::vec g = {1.0}, L = {100.0};
    arma::mat M = arma::vec({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}); // LOS matrix of get_channels_* / ray_commit
    arma::cube cr, ci, dl;
    quadriga_lib::get_channels_spherical<double>(&a, &a, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, &F, &F, &g, &L, &M, &cr, &ci, &dl, 0.0);
    printf("get_channels_spherical, LOS M = diag(1,-1): same circular pattern at TX and RX: |c| = %.3f\n", std::hypot(cr(0), ci(0)));
    quadriga_lib::get_channels_spherical<double>(&a, &b, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, &F, &F, &g, &L, &M, &cr, &ci, &dl, 0.0);
    printf("get_channels_spherical, LOS M = diag(1,-1): opposite circular patterns: |c| = %.3f\n", std::hypot(cr(0), ci(0)));
    std::vector<arma::vec> pw = {{1.0}}, pl = {{100.0}};
    std::vector<arma::mat> Mv = {M};
    arma::mat tx = arma::vec({0.0, 0.0, 0.0}), rx = arma::vec({100.0, 0.0, 0.0}), xpr;
    quadriga_lib::calc_cross_polarization_ratio<double>(pw, Mv, pl, tx, rx, &xpr, nullptr, true);
    printf("calc_cross_polarization_ratio, same M, include_los: circular XPR = %g, LHCP = %g, RHCP = %g\n", xpr(0, 3), xpr(0, 4), xpr(0, 5));
    return 0;
}
```

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-120/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# B-120: circular XPR columns of calc_cross_polarization_ratio vs get_channels_spherical for the LOS matrix diag(1, -1)
cd /sjc/quadriga-lib-fixes
D=build_linux/scratch/hunt/final/B-120
g++-12 -std=c++17 -O2 -fopenmp -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_LAPACK -I include -I src "$D/repro.cpp" -o "$D/repro" build_linux/libquadriga.a $(pkg-config --cflags --libs hdf5) || echo "build failed"
OMP_NUM_THREADS=1 "$D/repro" || echo "exit code $?"
```

Output (`bash run.sh`, 2026-10-17, library of the WP-04 work tree):
```text
get_channels_spherical, LOS M = diag(1,-1): same circular pattern at TX and RX: |c| = 1.000
get_channels_spherical, LOS M = diag(1,-1): opposite circular patterns: |c| = 0.000
calc_cross_polarization_ratio, same M, include_los: circular XPR = 0, LHCP = 0, RHCP = 0
```

**Cause.** Probably the circular transform assumes an `M` without the receive-side mirror, so for `M` from `get_channels_*` or the ray tracer the co- and cross-polar terms are swapped; to be confirmed in the fix session. Confirmed in WP-13: the old entries were r^H M t (conjugated circular basis at the receiver) instead of the r^T M t coupling that the channel generators use, so co- and cross-polar were swapped; the old columns are the reciprocals of the new ones (3), with 4 and 5 exchanged.

**Proposed fix.** Define the handedness convention and the per-receive (or per-transmit) meaning of the circular columns consistently with the linear ones and with `get_channels_*`, fix the transform, and correct the pinned test. Behaviour change of columns 3-5. File: `src/qd_tools_calc_xpr.cpp` (and the MEX / Python blocks).

**Test for the fix session.** Catch2: for `M = diag(1, -1)` the circular XPR is infinite (co-polar), consistent with `get_channels_spherical` for circular patterns at both ends; a reflection matrix gives the expected handedness change.

## B-121: Test defect: `test_bin` is compiled without `BUILD_WITH_AVX2`, so the AVX2-vs-generic cases of the ray-triangle and ray-point kernel tests never run

- Status: fixed (WP-12)
- Severity: S3, because the only tests that compare the AVX2 ray kernels with the generic ones have never run, so differences such as B-071 went unnoticed; the library itself is built with AVX2.
- Kind: test defect
- Mark: [R] (found by the WP-04 worker; confirmed by the manager, 2026-10-17)
- Where: `CMakeLists.txt` (test_bin target, around line 679-698: no `BUILD_WITH_AVX2` definition, while `quadriga_static` has `-DBUILD_WITH_AVX2=1`); `tests/catch2_tests/test_ray_triangle_intersect_internal.cpp:8, 370` and `tests/catch2_tests/test_ray_point_intersect_internal.cpp:9, 304, 589` (`#ifdef BUILD_WITH_AVX2`).

**What happens.** "RTI Internal - AVX2 comparison" is not compiled into `test_bin` at all; "RPI Internal - Simple Mode (AVX2 vs GENERIC)" and "RPI Internal - Dense Grid (AVX2 vs GENERIC)" are listed but their bodies are compiled out, so they pass with no assertions. No other test file uses `BUILD_WITH_AVX2`.

**Reproduce.**

Folder `build_linux/scratch/hunt/final/B-121/`.

`run.sh` (run from anywhere with `bash build_linux/scratch/hunt/final/B-121/run.sh`; it changes to the worktree root itself):
```bash
#!/bin/bash
# B-121: test_bin is compiled without BUILD_WITH_AVX2, so the AVX2-vs-generic cases of the ray kernel tests are compiled out
cd /sjc/quadriga-lib-fixes
echo "== BUILD_WITH_AVX2 in the compile flags: library / test_bin"
grep -c 'BUILD_WITH_AVX2' build_linux/CMakeFiles/quadriga_static.dir/flags.make
grep -c 'BUILD_WITH_AVX2' build_linux/CMakeFiles/test_bin.dir/flags.make
echo "== test files that depend on it"
grep -n '#ifdef BUILD_WITH_AVX2' tests/catch2_tests/*.cpp
echo "== test cases of the two files that test_bin knows (tags of all cases in those files)"
for f in test_ray_triangle_intersect_internal test_ray_point_intersect_internal; do
  n_src=$(grep -c '^TEST_CASE' tests/catch2_tests/$f.cpp)
  echo "$f.cpp: $n_src TEST_CASE lines in the source"
done
build_linux/test_bin --list-tests 2>/dev/null | grep -i -E 'avx2' | head -20
echo "(test cases listed above with AVX2 in the name: these are the ones that run)"
echo "== the listed RPI AVX2 cases run with an empty body (no assertions)"
for t in "RPI Internal - Simple Mode (AVX2 vs GENERIC)" "RPI Internal - Dense Grid (AVX2 vs GENERIC)"; do
  echo "$t: $(flock /tmp/quadriga-fixes-suite.lock build_linux/test_bin "$t" 2>&1 | grep -E "assertions|test cases" | tr "\n" " ")"
done
echo "== the RTI AVX2 case is not in test_bin at all"
build_linux/test_bin --list-tests 2>/dev/null | grep -c 'RTI Internal - AVX2 comparison'
```

Output (`bash run.sh`, 2026-10-17):
```text
== BUILD_WITH_AVX2 in the compile flags: library / test_bin
1
0
== test files that depend on it
tests/catch2_tests/test_ray_point_intersect_internal.cpp:9:#ifdef BUILD_WITH_AVX2
tests/catch2_tests/test_ray_point_intersect_internal.cpp:304:#ifdef BUILD_WITH_AVX2
tests/catch2_tests/test_ray_point_intersect_internal.cpp:589:#ifdef BUILD_WITH_AVX2
tests/catch2_tests/test_ray_triangle_intersect_internal.cpp:8:#ifdef BUILD_WITH_AVX2
tests/catch2_tests/test_ray_triangle_intersect_internal.cpp:370:#ifdef BUILD_WITH_AVX2
== test cases of the two files that test_bin knows (tags of all cases in those files)
test_ray_triangle_intersect_internal.cpp: 9 TEST_CASE lines in the source
test_ray_point_intersect_internal.cpp: 13 TEST_CASE lines in the source
  fast_atan2 - Large vector (exercises OpenMP and AVX2 bulk path)
  RPI Internal - Simple Mode (AVX2 vs GENERIC)
  RPI Internal - Dense Grid (AVX2 vs GENERIC)
  sos_val - the AVX2 kernel agrees with GENERIC at any distance from the origin
  sos_val - the automatic kernel is AVX2 when available
  sos_val - the AVX2 kernel stays within its bound up to 1e9 periods from the
  sos_val - the AVX2 kernel stays within its bound for the largest preset
(test cases listed above with AVX2 in the name: these are the ones that run)
== the listed RPI AVX2 cases run with an empty body (no assertions)
RPI Internal - Simple Mode (AVX2 vs GENERIC): test cases: 1 | 1 passed assertions: - none - 
RPI Internal - Dense Grid (AVX2 vs GENERIC): test cases: 1 | 1 passed assertions: - none - 
== the RTI AVX2 case is not in test_bin at all
0
```

**Cause.** The test sources check the macro, but only the library target defines it.

**Proposed fix.** Define `BUILD_WITH_AVX2` for `test_bin` exactly when the library is built with AVX2 (same CMake condition); compile the AVX2 test parts with the kernel's flags if they call the intrinsics directly; keep the run-time `quadriga_lib_has_AVX2()` guard. Expect the newly running cases to expose B-071. File: `CMakeLists.txt`.

**Test for the fix session.** `test_bin --list-tests` contains "RTI Internal - AVX2 comparison", and the three AVX2 cases report assertions; the case counts of the suite go up accordingly.


## B-122: `qrt_file_read_raw` is quadratic in the number of paths per origin (`reserve` per CIR): 240 000 paths take 7.5 s

- Status: fixed (WP-14): geometric growth of the output; linear scaling
- Severity: S3, because reading the raw paths of a large ray-tracing result takes seconds to minutes where linear time would take milliseconds; results are correct.
- Kind: performance
- Mark: [M] (found and measured by the WP-07 worker, 2026-10-19; details and timings in `md/fix_logs/WP-07.md`, "Found on the way")
- Where: `src/qd_channel_qrt.cpp`, `qrt_file_read_raw` (around line 1872 after WP-07): `out.reserve(...)` is called once per CIR with a slightly larger size, so the output vectors are reallocated and copied on every CIR.

**What happens.** On a generated 93 MB QRT file, each doubling of the paths per origin makes `qrt_file_read_raw` about 4 times slower: 0.43 s, 1.74 s and 7.50 s for 60 000, 120 000 and 240 000 paths; the same in the code before WP-07.

**Reproduce.** The WP-07 benchmark generator and timing scripts in `build_linux/scratch/fix/WP-07/` (see the log); time `qrt_file_read_raw` of one origin for 750, 1500 and 3000 CIRs.

**Proposed fix.** Reserve once per origin from the known path count (or grow geometrically). File: `src/qd_channel_qrt.cpp`.

**Test for the fix session.** Timing ratio check: doubling the paths at most doubles the time (plus noise); outputs unchanged (checksum).

## B-123: MEX and Python `qrt_file_read` build a 2^32-entry index vector (34 GB) for a valid file with 0 CIRs

- Status: fixed (WP-15): empty outputs for 0 CIRs, no large allocation
- Severity: S3, because a valid file without CIRs makes the wrapper allocate 34 GB (MemoryError in Python, "out of memory" in Octave) instead of returning empty outputs.
- Kind: defect (unsigned underflow)
- Mark: [R] (found by the WP-07 worker under `ulimit`, 2026-10-19; details in `md/fix_logs/WP-07.md`)
- Where: `api_mex/qrt_file_read.cpp:100`, `api_python/qpy_qrt_file_read.cpp:127`: the default CIR list is `regspace(0, no_cir - 1)` in 32 bit, which underflows for `no_cir = 0`.

**Reproduce.** A QRT file with 0 CIRs written with `qrt_file_init`, then `quadriga_lib.qrt_file_read(fn)` in Octave / Python under `ulimit -v 4000000`.

**Proposed fix.** Return empty outputs (or an empty CIR list) when `no_cir == 0`. Files: the two wrappers.

**Test for the fix session.** MOxUnit and pytest: a file with 0 CIRs returns empty outputs without a large allocation.
