# WP-17b log: final verification of the committed `fixes` branch

Date 2026-10-22. Start: bac8a45 (notes commit after a2d0a49), clean tree. Outputs in `build_linux/scratch/fix/WP-17b/`.
Only change to the tree: `tests/mex_tests/test_calc_angular_spreads_sphere.m` (see "Test change"). Nothing committed,
`md/bugs.md` and `md/bug_fix_plan.md` untouched.

Result in one line: every build, suite, fuzz campaign and repro meets its expectation except two points, listed at the
end: (1) MATLAB R2024a cannot load 16 MEX files since WP-10 (`GLIBCXX_3.4.29` from `std::to_chars`), and (2) one new
HDF5 fuzz input takes 17 s outside AFL, all of it inside HDF5's `H5Fopen` (the known upstream item B-119).

## Test change

`tests/mex_tests/test_calc_angular_spreads_sphere.m:106-113`: the "too few arguments" check used
`try ...; error('Expected an error'); catch; end`, which passes whether or not the wrapper throws. Rewritten as in
`test_acdf.m`: a `failed` flag, set only when the error message contains `Wrong number of input arguments.`, then
`assertTrue(failed)`. The call is `[~] = quadriga_lib.calc_angular_spread(az, el);` (the wrapper checks `nrhs` before
anything else, so `nlhs = 0` would also throw; `[~]` keeps the check independent of that).

Shown to fail when the wrapper does not throw (Octave, single test):
- passing call `calc_angular_spread(az, el, ones(size(az)))` put in temporarily: `FAILED (failure=1)`,
  `test_calc_angular_spreads_sphere:113`, `assertTrue` input 0;
- condition inverted temporarily (`failed = isempty(...)`): `FAILED (failure=1)` at line 113;
- restored: `OK (passed=1)`. `git diff` shows only the intended 5 + / 4 - lines.

No other `try/error/catch` check of this kind in `tests/mex_tests/` (the only other bare `catch` is the file cleanup
helper in `test_write_png.m`).

## Builds

Fresh directories: the previous build directories were moved aside, not deleted: `build_linux_pre17b/` (everything of
`build_linux/` except `scratch/`, which stayed in place) and `build_generic_pre17b/`, `build_asan_pre17b/`,
`build_asan15_pre17b/`, `build_afl_plain_pre17b/`, `build_afl_cmplog_pre17b/` (about 5 GB, gitignored; can be deleted).
Commands of the Baseline "Builds" table with `--parallel 12` (`build_linux/scratch/fix/WP-17b/builds.sh`, logs
`build_*.log`). One build at a time.

| Build | Directory | Time | Compiler warnings | Load before / after |
|---|---|---|---|---|
| base gcc-12, AVX2, MATLAB + Octave MEX, Python, tests; `cmake --install build_linux` | `build_linux` | 139 s | 0 | 3.0 / 11.4 |
| generic gcc-12 | `build_generic` | 84 s | 0 | 8.6 / 10.6 |
| ASan + UBSan gcc-12 | `build_asan` | 608 s | 0 | 10.0 / 4.0 |
| ASan + UBSan gcc-15 | `build_asan15` | 302 s | 0 | 3.7 / 5.7 |
| AFL ASan (afl-clang-fast, no AVX2) | `build_afl_plain` | 32 s | 0 | 5.0 / 8.3 |
| AFL CMPLOG | `build_afl_cmplog` | 24 s | 0 | 7.8 / 9.2 |

The four gcc builds also print two `CMake Warning` notices from the Catch2 external project at configure time (no
`URL_HASH`; `CMAKE_C_COMPILER` unused by the sub-project). They are not compiler warnings and come with every fresh
configure. Install: `lib/` (static library, Python module) and `+quadriga_lib/` (231 files) rewritten at 11:53-11:55.

## Suites

Catch2 from the repository root through `flock /tmp/quadriga-fixes-suite.lock`, `OMP_NUM_THREADS=12`
(`suite.sh`, outputs `suite_*.txt`). Sanitizer runs with
`ASAN_OPTIONS=halt_on_error=0:detect_leaks=1:suppress_equal_pcs=0 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0`.

| Run | Result | Expected | Wall | Load before / after |
|---|---|---|---|---|
| base `build_linux/test_bin` | `All tests passed (1348756 assertions in 1300 test cases)` | 1348756 / 1300 | 19 s | 7.9 / 6.6 |
| generic `build_generic/test_bin` | `All tests passed (1346833 assertions in 1299 test cases)` | 1346833 / 1299 | 19 s | 6.6 / 5.6 |
| base `[wave]` | `All tests passed (163474 assertions in 178 test cases)` | 163474 / 178 | 5 s | 5.4 / 5.3 |
| base `[acoustic]` | `All tests passed (42741 assertions in 59 test cases)` | 42741 / 59 | 1 s | 5.3 / 5.3 |
| generic `[wave]` | `All tests passed (163474 assertions in 178 test cases)` | 163474 / 178 | 5 s | 5.3 / 5.1 |
| generic `[acoustic]` | `All tests passed (42741 assertions in 59 test cases)` | 42741 / 59 | 1 s | 5.1 / 5.1 |
| ASan + UBSan gcc-12 | `All tests passed (1348756 assertions in 1300 test cases)`; 0 ASan reports, 0 leaks, 6 UBSan reports, all excluded `ray_progress` | as base, only the excluded reports | 131 s | 5.1 / 3.2 |
| ASan + UBSan gcc-15 | `All tests passed (1348756 assertions in 1300 test cases)`; same 6 UBSan reports, nothing else | as base, only the excluded reports | 90 s | 2.8 / 2.4 |

The 6 UBSan reports in both sanitizer builds: 3 x `armadillo_bits/access.hpp:26` and 3 x `Mat_meat.hpp:7490`
("reference binding to null pointer"), from `src/ray_progress.cpp:829`, `:830`, `:831` (2 each), called from
`tests/catch2_tests/test_wave_progress.cpp:299`. The brief names line 301; the WP-16 and WP-17a logs already give
line 299 (the acoustic session's test cleanup moved the call). Same reports, same source lines.

## MEX suite

`tests/quadriga_lib_mex_tests.m`, `OMP_NUM_THREADS=8`, from `tests/`, one runtime at a time (outputs `mex_*.txt`).
68 tests.

| Runtime | Result | Wall | Load before |
|---|---|---|---|
| Octave 11.1.0 | `FAILED (passed=67, failure=1)`: only `test_ray_mesh_interact:113` (0.98485 vs 0.69639, excluded) | 2 s | 2.6 |
| MATLAB R2024a (`matlab24 -batch`) | **`FAILED (passed=50, failure=18)`**: 17 x `Invalid MEX-file ... libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`, plus the excluded `test_ray_mesh_interact:113`. See list item 1 | 11 s | 3.2 |
| MATLAB R2024a, `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` (diagnosis only) | `FAILED (passed=67, failure=1)`: only the excluded failure | 7 s | 2.9 |
| MATLAB R2026a (`matlab -batch`) | `FAILED (passed=67, failure=1)`: only the excluded failure; exit code 0; no new `~/matlab_crash_dump.*` | 12 s | 2.6 |

(The suite script does not set an exit code on test failures; R2024a also exited with 0.)

## pytest

| Module | Result | Load before / after |
|---|---|---|
| `PYTHONPATH=/sjc/quadriga-lib-fixes/lib`, `OMP_NUM_THREADS=8` (module `0.12.2_AVX2` from the fresh install) | `817 passed, 1 skipped, 52 subtests passed in 2.44s`; skip: `test_ray_point_intersect.py:417` (no CUDA) | 2.4 / 2.6 |
| ASan + UBSan module (`bash build_linux/scratch/fix/build_asan_mod.sh 12`, 53 s, 0 warnings, against the fresh `build_asan`), `LD_PRELOAD` libasan + libubsan, `ASAN_OPTIONS=detect_leaks=0`, reports to `log_path` files | `816 passed, 2 skipped, 52 subtests passed in 4.16s`; 0 report files; skips: CUDA, and `test_qrt_file_reader.py:288` (RLIMIT_AS cannot make allocations fail under ASan) | 3.0 / 3.5 |

The loaded module was checked with `quadriga_lib.__file__` (`build_linux/scratch/hunt/G/asan_mod/...`).

## AFL

Folder `build_linux/scratch/fix/WP-17b/afl/` (model WP-14): harnesses `fz_*.cpp` copied unchanged from
`hunt/fuzz/`, `common.hpp` with the work directory changed to `afl/work`, `build_harness.sh`, `run_fuzz.sh` (copy of
the hunt script writing into `afl/out/`; `hunt/fuzz/out/` untouched), `replay.sh`, `replay_all.sh`. Seeds: the hunt
seeds plus `qdant/newfmt.qdant` (WP-10) and `obj/crlf_rel_null.obj`, `mtl/crlf.mtl` (WP-14). Harnesses built with the
Baseline command against the fresh AFL libraries; obj, mtl, csv TUs at `-O1`, the others at `-O2`; 14 binaries,
0 warnings; every seed runs with exit code 0 and no sanitizer report in both binaries. 900 s per instance, one ASan
main and one CMPLOG secondary per reader, two batches: qdant + obj + mtl + csv (12:23-12:38, load 2.6 / 12.3), then
conf + qrt + hdf5 (12:38-12:54, load 10.0 / 9.5; other sessions were busy).

Replay outside AFL: every queue, crash and hang input with the ASan harness, one at a time
(`ASAN_OPTIONS=allocator_may_return_null=1:max_allocation_size_mb=8000:hard_rss_limit_mb=16000:detect_leaks=1`,
120 s timeout; `replay_*.txt`, `replay_summary.txt`; load 7.0 / 2.5).

| Reader | Executions (main + cmp) | Edges (main) | Crashes | Hangs (in AFL) | Inputs replayed | Slowest replayed input |
|---|---|---|---|---|---|---|
| qdant | 124,428 + 787,973 = 912,401 | 2455 | 0 | 0 | 1330 | 0.037 s |
| obj | 124,647 + 845,502 = 970,149 | 616 | 0 | 0 | 1050 | 0.036 s |
| mtl | 249,697 + 786,837 = 1,036,534 | 605 | 0 | 0 | 588 | 0.036 s |
| csv | 246,202 + 751,423 = 997,625 | 804 | 0 | 0 | 784 | 0.037 s |
| conf | 107,872 + 646,515 = 754,387 | 949 | 0 | 0 | 912 | 0.058 s |
| qrt | 254,216 + 796,016 = 1,050,232 | 1323 | 0 | 0 | 710 | 0.030 s |
| hdf5 | 97,131 + 133,106 = 230,237 | 1206 | 0 | 2 (cmp) | 304 | **17.03 s** (`cmp/hangs/id:000000`); next 1.15 s (`cmp/hangs/id:000001`), all others <= 0.04 s |
| phase-0 HDF5 (`hunt/fuzz/out/hdf5/`, 11 crash + 20 hang inputs) | - | - | - | - | 31 | 0.037 s; all exit 0, no sanitizer report (malformed-file exceptions) |

All replayed inputs: exit code 0, no sanitizer report, no leak. The two HDF5 hang inputs are analysed in list item 2
(`afl/slow/`).

## Repros

All 122 `build_linux/scratch/hunt/final/<folder>/run.sh`, one after the other (`repro/run_all.sh`), with
`OMP_NUM_THREADS=2` (scripts that set their own value keep it), `OMP_THREAD_LIMIT=12` (C-8 asks for 16 threads),
`timeout 300`. Outputs `repro/<folder>.txt`; total 374 s, longest F-1 (26.7 s); load 1.6 before / 2.1 after; no
timeout, every script returned 0.

Allocation probes, read before running: A-14 (`ulimit -v 8000000`), B-009 (`ulimit -v 30000000`) and C-11 (b)
(`ulimit -v 8000000`) are plain builds under a limit; C-15 (about 800 MB before the fix) and B-119 (about 4 GB zero-fill
inside HDF5) are plain and bounded; the ASan scripts D-11, B-002 and F-6 test overflow guards that now throw before
allocating. No script allocates without a bound, so none was skipped.

Extra checks where a script cannot show the fixed behaviour (`repro/checks/`, `repro/F-6_*.txt`): A-17 with the new
`propagation_speed = 344`; A-20 empty destination on a writable object (the script's object is still read-only);
C-10 and B-001 run without the output filter; B-006 Python return type; F-6 MATLAB part in R2026a (R2024a cannot load
the MEX file) and its remaining cases in Octave.

Verdicts: "fixed" = the values show the corrected behaviour; "documented" / "as expected" = the item's recorded
resolution (documentation, decision, upstream); "NOT FIXED" = none.

| Folder | Item | Output (one line) | Verdict |
|---|---|---|---|
| A-01 | B-006 | ASan: `get_channels_spherical(use_avx2=true) returned, 200 coefficients`, no report | fixed |
| A-02 | B-007 | ASan: NaN angle gives `V_re = nan`, no report | fixed |
| A-03 | B-037 | scale 1e-8: AVX2 `2.354853 2.019394 1.292356` = generic | fixed |
| A-04 | B-008 | coupling `[1 0] [1 0] [0 1] [0 1]` as expected; empty `coupling_im` returns, no report | fixed |
| A-06 | B-009 | output keeps `name=myant fc=3.5e+09 n_ports=1` | fixed |
| A-07 | B-010 | heading 180: `bw_az=89.83 point=(-180.0, 0.0)`; zenith `bw_el=89.24` | fixed |
| A-08 | B-011 | beam at el=90: pol 1 `n_ports=1`, pol 2 `n_ports=2` (expected), no report | fixed |
| A-09 | B-012 | `max abs(e_theta) written 2.01e-06, read back 2.01e-06` | fixed |
| A-10 | B-038 | `123.456789 -> 123.456790` (float precision, error 1e-6) | fixed |
| A-11 | B-039 | invalid entry throws, `after: file exists = 1` | fixed |
| A-12 | B-040 | `read id=1: error 'Could not find an arrayant object with ID 1'` | fixed |
| A-13 | B-041 | layout without argument replaced by `1 2 3 4` | documented |
| A-14 | B-042 | `'NoElements' (200000000) is larger than the number of elements found`, `max RSS 5016 kB` | fixed |
| A-15 | B-043 | `is_valid()` = `is_valid(false)` = the grid error | fixed |
| A-16 | B-044 | ULA: 4 ports, identity coupling, own positions; 3GPP positions 0 | fixed |
| A-17 | B-045 | default speed `2.000 / 2.000`; extra check with speed 344: `2.000 / 0.506` (expected 0.506) | fixed |
| A-18 | B-046 | scale 1e-8: `7.512 dBi`, `remove_zeros -> 37 x 73` | fixed |
| A-20 | B-101 | read-only: `Cannot copy elements of a read-only array antenna object.`; extra check: `Input 'destination' is empty.` | fixed |
| B-001 | B-013 | ASan: no report in 3 cases; release: `Malformed HDF5 file: dataset 'coeff_re' has extent 50 ..., expected 1` (uncaught in the probe) | fixed |
| B-002 | B-014 | `Storage layout has too many slots: nx * ny * nz * nw must not exceed 4294967295.`; ASan: no READ/WRITE report | fixed |
| B-003 | B-001 | `par fields read back: b=2` / `x=99` (expected) | fixed |
| B-004 | B-015 | `0 non-zero bytes stored after the terminating NUL`; ASan no report | fixed |
| B-005 | B-016 | `aod alone: 2 element(s), aod(0) = -0.305922` (the "stale 12345" text is a fixed label) | fixed |
| B-006 | B-002 | `negtiny: 5 taps`; negative / NaN delays throw; normal use `returned 4 taps`; ASan no report; Python returns (the script's `.shape` print fails on the list result) | fixed |
| B-007 | B-017 | `TX1 orientation [0 0 0.5]`, `read_raw round trip: 0 mismatches` (v5, v6) | fixed |
| B-008 | B-018 | `mtl_ind written {1 0 2}, read back {1 0 2}` | fixed |
| B-009 | B-048 | `no_path = 20000000 does not fit into the file`, max RSS 4.3 MB, 0.00 s (all 3 cases) | fixed |
| B-010 | B-049 | `i_cir = 2^32: CIR index exceeds number of CIRs` (cache and no cache) | fixed |
| B-011 | B-050 | all round trips `is_valid(read)=''`, snapshot counts kept, `11 of 11` dset names | fixed |
| B-012 | B-051 | caller's handles stay valid, `H5Dwrite ... returned 0`; `open file ids ... 1 (caller holds 1)` | fixed |
| B-013 | B-052 | file grows `160108 -> 792204` bytes over 5 rewrites, old link removed | fixed (decision (a): space reclaimed only by `h5repack`, documented; same output in the WP-02 log) |
| B-014 | B-053 | `string of length 28` = "written by another HDF5 tool" | fixed |
| B-015 | B-054 | offset 0.02 s: `0.0005 s, 36 taps` (offset 0: 0.0007 s) | fixed |
| B-016 | B-055 | aliased outputs = separate outputs (`1 0.5 0.5 0.32 0.68`) | fixed |
| B-017 | B-056 | `80000 faces: 0.019 s` (linear) | fixed |
| B-018 | B-057 | CRLF `Kd = 1 0 0`; bad face index throws for `face_ind` only; no-output call validates | fixed |
| B-019 | B-058 | constant / NaN data give the first colour; `min_val > max_val` and empty data throw; UBSan what-if: no report | fixed |
| B-020 | B-059, B-060 | `add_paths` count mismatch throws; `normalize_M = 5` throws; gain documented as sum (4) | fixed |
| B-022 | B-103 | per-antenna input `fix_taps=1/2: 2x2x6` (docs now say shared input only) | fixed (docs) |
| B-119 | B-119 | H5Fopen input `12.62 s wall, 4209968 kB peak RSS`, then `Storage index ... corrupted`; ASan: request `0xffffffff00000430` in `H5Gopen2` | as expected (won't fix, upstream HDF5) |
| B-120 | B-120 | same circular pattern `abs(c) = 1.000`, opposite `0.000`; XPR = inf for the same M | fixed |
| B-121 | B-121 | `BUILD_WITH_AVX2` in test_bin flags `1`; RPI AVX2 cases 81 and 1281 assertions; RTI AVX2 case listed (`1`) | fixed |
| C-1 | B-003 | `delay*c = 156.568542 m (expected 156.568542)` (spherical, multifreq) | fixed |
| C-10 | B-061 | `Input 'interact_coord_1' must have 3 rows and sum(no_interact_1) = 10 columns.`, no report | fixed |
| C-11 | B-062 | (a) generic ASan `returned, hr(0,0,0) = 1`, no report; (b) under `ulimit -v`: `returned` | fixed |
| C-12 | B-063 | IEEE case alone and after the HDF case: all passed, no `test_qd_ieee_chan.cpp:318` report | fixed |
| C-13 | B-064 | `XPR_NLOS_linear must be NAN (model default) or a finite value larger than 0.` | fixed |
| C-14 | B-065 | non-finite `observation_time` and tiny `quantize` throw, no float-cast report | fixed |
| C-15 | B-066 | `max RSS = 4460 kB, wall = 0.00 s` | fixed |
| C-16 | B-067 | `Input 'bins' must be finite and strictly ascending.` | fixed |
| C-17 | B-068 | `n_out = 4, abs(c)^2 = 1.000 0.500 0.000 0.250` | documented |
| C-18 | B-069 | `V-XPR = inf, H-XPR = 4` | documented |
| C-19 | B-070 | 1000 Hz: phase 0; EM-equivalent 8.75e8 Hz: -3.1416 | documented |
| C-2 | B-019 | absolute delays `9.8e-08`; relative delays `2.000e+00` | documented |
| C-20 | B-105 | `All tests passed (22 assertions)`, no `runtime error` | fixed |
| C-21 | B-106 | split from 165 deg (4 rings), not at 90 deg | fixed |
| C-22 | B-106 | `nREF = 0, nTRA = 0` for invalid types; `sizeof(path) = 56` (header: 56, `static_assert`) | fixed |
| C-3 | B-020 | tau 0.12 s: `abs(H - H_exact) = 1.65e-08` in both builds | fixed |
| C-4 | B-026 | tau 5 s: `abs(H) = 1`, error `5.1e-07` in both builds | fixed |
| C-5 | B-021 | taps `0.000 / 10.000 ns` for 0-4 floors | fixed |
| C-6 | B-022 | 41-42 taps at every distance and floor count | fixed |
| C-7 | B-023 | `max abs(H_ul^T - H_dl) = 0.000e+00`, same Doppler sign | fixed |
| C-8 | B-024 | 1 and 16 (capped at 12) threads: `X(0,0) = -0.416146837` = cos(2) | fixed |
| C-9 | B-025 | `0 are NaN` (2 and 3 collinear points) | fixed |
| D-1 | B-026 | float `max abs(s) = 1`, errors ~1.2e-7 in both builds; double values = reference | fixed |
| D-10 | B-077 | ASan: `empty mesh: returned 0 ids` | fixed |
| D-11 | B-078 | ASan: `EXCEPTION Input 'n_div' is too large.` | fixed |
| D-12 | B-079 | aliased `sbs = (0 0 -1)`; in place `max abs(diff) = 0` | fixed |
| D-13 | B-080 | ASan with leak check: `returned -3 (failure)`, no leak | fixed |
| D-14 | B-081 | ids returned as given (`0 1`, `1 2`), docs 0-based | fixed |
| D-15 | B-082 | `Input 'radius' must be a finite positive number.` | fixed |
| D-16 | B-083 | `n_in = n_out = 40000: 0.002 s` | fixed |
| D-17 | B-084 | `distinct reason vectors over 6 calls: 1` (1 and 4 threads) | fixed |
| D-18 | B-107 | `xo = nan -> nan`; clamping documented | fixed |
| D-19 | B-108 | 10-column mesh accepted; index-map directions as documented | fixed |
| D-2 | B-027 | `0 of 12 points misclassified` at all offsets, both builds | fixed |
| D-3 | B-028 | overlap found with vertices on the face planes (both builds) | fixed |
| D-4 | B-071 | GENERIC and AVX2 both `fbs_ind=1 sbs_ind=2` | fixed |
| D-5 | B-072 | dz = 0 in plane z = Zmin: `fbs_ind = 1` in both kernels | fixed |
| D-6 | B-073 | default segmentation, `use_kernel=0: OK, total hits 2999` (both builds) | fixed |
| D-7 | B-074 | invalid kernels throw; generic build: `AVX2 kernel requested but not available` | fixed |
| D-8 | B-075 | `fast_atan2(1, nan) = nan`; tiny / huge vectors `el=0.6154797` | fixed |
| D-9 | B-076 | AVX2 lane 0 identical with or without a slow-path neighbour, `abs(X)-1 = -3.4e-08` | fixed |
| E-1 | B-109 | `Output 'val' cannot be an input or share memory with one.`; `phase_b` overlap rejected | fixed |
| E-2 | B-034 | Octave: `Input 'seed' must be an integer from -1 to 9223372036854775807.`; Python `TypeError` | fixed |
| E-3 | B-110 | `Input 'preset' is not a known preset.` (Octave and Python) | fixed |
| F-1 | B-029 | R2026a with the worktree MEX: `rc=0 newdumps=[]` (OMP 8 and 1); R2024a rc=0; the old-link what-if control: rc=137 and a dump, as the script expects | fixed |
| F-10 | B-088 | defaults `as=1.627282 es=0.190512` in MATLAB, C++ and Python | fixed |
| F-11 | B-089 | MATLAB padding `0 0 0 0`, same as Python | fixed |
| F-12 | B-090 | descending carriers accepted; equal carriers rejected (MATLAB, Python) | fixed |
| F-13 | B-091 | `i_irs=0` rejected (1-based), `i_irs=1/2` OK | fixed |
| F-14 | B-092 | MATLAB: `... do not form a frequency set ...; pass 'id'`; Python single dict | fixed |
| F-15 | B-093 | `Inputs with more than 4 dimensions are not supported.` | fixed |
| F-16 | B-094 | `Entries in 'obj_ind' cannot be 0 (1-based index).` | fixed |
| F-17 | B-112 | `class(interact_type{1}) = uint8`; doc now says uint8 (the script's "(doc: uint32)" is a fixed label) | fixed |
| F-18 | B-095 | lod 7: `coord [3 9 2]` in MATLAB and C++, `generate_diffraction_paths lod=7 OK` | fixed |
| F-19 | B-113 | MATLAB 0 paths + fake LOS: coeff 0, LOS delay; C++ returns | fixed |
| F-2 | B-030 | Octave and R2024a: `Sparse inputs are not supported.`; ASan what-if: no report; no dump | fixed |
| F-20 | B-114 | usage 3 on struct arrays OK, `nlhs=0: OK` | fixed |
| F-21 | B-115 | error ID `quadriga_lib:CPPerror`; docs "2 to 11 outputs", row vectors "stored as given"; `interp` 2-D without `yI` rejected | fixed |
| F-3 | B-031 | `i_cir after call = [1 2]`, `snap after call = [2 3]` (Octave, R2024a) | fixed |
| F-4 | B-032 | `Number of snapshots in no_interact must match coefficients.` (Octave, R2024a), no dump | fixed |
| F-5 | B-033 | `Cell element 1 of 'obj_names' is not a string.` (Octave, R2026a), no dump | fixed |
| F-6 | B-034 | C++ `Input 'n_div' is too large.`; Octave `Input 'M' must be an integer ...`; R2024a: `Invalid MEX-file` (list item 1); R2026a (extra run): `Input 'M' must be an integer from 0 to 4294967295.`, no dump; Octave `element=1.7`, `icosphere(-1)`, `icosphere(NaN)` rejected | fixed |
| F-7 | B-085 | `fast_sincos(x, []) = sin(x)` | fixed |
| F-8 | B-086 | `center_freq read back = 2.99792e+08` (299792458) | fixed |
| F-9 | B-087 | default = 120 deg (`max diff 0`), MATLAB and Python | fixed |
| G-1 | B-004 | `max abs(separate - combined) = 0`, n = 100000 return code 0; ASan module no report | fixed |
| G-10 | B-074 | `use_kernel` 3, 99, -1: `ValueError` | fixed |
| G-11 | B-095 | lod 7 `coord (3, 9, 1)`; lod 8 / -1 rejected; empty input rejected (docs say 0-7) | fixed |
| G-12 | B-116 | `[2 1 1 1] int64`, declared `py::array_t<py::ssize_t>` | fixed |
| G-13 | B-117 | dtypes int64, 3-D speaker fields; docs give these | documented |
| G-14 | B-105 | ASan module: `ValueError Delay and power vectors must not be empty.`, no report | fixed |
| G-15 | B-118 | 1-element `center_freq`: no warning; 2 elements: clear `ValueError` | fixed |
| G-2 | B-035 | `local_angles only` = `dist + local_angles` values | fixed |
| G-3 | B-005 | strided arrays read back as expected; ASan module no report | fixed |
| G-4 | B-036 | `clean exception: ValueError ...`, `still alive` | fixed |
| G-5 | B-096 | `clean exception: MemoryError` | fixed |
| G-6 | B-097 | scalars, 0-d, big-endian, bool written and read back | fixed |
| G-7 | B-098 | `(True, True, True, True)` | fixed |
| G-8 | B-099 | the old kwarg names raise `TypeError`; docs follow the registered names | documented |
| G-9 | B-100 | 2-D position: `ValueError: Input 'tx_pos' has incorrect number of elements.` | fixed |

MATLAB repros of B-029 / B-032 / B-033 / B-034: no crash and no crash dump from the library's MEX files (F-1, F-4, F-5,
F-6 incl. the extra R2026a run). The only new dump of the whole session, `~/matlab_crash_dump.2275384-1` (13:07), comes
from F-1's deliberate control (the what-if MEX linked without `-z nodelete`, expected rc=137). It was left in place.

## Tool versions

Same as the Baseline section.

| Tool | Version |
|---|---|
| g++ | 12.5.0 (Ubuntu 12.5.0-9ubuntu1); 15.2.0 (Ubuntu 15.2.0-16ubuntu1) |
| clang / afl-clang-fast | clang 21.1.8 (6ubuntu1); AFL++ 4.33c (afl-fuzz++4.33c, afl-cc++4.33c) |
| cmake | 4.2.3 |
| Armadillo (system, `/usr/include`) | 15.2.1 |
| HDF5 (system, serial) | 1.14.6 |
| Octave | 11.1.0 |
| MATLAB | R2026a 26.1.0.3203278 (`matlab`); R2024a 24.1.0.3050227 Update 8 (`matlab24`) |
| Python / numpy / pytest | 3.14.4 / 2.3.5 / 9.0.2 |

## Did not meet the expectation

1. **MATLAB R2024a cannot load 16 MEX files** (MEX suite expectation "all but the excluded test" not met in R2024a:
   50 of 68).
   - Error: `Invalid MEX-file '.../+quadriga_lib/arrayant_generate.mexa64': /usr/local/MATLAB/R2024a/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`.
   - Cause: `std::to_chars` for float / double in `src/qd_arrayant_qdant.cpp:414` and `:478` (added in d0f4a31,
     WP-10, the qdant writer for B-012 / B-038). `objdump -T arrayant_generate.mexa64` shows
     `_ZSt8to_charsPcS_f` and `_ZSt8to_charsPcS_dSt12chars_formati` at `GLIBCXX_3.4.29`. R2024a's bundled libstdc++
     ends at `GLIBCXX_3.4.28`; R2026a's has 3.4.30, Octave uses the system library.
   - Affected (every MEX file that links the arrayant code): `arrayant_calc_beamwidth`, `arrayant_calc_directivity`,
     `arrayant_combine_pattern`, `arrayant_concat`, `arrayant_copy_element`, `arrayant_export_obj_file`,
     `arrayant_generate`, `arrayant_interpolate`, `arrayant_qdant_read`, `arrayant_qdant_write`,
     `arrayant_rotate_pattern`, `get_channels_ieee_indoor`, `get_channels_irs`, `get_channels_multifreq`,
     `get_channels_planar`, `get_channels_spherical`.
   - Since when: the previous build (`build_linux_pre17b`) has the same symbols. The last R2024a suite run in the logs
     is from WP-06, before WP-10; later packages ran only R2026a and Octave, so this went unnoticed.
   - The code is otherwise fine in R2024a: with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` the suite gives
     67 of 68 (only the excluded failure).
   - The project already avoids these overloads elsewhere: `src/obj_file_write.cpp:645` says "snprintf-based to avoid
     the std::to_chars float overloads (GLIBCXX_3.4.29 / GCC 11)". A fix would do the same in
     `qd_arrayant_qdant.cpp`. Not changed here (report only).
   - Evidence: `mex_r2024a.txt`, `mex_r2024a_preload.txt`, `repro/F-6.txt` (the F-6 MATLAB part runs in R2024a and
     hits the same error).

2. **One HDF5 fuzz input takes 17 s outside AFL** (AFL acceptance "no input slower than 2 s" not met).
   - Input: `afl/out/hdf5/cmp/hangs/id:000000` (copy `afl/slow/hang0.bin`, 10,121 bytes); its sibling
     `cmp/hangs/id:000001` takes 1.15 s. No crash, no sanitizer report, exit code 0.
   - Release harness (`afl/slow/fz_release`, built against `build_linux`): 16.7 s wall, 2.7 s user, 14.0 s sys,
     4.2 GB peak RSS.
   - Where the time goes (all inside HDF5 1.14.6, not in quadriga-lib):
     - Stack samples at 2, 7 and 12 s are all in `memset` <- `H5FD_read` <- `H5F__accum_read` <- ... <-
       `H5F__super_read` <- `H5F_open` <- `H5Fopen` <- `qHDF_open_file` (`afl/slow/sample.sh`).
     - A bare `H5Fopen` + `H5Fclose` without quadriga-lib (`afl/slow/h5open_only.c`) takes 1.81 s and 4.2 GB on this
       input, and 1.85 s / 4.2 GB on B-119's `h5fopen_zero_fill.bin`.
     - Each of the 9 library calls the harness makes (`hdf5_read_layout`, 3 x `hdf5_read_channel`,
       3 x `hdf5_read_dset_names`, 2 x `hdf5_read_dset`) takes 1.80-1.87 s, the cost of one `H5Fopen`
       (`afl/slow/time_calls_hang0.txt`); 9 x 1.8 s = the 17 s.
   - This is the known B-119 behaviour (HDF5 zero-fills about 4 GB in `H5Fopen` for a crafted superblock; upstream,
     won't fix), found again by the new campaign. The phase-0 HDF5 inputs and the B-119 repro behave as recorded. No
     quadriga-lib defect, but the acceptance criterion as written is not met.

Notes (no defect, expectation wording only):
- The excluded UBSan reports come via `test_wave_progress.cpp:299`, not `:301` as in the brief; same reports and
  source lines as in the WP-16 and WP-17a logs.
- F-1 left one MATLAB crash dump on purpose (its old-link control), see "Repros".

## Round 2 (2026-10-22): fixes R1-R6, re-verification

Start: `fixes` at 4302d4e (main merged by the manager); the round-1 test fix and this log were kept; the manager's
uncommitted edit of `md/bug_fix_plan.md` was not touched. Decisions (Stephan, 2026-10-22): MATLAB R2024a is a required
target; non-finite CSV values are rejected; `thk` goes in the canonical CSV column order with default 0. Outputs in
`build_linux/scratch/fix/WP-17b/r2/`. Nothing committed.

Result: R1-R6 done. Round-1 list item 1 (R2024a) is fixed: all MEX files load in R2024a, 67 of 68. Written qdant files
are byte-identical to the old writer's, and writing is 25 % faster. Every gate met; one deviation: the AFL harness TUs
of obj / mtl / csv are built at `-O0`, not `-O1` (see "AFL").

### Changes

| Item | Change | Files |
|---|---|---|
| R1 | `std::to_chars` replaced. Shortest float text: `%.*e` digits (the correctly rounded ones; a neighbour only where they do not read back with `strtof`), then fixed or scientific, whichever is shorter (fixed on a tie), integers in fixed form written exactly, as libstdc++ does. Fixed decimals: exact rounding of `value * 10^k` (product and `fma` error term, ties to even) printed with integer arithmetic, `snprintf("%.*f")` as fallback for large values. Output always uses `.`, whatever `LC_NUMERIC` says (`snprintf` / `strtof` follow the locale, `to_chars` did not). `<charconv>` include dropped | `src/qd_arrayant_qdant.cpp` |
| R2 | `mtl_ind` comment: "1-based material index, 0 = no material" | `include/quadriga_tools.hpp:114` |
| R3 | `.mtl` reader strips leading spaces and tabs from every line before the keyword match (also for `newmtl`) | `src/obj_file_read.cpp` |
| R4 | CSV cell: `std::stod(v, &n_read)`; the whole cell must be read and the value finite, otherwise the existing "... is not numeric ('...')" error. The doc blocks (C++, MATLAB, Python) do not describe the cell rule (they link to Data Formats), so only the parser comment changed | `src/obj_file_read.cpp` |
| R5 | `kOrder` gets `thk` after `h` (position of the column table of `format_materials.md`), default 0; doc blocks (C++, MATLAB, Python) now list `side` = 3.29 among the defaults | `src/obj_file_write.cpp`, `api_mex/obj_file_write.cpp`, `api_python/qpy_obj_file_write.cpp` |
| R6 | Material-model link text replaced by the two-line wording of `src/obj_file_read.cpp:455-456` (doc blocks only) | `api_mex/ray_mesh_interact.cpp`, `api_mex/calc_diffraction_gain.cpp`, `api_python/qpy_calc_diffraction_gain.cpp`, `api_python/qpy_obj_file_write.cpp`, `api_python/qpy_mitsuba_xml_file_write.cpp` |

Tests (Catch2 `test_bin` +3 cases, +24 assertions):
- `test_obj_file_read.cpp`: "Indented .mtl statements" (`    Kd 1 0 0` gives Kd = 1 0 0, `\td 0.5` gives alpha 0.5,
  an indented `newmtl` is found); "CSV cells must be one finite number" (`1.5x`, `nan`, `inf`, `-inf`, `NaN`, `1.5 2`,
  `0x` throw "column 'a' is not numeric"; ` 2.5 `, `1e3`, `-0.5` and an empty cell are accepted).
- `test_obj_file_write.cpp`: the pinned header / default row now contain `thk` after `h`; new section "thk is
  written in the canonical position" (header `name,h,thk,att`, round trip through `obj_file_read`).
- `test_qdant_write.cpp`: "Number text and locale": the `ElementPosition` text of values where `%g` and `to_chars`
  differ (`1200,-0,123456 1e-04,1e-05,0.1 2.5e-07,1e+15,16777216`), and the same file bytes under a decimal-comma
  locale if one is installed (otherwise a `WARN` and that part is skipped; this machine has none, so it was run with a
  scratch `de_DE.UTF-8` built by `localedef` into `r2/locale/` and `LOCPATH`).
- `tests/mex_tests/test_obj_file_write.m`, `tests/python_tests/test_obj_file_write.py`: pinned column list and default
  row with `thk`.

Failing before the fix: the three changed Catch2 files built against a copy of the pre-change library
(`r2/bench/libquadriga_old.a`, 4302d4e; `r2/tests_before/before_old_lib_2.txt`): indented `.mtl` 2 of 4 assertions
fail; CSV cells all 7 bad cells accepted (7 failures); column order 4 failures (`name,h,att,thk`). The qdant
number-text test passes there by design: it pins the text the old `to_chars` writer produced (2 assertions, with the
scratch locale).

### R1 evidence

- objdump gate: `objdump -T +quadriga_lib/*.mexa64 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -u` gives 3.4.9, 3.4.11,
  3.4.18, 3.4.19, 3.4.20, 3.4.21, 3.4.26; nothing above 3.4.28, no `to_chars` symbol. (The Python module references
  3.4.30 through other symbols; Python loads the system libstdc++, so this does not matter.)
- Byte identity of written files (`r2/qdant_hash/`): an `LD_PRELOAD` shim (`capture_shim.c`) copies every `.qdant`
  file when it is closed after writing; `capture.sh` runs the full Catch2 suite, the MEX suite (Octave), pytest,
  repros A-09 and A-10 and `probe.cpp` (edge values for both number paths: -0, nan, +-inf, subnormals, 3.4e38, values
  where `%g` and `to_chars` choose differently, patterns from 0 to -300 dB with rounding ties; double and float;
  3GPP, multibeam, 3-frequency speaker). Before: 317 files, two runs identical (order + SHA-256). After: the same 317
  files byte-identical with the final code (`diff_before_after2.txt` empty), plus one file of the new test.
- Exhaustive check against `std::to_chars` (`r2/fmt_check/`; the helpers are copied verbatim from the source by `sed`):
  all 4,294,967,296 float bit patterns, 0 mismatches (`floats_all_2.txt`, 1553 s at 12 threads). The first version
  had 253 million mismatches (`floats_all.txt`): exact ties (2^-12 printed `...063` instead of `...062`, a double
  rounding in my distance test) and integers in fixed form (33554448 printed as `33554450`; libstdc++ writes the
  exact integer). Both rules were fixed before the second sweep. Fixed decimals: 315 million values (random dB levels
  and phases, values near decimal ties, exact binary ties, random doubles; decimals 0-6), 0 mismatches in the C locale
  and under `de_DE.UTF-8`; float ranges under `de_DE.UTF-8` (67 million values), 0 mismatches.
- Write time (`r2/bench/`, `bench_write.cpp`, fastest of 5 writes, 7 alternating runs pinned to idle core 11,
  `flock /tmp/bench.lock`, load 3.8 before / 3.2 after; all runs write identical bytes):

| File | Old (`to_chars`) | New, first version (`snprintf` per value) | New, final |
|---|---|---|---|
| 3GPP 4x4 dual-pol, 32 elements, 181 x 361 (52 MB) | 0.436-0.440 s | 1.240-1.284 s (2.8x slower) | 0.328-0.330 s (0.75x) |
| speaker, 3 frequencies (2 MB) | 0.0278-0.0281 s | 0.068-0.070 s | 0.0227-0.0228 s (0.81x) |

  Old vs old spread below 1 %. The first version was 2.8 times slower (glibc `printf` per pattern value), hence the
  integer fast path of the fixed decimals; it is the shipped version.

### Builds

| Build | Kind | Time | Warnings |
|---|---|---|---|
| `build_linux` + install, at 4302d4e (for the "before" capture) | incremental | - | 0 |
| `build_linux` + install, final code | incremental | 71 s, then 49 s after the fast path | 0 |
| `build_generic`, final code | incremental | 71 s, then 8 s | 0 |
| `build_asan`, final code | incremental | 579 s, then 34 s | 0 |
| `build_afl_plain`, `build_afl_cmplog` | incremental (48 files each) | 23 s, 15 s | 0 |

### Suites (final code)

| Run | Result | Wall | Load before / after |
|---|---|---|---|
| base `build_linux/test_bin` | `All tests passed (1348780 assertions in 1303 test cases)` (round 1: 1348756 / 1300; +3 cases) | 20 s | 3.4 / 3.1 |
| generic | `All tests passed (1346857 assertions in 1302 test cases)` (1346833 / 1299) | 19 s | 3.1 / 3.4 |
| `[wave]` base / generic | `All tests passed (163474 assertions in 178 test cases)` in both | 5 s | 3.4 / 3.3 |
| `[acoustic]` base / generic | `All tests passed (42741 assertions in 59 test cases)` in both | 1 s | 3.3 / 3.4 |
| ASan + UBSan gcc-12 (`suppress_equal_pcs=0`, leak check) | `All tests passed (1348780 assertions in 1303 test cases)`; only the 6 excluded `ray_progress` reports (`ray_progress.cpp:829-831` via `test_wave_progress.cpp:299`), no ASan report, no leak | 131 s | 3.0 / 3.0 |
| MEX Octave 11.1 | `FAILED (passed=67, failure=1)`, only `test_ray_mesh_interact:113` | 1 s | 2.4 |
| MEX R2024a (no preload) | `FAILED (passed=67, failure=1)`, only the excluded failure; no `Invalid MEX-file` | 7 s | 2.4 |
| MEX R2026a | `FAILED (passed=67, failure=1)`, only the excluded failure; exit code 0, no new crash dump | 11 s | 2.4 |
| pytest | `817 passed, 1 skipped, 52 subtests passed` (skip: CUDA) | 3 s | 2.5 / 2.4 |

### AFL (obj, mtl, csv)

Folder `r2/afl/`: harnesses copied from round 1, `common.hpp` with `r2/afl/work`, the round-1 seeds plus
`mtl/indent.mtl` (indented statements, tabs) and `csv/edge.csv` (exponents, blanks, empty cells, a `thk` column).

Deviation: harness TUs at `-O0`, not `-O1`. With the round-2 reader, all three ASan harnesses built at `-O1` crash on
every valid seed (exit 1 after 90 s in ASan's abort path): SEGV in AFL's own coverage-counter increment
(`movzbl (%rcx,%rax,1); inc`, `rcx = __afl_area_initial`, `rax = 0x556d2a60`, a garbage guard index) inside
`arma::Mat<double>::init_warm` called from `obj_file_read.cpp:644`, the signature of the WP-14 note (0x556d2aa0 there).
Controls (`r2/afl/dbg/`): the same harnesses built by gcc-12 with ASan + UBSan against `build_asan` and by
afl-clang-fast at `-O0` run every seed clean; an `-O1` harness with `obj_file_read.cpp` of 4302d4e (compiled with the
library's flags) is clean, with the current file it crashes. R3 / R4 only shifted the code layout that triggers the
toolchain fault; no library defect. (The obj CMPLOG twin built at `-O1` ran its seeds clean; all six binaries of the
campaign are `-O0`.)

Campaign 15:33-15:48, 900 s per instance, ASan main + CMPLOG per reader (6 processes), load 2.1 before / 10.7 after
(other sessions). Replay of every queue, crash and hang input with the ASan harnesses (`replay_*.txt`), load 9.4 / 4.5:

| Reader | Executions (main + cmp) | Edges (main) | Crashes | Hangs | Inputs replayed | Slowest replayed input |
|---|---|---|---|---|---|---|
| obj | 259,367 + 767,495 = 1,026,862 | 736 | 0 | 0 | 1252 | 0.031 s |
| mtl | 121,580 + 651,959 = 773,539 | 730 | 0 | 0 | 647 | 0.037 s |
| csv | 123,098 + 696,340 = 819,438 | 935 | 0 | 0 | 939 | 0.031 s |

All replayed inputs: exit code 0, no sanitizer report.

### Sentences of `tools/data_formats_md/format_materials.md` that R4 / R5 make inaccurate (file not edited)

- Line 35: "A cell that does not start with a number is an error." After R4 a cell must be one finite number in full:
  `1.5x`, `1.5 2`, `nan`, `inf` are errors too.
- R5: no sentence becomes inaccurate (line 39 says only that `obj_file_write` writes the table "in the same form").
  Observation: the column table starts with `fRef`, while `kOrder` keeps `fRef` after `alphaB`; R5 asked only for
  `thk`, so `fRef` was left where it was.

### Did not meet the expectation (round 2)

None. The AFL harness optimisation level differs from the brief (`-O0` instead of `-O1`), for the toolchain reason
above. Round-1 list item 2 (the HDF5 input that spends 17 s in HDF5's `H5Fopen`, B-119) is unchanged; HDF5 was not
part of round 2.
