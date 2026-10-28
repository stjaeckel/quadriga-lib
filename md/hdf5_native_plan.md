# Project plan: native HDF5 I/O for quadriga-lib (file format v3)

Branch `hdf5-native` (from v0.12.4, `26da9f93`), worktree `/home/stephan/quadriga-lib-hdf5` of `/sjc/quadriga-lib`.
Local only: nothing is pushed, nothing is merged. The user reviews and merges.

## 1. Goal

Replace the libhdf5 dependency of quadriga-lib with a lean, header-only C++ implementation (`src/qd_hdf5.hpp`)
that writes and reads the quadriga-lib channel file format directly as binary files.

- Files written by quadriga-lib stay valid HDF5 files: libhdf5 (1.8 and newer), h5py, MATLAB and HDFView can read
  them and see the same tree as today.
- The quadriga-lib file version goes from 2 to 3. v2 files are no longer supported (clear error on read).
- The reader only has to read files written by `qd_hdf5.hpp`. It is optimized for that format and does not handle
  generic HDF5 files.
- libhdf5 is removed from the library, the MATLAB/Octave MEX files, the Python module, the build system,
  packaging and docs. It stays only as an optional test dependency.

## 2. Decisions (settled with the user, 2026-10-28)

| ID  | Topic                | Decision |
|-----|----------------------|----------|
| D1  | Logical schema       | Keep the v2 tree: root datasets `ChannelDims`, `Order`, `Version`; one group `channel_<id>` per written slot with the v2 attributes; `Snap_<i>` groups for multi-snapshot channels; group `Data` for unstructured `par_*` datasets. Same names, shapes, element types as v2. The per-object metadata overhead (~1.2–1.5 KB per snapshot) is accepted. |
| D2  | Version              | Root dataset `Version` = 3. Reading a file with another version (or no version) throws `std::invalid_argument` with a clear message. |
| D3  | Slot lookup          | New root dataset `ChannelRef`: one HDF5 object reference (8 bytes, H5R_OBJECT, i.e. the object-header address) per slot, parallel to `Order`; empty slot = reference to address 0 (null reference as h5py writes it — verify in the spike, P2.1). The reader seeks straight to the channel group; it never scans root links. `Order` and channel IDs keep their v2 meaning (`hdf5_read_layout` unchanged). |
| D4  | Reader scope         | Only files written by `qd_hdf5.hpp`. v2 files and files changed by other tools (h5py/MATLAB edits, `h5repack` output, variable-length strings, chunked data, dense link storage) are rejected with a clear `std::invalid_argument`. |
| D5  | Compatibility floor  | HDF5 1.8+: superblock version 2, version-2 object headers with Jenkins lookup3 checksums, compact (message-based) link storage, link creation order tracked in `Data` groups. |
| D6  | Hardening            | Same standard as v2: every size, count and address read from the file is checked against the file size before anything is allocated; metadata checksums are verified on read; malformed files throw `std::invalid_argument`; failing writes throw `std::runtime_error`. |
| D7  | Dead space           | Overwriting a slot leaves the old data in the file. No compaction, no free-space reuse. The docs drop the `h5repack` hint (the reader won't read its output) and say: copy the channels into a new file to reclaim space. |
| D8  | Concurrency          | Single writer per file, no file locking; documented. Readers are safe while nobody writes. No global state in `qd_hdf5.hpp`, so reading different files from parallel threads is safe. |
| D9  | `get_HDF5_version`   | Removed everywhere: C++ (`quadriga_channel.hpp`), MATLAB/Octave (`api_mex/hdf5_version.cpp`), Python (`components()` output), their tests and docs. |
| D10 | libhdf5 interop test | Optional Catch2 test, compiled only when CMake finds a system libhdf5 (`find_package(HDF5 COMPONENTS C)`); links libhdf5 into `test_bin` only. It opens files written by the library with libhdf5 and compares the full tree and data. Installed here: libhdf5-dev 1.14.6. h5py is not used. |
| D11 | Purge scope          | Everything: CMake (`HDF5_PATH`, `HDF5_STATIC`, ExternalProject, find_package for the library, install of `libhdf5`, the `-z nodelete` MEX workaround if only HDF5 needs it — verify with MATLAB R2026a `clear mex` + exit), `GNUmakefile`, Windows `Makefile`, `external/hdf5-1.14.6.zip`, `Dockerfile.ubuntu`, `debian/*`, `pyproject.toml`, README files, install docs in `tools/download_md/`, and code comments that mention libhdf5 behavior. |
| D12 | Spec review stops    | The manager acts as reviewer at review points A, B and C of `md/SPEC_CPP_API.md` and logs each decision. It stops for the user only on the escalation conditions in §8. |
| D13 | Commits              | Four milestone commits on `hdf5-native`: (1) this plan, (2) extended tests with `src/` unchanged, (3) the spec after its three reviews, (4) implementation, build purge and docs after the three code reviews. Nothing is pushed or merged. |
| D14 | Test gates           | Before commits 2 and 4, all suites must be green: Catch2, Python (pytest), MATLAB R2024a, MATLAB R2026a, Octave. |
| D15 | Benchmark            | Two mixes of ~5 GB each, v2 (libhdf5, v0.12.4) against v3: (a) MIMO: ~2000 channels × 10 snapshots × 16×4 antennas × 200 paths; (b) ray-tracing route: ~20k channels × 100 snapshots × SISO × 20 paths with all path fields. Write and read times are measured. |

## 3. Current state (v0.12.4)

- `src/qd_channel_hdf5.cpp` (1914 lines): all HDF5 code, using the libhdf5 C API. Public functions, declared in
  `include/quadriga_channel.hpp`: `get_HDF5_version`, `hdf5_create`, `hdf5_write<dtype>`, `hdf5_read_layout`,
  `hdf5_reshape_layout`, `hdf5_read_channel<dtype>`, `hdf5_read_dset`, `hdf5_read_dset_names`, `hdf5_write_dset`.
  `any_type_id` (`src/any_type_id.cpp`) is used for the unstructured data types and stays unchanged.
- Wrappers: `api_mex/hdf5_*.cpp` (9 files incl. `hdf5_version.cpp`), `api_python/qpy_hdf5_*.cpp` (8 files),
  `api_python/qpy_components.cpp` (prints the HDF5 version), `api_python/python_channel.cpp`, adapters.
- Tests: `tests/catch2_tests/test_hdf_functions.cpp` (one large test case), `tests/catch2_tests/test_hdf_robustness.cpp`
  (patches files with the libhdf5 C API to build malformed inputs; also checks side effects on a caller's libhdf5
  state and strings written by other tools), `tests/python_tests/test_hdf5_functions.py`, `tests/mex_tests/test_hdf_rw.m`,
  `tests/test_hdf_rw.py` (outside the suites), version tests in all three languages.
- Known v2 costs: every `hdf5_write` reads and rewrites the whole `Order` index (256 KB with the default 65536 slots,
  up to 16 GB at the 2^32 limit); libhdf5 B-tree and metadata-cache overhead; one HDF5 open/close per call.

## 4. Target design (to be detailed in the spec)

### 4.1 Logical schema v3
Identical to v2 (D1) plus `ChannelRef` (D3) and `Version = 3` (D2). The spec lists every object with name, HDF5
class, element type, dataspace and when it is present. This includes the v2 conventions the tests pin down: NaN
defaults for `CenterFrequency`/positions, zero orientations, the 256-byte NUL-terminated `Name`, empty snapshots
stored with extent-0 datasets, scalar vs. rank-1 vs. rank-2/3 unstructured data, an empty string stored as one NUL
byte, and `Row` read back as `Col`.

### 4.2 On-disk structures (HDF5 File Format Specification, version 3.0 document, published by The HDF Group)
Only this subset is implemented:
- Superblock version 2 (with checksum, EOF address, root object header address).
- Version-2 object headers ("OHDR"), continuation blocks ("OCHK"), Jenkins lookup3 checksum.
- Messages: NIL (padding/slack), Dataspace, Link Info, Datatype (fixed-point, IEEE float, fixed-length string,
  object reference), Fill Value, Link, Data Layout (compact and contiguous), Group Info, Attribute, Object Header
  Continuation. Others are not needed.
- Attributes are stored in the object header (compact). Unstructured `par_*` datasets keep creation order
  (link messages carry creation-order values; Link Info flags creation-order tracking).

### 4.3 Write strategy (performance)
- Append-only: all objects of a channel (group header, snapshot groups, datasets, attributes, data) are built in
  one memory buffer and written with one `fwrite` at the end of the file. Small datasets use compact layout (data
  inside the object header); larger ones are contiguous and follow their header.
- In-place updates after the append, in an order that keeps the file consistent if the write is interrupted:
  the new link in the parent group, the `Order` and `ChannelRef` entries of the slot (a few bytes, not the whole
  index), then the superblock EOF address and checksum.
- Growable link lists (root group, `Data` groups): object-header chunks get reserved NIL-message slack. When a chunk
  is full, a new continuation chunk with geometrically growing capacity is appended and linked from slack reserved
  in the previous chunk. Changing a chunk means rewriting only that chunk and its checksum. How libhdf5 deals with
  more compact links than the Group Info "max compact" value (16-bit field) is the main technical risk. The spike
  (P2.1) settles it at 10^6 links. The fallback is an old-style symbol-table root group (v1 B-tree + local heap with
  zero-padded, append-ordered names), which is valid inside a v2 object header.
- Overwrite: the parent's link message for the old group becomes a NIL message in place (D7: no space reuse).
- `hdf5_reshape_layout`: rewrite the `ChannelDims` data in place.
- Read: open the file, verify superblock + `Version`, read one `Order`/`ChannelRef` entry, parse the channel's object
  headers (adjacent in the file), and read the data straight into Armadillo memory. Float→double conversion uses one
  buffer.

### 4.4 Code structure
- `src/qd_hdf5.hpp`: private, header-only, not installed. Self-documenting: a file header explains purpose, the
  supported HDF5 subset, and the published specification it follows. Minimal interface (file handle with 64-bit
  I/O, object-header builder, message encoders/decoders, checksum, reader for the subset). Everything in a
  namespace; helpers used once are lambdas, file-local helpers are `static inline` or in an anonymous namespace
  (user rules, §9).
- `src/qd_channel_hdf5.cpp`: rewritten on top of `qd_hdf5.hpp`; keeps all public functions except
  `get_HDF5_version`; keeps the `/*!MD ... MD!*/` documentation blocks, updated for v3.
- Portability: C++17, compiles with GCC, Clang and MSVC (no `__builtin_*`, no POSIX-only calls; 64-bit file offsets
  via `_fseeki64`/`_ftelli64` on MSVC; little-endian hosts only, checked at compile or start time). `-Wall -Wextra
  -Wpedantic` clean. No BLAS/LAPACK.

## 5. Phases

Each phase has an owner (sub-agent, effort high, unless noted), outputs, and a gate the manager checks before moving
on. Internal files (briefs, reports, logs, spikes, benchmark data) go to `md/hdf5_internal/` (excluded from git via
`.git/info/exclude`) or outside the repo; they are never committed.

### P0 — Setup (done in the planning session)
- Worktree and branch created. `md/hdf5_internal/` excluded from git.
- Gate: the user approved this plan → **commit 1** (this file only).

### P1 — Extend the tests (src unchanged)
Goal: every regression of the public HDF5 functions is caught, independent of the on-disk version.
- Inventory every public HDF5 function, every documented behavior (the MD blocks in `qd_channel_hdf5.cpp` and the
  header comments) and every branch of `qd_channel_hdf5.cpp`. Write a coverage matrix (internal) mapping each item to
  a test. Measure line coverage of `qd_channel_hdf5.cpp` with gcov/lcov (internal build dir) and close gaps that are
  reachable through the public API.
- Catch2: split the single large test case in `test_hdf_functions.cpp` into focused test cases (`"hdf5_create - ..."`,
  `"hdf5_write - ..."` etc.) and add the missing ones. Cover round trips for float and double, all optional channel
  fields, empty snapshots, mixed delay shapes, all unstructured types and ranks, names (255-char limit), slot
  overwrite (return value, `par_*` removal), `hdf5_write_dset` on empty and filled slots, duplicates, reshape,
  out-of-range indices, layout limits, missing files, `assume_valid`, and error types.
- Format-specific tests (anything that patches files with libhdf5, foreign strings, caller's libhdf5 state) stay in
  `test_hdf_robustness.cpp` and are listed in the coverage matrix as "v2-specific, replaced in P4".
- Add the optional libhdf5 interop test (D10) and its CMake hook. It checks the full tree written by the library
  against the channel objects, written so that it passes on v2 now and on v3 later (`Version` and `ChannelRef` are
  checked conditionally on the version found).
- Python (`test_hdf5_functions.py`) and MATLAB/Octave (`test_hdf_rw.m`): cover each wrapper's argument handling,
  defaults and errors; core behavior stays in Catch2. MATLAB tests must also pass in Octave.
- Benchmark program `tests/benchmarks_cpp/benchmark_hdf5.cpp` (committed): generates the two D15 mixes with library
  functions, measures `hdf5_write` and `hdf5_read_channel` over all slots, and drops the page cache for the file before
  reading (`posix_fadvise(POSIX_FADV_DONTNEED)` on Linux). Run the v2 baseline now and save it to the internal folder
  (see §7 for the measurement rules).
- Rules: tests use the library API (`html_docu/quadriga_lib_api_cpp.md`) for fixtures instead of hand-written code;
  only the functions under test are not used as helpers. `md/SPEC_CPP_API.md` "TESTS" section applies. No bloat.
- Gate: all five suites green on the unchanged v2 code (D14); `git diff 26da9f93 -- src include api_mex api_python`
  is empty; review point B decided by the manager → **commit 2**.

### P2 — Spike and specification
- P2.1 Spike (internal, not committed): small C++ prototypes that write the key structures and read them back with
  libhdf5 1.14.6. They settle: compact link storage with >65535 links in the root group and in a `Data` group (read
  correctness and libhdf5 open/lookup time), growing continuation chunks, the null object reference, compact vs.
  contiguous dataset thresholds, attribute message version, fill-value message requirements, and the exact
  superblock/EOF update order. Results go into the spec.
- P2.2 Spec `md/hdf5_v3_spec.md`, complete enough to implement without the plan. It contains:
  logical schema v3 (every object); the byte layout of every structure written, with field values; the reader's
  validation rules and error messages; the algorithm of each public function (create, write, overwrite, write_dset,
  reshape, read_layout, read_channel, read_dset, read_dset_names), with the in-place update order; the
  `qd_hdf5.hpp` interface (names, responsibilities, ownership); portability rules (MSVC, endianness, 64-bit offsets);
  the new robustness tests (byte-level corruption of v3 files, replacing the libhdf5-based ones); the exact list of
  build-system, packaging, wrapper and doc changes (D9, D11); the public API doc changes (MD blocks, header comments,
  D7, D8); the benchmark protocol.
- References in the spec may point to the published HDF5 File Format Specification and this plan's decision IDs.

### P3 — Three independent spec reviews
- Three fresh reviewer agents (high effort), run one after another. Each gets the spec, this plan, the v0.12.4 sources
  and the published HDF5 format spec, but not the earlier review reports. Suggested focus (each still reviews
  everything): R1 HDF5 conformance (byte layouts, checksums, message versions vs. the published spec and libhdf5
  1.14 behavior); R2 API behavior parity with v2 and the P1 tests, error handling, hardening; R3 implementability,
  leanness, MSVC/portability, performance of the write/read paths.
- After each review the manager (or a spec agent) fixes the spec and logs what was accepted or rejected and why.
- Gate: no open blocking finding → **commit 3** (spec, plus plan updates if any).

### P4 — Implementation
The manager decides the number of agents and the order. Suggested order, with parallel work only where the files
do not overlap:
1. `src/qd_hdf5.hpp` with its own focused Catch2 tests (allowed: tests may include internal headers), checked with
   the libhdf5 interop test as soon as files can be written.
2. Rewrite `src/qd_channel_hdf5.cpp` on top of it; all P1 API tests must pass unchanged, except the tests listed as
   v2-specific (D4) and the version tests (D9).
3. Replace `test_hdf_robustness.cpp` with v3 byte-level corruption tests (no libhdf5). Move the interop tests there
   or into their own file.
4. Remove `get_HDF5_version` (D9) across C++, MEX, Python, tests and docs.
5. Build-system and packaging purge (D11); README/install docs; remove `external/hdf5-1.14.6.zip`.
6. Update the MD documentation blocks and the header comments (v3, D7, D8, D4); regenerate the docs with
   `make documentation` and check that `html_docu/quadriga_lib_api_*.md` reflect the change.
- Changes to P1 tests are allowed only for the documented v2-specific items. Any other change needs a logged
  justification and must not weaken a check.
- Gate: all five suites green; the libhdf5 interop test green; a build without libhdf5 installed works (check with a
  CMake configure that hides HDF5, e.g. `-DCMAKE_DISABLE_FIND_PACKAGE_HDF5=ON`); no compiler warnings.

### P5 — Three independent code reviews
- Three fresh reviewer agents (high effort), one after another, each with the full diff against `26da9f93`, the spec
  and the review checklist (§9). Suggested focus: C1 correctness, HDF5 conformance and hardening (incl. fuzz-style
  corruption of v3 files); C2 API, docs, leanness and naming against §9; C3 portability (MSVC reading of the code,
  64-bit offsets, endianness), build/packaging purge completeness, and performance.
- The manager has findings fixed after each review and re-runs the gates.

### P6 — Benchmark and final commit
- Run the D15 benchmark on v3 with the same binary settings as the v2 baseline (§7). Report write/read throughput,
  file sizes, and the speedup per mix.
- Final check: §9 checklist, the five suites, the interop test, `git status` clean except intended changes, no file in
  `md/hdf5_internal/` tracked.
- **Commit 4** (implementation, tests, purge, docs, spec/plan updates). Do not push, do not merge.
- Write the final report (internal) and a short summary for the user: what changed, benchmark table, any deviations
  from this plan, open items.

## 6. Commands and environment
- Build and Catch2: `make test` (once), then `cmake --build build_linux --parallel` and `build_linux/test_bin "hdf5*"`.
  Run from the repository root.
- Python: `make python` then `python3 -m pytest tests/python_tests -x -s` (the module lands in `lib/`).
- MATLAB R2024a: `matlab24 -batch "run('tests/quadriga_lib_mex_tests.m');"`; R2026a: `matlab -batch ...`.
  The MEX files must be built for each MATLAB release (the GNUmakefile uses `MATLAB_BIN = matlab24`; for R2026a,
  configure a separate build dir or rebuild with the R2026a `mex`).
- Octave: `octave --eval "cd tests; quadriga_lib_mex_tests;"` (Octave builds are enabled once static libhdf5 is gone).
- Compiler: GNUmakefile pins `gcc_version = 12` (MATLAB limit); keep it.
- Shared machine (from the user's global instructions, pass on to every sub-agent): check `cat /proc/loadavg` and
  `pgrep -c -f claude` before heavy work; when the 1-min load is above ~16, size `-j`/`OMP_NUM_THREADS` so the total
  stays near 32; expect 3–5× slower runs under load; never kill or renice processes this session did not start.
  No GPU is needed.

## 7. Benchmark protocol
- Same compiler and flags for v2 and v3. v2 = build of `26da9f93` (with libhdf5 1.14.6), v3 = final branch.
- Files go to a disk directory outside the repo (e.g. `/home/stephan/hdf5_bench/`, ~10 GB free needed per run;
  1.7 TB free now), and are deleted afterwards. Not to `/tmp` (tmpfs, RAM).
- Run under `flock /tmp/bench.lock`. Record the load average before and after. If the load was above ~4, the numbers
  are provisional: retry later; report them only as provisional.
- Report: write time and throughput, read time and throughput (page cache dropped for the file before reading),
  file size, per mix and version.

## 8. Manager setup and escalation
- One long-running manager session at medium effort. It owns the plan, starts sub-agents at high effort for the work
  in P1–P6, checks the gates itself (runs the suites, reads the diffs), and makes the review-point decisions (D12).
- Manager memory: `md/hdf5_internal/manager_log.md` — current phase, decisions with reasons, gate results, open
  items, agent reports index. Updated after every step so the run can resume after a context compaction or restart.
- Sub-agent briefs are written to `md/hdf5_internal/briefs/`, reports to `md/hdf5_internal/reports/`. Each brief
  includes the relevant decisions, the §9 checklist, the machine-load rules (§6) and the paths.
- No GPU work; no parallel sub-agents that write the same files.
- Stop and ask the user only if:
  - a public API change beyond D9 seems necessary;
  - libhdf5 cannot read a structure that the design depends on, and neither the planned fallback nor a small
    deviation fixes it;
  - a P1 test would have to be weakened for a reason outside D4/D9;
  - the v3 benchmark is slower than v2 on either mix after one round of optimization;
  - a gate keeps failing after two fix rounds.

## 9. Review checklist (applies to every review and gate)
- References: produced source and test files must not reference temporary documents in `md/` (this plan, the spec,
  internal files). They stand on their own and may cite published material (e.g. the HDF5 File Format
  Specification) and the persistent `md/SPEC*.md` files. Internal remarks and tracking IDs (D1, R2, ...) only appear
  in the plan, the spec and internal files.
- No code bloat: reuse existing helpers; lean, readable structure. Given only the source files, a reader (human or
  AI) understands purpose and mechanics.
- Public API fits `html_docu/quadriga_lib_api_cpp.md`: no new public functions; docs say what is needed to use the
  functions correctly, not implementation details.
- Naming and scope: clear names. Helpers used once are lambdas in the calling function; helpers reused within one
  `.cpp` are `static inline` or in an anonymous namespace; helpers used across files go into a private header with
  its own documentation (`qd_hdf5.hpp`).
- Related public functions may share one `.cpp` (the read/write functions of the file format stay together in
  `qd_channel_hdf5.cpp`).
- BLAS/LAPACK disabled; C++17; MSVC-compatible (not available here, review by reading); no warnings with
  `-Wall -Wextra -Wpedantic`.
- Tests use the library API for fixtures (e.g. geometry helpers) instead of re-implementing them; only the functions
  under test are not used as helpers; no unnecessary bulk.
- `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md` are respected.

## 10. Risks
| Risk | Mitigation |
|------|------------|
| libhdf5 rejects or slowly reads very large compact link lists | Spike P2.1 at 10^6 links; fallback: symbol-table root group (v1 B-tree + local heap) inside the v2 object header. |
| Subtle conformance errors that libhdf5 tolerates but other readers don't | Interop test with libhdf5 on every test file; R1/C1 reviews against the published spec. |
| MSVC build breaks (not testable here) | Portable code rules (§4.4), C3 review; avoid compiler-specific builtins. |
| `-z nodelete` removal destabilizes MATLAB | Verify `clear mex` and MATLAB exit in R2024a and R2026a before removing; keep it if in doubt and log why. |
| Benchmark noise from other sessions | `flock`, load recording, provisional labeling (§7). |
| Behavior drift from v2 | P1 tests written against v2 first, then run unchanged on v3. |
