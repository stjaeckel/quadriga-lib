# HDF5 channel files without libhdf5 (file format version 3)

Maintainer notes on the native HDF5 I/O of quadriga-lib: what changed against quadriga-lib 0.12.4 (commit `26da9f93`, file format version 2, libhdf5), why, how the format and the code work, and what was measured. State: commit `e2e1a32` on top of `a719e43` (implementation) and `8e078a5` (extended tests). The code is the reference: byte layouts and constants in `src/qd_hdf5.hpp`, the schema and the public functions in `src/qd_channel_hdf5.cpp`. Byte layouts follow the *HDF5 File Format Specification Version 3.0* (The HDF Group); section names below are from that document.

## 1. Summary

- libhdf5 is gone from the library, the MEX files, the Python module, the build system and the packages. A private header-only layer, `src/qd_hdf5.hpp` (1667 lines), writes and reads the small HDF5 subset the channel files need; `src/qd_channel_hdf5.cpp` (1914 → 1493 lines) keeps all public functions except `get_HDF5_version`.
- The files stay ordinary HDF5 files with the v2 object tree, plus a root dataset `ChannelRef` (one object reference per slot) and `Version = 3`. Version-2 files and files modified with other HDF5 tools are rejected.
- Writes are append-only: one block at the end of the file, then a few small in-place updates in an order that keeps every intermediate state readable. A write no longer rewrites the whole `Order` index.
- Benchmark (5 GB files): writes 3.4× (MIMO) and 36× (ray-tracing route) faster, reads 1.8× and 24× faster; the route file is half the size (section 9).
- `tools/hdf5_convert_v2.py` converts version-2 files (h5py reads, the quadriga_lib Python module writes); the Python `hdf5_*dset` functions gained a `prefix` argument (section 10).

## 2. Motivation: costs of the libhdf5-based version 2

- **Dependency and packaging.** CMake carried `HDF5_PATH`, `HDF5_STATIC`, a bundled `external/hdf5-1.14.6.zip` (42 MB) built through `ExternalProject`, a prebuilt-HDF5 autodetect for MSVC, installation of `libhdf5`, and HDF5 link flags for every MEX file, the Python module and the tests. Debian, Docker, conda and the READMEs required `libhdf5-dev`/`hdf5`. A static libhdf5 forced Octave off ("version clash").
- **MATLAB.** MATLAB bundles its own libhdf5; the MEX files linked a differently named system or static copy. The thread-safe libhdf5 creates pthread keys with destructors that it never deletes, and MATLAB R2026a unloads MEX files on `clear mex` and at exit, so all MEX files were linked with `-Wl,-z,nodelete`. Without the flag, v2 MEX files still crash R2026a at exit (re-checked as a control, section 11).
- **Write cost.** Every `hdf5_write` read and rewrote the whole `Order` index (256 KB with the default 65536 slots, up to 16 GB at the 2^32 − 1 slot limit), and libhdf5 opened and closed the file, the B-trees and the metadata cache on every call. The ray-tracing mix was CPU-bound in libhdf5: ≈ 0.18 ms per snapshot written, 0.23 ms read, ≈ 1300 system calls per channel write and ≈ 2900 per read (section 9).
- **Metadata.** libhdf5 stored ≈ 4.9 KB of metadata per snapshot of a SISO channel with 20 paths (payload ≈ 2.3 KB); version 3 stores ≈ 1.2 KB.

## 3. File format version 3

### 3.1 Logical schema

The v2 tree plus `ChannelRef` and `Version = 3`. HDF5 extents are listed slowest dimension first: an Armadillo object `[rows, cols, slices]` has the extents `{slices, cols, rows}` and the same memory order.

| Object | Type, dataspace | Content |
|--------|-----------------|---------|
| `/ChannelDims` | uint32 `[4]`, contiguous | layout `{nx, ny, nz, nw}`, each ≥ 1, product `n_slot` ≤ 2^32 − 1 |
| `/Order` | uint32 `[n_slot]`, contiguous | channel ID per slot, 0 = empty (returned by `hdf5_read_layout`) |
| `/Version` | uint32 scalar, compact | 3 |
| `/ChannelRef` | object reference `[n_slot]`, contiguous | object-header address of the slot's channel group; 0 = empty |
| `/channel_<id>` | group | one per occupied slot; decimal ID without leading zeros |
| attributes of the channel group | scalar unless noted | `NumSnap`, `NumTx`, `NumRx` (uint32); `Initial_position` (int32); `CenterFrequency` (float32, NaN); `tx_position`, `rx_position` (float32 `[3]`, NaN); `tx_orientation`, `rx_orientation` (float32 `[3]`, 0); `Name` (fixed string of 256 bytes, NUL-terminated, ≤ 255 bytes of text); `NumPath` only in a 1-snapshot channel with path data |
| per-snapshot channel values | float32 `[n_snap]` / `[n_snap, 3]` | `CenterFrequency`, positions, orientations as datasets of the same name when they have `n_snap ≥ 2` values (the attribute keeps its default) |
| snapshot data | in the channel group (`n_snap = 1`) or in `Snap_0 … Snap_<n−1>` (each with `NumPath`) | `coeff_re`, `coeff_im` `{np, n_tx, n_rx}`; `delay` `{np, n_tx, n_rx}` or `{np, 1, 1}`; `path_gain`, `path_length` `{np}`; `path_polarization` `{np, 8}`; `path_angles` `{4, np}`; `path_fbs_pos`, `path_lbs_pos` `{np, 3}`; `no_interact` uint32 `{np}`; `interact_coord` `{sum(no_interact), 3}`; all float32 except `no_interact` |
| `/channel_<id>/Data` | group, creation order tracked and indexed | unstructured datasets `<prefix><name>` (default prefix `par_`) |

Conventions pinned by the tests (all unchanged from v2): if any per-snapshot field has `n_snap` entries, every snapshot is written, snapshots without paths as extent-0 datasets; a channel with only unstructured data, and every group created by `hdf5_write_dset`, keeps the default attributes and no name; a position matrix with 3 rows and a column count other than 1 or `n_snap` is silently not stored. `Data` datasets: `std::string` as a fixed-length string of `max(len, 1)` bytes (an empty string is one NUL byte), scalars scalar, `Col`/`Row` rank 1 (a `Row` comes back as `Col`), `Mat` `{cols, rows}`, `Cube` `{slices, cols, rows}`; element types float32, float64, uint64, int64, uint32, int32, raw bytes without conversion.

Slots: slot `(ix, iy, iz, iw)` = `ix + nx·(iy + ny·(iz + nz·iw))`. Channel IDs start at 1; a new ID is one above the ID of the last root link (the largest ever linked), and an overwrite gives the slot a new ID, as in v2. All reading functions find a slot through `ChannelRef` alone and never search the root links.

### 3.2 HDF5 subset

| Structure | Version, encoding |
|-----------|-------------------|
| Superblock | version 2, 48 bytes at offset 0: signature, version, offset/length size 8, consistency flags 0 (reader accepts 0 or 1: libhdf5 sets bit 0 while a file is open read-write and leaves it after a crash), base address 0, extension address undefined (`ff`×8; 0 makes libhdf5 fail), end-of-file (EOF) address, root header address 48, lookup3 checksum of bytes 0–43 |
| Object header | version 2: chunk 0 `OHDR` + version + flags + 1- or 2-byte body size; continuation chunks `OCHK`; every chunk ends with a lookup3 checksum; body ≤ 65535 bytes; flags only carry the size-field width (no times, no attribute creation order) |
| Messages | NIL, Dataspace v2 (no maximum extents), Link Info v0, Datatype v1, Fill Value v3, Link v1 (hard links, 1- or 2-byte name length), Data Layout v3 (compact or contiguous), Group Info v0, Attribute v3, Object Header Continuation, Attribute Info v0 |
| Datatypes | byte-identical to libhdf5 1.14.6 output: little-endian int32/uint32/int64/uint64, IEEE float32/float64, fixed-length ASCII string with NUL-terminated padding, object reference (8 bytes) |
| Message flags | as libhdf5 writes them: 0x01 (constant) on Datatype, Fill Value and Group Info, 0x04 on Attribute Info, 0 otherwise |
| Checksum | Jenkins lookup3 `hashlittle(…, 0)`, byte-wise as libhdf5's `H5_checksum_lookup3` (no aligned fast path, which reads past the end) |

Rules and the reasons for them (measured with libhdf5 1.14.6 on prototype files):

- **Chunk tail rule.** Free space at the end of a chunk is nothing, a gap of 1–3 zero bytes, or exactly one NIL message; never two NIL messages, never a gap and a NIL message together, never a NIL message before another message (a removal rewrites the chunk with the following messages moved forward). libhdf5 rejects a gap in a chunk with a NIL message, and on a read-write open it merges adjacent NIL messages and wraps sizes above 65535 into a corrupt header; bodies ≤ 65535 keep any free space within the 16-bit message size.
- **Attribute Info message** in every header with attributes (22 bytes): without it libhdf5 reports 0 attributes (`H5Oget_info3`, h5py `len(attrs)`, MATLAB `h5info`, HDFView), although iteration and open by name work.
- **Compact vs contiguous.** Datasets with ≤ 4096 bytes of raw data (`compact_limit`) are compact: the data is inside the checksummed header and is read with it. Larger datasets are contiguous, their data follows the header directly and is not checksummed (lookup3 runs at ≈ 2.8 GB/s). `ChannelDims`, `Order` and `ChannelRef` are always contiguous because they are updated in place without touching a checksum. An empty contiguous dataset has the undefined address (address 0 is rejected by libhdf5 1.14).
- **Null reference** = address 0 (libhdf5 tests `addr == 0`); valid references dereference to the channel groups.
- Fill Value written as libhdf5 does (`03 09` compact, `03 0a` contiguous); no modification time, no filters.

File written by `hdf5_create` (offsets fixed; readers still locate every object through the root links):

| Offset | Size | Object |
|-------:|-----:|--------|
| 0 | 48 | superblock, EOF = `779 + 12·n_slot` |
| 48 | 453 | root chunk 0: Link Info, Group Info, links `ChannelDims`, `Order`, `Version`, `ChannelRef`, one NIL of 320 bytes (the empty root directory) |
| 501 | 71 + 16 | `ChannelDims` header and data |
| 588 | 53 | `Version` header (compact) |
| 641 | 71 + 4·n_slot | `Order` header and data (zeros) |
| 712 + 4·n_slot | 67 + 8·n_slot | `ChannelRef` header and data (zeros, written in blocks of ≤ 1 MiB) |

### 3.3 Root links and growable groups

The root group can hold millions of links and is changed on every write. Its links to the channel groups are never stored in root chunk 0:

- **Directory.** Root chunk 0 ends with room for 16 Continuation messages (`root_directory_entries`, 20 bytes each). Entries point to *link chunks*; when a directory chunk has only 20 bytes left, its last entry points to a new *directory chunk* (body 16380 = 819 entries, `directory_chunk_body`). Chunk 0 thus holds 15 link-chunk entries, each directory chunk 818.
- **Link chunks** hold only Link messages and the tail. Link chunk `k` has the body `min(1024·2^k, 8192)` (`link_chunk_body_first`, `link_chunk_body_max`). A new link goes into the last link chunk; if it does not fit, a new link chunk is appended in the block and its directory entry is written in place.
- **Lookup without scanning.** IDs only grow, link chunks lie in the file in creation order, and a channel's link is appended to the last link chunk before the channel group is emitted. So the link of the group at address `A` is in the link chunk with the largest address below `A` (binary search over the directory), and the last link of the last link chunk carries the largest ID. A writer reads root chunk 0, the directory chunks, the last link chunk and, for an overwrite, the old link's chunk; it rewrites one link chunk (≤ 8 KB) or one directory chunk.
- **Capacity.** Chunk 0's 15 link chunks hold ≈ 3900 links (≈ 28 bytes each); 20 000 channels need ≈ 72 link chunks, 10^6 channels ≈ 3400 link chunks in 5 directory chunks. A plain chain of chunks would read all root links per write (≈ 28 MB at 10^6 channels). 8 KB link chunks with the 16 KiB header window were the fastest combination tried (section 9).
- **Why compact links and not dense storage or a symbol table:** libhdf5 reads compact link lists far beyond the Group Info limits without complaint (it converts a group only when it adds a link itself): 10^6 root links open in 1 ms, the first lookup by name takes 0.2 s (later lookups ≤ 8 ms, a linear scan), a full iteration 0.3–0.4 s (libhdf5 1.14.6; also checked with MATLAB's 1.10.11 and 1.14.4, and with up to 12 487 continuation chunks). Dense storage (fractal heap + B-tree) or an old-style symbol table would need much more code.

Growable group headers (`chain_links`): only two insertions happen after a header is written. A channel group without `Data` keeps a 19-byte reserve, exactly one Link message named `Data`, which `hdf5_write_dset` fills. A `Data` group keeps ≥ 20 bytes for a Continuation message; when its last chunk is full, a new chunk with the body `max(L + 20, min(16384, 2 · previous body))` is appended and linked from the old last chunk, and the creation index in the Link Info message (chunk 0) is incremented. The free space of a chunk is always computed from its messages, never assumed, and a rewrite compacts the chunk. A header longer than 65535 bytes (channel groups with more than ≈ 2500 snapshots, `Data` groups with very many datasets) is split into continuation chunks that follow chunk 0 directly; only the last chunk has free space.

### 3.4 Compatibility with other HDF5 readers

- Every structure exists since libhdf5 1.8.0 and is at or below the versions libhdf5 writes for 1.8-compatible files (from libhdf5's version-bound tables; 1.8 itself was not run). Tested: libhdf5 1.14.6 (C API; the interop test walks the full tree, values, attribute counts, creation order and references; a read-write open/close leaves files byte-identical), MATLAB R2024a (libhdf5 1.10.11) and R2026a (1.14.4) on prototype files of the same structures, and h5py 3.16, which reads the converter's outputs in its tests (root datasets, attributes, `Data` datasets). HDFView was not tested.
- MATLAB `h5read` of `ChannelRef` fails ("Reference value is invalid.") when any slot is empty. This is a MATLAB limitation (also for libhdf5-written files); users do not need `ChannelRef`.
- Files must not be modified with other tools. When libhdf5 adds a link to a large compact group it converts the group to dense storage, which takes 7–76 s at 10^4–10^5 links (deleting a link: 5 s to more than 5 min); the library then rejects the group. Objects that libhdf5 creates carry the "times stored" header flag and are rejected too. Some edits stay readable (a group added to a small root, a dataset deleted from `Data`); the docs say such files "are not supported and may be rejected".

## 4. Writing and reading

### 4.1 Write path

1. All inputs are validated, and every file structure the writer will read or rewrite is read and checked, before the first byte is written. A writer that throws leaves the file byte-identical. Writers read only what they need: `hdf5_reshape_layout` the superblock, root chunk 0 and the four root dataset headers; `hdf5_write` in addition the directory chunks, the last link chunk and the old link's chunk (never the old channel group); `hdf5_write_dset` the directory and last link chunk for an empty slot, the channel group (and its `Data` group) for a filled slot.
2. The new objects are emitted as one block from the stored EOF (not the physical file end) through a 1 MiB buffer (`io_block`). Float data of ≥ 1 MiB is written directly from Armadillo memory; double input is converted in ≤ 1 MiB pieces. A ray-tracing channel is one `write` call. Objects are emitted in post-order (children before their group, so each group header knows its children's addresses): snapshot datasets and `Snap_<i>` headers, channel-level datasets, unstructured datasets and `Data`, the new root directory/link chunk if any, then the channel group. The link chunk must precede the channel group (lookup rule of 3.3).
3. In-place updates, in this order:

| Step | Writes | Library sees after an interruption here | libhdf5 sees |
|------|--------|-----------------------------------------|--------------|
| 1 | the block past the stored EOF | old state (the bytes are overwritten by the next write) | old state; a read-write close truncates the bytes |
| 2 | superblock: new EOF and checksum | old state; the block is dead space | old state |
| 3 | one chunk: the link in the last link chunk, or a new directory entry | old state | the new group in addition |
| 4a | `ChannelRef[slot]` (8 bytes) | new channel; `hdf5_read_layout` shows the old ID | both groups |
| 4b | `Order[slot]` (4 bytes) | new state | both groups, the old one unreferenced |
| 5 | overwrite only: the old link's chunk without it | new state | new state |

Why EOF comes first: libhdf5 ignores bytes past the stored EOF and truncates them on a read-write close, and it cannot open an object past the stored EOF. With the link written before the EOF update, any libhdf5 read-write session in between would cut off the new object and leave a dangling link. New IDs come from the last root link, so a link left by an interrupted write is never reused. A crash between 4a and 4b of an `hdf5_write_dset` on an empty slot leaves `Order[slot] = 0` with a filled `ChannelRef`; this is not repaired automatically (the next `hdf5_write` to the slot fixes it). `hdf5_write_dset` on a filled slot writes the block, the superblock, then either the channel-group chunk that receives the `Data` link or `Data` chunk 0 (creation index + 1) and its last chunk; `Order` and `ChannelRef` stay.

The file is opened unbuffered, so the writes reach the operating system in program order. There is no `fsync`: the order protects against an interrupted process, not against power loss (as v2). A torn chunk write breaks that chunk's checksum: libhdf5 can no longer iterate the root, the library's readers are unaffected (they use `ChannelRef`), and a later write that needs the chunk throws.

**Overwrite = dead space.** The old channel group and its data stay in the file; only the old root link is removed. The file never shrinks and free space is not reused. To reclaim space, copy the channels into a new file (`hdf5_read_channel` + `hdf5_write`); `h5repack` output is not readable any more. **Reshape** writes the 16 bytes of `ChannelDims`. **Create** truncates and writes the image of 3.2 (no interruption guarantee).

### 4.2 Read path

`open`: superblock, root chunk 0, the four root links found by name, `Version`, `ChannelDims`, `Order`, `ChannelRef` headers, and a check that the root chunk, the root dataset headers and the three index data ranges do not overlap. A read then takes one `ChannelRef` entry and parses the channel group and its children, whose headers are adjacent in the file. Reads of known length (superblock, continuation chunks, contiguous data, index entries) go directly into their destination; contiguous float data lands in Armadillo memory (`float`) or goes through one float buffer per dataset (`double`). Only chunk 0 of a header, whose length is known after its prefix, goes through a 16 KiB window (`header_window`) that starts 4 KiB before the address (`header_lookback`): a snapshot group header lies after its datasets and is read first. Any write drops the window. All reads and writes are issued in pieces of at most 1 GiB (`io_max`; the Windows C runtime rejects single calls above 2 GiB). Unstructured datasets are returned in creation order (sorted by the creation-order field of the links).

### 4.3 Concurrency

One writer per file at a time; files are not locked (v2 inherited libhdf5's file locking). Reading is safe while nobody writes. `qd_hdf5.hpp` has no global or static mutable state (only constant tables), so different files can be read and written from parallel threads. The HDF5 code uses no OpenMP.

## 5. Reader validation and error messages

Every function validates everything it reads; violations throw `std::invalid_argument`. The rules in short:

- **Bounds:** every address and length is checked as `n ≤ end && addr ≥ lower && addr ≤ end − n` (`in_bounds`), `addr + n` is never computed; products are checked as `a != 0 && b > max / a` (`multiply`); no compiler builtins. Objects must lie in `[48, EOF)`, and `EOF ≤ file size` (a larger file is valid: bytes past EOF are ignored).
- **Checksums** of the superblock and of every chunk read, by readers and writers.
- **Chunks:** header flags are checked before the size field (libhdf5-created headers carry times and have the field elsewhere); body ≤ 65535, continuation length in `[8, 65543]`; messages must fit; continuation chunks are read in the order found, each at or after the end of the previous one (no cycles, bounded work).
- **Messages and objects:** only the types and versions of 3.2; dense link or attribute storage (defined fractal-heap address) → `Unsupported`; shared messages rejected; NIL data ignored. A dataset has exactly one Dataspace, Datatype and Layout message; a group exactly one Link Info and Group Info and one Attribute Info if it has attributes. Data size must equal `n_elem · element size`. Datatypes are matched byte by byte; element types are exact (structured data float32, `no_interact` uint32, attributes as in 3.1; v2 converted any numeric type).
- **Per-call budget:** the bytes of all header chunks and contiguous data a call reads for a channel (plus 8 bytes per slice of an empty rank-3 dataset, which Armadillo allocates) may not exceed EOF. A library file reads each object once; crafted files with many links to one object cannot amplify memory or time beyond a multiple of the file size. No allocation is sized from file values before this and the extent checks pass.
- **Schema:** the v2 checks (attribute presence and types, extents from `NumPath`/`NumTx`/`NumRx`, except dimensions 1–2 of `delay`; `coeff_re`/`coeff_im` together; `interact_coord` rows = `sum(no_interact)`; `Snap_<i>` names in canonical decimal, complete and unique); `Version` = 3; index consistency.
- **Writers** additionally require exactly the root structure the library writes (chunk 0: Link Info, Group Info, the four links in order, then only Continuation messages; directory chunks only Continuation messages; link chunks only `channel_<id>` links with increasing IDs, the last one non-empty), and that no chunk they rewrite overlaps the index data. Anything else → `Unsupported` before the file is changed.

| Situation | Exception: message |
|-----------|--------------------|
| missing file (`hdf5_read_layout` returns zeros, `hdf5_write`/`hdf5_write_dset` create the file) | invalid_argument: `File does not exist.` |
| shorter than 8 bytes or wrong signature | invalid_argument: `Not an HDF5 file.` |
| valid HDF5 outside the subset: superblock ≠ 2 (every v2 file, libhdf5 writes version 0), other root links, dense storage, writer structure rules | invalid_argument: `Unsupported HDF5 file: quadriga-lib reads only channel files of version 3 that it wrote itself; files written by quadriga-lib 0.12.4 and earlier (version 2) and files modified with other HDF5 tools are not supported. Version-2 files can be converted with tools/hdf5_convert_v2.py.` |
| `Version` ≠ 3 | invalid_argument: `Unsupported channel file version <n>; this version of quadriga-lib reads version 3 only.` |
| `ChannelDims`/`Order`/`ChannelRef` inconsistent | invalid_argument: `Storage index in HDF file is corrupted.` |
| any other rule | invalid_argument: `Malformed HDF5 file: <what>.`, e.g. `checksum mismatch at address <a>`, `address <a> is outside the file`, `unsupported object header flags at address <a>`, `continuation chunk at address <a> is out of order`, `the channel references more data than the file holds`, `the storage index overlaps the object header at address <a>`, `cannot read <n> bytes at address <a>`, and the v2 texts for schema checks |
| file cannot be opened | invalid_argument: `Error opening file.` (`hdf5_write`: `… It may be opened by another program.`; `hdf5_create`: `Error creating file. It may be already opened.`) |
| write, seek before a write, or close fails | runtime_error: `Cannot write file '<fn>'.` |
| no channel ID left | runtime_error: `No free channel ID left in the HDF5 file.` |
| argument errors | the v2 texts; new: `Parameter name is too long.` (> 65000 bytes), `String '<name>' is too long: at most 4294967295 bytes can be stored.` |

`std::logic_error("Channel group address mismatch.")` marks a writer bug (computed vs emitted address), never a file.

## 6. Behavior changes for users (against 0.12.4)

- **Files:** version-2 files are rejected by every function, also by the writers (appending to an old results file now fails); convert them with `tools/hdf5_convert_v2.py`. Files modified with other HDF5 tools are not supported.
- **API:** removed C++ `get_HDF5_version`, MEX `quadriga_lib.hdf5_version` and the `HDF5 = …` line of Python `components()`; Python `hdf5_write_dset`, `hdf5_read_dset`, `hdf5_read_dset_names` take an optional `prefix='par_'` (as MEX and C++ already did; `prefix=''` lists all datasets).
- An existing file that cannot be read (e.g. no read permission) throws `Error opening file` from every function; v2's `hdf5_read_layout` returned zeros for it and the writers truncated it and created a new file.
- Unstructured data of foreign types (variable-length or NUL-padded strings, other numeric types, rank > 3) throw instead of returning an empty `std::any`; attributes must have exactly their documented types.
- No file locking (one writer per file is documented); different `Malformed HDF5 file: …` texts for corrupt files; `hdf5_read_layout` without `channelID` and `hdf5_reshape_layout` validate `Order` and `ChannelRef` too.
- Dataset names over 65000 bytes and strings of 4 GiB or more throw before the file is touched; duplicate par names throw before a missing file is created.
- MEX/Python `hdf5_create_file`: "File already exists." for every existing file, also one that cannot be read.
- Docs: the `h5repack` hint is replaced by "copy the channels into a new file"; every function documents the version requirement and the single-writer rule.

## 7. Code structure and portability

`src/qd_hdf5.hpp` (namespace `qd::hdf5`, private, not installed; included by `qd_channel_hdf5.cpp` and two internal tests), in file order:

| Part | Responsibility |
|------|----------------|
| constants, `malformed`, `unsupported` | design constants (values in 3.2–4.2), message types, the two error families |
| `lookup3` | checksum |
| `put_le`, `get_le`, `in_bounds`, `multiply`, `parse_decimal` | little-endian fields, overflow-safe bounds and products, decimal IDs (≤ 10 digits, no leading zero, ≤ 2^32 − 1) |
| `type_t`, `dtype`, `datatype_bytes`, `encode_datatype`, `decode_datatype` | element types; one constant byte table for encoding and exact decoding |
| `class file` | `FILE*` owner: UTF-8 names, unbuffered, 64-bit offsets, bounds-checked `read`, windowed `read_header`, `write` (both directions in pieces of ≤ 1 GiB), `close`, static `exists` |
| `superblock`, `read_superblock`, `superblock_bytes` | superblock checks and encoding |
| `chunk`, `patch`, message encoders, `encode_chunk`, `parse_chunk`, `read_chunk`, `rewrite_chunk`, `object`, `read_object` | chunk images with the tail rule, parsing and verification, in-place rewrites, a header's chunks in reading order |
| `link`, `attribute`, `group`, `dataset`, `decode_*`, `read_raw` | decoded views (valid while the object lives), data reads |
| `output`, `emit_dataset`, `header` | the append block, dataset emission, group-header builder (adds Attribute Info, splits at 65535) |
| `chain_links` | growable channel and `Data` group headers |
| `root_links` | root directory and link chunks, structure checks, binary search, insertion size, removal |

`src/qd_channel_hdf5.cpp` holds the schema (names, defaults, field mapping), `qHDF_open`, argument and channel validation, block assembly for channels and `any_type_id` data, reading channels and `std::any` values, the update order (`qHDF_commit_channel`), the remaining messages, and the `/*!MD … MD!*/` API docs.

Portability: C++17, no compiler builtins, `-Wall -Wextra -Wpedantic` clean with g++-12 and clang 21. Under `_WIN32`: `_fseeki64`/`_ftelli64`, file names via `std::filesystem::u8path` and `_wfsopen(…, _SH_DENYNO)` (shared like `fopen`; `_wfopen` triggers C4996 at `/W3`, `_wfopen_s` would not share), no `<windows.h>`; elsewhere `fopen`/`fseek` with `static_assert(sizeof(long) >= 8)` (32-bit POSIX targets do not compile, deliberately). File existence is decided by `errno` (`ENOENT`/`ENOTDIR`) after a failed open; no `stat`, no `std::filesystem` outside Windows, no floating-point `from_chars`, so the MEX files need at most GLIBCXX_3.4.26 (MATLAB limit 3.4.28; build with g++-12). Little-endian hosts only (`#error` otherwise; all MSVC targets are little-endian). The Windows paths were reviewed by reading only.

## 8. Tests

| File | Covers |
|------|--------|
| `tests/catch2_tests/test_hdf_functions.cpp` | 27 cases, public API only, independent of the file version: create, layout, reshape, slot indexing, float/double round trips with exact `(R)(float)x` oracles, empty snapshots, field subsets, names (255 bytes), all unstructured types and ranks, overwrite, par-only channels, `assume_valid`, `hdf5_write_dset`, read functions, argument and file errors |
| `tests/catch2_tests/test_hdf_robustness_internal.cpp` | 14 cases, byte-level corruption without libhdf5 (`qd_hdf5.hpp` used only to locate and re-sign chunks): superblock, version and index, `ChannelRef` targets, object headers, channel semantics, shared targets and header budget, huge extents without allocation, index overlap, unreadable file, writer-side root and chain structures, a byte-flip sweep over every header byte (plain, and re-signed to reach the parsers) and the index data, every intermediate state of interrupted writes, 8000 channels across the root directory |
| `tests/catch2_tests/test_qd_hdf5_internal.cpp` | 9 cases on the header: lookup3 test vectors (incl. a libhdf5-written superblock), datatype bytes, Attribute Info and message flags, tail rule, header splitting, `chain_links`, `root_links`, file bounds and window, 64-bit offsets (sparse file at 2^32 + 16), decimal names |
| `tests/catch2_tests/test_hdf5_interop.cpp` | optional, 5 cases with libhdf5 ≥ 1.12 (`H5Ovisit3`): full tree and values, attribute counts, every interrupted-overwrite state incl. a read-write open/close, 8000 root links, files modified with libhdf5 (rejected or read consistently) |
| `tests/python_tests/test_hdf5_functions.py`, `tests/mex_tests/test_hdf_rw.m` | wrapper argument handling, defaults, index conversion, types and errors (Python incl. `prefix`); the MEX test also runs in Octave |
| `tests/python_tests/test_hdf5_convert_v2.py` | the converter as a subprocess (section 10); skipped without h5py |

Running them (from the repository root):
- Catch2: configure with `-D ENABLE_TESTS=ON` (g++-12), then `build_linux/test_bin "hdf5*"` (API, robustness and interop cases) and `build_linux/test_bin "qd_hdf5*"`. The configure line "HDF5 interop test enabled (HDF5 x.y.z)" shows whether the interop test is built; `-DCMAKE_DISABLE_FIND_PACKAGE_HDF5=ON` builds without it.
- Sanitizers: build with `-fsanitize=address,undefined` and run the robustness cases with `ASAN_OPTIONS=max_allocation_size_mb=1024:allocator_may_return_null=0`, so that an allocation beyond 1 GB fails (ASan alone does not flag large allocations).
- Python: `make python`, then `make python_test` (pytest with `PYTHONPATH=<checkout>/lib`, so the fresh build is tested and not an installed module). The converter test needs h5py, e.g. in a conda environment with h5py.
- MEX: `matlab -batch "run('tests/quadriga_lib_mex_tests.m');"`, Octave `cd tests; quadriga_lib_mex_tests`. The runner exits 0 also when tests fail: check the output for `FAILED`.
- Converter fixtures: `tests/data/hdf5_v2_{written,edited,lossy}.hdf5` and the oracle `hdf5_v2_expected.npz` (260 KB) were made by `tests/data/make_hdf5_v2_fixtures.py` with the Python module of 0.12.4; its header has the recipe. Only rerun it to change the fixtures (the HDF5 files then differ in timestamps, the oracle does not).

Results at the commits (the total Catch2 count depends on build options such as AVX2; the HDF5 cases do not): `a719e43` Catch2 all passed, 46 `hdf5*` cases / 32 755 assertions incl. interop and 9 `qd_hdf5*` cases, pytest 886 passed, MATLAB R2024a, R2026a and Octave all HDF5 tests passed, the build without libhdf5 0 warnings and no `H5` symbols, ASan/UBSan clean; `e2e1a32` Catch2 all passed, pytest 896 passed (with h5py, converter included). Fuzzing (mutation harness on three library-written seed files, ASan + UBSan, allocation cap 256 MB; 1–8 mutations per iteration, mostly in headers, then re-signed; readers on 5 slots, writers on a copy with the byte-identical-or-reads-back oracle): about 1.2 million iterations in total; the defects found (unbounded allocations from huge extents, index data overlapping headers, link amplification, unreadable files being replaced) were fixed, and the ≈ 800 000 iterations after the fixes gave no finding.

## 9. Performance

Protocol: `tests/benchmarks_cpp/benchmark_hdf5.cpp` (public API only; compiles against both versions) writes all slots with `hdf5_write<float>`, then flushes the file (`fsync`, timed separately), drops its page cache (`posix_fadvise(POSIX_FADV_DONTNEED)`) and reads every slot with `hdf5_read_channel<float>`; data from four seeded template channels varied per slot; a read check counts mismatches (`bad=0` in all runs). v2 = `26da9f93` with the system libhdf5 1.14.6 (shared), v3 = `a719e43`; same configure (g++-12, `-O3 -fopenmp`, AVX2, static library, no BLAS), `OMP_NUM_THREADS=1`, files on ext4/NVMe, AMD Ryzen 9 5950X. Three rounds alternating v2/v3 under `flock /tmp/bench.lock`, 2026-10-29. Rerun: build the library, then `g++-12 -std=c++17 -O3 -fopenmp -I include tests/benchmarks_cpp/benchmark_hdf5.cpp <build>/libquadriga.a` and `benchmark_hdf5 -d <dir on disk> [-m a|b|both] [-s scale]`.

Mixes: a) MIMO, 3200 channels × 10 snapshots × 16×4 antennas × 200 paths (coefficients, per-pair delays, positions, orientations); b) ray-tracing route, 7000 channels × 100 snapshots × SISO × 20 paths with every path field (7000 instead of 20 000 channels keeps the v2 file at 5 GB).

| Mix | v2 write | v3 write | v2 read | v3 read (CPU) | File v2 → v3 |
|-----|---------:|---------:|--------:|--------------:|-------------:|
| a MIMO | 5.39 s | 1.57 s (3.4×) | 8.17 s | 4.43 s (2.80 s), 1.8× | 4995 → 4935 MB |
| b route | 123.8 s | 3.48 s (36×) | 150.4 s | 6.33 s (5.21 s), 24× | 5012 → 2470 MB |

Mean of rounds 2–3 (1-min load 1.21–1.48 before and after each run); round 1 (load 6.1 → 4.7, provisional) agrees within ≈ 1 %, and the v2 baseline of the previous day (load 2.2–2.5) within 3 %. Write CPU time equals wall time; `fsync` adds 0.96 s (a) and 0.53 s (b) for v3, 1.37 s and 1.28 s for v2. Compare times, not MB/s, for mix b (the v3 file is half the size).

- **Metadata per snapshot** (mix b): v3 ≈ 1.2 KB (3529 bytes per snapshot on disk, ≈ 2.3 KB payload), v2 ≈ 4.9 KB. Mix a: ≈ 6 KB metadata per 1.54 MB channel in v3 (v2 ≈ 25 KB).
- **System calls per call** (`strace`, 20 channels into a 1000-slot file):

| | v2 write | v3 write | v2 read | v3 read |
|---|---:|---:|---:|---:|
| mix a (1.54 MB channel) | 73.5 | 25.9 (6 write, 5 read, 12 lseek) | 145 (1.58 MB read) | 156 (2.24 MB read) |
| mix b (354 KB channel) | 1309 | 24.0 (5 write, 5 read, 11 lseek) | 2879 (1.07 MB read) | 116 (0.87 MB read) |

  The v3 read counts were taken before the window look-back of 4.2; with it a mix-b read issues 30 instead of 56 reads (0.44 MB), and mix-b reads became ≈ 20 % faster together with the allocation-free datatype decoding. A v3 write reads ≈ 17 KB first (superblock, root window, index entries, last link chunk) and writes the block plus four small updates. Each read or write is `lseek` + `read`/`write`; `pread` would halve the calls but is POSIX-only.
- **Per-call cost** (10^5-slot file, 10^5 minimal channels: 1 snapshot, SISO, 1 path; open to close; load 1.7–2.9): write 74–75 µs, overwrite 99–100 µs, read 26–27 µs, ≈ 25 system calls per write (measured before the final read-path changes). It tuned the two constants: 16 KB link chunks with a 64 KiB window gave 88–90 / 120–124 / 38 µs; 4 KB chunks equal 8 KB within noise but need more directory chunks at 10^6 channels; a 64 KiB window made mix-a reads slower than v2 (each 51 KB coefficient header loaded 64 KiB of data), 8 KiB equals 16 KiB.

## 10. Converter `tools/hdf5_convert_v2.py`

`python tools/hdf5_convert_v2.py [--force] input.hdf5 output.hdf5` reads a version-2 file with h5py (read-only) and writes a version-3 file with the same layout through `hdf5_create_file`, `hdf5_write_channel` and `hdf5_write_dset` of the quadriga_lib Python module; nothing of the library is re-implemented. Requirements: Python ≥ 3.10, numpy, h5py ≥ 3.0, a quadriga_lib module newer than 0.12.4. Module selection: a normal `import quadriga_lib` (installed or chosen by `PYTHONPATH`); only if that fails, `../lib` next to the script (a `make python` build). The script checks that the module writes `Version` 3 and otherwise names the module path and the `PYTHONPATH` remedy. MATLAB and Octave users run it with Python; there is no MATLAB converter.

- **Equivalence:** every slot reads back with `hdf5_read_channel` as the input did with 0.12.4: same fields, shapes, element bytes (float32 values are carried bit-exact, NaN payloads, ±0 and ±Inf included) and unstructured datasets with any prefix in creation order. The script re-implements the 0.12.4 *reader* rules, so it also reads files of older writers (missing empty `Snap_<i>` groups padded) and of other tools (h5py handles dense storage, chunked and compressed data, other byte orders). Channel IDs are renumbered in slot order.
- **Deliberate repairs** (0.12.4 read these channels but its own validation rejected them): a missing trailing empty `delay` of a channel without coefficients becomes `(1, 1, 0)` instead of `(0, 0, 0)`; delays of shape `(1, n_tx, np)` or `(n_rx, 1, np)` (writers of 2023-11 to 2024-07) are broadcast to `(n_rx, n_tx, np)`. Channel groups without orientations (edits only) read back with zero orientations.
- **Not convertible (warning, exit 3):** occupied slots without channel data (left empty), names of channels without structured data, name bytes beyond 255, datasets with names outside `[A-Za-z0-9_]`, unsupported types or rank > 3, strings that are not valid UTF-8. These arise only from C++ `assume_valid` or edits with other tools. Any other channel the library rejects ends the run (a channel is never converted partially).
- **Safety:** the output is written into a fresh temporary folder next to it and moved into place with `os.replace` only when complete; an existing output needs `--force`; the input is never changed; errors, Ctrl-C, SIGTERM and SIGHUP remove the temporary folder.
- **Exit codes:** 0 complete; 3 complete except the warned items; 1 error, nothing written; 2 wrong arguments; 130 interrupted, nothing written. Messages use argparse style (`hdf5_convert_v2.py: error: …`), slot indices 0-based.
- **Timing:** a 966 MB v2 file (1000 channels, 8×8 antennas, 20 snapshots, 50 paths, two par datasets each) converted in 34.9 s with 83 MB peak memory (provisional: load 6.4). Most of the time is h5py's per-object overhead (≈ 220 000 dataset opens); the library writes took 2.7 s. Memory: `Order` plus one channel.

## 11. Build and packaging

Removed: in `CMakeLists.txt` `HDF5_PATH`, `HDF5_STATIC`, the bundled-source build (`ExternalProject`), the MSVC prebuilt autodetect, `find_package(HDF5)` for the library, the Octave switch-off, HDF5 include and link flags for the library, MEX, Python and tests, installation of libhdf5, `-Wl,-z,nodelete`; `GNUmakefile` and the Windows `Makefile` HDF5 variables and targets; `external/hdf5-1.14.6.zip`; `libhdf5-dev` in `Dockerfile.ubuntu` and `debian/control`, the HDF5 options in `debian/rules` and `debian/copyright`; `libhdf5-dev`/conda `hdf5` in the READMEs and install pages; HDF5 flags in `tools/acoustic/build_band_probe.sh` and `tools/sos_port/review_prototypes/README.md`. The library, the MEX files and the Python module contain no `H5` symbols and need no libhdf5 at run time (`nm`, `ldd`).

`-z nodelete`: its only reason was libhdf5. Checked in MATLAB R2024a and R2026a, three runs each: MEX files built without the flag ran three cycles of all `hdf5_*` functions plus other MEX functions (OpenMP included), `clear mex` after each cycle, then `exit`, without a crash. Control: v2 MEX files with libhdf5 and without the flag crashed R2026a at exit in both runs, so the check detects the original defect.

Optional test dependency: `find_package(HDF5 1.12 MODULE QUIET COMPONENTS C)` in the test section only; when found, `test_hdf5_interop.cpp` is compiled and libhdf5 is linked into `test_bin` only, with `${HDF5_DEFINITIONS}`. An older HDF5 skips the test instead of breaking the build; the Debian package build (no `libhdf5-dev`) skips it.

## 12. Open items and known limitations

- `cmake --install` from a build directory that once built `hdf5_version.mex*` installs that stale file again; delete it (or run `make tidy`) when updating an old build directory. Worth a release note.
- Not run: libhdf5 1.8, HDFView, a full-tree comparison with h5py; MSVC/MinGW builds and the Windows paths (`_fseeki64`, `_wfsopen`, `u8path`, 1 GiB pieces); no test issues a single read or write above 1 GiB.
- Residual gap for crafted files: a valid object header placed inside the `Order`/`ChannelRef` data of a header that the writer does not read can be overwritten by a slot update (only that crafted object is damaged).
- Interrupted `hdf5_write_dset` on an empty slot between its last two updates leaves `Order[slot] = 0` (4.1); not repaired automatically.
- v2 wrapper behavior kept unchanged (pinned or out of scope): Python `hdf5_write_dset` checks `ix > nx` instead of `>=` (two different messages) and accepts empty arrays; Python `hdf5_read_channel` truncates float indices and ignores `snap` on slots without structured data; a par-only channel loses its name; an empty channel without par data is not written.
- `README_pip.md` lists wheels for Python 3.9–3.13, while the converter needs Python ≥ 3.10.
