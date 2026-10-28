# Native HDF5 channel files (file format version 3): specification

Branch `hdf5-native`. This document is the implementation specification for replacing libhdf5 in quadriga-lib by
the private header `src/qd_hdf5.hpp`. It is complete without the plan (`md/hdf5_native_plan.md`), whose decision
ids D1–D15 it cites. Byte layouts follow the *HDF5 File Format Specification Version 3.0* (The HDF Group), cited
as "FF spec" with its section names. All layouts below were written by prototypes and read back with libhdf5 1.14.6
(C API) and with MATLAB's libhdf5 1.10.11 (R2024a) and 1.14.4 (R2026a): tree walk (`H5Ovisit3`), values,
attributes by iteration (`H5Aiterate2`) and by count (`H5Oget_info3` num_attrs, `H5Aget_num_attrs`, MATLAB `h5info`),
creation order, object references, and a read-write open/close that leaves the file byte-identical. libhdf5 1.8 was
not available: 1.8 compatibility follows from libhdf5's own version-bound tables, not from a measurement. All
structures used — superblock v2, object header v2, Dataspace v2, Datatype v1, Fill Value v3, Data Layout v3,
Attribute v3, Attribute Info v0, Link Info v0, Group Info v0, Link v1, Continuation, NIL — are at or below the
versions libhdf5 writes for 1.8-compatible files, and all exist since libhdf5 1.8.0.

Notation: all integers little-endian; `UNDEF` = `ff ff ff ff ff ff ff ff` (undefined address); sizes in bytes;
"body" of an object-header chunk = the bytes between the chunk prefix and the checksum (messages and gap);
"message size" = 4-byte message header + message data; HDF5 dataspace extents are listed slowest dimension first
(an Armadillo object `[rows, cols, slices]` has the extents `{slices, cols, rows}` and the same memory order).

Contents: 1 Logical schema · 2 Byte layouts · 3 File layout, growth and update order · 4 Reader validation and
errors · 5 Public functions · 6 `qd_hdf5.hpp` · 7 Portability · 8 Tests · 9 Build, packaging, wrappers ·
10 API documentation · 11 Benchmark · Appendix A constants.

---------------------------------------------------------------------------------------------------------------------

## 1. Logical schema v3

The v2 tree (D1) plus `ChannelRef` (D3) and `Version = 3` (D2). Every dataset and attribute has a simple dataspace
whose maximum extents equal its extents (not stored, see 2.5) or a scalar dataspace. No other objects, attributes or
links exist in a file written by the library.

### 1.1 Root group `/`

No attributes. Links are not creation-ordered.

| Object | Class | Element type | Dataspace | Storage | Content |
|--------|-------|--------------|-----------|---------|---------|
| `ChannelDims` | dataset | uint32 | `[4]` | contiguous | `{nx, ny, nz, nw}`, each ≥ 1 (an entry 0 is a corrupted index, 4.2), product ≤ 2^32 − 1 |
| `Order` | dataset | uint32 | `[n_slot]` | contiguous | channel ID per slot, 0 = empty; `n_slot = nx·ny·nz·nw` |
| `Version` | dataset | uint32 | scalar | compact | 3 |
| `ChannelRef` | dataset | object reference | `[n_slot]` | contiguous | object-header address of the slot's channel group; 0 = empty slot (null reference) |
| `channel_<id>` | group | | | | one per occupied slot; `<id>` = decimal channel ID without leading zeros |

Slot index of `(ix, iy, iz, iw)`: `ix + nx·(iy + ny·(iz + nz·iw))`. Channel IDs are unique, start at 1, and a new
ID is one above the largest ID ever linked in the root (3.4); an overwrite gives the slot a new ID. `Order` and the
IDs keep their v2 meaning: `hdf5_read_layout` returns `Order`. All data-reading functions locate a slot through
`ChannelRef` only (D3); they never search root links.

### 1.2 Channel group `/channel_<id>`

Attributes (always all ten; `NumPath` only as noted). The defaults are written for every channel group and replaced
by `hdf5_write` as described:

| Attribute | Type | Dataspace | Default | Written value |
|-----------|------|-----------|---------|---------------|
| `NumSnap` | uint32 | scalar | 0 | `n_snap` |
| `NumTx` | uint32 | scalar | 0 | `n_tx()` |
| `NumRx` | uint32 | scalar | 0 | `n_rx()` |
| `Initial_position` | int32 | scalar | 0 | `initial_position` |
| `CenterFrequency` | float32 | scalar | NaN | the value, if `center_frequency` has 1 element |
| `tx_position` | float32 | `[3]` | NaN ×3 | the column, if `tx_pos` is `[3, 1]` |
| `rx_position` | float32 | `[3]` | NaN ×3 | the column, if `rx_pos` is `[3, 1]` |
| `tx_orientation` | float32 | `[3]` | 0 ×3 | the column, if `tx_orientation` is `[3, 1]` |
| `rx_orientation` | float32 | `[3]` | 0 ×3 | the column, if `rx_orientation` is `[3, 1]` |
| `Name` | fixed-length string, 256 bytes, NUL-terminated padding, ASCII | scalar | 256 NUL bytes | the first ≤ 255 bytes of `name`, NUL-padded to 256 |
| `NumPath` | uint32 | scalar | — | only in a 1-snapshot channel with snapshot data: `n_path(0)` |

"Written value" applies only when the channel has structured data, i.e. when `hdf5_write` writes the channel data
(`!ch->empty()`, or any channel with `assume_valid = true`). A channel with only unstructured data, and every channel
group created by `hdf5_write_dset`, keeps all defaults (its name is not stored; v2 behavior, pinned by tests).

Datasets in the channel group:

| Dataset | Type | Dataspace | Present if |
|---------|------|-----------|------------|
| `CenterFrequency` | float32 | `[n_snap]` | `center_frequency` has `n_snap` elements and `n_snap ≥ 2` (the attribute keeps NaN) |
| `tx_position`, `rx_position`, `tx_orientation`, `rx_orientation` | float32 | `[n_snap, 3]` | the matrix has `n_snap` columns and `n_snap ≥ 2` (the attribute keeps its default) |
| snapshot datasets (1.3) | | | `n_snap = 1` and the channel has snapshot data |

An empty position/orientation/frequency keeps the default. A matrix with 3 rows and a column count other than 1 or
`n_snap` is silently not stored (v2). A matrix with a row count other than 3 throws `'tx_pos' must have 3 rows.`
(with the field name) before the file is changed.

Links of the channel group: the datasets above, `Snap_<i>` groups (`n_snap > 1`), `Data` (if unstructured data).

### 1.3 Snapshot data

A channel has *snapshot data* if at least one per-snapshot field has exactly `n_snap` entries:
`coeff` (needs both `coeff_re` and `coeff_im`), `delay`, `path_gain`, `path_length`, `path_polarization`,
`path_angles`, `path_fbs_pos`, `path_lbs_pos`, `no_interact`, `interact_coord`. Then every snapshot is written, also
snapshots without paths (datasets with extent 0), and every present field is written for every snapshot. Without
snapshot data there are no `Snap_<i>` groups, no snapshot datasets and no `NumPath`.

Location: `n_snap = 1`: in the channel group; `n_snap > 1`: in groups `Snap_0` … `Snap_<n_snap−1>` (decimal, no
leading zeros), each with the attribute `NumPath` (uint32 scalar, `n_path(i)`). Snapshot groups have no other
attributes and only the dataset links below. Datasets of snapshot `i` with `np = n_path(i)` (all float32 except
`no_interact`):

| Dataset | Armadillo object | HDF5 extents |
|---------|------------------|--------------|
| `coeff_re`, `coeff_im` | `Cube [n_rx, n_tx, np]` | `{np, n_tx, n_rx}` |
| `delay` | `Cube [n_rx, n_tx, np]` or `[1, 1, np]` | `{np, n_tx, n_rx}` or `{np, 1, 1}` |
| `path_gain`, `path_length` | `Col [np]` | `{np}` |
| `path_polarization` | `Mat [8, np]` | `{np, 8}` |
| `path_angles` | `Mat [np, 4]` | `{4, np}` |
| `path_fbs_pos`, `path_lbs_pos` | `Mat [3, np]` | `{np, 3}` |
| `no_interact` | `u32_vec [np]` (uint32) | `{np}` |
| `interact_coord` | `Mat [3, sum(no_interact)]` | `{sum, 3}` |

`coeff_re[i]` and `coeff_im[i]` of different size throw `Size mismatch between 'coeff_re[i]' and 'coeff_im[i]'.`
before the file is changed. Write order of the datasets (= link order): the order of the table.

### 1.4 Unstructured data `/channel_<id>/Data`

Present iff the slot holds at least one unstructured dataset. The group tracks and indexes link creation order
(D5); links carry creation-order values 0, 1, …; the order of the link messages equals the creation order. Dataset
name: `prefix + name` (`hdf5_write`: `par_`; `hdf5_write_dset`: its `prefix`, default `par_`).

| `any_type_id` | Stored as | Dataspace |
|---------------|-----------|-----------|
| 9 `std::string` | fixed-length string, NUL-terminated padding, ASCII, size `max(len, 1)`; the bytes of the string, no added NUL; an empty string is one NUL byte | scalar |
| 10–15 scalar | element type by `id % 10` (below) | scalar |
| 20–25 `Mat` | | `{cols, rows}` |
| 30–35 `Cube` | | `{slices, cols, rows}` |
| 40–45 `Col`, 50–55 `Row` | | `{n_elem}` (a `Row` comes back as `Col`) |

Element type by `id % 10`: 0 float32, 1 float64, 2 uint64, 3 int64, 4 uint32, 5 int32. Data are the raw bytes of the
object (no conversion). Other types throw `Unsupported data type for '<name>'` before the file is changed.

### 1.5 Notes for users of other HDF5 tools

The files can be read by any HDF5 reader. They must not be modified by other tools: libhdf5 converts the root group
to dense link storage when it adds a link (slow: seconds to minutes at 10^4–10^5 channels) and the library then
rejects the file (D4). MATLAB's `h5read` of `ChannelRef` fails ("Reference value is invalid.") when any slot is
empty; this is a MATLAB limitation (also for libhdf5-written files) and not mentioned in the API docs, since users do
not need `ChannelRef`.

---------------------------------------------------------------------------------------------------------------------

## 2. Byte layouts

Only the structures below are written. Offsets are relative to the start of the structure.

### 2.1 Superblock version 2 (FF spec "Superblock", version 2), 48 bytes at file offset 0

| Off | Size | Field | Written | Reader accepts |
|-----|------|-------|---------|----------------|
| 0 | 8 | signature | `89 48 44 46 0d 0a 1a 0a` | equal, else `Not an HDF5 file.` |
| 8 | 1 | version | 2 | 2, else "unsupported" (4.2) |
| 9 | 1 | size of offsets | 8 | 8 |
| 10 | 1 | size of lengths | 8 | 8 |
| 11 | 1 | file consistency flags | 0 | 0 or 1 (bit 0 = left by a crashed libhdf5 read-write session) |
| 12 | 8 | base address | 0 | 0 |
| 20 | 8 | superblock extension address | `UNDEF` | `UNDEF` (address 0 makes libhdf5 fail) |
| 28 | 8 | end-of-file address (EOF) | end of the last allocated byte | `48 < EOF ≤ file size` |
| 36 | 8 | root group object header address | 48 | `48 ≤ root < EOF` |
| 44 | 4 | checksum | lookup3 of bytes 0..43 | must match |

A file larger than EOF is valid (libhdf5 ignores the extra bytes and truncates them on a read-write close). The
writer always appends at EOF, never at the physical end of the file.

### 2.2 Checksum

Jenkins lookup3 `hashlittle(key, length, initval = 0)` (FF spec "Appendix C"; Bob Jenkins' public-domain
`lookup3.c`), over the bytes before the checksum field. Implemented byte-wise, as libhdf5's `H5_checksum_lookup3`:
each 12-byte round assembles `a`, `b`, `c` from single bytes, the last 1–12 bytes are added by a `switch`; not the
aligned fast path of `lookup3.c`, which reads whole words through a cast pointer and up to 3 bytes past the end
(undefined behavior, flagged by ASan). The byte-wise form is also independent of the host byte order. Test vectors: `""` → `0xdeadbeef`;
`"Four score and seven years ago"` (30 bytes) → `0x17770551`; the 44 superblock bytes
`89 48 44 46 0d 0a 1a 0a 02 08 08 00 | 00×8 | ff×8 | 9d 0f 00 00 00 00 00 00 | 30 00 00 00 00 00 00 00`
(written by libhdf5 1.14.6) → `0x60d52a4e` (stored as `4e 2a d5 60`).

### 2.3 Object header version 2 (FF spec "Version 2 Object Header")

Chunk 0:

| Off | Size | Field | Written | Reader accepts |
|-----|------|-------|---------|----------------|
| 0 | 4 | signature | `4f 48 44 52` ("OHDR") | equal |
| 4 | 1 | version | 2 | 2 |
| 5 | 1 | flags | bits 0–1 = width code of the next field (0: 1 byte if body < 256, 1: 2 bytes); bits 2–7 = 0 (no attribute creation order, no phase-change values, no times) | bits 2–7 = 0; width code 0 or 1 |
| 6 | 1 or 2 | size of chunk 0 | body size (messages + gap) | 1 ≤ body ≤ 65535, chunk within EOF |
| 7/8 | body | messages, then gap | | |
| end | 4 | checksum | lookup3 from the signature to the end of the gap | must match |

Continuation chunk: `4f 43 48 4b` ("OCHK"), messages, gap, 4-byte checksum (lookup3 from the signature to the end of
the gap). Its length (held by the Continuation message) includes signature and checksum: `length = body + 8`.

Message header (no creation-order field, header flag bit 2 = 0): type (1), data size (2), message flags (1), then
the data. Message flags written as libhdf5 1.14.6 writes them: `0x01` (constant) on Datatype, Fill Value and Group
Info messages, `0x04` (do not share) on Attribute Info messages, `0x00` on all others. The reader ignores the flags
(so `0x00` on Group Info or Attribute Info is accepted too) except bit 1 (shared message), which it rejects.

Chunk tail rule (every chunk the library writes or rewrites): after the last message, the remaining free space
`F` of the body is filled with nothing (`F = 0`), a gap of `F` zero bytes (`F = 1…3`; legal only because the chunk
then has no NIL message), or exactly one NIL message of data size `F − 4` (`F ≥ 4`). Never two NIL messages in a
chunk, never a NIL message and a gap in one chunk, never a NIL message before another message (removing a message
rewrites the chunk with the following messages moved forward). Chunk bodies are at most 65535 bytes, so free space
always fits the 16-bit message size (libhdf5 merges adjacent NIL messages when it opens a file read-write and would
wrap larger sizes).

### 2.4 Messages

| Type | Name (FF spec) | Data written | Data size |
|------|----------------|--------------|-----------|
| 0x00 | NIL | zero bytes | 0…65535 |
| 0x01 | Dataspace, version 2 | `02 rank 00 type` then `rank` × 8-byte extents; type 0 = scalar (rank 0), 1 = simple; flags 0 (no maximum extents) | 4 + 8·rank |
| 0x02 | Link Info, version 0 | `00 00` + `UNDEF` (fractal heap) + `UNDEF` (name index) | 18 |
| 0x02 | Link Info, version 0, creation order tracked and indexed (`Data` groups) | `00 03` + max creation index (8, = number of links ever created in the group) + `UNDEF` ×3 | 34 |
| 0x03 | Datatype, version 1 | table 2.5 | 8, 12 or 20 |
| 0x05 | Fill Value, version 3 | `03 09` (compact dataset: allocation early, fill write "if set") or `03 0a` (contiguous: allocation late, "if set") — what libhdf5 writes; message flags 0x01 | 2 |
| 0x06 | Link, version 1, hard link | `01 flags` + [creation order, 8, if flags bit 2] + name length (1 byte if < 256: flags bits 0–1 = 0; 2 bytes: = 1) + name (no NUL) + object-header address (8). Flags: `0x00`/`0x01`, plus `0x04` in `Data` groups | 11 + n (+1, +8) |
| 0x08 | Data Layout, version 3, compact | `03 00` + size (2) + raw data | 4 + size |
| 0x08 | Data Layout, version 3, contiguous | `03 01` + address (8) + size (8); size 0 ⇒ address `UNDEF` (address 0 is rejected by libhdf5 1.14) | 18 |
| 0x0A | Group Info, version 0 | `00 00`; message flags 0x01 | 2 |
| 0x0C | Attribute, version 3 | `03 00` + name size (2, incl. NUL) + datatype size (2) + dataspace size (2) + `00` (ASCII name) + name + NUL + datatype data + dataspace data + raw value | 9 + name + 1 + dt + ds + value |
| 0x10 | Object Header Continuation | address (8) + length (8) of the continuation chunk | 16 |
| 0x15 | Attribute Info, version 0 | `00 00` (creation order neither tracked nor indexed) + `UNDEF` (fractal heap) + `UNDEF` (name index); message flags 0x04. Message bytes: `15 12 00 04 00 00` + `ff`×16 | 18 |

Message sizes used below: Link Info 22 / 38, Group Info 6, Attribute Info 22, Continuation 20; a Link message without creation order
is `15 + name length` bytes (root channel link `23 + digits`, `Data` link 19), with creation order (`Data` members)
`23 + name length` (+1 if the name is ≥ 256 bytes).

The dataspace of a dataset with extent 0 is written like any other (`02 rank 00 01` + extents containing 0).
No maximum extents: libhdf5 then reports maximum extents equal to the extents (checked by the interop test).
Attribute Info is required in every version-2 header that has attributes: libhdf5 counts the attributes of such a
header only through it (`H5Oget_info3` num_attrs, `H5Aget_num_attrs`, h5py `len(attrs)`, MATLAB `h5info`, HDFView
report 0 attributes without it, although iteration and open by name work). libhdf5 adds it when it creates the first
attribute; it costs 22 bytes per channel and snapshot group.
No Modification Time message, no Filter Pipeline.

### 2.5 Datatype messages (version 1; byte-identical to libhdf5 1.14.6 output)

| Type | Bytes |
|------|-------|
| int32 / uint32 | `10 08 00 00 04 00 00 00 00 00 20 00` / `10 00 00 00 04 00 00 00 00 00 20 00` |
| int64 / uint64 | `10 08 00 00 08 00 00 00 00 00 40 00` / `10 00 00 00 08 00 00 00 00 00 40 00` |
| float32 | `11 20 1f 00 04 00 00 00 00 00 20 00 17 08 00 17 7f 00 00 00` |
| float64 | `11 20 3f 00 08 00 00 00 00 00 40 00 34 0b 00 34 ff 03 00 00` |
| fixed string of n bytes | `13 00 00 00` + n (4 bytes): NUL-terminated padding, ASCII |
| object reference | `17 00 00 00 08 00 00 00` |

The reader accepts exactly these encodings (byte comparison; for strings the first four bytes and `n ≥ 1`) and rejects
every other datatype (other integer sizes, NUL-padded or variable-length strings, compound types, …: D4).

### 2.6 Object header contents

Messages in this order (the reader does not depend on the order except where noted in 3.4):

| Object | Messages |
|--------|----------|
| dataset | Dataspace, Datatype, Fill Value, Data Layout |
| channel group, snapshot group | Link Info (flags 0), Group Info, Attribute Info, Attribute …, Link … |
| `Data` group | Link Info (flags 3), Group Info, Link … (with creation order) |
| root group, chunk 0 | Link Info (flags 0), Group Info, Links `ChannelDims`, `Order`, `Version`, `ChannelRef`, directory (Continuation …), NIL (3.4) |

Compact or contiguous: a dataset is compact if its raw data size is ≤ 4096 bytes (also 0), contiguous otherwise; the
root datasets `ChannelDims`, `Order`, `ChannelRef` are always contiguous (they are updated in place without
touching a checksum), `Version` is compact. Compact data is part of the checksummed header; contiguous data is not
checksummed (lookup3 runs at ≈ 2.8 GB/s, ≈ 0.35 ms per MB). Contiguous raw data follows its dataset header directly.

Dataset header size: compact: `prefix + 4+4+8·rank + 4+dt + 6 + 4+4+data + 4`; contiguous:
`prefix + 4+4+8·rank + 4+dt + 6 + 22 + 4` with `prefix` = 7 (body < 256) or 8; `dt` = 8 (string, reference),
12 (integers), 20 (floats). Example: `path_gain` with 5 paths = 85 bytes.

---------------------------------------------------------------------------------------------------------------------

## 3. File layout, growth and update order

### 3.1 The file written by `hdf5_create` (verified with libhdf5 for n_slot = 4)

| Offset | Size | Object |
|--------|------|--------|
| 0 | 48 | superblock, EOF = `779 + 12·n_slot`, root = 48 |
| 48 | 453 | root chunk 0: flags `0x01`, body 441 = Link Info 22 + Group Info 6 + links `ChannelDims` 26, `Order` 20, `Version` 22, `ChannelRef` 25 + one NIL of 320 bytes (the empty directory, 16 entries) |
| 501 | 71 | `ChannelDims` header (contiguous, data address 572, size 16) |
| 572 | 16 | `ChannelDims` data |
| 588 | 53 | `Version` header (compact, value 3) |
| 641 | 71 | `Order` header (contiguous, data address 712, size `4·n_slot`) |
| 712 | 4·n_slot | `Order` data, zeros |
| 712 + 4·n_slot | 67 | `ChannelRef` header (contiguous, data address `779 + 4·n_slot`, size `8·n_slot`) |
| 779 + 4·n_slot | 8·n_slot | `ChannelRef` data, zeros |

The zeros are written in blocks of at most 1 MiB (a 2^32 − 1 slot layout writes 48 GB without a 48 GB buffer). The
layout is fixed, but readers find every object through the root links, not through these offsets.

### 3.2 Writing objects (the append block)

Every write appends one *block* at EOF. All validation is done on the inputs before any byte is emitted (errors are
thrown before the existing file is changed; a missing file is created first, as in v2). The block is then emitted
sequentially from the stored EOF through a 1 MiB output buffer, which is written when full; contiguous data larger
than the buffer is written directly from the Armadillo memory (`float` input) or converted in ≤ 1 MiB pieces
(`double` input). A block ≤ 1 MiB (every block of mix b) is thus one write, a larger one about one write per MiB, so
the number of system calls does not grow materially, while the peak memory is bounded by 1 MiB plus the caller's
data instead of a full float copy of the channel. Consistency does not depend on a single write: all block bytes lie
past the stored EOF, which no reader sees before step 2 of 3.7. Objects are emitted in post-order: children before
their group, so every group header knows its children's addresses. A dataset is its header followed directly by its
contiguous data (if any). Address of an object = EOF + its offset in the block.

Header splitting (one rule): message `m` goes into the current chunk if `used + size(m) + (m is the last message ?
reserve : 20) ≤ 65535`; otherwise a Continuation message closes the chunk (body = used + 20, free space 0, no NIL, no
gap) and points to the next chunk, which follows directly in the block. So free space only ever exists in the last
chunk, and the last chunk always holds its reserve. Splitting happens only for channel groups with more than ≈ 2500
snapshots and `Data` groups with very many datasets. The *reserve* is free space for later growth at the end of the
last chunk, filled with a NIL message (or a gap / nothing by the tail rule):

| Header | Reserve in its last chunk |
|--------|---------------------------|
| channel group without `Data` | 19 bytes (exactly the `Data` link; after its insertion the free space is 0) |
| channel group with `Data`, snapshot groups, datasets | 0 |
| `Data` group | 20 (a Continuation message) |

Channel block of `hdf5_write` (offsets increasing):
1. for `n_snap > 1` with snapshot data, for each `i`: the snapshot datasets of snapshot `i` (order of 1.3), then the
   header of `Snap_i`;
2. the channel-level datasets (`CenterFrequency`, `tx_position`, `rx_position`, `tx_orientation`, `rx_orientation`;
   for `n_snap = 1` then the snapshot datasets);
3. the unstructured datasets in `par_names` order, then the `Data` group header;
4. the new root directory chunk (if needed), then the new root link chunk (if needed), in this order (3.4; the
   directory chunk names the link chunk, so the link chunk must follow it, 4.1 rule 3);
5. the channel group header (attributes in the order of 1.2, then links in the order: channel-level datasets or
   snapshot datasets, `Snap_0…`, `Data`).

Item 4 must precede item 5: the link chunk that receives the channel's root link then always lies before the channel
group, which the link removal relies on (3.4).

### 3.3 Growable group headers: channel and `Data` groups (chain)

Only two kinds of link insertion happen after a group header is written: the `Data` link into a channel group, and a
dataset link into a `Data` group (`hdf5_write_dset`). Both use the chain rule:
- The writer reads the whole header (chunk 0 and its continuation chunks, 4.1 rule 3), checks names, and takes the last
  chunk in reading order. Its free space is computed, not assumed: `F` = body − Σ sizes of all non-NIL messages of
  the chunk (a rewrite compacts the chunk: its non-NIL messages in order, then the new message, then the tail of
  2.3). In files written by the library `F` is the reserve or the NIL at the end; after edits by other tools
  (libhdf5 turns deleted messages into NIL messages, possibly several, and fills free space with its own messages)
  it may be anything. `L` = message size of the new link.
- Channel group (only the `Data` link, `L = 19`, is ever inserted): insert in place if `L ≤ F`, else throw the
  `Unsupported HDF5 file` message (4.2) before the file is changed. There is no continuation branch for channel
  groups.
- `Data` group: insert in place if `F − L ≥ 20`; otherwise, if `F < 20`, throw `Unsupported` before the file is
  changed; otherwise a new continuation chunk is appended in the block, holding the link and a NIL message, with body
  `max(L + 20, min(16384, 2 · body of the last chunk))`; the last chunk gets a Continuation message to it (20 bytes of
  its free space; tail rule for the rest) and is rewritten in place.
- For a `Data` group the Link Info message must have flags 3 and a max creation index < 2^63 (else `Unsupported`);
  it gets max creation index + 1, and the new link gets the old value as its creation order; chunk 0 is rewritten
  (together with the link if chunk 0 is the last chunk).

### 3.4 Root links: directory and link chunks

Root links to channel groups are never stored in root chunk 0. They live in *link chunks* (continuation chunks
holding only Link messages and the tail), which are referenced from a *directory*: the Continuation messages in
root chunk 0 after the four dataset links (16 entries of 20 bytes), continued by *directory chunks* (continuation
chunks holding only Continuation messages and the tail; body 16380 = 819 entries).

- Directory entries are appended in order. A directory chunk (chunk 0 or a directory chunk) takes a link-chunk entry
  while at least 40 bytes are free; when only 20 bytes are free, the next link chunk is entered into a new directory
  chunk, and the last 20 bytes receive the Continuation message to that new directory chunk. Hence: a directory
  chunk without free space is full and its last entry points to the next directory chunk; every other entry points
  to a link chunk. Chunk 0 holds up to 15 link-chunk entries, each directory chunk 818. Entries are read in order;
  each must start at or after the end of the chunk named by the previous entry (4.1 rule 3).
- Link chunk `k` (counted from 0 over all link chunks) has body `min(1024 · 2^k, 16384)`. A new link is appended to
  the last link chunk if it fits (`L ≤ F`, tail rule for the rest); otherwise a new link chunk is created in the
  block (with the link and the tail), and its directory entry is inserted in place (3.7, step 3).
- Link chunks and directory chunks lie in the file in creation order (increasing addresses), and every root link is
  appended to the last link chunk existing when its channel group is written, which lies before that group (3.2).
  Therefore the link of the channel group at address `A` is in the link chunk with the largest address below `A`
  (binary search over the link-chunk entries). The writer removes the Link message whose target address is `A`; if
  there is none (not possible in files written by the library), nothing is removed.
- New channel ID: channel IDs only grow, links are appended in ID order, and the newest link is removed only after a
  newer one was appended. So the last Link message of the last link chunk names the largest ID ever linked; the new
  ID is that ID + 1 (1 if there is no link chunk; a last link chunk without links is `Unsupported`, 4.1 rule 8). This equals v2's "largest ID in `Order` + 1, skipping existing
  group names" for every file written by the library, without reading `Order`. ID > 2^32 − 1 throws
  `std::runtime_error("No free channel ID left in the HDF5 file.")`.

Costs (chunk bodies of 16 KB; lookup3 ≈ 0.35 ms/MB): at 20 000 channels the links (≈ 28 bytes each) fill ≈ 40 link
chunks; a write reads chunk 0, all directory chunks (one below ≈ 480 000 links) and the last link chunk, i.e.
≤ 34 KB (directory and link chunks are read directly, 3.6; `open` adds the 64 KiB header-window loads for root
chunk 0 and the root dataset headers), and rewrites one chunk. At 10^6 channels: 3 directory chunks; a write reads
≈ 66 KB and checksums ≈ 82 KB (≈ 30 µs); an overwrite additionally reads one link chunk. A chain without directory would read all ≈ 28 MB of root links per write at
10^6 channels. libhdf5 reads such root groups correctly and fast (10^6 links: H5Fopen 1 ms, first lookup by name
0.2 s, iteration 0.4 s; 7116 continuation chunks tested).

### 3.5 Writer data flow

All functions that change a file first read and validate everything they will need or modify (superblock, root,
the slot's entries, directory and link chunks, channel and `Data` headers) and validate the inputs, then emit the
block (3.2) and perform the remaining writes of 3.7 without reading again. A malformed file therefore throws `std::invalid_argument` before it
is changed. When the same chunk is changed twice (e.g. the new root link and the removed old link are in the same
link chunk), the second rewrite starts from the image after the first change.

The file is opened unbuffered (`setvbuf(f, nullptr, _IONBF, 0)`), so every write reaches the operating system in
program order. There is no `fsync`: the order protects against an interrupted process (exception, crash, kill), not
against power loss (as libhdf5 and v2).

### 3.6 Reader data flow

Reads whose length is known in advance go directly into their destination: the superblock, contiguous data (always
> 4096 bytes, 2.6), continuation chunks (length from the Continuation message, so also directory and link chunks),
`Order`/`ChannelRef`/`ChannelDims` data and entries. Only chunk 0 of an object header, whose size is known only after
its prefix, is read through a 64 KiB window in the file handle: a header read that is not inside the window loads
`[addr, addr + 64 KiB)` (limited to the file size) and copies from it. Within a channel block the reader visits a
snapshot's datasets in link order, i.e. in increasing addresses, so consecutive header reads hit the window.
Contiguous float data is read straight into the Armadillo memory (`dtype = float`) or through one float buffer per
dataset (`dtype = double`). Any write drops the window. (A 1 MiB centered window was measured to cost 80–370 µs per
load and about four loads per call; the 64 KiB forward window costs 6–26 µs.)

### 3.7 Update order and interrupted writes

`hdf5_write` (and `hdf5_write_dset` on an empty slot, without step 5):

| Step | Writes | If interrupted after this step: library reads | libhdf5 / h5py / MATLAB see |
|------|--------|---------------------------------------------|----------------------------|
| 1 | the block at the stored EOF | old state (bytes past EOF are ignored and overwritten by the next write) | old state; a read-write close truncates the bytes |
| 2 | superblock: EOF = end of block, checksum | old state; the block is dead space | old state |
| 3 | one chunk rewrite: the new link in the last link chunk, or the directory entry of a new link chunk (in chunk 0 or the last directory chunk) | old state | the new group `/channel_<new>` in addition (not referenced by `Order`) |
| 4a | `ChannelRef[slot]` (8 bytes) | read functions return the new channel; `hdf5_read_layout` returns the old ID | both groups |
| 4b | `Order[slot]` (4 bytes) | new state | both groups (the old one unreferenced) |
| 5 | overwrite only: the link chunk holding the old link, rewritten without it | new state | new state |

Every intermediate state is a valid file for libhdf5 and for the library, and the next write works: new IDs come
from the last root link, so an orphaned link from an interrupted write is never reused. Leftovers are dead space
(D7): an unreferenced group (after 3 or 4a) or a still-linked old group (after 4b). After an interruption between 4a
and 4b, a later overwrite of that slot removes the link of the group `ChannelRef` points to; the older group stays
linked. `Order` stays stale after such an interruption until the slot is written again with `hdf5_write`
(`hdf5_write_dset` on a filled slot never writes `Order`, 5.3): for a slot first filled by an interrupted
`hdf5_write_dset`, `Order[slot]` remains 0, so `hdf5_read_layout` (and the MEX/Python `hdf5_read_channel` wrappers,
which select slots by it) report the slot as empty while `hdf5_read_dset` and `hdf5_read_channel` return its data.
This leftover is accepted (it needs a crash between two adjacent small writes) and is not repaired automatically.

`hdf5_write_dset` on a filled slot: step 1 block (dataset header and data; plus a new `Data` group, or a new chain
chunk), step 2 superblock, step 3: either the channel group chunk that receives the `Data` link (one rewrite), or
3a `Data` chunk 0 (max creation index + 1, plus the link or Continuation if chunk 0 is the last chunk) and 3b the last
chunk (link or Continuation). After 3a alone the creation index has a gap, which is harmless. Readers see the dataset
after step 3 (3b).

`hdf5_reshape_layout`: one 16-byte write of the `ChannelDims` data. `hdf5_create`: no guarantee (the file is
truncated first).

A torn chunk write (power loss in the middle of step 3 or 5) breaks that chunk's checksum: libhdf5 can then no longer
iterate the root group; the library's read functions are not affected (they use `ChannelRef`), but a later write
that needs the damaged chunk throws `std::invalid_argument`. A torn superblock makes the file unreadable for both.

---------------------------------------------------------------------------------------------------------------------

## 4. Reader validation and errors

### 4.1 Rules (D4, D6)

Applied by every function to everything it reads; all violations throw `std::invalid_argument`.
0. Superblock, in this order: file ≥ 8 bytes and signature (else `Not an HDF5 file.`), version byte = 2 (else
   unsupported, 4.2), then the remaining fields of 2.1 and the checksum.
1. Bounds: every address and length is checked against EOF before anything is read or allocated, in the
   overflow-safe form `length ≤ EOF && addr ≥ 48 && addr ≤ EOF − length`; `addr + length` is never computed before
   this check (`UNDEF` is an ordinary value in the file). The same form is used by `file::read` (against the file
   size) and for every `offset + size` inside a chunk image. Products are checked as `a != 0 && b > max / a`, without
   builtins. EOF ≤ file size.
2. Checksums: superblock and every chunk read (also by writers) are verified.
3. Chunks: chunk-0 size-field width code 0 or 1; every chunk (chunk 0 and continuation) has a body ≤ 65535, i.e. a
   chunk image of at most 65543 bytes and a continuation length in `[8, 65543]` (else `Malformed`); messages must fit the body; 1–3 trailing bytes
   are the gap; a gap together with a NIL message is rejected. Continuation chunks are read in the order their
   Continuation messages are found (queue); each must start at or after the end of the chunk read before it. This rules out cycles and repeated chunks and bounds the bytes read per header by the file
   size.
4. Messages: only the types of 2.4; Link Info version 0 with flags 0 or 3 and fractal heap / name index `UNDEF`
   (dense storage → rejected); Group Info version 0, flags 0; Link version 1, hard links only (flags bits 3, 4 = 0);
   Attribute version 3 with flags 0; Dataspace version 2, flags 0, type scalar with rank 0 or simple with rank 1–3;
   Datatype as in 2.5; Fill Value version 3 (content ignored); Data Layout version 3, class compact or contiguous
   (chunked/virtual → rejected); Attribute Info version 0, flags 0, fractal heap and name index `UNDEF` (dense
   attribute storage → rejected); message flag bit 1 (shared) → rejected. The data bytes of NIL messages are
   ignored (libhdf5 leaves stale bytes in them after its own edits).
5. Objects: a dataset has exactly one Dataspace, Datatype and Data Layout message (Fill Value optional), no Link or
   Attribute messages; a group has exactly one Link Info and Group Info message, at most one Attribute Info message
   (exactly one if it has Attribute messages), and no Dataspace/Datatype/Layout. Datasets have no Attribute Info.
   Data size: `n_elem · element size` (overflow-checked) must equal the compact size or the contiguous size; a
   contiguous size 0 needs address `UNDEF`, otherwise the bounds of rule 1. Attributes likewise (value size).
   Total: per call, the reader sums the stored raw size of every dataset it reads (compact and contiguous); if the sum
   exceeds EOF → `Malformed HDF5 file: the channel references more data than the file holds.` This bounds the sum of
   all allocations by the file size, also when several links point to the same object (library-written files never
   share targets).
6. Schema: as in section 1 with the v2 checks: the root links and dataset types/shapes of 1.1 (`Order` and
   `ChannelRef` contiguous with `n_slot` elements, `n_slot` = product of `ChannelDims` ≤ 2^32 − 1); `ChannelRef`
   entries are 0 or point to a group header inside EOF; channel attributes with exactly their type and value count
   (`NumSnap` required; `NumTx`, `NumRx`, `tx_position`, `rx_position` required if `NumSnap > 0`; others optional);
   `Name` a 256-byte fixed string; snapshot datasets with the extents of 1.3 derived from `NumPath`, `NumTx`,
   `NumRx`, except `delay`, whose extents 1 and 2 are not checked (v2: any `[n_rx', n_tx', np]` is read back);
   `coeff_re` and `coeff_im` both present or both absent; `interact_coord` with `sum(no_interact)` rows; for
   `NumSnap > 1` either no `Snap_<i>` links or exactly `Snap_0 … Snap_<NumSnap−1>` (`Snap_` + 1–10 decimal digits,
   no leading zero, value < `NumSnap`, no duplicates), every snapshot with the same set of datasets as `Snap_0`; `Data` links point to datasets of the types of 1.4; strings are scalar. Unknown links in a
   channel group are ignored. Allocations sized by `NumSnap` happen only after the snapshot links were counted.
7. Version: the `Version` value must be 3.
8. Root structure (checked by the writers; the read functions never need it, D3). Every `open(fn, update)` (5)
   checks the chunk-0 part on the root chunk-0 image it already holds, so no writer changes a file with a foreign
   root; the directory-chunk and link-chunk parts are checked by the writers that read those chunks (`hdf5_write`,
   `hdf5_write_dset` on an empty slot). Root
   chunk 0 holds exactly Link Info (flags 0), Group Info, the links `ChannelDims`, `Order`, `Version`, `ChannelRef` in
   this order, then only Continuation messages, then the tail, with free space 0 or a multiple of 20; a directory
   chunk holds only Continuation messages and the tail, free space 0 or a multiple of 20; an entry names a directory
   chunk iff it is the last entry of a directory chunk without free space, and that chunk starts with `OCHK` followed
   by a Continuation message or the tail; every other entry names a link chunk, which holds only hard Link messages
   named `channel_` + 1–10 decimal digits (no leading zero, value 1 … 2^32 − 1) with increasing IDs, and the tail;
   the last link chunk holds at least one link. Checked are the chunks the writer reads (chunk 0, all directory
   chunks, the last link chunk, the link chunk found for the old group); the other entries are checked only for
   their order (rule 3), their targets are not read. Any violation (e.g. a link added to
   the root by libhdf5, which places it into the free space of chunk 0) → the `Unsupported HDF5 file` message:
   the file is valid HDF5 but outside the subset (D4).

### 4.2 Error messages

v2 messages are kept where the situation exists in v3 (several are pinned by the wrapper tests).

| Situation | Exception | Message |
|-----------|-----------|---------|
| file missing: `hdf5_read_channel`, `hdf5_read_dset`, `hdf5_read_dset_names`, `hdf5_reshape_layout` (`hdf5_read_layout` returns zeros; `hdf5_write`, `hdf5_write_dset` create the file) | invalid_argument | `File does not exist.` |
| file shorter than 8 bytes or wrong signature | invalid_argument | `Not an HDF5 file.` (pinned by MEX and Python tests) |
| HDF5 file outside the subset at file level: superblock version ≠ 2 (e.g. every v2 file, which libhdf5 writes with superblock version 0), offset/length sizes ≠ 8, base ≠ 0, extension defined, root chunk 0 without the four dataset links or with dense link storage, any violation of 4.1 rule 8 (writers), a growable group without room for the new link or with an unexpected Link Info (3.3, writers) | invalid_argument | `Unsupported HDF5 file: quadriga-lib reads only channel files of version 3 that it wrote itself; files written by quadriga-lib 0.12.4 and earlier (version 2) and files modified with other HDF5 tools are not supported.` |
| `Version` ≠ 3 | invalid_argument | `Unsupported channel file version <n>; this version of quadriga-lib reads version 3 only.` |
| `ChannelDims`/`Order`/`ChannelRef` inconsistent (product > 2^32 − 1, extents ≠ product, wrong type or layout) | invalid_argument | `Storage index in HDF file is corrupted.` |
| any other rule of 4.1 | invalid_argument | `Malformed HDF5 file: <what>.` e.g. `… checksum mismatch at address <a>`, `… address <a> is outside the file`, `… dataset 'coeff_re' has extent 1 in dimension 0, expected 50`, `… attribute 'NumSnap' is missing`, `… attribute 'Name' must be a fixed-length string of 256 bytes` (v2 texts for the semantic checks) |
| cannot open for reading, or the size cannot be determined | invalid_argument | `Error opening file.` |
| a seek or read fails while reading (in readers, and in writers before their first write; e.g. a file truncated by another process, a directory passed as `fn`) | invalid_argument | `Malformed HDF5 file: cannot read <n> bytes at address <a>.` |
| cannot open for writing | invalid_argument | `hdf5_write`: `Error opening file. It may be opened by another program.`; others `Error opening file.` |
| cannot create (`hdf5_create`) | invalid_argument | `Error creating file. It may be already opened.` |
| argument errors | invalid_argument | unchanged v2 texts: `Storage layout dimension sizes cannot be 0.`, `Storage layout has too many slots: nx * ny * nz * nw must not exceed 4294967295.`, `Index out of bound.`, `Mismatch in number of elements in storage index.`, `Parameter name must only contain letters, numbers and the underscore '_'.`, `NULL pointer was passed as data object.`, `Unsupported data type for '<name>'`, `Dataset '<name>' already exists.`, `Number of elements in 'par_data' must match number of elements in 'par_name'.`, `Too many snapshots.`, `Number of elements returned by 'n_path()' does not match number of snapshots.`, `Size mismatch between …`, `'<field>' must have 3 rows.` |
| dataset name longer than 65000 bytes (a link message must fit 16 bits) | invalid_argument | `Parameter name is too long.` (new; v2 had no practical limit) |
| a write, a seek before a write, or the close after writing fails (disk full, I/O error) | runtime_error | `Cannot write file '<fn>'.` (the only runtime_error besides the channel ID; all other file errors are invalid_argument) |
| no channel ID left | runtime_error | `No free channel ID left in the HDF5 file.` |

---------------------------------------------------------------------------------------------------------------------

## 5. Public functions

Common step `open(fn, mode)`: open (`rb` or `r+b`), size, superblock (4.1), root chunk 0; find `ChannelDims`,
`Order`, `Version`, `ChannelRef` by name among the Link messages of root chunk 0 (each exactly once, else
`Unsupported`); readers ignore other Link and Continuation messages there, `open(fn, update)` applies the chunk-0 part
of 4.1 rule 8. Then `Version` (value 3), `ChannelDims`, `Order`, `ChannelRef` headers. Result: layout `{nx, ny, nz, nw}`, `n_slot`, data addresses of `Order`
and `ChannelRef`, `ChannelDims` data address, EOF, root chunk 0 image (writers). Slot `(ix, iy, iz, iw)` outside the
layout: readers return "empty", writers throw `Index out of bound.`.

### 5.1 `hdf5_create(fn, nx, ny, nz, nw)`
1. Any dimension 0 → `Storage layout dimension sizes cannot be 0.`; product > 2^32 − 1 → too-many-slots error
   (before the file is touched).
2. `fopen(fn, "wb")` (truncates an existing file); failure → `Error creating file. It may be already opened.`
3. Write the image of 3.1 (headers in one buffer, zeros in ≤ 1 MiB blocks); close; failure → runtime_error.

### 5.2 `hdf5_write(ch, fn, ix, iy, iz, iw, assume_valid)`
1. v2 validation unchanged: `is_valid()` unless `assume_valid`; `channel_is_empty = !assume_valid && ch->empty()`;
   par count mismatch; empty channel without par → return 0 (no file is created); every par name (`par_` + name:
   alphanumeric/underscore, ≤ 65000 bytes, not duplicated within the channel → `Dataset 'par_<x>' already exists.`)
   and type checked.
2. File missing → `hdf5_create(fn)` (default layout). `open(fn, update)`; slot (throws if outside).
3. `old = ChannelRef[slot]`. Read the root directory (chunk 0 and all directory chunks), the last link chunk (new
   ID, 3.4), and for `old ≠ 0` the link chunk holding the old link; check them against 4.1 rule 8.
4. Validate the channel data in v2 order before emitting anything (if not `channel_is_empty`): `n_snap ≤ 2^32 − 1`
   (`Too many snapshots.`), `n_path()` count, rows of positions and orientations, coefficient sizes per snapshot.
   Build the block at base EOF (3.2): the channel data (1.2, 1.3); par datasets and `Data` group if `par_names` is not
   empty; root directory/link chunks; channel group with defaults overwritten by the channel values.
5. Steps 1–5 of 3.7. Close (flush errors → runtime_error). Return `old ≠ 0`.

### 5.3 `hdf5_write_dset(fn, par_name, par_data, ix, iy, iz, iw, prefix)`
1. `name = prefix + par_name`; name and type checks (as `qHDF_check_par` in v2, plus the length limit) before the
   file is touched. File missing → `hdf5_create(fn)`. `open(fn, update)`; slot.
2. `ChannelRef[slot] = 0` (empty slot): new ID as in 5.2; block: dataset, `Data` group (creation index 1, link with
   creation order 0), root directory/link chunks, channel group with default attributes and the `Data` link
   (reserve 0); steps 1–4 of 3.7.
3. Filled slot: read the channel group header; no `Data` link → block: dataset and a new `Data` group; step 3 inserts
   the `Data` link into the channel group (3.3; uses the 19-byte reserve). `Data` link present → read the `Data`
   header; an existing link named `name` → `Dataset '<name>' already exists.` (nothing written); block: dataset (and a
   chain chunk if needed); steps 3a/3b (3.7).
4. `Order` and `ChannelRef` are not changed for a filled slot.

### 5.4 `hdf5_reshape_layout(fn, nx, ny, nz, nw)`
File missing → `File does not exist.`; new product > 2^32 − 1 → too-many-slots error (before opening);
`open(fn, update)`; product of the stored layout ≠ new product → `Mismatch in number of elements in storage index.`;
write the 16 bytes of `ChannelDims`; close.

### 5.5 `hdf5_read_layout(fn, channelID)`
File missing → `{0, 0, 0, 0}` (`channelID` untouched, v2). `open(fn, read)`; if `channelID` is given, `set_size`
to `n_slot` and read all of `Order` into it. Return the layout.

### 5.6 `hdf5_read_channel<dtype>(fn, ix, iy, iz, iw)`
File missing → `File does not exist.`; `open(fn, read)`; slot outside, or `ChannelRef[slot] = 0` → empty channel.
Read the channel group. `Data` present: every link whose name starts with `par_`, in creation order (the links are
sorted by their creation-order field, stable; equal to the message order in library files, but libhdf5 may move
messages when it edits a group):
dataset → `std::any` (as 5.8), name without prefix. Then `Name` (the bytes before the first NUL, v2; missing → ""), `NumSnap`, `Initial_position`
(optional); `NumSnap = 0` → return. Then exactly v2: `NumTx`, `NumRx`; `CenterFrequency` dataset `[n_snap]` or
attribute (NaN → empty); positions/orientations dataset `[n_snap, 3]` or attribute `[3]` (orientations optional);
snapshot data from the channel group (`n_snap = 1`) or `Snap_<i>` (4.1 rule 6): `NumPath` (optional, default 0) and
the datasets of 1.3 with their extents checked; `interact_coord` read only if `sum(no_interact) > 0`, otherwise
`[3, 0]`. Structured data converted from float32 to `dtype`. v2's padding of missing snapshot groups (files of
older library versions) is dropped.

### 5.7 `hdf5_read_dset_names(fn, par_names, ix, iy, iz, iw, prefix)`
File missing → `File does not exist.`; empty/outside slot or no `Data` → 0. Names of `Data` links starting with
`prefix`, in creation order (sorted by the creation-order field, as 5.6), without the prefix, *appended* to `*par_names` if not `nullptr` (v2). Returns the
count. The link targets are not opened.

### 5.8 `hdf5_read_dset(fn, par_name, ix, iy, iz, iw, prefix)`
File missing → `File does not exist.`; empty/outside slot, no `Data`, or no link `prefix + par_name` → empty
`std::any`. Otherwise the dataset (4.1 rules 4–6): scalar → `T`; rank 1 → `arma::Col<T>`; rank 2 →
`arma::Mat<T>(dims[1], dims[0])`; rank 3 → `arma::Cube<T>(dims[2], dims[1], dims[0])`, `T` = float, double,
unsigned long long, long long, unsigned, int; string → `std::string` without trailing NUL bytes (inner NULs kept).
A dataset of another type or shape throws (D4; v2 returned an empty `std::any`, which only happened for files
written by other tools).

### 5.9 Behavior parity with v2 (pinned by the P1 tests; all must hold unchanged)

- create: layout up to 4D; default 65536×1×1×1; overwrites existing files; zero dimension and > 2^32 − 1 slots throw
  before the file is created; uncreatable or read-only file → invalid_argument.
- read_layout: returns `{nx, ny, nz, nw}`; missing file → zeros, no error; non-HDF5 file → invalid_argument
  (`Not an HDF5 file`); `channelID` has `n_slot` entries, 0 = empty, stale content replaced.
- reshape: total count must stay (incl. 0 and > 2^32 − 1 → invalid_argument, layout unchanged); slots keep their
  linear order; missing / non-HDF5 / read-only file → invalid_argument.
- write: single precision for float and double input, values exactly `(R)(float)x`; file created with the default
  layout; returns 0 / 1; overwrite replaces the slot completely (old group and all its `par_*`, also from
  `hdf5_write_dset`, gone for libhdf5 too); index outside → invalid_argument (each dimension); `assume_valid`;
  name ≤ 255 bytes (UTF-8 bytes kept, 300 rejected by validation, truncated with `assume_valid`); unstructured types
  and ranks of 1.4; slot index formula; new ID = largest + 1, overwrite gets a new ID, failed writes use no ID;
  empty channel without par → 0 and no file; empty channel with `assume_valid` is written; par-only channel keeps the
  defaults (no name); `initial_position`-only and positions-only channels round trip (no snapshot groups); writer-side
  rejections (coefficient size mismatch, positions without 3 rows) and duplicate par names leave slot and IDs
  unchanged; unsupported par types (`cx_mat`, `std::vector`, `char`, `const char*`, `unsigned short`, empty
  `std::any`) → invalid_argument.
- read_channel: empty slot or index outside → empty channel; every per-snapshot field one entry per snapshot,
  empty snapshots keep their shape (`delay` `[n_rx, n_tx, 0]` or `[1, 1, 0]`, LOS-only `interact_coord` `[3, 0]`);
  missing `center_frequency` → empty; empty orientations → `[3, 1]` zeros; unstructured data in write order, only
  `par_` datasets, no type conversion; float/double read parity.
- read_dset: empty `std::any` for missing slot/group/name or index outside; scalar/Col (also 1 element)/Mat/Cube;
  Row → Col; strings without trailing NULs, empty string → empty, inner NUL kept.
- read_dset_names: names without prefix in write order; prefix selects; `nullptr` allowed; empty/outside slot → 0.
- write_dset: empty slot gets a new ID, filled slot keeps its ID; `Data` created if missing; appended in creation
  order; custom prefix; name and type errors before the file is changed; existing name → invalid_argument; a later
  `hdf5_write` replaces these datasets.
- Error types: all argument and file errors `std::invalid_argument`; read-only file for every writer →
  invalid_argument.
- libhdf5 sees exactly the documented tree (interop test): object set, attribute sets, types, dataspaces, values,
  NaN/zero defaults, creation order in `Data`, `ChannelRef` dereferencing to the slot's group, null for empty slots.

Intentional differences (D4, D8; none is pinned by a P1 test outside the v2-specific cases):
- v2 files and files modified by other HDF5 tools are rejected; variable-length and NUL-padded strings, other
  numeric types and rank > 3 in `Data` are no longer read (throw instead of reading / returning an empty `std::any`).
- No file locking (D8): v2 (libhdf5 ≥ 1.10) refused files locked by another libhdf5 process; v3 does not.
- Corrupt files: different `Malformed HDF5 file: …` texts; still `std::invalid_argument`.
- The writer never reads all of `Order`; the "same ID in several slots" check of v2's link deletion is dropped (not
  reachable through the API).
- `hdf5_read_layout` without `channelID` and `hdf5_reshape_layout` validate `Order` and `ChannelRef` too (v2 read only
  `ChannelDims`): a broken index gives `Storage index in HDF file is corrupted.` (from reshape, v2 could give
  `Mismatch …`).
- `hdf5_read_dset_names` does not open the link targets (v2 skipped non-dataset links; library files have none).
- Attributes must have exactly their documented types (v2 converted any numeric type).
- Duplicate par names in one channel throw before a missing file is created (v2: after creating it).
- `Parameter name is too long.` for names > 65000 bytes (new limit).
- `hdf5_write` on files changed by other tools may throw `Unsupported HDF5 file` where v2 wrote (3.3, 4.1 rule 8).

---------------------------------------------------------------------------------------------------------------------

## 6. `src/qd_hdf5.hpp`

Private, header-only, not installed, included only by `src/qd_channel_hdf5.cpp` and the internal tests. Namespace
`qd::hdf5` (as `qd::` in `bits.hpp`). Free functions `static inline`; no global or static mutable state (D8). The
file header explains: purpose (write and read the subset of HDF5 used by quadriga-lib channel files), the subset
(sections 2 and 3.3–3.4 of this document in words, without referencing it), the published specification it follows
(*HDF5 File Format Specification Version 3.0*, section names), and compatibility (libhdf5 ≥ 1.8). Contents and
ownership:

| Name | Responsibility |
|------|----------------|
| constants | `undef`, `max_chunk_body` 65535, `compact_limit` 4096, directory and link-chunk sizes (Appendix A) |
| `lookup3(const uint8_t*, size_t)` | checksum (2.2) |
| `put_le`, `get_le` | little-endian integer encoding into a `std::vector<uint8_t>` / from a pointer |
| `enum class type_t {f32, f64, u64, i64, u32, i32, str, ref}`, `struct dtype {type_t t; uint32_t size;}` | element types; `encode_datatype`, `decode_datatype` (table 2.5, exact match) |
| `class file` | owns a `FILE*` (opened `rb`, `r+b` or `wb`, unbuffered; UTF-8 names, 7); `static bool exists(fn)`; `size()`; `read(addr, dst, n)` (direct) and `read_header(addr, …)` through the 64 KiB window (3.6) and the overflow-safe bounds check of 4.1 rule 1 against the size given at open; seek/read failures → `invalid_argument`; `write(addr, src, n)` (`runtime_error`); `close()` (`runtime_error` on failure; the destructor closes silently); 64-bit offsets (7) |
| `superblock` | `{eof, root}`; `read_superblock(file&)` (4.1), `superblock_bytes(eof, root)` |
| `struct chunk` | address, length, OHDR or OCHK, owned image bytes, message list (type, flags, offset, size), free space at the end; `read_chunk(file&, addr, length_or_0_for_ohdr, eof)` (verifies), `encode_chunk(...)` (messages + body size → image with tail rule and checksum) |
| `struct object` | the chunks of one header in reading order; `read_object(file&, addr, eof)` (4.1 rule 3) |
| `struct link`, `attribute`, `dataset`, `group` | decoded views (name, target, creation order / type, dataspace, value bytes / layout, data location); `decode_group(const object&)`, `decode_dataset(const object&, eof)`; views point into the object's images and are valid while the object lives |
| `read_raw(file&, const dataset&, void* dst)` | compact copy or contiguous read |
| `class header` | object-header builder: `dataspace`, `datatype`, `fill_value`, `layout_compact`, `layout_contiguous`, `link_info`, `group_info`, `attribute_info` (added automatically before the first attribute), `link`, `attribute`,
`continuation`; `emit(std::vector<uint8_t>& block, uint64_t base, size_t reserve)` → address of chunk 0 (splitting, 3.2) |
| `emit_dataset(block, base, dtype, rank, dims, data, n_bytes, force_contiguous)` | dataset header + contiguous data (2.6) |
| `struct patch {uint64_t addr; std::vector<uint8_t> bytes;}` | an in-place rewrite to be written later in a given order |
| `class chain_links` | growable group (3.3): `load(file&, addr, eof)`, `find(name)`, `prepare_insert(link message, block, base) → patches` (computes the real free space with compaction; channel groups: insert into the reserve or throw `Unsupported`; `Data` groups: insert, new chunk, or throw; keeps the Link Info creation index) |
| `class root_links` | root directory (3.4): `load(file&, chunk0, eof, all_directory_chunks)` (validates 4.1 rule 8), `last_link()`, `prepare_insert(...) → patches` (new link chunk and directory chunk go into the block), `prepare_remove(target address) → patch` |

What stays in `src/qd_channel_hdf5.cpp`: the schema (names, attribute defaults, field-to-dataset mapping, ID naming
`channel_<id>`, parsing the ID of the last root link), the public functions and their MD blocks, argument and
channel validation, the `open` step of section 5, block assembly for channels and unstructured data
(`any_type_id` mapping), reading channels and `std::any` values, the update order of 3.7, and all error messages of
4.2 except the generic `Malformed HDF5 file` texts raised in `qd_hdf5.hpp`. File-local helpers keep the `qHDF_`
prefix; helpers used once are lambdas, except self-contained steps of more than ≈ 20 lines, which may be named
file-local functions directly above their single caller (`md/SPEC_CPP_API.md`, "File-local helpers"); helpers used by
several functions are `static inline`.

Not in the header: anything about libhdf5, symbol tables, B-trees, heaps, dense storage, chunked layouts, filters,
variable-length types, attribute dense storage, superblock versions other than 2, user blocks.

---------------------------------------------------------------------------------------------------------------------

## 7. Portability

- C++17; GCC, Clang, MSVC; no compiler builtins (`__builtin_mul_overflow` of v2 is replaced by explicit checks), no
  POSIX-only calls in library code; no warnings with `-Wall -Wextra -Wpedantic`.
- 64-bit file offsets: `_fseeki64`/`_ftelli64` under `_WIN32` (MSVC and MinGW), `std::fseek`/`std::ftell` elsewhere
  with `static_assert(sizeof(long) >= 8)` (64-bit Linux and macOS).
- Little-endian hosts only: `#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__` →
  `#error` (all MSVC targets are little-endian). Encoding uses explicit byte shifts; bulk data (floats, integers,
  references) is copied as raw bytes.
- MATLAB libstdc++ limit: nothing newer than GLIBCXX_3.4.28 (GCC 10) in library code: no `std::from_chars` /
  `std::to_chars` for floating point, no `std::filesystem` outside the `_WIN32` branch below (the limit concerns the
  Linux MATLAB; file existence by `file::exists`); decimal parsing of
  channel and snapshot numbers by a digit loop. Build with g++-12 and check the MEX files' GLIBCXX versions.
- File names are UTF-8 (as libhdf5 1.14 treats them on Windows, and as the Python wrapper passes them). Under
  `_WIN32` only: `std::filesystem::path p = std::filesystem::u8path(fn)` and `_wfopen(p.c_str(), mode)` (wide mode
  string, e.g. `L"rb"`); no `<windows.h>`. Elsewhere `std::fopen(fn.c_str(), mode)`. The conversion lives in one place,
  `qd::hdf5::file` (open and the static `file::exists(fn)` used for the existence checks of section 5). An invalid
  UTF-8 name (`u8path` throws or the open fails) → `Error opening file.`; for `file::exists` → `false` (then
  `File does not exist.` where v2 checked existence).
- No OpenMP in the HDF5 code; reading different files from parallel threads is safe; one writer per file (D8).

---------------------------------------------------------------------------------------------------------------------

## 8. Tests in P4

### 8.1 Allowed changes to the P1 tests (all other P1 tests pass unchanged)

| File | Change |
|------|--------|
| `tests/catch2_tests/test_hdf_robustness.cpp` | deleted (`git rm`): the cases "hdf5_read_channel - malformed files throw instead of overflowing buffers", "hdf5 - no side effects on the caller's HDF5 state", "hdf5_read_dset - strings written by other tools" (v2-specific, libhdf5-based) and "get_HDF5_version - …" (D9) are removed; the cases of 8.2 replace them in the new file `test_hdf_robustness_internal.cpp` (`md/SPEC_CPP_API.md`: tests that include internal headers carry the `_internal` suffix) |
| `tests/mex_tests/test_hdf_rw.m` | remove the section `%% ===== HDF5 version =====` (`quadriga_lib.hdf5_version` check) and "and hdf5_version" in the header comment (D9) |
| `tests/catch2_tests/test_hdf5_interop.cpp` | the existing case stays unchanged; new cases of 8.4 are added; the header comment may mention them |
| `tests/catch2_tests/test_hdf_functions.cpp` | header comment only: line 8 names `test_hdf_robustness.cpp` → `test_hdf_robustness_internal.cpp` (and "files changed by other tools" → "files changed by other tools are rejected"); no test change |
| `tests/python_tests/test_hdf5_functions.py` | no change (checked for D9: no Python test calls `components()` or checks a version; `tests/benchmarks_mex/python_large_data_test.py` only prints `components()` and keeps working without the HDF5 line) |
| `tests/benchmarks_cpp/benchmark_hdf5.cpp` | only the compile comment (drop the optional libhdf5 part) |

### 8.2 `tests/catch2_tests/test_hdf_robustness_internal.cpp` (v3 byte-level corruption, no libhdf5; new file)

The file includes `qd_hdf5.hpp` only as a patch tool: to locate an object (root links, `ChannelRef`, child links)
and to rewrite a modified chunk with a correct checksum (`read_object`, `encode_chunk`, `lookup3`), so that the
semantic checks behind the checksums are reached. The oracle is the documented behavior: `std::invalid_argument` (or
the stated result), no other exception, no crash, no allocation beyond the file size. In the sanitizer build these
cases run with `ASAN_OPTIONS=max_allocation_size_mb=1024:allocator_may_return_null=0`, so an allocation larger than
1 GB fails the test (ASan alone does not flag large allocations). Each case writes a fresh file with the public API, patches it, and removes it at the end. Cases (one
`TEST_CASE` per group, one `SECTION` per patch):

1. File level: empty file and 7-byte file → `Not an HDF5 file.`; wrong signature byte → same; superblock checksum
   byte flipped → invalid_argument; superblock version byte set to 0 → the `Unsupported HDF5 file` message (checked
   with `CHECK_THROWS_WITH`); EOF > file size (last 100 bytes cut) → invalid_argument from read_layout, read_channel
   and write; root address beyond EOF → invalid_argument; extension address 0 → unsupported.
2. Accepted variants: consistency flag bit 0 set (re-signed) → all functions work; 4 KB of garbage after EOF →
   reads work, a following `hdf5_write` appends at EOF and the file then reads back completely.
3. Version and index: `Version` 2 and 4 (compact data patched, header re-signed) → exact message
   `Unsupported channel file version 2; …`; root link `Version` removed → unsupported; `ChannelDims` = {3,1,1,1}
   with 4 slots (contiguous data, no checksum) → `Storage index in HDF file is corrupted.` from every function (from
   the old side-effects case); `ChannelDims` product > 2^32 − 1 → same; `Order` contiguous address beyond EOF →
   invalid_argument; `ChannelRef[slot]` = EOF + 8, = address of a dataset header, = address inside raw data,
   = `UNDEF` → invalid_argument; `ChannelRef[slot]` = 0 → empty channel (no throw); a `ChannelDims` entry 0 →
   corrupted index. Bounds overflow: a contiguous layout with address `UNDEF` and size 16, and with address 2^63 and
   size 2^63 → invalid_argument without allocation.
4. Object headers: chunk-0 size field enlarged beyond EOF (re-signed) → invalid_argument without allocation; body
   > 65535 → invalid_argument; message size beyond the body; unknown message type (e.g. 0x11 symbol table) in a
   dataset; shared-message flag; Continuation pointing beyond EOF, backwards (to the chunk itself) and twice to the
   same chunk; a Continuation length of 65544 (re-signed) and of several GB (no allocation); Link Info with a
   defined fractal-heap address; Attribute Info with a defined fractal-heap address,
   Attribute Info missing in a channel group, two Attribute Info messages, Attribute Info in a dataset; chunked
   layout class; Datatype int16 / NUL-padded
   string / float with another exponent bias; Dataspace with flags 1, with rank 4, null dataspace → each
   invalid_argument.
5. Channel semantics (re-signed, from the v2 malformed case): `NumPath` 50 → 1; `NumRx` 4 → 1; link `coeff_im`
   removed; `delay` dataspace of rank 1 and rank 4; `Name` as a 300-byte string; `NumPath` 2^30 with `coeff_re`,
   `coeff_im`, `delay` extents `{2^30, 2, 4}` but the original data (compact/contiguous size mismatch, no 64 GB
   allocation); `NumSnap` 4·10^9 with one `Snap_0` group (no 4·10^9-entry allocation); `Snap_1` missing in a
   3-snapshot channel; a link `Snap_07`; `interact_coord` with one row less than `sum(no_interact)`; compact `par_`
   data size ≠ elements × type size; a `Data` link pointing to a group; a rank-4 `par_` dataset → invalid_argument
   from `hdf5_read_channel` and `hdf5_read_dset` (changed from v2's empty `std::any`); `coeff_im` without `coeff_re`;
   a link `Snap_4294967296` and `Snap_` with 25 digits. Shared targets: in a 3-snapshot channel, `Snap_1` and `Snap_2`
   re-pointed to `Snap_0` with `Snap_0`'s data enlarged so that the sum of read sizes exceeds EOF → the
   `references more data than the file holds` message; the variant within the limit reads back without throwing.
   The unpatched file reads back.
6. Writers on malformed files: last root link chunk with a flipped byte, a `Data` chunk with a flipped byte, and the
   `ChannelDims` corruption → `hdf5_write`, `hdf5_write_dset`, `hdf5_reshape_layout` throw invalid_argument and the
   file is byte-identical afterwards. Root structure (4.1 rule 8, re-signed): a foreign link in chunk 0 after the
   directory entries; chunk 0 free space not a multiple of 20; in the *last* link chunk and in the link chunk holding
   an overwritten slot's link: a Continuation message, a link `channel_07`, `channel_4294967296`, decreasing IDs;
   a last link chunk without links → `hdf5_write` and `hdf5_write_dset` (empty slot) throw the `Unsupported HDF5
   file` message, file byte-identical; the foreign link in chunk 0 also makes `hdf5_write_dset` (filled slot) and
   `hdf5_reshape_layout` throw it (checked at `open(fn, update)`); read functions unaffected. A NIL message with non-zero data bytes is
   accepted. Chain growth (3.3, re-signed): a channel group created by `hdf5_write_dset` whose `Data` link was turned
   into a 19-byte NIL (free space 19) and one with free space 4 → `hdf5_write_dset` adds the `Data` link (19 ≤ F)
   resp. throws `Unsupported` (4 < 19), file byte-identical when it throws; a `Data` group whose last chunk has free
   space 10 and no room for the link → `Unsupported`; a `Data` group with Link Info flags 0 → `Unsupported`; a last
   chunk with two NIL messages (one in the middle) → compacted, the link is added and the group reads back in
   creation order.
7. Byte-flip sweep on a small file (2 slots, one 3-snapshot channel with all fields and 3 par datasets, one
   `hdf5_write_dset`-extended slot, one overwritten slot; ≈ 10 KB), one variant at a time: (a) every byte of the
   superblock and of every header chunk (positions from `read_object`, all chunks) flipped (`x ^ 0xff`) — this checks
   checksum verification; (b) every header byte flipped with `x ^ 0xff` and with `x ^ 0x01`, then the chunk re-signed
   with `lookup3` — this reaches the parsers (length fields, counts, extents, name lengths, message sizes,
   continuation addresses); (c) every byte of the `ChannelDims`, `Order` and `ChannelRef` data (not checksummed).
   Oracle: each read function returns normally or throws `std::invalid_argument`; on a copy, `hdf5_write_dset` to the
   filled slot and an `hdf5_write` overwrite either succeed and read back, or throw `std::invalid_argument` with the
   file byte-identical. Nothing else. Size: ≈ 9000 variants in (a), ≈ 18000 in (b), each with 4 read calls (≈ 15 µs)
   and 2 writer calls on a fresh copy (≈ 80–100 µs). To keep the case ≤ ≈ 2 s (release build), the read calls run on
   every variant of (a) and (c) and on every 2nd variant of (b), the writer calls on every 8th variant of (a), (b) and
   (c), with a fixed stride so the positions are deterministic and cover every chunk. The sanitizer build runs the
   same case with the allocation limit above (several times slower; acceptable there).
8. Interrupted writes (3.7). File A: 50 minimal channels in slots 0–49 (link chunk 0 with body 1024 holds 41 links:
   IDs 1–9 take 24 bytes, IDs 10–99 25 bytes), slot 50 empty. File B: A after an `hdf5_write` overwrite of slot 0. The
   old link is in link chunk 0, the new link is appended to link chunk 1; the test asserts this premise with
   `read_object` of the root (so a constant tuned in P4 cannot silently break it). So every step changes different
   bytes: block = B's bytes
   `[EOF_A, EOF_B)`; step 2 = B's superblock; step 3 = B's link chunk 1; step 4a = B's 8 bytes of `ChannelRef[0]`;
   4b = B's 4 bytes of `Order[0]`; step 5 = B's link chunk 0 (addresses from `qd_hdf5.hpp`: `read_object` of the root
   and of the datasets). States after 1, 2, 3, 4a, 4b are A plus these regions in the order of 3.7; the state after 5
   must equal B. For each state: `hdf5_read_layout` and `hdf5_read_channel` of all slots return the old or new content
   as in the table; then an `hdf5_write` to slot 50 and an overwrite of slot 0 succeed and read back. A second case
   with the old and new link in the same link chunk (2 channels) checks only the final state. Same for
   `hdf5_write_dset`: empty slot (steps 1–4b, including the stale `Order` of 3.7) and filled slot with a new `Data`
   group and with an append (steps 1–3b). (The libhdf5 view of these states: 8.4.)
9. Many channels: 8000 minimal channels (1 snapshot, SISO, 1 path) into an 8000-slot file (crosses from root chunk 0
   into the first directory chunk, 3.4), overwrite slots 0, 3999, 7999 and the last-written slot twice; IDs as in v2
   (largest + 1), all slots read back. Runtime target < 2 s.

### 8.3 `tests/catch2_tests/test_qd_hdf5_internal.cpp` (focused tests of the header)

- `lookup3`: the three vectors of 2.2 (independent oracle: published values and a libhdf5-written superblock).
- Datatype encodings equal table 2.5 byte by byte; `decode_datatype` rejects one-byte changes.
- Message bytes written by `header`: Attribute Info `15 12 00 04 00 00` + `ff`×16, placed after Group Info and
  before the first Attribute message, only in headers with attributes; message flags 0x01 on Datatype, Fill Value and
  Group Info.
- `encode_chunk` + `read_chunk`: round trip for free space 0, 1, 2, 3, 4, 20 (tail rule: nothing / gap / NIL);
  checksum and body size fields; a chunk with a gap and a NIL is rejected.
- `header::emit`: splitting at 65535 with continuation chunks directly after chunk 0; reserve in the last chunk;
  `read_object` returns all messages in order.
- `chain_links`: `Data` group: insert until three chunks exist; free-space invariant ≥ 20; creation index; message
  order = creation order. Channel group: the `Data` link fills the 19-byte reserve exactly (free space 0 afterwards).
- `header::emit` packing rule of 3.2 at the boundary: messages that fit exactly with the reserve, and one byte more.
- `file::exists` and open with a non-ASCII UTF-8 name (on Linux the bytes are passed through).
- `root_links` with the constants of Appendix A: the 16th directory entry is the next directory chunk; link chunk
  sizes 1024 … 16384; binary search finds the chunk for every link; removal keeps the order; `last_link()` after
  removals.
- `file`: reading beyond the size throws invalid_argument, also for `addr = UNDEF`, `n = 16` and `addr = n = 2^63`
  (overflow-safe check); header window: header reads inside, crossing the end of and outside the window return the
  file bytes; known-length reads bypass it; a write drops the window. 64-bit offsets: write and read at offset
  2^32 + 16 (`#if !defined(_WIN32)`, sparse file, removed afterwards).

### 8.4 `tests/catch2_tests/test_hdf5_interop.cpp` (optional, libhdf5 found by CMake)

The file uses only the public API and libhdf5; it does not include `qd_hdf5.hpp` (so it keeps its name). A case that
needs the internal header goes into `test_hdf_robustness_internal.cpp` instead. Unchanged case plus:
- attribute counts: for every object of the files written by the existing case (also its intermediate files),
  `H5Oget_info3(…, H5O_INFO_NUM_ATTRS).num_attrs` equals the number of names returned by `H5Aiterate2`;
- the intermediate states of the `hdf5_write` overwrite of 8.2 item 8, built here without the internal header: the
  same files A and B (public API); the changed regions are the byte differences between A and B over
  `[0, EOF_A)` plus the block `[EOF_A, EOF_B)` (EOF from superblock bytes 28–35); the `Order`/`ChannelRef` entry
  regions lie inside the data ranges given by `H5Dget_offset`/`H5Dget_storage_size`, the superblock is `[0, 48)`, and
  the two remaining regions are the link chunks — the higher address is step 3, the lower step 5 (by construction of
  A and B). Each state opens in libhdf5, the tree walk succeeds, and a read-write open/close leaves the file
  byte-identical when EOF = file size (and truncates to EOF otherwise);
- the 8000-channel file of 8.2 item 9: `H5Literate2` over the root finds exactly the expected names,
  `H5Oopen` by name of first/middle/last, `ChannelRef` dereferences;
- files modified by libhdf5. General oracle: each library function either throws `std::invalid_argument` or returns
  a result consistent with the modified tree (a deleted dataset is absent, all other data intact); never a crash or
  another exception; a writer either throws before changing the file or leaves a file that libhdf5 and the library
  read consistently. libhdf5 inserts into compact storage only while the group has fewer than 8 links (Group Info
  default), otherwise it converts the group to dense storage. Specific files and expectations:
  (a) `H5Gcreate2` of a root group in a file with ≤ 3 channels (root stays compact): read functions work, every
  writer except `hdf5_create` (which replaces the file) throws `Unsupported HDF5 file` at `open` (4.1 rule 8); the same in the file of the existing case (9 root links → dense
  root): every function throws `Unsupported HDF5 file`;
  (b) a variable-length string dataset added to a `Data` group with ≤ 6 links (stays compact): `hdf5_read_dset_names`
  lists it, `hdf5_read_dset` of that name and `hdf5_read_channel` throw; added to `/channel_1/Data` of the existing
  case (10 links → dense): all functions on that slot throw `Unsupported HDF5 file`, other slots work;
  (c) `H5Ldelete` of `par_note` in the two-dataset `Data` group of slot 2 of the existing case: `hdf5_read_channel`,
  `hdf5_read_dset`, `hdf5_read_dset_names` return only `values`; `hdf5_write_dset` to that slot appends and the slot
  reads back in creation order; libhdf5 reads the result;
- a file written by libhdf5 with default settings (superblock version 0, as every v2 file) → the
  `Unsupported HDF5 file` message.

---------------------------------------------------------------------------------------------------------------------

## 9. Build system, packaging, wrappers (D9, D11)

From `git grep -i -E 'hdf5|H5[A-Z]|h5repack|nodelete|libhdf'` (binary matches in pdf/png/zip files are false hits).

| File | Change |
|------|--------|
| `CMakeLists.txt` | remove: `HDF5_PATH`/`_HDF5_PATH` (l. 32–34), option `HDF5_STATIC` + comment (l. 39–41), `hdf5_version` (l. 51), `HDF5_ZIP` (l. 83), `BUILD_HDF5` (l. 155), MSVC prebuilt autodetect (l. 223–230), `find_package(HDF5 … C HL)` for the library (l. 232–244), Octave disabling for static HDF5 (l. 246–252), `_HDF5_PATH` handling (l. 254–263), `ExternalProject_Add(hdf5_dep …)` (l. 265–301), `HDF5_LINK_LIBS` (l. 303–307), `${HDF5_INCLUDE_DIRS}` in both libraries (l. 420, 452), `hdf5_dep` dependencies (l. 438–440, 654–656), shared-lib HDF5 linking (l. 477–485), `${HDF5_LINK_LIBS}` in MEX (l. 529, 545), Python (l. 660) and test (l. 714–716) links, test include `${HDF5_INCLUDE_DIRS}` (l. 702), install of `libhdf5` (l. 779–781); rewrite the comments on HDF5 linking and sonames (l. 196–203, 497–500) to the remaining facts (MEX files link the static library); interop hook (l. 682–694): keep `find_package(HDF5 MODULE QUIET COMPONENTS C)` for `test_bin` only, drop the `BUILD_HDF5`/`_HDF5_PATH` condition and update its comment; `-Wl,-z,nodelete` (l. 537–538, 547): **verify in MATLAB R2024a and R2026a with `clear mex` and MATLAB exit before removing** (build without the flag, call several `hdf5_*` and other MEX functions, `clear mex`, call again, `exit`, three times per release; remove flag and comment only if no crash, otherwise keep and reword the comment) |
| `GNUmakefile` | l. 3 "Set Armadillo and HDF5 sources" → "Set Armadillo sources" (it also documents `arma_internal`); remove l. 4 (comment) and `hdf5_internal` (l. 5), `-D HDF5_STATIC=…` (l. 47, 63), `rm -rf external/hdf5-prebuilt` in `tidy` (l. 248); keep `rm *.hdf5` in `clean` (test files) |
| `Makefile` (Windows) | remove the HDF5 prebuild variables (l. 13–20, keep Catch2/Armadillo), `hdf5` from `external:` (l. 61), the `hdf5` target (l. 63–70), `rmdir $(HDF5_PREBUILT)` in `tidy` (l. 139); fix the comment of l. 13 |
| `external/hdf5-1.14.6.zip` | delete (`git rm`) |
| `.gitignore` | remove `external/hdf5-*/` (l. 12) |
| `Dockerfile.ubuntu` | remove `libhdf5-dev` (l. 48) |
| `debian/control` | remove `libhdf5-dev` from Build-Depends (l. 13); the interop test is then skipped in the package build |
| `debian/rules` | remove `-DHDF5_STATIC=OFF` (l. 41, 59); comment l. 92: drop `libhdf5` |
| `debian/copyright` | remove HDF5 from the bundled-library list (l. 14) |
| `pyproject.toml` | no HDF5 entry; nothing to change (wheels no longer build HDF5 from the zip because the library no longer looks for it) |
| `README.md` | remove `libhdf5-dev` (l. 19) and `hdf5` from the conda line (l. 55) |
| `README_win.md` | `lib/` row (l. 14) and linking sentence (l. 65): only `quadriga.lib` |
| `tools/download_md/install_source.md` | remove `libhdf5-dev` (l. 14) and conda `hdf5` (l. 51) |
| `tools/download_md/insall_binaries.md` | l. 39: link only against `quadriga.lib` / `libquadriga.a` |
| `tools/acoustic/build_band_probe.sh` | drop the HDF5 include path, the rpath and `libhdf5.so` (l. 8, 10) and libhdf5's transitive dependencies `libcrypto.so libcurl.so libsz.so libz.so` (l. 11–12); keep `-ldl -lm -lpthread` |
| `tools/sos_port/review_prototypes/README.md` | drop `-I/usr/include/hdf5/serial` (l. 23), `libhdf5.so` and `-lcurl -lcrypto -lz -lsz` (l. 24) |
| `tools/thin_sheet_test/thin_sheet.ipynb` | stored `components()` output (l. 15, `"HDF5         = 2.1.0\n"`): remove that line from the stored output |
| `include/quadriga_channel.hpp` | remove `get_HDF5_version` and its comment (l. 188–189); comments as in section 10 |
| `src/qd_channel_hdf5.cpp` | rewritten (section 5, 6); `get_HDF5_version` and its MD block removed |
| `src/qd_hdf5.hpp` | new (section 6) |
| `api_mex/hdf5_version.cpp` | delete (D9); also delete the local build artifacts `+quadriga_lib/hdf5_version.mex*` so tests cannot pick up a stale file |
| `api_mex/hdf5_write_channel.cpp`, `api_python/qpy_hdf5_write_channel.cpp` | doc line "The file does not shrink; use `h5repack` …" (l. 20 / l. 21) → section 10 text |
| `api_python/qpy_components.cpp` | remove the `HDF5 = …` line (l. 40) |
| `tests/…` | section 8 |
| `html_docu/quadriga_lib_api_{cpp,mex,python}.md`, `html_docu/{cpp,mex,python}_api.html`, `html_docu/download.html` | regenerate with `make documentation` (never hand-edit); check that `get_HDF5_version`/`hdf5_version` are gone, the h5repack text is replaced and the install docs no longer mention HDF5 |
| unchanged on purpose | `src/any_type_id.cpp` (HDF5 mentions describe the file functions), `api_mex/get_channels_ieee_indoor.cpp`, `api_python/qpy_get_channels_ieee_indoor.cpp`, `api_python/python_*.hpp` comments, `md/SPEC_*.md`, `md/bug_hunt_brief.md`, `tools/prompts/…` (the files are still HDF5 files) |

Gate check after the purge: configure with `-DCMAKE_DISABLE_FIND_PACKAGE_HDF5=ON` builds the library, MEX, Python
and tests (without the interop test); `nm` of the library and of a MEX file shows no `H5` symbol.

---------------------------------------------------------------------------------------------------------------------

## 10. Public API documentation changes

MD blocks in `src/qd_channel_hdf5.cpp` (bullets added or replaced; everything else unchanged):
- `get_HDF5_version`: block removed.
- `hdf5_create`: add "- Files use the quadriga-lib channel file format version 3, an HDF5 file readable by any HDF5
  reader (h5py, MATLAB, HDFView). Modify files only with the quadriga-lib functions: files changed by other HDF5
  tools are rejected."
- `hdf5_write`: replace "HDF5 does not reuse the space of the removed data: the file does not shrink; use
  `h5repack` to reclaim the space." by "The space of the removed data is not reused, so the file does not shrink;
  to reclaim it, copy the channels into a new file ([[hdf5_read_channel]], [[hdf5_write]])." Add "- One writer per
  file at a time; files are not locked. Reading is safe while no one writes; different files can be read and
  written from parallel threads."
- `hdf5_write_dset`: add the same concurrency bullet (short form: "One writer per file at a time; files are not
  locked.").
- `hdf5_read_layout`: "throws if the file exists but is not a valid HDF5 file" → "… is not a channel file of version
  3 (see [[hdf5_read_channel]])".
- `hdf5_read_channel`: replace the throw bullet by "- Throws `std::invalid_argument` if the file is malformed (e.g.
  a dataset does not have the size given by the attributes of the channel, or a required dataset or attribute is
  missing) or is not a channel file of version 3 written by quadriga-lib: files written by quadriga-lib 0.12.4 and
  earlier (version 2) and files modified with other HDF5 tools are rejected."
- `hdf5_read_dset`: "Element types: float, double, 32- and 64-bit signed and unsigned integers." (drop "Datasets of
  other types or with more than 3 dimensions are returned as empty `std::any`."); "Strings are returned without
  trailing NUL characters (an empty string written by [[hdf5_write_dset]] comes back empty)." (drop the
  variable-length sentence).

Header comments in `include/quadriga_channel.hpp`: remove `get_HDF5_version`; `hdf5_write`: "the file does not
shrink, use h5repack for that" → "the file does not shrink; copy the channels into a new file to reclaim the space";
add "- One writer per file at a time (no file locking)"; `hdf5_read_channel`: "Throws std::invalid_argument if the
file is malformed or not a version-3 channel file (version-2 files and files modified with other HDF5 tools are
rejected)".

Wrapper MD blocks (MEX `api_mex/hdf5_*.cpp`, Python `api_python/qpy_hdf5_*.cpp`):
- `hdf5_write_channel` (both, l. 20 / 21): "The file does not shrink; use `h5repack` to reclaim the space of the
  removed data" → "The file does not shrink; copy the channels into a new file to reclaim the space of the removed
  data". Add "- One writer per file at a time; files are not locked."
- `hdf5_write_dset` (both): add "- One writer per file at a time; files are not locked."
- `hdf5_read_layout` (MEX l. 18 "errors if the file exists but is not a valid HDF5 file"; Python l. 19 "Raises an
  error if the file exists but is not a valid HDF5 file") and `hdf5_reshape_layout` (MEX l. 20, Python l. 19 "Errors
  if the file does not exist or is not a valid HDF5 file"): "not a valid HDF5 file" → "not a quadriga-lib channel
  file of version 3 (files written by quadriga-lib 0.12.4 and earlier and files modified with other HDF5 tools are
  rejected)".
- `hdf5_read_channel` (both): add "- Errors if the file is not a quadriga-lib channel file of version 3: files
  written by quadriga-lib 0.12.4 and earlier and files modified with other HDF5 tools are rejected."
- `hdf5_create_file` (both): add "- Files can be read by any HDF5 tool but must be modified only with quadriga-lib."
Regenerate the MEX `.m` help and the HTML/Markdown docs (section 9).

---------------------------------------------------------------------------------------------------------------------

## 11. Benchmark protocol (plan §7, D15)

- Program: `tests/benchmarks_cpp/benchmark_hdf5.cpp`, unchanged except its compile comment (8.1) (public API only: `hdf5_create`, `hdf5_write<float>`,
  `hdf5_read_channel<float>`; page cache of the file dropped before reading). Mixes as in the program: a) MIMO 3200
  channels × 10 snapshots × 16×4 × 200 paths (≈ 5 GB in v2); b) ray-tracing route 7000 channels × 100 snapshots ×
  SISO × 20 paths with all path fields (≈ 5 GB in v2; reduced from the plan's 20 000 channels to keep the v2 file at
  5 GB).
- Builds: g++-12, `-std=gnu++17 -O3 -fopenmp`, AVX2 on, static library, no BLAS/LAPACK — the same configure options
  for v2 (`26da9f93`, with the system libhdf5 1.14.6 shared) and v3 (final branch, no `HDF5_STATIC` option). Benchmark
  binary: `g++-12 -std=c++17 -O3 -fopenmp -Wall -Wextra -Wpedantic -I<include> benchmark_hdf5.cpp
  <build>/libquadriga.a` (v2 additionally `-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5`).
- Run: `flock /tmp/bench.lock`, 3 runs alternating mix a and b, `OMP_NUM_THREADS=1`, `cat /proc/loadavg` before and
  after each run; files on disk in `/home/stephan/hdf5_bench/` (not tmpfs), deleted after each run. Load > 4 →
  provisional, repeat later.
- Report per mix and version: medians of write wall/CPU time, write + fsync, read wall/CPU time, file size, the
  `bad=` count (must be 0). Compare times for the same work (mix b's v3 file is smaller, so MB/s is not comparable
  across versions for b). v2 baseline (2026-10-28, load 2.2–2.5): a: write 5.52 s, write + fsync 6.85 s, read
  8.44 s, 4995.1 MB; b: write 127.63 s, write + fsync 128.71 s, read 158.13 s, 5012.1 MB.
- Acceptance: v3 not slower than v2 for either mix (otherwise one optimization round, then escalate, plan §8).
- Additional P4 check (not part of the D15 table): a micro-benchmark of 10^5 minimal channels written and overwritten
  once, to confirm that one minimal-channel write (open to close, including the 64 KiB header-window loads and the
  root bookkeeping of 3.4) stays below ≈ 100 µs (measured I/O pattern alone: ≈ 41 µs). P4 tunes the constants of
  Appendix A with it (the structure of 3.3–3.4 stays; changed values are entered in Appendix A).

---------------------------------------------------------------------------------------------------------------------

## Appendix A: constants

The values are derived from measured lookup3 throughput and the size estimates of 3.4, not from an end-to-end
measurement; P4 tunes them with the 10^5-write micro-benchmark of section 11.

| Constant | Value | Reason |
|----------|-------|--------|
| max chunk body | 65535 | free space must fit a 16-bit NIL size |
| compact limit | 4096 bytes of raw data | small datasets in the checksummed header (one read, 14 bytes smaller layout message); large data unchecksummed and read directly into Armadillo memory; every value ≤ 65531 reads in libhdf5 |
| root chunk 0 directory | 16 entries (320 bytes) | 15 link chunks cover ≈ 7300 channel links before the first directory chunk; small fixed root header |
| directory chunk body | 16380 (819 entries) | one 16 KB chunk covers ≈ 480 000 links (818 link chunks of 16 KB) |
| link chunk body | `min(1024 · 2^k, 16384)` | small files stay small; rewrite + checksum per write ≤ 32 KB |
| chain growth chunk body | `max(L + 20, min(16384, 2 · previous body))` | `Data` groups grow geometrically |
| reserves | 19 (channel group without `Data`), 20 (`Data` group) | 3.2 |
| header read window | 64 KiB, forward, chunk-0 header reads only; all reads of known length direct | 3.6 |
| output buffer | 1 MiB; larger contiguous data written directly | 3.2 |
| zero-fill block | 1 MiB | 3.1 |
| dataset name limit | 65000 bytes | link message ≤ 65535 bytes |
