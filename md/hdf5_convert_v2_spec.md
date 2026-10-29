# Converter for channel files of version 2: `tools/hdf5_convert_v2.py` (specification)

Branch `hdf5-native`. Since file format version 3 (`md/hdf5_v3_spec.md`), quadriga-lib rejects channel files of
version 2, written by quadriga-lib 0.12.4 and earlier (plan decisions D2, D4). This tool converts such a file into a
version-3 file. It reads the version-2 file with **h5py** and writes the version-3 file with the **quadriga_lib
Python API** of the installed version. The C++ library API stays unchanged; nothing in quadriga-lib reads version 2
any more. Two small library changes come with the tool: an optional `prefix` argument for three Python wrappers (5.1)
and a pointer to the tool in the library's rejection message (5.2).

Contents: 1 Usage · 2 What is read and how it maps · 3 Algorithm · 4 Tests · 5 Library and documentation changes ·
6 Size and rules · Appendix A v2 writer history.

------------------------------------------------------------------------------------------------------------------

## 1. Purpose, usage, requirements, exit codes

```
python tools/hdf5_convert_v2.py [--force] input.hdf5 output.hdf5
```

- `input`: a channel file of version 2. It is opened read-only and never changed.
- `output`: the version-3 file to create, with the same storage layout. An existing `output` is an error unless
  `--force` is given; then it is replaced, but only after the conversion succeeded.
- No other options. Standard library `argparse` (`-h` prints the usage and the docstring summary).
- Requirements: Python ≥ 3.10, numpy, h5py ≥ 3.0, and a quadriga_lib Python module that writes channel files of
  version 3 (any version newer than 0.12.4). There is no version-number check: the script checks the `Version` that
  the module actually writes (#14). Linux, Windows (the Windows wheels of quadriga-lib and h5py) and macOS alike.
- Module search: a normal `import quadriga_lib` (the installed module, or the one that `PYTHONPATH` selects); only if
  that fails, `<script folder>/../lib` (a `make python` build in a source checkout) is inserted at the front of
  `sys.path` and the import is repeated. An older installed module is caught by #14, whose message names the
  `PYTHONPATH` remedy.
- MATLAB/Octave users run the same script with Python (e.g. `system('python tools/hdf5_convert_v2.py a.hdf5
  b.hdf5')`); there is no MATLAB converter.
- Slot indices in messages are 0-based (MATLAB: add 1).

Messages go to stderr in the argparse style `hdf5_convert_v2.py: error: <text>` / `hdf5_convert_v2.py: warning:
<text>`. Checks in this order (the first failing one ends the run):

| # | Situation | Exit | Message (`<in>`, `<out>` as given on the command line) |
|---|-----------|------|---------|
| 1 | wrong arguments | 2 | argparse usage error |
| 2 | input missing | 1 | `input file '<in>' does not exist` |
| 3 | h5py not importable | 1 | `h5py is needed to read version-2 files: pip install h5py (or conda install h5py)` |
| 4 | input not HDF5 (`h5py.is_hdf5` false, also for a directory) | 1 | `'<in>' is not an HDF5 file` |
| 5 | h5py cannot open the input (locked by a writer, no permission) | 1 | `cannot open '<in>': <h5py message>` |
| 6 | root dataset `Version`, `ChannelDims` or `Order` missing | 1 | `'<in>' is not a quadriga-lib channel file (dataset '<name>' is missing)` |
| 7 | `Version` = 3 | 1 | `'<in>' is already a channel file of version 3; nothing to convert` |
| 8 | `Version` ≠ 2 | 1 | `'<in>' is a channel file of version <n>; only version 2 can be converted` |
| 9 | product of `ChannelDims` ≠ length of `Order` | 1 | `'<in>': the storage index is corrupted (ChannelDims and Order do not match)` |
| 10 | `<out>` is a directory | 1 | `output '<out>' is a directory` |
| 11 | input and output are the same file (`os.path.samefile` when the output exists) | 1 | `input and output are the same file; choose another output name` |
| 12 | output exists, no `--force` | 1 | `output file '<out>' exists; use --force to replace it` |
| 13 | quadriga_lib not importable | 1 | `the quadriga_lib Python module (a version newer than 0.12.4) is needed: pip install quadriga-lib, or add the folder with the built module (quadriga-lib/lib) to PYTHONPATH` |
| 14 | the module writes a version other than 3 (checked on the new temporary output) | 1 | `quadriga_lib at <module path> writes channel files of version <n>; converting needs a quadriga-lib version newer than 0.12.4 (one that writes version 3); a newer module can be selected with PYTHONPATH=<folder with the module>` |
| 15 | the temporary folder or file cannot be created | 1 | `cannot create a temporary file next to '<out>': <message>` |
| 16 | a slot cannot be read or written (malformed v2 data, a channel the library rejects) | 1 | `slot (ix, iy, iz, iw), group 'channel_<id>': <exception text>; nothing was written` |
| 17 | any other exception | 1 | `<exception type>: <text>; nothing was written` (no traceback) |
| 18 | interrupt (Ctrl-C; SIGTERM and SIGHUP where the platform has them, turned into the same interrupt by a signal handler) | 130 | `interrupted; nothing was written` |
| 19 | conversion complete, every item converted | 0 | stdout: `Converted <k> channels from version 2 to version 3: '<out>'` |
| 20 | conversion complete, some items could not be stored in version 3 (2.6) | 3 | one warning per item during the run (`slot (ix, iy, iz, iw): <what>`), then the line of #19 plus `; <m> items were not converted (see the warnings)` |

Exits 1 and 130 leave no output and no temporary files behind (3.2). Exit 3 means: the output was written and is
complete except for the items named in the warnings.

------------------------------------------------------------------------------------------------------------------

## 2. What is read and how it maps

### 2.1 Correctness criterion

For every slot, `hdf5_read_channel` of the installed quadriga-lib on the output returns what the 0.12.4 reader
(`hdf5_read_channel`, C++ and Python) returned on the input: the same fields, shapes, element bytes and unstructured
data, with the same element types, in the same order. `hdf5_read_dset` and `hdf5_read_dset_names` with any prefix
return the same datasets in the same order. `hdf5_read_layout` returns the same layout and the same occupied slots
(except the empty-channel slots of 2.6); the channel IDs are renumbered (2.2).

Deviations, and only these:
1. the items of 2.6 (warned, exit 3);
2. trailing `delay` padding without coefficients `(1, 1, 0)` instead of `(0, 0, 0)` (2.4);
3. delays of older writers in the shape `(1, n_tx, np)` or `(n_rx, 1, np)` are expanded to `(n_rx, n_tx, np)` (2.4);
4. a channel group without both orientation attributes and datasets (edits only; library files always have the
   default attributes) reads back with `(3, 1)` zero orientations, where 0.12.4 returned none.

The converter re-implements the reading rules of the 0.12.4 reader (`hdf5_read_channel` in `src/qd_channel_hdf5.cpp`
of 26da9f93), not the writers: so it reads every layout that 0.12.4 read, including files of older writers
(Appendix A) and files changed with other HDF5 tools (h5py reads every HDF5 storage layout). It does not repeat the
0.12.4 extent checks against `NumTx`, `NumRx`, `NumPath` (they guarded C++ allocations); shapes come from the
datasets and the library validates each channel when it writes it (`channel::is_valid`). A slot that 0.12.4
rejected as malformed is therefore converted if its data form a valid channel, and fails with #16 otherwise.

Notation: the Python API uses the Armadillo shape of each object (`(n_rx, n_tx, n_path)` etc.), HDF5 (and h5py)
stores the reversed extents with the same memory order. So every array read with h5py is transposed with `.T`
(all axes reversed) to get the dict value; 1-D arrays and scalars are unchanged.

### 2.2 Root group

| Object | Use |
|--------|-----|
| `Version` (scalar, any integer type) | must be 2 (#6–#8) |
| `ChannelDims` (`[4]`) | `nx, ny, nz, nw`; the output is created with `hdf5_create_file(tmp, nx, ny, nz, nw)` |
| `Order` (`[n_slot]`) | channel ID per slot, 0 = empty; read at once (`f['Order'][()]`, as 0.12.4 did on every call) |
| `channel_<id>` | read for every slot with `Order[slot] = id ≠ 0`; groups not referenced by `Order` (none in library files) are ignored; an ID used by several slots is read for each of them (as 0.12.4) |

Slot `s` ↔ `(ix, iy, iz, iw)`: `ix = s % nx`, `iy = (s // nx) % ny`, `iz = (s // (nx·ny)) % nz`,
`iw = s // (nx·ny·nz)`. Slots are converted in increasing `s`; the version-3 writer gives new IDs 1, 2, … in that
order, so the IDs differ from the input (they have no meaning outside the file; `hdf5_read_layout` of the C++ API
shows them).

### 2.3 Channel group `channel_<id>` → channel dict (keys of `hdf5_write_channel`)

| v2 object (attribute "A", dataset "D") | dict entry |
|---|---|
| A `Name` (fixed-length string ≤ 256 bytes) | `name`: the bytes before the first NUL, passed as `bytes` (pybind11 stores them unchanged, so a name in any encoding is kept byte-exact); `b''` if missing. h5py's default read already stops at the first NUL (older writers left undefined bytes after it, Appendix A). Always set: without it the library uses the default name `"empty"`. More than 255 bytes (no NUL in 256 bytes): see 2.6 |
| A `NumSnap` (required) | `n_snap`; not passed (the library derives it from the fields) |
| A `Initial_position` (optional, default 0) | `initial_position` (Python int); always set |
| — `n_snap = 0`: | stop here; every other object of the group except `Data` is ignored (writers before June 2024 left empty `CenterFrequency` and position datasets there) |
| A `NumTx`, `NumRx` (required) | `n_tx`, `n_rx` for the coefficient padding and the delay expansion (2.4) |
| D `CenterFrequency` | `center_frequency` = the dataset (float32, `(n_snap,)`) |
| else A `CenterFrequency` | NaN → key omitted (reads back empty); otherwise a float32 array `(1,)` |
| D `tx_position`, `rx_position` (`{n_snap, 3}`) | same key, `.T` → `(3, n_snap)` |
| else A `tx_position`, `rx_position` (`[3]`, required) | same key, `(3, 1)` (NaN default kept as NaN, as 0.12.4) |
| D / A `tx_orientation`, `rx_orientation` | as positions; both missing → key omitted (deviation 4 of 2.1) |
| snapshot data | 2.4 |
| G `Data` | 2.5 |

Structured values stay float32 and no arithmetic is applied to them: float32 → double (pybind11) → float32 (writer)
is exact, so the output holds the input's bits (quiet-NaN payloads, ±0 and ±Inf included; a signalling NaN, which
only bit manipulation creates, comes back quiet, as from 0.12.4's Python reader, which reads as double). A numeric type other than
float32 (only after edits with other tools) is converted with `astype(np.float32)` when it is read, which is what
0.12.4 `hdf5_read_channel<float>` did; `no_interact` of another integer type is converted with `astype(np.uint32)`
(0.12.4 read it as `H5T_NATIVE_UINT`).

### 2.4 Snapshot data

Snapshot groups: `n_snap = 1`: the channel group itself (index 0). `n_snap > 1`: the links of the channel group named
`Snap_<i>` with `i` in canonical decimal (regex `Snap_(0|[1-9][0-9]{0,9})`) and `i < n_snap`, sorted by `i` (as
0.12.4). If some exist and `n_snap − count > 2^20`: error #16 with 0.12.4's text `'NumSnap' is much larger than the
number of snapshots in the file` (padding would only allocate memory). `NumPath` is not read.

Per snapshot group `s` with index `i`, every present dataset is appended to its list, after padding the list to
length `i` (missing snapshots of older writers, Appendix A); after the last group, every non-empty list is padded to
`n_snap`. A list that no snapshot has stays absent (key omitted). This is exactly 0.12.4's `j_<field>` counter logic.

| v2 dataset in `s` | dict key | entry | padding entry (leading / trailing) |
|---|---|---|---|
| `coeff_re`, `coeff_im` (if `coeff_re` exists; `coeff_im` then required) | `coeff_re`, `coeff_im` (the real form of `hdf5_write_channel`) | `.T` → `(n_rx, n_tx, np)` | `(n_rx, n_tx, 0)` from the attributes |
| `delay` | `delay` | `.T` → `(d0, d1, np)` (a rank other than 3 → #16 with 0.12.4's text `dataset 'delay' has <n> dimensions, expected 3`); if the snapshot has `coeff_re` and `(d0, d1)` is neither `(n_rx, n_tx)` nor `(1, 1)`: `np.broadcast_to(d, (n_rx, n_tx, np)).copy()` (deviation 3) | leading: `(d0, d1, 0)` of the (expanded) dataset; trailing: `(n_rx, n_tx, 0)` if the channel has `coeff_re`, else `(1, 1, 0)` (deviation 2) |
| `path_gain`, `path_length` | same | `(np,)` | `(0,)` |
| `path_polarization` (`{np, 8}`, interleaved re/im pairs) | `path_polarization` | `np.ascontiguousarray(d[()], dtype=np.float32).view(np.complex64).T` → `(4, np)`; a view, no arithmetic (`p[0::2] + 1j·p[1::2]` would turn ±Inf/NaN parts into NaN and −0 into +0); the library interleaves it back | complex64 `(4, 0)` |
| `path_angles` (`{4, np}`) | `path_angles` | `.T` → `(np, 4)` | `(0, 4)` |
| `path_fbs_pos`, `path_lbs_pos` | `fbs_pos`, `lbs_pos` | `.T` → `(3, np)` | `(3, 0)` |
| `no_interact` | `no_interact` | `(np,)` uint32 | `(0,)` uint32 |
| — with it: `interact_coord` | `interact_coord` | `.T` → `(3, sum)` if `sum(no_interact) > 0` (dataset required then); otherwise `(3, 0)` whether the dataset exists or not | `(3, 0)` |

Deviation 2 (trailing `delay` padding): 0.12.4 padded trailing snapshots with `(n_rx, n_tx, 0)` also for channels
without coefficients, where `n_rx = n_tx = 0`. That `(0, 0, 0)` entry makes the channel invalid (`Size mismatch in
'delay[i]'`), so 0.12.4's own Python and MATLAB readers rejected these channels (only files of writers before 0.12.4
with an empty last snapshot). With coefficients 0.12.4's `(n_rx, n_tx, 0)` is kept.

Deviation 3 (delay expansion): writers from 2023-11-13 to 2024-07-03 accepted delays with rows in {1, n_rx} and
columns in {1, n_tx}; the current validation accepts only `(n_rx, n_tx, np)` and `(1, 1, np)` and would reject the
channel. A singleton dimension means "the same for all antennas", so the expansion does not change any value.

Mixed delay shapes (`(n_rx, n_tx, np)` in one snapshot, `(1, 1, np)` in another), empty snapshots (extent-0
datasets of 0.12.4), LOS-only snapshots (`no_interact` all 0, `interact_coord` `{0, 3}` or absent) and
single-snapshot channels need no special code: they follow from the table.

### 2.5 Unstructured data `channel_<id>/Data`

Link order: creation order (`g.id.links.iterate(..., idx_type=h5py.h5.INDEX_CRT_ORDER)`); if the group does not
track it (only groups made by other tools; h5py raises `RuntimeError`), name order (`INDEX_NAME`), as 0.12.4. Links
that are not datasets are ignored (0.12.4 ignored them too). Every dataset is converted, with any prefix, keeping the
link order:
- the leading run of datasets whose names start with `par_` goes into the `par` dict of `hdf5_write_channel`, under
  the names without the prefix (Python dicts keep insertion order, the library writes `par` in that order);
- from the first dataset without the `par_` prefix on (written by MEX or C++ `hdf5_write_dset` with a `prefix`), it
  and every later dataset are written one by one with `hdf5_write_dset(tmp, ix, iy, iz, iw, name=<full link name>,
  data=<value>, prefix='')` (5.1), which appends in creation order.

So the creation order of all datasets is kept, and library files without custom prefixes need one call per slot.

Value of a dataset (rules of 0.12.4 `qHDF_read_data`; classified by the HDF5 type of `d.id.get_type()`, not by the
numpy dtype, because h5py reads an HDF5 enum as an integer array):

| HDF5 type | Rank (scalar = 0) | Value passed |
|---|---|---|
| `TypeFloatID` or `TypeIntegerID` (not `TypeEnumID`), size 4 or 8, any byte order | 0 | numpy scalar of the native-order type (`np.float32`, `np.float64`, `np.uint64`, `np.int64`, `np.uint32`, `np.int32`) → stored as a scalar of that type (never `.item()`, which would turn float32 into float64 and uint32 into int64) |
| same | 1 | 1-D array (stored as a vector; also with one element) |
| same | 2, 3 | `.T` (Armadillo shape, also `(n, 1)`, `(1, 1)` and extents 0) |
| `TypeStringID`, fixed length, one element (rank 0–3) | — | the raw bytes (read with the file type as memory type, `d.id.read(..., mtype=d.id.get_type())`, because h5py's default read stops at the first NUL) with trailing NULs removed, inner NULs kept, decoded as UTF-8 → `str` (an empty string, stored as one NUL, gives `''`) |
| `TypeStringID`, variable length, one element (h5py/MATLAB edits) | — | the bytes before the first NUL (0.12.4 read it as a C string), decoded as UTF-8 → `str` |
| anything else (other types and sizes, enums, compound, string arrays, rank > 3, null dataspace) | | not converted (2.6) |

### 2.6 Items that version 3 cannot store (warning, exit 3)

All arise only from the C++ API with `assume_valid = true` or from edits with other tools; files written through the
Python or MATLAB functions never contain them.

| Item | Reason | Result |
|---|---|---|
| Slot with `n_snap = 0`, `initial_position = 0`, no convertible `Data`, empty name (C++ `assume_valid` with an empty channel) | the library writes nothing for an empty channel | slot left empty; reads back as before (empty channel), only the occupancy differs. Warning `empty channel; the slot is left empty` |
| Same with a non-empty name, with or without `Data` datasets | a name alone is not channel data; the library stores it only with structured data | name dropped (datasets, if any, converted). Warning `channel name '<name>' is dropped: version 3 stores a name only together with channel data` |
| Name of 256 bytes without NUL (writers before 0.12.4 with `assume_valid`) | at most 255 bytes | if the 256 bytes are valid UTF-8: cut at the last character boundary ≤ 255 bytes (`name[:255].decode('utf-8', 'ignore').encode()`), so that the name stays readable as `str`; otherwise the first 255 bytes. Warning `channel name truncated to <k> bytes` |
| Dataset whose name has characters other than letters, digits and `_` (edits) | rejected by the writers | not converted, warning |
| Dataset of an unsupported type or shape (table 2.5, last row) or a string that is not valid UTF-8 (C++ only) | not storable / no Python `str` | not converted, warning naming the reason |

A channel that the library rejects for any other reason (e.g. `assume_valid` data with `Delays are missing or
incomplete.`) is not an item: the run fails with #16 (a channel is not converted partially).

------------------------------------------------------------------------------------------------------------------

## 3. Algorithm

```
parse arguments; checks #2–#12 (h5py opens the input read-only once and keeps it open)
import quadriga_lib (#13)
tmpdir = tempfile.mkdtemp(dir=<folder of out>, prefix='.hdf5_convert_v2-')        # #15
tmp = tmpdir/<basename of out>
try:
    hdf5_create_file(tmp, nx, ny, nz, nw)
    open tmp with h5py read-only, read Version, close; ≠ 3 → #14
    for each slot s with Order[s] = id ≠ 0, in increasing s:
        chan, par, extra, warnings = read_slot(input["channel_<id>"])    # 2.3–2.6; exceptions → #16
        print the warnings (slot index)
        if chan is not empty (n_snap > 0 or initial_position ≠ 0) or par:
            hdf5_write_channel(tmp, chan, par or None, ix, iy, iz, iw)     # exceptions → #16
        for name, value in extra:                                        # datasets after the first non-par_ one
            hdf5_write_dset(tmp, ix, iy, iz, iw, name, value, prefix='') # exceptions → #16
        if nothing was written: 2.6 warning (empty channel)
    close the input
    os.replace(tmp, out)                                                 # atomic; replaces out with --force
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)                            # only this run's own folder
print the summary (#19/#20); return 0 or 3
```

3.1 Memory: `Order` (4 bytes per slot, as 0.12.4 read it) plus one channel at a time.

3.2 Atomicity and safety: the output is written into a fresh folder created by this run next to the output (same
file system, so `os.replace` is atomic) and appears only complete. Nothing that existed before is ever deleted:
not a file named like the temporary one, not the input (checks #10–#12 run before), and two concurrent runs with the
same output do not touch each other's files. On every error, exception (#16, #17) and interrupt (#18, also SIGTERM and SIGHUP) the
`finally` block removes the folder; an existing output stays untouched until the replace.

3.3 Writes: `hdf5_write_channel` (one call per slot) writes the structured data and the leading `par_` datasets; a
slot with only unstructured data is written with the channel dict `{name, initial_position}` (an empty channel) plus
`par`, which the library stores like 0.12.4 (group with default attributes); a slot whose first dataset has a custom
prefix is created by its first `hdf5_write_dset`, which also gives the group the default attributes.

3.4 No progress output: a slot write takes about 0.1 ms of CPU time (20 000 `hdf5_write_channel` calls into a
65 536-slot file: 2.1 s, measured under load, provisional); the runtime is set by the amount of data. Output:
warnings and the summary only.

3.5 Files changed with other HDF5 tools are converted as far as 0.12.4 read them (2.1, 2.5); h5py handles dense link
storage, chunked and compressed datasets, other byte orders. Only the 2.6 items are lost, each with a warning.

3.6 The output is only read (never written) with h5py, so it stays inside the version-3 subset (D4).

------------------------------------------------------------------------------------------------------------------

## 4. Tests: `tests/python_tests/test_hdf5_convert_v2.py`

unittest style (`md/SPEC_PYTHON_API.md` layout, one `test_case` class), the whole class skipped with
`@unittest.skipIf(h5py is None, "h5py is not installed (needed to read version-2 files)")`. The script is run as a
subprocess (`sys.executable tools/hdf5_convert_v2.py ...`, path relative to the test file), so the real CLI, exit
codes and messages are tested. Files go to a `tempfile.TemporaryDirectory`.

Module selection: the test imports `quadriga_lib` as the other test files do (`../../lib` appended to `sys.path`),
and runs the script with the folder of the module it imported prepended to `PYTHONPATH` in the subprocess
environment, so the test and the script always use the same module: the installed one, the one `PYTHONPATH`
selects (`make test` and `make python_test` run pytest with `PYTHONPATH=<checkout>/lib`, so they test the build
of `make python`), or `lib`. `setUpClass` checks that the module writes `Version` 3 (create a small file, read it with
h5py) and fails with the module path in the message otherwise (no skip: a wrong module is a setup error).

### 4.1 Fixtures (`tests/data/`, committed, about 200–250 KB in total, at most 300 KB)

Written once by `tests/data/make_hdf5_v2_fixtures.py` (committed, about 150 lines). Its header gives purpose and
recipe: build the Python module of quadriga-lib 0.12.4 from a checkout of commit 26da9f93 (release 0.12.4)
(`cmake -S <src> -B <build> -DENABLE_PYTHON=ON -DPython3_EXECUTABLE=<python> -DCMAKE_CXX_COMPILER=g++-12
-DENABLE_MATLAB=OFF -DENABLE_OCTAVE=OFF -DENABLE_TESTS=OFF`, `cmake --build <build> --target quadriga_py`), then run
`PYTHONPATH=<folder with the built module> python tests/data/make_hdf5_v2_fixtures.py` in an environment with h5py
(the 0.12.4 module and h5py work in one process). It does not check `quadriga_lib.version()` (builds report e.g.
`0.12.4_AVX2`, and later versions also report 0.12.4 until the release); it checks that every file it writes has
`Version` 2 and stops otherwise. Random values from `np.random.default_rng(<fixed seed>)`. Small layouts only (the
default layout would add 256 KB of `Order` per file). The oracle is reproducible; the HDF5 files differ in their
timestamps on each run.

1. `hdf5_v2_written.hdf5` — only 0.12.4 functions. Created `(4, 2, 2, 1)`, reshaped to `(2, 4, 2, 1)`; slots:
   - multi-snapshot channel, all keys, 3 snapshots with paths `(3, 0, 2)` (empty middle snapshot), `delay`
     `(n_rx, n_tx, np)` in snapshot 0 and `(1, 1, np)` in the others (mixed shapes), `no_interact` with 0 and > 0
     entries, per-snapshot `tx_position`, `rx_orientation`, `center_frequency`, shared `rx_position`,
     `tx_orientation`, `initial_position = 2`, UTF-8 name, float64 values that are not float32-exact (the file holds
     the rounded value); `path_polarization` and `coeff` with −0.0, NaN and ±Inf in real and imaginary parts;
   - single-snapshot channel from float32 arrays, LOS only (`interact_coord` `(3, 0)`), no orientations (read back
     as zeros), no `center_frequency` (read back absent), name of 255 bytes;
   - single-snapshot channel with 0 paths; a channel with positions only; a channel with only `initial_position`;
   - `par` data on the first channel: str, `''`, str with an inner NUL, scalars of all six numeric types, 1-D
     (also length 1), 2-D (also `(n, 1)` and `(0, 3)`), 3-D;
   - a par-only slot (`hdf5_write_channel({}, par)`), a slot made by `hdf5_write_dset` on an empty slot (two
     datasets), a filled slot with an added `hdf5_write_dset`, an overwritten slot (written twice: dead space),
     empty slots.
2. `hdf5_v2_edited.hdf5` — 0.12.4 output changed with h5py into layouts of older writers and of other tools (all
   read by 0.12.4), every case convertible without loss:
   - a 4-snapshot channel with coefficients and `(1, 1, np)` delays whose empty `Snap_1` and `Snap_3` groups are
     deleted;
   - a channel without coefficients (`path_gain` and `(1, 1, np)` `delay`) whose empty last `Snap_` group is deleted
     (deviation 2);
   - a channel with `n_rx = 2`, `n_tx = 3` whose `delay` datasets are rewritten with the old shapes `{np, 3, 1}` and
     `{np, 1, 2}` (deviation 3);
   - a LOS-only snapshot without `interact_coord`; a single-snapshot channel with 0 paths whose datasets are
     deleted; a `Name` with bytes after the NUL (raw attribute write); a par-only group with `Initial_position = 5`;
   - a `Data` group without creation-order tracking (rebuilt by h5py with `track_order=False`, datasets created in
     non-alphabetical order);
   - `par_` datasets added by h5py: a variable-length string, a big-endian int32 array, a chunked gzip-compressed
     float64 matrix;
   - custom prefixes as written by MEX `hdf5_write_dset(..., prefix)`: in one `Data` group the order `par_a`,
     `x_b`, `par_c`, `y_d` (added by h5py; the group tracks creation order), and a slot whose only dataset is `x_e`.
3. `hdf5_v2_lossy.hdf5` — the items of 2.6, made with h5py from 0.12.4 output, one structured channel only: an empty
   occupied slot (par-only group with `Data` deleted), a name-only slot (same plus `Name` set), a name-only slot with
   `par` data, a 256-byte UTF-8 name without NUL whose 255th and 256th bytes belong to one 2-byte character,
   `par_bad-name`, an int16 dataset, a rank-4 dataset, an enum dataset, and — alone in its own slot, because 0.12.4's
   Python reader fails for the whole slot on it — a fixed-length string with invalid UTF-8.
4. `hdf5_v2_expected.npz` — the oracle: for each file, what 0.12.4 returns, saved with `np.savez_compressed`
   (loaded with `allow_pickle=False`; no pickle, so any numpy version loads it). Keys: `<file>/layout` (4 values of
   `hdf5_read_layout`), `<file>/has_data` (its occupancy mask), and per occupied slot `s` (linear index) the output of
   0.12.4 `hdf5_read_channel(fn, ix, iy, iz, iw)`: `<file>/<s>/chan/<key>` for array and scalar entries,
   `<file>/<s>/chan/<key>/<i>` for list entries, `<file>/<s>/par/<name>` for `par` (every value stored as
   `np.asarray(value)`, which keeps dtype and shape), `<file>/<s>/chan_keys` and `<file>/<s>/par_keys` (the key
   order, as str arrays); `<file>/<s>/error` with the exception text if 0.12.4's Python reader raised for the slot
   (any failure: e.g. its own validation for the deviation-2 channel, `UnicodeDecodeError` for the invalid-UTF-8
   string). 0.12.4's Python reader has no `prefix` argument; custom-prefix datasets are checked with h5py (4.3).

### 4.2 Cases

| Test | Checks |
|---|---|
| `test_written` | convert file 1: exit 0, no stderr, summary line; layout and `has_data` of the output equal the oracle; every slot equal (4.3) |
| `test_edited` | file 2: exit 0; equal to the oracle and the h5py check; the slot with `error` in the oracle (deviation 2): the channel reads back, `delay` has `n_snap` entries, the trailing one `(1, 1, 0)`, all other fields as stored; the deviation-3 slot: `delay` is `(2, 3, np)`, each entry equal to the stored value of its row/column; `hdf5_read_dset_names(out, ..., prefix='')` lists `par_a, x_b, par_c, y_d` in that order |
| `test_lossy` | file 3: exit 3; one warning per 2.6 item with its slot and name (substring checks); the structured channel and the convertible parts equal the oracle; the empty and the name-only slots are empty in the output; the 256-byte name reads back as its first 254 bytes; the skipped datasets are absent; the invalid-UTF-8 slot (oracle `error`) holds every other dataset of the slot |
| `test_output_exists` | output present: exit 1, `--force` in the message, file unchanged; with `--force`: exit 0, replaced |
| `test_rejected_inputs` | missing input; a text file (`not an HDF5 file`); an h5py file without `Version`; a v2 fixture copy with `Version` set to 7 by h5py; a file written by the installed quadriga_lib (`already a channel file of version 3`); a v2 copy with `Order` one element short (#9); input = output; output is a directory; output in a missing folder (#15); a v2 copy with a string `Version` (#17, `ValueError: …; nothing was written`). Each: exit 1, the message as the start of the only stderr line, no output created; and a missing argument: exit 2 |
| `test_failure_leaves_nothing` | a copy of file 1 where h5py deleted `coeff_im` of a slot after the first occupied one: exit 1, message names the slot, no output, and the output folder holds no other new entry (no temporary folder left) |
| `test_missing_modules` | the script run with `sys.modules['h5py'] = None` (and then `['quadriga_lib'] = None`) set before `runpy.run_path`: exit 1 and the install hint; a stub `quadriga_lib` whose `hdf5_create_file` writes `Version` 2 (#14): exit 1, the message with the stub path, nothing left |
| `test_interrupted` | the script run with `runpy` after replacing `hdf5_write_channel` by a function that raises `KeyboardInterrupt`, and (POSIX) one that sends SIGTERM and SIGHUP to its own process (#18): exit 130, the message, no output and no temporary folder |
| `test_input_unchanged` | the input files' bytes (hash) are unchanged after the conversions |

### 4.3 Comparison

Per slot, against the oracle: the dicts from `hdf5_read_channel(out, ix, iy, iz, iw)` (chan[0] and par) have exactly
the oracle's keys in the oracle's order (`list(d) == chan_keys / par_keys`); lists have the oracle's length; every
value: `np.asarray(actual)` has the oracle's dtype and shape and the same bytes (`a.tobytes() == b.tobytes()`; this
also pins −0.0 and NaN payloads); str by `==`.

h5py check, independent of the converter, for every occupied slot of the input that is not a 2.6 slot: the `Data`
links of input and output have the same names in the same order (minus the 2.6 datasets); each output dataset has
the input dataset's native dtype (strings: the stored bytes without trailing NULs) and shape and the same bytes; the
`Initial_position` attributes are equal (the Python reader omits it for `n_snap = 0`).

### 4.4 Manual check (not committed)

Convert a version-2 file of about 1 GB (e.g. made by `tests/benchmarks_cpp/benchmark_hdf5.cpp` with the 0.12.4
build) and report the runtime and peak memory under `flock /tmp/bench.lock` with the load average, so that the docs
can name the expected speed honestly.

------------------------------------------------------------------------------------------------------------------

## 5. Library and documentation changes

### 5.1 Optional `prefix` for three Python wrappers (MEX parity)

The C++ functions already take `prefix` (default `"par_"`); the MEX wrappers pass it on; the Python wrappers always
used the default. Each gets an optional last argument `prefix='par_'`, so existing calls are unchanged:

| File | Change |
|---|---|
| `api_python/qpy_hdf5_write_dset.cpp` | signature `hdf5_write_dset(fn, ix, iy, iz, iw, name, data, prefix)`; pass `prefix` to `quadriga_lib::hdf5_write_dset(fn, name, &any_data, ix, iy, iz, iw, prefix)` |
| `api_python/qpy_hdf5_read_dset.cpp` | `hdf5_read_dset(fn, ix, iy, iz, iw, name, prefix)`; pass it to `quadriga_lib::hdf5_read_dset`; drop the comment about the default prefix |
| `api_python/qpy_hdf5_read_dset_names.cpp` | `hdf5_read_dset_names(fn, ix, iy, iz, iw, prefix)`; pass it to `quadriga_lib::hdf5_read_dset_names` |
| `api_python/python_channel.cpp` | registrations: `py::arg("prefix") = "par_"` after `data` / `name` / `iw`; update the commented declarations at the end of the three wrapper files |

MD blocks (Usage lines get `, prefix` at the end; Inputs get `- **`prefix`** — Prefix prepended to `name` in the HDF5
file; alphanumeric and underscores only; str; default: `'par_'``, in the wording of the MEX blocks):
- `hdf5_write_dset`: "Dataset is stored under `prefix + name`" stays; "`name` must contain only alphanumeric
  characters and underscores" → "`prefix + name` must …".
- `hdf5_read_dset`: "The dataset is looked up under `'par_' + name` — the `par_` prefix is prepended internally" →
  "The dataset is looked up under `prefix + name` (default prefix `par_`)"; `name` input: "Dataset name without the
  prefix".
- `hdf5_read_dset_names`: "Datasets are identified by the `par_` prefix; the returned names have that prefix
  stripped" → "Lists the datasets whose names start with `prefix` (default `par_`; `''` lists all); the returned
  names have the prefix stripped"; output description accordingly.
- `hdf5_read_channel` and `hdf5_write_channel` are unchanged (`par` is always `par_`).

Wrapper tests in `tests/python_tests/test_hdf5_functions.py`, new method `test_dset_prefix`: `hdf5_write_dset` with
`prefix='x_'` (positional and keyword) next to default datasets; `hdf5_read_dset_names` without prefix lists only the
`par_` names, with `prefix='x_'` only the `x_` names (stripped), with `prefix=''` all full names in creation order;
`hdf5_read_dset(name=..., prefix='x_')` returns the value, without the prefix `None`; `prefix='x-'` raises
`ValueError` ("Parameter name must only contain …"); `hdf5_read_channel`'s `par` holds only the `par_` datasets.
The existing pybind11 `TypeError` checks stay.

### 5.2 Pointer in the library's rejection message

`src/qd_hdf5.hpp` (`unsupported()`, about line 120): append " Version-2 files can be converted with
tools/hdf5_convert_v2.py." to the `Unsupported HDF5 file: …` text. The pinned copies follow:
`tests/catch2_tests/test_hdf_robustness_internal.cpp` (about line 56) and `tests/catch2_tests/test_hdf5_interop.cpp`
(about line 514); the message table in `md/hdf5_v3_spec.md` 4.2 too. No Python or MEX test pins the full text.

### 5.3 Documentation

Users meet the problem as the rejection error, then read the `hdf5_read_channel` docs; the tool is named there and
in the READMEs. Exact changes:

1. `src/qd_channel_hdf5.cpp`, MD block `hdf5_read_channel` (the bullet ending "…files modified with other HDF5
   tools are not supported and may be rejected."): append the bullet "- Files of quadriga-lib 0.12.4 and earlier
   can be converted to version 3 with the Python script `tools/hdf5_convert_v2.py` of the quadriga-lib sources (needs
   numpy, h5py and the quadriga_lib Python module): `python tools/hdf5_convert_v2.py old.hdf5 new.hdf5`."
2. `include/quadriga_channel.hpp`, comment of `hdf5_read_channel`: after the version-3 sentence add "Convert
   version-2 files with tools/hdf5_convert_v2.py."
3. `api_mex/hdf5_read_channel.cpp`, after the bullet "Errors if the file is not a quadriga-lib channel file of
   version 3: …": "- Files of quadriga-lib 0.12.4 and earlier can be converted with the Python script
   `tools/hdf5_convert_v2.py` of the quadriga-lib sources (needs Python with numpy, h5py and quadriga-lib), e.g.
   `system('python tools/hdf5_convert_v2.py old.hdf5 new.hdf5')`." (wrapped at 100 characters)
4. `api_python/qpy_hdf5_read_channel.cpp`, after the corresponding bullet: "- Convert files of quadriga-lib 0.12.4
   and earlier with `tools/hdf5_convert_v2.py` of the quadriga-lib sources (needs h5py):
   `python tools/hdf5_convert_v2.py old.hdf5 new.hdf5`."
5. `README.md`, new section after "Installation" (before "Distribution License"):
   ```
   ## Channel files of quadriga-lib 0.12.4 and earlier
   The HDF5 channel files use file format version 3; files of version 2 (written by quadriga-lib 0.12.4 and
   earlier) are rejected. Convert them once with `python tools/hdf5_convert_v2.py old.hdf5 new.hdf5` (needs numpy,
   h5py and the quadriga_lib Python module; also for MATLAB and Octave users).
   ```
6. `README_pip.md` and `README_win.md` (the pip and the Windows packages contain no `tools/`): the same section, with
   the script as a link to `https://github.com/stjaeckel/quadriga-lib/blob/main/tools/hdf5_convert_v2.py` and the
   install line `pip install quadriga-lib h5py` (`README_win.md`: after "Python Setup"; the script runs natively
   with the Windows wheels).
7. Regenerate the docs (`make documentation`) so that `html_docu/` and the MEX `.m` help carry items 1, 3, 4 and the
   5.1 wrapper docs.
8. Not changed: the other MD blocks that mention 0.12.4 (`hdf5_read_layout`, `hdf5_reshape_layout`,
   `hdf5_write_channel`; they refer to or match `hdf5_read_channel`) and the install pages in `tools/download_md/`
   (they describe building, not files).

------------------------------------------------------------------------------------------------------------------

## 6. Size and rules

- One script, standard library + numpy + h5py + quadriga_lib. Target about 300 lines including the license header
  (`#!/usr/bin/env python3` and the SPDX lines of `tools/*.py`) and a module docstring that serves as the user
  documentation (usage, requirements, exit codes, the 2.6 limits in one paragraph). The docstring may cite the
  0.12.4 reader by version, not `md/` documents or tracking ids.
- Functions: `main(argv)`, `read_slot(group)` (2.3/2.4, with a local `put(key, i, value, pad)` helper), `read_par(dset)`
  (2.5), `link_names(group)` (creation/name order). Nothing of the library is re-implemented: layout, slot
  indexing of the output, validation, float32 storage and IDs are done by `hdf5_create_file`,
  `hdf5_write_channel` and `hdf5_write_dset`.
- No progress output, no caching, no parallelism, no `--verify` option (the tests compare the outputs).
- Fixture generator and test follow §9 of the plan: no `md/` references in either file, tests remove their files.

------------------------------------------------------------------------------------------------------------------

## Appendix A. v2 writers (history of `src/qd_channel.cpp` / `src/qd_channel_hdf5.cpp`)

| Writers | Layout | Read by 0.12.4 |
|---|---|---|
| 2023-09-12 (ec506d8) to 2023-11-13 (b411cdf), before the first tag | channel properties as datasets (`NumSnap`, `Name`, …), `Version` from 2023-10-10, `path_coord` until 2023-10-25 | no: `attribute 'NumSnap' is missing`; the converter fails the same way (#16) |
| 2023-11-13 (b411cdf) to 0.12.2 (incl. v0.1.14-alpha, v0.11.x) | attributes as in version 3; snapshots without any data have no `Snap_<i>` group (single snapshot: no datasets, no `NumPath`); within a snapshot, empty fields (e.g. `interact_coord` of LOS-only snapshots) are not written; before 2024-06-17 (252e68c) a channel with `n_snap = 0` got extent-0 `CenterFrequency`/position datasets; before 2024-07-03 (4b17a14) delays of shape `(1, n_tx, np)` and `(n_rx, 1, np)` were accepted and stored (expanded, 2.4); `Name` written from `name.c_str()` with 256 bytes (bytes after the NUL undefined, no NUL for names ≥ 256 bytes, which only `assume_valid` allowed); `Initial_position` written also for channels with only unstructured data | yes, missing snapshots padded (2.4) |
| 0.12.4 (26da9f93) | every snapshot written, empty ones with extent-0 datasets; `Name` NUL-padded, at most 255 bytes; channels with only unstructured data keep all default attributes | yes |
