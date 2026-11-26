+++
title = "HDF5 channel files"
summary = "HDF5 file of channels with antennas: a grid of storage places, one channel with coefficients, delays, path data and metadata in each"
order = 4
+++

An HDF5 channel file stores finished channels, after the antennas are applied: for each link one [channel](cpp_api.html#channel) with its coefficients and delays for every antenna pair and snapshot, optionally the paths without antennas, and any metadata the user adds. The file is organized as a grid of storage places, the **slots**: up to four indices, e.g. Tx, Rx and frequency, address one slot, and each slot holds one channel. A simulation writes each link into its slot and later reads any subset back. Numbers are stored in single precision.

HDF5 channel files and [QRT files](formats.html#qrt) store different stages of a simulation: a QRT file holds propagation paths before antennas are applied, organized by origin and CIR position; an HDF5 channel file holds finished channels from any source (GBSM, layouts, channels made from QRT paths), ready for the frequency response, statistics or a link-level simulation. The table on the [QRT](formats.html#qrt) page compares both.

Related functions:

- C++ [hdf5_create_file](cpp_api.html#hdf5_create_file), [hdf5_write_channel](cpp_api.html#hdf5_write_channel), [hdf5_read_channel](cpp_api.html#hdf5_read_channel), [hdf5_read_layout](cpp_api.html#hdf5_read_layout), [hdf5_reshape_layout](cpp_api.html#hdf5_reshape_layout), [hdf5_write_dset](cpp_api.html#hdf5_write_dset), [hdf5_read_dset](cpp_api.html#hdf5_read_dset), [hdf5_read_dset_names](cpp_api.html#hdf5_read_dset_names)
- MATLAB / Octave [hdf5_create_file](mex_api.html#hdf5_create_file), [hdf5_write_channel](mex_api.html#hdf5_write_channel), [hdf5_read_channel](mex_api.html#hdf5_read_channel), [hdf5_read_layout](mex_api.html#hdf5_read_layout), [hdf5_reshape_layout](mex_api.html#hdf5_reshape_layout), [hdf5_write_dset](mex_api.html#hdf5_write_dset), [hdf5_read_dset](mex_api.html#hdf5_read_dset), [hdf5_read_dset_names](mex_api.html#hdf5_read_dset_names)
- Python [hdf5_create_file](python_api.html#hdf5_create_file), [hdf5_write_channel](python_api.html#hdf5_write_channel), [hdf5_read_channel](python_api.html#hdf5_read_channel), [hdf5_read_layout](python_api.html#hdf5_read_layout), [hdf5_reshape_layout](python_api.html#hdf5_reshape_layout), [hdf5_write_dset](python_api.html#hdf5_write_dset), [hdf5_read_dset](python_api.html#hdf5_read_dset), [hdf5_read_dset_names](python_api.html#hdf5_read_dset_names)

## Slots and storage layout {#hdf5-slots}

- The slots form a grid of `nx · ny · nz · nw` places, the **storage layout** of the file, at most 4294967295 (2^32 − 1) in total. The number of slots is fixed when the file is created; [hdf5_reshape_layout](cpp_api.html#hdf5_reshape_layout) changes only the dimensions.
- Slot `(ix, iy, iz, iw)` (0-based) has the linear index `ix + nx · (iy + ny · (iz + nz · iw))`; the MATLAB / Octave functions take the same indices 1-based.
- A slot holds one channel or nothing. Writing a channel to an occupied slot replaces everything in it, also its metadata. The space of replaced data is not reused, so the file does not shrink.

## Compatibility {#hdf5-compatibility}

- The files are HDF5 files of the quadriga-lib channel format, version 4. Standard HDF5 readers open them, e.g. h5py or MATLAB `h5read`; the library itself needs no HDF5 installation.
- Change files only with the functions of the library: files changed by other HDF5 tools are not supported and may be rejected.
- Files of version 2 (quadriga-lib 0.12.4 and earlier) are rejected; `tools/hdf5_convert_v2.py` in the source package converts them (needs h5py). Files of version 3 (development versions only) are rejected.
- Files are not locked: one writer per file at a time. Reading is safe while no one writes.

## Object tree {#hdf5-object-tree}

The file is a tree of groups (folders), datasets (arrays) and attributes (small values attached to a group). The names are those of the [channel](cpp_api.html#channel) members. Shapes are given as HDF5 and h5py list them, the slowest dimension first: a C++ / MATLAB array `[rows, cols, slices]` is stored as `(slices, cols, rows)` with the same memory order, so MATLAB `h5read` returns the API shape. Values are float32 unless noted.

**Root group**:

| Name | Type, shape | Content |
| ---- | ----------- | ------- |
| `version` | uint32, scalar | Format version, 4 |
| `layout` | uint32 `(4)` | Storage layout `{nx, ny, nz, nw}` |
| `channel_id` | uint32 `(n_slot)` | Channel ID of each slot by linear index; 0 = empty |
| `channel_ref` | object reference `(n_slot)` | Reference to the channel group of each slot; 0 = empty. The library finds a slot through it. MATLAB `h5read` cannot read it while any slot is empty |
| `channel_<id>` | group | One per occupied slot; `<id>` is its channel ID in decimal |

**Channel group** `channel_<id>`: the attributes describe the channel. A value that was not set keeps its default:

| Name | Type, shape | Content | Default |
| ---- | ----------- | ------- | ------- |
| `name` | string, 256 bytes | Channel name, NUL-terminated | empty |
| `n_snap` | uint32 | Number of snapshots | 0 |
| `n_tx`, `n_rx` | uint32 | Number of Tx and Rx antennas | 0 |
| `n_path` | uint32 | Number of paths; only in a channel with one snapshot that has path data | |
| `initial_position` | int32 | Reference snapshot, 0-based | 0 |
| `center_freq` | float32 | Center frequency in Hz | NaN |
| `tx_pos`, `rx_pos` | float32 `(3)` | Tx and Rx position in m | NaN |
| `tx_orientation`, `rx_orientation` | float32 `(3)` | Tx and Rx orientation in rad | 0 |

Center frequency, positions and orientations are stored once as the attribute when they apply to all snapshots. When they have one value per snapshot, they are datasets of the same name in the channel group instead, `center_freq` `(n_snap)` and the others `(n_snap, 3)`, and the attribute keeps its default; readers check for the dataset first. A slot that holds only metadata (written with [hdf5_write_dset](cpp_api.html#hdf5_write_dset)) has a channel group with all defaults and only the `par` group.

**Snapshot data**: directly in the channel group for a channel with one snapshot, otherwise in one group `snap_<i>` per snapshot (`i` from 0), each with an `n_path` attribute. `n_path` is the number of paths of the snapshot:

| Name | Type, shape | Content |
| ---- | ----------- | ------- |
| `coeff_re`, `coeff_im` | `(n_path, n_tx, n_rx)` | Coefficients, real and imaginary part |
| `delay` | `(n_path, n_tx, n_rx)` or `(n_path, 1, 1)` | Delays in s |
| `path_gain` | `(n_path)` | Path gain, linear, without antennas |
| `path_length` | `(n_path)` | Path length in m |
| `polmat` | `(n_path, 8)` | Polarization transfer matrix |
| `path_angles` | `(4, n_path)` | AOD, EOD, AOA, EOA in rad |
| `fbs_pos`, `lbs_pos` | `(n_path, 3)` | First- and last-bounce scatterer positions in m |
| `n_interact` | uint32 `(n_path)` | Number of interaction points of each path |
| `interact_coord` | `(sum(n_interact), 3)` | Interaction points of all paths in m |

A dataset exists only if the channel has the value. If any snapshot has it, every snapshot stores it, a snapshot without paths as an empty dataset.

**Metadata**: the group `par` in the channel group, present only if the slot has metadata, holds one dataset per value, named `par_<name>` (the prefix can be chosen in [hdf5_write_dset](cpp_api.html#hdf5_write_dset)). Names hold only letters, digits and `_`. A string is a fixed-length string, a scalar a scalar dataset, a vector rank 1, a matrix `(cols, rows)`, a cube `(slices, cols, rows)`; element types float32, float64, uint32, int32, uint64, int64, stored without conversion. The group keeps the order in which the values were written, and [hdf5_read_channel](cpp_api.html#hdf5_read_channel) returns the values with the prefix `par_` in that order.
