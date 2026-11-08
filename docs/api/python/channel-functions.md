# Channel functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-channel-functions

| Function | Summary |
| --- | --- |
| [baseband_freq_response](#baseband_freq_response) | Compute the baseband frequency response of a MIMO channel |
| [channel_export_obj_file](#channel_export_obj_file) | Export propagation paths to a Wavefront OBJ file for 3D visualization |
| [hdf5_create_file](#hdf5_create_file) | Create a new HDF5 channel file with a custom storage layout |
| [hdf5_read_channel](#hdf5_read_channel) | Reads channel data from HDF5 files |
| [hdf5_read_dset](#hdf5_read_dset) | Read a single unstructured dataset from an HDF5 file |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Read the names of unstructured datasets from an HDF5 file |
| [hdf5_read_layout](#hdf5_read_layout) | Read the storage layout of channel data inside an HDF5 file |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Reshapes the storage layout inside an existing HDF5 file |
| [hdf5_write_channel](#hdf5_write_channel) | Write one or more channel objects to an HDF5 file |
| [hdf5_write_dset](#hdf5_write_dset) | Write a single unstructured dataset to an HDF5 file |
| [qrt_file_parse](#qrt_file_parse) | Read metadata from a QRT file |
| [qrt_file_read](#qrt_file_read) | Read ray-tracing CIR data from a QRT file |
| [quantize_delays](#quantize_delays) | Map path delays to a fixed tap grid using two-tap power-weighted interpolation |

---
<a id="baseband_freq_response"></a>
## baseband_freq_response
Compute the baseband frequency response of a MIMO channel

Also in: [C++](../cpp/channel-functions.md#baseband_freq_response) · [MATLAB](../mex/channel-functions.md#baseband_freq_response)

- Transforms time-domain channel coefficients into the frequency-domain transfer function H(f)
- Snapshots are passed as Python lists; each list item is one snapshot, processed in parallel via OpenMP
- Two per-snapshot modes, selected by the ndim of the coefficient arrays:
  - single-freq: 3D items `(n_rx, n_tx, n_path)`; DFT over path delays at the requested carriers (AVX2)
  - multi-freq: 4D items `(n_rx, n_tx, n_path, n_freq_in)`; SLERP interpolation across `freq_in`,
    then delay-induced phase applied per output carrier
- All list items must share ndim, `n_rx`, `n_tx` (and `n_freq_in` for 4D); `n_path` may vary per snapshot
- Coefficients are given either as one complex array `coeff`, or as split `coeff_re` + `coeff_im`;
  supplying both forms is an error
- Output carriers are defined one of two mutually exclusive ways:
  - `bandwidth` paired with `pilot_grid` (or `carriers` for an evenly spaced grid) — single-freq only
  - `freq_in` + `freq_out` as absolute frequencies — required for multi-freq, optional for single-freq
- For single-freq with `freq_in` + `freq_out`, the baseband grid is derived internally; the phase is
  referenced to `freq_in[0]` when given, else to `min(freq_out)`
- `delay` may be broadcast: a `(1, 1, n_path)` item applies the same delays to every RX/TX pair;
  for multi-freq a 3D delay item is reused for all input frequencies
- `snap` selects a 0-based subset of snapshots; omitted processes all snapshots
- single-freq: the phase `bandwidth · pilot_grid · delay` (in cycles) is computed in double precision and reduced to
  one period before sin / cos are evaluated in single precision; coefficients and sums are single precision, double
  inputs are narrowed to float, results widened back; multi-freq computes in double precision

### Usage:
```python
# Single-freq, bandwidth + evenly spaced carriers
hmat = quadriga_lib.channel.baseband_freq_response( coeff=coeff, delay=delay, bandwidth=100e6, \
    carriers=128 )

# Single-freq, absolute frequencies
hmat = quadriga_lib.channel.baseband_freq_response( coeff=coeff, delay=delay, freq_in=freq_in, \
    freq_out=freq_out )

# Multi-freq, split real / imaginary parts
hmat = quadriga_lib.channel.baseband_freq_response( coeff_re=cre, coeff_im=cim, delay=delay, \
    freq_in=freq_in, freq_out=freq_out )
```

### Inputs:
- **`coeff`** — Complex channel coefficients; list of length `n_snap`; each item `(n_rx, n_tx, n_path)`
  (single-freq) or `(n_rx, n_tx, n_path, n_freq_in)` (multi-freq); mutually exclusive with
  `coeff_re` / `coeff_im`; default: `None`
- **`delay`** — Path delays in seconds; list of length `n_snap`; each item shaped like the matching
  `coeff` item or broadcast as `(1, 1, n_path)` (with an optional 4th dimension `n_freq_in`)
- **`bandwidth`** — Baseband bandwidth in Hz; paired with `pilot_grid` or `carriers`; single-freq only;
  cannot be combined with `freq_out`; default: `0.0`
- **`carriers`** — Number of evenly spaced carriers on `[0, 1]`; used only when both `pilot_grid` and
  `freq_out` are omitted; default: 128
- **`pilot_grid`** — Normalized sub-carrier positions, `0.0` = center, `1.0` = center + `bandwidth`;
  1D array of length `n_carrier`; default: `None`
- **`snap`** — 0-based snapshot indices to process; 1D array of length `n_out`; omitted processes
  all snapshots; default: `None`
- **`coeff_re`** — Real part of channel coefficients; same list/shape layout as `coeff`; must be
  paired with `coeff_im`; mutually exclusive with `coeff`; default: `None`
- **`coeff_im`** — Imaginary part of channel coefficients; same layout as `coeff_re`; default: `None`
- **`freq_in`** — Input sample frequencies in Hz; 1D array; required for multi-freq input; for
  single-freq input used together with `freq_out` as the phase reference; default: `None`
- **`freq_out`** — Absolute output carrier frequencies in Hz; 1D array of length `n_carrier`;
  required for multi-freq input; may replace `bandwidth` + `pilot_grid` for single-freq; default: `None`
- **`remove_delay_phase`** — If `True`, undo the delay-induced phase baked in by the channel generator before SLERP,
  then re-apply it analytically per output carrier; multi-freq only; needs channels generated with
  `use_absolute_delays = True`: the coefficients of `get_channels_*` carry the phase of the absolute path length also
  when `delay` is relative, and with relative delays the phase of the TX-RX distance is interpolated across `freq_in`,
  which gives wrong phases (magnitudes stay right); default: `True`

### Outputs:
- **`hmat`** — Complex frequency-domain channel matrix; shape `(n_rx, n_tx, n_carrier, n_out)`,
  where `n_out` is the number of processed snapshots

### See also:
- [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (single-frequency channel generator)
- [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq) (multi-frequency channel generator)

---
<a id="channel_export_obj_file"></a>
## channel_export_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

Also in: [C++](../cpp/channel-class.md#.write_paths_to_obj_file) · [MATLAB](../mex/channel-functions.md#channel_export_obj_file)

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g. for use in Blender)
- Tubes are color-coded by path gain using a selected colormap; tube radius also scales with gain
- Paths below `gain_min` are excluded; `max_no_paths` caps the total number of exported paths
- Paths with non-finite coordinates are skipped
- Takes raw channel data fields directly; no channel dict is required
- The per-snapshot fields `no_interact`, `interact_coord` and `coeff` each accept either a `list`
  (one entry per snapshot, ragged) or a single padded `ndarray` (MATLAB-style nD layout)

### Usage:
```python
quadriga_lib.channel.channel_export_obj_file( fn, max_no_paths, gain_max, gain_min, colormap, \
    radius_max, radius_min, n_edges, rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff, \
    coeff_re, coeff_im, i_snap )
```

### Inputs:
- **`fn`** — Output `.obj` file path; str
- **`max_no_paths`** — Max paths to export; 0 includes all paths above `gain_min`; default: 0
- **`gain_max`** — Upper gain threshold in dB for color/radius mapping; higher values are clipped; default: -60.0
- **`gain_min`** — Lower gain threshold in dB; paths below this are excluded; default: -140.0
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`radius_max`** — Tube radius at maximum gain; default: 0.05
- **`radius_min`** — Tube radius at minimum gain; default: 0.01
- **`n_edges`** — Vertices per tube cross-section; must be >= 3; default: 5
- **`rx_pos`** — Receiver positions; ndarray `(3, n_snap)` or `(3, 1)`
- **`tx_pos`** — Transmitter positions; ndarray `(3, n_snap)` or `(3, 1)`
- **`no_interact`** — Number of interaction points of each path with the environment; uint32;
  either a `list` of `n_snap` arrays of shape `(n_path,)`, or an ndarray `(n_path, n_snap)`
- **`interact_coord`** — Interaction coordinates; either a `list` of `n_snap` arrays of shape
  `(3, sum(no_interact))`, or an ndarray `(3, max(sum(no_interact)), n_snap)`; fewer than `sum(no_interact)`
  columns in a snapshot raise an error
- **`center_freq`** — Center frequency in Hz; ndarray `(n_snap,)` or scalar
- **`coeff`** — Channel coefficients, complex valued; either a `list` of `n_snap` arrays of shape
  `(n_rx, n_tx, n_path)`, or an ndarray `(n_rx, n_tx, n_path, n_snap)`; provide this or
  `coeff_re`/`coeff_im`, not both; default: None
- **`coeff_re`** — Real part of the channel coefficients; same format as `coeff` but real-valued;
  must be paired with `coeff_im`; default: None
- **`coeff_im`** — Imaginary part of the channel coefficients; same format as `coeff_re`;
  must be paired with `coeff_re`; default: None
- **`i_snap`** — Snapshot indices to include; 0-based; `None` or empty exports all; default: None

### Outputs:
- None; the OBJ file is written directly to disk

---
<a id="hdf5_create_file"></a>
## hdf5_create_file
Create a new HDF5 channel file with a custom storage layout

Also in: [C++](../cpp/channel-functions.md#hdf5_create) · [MATLAB](../mex/channel-functions.md#hdf5_create_file)

- Initializes a new HDF5 file for storing channel data
- Defines a 4D storage layout `(nx, ny, nz, nw)`; each index combination maps to one channel slot
- Typical dimension mapping: nx = BS, ny = UE, nz = frequency, nw = scenario/repetition
- The total slot count is fixed at creation; the layout can be reshaped with [hdf5_reshape_layout](#hdf5_reshape_layout)
- Raises an error if the target file already exists; delete it first to recreate it
- Files can be read by any HDF5 tool but must be modified only with quadriga-lib

### Usage:
```python
storage_space = quadriga_lib.channel.hdf5_create_file( fn, nx, ny, nz, nw )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file to create; str
- **`nx`** — Number of elements on the x-dimension; default: 65536
- **`ny`** — Number of elements on the y-dimension; default: 1
- **`nz`** — Number of elements on the z-dimension; default: 1
- **`nw`** — Number of elements on the w-dimension; default: 1

### Outputs:
- **`storage_space`** — Storage dimensions used by the new file, `[nx, ny, nz, nw]`; `(4,)`; int64

### See also:
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data)
- [hdf5_write_dset](#hdf5_write_dset) (for writing arbitrary unstructured data)

---
<a id="hdf5_read_channel"></a>
## hdf5_read_channel
Reads channel data from HDF5 files

Also in: [C++](../cpp/channel-functions.md#hdf5_read_channel) · [MATLAB](../mex/channel-functions.md#hdf5_read_channel)

- Reads structured channel data and any unstructured datasets from a 4D indexed HDF5 file
- Each of ix, iy, iz, iw may be a scalar, vector, or omitted (= None)
- Datasets span `n_rx` RX antennas, `n_tx` TX antennas, `n_path` paths and `n_snap` snapshots;
  snapshots typically index positions along a trajectory or frequencies
- Slots are visited in column-major order and empty slots are skipped.
- Not every dataset spans all dimensions; only datasets present in the file are returned
- Per-snapshot data is returned as a `list` of arrays, one per snapshot, since `n_path` may differ between snapshots
- Structured fields are stored in single precision in the file and returned in double.
- Unstructured datasets keep their stored type and shape.
- `snap` selects a subset of snapshots; if `None`, all snapshots are read
- Errors if the file is not a quadriga-lib channel file of version 3: files written by quadriga-lib 0.12.4 and
  earlier are rejected; files modified with other HDF5 tools are not supported and may be rejected.
- Convert files of quadriga-lib 0.12.4 and earlier with `tools/hdf5_convert_v2.py` of the quadriga-lib sources
  (needs h5py): `python tools/hdf5_convert_v2.py old.hdf5 new.hdf5`.

### Usage:
```python
chan, par = quadriga_lib.channel.hdf5_read_channel( fn, ix, iy, iz, iw, snap, stack )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — 0-based slot indices along dimension X; scalar or vector; default: `0 ... nx-1`
- **`iy`** — 0-based slot indices along dimension Y; scalar or vector; default: `0 ... ny-1`
- **`iz`** — 0-based slot indices along dimension Z; scalar or vector; default: `0 ... nz-1`
- **`iw`** — 0-based slot indices along dimension W; scalar or vector; default: `0 ... nw-1`
- **`snap`** — Snapshot indices to read; 0-based; default: all snapshots. Only allowed when the total selection is a single slot.
- **`stack`** — If `True`, stack snapshots. Default: `False` (return as list)

### Outputs:
- **`chan`** — List of dicts with the following keys:
  | Key                | Description                                                              | Shape `stack = False`             | Shape `stack = True`                 |
  | ------------------ | ------------------------------------------------------------------------ | --------------------------------- | ------------------------------------ |
  | `name`             | Channel name                                                             | str                               | str                                  |
  | `tx_position`      | Transmitter positions (AP for downlink, STA for uplink)                  | `(3, n_snap)` or `(3, 1)`         | `(3, n_snap)` or `(3, 1)`            |
  | `rx_position`      | Receiver positions (STA for downlink, AP for uplink)                     | `(3, n_snap)` or `(3, 1)`         | `(3, n_snap)` or `(3, 1)`            |
  | `tx_orientation`   | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | `(3, n_snap)` or `(3, 1)`         | `(3, n_snap)` or `(3, 1)`            |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | `(3, n_snap)` or `(3, 1)`         | `(3, n_snap)` or `(3, 1)`            |
  | `coeff`            | Channel coefficients, complex-valued                                     | list of `(n_rx, n_tx, n_path_s)`  | `(n_rx, n_tx, n_path, n_snap)`       |
  | `delay`            | Propagation delays in seconds                                            | list of `(n_rx, n_tx, n_path_s)`  | `(n_rx, n_tx, n_path, n_snap)`       |
  | `path_gain`        | Path gain before antenna, linear scale                                   | list of `(n_path_s,)`             | `(n_path, n_snap)`                   |
  | `path_length`      | Path length in m                                                         | list of `(n_path_s,)`             | `(n_path, n_snap)`                   |
  | `path_polarization`| Polarization transfer function, complex                                  | list of `(4, n_path_s)`           | `(4, n_path, n_snap)`                |
  | `path_angles`      | Departure and arrival angles (AOD, EOD, AOA, EOA) in rad                 | list of `(n_path_s, 4)`           | `(n_path, 4, n_snap)`                |
  | `fbs_pos`          | First-bounce scatterer positions                                         | list of `(3, n_path_s)`           | `(3, n_path, n_snap)`                |
  | `lbs_pos`          | Last-bounce scatterer positions                                          | list of `(3, n_path_s)`           | `(3, n_path, n_snap)`                |
  | `no_interact`      | Number of interaction points per path; int64                             | list of `(n_path_s,)`             | `(n_path, n_snap)`                   |
  | `interact_coord`   | Interaction coordinates                                                  | list of `(3, sum(no_interact_s))` | `(3, max(sum(no_interact)), n_snap)` |
  | `center_frequency` | Center Frequency in Hz                                                   | `(n_snap,)` or scalar             | `(n_snap,)` or scalar                |
  | `initial_position` | Index of reference position; 1-based                                     | int32, scalar                     | int32, scalar                        |

- **`par`** — Dict (single slot) or list of dicts (multiple slots) containing the unstructured data in the file.

### See also:
- [hdf5_read_layout](#hdf5_read_layout) (for reading the layout in the file)
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data)
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)

---
<a id="hdf5_read_dset"></a>
## hdf5_read_dset
Read a single unstructured dataset from an HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_read_dset) · [MATLAB](../mex/channel-functions.md#hdf5_read_dset)

- Reads one user-defined dataset from the slot addressed by the 0-based indices `(ix, iy, iz, iw)`
- The dataset is looked up under `prefix + name` (default prefix `par_`)
- The returned type and shape are defined by the dataset's HDF5 dataspace: a scalar dataset gives a Python
  number, a vector a numpy array of shape `(n,)` (also with a single element), a matrix or 3D array a numpy array
  of the stored shape
- Returns `None` if the dataset does not exist at the requested slot
- Errors if the file does not exist or is not a quadriga-lib channel file of version 3 (see
  [hdf5_read_channel](#hdf5_read_channel))
- Supported types: str, scalar, vector, 2D array, and 3D array

### Usage:
```python
dset = quadriga_lib.channel.hdf5_read_dset( fn, ix, iy, iz, iw, name, prefix )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`name`** — Dataset name without the prefix, e.g. `'carrier_frequency'`; str
- **`prefix`** — Prefix prepended to `name` when looking up the dataset; str; default: `'par_'`

### Outputs:
- **`dset`** — Dataset contents; type and shape are defined by the HDF5 dataspace; `None` if the dataset is missing

### See also:
- [hdf5_read_dset_names](#hdf5_read_dset_names) (for reading names of already written datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)
- [hdf5_read_channel](#hdf5_read_channel) (for reading structured channel data)

---
<a id="hdf5_read_dset_names"></a>
## hdf5_read_dset_names
Read the names of unstructured datasets from an HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_read_dset_names) · [MATLAB](../mex/channel-functions.md#hdf5_read_dset_names)

- Lists all unstructured datasets stored at the slot addressed by the 0-based indices `(ix, iy, iz, iw)`
- Lists the datasets whose names start with `prefix` (default `par_`; `''` lists all); the returned names have the
  prefix stripped
- Returns an empty list if no unstructured datasets are present at the slot
- Errors if the file does not exist or is not a quadriga-lib channel file of version 3 (see
  [hdf5_read_channel](#hdf5_read_channel))

### Usage:
```python
names = quadriga_lib.channel.hdf5_read_dset_names( fn, ix, iy, iz, iw, prefix )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`prefix`** — Prefix used to identify unstructured datasets; `''` lists all datasets; str; default: `'par_'`

### Outputs:
- **`names`** — Names of the unstructured datasets at the slot whose names start with `prefix`, with the prefix
  stripped, in creation order; list of str

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)

---
<a id="hdf5_read_layout"></a>
## hdf5_read_layout
Read the storage layout of channel data inside an HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_read_layout) · [MATLAB](../mex/channel-functions.md#hdf5_read_layout)

- Returns the dimensions of the 4D channel-slot grid stored inside an HDF5 file
- Also reports which slots already hold data, so free slots can be found without scanning the file
- Returns `(0, 0, 0, 0)` dimensions if the file does not exist
- Raises an error if the file exists but is not a quadriga-lib channel file of version 3 (files written by
  quadriga-lib 0.12.4 and earlier are rejected; files modified with other HDF5 tools are not supported and may be
  rejected)

### Usage:
```python
storage_dims, has_data = quadriga_lib.channel.hdf5_read_layout( fn )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; str

### Outputs:
- **`storage_dims`** — Size of the storage space `[nx, ny, nz, nw]`; `(4,)`; int64
- **`has_data`** — Slot occupancy mask; 1 where a slot holds data, 0 otherwise; uint32; `(nx, ny, nz, nw)`

### See also:
- [hdf5_create_file](#hdf5_create_file) (for creating a file with a custom storage layout)
- [hdf5_read_channel](#hdf5_read_channel) (for reading channel data)

---
<a id="hdf5_reshape_layout"></a>
## hdf5_reshape_layout
Reshapes the storage layout inside an existing HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_reshape_layout) · [MATLAB](../mex/channel-functions.md#hdf5_reshape_layout)

- Changes the 4D slot grid `(nx, ny, nz, nw)` of an existing HDF5 channel file
- The total number of slots (`nx · ny · nz · nw`) must match the original layout
- Only the dimension metadata is updated; stored channel data is not moved
- Errors if the file does not exist or is not a quadriga-lib channel file of version 3 (files written by
  quadriga-lib 0.12.4 and earlier are rejected; files modified with other HDF5 tools are not supported and may be
  rejected)
- One writer per file at a time; files are not locked

### Usage:
```python
quadriga_lib.channel.hdf5_reshape_layout( fn, nx, ny, nz, nw )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file to create; str
- **`nx`** — Number of elements on the x-dimension; default: 65536
- **`ny`** — Number of elements on the y-dimension; default: 1
- **`nz`** — Number of elements on the z-dimension; default: 1
- **`nw`** — Number of elements on the w-dimension; default: 1

---
<a id="hdf5_write_channel"></a>
## hdf5_write_channel
Write one or more channel objects to an HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_write) · [MATLAB](../mex/channel-functions.md#hdf5_write_channel)

- Writes a list of channel dicts into 4D storage slots (one slot per list entry)
- `chan` may also be a single dict, in which case one channel is written
- Optional unstructured data (`par`) can be passed as a matching list of dicts (or a single dict)
- Creates the file with a default layout if it does not exist; an existing file must be a quadriga-lib channel
  file of version 3 (files of quadriga-lib 0.12.4 and earlier are rejected, see [hdf5_read_channel](#hdf5_read_channel))
- A warning is issued if any selected slot already contains data. That slot is replaced completely: its old
  structured data and all its unstructured datasets (also those written with [hdf5_write_dset](#hdf5_write_dset)) are removed.
  The file does not shrink; copy the channels into a new file to reclaim the space of the removed data
- One writer per file at a time; files are not locked
- Structured data is stored in single precision regardless of the input precision
- Unstructured datasets retain their numpy dtype and shape; supported types as in [hdf5_write_dset](#hdf5_write_dset)
- Each scalar index input is broadcast to all channels; each vector index must have one entry per channel
- If the file does not exist, the layout is `[max(len(chan), max(ix)+1), max(iy)+1, max(iz)+1, max(iw)+1]`
- Channel dict field layout matches [hdf5_read_channel](#hdf5_read_channel)
- Per-snapshot fields accept the list model (list of arrays) or the stack model
  (single array with the snapshot index as the last axis), detected per field
- Coefficients may be passed as a complex `coeff`, or as separate real `coeff_re` and `coeff_im` (the two forms are mutually exclusive)
- Slot indices are 0-based

### Usage:
```python
storage_dims = quadriga_lib.channel.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`chan`** — Structured channel data; a channel dict or a list of channel dicts; field layout matches [hdf5_read_channel](#hdf5_read_channel)
- **`par`** — Unstructured data; a dict or a list of dicts with the same number of entries as `chan`. Dict keys become HDF5 dataset
  names per slot; `None` values are skipped. Pass `None` to disable; default: `None`
- **`ix`** — 0-based slot indices along dimension X; scalar or vector of length `len(chan)`; default: `0 ... len(chan)-1`
- **`iy`** — 0-based slot indices along dimension Y; scalar or vector of length `len(chan)`; default: 0
- **`iz`** — 0-based slot indices along dimension Z; scalar or vector of length `len(chan)`; default: 0
- **`iw`** — 0-based slot indices along dimension W; scalar or vector of length `len(chan)`; default: 0

### Outputs:
- **`storage_dims`** — Storage layout dimensions of the file `[nx, ny, nz, nw]`; `(4,)`; int64

### See also:
- [hdf5_create_file](#hdf5_create_file) (for creating a file with a custom storage layout)
- [hdf5_reshape_layout](#hdf5_reshape_layout) (to change the layout later)
- [hdf5_read_channel](#hdf5_read_channel) (for reading channel data)

---
<a id="hdf5_write_dset"></a>
## hdf5_write_dset
Write a single unstructured dataset to an HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_write_dset) · [MATLAB](../mex/channel-functions.md#hdf5_write_dset)

- Dataset is stored under `prefix + name` at slot `(ix, iy, iz, iw)`
- `prefix + name` must contain only alphanumeric characters and underscores
- The file must already exist (use [hdf5_create_file](#hdf5_create_file) first) and be a quadriga-lib channel file of
  version 3 (see [hdf5_read_channel](#hdf5_read_channel))
- A dataset of the same name at the same slot is not overwritten; an error is thrown instead
- The dataset is added to the slot; everything else stored there is kept. A later [hdf5_write_channel](#hdf5_write_channel) to the
  same slot replaces the slot completely, including the datasets written with this function
- Supported types: str; Python float (stored as double), int and bool (stored as int64; a bool is stored as the
  integer 0 or 1 and reads back as an int); numpy scalars and 0-d arrays; numpy arrays with 1, 2 or 3 dimensions
  in any memory layout and byte order; numeric element types: float32, float64, int32, uint32, int64, uint64.
  Other types (e.g. int16, numpy bool, complex) raise an error
- 1D arrays are stored as vectors and read back with shape `(n,)`, also with a single element (not as a scalar);
  2D and 3D arrays keep their shape, also `(n, 1)` and `(1, 1)`
- One writer per file at a time; files are not locked

### Usage:
```python
quadriga_lib.channel.hdf5_write_dset( fn, ix, iy, iz, iw, name, data, prefix )
```

### Input Arguments:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; alphanumeric and underscores only; str
- **`data`** — Data to be written; type must be supported (see above); cannot be empty
- **`prefix`** — Prefix prepended to `name` in the HDF5 file; alphanumeric and underscores only; str; default: `'par_'`

### See also:
- [hdf5_read_dset_names](#hdf5_read_dset_names) (for reading names of already written datasets)
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)

---
<a id="qrt_file_parse"></a>
## qrt_file_parse
Read metadata from a QRT file

Also in: [C++](../cpp/channel-functions.md#qrt_file_parse) · [MATLAB](../mex/channel-functions.md#qrt_file_parse)

- Parses a QRT file and extracts snapshot counts, origin/destination counts, frequency count, CIR
  offsets, names, positions, orientations, and file version
- When `no_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and
  `cir_offset` reflect this

### Usage:
```python
no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version, center_freq, \
    cir_pos, cir_orientation, orig_pos, orig_orientation = quadriga_lib.channel.qrt_file_parse( fn )
```

### Inputs:
- **`fn`** — Path to the QRT file

### Outputs:
- **`no_cir`** — Number of channel snapshots per origin point
- **`no_orig`** — Number of origin points (TX)
- **`no_dest`** — Number of destination points (RX)
- **`no_freq`** — Number of frequency bands
- **`cir_offset`** — CIR offset per destination; `(no_dest,)`
- **`orig_names`** — Names of the origin points (TX); list of strings; length `no_orig`
- **`dest_names`** — Names of the destination points (RX); list of strings; length `no_dest`
- **`version`** — QRT file version number
- **`center_freq`** — Frequencies as stored in the file; GHz for EM mode (v4/v5), Hz for scalar mode (v6); `(no_freq,)`
- **`cir_pos`** — CIR positions in Cartesian coordinates; `(no_cir, 3)`
- **`cir_orientation`** — CIR orientations as Euler angles; `(no_cir, 3)`
- **`orig_pos`** — Origin (TX) positions in Cartesian coordinates; `(no_orig, 3)`
- **`orig_orientation`** — Origin (TX) orientations as Euler angles; `(no_orig, 3)`

---
<a id="qrt_file_read"></a>
## qrt_file_read
Read ray-tracing CIR data from a QRT file

Also in: [C++](../cpp/channel-functions.md#qrt_file_read) · [MATLAB](../mex/channel-functions.md#qrt_file_read)

- Reads channel impulse response data from QRT files
- A file read cache is initialized once and reused across all requested snapshots, which
  significantly speeds up multi-snapshot reads
- File reading and the conversion to numpy arrays are overlapped: a background thread reads the
  next snapshot into a double buffer while the main thread converts the previous one
- If `downlink = True`, origin is TX and destination is RX; if `False`, the roles are swapped
- Per-snapshot outputs are returned as lists with one entry per requested snapshot

### Usage:
```python
center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, \
    path_length, M, aod, eod, aoa, eoa, path_coord, no_int, coord, \
    interact_type = quadriga_lib.channel.qrt_file_read( fn, i_cir, i_orig, downlink, normalize_M )
```

### Inputs:
- **`fn`** — Path to the QRT file
- **`i_cir`** — Snapshot indices to read, 0-based; `(n_out,)` or `None`; if `None` or empty, all snapshots are read (for a file
  without CIRs, all outputs are empty with `n_out = 0`); default: `None`
- **`i_orig`** — Origin index, 0-based (origin = TX for downlink); scalar; default: 0
- **`downlink`** — If `True`, origin=TX and destination=RX; if `False`, the roles are swapped; default: `True`
- **`normalize_M`** — Controls `M` and `path_gain` scaling; 0 = values as stored in the QRT file with
  `path_gain` = -PL; 1 = `M` columns scaled to max power 1 and `path_gain` = -PL minus material losses; default: 1
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm
    | `normalize_M` | `M`                   | `path_gain`                      |
    | :-----------: | :-------------------: | :------------------------------: |
    | 0             | As stored in QRT file | -PL                              |
    | 1             | Max column power = 1  | -PL minus material losses        |

    Other values raise an error.

### Outputs:
- **`center_freq`** — Center frequencies in Hz; `(n_freq,)`
- **`tx_pos`** — Transmitter positions in Cartesian coordinates; `(3, n_out)`
- **`tx_orientation`** — Transmitter orientations as Euler angles (bank, tilt, heading); `(3, n_out)`
- **`rx_pos`** — Receiver positions in Cartesian coordinates; `(3, n_out)`
- **`rx_orientation`** — Receiver orientations as Euler angles (bank, tilt, heading); `(3, n_out)`
- **`fbs_pos`** — First-bounce scatterer positions; list of length `n_out`; entries `(3, n_path)`
- **`lbs_pos`** — Last-bounce scatterer positions; list of length `n_out`; entries `(3, n_path)`
- **`path_gain`** — Path gain in linear scale; list of length `n_out`; entries `(n_path, n_freq)`
- **`path_length`** — Absolute path length from TX to RX phase center; list of length `n_out`; entries `(n_path,)`
- **`M`** — Polarization transfer matrix, stored as interleaved real/imaginary pairs; list of length `n_out`; entries `(8, n_path, n_freq)`, or `(2, n_path, n_freq)` for v6 files
- **`aod`** — Departure azimuth angles; list of length `n_out`; entries `(n_path,)`
- **`eod`** — Departure elevation angles; list of length `n_out`; entries `(n_path,)`
- **`aoa`** — Arrival azimuth angles; list of length `n_out`; entries `(n_path,)`
- **`eoa`** — Arrival elevation angles; list of length `n_out`; entries `(n_path,)`
- **`path_coord`** — Interaction coordinates per path; list of length `n_out`; each entry is a list of length `n_path` with arrays `(3, n_interact + 2)`
- **`no_int`** — Number of mesh interactions per path (0 indicates LOS); list of length `n_out`; entries `(n_path,)`
- **`coord`** — Interaction coordinates concatenated across paths; list of length `n_out`; entries `(3, sum(no_int))`
- **`interact_type`** — Interaction type codes concatenated across paths (matching `coord`); list of length `n_out`;
  entries `(sum(no_int),)`. Empty for v4 legacy files. GBSM paths written with the C++ API carry 129 (GR) and
  193 (NLOS); LOS and GR paths are not marked as GBSM.

### See also:
- [generate](array-antenna-functions.md#generate) (for generating antenna arrays)
- [get_channels_planar](channel-generation-functions.md#get_channels_planar) (for embedding antennas using departure and arrival angles)
- [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (for embedding antennas using FBS/LBS positions)
- [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq) (for multi-frequency antenna embedding)

---
<a id="quantize_delays"></a>
## quantize_delays
Map path delays to a fixed tap grid using two-tap power-weighted interpolation

Also in: [C++](../cpp/channel-functions.md#quantize_delays) · [MATLAB](../mex/channel-functions.md#quantize_delays)

- Each path delay is approximated by two adjacent taps with coefficients scaled by (1−δ)^α and δ^α,
  where δ is the fractional offset within the bin and α is `power_exponent`
- Two-tap interpolation avoids discontinuities when delays cross tap boundaries
- Use `power_exponent = 1.0` for narrowband (linear interpolation) or `0.5` for wideband
  (incoherent power preservation)
- If all fractional per-tap offsets are below 0.01 or above 0.99, weight computation is skipped
  (nearest-neighbor selection) but tap-selection logic still applies
- Snapshots are passed as Python lists; each list item is one snapshot and `n_path` may vary per item
- Coefficients are given either as one complex list `coeff`, or as split `coeff_re` + `coeff_im`;
  supplying both forms is an error. The input and output forms are independent
- `complex=True` returns one combined complex coefficient output; `complex=False` (default) returns
  separate real `coeff_re_q` and `coeff_im_q`
- Input `delay` items may be per-antenna `(n_rx, n_tx, n_path)` or shared `(1, 1, n_path)`;
  shared delays are expanded internally when `fix_taps` is 0 or 3
- Output arrays are zero-padded along the tap dimension so that all snapshots share the same `n_taps`
- Delays must be non-negative, as for a tapped delay line, with a tolerance of `tap_spacing / 2`: delays in
  `(-tap_spacing / 2, 0)` are treated as 0, so all of their power goes to tap 0. Delays at or below
  `-tap_spacing / 2` raise `ValueError`, as do NaN, Inf and delays of 2^31 taps or more (`delay / tap_spacing < 2^31`,
  e.g. 10.7 s at 5 ns or 12.4 h at 48 kHz)
- Relative delays (`use_absolute_delays = false` in [get_channels_spherical](channel-generation-functions.md#get_channels_spherical), [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq) and
  [get_channels_planar](channel-generation-functions.md#get_channels_planar)) are negative for element pairs that are closer to each other than the phase centres, by up
  to about (half the TX array size + half the RX array size) / propagation speed (both arrays endfire). They stay
  above `-tap_spacing / 2` only while these half sizes add up to less than `tap_spacing / 2` times the propagation
  speed: about 0.75 m for radio at 5 ns taps, but only about 3.6 mm for sound at 48 kHz taps, i.e. practically every
  acoustic array. In those cases use absolute delays (`use_absolute_delays = true`)
- Run time and memory depend on the number of paths and taps, not on the absolute delay values

### Usage:
```python
# Split real / imaginary output
coeff_re_q, coeff_im_q, delay_q = quadriga_lib.channel.quantize_delays( coeff_re, coeff_im, delay )

# Complex input and / or output (keyword)
coeff_q, delay_q = quadriga_lib.channel.quantize_delays( coeff=coeff, delay=delay, complex=True )
```

### Inputs:
- **`coeff_re`** — Real part of channel coefficients; list of length `n_snap`; each item
  `(n_rx, n_tx, n_path)`; must be paired with `coeff_im`; mutually exclusive with `coeff`;
  default: `None`
- **`coeff_im`** — Imaginary part of channel coefficients; same layout as `coeff_re`; default: `None`
- **`delay`** — Path delays in seconds; list of length `n_snap`; each item `(n_rx, n_tx, n_path)` or
  shared `(1, 1, n_path)`; non-negative with a tolerance of `tap_spacing / 2`, finite, below
  `2^31 * tap_spacing`
- **`tap_spacing`** — Delay bin spacing in seconds, positive and finite; 5 ns corresponds to 200 MHz
  sampling rate; default: 5e-9
- **`max_no_taps`** — Maximum number of output taps; 0 = unlimited; default: 48
- **`power_exponent`** — Interpolation exponent α; default: 1.0
- **`fix_taps`** — Delay grid sharing mode; 0 = per tx-rx pair and snapshot, 1 = single shared grid,
  2 = per snapshot, 3 = per tx-rx pair across all snapshots; default: 0
- **`stack`** — If `True`, stack snapshots into a 4D array; if `False`, return a list of per-snapshot
  arrays; default: `False`
- **`complex`** — If `True`, return combined complex coefficients `coeff_q`; if `False`, return
  separate `coeff_re_q` and `coeff_im_q`; default: `False`
- **`coeff`** — Complex channel coefficients; list of length `n_snap`; each item `(n_rx, n_tx, n_path)`;
  mutually exclusive with `coeff_re` / `coeff_im`; default: `None`

### Outputs:
- **`coeff_q`** — Combined complex output coefficients; returned when `complex` is `True`; list of
  `(n_rx, n_tx, n_taps)` when `stack` is `False`, else `(n_rx, n_tx, n_taps, n_snap)`
- **`coeff_re_q`** — Output coefficients, real part; returned when `complex` is `False`; list of
  `(n_rx, n_tx, n_taps)` when `stack` is `False`, else `(n_rx, n_tx, n_taps, n_snap)`
- **`coeff_im_q`** — Output coefficients, imaginary part; returned when `complex` is `False`; same
  shape as `coeff_re_q`
- **`delay_q`** — Output delays in seconds; list of `(n_rx, n_tx, n_taps)` or `(1, 1, n_taps)` when
  `stack` is `False`, else `(n_rx, n_tx, n_taps, n_snap)` or `(1, 1, n_taps, n_snap)`; the `(1, 1, ...)`
  shape only for shared delay input with `fix_taps` 1 or 2
