# Channel functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/mex_api.html#sec-channel-functions

| Function | Summary |
| --- | --- |
| [baseband_freq_response](#baseband_freq_response) | Compute the baseband frequency response of a MIMO channel |
| [channel_export_obj_file](#channel_export_obj_file) | Export propagation paths to a Wavefront OBJ file for 3D visualization |
| [hdf5_create_file](#hdf5_create_file) | Create a new HDF5 channel file with a custom storage layout |
| [hdf5_read_channel](#hdf5_read_channel) | Read one or more channel objects from an HDF5 file |
| [hdf5_read_dset](#hdf5_read_dset) | Read a single unstructured dataset from an HDF5 file |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Read names of unstructured datasets stored at a 4D slot in an HDF5 file |
| [hdf5_read_layout](#hdf5_read_layout) | Read the storage layout of channel data inside an HDF5 file |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Reshape the storage layout inside an existing HDF5 file |
| [hdf5_write_channel](#hdf5_write_channel) | Write one or more channel objects to an HDF5 file |
| [hdf5_write_dset](#hdf5_write_dset) | Write a single unstructured dataset to an HDF5 file |
| [qrt_file_parse](#qrt_file_parse) | Read metadata from a QRT file |
| [qrt_file_read](#qrt_file_read) | Read ray-tracing CIR data from a QRT file |
| [quantize_delays](#quantize_delays) | Map path delays to a fixed tap grid using two-tap power-weighted interpolation |

---
<a id="baseband_freq_response"></a>
## baseband_freq_response
Compute the baseband frequency response of a MIMO channel

Also in: [C++](../cpp/channel-functions.md#baseband_freq_response) · [Python](../python/channel-functions.md#baseband_freq_response)

- Three dispatch modes depending on the dimensionality of coeff_re / coeff_im / delay  and the
  presence of `i_snap` — see the table below
- Single-frequency mode: computes the frequency-domain channel matrix at the carrier positions via
  DFT over time-domain path coefficients and delays
- Multi-frequency mode: interpolates complex channel coefficients from the input frequency grid `center_freq` to the
  output grid using SLERP, then applies delay-induced phase rotation per output carrier; only the first frequency slab
  of `delay` is used. The delay-induced phase is removed from the input coefficients before the interpolation, so
  channels from [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq) / [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) must be generated with
  `use_absolute_delays = true`: their coefficients carry the phase of the absolute path length also when `delay` is
  relative, and with relative delays the phase of the TX-RX distance is interpolated across `center_freq`, which gives
  wrong phases (magnitudes stay right)
- Multi-snapshot mode: applies the single-frequency backend in parallel across snapshots via OpenMP;
  `i_snap` (1-based MATLAB indices) optionally selects a subset
- Carrier positions are specified one of two ways: as a normalized `pilot_grid` paired with `bandwidth`
  (where `0.0` corresponds to `center_freq(1)` and `1.0` to `center_freq(1) + bandwidth`), or as absolute
  frequencies via `carrier_freq`; supplying both pairs is an error
- Single-frequency and multi-snapshot modes with `carrier_freq` (same rules as the Python wrapper): the carriers may
  come in any order and the output keeps that order; the phase reference is `center_freq(1)` when given, else the
  lowest carrier; more than one carrier with a zero span (e.g. duplicates only) is an error
- Multi-frequency inputs always require `center_freq`; if `carrier_freq` is omitted it is derived as
  `center_freq(1) + pilot_grid · bandwidth`
- Multi-snapshot inputs require `center_freq` to be omitted or scalar (no combined multi-freq + multi-snap)
- Single-frequency inputs ignore `center_freq` when `pilot_grid` + `bandwidth` are given
- `delay` supports broadcasting: shape `[1, 1, n_path, ...]` applies the same delays to all RX/TX pairs
- Single-frequency and multi-snapshot modes: the phase `bandwidth · pilot_grid · delay` (in cycles) is computed in
  double precision and reduced to one period before sin / cos are evaluated in single precision; coefficients and sums
  are single precision; uses AVX2 where supported; double inputs are narrowed to float internally, results widened
  back. Multi-frequency mode computes in double precision

### Usage:
```matlab
[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( coeff_re, coeff_im, delay, ...
    pilot_grid, bandwidth, center_freq, carrier_freq, i_snap );
```

### Dispatch modes:
| Mode             | Triggered by                             | `coeff_re` shape                  | Output shape                       |
|------------------|------------------------------------------|-----------------------------------|------------------------------------|
| Single-frequency | 3D `coeff_re` and `i_snap` omitted       | `[n_rx, n_tx, n_path]`            | `[n_rx, n_tx, n_carrier]`          |
| Multi-frequency  | 4D `coeff_re` and `i_snap` omitted       | `[n_rx, n_tx, n_path, n_freq]`    | `[n_rx, n_tx, n_carrier]`          |
| Multi-snapshot   | `i_snap` supplied (may be 0 for all)     | `[n_rx, n_tx, n_path, n_snap]`    | `[n_rx, n_tx, n_carrier, n_out]`   |

### Inputs:
- **`coeff_re`** — Real part of channel coefficients; `[n_rx, n_tx, n_path]` (single-freq) or
  `[n_rx, n_tx, n_path, n_freq]` (multi-freq) or `[n_rx, n_tx, n_path, n_snap]` (multi-snap)
- **`coeff_im`** — Imaginary part of channel coefficients; same shape as `coeff_re`
- **`delay`** — Path delays in seconds; same shape as `coeff_re`, optionally broadcast over RX/TX with
  shape `[1, 1, n_path]` or `[1, 1, n_path, ...]`
- **`pilot_grid`** — Normalized sub-carrier positions; `0.0` = center, `1.0` = center + bandwidth;
  must be paired with `bandwidth`; `[n_carrier, 1]`
- **`bandwidth`** — Total baseband bandwidth in Hz; must be paired with `pilot_grid`
- **`center_freq`** — Input sample frequencies; required for multi-frequency inputs; length must equal
  the 4th dimension of `coeff_re`; for multi-snap must be omitted or scalar; with `carrier_freq` in single-frequency
  and multi-snap mode, `center_freq(1)` is the phase reference (default: the lowest carrier); `[n_freq, 1]`
- **`carrier_freq`** — Absolute output carrier frequencies in Hz; cannot be combined with
  `pilot_grid` + `bandwidth`; `[n_carrier, 1]`
- **`i_snap`** — Triggers multi-snap mode. Scalar `0` processes all snapshots; a positive vector of 1-based
  indices processes the selected subset. Omitting this argument or passing `[]` keeps the function in single/multi-frequency mode.

### Outputs:
- **`hmat_re`** — Real part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier]`
  (single/multi-freq) or `[n_rx, n_tx, n_carrier, n_out]` (multi-snap)
- **`hmat_im`** — Imaginary part of the frequency-domain channel matrix; same shape as `hmat_re`

### See also:
- [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (single-frequency channel generator)
- [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq) (multi-frequency channel generator)

---
<a id="channel_export_obj_file"></a>
## channel_export_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

Also in: [C++](../cpp/channel-class.md#.write_paths_to_obj_file) · [Python](../python/channel-functions.md#channel_export_obj_file)

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g., for use in Blender)
- Tubes are color-coded by path gain using a selected colormap; tube radius also scales with gain
- Paths below `gain_min` are excluded; `max_no_paths` limits the total number of exported paths
- The function takes raw channel data fields directly; no MATLAB channel struct is needed

### Usage:
```matlab
quadriga_lib.channel_export_obj_file( fn, max_no_paths, gain_max, gain_min, colormap, radius_max,  radius_min, ...
    n_edges, rx_pos, tx_pos, no_interact, interact_coord, center_freq, coeff_re, coeff_im, i_snap );
```

### Inputs:
- **`fn`** — Output `.obj` file path
- **`max_no_paths`** — Max paths to export; 0 includes all paths above `gain_min`; default: 0
- **`gain_max`** — Upper gain threshold in dB for color/radius mapping; higher values are clipped; default: -60.0
- **`gain_min`** — Lower gain threshold in dB; paths below this are excluded; default: -140.0
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`radius_max`** — Tube radius at maximum gain; default: 0.05
- **`radius_min`** — Tube radius at minimum gain; default: 0.01
- **`n_edges`** — Vertices per tube cross-section; must be >= 3; default: 5
- **`rx_pos`** — Receiver positions; `[3, n_snap]` or `[3, 1]`
- **`tx_pos`** — Transmitter positions; `[3, n_snap]` or `[3, 1]`
- **`no_interact`** — Number of interaction points of paths with the environment; uint32; `[n_path, n_snap]`
- **`interact_coord`** — Interaction coordinates; `[3, max(sum(no_interact)), n_snap]`; fewer than `sum(no_interact)`
  columns in a snapshot raise an error
- **`center_freq`** — Center frequency in Hz; `[n_snap]` or scalar
- **`coeff_re`** — Channel coefficients, real part; `[n_rx, n_tx, n_path, n_snap]`
- **`coeff_im`** — Channel coefficients, imaginary part; `[n_rx, n_tx, n_path, n_snap]`
- **`i_snap`** — Snapshot indices to include; 1-based; range [1 ... n_snap]; empty exports all

### Outputs:
- This function writes the OBJ file directly to disk and does not return any data

---
<a id="hdf5_create_file"></a>
## hdf5_create_file
Create a new HDF5 channel file with a custom storage layout

Also in: [C++](../cpp/channel-functions.md#hdf5_create) · [Python](../python/channel-functions.md#hdf5_create_file)

- Initializes a new HDF5 file for storing wireless channel data
- Defines a 4D layout `(nx, ny, nz, nw)` where each index combination maps to one channel storage slot
- Typical dimension mapping: nx = BS, ny = UE, nz = frequency, nw = scenario/repetition
- Storage layout is fixed at creation and cannot be altered later, except by reshaping while
  keeping the total slot count constant
- Errors if the target file already exists; delete it first to recreate it
- Files can be read by any HDF5 tool but must be modified only with quadriga-lib

### Usage:
```matlab
storage_space = quadriga_lib.hdf5_create_file( fn, storage_dims );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file to create; string
- **`storage_dims`** — Size of the storage layout; vector with 1-4 elements, i.e. `[nx]`, `[nx, ny]`, 
  `[nx, ny, nz]` or `[nx, ny, nz, nw]`; default: `[65536, 1, 1, 1]`

### Output:
- **`storage_space`** — Actual storage dimensions used; `[4]`; uint32

### See also:
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data)
- [hdf5_write_dset](#hdf5_write_dset) (for writing arbitrary unstructured data)

---
<a id="hdf5_read_channel"></a>
## hdf5_read_channel
Read one or more channel objects from an HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_read_channel) · [Python](../python/channel-functions.md#hdf5_read_channel)

- Reads structured channel data and any unstructured datasets from a 4D indexed HDF5 file
- Each of ix, iy, iz, iw may be a scalar, vector, or omitted (omitted/empty = read full extent along that dimension)
- Slots are visited in column-major order and empty slots are skipped.
- Structured fields are stored in single precision in the file and returned in double.
- Unstructured datasets keep their stored type and shape.
- If no data is found, both outputs are empty `0x0` structs.
- Errors if the file is not a quadriga-lib channel file of version 3: files written by quadriga-lib
  0.12.4 and earlier are rejected; files modified with other HDF5 tools are not supported and may be
  rejected.
- Files of quadriga-lib 0.12.4 and earlier can be converted with the Python script
  `tools/hdf5_convert_v2.py` of the quadriga-lib sources (needs Python with numpy, h5py and
  quadriga-lib), e.g. `system('python tools/hdf5_convert_v2.py old.hdf5 new.hdf5')`.

### Usage:
```matlab
[ chan, par ] = quadriga_lib.hdf5_read_channel( fn, ix, iy, iz, iw, snap );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`ix`** — 1-based slot indices along dimension X; scalar or vector; default: `1:nx`
- **`iy`** — 1-based slot indices along dimension Y; scalar or vector; default: `1:ny`
- **`iz`** — 1-based slot indices along dimension Z; scalar or vector; default: `1:nz`
- **`iw`** — 1-based slot indices along dimension W; scalar or vector; default: `1:nw`
- **`snap`** — Snapshot indices to read; 1-based; default: all snapshots. Only allowed
  when the total selection is a single slot.

### Outputs:
- **`chan`** — Struct array of length `N` (number of non-empty slots in the selection) with the channel data. Fields per element:
  | Field              | Description                                                              | Type / Size                         |
  | ------------------ | ------------------------------------------------------------------------ | ----------------------------------- |
  | `name`             | Channel name                                                             | String                              |
  | `tx_position`      | Transmitter positions (AP for downlink, STA for uplink)                  | `[3, 1]` or `[3, n_snap]`           |
  | `rx_position`      | Receiver positions (STA for downlink, AP for uplink)                     | `[3, 1]` or `[3, n_snap]`           |
  | `tx_orientation`   | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | `[3, 1]` or `[3, n_snap]`           |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | `[3, 1]` or `[3, n_snap]`           |
  | `coeff_re`         | Channel coefficients, real part                                          | `[n_rx, n_tx, n_path, n_snap]`      |
  | `coeff_im`         | Channel coefficients, imaginary part                                     | `[n_rx, n_tx, n_path, n_snap]`      |
  | `delay`            | Propagation delays in seconds                                            | `[n_rx, n_tx, n_path, n_snap]`      |
  | `path_gain`        | Path gain before antenna, linear scale                                   | `[n_path, n_snap]`                  |
  | `path_length`      | Path length in m                                                         | `[n_path, n_snap]`                  |
  | `path_polarization`| Polarization transfer function, interleaved complex                      | `[8, n_path, n_snap]`               |
  | `path_angles`      | Departure and arrival angles [AOD, EOD, AOA, EOA] in rad                 | `[n_path, 4, n_snap]`               |
  | `fbs_pos`          | First-bounce scatterer positions                                         | `[3, n_path, n_snap]`               |
  | `lbs_pos`          | Last-bounce scatterer positions                                          | `[3, n_path, n_snap]`               |
  | `no_interact`      | Number of interaction points per path; uint32                            | `[n_path, n_snap]`                  |
  | `interact_coord`   | Interaction coordinates                                                  | `[3, max(sum(no_interact)), n_snap]`|
  | `center_frequency` | Center Frequency in Hz                                                   | Scalar or `[n_snap]`                |
  | `initial_position` | Index of reference position; 1-based                                     | int32, scalar                       |
- **`par`** — Unstructured datasets as a `1xN` struct array matching `chan`. The field set is the
  **union** of dataset names across all loaded channels (without the `par_` prefix). For channels that
  do not contain a given field, the corresponding element is left empty. If no unstructured data is present anywhere,
  `par` is an empty `0x0` struct.

### See also:
- [hdf5_read_layout](#hdf5_read_layout) (for reading the layout in the file)
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data)
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)

---
<a id="hdf5_read_dset"></a>
## hdf5_read_dset
Read a single unstructured dataset from an HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_read_dset) · [Python](../python/channel-functions.md#hdf5_read_dset)

- Reads a user-defined dataset stored under `prefix + name` (e.g. `"par_carrier_frequency"`)
- Type and shape of the returned data are determined by the dataset's HDF5 dataspace
- Returns an empty `[]` matrix if the dataset does not exist at the requested slot
- Errors if the file does not exist or is not a quadriga-lib channel file of version 3 (see
  [hdf5_read_channel](#hdf5_read_channel))
- Supported types: string, scalar, vector (row or column), 2D matrix, and 3D array; numeric element 
  types: single, double, int32, uint32, int64, uint64

### Usage:
```matlab
dset = quadriga_lib.hdf5_read_dset( fn, location, name, prefix );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`location`** — Slot location inside the file; 1-based; vector with 1-4 elements, i.e. `[ix]`,
  `[ix, iy]`, `[ix, iy, iz]` or `[ix, iy, iz, iw]`; pass `[]` for default `[1, 1, 1, 1]`
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; string
- **`prefix`** — Prefix prepended to `name` when looking up the dataset; string; default: `'par_'`

### Outputs:
- **`dset`** — Dataset contents; type and shape are defined by the HDF5 dataspace; empty `[]` if the dataset is missing

### See also:
- [hdf5_read_dset_names](#hdf5_read_dset_names) (for reading names of already written datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)

---
<a id="hdf5_read_dset_names"></a>
## hdf5_read_dset_names
Read names of unstructured datasets stored at a 4D slot in an HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_read_dset_names) · [Python](../python/channel-functions.md#hdf5_read_dset_names)

- Finds all datasets whose HDF5 name starts with `prefix` at slot `(ix, iy, iz, iw)`
- Returned names have the prefix stripped
- Returns an empty cell array if no matching datasets are present at the slot
- Errors if the file does not exist or is not a quadriga-lib channel file of version 3 (see
  [hdf5_read_channel](#hdf5_read_channel))

### Usage:
```matlab
names = quadriga_lib.hdf5_read_dset_names( fn, location, prefix );
```

### Inputs:
- **`fn`** — Path to the HDF5 file; string
- **`location`** — Slot location inside the file; 1-based; vector with 1-4 elements, i.e. `[ix]`, `[ix, iy]`,
  `[ix, iy, iz]` or `[ix, iy, iz, iw]`; default:  `[1, 1, 1, 1]`
- **`prefix`** — Prefix used to identify unstructured datasets; string; default: `'par_'`

### Outputs:
- **`names`** — Names of all datasets at the given slot, with the prefix stripped; cell array of strings

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)
- [hdf5_write_dset](#hdf5_write_dset) (for writing individual unstructured datasets)

---
<a id="hdf5_read_layout"></a>
## hdf5_read_layout
Read the storage layout of channel data inside an HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_read_layout) · [Python](../python/channel-functions.md#hdf5_read_layout)

- Returns the dimensions of the 4D channel slot grid stored inside an HDF5 file
- Returns `[0, 0, 0, 0]` if the file does not exist; errors if the file exists but is not a
  quadriga-lib channel file of version 3 (files written by quadriga-lib 0.12.4 and earlier are
  rejected; files modified with other HDF5 tools are not supported and may be rejected)
- Also reports which slots already contain data, so callers can locate free slots without scanning the file

### Usage:
```matlab
[ storage_dims, has_data ] = quadriga_lib.hdf5_read_layout( fn );
```

### Input:
- **`fn`** — Filename of the HDF5 file; string

### Outputs:
- **`storage_dims`** — Size of the storage space `[nx, ny, nz, nw]`; `[4]`;  uint32
- **`has_data`** — Slot occupancy mask; `true` where data exists, `false` otherwise; `[nx, ny, nz, nw]`; logical

### See also:
- [hdf5_create_file](#hdf5_create_file) (for creating a file with a custom storage layout)
- [hdf5_reshape_layout](#hdf5_reshape_layout) (to change the layout later)

---
<a id="hdf5_reshape_layout"></a>
## hdf5_reshape_layout
Reshape the storage layout inside an existing HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_reshape_layout) · [Python](../python/channel-functions.md#hdf5_reshape_layout)

- Changes the 4D slot grid `(nx, ny, nz, nw)` of an existing HDF5 channel file
- The total number of slots (`nx · ny · nz · nw`) must match the original layout
- Only the dimension metadata is updated; stored channel data is not moved
- Errors if the file does not exist or is not a quadriga-lib channel file of version 3 (files
  written by quadriga-lib 0.12.4 and earlier are rejected; files modified with other HDF5 tools are
  not supported and may be rejected)
- One writer per file at a time; files are not locked

### Usage:
```matlab
storage_space = quadriga_lib.hdf5_reshape_layout( fn, storage_dims );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`storage_dims`** — New storage layout; vector with 1-4 elements,
  i.e. `[nx]`, `[nx, ny]`, `[nx, ny, nz]` or `[nx, ny, nz, nw]`; default: `[65536, 1, 1, 1]`

### Outputs:
- **`storage_space`** — New storage dimensions `[nx, ny, nz, nw]`; `[4]`; uint32

### See also:
- [hdf5_create_file](#hdf5_create_file) (for creating a file with a custom storage layout)
- [hdf5_read_layout](#hdf5_read_layout) (for reading the existing layout)

---
<a id="hdf5_write_channel"></a>
## hdf5_write_channel
Write one or more channel objects to an HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_write) · [Python](../python/channel-functions.md#hdf5_write_channel)

- Writes a struct array of channels into 4D slots (one slot per array element)
- Optional unstructured data can be passed as a matching struct array
- Creates the file with a sensible default layout if it does not yet exist; an existing file must be
  a quadriga-lib channel file of version 3 (files of quadriga-lib 0.12.4 and earlier are rejected,
  see [hdf5_read_channel](#hdf5_read_channel))
- A warning is issued if any selected slot already contains data. That slot is replaced completely:
  its old structured data and all its unstructured datasets (also those written with
  [hdf5_write_dset](#hdf5_write_dset)) are removed. The file does not shrink; copy the channels into a new file to
  reclaim the space of the removed data
- One writer per file at a time; files are not locked
- Structured data is stored in single precision regardless of MATLAB input precision
- Unstructured datasets retain their MATLAB type and shape (see [hdf5_write_dset](#hdf5_write_dset))
- Each scalar location input is broadcast to all `numel(chan)` channels; each vector input must have exactly `numel(chan)` elements
- If the file does not exist, it is created with layout `[max(numel(chan), max(ix)), max(iy), max(iz), max(iw)]`

### Usage:
```matlab
storage_dims = quadriga_lib.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`chan`** — Structured channel data; non-empty struct array; field layout matches [hdf5_read_channel](#hdf5_read_channel)
- **`par`** — Unstructured data; struct array of the same size as `chan`. Field names become HDF5 dataset
  names per slot (each prefixed with `par_`). Empty fields are skipped. Pass `[]` or omit to disable.
  Field values follow the type rules of [hdf5_write_dset](#hdf5_write_dset) (at most 3 dimensions).
- **`ix`** — 1-based slot index along dimension X; scalar or vector of length `numel(chan)`; default `1:numel(chan)`
- **`iy`** — 1-based slot index along dimension Y; scalar or vector of length `numel(chan)`; default `1`
- **`iz`** — 1-based slot index along dimension Z; scalar or vector of length `numel(chan)`; default `1`
- **`iw`** — 1-based slot index along dimension W; scalar or vector of length `numel(chan)`; default `1`

### Outputs:
- **`storage_dims`** — Storage layout dimensions of the file `[nx, ny, nz, nw]`; `[4]`; uint32

### See also:
- [hdf5_create_file](#hdf5_create_file) (for creating a file with a custom storage layout)
- [hdf5_reshape_layout](#hdf5_reshape_layout) (to change the layout later)
- [hdf5_read_channel](#hdf5_read_channel) (for reading channel data)

---
<a id="hdf5_write_dset"></a>
## hdf5_write_dset
Write a single unstructured dataset to an HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_write_dset) · [Python](../python/channel-functions.md#hdf5_write_dset)

- Dataset is stored under `prefix + name` at slot `(ix, iy, iz, iw)`
- `name` must contain only alphanumeric characters and underscores
- The file must already exist (use [hdf5_create_file](#hdf5_create_file) first) and be a quadriga-lib channel file of
  version 3 (see [hdf5_read_channel](#hdf5_read_channel))
- A dataset of the same name at the same slot is not overwritten; an error is thrown instead
- The dataset is added to the slot; everything else stored there is kept. A later [hdf5_write_channel](#hdf5_write_channel) to the
  same slot replaces the slot completely, including the datasets written with this function
- Supported types: string, scalar, vector (row or column), 2D matrix, and 3D array; numeric element
  types: single, double, int32, uint32, int64, uint64
- Arrays with more than 3 dimensions, sparse and complex arrays are rejected
- Row vectors are stored as given (`[1, n]`)
- One writer per file at a time; files are not locked

### Usage:
```matlab
storage_dims = quadriga_lib.hdf5_write_dset( fn, location, name, data, prefix );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`location`** — Slot location inside the file; 1-based; vector with 1-4 elements, i.e. `[ix]`,
  `[ix, iy]`, `[ix, iy, iz]` or `[ix, iy, iz, iw]`; pass `[]` for default `[1, 1, 1, 1]`
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; alphanumeric and underscores only; string
- **`data`** — Data to be written; type must be supported (see above); cannot be empty
- **`prefix`** — Prefix prepended to `name` in the HDF5 file; string; default: `'par_'`

### Outputs:
- **`storage_dims`** — Storage layout dimensions of the file `[nx, ny, nz, nw]`; `[4]`; uint32

### See also:
- [hdf5_read_dset_names](#hdf5_read_dset_names) (for reading names of already written datasets)
- [hdf5_read_dset](#hdf5_read_dset) (for reading individual unstructured datasets)

---
<a id="qrt_file_parse"></a>
## qrt_file_parse
Read metadata from a QRT file

Also in: [C++](../cpp/channel-functions.md#qrt_file_parse) · [Python](../python/channel-functions.md#qrt_file_parse)

- Parses a QRT file and extracts snapshot counts, origin/destination counts, frequency count,
  CIR offsets, names, positions, orientations, and file version
- When `no_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and
  `cir_offset` reflect this

### Usage:
```matlab
[ no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version, center_freq, ...
     cir_pos, cir_orientation, orig_pos, orig_orientation ] = quadriga_lib.qrt_file_parse( fn );
```

### Input:
- **`fn`** — Path to the QRT file; string

### Outputs:
- **`no_cir`** — Number of channel snapshots per origin point; uint64 scalar
- **`no_orig`** — Number of origin points (TX); uint64 scalar
- **`no_dest`** — Number of destination points (RX); uint64 scalar
- **`no_freq`** — Number of frequency bands; uint64 scalar
- **`cir_offset`** — CIR offset per destination; uint64; `[no_dest]`
- **`orig_names`** — Names of origin points; cell array of strings; `[no_orig]`
- **`dest_names`** — Names of destination points; cell array of strings; `[no_dest]`
- **`version`** — QRT file version number; int32 scalar
- **`center_freq`** — Frequencies as stored in the file; GHz for EM mode (v4/v5), Hz for scalar mode (v6); single; `[no_freq]`
- **`cir_pos`** — CIR positions in Cartesian coordinates; single; `[no_cir, 3]`
- **`cir_orientation`** — CIR orientations as Euler angles; single; `[no_cir, 3]`
- **`orig_pos`** — Origin (TX) positions in Cartesian coordinates; single; `[no_orig, 3]`
- **`orig_orientation`** — Origin (TX) orientations as Euler angles; single; `[no_orig, 3]`

---
<a id="qrt_file_read"></a>
## qrt_file_read
Read ray-tracing CIR data from a QRT file

Also in: [C++](../cpp/channel-functions.md#qrt_file_read) · [Python](../python/channel-functions.md#qrt_file_read)

- Reads channel impulse response data from QRT files
- All output arguments are optional; MATLAB only computes outputs that are requested
- If `downlink = true`, origin is TX and destination is RX; if `false`, roles are swapped

### Usage:
```matlab
[ center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, ...
    path_length, M, aod, eod, aoa, eoa, path_coord, no_int, coord, interact_type ] = ...
    quadriga_lib.qrt_file_read( fn, i_cir, i_orig, downlink, normalize_M );
```

### Inputs:
- **`fn`** — Path to the QRT file; string
- **`i_cir`** — Snapshot indices; 1-based; uint64; `[n_out]` or empty; default: `[]` (read all; for a file
  without CIRs, all outputs are empty with `n_out = 0`)
- **`i_orig`** — Origin index; 1-based; uint64; scalar; default: 1
- **`downlink`** — If `true`, origin=TX, destination=RX; if `false`, roles are swapped; logical scalar; default: `true`
- **`normalize_M`** — Controls `M` and `path_gain` scaling where PL is the propagation-only path loss
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm
    | `normalize_M` | `M`                   | `path_gain`                      |
    | :-----------: | :-------------------: | :------------------------------: |
    | 0             | As stored in QRT file | -PL                              |
    | 1             | Max column power = 1  | -PL minus material losses        |

    Other values are rejected with an error.

### Outputs:
- **`center_freq`** — Center frequency in Hz; `[n_freq]`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `[3, n_out]`
- **`tx_orientation`** — Transmitter orientation (bank, tilt, heading); `[3, n_out]`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `[3, n_out]`
- **`rx_orientation`** — Receiver orientation (bank, tilt, heading); `[3, n_out]`
- **`fbs_pos`** — First-bounce scatterer positions; Cell of length `n_out`; elements `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions;  Cell of length `n_out`; elements `[3, n_path]`
- **`path_gain`** — Path gain in linear scale; Cell of length `n_out`; elements `[n_path, n_freq]`
- **`path_length`** — Absolute path length TX to RX phase center; Cell of length `n_out`; elements `[n_path]`
- **`M`** — Polarization transfer matrix; Cell of length `n_out`;
  elements `[8, n_path, n_freq]` or `[2, n_path, n_freq]` for v6 files
- **`aod`** — Departure azimuth angles; Cell of length `n_out`; elements `[n_path]`
- **`eod`** — Departure elevation angles; Cell of length `n_out`; elements `[n_path]`
- **`aoa`** — Arrival azimuth angles; Cell of length `n_out`; elements `[n_path]`
- **`eoa`** — Arrival elevation angles; Cell of length `n_out`; elements `[n_path]`
- **`path_coord`** — Interaction coordinates per path; Cell of length `n_out`;
  elements Cell of length `n_path`, each `[3, n_interact + 2]`
- **`no_int`** — Number of mesh interactions per path; 0 indicates LOS; uint32;
  Cell of length `n_out`; elements `[n_path]`
- **`coord`** — Interaction coordinates (flat, concatenated across paths); single;
  Cell of length `n_out`; elements `[3, sum(no_int)]`
- **`interact_type`** — Interaction type codes (flat, concatenated across paths, matching `coord`); uint8;
  Cell of length `n_out`; elements `[sum(no_int)]`. Empty for v4 legacy files. GBSM paths written with the C++ API
  carry 129 (GR) and 193 (NLOS); LOS and GR paths are not marked as GBSM.

### See also:
- [arrayant_generate](array-antenna-functions.md#arrayant_generate) (for generating antenna arrays)
- [get_channels_planar](channel-generation-functions.md#get_channels_planar) (for embedding antennas using departure and arrival angles)
- [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (for embedding antennas using FBS/LBS positions)
- [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq) (for multi-frequency antenna embedding)

---
<a id="quantize_delays"></a>
## quantize_delays
Map path delays to a fixed tap grid using two-tap power-weighted interpolation

Also in: [C++](../cpp/channel-functions.md#quantize_delays) · [Python](../python/channel-functions.md#quantize_delays)

- Each path delay is approximated by two adjacent taps with coefficients scaled by (1−δ)^α and δ^α, 
  where δ is the fractional offset within the bin and α is `power_exponent`
- Two-tap interpolation avoids discontinuities when delays cross tap boundaries
- Use `power_exponent = 1.0` for narrowband (linear interpolation) or `0.5` for wideband (incoherent power preservation)
- If all fractional per-tap offsets are below 0.01 or above 0.99, weight computation is skipped 
  (nearest neighor selection) but tap-selection logic still applies
- Input `delay` may be per-antenna `[n_rx, n_tx, n_path, n_snap]` or shared `[1, 1, n_path, n_snap]`; 
  shared delays are expanded internally when `fix_taps` is 0 or 3
- Output arrays are zero-padded along the tap dimension so that all snapshots share the same `n_taps`
- Delays must be non-negative, as for a tapped delay line, with a tolerance of `tap_spacing / 2`: delays in
  `(-tap_spacing / 2, 0)` are treated as 0, so all of their power goes to tap 0. Delays at or below
  `-tap_spacing / 2` raise an error, as do NaN, Inf and delays of 2^31 taps or more (`delay / tap_spacing < 2^31`,
  e.g. 10.7 s at 5 ns or 12.4 h at 48 kHz)
- Relative delays (`use_absolute_delays = false` in [get_channels_spherical](channel-generation-functions.md#get_channels_spherical), [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq) and
  [get_channels_planar](channel-generation-functions.md#get_channels_planar)) are negative for element pairs that are closer to each other than the phase centres, by up
  to about (half the TX array size + half the RX array size) / propagation speed (both arrays endfire). They stay
  above `-tap_spacing / 2` only while these half sizes add up to less than `tap_spacing / 2` times the propagation
  speed: about 0.75 m for radio at 5 ns taps, but only about 3.6 mm for sound at 48 kHz taps, i.e. practically every
  acoustic array. In those cases use absolute delays (`use_absolute_delays = true`)
- Run time and memory depend on the number of paths and taps, not on the absolute delay values

### Usage:
```matlab
[ coeff_re_q, coeff_im_q, delay_q ] = quadriga_lib.quantize_delays( coeff_re, coeff_im, delay, ...
    tap_spacing, max_no_taps, power_exponent, fix_taps );
```

### Inputs:
- **`coeff_re`** — Channel coefficients, real part; `[n_rx, n_tx, n_path, n_snap]`
- **`coeff_im`** — Channel coefficients, imaginary part; `[n_rx, n_tx, n_path, n_snap]`
- **`delay`** — Path delays in seconds; `[n_rx, n_tx, n_path, n_snap]` or `[1, 1, n_path, n_snap]`; non-negative
  with a tolerance of `tap_spacing / 2`, finite, below `2^31 * tap_spacing`
- **`tap_spacing`** — Delay bin spacing in seconds, positive and finite; 5 ns corresponds to 200 MHz sampling rate; default: 5e-9
- **`max_no_taps`** — Maximum number of output taps; 0 = unlimited; default: 48
- **`power_exponent`** — Interpolation exponent α; default: 1.0
- **`fix_taps`** — Delay grid sharing mode; default: 0
  | Value | Meaning                                                                                         |
  | :---: | ----------------------------------------------------------------------------------------------- |
  | 0     | Per tx-rx pair and snapshot; output delays `[n_rx, n_tx, n_taps, n_snap]`                       |
  | 1     | Single shared grid across all snapshots and tx-rx pairs; output delays `[1, 1, n_taps, n_snap]` for shared delay input, `[n_rx, n_tx, n_taps, n_snap]` for per-antenna input |
  | 2     | Per snapshot; output delays `[1, 1, n_taps, n_snap]` for shared delay input, `[n_rx, n_tx, n_taps, n_snap]` for per-antenna input; each snapshot independent |
  | 3     | Per tx-rx pair across all snapshots; output delays `[n_rx, n_tx, n_taps, n_snap]`               |

### Outputs:
- **`coeff_re_q`** — Output coefficients, real part; `[n_rx, n_tx, n_taps, n_snap]`
- **`coeff_im_q`** — Output coefficients, imaginary part; `[n_rx, n_tx, n_taps, n_snap]`
- **`delay_q`** — Output delays in seconds; `[n_rx, n_tx, n_taps, n_snap]`, or `[1, 1, n_taps, n_snap]` for shared delay
  input with `fix_taps` 1 or 2
