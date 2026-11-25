# Channel functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/mex_api.html#sec-channel-functions

| Function | Summary |
| --- | --- |
| [baseband_freq_response](#baseband_freq_response) | Compute the baseband frequency response of a MIMO channel |
| [baseband_freq_response_multi](#baseband_freq_response_multi) | Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients |
| [channel_export_obj_file](#channel_export_obj_file) | Export propagation paths to a Wavefront OBJ file for 3D visualization |
| [crossfade_segments](#crossfade_segments) | Joins the channels of consecutive segments of one link by a power cross-fade |
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

- Computes the frequency-domain channel matrix at the carrier positions `pilot_grid` by a DFT over the time-domain
  path coefficients and delays
- One snapshot is a 3D array; a batch of snapshots is a cell array of such arrays (`coeff_re`, `coeff_im` and `delay`
  cell arrays of the same length), processed in parallel; the number of paths may differ between snapshots
- A channel struct (e.g. from [hdf5_read_channel](#hdf5_read_channel)) holds its snapshots on the 4th dimension, zero-padded;
  `num2cell(ch.coeff_re, [1 2 3])`, and the same for `coeff_im` and `delay`, gives the cell arrays (the zero-padded
  paths do not change the result)
- Multi-frequency coefficients (e.g. from [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq)) and absolute carrier frequencies are handled by
  [baseband_freq_response_multi](#baseband_freq_response_multi); the two functions take different arguments, so they are two functions, while
  [interpolate_pattern](antennas-transducers-and-patterns.md#interpolate_pattern) serves both of its cases with one argument set
- `delay` supports broadcasting: shape `[1, 1, n_path]` applies the same delays to all RX/TX pairs
- The phase `bandwidth · pilot_grid · delay` (in cycles) is computed in double precision and reduced to one period
  before sin / cos are evaluated in single precision; coefficients and sums are single precision; double inputs are
  narrowed to float internally, results widened back

### Usage:
```matlab
[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( coeff_re, coeff_im, delay, pilot_grid, bandwidth, ...
    i_snap, use_kernel );
```

### Inputs:
- **`coeff_re`** — Real part of the channel coefficients; `[n_rx, n_tx, n_path]`, or a cell array of length `n_snap`
  with one such array per snapshot
- **`coeff_im`** — Imaginary part of the channel coefficients; same layout as `coeff_re`
- **`delay`** — Path delays in seconds; same layout as `coeff_re`, each array `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`
- **`pilot_grid`** — Sub-carrier positions relative to the bandwidth, `0.0` = carrier, `1.0` = carrier + `bandwidth`;
  `[n_carrier, 1]`
- **`bandwidth`** — Baseband bandwidth in Hz
- **`i_snap`** — Snapshots to process, 1-based; `[n_out, 1]`; default: `[]` (all snapshots)
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (error if unavailable); default: 0

### Outputs:
- **`hmat_re`** — Real part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier, n_out]`
- **`hmat_im`** — Imaginary part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier, n_out]`

### See also:
- [baseband_freq_response_multi](#baseband_freq_response_multi) (multi-frequency coefficients, absolute carrier frequencies)
- [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (single-frequency channel generator)

---
<a id="baseband_freq_response_multi"></a>
## baseband_freq_response_multi
Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients

Also in: [C++](../cpp/channel-functions.md#baseband_freq_response_multi) · [Python](../python/channel-functions.md#baseband_freq_response_multi)

- Interpolates the complex channel coefficients from the input frequencies `freq` to the output carrier
  frequencies `freq_new` using SLERP (magnitude and unwrapped phase each linear), then applies the delay-induced
  phase `exp(-j·2π·freq_new·delay)` per output carrier in double precision
- One snapshot is a 4D array `[n_rx, n_tx, n_path, n_freq]`; a batch of snapshots is a cell array of such arrays
  (`coeff_re`, `coeff_im` and `delay` cell arrays of the same length), processed in parallel; the number of paths may
  differ between snapshots
- Single-frequency coefficients with absolute output carriers: `n_freq = 1` (a 3D array), `freq` the frequency the
  coefficients were generated at; with `remove_delay_phase = true` (the default) and coefficients that carry the delay
  phase at `freq`, as the channel generators return them, this gives the result of [baseband_freq_response](#baseband_freq_response) with
  `pilot_grid · bandwidth = freq_new - freq`
- `delay` is frequency-independent: `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`; of a 4D `delay` (as returned by
  [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq)) only the first frequency is used
- With `remove_delay_phase`, the channels from [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq) / [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) must be
  generated with `use_absolute_delays = true`: their coefficients carry the phase of the absolute path length also
  when `delay` is relative, and with relative delays the phase of the TX-RX distance is interpolated across `freq`,
  which gives wrong phases (magnitudes stay right)
- Output frequencies outside the range of `freq` use constant extrapolation from the nearest endpoint

### Usage:
```matlab
[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response_multi( coeff_re, coeff_im, delay, freq, freq_new, ...
    remove_delay_phase, i_snap, use_kernel );
```

### Inputs:
- **`coeff_re`** — Real part of the channel coefficients; `[n_rx, n_tx, n_path, n_freq]`, or a cell array of length
  `n_snap` with one such array per snapshot
- **`coeff_im`** — Imaginary part of the channel coefficients; same layout as `coeff_re`
- **`delay`** — Path delays in seconds; an array `[n_rx, n_tx, n_path]` or `[1, 1, n_path]` (a 4th dimension of length
  `n_freq` is allowed, only the first frequency is used), or a cell array of length `n_snap` with one such array per
  snapshot
- **`freq`** — Input sample frequencies in Hz, sorted ascending; `[n_freq, 1]`
- **`freq_new`** — Output carrier frequencies in Hz; `[n_carrier, 1]`
- **`remove_delay_phase`** — Remove the delay phase `exp(-j·2π·freq·delay)` from the coefficients before the
  interpolation and apply it again at `freq_new`; `true` for the output of the channel generators, `false` for pure
  envelope coefficients; default: `true`
- **`i_snap`** — Snapshots to process, 1-based; `[n_out, 1]`; default: `[]` (all snapshots)
- **`use_kernel`** — Kernel of the sin / cos and SLERP evaluation: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (error if unavailable); default: 0

### Outputs:
- **`hmat_re`** — Real part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier, n_out]`
- **`hmat_im`** — Imaginary part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier, n_out]`

### See also:
- [baseband_freq_response](#baseband_freq_response) (single-frequency coefficients, relative carrier positions)
- [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq) (multi-frequency channel generator)

---
<a id="channel_export_obj_file"></a>
## channel_export_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

Also in: [C++](../cpp/channel-functions.md#channel_export_obj_file) · [Python](../python/channel-functions.md#channel.export_obj_file)

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g., for use in Blender)
- Tubes are color-coded by path gain using a selected colormap; tube radius also scales with gain
- Paths below `gain_min_dB` are excluded; `max_paths` limits the total number of exported paths
- Paths with non-finite coordinates are skipped

### Usage:
```matlab
quadriga_lib.channel_export_obj_file( fn, chan, max_paths, gain_max_dB, gain_min_dB, colormap, i_snap, ...
    radius_max, radius_min, n_edges );
```

### Inputs:
- **`fn`** — Output `.obj` file path
- **`chan`** — Channel; 1x1 struct with the fields of [hdf5_read_channel](#hdf5_read_channel); needs `rx_pos`, `tx_pos`,
  `center_freq`, `n_interact`, `interact_coord` and `coeff_re` / `coeff_im` with `delay`, or `polmat`;
  columns of `interact_coord` beyond `sum(n_interact)` are ignored
- **`max_paths`** — Max paths to export; 0 includes all paths above `gain_min_dB`; default: 0
- **`gain_max_dB`** — Upper gain threshold in dB for color/radius mapping; higher values are clipped; default: -60.0
- **`gain_min_dB`** — Lower gain threshold in dB; paths below this are excluded; default: -140.0
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`i_snap`** — Snapshot indices to include; 1-based; range [1 ... n_snap]; default: `[]` (all snapshots)
- **`radius_max`** — Tube radius at maximum gain; default: 0.05
- **`radius_min`** — Tube radius at minimum gain; default: 0.01
- **`n_edges`** — Vertices per tube cross-section; must be >= 3; default: 5

---
<a id="crossfade_segments"></a>
## crossfade_segments
Joins the channels of consecutive segments of one link by a power cross-fade

Also in: [C++](../cpp/channel-functions.md#crossfade_segments) · [Python](../python/channel-functions.md#crossfade_segments)

- Joins the channels of the segments of one link, e.g. of a track whose scenario changes, into one channel; consecutive
  segments share snapshots, over which the paths of the old segment fade out and those of the new one fade in
- [get_gbsm_channels](network-layouts-and-mobility.md#get_gbsm_channels) joins the segments of tracks with this function; it is a port of the channel merger of QuaDRiGa

### Usage:
```matlab
channel = quadriga_lib.crossfade_segments( segments, n_overlap, overlap, par );
```

### Inputs:
- **`segments`** — Channels of one link in track order, valid, with coefficients, equal `n_rx`, `n_tx`, form and
  `center_freq` (at most one value); struct array, fields in [hdf5_read_channel](#hdf5_read_channel); `[1, n_segment]`
- **`n_overlap`** — Snapshots shared with the preceding segment, entry 1 is 0, at most the snapshots of the preceding
  segment after its own shared ones; `[n_segment]`
- **`overlap`** — Part of the shared snapshots used for the ramp, in [0, 1]; default: 0.5
- **`par`** — Unstructured data of the segments with the fields `tap` (tap index per path or tap, uint32), `path_type`
  (0 = LOS, 1 = GR, 2 = NLOS per path or tap, uint32, optional) and `n_ray_tap` (rays per tap, `'taps'` only); struct
  array of the size of `segments` that has the field `tap` or `path_type` (other fields are ignored); `[]` = no tap
  indices, all paths are NLOS; default: `[]`

### Outputs:
- **`channel`** — The joined channel, 1x1 struct, fields in [hdf5_read_channel](#hdf5_read_channel); `par` is not carried

### Notes:
- **Snapshots**: segment k starts with `n_overlap(k)` snapshots that are also the last ones of segment k - 1; the output
  has `sum(n_snap(k) - n_overlap(k))` snapshots, summed over k
- **Ramp**: in the last `n_r = round(overlap · n_overlap(k))` shared snapshots, the old paths fade out with `1 - w(i)`
  and the new ones fade in with `w(i)`, in power, with `w(i) = 0.5 · (1 + sin((i / (n_r + 1) - 0.5) · π))` for i = 1,
  ..., n_r. Before the ramp, the snapshots are those of the old segment
- **LOS and GR** (types 0 and 1 of the entry `path_type` of `par`; without it, tap index 0 of the entry `tap` and tap
  index 3 held by one ray, without index 4 or 5, as [ssf_rays](geometry-based-stochastic-model.md#ssf_rays) numbers them): one path with the data of the new
  segment (of the old one where the new one has no power), scaled to
  `abs(h)^2 = (1 - w) · abs(h_old)^2 + w · abs(h_new)^2` by the path gain (rays) or the tap power (taps)
- **Rays** (channels with `path_gain`): the NLOS paths of both segments side by side, with the amplitudes `sqrt(1 - w)`
  and `sqrt(w)`; `path_length`, `path_angles`, `fbs_pos` and `lbs_pos` are kept where all segments have them
- **Taps** (without `path_gain`, one tap count per segment): one set of tap slots for all snapshots (LOS, GR, NLOS); the
  NLOS taps of the old segment hold `1 - w` and those of the new segment `w` of their mean power over the ramp; slots
  without power hold zero coefficients and the smallest delay of the snapshot
- **Other data**: delays as given (one reference for all segments); positions interpolated in the ramp, orientations
  along the shorter way; `name` and `center_freq` of segment 1; `n_interact` and `interact_coord` are not carried
- **Tap indices**: the C++ channels carry them in their `par_data`; in MATLAB they are the input `par`, as in
  [hdf5_write_channel](#hdf5_write_channel): the entries `tap`, `path_type` and `n_ray_tap` of each segment, as returned by
  [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel)

### See also:
- [get_gbsm_channels](network-layouts-and-mobility.md#get_gbsm_channels) (joins the segments of tracks with this function)
- [gbsm_single_link_channel](geometry-based-stochastic-model.md#gbsm_single_link_channel) (channels of one segment)

---
<a id="hdf5_create_file"></a>
## hdf5_create_file
Create a new HDF5 channel file with a custom storage layout

Also in: [C++](../cpp/channel-functions.md#hdf5_create_file) · [Python](../python/channel-functions.md#hdf5_create_file)

- Initializes a new HDF5 file for storing wireless channel data
- Defines a 4D layout `(nx, ny, nz, nw)` where each index combination maps to one channel storage slot
- Typical dimension mapping: nx = BS, ny = UE, nz = frequency, nw = scenario/repetition
- Storage layout is fixed at creation and cannot be altered later, except by reshaping while
  keeping the total slot count constant
- Errors if the target file already exists (also one that cannot be read); delete it first to recreate it
- Files can be read by any HDF5 tool but must be modified only with quadriga-lib

### Usage:
```matlab
quadriga_lib.hdf5_create_file( fn, nx, ny, nz, nw );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file to create; string
- **`nx`** — Number of slots along dimension X; default: 65536
- **`ny`** — Number of slots along dimension Y; default: 1
- **`nz`** — Number of slots along dimension Z; default: 1
- **`nw`** — Number of slots along dimension W; default: 1

### See also:
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data)
- [hdf5_write_dset](#hdf5_write_dset) (for writing arbitrary unstructured data)
- [hdf5_read_layout](#hdf5_read_layout) (for reading the layout)

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
- Errors if the file does not exist or is not a quadriga-lib channel file; files modified with other HDF5 tools
  are not supported and may be rejected.

### Usage:
```matlab
[ chan, par ] = quadriga_lib.hdf5_read_channel( fn, ix, iy, iz, iw, i_snap );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`ix`** — 1-based slot indices along dimension X; scalar or vector; default: `1:nx`
- **`iy`** — 1-based slot indices along dimension Y; scalar or vector; default: `1:ny`
- **`iz`** — 1-based slot indices along dimension Z; scalar or vector; default: `1:nz`
- **`iw`** — 1-based slot indices along dimension W; scalar or vector; default: `1:nw`
- **`i_snap`** — Snapshot indices to read; 1-based; default: all snapshots. Applies to every
  selected slot; errors if an index exceeds the number of snapshots of a selected slot.

### Outputs:
- **`chan`** — Struct array of length `N` (number of non-empty slots in the selection) with the channel data. Fields per element:
  | Field              | Description                                                              | Type / Size                         |
  | ------------------ | ------------------------------------------------------------------------ | ----------------------------------- |
  | `name`             | Channel name                                                             | String                              |
  | `tx_pos`           | Transmitter positions (AP for downlink, STA for uplink)                  | `[3, 1]` or `[3, n_snap]`           |
  | `rx_pos`           | Receiver positions (STA for downlink, AP for uplink)                     | `[3, 1]` or `[3, n_snap]`           |
  | `tx_orientation`   | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | `[3, 1]` or `[3, n_snap]`           |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | `[3, 1]` or `[3, n_snap]`           |
  | `coeff_re`         | Channel coefficients, real part                                          | `[n_rx, n_tx, n_path, n_snap]`      |
  | `coeff_im`         | Channel coefficients, imaginary part                                     | `[n_rx, n_tx, n_path, n_snap]`      |
  | `delay`            | Propagation delays in seconds                                            | `[n_rx, n_tx, n_path, n_snap]`      |
  | `path_gain`        | Path gain before antenna, linear scale                                   | `[n_path, n_snap]`                  |
  | `path_length`      | Path length in m                                                         | `[n_path, n_snap]`                  |
  | `polmat`           | Polarization transfer matrix, interleaved complex                        | `[8, n_path, n_snap]`               |
  | `path_angles`      | Departure and arrival angles [AOD, EOD, AOA, EOA] in rad                 | `[n_path, 4, n_snap]`               |
  | `fbs_pos`          | First-bounce scatterer positions                                         | `[3, n_path, n_snap]`               |
  | `lbs_pos`          | Last-bounce scatterer positions                                          | `[3, n_path, n_snap]`               |
  | `n_interact`       | Number of interaction points per path; uint32                            | `[n_path, n_snap]`                  |
  | `interact_coord`   | Interaction coordinates                                                  | `[3, max(sum(n_interact)), n_snap]`|
  | `center_freq`      | Center Frequency in Hz                                                   | Scalar or `[n_snap]`                |
  | `initial_position` | Reference snapshot (start of the current segment); 1-based; absent = 1   | int32, scalar                       |
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
- Errors if the file does not exist or is not a quadriga-lib channel file
- Supported types: string, scalar, vector (row or column), 2D matrix, and 3D array; numeric element 
  types: single, double, int32, uint32, int64, uint64

### Usage:
```matlab
dset = quadriga_lib.hdf5_read_dset( fn, name, ix, iy, iz, iw, prefix );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; string
- **`ix`** — 1-based slot index along dimension X; scalar; default: 1
- **`iy`** — 1-based slot index along dimension Y; scalar; default: 1
- **`iz`** — 1-based slot index along dimension Z; scalar; default: 1
- **`iw`** — 1-based slot index along dimension W; scalar; default: 1
- **`prefix`** — Prefix prepended to `name` when looking up the dataset; string; `''` looks up `name` as given; default (omitted or `[]`): `'par_'`

### Output:
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
- Errors if the file does not exist or is not a quadriga-lib channel file

### Usage:
```matlab
names = quadriga_lib.hdf5_read_dset_names( fn, ix, iy, iz, iw, prefix );
```

### Inputs:
- **`fn`** — Path to the HDF5 file; string
- **`ix`** — 1-based slot index along dimension X; scalar; default: 1
- **`iy`** — 1-based slot index along dimension Y; scalar; default: 1
- **`iz`** — 1-based slot index along dimension Z; scalar; default: 1
- **`iw`** — 1-based slot index along dimension W; scalar; default: 1
- **`prefix`** — Prefix used to identify unstructured datasets; string; `''` lists all datasets with their full names; default (omitted or `[]`): `'par_'`

### Output:
- **`names`** — Names of all datasets at the given slot, with the prefix stripped; cell array of strings; `[n_names, 1]`

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
  quadriga-lib channel file
- Also reports which slots already contain data, so callers can locate free slots without scanning the file

### Usage:
```matlab
[ storage_dims, has_data ] = quadriga_lib.hdf5_read_layout( fn );
```

### Input:
- **`fn`** — Filename of the HDF5 file; string

### Outputs:
- **`storage_dims`** — Size of the storage space `[nx, ny, nz, nw]`; uint32; `[1, 4]`
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
- Errors if the file does not exist or is not a quadriga-lib channel file
- One writer per file at a time; files are not locked

### Usage:
```matlab
quadriga_lib.hdf5_reshape_layout( fn, nx, ny, nz, nw );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`nx`** — New number of slots along dimension X
- **`ny`** — New number of slots along dimension Y; default: 1
- **`nz`** — New number of slots along dimension Z; default: 1
- **`nw`** — New number of slots along dimension W; default: 1

### See also:
- [hdf5_create_file](#hdf5_create_file) (for creating a file with a custom storage layout)
- [hdf5_read_layout](#hdf5_read_layout) (for reading the existing layout)

---
<a id="hdf5_write_channel"></a>
## hdf5_write_channel
Write one or more channel objects to an HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_write_channel) · [Python](../python/channel-functions.md#hdf5_write_channel)

- Writes a struct array of channels into 4D slots (one slot per array element)
- Optional unstructured data can be passed as a matching struct array
- Creates the file if it does not exist, with the layout that just holds all slots (largest index per dimension);
  use [hdf5_create_file](#hdf5_create_file) for another layout
- All inputs are checked before anything is written: every slot inside the layout of an existing file, no slot
  twice, all channels valid
- A slot that already contains data is replaced completely: its old structured data and all its unstructured
  datasets (also those written with [hdf5_write_dset](#hdf5_write_dset)) are removed; `code` shows which slots were replaced. The
  file does not shrink; copy the channels into a new file to reclaim the space of the removed data
- One writer per file at a time; files are not locked
- Structured data is stored in single precision regardless of MATLAB input precision
- Unstructured datasets retain their MATLAB type and shape (see [hdf5_write_dset](#hdf5_write_dset))
- Each scalar location input is broadcast to all `numel(chan)` channels; each vector input must have exactly `numel(chan)` elements

### Usage:
```matlab
code = quadriga_lib.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`chan`** — Structured channel data; non-empty struct array; field layout matches [hdf5_read_channel](#hdf5_read_channel)
- **`par`** — Unstructured data; struct array of the same size as `chan`. Field names become HDF5 dataset
  names per slot (each prefixed with `par_`). Empty fields are skipped; default: `[]` (none).
  Field values follow the type rules of [hdf5_write_dset](#hdf5_write_dset) (at most 3 dimensions).
- **`ix`** — 1-based slot index along dimension X; scalar or vector of length `numel(chan)`; default: `1:numel(chan)`
- **`iy`** — 1-based slot index along dimension Y; scalar or vector of length `numel(chan)`; default: 1
- **`iz`** — 1-based slot index along dimension Z; scalar or vector of length `numel(chan)`; default: 1
- **`iw`** — 1-based slot index along dimension W; scalar or vector of length `numel(chan)`; default: 1

### Output:
- **`code`** — Per channel: `0` if the slot was empty, `1` if existing data was replaced; `[numel(chan), 1]`; int32

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
- `name` must not be empty and must contain only alphanumeric characters and underscores
- Creates the file with the layout `[ix, iy, iz, iw]` if it does not exist; use [hdf5_create_file](#hdf5_create_file) for another
  layout
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
quadriga_lib.hdf5_write_dset( fn, name, data, ix, iy, iz, iw, prefix );
```

### Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; alphanumeric and underscores only; string
- **`data`** — Data to be written; type must be supported (see above); cannot be empty
- **`ix`** — 1-based slot index along dimension X; scalar; default: 1
- **`iy`** — 1-based slot index along dimension Y; scalar; default: 1
- **`iz`** — 1-based slot index along dimension Z; scalar; default: 1
- **`iw`** — 1-based slot index along dimension W; scalar; default: 1
- **`prefix`** — Prefix prepended to `name` in the HDF5 file; string; `''` stores `name` without a prefix; default (omitted or `[]`): `'par_'`

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
- When `n_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and
  `cir_offset` reflect this

### Usage:
```matlab
[ n_cir, n_orig, n_dest, n_freq, cir_offset, orig_names, dest_names, version, center_freq, ...
     cir_pos, cir_orientation, orig_pos, orig_orientation ] = quadriga_lib.qrt_file_parse( fn );
```

### Input:
- **`fn`** — Path to the QRT file; string

### Outputs:
- **`n_cir`** — Number of channel snapshots per origin point; uint64 scalar
- **`n_orig`** — Number of origin points (TX); uint64 scalar
- **`n_dest`** — Number of destination points (RX); uint64 scalar
- **`n_freq`** — Number of frequency bands; uint64 scalar
- **`cir_offset`** — Number of CIRs before each destination (an offset, not an index; the first entry is 0); the CIRs of
  destination `d` start at `i_cir = cir_offset(d) + 1`; uint64; `[n_dest, 1]`
- **`orig_names`** — Names of origin points; cell array of strings; `[n_orig, 1]`
- **`dest_names`** — Names of destination points; cell array of strings; `[n_dest, 1]`
- **`version`** — QRT file version number; int32 scalar
- **`center_freq`** — Center frequencies in Hz, converted from GHz for files that store GHz (v4, and v5 when all stored values are below 1000); single; `[n_freq, 1]`
- **`cir_pos`** — CIR positions in Cartesian coordinates; single; `[n_cir, 3]`
- **`cir_orientation`** — CIR orientations as Euler angles; single; `[n_cir, 3]`
- **`orig_pos`** — Origin (TX) positions in Cartesian coordinates; single; `[n_orig, 3]`
- **`orig_orientation`** — Origin (TX) orientations as Euler angles; single; `[n_orig, 3]`

---
<a id="qrt_file_read"></a>
## qrt_file_read
Read ray-tracing CIR data from a QRT file

Also in: [C++](../cpp/channel-functions.md#qrt_file_read) · [Python](../python/channel-functions.md#qrt_file_read)

- Reads channel impulse response data from QRT files
- All output arguments are optional; MATLAB only computes outputs that are requested
- Reads in chunks of 16 snapshots; besides the output, at most two chunks are held in memory
- If `uplink = false`, origin is TX and destination is RX; if `true`, roles are swapped and `polmat` is the transpose
  of the downlink `polmat` (reciprocal channel, no conjugation) with the same `path_gain`; `n_interact`, `coord` and
  `interact_type` keep the stored order (origin to destination) in both directions, and the type codes describe
  that traversal

### Usage:
```matlab
[ center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, ...
    path_length, polmat, aod, eod, aoa, eoa, path_coord, n_interact, coord, interact_type ] = ...
    quadriga_lib.qrt_file_read( fn, i_cir, i_orig, uplink, normalize_polmat );
```

### Inputs:
- **`fn`** — Path to the QRT file; string
- **`i_cir`** — Snapshot indices; 1-based; `[n_out]` or empty; default: `[]` (read all; for a file
  without CIRs, all outputs are empty with `n_out = 0`)
- **`i_orig`** — Origin index; 1-based; scalar; default: 1
- **`uplink`** — If `false`, origin=TX, destination=RX; if `true`, roles are swapped (origin=RX, destination=TX); logical scalar; default: `false`
- **`normalize_polmat`** — Controls `polmat` and `path_gain` scaling where PL is the propagation-only path loss; default: 1
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm
    | `normalize_polmat` | `polmat`                                   | `path_gain`                      |
    | :-----------: | :---------------------------------------------: | :------------------------------: |
    | 0             | As stored in QRT file                           | -PL                              |
    | 1             | Max column power = 1; uplink: max row power = 1 | -PL minus material losses        |

    Other values are rejected with an error.

### Outputs:
- **`center_freq`** — Center frequency in Hz; `[n_freq, 1]`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `[3, n_out]`
- **`tx_orientation`** — Transmitter orientation (bank, tilt, heading); `[3, n_out]`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `[3, n_out]`
- **`rx_orientation`** — Receiver orientation (bank, tilt, heading); `[3, n_out]`
- **`fbs_pos`** — First-bounce scatterer positions; Cell of length `n_out`; elements `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; Cell of length `n_out`; elements `[3, n_path]`
- **`path_gain`** — Path gain in linear scale; Cell of length `n_out`; elements `[n_path, n_freq]`
- **`path_length`** — Absolute path length TX to RX phase center; Cell of length `n_out`; elements `[n_path, 1]`
- **`polmat`** — Polarization transfer matrix; Cell of length `n_out`;
  elements `[8, n_path, n_freq]` or `[2, n_path, n_freq]` for v6 files
- **`aod`** — Departure azimuth angles; Cell of length `n_out`; elements `[n_path, 1]`
- **`eod`** — Departure elevation angles; Cell of length `n_out`; elements `[n_path, 1]`
- **`aoa`** — Arrival azimuth angles; Cell of length `n_out`; elements `[n_path, 1]`
- **`eoa`** — Arrival elevation angles; Cell of length `n_out`; elements `[n_path, 1]`
- **`path_coord`** — Interaction coordinates per path; Cell of length `n_out`;
  elements Cell of length `n_path`, each `[3, n_interact + 2]`
- **`n_interact`** — Number of mesh interactions per path; 0 indicates LOS; uint32;
  Cell of length `n_out`; elements `[n_path, 1]`
- **`coord`** — Interaction coordinates (flat, concatenated across paths); single;
  Cell of length `n_out`; elements `[3, sum(n_interact)]`
- **`interact_type`** — Interaction type codes (flat, concatenated across paths, matching `coord`); uint8;
  Cell of length `n_out`; elements `[sum(n_interact), 1]`. Empty for v4 legacy files. GBSM paths written with the C++ API
  carry 129 (GR) and 193 (NLOS); LOS and GR paths are not marked as GBSM.

### See also:
- [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) (for generating antenna arrays)
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
  (nearest neighbor selection) but tap-selection logic still applies
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
    tap_spacing, max_taps, power_exponent, fix_taps );
```

### Inputs:
- **`coeff_re`** — Channel coefficients, real part; `[n_rx, n_tx, n_path, n_snap]`
- **`coeff_im`** — Channel coefficients, imaginary part; `[n_rx, n_tx, n_path, n_snap]`
- **`delay`** — Path delays in seconds; `[n_rx, n_tx, n_path, n_snap]` or `[1, 1, n_path, n_snap]`; non-negative
  with a tolerance of `tap_spacing / 2`, finite, below `2^31 * tap_spacing`
- **`tap_spacing`** — Delay bin spacing in seconds, positive and finite; 5 ns corresponds to 200 MHz sampling rate; default: 5e-9
- **`max_taps`** — Maximum number of output taps; 0 = unlimited; default: 48
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
