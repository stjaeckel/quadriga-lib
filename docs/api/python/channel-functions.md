# Channel functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/python_api.html#sec-channel-functions

| Function | Summary |
| --- | --- |
| [baseband_freq_response](#baseband_freq_response) | Compute the baseband frequency response of a MIMO channel |
| [baseband_freq_response_multi](#baseband_freq_response_multi) | Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients |
| [channel.export_obj_file](#channel.export_obj_file) | Export propagation paths to a Wavefront OBJ file for 3D visualization |
| [crossfade_segments](#crossfade_segments) | Joins the channels of consecutive segments of one link by a power cross-fade |
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

- Computes the frequency-domain channel matrix at the carrier positions `pilot_grid` by a DFT over the time-domain
  path coefficients and delays
- Snapshots are passed as Python lists; each list item is one snapshot, processed in parallel; the number of paths
  may differ between snapshots, `n_rx` and `n_tx` may not
- Multi-frequency coefficients (e.g. from [channel.multifreq](channel-generation-functions.md#channel.multifreq)) and absolute carrier frequencies are handled by
  [baseband_freq_response_multi](#baseband_freq_response_multi); the two functions take different arguments, so they are two functions,
  while [interpolate_pattern](antennas-transducers-and-patterns.md#interpolate_pattern) serves both of its cases with one argument set
- `delay` may be broadcast: a `(1, 1, n_path)` item applies the same delays to every RX/TX pair
- The phase `bandwidth · pilot_grid · delay` (in cycles) is computed in double precision and reduced to one period
  before sin / cos are evaluated in single precision; coefficients and sums are single precision, double inputs are
  narrowed to float, results widened back

### Usage:
```python
hmat = quadriga_lib.channel.baseband_freq_response( coeff, delay, pilot_grid, bandwidth, *, i_snap, use_kernel )
```

### Inputs:
- **`coeff`** — Complex channel coefficients; list of length `n_snap`; each item `(n_rx, n_tx, n_path)`
- **`delay`** — Path delays in seconds; list of length `n_snap`; each item `(n_rx, n_tx, n_path)` or `(1, 1, n_path)`
- **`pilot_grid`** — Sub-carrier positions relative to the bandwidth, `0.0` = carrier, `1.0` = carrier + `bandwidth`;
  1D array of length `n_carrier`; `np.linspace(0, 1, n)` gives `n` evenly spaced carriers
- **`bandwidth`** — Baseband bandwidth in Hz
- **`i_snap`** — 0-based snapshot indices to process; 1D array of length `n_out`; default: `None` (all snapshots)
- **`use_kernel`** — Kernel: 0 = auto (AVX2 if available, else GENERIC), 1 = GENERIC, 2 = AVX2 (raises if unavailable); default: 0

### Output:
- **`hmat`** — Complex frequency-domain channel matrix; shape `(n_rx, n_tx, n_carrier, n_out)`

### See also:
- [baseband_freq_response_multi](#baseband_freq_response_multi) (multi-frequency coefficients, absolute carrier frequencies)
- [channel.spherical](channel-generation-functions.md#channel.spherical) (single-frequency channel generator)

---
<a id="baseband_freq_response_multi"></a>
## baseband_freq_response_multi
Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients

Also in: [C++](../cpp/channel-functions.md#baseband_freq_response_multi) · [MATLAB](../mex/channel-functions.md#baseband_freq_response_multi)

- Interpolates the complex channel coefficients from the input frequencies `freq` to the output carrier
  frequencies `freq_new` using SLERP (magnitude and unwrapped phase each linear), then applies the delay-induced
  phase `exp(-j·2π·freq_new·delay)` per output carrier in double precision
- Snapshots are passed as Python lists; each list item is one snapshot, processed in parallel; the number of paths
  may differ between snapshots, `n_rx` and `n_tx` may not
- Single-frequency coefficients with absolute output carriers: items `(n_rx, n_tx, n_path)`, `freq` the frequency
  the coefficients were generated at; with `remove_delay_phase = True` (the default) and coefficients that carry the
  delay phase at `freq`, as the channel generators return them, this gives the result of [baseband_freq_response](#baseband_freq_response)
  with `pilot_grid · bandwidth = freq_new - freq`
- `delay` is frequency-independent: items `(n_rx, n_tx, n_path)` or `(1, 1, n_path)`; of a 4D item (as returned by
  [channel.multifreq](channel-generation-functions.md#channel.multifreq)) only the first frequency is used
- With `remove_delay_phase`, the channels from [channel.multifreq](channel-generation-functions.md#channel.multifreq) / [channel.spherical](channel-generation-functions.md#channel.spherical) must be generated with
  `use_absolute_delays = True`: their coefficients carry the phase of the absolute path length also when `delay` is
  relative, and with relative delays the phase of the TX-RX distance is interpolated across `freq`, which gives
  wrong phases (magnitudes stay right)
- Output frequencies outside the range of `freq` use constant extrapolation from the nearest endpoint

### Usage:
```python
hmat = quadriga_lib.channel.baseband_freq_response_multi( coeff, delay, freq, freq_new, *, remove_delay_phase, \
    i_snap, use_kernel )
```

### Inputs:
- **`coeff`** — Complex channel coefficients; list of length `n_snap`; each item `(n_rx, n_tx, n_path, n_freq)`; for
  `n_freq = 1` also `(n_rx, n_tx, n_path)`
- **`delay`** — Path delays in seconds; list of length `n_snap`; each item `(n_rx, n_tx, n_path)` or `(1, 1, n_path)`,
  with an optional 4th dimension of length `n_freq` (only the first frequency is used)
- **`freq`** — Input sample frequencies in Hz, sorted ascending; 1D array of length `n_freq`
- **`freq_new`** — Output carrier frequencies in Hz; 1D array of length `n_carrier`
- **`remove_delay_phase`** — Remove the delay phase `exp(-j·2π·freq·delay)` from the coefficients before the
  interpolation and apply it again at `freq_new`; `True` for the output of the channel generators, `False` for pure
  envelope coefficients; default: `True`
- **`i_snap`** — 0-based snapshot indices to process; 1D array of length `n_out`; default: `None` (all snapshots)
- **`use_kernel`** — Kernel of the sin / cos and SLERP evaluation: 0 = auto (AVX2 if available, else GENERIC),
  1 = GENERIC, 2 = AVX2 (raises if unavailable); default: 0

### Output:
- **`hmat`** — Complex frequency-domain channel matrix; shape `(n_rx, n_tx, n_carrier, n_out)`

### See also:
- [baseband_freq_response](#baseband_freq_response) (single-frequency coefficients, relative carrier positions)
- [channel.multifreq](channel-generation-functions.md#channel.multifreq) (multi-frequency channel generator)

---
<a id="channel.export_obj_file"></a>
## channel.export_obj_file
Export propagation paths to a Wavefront OBJ file for 3D visualization

Also in: [C++](../cpp/channel-functions.md#channel_export_obj_file) · [MATLAB](../mex/channel-functions.md#channel_export_obj_file)

- Writes ray-traced paths as tube geometry to a `.obj` file (e.g. for use in Blender)
- Tubes are color-coded by path gain using a selected colormap; tube radius also scales with gain
- Paths below `gain_min_dB` are excluded; `max_paths` caps the total number of exported paths
- Paths with non-finite coordinates are skipped

### Usage:
```python
quadriga_lib.channel.export_obj_file( fn, chan, *, max_paths, gain_max_dB, gain_min_dB, colormap, i_snap, \
    radius_max, radius_min, n_edges )
```

### Inputs:
- **`fn`** — Output `.obj` file path
- **`chan`** — Channel; dict with the keys of [hdf5_read_channel](#hdf5_read_channel), list or stacked layout; needs `rx_pos`,
  `tx_pos`, `center_freq`, `n_interact`, `interact_coord` and `coeff` with `delay`, or `polmat`; in the stacked
  layout, columns of `interact_coord` beyond `sum(n_interact)` are ignored
- **`max_paths`** — Max paths to export; 0 includes all paths above `gain_min_dB`; default: 0
- **`gain_max_dB`** — Upper gain threshold in dB for color/radius mapping; higher values are clipped; default: -60.0
- **`gain_min_dB`** — Lower gain threshold in dB; paths below this are excluded; default: -140.0
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`i_snap`** — Snapshot indices to include; 0-based; `None` or empty exports all; default: `None`
- **`radius_max`** — Tube radius at maximum gain; default: 0.05
- **`radius_min`** — Tube radius at minimum gain; default: 0.01
- **`n_edges`** — Vertices per tube cross-section; must be >= 3; default: 5

---
<a id="crossfade_segments"></a>
## crossfade_segments
Joins the channels of consecutive segments of one link by a power cross-fade

Also in: [C++](../cpp/channel-functions.md#crossfade_segments) · [MATLAB](../mex/channel-functions.md#crossfade_segments)

- Joins the channels of the segments of one link, e.g. of a track whose scenario changes, into one channel; consecutive
  segments share snapshots, over which the paths of the old segment fade out and those of the new one fade in
- [get_gbsm_channels](network-layouts-and-mobility.md#get_gbsm_channels) joins the segments of tracks with this function; it is a port of the channel merger of QuaDRiGa

### Usage:
```python
channel = quadriga_lib.channel.crossfade_segments( segments, n_overlap, *, overlap, par, stack )
```

### Inputs:
- **`segments`** — Channels of one link in track order, valid, with coefficients, equal `n_rx`, `n_tx`, form and
  `center_freq` (at most one value); list of channel dicts, keys in [hdf5_read_channel](#hdf5_read_channel)
- **`n_overlap`** — Snapshots shared with the preceding segment, the first entry is 0, at most the snapshots of the
  preceding segment after its own shared ones; `(n_segment,)`
- **`overlap`** — Part of the shared snapshots used for the ramp, in [0, 1]; default: 0.5
- **`par`** — Unstructured data of the segments: list of dicts, one per segment, with the keys `tap` (tap index per path
  or tap, uint32), `path_type` (0 = LOS, 1 = GR, 2 = NLOS per path or tap, uint32, optional) and `n_ray_tap` (rays per
  tap, `"taps"` only); a list that is given must have the key `tap` or `path_type` (other keys are ignored); `None` = no
  tap indices, all paths are NLOS; default: None
- **`stack`** — Snapshot layout of the channel as in [hdf5_read_channel](#hdf5_read_channel); default: `False`

### Outputs:
- **`channel`** — The joined channel, dict, keys in [hdf5_read_channel](#hdf5_read_channel)

### Notes:
- **Snapshots**: segment $k$ starts with `n_overlap[k]` snapshots that are also the last ones of segment $k - 1$; the
  output has $\sum_k (n_{\mathrm{snap},k} - n_{\mathrm{overlap},k})$ snapshots
- **Ramp**: in the last $n_r = \mathrm{round}(\mathrm{overlap} \cdot n_{\mathrm{overlap},k})$ shared snapshots, the old
  paths fade out with $1 - w_i$ and the new ones fade in with $w_i$, in power, with

  $$w_i = \frac{1}{2} \left( 1 + \sin\left( \left( \frac{i}{n_r + 1} - \frac{1}{2} \right) \pi \right) \right), \quad i = 1, \dots, n_r$$

  Before the ramp, the snapshots are those of the old segment
- **LOS and GR** (types 0 and 1 of the entry `path_type` of `par`; without it, tap index 0 of the entry `tap` and tap
  index 3 held by one ray, without index 4 or 5, as [ssf_rays](geometry-based-stochastic-model.md#ssf_rays) numbers them): one path with the data of the new
  segment (of the old one where the new one has no power), scaled to
  $\lvert h \rvert^2 = (1 - w) \lvert h_\mathrm{old} \rvert^2 + w \lvert h_\mathrm{new} \rvert^2$ by the path gain
  (rays) or the tap power (taps)
- **Rays** (channels with `path_gain`): the NLOS paths of both segments side by side, with the amplitudes $\sqrt{1 - w}$
  and $\sqrt{w}$; `path_length`, `path_angles`, `fbs_pos` and `lbs_pos` are kept where all segments have them
- **Taps** (without `path_gain`, one tap count per segment): one set of tap slots for all snapshots (LOS, GR, NLOS); the
  NLOS taps of the old segment hold $1 - w$ and those of the new segment $w$ of their mean power over the ramp; slots
  without power hold zero coefficients and the smallest delay of the snapshot
- **Other data**: delays as given (one reference for all segments); positions interpolated in the ramp, orientations
  along the shorter way; `name` and `center_freq` of segment 0; `par`, `n_interact` and `interact_coord` are not carried
- **`par`**: the C++ channels carry the tap indices in their `par_data`; in Python they are the input `par`, as in
  [hdf5_write_channel](#hdf5_write_channel), with the entries of each segment as [gbsm.single_link_channel](geometry-based-stochastic-model.md#gbsm.single_link_channel) returns them

### See also:
- [get_gbsm_channels](network-layouts-and-mobility.md#get_gbsm_channels) (joins the segments of tracks with this function)
- [gbsm.single_link_channel](geometry-based-stochastic-model.md#gbsm.single_link_channel) (channels of one segment)

---
<a id="hdf5_create_file"></a>
## hdf5_create_file
Create a new HDF5 channel file with a custom storage layout

Also in: [C++](../cpp/channel-functions.md#hdf5_create_file) · [MATLAB](../mex/channel-functions.md#hdf5_create_file)

- Initializes a new HDF5 file for storing channel data
- Defines a 4D storage layout `(nx, ny, nz, nw)`; each index combination maps to one channel slot
- Typical dimension mapping: nx = BS, ny = UE, nz = frequency, nw = scenario/repetition
- The total slot count is fixed at creation; the layout can be reshaped with [hdf5_reshape_layout](#hdf5_reshape_layout)
- Raises an error if the target file already exists (also one that cannot be read); delete it first to recreate it
- Files can be read by any HDF5 tool but must be modified only with quadriga-lib

### Usage:
```python
quadriga_lib.channel.hdf5_create_file( fn, nx, ny, nz, nw )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file to create
- **`nx`** — Number of elements on the x-dimension; default: 65536
- **`ny`** — Number of elements on the y-dimension; default: 1
- **`nz`** — Number of elements on the z-dimension; default: 1
- **`nw`** — Number of elements on the w-dimension; default: 1

### See also:
- [hdf5_write_channel](#hdf5_write_channel) (for writing channel data)
- [hdf5_write_dset](#hdf5_write_dset) (for writing arbitrary unstructured data)
- [hdf5_read_layout](#hdf5_read_layout) (for reading the layout)

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
- `i_snap` selects a subset of snapshots; if `None`, all snapshots are read
- Errors if the file does not exist or is not a quadriga-lib channel file; files modified with other HDF5 tools are
  not supported and may be rejected.

### Usage:
```python
chan, par = quadriga_lib.channel.hdf5_read_channel( fn, ix, iy, iz, iw, *, i_snap, stack )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file
- **`ix`** — 0-based slot indices along dimension X; scalar or vector; default: `0 ... nx-1`
- **`iy`** — 0-based slot indices along dimension Y; scalar or vector; default: `0 ... ny-1`
- **`iz`** — 0-based slot indices along dimension Z; scalar or vector; default: `0 ... nz-1`
- **`iw`** — 0-based slot indices along dimension W; scalar or vector; default: `0 ... nw-1`
- **`i_snap`** — Snapshot indices to read; 0-based; default: all snapshots. Applies to every selected slot; raises an error if an index exceeds the number of snapshots of a selected slot.
- **`stack`** — If `True`, stack snapshots; default: `False` (return as list)

### Outputs:
- **`chan`** — List of dicts with the following keys:
  | Key                | Description                                                              | Shape `stack = False`             | Shape `stack = True`                 |
  | ------------------ | ------------------------------------------------------------------------ | --------------------------------- | ------------------------------------ |
  | `name`             | Channel name                                                             | str                               | str                                  |
  | `tx_pos`           | Transmitter positions (AP for downlink, STA for uplink)                  | `(3, n_snap)` or `(3, 1)`         | `(3, n_snap)` or `(3, 1)`            |
  | `rx_pos`           | Receiver positions (STA for downlink, AP for uplink)                     | `(3, n_snap)` or `(3, 1)`         | `(3, n_snap)` or `(3, 1)`            |
  | `tx_orientation`   | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | `(3, n_snap)` or `(3, 1)`         | `(3, n_snap)` or `(3, 1)`            |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | `(3, n_snap)` or `(3, 1)`         | `(3, n_snap)` or `(3, 1)`            |
  | `coeff`            | Channel coefficients, complex-valued                                     | list of `(n_rx, n_tx, n_path_s)`  | `(n_rx, n_tx, n_path, n_snap)`       |
  | `delay`            | Propagation delays in seconds                                            | list of `(n_rx, n_tx, n_path_s)`  | `(n_rx, n_tx, n_path, n_snap)`       |
  | `path_gain`        | Path gain before antenna, linear scale                                   | list of `(n_path_s,)`             | `(n_path, n_snap)`                   |
  | `path_length`      | Path length in m                                                         | list of `(n_path_s,)`             | `(n_path, n_snap)`                   |
  | `polmat`           | Polarization transfer matrix, real, rows as in [channel.planar](channel-generation-functions.md#channel.planar)        | list of `(8, n_path_s)`           | `(8, n_path, n_snap)`                |
  | `path_angles`      | Departure and arrival angles (AOD, EOD, AOA, EOA) in rad                 | list of `(n_path_s, 4)`           | `(n_path, 4, n_snap)`                |
  | `fbs_pos`          | First-bounce scatterer positions                                         | list of `(3, n_path_s)`           | `(3, n_path, n_snap)`                |
  | `lbs_pos`          | Last-bounce scatterer positions                                          | list of `(3, n_path_s)`           | `(3, n_path, n_snap)`                |
  | `n_interact`       | Number of interaction points per path; int64                             | list of `(n_path_s,)`             | `(n_path, n_snap)`                   |
  | `interact_coord`   | Interaction coordinates                                                  | list of `(3, sum(n_interact_s))`  | `(3, max(sum(n_interact)), n_snap)`  |
  | `center_freq`      | Center Frequency in Hz                                                   | `(n_snap,)` or `(1,)`             | `(n_snap,)` or `(1,)`                |
  | `initial_position` | Reference snapshot (start of the current segment); 0-based               | int                               | int                                  |

- **`par`** — List of dicts with the unstructured data, one per entry of `chan` (empty dict if the slot has none).

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
- Errors if the file does not exist or is not a quadriga-lib channel file
- Supported types: str, scalar, vector, 2D array, and 3D array

### Usage:
```python
dset = quadriga_lib.channel.hdf5_read_dset( fn, name, ix, iy, iz, iw, *, prefix )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file
- **`name`** — Dataset name without the prefix, e.g. `'carrier_frequency'`; str
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`prefix`** — Prefix prepended to `name` when looking up the dataset; str; default: `'par_'`

### Output:
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
- Errors if the file does not exist or is not a quadriga-lib channel file

### Usage:
```python
names = quadriga_lib.channel.hdf5_read_dset_names( fn, ix, iy, iz, iw, *, prefix )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`prefix`** — Prefix used to identify unstructured datasets; `''` lists all datasets; str; default: `'par_'`

### Output:
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
- Raises an error if the file exists but is not a quadriga-lib channel file

### Usage:
```python
storage_dims, has_data = quadriga_lib.channel.hdf5_read_layout( fn )
```

### Input:
- **`fn`** — Filename of the HDF5 file

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
- Errors if the file does not exist or is not a quadriga-lib channel file
- One writer per file at a time; files are not locked

### Usage:
```python
quadriga_lib.channel.hdf5_reshape_layout( fn, nx, ny, nz, nw )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file
- **`nx`** — Number of elements on the x-dimension
- **`ny`** — Number of elements on the y-dimension; default: 1
- **`nz`** — Number of elements on the z-dimension; default: 1
- **`nw`** — Number of elements on the w-dimension; default: 1

---
<a id="hdf5_write_channel"></a>
## hdf5_write_channel
Write one or more channel objects to an HDF5 file

Also in: [C++](../cpp/channel-functions.md#hdf5_write_channel) · [MATLAB](../mex/channel-functions.md#hdf5_write_channel)

- Writes a list of channel dicts into 4D storage slots (one slot per list entry)
- `chan` may also be a single dict, in which case one channel is written
- Optional unstructured data (`par`) can be passed as a matching list of dicts (or a single dict)
- Creates the file if it does not exist, with the layout that just holds all slots (largest index + 1 per
  dimension); use [hdf5_create_file](#hdf5_create_file) for another layout
- All inputs are checked before anything is written: every slot inside the layout of an existing file, no slot
  twice, all channels valid
- A slot that already contains data is replaced completely: its old structured data and all its unstructured
  datasets (also those written with [hdf5_write_dset](#hdf5_write_dset)) are removed; `code` shows which slots were replaced.
  The file does not shrink; copy the channels into a new file to reclaim the space of the removed data
- One writer per file at a time; files are not locked
- Structured data is stored in single precision regardless of the input precision
- Unstructured datasets retain their numpy dtype and shape; supported types as in [hdf5_write_dset](#hdf5_write_dset)
- Each scalar index input is broadcast to all channels; each vector index must have one entry per channel
- Channel dict field layout matches [hdf5_read_channel](#hdf5_read_channel)
- Per-snapshot fields accept the list model (list of arrays) or the stack model
  (single array with the snapshot index as the last axis), detected per field
- Coefficients may be passed as a complex `coeff`, or as separate real `coeff_re` and `coeff_im` (the two forms are mutually exclusive)
- Slot indices are 0-based

### Usage:
```python
code = quadriga_lib.channel.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file
- **`chan`** — Structured channel data; a channel dict or a list of channel dicts; field layout matches [hdf5_read_channel](#hdf5_read_channel)
- **`par`** — Unstructured data; a dict or a list of dicts with the same number of entries as `chan`. Dict keys become HDF5 dataset
  names per slot; `None` values are skipped. Pass `None` to disable; default: `None`
- **`ix`** — 0-based slot indices along dimension X; scalar or vector of length `len(chan)`; default: `0 ... len(chan)-1`
- **`iy`** — 0-based slot indices along dimension Y; scalar or vector of length `len(chan)`; default: 0
- **`iz`** — 0-based slot indices along dimension Z; scalar or vector of length `len(chan)`; default: 0
- **`iw`** — 0-based slot indices along dimension W; scalar or vector of length `len(chan)`; default: 0

### Output:
- **`code`** — Per channel: `0` if the slot was empty, `1` if existing data was replaced; `(len(chan),)`; int64

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
- `name` must not be empty; `prefix + name` must contain only alphanumeric characters and underscores
- Creates the file with the layout `(ix + 1, iy + 1, iz + 1, iw + 1)` if it does not exist; use [hdf5_create_file](#hdf5_create_file)
  for another layout
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
quadriga_lib.channel.hdf5_write_dset( fn, name, data, ix, iy, iz, iw, *, prefix )
```

### Inputs:
- **`fn`** — Filename of the HDF5 file
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; alphanumeric and underscores only; str
- **`data`** — Data to be written; type must be supported (see above); cannot be empty
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
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
- When `n_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and
  `cir_offset` reflect this

### Usage:
```python
n_cir, n_orig, n_dest, n_freq, cir_offset, orig_names, dest_names, version, center_freq, \
    cir_pos, cir_orientation, orig_pos, orig_orientation = quadriga_lib.channel.qrt_file_parse( fn )
```

### Input:
- **`fn`** — Path to the QRT file

### Outputs:
- **`n_cir`** — Number of channel snapshots per origin point
- **`n_orig`** — Number of origin points (TX)
- **`n_dest`** — Number of destination points (RX)
- **`n_freq`** — Number of frequency bands
- **`cir_offset`** — Number of CIRs before each destination (offset of its first CIR), the first entry 0; int64; `(n_dest,)`
- **`orig_names`** — Names of the origin points (TX); list of strings; length `n_orig`
- **`dest_names`** — Names of the destination points (RX); list of strings; length `n_dest`
- **`version`** — QRT file version number
- **`center_freq`** — Center frequencies in Hz, converted from GHz for files that store GHz (v4, and v5 when all stored values are below 1000); `(n_freq,)`
- **`cir_pos`** — CIR positions in Cartesian coordinates; `(n_cir, 3)`
- **`cir_orientation`** — CIR orientations as Euler angles; `(n_cir, 3)`
- **`orig_pos`** — Origin (TX) positions in Cartesian coordinates; `(n_orig, 3)`
- **`orig_orientation`** — Origin (TX) orientations as Euler angles; `(n_orig, 3)`

---
<a id="qrt_file_read"></a>
## qrt_file_read
Read ray-tracing CIR data from a QRT file

Also in: [C++](../cpp/channel-functions.md#qrt_file_read) · [MATLAB](../mex/channel-functions.md#qrt_file_read)

- Reads channel impulse response data from QRT files
- Reads in chunks of 16 snapshots; besides the output, at most two chunks are held in memory
- If `uplink = False`, origin is TX and destination is RX; if `True`, roles are swapped and `polmat` is the transpose
  of the downlink `polmat` (reciprocal channel, no conjugation) with the same `path_gain`; `n_interact`, `coord` and
  `interact_type` keep the stored order (origin to destination) in both directions, and the type codes describe
  that traversal
- Per-snapshot outputs are returned as lists with one entry per requested snapshot

### Usage:
```python
center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, path_gain, \
    path_length, polmat, aod, eod, aoa, eoa, path_coord, n_interact, coord, \
    interact_type = quadriga_lib.channel.qrt_file_read( fn, i_cir, i_orig, *, uplink, normalize_polmat )
```

### Inputs:
- **`fn`** — Path to the QRT file
- **`i_cir`** — Snapshot indices to read, 0-based; `(n_out,)` or `None`; if `None` or empty, all snapshots are read (for a file
  without CIRs, all outputs are empty with `n_out = 0`); default: `None`
- **`i_orig`** — Origin index, 0-based (origin = TX for downlink); scalar; default: 0
- **`uplink`** — If `False`, origin=TX and destination=RX; if `True`, the roles are swapped (origin=RX, destination=TX); default: `False`
- **`normalize_polmat`** — Controls `polmat` and `path_gain` scaling; 0 = values as stored in the QRT file with
  `path_gain` = -PL; 1 = `polmat` columns (uplink: rows) scaled to max power 1 and `path_gain` = -PL minus material losses; default: 1
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm
    | `normalize_polmat` | `polmat`                                   | `path_gain`                      |
    | :-----------: | :---------------------------------------------: | :------------------------------: |
    | 0             | As stored in QRT file                           | -PL                              |
    | 1             | Max column power = 1; uplink: max row power = 1 | -PL minus material losses        |

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
- **`polmat`** — Polarization transfer matrix, stored as interleaved real/imaginary pairs; list of length `n_out`; entries `(8, n_path, n_freq)`, or `(2, n_path, n_freq)` for v6 files
- **`aod`** — Departure azimuth angles; list of length `n_out`; entries `(n_path,)`
- **`eod`** — Departure elevation angles; list of length `n_out`; entries `(n_path,)`
- **`aoa`** — Arrival azimuth angles; list of length `n_out`; entries `(n_path,)`
- **`eoa`** — Arrival elevation angles; list of length `n_out`; entries `(n_path,)`
- **`path_coord`** — Interaction coordinates per path; list of length `n_out`; each entry is a list of length `n_path` with arrays `(3, n_interact + 2)`
- **`n_interact`** — Number of mesh interactions per path (0 indicates LOS); list of length `n_out`; entries int64 `(n_path,)`
- **`coord`** — Interaction coordinates concatenated across paths; list of length `n_out`; entries `(3, sum(n_interact))`
- **`interact_type`** — Interaction type codes concatenated across paths (matching `coord`); list of length `n_out`;
  entries uint8 `(sum(n_interact),)`. Empty for v4 legacy files. GBSM paths written with the C++ API carry 129 (GR) and
  193 (NLOS); LOS and GR paths are not marked as GBSM.

### See also:
- [generate_antenna](antennas-transducers-and-patterns.md#generate_antenna) (for generating antenna arrays)
- [channel.planar](channel-generation-functions.md#channel.planar) (for embedding antennas using departure and arrival angles)
- [channel.spherical](channel-generation-functions.md#channel.spherical) (for embedding antennas using FBS/LBS positions)
- [channel.multifreq](channel-generation-functions.md#channel.multifreq) (for multi-frequency antenna embedding)

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
- Input `delay` items may be per-antenna `(n_rx, n_tx, n_path)` or shared `(1, 1, n_path)`;
  shared delays are expanded internally when `fix_taps` is 0 or 3
- Output arrays are zero-padded along the tap dimension so that all snapshots share the same `n_taps`
- Delays must be non-negative, as for a tapped delay line, with a tolerance of `tap_spacing / 2`: delays in
  `(-tap_spacing / 2, 0)` are treated as 0, so all of their power goes to tap 0. Delays at or below
  `-tap_spacing / 2` raise `ValueError`, as do NaN, Inf and delays of 2^31 taps or more (`delay / tap_spacing < 2^31`,
  e.g. 10.7 s at 5 ns or 12.4 h at 48 kHz)
- Relative delays (`use_absolute_delays = false` in [channel.spherical](channel-generation-functions.md#channel.spherical), [channel.multifreq](channel-generation-functions.md#channel.multifreq) and
  [channel.planar](channel-generation-functions.md#channel.planar)) are negative for element pairs that are closer to each other than the phase centres, by up
  to about (half the TX array size + half the RX array size) / propagation speed (both arrays endfire). They stay
  above `-tap_spacing / 2` only while these half sizes add up to less than `tap_spacing / 2` times the propagation
  speed: about 0.75 m for radio at 5 ns taps, but only about 3.6 mm for sound at 48 kHz taps, i.e. practically every
  acoustic array. In those cases use absolute delays (`use_absolute_delays = true`)
- Run time and memory depend on the number of paths and taps, not on the absolute delay values

### Usage:
```python
coeff_q, delay_q = quadriga_lib.channel.quantize_delays( coeff, delay, *, tap_spacing, max_taps, \
    power_exponent, fix_taps, stack )
```

### Inputs:
- **`coeff`** — Complex channel coefficients; list of length `n_snap`; each item `(n_rx, n_tx, n_path)`
- **`delay`** — Path delays in seconds; list of length `n_snap`; each item `(n_rx, n_tx, n_path)` or
  shared `(1, 1, n_path)`; non-negative with a tolerance of `tap_spacing / 2`, finite, below
  `2^31 * tap_spacing`
- **`tap_spacing`** — Delay bin spacing in seconds, positive and finite; 5 ns corresponds to 200 MHz
  sampling rate; default: 5e-9
- **`max_taps`** — Maximum number of output taps; 0 = unlimited; default: 48
- **`power_exponent`** — Interpolation exponent α; default: 1.0
- **`fix_taps`** — Delay grid sharing mode; 0 = per tx-rx pair and snapshot, 1 = single shared grid,
  2 = per snapshot, 3 = per tx-rx pair across all snapshots; default: 0
- **`stack`** — If `True`, stack snapshots into a 4D array; if `False`, return a list of per-snapshot
  arrays; default: `False`

### Outputs:
- **`coeff_q`** — Complex output coefficients; list of `(n_rx, n_tx, n_taps)` when `stack` is `False`, else
  `(n_rx, n_tx, n_taps, n_snap)`
- **`delay_q`** — Output delays in seconds; list of `(n_rx, n_tx, n_taps)` or `(1, 1, n_taps)` when
  `stack` is `False`, else `(n_rx, n_tx, n_taps, n_snap)` or `(1, 1, n_taps, n_snap)`; the `(1, 1, ...)`
  shape only for shared delay input with `fix_taps` 1 or 2
