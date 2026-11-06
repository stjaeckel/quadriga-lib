# Channel functions

Quadriga-Lib 0.12.5 · Web page: http://quadriga-lib.org/cpp_api.html#sec-channel-functions

| Function | Summary |
| --- | --- |
| [any_type_id](#any_type_id) | Get type ID and raw access from a `std::any` object |
| [baseband_freq_response](#baseband_freq_response) | Compute the baseband frequency response of a MIMO channel |
| [baseband_freq_response_multi](#baseband_freq_response_multi) | Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients |
| [baseband_freq_response_vec](#baseband_freq_response_vec) | Compute the baseband frequency response of multiple MIMO channels |
| [hdf5_create](#hdf5_create) | Create a new HDF5 channel file with a defined storage layout |
| [hdf5_read_channel](#hdf5_read_channel) | Read a channel object from an HDF5 file at a specified 4D index |
| [hdf5_read_dset](#hdf5_read_dset) | Read an unstructured dataset from an HDF5 file at a specified 4D index |
| [hdf5_read_dset_names](#hdf5_read_dset_names) | Read names of unstructured datasets stored at a 4D slot in an HDF5 file |
| [hdf5_read_layout](#hdf5_read_layout) | Read the storage layout of an HDF5 channel file |
| [hdf5_reshape_layout](#hdf5_reshape_layout) | Reshape the 4D storage layout of an existing HDF5 channel file |
| [hdf5_write](#hdf5_write) | Write a channel object to an HDF5 file at a specified 4D index |
| [hdf5_write_dset](#hdf5_write_dset) | Write a single unstructured dataset to an HDF5 file at a specified 4D index |
| [path](#path) | Class for storing and managing a single propagation path with a compact fixed-size header |
| [qrt_file_append](#qrt_file_append) | Append one transmitter's path data to an existing QRT file |
| [qrt_file_init](#qrt_file_init) | Create a new QRT file and write its metadata header |
| [qrt_file_parse](#qrt_file_parse) | Read metadata from a QRT file |
| [qrt_file_read](#qrt_file_read) | Read ray-tracing CIR data from a QRT file |
| [qrt_file_read_raw](#qrt_file_read_raw) | Read raw ray-tracing path data from a QRT file into path objects |
| [qrt_paths_from_scatterers](#qrt_paths_from_scatterers) | Path objects of one CIR from spherical path data, for qrt_file_append |
| [qrt_read_cache_init](#qrt_read_cache_init) | Initialize a QRT read cache for fast repeated access |
| [quantize_delays](#quantize_delays) | Map path delays to a fixed tap grid using two-tap power-weighted interpolation |

---
<a id="any_type_id"></a>
## any_type_id
Get type ID and raw access from a `std::any` object

Not in: MATLAB, Python

- Inspects a `std::any` object and returns an integer type identifier for its contents
- Optionally retrieves dimensions (rows, columns, slices) for Armadillo matrix/cube/vector types; for `std::string`, `dims[0]` is the string length, `dims[1]`/`dims[2]` are zero
- Optionally retrieves a raw `void*` to the internal data — not type-safe, bypasses `const` protection; use with caution

### Declaration:
```cpp
int quadriga_lib::any_type_id(
    const std::any *data,
    unsigned long long *dims = nullptr,
    void **dataptr = nullptr);
```

### Inputs:
- **`data`** — Pointer to the `std::any` object to inspect

### Outputs:
- **`dims`** *(optional)* — Array of 3 values filled with `[rows, cols, slices]` of the contained Armadillo object
- **`dataptr`** *(optional)* — Receives a raw pointer to the object's internal data

### Returns:
- Integer type ID of the contained value:
  | ID  | Type                      | ID  | Type                   | ID  | Type                      |
  | --- | ------------------------- | --- | ---------------------- | --- | ------------------------- |
  | -2  | `no value`                | -1  | `unsupported type`     | 9   | `std::string`             |
  | 10  | `float`                   | 11  | `double`               | 12  | `unsigned long long int`  |
  | 13  | `long long int`           | 14  | `unsigned int`         | 15  | `int`                     |
  | 20  | `arma::Mat<float>`        | 21  | `arma::Mat<double>`    | 22  | `arma::Mat<arma::uword>`  |
  | 23  | `arma::Mat<arma::sword>`  | 24  | `arma::Mat<unsigned>`  | 25  | `arma::Mat<int>`          |
  | 30  | `arma::Cube<float>`       | 31  | `arma::Cube<double>`   | 32  | `arma::Cube<arma::uword>` |
  | 33  | `arma::Cube<arma::sword>` | 34  | `arma::Cube<unsigned>` | 35  | `arma::Cube<int>`         |
  | 40  | `arma::Col<float>`        | 41  | `arma::Col<double>`    | 42  | `arma::Col<arma::uword>`  |
  | 43  | `arma::Col<arma::sword>`  | 44  | `arma::Col<unsigned>`  | 45  | `arma::Col<int>`          |
  | 50  | `arma::Row<float>`        | 51  | `arma::Row<double>`    | 52  | `arma::Row<arma::uword>`  |
  | 53  | `arma::Row<arma::sword>`  | 54  | `arma::Row<unsigned>`  | 55  | `arma::Row<int>`          |

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (uses `any_type_id` to read dataset from HDF5 file)
- [hdf5_write_dset](#hdf5_write_dset) (HDF5 dataset writer)

---
<a id="baseband_freq_response"></a>
## baseband_freq_response
Compute the baseband frequency response of a MIMO channel

Also in: [MATLAB](../mex/channel-functions.md#baseband_freq_response) · [Python](../python/channel-functions.md#baseband_freq_response)

- Computes the frequency-domain channel matrix `H` at given sub-carrier positions via DFT over time-domain
  path coefficients and delays
- `delay` supports broadcasting: shape `[1, 1, n_path]` applies the same delays to all RX/TX pairs
- `pilot_grid` values are normalized to bandwidth: `0.0` = center frequency, `1.0` = center + bandwidth
- The phase `bandwidth · pilot_grid · delay` (in cycles) is computed in double precision and reduced to one period
  before sin / cos are evaluated in single precision, so long (absolute) delays keep their accuracy; coefficients and
  sums are single precision; uses AVX2 for 8-carrier parallel computation; double inputs are narrowed to float
  internally, results widened back
- Safe to call in a loop over snapshots and parallelize with OpenMP

### Declaration:
```cpp
void quadriga_lib::baseband_freq_response(
    const arma::Cube<dtype> *coeff_re,
    const arma::Cube<dtype> *coeff_im,
    const arma::Cube<dtype> *delay,
    const arma::Col<dtype> *pilot_grid,
    const double bandwidth,
    arma::Cube<dtype> *hmat_re,
    arma::Cube<dtype> *hmat_im,
    arma::Cube<std::complex<dtype>> *hmat = nullptr);
```

### Inputs:
- **`coeff_re`** — Real part of time-domain channel coefficients; `[n_rx, n_tx, n_path]`
- **`coeff_im`** — Imaginary part of time-domain channel coefficients; `[n_rx, n_tx, n_path]`
- **`delay`** — Path delays in seconds; `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`
- **`pilot_grid`** — Normalized sub-carrier positions in range `[0.0, 1.0]`; `[n_carriers]`
- **`bandwidth`** — Total baseband bandwidth

### Outputs:
- **`hmat_re`** *(optional)* — Real part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carriers]`
- **`hmat_im`** *(optional)* — Imaginary part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carriers]`
- **`hmat`** *(optional)* — Complex-valued frequency-domain channel matrix; `[n_rx, n_tx, n_carriers]`

### See also:
- [baseband_freq_response_vec](#baseband_freq_response_vec) (vectorized version)
- [baseband_freq_response_multi](#baseband_freq_response_multi) (multi-freq counterpart)
- [get_channels_planar](channel-generation-functions.md#get_channels_planar) (for generating coeff and delay)
- [get_channels_spherical](channel-generation-functions.md#get_channels_spherical) (for generating coeff and delay)

---
<a id="baseband_freq_response_multi"></a>
## baseband_freq_response_multi
Compute the wideband frequency response of a MIMO channel with frequency-dependent coefficients

Not in: MATLAB, Python

- Interpolates complex channel coefficients from a coarse input frequency grid (`freq_in`) to a dense
  output grid (`freq_out`) using SLERP: magnitude and unwrapped phase are each interpolated linearly along the shortest arc
- Applies delay-induced phase rotation `exp(-j·2·pi·freq_out·delay)` per output carrier in double
  precision to preserve accuracy at high carrier frequencies
- Only `delay[0]` is used; all entries in the `delay` vector should be identical
  (path geometry is frequency-independent)
- `delay` cube supports `[1, 1, n_path]` (planar wave) or `[n_rx, n_tx, n_path]` (spherical wave)
- Output frequencies outside the range of `freq_in` use constant extrapolation from the nearest endpoint
- At least one of `hmat_re`/`hmat_im` or `hmat` must be non-null

### Declaration:
```cpp
void quadriga_lib::baseband_freq_response_multi(
    const std::vector<arma::Cube<dtype>> &coeff_re,
    const std::vector<arma::Cube<dtype>> &coeff_im,
    const std::vector<arma::Cube<dtype>> &delay,
    const arma::Col<dtype> &freq_in,
    const arma::Col<dtype> &freq_out,
    arma::Cube<dtype> *hmat_re = nullptr,
    arma::Cube<dtype> *hmat_im = nullptr,
    arma::Cube<std::complex<dtype>> *hmat = nullptr,
    bool remove_delay_phase = true);
```

### Inputs:
- **`coeff_re`** — Real part of channel coefficients at each input frequency, vector of `n_freq_in` cubes `[n_rx, n_tx, n_path]`
- **`coeff_im`** — Imaginary part of channel coefficients at each input frequency, same structure as `coeff_re`
- **`delay`** — Path delays in seconds, vector of `n_freq_in` cubes; only `delay[0]` is used; shape `[n_rx, n_tx, n_path]` or `[1, 1, n_path]`
- **`freq_in`** — Input sample frequencies, sorted ascending; `[n_freq_in]`
- **`freq_out`** — Output carrier frequencies (absolute); `[n_carrier]`
- **`remove_delay_phase`** *(optional)* — Removes baked-in `exp(-j·2π·freq_in[f]·delay)` before SLERP and re-applies
  analytically at output frequencies; must be `true` for output from [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq) or
  [get_channels_spherical](channel-generation-functions.md#get_channels_spherical), `false` for pure envelope coefficients. These channels must be generated with
  `use_absolute_delays = true`: their coefficients carry the phase of the absolute path length also when `delay` holds
  relative delays. With relative delays the phase of the TX-RX distance stays in the coefficients and is interpolated
  across the `freq_in` grid, which gives wrong phases (magnitudes stay right) unless the `freq_in` spacing is far
  below `c / (TX-RX distance)`

### Outputs:
- **`hmat_re`** *(optional)* — Real part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier]`
- **`hmat_im`** *(optional)* — Imaginary part of the frequency-domain channel matrix; `[n_rx, n_tx, n_carrier]`
- **`hmat`** *(optional)* — Complex-valued frequency-domain channel matrix; `[n_rx, n_tx, n_carrier]`

### See also:
- [baseband_freq_response](#baseband_freq_response) (single-snapshot narrowband version)
- [baseband_freq_response_vec](#baseband_freq_response_vec) (batched narrowband version)
- [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq) (produces the multi-frequency input coefficients)

---
<a id="baseband_freq_response_vec"></a>
## baseband_freq_response_vec
Compute the baseband frequency response of multiple MIMO channels

Not in: MATLAB, Python

- Batch wrapper around [baseband_freq_response](#baseband_freq_response), applying it across snapshots in parallel via OpenMP
- Each element of the input vectors is a cube of shape `[n_rx, n_tx, n_path]`; `delay` supports broadcasting to `[1, 1, n_path]`
- Output vectors have length `n_out`: either `n_snap` (all snapshots) or `length(i_snap)` (subset)
- Same arithmetic as [baseband_freq_response](#baseband_freq_response): phase in double precision, sin / cos and sums in single precision

### Declaration:
```cpp
void quadriga_lib::baseband_freq_response_vec(
    const std::vector<arma::Cube<dtype>> *coeff_re,
    const std::vector<arma::Cube<dtype>> *coeff_im,
    const std::vector<arma::Cube<dtype>> *delay,
    const arma::Col<dtype> *pilot_grid,
    const double bandwidth,
    std::vector<arma::Cube<dtype>> *hmat_re = nullptr,
    std::vector<arma::Cube<dtype>> *hmat_im = nullptr,
    const arma::uvec *i_snap = nullptr);
```

### Inputs:
- **`coeff_re`** — Real part of time-domain channel coefficients, vector of `n_snap` cubes `[n_rx, n_tx, n_path]`
- **`coeff_im`** — Imaginary part of time-domain channel coefficients, same structure as `coeff_re`
- **`delay`** — Path delays in seconds, same structure as `coeff_re`; each cube broadcastable to `[1, 1, n_path]`
- **`pilot_grid`** — Normalized sub-carrier positions in range `[0.0, 1.0]`; `[n_carriers]`
- **`bandwidth`** — Total baseband bandwidth
- **`i_snap`** *(optional)* — Snapshot indices to process; if omitted, all `n_snap` snapshots are processed; `[n_out]`

### Outputs:
- **`hmat_re`** *(optional)* — Real part of frequency-domain channel matrices, vector of `n_out` cubes `[n_rx, n_tx, n_carriers]`
- **`hmat_im`** *(optional)* — Imaginary part of frequency-domain channel matrices, same structure as `hmat_re`

### See also:
- [baseband_freq_response](#baseband_freq_response) (single-snapshot variant)
- [baseband_freq_response_multi](#baseband_freq_response_multi) (multi-freq counterpart)

---
<a id="hdf5_create"></a>
## hdf5_create
Create a new HDF5 channel file with a defined storage layout

Also in: [MATLAB](../mex/channel-functions.md#hdf5_create_file) · [Python](../python/channel-functions.md#hdf5_create_file)

- Initializes a new HDF5 file for storing wireless channel data; overwrites existing files.
- Defines a 4D layout (x, y, z, w) where each index combination maps to one channel storage slot.
- Typical dimension mapping: x = BS, y = UE, z = frequency, w = scenario/repetition.
- Layout can be reshaped later with [hdf5_reshape_layout](#hdf5_reshape_layout) if the total slot count stays constant.
- The total slot count `nx × ny × nz × nw` must not exceed 4294967295 (2^32 - 1); larger layouts throw before
  the file is created.
- Files use the quadriga-lib channel file format version 3, an HDF5 file readable by any HDF5 reader (h5py, MATLAB,
  HDFView). Modify files only with the quadriga-lib functions: files changed by other HDF5 tools are not supported
  and may be rejected.

### Declaration:
```cpp
void quadriga_lib::hdf5_create(
    std::string fn,
    unsigned nx = 65536,
    unsigned ny = 1,
    unsigned nz = 1,
    unsigned nw = 1);
```

### Inputs:
- **`fn`** — Path and filename of the HDF5 file to create
- **`nx`** *(optional)* — Size of x-dimension
- **`ny`** *(optional)* — Size of y-dimension
- **`nz`** *(optional)* — Size of z-dimension
- **`nw`** *(optional)* — Size of w-dimension

### See also:
- [hdf5_reshape_layout](#hdf5_reshape_layout) (change layout dimensions of an existing file)
- [hdf5_write](#hdf5_write) (write channel data into a slot)

---
<a id="hdf5_read_channel"></a>
## hdf5_read_channel
Read a channel object from an HDF5 file at a specified 4D index

Also in: [MATLAB](../mex/channel-functions.md#hdf5_read_channel) · [Python](../python/channel-functions.md#hdf5_read_channel)

- Returns an empty channel object (`n_snap() == 0`) if the slot is empty or outside the layout.
- Structured data is stored in single precision in the file and converted to `dtype` on read.
- Unstructured fields (`std::any`) retain their original stored type without conversion; see [hdf5_read_dset](#hdf5_read_dset)
  for the returned types.
- Round trip with [hdf5_write](#hdf5_write): every per-snapshot field comes back with one entry per snapshot, also for
  snapshots without paths (empty entries keep their shape); a channel written without `center_frequency` comes back
  with an empty `center_frequency`; empty `tx_orientation` / `rx_orientation` come back as `[3, 1]` zeros.
- Throws `std::invalid_argument` if the file is malformed (e.g. a dataset does not have the size given by the
  attributes of the channel, or a required dataset or attribute is missing) or is not a channel file of version 3
  written by quadriga-lib: files written by quadriga-lib 0.12.4 and earlier (version 2) are rejected; files modified
  with other HDF5 tools are not supported and may be rejected.
- Files of quadriga-lib 0.12.4 and earlier can be converted to version 3 with the Python script
  `tools/hdf5_convert_v2.py` of the quadriga-lib sources (needs numpy, h5py and the quadriga_lib Python module):
  `python tools/hdf5_convert_v2.py old.hdf5 new.hdf5`.

### Declaration:
```cpp
quadriga_lib::channel<dtype> quadriga_lib::hdf5_read_channel(
    std::string fn,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0);
```

### Inputs:
- **`fn`** — Path to the HDF5 file
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension

### Returns:
- Channel object at the specified slot; empty if no data is present

### See also:
- [hdf5_write](#hdf5_write) (write a channel object to a slot)
- [hdf5_read_layout](#hdf5_read_layout) (inspect slot occupancy before reading)

---
<a id="hdf5_read_dset"></a>
## hdf5_read_dset
Read an unstructured dataset from an HDF5 file at a specified 4D index

Also in: [MATLAB](../mex/channel-functions.md#hdf5_read_dset) · [Python](../python/channel-functions.md#hdf5_read_dset)

- Reads a user-defined dataset stored under `prefix + par_name` (e.g., `"par_carrier_frequency"`).
- Returns an empty `std::any` if the dataset does not exist at the specified slot or name.
- Use [any_type_id](#any_type_id) to determine the contained type and obtain a raw pointer.
- Returned types: a scalar dataset gives a scalar; a 1D dataset an `arma::Col`, also when it has a single element;
  a 2D dataset an `arma::Mat`; a 3D dataset an `arma::Cube`. An `arma::Row` written with [hdf5_write_dset](#hdf5_write_dset)
  therefore comes back as an `arma::Col`.
- Element types: float, double, 32- and 64-bit signed and unsigned integers.
- Strings are returned without trailing NUL characters (an empty string written by [hdf5_write_dset](#hdf5_write_dset) comes back
  empty).
- Throws if the file does not exist or is not a channel file of version 3 (see [hdf5_read_channel](#hdf5_read_channel)).

### Declaration:
```cpp
std::any quadriga_lib::hdf5_read_dset(
    std::string fn,
    std::string par_name,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    std::string prefix = "par_");
```

### Inputs:
- **`fn`** — Path to the HDF5 file
- **`par_name`** — Dataset name without prefix, e.g., `"carrier_frequency"`
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension
- **`prefix`** *(optional)* — Dataset name prefix prepended before `par_name`

### Returns:
- `std::any` containing the dataset, or empty `std::any` if not found

### See also:
- [hdf5_write_dset](#hdf5_write_dset) (write an unstructured dataset)
- [any_type_id](#any_type_id) (inspect the type held in a `std::any`)

---
<a id="hdf5_read_dset_names"></a>
## hdf5_read_dset_names
Read names of unstructured datasets stored at a 4D slot in an HDF5 file

Also in: [MATLAB](../mex/channel-functions.md#hdf5_read_dset_names) · [Python](../python/channel-functions.md#hdf5_read_dset_names)

- Finds all datasets whose HDF5 name starts with `prefix` at slot `(ix, iy, iz, iw)`; returned names exclude the prefix.
- Throws if the file does not exist or is not a channel file of version 3 (see [hdf5_read_channel](#hdf5_read_channel)).

### Declaration:
```cpp
arma::uword quadriga_lib::hdf5_read_dset_names(
    std::string fn,
    std::vector<std::string> *par_names,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    std::string prefix = "par_");
```

### Inputs:
- **`fn`** — Path to the HDF5 file
- **`par_names`** — Pointer to vector receiving dataset names (without prefix)
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension
- **`prefix`** *(optional)* — Prefix used to identify unstructured datasets

### Returns:
- Number of datasets found at the specified slot

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (read a dataset by name)
- [hdf5_write_dset](#hdf5_write_dset) (write an unstructured dataset)

---
<a id="hdf5_read_layout"></a>
## hdf5_read_layout
Read the storage layout of an HDF5 channel file

Also in: [MATLAB](../mex/channel-functions.md#hdf5_read_layout) · [Python](../python/channel-functions.md#hdf5_read_layout)

- Returns `{nx, ny, nz, nw}` describing the 4D slot grid of the file.
- Returns `{0, 0, 0, 0}` if the file does not exist; throws if the file exists but is not a channel file of
  version 3 (see [hdf5_read_channel](#hdf5_read_channel)).
- `channelID` entries are `0` for empty slots; length equals `nx × ny × nz × nw` (serialized linear index).

### Declaration:
```cpp
arma::u32_vec quadriga_lib::hdf5_read_layout(
    std::string fn,
    arma::u32_vec *channelID = nullptr);
```

### Inputs:
- **`fn`** — Path to the HDF5 file
- **`channelID`** *(optional)* — Pointer to vector receiving the serialized slot occupancy list; `[nx·ny·nz·nw]`

### Returns:
- Four-element vector `{nx, ny, nz, nw}` describing the layout dimensions

### See also:
- [hdf5_create](#hdf5_create) (create a file with a defined layout)
- [hdf5_reshape_layout](#hdf5_reshape_layout) (change layout dimensions of an existing file)

---
<a id="hdf5_reshape_layout"></a>
## hdf5_reshape_layout
Reshape the 4D storage layout of an existing HDF5 channel file

Also in: [MATLAB](../mex/channel-functions.md#hdf5_reshape_layout) · [Python](../python/channel-functions.md#hdf5_reshape_layout)

- Updates `{nx, ny, nz, nw}` of an existing file; total slot count `nx × ny × nz × nw` must remain unchanged.
- Throws if the new layout violates the total-count constraint or has more than 4294967295 (2^32 - 1) slots.
- Throws if the file does not exist or is not a channel file of version 3 (see [hdf5_read_channel](#hdf5_read_channel)).
- One writer per file at a time; files are not locked.

### Declaration:
```cpp
void quadriga_lib::hdf5_reshape_layout(
    std::string fn,
    unsigned nx,
    unsigned ny = 1,
    unsigned nz = 1,
    unsigned nw = 1);
```

### Inputs:
- **`fn`** — Path to the HDF5 file
- **`nx`** — New size of x-dimension
- **`ny`** *(optional)* — New size of y-dimension
- **`nz`** *(optional)* — New size of z-dimension
- **`nw`** *(optional)* — New size of w-dimension

### See also:
- [hdf5_create](#hdf5_create) (set initial layout at file creation)
- [hdf5_read_layout](#hdf5_read_layout) (query current layout)

---
<a id="hdf5_write"></a>
## hdf5_write
Write a channel object to an HDF5 file at a specified 4D index

Also in: [MATLAB](../mex/channel-functions.md#hdf5_write_channel) · [Python](../python/channel-functions.md#hdf5_write_channel)

- Writes a `quadriga_lib::channel<dtype>` object to slot `(ix, iy, iz, iw)` in the HDF5 file.
- Creates the file with default layout `(65536 × 1 × 1 × 1)` if it does not exist; an existing file must be a
  channel file of version 3 written by quadriga-lib (files of quadriga-lib 0.12.4 and earlier are rejected, see
  [hdf5_read_channel](#hdf5_read_channel)).
- If the slot already holds data, the slot is replaced completely: the old structured data and all unstructured
  datasets of the slot (`par_*`, including those added with [hdf5_write_dset](#hdf5_write_dset)) are removed, and only the new
  channel is stored there. The space of the removed data is not reused, so the file does not shrink; to reclaim
  it, copy the channels into a new file ([hdf5_read_channel](#hdf5_read_channel), [hdf5_write](#hdf5_write)).
- Throws if the index was not reserved during [hdf5_create](#hdf5_create).
- Structured data is always stored in single precision; input may be float or double.
- Unstructured data: supported types are string, double, float, (u)int32, (u)int64; up to 3D; storage order preserved.
- Set `assume_valid = true` to skip integrity validation (faster but unsafe for potentially corrupted data).
- The channel name is stored with at most 255 characters.
- One writer per file at a time; files are not locked. Reading is safe while no one writes; different files can be
  read and written from parallel threads.

### Declaration:
```cpp
int quadriga_lib::hdf5_write(
    const quadriga_lib::channel<dtype> *ch,
    std::string fn,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    bool assume_valid = false);
```

### Inputs:
- **`ch`** — Pointer to the channel object to write
- **`fn`** — Path to the HDF5 file
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension
- **`assume_valid`** *(optional)* — Skip channel integrity validation before writing

### Returns:
- `0` if the slot was empty, `1` if existing data in the slot was replaced

### See also:
- [hdf5_create](#hdf5_create) (create file and reserve layout)
- [hdf5_read_channel](#hdf5_read_channel) (read channel data from a slot)

---
<a id="hdf5_write_dset"></a>
## hdf5_write_dset
Write a single unstructured dataset to an HDF5 file at a specified 4D index

Also in: [MATLAB](../mex/channel-functions.md#hdf5_write_dset) · [Python](../python/channel-functions.md#hdf5_write_dset)

- Dataset is stored under `prefix + par_name`; name must contain only alphanumeric characters and underscores.
- Adds the dataset to the slot and keeps everything else stored there (channel data and other datasets); an empty
  slot is reserved for it. A later [hdf5_write](#hdf5_write) to the same slot replaces the slot completely, including the
  datasets written with this function.
- Supported scalar types: `std::string`, `unsigned`, `int`, `long long`, `unsigned long long`, `float`, `double`
- Supported Armadillo types: `arma::Col`, `arma::Row`, `arma::Mat`, `arma::Cube` with element types `float`, `double`, `int`, `unsigned`, `sword`, `uword`, `unsigned long long`
- `arma::Row` is converted to `arma::Col` before writing.
- Throws for unsupported types and invalid names, before the file is changed.
- Throws if a dataset with the same name already exists at the specified slot; no overwrite/update is supported.
- One writer per file at a time; files are not locked.
- Creates the file with the default layout of [hdf5_create](#hdf5_create) if it does not exist; an existing file must be
  a channel file of version 3 (see [hdf5_read_channel](#hdf5_read_channel)).

### Declaration:
```cpp
void quadriga_lib::hdf5_write_dset(
    std::string fn,
    std::string par_name,
    const std::any *par_data,
    unsigned ix = 0,
    unsigned iy = 0,
    unsigned iz = 0,
    unsigned iw = 0,
    std::string prefix = "par_");
```

### Inputs:
- **`fn`** — Path to the HDF5 file
- **`par_name`** — Dataset name without prefix; alphanumeric and underscores only
- **`par_data`** — Pointer to the data to write; type must be supported (see above)
- **`ix`** *(optional)* — Slot index in x-dimension
- **`iy`** *(optional)* — Slot index in y-dimension
- **`iz`** *(optional)* — Slot index in z-dimension
- **`iw`** *(optional)* — Slot index in w-dimension
- **`prefix`** *(optional)* — Prefix prepended to `par_name` in the HDF5 file

### See also:
- [hdf5_read_dset](#hdf5_read_dset) (read an unstructured dataset by name)
- [hdf5_read_dset_names](#hdf5_read_dset_names) (list available dataset names at a slot)
- [any_type_id](#any_type_id) (inspect the type held in a `std::any`)

---
<a id="path"></a>
## path
Class for storing and managing a single propagation path with a compact fixed-size header

Not in: MATLAB, Python

- Represents one ray from origin to destination through a sequence of `nSEG` interaction points
- A 56-byte header holds metadata, the first 2 interaction type codes, and the frequency-0
  transfer coefficients; a variable-length heap buffer holds coordinates, the remaining
  frequency coefficients, and the overflow interaction codes
- Two layout modes selected at initialization: EM carries a full 2x2 Jones matrix per frequency;
  SCALAR carries a single complex pressure coefficient per frequency
- Copyable and movable; the moved-from object is left in the valid empty state (`nSEG == 0`, `nFRQ == 1`, EM)

### Attributes:
| Attribute            | Size   | Description                                                       |
| -------------------- | ------ | ----------------------------------------------------------------- |
| `unsigned iC`        | scalar | Channel ID: the channel to which the path belongs                 |
| `uint8_t nREF`       | scalar | Number of reflections (interaction type codes 128-255 except 193) |
| `uint8_t nTRA`       | scalar | Number of transmissions / refractions (type codes 1-127)          |
| `uint8_t nSUB`       | scalar | Number of subdivisions                                            |
| `uint8_t nSCT`       | scalar | Number of scattering events (interaction type code 193, set by [qrt_paths_from_scatterers](#qrt_paths_from_scatterers); QRT files store the codes, not the counter) |
| `float length`       | scalar | Accumulated path length, origin through the last interaction (m)  |

### Data buffer layout (EM mode):
| Block         | Size (floats)     | Description                                             |
| ------------- | ----------------- | ------------------------------------------------------- |
| Coordinates   | `3 * nSEG`        | Interaction points `[x, y, z]` per segment              |
| Jones         | `8 * (nFRQ - 1)`  | Transfer matrices for frequencies 1..nFRQ-1, col-major  |
| Interactions  | `(nSEG + 1) / 4`  | Overflow type codes past the first 2, packed 4 per float, present for `nSEG > 2` |

### Data buffer layout (SCALAR mode):
| Block         | Size (floats)     | Description                                             |
| ------------- | ----------------- | ------------------------------------------------------- |
| Coordinates   | `3 * nSEG`        | Interaction points `[x, y, z]` per segment              |
| Coeff         | `2 * (nFRQ - 4)`  | Pressure coefficients for frequencies 4..nFRQ-1         |
| Interactions  | `(nSEG + 1) / 4`  | Overflow type codes past the first 2, packed 4 per float, present for `nSEG > 2` |

### Simple member functions:
| Method            | Description                                                                     |
| ----------------- | ------------------------------------------------------------------------------- |
| `.n_freq()`       | Returns the number of frequencies (1-127)                                       |
| `.n_seg()`        | Returns the number of segments                                                  |
| `.is_scalar()`    | Returns true for SCALAR layout, false for EM                                    |
| `.init()`         | (Re)initializes storage to a given segment / frequency layout                   |
| `.free()`         | Releases the data buffer and resets to the valid empty state                    |
| `.coord()`        | Returns a pointer to a segment's `[x, y, z]` coordinates                        |
| `.xpr_coeff()`    | Returns a pointer to a frequency's transfer coefficients                        |
| `.operator()`     | Returns a last-segment coordinate by index (0=x, 1=y, 2=z)                      |
| `.interaction_type_codes()` | Returns the full interaction type sequence as a vector                |
| `.set_interaction_type_codes()` | Sets the interaction type sequence; requires one code per segment |

### Function 'calc_length':
Calculates the total length of the path

- The stored member `length` spans origin through the last interaction; it never includes the leg to the destination
- The read-only overload adds only the final last-to-D leg to the stored member; the origin overloads recompute every leg and overwrite the member with the origin-to-last total
- Returns the full length including the leg to `D`; on an empty path (`nSEG == 0`) the read-only form returns NaN while the origin form returns the straight origin-to-destination distance
```cpp
float calc_length(float Dx, float Dy, float Dz) const;                         // member length + last to D distance
float calc_length(float Dx, float Dy, float Dz, float Ox, float Oy, float Oz); // calculate full path length, update member
float calc_length(const float *D, const float *O = nullptr);                   // 3-element overload
```
- **`Dx`, `Dy`, `Dz`, `*D`** — Destination coordinates
- **`Ox`, `Oy`, `Oz`, `*O`** — Origin coordinates (optional); if provided, accumulated path length is updated

### Function 'calc_gain':
Calculates the path gain (linear power) from the transfer coefficients at one frequency

- EM mode returns the maximum column power of the 2x2 Jones matrix; SCALAR mode returns the squared magnitude of the pressure coefficient
- A degenerate coefficient set (all-zero, underflowed, or containing NaN) returns 0, flagging the path for pruning
- With `fGHz > 0`, free-space path loss is folded in: `(lambda / (4 pi d))^2` in EM mode, `1 / d^2` (frequency-independent) in SCALAR mode
- The distance `d` is the stored `length` by default; pass `len > 0` to override it, e.g. to match a total length from the
  read-only `calc_length` before the member is updated
```cpp
float calc_gain(float fGHz = 0.0f, size_t freq = 0, float len = 0.0f) const;
```
- **`fGHz`** — Frequency in GHz; `> 0` applies path loss, `0` returns polarization power only. In SCALAR mode any
  positive value applies spherical spreading; the magnitude is ignored
- **`freq`** — Frequency index into the coefficient store, valid range `0` to `nFRQ - 1`
- **`len`** — Path length override in meters; `> 0` replaces the stored `length` for the path-loss term, `<= 0` uses the stored value

### Function 'xpr_update':
Left-multiplies the ray's transfer matrix by an interaction matrix, applies a power gain, and returns the resulting gain

- Updates the coefficient slot for one frequency in place: `M := coeff_update * M`, then scales by `sqrt(gain_update)` in amplitude
- `coeff_update == nullptr` applies only the gain; `gain_update == 1.0` applies only the multiply; both defaults leave the matrix unchanged and just measure it
- The update reads all old values before writing, so `coeff_update` may alias the slot; the return value uses the same gain definition as `calc_gain`
- Must be called after `extend` closes the path, so the stored `length` is final when path loss is requested
```cpp
float xpr_update(const float *coeff_update = nullptr, float gain_update = 1.0f, size_t freq = 0, float fGHz = 0.0f, float len = 0.0f);
```
- **`coeff_update`** — Interaction matrix, length 8 (EM) or 2 (SCALAR); `nullptr` skips the multiply
- **`gain_update`** — Power gain applied to the slot (converted to an amplitude factor internally); `1.0` skips the scaling
- **`freq`** — Frequency index into the coefficient store, valid range `0` to `nFRQ - 1`
- **`fGHz`** — Frequency in GHz for the returned gain; `> 0` applies path loss as in `calc_gain`, `0` returns polarization power only
- **`len`** — Path length override in meters; `> 0` replaces the stored `length` for the path-loss term, `<= 0` uses the stored value

### Function 'duplicate';
Copies the path into an existing target object and returns its length

- Performs a deep copy: the target receives an independent data buffer and all metadata
- The target's previous contents are released; the source is left unchanged
```cpp
float duplicate(path &target) const;
```
- **`target`** — Destination path, overwritten with a deep copy of the source

### Function 'extend':
Copies the path into a target and appends one new segment, returning the new total length

- The new coordinate is appended after the existing segments; the stored `length` grows by the origin-to-new-point distance of the appended leg
- The interaction `type` is recorded for the new segment, counters nREF, nTRA, nSUB, nSCT are not updated (left to the caller)
- The target receives an independent buffer sized for the extra segment; the source is left unchanged
- Throws if the source already holds the maximum of 255 segments
```cpp
float extend(path &target, float x, float y, float z, uint8_t type = 0) const;
```
- **`target`** — Destination path, overwritten with the extended copy
- **`x`, `y`, `z`** — Coordinates of the appended interaction point
- **`type`** — Interaction type code for the new segment; drives the reflection / transmission counters

---
<a id="qrt_file_append"></a>
## qrt_file_append
Append one transmitter's path data to an existing QRT file

Not in: MATLAB, Python

- Writes the path data for a single TX into the next free slot reserved by [qrt_file_init](#qrt_file_init), and records the TX name, position, and orientation.
- Paths are grouped by their CIR index (`path::iC`) and stored per CIR as: interaction counts, polarization coefficients, interaction coordinates, and interaction type codes.
- Validates every path before writing: `iC` must be within the file's CIR count, and each path's frequency count and layout mode (EM/scalar) must match the file exactly.
- Throws when all TX slots are already filled — the file holds at most the `no_orig` slots reserved at init.
- Only v5/6 files can be appended to. Positional data is converted to `float` on write.
- Returns the total number of paths written.

### Declaration:
```cpp
template <typename dtype>
size_t quadriga_lib::qrt_file_append(
    const std::string &fn,
    const std::vector<quadriga_lib::path> &path_data,
    const arma::Col<dtype> &orig_pos,
    const arma::Col<dtype> &orig_orientation = {0.0, 0.0, 0.0},
    const std::string &orig_name = "TX");
```

### Inputs:
- **`fn`** — Path to an existing QRT file created by [qrt_file_init](#qrt_file_init)
- **`path_data`** — Paths to write; each carries its CIR index in `iC` and must match the file's frequency count and layout mode
- **`orig_pos`** — Transmitter position; `[3]`
- **`orig_orientation`** — Transmitter orientation (bank, tilt, head) in rad; `[3]`
- **`orig_name`** — Transmitter name; at most 255 characters

### Returns:
- Number of paths written across all CIRs.

### See also:
- [qrt_file_init](#qrt_file_init) (create the file and reserve TX slots)
- [qrt_file_read](#qrt_file_read) (read the appended data back)

---
<a id="qrt_file_init"></a>
## qrt_file_init
Create a new QRT file and write its metadata header

Not in: MATLAB, Python

- Writes a v5 (EM) or v6 (scalar) header: frequencies, CIR positions/orientations, RX (MT) metadata, and a reserved BS (TX) region.
- Writes no path data. The BS position/orientation rows and the BS index table are reserved as zeros; each [qrt_file_append](#qrt_file_append) call fills the next free slot.
- CIR orientation is stored compressed: only the angle columns that carry a nonzero value after conversion to `float` are written,
  encoded in a per-file format byte. [qrt_file_append](#qrt_file_append) reads this byte from the file.
- `no_orig` fixes the number of TX slots; the file can hold at most that many appended TX blocks.
- Positional data is converted to `float` on write regardless of `dtype`.

### Declaration:
```cpp
template <typename dtype>
void quadriga_lib::qrt_file_init(
    const std::string &fn,
    const arma::Col<dtype> &freq,
    const arma::Mat<dtype> &cir_pos,
    const arma::Mat<dtype> &cir_orientation,
    const std::vector<std::string> &dest_names = {"RX"},
    const arma::u32_vec &cir_offset = {0},
    unsigned no_orig = 1,
    bool scalar_mode = false);
```

### Inputs:
- **`fn`** — Path to the QRT file to create (truncated if it exists; all inputs are checked before the file is
  opened, so an invalid input throws and leaves an existing file unchanged)
- **`freq`** — Frequencies in GHz (EM) or Hz (scalar); `[n_freq]`, 1 to 127 entries
- **`cir_pos`** — CIR positions in Cartesian coordinates; `[no_cir, 3]`
- **`cir_orientation`** — CIR orientations as Euler angles (bank, tilt, head); `[no_cir, 3]` or empty for none
- **`dest_names`** — Receiver (MT) names, at most 255 characters each; `[no_dest]`
- **`cir_offset`** — CIR offset for each receiver, 0-based, the first entry 0; `[no_dest]`, must equal `dest_names` in length
- **`no_orig`** — Number of origin (TX) slots to reserve; at least 1
- **`scalar_mode`** — `true` writes a v6 scalar-layout file, `false` a v5 EM file

### See also:
- [qrt_file_append](#qrt_file_append) (write one TX block into a reserved slot)
- [qrt_file_read](#qrt_file_read) (read CIR data back)

---
<a id="qrt_file_parse"></a>
## qrt_file_parse
Read metadata from a QRT file

Also in: [MATLAB](../mex/channel-functions.md#qrt_file_parse) · [Python](../python/channel-functions.md#qrt_file_parse)

- Parses a QRT file and extracts snapshot counts, origin/destination counts, frequency count, CIR offsets, names, positions, orientations, and file version.
- All output arguments are optional; pass `nullptr` to skip any.
- If `file` is `nullptr`, the file is opened internally and closed on return; if provided, the stream is left open.
- When `no_dest == 0` in the file, one implicit RX named `"RX"` is assumed; `dest_names` and `cir_offset` reflect this.
- Throws `std::invalid_argument` if the file is truncated, or if a count or offset in it does not fit the file size.

### Declaration:
```cpp
void quadriga_lib::qrt_file_parse(
    const std::string &fn,
    arma::uword *no_cir = nullptr,
    arma::uword *no_orig = nullptr,
    arma::uword *no_dest = nullptr,
    arma::uword *no_freq = nullptr,
    arma::uvec *cir_offset = nullptr,
    std::vector<std::string> *orig_names = nullptr,
    std::vector<std::string> *dest_names = nullptr,
    int *version = nullptr,
    arma::fvec *freq = nullptr,
    arma::fmat *cir_pos = nullptr,
    arma::fmat *cir_orientation = nullptr,
    arma::fmat *orig_pos = nullptr,
    arma::fmat *orig_orientation = nullptr,
    std::ifstream *file = nullptr);
```

### Inputs:
- **`fn`** — Path to the QRT file
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; pass `nullptr` to let the function open/close the file internally

### Outputs:
- **`no_cir`** *(optional)* — Number of channel snapshots per origin point
- **`no_orig`** *(optional)* — Number of origin points (TX)
- **`no_dest`** *(optional)* — Number of destination points (RX)
- **`no_freq`** *(optional)* — Number of frequency bands
- **`cir_offset`** *(optional)* — CIR offset per destination; `[no_dest]`
- **`orig_names`** *(optional)* — Names of origin points; `[no_orig]`
- **`dest_names`** *(optional)* — Names of destination points; `[no_dest]`
- **`version`** *(optional)* — QRT file version number
- **`freq`** *(optional)* — Frequencies as stored in the file; usually in GHz; `[no_freq]`
- **`cir_pos`** *(optional)* — CIR positions in Cartesian coordinates; `[no_cir, 3]`
- **`cir_orientation`** *(optional)* — CIR orientations as Euler angles; `[no_cir, 3]`
- **`orig_pos`** *(optional)* — Origin (TX) positions in Cartesian coordinates; `[no_orig, 3]`
- **`orig_orientation`** *(optional)* — Origin (TX) orientations as Euler angles; `[no_orig, 3]`

---
<a id="qrt_file_read"></a>
## qrt_file_read
Read ray-tracing CIR data from a QRT file

Also in: [MATLAB](../mex/channel-functions.md#qrt_file_read) · [Python](../python/channel-functions.md#qrt_file_read)

- Reads channel impulse response data for a specific snapshot index and origin point.
- All output arguments are optional; pass `nullptr` to skip any.
- If `downlink = true`, origin is TX and destination is RX; if `false`, roles are swapped.
- For tight-loop performance, pass a pre-opened `std::ifstream` and a [qrt_read_cache_init](#qrt_read_cache_init)-populated cache; reduces per-call I/O to 2 seeks and 4 reads.
- `fn` is ignored when both `file` and `cache` are provided.
- Reading a TX slot that was reserved by [qrt_file_init](#qrt_file_init) but not yet written by [qrt_file_append](#qrt_file_append) returns
  empty path outputs with zeroed positions, rather than throwing; an out-of-range index still throws.
- Throws `std::invalid_argument` if the file is truncated, or if a count or offset in it does not fit the file size.

### Declaration:
```cpp
void quadriga_lib::qrt_file_read(
    const std::string &fn,
    arma::uword i_cir = 0,
    arma::uword i_orig = 0,
    bool downlink = true,
    arma::Col<dtype> *center_frequency = nullptr,
    arma::Col<dtype> *tx_pos = nullptr,
    arma::Col<dtype> *tx_orientation = nullptr,
    arma::Col<dtype> *rx_pos = nullptr,
    arma::Col<dtype> *rx_orientation = nullptr,
    arma::Mat<dtype> *fbs_pos = nullptr,
    arma::Mat<dtype> *lbs_pos = nullptr,
    arma::Mat<dtype> *path_gain = nullptr,
    arma::Col<dtype> *path_length = nullptr,
    arma::Cube<dtype> *M = nullptr,
    arma::Col<dtype> *aod = nullptr,
    arma::Col<dtype> *eod = nullptr,
    arma::Col<dtype> *aoa = nullptr,
    arma::Col<dtype> *eoa = nullptr,
    std::vector<arma::Mat<dtype>> *path_coord = nullptr,
    int normalize_M = 1,
    arma::u32_vec *no_int = nullptr,
    arma::fmat *coord = nullptr,
    std::vector<uint8_t> *interact_type = nullptr,
    std::ifstream *file = nullptr,
    const qrt_read_cache *cache = nullptr);
```

### Inputs:
- **`fn`** — Path to the QRT file; ignored when both `file` and `cache` are provided
- **`i_cir`** — Snapshot index, 0-based
- **`i_orig`** — Origin index, 0-based
- **`downlink`** — If `true`, origin=TX, destination=RX; if `false`, roles are swapped
- **`normalize_M`** *(optional)* — Controls `M` and `path_gain` scaling where PL is the propagation-only path loss
  - v4/v5 (EM):    FSPL = 32.45 + 20·log10(f_GHz) + 20·log10(d_m)  [dB]
  - v6 (scalar):   20·log10(d_m) + α(f)·d_m  [dB], with α from ISO 9613-1 at T=20°C, RH=50%, p=1 atm

    | `normalize_M` | `M`                   | `path_gain`                      |
    | ------------- | --------------------- | -------------------------------- |
    | 0             | As stored in QRT file | -PL                              |
    | 1             | Max column power = 1  | -PL minus material losses        |

  Other values throw `std::invalid_argument`.
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; left open on return
- **`cache`** *(optional)* — Pre-populated cache from [qrt_read_cache_init](#qrt_read_cache_init)

### Outputs:
- **`center_frequency`** — Center frequency in Hz; `[n_freq]`
- **`tx_pos`** — Transmitter position in Cartesian coordinates; `[3]`
- **`tx_orientation`** — Transmitter orientation (bank, tilt, heading); `[3]`
- **`rx_pos`** — Receiver position in Cartesian coordinates; `[3]`
- **`rx_orientation`** — Receiver orientation (bank, tilt, heading); `[3]`
- **`fbs_pos`** — First-bounce scatterer positions; `[3, n_path]`
- **`lbs_pos`** — Last-bounce scatterer positions; `[3, n_path]`
- **`path_gain`** — Path gain on linear scale; `[n_path, n_freq]`
- **`path_length`** — Absolute path length TX to RX phase center; `[n_path]`
- **`M`** — Polarization transfer matrix; `[8, n_path, n_freq]` or `[2, n_path, n_freq]` for v6 files
- **`aod`** — Departure azimuth angles; `[n_path]`
- **`eod`** — Departure elevation angles; `[n_path]`
- **`aoa`** — Arrival azimuth angles; `[n_path]`
- **`eoa`** — Arrival elevation angles; `[n_path]`
- **`path_coord`** — Interaction coordinates per path; vector of length `n_path`, each `[3, n_interact + 2]`
- **`no_int`** — Number of mesh interactions per path; 0 indicates LOS; `[n_path]`
- **`coord`** — Interaction coordinates; `[3, sum(no_int)]`
- **`interact_type`** — Interaction type codes, concatenated per path and segmented by `no_int`; `[sum(no_int)]`.
  Empty for v4 legacy files. GBSM paths of [qrt_paths_from_scatterers](#qrt_paths_from_scatterers) carry 129 (GR) and 193 (NLOS); LOS and GR
  paths are not marked as GBSM.

### Example:
```cpp
std::ifstream stream("scene.qrt", std::ios::in | std::ios::binary);
auto cache = quadriga_lib::qrt_read_cache_init("scene.qrt", &stream);
arma::vec center_freq, tx_pos, rx_pos, path_length;
arma::mat path_gain; arma::cube M;
for (arma::uword ic = 0; ic < cache.no_cir; ++ic)
    for (arma::uword io = 0; io < cache.no_orig; ++io)
        quadriga_lib::qrt_file_read<double>("", ic, io, true,
            &center_freq, &tx_pos, nullptr, &rx_pos, nullptr,
            nullptr, nullptr, &path_gain, &path_length, &M,
            nullptr, nullptr, nullptr, nullptr, nullptr, 1,
            nullptr, nullptr, &stream, &cache);
```

### See also:
- [qrt_read_cache_init](#qrt_read_cache_init) (populate cache for fast repeated reads)
- [qrt_file_parse](#qrt_file_parse) (extract file metadata without reading CIR data)

---
<a id="qrt_file_read_raw"></a>
## qrt_file_read_raw
Read raw ray-tracing path data from a QRT file into path objects

Not in: MATLAB, Python

- Reassembles the stored path data for a single origin (TX) into a vector of [path](#path) objects — the inverse of [qrt_file_append](#qrt_file_append).
- Unlike [qrt_file_read](#qrt_file_read), which returns processed CIR data (channel matrices, angles, path loss), this returns the raw per-path storage:
  coordinates, polarization coefficients, and interaction type codes.
- Each returned path has `iC` set to its CIR index. Counters not stored in the file
  (`nREF`, `nTRA`, `nSUB`, `nSCT`) and `length` are left at their defaults.
- Only v05/v06 files can be read as raw paths; v4 legacy files are rejected.
- An origin slot that was reserved by [qrt_file_init](#qrt_file_init) but never written by [qrt_file_append](#qrt_file_append) returns an empty vector.
- Throws `std::invalid_argument` if the file is truncated, or if a count or offset in it does not fit the file size.
- For tight loops over many origins, pass a pre-opened `std::ifstream` and a [qrt_read_cache_init](#qrt_read_cache_init)-populated cache to avoid
  re-parsing the header on each call.

### Declaration:
```cpp
std::vector<quadriga_lib::path> quadriga_lib::qrt_file_read_raw(
    const std::string &fn,
    arma::uword i_orig = 0,
    std::ifstream *file = nullptr,
    const qrt_read_cache *cache = nullptr);
```

### Inputs:
- **`fn`** — Path to the QRT file; ignored when both `file` and `cache` are supplied
- **`i_orig`** — Origin index to read (for downlink, origin = TX); must be less than `no_orig`
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; pass `nullptr` to let the function open/close the file internally
- **`cache`** *(optional)* — Pre-parsed metadata from [qrt_read_cache_init](#qrt_read_cache_init); pass `nullptr` to parse the header on this call

### Returns:
- Vector of [path](#path) objects for the requested origin, one per stored path across all CIRs; empty if the origin slot is unwritten.

### See also:
- [qrt_file_append](#qrt_file_append) (write path data — the inverse operation)
- [qrt_read_cache_init](#qrt_read_cache_init) (populate cache for fast repeated reads)
- [qrt_file_read](#qrt_file_read) (read processed CIR data instead of raw paths)

---
<a id="qrt_paths_from_scatterers"></a>
## qrt_paths_from_scatterers
Path objects of one CIR from spherical path data, for qrt_file_append

Not in: MATLAB, Python

- Converts the paths of one Tx-Rx pair (e.g. of [ssf_paths](geometry-based-stochastic-model.md#ssf_paths) with `drift` mobility) into [path](#path) objects for
  [qrt_file_append](#qrt_file_append); one call per CIR, the objects of all CIRs of one origin appended together
- Interactions: LOS none; GR one reflection at `lbs_pos` (code 129, `nREF` = 1); NLOS two point scatterers at
  `fbs_pos`, `lbs_pos` (code 193, `nSCT` = 2); `length` = polyline from the Tx to the last interaction. Counters
  and `length` exist on the returned objects only ([qrt_file_read_raw](#qrt_file_read_raw) returns 0): files store the codes
- The path gain and the inverse free-space loss of the polyline are folded into the stored Jones matrices, so that
  [qrt_file_read](#qrt_file_read) (`downlink = true`, `normalize_M` 0 or 1) returns sqrt(`path_gain`)·`M` per complex entry
  within float32 accuracy, the polyline as `path_length` and `path_type` as `no_int`
- `tx_pos` must be the `orig_pos` of [qrt_file_append](#qrt_file_append) and `rx_pos` the CIR position of [qrt_file_init](#qrt_file_init) (which
  takes GHz): `qrt_file_read` recomputes the lengths from the stored positions; not checked
- Float32 storage: use local coordinates, shifted in x and y only (ground at z = 0); path length error 2.7 µm,
  0.31 mm, 1.0 mm for coordinates up to 150 m, 5 km, 20 km (phase 0.003, 0.39, 1.29 rad at 60 GHz)
- Memory: 80 to 105 bytes per path object at one frequency, e.g. 3 to 4 GB for 1000 UTs, 100 snapshots, 381 rays
- A path of length 0 (LOS with `tx_pos` = `rx_pos`) throws: `qrt_file_read` would return an infinite gain

### Declaration:
```cpp
std::vector<path> quadriga_lib::qrt_paths_from_scatterers(
    const arma::vec &center_frequency,
    const arma::vec &tx_pos,
    const arma::vec &rx_pos,
    const arma::mat &fbs_pos,
    const arma::mat &lbs_pos,
    const arma::mat &path_gain,
    const arma::cube &M,
    const arma::uvec &path_type,
    arma::uword i_cir = 0);
```

### Inputs:
- **`center_frequency`** — Frequencies in [Hz], 1 to 127 entries; `[n_freq]`
- **`tx_pos`** — Tx (origin) position in [m]; `[3]`
- **`rx_pos`** — Rx position of the CIR in [m]; `[3]`
- **`fbs_pos`**, **`lbs_pos`** — First- and last-bounce scatterer positions in [m] (GR: `lbs_pos`); `[3, n_path]`
- **`path_gain`** — Path gains (linear); `[n_path, n_freq]`
- **`M`** — Polarization transfer matrices, row order of [get_channels_spherical](channel-generation-functions.md#get_channels_spherical); `[8, n_path, n_freq]`
- **`path_type`** — Path type, 0 = LOS, 1 = GR, 2 = NLOS; `[n_path]`
- **`i_cir`** *(optional)* — CIR index, 0-based, stored in `path::iC`, < 2^32

### Returns:
- Path objects in input order, EM layout with `n_freq` frequencies; `[n_path]`

### See also:
- [qrt_file_append](#qrt_file_append) (writes the paths)
- [gbsm_rays_from_paths](geometry-based-stochastic-model.md#gbsm_rays_from_paths) (ray data from the paths read back)

---
<a id="qrt_read_cache_init"></a>
## qrt_read_cache_init
Initialize a QRT read cache for fast repeated access

Not in: MATLAB, Python

- Reads all fixed metadata from a QRT file into a `quadriga_lib::qrt_read_cache` struct.
- Pre-computes byte offsets so subsequent [qrt_file_read](#qrt_file_read) calls need only 2 seeks and 4 reads instead of re-parsing the header.
- Populate once, then pass the cache and a shared `std::ifstream` to [qrt_file_read](#qrt_file_read) for tight-loop performance.
- If `file` is `nullptr`, the file is opened internally and closed on return; if provided, the stream is left open.
- For a TX slot reserved by [qrt_file_init](#qrt_file_init) but not yet written, `orig_index` and `path_data_offset`
  are `0`; [qrt_file_read](#qrt_file_read) treats that as an empty (unwritten) origin.
- Throws `std::invalid_argument` if the file is truncated, or if a count or offset in it does not fit the file size.

### Declaration:
```cpp
quadriga_lib::qrt_read_cache quadriga_lib::qrt_read_cache_init(
    const std::string &fn,
    std::ifstream *file = nullptr);
```

### Inputs:
- **`fn`** — Path to the QRT file
- **`file`** *(optional)* — Pre-opened binary `std::ifstream`; pass `nullptr` to let the function open/close the file internally

### Returns:
- Populated `quadriga_lib::qrt_read_cache` struct with the following members:
  | Member             | Type         | Description                                                                                     |
  | ------------------ | ------------ | ----------------------------------------------------------------------------------------------- |
  | `version`          | `int`        | QRT file version                                                                                |
  | `no_orig`          | `unsigned`   | Number of origin (TX) positions                                                                 |
  | `no_cir`           | `unsigned`   | Number of CIRs per origin                                                                       |
  | `no_dest`          | `unsigned`   | Number of destinations (RX)                                                                     |
  | `no_freq`          | `unsigned`   | Number of frequency bands                                                                       |
  | `freq`             | `arma::fvec` | Frequency in GHz; `[no_freq]`                                                                   |
  | `cir_pos`          | `arma::fmat` | CIR positions; `[no_cir, 3]`                                                                    |
  | `cir_orientation`  | `arma::fmat` | CIR orientations (Euler); `[no_cir, 3]`                                                         |
  | `orig_pos_all`     | `arma::fmat` | Origin positions; `[no_orig, 3]`                                                                |
  | `orig_orientation` | `arma::fmat` | Origin orientations (Euler); `[no_orig, 3]`                                                     |
  | `orig_index`       | `arma::uvec` | Byte offsets from BOF to each origin data block; `[no_orig]`                                    |
  | `path_data_offset` | `arma::uvec` | Absolute offset to path_data_index array per origin; 0 if the TX slot is unwritten; `[no_orig]` |

---
<a id="quantize_delays"></a>
## quantize_delays
Map path delays to a fixed tap grid using two-tap power-weighted interpolation

Also in: [MATLAB](../mex/channel-functions.md#quantize_delays) · [Python](../python/channel-functions.md#quantize_delays)

- Each path delay is approximated by two adjacent taps with coefficients scaled by (1-delta)^alpha and delta^alpha,
  where delta is the fractional offset within the bin and alpha is `power_exponent`; this avoids discontinuities when 
  delays cross tap boundaries
- Use `power_exponent=1.0` for narrowband (linear interpolation) or `0.5` for wideband (incoherent power preservation)
- If all fractional offsets are below 0.01 or above 0.99, weight computation is skipped but tap-selection logic still applies
- Input `delay` may be per-antenna `[n_rx, n_tx, n_path_s]` or shared `[1, 1, n_path_s]`; shared delays are expanded 
  internally when `fix_taps` is 0 or 3, and output delays remain shared `[1, 1, n_taps]` when `fix_taps` is 1 or 2;
  per-antenna input always gives `[n_rx, n_tx, n_taps]` output delays (identical for all pairs when `fix_taps` is 1 or 2)
- `n_rx` and `n_tx` must be identical across all snapshots; `n_path_s` may differ per snapshot
- Delays must be non-negative, as for a tapped delay line, with a tolerance of `tap_spacing / 2`: delays in
  `(-tap_spacing / 2, 0)` are treated as 0, so all of their power goes to tap 0 (there is no tap below 0 to
  interpolate with; the error is below `tap_spacing / 2`, as for nearest-tap rounding). Delays at or below
  `-tap_spacing / 2` throw `std::invalid_argument`, as do NaN, Inf and delays of 2^31 taps or more
  (`delay / tap_spacing < 2^31`, e.g. 10.7 s at 5 ns or 12.4 h at 48 kHz)
- Relative delays (`use_absolute_delays = false` in [get_channels_spherical](channel-generation-functions.md#get_channels_spherical), [get_channels_multifreq](channel-generation-functions.md#get_channels_multifreq) and
  [get_channels_planar](channel-generation-functions.md#get_channels_planar)) are negative for element pairs that are closer to each other than the phase centres, by up
  to about (half the TX array size + half the RX array size) / propagation speed (both arrays endfire). They stay
  above `-tap_spacing / 2` only while these half sizes add up to less than `tap_spacing / 2` times the propagation
  speed: about 0.75 m for radio at 5 ns taps, but only about 3.6 mm for sound at 48 kHz taps, i.e. practically every
  acoustic array. In those cases use absolute delays (`use_absolute_delays = true`)
- Run time and memory depend on the number of paths and taps, not on the absolute delay values
- Outputs may alias inputs, also across arguments (e.g. `delay_quant == coeff_re`); all inputs are read before the
  outputs are written. The three outputs must be different objects.

### Declaration:
```cpp
void quadriga_lib::quantize_delays(
    const std::vector<arma::Cube<dtype>> *coeff_re,
    const std::vector<arma::Cube<dtype>> *coeff_im,
    const std::vector<arma::Cube<dtype>> *delay,
    std::vector<arma::Cube<dtype>> *coeff_re_quant,
    std::vector<arma::Cube<dtype>> *coeff_im_quant,
    std::vector<arma::Cube<dtype>> *delay_quant,
    dtype tap_spacing = (dtype)5.0e-9,
    arma::uword max_no_taps = 48,
    dtype power_exponent = (dtype)1.0,
    int fix_taps = 0);
```

### Inputs:
- **`coeff_re`** — Channel coefficients, real part; vector of length `n_snap`, each cube `[n_rx, n_tx, n_path_s]`
- **`coeff_im`** — Channel coefficients, imaginary part; same layout as `coeff_re`
- **`delay`** — Path delays in seconds; vector of length `n_snap`, each cube `[n_rx, n_tx, n_path_s]` or `[1, 1, n_path_s]`;
  non-negative with a tolerance of `tap_spacing / 2`, finite, below `2^31 * tap_spacing`
- **`tap_spacing`** *(optional)* — Delay bin spacing in seconds, positive and finite; 5 ns corresponds to 200 MHz sampling rate
- **`max_no_taps`** *(optional)* — Maximum number of output taps; 0 means unlimited
- **`power_exponent`** *(optional)* — Interpolation exponent alpha; 1.0 = narrowband, 0.5 = wideband power-preserving
- **`fix_taps`** *(optional)* — Delay grid sharing mode:
  | Value | Meaning                                                                                                                               |
  | ----- | ------------------------------------------------------------------------------------------------------------------------------------- |
  | 0     | Per tx-rx pair and snapshot; output delays `[n_rx, n_tx, n_taps]`                                                                     |
  | 1     | Single shared grid across all snapshots and tx-rx pairs; output delays `[1, 1, n_taps]` for shared delay input (`[n_rx, n_tx, n_taps]` for per-antenna input), identical for every snapshot |
  | 2     | Per snapshot; output delays `[1, 1, n_taps]` for shared delay input (`[n_rx, n_tx, n_taps]` for per-antenna input), but each snapshot has its own independent tap grid — taps do not align across snapshots |
  | 3     | Per tx-rx pair across all snapshots; output delays `[n_rx, n_tx, n_taps]`                                                             |

### Outputs:
- **`coeff_re_quant`** — Output coefficients, real part; vector of length `n_snap`, each cube `[n_rx, n_tx, n_taps]`
- **`coeff_im_quant`** — Output coefficients, imaginary part; same layout as `coeff_re_quant`
- **`delay_quant`** — Output delays in seconds; each cube `[n_rx, n_tx, n_taps]`, or `[1, 1, n_taps]` for shared delay input with
  `fix_taps` 1 or 2

### Example:
```cpp
std::vector<arma::Cube<double>> cre(2), cim(2), dl(2);
cre[0].set_size(1,1,3); cim[0].set_size(1,1,3); dl[0].set_size(1,1,3);
cre[1].set_size(1,1,2); cim[1].set_size(1,1,2); dl[1].set_size(1,1,2);
dl[0](0,0,1) = 12.5e-9; dl[0](0,0,2) = 33.4e-9; dl[1](0,0,1) = 10.0e-9;
std::vector<arma::Cube<double>> cre_q, cim_q, dl_q;
quadriga_lib::quantize_delays(&cre, &cim, &dl, &cre_q, &cim_q, &dl_q, 5.0e-9, 48, 1.0, 0);
```
