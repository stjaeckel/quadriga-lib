// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# hdf5_create_file
Create a new HDF5 channel file with a custom storage layout

- Initializes a new HDF5 file for storing channel data
- Defines a 4D storage layout `(nx, ny, nz, nw)`; each index combination maps to one channel slot
- Typical dimension mapping: nx = BS, ny = UE, nz = frequency, nw = scenario/repetition
- The total slot count is fixed at creation; the layout can be reshaped with [[hdf5_reshape_layout]]
- Raises an error if the target file already exists (also one that cannot be read); delete it first to recreate it
- Files can be read by any HDF5 tool but must be modified only with quadriga-lib

## Usage:
```
quadriga_lib.channel.hdf5_create_file( fn, nx, ny, nz, nw )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file to create
- **`nx`** — Number of elements on the x-dimension; default: 65536
- **`ny`** — Number of elements on the y-dimension; default: 1
- **`nz`** — Number of elements on the z-dimension; default: 1
- **`nw`** — Number of elements on the w-dimension; default: 1

## See also:
- [[hdf5_write_channel]] (for writing channel data)
- [[hdf5_write_dset]] (for writing arbitrary unstructured data)
- [[hdf5_read_layout]] (for reading the layout)
MD!*/

void hdf5_create_file(const py::handle &fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    // C++ throws if the file exists
    quadriga_lib::hdf5_create_file(qd_python_path(fn, "fn"), nx, ny, nz, nw);
}

/*!MD
# hdf5_read_channel
Reads channel data from HDF5 files

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

## Usage:
```
chan, par = quadriga_lib.channel.hdf5_read_channel( fn, ix, iy, iz, iw, *, i_snap, stack )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file
- **`ix`** — 0-based slot indices along dimension X; scalar or vector; default: `0 ... nx-1`
- **`iy`** — 0-based slot indices along dimension Y; scalar or vector; default: `0 ... ny-1`
- **`iz`** — 0-based slot indices along dimension Z; scalar or vector; default: `0 ... nz-1`
- **`iw`** — 0-based slot indices along dimension W; scalar or vector; default: `0 ... nw-1`
- **`i_snap`** — Snapshot indices to read; 0-based; default: all snapshots. Applies to every selected slot; raises an error if an index exceeds the number of snapshots of a selected slot.
- **`stack`** — If `True`, stack snapshots; default: `False` (return as list)

## Outputs:
- **`chan`** — List of dicts with the following keys:<br><br>
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
  | `polmat`           | Polarization transfer matrix, real, rows as in [[channel.planar]]        | list of `(8, n_path_s)`           | `(8, n_path, n_snap)`                |
  | `path_angles`      | Departure and arrival angles (AOD, EOD, AOA, EOA) in rad                 | list of `(n_path_s, 4)`           | `(n_path, 4, n_snap)`                |
  | `fbs_pos`          | First-bounce scatterer positions                                         | list of `(3, n_path_s)`           | `(3, n_path, n_snap)`                |
  | `lbs_pos`          | Last-bounce scatterer positions                                          | list of `(3, n_path_s)`           | `(3, n_path, n_snap)`                |
  | `n_interact`       | Number of interaction points per path; int64                             | list of `(n_path_s,)`             | `(n_path, n_snap)`                   |
  | `interact_coord`   | Interaction coordinates                                                  | list of `(3, sum(n_interact_s))`  | `(3, max(sum(n_interact)), n_snap)`  |
  | `center_freq`      | Center Frequency in Hz                                                   | `(n_snap,)` or `(1,)`             | `(n_snap,)` or `(1,)`                |
  | `initial_position` | Reference snapshot (start of the current segment); 0-based               | int                               | int                                  |

- **`par`** — List of dicts with the unstructured data, one per entry of `chan` (empty dict if the slot has none).

## See also:
- [[hdf5_read_layout]] (for reading the layout in the file)
- [[hdf5_write_channel]] (for writing channel data)
- [[hdf5_read_dset]] (for reading individual unstructured datasets)
- [[hdf5_write_dset]] (for writing individual unstructured datasets)
MD!*/

py::tuple hdf5_read_channel(const py::handle &fn,
                            py::handle ix, py::handle iy, py::handle iz, py::handle iw,
                            py::handle i_snap, py::handle stack)
{
    const std::string fn_s = qd_python_path(fn, "fn");
    const bool stack_b = qd_python_flag(stack, false, "stack");

    // Occupied slots of the selection (None or empty selects the full extent), then their channels
    arma::u32_vec channel_id;
    const arma::u32_vec dims = quadriga_lib::hdf5_read_layout(fn_s, &channel_id);
    const arma::u32_mat slots = qd_hdf5_select_slots(fn_s, dims, channel_id,
                                                     qd_python_numpy2arma_Col<unsigned>(ix, false, false, "ix"),
                                                     qd_python_numpy2arma_Col<unsigned>(iy, false, false, "iy"),
                                                     qd_python_numpy2arma_Col<unsigned>(iz, false, false, "iz"),
                                                     qd_python_numpy2arma_Col<unsigned>(iw, false, false, "iw"));
    const arma::uvec i_snap_a = qd_python_numpy2arma_Col<arma::uword>(i_snap, false, false, "i_snap");
    const auto channels = quadriga_lib::hdf5_read_channel_vec<double>(fn_s, slots, i_snap_a);

    py::list chan;
    py::list par_list;
    for (const auto &channel : channels)
    {
        // A slot flagged occupied but holding nothing is skipped
        if (channel.empty() && channel.par_names.empty())
            continue;

        // Validate structured data (par-only channels are not validated)
        if (!channel.empty())
        {
            const std::string err = channel.is_valid();
            if (!err.empty())
                throw std::runtime_error(err);
        }

        chan.append(qd_python_channel2dict(channel, arma::uvec(), false, stack_b));

        py::dict par;
        for (size_t k = 0; k < channel.par_names.size(); ++k)
            par[channel.par_names[k].c_str()] = qd_python_any2numpy(channel.par_data[k]);
        par_list.append(par);
    }

    return py::make_tuple(chan, par_list);
}

/*!MD
# hdf5_read_dset
Read a single unstructured dataset from an HDF5 file

- Reads one user-defined dataset from the slot addressed by the 0-based indices `(ix, iy, iz, iw)`
- The dataset is looked up under `prefix + name` (default prefix `par_`)
- The returned type and shape are defined by the dataset's HDF5 dataspace: a scalar dataset gives a Python
  number, a vector a numpy array of shape `(n,)` (also with a single element), a matrix or 3D array a numpy array
  of the stored shape
- Returns `None` if the dataset does not exist at the requested slot
- Errors if the file does not exist or is not a quadriga-lib channel file
- Supported types: str, scalar, vector, 2D array, and 3D array

## Usage:
```
dset = quadriga_lib.channel.hdf5_read_dset( fn, name, ix, iy, iz, iw, *, prefix )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file
- **`name`** — Dataset name without the prefix, e.g. `'carrier_frequency'`; str
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`prefix`** — Prefix prepended to `name` when looking up the dataset; str; default: `'par_'`

## Output:
- **`dset`** — Dataset contents; type and shape are defined by the HDF5 dataspace; `None` if the dataset is missing

## See also:
- [[hdf5_read_dset_names]] (for reading names of already written datasets)
- [[hdf5_write_dset]] (for writing individual unstructured datasets)
- [[hdf5_read_channel]] (for reading structured channel data)
MD!*/

py::object hdf5_read_dset(const py::handle &fn,
                          const std::string &name,
                          unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                          const std::string &prefix)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    std::any dset = quadriga_lib::hdf5_read_dset(fn_s, name, ix, iy, iz, iw, prefix);

    // Convert to numpy array / scalar / str; None if the dataset is missing
    return qd_python_any2numpy(dset);
}

/*!MD
# hdf5_read_dset_names
Read the names of unstructured datasets from an HDF5 file

- Lists all unstructured datasets stored at the slot addressed by the 0-based indices `(ix, iy, iz, iw)`
- Lists the datasets whose names start with `prefix` (default `par_`; `''` lists all); the returned names have the
  prefix stripped
- Returns an empty list if no unstructured datasets are present at the slot
- Errors if the file does not exist or is not a quadriga-lib channel file

## Usage:
```
names = quadriga_lib.channel.hdf5_read_dset_names( fn, ix, iy, iz, iw, *, prefix )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`prefix`** — Prefix used to identify unstructured datasets; `''` lists all datasets; str; default: `'par_'`

## Output:
- **`names`** — Names of the unstructured datasets at the slot whose names start with `prefix`, with the prefix
  stripped, in creation order; list of str

## See also:
- [[hdf5_read_dset]] (for reading individual unstructured datasets)
- [[hdf5_write_dset]] (for writing individual unstructured datasets)
MD!*/

py::list hdf5_read_dset_names(const py::handle &fn,
                              unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                              const std::string &prefix)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    return qd_python_copy2list(quadriga_lib::hdf5_read_dset_names(fn_s, ix, iy, iz, iw, prefix));
}

/*!MD
# hdf5_read_layout
Read the storage layout of channel data inside an HDF5 file

- Returns the dimensions of the 4D channel-slot grid stored inside an HDF5 file
- Also reports which slots already hold data, so free slots can be found without scanning the file
- Returns `(0, 0, 0, 0)` dimensions if the file does not exist
- Raises an error if the file exists but is not a quadriga-lib channel file

## Usage:
```
storage_dims, has_data = quadriga_lib.channel.hdf5_read_layout( fn )
```

## Input:
- **`fn`** — Filename of the HDF5 file

## Outputs:
- **`storage_dims`** — Size of the storage space `[nx, ny, nz, nw]`; `(4,)`; int64
- **`has_data`** — Slot occupancy mask; 1 where a slot holds data, 0 otherwise; uint32; `(nx, ny, nz, nw)`

## See also:
- [[hdf5_create_file]] (for creating a file with a custom storage layout)
- [[hdf5_read_channel]] (for reading channel data)
MD!*/

py::tuple hdf5_read_layout(const py::handle &fn)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    arma::Col<unsigned> channel_id;
    arma::Col<unsigned> storage_space = quadriga_lib::hdf5_read_layout(fn_s, &channel_id);

    const unsigned nx = storage_space.at(0), ny = storage_space.at(1),
                   nz = storage_space.at(2), nw = storage_space.at(3);

    if (channel_id.n_elem != (arma::uword)nx * ny * nz * nw)
        throw std::runtime_error("Corrupted storage index.");

    // storage_dims: 4-element vector [nx, ny, nz, nw] (copied integer output: int64)
    auto storage_dims = qd_python_copy2numpy<unsigned, py::ssize_t>(&storage_space);

    // has_data: 4D occupancy mask (1 where a slot holds data, 0 otherwise)
    std::vector<arma::Cube<unsigned>> has_data_cubes;
    auto has_data = qd_python_init_output(nx, ny, nz, nw, &has_data_cubes);
    for (unsigned w = 0; w < nw; ++w)
    {
        unsigned *dst = has_data_cubes[w].memptr();
        const unsigned *src = channel_id.memptr() + (size_t)w * nx * ny * nz;
        for (size_t i = 0, n = (size_t)nx * ny * nz; i < n; ++i)
            dst[i] = (src[i] != 0u) ? 1u : 0u;
    }

    return py::make_tuple(storage_dims, has_data);
}

/*!MD
# hdf5_reshape_layout
Reshapes the storage layout inside an existing HDF5 file

- Changes the 4D slot grid `(nx, ny, nz, nw)` of an existing HDF5 channel file
- The total number of slots (`nx · ny · nz · nw`) must match the original layout
- Only the dimension metadata is updated; stored channel data is not moved
- Errors if the file does not exist or is not a quadriga-lib channel file
- One writer per file at a time; files are not locked

## Usage:
```
quadriga_lib.channel.hdf5_reshape_layout( fn, nx, ny, nz, nw )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file
- **`nx`** — Number of elements on the x-dimension
- **`ny`** — Number of elements on the y-dimension; default: 1
- **`nz`** — Number of elements on the z-dimension; default: 1
- **`nw`** — Number of elements on the w-dimension; default: 1
MD!*/

void hdf5_reshape_layout(const py::handle &fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    quadriga_lib::hdf5_reshape_layout(fn_s, nx, ny, nz, nw);
}

/*!MD
# hdf5_write_channel
Write one or more channel objects to an HDF5 file

- Writes a list of channel dicts into 4D storage slots (one slot per list entry)
- `chan` may also be a single dict, in which case one channel is written
- Optional unstructured data (`par`) can be passed as a matching list of dicts (or a single dict)
- Creates the file if it does not exist, with the layout that just holds all slots (largest index + 1 per
  dimension); use [[hdf5_create_file]] for another layout
- All inputs are checked before anything is written: every slot inside the layout of an existing file, no slot
  twice, all channels valid
- A slot that already contains data is replaced completely: its old structured data and all its unstructured
  datasets (also those written with [[hdf5_write_dset]]) are removed; `code` shows which slots were replaced.
  The file does not shrink; copy the channels into a new file to reclaim the space of the removed data
- One writer per file at a time; files are not locked
- Structured data is stored in single precision regardless of the input precision
- Unstructured datasets retain their numpy dtype and shape; supported types as in [[hdf5_write_dset]]
- Each scalar index input is broadcast to all channels; each vector index must have one entry per channel
- Channel dict field layout matches [[hdf5_read_channel]]
- Per-snapshot fields accept the list model (list of arrays) or the stack model
  (single array with the snapshot index as the last axis), detected per field
- Coefficients may be passed as a complex `coeff`, or as separate real `coeff_re` and `coeff_im` (the two forms are mutually exclusive)
- Slot indices are 0-based

## Usage:
```
code = quadriga_lib.channel.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file
- **`chan`** — Structured channel data; a channel dict or a list of channel dicts; field layout matches [[hdf5_read_channel]]
- **`par`** — Unstructured data; a dict or a list of dicts with the same number of entries as `chan`. Dict keys become HDF5 dataset
  names per slot; `None` values are skipped. Pass `None` to disable; default: `None`
- **`ix`** — 0-based slot indices along dimension X; scalar or vector of length `len(chan)`; default: `0 ... len(chan)-1`
- **`iy`** — 0-based slot indices along dimension Y; scalar or vector of length `len(chan)`; default: 0
- **`iz`** — 0-based slot indices along dimension Z; scalar or vector of length `len(chan)`; default: 0
- **`iw`** — 0-based slot indices along dimension W; scalar or vector of length `len(chan)`; default: 0

## Output:
- **`code`** — Per channel: `0` if the slot was empty, `1` if existing data was replaced; `(len(chan),)`; int64

## See also:
- [[hdf5_create_file]] (for creating a file with a custom storage layout)
- [[hdf5_reshape_layout]] (to change the layout later)
- [[hdf5_read_channel]] (for reading channel data)
MD!*/

py::array_t<py::ssize_t> hdf5_write_channel(const py::handle &fn,
                                         py::handle chan,
                                         py::handle par,
                                         py::handle ix,
                                         py::handle iy,
                                         py::handle iz,
                                         py::handle iw)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    std::vector<quadriga_lib::channel<double>> channels = qd_python_list2channel(chan, false); // C++ validates, naming the entry
    const arma::uword n_chan = (arma::uword)channels.size();
    if (n_chan == 0)
        throw std::invalid_argument("Input 'chan' must contain at least one channel.");

    std::vector<py::dict> par_vec;
    if (!par.is_none())
    {
        if (py::isinstance<py::dict>(par))
            par_vec.emplace_back(py::reinterpret_borrow<py::dict>(par));
        else if (py::isinstance<py::list>(par))
        {
            for (py::handle item : py::reinterpret_borrow<py::list>(par))
            {
                if (!py::isinstance<py::dict>(item))
                    throw std::invalid_argument("Input 'par' must be a dict, a list of dicts, or None.");
                par_vec.emplace_back(py::reinterpret_borrow<py::dict>(item));
            }
        }
        else
            throw std::invalid_argument("Input 'par' must be a dict, a list of dicts, or None.");

        if ((arma::uword)par_vec.size() != n_chan)
            throw std::invalid_argument("Inputs 'par' and 'chan' must have the same number of entries.");
    }

    // Slot index: scalar, vector or None; 'ix' defaults to 0 ... n_chan-1, the others to 0
    auto parse_idx = [&](py::handle h, const char *name, bool is_x) -> arma::Col<unsigned>
    {
        arma::Col<unsigned> v = qd_python_numpy2arma_Col<unsigned>(h, false, false, name);
        if (v.is_empty())
        {
            if (is_x)
            {
                v.set_size(n_chan);
                for (arma::uword i = 0; i < n_chan; ++i)
                    v.at(i) = (unsigned)i;
            }
            else
                v = arma::Col<unsigned>(1, arma::fill::zeros);
        }
        if (v.n_elem != 1 && v.n_elem != n_chan)
            throw std::invalid_argument(std::string("Input '") + name +
                                        "' must be a scalar or a vector of length len(chan).");
        return v;
    };

    const arma::Col<unsigned> ix_vec = parse_idx(ix, "ix", true);
    const arma::Col<unsigned> iy_vec = parse_idx(iy, "iy", false);
    const arma::Col<unsigned> iz_vec = parse_idx(iz, "iz", false);
    const arma::Col<unsigned> iw_vec = parse_idx(iw, "iw", false);

    // Attach the unstructured 'par' data and resolve the slot of each channel (scalar broadcasting)
    arma::u32_mat slots(4, n_chan);
    for (arma::uword k = 0; k < n_chan; ++k)
    {
        quadriga_lib::channel<double> &c = channels[k];
        if (!par_vec.empty())
            for (auto item : par_vec[k])
            {
                if (item.second.is_none()) // empty / missing field -> skip
                    continue;
                std::string field_name = py::cast<std::string>(item.first);
                c.par_names.push_back(field_name);
                c.par_data.push_back(qd_python_anycast(item.second, field_name));
            }

        slots.at(0, k) = ix_vec.n_elem == 1 ? ix_vec.at(0) : ix_vec.at(k);
        slots.at(1, k) = iy_vec.n_elem == 1 ? iy_vec.at(0) : iy_vec.at(k);
        slots.at(2, k) = iz_vec.n_elem == 1 ? iz_vec.at(0) : iz_vec.at(k);
        slots.at(3, k) = iw_vec.n_elem == 1 ? iw_vec.at(0) : iw_vec.at(k);
    }

    // Write all channels (C++ checks all inputs first and creates a missing file)
    const arma::s32_vec code = quadriga_lib::hdf5_write_channel_vec(fn_s, channels, slots);
    return qd_python_copy2numpy<int, py::ssize_t>(&code);
}

/*!MD
# hdf5_write_dset
Write a single unstructured dataset to an HDF5 file

- Dataset is stored under `prefix + name` at slot `(ix, iy, iz, iw)`
- `name` must not be empty; `prefix + name` must contain only alphanumeric characters and underscores
- Creates the file with the layout `(ix + 1, iy + 1, iz + 1, iw + 1)` if it does not exist; use [[hdf5_create_file]]
  for another layout
- A dataset of the same name at the same slot is not overwritten; an error is thrown instead
- The dataset is added to the slot; everything else stored there is kept. A later [[hdf5_write_channel]] to the
  same slot replaces the slot completely, including the datasets written with this function
- Supported types: str; Python float (stored as double), int and bool (stored as int64; a bool is stored as the
  integer 0 or 1 and reads back as an int); numpy scalars and 0-d arrays; numpy arrays with 1, 2 or 3 dimensions
  in any memory layout and byte order; numeric element types: float32, float64, int32, uint32, int64, uint64.
  Other types (e.g. int16, numpy bool, complex) raise an error
- 1D arrays are stored as vectors and read back with shape `(n,)`, also with a single element (not as a scalar);
  2D and 3D arrays keep their shape, also `(n, 1)` and `(1, 1)`
- One writer per file at a time; files are not locked

## Usage:
```
quadriga_lib.channel.hdf5_write_dset( fn, name, data, ix, iy, iz, iw, *, prefix )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file
- **`name`** — Dataset name without prefix, e.g. `'carrier_frequency'`; alphanumeric and underscores only; str
- **`data`** — Data to be written; type must be supported (see above); cannot be empty
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`prefix`** — Prefix prepended to `name` in the HDF5 file; alphanumeric and underscores only; str; default: `'par_'`

## See also:
- [[hdf5_read_dset_names]] (for reading names of already written datasets)
- [[hdf5_read_dset]] (for reading individual unstructured datasets)
MD!*/

void hdf5_write_dset(const py::handle &fn,
                     const std::string &name,
                     const py::handle data,
                     unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                     const std::string &prefix)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    // C++ checks the name and the slot and creates a missing file
    quadriga_lib::hdf5_write_dset(fn_s, name, qd_python_anycast(data, name), ix, iy, iz, iw, prefix);
}
