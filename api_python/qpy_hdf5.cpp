// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Channel data and processing
SECTION!*/

/*!SECTION_DESC
A **channel** describes the propagation between a transmit and a receive antenna array: for each snapshot, a set of
propagation paths, each with a complex coefficient per antenna pair and a delay. The functions of this section store
channels in files, turn them into frequency responses, prepare them for link-level simulation, compute their statistics
and show the paths in 3D. The channels themselves come from the functions of section "Channel generators" or from
[[get_channels_layout]].<br><br>

**What a channel holds** (a dict, keys in [[hdf5_read_channel]]):

- The positions and orientations of the transmitter and the receiver, per snapshot or fixed
- The complex **coefficients** `coeff` and the **delays** `delay` of each snapshot, `(n_rx, n_tx, n_path)`
- Optionally, antenna-independent path data: path gain, path length, polarization transfer matrix, angles, scatterer
  positions and the interaction points of ray-traced paths
- Unstructured metadata: named values of any type

**In Python:**

- The functions are in the submodule `quadriga_lib.channel`, e.g. `quadriga_lib.channel.hdf5_read_channel`
- A channel is a dict with the member names of the C++ class `channel`, with one complex `coeff` for `coeff_re` and
  `coeff_im`; several channels are a list of dicts. A member with one entry per snapshot is a list of arrays, one per
  snapshot, or, with `stack = True` of [[hdf5_read_channel]], one array with the snapshot on the last axis,
  zero-padded where the number of paths differs
- [[hdf5_read_channel]] and [[hdf5_write_channel]] take and return the unstructured data as a list `par` with one
  dict per channel, one key per dataset

**Typical workflow:**

1. Generate the channels, or read them: [[hdf5_read_channel]] from a channel file, [[qrt_file_read]] from the output
   of the QRT ray tracer
2. Optionally, map the delays to a tap grid with [[quantize_delays]], or join the segments of a track with
   [[crossfade_segments]]
3. Compute the frequency response at the subcarriers with [[freq_response]], or with [[freq_response_mf]] for
   coefficients sampled at several frequencies
4. Evaluate the channels: [[calc_delay_spread]], [[calc_angular_spread]], [[calc_rician_k_factor]], [[calc_xpr]]
5. Store them with [[hdf5_write_channel]], or show the paths with [[export_channel_obj]]

**Terms** (full list in the [Glossary](cpp_api.html#glossary)):

- **Snapshot**: one sample of the channel in time or position; `n_snap` snapshots, each with its own number of paths
- **Path**: one propagation path from the transmitter to the receiver, e.g. the LOS path, a reflection or a ray of a
  cluster
- **Coefficient**: the complex amplitude $a$ of one path for one antenna pair, with the antenna patterns, the
  polarization and the path gain
- **Delay**: the propagation delay $\tau$ of a path in s, per antenna pair `(n_rx, n_tx, n_path)` or shared by all
  pairs `(1, 1, n_path)`
- **CIR**: the channel impulse response of one snapshot, its coefficients and delays
- **Frequency response**: the channel matrix at subcarrier $k$, $H_k = \sum_l a_l \, e^{-j 2\pi f_k \tau_l}$, with
  $f_k$ the offset of the subcarrier from the carrier (the absolute frequency for `_mf`)
- **Multi-frequency channel** (suffix `_mf`): coefficients sampled at several frequencies, the frequency on the 4th
  axis, as [[channel.multifreq]] produces them
- **Slot**: one storage place of an HDF5 channel file, addressed by four indices `(ix, iy, iz, iw)`, e.g. Tx, Rx and
  frequency; the file holds a fixed grid of slots, its **storage layout**
- **Origin**, **destination**: a QRT file holds the paths from each origin (a Tx position) to a list of CIR positions; a
  destination (a receiver) owns a run of consecutive CIR positions, e.g. a track

**Functions by task:**

| Task | Functions |
| ---- | --------- |
| Store and read channels | [[hdf5_create_file]], [[hdf5_write_channel]], [[hdf5_read_channel]], [[hdf5_read_layout]], [[hdf5_reshape_layout]] |
| Store and read metadata | [[hdf5_write_dset]], [[hdf5_read_dset]], [[hdf5_read_dset_names]] |
| Ray-tracing files | [[qrt_file_parse]], [[qrt_file_read]] |
| Frequency response | [[freq_response]], [[freq_response_mf]] |
| Process | [[quantize_delays]], [[crossfade_segments]] |
| Statistics | [[calc_delay_spread]], [[calc_angular_spread]], [[calc_rician_k_factor]], [[calc_xpr]] |
| Show | [[export_channel_obj]] |

**Conventions.** Coefficients and delays of several snapshots are lists with one array per snapshot. Indices are
0-based. HDF5 files store the structured channel data in single precision.
SECTION_DESC!*/

/*!MD
# hdf5_create_file
Create a new HDF5 channel file with a given storage layout

- Creates a channel file with `nx · ny · nz · nw` empty slots; the slot count is fixed, [[hdf5_reshape_layout]] changes
  only the dimensions
- [[hdf5_write_channel]] and [[hdf5_write_dset]] create a missing file themselves, with the layout that just holds their
  slots; this function sets any other one
- The file is an HDF5 file (quadriga-lib channel format, version 3) that any HDF5 reader opens, e.g. h5py, MATLAB or
  HDFView. Change it only with the functions of this section: files changed by other HDF5 tools may be rejected

## Usage:
```
quadriga_lib.channel.hdf5_create_file( fn, nx, ny, nz, nw )
```

## Inputs:
- **`fn`** — File name, str or path-like; the file must not exist
- **`nx`** — Number of slots along x, at least 1; default: 65536
- **`ny`**, **`nz`**, **`nw`** — Number of slots along y, z and w, each at least 1; at most 4294967295 (2^32 - 1)
  slots in total; default: 1

## See also:
- [[hdf5_write_channel]] (writes channels into slots)
- [[hdf5_read_layout]] (reads the layout and the occupied slots)
MD!*/

void hdf5_create_file(const py::handle &fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    // C++ throws if the file exists
    quadriga_lib::hdf5_create_file(qd_python_path(fn, "fn"), nx, ny, nz, nw);
}

/*!MD
# hdf5_read_channel
Read channel objects from one or several slots of an HDF5 channel file

- Returns the channels of the selected slots that hold data as a list of dicts `chan`, and their unstructured data as
  a list of dicts `par` of the same length
- `ix`, `iy`, `iz` and `iw` select the slots along each dimension; the selection is every combination, in
  column-major order (`ix` fastest), and empty slots are skipped ([[hdf5_read_layout]] tells which slots hold data)
- `i_snap` reads only the given snapshots of every selected slot (see Notes)

## Usage:
```
chan, par = quadriga_lib.channel.hdf5_read_channel( fn, ix, iy, iz, iw, *, i_snap, stack )
```

## Inputs:
- **`fn`** — File name of an existing channel file, str or path-like
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot indices along x, y, z and w, 0-based, inside the layout; int or
  array; default: `None` (all slots of the dimension)
- **`i_snap`** — Snapshot indices, 0-based, any order, repeats allowed, below `n_snap` of every selected slot that
  has snapshots; `(n_out,)`; default: `None` (all snapshots)
- **`stack`** — `False`: members with one entry per snapshot are lists of arrays; `True`: one array with the
  snapshot on the last axis, zero-padded to the largest number of paths; default: `False`

## Outputs:
- **`chan`** — Channels, one per selected slot that holds data; list of dicts with the keys:<br><br>
  | Key | Description | Shape `stack = False` | Shape `stack = True` |
  | --- | ----------- | --------------------- | -------------------- |
  | `name` | Channel name | str | str |
  | `tx_pos`, `rx_pos` | Tx and Rx positions in m | `(3, 1)` (fixed) or `(3, n_snap)` | as left |
  | `tx_orientation`, `rx_orientation` | Tx and Rx orientations (bank, tilt, heading) in rad | `(3, 1)` (fixed) or `(3, n_snap)` | as left |
  | `coeff` | Coefficients; complex128 | list of `(n_rx, n_tx, n_path)` | `(n_rx, n_tx, n_path, n_snap)` |
  | `delay` | Delays in s | list of `(n_rx, n_tx, n_path)` or `(1, 1, n_path)` | `(n_rx, n_tx, n_path, n_snap)` or `(1, 1, n_path, n_snap)` |
  | `path_gain` | Path gain without antennas, linear | list of `(n_path,)` | `(n_path, n_snap)` |
  | `path_length` | Path length in m | list of `(n_path,)` | `(n_path, n_snap)` |
  | `polmat` | Polarization transfer matrix (Glossary) | list of `(8, n_path)` | `(8, n_path, n_snap)` |
  | `path_angles` | AOD, EOD, AOA, EOA in rad | list of `(n_path, 4)` | `(n_path, 4, n_snap)` |
  | `fbs_pos`, `lbs_pos` | First- and last-bounce scatterer positions in m | list of `(3, n_path)` | `(3, n_path, n_snap)` |
  | `n_interact` | Number of interaction points per path, 0 = LOS; int64 | list of `(n_path,)` | `(n_path, n_snap)` |
  | `interact_coord` | Interaction points of all paths, concatenated | list of `(3, sum(n_interact))` | `(3, max(sum(n_interact)), n_snap)` |
  | `center_freq` | Center frequency in Hz | `(1,)` or `(n_snap,)` | as left |
  | `initial_position` | Reference snapshot, 0-based | int | int |

  `n_path` may differ between the snapshots; `stack = True` zero-pads to the largest one. A key exists if the channel
  has the member
- **`par`** — Unstructured data, one dict per entry of `chan` (empty if the slot has none); keys are the dataset
  names without the prefix `par_`, values as [[hdf5_read_dset]] returns them

## Notes:
- **Snapshot selection**: members with one entry per snapshot are sliced, also positions, orientations and
  `center_freq` stored per snapshot; values stored once and the unstructured data are kept. A slot without snapshots
  (only unstructured data) ignores `i_snap`
- **Round trip** with [[hdf5_write_channel]]: the values come back in single precision; empty orientations come back as
  `(3, 1)` zeros
- **Unstructured data**: the datasets of the slot with the prefix `par_`; datasets of another prefix are not read
- **Errors**: a missing file, an index outside the layout, an index in `i_snap` beyond the snapshots of a slot, or a
  file that is not a quadriga-lib channel file; files changed by other HDF5 tools may be rejected

## See also:
- [[hdf5_read_layout]] (finds the occupied slots)
- [[hdf5_write_channel]] (writes channels)
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
    const auto channels = quadriga_lib::hdf5_read_channel<double>(fn_s, slots, i_snap_a);

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
Read an unstructured dataset from a slot of an HDF5 channel file

- Reads the dataset `prefix + name` of slot `(ix, iy, iz, iw)`, e.g. `par_carrier_frequency`
- The type follows the stored dataset: str for a string, a Python number for a scalar, an array `(n,)` for a 1D
  dataset (also with one element), else an array of the stored shape; dtypes float32, float64, int32, uint32, int64,
  uint64

## Usage:
```
dset = quadriga_lib.channel.hdf5_read_dset( fn, name, ix, iy, iz, iw, *, prefix )
```

## Inputs:
- **`fn`** — File name of an existing channel file, str or path-like
- **`name`** — Dataset name without the prefix, e.g. `'carrier_frequency'`; str
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot index along x, y, z and w, 0-based; default: 0
- **`prefix`** — Prefix of the dataset name; `''` reads `name` as given; str; default: `'par_'`

## Output:
- **`dset`** — The dataset; `None` if the slot is empty or outside the layout or has no such dataset

## See also:
- [[hdf5_read_dset_names]] (lists the datasets of a slot)
- [[hdf5_write_dset]] (writes a dataset)
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
Read the names of the unstructured datasets of a slot of an HDF5 channel file

- Lists the datasets of slot `(ix, iy, iz, iw)` whose names start with `prefix`, without the prefix, in the order
  they were written

## Usage:
```
names = quadriga_lib.channel.hdf5_read_dset_names( fn, ix, iy, iz, iw, *, prefix )
```

## Inputs:
- **`fn`** — File name of an existing channel file, str or path-like
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot index along x, y, z and w, 0-based; default: 0
- **`prefix`** — Prefix of the listed names; `''` lists all datasets with their full names; str; default: `'par_'`

## Output:
- **`names`** — Dataset names; list of str, empty if the slot is empty or outside the layout or has none

## See also:
- [[hdf5_read_dset]] (reads a dataset by name)
- [[hdf5_write_dset]] (writes a dataset)
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
Read the storage layout of an HDF5 channel file

- Returns the dimensions of the slot grid and the slots that hold data
- A missing file gives `[0, 0, 0, 0]`, so the function also tests whether a file exists; a file that is not a channel
  file raises an error

## Usage:
```
storage_dims, has_data = quadriga_lib.channel.hdf5_read_layout( fn )
```

## Input:
- **`fn`** — File name, str or path-like

## Outputs:
- **`storage_dims`** — Layout `[nx, ny, nz, nw]`; int64; `(4,)`
- **`has_data`** — 1 for the slots that were written, also with an empty channel, else 0; uint32; `(nx, ny, nz, nw)`

## See also:
- [[hdf5_create_file]] (creates a file with a given layout)
- [[hdf5_reshape_layout]] (changes the dimensions)
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
Reshape the storage layout of an existing HDF5 channel file

- Sets new dimensions `[nx, ny, nz, nw]` with the slot count `nx · ny · nz · nw` of the file
- The data stays in place: the slot with the linear index `ix + nx * (iy + ny * (iz + nz * iw))` keeps its channel,
  which gets the slot index of that position in the new layout

## Usage:
```
quadriga_lib.channel.hdf5_reshape_layout( fn, nx, ny, nz, nw )
```

## Inputs:
- **`fn`** — File name of an existing channel file, str or path-like
- **`nx`** — New number of slots along x
- **`ny`**, **`nz`**, **`nw`** — New number of slots along y, z and w; default: 1

## See also:
- [[hdf5_create_file]] (sets the layout of a new file)
- [[hdf5_read_layout]] (reads the layout)
MD!*/

void hdf5_reshape_layout(const py::handle &fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    const std::string fn_s = qd_python_path(fn, "fn");

    quadriga_lib::hdf5_reshape_layout(fn_s, nx, ny, nz, nw);
}

/*!MD
# hdf5_write_channel
Write channel objects to one or several slots of an HDF5 channel file

- Stores `chan[k]` with the unstructured data `par[k]` in slot `(ix[k], iy[k], iz[k], iw[k])`; a scalar index
  applies to all channels
- A missing file is created with the layout that just holds the slots (the largest index + 1 per dimension);
  [[hdf5_create_file]] sets another layout
- A slot that holds data is replaced completely: its old channel and all its unstructured datasets, also those added
  with [[hdf5_write_dset]], are removed
- All inputs are checked before anything is written, so a failing check leaves the file unchanged

## Usage:
```
code = quadriga_lib.channel.hdf5_write_channel( fn, chan, par, ix, iy, iz, iw )
```

## Inputs:
- **`fn`** — File name, str or path-like
- **`chan`** — Channels to write, valid; a channel dict or a non-empty list of `n_chan` dicts, keys in
  [[hdf5_read_channel]]. Members with one entry per snapshot as a list of arrays or as one array with the snapshot on
  the last axis (`stack = True` of [[hdf5_read_channel]]), detected per key; the coefficients as complex `coeff` or as
  real `coeff_re` and `coeff_im`, not both
- **`par`** — Unstructured data; a dict or a list of `n_chan` dicts: a key is a dataset name (stored with the prefix
  `par_`), the values follow the type rules of [[hdf5_write_dset]], `None` values are skipped; default: `None`
  (none)
- **`ix`** — Slot index along x, 0-based; int or `n_chan` entries; default: `None` (`0 ... n_chan - 1`)
- **`iy`**, **`iz`**, **`iw`** — Slot indices along y, z and w, 0-based; int or `n_chan` entries; default:
  `None` (0)

## Output:
- **`code`** — `0` if the slot was empty, `1` if its data was replaced; int64; `(n_chan,)`

## Notes:
- A channel without data and without `par` entries replaces the data of an occupied slot (it then reads as an empty
  channel, and [[hdf5_read_layout]] reports it as written); it leaves an empty slot as it is and does not create a
  missing file
- The space of replaced data is not reused, so the file does not shrink; copying the channels into a new file reclaims
  it
- Files are not locked: one writer per file at a time, also for [[hdf5_write_dset]] and [[hdf5_reshape_layout]].
  Reading is safe while no one writes

## See also:
- [[hdf5_read_channel]] (reads the channels back)
- [[hdf5_create_file]] (creates a file with a given layout)
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
    const arma::s32_vec code = quadriga_lib::hdf5_write_channel(fn_s, channels, slots);
    return qd_python_copy2numpy<int, py::ssize_t>(&code);
}

/*!MD
# hdf5_write_dset
Write an unstructured dataset to a slot of an HDF5 channel file

- Adds the dataset `prefix + name` to slot `(ix, iy, iz, iw)`; everything else stored in the slot stays. An empty
  slot gets a channel without structured data
- A missing file is created with the layout `(ix + 1, iy + 1, iz + 1, iw + 1)`; [[hdf5_create_file]] sets another
  layout
- A dataset is written once: a dataset of the same name in the slot raises an error. A later [[hdf5_write_channel]] to
  the slot removes it with the rest of the slot
- Invalid names and types raise an error before the file is changed
- [[hdf5_read_channel]] returns the datasets with the prefix `par_` as the unstructured data `par`; [[hdf5_read_dset]]
  reads each one back

## Usage:
```
quadriga_lib.channel.hdf5_write_dset( fn, name, data, ix, iy, iz, iw, *, prefix )
```

## Inputs:
- **`fn`** — File name, str or path-like
- **`name`** — Dataset name without the prefix, not empty; `prefix + name` holds only letters, digits and `_`, at
  most 65000 characters; str
- **`data`** — A str; a Python float (stored as float64), int or bool (stored as int64, a bool as 0 or 1); a numpy
  scalar, or a numpy array with 0 to 3 dimensions in any memory layout and byte order; dtypes float32, float64, int32,
  uint32, int64, uint64, other dtypes (e.g. int16, bool, complex) raise an error. A 1D array is stored as a vector and
  comes back with shape `(n,)`, also with one element; 2D and 3D arrays keep their shape, also `(n, 1)` and `(1, 1)`
- **`ix`**, **`iy`**, **`iz`**, **`iw`** — Slot index along x, y, z and w, 0-based; inside the layout of an
  existing file; default: 0
- **`prefix`** — Prefix of the dataset name; `''` stores `name` without a prefix; str; default: `'par_'`

## See also:
- [[hdf5_read_dset]] (reads a dataset by name)
- [[hdf5_read_dset_names]] (lists the datasets of a slot)
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
