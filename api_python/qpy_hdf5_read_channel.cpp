// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Channel functions
SECTION!*/

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
- `snap` selects a subset of snapshots; if `None`, all snapshots are read

## Usage:
```
chan, par = quadriga_lib.channel.hdf5_read_channel( fn, ix, iy, iz, iw, snap, stack )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — 0-based slot indices along dimension X; scalar or vector; default: `0 ... nx-1`
- **`iy`** — 0-based slot indices along dimension Y; scalar or vector; default: `0 ... ny-1`
- **`iz`** — 0-based slot indices along dimension Z; scalar or vector; default: `0 ... nz-1`
- **`iw`** — 0-based slot indices along dimension W; scalar or vector; default: `0 ... nw-1`
- **`snap`** — Snapshot indices to read; 0-based; default: all snapshots. Only allowed when the total selection is a single slot.
- **`stack`** — If `True`, stack snapshots. Default: `False` (return as list)

## Outputs:
- **`chan`** — List of dicts with the following keys:<br><br>
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
  | `no_interact`      | Number of interaction points per path; uint32                            | list of `(n_path_s,)`             | `(n_path, n_snap)`                   |
  | `interact_coord`   | Interaction coordinates                                                  | list of `(3, sum(no_interact_s))` | `(3, max(sum(no_interact)), n_snap)` |
  | `center_frequency` | Center Frequency in Hz                                                   | `(n_snap,)` or scalar             | `(n_snap,)` or scalar                |
  | `initial_position` | Index of reference position; 1-based                                     | int32, scalar                     | int32, scalar                        |

- **`par`** — Dict (single slot) or list of dicts (multiple slots) containing the unstructured data in the file.

## See also:
- [[hdf5_read_layout]] (for reading the layout in the file)
- [[hdf5_write_channel]] (for writing channel data)
- [[hdf5_read_dset]] (for reading individual unstructured datasets)
- [[hdf5_write_dset]] (for writing individual unstructured datasets)
MD!*/

// Resolve a slot-index argument (scalar / 1-D array / list / None) to a 0-based
// index vector. None or empty selects the full dimension (0 ... n-1).
static arma::u32_vec parse_idx(const py::handle &h, unsigned n, const char *name)
{
    arma::u32_vec idx = qd_python_numpy2arma_Col<unsigned>(h, false);

    if (idx.is_empty()) // None / empty -> full dimension
    {
        idx.set_size(n);
        for (unsigned i = 0; i < n; ++i)
            idx.at(i) = i;
        return idx;
    }

    for (const auto v : idx)
        if (v >= n)
            throw std::invalid_argument(std::string("Slot index '") + name + "' is out of bound.");

    return idx;
}

py::tuple hdf5_read_channel(const std::string &fn,
                            py::handle ix, py::handle iy, py::handle iz, py::handle iw,
                            py::handle snap, bool stack)
{
    // Read the storage layout: 4D slot grid + per-slot occupancy (channelID == 0 -> empty)
    arma::u32_vec channelID;
    const arma::u32_vec storage = quadriga_lib::hdf5_read_layout(fn, &channelID);

    const unsigned nx = storage.at(0), ny = storage.at(1),
                   nz = storage.at(2), nw = storage.at(3);

    if (nx == 0)
        throw std::runtime_error("HDF5 file does not exist or has no layout.");

    if (channelID.n_elem != (arma::uword)nx * (arma::uword)ny * (arma::uword)nz * (arma::uword)nw)
        throw std::runtime_error("Corrupted storage index.");

    // Resolve the per-dimension slot selectors (None -> full dimension, bounds-checked)
    const auto sx = parse_idx(ix, nx, "ix");
    const auto sy = parse_idx(iy, ny, "iy");
    const auto sz = parse_idx(iz, nz, "iz");
    const auto sw = parse_idx(iw, nw, "iw");

    const arma::uword n_sel = sx.n_elem * sy.n_elem * sz.n_elem * sw.n_elem;

    // Snapshot selection is only meaningful for a single addressed slot
    const arma::uvec snap_a = qd_python_numpy2arma_Col<arma::uword>(snap, false);
    if (!snap_a.is_empty() && n_sel != 1)
        throw std::invalid_argument("'snap' is only allowed when a single slot is selected.");

    // Visit the selected slots in column-major order (ix varies fastest); skip empty slots
    py::list chan;
    py::list par_list;
    for (const auto w : sw)
        for (const auto z : sz)
            for (const auto y : sy)
                for (const auto x : sx)
                {
                    const unsigned lin = x + nx * (y + ny * (z + nz * w));
                    if (channelID.at(lin) == 0) // empty slot -> skip
                        continue;

                    const auto channel = quadriga_lib::hdf5_read_channel<double>(fn, x, y, z, w);

                    // Defensive: a slot flagged occupied but holding nothing -> skip
                    if (channel.empty() && channel.par_names.empty())
                        continue;

                    // Validate structured data (par-only channels are not validated)
                    if (!channel.empty())
                    {
                        const std::string err = channel.is_valid();
                        if (!err.empty())
                            throw std::runtime_error(err);
                    }

                    // Structured channel data -> dict
                    chan.append(qd_python_channel2dict(channel, snap_a, false, stack));

                    // Unstructured datasets -> dict
                    py::dict par;
                    for (size_t k = 0; k < channel.par_names.size(); ++k)
                        par[channel.par_names[k].c_str()] = qd_python_any2numpy(channel.par_data[k]);
                    par_list.append(par);
                }

    // par: single dict for a single addressed slot, list of dicts otherwise
    py::object par_out;
    if (n_sel == 1)
        par_out = par_list.empty() ? py::object(py::dict()) : py::object(par_list[0]);
    else
        par_out = par_list;

    return py::make_tuple(chan, par_out);
}

// m.def("hdf5_read_channel", &hdf5_read_channel,
//       py::arg("fn"),
//       py::arg("ix") = py::none(),
//       py::arg("iy") = py::none(),
//       py::arg("iz") = py::none(),
//       py::arg("iw") = py::none(),
//       py::arg("snap") = py::none(),
//       py::arg("stack") = false);