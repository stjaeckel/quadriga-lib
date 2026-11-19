// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"
#include "quadriga_lib_helper_functions.hpp"

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
- `i_snap` selects a subset of snapshots; if `None`, all snapshots are read
- Errors if the file does not exist or is not a quadriga-lib channel file; files modified with other HDF5 tools are
  not supported and may be rejected.

## Usage:
```
chan, par = quadriga_lib.channel.hdf5_read_channel( fn, ix, iy, iz, iw, *, i_snap, stack=False )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file
- **`ix`** — 0-based slot indices along dimension X; scalar or vector; default: `0 ... nx-1`
- **`iy`** — 0-based slot indices along dimension Y; scalar or vector; default: `0 ... ny-1`
- **`iz`** — 0-based slot indices along dimension Z; scalar or vector; default: `0 ... nz-1`
- **`iw`** — 0-based slot indices along dimension W; scalar or vector; default: `0 ... nw-1`
- **`i_snap`** — Snapshot indices to read; 0-based; default: all snapshots. Applies to every selected slot; raises an error if an index exceeds the number of snapshots of a selected slot.
- **`stack`** — If `True`, stack snapshots. Default: `False` (return as list)

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
  | `center_freq`      | Center Frequency in Hz                                                   | `(n_snap,)` or scalar             | `(n_snap,)` or scalar                |
  | `initial_position` | Reference snapshot (start of the current segment); 0-based               | int32, scalar                     | int32, scalar                        |

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
    const arma::u32_mat slots = qd_hdf5_select_slots(dims, channel_id,
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
