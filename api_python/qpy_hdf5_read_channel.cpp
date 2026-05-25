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

- Reads structured channel data and any unstructured datasets from one slot of an indexed HDF5 file
- A slot is addressed by the 0-based storage indices `(ix, iy, iz, iw)`
- Datasets span `n_rx` RX antennas, `n_tx` TX antennas, `n_path` paths and `n_snap` snapshots;
  snapshots typically index positions along a trajectory or frequencies
- Not every dataset spans all dimensions; only datasets present in the file are returned
- Per-snapshot data is returned as a `list` of arrays, one per snapshot, since `n_path` may differ between snapshots
- Stored in single precision, returned as double precision
- `snap` selects a subset of snapshots; if `None`, all snapshots are read

## Usage:
```
chan, par = quadriga_lib.channel.hdf5_read_channel( fn, ix, iy, iz, iw, snap )
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; str
- **`ix`** — Storage index for the x-dimension; 0-based; default: 0
- **`iy`** — Storage index for the y-dimension; 0-based; default: 0
- **`iz`** — Storage index for the z-dimension; 0-based; default: 0
- **`iw`** — Storage index for the w-dimension; 0-based; default: 0
- **`snap`** — Snapshot indices to read; 0-based; `(n_sel,)` or `None`; `None` reads all
  snapshots; default: `None`

## Outputs:
- **`chan`** — Dict with the channel data; only keys present in the file are included:<br><br>
  | Key                | Description                                                              | Type / Shape                         |
  | ------------------ | ------------------------------------------------------------------------ | ----------------------------------- |
  | `name`             | Channel name                                                             | String                              |
  | `tx_position`      | Transmitter positions (AP for downlink, STA for uplink)                  | `(3, n_snap)` or `(3, 1)`           |
  | `rx_position`      | Receiver positions (STA for downlink, AP for uplink)                     | `(3, n_snap)` or `(3, 1)`           |
  | `tx_orientation`   | Transmitter orientation, Euler angles (AP for downlink, STA for uplink)  | `(3, n_snap)` or `(3, 1)`           |
  | `rx_orientation`   | Receiver orientation, Euler angles (STA for downlink, AP for uplink)     | `(3, n_snap)` or `(3, 1)`           |
  | `coeff`            | Channel coefficients, complex-valued                                     | list of `(n_rx, n_tx, n_path_s)`    |
  | `delay`            | Propagation delays in seconds                                            | list of `(n_rx, n_tx, n_path_s)`    |
  | `path_gain`        | Path gain before antenna, linear scale                                   | list of `(n_path_s,)`               |
  | `path_length`      | Path length in m                                                         | list of `(n_path_s,)`               |
  | `path_polarization`| Polarization transfer function, complex                                  | list of `(4, n_path_s)`             |
  | `path_angles`      | Departure and arrival angles [AOD, EOD, AOA, EOA] in rad                 | list of `(n_path_s, 4)`             |
  | `fbs_pos`          | First-bounce scatterer positions                                         | list of `(3, n_path_s)`             |
  | `lbs_pos`          | Last-bounce scatterer positions                                          | list of `(3, n_path_s)`             |
  | `no_interact`      | Number of interaction points per path; uint32                            | list of `(n_path_s,)`               |
  | `interact_coord`   | Interaction coordinates                                                  | list of `(3, sum(no_interact_s))`   |
  | `center_frequency` | Center Frequency in Hz                                                   | `(n_snap,)` or scalar               |
  | `initial_position` | Index of reference position; 1-based                                     | int32, scalar                       |

- **`par`** —Dictionary of unstructured data

## See also:
- [[hdf5_read_layout]] (for reading the layout in the file)
- [[hdf5_write_channel]] (for writing channel data)
- [[hdf5_read_dset]] (for reading individual unstructured datasets)
- [[hdf5_write_dset]] (for writing individual unstructured datasets)
MD!*/

py::tuple hdf5_read_channel(const std::string &fn,
                            unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                            py::handle snap)
{
    // Read the channel object from the requested storage slot
    const auto channel = quadriga_lib::hdf5_read_channel<double>(fn, ix, iy, iz, iw);

    // Snapshot selection (None / empty reads all snapshots)
    const arma::uvec snap_a = qd_python_numpy2arma_Col<arma::uword>(snap, true);

    auto chan = qd_python_channel2dict(channel, snap_a);

    // Unstructured data
    py::dict par;
    if (!channel.par_names.empty())
        for (size_t n = 0; n < channel.par_names.size(); ++n)
            par[channel.par_names[n].c_str()] = qd_python_any2numpy(channel.par_data[n]);

    // Assemble the output
    return py::make_tuple(chan, par);
}

// pybind11 declaration:
// m.def("hdf5_read_channel", &hdf5_read_channel,
//       py::arg("fn"),
//       py::arg("ix") = 0,
//       py::arg("iy") = 0,
//       py::arg("iz") = 0,
//       py::arg("iw") = 0,
//       py::arg("snap") = py::none());