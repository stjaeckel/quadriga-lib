// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"
#include "qd_helpers_wrappers.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_READ_CHANNEL
Read one or more channel objects from an HDF5 file

- Reads structured channel data and any unstructured datasets from a 4D indexed HDF5 file
- Each of ix, iy, iz, iw may be a scalar, vector, or omitted (omitted/empty = read full extent along that dimension)
- Slots are visited in column-major order and empty slots are skipped.
- Structured fields are stored in single precision in the file and returned in double.
- Unstructured datasets keep their stored type and shape.
- If no data is found, both outputs are empty `0x0` structs.
- Errors if the file does not exist or is not a quadriga-lib channel file; files modified with other HDF5 tools
  are not supported and may be rejected.

## Usage:
```
[ chan, par ] = quadriga_lib.hdf5_read_channel( fn, ix, iy, iz, iw, i_snap );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`ix`** — 1-based slot indices along dimension X; scalar or vector; default: `1:nx`
- **`iy`** — 1-based slot indices along dimension Y; scalar or vector; default: `1:ny`
- **`iz`** — 1-based slot indices along dimension Z; scalar or vector; default: `1:nz`
- **`iw`** — 1-based slot indices along dimension W; scalar or vector; default: `1:nw`
- **`i_snap`** — Snapshot indices to read; 1-based; default: all snapshots. Applies to every
  selected slot; errors if an index exceeds the number of snapshots of a selected slot.

## Outputs:
- **`chan`** — Struct array of length `N` (number of non-empty slots in the selection) with the channel data. Fields per element:<br><br>
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

## See also:
- [[hdf5_read_layout]] (for reading the layout in the file)
- [[hdf5_write_channel]] (for writing channel data)
- [[hdf5_read_dset]] (for reading individual unstructured datasets)
- [[hdf5_write_dset]] (for writing individual unstructured datasets)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read filename
    const std::string fn = qd_mex_get_string(prhs[0], "", "fn");

    // Slot indices (1-based -> 0-based); omitted or [] selects the full extent
    auto get_idx = [&](int arg_idx, const char *name) -> arma::u32_vec
    {
        if (arg_idx >= nrhs || mxGetNumberOfElements(prhs[arg_idx]) == 0)
            return arma::u32_vec();
        return qd_mex_get_index_Col<unsigned>(prhs[arg_idx], name);
    };
    const arma::u32_vec ix = get_idx(1, "ix"), iy = get_idx(2, "iy"), iz = get_idx(3, "iz"), iw = get_idx(4, "iw");
    const arma::uvec i_snap = (nrhs < 6) ? arma::uvec() : qd_mex_get_index_Col(prhs[5], "i_snap"); // 1-based -> 0-based

    // Occupied slots of the selection, then their channels
    std::vector<quadriga_lib::channel<double>> channels;
    arma::u32_vec dims, channel_id;
    CALL_QD(dims = quadriga_lib::hdf5_read_layout(fn, &channel_id));
    CALL_QD(channels = quadriga_lib::hdf5_read_channel_vec<double>(fn, qd_hdf5_select_slots(dims, channel_id, ix, iy, iz, iw), i_snap));

    // Keep par-only channels (c.empty() ignores par_data); check the structured data
    channels.erase(std::remove_if(channels.begin(), channels.end(), [](const quadriga_lib::channel<double> &c)
                                  { return c.empty() && c.par_names.empty(); }),
                   channels.end());
    for (const auto &c : channels)
        if (!c.empty())
        {
            const std::string error_msg = c.is_valid();
            if (!error_msg.empty())
                qd_mex_cpp_error(error_msg);
        }

    // Output 1: chan struct array (qd_mex_channel2struct handles empty vector)
    plhs[0] = qd_mex_channel2struct(channels, false);

    // Output 2: par struct array with the union of par_names across all channels
    if (nlhs > 1)
    {
        // Collect union of field names, preserving first-seen order
        std::vector<std::string> par_field_names;
        for (const auto &c : channels)
            for (const auto &n : c.par_names)
                if (std::find(par_field_names.begin(), par_field_names.end(), n) == par_field_names.end())
                    par_field_names.push_back(n);

        if (par_field_names.empty() || channels.empty())
        {
            std::vector<std::string> none;
            plhs[1] = qd_mex_make_struct(none, 0);
        }
        else
        {
            plhs[1] = qd_mex_make_struct(par_field_names, channels.size());
            for (size_t k = 0; k < channels.size(); ++k)
            {
                const auto &c = channels[k];
                for (size_t n = 0; n < c.par_names.size(); ++n)
                    qd_mex_set_field(plhs[1], c.par_names[n], qd_mex_any2matlab(c.par_data[n]), k);
            }
        }
    }
}