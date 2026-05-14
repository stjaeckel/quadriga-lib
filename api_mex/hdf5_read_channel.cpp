// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_READ_CHANNEL
Read one or more channel objects from an HDF5 file

- Reads structured channel data and any unstructured datasets from a 4D indexed HDF5 file
- Each of ix, iy, iz, iw may be a scalar, vector, or omitted (omitted/empty = read full extent along that dimension)
- Slots are visited in column-major order empty slots are skipped
- Structured fields are stored in single precision in the file and returned in double
- Unstructured datasets keep their stored type and shape
- If no data is found, both outputs are empty `0x0` structs

## Usage:
```
[ chan, par ] = quadriga_lib.hdf5_read_channel( fn, ix, iy, iz, iw, snap );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`ix`** *(optional)* — 1-based slot indices along dimension X; scalar or vector; default: `1:nx`
- **`iy`** *(optional)* — 1-based slot indices along dimension Y; scalar or vector; default: `1:ny`
- **`iz`** *(optional)* — 1-based slot indices along dimension Z; scalar or vector; default: `1:nz`
- **`iw`** *(optional)* — 1-based slot indices along dimension W; scalar or vector; default: `1:nw`
- **`snap`** *(optional)* — Snapshot indices to read; 1-based; default: all snapshots. Only allowed
  when the total selection is a single slot.

## Outputs:
- **`chan`** — Struct array of length `N` (number of non-empty slots in the selection) with the channel data. Fields per element:<br><br>
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
    const std::string fn = qd_mex_get_string(prhs[0]);

    // Read storage layout (returns [0,0,0,0] if file does not exist)
    arma::Col<unsigned> storage_space;
    arma::Col<unsigned> channel_id;
    CALL_QD(storage_space = quadriga_lib::hdf5_read_layout(fn, &channel_id));

    const unsigned nx = storage_space.at(0);
    const unsigned ny = storage_space.at(1);
    const unsigned nz = storage_space.at(2);
    const unsigned nw = storage_space.at(3);

    if (nx == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "HDF5 file does not exist or has no layout.");

    // Parse index inputs: omitted/empty -> 1:n_dim; vector -> validated as-is
    auto parse_idx = [&](int arg_idx, unsigned n_dim, const char *name) -> arma::Col<unsigned>
    {
        if (arg_idx >= nrhs || mxGetNumberOfElements(prhs[arg_idx]) == 0)
        {
            arma::Col<unsigned> v(n_dim);
            for (unsigned i = 0; i < n_dim; ++i)
                v.at(i) = i + 1;
            return v;
        }
        arma::Col<unsigned> v = qd_mex_get_Col<unsigned>(prhs[arg_idx]);
        for (auto val : v)
            if (val == 0 || val > n_dim)
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Index '%s' out of bounds.", name);
        return v;
    };

    arma::Col<unsigned> ix_vec = parse_idx(1, nx, "ix");
    arma::Col<unsigned> iy_vec = parse_idx(2, ny, "iy");
    arma::Col<unsigned> iz_vec = parse_idx(3, nz, "iz");
    arma::Col<unsigned> iw_vec = parse_idx(4, nw, "iw");

    // Parse snap (optional)
    arma::uvec snap = (nrhs < 6) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[5]);

    const arma::uword n_sel = ix_vec.n_elem * iy_vec.n_elem * iz_vec.n_elem * iw_vec.n_elem;
    if (!snap.is_empty() && n_sel > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'snap' is only allowed when a single slot is requested.");

    // Iterate Cartesian product in column-major order, skipping empty slots.
    std::vector<quadriga_lib::channel<double>> channels;
    channels.reserve((size_t)n_sel);

    for (arma::uword w = 0; w < iw_vec.n_elem; ++w)
        for (arma::uword z = 0; z < iz_vec.n_elem; ++z)
            for (arma::uword y = 0; y < iy_vec.n_elem; ++y)
                for (arma::uword x = 0; x < ix_vec.n_elem; ++x)
                {
                    const unsigned ix0 = ix_vec.at(x) - 1;
                    const unsigned iy0 = iy_vec.at(y) - 1;
                    const unsigned iz0 = iz_vec.at(z) - 1;
                    const unsigned iw0 = iw_vec.at(w) - 1;

                    // Fast empty-slot check via channel_id (avoids per-slot HDF5 read)
                    const unsigned lin_idx = ix0 + iy0 * nx + iz0 * nx * ny + iw0 * nx * ny * nz;
                    if (channel_id.at((size_t)lin_idx) == 0)
                        continue;

                    quadriga_lib::channel<double> c;
                    CALL_QD(c = quadriga_lib::hdf5_read_channel<double>(fn, ix0, iy0, iz0, iw0));

                    // Keep par-only channels (c.empty() ignores par_data)
                    if (c.empty() && c.par_names.empty())
                        continue;

                    if (!c.empty())
                    {
                        std::string error_msg = c.is_valid();
                        if (!error_msg.empty())
                            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", error_msg.c_str());
                    }

                    channels.push_back(std::move(c));
                }

    // Snap subsetting (only valid for a single loaded channel)
    if (!snap.is_empty() && !channels.empty())
    {
        quadriga_lib::channel<double> &channel = channels[0];
        const arma::uword n_snap_full = channel.n_snap();
        for (auto &s : snap)
        {
            if (s == 0 || s > n_snap_full)
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Snapshot index out of bounds.");
            --s;
        }

        quadriga_lib::channel<double> sub;
        sub.name = channel.name;
        sub.initial_position = channel.initial_position;

        auto sub_mat = [&](const arma::Mat<double> &src, arma::Mat<double> &dst)
        {
            dst = (src.n_cols <= 1) ? src : arma::Mat<double>(src.cols(snap));
        };
        sub_mat(channel.rx_pos, sub.rx_pos);
        sub_mat(channel.tx_pos, sub.tx_pos);
        sub_mat(channel.rx_orientation, sub.rx_orientation);
        sub_mat(channel.tx_orientation, sub.tx_orientation);

        sub.center_frequency = (channel.center_frequency.n_elem <= 1)
                                   ? channel.center_frequency
                                   : arma::Col<double>(channel.center_frequency.elem(snap));

        auto sub_vec = [&](const auto &src, auto &dst)
        {
            if (src.empty())
                return;
            dst.reserve(snap.n_elem);
            for (arma::uword i : snap)
                dst.push_back(src[i]);
        };
        sub_vec(channel.coeff_re, sub.coeff_re);
        sub_vec(channel.coeff_im, sub.coeff_im);
        sub_vec(channel.delay, sub.delay);
        sub_vec(channel.path_gain, sub.path_gain);
        sub_vec(channel.path_length, sub.path_length);
        sub_vec(channel.path_polarization, sub.path_polarization);
        sub_vec(channel.path_angles, sub.path_angles);
        sub_vec(channel.path_fbs_pos, sub.path_fbs_pos);
        sub_vec(channel.path_lbs_pos, sub.path_lbs_pos);
        sub_vec(channel.no_interact, sub.no_interact);
        sub_vec(channel.interact_coord, sub.interact_coord);

        sub.par_names = std::move(channel.par_names);
        sub.par_data = std::move(channel.par_data);
        channel = std::move(sub);
    }

    // Output 1: chan struct array (qd_mex_channel2struct handles empty vector)
    if (nlhs > 0)
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