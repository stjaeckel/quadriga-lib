// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_READ_CHANNEL
Read a channel object from an HDF5 file

- Reads structured channel data and any unstructured datasets from a single 4D slot
- Structured fields are stored in single precision in the file and returned in double
- Unstructured datasets keep their stored type and shape
- The optional `snap` argument extracts a subset of snapshots from structured fields;
  unstructured datasets are returned in full and are not subsetted
- Empty/missing slot returns an empty struct (`0x0`) for both outputs

## Usage:
```
[ par, chan ] = quadriga_lib.hdf5_read_channel( fn, location, snap );
```

## Inputs:
- **`fn`** — Filename of the HDF5 file; string
- **`location`** *(optional)* — Slot location inside the file; 1-based; vector with 1-4 elements,
  i.e. `[ix]`, `[ix, iy]`, `[ix, iy, iz]` or `[ix, iy, iz, iw]`; default: `[1, 1, 1, 1]`
- **`snap`** *(optional)* — Snapshot indices to read; 1-based; default: all snapshots

## Outputs:
- **`par`** — Unstructured datasets as a MATLAB struct; field names follow the dataset names in the file
  (without prefix); empty `0x0` struct if no unstructured data is
  present
- **`chan`** — Struct containing the channel data with the following fields:<br><br>
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
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string fn = qd_mex_get_string(prhs[0]);
    const arma::u32_vec location = (nrhs < 2) ? arma::u32_vec() : qd_mex_get_Col<unsigned>(prhs[1]);
    arma::Col<arma::uword> snap = (nrhs < 3) ? arma::Col<arma::uword>() : qd_mex_get_Col<arma::uword>(prhs[2]);

    unsigned ix = location.is_empty() ? 1 : location.at(0);
    unsigned iy = location.n_elem > 1 ? location.at(1) : 1;
    unsigned iz = location.n_elem > 2 ? location.at(2) : 1;
    unsigned iw = location.n_elem > 3 ? location.at(3) : 1;

    if (ix == 0 || iy == 0 || iz == 0 || iw == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'location' cannot contain zeros (1-based indexing).");
    --ix, --iy, --iz, --iw;

    // Read channel from file (request double precision)
    quadriga_lib::channel<double> channel;
    CALL_QD(channel = quadriga_lib::hdf5_read_channel<double>(fn, ix, iy, iz, iw));

    if (!channel.empty())
    {
        std::string error_msg = channel.is_valid();
        if (!error_msg.empty())
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", error_msg.c_str());
    }

    // Validate snap indices (1-based) and convert to 0-based
    const arma::uword n_snap_full = channel.n_snap();
    if (!snap.is_empty())
        for (auto &s : snap)
        {
            if (s == 0 || s > n_snap_full)
                mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Snapshot index out of bounds.");
            --s;
        }

    // Apply snap subsetting on structured fields (par_data is not per-snapshot)
    if (!snap.is_empty())
    {
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

        // Move par_data over (not snapshot-dependent)
        sub.par_names = std::move(channel.par_names);
        sub.par_data = std::move(channel.par_data);
        channel = std::move(sub);
    }

    // Output 1: par struct (unstructured datasets)
    if (nlhs > 0)
    {
        if (channel.par_names.empty())
        {
            std::vector<std::string> none;
            plhs[0] = qd_mex_make_struct(none, 0);
        }
        else
        {
            plhs[0] = qd_mex_make_struct(channel.par_names);

            for (size_t n = 0; n < channel.par_names.size(); ++n)
            {
                const std::string &fname = channel.par_names[n];
                const std::any &dset = channel.par_data[n];
                qd_mex_set_field(plhs[0], fname, qd_mex_any2matlab(dset));
            }
        }
    }

    // Output 2: chan struct (structured data)
    if (nlhs > 1)
    {
        std::vector<quadriga_lib::channel<double>> vec;
        if (!channel.empty())
            vec.push_back(std::move(channel));
        plhs[1] = qd_mex_channel2struct(vec);
    }
}