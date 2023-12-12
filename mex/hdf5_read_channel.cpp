// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.cpp"

/*!SECTION
Channel functions
SECTION!*/

/*!MD
# HDF5_READ_CHANNEL
Read channel data from an HDF5 file

## Description:
Quadriga-Lib provides an HDF5-based solution for storing and organizing channel data. This data 
comprises various well-defined sets, including channel coefficients, positions of transmitters and 
receivers, as well as path data that reflects the interaction of radio waves with the environment. 
Typically, these datasets are multi-dimensional, encompassing data for `n_rx` receive antennas, 
`n_tx` transmit antennas, `n_path` propagation paths, and `n_snap` snapshots. Snapshots are 
particularly useful for recording data across different locations (such as along a trajectory) or 
various frequencies. It is important to note that not all datasets include all these dimensions.<br><br>

The library also supports the addition of extra datasets of any type or shape, which can be useful 
for incorporating descriptive data or analysis results. To facilitate data access, the function 
`quadriga_lib.hdf5_read_channel` is designed to read both structured and unstructured data from the 
file.

## Usage:

```
[ par, rx_position, tx_position, coeff_re, coeff_im, delay, center_freq, name, initial_pos, ...
   path_gain, path_length, path_polarization, path_angles, path_fbs_pos, path_lbs_pos, no_interact, ...
   interact_coord, rx_orientation, tx_orientation ] = quadriga_lib.hdf5_read_channel( fn, location, snap );
```

## Input Arguments:
- **`fn`**<br>
  Filename of the HDF5 file, string

- **`location`** (optional)<br>
  Storage location inside the file; 1-based; vector with 1-4 elements, i.e. `[ix]`, `[ix, iy]`, 
  `[ix,iy,iz]` or `[ix,iy,iz,iw]`; Default: `ix = iy = iz = iw = 1`

- **`snap`** (optional)<br>
  Snapshot range; optional; vector, default = read all

## Output Arguments:
- **`par`**<br>
  Unstructured data as struct, may be empty if no unstructured data is present

- **Structured data:** (outputs 2-19, single precision)
  `rx_position`    | Receiver positions                                       | `[3, n_snap]` or `[3, 1]`
  `tx_position`    | Transmitter positions                                    | `[3, n_snap]` or `[3, 1]`
  `coeff_re`       | Channel coefficients, real part                          | `[n_rx, n_tx, n_path, n_snap]`
  `coeff_im`       | Channel coefficients, imaginary part                     | `[n_rx, n_tx, n_path, n_snap]`
  `delay`          | Propagation delays in seconds                            | `[n_rx, n_tx, n_path, n_snap]` or `[1, 1, n_path, n_snap]`
  `center_freq`    | Center frequency in [Hz]                                 | `[n_snap, 1]` or scalar
  `name`           | Name of the channel                                      | String
  `initial_pos`    | Index of reference position, 1-based                     | uint32, scalar
  `path_gain`      | Path gain before antenna, linear scale                   | `[n_path, n_snap]`
  `path_length`    | Path length from TX to RX phase center in m              | `[n_path, n_snap]`
  `polarization`   | Polarization transfer function, interleaved complex      | `[8, n_path, n_snap]`
  `path_angles`    | Departure and arrival angles {AOD, EOD, AOA, EOA} in rad | `[n_path, 4, n_snap]`
  `path_fbs_pos`   | First-bounce scatterer positions                         | `[3, n_path, n_snap]`
  `path_lbs_pos`   | Last-bounce scatterer positions                          | `[3, n_path, n_snap]`
  `no_interact`    | Number interaction points of paths with the environment  | uint32, `[n_path, n_snap]`
  `interact_coord` | Interaction coordinates                                  | `[3, max(sum(no_interact)), n_snap]`
  `rx_orientation` | Transmitter orientation                                  | `[3, n_snap]` or `[3, 1]`
  `tx_orientation` | Receiver orientation                                     | `[3, n_snap]` or `[3, 1]`
  
## Caveat:
- Empty outputs are returned if data set does not exist in the file
- All structured data is stored in single precision. Hence, outputs are also in single precision.
- Unstructured datatypes are returned as stored in the HDF file (same type, dimensions and storage order)
- Typically, `n_path` may vary for each snapshot. In such cases, `n_path` is set to the maximum value found 
  within the range of snapshots, and any missing paths are padded with zeroes.
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - fn              Filename of the HDF5 file
    //  1 - location        Storage location, 1-based, vector with 1-4 elements, i.e. [ix], [ix, iy], [ix,iy,iz] or [ix,iy,iz,iw]
    //  2 - snap            Snapshot range, optional, vector, default = all

    // Outputs:
    //  0 - par             Unstructured data as struct, may be empty if no unstructured data is present
    //  1 - rx_position     Receiver positions, matrix of size [3, n_snap] or [3, 1]
    //  2 - tx_position     Transmitter positions, matrix of size [3, n_snap] or [3, 1]
    //  3 - coeff_re        Channel coefficients, real part, size [n_rx, n_tx, n_path, n_snap], zero-padded
    //  4 - coeff_im        Channel coefficients, imaginary part, size [n_rx, n_tx, n_path, n_snap], zero-padded
    //  5 - delay           Path delays in seconds, size [n_rx, n_tx, n_path, n_snap] or [1, 1, n_path, n_snap], zero-padded
    //  6 - center_freq     Center frequency in [Hz], scalar or size [n_snap, 1]
    //  7 - name            Name of the channel object
    //  8 - initial_pos     Index of reference position, values between 1 and n_snap (1-based)
    //  9 - path_gain       Path gain before antenna patterns, linear scale, size [n_path, n_snap], zero-padded
    // 10 - path_length     Absolute path length from TX to RX phase center in m, size [n_path, n_snap], zero-padded
    // 11 - path_polarization   Polarization transfer function, size [8, n_path, n_snap], interleaved complex, zero-padded
    // 12 - path_angles     Departure and arrival angles in rad {AOD, EOD, AOA, EOA}, size [n_path, 4, n_snap], zero-padded
    // 13 - path_fbs_pos    First-bounce scatterer positions, size [3, n_path, n_snap], zero-padded
    // 14 - path_lbs_pos    Last-bounce scatterer positions, size [3, n_path, n_snap], zero-padded
    // 15 - no_interact     Number interaction points of a path with the environment, 0 = LOS, [n_path, n_snap], zero-padded
    // 16 - interact_coord  Interaction coordinates, size [3, max(sum(no_interact)), n_snap], zero-padded
    // 17 - rx_orientation  Transmitter orientation, matrix of size [3, n_snap] or [3, 1] or []
    // 18 - tx_orientation  Receiver orientation, matrix of size [3, n_snap] or [3, 1] or []

    // Notes:
    // - Empty outputs are returned if data does not exist in the file
    // - All structured data is returned in single precision
    // - Unstructured datatypes are returned as stored in the HDF file
    // - Storage order of the unstructured data is maintained
    // - All unstructured vector types are returned as column vectors

    // Number of in and outputs
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_channel:no_input", "Filename is missing.");

    if (nlhs > 19)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_channel:no_input", "Too many output arguments.");

    // Read filename
    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_channel:wrong_type", "Input 'fn' must be a string");

    auto mx_fn = mxArrayToString(prhs[0]);
    std::string fn = std::string(mx_fn);
    mxFree(mx_fn);

    // Read location (1-based)
    arma::Col<unsigned> location;
    if (nrhs > 1)
        location = qd_mex_typecast_Col<unsigned>(prhs[1], "location");

    unsigned ix = location.empty() ? 1 : location.at(0);
    unsigned iy = location.n_elem > 1ULL ? location.at(1) : 1;
    unsigned iz = location.n_elem > 2ULL ? location.at(2) : 1;
    unsigned iw = location.n_elem > 3ULL ? location.at(3) : 1;

    if (ix == 0 || iy == 0 || iz == 0 || iw == 0)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_channel:wrong_type", "Input 'location' cannot contain zeros (1-based indexing)");
    --ix, --iy, --iz, --iw;

    // Read channel object from file
    quadriga_lib::channel<float> channel;
    try
    {
        channel = quadriga_lib::hdf5_read_channel<float>(fn, ix, iy, iz, iw);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_channel:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_channel:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Make sure the data is valid
    if (!channel.empty())
    {
        std::string error_msg = channel.is_valid();
        if (!error_msg.empty())
            mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_channel:unknown_error", error_msg.c_str());
    }

    // Read snapshot range
    arma::uvec snap; // Empty = read all
    if (nrhs > 2)
        snap = qd_mex_typecast_Col<unsigned long long>(prhs[2], "snap");

    // Update snapshot index
    unsigned long long n_snap_channel = channel.n_snap();
    if (snap.empty() && n_snap_channel != 0ULL)
    {
        snap.set_size(n_snap_channel);
        unsigned long long *p = snap.memptr();
        for (auto s = 0ULL; s < n_snap_channel; ++s)
            p[s] = s;
    }
    else if (n_snap_channel != 0ULL) // Check bounds
    {
        for (auto &p : snap)
            if (p < 1 || p > n_snap_channel)
                mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_channel:out_of_bound", "Snapshot index out of bound.");
            else // Convert 1-based to 0-based
                --p;
    }
    else // n_snap_channel == 0ULL
        snap.reset();

    unsigned long long n_snap = snap.n_elem;
    unsigned long long *i_snap = snap.memptr(); // Snapshot index

    // Get number of paths
    unsigned long long n_path = 0ULL;
    if (n_snap != 0ULL && n_snap_channel != 0ULL)
    {
        arma::uvec n_path_vec = channel.n_path();
        unsigned long long *p = n_path_vec.memptr();
        for (auto i = 0ULL; i < n_snap; ++i)
            n_path = p[i_snap[i]] > n_path ? p[i_snap[i]] : n_path;
    }

    // Write to MATLAB
    if (nlhs > 0 && channel.par_names.empty())
    {
        mwSize dims[2] = {0, 0};
        plhs[0] = mxCreateStructArray(2, dims, 0, NULL);
    }
    else if (nlhs > 0)
    {
        std::vector<const char *> field_names;
        for (const auto &str : channel.par_names)
            field_names.push_back(str.c_str());

        mwSize dims[2] = {1, 1}; // Creates a 1x1 struct array
        plhs[0] = mxCreateStructArray(2, dims, (int)field_names.size(), field_names.data());

        for (size_t n = 0; n < field_names.size(); ++n)
        {
            unsigned long long dims[3];
            void *dataptr;
            int type_id = quadriga_lib::any_type_id(&channel.par_data.at(n), dims, &dataptr);

            if (type_id == 9) // Strings
            {
                auto data = std::any_cast<std::string>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), mxCreateString(data.c_str()));
            }

            // Scalars
            else if (type_id == 10)
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab((float *)dataptr));
            else if (type_id == 11)
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab((double *)dataptr));
            else if (type_id == 12)
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab((unsigned long long int *)dataptr));
            else if (type_id == 13)
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab((long long int *)dataptr));
            else if (type_id == 14)
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab((unsigned int *)dataptr));
            else if (type_id == 15)
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab((int *)dataptr));

            // Matrices
            else if (type_id == 20)
            {
                auto data = std::any_cast<arma::Mat<float>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 21)
            {
                auto data = std::any_cast<arma::Mat<double>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 22)
            {
                auto data = std::any_cast<arma::Mat<unsigned long long>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 23)
            {
                auto data = std::any_cast<arma::Mat<long long>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 24)
            {
                auto data = std::any_cast<arma::Mat<unsigned>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 25)
            {
                auto data = std::any_cast<arma::Mat<int>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }

            // Cubes
            else if (type_id == 30)
            {
                auto data = std::any_cast<arma::Cube<float>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 31)
            {
                auto data = std::any_cast<arma::Cube<double>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 32)
            {
                auto data = std::any_cast<arma::Cube<unsigned long long>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 33)
            {
                auto data = std::any_cast<arma::Cube<long long>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 34)
            {
                auto data = std::any_cast<arma::Cube<unsigned>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 35)
            {
                auto data = std::any_cast<arma::Cube<int>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }

            // Vectors (Columns only)
            else if (type_id == 40)
            {
                auto data = std::any_cast<arma::Col<float>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 41)
            {
                auto data = std::any_cast<arma::Col<double>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 42)
            {
                auto data = std::any_cast<arma::Col<unsigned long long>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 43)
            {
                auto data = std::any_cast<arma::Col<long long>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 44)
            {
                auto data = std::any_cast<arma::Col<unsigned>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 45)
            {
                auto data = std::any_cast<arma::Col<int>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
        }
    }

    if (nlhs > 1 && channel.rx_pos.n_cols <= 1ULL)
        plhs[1] = qd_mex_copy2matlab(&channel.rx_pos);
    else if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&channel.rx_pos, n_snap, i_snap);

    if (nlhs > 2 && channel.tx_pos.n_cols <= 1ULL)
        plhs[2] = qd_mex_copy2matlab(&channel.tx_pos);
    else if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&channel.tx_pos, n_snap, i_snap);

    if (nlhs > 3)
        plhs[3] = qd_mex_vector2matlab(&channel.coeff_re, n_snap, i_snap);

    if (nlhs > 4)
        plhs[4] = qd_mex_vector2matlab(&channel.coeff_im, n_snap, i_snap);

    if (nlhs > 5)
        plhs[5] = qd_mex_vector2matlab(&channel.delay, n_snap, i_snap);

    if (nlhs > 6 && channel.center_frequency.n_elem <= 1ULL)
        plhs[6] = qd_mex_copy2matlab(&channel.center_frequency);
    else if (nlhs > 6)
        plhs[6] = qd_mex_copy2matlab(&channel.center_frequency, false, n_snap, i_snap);

    if (nlhs > 7) // There is always a default name
        plhs[7] = mxCreateString(channel.name.c_str());

    if (nlhs > 8)
        plhs[8] = qd_mex_copy2matlab(&channel.initial_position);

    if (nlhs > 9)
        plhs[9] = qd_mex_vector2matlab(&channel.path_gain, n_snap, i_snap);

    if (nlhs > 10)
        plhs[10] = qd_mex_vector2matlab(&channel.path_length, n_snap, i_snap);

    if (nlhs > 11)
        plhs[11] = qd_mex_vector2matlab(&channel.path_polarization, n_snap, i_snap);

    if (nlhs > 12)
        plhs[12] = qd_mex_vector2matlab(&channel.path_angles, n_snap, i_snap);

    if (nlhs > 13)
        plhs[13] = qd_mex_vector2matlab(&channel.path_fbs_pos, n_snap, i_snap);

    if (nlhs > 14)
        plhs[14] = qd_mex_vector2matlab(&channel.path_lbs_pos, n_snap, i_snap);

    if (nlhs > 15)
        plhs[15] = qd_mex_vector2matlab(&channel.no_interact, n_snap, i_snap);

    if (nlhs > 16)
        plhs[16] = qd_mex_vector2matlab(&channel.interact_coord, n_snap, i_snap);

    if (nlhs > 17 && channel.rx_orientation.n_cols <= 1ULL)
        plhs[17] = qd_mex_copy2matlab(&channel.rx_orientation);
    else if (nlhs > 17)
        plhs[17] = qd_mex_copy2matlab(&channel.rx_orientation, n_snap, i_snap);

    if (nlhs > 18 && channel.tx_orientation.n_cols <= 1ULL)
        plhs[18] = qd_mex_copy2matlab(&channel.tx_orientation);
    else if (nlhs > 18)
        plhs[18] = qd_mex_copy2matlab(&channel.tx_orientation, n_snap, i_snap);
}