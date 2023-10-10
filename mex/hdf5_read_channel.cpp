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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - fn              Filename of the QDANT file
    //  1 - location        Storage location, 1-based, vector with 1-4 elements, i.e. [ix], [ix, iy], [ix,iy,iz] or [ix,iy,iz,iw]
    //  2 - snap            Snapshot range, optional, vector, default = all

    // Outputs:
    //  0 - par             Unstructured data as struct, may be empty if no unstructured data is present
    //  1 - rx_position     Receiver positions, matrix of size [3, n_snap] or [3, 1]
    //  2 - tx_position     Transmitter positions, matrix of size [3, n_snap] or [3, 1]
    //  3 - coeff_re        Channel coefficients, real part, size [n_rx, n_tx, n_path, n_snap], zero-padded
    //  4 - coeff_im        Channel coefficients, imaginary part, size [n_rx, n_tx, n_path, n_snap], zero-padded
    //  5 - delay           Path delays in seconds, size [n_rx, n_tx, n_path, n_snap] or [1, 1, n_path, n_snap], zero-padded
    //  6 - center_frequency   Center frequency in [Hz], scalar or size [n_snap, 1]
    //  7 - name            Name of the channel object
    //  8 - initial_position   Index of reference position, values between 1 and n_snap (1-based)
    //  9 - path_gain       Path gain before antenna patterns, linear scale, size [n_path, n_snap], zero-padded
    // 10 - path_length     Absolute path length from TX to RX phase center in m, size [n_path, n_snap], zero-padded
    // 11 - path_polarization   Polarization transfer function, size [8, n_path, n_snap], interleaved complex, zero-padded
    // 12 - path_angles     Departure and arrival angles in rad {AOD, EOD, AOA, EOA}, size [n_path, 4, n_snap], zero-padded
    // 13 - path_coord      Interaction coordinates, size [3, n_coord, n_path, n_snap], NAN-padded
    // 14 - rx_orientation  Transmitter orientation, matrix of size [3, n_snap] or [3, 1] or []
    // 15 - tx_orientation  Receiver orientation, matrix of size [3, n_snap] or [3, 1] or []

    // Notes:
    // - Empty outputs are returned if data does not exist in the file
    // - All structured data is returned in single precision
    // - Unstructured datatypes are returned as stored in the HDF file
    // - Storage order of the unstructured data is maintained
    // - All unstructured vector types are returned as column vectors

    // Number of in and outputs
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_channel:no_input", "Filename is missing.");

    if (nlhs > 16)
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
        snap = qd_mex_typecast_Col<uword>(prhs[2], "snap");

    // Update snapshot index
    uword n_snap_channel = channel.n_snap();
    if (snap.empty() && n_snap_channel != 0ULL)
    {
        snap.set_size(n_snap_channel);
        uword *p = snap.memptr();
        for (uword s = 0ULL; s < n_snap_channel; ++s)
            p[s] = s;
    }
    else if (n_snap_channel != 0ULL) // Check bounds
    {
        for (uword &p : snap)
            if (p < 1 || p > n_snap_channel)
                mexErrMsgIdAndTxt("quadriga_lib:hdf5_read_channel:out_of_bound", "Snapshot index out of bound.");
            else // Convert 1-based to 0-based
                --p;
    }
    else // n_snap_channel == 0ULL
        snap.reset();

    uword n_snap = snap.n_elem;
    uword *i_snap = snap.memptr(); // Snapshot index

    // Get number of paths
    uword n_path = 0ULL;
    if (n_snap != 0ULL && n_snap_channel != 0ULL)
    {
        arma::uvec n_path_vec = channel.n_path();
        uword *p = n_path_vec.memptr();
        for (uword i = 0ULL; i < n_snap; ++i)
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
            uword dims[3];
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
                auto data = std::any_cast<arma::Mat<arma::uword>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 23)
            {
                auto data = std::any_cast<arma::Mat<arma::sword>>(channel.par_data.at(n));
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
                auto data = std::any_cast<arma::Cube<arma::uword>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 33)
            {
                auto data = std::any_cast<arma::Cube<arma::sword>>(channel.par_data.at(n));
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
                auto data = std::any_cast<arma::Col<arma::uword>>(channel.par_data.at(n));
                mxSetField(plhs[0], 0, field_names.at(n), qd_mex_copy2matlab(&data));
            }
            else if (type_id == 43)
            {
                auto data = std::any_cast<arma::Col<arma::sword>>(channel.par_data.at(n));
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
        plhs[13] = qd_mex_vector2matlab(&channel.path_coord, n_snap, i_snap, NAN);

    if (nlhs > 14 && channel.rx_orientation.n_cols <= 1ULL)
        plhs[14] = qd_mex_copy2matlab(&channel.rx_orientation);
    else if (nlhs > 14)
        plhs[14] = qd_mex_copy2matlab(&channel.rx_orientation, n_snap, i_snap);

    if (nlhs > 15 && channel.tx_orientation.n_cols <= 1ULL)
        plhs[15] = qd_mex_copy2matlab(&channel.tx_orientation);
    else if (nlhs > 15)
        plhs[15] = qd_mex_copy2matlab(&channel.tx_orientation, n_snap, i_snap);
}