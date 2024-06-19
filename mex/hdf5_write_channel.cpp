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
# HDF5_WRITE_CHANNEL
Writes channel data to HDF5 files

## Description:
Quadriga-Lib provides an HDF5-based solution for storing and organizing channel data. This function
can be used to write structured and unstructured data to an HDF5 file. 

## Usage:

```
storage_dims = quadriga_lib.hdf5_write_channel( fn, location, rx_position, tx_position, ...
   coeff_re, coeff_im, delay, center_freq, name, initial_pos, path_gain, path_length, ...
   path_polarization, path_angles, path_fbs_pos, path_lbs_pos, no_interact, interact_coord, ...
   rx_orientation, tx_orientation )
```

## Input Arguments:
- **`fn`**<br>
  Filename of the HDF5 file, string

- **`location`** (optional)<br>
  Storage location inside the file; 1-based; vector with 1-4 elements, i.e. `[ix]`, `[ix, iy]`, 
  `[ix,iy,iz]` or `[ix,iy,iz,iw]`; Default: `ix = iy = iz = iw = 1`

- **`par`**<br>
  Unstructured data as struct, can be empty if no unstructured data should be written

- **Structured data:** (inputs 4-21, single or double precision)
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

## Output Arguments:
- **`storage_dims`**<br>
  Size of the dimensions of the storage space, vector with 4 elements, i.e. `[nx,ny,nz,nw]`.

## Caveat:
- If the file exists already, the new data is added to the exisiting file
- If a new file is created, a storage layout is created to store the location of datasets in the file
- For `location = [ix]` storage layout is `[65536,1,1,1]` or `[ix,1,1,1]` if (`ix > 65536`)
- For `location = [ix,iy]` storage layout is `[1024,64,1,1]`
- For `location = [ix,iy,iz]` storage layout is `[256,16,16,1]`
- For `location = [ix,iy,iz,iw]` storage layout is `[128,8,8,8]`
- You can create a custom storage layout by creating the file first using "`hdf5_create_file`"
- You can reshape the storage layout by using "`hdf5_reshape_storage`", but the total number of elements must not change
- Inputs can be empty or missing.
- All structured data is written in single precision (but can can be provided as single or double)
- Unstructured datatypes are maintained in the HDF file
- Supported unstructured types: string, double, float, (u)int32, (u)int64
- Supported unstructured size: up to 3 dimensions
- Storage order of the unstructured data is maintained
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - fn              Filename of the HDF5 file
    //  1 - location        Storage location, 1-based, vector with 1-4 elements, i.e. [ix], [ix, iy], [ix,iy,iz] or [ix,iy,iz,iw]
    //  2 - par             Unstructured data as struct, can be empty if not needed
    //  3 - rx_position     Receiver positions, matrix of size [3, n_snap]
    //  4 - tx_position     Transmitter positions, matrix of size [3, n_snap] or [3, 1]
    //  5 - coeff_re        Channel coefficients, real part, size [n_rx, n_tx, n_path, n_snap], zero-padded
    //  6 - coeff_im        Channel coefficients, imaginary part, size [n_rx, n_tx, n_path, n_snap], zero-padded
    //  7 - delay           Path delays in seconds, size [n_rx, n_tx, n_path, n_snap] or [n_path, n_snap], zero-padded
    //  8 - center_frequency   Center frequency in [Hz], scalar or size [n_snap, 1]
    //  9 - name            Name of the channel object
    // 10 - initial_position   Index of reference position, values between 1 and n_snap (1-based)
    // 11 - path_gain       Path gain before antenna patterns, linear scale, size [n_path, n_snap], zero-padded
    // 12 - path_length     Absolute path length from TX to RX phase center in m, size [n_path, n_snap], zero-padded
    // 13 - path_polarization   Polarization transfer function, size [8, n_path, n_snap], interleaved complex, zero-padded
    // 14 - path_angles     Departure and arrival angles in rad {AOD, EOD, AOA, EOA}, size [n_path, 4, n_snap], zero-padded
    // 15 - path_fbs_pos    First-bounce scatterer positions, size [3, n_path, n_snap], zero-padded
    // 16 - path_lbs_pos    Last-bounce scatterer positions, size [3, n_path, n_snap], zero-padded
    // 17 - no_interact     Number interaction points of a path with the environment, 0 = LOS, [n_path, n_snap], zero-padded
    // 18 - interact_coord  Interaction coordinates, size [3, max(sum(no_interact)), n_snap], zero-padded
    // 19 - rx_orientation  Transmitter orientation, matrix of size [3, n_snap] or [3, 1] or []
    // 20 - tx_orientation  Receiver orientation, matrix of size [3, n_snap] or [3, 1] or []

    // Output:
    //  0 - storage_dims    Dimensions of the storage space in the file, 4-element vector

    // Notes:
    // - If a new file is created, a storage layout is created to store the location of datasets in the file
    //      - For location = [ix] storage layout is [65536,1,1,1] or [ix,1,1,1] if (ix > 65536)
    //      - For location = [ix,iy] storage layout is [1024,64,1,1]
    //      - For location = [ix,iy,iz] storage layout is [256,16,16,1]
    //      - For location = [ix,iy,iz,iw] storage layout is [128,8,8,8]
    // - You can create a custom storage layout by creating the file first using "hdf5_create_file"
    // - You can reshape the storage layout by using "hdf5_reshape_storage", but the total number of elements must not change
    // - Inputs can be empty or missing.
    // - All structured data is written in single precision (but can can be provided as single or double)
    // - Unstructured datatypes are maintained in the HDF file
    // - Supported unstructured types: string, double, float, (u)int32, (u)int64
    // - Supported unstructured size: up to 3 dimensions
    // - Storage order of the unstructured data is maintained

    // Number of in and outputs
    if (nrhs < 3)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_channel:no_input", "Need at least 3 inputs (file name, location and 1 dataset).");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_channel:no_output", "Incorrect number of output arguments.");

    // Read filename
    if (!mxIsClass(prhs[0], "char"))
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_channel:wrong_type", "Input 'fn' must be a string");

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
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_channel:wrong_type", "Input 'location' cannot contain zeros (1-based indexing)");
    --ix, --iy, --iz, --iw;

    // Read the storage space from the file - returns [0,0,0,0] if file does not exist
    arma::Col<unsigned> storage_space;
    try
    {
        storage_space = quadriga_lib::hdf5_read_layout(fn);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_channel:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_channel:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Create file
    try
    {
        if (storage_space.at(0) == 0)
        {
            unsigned nx = 1, ny = 1, nz = 1, nw = 1;
            if (location.n_elem == 1ULL)
                nx = ix > 65536U ? ix : 65536U;
            else if (location.n_elem == 2ULL)
                nx = ix > 1024U ? ix : 1024U,
                ny = iy > 64U ? iy : 64U;
            else if (location.n_elem == 3ULL)
                nx = ix > 256U ? ix : 256U,
                ny = iy > 16U ? iy : 16U,
                nz = iz > 16U ? iz : 16U;
            else if (location.n_elem == 4ULL)
                nx = ix > 128U ? ix : 128U,
                ny = iy > 8U ? iy : 8U,
                nz = iz > 8U ? iz : 8U,
                nw = iz > 8U ? iw : 8U;

            quadriga_lib::hdf5_create(fn, nx, ny, nz, nw);
            storage_space.at(0) = nx;
            storage_space.at(1) = ny;
            storage_space.at(2) = nz;
            storage_space.at(3) = nw;
        }
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_channel:create_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_channel:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Throw error if location exceeds storage space
    if (ix > storage_space.at(0) || iy > storage_space.at(1) || iz > storage_space.at(2) || iw > storage_space.at(3))
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_channel:wrong_type", "Location exceeds storage space in HDF file.");

    // Construct channel object from input data
    auto c = quadriga_lib::channel<float>();

    if (nrhs > 2 && mxGetNumberOfElements(prhs[2]) != 0) // Unstructured data as struct
    {
        if (!mxIsStruct(prhs[2])) // Check if the input is a struct
            mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_channel:wrong_type", "Input 'par' must be a struct.");

        int numFields = mxGetNumberOfFields(prhs[2]);
        for (int i = 0; i < numFields; ++i)
        {
            mxArray *fieldData = mxGetFieldByNumber(prhs[2], 0, i);
            if (mxGetNumberOfElements(fieldData) != 0)
            {
                const char *fieldName = mxGetFieldNameByNumber(prhs[2], i);
                std::string fieldString = "par." + std::string(fieldName);
                c.par_names.push_back(std::string(fieldName));

                if (mxIsClass(fieldData, "char"))
                {
                    auto chr = mxArrayToString(fieldData);
                    c.par_data.push_back(std::any(std::string(chr)));
                    mxFree(chr);
                }
                else
                    c.par_data.push_back(qd_mex_anycast(fieldData, fieldString, false));
            }
        }
    }

    if (nrhs > 3)
        c.rx_pos = qd_mex_typecast_Mat<float>(prhs[3]);

    if (nrhs > 4)
        c.tx_pos = qd_mex_typecast_Mat<float>(prhs[4]);

    if (nrhs > 5)
        c.coeff_re = qd_mex_matlab2vector_Cube<float>(prhs[5], 3);

    if (nrhs > 6)
        c.coeff_im = qd_mex_matlab2vector_Cube<float>(prhs[6], 3);

    if (nrhs > 8)
        c.center_frequency = qd_mex_typecast_Col<float>(prhs[8]);

    if (nrhs > 9 && mxGetNumberOfElements(prhs[9]) != 0)
    {
        if (!mxIsClass(prhs[9], "char"))
            mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_channel:wrong_type", "Channel 'name' must be a string");

        auto mx_name = mxArrayToString(prhs[9]);
        c.name = std::string(mx_name);
        mxFree(mx_name);
    }

    if (nrhs > 10)
        c.initial_position = qd_mex_get_scalar<int>(prhs[10], "initial_position", 0);

    if (nrhs > 11)
        c.path_gain = qd_mex_matlab2vector_Col<float>(prhs[11], 1);

    if (nrhs > 12)
        c.path_length = qd_mex_matlab2vector_Col<float>(prhs[12], 1);

    if (nrhs > 13)
        c.path_polarization = qd_mex_matlab2vector_Mat<float>(prhs[13], 2);

    if (nrhs > 14)
        c.path_angles = qd_mex_matlab2vector_Mat<float>(prhs[14], 2);

    if (nrhs > 15)
        c.path_fbs_pos = qd_mex_matlab2vector_Mat<float>(prhs[15], 2);

    if (nrhs > 16)
        c.path_lbs_pos = qd_mex_matlab2vector_Mat<float>(prhs[16], 2);

    if (nrhs > 17)
        c.no_interact = qd_mex_matlab2vector_Col<unsigned>(prhs[17], 1);

    if (nrhs > 18)
        c.interact_coord = qd_mex_matlab2vector_Mat<float>(prhs[18], 2);

    if (nrhs > 19)
        c.rx_orientation = qd_mex_typecast_Mat<float>(prhs[19]);

    if (nrhs > 20)
        c.tx_orientation = qd_mex_typecast_Mat<float>(prhs[20]);

    unsigned long long n_snap = c.n_snap();
    if (nrhs > 7) // Delay
    {
        unsigned long long n_dim = (unsigned long long)mxGetNumberOfDimensions(prhs[7]);
        unsigned long long n_cols = (unsigned long long)mxGetN(prhs[7]);
        if (n_dim == 2ULL && n_cols == n_snap)
        {
            auto tmp = qd_mex_matlab2vector_Cube<float>(prhs[7], 1);
            for (auto &d : tmp)
                c.delay.push_back(arma::Cube<float>(d.memptr(), 1, 1, d.n_elem, true));
        }
        else
            c.delay = qd_mex_matlab2vector_Cube<float>(prhs[7], 3);
    }

    // Prune the size of 'interact_coord'
    if (c.no_interact.size() == (size_t)n_snap && c.no_interact.size() == c.interact_coord.size())
        for (auto s = 0ULL; s < n_snap; ++s)
        {
            unsigned cnt = 0;
            for (auto &d : c.no_interact[s])
                cnt += d;

            if (c.interact_coord[s].n_cols > (unsigned long long)cnt)
                c.interact_coord[s].resize(c.interact_coord[s].n_rows, (unsigned long long)cnt);
        }

    // Write channel object to HDF5 file
    int return_code = 0;
    try
    {
        return_code = c.hdf5_write(fn, ix, iy, iz, iw);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_channel:write_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:hdf5_write_channel:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    if (return_code == 1)
        mexWarnMsgIdAndTxt("quadriga_lib:hdf5_write_channel:overwriting_exisiting_data", "Modifying or overwriting already existing dataset in file.");

    // Return storage space
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&storage_space, true);
}