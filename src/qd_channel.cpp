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

#include <stdexcept>
#include <cstring> // For std::memcopy
#include "quadriga_lib.hpp"

#include <hdf5.h>

// Instantiate templates
template class quadriga_lib::QUADRIGA_LIB_VERSION::channel<float>;
template class quadriga_lib::QUADRIGA_LIB_VERSION::channel<double>;

void quadriga_lib::qd_channel_hello()
{
    std::cout << H5_VERS_INFO << std::endl;

    hid_t file_id, dset_id, dspace_id, group_id; /* identifiers */
    herr_t status;

    // Setup dataset dimensions and input data
    int ndims = 1;
    hsize_t dims[1];
    dims[0] = 50;
    std::vector<double> data(50, 1);

    // Open a file
    file_id = H5Fcreate("new_file.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Create a group
    group_id = H5Gcreate2(file_id, "/group", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Create a dataset
    dspace_id = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate2(
        group_id, "dset1", H5T_STD_I32BE, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Write the data
    status = H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

    // Close dataset after writing
    status = H5Dclose(dset_id);

    // Retrieve result size and preallocate vector
    std::vector<double> result;
    dset_id = H5Dopen(file_id, "/group/dset1", H5P_DEFAULT);
    dspace_id = H5Dget_space(dset_id);
    ndims = H5Sget_simple_extent_ndims(dspace_id);
    hsize_t res_dims[1];
    status = H5Sget_simple_extent_dims(dspace_id, res_dims, NULL);
    int res_sz = 1;
    for (int i = 0; i < 1; i++) {
        res_sz *= res_dims[i];
    }
    result.resize(res_sz);

    // Read the data
    status = H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data());

    // Close the dataset and group
    status = H5Dclose(dset_id);
    status = H5Gclose(group_id);

    // Close the file
    status = H5Fclose(file_id);
}