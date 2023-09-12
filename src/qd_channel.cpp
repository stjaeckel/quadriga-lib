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
#include <cctype>  // For string type operations such as isalnum
#include "quadriga_lib.hpp"
#include <hdf5.h>

// Instantiate templates
template class quadriga_lib::QUADRIGA_LIB_VERSION::channel<float>;
template class quadriga_lib::QUADRIGA_LIB_VERSION::channel<double>;

// Return HDF5 version info
#define AUX(x) #x
#define STRINGIFY(x) AUX(x)
std::string quadriga_lib::get_HDF5_version()
{
    std::string str = STRINGIFY(H5_VERS_MAJOR);
    str += ".";
    str += STRINGIFY(H5_VERS_MINOR);
    str += ".";
    str += STRINGIFY(H5_VERS_RELEASE);
    return str;
}

// Helper function: File exists?
inline bool qHDF_file_exists(const std::string &name)
{
    if (FILE *file = std::fopen(name.c_str(), "r"))
    {
        fclose(file);
        return true;
    }
    else
        return false;
}

// Helper function: convert double to float
template <typename dtype>
inline void qHDF_cast_to_float(const dtype *in, float *out, unsigned n_elem)
{
    for (unsigned n = 0; n < n_elem; n++)
        out[n] = float(in[n]);
}

// Helper function: check if string contains only alphanumeric characters
bool qHDF_isalnum(const std::string &str)
{
    for (char ch : str)
        if (!std::isalnum(ch) && ch != '_')
            return false;
    return true;
}

// Helper function : get channel ID from HDF file
// Usage: 0 - Returns ID if it exists in the file, 0 otherwise
//        1 - Creates new channel ID. Reuses exisiting if it contains ONLY unstructured data
//        2 - Returns ID if it exists in the file, creates new one if it does not!
// Always returns 0 if index out of bound.
inline unsigned qHDF_get_channel_ID(hid_t file_id, unsigned ix, unsigned iy, unsigned iz, unsigned iw, unsigned usage = 0)
{
    // Read channel dims from file
    unsigned ChannelDims[4];
    hsize_t dims[4];
    hid_t dset_id = H5Dopen(file_id, "ChannelDims", H5P_DEFAULT);
    if (dset_id == H5I_INVALID_HID)
        H5Fclose(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    hid_t dspace_id = H5Dget_space(dset_id);
    unsigned ndims = H5Sget_simple_extent_ndims(dspace_id);
    if (ndims != 1)
        H5Fclose(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    H5Sget_simple_extent_dims(dspace_id, dims, NULL);
    if (dims[0] != 4)
        H5Fclose(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    H5Dread(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, ChannelDims);
    H5Sclose(dspace_id);
    H5Dclose(dset_id);

    // Check, if the requested storage location exists
    if (ix >= ChannelDims[0] || iy >= ChannelDims[1] || iz >= ChannelDims[2] || iw >= ChannelDims[3])
        return 0;

    // Load the storage index from file
    unsigned n_order = ChannelDims[0] * ChannelDims[1] * ChannelDims[2] * ChannelDims[3];
    unsigned *p_order = new unsigned[n_order];
    dset_id = H5Dopen(file_id, "Order", H5P_DEFAULT);
    if (dset_id == H5I_INVALID_HID)
        H5Fclose(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    dspace_id = H5Dget_space(dset_id);
    ndims = H5Sget_simple_extent_ndims(dspace_id);
    if (ndims != 1)
        H5Fclose(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    H5Sget_simple_extent_dims(dspace_id, dims, NULL);
    if (dims[0] != n_order)
        H5Fclose(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    H5Dread(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, p_order);

    unsigned storage_location = iw * ChannelDims[0] * ChannelDims[1] * ChannelDims[2] +
                                iz * ChannelDims[0] * ChannelDims[1] +
                                iy * ChannelDims[0] + ix;
    unsigned channel_index = p_order[storage_location];
    bool create_new = false;

    if (usage == 1)
    {
        // Check if there is an exisiting channel
        if (channel_index == 0) // No exisiting channel
            create_new = true;
        else // Check if there is channel data at this location
        {
            std::string dataset_name = "/channel_" + std::to_string(channel_index) + "/Name";
            htri_t exists = H5Lexists(file_id, dataset_name.c_str(), H5P_DEFAULT);
            if (exists > 0)
                create_new = true;
        }
    }
    else if (usage == 2 && channel_index == 0) // Create new index to store par data
        create_new = true;

    if (create_new)
    {
        // Find maximum value in storage index
        for (unsigned i = 0; i < n_order; i++)
            channel_index = p_order[i] > channel_index ? p_order[i] : channel_index;
        channel_index++;

        // Update storage index
        p_order[storage_location] = channel_index;
        H5Dwrite(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, p_order);
        H5Dclose(dset_id);
        H5Sclose(dspace_id);

        // Create group for storing channel data
        std::string group_name = "/channel_" + std::to_string(channel_index);
        hid_t group_id = H5Gcreate2(file_id, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(group_id);
    }
    delete[] p_order;
    return channel_index;
}

// Helper function to write unstructured data
inline void qHDF_write_par(hid_t group_id, const std::string *par_name, const std::any *par_data)
{
    hsize_t dims[4];
    hid_t dspace_scalar = H5Screate(H5S_SCALAR);

    if (par_name->length() == 0 || !qHDF_isalnum(*par_name))
        throw std::invalid_argument("Parameter name must only contain letters, numbers and the underscore '_'.");

    if (par_data == NULL)
        throw std::invalid_argument("NULL pointer was passed as data object.");

    htri_t exists = H5Lexists(group_id, par_name->c_str(), H5P_DEFAULT);
    if (exists > 0)
    {
        std::string error_msg = "Dataset '" + *par_name + "' already exists.";
        throw std::invalid_argument(error_msg);
    }

    // Strings
    if (par_data->type().name() == typeid(std::string).name())
    {
        auto *data = std::any_cast<std::string>(par_data);
        hid_t type_id = H5Tcopy(H5T_C_S1);
        H5Tset_size(type_id, data->length());
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), type_id, dspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->c_str());
        H5Dclose(dset_id);
        H5Tclose(type_id);
    }

    // Scalar types
    else if (par_data->type().name() == typeid(unsigned).name())
    {
        auto *data = std::any_cast<unsigned>(par_data);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_UINT, dspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(int).name())
    {
        auto *data = std::any_cast<int>(par_data);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_INT, dspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(long long).name())
    {
        auto *data = std::any_cast<long long>(par_data);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_LLONG, dspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(unsigned long long).name())
    {
        auto *data = std::any_cast<unsigned long long>(par_data);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_ULLONG, dspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(float).name())
    {
        auto *data = std::any_cast<float>(par_data);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_FLOAT, dspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(double).name())
    {
        auto *data = std::any_cast<double>(par_data);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_DOUBLE, dspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(dset_id);
    }

    // Vector types
    else if (par_data->type().name() == typeid(arma::Row<unsigned>).name())
    {
        auto *data = std::any_cast<arma::Row<unsigned>>(par_data);
        dims[0] = data->n_elem;
        hid_t dspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_UINT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Col<unsigned>).name())
    {
        auto *data = std::any_cast<arma::Col<unsigned>>(par_data);
        dims[0] = data->n_elem;
        hid_t dspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_UINT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Row<int>).name())
    {
        auto *data = std::any_cast<arma::Row<int>>(par_data);
        dims[0] = data->n_elem;
        hid_t dspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_INT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Col<int>).name())
    {
        auto *data = std::any_cast<arma::Col<int>>(par_data);
        dims[0] = data->n_elem;
        hid_t dspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_INT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Row<arma::uword>).name())
    {
        auto *data = std::any_cast<arma::Row<arma::uword>>(par_data);
        dims[0] = data->n_elem;
        hid_t dspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_UINT64, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Col<arma::uword>).name())
    {
        auto *data = std::any_cast<arma::Col<arma::uword>>(par_data);
        dims[0] = data->n_elem;
        hid_t dspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_UINT64, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Row<arma::sword>).name())
    {
        auto *data = std::any_cast<arma::Row<arma::sword>>(par_data);
        dims[0] = data->n_elem;
        hid_t dspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_INT64, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Col<arma::sword>).name())
    {
        auto *data = std::any_cast<arma::Col<arma::sword>>(par_data);
        dims[0] = data->n_elem;
        hid_t dspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_INT64, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Row<double>).name())
    {
        auto *data = std::any_cast<arma::Row<double>>(par_data);
        dims[0] = data->n_elem;
        hid_t dspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Col<double>).name())
    {
        auto *data = std::any_cast<arma::Col<double>>(par_data);
        dims[0] = data->n_elem;
        hid_t dspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Row<float>).name())
    {
        auto *data = std::any_cast<arma::Row<float>>(par_data);
        dims[0] = data->n_elem;
        hid_t dspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Col<float>).name())
    {
        auto *data = std::any_cast<arma::Col<float>>(par_data);
        dims[0] = data->n_elem;
        hid_t dspace_id = H5Screate_simple(1, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }

    // Matrix types
    else if (par_data->type().name() == typeid(arma::Mat<unsigned>).name())
    {
        auto *data = std::any_cast<arma::Mat<unsigned>>(par_data);
        dims[0] = data->n_cols;
        dims[1] = data->n_rows;
        hid_t dspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_UINT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Mat<int>).name())
    {
        auto *data = std::any_cast<arma::Mat<int>>(par_data);
        dims[0] = data->n_cols;
        dims[1] = data->n_rows;
        hid_t dspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_INT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Mat<arma::uword>).name())
    {
        auto *data = std::any_cast<arma::Mat<arma::uword>>(par_data);
        dims[0] = data->n_cols;
        dims[1] = data->n_rows;
        hid_t dspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_UINT64, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Mat<arma::sword>).name())
    {
        auto *data = std::any_cast<arma::Mat<arma::sword>>(par_data);
        dims[0] = data->n_cols;
        dims[1] = data->n_rows;
        hid_t dspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_INT64, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Mat<float>).name())
    {
        auto *data = std::any_cast<arma::Mat<float>>(par_data);
        dims[0] = data->n_cols;
        dims[1] = data->n_rows;
        hid_t dspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Mat<double>).name())
    {
        auto *data = std::any_cast<arma::Mat<double>>(par_data);
        dims[0] = data->n_cols;
        dims[1] = data->n_rows;
        hid_t dspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }

    // Cube types
    else if (par_data->type().name() == typeid(arma::Cube<unsigned>).name())
    {
        auto *data = std::any_cast<arma::Cube<unsigned>>(par_data);
        dims[0] = data->n_slices;
        dims[1] = data->n_cols;
        dims[2] = data->n_rows;
        hid_t dspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_UINT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Cube<int>).name())
    {
        auto *data = std::any_cast<arma::Cube<int>>(par_data);
        dims[0] = data->n_slices;
        dims[1] = data->n_cols;
        dims[2] = data->n_rows;
        hid_t dspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_INT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Cube<arma::uword>).name())
    {
        auto *data = std::any_cast<arma::Cube<arma::uword>>(par_data);
        dims[0] = data->n_slices;
        dims[1] = data->n_cols;
        dims[2] = data->n_rows;
        hid_t dspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_UINT64, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Cube<arma::sword>).name())
    {
        auto *data = std::any_cast<arma::Cube<arma::sword>>(par_data);
        dims[0] = data->n_slices;
        dims[1] = data->n_cols;
        dims[2] = data->n_rows;
        hid_t dspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_INT64, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Cube<float>).name())
    {
        auto *data = std::any_cast<arma::Cube<float>>(par_data);
        dims[0] = data->n_slices;
        dims[1] = data->n_cols;
        dims[2] = data->n_rows;
        hid_t dspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else if (par_data->type().name() == typeid(arma::Cube<double>).name())
    {
        auto *data = std::any_cast<arma::Cube<double>>(par_data);
        dims[0] = data->n_slices;
        dims[1] = data->n_cols;
        dims[2] = data->n_rows;
        hid_t dspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->memptr());
        H5Sclose(dspace_id);
        H5Dclose(dset_id);
    }
    else
    {
        std::string error_msg = "Unsupported data type for '" + *par_name + "'";
        throw std::invalid_argument(error_msg);
    }

    H5Sclose(dspace_scalar);
}

// CHANNEL METHODS : Return object dimensions
template <typename dtype>
arma::uword quadriga_lib::QUADRIGA_LIB_VERSION::channel<dtype>::n_snap() const
{
    return rx_pos.n_cols;
}
template <typename dtype>
arma::uword quadriga_lib::QUADRIGA_LIB_VERSION::channel<dtype>::n_rx() const
{
    if (coeff_re.size() != 0)
        return coeff_re[0].n_rows;
    else
        return 0;
}
template <typename dtype>
arma::uword quadriga_lib::QUADRIGA_LIB_VERSION::channel<dtype>::n_tx() const
{
    if (coeff_re.size() != 0)
        return coeff_re[0].n_cols;
    else
        return 0;
}
template <typename dtype>
arma::uvec quadriga_lib::QUADRIGA_LIB_VERSION::channel<dtype>::n_path() const
{
    arma::uword n_snap = rx_pos.n_cols;
    if (n_snap == 0)
        return arma::uvec();

    arma::uvec n_path(n_snap);
    arma::uword *p_path = n_path.memptr();
    if (coeff_re.size() == n_snap)
        for (arma::uword i = 0; i < n_snap; i++)
            p_path[i] = coeff_re[i].n_slices;
    else if (path_gain.size() == n_snap)
        for (arma::uword i = 0; i < n_snap; i++)
            p_path[i] = path_gain[i].n_elem;
    else if (path_length.size() == n_snap)
        for (arma::uword i = 0; i < n_snap; i++)
            p_path[i] = path_length[i].n_elem;
    else if (path_polarization.size() == n_snap)
        for (arma::uword i = 0; i < n_snap; i++)
            p_path[i] = path_polarization[i].n_cols;
    else if (path_angles.size() == n_snap)
        for (arma::uword i = 0; i < n_snap; i++)
            p_path[i] = path_angles[i].n_rows;
    else if (path_coord.size() == n_snap)
        for (arma::uword i = 0; i < n_snap; i++)
            p_path[i] = path_coord[i].n_slices;

    return n_path;
}

// CHANNEL METHOD : Validate correctness of the members
template <typename dtype>
std::string quadriga_lib::QUADRIGA_LIB_VERSION::channel<dtype>::is_valid() const
{
    if (rx_pos.n_rows != 3)
        return "'rx_pos' must have 3 rows.";

    if (rx_pos.n_cols == 0)
        return "There must be at least one entry in 'rx_pos'.";

    arma::uword n_snap = rx_pos.n_cols;
    arma::uword n_tx = 0, n_rx = 0;
    arma::uvec n_pth_v = this->n_path();
    arma::uword *n_pth = n_pth_v.memptr();

    if (tx_pos.n_rows != 3)
        return "'tx_pos' must have 3 rows.";

    if (tx_pos.n_cols != 1 && tx_pos.n_cols != n_snap)
        return "Number of columns in 'tx_pos' must be 1 or match the number of snapshots.";

    if (tx_orientation.n_elem != 0 && tx_orientation.n_rows != 3)
        return "'tx_orientation' must be empty or have 3 rows.";

    if (tx_orientation.n_elem != 0 && tx_orientation.n_cols != 1 && tx_orientation.n_cols != n_snap)
        return "Number of columns in 'tx_orientation' must be 1 or match the number of snapshots.";

    if (rx_orientation.n_elem != 0 && rx_orientation.n_rows != 3)
        return "'rx_orientation' must be empty or have 3 rows.";

    if (rx_orientation.n_elem != 0 && rx_orientation.n_cols != 1 && rx_orientation.n_cols != n_snap)
        return "Number of columns in 'rx_orientation' must be 1 or match the number of snapshots.";

    if (coeff_re.size() != 0 && coeff_re.size() != n_snap)
        return "'coeff_re' must be empty or match the number of snapshots.";

    if (coeff_re.size() == n_snap)
    {
        if (coeff_im.size() != n_snap)
            return "Imaginary part of channel coefficients 'coeff_im' is missing or incomplete.";

        if (delay.size() != n_snap)
            return "Delays are missing or incomplete.";

        n_rx = coeff_re[0].n_rows, n_tx = coeff_re[0].n_cols;

        for (arma::uword i = 0; i < n_snap; i++)
        {
            if (coeff_re[i].n_rows != n_rx || coeff_re[i].n_cols != n_tx || coeff_re[i].n_slices != n_pth[i])
                return "Size mismatch in 'coeff_re[" + std::to_string(i) + "]'.";

            if (coeff_im[i].n_rows != n_rx || coeff_im[i].n_cols != n_tx || coeff_im[i].n_slices != n_pth[i])
                return "Size mismatch in 'coeff_re[" + std::to_string(i) + "]'.";

            if ((delay[i].n_rows != 1 && delay[i].n_rows != n_rx) ||
                (delay[i].n_cols != 1 && delay[i].n_cols != n_tx) ||
                delay[i].n_slices != n_pth[i])
                return "Size mismatch in 'delay[" + std::to_string(i) + "]'.";
        }
    }

    if (path_gain.size() != 0 && path_gain.size() != n_snap)
        return "'path_gain' must be empty or match the number of snapshots.";

    if (path_gain.size() == n_snap)
        for (arma::uword i = 0; i < n_snap; i++)
            if (path_gain[i].n_elem != n_pth[i])
                return "Size mismatch in 'path_gain[" + std::to_string(i) + "]'.";

    if (path_length.size() != 0 && path_length.size() != n_snap)
        return "'path_length' must be empty or match the number of snapshots.";

    if (path_length.size() == n_snap)
        for (arma::uword i = 0; i < n_snap; i++)
            if (path_length[i].n_elem != n_pth[i])
                return "Size mismatch in 'path_length[" + std::to_string(i) + "]'.";

    if (path_polarization.size() != 0 && path_polarization.size() != n_snap)
        return "'path_polarization' must be empty or match the number of snapshots.";

    if (path_polarization.size() == n_snap)
        for (arma::uword i = 0; i < n_snap; i++)
            if (path_polarization[i].n_rows != 8 || path_polarization[i].n_cols != n_pth[i])
                return "Size mismatch in 'path_polarization[" + std::to_string(i) + "]'.";

    if (path_angles.size() != 0 && path_angles.size() != n_snap)
        return "'path_angles' must be empty or match the number of snapshots.";

    if (path_angles.size() == n_snap)
        for (arma::uword i = 0; i < n_snap; i++)
            if (path_angles[i].n_rows != n_pth[i] || path_angles[i].n_cols != 4)
                return "Size mismatch in 'path_angles[" + std::to_string(i) + "]'.";

    if (path_coord.size() != 0 && path_coord.size() != n_snap)
        return "'path_coord' must be empty or match the number of snapshots.";

    if (path_coord.size() == n_snap)
        for (arma::uword i = 0; i < n_snap; i++)
            if (path_coord[i].n_elem != 0 && (path_coord[i].n_rows != 3 || path_coord[i].n_slices != n_pth[i]))
                return "Size mismatch in 'path_coord[" + std::to_string(i) + "]'.";

    return "";
}

// Save data to HDF
template <typename dtype>
void quadriga_lib::QUADRIGA_LIB_VERSION::channel<dtype>::hdf5_write(std::string fn, unsigned ix, unsigned iy, unsigned iz, unsigned iw) const
{
    // Commonly reused variables
    hid_t file_id, dspace_id, dset_id, group_id, type_id, snap_id;
    herr_t status;
    hsize_t dims[4];
    unsigned ndims;

    // Validate the channel data
    std::string error_message = is_valid();
    if (error_message.length() != 0)
        throw std::invalid_argument(error_message.c_str());

    // Determine if we need to convert the data to float
    bool data_is_float = typeid(rx_pos).name() == typeid(arma::Mat<float>).name();
    float *data;

    // Create file
    if (!qHDF_file_exists(fn))
        quadriga_lib::hdf5_create(fn);

    // Open file for writing
    file_id = H5Fopen(fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID)
        throw std::invalid_argument("Error opening file.");

    // Get channel ID
    unsigned channel_index = qHDF_get_channel_ID(file_id, ix, iy, iz, iw, 1);
    if (channel_index == 0)
        H5Fclose(file_id), throw std::invalid_argument("Index out of bound.");

    // Open group for writing
    std::string group_name = "/channel_" + std::to_string(channel_index);
    group_id = H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);

    // Store channel name
    type_id = H5Tcopy(H5T_C_S1);
    H5Tset_size(type_id, name.length());
    dims[0] = 1;
    dspace_id = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate2(group_id, "Name", type_id, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, name.c_str());
    H5Dclose(dset_id);
    H5Sclose(dspace_id);
    H5Tclose(type_id);

    // Center frequency in [Hz]
    float tmp = float(center_frequency);
    dims[0] = 1;
    dspace_id = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate2(group_id, "CenterFrequency", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &tmp);
    H5Dclose(dset_id);
    H5Sclose(dspace_id);

    // Index of reference position
    if (initial_position != 0)
    {
        dspace_id = H5Screate_simple(1, dims, NULL);
        dset_id = H5Dcreate2(group_id, "Initial_position", H5T_NATIVE_INT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &initial_position);
        H5Dclose(dset_id);
        H5Sclose(dspace_id);
    }

    // Transmitter positions, matrix of size [3, n_snap] or [3, 1]
    dims[0] = tx_pos.n_cols;
    dims[1] = 3;
    dspace_id = H5Screate_simple(2, dims, NULL);
    dset_id = H5Dcreate2(group_id, "tx_position", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (data_is_float)
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, tx_pos.memptr());
    else
    {
        data = new float[tx_pos.n_elem];
        qHDF_cast_to_float(tx_pos.memptr(), data, tx_pos.n_elem);
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        delete[] data;
    }
    H5Dclose(dset_id);
    H5Sclose(dspace_id);

    // Receiver positions, matrix of size [3, n_snap]
    dims[0] = rx_pos.n_cols;
    dims[1] = 3;
    dspace_id = H5Screate_simple(2, dims, NULL);
    dset_id = H5Dcreate2(group_id, "rx_position", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (data_is_float)
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rx_pos.memptr());
    else
    {
        data = new float[rx_pos.n_elem];
        qHDF_cast_to_float(rx_pos.memptr(), data, rx_pos.n_elem);
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        delete[] data;
    }
    H5Dclose(dset_id);
    H5Sclose(dspace_id);

    // Transmitter orientation, matrix of size [3, n_snap] or [3, 1] or []
    if (tx_orientation.n_elem != 0)
    {
        dims[0] = tx_orientation.n_cols;
        dims[1] = 3;
        dspace_id = H5Screate_simple(2, dims, NULL);
        dset_id = H5Dcreate2(group_id, "tx_orientation", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (data_is_float)
            H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, tx_orientation.memptr());
        else
        {
            data = new float[tx_orientation.n_elem];
            qHDF_cast_to_float(tx_orientation.memptr(), data, tx_orientation.n_elem);
            H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            delete[] data;
        }
        H5Dclose(dset_id);
        H5Sclose(dspace_id);
    }

    // Receiver orientation, matrix of size [3, n_snap] or [3, 1] or []
    if (rx_orientation.n_elem != 0)
    {
        dims[0] = rx_orientation.n_cols;
        dims[1] = 3;
        dspace_id = H5Screate_simple(2, dims, NULL);
        dset_id = H5Dcreate2(group_id, "rx_orientation", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (data_is_float)
            H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rx_orientation.memptr());
        else
        {
            data = new float[rx_orientation.n_elem];
            qHDF_cast_to_float(rx_orientation.memptr(), data, rx_orientation.n_elem);
            H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            delete[] data;
        }
        H5Dclose(dset_id);
        H5Sclose(dspace_id);
    }

    // Snapshot data
    arma::uword n_snap = this->n_snap();
    for (arma::uword i = 0; i < n_snap; i++)
    {
        if (n_snap == 1)
            snap_id = group_id;
        else
        {
            std::string snap_name = "Snap_" + std::to_string(i);
            snap_id = H5Gcreate2(group_id, snap_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        }

        // Channel coefficients, vector (n_snap) of tensors of size [n_rx, n_tx, n_path]
        if (coeff_re.size() == n_snap && coeff_im.size() == n_snap && coeff_re[i].n_elem != 0 && coeff_im[i].n_elem != 0)
        {
            dims[0] = coeff_re[i].n_slices;
            dims[1] = coeff_re[i].n_cols;
            dims[2] = coeff_re[i].n_rows;
            dspace_id = H5Screate_simple(3, dims, NULL);

            // Real part
            dset_id = H5Dcreate2(snap_id, "coeff_re", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (data_is_float)
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, coeff_re[i].memptr());
            else
            {
                data = new float[coeff_re[i].n_elem];
                qHDF_cast_to_float(coeff_re[i].memptr(), data, coeff_re[i].n_elem);
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                delete[] data;
            }
            H5Dclose(dset_id);

            // Imaginary part
            dset_id = H5Dcreate2(snap_id, "coeff_im", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (data_is_float)
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, coeff_im[i].memptr());
            else
            {
                data = new float[coeff_im[i].n_elem];
                qHDF_cast_to_float(coeff_im[i].memptr(), data, coeff_im[i].n_elem);
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                delete[] data;
            }
            H5Dclose(dset_id);
            H5Sclose(dspace_id);
        }

        // Path delays in seconds, vector (n_snap) of tensors of size [n_rx, n_tx, n_path]
        if (delay.size() == n_snap && delay[i].n_elem != 0)
        {
            dims[0] = delay[i].n_slices;
            dims[1] = delay[i].n_cols;
            dims[2] = delay[i].n_rows;
            dspace_id = H5Screate_simple(3, dims, NULL);
            dset_id = H5Dcreate2(snap_id, "delay", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (data_is_float)
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, delay[i].memptr());
            else
            {
                data = new float[delay[i].n_elem];
                qHDF_cast_to_float(delay[i].memptr(), data, delay[i].n_elem);
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                delete[] data;
            }
            H5Dclose(dset_id);
            H5Sclose(dspace_id);
        }

        // Path gain before antenna patterns, vector (n_snap) of vectors of length [n_path]
        if (path_gain.size() == n_snap && path_gain[i].n_elem != 0)
        {
            dims[0] = path_gain[i].n_elem;
            dspace_id = H5Screate_simple(1, dims, NULL);
            dset_id = H5Dcreate2(snap_id, "path_gain", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (data_is_float)
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, path_gain[i].memptr());
            else
            {
                data = new float[path_gain[i].n_elem];
                qHDF_cast_to_float(path_gain[i].memptr(), data, path_gain[i].n_elem);
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                delete[] data;
            }
            H5Dclose(dset_id);
            H5Sclose(dspace_id);
        }

        // Absolute path length from TX to RX phase center, vector (n_snap) of vectors of length [n_path]
        if (path_length.size() == n_snap && path_length[i].n_elem != 0)
        {
            dims[0] = path_length[i].n_elem;
            dspace_id = H5Screate_simple(1, dims, NULL);
            dset_id = H5Dcreate2(snap_id, "path_length", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (data_is_float)
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, path_length[i].memptr());
            else
            {
                data = new float[path_length[i].n_elem];
                qHDF_cast_to_float(path_length[i].memptr(), data, path_length[i].n_elem);
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                delete[] data;
            }
            H5Dclose(dset_id);
            H5Sclose(dspace_id);
        }

        // Polarization transfer function, vector (n_snap) of matrices of size [8, n_path], interleaved complex
        if (path_polarization.size() == n_snap && path_polarization[i].n_elem != 0)
        {
            dims[0] = path_polarization[i].n_cols;
            dims[1] = path_polarization[i].n_rows;
            dspace_id = H5Screate_simple(2, dims, NULL);
            dset_id = H5Dcreate2(snap_id, "path_polarization", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (data_is_float)
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, path_polarization[i].memptr());
            else
            {
                data = new float[path_polarization[i].n_elem];
                qHDF_cast_to_float(path_polarization[i].memptr(), data, path_polarization[i].n_elem);
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                delete[] data;
            }
            H5Dclose(dset_id);
            H5Sclose(dspace_id);
        }

        // Departure and arrival angles, vector (n_snap) of matrices of size [n_path, 4], {AOD, EOD, AOA, EOA}
        if (path_angles.size() == n_snap && path_angles[i].n_elem != 0)
        {
            dims[0] = path_angles[i].n_cols;
            dims[1] = path_angles[i].n_rows;
            dspace_id = H5Screate_simple(2, dims, NULL);
            dset_id = H5Dcreate2(snap_id, "path_angles", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (data_is_float)
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, path_angles[i].memptr());
            else
            {
                data = new float[path_angles[i].n_elem];
                qHDF_cast_to_float(path_angles[i].memptr(), data, path_angles[i].n_elem);
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                delete[] data;
            }
            H5Dclose(dset_id);
            H5Sclose(dspace_id);
        }

        // Interaction coordinates, NAN-padded, vector (n_snap) of tensors of size [3, n_coord, n_path]
        if (path_coord.size() == n_snap && path_coord[i].n_elem != 0)
        {
            dims[0] = path_coord[i].n_slices;
            dims[1] = path_coord[i].n_cols;
            dims[2] = path_coord[i].n_rows;
            dspace_id = H5Screate_simple(3, dims, NULL);
            dset_id = H5Dcreate2(snap_id, "path_coord", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (data_is_float)
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, path_coord[i].memptr());
            else
            {
                data = new float[path_coord[i].n_elem];
                qHDF_cast_to_float(path_coord[i].memptr(), data, path_coord[i].n_elem);
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                delete[] data;
            }
            H5Dclose(dset_id);
            H5Sclose(dspace_id);
        }

        if (n_snap != 1)
            H5Gclose(snap_id);
    }

    // Write unstructured data
    for (std::size_t i = 0; i < par_names.size(); i++)
    {
        std::string par_name = "par_" + par_names[i];
        qHDF_write_par(group_id, &par_name, &par_data[i]);
    }

    // Close the group and file
    H5Gclose(group_id);
    H5Fclose(file_id);
}

// Create a new HDF file and set the index to to given storage layout
void quadriga_lib::hdf5_create(std::string fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    herr_t status;
    hid_t file_id, dspace_id, dset_id;
    hsize_t dims[4];

    if (nx == 0 || ny == 0 || nz == 0 || nw == 0)
        throw std::invalid_argument("Storage layout dimension sizes cannot be 0.");

    // Create and open file
    file_id = H5Fcreate(fn.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID) // May be invalid if file is currently opened
        throw std::invalid_argument("Error creating file. It may be already opened already.");

    // Store dimension size to file
    unsigned ChannelDims[4];
    ChannelDims[0] = nx, ChannelDims[1] = ny, ChannelDims[2] = nz, ChannelDims[3] = nw;
    dims[0] = 4;
    dspace_id = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate2(file_id, "ChannelDims", H5T_NATIVE_UINT32, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, ChannelDims);
    status = H5Sclose(dspace_id);
    status = H5Dclose(dset_id);

    // Create the index
    unsigned n_order = nx * ny * nz * nw;
    unsigned *p_order = new unsigned[n_order]();
    dims[0] = n_order;
    dspace_id = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate2(file_id, "Order", H5T_NATIVE_UINT32, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, p_order);
    status = H5Sclose(dspace_id);
    status = H5Dclose(dset_id);
    delete[] p_order;

    // Close the file
    status = H5Fclose(file_id);
}

// Writes unstructured data to a hdf5 file
void quadriga_lib::hdf5_write_unstructured(std::string fn, std::string par_name, const std::any *par_data,
                                           unsigned ix, unsigned iy, unsigned iz, unsigned iw)
{
    // Create file
    if (!qHDF_file_exists(fn))
        quadriga_lib::hdf5_create(fn);

    // Open file for writing
    hid_t file_id = H5Fopen(fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID)
        throw std::invalid_argument("Error opening file.");

    // Get channel ID
    unsigned channel_index = qHDF_get_channel_ID(file_id, ix, iy, iz, iw, 2);
    if (channel_index == 0)
        H5Fclose(file_id), throw std::invalid_argument("Index out of bound.");

    // Open group for writing
    std::string group_name = "/channel_" + std::to_string(channel_index);
    hid_t group_id = H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);

    // Write unstructured data to file
    qHDF_write_par(group_id, &par_name, par_data);

    // Close the group and file
    H5Gclose(group_id);
    H5Fclose(file_id);
}

// void quadriga_lib::qd_channel_hello()
// {
//     std::cout << H5_VERS_INFO << std::endl;

//     hid_t file_id, dset_id, dspace_id, group_id; /* identifiers */
//     herr_t status;

//     // Setup dataset dimensions and input data
//     int ndims = 1;
//     hsize_t dims[1];
//     dims[0] = 50;
//     std::vector<double> data(50, 1);

//     // Open a file
//     file_id = H5Fcreate("new_file.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

//     // Create a group
//     group_id = H5Gcreate2(file_id, "/group", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

//     // Create a dataset
//     dspace_id = H5Screate_simple(1, dims, NULL);
//     dset_id = H5Dcreate2(group_id, "dset1", H5T_STD_I32BE, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

//     // Write the data
//     status = H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

//     // Close dataset after writing
//     status = H5Dclose(dset_id);

//     // Retrieve result size and preallocate vector
//     std::vector<double> result;
//     dset_id = H5Dopen(file_id, "/group/dset1", H5P_DEFAULT);
//     dspace_id = H5Dget_space(dset_id);
//     ndims = H5Sget_simple_extent_ndims(dspace_id);
//     hsize_t res_dims[1];
//     status = H5Sget_simple_extent_dims(dspace_id, res_dims, NULL);
//     int res_sz = 1;
//     for (int i = 0; i < 1; i++)
//     {
//         res_sz *= res_dims[i];
//     }
//     result.resize(res_sz);

//     // Read the data
//     status = H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data());

//     // Close the dataset and group
//     status = H5Dclose(dset_id);
//     status = H5Gclose(group_id);

//     // Close the file
//     status = H5Fclose(file_id);
// }
