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
#include <map>

#include <hdf5.h>
#include "quadriga_channel.hpp"

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

// Close file
inline void qHDF_close_file(hid_t file_id)
{
    if (file_id == H5I_INVALID_HID)
        throw std::invalid_argument("Error closing file. Invalid File ID.");

    hid_t *obj_ids;

    // Check if the file is opened only once
    ssize_t count = H5Fget_obj_count(file_id, H5F_OBJ_FILE);
    if (count > 1)
        throw std::invalid_argument("Error closing file. File is opened multiple times.");

    // Close all attributes
    count = H5Fget_obj_count(file_id, H5F_OBJ_ATTR);
    if (count > 0)
    {
        obj_ids = new hid_t[count];
        H5Fget_obj_ids(file_id, H5F_OBJ_ATTR, count, obj_ids);
        for (ssize_t i = 0; i < count; ++i)
            H5Aclose(obj_ids[i]);
        delete[] obj_ids;
    }

    // Close all datatypes
    count = H5Fget_obj_count(file_id, H5F_OBJ_DATATYPE);
    if (count > 0)
    {
        obj_ids = new hid_t[count];
        H5Fget_obj_ids(file_id, H5F_OBJ_DATATYPE, count, obj_ids);
        for (ssize_t i = 0; i < count; ++i)
            H5Tclose(obj_ids[i]);
        delete[] obj_ids;
    }

    // Close all datasets
    count = H5Fget_obj_count(file_id, H5F_OBJ_DATASET);
    if (count > 0)
    {
        obj_ids = new hid_t[count];
        H5Fget_obj_ids(file_id, H5F_OBJ_DATASET, count, obj_ids);
        for (ssize_t i = 0; i < count; ++i)
            H5Dclose(obj_ids[i]);
        delete[] obj_ids;
    }

    // Close all groups
    count = H5Fget_obj_count(file_id, H5F_OBJ_GROUP);
    if (count > 0)
    {
        obj_ids = new hid_t[count];
        H5Fget_obj_ids(file_id, H5F_OBJ_GROUP, count, obj_ids);
        for (ssize_t i = 0; i < count; ++i)
            H5Gclose(obj_ids[i]);
        delete[] obj_ids;
    }

    // Check if we got all of them
    count = H5Fget_obj_count(file_id, H5F_OBJ_ALL);
    if (count > 1)
        throw std::invalid_argument("Error closing file. Objects are still open.");
    else if (count == 1)   // Only the file_id is left
        H5Fclose(file_id); // Bye
}

// Helper function: convert double to float
template <typename dtype>
inline void qHDF_cast_to_float(const dtype *in, float *out, unsigned long long n_elem)
{
    for (auto n = 0ULL; n < n_elem; ++n)
        out[n] = float(in[n]);
}

// Helper function: convert float to double
template <typename dtype>
inline void qHDF_cast_to_double(const dtype *in, double *out, unsigned long long n_elem)
{
    for (auto n = 0ULL; n < n_elem; ++n)
        out[n] = double(in[n]);
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
        qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    hid_t dspace_id = H5Dget_space(dset_id);
    int ndims = H5Sget_simple_extent_ndims(dspace_id);
    if (ndims != 1)
        qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    H5Sget_simple_extent_dims(dspace_id, dims, NULL);
    if (dims[0] != 4)
        qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    H5Dread(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, ChannelDims);
    H5Sclose(dspace_id);
    H5Dclose(dset_id);

    // Always returns 0 if index out of bound
    if (ix >= ChannelDims[0] || iy >= ChannelDims[1] || iz >= ChannelDims[2] || iw >= ChannelDims[3])
        return 0;

    // Load the storage index from file
    unsigned n_order = ChannelDims[0] * ChannelDims[1] * ChannelDims[2] * ChannelDims[3];
    unsigned *p_order = new unsigned[n_order];
    dset_id = H5Dopen(file_id, "Order", H5P_DEFAULT);
    if (dset_id == H5I_INVALID_HID)
        qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    dspace_id = H5Dget_space(dset_id);
    ndims = H5Sget_simple_extent_ndims(dspace_id);
    if (ndims != 1)
        qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    H5Sget_simple_extent_dims(dspace_id, dims, NULL);
    if (dims[0] != n_order)
        qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
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
        for (unsigned i = 0; i < n_order; ++i)
            channel_index = p_order[i] > channel_index ? p_order[i] : channel_index;
        ++channel_index;

        // Update storage index
        p_order[storage_location] = channel_index;
        H5Dwrite(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, p_order);

        // Create group for storing channel data
        std::string group_name = "/channel_" + std::to_string(channel_index);
        hid_t gcpl = H5Pcreate(H5P_GROUP_CREATE); // Group creation property list
        H5Pset_link_creation_order(gcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);
        hid_t group_id = H5Gcreate2(file_id, group_name.c_str(), H5P_DEFAULT, gcpl, H5P_DEFAULT);
        H5Gclose(group_id);
        H5Pclose(gcpl);
    }

    H5Dclose(dset_id);
    H5Sclose(dspace_id);

    delete[] p_order;
    return channel_index;
}

// Helper function to write unstructured data
inline void qHDF_write_par(hid_t file_id, hid_t group_id, const std::string *par_name, const std::any *par_data)
{

    if (par_name->length() == 0 || !qHDF_isalnum(*par_name))
        qHDF_close_file(file_id), throw std::invalid_argument("Parameter name must only contain letters, numbers and the underscore '_'.");

    if (par_data == nullptr)
        qHDF_close_file(file_id), throw std::invalid_argument("NULL pointer was passed as data object.");

    htri_t exists = H5Lexists(group_id, par_name->c_str(), H5P_DEFAULT);
    if (exists > 0)
    {
        std::string error_msg = "Dataset '" + *par_name + "' already exists.";
        qHDF_close_file(file_id), throw std::invalid_argument(error_msg);
    }

    // Get low-level access to the data
    unsigned long long dims_arma[3];
    hsize_t dims[3] = {1, 1, 1};
    void *dataptr;
    int datatype_id = quadriga_lib::any_type_id(par_data, dims_arma, &dataptr);

    // Strings
    if (datatype_id == 9)
    {
        hid_t type_id = H5Tcopy(H5T_C_S1);
        H5Tset_size(type_id, (hsize_t)dims_arma[0]);
        hid_t dspace_scalar = H5Screate(H5S_SCALAR);
        hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), type_id, dspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataptr);
        H5Dclose(dset_id);
        H5Sclose(dspace_scalar);
        H5Tclose(type_id);
        return;
    }

    // Mapping datatype_ids to HDF5 datatypes
    std::map<int, hid_t> datatype_map = {
        {10, H5T_NATIVE_FLOAT}, // Scalars
        {11, H5T_NATIVE_DOUBLE},
        {12, H5T_NATIVE_ULLONG},
        {13, H5T_NATIVE_LLONG},
        {14, H5T_NATIVE_UINT},
        {15, H5T_NATIVE_INT},
        {20, H5T_NATIVE_FLOAT}, // Matrices
        {21, H5T_NATIVE_DOUBLE},
        {22, H5T_NATIVE_ULLONG},
        {23, H5T_NATIVE_LLONG},
        {24, H5T_NATIVE_UINT},
        {25, H5T_NATIVE_INT},
        {30, H5T_NATIVE_FLOAT}, // Cubes
        {31, H5T_NATIVE_DOUBLE},
        {32, H5T_NATIVE_ULLONG},
        {33, H5T_NATIVE_LLONG},
        {34, H5T_NATIVE_UINT},
        {35, H5T_NATIVE_INT},
        {40, H5T_NATIVE_FLOAT}, // Column vectors
        {41, H5T_NATIVE_DOUBLE},
        {42, H5T_NATIVE_ULLONG},
        {43, H5T_NATIVE_LLONG},
        {44, H5T_NATIVE_UINT},
        {45, H5T_NATIVE_INT},
        {50, H5T_NATIVE_FLOAT}, // Row vectors
        {51, H5T_NATIVE_DOUBLE},
        {52, H5T_NATIVE_ULLONG},
        {53, H5T_NATIVE_LLONG},
        {54, H5T_NATIVE_UINT},
        {55, H5T_NATIVE_INT}};

    auto it = datatype_map.find(datatype_id);
    if (it != datatype_map.end())
    {
        hid_t dataType = it->second;
        if (it->first < 20) // Scalar types
        {
            hid_t dspace_scalar = H5Screate(H5S_SCALAR);
            hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), dataType, dspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataptr);
            H5Dclose(dset_id);
            H5Sclose(dspace_scalar);
        }
        else if (it->first < 30) // Matrix types
        {
            dims[0] = (hsize_t)dims_arma[1], dims[1] = (hsize_t)dims_arma[0];
            hid_t dspace_id = H5Screate_simple(2, dims, NULL);
            hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), dataType, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataptr);
            H5Sclose(dspace_id);
            H5Dclose(dset_id);
        }
        else if (it->first < 40) // Cube types
        {
            dims[0] = (hsize_t)dims_arma[2], dims[1] = (hsize_t)dims_arma[1], dims[2] = (hsize_t)dims_arma[0];
            hid_t dspace_id = H5Screate_simple(3, dims, NULL);
            hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), dataType, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataptr);
            H5Sclose(dspace_id);
            H5Dclose(dset_id);
        }
        else if (it->first < 60) // Vector types (Col and Row)
        {
            dims[0] = (hsize_t)dims_arma[0];
            hid_t dspace_id = H5Screate_simple(1, dims, NULL);
            hid_t dset_id = H5Dcreate2(group_id, par_name->c_str(), dataType, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataptr);
            H5Sclose(dspace_id);
            H5Dclose(dset_id);
        }
    }
    else
    {
        std::string error_msg = "Unsupported data type for '" + *par_name + "'";
        qHDF_close_file(file_id), throw std::invalid_argument(error_msg);
    }
}

// Helper function to write unstructured data
inline long long qHDF_read_par_names(hid_t group_id, std::vector<std::string> *par_names, const std::string *prefix = nullptr)
{
    // Get information about the group
    H5G_info_t group_info;
    if (H5Gget_info(group_id, &group_info) < 0)
        return -1LL; // Error

    size_t prefix_length = 0;
    if (prefix != nullptr)
        prefix_length = prefix->length();

    // Traverse the objects in the group by index
    long long no_par_names = 0LL;
    for (hsize_t i = 0; i < group_info.nlinks; ++i)
    {
        // Get the name of the object by its creation order
        char obj_name[256]; // Adjust size if needed
        ssize_t name_len = H5Lget_name_by_idx(group_id, ".", H5_INDEX_CRT_ORDER, H5_ITER_INC, i, obj_name, sizeof(obj_name), H5P_DEFAULT);
        if (name_len < 0)
            return -1LL; // Error

        // Get the type of the object by its creation order
        H5G_obj_t obj_type = H5Gget_objtype_by_idx(group_id, i);

        // Check if the object is not a group
        if (obj_type != H5G_GROUP)
        {
            if (prefix_length == 0)
                ++no_par_names, par_names->push_back(obj_name);

            else if (std::strncmp(obj_name, prefix->c_str(), prefix_length) == 0)
                ++no_par_names, par_names->push_back(obj_name + prefix_length);
        }
    }
    return no_par_names;
}

// Helper function to read a dataset from a HDF file
// Returns empty std::any on error
inline std::any qHDF_read_data(hid_t group_id, std::string dataset_name, bool float2double = false)
{
    if (group_id == H5I_INVALID_HID)
        return std::any();

    // Check if dataset exists
    htri_t exists = H5Lexists(group_id, dataset_name.c_str(), H5P_DEFAULT);
    if (exists <= 0)
        return std::any();

    // Open dataset
    hid_t dset_id = H5Dopen(group_id, dataset_name.c_str(), H5P_DEFAULT);

    if (dset_id == H5I_INVALID_HID)
        return std::any();

    // Obtain the data type id
    hid_t datatype_id = H5Dget_type(dset_id);
    if (datatype_id == H5I_INVALID_HID)
    {
        H5Dclose(dset_id);
        return std::any();
    }

    // Evaluate dataspace
    hid_t dspace_id = H5Dget_space(dset_id);
    if (dspace_id == H5I_INVALID_HID)
    {
        H5Tclose(datatype_id);
        H5Dclose(dset_id);
        return std::any();
    }

    H5S_class_t space_class = H5Sget_simple_extent_type(dspace_id);
    H5T_class_t type_class = H5Tget_class(datatype_id);

    int ndims = 1;
    hsize_t dims[3] = {1, 1, 1};
    if (space_class == H5S_SIMPLE)
        ndims = H5Sget_simple_extent_ndims(dspace_id),
        H5Sget_simple_extent_dims(dspace_id, dims, NULL);
    else if (space_class != H5S_SCALAR)
        ndims = 0;
    H5Sclose(dspace_id);

    if (ndims == 1 && dims[0] == 1) // Scalar types
    {
        if (type_class == H5T_STRING)
        {
            std::string value;
            size_t size = H5Tget_size(datatype_id);
            value.resize(size);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.data());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_FLOAT))
        {
            float value;
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            if (float2double)
                return (double)value;
            else
                return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_DOUBLE))
        {
            double value;
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_ULLONG))
        {
            unsigned long long int value;
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_LLONG))
        {
            long long int value;
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_UINT))
        {
            unsigned int value;
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_INT))
        {
            int value;
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }
    }
    else if (ndims == 1) // Vector types
    {
        if (H5Tequal(datatype_id, H5T_NATIVE_FLOAT))
        {
            if (float2double)
            {
                arma::Col<double> value((arma::uword)dims[0], arma::fill::none);
                float *data = new float[value.n_elem];
                H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                qHDF_cast_to_double(data, value.memptr(), value.n_elem);
                delete[] data;
                H5Tclose(datatype_id);
                H5Dclose(dset_id);
                return value;
            }
            else
            {
                arma::Col<float> value((arma::uword)dims[0], arma::fill::none);
                H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
                H5Tclose(datatype_id);
                H5Dclose(dset_id);
                return value;
            }
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_DOUBLE))
        {
            arma::Col<double> value((arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_ULLONG))
        {
            arma::Col<unsigned long long int> value((arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_LLONG))
        {
            arma::Col<long long int> value((arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_UINT))
        {
            arma::Col<unsigned int> value((arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_INT))
        {
            arma::Col<int> value((arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }
    }
    else if (ndims == 2) // Matrix types
    {
        if (H5Tequal(datatype_id, H5T_NATIVE_FLOAT))
        {
            if (float2double)
            {
                arma::Mat<double> value((arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
                float *data = new float[value.n_elem];
                H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                qHDF_cast_to_double(data, value.memptr(), value.n_elem);
                delete[] data;
                H5Tclose(datatype_id);
                H5Dclose(dset_id);
                return value;
            }
            else
            {
                arma::Mat<float> value((arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
                H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
                H5Tclose(datatype_id);
                H5Dclose(dset_id);
                return value;
            }
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_DOUBLE))
        {
            arma::Mat<double> value((arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_ULLONG))
        {
            arma::Mat<unsigned long long int> value((arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_LLONG))
        {
            arma::Mat<long long int> value((arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_UINT))
        {
            arma::Mat<unsigned int> value((arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_INT))
        {
            arma::Mat<int> value((arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }
    }
    else if (ndims == 3) // Cube types
    {
        if (H5Tequal(datatype_id, H5T_NATIVE_FLOAT))
        {
            if (float2double)
            {
                arma::Cube<double> value((arma::uword)dims[2], (arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
                float *data = new float[value.n_elem];
                H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                qHDF_cast_to_double(data, value.memptr(), value.n_elem);
                delete[] data;
                H5Tclose(datatype_id);
                H5Dclose(dset_id);
                return value;
            }
            else
            {
                arma::Cube<float> value((arma::uword)dims[2], (arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
                H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
                H5Tclose(datatype_id);
                H5Dclose(dset_id);
                return value;
            }
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_DOUBLE))
        {
            arma::Cube<double> value((arma::uword)dims[2], (arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_ULLONG))
        {
            arma::Cube<unsigned long long int> value((arma::uword)dims[2], (arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_LLONG))
        {
            arma::Cube<long long int> value((arma::uword)dims[2], (arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_UINT))
        {
            arma::Cube<unsigned int> value((arma::uword)dims[2], (arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }

        if (H5Tequal(datatype_id, H5T_NATIVE_INT))
        {
            arma::Cube<int> value((arma::uword)dims[2], (arma::uword)dims[1], (arma::uword)dims[0], arma::fill::none);
            H5Dread(dset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.memptr());
            H5Tclose(datatype_id);
            H5Dclose(dset_id);
            return value;
        }
    }

    H5Tclose(datatype_id);
    H5Dclose(dset_id);
    return std::any();
}

// CHANNEL METHODS : Return object dimensions
template <typename dtype>
unsigned long long quadriga_lib::channel<dtype>::n_snap() const
{
    auto s = 0ULL;

    if (center_frequency.n_elem > s) // 1 or s
        s = center_frequency.n_elem;

    if (tx_orientation.n_cols > s) // 1 or s
        s = tx_orientation.n_cols;

    if (rx_orientation.n_cols > s) // 1 or s
        s = rx_orientation.n_cols;

    if (rx_pos.n_cols > s) // 1 or s
        s = rx_pos.n_cols;

    if (tx_pos.n_cols > s) // 1 or s
        s = tx_pos.n_cols;

    if (coeff_re.size() > (size_t)s)
        s = (unsigned long long)coeff_re.size();

    if (delay.size() > (size_t)s)
        s = (unsigned long long)delay.size();

    if (path_gain.size() > (size_t)s)
        s = (unsigned long long)path_gain.size();

    if (path_length.size() > (size_t)s)
        s = (unsigned long long)path_length.size();

    if (path_polarization.size() > (size_t)s)
        s = (unsigned long long)path_polarization.size();

    if (path_angles.size() > (size_t)s)
        s = (unsigned long long)path_angles.size();

    if (path_fbs_pos.size() > (size_t)s)
        s = (unsigned long long)path_fbs_pos.size();

    if (path_lbs_pos.size() > (size_t)s)
        s = (unsigned long long)path_lbs_pos.size();

    if (no_interact.size() > (size_t)s)
        s = (unsigned long long)no_interact.size();

    return s;
}

template <typename dtype>
unsigned long long quadriga_lib::channel<dtype>::n_rx() const
{
    if (coeff_re.size() != 0)
        return coeff_re[0].n_rows;
    else
        return 0ULL;
}

template <typename dtype>
unsigned long long quadriga_lib::channel<dtype>::n_tx() const
{
    if (coeff_re.size() != 0)
        return coeff_re[0].n_cols;
    else
        return 0ULL;
}

template <typename dtype>
arma::uvec quadriga_lib::channel<dtype>::n_path() const
{
    unsigned long long n_snap = this->n_snap();
    if (n_snap == 0)
        return arma::uvec();

    arma::uvec n_path(n_snap);
    unsigned long long *p_path = n_path.memptr();
    if (coeff_re.size() == n_snap)
        for (auto i = 0ULL; i < n_snap; ++i)
            p_path[i] = coeff_re[i].n_slices;
    else if (path_gain.size() == n_snap)
        for (auto i = 0ULL; i < n_snap; ++i)
            p_path[i] = path_gain[i].n_elem;
    else if (path_length.size() == n_snap)
        for (auto i = 0ULL; i < n_snap; ++i)
            p_path[i] = path_length[i].n_elem;
    else if (path_polarization.size() == n_snap)
        for (auto i = 0ULL; i < n_snap; ++i)
            p_path[i] = path_polarization[i].n_cols;
    else if (path_angles.size() == n_snap)
        for (auto i = 0ULL; i < n_snap; ++i)
            p_path[i] = path_angles[i].n_rows;
    else if (no_interact.size() == n_snap)
        for (auto i = 0ULL; i < n_snap; ++i)
            p_path[i] = no_interact[i].n_elem;

    return n_path;
}

// Returns true if the channel object contains no data
template <typename dtype>
bool quadriga_lib::channel<dtype>::empty() const
{
    if (name != "empty")
        return false;

    if (center_frequency.n_elem != 0ULL)
        return false;

    if (tx_pos.n_elem != 0ULL)
        return false;

    if (rx_pos.n_elem != 0ULL)
        return false;

    if (tx_orientation.n_elem != 0ULL)
        return false;

    if (rx_orientation.n_elem != 0ULL)
        return false;

    if (coeff_re.size() != 0)
        return false;

    if (delay.size() != 0)
        return false;

    if (path_gain.size() != 0)
        return false;

    if (path_length.size() != 0)
        return false;

    if (path_polarization.size() != 0)
        return false;

    if (path_angles.size() != 0)
        return false;

    if (path_fbs_pos.size() != 0)
        return false;

    if (path_lbs_pos.size() != 0)
        return false;

    if (no_interact.size() != 0)
        return false;

    if (interact_coord.size() != 0)
        return false;

    if (initial_position != 0)
        return false;

    return true;
}

// CHANNEL METHOD : Validate correctness of the members
template <typename dtype>
std::string quadriga_lib::channel<dtype>::is_valid() const
{
    unsigned long long n_snap = this->n_snap();
    unsigned long long n_tx = 0ULL, n_rx = 0ULL;
    arma::uvec n_pth_v = this->n_path();

    if (n_pth_v.n_elem != n_snap)
        return "Number of elements returned by 'n_path()' does not match number of snapshots.";
    unsigned long long *n_pth = n_pth_v.memptr();

    if (n_snap != 0ULL && rx_pos.n_rows != 3ULL)
        return "'rx_pos' is missing or ill-formatted (must have 3 rows).";

    if (rx_pos.n_cols != 1ULL && rx_pos.n_cols != n_snap)
        return "Number of columns in 'rx_pos' must be 1 or match the number of snapshots.";

    if (n_snap != 0UL && tx_pos.n_rows != 3ULL)
        return "'tx_pos' is missing or ill-formatted (must have 3 rows).";

    if (tx_pos.n_cols != 1ULL && tx_pos.n_cols != n_snap)
        return "Number of columns in 'tx_pos' must be 1 or match the number of snapshots.";

    if (center_frequency.n_elem != 0ULL && center_frequency.n_elem != 1ULL && center_frequency.n_elem != n_snap)
        return "Number of entries in 'center_frequency' must be 0, 1 or match the number of snapshots.";

    if (tx_orientation.n_elem != 0ULL && tx_orientation.n_rows != 3ULL)
        return "'tx_orientation' must be empty or have 3 rows.";

    if (tx_orientation.n_elem != 0ULL && tx_orientation.n_cols != 1ULL && tx_orientation.n_cols != n_snap)
        return "Number of columns in 'tx_orientation' must be 1 or match the number of snapshots.";

    if (rx_orientation.n_elem != 0ULL && rx_orientation.n_rows != 3ULL)
        return "'rx_orientation' must be empty or have 3 rows.";

    if (rx_orientation.n_elem != 0ULL && rx_orientation.n_cols != 1ULL && rx_orientation.n_cols != n_snap)
        return "Number of columns in 'rx_orientation' must be 1 or match the number of snapshots.";

    if (coeff_re.size() != 0 && coeff_re.size() != n_snap)
        return "'coeff_re' must be empty or match the number of snapshots.";

    if (n_snap != 0ULL && coeff_re.size() == n_snap)
    {
        if (coeff_im.size() != n_snap)
            return "Imaginary part of channel coefficients 'coeff_im' is missing or incomplete.";

        if (delay.size() != n_snap)
            return "Delays are missing or incomplete.";

        n_rx = coeff_re[0].n_rows, n_tx = coeff_re[0].n_cols;

        for (auto i = 0ULL; i < n_snap; ++i)
        {
            if (coeff_re[i].n_rows != n_rx || coeff_re[i].n_cols != n_tx || coeff_re[i].n_slices != n_pth[i])
                return "Size mismatch in 'coeff_re[" + std::to_string(i) + "]'.";

            if (coeff_im[i].n_rows != n_rx || coeff_im[i].n_cols != n_tx || coeff_im[i].n_slices != n_pth[i])
                return "Size mismatch in 'coeff_im[" + std::to_string(i) + "]'.";

            if ((delay[i].n_rows != 1 && delay[i].n_rows != n_rx) ||
                (delay[i].n_cols != 1 && delay[i].n_cols != n_tx) ||
                delay[i].n_slices != n_pth[i])
                return "Size mismatch in 'delay[" + std::to_string(i) + "]'.";
        }
    }
    else if (!coeff_im.empty())
        return "Real part of channel coefficients 'coeff_re' is missing or incomplete.";
    else if (!delay.empty())
        for (auto i = 0ULL; i < n_snap; ++i)
            if (delay[i].n_rows != 1 || delay[i].n_cols != 1 || delay[i].n_slices != n_pth[i])
                return "Size mismatch in 'delay[" + std::to_string(i) + "]'.";

    if (path_gain.size() != 0 && path_gain.size() != n_snap)
        return "'path_gain' must be empty or match the number of snapshots.";

    if (n_snap != 0ULL && path_gain.size() == n_snap)
        for (auto i = 0ULL; i < n_snap; ++i)
            if (path_gain[i].n_elem != n_pth[i])
                return "Size mismatch in 'path_gain[" + std::to_string(i) + "]'.";

    if (path_length.size() != 0 && path_length.size() != n_snap)
        return "'path_length' must be empty or match the number of snapshots.";

    if (n_snap != 0ULL && path_length.size() == n_snap)
        for (auto i = 0ULL; i < n_snap; ++i)
            if (path_length[i].n_elem != n_pth[i])
                return "Size mismatch in 'path_length[" + std::to_string(i) + "]'.";

    if (path_polarization.size() != 0 && path_polarization.size() != n_snap)
        return "'path_polarization' must be empty or match the number of snapshots.";

    if (n_snap != 0ULL && path_polarization.size() == n_snap)
        for (auto i = 0ULL; i < n_snap; ++i)
            if (path_polarization[i].n_rows != 8ULL || path_polarization[i].n_cols != n_pth[i])
                return "Size mismatch in 'path_polarization[" + std::to_string(i) + "]'.";

    if (path_angles.size() != 0 && path_angles.size() != n_snap)
        return "'path_angles' must be empty or match the number of snapshots.";

    if (n_snap != 0ULL && path_angles.size() == n_snap)
        for (auto i = 0ULL; i < n_snap; ++i)
            if (path_angles[i].n_rows != n_pth[i] || path_angles[i].n_cols != 4)
                return "Size mismatch in 'path_angles[" + std::to_string(i) + "]'.";

    if (path_fbs_pos.size() != 0 && path_fbs_pos.size() != n_snap)
        return "'path_fbs_pos' must be empty or match the number of snapshots.";

    if (n_snap != 0ULL && path_fbs_pos.size() == n_snap)
        for (auto i = 0ULL; i < n_snap; ++i)
            if (path_fbs_pos[i].n_rows != 3ULL || path_fbs_pos[i].n_cols != n_pth[i])
                return "Size mismatch in 'path_fbs_pos[" + std::to_string(i) + "]'.";

    if (path_lbs_pos.size() != 0 && path_lbs_pos.size() != n_snap)
        return "'path_lbs_pos' must be empty or match the number of snapshots.";

    if (n_snap != 0ULL && path_lbs_pos.size() == n_snap)
        for (auto i = 0ULL; i < n_snap; ++i)
            if (path_lbs_pos[i].n_rows != 3ULL || path_lbs_pos[i].n_cols != n_pth[i])
                return "Size mismatch in 'path_lbs_pos[" + std::to_string(i) + "]'.";

    if (no_interact.size() != 0 && no_interact.size() != n_snap)
        return "'no_interact' must be empty or match the number of snapshots.";
    else if (no_interact.size() == n_snap && interact_coord.size() != n_snap)
        return "'no_interact' is provided but 'interact_coord' is missing or has wrong number of snapshots.";
    else if (no_interact.size() == 0 && interact_coord.size() != 0)
        return "'interact_coord' is provided but 'no_interact' is missing.";
    else if (no_interact.size() == n_snap && interact_coord.size() == n_snap)
        for (auto i = 0ULL; i < n_snap; ++i)
        {
            if (no_interact[i].n_elem != n_pth[i])
                return "Size mismatch in 'no_interact[" + std::to_string(i) + "]'.";

            if (interact_coord[i].n_rows != 3ULL)
                return "'interact_coord[" + std::to_string(i) + "]' must have 3 rows.";

            unsigned cnt = 0;
            for (const unsigned *p = no_interact[i].begin(); p < no_interact[i].end(); ++p)
                cnt += *p;

            if (interact_coord[i].n_cols != cnt)
                return "Number of columns in 'interact_coord[" + std::to_string(i) + "]' must match the sum of 'no_interact'.";
        }

    if (par_names.size() != par_data.size())
        return "Number of elements in 'par_data' must match number of elements in 'par_name'.";

    for (auto i = 0ULL; i < par_names.size(); ++i)
        if (any_type_id(&par_data[i]) < 0)
            return "Unsupported datatype in unstructured data.";

    return "";
}

// Save data to HDF
template <typename dtype>
int quadriga_lib::channel<dtype>::hdf5_write(std::string fn, unsigned ix, unsigned iy, unsigned iz, unsigned iw, bool assume_valid) const
{
    // Commonly reused variables
    hid_t file_id, dspace_id, dset_id, group_id, type_id, snap_id;
    hsize_t dims[4];
    int return_code = 0;

    // Validate the channel data
    bool channel_is_empty = false;
    if (!assume_valid)
    {
        std::string error_message = is_valid();
        if (error_message.length() != 0)
            throw std::invalid_argument(error_message.c_str());

        // Determine if the channel object has no structured data fields
        channel_is_empty = empty();
    }

    // Are there unstructured fields?
    if (channel_is_empty && par_names.size() == 0)
        return 0; // Nothing to be done!

    // Determine if we need to convert the data to float
    bool data_is_float = !channel_is_empty && typeid(rx_pos).name() == typeid(arma::Mat<float>).name();
    float *data;

    // Create file
    if (!qHDF_file_exists(fn))
        quadriga_lib::hdf5_create(fn);

    // Open file for writing
    htri_t status = H5Fis_hdf5(fn.c_str());
    if (status <= 0)
        throw std::invalid_argument("Not an HDF5 file.");
    file_id = H5Fopen(fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID)
        throw std::invalid_argument("Error opening file.");

    // Get channel ID
    return_code = (qHDF_get_channel_ID(file_id, ix, iy, iz, iw, 0) != 0) ? 1 : 0;
    unsigned channel_index = qHDF_get_channel_ID(file_id, ix, iy, iz, iw, 1);
    if (channel_index == 0)
        qHDF_close_file(file_id), throw std::invalid_argument("Index out of bound.");

    // Open group for writing
    std::string group_name = "/channel_" + std::to_string(channel_index);
    group_id = H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);

    // Store the number of snapshots
    if (!channel_is_empty)
    {
        unsigned n_snap = (unsigned)this->n_snap();
        hid_t dspace_scalar = H5Screate(H5S_SCALAR);
        dset_id = H5Dcreate2(group_id, "NumSnap", H5T_NATIVE_UINT, dspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &n_snap);
        H5Dclose(dset_id);
        H5Sclose(dspace_scalar);
    }

    // Store channel name
    if (!channel_is_empty)
    {
        type_id = H5Tcopy(H5T_C_S1);
        H5Tset_size(type_id, name.length());
        dims[0] = 1;
        dspace_id = H5Screate_simple(1, dims, NULL);
        dset_id = H5Dcreate2(group_id, "Name", type_id, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, name.c_str());
        H5Dclose(dset_id);
        H5Sclose(dspace_id);
        H5Tclose(type_id);
    }

    // Center frequency in [Hz]
    if (!center_frequency.empty())
    {
        dims[0] = (hsize_t)center_frequency.n_elem;
        dspace_id = H5Screate_simple(1, dims, NULL);
        dset_id = H5Dcreate2(group_id, "CenterFrequency", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (data_is_float)
            H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, center_frequency.memptr());
        else
        {
            data = new float[center_frequency.n_elem];
            qHDF_cast_to_float(center_frequency.memptr(), data, center_frequency.n_elem);
            H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            delete[] data;
        }
        H5Dclose(dset_id);
        H5Sclose(dspace_id);
    }

    // Index of reference position
    if (initial_position != 0)
    {
        dims[0] = 1;
        dspace_id = H5Screate_simple(1, dims, NULL);
        dset_id = H5Dcreate2(group_id, "Initial_position", H5T_NATIVE_INT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &initial_position);
        H5Dclose(dset_id);
        H5Sclose(dspace_id);
    }

    // Transmitter positions, matrix of size [3, n_snap] or [3, 1]
    if (tx_pos.n_elem != 0ULL)
    {
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
    }

    // Receiver positions, matrix of size [3, n_snap]
    if (rx_pos.n_elem != 0ULL)
    {
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
    }

    // Transmitter orientation, matrix of size [3, n_snap] or [3, 1] or []
    if (tx_orientation.n_elem != 0ULL)
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
    if (rx_orientation.n_elem != 0ULL)
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
    unsigned long long n_snapshots = this->n_snap();
    for (auto i = 0ULL; i < n_snapshots; ++i)
    {
        if (n_snapshots == 1ULL)
            snap_id = group_id;
        else
        {
            std::string snap_name = "Snap_" + std::to_string(i);
            snap_id = H5Gcreate2(group_id, snap_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        }

        // Channel coefficients, vector (n_snap) of tensors of size [n_rx, n_tx, n_path]
        if (coeff_re.size() == n_snapshots && coeff_im.size() == n_snapshots && coeff_re[i].n_elem != 0 && coeff_im[i].n_elem != 0)
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
        if (delay.size() == n_snapshots && delay[i].n_elem != 0ULL)
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
        if (path_gain.size() == n_snapshots && path_gain[i].n_elem != 0ULL)
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
        if (path_length.size() == n_snapshots && path_length[i].n_elem != 0ULL)
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
        if (path_polarization.size() == n_snapshots && path_polarization[i].n_elem != 0ULL)
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
        if (path_angles.size() == n_snapshots && path_angles[i].n_elem != 0ULL)
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

        // First-bounce scatterer positions, matrices of size [3, n_path]
        if (path_fbs_pos.size() == n_snapshots && path_fbs_pos[i].n_elem != 0ULL)
        {
            dims[0] = path_fbs_pos[i].n_cols;
            dims[1] = path_fbs_pos[i].n_rows;
            dspace_id = H5Screate_simple(2, dims, NULL);
            dset_id = H5Dcreate2(snap_id, "path_fbs_pos", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (data_is_float)
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, path_fbs_pos[i].memptr());
            else
            {
                data = new float[path_fbs_pos[i].n_elem];
                qHDF_cast_to_float(path_fbs_pos[i].memptr(), data, path_fbs_pos[i].n_elem);
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                delete[] data;
            }
            H5Dclose(dset_id);
            H5Sclose(dspace_id);
        }

        // Last-bounce scatterer positions, matrices of size [3, n_path]
        if (path_lbs_pos.size() == n_snapshots && path_lbs_pos[i].n_elem != 0ULL)
        {
            dims[0] = path_lbs_pos[i].n_cols;
            dims[1] = path_lbs_pos[i].n_rows;
            dspace_id = H5Screate_simple(2, dims, NULL);
            dset_id = H5Dcreate2(snap_id, "path_lbs_pos", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (data_is_float)
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, path_lbs_pos[i].memptr());
            else
            {
                data = new float[path_lbs_pos[i].n_elem];
                qHDF_cast_to_float(path_lbs_pos[i].memptr(), data, path_lbs_pos[i].n_elem);
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                delete[] data;
            }
            H5Dclose(dset_id);
            H5Sclose(dspace_id);
        }

        // Number interaction points of a path with the environment, 0 = LOS, vectors of length [n_path]
        if (no_interact.size() == n_snapshots && no_interact[i].n_elem != 0ULL)
        {
            dims[0] = no_interact[i].n_elem;
            dspace_id = H5Screate_simple(1, dims, NULL);
            dset_id = H5Dcreate2(snap_id, "no_interact", H5T_NATIVE_UINT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, no_interact[i].memptr());
            H5Dclose(dset_id);
            H5Sclose(dspace_id);
        }

        // Interaction coordinates, NAN-padded, vector (n_snap) of tensors of size [3, n_coord, n_path]
        if (interact_coord.size() == n_snapshots && interact_coord[i].n_elem != 0ULL)
        {
            dims[0] = interact_coord[i].n_cols;
            dims[1] = interact_coord[i].n_rows;
            dspace_id = H5Screate_simple(2, dims, NULL);
            dset_id = H5Dcreate2(snap_id, "interact_coord", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (data_is_float)
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, interact_coord[i].memptr());
            else
            {
                data = new float[interact_coord[i].n_elem];
                qHDF_cast_to_float(interact_coord[i].memptr(), data, interact_coord[i].n_elem);
                H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                delete[] data;
            }
            H5Dclose(dset_id);
            H5Sclose(dspace_id);
        }

        if (n_snapshots != 1ULL)
            H5Gclose(snap_id);
    }

    // Write unstructured data
    for (std::size_t i = 0; i < par_names.size(); ++i)
    {
        std::string par_name = "par_" + par_names[i];
        qHDF_write_par(file_id, group_id, &par_name, &par_data[i]);
    }

    // Close the group and file
    qHDF_close_file(file_id);

    return return_code;
}

// Instantiate templates
template class quadriga_lib::channel<float>;
template class quadriga_lib::channel<double>;

// Returns type ID of a std::any field:
int quadriga_lib::any_type_id(const std::any *par_data, unsigned long long *dims, void **dataptr)
{
    if (par_data == nullptr || !par_data->has_value())
        return -2;

    if (dims != nullptr) // Set dims to scalar type
        dims[0] = 1ULL, dims[1] = 1ULL, dims[2] = 1ULL;

    if (par_data->type().name() == typeid(std::string).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<std::string>(par_data);
            if (dims != nullptr)
                *dims = (unsigned long long)data->length();
            if (dataptr != nullptr)
                *dataptr = (void *)data->c_str();
        }
        return 9;
    }

    // Scalar types
    if (par_data->type().name() == typeid(float).name())
    {
        if (dataptr != nullptr)
            *dataptr = (void *)std::any_cast<float>(par_data);
        return 10;
    }

    if (par_data->type().name() == typeid(double).name())
    {
        if (dataptr != nullptr)
            *dataptr = (void *)std::any_cast<double>(par_data);
        return 11;
    }

    if (par_data->type().name() == typeid(unsigned long long int).name())
    {
        if (dataptr != nullptr)
            *dataptr = (void *)std::any_cast<unsigned long long int>(par_data);
        return 12;
    }

    if (par_data->type().name() == typeid(long long int).name())
    {
        if (dataptr != nullptr)
            *dataptr = (void *)std::any_cast<long long int>(par_data);
        return 13;
    }

    if (par_data->type().name() == typeid(unsigned int).name())
    {
        if (dataptr != nullptr)
            *dataptr = (void *)std::any_cast<unsigned int>(par_data);
        return 14;
    }

    if (par_data->type().name() == typeid(int).name())
    {
        if (dataptr != nullptr)
            *dataptr = (void *)std::any_cast<int>(par_data);
        return 15;
    }

    // Matrix types
    if (par_data->type().name() == typeid(arma::Mat<float>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Mat<float>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 20;
    }

    if (par_data->type().name() == typeid(arma::Mat<double>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Mat<double>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 21;
    }

    if (par_data->type().name() == typeid(arma::Mat<unsigned long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Mat<unsigned long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 22;
    }

    if (par_data->type().name() == typeid(arma::Mat<long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Mat<long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 23;
    }

    if (par_data->type().name() == typeid(arma::Mat<unsigned int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Mat<unsigned int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 24;
    }

    if (par_data->type().name() == typeid(arma::Mat<int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Mat<int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 25;
    }

    // Cube types
    if (par_data->type().name() == typeid(arma::Cube<float>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Cube<float>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols, dims[2] = data->n_slices;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 30;
    }

    if (par_data->type().name() == typeid(arma::Cube<double>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Cube<double>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols, dims[2] = data->n_slices;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 31;
    }

    if (par_data->type().name() == typeid(arma::Cube<unsigned long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Cube<unsigned long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols, dims[2] = data->n_slices;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 32;
    }

    if (par_data->type().name() == typeid(arma::Cube<long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Cube<long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols, dims[2] = data->n_slices;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 33;
    }

    if (par_data->type().name() == typeid(arma::Cube<unsigned int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Cube<unsigned int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols, dims[2] = data->n_slices;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 34;
    }

    if (par_data->type().name() == typeid(arma::Cube<int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Cube<int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_rows, dims[1] = data->n_cols, dims[2] = data->n_slices;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 35;
    }

    // Column vectors
    if (par_data->type().name() == typeid(arma::Col<float>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Col<float>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 40;
    }

    if (par_data->type().name() == typeid(arma::Col<double>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Col<double>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 41;
    }

    if (par_data->type().name() == typeid(arma::Col<unsigned long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Col<unsigned long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 42;
    }

    if (par_data->type().name() == typeid(arma::Col<long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Col<long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 43;
    }

    if (par_data->type().name() == typeid(arma::Col<unsigned int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Col<unsigned int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 44;
    }

    if (par_data->type().name() == typeid(arma::Col<int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Col<int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 45;
    }

    // Row vectors
    if (par_data->type().name() == typeid(arma::Row<float>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Row<float>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 50;
    }

    if (par_data->type().name() == typeid(arma::Row<double>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Row<double>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 51;
    }

    if (par_data->type().name() == typeid(arma::Row<unsigned long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Row<unsigned long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 52;
    }

    if (par_data->type().name() == typeid(arma::Row<long long int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Row<long long int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 53;
    }

    if (par_data->type().name() == typeid(arma::Row<unsigned int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Row<unsigned int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 54;
    }

    if (par_data->type().name() == typeid(arma::Row<int>).name())
    {
        if (dims != nullptr || dataptr != nullptr)
        {
            auto *data = std::any_cast<arma::Row<int>>(par_data);
            if (dims != nullptr)
                dims[0] = data->n_elem;
            if (dataptr != nullptr)
                *dataptr = (void *)data->memptr();
        }
        return 55;
    }
    return -1;
}

// Create a new HDF file and set the index to to given storage layout
void quadriga_lib::hdf5_create(std::string fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    hid_t file_id, dspace_id, dset_id;
    hsize_t dims[4];

    if (nx == 0 || ny == 0 || nz == 0 || nw == 0)
        throw std::invalid_argument("Storage layout dimension sizes cannot be 0.");

    // Initialize HDF5 library
    H5open();

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
    H5Dwrite(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, ChannelDims);
    H5Dclose(dset_id);
    H5Sclose(dspace_id);

    // Create the index
    unsigned n_order = nx * ny * nz * nw;
    unsigned *p_order = new unsigned[n_order]();
    dims[0] = n_order;
    dspace_id = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate2(file_id, "Order", H5T_NATIVE_UINT32, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, p_order);
    H5Dclose(dset_id);
    H5Sclose(dspace_id);
    delete[] p_order;

    // Write a version ID
    unsigned version = 2;
    hid_t dspace_scalar = H5Screate(H5S_SCALAR);
    dset_id = H5Dcreate2(file_id, "Version", H5T_NATIVE_UINT, dspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &version);
    H5Dclose(dset_id);
    H5Sclose(dspace_scalar);

    // Close the file
    qHDF_close_file(file_id);

    // Close HDF5 library
    H5close();
    return;
}

// Read storage layout from HDF file
arma::Col<unsigned> quadriga_lib::hdf5_read_layout(std::string fn, arma::Col<unsigned> *channelID)
{
    if (!qHDF_file_exists(fn))
        return arma::Col<unsigned>(4);

    // Initialize HDF5 library
    H5open();

    // Open file for reading
    htri_t status = H5Fis_hdf5(fn.c_str());
    if (status <= 0)
        throw std::invalid_argument("Not an HDF5 file.");
    hid_t file_id = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID)
        throw std::invalid_argument("Error opening file.");

    arma::Col<unsigned> ChannelDims(4, arma::fill::none);

    // Read channel dims from file
    hsize_t dims[4];
    hid_t dset_id = H5Dopen(file_id, "ChannelDims", H5P_DEFAULT);
    if (dset_id == H5I_INVALID_HID)
        qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    hid_t dspace_id = H5Dget_space(dset_id);
    int ndims = H5Sget_simple_extent_ndims(dspace_id);
    if (ndims != 1)
        qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    H5Sget_simple_extent_dims(dspace_id, dims, NULL);
    if (dims[0] != 4)
        qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    H5Dread(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, ChannelDims.memptr());
    H5Dclose(dset_id);
    H5Sclose(dspace_id);

    // Load the storage index from file
    if (channelID != nullptr)
    {
        unsigned *p_ChannelDims = ChannelDims.memptr();
        unsigned n_order = p_ChannelDims[0] * p_ChannelDims[1] * p_ChannelDims[2] * p_ChannelDims[3];
        channelID->set_size((arma::uword)n_order);

        dset_id = H5Dopen(file_id, "Order", H5P_DEFAULT);
        if (dset_id == H5I_INVALID_HID)
            qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
        dspace_id = H5Dget_space(dset_id);
        ndims = H5Sget_simple_extent_ndims(dspace_id);
        if (ndims != 1)
            qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
        H5Sget_simple_extent_dims(dspace_id, dims, NULL);
        if (dims[0] != n_order)
            qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
        H5Dread(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, channelID->memptr());
        H5Dclose(dset_id);
        H5Sclose(dspace_id);
    }

    // Close the file
    qHDF_close_file(file_id);

    // Close HDF5 library
    H5close();

    return ChannelDims;
}

// Read channel object from HDF5 file
template <typename dtype>
quadriga_lib::channel<dtype> quadriga_lib::hdf5_read_channel(std::string fn, unsigned ix, unsigned iy, unsigned iz, unsigned iw)
{
    if (!qHDF_file_exists(fn))
        throw std::invalid_argument("File does not exist.");

    // Create empty channel
    quadriga_lib::channel<dtype> c;
    hid_t file_id = -1;

    // Open file for reading
    htri_t status = H5Fis_hdf5(fn.c_str());
    if (status <= 0)
        throw std::invalid_argument("Not an HDF5 file.");
    file_id = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID)
        throw std::invalid_argument("Error opening file.");

    // Get channel ID
    unsigned channel_index = qHDF_get_channel_ID(file_id, ix, iy, iz, iw, 0);
    if (channel_index == 0)
    {
        qHDF_close_file(file_id);
        return quadriga_lib::channel<dtype>();
    }

    // Open group
    std::string group_name = "/channel_" + std::to_string(channel_index);
    hid_t group_id = H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);

    // Check if we need to convert data to double precision
    bool float2double = typeid(dtype).name() == typeid(double).name();

    std::any val;

    // Read unstructured data
    std::string prefix = "par_";
    std::vector<std::string> available_par;
    qHDF_read_par_names(group_id, &available_par, &prefix);

    for (std::string &name : available_par)
    {
        val = qHDF_read_data(group_id, prefix + name);
        if (quadriga_lib::any_type_id(&val) > 0)
            c.par_names.push_back(name), c.par_data.push_back(val);
    }

    // Read channel name
    val = qHDF_read_data(group_id, "Name");
    if (val.has_value())
        c.name = std::any_cast<std::string>(val);

    // Read the number of snapshots
    unsigned long long n_snapshots = 0ULL;
    val = qHDF_read_data(group_id, "NumSnap");
    if (val.has_value())
        n_snapshots = (unsigned long long)std::any_cast<unsigned int>(val);

    // If there are no snapshots, we don't nee to continue
    if (n_snapshots == 0ULL)
    {
        qHDF_close_file(file_id);
        return c;
    }

    // Center frequency in [Hz]
    val = qHDF_read_data(group_id, "CenterFrequency", float2double);
    if (val.has_value() && quadriga_lib::any_type_id(&val) < 20) // scalar vale
        c.center_frequency = (dtype)std::any_cast<dtype>(val);
    else if (val.has_value())
        c.center_frequency = std::any_cast<arma::Col<dtype>>(val);

    // Index of reference position
    val = qHDF_read_data(group_id, "Initial_position");
    if (val.has_value())
        c.initial_position = std::any_cast<int>(val);

    // Transmitter positions, matrix of size [3, n_snap] or [3, 1]
    val = qHDF_read_data(group_id, "tx_position", float2double);
    if (val.has_value())
        c.tx_pos = std::any_cast<arma::Mat<dtype>>(val);

    // Receiver positions, matrix of size [3, n_snap]
    val = qHDF_read_data(group_id, "rx_position", float2double);
    if (val.has_value())
        c.rx_pos = std::any_cast<arma::Mat<dtype>>(val);

    // Transmitter orientation, matrix of size [3, n_snap] or [3, 1] or []
    val = qHDF_read_data(group_id, "tx_orientation", float2double);
    if (val.has_value())
        c.tx_orientation = std::any_cast<arma::Mat<dtype>>(val);

    // Receiver orientation, matrix of size [3, n_snap] or [3, 1] or []
    val = qHDF_read_data(group_id, "rx_orientation", float2double);
    if (val.has_value())
        c.rx_orientation = std::any_cast<arma::Mat<dtype>>(val);

    for (auto i = 0ULL; i < n_snapshots; ++i)
    {
        // Open group containing snapshot data
        hid_t snap_id = group_id;
        if (n_snapshots != 1ULL)
        {
            std::string snap_name = "Snap_" + std::to_string(i);
            snap_id = H5Gopen2(group_id, snap_name.c_str(), H5P_DEFAULT);
        }

        // Channel coefficients, vector (n_snap) of tensors of size [n_rx, n_tx, n_path]
        val = qHDF_read_data(snap_id, "coeff_re", float2double);
        if (val.has_value())
        {
            c.coeff_re.push_back(std::any_cast<arma::Cube<dtype>>(val));
            val = qHDF_read_data(snap_id, "coeff_im", float2double);
            c.coeff_im.push_back(std::any_cast<arma::Cube<dtype>>(val));
        }

        // Path delays in seconds, vector (n_snap) of tensors of size [n_rx, n_tx, n_path]
        val = qHDF_read_data(snap_id, "delay", float2double);
        if (val.has_value())
            c.delay.push_back(std::any_cast<arma::Cube<dtype>>(val));

        // Path gain before antenna patterns, vector (n_snap) of vectors of length [n_path]
        val = qHDF_read_data(snap_id, "path_gain", float2double);
        if (val.has_value())
            c.path_gain.push_back(std::any_cast<arma::Col<dtype>>(val));

        // Absolute path length from TX to RX phase center, vector (n_snap) of vectors of length [n_path]
        val = qHDF_read_data(snap_id, "path_length", float2double);
        if (val.has_value())
            c.path_length.push_back(std::any_cast<arma::Col<dtype>>(val));

        // Polarization transfer function, vector (n_snap) of matrices of size [8, n_path], interleaved complex
        val = qHDF_read_data(snap_id, "path_polarization", float2double);
        if (val.has_value())
            c.path_polarization.push_back(std::any_cast<arma::Mat<dtype>>(val));

        // Departure and arrival angles, vector (n_snap) of matrices of size [n_path, 4], {AOD, EOD, AOA, EOA}
        val = qHDF_read_data(snap_id, "path_angles", float2double);
        if (val.has_value())
            c.path_angles.push_back(std::any_cast<arma::Mat<dtype>>(val));

        // First-bounce scatterer positions, matrices of size [3, n_path]
        val = qHDF_read_data(snap_id, "path_fbs_pos", float2double);
        if (val.has_value())
            c.path_fbs_pos.push_back(std::any_cast<arma::Mat<dtype>>(val));

        // Last-bounce scatterer positions, matrices of size [3, n_path]
        val = qHDF_read_data(snap_id, "path_lbs_pos", float2double);
        if (val.has_value())
            c.path_lbs_pos.push_back(std::any_cast<arma::Mat<dtype>>(val));

        // Number interaction points of a path with the environment, 0 = LOS
        val = qHDF_read_data(snap_id, "no_interact");
        if (val.has_value())
            c.no_interact.push_back(std::any_cast<arma::Col<unsigned>>(val));

        // Interaction coordinates of paths with the environment, matrices of size [3, sum(no_interact)]
        val = qHDF_read_data(snap_id, "interact_coord", float2double);
        if (val.has_value())
            c.interact_coord.push_back(std::any_cast<arma::Mat<dtype>>(val));

        if (n_snapshots != 1ULL)
            H5Gclose(snap_id);
    }

    qHDF_close_file(file_id);
    return c;
}
template quadriga_lib::channel<float> quadriga_lib::hdf5_read_channel(std::string fn, unsigned ix, unsigned iy, unsigned iz, unsigned iw);
template quadriga_lib::channel<double> quadriga_lib::hdf5_read_channel(std::string fn, unsigned ix, unsigned iy, unsigned iz, unsigned iw);

// Reshape storage layout
void quadriga_lib::hdf5_reshape_layout(std::string fn, unsigned nx, unsigned ny, unsigned nz, unsigned nw)
{
    if (!qHDF_file_exists(fn))
        throw std::invalid_argument("File does not exist.");

    // Open file for writing
    htri_t status = H5Fis_hdf5(fn.c_str());
    if (status <= 0)
        throw std::invalid_argument("Not an HDF5 file.");
    hid_t file_id = H5Fopen(fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID)
        throw std::invalid_argument("Error opening file.");

    // Read channel dims from file
    unsigned ChannelDims[4];
    hsize_t dims[4];
    hid_t dset_id = H5Dopen(file_id, "ChannelDims", H5P_DEFAULT);
    if (dset_id == H5I_INVALID_HID)
        qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    hid_t dspace_id = H5Dget_space(dset_id);
    int ndims = H5Sget_simple_extent_ndims(dspace_id);
    if (ndims != 1)
        qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    H5Sget_simple_extent_dims(dspace_id, dims, NULL);
    if (dims[0] != 4)
        qHDF_close_file(file_id), throw std::invalid_argument("Storage index in HDF file is corrupted.");
    H5Dread(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, ChannelDims);

    unsigned n_data_old = ChannelDims[0] * ChannelDims[1] * ChannelDims[2] * ChannelDims[3];
    unsigned n_data_new = nx * ny * nz * nw;

    if (n_data_new != n_data_old)
        qHDF_close_file(file_id), throw std::invalid_argument("Mismatch in number of elements in storage index.");

    ChannelDims[0] = nx;
    ChannelDims[1] = ny;
    ChannelDims[2] = nz;
    ChannelDims[3] = nw;
    H5Dwrite(dset_id, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, ChannelDims);

    H5Sclose(dspace_id);
    H5Dclose(dset_id);

    qHDF_close_file(file_id);
}

// Read unstructured data from HDF5 file
std::any quadriga_lib::hdf5_read_dset(std::string fn, std::string par_name,
                                      unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                                      std::string prefix)
{
    if (!qHDF_file_exists(fn))
        throw std::invalid_argument("File does not exist.");

    // Open file for reading
    htri_t status = H5Fis_hdf5(fn.c_str());
    if (status <= 0)
        throw std::invalid_argument("Not an HDF5 file.");
    hid_t file_id = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID)
        throw std::invalid_argument("Error opening file.");

    // Get channel ID
    unsigned channel_index = qHDF_get_channel_ID(file_id, ix, iy, iz, iw, 0);
    if (channel_index == 0)
    {
        qHDF_close_file(file_id);
        return std::any();
    }

    // Open group
    std::string group_name = "/channel_" + std::to_string(channel_index);
    hid_t group_id = H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);

    // Read data
    std::string name = prefix + par_name;
    std::any data = qHDF_read_data(group_id, name);

    // Close group and file
    qHDF_close_file(file_id);
    return data;
}

// Read names of the unstructured data fields from the HDF file
unsigned long long quadriga_lib::hdf5_read_dset_names(std::string fn, std::vector<std::string> *par_names,
                                                      unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                                                      std::string prefix)
{
    if (!qHDF_file_exists(fn))
        throw std::invalid_argument("File does not exist.");

    // Open file for reading
    htri_t status = H5Fis_hdf5(fn.c_str());
    if (status <= 0)
        throw std::invalid_argument("Not an HDF5 file.");
    hid_t file_id = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID)
        throw std::invalid_argument("Error opening file.");

    // Get channel ID
    unsigned channel_index = qHDF_get_channel_ID(file_id, ix, iy, iz, iw, 0);
    if (channel_index == 0)
    {
        qHDF_close_file(file_id);
        return 0ULL;
    }

    // Open group
    std::string group_name = "/channel_" + std::to_string(channel_index);
    hid_t group_id = H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);

    // Read names
    long long no_par_names = qHDF_read_par_names(group_id, par_names, &prefix);
    qHDF_close_file(file_id);

    if (no_par_names < 0LL)
        throw std::invalid_argument("Failed to get group info.");

    return (unsigned long long)no_par_names;
}

// Writes unstructured data to a hdf5 file
void quadriga_lib::hdf5_write_dset(std::string fn, std::string par_name, const std::any *par_data,
                                   unsigned ix, unsigned iy, unsigned iz, unsigned iw,
                                   std::string prefix)
{
    // Create file
    if (!qHDF_file_exists(fn))
        quadriga_lib::hdf5_create(fn);

    // Open file for writing
    htri_t status = H5Fis_hdf5(fn.c_str());
    if (status <= 0)
        throw std::invalid_argument("Not an HDF5 file.");
    hid_t file_id = H5Fopen(fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID)
        throw std::invalid_argument("Error opening file.");

    // Get channel ID
    unsigned channel_index = qHDF_get_channel_ID(file_id, ix, iy, iz, iw, 2);
    if (channel_index == 0)
        qHDF_close_file(file_id), throw std::invalid_argument("Index out of bound.");

    // Open group for writing
    std::string group_name = "/channel_" + std::to_string(channel_index);
    hid_t group_id = H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);

    // Write unstructured data to file
    std::string name = prefix + par_name;
    qHDF_write_par(file_id, group_id, &name, par_data);

    // Close the group and file
    qHDF_close_file(file_id);
}
