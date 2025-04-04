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

// A collection of small MEX helper functions to reduce copy and pasting code
// Include this file in th mex-cpp files to use the functions

#ifndef mex_helper_H
#define mex_helper_H

#include "mex.h"
#include <string>
#include <armadillo>
#include <utility> // for std::move
#include <cstring> // For std::memcopy
#include <any>
#include <string>

// Read a scalar input from MATLAB and convert it to a desired c++ output type
// - Casts to <dtype>
// - Returns NaN for empty input (0 in case of integer types)
template <typename dtype>
inline dtype qd_mex_get_scalar(const mxArray *input, std::string var_name = "", dtype default_value = dtype(NAN))
{
    // Set default value in case of empty input
    if (mxGetNumberOfElements(input) == 0)
        return default_value;

    if (mxIsComplex(input))
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Complex datatypes are not supported.");

    // Assign the MATLAB data to the correct pointer
    if (mxIsDouble(input))
    {
        double *tmp = (double *)mxGetData(input);
        return dtype(*tmp);
    }
    else if (mxIsSingle(input))
    {
        float *tmp = (float *)mxGetData(input);
        return dtype(*tmp);
    }
    else if (mxIsClass(input, "uint32"))
    {
        unsigned *tmp = (unsigned *)mxGetData(input);
        return dtype(*tmp);
    }
    else if (mxIsClass(input, "int32"))
    {
        int *tmp = (int *)mxGetData(input);
        return dtype(*tmp);
    }
    else if (mxIsClass(input, "uint64"))
    {
        unsigned long long *tmp = (unsigned long long *)mxGetData(input);
        return dtype(*tmp);
    }
    else if (mxIsClass(input, "int64"))
    {
        long long *tmp = (long long *)mxGetData(input);
        return dtype(*tmp);
    }
    else if (mxIsClass(input, "logical"))
    {
        bool *tmp = (bool *)mxGetData(input);
        return dtype(*tmp);
    }
    else if (var_name.empty())
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported data type.");
    else
    {
        std::string error_message = "Input '" + var_name + "' has an unsupported data type.";
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", error_message.c_str());
    }

    return default_value;
}

// Read a string input from MATLAB and convert it to a std::string
// - Returns default_value (e.g. "") for empty input
inline std::string qd_mex_get_string(const mxArray *input, std::string default_value = "")
{
    if (mxGetNumberOfElements(input) == 0)
        return default_value;

    if (!mxIsClass(input, "char"))
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Input must be a string (character array)");

    auto chr = mxArrayToString(input);
    std::string data = std::string(chr);
    mxFree(chr);
    
    return data;
}

// Reinterpret MATLAB Array to Armadillo Column Vector
template <typename dtype>
inline arma::Col<dtype> qd_mex_reinterpret_Col(const mxArray *input, bool create_copy = false)
{
    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(input); // Number of dimensions - either 2, 3 or 4
    const mwSize *dims = mxGetDimensions(input);               // Read number of elements elements per dimension
    unsigned d1 = (unsigned)dims[0];                           // Number of elements on first dimension
    unsigned d2 = (unsigned)dims[1];                           // Number of elements on second dimension
    unsigned d3 = n_dim < 3 ? 1 : (unsigned)dims[2];           // Number of elements on third dimension
    unsigned d4 = n_dim < 4 ? 1 : (unsigned)dims[3];           // Number of elements on fourth dimension
    unsigned n_data = d1 * d2 * d3 * d4;

    if (n_data == 0)
        return arma::Col<dtype>();

    return arma::Col<dtype>((dtype *)mxGetData(input), n_data, create_copy, !create_copy);
}

// Reinterpret MATLAB Array to Armadillo Matrix
template <typename dtype>
inline arma::Mat<dtype> qd_mex_reinterpret_Mat(const mxArray *input, bool create_copy = false)
{
    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(input); // Number of dimensions - either 2, 3 or 4
    const mwSize *dims = mxGetDimensions(input);               // Read number of elements elements per dimension
    unsigned d1 = (unsigned)dims[0];                           // Number of elements on first dimension
    unsigned d2 = (unsigned)dims[1];                           // Number of elements on second dimension
    unsigned d3 = n_dim < 3 ? 1 : (unsigned)dims[2];           // Number of elements on third dimension
    unsigned d4 = n_dim < 4 ? 1 : (unsigned)dims[3];           // Number of elements on fourth dimension
    unsigned n_data = d1 * d2 * d3 * d4;

    if (n_data == 0)
        return arma::Mat<dtype>();

    return arma::Mat<dtype>((dtype *)mxGetData(input), d1, d2 * d3 * d4, create_copy, !create_copy);
}

// Reinterpret MATLAB Array to Armadillo Cube
template <typename dtype>
inline arma::Cube<dtype> qd_mex_reinterpret_Cube(const mxArray *input, bool create_copy = false)
{
    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(input); // Number of dimensions - either 2 or 3
    const mwSize *dims = mxGetDimensions(input);               // Read number of elements elements per dimension
    unsigned d1 = (unsigned)dims[0];                           // Number of elements on first dimension
    unsigned d2 = (unsigned)dims[1];                           // Number of elements on second dimension
    unsigned d3 = n_dim < 3 ? 1 : (unsigned)dims[2];           // Number of elements on third dimension
    unsigned d4 = n_dim < 4 ? 1 : (unsigned)dims[3];           // Number of elements on fourth dimension

    if (d1 * d2 * d3 * d4 == 0)
        return arma::Cube<dtype>();

    return arma::Cube<dtype>((dtype *)mxGetData(input), d1, d2, d3 * d4, create_copy, !create_copy);
}

// Converts input to <std::any>
// - Contained types: scalar native c++, arma::Col, arma::Row, arma::Mat, arma::Cube
// - 4D Types get converted to arma::Cube with 3rd and 4th dimensions merged
// - Empty inputs read to empty outputs
// - Input datatypes are maintained
// - Does not copy data
// - See also: quadriga_lib::any_type_id
inline std::any qd_mex_anycast(const mxArray *input, std::string var_name = "", bool create_copy = false)
{
    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(input); // Number of dimensions - either 2 or 3
    const mwSize *dims = mxGetDimensions(input);               // Read number of elements elements per dimension
    unsigned d1 = (unsigned)dims[0];                           // Number of elements on first dimension
    unsigned d2 = (unsigned)dims[1];                           // Number of elements on second dimension
    unsigned d3 = n_dim < 3 ? 1 : (unsigned)dims[2];           // Number of elements on third dimension
    unsigned d4 = n_dim < 4 ? 1 : (unsigned)dims[3];           // Number of elements on fourth dimension
    unsigned d34 = d3 * d4, n_data = d1 * d2 * d34;

    if (mxIsComplex(input))
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Complex datatypes are not supported.");

    if (n_data == 0) // Return empty
        return std::any();

    // Generate pointers for the 6 supported MATLAB data types
    if (mxIsDouble(input))
    {
        if (d1 == 1 && d2 == 1 && d34 == 1)
            return *(double *)mxGetData(input);
        else if (d1 != 1 && d2 == 1 && d34 == 1)
            return arma::Col<double>((double *)mxGetData(input), n_data, create_copy, !create_copy);
        else if (d1 == 1 && d2 != 1 && d34 == 1)
            return arma::Mat<double>((double *)mxGetData(input), 1, n_data, create_copy, !create_copy);
        else if (d34 != 1)
            return arma::Cube<double>((double *)mxGetData(input), d1, d2, d34, create_copy, !create_copy);
        else
            return arma::Mat<double>((double *)mxGetData(input), d1, d2 * d34, create_copy, !create_copy);
    }
    if (mxIsSingle(input))
    {
        if (d1 == 1 && d2 == 1 && d34 == 1)
            return *(float *)mxGetData(input);
        else if (d1 != 1 && d2 == 1 && d34 == 1)
            return arma::Col<float>((float *)mxGetData(input), n_data, create_copy, !create_copy);
        else if (d1 == 1 && d2 != 1 && d34 == 1)
            return arma::Mat<float>((float *)mxGetData(input), 1, n_data, create_copy, !create_copy);
        else if (d34 != 1)
            return arma::Cube<float>((float *)mxGetData(input), d1, d2, d34, create_copy, !create_copy);
        else
            return arma::Mat<float>((float *)mxGetData(input), d1, d2 * d34, create_copy, !create_copy);
    }
    if (mxIsClass(input, "uint32"))
    {
        if (d1 == 1 && d2 == 1 && d34 == 1)
            return *(unsigned *)mxGetData(input);
        else if (d1 != 1 && d2 == 1 && d34 == 1)
            return arma::Col<unsigned>((unsigned *)mxGetData(input), n_data, create_copy, !create_copy);
        else if (d1 == 1 && d2 != 1 && d34 == 1)
            return arma::Mat<unsigned>((unsigned *)mxGetData(input), 1, n_data, create_copy, !create_copy);
        else if (d34 != 1)
            return arma::Cube<unsigned>((unsigned *)mxGetData(input), d1, d2, d34, create_copy, !create_copy);
        else
            return arma::Mat<unsigned>((unsigned *)mxGetData(input), d1, d2 * d34, create_copy, !create_copy);
    }
    if (mxIsClass(input, "int32"))
    {
        if (d1 == 1 && d2 == 1 && d34 == 1)
            return *(int *)mxGetData(input);
        else if (d1 != 1 && d2 == 1 && d34 == 1)
            return arma::Col<int>((int *)mxGetData(input), n_data, create_copy, !create_copy);
        else if (d1 == 1 && d2 != 1 && d34 == 1)
            return arma::Mat<int>((int *)mxGetData(input), 1, n_data, create_copy, !create_copy);
        else if (d34 != 1)
            return arma::Cube<int>((int *)mxGetData(input), d1, d2, d34, create_copy, !create_copy);
        else
            return arma::Mat<int>((int *)mxGetData(input), d1, d2 * d34, create_copy, !create_copy);
    }
    if (mxIsClass(input, "uint64"))
    {
        if (d1 == 1 && d2 == 1 && d34 == 1)
            return *(unsigned long long *)mxGetData(input);
        else if (d1 != 1 && d2 == 1 && d34 == 1)
            return arma::Col<unsigned long long>((unsigned long long *)mxGetData(input), n_data, create_copy, !create_copy);
        else if (d1 == 1 && d2 != 1 && d34 == 1)
            return arma::Mat<unsigned long long>((unsigned long long *)mxGetData(input), 1, n_data, create_copy, !create_copy);
        else if (d34 != 1)
            return arma::Cube<unsigned long long>((unsigned long long *)mxGetData(input), d1, d2, d34, create_copy, !create_copy);
        else
            return arma::Mat<unsigned long long>((unsigned long long *)mxGetData(input), d1, d2 * d34, create_copy, !create_copy);
    }
    if (mxIsClass(input, "int64"))
    {
        if (d1 == 1 && d2 == 1 && d34 == 1)
            return *(long long *)mxGetData(input);
        else if (d1 != 1 && d2 == 1 && d34 == 1)
            return arma::Col<long long>((long long *)mxGetData(input), n_data, create_copy, !create_copy);
        else if (d1 == 1 && d2 != 1 && d34 == 1)
            return arma::Mat<long long>((long long *)mxGetData(input), 1, n_data, create_copy, !create_copy);
        else if (d34 != 1)
            return arma::Cube<long long>((long long *)mxGetData(input), d1, d2, d34, create_copy, !create_copy);
        else
            return arma::Mat<long long>((long long *)mxGetData(input), d1, d2 * d34, create_copy, !create_copy);
    }

    // Throw error if type is not supported
    if (var_name.empty())
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported data type.");
    else
    {
        std::string error_message = "Input '" + var_name + "' has an unsupported data type.";
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", error_message.c_str());
    }

    return std::any();
}

// Reads input and converts it to desired C++ type, creates a copy of the input
template <typename dtype>
inline arma::Col<dtype> qd_mex_typecast_Col(const mxArray *input, std::string var_name = "", unsigned n_elem = 0)
{
    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(input); // Number of dimensions - either 2, 3 or 4
    const mwSize *dims = mxGetDimensions(input);               // Read number of elements elements per dimension
    unsigned d1 = (unsigned)dims[0];                           // Number of elements on first dimension
    unsigned d2 = (unsigned)dims[1];                           // Number of elements on second dimension
    unsigned d3 = n_dim < 3 ? 1 : (unsigned)dims[2];           // Number of elements on third dimension
    unsigned d4 = n_dim < 4 ? 1 : (unsigned)dims[3];           // Number of elements on fourth dimension
    unsigned n_data = d1 * d2 * d3 * d4;

    if (mxIsComplex(input))
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Complex datatypes are not supported.");

    if (n_data == 0) // Return empty
        return arma::Col<dtype>();

    if (n_elem != 0 && n_data != n_elem)
    {
        if (var_name.empty())
            mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Incorrect number of elements.");
        else
        {
            std::string error_message = "Input '" + var_name + "' has incorrect number of elements.";
            mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", error_message.c_str());
        }
    }

    // Generate pointers for the 6 supported MATLAB data types
    double *ptr_d = nullptr;
    float *ptr_f = nullptr;
    unsigned *ptr_ui = nullptr;
    int *ptr_i = nullptr;
    unsigned long long *ptr_ull = nullptr;
    long long *ptr_ll = nullptr;

    // Assign the MATLAB data to the correct pointer
    unsigned T = 0; // Type ID of input type
    if (mxIsDouble(input))
        T = 1, ptr_d = (double *)mxGetData(input);
    else if (mxIsSingle(input))
        T = 2, ptr_f = (float *)mxGetData(input);
    else if (mxIsClass(input, "uint32"))
        T = 3, ptr_ui = (unsigned *)mxGetData(input);
    else if (mxIsClass(input, "int32"))
        T = 4, ptr_i = (int *)mxGetData(input);
    else if (mxIsClass(input, "uint64"))
        T = 5, ptr_ull = (unsigned long long *)mxGetData(input);
    else if (mxIsClass(input, "int64"))
        T = 6, ptr_ll = (long long *)mxGetData(input);
    else if (var_name.empty())
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported data type.");
    else
    {
        std::string error_message = "Input '" + var_name + "' has an unsupported data type.";
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", error_message.c_str());
    }

    // Convert data to armadillo output
    arma::Col<dtype> output = arma::Col<dtype>(n_data, arma::fill::none);
    dtype *ptr = output.memptr();
    if (T == 1)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_d[m];
    else if (T == 2)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_f[m];
    else if (T == 3)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_ui[m];
    else if (T == 4)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_i[m];
    else if (T == 5)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_ull[m];
    else if (T == 6)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_ll[m];
    return output;
}

// Reads input and converts it to desired c++ type, creates a copy of the input
template <typename dtype>
inline arma::Mat<dtype> qd_mex_typecast_Mat(const mxArray *input, std::string var_name = "")
{
    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(input); // Number of dimensions - either 2, 3 or 4
    const mwSize *dims = mxGetDimensions(input);               // Read number of elements elements per dimension
    unsigned d1 = (unsigned)dims[0];                           // Number of elements on first dimension
    unsigned d2 = (unsigned)dims[1];                           // Number of elements on second dimension
    unsigned d3 = n_dim < 3 ? 1 : (unsigned)dims[2];           // Number of elements on third dimension
    unsigned d4 = n_dim < 4 ? 1 : (unsigned)dims[3];           // Number of elements on fourth dimension
    unsigned n_data = d1 * d2 * d3 * d4;

    if (mxIsComplex(input))
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Complex datatypes are not supported.");

    if (n_data == 0) // Return empty
        return arma::Mat<dtype>();

    // Generate pointers for the 6 supported MATLAB data types
    double *ptr_d = nullptr;
    float *ptr_f = nullptr;
    unsigned *ptr_ui = nullptr;
    int *ptr_i = nullptr;
    unsigned long long *ptr_ull = nullptr;
    long long *ptr_ll = nullptr;

    // Assign the MATLAB data to the correct pointer
    unsigned T = 0; // Type ID of input type
    if (mxIsDouble(input))
        T = 1, ptr_d = (double *)mxGetData(input);
    else if (mxIsSingle(input))
        T = 2, ptr_f = (float *)mxGetData(input);
    else if (mxIsClass(input, "uint32"))
        T = 3, ptr_ui = (unsigned *)mxGetData(input);
    else if (mxIsClass(input, "int32"))
        T = 4, ptr_i = (int *)mxGetData(input);
    else if (mxIsClass(input, "uint64"))
        T = 5, ptr_ull = (unsigned long long *)mxGetData(input);
    else if (mxIsClass(input, "int64"))
        T = 6, ptr_ll = (long long *)mxGetData(input);
    else if (var_name.empty())
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported data type.");
    else
    {
        std::string error_message = "Input '" + var_name + "' has an unsupported data type.";
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", error_message.c_str());
    }

    // Convert data to armadillo output
    auto output = arma::Mat<dtype>(d1, d2 * d3 * d4, arma::fill::none);
    dtype *ptr = output.memptr();
    if (T == 1)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_d[m];
    else if (T == 2)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_f[m];
    else if (T == 3)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_ui[m];
    else if (T == 4)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_i[m];
    else if (T == 5)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_ull[m];
    else if (T == 6)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_ll[m];
    return output;
}

// Reads input and converts it to desired c++ type, creates a copy of the input
template <typename dtype>
inline arma::Cube<dtype> qd_mex_typecast_Cube(const mxArray *input, std::string var_name = "")
{
    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(input); // Number of dimensions - either 2, 3 or 4
    const mwSize *dims = mxGetDimensions(input);               // Read number of elements elements per dimension
    unsigned d1 = (unsigned)dims[0];                           // Number of elements on first dimension
    unsigned d2 = (unsigned)dims[1];                           // Number of elements on second dimension
    unsigned d3 = n_dim < 3 ? 1 : (unsigned)dims[2];           // Number of elements on third dimension
    unsigned d4 = n_dim < 4 ? 1 : (unsigned)dims[3];           // Number of elements on fourth dimension
    unsigned n_data = d1 * d2 * d3 * d4;

    if (mxIsComplex(input))
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Complex datatypes are not supported.");

    if (n_data == 0) // Return empty
        return arma::Cube<dtype>();

    // Generate pointers for the 6 supported MATLAB data types
    double *ptr_d = nullptr;
    float *ptr_f = nullptr;
    unsigned *ptr_ui = nullptr;
    int *ptr_i = nullptr;
    unsigned long long *ptr_ull = nullptr;
    long long *ptr_ll = nullptr;

    // Assign the MATLAB data to the correct pointer
    unsigned T = 0; // Type ID of input type
    if (mxIsDouble(input))
        T = 1, ptr_d = (double *)mxGetData(input);
    else if (mxIsSingle(input))
        T = 2, ptr_f = (float *)mxGetData(input);
    else if (mxIsClass(input, "uint32"))
        T = 3, ptr_ui = (unsigned *)mxGetData(input);
    else if (mxIsClass(input, "int32"))
        T = 4, ptr_i = (int *)mxGetData(input);
    else if (mxIsClass(input, "uint64"))
        T = 5, ptr_ull = (unsigned long long *)mxGetData(input);
    else if (mxIsClass(input, "int64"))
        T = 6, ptr_ll = (long long *)mxGetData(input);
    else if (var_name.empty())
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported data type.");
    else
    {
        std::string error_message = "Input '" + var_name + "' has an unsupported data type.";
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", error_message.c_str());
    }

    // Convert data to armadillo output
    auto output = arma::Cube<dtype>(d1, d2, d3 * d4, arma::fill::none);
    dtype *ptr = output.memptr();
    if (T == 1)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_d[m];
    else if (T == 2)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_f[m];
    else if (T == 3)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_ui[m];
    else if (T == 4)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_i[m];
    else if (T == 5)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_ull[m];
    else if (T == 6)
        for (unsigned m = 0; m < n_data; ++m)
            ptr[m] = (dtype)ptr_ll[m];
    return output;
}

// Creates an mxArray based on the armadillo input type, copies content
template <typename dtype>
inline mxArray *qd_mex_copy2matlab(const dtype *input) // Scalar
{
    // Get classID from dtype
    mxClassID classID;
    if (typeid(dtype).name() == typeid(float).name())
        classID = mxSINGLE_CLASS;
    else if (typeid(dtype).name() == typeid(double).name())
        classID = mxDOUBLE_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned).name())
        classID = mxUINT32_CLASS;
    else if (typeid(dtype).name() == typeid(int).name())
        classID = mxINT32_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned long long).name())
        classID = mxUINT64_CLASS;
    else if (typeid(dtype).name() == typeid(long long).name())
        classID = mxINT64_CLASS;
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported datatype.");

    mxArray *output = mxCreateNumericMatrix(1, 1, classID, mxREAL);
    std::memcpy((dtype *)mxGetData(output), input, sizeof(dtype));
    return output;
}

template <typename dtype>
inline mxArray *qd_mex_copy2matlab(const arma::Row<dtype> *input) // Row Vector
{
    mxClassID classID;
    if (typeid(dtype).name() == typeid(float).name())
        classID = mxSINGLE_CLASS;
    else if (typeid(dtype).name() == typeid(double).name())
        classID = mxDOUBLE_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned).name())
        classID = mxUINT32_CLASS;
    else if (typeid(dtype).name() == typeid(int).name())
        classID = mxINT32_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned long long).name())
        classID = mxUINT64_CLASS;
    else if (typeid(dtype).name() == typeid(long long).name())
        classID = mxINT64_CLASS;
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported datatype.");

    mxArray *output = mxCreateNumericMatrix(1, (mwSize)input->n_elem, classID, mxREAL);
    std::memcpy((dtype *)mxGetData(output), input->memptr(), sizeof(dtype) * input->n_elem);
    return output;
}

template <typename dtype>
inline mxArray *qd_mex_copy2matlab(const arma::Col<dtype> *input,          // Column Vector
                                   bool transpose = false,                 // Transpose output
                                   unsigned long long ns = 0,              // Number of elements in output
                                   const unsigned long long *is = nullptr) // List of elements to copy, 0-based
{
    mxClassID classID;
    if (typeid(dtype).name() == typeid(float).name())
        classID = mxSINGLE_CLASS;
    else if (typeid(dtype).name() == typeid(double).name())
        classID = mxDOUBLE_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned).name())
        classID = mxUINT32_CLASS;
    else if (typeid(dtype).name() == typeid(int).name())
        classID = mxINT32_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned long long).name())
        classID = mxUINT64_CLASS;
    else if (typeid(dtype).name() == typeid(long long).name())
        classID = mxINT64_CLASS;
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported datatype.");

    if (input->empty())
        return mxCreateNumericMatrix(0, 0, classID, mxREAL);

    ns = ns == 0 ? input->n_elem : ns;
    mxArray *output = transpose ? mxCreateNumericMatrix(1, (mwSize)ns, classID, mxREAL)
                                : mxCreateNumericMatrix((mwSize)ns, 1, classID, mxREAL);
    dtype *ptr_o = (dtype *)mxGetData(output);
    const dtype *ptr_i = input->memptr();

    if (is == nullptr) // Copy all
        std::memcpy(ptr_o, ptr_i, sizeof(dtype) * input->n_elem);
    else // Copy selected
        for (unsigned long long i = 0ULL; i < ns; ++i)
            ptr_o[i] = is[i] >= input->n_elem ? *ptr_i : ptr_i[is[i]];

    return output;
}

template <typename dtype>
inline mxArray *qd_mex_copy2matlab(const arma::Mat<dtype> *input,          // Matrix
                                   unsigned long long ns = 0,              // Number of columns in output
                                   const unsigned long long *is = nullptr) // List of columns to copy, 0-based
{
    mxClassID classID;
    if (typeid(dtype).name() == typeid(float).name())
        classID = mxSINGLE_CLASS;
    else if (typeid(dtype).name() == typeid(double).name())
        classID = mxDOUBLE_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned).name())
        classID = mxUINT32_CLASS;
    else if (typeid(dtype).name() == typeid(int).name())
        classID = mxINT32_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned long long).name())
        classID = mxUINT64_CLASS;
    else if (typeid(dtype).name() == typeid(long long).name())
        classID = mxINT64_CLASS;
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported datatype.");

    if (input->empty())
        return mxCreateNumericMatrix(0, 0, classID, mxREAL);

    unsigned long long m = input->n_rows; // Rows
    ns = ns == 0 ? input->n_cols : ns;    // Output columns
    mxArray *output = mxCreateNumericMatrix((mwSize)m, (mwSize)ns, classID, mxREAL);
    dtype *ptr = (dtype *)mxGetData(output);

    if (is == nullptr) // Copy all
        std::memcpy(ptr, input->memptr(), sizeof(dtype) * input->n_elem);
    else // Copy selected
        for (unsigned long long i = 0ULL; i < ns; ++i)
        {
            unsigned long long k = is[i] >= input->n_cols ? 0ULL : is[i];
            std::memcpy(&ptr[i * m], input->colptr(k), sizeof(dtype) * m);
        }

    return output;
}

template <typename dtype>
inline mxArray *qd_mex_copy2matlab(arma::Cube<dtype> *input,               // Cube
                                   unsigned long long ns = 0,              // Number of columns in output
                                   const unsigned long long *is = nullptr) // List of columns to copy, 0-based
{
    mxClassID classID;
    if (typeid(dtype).name() == typeid(float).name())
        classID = mxSINGLE_CLASS;
    else if (typeid(dtype).name() == typeid(double).name())
        classID = mxDOUBLE_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned).name())
        classID = mxUINT32_CLASS;
    else if (typeid(dtype).name() == typeid(int).name())
        classID = mxINT32_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned long long).name())
        classID = mxUINT64_CLASS;
    else if (typeid(dtype).name() == typeid(long long).name())
        classID = mxINT64_CLASS;
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported datatype.");

    if (input->empty())
        return mxCreateNumericMatrix(0, 0, classID, mxREAL);

    unsigned long long m = input->n_rows * input->n_cols; // Rows and columns
    ns = ns == 0 ? input->n_slices : ns;                  // Slices
    mwSize dims[3] = {(mwSize)input->n_rows, (mwSize)input->n_cols, (mwSize)ns};
    mxArray *output = mxCreateNumericArray(3, dims, classID, mxREAL);
    dtype *ptr = (dtype *)mxGetData(output);

    if (is == nullptr) // Copy all
        std::memcpy(ptr, input->memptr(), sizeof(dtype) * input->n_elem);
    else // Copy selected
        for (unsigned long long i = 0ULL; i < ns; ++i)
        {
            unsigned long long k = is[i] >= input->n_slices ? 0ULL : is[i];
            std::memcpy(&ptr[i * m], input->slice_memptr(k), sizeof(dtype) * m);
        }

    return output;
}

// Creates an mxArray based on a vector of armadillo input types
// - adds one additional dimension, e.g. arma::Cube --> MATLAB 4D
// - optional input: reading order of vector elements
// - copies data
// - zero-padding of missing data
// - returns empty Matrix object is vector is empty
template <typename dtype>
inline mxArray *qd_mex_vector2matlab(const std::vector<arma::Col<dtype>> *input, unsigned long long ns = 0, const unsigned long long *is = nullptr, dtype padding = (dtype)0)
{
    // Get classID from dtype
    mxClassID classID;
    if (typeid(dtype).name() == typeid(float).name())
        classID = mxSINGLE_CLASS;
    else if (typeid(dtype).name() == typeid(double).name())
        classID = mxDOUBLE_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned).name())
        classID = mxUINT32_CLASS;
    else if (typeid(dtype).name() == typeid(int).name())
        classID = mxINT32_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned long long).name())
        classID = mxUINT64_CLASS;
    else if (typeid(dtype).name() == typeid(long long).name())
        classID = mxINT64_CLASS;
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported datatype.");

    if (input->empty())
        return mxCreateNumericMatrix(0, 0, classID, mxREAL);

    bool use_padding = padding != (dtype)0;

    // Get maximum input data dimensions
    unsigned long long m = 0ULL;
    for (auto &v : *input)
        m = v.n_rows > m ? v.n_rows : m;
    ns = ns == 0ULL ? input->size() : ns;

    mxArray *output = mxCreateNumericMatrix((mwSize)m, (mwSize)ns, classID, mxREAL);
    dtype *ptr = (dtype *)mxGetData(output);

    // Get snapshot range
    unsigned long long *js;
    if (is == nullptr)
    {
        js = new unsigned long long[input->size()];
        for (unsigned long long i = 0ULL; i < input->size(); ++i)
            js[i] = i;
    }
    else
        js = const_cast<unsigned long long *>(is); // Dirty, but fast

    // Copy data
    for (unsigned long long i = 0ULL; i < ns; ++i)
    {
        unsigned long long k = js[i] >= input->size() ? 0ULL : js[i];
        unsigned long long r = input->at(k).n_rows;

        if (use_padding && r != m)
            for (dtype *p = &ptr[i * m]; p < &ptr[(i + 1) * m]; ++p)
                *p = padding;

        std::memcpy(&ptr[i * m], input->at(k).memptr(), sizeof(dtype) * r);
    }

    if (is == nullptr)
        delete[] js;

    return output;
}

template <typename dtype>
inline mxArray *qd_mex_vector2matlab(const std::vector<arma::Mat<dtype>> *input, unsigned long long ns = 0, const unsigned long long *is = nullptr, dtype padding = (dtype)0)
{
    // Get classID from dtype
    mxClassID classID;
    if (typeid(dtype).name() == typeid(float).name())
        classID = mxSINGLE_CLASS;
    else if (typeid(dtype).name() == typeid(double).name())
        classID = mxDOUBLE_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned).name())
        classID = mxUINT32_CLASS;
    else if (typeid(dtype).name() == typeid(int).name())
        classID = mxINT32_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned long long).name())
        classID = mxUINT64_CLASS;
    else if (typeid(dtype).name() == typeid(long long).name())
        classID = mxINT64_CLASS;
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported datatype.");

    if (input->empty())
        return mxCreateNumericMatrix(0, 0, classID, mxREAL);

    bool use_padding = padding != (dtype)0;

    // Get maximum input data dimensions
    unsigned long long n_rows = 0ULL, n_cols = 0ULL;
    for (auto &v : *input)
        n_rows = v.n_rows > n_rows ? v.n_rows : n_rows,
        n_cols = v.n_cols > n_cols ? v.n_cols : n_cols;

    unsigned long long m = n_rows * n_cols;
    ns = ns == 0ULL ? input->size() : ns;

    mwSize dims[3] = {(mwSize)n_rows, (mwSize)n_cols, (mwSize)ns};
    mxArray *output = mxCreateNumericArray(3, dims, classID, mxREAL);
    dtype *ptr = (dtype *)mxGetData(output);

    // Get snapshot range
    unsigned long long *js;
    if (is == nullptr)
    {
        js = new unsigned long long[input->size()];
        for (unsigned long long i = 0ULL; i < input->size(); ++i)
            js[i] = i;
    }
    else
        js = const_cast<unsigned long long *>(is); // Dirty, but fast

    // Copy data
    for (unsigned long long i = 0ULL; i < ns; ++i)
    {
        unsigned long long k = js[i] >= input->size() ? 0ULL : js[i];
        unsigned long long r = input->at(k).n_rows, c = input->at(k).n_cols;

        if (use_padding && r * c != m)
            for (dtype *p = &ptr[i * m]; p < &ptr[(i + 1) * m]; ++p)
                *p = padding;

        if (r == n_rows)
            std::memcpy(&ptr[i * m], input->at(k).memptr(), sizeof(dtype) * r * c);
        else // Copy column by column
            for (unsigned long long ic = 0ULL; ic < c; ++ic)
                std::memcpy(&ptr[i * m + ic * n_rows],
                            input->at(k).colptr(ic), sizeof(dtype) * r);
    }

    if (is == nullptr)
        delete[] js;

    return output;
}

template <typename dtype>
inline mxArray *qd_mex_vector2matlab(const std::vector<arma::Cube<dtype>> *input, unsigned long long ns = 0, const unsigned long long *is = nullptr, dtype padding = (dtype)0)
{
    // Get classID from dtype
    mxClassID classID;
    if (typeid(dtype).name() == typeid(float).name())
        classID = mxSINGLE_CLASS;
    else if (typeid(dtype).name() == typeid(double).name())
        classID = mxDOUBLE_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned).name())
        classID = mxUINT32_CLASS;
    else if (typeid(dtype).name() == typeid(unsigned long long).name())
        classID = mxUINT64_CLASS;
    else if (typeid(dtype).name() == typeid(long long).name())
        classID = mxINT64_CLASS;
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported datatype.");

    if (input->empty())
        return mxCreateNumericMatrix(0, 0, classID, mxREAL);

    bool use_padding = padding != (dtype)0;

    // Get maximum input data dimensions
    unsigned long long n_rows = 0ULL, n_cols = 0ULL, n_slices = 0ULL;
    for (auto &v : *input)
        n_rows = v.n_rows > n_rows ? v.n_rows : n_rows,
        n_cols = v.n_cols > n_cols ? v.n_cols : n_cols,
        n_slices = v.n_slices > n_slices ? v.n_slices : n_slices;

    unsigned long long m = n_rows * n_cols * n_slices;
    ns = ns == 0ULL ? input->size() : ns;

    mwSize dims[4] = {(mwSize)n_rows, (mwSize)n_cols, (mwSize)n_slices, (mwSize)ns};
    mxArray *output = mxCreateNumericArray(4, dims, classID, mxREAL);
    dtype *ptr = (dtype *)mxGetData(output);

    // Get snapshot range
    unsigned long long *js;
    if (is == nullptr)
    {
        js = new unsigned long long[input->size()];
        for (unsigned long long i = 0ULL; i < input->size(); ++i)
            js[i] = i;
    }
    else
        js = const_cast<unsigned long long *>(is); // Dirty, but fast

    // Copy data
    for (unsigned long long i = 0ULL; i < ns; ++i)
    {
        unsigned long long k = js[i] >= input->size() ? 0ULL : js[i];
        unsigned long long r = input->at(k).n_rows, c = input->at(k).n_cols, s = input->at(k).n_slices;

        if (use_padding && r * c * s != m)
            for (dtype *p = &ptr[i * m]; p < &ptr[(i + 1) * m]; ++p)
                *p = padding;

        if (r == n_rows && c == n_cols)
            std::memcpy(&ptr[i * m], input->at(k).memptr(), sizeof(dtype) * r * c * s);
        else // Copy column by column
            for (unsigned long long is = 0ULL; is < s; ++is)
                for (unsigned long long ic = 0ULL; ic < c; ++ic)
                    std::memcpy(&ptr[i * m + is * n_rows * n_cols + ic * n_rows],
                                input->at(k).slice_colptr(is, ic), sizeof(dtype) * r);
    }

    if (is == nullptr)
        delete[] js;

    return output;
}

// Creates a std::vector of armadillo types from mxArray
// - e.g. MATLAB --> std::vector<arma::Col<dtype>>
// - vec_dim = Dimension used for std::vector, 0-based
// - Data on other dimensions are vectorized and casted to <dtype>
template <typename dtype>
std::vector<arma::Col<dtype>> qd_mex_matlab2vector_Col(const mxArray *input, unsigned vec_dim)
{
    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(input); // Number of dimensions - either 2, 3 or 4
    const mwSize *dims = mxGetDimensions(input);               // Read number of elements elements per dimension
    unsigned d1 = (unsigned)dims[0];                           // Number of elements on first dimension
    unsigned d2 = (unsigned)dims[1];                           // Number of elements on second dimension
    unsigned d3 = n_dim < 3 ? 1 : (unsigned)dims[2];           // Number of elements on third dimension
    unsigned d4 = n_dim < 4 ? 1 : (unsigned)dims[3];           // Number of elements on fourth dimension
    unsigned n_data = d1 * d2 * d3 * d4;

    if (mxIsComplex(input))
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Complex datatypes are not supported.");

    if (n_data == 0) // Return empty std::vector
        return std::vector<arma::Col<dtype>>();

    // Generate pointers for the 6 supported MATLAB data types
    double *ptr_d = nullptr;
    float *ptr_f = nullptr;
    unsigned *ptr_ui = nullptr;
    int *ptr_i = nullptr;
    unsigned long long *ptr_ull = nullptr;
    long long *ptr_ll = nullptr;

    // Assign the MATLAB data to the correct pointer
    unsigned T = 0; // Type ID of input type
    if (mxIsDouble(input))
        T = 1, ptr_d = (double *)mxGetData(input);
    else if (mxIsSingle(input))
        T = 2, ptr_f = (float *)mxGetData(input);
    else if (mxIsClass(input, "uint32"))
        T = 3, ptr_ui = (unsigned *)mxGetData(input);
    else if (mxIsClass(input, "int32"))
        T = 4, ptr_i = (int *)mxGetData(input);
    else if (mxIsClass(input, "uint64"))
        T = 5, ptr_ull = (unsigned long long *)mxGetData(input);
    else if (mxIsClass(input, "int64"))
        T = 6, ptr_ll = (long long *)mxGetData(input);
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported data type.");

    // Convert data to armadillo output
    auto output = std::vector<arma::Col<dtype>>();
    if (vec_dim == 0)
        for (unsigned n = 0; n < d1; ++n)
        {
            auto tmp = arma::Col<dtype>(d2 * d3 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            if (T == 1)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_d[m * d1 + n];
            else if (T == 2)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_f[m * d1 + n];
            else if (T == 3)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_ui[m * d1 + n];
            else if (T == 4)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_i[m * d1 + n];
            else if (T == 5)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_ull[m * d1 + n];
            else if (T == 6)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_ll[m * d1 + n];
            output.push_back(tmp);
        }
    else if (vec_dim == 1)
        for (unsigned n = 0; n < d2; ++n)
        {
            auto tmp = arma::Col<dtype>(d1 * d3 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            if (T == 1)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_d[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 2)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_f[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 3)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_ui[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 4)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_i[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 5)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_ull[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 6)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_ll[m34 * d2 * d1 + n * d1 + m1];
            output.push_back(tmp);
        }
    else if (vec_dim == 2)
        for (unsigned n = 0; n < d3; ++n)
        {
            auto tmp = arma::Col<dtype>(d1 * d2 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            if (T == 1)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_d[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 2)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_f[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 3)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_ui[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 4)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_i[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 5)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_ull[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 6)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_ll[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            output.push_back(tmp);
        }
    else if (vec_dim == 3)
        for (unsigned n = 0; n < d4; ++n)
        {
            auto tmp = arma::Col<dtype>(d1 * d2 * d3, arma::fill::none);
            dtype *ptr = tmp.memptr();
            if (T == 1)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_d[n * d3 * d2 * d1 + m];
            else if (T == 2)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_f[n * d3 * d2 * d1 + m];
            else if (T == 3)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_ui[n * d3 * d2 * d1 + m];
            else if (T == 4)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_i[n * d3 * d2 * d1 + m];
            else if (T == 5)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_ull[n * d3 * d2 * d1 + m];
            else if (T == 6)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_ll[n * d3 * d2 * d1 + m];
            output.push_back(tmp);
        }
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Armadillo object dimensions must be 0,1,2 or 3");

    return output;
}

// Creates a std::vector of armadillo types from mxArray
// - e.g. MATLAB --> std::vector<arma::Mat<dtype>>
template <typename dtype>
std::vector<arma::Mat<dtype>> qd_mex_matlab2vector_Mat(const mxArray *input, unsigned vec_dim)
{
    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(input); // Number of dimensions - either 2, 3 or 4
    const mwSize *dims = mxGetDimensions(input);               // Read number of elements elements per dimension
    unsigned d1 = (unsigned)dims[0];                           // Number of elements on first dimension
    unsigned d2 = (unsigned)dims[1];                           // Number of elements on second dimension
    unsigned d3 = n_dim < 3 ? 1 : (unsigned)dims[2];           // Number of elements on third dimension
    unsigned d4 = n_dim < 4 ? 1 : (unsigned)dims[3];           // Number of elements on fourth dimension
    unsigned n_data = d1 * d2 * d3 * d4;

    if (mxIsComplex(input))
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Complex datatypes are not supported.");

    if (n_data == 0) // Return empty std::vector
        return std::vector<arma::Mat<dtype>>();

    // Generate pointers for the 6 supported MATLAB data types
    double *ptr_d = nullptr;
    float *ptr_f = nullptr;
    unsigned *ptr_ui = nullptr;
    int *ptr_i = nullptr;
    unsigned long long *ptr_ull = nullptr;
    long long *ptr_ll = nullptr;

    // Assign the MATLAB data to the correct pointer
    unsigned T = 0; // Type ID of input type
    if (mxIsDouble(input))
        T = 1, ptr_d = (double *)mxGetData(input);
    else if (mxIsSingle(input))
        T = 2, ptr_f = (float *)mxGetData(input);
    else if (mxIsClass(input, "uint32"))
        T = 3, ptr_ui = (unsigned *)mxGetData(input);
    else if (mxIsClass(input, "int32"))
        T = 4, ptr_i = (int *)mxGetData(input);
    else if (mxIsClass(input, "uint64"))
        T = 5, ptr_ull = (unsigned long long *)mxGetData(input);
    else if (mxIsClass(input, "int64"))
        T = 6, ptr_ll = (long long *)mxGetData(input);
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported data type.");

    // Convert data to armadillo output
    auto output = std::vector<arma::Mat<dtype>>();
    if (vec_dim == 0)
        for (unsigned n = 0; n < d1; ++n)
        {
            auto tmp = arma::Mat<dtype>(d2, d3 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            if (T == 1)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_d[m * d1 + n];
            else if (T == 2)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_f[m * d1 + n];
            else if (T == 3)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_ui[m * d1 + n];
            else if (T == 4)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_i[m * d1 + n];
            else if (T == 5)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_ull[m * d1 + n];
            else if (T == 6)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_ll[m * d1 + n];
            output.push_back(tmp);
        }
    else if (vec_dim == 1)
        for (unsigned n = 0; n < d2; ++n)
        {
            auto tmp = arma::Mat<dtype>(d1, d3 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            if (T == 1)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_d[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 2)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_f[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 3)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_ui[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 4)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_i[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 5)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_ull[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 6)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_ll[m34 * d2 * d1 + n * d1 + m1];
            output.push_back(tmp);
        }
    else if (vec_dim == 2)
        for (unsigned n = 0; n < d3; ++n)
        {
            auto tmp = arma::Mat<dtype>(d1, d2 * d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            if (T == 1)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_d[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 2)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_f[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 3)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_ui[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 4)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_i[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 5)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_ull[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 6)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_ll[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            output.push_back(tmp);
        }
    else if (vec_dim == 3)
        for (unsigned n = 0; n < d4; ++n)
        {
            auto tmp = arma::Mat<dtype>(d1, d2 * d3, arma::fill::none);
            dtype *ptr = tmp.memptr();
            if (T == 1)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_d[n * d3 * d2 * d1 + m];
            else if (T == 2)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_f[n * d3 * d2 * d1 + m];
            else if (T == 3)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_ui[n * d3 * d2 * d1 + m];
            else if (T == 4)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_i[n * d3 * d2 * d1 + m];
            else if (T == 5)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_ull[n * d3 * d2 * d1 + m];
            else if (T == 6)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_ll[n * d3 * d2 * d1 + m];
            output.push_back(tmp);
        }
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Armadillo object dimensions must be 0,1,2 or 3");

    return output;
}

// Creates a std::vector of armadillo types from mxArray
// - e.g. MATLAB --> std::vector<arma::Cube<dtype>>
template <typename dtype>
std::vector<arma::Cube<dtype>> qd_mex_matlab2vector_Cube(const mxArray *input, unsigned vec_dim)
{
    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(input); // Number of dimensions - either 2, 3 or 4
    const mwSize *dims = mxGetDimensions(input);               // Read number of elements elements per dimension
    unsigned d1 = (unsigned)dims[0];                           // Number of elements on first dimension
    unsigned d2 = (unsigned)dims[1];                           // Number of elements on second dimension
    unsigned d3 = n_dim < 3 ? 1 : (unsigned)dims[2];           // Number of elements on third dimension
    unsigned d4 = n_dim < 4 ? 1 : (unsigned)dims[3];           // Number of elements on fourth dimension
    unsigned n_data = d1 * d2 * d3 * d4;

    if (mxIsComplex(input))
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Complex datatypes are not supported.");

    if (n_data == 0) // Return empty std::vector
        return std::vector<arma::Cube<dtype>>();

    // Generate pointers for the 6 supported MATLAB data types
    double *ptr_d = nullptr;
    float *ptr_f = nullptr;
    unsigned *ptr_ui = nullptr;
    int *ptr_i = nullptr;
    unsigned long long *ptr_ull = nullptr;
    long long *ptr_ll = nullptr;

    // Assign the MATLAB data to the correct pointer
    unsigned T = 0; // Type ID of input type
    if (mxIsDouble(input))
        T = 1, ptr_d = (double *)mxGetData(input);
    else if (mxIsSingle(input))
        T = 2, ptr_f = (float *)mxGetData(input);
    else if (mxIsClass(input, "uint32"))
        T = 3, ptr_ui = (unsigned *)mxGetData(input);
    else if (mxIsClass(input, "int32"))
        T = 4, ptr_i = (int *)mxGetData(input);
    else if (mxIsClass(input, "uint64"))
        T = 5, ptr_ull = (unsigned long long *)mxGetData(input);
    else if (mxIsClass(input, "int64"))
        T = 6, ptr_ll = (long long *)mxGetData(input);
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Unsupported data type.");

    // Convert data to armadillo output
    auto output = std::vector<arma::Cube<dtype>>();
    if (vec_dim == 0)
        for (unsigned n = 0; n < d1; ++n)
        {
            auto tmp = arma::Cube<dtype>(d2, d3, d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            if (T == 1)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_d[m * d1 + n];
            else if (T == 2)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_f[m * d1 + n];
            else if (T == 3)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_ui[m * d1 + n];
            else if (T == 4)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_i[m * d1 + n];
            else if (T == 5)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_ull[m * d1 + n];
            else if (T == 6)
                for (unsigned m = 0; m < d2 * d3 * d4; ++m)
                    ptr[m] = (dtype)ptr_ll[m * d1 + n];
            output.push_back(tmp);
        }
    else if (vec_dim == 1)
        for (unsigned n = 0; n < d2; ++n)
        {
            auto tmp = arma::Cube<dtype>(d1, d3, d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            if (T == 1)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_d[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 2)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_f[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 3)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_ui[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 4)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_i[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 5)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_ull[m34 * d2 * d1 + n * d1 + m1];
            else if (T == 6)
                for (unsigned m34 = 0; m34 < d3 * d4; ++m34)
                    for (unsigned m1 = 0; m1 < d1; ++m1)
                        ptr[m34 * d1 + m1] = (dtype)ptr_ll[m34 * d2 * d1 + n * d1 + m1];
            output.push_back(tmp);
        }
    else if (vec_dim == 2)
        for (unsigned n = 0; n < d3; ++n)
        {
            auto tmp = arma::Cube<dtype>(d1, d2, d4, arma::fill::none);
            dtype *ptr = tmp.memptr();
            if (T == 1)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_d[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 2)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_f[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 3)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_ui[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 4)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_i[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 5)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_ull[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            else if (T == 6)
                for (unsigned m4 = 0; m4 < d4; ++m4)
                    for (unsigned m12 = 0; m12 < d1 * d2; ++m12)
                        ptr[m4 * d2 * d1 + m12] = (dtype)ptr_ll[m4 * d3 * d2 * d1 + n * d2 * d1 + m12];
            output.push_back(tmp);
        }
    else if (vec_dim == 3)
        for (unsigned n = 0; n < d4; ++n)
        {
            auto tmp = arma::Cube<dtype>(d1, d2, d3, arma::fill::none);
            dtype *ptr = tmp.memptr();
            if (T == 1)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_d[n * d3 * d2 * d1 + m];
            else if (T == 2)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_f[n * d3 * d2 * d1 + m];
            else if (T == 3)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_ui[n * d3 * d2 * d1 + m];
            else if (T == 4)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_i[n * d3 * d2 * d1 + m];
            else if (T == 5)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_ull[n * d3 * d2 * d1 + m];
            else if (T == 6)
                for (unsigned m = 0; m < d1 * d2 * d3; ++m)
                    ptr[m] = (dtype)ptr_ll[n * d3 * d2 * d1 + m];
            output.push_back(tmp);
        }
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", "Armadillo object dimensions must be 0,1,2 or 3");

    return output;
}

// Creates an mxArray based on the armadillo type, Initializes mxArray and reinterprets armadillo object
inline mxArray *qd_mex_init_output(arma::Row<float> *input, unsigned long long n_elem) // 1D-Single Row Vector
{
    mxArray *output = mxCreateNumericMatrix(1, n_elem, mxSINGLE_CLASS, mxREAL);
    *input = arma::Row<float>((float *)mxGetData(output), n_elem, false, true);
    return output;
}
inline mxArray *qd_mex_init_output(arma::Col<float> *input, unsigned long long n_elem, bool transpose = false) // 1D-Single Column Vector
{
    mxArray *output = transpose ? mxCreateNumericMatrix(1, n_elem, mxSINGLE_CLASS, mxREAL)
                                : mxCreateNumericMatrix(n_elem, 1, mxSINGLE_CLASS, mxREAL);
    *input = arma::Col<float>((float *)mxGetData(output), n_elem, false, true);
    return output;
}
inline mxArray *qd_mex_init_output(arma::Mat<float> *input, unsigned long long n_rows, unsigned long long n_cols) // 2D-Single
{
    mxArray *output = mxCreateNumericMatrix(n_rows, n_cols, mxSINGLE_CLASS, mxREAL);
    *input = arma::Mat<float>((float *)mxGetData(output), n_rows, n_cols, false, true);
    return output;
}
inline mxArray *qd_mex_init_output(arma::Cube<float> *input, unsigned long long n_rows, unsigned long long n_cols, unsigned long long n_slices) // 3D-Single
{
    mwSize dims[3] = {(mwSize)n_rows, (mwSize)n_cols, (mwSize)n_slices};
    mxArray *output = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
    *input = arma::Cube<float>((float *)mxGetData(output), n_rows, n_cols, n_slices, false, true);
    return output;
}
inline mxArray *qd_mex_init_output(arma::Row<double> *input, unsigned long long n_elem) // 1D-Double Row Vector
{
    mxArray *output = mxCreateNumericMatrix(1, n_elem, mxDOUBLE_CLASS, mxREAL);
    *input = arma::Row<double>((double *)mxGetData(output), n_elem, false, true);
    return output;
}
inline mxArray *qd_mex_init_output(arma::Col<double> *input, unsigned long long n_elem, bool transpose = false) // 1D-Double Column Vector
{
    mxArray *output = transpose ? mxCreateNumericMatrix(1, n_elem, mxDOUBLE_CLASS, mxREAL)
                                : mxCreateNumericMatrix(n_elem, 1, mxDOUBLE_CLASS, mxREAL);
    *input = arma::Col<double>((double *)mxGetData(output), n_elem, false, true);
    return output;
}
inline mxArray *qd_mex_init_output(arma::Mat<double> *input, unsigned long long n_rows, unsigned long long n_cols) // 2D-Double
{
    mxArray *output = mxCreateNumericMatrix(n_rows, n_cols, mxDOUBLE_CLASS, mxREAL);
    *input = arma::Mat<double>((double *)mxGetData(output), n_rows, n_cols, false, true);
    return output;
}
inline mxArray *qd_mex_init_output(arma::Cube<double> *input, unsigned long long n_rows, unsigned long long n_cols, unsigned long long n_slices) // 3D-Double
{
    mwSize dims[3] = {(mwSize)n_rows, (mwSize)n_cols, (mwSize)n_slices};
    mxArray *output = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    *input = arma::Cube<double>((double *)mxGetData(output), n_rows, n_cols, n_slices, false, true);
    return output;
}
inline mxArray *qd_mex_init_output(arma::Row<unsigned> *input, unsigned long long n_elem) // 1D-UINT32 Row Vector
{
    mxArray *output = mxCreateNumericMatrix(1, n_elem, mxUINT32_CLASS, mxREAL);
    *input = arma::Row<unsigned>((unsigned *)mxGetData(output), n_elem, false, true);
    return output;
}
inline mxArray *qd_mex_init_output(arma::Col<unsigned> *input, unsigned long long n_elem, bool transpose = false) // 1D-UINT32 Column Vector
{
    mxArray *output = transpose ? mxCreateNumericMatrix(1, n_elem, mxUINT32_CLASS, mxREAL)
                                : mxCreateNumericMatrix(n_elem, 1, mxUINT32_CLASS, mxREAL);
    *input = arma::Col<unsigned>((unsigned *)mxGetData(output), n_elem, false, true);
    return output;
}
inline mxArray *qd_mex_init_output(arma::Mat<unsigned> *input, unsigned long long n_rows, unsigned long long n_cols) // 2D-UINT32
{
    mxArray *output = mxCreateNumericMatrix(n_rows, n_cols, mxUINT32_CLASS, mxREAL);
    *input = arma::Mat<unsigned>((unsigned *)mxGetData(output), n_rows, n_cols, false, true);
    return output;
}
inline mxArray *qd_mex_init_output(arma::Cube<unsigned> *input, unsigned long long n_rows, unsigned long long n_cols, unsigned long long n_slices) // 3D-UINT32
{
    mwSize dims[3] = {(mwSize)n_rows, (mwSize)n_cols, (mwSize)n_slices};
    mxArray *output = mxCreateNumericArray(3, dims, mxUINT32_CLASS, mxREAL);
    *input = arma::Cube<unsigned>((unsigned *)mxGetData(output), n_rows, n_cols, n_slices, false, true);
    return output;
}

#endif
