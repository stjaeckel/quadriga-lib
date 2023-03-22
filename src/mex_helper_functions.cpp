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

#include "mex.h"
#include <string>
#include <armadillo>
#include <utility> // for std::move
#include <cstring> // For std::memcopy

// Read a scalar input from MATLAB and convert it to a desired c++ output type
// Returns NaN for empty input (0 in case of integer types)
template <typename dtype>
inline dtype qd_mex_get_scalar(const mxArray *input, std::string var_name)
{
    std::string error_message = "Input '" + var_name + "' has an unsupported data type.";
    if (mxGetNumberOfElements(input) == 0)
        return dtype(NAN);
    else if (mxIsDouble(input))
    {
        double *tmp = (double *)mxGetData(input);
        return dtype(*tmp);
    }
    else if (mxIsSingle(input))
    {
        float *tmp = (float *)mxGetData(input);
        return dtype(*tmp);
    }
    else if (mxIsClass(input, "uint32") || mxIsClass(input, "int32"))
    {
        unsigned *tmp = (unsigned *)mxGetData(input);
        return dtype(*tmp);
    }
    else if (mxIsClass(input, "logical"))
    {
        bool *tmp = (bool *)mxGetData(input);
        return dtype(*tmp);
    }
    else
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", error_message.c_str());

    return dtype(0);
}

// Reinterpret MATLAB Array to Armadillo Column Vector
template <typename dtype>
inline arma::Col<dtype> qd_mex_reinterpret_Col(const mxArray *input)
{
    unsigned d1 = (unsigned)mxGetM(input); // Number of elements on first dimension
    unsigned d2 = (unsigned)mxGetN(input); // Number of elements on other dimensions
    return arma::Col<dtype>((dtype *)mxGetData(input), d1 * d2, false, true);
}

// Reinterpret MATLAB Array to Armadillo Matrix
template <typename dtype>
inline arma::Mat<dtype> qd_mex_reinterpret_Mat(const mxArray *input)
{
    unsigned d1 = (unsigned)mxGetM(input); // Number of elements on first dimension
    unsigned d2 = (unsigned)mxGetN(input); // Number of elements on other dimensions
    return arma::Mat<dtype>((dtype *)mxGetData(input), d1, d2, false, true);
}

// Reinterpret MATLAB Array to Armadillo Cube
template <typename dtype>
inline arma::Cube<dtype> qd_mex_reinterpret_Cube(const mxArray *input)
{
    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(input); // Number of dimensions - either 2 or 3
    const mwSize *dims = mxGetDimensions(input);               // Read number of elements elements per dimension
    unsigned d1 = (unsigned)dims[0];                           // Number of elements on first dimension
    unsigned d2 = (unsigned)dims[1];                           // Number of elements on second dimension
    unsigned d3 = n_dim < 3 ? 1 : (unsigned)dims[2];           // Number of elements on third dimension
    unsigned d4 = n_dim < 4 ? 1 : (unsigned)dims[3];           // Number of elements on fourth dimension
    return arma::Cube<dtype>((dtype *)mxGetData(input), d1, d2, d3 * d4, false, true);
}

// Reads input and converts it to desired c++ type, creates a copy of the input
template <typename dtype>
inline arma::Col<dtype> qd_mex_typecast_Col(const mxArray *input, std::string var_name)
{
    unsigned d1 = (unsigned)mxGetM(input); // Number of elements on first dimension
    unsigned d2 = (unsigned)mxGetN(input); // Number of elements on other dimensions

    if (mxIsDouble(input))
    {
        arma::Col<double> tmp = arma::Col<double>((double *)mxGetData(input), d1 * d2, false, true);
        return arma::conv_to<arma::Col<dtype>>::from(tmp);
    }
    else if (mxIsSingle(input))
    {
        arma::Col<float> tmp = arma::Col<float>((float *)mxGetData(input), d1 * d2, false, true);
        return arma::conv_to<arma::Col<dtype>>::from(tmp);
    }
    else if (mxIsClass(input, "uint32") || mxIsClass(input, "int32"))
    {
        arma::Col<int> tmp = arma::Col<int>((int *)mxGetData(input), d1 * d2, false, true);
        return arma::conv_to<arma::Col<dtype>>::from(tmp);
    }
    else if (mxIsClass(input, "uint64") || mxIsClass(input, "int64"))
    {
        arma::Col<long long> tmp = arma::Col<long long>((long long *)mxGetData(input), d1 * d2, false, true);
        return arma::conv_to<arma::Col<dtype>>::from(tmp);
    }
    else
    {
        std::string error_message = "Input '" + var_name + "' has an unsupported data type.";
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", error_message.c_str());
    }

    return arma::Col<dtype>();
}

// Reads input and converts it to desired c++ type, creates a copy of the input
template <typename dtype>
inline arma::Mat<dtype> qd_mex_typecast_Mat(const mxArray *input, std::string var_name)
{
    unsigned d1 = (unsigned)mxGetM(input); // Number of elements on first dimension
    unsigned d2 = (unsigned)mxGetN(input); // Number of elements on other dimensions

    if (mxIsDouble(input))
    {
        arma::Mat<double> tmp = arma::Mat<double>((double *)mxGetData(input), d1, d2, false, true);
        return arma::conv_to<arma::Mat<dtype>>::from(tmp); // Copy and cast data
    }
    else if (mxIsSingle(input))
    {
        arma::Mat<float> tmp = arma::Mat<float>((float *)mxGetData(input), d1, d2, false, true);
        return arma::conv_to<arma::Mat<dtype>>::from(tmp); // Copy and cast data
    }
    else if (mxIsClass(input, "uint32") || mxIsClass(input, "int32"))
    {
        arma::Mat<int> tmp = arma::Mat<int>((int *)mxGetData(input), d1, d2, false, true);
        return arma::conv_to<arma::Mat<dtype>>::from(tmp); // Copy and cast data
    }
    else if (mxIsClass(input, "uint64") || mxIsClass(input, "int64"))
    {
        arma::Mat<long long> tmp = arma::Mat<long long>((long long *)mxGetData(input), d1, d2, false, true);
        return arma::conv_to<arma::Mat<dtype>>::from(tmp); // Copy and cast data
    }
    else
    {
        std::string error_message = "Input '" + var_name + "' has an unsupported data type.";
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", error_message.c_str());
    }

    return arma::Mat<dtype>();
}

// Reads input and converts it to desired c++ type, creates a copy of the input
template <typename dtype>
inline arma::Cube<dtype> qd_mex_typecast_Cube(const mxArray *input, std::string var_name)
{
    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(input); // Number of dimensions - either 2 or 3
    const mwSize *dims = mxGetDimensions(input);               // Read number of elements elements per dimension
    unsigned d1 = (unsigned)dims[0];                           // Number of elements on first dimension
    unsigned d2 = (unsigned)dims[1];                           // Number of elements on second dimension
    unsigned d3 = n_dim < 3 ? 1 : (unsigned)dims[2];           // Number of elements on third dimension
    unsigned d4 = n_dim < 4 ? 1 : (unsigned)dims[3];           // Number of elements on fourth dimension

    if (mxIsDouble(input))
    {
        arma::Cube<double> tmp = arma::Cube<double>((double *)mxGetData(input), d1, d2, d3 * d4, false, true);
        return arma::conv_to<arma::Cube<dtype>>::from(tmp); // Copy and cast data
    }
    else if (mxIsSingle(input))
    {
        arma::Cube<float> tmp = arma::Cube<float>((float *)mxGetData(input), d1, d2, d3 * d4, false, true);
        return arma::conv_to<arma::Cube<dtype>>::from(tmp); // Copy and cast data
    }
    else if (mxIsClass(input, "uint32") || mxIsClass(input, "int32"))
    {
        arma::Cube<int> tmp = arma::Cube<int>((int *)mxGetData(input), d1, d2, d3 * d4, false, true);
        return arma::conv_to<arma::Cube<dtype>>::from(tmp); // Copy and cast data
    }
    else if (mxIsClass(input, "uint64") || mxIsClass(input, "int64"))
    {
        arma::Cube<long long> tmp = arma::Cube<long long>((long long *)mxGetData(input), d1, d2, d3 * d4, false, true);
        return arma::conv_to<arma::Cube<dtype>>::from(tmp); // Copy and cast data
    }
    else
    {
        std::string error_message = "Input '" + var_name + "' has an unsupported data type.";
        mexErrMsgIdAndTxt("MATLAB:unexpectedCPPexception", error_message.c_str());
    }

    return arma::Cube<dtype>();
}

// Creates an mxArray based on the armadillo input type, copies content
inline mxArray *qd_mex_copy2matlab(float *input) // Scalar Single
{
    mxArray *output = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    std::memcpy((float *)mxGetData(output), input, sizeof(float));
    return output;
}
inline mxArray *qd_mex_copy2matlab(arma::Row<float> *input) // 1D-Single Row Vector
{
    mxArray *output = mxCreateNumericMatrix(1, (mwSize)input->n_elem, mxSINGLE_CLASS, mxREAL);
    std::memcpy((float *)mxGetData(output), input->memptr(), sizeof(float) * input->n_elem);
    return output;
}
inline mxArray *qd_mex_copy2matlab(arma::Col<float> *input, bool transpose) // 1D-Single Column Vector
{
    mxArray *output = transpose ? mxCreateNumericMatrix(1, (mwSize)input->n_elem, mxSINGLE_CLASS, mxREAL)
                                : mxCreateNumericMatrix((mwSize)input->n_elem, 1, mxSINGLE_CLASS, mxREAL);
    std::memcpy((float *)mxGetData(output), input->memptr(), sizeof(float) * input->n_elem);
    return output;
}
inline mxArray *qd_mex_copy2matlab(arma::Col<float> *input)
{
    return qd_mex_copy2matlab(input, false);
}
inline mxArray *qd_mex_copy2matlab(arma::Mat<float> *input) // 2D-Single
{
    mxArray *output = mxCreateNumericMatrix((mwSize)input->n_rows, (mwSize)input->n_cols, mxSINGLE_CLASS, mxREAL);
    std::memcpy((float *)mxGetData(output), input->memptr(), sizeof(float) * input->n_elem);
    return output;
}
inline mxArray *qd_mex_copy2matlab(arma::Cube<float> *input) // 3D-Single
{
    mwSize dims[3] = {(mwSize)input->n_rows, (mwSize)input->n_cols, (mwSize)input->n_slices};
    mxArray *output = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
    std::memcpy((float *)mxGetData(output), input->memptr(), sizeof(float) * input->n_elem);
    return output;
}

inline mxArray *qd_mex_copy2matlab(double *input) // Scalar Double
{
    mxArray *output = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    std::memcpy((double *)mxGetData(output), input, sizeof(double));
    return output;
}
inline mxArray *qd_mex_copy2matlab(arma::Row<double> *input) // 1D-Double Row Vector
{
    mxArray *output = mxCreateNumericMatrix(1, (mwSize)input->n_elem, mxDOUBLE_CLASS, mxREAL);
    std::memcpy((double *)mxGetData(output), input->memptr(), sizeof(double) * input->n_elem);
    return output;
}
inline mxArray *qd_mex_copy2matlab(arma::Col<double> *input, bool transpose) // // 1D-Double Column Vector
{
    mxArray *output = transpose ? mxCreateNumericMatrix(1, (mwSize)input->n_elem, mxDOUBLE_CLASS, mxREAL)
                                : mxCreateNumericMatrix((mwSize)input->n_elem, 1, mxDOUBLE_CLASS, mxREAL);
    std::memcpy((double *)mxGetData(output), input->memptr(), sizeof(double) * input->n_elem);
    return output;
}
inline mxArray *qd_mex_copy2matlab(arma::Col<double> *input)
{
    return qd_mex_copy2matlab(input, false);
}
inline mxArray *qd_mex_copy2matlab(arma::Mat<double> *input) // 2D-Double
{
    mxArray *output = mxCreateNumericMatrix((mwSize)input->n_rows, (mwSize)input->n_cols, mxDOUBLE_CLASS, mxREAL);
    std::memcpy((double *)mxGetData(output), input->memptr(), sizeof(double) * input->n_elem);
    return output;
}
inline mxArray *qd_mex_copy2matlab(arma::Cube<double> *input) // 3D-Double
{
    mwSize dims[3] = {(mwSize)input->n_rows, (mwSize)input->n_cols, (mwSize)input->n_slices};
    mxArray *output = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    std::memcpy((double *)mxGetData(output), input->memptr(), sizeof(double) * input->n_elem);
    return output;
}

inline mxArray *qd_mex_copy2matlab(unsigned *input) // Scalar UINT32
{
    mxArray *output = mxCreateNumericMatrix(1, 1, mxUINT32_CLASS, mxREAL);
    std::memcpy((unsigned *)mxGetData(output), input, sizeof(unsigned));
    return output;
}
inline mxArray *qd_mex_copy2matlab(arma::Row<unsigned> *input) // 1D-UINT32 Row Vector
{
    mxArray *output = mxCreateNumericMatrix(1, (mwSize)input->n_elem, mxUINT32_CLASS, mxREAL);
    std::memcpy((unsigned *)mxGetData(output), input->memptr(), sizeof(unsigned) * input->n_elem);
    return output;
}
inline mxArray *qd_mex_copy2matlab(arma::Col<unsigned> *input, bool transpose) // 1D-UINT32 Column Vector
{
    mxArray *output = transpose ? mxCreateNumericMatrix(1, (mwSize)input->n_elem, mxUINT32_CLASS, mxREAL)
                                : mxCreateNumericMatrix((mwSize)input->n_elem, 1, mxUINT32_CLASS, mxREAL);
    std::memcpy((unsigned *)mxGetData(output), input->memptr(), sizeof(unsigned) * input->n_elem);
    return output;
}
inline mxArray *qd_mex_copy2matlab(arma::Col<unsigned> *input)
{
    return qd_mex_copy2matlab(input, false);
}
inline mxArray *qd_mex_copy2matlab(arma::Mat<unsigned> *input) // 2D-UINT32
{
    mxArray *output = mxCreateNumericMatrix((mwSize)input->n_rows, (mwSize)input->n_cols, mxUINT32_CLASS, mxREAL);
    std::memcpy((unsigned *)mxGetData(output), input->memptr(), sizeof(unsigned) * input->n_elem);
    return output;
}
inline mxArray *qd_mex_copy2matlab(arma::Cube<unsigned> *input) // 3D-UINT32
{
    mwSize dims[3] = {(mwSize)input->n_rows, (mwSize)input->n_cols, (mwSize)input->n_slices};
    mxArray *output = mxCreateNumericArray(3, dims, mxUINT32_CLASS, mxREAL);
    std::memcpy((unsigned *)mxGetData(output), input->memptr(), sizeof(unsigned) * input->n_elem);
    return output;
}
