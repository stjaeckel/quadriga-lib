// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef qd_helpers_mex_hpp
#define qd_helpers_mex_hpp

#include "mex.h"
#include <string>
#include <armadillo>
#include <utility>
#include <cstring>
#include <any>
#include <type_traits>
#include <array>
#include <limits>
#include <cmath>
#include <unordered_map>
#include <vector>
#include <complex>

// Raises a C++ error message in MATLAB / Octave. A C++ message names an entry of a vector input only by the prefix
// "Entry <i>: " or "Snapshot <i>: " with a 0-based <i>; every such prefix is rewritten to <i> + 1, so that MATLAB
// users read 1-based indices. User text inside a message (e.g. a file name) that contains such a prefix is rewritten
// as well; no library message is affected.
inline void qd_mex_cpp_error(std::string msg)
{
    for (const std::string p : {"Entry ", "Snapshot "})
        for (size_t pos = msg.find(p); pos != std::string::npos; pos = msg.find(p, pos))
        {
            const size_t i = pos + p.size(), end = msg.find_first_not_of("0123456789", i);
            pos = i;
            if (end == i || end - i > 18 || msg.compare(end, 2, ": ") != 0) // end == npos fails the compare
                continue;
            const std::string n = std::to_string(std::stoull(msg.substr(i, end - i)) + 1ULL);
            msg.replace(i, end - i, n);
            pos = i + n.size();
        }
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
}

// Macro to redirect any exception to MATLAB / Octave
#define CALL_QD(expr)                    \
    do                                   \
    {                                    \
        try                              \
        {                                \
            expr;                        \
        }                                \
        catch (const std::exception &ex) \
        {                                \
            qd_mex_cpp_error(ex.what()); \
        }                                \
    } while (0)

// Complex data uses the interleaved complex API (MATLAB R2018a, Octave 8.4 and later): the data of a complex array is
// one block of std::complex values. Output helpers take std::complex<float> or std::complex<double> as dtype.
template <typename T>
struct qd_mex_is_complex : std::false_type
{
};
template <typename T>
struct qd_mex_is_complex<std::complex<T>> : std::true_type
{
};

template <typename dtype>
constexpr mxComplexity qd_mex_complexity()
{
    return qd_mex_is_complex<dtype>::value ? mxCOMPLEX : mxREAL;
}

// Check if MATLAB array type matches the requested C++ dtype
template <typename dtype>
inline bool qd_mex_type_matches(const mxArray *input)
{
    if constexpr (std::is_same_v<dtype, double>)
        return mxIsDouble(input);
    else if constexpr (std::is_same_v<dtype, float>)
        return mxIsSingle(input);
    else if constexpr (std::is_same_v<dtype, int>)
        return mxIsClass(input, "int32");
    else if constexpr (std::is_same_v<dtype, unsigned>)
        return mxIsClass(input, "uint32");
    else if constexpr (std::is_same_v<dtype, long long>)
        return mxIsClass(input, "int64");
    else if constexpr (std::is_same_v<dtype, unsigned long long>)
        return mxIsClass(input, "uint64");
    else
        return false;
}

// Map C++ type to MATLAB class ID
template <typename dtype>
inline mxClassID qd_mex_class_id()
{
    if constexpr (qd_mex_is_complex<dtype>::value)
        return qd_mex_class_id<typename dtype::value_type>();
    else if constexpr (std::is_same_v<dtype, double>)
        return mxDOUBLE_CLASS;
    else if constexpr (std::is_same_v<dtype, float>)
        return mxSINGLE_CLASS;
    else if constexpr (std::is_same_v<dtype, int>)
        return mxINT32_CLASS;
    else if constexpr (std::is_same_v<dtype, unsigned>)
        return mxUINT32_CLASS;
    else if constexpr (std::is_same_v<dtype, long long>)
        return mxINT64_CLASS;
    else if constexpr (std::is_same_v<dtype, unsigned long long>)
        return mxUINT64_CLASS;
    else
        static_assert(!std::is_same_v<dtype, dtype>, "qd_mex_class_id: unsupported dtype");
}

// True if the value v (MATLAB type stype) can be converted to dtype without loss. For integer targets
// (except bool) this rejects NaN, Inf, fractional values and values outside the range of dtype; integer sources
// are checked against the range only. Floating-point and bool targets accept every value.
template <typename dtype, typename stype>
inline bool qd_mex_fits(stype v)
{
    if constexpr (std::is_same_v<dtype, stype> || !std::is_integral_v<dtype> || std::is_same_v<dtype, bool>)
        return true;
    else if constexpr (std::is_floating_point_v<stype>)
    {
        // max + 1 = 2^digits and min = -2^digits (signed) or 0 are exact in double
        constexpr double hi = 2.0 * (double)(std::numeric_limits<dtype>::max() / 2 + 1);
        constexpr double lo = std::is_signed_v<dtype> ? -hi : 0.0;
        const double x = (double)v;
        return x >= lo && x < hi && x == (double)(dtype)x; // NaN fails the comparisons; the cast is in range
    }
    else if constexpr (std::is_signed_v<stype> && !std::is_signed_v<dtype>)
        return v >= 0 && (unsigned long long)v <= (unsigned long long)std::numeric_limits<dtype>::max();
    else if constexpr (std::is_signed_v<stype>)
        return (long long)v >= (long long)std::numeric_limits<dtype>::min() &&
               (long long)v <= (long long)std::numeric_limits<dtype>::max();
    else
        return (unsigned long long)v <= (unsigned long long)std::numeric_limits<dtype>::max();
}

// Convert one value to dtype; sets "bad" if it does not fit (see qd_mex_fits) and returns 0 in that case
template <typename dtype, typename stype>
inline dtype qd_mex_cast(stype v, bool &bad)
{
    const bool ok = qd_mex_fits<dtype, stype>(v);
    bad |= !ok;
    return ok ? (dtype)v : dtype(0);
}

// Calls f with a typed pointer to the data of a real numeric or logical array (all integer classes, single,
// double, logical); any other class raises "Input 'x' has an unsupported data type." (var_name empty: "Unsupported data
// type."). The class is checked before the data pointer is read: Octave raises an internal error for mxGetData of
// classes without numeric data (e.g. function handles).
template <typename F>
inline void qd_mex_visit_data(const mxArray *input, const std::string &var_name, F &&f)
{
    const bool numeric = mxIsNumeric(input) || mxIsLogical(input);
    const void *d = numeric ? mxGetData(input) : nullptr;
    switch (numeric ? mxGetClassID(input) : mxUNKNOWN_CLASS)
    {
    case mxDOUBLE_CLASS:
        return f((const double *)d);
    case mxSINGLE_CLASS:
        return f((const float *)d);
    case mxINT8_CLASS:
        return f((const signed char *)d);
    case mxUINT8_CLASS:
        return f((const unsigned char *)d);
    case mxINT16_CLASS:
        return f((const short *)d);
    case mxUINT16_CLASS:
        return f((const unsigned short *)d);
    case mxINT32_CLASS:
        return f((const int *)d);
    case mxUINT32_CLASS:
        return f((const unsigned *)d);
    case mxINT64_CLASS:
        return f((const long long *)d);
    case mxUINT64_CLASS:
        return f((const unsigned long long *)d);
    case mxLOGICAL_CLASS:
        return f((const mxLogical *)d);
    default:
        break;
    }
    const std::string msg = var_name.empty() ? "Unsupported data type." : "Input '" + var_name + "' has an unsupported data type.";
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
}

// Error for integer input that does not fit dtype
// - Types up to 32 bit state the range, e.g. "Input 'x' must be an integer from 0 to 4294967295."
// - 64-bit types state only the sign, e.g. "Input 'x' must be a non-negative integer."; their limits lie far
//   beyond any value an input can sensibly have
// - one_based: the range starts at 1 (1-based index inputs)
template <typename dtype>
inline void qd_mex_integer_error(const std::string &var_name, bool scalar, bool one_based = false)
{
    std::string what; // Completes "must be ..."
    if constexpr (sizeof(dtype) < 8)
    {
        const std::string lo = one_based ? std::string("1") : std::to_string(std::numeric_limits<dtype>::min());
        what = std::string(scalar ? "an integer" : "integers") + " from " + lo + " to " + std::to_string(std::numeric_limits<dtype>::max());
    }
    else if (one_based)
        what = scalar ? "a positive integer" : "positive integers";
    else if constexpr (std::is_signed_v<dtype>)
        what = scalar ? "an integer" : "integers";
    else
        what = scalar ? "a non-negative integer" : "non-negative integers";
    const std::string msg = scalar ? (var_name.empty() ? std::string("Input") : "Input '" + var_name + "'") + " must be " + what + "."
                                   : (var_name.empty() ? std::string("Entries") : "Entries in '" + var_name + "'") + " must be " + what + ".";
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
}

// Reject complex (unless complex_ok) and sparse input, and input with more than max_dims dimensions
// (mxGetData of a sparse array holds only the non-zero values, so it cannot be read as a dense array)
inline void qd_mex_check_input(const mxArray *input, const std::string &var_name = "", size_t max_dims = 4,
                               bool complex_ok = false)
{
    if (!complex_ok && mxIsComplex(input))
    {
        if (var_name.empty())
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Complex inputs are not supported.");
        const std::string msg = "Input '" + var_name + "' is complex; complex inputs are not supported.";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    }
    if (mxIsSparse(input))
    {
        if (var_name.empty())
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Sparse inputs are not supported.");
        const std::string msg = "Input '" + var_name + "' is sparse; sparse inputs are not supported.";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    }
    if ((size_t)mxGetNumberOfDimensions(input) > max_dims)
    {
        const std::string msg = var_name.empty()
                                    ? "Inputs with more than " + std::to_string(max_dims) + " dimensions are not supported."
                                    : "Input '" + var_name + "' has more than " + std::to_string(max_dims) + " dimensions.";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    }
}

// Read up to 4 dimensions from a MATLAB array. Missing dims default to 1.
// Returns {d1, d2, d3, d4, n_data}. Rejects complex (unless complex_ok) and sparse input and input with more than
// 4 dimensions.
inline std::array<size_t, 5> qd_mex_get_dims(const mxArray *input, const std::string &var_name = "", bool complex_ok = false)
{
    qd_mex_check_input(input, var_name, 4, complex_ok);
    size_t n_dim = (size_t)mxGetNumberOfDimensions(input);
    const mwSize *dims = mxGetDimensions(input);
    size_t d1 = (size_t)dims[0];
    size_t d2 = (size_t)dims[1];
    size_t d3 = n_dim < 3 ? 1 : (size_t)dims[2];
    size_t d4 = n_dim < 4 ? 1 : (size_t)dims[3];
    return {d1, d2, d3, d4, d1 * d2 * d3 * d4};
}

// Used by qd_mex_get_scalar: reads the one element and casts it to <dtype>; sets "bad" when the value
// does not fit an integer type (see qd_mex_fits); returns default_value for empty input; more than one element
// raises "Input 'x' must be a scalar."
template <typename dtype>
inline dtype qd_mex_read_scalar(const mxArray *input, const std::string &var_name, dtype default_value, bool &bad)
{
    if (mxGetNumberOfElements(input) == 0)
        return default_value;

    qd_mex_check_input(input, var_name); // Rejects complex and sparse input
    if (mxGetNumberOfElements(input) != 1)
    {
        const std::string msg = var_name.empty() ? "Input must be a scalar." : "Input '" + var_name + "' must be a scalar.";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    }

    // Integer classes are read without a detour through double
    dtype value = default_value;
    qd_mex_visit_data(input, var_name, [&](auto *p) { value = qd_mex_cast<dtype>(*p, bad); });
    return value;
}

// Read a scalar input
// - Casts to <dtype>; returns default_value for empty input; more than one element is an error
// - Integer types (except bool) need an integer value in their range: NaN, Inf, fractional, negative (unsigned) or
//   too large values raise an error
// - Rejects complex and sparse input
template <typename dtype>
inline dtype qd_mex_get_scalar(const mxArray *input, std::string var_name, dtype default_value)
{
    bool bad = false;
    const dtype value = qd_mex_read_scalar<dtype>(input, var_name, default_value, bad);
    if (bad)
        qd_mex_integer_error<dtype>(var_name, true);
    return value;
}

// Overload without default value: returns NaN for empty input; only for floating-point types, integer and bool
// types must state their default
template <typename dtype>
inline dtype qd_mex_get_scalar(const mxArray *input, std::string var_name = "")
{
    static_assert(std::is_floating_point_v<dtype>, "qd_mex_get_scalar: integer and bool types need a default value.");
    return qd_mex_get_scalar<dtype>(input, var_name, dtype(NAN));
}

// Reads a count argument (number of coefficients, generators, ...) as double, so that a negative, non-finite or
// non-integer value is rejected with the name of the argument ("Input 'x' must be an integer >= 1.") instead of
// wrapping around in the conversion to an unsigned type; returns default_value for empty input
inline arma::uword qd_mex_get_count(const mxArray *input, const char *var_name, double default_value)
{
    const double v = qd_mex_get_scalar<double>(input, var_name, default_value);
    if (!(v >= 1.0) || !(v < 9.0e18) || v != std::floor(v))
    {
        const std::string msg = std::string("Input '") + var_name + "' must be an integer >= 1.";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    }
    return (arma::uword)v;
}

// Read the seed of a random number generator: -1 (random) or 0 ... 2^63-1
// - int64 and uint64 inputs are read without a detour through double, so that large seeds keep their value
// - Returns -1 for empty input
inline long long qd_mex_get_seed(const mxArray *input, const std::string &var_name = "seed")
{
    bool bad = false;
    const long long seed = qd_mex_read_scalar<long long>(input, var_name, -1LL, bad);
    if (bad || seed < -1LL)
    {
        const std::string msg = "Input '" + var_name + "' must be -1 or a non-negative integer.";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    }
    return seed;
}

// Read a 1-based index (MATLAB) and return it 0-based (C++); any integer type except bool
// - Returns 0 (the first index) for empty input
// - 0 raises "Input 'x' cannot be 0 (1-based index)."; NaN, Inf, fractional, negative or too large values raise
//   "Input 'x' must be an integer from 1 to <max>."
template <typename dtype = arma::uword>
inline dtype qd_mex_get_index(const mxArray *input, const std::string &var_name)
{
    static_assert(std::is_integral_v<dtype> && !std::is_same_v<dtype, bool>, "qd_mex_get_index: integer types only.");
    if (mxGetNumberOfElements(input) == 0)
        return dtype(0);

    bool bad = false;
    const dtype value = qd_mex_read_scalar<dtype>(input, var_name, dtype(0), bad);
    if constexpr (std::is_signed_v<dtype>)
        bad |= value < 0;
    if (bad)
        qd_mex_integer_error<dtype>(var_name, true, true);
    if (value == 0)
    {
        const std::string msg = "Input '" + var_name + "' cannot be 0 (1-based index).";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    }
    return value - 1;
}

// Read a string input from MATLAB and convert it to a std::string
// - Returns default_value (e.g. "") for empty input; errors name 'var_name' if given
inline std::string qd_mex_get_string(const mxArray *input, std::string default_value = "",
                                     const std::string &var_name = "")
{
    if (input != nullptr && mxGetNumberOfElements(input) == 0)
        return default_value;

    if (input == nullptr || !mxIsClass(input, "char"))
    {
        const std::string msg = var_name.empty() ? "Input must be a string (character array)."
                                                 : "Input '" + var_name + "' must be a string (character array).";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    }

    auto chr = mxArrayToString(input);
    std::string data = std::string(chr);
    mxFree(chr);

    return data;
}

// Read a string input whose empty value is meaningful (e.g. a name prefix)
// - [] (or any empty input that is not a char array) gives default_value, '' gives an empty string
inline std::string qd_mex_get_string_or_empty(const mxArray *input, std::string default_value,
                                              const std::string &var_name = "")
{
    if (input != nullptr && mxGetNumberOfElements(input) == 0)
        return mxIsChar(input) ? std::string() : default_value;
    return qd_mex_get_string(input, default_value, var_name);
}

// The reinterpret functions use the MATLAB data in place. Complex arrays are rejected (qd_mex_get_dims): their data
// holds interleaved real and imaginary parts and cannot be read as a real array; they are read with
// qd_mex_get_complex_Cube. Sparse arrays store only their non-zero values and are rejected as well.

// Reinterpret MATLAB Array to Armadillo Column Vector
template <typename dtype>
inline arma::Col<dtype> qd_mex_reinterpret_Col(const mxArray *input, const std::string &var_name = "")
{
    auto [d1, d2, d3, d4, n_data] = qd_mex_get_dims(input, var_name); // Rejects complex, sparse and 5-D input

    if (n_data == 0)
        return arma::Col<dtype>();

    return arma::Col<dtype>((dtype *)mxGetData(input), n_data, false, true);
}

// Reinterpret MATLAB Array to Armadillo Matrix
template <typename dtype>
inline arma::Mat<dtype> qd_mex_reinterpret_Mat(const mxArray *input, const std::string &var_name = "")
{
    auto [d1, d2, d3, d4, n_data] = qd_mex_get_dims(input, var_name); // Rejects complex, sparse and 5-D input

    if (n_data == 0)
        return arma::Mat<dtype>(d1, d2 * d3 * d4);

    return arma::Mat<dtype>((dtype *)mxGetData(input), d1, d2 * d3 * d4, false, true);
}

// Reinterpret MATLAB Array to Armadillo Cube
template <typename dtype>
inline arma::Cube<dtype> qd_mex_reinterpret_Cube(const mxArray *input, const std::string &var_name = "")
{
    auto [d1, d2, d3, d4, n_data] = qd_mex_get_dims(input, var_name); // Rejects complex, sparse and 5-D input

    if (n_data == 0)
        return arma::Cube<dtype>(d1, d2, d3 * d4);

    return arma::Cube<dtype>((dtype *)mxGetData(input), d1, d2, d3 * d4, false, true);
}

template <typename dtype>
inline std::vector<arma::Cube<dtype>> qd_mex_reinterpret_CubeVec(const mxArray *input, const std::string &var_name = "")
{
    auto [d1, d2, d3, d4, n_data] = qd_mex_get_dims(input, var_name); // Rejects complex, sparse and 5-D input
    if (n_data == 0)
        return {};

    dtype *data = (dtype *)mxGetData(input);
    const size_t stride = d1 * d2 * d3;
    std::vector<arma::Cube<dtype>> output;
    output.reserve(d4); // prevent later reallocation of aliasing cubes
    for (size_t i = 0; i < d4; ++i)
        output.emplace_back(data + i * stride, d1, d2, d3, false, true);

    return output;
}

// Converts input to <std::any>
// - Contained types: scalar native c++, arma::Col, arma::Mat (also for a row vector), arma::Cube
// - Inputs with more than 3 dimensions are rejected (a cube is the largest type)
// - Empty inputs read to empty outputs
// - Input datatypes are maintained (double, single, int32, uint32, int64, uint64)
// - Does not copy data
// - See also: quadriga_lib::any_type_id
inline std::any qd_mex_anycast(const mxArray *input, const std::string &var_name = "")
{
    qd_mex_check_input(input, var_name, 3); // Rejects complex and sparse input, and more than 3 dimensions

    const size_t n_dim = (size_t)mxGetNumberOfDimensions(input);
    const mwSize *dims = mxGetDimensions(input);
    const size_t d1 = (size_t)dims[0], d2 = (size_t)dims[1], d3 = n_dim < 3 ? 1 : (size_t)dims[2];
    const size_t n_data = d1 * d2 * d3;

    if (n_data == 0)
        return std::any();

    auto wrap = [&](auto *ptr) -> std::any
    {
        using dtype = std::remove_pointer_t<decltype(ptr)>;
        if (n_data == 1)
            return *ptr;
        if (d2 == 1 && d3 == 1)
            return arma::Col<dtype>(ptr, n_data, false, true);
        if (d1 == 1 && d3 == 1)
            return arma::Mat<dtype>(ptr, 1, n_data, false, true);
        if (d3 != 1)
            return arma::Cube<dtype>(ptr, d1, d2, d3, false, true);
        return arma::Mat<dtype>(ptr, d1, d2, false, true);
    };

    switch (mxGetClassID(input)) // mxGetData only for numeric classes (Octave raises an error for others)
    {
    case mxDOUBLE_CLASS:
        return wrap((double *)mxGetData(input));
    case mxSINGLE_CLASS:
        return wrap((float *)mxGetData(input));
    case mxUINT32_CLASS:
        return wrap((unsigned *)mxGetData(input));
    case mxINT32_CLASS:
        return wrap((int *)mxGetData(input));
    case mxUINT64_CLASS:
        return wrap((unsigned long long *)mxGetData(input));
    case mxINT64_CLASS:
        return wrap((long long *)mxGetData(input));
    default:
        break;
    }

    const std::string msg = var_name.empty() ? "Unsupported data type." : "Input '" + var_name + "' has an unsupported data type.";
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    return std::any();
}

// Copy-convert MATLAB array data into a pre-allocated C++ buffer.
// Dispatches on the input's MATLAB type; casts each element to `dtype`. For integer types, NaN, Inf, fractional and
// out-of-range values raise an error after the copy.
template <typename dtype>
inline void qd_mex_copy_convert(const mxArray *input, dtype *ptr, size_t n_data,
                                const std::string &var_name = "")
{
    bool bad = false; // Set when a value does not fit an integer dtype (NaN, Inf, fraction, out of range)
    auto copy = [&](auto *src)
    {
        bool b = false;
        for (size_t m = 0; m < n_data; ++m)
            ptr[m] = qd_mex_cast<dtype>(src[m], b);
        bad = b;
    };
    qd_mex_visit_data(input, var_name, copy);
    if (bad)
        qd_mex_integer_error<dtype>(var_name, false);
}

// Reads input and converts it to desired C++ type, creates a copy of the input
template <typename dtype>
inline arma::Col<dtype> qd_mex_typecast_Col(const mxArray *input, std::string var_name = "", size_t n_elem = 0)
{
    auto [d1, d2, d3, d4, n_data] = qd_mex_get_dims(input, var_name);
    if (n_data == 0)
        return arma::Col<dtype>();
    if (n_elem != 0 && n_data != n_elem)
    {
        std::string msg = var_name.empty() ? "Incorrect number of elements." : "Input '" + var_name + "' has incorrect number of elements.";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    }
    arma::Col<dtype> output(n_data, arma::fill::none);
    qd_mex_copy_convert<dtype>(input, output.memptr(), n_data, var_name);
    return output;
}

template <typename dtype>
inline arma::Mat<dtype> qd_mex_typecast_Mat(const mxArray *input, std::string var_name = "")
{
    auto [d1, d2, d3, d4, n_data] = qd_mex_get_dims(input, var_name);
    if (n_data == 0)
        return arma::Mat<dtype>(d1, d2 * d3 * d4);
    arma::Mat<dtype> output(d1, d2 * d3 * d4, arma::fill::none);
    qd_mex_copy_convert<dtype>(input, output.memptr(), n_data, var_name);
    return output;
}

template <typename dtype>
inline arma::Cube<dtype> qd_mex_typecast_Cube(const mxArray *input, std::string var_name = "")
{
    auto [d1, d2, d3, d4, n_data] = qd_mex_get_dims(input, var_name);
    if (n_data == 0)
        return arma::Cube<dtype>(d1, d2, d3 * d4);
    arma::Cube<dtype> output(d1, d2, d3 * d4, arma::fill::none);
    qd_mex_copy_convert<dtype>(input, output.memptr(), n_data, var_name);
    return output;
}

template <typename dtype>
inline std::vector<arma::Cube<dtype>> qd_mex_typecast_CubeVec(const mxArray *input, std::string var_name = "")
{
    const std::array<size_t, 5> dims = qd_mex_get_dims(input, var_name); // Not structured bindings: C++17 lambdas cannot capture those
    const size_t d1 = dims[0], d2 = dims[1], d3 = dims[2], d4 = dims[3];
    if (dims[4] == 0)
        return {};

    std::vector<arma::Cube<dtype>> output;
    output.reserve(d4);
    for (size_t i = 0; i < d4; ++i)
        output.emplace_back(d1, d2, d3, arma::fill::none);

    const size_t stride = d1 * d2 * d3;
    bool bad = false; // Set when a value does not fit an integer dtype (NaN, Inf, fraction, out of range)
    auto copy = [&](auto *src)
    {
        bool b = false;
        for (size_t i = 0; i < d4; ++i)
        {
            dtype *dst = output[i].memptr();
            const auto *s = src + i * stride;
            for (size_t m = 0; m < stride; ++m)
                dst[m] = qd_mex_cast<dtype>(s[m], b);
        }
        bad = b;
    };

    qd_mex_visit_data(input, var_name, copy);
    if (bad)
        qd_mex_integer_error<dtype>(var_name, false);

    return output;
}

// Quick input converters; var_name is used in error messages
template <typename dtype>
inline arma::Col<dtype> qd_mex_get_Col(const mxArray *input, bool copy = false, const std::string &var_name = "")
{
    return (qd_mex_type_matches<dtype>(input) && !copy)
               ? qd_mex_reinterpret_Col<dtype>(input, var_name)
               : qd_mex_typecast_Col<dtype>(input, var_name);
}

template <typename dtype>
inline arma::Mat<dtype> qd_mex_get_Mat(const mxArray *input, bool copy = false, const std::string &var_name = "")
{
    return (qd_mex_type_matches<dtype>(input) && !copy)
               ? qd_mex_reinterpret_Mat<dtype>(input, var_name)
               : qd_mex_typecast_Mat<dtype>(input, var_name);
}

template <typename dtype>
inline arma::Cube<dtype> qd_mex_get_Cube(const mxArray *input, bool copy = false, const std::string &var_name = "")
{
    return (qd_mex_type_matches<dtype>(input) && !copy)
               ? qd_mex_reinterpret_Cube<dtype>(input, var_name)
               : qd_mex_typecast_Cube<dtype>(input, var_name);
}

template <typename dtype>
inline std::vector<arma::Cube<dtype>> qd_mex_get_CubeVec(const mxArray *input, bool copy = false, const std::string &var_name = "")
{
    return (qd_mex_type_matches<dtype>(input) && !copy)
               ? qd_mex_reinterpret_CubeVec<dtype>(input, var_name)
               : qd_mex_typecast_CubeVec<dtype>(input, var_name);
}

// Complex input split into real and imaginary part, as a copy; for C++ functions that take _re / _im
// - Real input of any numeric class is read as complex with imaginary part 0 (MATLAB and Octave store a complex value
//   with imaginary part 0 as a real array, e.g. 3+0i); complex input must be single or double
// - Rejects sparse input and input with more than 4 dimensions; empty input gives empty objects with the dimensions
//   of the input; the Mat folds dimensions 2 to 4 into its columns, the Cube dimensions 3 and 4 into its slices
// Calls f with a typed pointer to the data of a complex single or double array; other classes raise "Input 'x' has an
// unsupported data type."
template <typename F>
inline void qd_mex_visit_complex(const mxArray *input, const std::string &var_name, F &&f)
{
    if (mxIsDouble(input))
        return f((const std::complex<double> *)mxGetData(input));
    if (mxIsSingle(input))
        return f((const std::complex<float> *)mxGetData(input));
    const std::string msg = var_name.empty() ? "Unsupported data type." : "Input '" + var_name + "' has an unsupported data type.";
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
}

template <typename dtype>
inline void qd_mex_split_complex(const mxArray *input, const std::string &var_name, size_t n_data, dtype *re, dtype *im)
{
    if (n_data == 0)
        return;
    if (!mxIsComplex(input))
    {
        qd_mex_copy_convert<dtype>(input, re, n_data, var_name);
        std::memset(im, 0, sizeof(dtype) * n_data);
        return;
    }
    qd_mex_visit_complex(input, var_name, [&](const auto *src)
                         {
        for (size_t i = 0; i < n_data; ++i)
            re[i] = (dtype)src[i].real(), im[i] = (dtype)src[i].imag(); });
}

template <typename dtype>
inline void qd_mex_get_complex_Col(const mxArray *input, const std::string &var_name, arma::Col<dtype> &re, arma::Col<dtype> &im)
{
    auto [d1, d2, d3, d4, n_data] = qd_mex_get_dims(input, var_name, true);
    re.set_size(n_data), im.set_size(n_data);
    qd_mex_split_complex<dtype>(input, var_name, n_data, re.memptr(), im.memptr());
}

template <typename dtype>
inline void qd_mex_get_complex_Mat(const mxArray *input, const std::string &var_name, arma::Mat<dtype> &re, arma::Mat<dtype> &im)
{
    auto [d1, d2, d3, d4, n_data] = qd_mex_get_dims(input, var_name, true);
    re.set_size(d1, d2 * d3 * d4), im.set_size(d1, d2 * d3 * d4);
    qd_mex_split_complex<dtype>(input, var_name, n_data, re.memptr(), im.memptr());
}

template <typename dtype>
inline void qd_mex_get_complex_Cube(const mxArray *input, const std::string &var_name, arma::Cube<dtype> &re, arma::Cube<dtype> &im)
{
    auto [d1, d2, d3, d4, n_data] = qd_mex_get_dims(input, var_name, true);
    re.set_size(d1, d2, d3 * d4), im.set_size(d1, d2, d3 * d4);
    qd_mex_split_complex<dtype>(input, var_name, n_data, re.memptr(), im.memptr());
}

// One cube per index of the 4th dimension, as qd_mex_get_CubeVec; empty input gives empty vectors
template <typename dtype>
inline void qd_mex_get_complex_CubeVec(const mxArray *input, const std::string &var_name,
                                       std::vector<arma::Cube<dtype>> &re, std::vector<arma::Cube<dtype>> &im)
{
    const std::array<size_t, 5> dims = qd_mex_get_dims(input, var_name, true); // Not structured bindings: C++17 lambdas cannot capture those
    const size_t d1 = dims[0], d2 = dims[1], d3 = dims[2], d4 = dims[3], stride = d1 * d2 * d3;
    re.clear(), im.clear();
    if (dims[4] == 0)
        return;
    if (!mxIsComplex(input))
    {
        re = qd_mex_typecast_CubeVec<dtype>(input, var_name);
        for (const auto &r : re)
            im.emplace_back(arma::size(r), arma::fill::zeros);
        return;
    }
    re.reserve(d4), im.reserve(d4);
    for (size_t i = 0; i < d4; ++i)
        re.emplace_back(d1, d2, d3, arma::fill::none), im.emplace_back(d1, d2, d3, arma::fill::none);
    qd_mex_visit_complex(input, var_name, [&](const auto *src)
                         {
        for (size_t i = 0; i < d4; ++i)
        {
            dtype *pr = re[i].memptr(), *pi = im[i].memptr();
            for (size_t m = 0; m < stride; ++m)
                pr[m] = (dtype)src[i * stride + m].real(), pi[m] = (dtype)src[i * stride + m].imag();
        } });
}

// Read a vector of 1-based indices (MATLAB) and return them 0-based (C++), as a copy
// - Returns an empty vector for empty input
// - A 0 raises "Entries in 'x' cannot be 0 (1-based index)."; NaN, Inf, fractional, negative or too large values
//   raise "Entries in 'x' must be integers from 1 to <max>."
template <typename dtype = arma::uword>
inline arma::Col<dtype> qd_mex_get_index_Col(const mxArray *input, const std::string &var_name)
{
    static_assert(std::is_integral_v<dtype> && std::is_unsigned_v<dtype>, "qd_mex_get_index_Col: unsigned integer types only.");
    const size_t n_data = qd_mex_get_dims(input, var_name)[4]; // Not a structured binding: C++17 lambdas cannot capture those
    if (n_data == 0)
        return arma::Col<dtype>();

    arma::Col<dtype> output(n_data, arma::fill::none);
    dtype *ptr = output.memptr();
    bool bad = false, zero = false;
    qd_mex_visit_data(input, var_name, [&](auto *src)
                      {
                          for (size_t m = 0; m < n_data; ++m)
                          {
                              ptr[m] = qd_mex_cast<dtype>(src[m], bad);
                              zero |= ptr[m] == 0;
                              --ptr[m]; // Wraps for 0, which raises the error below
                          } });
    if (bad)
        qd_mex_integer_error<dtype>(var_name, false, true);
    if (zero)
    {
        const std::string msg = "Entries in '" + var_name + "' cannot be 0 (1-based index).";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    }
    return output;
}

// Read a MATLAB cell array of strings (or a single char string) into a std::vector<std::string>
// - Every cell element must be a char array; unset elements (cell(1,1)) and other types raise an error
inline std::vector<std::string> qd_mex_get_strings(const mxArray *input, const std::string &var_name = "")
{
    std::vector<std::string> out;
    if (input == nullptr || mxIsEmpty(input))
        return out;

    if (mxIsChar(input)) // single string -> one entry
    {
        out.push_back(qd_mex_get_string(input, "", var_name));
        return out;
    }

    if (!mxIsCell(input))
    {
        const std::string msg = var_name.empty() ? "Input must be a cell array of strings."
                                                 : "Input '" + var_name + "' must be a cell array of strings.";
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
    }

    size_t n = mxGetNumberOfElements(input);
    out.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        const mxArray *element = mxGetCell(input, (mwIndex)i); // NULL for unset elements in MATLAB
        if (element == nullptr || !mxIsChar(element))
        {
            const std::string msg = "Cell element " + std::to_string(i + 1) +
                                    (var_name.empty() ? std::string("") : " of '" + var_name + "'") + " is not a string.";
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
        }
        out.push_back(qd_mex_get_string(element, "", var_name));
    }

    return out;
}

// Creates an mxArray based on the armadillo input type, copies content
template <typename dtype>
inline mxArray *qd_mex_copy2matlab(const dtype *input) // Scalar
{
    const mxClassID classID = qd_mex_class_id<dtype>();

    mxArray *output = mxCreateNumericMatrix(1, 1, classID, qd_mex_complexity<dtype>());
    std::memcpy((dtype *)mxGetData(output), input, sizeof(dtype));
    return output;
}

template <typename dtype>
inline mxArray *qd_mex_copy2matlab(const arma::Row<dtype> *input) // Row Vector
{
    const mxClassID classID = qd_mex_class_id<dtype>();

    mxArray *output = mxCreateNumericMatrix(1, (mwSize)input->n_elem, classID, qd_mex_complexity<dtype>());
    std::memcpy((dtype *)mxGetData(output), input->memptr(), sizeof(dtype) * input->n_elem);
    return output;
}

template <typename dtype>
inline mxArray *qd_mex_copy2matlab(const arma::Col<dtype> *input, // Column Vector
                                   bool transpose = false)        // Transpose output
{
    const mxClassID classID = qd_mex_class_id<dtype>();

    if (input->empty())
        return mxCreateNumericMatrix(0, 0, classID, qd_mex_complexity<dtype>());

    const mwSize n = (mwSize)input->n_elem;
    mxArray *output = transpose ? mxCreateNumericMatrix(1, n, classID, qd_mex_complexity<dtype>()) : mxCreateNumericMatrix(n, 1, classID, qd_mex_complexity<dtype>());
    std::memcpy((dtype *)mxGetData(output), input->memptr(), sizeof(dtype) * input->n_elem);
    return output;
}

template <typename dtype>
inline mxArray *qd_mex_copy2matlab(const arma::Mat<dtype> *input) // Matrix
{
    const mxClassID classID = qd_mex_class_id<dtype>();

    if (input->empty())
        return mxCreateNumericMatrix(0, 0, classID, qd_mex_complexity<dtype>());

    mxArray *output = mxCreateNumericMatrix((mwSize)input->n_rows, (mwSize)input->n_cols, classID, qd_mex_complexity<dtype>());
    std::memcpy((dtype *)mxGetData(output), input->memptr(), sizeof(dtype) * input->n_elem);
    return output;
}

template <typename dtype>
inline mxArray *qd_mex_copy2matlab(const arma::Cube<dtype> *input) // Cube
{
    const mxClassID classID = qd_mex_class_id<dtype>();

    if (input->empty())
        return mxCreateNumericMatrix(0, 0, classID, qd_mex_complexity<dtype>());

    mwSize dims[3] = {(mwSize)input->n_rows, (mwSize)input->n_cols, (mwSize)input->n_slices};
    mxArray *output = mxCreateNumericArray(3, dims, classID, qd_mex_complexity<dtype>());
    std::memcpy((dtype *)mxGetData(output), input->memptr(), sizeof(dtype) * input->n_elem);
    return output;
}

// Copies an arma::Col, Mat or Cube like qd_mex_copy2matlab, but an empty input keeps its dimensions (e.g. a Col
// [0, 1], a Mat [3, 0], a Cube [n_rx, n_tx, 0]) instead of becoming 0x0; for outputs whose shape is documented
template <typename T>
inline mxArray *qd_mex_copy2matlab_keep_shape(const T &x)
{
    using dtype = typename T::elem_type;
    if (!x.is_empty())
        return qd_mex_copy2matlab(&x);
    if constexpr (std::is_same_v<T, arma::Cube<dtype>>)
    {
        mwSize dims[3] = {(mwSize)x.n_rows, (mwSize)x.n_cols, (mwSize)x.n_slices};
        return mxCreateNumericArray(3, dims, qd_mex_class_id<dtype>(), qd_mex_complexity<dtype>());
    }
    else
        return mxCreateNumericMatrix((mwSize)x.n_rows, (mwSize)x.n_cols, qd_mex_class_id<dtype>(), qd_mex_complexity<dtype>());
}

inline mxArray *qd_mex_copy2matlab(const std::vector<std::string> *strings)
{
    size_t n_obj = strings->size();
    auto *cellArray = mxCreateCellMatrix((mwSize)n_obj, 1);
    for (size_t i = 0; i < n_obj; ++i)
    {
        auto *mxStr = mxCreateString(strings->at(i).c_str());
        mxSetCell(cellArray, i, mxStr);
    }
    return cellArray;
}

inline mxArray *qd_mex_copy2matlab(const std::vector<bool> *bools)
{
    if (!bools)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Null std::vector<bool> pointer.");

    const size_t n_obj = bools->size();

    if (n_obj == 0)
        return mxCreateLogicalMatrix(0, 0);

    mxArray *out = mxCreateLogicalMatrix((mwSize)n_obj, 1);
    mxLogical *dst = mxGetLogicals(out);

    for (size_t i = 0; i < n_obj; ++i)
        dst[i] = (*bools)[i] ? 1 : 0;

    return out;
}

// Copies a vector of Armadillo objects to an mxArray with one more dimension (e.g. arma::Cube -> MATLAB 4D); entries
// smaller than the largest one are zero-padded; an empty vector gives a 0x0 matrix
template <typename dtype>
inline mxArray *qd_mex_vector2matlab(const std::vector<arma::Col<dtype>> *input)
{
    const mxClassID classID = qd_mex_class_id<dtype>();

    if (input->empty())
        return mxCreateNumericMatrix(0, 0, classID, qd_mex_complexity<dtype>());

    size_t m = 0ULL;
    for (auto &v : *input)
        m = v.n_rows > m ? v.n_rows : m;
    const size_t ns = input->size();

    mxArray *output = mxCreateNumericMatrix((mwSize)m, (mwSize)ns, classID, qd_mex_complexity<dtype>());
    dtype *ptr = (dtype *)mxGetData(output);
    for (size_t i = 0ULL; i < ns; ++i)
        std::memcpy(&ptr[i * m], input->at(i).memptr(), sizeof(dtype) * input->at(i).n_rows);

    return output;
}

template <typename dtype>
inline mxArray *qd_mex_vector2matlab(const std::vector<arma::Mat<dtype>> *input)
{
    const mxClassID classID = qd_mex_class_id<dtype>();

    if (input->empty())
        return mxCreateNumericMatrix(0, 0, classID, qd_mex_complexity<dtype>());

    size_t n_rows = 0ULL, n_cols = 0ULL;
    for (auto &v : *input)
        n_rows = v.n_rows > n_rows ? v.n_rows : n_rows,
        n_cols = v.n_cols > n_cols ? v.n_cols : n_cols;
    const size_t m = n_rows * n_cols, ns = input->size();

    mwSize dims[3] = {(mwSize)n_rows, (mwSize)n_cols, (mwSize)ns};
    mxArray *output = mxCreateNumericArray(3, dims, classID, qd_mex_complexity<dtype>());
    dtype *ptr = (dtype *)mxGetData(output);

    for (size_t i = 0ULL; i < ns; ++i)
    {
        const arma::Mat<dtype> &v = input->at(i);
        if (v.n_rows == n_rows)
            std::memcpy(&ptr[i * m], v.memptr(), sizeof(dtype) * v.n_elem);
        else // Copy column by column
            for (size_t ic = 0ULL; ic < v.n_cols; ++ic)
                std::memcpy(&ptr[i * m + ic * n_rows], v.colptr(ic), sizeof(dtype) * v.n_rows);
    }

    return output;
}

template <typename dtype>
inline mxArray *qd_mex_vector2matlab(const std::vector<arma::Cube<dtype>> *input)
{
    const mxClassID classID = qd_mex_class_id<dtype>();

    if (input->empty())
        return mxCreateNumericMatrix(0, 0, classID, qd_mex_complexity<dtype>());

    size_t n_rows = 0ULL, n_cols = 0ULL, n_slices = 0ULL;
    for (auto &v : *input)
        n_rows = v.n_rows > n_rows ? v.n_rows : n_rows,
        n_cols = v.n_cols > n_cols ? v.n_cols : n_cols,
        n_slices = v.n_slices > n_slices ? v.n_slices : n_slices;
    const size_t m = n_rows * n_cols * n_slices, ns = input->size();

    mwSize dims[4] = {(mwSize)n_rows, (mwSize)n_cols, (mwSize)n_slices, (mwSize)ns};
    mxArray *output = mxCreateNumericArray(4, dims, classID, qd_mex_complexity<dtype>());
    dtype *ptr = (dtype *)mxGetData(output);

    for (size_t i = 0ULL; i < ns; ++i)
    {
        const arma::Cube<dtype> &v = input->at(i);
        if (v.n_rows == n_rows && v.n_cols == n_cols)
            std::memcpy(&ptr[i * m], v.memptr(), sizeof(dtype) * v.n_elem);
        else // Copy column by column
            for (size_t is = 0ULL; is < v.n_slices; ++is)
                for (size_t ic = 0ULL; ic < v.n_cols; ++ic)
                    std::memcpy(&ptr[i * m + is * n_rows * n_cols + ic * n_rows], v.slice_colptr(is, ic), sizeof(dtype) * v.n_rows);
    }

    return output;
}

// Complex output from the real and imaginary part of a C++ result (copy), with the shape of 're' (a Col gives
// [n, 1]); an empty 'im' gives imaginary part 0; an empty 're' gives a 0x0 array, or with keep_shape its dimensions.
// The output helpers above also take complex objects directly (e.g. arma::cx_mat, std::vector<arma::cx_cube>); with
// an optional std::complex output of the C++ function, qd_mex_init_output lets C++ write the complex result in place.
template <typename T>
inline mxArray *qd_mex_copy2matlab(const T *re, const T *im, bool keep_shape = false)
{
    using dtype = typename T::elem_type;
    if (!im->is_empty() && im->n_elem != re->n_elem)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Real and imaginary part of a complex output differ in size.");
    mwSize dims[3] = {(mwSize)re->n_rows, (mwSize)re->n_cols, 1}, n_dims = 2;
    if constexpr (arma::is_Cube<T>::value)
        dims[2] = (mwSize)re->n_slices, n_dims = 3;
    if (re->is_empty() && !keep_shape)
        dims[0] = dims[1] = 0, n_dims = 2;
    mxArray *output = mxCreateNumericArray(n_dims, dims, qd_mex_class_id<dtype>(), mxCOMPLEX);
    std::complex<dtype> *ptr = (std::complex<dtype> *)mxGetData(output);
    const dtype *pr = re->memptr(), *pi = im->is_empty() ? nullptr : im->memptr();
    for (size_t i = 0; i < re->n_elem; ++i)
        ptr[i] = std::complex<dtype>(pr[i], pi ? pi[i] : dtype(0));
    return output;
}

// Complex 4D output from a vector of real and imaginary cubes, zero-padded as qd_mex_vector2matlab; an empty 'im'
// vector or entry gives imaginary part 0
template <typename dtype>
inline mxArray *qd_mex_vector2matlab(const std::vector<arma::Cube<dtype>> *re, const std::vector<arma::Cube<dtype>> *im)
{
    if (!im->empty() && im->size() != re->size())
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Real and imaginary part of a complex output differ in size.");
    if (re->empty())
        return mxCreateNumericMatrix(0, 0, qd_mex_class_id<dtype>(), mxCOMPLEX);

    size_t n_rows = 0, n_cols = 0, n_slices = 0;
    for (const auto &v : *re)
        n_rows = std::max(n_rows, (size_t)v.n_rows), n_cols = std::max(n_cols, (size_t)v.n_cols),
        n_slices = std::max(n_slices, (size_t)v.n_slices);
    const size_t m = n_rows * n_cols * n_slices, ns = re->size();

    mwSize dims[4] = {(mwSize)n_rows, (mwSize)n_cols, (mwSize)n_slices, (mwSize)ns};
    mxArray *output = mxCreateNumericArray(4, dims, qd_mex_class_id<dtype>(), mxCOMPLEX); // Zero-initialized padding
    std::complex<dtype> *ptr = (std::complex<dtype> *)mxGetData(output);
    for (size_t i = 0; i < ns; ++i)
    {
        const arma::Cube<dtype> &r = re->at(i);
        const dtype *pi = (im->empty() || im->at(i).is_empty()) ? nullptr : im->at(i).memptr();
        if (pi && im->at(i).n_elem != r.n_elem)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Real and imaginary part of a complex output differ in size.");
        for (size_t s = 0, k = 0; s < r.n_slices; ++s)
            for (size_t c = 0; c < r.n_cols; ++c)
                for (size_t row = 0; row < r.n_rows; ++row, ++k)
                    ptr[i * m + (s * n_cols + c) * n_rows + row] = std::complex<dtype>(r[k], pi ? pi[k] : dtype(0));
    }
    return output;
}

// Creates a std::vector of armadillo objects from mxArray, one per index of dimension vec_dim (0-based); the data of
// the other three dimensions, in their order, fill the object made by make(n1, n2, n3)
// - Casts to <dtype>; reads all real classes (qd_mex_visit_data); integer dtypes need integer values in range
// - Rejects complex and sparse input, and input with more than 4 dimensions; var_name is used in error messages
template <typename dtype, typename obj, typename F>
std::vector<obj> qd_mex_matlab2vector(const mxArray *input, size_t vec_dim, const std::string &var_name, F &&make)
{
    auto [d1, d2, d3, d4, n_data] = qd_mex_get_dims(input, var_name);
    std::vector<obj> output;
    if (n_data == 0)
        return output;
    if (vec_dim > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Armadillo object dimensions must be 0,1,2 or 3");

    // Element (i, n, o) with i the index over the dimensions before vec_dim and o over those after it
    const size_t d[4] = {d1, d2, d3, d4}, dn = d[vec_dim];
    size_t r[3], inner = 1, outer = 1;
    for (size_t k = 0, j = 0; k < 4; ++k)
        if (k < vec_dim)
            r[j++] = d[k], inner *= d[k];
        else if (k > vec_dim)
            r[j++] = d[k], outer *= d[k];

    bool bad = false; // Set when a value does not fit an integer dtype (NaN, Inf, fraction, out of range)
    qd_mex_visit_data(input, var_name, [&](auto *src)
                      {
        output.reserve(dn);
        for (size_t n = 0; n < dn; ++n)
        {
            obj tmp = make(r[0], r[1], r[2]);
            dtype *ptr = tmp.memptr();
            for (size_t o = 0; o < outer; ++o)
                for (size_t i = 0; i < inner; ++i)
                    ptr[o * inner + i] = qd_mex_cast<dtype>(src[(o * dn + n) * inner + i], bad);
            output.push_back(std::move(tmp));
        } });
    if (bad)
        qd_mex_integer_error<dtype>(var_name, false);
    return output;
}

// MATLAB --> std::vector<arma::Col<dtype>>, see qd_mex_matlab2vector
template <typename dtype>
std::vector<arma::Col<dtype>> qd_mex_matlab2vector_Col(const mxArray *input, size_t vec_dim, const std::string &var_name = "")
{
    return qd_mex_matlab2vector<dtype, arma::Col<dtype>>(input, vec_dim, var_name, [](size_t a, size_t b, size_t c)
                                                         { return arma::Col<dtype>(a * b * c, arma::fill::none); });
}

// MATLAB --> std::vector<arma::Mat<dtype>> with n2 * n3 columns, see qd_mex_matlab2vector
template <typename dtype>
std::vector<arma::Mat<dtype>> qd_mex_matlab2vector_Mat(const mxArray *input, size_t vec_dim, const std::string &var_name = "")
{
    return qd_mex_matlab2vector<dtype, arma::Mat<dtype>>(input, vec_dim, var_name, [](size_t a, size_t b, size_t c)
                                                         { return arma::Mat<dtype>(a, b * c, arma::fill::none); });
}

// MATLAB --> std::vector<arma::Cube<dtype>>, see qd_mex_matlab2vector
template <typename dtype>
std::vector<arma::Cube<dtype>> qd_mex_matlab2vector_Cube(const mxArray *input, size_t vec_dim, const std::string &var_name = "")
{
    return qd_mex_matlab2vector<dtype, arma::Cube<dtype>>(input, vec_dim, var_name, [](size_t a, size_t b, size_t c)
                                                          { return arma::Cube<dtype>(a, b, c, arma::fill::none); });
}

// Converts input to a std::vector<bool> (non-zero = true); reads all real classes (qd_mex_visit_data); rejects complex
// and sparse input; any number of dimensions
inline std::vector<bool> qd_mex_matlab2vector_Bool(const mxArray *input, const std::string &var_name = "")
{
    qd_mex_check_input(input, var_name, (size_t)-1);
    const size_t n_data = (size_t)mxGetNumberOfElements(input);
    std::vector<bool> output;
    if (n_data == 0)
        return output;
    output.reserve(n_data);
    qd_mex_visit_data(input, var_name, [&](auto *src)
                      {
        for (size_t m = 0; m < n_data; ++m)
            output.push_back(src[m] != 0); });
    return output;
}

// Allocates a zero-initialized MATLAB output and maps the Armadillo object onto its memory
template <typename dtype>
inline mxArray *qd_mex_init_output(arma::Row<dtype> *input, size_t n_elem) // 1D Row Vector
{
    mxArray *output = mxCreateNumericMatrix(1, n_elem, qd_mex_class_id<dtype>(), qd_mex_complexity<dtype>());
    input->~Row();
    ::new (input) arma::Row<dtype>((dtype *)mxGetData(output), n_elem, false, true);
    return output;
}

template <typename dtype>
inline mxArray *qd_mex_init_output(arma::Col<dtype> *input, size_t n_elem, bool transpose = false) // 1D Column Vector
{
    mxArray *output = transpose ? mxCreateNumericMatrix(1, n_elem, qd_mex_class_id<dtype>(), qd_mex_complexity<dtype>())
                                : mxCreateNumericMatrix(n_elem, 1, qd_mex_class_id<dtype>(), qd_mex_complexity<dtype>());
    input->~Col();
    ::new (input) arma::Col<dtype>((dtype *)mxGetData(output), n_elem, false, true);
    return output;
}

// Output with the dimensions of the MATLAB array "like" (any number of dimensions), mapped as a flat column vector
template <typename dtype>
inline mxArray *qd_mex_init_output_like(arma::Col<dtype> *input, const mxArray *like)
{
    mxArray *output = mxCreateNumericArray(mxGetNumberOfDimensions(like), mxGetDimensions(like), qd_mex_class_id<dtype>(), qd_mex_complexity<dtype>());
    input->~Col();
    ::new (input) arma::Col<dtype>((dtype *)mxGetData(output), mxGetNumberOfElements(like), false, true);
    return output;
}

// True if two MATLAB arrays have the same dimensions
inline bool qd_mex_same_dims(const mxArray *a, const mxArray *b)
{
    const mwSize n_dim = mxGetNumberOfDimensions(a);
    if (mxGetNumberOfDimensions(b) != n_dim)
        return false;
    const mwSize *da = mxGetDimensions(a), *db = mxGetDimensions(b);
    for (mwSize i = 0; i < n_dim; ++i)
        if (da[i] != db[i])
            return false;
    return true;
}

template <typename dtype>
inline mxArray *qd_mex_init_output(arma::Mat<dtype> *input, size_t n_rows, size_t n_cols) // 2D
{
    mxArray *output = mxCreateNumericMatrix(n_rows, n_cols, qd_mex_class_id<dtype>(), qd_mex_complexity<dtype>());
    input->~Mat();
    ::new (input) arma::Mat<dtype>((dtype *)mxGetData(output), n_rows, n_cols, false, true);
    return output;
}

template <typename dtype>
inline mxArray *qd_mex_init_output(arma::Cube<dtype> *input, size_t n_rows, size_t n_cols, size_t n_slices) // 3D
{
    mwSize dims[3] = {(mwSize)n_rows, (mwSize)n_cols, (mwSize)n_slices};
    mxArray *output = mxCreateNumericArray(3, dims, qd_mex_class_id<dtype>(), qd_mex_complexity<dtype>());
    input->~Cube();
    ::new (input) arma::Cube<dtype>((dtype *)mxGetData(output), n_rows, n_cols, n_slices, false, true);
    return output;
}

template <typename dtype>
inline mxArray *qd_mex_init_output(std::vector<arma::Cube<dtype>> *input,
                                   size_t n_rows, size_t n_cols, size_t n_slices, size_t n_cubes) // 4D
{
    mwSize dims[4] = {(mwSize)n_rows, (mwSize)n_cols, (mwSize)n_slices, (mwSize)n_cubes};
    mxArray *output = mxCreateNumericArray(4, dims, qd_mex_class_id<dtype>(), qd_mex_complexity<dtype>());
    dtype *data = (dtype *)mxGetData(output);
    const size_t stride = n_rows * n_cols * n_slices;

    input->clear();
    input->reserve(n_cubes); // critical: avoid any later reallocation
    for (size_t i = 0; i < n_cubes; ++i)
        input->emplace_back(data + i * stride, n_rows, n_cols, n_slices, false, true);

    return output;
}

// 1 x N struct array with the given (unset) fields; a 0x0 struct without fields for an empty field list
inline mxArray *qd_mex_make_struct(const std::vector<std::string> &fields, size_t N = 1)
{
    mxArray *output;
    if (fields.empty())
    {
        mwSize dims[2] = {0, 0};
        output = mxCreateStructArray(2, dims, 0, NULL);
    }
    else
    {
        std::vector<const char *> field_names;
        for (const auto &str : fields)
            field_names.push_back(str.c_str());

        mwSize dims[2] = {1, (mwSize)N}; // Creates a 1xN struct array
        output = mxCreateStructArray(2, dims, (int)field_names.size(), field_names.data());
    }
    return output;
}

inline void qd_mex_set_field(mxArray *strct, const std::string &field, mxArray *data, size_t n = 0)
{
    mxSetField(strct, n, field.c_str(), data);
}

// Raises "Input 'x' must be a struct." ("Input must be a struct." for an empty var_name) unless input is a struct;
// one: the struct must have exactly one element ("... must be a 1x1 struct.")
inline void qd_mex_check_struct(const mxArray *input, const std::string &var_name = "", bool one = false)
{
    if (mxIsStruct(input) && (!one || mxGetNumberOfElements(input) == 1))
        return;
    const std::string msg = (var_name.empty() ? std::string("Input") : "Input '" + var_name + "'") +
                            (one ? " must be a 1x1 struct." : " must be a struct.");
    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", msg.c_str());
}

inline bool qd_mex_has_field(const mxArray *strct, const std::string &field)
{
    qd_mex_check_struct(strct);
    return mxGetField(strct, 0, field.c_str()) != nullptr;
}

inline mxArray *qd_mex_get_field(const mxArray *strct, const std::string &field, size_t n = 0, const std::string &var_name = "")
{
    const std::string in = var_name.empty() ? std::string("") : " in '" + var_name + "'";
    qd_mex_check_struct(strct, var_name);

    size_t n_elem = (size_t)mxGetNumberOfElements(strct);
    if (n >= n_elem)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Struct index " + std::to_string(n + 1) + in + " out of range (size " + std::to_string(n_elem) + ").").c_str());

    mxArray *data = mxGetField(strct, (mwIndex)n, field.c_str());
    if (data == nullptr)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Field '" + field + "' not found" + in + " at index " + std::to_string(n + 1) + ".").c_str());
    return data;
}

// Convert std::unordered_map<std::string, std::vector<dtype>> to a 1x1 MATLAB struct.
// Each map key becomes a field; each value vector becomes a column vector [n_mtl, 1].
// Field order follows 'field_order' when given (e.g. the CSV column order); otherwise the
// map's iteration order is used. An empty map yields an empty struct.
template <typename dtype>
inline mxArray *qd_mex_map2struct(const std::unordered_map<std::string, std::vector<dtype>> &map,
                                  const std::vector<std::string> &field_order = {})
{
    // Decide field order: explicit list (filtered to keys present), else map iteration order
    std::vector<std::string> fields;
    fields.reserve(map.size());
    if (!field_order.empty())
    {
        for (const auto &k : field_order)
            if (map.find(k) != map.end())
                fields.push_back(k);
    }
    else
        for (const auto &kv : map)
            fields.push_back(kv.first);

    mxClassID classID = qd_mex_class_id<dtype>();
    mxArray *output = qd_mex_make_struct(fields, 1); // 1x1 struct, one field per key

    for (const auto &key : fields)
    {
        const std::vector<dtype> &vec = map.at(key);
        mxArray *col = mxCreateNumericMatrix((mwSize)vec.size(), vec.empty() ? 0 : 1, classID, qd_mex_complexity<dtype>());
        if (!vec.empty())
            std::memcpy((dtype *)mxGetData(col), vec.data(), sizeof(dtype) * vec.size());
        qd_mex_set_field(output, key, col);
    }

    return output;
}

// Convert a 1x1 MATLAB struct to std::unordered_map<std::string, std::vector<dtype>>.
// Each field becomes a key; the field data (any numeric class) is read as a flat vector
// and cast to dtype. All fields must have the same number of elements (n_mtl); a mismatch
// throws. A non-struct input or an empty struct yields an empty map.
template <typename dtype>
inline std::unordered_map<std::string, std::vector<dtype>> qd_mex_struct2map(const mxArray *strct)
{
    std::unordered_map<std::string, std::vector<dtype>> map;
    if (strct == nullptr || !mxIsStruct(strct))
        return map;

    int n_fields = mxGetNumberOfFields(strct);
    if (n_fields == 0)
        return map;

    size_t n_mtl = 0;
    bool n_mtl_set = false;
    for (int f = 0; f < n_fields; ++f)
    {
        const char *name = mxGetFieldNameByNumber(strct, f);
        const mxArray *data = mxGetFieldByNumber(strct, 0, f);
        if (data == nullptr)
            continue;

        auto [d1, d2, d3, d4, n_data] = qd_mex_get_dims(data, name);

        if (!n_mtl_set)
            n_mtl = n_data, n_mtl_set = true;
        else if (n_data != n_mtl)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", ("Material property field '" + std::string(name) +
                                     "' has " + std::to_string(n_data) + " elements, expected " +
                                     std::to_string(n_mtl) + " (all fields must have the same length).").c_str());

        std::vector<dtype> vec(n_data);
        if (n_data != 0)
            qd_mex_copy_convert<dtype>(data, vec.data(), n_data, name);
        map[std::string(name)] = std::move(vec);
    }

    return map;
}

// Batch inputs: reads the entries of a cell array with 'get(entry, var_name)'; unset entries stay empty (no data).
// 'error' is the message when 'input' is not a cell array.
template <typename obj, typename F>
inline std::vector<obj> qd_mex_read_cell(const mxArray *input, const char *var_name, const char *error, F &&get)
{
    if (!mxIsCell(input))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "%s", error);

    const size_t n = mxGetNumberOfElements(input);
    std::vector<obj> out(n);
    for (size_t i = 0; i < n; ++i)
    {
        const mxArray *element = mxGetCell(input, (mwIndex)i); // NULL for unset elements
        if (element != nullptr)
            out[i] = get(element, var_name);
    }
    return out;
}

// Batch positions and orientations: a 3-element vector (row or column) applies to all snapshots, else [3, n_snap]
inline arma::mat qd_mex_get_xyz_batch(const mxArray *input, const char *var_name)
{
    return mxGetNumberOfElements(input) == 3 ? arma::mat(qd_mex_typecast_Col<double>(input, var_name, 3))
                                             : qd_mex_get_Mat<double>(input, false, var_name);
}

#endif
