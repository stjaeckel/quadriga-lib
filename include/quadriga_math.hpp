// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef quadriga_math_H
#define quadriga_math_H

#include <armadillo>

// If arma::uword and size_t are not the same width (e.g. 64 bit), the compiler will throw an error here
// This allows the use of "uword", "size_t" and "unsigned long long" interchangeably
// This requires a 64 bit platform, but will compile on Linux, Windows and macOS
static_assert(sizeof(arma::uword) == sizeof(unsigned long long), "arma::uword and unsigned long long have different sizes");
static_assert(sizeof(size_t) == sizeof(unsigned long long), "size_t and unsigned long long have different sizes");

namespace quadriga_lib
{
    // Rotation matrix from Euler angles
    // - A matrix of orientations [3, n] is the cube [3, n, 1]: arma::Cube<dtype>(const_cast<dtype *>(mat.memptr()), 3, n, 1, false, true)
    template <typename dtype>
    void calc_rotation_matrix(const arma::Cube<dtype> &orientation, // Orientation vectors (Euler rotations) in [rad], Size [3, n_row, n_col]
                              arma::Cube<dtype> &rotation,          // Output: Rotation matrices, column-major, Size [9, n_row, n_col]
                              bool transposed = false);             // Returns the transpose of R instead of R

    // Fast, approximate sine/cosine for single-precision vectors
    template <typename dtype>
    void fast_sincos(const arma::Col<dtype> &x,     // Input angles in radians; n = x.n_elem, float or double
                     arma::fvec *sin_x = nullptr,   // [out] If non-null, set to sin(x). Resized to length n if needed
                     arma::fvec *cos_x = nullptr,   // [out] If non-null, set to cos(x). Resized to length n if needed
                     int use_kernel = 0);           // Kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2

    // Fast, approximate arc-sine for single-precision vectors
    template <typename dtype>
    void fast_asin(const arma::Col<dtype> &x, // Input values in [-1, 1]; n = x.n_elem, float or double
                   arma::fvec &asin_x,        // [out] Set to asin(x). Resized to length n if needed
                   int use_kernel = 0);       // Kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2

    // Fast, approximate arc-cosine for single-precision vectors
    template <typename dtype>
    void fast_acos(const arma::Col<dtype> &x, // Input values in [-1, 1]; n = x.n_elem, float or double
                   arma::fvec &acos_x,        // [out] Set to acos(x). Resized to length n if needed
                   int use_kernel = 0);       // Kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2

    // Fast, approximate atan2(y, x) for single-precision vectors
    template <typename dtype>
    void fast_atan2(const arma::Col<dtype> &y, // Input y-coordinates, Length [n], float or double
                    const arma::Col<dtype> &x, // Input x-coordinates, Length [n], float or double
                    arma::fvec &atan2_yx,      // [out] Set to atan2(y, x) in radians. Resized to length n if needed
                    int use_kernel = 0);       // Kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2

    // Fast, approximate spherical interpolation (SLERP) for complex value pairs
    template <typename dtype>
    void fast_slerp(const arma::Col<dtype> &A_re, // Real part of source A, Length [n]
                    const arma::Col<dtype> &A_im, // Imaginary part of source A, Length [n]
                    const arma::Col<dtype> &B_re, // Real part of source B, Length [n]
                    const arma::Col<dtype> &B_im, // Imaginary part of source B, Length [n]
                    const arma::Col<dtype> &w,    // Per-element interpolation weight in [0,1], 0 = A, 1 = B, Length [n]
                    arma::fvec &X_re,             // [out] Real part of interpolated result, Length [n]
                    arma::fvec &X_im,             // [out] Imaginary part of interpolated result, Length [n]
                    int use_kernel = 0);          // Kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2

    // Fast, approximate geographic-to-Cartesian conversion
    template <typename dtype>
    void fast_geo2cart(const arma::Col<dtype> &az,                       // Input azimuth angles in radians, Length [n]
                       const arma::Col<dtype> &el,                       // Input elevation angles in radians, Length [n]
                       arma::Col<dtype> &x,                              // Output x-coordinates, Length [n]
                       arma::Col<dtype> &y,                              // Output y-coordinates, Length [n]
                       arma::Col<dtype> &z,                              // Output z-coordinates, Length [n]
                       arma::Col<dtype> *sin_az = nullptr,               // Optional output: sin(az), Length [n]
                       arma::Col<dtype> *cos_az = nullptr,               // Optional output: cos(az), Length [n]
                       arma::Col<dtype> *sin_el = nullptr,               // Optional output: sin(el), Length [n]
                       arma::Col<dtype> *cos_el = nullptr,               // Optional output: cos(el), Length [n]
                       const arma::Col<dtype> &len = arma::Col<dtype>(), // Optional input vector length, Length [n]
                       int use_kernel = 0);                              // Kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2

    // Fast, approximate Cartesian-to-geographic conversion
    template <typename dtype>
    void fast_cart2geo(const arma::Col<dtype> &x,       // Input x-coordinates, Length [n]
                       const arma::Col<dtype> &y,       // Input y-coordinates, Length [n]
                       const arma::Col<dtype> &z,       // Input z-coordinates, Length [n]
                       arma::Col<dtype> &az,            // Output azimuth angles in radians, Length [n]
                       arma::Col<dtype> &el,            // Output elevation angles in radians, Length [n]
                       arma::Col<dtype> *len = nullptr, // Vector length, Length [n]
                       int use_kernel = 0);             // Kernel: 0 = auto (AVX2 if available), 1 = GENERIC, 2 = AVX2

    // 2D linear interpolation of multiple data sets
    template <typename dtype>
    void interp_2D(const arma::Col<dtype> &x,      // x sample points of input; vector length nx
                   const arma::Col<dtype> &y,      // y sample points of input; vector length ny
                   const arma::Cube<dtype> &input, // Input data; size [ ny, nx, ne ], ne = multiple data sets
                   const arma::Col<dtype> &x_new,  // x sample points of output; vector length nx_new
                   const arma::Col<dtype> &y_new,  // y sample points of output; vector length ny_new
                   arma::Cube<dtype> &output);     // Output: Interpolated data, size [ ny_new, nx_new, ne ]

    // 1D linear interpolation of multiple data sets
    template <typename dtype>
    void interp_1D(const arma::Col<dtype> &x,     // Input sample points, vector length nx
                   const arma::Mat<dtype> &input, // Input data; size [ nx, ne ], ne = multiple data sets
                   const arma::Col<dtype> &x_new, // Output sample points, vector length nx_new
                   arma::Mat<dtype> &output);     // Output: Interpolated data, size [ nx_new, ne ]

    // Calculate the empirical averaged cumulative distribution function (CDF)
    // Input data matrix has samples in rows and data sets in columns.
    // Individual CDFs are computed per column. An averaged CDF is obtained by
    // quantile-space averaging. Inf and NaN values are excluded.
    template <typename dtype>
    void acdf(const arma::Mat<dtype> &data,            // Input data, Size [n_samples, n_sets]
              arma::Col<dtype> *bins = nullptr,        // Bin centers (in/out), Length [n_bins]
              arma::Mat<dtype> *cdf_per_set = nullptr, // Individual CDFs, Size [n_bins, n_sets]
              arma::Col<dtype> *cdf_avg = nullptr,     // Averaged CDF, Length [n_bins]
              arma::Col<dtype> *mu = nullptr,          // Mean 0.1-0.9 quantiles, Length [9]
              arma::Col<dtype> *sig = nullptr,         // Std of 0.1-0.9 quantiles, Length [9]
              arma::uword n_bins = 201);               // Number of auto-generated bins
}

#endif