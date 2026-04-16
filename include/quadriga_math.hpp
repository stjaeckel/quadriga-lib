// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
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
    // Fast, approximate sine/cosine for single-precision vectors
    template <typename dtype>
    void fast_sincos(const arma::Col<dtype> &x, // Input angles in radians; n = x.n_elem, float or double
                     arma::fvec *s = nullptr,   // [out] If non-null, set to sin(x). Resized to length n if needed
                     arma::fvec *c = nullptr);  // [out] If non-null, set to cos(x). Resized to length n if needed

    // Fast, approximate arc-sine for single-precision vectors
    template <typename dtype>
    void fast_asin(const arma::Col<dtype> &x, // Input values in [-1, 1]; n = x.n_elem, float or double
                   arma::fvec &s);            // [out] Set to asin(x). Resized to length n if needed

    // Fast, approximate arc-cosine for single-precision vectors
    template <typename dtype>
    void fast_acos(const arma::Col<dtype> &x, // Input values in [-1, 1]; n = x.n_elem, float or double
                   arma::fvec &c);            // [out] Set to acos(x). Resized to length n if needed

    // Fast, approximate atan2(y, x) for single-precision vectors
    template <typename dtype>
    void fast_atan2(const arma::Col<dtype> &y, // Input y-coordinates, Length [n], float or double
                    const arma::Col<dtype> &x, // Input x-coordinates, Length [n], float or double
                    arma::fvec &a);            // [out] Set to atan2(y, x) in radians. Resized to length n if needed

    // Fast, approximate spherical interpolation (SLERP) for complex value pairs
    template <typename dtype>
    void fast_slerp(const arma::Col<dtype> &Ar, // Real part of source A, Length [n]
                    const arma::Col<dtype> &Ai, // Imaginary part of source A, Length [n]
                    const arma::Col<dtype> &Br, // Real part of source B, Length [n]
                    const arma::Col<dtype> &Bi, // Imaginary part of source B, Length [n]
                    const arma::Col<dtype> &w,  // Per-element interpolation weight in [0,1], 0 = A, 1 = B, Length [n]
                    arma::fvec &Xr,             // [out] Real part of interpolated result, Length [n]
                    arma::fvec &Xi);            // [out] Imaginary part of interpolated result, Length [n]

    // Fast, approximate geographic-to-Cartesian conversion
    template <typename dtype>
    void fast_geo2cart(const arma::Col<dtype> &az, // Input azimuth angles in radians, Length [n]
                       const arma::Col<dtype> &el, // Input elevation angles in radians, Length [n]
                       arma::fvec &x,              // Output x-coordinates, Length [n]
                       arma::fvec &y,              // Output y-coordinates, Length [n]
                       arma::fvec &z,              // Output z-coordinates, Length [n]
                       arma::fvec *sAZ = nullptr,  // Optional output: sin(az), Length [n]
                       arma::fvec *cAZ = nullptr,  // Optional output: cos(az), Length [n]
                       arma::fvec *sEL = nullptr,  // Optional output: sin(el), Length [n]
                       arma::fvec *cEL = nullptr); // Optional output: cos(el), Length [n]

    // Fast, approximate Cartesian-to-geographic conversion
    template <typename dtype>
    void fast_cart2geo(const arma::Col<dtype> &x,       // Input x-coordinates, Length [n]
                       const arma::Col<dtype> &y,       // Input y-coordinates, Length [n]
                       const arma::Col<dtype> &z,       // Input z-coordinates, Length [n]
                       arma::Col<dtype> &az,            // Output azimuth angles in radians, Length [n]
                       arma::Col<dtype> &el,            // Output elevation angles in radians, Length [n]
                       arma::Col<dtype> *len = nullptr, // Vector length, Length [n]
                       int use_kernel = 0);             // Kernel: 0 = auto, 1 = GENERIC, 2 = AVX2

}

#endif