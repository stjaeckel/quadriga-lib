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
    // - Computes sin(x) and/or cos(x) for all elements of an Armadillo vector
    // - Uses AVX2 to process 8 floats per vector lane
    // - Parallelizes across cores with OpenMP
    // - Results are approximate and may differ from std::sinf / std::cosf
    // - Either 's' or 'c' may be nullptr to compute only one function
    // - Fallback to std::sinf / std::cosf if complied without AVX2 (or running with non-AVX2 CPU)
    template <typename dtype>
    void fast_sincos(const arma::Col<dtype> &x, // Input angles in radians; n = x.n_elem, float or double
                     arma::fvec *s = nullptr,   // [out] If non-null, set to sin(x). Resized to length n if needed
                     arma::fvec *c = nullptr);  // [out] If non-null, set to cos(x). Resized to length n if needed

    // Fast, approximate arc-sine for single-precision vectors
    // - Computes asin(x) for all elements of an Armadillo vector
    // - Uses AVX2 to process 8 floats per vector lane
    // - Parallelizes across cores with OpenMP
    // - Results are approximate and may differ from std::asinf
    // - For x in [-1, 1], the maximum error is approximately 2 ULP (~2.4e-7)
    // - Input values outside [-1, 1] produce NaN (IEEE compliant)
    // - Fallback to std::asinf if compiled without AVX2 (or running with non-AVX2 CPU)
    template <typename dtype>
    void fast_asin(const arma::Col<dtype> &x, // Input values in [-1, 1]; n = x.n_elem, float or double
                   arma::fvec &s);            // [out] Set to asin(x). Resized to length n if needed

    // Fast, approximate arc-cosine for single-precision vectors
    // - Computes acos(x) for all elements of an Armadillo vector
    // - Uses AVX2 to process 8 floats per vector lane
    // - Parallelizes across cores with OpenMP
    // - Results are approximate and may differ from std::acosf
    // - For x in [-1, 1], the maximum error is approximately 2 ULP (~2.4e-7)
    // - Input values outside [-1, 1] produce NaN (IEEE compliant)
    // - Fallback to std::acosf if compiled without AVX2 (or running with non-AVX2 CPU)
    template <typename dtype>
    void fast_acos(const arma::Col<dtype> &x, // Input values in [-1, 1]; n = x.n_elem, float or double
                   arma::fvec &c);            // [out] Set to acos(x). Resized to length n if needed

    // Fast, approximate atan2(y, x) for single-precision vectors
    // - Computes atan2(y, x) for all elements of two Armadillo vectors
    // - Uses AVX2 to process 8 floats per vector lane
    // - Parallelizes across cores with OpenMP
    // - Results are approximate and may differ from std::atan2f
    // - Maximum error is approximately 3 ULP (~3.6e-7) across the full domain
    // - atan2(0, 0) returns 0; atan2(0, 0) returns 0; atan2(±0, -0) returns ±0 (not ±pi)
    // - Fallback to std::atan2f if compiled without AVX2 (or running with non-AVX2 CPU)
    template <typename dtype>
    void fast_atan2(const arma::Col<dtype> &y, // Input y-coordinates, Length [n], float or double
                    const arma::Col<dtype> &x, // Input x-coordinates, Length [n], float or double
                    arma::fvec &a);            // [out] Set to atan2(y, x) in radians. Resized to length n if needed

    // Fast, approximate spherical interpolation (SLERP) for complex value pairs
    // - Interpolates between two complex-valued vectors using SLERP on normalized directions
    //   and linear interpolation of amplitudes
    // - Uses AVX2 to process 8 complex pairs per vector lane
    // - Parallelizes across cores with OpenMP
    // - Near-antipodal inputs (phase angle close to pi) smoothly transition to linear interpolation
    // - Both amplitudes negligible → output is zero
    // - Maximum error vs double-precision reference: ~100 ULP (~1.2e-5 relative, ~17.5 effective bits of 23)
    // - Fallback to scalar slerp_complex_mf if compiled without AVX2 (or running with non-AVX2 CPU)
    template <typename dtype>
    void fast_slerp(const arma::Col<dtype> &Ar, // Real part of source A, Length [n]
                    const arma::Col<dtype> &Ai, // Imaginary part of source A, Length [n]
                    const arma::Col<dtype> &Br, // Real part of source B, Length [n]
                    const arma::Col<dtype> &Bi, // Imaginary part of source B, Length [n]
                    const arma::Col<dtype> &w,  // Per-element interpolation weight in [0,1], 0 = A, 1 = B, Length [n]
                    arma::fvec &Xr,             // [out] Real part of interpolated result, Length [n]
                    arma::fvec &Xi);            // [out] Imaginary part of interpolated result, Length [n]

}

#endif