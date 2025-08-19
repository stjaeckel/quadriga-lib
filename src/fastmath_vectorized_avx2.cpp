// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

#include "fastmath_vectorized_avx2.h"
#include "fastmath_avx2.h"

#include <immintrin.h>
#include <limits.h>

#ifndef QD_OMP_THRESHOLD
#define QD_OMP_THRESHOLD 4096 // iterations of the inner loop before parallelizing
#endif

template <> // float
void qd_SINCOS_AVX2(const float *__restrict x,
                    float *__restrict s,
                    float *__restrict c,
                    size_t n_val) // multiple of 8
{
    const long long n_vec = (long long)n_val >> 3; // number of 8-float vectors

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 xv = _mm256_loadu_ps(x + off); // no alignment required
        __m256 sv, cv;
        _fm256_sincos256_ps(xv, &sv, &cv);
        _mm256_storeu_ps(s + off, sv);
        _mm256_storeu_ps(c + off, cv);
    }
}

template <> // double
void qd_SINCOS_AVX2(const double *__restrict x,
                    float *__restrict s,
                    float *__restrict c,
                    size_t n_val) // multiple of 8
{
    const long long n_vec = (long long)n_val >> 3; // number of 8-float vectors

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        const double *xp = x + off;            // 8 doubles
        __m256d x0 = _mm256_loadu_pd(xp);      // 0..3
        __m256d x1 = _mm256_loadu_pd(xp + 4);  // 4..7
        __m128 xf0 = _mm256_cvtpd_ps(x0);      // 4f
        __m128 xf1 = _mm256_cvtpd_ps(x1);      // 4f
        __m256 xv = _mm256_set_m128(xf1, xf0); // pack 8f
        __m256 sv, cv;
        _fm256_sincos256_ps(xv, &sv, &cv);
        _mm256_storeu_ps(s + off, sv);
        _mm256_storeu_ps(c + off, cv);
    }
}

template <> // float
void qd_SIN_AVX2(const float *__restrict x,
                 float *__restrict s,
                 size_t n_val) // multiple of 8
{
    const long long n_vec = (long long)n_val >> 3; // number of 8-float vectors

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 xv = _mm256_loadu_ps(x + off); // no alignment required
        __m256 sv, cv;
        _fm256_sincos256_ps(xv, &sv, &cv);
        _mm256_storeu_ps(s + off, sv);
    }
}

template <> // double
void qd_SIN_AVX2(const double *__restrict x,
                 float *__restrict s,
                 size_t n_val) // multiple of 8
{
    const long long n_vec = (long long)n_val >> 3; // number of 8-float vectors

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        const double *xp = x + off;            // 8 doubles
        __m256d x0 = _mm256_loadu_pd(xp);      // 0..3
        __m256d x1 = _mm256_loadu_pd(xp + 4);  // 4..7
        __m128 xf0 = _mm256_cvtpd_ps(x0);      // 4f
        __m128 xf1 = _mm256_cvtpd_ps(x1);      // 4f
        __m256 xv = _mm256_set_m128(xf1, xf0); // pack 8f
        __m256 sv, cv;
        _fm256_sincos256_ps(xv, &sv, &cv);
        _mm256_storeu_ps(s + off, sv);
    }
}

template <> // float
void qd_COS_AVX2(const float *__restrict x,
                 float *__restrict c,
                 size_t n_val) // multiple of 8
{
    const long long n_vec = (long long)n_val >> 3; // number of 8-float vectors

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 xv = _mm256_loadu_ps(x + off); // no alignment required
        __m256 sv, cv;
        _fm256_sincos256_ps(xv, &sv, &cv);
        _mm256_storeu_ps(c + off, cv);
    }
}

template <> // double
void qd_COS_AVX2(const double *__restrict x,
                 float *__restrict c,
                 size_t n_val) // multiple of 8
{
    const long long n_vec = (long long)n_val >> 3; // number of 8-float vectors

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        const double *xp = x + off;            // 8 doubles
        __m256d x0 = _mm256_loadu_pd(xp);      // 0..3
        __m256d x1 = _mm256_loadu_pd(xp + 4);  // 4..7
        __m128 xf0 = _mm256_cvtpd_ps(x0);      // 4f
        __m128 xf1 = _mm256_cvtpd_ps(x1);      // 4f
        __m256 xv = _mm256_set_m128(xf1, xf0); // pack 8f
        __m256 sv, cv;
        _fm256_sincos256_ps(xv, &sv, &cv);
        _mm256_storeu_ps(c + off, cv);
    }
}
