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
        const double *xp = x + off;
        __m256 sv, cv;
        _fm256_sincos256_pd(_mm256_loadu_pd(xp), _mm256_loadu_pd(xp + 4), &sv, &cv);
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
        const double *xp = x + off;
        __m256 sv, cv;
        _fm256_sincos256_pd(_mm256_loadu_pd(xp), _mm256_loadu_pd(xp + 4), &sv, &cv);
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
        const double *xp = x + off;
        __m256 sv, cv;
        _fm256_sincos256_pd(_mm256_loadu_pd(xp), _mm256_loadu_pd(xp + 4), &sv, &cv);
        _mm256_storeu_ps(c + off, cv);
    }
}

template <> // float
void qd_ASIN_AVX2(const float *__restrict x,
                  float *__restrict s,
                  size_t n_val) // multiple of 8
{
    const long long n_vec = (long long)n_val >> 3; // number of 8-float vectors

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 xv = _mm256_loadu_ps(x + off); // no alignment required
        _mm256_storeu_ps(s + off, _fm256_asin256_ps(xv));
    }
}

template <> // double
void qd_ASIN_AVX2(const double *__restrict x,
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
        _mm256_storeu_ps(s + off, _fm256_asin256_ps(xv));
    }
}

template <> // float
void qd_ACOS_AVX2(const float *__restrict x,
                  float *__restrict c,
                  size_t n_val) // multiple of 8
{
    const long long n_vec = (long long)n_val >> 3; // number of 8-float vectors

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 xv = _mm256_loadu_ps(x + off); // no alignment required
        _mm256_storeu_ps(c + off, _fm256_acos256_ps(xv));
    }
}

template <> // double
void qd_ACOS_AVX2(const double *__restrict x,
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
        _mm256_storeu_ps(c + off, _fm256_acos256_ps(xv));
    }
}

template <> // float
void qd_ATAN2_AVX2(const float *__restrict y,
                   const float *__restrict x,
                   float *__restrict a,
                   size_t n_val) // multiple of 8
{
    const long long n_vec = (long long)n_val >> 3; // number of 8-float vectors

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 yv = _mm256_loadu_ps(y + off);
        __m256 xv = _mm256_loadu_ps(x + off);
        _mm256_storeu_ps(a + off, _fm256_atan2256_ps(yv, xv));
    }
}

template <> // double
void qd_ATAN2_AVX2(const double *__restrict y,
                   const double *__restrict x,
                   float *__restrict a,
                   size_t n_val) // multiple of 8
{
    const long long n_vec = (long long)n_val >> 3; // number of 8-float vectors

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 yv = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_loadu_pd(y + off + 4)),
                                    _mm256_cvtpd_ps(_mm256_loadu_pd(y + off)));
        __m256 xv = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_loadu_pd(x + off + 4)),
                                    _mm256_cvtpd_ps(_mm256_loadu_pd(x + off)));
        _mm256_storeu_ps(a + off, _fm256_atan2256_ps(yv, xv));
    }
}

template <> // float
void qd_SLERP_AVX2(const float *__restrict Ar, const float *__restrict Ai,
                   const float *__restrict Br, const float *__restrict Bi,
                   const float *__restrict w,
                   float *__restrict Xr, float *__restrict Xi,
                   size_t n_val) // multiple of 8
{
    const long long n_vec = (long long)n_val >> 3; // number of 8-float vectors

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 vAr = _mm256_loadu_ps(Ar + off);
        __m256 vAi = _mm256_loadu_ps(Ai + off);
        __m256 vBr = _mm256_loadu_ps(Br + off);
        __m256 vBi = _mm256_loadu_ps(Bi + off);
        __m256 vw = _mm256_loadu_ps(w + off);
        __m256 vXr, vXi;
        _fm256_slerp_complex_ps(vAr, vAi, vBr, vBi, vw, &vXr, &vXi);
        _mm256_storeu_ps(Xr + off, vXr);
        _mm256_storeu_ps(Xi + off, vXi);
    }
}

template <> // double
void qd_SLERP_AVX2(const double *__restrict Ar, const double *__restrict Ai,
                   const double *__restrict Br, const double *__restrict Bi,
                   const double *__restrict w,
                   float *__restrict Xr, float *__restrict Xi,
                   size_t n_val) // multiple of 8
{
    const long long n_vec = (long long)n_val >> 3; // number of 8-float vectors

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);

        // Load 8 doubles as two groups of 4, convert each to float, pack into __m256
        __m256 vAr = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_loadu_pd(Ar + off + 4)),
                                     _mm256_cvtpd_ps(_mm256_loadu_pd(Ar + off)));
        __m256 vAi = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_loadu_pd(Ai + off + 4)),
                                     _mm256_cvtpd_ps(_mm256_loadu_pd(Ai + off)));
        __m256 vBr = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_loadu_pd(Br + off + 4)),
                                     _mm256_cvtpd_ps(_mm256_loadu_pd(Br + off)));
        __m256 vBi = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_loadu_pd(Bi + off + 4)),
                                     _mm256_cvtpd_ps(_mm256_loadu_pd(Bi + off)));
        __m256 vw = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_loadu_pd(w + off + 4)),
                                    _mm256_cvtpd_ps(_mm256_loadu_pd(w + off)));

        __m256 vXr, vXi;
        _fm256_slerp_complex_ps(vAr, vAi, vBr, vBi, vw, &vXr, &vXi);
        _mm256_storeu_ps(Xr + off, vXr);
        _mm256_storeu_ps(Xi + off, vXi);
    }
}