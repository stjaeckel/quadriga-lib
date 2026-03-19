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

#include "fastmath_vectorized_avx2.h"
#include "fastmath_avx2.h"

#include <immintrin.h>
#include <limits.h>
#include <cstring> // memcpy

#ifndef QD_OMP_THRESHOLD
#define QD_OMP_THRESHOLD 4096 // iterations of the inner loop before parallelizing
#endif

template <> // float
void qd_SINCOS_AVX2(const float *__restrict x,
                    float *__restrict s,
                    float *__restrict c,
                    size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3; // number of full 8-float vectors

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 xv = _mm256_loadu_ps(x + off);
        __m256 sv, cv;
        _fm256_sincos256_ps(xv, &sv, &cv);
        _mm256_storeu_ps(s + off, sv);
        _mm256_storeu_ps(c + off, cv);
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) float xp[8] = {};
        std::memcpy(xp, x + off, n_tail * sizeof(float));
        __m256 sv, cv;
        _fm256_sincos256_ps(_mm256_load_ps(xp), &sv, &cv);
        alignas(32) float sb[8], cb[8];
        _mm256_store_ps(sb, sv);
        _mm256_store_ps(cb, cv);
        std::memcpy(s + off, sb, n_tail * sizeof(float));
        std::memcpy(c + off, cb, n_tail * sizeof(float));
    }
}

template <> // double
void qd_SINCOS_AVX2(const double *__restrict x,
                    float *__restrict s,
                    float *__restrict c,
                    size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

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

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) double xp[8] = {};
        std::memcpy(xp, x + off, n_tail * sizeof(double));
        __m256 sv, cv;
        _fm256_sincos256_pd(_mm256_load_pd(xp), _mm256_load_pd(xp + 4), &sv, &cv);
        alignas(32) float sb[8], cb[8];
        _mm256_store_ps(sb, sv);
        _mm256_store_ps(cb, cv);
        std::memcpy(s + off, sb, n_tail * sizeof(float));
        std::memcpy(c + off, cb, n_tail * sizeof(float));
    }
}

template <> // float
void qd_SIN_AVX2(const float *__restrict x,
                 float *__restrict s,
                 size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 xv = _mm256_loadu_ps(x + off);
        __m256 sv, cv;
        _fm256_sincos256_ps(xv, &sv, &cv);
        _mm256_storeu_ps(s + off, sv);
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) float xp[8] = {};
        std::memcpy(xp, x + off, n_tail * sizeof(float));
        __m256 sv, cv;
        _fm256_sincos256_ps(_mm256_load_ps(xp), &sv, &cv);
        alignas(32) float sb[8];
        _mm256_store_ps(sb, sv);
        std::memcpy(s + off, sb, n_tail * sizeof(float));
    }
}

template <> // double
void qd_SIN_AVX2(const double *__restrict x,
                 float *__restrict s,
                 size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        const double *xp = x + off;
        __m256 sv, cv;
        _fm256_sincos256_pd(_mm256_loadu_pd(xp), _mm256_loadu_pd(xp + 4), &sv, &cv);
        _mm256_storeu_ps(s + off, sv);
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) double xp[8] = {};
        std::memcpy(xp, x + off, n_tail * sizeof(double));
        __m256 sv, cv;
        _fm256_sincos256_pd(_mm256_load_pd(xp), _mm256_load_pd(xp + 4), &sv, &cv);
        alignas(32) float sb[8];
        _mm256_store_ps(sb, sv);
        std::memcpy(s + off, sb, n_tail * sizeof(float));
    }
}

template <> // float
void qd_COS_AVX2(const float *__restrict x,
                 float *__restrict c,
                 size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 xv = _mm256_loadu_ps(x + off);
        __m256 sv, cv;
        _fm256_sincos256_ps(xv, &sv, &cv);
        _mm256_storeu_ps(c + off, cv);
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) float xp[8] = {};
        std::memcpy(xp, x + off, n_tail * sizeof(float));
        __m256 sv, cv;
        _fm256_sincos256_ps(_mm256_load_ps(xp), &sv, &cv);
        alignas(32) float cb[8];
        _mm256_store_ps(cb, cv);
        std::memcpy(c + off, cb, n_tail * sizeof(float));
    }
}

template <> // double
void qd_COS_AVX2(const double *__restrict x,
                 float *__restrict c,
                 size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        const double *xp = x + off;
        __m256 sv, cv;
        _fm256_sincos256_pd(_mm256_loadu_pd(xp), _mm256_loadu_pd(xp + 4), &sv, &cv);
        _mm256_storeu_ps(c + off, cv);
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) double xp[8] = {};
        std::memcpy(xp, x + off, n_tail * sizeof(double));
        __m256 sv, cv;
        _fm256_sincos256_pd(_mm256_load_pd(xp), _mm256_load_pd(xp + 4), &sv, &cv);
        alignas(32) float cb[8];
        _mm256_store_ps(cb, cv);
        std::memcpy(c + off, cb, n_tail * sizeof(float));
    }
}

template <> // float
void qd_ASIN_AVX2(const float *__restrict x,
                  float *__restrict s,
                  size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 xv = _mm256_loadu_ps(x + off);
        _mm256_storeu_ps(s + off, _fm256_asin256_ps(xv));
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) float xp[8] = {};
        std::memcpy(xp, x + off, n_tail * sizeof(float));
        alignas(32) float sb[8];
        _mm256_store_ps(sb, _fm256_asin256_ps(_mm256_load_ps(xp)));
        std::memcpy(s + off, sb, n_tail * sizeof(float));
    }
}

template <> // double
void qd_ASIN_AVX2(const double *__restrict x,
                  float *__restrict s,
                  size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        const double *xp = x + off;
        __m256d x0 = _mm256_loadu_pd(xp);
        __m256d x1 = _mm256_loadu_pd(xp + 4);
        __m128 xf0 = _mm256_cvtpd_ps(x0);
        __m128 xf1 = _mm256_cvtpd_ps(x1);
        __m256 xv = _mm256_set_m128(xf1, xf0);
        _mm256_storeu_ps(s + off, _fm256_asin256_ps(xv));
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) double xp[8] = {};
        std::memcpy(xp, x + off, n_tail * sizeof(double));
        __m256 xv = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_load_pd(xp + 4)),
                                    _mm256_cvtpd_ps(_mm256_load_pd(xp)));
        alignas(32) float sb[8];
        _mm256_store_ps(sb, _fm256_asin256_ps(xv));
        std::memcpy(s + off, sb, n_tail * sizeof(float));
    }
}

template <> // float
void qd_ACOS_AVX2(const float *__restrict x,
                  float *__restrict c,
                  size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 xv = _mm256_loadu_ps(x + off);
        _mm256_storeu_ps(c + off, _fm256_acos256_ps(xv));
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) float xp[8] = {};
        std::memcpy(xp, x + off, n_tail * sizeof(float));
        alignas(32) float cb[8];
        _mm256_store_ps(cb, _fm256_acos256_ps(_mm256_load_ps(xp)));
        std::memcpy(c + off, cb, n_tail * sizeof(float));
    }
}

template <> // double
void qd_ACOS_AVX2(const double *__restrict x,
                  float *__restrict c,
                  size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        const double *xp = x + off;
        __m256d x0 = _mm256_loadu_pd(xp);
        __m256d x1 = _mm256_loadu_pd(xp + 4);
        __m128 xf0 = _mm256_cvtpd_ps(x0);
        __m128 xf1 = _mm256_cvtpd_ps(x1);
        __m256 xv = _mm256_set_m128(xf1, xf0);
        _mm256_storeu_ps(c + off, _fm256_acos256_ps(xv));
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) double xp[8] = {};
        std::memcpy(xp, x + off, n_tail * sizeof(double));
        __m256 xv = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_load_pd(xp + 4)),
                                    _mm256_cvtpd_ps(_mm256_load_pd(xp)));
        alignas(32) float cb[8];
        _mm256_store_ps(cb, _fm256_acos256_ps(xv));
        std::memcpy(c + off, cb, n_tail * sizeof(float));
    }
}

template <> // float
void qd_ATAN2_AVX2(const float *__restrict y,
                   const float *__restrict x,
                   float *__restrict a,
                   size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 yv = _mm256_loadu_ps(y + off);
        __m256 xv = _mm256_loadu_ps(x + off);
        _mm256_storeu_ps(a + off, _fm256_atan2256_ps(yv, xv));
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) float yp[8] = {}, xp[8] = {};
        std::memcpy(yp, y + off, n_tail * sizeof(float));
        std::memcpy(xp, x + off, n_tail * sizeof(float));
        alignas(32) float ab[8];
        _mm256_store_ps(ab, _fm256_atan2256_ps(_mm256_load_ps(yp), _mm256_load_ps(xp)));
        std::memcpy(a + off, ab, n_tail * sizeof(float));
    }
}

template <> // double
void qd_ATAN2_AVX2(const double *__restrict y,
                   const double *__restrict x,
                   float *__restrict a,
                   size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

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

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) double yp[8] = {}, xp[8] = {};
        std::memcpy(yp, y + off, n_tail * sizeof(double));
        std::memcpy(xp, x + off, n_tail * sizeof(double));
        __m256 yv = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_load_pd(yp + 4)),
                                    _mm256_cvtpd_ps(_mm256_load_pd(yp)));
        __m256 xv = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_load_pd(xp + 4)),
                                    _mm256_cvtpd_ps(_mm256_load_pd(xp)));
        alignas(32) float ab[8];
        _mm256_store_ps(ab, _fm256_atan2256_ps(yv, xv));
        std::memcpy(a + off, ab, n_tail * sizeof(float));
    }
}

template <> // float
void qd_SLERP_AVX2(const float *__restrict Ar, const float *__restrict Ai,
                   const float *__restrict Br, const float *__restrict Bi,
                   const float *__restrict w,
                   float *__restrict Xr, float *__restrict Xi,
                   size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

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

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) float pAr[8] = {}, pAi[8] = {}, pBr[8] = {}, pBi[8] = {}, pw[8] = {};
        std::memcpy(pAr, Ar + off, n_tail * sizeof(float));
        std::memcpy(pAi, Ai + off, n_tail * sizeof(float));
        std::memcpy(pBr, Br + off, n_tail * sizeof(float));
        std::memcpy(pBi, Bi + off, n_tail * sizeof(float));
        std::memcpy(pw, w + off, n_tail * sizeof(float));
        __m256 vXr, vXi;
        _fm256_slerp_complex_ps(_mm256_load_ps(pAr), _mm256_load_ps(pAi),
                                _mm256_load_ps(pBr), _mm256_load_ps(pBi),
                                _mm256_load_ps(pw), &vXr, &vXi);
        alignas(32) float oXr[8], oXi[8];
        _mm256_store_ps(oXr, vXr);
        _mm256_store_ps(oXi, vXi);
        std::memcpy(Xr + off, oXr, n_tail * sizeof(float));
        std::memcpy(Xi + off, oXi, n_tail * sizeof(float));
    }
}

template <> // double
void qd_SLERP_AVX2(const double *__restrict Ar, const double *__restrict Ai,
                   const double *__restrict Br, const double *__restrict Bi,
                   const double *__restrict w,
                   float *__restrict Xr, float *__restrict Xi,
                   size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);

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

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) double pAr[8] = {}, pAi[8] = {}, pBr[8] = {}, pBi[8] = {}, pw[8] = {};
        std::memcpy(pAr, Ar + off, n_tail * sizeof(double));
        std::memcpy(pAi, Ai + off, n_tail * sizeof(double));
        std::memcpy(pBr, Br + off, n_tail * sizeof(double));
        std::memcpy(pBi, Bi + off, n_tail * sizeof(double));
        std::memcpy(pw, w + off, n_tail * sizeof(double));

        __m256 vAr_ = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_load_pd(pAr + 4)),
                                      _mm256_cvtpd_ps(_mm256_load_pd(pAr)));
        __m256 vAi_ = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_load_pd(pAi + 4)),
                                      _mm256_cvtpd_ps(_mm256_load_pd(pAi)));
        __m256 vBr_ = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_load_pd(pBr + 4)),
                                      _mm256_cvtpd_ps(_mm256_load_pd(pBr)));
        __m256 vBi_ = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_load_pd(pBi + 4)),
                                      _mm256_cvtpd_ps(_mm256_load_pd(pBi)));
        __m256 vw_ = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_load_pd(pw + 4)),
                                     _mm256_cvtpd_ps(_mm256_load_pd(pw)));

        __m256 vXr, vXi;
        _fm256_slerp_complex_ps(vAr_, vAi_, vBr_, vBi_, vw_, &vXr, &vXi);
        alignas(32) float oXr[8], oXi[8];
        _mm256_store_ps(oXr, vXr);
        _mm256_store_ps(oXi, vXi);
        std::memcpy(Xr + off, oXr, n_tail * sizeof(float));
        std::memcpy(Xi + off, oXi, n_tail * sizeof(float));
    }
}

template <> // float
void qd_GEO2CART_AVX2(const float *__restrict az, const float *__restrict el,
                      float *__restrict x, float *__restrict y, float *__restrict z,
                      float *__restrict sAZ, float *__restrict cAZ,
                      float *__restrict sEL, float *__restrict cEL,
                      size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 vaz = _mm256_loadu_ps(az + off);
        __m256 vel = _mm256_loadu_ps(el + off);
        __m256 vx, vy, vz, vsa, vca, vse, vce;
        _fm256_geo2cart_ps(vaz, vel, &vx, &vy, &vz, &vsa, &vca, &vse, &vce);
        _mm256_storeu_ps(x + off, vx);
        _mm256_storeu_ps(y + off, vy);
        _mm256_storeu_ps(z + off, vz);
        if (sAZ)
            _mm256_storeu_ps(sAZ + off, vsa);
        if (cAZ)
            _mm256_storeu_ps(cAZ + off, vca);
        if (sEL)
            _mm256_storeu_ps(sEL + off, vse);
        if (cEL)
            _mm256_storeu_ps(cEL + off, vce);
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) float paz[8] = {}, pel[8] = {};
        std::memcpy(paz, az + off, n_tail * sizeof(float));
        std::memcpy(pel, el + off, n_tail * sizeof(float));
        __m256 vx, vy, vz, vsa, vca, vse, vce;
        _fm256_geo2cart_ps(_mm256_load_ps(paz), _mm256_load_ps(pel),
                           &vx, &vy, &vz, &vsa, &vca, &vse, &vce);
        alignas(32) float bx[8], by[8], bz[8];
        _mm256_store_ps(bx, vx);
        _mm256_store_ps(by, vy);
        _mm256_store_ps(bz, vz);
        std::memcpy(x + off, bx, n_tail * sizeof(float));
        std::memcpy(y + off, by, n_tail * sizeof(float));
        std::memcpy(z + off, bz, n_tail * sizeof(float));
        if (sAZ)
        {
            alignas(32) float b[8];
            _mm256_store_ps(b, vsa);
            std::memcpy(sAZ + off, b, n_tail * sizeof(float));
        }
        if (cAZ)
        {
            alignas(32) float b[8];
            _mm256_store_ps(b, vca);
            std::memcpy(cAZ + off, b, n_tail * sizeof(float));
        }
        if (sEL)
        {
            alignas(32) float b[8];
            _mm256_store_ps(b, vse);
            std::memcpy(sEL + off, b, n_tail * sizeof(float));
        }
        if (cEL)
        {
            alignas(32) float b[8];
            _mm256_store_ps(b, vce);
            std::memcpy(cEL + off, b, n_tail * sizeof(float));
        }
    }
}

template <> // double
void qd_GEO2CART_AVX2(const double *__restrict az, const double *__restrict el,
                      float *__restrict x, float *__restrict y, float *__restrict z,
                      float *__restrict sAZ, float *__restrict cAZ,
                      float *__restrict sEL, float *__restrict cEL,
                      size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        const double *azp = az + off;
        const double *elp = el + off;

        __m256 vsa, vca;
        _fm256_sincos256_pd(_mm256_loadu_pd(azp), _mm256_loadu_pd(azp + 4), &vsa, &vca);

        __m256 vse, vce;
        _fm256_sincos256_pd(_mm256_loadu_pd(elp), _mm256_loadu_pd(elp + 4), &vse, &vce);

        _mm256_storeu_ps(x + off, _mm256_mul_ps(vce, vca));
        _mm256_storeu_ps(y + off, _mm256_mul_ps(vce, vsa));
        _mm256_storeu_ps(z + off, vse);
        if (sAZ)
            _mm256_storeu_ps(sAZ + off, vsa);
        if (cAZ)
            _mm256_storeu_ps(cAZ + off, vca);
        if (sEL)
            _mm256_storeu_ps(sEL + off, vse);
        if (cEL)
            _mm256_storeu_ps(cEL + off, vce);
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) double paz[8] = {}, pel[8] = {};
        std::memcpy(paz, az + off, n_tail * sizeof(double));
        std::memcpy(pel, el + off, n_tail * sizeof(double));

        __m256 vsa, vca;
        _fm256_sincos256_pd(_mm256_load_pd(paz), _mm256_load_pd(paz + 4), &vsa, &vca);

        __m256 vse, vce;
        _fm256_sincos256_pd(_mm256_load_pd(pel), _mm256_load_pd(pel + 4), &vse, &vce);

        alignas(32) float bx[8], by[8], bz[8];
        _mm256_store_ps(bx, _mm256_mul_ps(vce, vca));
        _mm256_store_ps(by, _mm256_mul_ps(vce, vsa));
        _mm256_store_ps(bz, vse);
        std::memcpy(x + off, bx, n_tail * sizeof(float));
        std::memcpy(y + off, by, n_tail * sizeof(float));
        std::memcpy(z + off, bz, n_tail * sizeof(float));
        if (sAZ)
        {
            alignas(32) float b[8];
            _mm256_store_ps(b, vsa);
            std::memcpy(sAZ + off, b, n_tail * sizeof(float));
        }
        if (cAZ)
        {
            alignas(32) float b[8];
            _mm256_store_ps(b, vca);
            std::memcpy(cAZ + off, b, n_tail * sizeof(float));
        }
        if (sEL)
        {
            alignas(32) float b[8];
            _mm256_store_ps(b, vse);
            std::memcpy(sEL + off, b, n_tail * sizeof(float));
        }
        if (cEL)
        {
            alignas(32) float b[8];
            _mm256_store_ps(b, vce);
            std::memcpy(cEL + off, b, n_tail * sizeof(float));
        }
    }
}

template <> // float
void qd_CART2GEO_AVX2(const float *__restrict x, const float *__restrict y, const float *__restrict z,
                      float *__restrict az, float *__restrict el, size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 vx = _mm256_loadu_ps(x + off);
        __m256 vy = _mm256_loadu_ps(y + off);
        __m256 vz = _mm256_loadu_ps(z + off);
        __m256 vaz, vel;
        _fm256_cart2geo_ps(vx, vy, vz, &vaz, &vel);
        _mm256_storeu_ps(az + off, vaz);
        _mm256_storeu_ps(el + off, vel);
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) float px[8] = {}, py[8] = {}, pz[8] = {};
        std::memcpy(px, x + off, n_tail * sizeof(float));
        std::memcpy(py, y + off, n_tail * sizeof(float));
        std::memcpy(pz, z + off, n_tail * sizeof(float));
        __m256 vaz, vel;
        _fm256_cart2geo_ps(_mm256_load_ps(px), _mm256_load_ps(py), _mm256_load_ps(pz), &vaz, &vel);
        alignas(32) float baz[8], bel[8];
        _mm256_store_ps(baz, vaz);
        _mm256_store_ps(bel, vel);
        std::memcpy(az + off, baz, n_tail * sizeof(float));
        std::memcpy(el + off, bel, n_tail * sizeof(float));
    }
}

template <> // double
void qd_CART2GEO_AVX2(const double *__restrict x, const double *__restrict y, const double *__restrict z,
                      float *__restrict az, float *__restrict el, size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 vx = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_loadu_pd(x + off + 4)),
                                    _mm256_cvtpd_ps(_mm256_loadu_pd(x + off)));
        __m256 vy = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_loadu_pd(y + off + 4)),
                                    _mm256_cvtpd_ps(_mm256_loadu_pd(y + off)));
        __m256 vz = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_loadu_pd(z + off + 4)),
                                    _mm256_cvtpd_ps(_mm256_loadu_pd(z + off)));
        __m256 vaz, vel;
        _fm256_cart2geo_ps(vx, vy, vz, &vaz, &vel);
        _mm256_storeu_ps(az + off, vaz);
        _mm256_storeu_ps(el + off, vel);
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) double px[8] = {}, py[8] = {}, pz[8] = {};
        std::memcpy(px, x + off, n_tail * sizeof(double));
        std::memcpy(py, y + off, n_tail * sizeof(double));
        std::memcpy(pz, z + off, n_tail * sizeof(double));
        __m256 vx = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_load_pd(px + 4)),
                                    _mm256_cvtpd_ps(_mm256_load_pd(px)));
        __m256 vy = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_load_pd(py + 4)),
                                    _mm256_cvtpd_ps(_mm256_load_pd(py)));
        __m256 vz = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_load_pd(pz + 4)),
                                    _mm256_cvtpd_ps(_mm256_load_pd(pz)));
        __m256 vaz, vel;
        _fm256_cart2geo_ps(vx, vy, vz, &vaz, &vel);
        alignas(32) float baz[8], bel[8];
        _mm256_store_ps(baz, vaz);
        _mm256_store_ps(bel, vel);
        std::memcpy(az + off, baz, n_tail * sizeof(float));
        std::memcpy(el + off, bel, n_tail * sizeof(float));
    }
}