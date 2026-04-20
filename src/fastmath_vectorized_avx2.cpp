// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "fastmath_vectorized_avx2.h"
#include "fastmath_avx2.h"

#include <immintrin.h>
#include <limits.h>
#include <cstring> // std::memcpy

#ifndef QD_OMP_THRESHOLD
#define QD_OMP_THRESHOLD 4096 // iterations of the inner loop before parallelizing
#endif

template <> // float
void qd_SINCOS_AVX2(const float *x,
                    float *s,
                    float *c,
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
void qd_SINCOS_AVX2(const double *x,
                    float *s,
                    float *c,
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
void qd_SIN_AVX2(const float *x,
                 float *s,
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
void qd_SIN_AVX2(const double *x,
                 float *s,
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
void qd_COS_AVX2(const float *x,
                 float *c,
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
void qd_COS_AVX2(const double *x,
                 float *c,
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
void qd_ASIN_AVX2(const float *x,
                  float *s,
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
void qd_ASIN_AVX2(const double *x,
                  float *s,
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
void qd_ACOS_AVX2(const float *x,
                  float *c,
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
void qd_ACOS_AVX2(const double *x,
                  float *c,
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
void qd_ATAN2_AVX2(const float *y,
                   const float *x,
                   float *a,
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
void qd_ATAN2_AVX2(const double *y,
                   const double *x,
                   float *a,
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
void qd_SLERP_AVX2(const float *Ar, const float *Ai,
                   const float *Br, const float *Bi,
                   const float *w,
                   float *Xr, float *Xi,
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
void qd_SLERP_AVX2(const double *Ar, const double *Ai,
                   const double *Br, const double *Bi,
                   const double *w,
                   float *Xr, float *Xi,
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
void qd_GEO2CART_AVX2(const float *az, const float *el, const float *len,
                      float *x, float *y, float *z,
                      float *sAZ, float *cAZ, float *sEL, float *cEL,
                      size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;
    const bool has_len = (len != nullptr);

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        const __m256 vaz = _mm256_loadu_ps(az + off);
        const __m256 vel = _mm256_loadu_ps(el + off);
        __m256 vx, vy, vz, vsa, vca, vse, vce;
        _fm256_geo2cart_ps(vaz, vel, &vx, &vy, &vz, &vsa, &vca, &vse, &vce);
        if (has_len)
        {
            const __m256 vlen = _mm256_loadu_ps(len + off);
            vx = _mm256_mul_ps(vx, vlen);
            vy = _mm256_mul_ps(vy, vlen);
            vz = _mm256_mul_ps(vz, vlen);
        }
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

        // Stage tail inputs/outputs through aligned 8-float scratch buffers.
        alignas(32) float scratch_in[8] = {};
        alignas(32) float scratch_out[8];

        auto load_tail = [&](const float *src) -> __m256
        {
            std::memcpy(scratch_in, src + off, n_tail * sizeof(float));
            return _mm256_load_ps(scratch_in);
        };
        auto store_tail = [&](float *dst, __m256 v)
        {
            if (!dst)
                return;
            _mm256_store_ps(scratch_out, v);
            std::memcpy(dst + off, scratch_out, n_tail * sizeof(float));
        };

        const __m256 vaz = load_tail(az);
        const __m256 vel = load_tail(el);
        __m256 vx, vy, vz, vsa, vca, vse, vce;
        _fm256_geo2cart_ps(vaz, vel, &vx, &vy, &vz, &vsa, &vca, &vse, &vce);
        if (has_len)
        {
            const __m256 vlen = load_tail(len);
            vx = _mm256_mul_ps(vx, vlen);
            vy = _mm256_mul_ps(vy, vlen);
            vz = _mm256_mul_ps(vz, vlen);
        }

        store_tail(x, vx);
        store_tail(y, vy);
        store_tail(z, vz);
        store_tail(sAZ, vsa);
        store_tail(cAZ, vca);
        store_tail(sEL, vse);
        store_tail(cEL, vce);
    }
}

template <> // double
void qd_GEO2CART_AVX2(const double *az, const double *el, const double *len,
                      double *x, double *y, double *z,
                      double *sAZ, double *cAZ, double *sEL, double *cEL,
                      size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;
    const bool has_len = (len != nullptr);

    auto store8 = [](double *dst, __m256 v)
    {
        _mm256_storeu_pd(dst, _mm256_cvtps_pd(_mm256_castps256_ps128(v)));
        _mm256_storeu_pd(dst + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)));
    };

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        const double *azp = az + off;
        const double *elp = el + off;

        __m256 vsa, vca, vse, vce;
        _fm256_sincos256_pd(_mm256_loadu_pd(azp), _mm256_loadu_pd(azp + 4), &vsa, &vca);
        _fm256_sincos256_pd(_mm256_loadu_pd(elp), _mm256_loadu_pd(elp + 4), &vse, &vce);

        __m256 vx = _mm256_mul_ps(vce, vca);
        __m256 vy = _mm256_mul_ps(vce, vsa);
        __m256 vz = vse;

        if (has_len)
        {
            const double *lp = len + off;
            const __m128 lo = _mm256_cvtpd_ps(_mm256_loadu_pd(lp));
            const __m128 hi = _mm256_cvtpd_ps(_mm256_loadu_pd(lp + 4));
            const __m256 vlen = _mm256_set_m128(hi, lo);
            vx = _mm256_mul_ps(vx, vlen);
            vy = _mm256_mul_ps(vy, vlen);
            vz = _mm256_mul_ps(vz, vlen);
        }

        store8(x + off, vx);
        store8(y + off, vy);
        store8(z + off, vz);
        if (sAZ)
            store8(sAZ + off, vsa);
        if (cAZ)
            store8(cAZ + off, vca);
        if (sEL)
            store8(sEL + off, vse);
        if (cEL)
            store8(cEL + off, vce);
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;

        alignas(32) double scratch_in[8] = {};
        alignas(32) double scratch_out[8];

        auto load_tail_pd = [&](const double *src, __m256d &lo, __m256d &hi)
        {
            std::memcpy(scratch_in, src + off, n_tail * sizeof(double));
            lo = _mm256_load_pd(scratch_in);
            hi = _mm256_load_pd(scratch_in + 4);
        };
        auto store_tail = [&](double *dst, __m256 v)
        {
            if (!dst)
                return;
            _mm256_store_pd(scratch_out, _mm256_cvtps_pd(_mm256_castps256_ps128(v)));
            _mm256_store_pd(scratch_out + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)));
            std::memcpy(dst + off, scratch_out, n_tail * sizeof(double));
        };

        __m256d vaz_lo, vaz_hi, vel_lo, vel_hi;
        load_tail_pd(az, vaz_lo, vaz_hi);
        load_tail_pd(el, vel_lo, vel_hi);

        __m256 vsa, vca, vse, vce;
        _fm256_sincos256_pd(vaz_lo, vaz_hi, &vsa, &vca);
        _fm256_sincos256_pd(vel_lo, vel_hi, &vse, &vce);

        __m256 vx = _mm256_mul_ps(vce, vca);
        __m256 vy = _mm256_mul_ps(vce, vsa);
        __m256 vz = vse;

        if (has_len)
        {
            __m256d vlen_lo, vlen_hi;
            load_tail_pd(len, vlen_lo, vlen_hi);
            const __m256 vlen = _mm256_set_m128(_mm256_cvtpd_ps(vlen_hi), _mm256_cvtpd_ps(vlen_lo));
            vx = _mm256_mul_ps(vx, vlen);
            vy = _mm256_mul_ps(vy, vlen);
            vz = _mm256_mul_ps(vz, vlen);
        }

        store_tail(x, vx);
        store_tail(y, vy);
        store_tail(z, vz);
        store_tail(sAZ, vsa);
        store_tail(cAZ, vca);
        store_tail(sEL, vse);
        store_tail(cEL, vce);
    }
}

template <> // float
void qd_CART2GEO_AVX2(const float *x, const float *y, const float *z,
                      float *az, float *el, float *len, size_t n_val)
{
    const long long n_vec = (long long)n_val >> 3;

#pragma omp parallel for schedule(static) if (n_vec >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_vec; ++i)
    {
        const size_t off = (static_cast<size_t>(i) << 3);
        __m256 vx = _mm256_loadu_ps(x + off);
        __m256 vy = _mm256_loadu_ps(y + off);
        __m256 vz = _mm256_loadu_ps(z + off);

        __m256 vaz, vel, vlen;
        _fm256_cart2geo_ps(vx, vy, vz, &vaz, &vel, &vlen);

        _mm256_storeu_ps(az + off, vaz);
        _mm256_storeu_ps(el + off, vel);
        if (len)
            _mm256_storeu_ps(len + off, vlen);
    }

    const size_t n_tail = n_val & 7;
    if (n_tail)
    {
        const size_t off = n_val - n_tail;
        alignas(32) float px[8] = {}, py[8] = {}, pz[8] = {};
        std::memcpy(px, x + off, n_tail * sizeof(float));
        std::memcpy(py, y + off, n_tail * sizeof(float));
        std::memcpy(pz, z + off, n_tail * sizeof(float));

        __m256 vaz, vel, vlen;
        _fm256_cart2geo_ps(_mm256_load_ps(px), _mm256_load_ps(py), _mm256_load_ps(pz), &vaz, &vel, &vlen);

        alignas(32) float baz[8], bel[8], blen[8];
        _mm256_store_ps(baz, vaz);
        _mm256_store_ps(bel, vel);
        _mm256_store_ps(blen, vlen);
        std::memcpy(az + off, baz, n_tail * sizeof(float));
        std::memcpy(el + off, bel, n_tail * sizeof(float));
        if (len)
            std::memcpy(len + off, blen, n_tail * sizeof(float));
    }
}

template <> // double
void qd_CART2GEO_AVX2(const double *x, const double *y, const double *z,
                      double *az, double *el, double *len, size_t n_val)
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

        __m256 vaz, vel, vlen;
        _fm256_cart2geo_ps(vx, vy, vz, &vaz, &vel, &vlen);

        _mm256_storeu_pd(az + off, _mm256_cvtps_pd(_mm256_castps256_ps128(vaz)));
        _mm256_storeu_pd(az + off + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(vaz, 1)));
        _mm256_storeu_pd(el + off, _mm256_cvtps_pd(_mm256_castps256_ps128(vel)));
        _mm256_storeu_pd(el + off + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(vel, 1)));
        if (len)
        {
            _mm256_storeu_pd(len + off, _mm256_cvtps_pd(_mm256_castps256_ps128(vlen)));
            _mm256_storeu_pd(len + off + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(vlen, 1)));
        }
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

        __m256 vaz, vel, vlen;
        _fm256_cart2geo_ps(vx, vy, vz, &vaz, &vel, &vlen);

        alignas(32) double daz[8], del[8], dlen[8];
        _mm256_store_pd(daz, _mm256_cvtps_pd(_mm256_castps256_ps128(vaz)));
        _mm256_store_pd(daz + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(vaz, 1)));
        std::memcpy(az + off, daz, n_tail * sizeof(double));

        _mm256_store_pd(del, _mm256_cvtps_pd(_mm256_castps256_ps128(vel)));
        _mm256_store_pd(del + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(vel, 1)));
        std::memcpy(el + off, del, n_tail * sizeof(double));

        if (len)
        {
            _mm256_store_pd(dlen, _mm256_cvtps_pd(_mm256_castps256_ps128(vlen)));
            _mm256_store_pd(dlen + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(vlen, 1)));
            std::memcpy(len + off, dlen, n_tail * sizeof(double));
        }
    }
}