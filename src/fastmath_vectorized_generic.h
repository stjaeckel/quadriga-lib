// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef quadriga_lib_fastmath_vec_generic_H
#define quadriga_lib_fastmath_vec_generic_H

#include <math.h>
#include <stddef.h>
#include <limits.h>
#include "slerp.h"

#ifndef QD_OMP_THRESHOLD
#define QD_OMP_THRESHOLD 4096 // iterations of the inner loop before parallelizing
#endif

// Scalar double-precision Cody–Waite range reduction: map x into [-pi, pi].
// Two-constant split: CW_C1 + CW_C2 = 2*pi, with CW_C1 exactly representable.
// Matches the AVX2 _fm256_range_reduce_2pi_pd helper.
static const double QD_INV_TWO_PI = 0.15915494309189533577; // 1 / (2*pi)
static const double QD_CW_C1 = 6.28125;                     // high bits of 2*pi
static const double QD_CW_C2 = 1.9353071795864769253e-03;   // 2*pi - CW_C1

static inline double qd_range_reduce_2pi(double x)
{
    double n = round(x * QD_INV_TWO_PI); // nearest integer multiple of 2*pi
    x -= n * QD_CW_C1;                   // exact by Sterbenz when n*C1 ≈ x
    x -= n * QD_CW_C2;                   // mop up low bits
    return x;
}

template <typename dtype>
void qd_SINCOS_GENERIC(const dtype * x,
                       float * s,
                       float * c,
                       size_t n_val)
{
    const long long n_val_ll = (long long)n_val;
#pragma omp parallel for schedule(static) if (n_val_ll >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_val_ll; ++i)
    {
        const float xi = (float)qd_range_reduce_2pi((double)x[i]);
        s[i] = sinf(xi);
        c[i] = cosf(xi);
    }
}

template <typename dtype>
void qd_SIN_GENERIC(const dtype * x,
                    float * s,
                    size_t n_val)
{
    const long long n_val_ll = (long long)n_val;
#pragma omp parallel for schedule(static) if (n_val_ll >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_val_ll; ++i)
    {
        const float xi = (float)qd_range_reduce_2pi((double)x[i]);
        s[i] = sinf(xi);
    }
}

template <typename dtype>
void qd_COS_GENERIC(const dtype * x,
                    float * c,
                    size_t n_val)
{
    const long long n_val_ll = (long long)n_val;
#pragma omp parallel for schedule(static) if (n_val_ll >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_val_ll; ++i)
    {
        const float xi = (float)qd_range_reduce_2pi((double)x[i]);
        c[i] = cosf(xi);
    }
}

template <typename dtype>
void qd_ASIN_GENERIC(const dtype * x,
                     float * s,
                     size_t n_val)
{
    const long long n_val_ll = (long long)n_val;
#pragma omp parallel for schedule(static) if (n_val_ll >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_val_ll; ++i)
    {
        const float xi = (float)x[i];
        s[i] = asinf(xi);
    }
}

template <typename dtype>
void qd_ACOS_GENERIC(const dtype * x,
                     float * c,
                     size_t n_val)
{
    const long long n_val_ll = (long long)n_val;
#pragma omp parallel for schedule(static) if (n_val_ll >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_val_ll; ++i)
    {
        const float xi = (float)x[i];
        c[i] = acosf(xi);
    }
}

template <typename dtype>
void qd_ATAN2_GENERIC(const dtype * y,
                      const dtype * x,
                      float * a,
                      size_t n_val)
{
    const long long n_val_ll = (long long)n_val;
#pragma omp parallel for schedule(static) if (n_val_ll >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_val_ll; ++i)
    {
        const float yi = (float)y[i];
        const float xi = (float)x[i];
        a[i] = atan2f(yi, xi);
    }
}

template <typename dtype>
void qd_SLERP_GENERIC(const dtype * Ar, const dtype * Ai,
                      const dtype * Br, const dtype * Bi,
                      const dtype * w,
                      float * Xr, float * Xi,
                      size_t n_val)
{
    const long long n_val_ll = (long long)n_val;
#pragma omp parallel for schedule(static) if (n_val_ll >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_val_ll; ++i)
    {
        float xr, xi;
        slerp_complex_mf<float>((float)Ar[i], (float)Ai[i],
                                (float)Br[i], (float)Bi[i],
                                (float)w[i], xr, xi);
        Xr[i] = xr;
        Xi[i] = xi;
    }
}

template <typename dtype>
void qd_GEO2CART_GENERIC(const dtype *az, const dtype *el, const dtype *len,
                         dtype *x, dtype *y, dtype *z,
                         dtype *sAZ, dtype *cAZ, dtype *sEL, dtype *cEL,
                         size_t n_val)
{
    const long long n_val_ll = (long long)n_val;
    if (len)
    {
#pragma omp parallel for schedule(static) if (n_val_ll >= QD_OMP_THRESHOLD)
        for (long long i = 0; i < n_val_ll; ++i)
        {
            const dtype azi = az[i], eli = el[i], leni = len[i];
            dtype sa = std::sin(azi), ca = std::cos(azi);
            dtype se = std::sin(eli), ce = std::cos(eli);
            x[i] = ce * ca * leni;
            y[i] = ce * sa * leni;
            z[i] = se * leni;
            if (sAZ)
                sAZ[i] = sa;
            if (cAZ)
                cAZ[i] = ca;
            if (sEL)
                sEL[i] = se;
            if (cEL)
                cEL[i] = ce;
        }
    }
    else
    {
#pragma omp parallel for schedule(static) if (n_val_ll >= QD_OMP_THRESHOLD)
        for (long long i = 0; i < n_val_ll; ++i)
        {
            const dtype azi = az[i], eli = el[i];
            dtype sa = std::sin(azi), ca = std::cos(azi);
            dtype se = std::sin(eli), ce = std::cos(eli);
            x[i] = ce * ca;
            y[i] = ce * sa;
            z[i] = se;
            if (sAZ)
                sAZ[i] = sa;
            if (cAZ)
                cAZ[i] = ca;
            if (sEL)
                sEL[i] = se;
            if (cEL)
                cEL[i] = ce;
        }
    }
}

template <typename dtype>
void qd_CART2GEO_GENERIC(const dtype * x, const dtype * y, const dtype * z,
                         dtype * az, dtype * el, dtype * len, size_t n_val)
{
    const long long n_val_ll = (long long)n_val;
#pragma omp parallel for schedule(static) if (n_val_ll >= QD_OMP_THRESHOLD)
    for (long long i = 0; i < n_val_ll; ++i)
    {
        dtype xi = x[i], yi = y[i], zi = z[i];
        dtype r = std::sqrt(xi * xi + yi * yi + zi * zi);
        if (len)
            len[i] = r;
        az[i] = std::atan2(yi, xi);
        dtype zn = r > dtype(0) ? zi / r : dtype(0);
        zn = zn > dtype(1) ? dtype(1) : (zn < dtype(-1) ? dtype(-1) : zn);
        el[i] = std::asin(zn);
    }
}

#endif