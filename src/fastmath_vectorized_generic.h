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

// GENERIC reference / fallback vectorized math functions
// - Vector lengths must be multiple of 1
// - Input / output vectors do not need to be aligned
// - No sanity checks - incorrect argument formatting leads to undefined behavior

#ifndef quadriga_lib_fastmath_vec_generic_H
#define quadriga_lib_fastmath_vec_generic_H

#include <math.h>
#include <stddef.h>
#include <limits.h>
#include "slerp.h"

// Scalar double-precision Cody–Waite range reduction: map x into [-pi, pi].
// Two-constant split: CW_C1 + CW_C2 = 2*pi, with CW_C1 exactly representable.
// Matches the AVX2 _fm256_range_reduce_2pi_pd helper.
static const double QD_INV_TWO_PI = 0.15915494309189533577;  // 1 / (2*pi)
static const double QD_CW_C1 = 6.28125;                     // high bits of 2*pi
static const double QD_CW_C2 = 1.9353071795864769253e-03;   // 2*pi - CW_C1

static inline double qd_range_reduce_2pi(double x)
{
    double n = round(x * QD_INV_TWO_PI);    // nearest integer multiple of 2*pi
    x -= n * QD_CW_C1;                      // exact by Sterbenz when n*C1 ≈ x
    x -= n * QD_CW_C2;                      // mop up low bits
    return x;
}

#ifndef QD_OMP_THRESHOLD
#define QD_OMP_THRESHOLD 4096 // iterations of the inner loop before parallelizing
#endif

template <typename dtype>
void qd_SINCOS_GENERIC(const dtype *__restrict x,
                       float *__restrict s,
                       float *__restrict c,
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
void qd_SIN_GENERIC(const dtype *__restrict x,
                    float *__restrict s,
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
void qd_COS_GENERIC(const dtype *__restrict x,
                    float *__restrict c,
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
void qd_ASIN_GENERIC(const dtype *__restrict x,
                     float *__restrict s,
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
void qd_ACOS_GENERIC(const dtype *__restrict x,
                     float *__restrict c,
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
void qd_SLERP_GENERIC(const dtype *__restrict Ar, const dtype *__restrict Ai,
                       const dtype *__restrict Br, const dtype *__restrict Bi,
                       const dtype *__restrict w,
                       float *__restrict Xr, float *__restrict Xi,
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

#endif