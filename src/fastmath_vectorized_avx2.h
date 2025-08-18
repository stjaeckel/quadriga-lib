// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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

// AVX2 accelerated vectorized math functions
// - Vector lengths must be multiple of 8
// - Input / output vectors do not need to be aligned
// - No sanity checks - incorrect argument formatting leads to undefined behavior

#ifndef quadriga_lib_fastmath_vec_avx2_H
#define quadriga_lib_fastmath_vec_avx2_H

#include "fastmath_avx2.h"

#include <immintrin.h>
#include <stddef.h>
#include <limits.h>

#ifndef QD_OMP_THRESHOLD
#define QD_OMP_THRESHOLD 4096 // iterations of the inner loop before parallelizing
#endif

void qd_SINCOS_AVX2(const float *__restrict x,
                    float *__restrict s,
                    float *__restrict c,
                    size_t n_val) // multiple of 8
{
    const size_t n_vec = n_val >> 3; // number of 8-float vectors
    size_t done = 0;

    while (done < n_vec)
    {
        size_t blk = n_vec - done;
        if (blk > (size_t)INT_MAX)
            blk = (size_t)INT_MAX; // MSVC OpenMP 'for' needs signed int
        const int iters = (int)blk;

        const float *__restrict xb = x + (done << 3);
        float *__restrict sb = s + (done << 3);
        float *__restrict cb = c + (done << 3);

#pragma omp parallel for schedule(static) if (iters >= QD_OMP_THRESHOLD)
        for (int i = 0; i < iters; ++i)
        {
            __m256 xv, sv, cv;
            xv = _mm256_loadu_ps(xb + ((size_t)i << 3)); // no alignment required
            _fm256_sincos256_ps(xv, &sv, &cv);
            _mm256_storeu_ps(sb + ((size_t)i << 3), sv);
            _mm256_storeu_ps(cb + ((size_t)i << 3), cv);
        }

        done += blk;
    }
}

void qd_SIN_AVX2(const float *__restrict x,
                 float *__restrict s,
                 size_t n_val) // multiple of 8
{
    const size_t n_vec = n_val >> 3; // number of 8-float vectors
    size_t done = 0;

    while (done < n_vec)
    {
        size_t blk = n_vec - done;
        if (blk > (size_t)INT_MAX)
            blk = (size_t)INT_MAX; // MSVC OpenMP 'for' needs signed int
        const int iters = (int)blk;

        const float *__restrict xb = x + (done << 3);
        float *__restrict sb = s + (done << 3);

#pragma omp parallel for schedule(static) if (iters >= QD_OMP_THRESHOLD)
        for (int i = 0; i < iters; ++i)
        {
            __m256 xv, sv, cv;
            xv = _mm256_loadu_ps(xb + ((size_t)i << 3)); // no alignment required
            _fm256_sincos256_ps(xv, &sv, &cv);
            _mm256_storeu_ps(sb + ((size_t)i << 3), sv);
        }

        done += blk;
    }
}

void qd_COS_AVX2(const float *__restrict x,
                 float *__restrict c,
                 size_t n_val) // multiple of 8
{
    const size_t n_vec = n_val >> 3; // number of 8-float vectors
    size_t done = 0;

    while (done < n_vec)
    {
        size_t blk = n_vec - done;
        if (blk > (size_t)INT_MAX)
            blk = (size_t)INT_MAX; // MSVC OpenMP 'for' needs signed int
        const int iters = (int)blk;

        const float *__restrict xb = x + (done << 3);
        float *__restrict cb = c + (done << 3);

#pragma omp parallel for schedule(static) if (iters >= QD_OMP_THRESHOLD)
        for (int i = 0; i < iters; ++i)
        {
            __m256 xv, sv, cv;
            xv = _mm256_loadu_ps(xb + ((size_t)i << 3)); // no alignment required
            _fm256_sincos256_ps(xv, &sv, &cv);
            _mm256_storeu_ps(cb + ((size_t)i << 3), cv);
        }

        done += blk;
    }
}

#endif
