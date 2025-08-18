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

// GENERIC reference / fallback vectorized math functions
// - Vector lengths must be multiple of 1
// - Input / output vectors do not need to be aligned
// - No sanity checks - incorrect argument formatting leads to undefined behavior

#ifndef quadriga_lib_fastmath_vec_generic_H
#define quadriga_lib_fastmath_vec_generic_H

#include <math.h>
#include <stddef.h>
#include <limits.h>

#ifndef QD_OMP_THRESHOLD
#define QD_OMP_THRESHOLD 4096 // iterations of the inner loop before parallelizing
#endif

void qd_SINCOS_GENERIC(const float *__restrict x,
                       float *__restrict s,
                       float *__restrict c,
                       size_t n_val)
{
    size_t done = 0;

    while (done < n_val)
    {
        size_t blk = n_val - done;
        if (blk > (size_t)INT_MAX)
            blk = (size_t)INT_MAX; // MSVC OpenMP requires signed int
        const int iters = (int)blk;

        const float *__restrict xb = x + done;
        float *__restrict sb = s + done;
        float *__restrict cb = c + done;

#pragma omp parallel for schedule(static) if (iters >= QD_OMP_THRESHOLD)
        for (int i = 0; i < iters; ++i)
        {
            const float xi = xb[i];
            sb[i] = sinf(xi);
            cb[i] = cosf(xi);
        }

        done += blk;
    }
}

void qd_SIN_GENERIC(const float *__restrict x,
                    float *__restrict s,
                    size_t n_val)
{
    size_t done = 0;

    while (done < n_val)
    {
        size_t blk = n_val - done;
        if (blk > (size_t)INT_MAX)
            blk = (size_t)INT_MAX; // MSVC OpenMP requires signed int
        const int iters = (int)blk;

        const float *__restrict xb = x + done;
        float *__restrict sb = s + done;

#pragma omp parallel for schedule(static) if (iters >= QD_OMP_THRESHOLD)
        for (int i = 0; i < iters; ++i)
        {
            const float xi = xb[i];
            sb[i] = sinf(xi);
        }

        done += blk;
    }
}

void qd_COS_GENERIC(const float *__restrict x,
                    float *__restrict c,
                    size_t n_val)
{
    size_t done = 0;

    while (done < n_val)
    {
        size_t blk = n_val - done;
        if (blk > (size_t)INT_MAX)
            blk = (size_t)INT_MAX; // MSVC OpenMP requires signed int
        const int iters = (int)blk;

        const float *__restrict xb = x + done;
        float *__restrict cb = c + done;

#pragma omp parallel for schedule(static) if (iters >= QD_OMP_THRESHOLD)
        for (int i = 0; i < iters; ++i)
        {
            const float xi = xb[i];
            cb[i] = cosf(xi);
        }

        done += blk;
    }
}

#endif
