// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <immintrin.h>
#include <cstddef>
#include <vector>

#include "quadriga_lib_avx2_functions.hpp"
#include "fastmath_avx2.h"

// Number of positions per block (one AVX2 vector), number of generators accumulated together,
// number of sinusoids summed in single precision before the partial sum is added in double precision
#define QD_SOS_VEC 8
#define QD_SOS_GEN 4
#define QD_SOS_CHUNK 16

// AVX2 accelerated implementation of the sum-of-sinusoids evaluation
// Vector lanes run over positions. All operations are lane-wise, so a value does not depend on the lane
// or on the other positions of its block; an incomplete last block is padded and the padding discarded.
void qd_SOS_AVX2(const double *fx, const double *fy, const double *fz,
                 const double *wc, const double *ws,
                 const size_t n_coeff,
                 const size_t n_gen,
                 const double *px, const double *py, const double *pz,
                 const size_t n_pos,
                 double *val)
{
    const long long n_block = (long long)((n_pos + QD_SOS_VEC - 1) / QD_SOS_VEC);

    // Weights in single precision
    std::vector<float> wcf(n_coeff * n_gen), wsf(n_coeff * n_gen);
    for (size_t i = 0; i < n_coeff * n_gen; ++i)
        wcf[i] = (float)wc[i], wsf[i] = (float)ws[i];

#pragma omp parallel
    {
        // Cosine and sine of all sinusoids at the positions of one block, Size [8, n_coeff]
        float *C = (float *)_mm_malloc(n_coeff * QD_SOS_VEC * sizeof(float), 32);
        float *S = (float *)_mm_malloc(n_coeff * QD_SOS_VEC * sizeof(float), 32);

        const __m256d two_pi = _mm256_set1_pd(6.283185307179586);

#pragma omp for
        for (long long i_block = 0; i_block < n_block; ++i_block)
        {
            const size_t i_start = (size_t)i_block * QD_SOS_VEC;
            const size_t n_valid = (n_pos - i_start < QD_SOS_VEC) ? n_pos - i_start : QD_SOS_VEC;

            double bx[QD_SOS_VEC] = {0.0}, by[QD_SOS_VEC] = {0.0}, bz[QD_SOS_VEC] = {0.0};
            for (size_t k = 0; k < n_valid; ++k)
                bx[k] = px[i_start + k], by[k] = py[i_start + k], bz[k] = pz[i_start + k];

            const __m256d xl = _mm256_loadu_pd(bx), xh = _mm256_loadu_pd(bx + 4);
            const __m256d yl = _mm256_loadu_pd(by), yh = _mm256_loadu_pd(by + 4);
            const __m256d zl = _mm256_loadu_pd(bz), zh = _mm256_loadu_pd(bz + 4);

            for (size_t n = 0; n < n_coeff; ++n)
            {
                const __m256d vx = _mm256_set1_pd(fx[n]), vy = _mm256_set1_pd(fy[n]), vz = _mm256_set1_pd(fz[n]);

                // Phase in cycles, reduced to [-0.5, 0.5] in double precision
                __m256d tl = _mm256_fmadd_pd(vx, xl, _mm256_fmadd_pd(vy, yl, _mm256_mul_pd(vz, zl)));
                __m256d th = _mm256_fmadd_pd(vx, xh, _mm256_fmadd_pd(vy, yh, _mm256_mul_pd(vz, zh)));
                tl = _mm256_sub_pd(tl, _mm256_round_pd(tl, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                th = _mm256_sub_pd(th, _mm256_round_pd(th, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

                __m256 s8, c8;
                _fm256_sincos256_pd(_mm256_mul_pd(tl, two_pi), _mm256_mul_pd(th, two_pi), &s8, &c8);
                _mm256_store_ps(C + n * QD_SOS_VEC, c8);
                _mm256_store_ps(S + n * QD_SOS_VEC, s8);
            }

            for (size_t g0 = 0; g0 < n_gen; g0 += QD_SOS_GEN)
            {
                const size_t ng = (n_gen - g0 < QD_SOS_GEN) ? n_gen - g0 : QD_SOS_GEN;

                __m256d acc_l[QD_SOS_GEN], acc_h[QD_SOS_GEN];
                for (size_t j = 0; j < QD_SOS_GEN; ++j)
                    acc_l[j] = _mm256_setzero_pd(), acc_h[j] = _mm256_setzero_pd();

                for (size_t n0 = 0; n0 < n_coeff; n0 += QD_SOS_CHUNK)
                {
                    const size_t n1 = (n_coeff - n0 < QD_SOS_CHUNK) ? n_coeff : n0 + QD_SOS_CHUNK;

                    __m256 part[QD_SOS_GEN];
                    for (size_t j = 0; j < QD_SOS_GEN; ++j)
                        part[j] = _mm256_setzero_ps();

                    for (size_t n = n0; n < n1; ++n)
                    {
                        const __m256 c8 = _mm256_load_ps(C + n * QD_SOS_VEC);
                        const __m256 s8 = _mm256_load_ps(S + n * QD_SOS_VEC);
                        for (size_t j = 0; j < ng; ++j)
                        {
                            const size_t iw = (g0 + j) * n_coeff + n;
                            part[j] = _mm256_fmadd_ps(c8, _mm256_set1_ps(wcf[iw]), part[j]);
                            part[j] = _mm256_fnmadd_ps(s8, _mm256_set1_ps(wsf[iw]), part[j]);
                        }
                    }

                    for (size_t j = 0; j < ng; ++j)
                    {
                        acc_l[j] = _mm256_add_pd(acc_l[j], _mm256_cvtps_pd(_mm256_castps256_ps128(part[j])));
                        acc_h[j] = _mm256_add_pd(acc_h[j], _mm256_cvtps_pd(_mm256_extractf128_ps(part[j], 1)));
                    }
                }

                for (size_t j = 0; j < ng; ++j)
                {
                    double tmp[QD_SOS_VEC];
                    _mm256_storeu_pd(tmp, acc_l[j]);
                    _mm256_storeu_pd(tmp + 4, acc_h[j]);
                    double *out = &val[(g0 + j) * n_pos + i_start];
                    for (size_t k = 0; k < n_valid; ++k)
                        out[k] = tmp[k];
                }
            }
        }

        _mm_free(C);
        _mm_free(S);
    }
}
