// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <immintrin.h>
#include <cstddef>
#include <new>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "quadriga_lib_avx2_functions.hpp"
#include "fastmath_avx2.h"

// Number of positions per block (one AVX2 vector) and number of sinusoids summed in single precision
// before the partial sum is added in double precision
#define QD_SOS_VEC 8
#define QD_SOS_CHUNK 16

// Memory for the cosines and sines of one tile of position blocks, per thread. A tile is evaluated for one
// group of generators after the other, so that the weights of a group are read from the first-level cache
// for all blocks of the tile while the cosines and sines stay in the second-level cache.
#define QD_SOS_TILE_BYTES 131072
#define QD_SOS_TILE_MAX 16

// Sum of the sinusoids for a group of G generators (1 to 8) at the 8 positions of one block
// - C, S: cosine and sine of each sinusoid at the 8 positions, Size [8, n_coeff], 32-byte aligned
// - w: weights of the group, for each sinusoid G pairs { a·cos(phase), a·sin(phase) }, Size [2·G, n_coeff]
// - acc_l, acc_h: sums at positions 0-3 and 4-7 for each generator of the group, length G
// Arithmetic per (position, generator), the same for every G: single-precision partial sums over 16
// sinusoids, cosine term then sine term in the order of the sinusoids, added in double precision.
// The partial sums are named variables and G is a compile-time constant, so they are kept in registers
// and the accumulation has no branch on the size of the group.
template <int G>
static inline void qd_sos_accumulate(const float *C, const float *S, const float *w, const size_t n_coeff,
                                     __m256d *acc_l, __m256d *acc_h)
{
    __m256d l0 = _mm256_setzero_pd(), l1 = l0, l2 = l0, l3 = l0, l4 = l0, l5 = l0, l6 = l0, l7 = l0;
    __m256d h0 = l0, h1 = l0, h2 = l0, h3 = l0, h4 = l0, h5 = l0, h6 = l0, h7 = l0;

#define QD_SOS_TERM(k) p##k = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(wn + 2 * k + 1), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(wn + 2 * k), p##k))
#define QD_SOS_ADD(k) l##k = _mm256_add_pd(l##k, _mm256_cvtps_pd(_mm256_castps256_ps128(p##k))), \
                      h##k = _mm256_add_pd(h##k, _mm256_cvtps_pd(_mm256_extractf128_ps(p##k, 1)))

    for (size_t n0 = 0; n0 < n_coeff; n0 += QD_SOS_CHUNK)
    {
        const size_t n1 = (n_coeff - n0 < QD_SOS_CHUNK) ? n_coeff : n0 + QD_SOS_CHUNK;

        __m256 p0 = _mm256_setzero_ps(), p1 = p0, p2 = p0, p3 = p0, p4 = p0, p5 = p0, p6 = p0, p7 = p0;
        for (size_t n = n0; n < n1; ++n)
        {
            const __m256 c8 = _mm256_load_ps(C + n * QD_SOS_VEC), s8 = _mm256_load_ps(S + n * QD_SOS_VEC);
            const float *wn = w + n * 2 * G;
            QD_SOS_TERM(0);
            if (G > 1)
                QD_SOS_TERM(1);
            if (G > 2)
                QD_SOS_TERM(2);
            if (G > 3)
                QD_SOS_TERM(3);
            if (G > 4)
                QD_SOS_TERM(4);
            if (G > 5)
                QD_SOS_TERM(5);
            if (G > 6)
                QD_SOS_TERM(6);
            if (G > 7)
                QD_SOS_TERM(7);
        }

        QD_SOS_ADD(0);
        if (G > 1)
            QD_SOS_ADD(1);
        if (G > 2)
            QD_SOS_ADD(2);
        if (G > 3)
            QD_SOS_ADD(3);
        if (G > 4)
            QD_SOS_ADD(4);
        if (G > 5)
            QD_SOS_ADD(5);
        if (G > 6)
            QD_SOS_ADD(6);
        if (G > 7)
            QD_SOS_ADD(7);
    }

#undef QD_SOS_TERM
#undef QD_SOS_ADD

    acc_l[0] = l0, acc_h[0] = h0;
    if (G > 1)
        acc_l[1] = l1, acc_h[1] = h1;
    if (G > 2)
        acc_l[2] = l2, acc_h[2] = h2;
    if (G > 3)
        acc_l[3] = l3, acc_h[3] = h3;
    if (G > 4)
        acc_l[4] = l4, acc_h[4] = h4;
    if (G > 5)
        acc_l[5] = l5, acc_h[5] = h5;
    if (G > 6)
        acc_l[6] = l6, acc_h[6] = h6;
    if (G > 7)
        acc_l[7] = l7, acc_h[7] = h7;
}

// One group of G generators for the n_block blocks of a tile
// - C, S: cosine and sine for the blocks of the tile, Size [8, n_coeff, n_block]
// - i_first: index of the first position of the tile; val: output column of the first generator of the group
template <int G>
static inline void qd_sos_group(const float *C, const float *S, const float *w, const size_t n_coeff,
                                const size_t n_block, const size_t i_first, const size_t n_pos, double *val)
{
    const size_t n_slice = n_coeff * QD_SOS_VEC;
    for (size_t b = 0; b < n_block; ++b)
    {
        __m256d acc_l[8], acc_h[8];
        qd_sos_accumulate<G>(C + b * n_slice, S + b * n_slice, w, n_coeff, acc_l, acc_h);

        const size_t i_start = i_first + b * QD_SOS_VEC;
        if (n_pos - i_start >= QD_SOS_VEC)
            for (int j = 0; j < G; ++j)
            {
                double *out = val + (size_t)j * n_pos + i_start;
                _mm256_storeu_pd(out, acc_l[j]);
                _mm256_storeu_pd(out + 4, acc_h[j]);
            }
        else // Incomplete last block: the padding is discarded
            for (int j = 0; j < G; ++j)
            {
                double tmp[QD_SOS_VEC];
                _mm256_storeu_pd(tmp, acc_l[j]);
                _mm256_storeu_pd(tmp + 4, acc_h[j]);
                double *out = val + (size_t)j * n_pos + i_start;
                for (size_t k = 0; k < n_pos - i_start; ++k)
                    out[k] = tmp[k];
            }
    }
}

// AVX2 accelerated implementation of the sum-of-sinusoids evaluation
// Vector lanes run over positions. All operations are lane-wise, so a value does not depend on the lane
// or on the other positions of its block; an incomplete last block is padded and the padding discarded.
// The generators are split into groups of 8, followed by one group with the remaining 1 to 7, so that no
// generator is padded. The value of a generator does not depend on the group it falls into. The weights
// of a group start at 2 · n_coeff · (index of its first generator) in the packed array.
void qd_SOS_AVX2(const double *fx, const double *fy, const double *fz,
                 const double *wc, const double *ws,
                 const size_t n_coeff,
                 const size_t n_gen,
                 const double *px, const double *py, const double *pz,
                 const size_t n_pos,
                 double *val)
{
    if (n_coeff == 0 || n_gen == 0 || n_pos == 0)
        return;

    const size_t n_block = (n_pos + QD_SOS_VEC - 1) / QD_SOS_VEC;
    const size_t n_slice = n_coeff * QD_SOS_VEC; // Floats per block and table, a multiple of 8

    // Blocks per tile. With fewer than 8 generators all weights are small and a tile has no benefit.
    size_t tile = QD_SOS_TILE_BYTES / (2 * n_slice * sizeof(float));
    tile = (tile > QD_SOS_TILE_MAX) ? QD_SOS_TILE_MAX : tile;
    tile = (tile < 1 || n_gen < 8) ? 1 : tile;
    tile = (tile > n_block) ? n_block : tile;
    const long long n_tile = (long long)((n_block + tile - 1) / tile);

    // Buffers are allocated before the parallel region, so that a failed allocation throws where it can be
    // caught: the packed weights in single precision, and per thread the cosines and sines of one tile
#ifdef _OPENMP
    const int n_threads = omp_get_max_threads();
#else
    const int n_threads = 1;
#endif
    float *weights = (float *)_mm_malloc(2 * n_coeff * n_gen * sizeof(float), 32);
    float *buffer = (float *)_mm_malloc(2 * tile * n_slice * (size_t)n_threads * sizeof(float), 32);
    if (weights == nullptr || buffer == nullptr)
    {
        _mm_free(weights);
        _mm_free(buffer);
        throw std::bad_alloc();
    }

    for (size_t g0 = 0; g0 < n_gen; g0 += 8)
    {
        const size_t G = (n_gen - g0 < 8) ? n_gen - g0 : 8;
        float *w = weights + 2 * n_coeff * g0;
        for (size_t j = 0; j < G; ++j)
        {
            const double *pc = wc + (g0 + j) * n_coeff, *ps = ws + (g0 + j) * n_coeff;
            for (size_t n = 0; n < n_coeff; ++n)
                w[n * 2 * G + 2 * j] = (float)pc[n], w[n * 2 * G + 2 * j + 1] = (float)ps[n];
        }
    }

#pragma omp parallel num_threads(n_threads)
    {
        // Cosine and sine of all sinusoids at the positions of the blocks of one tile, Size [8, n_coeff, tile]
#ifdef _OPENMP
        float *C = buffer + 2 * tile * n_slice * (size_t)omp_get_thread_num(), *S = C + tile * n_slice;
#else
        float *C = buffer, *S = C + tile * n_slice;
#endif

        const __m256d two_pi = _mm256_set1_pd(6.283185307179586);

#pragma omp for
        for (long long i_tile = 0; i_tile < n_tile; ++i_tile)
        {
            const size_t b_first = (size_t)i_tile * tile;
            const size_t nb = (n_block - b_first < tile) ? n_block - b_first : tile;

            for (size_t b = 0; b < nb; ++b)
            {
                const size_t i_start = (b_first + b) * QD_SOS_VEC;
                const size_t n_valid = (n_pos - i_start < QD_SOS_VEC) ? n_pos - i_start : QD_SOS_VEC;

                double bx[QD_SOS_VEC] = {0.0}, by[QD_SOS_VEC] = {0.0}, bz[QD_SOS_VEC] = {0.0};
                for (size_t k = 0; k < n_valid; ++k)
                    bx[k] = px[i_start + k], by[k] = py[i_start + k], bz[k] = pz[i_start + k];

                const __m256d xl = _mm256_loadu_pd(bx), xh = _mm256_loadu_pd(bx + 4);
                const __m256d yl = _mm256_loadu_pd(by), yh = _mm256_loadu_pd(by + 4);
                const __m256d zl = _mm256_loadu_pd(bz), zh = _mm256_loadu_pd(bz + 4);

                float *Cb = C + b * n_slice, *Sb = S + b * n_slice;
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
                    _mm256_store_ps(Cb + n * QD_SOS_VEC, c8);
                    _mm256_store_ps(Sb + n * QD_SOS_VEC, s8);
                }
            }

            const size_t i_first = b_first * QD_SOS_VEC;
            for (size_t g0 = 0; g0 < n_gen; g0 += 8)
            {
                const float *w = weights + 2 * n_coeff * g0;
                double *out = val + g0 * n_pos;
                switch (n_gen - g0)
                {
                case 1:
                    qd_sos_group<1>(C, S, w, n_coeff, nb, i_first, n_pos, out);
                    break;
                case 2:
                    qd_sos_group<2>(C, S, w, n_coeff, nb, i_first, n_pos, out);
                    break;
                case 3:
                    qd_sos_group<3>(C, S, w, n_coeff, nb, i_first, n_pos, out);
                    break;
                case 4:
                    qd_sos_group<4>(C, S, w, n_coeff, nb, i_first, n_pos, out);
                    break;
                case 5:
                    qd_sos_group<5>(C, S, w, n_coeff, nb, i_first, n_pos, out);
                    break;
                case 6:
                    qd_sos_group<6>(C, S, w, n_coeff, nb, i_first, n_pos, out);
                    break;
                case 7:
                    qd_sos_group<7>(C, S, w, n_coeff, nb, i_first, n_pos, out);
                    break;
                default:
                    qd_sos_group<8>(C, S, w, n_coeff, nb, i_first, n_pos, out);
                }
            }
        }
    }

    _mm_free(weights);
    _mm_free(buffer);
}
