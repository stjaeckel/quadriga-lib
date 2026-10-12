// Variant of qd_SOS_AVX2 with a branch-free accumulation for full groups of generators: the partial sums
// are named variables (registers), not an array indexed by a loop with a run-time bound. The arithmetic per
// (position, generator) is unchanged: float partial sums over 16 sinusoids, cos term then sin term, added in
// double. Template parameter: generators per full group (4 or 8). Remaining generators use the 1-generator path.
#include <immintrin.h>
#include <cstddef>
#include <vector>
#include "fastmath_avx2.h"
#define VEC 8
#define CHUNK 16

template <int G>
static inline void acc_group(const float *C, const float *S, const float *w, size_t n_coeff, __m256d *al, __m256d *ah)
{
    __m256d l0 = _mm256_setzero_pd(), l1 = l0, l2 = l0, l3 = l0, l4 = l0, l5 = l0, l6 = l0, l7 = l0;
    __m256d h0 = l0, h1 = l0, h2 = l0, h3 = l0, h4 = l0, h5 = l0, h6 = l0, h7 = l0;
    for (size_t n0 = 0; n0 < n_coeff; n0 += CHUNK)
    {
        const size_t n1 = (n_coeff - n0 < CHUNK) ? n_coeff : n0 + CHUNK;
        __m256 p0 = _mm256_setzero_ps(), p1 = p0, p2 = p0, p3 = p0, p4 = p0, p5 = p0, p6 = p0, p7 = p0;
        for (size_t n = n0; n < n1; ++n)
        {
            const __m256 c8 = _mm256_load_ps(C + n * VEC), s8 = _mm256_load_ps(S + n * VEC);
            p0 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(w + n * 2 * G + 1), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(w + n * 2 * G + 0), p0));
            p1 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(w + n * 2 * G + 3), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(w + n * 2 * G + 2), p1));
            p2 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(w + n * 2 * G + 5), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(w + n * 2 * G + 4), p2));
            p3 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(w + n * 2 * G + 7), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(w + n * 2 * G + 6), p3));
            if (G > 4)
            {
                p4 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(w + n * 2 * G + 9), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(w + n * 2 * G + 8), p4));
                p5 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(w + n * 2 * G + 11), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(w + n * 2 * G + 10), p5));
                p6 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(w + n * 2 * G + 13), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(w + n * 2 * G + 12), p6));
                p7 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(w + n * 2 * G + 15), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(w + n * 2 * G + 14), p7));
            }
        }
#define ADD(k) l##k = _mm256_add_pd(l##k, _mm256_cvtps_pd(_mm256_castps256_ps128(p##k))), h##k = _mm256_add_pd(h##k, _mm256_cvtps_pd(_mm256_extractf128_ps(p##k, 1)))
        ADD(0); ADD(1); ADD(2); ADD(3);
        if (G > 4) { ADD(4); ADD(5); ADD(6); ADD(7); }
    }
    al[0] = l0, al[1] = l1, al[2] = l2, al[3] = l3, ah[0] = h0, ah[1] = h1, ah[2] = h2, ah[3] = h3;
    if (G > 4) al[4] = l4, al[5] = l5, al[6] = l6, al[7] = l7, ah[4] = h4, ah[5] = h5, ah[6] = h6, ah[7] = h7;
}


// Tiled kernel: generators are padded to a multiple of 8 with zero weights; positions are processed in tiles of
// 'tile' blocks of 8; within a tile the loop over generator groups is the outer loop, so the weights of a group
// (16 * n_coeff floats) are read from the first-level cache for all blocks of the tile.
void var_SOS_AVX2_T8(const double *fx, const double *fy, const double *fz, const double *wc, const double *ws, const size_t n_coeff, const size_t n_gen,
                     const double *px, const double *py, const double *pz, const size_t n_pos, double *val)
{
    const int G = 8;
    const long long n_block = (long long)((n_pos + VEC - 1) / VEC);
    const size_t n_grp = (n_gen + G - 1) / G;
    size_t tile = (128 * 1024) / (n_coeff * VEC * sizeof(float) * 2);
    tile = tile < 1 ? 1 : (tile > 16 ? 16 : tile);
    const long long n_tile = (n_block + (long long)tile - 1) / (long long)tile;

    float *wp = (float *)_mm_malloc(n_grp * n_coeff * 2 * G * sizeof(float), 64);
    for (size_t gr = 0; gr < n_grp; ++gr)
        for (size_t n = 0; n < n_coeff; ++n)
            for (size_t j = 0; j < (size_t)G; ++j)
            {
                const size_t g = gr * G + j;
                wp[(gr * n_coeff + n) * 2 * G + 2 * j] = (g < n_gen) ? (float)wc[g * n_coeff + n] : 0.0f;
                wp[(gr * n_coeff + n) * 2 * G + 2 * j + 1] = (g < n_gen) ? (float)ws[g * n_coeff + n] : 0.0f;
            }
#pragma omp parallel
    {
        float *C = (float *)_mm_malloc(tile * n_coeff * VEC * sizeof(float), 64), *S = (float *)_mm_malloc(tile * n_coeff * VEC * sizeof(float), 64);
        const __m256d two_pi = _mm256_set1_pd(6.283185307179586);
#pragma omp for
        for (long long i_tile = 0; i_tile < n_tile; ++i_tile)
        {
            const size_t b0 = (size_t)i_tile * tile, nb = ((size_t)n_block - b0 < tile) ? (size_t)n_block - b0 : tile;
            for (size_t b = 0; b < nb; ++b)
            {
                const size_t i_start = (b0 + b) * VEC, n_valid = (n_pos - i_start < VEC) ? n_pos - i_start : VEC;
                double bx[VEC] = {0.0}, by[VEC] = {0.0}, bz[VEC] = {0.0};
                for (size_t k = 0; k < n_valid; ++k)
                    bx[k] = px[i_start + k], by[k] = py[i_start + k], bz[k] = pz[i_start + k];
                const __m256d xl = _mm256_loadu_pd(bx), xh = _mm256_loadu_pd(bx + 4), yl = _mm256_loadu_pd(by), yh = _mm256_loadu_pd(by + 4), zl = _mm256_loadu_pd(bz), zh = _mm256_loadu_pd(bz + 4);
                float *Cb = C + b * n_coeff * VEC, *Sb = S + b * n_coeff * VEC;
                for (size_t n = 0; n < n_coeff; ++n)
                {
                    const __m256d vx = _mm256_set1_pd(fx[n]), vy = _mm256_set1_pd(fy[n]), vz = _mm256_set1_pd(fz[n]);
                    __m256d tl = _mm256_fmadd_pd(vx, xl, _mm256_fmadd_pd(vy, yl, _mm256_mul_pd(vz, zl)));
                    __m256d th = _mm256_fmadd_pd(vx, xh, _mm256_fmadd_pd(vy, yh, _mm256_mul_pd(vz, zh)));
                    tl = _mm256_sub_pd(tl, _mm256_round_pd(tl, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                    th = _mm256_sub_pd(th, _mm256_round_pd(th, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                    __m256 s8, c8;
                    _fm256_sincos256_pd(_mm256_mul_pd(tl, two_pi), _mm256_mul_pd(th, two_pi), &s8, &c8);
                    _mm256_store_ps(Cb + n * VEC, c8), _mm256_store_ps(Sb + n * VEC, s8);
                }
            }
            __m256d al[8], ah[8];
            for (size_t gr = 0; gr < n_grp; ++gr)
            {
                const size_t ng = (n_gen - gr * G < (size_t)G) ? n_gen - gr * G : (size_t)G;
                for (size_t b = 0; b < nb; ++b)
                {
                    acc_group<8>(C + b * n_coeff * VEC, S + b * n_coeff * VEC, wp + gr * n_coeff * 2 * G, n_coeff, al, ah);
                    const size_t i_start = (b0 + b) * VEC, n_valid = (n_pos - i_start < VEC) ? n_pos - i_start : VEC;
                    for (size_t j = 0; j < ng; ++j)
                    {
                        double *out = &val[(gr * G + j) * n_pos + i_start];
                        if (n_valid == VEC) { _mm256_storeu_pd(out, al[j]), _mm256_storeu_pd(out + 4, ah[j]); }
                        else { double tmp[VEC]; _mm256_storeu_pd(tmp, al[j]), _mm256_storeu_pd(tmp + 4, ah[j]); for (size_t k = 0; k < n_valid; ++k) out[k] = tmp[k]; }
                    }
                }
            }
        }
        _mm_free(C), _mm_free(S);
    }
    _mm_free(wp);
}
