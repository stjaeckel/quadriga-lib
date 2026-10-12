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
static inline void acc_group(const float *C, const float *S, const float *wc, const float *ws, size_t n_coeff, __m256d *al, __m256d *ah)
{
    __m256d l0 = _mm256_setzero_pd(), l1 = l0, l2 = l0, l3 = l0, l4 = l0, l5 = l0, l6 = l0, l7 = l0;
    __m256d h0 = l0, h1 = l0, h2 = l0, h3 = l0, h4 = l0, h5 = l0, h6 = l0, h7 = l0;
    const float *c0 = wc, *c1 = wc + n_coeff, *c2 = wc + 2 * n_coeff, *c3 = wc + 3 * n_coeff;
    const float *s0 = ws, *s1 = ws + n_coeff, *s2 = ws + 2 * n_coeff, *s3 = ws + 3 * n_coeff;
    const float *c4 = wc + (G > 4 ? 4 : 0) * n_coeff, *c5 = wc + (G > 4 ? 5 : 0) * n_coeff, *c6 = wc + (G > 4 ? 6 : 0) * n_coeff, *c7 = wc + (G > 4 ? 7 : 0) * n_coeff;
    const float *s4 = ws + (G > 4 ? 4 : 0) * n_coeff, *s5 = ws + (G > 4 ? 5 : 0) * n_coeff, *s6 = ws + (G > 4 ? 6 : 0) * n_coeff, *s7 = ws + (G > 4 ? 7 : 0) * n_coeff;
    for (size_t n0 = 0; n0 < n_coeff; n0 += CHUNK)
    {
        const size_t n1 = (n_coeff - n0 < CHUNK) ? n_coeff : n0 + CHUNK;
        __m256 p0 = _mm256_setzero_ps(), p1 = p0, p2 = p0, p3 = p0, p4 = p0, p5 = p0, p6 = p0, p7 = p0;
        for (size_t n = n0; n < n1; ++n)
        {
            const __m256 c8 = _mm256_load_ps(C + n * VEC), s8 = _mm256_load_ps(S + n * VEC);
            p0 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(s0 + n), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(c0 + n), p0));
            p1 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(s1 + n), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(c1 + n), p1));
            p2 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(s2 + n), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(c2 + n), p2));
            p3 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(s3 + n), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(c3 + n), p3));
            if (G > 4)
            {
                p4 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(s4 + n), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(c4 + n), p4));
                p5 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(s5 + n), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(c5 + n), p5));
                p6 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(s6 + n), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(c6 + n), p6));
                p7 = _mm256_fnmadd_ps(s8, _mm256_broadcast_ss(s7 + n), _mm256_fmadd_ps(c8, _mm256_broadcast_ss(c7 + n), p7));
            }
        }
#define ADD(k) l##k = _mm256_add_pd(l##k, _mm256_cvtps_pd(_mm256_castps256_ps128(p##k))), h##k = _mm256_add_pd(h##k, _mm256_cvtps_pd(_mm256_extractf128_ps(p##k, 1)))
        ADD(0); ADD(1); ADD(2); ADD(3);
        if (G > 4) { ADD(4); ADD(5); ADD(6); ADD(7); }
    }
    al[0] = l0, al[1] = l1, al[2] = l2, al[3] = l3, ah[0] = h0, ah[1] = h1, ah[2] = h2, ah[3] = h3;
    if (G > 4) al[4] = l4, al[5] = l5, al[6] = l6, al[7] = l7, ah[4] = h4, ah[5] = h5, ah[6] = h6, ah[7] = h7;
}

static inline void acc_one(const float *C, const float *S, const float *wc, const float *ws, size_t n_coeff, __m256d *al, __m256d *ah)
{
    __m256d l0 = _mm256_setzero_pd(), h0 = l0;
    for (size_t n0 = 0; n0 < n_coeff; n0 += CHUNK)
    {
        const size_t n1 = (n_coeff - n0 < CHUNK) ? n_coeff : n0 + CHUNK;
        __m256 p0 = _mm256_setzero_ps();
        for (size_t n = n0; n < n1; ++n)
            p0 = _mm256_fnmadd_ps(_mm256_load_ps(S + n * VEC), _mm256_broadcast_ss(ws + n), _mm256_fmadd_ps(_mm256_load_ps(C + n * VEC), _mm256_broadcast_ss(wc + n), p0));
        l0 = _mm256_add_pd(l0, _mm256_cvtps_pd(_mm256_castps256_ps128(p0))), h0 = _mm256_add_pd(h0, _mm256_cvtps_pd(_mm256_extractf128_ps(p0, 1)));
    }
    al[0] = l0, ah[0] = h0;
}

template <int G>
static void kernel(const double *fx, const double *fy, const double *fz, const double *wc, const double *ws, const size_t n_coeff, const size_t n_gen,
                   const double *px, const double *py, const double *pz, const size_t n_pos, double *val)
{
    const long long n_block = (long long)((n_pos + VEC - 1) / VEC);
    std::vector<float> wcf(n_coeff * n_gen), wsf(n_coeff * n_gen);
    for (size_t i = 0; i < n_coeff * n_gen; ++i)
        wcf[i] = (float)wc[i], wsf[i] = (float)ws[i];
#pragma omp parallel
    {
        float *C = (float *)_mm_malloc(n_coeff * VEC * sizeof(float), 32), *S = (float *)_mm_malloc(n_coeff * VEC * sizeof(float), 32);
        const __m256d two_pi = _mm256_set1_pd(6.283185307179586);
#pragma omp for
        for (long long i_block = 0; i_block < n_block; ++i_block)
        {
            const size_t i_start = (size_t)i_block * VEC, n_valid = (n_pos - i_start < VEC) ? n_pos - i_start : VEC;
            double bx[VEC] = {0.0}, by[VEC] = {0.0}, bz[VEC] = {0.0};
            for (size_t k = 0; k < n_valid; ++k)
                bx[k] = px[i_start + k], by[k] = py[i_start + k], bz[k] = pz[i_start + k];
            const __m256d xl = _mm256_loadu_pd(bx), xh = _mm256_loadu_pd(bx + 4), yl = _mm256_loadu_pd(by), yh = _mm256_loadu_pd(by + 4), zl = _mm256_loadu_pd(bz), zh = _mm256_loadu_pd(bz + 4);
            for (size_t n = 0; n < n_coeff; ++n)
            {
                const __m256d vx = _mm256_set1_pd(fx[n]), vy = _mm256_set1_pd(fy[n]), vz = _mm256_set1_pd(fz[n]);
                __m256d tl = _mm256_fmadd_pd(vx, xl, _mm256_fmadd_pd(vy, yl, _mm256_mul_pd(vz, zl)));
                __m256d th = _mm256_fmadd_pd(vx, xh, _mm256_fmadd_pd(vy, yh, _mm256_mul_pd(vz, zh)));
                tl = _mm256_sub_pd(tl, _mm256_round_pd(tl, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                th = _mm256_sub_pd(th, _mm256_round_pd(th, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                __m256 s8, c8;
                _fm256_sincos256_pd(_mm256_mul_pd(tl, two_pi), _mm256_mul_pd(th, two_pi), &s8, &c8);
                _mm256_store_ps(C + n * VEC, c8), _mm256_store_ps(S + n * VEC, s8);
            }
            size_t g = 0;
            __m256d al[8], ah[8];
            auto put = [&](size_t gg, __m256d l, __m256d h) {
                double *out = &val[gg * n_pos + i_start];
                if (n_valid == VEC) { _mm256_storeu_pd(out, l), _mm256_storeu_pd(out + 4, h); }
                else { double tmp[VEC]; _mm256_storeu_pd(tmp, l), _mm256_storeu_pd(tmp + 4, h); for (size_t k = 0; k < n_valid; ++k) out[k] = tmp[k]; } };
            for (; g + G <= n_gen; g += G)
            {
                acc_group<G>(C, S, &wcf[g * n_coeff], &wsf[g * n_coeff], n_coeff, al, ah);
                for (int j = 0; j < G; ++j)
                    put(g + j, al[j], ah[j]);
            }
            for (; g < n_gen; ++g)
            {
                acc_one(C, S, &wcf[g * n_coeff], &wsf[g * n_coeff], n_coeff, al, ah);
                put(g, al[0], ah[0]);
            }
        }
        _mm_free(C), _mm_free(S);
    }
}
#define DEF(name, G) void name(const double *fx, const double *fy, const double *fz, const double *wc, const double *ws, const size_t n_coeff, const size_t n_gen, const double *px, const double *py, const double *pz, const size_t n_pos, double *val) { kernel<G>(fx, fy, fz, wc, ws, n_coeff, n_gen, px, py, pz, n_pos, val); }
DEF(var_SOS_AVX2_H4, 4)
DEF(var_SOS_AVX2_H8, 8)
