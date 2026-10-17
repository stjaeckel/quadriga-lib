// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <immintrin.h>
#include <iostream>
#include <vector>

#include "quadriga_lib_avx2_functions.hpp"

// Vector size for AVX2
#define VEC_SIZE 8

// sin and cos of 2 * pi * r for r in [-0.5, 0.5] (one period, already reduced in double precision by the caller).
// Quadrant reduction q = nearest integer of 4 * r (exact in float), t = (4 * r - q) * pi / 2 in [-pi/4, pi/4], then
// the Cephes single-precision polynomials of _fm256_sincos256_ps and the quadrant swap / signs.
static inline void qd_sincos_2pi_reduced(__m256 r, __m256 *s, __m256 *c)
{
    const __m256 four = _mm256_set1_ps(4.0f);
    const __m256 pi_2 = _mm256_set1_ps(1.57079632679489662f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 n_half = _mm256_set1_ps(-0.5f);
    const __m256i i32_1 = _mm256_set1_epi32(1);
    const __m256i i32_2 = _mm256_set1_epi32(2);

    const __m256 Sp0 = _mm256_set1_ps(-1.9515295891E-4f);
    const __m256 Sp1 = _mm256_set1_ps(8.3321608736E-3f);
    const __m256 Sp2 = _mm256_set1_ps(-1.6666654611E-1f);
    const __m256 Cp0 = _mm256_set1_ps(2.443315711809948E-005f);
    const __m256 Cp1 = _mm256_set1_ps(-1.388731625493765E-003f);
    const __m256 Cp2 = _mm256_set1_ps(4.166664568298827E-002f);

    __m256 u = _mm256_mul_ps(r, four);                                        // In [-2, 2], exact
    __m256 n = _mm256_round_ps(u, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); // Quadrant -2 ... 2
    __m256i q = _mm256_cvtps_epi32(n);
    __m256 t = _mm256_mul_ps(_mm256_sub_ps(u, n), pi_2); // In [-pi/4, pi/4]

    __m256 z = _mm256_mul_ps(t, t);
    __m256 z2 = _mm256_mul_ps(z, z);

    // sin(t) ~ t * (1 + Sp2*z + Sp1*z^2 + Sp0*z^3)
    __m256 t0s = _mm256_fmadd_ps(Sp2, z, one);
    __m256 t1s = _mm256_fmadd_ps(Sp0, z, Sp1);
    __m256 ys = _mm256_mul_ps(_mm256_fmadd_ps(z2, t1s, t0s), t);

    // cos(t) ~ 1 - 0.5*z + z^2*(Cp2 + Cp1*z + Cp0*z^2)
    __m256 te = _mm256_fmadd_ps(Cp0, z2, Cp2);
    __m256 ev = _mm256_fmadd_ps(z2, te, one);
    __m256 to = _mm256_fmadd_ps(Cp1, z2, n_half);
    __m256 yc = _mm256_fmadd_ps(z, to, ev);

    // sin(t + q*pi/2), cos(t + q*pi/2): swap for odd q; sin negated for q & 2, cos negated for (q + 1) & 2
    __m256 swap = _mm256_castsi256_ps(_mm256_slli_epi32(q, 31));
    __m256 sign_sin = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(q, i32_2), 30));
    __m256 sign_cos = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(_mm256_add_epi32(q, i32_1), i32_2), 30));
    *s = _mm256_xor_ps(_mm256_blendv_ps(ys, yc, swap), sign_sin);
    *c = _mm256_xor_ps(_mm256_blendv_ps(yc, ys, swap), sign_cos);
}

// AVX2 accelerated implementation of DFT
// The phase x = freq * delay (in cycles) is computed in double precision and reduced to x - round(x), which is in
// [-0.5, 0.5], before sin / cos are evaluated in single precision; coefficients and sums are single precision
template <typename dtype>
void qd_DFT_AVX2(const dtype *__restrict CFr,    // Channel coefficients, real part, Size [n_ant, n_path]
                 const dtype *__restrict CFi,    // Channel coefficients, imaginary part, Size [n_ant, n_path]
                 const dtype *__restrict DL,     // Path delays in seconds, Size [n_ant, n_path] or [1, n_path]
                 const size_t n_ant,             // Number of MIMO sub-links
                 const size_t n_path,            // Number multipath components
                 const bool planar_wave,         // Indicator that same delays are used for all antennas
                 const double *__restrict freq,  // Carrier frequencies -bandwidth * pilot_grid in [Hz], aligned to 32 byte, Size [ n_carrier ]
                 const size_t n_carrier,         // Number of carriers, mutiple of 8
                 float *__restrict Hr,           // Channel matrix, real part, aligned to 32 byte, Size [ n_carrier, n_ant ]
                 float *__restrict Hi)           // Channel matrix, imaginary part, aligned to 32 byte, Size [ n_carrier, n_ant ]
{
    // Preallocate per-path, per-antenna scalars to avoid per-block casts
    std::vector<float> crf(n_path), cif(n_path);
    std::vector<double> dld(n_path);

    for (size_t i_ant = 0; i_ant < n_ant; ++i_ant) // Antenna loop
    {
        // Fill scalars once per (antenna, path)
        for (size_t i_path = 0; i_path < n_path; ++i_path)
        {
            const size_t idx = i_path * n_ant + i_ant;
            crf[i_path] = (float)CFr[idx];
            cif[i_path] = (float)CFi[idx];
            dld[i_path] = planar_wave ? (double)DL[i_path] : (double)DL[idx];
        }

        const size_t base = i_ant * n_carrier;
        for (size_t i_carrier = 0; i_carrier < n_carrier; i_carrier += VEC_SIZE)
        {
            const size_t o = base + i_carrier;

            __m256 hr = _mm256_setzero_ps();
            __m256 hi = _mm256_setzero_ps();

            const __m256d f_lo = _mm256_load_pd(&freq[i_carrier]);
            const __m256d f_hi = _mm256_load_pd(&freq[i_carrier + 4]);

            for (size_t i_path = 0; i_path < n_path; ++i_path)
            {
                const __m256 cr = _mm256_set1_ps(crf[i_path]);
                const __m256 ci = _mm256_set1_ps(cif[i_path]);
                const __m256d dl = _mm256_set1_pd(dld[i_path]);

                // Phase in cycles, reduced to [-0.5, 0.5] in double precision
                __m256d x_lo = _mm256_mul_pd(f_lo, dl);
                __m256d x_hi = _mm256_mul_pd(f_hi, dl);
                x_lo = _mm256_sub_pd(x_lo, _mm256_round_pd(x_lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                x_hi = _mm256_sub_pd(x_hi, _mm256_round_pd(x_hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                const __m256 r = _mm256_set_m128(_mm256_cvtpd_ps(x_hi), _mm256_cvtpd_ps(x_lo));

                __m256 vsin, vcos;
                qd_sincos_2pi_reduced(r, &vsin, &vcos);

                hr = _mm256_fnmadd_ps(vsin, ci, hr);
                hr = _mm256_fmadd_ps(vcos, cr, hr);

                hi = _mm256_fmadd_ps(vsin, cr, hi);
                hi = _mm256_fmadd_ps(vcos, ci, hi);
            }

            _mm256_store_ps(&Hr[o], hr);
            _mm256_store_ps(&Hi[o], hi);
        }
    }
}

template void qd_DFT_AVX2(const float *CFr,
                          const float *CFi,
                          const float *DL,
                          const size_t n_ant,
                          const size_t n_path,
                          const bool planar_wave,
                          const double *freq,
                          const size_t n_carrier,
                          float *Hr,
                          float *Hi);

template void qd_DFT_AVX2(const double *CFr,
                          const double *CFi,
                          const double *DL,
                          const size_t n_ant,
                          const size_t n_path,
                          const bool planar_wave,
                          const double *freq,
                          const size_t n_carrier,
                          float *Hr,
                          float *Hi);