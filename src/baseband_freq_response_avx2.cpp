// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <immintrin.h>
#include <iostream>
#include <vector>

#include "quadriga_lib_avx2_functions.hpp"
#include "fastmath_avx2.h"

// Vector size for AVX2
#define VEC_SIZE 8

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
                 const double *__restrict freq,  // Carrier frequencies -bandwidth * pilot_grid in [Hz], Size [ n_carrier ]
                 const size_t n_carrier,         // Number of carriers, mutiple of 8
                 float *__restrict Hr,           // Channel matrix, real part, Size [ n_carrier, n_ant ]
                 float *__restrict Hi)           // Channel matrix, imaginary part, Size [ n_carrier, n_ant ]
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

            const __m256d f_lo = _mm256_loadu_pd(&freq[i_carrier]);
            const __m256d f_hi = _mm256_loadu_pd(&freq[i_carrier + 4]);

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
                _fm256_sincos_2pi_reduced_ps(r, &vsin, &vcos);

                hr = _mm256_fnmadd_ps(vsin, ci, hr);
                hr = _mm256_fmadd_ps(vcos, cr, hr);

                hi = _mm256_fmadd_ps(vsin, cr, hi);
                hi = _mm256_fmadd_ps(vcos, ci, hi);
            }

            _mm256_storeu_ps(&Hr[o], hr);
            _mm256_storeu_ps(&Hi[o], hi);
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