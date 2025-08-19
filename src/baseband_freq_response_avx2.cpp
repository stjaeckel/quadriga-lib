// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

#include <immintrin.h>
#include <iostream>

#include "fastmath_avx2.h"
#include "quadriga_lib_avx2_functions.hpp"

// Vector size for AVX2
#define VEC_SIZE 8

// AVX2 accelerated implementation of DFT
template <typename dtype>
void qd_DFT_AVX2(const dtype *__restrict CFr,    // Channel coefficients, real part, Size [n_ant, n_path]
                 const dtype *__restrict CFi,    // Channel coefficients, imaginary part, Size [n_ant, n_path]
                 const dtype *__restrict DL,     // Path delays in seconds, Size [n_ant, n_path] or [1, n_path]
                 const size_t n_ant,             // Number of MIMO sub-links
                 const size_t n_path,            // Number multipath components
                 const bool planar_wave,         // Indicator that same delays are used for all antennas
                 const float *__restrict phasor, // Phasor, -pi/2 to pi/2, aligned to 32 byte, Size [ n_carrier ]
                 const size_t n_carrier,         // Number of carriers, mutiple of 8
                 float *__restrict Hr,           // Channel matrix, real part, aligned to 32 byte, Size [ n_carrier, n_ant ]
                 float *__restrict Hi)           // Channel matrix, imaginary part, aligned to 32 byte, Size [ n_carrier, n_ant ]
{
    // Preallocate per-path, per-antenna float scalars to avoid per-block casts
    std::vector<float> crf(n_path), cif(n_path), dlf(n_path);

    for (size_t i_ant = 0; i_ant < n_ant; ++i_ant) // Antenna loop
    {
        // Fill float scalars once per (antenna, path)
        for (size_t i_path = 0; i_path < n_path; ++i_path)
        {
            const size_t idx = i_path * n_ant + i_ant;
            crf[i_path] = (float)CFr[idx];
            cif[i_path] = (float)CFi[idx];
            dlf[i_path] = planar_wave ? (float)DL[i_path] : (float)DL[idx];
        }

        const size_t base = i_ant * n_carrier;
        for (size_t i_carrier = 0; i_carrier < n_carrier; i_carrier += VEC_SIZE)
        {
            const size_t o = base + i_carrier;

            __m256 hr = _mm256_setzero_ps();
            __m256 hi = _mm256_setzero_ps();

            const __m256 vphasor = _mm256_load_ps(&phasor[i_carrier]);

            for (size_t i_path = 0; i_path < n_path; ++i_path)
            {
                const __m256 cr = _mm256_set1_ps(crf[i_path]);
                const __m256 ci = _mm256_set1_ps(cif[i_path]);
                const __m256 dl = _mm256_set1_ps(dlf[i_path]);

                __m256 theta = _mm256_mul_ps(vphasor, dl);
                __m256 vsin, vcos;
                _fm256_sincos256_ps(theta, &vsin, &vcos);

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
                          const float *phasor,
                          const size_t n_carrier,
                          float *Hr,
                          float *Hi);

template void qd_DFT_AVX2(const double *CFr,
                          const double *CFi,
                          const double *DL,
                          const size_t n_ant,
                          const size_t n_path,
                          const bool planar_wave,
                          const float *phasor,
                          const size_t n_carrier,
                          float *Hr,
                          float *Hi);