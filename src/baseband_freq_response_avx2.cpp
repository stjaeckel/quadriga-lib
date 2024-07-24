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
#include "avx_mathfun.h"
#include "baseband_freq_response_avx2.hpp"

// Vector size for AVX2
#define VEC_SIZE 8

// AVX2 accelerated implementation of DFT
template <typename dtype>
void qd_DFT_AVX2(const dtype *CFr,       // Channel coefficients, real part, Size [n_ant, n_path]
                 const dtype *CFi,       // Channel coefficients, imaginary part, Size [n_ant, n_path]
                 const dtype *DL,        // Path delays in seconds, Size [n_ant, n_path] or [1, n_path]
                 const size_t n_ant,     // Number of MIMO sub-links
                 const size_t n_path,    // Number multipath components
                 const bool planar_wave, // Indicator that same delays are used for all antennas
                 const float *phasor,    // Phasor, -pi/2 to pi/2, aligned to 32 byte, Size [ n_carrier ]
                 const size_t n_carrier, // Number of carriers, mutiple of 8
                 float *Hr,              // Channel matrix, real part, Size [ n_carrier, n_ant ]
                 float *Hi)              // Channel matrix, imaginary part, Size [ n_carrier, n_ant ]
{
    for (size_t i_path = 0; i_path < n_path; ++i_path) // Path loop
        for (size_t i_ant = 0; i_ant < n_ant; ++i_ant) // Antenna loop
        {
            size_t i = i_path * n_ant + i_ant;
            __m256 cr = _mm256_set1_ps((float)CFr[i]);
            __m256 ci = _mm256_set1_ps((float)CFi[i]);
            __m256 dl = planar_wave ? _mm256_set1_ps((float)DL[i_path]) : _mm256_set1_ps((float)DL[i]);

            for (size_t i_carrier = 0; i_carrier < n_carrier; i_carrier += VEC_SIZE)
            {
                __m256 xmm0 = _mm256_load_ps(&phasor[i_carrier]); // Load phasor
                xmm0 = _mm256_mul_ps(xmm0, dl);                   // Multiply with delay
                __m256 xmm1, xmm2;                                // Registers to store the sine and cosine
                sincos256_ps(xmm0, &xmm1, &xmm2);                 // Calculate sine and cosine

                size_t o = i_ant * n_carrier + i_carrier;
                xmm0 = _mm256_load_ps(&Hr[o]);           // Load previous Hr
                xmm0 = _mm256_fnmadd_ps(xmm1, ci, xmm0); // Hr - si * ci
                xmm0 = _mm256_fmadd_ps(xmm2, cr, xmm0);  // Hr + co * cr - si * ci
                _mm256_store_ps(&Hr[o], xmm0);           // Update Hr in memory

                xmm0 = _mm256_load_ps(&Hi[o]);          // Load previous Hi
                xmm0 = _mm256_fmadd_ps(xmm1, cr, xmm0); // Hi + si * cr
                xmm0 = _mm256_fmadd_ps(xmm2, ci, xmm0); // Hi + co * ci + si * cr
                _mm256_store_ps(&Hi[o], xmm0);          // Update Hi in memory
            }
        }
}

template void qd_DFT_AVX2(const float *CFr,       // Channel coefficients, real part, Size [n_ant, n_path]
                          const float *CFi,       // Channel coefficients, imaginary part, Size [n_ant, n_path]
                          const float *DL,        // Path delays in seconds, Size [n_ant, n_path] or [1, n_path]
                          const size_t n_ant,     // Number of MIMO sub-links
                          const size_t n_path,    // Number multipath components
                          const bool planar_wave, // Indicator that same delays are used for all antennas
                          const float *phasor,    // Phasor, -pi/2 to pi/2, aligned to 32 byte, Size [ n_carrier ]
                          const size_t n_carrier, // Number of carriers, mutiple of 8
                          float *Hr,              // Channel matrix, real part, Size [ n_carrier, n_ant ]
                          float *Hi);             // Channel matrix, imaginary part, Size [ n_carrier, n_ant ]

template void qd_DFT_AVX2(const double *CFr,      // Channel coefficients, real part, Size [n_ant, n_path]
                          const double *CFi,      // Channel coefficients, imaginary part, Size [n_ant, n_path]
                          const double *DL,       // Path delays in seconds, Size [n_ant, n_path] or [1, n_path]
                          const size_t n_ant,     // Number of MIMO sub-links
                          const size_t n_path,    // Number multipath components
                          const bool planar_wave, // Indicator that same delays are used for all antennas
                          const float *phasor,    // Phasor, -pi/2 to pi/2, aligned to 32 byte, Size [ n_carrier ]
                          const size_t n_carrier, // Number of carriers, mutiple of 8
                          float *Hr,              // Channel matrix, real part, Size [ n_carrier, n_ant ]
                          float *Hi);