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

#ifndef baseband_freq_response_avx2_H
#define baseband_freq_response_avx2_H

#include <cstddef>

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
                 float *Hi);             // Channel matrix, imaginary part, Size [ n_carrier, n_ant ]

#endif