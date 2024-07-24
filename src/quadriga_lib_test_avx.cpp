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
#include "quadriga_lib_test_avx.hpp"

// Perform a simple test calculation
void quadriga_lib::avx2_test(const float *X, // Aligned memory, 16 floats
                             float *Z)       // Aligned memory, 8 floats
{
    __m256 tx = _mm256_load_ps(X);
    __m256 ty = _mm256_load_ps(&X[8]);
    const __m256 r2 = _mm256_set1_ps(2.0f);
    __m256 z = _mm256_fmadd_ps(tx, ty, r2);
    _mm256_store_ps(Z, z);
}
