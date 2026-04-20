// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <immintrin.h>
#include "quadriga_lib_avx2_functions.hpp"

// Perform a simple test calculation
void qd_TEST_AVX2(const float *X, // Aligned memory, 16 floats
                  float *Z)       // 8 floats
{
    __m256 tx = _mm256_load_ps(X);
    __m256 ty = _mm256_load_ps(&X[8]);
    const __m256 r2 = _mm256_set1_ps(2.0f);
    __m256 z = _mm256_fmadd_ps(tx, ty, r2);
    _mm256_storeu_ps(Z, z); // Unaligned store
}
