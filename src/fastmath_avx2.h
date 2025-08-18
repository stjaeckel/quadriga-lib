// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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

// AVX2 accelerated math functions

#ifndef quadriga_lib_fastmath_avx2_H
#define quadriga_lib_fastmath_avx2_H

#include <immintrin.h>

// Note: A SINE / COSINE-only version provides no significant performance advantage
// Just compute both and drop the unwanted one
static inline void _fm256_sincos256_ps(__m256 x, __m256 *__restrict s, __m256 *__restrict c)
{
    // Bit masks
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000u));
    const __m256 inv_sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i pi32_0 = _mm256_set1_epi32(0);
    const __m256i pi32_1 = _mm256_set1_epi32(1);
    const __m256i pi32_inv1 = _mm256_set1_epi32(~1);
    const __m256i pi32_2 = _mm256_set1_epi32(2);
    const __m256i pi32_4 = _mm256_set1_epi32(4);

    // Constants
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 n_half = _mm256_set1_ps(-0.5f);
    const __m256 FOPI = _mm256_set1_ps(1.27323954473516f); // 4/pi

    // Cody–Waite split constants
    const __m256 DP1 = _mm256_set1_ps(-0.78515625f);
    const __m256 DP2 = _mm256_set1_ps(-2.4187564849853515625e-4f);
    const __m256 DP3 = _mm256_set1_ps(-3.77489497744594108e-8f);

    // Polynomial coeffs (Cephes), single-precision
    const __m256 Sp0 = _mm256_set1_ps(-1.9515295891E-4f);
    const __m256 Sp1 = _mm256_set1_ps(8.3321608736E-3f);
    const __m256 Sp2 = _mm256_set1_ps(-1.6666654611E-1f);

    const __m256 Cp0 = _mm256_set1_ps(2.443315711809948E-005f);
    const __m256 Cp1 = _mm256_set1_ps(-1.388731625493765E-003f);
    const __m256 Cp2 = _mm256_set1_ps(4.166664568298827E-002f);

    // sign for sine; |x|
    __m256 sign_bit_sin = _mm256_and_ps(x, sign_mask);
    x = _mm256_and_ps(x, inv_sign_mask);

    // Scale by 4/pi and get j = (int)y;  j = (j+1) & (~1)
    __m256 y = _mm256_mul_ps(x, FOPI);
    __m256i j = _mm256_cvttps_epi32(y);
    j = _mm256_add_epi32(j, pi32_1);
    j = _mm256_and_si256(j, pi32_inv1);
    y = _mm256_cvtepi32_ps(j);

    // swap sign flag for sine and cosine sign bit
    __m256 swap_sin = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(j, pi32_4), 29));
    __m256 sign_bit_cos = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_andnot_si256(_mm256_sub_epi32(j, pi32_2), pi32_4), 29));

    // polynomial selection mask for sine: (j & 2) == 0
    __m256 poly = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(j, pi32_2), pi32_0));

    // Range reduction: x = ((x - y*DP1) - y*DP2) - y*DP3 (as fused adds)
    x = _mm256_fmadd_ps(y, DP1, x);
    x = _mm256_fmadd_ps(y, DP2, x);
    x = _mm256_fmadd_ps(y, DP3, x);

    // Common powers
    __m256 z = _mm256_mul_ps(x, x);
    __m256 z2 = _mm256_mul_ps(z, z);

    // -------- Estrin: SIN --------
    // sin(x) ≈ x * (1 + Sp2*z + Sp1*z^2 + Sp0*z^3)
    __m256 t0s = _mm256_fmadd_ps(Sp2, z, one); // 1 + Sp2*z
    __m256 t1s = _mm256_fmadd_ps(Sp0, z, Sp1); // Sp1 + Sp0*z
    __m256 Qs = _mm256_fmadd_ps(z2, t1s, t0s); // t0s + z2*t1s
    __m256 ys = _mm256_mul_ps(Qs, x);          // x * Qs

    // -------- Estrin: COS --------
    // cos(x) ≈ 1 - 0.5*z + z^2*(Cp2 + Cp1*z + Cp0*z^2)
    // Even: 1 + z2*(Cp2 + Cp0*z2)
    __m256 te = _mm256_fmadd_ps(Cp0, z2, Cp2); // Cp2 + Cp0*z2
    __m256 ev = _mm256_fmadd_ps(z2, te, one);  // 1 + z2*te
    // Odd: z*(-0.5 + Cp1*z2)
    __m256 to = _mm256_fmadd_ps(Cp1, z2, n_half); // -0.5 + Cp1*z2
    __m256 yc = _mm256_fmadd_ps(z, to, ev);       // even + odd

    // Select between sin/cos polys for sine/cosine results
    __m256 sin_approx = _mm256_or_ps(_mm256_and_ps(poly, ys), _mm256_andnot_ps(poly, yc));
    __m256 cos_approx = _mm256_or_ps(_mm256_and_ps(poly, yc), _mm256_andnot_ps(poly, ys));

    // Apply signs
    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sin);
    *s = _mm256_xor_ps(sin_approx, sign_bit_sin);
    *c = _mm256_xor_ps(cos_approx, sign_bit_cos);
}

#endif