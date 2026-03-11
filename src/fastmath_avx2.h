// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
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

// AVX2 accelerated arc-sine (single precision, full [-1, 1] domain)
// Uses Cephes-style rational polynomial with half-angle identity for |x| > 0.5
// Max error: ~2 ULP
static inline __m256 _fm256_asin256_ps(__m256 x)
{
    // Constants
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000u));
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 pio2 = _mm256_set1_ps(1.5707963267948966f); // pi/2

    // Cephes single-precision asin polynomial coefficients (degree-5 on x^2)
    // asin(x) ≈ x + x^3 * P(x^2)
    const __m256 P0 = _mm256_set1_ps(4.2163199048E-2f);
    const __m256 P1 = _mm256_set1_ps(2.4181311049E-2f);
    const __m256 P2 = _mm256_set1_ps(4.5470025998E-2f);
    const __m256 P3 = _mm256_set1_ps(7.4953002686E-2f);
    const __m256 P4 = _mm256_set1_ps(1.6666752422E-1f);

    // Extract sign and compute |x|
    __m256 sign = _mm256_and_ps(x, sign_mask);
    __m256 ax = _mm256_and_ps(x, abs_mask);

    // Region selection: mask is all-ones where |x| > 0.5
    __m256 big = _mm256_cmp_ps(ax, half, _CMP_GT_OQ);

    // Region A (|x| <= 0.5): u = x^2,    base = |x|
    // Region B (|x| >  0.5): u = (1-|x|)/2, base = sqrt(u)
    __m256 uA = _mm256_mul_ps(ax, ax);
    __m256 uB = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
    __m256 u = _mm256_blendv_ps(uA, uB, big);

    __m256 base_B = _mm256_sqrt_ps(u);
    __m256 base = _mm256_blendv_ps(ax, base_B, big);

    // Evaluate P(u) = P0*u^4 + P1*u^3 + P2*u^2 + P3*u + P4  (Cephes degree-4)
    // Estrin decomposition: (P0*u^2 + P1*u + P2)*u^2 + (P3*u + P4)
    //   level 0: t0 = P0*u + P1,  t1 = P3*u + P4  (parallel)
    //   level 1: t2 = t0*u + P2                     (serial on t0)
    //   level 2: poly = t2*u^2 + t1                  (serial on t2, uses t1)
    __m256 u2 = _mm256_mul_ps(u, u);
    __m256 t0 = _mm256_fmadd_ps(P0, u, P1);  // P0*u + P1
    __m256 t1 = _mm256_fmadd_ps(P3, u, P4);  // P3*u + P4
    __m256 t2 = _mm256_fmadd_ps(t0, u, P2);  // (P0*u + P1)*u + P2
    __m256 poly = _mm256_fmadd_ps(t2, u2, t1); // t2*u^2 + t1

    // r = base + base * u * poly  =  base * (1 + u * poly)
    __m256 r = _mm256_fmadd_ps(_mm256_mul_ps(base, u), poly, base);

    // Reconstruction
    // Region A: asin = r           (with original sign)
    // Region B: asin = pi/2 - 2*r  (with original sign)
    __m256 rB = _mm256_fnmadd_ps(two, r, pio2); // pi/2 - 2*r
    __m256 result = _mm256_blendv_ps(r, rB, big);

    // Restore sign
    return _mm256_or_ps(result, sign);
}

// AVX2 accelerated arc-cosine (single precision)
// Thin wrapper: acos(x) = pi/2 - asin(x)
static inline __m256 _fm256_acos256_ps(__m256 x)
{
    // Constants
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000u));
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 pio2 = _mm256_set1_ps(1.5707963267948966f); // pi/2
    const __m256 pi = _mm256_set1_ps(3.14159265358979323846f);

    // Same Cephes polynomial coefficients as asin
    const __m256 P0 = _mm256_set1_ps(4.2163199048E-2f);
    const __m256 P1 = _mm256_set1_ps(2.4181311049E-2f);
    const __m256 P2 = _mm256_set1_ps(4.5470025998E-2f);
    const __m256 P3 = _mm256_set1_ps(7.4953002686E-2f);
    const __m256 P4 = _mm256_set1_ps(1.6666752422E-1f);

    // Sign and |x|
    __m256 sign = _mm256_and_ps(x, sign_mask);
    __m256 ax = _mm256_and_ps(x, abs_mask);

    // Region selection mask: true where |x| > 0.5
    __m256 big = _mm256_cmp_ps(ax, half, _CMP_GT_OQ);

    // Region A: u = x^2,         base = |x|
    // Region B: u = (1-|x|)/2,   base = sqrt(u)
    __m256 uA = _mm256_mul_ps(ax, ax);
    __m256 uB = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
    __m256 u = _mm256_blendv_ps(uA, uB, big);

    __m256 base_B = _mm256_sqrt_ps(u);
    __m256 base = _mm256_blendv_ps(ax, base_B, big);

    // Same polynomial: P(u) = P0*u^4 + P1*u^3 + P2*u^2 + P3*u + P4
    __m256 u2 = _mm256_mul_ps(u, u);
    __m256 t0 = _mm256_fmadd_ps(P0, u, P1);
    __m256 t1 = _mm256_fmadd_ps(P3, u, P4);
    __m256 t2 = _mm256_fmadd_ps(t0, u, P2);
    __m256 poly = _mm256_fmadd_ps(t2, u2, t1);

    // r = base + base * u * poly  (core asin approximation on reduced argument)
    __m256 r = _mm256_fmadd_ps(_mm256_mul_ps(base, u), poly, base);

    // Reconstruction (avoids catastrophic cancellation near |x| → 1):
    //   Region A:          acos(x) = pi/2 - r  (with sign of x applied to r)
    //   Region B, x > 0:   acos(x) = 2*r
    //   Region B, x < 0:   acos(x) = pi - 2*r
    __m256 neg = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);

    __m256 rA = _mm256_sub_ps(pio2, _mm256_or_ps(r, sign)); // pi/2 - (±r)
    __m256 rB_pos = _mm256_mul_ps(two, r);                   // 2*r
    __m256 rB_neg = _mm256_fnmadd_ps(two, r, pi);            // pi - 2*r
    __m256 rB = _mm256_blendv_ps(rB_pos, rB_neg, neg);

    return _mm256_blendv_ps(rA, rB, big);
}

#endif