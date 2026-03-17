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

// AVX2 signum: returns -1.0f, 0.0f, or +1.0f per lane
static inline __m256 _fm256_signum_ps(__m256 x)
{
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    __m256 pos = _mm256_and_ps(_mm256_cmp_ps(x, zero, _CMP_GT_OQ), one);     // x > 0 → 1.0
    __m256 neg = _mm256_and_ps(_mm256_cmp_ps(x, zero, _CMP_LT_OQ), neg_one); // x < 0 → -1.0
    return _mm256_or_ps(pos, neg);
}

// Clamp to [-1, +1]
static inline __m256 _fm256_clamp_pm1_ps(__m256 x)
{
    return _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(1.0f)), _mm256_set1_ps(-1.0f));
}

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

// Double-input variant: takes two __m256d (8 doubles as low/high halves),
// does Cody–Waite range reduction to [-pi/4, pi/4] in double precision,
// converts to float, then evaluates the same Cephes polynomial + reconstruction.
// This avoids the cascaded reduction problem where the float kernel would
// re-reduce an already-reduced input, losing precision near octant boundaries.
static inline void _fm256_sincos256_pd(__m256d xl, __m256d xh, __m256 *__restrict s, __m256 *__restrict c)
{
    // --- Double-precision constants for range reduction ---
    const __m256d abs_mask_d = _mm256_castsi256_pd(
        _mm256_set_epi64x(0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFFLL,
                          0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFFLL));
    const __m256d FOPI_D = _mm256_set1_pd(1.2732395447351626862); // 4/pi

    // Two-constant Cody–Waite split of pi/4:
    //   DP1_D has few mantissa bits → j*DP1_D is exact for reasonable j
    //   DP1_D + DP2_D = pi/4 to full double precision
    const __m256d DP1_D = _mm256_set1_pd(0.78515625);
    const __m256d DP2_D = _mm256_set1_pd(2.4191339744830961566e-4);

    const __m128i one_128 = _mm_set1_epi32(1);
    const __m128i inv1_128 = _mm_set1_epi32(~1);

    // --- Process low 4 doubles ---
    __m256d sign_dl = _mm256_andnot_pd(abs_mask_d, xl);       // sign bits
    __m256d axl = _mm256_and_pd(xl, abs_mask_d);              // |x|
    __m256d yl = _mm256_mul_pd(axl, FOPI_D);                  // |x| * 4/pi
    __m128i jl = _mm256_cvttpd_epi32(yl);                     // truncate → 4 int32
    jl = _mm_and_si128(_mm_add_epi32(jl, one_128), inv1_128); // (j+1) & ~1
    __m256d jdl = _mm256_cvtepi32_pd(jl);                     // j → double
    axl = _mm256_fnmadd_pd(jdl, DP1_D, axl);                  // ax -= j*DP1
    axl = _mm256_fnmadd_pd(jdl, DP2_D, axl);                  // ax -= j*DP2

    // --- Process high 4 doubles ---
    __m256d sign_dh = _mm256_andnot_pd(abs_mask_d, xh);
    __m256d axh = _mm256_and_pd(xh, abs_mask_d);
    __m256d yh = _mm256_mul_pd(axh, FOPI_D);
    __m128i jh = _mm256_cvttpd_epi32(yh);
    jh = _mm_and_si128(_mm_add_epi32(jh, one_128), inv1_128);
    __m256d jdh = _mm256_cvtepi32_pd(jh);
    axh = _mm256_fnmadd_pd(jdh, DP1_D, axh);
    axh = _mm256_fnmadd_pd(jdh, DP2_D, axh);

    // --- Pack into 8-wide float/int vectors ---
    __m256 x8 = _mm256_set_m128(_mm256_cvtpd_ps(axh), _mm256_cvtpd_ps(axl));
    __m256i j = _mm256_set_m128i(jh, jl);
    __m256 sign_bit_sin = _mm256_set_m128(_mm256_cvtpd_ps(sign_dh),
                                          _mm256_cvtpd_ps(sign_dl));

    // --- From here: identical to _fm256_sincos256_ps after its range reduction ---
    const __m256i pi32_0 = _mm256_set1_epi32(0);
    const __m256i pi32_2 = _mm256_set1_epi32(2);
    const __m256i pi32_4 = _mm256_set1_epi32(4);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 n_half = _mm256_set1_ps(-0.5f);

    const __m256 Sp0 = _mm256_set1_ps(-1.9515295891E-4f);
    const __m256 Sp1 = _mm256_set1_ps(8.3321608736E-3f);
    const __m256 Sp2 = _mm256_set1_ps(-1.6666654611E-1f);
    const __m256 Cp0 = _mm256_set1_ps(2.443315711809948E-005f);
    const __m256 Cp1 = _mm256_set1_ps(-1.388731625493765E-003f);
    const __m256 Cp2 = _mm256_set1_ps(4.166664568298827E-002f);

    // Swap/sign/poly flags from j
    __m256 swap_sin = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(j, pi32_4), 29));
    __m256 sign_bit_cos = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_andnot_si256(_mm256_sub_epi32(j, pi32_2), pi32_4), 29));
    __m256 poly = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(j, pi32_2), pi32_0));

    // Common powers
    __m256 z = _mm256_mul_ps(x8, x8);
    __m256 z2 = _mm256_mul_ps(z, z);

    // Estrin: SIN   sin(x) ~ x * (1 + Sp2*z + Sp1*z^2 + Sp0*z^3)
    __m256 t0s = _mm256_fmadd_ps(Sp2, z, one);
    __m256 t1s = _mm256_fmadd_ps(Sp0, z, Sp1);
    __m256 Qs = _mm256_fmadd_ps(z2, t1s, t0s);
    __m256 ys = _mm256_mul_ps(Qs, x8);

    // Estrin: COS   cos(x) ~ 1 - 0.5*z + z^2*(Cp2 + Cp1*z + Cp0*z^2)
    __m256 te = _mm256_fmadd_ps(Cp0, z2, Cp2);
    __m256 ev = _mm256_fmadd_ps(z2, te, one);
    __m256 to = _mm256_fmadd_ps(Cp1, z2, n_half);
    __m256 yc = _mm256_fmadd_ps(z, to, ev);

    // Select between sin/cos polys
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
    __m256 t0 = _mm256_fmadd_ps(P0, u, P1);    // P0*u + P1
    __m256 t1 = _mm256_fmadd_ps(P3, u, P4);    // P3*u + P4
    __m256 t2 = _mm256_fmadd_ps(t0, u, P2);    // (P0*u + P1)*u + P2
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
    __m256 rB_pos = _mm256_mul_ps(two, r);                  // 2*r
    __m256 rB_neg = _mm256_fnmadd_ps(two, r, pi);           // pi - 2*r
    __m256 rB = _mm256_blendv_ps(rB_pos, rB_neg, neg);

    return _mm256_blendv_ps(rA, rB, big);
}

// AVX2 accelerated atan2(y, x) (single precision, full domain)
//
// Algorithm:
//   1. Octant reduction via min/max swap  →  ratio in [0, 1]
//   2. Cephes two-region reduction (threshold at tan(pi/8) ≈ 0.4142):
//        Region A (ratio <= 0.4142): evaluate polynomial directly
//        Region B (ratio >  0.4142): reduce via (ratio-1)/(ratio+1), offset pi/4
//      Both regions share a single division by blending numerator/denominator
//      before dividing (the reduced argument is always in [-0.4142, 0.4142]).
//   3. Cephes degree-7 minimax polynomial on the reduced argument
//   4. Quadrant reconstruction with sign copy
//
// Max error: < 2 ULP  (polynomial kernel < 0.3 ULP; reconstruction adds ~1 ULP)
//
// Special cases (IEEE-compliant except as noted):
//   atan2(±y, 0)   → ±pi/2        atan2(0, +x)    → 0
//   atan2(±0, -x)  → ±pi          atan2(±0, +0)   → ±0
// Note: atan2(±0, -0) returns ±0 (not ±pi); irrelevant for well-conditioned inputs.
static inline __m256 _fm256_atan2256_ps(__m256 y, __m256 x)
{
    // Bit masks
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000u));
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    // Constants
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 pio4 = _mm256_set1_ps(0.78539816339744830962f);   // pi/4
    const __m256 pio2 = _mm256_set1_ps(1.5707963267948966f);       // pi/2
    const __m256 pi = _mm256_set1_ps(3.14159265358979324f);        // pi
    const __m256 tanpio8 = _mm256_set1_ps(0.4142135623730950488f); // tan(pi/8)

    // Cephes single-precision atan polynomial coefficients (degree 7 odd, on [-0.4142, 0.4142])
    // atan(a) ≈ a + a * z * P(z),  z = a^2
    // P(z) = Ap3 + Ap2*z + Ap1*z^2 + Ap0*z^3
    const __m256 Ap0 = _mm256_set1_ps(8.05374449538e-2f);
    const __m256 Ap1 = _mm256_set1_ps(-1.38776856032e-1f);
    const __m256 Ap2 = _mm256_set1_ps(1.99777106478e-1f);
    const __m256 Ap3 = _mm256_set1_ps(-3.33329491539e-1f);

    // |x|, |y|
    __m256 ax = _mm256_and_ps(x, abs_mask);
    __m256 ay = _mm256_and_ps(y, abs_mask);

    // Octant reduction: work with min/max of |x|, |y|
    __m256 mn = _mm256_min_ps(ax, ay);
    __m256 mx = _mm256_max_ps(ax, ay);

    // Cephes two-region reduction (decided BEFORE dividing):
    //   big = (mn > tan(pi/8) * mx)   i.e. ratio = mn/mx > 0.4142
    //   Region A: num = mn,      den = mx       → ratio = mn/mx ∈ [0, 0.4142]
    //   Region B: num = mn - mx, den = mn + mx  → ratio = (mn/mx-1)/(mn/mx+1) ∈ (-0.4142, 0]
    __m256 big = _mm256_cmp_ps(mn, _mm256_mul_ps(tanpio8, mx), _CMP_GT_OQ);

    __m256 num = _mm256_blendv_ps(mn, _mm256_sub_ps(mn, mx), big);
    __m256 den = _mm256_blendv_ps(mx, _mm256_add_ps(mn, mx), big);

    // Guard: if den == 0 (both x and y are zero), set den = 1 to avoid 0/0.
    // num is also 0 in this case, so ratio = 0 → result = 0.
    __m256 den_zero = _mm256_cmp_ps(den, zero, _CMP_EQ_OQ);
    den = _mm256_blendv_ps(den, one, den_zero);

    __m256 a = _mm256_div_ps(num, den);

    // -------- Cephes polynomial: atan(a) ≈ a + a*z*P(z), z = a^2 --------
    // Estrin evaluation of P(z) = Ap3 + Ap2*z + Ap1*z^2 + Ap0*z^3
    //                           = (Ap3 + Ap2*z) + z^2*(Ap1 + Ap0*z)
    __m256 z = _mm256_mul_ps(a, a);
    __m256 z2 = _mm256_mul_ps(z, z);

    __m256 t0 = _mm256_fmadd_ps(Ap2, z, Ap3);  // Ap3 + Ap2*z
    __m256 t1 = _mm256_fmadd_ps(Ap0, z, Ap1);  // Ap1 + Ap0*z
    __m256 poly = _mm256_fmadd_ps(t1, z2, t0); // (Ap3+Ap2*z) + z^2*(Ap1+Ap0*z)

    // r = a + a*z*poly  =  fma(a*z, poly, a)
    __m256 r = _mm256_fmadd_ps(_mm256_mul_ps(a, z), poly, a);

    // Add pi/4 offset for the big-ratio region
    r = _mm256_blendv_ps(r, _mm256_add_ps(r, pio4), big);

    // Octant fixup: if |y| > |x|, we computed atan(|x|/|y|); apply identity:
    //   atan(a) + atan(1/a) = pi/2  →  r = pi/2 - r
    __m256 swap = _mm256_cmp_ps(ay, ax, _CMP_GT_OQ);
    r = _mm256_blendv_ps(r, _mm256_sub_ps(pio2, r), swap);

    // Quadrant fixup: if x < 0  →  r = pi - r
    __m256 x_neg = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
    r = _mm256_blendv_ps(r, _mm256_sub_ps(pi, r), x_neg);

    // Apply sign of y (r is non-negative here, so OR with sign bit suffices)
    r = _mm256_or_ps(r, _mm256_and_ps(y, sign_mask));

    return r;
}

// Convert geographic (azimuth, elevation) to Cartesian (x, y, z)
// Optionally returns the intermediate sincos values via restrict pointers.
// Pass NULL for any intermediate you don't need.
//   x = cos(el) * cos(az)
//   y = cos(el) * sin(az)
//   z = sin(el)
static inline void _fm256_geo2cart_ps(__m256 az, __m256 el, __m256 *__restrict x, __m256 *__restrict y, __m256 *__restrict z,
                                      __m256 *__restrict sAZ = nullptr, __m256 *__restrict cAZ = nullptr,
                                      __m256 *__restrict sEL = nullptr, __m256 *__restrict cEL = nullptr)
{
    __m256 sa, ca, se, ce;
    _fm256_sincos256_ps(az, &sa, &ca);
    _fm256_sincos256_ps(el, &se, &ce);

    *x = _mm256_mul_ps(ce, ca);
    *y = _mm256_mul_ps(ce, sa);
    *z = se;

    if (sAZ)
        *sAZ = sa;
    if (cAZ)
        *cAZ = ca;
    if (sEL)
        *sEL = se;
    if (cEL)
        *cEL = ce;
}

// Convert Cartesian (x, y, z) to geographic (azimuth, elevation)
// z is clamped to [-1, +1] before asin to guard against out-of-range
// values from upstream matrix multiplications or FMA rounding.
//   az = atan2(y, x)
//   el = asin(clamp(z, -1, 1))
static inline void _fm256_cart2geo_ps(__m256 x, __m256 y, __m256 z, __m256 *__restrict az, __m256 *__restrict el)
{
    *az = _fm256_atan2256_ps(y, x);
    *el = _fm256_asin256_ps(_fm256_clamp_pm1_ps(z));
}

// Convert complex RE/IM to polar form (magnitude and phase angle)
// abs = sqrt(re² + im²),  arg = atan2(im, re)
static inline void _fm256_absarg_ps(__m256 re, __m256 im, __m256 *__restrict abs, __m256 *__restrict arg)
{
    *abs = _mm256_sqrt_ps(_mm256_fmadd_ps(re, re, _mm256_mul_ps(im, im)));
    *arg = _fm256_atan2256_ps(im, re);
}

// Convert polar form (magnitude and phase angle) to complex RE/IM
// Equivalent to std::polar(abs, arg): re = abs * cos(arg),  im = abs * sin(arg)
static inline void _fm256_polar_ps(__m256 abs, __m256 arg, __m256 *__restrict re, __m256 *__restrict im)
{
    __m256 s, c;
    _fm256_sincos256_ps(arg, &s, &c);
    *re = _mm256_mul_ps(abs, c);
    *im = _mm256_mul_ps(abs, s);
}

// AVX2 accelerated spherical interpolation (SLERP) for complex value pairs
// Processes 8 complex pairs simultaneously with per-lane interpolation weight.
// Behaviour matches the scalar slerp_complex_mf<float> template except:
//   - sin(Phase) computed via sqrt(1 - cos²Phase) instead of sincos polynomial
//     (eliminates one full sincos call, improves accuracy by ~2 ULP)
//   - Weighted direction sum promoted to double precision to prevent catastrophic
//     cancellation when the interpolated direction crosses a component zero
//     (reduces worst-case ULP from millions to ~10–15)
//
// Otherwise identical:
//   - Spherical interpolation of normalised direction
//   - Linear interpolation of amplitude
//   - Smooth transition to linear fallback for near-antipodal directions
//   - Zero output when both input amplitudes are negligible
//
// Parameters (all __m256, i.e. 8 lanes of float):
//   Ar, Ai  – real/imag parts of source A
//   Br, Bi  – real/imag parts of source B
//   w       – interpolation weight per lane  (0 → A, 1 → B)
//   Xr, Xi  – output real/imag parts (written via restrict pointer)
static inline void _fm256_slerp_complex_ps(__m256 Ar, __m256 Ai, __m256 Br, __m256 Bi, __m256 w,
                                           __m256 *__restrict Xr, __m256 *__restrict Xi)
{
    // ---- Constants ----
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);

    // Guard constants matching the scalar implementation:
    // R0 = eps^3 ≈ 1.694066e-21  – added to Phase and sinPhase to keep the
    //   ratio sin(w*Phase)/sin(Phase) well-defined as Phase → 0.
    //   When Phase ≈ 0 (A ≈ B), both numerator and denominator approach R0,
    //   giving the correct limit wp → w, wn → 1-w.
    // R1 = eps   ≈ 1.192093e-7   – amplitude threshold for "tiny" vectors
    const __m256 R0 = _mm256_set1_ps(1.6940659e-21f);
    const __m256 R1 = _mm256_set1_ps(1.1920929e-7f);

    // Transition-zone boundaries: pure-linear below tL, spherical above tS
    const __m256 tL = _mm256_set1_ps(-0.999f);
    const __m256 tS = _mm256_set1_ps(-0.99f);
    const __m256 dT = _mm256_set1_ps(1.0f / (-0.99f - (-0.999f))); // 1 / (tS - tL)

    // ---- Weights ----
    __m256 wB = w;
    __m256 wA = _mm256_sub_ps(one, w);

    // ---- Amplitudes ----
    __m256 ampA = _mm256_sqrt_ps(_mm256_fmadd_ps(Ar, Ar, _mm256_mul_ps(Ai, Ai)));
    __m256 ampB = _mm256_sqrt_ps(_mm256_fmadd_ps(Br, Br, _mm256_mul_ps(Bi, Bi)));

    // ---- Tiny-amplitude masks ----
    __m256 tinyA = _mm256_cmp_ps(ampA, R1, _CMP_LT_OQ);
    __m256 tinyB = _mm256_cmp_ps(ampB, R1, _CMP_LT_OQ);
    __m256 both_tiny = _mm256_and_ps(tinyA, tinyB);
    __m256 either_tiny = _mm256_or_ps(tinyA, tinyB);

    // ---- Normalise (safe division: replace tiny amplitudes with 1 to avoid 0/0) ----
    __m256 inv_ampA = _mm256_div_ps(one, _mm256_blendv_ps(ampA, one, tinyA));
    __m256 inv_ampB = _mm256_div_ps(one, _mm256_blendv_ps(ampB, one, tinyB));

    // Normalised direction; zeroed where amplitude is tiny
    __m256 gAr = _mm256_andnot_ps(tinyA, _mm256_mul_ps(Ar, inv_ampA));
    __m256 gAi = _mm256_andnot_ps(tinyA, _mm256_mul_ps(Ai, inv_ampA));
    __m256 gBr = _mm256_andnot_ps(tinyB, _mm256_mul_ps(Br, inv_ampB));
    __m256 gBi = _mm256_andnot_ps(tinyB, _mm256_mul_ps(Bi, inv_ampB));

    // ---- Cosine of phase angle between normalised directions ----
    // Set to -1 if either amplitude is tiny (forces linear path)
    __m256 cPhase = _mm256_fmadd_ps(gAr, gBr, _mm256_mul_ps(gAi, gBi));
    cPhase = _mm256_blendv_ps(cPhase, neg_one, either_tiny);

    // ---- Region masks ----
    // linear_int: cPhase < tS  →  need linear result (pure linear or transition zone)
    // do_slerp:   cPhase > tL  →  need spherical result (pure spherical or transition zone)
    __m256 linear_int = _mm256_cmp_ps(cPhase, tS, _CMP_LT_OQ);
    __m256 do_slerp = _mm256_cmp_ps(cPhase, tL, _CMP_GT_OQ);

    // ---- Linear fallback (computed for all lanes, selected by mask later) ----
    __m256 fXr = _mm256_fmadd_ps(wA, Ar, _mm256_mul_ps(wB, Br));
    __m256 fXi = _mm256_fmadd_ps(wA, Ai, _mm256_mul_ps(wB, Bi));

    // ---- Spherical path (computed for all lanes, selected by mask later) ----
    // Clamp cPhase to [-1, 1] for acos and sqrt safety.
    // Upper clamp: FMA dot product of approximately-unit vectors can exceed 1.0.
    // Lower clamp: prevents sqrt(1 - c²) from receiving a negative argument in
    //   masked-out lanes where cPhase < -1 due to FMA rounding.  Those lanes are
    //   never selected (cPhase < tL = -0.999 → !do_slerp), but SIMD computes
    //   them anyway — without the clamp, sqrt of negative produces NaN.
    __m256 cPhase_clamped = _mm256_max_ps(_mm256_min_ps(cPhase, one), neg_one);

    // Phase angle: still needed for sin(w*Phase) / sin(Phase) weight computation.
    // R0 guard ensures Phase > 0 even when cPhase = 1 exactly (A ≡ B), so that
    // sin(w*Phase) ≈ w*R0 and sinPhase ≈ R0, giving the correct ratio w.
    __m256 Phase = _mm256_add_ps(_fm256_acos256_ps(cPhase_clamped), R0);

    // sin(Phase) via Pythagorean identity: sin(Ω) = sqrt(1 - cos²Ω).
    // This replaces a full sincos polynomial call with a single FMA + sqrt (~0.5 ULP),
    // eliminating ~2 ULP of error from the polynomial approximation.
    // R0 guard matches the one on Phase: when cPhase = 1 exactly, both Phase and
    // sinPhase are ≈ R0, keeping the ratio well-defined.
    __m256 sinPhase = _mm256_add_ps(_mm256_sqrt_ps(_mm256_fnmadd_ps(cPhase_clamped, cPhase_clamped, one)), R0);
    __m256 sPhase = _mm256_div_ps(one, sinPhase);

    // SLERP weight factors: wp = sin(wB*Phase) / sin(Phase)
    //                       wn = sin(wA*Phase) / sin(Phase)
    __m256 sinWB, cosWB_unused, sinWA, cosWA_unused;
    _fm256_sincos256_ps(_mm256_mul_ps(wB, Phase), &sinWB, &cosWB_unused);
    _fm256_sincos256_ps(_mm256_mul_ps(wA, Phase), &sinWA, &cosWA_unused);

    __m256 wp = _mm256_mul_ps(sinWB, sPhase);
    __m256 wn = _mm256_mul_ps(sinWA, sPhase);

    // Spherical interpolation of normalised direction
    __m256 gXr = _mm256_fmadd_ps(wn, gAr, _mm256_mul_ps(wp, gBr));
    __m256 gXi = _mm256_fmadd_ps(wn, gAi, _mm256_mul_ps(wp, gBi));

    // Linear interpolation of amplitude
    __m256 ampX = _mm256_fmadd_ps(wA, ampA, _mm256_mul_ps(wB, ampB));

    // Pure spherical result = direction * amplitude
    __m256 sXr = _mm256_mul_ps(gXr, ampX);
    __m256 sXi = _mm256_mul_ps(gXi, ampX);

    // ---- Transition-zone blending (tL < cPhase < tS) ----
    // m = (tS - cPhase) / (tS - tL)  ∈ [0, 1],  n = 1 - m
    // result = n * spherical + m * linear
    __m256 m = _mm256_mul_ps(_mm256_sub_ps(tS, cPhase), dT);
    __m256 n = _mm256_sub_ps(one, m);
    __m256 tXr = _mm256_fmadd_ps(n, sXr, _mm256_mul_ps(m, fXr));
    __m256 tXi = _mm256_fmadd_ps(n, sXi, _mm256_mul_ps(m, fXi));

    // ---- Assemble results per region ----
    // Regions are mutually exclusive:
    //   cPhase >= tS              →  pure spherical    (do_slerp & !linear_int)
    //   tL < cPhase < tS          →  transition blend  (do_slerp &  linear_int)
    //   cPhase <= tL              →  pure linear       (!do_slerp, linear_int is always true here)
    //   both amplitudes tiny      →  zero              (overrides everything)
    __m256 transition = _mm256_and_ps(do_slerp, linear_int);
    __m256 pure_slerp = _mm256_andnot_ps(linear_int, do_slerp);

    // Start with linear result (covers the pure-linear region)
    __m256 resXr = fXr;
    __m256 resXi = fXi;

    // Overwrite transition-zone lanes
    resXr = _mm256_blendv_ps(resXr, tXr, transition);
    resXi = _mm256_blendv_ps(resXi, tXi, transition);

    // Overwrite pure-spherical lanes
    resXr = _mm256_blendv_ps(resXr, sXr, pure_slerp);
    resXi = _mm256_blendv_ps(resXi, sXi, pure_slerp);

    // Zero out lanes where both amplitudes are negligible
    resXr = _mm256_andnot_ps(both_tiny, resXr);
    resXi = _mm256_andnot_ps(both_tiny, resXi);

    *Xr = resXr;
    *Xi = resXi;
}

#endif