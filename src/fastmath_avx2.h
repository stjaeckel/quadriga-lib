// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// AVX2 accelerated math functions

#ifndef quadriga_lib_fastmath_avx2_H
#define quadriga_lib_fastmath_avx2_H

#include <immintrin.h>

// The fast paths must be inlined into the vector loops; the rarely used large-argument paths are kept out of line so
// that they do not prevent this.
#if defined(_MSC_VER)
#define QD_FM_INLINE static __forceinline
#define QD_FM_NOINLINE static __declspec(noinline)
#define QD_FM_UNLIKELY(x) (x)
#else
#define QD_FM_INLINE static inline __attribute__((always_inline))
#define QD_FM_NOINLINE static __attribute__((noinline, cold, unused))
#define QD_FM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

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

// Reciprocal with one Newton–Raphson refinement: r = 1/x, ~23-bit accuracy.
// rcp gives ~12 bits; one NR step: r₁ = r₀ · (2 − x · r₀)
// Caller must ensure x is not zero or denormal.
static inline __m256 _fm256_rcp_nr_ps(__m256 x)
{
    __m256 r0 = _mm256_rcp_ps(x);
    return _mm256_mul_ps(r0, _mm256_fnmadd_ps(x, r0, _mm256_set1_ps(2.0f)));
}

// Reciprocal square root with one Newton–Raphson refinement: r = 1/sqrt(x), ~23-bit accuracy.
// rsqrt gives ~12 bits; one NR step: r₁ = 0.5 · r₀ · (3 − x · r₀²)
// Caller must ensure x is not zero or denormal.
static inline __m256 _fm256_rsqrt_nr_ps(__m256 x)
{
    __m256 r0 = _mm256_rsqrt_ps(x);
    return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), r0),
                         _mm256_fnmadd_ps(x, _mm256_mul_ps(r0, r0), _mm256_set1_ps(3.0f)));
}

// Polynomial part of the AVX2 sine / cosine (Cephes single-precision polynomials), shared by all sin / cos kernels.
// Input: reduced argument x in [-pi/4, pi/4] (plus rounding) and an index k (negative values are allowed):
//   quadrant = false: k = j is an even octant index, s = sin(x + j * pi/4), c = cos(x + j * pi/4)
//   quadrant = true:  k = q is a quadrant index,     s = sin(x + q * pi/2), c = cos(x + q * pi/2)
// Only bits 1 and 2 of j (bits 0 and 1 of q) are used. Both forms give the same bits for j = 2 * q.
// blend_select: select the sine / cosine polynomial with blendv (faster in fast_sincos, fast_geo2cart and the
// baseband kernel) or with and / andnot / or (faster in the antenna interpolation kernel and SLERP; measured); both
// give the same bits.
// Note: A SINE / COSINE-only version provides no significant performance advantage
// Just compute both and drop the unwanted one
template <bool quadrant = false, bool blend_select = true>
QD_FM_INLINE void _fm256_sincos_poly_ps(__m256 x, __m256i k, __m256 *s, __m256 *c)
{
    constexpr int sh = quadrant ? 1 : 0;             // k = j >> sh
    const __m256i bit1 = _mm256_set1_epi32(2 >> sh); // bit 1 of j
    const __m256i bit2 = _mm256_set1_epi32(4 >> sh); // bit 2 of j
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 n_half = _mm256_set1_ps(-0.5f);

    // Polynomial coeffs (Cephes), single-precision
    const __m256 Sp0 = _mm256_set1_ps(-1.9515295891E-4f);
    const __m256 Sp1 = _mm256_set1_ps(8.3321608736E-3f);
    const __m256 Sp2 = _mm256_set1_ps(-1.6666654611E-1f);

    const __m256 Cp0 = _mm256_set1_ps(2.443315711809948E-005f);
    const __m256 Cp1 = _mm256_set1_ps(-1.388731625493765E-003f);
    const __m256 Cp2 = _mm256_set1_ps(4.166664568298827E-002f);

    // sign flip of the sine where j & 4; of the cosine where (j + 2) & 4 (bit 2 moved to the sign bit)
    __m256 swap_sin = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(k, bit2), 29 + sh));
    __m256 sign_bit_cos = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_and_si256(_mm256_add_epi32(k, bit1), bit2), 29 + sh));

    // polynomials swapped where j & 2 (bit 1 of j moved to the sign bit for blendv, or a mask of all ones)
    __m256 swap = blend_select ? _mm256_castsi256_ps(_mm256_slli_epi32(k, 30 + sh))
                               : _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(k, bit1), bit1));

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
    __m256 sin_approx, cos_approx;
    if (blend_select)
        sin_approx = _mm256_blendv_ps(ys, yc, swap), cos_approx = _mm256_blendv_ps(yc, ys, swap);
    else
        sin_approx = _mm256_or_ps(_mm256_andnot_ps(swap, ys), _mm256_and_ps(swap, yc)),
        cos_approx = _mm256_or_ps(_mm256_andnot_ps(swap, yc), _mm256_and_ps(swap, ys));

    // Apply signs
    *s = _mm256_xor_ps(sin_approx, swap_sin);
    *c = _mm256_xor_ps(cos_approx, sign_bit_cos);
}

// Double-precision range reduction by multiples of 2*pi: returns x - n * 2*pi with n = round(x / (2*pi)), in
// [-pi, pi] (plus rounding) for |x| < 2^45 * 2*pi (n * CW_C1 is exact). Two-constant split CW_C1 + CW_C2 = 2*pi with
// CW_C1 exactly representable in 8 bits, as qd_range_reduce_2pi of the generic kernel. Inf and NaN give NaN.
QD_FM_INLINE __m256d _fm256_range_reduce_2pi_pd(__m256d x)
{
    const __m256d inv_two_pi = _mm256_set1_pd(0.15915494309189533577); // 1 / (2*pi)
    const __m256d CW_C1 = _mm256_set1_pd(6.28125);                     // high bits of 2*pi
    const __m256d CW_C2 = _mm256_set1_pd(1.9353071795864769253e-03);   // 2*pi - CW_C1
    __m256d n = _mm256_round_pd(_mm256_mul_pd(x, inv_two_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_pd(n, CW_C1, x); // exact (Sterbenz)
    return _mm256_fnmadd_pd(n, CW_C2, x);
}

// Octant reduction of 4 doubles: |x| = j * pi/4 + r with an even j and r in [-pi/4, pi/4] (Cody–Waite, two constants).
// j is an int32; for |x| * 4/pi >= 2^31 - 1, inf and NaN the conversion overflows and j is negative.
QD_FM_INLINE void _fm256_octant_reduce_pd(__m256d x, __m128 *r, __m128i *j, __m128 *sign)
{
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

    __m256d sign_d = _mm256_andnot_pd(abs_mask_d, x);         // sign bits
    __m256d ax = _mm256_and_pd(x, abs_mask_d);                // |x|
    __m256d y = _mm256_mul_pd(ax, FOPI_D);                    // |x| * 4/pi
    __m128i jj = _mm256_cvttpd_epi32(y);                      // truncate → 4 int32
    jj = _mm_and_si128(_mm_add_epi32(jj, one_128), inv1_128); // (j+1) & ~1
    __m256d jd = _mm256_cvtepi32_pd(jj);                      // j → double
    ax = _mm256_fnmadd_pd(jd, DP1_D, ax);                     // ax -= j*DP1
    ax = _mm256_fnmadd_pd(jd, DP2_D, ax);                     // ax -= j*DP2

    *r = _mm256_cvtpd_ps(ax);
    *j = jj;
    *sign = _mm256_cvtpd_ps(sign_d);
}

// Quadrant reduction of 8 values given as two halves of doubles, for large arguments: x = q * pi/2 + r with
// q = round(x * 2/pi) kept in double (no int32 overflow) and r in [-pi/4, pi/4] (two-constant split of pi/2, PIO2_1
// exactly representable in 8 bits; error below 1e-8 for |x| < 2^37). Returns r as float and the low 32 bits of q in
// two's complement (taken from the mantissa of q + 1.5 * 2^52). sin(x) = sin(r + q * pi/2), so no sign handling is
// needed. NaN gives NaN in r.
QD_FM_INLINE void _fm256_quadrant_reduce_pd(__m256d xl, __m256d xh, __m256 *r, __m256i *q)
{
    const __m256d TOPI_D = _mm256_set1_pd(0.63661977236758134308); // 2/pi
    const __m256d PIO2_1 = _mm256_set1_pd(1.5703125);
    const __m256d PIO2_2 = _mm256_set1_pd(4.8382679489661923132e-4); // pi/2 - PIO2_1
    const __m256d shift = _mm256_set1_pd(6755399441055744.0);        // 1.5 * 2^52
    // Magic-number rounding: x * 2/pi + 1.5 * 2^52 rounds to an integer (exact only without -ffast-math, which may
    // fold the subtraction below); the low mantissa word of the sum holds q mod 2^32
    __m256d tl = _mm256_fmadd_pd(xl, TOPI_D, shift), th = _mm256_fmadd_pd(xh, TOPI_D, shift);
    __m256d ql = _mm256_sub_pd(tl, shift), qh = _mm256_sub_pd(th, shift);
    xl = _mm256_fnmadd_pd(ql, PIO2_2, _mm256_fnmadd_pd(ql, PIO2_1, xl));
    xh = _mm256_fnmadd_pd(qh, PIO2_2, _mm256_fnmadd_pd(qh, PIO2_1, xh));
    *r = _mm256_set_m128(_mm256_cvtpd_ps(xh), _mm256_cvtpd_ps(xl));
    __m256 bl = _mm256_castpd_ps(tl), bh = _mm256_castpd_ps(th);
    __m256 b = _mm256_shuffle_ps(bl, bh, _MM_SHUFFLE(2, 0, 2, 0)); // l0 l1 h0 h1 | l2 l3 h2 h3
    b = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(b), _MM_SHUFFLE(3, 1, 2, 0)));
    *q = _mm256_castps_si256(b);
}

// Lanes with |x| > 2^37 (and inf) of two halves of doubles are reduced by multiples of 2*pi, repeatedly above
// 2^45 * 2*pi, where one step is not exact (each step shrinks |x| by about 2^-50); inf becomes NaN. The other lanes
// are not changed.
QD_FM_INLINE void _fm256_reduce_very_large_pd(__m256d *xl, __m256d *xh)
{
    const __m256d abs_mask_d = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
    const __m256d limit = _mm256_set1_pd(137438953472.0); // 2^37
    __m256d big_l = _mm256_cmp_pd(_mm256_and_pd(*xl, abs_mask_d), limit, _CMP_GT_OQ);
    __m256d big_h = _mm256_cmp_pd(_mm256_and_pd(*xh, abs_mask_d), limit, _CMP_GT_OQ);
    while (_mm256_movemask_pd(_mm256_or_pd(big_l, big_h)))
    {
        *xl = _mm256_blendv_pd(*xl, _fm256_range_reduce_2pi_pd(*xl), big_l);
        *xh = _mm256_blendv_pd(*xh, _fm256_range_reduce_2pi_pd(*xh), big_h);
        big_l = _mm256_cmp_pd(_mm256_and_pd(*xl, abs_mask_d), limit, _CMP_GT_OQ);
        big_h = _mm256_cmp_pd(_mm256_and_pd(*xh, abs_mask_d), limit, _CMP_GT_OQ);
    }
}

// sin / cos of 8 large arguments given as two halves of doubles: very large lanes are first reduced by multiples of
// 2*pi (_fm256_reduce_very_large_pd), then all lanes are reduced to a quadrant in double precision. Max. error 1.3e-7
// for |x| <= 1e10; |s|, |c| <= 1 for all finite x; inf and NaN give NaN.
QD_FM_NOINLINE void _fm256_sincos256_big_pd(__m256d xl, __m256d xh, __m256 *s, __m256 *c)
{
    _fm256_reduce_very_large_pd(&xl, &xh);
    __m256 r;
    __m256i q;
    _fm256_quadrant_reduce_pd(xl, xh, &r, &q);
    _fm256_sincos_poly_ps<true>(r, q, s, c);
}

// sin / cos of 8 large float arguments (|x| > 2^22 in some lanes): as _fm256_sincos256_big_pd, inline, with the check
// for very large lanes done on the float input.
QD_FM_INLINE void _fm256_sincos256_big_ps(__m256 x, __m256 *s, __m256 *c)
{
    const __m256 inv_sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 limit = _mm256_set1_ps(137438953472.0f); // 2^37
    __m256d xl = _mm256_cvtps_pd(_mm256_castps256_ps128(x)), xh = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
    if (QD_FM_UNLIKELY(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_and_ps(x, inv_sign_mask), limit, _CMP_GT_OQ))))
        _fm256_reduce_very_large_pd(&xl, &xh);
    __m256 r;
    __m256i q;
    _fm256_quadrant_reduce_pd(xl, xh, &r, &q);
    _fm256_sincos_poly_ps<true>(r, q, s, c);
}

// Double-input variant: takes two __m256d (8 doubles as low/high halves),
// does Cody–Waite range reduction to [-pi/4, pi/4] in double precision,
// converts to float, then evaluates the same Cephes polynomial + reconstruction.
// This avoids the cascaded reduction problem where the float kernel would
// re-reduce an already-reduced input, losing precision near octant boundaries.
// Lanes whose octant index does not fit into an int32 (|x| > 1.68e9, inf, NaN) are computed by the large-argument
// variant and blended in; the other lanes are not changed by this, so the result of a lane does not depend on the
// other lanes. Max. error 1.3e-7 for |x| <= 1e10; |s|, |c| <= 1 for all finite x; inf and NaN give NaN.
QD_FM_INLINE void _fm256_sincos256_pd(__m256d xl, __m256d xh, __m256 *s, __m256 *c)
{
    __m128 rl, rh, sl, sh;
    __m128i jl, jh;
    _fm256_octant_reduce_pd(xl, &rl, &jl, &sl);
    _fm256_octant_reduce_pd(xh, &rh, &jh, &sh);
    __m256i j = _mm256_set_m128i(jh, jl);

    __m256 sin_r, cos_r;
    _fm256_sincos_poly_ps(_mm256_set_m128(rh, rl), j, &sin_r, &cos_r);
    *s = _mm256_xor_ps(sin_r, _mm256_set_m128(sh, sl));
    *c = cos_r;

    if (QD_FM_UNLIKELY(_mm256_movemask_ps(_mm256_castsi256_ps(j)))) // Some lane overflowed: large, inf or NaN
    {
        __m256 sb, cb;
        _fm256_sincos256_big_pd(xl, xh, &sb, &cb);
        *s = _mm256_blendv_ps(*s, sb, _mm256_castsi256_ps(j));
        *c = _mm256_blendv_ps(*c, cb, _mm256_castsi256_ps(j));
    }
}

// Single-precision input, small arguments only: Cody–Waite range reduction in float, exact enough for |x| <= 2^22
// (max. error 1.3e-7); larger arguments give wrong results (see _fm256_sincos256_ps). For callers whose arguments
// are known to be small, and for loops that send vectors with large lanes to _fm256_sincos256_ps
// (_fm256_any_large_ps). blend_select: see _fm256_sincos_poly_ps.
template <bool blend_select = true>
QD_FM_INLINE void _fm256_sincos256_small_ps(__m256 x, __m256 *s, __m256 *c)
{
    // Bit masks
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000u));
    const __m256 inv_sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i pi32_1 = _mm256_set1_epi32(1);
    const __m256i pi32_inv1 = _mm256_set1_epi32(~1);

    // Constants
    const __m256 FOPI = _mm256_set1_ps(1.27323954473516f); // 4/pi

    // Cody–Waite split constants
    const __m256 DP1 = _mm256_set1_ps(-0.78515625f);
    const __m256 DP2 = _mm256_set1_ps(-2.4187564849853515625e-4f);
    const __m256 DP3 = _mm256_set1_ps(-3.77489497744594108e-8f);

    // sign for sine; |x|
    __m256 sign_bit_sin = _mm256_and_ps(x, sign_mask);
    x = _mm256_and_ps(x, inv_sign_mask);

    // Scale by 4/pi and get j = (int)y;  j = (j+1) & (~1)
    __m256 y = _mm256_mul_ps(x, FOPI);
    __m256i j = _mm256_cvttps_epi32(y);
    j = _mm256_add_epi32(j, pi32_1);
    j = _mm256_and_si256(j, pi32_inv1);
    y = _mm256_cvtepi32_ps(j);

    // Range reduction: x = ((x - y*DP1) - y*DP2) - y*DP3 (as fused adds)
    x = _mm256_fmadd_ps(y, DP1, x);
    x = _mm256_fmadd_ps(y, DP2, x);
    x = _mm256_fmadd_ps(y, DP3, x);

    __m256 sin_r, cos_r;
    _fm256_sincos_poly_ps<false, blend_select>(x, j, &sin_r, &cos_r);
    *s = _mm256_xor_ps(sin_r, sign_bit_sin);
    *c = cos_r;
}

// Mask of the lanes that are too large for _fm256_sincos256_small_ps (|x| > 2^22, inf; not NaN)
QD_FM_INLINE __m256 _fm256_large_lanes_ps(__m256 x)
{
    const __m256 inv_sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    return _mm256_cmp_ps(_mm256_and_ps(x, inv_sign_mask), _mm256_set1_ps(4194304.0f), _CMP_GT_OQ);
}

// True if any lane of x is too large for _fm256_sincos256_small_ps
QD_FM_INLINE bool _fm256_any_large_ps(__m256 x)
{
    return _mm256_movemask_ps(_fm256_large_lanes_ps(x)) != 0;
}

// Single-precision input, any argument: lanes with |x| <= 2^22 as _fm256_sincos256_small_ps; lanes with |x| > 2^22
// (and inf) are computed by the large-argument variant (quadrant reduction in double precision) from the same float
// value and blended in (if all lanes are large, only by that), so the result of a lane does not depend on the other
// lanes. Max. error 1.3e-7 for |x| <= 1e10; |s|, |c| <= 1 for all finite x; inf and NaN give NaN.
// blend_select: see _fm256_sincos_poly_ps.
template <bool blend_select = true>
QD_FM_INLINE void _fm256_sincos256_ps(__m256 x, __m256 *s, __m256 *c)
{
    const __m256 big = _fm256_large_lanes_ps(x);
    const int big_lanes = _mm256_movemask_ps(big);
    if (QD_FM_UNLIKELY(big_lanes == 0xFF)) // All lanes large
    {
        _fm256_sincos256_big_ps(x, s, c);
        return;
    }
    _fm256_sincos256_small_ps<blend_select>(x, s, c);
    if (QD_FM_UNLIKELY(big_lanes)) // Some lanes large: blend in the large-argument results
    {
        __m256 sb, cb;
        _fm256_sincos256_big_ps(x, &sb, &cb);
        *s = _mm256_blendv_ps(*s, sb, big);
        *c = _mm256_blendv_ps(*c, cb, big);
    }
}

// sin and cos of 2*pi*r for r in [-0.5, 0.5] (one period, already reduced in double precision by the caller; used by
// the baseband DFT kernel). Quadrant q = nearest integer of 4*r (exact in float), t = (4*r - q) * pi/2 in
// [-pi/4, pi/4], then the shared polynomial part.
QD_FM_INLINE void _fm256_sincos_2pi_reduced_ps(__m256 r, __m256 *s, __m256 *c)
{
    const __m256 four = _mm256_set1_ps(4.0f);
    const __m256 pi_2 = _mm256_set1_ps(1.57079632679489662f);
    __m256 u = _mm256_mul_ps(r, four);                                            // In [-2, 2], exact
    __m256 n = _mm256_round_ps(u, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); // Quadrant -2 ... 2
    _fm256_sincos_poly_ps<true>(_mm256_mul_ps(_mm256_sub_ps(u, n), pi_2), _mm256_cvtps_epi32(n), s, c);
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
// Special cases (IEEE-compliant, as std::atan2):
//   atan2(±y, 0)   → ±pi/2        atan2(0, +x)    → 0
//   atan2(±0, -x)  → ±pi          atan2(±0, +0)   → ±0
//   atan2(±0, -0)  → ±pi          atan2(±inf, ±inf) → ±pi/4, ±3pi/4
//   NaN in x or y  → NaN
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

    // Both infinite (mn = inf): use |x| = |y| = 1, i.e. the diagonal of the quadrant
    __m256 both_inf = _mm256_cmp_ps(mn, _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000)), _CMP_EQ_OQ);
    mn = _mm256_blendv_ps(mn, one, both_inf);
    mx = _mm256_blendv_ps(mx, one, both_inf);

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

    // Quadrant fixup: if x is negative (sign bit set, also for x = -0)  →  r = pi - r
    r = _mm256_blendv_ps(r, _mm256_sub_ps(pi, r), x);

    // Apply sign of y (r is non-negative here, so OR with sign bit suffices)
    r = _mm256_or_ps(r, _mm256_and_ps(y, sign_mask));

    // NaN in x or y: min / max above drop a NaN, so set the result to NaN explicitly
    return _mm256_or_ps(r, _mm256_cmp_ps(x, y, _CMP_UNORD_Q));
}

// Convert geographic (azimuth, elevation) to Cartesian (x, y, z)
// Optionally returns the intermediate sincos values via restrict pointers.
// Pass NULL for any intermediate you don't need.
//   x = cos(el) * cos(az)
//   y = cos(el) * sin(az)
//   z = sin(el)
static inline void _fm256_geo2cart_ps(__m256 az, __m256 el, __m256 * x, __m256 * y, __m256 * z,
                                      __m256 * sAZ = nullptr, __m256 * cAZ = nullptr,
                                      __m256 * sEL = nullptr, __m256 * cEL = nullptr)
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

// Convert Cartesian (x, y, z) to geographic (azimuth, elevation, length)
//   len = sqrt(x^2 + y^2 + z^2)
//   az  = atan2(y, x)
//   el  = asin(clamp(z / len, -1, 1))
// Clamp guards against len == 0 (produces NaN/Inf from div) and FMA rounding.
static inline void _fm256_cart2geo_ps(__m256 x, __m256 y, __m256 z,
                                      __m256 * az, __m256 * el, __m256 * len)
{
    __m256 r2 = _mm256_fmadd_ps(z, z, _mm256_fmadd_ps(y, y, _mm256_mul_ps(x, x)));
    __m256 r  = _mm256_sqrt_ps(r2);
    *len = r;
    *az  = _fm256_atan2256_ps(y, x);
    // Mask: select z/r where r > 0, else 0 — matches GENERIC behavior at origin
    __m256 mask = _mm256_cmp_ps(r, _mm256_setzero_ps(), _CMP_GT_OQ);
    __m256 zn   = _mm256_and_ps(mask, _mm256_div_ps(z, r));
    *el  = _fm256_asin256_ps(_fm256_clamp_pm1_ps(zn));
    *el  = _mm256_or_ps(*el, _mm256_cmp_ps(r, r, _CMP_UNORD_Q)); // NaN length gives NaN
}

// Half of _fm256_cart2geo_pd (4 doubles): length, x / length and y / length (x and y unscaled if the length is 0, inf
// or NaN), z / length (0 if the length is not positive)
QD_FM_INLINE void _fm256_cart2geo_half_pd(__m256d x, __m256d y, __m256d z, __m128 *xs, __m128 *ys, __m128 *zn,
                                          __m256d *len)
{
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d inf = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FF0000000000000LL));
    *len = _mm256_sqrt_pd(_mm256_fmadd_pd(z, z, _mm256_fmadd_pd(y, y, _mm256_mul_pd(x, x))));
    const __m256d inv = _mm256_div_pd(one, *len);
    const __m256d pos = _mm256_cmp_pd(*len, _mm256_setzero_pd(), _CMP_GT_OQ);
    const __m256d scale = _mm256_blendv_pd(one, inv, _mm256_and_pd(pos, _mm256_cmp_pd(*len, inf, _CMP_LT_OQ)));
    *xs = _mm256_cvtpd_ps(_mm256_mul_pd(x, scale));
    *ys = _mm256_cvtpd_ps(_mm256_mul_pd(y, scale));
    *zn = _mm256_cvtpd_ps(_mm256_and_pd(pos, _mm256_mul_pd(z, inv)));
}

// Double-precision input (8 values as low / high halves): the length and z / length are computed in double precision,
// so that vectors whose squared components underflow or overflow in float keep their direction; x and y are scaled
// by 1 / length before they are narrowed to float for atan2. Returns az and el in float, the length in double.
// Special values as in the generic kernel: length 0 gives el = 0; NaN gives NaN in az, el and len.
QD_FM_INLINE void _fm256_cart2geo_pd(__m256d xl, __m256d xh, __m256d yl, __m256d yh, __m256d zl, __m256d zh,
                                     __m256 *az, __m256 *el, __m256d *len_l, __m256d *len_h)
{
    __m128 xs_l, xs_h, ys_l, ys_h, zn_l, zn_h;
    _fm256_cart2geo_half_pd(xl, yl, zl, &xs_l, &ys_l, &zn_l, len_l);
    _fm256_cart2geo_half_pd(xh, yh, zh, &xs_h, &ys_h, &zn_h, len_h);
    *az = _fm256_atan2256_ps(_mm256_set_m128(ys_h, ys_l), _mm256_set_m128(xs_h, xs_l));
    *el = _fm256_asin256_ps(_fm256_clamp_pm1_ps(_mm256_set_m128(zn_h, zn_l)));
    // NaN length gives NaN: the all-ones mask of a NaN length converts to a NaN float
    const __m128 nan_l = _mm256_cvtpd_ps(_mm256_cmp_pd(*len_l, *len_l, _CMP_UNORD_Q));
    const __m128 nan_h = _mm256_cvtpd_ps(_mm256_cmp_pd(*len_h, *len_h, _CMP_UNORD_Q));
    *el = _mm256_or_ps(*el, _mm256_set_m128(nan_h, nan_l));
}

// Convert complex RE/IM to polar form (magnitude and phase angle)
// abs = sqrt(re² + im²),  arg = atan2(im, re)
static inline void _fm256_absarg_ps(__m256 re, __m256 im, __m256 * abs, __m256 * arg)
{
    *abs = _mm256_sqrt_ps(_mm256_fmadd_ps(re, re, _mm256_mul_ps(im, im)));
    *arg = _fm256_atan2256_ps(im, re);
}

// AVX2 accelerated spherical interpolation (SLERP) for complex value pairs
// Processes 8 complex pairs simultaneously with per-lane interpolation weight.
// Behaviour matches the scalar slerp_complex_mf<float> template except:
//   - sin(Phase) computed via sqrt(1 - cos²Phase) instead of sincos polynomial
//     (eliminates one full sincos call, improves accuracy by ~2 ULP)
//   - Weighted direction sum computed in single precision, with one FMA per component
//   - Nearly parallel directions (cos(Phase) > 0.99998, Phase < 0.0063) use the normalised linear interpolation of
//     the directions times the interpolated amplitude; its phase differs from SLERP by less than 4e-9 rad
//   - Optionally (zero_if_both_tiny = false) the linear result instead of zero when both amplitudes are tiny
//
// Otherwise identical:
//   - Spherical interpolation of normalised direction
//   - Linear interpolation of amplitude
//   - Smooth transition to linear fallback for near-antipodal directions
//   - Zero output when both input amplitudes are negligible (zero_if_both_tiny = true, as slerp_complex_mf)
// The result of a lane depends only on the inputs of that lane.
//
// Parameters (all __m256, i.e. 8 lanes of float):
//   Ar, Ai  – real/imag parts of source A
//   Br, Bi  – real/imag parts of source B
//   w       – interpolation weight per lane  (0 → A, 1 → B)
//   Xr, Xi  – output real/imag parts (written via restrict pointer)
//   zero_if_both_tiny – true: zero where both amplitudes are below 1.19e-7; false: linear interpolation there (as the
//                       generic antenna interpolation)
static inline void _fm256_slerp_complex_ps(__m256 Ar, __m256 Ai, __m256 Br, __m256 Bi, __m256 w,
                                           __m256 * Xr, __m256 * Xi, const bool zero_if_both_tiny = true)
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
    //   R1_sq = R1² ≈ 1.421085e-14 – squared threshold for mag_sq comparison
    const __m256 R0 = _mm256_set1_ps(1.6940659e-21f);
    const __m256 R1_sq = _mm256_set1_ps(1.1920929e-7f * 1.1920929e-7f);

    // Transition-zone boundaries: pure-linear below tL, spherical above tS
    const __m256 tL = _mm256_set1_ps(-0.999f);
    const __m256 tS = _mm256_set1_ps(-0.99f);
    const __m256 dT = _mm256_set1_ps(1.0f / (-0.99f - (-0.999f))); // 1 / (tS - tL)

    // ---- Weights ----
    __m256 wB = w;
    __m256 wA = _mm256_sub_ps(one, w);

    // ---- Squared magnitudes ----
    __m256 mag_sq_A = _mm256_fmadd_ps(Ar, Ar, _mm256_mul_ps(Ai, Ai));
    __m256 mag_sq_B = _mm256_fmadd_ps(Br, Br, _mm256_mul_ps(Bi, Bi));

    // ---- Tiny-amplitude masks (on squared magnitude to avoid sqrt before test) ----
    __m256 tinyA = _mm256_cmp_ps(mag_sq_A, R1_sq, _CMP_LT_OQ);
    __m256 tinyB = _mm256_cmp_ps(mag_sq_B, R1_sq, _CMP_LT_OQ);
    __m256 both_tiny = _mm256_and_ps(tinyA, tinyB);
    __m256 either_tiny = _mm256_or_ps(tinyA, tinyB);

    // ---- Amplitudes & inverse amplitudes via rsqrt+NR ----
    // inv_amp = 1/sqrt(mag_sq),  amp = mag_sq * inv_amp = sqrt(mag_sq)
    // Guard: replace tiny mag_sq with 1.0 so rsqrt doesn't see 0/denorm
    __m256 inv_ampA = _fm256_rsqrt_nr_ps(_mm256_blendv_ps(mag_sq_A, one, tinyA));
    __m256 inv_ampB = _fm256_rsqrt_nr_ps(_mm256_blendv_ps(mag_sq_B, one, tinyB));
    __m256 ampA = _mm256_mul_ps(mag_sq_A, inv_ampA);
    __m256 ampB = _mm256_mul_ps(mag_sq_B, inv_ampB);

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

    // Linear interpolation of amplitude
    __m256 ampX = _mm256_fmadd_ps(wA, ampA, _mm256_mul_ps(wB, ampB));

    // ---- Nearly parallel directions: normalised linear interpolation of the directions ----
    // For cos(Φ) > 0.99998 (Φ < 0.0063 rad) the direction of wA * gA + wB * gB differs from the SLERP direction by
    // less than Φ³/64 (< 4e-9 rad). Its length is sqrt(1 - 2 wA wB (1 - cos(Φ))), short by up to Φ²/8 (5e-6);
    // 1 + wA wB (1 - cos(Φ)) corrects that to 4e-11. (Linear interpolation of the complex values themselves would
    // weight the directions by the amplitudes.) Tiny-amplitude lanes have cPhase forced to -1, so they never enter
    // here. The fast path (all lanes nearly parallel) and the blend at the end give the same lane results.
    __m256 near = _mm256_cmp_ps(cPhase, _mm256_set1_ps(0.99998f), _CMP_GT_OQ);
    const int near_lanes = _mm256_movemask_ps(near);
    __m256 nXr = _mm256_setzero_ps(), nXi = _mm256_setzero_ps();
    if (near_lanes)
    {
        __m256 nscale = _mm256_mul_ps(ampX, _mm256_fmadd_ps(_mm256_mul_ps(wA, wB), _mm256_sub_ps(one, cPhase), one));
        nXr = _mm256_mul_ps(_mm256_fmadd_ps(wA, gAr, _mm256_mul_ps(wB, gBr)), nscale);
        nXi = _mm256_mul_ps(_mm256_fmadd_ps(wA, gAi, _mm256_mul_ps(wB, gBi)), nscale);
        if (near_lanes == 0xFF)
        {
            *Xr = nXr;
            *Xi = nXi;
            return;
        }
    }

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
    __m256 sPhase = _fm256_rcp_nr_ps(sinPhase);

    // SLERP weight factors: wp = sin(wB*Phase) / sin(Phase)
    //                       wn = sin(wA*Phase) / sin(Phase)
    // Since wA = 1 - wB, we have wA*Phase = Phase - wB*Phase, so:
    //   sin(wA*Phase) = sin(Phase)*cos(wB*Phase) - cos(Phase)*sin(wB*Phase)
    // This eliminates one full sincos call (~28 cycles).
    __m256 sinWB, cosWB;
    _fm256_sincos256_small_ps<false>(_mm256_mul_ps(wB, Phase), &sinWB, &cosWB); // wB * Phase in [0, pi]
    __m256 sinWA = _mm256_fmsub_ps(sinPhase, cosWB, _mm256_mul_ps(cPhase_clamped, sinWB));

    __m256 wp = _mm256_mul_ps(sinWB, sPhase);
    __m256 wn = _mm256_mul_ps(sinWA, sPhase);

    // Spherical interpolation of normalised direction
    __m256 gXr = _mm256_fmadd_ps(wn, gAr, _mm256_mul_ps(wp, gBr));
    __m256 gXi = _mm256_fmadd_ps(wn, gAi, _mm256_mul_ps(wp, gBi));

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
    //   cPhase > 0.99998          →  normalised linear (inside the pure spherical region)
    //   both amplitudes tiny      →  zero              (overrides everything, if zero_if_both_tiny)
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

    // Overwrite nearly parallel lanes
    if (near_lanes)
    {
        resXr = _mm256_blendv_ps(resXr, nXr, near);
        resXi = _mm256_blendv_ps(resXi, nXi, near);
    }

    // Zero out lanes where both amplitudes are negligible
    if (zero_if_both_tiny)
    {
        resXr = _mm256_andnot_ps(both_tiny, resXr);
        resXi = _mm256_andnot_ps(both_tiny, resXi);
    }

    *Xr = resXr;
    *Xi = resXi;
}

#endif