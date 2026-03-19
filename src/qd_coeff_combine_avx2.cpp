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

#include <immintrin.h>
#include <cstring>
#include "fastmath_avx2.h"
#include "quadriga_lib_avx2_functions.hpp"

// ---------------------------------------------------------------------------
// Jones product kernel (8 float lanes)
// ---------------------------------------------------------------------------
// Computes the complex MIMO coefficient from interpolated antenna patterns
// and the 2x2 polarization transfer matrix for 8 TX-RX links in parallel.
//
//   c = Vr * M_VV * Vt  +  Hr * M_HV * Vt  +  Vr * M_VH * Ht  +  Hr * M_HH * Ht
//
// Each term is a complex triple product Re/Im{Rx * M * Tx}, factored as:
//   U = Rx_re*M_re - Rx_im*M_im
//   V = Rx_re*M_im + Rx_im*M_re
//   re += U*Tx_re - V*Tx_im
//   im += V*Tx_re + U*Tx_im
//
// This factorisation halves the number of multiplies vs. the direct expansion
// (8 mul + 24 FMA across all 4 components, down from 32 mul + 32 FMA).
static inline void jones_product_8(__m256 Vrr, __m256 Vri, __m256 Hrr, __m256 Hri,
                                   __m256 Vtr, __m256 Vti, __m256 Htr, __m256 Hti,
                                   __m256 M0, __m256 M1, __m256 M2, __m256 M3,
                                   __m256 M4, __m256 M5, __m256 M6, __m256 M7,
                                   __m256 *__restrict re_out, __m256 *__restrict im_out)
{
    __m256 Q, T, U, V;

    // ---- VV component: Vr * M_VV * Vt ----
    Q = _mm256_mul_ps(Vri, M1);      // Vri * Im_VV
    T = _mm256_mul_ps(Vri, M0);      // Vri * Re_VV
    U = _mm256_fmsub_ps(Vrr, M0, Q); // Vrr*Re_VV - Vri*Im_VV
    V = _mm256_fmadd_ps(Vrr, M1, T); // Vrr*Im_VV + Vri*Re_VV

    __m256 re = _mm256_mul_ps(U, Vtr); // U * Vtr
    re = _mm256_fnmadd_ps(V, Vti, re); // - V * Vti
    __m256 im = _mm256_mul_ps(V, Vtr); // V * Vtr
    im = _mm256_fmadd_ps(U, Vti, im);  // + U * Vti

    // ---- HV component: Hr * M_HV * Vt ----
    Q = _mm256_mul_ps(Hri, M3);
    T = _mm256_mul_ps(Hri, M2);
    U = _mm256_fmsub_ps(Hrr, M2, Q);
    V = _mm256_fmadd_ps(Hrr, M3, T);

    re = _mm256_fmadd_ps(U, Vtr, re);
    re = _mm256_fnmadd_ps(V, Vti, re);
    im = _mm256_fmadd_ps(V, Vtr, im);
    im = _mm256_fmadd_ps(U, Vti, im);

    // ---- VH component: Vr * M_VH * Ht ----
    Q = _mm256_mul_ps(Vri, M5);
    T = _mm256_mul_ps(Vri, M4);
    U = _mm256_fmsub_ps(Vrr, M4, Q);
    V = _mm256_fmadd_ps(Vrr, M5, T);

    re = _mm256_fmadd_ps(U, Htr, re);
    re = _mm256_fnmadd_ps(V, Hti, re);
    im = _mm256_fmadd_ps(V, Htr, im);
    im = _mm256_fmadd_ps(U, Hti, im);

    // ---- HH component: Hr * M_HH * Ht ----
    Q = _mm256_mul_ps(Hri, M7);
    T = _mm256_mul_ps(Hri, M6);
    U = _mm256_fmsub_ps(Hrr, M6, Q);
    V = _mm256_fmadd_ps(Hrr, M7, T);

    re = _mm256_fmadd_ps(U, Htr, re);
    re = _mm256_fnmadd_ps(V, Hti, re);
    im = _mm256_fmadd_ps(V, Htr, im);
    im = _mm256_fmadd_ps(U, Hti, im);

    *re_out = re;
    *im_out = im;
}

// ---------------------------------------------------------------------------
// Phase rotation kernel (8 float lanes)
// ---------------------------------------------------------------------------
// Applies:  coeff = (re*cp + im*sp,  im*cp - re*sp) * amplitude
//
static inline void phase_rotate_8(__m256 re, __m256 im, __m256 sp, __m256 cp, __m256 amp,
                                  __m256 *__restrict out_re, __m256 *__restrict out_im)
{
    __m256 cr = _mm256_fmadd_ps(re, cp, _mm256_mul_ps(im, sp));
    __m256 ci = _mm256_fmsub_ps(im, cp, _mm256_mul_ps(re, sp));
    *out_re = _mm256_mul_ps(cr, amp);
    *out_im = _mm256_mul_ps(ci, amp);
}

// ---------------------------------------------------------------------------
// Float specialization
// ---------------------------------------------------------------------------
// All arithmetic in single precision. Phase fmod keeps the sincos input
// small so the float-only Cody-Waite range reduction stays accurate.
template <>
void qd_coeff_combine_avx2<float>(const float *__restrict pVrr, const float *__restrict pVri,
                                  const float *__restrict pHrr, const float *__restrict pHri,
                                  const float *__restrict pVtr, const float *__restrict pVti,
                                  const float *__restrict pHtr, const float *__restrict pHti,
                                  const float *__restrict pM,
                                  const float *__restrict p_delays,
                                  float wave_number,
                                  float wavelength,
                                  float path_amplitude,
                                  float *__restrict p_coeff_re,
                                  float *__restrict p_coeff_im,
                                  size_t n_links)
{
    // Broadcast polarization transfer matrix
    __m256 M0 = _mm256_set1_ps(pM[0]), M1 = _mm256_set1_ps(pM[1]);
    __m256 M2 = _mm256_set1_ps(pM[2]), M3 = _mm256_set1_ps(pM[3]);
    __m256 M4 = _mm256_set1_ps(pM[4]), M5 = _mm256_set1_ps(pM[5]);
    __m256 M6 = _mm256_set1_ps(pM[6]), M7 = _mm256_set1_ps(pM[7]);

    // Broadcast scalar constants
    __m256 vWN = _mm256_set1_ps(wave_number);
    __m256 vWL = _mm256_set1_ps(wavelength);
    __m256 vInvWL = _mm256_set1_ps(1.0f / wavelength);
    __m256 vAmp = _mm256_set1_ps(path_amplitude);

    // Mask LUT for tail handling: first 8 entries are all-ones, next 8 are zero.
    // Selecting at offset (8 - tail) gives the correct mask for 'tail' active lanes.
    alignas(32) static const int32_t mask_lut[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};

    size_t n8 = n_links & ~size_t(7); // largest multiple of 8 <= n_links
    size_t R = 0;

    for (; R < n_links; R += 8)
    {
        __m256 Vrr, Vri, Hrr, Hri, Vtr, Vti, Htr, Hti, dl;

        if (R < n8) // Full 8-wide vector
        {
            Vrr = _mm256_loadu_ps(pVrr + R);
            Vri = _mm256_loadu_ps(pVri + R);
            Hrr = _mm256_loadu_ps(pHrr + R);
            Hri = _mm256_loadu_ps(pHri + R);
            Vtr = _mm256_loadu_ps(pVtr + R);
            Vti = _mm256_loadu_ps(pVti + R);
            Htr = _mm256_loadu_ps(pHtr + R);
            Hti = _mm256_loadu_ps(pHti + R);
            dl = _mm256_loadu_ps(p_delays + R);
        }
        else // Tail: masked load pads inactive lanes with zero
        {
            __m256i mask = _mm256_loadu_si256((const __m256i *)(mask_lut + 8 - (n_links - R)));
            Vrr = _mm256_maskload_ps(pVrr + R, mask);
            Vri = _mm256_maskload_ps(pVri + R, mask);
            Hrr = _mm256_maskload_ps(pHrr + R, mask);
            Hri = _mm256_maskload_ps(pHri + R, mask);
            Vtr = _mm256_maskload_ps(pVtr + R, mask);
            Vti = _mm256_maskload_ps(pVti + R, mask);
            Htr = _mm256_maskload_ps(pHtr + R, mask);
            Hti = _mm256_maskload_ps(pHti + R, mask);
            dl = _mm256_maskload_ps(p_delays + R, mask);
        }

        // Jones product
        __m256 re, im;
        jones_product_8(Vrr, Vri, Hrr, Hri, Vtr, Vti, Htr, Hti,
                        M0, M1, M2, M3, M4, M5, M6, M7, &re, &im);

        // Phase = wave_number * fmod(dl, wavelength)
        __m256 ratio = _mm256_mul_ps(dl, vInvWL);
        __m256 trunc_r = _mm256_round_ps(ratio, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256 rem = _mm256_fnmadd_ps(trunc_r, vWL, dl); // dl - trunc(dl/wl)*wl
        __m256 phase = _mm256_mul_ps(vWN, rem);

        // Sincos (float range reduction, safe because fmod keeps phase small)
        __m256 sp, cp;
        _fm256_sincos256_ps(phase, &sp, &cp);

        // Phase rotation + amplitude scaling
        __m256 out_re, out_im;
        phase_rotate_8(re, im, sp, cp, vAmp, &out_re, &out_im);

        // Store
        if (R < n8)
        {
            _mm256_storeu_ps(p_coeff_re + R, out_re);
            _mm256_storeu_ps(p_coeff_im + R, out_im);
        }
        else
        {
            __m256i mask = _mm256_loadu_si256((const __m256i *)(mask_lut + 8 - (n_links - R)));
            _mm256_maskstore_ps(p_coeff_re + R, mask, out_re);
            _mm256_maskstore_ps(p_coeff_im + R, mask, out_im);
        }
    }
}

// ---------------------------------------------------------------------------
// Double specialization
// ---------------------------------------------------------------------------
// Strategy: convert antenna patterns from double to float, run the Jones
// product in 8-wide float SIMD for full throughput.  For the phase, keep
// delays in double for the fmod and feed the double-precision phases into
// _fm256_sincos256_pd which does Cody-Waite range reduction in double
// before evaluating the polynomial in float.  Results are converted back
// to double for storage.
//
// This gives the full 8x SIMD width while maintaining double-precision
// accuracy in the phase-critical path.
//

// Helper: load 8 consecutive doubles, convert to 8 packed floats
static inline __m256 load_8d_as_8f(const double *__restrict p)
{
    __m128 lo = _mm256_cvtpd_ps(_mm256_loadu_pd(p));
    __m128 hi = _mm256_cvtpd_ps(_mm256_loadu_pd(p + 4));
    return _mm256_set_m128(hi, lo);
}

// Helper: store 8 packed floats as 8 consecutive doubles
static inline void store_8f_as_8d(double *__restrict p, __m256 v)
{
    _mm256_storeu_pd(p, _mm256_cvtps_pd(_mm256_castps256_ps128(v)));
    _mm256_storeu_pd(p + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)));
}

template <>
void qd_coeff_combine_avx2<double>(const double *__restrict pVrr, const double *__restrict pVri,
                                   const double *__restrict pHrr, const double *__restrict pHri,
                                   const double *__restrict pVtr, const double *__restrict pVti,
                                   const double *__restrict pHtr, const double *__restrict pHti,
                                   const double *__restrict pM,
                                   const double *__restrict p_delays,
                                   double wave_number,
                                   double wavelength,
                                   double path_amplitude,
                                   double *__restrict p_coeff_re,
                                   double *__restrict p_coeff_im,
                                   size_t n_links)
{
    // Broadcast polarization transfer matrix (convert to float once)
    __m256 M0 = _mm256_set1_ps((float)pM[0]), M1 = _mm256_set1_ps((float)pM[1]);
    __m256 M2 = _mm256_set1_ps((float)pM[2]), M3 = _mm256_set1_ps((float)pM[3]);
    __m256 M4 = _mm256_set1_ps((float)pM[4]), M5 = _mm256_set1_ps((float)pM[5]);
    __m256 M6 = _mm256_set1_ps((float)pM[6]), M7 = _mm256_set1_ps((float)pM[7]);

    // Scalar constants (double for phase path, float for amplitude)
    __m256d vWN_d = _mm256_set1_pd(wave_number);
    __m256d vWL_d = _mm256_set1_pd(wavelength);
    __m256d vInvWL_d = _mm256_set1_pd(1.0 / wavelength);
    __m256 vAmp = _mm256_set1_ps((float)path_amplitude);

    size_t n8 = n_links & ~size_t(7);
    size_t R = 0;

    for (; R < n8; R += 8)
    {
        // Load antenna patterns: 8 doubles → 8 floats
        __m256 Vrr = load_8d_as_8f(pVrr + R), Vri = load_8d_as_8f(pVri + R);
        __m256 Hrr = load_8d_as_8f(pHrr + R), Hri = load_8d_as_8f(pHri + R);
        __m256 Vtr = load_8d_as_8f(pVtr + R), Vti = load_8d_as_8f(pVti + R);
        __m256 Htr = load_8d_as_8f(pHtr + R), Hti = load_8d_as_8f(pHti + R);

        // Jones product in float
        __m256 re, im;
        jones_product_8(Vrr, Vri, Hrr, Hri, Vtr, Vti, Htr, Hti,
                        M0, M1, M2, M3, M4, M5, M6, M7, &re, &im);

        // Phase computation in double: phase = wave_number * fmod(dl, wavelength)
        __m256d dl_lo = _mm256_loadu_pd(p_delays + R);
        __m256d dl_hi = _mm256_loadu_pd(p_delays + R + 4);

        __m256d r_lo = _mm256_mul_pd(dl_lo, vInvWL_d);
        __m256d r_hi = _mm256_mul_pd(dl_hi, vInvWL_d);
        __m256d t_lo = _mm256_round_pd(r_lo, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256d t_hi = _mm256_round_pd(r_hi, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256d rem_lo = _mm256_fnmadd_pd(t_lo, vWL_d, dl_lo);
        __m256d rem_hi = _mm256_fnmadd_pd(t_hi, vWL_d, dl_hi);
        __m256d phase_lo = _mm256_mul_pd(vWN_d, rem_lo);
        __m256d phase_hi = _mm256_mul_pd(vWN_d, rem_hi);

        // Sincos with double-precision range reduction, float polynomial
        __m256 sp, cp;
        _fm256_sincos256_pd(phase_lo, phase_hi, &sp, &cp);

        // Phase rotation + amplitude scaling (float)
        __m256 out_re, out_im;
        phase_rotate_8(re, im, sp, cp, vAmp, &out_re, &out_im);

        // Store: convert float results back to double
        store_8f_as_8d(p_coeff_re + R, out_re);
        store_8f_as_8d(p_coeff_im + R, out_im);
    }

    // ---- Tail: zero-padded buffers for the remaining 1-7 links ----
    if (R < n_links)
    {
        size_t tail = n_links - R;

        // Zero-initialised aligned buffers for pattern data (converted to float)
        alignas(32) float bVrr[8] = {}, bVri[8] = {}, bHrr[8] = {}, bHri[8] = {};
        alignas(32) float bVtr[8] = {}, bVti[8] = {}, bHtr[8] = {}, bHti[8] = {};

        // Zero-initialised aligned buffers for delays (kept in double)
        alignas(32) double bDl[8] = {};

        for (size_t i = 0; i < tail; ++i)
        {
            bVrr[i] = (float)pVrr[R + i];
            bVri[i] = (float)pVri[R + i];
            bHrr[i] = (float)pHrr[R + i];
            bHri[i] = (float)pHri[R + i];
            bVtr[i] = (float)pVtr[R + i];
            bVti[i] = (float)pVti[R + i];
            bHtr[i] = (float)pHtr[R + i];
            bHti[i] = (float)pHti[R + i];
            bDl[i] = p_delays[R + i];
        }

        // Jones product
        __m256 re, im;
        jones_product_8(_mm256_load_ps(bVrr), _mm256_load_ps(bVri),
                        _mm256_load_ps(bHrr), _mm256_load_ps(bHri),
                        _mm256_load_ps(bVtr), _mm256_load_ps(bVti),
                        _mm256_load_ps(bHtr), _mm256_load_ps(bHti),
                        M0, M1, M2, M3, M4, M5, M6, M7, &re, &im);

        // Phase in double
        __m256d dl_lo = _mm256_load_pd(bDl);
        __m256d dl_hi = _mm256_load_pd(bDl + 4);

        __m256d r_lo = _mm256_mul_pd(dl_lo, vInvWL_d);
        __m256d r_hi = _mm256_mul_pd(dl_hi, vInvWL_d);
        __m256d t_lo = _mm256_round_pd(r_lo, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256d t_hi = _mm256_round_pd(r_hi, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256d rem_lo = _mm256_fnmadd_pd(t_lo, vWL_d, dl_lo);
        __m256d rem_hi = _mm256_fnmadd_pd(t_hi, vWL_d, dl_hi);
        __m256d phase_lo = _mm256_mul_pd(vWN_d, rem_lo);
        __m256d phase_hi = _mm256_mul_pd(vWN_d, rem_hi);

        __m256 sp, cp;
        _fm256_sincos256_pd(phase_lo, phase_hi, &sp, &cp);

        __m256 out_re, out_im;
        phase_rotate_8(re, im, sp, cp, vAmp, &out_re, &out_im);

        // Extract valid lanes back to double
        alignas(32) float tmp_re[8], tmp_im[8];
        _mm256_store_ps(tmp_re, out_re);
        _mm256_store_ps(tmp_im, out_im);

        for (size_t i = 0; i < tail; ++i)
        {
            p_coeff_re[R + i] = (double)tmp_re[i];
            p_coeff_im[R + i] = (double)tmp_im[i];
        }
    }
}
