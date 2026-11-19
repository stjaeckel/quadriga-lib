// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib_generic_functions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

// sin and cos of 2 * pi * r for r in [-0.5, 0.5] (one period, already reduced in double precision by the caller).
// Same algorithm as the AVX2 DFT kernel (_fm256_sincos_2pi_reduced_ps and the shared polynomial part
// _fm256_sincos_poly_ps in fastmath_avx2.h): quadrant q = nearest integer of 4 * r (magic-number rounding, exact for
// |4 * r| < 2^22, and only without -ffast-math, which may fold (u + magic) - magic to u), t = (4 * r - q) * pi / 2 in
// [-pi/4, pi/4], Cephes single-precision polynomials, quadrant swap / signs.
// Branch-free, so that loops calling it can be vectorized.
static inline void qd_sincos_2pi_reduced(float r, float &s, float &c)
{
    constexpr float magic = 12582912.0f; // 1.5 * 2^23
    const float u = 4.0f * r;
    const float n = (u + magic) - magic;
    const int q = (int)n;
    const float t = (u - n) * 1.57079632679489662f;
    const float z = t * t, z2 = z * z;
    const float ys = ((1.0f - 1.6666654611E-1f * z) + z2 * (8.3321608736E-3f - 1.9515295891E-4f * z)) * t;
    const float yc = (1.0f + z2 * (4.166664568298827E-002f + 2.443315711809948E-005f * z2)) + z * (-0.5f - 1.388731625493765E-003f * z2);
    const float ss = (q & 1) ? yc : ys, cc = (q & 1) ? ys : yc;
    s = (q & 2) ? -ss : ss;
    c = ((q + 1) & 2) ? -cc : cc;
}

template <typename dtype>
void qd_DFT_GENERIC(const dtype *__restrict CFr,    // Channel coefficients, real part, Size [n_ant, n_path]
                    const dtype *__restrict CFi,    // Channel coefficients, imaginary part, Size [n_ant, n_path]
                    const dtype *__restrict DL,     // Path delays in seconds, Size [n_ant, n_path] or [1, n_path]
                    const size_t n_ant,             // Number of MIMO sub-links
                    const size_t n_path,            // Number of multipath components
                    const bool planar_wave,         // Indicator that same delays are used for all antennas
                    const double *__restrict freq,  // Carrier frequencies -bandwidth * pilot_grid in [Hz], Length [ n_carrier ]
                    const size_t n_carrier,         // Number of carriers
                    float *__restrict Hr,           // Channel matrix, real part, Size [ n_carrier, n_ant ]
                    float *__restrict Hi)           // Channel matrix, imaginary part, Size [ n_carrier, n_ant ]
{
    constexpr double magic = 6755399441055744.0; // 1.5 * 2^52
    constexpr double x_limit = 1125899906842624.0; // 2^50

    double max_freq = 0.0;
    for (size_t k = 0; k < n_carrier; ++k)
        max_freq = std::max(max_freq, std::abs(freq[k]));

    for (size_t i_ant = 0; i_ant < n_ant; ++i_ant)
    {
        float *__restrict hr = &Hr[i_ant * n_carrier];
        float *__restrict hi = &Hi[i_ant * n_carrier];
        for (size_t k = 0; k < n_carrier; ++k)
            hr[k] = 0.0f, hi[k] = 0.0f;

        for (size_t i_path = 0; i_path < n_path; ++i_path)
        {
            const size_t idx = i_path * n_ant + i_ant;
            const float cr = (float)CFr[idx];
            const float ci = (float)CFi[idx];
            const double dl = planar_wave ? (double)DL[i_path] : (double)DL[idx];

            if (std::abs(dl) * max_freq < x_limit) // Also false for NaN and Inf
                for (size_t k = 0; k < n_carrier; ++k)
                {
                    const double x = freq[k] * dl; // Phase in cycles
                    float s, c;
                    qd_sincos_2pi_reduced((float)(x - ((x + magic) - magic)), s, c);
                    hr[k] += c * cr - s * ci;
                    hi[k] += c * ci + s * cr;
                }
            else
                for (size_t k = 0; k < n_carrier; ++k)
                {
                    const double x = freq[k] * dl;
                    const float theta = (float)(6.283185307179586 * (x - std::rint(x))); // NaN for non-finite x
                    const float s = std::sin(theta), c = std::cos(theta);
                    hr[k] += c * cr - s * ci;
                    hi[k] += c * ci + s * cr;
                }
        }
    }
}

template void qd_DFT_GENERIC(const float *__restrict CFr, const float *__restrict CFi, const float *__restrict DL, const size_t n_ant,
                             const size_t n_path, const bool planar_wave, const double *__restrict freq, const size_t n_carrier,
                             float *__restrict Hr, float *__restrict Hi);
template void qd_DFT_GENERIC(const double *__restrict CFr, const double *__restrict CFi, const double *__restrict DL, const size_t n_ant,
                             const size_t n_path, const bool planar_wave, const double *__restrict freq, const size_t n_carrier,
                             float *__restrict Hr, float *__restrict Hi);
