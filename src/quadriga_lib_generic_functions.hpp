// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// A collection of non-accelerated functions

#ifndef quadriga_lib_genericfun_H
#define quadriga_lib_genericfun_H

#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <cstdint>

// Assemble a point-major CSR hit list from a set of packed hit buffers
// Each hit is a 64 bit record, the upper 32 bit hold the 0-based point index
// and the lower 32 bit the 0-based ray index. The buffers are concatenated in
// the given order, which also fixes the order of the ray indices within each
// point's block. Used by the GENERIC, AVX2 and CUDA implementations of
// RayPointIntersect, defined in ray_point_intersect_generic.cpp
void qd_RPI_assemble(const unsigned long long *const *hit_buffer, // Packed hit records, one pointer per buffer, length n_buffer
                     const size_t *n_hit_buffer,                  // Number of records in each buffer, length n_buffer
                     const size_t n_buffer,                       // Number of buffers, may be 0
                     const size_t n_point,                        // Number of points
                     std::vector<unsigned> *hit_index,            // Output: flat list of ray indices grouped by point, resized to the total hit count
                     unsigned *hit_offset);                       // Output: 0-based start of each point's block, length n_point + 1

// Generic C++ implementation of RayTriangleIntersect
template <typename dtype>
void qd_RTI_GENERIC(const dtype *Tx, const dtype *Ty, const dtype *Tz, // First vertex coordinate in GCS, length n_mesh
                    const dtype *Ux, const dtype *Uy, const dtype *Uz, // Second vertex coordinate in GCS, length n_mesh
                    const dtype *Vx, const dtype *Vy, const dtype *Vz, // Third vertex coordinate in GCS, length n_mesh
                    const size_t n_mesh,                               // Number of triangles (multiple of VEC_SIZE)
                    const unsigned *SMI,                               // List of sub-mesh indices, length n_sub
                    const dtype *Xmin, const dtype *Xmax,              // Minimum and maximum x-values of the AABB, length n_sub_s
                    const dtype *Ymin, const dtype *Ymax,              // Minimum and maximum y-values of the AABB, length n_sub_s
                    const dtype *Zmin, const dtype *Zmax,              // Minimum and maximum z-values of the AABB, length n_sub_s
                    const size_t n_sub,                                // Number of sub-meshes (not aligned, i.e. n_sub <= n_sub_s)
                    const dtype *Ox, const dtype *Oy, const dtype *Oz, // Ray origin in GCS, length n_ray
                    const dtype *Dx, const dtype *Dy, const dtype *Dz, // Ray destination in GCS, length n_ray
                    const size_t n_ray,                                // Number of rays
                    dtype *Wf,                                         // Normalized distance (0-1) of FBS hit, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                    dtype *Ws,                                         // Normalized distance (0-1) of SBS hit, must be >= Wf, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                    unsigned *If,                                      // Index of mesh element hit at FBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                    unsigned *Is,                                      // Index of mesh element hit at SBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                    unsigned *hit_cnt = nullptr);                      // Number of hits between orig and dest, length n_ray, uninitialized, optional

// Generic C++ implementation of RayPointIntersect
template <typename dtype>
void qd_RPI_GENERIC(const dtype *Px, const dtype *Py, const dtype *Pz,    // Point coordinates, length n_point
                    const size_t n_point,                                 // Number of points
                    const unsigned *SCI,                                  // List of sub-cloud indices, length n_sub
                    const dtype *Xmin, const dtype *Xmax,                 // Minimum and maximum x-values of the AABB, length n_sub_s
                    const dtype *Ymin, const dtype *Ymax,                 // Minimum and maximum y-values of the AABB, length n_sub_s
                    const dtype *Zmin, const dtype *Zmax,                 // Minimum and maximum z-values of the AABB, length n_sub_s
                    const size_t n_sub,                                   // Number of sub-clouds (n_sub <= n_sub_s)
                    const dtype *T1x, const dtype *T1y, const dtype *T1z, // First ray vertex coordinate in GCS, length n_ray
                    const dtype *T2x, const dtype *T2y, const dtype *T2z, // Second ray vertex coordinate in GCS, length n_ray
                    const dtype *T3x, const dtype *T3y, const dtype *T3z, // Third ray vertex coordinate in GCS, length n_ray
                    const dtype *Nx, const dtype *Ny, const dtype *Nz,    // Ray tube normal vector, length n_ray
                    const dtype *D1x, const dtype *D1y, const dtype *D1z, // First ray direction in GCS, length n_ray
                    const dtype *D2x, const dtype *D2y, const dtype *D2z, // Second ray direction in GCS, length n_ray
                    const dtype *D3x, const dtype *D3y, const dtype *D3z, // Third ray direction in GCS, length n_ray
                    const dtype *rD1, const dtype *rD2, const dtype *rD3, // Inverse Dot product of ray direction and normal vector
                    const size_t n_ray,                                   // Number of rays
                    std::vector<unsigned> *hit_index,                     // Output: flat list of 0-based ray indices grouped by point, resized by the kernel
                    unsigned *hit_offset);                                // Output: 0-based start of each point's block, length n_point + 1, allocated by the caller

// sin and cos of 2 * pi * r for r in [-0.5, 0.5] (one period, already reduced in double precision by the caller).
// Same algorithm as the AVX2 DFT kernel (_fm256_sincos_2pi_reduced_ps and the shared polynomial part
// _fm256_sincos_poly_ps in fastmath_avx2.h): quadrant q = nearest integer of 4 * r (magic-number rounding, exact for
// |4 * r| < 2^22, and only without -ffast-math, which may fold (u + magic) - magic to u), t = (4 * r - q) * pi / 2 in
// [-pi/4, pi/4], Cephes single-precision polynomials, quadrant swap / signs.
// Branch-free, so that loops calling it can be vectorized.
inline void qd_sincos_2pi_reduced(float r, float &s, float &c)
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

// Generic C++ implementation of DFT
// The phase x = freq * delay (in cycles) is computed in double precision and reduced to x - round(x), which is in
// [-0.5, 0.5], before sin / cos are evaluated in single precision; coefficients and sums are single precision.
// Paths are processed one after the other, the carrier loop is innermost and branch-free (vectorizable): the rounding
// uses the magic-number method, exact for |x| < 2^51 and only without -ffast-math (which may fold it away); paths with
// larger or non-finite phases use a scalar loop.
template <typename dtype>
void qd_DFT_GENERIC(const dtype *__restrict CFr,    // Channel coefficients, real part, Size [n_ant, n_path]
                    const dtype *__restrict CFi,    // Channel coefficients, imaginary part, Size [n_ant, n_path]
                    const dtype *__restrict DL,     // Path delays in seconds, Size [n_ant, n_path] or [1, n_path]
                    const size_t n_ant,             // Number of MIMO sub-links
                    const size_t n_path,            // Number multipath components
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

// Sum-of-sinusoids evaluation, double precision
// val(p, g) = sum_n wc(n, g) · cos( 2·pi · t ) - ws(n, g) · sin( 2·pi · t ), t = fx(n)·px(p) + fy(n)·py(p) + fz(n)·pz(p)
// The argument t is reduced to one period before the trigonometric functions are evaluated
void qd_SOS_GENERIC(const double *fx, const double *fy, const double *fz, // Frequencies in [1/m], length n_coeff
                    const double *wc, const double *ws,                   // Weights a·cos(phase), a·sin(phase), Size [n_coeff, n_gen]
                    const size_t n_coeff,                                 // Number of sinusoids
                    const size_t n_gen,                                   // Number of generators
                    const double *px, const double *py, const double *pz, // Positions in [m], length n_pos
                    const size_t n_pos,                                   // Number of positions
                    double *val);                                         // Output: Values, Size [n_pos, n_gen], uninitialized

#endif
