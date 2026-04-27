// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// A collection of non-accelerated functions

#ifndef quadriga_lib_genericfun_H
#define quadriga_lib_genericfun_H

#include <cstddef>
#include <vector>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <cstdint>

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
                    std::vector<unsigned> *p_hit);                        // Output: Array of std::vector containing list of points that were hit by a ray, length n_ray

// Generic C++ implementation of DFT
template <typename dtype>
void qd_DFT_GENERIC(const dtype *__restrict CFr,    // Channel coefficients, real part, Size [n_ant, n_path]
                    const dtype *__restrict CFi,    // Channel coefficients, imaginary part, Size [n_ant, n_path]
                    const dtype *__restrict DL,     // Path delays in seconds, Size [n_ant, n_path] or [1, n_path]
                    const size_t n_ant,             // Number of MIMO sub-links
                    const size_t n_path,            // Number multipath components
                    const bool planar_wave,         // Indicator that same delays are used for all antennas
                    const float *__restrict phasor, // Phasor, Length [ n_carrier ]
                    const size_t n_carrier,         // Number of carriers
                    float *__restrict Hr,           // Channel matrix, real part, Size [ n_carrier, n_ant ]
                    float *__restrict Hi)           // Channel matrix, imaginary part, Size [ n_carrier, n_ant ]
{
    std::vector<float> crf(n_path), cif(n_path), dlf(n_path);

    for (size_t i_ant = 0; i_ant < n_ant; ++i_ant)
    {
        for (size_t i_path = 0; i_path < n_path; ++i_path)
        {
            const size_t idx = i_path * n_ant + i_ant;
            crf[i_path] = (float)CFr[idx];
            cif[i_path] = (float)CFi[idx];
            dlf[i_path] = planar_wave ? (float)DL[i_path] : (float)DL[idx];
        }

        const size_t base = i_ant * n_carrier;

        for (size_t k = 0; k < n_carrier; ++k)
        {
            float hr = 0.0f;
            float hi = 0.0f;
            const float pk = phasor[k];

            for (size_t i_path = 0; i_path < n_path; ++i_path)
            {
                const float theta = pk * dlf[i_path];
                const float s = std::sin(theta);
                const float c = std::cos(theta);

                hr += c * crf[i_path] - s * cif[i_path];
                hi += c * cif[i_path] + s * crf[i_path];
            }

            Hr[base + k] = hr;
            Hi[base + k] = hi;
        }
    }
}

#endif
