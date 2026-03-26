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

// A collection of non-accelerated functions

#ifndef quadriga_lib_genericfun_H
#define quadriga_lib_genericfun_H

#include <cstddef>

// Generic C++ implementation of RayTriangleIntersect
template <typename dtype>
inline void qd_RTI_GENERIC(const dtype *Tx, const dtype *Ty, const dtype *Tz,    // First vertex coordinate in GCS, length n_mesh
                           const dtype *E1x, const dtype *E1y, const dtype *E1z, // Edge 1 from first vertex to second vertex, length n_mesh
                           const dtype *E2x, const dtype *E2y, const dtype *E2z, // Edge 2 from first vertex to third vertex, length n_mesh
                           const size_t n_mesh,                                  // Number of triangles (multiple of VEC_SIZE)
                           const unsigned *SMI,                                  // List of sub-mesh indices, length n_sub
                           const dtype *Xmin, const dtype *Xmax,                 // Minimum and maximum x-values of the AABB, length n_sub_s
                           const dtype *Ymin, const dtype *Ymax,                 // Minimum and maximum y-values of the AABB, length n_sub_s
                           const dtype *Zmin, const dtype *Zmax,                 // Minimum and maximum z-values of the AABB, length n_sub_s
                           const size_t n_sub,                                   // Number of sub-meshes (not aligned, i.e. n_sub <= n_sub_s)
                           const dtype *Ox, const dtype *Oy, const dtype *Oz,    // Ray origin in GCS, length n_ray
                           const dtype *Dx, const dtype *Dy, const dtype *Dz,    // Vector from ray origin to ray destination, length n_ray
                           const size_t n_ray,                                   // Number of rays
                           dtype *Wf,                                            // Normalized distance (0-1) of FBS hit, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                           dtype *Ws,                                            // Normalized distance (0-1) of SBS hit, must be >= Wf, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                           unsigned *If,                                         // Index of mesh element hit at FBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                           unsigned *Is,                                         // Index of mesh element hit at SBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                           unsigned *hit_cnt = nullptr)                          // Number of hits between orig and dest, length n_ray, uninitialized, optional
{
    if (n_mesh >= INT32_MAX)
        throw std::invalid_argument("Number of triangles exceeds maximum supported number.");
    if (n_ray >= INT32_MAX)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

    bool count_hits = hit_cnt != nullptr;

    const int n_mesh_int = (int)n_mesh;
    const int n_ray_int = (int)n_ray;

#pragma omp parallel for
    for (int i_ray = 0; i_ray < n_ray_int; ++i_ray) // Ray loop
    {
        // Load origin
        dtype ox = Ox[i_ray];
        dtype oy = Oy[i_ray];
        dtype oz = Oz[i_ray];

        // Load vector from ray origin to ray destination
        dtype dx = Dx[i_ray];
        dtype dy = Dy[i_ray];
        dtype dz = Dz[i_ray];

        // Initialize local variables
        dtype W_fbs = dtype(1.0);        // Set FBS location equal to dest
        dtype W_sbs = W_fbs;             // Set SBS location equal to FBS location
        unsigned I_fbs = 0U, I_sbs = 0U; // Set FBS index to 0
        unsigned hit_counter = 0U;       // Hit counter

        // Step 1 - Check intersection with the AABBs of the sub-meshes (slab-method)
        dtype dx_i = dtype(1.0) / dx;
        dtype dy_i = dtype(1.0) / dy;
        dtype dz_i = dtype(1.0) / dz;

        arma::s32_vec sub_mesh_hit(n_sub, arma::fill::none);
        int *p_sub_mesh_hit = sub_mesh_hit.memptr();

        for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            dtype t0_low = (Xmin[i_sub] - ox) * dx_i;
            dtype t0_high = (Xmax[i_sub] - ox) * dx_i;
            dtype t1_low = (Ymin[i_sub] - oy) * dy_i;
            dtype t1_high = (Ymax[i_sub] - oy) * dy_i;
            dtype t2_low = (Zmin[i_sub] - oz) * dz_i;
            dtype t2_high = (Zmax[i_sub] - oz) * dz_i;

            bool M = t0_low >= t0_high;
            dtype T = M ? t0_high : t0_low;
            t0_high = M ? t0_low : t0_high;
            t0_low = T;

            M = t1_low >= t1_high;
            T = M ? t1_high : t1_low;
            t1_high = M ? t1_low : t1_high;
            t1_low = T;

            M = t2_low >= t2_high;
            T = M ? t2_high : t2_low;
            t2_high = M ? t2_low : t2_high;
            t2_low = T;

            M = t0_low >= t1_low;
            t0_low = M ? t0_low : t1_low;
            M = t0_low >= t2_low;
            t0_low = M ? t0_low : t2_low;

            M = t0_high <= t1_high;
            t0_high = M ? t0_high : t1_high;
            M = t0_high <= t2_high;
            t0_high = M ? t0_high : t2_high;

            bool C1 = t0_high > dtype(0.0) && t0_high >= t0_low && t0_low <= dtype(1.0);
            p_sub_mesh_hit[i_sub] = int(C1);
        }

        // Step 2 - Check intersection with triangles within the sub-meshes
        for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            if (p_sub_mesh_hit[i_sub] == 0)
                continue;

            int i_mesh_start = (int)SMI[i_sub];
            int i_mesh_end = (i_sub == n_sub - 1) ? n_mesh_int : (int)SMI[i_sub + 1];

            for (int i_mesh = i_mesh_start; i_mesh < i_mesh_end; ++i_mesh) // Mesh loop
            {
                dtype tx = Tx[i_mesh];
                dtype ty = Ty[i_mesh];
                dtype tz = Tz[i_mesh];

                tx = ox - tx;
                ty = oy - ty;
                tz = oz - tz;

                dtype e1x = E1x[i_mesh], e2x = E2x[i_mesh];
                dtype e1y = E1y[i_mesh], e2y = E2y[i_mesh];
                dtype e1z = E1z[i_mesh], e2z = E2z[i_mesh];

                // Calculate 1st barycentric coordinate U
                dtype PQ = e2z * dy - e2y * dz;
                dtype DT = e1x * PQ;
                dtype U = tx * PQ;

                PQ = e2x * dz - e2z * dx;
                DT = e1y * PQ + DT;
                U = ty * PQ + U;

                PQ = e2y * dx - e2x * dy;
                DT = e1z * PQ + DT;
                U = tz * PQ + U;

                // Calculate 2nd barycentric coordinate (V) and normalized intersect position (W)
                PQ = e1z * ty - e1y * tz;
                dtype V = dx * PQ;
                dtype W = e2x * PQ;

                PQ = e1x * tz - e1z * tx;
                V = dy * PQ + V;
                W = e2y * PQ + W;

                PQ = e1y * tx - e1x * ty;
                V = dz * PQ + V;
                W = e2z * PQ + W;

                // Inverse of DT
                DT = dtype(1.0) / DT;
                U = U * DT;
                V = V * DT;
                W = W * DT;

                // Check intersect conditions
                bool C1 = (U >= dtype(0.0)) & (V >= dtype(0.0)) & ((U + V) <= dtype(1.0)) & (W >= dtype(0.0)) & (W < dtype(1.0));

                if (!C1)
                    continue;

                if (count_hits)
                    ++hit_counter;

                if (W < W_fbs)
                {
                    W_sbs = W_fbs;
                    I_sbs = I_fbs;
                    W_fbs = W;
                    I_fbs = i_mesh;
                }
                else if (W < W_sbs)
                {
                    W_sbs = W;
                    I_sbs = i_mesh;
                }
            }
        }

        // Update output memory
        Wf[i_ray] = W_fbs;
        Ws[i_ray] = W_sbs;
        If[i_ray] = (Wf[i_ray] < dtype(1.0)) ? I_fbs + 1 : 0;
        Is[i_ray] = (Ws[i_ray] < dtype(1.0)) ? I_sbs + 1 : 0;
        if (count_hits)
            hit_cnt[i_ray] = hit_counter;
    }
}

// Generic C++ implementation of RayPointIntersect
template <typename dtype>
static inline void qd_RPI_GENERIC(const dtype *Px, const dtype *Py, const dtype *Pz,    // Point coordinates, length n_point
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
                                  std::vector<unsigned> *p_hit)                         // Output: Array of std::vector containing list of points that were hit by a ray, length n_ray
{
    if (n_point >= INT32_MAX)
        throw std::invalid_argument("Number of points exceeds maximum supported number.");
    if (n_ray >= INT32_MAX)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

    // Constant values needed for some operations
    const int n_point_i = (int)n_point; // Number of points as int
    const int n_ray_i = (int)n_ray;     // Number of rays as int

    constexpr dtype slack = dtype(1.0e-5);

#pragma omp parallel for
    for (int i_ray = 0; i_ray < n_ray_i; ++i_ray) // Ray loop
    {
        // Initialize indicator for sub-cloud hits
        arma::s32_vec sub_hit(n_sub, arma::fill::zeros);
        int *p_sub_hit = sub_hit.memptr();

        // Load origin
        dtype ox[3], oy[3], oz[3];
        ox[0] = T1x[i_ray], oy[0] = T1y[i_ray], oz[0] = T1z[i_ray];
        ox[1] = T2x[i_ray], oy[1] = T2y[i_ray], oz[1] = T2z[i_ray];
        ox[2] = T3x[i_ray], oy[2] = T3y[i_ray], oz[2] = T3z[i_ray];

        // Load direction
        dtype dx[3], dy[3], dz[3];
        dx[0] = D1x[i_ray], dy[0] = D1y[i_ray], dz[0] = D1z[i_ray];
        dx[1] = D2x[i_ray], dy[1] = D2y[i_ray], dz[1] = D2z[i_ray];
        dx[2] = D3x[i_ray], dy[2] = D3y[i_ray], dz[2] = D3z[i_ray];

        // Load normal vector
        dtype nx = Nx[i_ray], ny = Ny[i_ray], nz = Nz[i_ray];

        // Load inverse dot product
        dtype rdx = rD1[i_ray], rdy = rD2[i_ray], rdz = rD3[i_ray];

        // Step 1 - Check for possible hits
        // - Move the wavefront forward relative to the distance between vertex origin and AABB corner point
        // - Construct second AABB from advanced wavefronts
        // - If AABBs overlap, there is a potential hit and individual points must be checked in step 2

        for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            // Load point bounding box, add some slack for numeric stability
            dtype b0_low = Xmin[i_sub] - slack, b0_high = Xmax[i_sub] + slack,
                  b1_low = Ymin[i_sub] - slack, b1_high = Ymax[i_sub] + slack,
                  b2_low = Zmin[i_sub] - slack, b2_high = Zmax[i_sub] + slack;

            // AABB corner points
            dtype rx[8] = {b0_low, b0_low, b0_low, b0_low, b0_high, b0_high, b0_high, b0_high};
            dtype ry[8] = {b1_low, b1_low, b1_high, b1_high, b1_low, b1_low, b1_high, b1_high};
            dtype rz[8] = {b2_low, b2_high, b2_low, b2_high, b2_low, b2_high, b2_low, b2_high};

            // Initialize coordinates for the vertex box
            constexpr dtype inf = std::numeric_limits<dtype>::infinity();
            dtype a0_low = inf, a0_high = -inf,
                  a1_low = inf, a1_high = -inf,
                  a2_low = inf, a2_high = -inf;

            // Calculate the vertex box at the advanced wavefront
            for (int i = 0; i < 8; ++i)
            {
                // Distance between vertex origin and wavefront at corner point
                dtype d = rdx * ((rx[i] - ox[0]) * nx + (ry[i] - oy[0]) * ny + (rz[i] - oz[0]) * nz);

                // Update vertex box at advanced wavefront
                dtype V = d * dx[0] + ox[0];
                a0_low = (V < a0_low) ? V : a0_low;
                a0_high = (V > a0_high) ? V : a0_high;

                V = d * dy[0] + oy[0];
                a1_low = (V < a1_low) ? V : a1_low;
                a1_high = (V > a1_high) ? V : a1_high;

                V = d * dz[0] + oz[0];
                a2_low = (V < a2_low) ? V : a2_low;
                a2_high = (V > a2_high) ? V : a2_high;

                // 2nd vertex
                d = rdy * ((rx[i] - ox[1]) * nx + (ry[i] - oy[1]) * ny + (rz[i] - oz[1]) * nz);

                V = d * dx[1] + ox[1];
                a0_low = (V < a0_low) ? V : a0_low;
                a0_high = (V > a0_high) ? V : a0_high;

                V = d * dy[1] + oy[1];
                a1_low = (V < a1_low) ? V : a1_low;
                a1_high = (V > a1_high) ? V : a1_high;

                V = d * dz[1] + oz[1];
                a2_low = (V < a2_low) ? V : a2_low;
                a2_high = (V > a2_high) ? V : a2_high;

                // 3rd vertex
                d = rdz * ((rx[i] - ox[2]) * nx + (ry[i] - oy[2]) * ny + (rz[i] - oz[2]) * nz);

                V = d * dx[2] + ox[2];
                a0_low = (V < a0_low) ? V : a0_low;
                a0_high = (V > a0_high) ? V : a0_high;

                V = d * dy[2] + oy[2];
                a1_low = (V < a1_low) ? V : a1_low;
                a1_high = (V > a1_high) ? V : a1_high;

                V = d * dz[2] + oz[2];
                a2_low = (V < a2_low) ? V : a2_low;
                a2_high = (V > a2_high) ? V : a2_high;
            }

            // Check for a potential overlap between the AABBs
            if (a0_high >= b0_low && a0_low <= b0_high &&
                a1_high >= b1_low && a1_low <= b1_high &&
                a2_high >= b2_low && a2_low <= b2_high)
                p_sub_hit[i_sub] = 1;
        }

        // Step 2 - Check intersection with points within the sub-clouds

        for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            // Skip if sub-cloud was not hit
            if (p_sub_hit[i_sub] == 0)
                continue;

            int i_point_start = (int)SCI[i_sub];
            int i_point_end = (i_sub == n_sub - 1) ? n_point_i : (int)SCI[i_sub + 1];

            for (int i_point = i_point_start; i_point < i_point_end; ++i_point) // Point loop
            {
                // Load point coordinate
                dtype rx = Px[i_point];
                dtype ry = Py[i_point];
                dtype rz = Pz[i_point];

                // Distance between vertex origin and wavefront at point
                dtype d = rdx * ((rx - ox[0]) * nx + (ry - oy[0]) * ny + (rz - oz[0]) * nz);

                // Vertex position at advanced wavefront
                dtype Vx = d * dx[0] + ox[0],
                      Vy = d * dy[0] + oy[0],
                      Vz = d * dz[0] + oz[0];

                // Calculate edge from W1 to W2
                d = rdy * ((rx - ox[1]) * nx + (ry - oy[1]) * ny + (rz - oz[1]) * nz);

                dtype e1x = d * dx[1] + ox[1] - Vx,
                      e1y = d * dy[1] + oy[1] - Vy,
                      e1z = d * dz[1] + oz[1] - Vz;

                // Calculate edge from W1 to W3
                d = rdz * ((rx - ox[2]) * nx + (ry - oy[2]) * ny + (rz - oz[2]) * nz);

                dtype e2x = d * dx[2] + ox[2] - Vx,
                      e2y = d * dy[2] + oy[2] - Vy,
                      e2z = d * dz[2] + oz[2] - Vz;

                // Calculate vector from V to R
                dtype tx = rx - Vx,
                      ty = ry - Vy,
                      tz = rz - Vz;

                // Calculate 1st barycentric coordinate
                dtype PQ = e2z * ny - e2y * nz;
                dtype DT = e1x * PQ;
                dtype U = tx * PQ;

                PQ = e2x * nz - e2z * nx;
                DT = e1y * PQ + DT;
                U = ty * PQ + U;

                PQ = e2y * nx - e2x * ny;
                DT = e1z * PQ + DT;
                U = tz * PQ + U;

                // Calculate 2nd barycentric coordinate
                PQ = e1z * ty - e1y * tz;
                dtype V = nx * PQ;

                PQ = e1x * tz - e1z * tx,
                V = ny * PQ + V;

                PQ = e1y * tx - e1x * ty,
                V = nz * PQ + V;

                // Inverse of DT
                DT = dtype(1.0) / DT;

                U = U * DT;
                V = V * DT;

                // Check intersect conditions
                bool C1 = (U >= dtype(0.0)) & (V >= dtype(0.0)) & ((U + V) <= dtype(1.0)) & (d >= dtype(0.0));

                // Add point to points list
                if (C1)
                    p_hit[i_ray].push_back(i_point);
            }
        }
    }
}

// Generic C++ implementation of DFT
template <typename dtype>
static inline void qd_DFT_GENERIC(const dtype *__restrict CFr,    // Channel coefficients, real part, Size [n_ant, n_path]
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
