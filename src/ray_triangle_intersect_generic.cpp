// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib_generic_functions.hpp"

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
                    unsigned *hit_cnt)                                 // Number of hits between orig and dest, length n_ray, uninitialized, optional
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
        dtype dx = Dx[i_ray] - ox;
        dtype dy = Dy[i_ray] - oy;
        dtype dz = Dz[i_ray] - oz;

        // Initialize local variables
        dtype W_fbs = dtype(1.0);        // Set FBS location equal to dest
        dtype W_sbs = W_fbs;             // Set SBS location equal to FBS location
        unsigned I_fbs = 0U, I_sbs = 0U; // Set FBS index to 0
        unsigned hit_counter = 0U;       // Hit counter

        // Step 1 - Check intersection with the AABBs of the sub-meshes (slab-method)
        dtype dx_i = dtype(1.0) / dx;
        dtype dy_i = dtype(1.0) / dy;
        dtype dz_i = dtype(1.0) / dz;

        std::vector<int> sub_mesh_hit(n_sub);
        int *p_sub_mesh_hit = sub_mesh_hit.data();

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

                dtype e1x = Ux[i_mesh] - tx, e2x = Vx[i_mesh] - tx;
                dtype e1y = Uy[i_mesh] - ty, e2y = Vy[i_mesh] - ty;
                dtype e1z = Uz[i_mesh] - tz, e2z = Vz[i_mesh] - tz;

                tx = ox - tx;
                ty = oy - ty;
                tz = oz - tz;

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

template void qd_RTI_GENERIC<float>(const float *Tx, const float *Ty, const float *Tz,
                                    const float *Ux, const float *Uy, const float *Uz, const float *Vx, const float *Vy, const float *Vz,
                                    size_t n_mesh, const unsigned *SMI, const float *Xmin, const float *Xmax, const float *Ymin, const float *Ymax,
                                    const float *Zmin, const float *Zmax, size_t n_sub, const float *Ox, const float *Oy, const float *Oz,
                                    const float *Dx, const float *Dy, const float *Dz, size_t n_ray, float *Wf, float *Ws, unsigned *If,
                                    unsigned *Is, unsigned *hit_cnt);

template void qd_RTI_GENERIC<double>(const double *Tx, const double *Ty, const double *Tz,
                                     const double *Ux, const double *Uy, const double *Uz, const double *Vx, const double *Vy, const double *Vz,
                                     size_t n_mesh, const unsigned *SMI, const double *Xmin, const double *Xmax, const double *Ymin, const double *Ymax,
                                     const double *Zmin, const double *Zmax, size_t n_sub, const double *Ox, const double *Oy, const double *Oz,
                                     const double *Dx, const double *Dy, const double *Dz, size_t n_ray, double *Wf, double *Ws, unsigned *If,
                                     unsigned *Is, unsigned *hit_cnt);
