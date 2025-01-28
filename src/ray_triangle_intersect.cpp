// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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

#include <cstring> // For std::memcopy

#include <cmath> // For std::isnan
#include "quadriga_tools.hpp"
#include "ray_triangle_intersect_avx2.hpp"

// Vector size for AVX2
#define VEC_SIZE 8ULL

// Testing for AVX2 support at runtime
#if defined(_MSC_VER) // Windows
#include <intrin.h>
#include <malloc.h> // Include for _aligned_malloc and _aligned_free
#else               // Linux
#include <cpuid.h>
#endif

static bool isAVX2Supported()
{
    std::vector<int> cpuidInfo(4);

#if defined(_MSC_VER) // Windows
    __cpuidex(cpuidInfo.data(), 7, 0);
#else // Linux
    __cpuid_count(7, 0, cpuidInfo[0], cpuidInfo[1], cpuidInfo[2], cpuidInfo[3]);
#endif

    return (cpuidInfo[1] & (1 << 5)) != 0; // Check the AVX2 bit in EBX
}

// Generic C++ implementation of RayTriangleIntersect
static inline void qd_RTI_GENERIC(const float *Tx, const float *Ty, const float *Tz,    // First vertex coordinate in GCS, length n_mesh
                                  const float *E1x, const float *E1y, const float *E1z, // Edge 1 from first vertex to second vertex, length n_mesh
                                  const float *E2x, const float *E2y, const float *E2z, // Edge 2 from first vertex to third vertex, length n_mesh
                                  const size_t n_mesh,                                  // Number of triangles (multiple of VEC_SIZE)
                                  const unsigned *SMI,                                  // List of sub-mesh indices, length n_sub
                                  const float *Xmin, const float *Xmax,                 // Minimum and maximum x-values of the AABB, aligned to 32 byte, length n_sub_s
                                  const float *Ymin, const float *Ymax,                 // Minimum and maximum y-values of the AABB, aligned to 32 byte, length n_sub_s
                                  const float *Zmin, const float *Zmax,                 // Minimum and maximum z-values of the AABB, aligned to 32 byte, length n_sub_s
                                  const size_t n_sub,                                   // Number of sub-meshes (not aligned, i.e. n_sub <= n_sub_s)
                                  const float *Ox, const float *Oy, const float *Oz,    // Ray origin in GCS, length n_ray
                                  const float *Dx, const float *Dy, const float *Dz,    // Vector from ray origin to ray destination, length n_ray
                                  const size_t n_ray,                                   // Number of rays
                                  float *Wf,                                            // Normalized distance (0-1) of FBS hit, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                                  float *Ws,                                            // Normalized distance (0-1) of SBS hit, must be >= Wf, 0 = orig, 1 = dest (no hit), length n_ray, uninitialized
                                  unsigned *If,                                         // Index of mesh element hit at FBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                                  unsigned *Is,                                         // Index of mesh element hit at SBS location, 1-based, 0 = no hit, length n_ray, uninitialized
                                  unsigned *hit_cnt = nullptr)                          // Number of hits between orig and dest, length n_ray, uninitialized, optional
{
    if (n_mesh >= INT32_MAX)
        throw std::invalid_argument("Number of triangles exceeds maximum supported number.");
    if (n_ray >= INT32_MAX)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

    bool count_hits = hit_cnt != nullptr;

    // Constant values needed for some operations
    const int n_mesh_int = (int)n_mesh; // Number of triangles as int
    const int n_ray_int = (int)n_ray;   // Number of rays as int

#pragma omp parallel for
    for (int i_ray = 0; i_ray < n_ray_int; ++i_ray) // Ray loop
    {
        // Load origin
        float ox = Ox[i_ray];
        float oy = Oy[i_ray];
        float oz = Oz[i_ray];

        // Load vector from ray origin to ray destination
        float dx = Dx[i_ray];
        float dy = Dy[i_ray];
        float dz = Dz[i_ray];

        // Initialize local variables
        float W_fbs = 1.0f;              // Set FBS location equal to dest
        float W_sbs = W_fbs;             // Set SBS location equal to FBS location
        unsigned I_fbs = 0U, I_sbs = 0U; // Set FBS index to 0
        unsigned hit_counter = 0U;       // Hit counter

        // Step 1 - Check intersection with the AABBs of the sub-meshes (slab-method)
        // See: https://en.wikipedia.org/wiki/Slab_method

        // Inverse of the direction (may be infinite if ray is parallel to an axis)
        float dx_i = 1.0f / dx;
        float dy_i = 1.0f / dy;
        float dz_i = 1.0f / dz;

        arma::s32_vec sub_mesh_hit(n_sub, arma::fill::none);
        int *p_sub_mesh_hit = sub_mesh_hit.memptr();
        for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            // Calculate the intersections of the ray with the two planes orthogonal to the i-th coordinate axis
            float t0_low = (Xmin[i_sub] - ox) * dx_i;
            float t0_high = (Xmax[i_sub] - ox) * dx_i;
            float t1_low = (Ymin[i_sub] - oy) * dy_i;
            float t1_high = (Ymax[i_sub] - oy) * dy_i;
            float t2_low = (Zmin[i_sub] - oz) * dz_i;
            float t2_high = (Zmax[i_sub] - oz) * dz_i;

            // Calculate the close and far extrema of the segment within the i-th slab
            bool M = t0_low >= t0_high;
            float T = M ? t0_high : t0_low;
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

            // Calculate the intersection of all segments
            M = t0_low >= t1_low;
            t0_low = M ? t0_low : t1_low;
            M = t0_low >= t2_low;
            t0_low = M ? t0_low : t2_low;

            M = t0_high <= t1_high;
            t0_high = M ? t0_high : t1_high;
            M = t0_high <= t2_high;
            t0_high = M ? t0_high : t2_high;

            // If t0_high < 0, the ray is intersecting AABB, but the whole AABB is behind us
            // If t0_low > t0_high, ray doesn't intersect AABB
            // If t0_low > 1, the destination point lays before the AABB
            bool C1 = t0_high > 0.0f && t0_high >= t0_low && t0_low <= 1.0f;
            p_sub_mesh_hit[i_sub] = int(C1);
        }

        // Step 2 - Check intersection with triangles within the sub-meshes

        for (size_t i_sub = 0; i_sub < n_sub; ++i_sub)
        {
            // Skip if sub-mesh was not hit
            if (p_sub_mesh_hit[i_sub] == 0)
                continue;

            int i_mesh_start = (int)SMI[i_sub];
            int i_mesh_end = (i_sub == n_sub - 1) ? n_mesh_int : (int)SMI[i_sub + 1];

            for (int i_mesh = i_mesh_start; i_mesh < i_mesh_end; ++i_mesh) // Mesh loop
            {
                // Load first vertex coordinate
                float tx = Tx[i_mesh];
                float ty = Ty[i_mesh];
                float tz = Tz[i_mesh];

                // Calculate vector from first vertex coordinate V1 to origin O
                tx = ox - tx;
                ty = oy - ty;
                tz = oz - tz;

                // Load the two triangle edges E1 and E2
                float e1x = E1x[i_mesh], e2x = E2x[i_mesh];
                float e1y = E1y[i_mesh], e2y = E2y[i_mesh];
                float e1z = E1z[i_mesh], e2z = E2z[i_mesh];

                // Calculate 1st barycentric coordinate U
                float PQ = e2z * dy - e2y * dz;
                float DT = e1x * PQ;
                float U = tx * PQ;

                PQ = e2x * dz - e2z * dx;
                DT = e1y * PQ + DT;
                U = ty * PQ + U;

                PQ = e2y * dx - e2x * dy;
                DT = e1z * PQ + DT;
                U = tz * PQ + U;

                // Calculate and 2nd barycentric coordinate (V) and normalized intersect position (W)
                PQ = e1z * ty - e1y * tz;
                float V = dx * PQ;
                float W = e2x * PQ;

                PQ = e1x * tz - e1z * tx;
                V = dy * PQ + V;
                W = e2y * PQ + W;

                PQ = e1y * tx - e1x * ty;
                V = dz * PQ + V;
                W = e2z * PQ + W;

                // Inverse of DT
                DT = 1.0f / DT;

                U = U * DT;
                V = V * DT;
                W = W * DT;

                // Check intersect conditions
                bool C1 = (U >= 0.0f) & (V >= 0.0f) & ((U + V) <= 1.0f) & (W >= 0.0f) & (W < 1.0f);

                // Fast exit if no hit was detected
                if (!C1)
                    continue;

                // Count hits
                if (count_hits)
                    ++hit_counter;

                // Update FBS and SBS position
                if (W < W_fbs)
                {                   // Update FBS and SBS
                    W_sbs = W_fbs;  // The previous FBS becomes the new SBS
                    I_sbs = I_fbs;  // Update SBS index
                    W_fbs = W;      // Store the new FBS position
                    I_fbs = i_mesh; // Set new FBS index (0-based)
                }
                else if (W < W_sbs)
                {                   // Update only SBS
                    W_sbs = W;      // Store the new SBS position
                    I_sbs = i_mesh; // Set new SBS index (0-based)
                }
            }
        }

        // Update output memory
        Wf[i_ray] = W_fbs;                              // Extract first value from W_fbs
        Ws[i_ray] = W_sbs;                              // First value from W_sbs
        If[i_ray] = (Wf[i_ray] < 1.0f) ? I_fbs + 1 : 0; // Set FBS index (1-based)
        Is[i_ray] = (Ws[i_ray] < 1.0f) ? I_sbs + 1 : 0; // Set SBS index (1-based)

        if (count_hits)
            hit_cnt[i_ray] = hit_counter;
    }
}

template <typename dtype>
void quadriga_lib::ray_triangle_intersect(const arma::Mat<dtype> *orig, const arma::Mat<dtype> *dest, const arma::Mat<dtype> *mesh,
                                          arma::Mat<dtype> *fbs, arma::Mat<dtype> *sbs, arma::Col<unsigned> *no_interact,
                                          arma::Col<unsigned> *fbs_ind, arma::Col<unsigned> *sbs_ind,
                                          const arma::Col<unsigned> *sub_mesh_index, bool transpose_inputs)
{
    // Input validation
    if (orig == nullptr)
        throw std::invalid_argument("Input 'orig' cannot be NULL.");
    if (orig->n_elem == 0)
        throw std::invalid_argument("Inputs cannot be empty.");
    if (!transpose_inputs && orig->n_cols < 3)
        throw std::invalid_argument("Input 'orig' must have at least 3 columns containing x,y,z coordinates.");
    if (transpose_inputs && orig->n_rows < 3)
        throw std::invalid_argument("Input 'orig' must have at least 3 rows containing x,y,z coordinates.");

    arma::uword n_ray = transpose_inputs ? orig->n_cols : orig->n_rows;
    arma::uword o_ray = transpose_inputs ? orig->n_rows : orig->n_cols;

    if (dest == nullptr && !transpose_inputs && orig->n_cols < 6)
        throw std::invalid_argument("If input 'dest' is NULL, 'orig' must have at least 6 columns containing x,y,z coordinates.");
    if (dest == nullptr && transpose_inputs && orig->n_rows < 6)
        throw std::invalid_argument("If input 'dest' is NULL, 'orig' must have at least 6 rows containing x,y,z coordinates.");
    if (dest != nullptr && !transpose_inputs && dest->n_cols < 3)
        throw std::invalid_argument("Input 'dest' must have at least 3 columns containing x,y,z coordinates.");
    if (dest != nullptr && transpose_inputs && dest->n_rows < 3)
        throw std::invalid_argument("Input 'dest' must have at least 3 rows containing x,y,z coordinates.");

    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (mesh->n_elem == 0)
        throw std::invalid_argument("Inputs cannot be empty.");
    if (!transpose_inputs && mesh->n_cols < 9)
        throw std::invalid_argument("Input 'mesh' must have at least 9 columns containing x,y,z coordinates of 3 vertices.");
    if (transpose_inputs && mesh->n_rows < 9)
        throw std::invalid_argument("Input 'mesh' must have at least 9 rows containing x,y,z coordinates of 3 vertices.");

    arma::uword n_mesh = transpose_inputs ? mesh->n_cols : mesh->n_rows;

    // The mesh can have more than 9 values (for additional data in memory, such as normal vectors)
    // We assume the first 9 values to be the vertex coordinates, the rest is ignored
    arma::uword o_mesh = transpose_inputs ? mesh->n_rows : mesh->n_cols;

    if (dest != nullptr && orig->n_elem != dest->n_elem)
        throw std::invalid_argument("Number of elements in 'orig' and 'dest' dont match.");

    // Convert orig and dest to aligned floats and calculate (dest - orig)
    arma::fmat origA = arma::fmat(n_ray, 3ULL, arma::fill::none);
    arma::fmat dest_minus_origA = arma::fmat(n_ray, 3ULL, arma::fill::none);
    {
        const dtype *p_orig = orig->memptr();
        float *p_origA = origA.memptr(), *p_dest_minus_origA = dest_minus_origA.memptr();

        if (dest == nullptr) // dest is included in orig
        {
            if (transpose_inputs)
            {
                for (arma::uword i_ray = 0ULL; i_ray < n_ray; ++i_ray)
                {
                    arma::uword offset = i_ray * o_ray; // Ray offset

                    dtype v_orig = p_orig[offset];                                     // Load orig_x
                    p_origA[i_ray] = (float)v_orig;                                    // Cast to float
                    p_dest_minus_origA[i_ray] = float(p_orig[offset + 3ULL] - v_orig); // Calculate dest_x - orig_x

                    v_orig = p_orig[offset + 1ULL];                                            // Load orig_y
                    p_origA[i_ray + n_ray] = (float)v_orig;                                    // Cast to float
                    p_dest_minus_origA[i_ray + n_ray] = float(p_orig[offset + 4ULL] - v_orig); // Calculate dest_y - orig_y

                    v_orig = p_orig[offset + 2ULL];                                                   // Load orig_z
                    p_origA[i_ray + 2ULL * n_ray] = (float)v_orig;                                    // Cast to float
                    p_dest_minus_origA[i_ray + 2ULL * n_ray] = float(p_orig[offset + 5ULL] - v_orig); // Calculate dest_z - orig_z
                }
            }
            else
            {
                arma::uword offset = 3ULL * n_ray;
                for (arma::uword i_elem = 0ULL; i_elem < offset; ++i_elem)
                {
                    dtype v_orig = p_orig[i_elem];                                        // Load orig
                    p_origA[i_elem] = (float)v_orig;                                      // Cast to float
                    p_dest_minus_origA[i_elem] = float(p_orig[i_elem + offset] - v_orig); // Calculate dest - orig
                }
            }
        }
        else // Separate dest
        {
            const dtype *p_dest = dest->memptr();
            if (transpose_inputs)
            {
                for (arma::uword i_ray = 0ULL; i_ray < n_ray; ++i_ray)
                {
                    arma::uword offset = i_ray * o_ray; // Ray offset

                    dtype v_orig = p_orig[offset];                              // Load orig_x
                    p_origA[i_ray] = (float)v_orig;                             // Cast to float
                    p_dest_minus_origA[i_ray] = float(p_dest[offset] - v_orig); // Calculate dest_x - orig_x

                    v_orig = p_orig[offset + 1ULL];                                            // Load orig_y
                    p_origA[i_ray + n_ray] = (float)v_orig;                                    // Cast to float
                    p_dest_minus_origA[i_ray + n_ray] = float(p_dest[offset + 1ULL] - v_orig); // Calculate dest_y - orig_y

                    v_orig = p_orig[offset + 2ULL];                                                   // Load orig_z
                    p_origA[i_ray + 2ULL * n_ray] = (float)v_orig;                                    // Cast to float
                    p_dest_minus_origA[i_ray + 2ULL * n_ray] = float(p_dest[offset + 2ULL] - v_orig); // Calculate dest_z - orig_z
                }
            }
            else
            {
                for (arma::uword i_elem = 0ULL; i_elem < 3ULL * n_ray; ++i_elem)
                {
                    dtype v_orig = p_orig[i_elem];                               // Load orig
                    p_origA[i_elem] = (float)v_orig;                             // Cast to float
                    p_dest_minus_origA[i_elem] = float(p_dest[i_elem] - v_orig); // Calculate dest - orig
                }
            }
        }
    }

    // Check if the sub-mesh indices are valid
    arma::uword n_sub = 1ULL;                                     // Number of sub-meshes (at least 1)
    arma::Col<unsigned> smi(1);                                   // Sub-mesh-index (local copy)
    if (sub_mesh_index != nullptr && sub_mesh_index->n_elem != 0) // Input is available
    {
        n_sub = sub_mesh_index->n_elem;
        const unsigned *p_sub = sub_mesh_index->memptr();

        if (*p_sub != 0U)
            throw std::invalid_argument("First sub-mesh must start at index 0.");

        for (arma::uword i = 1ULL; i < n_sub; ++i)
        {
            if (p_sub[i] <= p_sub[i - 1ULL])
                throw std::invalid_argument("Sub-mesh indices must be sorted in ascending order.");

            if (p_sub[i] % VEC_SIZE != 0ULL)
                throw std::invalid_argument("Sub-meshes must be aligned with the SIMD vector size (8 for AVX2, 32 for CUDA).");
        }

        if (p_sub[n_sub - 1ULL] >= (unsigned)n_mesh)
            throw std::invalid_argument("Sub-mesh indices cannot exceed number of mesh elements.");

        smi = *sub_mesh_index;
    }

    // Alignment to 32 byte addresses is required when loading data into AVX2 registers
    // Not doing this may cause segmentation faults (e.g. in MATLAB)
    arma::uword n_mesh_s = (n_mesh % VEC_SIZE == 0ULL) ? n_mesh : VEC_SIZE * (n_mesh / VEC_SIZE + 1ULL);
    arma::uword n_sub_s = (n_sub % VEC_SIZE == 0ULL) ? n_sub : VEC_SIZE * (n_sub / VEC_SIZE + 1ULL);

#if defined(_MSC_VER) // Windows
    float *Tx = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *Ty = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *Tz = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *E1x = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *E1y = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *E1z = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *E2x = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *E2y = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *E2z = (float *)_aligned_malloc(n_mesh_s * sizeof(float), 32);
    float *Xmin = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Xmax = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Ymin = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Ymax = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Zmin = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
    float *Zmax = (float *)_aligned_malloc(n_sub_s * sizeof(float), 32);
#else // Linux
    float *Tx = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *Ty = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *Tz = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *E1x = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *E1y = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *E1z = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *E2x = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *E2y = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *E2z = (float *)aligned_alloc(32, n_mesh_s * sizeof(float));
    float *Xmin = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Xmax = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Ymin = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Ymax = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Zmin = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
    float *Zmax = (float *)aligned_alloc(32, n_sub_s * sizeof(float));
#endif

    // Convert mesh to float and write to aligned memory
    // Calculate bounding box for each sub-meshes
    const dtype *p_mesh = mesh->memptr();
    const unsigned *p_sub = smi.memptr();

    // Set parameters for the first AABB
    arma::uword i_sub = 0ULL, i_next = (n_sub == 1ULL) ? n_mesh - 1ULL : (arma::uword)p_sub[1] - 1ULL;
    float x_min = INFINITY, x_max = -INFINITY,
          y_min = INFINITY, y_max = -INFINITY,
          z_min = INFINITY, z_max = -INFINITY;

    // Lambda to process each mesh element and write the data to the aligned memory
    auto process_mesh_element = [&](arma::uword i_mesh, dtype x1, dtype y1, dtype z1, dtype x2, dtype y2, dtype z2, dtype x3, dtype y3, dtype z3)
    {
        // Typecast to float and update AABB
        float xf = (float)x1, yf = (float)y1, zf = (float)z1;
        x_min = (xf < x_min) ? xf : x_min, x_max = (xf > x_max) ? xf : x_max;
        y_min = (yf < y_min) ? yf : y_min, y_max = (yf > y_max) ? yf : y_max;
        z_min = (zf < z_min) ? zf : z_min, z_max = (zf > z_max) ? zf : z_max;

        // Write to aligned memory
        Tx[i_mesh] = xf, Ty[i_mesh] = yf, Tz[i_mesh] = zf;

        // Typecast to float and update AABB
        xf = (float)x2, yf = (float)y2, zf = (float)z2;
        x_min = (xf < x_min) ? xf : x_min, x_max = (xf > x_max) ? xf : x_max;
        y_min = (yf < y_min) ? yf : y_min, y_max = (yf > y_max) ? yf : y_max;
        z_min = (zf < z_min) ? zf : z_min, z_max = (zf > z_max) ? zf : z_max;

        // Calculate edge and write to aligned memory
        E1x[i_mesh] = float(x2 - x1);
        E1y[i_mesh] = float(y2 - y1);
        E1z[i_mesh] = float(z2 - z1);

        // Typecast to float and update AABB
        xf = (float)x3, yf = (float)y3, zf = (float)z3;
        x_min = (xf < x_min) ? xf : x_min, x_max = (xf > x_max) ? xf : x_max;
        y_min = (yf < y_min) ? yf : y_min, y_max = (yf > y_max) ? yf : y_max;
        z_min = (zf < z_min) ? zf : z_min, z_max = (zf > z_max) ? zf : z_max;

        // Calculate edge and write to aligned memory
        E2x[i_mesh] = float(x3 - x1);
        E2y[i_mesh] = float(y3 - y1);
        E2z[i_mesh] = float(z3 - z1);

        // Update sub-mesh data for the next AABB
        if (i_mesh == i_next)
        {
            // Write current AABB data to aligned memory
            Xmin[i_sub] = x_min, Xmax[i_sub] = x_max;
            Ymin[i_sub] = y_min, Ymax[i_sub] = y_max;
            Zmin[i_sub] = z_min, Zmax[i_sub] = z_max;

            // Reset registers
            x_min = INFINITY, x_max = -INFINITY,
            y_min = INFINITY, y_max = -INFINITY,
            z_min = INFINITY, z_max = -INFINITY;

            // Update counters
            ++i_sub;
            i_next = (i_sub == n_sub - 1ULL) ? n_mesh - 1ULL : (arma::uword)p_sub[i_sub + 1ULL] - 1ULL;
        }
    };

    if (transpose_inputs)
    {
        for (arma::uword i_mesh = 0ULL; i_mesh < n_mesh; ++i_mesh)
        {
            arma::uword offset = o_mesh * i_mesh;

            // Load first vertex
            dtype x1 = p_mesh[offset],
                  y1 = p_mesh[offset + 1ULL],
                  z1 = p_mesh[offset + 2ULL];

            // Load second vertex
            dtype x2 = p_mesh[offset + 3ULL],
                  y2 = p_mesh[offset + 4ULL],
                  z2 = p_mesh[offset + 5ULL];

            // Load third vertex
            dtype x3 = p_mesh[offset + 6ULL],
                  y3 = p_mesh[offset + 7ULL],
                  z3 = p_mesh[offset + 8ULL];

            process_mesh_element(i_mesh, x1, y1, z1, x2, y2, z2, x3, y3, z3);
        }
    }
    else
    {
        for (arma::uword i_mesh = 0ULL; i_mesh < n_mesh; ++i_mesh)
        {
            // Load first vertex
            dtype x1 = p_mesh[i_mesh],
                  y1 = p_mesh[i_mesh + n_mesh],
                  z1 = p_mesh[i_mesh + 2ULL * n_mesh];

            // Load second vertex
            dtype x2 = p_mesh[i_mesh + 3ULL * n_mesh],
                  y2 = p_mesh[i_mesh + 4ULL * n_mesh],
                  z2 = p_mesh[i_mesh + 5ULL * n_mesh];

            // Load third vertex
            dtype x3 = p_mesh[i_mesh + 6ULL * n_mesh],
                  y3 = p_mesh[i_mesh + 7ULL * n_mesh],
                  z3 = p_mesh[i_mesh + 8ULL * n_mesh];

            process_mesh_element(i_mesh, x1, y1, z1, x2, y2, z2, x3, y3, z3);
        }
    }

    // Add padding to the aligned mesh data
    for (arma::uword i_mesh = n_mesh; i_mesh < n_mesh_s; ++i_mesh)
    {
        Tx[i_mesh] = 0.0f, Ty[i_mesh] = 0.0f, Tz[i_mesh] = 0.0f,
        E1x[i_mesh] = 0.0f, E1y[i_mesh] = 0.0f, E1z[i_mesh] = 0.0f,
        E2x[i_mesh] = 0.0f, E2y[i_mesh] = 0.0f, E2z[i_mesh] = 0.0f;
    }

    // Add padding to the aligned AABB data
    for (arma::uword i_sub = n_sub; i_sub < n_sub_s; ++i_sub)
    {
        Xmin[i_sub] = 0.0f, Xmax[i_sub] = 0.0f;
        Ymin[i_sub] = 0.0f, Ymax[i_sub] = 0.0f;
        Zmin[i_sub] = 0.0f, Zmax[i_sub] = 0.0f;
    }

    // Define and initialize temporary variables
    arma::fvec Wf(n_ray), Ws(n_ray);    // Normalized FBS and SBS hit distances, initialized to 0
    arma::u32_vec If(n_ray), Is(n_ray); // Index of mesh element hit at FBS/SBS, initialized to 0
    arma::u32_vec hit_cnt(n_ray);       // Hit counter

    // Pointer to hit counter
    unsigned *p_hit_cnt = (no_interact == nullptr) ? nullptr : hit_cnt.memptr();

    if (isAVX2Supported()) // CPU support for AVX2
    {
        qd_RTI_AVX2(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh_s,
                    smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                    origA.colptr(0), origA.colptr(1), origA.colptr(2),
                    dest_minus_origA.colptr(0), dest_minus_origA.colptr(1), dest_minus_origA.colptr(2),
                    n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
    }
    else
    {
        qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                       smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                       origA.colptr(0), origA.colptr(1), origA.colptr(2),
                       dest_minus_origA.colptr(0), dest_minus_origA.colptr(1), dest_minus_origA.colptr(2),
                       n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
    }

    // Free aligned memory
#if defined(_MSC_VER) // Windows
    _aligned_free(Tx), _aligned_free(Ty), _aligned_free(Tz);
    _aligned_free(E1x), _aligned_free(E1y), _aligned_free(E1z);
    _aligned_free(E2x), _aligned_free(E2y), _aligned_free(E2z);
    _aligned_free(Xmin), _aligned_free(Xmax);
    _aligned_free(Ymin), _aligned_free(Ymax);
    _aligned_free(Zmin), _aligned_free(Zmax);
#else // Linux
    free(Tx), free(Ty), free(Tz);
    free(E1x), free(E1y), free(E1z);
    free(E2x), free(E2y), free(E2z);
    free(Xmin), free(Xmax);
    free(Ymin), free(Ymax);
    free(Zmin), free(Zmax);
#endif

    // Pointer to origin coordinates
    const dtype *ox = orig->colptr(0), *oy = orig->colptr(1), *oz = orig->colptr(2);

    // Pointer to dest_minus_origA
    float *dx = dest_minus_origA.colptr(0), *dy = dest_minus_origA.colptr(1), *dz = dest_minus_origA.colptr(2);

    // Compute FBS location in GCS
    if (fbs != nullptr)
    {
        if (fbs->n_rows != n_ray || fbs->n_cols != 3)
            fbs->set_size(n_ray, 3);

        dtype *px = fbs->colptr(0), *py = fbs->colptr(1), *pz = fbs->colptr(2);
        float *w = Wf.memptr();

        for (arma::uword i = 0; i < n_ray; ++i)
        {
            px[i] = ox[i] + dtype(w[i] * dx[i]);
            py[i] = oy[i] + dtype(w[i] * dy[i]);
            pz[i] = oz[i] + dtype(w[i] * dz[i]);
        }
    }

    // Compute SBS location in GCS
    if (sbs != nullptr)
    {
        if (sbs->n_rows != n_ray || sbs->n_cols != 3)
            sbs->set_size(n_ray, 3);

        dtype *px = sbs->colptr(0), *py = sbs->colptr(1), *pz = sbs->colptr(2);
        float *w = Ws.memptr();

        for (arma::uword i = 0; i < n_ray; ++i)
        {
            px[i] = ox[i] + dtype(w[i] * dx[i]);
            py[i] = oy[i] + dtype(w[i] * dy[i]);
            pz[i] = oz[i] + dtype(w[i] * dz[i]);
        }
    }

    // Copy the rest
    arma::uword no_bytes = (arma::uword)n_ray * sizeof(unsigned);
    if (no_interact != nullptr)
    {
        if (no_interact->n_elem != n_ray)
            no_interact->set_size(n_ray);
        std::memcpy(no_interact->memptr(), p_hit_cnt, no_bytes);
    }
    if (fbs_ind != nullptr)
    {
        if (fbs_ind->n_elem != n_ray)
            fbs_ind->set_size(n_ray);
        std::memcpy(fbs_ind->memptr(), If.memptr(), no_bytes);
    }
    if (sbs_ind != nullptr)
    {
        if (sbs_ind->n_elem != n_ray)
            sbs_ind->set_size(n_ray);
        std::memcpy(sbs_ind->memptr(), Is.memptr(), no_bytes);
    }
}

template void quadriga_lib::ray_triangle_intersect(const arma::Mat<float> *orig, const arma::Mat<float> *dest, const arma::Mat<float> *mesh,
                                                   arma::Mat<float> *fbs, arma::Mat<float> *sbs, arma::Col<unsigned> *no_interact,
                                                   arma::Col<unsigned> *fbs_ind, arma::Col<unsigned> *sbs_ind,
                                                   const arma::Col<unsigned> *sub_mesh_index, bool transpose_inputs);

template void quadriga_lib::ray_triangle_intersect(const arma::Mat<double> *orig, const arma::Mat<double> *dest, const arma::Mat<double> *mesh,
                                                   arma::Mat<double> *fbs, arma::Mat<double> *sbs, arma::Col<unsigned> *no_interact,
                                                   arma::Col<unsigned> *fbs_ind, arma::Col<unsigned> *sbs_ind,
                                                   const arma::Col<unsigned> *sub_mesh_index, bool transpose_inputs);