// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib.hpp"
#include "quadriga_lib_generic_functions.hpp"

#if BUILD_WITH_AVX2
#include "quadriga_lib_avx2_functions.hpp"
#endif

#if BUILD_WITH_CUDA
#include "quadriga_lib_cuda_functions.hpp"
#endif

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# ray_point_intersect
Calculate intersections of ray beams with points in 3D space

- Models rays as volumetric beams defined by a triangular wavefront that diverges from the origin, enabling energy spread simulation.
- Returns, for each point, the list of 0-based ray indices whose beam intersects that point.
- All internal computations use single precision.

## Declaration:
```
std::vector<arma::u32_vec> quadriga_lib::ray_point_intersect(
    const arma::Mat<dtype> *points,
    const arma::Mat<dtype> *orig,
    const arma::Mat<dtype> *trivec,
    const arma::Mat<dtype> *tridir,
    const arma::u32_vec *sub_cloud_index = nullptr,
    arma::u32_vec *hit_count = nullptr,
    int use_kernel = 0,
    int gpu_id = 0);
```

## Inputs:
- **`points`** — 3D point cloud coordinates; `[n_points, 3]`
- **`orig`** — Ray origin positions in global Cartesian coordinates; `[n_ray, 3]`
- **`trivec`** — Vectors from ray origin center to triangular wavefront vertices, order `[v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z]`; `[n_ray, 9]`
- **`tridir`** — Direction vectors of the three vertex-rays in Cartesian coordinates (need not be normalized), order `[d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z]`; `[n_ray, 9]`
- **`sub_cloud_index`** *(optional)* — Segment boundary indices for the point cloud (see [[point_cloud_segmentation]]); `[n_sub]`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable; auto mode selects CUDA when `n_points >= 10000` and CUDA is available, else AVX2, else GENERIC.
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA

## Optional output:
- **`hit_count`** — Number of rays intersecting each point; `[n_points]`

## Returns:
- `std::vector<arma::u32_vec>` — Per-point list of 0-based ray indices that intersected that point; length `n_points`

## See also:
- [[icosphere]] (generate ray beams)
- [[point_cloud_segmentation]] (generate sub-cloud index)
- [[subdivide_rays]] (subdivide beams into sub-beams)
- [[ray_triangle_intersect]] (ray–triangle intersection)
- [[ray_mesh_interact]] (beam–mesh interaction)
MD!*/

template <typename dtype>
std::vector<arma::u32_vec> quadriga_lib::ray_point_intersect(const arma::Mat<dtype> *points,
                                                             const arma::Mat<dtype> *orig,
                                                             const arma::Mat<dtype> *trivec,
                                                             const arma::Mat<dtype> *tridir,
                                                             const arma::u32_vec *sub_cloud_index,
                                                             arma::u32_vec *hit_count,
                                                             int use_kernel, int gpu_id)
{
    // Suppress unused-parameter warning when CUDA support is disabled at compile time
#if !BUILD_WITH_CUDA
    (void)gpu_id;
#endif

    // Input validation
    if (points == nullptr || points->n_elem == 0)
        throw std::invalid_argument("Input 'points' cannot be NULL.");
    if (orig == nullptr || orig->n_elem == 0)
        throw std::invalid_argument("Input 'orig' cannot be NULL.");
    if (trivec == nullptr)
        throw std::invalid_argument("Input 'trivec' cannot be NULL.");
    if (tridir == nullptr)
        throw std::invalid_argument("Input 'tridir' cannot be NULL.");

    if (points->n_cols != 3)
        throw std::invalid_argument("Input 'points' must have 3 columns containing x,y,z coordinates.");
    if (orig->n_cols != 3)
        throw std::invalid_argument("Input 'orig' must have 3 columns containing x,y,z coordinates.");
    if (trivec->n_cols != 9)
        throw std::invalid_argument("Input 'trivec' must have 9 columns containing x,y,z coordinates of ray tube vertices.");
    if (tridir->n_cols != 9)
        throw std::invalid_argument("Input 'tridir' must have 9 columns containing ray directions in Cartesian format.");

    size_t n_ray_t = orig->n_rows;
    size_t n_point_t = (size_t)points->n_rows;
    int n_ray_i = (int)n_ray_t;

    // Bound check
    if (n_point_t >= INT32_MAX)
        throw std::invalid_argument("Number of points exceeds maximum supported number.");
    if (n_ray_t >= INT32_MAX)
        throw std::invalid_argument("Number of rays exceeds maximum supported number.");

    if (trivec->n_rows != n_ray_t)
        throw std::invalid_argument("Number of rows in 'orig' and 'trivec' dont match.");
    if (tridir->n_rows != n_ray_t)
        throw std::invalid_argument("Number of rows in 'orig' and 'tridir' dont match.");

    // Determine which compute kernel to use
    // kernel: 1 = GENERIC, 2 = AVX2, 3 = CUDA
    int kernel = 1;      // Default to GENERIC
    if (use_kernel == 1) // GENERIC requested
    {
        kernel = 1;
    }
    else if (use_kernel == 2) // AVX2 requested
    {
        if (!quadriga_lib::quadriga_lib_has_AVX2())
            throw std::invalid_argument("AVX2 kernel requested but not available (compile with BUILD_WITH_AVX2 and run on AVX2-capable CPU).");
        kernel = 2;
    }
    else if (use_kernel == 3) // CUDA requested
    {
        if (!quadriga_lib::quadriga_lib_has_CUDA())
            throw std::invalid_argument("CUDA kernel requested but not available (compile with BUILD_WITH_CUDA and run on CUDA-capable GPU).");
        kernel = 3;
    }
    else // Auto-select (use_kernel == 0)
    {
        if (n_point_t >= 10000 && quadriga_lib::quadriga_lib_has_CUDA())
            kernel = 3;
        else if (quadriga_lib::quadriga_lib_has_AVX2())
            kernel = 2;
        else
            kernel = 1;
    }

    // Determine SIMD vector size based on selected kernel
    size_t vec_size = (kernel == 2) ? 8ULL : 1ULL;

    // Check if the sub-cloud indices are valid
    size_t n_sub_t = 1;                                             // Number of sub-clouds (at least 1)
    arma::u32_vec sci(1, arma::fill::zeros);                        // Sub-cloud-index (local copy)
    if (sub_cloud_index != nullptr && sub_cloud_index->n_elem != 0) // Input is available
    {
        n_sub_t = (size_t)sub_cloud_index->n_elem;
        const unsigned *p_sub = sub_cloud_index->memptr();

        if (*p_sub != 0U)
            throw std::invalid_argument("First sub-cloud must start at index 0.");

        for (size_t i = 1; i < n_sub_t; ++i)
        {
            if (p_sub[i] <= p_sub[i - 1])
                throw std::invalid_argument("Sub-cloud indices must be sorted in ascending order.");

            if (vec_size > 1ULL && p_sub[i] % vec_size != 0)
                throw std::invalid_argument("Sub-clouds must be aligned with the SIMD vector size (8 for AVX2).");
        }

        if (p_sub[n_sub_t - 1] >= (unsigned)n_point_t)
            throw std::invalid_argument("Sub-cloud indices cannot exceed number of points.");

        sci = *sub_cloud_index;
    }

    // Prepare ray data
    auto trivecA = arma::fmat(n_ray_t, 9, arma::fill::none); // Vertex origins in GCS
    auto normalA = arma::fmat(n_ray_t, 3, arma::fill::none); // Normal vector
    auto dirA = arma::fmat(n_ray_t, 9, arma::fill::none);    // Vertex directions (Cartesian)
    auto invDotA = arma::fmat(n_ray_t, 3, arma::fill::none); // Inverse dot product
    {
        const dtype *p_orig = orig->memptr();     // Origin pointer
        const dtype *p_trivec = trivec->memptr(); // Trivec pointer
        const dtype *p_tridir = tridir->memptr(); // Direction pointer

        float *p_trivecA = trivecA.memptr();
        float *p_normalA = normalA.memptr();
        float *p_dirA = dirA.memptr();
        float *p_invDotA = invDotA.memptr();

#pragma omp parallel for
        for (int i_ray = 0; i_ray < n_ray_i; ++i_ray)
        {
            // Load origin
            dtype Nx = p_orig[i_ray],
                  Ny = p_orig[i_ray + n_ray_i],
                  Nz = p_orig[i_ray + 2 * n_ray_i];

            // Load first vertex
            dtype Vx = p_trivec[i_ray],
                  Vy = p_trivec[i_ray + n_ray_i],
                  Vz = p_trivec[i_ray + 2 * n_ray_i];

            // Calculate first vertex location in GCS
            p_trivecA[i_ray] = float(Nx + Vx);
            p_trivecA[i_ray + n_ray_i] = float(Ny + Vy);
            p_trivecA[i_ray + 2 * n_ray_i] = float(Nz + Vz);

            // Load second vertex
            dtype Ux = p_trivec[i_ray + 3 * n_ray_i],
                  Uy = p_trivec[i_ray + 4 * n_ray_i],
                  Uz = p_trivec[i_ray + 5 * n_ray_i];

            // Calculate second vertex location in GCS
            p_trivecA[i_ray + 3 * n_ray_i] = float(Nx + Ux);
            p_trivecA[i_ray + 4 * n_ray_i] = float(Ny + Uy);
            p_trivecA[i_ray + 5 * n_ray_i] = float(Nz + Uz);

            // Calculate edge from first to second vertex
            Ux -= Vx, Uy -= Vy, Uz -= Vz;

            // Process third vertex
            dtype tmp = p_trivec[i_ray + 6 * n_ray_i];
            p_trivecA[i_ray + 6 * n_ray_i] = float(Nx + tmp);
            Vx = tmp - Vx;

            tmp = p_trivec[i_ray + 7 * n_ray_i];
            p_trivecA[i_ray + 7 * n_ray_i] = float(Ny + tmp);
            Vy = tmp - Vy;

            tmp = p_trivec[i_ray + 8 * n_ray_i];
            p_trivecA[i_ray + 8 * n_ray_i] = float(Nz + tmp);
            Vz = tmp - Vz;

            // Calculate Normal Vector
            Nx = Uy * Vz - Uz * Vy;
            Ny = Uz * Vx - Ux * Vz;
            Nz = Ux * Vy - Uy * Vx;

            // Convert to float
            p_normalA[i_ray] = float(Nx);
            p_normalA[i_ray + n_ray_i] = float(Ny);
            p_normalA[i_ray + 2 * n_ray_i] = float(Nz);

            // Load first vertex direction
            Vx = p_tridir[i_ray];
            Vy = p_tridir[i_ray + n_ray_i];
            Vz = p_tridir[i_ray + 2 * n_ray_i];

            // Normalize it, if needed
            tmp = Vx * Vx + Vy * Vy + Vz * Vz;
            if (std::abs(tmp - (dtype)1.0) > (dtype)2e-7)
                tmp = std::sqrt((dtype)1.0 / tmp), Vx *= tmp, Vy *= tmp, Vz *= tmp;

            // Store as float
            p_dirA[i_ray] = float(Vx);
            p_dirA[i_ray + n_ray_i] = float(Vy);
            p_dirA[i_ray + 2 * n_ray_i] = float(Vz);

            // Calculate inverse DotProduct from Normal Vector and Vertex direction
            tmp = Vx * Nx + Vy * Ny + Vz * Nz;
            p_invDotA[i_ray] = float((dtype)1.0 / tmp);

            // Load second vertex direction
            Vx = p_tridir[i_ray + 3 * n_ray_i];
            Vy = p_tridir[i_ray + 4 * n_ray_i];
            Vz = p_tridir[i_ray + 5 * n_ray_i];

            // Normalize it, if needed
            tmp = Vx * Vx + Vy * Vy + Vz * Vz;
            if (std::abs(tmp - (dtype)1.0) > (dtype)2e-7)
                tmp = std::sqrt((dtype)1.0 / tmp), Vx *= tmp, Vy *= tmp, Vz *= tmp;

            // Store as float
            p_dirA[i_ray + 3 * n_ray_i] = float(Vx);
            p_dirA[i_ray + 4 * n_ray_i] = float(Vy);
            p_dirA[i_ray + 5 * n_ray_i] = float(Vz);

            // Calculate inverse DotProduct from Normal Vector and Vertex direction
            tmp = Vx * Nx + Vy * Ny + Vz * Nz;
            p_invDotA[i_ray + n_ray_i] = float((dtype)1.0 / tmp);

            // Load third vertex direction
            Vx = p_tridir[i_ray + 6 * n_ray_i];
            Vy = p_tridir[i_ray + 7 * n_ray_i];
            Vz = p_tridir[i_ray + 8 * n_ray_i];

            // Normalize it, if needed
            tmp = Vx * Vx + Vy * Vy + Vz * Vz;
            if (std::abs(tmp - (dtype)1.0) > (dtype)2e-7)
                tmp = std::sqrt((dtype)1.0 / tmp), Vx *= tmp, Vy *= tmp, Vz *= tmp;

            // Store as float
            p_dirA[i_ray + 6 * n_ray_i] = float(Vx);
            p_dirA[i_ray + 7 * n_ray_i] = float(Vy);
            p_dirA[i_ray + 8 * n_ray_i] = float(Vz);

            // Calculate inverse DotProduct from Normal Vector and Vertex direction
            tmp = Vx * Nx + Vy * Ny + Vz * Nz;
            p_invDotA[i_ray + 2 * n_ray_i] = float((dtype)1.0 / tmp);
        }
    }

    // Pad to multiple of vec_size for AVX2 kernel (no tail handling)
    size_t n_point_s = (n_point_t % vec_size == 0) ? n_point_t : vec_size * (n_point_t / vec_size + 1);
    size_t n_sub_s = (n_sub_t % vec_size == 0) ? n_sub_t : vec_size * (n_sub_t / vec_size + 1);

    // Point data in SoA layout
    arma::fvec Px_vec(n_point_s, arma::fill::none), Py_vec(n_point_s, arma::fill::none), Pz_vec(n_point_s, arma::fill::none);
    float *Px = Px_vec.memptr(), *Py = Py_vec.memptr(), *Pz = Pz_vec.memptr();

    // AABB data per sub-cloud
    arma::fvec Xmin_vec(n_sub_s, arma::fill::none), Xmax_vec(n_sub_s, arma::fill::none);
    arma::fvec Ymin_vec(n_sub_s, arma::fill::none), Ymax_vec(n_sub_s, arma::fill::none);
    arma::fvec Zmin_vec(n_sub_s, arma::fill::none), Zmax_vec(n_sub_s, arma::fill::none);
    float *Xmin = Xmin_vec.memptr(), *Xmax = Xmax_vec.memptr();
    float *Ymin = Ymin_vec.memptr(), *Ymax = Ymax_vec.memptr();
    float *Zmin = Zmin_vec.memptr(), *Zmax = Zmax_vec.memptr();

    // Convert points to float and compute bounding boxes
    // Calculate bounding box for each sub-cloud
    const dtype *p_points = points->memptr();
    const unsigned *p_sub = sci.memptr();

    // Set parameters for the first AABB
    size_t i_sub = 0, i_next = (n_sub_t == 1) ? n_point_t - 1 : (size_t)p_sub[1] - 1;
    float x_min = INFINITY, x_max = -INFINITY,
          y_min = INFINITY, y_max = -INFINITY,
          z_min = INFINITY, z_max = -INFINITY;

    for (size_t i_point = 0; i_point < n_point_t; ++i_point)
    {
        // Load point
        dtype x = p_points[i_point],
              y = p_points[i_point + n_point_t],
              z = p_points[i_point + 2 * n_point_t];

        // Typecast to float and update AABB
        float xf = (float)x, yf = (float)y, zf = (float)z;
        x_min = (xf < x_min) ? xf : x_min, x_max = (xf > x_max) ? xf : x_max;
        y_min = (yf < y_min) ? yf : y_min, y_max = (yf > y_max) ? yf : y_max;
        z_min = (zf < z_min) ? zf : z_min, z_max = (zf > z_max) ? zf : z_max;

        // Write to float buffer
        Px[i_point] = xf, Py[i_point] = yf, Pz[i_point] = zf;

        // Update sub-cloud data for the next AABB
        if (i_point == i_next)
        {
            // Write current AABB data
            Xmin[i_sub] = x_min, Xmax[i_sub] = x_max;
            Ymin[i_sub] = y_min, Ymax[i_sub] = y_max;
            Zmin[i_sub] = z_min, Zmax[i_sub] = z_max;

            // Reset registers
            x_min = INFINITY, x_max = -INFINITY,
            y_min = INFINITY, y_max = -INFINITY,
            z_min = INFINITY, z_max = -INFINITY;

            // Update counters
            ++i_sub;
            i_next = (i_sub == n_sub_t - 1) ? n_point_t - 1 : (size_t)p_sub[i_sub + 1] - 1;
        }
    }

    // Add padding to the point data
    for (size_t i_point = n_point_t; i_point < n_point_s; ++i_point)
        Px[i_point] = 0.0f, Py[i_point] = 0.0f, Pz[i_point] = 0.0f;

    // Add padding to the AABB data
    for (size_t i_sub = n_sub_t; i_sub < n_sub_s; ++i_sub)
    {
        Xmin[i_sub] = 0.0f, Xmax[i_sub] = 0.0f;
        Ymin[i_sub] = 0.0f, Ymax[i_sub] = 0.0f;
        Zmin[i_sub] = 0.0f, Zmax[i_sub] = 0.0f;
    }

    // Output container
    std::vector<std::vector<unsigned>> hit_vec(n_ray_t);
    std::vector<unsigned> *p_hit = hit_vec.data();

    // Dispatch to selected kernel
    if (kernel == 3) // CUDA
    {
#if BUILD_WITH_CUDA
        qd_RPI_CUDA(Px, Py, Pz, n_point_s,
                    sci.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub_t,
                    trivecA.colptr(0), trivecA.colptr(1), trivecA.colptr(2),
                    trivecA.colptr(3), trivecA.colptr(4), trivecA.colptr(5),
                    trivecA.colptr(6), trivecA.colptr(7), trivecA.colptr(8),
                    normalA.colptr(0), normalA.colptr(1), normalA.colptr(2),
                    dirA.colptr(0), dirA.colptr(1), dirA.colptr(2),
                    dirA.colptr(3), dirA.colptr(4), dirA.colptr(5),
                    dirA.colptr(6), dirA.colptr(7), dirA.colptr(8),
                    invDotA.colptr(0), invDotA.colptr(1), invDotA.colptr(2),
                    n_ray_t, p_hit, gpu_id);
#endif
    }
    else if (kernel == 2) // AVX2
    {
#if BUILD_WITH_AVX2
        qd_RPI_AVX2(Px, Py, Pz, n_point_s,
                    sci.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub_t,
                    trivecA.colptr(0), trivecA.colptr(1), trivecA.colptr(2),
                    trivecA.colptr(3), trivecA.colptr(4), trivecA.colptr(5),
                    trivecA.colptr(6), trivecA.colptr(7), trivecA.colptr(8),
                    normalA.colptr(0), normalA.colptr(1), normalA.colptr(2),
                    dirA.colptr(0), dirA.colptr(1), dirA.colptr(2),
                    dirA.colptr(3), dirA.colptr(4), dirA.colptr(5),
                    dirA.colptr(6), dirA.colptr(7), dirA.colptr(8),
                    invDotA.colptr(0), invDotA.colptr(1), invDotA.colptr(2),
                    n_ray_t, p_hit);
#endif
    }
    else // GENERIC
    {
        qd_RPI_GENERIC(Px, Py, Pz, n_point_t,
                       sci.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub_t,
                       trivecA.colptr(0), trivecA.colptr(1), trivecA.colptr(2),
                       trivecA.colptr(3), trivecA.colptr(4), trivecA.colptr(5),
                       trivecA.colptr(6), trivecA.colptr(7), trivecA.colptr(8),
                       normalA.colptr(0), normalA.colptr(1), normalA.colptr(2),
                       dirA.colptr(0), dirA.colptr(1), dirA.colptr(2),
                       dirA.colptr(3), dirA.colptr(4), dirA.colptr(5),
                       dirA.colptr(6), dirA.colptr(7), dirA.colptr(8),
                       invDotA.colptr(0), invDotA.colptr(1), invDotA.colptr(2),
                       n_ray_t, p_hit);
    }

    // Count hits per point
    arma::u32_vec cnt_vec(n_point_s, arma::fill::zeros);
    unsigned *p_cnt = cnt_vec.memptr();

    for (size_t i_ray = 0; i_ray < n_ray_t; ++i_ray)
        for (size_t i_hit = 0; i_hit < p_hit[i_ray].size(); ++i_hit)
            ++p_cnt[p_hit[i_ray][i_hit]];

    if (hit_count != nullptr)
    {
        if (hit_count->n_elem != n_point_t)
            hit_count->set_size(n_point_t);

        std::memcpy(hit_count->memptr(), p_cnt, n_point_t * sizeof(unsigned));
    }

    // Generate output
    std::vector<arma::u32_vec> output(n_point_t);

    for (size_t i_point = 0; i_point < n_point_t; ++i_point)
    {
        if (p_cnt[i_point] != 0)
            output[i_point].set_size(p_cnt[i_point]);
        p_cnt[i_point] = 0;
    }

    for (size_t i_ray = 0; i_ray < n_ray_t; ++i_ray)
        for (size_t i_hit = 0; i_hit < p_hit[i_ray].size(); ++i_hit)
        {
            unsigned i_point = p_hit[i_ray][i_hit];
            if (i_point < n_point_t)
                output[i_point].at(p_cnt[i_point]++) = (unsigned)i_ray;
        }

    return output;
}

template std::vector<arma::u32_vec> quadriga_lib::ray_point_intersect(const arma::Mat<float> *points,
                                                                      const arma::Mat<float> *orig,
                                                                      const arma::Mat<float> *trivec,
                                                                      const arma::Mat<float> *tridir,
                                                                      const arma::u32_vec *sub_cloud_index,
                                                                      arma::u32_vec *hit_count,
                                                                      int use_kernel, int gpu_id);

template std::vector<arma::u32_vec> quadriga_lib::ray_point_intersect(const arma::Mat<double> *points,
                                                                      const arma::Mat<double> *orig,
                                                                      const arma::Mat<double> *trivec,
                                                                      const arma::Mat<double> *tridir,
                                                                      const arma::u32_vec *sub_cloud_index,
                                                                      arma::u32_vec *hit_count,
                                                                      int use_kernel, int gpu_id);