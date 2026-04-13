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

#include "quadriga_lib.hpp"
#include "quadriga_lib_generic_functions.hpp"

#if BUILD_WITH_AVX2
#include "quadriga_lib_avx2_functions.hpp"
#endif

#if BUILD_WITH_CUDA
#include "quadriga_lib_cuda_functions.hpp"
#endif

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# ray_triangle_intersect
Calculates the intersection of rays and triangles in three dimensions

## Description:
- Implements the Möller–Trumbore algorithm to compute intersections between rays and triangles in 3D.
- Supports three compute kernels: **GENERIC** (scalar), **AVX2** (SIMD, 8 triangles in parallel), and **CUDA** (GPU).
- The `use_kernel` parameter selects the kernel: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA.
- In auto mode (0), CUDA is selected only when `n_ray >= 10000` and a CUDA-capable GPU is available;
  otherwise AVX2 is preferred if available, falling back to GENERIC.
- Can detect first and second intersections (FBS/SBS), number of intersections, and intersection indices.
- Allowed datatypes (`dtype`): `float` or `double`
- Internal computations are carried out in **single precision** regardless of `dtype`.

## Declaration:
```
void quadriga_lib::ray_triangle_intersect(
                const arma::Mat<dtype> *orig,
                const arma::Mat<dtype> *dest,
                const arma::Mat<dtype> *mesh,
                arma::Mat<dtype> *fbs = nullptr,
                arma::Mat<dtype> *sbs = nullptr,
                arma::u32_vec *no_interact = nullptr,
                arma::u32_vec *fbs_ind = nullptr,
                arma::u32_vec *sbs_ind = nullptr,
                const arma::u32_vec *sub_mesh_index = nullptr,
                const arma::Mat<dtype> *aabb = nullptr,
                int use_kernel = 0,
                int gpu_id = 0);
```

## Arguments:
- `const arma::Mat<dtype> ***orig**` (input)<br>
  Ray origins in global coordinate system (GCS). Size: `[n_ray, 3]`.

- `const arma::Mat<dtype> ***dest**` (input)<br>
  Ray destinations in GCS. Size: `[n_ray, 3]`.

- `const arma::Mat<dtype> ***mesh**` (input)<br>
  Triangular surface mesh. Size: `[n_mesh, 9]`, where each row contains the 3 vertices
  `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`.

- `arma::Mat<dtype> ***fbs**` (optional output)<br>
  First-bounce surface intersection points (FBS). Size: `[n_ray, 3]`.

- `arma::Mat<dtype> ***sbs**` (optional output)<br>
  Second-bounce surface intersection points (SBS). Size: `[n_ray, 3]`.

- `arma::u32_vec ***no_interact**` (optional output)<br>
  Number of intersections per ray (0, 1, or 2). Size: `[n_ray]`.

- `arma::u32_vec ***fbs_ind**` (optional output)<br>
  1-based index of the first intersected mesh element, 0 = no intersection. Size: `[n_ray]`.

- `arma::u32_vec ***sbs_ind**` (optional output)<br>
  1-based index of the second intersected mesh element, 0 = no second intersection. Size: `[n_ray]`.

- `const arma::u32_vec ***sub_mesh_index**` (optional input)<br>
  Indexes indicating start of sub-meshes in `mesh`. Size: `[n_sub]`. Enables faster processing via segmentation.

- `const arma::Mat<dtype> ***aabb**` (optional input)<br>
  Pre-computed axis-aligned bounding boxes per sub-mesh. Size: `[n_sub, 6]`, where each row contains
  `{x_min, x_max, y_min, y_max, z_min, z_max}`. If `nullptr`, AABBs are computed internally from `mesh`.

- `bool **transpose_inputs** = false` (optional input)<br>
  If `true`, treats `orig`/`dest` as `[3, n_ray]` and `mesh` as `[9, n_mesh]`.

- `int **use_kernel** = 0` (optional input)<br>
  Selects the compute kernel: 0 = auto (default), 1 = GENERIC (scalar CPU), 2 = AVX2 (SIMD),
  3 = CUDA (GPU). An error is thrown if the requested kernel is not available at runtime.

- `int **gpu_id** = 0` (optional input)<br>
  GPU device ID for CUDA kernel. Ignored when not using CUDA.

## See also:
- <a href="#obj_file_read">obj_file_read</a> (for loading `mesh` from an OBJ file)
- <a href="#icosphere">icosphere</a> (for generating beams)
- <a href="#triangle_mesh_segmentation">triangle_mesh_segmentation</a> (for calculating sub-meshes)
- <a href="#ray_point_intersect">ray_point_intersect</a> (for calculating beam interactions with sampling points)
- <a href="#subdivide_rays">subdivide_rays</a> (for subdivides ray beams into sub beams)
MD!*/

template <typename dtype>
void quadriga_lib::ray_triangle_intersect(const arma::Mat<dtype> *orig, const arma::Mat<dtype> *dest, const arma::Mat<dtype> *mesh,
                                          arma::Mat<dtype> *fbs, arma::Mat<dtype> *sbs, arma::u32_vec *no_interact,
                                          arma::u32_vec *fbs_ind, arma::u32_vec *sbs_ind,
                                          const arma::u32_vec *sub_mesh_index, const arma::Mat<dtype> *aabb,
                                          int use_kernel, int gpu_id)
{
    // Suppress unused-parameter warning when CUDA support is disabled at compile time
#if !BUILD_WITH_CUDA
    (void)gpu_id;
#endif

    // Input validation
    if (orig == nullptr)
        throw std::invalid_argument("Input 'orig' cannot be NULL.");
    if (orig->n_elem == 0)
        throw std::invalid_argument("Inputs cannot be empty.");
    if (orig->n_cols < 3)
        throw std::invalid_argument("Input 'orig' must have at least 3 columns containing x,y,z coordinates.");

    arma::uword n_ray = orig->n_rows;

    if (dest == nullptr)
        throw std::invalid_argument("Input 'dest' cannot be NULL.");
    if (dest->n_cols < 3)
        throw std::invalid_argument("Input 'dest' must have at least 3 columns containing x,y,z coordinates.");
    if (dest->n_rows < n_ray)
        throw std::invalid_argument("Number of elements in 'orig' and 'dest' dont match.");

    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (mesh->n_elem == 0)
        throw std::invalid_argument("Inputs cannot be empty.");
    if (mesh->n_cols < 9)
        throw std::invalid_argument("Input 'mesh' must have at least 9 columns containing x,y,z coordinates of 3 vertices.");

    arma::uword n_mesh = mesh->n_rows;

    // Calculate (dest - orig)
    arma::Mat<dtype> dest_minus_orig(n_ray, 3ULL, arma::fill::none);
    {
        const dtype *p_orig = orig->memptr();
        const dtype *p_dest = dest->memptr();
        dtype *p_dmo = dest_minus_orig.memptr();

        for (arma::uword i_elem = 0; i_elem < 3 * n_ray; ++i_elem)
            p_dmo[i_elem] = p_dest[i_elem] - p_orig[i_elem];
    }

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
        if (n_ray >= 10000ULL && quadriga_lib::quadriga_lib_has_CUDA())
            kernel = 3;
        else if (quadriga_lib::quadriga_lib_has_AVX2())
            kernel = 2;
        else
            kernel = 1;
    }

    // Check if the sub-mesh indices are valid
    arma::uword n_sub = 1ULL;                                     // Number of sub-meshes (at least 1)
    arma::u32_vec smi(1);                                         // Sub-mesh-index (local copy)
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
        }

        if (p_sub[n_sub - 1ULL] >= (unsigned)n_mesh)
            throw std::invalid_argument("Sub-mesh indices cannot exceed number of mesh elements.");

        smi = *sub_mesh_index;
    }

    // Compute or validate AABB
    arma::Mat<dtype> aabb_computed;
    if (aabb == nullptr)
    {
        aabb_computed = quadriga_lib::triangle_mesh_aabb(mesh, sub_mesh_index);
        aabb = &aabb_computed;
    }
    else
    {
        if (aabb->n_rows != n_sub)
            throw std::invalid_argument("Number of rows in 'aabb' must match number of sub-meshes.");
        if (aabb->n_cols < 6)
            throw std::invalid_argument("Input 'aabb' must have at least 6 columns [x_min, x_max, y_min, y_max, z_min, z_max].");
    }

    // AABB data per sub-mesh
    const dtype *Xmin = aabb->colptr(0), *Xmax = aabb->colptr(1);
    const dtype *Ymin = aabb->colptr(2), *Ymax = aabb->colptr(3);
    const dtype *Zmin = aabb->colptr(4), *Zmax = aabb->colptr(5);

    // Convert mesh (first vertex + two edges)
    const dtype *Tx = mesh->colptr(0), *Ty = mesh->colptr(1), *Tz = mesh->colptr(2);

    arma::Col<dtype> E1x_vec(n_mesh, arma::fill::none), E1y_vec(n_mesh, arma::fill::none), E1z_vec(n_mesh, arma::fill::none);
    arma::Col<dtype> E2x_vec(n_mesh, arma::fill::none), E2y_vec(n_mesh, arma::fill::none), E2z_vec(n_mesh, arma::fill::none);

    dtype *E1x = E1x_vec.memptr(), *E1y = E1y_vec.memptr(), *E1z = E1z_vec.memptr();
    dtype *E2x = E2x_vec.memptr(), *E2y = E2y_vec.memptr(), *E2z = E2z_vec.memptr();

    const dtype *p_mesh = mesh->memptr();
    for (arma::uword i_mesh = 0; i_mesh < n_mesh; ++i_mesh)
    {
        dtype x1 = Tx[i_mesh];
        dtype y1 = Ty[i_mesh];
        dtype z1 = Tz[i_mesh];

        dtype x2 = p_mesh[i_mesh + 3 * n_mesh];
        dtype y2 = p_mesh[i_mesh + 4 * n_mesh];
        dtype z2 = p_mesh[i_mesh + 5 * n_mesh];

        dtype x3 = p_mesh[i_mesh + 6 * n_mesh];
        dtype y3 = p_mesh[i_mesh + 7 * n_mesh];
        dtype z3 = p_mesh[i_mesh + 8 * n_mesh];

        E1x[i_mesh] = x2 - x1, E1y[i_mesh] = y2 - y1, E1z[i_mesh] = z2 - z1;
        E2x[i_mesh] = x3 - x1, E2y[i_mesh] = y3 - y1, E2z[i_mesh] = z3 - z1;
    }

    // Define and initialize temporary variables
    arma::Col<dtype> Wf(n_ray), Ws(n_ray); // Normalized FBS and SBS hit distances, initialized to 0
    arma::u32_vec If(n_ray), Is(n_ray);    // Index of mesh element hit at FBS/SBS, initialized to 0
    arma::u32_vec hit_cnt(n_ray);          // Hit counter

    // Pointer to hit counter
    unsigned *p_hit_cnt = (no_interact == nullptr) ? nullptr : hit_cnt.memptr();

    // Dispatch to selected kernel
    if (kernel == 3) // CUDA
    {
#if BUILD_WITH_CUDA
        qd_RTI_CUDA<dtype>(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                           smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                           orig->colptr(0), orig->colptr(1), orig->colptr(2),
                           dest_minus_orig.colptr(0), dest_minus_orig.colptr(1), dest_minus_orig.colptr(2),
                           n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt, gpu_id);
#endif
    }
    else if (kernel == 2) // AVX2
    {
#if BUILD_WITH_AVX2
        qd_RTI_AVX2<dtype>(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                           smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                           orig->colptr(0), orig->colptr(1), orig->colptr(2),
                           dest_minus_orig.colptr(0), dest_minus_orig.colptr(1), dest_minus_orig.colptr(2),
                           n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
#endif
    }
    else // GENERIC
    {
        qd_RTI_GENERIC<dtype>(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                              smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                              orig->colptr(0), orig->colptr(1), orig->colptr(2),
                              dest_minus_orig.colptr(0), dest_minus_orig.colptr(1), dest_minus_orig.colptr(2),
                              n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
    }

    // Pointer to origin coordinates
    const dtype *ox = orig->colptr(0), *oy = orig->colptr(1), *oz = orig->colptr(2);

    // Pointer to dest_minus_origA
    dtype *dx = dest_minus_orig.colptr(0), *dy = dest_minus_orig.colptr(1), *dz = dest_minus_orig.colptr(2);

    // Compute FBS location in GCS
    if (fbs != nullptr)
    {
        if (fbs->n_rows != n_ray || fbs->n_cols != 3)
            fbs->set_size(n_ray, 3);

        dtype *px = fbs->colptr(0), *py = fbs->colptr(1), *pz = fbs->colptr(2);
        dtype *w = Wf.memptr();

        for (arma::uword i = 0; i < n_ray; ++i)
        {
            px[i] = ox[i] + w[i] * dx[i];
            py[i] = oy[i] + w[i] * dy[i];
            pz[i] = oz[i] + w[i] * dz[i];
        }
    }

    // Compute SBS location in GCS
    if (sbs != nullptr)
    {
        if (sbs->n_rows != n_ray || sbs->n_cols != 3)
            sbs->set_size(n_ray, 3);

        dtype *px = sbs->colptr(0), *py = sbs->colptr(1), *pz = sbs->colptr(2);
        dtype *w = Ws.memptr();

        for (arma::uword i = 0; i < n_ray; ++i)
        {
            px[i] = ox[i] + w[i] * dx[i];
            py[i] = oy[i] + w[i] * dy[i];
            pz[i] = oz[i] + w[i] * dz[i];
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
                                                   arma::Mat<float> *fbs, arma::Mat<float> *sbs, arma::u32_vec *no_interact,
                                                   arma::u32_vec *fbs_ind, arma::u32_vec *sbs_ind,
                                                   const arma::u32_vec *sub_mesh_index, const arma::Mat<float> *aabb,
                                                   int use_kernel, int gpu_id);

template void quadriga_lib::ray_triangle_intersect(const arma::Mat<double> *orig, const arma::Mat<double> *dest, const arma::Mat<double> *mesh,
                                                   arma::Mat<double> *fbs, arma::Mat<double> *sbs, arma::u32_vec *no_interact,
                                                   arma::u32_vec *fbs_ind, arma::u32_vec *sbs_ind,
                                                   const arma::u32_vec *sub_mesh_index, const arma::Mat<double> *aabb,
                                                   int use_kernel, int gpu_id);