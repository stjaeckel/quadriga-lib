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
- Supports a compact input format where origin and destination coordinates are stored together.
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
                bool transpose_inputs = false,
                int use_kernel = 0,
                int gpu_id = 0);
```

## Arguments:
- `const arma::Mat<dtype> ***orig**` (input)<br>
  Ray origins in global coordinate system (GCS). Size: `[n_ray, 3]`, or `[n_ray, 6]` if `dest == nullptr`.
  In the latter case, columns must be ordered as `{xo, yo, zo, xd, yd, zd}`.

- `const arma::Mat<dtype> ***dest**` (input)<br>
  Ray destinations in GCS. Size: `[n_ray, 3]`. Set to `nullptr` if `orig` has 6 columns and contains
  both origin and destination.

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
  When using AVX2, sub-mesh start indices must be aligned to multiples of 8.

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
                                          const arma::u32_vec *sub_mesh_index, bool transpose_inputs,
                                          int use_kernel, int gpu_id)
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

    // Convert orig and dest to float and calculate (dest - orig)
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

    // Determine SIMD vector size based on selected kernel
    arma::uword vec_size = (kernel == 2) ? 8ULL : 1ULL;

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

            if (vec_size > 1ULL && p_sub[i] % vec_size != 0ULL)
                throw std::invalid_argument("Sub-meshes must be aligned with the SIMD vector size (8 for AVX2).");
        }

        if (p_sub[n_sub - 1ULL] >= (unsigned)n_mesh)
            throw std::invalid_argument("Sub-mesh indices cannot exceed number of mesh elements.");

        smi = *sub_mesh_index;
    }

    // Pad mesh and sub-mesh counts to multiples of vec_size for SIMD processing
    arma::uword n_mesh_s = (n_mesh % vec_size == 0ULL) ? n_mesh : vec_size * (n_mesh / vec_size + 1ULL);
    arma::uword n_sub_s = (n_sub % vec_size == 0ULL) ? n_sub : vec_size * (n_sub / vec_size + 1ULL);

    // Mesh data in SoA layout (first vertex + two edges)
    arma::fvec Tx_vec(n_mesh_s, arma::fill::none), Ty_vec(n_mesh_s, arma::fill::none), Tz_vec(n_mesh_s, arma::fill::none);
    arma::fvec E1x_vec(n_mesh_s, arma::fill::none), E1y_vec(n_mesh_s, arma::fill::none), E1z_vec(n_mesh_s, arma::fill::none);
    arma::fvec E2x_vec(n_mesh_s, arma::fill::none), E2y_vec(n_mesh_s, arma::fill::none), E2z_vec(n_mesh_s, arma::fill::none);
    float *Tx = Tx_vec.memptr(), *Ty = Ty_vec.memptr(), *Tz = Tz_vec.memptr();
    float *E1x = E1x_vec.memptr(), *E1y = E1y_vec.memptr(), *E1z = E1z_vec.memptr();
    float *E2x = E2x_vec.memptr(), *E2y = E2y_vec.memptr(), *E2z = E2z_vec.memptr();

    // AABB data per sub-mesh
    arma::fvec Xmin_vec(n_sub_s, arma::fill::none), Xmax_vec(n_sub_s, arma::fill::none);
    arma::fvec Ymin_vec(n_sub_s, arma::fill::none), Ymax_vec(n_sub_s, arma::fill::none);
    arma::fvec Zmin_vec(n_sub_s, arma::fill::none), Zmax_vec(n_sub_s, arma::fill::none);
    float *Xmin = Xmin_vec.memptr(), *Xmax = Xmax_vec.memptr();
    float *Ymin = Ymin_vec.memptr(), *Ymax = Ymax_vec.memptr();
    float *Zmin = Zmin_vec.memptr(), *Zmax = Zmax_vec.memptr();

    // Convert mesh to float and write to SoA layout
    // Calculate bounding box for each sub-meshes
    const dtype *p_mesh = mesh->memptr();
    const unsigned *p_sub = smi.memptr();

    // Set parameters for the first AABB
    arma::uword i_sub = 0ULL, i_next = (n_sub == 1ULL) ? n_mesh - 1ULL : (arma::uword)p_sub[1] - 1ULL;
    float x_min = INFINITY, x_max = -INFINITY,
          y_min = INFINITY, y_max = -INFINITY,
          z_min = INFINITY, z_max = -INFINITY;

    // Lambda to process each mesh element and write the data to SoA buffers
    auto process_mesh_element = [&](arma::uword i_mesh, dtype x1, dtype y1, dtype z1, dtype x2, dtype y2, dtype z2, dtype x3, dtype y3, dtype z3)
    {
        // Typecast to float and update AABB
        float xf = (float)x1, yf = (float)y1, zf = (float)z1;
        x_min = (xf < x_min) ? xf : x_min, x_max = (xf > x_max) ? xf : x_max;
        y_min = (yf < y_min) ? yf : y_min, y_max = (yf > y_max) ? yf : y_max;
        z_min = (zf < z_min) ? zf : z_min, z_max = (zf > z_max) ? zf : z_max;

        // Write to SoA buffers
        Tx[i_mesh] = xf, Ty[i_mesh] = yf, Tz[i_mesh] = zf;

        // Typecast to float and update AABB
        xf = (float)x2, yf = (float)y2, zf = (float)z2;
        x_min = (xf < x_min) ? xf : x_min, x_max = (xf > x_max) ? xf : x_max;
        y_min = (yf < y_min) ? yf : y_min, y_max = (yf > y_max) ? yf : y_max;
        z_min = (zf < z_min) ? zf : z_min, z_max = (zf > z_max) ? zf : z_max;

        // Calculate edge and write to SoA buffers
        E1x[i_mesh] = float(x2 - x1);
        E1y[i_mesh] = float(y2 - y1);
        E1z[i_mesh] = float(z2 - z1);

        // Typecast to float and update AABB
        xf = (float)x3, yf = (float)y3, zf = (float)z3;
        x_min = (xf < x_min) ? xf : x_min, x_max = (xf > x_max) ? xf : x_max;
        y_min = (yf < y_min) ? yf : y_min, y_max = (yf > y_max) ? yf : y_max;
        z_min = (zf < z_min) ? zf : z_min, z_max = (zf > z_max) ? zf : z_max;

        // Calculate edge and write to SoA buffers
        E2x[i_mesh] = float(x3 - x1);
        E2y[i_mesh] = float(y3 - y1);
        E2z[i_mesh] = float(z3 - z1);

        // Update sub-mesh data for the next AABB
        if (i_mesh == i_next)
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

    // Add padding to the mesh data
    for (arma::uword i_mesh = n_mesh; i_mesh < n_mesh_s; ++i_mesh)
    {
        Tx[i_mesh] = 0.0f, Ty[i_mesh] = 0.0f, Tz[i_mesh] = 0.0f,
        E1x[i_mesh] = 0.0f, E1y[i_mesh] = 0.0f, E1z[i_mesh] = 0.0f,
        E2x[i_mesh] = 0.0f, E2y[i_mesh] = 0.0f, E2z[i_mesh] = 0.0f;
    }

    // Add padding to the AABB data
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

    // Dispatch to selected kernel
    if (kernel == 3) // CUDA
    {
#if BUILD_WITH_CUDA
        qd_RTI_CUDA(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh_s,
                    smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                    origA.colptr(0), origA.colptr(1), origA.colptr(2),
                    dest_minus_origA.colptr(0), dest_minus_origA.colptr(1), dest_minus_origA.colptr(2),
                    n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt, gpu_id);
#endif
    }
    else if (kernel == 2) // AVX2
    {
#if BUILD_WITH_AVX2
        qd_RTI_AVX2(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh_s,
                    smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                    origA.colptr(0), origA.colptr(1), origA.colptr(2),
                    dest_minus_origA.colptr(0), dest_minus_origA.colptr(1), dest_minus_origA.colptr(2),
                    n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
#endif
    }
    else // GENERIC
    {
        qd_RTI_GENERIC(Tx, Ty, Tz, E1x, E1y, E1z, E2x, E2y, E2z, n_mesh,
                       smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                       origA.colptr(0), origA.colptr(1), origA.colptr(2),
                       dest_minus_origA.colptr(0), dest_minus_origA.colptr(1), dest_minus_origA.colptr(2),
                       n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
    }

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
                                                   arma::Mat<float> *fbs, arma::Mat<float> *sbs, arma::u32_vec *no_interact,
                                                   arma::u32_vec *fbs_ind, arma::u32_vec *sbs_ind,
                                                   const arma::u32_vec *sub_mesh_index, bool transpose_inputs,
                                                   int use_kernel, int gpu_id);

template void quadriga_lib::ray_triangle_intersect(const arma::Mat<double> *orig, const arma::Mat<double> *dest, const arma::Mat<double> *mesh,
                                                   arma::Mat<double> *fbs, arma::Mat<double> *sbs, arma::u32_vec *no_interact,
                                                   arma::u32_vec *fbs_ind, arma::u32_vec *sbs_ind,
                                                   const arma::u32_vec *sub_mesh_index, bool transpose_inputs,
                                                   int use_kernel, int gpu_id);