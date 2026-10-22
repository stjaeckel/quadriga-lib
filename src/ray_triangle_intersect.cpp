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
# ray_triangle_intersect
Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm

- Counts the total number of intersections between `orig` and `dest`
- Computes the coordinates and object IDs of the first two intersections per ray (FBS/SBS)
- The AVX2 and CUDA kernels compute in single precision, GENERIC in the precision of the input; with `double` input,
  the AVX2 kernel forms the vectors between ray origin, vertices and bounding boxes in `double` before rounding them, so
  its precision depends on the size of the scene, not on its distance from the coordinate origin (e.g. UTM coordinates);
  the CUDA kernel rounds the coordinates themselves, so its errors grow with the distance from the origin
- Triangle edges and vertices count as hits; hits at the same distance (e.g. on a shared edge) are reported in the
  order of the mesh index, the lower index first (all kernels)

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

## Inputs:
- **`orig`** — Ray origins in GCS; `[n_ray, 3]`
- **`dest`** — Ray destinations in GCS; `[n_ray, 3]`
- **`mesh`** — Triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; further columns are ignored; `[n_mesh, 9]`
- **`sub_mesh_index`** (optional) — Start indices of sub-meshes in `mesh`; enables AABB-accelerated traversal; `[n_sub]`
- **`aabb`** (optional) — Pre-computed axis-aligned bounding boxes per sub-mesh; each row: `{x_min x_max y_min y_max z_min z_max}`; if `nullptr`, AABBs are computed from `mesh`; `[n_sub, 6]`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable and for other values; auto mode selects CUDA when `n_ray >= 500` and CUDA is available, else AVX2, else GENERIC.
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA

## Outputs:
- **`fbs`** (optional) — First-bounce intersection points in GCS; `[n_ray, 3]`
- **`sbs`** (optional) — Second-bounce intersection points in GCS; `[n_ray, 3]`
- **`no_interact`** (optional) — Total number of intersections per ray between `orig` and `dest`; `[n_ray]`
- **`fbs_ind`** (optional) — 1-based index of first intersected mesh element; 0 = none; `[n_ray]`
- **`sbs_ind`** (optional) — 1-based index of second intersected mesh element; 0 = none; `[n_ray]`
- `fbs` and `sbs` may be the same objects as `orig` or `dest` (e.g. to advance rays in place)

## See also:
- [[obj_file_read]] (load mesh from OBJ file)
- [[triangle_mesh_segmentation]] (compute sub-mesh indices and AABBs)
- [[ray_point_intersect]] (beam interactions with sampling points)
- [[icosphere]] (generate ray beams)
- [[subdivide_rays]] (split ray beams into sub-beams)
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
    if (dest->n_rows != n_ray)
        throw std::invalid_argument("Number of elements in 'orig' and 'dest' dont match.");

    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (mesh->n_elem == 0)
        throw std::invalid_argument("Inputs cannot be empty.");
    if (mesh->n_cols < 9)
        throw std::invalid_argument("Input 'mesh' must have at least 9 columns containing x,y,z coordinates of 3 vertices.");

    arma::uword n_mesh = mesh->n_rows;

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
    else if (use_kernel == 0) // Auto-select
    {
        if (n_ray >= 500 && quadriga_lib::quadriga_lib_has_CUDA())
            kernel = 3;
        else if (quadriga_lib::quadriga_lib_has_AVX2())
            kernel = 2;
        else
            kernel = 1;
    }
    else
        throw std::invalid_argument("Input 'use_kernel' must be 0, 1, 2 or 3.");

    // Check if the sub-mesh indices are valid
    arma::uword n_sub = 1ULL;                                     // Number of sub-meshes (at least 1)
    arma::u32_vec smi(1, arma::fill::zeros);                      // Sub-mesh-index (local copy)
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

    // Mesh vertex pointers
    const dtype *Tx = mesh->colptr(0), *Ty = mesh->colptr(1), *Tz = mesh->colptr(2);
    const dtype *Ux = mesh->colptr(3), *Uy = mesh->colptr(4), *Uz = mesh->colptr(5);
    const dtype *Vx = mesh->colptr(6), *Vy = mesh->colptr(7), *Vz = mesh->colptr(8);

    // Pointer to origin and destination coordinates
    const dtype *ox = orig->colptr(0), *oy = orig->colptr(1), *oz = orig->colptr(2);
    const dtype *dx = dest->colptr(0), *dy = dest->colptr(1), *dz = dest->colptr(2);

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
        qd_RTI_CUDA<dtype>(Tx, Ty, Tz, Ux, Uy, Uz, Vx, Vy, Vz, n_mesh,
                           smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                           ox, oy, oz, dx, dy, dz,
                           n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt, gpu_id);
#endif
    }
    else if (kernel == 2) // AVX2
    {
#if BUILD_WITH_AVX2
        qd_RTI_AVX2<dtype>(Tx, Ty, Tz, Ux, Uy, Uz, Vx, Vy, Vz, n_mesh,
                           smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                           ox, oy, oz, dx, dy, dz,
                           n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
#endif
    }
    else // GENERIC
    {
        qd_RTI_GENERIC<dtype>(Tx, Ty, Tz, Ux, Uy, Uz, Vx, Vy, Vz, n_mesh,
                              smi.memptr(), Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                              ox, oy, oz, dx, dy, dz,
                              n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
    }

    // An output may be the same object as 'orig' or 'dest' (e.g. when rays are advanced in place). Both outputs
    // need the original coordinates, and resizing an output would free the input's memory, so the coordinates
    // are copied before the first output is written.
    arma::Mat<dtype> od_copy;
    if ((fbs != nullptr && (fbs == orig || fbs == dest)) || (sbs != nullptr && (sbs == orig || sbs == dest)))
    {
        od_copy.set_size(n_ray, 6);
        std::memcpy(od_copy.colptr(0), ox, n_ray * sizeof(dtype));
        std::memcpy(od_copy.colptr(1), oy, n_ray * sizeof(dtype));
        std::memcpy(od_copy.colptr(2), oz, n_ray * sizeof(dtype));
        std::memcpy(od_copy.colptr(3), dx, n_ray * sizeof(dtype));
        std::memcpy(od_copy.colptr(4), dy, n_ray * sizeof(dtype));
        std::memcpy(od_copy.colptr(5), dz, n_ray * sizeof(dtype));
        ox = od_copy.colptr(0), oy = od_copy.colptr(1), oz = od_copy.colptr(2);
        dx = od_copy.colptr(3), dy = od_copy.colptr(4), dz = od_copy.colptr(5);
    }

    // Compute FBS location in GCS
    if (fbs != nullptr)
    {
        if (fbs->n_rows != n_ray || fbs->n_cols != 3)
            fbs->set_size(n_ray, 3);

        dtype *px = fbs->colptr(0), *py = fbs->colptr(1), *pz = fbs->colptr(2);
        dtype *w = Wf.memptr();

        for (arma::uword i = 0; i < n_ray; ++i)
        {
            px[i] = ox[i] + w[i] * (dx[i] - ox[i]);
            py[i] = oy[i] + w[i] * (dy[i] - oy[i]);
            pz[i] = oz[i] + w[i] * (dz[i] - oz[i]);
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
            px[i] = ox[i] + w[i] * (dx[i] - ox[i]);
            py[i] = oy[i] + w[i] * (dy[i] - oy[i]);
            pz[i] = oz[i] + w[i] * (dz[i] - oz[i]);
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