// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_site.hpp"
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
    const arma::Mat<dtype> &orig,
    const arma::Mat<dtype> &dest,
    const arma::Mat<dtype> &mesh,
    arma::Mat<dtype> *fbs = nullptr,
    arma::Mat<dtype> *sbs = nullptr,
    arma::u32_vec *n_interact = nullptr,
    arma::u32_vec *fbs_ind = nullptr,
    arma::u32_vec *sbs_ind = nullptr,
    const arma::u32_vec &sub_mesh_ind = arma::u32_vec(),
    const arma::Mat<dtype> &aabb = arma::Mat<dtype>(),
    int use_kernel = 0,
    int gpu_id = 0);
```

## Inputs:
- **`orig`** — Ray origins in GCS; `[n_ray, 3]`
- **`dest`** — Ray destinations in GCS; `[n_ray, 3]`
- **`mesh`** — Triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; further columns are ignored; `[n_mesh, 9]`
- **`sub_mesh_ind`** (optional) — Start indices of sub-meshes in `mesh`; enables AABB-accelerated traversal; `[n_sub]`
- **`aabb`** (optional) — Pre-computed axis-aligned bounding boxes per sub-mesh; each row: `{x_min x_max y_min y_max z_min z_max}`; if empty, AABBs are computed from `mesh`; `[n_sub, 6]`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable and for other values; auto mode selects CUDA when `n_ray >= 500` and CUDA is available, else AVX2, else GENERIC.
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA

## Outputs:
- **`fbs`** (optional) — First-bounce intersection points in GCS; `[n_ray, 3]`
- **`sbs`** (optional) — Second-bounce intersection points in GCS; `[n_ray, 3]`
- **`n_interact`** (optional) — Total number of intersections per ray between `orig` and `dest`; `[n_ray]`
- **`fbs_ind`** (optional) — 1-based index of first intersected mesh element; 0 = none; `[n_ray]`
- **`sbs_ind`** (optional) — 1-based index of second intersected mesh element; 0 = none; `[n_ray]`

## See also:
- [[obj_file_read]] (load mesh from OBJ file)
- [[triangle_mesh_segmentation]] (compute sub-mesh indices and AABBs)
- [[ray_point_intersect]] (beam interactions with sampling points)
- [[icosphere]] (generate ray beams)
- [[subdivide_rays]] (split ray beams into sub-beams)
MD!*/

template <typename dtype>
void quadriga_lib::ray_triangle_intersect(const arma::Mat<dtype> &orig, const arma::Mat<dtype> &dest, const arma::Mat<dtype> &mesh,
                                          arma::Mat<dtype> *fbs, arma::Mat<dtype> *sbs, arma::u32_vec *n_interact,
                                          arma::u32_vec *fbs_ind, arma::u32_vec *sbs_ind,
                                          const arma::u32_vec &sub_mesh_ind, const arma::Mat<dtype> &aabb,
                                          int use_kernel, int gpu_id)
{
    // Suppress unused-parameter warning when CUDA support is disabled at compile time
#if !BUILD_WITH_CUDA
    (void)gpu_id;
#endif

    // Input validation
    if (orig.n_elem == 0)
        throw std::invalid_argument("Inputs cannot be empty.");
    if (orig.n_cols < 3)
        throw std::invalid_argument("Input 'orig' must have at least 3 columns containing x,y,z coordinates.");

    arma::uword n_ray = orig.n_rows;

    if (dest.n_cols < 3)
        throw std::invalid_argument("Input 'dest' must have at least 3 columns containing x,y,z coordinates.");
    if (dest.n_rows != n_ray)
        throw std::invalid_argument("Number of elements in 'orig' and 'dest' do not match.");

    if (mesh.n_elem == 0)
        throw std::invalid_argument("Inputs cannot be empty.");
    if (mesh.n_cols < 9)
        throw std::invalid_argument("Input 'mesh' must have at least 9 columns containing x,y,z coordinates of 3 vertices.");

    arma::uword n_mesh = mesh.n_rows;

    qd_check_alias(fbs, "fbs", &orig, &dest, &mesh, &aabb);
    qd_check_alias(sbs, "sbs", &orig, &dest, &mesh, &aabb);
    qd_check_alias_outputs(fbs, "fbs", sbs, "sbs");
    qd_check_alias(n_interact, "n_interact", &sub_mesh_ind);
    qd_check_alias(fbs_ind, "fbs_ind", &sub_mesh_ind);
    qd_check_alias(sbs_ind, "sbs_ind", &sub_mesh_ind);
    qd_check_alias_outputs(n_interact, "n_interact", fbs_ind, "fbs_ind");
    qd_check_alias_outputs(n_interact, "n_interact", sbs_ind, "sbs_ind");
    qd_check_alias_outputs(fbs_ind, "fbs_ind", sbs_ind, "sbs_ind");

    const int kernel = qd_kernel_auto_cuda(use_kernel, n_ray);

    // Sub-mesh indices; without them the mesh is one sub-mesh
    arma::uword n_sub = qd_check_sub_ind(sub_mesh_ind, n_mesh, "sub_mesh_ind", "mesh face");
    const unsigned first_face = 0;
    const unsigned *p_smi = (n_sub == 0) ? &first_face : sub_mesh_ind.memptr();
    n_sub = (n_sub == 0) ? 1 : n_sub;

    // Compute or validate AABB
    qd_check_aabb(aabb, n_sub);
    const arma::Mat<dtype> aabb_computed = aabb.empty() ? quadriga_lib::triangle_mesh_aabb(mesh, sub_mesh_ind) : arma::Mat<dtype>();
    const arma::Mat<dtype> *p_aabb = aabb.empty() ? &aabb_computed : &aabb;

    // AABB data per sub-mesh
    const dtype *Xmin = p_aabb->colptr(0), *Xmax = p_aabb->colptr(1);
    const dtype *Ymin = p_aabb->colptr(2), *Ymax = p_aabb->colptr(3);
    const dtype *Zmin = p_aabb->colptr(4), *Zmax = p_aabb->colptr(5);

    // Mesh vertex pointers
    const dtype *Tx = mesh.colptr(0), *Ty = mesh.colptr(1), *Tz = mesh.colptr(2);
    const dtype *Ux = mesh.colptr(3), *Uy = mesh.colptr(4), *Uz = mesh.colptr(5);
    const dtype *Vx = mesh.colptr(6), *Vy = mesh.colptr(7), *Vz = mesh.colptr(8);

    const dtype *ox = orig.colptr(0), *oy = orig.colptr(1), *oz = orig.colptr(2);
    const dtype *dx = dest.colptr(0), *dy = dest.colptr(1), *dz = dest.colptr(2);

    // Kernel outputs, every element is written by the kernel
    arma::Col<dtype> Wf(n_ray), Ws(n_ray); // Normalized FBS and SBS hit distances
    arma::u32_vec If(n_ray), Is(n_ray);    // Index of mesh element hit at FBS/SBS, 1-based, 0 = no hit
    arma::u32_vec hit_cnt(n_ray);

    unsigned *p_hit_cnt = (n_interact == nullptr) ? nullptr : hit_cnt.memptr();

    if (kernel == 3) // CUDA
    {
#if BUILD_WITH_CUDA
        qd_RTI_CUDA<dtype>(Tx, Ty, Tz, Ux, Uy, Uz, Vx, Vy, Vz, n_mesh,
                           p_smi, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                           ox, oy, oz, dx, dy, dz,
                           n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt, gpu_id);
#endif
    }
    else if (kernel == 2) // AVX2
    {
#if BUILD_WITH_AVX2
        qd_RTI_AVX2<dtype>(Tx, Ty, Tz, Ux, Uy, Uz, Vx, Vy, Vz, n_mesh,
                           p_smi, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                           ox, oy, oz, dx, dy, dz,
                           n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
#endif
    }
    else // GENERIC
    {
        qd_RTI_GENERIC<dtype>(Tx, Ty, Tz, Ux, Uy, Uz, Vx, Vy, Vz, n_mesh,
                              p_smi, Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, n_sub,
                              ox, oy, oz, dx, dy, dz,
                              n_ray, Wf.memptr(), Ws.memptr(), If.memptr(), Is.memptr(), p_hit_cnt);
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
    if (n_interact != nullptr)
    {
        if (n_interact->n_elem != n_ray)
            n_interact->set_size(n_ray);
        std::memcpy(n_interact->memptr(), p_hit_cnt, no_bytes);
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

template void quadriga_lib::ray_triangle_intersect(const arma::Mat<float> &orig, const arma::Mat<float> &dest, const arma::Mat<float> &mesh,
                                                   arma::Mat<float> *fbs, arma::Mat<float> *sbs, arma::u32_vec *n_interact,
                                                   arma::u32_vec *fbs_ind, arma::u32_vec *sbs_ind,
                                                   const arma::u32_vec &sub_mesh_ind, const arma::Mat<float> &aabb,
                                                   int use_kernel, int gpu_id);

template void quadriga_lib::ray_triangle_intersect(const arma::Mat<double> &orig, const arma::Mat<double> &dest, const arma::Mat<double> &mesh,
                                                   arma::Mat<double> *fbs, arma::Mat<double> *sbs, arma::u32_vec *n_interact,
                                                   arma::u32_vec *fbs_ind, arma::u32_vec *sbs_ind,
                                                   const arma::u32_vec &sub_mesh_ind, const arma::Mat<double> &aabb,
                                                   int use_kernel, int gpu_id);
