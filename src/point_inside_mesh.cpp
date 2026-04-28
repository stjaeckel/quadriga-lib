// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "quadriga_lib_helper_functions.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# point_inside_mesh
Test whether 3D points are inside a triangle mesh using raycasting

- Always casts 4 rays per point in near-tetrahedral directions (rotated regular tetrahedron,
  scaled to 1000 m) for inside/outside detection
- When `distance > 0`, adds icosphere-sampled rays at subdivision level ⌈distance⌉ + 1
  (e.g. subdiv 2 for distance ≤ 1 m, subdiv 3 for ≤ 2 m), substantially increasing ray count
- A point is inside if any ray hits a face with a negative incidence angle, or if the ray
  thickness at FBS is below 1 mm (surface proximity)
- Mesh must be watertight with all normals pointing outward
- If `obj_ind` is provided, returns the 1-based enclosing object index instead of binary 0/1

## Declaration:
```
arma::u32_vec quadriga_lib::point_inside_mesh(
    const arma::Mat<dtype> *points,
    const arma::Mat<dtype> *mesh,
    const arma::u32_vec *obj_ind = nullptr,
    dtype distance = 0.0);
```

## Inputs:
- **`points`** — 3D coordinates of test points; `[n_points, 3]`
- **`mesh`** — Triangle faces in row-major vertex format  `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`obj_ind`** *(optional)* — 1-based object index per mesh element; enables per-object output; `[n_mesh]`
- **`distance`** *(optional)* — Surface proximity threshold; points within this distance
  of the mesh surface are classified as inside; increases ray count to 4 + N_icosphere(⌈distance⌉ + 1);
  range: 0–20 m (default: 0)

## Returns:
- `arma::u32_vec`, size `[n_points]`; `0` = outside, `1` = inside any object (no `obj_ind`), or 1-based object index (with `obj_ind`)

## See also:
- [[obj_file_read]] (for reading `mesh` and `obj_ind` from an .obj file)
MD!*/

template <typename dtype>
arma::u32_vec quadriga_lib::point_inside_mesh(const arma::Mat<dtype> *points,
                                              const arma::Mat<dtype> *mesh,
                                              const arma::u32_vec *obj_ind,
                                              dtype distance)
{
    if (points == nullptr || points->n_cols != 3)
        throw std::invalid_argument("Input 'points' must have 3 columns.");

    if (mesh == nullptr || mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns.");

    if (obj_ind != nullptr && obj_ind->n_elem != mesh->n_rows && obj_ind->n_elem != 0)
        throw std::invalid_argument("Number of elements in 'obj_ind' does not match number of rows in 'mesh'.");

    if (distance < 0.0 || distance > 20.0)
        throw std::invalid_argument("Distance must be in between 0 and 20 meters.");

    if (points->n_rows == 0)
        return arma::u32_vec();

    arma::uword n_points = points->n_rows;
    arma::uword n_mesh = mesh->n_rows;

    // Generate 4 rays for "inside" detection [x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3 ]
    dtype dir[12] = {1.0, -0.5, -0.25, -0.25, 0.0, 0.866, -0.433, -0.433, 0.0, 0.0, 0.866, -0.866};
    qd_rotate_inplace<dtype>(0.1745f, 0.3491f, 0.5236f, dir, 4, true);
    qd_multiply_scalar<dtype>(1000.0f, dir, 12);
    arma::uword n_cast = 4;

    // Generate additional rays for distance check
    arma::Mat<dtype> direction;
    if (distance > 0.0)
    {
        arma::Mat<dtype> tmp;
        auto n_div = (arma::uword)std::ceil(distance) + 1;
        quadriga_lib::icosphere<dtype>(n_div, 1000.0, &tmp);
        n_cast += tmp.n_rows;

        direction.set_size(n_cast, 3);
        size_t n_bytes = 4 * sizeof(dtype);
        std::memcpy(direction.colptr(0), dir, n_bytes);
        std::memcpy(direction.colptr(1), &dir[4], n_bytes);
        std::memcpy(direction.colptr(2), &dir[8], n_bytes);

        n_bytes = tmp.n_rows * sizeof(dtype);
        std::memcpy(direction.colptr(0) + 4, tmp.colptr(0), n_bytes);
        std::memcpy(direction.colptr(1) + 4, tmp.colptr(1), n_bytes);
        std::memcpy(direction.colptr(2) + 4, tmp.colptr(2), n_bytes);
    }

    // Generate ray origin and destination points
    arma::Mat<dtype> orig(n_points * n_cast, 3, arma::fill::none);
    arma::Mat<dtype> dest(n_points * n_cast, 3, arma::fill::none);
    dtype *p_orig = orig.memptr();
    dtype *p_dest = dest.memptr();

    qd_repeat_sequence(points->memptr(), n_points * 3, n_cast, 1, p_orig);

    if (direction.n_elem != 0)
    {
        qd_repeat_sequence(direction.colptr(0), n_cast, 1, n_points, dest.colptr(0));
        qd_repeat_sequence(direction.colptr(1), n_cast, 1, n_points, dest.colptr(1));
        qd_repeat_sequence(direction.colptr(2), n_cast, 1, n_points, dest.colptr(2));

        dtype scale = distance * dtype(0.001);
        for (arma::uword i = 0; i < orig.n_elem; ++i)
        {
            dtype orig_tmp = (i % n_cast < 4) ? p_orig[i] : p_dest[i] * scale + p_orig[i];
            p_dest[i] += p_orig[i];
            p_orig[i] = orig_tmp;
        }
    }
    else
    {
        qd_repeat_sequence(dir, n_cast, 1, n_points, dest.colptr(0));
        qd_repeat_sequence(&dir[4], n_cast, 1, n_points, dest.colptr(1));
        qd_repeat_sequence(&dir[8], n_cast, 1, n_points, dest.colptr(2));

        for (arma::uword i = 0; i < orig.n_elem; ++i)
            p_dest[i] += p_orig[i];
    }

    // Calculate BVH tree
    arma::Mat<dtype> meshR;
    arma::u32_vec sub_mesh_index(1);
    size_t target_size = 10 * (size_t)std::ceil(std::sqrt((double)n_mesh));
    target_size = (target_size < 1024) ? 1024 : target_size;
    quadriga_lib::triangle_mesh_segmentation(mesh, &meshR, &sub_mesh_index, target_size, 8);

    // Calculate intersections
    arma::Mat<dtype> fbs, sbs;
    arma::u32_vec no_interact, fbs_ind, sbs_ind;
    ray_triangle_intersect(&orig, &dest, &meshR, &fbs, &sbs, &no_interact, &fbs_ind, &sbs_ind, &sub_mesh_index);

    // Calculate the incidence angles
    arma::Mat<dtype> mtl_prop(meshR.n_rows, 9, arma::fill::zeros);
    arma::Col<dtype> fbs_angle, thickness;
    quadriga_lib::ray_mesh_interact<dtype>(0, 1.0e9, &orig, &dest, &fbs, &sbs, &meshR, &mtl_prop, &fbs_ind, &sbs_ind,
                                           nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, &fbs_angle, &thickness);

    // Check inside condition
    arma::u32_vec output(n_points);
    unsigned *p_out = output.memptr();
    unsigned *p_fbs_ind = fbs_ind.memptr();
    const unsigned *p_obj_ind = (obj_ind == nullptr || obj_ind->n_elem == 0) ? nullptr : obj_ind->memptr();
    dtype *p_fbs_angle = fbs_angle.memptr();
    dtype *p_thickness = thickness.memptr();

    arma::uword i_ang = 0;
    for (arma::uword i_point = 0; i_point < n_points; ++i_point)
        for (arma::uword i_cast = 0; i_cast < n_cast; ++i_cast)
        {
            arma::uword i_ray = i_point * n_cast + i_cast;
            if (p_fbs_ind[i_ray] == 0) // No interaction with the mesh
                continue;
            if (p_out[i_point] == 0 && (p_fbs_angle[i_ang] < 0.0 || p_thickness[i_ang] < 0.001))
            {
                if (p_obj_ind) // Return object index
                    p_out[i_point] = p_obj_ind[p_fbs_ind[i_ray] - 1];
                else
                    p_out[i_point] = 1;
            }
            ++i_ang;
        }

    return output;
}

template arma::u32_vec quadriga_lib::point_inside_mesh(const arma::Mat<float> *points,
                                                       const arma::Mat<float> *mesh,
                                                       const arma::u32_vec *obj_ind,
                                                       float distance);

template arma::u32_vec quadriga_lib::point_inside_mesh(const arma::Mat<double> *points,
                                                       const arma::Mat<double> *mesh,
                                                       const arma::u32_vec *obj_ind,
                                                       double distance);