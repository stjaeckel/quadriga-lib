// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Triangle meshes: point_inside_mesh, obj_overlap_test, subdivide_triangles, triangle_mesh_segmentation,
// triangle_mesh_aabb, triangle_mesh_split, and the mesh generators icosphere, cube and plane

#include "fuzz_target.hpp"

void fuzz_seeds(const fz::seeds &) {}

// A coordinate: often a small integer (shared vertices, coplanar and degenerate faces), else a fuzzed value
template <typename T>
static T coord(fz::input &in)
{
    return in.ConsumeBool() ? T(in.ConsumeIntegralInRange<int>(-2, 2)) : in.value<T>();
}

template <typename T>
static arma::Mat<T> geo(fz::input &in, arma::uword n_rows, arma::uword n_cols)
{
    arma::Mat<T> x(n_rows, n_cols);
    for (auto &v : x)
        v = coord<T>(in);
    return x;
}

// A size that mostly matches n: n, sometimes 0, n - 1 or n + 1
static arma::uword near(fz::input &in, arma::uword n)
{
    switch (in.ConsumeIntegralInRange<int>(0, 7))
    {
    case 0:
        return 0;
    case 1:
        return n == 0 ? 0 : n - 1;
    case 2:
        return n + 1;
    default:
        return n;
    }
}

// A subdivision count: small, or so large that the output size overflows 64 bits
static arma::uword n_div(fz::input &in)
{
    const arma::uword big[] = {1ULL << 32, (1ULL << 32) + 1, 1ULL << 62, std::numeric_limits<arma::uword>::max()};
    return in.ConsumeIntegralInRange<int>(0, 15) == 0 ? big[in.ConsumeIntegralInRange<int>(0, 3)] : in.size(3);
}

// Object or material indices: few distinct small values, or fuzzed ones
static arma::uvec indices(fz::input &in, arma::uword n)
{
    arma::uvec x(n);
    const bool small = in.ConsumeBool();
    for (auto &v : x)
        v = small ? in.ConsumeIntegralInRange<arma::uword>(0, 3) : in.index(n);
    return x;
}

// A vector of length 0 to 4 for the scale, rotation or location of cube and plane
static arma::vec param(fz::input &in)
{
    arma::vec x(in.size(4));
    for (auto &v : x)
        v = in.value<double>();
    return x;
}

template <typename T>
static void run(fz::input &in)
{
    // Mesh: fuzzed, or a generated cube or plane (closed and open surfaces)
    arma::Mat<T> mesh;
    const int mesh_mode = in.ConsumeIntegralInRange<int>(0, 3);
    if (mesh_mode < 2)
    {
        const arma::vec scale = param(in), rotation = param(in), location = param(in);
        const arma::uword n = n_div(in);
        if (mesh_mode == 0)
            fz::clean([&] { mesh = quadriga_lib::cube<T>(scale, rotation, location, n); });
        else
            fz::clean([&] { mesh = quadriga_lib::plane<T>(scale, rotation, location, n); });
    }
    else
        mesh = geo<T>(in, in.size(64), near(in, 9));
    if (mesh.n_rows > 64) // Keep the cost per input small
        mesh = mesh.rows(0, 63);
    const arma::uword n_mesh = mesh.n_rows;

    // Points inside, with and without objects; a distance above 3 m adds many rays, so it is rare
    arma::Mat<T> points = geo<T>(in, in.size(16), near(in, 3));
    arma::uvec obj_ind = indices(in, near(in, n_mesh));
    T distance = in.ConsumeIntegralInRange<int>(0, 7) == 0 ? in.value<T>() : in.ConsumeFloatingPointInRange<T>(T(0), T(3));
    const bool use_obj = in.ConsumeBool();
    fz::clean([&] { quadriga_lib::point_inside_mesh<T>(&points, &mesh, use_obj ? &obj_ind : nullptr, distance); });

    // Overlap of objects
    std::vector<std::string> reason(in.size(2));
    const T tolerance = in.value<T>();
    const bool use_reason = in.ConsumeBool();
    fz::clean([&] { quadriga_lib::obj_overlap_test<T>(&mesh, &obj_ind, use_reason ? &reason : nullptr, tolerance); });

    // Subdivision, also in place
    arma::uvec mtl_ind = indices(in, near(in, n_mesh)), mtl_ind_out(in.size(3));
    const arma::uword n_div_t = in.ConsumeIntegralInRange<int>(0, 15) == 0 ? n_div(in) : in.size(2);
    const bool in_place = in.ConsumeBool(), use_mtl = in.ConsumeBool();
    arma::Mat<T> sub = mesh, sub_out(in.size(3), in.size(10));
    arma::uvec mtl_sub = mtl_ind;
    fz::clean([&]
              { quadriga_lib::subdivide_triangles<T>(n_div_t, &sub, in_place ? &sub : &sub_out, use_mtl ? &mtl_sub : nullptr,
                                                     in_place ? &mtl_sub : &mtl_ind_out); });

    // Segmentation; vec_size is small or the largest value
    arma::Mat<T> meshR;
    arma::u32_vec sub_mesh_index, mesh_index;
    arma::uvec mtl_ind_seg;
    const arma::uword target_size = in.size(70);
    const arma::uword vec_size = in.ConsumeIntegralInRange<int>(0, 15) == 0 ? std::numeric_limits<arma::uword>::max() : in.size(33);
    const int seg_mask = in.ConsumeIntegralInRange<int>(0, 7);
    fz::clean([&]
              { quadriga_lib::triangle_mesh_segmentation<T>(&mesh, (seg_mask & 1) ? &meshR : nullptr, &sub_mesh_index, target_size, vec_size,
                                                            (seg_mask & 2) ? &mtl_ind : nullptr, (seg_mask & 2) ? &mtl_ind_seg : nullptr,
                                                            (seg_mask & 4) ? &mesh_index : nullptr); });

    // AABB of fuzzed or segmented sub-meshes
    arma::u32_vec smi(in.size(8));
    for (auto &x : smi)
        x = (unsigned)in.index(n_mesh);
    if (in.ConsumeBool())
        smi = arma::sort(smi);
    const arma::Mat<T> &aabb_mesh = (in.ConsumeBool() && meshR.n_rows != 0) ? meshR : mesh;
    const arma::u32_vec *aabb_smi = in.ConsumeBool() ? &smi : (in.ConsumeBool() ? &sub_mesh_index : nullptr);
    const arma::uword vec_size_aabb = in.size(9);
    fz::clean([&] { quadriga_lib::triangle_mesh_aabb<T>(&aabb_mesh, aabb_smi, vec_size_aabb); });

    // Split along each axis selector, also invalid ones; the outputs may be distinct, the same object, or the input
    arma::Mat<T> split_mesh = mesh, meshA(in.size(2), 9), meshB;
    arma::Col<int> split_ind(in.size(2));
    arma::Mat<T> *outA = &meshA, *outB = &meshB;
    switch (in.ConsumeIntegralInRange<int>(0, 7))
    {
    case 0:
        outB = &meshA;
        break;
    case 1:
        outA = &split_mesh;
        break;
    case 2:
        outB = &split_mesh;
        break;
    }
    const int axis = in.ConsumeIntegralInRange<int>(-1, 4);
    const bool use_split_ind = in.ConsumeBool();
    fz::clean([&] { quadriga_lib::triangle_mesh_split<T>(&split_mesh, outA, outB, axis, use_split_ind ? &split_ind : nullptr); });

    // Icosphere with all output combinations
    arma::Mat<T> center, vert, direction;
    arma::Col<T> length;
    const arma::uword n_div_i = n_div(in);
    const T radius = in.value<T>();
    const int ico_mask = in.ConsumeIntegralInRange<int>(0, 15);
    fz::clean([&]
              { quadriga_lib::icosphere<T>(n_div_i, radius, (ico_mask & 1) ? &center : nullptr, (ico_mask & 2) ? &length : nullptr,
                                           (ico_mask & 4) ? &vert : nullptr, (ico_mask & 8) ? &direction : nullptr, in.ConsumeBool()); });
}

void fuzz_one(const uint8_t *data, size_t size)
{
    fz::input in(data, size);
    if (in.ConsumeBool())
        run<double>(in);
    else
        run<float>(in);
}
