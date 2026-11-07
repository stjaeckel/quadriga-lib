// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Ray intersections: ray_triangle_intersect (sub-meshes, AABBs, kernels, outputs aliasing the inputs),
// ray_point_intersect (sub-clouds, kernels, all outputs) and subdivide_rays (indices, pre-sized outputs)

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

// Start indices of sub-meshes or sub-clouds of n elements: ascending from 0, or fuzzed values
static arma::u32_vec starts(fz::input &in, arma::uword n)
{
    arma::u32_vec s(in.size(8));
    if (in.ConsumeBool())
    {
        unsigned v = 0;
        for (auto &x : s)
            x = v, v += in.ConsumeIntegralInRange<unsigned>(1, 4);
    }
    else
        for (auto &x : s)
            x = (unsigned)in.index(n);
    return s;
}

template <typename T>
static void run(fz::input &in)
{
    // Rays: fuzzed, or the beams of an icosphere from a fuzzed origin
    arma::uword n_ray = in.size(64);
    arma::Mat<T> orig, trivec, tridir;
    if (in.ConsumeBool())
    {
        arma::Mat<T> center;
        fz::clean([&] { quadriga_lib::icosphere<T>(in.ConsumeIntegralInRange<arma::uword>(1, 2), T(1), &center, nullptr, &trivec, &tridir, true); });
        n_ray = center.n_rows;
        orig = center;
        for (arma::uword c = 0; c < 3 && n_ray != 0; ++c)
            orig.col(c) += coord<T>(in);
    }
    else
    {
        orig = geo<T>(in, n_ray, near(in, 3));
        trivec = geo<T>(in, near(in, n_ray), near(in, 9));
        tridir = geo<T>(in, near(in, n_ray), near(in, 9));
    }
    arma::Mat<T> dest = geo<T>(in, near(in, n_ray), near(in, 3));

    arma::uword n_mesh = in.size(64);
    arma::Mat<T> mesh = geo<T>(in, n_mesh, near(in, 9));
    for (arma::uword i = 0; i < n_mesh && mesh.n_cols >= 9; ++i) // Some faces with two equal vertices
        if (in.ConsumeIntegralInRange<int>(0, 15) == 0)
            for (arma::uword c = 0; c < 3; ++c)
                mesh(i, 3 + c) = mesh(i, c);

    // Sub-meshes and AABBs: none, fuzzed, or computed by the library
    arma::u32_vec sub_mesh_index;
    arma::Mat<T> aabb;
    int smi_mode = in.ConsumeIntegralInRange<int>(0, 2), aabb_mode = in.ConsumeIntegralInRange<int>(0, 2);
    if (smi_mode == 1)
        sub_mesh_index = starts(in, n_mesh);
    if (smi_mode == 2)
        fz::clean([&]
                  { arma::Mat<T> meshR;
                    quadriga_lib::triangle_mesh_segmentation<T>(&mesh, &meshR, &sub_mesh_index, in.ConsumeIntegralInRange<arma::uword>(1, 16), in.ConsumeIntegralInRange<arma::uword>(1, 8));
                    mesh = meshR; });
    if (aabb_mode == 1)
        aabb = geo<T>(in, near(in, std::max(sub_mesh_index.n_elem, (arma::uword)1)), near(in, 6));
    if (aabb_mode == 2)
        fz::clean([&] { aabb = quadriga_lib::triangle_mesh_aabb<T>(&mesh, &sub_mesh_index, in.ConsumeIntegralInRange<arma::uword>(1, 8)); });

    // Outputs: missing, pre-sized wrongly, or the same objects as the inputs
    const int kernel = in.ConsumeIntegralInRange<int>(-1, 4);
    arma::Mat<T> fbs_own = geo<T>(in, in.size(3), in.size(4)), sbs_own;
    arma::u32_vec no_interact(in.size(2)), fbs_ind, sbs_ind;
    arma::Mat<T> orig_copy = orig, dest_copy = dest;
    arma::Mat<T> *fbs = nullptr, *sbs = nullptr;
    switch (in.ConsumeIntegralInRange<int>(0, 3))
    {
    case 1:
        fbs = &fbs_own, sbs = &sbs_own;
        break;
    case 2:
        fbs = &orig_copy, sbs = &dest_copy;
        break;
    case 3:
        fbs = &dest_copy, sbs = &orig_copy;
        break;
    }
    const bool all = in.ConsumeBool();
    fz::clean([&]
              { quadriga_lib::ray_triangle_intersect<T>(&orig_copy, &dest_copy, &mesh, fbs, sbs, all ? &no_interact : nullptr,
                                                        all ? &fbs_ind : nullptr, &sbs_ind, smi_mode ? &sub_mesh_index : nullptr,
                                                        aabb_mode ? &aabb : nullptr, kernel); });

    // Point cloud: fuzzed, or reorganized by the library
    arma::Mat<T> points = geo<T>(in, in.size(64), near(in, 3));
    arma::u32_vec sub_cloud_index;
    int sci_mode = in.ConsumeIntegralInRange<int>(0, 2);
    if (sci_mode == 1)
        sub_cloud_index = starts(in, points.n_rows);
    if (sci_mode == 2)
        fz::clean([&]
                  { arma::Mat<T> pointsR;
                    quadriga_lib::point_cloud_segmentation<T>(&points, &pointsR, &sub_cloud_index, in.ConsumeIntegralInRange<arma::uword>(1, 16), in.ConsumeIntegralInRange<arma::uword>(1, 8));
                    points = pointsR; });

    std::vector<unsigned> hit_index(in.size(3));
    arma::u32_vec hit_offset(in.size(3)), hit_count;
    std::vector<arma::u32_vec> hits_per_point(in.size(3));
    const int kernel_p = in.ConsumeIntegralInRange<int>(-1, 4), out_mask = in.ConsumeIntegralInRange<int>(0, 15);
    fz::clean([&]
              { quadriga_lib::ray_point_intersect<T>(points, orig, trivec, tridir, (out_mask & 1) ? &hit_index : nullptr,
                                                     (out_mask & 2) ? &hit_offset : nullptr, (out_mask & 4) ? &hit_count : nullptr,
                                                     (out_mask & 8) ? &hits_per_point : nullptr, sci_mode ? &sub_cloud_index : nullptr, kernel_p); });

    // Subdivision: spherical or Cartesian directions, fuzzed indices, outputs of fuzzed sizes
    arma::Mat<T> tridir_s = in.ConsumeBool() ? tridir : geo<T>(in, near(in, orig.n_rows), near(in, 6));
    arma::u32_vec index(in.size(16));
    for (auto &x : index)
        x = (unsigned)in.index(orig.n_rows);
    const bool transposed = in.ConsumeBool(), use_index = in.ConsumeBool(), use_dest = in.ConsumeBool();
    arma::Mat<T> origN(in.size(80), in.size(10)), trivecN(in.size(80), in.size(10)), tridirN, destN(in.size(80), in.size(10));
    if (in.ConsumeBool()) // Large enough and of the right size for the outputs to be reused
    {
        const arma::uword n = 4 * (use_index && index.n_elem ? index.n_elem : orig.n_rows) + in.size(2);
        origN.set_size(transposed ? 3 : n, transposed ? n : 3);
        trivecN.set_size(transposed ? 9 : n, transposed ? n : 9);
        tridirN.set_size(transposed ? tridir_s.n_cols : n, transposed ? n : tridir_s.n_cols);
        destN.set_size(transposed ? 3 : n, transposed ? n : 3);
    }
    fz::clean([&]
              { quadriga_lib::subdivide_rays<T>(orig, trivec, tridir_s, use_dest ? &dest : nullptr, &origN, &trivecN, &tridirN,
                                                &destN, use_index ? &index : nullptr, transposed); });
}

void fuzz_one(const uint8_t *data, size_t size)
{
    fz::input in(data, size);
    if (in.ConsumeBool())
        run<double>(in);
    else
        run<float>(in);
}
