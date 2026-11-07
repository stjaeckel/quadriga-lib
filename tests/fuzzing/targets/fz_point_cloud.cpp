// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Point clouds: point_cloud_segmentation (target and vector sizes, optional outputs), point_cloud_aabb (fuzzed and
// segmented sub-clouds) and point_cloud_split (all axis selectors, outputs that are the same objects)

#include "fuzz_target.hpp"

void fuzz_seeds(const fz::seeds &) {}

// A coordinate: often a small integer (equal and collinear points), else a fuzzed value
template <typename T>
static T coord(fz::input &in)
{
    return in.ConsumeBool() ? T(in.ConsumeIntegralInRange<int>(-2, 2)) : in.value<T>();
}

template <typename T>
static void run(fz::input &in)
{
    // Points with 3 columns, sometimes 2 or 4
    const arma::uword n_cols = in.ConsumeIntegralInRange<int>(0, 7) == 0 ? in.ConsumeIntegralInRange<arma::uword>(2, 4) : 3;
    arma::Mat<T> points(in.size(64), n_cols);
    for (auto &v : points)
        v = coord<T>(in);
    const arma::uword n_points = points.n_rows;

    // Segmentation; vec_size is small or the largest value
    arma::Mat<T> pointsR(in.size(3), 3);
    arma::u32_vec sub_cloud_index, forward_index(in.size(3)), reverse_index;
    const arma::uword target_size = in.size(70);
    const arma::uword vec_size = in.ConsumeIntegralInRange<int>(0, 15) == 0 ? std::numeric_limits<arma::uword>::max() : in.size(33);
    const int seg_mask = in.ConsumeIntegralInRange<int>(0, 15);
    fz::clean([&]
              { quadriga_lib::point_cloud_segmentation<T>(&points, (seg_mask & 1) ? &pointsR : nullptr, (seg_mask & 2) ? &sub_cloud_index : nullptr,
                                                          target_size, vec_size, (seg_mask & 4) ? &forward_index : nullptr,
                                                          (seg_mask & 8) ? &reverse_index : nullptr); });

    // AABB of fuzzed or segmented sub-clouds
    arma::u32_vec sci(in.size(8));
    for (auto &x : sci)
        x = (unsigned)in.index(n_points);
    if (in.ConsumeBool())
        sci = arma::sort(sci);
    const arma::Mat<T> &aabb_points = (in.ConsumeBool() && pointsR.n_rows != 0) ? pointsR : points;
    const int aabb_mode = in.ConsumeIntegralInRange<int>(0, 2);
    const arma::u32_vec *aabb_sci = aabb_mode == 0 ? nullptr : (aabb_mode == 1 ? &sci : &sub_cloud_index);
    const arma::uword vec_size_aabb = in.size(9);
    fz::clean([&] { quadriga_lib::point_cloud_aabb<T>(&aabb_points, aabb_sci, vec_size_aabb); });

    // Split; the outputs may be distinct, the same object, or the input itself
    arma::Mat<T> copy = points, pointsA(in.size(2), 3), pointsB;
    arma::Col<int> split_ind(in.size(2));
    arma::Mat<T> *outA = &pointsA, *outB = &pointsB;
    switch (in.ConsumeIntegralInRange<int>(0, 7))
    {
    case 0:
        outB = &pointsA;
        break;
    case 1:
        outA = &copy;
        break;
    case 2:
        outB = &copy;
        break;
    }
    const int axis = in.ConsumeIntegralInRange<int>(-1, 4);
    const bool use_split_ind = in.ConsumeBool();
    fz::clean([&] { quadriga_lib::point_cloud_split<T>(&copy, outA, outB, axis, use_split_ind ? &split_ind : nullptr); });
}

void fuzz_one(const uint8_t *data, size_t size)
{
    fz::input in(data, size);
    if (in.ConsumeBool())
        run<double>(in);
    else
        run<float>(in);
}
