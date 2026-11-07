// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Path geometry: coord2path (both directions, all outputs), combine_irs_coord (reversed segments, path subsets),
// path_to_tube (radius, edge count, turns) and generate_diffraction_paths (all levels of detail)

#include "fuzz_target.hpp"

void fuzz_seeds(const fz::seeds &) {}

// A coordinate: often a small integer (co-located points, straight lines and turns), else a fuzzed value
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

// Interaction counts of up to 16 paths: mostly 0 to 4, sometimes any 32-bit value (sums that wrap in 32 bits)
static arma::u32_vec counts(fz::input &in)
{
    arma::u32_vec x(in.size(16));
    for (auto &v : x)
        v = in.ConsumeIntegralInRange<int>(0, 15) == 0 ? in.ConsumeIntegral<unsigned>() : in.ConsumeIntegralInRange<unsigned>(0, 4);
    return x;
}

// Interaction coordinates for the counts: [3, sum], or near that size; never more than 256 columns
template <typename T>
static arma::Mat<T> interactions(fz::input &in, const arma::u32_vec &no_interact)
{
    arma::uword sum = 0;
    for (auto v : no_interact)
        sum += v;
    return geo<T>(in, near(in, 3), near(in, std::min(sum, (arma::uword)256)));
}

template <typename T>
static void run(fz::input &in)
{
    // Coordinates to paths
    {
        T p[6];
        for (auto &v : p)
            v = coord<T>(in);
        arma::u32_vec no_interact = counts(in);
        arma::Mat<T> interact_coord = interactions<T>(in, no_interact);
        arma::Col<T> path_length(in.size(2));
        arma::Mat<T> fbs_pos, lbs_pos(in.size(4), in.size(2)), path_angles;
        std::vector<arma::Mat<T>> path_coord(in.size(2));
        const int mask = in.ConsumeIntegralInRange<int>(0, 31);
        const bool reverse = in.ConsumeBool();
        fz::clean([&]
                  { quadriga_lib::coord2path<T>(p[0], p[1], p[2], p[3], p[4], p[5], &no_interact, &interact_coord,
                                                (mask & 1) ? &path_length : nullptr, (mask & 2) ? &fbs_pos : nullptr,
                                                (mask & 4) ? &lbs_pos : nullptr, (mask & 8) ? &path_angles : nullptr,
                                                (mask & 16) ? &path_coord : nullptr, reverse); });
    }

    // IRS segments
    {
        T irs[3];
        for (auto &v : irs)
            v = coord<T>(in);
        arma::u32_vec no_interact_1 = counts(in), no_interact_2 = counts(in), no_interact(in.size(2));
        arma::Mat<T> coord_1 = interactions<T>(in, no_interact_1), coord_2 = interactions<T>(in, no_interact_2), interact_coord;
        std::vector<bool> active_path(near(in, no_interact_1.n_elem * no_interact_2.n_elem));
        for (size_t i = 0; i < active_path.size(); ++i)
            active_path[i] = in.ConsumeBool();
        const bool rev_1 = in.ConsumeBool(), rev_2 = in.ConsumeBool(), use_active = in.ConsumeBool();
        fz::clean([&]
                  { quadriga_lib::combine_irs_coord<T>(irs[0], irs[1], irs[2], &no_interact_1, &coord_1, &no_interact_2, &coord_2,
                                                       &no_interact, &interact_coord, rev_1, rev_2, use_active ? &active_path : nullptr); });
    }

    // Tubes around a path; up to 16 points, up to 16 edges
    {
        arma::Mat<T> path_coord = geo<T>(in, near(in, 3), in.size(16)), vert(in.size(4), in.size(2));
        arma::umat faces;
        const T radius = in.ConsumeBool() ? T(0.1) : in.value<T>();
        const arma::uword n_edges = in.size(16);
        fz::clean([&] { quadriga_lib::path_to_tube<T>(&path_coord, &vert, &faces, radius, n_edges); });
    }

    // Diffraction ellipsoids
    {
        const arma::uword n_pos = in.size(16);
        arma::Mat<T> orig = geo<T>(in, n_pos, near(in, 3)), dest = geo<T>(in, near(in, n_pos), near(in, 3));
        const T frequency = in.ConsumeBool() ? T(3.5e9) : in.value<T>();
        const int lod = in.ConsumeIntegralInRange<int>(-1, 8);
        arma::Cube<T> ray_x, ray_y(in.size(2), in.size(2), in.size(2)), ray_z, weight;
        fz::clean([&] { quadriga_lib::generate_diffraction_paths<T>(orig, dest, frequency, lod, ray_x, ray_y, ray_z, weight); });
    }
}

void fuzz_one(const uint8_t *data, size_t size)
{
    fz::input in(data, size);
    if (in.ConsumeBool())
        run<double>(in);
    else
        run<float>(in);
}
