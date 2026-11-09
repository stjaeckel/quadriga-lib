// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Array antenna interpolation: arrayant_interpolate and arrayant_interpolate_multi on a fuzzed arrayant (grids,
// patterns, positions and coupling of consistent or mismatched sizes) with fuzzed angles, element indices,
// orientations, element positions, optional outputs, target frequencies and the available kernels

#include "fuzz_target.hpp"

// A matrix with 1 row, n_out rows or a fuzzed number of rows, and n_ang columns (or a fuzzed number)
template <typename T>
static arma::Mat<T> angles(fz::input &in, arma::uword n_out, arma::uword n_ang)
{
    arma::uword rows = 1;
    switch (in.ConsumeIntegralInRange<int>(0, 4))
    {
    case 1:
        rows = n_out;
        break;
    case 2:
        rows = in.size(9);
        break;
    }
    return in.mat<T>(rows, in.ConsumeIntegralInRange<int>(0, 7) == 0 ? in.size(17) : n_ang);
}

template <typename T>
static void run(fz::input &in)
{
    quadriga_lib::arrayant<T> a = fz::arrayant<T>(in);
    const arma::uword n_e = a.e_theta_re.n_slices;

    arma::uvec i_element;
    if (in.ConsumeBool())
    {
        i_element.set_size(in.size(8));
        for (auto &i : i_element)
            i = in.index(n_e);
    }
    const arma::uword n_out = i_element.is_empty() ? n_e : i_element.n_elem;
    const arma::uword n_ang = in.size(16);

    arma::Mat<T> az = angles<T>(in, n_out, n_ang), el = angles<T>(in, n_out, n_ang);
    if (in.ConsumeBool())
        el.set_size(az.n_rows, az.n_cols), el.zeros();

    arma::Cube<T> orientation;
    const bool use_orientation = in.ConsumeBool();
    switch (in.ConsumeIntegralInRange<int>(0, 4))
    {
    case 0:
        orientation = in.cube<T>(3, 1, 1);
        break;
    case 1:
        orientation = in.cube<T>(3, n_out, 1);
        break;
    case 2:
        orientation = in.cube<T>(3, 1, n_ang);
        break;
    case 3:
        orientation = in.cube<T>(3, n_out, n_ang);
        break;
    default:
        orientation = in.cube<T>(in.size(4), in.size(n_out + 1), in.size(n_ang + 1));
    }

    arma::Mat<T> element_pos_i;
    const bool use_pos = in.ConsumeBool();
    element_pos_i = in.ConsumeBool() ? in.mat<T>(3, n_out) : in.mat<T>(in.size(4), in.size(n_out + 1));

    // Kernel: auto, GENERIC, or AVX2 where available
    const int use_kernel = in.ConsumeIntegralInRange<int>(0, quadriga_lib::quadriga_lib_has_AVX2() ? 2 : 1);

    // Optional outputs, each sometimes nullptr; outputs sometimes with a stale size
    arma::Mat<T> V_re, V_im, H_re, H_im, dist, az_loc, el_loc, gamma;
    uint8_t out_mask = in.ConsumeIntegral<uint8_t>();
    if (in.ConsumeBool())
        V_re.set_size(in.size(4), in.size(4)), dist.set_size(in.size(4), in.size(4));
    auto opt = [&](arma::Mat<T> &m, int bit) { return (out_mask >> bit) & 1 ? &m : nullptr; };

    fz::clean([&]
              { quadriga_lib::arrayant_interpolate<T>(a, az, el, V_re, V_im, H_re, H_im, &i_element,
                                                      use_orientation ? &orientation : nullptr, use_pos ? &element_pos_i : nullptr,
                                                      opt(dist, 4), opt(az_loc, 5), opt(el_loc, 6), opt(gamma, 7), use_kernel); });

    // Same arrayant at up to three frequencies (the copies may get a different grid size)
    std::vector<quadriga_lib::arrayant<T>> vec;
    arma::uword n_freq_ant = in.size(3);
    for (arma::uword f = 0; f < n_freq_ant; ++f)
    {
        vec.push_back(in.ConsumeIntegralInRange<int>(0, 5) == 0 ? fz::arrayant<T>(in) : a);
        vec.back().center_frequency = in.value<T>();
    }
    arma::Col<T> freq = in.col<T>(in.size(5)); // May be empty: the center frequencies of the entries
    arma::Cube<T> CV_re, CV_im, CH_re, CH_im;
    fz::clean([&]
              { quadriga_lib::arrayant_interpolate_multi<T>(vec, az, el, CV_re, CV_im, CH_re, CH_im, &freq, &i_element,
                                                            use_orientation ? &orientation : nullptr,
                                                            use_pos ? &element_pos_i : nullptr, use_kernel); });
}

void fuzz_seeds(const fz::seeds &) {}

void fuzz_one(const uint8_t *data, size_t size)
{
    fz::input in(data, size);
    if (in.ConsumeBool())
        run<float>(in);
    else
        run<double>(in);
}
