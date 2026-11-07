// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Sum-of-sinusoids generators with fuzzed frequencies, phases, ACF tables and positions (also with outputs that
// alias inputs): sos_preset, sos_init_phases, sos_acfi, sos_val, sos_map, sos_randn / sos_rand / sos_randi,
// sos_remove_bias, sos_acf_2d, sos_acf_approx, sos_calc_mse, sos_acf_estimate, sos_generate

#include "fuzz_target.hpp"

// A seed >= 0 or an invalid negative one (-1, the system random device, is left out to keep inputs repeatable)
static arma::sword seed(fz::input &in)
{
    return in.ConsumeIntegralInRange<int>(0, 15) == 0 ? -arma::sword(in.ConsumeIntegralInRange<int>(2, 3)) : in.ConsumeIntegralInRange<arma::sword>(0, 1000);
}

static std::string preset_name(fz::input &in)
{
    const char *names[] = {"Comb128", "Exp128", "Gauss128", "Comb256", "Exp512", "Gauss1024", "Comb", "comb128", ""};
    return in.ConsumeIntegralInRange<int>(0, 7) != 0 ? names[in.ConsumeIntegralInRange<int>(0, 8)] : in.ConsumeRandomLengthString(10);
}

// An ACF table of up to 16 points: equally spaced from 0 (valid), or any values
template <typename T>
static void acf_table(fz::input &in, arma::Col<T> &acf, arma::Col<T> &dist)
{
    const arma::uword n = in.size(16);
    acf = in.col<T>(in.ConsumeIntegralInRange<int>(0, 7) == 0 ? in.size(17) : n);
    if (in.ConsumeBool())
        acf = arma::clamp(acf / T(100), T(-0.99), T(1));
    if (in.ConsumeIntegralInRange<int>(0, 3) != 0)
    {
        dist = arma::regspace<arma::Col<T>>(0, n == 0 ? 0 : n - 1) * (in.ConsumeBool() ? T(1) : std::abs(in.value<T>()));
        dist.resize(n); // regspace(0, 0) has one element
    }
    else
        dist = in.col<T>(in.ConsumeBool() ? n : in.size(17));
}

template <typename T>
static void run(fz::input &in)
{
    // Frequencies: from a preset (up to 1024 x 3) or fuzzed (up to 16 x 4)
    arma::Mat<T> freq;
    arma::Col<T> acf, dist;
    if (in.ConsumeBool())
        fz::clean([&] { quadriga_lib::sos_preset<T>(preset_name(in), freq, &acf, &dist, in.value<T>()); });
    else
        freq = in.mat<T>(in.size(16), in.ConsumeIntegralInRange<int>(0, 7) == 0 ? in.size(4) : in.ConsumeIntegralInRange<arma::uword>(1, 3)),
        acf_table<T>(in, acf, dist);
    if (in.ConsumeBool() && freq.n_rows > 16)
        freq = freq.rows(0, in.ConsumeIntegralInRange<arma::uword>(0, 15));
    if (acf.n_elem > 16) // the 200 points of a preset make the 2-D grids slow
        acf.resize(in.size(16)), dist.resize(acf.n_elem);

    // Phases: drawn or fuzzed
    arma::Mat<T> phase, phase_b;
    const arma::uword n_gen = in.size(4);
    if (in.ConsumeBool())
        fz::clean([&] { quadriga_lib::sos_init_phases<T>(freq.n_rows, n_gen, phase, seed(in)); });
    else
        phase = in.mat<T>(in.ConsumeBool() ? freq.n_rows : in.size(17), n_gen);
    phase_b = in.ConsumeBool() ? phase : in.mat<T>(phase.n_rows, in.ConsumeBool() ? phase.n_cols : in.size(4));

    // Positions with 1 to 3 columns (sometimes 0 or 4), up to 16 rows
    auto positions = [&]
    {
        arma::Mat<T> p = in.mat<T>(in.size(16), in.ConsumeIntegralInRange<int>(0, 7) == 0 ? in.size(4) : in.ConsumeIntegralInRange<arma::uword>(1, 3));
        if (in.ConsumeBool())
            p /= T(10);
        return p;
    };
    arma::Mat<T> pos = positions(), pos_b = positions(), val;
    const bool use_b = in.ConsumeBool(), use_phase_b = in.ConsumeBool(), use_acf = in.ConsumeBool(), uniform = in.ConsumeBool();
    const int kernel = in.ConsumeIntegralInRange<int>(-1, 3);
    const T scale_b = in.ConsumeBool() ? T(1) : in.value<T>();

    switch (in.ConsumeIntegralInRange<int>(0, 10))
    {
    case 0:
    {
        arma::Col<T> d = arma::abs(in.col<T>(in.size(16))), v;
        if (in.ConsumeBool() && d.n_elem != 0)
            d(0) = -d(0);
        fz::clean([&] { quadriga_lib::sos_acfi(acf, dist, d, in.ConsumeBool() ? v : d); });
        break;
    }
    case 1:
    {
        // val is a fresh matrix or one of the inputs
        arma::Mat<T> *out[] = {&val, &pos, &pos_b, &phase, &phase_b};
        arma::Mat<T> *v = out[in.ConsumeIntegralInRange<int>(0, 7) < 5 ? 0 : in.ConsumeIntegralInRange<int>(0, 4)];
        fz::clean([&] { quadriga_lib::sos_val(freq, phase, pos, *v, use_b ? &pos_b : nullptr, use_phase_b ? &phase_b : nullptr,
                                              use_acf ? &acf : nullptr, use_acf ? &dist : nullptr, scale_b, uniform, kernel); });
        break;
    }
    case 2:
    {
        arma::Col<T> x = in.col<T>(in.size(8)), y = in.col<T>(in.size(8)), z = in.col<T>(in.size(8));
        std::vector<arma::Cube<T>> map(in.size(3));
        if (in.ConsumeBool() && !map.empty())
            map[0].set_size(in.size(3), in.size(3), in.size(3));
        const int mode = in.ConsumeIntegralInRange<int>(0, 3);
        fz::clean([&] { quadriga_lib::sos_map(freq, phase, x, map, mode & 1 ? &y : nullptr, mode & 2 ? &z : nullptr, uniform, kernel); });
        break;
    }
    case 3:
    {
        arma::Col<T> dd = in.col<T>(in.size(4)), dd_b = in.col<T>(in.ConsumeBool() ? dd.n_elem : in.size(4));
        if (in.ConsumeBool())
            dd = arma::abs(dd), dd_b = arma::abs(dd_b);
        const std::string preset = preset_name(in);
        const arma::sword s = seed(in);
        arma::Mat<T> *v = in.ConsumeBool() ? &val : &pos; // val may be pos
        const int which = in.ConsumeIntegralInRange<int>(0, 2);
        const arma::uword imax = in.ConsumeIntegralInRange<int>(0, 3) == 0 ? in.ConsumeIntegral<arma::uword>() : in.size(10);
        if (which == 0)
            fz::clean([&] { quadriga_lib::sos_randn(dd, pos, *v, use_b ? &pos_b : nullptr, use_phase_b ? &dd_b : nullptr, preset, s); });
        else if (which == 1)
            fz::clean([&] { quadriga_lib::sos_rand(dd, pos, *v, use_b ? &pos_b : nullptr, use_phase_b ? &dd_b : nullptr, preset, s); });
        else
            fz::clean([&] { quadriga_lib::sos_randi(dd, pos, imax, *v, use_b ? &pos_b : nullptr, use_phase_b ? &dd_b : nullptr, preset, s); });
        break;
    }
    case 4:
    {
        arma::Mat<T> offset, mean_before;
        const int outs = in.ConsumeIntegralInRange<int>(0, 3);
        arma::Mat<T> *off = outs & 1 ? (in.ConsumeBool() ? &offset : &phase_b) : nullptr; // phase_b as output may clash
        fz::clean([&] { quadriga_lib::sos_remove_bias(freq, phase, pos, use_b ? &pos_b : nullptr, use_phase_b ? &phase_b : nullptr,
                                                      use_acf ? &acf : nullptr, use_acf ? &dist : nullptr, off,
                                                      outs & 2 ? &mean_before : nullptr, scale_b); });
        break;
    }
    case 5:
    {
        arma::Mat<T> a2;
        arma::Col<T> d2;
        fz::clean([&] { quadriga_lib::sos_acf_2d(acf, dist, a2, in.ConsumeBool() ? &d2 : nullptr); });
        break;
    }
    case 6:
    {
        arma::Cube<T> approx;
        arma::Col<T> d2;
        fz::clean([&] { quadriga_lib::sos_acf_approx(freq, dist, approx, in.ConsumeBool() ? (in.ConsumeBool() ? &d2 : &dist) : nullptr); });
        break;
    }
    case 7:
    {
        T core, all;
        arma::Col<T> tail;
        arma::Mat<T> dir;
        const uint8_t outs = in.ConsumeIntegral<uint8_t>();
        const arma::uword n_dir = in.size(6);
        fz::clean([&] { quadriga_lib::sos_calc_mse(freq, acf, dist, outs & 1 ? &core : nullptr, outs & 2 ? &all : nullptr,
                                                   outs & 4 ? (outs & 16 ? &acf : &tail) : nullptr, outs & 8 ? &dir : nullptr, n_dir); });
        break;
    }
    case 8:
    {
        arma::Col<T> bins = in.col<T>(in.size(8)), est, d_est, dual;
        if (in.ConsumeBool())
            bins = arma::sort(arma::abs(bins));
        arma::Col<T> ph = phase.n_elem ? arma::Col<T>(phase.col(0)) : in.col<T>(in.size(17));
        const arma::uword n_pos = in.size(32);
        const uint8_t outs = in.ConsumeIntegral<uint8_t>();
        fz::clean([&] { quadriga_lib::sos_acf_estimate(freq, ph, bins, n_pos, outs & 8 ? bins : est, outs & 1 ? &d_est : nullptr,
                                                       outs & 2 ? &dual : nullptr, outs & 4 ? &acf : nullptr,
                                                       outs & 4 ? &dist : nullptr, seed(in)); });
        break;
    }
    case 9:
    {
        // A search grid of at most a few hundred points: freq_max - freq_min below 30 / max(dist)
        arma::Mat<T> f_out, dir;
        arma::Col<T> ase;
        const arma::uword n_coeff = in.size(8), n_dim = in.size(4), n_dir = in.size(4);
        const bool uniform_smp = in.ConsumeBool(), random_init = in.ConsumeBool();
        const T d_max = dist.n_elem ? std::abs(dist.max()) : T(1);
        const T unit = (d_max > T(0) && std::isfinite(d_max)) ? T(1) / d_max : T(1);
        T f_min = 0, f_max = 0;
        if (in.ConsumeBool())
            f_min = in.ConsumeFloatingPointInRange<T>(T(-1), T(15)) * unit, f_max = in.ConsumeFloatingPointInRange<T>(T(-1), T(30)) * unit;
        if (in.ConsumeIntegralInRange<int>(0, 7) == 0)
            f_min = in.value<T>(), f_max = f_min;
        const uint8_t outs = in.ConsumeIntegral<uint8_t>();
        fz::clean([&] { quadriga_lib::sos_generate(acf, dist, n_coeff, f_out, outs & 1 ? (outs & 4 ? &acf : &ase) : nullptr,
                                                   outs & 2 ? &dir : nullptr, n_dim, uniform_smp, n_dir, random_init, f_min, f_max,
                                                   seed(in)); });
        break;
    }
    default:
    {
        arma::Mat<T> f;
        arma::Col<T> a, d;
        const uint8_t outs = in.ConsumeIntegral<uint8_t>();
        fz::clean([&] { quadriga_lib::sos_preset<T>(preset_name(in), outs & 4 ? freq : f, outs & 1 ? &a : nullptr,
                                                    outs & 2 ? &d : nullptr, in.value<T>()); });
        fz::clean([&] { quadriga_lib::sos_init_phases<T>(in.size(1100), in.size(5), phase, seed(in)); });
    }
    }
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
