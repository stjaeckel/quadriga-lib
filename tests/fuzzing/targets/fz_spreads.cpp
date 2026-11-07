// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Channel statistics: acdf, calc_delay_spread, calc_angular_spreads_sphere, calc_rician_k_factor and
// calc_cross_polarization_ratio with fuzzed (empty, ragged, zero, negative, non-finite) delays, angles and powers

#include "fuzz_target.hpp"

void fuzz_seeds(const fz::seeds &) {}

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

// A value: often a small non-negative one (powers, delays, lengths), else a fuzzed value
template <typename T>
static T val(fz::input &in)
{
    return in.ConsumeBool() ? T(in.ConsumeIntegralInRange<int>(0, 4)) * T(0.25) : in.value<T>();
}

// One column per CIR, with lengths that mostly match the reference lengths n_path
template <typename T>
static std::vector<arma::Col<T>> cols(fz::input &in, const std::vector<arma::uword> &n_path)
{
    std::vector<arma::Col<T>> x(near(in, n_path.size()));
    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i].set_size(near(in, i < n_path.size() ? n_path[i] : 1));
        for (auto &v : x[i])
            v = val<T>(in);
    }
    return x;
}

// Positions of size [3, 1] or [3, n_cir], sometimes of another size
template <typename T>
static arma::Mat<T> pos(fz::input &in, arma::uword n_cir)
{
    arma::Mat<T> x(near(in, 3), in.ConsumeBool() ? near(in, 1) : near(in, n_cir));
    for (auto &v : x)
        v = val<T>(in);
    return x;
}

template <typename T>
static void run(fz::input &in)
{
    // ACDF; bins are missing, empty (generated) or given
    {
        arma::Mat<T> data(in.size(32), in.size(4));
        for (auto &v : data)
            v = in.value<T>();
        arma::Col<T> bins(in.ConsumeBool() ? 0 : in.size(8)), cdf_avg, mu, sig;
        for (auto &v : bins)
            v = val<T>(in);
        if (in.ConsumeBool())
            bins = arma::sort(bins);
        arma::Mat<T> cdf_per_set;
        const arma::uword n_bins = in.size(300);
        const int mask = in.ConsumeIntegralInRange<int>(0, 31);
        fz::clean([&]
                  { quadriga_lib::acdf<T>(data, (mask & 1) ? &bins : nullptr, (mask & 2) ? &cdf_per_set : nullptr,
                                          (mask & 4) ? &cdf_avg : nullptr, (mask & 8) ? &mu : nullptr, (mask & 16) ? &sig : nullptr, n_bins); });
    }

    // Reference sizes: up to 8 CIRs with up to 32 paths each
    std::vector<arma::uword> n_path(in.size(8));
    for (auto &n : n_path)
        n = in.size(32);
    const arma::uword n_cir = n_path.size();

    // Delay spread
    {
        auto delays = cols<T>(in, n_path), powers = cols<T>(in, n_path);
        const T threshold = in.ConsumeBool() ? T(100) : in.value<T>();
        const T granularity = in.ConsumeBool() ? T(0) : in.value<T>();
        arma::Col<T> mean_delay(in.size(2));
        const bool use_mean = in.ConsumeBool();
        fz::clean([&] { quadriga_lib::calc_delay_spread<T>(delays, powers, threshold, granularity, use_mean ? &mean_delay : nullptr); });
    }

    // Angular spreads, with and without wrapping, bank angle and quantization
    {
        auto az = cols<T>(in, n_path), el = cols<T>(in, n_path), powers = cols<T>(in, n_path);
        arma::Col<T> as, es;
        arma::Mat<T> orientation(in.size(4), in.size(2));
        std::vector<arma::Col<T>> phi(in.size(2)), theta;
        const int mask = in.ConsumeIntegralInRange<int>(0, 31);
        const bool disable_wrapping = in.ConsumeBool(), calc_bank_angle = in.ConsumeBool();
        const T quantize = in.ConsumeBool() ? T(0) : in.value<T>();
        fz::clean([&]
                  { quadriga_lib::calc_angular_spreads_sphere<T>(az, el, powers, (mask & 1) ? &as : nullptr, (mask & 2) ? &es : nullptr,
                                                                 (mask & 4) ? &orientation : nullptr, (mask & 8) ? &phi : nullptr,
                                                                 (mask & 16) ? &theta : nullptr, disable_wrapping, calc_bank_angle, quantize); });
    }

    // Rician K-factor and cross-polarization ratio
    {
        auto powers = cols<T>(in, n_path), path_length = cols<T>(in, n_path);
        arma::Mat<T> tx_pos = pos<T>(in, n_cir), rx_pos = pos<T>(in, n_cir);
        const T window_size = in.ConsumeBool() ? T(0.01) : in.value<T>();
        arma::Col<T> kf, pg(in.size(2));
        const int mask = in.ConsumeIntegralInRange<int>(0, 3);
        fz::clean([&]
                  { quadriga_lib::calc_rician_k_factor<T>(powers, path_length, tx_pos, rx_pos, (mask & 1) ? &kf : nullptr,
                                                          (mask & 2) ? &pg : nullptr, window_size); });

        std::vector<arma::Mat<T>> M(near(in, n_cir));
        for (size_t i = 0; i < M.size(); ++i)
        {
            M[i].set_size(near(in, 8), near(in, i < n_cir ? n_path[i] : 1));
            for (auto &v : M[i])
                v = in.value<T>();
        }
        arma::Mat<T> xpr(in.size(2), in.size(7));
        const bool include_los = in.ConsumeBool();
        const int mask_x = in.ConsumeIntegralInRange<int>(0, 3);
        fz::clean([&]
                  { quadriga_lib::calc_cross_polarization_ratio<T>(powers, M, path_length, tx_pos, rx_pos, (mask_x & 1) ? &xpr : nullptr,
                                                                   (mask_x & 2) ? &pg : nullptr, include_los, window_size); });
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
