// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Channel coefficient tools on fuzzed coefficient and delay cubes (consistent or mismatched sizes): quantize_delays
// (also with outputs that alias inputs), baseband_freq_response, baseband_freq_response_vec and
// baseband_freq_response_multi

#include "fuzz_target.hpp"

template <typename T>
static void run(fz::input &in)
{
    // Up to 4 snapshots (or frequencies) of up to 4 x 4 antennas and 8 paths
    const arma::uword n_snap = in.size(4), n_rx = in.size(4), n_tx = in.size(4);
    const bool consistent = in.ConsumeIntegralInRange<int>(0, 3) != 0;
    const bool shared = in.ConsumeBool();
    const T delay_scale = in.ConsumeBool() ? T(1.0e-8) : T(1);
    std::vector<arma::Cube<T>> cre, cim, dl;
    for (arma::uword s = 0; s < n_snap; ++s)
    {
        arma::uword n_path = in.size(8);
        auto sz = [&](arma::uword n) { return consistent ? n : in.size(n + 1); };
        cre.push_back(in.cube<T>(n_rx, n_tx, n_path));
        cim.push_back(in.cube<T>(sz(n_rx), sz(n_tx), sz(n_path)));
        dl.push_back(shared ? in.cube<T>(sz(1), sz(1), sz(n_path)) : in.cube<T>(sz(n_rx), sz(n_tx), sz(n_path)));
        dl.back() *= delay_scale;
    }
    if (!consistent && in.ConsumeBool())
        (in.ConsumeBool() ? cim : dl).resize(in.size(5));

    // quantize_delays: outputs are fresh vectors or any of the inputs (all three must differ)
    {
        std::vector<arma::Cube<T>> ire = cre, iim = cim, idl = dl, o1, o2, o3;
        std::vector<arma::Cube<T>> *obj[] = {&o1, &o2, &o3, &ire, &iim, &idl};
        auto pick = [&](int dflt) { return obj[in.ConsumeIntegralInRange<int>(0, 7) < 6 ? dflt : in.ConsumeIntegralInRange<int>(0, 5)]; };
        std::vector<arma::Cube<T>> *q_re = pick(0), *q_im = pick(1), *q_dl = pick(2);
        T tap = in.ConsumeBool() ? T(5.0e-9) : in.value<T>();
        arma::uword max_taps = in.ConsumeIntegralInRange<int>(0, 7) == 0 ? std::numeric_limits<arma::uword>::max() : in.size(64);
        T exponent = in.ConsumeBool() ? T(1) : in.value<T>();
        int fix_taps = in.ConsumeIntegralInRange<int>(-1, 4);
        fz::clean([&] { quadriga_lib::quantize_delays(&ire, &iim, &idl, q_re, q_im, q_dl, tap, max_taps, exponent, fix_taps); });
    }

    // Outputs: nullptr or an object, sometimes with a stale size
    const uint8_t out_mask = in.ConsumeIntegral<uint8_t>();
    arma::Col<T> pilot = in.col<T>(in.size(16));
    const double bandwidth = in.value<double>();

    // baseband_freq_response on one snapshot
    {
        static const arma::Cube<T> empty;
        const arma::uword s = n_snap == 0 ? 0 : in.ConsumeIntegralInRange<arma::uword>(0, n_snap - 1);
        const arma::Cube<T> &a = n_snap ? cre[s] : empty, &b = s < cim.size() ? cim[s] : empty, &d = s < dl.size() ? dl[s] : empty;
        arma::Cube<T> h_re, h_im;
        arma::Cube<std::complex<T>> h;
        if (out_mask & 8)
            h_re.set_size(in.size(3), in.size(3), in.size(3)), h.set_size(in.size(3), in.size(3), in.size(3));
        fz::clean([&] { quadriga_lib::baseband_freq_response(&a, &b, &d, (out_mask & 16) ? nullptr : &pilot, bandwidth,
                                                             (out_mask & 1) ? &h_re : nullptr, (out_mask & 2) ? &h_im : nullptr,
                                                             (out_mask & 4) ? &h : nullptr); });
    }

    // baseband_freq_response_vec on all or some snapshots
    {
        arma::uvec i_snap(in.size(5));
        for (auto &i : i_snap)
            i = in.index(n_snap);
        std::vector<arma::Cube<T>> h_re, h_im;
        if (out_mask & 8)
            h_re.resize(in.size(3));
        fz::clean([&] { quadriga_lib::baseband_freq_response_vec(&cre, &cim, &dl, &pilot, bandwidth,
                                                                 (out_mask & 1) ? &h_re : nullptr, (out_mask & 2) ? &h_im : nullptr,
                                                                 (out_mask & 32) ? &i_snap : nullptr); });
    }

    // baseband_freq_response_multi with the snapshots as input frequencies
    {
        arma::Col<T> freq_in = in.col<T>(in.ConsumeBool() ? n_snap : in.size(5)), freq_out = in.col<T>(in.size(16));
        if (in.ConsumeBool())
            freq_in = arma::sort(freq_in);
        if (in.ConsumeBool())
            freq_in *= T(1.0e9), freq_out *= T(1.0e9);
        arma::Cube<T> h_re, h_im;
        arma::Cube<std::complex<T>> h;
        const bool remove_phase = in.ConsumeBool();
        fz::clean([&] { quadriga_lib::baseband_freq_response_multi(cre, cim, dl, freq_in, freq_out, (out_mask & 1) ? &h_re : nullptr,
                                                                   (out_mask & 2) ? &h_im : nullptr, (out_mask & 4) ? &h : nullptr,
                                                                   remove_phase); });
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
