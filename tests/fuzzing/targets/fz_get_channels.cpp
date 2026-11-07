// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Channel coefficients from paths and array antennas: get_channels_spherical, get_channels_planar, get_channels_irs
// and get_channels_multifreq with generated or fuzzed arrayants, fuzzed positions, orientations and path data
// (consistent or mismatched sizes), optional outputs and options

#include "fuzz_target.hpp"

// A generated arrayant on a 30 degree grid (with fuzzed positions and coupling), or a fuzzed one
template <typename T>
static quadriga_lib::arrayant<T> antenna(fz::input &in)
{
    quadriga_lib::arrayant<T> a;
    switch (in.ConsumeIntegralInRange<int>(0, 5))
    {
    case 0:
        return quadriga_lib::generate_arrayant_omni<T>(T(30));
    case 1:
        return quadriga_lib::generate_arrayant_xpol<T>(T(30));
    case 2:
        fz::clean([&] { a = quadriga_lib::generate_arrayant_ula<T>(in.size(4), T(3.5e9), in.value<T>(), nullptr, T(30)); });
        break;
    case 3:
        fz::clean([&] { a = quadriga_lib::generate_arrayant_3GPP<T>(in.size(2), in.size(2), T(3.5e9), in.ConsumeIntegralInRange<unsigned>(1, 6),
                                                                    T(5), T(0.5), 1, 1, T(0.5), T(0.5), nullptr, T(30)); });
        break;
    default:
        return fz::arrayant<T>(in);
    }
    if (in.ConsumeBool() && a.n_elements() != 0)
        a.element_pos = in.mat<T>(3, a.n_elements());
    if (in.ConsumeBool() && a.n_elements() != 0)
    {
        arma::uword n_p = in.size(4);
        a.coupling_re = in.mat<T>(a.n_elements(), n_p), a.coupling_im = in.mat<T>(a.n_elements(), n_p);
    }
    return a;
}

// Path data of n_path paths; each size is wrong in about one input of four
template <typename T>
struct paths
{
    arma::Mat<T> fbs, lbs, M;
    arma::Col<T> gain, length;
    paths(fz::input &in, arma::uword n_path, arma::uword m_rows = 8)
    {
        const bool consistent = in.ConsumeIntegralInRange<int>(0, 3) != 0;
        auto sz = [&](arma::uword n) { return consistent ? n : in.size(n + 1); };
        fbs = in.mat<T>(sz(3), sz(n_path)), lbs = in.mat<T>(sz(3), sz(n_path));
        gain = in.col<T>(sz(n_path)), length = in.col<T>(sz(n_path));
        M = in.mat<T>(sz(m_rows), sz(n_path));
        if (in.ConsumeBool()) // non-negative path lengths
            length = arma::abs(length);
    }
};

template <typename T>
static void run(fz::input &in)
{
    quadriga_lib::arrayant<T> tx = antenna<T>(in), rx = antenna<T>(in);
    T p[12];
    for (auto &v : p)
        v = in.value<T>();
    const T fc = in.ConsumeBool() ? T(3.5e9) : in.value<T>();
    const bool absolute = in.ConsumeBool(), fake_los = in.ConsumeBool();
    const uint8_t outs = in.ConsumeIntegral<uint8_t>();
    arma::Cube<T> c_re, c_im, dl, aod, eod, aoa, eoa;
    auto opt = [&](arma::Cube<T> &x, int bit) { return (outs >> bit) & 1 ? &x : nullptr; };

    switch (in.ConsumeIntegralInRange<int>(0, 3))
    {
    case 0:
    {
        paths<T> q(in, in.size(8));
        const bool avx2 = in.ConsumeBool();
        fz::clean([&] { quadriga_lib::get_channels_spherical(&tx, &rx, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                                                             p[10], p[11], &q.fbs, &q.lbs, &q.gain, &q.length, &q.M, opt(c_re, 0),
                                                             opt(c_im, 1), opt(dl, 2), fc, absolute, fake_los, opt(aod, 3),
                                                             opt(eod, 4), opt(aoa, 5), opt(eoa, 6), avx2); });
        break;
    }
    case 1:
    {
        const arma::uword n_path = in.size(8);
        paths<T> q(in, n_path);
        arma::Col<T> a1 = in.col<T>(n_path), a2 = in.col<T>(n_path), a3 = in.col<T>(in.ConsumeBool() ? n_path : in.size(9)),
                     a4 = in.col<T>(n_path), doppler;
        fz::clean([&] { quadriga_lib::get_channels_planar(&tx, &rx, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10],
                                                          p[11], &a1, &a2, &a3, &a4, &q.gain, &q.length, &q.M, opt(c_re, 0),
                                                          opt(c_im, 1), opt(dl, 2), fc, absolute, fake_los,
                                                          (outs & 8) ? &doppler : nullptr); });
        break;
    }
    case 2:
    {
        quadriga_lib::arrayant<T> irs = antenna<T>(in), irs2 = antenna<T>(in);
        T ip[6];
        for (auto &v : ip)
            v = in.value<T>();
        const arma::uword n1 = in.size(6), n2 = in.size(6);
        paths<T> q1(in, n1), q2(in, n2);
        const arma::uword i_irs = in.index(irs.n_ports());
        const T threshold = in.ConsumeBool() ? T(-140) : in.value<T>();
        std::vector<bool> active(in.ConsumeBool() ? n1 * n2 : in.size(n1 * n2 + 1));
        for (size_t i = 0; i < active.size(); ++i)
            active[i] = in.ConsumeBool();
        const bool use_irs2 = in.ConsumeBool(), use_active = in.ConsumeBool();
        fz::clean([&] { quadriga_lib::get_channels_irs(&tx, &rx, &irs, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10],
                                                       p[11], ip[0], ip[1], ip[2], ip[3], ip[4], ip[5], &q1.fbs, &q1.lbs, &q1.gain,
                                                       &q1.length, &q1.M, &q2.fbs, &q2.lbs, &q2.gain, &q2.length, &q2.M,
                                                       opt(c_re, 0), opt(c_im, 1), opt(dl, 2), i_irs, threshold, fc, absolute,
                                                       opt(aod, 3), opt(eod, 4), opt(aoa, 5), opt(eoa, 6),
                                                       use_irs2 ? &irs2 : nullptr, use_active ? &active : nullptr); });
        break;
    }
    default:
    {
        // Up to 3 frequencies per array, path gain [n_path, n_freq_in], M [8 or 2, n_path, n_freq_in]
        std::vector<quadriga_lib::arrayant<T>> tx_vec, rx_vec;
        for (arma::uword f = 0, n = in.size(3); f < n; ++f)
            tx_vec.push_back(in.ConsumeIntegralInRange<int>(0, 5) == 0 ? antenna<T>(in) : tx), tx_vec.back().center_frequency = in.value<T>();
        for (arma::uword f = 0, n = in.size(3); f < n; ++f)
            rx_vec.push_back(in.ConsumeIntegralInRange<int>(0, 5) == 0 ? antenna<T>(in) : rx), rx_vec.back().center_frequency = in.value<T>();
        const arma::uword n_path = in.size(8), n_fin = in.size(3);
        paths<T> q(in, n_path);
        const bool consistent = in.ConsumeIntegralInRange<int>(0, 3) != 0;
        auto sz = [&](arma::uword n) { return consistent ? n : in.size(n + 1); };
        arma::Mat<T> gain = in.mat<T>(sz(n_path), sz(n_fin));
        arma::Cube<T> M = in.cube<T>(in.ConsumeBool() ? 8 : (consistent ? 2 : in.size(9)), sz(n_path), sz(n_fin));
        arma::Col<T> f_in = in.col<T>(sz(n_fin)), f_out = in.col<T>(in.size(6));
        if (in.ConsumeBool())
            f_in = arma::sort(f_in);
        const T speed = in.ConsumeBool() ? T(299792458.0) : (in.ConsumeBool() ? T(343) : in.value<T>());
        std::vector<arma::Cube<T>> v_re, v_im, v_dl;
        fz::clean([&] { quadriga_lib::get_channels_multifreq(tx_vec, rx_vec, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                                                             p[10], p[11], q.fbs, q.lbs, gain, q.length, M, f_in, f_out, v_re, v_im,
                                                             v_dl, absolute, fake_los, speed); });
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
