// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Antenna array generators with fuzzed parameters: generate_antenna per type with its own inputs (3GPP and ula also
// with a fuzzed custom pattern), all types with tuples of fuzzed lengths and values (also inputs a type does not use),
// and generate_speaker

#include "fuzz_target.hpp"

// A grid resolution in degree: special values, or 5 to 90 degree (finer grids are slow, not wrong)
template <typename T>
static T resolution(fz::input &in)
{
    const T special[] = {T(0), -T(0), T(-1), T(1), T(4.9), T(90), T(91), T(1000), std::numeric_limits<T>::quiet_NaN(),
                         std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
                         std::numeric_limits<T>::denorm_min(), std::numeric_limits<T>::max(),
                         std::numeric_limits<T>::lowest()};
    if (in.ConsumeIntegralInRange<int>(0, 15) == 0)
        return special[in.ConsumeIntegralInRange<size_t>(0, sizeof(special) / sizeof(T) - 1)];
    return in.ConsumeFloatingPointInRange<T>(T(5), T(90));
}

// Grids of 1 degree or finer (also chosen by the library for res <= 0) get fewer elements
template <typename T>
static bool coarse(T res)
{
    return res >= T(4.9) || res == std::numeric_limits<T>::max() || res == std::numeric_limits<T>::infinity();
}

// A custom element pattern: none, a generated one, or one of up to 6 x 8 samples and 3 elements with any values
template <typename T>
static quadriga_lib::arrayant<T> pattern(fz::input &in, bool &use)
{
    quadriga_lib::arrayant<T> p;
    const int mode = in.ConsumeIntegralInRange<int>(0, 3);
    use = mode != 0;
    if (mode == 1)
        fz::clean([&] { p = quadriga_lib::generate_antenna<T>("custom", T(30), NAN, {in.value<T>(), in.value<T>(), in.value<T>()}); });
    else if (mode >= 2)
    {
        arma::uword n_el = in.size(6), n_az = in.size(8), n_e = in.size(3);
        p.e_theta_re = in.cube<T>(n_el, n_az, n_e), p.e_theta_im = in.cube<T>(n_el, n_az, n_e);
        p.e_phi_re = in.cube<T>(n_el, n_az, n_e), p.e_phi_im = in.cube<T>(n_el, n_az, n_e);
        if (mode == 2)
        {
            p.azimuth_grid = arma::linspace<arma::Col<T>>(T(-3.14159265), T(3.14159265), n_az);
            p.elevation_grid = arma::linspace<arma::Col<T>>(T(-1.5707963), T(1.5707963), n_el);
        }
        else
            p.azimuth_grid = in.col<T>(in.size(9)), p.elevation_grid = in.col<T>(in.size(7));
        if (in.ConsumeBool())
            p.element_pos = in.mat<T>(in.size(4), in.size(4));
        if (in.ConsumeBool())
            p.port_weights_re = in.mat<T>(in.size(4), in.size(4)), p.port_weights_im = in.mat<T>(in.size(4), in.size(4));
    }
    return p;
}

// An integer tuple of 0 to n_max + 1 entries (one too many sometimes) with values up to v_max, or rarely a huge value
static arma::uvec utuple(fz::input &in, arma::uword n_max, arma::uword v_max)
{
    arma::uvec t(in.size(n_max + 1));
    for (arma::uword &x : t)
        x = in.ConsumeIntegralInRange<int>(0, 31) == 0 ? (arma::uword(1) << 40) : in.ConsumeIntegralInRange<arma::uword>(0, v_max);
    return t;
}

// A real tuple of 0 to n_max + 1 entries with any values (NaN selects the default of an entry)
static arma::vec vtuple(fz::input &in, arma::uword n_max)
{
    return in.col<double>(in.size(n_max + 1));
}

template <typename T>
static void run(fz::input &in)
{
    const T res = resolution<T>(in);
    const bool c = coarse(res);

    switch (in.ConsumeIntegralInRange<int>(0, 10))
    {
    case 0:
        fz::clean([&] { quadriga_lib::generate_antenna<T>("omni", res); });
        break;
    case 1:
        fz::clean([&] { quadriga_lib::generate_antenna<T>("xpol", res); });
        break;
    case 2:
        fz::clean([&] { quadriga_lib::generate_antenna<T>("dipole", res); });
        break;
    case 3:
        fz::clean([&] { quadriga_lib::generate_antenna<T>("half-wave-dipole", res); });
        break;
    case 4:
    {
        T az = in.value<T>(), el = in.value<T>(), floor_dB = in.value<T>();
        fz::clean([&] { quadriga_lib::generate_antenna<T>("custom", res, NAN, {az, el, floor_dB}); });
        break;
    }
    case 5:
    {
        arma::uword N = in.size(c ? 8 : 2);
        double fc = in.value<double>(), spacing = in.value<double>();
        bool use;
        quadriga_lib::arrayant<T> p = pattern<T>(in, use);
        fz::clean([&] { quadriga_lib::generate_antenna<T>("ula", res, fc, {}, {1, N}, {spacing}, {}, {}, {}, {}, {}, {}, fz::opt(use, p)); });
        break;
    }
    case 6:
    {
        arma::uvec panel = utuple(in, 4, c ? 3 : 1), ports = utuple(in, 2, 3), pol = utuple(in, 2, 7);
        arma::vec spacing = vtuple(in, 4), element = vtuple(in, 5);
        arma::mat steer = in.mat<double>(in.size(3), in.size(3));
        double fc = in.ConsumeBool() ? 3.5e9 : in.value<double>();
        bool use, apply = in.ConsumeBool();
        quadriga_lib::arrayant<T> p = pattern<T>(in, use);
        fz::clean([&] { quadriga_lib::generate_antenna<T>("3GPP", res, fc, element, panel, spacing, pol, ports, steer, {}, {}, {}, fz::opt(use, p), false, apply); });
        break;
    }
    case 7:
    {
        arma::uvec panel = utuple(in, 4, c ? 3 : 1), pol = utuple(in, 2, 7);
        arma::vec spacing = vtuple(in, 4), element = vtuple(in, 3);
        arma::mat beams = in.mat<double>(in.size(4), in.size(4));
        double fc = in.ConsumeBool() ? 3.5e9 : in.value<double>();
        bool separate = in.ConsumeBool(), apply = in.ConsumeBool();
        fz::clean([&] { quadriga_lib::generate_antenna<T>("multibeam", res, fc, element, panel, spacing, pol, {}, {}, beams, {}, {}, {}, separate, apply); });
        break;
    }
    case 8:
    {
        const char *types[] = {"omni", "xpol", "dipole", "short-dipole", "half-wave-dipole", "custom", "ula", "3GPP", "3gpp", "multibeam", "3GPP-UT", "3GPP-CPE"};
        std::string type = in.ConsumeIntegralInRange<int>(0, 15) != 0 ? types[in.ConsumeIntegralInRange<int>(0, 11)] : in.ConsumeRandomLengthString(12);
        arma::uvec panel = utuple(in, 4, c ? 3 : 1), ports = utuple(in, 2, 3), pol = utuple(in, 2, 7);
        arma::vec spacing = vtuple(in, 4), element = vtuple(in, 5);
        arma::mat steer = in.mat<double>(in.size(3), in.size(3)), beams = in.mat<double>(in.size(4), in.size(4));
        double fc = in.ConsumeBool() ? 3.5e9 : in.value<double>();
        bool use, separate = in.ConsumeBool(), apply = in.ConsumeBool();
        quadriga_lib::arrayant<T> p = pattern<T>(in, use);
        fz::clean([&] { quadriga_lib::generate_antenna<T>(type, res, fc, element, panel, spacing, pol, ports, steer, beams, {}, {}, fz::opt(use, p), separate, apply); });
        break;
    }
    case 9:
    {
        const char *types[] = {"3GPP-UT", "3gpp-ut", "3GPP-CPE", "3gpp-cpe"};
        arma::uvec pol = utuple(in, 2, 7), locations = utuple(in, c ? 10 : 2, 10);
        arma::vec element = vtuple(in, 5), device_size = vtuple(in, 3);
        double fc = in.ConsumeBool() ? 3.5e9 : in.value<double>();
        bool use;
        quadriga_lib::arrayant<T> p = pattern<T>(in, use);
        const char *type = types[in.ConsumeIntegralInRange<int>(0, 3)];
        fz::clean([&] { quadriga_lib::generate_antenna<T>(type, res, fc, element, {}, {}, pol, {}, {}, {}, device_size, locations, fz::opt(use, p)); });
        break;
    }
    default:
    {
        const char *drivers[] = {"piston", "horn", "omni", ""}, *types[] = {"monopole", "hemisphere", "dipole", "cardioid", ""};
        std::string driver = in.ConsumeBool() ? drivers[in.ConsumeIntegralInRange<int>(0, 3)] : in.ConsumeRandomLengthString(10);
        std::string type = in.ConsumeBool() ? types[in.ConsumeIntegralInRange<int>(0, 4)] : in.ConsumeRandomLengthString(12);
        T radius = in.value<T>(), f_low = in.value<T>(), f_high = in.value<T>(), s_low = in.value<T>(), s_high = in.value<T>(),
          sens = in.value<T>(), hor = in.value<T>(), ver = in.value<T>(), f_ctrl = in.value<T>(), bw = in.value<T>(),
          bh = in.value<T>();
        if (in.ConsumeBool()) // cutoffs in the audible range give many third-octave bands
            f_low = in.ConsumeFloatingPointInRange<T>(T(10), T(30000)), f_high = in.ConsumeFloatingPointInRange<T>(T(10), T(30000));
        arma::Col<T> freq = in.col<T>(in.size(6));
        fz::clean([&] { quadriga_lib::generate_speaker<T>(driver, radius, f_low, f_high, s_low, s_high, sens, type, hor, ver,
                                                          f_ctrl, bw, bh, freq, res); });
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
