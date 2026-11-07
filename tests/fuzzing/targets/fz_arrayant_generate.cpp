// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Array antenna generators with fuzzed parameters: generate_arrayant_omni, _xpol, _dipole, _half_wave_dipole, _custom,
// _ula, _3GPP (also with a fuzzed custom pattern), _multibeam and generate_speaker

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
        fz::clean([&] { p = quadriga_lib::generate_arrayant_custom<T>(in.value<T>(), in.value<T>(), in.value<T>(), T(30)); });
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
            p.coupling_re = in.mat<T>(in.size(4), in.size(4)), p.coupling_im = in.mat<T>(in.size(4), in.size(4));
    }
    return p;
}

template <typename T>
static void run(fz::input &in)
{
    const T res = resolution<T>(in);
    const bool c = coarse(res);

    switch (in.ConsumeIntegralInRange<int>(0, 8))
    {
    case 0:
        fz::clean([&] { quadriga_lib::generate_arrayant_omni<T>(res); });
        break;
    case 1:
        fz::clean([&] { quadriga_lib::generate_arrayant_xpol<T>(res); });
        break;
    case 2:
        fz::clean([&] { quadriga_lib::generate_arrayant_dipole<T>(res); });
        break;
    case 3:
        fz::clean([&] { quadriga_lib::generate_arrayant_half_wave_dipole<T>(res); });
        break;
    case 4:
    {
        T az = in.value<T>(), el = in.value<T>(), rear = in.value<T>();
        fz::clean([&] { quadriga_lib::generate_arrayant_custom<T>(az, el, rear, res); });
        break;
    }
    case 5:
    {
        arma::uword N = in.size(c ? 8 : 2);
        T fc = in.value<T>(), spacing = in.value<T>();
        bool use;
        quadriga_lib::arrayant<T> p = pattern<T>(in, use);
        fz::clean([&] { quadriga_lib::generate_arrayant_ula<T>(N, fc, spacing, use ? &p : nullptr, res); });
        break;
    }
    case 6:
    {
        arma::uword M = in.size(c ? 3 : 1), N = in.size(c ? 3 : 1), Mg = in.size(c ? 2 : 1), Ng = in.size(c ? 2 : 1);
        T fc = in.value<T>(), tilt = in.value<T>(), spacing = in.value<T>(), dgv = in.value<T>(), dgh = in.value<T>();
        unsigned pol = in.ConsumeIntegralInRange<unsigned>(0, 7);
        bool use;
        quadriga_lib::arrayant<T> p = pattern<T>(in, use);
        fz::clean([&] { quadriga_lib::generate_arrayant_3GPP<T>(M, N, fc, pol, tilt, spacing, Mg, Ng, dgv, dgh,
                                                                use ? &p : nullptr, res); });
        break;
    }
    case 7:
    {
        arma::uword M = in.size(c ? 3 : 1), N = in.size(c ? 3 : 1);
        arma::uword n_beam = in.size(4);
        arma::Col<T> az = in.col<T>(n_beam), el = in.col<T>(in.ConsumeBool() ? n_beam : in.size(4)),
                     w = in.col<T>(in.ConsumeBool() ? n_beam : in.size(4));
        T fc = in.value<T>(), spacing = in.value<T>(), az3 = in.value<T>(), el3 = in.value<T>(), rear = in.value<T>();
        unsigned pol = in.ConsumeIntegralInRange<unsigned>(0, 4);
        bool separate = in.ConsumeBool(), apply = in.ConsumeBool();
        fz::clean([&] { quadriga_lib::generate_arrayant_multibeam<T>(M, N, az, el, w, fc, pol, spacing, az3, el3, rear,
                                                                     res, separate, apply); });
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
