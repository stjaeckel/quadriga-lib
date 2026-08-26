// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef QD_ATMOSPHERIC_ABSORPTION_H
#define QD_ATMOSPHERIC_ABSORPTION_H

#include <cmath>

// Speed of sound in air at 20 degrees Celsius, ISO 9613-1 reference conditions [m/s]
static constexpr double qd_speed_of_sound = 342.76271;

// Speed of light in vacuum [m/s]
static constexpr double qd_speed_of_light = 299792458.0;

// Scalar (acoustic) mode maps an acoustic frequency onto an EM-equivalent frequency such that
// c_light / f_eq equals the acoustic wavelength. Wrappers convert at the API boundary; the core
// and the material model in ray_mesh_interact always work in EM-equivalent units.
//   f_eq = f_acoustic * qd_acoustic_freq_scale
static constexpr double qd_acoustic_freq_scale = qd_speed_of_light / qd_speed_of_sound;

// Atmospheric absorption coefficient alpha(f) in dB/m according to ISO 9613-1.
// Defaults: T = 20 degrees Celsius, RH = 50 %, p = 1 atm (indoor sea-level).
// Input frequency is given in Hertz.
static inline double qd_iso9613_alpha(double f_Hz,
                                      double T_celsius = 20.0,
                                      double RH_percent = 50.0,
                                      double p_kPa = 101.325)
{
    constexpr double T0 = 293.15;   // Reference temperature [K]
    constexpr double T01 = 273.16;  // Triple-point isotherm [K]
    constexpr double p_r = 101.325; // Reference pressure [kPa]

    const double T = T_celsius + 273.15; // Absolute temperature [K]

    // Saturation vapor pressure ratio and molar concentration of water vapor [%]
    const double C = -6.8346 * std::pow(T01 / T, 1.261) + 4.6151;
    const double psat = std::pow(10.0, C);
    const double h = RH_percent * psat / (p_kPa / p_r);

    // Relaxation frequencies for O2 and N2 [Hz]
    const double pa_pr = p_kPa / p_r;
    const double T_T0 = T / T0;
    const double f_rO = pa_pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
    const double f_rN = pa_pr * std::pow(T_T0, -0.5) * (9.0 + 280.0 * h * std::exp(-4.170 * (std::pow(T_T0, -1.0 / 3.0) - 1.0)));

    // Absorption coefficient [dB/m]
    const double f2 = f_Hz * f_Hz;
    const double classical = 1.84e-11 / pa_pr * std::sqrt(T_T0);
    const double rot_O = 0.01275 * std::exp(-2239.1 / T) / (f_rO + f2 / f_rO);
    const double rot_N = 0.1068 * std::exp(-3352.0 / T) / (f_rN + f2 / f_rN);

    return 8.686 * f2 * (classical + std::pow(T_T0, -2.5) * (rot_O + rot_N));
}

// Gain at 1 m distance in dB, i.e. the path-loss reference excluding the 20*log10(d) spreading term.
// EM mode: FSPL reference, -32.45 - 20*log10(f_GHz). Scalar (acoustic) mode: pure 1/r spreading, 0 dB.
// Input frequency is given in Hertz.
static inline double qd_gain_at_1m(double f_Hz, bool scalar_mode)
{
    return scalar_mode ? 0.0 : (-32.45 - 20.0 * std::log10(f_Hz * 1.0e-9));
}

#endif
