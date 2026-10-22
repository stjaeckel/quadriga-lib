// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_tools.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <complex>
#include <cmath>

// Self-contained ITU-R P.2040 transition-gain reference. This does NOT call into
// quadriga-lib; it is an independent implementation used as ground truth so the
// tests validate the library blind rather than against its own output.
static inline double calc_transition_gain(int interaction_type,       // (0) Reflection, (1) Transmission, (2) Refraction
                                          double incidence_angle_deg, // Angle between face normal and ray (as in ITU P.2040-1) (degree)
                                          double dist1,               // Medium 1 travel distance (meters)
                                          double dist2,               // Medium 2 travel distance (meters) OR distance after reflection
                                          std::complex<double> eta1,  // relative permittivity of medium 1
                                          std::complex<double> eta2)  // relative permittivity of medium 2
{
    double deg2rad = arma::datum::pi / 180.0;

    // Calculate gain from ITU-R P.2040:
    double cos_th = std::cos(incidence_angle_deg * deg2rad); // Incidence on boundary
    double sin_th = std::sqrt(1.0 - cos_th * cos_th);        // Trigonometric identity
    std::complex<double> cos_th2 = std::sqrt(1.0 - eta1 / eta2 * sin_th * sin_th);

    // Medium 1 loss
    double tan_delta = std::imag(eta1) / std::real(eta1); // Loss tangent, Rec. ITU-R P.2040-1, eq. (13)
    double cos_delta = std::cos(std::atan(tan_delta));
    double Delta = 2.0 * cos_delta / (1.0 - cos_delta);
    Delta = std::sqrt(Delta) * 0.0477135 / (10.0 * std::sqrt(std::real(eta1)));
    double A = 8.686 * dist1 / Delta;                // Attenuation in db/m, ITU-R P.2040-1, eq. (26)
    double medium_1_gain = std::pow(10.0, -0.1 * A); // Gain caused by conductive medium in linear scale

    // Medium 2 loss
    if (interaction_type != 0) // Use eta1 for reflection
    {
        tan_delta = std::imag(eta2) / std::real(eta2); // Loss tangent, Rec. ITU-R P.2040-1, eq. (13)
        cos_delta = std::cos(std::atan(tan_delta));
        Delta = 2.0 * cos_delta / (1.0 - cos_delta);
        Delta = std::sqrt(Delta) * 0.0477135 / (10.0 * std::sqrt(std::real(eta2)));
    }
    A = 8.686 * dist2 / Delta;                       // Attenuation in db/m, ITU-R P.2040-1, eq. (26)
    double medium_2_gain = std::pow(10.0, -0.1 * A); // Gain caused by conductive medium in linear scale

    eta1 = std::sqrt(eta1);
    eta2 = std::sqrt(eta2);

    // Reflection coefficients
    std::complex<double> R_te = (eta1 * cos_th - eta2 * cos_th2) / (eta1 * cos_th + eta2 * cos_th2);
    std::complex<double> R_tm = (eta2 * cos_th - eta1 * cos_th2) / (eta2 * cos_th + eta1 * cos_th2);

    // Transmission coefficients
    std::complex<double> T_te = (2.0 * eta1 * cos_th) / (eta1 * cos_th + eta2 * cos_th2);
    std::complex<double> T_tm = (2.0 * eta1 * cos_th) / (eta2 * cos_th + eta1 * cos_th2);

    double reflection_gain = 0.5 * (std::norm(R_te) + std::norm(R_tm));
    double refraction_gain = 0.5 * (std::norm(T_te) + std::norm(T_tm));

    double total_gain = 0.0;
    if (interaction_type == 0) // Reflection
        total_gain = medium_1_gain * reflection_gain * medium_2_gain;
    else if (interaction_type == 1) // Transmission
        total_gain = medium_1_gain * (1.0 - reflection_gain) * medium_2_gain;
    else if (interaction_type == 2) // Refraction
        total_gain = medium_1_gain * refraction_gain * medium_2_gain;

    return total_gain;
}

// Convert a per-face material matrix [n_face, 9] with columns
// {a,b,c,d,att,attB,alpha,alphaB,fRef} into the (mtl_ind, mtl_prop-map) pair.
// Identical rows are deduplicated and mtl_ind is 1-based (0 = no material), so the
// result matches what obj_file_read emits and the current calc_diffraction_gain API.
static inline void mtl_matrix_to_map(const arma::mat &M,
                                     arma::uvec &mtl_ind,
                                     std::unordered_map<std::string, std::vector<double>> &mtl_prop)
{
    static const char *names[9] = {"a", "b", "c", "d", "att", "attB", "alpha", "alphaB", "fRef"};
    const arma::uword n_face = M.n_rows;

    mtl_ind.set_size(n_face);
    std::vector<arma::uword> uniq; // row index of each distinct material
    for (arma::uword f = 0; f < n_face; ++f)
    {
        arma::uword m = 0;
        bool found = false;
        for (; m < uniq.size(); ++m)
            if (arma::approx_equal(M.row(f), M.row(uniq[m]), "absdiff", 0.0))
            {
                found = true;
                break;
            }
        if (!found)
        {
            m = (arma::uword)uniq.size();
            uniq.push_back(f);
        }
        mtl_ind(f) = m + 1;
    }

    mtl_prop.clear();
    for (int c = 0; c < 9; ++c)
    {
        std::vector<double> col(uniq.size());
        for (size_t m = 0; m < uniq.size(); ++m)
            col[m] = M.at(uniq[m], c);
        mtl_prop[names[c]] = std::move(col);
    }
}

// Fresnel field reflection coefficients for a medium-1 -> medium-2 interface, using
// sqrt(eta) as the wave impedance (non-magnetic media). theta is the incidence angle
// measured from the surface normal (radians). Also returns cos(theta_t) inside medium 2.
static inline void fresnel(std::complex<double> eta1, std::complex<double> eta2, double theta,
                           std::complex<double> &R_te, std::complex<double> &R_tm,
                           std::complex<double> &cos_tt)
{
    double cos_th = std::cos(theta), sin_th = std::sin(theta);
    cos_tt = std::sqrt(1.0 - (eta1 / eta2) * sin_th * sin_th);
    std::complex<double> n1 = std::sqrt(eta1), n2 = std::sqrt(eta2);
    R_te = (n1 * cos_th - n2 * cos_tt) / (n1 * cos_th + n2 * cos_tt);
    R_tm = (n2 * cos_th - n1 * cos_tt) / (n2 * cos_th + n1 * cos_tt);
}

// Geometry (path coordinates) and normal-incidence transmission against the ITU reference.
// Ported to the new API: inputs by reference, the xprmat output occupies the slot between
// gain and coord (passed as nullptr here). Coordinates are pure ellipsoid geometry and are
// unchanged by the physics refactor; the transmission case is normal incidence into a single
// interface (dest inside the cube), where refraction and Fabry-Perot are dormant, so the
// value still equals the independent reference.
TEST_CASE("Calc Diffraction Gain")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});

    arma::mat mtl_prop, orig, dest;
    arma::vec gain, tv;
    arma::cube coord, tc;

    mtl_prop = {{1.5, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    // Entire path outside
    orig = {{-10.0, 0.0, 0.5}};
    dest = {{-5.0, 0.0, 0.5}};
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 1.0e9, 1, &gain, nullptr, &coord, 0);

    tv = {1.0};
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-13));

    tc.set_size(3, 2, 1);
    tc.slice(0) = {{-8.75, -6.25}, {0.0, 0.0}, {0.5, 0.5}};
    CHECK(arma::approx_equal(coord, tc, "absdiff", 1e-13));

    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 1.0e9, 2, &gain, nullptr, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-13));
    CHECK(arma::approx_equal(coord, tc, "absdiff", 1e-13));

    tc.set_size(3, 3, 1);
    tc.slice(0) = {{-10.0 + 5.0 * 0.1464, -7.5, -10.0 + 5.0 * 0.8536}, {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 1.0e9, 3, &gain, nullptr, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-13));
    CHECK(arma::approx_equal(coord, tc, "absdiff", 1e-13));

    tc.set_size(3, 4, 1);
    tc.slice(0) = {{-10.0 + 5.0 * 0.0955, -10.0 + 5.0 * 0.3455, -10.0 + 5.0 * 0.6545, -10.0 + 5.0 * 0.9045}, {0.0, 0.0, 0.0, 0.0}, {0.5, 0.5, 0.5, 0.5}};
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 1.0e9, 4, &gain, nullptr, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-13));
    CHECK(arma::approx_equal(coord, tc, "absdiff", 1e-13));

    // Single path outside to inside
    std::complex<double> eta1(1.0, 0.0);                                // Air
    std::complex<double> eta2(mtl_prop(0, 0), -1.798 * mtl_prop(0, 2)); // @ 10 GHz

    double total_gain = calc_transition_gain(1, 0.0, 1.0, 1.5, eta1, eta2);

    tv = {total_gain};
    orig = {{-10.0, 0.0, 0.5}};
    dest = {{0.5, 0.0, 0.5}};
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 0, &gain, nullptr, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-7));

    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 5, &gain, nullptr, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-7));

    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 6, &gain, nullptr, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-7));

    // 2 segments, (1) outside to inside, (2) inside
    orig = {{-1.5, 0.0, 0.5}};
    dest = {{0.5, 0.0, 0.5}};
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 5, &gain, nullptr, &coord);
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-7));
}

// In-medium absorption via the explicit alpha (dB/m) column. eps=1 (no Fresnel), normal
// incidence, n=1 so refraction is a no-op and the refracted path equals the geometric 1.5 m.
TEST_CASE("Calc Diffraction Gain - Alpha in-medium absorption")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});

    // eps_r = 1 (no Fresnel), sigma = 0, alpha = 4 dB/m, all exponents 0, fRef = 1
    arma::mat mtl_prop = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{0.5, 0.0, 0.5}};

    arma::vec gain;
    quadriga_lib::calc_diffraction_gain(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 0, &gain);

    // 1.5 m * 4 dB/m = 6 dB  ->  gain = 10^-0.6
    arma::vec tv = {std::pow(10.0, -0.1 * 4.0 * 1.5)};
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-7));
}

// Penetration loss (att, dB) with power-law frequency scaling: att=3 dB at fRef=2 GHz,
// attB=1  ->  at 10 GHz: 3*(10/2)^1 = 15 dB. eps=1 so no Fresnel term is mixed in.
TEST_CASE("Calc Diffraction Gain - Penetration loss frequency scaling")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});

    arma::mat mtl_prop = {{1.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{0.5, 0.0, 0.5}};

    arma::vec gain;
    quadriga_lib::calc_diffraction_gain(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 0, &gain);

    arma::vec tv = {std::pow(10.0, -1.5)}; // 10^-1.5
    CHECK(arma::approx_equal(gain, tv, "absdiff", 1e-10));
}

// Two materials parameterized at different reference frequencies but numerically identical
// at every frequency must produce identical gain. Pure invariance: robust to the physics
// details. lod=3 exercises the full multi-arc, multi-hit ray-state machine.
TEST_CASE("Calc Diffraction Gain - fRef parameterization equivalence")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});

    // At every f:  eps_r = 1.5*f,  sigma = 0.001*f,  att = 2*f dB,  alpha = 0.5*f dB/m
    arma::mat mat_A = {{1.5, 1.0, 0.001, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0}}; // fRef = 1
    arma::mat mat_B = {{3.0, 1.0, 0.002, 1.0, 4.0, 1.0, 1.0, 1.0, 2.0}}; // fRef = 2
    mat_A = repmat(mat_A, 12, 1);
    mat_B = repmat(mat_B, 12, 1);
    arma::uvec mtl_ind_A, mtl_ind_B;
    std::unordered_map<std::string, std::vector<double>> mtl_map_A, mtl_map_B;
    mtl_matrix_to_map(mat_A, mtl_ind_A, mtl_map_A);
    mtl_matrix_to_map(mat_B, mtl_ind_B, mtl_map_B);

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{0.5, 0.0, 0.5}};

    arma::vec gain_A, gain_B;
    quadriga_lib::calc_diffraction_gain(orig, dest, cube, mtl_ind_A, mtl_map_A, 10.0e9, 3, &gain_A);
    quadriga_lib::calc_diffraction_gain(orig, dest, cube, mtl_ind_B, mtl_map_B, 10.0e9, 3, &gain_B);

    CHECK(arma::approx_equal(gain_A, gain_B, "absdiff", 1e-12));
}

// Scalar transmission (interaction_type 4) vs EM TE/TM averaging (type 1). Single wall
// crossing (dest inside), so refraction/Fabry-Perot stay dormant and the energy-conserving
// partition is exercised in isolation. Migrated: xprmat nullptr slot inserted between gain
// and coord in the fully positional call.
TEST_CASE("Calc Diffraction Gain - Scalar mode")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});

    // eps_r = 2 (light->dense boundary) + att = 6 dB isolation. With the energy-conserving
    // partition the scalar path carries the same Fresnel boundary loss (1 - |R_TE(theta)|^2)
    // as EM on a light->dense crossing. Only eps<1 (dense->light) stays pass-through.
    arma::mat mtl_prop = {{2.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0}}; // col 4 = att
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    arma::vec g_em_n, g_sc_n, g_em_o, g_sc_o;

    // Normal incidence: dest INSIDE the cube -> single wall crossing
    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{0.5, 0.0, 0.5}};
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 0,
                                                &g_em_n, nullptr, nullptr, 0, nullptr, 0, 0, false);
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 0,
                                                &g_sc_n, nullptr, nullptr, 0, nullptr, 0, 0, true);

    // Oblique incidence (~39 deg off normal at the west wall): dest still INSIDE -> single crossing
    orig = {{-10.0, -8.0, 0.5}};
    dest = {{0.5, 0.5, 0.5}};
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 0,
                                                &g_em_o, nullptr, nullptr, 0, nullptr, 0, 0, false);
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 0,
                                                &g_sc_o, nullptr, nullptr, 0, nullptr, 0, 0, true);

    // Scalar transmission conserves energy on light->dense: gain = att_lin*(1 - |R_TE(theta)|^2),
    // angle-dependent, and equal to EM at normal incidence (TE == TM there).
    double att_lin = std::pow(10.0, -0.6); // -6 dB isolation
    double s = std::sqrt(2.0);             // sqrt(eps_r)
    double R0 = (1.0 - s) / (1.0 + s);     // normal-incidence reflection coefficient
    double g_expect_n = att_lin * (1.0 - R0 * R0);

    CHECK(std::abs(g_sc_n(0) - g_expect_n) < 1e-9); // exact at normal incidence
    CHECK(std::abs(g_sc_n(0) - g_em_n(0)) < 1e-9);  // scalar == EM at normal (TE == TM)
    CHECK(g_sc_o(0) > 0.0);
    CHECK(g_sc_o(0) < g_sc_n(0)); // more reflection at oblique -> less transmission

    // EM also loses more toward oblique; at oblique it transmits slightly more than scalar
    // because the TM component reflects less than TE (only TE enters the scalar coefficient).
    CHECK(g_em_n(0) > 0.0);
    CHECK(g_em_o(0) < g_em_n(0));
    CHECK(g_em_o(0) > g_sc_o(0));
}

// Unpack one interleaved-complex column of an xprmat into a Jones entry. Row pairs are
// (Re, Im). For EM [8,n_pos] the column order is VV, HV, VH, HH (col-major 2x2): use base
// 0, 2, 4, 6. For scalar [2,n_pos] the single coefficient is base 0.
static inline std::complex<double> xpr_cx(const arma::mat &X, arma::uword base, arma::uword col)
{
    return std::complex<double>(X(base, col), X(base + 1, col));
}

// Unobstructed path: the polarization transfer matrix is initialized to unity and never
// touched, so after gain normalization EM returns the exact identity Jones matrix and scalar
// returns (1,0). This pins the output layout and the "no interaction -> identity" contract.
TEST_CASE("Calc Diffraction Gain - xprmat identity on clear path")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});
    arma::mat mtl_prop = {{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    // Path entirely outside the cube -> no interaction
    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{-5.0, 0.0, 0.5}};

    // EM mode: xprmat is [8,1], identity
    arma::vec gain;
    arma::mat xpr;
    quadriga_lib::calc_diffraction_gain(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 0, &gain, &xpr);

    CHECK(xpr.n_rows == 8);
    CHECK(xpr.n_cols == 1);
    CHECK(std::abs(gain(0) - 1.0) < 1e-12);

    std::complex<double> VV = xpr_cx(xpr, 0, 0), HV = xpr_cx(xpr, 2, 0);
    std::complex<double> VH = xpr_cx(xpr, 4, 0), HH = xpr_cx(xpr, 6, 0);
    CHECK(std::abs(VV - std::complex<double>(1.0, 0.0)) < 1e-12);
    CHECK(std::abs(HH - std::complex<double>(1.0, 0.0)) < 1e-12);
    CHECK(std::abs(HV) < 1e-12);
    CHECK(std::abs(VH) < 1e-12);

    // Scalar mode: xprmat is [2,1], value (1,0)
    arma::vec gain_s;
    arma::mat xpr_s;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 0,
                                                &gain_s, &xpr_s, nullptr, 0, nullptr, 0, 0, true);
    CHECK(xpr_s.n_rows == 2);
    CHECK(xpr_s.n_cols == 1);
    CHECK(std::abs(gain_s(0) - 1.0) < 1e-12);
    CHECK(std::abs(xpr_cx(xpr_s, 0, 0) - std::complex<double>(1.0, 0.0)) < 1e-12);
}

// Normal-incidence transmission into a lossless dielectric (eps_r=2). There is real Fresnel
// loss so gain < 1, but TE == TM at normal incidence, so after gain normalization the
// transfer matrix is still the identity (no depolarization). This is the core contract:
// xprmat encodes polarization coupling, not total power.
TEST_CASE("Calc Diffraction Gain - xprmat normal incidence carries no depolarization")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});
    // eps_r = 2, lossless (sigma=0, att=0, alpha=0)
    arma::mat mtl_prop = {{2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{0.5, 0.0, 0.5}}; // along x -> normal incidence, dest inside

    arma::vec gain;
    arma::mat xpr;
    quadriga_lib::calc_diffraction_gain(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 0, &gain, &xpr);

    // Fresnel power transmission at normal incidence
    double R0 = (1.0 - std::sqrt(2.0)) / (1.0 + std::sqrt(2.0));
    CHECK(std::abs(gain(0) - (1.0 - R0 * R0)) < 1e-9);

    std::complex<double> VV = xpr_cx(xpr, 0, 0), HV = xpr_cx(xpr, 2, 0);
    std::complex<double> VH = xpr_cx(xpr, 4, 0), HH = xpr_cx(xpr, 6, 0);

    // Normalized diagonal has unit magnitude; off-diagonal is zero
    CHECK(std::abs(std::abs(VV) - 1.0) < 1e-9);
    CHECK(std::abs(std::abs(HH) - 1.0) < 1e-9);
    CHECK(std::abs(HV) < 1e-9);
    CHECK(std::abs(VH) < 1e-9);

    // VV and HH are identical (TE == TM): ratio is real, equal to 1
    std::complex<double> ratio = VV / HH;
    CHECK(std::abs(ratio - std::complex<double>(1.0, 0.0)) < 1e-9);

    // Normalization identity: 0.5 * sum|xprmat|^2 == 1
    double p = 0.5 * (std::norm(VV) + std::norm(HV) + std::norm(VH) + std::norm(HH));
    CHECK(std::abs(p - 1.0) < 1e-9);
}

// Oblique transmission into a lossless dielectric, closed form. The ray lies in a horizontal
// plane (dz=0) and the west-wall normal is horizontal, so the plane of incidence is horizontal
// and V=z is perpendicular to it. In the Ludwig-3 basis this makes the Jones matrix diagonal:
// VV carries the TE coefficient, HH the TM coefficient, and the cross terms vanish exactly.
// Expected magnitudes follow from Fresnel + the gain-normalization identity, so nothing is
// taken from a reference run.
TEST_CASE("Calc Diffraction Gain - xprmat oblique depolarization")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});
    arma::mat mtl_prop = {{2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}; // eps_r=2, lossless
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    arma::mat orig = {{-10.0, -8.0, 0.5}};
    arma::mat dest = {{0.5, 0.5, 0.5}}; // dz=0; enters west wall, dest inside

    arma::vec gain;
    arma::mat xpr;
    quadriga_lib::calc_diffraction_gain(orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 0, &gain, &xpr);

    // Incidence angle from the horizontal ray direction against the x-normal wall
    double dx = 10.5, dy = 8.5;
    double theta = std::acos(std::abs(dx) / std::sqrt(dx * dx + dy * dy));

    std::complex<double> R_te, R_tm, cos_tt;
    fresnel(std::complex<double>(1.0, 0.0), std::complex<double>(2.0, 0.0), theta, R_te, R_tm, cos_tt);
    double TE2 = 1.0 - std::norm(R_te); // energy transmission, TE
    double TM2 = 1.0 - std::norm(R_tm); // energy transmission, TM
    double g = 0.5 * (TE2 + TM2);       // expected EM gain

    CHECK(std::abs(gain(0) - g) < 1e-6);

    std::complex<double> VV = xpr_cx(xpr, 0, 0), HV = xpr_cx(xpr, 2, 0);
    std::complex<double> VH = xpr_cx(xpr, 4, 0), HH = xpr_cx(xpr, 6, 0);

    // The library normalizes each column to unit largest singular value (for a diagonal matrix,
    // the max column 2-norm). HH (TM) transmits more here, so |HH| = 1 and |VV| = sqrt(TE2/TM2).
    CHECK(std::abs(std::abs(HH) - 1.0) < 1e-6);
    CHECK(std::abs(std::abs(VV) - std::sqrt(TE2 / TM2)) < 1e-6);
    CHECK(std::abs(VV) < std::abs(HH));

    // Cross terms vanish (decoupled geometry)
    CHECK(std::abs(HV) < 1e-9);
    CHECK(std::abs(VH) < 1e-9);

    // Ratio is real (common propagation phase cancels) and equals sqrt(TE2/TM2)
    std::complex<double> ratio = VV / HH;
    CHECK(std::abs(std::imag(ratio)) < 1e-9);
    CHECK(std::abs(std::real(ratio) - std::sqrt(TE2 / TM2)) < 1e-6);

    // Unit largest singular value (diagonal case: max column norm == 1)
    CHECK(std::abs(std::max(std::abs(VV), std::abs(HH)) - 1.0) < 1e-6);
}

// Analytic thin-slab (Fabry-Perot) resolution. A lossless slab of two parallel faces is
// tuned to half-wave (beta=pi) and quarter-wave (beta=pi/2) optical thickness at normal
// incidence. With resolution ON (thin_slab_threshold=0) the gain follows the closed-form
// Airy transmittance T = 1/(1 + F*sin^2(beta)), F = 4R/(1-R)^2: ~1 at half-wave (resonant
// transparency) and (1-R)^2/(1+R)^2 at quarter-wave. With resolution OFF (threshold=1) the
// internal interference is discarded, so the two thicknesses give the *same* incoherent gain.
TEST_CASE("Calc Diffraction Gain - Fabry-Perot thin-slab resolution")
{
    const double c = 299792458.0;
    const double f = 10.0e9;
    const double lambda0 = c / f;
    const double n = 1.5; // eps_r = 2.25
    const double R = std::pow((n - 1.0) / (n + 1.0), 2.0);
    const double F = 4.0 * R / std::pow(1.0 - R, 2.0);

    const double t_half = lambda0 / (2.0 * n);    // beta = pi
    const double t_quarter = lambda0 / (4.0 * n); // beta = pi/2

    arma::mat mtl_prop = {{n * n, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}; // eps_r=2.25, lossless
    mtl_prop = repmat(mtl_prop, 12, 1);

    auto run = [&](double thickness, double threshold) -> double
    {
        arma::mat slab = quadriga_lib::cube<double>({thickness / 2.0, 5.0, 5.0}, {}, {thickness / 2.0, 0.0, 2.001});
        arma::uvec mtl_ind;
        std::unordered_map<std::string, std::vector<double>> mtl_map;
        mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);
        arma::mat orig = {{-10.0, 0.0, 0.0}};
        arma::mat dest = {{10.0, 0.0, 0.0}}; // through both faces, normal incidence
        arma::vec gain;
        quadriga_lib::calc_diffraction_gain<double>(orig, dest, slab, mtl_ind, mtl_map, f, 0,
                                                    &gain, nullptr, nullptr, 0, nullptr, 0, 0, false, threshold);
        return gain(0);
    };

    double g_half_on = run(t_half, 0.0);
    double g_quarter_on = run(t_quarter, 0.0);
    double g_half_off = run(t_half, 1.0);
    double g_quarter_off = run(t_quarter, 1.0);

    auto airy = [&](double beta)
    { return 1.0 / (1.0 + F * std::pow(std::sin(beta), 2.0)); };

    // Resolution ON: closed-form Airy transmittance
    CHECK(std::abs(g_half_on - airy(arma::datum::pi)) < 1e-5);          // ~1.0
    CHECK(std::abs(g_quarter_on - airy(arma::datum::pi / 2.0)) < 1e-5); // (1-R)^2/(1+R)^2

    // Resolution OFF: incoherent, so independent of slab thickness
    CHECK(std::abs(g_half_off - g_quarter_off) < 1e-9);

    // Interference signature: resonance enhances above the incoherent value, anti-resonance
    // suppresses below it. Robust to whichever incoherent convention OFF uses.
    CHECK(g_half_on > g_half_off + 1e-3);
    CHECK(g_quarter_on < g_quarter_off - 1e-3);
}

// Refraction path-length: a thick, strongly absorbing slab (two parallel faces) at normal
// and 45-degree incidence. The in-medium absorption acts over the REFRACTED path t/cos(theta_t),
// with cos(theta_t) from Snell's law, not the geometric chord. Absorption is heavy enough
// (~20 dB one way) that internal reflections are negligible, so with resolution never
// (threshold=1) the result is a clean single pass. Scalar mode keeps a single coefficient.
TEST_CASE("Calc Diffraction Gain - Refraction in-medium path length")
{
    const double f = 10.0e9;
    const double eps_r = 2.25;
    const double alpha = 40.0; // dB/m -> ~20 dB one-way over t
    const double t = 0.5;      // slab thickness along x (m)

    arma::mat mtl_prop = {{eps_r, 0.0, 0.0, 0.0, 0.0, 0.0, alpha, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);

    arma::mat slab = quadriga_lib::cube<double>({t / 2.0, 5.0, 5.0}, {}, {t / 2.0, 0.0, 2.001});
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    auto run = [&](const arma::mat &orig, const arma::mat &dest) -> double
    {
        arma::vec gain;
        quadriga_lib::calc_diffraction_gain<double>(orig, dest, slab, mtl_ind, mtl_map, f, 0,
                                                    &gain, nullptr, nullptr, 0, nullptr, 0, 0, true, 1.0);
        return gain(0);
    };

    // Closed-form single pass: (1 - |R_TE(theta)|^2)^2 * 10^(-0.1 * alpha * t/cos(theta_t))
    auto expect = [&](double theta) -> double
    {
        std::complex<double> R_te, R_tm, cos_tt;
        fresnel(std::complex<double>(1.0, 0.0), std::complex<double>(eps_r, 0.0), theta, R_te, R_tm, cos_tt);
        double L = t / std::real(cos_tt);
        double interface = std::pow(1.0 - std::norm(R_te), 2.0);
        return interface * std::pow(10.0, -0.1 * alpha * L);
    };

    // Normal incidence: theta_t = 0, L = t
    double g0 = run({{-10.0, 0.0, 0.0}}, {{10.0, 0.0, 0.0}});
    CHECK(std::abs(g0 - expect(0.0)) < 1e-5);

    // 45-degree incidence (direction (1,1,0)): refraction lengthens the in-medium path
    double g45 = run({{-10.0, -10.0, 0.0}}, {{10.0, 10.0, 0.0}});
    CHECK(std::abs(g45 - expect(arma::datum::pi / 4.0)) < 1e-5);

    // Oblique loses more: stronger reflection plus a longer refracted path
    CHECK(g45 < g0);
}

// Multi-material stack at normal incidence with resolution never (threshold=1). Two adjacent
// lossless slabs (eps_r=2.25 then eps_r=4) share a coincident interface, so the ray crosses
// three boundaries: air->mat1, mat1->mat2, mat2->air. Without cavity resolution the gain is
// the incoherent product of the per-interface transmissions (1 - |R|^2). This assumes the two
// opposite-facing coincident faces collapse to a single mat1->mat2 interface (no phantom air
// gap). Only threshold=1 is meaningful here: the recursive Fabry-Perot cannot be resolved
// across more than one layer.
TEST_CASE("Calc Diffraction Gain - Multi-material stack")
{
    const double f = 10.0e9;
    const double n1 = 1.5; // eps_r = 2.25
    const double n2 = 2.0; // eps_r = 4
    const double t1 = 0.010, t2 = 0.010;

    arma::mat box1 = quadriga_lib::cube<double>({t1 / 2.0, 5.0, 5.0}, {}, {t1 / 2.0, 0.0, 2.001});
    arma::mat box2 = quadriga_lib::cube<double>({t2 / 2.0, 5.0, 5.0}, {}, {t1 + t2 / 2.0, 0.0, 2.001});
    arma::mat mesh = arma::join_cols(box1, box2);

    arma::mat p1 = {{n1 * n1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::mat p2 = {{n2 * n2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::mat mtl_prop = arma::join_cols(repmat(p1, 12, 1), repmat(p2, 12, 1));
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    arma::mat orig = {{-10.0, 0.0, 0.0}};
    arma::mat dest = {{20.0, 0.0, 0.0}}; // through both slabs, normal incidence

    arma::vec gain;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, mesh, mtl_ind, mtl_map, f, 0,
                                                &gain, nullptr, nullptr, 0, nullptr, 0, 0, false, 1.0);

    // Product of interface transmissions (normal incidence, TE == TM)
    double Ra = std::pow((1.0 - n1) / (1.0 + n1), 2.0); // air -> mat1
    double Rb = std::pow((n1 - n2) / (n1 + n2), 2.0);   // mat1 -> mat2
    double Rc = std::pow((n2 - 1.0) / (n2 + 1.0), 2.0); // mat2 -> air
    double g_expect = (1.0 - Ra) * (1.0 - Rb) * (1.0 - Rc);

    CHECK(std::abs(gain(0) - g_expect) < 1e-6);
}

// A material index of 0 means "no material": the face is intersected geometrically but applies
// no transition. A cube with all indices 0 is fully transparent (gain exactly 1), while the
// same geometry with a real lossy material clearly attenuates. This proves transparency comes
// from the 0 index, not from the ray missing the mesh.
TEST_CASE("Calc Diffraction Gain - Zero material index is transparent")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});

    // Build a valid map (one material) but assign index 0 to every face
    arma::mat dummy = {{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    arma::uvec tmp;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(dummy, tmp, mtl_map);
    arma::uvec mtl_ind_zero = arma::zeros<arma::uvec>(12);

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{10.0, 0.0, 0.5}}; // straight through both walls

    arma::vec gain_zero;
    quadriga_lib::calc_diffraction_gain(orig, dest, cube, mtl_ind_zero, mtl_map, 10.0e9, 0, &gain_zero);
    CHECK(std::abs(gain_zero(0) - 1.0) < 1e-13);

    // Same geometry, real material (att = 6 dB per crossing, eps_r = 1 so no Fresnel)
    arma::mat lossy = {{1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0}};
    lossy = repmat(lossy, 12, 1);
    arma::uvec mtl_ind_l;
    std::unordered_map<std::string, std::vector<double>> mtl_map_l;
    mtl_matrix_to_map(lossy, mtl_ind_l, mtl_map_l);

    arma::vec gain_lossy;
    quadriga_lib::calc_diffraction_gain(orig, dest, cube, mtl_ind_l, mtl_map_l, 10.0e9, 0, &gain_lossy);
    CHECK(gain_lossy(0) < 0.5); // two crossings of 6 dB -> clearly attenuated
}

// Input validation. Each malformed argument must raise std::invalid_argument. Ordering follows
// the guards in the implementation; lod is validated downstream in generate_diffraction_paths
// and is not exercised here.
TEST_CASE("Calc Diffraction Gain - Input validation")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    arma::mat mtl_prop = repmat(arma::mat({{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}), 12, 1);
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{0.5, 0.0, 0.5}};
    arma::vec gain;

    // orig without 3 columns
    arma::mat bad_orig = {{-10.0, 0.0}};
    CHECK_THROWS_AS(quadriga_lib::calc_diffraction_gain(bad_orig, dest, cube, mtl_ind, mtl_map, 10.0e9, 0, &gain), std::invalid_argument);

    // dest without 3 columns
    arma::mat bad_dest = {{0.5, 0.0}};
    CHECK_THROWS_AS(quadriga_lib::calc_diffraction_gain(orig, bad_dest, cube, mtl_ind, mtl_map, 10.0e9, 0, &gain), std::invalid_argument);

    // mesh without 9 columns
    arma::mat bad_mesh(5, 8, arma::fill::zeros);
    CHECK_THROWS_AS(quadriga_lib::calc_diffraction_gain(orig, dest, bad_mesh, mtl_ind, mtl_map, 10.0e9, 0, &gain), std::invalid_argument);

    // dest row count not matching orig
    arma::mat dest2 = {{0.5, 0.0, 0.5}, {0.6, 0.1, 0.5}};
    CHECK_THROWS_AS(quadriga_lib::calc_diffraction_gain(orig, dest2, cube, mtl_ind, mtl_map, 10.0e9, 0, &gain), std::invalid_argument);

    // mtl_ind length not matching face count
    arma::uvec mtl_ind_short = arma::ones<arma::uvec>(5);
    CHECK_THROWS_AS(quadriga_lib::calc_diffraction_gain(orig, dest, cube, mtl_ind_short, mtl_map, 10.0e9, 0, &gain), std::invalid_argument);

    // Non-positive center frequency
    CHECK_THROWS_AS(quadriga_lib::calc_diffraction_gain(orig, dest, cube, mtl_ind, mtl_map, 0.0, 0, &gain), std::invalid_argument);

    // Material index exceeding 16383, the 14-bit index of the state words
    arma::uvec mtl_ind_big = mtl_ind;
    mtl_ind_big(0) = 40000;
    CHECK_THROWS_AS(quadriga_lib::calc_diffraction_gain(orig, dest, cube, mtl_ind_big, mtl_map, 10.0e9, 0, &gain), std::invalid_argument);
}

// Single-precision instantiation smoke test. Same alpha-absorption geometry as the double
// case (eps_r=1, 4 dB/m over 1.5 m -> 10^-0.6), run through the float template with a loose
// tolerance to confirm the float path compiles and produces the expected value.
TEST_CASE("Calc Diffraction Gain - Float instantiation")
{
    arma::fmat cube = quadriga_lib::cube<float>({}, {}, {0.0f, 0.0f, 0.001f});
    arma::uvec mtl_ind = arma::ones<arma::uvec>(12);

    std::unordered_map<std::string, std::vector<float>> mtl_map;
    mtl_map["a"] = {1.0f};
    mtl_map["b"] = {0.0f};
    mtl_map["c"] = {0.0f};
    mtl_map["d"] = {0.0f};
    mtl_map["att"] = {0.0f};
    mtl_map["attB"] = {0.0f};
    mtl_map["alpha"] = {4.0f};
    mtl_map["alphaB"] = {0.0f};
    mtl_map["fRef"] = {1.0f};

    arma::fmat orig = {{-10.0f, 0.0f, 0.5f}};
    arma::fmat dest = {{0.5f, 0.0f, 0.5f}};

    arma::fvec gain;
    quadriga_lib::calc_diffraction_gain<float>(orig, dest, cube, mtl_ind, mtl_map, 10.0e9f, 0, &gain);

    CHECK(std::abs(gain(0) - std::pow(10.0f, -0.6f)) < 1e-5f);
}
// Independent ISO 9613-1 air-absorption reference, transcribed from the standard rather than
// from the library header, so the tests validate the implementation blind. Returns alpha in
// dB/m. Defaults are the library defaults: T = 20 C, RH = 50 %, p = 1 atm.
static inline double ref_iso9613_alpha(double f_Hz,
                                       double T_celsius = 20.0,
                                       double RH_percent = 50.0,
                                       double p_kPa = 101.325)
{
    const double T_ref = 293.15;    // Reference air temperature [K]
    const double T_triple = 273.16; // Triple-point isotherm [K]
    const double p_ref = 101.325;   // Reference static pressure [kPa]

    const double T = T_celsius + 273.15;
    const double p_ratio = p_kPa / p_ref;
    const double T_ratio = T / T_ref;

    // Molar concentration of water vapor, in percent
    const double C = 4.6151 - 6.8346 * std::pow(T_triple / T, 1.261);
    const double h = RH_percent * std::pow(10.0, C) / p_ratio;

    // Oxygen and nitrogen relaxation frequencies [Hz]
    const double f_rO = p_ratio * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
    const double f_rN = p_ratio / std::sqrt(T_ratio) *
                        (9.0 + 280.0 * h * std::exp(-4.170 * (std::pow(T_ratio, -1.0 / 3.0) - 1.0)));

    // Classical, rotational-O2 and rotational-N2 contributions
    const double f2 = f_Hz * f_Hz;
    const double term_cl = 1.84e-11 / p_ratio * std::sqrt(T_ratio);
    const double term_O = 0.01275 * std::exp(-2239.1 / T) / (f_rO + f2 / f_rO);
    const double term_N = 0.1068 * std::exp(-3352.0 / T) / (f_rN + f2 / f_rN);

    return 8.686 * f2 * (term_cl + std::pow(T_ratio, -2.5) * (term_O + term_N));
}

// Reference propagation gain from geometric spreading (and, in scalar mode, air absorption).
// EM:     32.45 + 20*log10(f_GHz) + 20*log10(d_m)  dB of loss
// Scalar: 20*log10(d_m) + alpha(f)*d_m             dB of loss
static inline double ref_spreading_gain(double f_Hz, double dist, bool scalar_mode)
{
    double gain_at_1m = scalar_mode ? 0.0 : (-32.45 - 20.0 * std::log10(f_Hz * 1.0e-9));
    double alpha = scalar_mode ? ref_iso9613_alpha(f_Hz) : 0.0;
    return std::pow(10.0, 0.1 * (gain_at_1m - 20.0 * std::log10(dist) - alpha * dist));
}

// Scalar mode takes an EM-equivalent frequency at the library API: c_light / f_eq is the
// acoustic wavelength. These two must match qd_atmospheric_absorption.hpp. Note the effective
// acoustic wave speed is NOT 343: the scale is formed with c = 3e8 while the phase uses the
// exact speed of light, so the two differ by 0.07 %.
static constexpr double ACOUSTIC_FREQ_SCALE = 3.0e8 / 343.0;                     // 874635.59
static constexpr double ACOUSTIC_WAVE_SPEED = 299792458.0 / ACOUSTIC_FREQ_SCALE; // 342.76276

// Reference propagation phase exp(-j*2*pi*d/lambda), unit magnitude. The wave speed selects
// EM (speed of light) vs. acoustic (speed of sound at 20 C, ISO 9613-1 reference conditions).
static inline std::complex<double> ref_prop_phase(double f_Hz, double dist, bool scalar_mode)
{
    double wave_speed = scalar_mode ? ACOUSTIC_WAVE_SPEED : 299792458.0;
    return std::polar(1.0, -2.0 * arma::datum::pi * f_Hz * dist / wave_speed);
}

// Build the standard 12-face cube plus a lossless, index-1 material map. Used by the
// apply_fspl / apply_phase tests, which mostly run on clear paths where the material is inert.
static inline void fspl_test_scene(arma::mat &cube, arma::uvec &mtl_ind,
                                   std::unordered_map<std::string, std::vector<double>> &mtl_map)
{
    cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});
    arma::mat mtl_prop = {{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);
}

// apply_fspl on an unobstructed path: the diffraction gain is exactly 1, so the returned gain
// must equal the bare free-space term. Also pins the default (apply_fspl = false) to 1, which
// is the regression guard for existing callers.
TEST_CASE("Calc Diffraction Gain - FSPL EM mode on clear path")
{
    arma::mat cube;
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    fspl_test_scene(cube, mtl_ind, mtl_map);

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{-5.0, 0.0, 0.5}}; // 5 m, entirely outside the cube
    const double dist = 5.0, freq = 10.0e9;

    arma::vec g_rel, g_abs;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, 0,
                                                &g_rel, nullptr, nullptr, 0, nullptr, 0, 0, false, 0.0, false, false);
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, 0,
                                                &g_abs, nullptr, nullptr, 0, nullptr, 0, 0, false, 0.0, true, false);

    CHECK(std::abs(g_rel(0) - 1.0) < 1e-13); // default is unchanged: relative to FSPL

    double expect = ref_spreading_gain(freq, dist, false);
    CHECK(std::abs(g_abs(0) / expect - 1.0) < 1e-12);

    // Sanity on the absolute level: 32.45 + 20*log10(10) + 20*log10(5) = 66.4288 dB
    CHECK(std::abs(-10.0 * std::log10(g_abs(0)) - 66.4288) < 0.001);
}

// The free-space term is a pure multiplicative scaling: it must not interact with the material
// losses already folded into the gain. Checked on an obstructed path and across lod levels,
// which also confirms the reference distance is the direct TX-RX line and not the arc length.
TEST_CASE("Calc Diffraction Gain - FSPL is separable from material loss")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});
    arma::mat mtl_prop = {{2.0, 0.0, 0.01, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{10.0, 0.0, 0.5}}; // 20 m, straight through both walls
    const double dist = 20.0, freq = 3.5e9;
    const double fspl = ref_spreading_gain(freq, dist, false);

    for (int lod : {0, 2, 4})
    {
        arma::vec g_rel, g_abs;
        quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, lod,
                                                    &g_rel, nullptr, nullptr, 0, nullptr, 0, 0, false, 0.0, false, false);
        quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, lod,
                                                    &g_abs, nullptr, nullptr, 0, nullptr, 0, 0, false, 0.0, true, false);

        CHECK(g_rel(0) > 0.0);
        CHECK(g_rel(0) < 1.0); // material loss is present
        CHECK(std::abs(g_abs(0) / (g_rel(0) * fspl) - 1.0) < 1e-12);
    }
}

// Inverse-square law: doubling the TX-RX distance costs exactly 6.0206 dB in EM mode, and the
// per-row distance is used for a multi-position call rather than a single shared value.
TEST_CASE("Calc Diffraction Gain - FSPL distance scaling and multi-position")
{
    arma::mat cube;
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    fspl_test_scene(cube, mtl_ind, mtl_map);

    const double freq = 2.4e9;

    // Three clear paths of 5 m, 10 m and 13 m (3-4-12 triple), all west of the cube
    arma::mat orig = {{-20.0, 0.0, 0.5}, {-20.0, 0.0, 0.5}, {-20.0, 0.0, 0.5}};
    arma::mat dest = {{-15.0, 0.0, 0.5}, {-10.0, 0.0, 0.5}, {-15.0, 12.0, 0.5}};
    arma::vec dist = {5.0, 10.0, 13.0};

    arma::vec gain;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, 0,
                                                &gain, nullptr, nullptr, 0, nullptr, 0, 0, false, 0.0, true, false);

    REQUIRE(gain.n_elem == 3);
    for (arma::uword i = 0; i < 3; ++i)
        CHECK(std::abs(gain(i) / ref_spreading_gain(freq, dist(i), false) - 1.0) < 1e-12);

    // 5 m -> 10 m is exactly a factor of 4 in power
    CHECK(std::abs(gain(0) / gain(1) - 4.0) < 1e-10);
    CHECK(std::abs(10.0 * std::log10(gain(0) / gain(1)) - 6.020599913279624) < 1e-10);
}

// Scalar (acoustic) mode uses 1/r spreading with no frequency-dependent reference, plus ISO
// 9613-1 air absorption. At 1 kHz absorption is negligible over 10 m and the result collapses
// to pure spreading; at 20 kHz it is a clearly measurable extra loss.
TEST_CASE("Calc Diffraction Gain - FSPL scalar mode with air absorption")
{
    arma::mat cube;
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    fspl_test_scene(cube, mtl_ind, mtl_map);

    arma::mat orig = {{-20.0, 0.0, 0.5}};
    arma::mat dest = {{-10.0, 0.0, 0.5}}; // 10 m clear path
    const double dist = 10.0;

    for (double freq : {1.0e3, 20.0e3})
    {
        arma::vec gain;
        quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map,
                                                    freq * ACOUSTIC_FREQ_SCALE, 0,
                                                    &gain, nullptr, nullptr, 0, nullptr, 0, 0, true, 0.0, true, false);
        CHECK(std::abs(gain(0) / ref_spreading_gain(freq, dist, true) - 1.0) < 1e-8);
    }

    // 1 kHz: absorption is only ~0.047 dB over 10 m, so spreading dominates
    arma::vec g_low;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map,
                                                1.0e3 * ACOUSTIC_FREQ_SCALE, 0,
                                                &g_low, nullptr, nullptr, 0, nullptr, 0, 0, true, 0.0, true, false);
    CHECK(ref_iso9613_alpha(1.0e3) * dist < 0.1);                   // absorption stays minor
    CHECK(std::abs(g_low(0) / (1.0 / (dist * dist)) - 1.0) < 0.02); // within 2 % of 1/d^2

    // 20 kHz: absorption adds real loss on top of spreading
    arma::vec g_high;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map,
                                                20.0e3 * ACOUSTIC_FREQ_SCALE, 0,
                                                &g_high, nullptr, nullptr, 0, nullptr, 0, 0, true, 0.0, true, false);
    CHECK(g_high(0) < g_low(0));
    // The ratio isolates the absorption difference; both endpoints carry their own alpha
    double extra_dB = -10.0 * std::log10(g_high(0) / g_low(0));
    double expect_dB = (ref_iso9613_alpha(20.0e3) - ref_iso9613_alpha(1.0e3)) * dist;
    CHECK(std::abs(extra_dB - expect_dB) < 1e-6);

    // Scalar and EM disagree at the same frequency: EM carries the 32.45 + 20*log10(f_GHz) term
    arma::vec g_em;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 20.0e3, 0,
                                                &g_em, nullptr, nullptr, 0, nullptr, 0, 0, false, 0.0, true, false);
    CHECK(g_em(0) > g_high(0)); // 20 kHz EM has a huge wavelength -> far less spreading loss
}

// apply_phase multiplies xprmat by exp(-j*2*pi*d/lambda). On a clear path the Jones matrix is
// the identity, so every diagonal entry must equal the bare phasor and xprmat must stay
// normalized (unit magnitude). Off-diagonal entries stay zero.
TEST_CASE("Calc Diffraction Gain - Propagation phase on clear path")
{
    arma::mat cube;
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    fspl_test_scene(cube, mtl_ind, mtl_map);

    arma::mat orig = {{-20.0, 0.0, 0.5}};
    arma::mat dest = {{-13.0, 0.0, 0.5}}; // 7 m clear path
    const double dist = 7.0, freq = 2.0e9;

    arma::mat X;
    arma::vec gain;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, 0,
                                                &gain, &X, nullptr, 0, nullptr, 0, 0, false, 0.0, false, true);

    REQUIRE(X.n_rows == 8);
    REQUIRE(X.n_cols == 1);

    std::complex<double> ph = ref_prop_phase(freq, dist, false);
    CHECK(std::abs(xpr_cx(X, 0, 0) - ph) < 1e-9); // VV
    CHECK(std::abs(xpr_cx(X, 6, 0) - ph) < 1e-9); // HH
    CHECK(std::abs(xpr_cx(X, 2, 0)) < 1e-12);     // HV
    CHECK(std::abs(xpr_cx(X, 4, 0)) < 1e-12);     // VH

    // Normalization is untouched: the phasor has unit magnitude
    CHECK(std::abs(std::abs(xpr_cx(X, 0, 0)) - 1.0) < 1e-12);

    // ... and the gain is unaffected by apply_phase
    CHECK(std::abs(gain(0) - 1.0) < 1e-13);

    // Scalar mode: single coefficient, wavelength from the speed of sound
    arma::mat Xs;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map,
                                                1.0e3 * ACOUSTIC_FREQ_SCALE, 0,
                                                nullptr, &Xs, nullptr, 0, nullptr, 0, 0, true, 0.0, false, true);
    REQUIRE(Xs.n_rows == 2);
    CHECK(std::abs(xpr_cx(Xs, 0, 0) - ref_prop_phase(1.0e3, dist, true)) < 1e-9);

    // The acoustic phasor must differ from the EM one at the same frequency and distance
    CHECK(std::abs(xpr_cx(Xs, 0, 0) - ref_prop_phase(1.0e3, dist, false)) > 1e-3);
}

// Two distances separated by exactly one wavelength must yield the same phasor. This pins the
// wave speed used to form lambda and exercises the internal modulo reduction of the distance.
TEST_CASE("Calc Diffraction Gain - Propagation phase wraps at one wavelength")
{
    arma::mat cube;
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    fspl_test_scene(cube, mtl_ind, mtl_map);

    const double freq = 1.0e9;
    const double lambda = 299792458.0 / freq; // ~0.29979 m

    arma::mat orig = {{-40.0, 0.0, 0.5}, {-40.0, 0.0, 0.5}};
    arma::mat dest = {{-30.0, 0.0, 0.5}, {-30.0 + lambda, 0.0, 0.5}};

    arma::mat X;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, 0,
                                                nullptr, &X, nullptr, 0, nullptr, 0, 0, false, 0.0, false, true);

    REQUIRE(X.n_cols == 2);
    CHECK(std::abs(xpr_cx(X, 0, 0) - xpr_cx(X, 0, 1)) < 1e-9);

    // Half a wavelength must instead invert the sign
    arma::mat dest_half = {{-30.0, 0.0, 0.5}, {-30.0 + 0.5 * lambda, 0.0, 0.5}};
    arma::mat Xh;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest_half, cube, mtl_ind, mtl_map, freq, 0,
                                                nullptr, &Xh, nullptr, 0, nullptr, 0, 0, false, 0.0, false, true);
    CHECK(std::abs(xpr_cx(Xh, 0, 0) + xpr_cx(Xh, 0, 1)) < 1e-9);

    // Same acoustic check, using the speed of sound to form lambda
    const double f_ac = 2.0e3, lambda_ac = ACOUSTIC_WAVE_SPEED / f_ac;
    arma::mat dest_ac = {{-30.0, 0.0, 0.5}, {-30.0 + lambda_ac, 0.0, 0.5}};
    arma::mat Xa;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest_ac, cube, mtl_ind, mtl_map,
                                                f_ac * ACOUSTIC_FREQ_SCALE, 0,
                                                nullptr, &Xa, nullptr, 0, nullptr, 0, 0, true, 0.0, false, true);
    CHECK(std::abs(xpr_cx(Xa, 0, 0) - xpr_cx(Xa, 0, 1)) < 1e-9);
}

// On an obstructed path the propagation phase must left-multiply the material Jones matrix
// entry by entry without altering its magnitude, i.e. it composes with (rather than replaces)
// the phase the material interactions already contributed.
TEST_CASE("Calc Diffraction Gain - Propagation phase composes with material phase")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});
    arma::mat mtl_prop = {{5.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    // Oblique crossing so that VV and HH differ and pick up material phase
    arma::mat orig = {{-10.0, -8.0, 0.5}};
    arma::mat dest = {{0.5, 0.5, 0.5}};
    const double dx = 10.5, dy = 8.5;
    const double dist = std::sqrt(dx * dx + dy * dy);
    const double freq = 6.0e9;

    arma::mat X0, X1;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, 0,
                                                nullptr, &X0, nullptr, 0, nullptr, 0, 0, false, 0.0, false, false);
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, 0,
                                                nullptr, &X1, nullptr, 0, nullptr, 0, 0, false, 0.0, false, true);

    std::complex<double> ph = ref_prop_phase(freq, dist, false);
    for (arma::uword base : {0u, 2u, 4u, 6u})
    {
        std::complex<double> a = xpr_cx(X0, base, 0);
        std::complex<double> b = xpr_cx(X1, base, 0);
        CHECK(std::abs(b - a * ph) < 1e-9);
        CHECK(std::abs(std::abs(b) - std::abs(a)) < 1e-12); // magnitudes preserved
    }

    // The material actually contributed a non-trivial Jones matrix here, so the composition
    // above is not a trivial identity-times-phasor check
    arma::vec ident(8, arma::fill::zeros);
    ident(0) = 1.0, ident(6) = 1.0; // ReVV and ReHH
    CHECK(arma::abs(X0.col(0) - ident).max() > 1e-6);
}

// The two flags are independent: apply_fspl touches only gain, apply_phase touches only
// xprmat, and enabling both reproduces each of them exactly.
TEST_CASE("Calc Diffraction Gain - apply_fspl and apply_phase are independent")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});
    arma::mat mtl_prop = {{2.5, 0.0, 0.02, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0}};
    mtl_prop = repmat(mtl_prop, 12, 1);
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    arma::mat orig = {{-10.0, 0.0, 0.5}};
    arma::mat dest = {{10.0, 0.0, 0.5}};
    const double freq = 28.0e9;

    arma::vec g00, g10, g01, g11;
    arma::mat X00, X10, X01, X11;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, 2,
                                                &g00, &X00, nullptr, 0, nullptr, 0, 0, false, 0.0, false, false);
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, 2,
                                                &g10, &X10, nullptr, 0, nullptr, 0, 0, false, 0.0, true, false);
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, 2,
                                                &g01, &X01, nullptr, 0, nullptr, 0, 0, false, 0.0, false, true);
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, freq, 2,
                                                &g11, &X11, nullptr, 0, nullptr, 0, 0, false, 0.0, true, true);

    // apply_fspl leaves xprmat alone, apply_phase leaves gain alone
    CHECK(arma::approx_equal(X10, X00, "absdiff", 1e-14));
    CHECK(arma::approx_equal(g01, g00, "absdiff", 1e-14));

    // Combined call reproduces each single-flag result
    CHECK(arma::approx_equal(g11, g10, "absdiff", 1e-14));
    CHECK(arma::approx_equal(X11, X01, "absdiff", 1e-14));

    // ... and both effects are actually present
    CHECK(g10(0) < 1e-3 * g00(0));                      // 28 GHz over 20 m is a big loss
    CHECK(arma::abs(X01 - X00).max() > 1e-3);           // phase was applied
    double p00 = 0.0, p01 = 0.0;
    for (arma::uword k = 0; k < X00.n_elem; ++k)
        p00 += X00(k) * X00(k), p01 += X01(k) * X01(k);
    CHECK(std::abs(p01 - p00) < 1e-12); // normalization held
}

// Degenerate geometry: co-located TX and RX make the free-space term and the phase undefined.
// The implementation must skip both rather than emit inf/NaN, and must not disturb other rows.
TEST_CASE("Calc Diffraction Gain - Co-located TX and RX is finite")
{
    arma::mat cube;
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    fspl_test_scene(cube, mtl_ind, mtl_map);

    arma::mat orig = {{-20.0, 0.0, 0.5}, {-20.0, 0.0, 0.5}};
    arma::mat dest = {{-20.0, 0.0, 0.5}, {-15.0, 0.0, 0.5}}; // row 0 degenerate, row 1 normal

    arma::vec gain;
    arma::mat X;
    quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 3.5e9, 0,
                                                &gain, &X, nullptr, 0, nullptr, 0, 0, false, 0.0, true, true);

    CHECK(std::isfinite(gain(0)));
    CHECK(X.col(0).is_finite());

    // The valid row is still correct
    CHECK(std::abs(gain(1) / ref_spreading_gain(3.5e9, 5.0, false) - 1.0) < 1e-12);
    CHECK(std::abs(xpr_cx(X, 0, 1) - ref_prop_phase(3.5e9, 5.0, false)) < 1e-9);
}

// Single-precision instantiation of the new flags. Float carries ~7 significant digits, so the
// phase is checked at a short distance and a low frequency to keep the argument small, and the
// tolerances are loosened accordingly.
TEST_CASE("Calc Diffraction Gain - Float instantiation with FSPL and phase")
{
    arma::fmat cube = quadriga_lib::cube<float>({}, {}, {0.0f, 0.0f, 0.001f});
    arma::uvec mtl_ind = arma::ones<arma::uvec>(12);

    std::unordered_map<std::string, std::vector<float>> mtl_map;
    mtl_map["a"] = {1.5f};
    mtl_map["b"] = {0.0f};
    mtl_map["c"] = {0.0f};
    mtl_map["d"] = {0.0f};
    mtl_map["att"] = {0.0f};
    mtl_map["attB"] = {0.0f};
    mtl_map["alpha"] = {0.0f};
    mtl_map["alphaB"] = {0.0f};
    mtl_map["fRef"] = {1.0f};

    arma::fmat orig = {{-20.0f, 0.0f, 0.5f}};
    arma::fmat dest = {{-16.0f, 0.0f, 0.5f}}; // 4 m clear path
    const double dist = 4.0, freq = 1.0e8;

    arma::fvec gain;
    arma::fmat X;
    quadriga_lib::calc_diffraction_gain<float>(orig, dest, cube, mtl_ind, mtl_map, (float)freq, 0,
                                               &gain, &X, nullptr, 0, nullptr, 0, 0, false, 0.0, true, true);

    CHECK(std::abs(gain(0) / (float)ref_spreading_gain(freq, dist, false) - 1.0f) < 1e-4f);

    std::complex<double> ph = ref_prop_phase(freq, dist, false);
    CHECK(std::abs((double)X(0, 0) - ph.real()) < 1e-5);
    CHECK(std::abs((double)X(1, 0) - ph.imag()) < 1e-5);
    CHECK(std::abs(std::hypot((double)X(0, 0), (double)X(1, 0)) - 1.0) < 1e-6);
}
// Level of detail 0-7 (lod 7: 1 path, 10 segments); coord holds the n_seg-1 waypoints of the generated paths;
// out-of-range lod and empty input give errors that name the input
TEST_CASE("Calc Diffraction Gain - lod range, coord shape and empty input")
{
    arma::mat cube = quadriga_lib::cube<double>({}, {}, {0.0, 0.0, 0.001});
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    arma::mat mtl_prop = repmat(arma::mat({{1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}), 12, 1);
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);

    arma::mat orig = {{-10.0, 0.0, 0.5}, {-10.0, 0.0, 0.2}};
    arma::mat dest = {{-5.0, 0.0, 0.5}, {-5.0, 0.0, 0.3}};
    arma::vec gain;
    arma::mat xprmat;
    arma::cube coord;

    const int lods[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const arma::uword n_wp[8] = {0, 2, 2, 3, 4, 1, 1, 9};
    for (int k = 0; k < 8; ++k)
    {
        INFO("lod " << lods[k]);
        quadriga_lib::calc_diffraction_gain<double>(orig, dest, cube, mtl_ind, mtl_map, 1.0e9, lods[k], &gain, &xprmat, &coord);
        CHECK(coord.n_rows == 3);
        CHECK(coord.n_cols == n_wp[k]);
        CHECK(coord.n_slices == 2);
        CHECK(arma::approx_equal(gain, arma::vec({1.0, 1.0}), "absdiff", 1e-13)); // path outside the cube

        if (lods[k] != 0) // coord matches the waypoints of generate_diffraction_paths
        {
            arma::cube rx, ry, rz, w;
            quadriga_lib::generate_diffraction_paths<double>(orig, dest, 1.0e9, lods[k], rx, ry, rz, w);
            CHECK(w.n_slices == n_wp[k] + 1);
            CHECK(rx.n_slices == n_wp[k]);
        }
    }

    auto message = [&](const arma::mat &o, const arma::mat &d, int lod) -> std::string
    {
        try
        {
            quadriga_lib::calc_diffraction_gain<double>(o, d, cube, mtl_ind, mtl_map, 1.0e9, lod, &gain, &xprmat, &coord);
        }
        catch (const std::invalid_argument &e)
        {
            return e.what();
        }
        return "no error";
    };
    CHECK(message(orig, dest, -1) == "Input 'lod' must be 0-7.");
    CHECK(message(orig, dest, 8) == "Input 'lod' must be 0-7.");
    CHECK(message(arma::mat(0, 3), arma::mat(0, 3), 2) == "Inputs 'orig' and 'dest' cannot be empty.");
    CHECK(message(arma::mat(0, 3), arma::mat(0, 3), 0) == "Inputs 'orig' and 'dest' cannot be empty.");
    CHECK_THROWS_AS(quadriga_lib::calc_diffraction_gain<double>(arma::mat(0, 3), arma::mat(0, 3), cube, mtl_ind, mtl_map, 1.0e9, 0, &gain),
                    std::invalid_argument);
}

TEST_CASE("Calc Diffraction Gain - verbose debug output with one position")
{
    // verbose = 2 with one position prints one line per ray and interaction. Two cubes beside the direct path: the
    // inner rays pass between them and leave the R-set first, so in the next iteration of a segment the R-set index
    // of a ray differs from its G-set index. The print used the G-set index for the R-set matrices (out of bounds).
    // Check: the print does not throw, and the destination printed for a ray is the same in every iteration of a
    // segment (it is the waypoint of that segment). The summary lines "iG <i> segP(...)" at the end are skipped.
    arma::uvec mtl_ind;
    std::unordered_map<std::string, std::vector<double>> mtl_map;
    arma::mat mtl_prop = repmat(arma::mat({{1.5, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}), 24, 1);
    mtl_matrix_to_map(mtl_prop, mtl_ind, mtl_map);
    arma::mat orig = {{-10.0, 0.0, 0.0}}, dest = {{10.0, 0.0, 0.0}};

    for (double yoff : {0.6, 1.0})
    {
        INFO("yoff " << yoff);
        arma::mat mesh = arma::join_cols(quadriga_lib::cube<double>({0.5}, {}, {0.0, yoff, 0.001}),
                                         quadriga_lib::cube<double>({0.5}, {}, {0.0, -yoff, 0.001}));
        arma::vec gain, gain_quiet;
        quadriga_lib::calc_diffraction_gain<double>(orig, dest, mesh, mtl_ind, mtl_map, 0.3e9, 4, &gain_quiet);

        std::ostringstream out;
        std::streambuf *old_buf = std::cout.rdbuf(out.rdbuf());
        bool threw = false;
        try
        {
            quadriga_lib::calc_diffraction_gain<double>(orig, dest, mesh, mtl_ind, mtl_map, 0.3e9, 4, &gain, nullptr, nullptr, 2);
        }
        catch (...)
        {
            threw = true;
        }
        std::cout.rdbuf(old_buf);
        REQUIRE(!threw);
        CHECK(arma::approx_equal(gain, gain_quiet, "absdiff", 0.0)); // the print does not change the result

        // Parse "Seg. <iS>.<iT>" and "iG <i>, ... D(<x>,<y>,<z>), ..." lines
        std::map<std::pair<int, int>, std::string> dest_first; // (iS, iG) -> D(...) of iteration 0
        std::istringstream lines(out.str());
        std::string line;
        int iS = -1, iT = -1, n_later = 0;
        while (std::getline(lines, line))
        {
            size_t p = line.find("Seg. ");
            if (p != std::string::npos)
            {
                REQUIRE(std::sscanf(line.c_str() + p, "Seg. %d.%d", &iS, &iT) == 2);
                continue;
            }
            if (line.rfind("iG ", 0) != 0 || line.find(", nH ") == std::string::npos) // per-interaction lines only
                continue;
            int iG = std::atoi(line.c_str() + 3);
            size_t d0 = line.find(", D("), d1 = line.find(')', d0);
            REQUIRE(d0 != std::string::npos);
            std::string d = line.substr(d0, d1 - d0);
            if (iT == 0)
                dest_first[{iS, iG}] = d;
            else
            {
                ++n_later;
                REQUIRE(dest_first.count({iS, iG}) == 1);
                CHECK(dest_first[{iS, iG}] == d);
            }
        }
        CHECK(n_later > 0); // the scene has a segment with a second iteration
    }
}
