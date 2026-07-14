// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_tools.hpp"

#include <iostream>
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

    // Material index exceeding 32767
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