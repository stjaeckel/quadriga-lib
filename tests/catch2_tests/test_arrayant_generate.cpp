// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Array antenna generators: grid sizes, field values at the main lobe and directivities (short dipole 1.76 dBi,
// half-wave dipole 2.15 dBi, stored values for the custom pattern); the half-wave dipole pattern as a closed form.
// TR 38.901 panel arrays (generate_arrayant_3GPP, _ula, _multibeam, arrayant_generate) against closed forms written
// independently of the library's loop: the Table 7.3-1 shape, Model-1 (eq. 7.3-3) and Model-2 (eq. 7.3-4/7.3-5), the
// element positions and the sub-array weights of eq. 7.3-1 for the calibration configurations of Tables 7.8-2, 7.8-2A,
// 7.8-6A and 7.8-6C; the kept elements against the combined ports; the beam directions of tilt and pan; the fill rules
// of the tuples (exact) and the input validation. Multibeam: the MRT gain of a single beam, the port and element order.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "quadriga_lib.hpp"
#include "qd_test_arrayant.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <string>

static const double qd_nan = std::numeric_limits<double>::quiet_NaN(), qd_inf = std::numeric_limits<double>::infinity();

// One beam (or another single column) as a matrix [n, 1]
static arma::mat column(const arma::vec &v)
{
    return arma::mat(v);
}

// TR 38.901 Table 7.3-1 power pattern in dB at azimuth az and elevation el in degree; c = [phi_3dB, theta_3dB, SLA_V, A_max]
static double tr_pattern_dB(double az, double el, const double *c)
{
    const double theta = 90.0 - el; // Zenith angle
    const double A_V = -std::min(12.0 * std::pow((theta - 90.0) / c[1], 2.0), c[2]);
    const double A_H = -std::min(12.0 * std::pow(az / c[0], 2.0), c[3]);
    return -std::min(-(A_V + A_H), c[3]);
}

// Largest difference of all pattern, position and coupling members (exact comparisons use 0)
template <typename dtype>
static double max_diff(const quadriga_lib::arrayant<dtype> &a, const quadriga_lib::arrayant<dtype> &b)
{
    if (arma::size(a.e_theta_re) != arma::size(b.e_theta_re) || arma::size(a.coupling_re) != arma::size(b.coupling_re) ||
        arma::size(a.element_pos) != arma::size(b.element_pos) || a.azimuth_grid.n_elem != b.azimuth_grid.n_elem)
        return qd_inf;
    return std::max({(double)arma::abs(a.e_theta_re - b.e_theta_re).max(), (double)arma::abs(a.e_theta_im - b.e_theta_im).max(),
                     (double)arma::abs(a.e_phi_re - b.e_phi_re).max(), (double)arma::abs(a.e_phi_im - b.e_phi_im).max(),
                     (double)arma::abs(a.element_pos - b.element_pos).max(), (double)arma::abs(a.coupling_re - b.coupling_re).max(),
                     (double)arma::abs(a.coupling_im - b.coupling_im).max(), (double)arma::abs(a.azimuth_grid - b.azimuth_grid).max(),
                     std::abs((double)a.center_freq - (double)b.center_freq)});
}

// Expected element positions [3, n_el] and coupling [n_el, n_ports] of a panel array from the element index
// e = p + P (m + M (n + N (mg + Mg ng))): TR 38.901 eq. 7.3-1 per sub-array of K = M / M_P vertical elements (1-based
// index k, theta_etilt = 90 deg + tilt), by analogy w = exp(-j 2 pi (l - (L+1)/2) d_H sin(pan)) over L = N / N_P
// horizontal elements; the common phase of a port is moved to the sub-array centre
struct panel_layout
{
    arma::uword M, N, Mg, Ng, MP, NP, P;
    double dV, dH, dgV, dgH;
    arma::mat steer; // [Mg Ng, 2] in degree
};

static void expected_layout(const panel_layout &c, double lambda, arma::mat &pos, arma::cx_mat &cpl)
{
    const arma::uword n_el = c.P * c.M * c.N * c.Mg * c.Ng, K = c.M / c.MP, L = c.N / c.NP;
    const double deg = arma::datum::pi / 180.0;
    pos.zeros(3, n_el), cpl.zeros(n_el, n_el / (K * L));
    for (arma::uword e = 0; e < n_el; ++e)
    {
        arma::uword r = e;
        const arma::uword p = r % c.P;
        r /= c.P;
        const arma::uword m = r % c.M;
        r /= c.M;
        const arma::uword n = r % c.N;
        r /= c.N;
        const arma::uword mg = r % c.Mg, ng = r / c.Mg, panel = mg + c.Mg * ng;
        pos(1, e) = (double(n) * c.dH - 0.5 * double(c.N - 1) * c.dH + double(ng) * c.dgH - 0.5 * double(c.Ng - 1) * c.dgH) * lambda;
        pos(2, e) = (double(m) * c.dV - 0.5 * double(c.M - 1) * c.dV + double(mg) * c.dgV - 0.5 * double(c.Mg - 1) * c.dgV) * lambda;
        const double k = double(m % K + 1), l = double(n % L + 1);
        const double theta_etilt = (90.0 + c.steer(panel, 0)) * deg, pan = c.steer(panel, 1) * deg;
        const double phase = -2.0 * arma::datum::pi * ((k - 0.5 * double(K + 1)) * c.dV * std::cos(theta_etilt) +
                                                       (l - 0.5 * double(L + 1)) * c.dH * std::sin(pan));
        const arma::uword port = p + c.P * (m / K + c.MP * (n / L + c.NP * panel));
        cpl(e, port) = std::polar(1.0 / std::sqrt(double(K * L)), phase);
    }
}

// Peak (elevation, azimuth) in degree of the power of element i
template <typename dtype>
static void peak_direction(const quadriga_lib::arrayant<dtype> &a, arma::uword i, double &el, double &az)
{
    const arma::Mat<dtype> pw = arma::square(a.e_theta_re.slice(i)) + arma::square(a.e_theta_im.slice(i)) +
                                arma::square(a.e_phi_re.slice(i)) + arma::square(a.e_phi_im.slice(i));
    const arma::uword k = pw.index_max();
    el = double(a.elevation_grid(k % a.n_elevation())) * 180.0 / arma::datum::pi;
    az = double(a.azimuth_grid(k / a.n_elevation())) * 180.0 / arma::datum::pi;
}

TEST_CASE("Generate Arrayant - Minimal test omni")
{
    auto ant = quadriga_lib::generate_arrayant_omni<float>();
    CHECK(ant.name == "omni");
    CHECK(ant.n_elevation() == 181);
    CHECK(ant.n_azimuth() == 361);

    auto Z = arma::fmat(181, 361, arma::fill::zeros);
    auto O = arma::fmat(181, 361, arma::fill::ones);
    CHECK(arma::approx_equal(ant.e_theta_re.slice(0), O, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_phi_re.slice(0), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_theta_im.slice(0), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_phi_im.slice(0), Z, "absdiff", 1e-7));

    auto ant2 = quadriga_lib::generate_arrayant_omni<double>(10.0);
    CHECK(ant2.name == "omni");
    CHECK(ant2.n_elevation() == 19);
    CHECK(ant2.n_azimuth() == 37);
}

TEST_CASE("Generate Arrayant - Minimal test xpol")
{
    auto ant = quadriga_lib::generate_arrayant_xpol<float>();
    CHECK(ant.name == "xpol");
    CHECK(ant.n_elevation() == 181);
    CHECK(ant.n_azimuth() == 361);
    CHECK(ant.n_elements() == 2);

    auto Z = arma::fmat(181, 361, arma::fill::zeros);
    auto O = arma::fmat(181, 361, arma::fill::ones);
    CHECK(arma::approx_equal(ant.e_theta_re.slice(0), O, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_theta_re.slice(1), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_phi_re.slice(0), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_phi_re.slice(1), O, "absdiff", 1e-7));

    CHECK(arma::approx_equal(ant.e_theta_im.slice(0), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_theta_im.slice(1), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_phi_im.slice(0), Z, "absdiff", 1e-7));
    CHECK(arma::approx_equal(ant.e_phi_im.slice(1), Z, "absdiff", 1e-7));

    auto ant2 = quadriga_lib::generate_arrayant_xpol<double>(10.0);
    CHECK(ant2.n_elevation() == 19);
    CHECK(ant2.n_azimuth() == 37);
}

TEST_CASE("Generate Arrayant - Minimal test dipole")
{
    auto ant = quadriga_lib::generate_arrayant_dipole<float>();
    float directivity = qd_directivity(ant, 0);
    CHECK(std::abs(directivity - 1.760964f) < 0.0001f);
    REQUIRE_THROWS_AS(qd_directivity(ant, 1), std::invalid_argument);

    auto ant2 = quadriga_lib::generate_arrayant_dipole<double>(5.0);
    CHECK(ant2.n_elevation() == 37);
    CHECK(ant2.n_azimuth() == 73);
    double directivity2 = qd_directivity(ant2, 0);
    CHECK(std::abs(directivity2 - 1.760964) < 0.01f);
}

TEST_CASE("Generate Arrayant - Minimal test Half-wave dipole")
{
    auto ant = quadriga_lib::generate_arrayant_half_wave_dipole<float>();
    float directivity = qd_directivity(ant, 0);
    CHECK(std::abs(directivity - 2.15f) < 0.001f);

    auto ant2 = quadriga_lib::generate_arrayant_half_wave_dipole<double>(5.0);
    CHECK(ant2.n_elevation() == 37);
    CHECK(ant2.n_azimuth() == 73);
    double directivity2 = qd_directivity(ant2, 0);
    CHECK(std::abs(directivity2 - 2.15) < 0.01f);
}

TEST_CASE("Generate Arrayant - Custom")
{
    auto ant = quadriga_lib::generate_arrayant_custom<float>(10.0, 10.0);
    float directivity = qd_directivity(ant, 0);
    CHECK(std::abs(directivity - 25.627f) < 0.001f);

    auto ant2 = quadriga_lib::generate_arrayant_custom<double>(10.0, 10.0, 0.0, 5.0);
    CHECK(ant2.n_elevation() == 37);
    CHECK(ant2.n_azimuth() == 73);
    double directivity2 = qd_directivity(ant2, 0);
    CHECK(std::abs(directivity2 - 25.62) < 0.01f);
}

TEST_CASE("generate_arrayant_3GPP - the element follows the Table 7.3-1 shape, also with other constants")
{
    // Field in dB = G_max + A_dB of Table 7.3-1 at every grid point: the default constants, those of Table 7.3-2
    // (handheld UT) and others with theta_3dB != phi_3dB and A_max != SLA_V; float equals double
    const double c_def[5] = {65.0, 65.0, 30.0, 30.0, 8.0}, c_ut[5] = {125.0, 125.0, 22.5, 22.5, 5.3}, c_x[5] = {50.0, 80.0, 25.0, 35.0, 6.0};
    const double rad2deg = 180.0 / arma::datum::pi;
    for (const double *c : {c_def, c_ut, c_x})
    {
        const arma::vec element = (c == c_def) ? arma::vec() : arma::vec(c, 5);
        const auto d = quadriga_lib::generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {}, {}, element, {}, 5.0);
        const auto f = quadriga_lib::generate_arrayant_3GPP<float>({}, 3.5e9, {}, {}, {}, {}, element, {}, 5.0);
        REQUIRE(d.n_elements() == 1);
        REQUIRE(d.n_azimuth() == 73);
        REQUIRE(d.n_elevation() == 37);
        double d_dB = 0.0;
        for (arma::uword j = 0; j < d.n_azimuth(); ++j)
            for (arma::uword i = 0; i < d.n_elevation(); ++i)
            {
                const double expected = c[4] + tr_pattern_dB(d.azimuth_grid(j) * rad2deg, d.elevation_grid(i) * rad2deg, c);
                d_dB = std::max(d_dB, std::abs(20.0 * std::log10(d.e_theta_re(i, j, 0)) - expected));
            }
        INFO("phi_3dB = " << c[0]);
        CHECK(d_dB < 1e-10);
        CHECK(arma::abs(d.e_theta_im).max() == 0.0);
        CHECK(arma::abs(d.e_phi_re).max() == 0.0);
        CHECK(arma::abs(d.e_phi_im).max() == 0.0);
        CHECK(arma::abs(arma::conv_to<arma::cube>::from(f.e_theta_re) - d.e_theta_re).max() < 1e-5);
    }
}

TEST_CASE("generate_arrayant_3GPP - G_max NaN sets the peak gain to the directivity")
{
    // The default element has its peak at G_max = 8 dBi, below its directivity of 9.83 dBi; NaN normalizes the peak to
    // the directivity (the directivity on the 1 degree grid, so 0.01 dB)
    const auto a = quadriga_lib::generate_arrayant_3GPP<double>();
    CHECK(std::abs(20.0 * std::log10(a.e_theta_re.max()) - 8.0) < 1e-12);
    CHECK(std::abs(qd_directivity(a, 0) - 9.83) < 0.01);
    for (const arma::vec &element : {arma::vec{65.0, 65.0, 30.0, 30.0, qd_nan}, arma::vec{125.0, qd_nan, 22.5, qd_nan, qd_nan}})
    {
        const auto b = quadriga_lib::generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {}, {}, element);
        CHECK(std::abs(20.0 * std::log10(b.e_theta_re.max()) - qd_directivity(b, 0)) < 0.01);
    }
}

TEST_CASE("generate_arrayant_3GPP - turned polarizations follow TR 38.901 Model-1, also on the rotation axis")
{
    // pol = 2 and 3 turn the vertical TR element by the slant zeta (0 and 90 degree; +45 and -45 degree) about its
    // boresight. TR 38.901 eq. 7.3-3 with the angles of eq. 7.1-7 (alpha = beta = 0, gamma = zeta), theta the zenith angle:
    // cos(theta'') = cos(zeta) cos(theta) - sin(zeta) sin(phi) sin(theta),
    // phi'' = arg(sin(theta) cos(phi) + j (sin(zeta) cos(theta) + cos(zeta) sin(phi) sin(theta))),
    // F_theta = sqrt(A''(theta'', phi'')) cos(psi), F_phi = sqrt(A''(theta'', phi'')) sin(psi), with cos(psi) and sin(psi)
    // of eq. 7.3-3a/b (their denominator is sin(theta'')) and A'' the Table 7.3-1 element with G_max = 8 dBi.
    // Off the axis the turned element is resampled from the 1 degree grid. Within 20 dB of the peak no min() of
    // Table 7.3-1 is active and sqrt(A) = exp(-a x^2) in each element angle x in degree (|x| <= 84), a = 0.6 ln(10) / 65^2;
    // linear interpolation over h = 1 degree errs by at most h^2 / 8 |4 a^2 x^2 - 2 a| <= 3e-4 of |F| per direction, so
    // the field error stays below 1e-3 of |F| (0.009 dB). On the axis (theta'' = 0 or 180 degree, on the grid) the
    // polarization is not rotated: phi'' makes the local theta unit vector equal to the global one, F_phi = 0 (rounding
    // only). zeta = 0 is the vertical element without resampling; float equals double everywhere.
    const double deg = arma::datum::pi / 180.0, c_tr[4] = {65.0, 65.0, 30.0, 30.0};
    auto F = [&](double th2, double ph2) // sqrt(A'') at the turned angles in rad
    { return std::pow(10.0, 0.05 * (8.0 + tr_pattern_dB(ph2 / deg, 90.0 - th2 / deg, c_tr))); };
    const auto v = quadriga_lib::generate_arrayant_3GPP<double>({}, 3.5e9);
    const double peak = v.e_theta_re.max();
    for (arma::uword pol : {2ULL, 3ULL})
    {
        const auto d = quadriga_lib::generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {pol});
        const auto f = quadriga_lib::generate_arrayant_3GPP<float>({}, 3.5e9, {}, {}, {pol});
        REQUIRE(d.n_elements() == 2);
        const double zeta[2] = {pol == 2 ? 0.0 : 45.0 * deg, pol == 2 ? 90.0 * deg : -45.0 * deg};
        double d_off = 0.0, d_axis = 0.0, d_float = 0.0;
        arma::uword n_off = 0, n_axis = 0;
        for (arma::uword s = 0; s < 2; ++s)
            for (arma::uword j = 0; j < d.n_azimuth(); ++j)
                for (arma::uword i = 0; i < d.n_elevation(); ++i)
                {
                    const double th = 0.5 * arma::datum::pi - d.elevation_grid(i), ph = d.azimuth_grid(j);
                    const double cz = std::cos(zeta[s]), sz = std::sin(zeta[s]);
                    const double x2 = std::sin(th) * std::cos(ph), y2 = sz * std::cos(th) + cz * std::sin(ph) * std::sin(th);
                    const double z2 = cz * std::cos(th) - sz * std::sin(ph) * std::sin(th), w = std::sqrt(x2 * x2 + y2 * y2);
                    if (w < 1e-9) // Axis: the global theta unit vector in the turned frame, (ex, ey, 0), is the local one
                    {
                        const double ex = std::cos(th) * std::cos(ph), ey = cz * std::cos(th) * std::sin(ph) - sz * std::sin(th);
                        const double A = F(z2 > 0.0 ? 0.0 : arma::datum::pi, std::atan2(z2 * ey, z2 * ex));
                        d_axis = std::max({d_axis, std::abs(d.e_theta_re(i, j, s) - A), std::abs(d.e_phi_re(i, j, s))});
                        ++n_axis;
                    }
                    else if (const double A = F(std::atan2(w, z2), std::atan2(y2, x2)); A >= 0.1 * peak) // Within 20 dB
                    {
                        const double Ft = A * (cz * std::sin(th) + sz * std::sin(ph) * std::cos(th)) / w, Fp = A * sz * std::cos(ph) / w;
                        d_off = std::max(d_off, std::hypot(d.e_theta_re(i, j, s) - Ft, d.e_phi_re(i, j, s) - Fp) / A);
                        ++n_off;
                    }
                    d_float = std::max({d_float, std::abs((double)f.e_theta_re(i, j, s) - d.e_theta_re(i, j, s)),
                                        std::abs((double)f.e_phi_re(i, j, s) - d.e_phi_re(i, j, s))});
                }
        INFO("pol = " << pol << ", " << n_off << " points off the axis, " << n_axis << " on it");
        CHECK(n_axis == (pol == 2 ? 2 * d.n_azimuth() + 2 : 4)); // zeta = 0: the grid poles
        CHECK(d_off < 1e-3);
        CHECK(d_axis < 1e-12 * peak);
        CHECK(d_float < 1e-5); // Single-precision field tolerance
        CHECK(arma::abs(d.e_theta_im).max() + arma::abs(d.e_phi_im).max() == 0.0);
    }
    const auto d = quadriga_lib::generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {2});
    CHECK(arma::approx_equal(d.e_theta_re.slice(0), v.e_theta_re.slice(0), "absdiff", 0.0));
    CHECK(arma::abs(d.e_phi_re.slice(0)).max() == 0.0);
}

TEST_CASE("generate_arrayant_3GPP - Model-2 follows eq. 7.3-4 and 7.3-5")
{
    // F_theta = sqrt(A') cos(zeta), F_phi = sqrt(A') sin(zeta) with A' = |F_theta|^2 + |F_phi|^2 of the element: on the
    // TR element sqrt(A') is the vertical element (pol 1 with zeta = 0, pol 2 with 0 and 90, pol 3 with +45 and -45 degree)
    const auto v = quadriga_lib::generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {}, {}, {}, {}, 5.0);
    const double s = std::sqrt(0.5), peak = v.e_theta_re.max();
    const double cz[3][2] = {{1.0, 0.0}, {1.0, 0.0}, {s, s}}, sz[3][2] = {{0.0, 0.0}, {0.0, 1.0}, {s, -s}};
    for (arma::uword pol = 1; pol <= 3; ++pol)
    {
        const auto a = quadriga_lib::generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {pol, 2}, {}, {}, {}, 5.0);
        REQUIRE(a.n_elements() == (pol == 1 ? 1 : 2));
        for (arma::uword i = 0; i < a.n_elements(); ++i)
        {
            INFO("pol = " << pol << ", element " << i);
            CHECK(arma::approx_equal(a.e_theta_re.slice(i), v.e_theta_re.slice(0) * cz[pol - 1][i], "absdiff", 1e-12 * peak));
            CHECK(arma::approx_equal(a.e_phi_re.slice(i), v.e_theta_re.slice(0) * sz[pol - 1][i], "absdiff", 1e-12 * peak));
        }
        CHECK(arma::abs(a.e_theta_im).max() == 0.0);
        CHECK(arma::abs(a.e_phi_im).max() == 0.0);
    }

    // A one-element pattern with a phi component and a phase (|F|^2 = 1): Model-2 replaces its polarization, Model-1
    // with pol = 1 uses it as given
    auto p = quadriga_lib::generate_arrayant_omni<double>(10.0);
    p.e_theta_re.fill(0.36), p.e_theta_im.fill(0.48), p.e_phi_im.fill(0.8);
    const auto m2 = quadriga_lib::generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {3, 2}, {}, {}, p);
    REQUIRE(m2.n_elements() == 2);
    CHECK(arma::abs(m2.e_theta_re - s).max() < 1e-12);
    CHECK(arma::abs(m2.e_phi_re.slice(0) - s).max() < 1e-12);
    CHECK(arma::abs(m2.e_phi_re.slice(1) + s).max() < 1e-12);
    CHECK(arma::abs(m2.e_theta_im).max() + arma::abs(m2.e_phi_im).max() == 0.0);
    const auto m1 = quadriga_lib::generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {1, 1}, {}, {}, p);
    p.center_freq = 3.5e9;
    CHECK(max_diff(m1, p) == 0.0);
}

TEST_CASE("generate_arrayant_3GPP - TR 38.901 calibration configurations: counts, positions and sub-array weights")
{
    // (M, N, P, Mg, Ng; M_P, N_P) of Tables 7.8-2 (config 1: one port per polarization and panel; config 2: one port per
    // element), 7.8-2A (configs 3, 4), 7.8-6A (UMi configs 1, 2 and the indoor config with 3 columns per port) and 7.8-6C;
    // Model-2, +/-45 degree slants for P = 2, the downtilt 12 degree (theta_etilt = 102 degree) of Table 7.8-6A. The
    // element and port counts are the TR products P M N Mg Ng and P M_P N_P Mg Ng; res 30 degree, as only the structure is
    // checked. Positions and weights against expected_layout (eq. 7.3-1 per sub-array).
    struct cfg
    {
        const char *name;
        arma::uvec panel, ports;
        arma::vec spacing;
        arma::mat steer;
        double fc;
        panel_layout lay;
        arma::uword n_el, n_port;
    };
    const arma::mat t12 = {{12.0, 0.0}};
    const std::vector<cfg> cfgs = {
        {"7.8-2 config 1", {4, 4, 1, 2}, {1, 1}, {0.5, 0.5, 2.5, 2.5}, {12.0}, 30e9, {4, 4, 1, 2, 1, 1, 2, 0.5, 0.5, 2.5, 2.5, {{12.0, 0.0}, {12.0, 0.0}}}, 64, 4},
        {"7.8-2 config 2", {2, 2}, {}, {0.5}, {}, 30e9, {2, 2, 1, 1, 2, 2, 1, 0.5, 0.5, 0.0, 0.0, {{0.0, 0.0}}}, 4, 4},
        {"7.8-2A config 3", {8, 16}, {8, 16}, {0.5}, {}, 7e9, {8, 16, 1, 1, 8, 16, 2, 0.5, 0.5, 0.0, 0.0, {{0.0, 0.0}}}, 256, 256},
        {"7.8-2A config 4", {64, 16}, {16, 16}, {0.5}, {12.0}, 7e9, {64, 16, 1, 1, 16, 16, 2, 0.5, 0.5, 0.0, 0.0, t12}, 2048, 512},
        {"7.8-6A config 1", {24, 32}, {8, 32}, {0.7, 0.5}, {12.0}, 7e9, {24, 32, 1, 1, 8, 32, 2, 0.7, 0.5, 0.0, 0.0, t12}, 1536, 512},
        {"7.8-6A config 2", {64, 16}, {8, 16}, {0.5}, {12.0}, 15e9, {64, 16, 1, 1, 8, 16, 2, 0.5, 0.5, 0.0, 0.0, t12}, 2048, 256},
        {"7.8-6A indoor", {8, 24}, {8, 8}, {0.5}, {{0.0, 20.0}}, 7e9, {8, 24, 1, 1, 8, 8, 2, 0.5, 0.5, 0.0, 0.0, {{0.0, 20.0}}}, 384, 128},
        {"7.8-6C", {8, 16}, {8, 16}, {0.5}, {}, 7e9, {8, 16, 1, 1, 8, 16, 2, 0.5, 0.5, 0.0, 0.0, {{0.0, 0.0}}}, 256, 256}};

    const double rad2deg = 180.0 / arma::datum::pi, c_tr[4] = {65.0, 65.0, 30.0, 30.0}, s = std::sqrt(0.5);
    for (const auto &c : cfgs)
    {
        INFO(c.name);
        const arma::uword P = c.lay.P;
        const auto a = quadriga_lib::generate_arrayant_3GPP<double>(c.panel, c.fc, c.spacing, c.ports, {P == 2 ? 3ULL : 1ULL, 2ULL}, c.steer, {}, {}, 30.0);
        REQUIRE(a.n_elements() == c.n_el);
        REQUIRE(a.n_ports() == c.n_port);

        const double lambda = 299792458.0 / c.fc;
        arma::mat pos;
        arma::cx_mat cpl;
        expected_layout(c.lay, lambda, pos, cpl);
        const double aperture = arma::abs(pos).max();
        CHECK(arma::abs(a.element_pos - pos).max() < 1e-12 * aperture);
        CHECK(arma::abs(arma::cx_mat(a.coupling_re, a.coupling_im) - cpl).max() < 1e-12);

        // Element e has the pattern of polarization e % P: Model-2 of the TR element
        double d_pat = 0.0;
        for (arma::uword e = 0; e < a.n_elements(); ++e)
            for (arma::uword j = 0; j < a.n_azimuth(); ++j)
                for (arma::uword i = 0; i < a.n_elevation(); ++i)
                {
                    const double F = std::pow(10.0, 0.05 * (8.0 + tr_pattern_dB(a.azimuth_grid(j) * rad2deg, a.elevation_grid(i) * rad2deg, c_tr)));
                    const double cz = P == 1 ? 1.0 : s, sz = P == 1 ? 0.0 : (e % 2 == 0 ? s : -s);
                    d_pat = std::max({d_pat, std::abs(a.e_theta_re(i, j, e) - F * cz), std::abs(a.e_phi_re(i, j, e) - F * sz)});
                }
        CHECK(d_pat < 1e-12);
    }
}

TEST_CASE("generate_arrayant_3GPP - kept elements with coupling equal the combined ports at the sub-array centres")
{
    // Two panels with their own steering rows and sub-arrays of 2 x 2 elements; apply_weights combines each port at its
    // sub-array centre (the mean position of its elements). The port responses, with the phase of the element or port
    // positions, agree at all grid directions within the rounding of k * aperture.
    const arma::uvec panel = {4, 4, 1, 2}, ports = {2, 2}, pol = {3, 2};
    const arma::vec spacing = {0.5, 0.6, 2.5, 3.0};
    const arma::mat steer = {{10.0, 20.0}, {-5.0, -30.0}};
    const double fc = 3.5e9, k = 2.0 * arma::datum::pi * fc / 299792458.0;
    const auto kept = quadriga_lib::generate_arrayant_3GPP<double>(panel, fc, spacing, ports, pol, steer, {}, {}, 10.0);
    const auto comb = quadriga_lib::generate_arrayant_3GPP<double>(panel, fc, spacing, ports, pol, steer, {}, {}, 10.0, true);
    REQUIRE(kept.n_elements() == 64);
    REQUIRE(kept.n_ports() == 16);
    REQUIRE(comb.n_elements() == 16);
    REQUIRE(comb.n_ports() == 16);

    const arma::cx_mat cpl(kept.coupling_re, kept.coupling_im);
    for (arma::uword j = 0; j < 16; ++j)
    {
        const arma::uvec members = arma::find(arma::abs(cpl.col(j)) > 0.0);
        REQUIRE(members.n_elem == 4);
        CHECK(arma::abs(arma::mean(kept.element_pos.cols(members), 1) - comb.element_pos.col(j)).max() < 1e-12 * arma::abs(kept.element_pos).max());
    }

    arma::mat az(1, kept.n_azimuth() * kept.n_elevation()), el(1, az.n_elem);
    for (arma::uword j = 0, i_ang = 0; j < kept.n_azimuth(); ++j)
        for (arma::uword i = 0; i < kept.n_elevation(); ++i, ++i_ang)
            az(i_ang) = kept.azimuth_grid(j), el(i_ang) = kept.elevation_grid(i);
    arma::mat V_re, V_im, H_re, H_im, dist, Vc_re, Vc_im, Hc_re, Hc_im, dist_c;
    quadriga_lib::arrayant_interpolate(kept, az, el, V_re, V_im, H_re, H_im, {}, {}, {}, &dist);
    quadriga_lib::arrayant_interpolate(comb, az, el, Vc_re, Vc_im, Hc_re, Hc_im, {}, {}, {}, &dist_c);
    const arma::cx_mat ph(arma::cos(k * dist), -arma::sin(k * dist)), ph_c(arma::cos(k * dist_c), -arma::sin(k * dist_c));
    const arma::cx_mat Vp = arma::cx_mat(V_re, V_im) % ph, Hp = arma::cx_mat(H_re, H_im) % ph;
    arma::cx_mat V_kept(16, az.n_elem, arma::fill::zeros), H_kept(16, az.n_elem, arma::fill::zeros); // No BLAS: sums by hand
    for (arma::uword e = 0; e < 64; ++e)
        for (arma::uword j = 0; j < 16; ++j)
            V_kept.row(j) += cpl(e, j) * Vp.row(e), H_kept.row(j) += cpl(e, j) * Hp.row(e);
    const arma::cx_mat V_comb = arma::cx_mat(Vc_re, Vc_im) % ph_c, H_comb = arma::cx_mat(Hc_re, Hc_im) % ph_c;
    const double peak = std::max(arma::abs(V_kept).max(), arma::abs(H_kept).max());
    CHECK(arma::abs(V_kept - V_comb).max() < 1e-12 * peak);
    CHECK(arma::abs(H_kept - H_comb).max() < 1e-12 * peak);
}

TEST_CASE("generate_arrayant_3GPP - a positive electrical tilt points the beam down")
{
    // A positive electrical tilt is a downtilt (TR 38.901): 8 vertical elements combined into one port with tilt 10
    // degree, the main beam at elevation -10 degree
    const auto ant = quadriga_lib::generate_arrayant_3GPP<double>({8}, 3.5e9, {}, {1}, {}, {10.0}, {}, {}, 1.0, true);
    REQUIRE(ant.n_elements() == 1);
    double el, az;
    peak_direction(ant, 0, el, az);
    CHECK(std::abs(el + 10.0) < 1.01);
}

TEST_CASE("generate_arrayant_3GPP - a positive pan turns the beam towards +y")
{
    // 8 columns of omni elements combined into one port (N_P = 1) with pan 20 degree: the main beam at azimuth +20 degree
    const auto omni = quadriga_lib::generate_arrayant_omni<double>(1.0);
    const auto ant = quadriga_lib::generate_arrayant_3GPP<double>({1, 8}, 3.5e9, {}, {0, 1}, {}, {0.0, 20.0}, {}, omni, 1.0, true);
    REQUIRE(ant.n_elements() == 1);
    double el, az;
    peak_direction(ant, 0, el, az);
    CHECK(std::abs(az - 20.0) < 1.01);
    CHECK(std::abs(el) < 1.01);
}

TEST_CASE("generate_arrayant_3GPP - each panel has its own steering row")
{
    // Two co-located panels (panel spacing 0) of 8 x 1 omni elements, one port per panel, tilt 10 and -20 degree: the
    // beams point to elevation -10 and +20 degree from the same position
    const auto omni = quadriga_lib::generate_arrayant_omni<double>(1.0), omni_10 = quadriga_lib::generate_arrayant_omni<double>(10.0);
    const arma::mat steer = {{10.0, 0.0}, {-20.0, 0.0}};
    const auto kept = quadriga_lib::generate_arrayant_3GPP<double>({8, 1, 1, 2}, 3.5e9, {0.5, 0.5, 0.0, 0.0}, {1}, {}, steer, {}, omni_10);
    CHECK(arma::approx_equal(kept.element_pos.cols(0, 7), kept.element_pos.cols(8, 15), "absdiff", 0.0));
    const auto ant = quadriga_lib::generate_arrayant_3GPP<double>({8, 1, 1, 2}, 3.5e9, {0.5, 0.5, 0.0, 0.0}, {1}, {}, steer, {}, omni, 1.0, true);
    REQUIRE(ant.n_elements() == 2);
    CHECK(arma::abs(ant.element_pos).max() < 1e-12 * 0.3);
    double el, az;
    peak_direction(ant, 0, el, az);
    CHECK(std::abs(el + 10.0) < 1.01);
    peak_direction(ant, 1, el, az);
    CHECK(std::abs(el - 20.0) < 1.01);
}

TEST_CASE("generate_arrayant_3GPP - omitted entries, 0 and NaN select the defaults")
{
    // Fill rules of the tuples, once: omitted entries, 0 (integer tuples) and NaN (real tuples) give the explicit defaults
    // with the same arithmetic (exact); dependent defaults d_H = d_V, d_gV = M d_V, d_gH = N d_H, ports [M, N],
    // theta_3dB = phi_3dB, A_max = SLA_V; one steering row for all panels, one column = pan 0
    const double n = qd_nan;
    auto g = [](const arma::uvec &panel, const arma::vec &spacing, const arma::uvec &ports, const arma::uvec &pol,
                const arma::mat &steer, const arma::vec &element)
    { return quadriga_lib::generate_arrayant_3GPP<double>(panel, 3.5e9, spacing, ports, pol, steer, element, {}, 30.0); };

    const auto ref1 = g({1, 1, 1, 1}, {0.5, 0.5, 0.5, 0.5}, {1, 1}, {1, 1}, {0.0, 0.0}, {65.0, 65.0, 30.0, 30.0, 8.0});
    CHECK(max_diff(g({}, {}, {}, {}, {}, {}), ref1) == 0.0);
    CHECK(max_diff(g({0, 0, 0, 0}, {n, n, n, n}, {0, 0}, {0, 0}, {n, n}, {n, n, n, n, 8.0}), ref1) == 0.0);

    const arma::uvec panel = {3, 2, 2, 2};
    const auto ref2 = g(panel, {0.75, 0.75, 2.25, 1.5}, {3, 2}, {3, 2}, {{0.0, 0.0}}, {50.0, 50.0, 20.0, 20.0, 8.0});
    CHECK(max_diff(g(panel, {0.75}, {}, {3, 2}, {}, {50.0, n, 20.0}), ref2) == 0.0);
    CHECK(max_diff(g(panel, {0.75, n, n, n}, {0, 0}, {3, 2}, {n, n}, {50.0, n, 20.0, n}), ref2) == 0.0);
    CHECK(max_diff(g(panel, {}, {0, 1}, {}, {}, {}), g(panel, {}, {3, 1}, {}, {}, {})) == 0.0);
    CHECK(max_diff(g(panel, {}, {}, {0, 2}, {}, {}), g(panel, {}, {}, {1, 2}, {}, {})) == 0.0);

    const arma::mat rows = {{5.0, -10.0}, {5.0, -10.0}, {5.0, -10.0}, {5.0, -10.0}};
    const auto ref3 = g(panel, {}, {1, 1}, {}, rows, {});
    CHECK(max_diff(g(panel, {}, {1, 1}, {}, {5.0, -10.0}, {}), ref3) == 0.0);
    CHECK(max_diff(g(panel, {}, {1, 1}, {}, arma::mat{5.0}, {}), g(panel, {}, {1, 1}, {}, {5.0, 0.0}, {})) == 0.0);
    CHECK(max_diff(g(panel, {}, {1, 1}, {}, arma::mat{{n, n}}, {}), g(panel, {}, {1, 1}, {}, {}, {})) == 0.0);
}

TEST_CASE("generate_arrayant_ula - the 3GPP panel path with the omni element")
{
    // N omni elements along y at (n - (N-1)/2) spacing lambda, one port each; the same array as arrayant_generate with
    // type 'ula' and panel [1, N]; N = 0 is 1, NaN spacing is 0.5; panel = 8 of type 'ula' is a vertical column
    const double fc = 2.4e9, lambda = 299792458.0 / fc;
    const auto u = quadriga_lib::generate_arrayant_ula<double>(5, fc, 0.6, {}, 10.0);
    const auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);
    REQUIRE(u.n_elements() == 5);
    CHECK(u.name == "ula");
    CHECK(u.center_freq == fc);
    for (arma::uword i = 0; i < 5; ++i)
        CHECK(arma::approx_equal(u.e_theta_re.slice(i), omni.e_theta_re.slice(0), "absdiff", 0.0));
    CHECK(arma::approx_equal(u.coupling_re, arma::mat(5, 5, arma::fill::eye), "absdiff", 0.0));
    CHECK(arma::abs(u.coupling_im).max() == 0.0);
    const arma::rowvec y = arma::regspace<arma::rowvec>(-2.0, 2.0) * 0.6 * lambda;
    CHECK(arma::abs(u.element_pos.row(1) - y).max() < 1e-12 * 1.2 * lambda);
    CHECK(arma::abs(u.element_pos.row(0)).max() + arma::abs(u.element_pos.row(2)).max() == 0.0);

    CHECK(max_diff(quadriga_lib::arrayant_generate<double>("ula", 10.0, fc, {1, 5}, {0.6}), u) == 0.0);
    CHECK(max_diff(quadriga_lib::generate_arrayant_ula<double>(0, fc, 0.5, {}, 10.0), quadriga_lib::generate_arrayant_ula<double>(1, fc, 0.5, {}, 10.0)) == 0.0);
    CHECK(max_diff(quadriga_lib::generate_arrayant_ula<double>(3, fc, qd_nan, {}, 10.0), quadriga_lib::generate_arrayant_ula<double>(3, fc, 0.5, {}, 10.0)) == 0.0);

    const auto col = quadriga_lib::arrayant_generate<double>("ula", 10.0, fc, {8});
    REQUIRE(col.n_elements() == 8);
    CHECK(arma::abs(col.element_pos.row(1)).max() == 0.0);
    CHECK(std::abs(col.element_pos(2, 7) - 3.5 * 0.5 * lambda) < 1e-12 * lambda);
}

// The ULA and 3GPP generators use only the radiation pattern of a custom pattern; its element positions and
// coupling are ignored (as documented); the elements of the pattern are the fastest index.
TEST_CASE("generate_arrayant_ula - ULA and 3GPP ignore positions and coupling of a custom pattern")
{
    auto p = quadriga_lib::generate_arrayant_omni<double>(10.0);
    quadriga_lib::arrayant_copy_element(p, 0, {1});
    p.element_pos = {{0.3, 0.3}, {0.0, 0.0}, {0.7, -0.7}};
    p.coupling_re = arma::mat(2, 1, arma::fill::ones); // 2 elements -> 1 port
    p.coupling_im = arma::mat(2, 1, arma::fill::zeros);
    p.center_freq = 1e9;

    const double lambda = 299792458.0 / 3e9;
    auto u = quadriga_lib::generate_arrayant_ula<double>(2, 3e9, 0.5, p);
    CHECK(u.n_elements() == 4);
    CHECK(u.n_ports() == 4);
    CHECK(arma::approx_equal(u.coupling_re, arma::mat(4, 4, arma::fill::eye), "absdiff", 0.0));
    CHECK(arma::all(arma::vectorise(u.coupling_im) == 0.0));
    CHECK(arma::all(u.element_pos.row(0) == 0.0));
    CHECK(arma::all(u.element_pos.row(2) == 0.0));
    arma::rowvec y = {-0.25 * lambda, -0.25 * lambda, 0.25 * lambda, 0.25 * lambda};
    CHECK(arma::approx_equal(u.element_pos.row(1), y, "absdiff", 1e-12));
    CHECK(u.center_freq == 3e9);

    auto g = quadriga_lib::generate_arrayant_3GPP<double>({}, 3e9, {}, {}, {}, {}, {}, p);
    CHECK(g.n_elements() == 2);
    CHECK(g.n_ports() == 2);
    CHECK(arma::approx_equal(g.coupling_re, arma::mat(2, 2, arma::fill::eye), "absdiff", 0.0));
    CHECK(arma::all(arma::vectorise(g.element_pos) == 0.0));

    // Vertical stacking: z positions come from the array geometry only
    auto g2 = quadriga_lib::generate_arrayant_3GPP<double>({2}, 3e9, {}, {}, {}, {}, {}, p);
    CHECK(g2.n_elements() == 4);
    CHECK(g2.n_ports() == 4);
    CHECK(arma::all(g2.element_pos.row(0) == 0.0));
    arma::rowvec z = {-0.25 * lambda, -0.25 * lambda, 0.25 * lambda, 0.25 * lambda};
    CHECK(arma::approx_equal(g2.element_pos.row(2), z, "absdiff", 1e-12));
}

TEST_CASE("generate_arrayant_3GPP - float equals double")
{
    // Table 7.8-6A config 1 with 4 columns per port, tilt and pan: patterns 1e-5 absolute, positions 1e-6 of the aperture, coupling phases
    // 1e-6 (1 + k aperture), the float rounding of the phase argument
    const arma::uvec panel = {24, 32}, ports = {8, 8}, pol = {3, 2};
    const arma::vec spacing = {0.7, 0.5};
    const arma::mat steer = {12.0, 7.0};
    const double fc = 7e9, k = 2.0 * arma::datum::pi * fc / 299792458.0;
    const auto d = quadriga_lib::generate_arrayant_3GPP<double>(panel, fc, spacing, ports, pol, steer, {}, {}, 30.0);
    const auto f = quadriga_lib::generate_arrayant_3GPP<float>(panel, fc, spacing, ports, pol, steer, {}, {}, 30.0);
    REQUIRE(f.n_elements() == d.n_elements());
    const double aperture = arma::abs(d.element_pos).max();
    CHECK(arma::abs(arma::conv_to<arma::cube>::from(f.e_theta_re) - d.e_theta_re).max() < 1e-5);
    CHECK(arma::abs(arma::conv_to<arma::cube>::from(f.e_phi_re) - d.e_phi_re).max() < 1e-5);
    CHECK(arma::abs(arma::conv_to<arma::mat>::from(f.element_pos) - d.element_pos).max() < 1e-6 * aperture);
    const arma::cx_mat cd(d.coupling_re, d.coupling_im), cf(arma::conv_to<arma::mat>::from(f.coupling_re), arma::conv_to<arma::mat>::from(f.coupling_im));
    CHECK(arma::abs(cf - cd).max() < 1e-6 * (1.0 + k * aperture));
}

TEST_CASE("generate_arrayant_3GPP - input validation")
{
    using quadriga_lib::generate_arrayant_3GPP;
    const double n = qd_nan, inf = qd_inf;
    REQUIRE_NOTHROW(generate_arrayant_3GPP<double>({4, 2, 1, 2}, 3.5e9, {0.5, 0.5, 0.0, 0.0}, {2, 1}, {3, 2}, {{10.0, 5.0}, {0.0, 0.0}}, {65.0, 65.0, 30.0, 30.0, n}, {}, 30.0));

    SECTION("panel")
    {
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({1, 1, 1, 1, 1}), "Input 'panel' must have at most 4 entries.");
        CHECK_THROWS_AS(generate_arrayant_3GPP<double>({1ULL << 40, 1ULL << 40}), std::invalid_argument);
    }
    SECTION("center_freq")
    {
        for (double fc : {0.0, -1.0, n, inf})
            CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({}, fc), "Input 'center_freq' must be positive and finite.");
    }
    SECTION("spacing")
    {
        CHECK_THROWS_AS(generate_arrayant_3GPP<double>({}, 3.5e9, {0.5, 0.5, 0.5, 0.5, 0.5}), std::invalid_argument);
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({}, 3.5e9, {n, -0.1}), "Input 'spacing' (d_H) must be finite and >= 0.");
        for (double x : {-0.5, inf, -inf})
            for (arma::uword i = 0; i < 4; ++i)
            {
                arma::vec s(4, arma::fill::value(n));
                s(i) = x;
                CHECK_THROWS_AS(generate_arrayant_3GPP<double>({}, 3.5e9, s), std::invalid_argument);
            }
    }
    SECTION("ports")
    {
        CHECK_THROWS_AS(generate_arrayant_3GPP<double>({4, 4}, 3.5e9, {}, {1, 1, 1}), std::invalid_argument);
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({4, 4}, 3.5e9, {}, {3}), "Input 'ports' (M_P) must divide 'panel' (M).");
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({4, 4}, 3.5e9, {}, {0, 8}), "Input 'ports' (N_P) must divide 'panel' (N).");
    }
    SECTION("pol")
    {
        CHECK_THROWS_AS(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {1, 1, 1}), std::invalid_argument);
        for (arma::uword p : {4ULL, 5ULL, 6ULL})
            CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {p}),
                              "Input 'pol' (polarization) values 4 to 6 are reserved; elements are combined through 'ports'.");
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {7}), "Input 'pol' (polarization) must be 1, 2 or 3.");
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {1, 3}), "Input 'pol' (model) must be 1 or 2.");
    }
    SECTION("steer")
    {
        CHECK_THROWS_AS(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {}, {0.0, 0.0, 0.0}), std::invalid_argument);
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({2, 1, 1, 3}, 3.5e9, {}, {}, {}, {{0.0, 0.0}, {0.0, 0.0}}),
                          "Input 'steer' must have one row or one row per panel.");
        for (double x : {91.0, -90.5, inf})
            CHECK_THROWS_AS(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {}, {0.0, x}), std::invalid_argument);
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({2, 2, 1, 2}, 3.5e9, {}, {2, 1}, {}, {{0.0, 5.0}, {1.0, 0.0}}),
                          "Input 'steer' has a non-zero tilt, but 'ports' gives one element per sub-array in the vertical; tilt needs combined elements.");
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({2, 2, 1, 2}, 3.5e9, {}, {1, 2}, {}, {{1.0, 0.0}, {0.0, 5.0}}),
                          "Input 'steer' has a non-zero pan, but 'ports' gives one element per sub-array in the horizontal; pan needs combined elements.");
    }
    SECTION("element")
    {
        CHECK_THROWS_AS(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {}, {}, {65.0, 65.0, 30.0, 30.0, 8.0, 1.0}), std::invalid_argument);
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {}, {}, {0.0}), "Input 'element' (phi_3dB) must be finite and > 0.");
        CHECK_THROWS_AS(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {}, {}, {n, inf}), std::invalid_argument);
        CHECK_THROWS_AS(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {}, {}, {n, n, -1.0}), std::invalid_argument);
        CHECK_THROWS_AS(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {}, {}, {n, n, n, inf}), std::invalid_argument);
        CHECK_THROWS_AS(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {}, {}, {n, n, n, n, -inf}), std::invalid_argument);
    }
    SECTION("pattern")
    {
        auto p = quadriga_lib::generate_arrayant_omni<double>(10.0);
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {}, {}, {65.0}, p), "Inputs 'element' and 'pattern' cannot be given together.");
        quadriga_lib::arrayant_copy_element(p, 0, {1});
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {2}, {}, {}, p), "Input 'pol' must be [1, 1] for a 'pattern' with more than one element.");
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {1, 2}, {}, {}, p), "Input 'pol' must be [1, 1] for a 'pattern' with more than one element.");
        CHECK_THROWS_WITH(generate_arrayant_3GPP<double>({2}, 1e-300, {}, {}, {}, {}, {}, p), "Inputs 'spacing' and 'center_freq' give non-finite element positions.");
        p.azimuth_grid.resize(3);
        CHECK_THROWS_AS(generate_arrayant_3GPP<double>({}, 3.5e9, {}, {}, {}, {}, {}, p), std::invalid_argument);
    }
}

TEST_CASE("generate_arrayant_custom - input validation")
{
    // NaN selects the defaults (90, 90, 0); a beamwidth must be finite and > 0, rear_gain_lin in [0, 1)
    const auto ref = quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0, 10.0);
    CHECK(max_diff(quadriga_lib::generate_arrayant_custom<double>(qd_nan, qd_nan, qd_nan, 10.0), ref) == 0.0);
    for (double bw : {0.0, -5.0, qd_inf})
    {
        CHECK_THROWS_WITH(quadriga_lib::generate_arrayant_custom<double>(bw, 90.0, 0.0, 10.0), "Input 'az_3dB' must be finite and > 0, or NaN for the default.");
        CHECK_THROWS_WITH(quadriga_lib::generate_arrayant_custom<double>(90.0, bw, 0.0, 10.0), "Input 'el_3dB' must be finite and > 0, or NaN for the default.");
    }
    for (double r : {-0.1, 1.0, 1.5, qd_inf})
        CHECK_THROWS_WITH(quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, r, 10.0), "Input 'rear_gain_lin' must be in [0, 1), or NaN for the default.");
}

TEST_CASE("Array antenna generators - grid resolution")
{
    // NaN and values <= 0 select the default (1 degree); infinite values and values below 0.1 or above 90 degree throw
    for (double res : {qd_inf, -qd_inf, 0.09, 1.0e-300, 90.5, 1000.0})
    {
        INFO("res = " << res);
        CHECK_THROWS_AS(quadriga_lib::generate_arrayant_omni<double>(res), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::generate_arrayant_xpol<double>(res), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::generate_arrayant_dipole<double>(res), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::generate_arrayant_half_wave_dipole<double>(res), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::generate_arrayant_custom<double>(90.0, 90.0, 0.0, res), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::generate_arrayant_ula<double>(2, 3.0e9, 0.5, {}, res), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::generate_arrayant_3GPP<double>({}, 3.0e9, {}, {}, {}, {}, {}, {}, res), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::generate_arrayant_multibeam<double>({}, 3.0e9, {}, {}, {}, {}, res), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::arrayant_generate<double>("omni", res), std::invalid_argument);
    }
    CHECK_THROWS_AS(quadriga_lib::generate_arrayant_omni<float>(std::numeric_limits<float>::denorm_min()), std::invalid_argument);
    for (double res : {qd_nan, 0.0, -5.0})
        CHECK(quadriga_lib::generate_arrayant_omni<double>(res).n_azimuth() == 361);
    CHECK(quadriga_lib::generate_arrayant_omni<double>(90.0).n_azimuth() == 5);
}

TEST_CASE("arrayant_generate - each type once, with its default resolution")
{
    // The dispatcher equals the generator of each type (exact); res NaN or <= 0 is 10 degree for omni and xpol, 1 degree
    // otherwise; every result has the given center frequency
    using namespace quadriga_lib;
    const double fc = 2.0e9;
    auto same = [&](const arrayant<double> &a, arrayant<double> b)
    {
        b.center_freq = fc;
        return max_diff(a, b) == 0.0 && a.center_freq == fc;
    };
    CHECK(same(arrayant_generate<double>("omni", qd_nan, fc), generate_arrayant_omni<double>(10.0)));
    CHECK(same(arrayant_generate<double>("xpol", 0.0, fc), generate_arrayant_xpol<double>(10.0)));
    CHECK(same(arrayant_generate<double>("dipole", -1.0, fc), generate_arrayant_dipole<double>(1.0)));
    CHECK(same(arrayant_generate<double>("short-dipole", 5.0, fc), generate_arrayant_dipole<double>(5.0)));
    CHECK(same(arrayant_generate<double>("half-wave-dipole", qd_nan, fc), generate_arrayant_half_wave_dipole<double>(1.0)));
    CHECK(same(arrayant_generate<double>("custom", 10.0, fc, {}, {}, {}, {}, {}, {30.0, 20.0, 10.0}),
               generate_arrayant_custom<double>(30.0, 20.0, std::pow(10.0, -1.0), 10.0)));
    CHECK(same(arrayant_generate<double>("custom", 10.0, fc), generate_arrayant_custom<double>(90.0, 90.0, 0.0, 10.0)));
    CHECK(same(arrayant_generate<double>("3GPP", 10.0, fc, {2, 2}, {}, {1}, {3, 2}, {6.0}), generate_arrayant_3GPP<double>({2, 2}, fc, {}, {1}, {3, 2}, {6.0}, {}, {}, 10.0)));
    CHECK(same(arrayant_generate<double>("3gpp", qd_nan, fc), generate_arrayant_3GPP<double>({}, fc)));
    CHECK(same(arrayant_generate<double>("ula", 10.0, fc, {1, 3}, {}, {}, {}, {}, {}, generate_arrayant_xpol<double>(10.0)),
               generate_arrayant_ula<double>(3, fc, 0.5, generate_arrayant_xpol<double>(10.0))));
    const arma::mat beams = {{10.0, -20.0}, {0.0, 5.0}};
    CHECK(same(arrayant_generate<double>("multibeam", 10.0, fc, {2, 2}, {}, {}, {2}, {}, {}, {}, beams, true, false),
               generate_arrayant_multibeam<double>({2, 2}, fc, {}, {2}, {}, beams, 10.0, true, false)));
    CHECK(arrayant_generate<float>("omni").center_freq == 299792458.0f);
}

TEST_CASE("arrayant_generate - input validation")
{
    using quadriga_lib::arrayant_generate;
    const auto p = quadriga_lib::generate_arrayant_omni<double>(10.0);
    const arma::mat beams = column({0.0, 0.0});
    REQUIRE_NOTHROW(arrayant_generate<double>("3GPP", 30.0, 3.5e9, {2}, {0.5}, {1}, {3}, {5.0}, {}, p, {}, false, true));

    SECTION("type")
    {
        CHECK_THROWS_WITH(arrayant_generate<double>("multibeam_sep"), "Input 'type' has an unsupported value 'multibeam_sep'.");
        CHECK_THROWS_AS(arrayant_generate<double>("OMNI"), std::invalid_argument);
    }
    SECTION("inputs a type does not use")
    {
        CHECK_THROWS_WITH(arrayant_generate<double>("omni", 10.0, 3e9, {2}), "Input 'panel' is not used by type 'omni'.");
        CHECK_THROWS_WITH(arrayant_generate<double>("xpol", 10.0, 3e9, {}, {0.5}), "Input 'spacing' is not used by type 'xpol'.");
        CHECK_THROWS_WITH(arrayant_generate<double>("dipole", 10.0, 3e9, {}, {}, {1}), "Input 'ports' is not used by type 'dipole'.");
        CHECK_THROWS_WITH(arrayant_generate<double>("custom", 10.0, 3e9, {}, {}, {}, {1}), "Input 'pol' is not used by type 'custom'.");
        CHECK_THROWS_WITH(arrayant_generate<double>("multibeam", 10.0, 3e9, {}, {}, {}, {}, {5.0}), "Input 'steer' is not used by type 'multibeam'.");
        CHECK_THROWS_WITH(arrayant_generate<double>("ula", 10.0, 3e9, {}, {}, {}, {}, {}, {65.0}), "Input 'element' is not used by type 'ula'.");
        CHECK_THROWS_WITH(arrayant_generate<double>("multibeam", 10.0, 3e9, {}, {}, {}, {}, {}, {}, p), "Input 'pattern' is not used by type 'multibeam'.");
        CHECK_THROWS_WITH(arrayant_generate<double>("3GPP", 10.0, 3e9, {}, {}, {}, {}, {}, {}, {}, beams), "Input 'beams' is not used by type '3GPP'.");
        CHECK_THROWS_WITH(arrayant_generate<double>("ula", 10.0, 3e9, {}, {}, {}, {}, {}, {}, {}, {}, true), "Input 'separate_beams' is not used by type 'ula'.");
        CHECK_THROWS_WITH(arrayant_generate<double>("half-wave-dipole", 10.0, 3e9, {}, {}, {}, {}, {}, {}, {}, {}, false, true),
                          "Input 'apply_weights' is not used by type 'half-wave-dipole'.");
    }
    SECTION("center_freq")
    {
        for (double fc : {0.0, qd_nan, qd_inf})
            CHECK_THROWS_WITH(arrayant_generate<double>("omni", 10.0, fc), "Input 'center_freq' must be positive and finite.");
    }
    SECTION("element of custom and multibeam")
    {
        CHECK_THROWS_AS(arrayant_generate<double>("custom", 10.0, 3e9, {}, {}, {}, {}, {}, {90.0, 90.0, 10.0, 1.0}), std::invalid_argument);
        CHECK_THROWS_WITH(arrayant_generate<double>("custom", 10.0, 3e9, {}, {}, {}, {}, {}, {0.0}), "Input 'element' (az_3dB) must be finite and > 0.");
        CHECK_THROWS_WITH(arrayant_generate<double>("custom", 10.0, 3e9, {}, {}, {}, {}, {}, {qd_nan, qd_inf}), "Input 'element' (el_3dB) must be finite and > 0.");
        for (double x : {0.0, -3.0, qd_inf})
            CHECK_THROWS_WITH(arrayant_generate<double>("multibeam", 10.0, 3e9, {}, {}, {}, {}, {}, {qd_nan, qd_nan, x}),
                              "Input 'element' (floor_dB) must be finite and > 0, or NaN for no floor.");
    }
    SECTION("beams")
    {
        CHECK_THROWS_WITH(arrayant_generate<double>("multibeam", 10.0, 3e9, {}, {}, {}, {}, {}, {}, {}, arma::mat{{0.0, 1.0}}), "Input 'beams' must have 2 or 3 rows.");
        CHECK_THROWS_AS(arrayant_generate<double>("multibeam", 10.0, 3e9, {}, {}, {}, {}, {}, {}, {}, arma::mat(4, 1, arma::fill::zeros)), std::invalid_argument);
        CHECK_THROWS_WITH(arrayant_generate<double>("multibeam", 10.0, 3e9, {}, {}, {}, {}, {}, {}, {}, column({0.0, qd_nan})), "Input 'beams' must be finite.");
    }
}

TEST_CASE("generate_arrayant_multibeam - a single beam has the MRT gain of the array")
{
    // One beam towards a grid direction: the weights are phasors that align the 36 element contributions there, so the
    // combined port has |F| = 36 |F_element| in that direction (planar wave; MRT is optimal for one beam, so the refinement
    // keeps the requested angles)
    const double az = 20.0, el = -6.0;
    const arma::mat beams = column({az, el});
    const auto ant = quadriga_lib::generate_arrayant_multibeam<double>({6, 6}, 3.75e9, {0.4}, {}, {}, beams, 2.0);
    REQUIRE(ant.n_elements() == 36);
    REQUIRE(ant.n_ports() == 1);
    CHECK(ant.name == "multibeam");
    CHECK(arma::abs(arma::abs(arma::cx_mat(ant.coupling_re, ant.coupling_im)) - 1.0).max() < 1e-12);
    const auto comb = quadriga_lib::arrayant_combine_pattern(ant);
    const arma::uword i = arma::uword(std::lround((el + 90.0) / 2.0)), j = arma::uword(std::lround((az + 180.0) / 2.0));
    REQUIRE(std::abs(comb.azimuth_grid(j) * 180.0 / arma::datum::pi - az) < 1e-9);
    REQUIRE(std::abs(comb.elevation_grid(i) * 180.0 / arma::datum::pi - el) < 1e-9);
    const double F = std::abs(std::complex<double>(comb.e_theta_re(i, j, 0), comb.e_theta_im(i, j, 0)));
    CHECK(std::abs(F / (36.0 * ant.e_theta_re(i, j, 0)) - 1.0) < 1e-9);
}

TEST_CASE("generate_arrayant_multibeam - beams, weights, ports and polarization")
{
    // Two beams with weights 2 and 1 equal three unit beams with the first one twice (the same sum); 2 rows of 'beams'
    // equal weights 1; omitted beams equal [0; 0; 1]. Ports: one per polarization, with separate beams one per
    // polarization and beam (polarization fastest); elements: polarization fastest, the weights equal for both
    const double fc = 3.75e9;
    const arma::mat b2 = {{20.0, 0.0}, {-7.0, 30.0}, {2.0, 1.0}}, b3 = {{20.0, 0.0, 20.0}, {-7.0, 30.0, -7.0}};
    const auto a2 = quadriga_lib::generate_arrayant_multibeam<double>({6, 6}, fc, {0.4}, {}, {}, b2, 15.0);
    const auto a3 = quadriga_lib::generate_arrayant_multibeam<double>({6, 6}, fc, {0.4}, {}, {}, b3, 15.0);
    CHECK(arma::abs(a2.coupling_re - a3.coupling_re).max() < 1e-12);
    CHECK(arma::abs(a2.coupling_im - a3.coupling_im).max() < 1e-12);

    const arma::mat b3w = arma::join_cols(b3, arma::mat(1, 3, arma::fill::ones));
    CHECK(max_diff(a3, quadriga_lib::generate_arrayant_multibeam<double>({6, 6}, fc, {0.4}, {}, {}, b3w, 15.0)) == 0.0);
    CHECK(max_diff(quadriga_lib::generate_arrayant_multibeam<double>({2, 2}, fc), quadriga_lib::generate_arrayant_multibeam<double>({2, 2}, fc, {}, {}, {}, column({0.0, 0.0, 1.0}))) == 0.0);

    const auto s = quadriga_lib::generate_arrayant_multibeam<double>({6, 6}, fc, {0.4}, {}, {}, b3, 15.0, true);
    CHECK(s.n_elements() == 36);
    CHECK(s.n_ports() == 3);

    for (bool sep : {false, true})
    {
        const auto v = quadriga_lib::generate_arrayant_multibeam<double>({3, 2}, fc, {}, {}, {}, b3, 15.0, sep);
        const auto x = quadriga_lib::generate_arrayant_multibeam<double>({3, 2}, fc, {}, {3, 2}, {}, b3, 15.0, sep);
        REQUIRE(x.n_elements() == 12);
        REQUIRE(x.n_ports() == 2 * v.n_ports());
        for (arma::uword e = 0; e < 6; ++e)
            for (arma::uword o = 0; o < v.n_ports(); ++o)
                for (arma::uword p = 0; p < 2; ++p)
                {
                    CHECK(arma::approx_equal(x.element_pos.col(2 * e + p), v.element_pos.col(e), "absdiff", 0.0));
                    CHECK(x.coupling_re(2 * e + p, 2 * o + p) == v.coupling_re(e, o));
                    CHECK(x.coupling_im(2 * e + p, 2 * o + p) == v.coupling_im(e, o));
                    CHECK(x.coupling_re(2 * e + p, 2 * o + 1 - p) == 0.0);
                }
    }

    // apply_weights: one pattern per port
    const auto c = quadriga_lib::generate_arrayant_multibeam<double>({6, 6}, fc, {0.4}, {}, {}, b3, 15.0, false, true);
    CHECK(c.n_elements() == 1);
    CHECK(max_diff(c, quadriga_lib::arrayant_combine_pattern(a3)) < 1e-12 * arma::abs(c.e_theta_re).max());

    // The element shape: NaN beamwidths are 120 degree; a floor of 20 dB
    CHECK(max_diff(quadriga_lib::generate_arrayant_multibeam<double>({2, 2}, fc, {}, {}, {qd_nan, qd_nan, qd_nan}, b3, 10.0),
                   quadriga_lib::generate_arrayant_multibeam<double>({2, 2}, fc, {}, {}, {120.0, 120.0}, b3, 10.0)) == 0.0);
    const auto f = quadriga_lib::generate_arrayant_multibeam<double>({}, fc, {}, {}, {90.0, 60.0, 20.0}, {}, 10.0);
    const auto f_ref = quadriga_lib::generate_arrayant_custom<double>(90.0, 60.0, 0.01, 10.0);
    CHECK(arma::approx_equal(f.e_theta_re, f_ref.e_theta_re, "absdiff", 0.0));
}

// A beam towards a pole, where the element pattern is exactly 0, still gets finite, non-zero beam weights and one
// port per polarization
TEST_CASE("generate_arrayant_multibeam - beams towards the poles")
{
    for (double el : {90.0, -90.0})
        for (arma::uword pol : {1ULL, 2ULL, 3ULL})
        {
            quadriga_lib::arrayant<double> a;
            REQUIRE_NOTHROW(a = quadriga_lib::generate_arrayant_multibeam<double>({2, 2}, 3e9, {}, {pol}, {}, column({0.0, el})));
            const arma::uword n_el = pol == 1 ? 4 : 8;
            CHECK(a.n_elements() == n_el);
            CHECK(a.n_ports() == (pol == 1 ? 1 : 2));
            CHECK(a.is_valid().empty());
            CHECK(arma::is_finite(a.coupling_re));
            CHECK(arma::is_finite(a.coupling_im));
            CHECK(arma::accu(arma::abs(a.coupling_re)) + arma::accu(arma::abs(a.coupling_im)) > 0.0);
        }

    // Separate beams, one of them towards the zenith
    auto b = quadriga_lib::generate_arrayant_multibeam<double>({2, 2}, 3e9, {}, {2}, {}, arma::mat{{0.0, 30.0}, {90.0, 0.0}}, 10.0, true);
    CHECK(b.n_elements() == 8);
    CHECK(b.n_ports() == 4);
    CHECK(b.is_valid().empty());
}

TEST_CASE("generate_arrayant_multibeam - turned polarizations: float equals double, also on the rotation axis")
{
    // The beams of pol = 2 and 3 combine turned elements; their rotation axes lie on the default 1 degree grid
    for (arma::uword pol : {2ULL, 3ULL})
    {
        const auto d = quadriga_lib::generate_arrayant_multibeam<double>({2, 2}, 3.5e9, {}, {pol});
        const auto f = quadriga_lib::generate_arrayant_multibeam<float>({2, 2}, 3.5e9, {}, {pol});
        const double peak = std::max(arma::abs(d.e_theta_re).max(), arma::abs(d.e_phi_re).max());
        const double d_float = std::max({arma::abs(arma::conv_to<arma::cube>::from(f.e_theta_re) - d.e_theta_re).max(),
                                         arma::abs(arma::conv_to<arma::cube>::from(f.e_phi_re) - d.e_phi_re).max(),
                                         arma::abs(arma::conv_to<arma::cube>::from(f.e_theta_im) - d.e_theta_im).max(),
                                         arma::abs(arma::conv_to<arma::cube>::from(f.e_phi_im) - d.e_phi_im).max()});
        INFO("pol = " << pol);
        CHECK(d_float < 2e-5 * peak); // Float rounding of the polarization angle grows as 1 / (distance from the axis)
    }
}

TEST_CASE("generate_arrayant_half_wave_dipole - pattern equals cos(pi/2 sin(el)) / cos(el), also at the poles")
{
    const auto d = quadriga_lib::generate_arrayant_half_wave_dipole<double>();
    const auto f = quadriga_lib::generate_arrayant_half_wave_dipole<float>();
    const double scale = 1.280968208215292, pih = 0.5 * arma::datum::pi;
    double d_model = 0.0, d_float = 0.0;
    for (arma::uword i = 1; i + 1 < d.n_elevation(); ++i)
    {
        const double el = d.elevation_grid(i);
        d_model = std::max(d_model, std::abs(d.e_theta_re(i, 0, 0) - scale * std::cos(pih * std::sin(el)) / std::cos(el)));
    }
    for (arma::uword i = 0; i < d.n_elevation(); ++i)
        d_float = std::max(d_float, std::abs((double)f.e_theta_re(i, 0, 0) - d.e_theta_re(i, 0, 0)));
    CHECK(d_model < 1e-12);
    CHECK(d_float < 1e-6);
    for (arma::uword i : {0ULL, 180ULL})
    {
        CHECK(std::abs(d.e_theta_re(i, 7, 0)) < 1e-12); // Limit 0 at the poles
        CHECK(std::abs(f.e_theta_re(i, 7, 0)) < 1e-6f);
    }
}
