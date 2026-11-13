// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include <catch2/catch_test_macros.hpp>
#include "quadriga_tools.hpp"
#include "quadriga_arrayant.hpp"

#include <cmath>

TEST_CASE("calc_angular_spreads_sphere - Single path returns zero spread")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.5};
    el[0] = {0.3};
    powers[0] = {1.0};

    arma::vec as, es;
    arma::mat orient;

    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, powers, &as, &es, &orient, nullptr, nullptr, false, true);

    REQUIRE(as.n_elem == 1);
    REQUIRE(es.n_elem == 1);
    CHECK(as(0) < 1e-10);
    CHECK(es(0) < 1e-10);
    CHECK(std::abs(orient(2, 0) - 0.5) < 1e-10);  // heading ~ azimuth
    CHECK(std::abs(orient(1, 0) + 0.3) < 1e-10);  // tilt = -elevation (a positive tilt points down)
}

TEST_CASE("calc_angular_spreads_sphere - Two symmetric azimuth paths")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.1, -0.1};
    el[0] = {0.0, 0.0};
    powers[0] = {1.0, 1.0};

    arma::vec as, es;
    arma::mat orient;

    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, powers, &as, &es, &orient, nullptr, nullptr, false, true);

    CHECK(std::abs(as(0) - 0.1) < 1e-6);
    CHECK(es(0) < 1e-6);
    CHECK(std::abs(orient(2, 0)) < 1e-6);
}

TEST_CASE("calc_angular_spreads_sphere - Two symmetric elevation paths")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.0, 0.0};
    el[0] = {0.2, -0.2};
    powers[0] = {1.0, 1.0};

    arma::vec as, es;
    arma::mat orient;
    std::vector<arma::vec> phi, theta;

    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, powers, &as, &es, &orient, &phi, &theta, false, true);

    REQUIRE(phi.size() == 1);
    REQUIRE(phi[0].n_elem == 2);

    // After bank rotation, AS >= ES
    CHECK(as(0) >= es(0) - 1e-6);
    // Total angular extent preserved
    double total = std::sqrt(as(0) * as(0) + es(0) * es(0));
    CHECK(std::abs(total - 0.2) < 1e-4);
}

TEST_CASE("calc_angular_spreads_sphere - Pole paths (near zenith)")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.0, 1.5707963, 3.1415927, -1.5707963};
    el[0] = {1.4, 1.4, 1.4, 1.4};
    powers[0] = {1.0, 1.0, 1.0, 1.0};

    arma::vec as, es;
    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, powers, &as, &es, nullptr, nullptr, nullptr, false, true);

    CHECK(as(0) < 0.5);
    CHECK(es(0) < 0.5);
}

TEST_CASE("calc_angular_spreads_sphere - Variable path counts per CIR")
{
    std::vector<arma::vec> az(3), el(3), powers(3);
    az[0] = {0.1, -0.1};                    // 2 paths
    el[0] = {0.0, 0.0};
    powers[0] = {1.0, 1.0};
    az[1] = {0.2, -0.2, 0.0};              // 3 paths
    el[1] = {0.0, 0.0, 0.0};
    powers[1] = {1.0, 1.0, 1.0};
    az[2] = {1.0, -1.0, 0.5, -0.5, 0.0};  // 5 paths
    el[2] = {0.0, 0.0, 0.0, 0.0, 0.0};
    powers[2] = {1.0, 1.0, 1.0, 1.0, 1.0};

    arma::vec as, es;
    std::vector<arma::vec> phi, theta;

    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as, &es,
                                              (arma::mat *)nullptr, &phi, &theta, false, true);

    REQUIRE(as.n_elem == 3);
    REQUIRE(phi.size() == 3);
    CHECK(phi[0].n_elem == 2);
    CHECK(phi[1].n_elem == 3);
    CHECK(phi[2].n_elem == 5);
    CHECK(as(0) < as(1));
    CHECK(as(1) < as(2));
}

TEST_CASE("calc_angular_spreads_sphere - No bank angle calculation")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.0, 0.0};
    el[0] = {0.2, -0.2};
    powers[0] = {1.0, 1.0};

    arma::vec as_bank, es_bank, as_nobank, es_nobank;
    arma::mat orient_bank, orient_nobank;

    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as_bank, &es_bank, &orient_bank,
                                              (std::vector<arma::vec> *)nullptr,
                                              (std::vector<arma::vec> *)nullptr,
                                              false, true);
    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as_nobank, &es_nobank, &orient_nobank,
                                              (std::vector<arma::vec> *)nullptr,
                                              (std::vector<arma::vec> *)nullptr,
                                              false, false);

    CHECK(std::abs(orient_nobank(0, 0)) < 1e-10); // bank = 0 when disabled
    CHECK(as_bank(0) >= as_nobank(0) - 1e-6);
}

TEST_CASE("calc_angular_spreads_sphere - Disable wrapping")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.1, -0.1, 0.05};
    el[0] = {0.2, -0.2, 0.0};
    powers[0] = {1.0, 1.0, 1.0};

    arma::vec as_wrap, es_wrap, as_raw, es_raw;
    arma::mat orient_raw;
    std::vector<arma::vec> phi_raw, theta_raw;

    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, powers, &as_wrap, &es_wrap, nullptr, nullptr, nullptr, false, true);
    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as_raw, &es_raw, &orient_raw,
                                              &phi_raw, &theta_raw, true);

    // Orientation should be zero when wrapping is disabled
    CHECK(std::abs(orient_raw(0, 0)) < 1e-10);
    CHECK(std::abs(orient_raw(1, 0)) < 1e-10);
    CHECK(std::abs(orient_raw(2, 0)) < 1e-10);

    // phi/theta should equal the input az/el
    REQUIRE(phi_raw.size() == 1);
    CHECK(arma::approx_equal(phi_raw[0], az[0], "absdiff", 1e-14));
    CHECK(arma::approx_equal(theta_raw[0], el[0], "absdiff", 1e-14));
}

TEST_CASE("calc_angular_spreads_sphere - Quantization groups nearby paths")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.0, 0.01};
    el[0] = {0.0, 0.0};
    powers[0] = {1.0, 1.0};

    arma::vec as_raw, as_quant;

    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as_raw,
                                              (arma::vec *)nullptr, (arma::mat *)nullptr,
                                              (std::vector<arma::vec> *)nullptr,
                                              (std::vector<arma::vec> *)nullptr,
                                              false, true, 0.0);
    quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as_quant,
                                              (arma::vec *)nullptr, (arma::mat *)nullptr,
                                              (std::vector<arma::vec> *)nullptr,
                                              (std::vector<arma::vec> *)nullptr,
                                              false, true, 3.0);

    CHECK(as_quant(0) <= as_raw(0) + 1e-8);
}

TEST_CASE("calc_angular_spreads_sphere - Wrap-around at +/- pi")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {3.0, -3.0};
    el[0] = {0.0, 0.0};
    powers[0] = {1.0, 1.0};

    arma::vec as, es;
    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, powers, &as, &es, nullptr, nullptr, nullptr, false, true);

    double gap = 2.0 * (3.14159265358979 - 3.0);
    CHECK(as(0) < gap);
    CHECK(as(0) > 0.0);
}

TEST_CASE("calc_angular_spreads_sphere - Float precision")
{
    std::vector<arma::fvec> az(1), el(1), powers(1);
    az[0] = {0.1f, -0.1f, 0.0f};
    el[0] = {0.0f, 0.0f, 0.0f};
    powers[0] = {1.0f, 1.0f, 2.0f};

    arma::fvec as, es;
    quadriga_lib::calc_angular_spreads_sphere<float>(az, el, powers, &as, &es, nullptr, nullptr, nullptr, false, true);

    REQUIRE(as.n_elem == 1);
    CHECK(as(0) > 0.0f);
    CHECK(es(0) < 0.01f);
}

TEST_CASE("calc_angular_spreads_sphere - Output shapes and valid ranges")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.3, -0.2, 0.1, -0.4, 0.0};
    el[0] = {0.1, -0.1, 0.05, -0.05, 0.0};
    powers[0] = {1.0, 1.0, 1.0, 1.0, 1.0};

    arma::vec as, es;
    arma::mat orient;
    std::vector<arma::vec> phi, theta;

    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, powers, &as, &es, &orient, &phi, &theta, false, true);

    REQUIRE(as.n_elem == 1);
    REQUIRE(es.n_elem == 1);
    REQUIRE(orient.n_rows == 3);
    REQUIRE(orient.n_cols == 1);
    REQUIRE(phi.size() == 1);
    REQUIRE(phi[0].n_elem == 5);
    REQUIRE(theta.size() == 1);
    REQUIRE(theta[0].n_elem == 5);

    for (arma::uword i = 0; i < 5; i++)
    {
        CHECK(phi[0](i) >= -3.15);
        CHECK(phi[0](i) <= 3.15);
        CHECK(theta[0](i) >= -1.58);
        CHECK(theta[0](i) <= 1.58);
    }
}

TEST_CASE("calc_angular_spreads_sphere - Error on empty input")
{
    std::vector<arma::vec> az, el, powers;
    arma::vec as;
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as),
                    std::invalid_argument);
}

TEST_CASE("calc_angular_spreads_sphere - Error on mismatched vector sizes")
{
    std::vector<arma::vec> az(2), el(1), powers(2);
    az[0] = {0.1};
    az[1] = {0.2};
    el[0] = {0.0};
    powers[0] = {1.0};
    powers[1] = {1.0};

    arma::vec as;
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as),
                    std::invalid_argument);
}

TEST_CASE("calc_angular_spreads_sphere - Error on mismatched path count within CIR")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.1, 0.2};
    el[0] = {0.0};         // Length mismatch
    powers[0] = {1.0, 1.0};

    arma::vec as;
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere(az, el, powers, &as),
                    std::invalid_argument);
}

TEST_CASE("calc_angular_spreads_sphere - Selective outputs (nullptr)")
{
    std::vector<arma::vec> az(1), el(1), powers(1);
    az[0] = {0.1, -0.1, 0.0};
    el[0] = {0.0, 0.0, 0.0};
    powers[0] = {1.0, 1.0, 1.0};

    arma::vec as;
    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, powers, &as,
                                                      (arma::vec *)nullptr,
                                                      (arma::mat *)nullptr,
                                                      (std::vector<arma::vec> *)nullptr,
                                                      (std::vector<arma::vec> *)nullptr,
                                                      false, true, 0.0);
    CHECK(as(0) > 0.0);
}

TEST_CASE("calc_angular_spreads_sphere - quantize must be 0 or a finite value >= 1e-3 deg")
{
    std::vector<arma::vec> az = {{0.1, 0.2}}, el = {{0.0, 0.05}}, pw = {{1.0, 1.0}};
    arma::vec as;
    auto run = [&](double q, bool wrap)
    {
        quadriga_lib::calc_angular_spreads_sphere<double>(az, el, pw, &as, nullptr, nullptr, nullptr, nullptr, !wrap, true, q);
    };
    for (bool wrap : {true, false})
    {
        INFO("wrapping " << wrap);
        for (double q : {1.0e-8, 9.99e-4, -1.0, -1.0e-8, (double)NAN, (double)INFINITY, -(double)INFINITY})
        {
            INFO("quantize " << q);
            CHECK_THROWS_AS(run(q, wrap), std::invalid_argument);
        }
        CHECK_NOTHROW(run(0.0, wrap));
        CHECK_NOTHROW(run(1.0e-3, wrap));
        CHECK(std::isfinite(as(0)));
        CHECK_NOTHROW(run(400.0, wrap));
    }
    std::vector<arma::fvec> azf = {{0.1f, 0.2f}}, elf = {{0.0f, 0.05f}}, pwf = {{1.0f, 1.0f}};
    arma::fvec asf;
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere<float>(azf, elf, pwf, &asf, nullptr, nullptr, nullptr, nullptr,
                                                                     false, true, 1.0e-8f),
                    std::invalid_argument);
    CHECK_NOTHROW(quadriga_lib::calc_angular_spreads_sphere<float>(azf, elf, pwf, &asf, nullptr, nullptr, nullptr, nullptr,
                                                                   false, true, 1.0e-3f));
}

TEST_CASE("calc_angular_spreads_sphere - Defaults: wrapping off, bank angle off (as in MATLAB and Python)")
{
    std::vector<arma::vec> az = {{0.1, 2.9, -3.0}}, el = {{0.1, -0.2, 0.3}}, pw = {{1.0, 0.8, 0.5}};
    arma::vec as, es, as0, es0;
    arma::mat orient;
    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, pw, &as, &es, &orient);
    quadriga_lib::calc_angular_spreads_sphere<double>(az, el, pw, &as0, &es0, nullptr, nullptr, nullptr, true, false);
    CHECK(as(0) == as0(0));
    CHECK(es(0) == es0(0));
    CHECK(std::abs(as(0) - 1.627282) < 1.0e-6); // MATLAB / Python default result on this input
    CHECK(std::abs(es(0) - 0.190512) < 1.0e-6);
    CHECK(arma::all(arma::vectorise(orient) == 0.0)); // no rotation without wrapping
}

TEST_CASE("calc_angular_spreads_sphere - outputs that alias inputs")
{
    std::vector<arma::vec> az = {arma::vec({0.1, -0.1, 0.3})}, el = {arma::vec({0.0, 0.1, -0.1})}, pw = {arma::vec({1.0, 1.0, 0.5})};
    arma::vec as, es;
    arma::mat orient;
    std::vector<arma::vec> phi, theta;
    CHECK_NOTHROW(quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &es, &orient, &phi, &theta, false));

    auto az2 = az;
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere(az2, el, pw, &as, &es, &orient, &az2, &theta, false), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &es, &orient, &phi, &phi, false), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &as, &as), std::invalid_argument);
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere(az2, el, pw, &az2[0]), std::invalid_argument);
    CHECK(arma::approx_equal(az2[0], az[0], "absdiff", 0.0));

    arma::vec view(orient.memptr(), 1, false, true);
    CHECK_THROWS_AS(quadriga_lib::calc_angular_spreads_sphere(az, el, pw, &view, (arma::vec *)nullptr, &orient), std::invalid_argument);
}

TEST_CASE("calc_angular_spreads_sphere - phi and theta are the local angles of the output orientation")
{
    // An antenna with the output orientation points at the mean direction; the rotated angles equal the local angles
    // that arrayant_interpolate returns for this orientation, with and without the bank angle. A mean direction 20
    // degree above the horizon gives tilt -20 degree (a positive tilt points down).
    const double d = arma::datum::pi / 180.0;
    const std::vector<arma::vec> az = {arma::vec{28.0 * d, 32.0 * d, 30.0 * d, 35.0 * d}};
    const std::vector<arma::vec> el = {arma::vec{18.0 * d, 22.0 * d, 20.0 * d, 19.0 * d}};
    const std::vector<arma::vec> pw = {arma::vec{1.0, 1.0, 1.0, 2.0}};
    const auto omni = quadriga_lib::generate_arrayant_omni<double>(10.0);
    for (bool bank : {false, true})
    {
        arma::mat ori;
        std::vector<arma::vec> phi, theta;
        quadriga_lib::calc_angular_spreads_sphere<double>(az, el, pw, nullptr, nullptr, &ori, &phi, &theta, false, bank);
        CHECK(std::abs(ori(1, 0) + 19.6 * d) < 0.1 * d);
        const arma::mat A = az[0].t(), E = el[0].t();
        const arma::cube O(ori.memptr(), 3, 1, 1);
        arma::mat Vr, Vi, Hr, Hi, az_loc, el_loc;
        const arma::uvec *i_el = nullptr;
        const arma::mat *pos = nullptr;
        arma::mat *dist = nullptr;
        quadriga_lib::arrayant_interpolate(omni, A, E, Vr, Vi, Hr, Hi, i_el, &O, pos, dist, &az_loc, &el_loc);
        CHECK(arma::approx_equal(phi[0], arma::vec(az_loc.t()), "absdiff", 1e-6));
        CHECK(arma::approx_equal(theta[0], arma::vec(el_loc.t()), "absdiff", 1e-6));
    }
}
