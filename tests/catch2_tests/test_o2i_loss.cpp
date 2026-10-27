// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "o2i_loss*"

// Verification suite for quadriga_lib::o2i_loss.
//
// Pins down: the building penetration loss of the low-loss, high-loss and low-loss A models, the backward
// compatible model, the car penetration losses, the model choice by the cumulative weights, the indoor distance
// as the minimum of two uniforms, the optional outputs, pre-sized outputs, the empty input and the input
// validation.
// Oracles, independent of the function:
//  - TR 38.901 v19.5.0 Table 7.4.3-1 (material losses, f in GHz: glass 2 + 0.2 f, IRR glass 25.4 + 0.11 f,
//    concrete 5 + 4 f, plywood 1.03 + 0.17 f), Table 7.4.3-2 (PL_tw = 5 - 10·lg(sum p_i·10^(-L_i/10)),
//    PL_in = 0.5·d2D-in, sigma_P 4.4 / 6.5 / 4.4 dB), Table 7.4.3-3 (PL_tw = 20 dB, one uniform d2D-in,
//    sigma_P = 0), clause 7.4.3.2 (car: N(9, 5²), metallized windows N(20, 5²)), clause 7.4.3.1 (d2D-in is the
//    minimum of two uniforms between 0 and the maximum indoor distance)
//  - the model choice rule: the smallest j with u < (w_0 + ... + w_j) / sum(w); if there is none, the last j with
//    w_j > 0
// Tolerances: dB values from short chains of lg and pow 1e-12; distances 1e-12; the distribution of the minimum of
// two uniforms on a 100 x 100 grid of mid-points within 2/100.
// Conventions: frequencies in Hz at the interface, f in GHz in formulas; rnd columns 0 to 2 are Uniform(0,1)
// (model choice, d2D-in #1, d2D-in #2), column 3 is Normal(0,1) (deviation).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "quadriga_gbsm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

using Catch::Matchers::Message;

namespace
{
    const double inf_o2i = std::numeric_limits<double>::infinity();

    // TR 38.901 Table 7.4.3-2 wall loss: 5 - 10·lg(p1·10^(-L1/10) + p2·10^(-L2/10))
    double tw(double p1, double L1, double p2, double L2)
    {
        return 5.0 - 10.0 * std::log10(p1 * std::pow(10.0, -L1 / 10.0) + p2 * std::pow(10.0, -L2 / 10.0));
    }

    // Oracle of one building model; f in GHz; returns PL_tw + PL_in + sigma_P·N
    double ref_building(const std::string &model, double f, double U1, double U2, double N, double d_max)
    {
        const double L_glass = 2.0 + 0.2 * f, L_irr = 25.4 + 0.11 * f, L_conc = 5.0 + 4.0 * f,
                     L_ply = 1.03 + 0.17 * f;
        const double din = std::min(U1, U2) * d_max;
        if (model == "low")
            return tw(0.3, L_glass, 0.7, L_conc) + 0.5 * din + 4.4 * N;
        if (model == "high")
            return tw(0.7, L_irr, 0.3, L_conc) + 0.5 * din + 6.5 * N;
        if (model == "low_A")
            return tw(0.3, L_glass, 0.7, L_ply) + 0.5 * din + 4.4 * N;
        return NAN;
    }

    // Random values of 5 UTs: U model, U1, U2, N
    const arma::mat rnd5 = {{0.10, 0.30, 0.70, 0.50},
                            {0.50, 0.90, 0.20, -1.30},
                            {0.99, 0.05, 0.95, 2.10},
                            {0.00, 1.00, 1.00, 0.00},
                            {1.00, 0.00, 0.40, -0.70}};
}

TEST_CASE("o2i_loss - wall and indoor losses follow Tables 7.4.3-1 and 7.4.3-2")
{
    const arma::vec fc = {3.5e9, 28.0e9, 60.0e9};
    const std::vector<std::string> models = {"low", "high", "low_A"};

    QD_DIAG("\n# diag: o2i_loss building models vs TR Tables 7.4.3-1 and 7.4.3-2\n");
    QD_DIAG("model,d_max,ut,f_GHz,loss,ref,diff\n");
    for (arma::uword j = 0; j < 3; ++j)
        for (double d_max : {25.0, 10.0})
        {
            arma::vec w(3, arma::fill::zeros);
            w(j) = 1.0;
            arma::mat loss;
            arma::vec d2d_in;
            arma::uvec idx;
            quadriga_lib::o2i_loss(fc, rnd5, models, w, loss, d_max, &d2d_in, &idx);
            REQUIRE(loss.n_rows == 5);
            REQUIRE(loss.n_cols == 3);
            REQUIRE(d2d_in.n_elem == 5);
            REQUIRE(idx.n_elem == 5);
            for (arma::uword u = 0; u < 5; ++u)
            {
                INFO("model " << models[j] << ", d_max " << d_max << ", UT " << u);
                CHECK(idx(u) == j);
                CHECK(std::abs(d2d_in(u) - std::min(rnd5(u, 1), rnd5(u, 2)) * d_max) < 1e-12);
                for (arma::uword k = 0; k < 3; ++k)
                {
                    const double f = fc(k) / 1e9;
                    double ref = ref_building(models[j], f, rnd5(u, 1), rnd5(u, 2), rnd5(u, 3), d_max);
                    QD_DIAG("%s,%g,%llu,%g,%.12f,%.12f,%.3e\n", models[j].c_str(), d_max, (unsigned long long)u, f,
                            loss(u, k), ref, loss(u, k) - ref);
                    CHECK(std::abs(loss(u, k) - ref) < 1e-12);
                }
            }
        }
}

TEST_CASE("o2i_loss - the backward compatible model of Table 7.4.3-3")
{
    const arma::vec fc = {2.0e9, 3.5e9, 5.9e9};
    arma::mat loss, loss2;
    arma::vec d2d_in;
    quadriga_lib::o2i_loss(fc, rnd5, {"legacy_6GHz"}, arma::vec{1.0}, loss, 25.0, &d2d_in);
    REQUIRE(loss.n_rows == 5);
    REQUIRE(loss.n_cols == 3);
    for (arma::uword u = 0; u < 5; ++u)
        for (arma::uword k = 0; k < 3; ++k)
        {
            INFO("UT " << u << ", fc " << fc(k));
            CHECK(std::abs(loss(u, k) - (20.0 + 0.5 * rnd5(u, 1) * 25.0)) < 1e-12);
            CHECK(std::abs(d2d_in(u) - rnd5(u, 1) * 25.0) < 1e-12);
        }

    // Columns 2 and 3 have no influence
    arma::mat r = rnd5;
    r.col(2) = arma::vec{0.9, 0.0, 0.3, 0.6, 0.1};
    r.col(3) = arma::vec{-2.0, 3.0, 0.1, 1.0, -1.0};
    arma::vec d2d_in2;
    quadriga_lib::o2i_loss(fc, r, {"legacy_6GHz"}, arma::vec{1.0}, loss2, 25.0, &d2d_in2);
    CHECK(arma::all(arma::vectorise(loss2 == loss)));
    CHECK(arma::all(d2d_in2 == d2d_in));
}

TEST_CASE("o2i_loss - car penetration loss")
{
    const arma::vec fc = {0.6e9, 3.5e9, 28.0e9, 60.0e9};
    for (const char *m : {"car", "car_metal"})
    {
        const double mu = std::string(m) == "car" ? 9.0 : 20.0;
        arma::mat loss;
        arma::vec d2d_in;
        quadriga_lib::o2i_loss(fc, rnd5, {m}, arma::vec{1.0}, loss, 25.0, &d2d_in);
        REQUIRE(loss.n_rows == 5);
        REQUIRE(loss.n_cols == 4);
        for (arma::uword u = 0; u < 5; ++u)
        {
            INFO("model " << m << ", UT " << u);
            CHECK(d2d_in(u) == 0.0);
            CHECK(std::abs(loss(u, 0) - (mu + 5.0 * rnd5(u, 3))) < 1e-12);
            for (arma::uword k = 1; k < 4; ++k)
                CHECK(loss(u, k) == loss(u, 0)); // independent of f
        }
    }
}

TEST_CASE("o2i_loss - the model is chosen by the cumulative weights")
{
    const arma::vec fc = {3.5e9};
    const std::vector<std::string> models = {"low", "high", "low_A", "car"};

    // 1000 UTs with u = (k + 0.5)/1000, weights {0.2, 0, 0.5, 0.3}
    const arma::uword n = 1000;
    arma::mat r(n, 4, arma::fill::value(0.5));
    for (arma::uword k = 0; k < n; ++k)
        r(k, 0) = (double(k) + 0.5) / double(n);
    arma::mat loss;
    arma::uvec idx;
    quadriga_lib::o2i_loss(fc, r, models, arma::vec{0.2, 0.0, 0.5, 0.3}, loss, 25.0, nullptr, &idx);
    REQUIRE(idx.n_elem == n);
    arma::uvec cnt(4, arma::fill::zeros);
    for (arma::uword k = 0; k < n; ++k)
        if (idx(k) < 4)
            ++cnt(idx(k));
    QD_DIAG("\n# diag: model counts for weights {0.2, 0, 0.5, 0.3}\nlow,high,low_A,car\n%llu,%llu,%llu,%llu\n",
            (unsigned long long)cnt(0), (unsigned long long)cnt(1), (unsigned long long)cnt(2), (unsigned long long)cnt(3));
    CHECK(cnt(0) == 200);
    CHECK(cnt(1) == 0);
    CHECK(cnt(2) == 500);
    CHECK(cnt(3) == 300);
    // The choice is monotone in u: the first 200 are 'low', then 'low_A', then 'car'
    CHECK(idx(199) == 0);
    CHECK(idx(200) == 2);
    CHECK(idx(699) == 2);
    CHECK(idx(700) == 3);

    // u = 1 picks the last non-zero weight; u = 0 never picks a zero weight
    const arma::mat r1 = {{1.0, 0.5, 0.5, 0.0}}, r0 = {{0.0, 0.5, 0.5, 0.0}};
    quadriga_lib::o2i_loss(fc, r1, models, arma::vec{0.5, 0.5, 0.0, 0.0}, loss, 25.0, nullptr, &idx);
    CHECK(idx(0) == 1);
    quadriga_lib::o2i_loss(fc, r1, models, arma::vec{0.25, 0.25, 0.25, 0.25}, loss, 25.0, nullptr, &idx);
    CHECK(idx(0) == 3);
    quadriga_lib::o2i_loss(fc, r0, models, arma::vec{0.0, 0.0, 0.6, 0.4}, loss, 25.0, nullptr, &idx);
    CHECK(idx(0) == 2);
    quadriga_lib::o2i_loss(fc, r0, models, arma::vec{0.0, 1.0, 0.0, 0.0}, loss, 25.0, nullptr, &idx);
    CHECK(idx(0) == 1);

    // Weights that sum to 1 within 1e-6 are normalized by their sum
    const arma::mat rh = {{0.4999999, 0.5, 0.5, 0.0}, {0.5000001, 0.5, 0.5, 0.0}};
    quadriga_lib::o2i_loss(fc, rh, {"low", "high"}, arma::vec{0.5000004, 0.5000004}, loss, 25.0, nullptr, &idx);
    CHECK(idx(0) == 0);
    CHECK(idx(1) == 1);
}

TEST_CASE("o2i_loss - indoor distance is the minimum of two uniforms")
{
    const arma::vec fc = {3.5e9};
    const double d_max = 25.0;
    const arma::uword n = 100;
    arma::mat r(n * n, 4, arma::fill::zeros);
    for (arma::uword i = 0; i < n; ++i)
        for (arma::uword j = 0; j < n; ++j)
        {
            r(i * n + j, 0) = (double(i) + 0.5) / double(n); // mixes low and high
            r(i * n + j, 1) = (double(i) + 0.5) / double(n);
            r(i * n + j, 2) = (double(j) + 0.5) / double(n);
        }
    arma::mat loss;
    arma::vec d2d_in;
    quadriga_lib::o2i_loss(fc, r, {"low", "high"}, arma::vec{0.4, 0.6}, loss, d_max, &d2d_in);
    REQUIRE(d2d_in.n_elem == n * n);

    double max_err = 0.0;
    for (arma::uword i = 0; i < n * n; ++i)
        max_err = std::max(max_err, std::abs(d2d_in(i) - std::min(r(i, 1), r(i, 2)) * d_max));
    CHECK(max_err < 1e-12);

    QD_DIAG("\n# diag: CDF of d2D-in / d_max on a 100 x 100 grid\nx,fraction,1-(1-x)^2,diff\n");
    for (double x : {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99})
    {
        double frac = double(arma::accu(d2d_in <= x * d_max)) / double(n * n);
        double ref = 1.0 - (1.0 - x) * (1.0 - x);
        QD_DIAG("%g,%.4f,%.4f,%.4f\n", x, frac, ref, frac - ref);
        INFO("x " << x);
        CHECK(std::abs(frac - ref) <= 0.02);
    }
}

TEST_CASE("o2i_loss - optional outputs and sizes")
{
    const arma::vec fc = {3.5e9, 28.0e9};
    const std::vector<std::string> models = {"low", "high", "legacy_6GHz", "car"};
    const arma::vec w = {0.3, 0.3, 0.2, 0.2};

    arma::mat L_ref;
    arma::vec d_ref;
    arma::uvec i_ref;
    quadriga_lib::o2i_loss(fc, rnd5, models, w, L_ref, 20.0, &d_ref, &i_ref);
    REQUIRE(L_ref.n_rows == 5);
    REQUIRE(L_ref.n_cols == 2);
    REQUIRE(d_ref.n_elem == 5);
    REQUIRE(i_ref.n_elem == 5);

    // Each optional output alone gives the same values
    arma::mat L1, L2, L3;
    arma::vec d1;
    arma::uvec i1;
    quadriga_lib::o2i_loss(fc, rnd5, models, w, L1, 20.0);
    quadriga_lib::o2i_loss(fc, rnd5, models, w, L2, 20.0, &d1);
    quadriga_lib::o2i_loss(fc, rnd5, models, w, L3, 20.0, nullptr, &i1);
    CHECK(arma::all(arma::vectorise(L1 == L_ref)));
    CHECK(arma::all(arma::vectorise(L2 == L_ref)));
    CHECK(arma::all(arma::vectorise(L3 == L_ref)));
    CHECK(arma::all(d1 == d_ref));
    CHECK(arma::all(i1 == i_ref));

    // Pre-sized outputs with stale content are overwritten in place
    arma::mat Lp(5, 2, arma::fill::value(7.0));
    arma::vec dp(5, arma::fill::value(7.0));
    arma::uvec ip(5, arma::fill::value(7));
    const double *pL = Lp.memptr(), *pd = dp.memptr();
    const arma::uword *pi = ip.memptr();
    quadriga_lib::o2i_loss(fc, rnd5, models, w, Lp, 20.0, &dp, &ip);
    CHECK(Lp.memptr() == pL);
    CHECK(dp.memptr() == pd);
    CHECK(ip.memptr() == pi);
    CHECK(arma::all(arma::vectorise(Lp == L_ref)));
    CHECK(arma::all(dp == d_ref));
    CHECK(arma::all(ip == i_ref));

    // Wrongly sized outputs are resized
    arma::mat Lw(2, 7, arma::fill::value(7.0));
    arma::vec dw(1, arma::fill::value(7.0));
    arma::uvec iw(9, arma::fill::value(7));
    quadriga_lib::o2i_loss(fc, rnd5, models, w, Lw, 20.0, &dw, &iw);
    REQUIRE(Lw.n_rows == 5);
    REQUIRE(Lw.n_cols == 2);
    CHECK(arma::all(arma::vectorise(Lw == L_ref)));
    CHECK(arma::all(dw == d_ref));
    CHECK(arma::all(iw == i_ref));

    // Repeated call: identical (Lw, dw, iw were computed by an earlier call)
    arma::mat Lr;
    quadriga_lib::o2i_loss(fc, rnd5, models, w, Lr, 20.0);
    CHECK(arma::all(arma::vectorise(Lr == Lw)));

    // No UTs
    const arma::mat r0(0, 4);
    arma::mat L0(3, 3, arma::fill::ones);
    arma::vec d0(3, arma::fill::ones);
    arma::uvec i0(3, arma::fill::ones);
    CHECK_NOTHROW(quadriga_lib::o2i_loss(fc, r0, models, w, L0, 20.0, &d0, &i0));
    CHECK(L0.n_rows == 0);
    CHECK(L0.n_cols == 2);
    CHECK(d0.n_elem == 0);
    CHECK(i0.n_elem == 0);
}

TEST_CASE("o2i_loss - input validation")
{
    const arma::vec fc = {3.5e9, 28.0e9};
    const std::vector<std::string> models = {"low", "high"};
    const arma::vec w = {0.4, 0.6};
    arma::mat loss;

    auto call = [&](const arma::vec &f, const arma::mat &r, const std::vector<std::string> &m, const arma::vec &ww,
                    double d_max)
    { quadriga_lib::o2i_loss(f, r, m, ww, loss, d_max); };

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(call(fc, rnd5, models, w, 25.0));
        CHECK_NOTHROW(call(fc, rnd5, {"low", "high", "low_A", "legacy_6GHz", "car", "car_metal"}, arma::vec{0.1, 0.1, 0.2, 0.2, 0.2, 0.2}, 0.0));
        CHECK_NOTHROW(call(fc, rnd5, models, arma::vec{0.4, 0.6 + 9e-7}, 25.0));
        CHECK_NOTHROW(call(fc, arma::mat(0, 4), models, w, 25.0));
        arma::vec d2d_in;
        arma::uvec idx;
        CHECK_NOTHROW(quadriga_lib::o2i_loss(fc, rnd5, models, w, loss, 25.0, &d2d_in, &idx));
        CHECK_NOTHROW(quadriga_lib::o2i_loss(fc, rnd5, models, w, loss));
    }
    SECTION("fc")
    {
        CHECK_THROWS_MATCHES(call(arma::vec(), rnd5, models, w, 25.0), std::invalid_argument,
                             Message("Input 'fc' cannot be empty."));
        for (double v : {0.0, -1.0, double(NAN), inf_o2i})
        {
            INFO("value " << v);
            CHECK_THROWS_MATCHES(call(arma::vec{3.5e9, v}, rnd5, models, w, 25.0), std::invalid_argument,
                                 Message("Input 'fc' must be finite and > 0."));
        }
    }
    SECTION("rnd")
    {
        CHECK_THROWS_MATCHES(call(fc, arma::mat(5, 6, arma::fill::value(0.5)), models, w, 25.0), std::invalid_argument,
                             Message("Input 'rnd' must have 4 columns."));
        CHECK_THROWS_MATCHES(call(fc, arma::mat(5, 3, arma::fill::value(0.5)), models, w, 25.0), std::invalid_argument,
                             Message("Input 'rnd' must have 4 columns."));
        CHECK_THROWS_MATCHES(call(fc, arma::mat(0, 3), models, w, 25.0), std::invalid_argument,
                             Message("Input 'rnd' must have 4 columns."));
        for (arma::uword c = 0; c < 3; ++c)
            for (double v : {-1e-300, 1.0 + 1e-15, double(NAN), inf_o2i})
            {
                arma::mat r = rnd5;
                r(2, c) = v;
                INFO("column " << c << ", value " << v);
                CHECK_THROWS_MATCHES(call(fc, r, models, w, 25.0), std::invalid_argument,
                                     Message("Uniform columns of 'rnd' must be in the range from 0 to 1."));
            }
        for (double v : {double(NAN), inf_o2i, -inf_o2i})
        {
            arma::mat r = rnd5;
            r(4, 3) = v;
            INFO("value " << v);
            CHECK_THROWS_MATCHES(call(fc, r, models, w, 25.0), std::invalid_argument,
                                 Message("Column 3 of 'rnd' must be finite."));
        }
    }
    SECTION("models")
    {
        CHECK_THROWS_MATCHES(call(fc, rnd5, {}, arma::vec(), 25.0), std::invalid_argument,
                             Message("Input 'models' cannot be empty."));
        CHECK_THROWS_MATCHES(call(fc, rnd5, {"low", "high_v14"}, w, 25.0), std::invalid_argument,
                             Message("Model 'high_v14' is not supported."));
        CHECK_THROWS_MATCHES(call(fc, rnd5, {"Low", "high"}, w, 25.0), std::invalid_argument,
                             Message("Model 'Low' is not supported."));
        CHECK_THROWS_MATCHES(call(fc, rnd5, {"low", "mmMAGIC"}, w, 25.0), std::invalid_argument,
                             Message("Model 'mmMAGIC' is not supported."));
        CHECK_THROWS_MATCHES(call(fc, rnd5, {"low", ""}, w, 25.0), std::invalid_argument,
                             Message("Model '' is not supported."));
    }
    SECTION("weights")
    {
        CHECK_THROWS_MATCHES(call(fc, rnd5, models, arma::vec{1.0}, 25.0), std::invalid_argument,
                             Message("Number of elements in 'weights' does not match number of models."));
        CHECK_THROWS_MATCHES(call(fc, rnd5, models, arma::vec{0.2, 0.3, 0.5}, 25.0), std::invalid_argument,
                             Message("Number of elements in 'weights' does not match number of models."));
        CHECK_THROWS_MATCHES(call(fc, rnd5, models, arma::vec{-0.1, 1.1}, 25.0), std::invalid_argument,
                             Message("Input 'weights' must be finite and >= 0."));
        CHECK_THROWS_MATCHES(call(fc, rnd5, models, arma::vec{NAN, 1.0}, 25.0), std::invalid_argument,
                             Message("Input 'weights' must be finite and >= 0."));
        CHECK_THROWS_MATCHES(call(fc, rnd5, models, arma::vec{inf_o2i, 1.0}, 25.0), std::invalid_argument,
                             Message("Input 'weights' must be finite and >= 0."));
        CHECK_THROWS_MATCHES(call(fc, rnd5, models, arma::vec{0.4, 0.5}, 25.0), std::invalid_argument,
                             Message("Input 'weights' must sum to 1."));
        CHECK_THROWS_MATCHES(call(fc, rnd5, models, arma::vec{0.4, 0.6 + 2e-6}, 25.0), std::invalid_argument,
                             Message("Input 'weights' must sum to 1."));
        CHECK_THROWS_MATCHES(call(fc, rnd5, models, arma::vec{0.0, 0.0}, 25.0), std::invalid_argument,
                             Message("Input 'weights' must sum to 1."));
    }
    SECTION("d2d_in_max")
    {
        for (double v : {-1.0, double(NAN), inf_o2i})
        {
            INFO("value " << v);
            CHECK_THROWS_MATCHES(call(fc, rnd5, models, w, v), std::invalid_argument,
                                 Message("Input 'd2d_in_max' must be finite and >= 0."));
        }
    }
}
