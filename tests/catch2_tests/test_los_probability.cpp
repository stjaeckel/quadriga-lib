// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_GBSM=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_GBSM=1 ./build_linux/test_bin "los_probability*"

// Verification suite for quadriga_lib::los_probability.
//
// Pins down: the LOS probability of the TR 38.901 models on both sides of every breakpoint, the 38.811 elevation
// tables, the QuaDRiGa industrial model, the fixed models LOS and NLOS, the limit to [0, 1], the edge rules of the
// InF high-BS and SMa models, the soft LOS state of eq. (7.6-18) including the accuracy of its inverse normal CDF
// on both tails, the optional inputs, pre-sized outputs and the input validation.
// Oracles, independent of the function:
//  - TR 38.901 v19.5.0 Table 7.4.2-1 typed as printed (InF defaults of Table 7.8-7, SMa clutter of the table),
//    limited to [0, 1]; for UMa above hUT = 23 m (not covered by the TR) C' = 1
//  - the edge rules: InF SH/DH with hU >= h_c gives 1, with hB <= hU < h_c the rule of SL/DL; SMa with hB <= hU
//    gives 1, a clutter height below hU gives no term of that clutter type
//  - the 38.811 tables (elevation grid -90, 0, 10, ..., 90 deg, values in percent) with a linear interpolation
//    typed here; QuaDRiGa industrial min(1, exp(-(d + 20)/80)·exp(min(hB, 8)/5 - 1))
//  - eq. (7.6-18) LOS_soft = 1/2 + atan(sqrt(20/lambda)·(G + F(p)))/pi with a test-side inverse normal CDF:
//    bisection for x <= 0 on 0.5·erfc(-x/sqrt(2)) = min(p, 1 - p), F = x for p <= 0.5 and -x otherwise (the
//    symmetric form keeps the upper tail exact)
// Tolerances: probabilities 1e-14 (values <= 1 from short chains of exp and pow); soft LOS
// k·1e-10·max(1, abs(F))/pi + 1e-15 with k = sqrt(20/lambda) (the accuracy target 1e-10·max(1, abs(F)) of the
// inverse normal CDF, scaled by the slope of atan(k·x)/pi, which is at most k/pi); the inverse CDF itself
// 1e-10·max(1, abs(F)); fractions of a quantile grid of K points within 1/K.
// Conventions: distances and heights in meters; frequencies in Hz; lambda = c/fc with c = 299792458 m/s.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_gbsm.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")

namespace
{
    const double pi = 3.141592653589793;
    const double c0 = 299792458.0;
    const double inf = std::numeric_limits<double>::infinity();

    double clamp01(double p) { return p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p); }

    // ---- TR 38.901 Table 7.4.2-1, typed as printed ----

    double tr_rma(double d) { return d <= 10.0 ? 1.0 : std::exp(-(d - 10.0) / 1000.0); }

    double tr_umi(double d) { return d <= 18.0 ? 1.0 : 18.0 / d + std::exp(-d / 36.0) * (1.0 - 18.0 / d); }

    // UMa without the limit to [0, 1]; C'(hUT) = 1 above 23 m (outside the TR range)
    double tr_uma_raw(double d, double hU)
    {
        if (d <= 18.0)
            return 1.0;
        double Cp = hU <= 13.0 ? 0.0 : (hU <= 23.0 ? std::pow((hU - 13.0) / 10.0, 1.5) : 1.0);
        return (18.0 / d + std::exp(-d / 63.0) * (1.0 - 18.0 / d)) *
               (1.0 + Cp * 5.0 / 4.0 * std::pow(d / 100.0, 3.0) * std::exp(-d / 150.0));
    }

    double tr_inh_mixed(double d)
    {
        if (d <= 1.2)
            return 1.0;
        if (d < 6.5)
            return std::exp(-(d - 1.2) / 4.7);
        return std::exp(-(d - 6.5) / 32.6) * 0.32;
    }

    double tr_inh_open(double d)
    {
        if (d <= 5.0)
            return 1.0;
        if (d <= 49.0)
            return std::exp(-(d - 5.0) / 70.8);
        return std::exp(-(d - 49.0) / 211.7) * 0.54;
    }

    // InF-SL, InF-DL: k = -d_clutter / ln(1 - r)
    double tr_inf_low(double d, double d_clutter, double r)
    {
        double k = -d_clutter / std::log(1.0 - r);
        return std::exp(-d / k);
    }

    // InF-SH, InF-DH: k = -d_clutter / ln(1 - r) · (hBS - hUT) / (hc - hUT), written for hBS > hc > hUT
    double tr_inf_high(double d, double d_clutter, double r, double hc, double hB, double hU)
    {
        double k = -d_clutter / std::log(1.0 - r) * (hB - hU) / (hc - hU);
        return std::exp(-d / k);
    }

    // SMa (TR v19), written for hUT below all clutter heights except the residential one (which has max(., 0))
    double tr_sma(double d, double hB, double hU, double r_veg)
    {
        if (d <= 10.0)
            return 1.0;
        const double dcl = 30.0;
        double kc = -std::log(1.0 - 0.02) / dcl * (20.0 - hU) / (hB - hU);
        double kr = -std::log(1.0 - 0.18) / dcl * std::max(8.0 - hU, 0.0) / (hB - hU);
        double kv = -std::log(1.0 - r_veg) / dcl * (15.0 - hU) / (hB - hU);
        return std::exp(-d * kc) * std::exp(-d * kr) * std::exp(-d * kv);
    }

    // ---- TR 38.811 tables (percent) over the elevation grid -90, 0, 10, ..., 90 deg ----

    const double tab_dense_urban[11] = {0, 0, 28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1};
    const double tab_urban[11] = {0, 0, 24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2};
    const double tab_suburban[11] = {0, 0, 78.2, 86.9, 91.9, 92.9, 93.5, 94.0, 94.9, 95.2, 99.8};

    double ref_811(const double *tab, double hB, double d)
    {
        const double grid[11] = {-90, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
        double el = std::atan2(hB, d) * 180.0 / pi;
        if (el <= grid[0])
            return 0.01 * tab[0];
        if (el >= grid[10])
            return 0.01 * tab[10];
        int i = 0;
        while (i < 9 && el >= grid[i + 1])
            ++i;
        double w = (el - grid[i]) / (grid[i + 1] - grid[i]);
        return 0.01 * (tab[i] + w * (tab[i + 1] - tab[i]));
    }

    double ref_industrial(double d, double hB)
    {
        return std::min(1.0, std::exp(-(d + 20.0) / 80.0) * std::exp(std::min(hB, 8.0) / 5.0 - 1.0));
    }

    // ---- Inverse normal CDF by bisection (symmetric form) ----

    double Phi(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

    double ref_probit(double p)
    {
        if (p <= 0.0)
            return -inf;
        if (p >= 1.0)
            return inf;
        const double t = p <= 0.5 ? p : 1.0 - p; // 1 - p is exact for p >= 0.5
        double lo = -39.0, hi = 0.0;
        for (int it = 0; it < 400 && hi - lo >= 1e-15 * std::max(1.0, std::abs(lo)); ++it)
        {
            double mid = 0.5 * (lo + hi);
            if (Phi(mid) < t)
                lo = mid;
            else
                hi = mid;
        }
        double x = 0.5 * (lo + hi);
        return p <= 0.5 ? x : -x;
    }

    double ref_soft(double p, double g, double fc)
    {
        double k = std::sqrt(20.0 / (c0 / fc));
        return 0.5 + std::atan(k * (g + ref_probit(p))) / pi;
    }

    // Tolerance of a soft LOS value: probit accuracy scaled by the slope of atan(k·x)/pi
    double tol_soft(double F, double fc)
    {
        double k = std::sqrt(20.0 / (c0 / fc));
        double aF = std::isfinite(F) ? std::abs(F) : 0.0;
        return k * 1e-10 * std::max(1.0, aF) / pi + 1e-15;
    }

    // Single-link call
    double p_one(const std::string &model, double d, double hB, double hU, double dcl = NAN, double r = NAN,
                 double hc = NAN)
    {
        arma::vec p;
        quadriga_lib::los_probability(model, arma::vec{d}, arma::vec{hB}, arma::vec{hU}, p, dcl, r, hc);
        return p.n_elem == 1 ? p(0) : NAN;
    }

    std::string err_msg(const std::function<void()> &f)
    {
        try
        {
            f();
        }
        catch (const std::invalid_argument &e)
        {
            return e.what();
        }
        catch (...)
        {
            return "<other exception>";
        }
        return "<no exception>";
    }

    const std::vector<std::string> all_models = {
        "38.901_RMa", "38.901_UMi", "38.901_UMa", "38.901_InH_Mixed", "38.901_InH_Open", "38.901_InF_SL",
        "38.901_InF_DL", "38.901_InF_SH", "38.901_InF_DH", "38.901_InF_HH", "38.901_SMa", "38.811_DenseUrban",
        "38.811_Urban", "38.811_Suburban", "38.811_Rural", "QuaDRiGa_Industrial", "LOS", "NLOS"};
}

TEST_CASE("los_probability - TR 38.901 models match Table 7.4.2-1")
{
    const double tol = 1e-14;
    QD_DIAG("\n# diag: los_probability vs TR 38.901 Table 7.4.2-1\n");
    QD_DIAG("model,d,hB,hU,p,ref,diff\n");

    auto chk = [&](const std::string &model, double d, double hB, double hU, double ref, double dcl = NAN,
                   double r = NAN, double hc = NAN)
    {
        double p = p_one(model, d, hB, hU, dcl, r, hc);
        ref = clamp01(ref);
        QD_DIAG("%s,%g,%g,%g,%.15f,%.15f,%.3e\n", model.c_str(), d, hB, hU, p, ref, p - ref);
        INFO("model " << model << ", d " << d << ", hB " << hB << ", hU " << hU);
        CHECK(std::abs(p - ref) < tol);
    };

    for (double d : {5.0, 10.0, 10.001, 500.0, 5000.0})
        chk("38.901_RMa", d, 35.0, 1.5, tr_rma(d));
    for (double d : {0.0, 18.0, 18.5, 100.0, 1000.0})
        chk("38.901_UMi", d, 10.0, 1.5, tr_umi(d));
    for (double hU : {1.5, 15.0, 22.5, 25.0})
        for (double d : {10.0, 18.0, 30.0, 100.0, 400.0, 1000.0})
            chk("38.901_UMa", d, 25.0, hU, tr_uma_raw(d, hU));
    for (double d : {1.0, 1.2, 3.0, 6.5, 6.6, 50.0})
        chk("38.901_InH_Mixed", d, 3.0, 1.0, tr_inh_mixed(d));
    for (double d : {3.0, 5.0, 20.0, 49.0, 50.0, 200.0})
        chk("38.901_InH_Open", d, 3.0, 1.0, tr_inh_open(d));

    // InF with the defaults of Table 7.8-7 and with explicit arguments; hB 8, hU 1.5
    const double hB = 8.0, hU = 1.5;
    for (double d : {1.0, 5.0, 20.0, 50.0})
    {
        chk("38.901_InF_SL", d, hB, hU, tr_inf_low(d, 10.0, 0.2));
        chk("38.901_InF_DL", d, hB, hU, tr_inf_low(d, 2.0, 0.6));
        chk("38.901_InF_SH", d, hB, hU, tr_inf_high(d, 10.0, 0.2, 2.0, hB, hU));
        chk("38.901_InF_DH", d, hB, hU, tr_inf_high(d, 2.0, 0.6, 6.0, hB, hU));
        chk("38.901_InF_SL", d, hB, hU, tr_inf_low(d, 7.0, 0.35), 7.0, 0.35, 4.0);
        chk("38.901_InF_DL", d, hB, hU, tr_inf_low(d, 3.0, 0.5), 3.0, 0.5, 4.0);
        chk("38.901_InF_SH", d, hB, hU, tr_inf_high(d, 7.0, 0.35, 4.0, hB, hU), 7.0, 0.35, 4.0);
        chk("38.901_InF_DH", d, hB, hU, tr_inf_high(d, 3.0, 0.5, 5.0, hB, hU), 3.0, 0.5, 5.0);
        chk("38.901_InF_HH", d, hB, hU, 1.0);
    }

    // SMa with hB 35 m (Table 7.2-5)
    for (double r_veg : {0.0, 0.1, 0.2})
        for (double hUs : {1.5, 4.5, 13.5})
            for (double d : {5.0, 10.0, 11.0, 100.0, 1000.0})
            {
                double r = r_veg == 0.0 ? NAN : r_veg; // NaN = default 0
                chk("38.901_SMa", d, 35.0, hUs, tr_sma(d, 35.0, hUs, r_veg), NAN, r);
            }
}

TEST_CASE("los_probability - 38.811, QuaDRiGa and fixed models")
{
    const double tol = 1e-14;
    QD_DIAG("\n# diag: los_probability 38.811 tables and QuaDRiGa industrial\n");
    QD_DIAG("model,d,hB,p,ref,diff\n");

    struct M
    {
        const char *name;
        const double *tab;
    };
    const M m811[4] = {{"38.811_DenseUrban", tab_dense_urban}, {"38.811_Urban", tab_urban}, {"38.811_Suburban", tab_suburban}, {"38.811_Rural", tab_suburban}};

    // Elevations 5, 10, 45, 87, 90 deg and below the horizon
    std::vector<std::pair<double, double>> geo; // (d, hB)
    for (double el : {5.0, 10.0, 87.0})
        geo.push_back({100.0, 100.0 * std::tan(el * pi / 180.0)});
    geo.push_back({50.0, 50.0}); // 45 deg
    geo.push_back({0.0, 10.0});  // 90 deg
    geo.push_back({50.0, -10.0}); // below the horizon
    geo.push_back({50.0, 0.0});   // at the horizon
    geo.push_back({0.0, -10.0});  // -90 deg

    for (const auto &m : m811)
        for (const auto &g : geo)
        {
            double p = p_one(m.name, g.first, g.second, 1.5, 3.0, 0.5, 1.0); // clutter arguments are ignored
            double ref = ref_811(m.tab, g.second, g.first);
            QD_DIAG("%s,%g,%g,%.15f,%.15f,%.3e\n", m.name, g.first, g.second, p, ref, p - ref);
            INFO("model " << m.name << ", d " << g.first << ", hB " << g.second);
            CHECK(std::abs(p - ref) < tol);
        }

    // The tables at the grid points: 90 deg gives the last entry, below the horizon 0
    CHECK(std::abs(p_one("38.811_Urban", 0.0, 10.0, 1.5) - 0.992) < tol);
    CHECK(p_one("38.811_Urban", 50.0, -10.0, 1.5) == 0.0);

    for (double hB : {4.0, 8.0, 12.0})
        for (double d : {0.0, 10.0, 50.0, 200.0, 1000.0})
        {
            double p = p_one("QuaDRiGa_Industrial", d, hB, 1.5);
            double ref = ref_industrial(d, hB);
            QD_DIAG("QuaDRiGa_Industrial,%g,%g,%.15f,%.15f,%.3e\n", d, hB, p, ref, p - ref);
            INFO("Industrial, d " << d << ", hB " << hB);
            CHECK(std::abs(p - ref) < tol);
        }
    CHECK(p_one("QuaDRiGa_Industrial", 0.0, 8.0, 1.5) == 1.0); // exp(0.35) limited to 1

    for (double d : {0.0, 10.0, 1000.0})
    {
        CHECK(p_one("LOS", d, 25.0, 1.5) == 1.0);
        CHECK(p_one("NLOS", d, 25.0, 1.5) == 0.0);
    }
}

TEST_CASE("los_probability - probabilities stay in [0, 1]")
{
    // UMa for hUT > 13 m exceeds 1 just above 18 m
    double r1 = tr_uma_raw(18.1, 20.0), r2 = tr_uma_raw(18.1, 22.5), r3 = tr_uma_raw(20.0, 20.0);
    QD_DIAG("\n# diag: UMa TR formula above 1\nd,hU,tr,p\n");
    QD_DIAG("18.1,20,%.6f,%.15f\n", r1, p_one("38.901_UMa", 18.1, 25.0, 20.0));
    QD_DIAG("18.1,22.5,%.6f,%.15f\n", r2, p_one("38.901_UMa", 18.1, 25.0, 22.5));
    QD_DIAG("20,20,%.6f,%.15f\n", r3, p_one("38.901_UMa", 20.0, 25.0, 20.0));
    REQUIRE(std::abs(r1 - 1.00246) < 5e-6);
    REQUIRE(std::abs(r2 - 1.00470) < 5e-6);
    REQUIRE(std::abs(r3 - 0.97779) < 5e-6);
    CHECK(p_one("38.901_UMa", 18.1, 25.0, 20.0) == 1.0);
    CHECK(p_one("38.901_UMa", 18.1, 25.0, 22.5) == 1.0);
    CHECK(std::abs(p_one("38.901_UMa", 20.0, 25.0, 20.0) - r3) < 1e-14);

    // Every model over a grid of distances and heights
    const arma::vec dg = {0.0, 0.5, 1.0, 5.0, 10.0, 18.0, 18.1, 18.3, 25.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 1e5};
    const std::vector<double> hBg = {-5.0, 0.0, 1.5, 4.0, 8.0, 10.0, 25.0, 35.0};
    const std::vector<double> hUg = {0.0, 1.5, 2.0, 6.0, 13.5, 17.0, 20.0, 22.5, 30.0};
    arma::uword n_bad = 0, n_tot = 0;
    for (const auto &m : all_models)
        for (double hB : hBg)
            for (double hU : hUg)
            {
                arma::vec p;
                quadriga_lib::los_probability(m, dg, arma::vec{hB}, arma::vec{hU}, p);
                REQUIRE(p.n_elem == dg.n_elem);
                for (arma::uword i = 0; i < p.n_elem; ++i)
                {
                    ++n_tot;
                    if (!(p(i) >= 0.0 && p(i) <= 1.0))
                    {
                        ++n_bad;
                        INFO("model " << m << ", d " << dg(i) << ", hB " << hB << ", hU " << hU << ", p " << p(i));
                        CHECK(false);
                    }
                }
            }
    QD_DIAG("\n# diag: grid of all models\nn_values,n_outside\n%llu,%llu\n", (unsigned long long)n_tot,
            (unsigned long long)n_bad);
    CHECK(n_bad == 0);
}

TEST_CASE("los_probability - InF edge rules")
{
    const double tol = 1e-14;
    const double d = 20.0;

    // UT at or above the clutter height: 1
    CHECK(p_one("38.901_InF_SH", d, 8.0, 2.0) == 1.0); // h_c = 2 m
    CHECK(p_one("38.901_InF_SH", d, 8.0, 3.0) == 1.0);
    CHECK(p_one("38.901_InF_DH", d, 8.0, 6.0) == 1.0); // h_c = 6 m
    CHECK(p_one("38.901_InF_DH", d, 8.0, 7.0, 2.0, 0.6, 6.5) == 1.0); // explicit h_c = 6.5 m below hU = 7 m
    CHECK(p_one("38.901_InF_SH", d, 8.0, 3.0, NAN, NAN, 3.0) == 1.0); // hU = h_c

    // BS below the UT, both inside the clutter, and BS at the UT height: the SL/DL rule
    for (double hB : {1.0, 2.5})
    {
        INFO("hB " << hB);
        CHECK(std::abs(p_one("38.901_InF_DH", d, hB, 2.5) - tr_inf_low(d, 2.0, 0.6)) < tol);
        CHECK(std::abs(p_one("38.901_InF_SH", d, hB, 2.5, 5.0, 0.3, 3.0) - tr_inf_low(d, 5.0, 0.3)) < tol);
    }
    CHECK(std::abs(p_one("38.901_InF_SH", d, 1.5, 1.5) - tr_inf_low(d, 10.0, 0.2)) < tol);

    // r = 0: no clutter, always LOS
    for (const char *m : {"38.901_InF_SL", "38.901_InF_DL", "38.901_InF_SH", "38.901_InF_DH"})
    {
        INFO("model " << m);
        CHECK(p_one(m, 50.0, 8.0, 1.5, NAN, 0.0) == 1.0);
        CHECK(p_one(m, 50.0, 1.0, 1.5, NAN, 0.0) == 1.0);
    }
}

TEST_CASE("los_probability - SMa edge rules")
{
    const double tol = 1e-14;

    // BS at or below the UT: 1
    CHECK(p_one("38.901_SMa", 100.0, 10.0, 10.0) == 1.0);
    CHECK(p_one("38.901_SMa", 100.0, 5.0, 10.0, NAN, 0.2) == 1.0);

    // UT at 18 m, above the vegetation (15 m) and the residential height (8 m): only the commercial term remains
    const double hB = 35.0, hU = 18.0;
    for (double r_veg : {0.0, 0.2})
        for (double d : {11.0, 100.0, 1000.0})
        {
            double kc = -std::log(1.0 - 0.02) / 30.0 * (20.0 - hU) / (hB - hU);
            double ref = std::exp(-d * kc);
            double lit = tr_sma(d, hB, hU, r_veg); // TR printed literally: a negative vegetation term
            double p = p_one("38.901_SMa", d, hB, hU, NAN, r_veg);
            INFO("r_veg " << r_veg << ", d " << d);
            CHECK(std::abs(p - ref) < tol);
            if (r_veg > 0.0)
                CHECK(lit > ref * (1.0 + 1e-6));
        }

    // Clutter size as argument (60 m), r_veg 0.1, hB 35 m, hU 1.5 m
    const double s = (-std::log(0.98) * 18.5 - std::log(0.82) * 6.5 - std::log(0.9) * 13.5) / 60.0 / 33.5;
    CHECK(std::abs(p_one("38.901_SMa", 100.0, 35.0, 1.5, 60.0, 0.1) - std::exp(-100.0 * s)) < tol);
}

TEST_CASE("los_probability - soft LOS is 0.5 at G = -F(d) and matches p over many G")
{
    const arma::vec fc = {3.5e9, 28.0e9};

    // G = -F(p) gives 0.5
    {
        const arma::vec d = {50.0, 500.0, 3000.0, 20000.0, 12.0};
        arma::vec p;
        quadriga_lib::los_probability("38.901_RMa", d, arma::vec{35.0}, arma::vec{1.5}, p);
        REQUIRE(p.n_elem == d.n_elem);
        arma::vec g(d.n_elem);
        for (arma::uword i = 0; i < d.n_elem; ++i)
            g(i) = -ref_probit(p(i));
        arma::mat soft;
        quadriga_lib::los_probability("38.901_RMa", d, arma::vec{35.0}, arma::vec{1.5}, p, NAN, NAN, NAN, &g, &fc, &soft);
        REQUIRE(soft.n_rows == d.n_elem);
        REQUIRE(soft.n_cols == 2);
        QD_DIAG("\n# diag: soft LOS at G = -F(p)\nd,p,F,fc,soft,diff,tol\n");
        for (arma::uword i = 0; i < d.n_elem; ++i)
            for (arma::uword k = 0; k < 2; ++k)
            {
                double tol = tol_soft(-g(i), fc(k));
                QD_DIAG("%g,%.6e,%.6f,%g,%.17f,%.3e,%.3e\n", d(i), p(i), -g(i), fc(k), soft(i, k), soft(i, k) - 0.5, tol);
                INFO("d " << d(i) << ", fc " << fc(k));
                CHECK(std::abs(soft(i, k) - 0.5) < tol);
            }
    }

    // Quantile grid of G: the fraction of LOS_soft > 0.5 equals p within 1/K
    {
        const arma::uword K = 10000;
        arma::vec G(K);
        for (arma::uword k = 0; k < K; ++k)
            G(k) = ref_probit((double(k) + 0.5) / double(K));
        QD_DIAG("\n# diag: fraction of soft LOS > 0.5 over a quantile grid of G\nd,p,fraction,diff\n");
        for (double dd : {20.0, 200.0, 693.147, 1000.0, 2500.0})
        {
            arma::vec d(K, arma::fill::value(dd)), p;
            arma::mat soft;
            quadriga_lib::los_probability("38.901_RMa", d, arma::vec{35.0}, arma::vec{1.5}, p, NAN, NAN, NAN, &G,
                                          &fc, &soft);
            REQUIRE(soft.n_rows == K);
            const double pr = tr_rma(dd);
            for (arma::uword k = 0; k < 2; ++k)
            {
                double frac = double(arma::accu(soft.col(k) > 0.5)) / double(K);
                QD_DIAG("%g,%.6f,%.6f,%.2e\n", dd, pr, frac, frac - pr);
                INFO("d " << dd << ", fc " << fc(k));
                CHECK(std::abs(frac - pr) <= 1.0 / double(K) + 1e-12);
            }
        }
    }

    // p = 1 gives 1, p = 0 gives 0, for any finite G
    {
        const arma::vec g = {-3.0, 0.0, 3.0};
        const arma::vec d = {5.0, 100.0, 1000.0};
        arma::vec p;
        arma::mat soft;
        quadriga_lib::los_probability("LOS", d, arma::vec{25.0}, arma::vec{1.5}, p, NAN, NAN, NAN, &g, &fc, &soft);
        CHECK(arma::all(arma::vectorise(soft) == 1.0));
        quadriga_lib::los_probability("NLOS", d, arma::vec{25.0}, arma::vec{1.5}, p, NAN, NAN, NAN, &g, &fc, &soft);
        CHECK(arma::all(arma::vectorise(soft) == 0.0));
        quadriga_lib::los_probability("38.901_RMa", arma::vec{5.0, 10.0, 1.0}, arma::vec{25.0}, arma::vec{1.5}, p, NAN,
                                      NAN, NAN, &g, &fc, &soft);
        CHECK(arma::all(arma::vectorise(soft) == 1.0));
    }

    // Eq. (7.6-18) typed, at 3.5 and 28 GHz
    {
        const arma::vec d = {30.0, 100.0, 300.0, 18.0}, g = {0.7, -1.2, 2.1, -0.4};
        arma::vec p;
        arma::mat soft;
        quadriga_lib::los_probability("38.901_UMa", d, arma::vec{25.0}, arma::vec{1.5}, p, NAN, NAN, NAN, &g, &fc, &soft);
        REQUIRE(soft.n_rows == 4);
        QD_DIAG("\n# diag: soft LOS vs eq. (7.6-18)\nd,G,fc,soft,ref,diff\n");
        for (arma::uword i = 0; i < 4; ++i)
            for (arma::uword k = 0; k < 2; ++k)
            {
                double pr = tr_uma_raw(d(i), 1.5);
                double ref = ref_soft(pr, g(i), fc(k));
                QD_DIAG("%g,%g,%g,%.15f,%.15f,%.3e\n", d(i), g(i), fc(k), soft(i, k), ref, soft(i, k) - ref);
                INFO("d " << d(i) << ", fc " << fc(k));
                CHECK(std::abs(soft(i, k) - ref) < tol_soft(ref_probit(pr), fc(k)));
            }
    }
}

TEST_CASE("los_probability - the inverse normal CDF of the soft LOS is accurate")
{
    // RMa: p = exp(-(d - 10)/1000). fc = 1 kHz makes k = sqrt(20/lambda) about 8e-3, so that
    // F_rec = tan(pi·(LOS_soft - 1/2)) / k recovers F(p) with an error of about 2e-14 from the rounding of LOS_soft
    const arma::vec fc = {1.0e3};
    const double k = std::sqrt(20.0 / (c0 / fc(0)));

    std::vector<double> dv;
    for (int j = 0; j <= 60; ++j) // lower tail: p from 1e-12 to 0.5, logarithmic
    {
        double lp = -12.0 + (12.0 + std::log10(0.5)) * double(j) / 60.0;
        dv.push_back(10.0 - 1000.0 * std::log(std::pow(10.0, lp)));
    }
    for (int j = 2; j <= 24; ++j) // upper tail: 1 - p from 0.1 down to 1e-12
    {
        double q = std::pow(10.0, -0.5 * double(j));
        dv.push_back(10.0 - 1000.0 * std::log1p(-q));
    }
    const arma::vec d(dv);
    const arma::vec g(d.n_elem, arma::fill::zeros);
    arma::vec p;
    arma::mat soft;
    quadriga_lib::los_probability("38.901_RMa", d, arma::vec{35.0}, arma::vec{1.5}, p, NAN, NAN, NAN, &g, &fc, &soft);
    REQUIRE(soft.n_rows == d.n_elem);

    QD_DIAG("\n# diag: inverse normal CDF through the soft LOS\np,1-p,F_rec,F_ref,rel_err\n");
    double max_rel = 0.0;
    for (arma::uword i = 0; i < d.n_elem; ++i)
    {
        double F_ref = ref_probit(p(i));
        double F_rec = std::tan(pi * (soft(i, 0) - 0.5)) / k;
        double rel = std::abs(F_rec - F_ref) / std::max(1.0, std::abs(F_ref));
        max_rel = std::max(max_rel, rel);
        QD_DIAG("%.6e,%.6e,%.15f,%.15f,%.3e\n", p(i), 1.0 - p(i), F_rec, F_ref, rel);
        INFO("p " << p(i) << ", 1 - p " << 1.0 - p(i));
        CHECK(rel < 1e-10);
    }
    QD_DIAG("max_rel,%.3e\n", max_rel);
    // The grid covers both tails down to 1e-12
    CHECK(tr_rma(d.max()) < 1.1e-12);
    CHECK(1.0 - tr_rma(d.min()) < 1.1e-12);
}

TEST_CASE("los_probability - optional inputs")
{
    const arma::vec d = {20.0, 100.0, 400.0}, g = {0.3, -0.5, 1.1}, fc = {3.5e9};
    arma::vec p, p_ref;
    arma::mat soft;

    // los_soft requires rnd_g and fc
    CHECK(err_msg([&]
                  { quadriga_lib::los_probability("38.901_UMi", d, arma::vec{10.0}, arma::vec{1.5}, p, NAN, NAN, NAN, nullptr, &fc, &soft); }) ==
          "Output 'los_soft' requires 'rnd_g' and 'fc'.");
    CHECK(err_msg([&]
                  { quadriga_lib::los_probability("38.901_UMi", d, arma::vec{10.0}, arma::vec{1.5}, p, NAN, NAN, NAN, &g, nullptr, &soft); }) ==
          "Output 'los_soft' requires 'rnd_g' and 'fc'.");

    // rnd_g and fc without los_soft are ignored (not validated)
    quadriga_lib::los_probability("38.901_UMi", d, arma::vec{10.0}, arma::vec{1.5}, p_ref);
    const arma::vec g_bad = {NAN}, fc_bad;
    CHECK_NOTHROW(quadriga_lib::los_probability("38.901_UMi", d, arma::vec{10.0}, arma::vec{1.5}, p, NAN, NAN, NAN, &g_bad, &fc_bad));
    CHECK(arma::all(p == p_ref));
    CHECK_NOTHROW(quadriga_lib::los_probability("38.901_UMi", d, arma::vec{10.0}, arma::vec{1.5}, p, NAN, NAN, NAN, &g, &fc));
    CHECK(arma::all(p == p_ref));

    // A single height applies to all links
    for (const auto &m : all_models)
    {
        arma::vec p1, p2;
        arma::mat s1, s2;
        quadriga_lib::los_probability(m, d, arma::vec{8.0}, arma::vec{1.5}, p1, NAN, NAN, NAN, &g, &fc, &s1);
        quadriga_lib::los_probability(m, d, arma::vec{8.0, 8.0, 8.0}, arma::vec{1.5, 1.5, 1.5}, p2, NAN, NAN, NAN, &g, &fc, &s2);
        INFO("model " << m);
        CHECK(arma::all(p1 == p2));
        CHECK(arma::all(arma::vectorise(s1 == s2)));
    }
    arma::vec p1, p2;
    quadriga_lib::los_probability("38.901_UMa", d, arma::vec{25.0}, arma::vec{1.5, 17.0, 22.5}, p1);
    quadriga_lib::los_probability("38.901_UMa", d, arma::vec{25.0, 25.0, 25.0}, arma::vec{1.5, 17.0, 22.5}, p2);
    CHECK(arma::all(p1 == p2));
    quadriga_lib::los_probability("38.901_InF_DH", d, arma::vec{8.0, 4.0, 1.0}, arma::vec{1.5}, p1);
    quadriga_lib::los_probability("38.901_InF_DH", d, arma::vec{8.0, 4.0, 1.0}, arma::vec{1.5, 1.5, 1.5}, p2);
    CHECK(arma::all(p1 == p2));
}

TEST_CASE("los_probability - pre-sized outputs, empty input and repeated calls")
{
    const arma::vec d = {20.0, 100.0, 400.0}, hB = {25.0, 30.0, 35.0}, hU = {1.5, 15.0, 22.5}, g = {0.3, -0.5, 1.1};
    const arma::vec fc = {3.5e9, 28.0e9};
    arma::vec p_ref;
    arma::mat s_ref;
    quadriga_lib::los_probability("38.901_UMa", d, hB, hU, p_ref, NAN, NAN, NAN, &g, &fc, &s_ref);
    REQUIRE(p_ref.n_elem == 3);
    REQUIRE(s_ref.n_rows == 3);
    REQUIRE(s_ref.n_cols == 2);

    // Correctly pre-sized outputs with stale content are overwritten in place
    arma::vec p(3, arma::fill::value(7.0));
    arma::mat s(3, 2, arma::fill::value(7.0));
    const double *pp = p.memptr(), *ps = s.memptr();
    quadriga_lib::los_probability("38.901_UMa", d, hB, hU, p, NAN, NAN, NAN, &g, &fc, &s);
    CHECK(p.memptr() == pp);
    CHECK(s.memptr() == ps);
    CHECK(arma::all(p == p_ref));
    CHECK(arma::all(arma::vectorise(s == s_ref)));

    // Wrongly sized outputs are resized
    arma::vec p2(5, arma::fill::value(7.0));
    arma::mat s2(1, 7, arma::fill::value(7.0));
    quadriga_lib::los_probability("38.901_UMa", d, hB, hU, p2, NAN, NAN, NAN, &g, &fc, &s2);
    CHECK(arma::all(p2 == p_ref));
    REQUIRE(s2.n_rows == 3);
    REQUIRE(s2.n_cols == 2);
    CHECK(arma::all(arma::vectorise(s2 == s_ref)));

    // Repeated call: identical
    arma::vec p3;
    arma::mat s3;
    quadriga_lib::los_probability("38.901_UMa", d, hB, hU, p3, NAN, NAN, NAN, &g, &fc, &s3);
    CHECK(arma::all(p3 == p_ref));
    CHECK(arma::all(arma::vectorise(s3 == s_ref)));

    // No links: empty outputs; heights with 0 or 1 elements
    const arma::vec e;
    arma::vec p0(4, arma::fill::ones);
    arma::mat s0(2, 2, arma::fill::ones);
    CHECK_NOTHROW(quadriga_lib::los_probability("38.901_UMa", e, e, e, p0, NAN, NAN, NAN, &e, &fc, &s0));
    CHECK(p0.n_elem == 0);
    CHECK(s0.n_rows == 0);
    CHECK(s0.n_cols == 2);
    CHECK_NOTHROW(quadriga_lib::los_probability("38.901_UMa", e, arma::vec{25.0}, arma::vec{1.5}, p0));
    CHECK(p0.n_elem == 0);
}

TEST_CASE("los_probability - input validation")
{
    const arma::vec d = {20.0, 100.0, 400.0}, hB = {25.0}, hU = {1.5, 15.0, 22.5}, g = {0.3, -0.5, 1.1};
    const arma::vec fc = {3.5e9};
    arma::vec p;
    arma::mat soft;

    auto call = [&](const std::string &m, const arma::vec &dd, const arma::vec &hb, const arma::vec &hu, double dcl,
                    double r, double hc, const arma::vec *gg, const arma::vec *ff, arma::mat *ss)
    { return err_msg([&]
                     { quadriga_lib::los_probability(m, dd, hb, hu, p, dcl, r, hc, gg, ff, ss); }); };
    auto call_m = [&](const std::string &m)
    { return call(m, d, hB, hU, NAN, NAN, NAN, nullptr, nullptr, nullptr); };
    auto call_d = [&](const arma::vec &dd)
    { return call("38.901_UMa", dd, hB, arma::vec{1.5}, NAN, NAN, NAN, nullptr, nullptr, nullptr); };
    auto call_h = [&](const arma::vec &hb, const arma::vec &hu)
    { return call("38.901_UMa", d, hb, hu, NAN, NAN, NAN, nullptr, nullptr, nullptr); };
    auto call_c = [&](const std::string &m, double dcl, double r, double hc)
    { return call(m, d, hB, hU, dcl, r, hc, nullptr, nullptr, nullptr); };
    auto call_s = [&](const arma::vec *gg, const arma::vec *ff)
    { return call("38.901_UMa", d, hB, hU, NAN, NAN, NAN, gg, ff, &soft); };

    SECTION("valid baseline")
    {
        for (const auto &m : all_models)
        {
            INFO("model " << m);
            CHECK(call_m(m) == "<no exception>");
            CHECK(call(m, d, hB, hU, 5.0, 0.0, 0.0, &g, &fc, &soft) == "<no exception>");
            CHECK(call(m, d, hB, hU, 5.0, 0.99, 3.0, &g, &fc, &soft) == "<no exception>");
        }
        CHECK(call_d(arma::vec{0.0, 1e6}) == "<no exception>");
        CHECK(call_h(arma::vec{-10.0, 0.0, 25.0}, arma::vec{0.0}) == "<no exception>");
        CHECK_NOTHROW(quadriga_lib::los_probability("38.901_UMa", d, hB, hU, p));
    }
    SECTION("model")
    {
        CHECK(call_m("foo") == "Model 'foo' is not supported.");
        CHECK(call_m("38.901_uma") == "Model '38.901_uma' is not supported.");
        CHECK(call_m("") == "Model '' is not supported.");
    }
    SECTION("d2d_out")
    {
        for (double v : {-1.0, -1e-300, double(NAN), inf, -inf})
        {
            INFO("value " << v);
            CHECK(call_d(arma::vec{10.0, v}) == "Input 'd2d_out' must be finite and >= 0.");
        }
    }
    SECTION("h_bs")
    {
        CHECK(call_h(arma::vec{25.0, 25.0}, hU) == "Number of elements in 'h_bs' must be 1 or match 'd2d_out'.");
        CHECK(call_h(arma::vec(), hU) == "Number of elements in 'h_bs' must be 1 or match 'd2d_out'.");
        CHECK(call_h(arma::vec{NAN}, hU) == "Input 'h_bs' must be finite.");
        CHECK(call_h(arma::vec{25.0, inf, 25.0}, hU) == "Input 'h_bs' must be finite.");
    }
    SECTION("h_ut")
    {
        CHECK(call_h(hB, arma::vec{1.5, 1.5}) == "Number of elements in 'h_ut' must be 1 or match 'd2d_out'.");
        CHECK(call_h(hB, arma::vec()) == "Number of elements in 'h_ut' must be 1 or match 'd2d_out'.");
        CHECK(call_h(hB, arma::vec{1.5, -inf, 1.5}) == "Input 'h_ut' must be finite.");
    }
    SECTION("d_clutter")
    {
        for (const char *m : {"38.901_InF_SL", "38.901_SMa", "LOS"}) // checked for every model
            for (double v : {0.0, -1.0, inf})
            {
                INFO("model " << m << ", value " << v);
                CHECK(call_c(m, v, NAN, NAN) == "Input 'd_clutter' must be NaN or > 0.");
            }
    }
    SECTION("r_clutter")
    {
        for (const char *m : {"38.901_InF_DL", "38.901_SMa", "38.811_Urban"})
            for (double v : {-0.1, 1.0, 1.5, inf, -inf})
            {
                INFO("model " << m << ", value " << v);
                CHECK(call_c(m, NAN, v, NAN) == "Input 'r_clutter' must be NaN or in the range from 0 to 1, 1 excluded.");
            }
    }
    SECTION("h_clutter")
    {
        for (const char *m : {"38.901_InF_SH", "NLOS"})
            for (double v : {-1.0, inf})
            {
                INFO("model " << m << ", value " << v);
                CHECK(call_c(m, NAN, NAN, v) == "Input 'h_clutter' must be NaN or >= 0.");
            }
    }
    SECTION("los_soft")
    {
        CHECK(call_s(nullptr, &fc) == "Output 'los_soft' requires 'rnd_g' and 'fc'.");
        CHECK(call_s(&g, nullptr) == "Output 'los_soft' requires 'rnd_g' and 'fc'.");
        const arma::vec g2 = {0.3, 0.1}, g_nan = {0.3, NAN, 0.1}, g_inf = {0.3, inf, 0.1};
        CHECK(call_s(&g2, &fc) == "Number of elements in 'rnd_g' does not match number of elements in 'd2d_out'.");
        CHECK(call_s(&g_nan, &fc) == "Input 'rnd_g' must be finite.");
        CHECK(call_s(&g_inf, &fc) == "Input 'rnd_g' must be finite.");
        const arma::vec f0, f1 = {3.5e9, 0.0}, f2 = {-1.0}, f3 = {NAN}, f4 = {inf};
        CHECK(call_s(&g, &f0) == "Input 'fc' cannot be empty.");
        for (const arma::vec *f : {&f1, &f2, &f3, &f4})
            CHECK(call_s(&g, f) == "Input 'fc' must be finite and > 0.");
    }
}
