// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for quadriga_lib::scenpar_validate.
//
// Pins down: completion of a parameter set with defaults (empty map, 138-key legacy map), rejection
// of unknown names, the value checks per validation class for a set that did not come from a file,
// the 'validate = false' mode, the LSP cross-correlation matrix and the positive-definite flag.
// Left to other suites: the file grammar and the same value checks through a file
// (test_conf_file_read).
//
// Oracles: key order and defaults are typed from QuaDRiGa's set_scenario_table.m, the validation
// classes from check_scenario_parameter_table.m with the changes Q11 to Q14 of the plan, the matrix
// positions from get.lsp_xcorr in qd_builder.m. The positive-definite oracle for 3 coupled LSPs is the
// determinant of the 3 x 3 correlation matrix.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Diagnostic output. Comment out to silence the tables. The assertions run either way.
#define QD_TEST_VERBOSE

#ifdef QD_TEST_VERBOSE
#include <cstdio>
#define QD_DIAG(...) std::printf(__VA_ARGS__)
#else
#define QD_DIAG(...) ((void)0)
#endif

namespace
{
    const double INF = std::numeric_limits<double>::infinity();
    const double QNAN = std::numeric_limits<double>::quiet_NaN();

    // Oracle: canonical key order. The 138 numeric legacy keys are typed from the struct(...) call in
    // QuaDRiGa's set_scenario_table.m (without 'SubpathMethod' and 'GR_epsilon'), the 27 new keys from section 7.2 of the plan.
    const std::vector<std::string> &ref_key_order()
    {
        static const std::vector<std::string> k = {
        "NumClusters", "NumSubPaths", "DS_mu", "DS_sigma", "DS_lambda", "DS_omega", "DS_gamma", "DS_epsilon",
        "DS_zeta", "DS_alpha", "DS_delta", "DS_kappa", "DS_tau", "DS_beta", "AS_D_mu", "AS_D_sigma",
        "AS_D_lambda", "AS_D_omega", "AS_D_gamma", "AS_D_epsilon", "AS_D_zeta", "AS_D_alpha", "AS_D_delta",
        "AS_D_kappa", "AS_D_tau", "AS_D_beta", "AS_A_mu", "AS_A_sigma", "AS_A_lambda", "AS_A_omega",
        "AS_A_gamma", "AS_A_epsilon", "AS_A_zeta", "AS_A_alpha", "AS_A_delta", "AS_A_kappa", "AS_A_tau",
        "AS_A_beta", "ES_D_mu", "ES_D_sigma", "ES_D_lambda", "ES_D_omega", "ES_D_gamma", "ES_D_epsilon",
        "ES_D_zeta", "ES_D_alpha", "ES_D_delta", "ES_D_kappa", "ES_D_tau", "ES_D_beta", "ES_D_mu_A",
        "ES_D_mu_min", "ES_A_mu", "ES_A_sigma", "ES_A_lambda", "ES_A_omega", "ES_A_gamma", "ES_A_epsilon",
        "ES_A_zeta", "ES_A_alpha", "ES_A_delta", "ES_A_kappa", "ES_A_tau", "ES_A_beta", "SF_sigma",
        "SF_lambda", "SF_omega", "SF_delta", "SF_kappa", "SF_tau", "SF_beta", "KF_mu", "KF_sigma", "KF_lambda",
        "KF_omega", "KF_gamma", "KF_epsilon", "KF_zeta", "KF_alpha", "KF_delta", "KF_kappa", "KF_tau",
        "KF_beta", "XPR_mu", "XPR_sigma", "XPR_lambda", "XPR_omega", "XPR_gamma", "XPR_epsilon", "XPR_zeta",
        "XPR_alpha", "XPR_delta", "XPR_kappa", "XPR_tau", "XPR_beta", "r_DS", "PerClusterDS",
        "PerClusterDS_gamma", "PerClusterDS_min", "PerClusterAS_D", "PerClusterAS_A", "PerClusterES_D",
        "PerClusterES_A", "absTOA_mu", "absTOA_sigma", "absTOA_lambda", "LOS_scatter_radius", "SC_lambda",
        "LNS_ksi", "GR_enabled", "asD_ds", "asA_ds", "asA_sf", "asD_sf", "ds_sf", "asD_asA",
        "asD_kf", "asA_kf", "ds_kf", "sf_kf", "esD_ds", "esA_ds", "esA_sf", "esD_sf", "esD_esA", "esD_asD",
        "esD_asA", "esA_asD", "esA_asA", "esD_kf", "esA_kf", "xpr_ds", "xpr_kf", "xpr_sf", "xpr_asd",
        "xpr_asa", "xpr_esd", "xpr_esa",
        // New keys for TR 38.901 v19.5.0
        "LSP_fc_min", "ES_D_mu_B", "ES_D_mu_C", "ES_D_mu_D", "EoD_offset_e", "EoD_offset_e_gamma",
        "EoD_offset_a", "EoD_offset_a_gamma", "EoD_offset_b", "EoD_offset_c", "EoD_offset_c_gamma",
        "EoD_offset_c_hUT", "EoD_offset_hcl", "PerClusterDS_fc_min", "PerClusterES_D_scale", "absTOA_Lmax",
        "NumClusters_min", "NF_k1", "NF_alpha", "NF_beta", "SNS_mu", "SNS_sigma", "SNS_A", "SNS_B", "SNS_R",
        "SNS_delta_var", "PolVar_sigma"};
        return k;
    }

    bool ends_with(const std::string &s, const std::string &e)
    {
        return s.size() >= e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0;
    }

    // Oracle: default value of a key, written as rules read from set_scenario_table.m.
    // Everything that is not named here is 0, including all 27 new keys.
    double ref_default(const std::string &k)
    {
        if (k == "NumClusters") return 6.0;
        if (k == "NumSubPaths") return 20.0;
        if (k == "r_DS") return 2.0;
        if (k == "LNS_ksi") return 3.0;
        if (k == "absTOA_mu") return -100.0;
        if (k == "absTOA_lambda" || k == "SC_lambda") return 0.0;
        if (k == "ES_D_mu_min") return -INF;
        if (ends_with(k, "_lambda")) return 20.0; // the 8 decorrelation distances
        if (k == "DS_mu") return -6.0;
        if (k == "DS_sigma") return 0.5;
        if (k == "AS_D_mu" || k == "AS_A_mu") return 1.0;
        if (k == "ES_D_mu" || k == "ES_A_mu") return 0.3;
        if (k == "AS_D_sigma" || k == "AS_A_sigma" || k == "ES_D_sigma" || k == "ES_A_sigma") return 0.2;
        if (k == "SF_sigma") return 8.0;
        if (k == "XPR_sigma") return 10.0;
        return 0.0;
    }

    // Oracle: position of the 28 correlation keys in the 8 x 8 matrix, 0-based, typed from get.lsp_xcorr in
    // qd_builder.m. Order of rows and columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR.
    struct xc_pos
    {
        const char *key;
        int row, col;
    };
    const xc_pos XC[28] = {
        {"ds_kf", 0, 1}, {"ds_sf", 0, 2}, {"asD_ds", 0, 3}, {"asA_ds", 0, 4}, {"esD_ds", 0, 5}, {"esA_ds", 0, 6},
        {"xpr_ds", 0, 7}, {"sf_kf", 1, 2}, {"asD_kf", 1, 3}, {"asA_kf", 1, 4}, {"esD_kf", 1, 5}, {"esA_kf", 1, 6},
        {"xpr_kf", 1, 7}, {"asD_sf", 2, 3}, {"asA_sf", 2, 4}, {"esD_sf", 2, 5}, {"esA_sf", 2, 6}, {"xpr_sf", 2, 7},
        {"asD_asA", 3, 4}, {"esD_asD", 3, 5}, {"esA_asD", 3, 6}, {"xpr_asd", 3, 7}, {"esD_asA", 4, 5},
        {"esA_asA", 4, 6}, {"xpr_asa", 4, 7}, {"esD_esA", 5, 6}, {"xpr_esd", 5, 7}, {"xpr_esa", 6, 7}};

    // Exact comparison that treats equal infinities as equal
    bool same(double a, double b) { return a == b; }

    // Number of entries of 'sp' that differ from the default oracle; -1 if the key set is not the 165 keys
    int n_not_default(const std::unordered_map<std::string, double> &sp, const std::string &except = "")
    {
        if (sp.size() != ref_key_order().size())
            return -1;
        int n = 0;
        for (const std::string &k : ref_key_order())
        {
            auto it = sp.find(k);
            if (it == sp.end())
                return -1;
            if (k != except && !same(it->second, ref_default(k)))
                ++n;
        }
        return n;
    }

    typedef std::unordered_map<std::string, double> spmap;

    // One key at one value fails the checks, passes without checks and is then returned unchanged
    void rejects_value(const std::string &key, double value)
    {
        INFO("key " << key << " = " << value);
        spmap sp;
        sp[key] = value;
        CHECK_THROWS_AS(quadriga_lib::scenpar_validate(sp), std::invalid_argument);
        CHECK(sp.size() == 1); // a failed call does not complete the set

        CHECK_NOTHROW(quadriga_lib::scenpar_validate(sp, "legacy", nullptr, nullptr, false));
        CHECK(sp.size() == 165);
        CHECK((sp[key] == value || (value != value && sp[key] != sp[key])));
    }

    // One key at one value passes and keeps its value
    void accepts(const std::string &key, double value)
    {
        INFO("key " << key << " = " << value);
        spmap sp;
        sp[key] = value;
        CHECK_NOTHROW(quadriga_lib::scenpar_validate(sp));
        CHECK(sp.size() == 165);
        CHECK(sp[key] == value);
    }
}

TEST_CASE("scenpar_validate - an empty map returns the 165 default values")
{
    spmap sp;
    arma::mat X;
    std::vector<std::string> order;

    bool ok = quadriga_lib::scenpar_validate(sp, "legacy", &X, &order);

    CHECK(ok);
    CHECK(sp.size() == 165);
    CHECK(n_not_default(sp) == 0);
    CHECK(order == ref_key_order());
    REQUIRE((X.n_rows == 8 && X.n_cols == 8));
    CHECK(arma::approx_equal(X, arma::eye(8, 8), "absdiff", 0.0));

    QD_DIAG("\n# diag: scenpar_validate defaults, first and last 5 keys of key_order\n");
    QD_DIAG("index,key,value,ref\n");
    for (size_t i = 0; i < order.size(); ++i)
        if (i < 5 || i + 5 >= order.size())
            QD_DIAG("%d,%s,%g,%g\n", (int)i, order[i].c_str(), sp.count(order[i]) ? sp[order[i]] : QNAN,
                    ref_default(order[i]));

    // The default arguments give the same result
    spmap sp2;
    CHECK(quadriga_lib::scenpar_validate(sp2) == ok);
    CHECK(sp2 == sp);
}

TEST_CASE("scenpar_validate - a legacy set with 138 numeric keys passes and grows to 165")
{
    // Distinct, valid, non-default values for the 138 legacy keys:
    // integers for the two counts, small correlations, positive values elsewhere
    spmap sp, given;
    for (int i = 0; i < 138; ++i)
    {
        const std::string &k = ref_key_order()[i];
        double v = 0.5 + 0.25 * i;
        if (k == "NumClusters" || k == "NumSubPaths")
            v = 30.0 + i;
        if (i >= 110) // the 28 correlations
            v = 0.001 * (i - 109);
        sp[k] = v;
    }
    given = sp;
    REQUIRE(sp.size() == 138);

    arma::mat X;
    bool ok = quadriga_lib::scenpar_validate(sp, "mmMAGIC", &X);

    CHECK(ok);
    CHECK(sp.size() == 165);
    int n_kept = 0, n_new_default = 0;
    for (int i = 0; i < 165; ++i)
    {
        const std::string &k = ref_key_order()[i];
        if (i < 138 && sp.count(k) && sp[k] == given[k])
            ++n_kept;
        if (i >= 138 && sp.count(k) && sp[k] == 0.0)
            ++n_new_default;
    }
    CHECK(n_kept == 138);        // given values are not changed
    CHECK(n_new_default == 27);  // the 27 new keys are added with default 0

    // The matrix is built from the given correlations
    REQUIRE((X.n_rows == 8 && X.n_cols == 8));
    arma::mat ref = arma::eye(8, 8);
    for (int i = 0; i < 28; ++i)
    {
        ref(XC[i].row, XC[i].col) = given[XC[i].key];
        ref(XC[i].col, XC[i].row) = given[XC[i].key];
    }
    CHECK(arma::approx_equal(X, ref, "absdiff", 0.0));
}

TEST_CASE("scenpar_validate - each correlation key lands in two symmetric positions of lsp_xcorr")
{
    for (int i = 0; i < 28; ++i)
    {
        INFO("key " << XC[i].key);
        const double v = 0.01 * (i + 1) * ((i % 2) ? -1.0 : 1.0);
        spmap sp;
        sp[XC[i].key] = v;
        arma::mat X;
        CHECK(quadriga_lib::scenpar_validate(sp, "legacy", &X));
        REQUIRE((X.n_rows == 8 && X.n_cols == 8));

        arma::mat ref = arma::eye(8, 8);
        ref(XC[i].row, XC[i].col) = v;
        ref(XC[i].col, XC[i].row) = v;
        CHECK(arma::approx_equal(X, ref, "absdiff", 0.0));
    }
}

TEST_CASE("scenpar_validate - the return value tells if lsp_xcorr is positive definite")
{
    // Oracle: for 3 coupled LSPs with correlations a, b, c the matrix is positive definite only if
    // abs(a), abs(b), abs(c) < 1 and det = 1 + 2abc - a^2 - b^2 - c^2 > 0. DS-KF-SF: a = ds_kf, b = ds_sf, c = sf_kf.
    QD_DIAG("\n# diag: scenpar_validate positive-definite flag, DS-KF-SF triple\n");
    QD_DIAG("ds_kf,ds_sf,sf_kf,det,flag,ref\n");
    const double grid[] = {-0.95, -0.6, -0.2, 0.0, 0.3, 0.7, 0.95};
    int n_true = 0, n_false = 0;
    for (double a : grid)
        for (double b : grid)
            for (double c : grid)
            {
                const double det = 1.0 + 2.0 * a * b * c - a * a - b * b - c * c;
                if (std::abs(det) < 1e-9) // the oracle is not decisive at the boundary
                    continue;
                const bool ref = det > 0.0;
                spmap sp = {{"ds_kf", a}, {"ds_sf", b}, {"sf_kf", c}};
                bool ok = !ref;
                CHECK_NOTHROW(ok = quadriga_lib::scenpar_validate(sp));
                INFO("ds_kf " << a << ", ds_sf " << b << ", sf_kf " << c << ", det " << det);
                CHECK(ok == ref);
                (ref ? n_true : n_false) += 1;
                if (a == 0.7 && b == 0.7)
                    QD_DIAG("%g,%g,%g,%.4f,%d,%d\n", a, b, c, det, (int)ok, (int)ref);
            }
    QD_DIAG("cases: %d positive definite, %d not\n", n_true, n_false);
    CHECK(n_true > 50); // both outcomes are exercised
    CHECK(n_false > 50);

    // A triple that does not involve the first rows: ESD-ESA-XPR
    spmap sp = {{"esD_esA", 0.9}, {"xpr_esd", 0.9}, {"xpr_esa", -0.9}};
    CHECK_FALSE(quadriga_lib::scenpar_validate(sp));
    sp = {{"esD_esA", 0.9}, {"xpr_esd", 0.9}, {"xpr_esa", 0.9}};
    CHECK(quadriga_lib::scenpar_validate(sp));

    // Singular: one correlation of exactly 1
    sp = {{"xpr_esa", 1.0}};
    CHECK_FALSE(quadriga_lib::scenpar_validate(sp));

    // Without checks: values outside [-1, 1] and NaN give 'false', not an error
    sp = {{"ds_kf", 3.0}};
    CHECK_FALSE(quadriga_lib::scenpar_validate(sp, "legacy", nullptr, nullptr, false));
    sp = {{"ds_kf", QNAN}};
    CHECK_FALSE(quadriga_lib::scenpar_validate(sp, "legacy", nullptr, nullptr, false));
}

TEST_CASE("scenpar_validate - value checks per validation class")
{
    SECTION("valid baseline")
    {
        accepts("DS_mu", -7.0);
        accepts("NumClusters", 12.0);
    }

    SECTION("NaN is an error for every key (Q12)")
    {
        for (const std::string &k : ref_key_order())
            rejects_value(k, QNAN);
    }

    SECTION("integer and > 0")
    {
        for (const char *k : {"NumClusters", "NumSubPaths"})
        {
            accepts(k, 1.0);
            accepts(k, 40.0);
            accepts(k, 1000.0); // Q11: upper bound 1000
            rejects_value(k, 1001.0);
            rejects_value(k, 1e300);
            rejects_value(k, 0.0);
            rejects_value(k, -1.0);
            rejects_value(k, 2.5);
            rejects_value(k, INF);
        }
    }

    SECTION("positive")
    {
        accepts("r_DS", 1e-9);
        rejects_value("r_DS", 0.0);
        rejects_value("r_DS", -1.0);
        rejects_value("r_DS", INF);
    }

    SECTION("non-negative")
    {
        for (const char *k : {"PerClusterAS_D", "PerClusterAS_A", "PerClusterES_D", "PerClusterES_A", "LNS_ksi",
                              "LOS_scatter_radius", "GR_enabled", "SC_lambda", "DS_lambda", "AS_D_lambda",
                              "AS_A_lambda", "ES_D_lambda", "ES_A_lambda", "SF_lambda", "KF_lambda", "XPR_lambda",
                              "PerClusterDS", "PerClusterDS_min", "absTOA_lambda",
                              "LSP_fc_min", "EoD_offset_b", "EoD_offset_hcl", "PerClusterDS_fc_min",
                              "PerClusterES_D_scale", "absTOA_Lmax", "SNS_sigma", "SNS_A", "SNS_B", "SNS_R",
                              "SNS_delta_var", "PolVar_sigma"})
        {
            accepts(k, 0.0);
            accepts(k, 3.5);
            rejects_value(k, -1e-9);
            rejects_value(k, INF);
        }
    }

    SECTION("real and unchecked keys accept any value except NaN")
    {
        // Every key that is in none of the bounded classes: the 35 'real' and 53 unchecked legacy keys
        // and the 10 new 'real' keys
        const std::vector<std::string> bounded = {
            "NumClusters", "NumSubPaths", "r_DS", "PerClusterAS_D", "PerClusterAS_A", "PerClusterES_D",
            "PerClusterES_A", "LNS_ksi", "LOS_scatter_radius", "GR_enabled", "SC_lambda", "DS_lambda", "AS_D_lambda",
            "AS_A_lambda", "ES_D_lambda", "ES_A_lambda", "SF_lambda", "KF_lambda", "XPR_lambda", "PerClusterDS",
            "PerClusterDS_min", "absTOA_lambda", "LSP_fc_min", "EoD_offset_b", "EoD_offset_hcl",
            "PerClusterDS_fc_min", "PerClusterES_D_scale", "absTOA_Lmax", "NumClusters_min", "NF_k1", "NF_alpha",
            "NF_beta", "SNS_mu", "SNS_sigma", "SNS_A", "SNS_B", "SNS_R", "SNS_delta_var", "PolVar_sigma"};
        int n = 0;
        for (int i = 0; i < 165; ++i)
        {
            const std::string &k = ref_key_order()[i];
            bool is_bounded = (i >= 110 && i < 138); // correlations
            for (const std::string &b : bounded)
                is_bounded = is_bounded || b == k;
            if (is_bounded)
                continue;
            ++n;
            accepts(k, -123.5);
            accepts(k, INF);
            accepts(k, -INF);
        }
        CHECK(n == 35 + 53 + 10);
    }

    SECTION("correlation in [-1, 1], the 7 xpr_ keys included (Q13)")
    {
        for (int i = 0; i < 28; ++i)
        {
            accepts(XC[i].key, 1.0);
            accepts(XC[i].key, -1.0);
            rejects_value(XC[i].key, 1.0 + 1e-9);
            rejects_value(XC[i].key, -1.0 - 1e-9);
            rejects_value(XC[i].key, INF);
        }
    }

    SECTION("in [0, 1]")
    {
        accepts("SNS_mu", 0.0);
        accepts("SNS_mu", 1.0);
        rejects_value("SNS_mu", -1e-9);
        rejects_value("SNS_mu", 1.0 + 1e-9);
    }

    SECTION("integer and >= 0")
    {
        accepts("NumClusters_min", 0.0);
        accepts("NumClusters_min", 6.0);
        accepts("NF_k1", 0.0);
        accepts("NF_k1", 4.0);
        accepts("NF_k1", 1000.0); // Q11: upper bound 1000
        rejects_value("NF_k1", 1001.0);
        rejects_value("NumClusters_min", 1.5);
        rejects_value("NumClusters_min", -1.0);
        rejects_value("NF_k1", 0.5);
        rejects_value("NF_k1", -1.0);
        rejects_value("NF_k1", INF);
    }

    SECTION("the bound 1000 of the counts is named in the message (Q11)")
    {
        // NumClusters_min = 1001 also exceeds NumClusters = 1000; the class check comes first and names the bound
        auto message = [](spmap sp) -> std::string
        {
            try
            {
                quadriga_lib::scenpar_validate(sp);
            }
            catch (const std::invalid_argument &e)
            {
                return e.what();
            }
            return "<no error>";
        };
        CHECK(message({{"NumClusters", 1000.0}, {"NumClusters_min", 1000.0}}) == "<no error>");
        CHECK(message({{"NumClusters", 1000.0}, {"NumClusters_min", 1001.0}}) ==
              "Value of 'NumClusters_min' must be an integer from 0 to 1000.");
        CHECK(message({{"NumSubPaths", 1001.0}}) == "Value of 'NumSubPaths' must be an integer from 1 to 1000.");
        CHECK(message({{"NumClusters", 0.0}}) == "Value of 'NumClusters' must be an integer from 1 to 1000.");
        CHECK(message({{"NF_k1", -1.0}}) == "Value of 'NF_k1' must be an integer from 0 to 1000.");
        QD_DIAG("\n# diag: scenpar_validate count bound: %s\n",
                message({{"NumClusters", 1000.0}, {"NumClusters_min", 1001.0}}).c_str());
    }

    SECTION("cross-key checks")
    {
        spmap sp = {{"NumClusters", 20.0}, {"NumClusters_min", 20.0}};
        CHECK_NOTHROW(quadriga_lib::scenpar_validate(sp));
        sp = {{"NumClusters", 20.0}, {"NumClusters_min", 21.0}};
        CHECK_THROWS_AS(quadriga_lib::scenpar_validate(sp), std::invalid_argument);
        CHECK_NOTHROW(quadriga_lib::scenpar_validate(sp, "legacy", nullptr, nullptr, false));
        rejects_value("NumClusters_min", 7.0); // default NumClusters is 6

        sp = {{"NF_alpha", 1.53}, {"NF_beta", 1.42}};
        CHECK_NOTHROW(quadriga_lib::scenpar_validate(sp));
        rejects_value("NF_alpha", 1.53);
        rejects_value("NF_beta", 1.42);
    }

    SECTION("subpath_method is one of three names (Q14)")
    {
        spmap sp;
        CHECK_NOTHROW(quadriga_lib::scenpar_validate(sp, "legacy"));
        CHECK_NOTHROW(quadriga_lib::scenpar_validate(sp, "Laplacian"));
        CHECK_NOTHROW(quadriga_lib::scenpar_validate(sp, "mmMAGIC"));
        CHECK_THROWS_AS(quadriga_lib::scenpar_validate(sp, "Legacy"), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::scenpar_validate(sp, ""), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::scenpar_validate(sp, "other"), std::invalid_argument);
        CHECK_NOTHROW(quadriga_lib::scenpar_validate(sp, "other", nullptr, nullptr, false));
    }
}

TEST_CASE("scenpar_validate - input validation")
{
    spmap sp = {{"DS_mu", -7.0}};
    arma::mat X(2, 2);
    X.fill(7.0);
    std::vector<std::string> order = {"marker"};

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::scenpar_validate(sp, "legacy", &X, &order));
        CHECK(sp.size() == 165);
        CHECK(order.size() == 165);
        CHECK((X.n_rows == 8 && X.n_cols == 8));
    }

    SECTION("unknown key, with and without value checks (Q11)")
    {
        sp["DS_mu_typo"] = 1.0;
        CHECK_THROWS_AS(quadriga_lib::scenpar_validate(sp, "legacy", &X, &order), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::scenpar_validate(sp, "legacy", &X, &order, false), std::invalid_argument);
        CHECK(sp.size() == 2); // nothing was written
        CHECK(order == std::vector<std::string>{"marker"});
        CHECK((X.n_rows == 2 && X.n_cols == 2 && X(1, 1) == 7.0));
    }

    SECTION("the legacy key 'GR_epsilon' is not supported")
    {
        sp["GR_epsilon"] = 10.0;
        CHECK_THROWS_AS(quadriga_lib::scenpar_validate(sp), std::invalid_argument);
        CHECK_THROWS_AS(quadriga_lib::scenpar_validate(sp, "legacy", nullptr, nullptr, false), std::invalid_argument);
    }

    SECTION("'SubpathMethod' is not a numeric key")
    {
        sp["SubpathMethod"] = 1.0;
        CHECK_THROWS_AS(quadriga_lib::scenpar_validate(sp), std::invalid_argument);
    }

    SECTION("failed value check leaves the outputs untouched")
    {
        sp["r_DS"] = -1.0;
        CHECK_THROWS_AS(quadriga_lib::scenpar_validate(sp, "legacy", &X, &order), std::invalid_argument);
        CHECK(sp.size() == 2);
        CHECK(sp["r_DS"] == -1.0);
        CHECK(order == std::vector<std::string>{"marker"});
        CHECK((X.n_rows == 2 && X.n_cols == 2 && X(1, 1) == 7.0));
    }
}

TEST_CASE("scenpar_validate - a pre-sized lsp_xcorr is written in place and the call is repeatable")
{
    spmap sp = {{"ds_kf", -0.4}, {"xpr_esa", 0.25}};
    arma::mat X(8, 8);
    X.fill(7.0);
    const double *mem = X.memptr();

    bool ok = quadriga_lib::scenpar_validate(sp, "legacy", &X);
    CHECK(ok);
    CHECK(X.memptr() == mem);
    arma::mat ref = arma::eye(8, 8);
    ref(0, 1) = ref(1, 0) = -0.4;
    ref(6, 7) = ref(7, 6) = 0.25;
    CHECK(arma::approx_equal(X, ref, "absdiff", 0.0));

    // A complete set is a fixed point: validating it again changes nothing
    spmap before = sp;
    arma::mat X2;
    CHECK(quadriga_lib::scenpar_validate(sp, "legacy", &X2) == ok);
    CHECK(sp == before);
    CHECK(arma::approx_equal(X2, ref, "absdiff", 0.0));
}
