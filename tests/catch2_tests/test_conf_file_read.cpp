// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for quadriga_lib::conf_file_read.
//
// Pins down: the line grammar of QuaDRiGa scenario files (*.conf), the default parameter set, the
// path-loss outputs, the mapping of the 28 correlation keys into the 8 x 8 LSP cross-correlation
// matrix, the positive-definite flag, the value checks per validation class and the error behaviour.
// Left to other suites: the value checks on a parameter set without a file (test_scenpar_validate),
// the directory listing (test_conf_file_list).
//
// Oracles: expected values are written by hand next to the file content of each case. Key order and
// defaults are typed from QuaDRiGa's set_scenario_table.m, the matrix positions from get.lsp_xcorr in
// qd_builder.m. Numbers are compared exactly: the parser and the compiler both round a decimal
// literal correctly, so the same literal gives the same double.
//
// Conventions: every case writes its own file 'test_conf_file_read_*.conf' into the working directory
// and removes it. Angles in the files are elevation angles, as everywhere in QuaDRiGa.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
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

    // Oracle: canonical key order. The 139 numeric legacy keys are typed from the struct(...) call in
    // QuaDRiGa's set_scenario_table.m (without 'SubpathMethod'), the 27 new keys from section 7.2 of the plan.
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
        "LNS_ksi", "GR_enabled", "GR_epsilon", "asD_ds", "asA_ds", "asA_sf", "asD_sf", "ds_sf", "asD_asA",
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

    // Number of entries of 'sp' that differ from the default oracle; -1 if the key set is not the 166 keys
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

    // All outputs of one call
    struct conf_out
    {
        bool ok = false;
        std::unordered_map<std::string, double> scenpar;
        std::string subpath_method, pl_model;
        std::unordered_map<std::string, std::vector<double>> plpar;
        arma::mat lsp_xcorr;
        std::vector<std::string> key_order, pl_order, short_names, ignored_keys;
    };

    // Removes a file when it goes out of scope
    struct tmp_file
    {
        std::string fn;
        tmp_file(const std::string &name, const std::string &content) : fn(name)
        {
            std::ofstream f(fn, std::ios::binary);
            f << content;
        }
        ~tmp_file() { std::remove(fn.c_str()); }
    };

    conf_out read_all(const std::string &fn, bool validate = true)
    {
        conf_out r;
        r.ok = quadriga_lib::conf_file_read(fn, r.scenpar, &r.subpath_method, &r.pl_model, &r.plpar, &r.lsp_xcorr,
                                            &r.key_order, &r.pl_order, &r.short_names, &r.ignored_keys, validate);
        return r;
    }

    // Write 'content' to a file, read it with all outputs, remove the file
    conf_out read_text(const std::string &content, bool validate = true)
    {
        tmp_file f("test_conf_file_read_tmp.conf", content);
        return read_all(f.fn, validate);
    }

    // Value of a key, NaN if the key is missing
    double val(const conf_out &r, const std::string &key)
    {
        auto it = r.scenpar.find(key);
        return it == r.scenpar.end() ? QNAN : it->second;
    }

    // A line that is well formed but fails a value check: error with validate = true, accepted with false
    void rejects_value(const std::string &line)
    {
        INFO("line: " << line);
        CHECK_THROWS_AS(read_text(line + "\n"), std::invalid_argument);
        CHECK_NOTHROW(read_text(line + "\n", false));
    }

    // A line that passes
    void accepts(const std::string &line)
    {
        INFO("line: " << line);
        CHECK_NOTHROW(read_text(line + "\n"));
    }

    // A line that violates the grammar: error in both modes
    void rejects_grammar(const std::string &line)
    {
        INFO("line: " << line);
        CHECK_THROWS_AS(read_text(line + "\n"), std::invalid_argument);
        CHECK_THROWS_AS(read_text(line + "\n", false), std::invalid_argument);
    }

    // Correlation keys of 3GPP_38.901_UMa_LOS.conf (v16.1.0), typed from the shipped file
    const char *UMA_LOS_XCORR =
        "ds_kf = -0.4\nds_sf = -0.4\nasD_ds = 0.4\nasA_ds = 0.8\nesD_ds = -0.2\nasA_kf = -0.2\n"
        "asD_sf = -0.5\nasA_sf = -0.5\nesA_sf = -0.8\nesD_asD = 0.5\nesD_asA = -0.3\nesA_asA = 0.4\n";
}

TEST_CASE("conf_file_read - an empty file gives the 166 default values")
{
    conf_out r = read_text("");

    CHECK(r.ok); // the default correlation matrix is the identity
    CHECK(r.scenpar.size() == 166);
    CHECK(n_not_default(r.scenpar) == 0);
    CHECK(same(val(r, "ES_D_mu_min"), -INF));
    CHECK(r.subpath_method == "legacy");
    CHECK(r.pl_model.empty());
    CHECK(r.plpar.empty());
    CHECK(r.pl_order.empty());
    CHECK(r.short_names.empty());
    CHECK(r.ignored_keys.empty());

    REQUIRE(r.key_order.size() == 166);
    CHECK(r.key_order == ref_key_order());
    CHECK(r.key_order[105] == "absTOA_lambda"); // 'SubpathMethod' is legacy field 107, between these two
    CHECK(r.key_order[106] == "LOS_scatter_radius");

    REQUIRE((r.lsp_xcorr.n_rows == 8 && r.lsp_xcorr.n_cols == 8));
    CHECK(arma::approx_equal(r.lsp_xcorr, arma::eye(8, 8), "absdiff", 0.0));

    QD_DIAG("\n# diag: conf_file_read defaults that are not 0\n");
    QD_DIAG("key,value,ref\n");
    for (const std::string &k : ref_key_order())
        if (ref_default(k) != 0.0)
            QD_DIAG("%s,%g,%g\n", k.c_str(), val(r, k), ref_default(k));

    // A file with comments and blank lines only is the same as an empty file
    conf_out c = read_text("% Comment\n\n   \t\n%% DS_mu = 3\n   % indented comment\r\n");
    CHECK(n_not_default(c.scenpar) == 0);
}

TEST_CASE("conf_file_read - line grammar: comments, white space, line ends and number forms")
{
    const std::string content =
        "% Comment line with an = sign\r\n"
        "\n"
        "   \t \n"
        "DS_mu = -6.28   % comment = with equal sign\n"
        "\tDS_sigma\t=\t0.39\t% tabs\n"
        "AS_D_mu=1.5\n"
        "AS_D_sigma = 1e-1\r\n"
        "AS_A_mu = 2E0\n"
        "AS_A_sigma = 2.5e+1\n"
        "ES_D_mu = 1.5E-2\n"
        "ES_D_sigma = .5\n"
        "ES_A_mu = 5.\n"
        "ES_A_sigma = +2\n"
        "ES_D_mu_min = -3\n"
        "DS_gamma = -Inf\n"
        "AS_D_gamma = Inf\n"
        "AS_A_gamma = +INF\n"
        "ES_A_gamma = -inf\r\n"
        "% DS_lambda = 99\n"
        "  % KF_mu = 7\n"
        "DS_lambda = 30 % KF_sigma = 3\n"
        "SC_lambda = 50\n"
        "SC_lambda = 60\n"
        "XPR_mu = 9 % non-ASCII bytes: caf\xc3\xa9 \xe9\xff\n"
        "SubpathMethod = mmMAGIC   % string value\n"
        "MIMOSA_like = 6.2e-07\n"
        "PerClusterAS_D = 6.2e-07\n"
        "NumClusters = 12"; // no line end at the end of the file

    struct kv
    {
        const char *key;
        double ref;
    };
    const kv expect[] = {
        {"DS_mu", -6.28}, {"DS_sigma", 0.39}, {"AS_D_mu", 1.5}, {"AS_D_sigma", 0.1}, {"AS_A_mu", 2.0},
        {"AS_A_sigma", 25.0}, {"ES_D_mu", 0.015}, {"ES_D_sigma", 0.5}, {"ES_A_mu", 5.0}, {"ES_A_sigma", 2.0},
        {"ES_D_mu_min", -3.0}, {"DS_gamma", -INF}, {"AS_D_gamma", INF}, {"AS_A_gamma", INF}, {"ES_A_gamma", -INF},
        {"DS_lambda", 30.0}, {"SC_lambda", 60.0}, {"XPR_mu", 9.0}, {"PerClusterAS_D", 6.2e-07}, {"NumClusters", 12.0}};
    const int n_expect = (int)(sizeof(expect) / sizeof(expect[0]));

    conf_out r = read_text(content);

    QD_DIAG("\n# diag: conf_file_read line grammar\n");
    QD_DIAG("key,value,ref,diff\n");
    for (int i = 0; i < n_expect; ++i)
    {
        double v = val(r, expect[i].key);
        QD_DIAG("%s,%.17g,%.17g,%.3e\n", expect[i].key, v, expect[i].ref, v - expect[i].ref);
        INFO("key " << expect[i].key);
        CHECK(same(v, expect[i].ref));
    }

    // Nothing else moved: comments did not assign KF_mu, KF_sigma or DS_lambda = 99
    CHECK(n_not_default(r.scenpar) == n_expect);
    CHECK(same(val(r, "KF_mu"), 0.0));
    CHECK(same(val(r, "KF_sigma"), 0.0));
    CHECK(r.subpath_method == "mmMAGIC");
    CHECK(r.ignored_keys == std::vector<std::string>{"MIMOSA_like"});
    CHECK(r.ok);
}

TEST_CASE("conf_file_read - unknown keys are ignored and reported once in file order")
{
    conf_out r = read_text("Zeta_unknown = 1\n"
                           "DS_mu = -7\n"
                           "Alpha_unknown = some words 1 2 3 ; = !\n"
                           "Zeta_unknown = 2\n"
                           "ds_mu = 4\n" // keys are case-sensitive
                           "ShortName = Xy\n"
                           "PL_A = 3\n"
                           "XPL_A = 3\n"
                           "My_PL_key = 3\n"); // Q7: 'PL_' in the middle is not a path-loss key

    CHECK(r.ignored_keys == std::vector<std::string>{"Zeta_unknown", "Alpha_unknown", "ds_mu", "XPL_A", "My_PL_key"});
    CHECK(same(val(r, "DS_mu"), -7.0));
    CHECK(n_not_default(r.scenpar) == 1);
    CHECK(r.scenpar.size() == 166); // unknown keys do not enter scenpar
    CHECK(r.plpar.size() == 1);
    CHECK(r.plpar.count("A") == 1);
}

TEST_CASE("conf_file_read - ShortName lines are returned in file order")
{
    conf_out r = read_text("ShortName = Ul   % first\n"
                           "% ShortName = InComment\n"
                           "DS_mu = -7 % ShortName = AfterValue\n"
                           "ShortName=UMa_2\n");

    CHECK(r.short_names == std::vector<std::string>{"Ul", "UMa_2"});
    CHECK(r.ignored_keys.empty()); // 'ShortName' is a known key
    CHECK(r.scenpar.count("ShortName") == 0);
}

TEST_CASE("conf_file_read - path-loss keys: model, several values, file order, duplicates")
{
    SECTION("no PL_ key")
    {
        conf_out r = read_text("DS_mu = -7\n");
        CHECK(r.pl_model.empty());
        CHECK(r.plpar.empty());
        CHECK(r.pl_order.empty());
    }

    SECTION("model only")
    {
        conf_out r = read_text("PL_model = logdist_simple % comment\n");
        CHECK(r.pl_model == "logdist_simple");
        CHECK(r.plpar.empty()); // the model is not an entry of plpar
        CHECK(r.pl_order.empty());
    }

    SECTION("parameters without a model")
    {
        conf_out r = read_text("PL_A = 21\n");
        CHECK(r.pl_model.empty());
        CHECK(r.plpar.size() == 1);
        CHECK(r.pl_order == std::vector<std::string>{"A"});
    }

    SECTION("values, order and duplicates")
    {
        conf_out r = read_text("PL_sig1 = 4\n"
                               "PL_model = dual_slope\n"
                               "PL_A = 21.5\n"
                               "PL_B = 1 2.5\t-3e1 +Inf  % several values\n"
                               "PL_E3n=-0.6\r\n"
                               "PL_A = 22   % stated twice: last value, first position\n"
                               "PL_model = nlos\n");

        CHECK(r.pl_model == "nlos");
        CHECK(r.pl_order == std::vector<std::string>{"sig1", "A", "B", "E3n"});
        REQUIRE(r.plpar.size() == 4);
        CHECK(r.plpar.count("model") == 0);
        CHECK(r.plpar["sig1"] == std::vector<double>{4.0});
        CHECK(r.plpar["A"] == std::vector<double>{22.0});
        CHECK(r.plpar["B"] == std::vector<double>{1.0, 2.5, -30.0, INF});
        CHECK(r.plpar["E3n"] == std::vector<double>{-0.6});
        CHECK(n_not_default(r.scenpar) == 0); // path-loss keys do not touch scenpar
        CHECK(r.ignored_keys.empty());

        QD_DIAG("\n# diag: conf_file_read path-loss keys\n");
        QD_DIAG("position,key,n_values,first_value\n");
        for (size_t i = 0; i < r.pl_order.size(); ++i)
        {
            const std::vector<double> &v = r.plpar[r.pl_order[i]];
            QD_DIAG("%d,%s,%d,%g\n", (int)i, r.pl_order[i].c_str(), (int)v.size(), v.empty() ? QNAN : v[0]);
        }
    }
}

TEST_CASE("conf_file_read - each correlation key lands in two symmetric positions of lsp_xcorr")
{
    // One file per key, the key alone at a distinct value: exactly (row, col) and (col, row) move
    QD_DIAG("\n# diag: conf_file_read correlation key positions (0-based)\n");
    QD_DIAG("key,value,row,col,found_row,found_col,n_changed\n");
    for (int i = 0; i < 28; ++i)
    {
        const double v = 0.01 * (i + 1) * ((i % 2) ? -1.0 : 1.0);
        conf_out r = read_text(std::string(XC[i].key) + " = " + std::to_string(v) + "\n");
        INFO("key " << XC[i].key);
        REQUIRE((r.lsp_xcorr.n_rows == 8 && r.lsp_xcorr.n_cols == 8));

        arma::mat ref = arma::eye(8, 8);
        ref(XC[i].row, XC[i].col) = v;
        ref(XC[i].col, XC[i].row) = v;
        CHECK(arma::approx_equal(r.lsp_xcorr, ref, "absdiff", 1e-15)); // to_string keeps 6 decimals, v has 2
        CHECK(r.ok);

        int n_changed = 0, fr = -1, fc = -1;
        for (int a = 0; a < 8; ++a)
            for (int b = 0; b < 8; ++b)
                if (r.lsp_xcorr(a, b) != (a == b ? 1.0 : 0.0))
                {
                    ++n_changed;
                    if (a < b)
                        fr = a, fc = b;
                }
        QD_DIAG("%s,%.2f,%d,%d,%d,%d,%d\n", XC[i].key, v, XC[i].row, XC[i].col, fr, fc, n_changed);
        CHECK(n_changed == 2);
    }

    // All 28 at once fill the matrix completely, and the scenpar value equals the matrix element
    std::string all;
    arma::mat ref = arma::eye(8, 8);
    for (int i = 0; i < 28; ++i)
    {
        const double v = 0.001 * (i + 1);
        all += std::string(XC[i].key) + " = " + std::to_string(v) + "\n";
        ref(XC[i].row, XC[i].col) = v;
        ref(XC[i].col, XC[i].row) = v;
    }
    conf_out r = read_text(all);
    REQUIRE((r.lsp_xcorr.n_rows == 8 && r.lsp_xcorr.n_cols == 8));
    CHECK(arma::approx_equal(r.lsp_xcorr, ref, "absdiff", 1e-15));
    CHECK(n_not_default(r.scenpar) == 28);
}

TEST_CASE("conf_file_read - the return value tells if lsp_xcorr is positive definite")
{
    // Oracle for the 3-parameter case: a 3 x 3 correlation matrix with off-diagonals a, b, c is positive
    // definite only if its determinant 1 + 2abc - a^2 - b^2 - c^2 is > 0.
    // (0.9, 0.9, -0.9): 1 - 1.458 - 2.43 = -2.888 < 0, although each value is a valid correlation.
    struct pd_case
    {
        const char *name;
        std::string content;
        bool ref;
    };
    const pd_case cases[] = {
        {"identity", "", true},
        {"one correlation of 1 (singular)", "asD_ds = 1\n", false},
        {"one correlation of -1 (singular)", "esA_kf = -1\n", false},
        {"one correlation of 0.999999", "asD_ds = 0.999999\n", true},
        {"indefinite DS-KF-SF triple", "ds_kf = 0.9\nds_sf = 0.9\nsf_kf = -0.9\n", false},
        {"same triple, consistent signs", "ds_kf = 0.9\nds_sf = 0.9\nsf_kf = 0.9\n", true},
        {"3GPP_38.901_UMa_LOS", UMA_LOS_XCORR, true}};

    QD_DIAG("\n# diag: conf_file_read positive-definite flag\n");
    QD_DIAG("case,flag,ref\n");
    for (const pd_case &c : cases)
    {
        INFO("case " << c.name);
        conf_out r;
        REQUIRE_NOTHROW(r = read_text(c.content)); // not positive definite is not an error (Q15)
        QD_DIAG("%s,%d,%d\n", c.name, (int)r.ok, (int)c.ref);
        CHECK(r.ok == c.ref);
    }

    // Without value checks a correlation outside [-1, 1] is returned as it is, and the flag is false
    conf_out r = read_text("xpr_ds = 2\n", false);
    REQUIRE((r.lsp_xcorr.n_rows == 8 && r.lsp_xcorr.n_cols == 8));
    CHECK(r.lsp_xcorr(0, 7) == 2.0);
    CHECK(r.lsp_xcorr(7, 0) == 2.0);
    CHECK_FALSE(r.ok);
}

TEST_CASE("conf_file_read - value checks per validation class")
{
    SECTION("valid baseline")
    {
        accepts("");
        accepts("NumClusters = 12");
        conf_out r = read_text("DS_lambda = -1\n", false); // validate = false returns the value unchecked
        CHECK(same(val(r, "DS_lambda"), -1.0));
    }

    SECTION("integer and > 0")
    {
        accepts("NumClusters = 1");
        accepts("NumSubPaths = 2e1");
        rejects_value("NumClusters = 0");
        rejects_value("NumClusters = -3");
        rejects_value("NumClusters = 1.5");
        rejects_value("NumSubPaths = Inf");
    }

    SECTION("positive")
    {
        accepts("r_DS = 1e-3");
        rejects_value("r_DS = 0");
        rejects_value("r_DS = -2");
        rejects_value("r_DS = Inf");
    }

    SECTION("non-negative")
    {
        const char *keys[] = {"PerClusterAS_D", "PerClusterAS_A", "PerClusterES_D", "PerClusterES_A", "LNS_ksi",
                              "LOS_scatter_radius", "GR_enabled", "SC_lambda", "DS_lambda", "AS_D_lambda",
                              "AS_A_lambda", "ES_D_lambda", "ES_A_lambda", "SF_lambda", "KF_lambda", "XPR_lambda",
                              "PerClusterDS", "PerClusterDS_min", "absTOA_lambda",
                              "LSP_fc_min", "EoD_offset_b", "EoD_offset_hcl", "PerClusterDS_fc_min",
                              "PerClusterES_D_scale", "absTOA_Lmax", "SNS_sigma", "SNS_A", "SNS_B", "SNS_R",
                              "SNS_delta_var", "PolVar_sigma"};
        for (const char *k : keys)
        {
            accepts(std::string(k) + " = 0");
            accepts(std::string(k) + " = 2.5");
            rejects_value(std::string(k) + " = -0.001");
            rejects_value(std::string(k) + " = Inf"); // Inf is allowed for 'real' and unchecked keys only
        }
    }

    SECTION("real: any value, Inf included")
    {
        const char *keys[] = {"DS_mu", "DS_sigma", "DS_gamma", "DS_delta", "ES_D_mu_A", "ES_D_mu_min", "SF_sigma",
                              "KF_mu", "XPR_mu", "PerClusterDS_gamma", "absTOA_mu", "absTOA_sigma",
                              "ES_D_mu_B", "ES_D_mu_C", "ES_D_mu_D", "EoD_offset_e", "EoD_offset_e_gamma",
                              "EoD_offset_a", "EoD_offset_a_gamma", "EoD_offset_c", "EoD_offset_c_gamma",
                              "EoD_offset_c_hUT"};
        for (const char *k : keys)
        {
            accepts(std::string(k) + " = -123.5");
            accepts(std::string(k) + " = Inf");
            accepts(std::string(k) + " = -Inf");
        }
    }

    SECTION("unchecked legacy keys: any value, Inf included")
    {
        const char *keys[] = {"DS_omega", "AS_D_epsilon", "AS_A_zeta", "ES_D_alpha", "ES_A_kappa", "SF_tau",
                              "KF_beta", "XPR_omega", "GR_epsilon"};
        for (const char *k : keys)
        {
            accepts(std::string(k) + " = -123.5");
            accepts(std::string(k) + " = Inf");
        }
    }

    SECTION("correlation in [-1, 1], the 7 xpr_ keys included (Q13)")
    {
        for (int i = 0; i < 28; ++i)
        {
            accepts(std::string(XC[i].key) + " = 1");
            accepts(std::string(XC[i].key) + " = -1");
            rejects_value(std::string(XC[i].key) + " = 1.001");
            rejects_value(std::string(XC[i].key) + " = -1.001");
            rejects_value(std::string(XC[i].key) + " = -Inf");
        }
    }

    SECTION("in [0, 1]")
    {
        accepts("SNS_mu = 0");
        accepts("SNS_mu = 0.56");
        accepts("SNS_mu = 1");
        rejects_value("SNS_mu = -0.01");
        rejects_value("SNS_mu = 1.01");
        rejects_value("SNS_mu = Inf");
    }

    SECTION("integer and >= 0")
    {
        accepts("NumClusters_min = 0");
        accepts("NumClusters_min = 6"); // equal to the default of NumClusters
        accepts("NF_k1 = 4");
        rejects_value("NumClusters_min = 2.5");
        rejects_value("NumClusters_min = -1");
        rejects_value("NF_k1 = 0.5");
        rejects_value("NF_k1 = -2");
        rejects_value("NF_k1 = Inf");
    }

    SECTION("NumClusters_min must not exceed NumClusters")
    {
        accepts("NumClusters = 20\nNumClusters_min = 15");
        accepts("NumClusters_min = 15\nNumClusters = 20"); // the order of the lines does not matter
        accepts("NumClusters = 20\nNumClusters_min = 20");
        rejects_value("NumClusters = 20\nNumClusters_min = 21");
        rejects_value("NumClusters_min = 7"); // default NumClusters is 6
    }

    SECTION("NF_alpha and NF_beta are both 0 or both > 0")
    {
        accepts("NF_alpha = 1.93\nNF_beta = 1.33");
        rejects_value("NF_alpha = 1.93");
        rejects_value("NF_beta = 1.33");
    }

    SECTION("SubpathMethod is one of three names (Q14)")
    {
        accepts("SubpathMethod = legacy");
        accepts("SubpathMethod = Laplacian");
        accepts("SubpathMethod = mmMAGIC");
        rejects_value("SubpathMethod = Legacy");
        rejects_value("SubpathMethod = mmmagic");
        rejects_value("SubpathMethod = other");
        conf_out r = read_text("SubpathMethod = other\n", false);
        CHECK(r.subpath_method == "other");
    }
}

TEST_CASE("conf_file_read - input validation")
{
    SECTION("valid baseline")
    {
        accepts("DS_mu = -6.28 % comment\nPL_model = nlos\nPL_A = 1 2\nShortName = Ab\nFoo = bar");
    }

    SECTION("file cannot be opened")
    {
        std::unordered_map<std::string, double> sp;
        CHECK_THROWS_AS(quadriga_lib::conf_file_read("test_conf_file_read_missing.conf", sp), std::runtime_error);
    }

    SECTION("numbers outside the grammar (Q5)")
    {
        rejects_grammar("DS_mu = NaN");
        rejects_grammar("DS_mu = nan");
        rejects_grammar("DS_mu = 1+2i");
        rejects_grammar("DS_mu = 2j");
        rejects_grammar("DS_mu = 1e");
        rejects_grammar("DS_mu = 1e+");
        rejects_grammar("DS_mu = e3");
        rejects_grammar("DS_mu = --1");
        rejects_grammar("DS_mu = 1,5");
        rejects_grammar("DS_mu = 1.2.3");
        rejects_grammar("DS_mu = .");
        rejects_grammar("DS_mu = 0x10");
        rejects_grammar("DS_mu = Infinity");
        rejects_grammar("DS_mu = - 1");
        rejects_grammar("PL_A = 1e");
        rejects_grammar("PL_A = NaN");
        rejects_grammar("PL_A = 1 2 x");
        rejects_grammar("PL_A = 1,2");
    }

    SECTION("word for a numeric key, number for a string key (Q6)")
    {
        rejects_grammar("DS_mu = abc");
        rejects_grammar("NumClusters = legacy");
        rejects_grammar("SubpathMethod = 5");
        rejects_grammar("SubpathMethod = legacy2 x");
    }

    SECTION("more than one value, trailing text, missing value")
    {
        rejects_grammar("DS_mu = 1 2");
        rejects_grammar("GR_enabled = 1;"); // as in TwoRayGR.conf of QuaDRiGa-NG
        rejects_grammar("DS_mu = 1 abc");
        rejects_grammar("DS_mu =");
        rejects_grammar("DS_mu = % -6");
        rejects_grammar("SubpathMethod =");
        rejects_grammar("SubpathMethod = legacy mmMAGIC");
        rejects_grammar("ShortName =");
        rejects_grammar("ShortName = a b");
        rejects_grammar("ShortName = a-b");
    }

    SECTION("path-loss lines (Q7, Q8)")
    {
        rejects_grammar("PL_ = 1");
        rejects_grammar("PL_A =");
        rejects_grammar("PL_A = % 3");
        rejects_grammar("PL_model =");
        rejects_grammar("PL_model = % foo");
        rejects_grammar("PL_model = a b");
        rejects_grammar("PL_model = a-b");
    }

    SECTION("key side of the line")
    {
        rejects_grammar("DS mu = 1");
        rejects_grammar("DS-mu = 1");
        rejects_grammar("= 1");
        rejects_grammar("Foo bar = 1"); // also for unknown keys
    }

    SECTION("line without '=' that is neither blank nor a comment")
    {
        rejects_grammar("DS_mu -6.28");
        rejects_grammar("some text");
        rejects_grammar("DS_mu % = 3"); // the '=' is in the comment
    }

    SECTION("an error in a late line rejects the whole file")
    {
        rejects_grammar("DS_mu = -7\nDS_sigma = 0.3\n\nKF_mu = x");
    }
}

TEST_CASE("conf_file_read - error messages name the file and the line")
{
    auto message = [](const std::string &content) -> std::string
    {
        tmp_file f("test_conf_file_read_msg.conf", content);
        std::unordered_map<std::string, double> sp;
        try
        {
            quadriga_lib::conf_file_read(f.fn, sp);
        }
        catch (const std::invalid_argument &e)
        {
            return e.what();
        }
        return "";
    };

    QD_DIAG("\n# diag: conf_file_read error messages\n");

    std::string m = message("% line 1\nDS_sigma = 0.3\nDS_mu = abc\n"); // grammar error in line 3
    QD_DIAG("grammar: %s\n", m.c_str());
    CHECK(m.find("test_conf_file_read_msg.conf") != std::string::npos);
    CHECK(m.find("line 3") != std::string::npos);
    CHECK(m.find("'DS_mu'") != std::string::npos);

    m = message("\r\nDS_lambda = -1\r\n"); // value check fails for the key stated in line 2
    QD_DIAG("value:   %s\n", m.c_str());
    CHECK(m.find("test_conf_file_read_msg.conf") != std::string::npos);
    CHECK(m.find("line 2") != std::string::npos);
    CHECK(m.find("'DS_lambda'") != std::string::npos);

    m = message("DS_mu = 1\nDS_lambda = 5\nDS_lambda = -1\n"); // the last statement of a key counts
    QD_DIAG("dup:     %s\n", m.c_str());
    CHECK(m.find("line 3") != std::string::npos);
}

TEST_CASE("conf_file_read - a failed call leaves all outputs untouched")
{
    conf_out r;
    r.scenpar["marker"] = 42.0;
    r.subpath_method = "marker";
    r.pl_model = "marker";
    r.plpar["marker"] = {1.0, 2.0};
    r.lsp_xcorr.set_size(2, 3);
    r.lsp_xcorr.fill(7.0);
    r.key_order = {"marker"};
    r.pl_order = {"marker"};
    r.short_names = {"marker"};
    r.ignored_keys = {"marker"};

    auto call = [&r](const std::string &content)
    {
        tmp_file f("test_conf_file_read_untouched.conf", content);
        quadriga_lib::conf_file_read(f.fn, r.scenpar, &r.subpath_method, &r.pl_model, &r.plpar, &r.lsp_xcorr,
                                     &r.key_order, &r.pl_order, &r.short_names, &r.ignored_keys);
    };

    SECTION("grammar error in the last line")
    {
        CHECK_THROWS_AS(call("DS_mu = -7\nPL_model = nlos\nPL_A = 3\nShortName = Ab\nFoo = 1\nKF_mu = x\n"),
                        std::invalid_argument);
    }
    SECTION("value check")
    {
        CHECK_THROWS_AS(call("DS_mu = -7\nPL_model = nlos\nPL_A = 3\nShortName = Ab\nFoo = 1\nr_DS = 0\n"),
                        std::invalid_argument);
    }
    SECTION("missing file")
    {
        CHECK_THROWS_AS(quadriga_lib::conf_file_read("test_conf_file_read_missing.conf", r.scenpar, &r.subpath_method,
                                                     &r.pl_model, &r.plpar, &r.lsp_xcorr, &r.key_order, &r.pl_order,
                                                     &r.short_names, &r.ignored_keys),
                        std::runtime_error);
    }

    CHECK(r.scenpar.size() == 1);
    CHECK(r.scenpar.count("marker") == 1);
    CHECK(r.subpath_method == "marker");
    CHECK(r.pl_model == "marker");
    CHECK(r.plpar.size() == 1);
    CHECK((r.lsp_xcorr.n_rows == 2 && r.lsp_xcorr.n_cols == 3 && r.lsp_xcorr(1, 2) == 7.0));
    CHECK(r.key_order == std::vector<std::string>{"marker"});
    CHECK(r.pl_order == std::vector<std::string>{"marker"});
    CHECK(r.short_names == std::vector<std::string>{"marker"});
    CHECK(r.ignored_keys == std::vector<std::string>{"marker"});
}

TEST_CASE("conf_file_read - optional outputs alone and together, stale content is replaced")
{
    const std::string content = std::string("DS_mu = -7.1\nSubpathMethod = Laplacian\nPL_model = nlos\nPL_A = 3 4\n"
                                            "PL_B = 5\nShortName = Ab\nFoo = 1\n") + UMA_LOS_XCORR;
    tmp_file f("test_conf_file_read_optional.conf", content);

    conf_out all = read_all(f.fn);
    CHECK(all.ok);
    CHECK(same(val(all, "DS_mu"), -7.1));
    CHECK(all.subpath_method == "Laplacian");
    CHECK(all.pl_model == "nlos");
    CHECK(all.pl_order == std::vector<std::string>{"A", "B"});
    CHECK(all.short_names == std::vector<std::string>{"Ab"});
    CHECK(all.ignored_keys == std::vector<std::string>{"Foo"});
    REQUIRE((all.lsp_xcorr.n_rows == 8 && all.lsp_xcorr.n_cols == 8));
    CHECK(all.lsp_xcorr(0, 4) == 0.8);   // asA_ds: DS vs ASA
    CHECK(all.lsp_xcorr(6, 2) == -0.8);  // esA_sf: SF vs ESA
    CHECK(all.lsp_xcorr(1, 4) == -0.2);  // asA_kf: KF vs ASA

    // Mandatory output only; previous content of scenpar is discarded
    std::unordered_map<std::string, double> sp;
    sp["marker"] = 42.0;
    sp["DS_sigma"] = 99.0;
    CHECK(quadriga_lib::conf_file_read(f.fn, sp) == all.ok);
    CHECK(sp == all.scenpar);

    // Each optional output alone, starting from stale content
    std::string s = "stale";
    quadriga_lib::conf_file_read(f.fn, sp, &s);
    CHECK(s == all.subpath_method);

    s = "stale";
    quadriga_lib::conf_file_read(f.fn, sp, nullptr, &s);
    CHECK(s == all.pl_model);

    std::unordered_map<std::string, std::vector<double>> pl;
    pl["stale"] = {1.0};
    quadriga_lib::conf_file_read(f.fn, sp, nullptr, nullptr, &pl);
    CHECK(pl == all.plpar);

    std::vector<std::string> v = {"stale", "stale"};
    quadriga_lib::conf_file_read(f.fn, sp, nullptr, nullptr, nullptr, nullptr, &v);
    CHECK(v == all.key_order);

    v = {"stale", "stale", "stale"};
    quadriga_lib::conf_file_read(f.fn, sp, nullptr, nullptr, nullptr, nullptr, nullptr, &v);
    CHECK(v == all.pl_order);

    v = {"stale", "stale"};
    quadriga_lib::conf_file_read(f.fn, sp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &v);
    CHECK(v == all.short_names);

    v = {"stale", "stale"};
    quadriga_lib::conf_file_read(f.fn, sp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &v);
    CHECK(v == all.ignored_keys);

    // lsp_xcorr: a correctly sized output is written in place, a wrong size is resized
    arma::mat X(8, 8);
    X.fill(7.0);
    const double *mem = X.memptr();
    quadriga_lib::conf_file_read(f.fn, sp, nullptr, nullptr, nullptr, &X);
    CHECK(X.memptr() == mem);
    CHECK(arma::approx_equal(X, all.lsp_xcorr, "absdiff", 0.0));

    arma::mat Y(3, 5);
    Y.fill(7.0);
    quadriga_lib::conf_file_read(f.fn, sp, nullptr, nullptr, nullptr, &Y);
    REQUIRE((Y.n_rows == 8 && Y.n_cols == 8));
    CHECK(arma::approx_equal(Y, all.lsp_xcorr, "absdiff", 0.0));

    // Repeated calls give identical results
    conf_out again = read_all(f.fn);
    CHECK(again.scenpar == all.scenpar);
    CHECK(again.plpar == all.plpar);
    CHECK(again.ok == all.ok);
}
