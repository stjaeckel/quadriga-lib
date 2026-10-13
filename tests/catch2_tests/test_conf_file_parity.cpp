// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Parity of quadriga_lib::conf_file_read and conf_file_list with the MATLAB parser of QuaDRiGa-NG.
//
// Pins down: for every conf file shipped with QuaDRiGa-NG the C++ reader returns the same scenpar,
// plpar, lsp_xcorr and positive-definite flag as qd_builder does in MATLAB, and the lister returns the
// same entries as qd_builder.supported_scenarios(1). This is the acceptance test for "drop-in".
//
// Oracle: 'tests/data/conf_file_parity_ref.txt', written by 'tests/data/conf_file_parity_ref.m', which
// loads every scenario with qd_builder and prints all values with 17 significant digits. Values are
// compared exactly: MATLAB's str2double and the C++ reader both round a decimal number correctly.
//
// The case needs the QuaDRiGa-NG checkout in /sjc/QuaDRiGa-NG. It is skipped (passes without checks) if
// the config folder or the reference file is missing. The reference must be regenerated when a conf file
// in QuaDRiGa-NG changes.
//
// 'GR_epsilon' is in the MATLAB struct but is not a key of the C++ reader. The reference value must be 0 (the
// MATLAB default) for every file; the key is then left out of the comparison.
//
// Known exception: 'TwoRayGR.conf' has the line 'GR_enabled = 1;'. MATLAB reads 1, the C++ reader rejects
// the trailing ';' (clean grammar). Until the file is fixed, the case accepts that this one file throws
// and compares all others.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
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
    const char *PARITY_CONFIG_DIR = "/sjc/QuaDRiGa-NG/quadriga_src/config";
    const char *PARITY_REF_FILE = "tests/data/conf_file_parity_ref.txt";

    // One scenario block of the reference file
    struct ref_scen
    {
        std::string name, subpath_method, pl_model;
        bool chk = false, model_first = true;
        std::vector<std::string> sp_keys, pl_keys;
        std::vector<double> sp_vals, xc;
        std::vector<std::vector<double>> pl_vals;
    };
}

TEST_CASE("Diagnostic - conf_file_read parity with QuaDRiGa on the shipped conf files")
{
    if (!std::filesystem::is_directory(PARITY_CONFIG_DIR) || !std::filesystem::is_regular_file(PARITY_REF_FILE))
    {
        QD_DIAG("\n# diag: conf_file_read parity SKIPPED, QuaDRiGa-NG config folder or reference file not found\n");
        return;
    }

    // Read the reference
    std::vector<ref_scen> ref;
    std::vector<std::string> list_scen, list_file;
    {
        std::ifstream f(PARITY_REF_FILE);
        std::string line, tag, key, text;
        while (std::getline(f, line))
        {
            std::istringstream is(line);
            is >> tag;
            if (tag == "list")
            {
                is >> key >> text;
                list_scen.push_back(key);
                list_file.push_back(text);
            }
            else if (tag == "scenario")
            {
                ref.emplace_back();
                is >> ref.back().name;
            }
            else if (tag == "chk")
            {
                int c = 0;
                is >> c;
                ref.back().chk = c != 0;
            }
            else if (tag == "sp")
            {
                is >> key >> text;
                if (key == "GR_epsilon") // not supported by the C++ reader; no shipped file states it
                {
                    REQUIRE(std::strtod(text.c_str(), nullptr) == 0.0);
                    continue;
                }
                ref.back().sp_keys.push_back(key);
                ref.back().sp_vals.push_back(std::strtod(text.c_str(), nullptr));
            }
            else if (tag == "sps")
            {
                is >> key >> text;
                REQUIRE(key == "SubpathMethod");
                ref.back().subpath_method = text;
            }
            else if (tag == "plm")
            {
                is >> ref.back().pl_model;
                ref.back().model_first = ref.back().pl_keys.empty();
            }
            else if (tag == "pl")
            {
                size_t n = 0;
                is >> key >> n;
                std::vector<double> v;
                while (is >> text)
                    v.push_back(std::strtod(text.c_str(), nullptr));
                REQUIRE(v.size() == n);
                ref.back().pl_keys.push_back(key);
                ref.back().pl_vals.push_back(v);
            }
            else if (tag == "xc")
            {
                while (is >> text)
                    ref.back().xc.push_back(std::strtod(text.c_str(), nullptr));
            }
        }
    }
    REQUIRE(ref.size() >= 91);

    QD_DIAG("\n# diag: conf_file_read parity with QuaDRiGa, %d scenarios in the reference\n", (int)ref.size());
    QD_DIAG("scenario,n_scenpar_diff,n_plpar_diff,n_xcorr_diff,flag,ref_flag,n_ignored,note\n");

    int n_compared = 0, n_clean = 0;
    std::vector<std::string> threw;
    for (const ref_scen &r : ref)
    {
        INFO("scenario " << r.name);
        REQUIRE(r.sp_keys.size() == 138);
        REQUIRE(r.xc.size() == 64);

        std::unordered_map<std::string, double> sp;
        std::unordered_map<std::string, std::vector<double>> pl;
        std::string subpath_method, pl_model;
        arma::mat X;
        std::vector<std::string> key_order, pl_order, short_names, ignored;
        bool ok = false;
        try
        {
            ok = quadriga_lib::conf_file_read(std::string(PARITY_CONFIG_DIR) + "/" + r.name + ".conf", sp,
                                              &subpath_method, &pl_model, &pl, &X, &key_order, &pl_order,
                                              &short_names, &ignored);
        }
        catch (const std::exception &e)
        {
            threw.push_back(r.name);
            QD_DIAG("%s,,,,,,,THROWS: %s\n", r.name.c_str(), e.what());
            continue;
        }
        ++n_compared;

        // scenpar: the 138 legacy values, in the MATLAB field order, and 27 new keys at their default 0
        int n_sp = 0;
        REQUIRE(sp.size() == 165);
        REQUIRE(key_order.size() == 165);
        for (size_t i = 0; i < 138; ++i)
        {
            if (key_order[i] != r.sp_keys[i] || !(sp[key_order[i]] == r.sp_vals[i]))
            {
                ++n_sp;
                QD_DIAG("  %s: %s = %.17g, ref %s = %.17g\n", r.name.c_str(), key_order[i].c_str(),
                        sp[key_order[i]], r.sp_keys[i].c_str(), r.sp_vals[i]);
            }
        }
        for (size_t i = 138; i < 165; ++i)
            if (sp[key_order[i]] != 0.0)
                ++n_sp;
        if (subpath_method != r.subpath_method)
            ++n_sp;
        CHECK(n_sp == 0);

        // plpar: model, keys in file order, all values
        int n_pl = 0;
        if (pl_model != r.pl_model || pl_order != r.pl_keys || pl.size() != r.pl_keys.size() || !r.model_first)
            ++n_pl;
        for (size_t i = 0; i < r.pl_keys.size(); ++i)
            if (pl.count(r.pl_keys[i]) == 0 || pl[r.pl_keys[i]] != r.pl_vals[i])
            {
                ++n_pl;
                QD_DIAG("  %s: PL_%s differs\n", r.name.c_str(), r.pl_keys[i].c_str());
            }
        CHECK(n_pl == 0);

        // lsp_xcorr and the positive-definite flag
        int n_xc = 0;
        REQUIRE(X.n_elem == 64);
        for (size_t i = 0; i < 64; ++i)
            if (!(X(i) == r.xc[i]))
                ++n_xc;
        CHECK(n_xc == 0);
        CHECK(ok == r.chk);

        CHECK(ignored.empty());

        const bool clean = n_sp == 0 && n_pl == 0 && n_xc == 0 && ok == r.chk && ignored.empty();
        n_clean += clean ? 1 : 0;
        if (!clean)
            QD_DIAG("%s,%d,%d,%d,%d,%d,%d,DIFFERS\n", r.name.c_str(), n_sp, n_pl, n_xc, (int)ok, (int)r.chk,
                    (int)ignored.size());
    }
    QD_DIAG("summary: %d files compared, %d identical, %d threw\n", n_compared, n_clean, (int)threw.size());

    // Only the known exception may throw
    for (const std::string &name : threw)
        CHECK(name == "TwoRayGR");
    CHECK(n_compared + (int)threw.size() == (int)ref.size());

    // The lister returns the entries of supported_scenarios(1): same names, same files, same order
    std::vector<std::string> scen, files, dirs;
    arma::uword n = quadriga_lib::conf_file_list({PARITY_CONFIG_DIR}, scen, &files, &dirs);
    QD_DIAG("lister: %d entries, reference %d\n", (int)n, (int)list_scen.size());
    CHECK(scen == list_scen);
    CHECK(files == list_file);

    std::vector<std::string> scen0;
    CHECK(quadriga_lib::conf_file_list({PARITY_CONFIG_DIR}, scen0, nullptr, nullptr, false) == ref.size());
}
