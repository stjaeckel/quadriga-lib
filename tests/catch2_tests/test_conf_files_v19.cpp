// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for the 23 scenario files 'data/config/3GPP_38.901_v19_*.conf'.
//
// Pins down: all files are read by quadriga_lib::conf_file_read without ignored keys, pass the value
// checks and have a positive definite LSP correlation matrix; the keys for the EoD offset and for the
// ESD mean reproduce the formulas of 3GPP TR 38.901 v19.5.0; a set of table values that changed with
// Release 19 or that belong to the new components.
// Left to other suites: the grammar and the value checks (test_conf_file_read).
//
// Oracles: the formulas and numbers of TR 38.901 v19.5.0 (Tables 7.5-6 to 7.5-12, 7.6.9-1, 7.6-13-1,
// 7.6.14.1.2-1/-2, 7.6.15-1, 7.6.3.1-2), typed into this file from the TR, independently of the files.
//
// Conventions: QuaDRiGa uses elevation angles (+90 deg is up), the TR zenith angles. The EoD offset is
// the ZOD offset of the TR with the opposite sign; spreads are the same in both conventions.
// The key formulas are those of section 7.2 of md/conf_file_parser.md, written out in this file. No
// library code evaluates them yet.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <algorithm>
#include <cmath>
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
    typedef std::unordered_map<std::string, double> spmap;
    const double RAD2DEG = 180.0 / 3.14159265358979323846;

    const std::vector<std::string> &v19_names()
    {
        static const std::vector<std::string> n = {
            "InF_LOS", "InF_NLOS_DH", "InF_NLOS_DL", "InF_NLOS_SH", "InF_NLOS_SL", "Indoor_LOS", "Indoor_NLOS",
            "RMa_LOS", "RMa_LOS_O2I", "RMa_NLOS", "RMa_NLOS_O2I", "SMa_LOS", "SMa_LOS_O2I", "SMa_NLOS", "SMa_NLOS_O2I",
            "UMa_LOS", "UMa_LOS_O2I", "UMa_NLOS", "UMa_NLOS_O2I", "UMi_LOS", "UMi_LOS_O2I", "UMi_NLOS", "UMi_NLOS_O2I"};
        return n;
    }

    spmap read_v19(const std::string &name, std::unordered_map<std::string, std::vector<double>> *pl = nullptr,
                   std::string *pl_model = nullptr)
    {
        spmap sp;
        quadriga_lib::conf_file_read("data/config/3GPP_38.901_v19_" + name + ".conf", sp, nullptr, pl_model, pl);
        return sp;
    }

    // Key formula of the plan: EoD offset in degrees, elevation, positive = upwards
    double eod_offset_keys(spmap &p, double fc, double d2D, double hBS, double hUT)
    {
        const double lf = std::log10(std::max(fc, p["LSP_fc_min"]));
        double off = p["EoD_offset_e"] + p["EoD_offset_e_gamma"] * lf;
        if (p["EoD_offset_b"] > 0.0)
            off += std::pow(10.0, (p["EoD_offset_a"] + p["EoD_offset_a_gamma"] * lf) * std::log10(std::max(p["EoD_offset_b"], d2D)) +
                                      p["EoD_offset_c"] + p["EoD_offset_c_gamma"] * lf + p["EoD_offset_c_hUT"] * hUT);
        const double hcl = p["EoD_offset_hcl"];
        if (hcl > 0.0 && hUT < hcl)
            off += RAD2DEG * (std::atan((hcl - hBS) / d2D) - std::atan((hUT - hBS) / d2D));
        return off;
    }

    // Key formula of the plan: mean of log10(ESD / 1 deg), without the legacy omega, epsilon, zeta, alpha terms
    double esd_mean_keys(spmap &p, double fc, double d2D, double hBS, double hUT)
    {
        const double lf = std::log10(p["ES_D_omega"] + std::max(fc, p["LSP_fc_min"]));
        return std::max(p["ES_D_mu_min"], p["ES_D_mu"] + p["ES_D_gamma"] * lf + p["ES_D_mu_A"] * d2D / 1000.0 +
                                              p["ES_D_mu_B"] * hUT + p["ES_D_mu_C"] * std::abs(hUT - hBS) +
                                              p["ES_D_mu_D"] * std::max(hUT - hBS, 0.0));
    }

    // TR 38.901 v19.5.0, Tables 7.5-7, 7.5-8, 7.5-9, 7.5-12: mu_offset,ZOD in degrees (zenith convention)
    // 'state' is LOS, NLOS or O2I (the column of the table)
    double tr_zod_offset(const std::string &scen, const std::string &state, double fc, double d2D, double hBS, double hUT)
    {
        if (scen == "UMa" && state == "NLOS")
        {
            const double lf = std::log10(std::max(fc, 6.0)); // note 4
            const double a = 0.208 * lf - 0.782, b = 25.0, c = -0.13 * lf + 2.03, e = 7.66 * lf - 5.96;
            return e - std::pow(10.0, a * std::log10(std::max(b, d2D)) + c - 0.07 * (hUT - 1.5));
        }
        if (scen == "UMi" && state == "NLOS")
            return -std::pow(10.0, -1.5 * std::log10(std::max(10.0, d2D)) + 3.3);
        if (scen == "RMa" && state != "LOS") // literal heights of Table 7.5-9; the caller uses hBS = 35, hUT = 1.5
            return RAD2DEG * (std::atan((35.0 - 3.5) / d2D) - std::atan((35.0 - 1.5) / d2D));
        if (scen == "SMa" && state != "LOS")
            return hUT < 3.5 ? RAD2DEG * (std::atan((hBS - 3.5) / d2D) - std::atan((hBS - hUT) / d2D)) : 0.0;
        return 0.0;
    }

    // TR 38.901 v19.5.0, Tables 7.5-7 to 7.5-12: mu_lgZSD
    double tr_zsd_mean(const std::string &scen, const std::string &state, double fc, double d2D, double hBS, double hUT)
    {
        const bool los = state == "LOS";
        if (scen == "UMa")
            return std::max(-0.5, -2.1 * (d2D / 1000.0) - 0.01 * (hUT - 1.5) + (los ? 0.75 : 0.9));
        if (scen == "UMi")
            return los ? std::max(-0.21, -14.8 * (d2D / 1000.0) + 0.01 * std::abs(hUT - hBS) + 0.83)
                       : std::max(-0.5, -3.1 * (d2D / 1000.0) + 0.01 * std::max(hUT - hBS, 0.0) + 0.2);
        if (scen == "RMa")
            return los ? std::max(-1.0, -0.17 * (d2D / 1000.0) - 0.01 * (hUT - 1.5) + 0.22)
                       : std::max(-1.0, -0.19 * (d2D / 1000.0) - 0.01 * (hUT - 1.5) + 0.28);
        if (scen == "Indoor")
            return los ? -1.43 * std::log10(1.0 + std::max(fc, 6.0)) + 2.228 : 1.08;
        if (scen == "InF")
            return los ? 1.35 : 1.2;
        return 0.14; // SMa
    }

    // Split a file name into scenario and the table column that the ESD and EoD offset rows follow:
    // UMa and UMi O2I links use the row of the outdoor state (note 5), RMa and SMa have an O2I column
    void split_name(const std::string &name, std::string &scen, std::string &state, bool &los_file, bool &o2i)
    {
        scen = name.substr(0, name.find('_'));
        o2i = name.find("O2I") != std::string::npos;
        const bool outdoor_los = name.find("NLOS") == std::string::npos;
        los_file = outdoor_los && !o2i;
        state = outdoor_los ? "LOS" : "NLOS";
        if (o2i && (scen == "RMa" || scen == "SMa"))
            state = "O2I";
    }
}

TEST_CASE("conf files v19 - all 23 files are read without ignored keys and pass the value checks")
{
    // The folder holds exactly the 23 files
    std::vector<std::string> scen;
    arma::uword n = quadriga_lib::conf_file_list({"data/config"}, scen);
    CHECK(n == 23);
    REQUIRE(scen.size() == v19_names().size());
    for (size_t i = 0; i < scen.size(); ++i)
        CHECK(scen[i] == "3GPP_38.901_v19_" + v19_names()[i]);

    QD_DIAG("\n# diag: conf files v19, read result\n");
    QD_DIAG("file,positive_definite,n_ignored,n_plpar,pl_model,NumClusters,n_keys_not_default\n");
    spmap def;
    quadriga_lib::scenpar_validate(def);
    for (const std::string &name : v19_names())
    {
        INFO("file " << name);
        spmap sp;
        std::string subpath_method, pl_model;
        std::unordered_map<std::string, std::vector<double>> pl;
        std::vector<std::string> short_names, ignored;
        bool ok = false;
        REQUIRE_NOTHROW(ok = quadriga_lib::conf_file_read("data/config/3GPP_38.901_v19_" + name + ".conf", sp,
                                                          &subpath_method, &pl_model, &pl, nullptr, nullptr, nullptr,
                                                          &short_names, &ignored));
        int n_changed = 0;
        for (const auto &e : sp)
            n_changed += (e.second != def[e.first]) ? 1 : 0;
        QD_DIAG("%s,%d,%d,%d,%s,%g,%d\n", name.c_str(), (int)ok, (int)ignored.size(), (int)pl.size(), pl_model.c_str(),
                sp["NumClusters"], n_changed);

        CHECK(ok); // positive definite
        CHECK(ignored.empty());
        CHECK(short_names.empty());
        CHECK(subpath_method == "legacy");
        CHECK(!pl_model.empty());
        CHECK(!pl.empty());

        // Common to all v19 files
        CHECK(sp["PerClusterES_D_scale"] == 0.375); // eq. (7.6-7)
        CHECK(sp["PerClusterES_D"] > 0.0);
        CHECK(sp["PerClusterDS_fc_min"] == 0.0); // the TR has no clamp for the cluster DS
        CHECK(sp["PolVar_sigma"] == 3.0);        // clause 7.6.16
        CHECK(sp["NumSubPaths"] == 20.0);
        CHECK(sp["absTOA_Lmax"] == 0.0);

        // LOS files have a K-factor, all others switch the LOS path off and carry one extra cluster
        std::string s, state;
        bool los_file, o2i;
        split_name(name, s, state, los_file, o2i);
        if (los_file)
            CHECK(sp["KF_mu"] > 0.0);
        else
            CHECK(sp["KF_mu"] == -100.0);

        // Absolute time of arrival only in NLOS and NLOS_O2I files (decision O10); active if absTOA_mu > -30
        const bool nlos_outdoor = name.find("NLOS") != std::string::npos;
        CHECK((sp["absTOA_mu"] > -30.0) == nlos_outdoor);
    }
}

TEST_CASE("conf files v19 - the EoD offset keys give minus the ZOD offset of the TR")
{
    // hBS: scenario default of Table 7.4.1-1; SMa has a range of 25 to 35 m, 30 m is used
    QD_DIAG("\n# diag: conf files v19, EoD offset [deg] from the keys vs. -mu_offset,ZOD of TR 38.901 Tables 7.5-7 to 7.5-12\n");
    QD_DIAG("file,fc_GHz,d2D_m,hBS_m,hUT_m,eod_keys,minus_zod_tr,diff\n");
    int n_nonzero = 0;
    for (const std::string &name : v19_names())
    {
        std::string scen, state;
        bool los_file, o2i;
        split_name(name, scen, state, los_file, o2i);
        spmap p = read_v19(name);

        const double hBS = scen == "UMa" ? 25.0 : scen == "UMi" ? 10.0 : scen == "RMa" ? 35.0 : scen == "SMa" ? 30.0 : 3.0;
        const double hUTs[] = {1.5, 3.0, 10.0};
        for (double fc : {3.5, 6.0, 28.0})
            for (double d2D : {15.0, 100.0, 1000.0})
                for (double hUT : hUTs)
                {
                    if (scen == "RMa" && hUT != 1.5) // the TR formula for RMa is written for hUT = 1.5 m only
                        continue;
                    const double key = eod_offset_keys(p, fc, d2D, hBS, hUT);
                    const double ref = -tr_zod_offset(scen, state, fc, d2D, hBS, hUT);
                    INFO("file " << name << ", fc " << fc << ", d2D " << d2D << ", hUT " << hUT);
                    CHECK(std::abs(key - ref) < 1e-10); // folded constants differ by rounding only
                    n_nonzero += ref != 0.0 ? 1 : 0;
                    if (ref != 0.0 && d2D == 100.0 && fc != 6.0 && hUT == 1.5)
                        QD_DIAG("%s,%.1f,%.0f,%.0f,%.1f,%.6f,%.6f,%.2e\n", name.c_str(), fc, d2D, hBS, hUT, key, ref, key - ref);
                }

        // Files without an offset in the TR state no offset key
        if (tr_zod_offset(scen, state, 28.0, 100.0, hBS, 1.5) == 0.0)
            for (const char *k : {"EoD_offset_e", "EoD_offset_e_gamma", "EoD_offset_a", "EoD_offset_a_gamma", "EoD_offset_b",
                                  "EoD_offset_c", "EoD_offset_c_gamma", "EoD_offset_c_hUT", "EoD_offset_hcl"})
            {
                INFO("file " << name << ", key " << k);
                CHECK(p[k] == 0.0);
            }
    }
    CHECK(n_nonzero > 100);

    // Examples of the plan, section 7.2
    spmap p = read_v19("UMi_NLOS");
    CHECK(std::abs(eod_offset_keys(p, 3.5, 100.0, 10.0, 1.5) - 1.9953) < 1e-4); // +2.0 deg at 100 m
    p = read_v19("RMa_NLOS");
    CHECK(std::abs(eod_offset_keys(p, 0.7, 500.0, 35.0, 1.5) - 0.23) < 0.005);  // +0.23 deg at 500 m
    p = read_v19("UMa_NLOS");
    CHECK(std::abs(eod_offset_keys(p, 28.0, 1000.0, 25.0, 1.5) + 2.62) < 0.01); // about -2.6 deg
    CHECK(eod_offset_keys(p, 6.0, 100.0, 25.0, 1.5) > 0.0);                     // positive = upwards at short range
    CHECK(eod_offset_keys(p, 3.5, 100.0, 25.0, 1.5) == eod_offset_keys(p, 6.0, 100.0, 25.0, 1.5)); // clamp at 6 GHz
}

TEST_CASE("conf files v19 - the ESD keys give the ZSD mean of the TR")
{
    QD_DIAG("\n# diag: conf files v19, ESD mean [log10(deg)] from the keys vs. mu_lgZSD of TR 38.901 Tables 7.5-7 to 7.5-12\n");
    QD_DIAG("file,fc_GHz,d2D_m,hBS_m,hUT_m,esd_keys,zsd_tr,diff\n");
    for (const std::string &name : v19_names())
    {
        std::string scen, state;
        bool los_file, o2i;
        split_name(name, scen, state, los_file, o2i);
        spmap p = read_v19(name);

        const double hBSs[] = {10.0, 25.0};
        const double hUTs[] = {1.5, 12.0, 22.5};
        for (double fc : {3.5, 28.0})
            for (double d2D : {10.0, 200.0, 5000.0})
                for (double hBS : hBSs)
                    for (double hUT : hUTs)
                    {
                        const double key = esd_mean_keys(p, fc, d2D, hBS, hUT);
                        const double ref = tr_zsd_mean(scen, state, fc, d2D, hBS, hUT);
                        INFO("file " << name << ", fc " << fc << ", d2D " << d2D << ", hBS " << hBS << ", hUT " << hUT);
                        CHECK(std::abs(key - ref) < 1e-12);
                        if (d2D == 200.0 && fc == 3.5 && hBS == 10.0 && hUT == 12.0)
                            QD_DIAG("%s,%.1f,%.0f,%.0f,%.1f,%.6f,%.6f,%.2e\n", name.c_str(), fc, d2D, hBS, hUT, key, ref, key - ref);
                    }

        // PerClusterES_D is 3/8 * 10^(constant of the ESD mean); Indoor LOS: mean at 6 GHz
        const double c = scen == "UMa" ? (state == "LOS" ? 0.75 : 0.9) : scen == "UMi" ? (state == "LOS" ? 0.83 : 0.2)
                       : scen == "RMa" ? (state == "LOS" ? 0.22 : 0.28) : tr_zsd_mean(scen, state, 6.0, 0.0, 0.0, 0.0);
        INFO("file " << name);
        CHECK(std::abs(p["PerClusterES_D"] - 0.375 * std::pow(10.0, c)) < 0.5e-3); // 3 decimals in the file
    }
}

TEST_CASE("conf files v19 - table values of TR 38.901 v19.5.0")
{
    struct kv
    {
        const char *file, *key;
        double ref;
    };
    // Typed from the TR. Frequency terms: X_mu + X_gamma * log10(X_omega + fc), X_sigma + X_delta * log10(X_omega + fc)
    const kv expect[] = {
        // Table 7.5-6 Part-1, values updated in Release 19 (note 8)
        {"UMi_LOS", "DS_mu", -7.28}, {"UMi_LOS", "DS_gamma", -0.18}, {"UMi_LOS", "DS_omega", 1}, {"UMi_LOS", "DS_sigma", 0.39},
        {"UMi_LOS", "AS_D_sigma", 0.29}, {"UMi_LOS", "AS_D_delta", 0.08}, {"UMi_LOS", "AS_A_mu", 1.66}, {"UMi_LOS", "AS_A_gamma", -0.07},
        {"UMi_LOS", "AS_A_sigma", 0.26}, {"UMi_LOS", "AS_A_delta", 0.021}, {"UMi_LOS", "ES_A_mu", 0.81}, {"UMi_LOS", "ES_A_gamma", -0.11},
        {"UMi_LOS", "ES_A_sigma", 0.29}, {"UMi_LOS", "ES_A_delta", -0.03},
        {"UMi_NLOS", "DS_mu", -6.87}, {"UMi_NLOS", "DS_gamma", -0.22}, {"UMi_NLOS", "DS_sigma", 0.22}, {"UMi_NLOS", "DS_delta", 0.19},
        {"UMi_NLOS", "AS_D_mu", 1.54}, {"UMi_NLOS", "AS_D_gamma", -0.24}, {"UMi_NLOS", "AS_D_sigma", 0.33}, {"UMi_NLOS", "AS_D_delta", 0.10},
        {"UMi_NLOS", "AS_A_mu", 1.76}, {"UMi_NLOS", "AS_A_gamma", -0.07}, {"UMi_NLOS", "AS_A_sigma", 0.27}, {"UMi_NLOS", "AS_A_delta", 0.05},
        {"UMi_NLOS", "ES_A_mu", 0.92}, {"UMi_NLOS", "ES_A_gamma", -0.03}, {"UMi_NLOS", "ES_A_sigma", 0.35}, {"UMi_NLOS", "ES_A_delta", -0.05},
        {"UMa_LOS", "DS_mu", -7.067}, {"UMa_LOS", "DS_gamma", -0.0794}, {"UMa_LOS", "DS_sigma", 0.57}, {"UMa_LOS", "DS_delta", 0.026},
        {"UMa_LOS", "DS_omega", 0}, {"UMa_LOS", "AS_D_mu", 0.92}, {"UMa_LOS", "AS_D_gamma", 0}, {"UMa_LOS", "AS_D_sigma", 0.31},
        {"UMa_LOS", "AS_A_mu", 1.76}, {"UMa_LOS", "AS_A_sigma", 0.19}, {"UMa_LOS", "ES_A_mu", 0.96}, {"UMa_LOS", "ES_A_sigma", 0.15},
        {"UMa_LOS", "PerClusterAS_D", 3.58},
        {"UMa_NLOS", "DS_mu", -6.47}, {"UMa_NLOS", "DS_gamma", -0.134}, {"UMa_NLOS", "AS_D_mu", 1.09}, {"UMa_NLOS", "AS_D_gamma", 0},
        {"UMa_NLOS", "AS_D_sigma", 0.44}, {"UMa_NLOS", "AS_A_mu", 2.04}, {"UMa_NLOS", "AS_A_gamma", -0.25}, {"UMa_NLOS", "AS_A_sigma", 0.17},
        {"UMa_NLOS", "AS_A_delta", -0.03}, {"UMa_NLOS", "ES_A_mu", 1.445}, {"UMa_NLOS", "ES_A_gamma", -0.2856}, {"UMa_NLOS", "ES_A_sigma", 0.17},
        {"UMa_NLOS", "PerClusterAS_D", 1.8},
        {"UMa_LOS_O2I", "AS_D_mu", 0.58}, {"UMa_LOS_O2I", "AS_D_sigma", 0.7}, {"UMa_LOS_O2I", "PerClusterAS_D", 1.8},
        {"UMa_NLOS_O2I", "AS_D_mu", 0.58}, {"UMa_NLOS_O2I", "AS_D_sigma", 0.7}, {"UMa_NLOS_O2I", "PerClusterAS_D", 1.8},
        // Lower frequency limit, Table 7.5-6 notes 6 and 7, ZSD tables note 4
        {"UMa_LOS", "LSP_fc_min", 6}, {"UMa_NLOS_O2I", "LSP_fc_min", 6}, {"UMi_LOS", "LSP_fc_min", 2}, {"UMi_NLOS", "LSP_fc_min", 2},
        {"Indoor_LOS", "LSP_fc_min", 6}, {"Indoor_NLOS", "LSP_fc_min", 6}, {"RMa_NLOS", "LSP_fc_min", 0}, {"SMa_LOS", "LSP_fc_min", 0},
        {"InF_LOS", "LSP_fc_min", 0},
        // Table 7.5-6 Part-4, SMa
        {"SMa_LOS", "DS_mu", -7.42}, {"SMa_LOS", "DS_sigma", 0.60}, {"SMa_LOS", "KF_mu", 9}, {"SMa_LOS", "KF_sigma", 7},
        {"SMa_LOS", "NumClusters", 15}, {"SMa_LOS", "r_DS", 2.4}, {"SMa_LOS", "PerClusterAS_D", 2.08}, {"SMa_LOS", "esA_sf", -0.8},
        {"SMa_NLOS", "DS_mu", -7.20}, {"SMa_NLOS", "AS_A_mu", 1.63}, {"SMa_NLOS", "NumClusters", 15}, {"SMa_NLOS", "r_DS", 1.5},
        {"SMa_NLOS", "XPR_mu", 4}, {"SMa_NLOS", "ds_sf", -0.13}, {"SMa_NLOS", "esA_sf", -0.47}, {"SMa_NLOS", "esA_asA", 0.36},
        {"SMa_NLOS_O2I", "SF_sigma", 8}, {"SMa_NLOS", "ES_D_mu", 0.14}, {"SMa_NLOS", "ES_D_sigma", 0.16},
        {"SMa_NLOS", "EoD_offset_hcl", 3.5}, {"SMa_LOS_O2I", "EoD_offset_hcl", 3.5}, {"RMa_NLOS", "EoD_offset_hcl", 3.5},
        // Number of clusters: N for LOS, N + 1 otherwise; lower end of the range D1 likewise (Table 7.6.15-1)
        {"UMi_LOS", "NumClusters", 12}, {"UMi_NLOS", "NumClusters", 20}, {"UMi_NLOS_O2I", "NumClusters", 13},
        {"UMa_NLOS", "NumClusters", 21}, {"RMa_LOS", "NumClusters", 11}, {"RMa_NLOS", "NumClusters", 11},
        {"Indoor_LOS", "NumClusters", 15}, {"Indoor_NLOS", "NumClusters", 20}, {"InF_LOS", "NumClusters", 25},
        {"InF_NLOS_DL", "NumClusters", 26},
        {"UMi_LOS", "NumClusters_min", 6}, {"UMi_NLOS", "NumClusters_min", 7}, {"UMi_LOS_O2I", "NumClusters_min", 7},
        {"UMa_LOS", "NumClusters_min", 10}, {"UMa_NLOS", "NumClusters_min", 16}, {"UMa_NLOS_O2I", "NumClusters_min", 11},
        {"Indoor_LOS", "NumClusters_min", 7}, {"Indoor_NLOS", "NumClusters_min", 7}, {"RMa_LOS", "NumClusters_min", 0},
        {"SMa_NLOS", "NumClusters_min", 0}, {"InF_LOS", "NumClusters_min", 0},
        // Table 7.6.9-1, absolute time of arrival
        {"Indoor_NLOS", "absTOA_mu", -8.6}, {"Indoor_NLOS", "absTOA_sigma", 0.1}, {"Indoor_NLOS", "absTOA_lambda", 10},
        {"UMi_NLOS", "absTOA_mu", -7.5}, {"UMi_NLOS", "absTOA_sigma", 0.5}, {"UMi_NLOS_O2I", "absTOA_lambda", 15},
        {"UMa_NLOS", "absTOA_mu", -7.4}, {"UMa_NLOS", "absTOA_sigma", 0.2}, {"UMa_NLOS_O2I", "absTOA_lambda", 50},
        {"RMa_NLOS", "absTOA_mu", -8.33}, {"RMa_NLOS", "absTOA_sigma", 0.26}, {"RMa_NLOS_O2I", "absTOA_lambda", 50},
        {"SMa_NLOS", "absTOA_mu", -7.702}, {"SMa_NLOS", "absTOA_sigma", 0.4}, {"SMa_NLOS_O2I", "absTOA_lambda", 50},
        {"InF_NLOS_SL", "absTOA_mu", -7.5}, {"InF_NLOS_SL", "absTOA_sigma", 0.4}, {"InF_NLOS_SL", "absTOA_lambda", 6},
        {"InF_NLOS_DL", "absTOA_lambda", 6}, {"InF_NLOS_SH", "absTOA_lambda", 11}, {"InF_NLOS_DH", "absTOA_lambda", 11},
        {"UMa_LOS", "absTOA_mu", -100}, {"UMa_LOS_O2I", "absTOA_mu", -100}, {"InF_LOS", "absTOA_mu", -100},
        // Table 7.6-13-1, near field
        {"UMa_LOS", "NF_k1", 2}, {"UMa_NLOS", "NF_alpha", 1.93}, {"UMa_LOS_O2I", "NF_beta", 1.33},
        {"UMi_NLOS", "NF_k1", 2}, {"UMi_LOS", "NF_alpha", 1.53}, {"UMi_NLOS_O2I", "NF_beta", 1.42},
        {"Indoor_LOS", "NF_k1", 4}, {"Indoor_NLOS", "NF_alpha", 1.25}, {"Indoor_LOS", "NF_beta", 1.27},
        {"InF_NLOS_DH", "NF_k1", 4}, {"InF_LOS", "NF_alpha", 1.38}, {"InF_NLOS_SL", "NF_beta", 1.26},
        {"RMa_LOS", "NF_alpha", 0}, {"SMa_NLOS", "NF_beta", 0}, {"SMa_LOS", "NF_k1", 0},
        // Tables 7.6.14.1.2-1 and 7.6.14.1.2-2, spatial non-stationarity
        {"RMa_LOS", "SNS_mu", 0.14}, {"RMa_NLOS", "SNS_sigma", 0.08}, {"RMa_LOS_O2I", "SNS_A", 0.16}, {"RMa_LOS", "SNS_B", 0.74},
        {"RMa_NLOS", "SNS_R", 60}, {"RMa_LOS", "SNS_delta_var", 0.0016},
        {"SMa_LOS", "SNS_mu", 0.24}, {"SMa_NLOS", "SNS_sigma", 0.07}, {"SMa_LOS", "SNS_A", 0.06}, {"SMa_NLOS_O2I", "SNS_B", 0.56},
        {"SMa_LOS", "SNS_R", 23}, {"SMa_NLOS", "SNS_delta_var", 0.0013},
        {"UMa_LOS", "SNS_mu", 0.56}, {"UMa_NLOS", "SNS_sigma", 0.20}, {"UMa_LOS", "SNS_A", 0.15}, {"UMa_NLOS", "SNS_B", 0.45},
        {"UMa_LOS_O2I", "SNS_R", 33}, {"UMa_NLOS", "SNS_delta_var", 0.0015},
        {"UMi_LOS", "SNS_mu", 0.49}, {"UMi_NLOS", "SNS_sigma", 0.18}, {"UMi_LOS", "SNS_A", 0.12}, {"UMi_NLOS", "SNS_B", 0.48},
        {"UMi_LOS", "SNS_R", 50}, {"UMi_NLOS_O2I", "SNS_delta_var", 0.001},
        {"Indoor_LOS", "SNS_mu", 0.31}, {"Indoor_NLOS", "SNS_sigma", 0.08}, {"Indoor_LOS", "SNS_A", 0}, {"Indoor_NLOS", "SNS_B", 0.6},
        {"Indoor_LOS", "SNS_R", 0}, {"Indoor_NLOS", "SNS_delta_var", 0.0011},
        {"InF_LOS", "SNS_mu", 0.32}, {"InF_NLOS_DL", "SNS_sigma", 0.06}, {"InF_NLOS_SH", "SNS_A", 0}, {"InF_LOS", "SNS_B", 0.57},
        {"InF_NLOS_DH", "SNS_R", 0}, {"InF_LOS", "SNS_delta_var", 0.002},
        // Table 7.6.3.1-2, spatial consistency
        {"RMa_LOS", "SC_lambda", 50}, {"RMa_NLOS", "SC_lambda", 60}, {"RMa_LOS_O2I", "SC_lambda", 15},
        {"UMi_LOS", "SC_lambda", 12}, {"UMi_NLOS", "SC_lambda", 15}, {"UMi_NLOS_O2I", "SC_lambda", 15},
        {"UMa_LOS", "SC_lambda", 40}, {"UMa_NLOS", "SC_lambda", 50}, {"UMa_LOS_O2I", "SC_lambda", 15},
        {"SMa_LOS", "SC_lambda", 40}, {"SMa_NLOS", "SC_lambda", 50}, {"SMa_NLOS_O2I", "SC_lambda", 15},
        {"Indoor_LOS", "SC_lambda", 10}, {"Indoor_NLOS", "SC_lambda", 10}, {"InF_LOS", "SC_lambda", 10}, {"InF_NLOS_SL", "SC_lambda", 10},
        // Shadow fading, Table 7.4.1-1 (InF LOS is 4.3 dB) and Table 7.5-6 (O2I)
        {"InF_LOS", "SF_sigma", 4.3}, {"InF_NLOS_SL", "SF_sigma", 5.7}, {"InF_NLOS_DL", "SF_sigma", 7.2}, {"InF_NLOS_SH", "SF_sigma", 5.9},
        {"InF_NLOS_DH", "SF_sigma", 4.0}, {"UMi_NLOS", "SF_sigma", 7.82}, {"UMa_NLOS", "SF_sigma", 6}, {"UMa_LOS_O2I", "SF_sigma", 7},
        {"RMa_NLOS_O2I", "SF_sigma", 8}, {"Indoor_NLOS", "SF_sigma", 8.03}, {"Indoor_LOS", "SF_sigma", 3}};

    int n = 0;
    std::string last;
    spmap p;
    for (const kv &e : expect)
    {
        if (last != e.file)
            p = read_v19(e.file), last = e.file;
        INFO("file " << e.file << ", key " << e.key);
        REQUIRE(p.count(e.key) == 1);
        CHECK(p[e.key] == e.ref);
        ++n;
    }
    QD_DIAG("\n# diag: conf files v19, %d table values compared\n", n);
}

TEST_CASE("conf files v19 - path-loss constants of RMa and SMa follow Table 7.4.1-1")
{
    // Oracle: the TR formula evaluated at the building height h and street width W that the file states in its
    // comments (RMa: h = 5 m, W = 20 m; SMa: h = 10 m, W = 10 m). Constants are rounded to 3 decimals in the files.
    //   LOS:  A1 = 20 + min(0.03 h^1.72, 10);  B = 20 log10(40 pi / 3) - min(0.044 h^1.72, 14.77);  D = 0.002 log10(h)
    //         break point dBP = 2 pi hBS hUT fc / c: E = 2 pi / 0.3 with fc in GHz, hE = 0 (note 5)
    //   NLOS: Bn = 161.04 - 7.1 log10(W) + 7.5 log10(h) - 3 * 43.42 + 4.97;  Dn = -24.37 + 3 * 3.1;  D2n = 3.7 h^2
    const double pi = 3.14159265358979323846;
    QD_DIAG("\n# diag: conf files v19, derived path-loss constants\n");
    QD_DIAG("file,key,value,ref\n");
    for (const char *scen : {"RMa", "SMa"})
    {
        const double h = std::string(scen) == "RMa" ? 5.0 : 10.0, W = std::string(scen) == "RMa" ? 20.0 : 10.0;
        const double A1 = 20.0 + std::min(0.03 * std::pow(h, 1.72), 10.0);
        const double B = 20.0 * std::log10(40.0 * pi / 3.0) - std::min(0.044 * std::pow(h, 1.72), 14.77);
        const double D = 0.002 * std::log10(h), E = 2.0 * pi / 0.3;
        const double Bn = 161.04 - 7.1 * std::log10(W) + 7.5 * std::log10(h) - 3.0 * 43.42 + 4.97;

        std::unordered_map<std::string, std::vector<double>> pl;
        std::string model;
        read_v19(std::string(scen) + "_LOS", &pl, &model);
        INFO("scenario " << scen);
        CHECK(model == "dual_slope");
        REQUIRE((pl.count("A1") && pl.count("A2") && pl.count("B") && pl.count("C") && pl.count("D") && pl.count("E") && pl.count("hE")));
        QD_DIAG("%s_LOS,A1,%g,%.5f\n%s_LOS,B,%g,%.5f\n%s_LOS,D,%g,%.6f\n%s_LOS,E,%g,%.5f\n", scen, pl["A1"][0], A1, scen,
                pl["B"][0], B, scen, pl["D"][0], D, scen, pl["E"][0], E);
        CHECK(std::abs(pl["A1"][0] - A1) < 0.5e-3);
        CHECK(pl["A2"][0] == 40.0);
        CHECK(std::abs(pl["B"][0] - B) < 0.5e-3);
        CHECK(pl["C"][0] == 20.0);
        CHECK(std::abs(pl["D"][0] - D) < 0.5e-4);
        CHECK(std::abs(pl["E"][0] - E) < 0.5e-3);
        CHECK(pl["hE"][0] == 0.0);
        CHECK(pl["sig1"][0] == 4.0);
        CHECK(pl["sig2"][0] == 6.0);

        read_v19(std::string(scen) + "_NLOS", &pl, &model);
        CHECK(model == "nlos");
        REQUIRE((pl.count("An") && pl.count("Bn") && pl.count("Cn") && pl.count("Dn") && pl.count("D2n") && pl.count("Fn") &&
                 pl.count("G1n") && pl.count("G2n") && pl.count("A1") && pl.count("B")));
        CHECK((pl.count("E") == 1 && pl.count("hE") == 1)); // needed by the QuaDRiGa model also if the LOS term is off
        QD_DIAG("%s_NLOS,Bn,%g,%.5f\n%s_NLOS,D2n,%g,%.5f\n", scen, pl["Bn"][0], Bn, scen, pl["D2n"][0], 3.7 * h * h);
        CHECK(pl["An"][0] == 43.42);
        CHECK(std::abs(pl["Bn"][0] - Bn) < 0.5e-3);
        CHECK(pl["Cn"][0] == 20.0);
        CHECK(std::abs(pl["Dn"][0] - (-24.37 + 3.0 * 3.1)) < 1e-9);
        CHECK(std::abs(pl["D2n"][0] - 3.7 * h * h) < 1e-9);
        CHECK(pl["Fn"][0] == -3.1);
        CHECK(pl["G1n"][0] == -3.2);
        CHECK(pl["G2n"][0] == 11.75);

        // RMa NLOS is max(LOS, NLOS'); SMa NLOS has no LOS term in the TR, it is switched off in the file
        if (std::string(scen) == "RMa")
            CHECK(std::abs(pl["A1"][0] - A1) < 0.5e-3);
        else
            CHECK((pl["A1"][0] == 0.0 && pl["A2"][0] == 0.0 && pl["B"][0] == 0.0 && pl["C"][0] == 0.0));
    }
}
