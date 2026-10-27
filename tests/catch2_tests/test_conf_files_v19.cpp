// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Diagnostic output: off by default. QD_DIAG_CONF=1 prints the tables of this file, QD_DIAG=1 those of all areas
// (see qd_test_diag.hpp); the assertions run either way. Example:
//   QD_DIAG_CONF=1 ./build_linux/test_bin "conf files v19*"

// Verification suite for the 23 scenario files 'data/config/3GPP_38.901_v19_*.conf'.
//
// Pins down: all files are read by quadriga_lib::conf_file_read without ignored keys, pass the value
// checks and have a positive definite LSP correlation matrix; the keys for the EoD offset and for the
// ESD mean reproduce the formulas of 3GPP TR 38.901 v19.5.0; a set of table values that changed with
// Release 19 or that belong to the new components; the derived path-loss constants and the path-loss keys of
// Table 7.4.1-1 (the path loss itself is verified in test_path_loss).
// Left to other suites: the grammar and the value checks (test_conf_file_read).
//
// Oracles: the formulas and numbers of TR 38.901 v19.5.0 (Tables 7.4.1-1, 7.5-6 to 7.5-12, 7.6.9-1, 7.6-13-1,
// 7.6.14.1.2-1/-2, 7.6.15-1, 7.6.3.1-2), typed into this file from the TR, independently of the files; the hall
// and room dimensions of the InF calibration (Table 7.8-7, clause 7.8.4) and of the indoor office (Table 7.2-2).
//
// Conventions: QuaDRiGa uses elevation angles (+90 deg is up), the TR zenith angles. The EoD offset is
// the ZOD offset of the TR with the opposite sign; spreads are the same in both conventions.
// The key formulas are written out in this file, independently of the library functions that evaluate them.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_gbsm.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "qd_test_diag.hpp"
QD_TEST_DIAG_SWITCH("QD_DIAG_CONF")

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

    // Key formula: EoD offset in degrees, elevation, positive = upwards
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

    // Key formula: mean of log10(ESD / 1 deg), without the legacy omega, epsilon, zeta, alpha terms
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

    // Hall or room of a file: TR 38.901 Table 7.8-7 (InF calibration, clause 7.8.4): InF-SL and InF-DH in the small
    // hall 120 m x 60 m, InF-DL and InF-SH in the big hall 300 m x 150 m, room height 10 m; Table 7.2-2 (indoor
    // office): 120 m x 50 m x 3 m. Returns false for files without a hall.
    bool hall_of(const std::string &name, double &len, double &wid, double &hgt)
    {
        if (name == "InF_NLOS_SL" || name == "InF_NLOS_DH")
            len = 120.0, wid = 60.0, hgt = 10.0;
        else if (name == "InF_NLOS_DL" || name == "InF_NLOS_SH")
            len = 300.0, wid = 150.0, hgt = 10.0;
        else if (name == "Indoor_NLOS")
            len = 120.0, wid = 50.0, hgt = 3.0;
        else
            return false;
        return true;
    }

    // Volume-to-surface ratio V/S of a closed box
    double vs_ratio(double len, double wid, double hgt)
    {
        return len * wid * hgt / (2.0 * (len * wid + len * hgt + wid * hgt));
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

        // LOS files have a K-factor, all others switch the LOS path off and carry one extra cluster
        std::string s, state;
        bool los_file, o2i;
        split_name(name, s, state, los_file, o2i);
        if (los_file)
            CHECK(sp["KF_mu"] > 0.0);
        else
            CHECK(sp["KF_mu"] == -100.0);

        // Absolute time of arrival only in NLOS and NLOS_O2I files: the TR gives it for NLOS links, and the O2I
        // links of an NLOS file are NLOS links; active if absTOA_mu > -30
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

    // Worked examples of the EoD offset at single points
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

TEST_CASE("conf files v19 - state distances, indoor distance and TOA bound per file")
{
    // Oracles: TR 38.901 v19.5.0 Table 7.6.3.1-2 (LOS/NLOS state: RMa 60 m, UMi, UMa, SMa 50 m, InH 10 m,
    // InF d_clutter / 2 with d_clutter 10 m for SL, SH and 2 m for DL, DH; indoor/outdoor state 50 m for RMa, UMi,
    // UMa and SMa); clause 7.4.3.1 (maximum d2D-in: 25 m for UMa, UMi and SMa UTs in commercial buildings, 10 m for
    // RMa); clause 7.6.9 (excess delay bound 2L/c, L = max(length, width, height) of the hall or room).
    // The state distances are stored in the NLOS files, the indoor/outdoor values in the O2I files; all other
    // files have 0.
    QD_DIAG("\n# diag: conf files v19, state distances, d2D-in maximum and absTOA_Lmax\n");
    QD_DIAG("file,LOS_state_lambda,IO_state_lambda,O2I_d2D_in_max,absTOA_Lmax,L_ref\n");
    for (const std::string &name : v19_names())
    {
        INFO("file " << name);
        spmap sp = read_v19(name);
        std::string scen, state;
        bool los_file, o2i;
        split_name(name, scen, state, los_file, o2i);
        const bool nlos_file = name.find("NLOS") != std::string::npos && !o2i;

        double los_ref = 0.0;
        if (nlos_file)
            los_ref = scen == "RMa" ? 60.0 : scen == "Indoor" ? 10.0
                    : scen == "InF" ? ((name == "InF_NLOS_SL" || name == "InF_NLOS_SH") ? 10.0 : 2.0) / 2.0
                                    : 50.0;
        const double io_ref = o2i ? 50.0 : 0.0;
        const double d2d_ref = o2i ? (scen == "RMa" ? 10.0 : 25.0) : 0.0;

        double len = 0.0, wid = 0.0, hgt = 0.0, L_ref = 0.0;
        if (hall_of(name, len, wid, hgt))
            L_ref = std::max(len, std::max(wid, hgt));

        QD_DIAG("%s,%g,%g,%g,%g,%g\n", name.c_str(), sp["LOS_state_lambda"], sp["IO_state_lambda"],
                sp["O2I_d2D_in_max"], sp["absTOA_Lmax"], L_ref);
        CHECK(sp["LOS_state_lambda"] == los_ref);
        CHECK(sp["IO_state_lambda"] == io_ref);
        CHECK(sp["O2I_d2D_in_max"] == d2d_ref);
        CHECK(sp["absTOA_Lmax"] == L_ref);
    }
}

TEST_CASE("conf files v19 - the InF delay spread follows the V/S rule of Table 7.5-6 Part-3")
{
    // Oracle: TR 38.901 v19.5.0 Table 7.5-6 Part-3: mu_lgDS = log10(26 (V/S) + 14) - 9.35 (LOS) and
    // log10(30 (V/S) + 32) - 9.44 (NLOS), V/S of the calibration hall of Table 7.8-7 (closed box, room height
    // 10 m); the InF LOS file serves all sub-scenarios and uses the average V/S of the two halls.
    // The key rule: mu_lgDS = DS_mu + log10(DS_VS_a * DS_VS + DS_VS_b). Files without a hall have no V/S keys.
    // The files write V/S rounded to 2 decimals (4, 50/11 as 4.55, 47/11 as 4.27): derived values are written
    // with as few digits as keep the LSP mean within 0.1 %. The rounding moves mu_lgDS by at most 3.52e-4
    // (log10((30 * 4.55 + 32) / (30 * 50/11 + 32))); the tolerance is that limit, log10(1.001) = 4.34e-4.
    const double tol_mu = std::log10(1.001);
    const double vs_small = vs_ratio(120.0, 60.0, 10.0), vs_big = vs_ratio(300.0, 150.0, 10.0);
    QD_DIAG("\n# diag: conf files v19, InF delay spread [log10(s)]\n");
    QD_DIAG("file,DS_mu,DS_VS,DS_VS_a,DS_VS_b,mu_keys,mu_tr\n");
    for (const std::string &name : v19_names())
    {
        INFO("file " << name);
        spmap sp = read_v19(name);
        if (name.compare(0, 4, "InF_") != 0)
        {
            CHECK(sp["DS_VS"] == 0.0);
            CHECK(sp["DS_VS_a"] == 0.0);
            CHECK(sp["DS_VS_b"] == 0.0);
            continue;
        }
        const bool los = name == "InF_LOS";
        double len = 0.0, wid = 0.0, hgt = 0.0;
        const double vs = los ? 0.5 * (vs_small + vs_big) : (hall_of(name, len, wid, hgt), vs_ratio(len, wid, hgt));
        const double mu_tr = los ? std::log10(26.0 * vs + 14.0) - 9.35 : std::log10(30.0 * vs + 32.0) - 9.44;
        const double mu_keys = sp["DS_mu"] + std::log10(sp["DS_VS_a"] * sp["DS_VS"] + sp["DS_VS_b"]);
        QD_DIAG("%s,%g,%.17g,%g,%g,%.15f,%.15f\n", name.c_str(), sp["DS_mu"], sp["DS_VS"], sp["DS_VS_a"],
                sp["DS_VS_b"], mu_keys, mu_tr);

        CHECK(sp["DS_mu"] == (los ? -9.35 : -9.44));
        CHECK(sp["DS_VS_a"] == (los ? 26.0 : 30.0));
        CHECK(sp["DS_VS_b"] == (los ? 14.0 : 32.0));
        CHECK(std::abs(sp["DS_VS"] - std::round(100.0 * vs) / 100.0) < 1e-12); // V/S rounded to 2 decimals
        CHECK(std::abs(mu_keys - mu_tr) < tol_mu);
    }
    CHECK(std::abs(vs_small - 4.0) < 1e-15);
    CHECK(std::abs(vs_big - 50.0 / 11.0) < 1e-15);
}

TEST_CASE("conf files v19 - path-loss constants of RMa and SMa follow Table 7.4.1-1")
{
    // Oracle: the TR formula evaluated at the building height h and street width W that the file states in its
    // comments (RMa: h = 5 m, W = 20 m; SMa: h = 10 m, W = 10 m), with c = 3e8 m/s.
    //   LOS:  A1 = 20 + min(0.03 h^1.72, 10);  B = 20 log10(40 pi / 3) - min(0.044 h^1.72, 14.77);  D = 0.002 log10(h)
    //         break point dBP = 2 pi hBS hUT fc / c: E = 2 pi / 0.3 = 20 pi / 3 with fc in GHz, hE = 0 (note 5)
    //   NLOS: Bn = 161.04 - 7.1 log10(W) + 7.5 log10(h) - 3 * 43.42 + 4.97;  Dn = -24.37 + 3 * 3.1;  D2n = 3.7 h^2
    // The files write the derived constants with 4 significant digits (SMa A1 with 5, where 4 digits would change
    // the loss by more than 0.01 dB): RMa A1, B, Bn, E with 2 decimals, D with 6; SMa A1 with 3, B and E with 2.
    // Each value equals the exact constant rounded to that many decimals: it has at most that many decimals, and
    // it is within half a unit of the last digit (plus 1e-12) of the exact constant.
    const double pi = 3.14159265358979323846;
    auto rounded = [](double file, double exact, int decimals)
    {
        const double s = std::pow(10.0, decimals);
        const bool digits = std::abs(file * s - std::round(file * s)) < 1e-6;
        return digits && std::abs(file - exact) <= 0.5 / s + 1e-12;
    };
    QD_DIAG("\n# diag: conf files v19, derived path-loss constants\n");
    QD_DIAG("file,key,value,ref\n");
    for (const char *scen : {"RMa", "SMa"})
    {
        const bool rma = std::string(scen) == "RMa";
        const double h = rma ? 5.0 : 10.0, W = rma ? 20.0 : 10.0;
        const double A1 = 20.0 + std::min(0.03 * std::pow(h, 1.72), 10.0);
        const double B = 20.0 * std::log10(40.0 * pi / 3.0) - std::min(0.044 * std::pow(h, 1.72), 14.77);
        const double D = 0.002 * std::log10(h), E = 2.0 * pi / 0.3;
        const double Bn = 161.04 - 7.1 * std::log10(W) + 7.5 * std::log10(h) - 3.0 * 43.42 + 4.97;

        for (const char *suffix : {"_LOS", "_LOS_O2I"})
        {
            std::unordered_map<std::string, std::vector<double>> pl;
            std::string model;
            read_v19(std::string(scen) + suffix, &pl, &model);
            INFO("file " << scen << suffix);
            CHECK(model == "dual_slope");
            REQUIRE((pl.count("A1") && pl.count("A2") && pl.count("B") && pl.count("C") && pl.count("D") && pl.count("E") && pl.count("hE")));
            QD_DIAG("%s%s,A1,%g,%.6f\n%s%s,B,%g,%.6f\n%s%s,D,%g,%.9f\n%s%s,E,%g,%.6f\n", scen, suffix, pl["A1"][0], A1,
                    scen, suffix, pl["B"][0], B, scen, suffix, pl["D"][0], D, scen, suffix, pl["E"][0], E);
            CHECK(rounded(pl["A1"][0], A1, rma ? 2 : 3));
            CHECK(pl["A2"][0] == 40.0);
            CHECK(rounded(pl["B"][0], B, 2));
            CHECK(pl["C"][0] == 20.0);
            if (rma)
                CHECK(rounded(pl["D"][0], D, 6));
            else
                CHECK(pl["D"][0] == 0.002); // exact for h = 10 m
            CHECK(rounded(pl["E"][0], E, 2));
            CHECK(pl["hE"][0] == 0.0);
            if (std::string(suffix) == "_LOS")
            {
                CHECK(pl["sig1"][0] == 4.0);
                CHECK(pl["sig2"][0] == 6.0);
            }
        }

        for (const char *suffix : {"_NLOS", "_NLOS_O2I"})
        {
            std::unordered_map<std::string, std::vector<double>> pl;
            std::string model;
            read_v19(std::string(scen) + suffix, &pl, &model);
            INFO("file " << scen << suffix);
            CHECK(model == "nlos");
            REQUIRE((pl.count("An") && pl.count("Bn") && pl.count("Cn") && pl.count("Dn") && pl.count("D2n") && pl.count("Fn") &&
                     pl.count("G1n") && pl.count("G2n")));
            QD_DIAG("%s%s,Bn,%g,%.6f\n%s%s,D2n,%g,%.5f\n", scen, suffix, pl["Bn"][0], Bn, scen, suffix, pl["D2n"][0], 3.7 * h * h);
            CHECK(pl["An"][0] == 43.42);
            CHECK(rounded(pl["Bn"][0], Bn, 2));
            CHECK(pl["Cn"][0] == 20.0);
            CHECK(std::abs(pl["Dn"][0] - (-24.37 + 3.0 * 3.1)) < 1e-9);
            CHECK(std::abs(pl["D2n"][0] - 3.7 * h * h) < 1e-9);
            CHECK(pl["Fn"][0] == -3.1);
            CHECK(pl["G1n"][0] == -3.2);
            CHECK(pl["G2n"][0] == 11.75);

            // RMa NLOS is max(LOS, NLOS'), with the LOS constants of the LOS file; SMa NLOS has no LOS term in the
            // TR: PL_LOSmax = 0 switches it off, and the keys of the LOS term are not written
            if (rma)
            {
                REQUIRE((pl.count("A1") && pl.count("B") && pl.count("D") && pl.count("E")));
                CHECK(rounded(pl["A1"][0], A1, 2));
                CHECK(rounded(pl["B"][0], B, 2));
                CHECK(rounded(pl["D"][0], D, 6));
                CHECK(rounded(pl["E"][0], E, 2));
                CHECK(pl.count("LOSmax") == 0);
            }
            else
            {
                REQUIRE(pl.count("LOSmax") == 1);
                CHECK(pl["LOSmax"][0] == 0.0);
                for (const char *k : {"A1", "A2", "B", "C", "E", "hE"})
                    CHECK(pl.count(k) == 0);
            }
        }
    }

    // Break point of note 1 (UMa, UMi) and the LOS term of the InH and InF NLOS files: E = 4 / 0.3 = 40/3, written
    // with 2 decimals as 13.33; UMa has the random environment height of note 1 (PL_hE_rand = 1); InF-DL takes the
    // maximum with the InF-SL term 33 + 25.5 log10(d3D) + 20 log10(fc) as the additional term of the model "nlos"
    for (const std::string &name : v19_names())
    {
        std::unordered_map<std::string, std::vector<double>> pl;
        std::string model;
        read_v19(name, &pl, &model);
        INFO("file " << name);
        const std::string scen = name.substr(0, name.find('_'));
        const bool nlos = name.find("NLOS") != std::string::npos;
        if (scen == "UMa" || scen == "UMi" || ((scen == "Indoor" || scen == "InF") && nlos))
        {
            REQUIRE(pl.count("E") == 1);
            CHECK(pl["E"][0] == 13.33);
            CHECK(std::abs(pl["E"][0] - std::round(100.0 * 40.0 / 3.0) / 100.0) < 1e-12);
        }
        CHECK(pl.count("hE_rand") == (scen == "UMa" ? 1u : 0u));
        if (scen == "UMa")
            CHECK(pl["hE_rand"][0] == 1.0);
        const bool dl = name == "InF_NLOS_DL";
        CHECK(pl.count("Ax") == (dl ? 1u : 0u));
        CHECK(pl.count("Bx") == (dl ? 1u : 0u));
        CHECK(pl.count("Cx") == (dl ? 1u : 0u));
        if (dl)
            CHECK((pl["Ax"][0] == 25.5 && pl["Bx"][0] == 33.0 && pl["Cx"][0] == 20.0));
        CHECK(pl.count("LOSmax") == (scen == "SMa" && nlos ? 1u : 0u));
    }
}
