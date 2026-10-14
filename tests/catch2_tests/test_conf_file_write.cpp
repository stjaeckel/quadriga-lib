// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Verification suite for quadriga_lib::conf_file_write.
//
// Pins down: the text layout of a written file (header, validity range, short names, section order and
// headers, overview formulas, correlation matrix, paragraphs, column alignment, comments with units, path-loss
// comments and formula blocks per model), the number format, which keys are written with and without
// 'write_defaults', the order of the path-loss keys, the round trip of the conf files in data/config, overwriting,
// the safe write (a failed write leaves an existing file unchanged), the independence of the global locale, and the
// input validation.
// Left to other suites: the grammar and the value checks of the reader (test_conf_file_read,
// test_scenpar_validate).
//
// Oracles: the layout is a complete file typed by hand from the rules of QuaDRiGa's write_conf_file.m
// (write_par_line: name, ' = ', padding to 19 characters, value padded to 13, ' % ', comment), with the
// decisions of review point A: a key is written if its value differs from its default, sections without a
// written key are left out, no trailing spaces, sections 9 (EoD offset) and 10 (optional components) are new.
// Number texts are typed by hand (the shortest of 15, 16, 17 significant digits that reads back exactly).
// The round trip uses conf_file_read as a supporting function: read, write, read again must give the same
// values bit for bit.
//
// Conventions: every case writes its own files 'test_conf_file_write_*.conf' into the working directory and
// removes them.

#include <catch2/catch_test_macros.hpp>

#include "quadriga_tools.hpp"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <csignal>
#include <sys/resource.h>
#include <unistd.h>
#endif

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

    const char *LIB_CONFIG_DIR = "data/config";

    // The 165 numeric keys, typed from QuaDRiGa's set_scenario_table.m (without 'SubpathMethod' and
    // 'GR_epsilon') and the 27 keys for TR 38.901 v19.5.0
    const std::vector<std::string> KEYS = {
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
        "LSP_fc_min", "ES_D_mu_B", "ES_D_mu_C", "ES_D_mu_D", "EoD_offset_e", "EoD_offset_e_gamma",
        "EoD_offset_a", "EoD_offset_a_gamma", "EoD_offset_b", "EoD_offset_c", "EoD_offset_c_gamma",
        "EoD_offset_c_hUT", "EoD_offset_hcl", "PerClusterDS_fc_min", "PerClusterES_D_scale", "absTOA_Lmax",
        "NumClusters_min", "NF_k1", "NF_alpha", "NF_beta", "SNS_mu", "SNS_sigma", "SNS_A", "SNS_B", "SNS_R",
        "SNS_delta_var", "PolVar_sigma"};

    // Removes a file when it goes out of scope, and at construction
    struct tmp_name
    {
        std::string fn;
        explicit tmp_name(const std::string &name) : fn(name) { std::remove(fn.c_str()); }
        ~tmp_name() { std::remove(fn.c_str()); }
    };

    bool file_exists(const std::string &fn) { return std::filesystem::is_regular_file(fn); }

    std::string slurp(const std::string &fn)
    {
        std::ifstream f(fn, std::ios::binary);
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    void put(const std::string &fn, const std::string &text)
    {
        std::ofstream f(fn, std::ios::binary);
        f << text;
    }

    bool bits_equal(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

    // Everything conf_file_read returns, read with validate = true
    struct conf_data
    {
        std::unordered_map<std::string, double> scenpar;
        std::string subpath_method = "legacy", pl_model;
        std::unordered_map<std::string, std::vector<double>> plpar;
        std::vector<std::string> pl_order, short_names, ignored_keys;
    };

    conf_data read_back(const std::string &fn)
    {
        conf_data d;
        quadriga_lib::conf_file_read(fn, d.scenpar, &d.subpath_method, &d.pl_model, &d.plpar, nullptr, nullptr,
                                     &d.pl_order, &d.short_names, &d.ignored_keys, true);
        return d;
    }

    void write_data(const std::string &fn, const conf_data &d, bool write_defaults)
    {
        quadriga_lib::conf_file_write(fn, d.scenpar, d.subpath_method, d.pl_model, &d.plpar, &d.pl_order,
                                      &d.short_names, "", nullptr, nullptr, write_defaults);
    }

    // Differences between two data sets as text, empty if they are identical bit for bit
    std::string differences(const conf_data &a, const conf_data &b)
    {
        std::string d;
        if (a.scenpar.size() != b.scenpar.size())
            d += " scenpar size;";
        for (const auto &kv : a.scenpar)
        {
            auto it = b.scenpar.find(kv.first);
            if (it == b.scenpar.end() || !bits_equal(kv.second, it->second))
                d += " " + kv.first + ";";
        }
        if (a.subpath_method != b.subpath_method)
            d += " SubpathMethod;";
        if (a.pl_model != b.pl_model)
            d += " PL_model;";
        if (a.pl_order != b.pl_order)
            d += " pl_order;";
        if (a.short_names != b.short_names)
            d += " short_names;";
        if (a.plpar.size() != b.plpar.size())
            d += " plpar size;";
        for (const auto &kv : a.plpar)
        {
            auto it = b.plpar.find(kv.first);
            bool same = it != b.plpar.end() && it->second.size() == kv.second.size();
            for (size_t i = 0; same && i < kv.second.size(); ++i)
                same = bits_equal(kv.second[i], it->second[i]);
            if (!same)
                d += " PL_" + kv.first + ";";
        }
        return d;
    }

    // Lines of a text without the line ends
    std::vector<std::string> lines_of(const std::string &text)
    {
        std::vector<std::string> out;
        std::stringstream ss(text);
        std::string line;
        while (std::getline(ss, line))
            out.push_back(line);
        return out;
    }

    // Name before the '=' of an assignment line, empty for comment and blank lines
    std::string key_of(const std::string &line)
    {
        if (line.empty() || line[0] == '%' || line.find('=') == std::string::npos)
            return "";
        std::string k = line.substr(0, line.find('='));
        while (!k.empty() && k.back() == ' ')
            k.pop_back();
        return k;
    }

    // Number of assignment lines for 'key'
    int n_lines(const std::string &text, const std::string &key)
    {
        int n = 0;
        for (const auto &l : lines_of(text))
            n += key_of(l) == key ? 1 : 0;
        return n;
    }

    // Value text of the first assignment line of 'key' (between '=' and '%', trimmed), "<none>" if missing
    std::string value_text(const std::string &text, const std::string &key)
    {
        for (const auto &l : lines_of(text))
            if (key_of(l) == key)
            {
                size_t b = l.find('=') + 1, e = l.find('%');
                if (e == std::string::npos)
                    e = l.size();
                while (b < e && l[b] == ' ')
                    ++b;
                while (e > b && l[e - 1] == ' ')
                    --e;
                return l.substr(b, e - b);
            }
        return "<none>";
    }

    // Comment text of the first assignment line of 'key', "<none>" if the key or the comment is missing
    std::string comment_text(const std::string &text, const std::string &key)
    {
        for (const auto &l : lines_of(text))
            if (key_of(l) == key)
                return l.find(" % ") == std::string::npos ? "<none>" : l.substr(l.find(" % ") + 3);
        return "<none>";
    }

    // Overview line of an LSP in section 0, e.g. "% ESD = ...", without the line end; "<none>" if missing
    std::string overview_line(const std::string &text, const std::string &label)
    {
        const std::string start = "% " + label + std::string(4 - label.size(), ' ') + "= ";
        for (const auto &l : lines_of(text))
            if (l.compare(0, start.size(), start) == 0)
                return l;
        return "<none>";
    }

    // Number of assignment lines in a text
    int n_assignments(const std::string &text)
    {
        int n = 0;
        for (const auto &l : lines_of(text))
            n += key_of(l).empty() ? 0 : 1;
        return n;
    }

    // Write 'scenpar' alone, read it back; returns the differences to the input completed with defaults
    std::string round_trip(const std::unordered_map<std::string, double> &scenpar, const std::string &fn,
                           bool write_defaults, std::string *text = nullptr)
    {
        quadriga_lib::conf_file_write(fn, scenpar, "legacy", "", nullptr, nullptr, nullptr, "", nullptr, nullptr,
                                      write_defaults);
        if (!file_exists(fn))
            return " <file not written>";
        if (text != nullptr)
            *text = slurp(fn);
        conf_data in, out = read_back(fn);
        in.scenpar = scenpar;
        quadriga_lib::scenpar_validate(in.scenpar);
        in.pl_order = out.pl_order, in.plpar = out.plpar; // path loss is not part of this check
        std::string d = differences(in, out);
        if (!out.ignored_keys.empty())
            d += " ignored keys;";
        return d;
    }

    // Round trip of all conf files in a directory; returns the number of files compared
    int round_trip_dir(const std::string &dir, std::vector<std::string> &skipped)
    {
        std::vector<std::string> names;
        for (const auto &e : std::filesystem::directory_iterator(dir))
            if (e.path().extension() == ".conf")
                names.push_back(e.path().filename().string());
        std::sort(names.begin(), names.end());

        tmp_name f1("test_conf_file_write_rt.conf");
        int n = 0;
        QD_DIAG("\n# diag: conf_file_write round trip, %s\n", dir.c_str());
        QD_DIAG("file,write_defaults,n_assign_orig,n_assign_written,differences,idempotent\n");
        for (const auto &name : names)
        {
            conf_data orig;
            try
            {
                orig = read_back(dir + "/" + name);
            }
            catch (const std::invalid_argument &)
            {
                skipped.push_back(name);
                QD_DIAG("%s,-,-,-,rejected by conf_file_read,-\n", name.c_str());
                continue;
            }
            for (bool wd : {false, true})
            {
                INFO("file " << name << ", write_defaults " << wd);
                std::remove(f1.fn.c_str());
                write_data(f1.fn, orig, wd);
                REQUIRE(file_exists(f1.fn));
                const std::string text1 = slurp(f1.fn);
                const conf_data back = read_back(f1.fn);
                const std::string d = differences(orig, back);
                write_data(f1.fn, back, wd); // same file name, so that the header is the same
                const bool idem = slurp(f1.fn) == text1;
                QD_DIAG("%s,%d,%d,%d,%s,%d\n", name.c_str(), (int)wd, n_assignments(slurp(dir + "/" + name)),
                        n_assignments(text1), d.empty() ? "none" : d.c_str(), (int)idem);
                CHECK(d.empty());
                CHECK(back.ignored_keys.empty());
                CHECK(idem);
            }
            ++n;
        }
        return n;
    }
}

TEST_CASE("conf_file_write - layout of a written file")
{
    tmp_name f("test_conf_file_write_layout.conf");

    const std::unordered_map<std::string, double> scenpar = {
        {"DS_mu", -6.28}, {"DS_sigma", 0.39}, {"DS_omega", 0.5}, {"DS_gamma", -0.204}, {"DS_epsilon", 0.1},
        {"DS_delta", 0.02}, {"KF_mu", -100.0}, {"SF_sigma", 6.0}, {"ES_D_mu", 0.9}, {"ES_D_mu_A", -2.1},
        {"ES_D_mu_min", -0.5}, {"ES_D_mu_B", -0.01}, {"XPR_mu", 7.0}, {"XPR_sigma", 3.0}, {"LSP_fc_min", 6.0},
        {"DS_lambda", 40.0}, {"SF_lambda", 0.0}, {"KF_lambda", 50.0}, {"ds_kf", 0.05}, {"ds_sf", -0.4},
        {"asD_ds", 0.4}, {"NumClusters", 21.0}, {"r_DS", 2.3}, {"LNS_ksi", 0.0}, {"PerClusterDS", 6.5622},
        {"PerClusterES_D_scale", 0.375}, {"EoD_offset_a", -1.5}, {"EoD_offset_b", 10.0}, {"EoD_offset_c", 3.3},
        {"SC_lambda", 50.0}, {"GR_enabled", 1.0}, {"absTOA_mu", -99.5}, {"absTOA_sigma", 0.3}, {"NF_k1", 2.0},
        {"PolVar_sigma", 3.0}};
    const std::unordered_map<std::string, std::vector<double>> plpar = {
        {"A1", {22.0}}, {"B", {28.0}}, {"An", {39.08}}, {"Exn", {1.0, 2.5}}};
    const std::vector<std::string> pl_order = {"A1", "B", "An", "Exn"};
    const std::vector<std::string> short_names = {"Ab", "Cd"};
    const std::vector<std::string> ref_sec = {"Table 1", "Table 2", "", "Table 4", "Table 5",
                                              "Table 6", "Table 7", "Table 8", "Table 9", "Table 10"};
    const arma::mat valid_range = {{0.5, 100.0}, {10.0, 10.0}, {QNAN, 25.0}, {1.5, 22.5}};

    // Oracle: the complete file, typed from the rules of write_conf_file.m (see header comment)
    const std::string expected =
        "% Config file for scenario \"test_conf_file_write_layout\"\n"
        "% See: Test reference\n"
        "\n"
        "% Valid frequency range:        0.5 to 100 GHz\n"
        "% Valid distance range:         10 m\n"
        "% Valid MT antenna height:      1.5 to 22.5 m\n"
        "\n"
        "ShortName =            Ab           % You can use this short name to load the file\n"
        "ShortName =            Cd           % You can use this short name to load the file\n"
        "\n"
        "% ==================================================================================================\n"
        "% Large scale distributions\n"
        "% See: Test reference; Table 1\n"
        "% ==================================================================================================\n"
        "\n"
        "% DS  = -6.28 - 0.204 * log10( 0.5 + max( fGHz, 6 ) ) + 0.1 * log10( d2D ) + Xds * ( 0.39 + 0.02 * log10( 0.5 + max( fGHz, 6 ) ) )\n"
        "% KF  = -100 + Xkf * ( 0 )\n"
        "% SF  = Xsf * ( 6 )\n"
        "% ASD = 1 + Xasd * ( 0.2 )\n"
        "% ASA = 1 + Xasa * ( 0.2 )\n"
        "% ESD = max( -0.5, 0.9 - 2.1 * d2D / 1000 - 0.01 * hUT ) + Xesd * ( 0.2 )\n"
        "% ESA = 0.3 + Xesa * ( 0.2 )\n"
        "% XPR = 7 + Xxpr * ( 3 )\n"
        "\n"
        "DS_mu =               -6.28         % delay spread [log10(s)] @ 0.5 GHz, 1 m TX-RX dist.\n"
        "DS_sigma =             0.39         % delay spread STD [log10(s)] @ 0.5 GHz\n"
        "DS_omega =             0.5          % reference frequency offset for the DS [GHz]\n"
        "DS_gamma =            -0.204        % freq.-dep. of DS [log10(s)/log10(GHz)]\n"
        "DS_epsilon =           0.1          % TX-RX 2D dist.-dep. of DS [log10(s)/log10(m)]\n"
        "DS_delta =             0.02         % freq.-dep. of DS STD [log10(s)/log10(GHz)]\n"
        "\n"
        "KF_mu =               -100          % Ricean K-factor [dB]\n"
        "\n"
        "SF_sigma =             6            % Shadow Fading STD [dB]\n"
        "\n"
        "ES_D_mu =              0.9          % elevation of departure angle spread [log10(deg)]\n"
        "ES_D_mu_A =           -2.1          % TX-RX 2D dist.-dep. of ESD [log10(deg)/km]\n"
        "ES_D_mu_min =         -0.5          % minimum ESD reference value [log10(deg)]\n"
        "ES_D_mu_B =           -0.01         % RX height-dep. of ESD [log10(deg)/m]\n"
        "\n"
        "XPR_mu =               7            % cross-polarization ratio [dB]\n"
        "XPR_sigma =            3            % cross-polarization ratio STD [dB]\n"
        "\n"
        "LSP_fc_min =           6            % lower limit of fc for the frequency-dependent terms [GHz]\n"
        "\n"
        "% ==================================================================================================\n"
        "% Large-Scale fading decorrelation distances\n"
        "% See: Test reference; Table 2\n"
        "% ==================================================================================================\n"
        "\n"
        "DS_lambda =            40           % DS decorrelation distance [m]\n"
        "SF_lambda =            0            % SF decorrelation distance [m]\n"
        "KF_lambda =            50           % KF decorrelation distance [m]\n"
        "\n"
        "% ==================================================================================================\n"
        "% Inter-parameter correlations\n"
        "% ==================================================================================================\n"
        "\n"
        "%         DS     KF     SF     ASD    ASA    ESD    ESA    XPR\n"
        "%     |   1      0.05  -0.4    0.4    0      0      0      0     | DS\n"
        "%     |   0.05   1      0      0      0      0      0      0     | KF\n"
        "%     |  -0.4    0      1      0      0      0      0      0     | SF\n"
        "% R = |   0.4    0      0      1      0      0      0      0     | ASD\n"
        "%     |   0      0      0      0      1      0      0      0     | ASA\n"
        "%     |   0      0      0      0      0      1      0      0     | ESD\n"
        "%     |   0      0      0      0      0      0      1      0     | ESA\n"
        "%     |   0      0      0      0      0      0      0      1     | XPR\n"
        "\n"
        "ds_kf =                0.05         % DS vs. KF\n"
        "ds_sf =               -0.4          % DS vs. SF\n"
        "asD_ds =               0.4          % DS vs. ASD\n"
        "\n"
        "% ==================================================================================================\n"
        "% Model parameters\n"
        "% See: Test reference; Table 4\n"
        "% ==================================================================================================\n"
        "\n"
        "NumClusters =          21           % number of clusters\n"
        "SubpathMethod =        mmMAGIC      % subpath mapping method (legacy, Laplacian or mmMAGIC)\n"
        "\n"
        "r_DS =                 2.3          % delay scaling factor\n"
        "LNS_ksi =              0            % per cluster shadowing STD [dB]\n"
        "\n"
        "PerClusterDS =         6.5622       % cluster delay spread [ns]\n"
        "PerClusterES_D_scale =  0.375        % cluster ESD as fraction of 10^(ESD mean); 0 = use PerClusterES_D\n"
        "\n"
        "% ==================================================================================================\n"
        "% EoD offset (elevation of departure, TR: ZOD offset)\n"
        "% See: Test reference; Table 9\n"
        "% ==================================================================================================\n"
        "\n"
        "EoD_offset_a =        -1.5          % distance exponent\n"
        "EoD_offset_b =         10           % minimum 2D distance [m]; 0 = no power term\n"
        "EoD_offset_c =         3.3          % constant of the exponent\n"
        "\n"
        "% ==================================================================================================\n"
        "% Decorrelation distance for the small-scale fading spatial consistency\n"
        "% See: Test reference; Table 5\n"
        "% ==================================================================================================\n"
        "\n"
        "SC_lambda =            50           % decorrelation distance [m]; 0 = disabled\n"
        "\n"
        "% ==================================================================================================\n"
        "% Parameters for the ground reflection extension\n"
        "% See: Test reference; Table 6\n"
        "% ==================================================================================================\n"
        "\n"
        "GR_enabled =           1            % Enables the explicit ground reflection model\n"
        "\n"
        "% ==================================================================================================\n"
        "% Absolute time of arrival offset\n"
        "% See: Test reference; Table 7\n"
        "% ==================================================================================================\n"
        "\n"
        "absTOA_mu =           -99.5         % absolute time of arrival offset reference value [log10(s)]\n"
        "absTOA_sigma =         0.3          % absolute time of arrival offset reference STD [log10(s)]\n"
        "\n"
        "% ==================================================================================================\n"
        "% Optional modelling components\n"
        "% See: Test reference; Table 10\n"
        "% ==================================================================================================\n"
        "\n"
        "NF_k1 =                2            % number of strongest clusters with scaling factor 1\n"
        "\n"
        "PolVar_sigma =         3            % STD of the polarization variability powers [dB]\n"
        "\n"
        "% ==================================================================================================\n"
        "% Path-loss model\n"
        "% See: Test reference; Table 8\n"
        "% ==================================================================================================\n"
        "\n"
        "% Formula for dual-slope (LOS) pathloss model:\n"
        "% (Distance in meters, frequency in GHz)\n"
        "%\n"
        "%     PL = PL1 for d2D <= dBP | PL2 for d2D > dBP\n"
        "%    PL1 = A1 * log10( d3D ) + B + C * log10( fGHz ) + D * d3D\n"
        "%    PL2 = PL1( dBP ) + A2 * log10( d3D / dBP )\n"
        "%    dBP = E * ( hBS-hE ) * ( hMS-hE ) * fGHz\n"
        "\n"
        "% Formula for 3GPP NLOS pathloss model:\n"
        "% (Distances and heights in meters, frequency in GHz)\n"
        "%\n"
        "%    PLn =  An * log10( d3D )\n"
        "%        +  Bn\n"
        "%        +  Cn * log10( fGHz )\n"
        "%        +  Dn * log10( hBS )\n"
        "%        + D1n * log10( hBS ) / hBS\n"
        "%        + D2n * log10( hBS ) / hBS^2\n"
        "%        + D3n * hBS\n"
        "%        +  En * log10( hUT )\n"
        "%        + E1n * log10( hUT ) / hUT\n"
        "%        + E2n * log10( hUT ) / hUT^2\n"
        "%        + E3n * hUT\n"
        "%        +  Fn * log10( hBS ) * log10( d3d )\n"
        "%        + G1n * log10^2( G2n * hUT )\n"
        "%\n"
        "%     PL = max( PL_LOS, PLn )\n"
        "\n"
        "PL_model =             nlos\n"
        "PL_A1 =                22           % TX-RX 3D dist.-dep. of the LOS-PL before break-point [dB/log10(m)]\n"
        "PL_B =                 28           % Reference LOS-PL in [dB] @ 1 GHz, 1 m TX-RX dist.\n"
        "PL_An =                39.08        % TX-RX 3D dist.-dep. of NLOS-PL [dB/log10(m)]\n"
        "PL_Exn =               1 2.5\n";

    quadriga_lib::conf_file_write(f.fn, scenpar, "mmMAGIC", "nlos", &plpar, &pl_order, &short_names,
                                  "Test reference", &ref_sec, &valid_range, false);
    REQUIRE(file_exists(f.fn));
    const std::string text = slurp(f.fn);

    // Line-by-line comparison, so that a failure names the first line that differs
    const std::vector<std::string> got = lines_of(text), exp = lines_of(expected);
    QD_DIAG("\n# diag: conf_file_write layout, lines written %zu, expected %zu\n", got.size(), exp.size());
    size_t n_diff = 0;
    for (size_t i = 0; i < std::max(got.size(), exp.size()); ++i)
    {
        const std::string g = i < got.size() ? got[i] : "<missing>", e = i < exp.size() ? exp[i] : "<missing>";
        if (g != e && n_diff++ < 5)
            QD_DIAG("line %zu\n  got: %s\n  exp: %s\n", i + 1, g.c_str(), e.c_str());
    }
    CHECK(n_diff == 0);
    CHECK(text == expected); // also pins the line ends and the end of the file
}

TEST_CASE("conf_file_write - numbers are written with the fewest digits that read back exactly")
{
    tmp_name f("test_conf_file_write_numbers.conf");

    // Oracle: value, expected text, typed by hand. Keys of class "any value" so that every number passes.
    struct num_case
    {
        const char *key;
        double value;
        const char *text;
    };
    const num_case cases[] = {
        {"DS_mu", -6.28, "-6.28"},                                     // decimal from a file, 15 digits suffice
        {"DS_omega", 0.1 + 0.2, "0.30000000000000004"},                // needs 17 digits
        {"DS_epsilon", 1.0 / 3.0, "0.3333333333333333"},               // needs 16 digits
        {"DS_zeta", -0.0, "-0"},                                       // negative zero differs from the default 0
        {"DS_alpha", 5e-324, "4.94065645841247e-324"},                 // smallest subnormal
        {"DS_kappa", 1e300, "1e+300"},                                 // exponent with sign
        {"DS_tau", std::numeric_limits<double>::max(), "1.7976931348623157e+308"}, // 15 and 16 digits overflow
        {"DS_beta", INF, "Inf"},                                       // infinity
        {"KF_omega", 6.2e-7, "6.2e-07"},                               // small exponent as in the MIMOSA files
        {"ES_D_mu_min", -INF, "<none>"},                               // the default, not written
        {"AS_D_omega", -INF, "-Inf"}};
    const std::unordered_map<std::string, std::vector<double>> plpar = {{"A", {1.0 / 3.0, -INF, 2.0}}};

    std::unordered_map<std::string, double> scenpar;
    for (const auto &c : cases)
        scenpar[c.key] = c.value;

    quadriga_lib::conf_file_write(f.fn, scenpar, "legacy", "", &plpar);
    REQUIRE(file_exists(f.fn));
    const std::string text = slurp(f.fn);
    const conf_data back = read_back(f.fn);

    QD_DIAG("\n# diag: conf_file_write number text\n");
    QD_DIAG("key,value_17g,text,expected,bits_equal_after_read\n");
    for (const auto &c : cases)
    {
        INFO("key " << c.key);
        const std::string t = value_text(text, c.key);
        const bool same = bits_equal(back.scenpar.at(c.key), c.value);
        QD_DIAG("%s,%.17g,%s,%s,%d\n", c.key, c.value, t.c_str(), c.text, (int)same);
        CHECK(t == c.text);
        CHECK(same);
    }
    const std::string t = value_text(text, "PL_A");
    QD_DIAG("PL_A,-,%s,0.3333333333333333 -Inf 2,-\n", t.c_str());
    CHECK(t == "0.3333333333333333 -Inf 2");
    REQUIRE(back.plpar.count("A") == 1);
    REQUIRE(back.plpar.at("A").size() == 3);
    CHECK(bits_equal(back.plpar.at("A")[0], 1.0 / 3.0));
    CHECK(bits_equal(back.plpar.at("A")[1], -INF));
    CHECK(bits_equal(back.plpar.at("A")[2], 2.0));
}

TEST_CASE("conf_file_write - without write_defaults exactly the keys that differ from their default are written")
{
    tmp_name f("test_conf_file_write_defaults.conf");
    std::string text, d;

    SECTION("empty parameter set: header line only")
    {
        d = round_trip({}, f.fn, false, &text);
        CHECK(d.empty());
        CHECK(text == "% Config file for scenario \"test_conf_file_write_defaults\"\n");
    }

    SECTION("a value equal to its default is not written")
    {
        d = round_trip({{"DS_mu", -6.0}, {"NumClusters", 6.0}, {"ES_D_mu_min", -INF}, {"absTOA_mu", -100.0}}, f.fn,
                       false, &text);
        CHECK(d.empty());
        CHECK(n_assignments(text) == 0);
    }

    // Cases that QuaDRiGa's write_conf_file.m loses, measured in MATLAB at review point A
    struct loss_case
    {
        const char *what;
        std::unordered_map<std::string, double> scenpar;
        std::vector<std::string> written;
    };
    const loss_case cases[] = {
        {"zero with a non-zero default", {{"LNS_ksi", 0.0}, {"DS_lambda", 0.0}}, {"LNS_ksi", "DS_lambda"}},
        {"one small correlation", {{"ds_kf", 5e-4}}, {"ds_kf"}},
        {"absTOA below -99", {{"absTOA_mu", -99.5}, {"absTOA_sigma", 0.3}}, {"absTOA_mu", "absTOA_sigma"}},
        {"LOSonly.conf", {{"NumClusters", 1.0}, {"SF_sigma", 0.0}, {"KF_mu", 0.0}, {"KF_sigma", 0.0}},
         {"NumClusters", "SF_sigma"}},
        {"TwoRayGR.conf",
         {{"NumClusters", 2.0}, {"SF_sigma", 0.0}, {"KF_mu", 0.0}, {"KF_sigma", 0.0}, {"DS_sigma", 0.0},
          {"AS_D_sigma", 0.0}, {"AS_A_sigma", 0.0}, {"ES_D_sigma", 0.0}, {"ES_A_sigma", 0.0}, {"XPR_mu", 100.0},
          {"XPR_sigma", 0.0}, {"GR_enabled", 1.0}},
         {"NumClusters", "SF_sigma", "DS_sigma", "AS_D_sigma", "AS_A_sigma", "ES_D_sigma", "ES_A_sigma",
          "XPR_mu", "XPR_sigma", "GR_enabled"}},
        {"a v19 key alone", {{"PolVar_sigma", 3.0}}, {"PolVar_sigma"}}};

    SECTION("values that QuaDRiGa's writer loses")
    {
        QD_DIAG("\n# diag: conf_file_write keys written without write_defaults\n");
        QD_DIAG("case,n_written,n_expected,differences\n");
        for (const auto &c : cases)
        {
            INFO("case " << c.what);
            d = round_trip(c.scenpar, f.fn, false, &text);
            QD_DIAG("%s,%d,%zu,%s\n", c.what, n_assignments(text), c.written.size(), d.empty() ? "none" : d.c_str());
            CHECK(d.empty());
            CHECK(n_assignments(text) == (int)c.written.size());
            for (const auto &k : c.written)
            {
                INFO("key " << k);
                CHECK(n_lines(text, k) == 1);
            }
        }
    }

    SECTION("SubpathMethod is written only if it is not legacy")
    {
        quadriga_lib::conf_file_write(f.fn, {}, "legacy");
        CHECK(n_lines(slurp(f.fn), "SubpathMethod") == 0);
        quadriga_lib::conf_file_write(f.fn, {}, "Laplacian");
        CHECK(value_text(slurp(f.fn), "SubpathMethod") == "Laplacian");
        CHECK(read_back(f.fn).subpath_method == "Laplacian");
    }
}

TEST_CASE("conf_file_write - with write_defaults every key is written once")
{
    tmp_name f("test_conf_file_write_all.conf");
    std::string text;

    // A default set and one with a changed value give the same number of lines
    for (const auto &scenpar : {std::unordered_map<std::string, double>{},
                                std::unordered_map<std::string, double>{{"DS_mu", -7.5}, {"NF_k1", 3.0}}})
    {
        const std::string d = round_trip(scenpar, f.fn, true, &text);
        QD_DIAG("\n# diag: conf_file_write write_defaults = true, %zu given keys: %d assignments, %d lines\n",
                scenpar.size(), n_assignments(text), (int)lines_of(text).size());
        CHECK(d.empty());
        CHECK(n_assignments(text) == 166); // 165 numeric keys and SubpathMethod
        for (const auto &k : KEYS)
        {
            INFO("key " << k);
            CHECK(n_lines(text, k) == 1);
        }
        CHECK(n_lines(text, "SubpathMethod") == 1);
        CHECK(value_text(text, "SubpathMethod") == "legacy");
    }

    // All sections except the path loss (no PL input) are present, in the decided order
    const std::vector<std::string> titles = {
        "% Large scale distributions", "% Large-Scale fading decorrelation distances", "% Inter-parameter correlations",
        "% Model parameters", "% EoD offset (elevation of departure, TR: ZOD offset)",
        "% Decorrelation distance for the small-scale fading spatial consistency",
        "% Parameters for the ground reflection extension",
        "% Absolute time of arrival offset", "% Optional modelling components"};
    size_t pos = 0;
    for (const auto &t : titles)
    {
        INFO("section " << t);
        const size_t p = text.find("\n" + t + "\n", pos);
        CHECK(p != std::string::npos);
        if (p != std::string::npos)
            pos = p;
    }
    CHECK(text.find("% Path-loss model") == std::string::npos);
}

TEST_CASE("conf_file_write - path-loss comments and formulas depend on the model")
{
    tmp_name f("test_conf_file_write_pl.conf");

    // Oracle: comments typed from write_conf_file.m, lines 568 to 692 ("Elevaion" corrected)
    const std::unordered_map<std::string, std::vector<double>> plpar = {
        {"A", {1.0}}, {"A1", {2.0}}, {"A2", {3.0}}, {"B", {4.0}}, {"C", {5.0}}, {"D", {6.0}}, {"hE", {7.0}},
        {"Bn", {8.0}}, {"usePLa", {1.0}}, {"Xyz", {9.0}}};
    const std::string A_def = "TX-RX 3D dist.-dep. of the PL [dB/log10(m)]";
    const std::string A1_los = "TX-RX 3D dist.-dep. of the LOS-PL before break-point [dB/log10(m)]";
    const std::string A1_def = "TX-RX 3D dist.-dep. of the PL before break-point [dB/log10(m)]";
    const std::string A2_los = "TX-RX 3D dist.-dep. of the LOS-PL after break-point [dB/log10(m)]";
    const std::string A2_def = "TX-RX 3D dist.-dep. of the PL after break-point [dB/log10(m)]";
    const std::string B_los = "Reference LOS-PL in [dB] @ 1 GHz, 1 m TX-RX dist.";
    const std::string B_def = "Reference PL in [dB] @ 1 GHz, 1 m TX-RX dist.";
    const std::string C_los = "Freq.-dep. of the LOS-PL in [dB/log10(GHz)]";
    const std::string C_def = "Freq.-dep. of the PL in [dB/log10(GHz)]";
    const std::string D_los = "TX-RX 3D dist.-dep. of LOS-PL [dB/m]";
    const std::string D_sat = "Elevation angle dep. of PL [dB/rad]";
    const std::string D_def = "TX-RX 3D dist.-dep. of PL [dB/m]";

    struct pl_case
    {
        const char *model;
        std::string A, A1, A2, B, C, D;
        bool hata, dual, triple, nlos; // formula blocks
    };
    const pl_case cases[] = {
        {"nlos", A_def, A1_los, A2_los, B_los, C_los, D_los, false, true, false, true},
        {"dual_slope", A_def, A1_def, A2_def, B_def, C_def, D_def, false, true, false, false},
        {"constant", "Reference PL in [dB]", A1_def, A2_def, B_def, C_def, D_def, false, false, false, false},
        {"satellite", A_def, A1_def, A2_def, B_def, C_def, D_sat, false, false, false, false},
        {"logdist", A_def, A1_def, A2_def, B_def, C_def, D_def, true, false, false, false},
        {"tripple_slope", A_def, A1_def, A2_def, B_def, C_def, D_def, false, false, true, false},
        {"winner_los", A_def, A1_def, A2_def, B_def, C_def, D_def, false, false, false, false},
        {"", A_def, A1_def, A2_def, B_def, C_def, D_def, false, false, false, false}};

    QD_DIAG("\n# diag: conf_file_write path-loss comments per model\n");
    QD_DIAG("model,A,A1,D,formula_blocks\n");
    for (const auto &c : cases)
    {
        INFO("model '" << c.model << "'");
        quadriga_lib::conf_file_write(f.fn, {}, "legacy", c.model, &plpar);
        REQUIRE(file_exists(f.fn));
        const std::string text = slurp(f.fn);
        QD_DIAG("%s,%s,%s,%s,%d%d%d%d\n", c.model, comment_text(text, "PL_A").c_str(),
                comment_text(text, "PL_A1").c_str(), comment_text(text, "PL_D").c_str(),
                (int)(text.find("% Formula for Hata") != std::string::npos),
                (int)(text.find("% Formula for dual-slope") != std::string::npos),
                (int)(text.find("% Formula for triple-slope") != std::string::npos),
                (int)(text.find("% Formula for 3GPP NLOS") != std::string::npos));
        CHECK(comment_text(text, "PL_A") == c.A);
        CHECK(comment_text(text, "PL_A1") == c.A1);
        CHECK(comment_text(text, "PL_A2") == c.A2);
        CHECK(comment_text(text, "PL_B") == c.B);
        CHECK(comment_text(text, "PL_C") == c.C);
        CHECK(comment_text(text, "PL_D") == c.D);
        CHECK(comment_text(text, "PL_hE") == "Environment height in [m]");
        CHECK(comment_text(text, "PL_Bn") == "Reference NLOS-PL in [dB] @ 1 GHz, 1 m TX-RX dist.");
        CHECK(comment_text(text, "PL_usePLa") == "Enable/disable attenuation by atmospheric gases");
        CHECK(comment_text(text, "PL_Xyz") == "<none>");
        CHECK((text.find("% Formula for Hata pathloss model:") != std::string::npos) == c.hata);
        CHECK((text.find("% Formula for dual-slope (LOS) pathloss model:") != std::string::npos) == c.dual);
        CHECK((text.find("% Formula for triple-slope (LOS+GR) pathloss model:") != std::string::npos) == c.triple);
        CHECK((text.find("% Formula for 3GPP NLOS pathloss model:") != std::string::npos) == c.nlos);
        CHECK(n_lines(text, "PL_model") == (c.model[0] == 0 ? 0 : 1));
        CHECK(read_back(f.fn).pl_model == c.model);
    }
}

TEST_CASE("conf_file_write - path-loss keys are written in the given order, else sorted by name")
{
    tmp_name f("test_conf_file_write_order.conf");
    const std::unordered_map<std::string, std::vector<double>> plpar = {{"Bn", {1.0}}, {"A", {2.0}}, {"hE", {3.0}}};
    const std::vector<std::string> order = {"hE", "A", "Bn"};

    quadriga_lib::conf_file_write(f.fn, {}, "legacy", "logdist", &plpar, &order);
    CHECK(read_back(f.fn).pl_order == order);

    quadriga_lib::conf_file_write(f.fn, {}, "legacy", "logdist", &plpar);
    CHECK(read_back(f.fn).pl_order == std::vector<std::string>({"A", "Bn", "hE"})); // byte-wise: A < B < h

    // Model only: the section has the model line and no other key
    quadriga_lib::conf_file_write(f.fn, {}, "legacy", "constant");
    REQUIRE(file_exists(f.fn));
    const conf_data d = read_back(f.fn);
    CHECK(d.pl_model == "constant");
    CHECK(d.plpar.empty());
    CHECK(n_assignments(slurp(f.fn)) == 1);
}

TEST_CASE("conf_file_write - round trip of the conf files in data/config is exact")
{
    REQUIRE(std::filesystem::is_directory(LIB_CONFIG_DIR));
    std::vector<std::string> skipped;
    const int n = round_trip_dir(LIB_CONFIG_DIR, skipped);
    CHECK(n == 23);
    CHECK(skipped.empty());
}

TEST_CASE("conf_file_write - an existing file is overwritten, a failed call changes no file")
{
    tmp_name f("test_conf_file_write_exist.conf");

    put(f.fn, "garbage that is not a conf file\n");
    quadriga_lib::conf_file_write(f.fn, {{"DS_mu", -7.0}});
    CHECK(slurp(f.fn).find("garbage") == std::string::npos);
    CHECK(read_back(f.fn).scenpar.at("DS_mu") == -7.0);

    put(f.fn, "garbage\n");
    CHECK_THROWS_AS(quadriga_lib::conf_file_write(f.fn, {{"DS_mu", QNAN}}), std::invalid_argument);
    CHECK(slurp(f.fn) == "garbage\n");

    std::remove(f.fn.c_str());
    CHECK_THROWS_AS(quadriga_lib::conf_file_write(f.fn, {{"NumClusters", 0.0}}), std::invalid_argument);
    CHECK(!file_exists(f.fn));
}

TEST_CASE("conf_file_write - input validation")
{
    tmp_name f("test_conf_file_write_valid.conf");
    const std::unordered_map<std::string, double> sp = {{"DS_mu", -7.0}};
    const std::unordered_map<std::string, std::vector<double>> pl = {{"A", {1.0}}, {"B", {2.0, 3.0}}};
    const std::vector<std::string> order = {"B", "A"}, names = {"Xy"}, refs = {"a", "", "c"};
    const arma::mat vr(4, 2, arma::fill::ones);

    // Every invalid call must throw and leave no file behind
    auto rejects = [&](auto call)
    {
        std::remove(f.fn.c_str());
        CHECK_THROWS_AS(call(), std::invalid_argument);
        CHECK(!file_exists(f.fn));
    };

    SECTION("valid baseline")
    {
        CHECK_NOTHROW(quadriga_lib::conf_file_write(f.fn, sp, "Laplacian", "logdist", &pl, &order, &names, "ref",
                                                    &refs, &vr, true));
        CHECK(file_exists(f.fn));
        CHECK_NOTHROW(quadriga_lib::conf_file_write(f.fn, sp));
    }

    SECTION("fn")
    {
        rejects([&]() { quadriga_lib::conf_file_write("", sp); });
        CHECK_THROWS_AS(quadriga_lib::conf_file_write("test_conf_file_write_no_such_dir/x.conf", sp), std::runtime_error);
        CHECK(!std::filesystem::exists("test_conf_file_write_no_such_dir"));

        // B2: a line break in the file name would end the header comment
        for (const char *name : {"test_conf_file_write_x\nKF_mu = 5 %.conf", "test_conf_file_write_x\rKF_mu = 5 %.conf"})
        {
            tmp_name t(name);
            CHECK_THROWS_AS(quadriga_lib::conf_file_write(name, sp), std::invalid_argument);
            CHECK(!file_exists(name));
        }
    }

    SECTION("scenpar")
    {
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, {{"DS_mux", 1.0}}); });       // unknown key
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, {{"GR_epsilon", 1.0}}); });   // not supported
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, {{"DS_mu", QNAN}}); });       // NaN
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, {{"NumClusters", 2.5}}); });  // class int > 0
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, {{"ds_kf", 1.5}}); });        // class correlation
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, {{"DS_lambda", INF}}); });    // class finite >= 0
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, {{"NF_alpha", 1.0}}); });     // NF_alpha without NF_beta
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, {{"NumClusters", 1001.0}}); }); // Q11: bound 1000
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, {{"NumSubPaths", 1001.0}}); });
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, {{"NF_k1", 1001.0}}); });
        CHECK_NOTHROW(quadriga_lib::conf_file_write(f.fn, {{"NumClusters", 1000.0}, {"NumSubPaths", 1000.0},
                                                           {"NumClusters_min", 1000.0}, {"NF_k1", 1000.0}}));
    }

    SECTION("subpath_method")
    {
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, sp, "bla"); });
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, sp, ""); });
    }

    SECTION("pl_model")
    {
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, sp, "legacy", "two words"); });
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, sp, "legacy", "a-b"); });
    }

    SECTION("plpar")
    {
        using plmap = std::unordered_map<std::string, std::vector<double>>;
        const plmap model = {{"model", {1.0}}}, space = {{"A B", {1.0}}}, empty_name = {{"", {1.0}}},
                    no_value = {{"A", {}}}, nan = {{"A", {1.0, QNAN}}};
        for (const plmap *p : {&model, &space, &empty_name, &no_value, &nan})
            rejects([&]() { quadriga_lib::conf_file_write(f.fn, sp, "legacy", "logdist", p); });
    }

    SECTION("pl_order")
    {
        const std::vector<std::string> missing = {"A"}, extra = {"A", "B", "C"}, twice = {"A", "A"};
        for (const auto *o : {&missing, &extra, &twice})
            rejects([&]() { quadriga_lib::conf_file_write(f.fn, sp, "legacy", "logdist", &pl, o); });
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, sp, "legacy", "logdist", nullptr, &order); });
    }

    SECTION("short_names")
    {
        const std::vector<std::string> space = {"A b"}, empty = {""}, twice = {"Ab", "Cd", "Ab"}; // Q3: distinct
        for (const auto *s : {&space, &empty, &twice})
            rejects([&]() { quadriga_lib::conf_file_write(f.fn, sp, "legacy", "", nullptr, nullptr, s); });
        const std::vector<std::string> case_differs = {"Ab", "AB"};
        CHECK_NOTHROW(quadriga_lib::conf_file_write(f.fn, sp, "legacy", "", nullptr, nullptr, &case_differs));
    }

    SECTION("ref and ref_sec")
    {
        rejects([&]() { quadriga_lib::conf_file_write(f.fn, sp, "legacy", "", nullptr, nullptr, nullptr, "a\nb"); });
        const std::vector<std::string> cr = {"a\rb"}, eleven(11, "x");
        for (const auto *r : {&cr, &eleven})
            rejects([&]() { quadriga_lib::conf_file_write(f.fn, sp, "legacy", "", nullptr, nullptr, nullptr, "", r); });
        const std::vector<std::string> ten(10, "x");
        CHECK_NOTHROW(quadriga_lib::conf_file_write(f.fn, sp, "legacy", "", nullptr, nullptr, nullptr, "", &ten));
    }

    SECTION("valid_range")
    {
        const arma::mat r32(3, 2, arma::fill::ones), r24(2, 4, arma::fill::ones), r42(4, 2, arma::fill::ones), r0;
        for (const auto *r : {&r32, &r24})
            rejects([&]() { quadriga_lib::conf_file_write(f.fn, sp, "legacy", "", nullptr, nullptr, nullptr, "", nullptr, r); });
        CHECK_NOTHROW(quadriga_lib::conf_file_write(f.fn, sp, "legacy", "", nullptr, nullptr, nullptr, "", nullptr, &r0));
        CHECK_NOTHROW(quadriga_lib::conf_file_write(f.fn, sp, "legacy", "", nullptr, nullptr, nullptr, "", nullptr, &r42));
    }
}

TEST_CASE("conf_file_write - the ESD overview line shows the distance and height terms and the minimum")
{
    // Oracle: the lines typed by hand from the rule of Q10: the terms of ES_D_mu_A ... ES_D_mu_D follow the
    // log10 terms with the same sign style; max( mu_min, mean ) only if ES_D_mu_min is not -Inf; with
    // LSP_fc_min > 0 the frequency is written as max( fGHz, fc_min ), in the mean and in the STD term.
    tmp_name f("test_conf_file_write_esd.conf");

    SECTION("height terms, no minimum")
    {
        quadriga_lib::conf_file_write(f.fn, {{"ES_D_mu", 1.2}, {"ES_D_gamma", 0.1}, {"ES_D_mu_C", 0.05},
                                             {"ES_D_mu_D", -0.03}});
        const std::string text = slurp(f.fn);
        QD_DIAG("\n# diag: conf_file_write ESD overview\n%s\n%s\n", overview_line(text, "ESD").c_str(),
                overview_line(text, "DS").c_str());
        CHECK(overview_line(text, "ESD") ==
              "% ESD = 1.2 + 0.1 * log10( fGHz ) + 0.05 * abs( hUT - hBS ) - 0.03 * max( hUT - hBS, 0 ) + Xesd * ( 0.2 )");
        CHECK(overview_line(text, "DS") == "% DS  = -6 + Xds * ( 0.5 )");
    }

    SECTION("minimum, distance term and lower frequency limit")
    {
        quadriga_lib::conf_file_write(f.fn, {{"ES_D_mu", 1.2}, {"ES_D_gamma", 0.1}, {"ES_D_delta", -0.02},
                                             {"ES_D_mu_A", 0.5}, {"ES_D_mu_min", 0.25}, {"LSP_fc_min", 2.5}});
        const std::string text = slurp(f.fn);
        QD_DIAG("%s\n", overview_line(text, "ESD").c_str());
        CHECK(overview_line(text, "ESD") == "% ESD = max( 0.25, 1.2 + 0.1 * log10( max( fGHz, 2.5 ) ) + 0.5 * d2D / "
                                            "1000 ) + Xesd * ( 0.2 - 0.02 * log10( max( fGHz, 2.5 ) ) )");
        CHECK(overview_line(text, "ESA") == "% ESA = 0.3 + Xesa * ( 0.2 )");
    }

    SECTION("a minimum without other terms")
    {
        quadriga_lib::conf_file_write(f.fn, {{"ES_D_mu_min", -0.0}, {"ES_D_mu_B", 0.2}});
        CHECK(overview_line(slurp(f.fn), "ESD") == "% ESD = max( -0, 0.3 + 0.2 * hUT ) + Xesd * ( 0.2 )");
    }
}

TEST_CASE("conf_file_write - a failed write leaves an existing file unchanged")
{
#if !defined(__linux__)
    SKIP("needs Linux: file size limit and /dev/full");
#else
    // Decision B1: the text goes to a temporary file in the same directory, which is renamed over the target
    // only after it was closed without error. A failed call leaves an existing file unchanged and no temporary
    // file behind. The write error is provoked by a file size limit of 2048 bytes (SIGXFSZ ignored, so that
    // write() fails with EFBIG); the limit is lifted before any assertion, since it also applies to a
    // redirected stdout.
    const std::string dir = "test_conf_file_write_fail_dir";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directory(dir);
    const std::string fn = dir + "/test_conf_file_write_fail.conf";
    const std::string old_content = "% older content that must survive\nDS_mu = -7\n";

    // Number of files in 'dir' whose name contains ".tmp"
    auto n_tmp = [&dir]()
    {
        int n = 0;
        for (const auto &e : std::filesystem::directory_iterator(dir))
            n += e.path().filename().string().find(".tmp") != std::string::npos ? 1 : 0;
        return n;
    };

    const std::unordered_map<std::string, double> sp = {{"DS_mu", -7.5}, {"KF_mu", 3.0}, {"XPR_mu", 8.0}};
    std::unordered_map<std::string, std::vector<double>> pl; // 60 keys: about 3.4 KB, less than the stream buffer
    for (int i = 0; i < 60; ++i)
        pl["K" + std::to_string(i)] = {1.0 / 3.0};

    QD_DIAG("\n# diag: conf_file_write under a file size limit of 2048 bytes\n");
    QD_DIAG("write_defaults,text_bytes,runtime_error,other_error,old_content_kept,n_tmp\n");
    for (bool wd : {false, true})
    {
        INFO("write_defaults " << wd);

        // Size of the text without a limit
        quadriga_lib::conf_file_write(fn, sp, "legacy", "", &pl, nullptr, nullptr, "", nullptr, nullptr, wd);
        const size_t n_bytes = slurp(fn).size();
        REQUIRE(n_bytes > 2048);

        put(fn, old_content);
        bool runtime_error = false, other_error = false;
        {
            struct rlimit old_limit, limit;
            getrlimit(RLIMIT_FSIZE, &old_limit);
            limit = old_limit;
            limit.rlim_cur = 2048;
            void (*old_handler)(int) = std::signal(SIGXFSZ, SIG_IGN);
            setrlimit(RLIMIT_FSIZE, &limit);
            try
            {
                quadriga_lib::conf_file_write(fn, sp, "legacy", "", &pl, nullptr, nullptr, "", nullptr, nullptr, wd);
            }
            catch (const std::runtime_error &)
            {
                runtime_error = true;
            }
            catch (...)
            {
                other_error = true;
            }
            setrlimit(RLIMIT_FSIZE, &old_limit);
            std::signal(SIGXFSZ, old_handler);
        }
        const bool kept = slurp(fn) == old_content;
        QD_DIAG("%d,%zu,%d,%d,%d,%d\n", (int)wd, n_bytes, (int)runtime_error, (int)other_error, (int)kept, n_tmp());
        CHECK(runtime_error);
        CHECK(!other_error);
        CHECK(kept);
        CHECK(n_tmp() == 0);
    }

    // A device without space: the temporary file cannot be created next to it
    if (std::filesystem::exists("/dev/full") && geteuid() != 0)
    {
        CHECK_THROWS_AS(quadriga_lib::conf_file_write("/dev/full", sp), std::runtime_error);
        CHECK_THROWS_AS(quadriga_lib::conf_file_write("/dev/full", sp, "legacy", "", nullptr, nullptr, nullptr, "",
                                                      nullptr, nullptr, true),
                        std::runtime_error);
    }

    // A read-only existing file is not replaced (skipped where the permission is not enforced)
    const std::string fn_ro = dir + "/test_conf_file_write_ro.conf";
    put(fn_ro, old_content);
    std::filesystem::permissions(fn_ro, std::filesystem::perms::owner_read);
    if (!std::ofstream(fn_ro, std::ios::app).is_open())
    {
        CHECK_THROWS_AS(quadriga_lib::conf_file_write(fn_ro, sp), std::runtime_error);
        CHECK(slurp(fn_ro) == old_content);
        CHECK(n_tmp() == 0);
    }
    std::filesystem::permissions(fn_ro, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);

    // A read-only directory: no new file (skipped where the permission is not enforced)
    const std::string dir_ro = dir + "/ro";
    std::filesystem::create_directory(dir_ro);
    std::filesystem::permissions(dir_ro, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
    if (!std::ofstream(dir_ro + "/probe").is_open())
    {
        CHECK_THROWS_AS(quadriga_lib::conf_file_write(dir_ro + "/x.conf", sp), std::runtime_error);
        CHECK(!std::filesystem::exists(dir_ro + "/x.conf"));
    }
    std::filesystem::permissions(dir_ro, std::filesystem::perms::owner_all);

    // An existing file is replaced and keeps its permission bits; a new file gets those of std::ofstream
    const std::string fn_perm = dir + "/test_conf_file_write_perm.conf";
    const auto rw = std::filesystem::perms::owner_read | std::filesystem::perms::owner_write;
    put(fn_perm, old_content);
    std::filesystem::permissions(fn_perm, rw);
    CHECK_NOTHROW(quadriga_lib::conf_file_write(fn_perm, sp));
    CHECK(std::filesystem::status(fn_perm).permissions() == rw);
    CHECK(read_back(fn_perm).scenpar.at("XPR_mu") == 8.0);

    const std::string fn_new = dir + "/test_conf_file_write_new.conf", fn_ref = dir + "/reference_file";
    put(fn_ref, "x");
    CHECK_NOTHROW(quadriga_lib::conf_file_write(fn_new, sp));
    CHECK(std::filesystem::status(fn_new).permissions() == std::filesystem::status(fn_ref).permissions());

    // A symbolic link is written through: the link stays, its target gets the text
    std::error_code ec;
    std::filesystem::create_symlink("test_conf_file_write_target.conf", dir + "/link.conf", ec);
    if (!ec)
    {
        put(dir + "/test_conf_file_write_target.conf", old_content);
        CHECK_NOTHROW(quadriga_lib::conf_file_write(dir + "/link.conf", sp));
        CHECK(std::filesystem::is_symlink(dir + "/link.conf"));
        CHECK(read_back(dir + "/test_conf_file_write_target.conf").scenpar.at("XPR_mu") == 8.0);
    }

    // A symbolic link to itself cannot be resolved
    std::filesystem::create_symlink("loop.conf", dir + "/loop.conf", ec);
    if (!ec)
        CHECK_THROWS_AS(quadriga_lib::conf_file_write(dir + "/loop.conf", sp), std::runtime_error);
    CHECK(n_tmp() == 0);

    std::filesystem::remove_all(dir);
#endif
}

TEST_CASE("conf_file_write - numbers do not depend on the global locale")
{
    // Oracle: the same file read and written in the classic locale; values compared bit for bit, texts exactly.
    // The locale must have a decimal comma, so that a locale-dependent conversion would read '-6.28' as -6.
    tmp_name f_in("test_conf_file_write_locale_in.conf"), f_out("test_conf_file_write_locale_out.conf");
    put(f_in.fn, "DS_mu = -6.28\nDS_lambda = 1.5e3\nDS_sigma = .25\nPL_A = -6.28 1.5e3 .25\n");

    const conf_data c = read_back(f_in.fn);
    write_data(f_out.fn, c, true);
    const std::string text_c = slurp(f_out.fn);
    CHECK(c.scenpar.at("DS_mu") == -6.28);
    CHECK(c.scenpar.at("DS_lambda") == 1500.0);
    CHECK(c.scenpar.at("DS_sigma") == 0.25);

    // Switch both the C and the C++ global locale; restored when 'guard' goes out of scope, also on an exception
    struct locale_guard
    {
        std::string c_name = std::setlocale(LC_ALL, nullptr);
        std::locale cpp_locale;
        ~locale_guard()
        {
            std::setlocale(LC_ALL, c_name.c_str());
            std::locale::global(cpp_locale);
        }
    };

    std::string used, decimal_point, text_l;
    conf_data l;
    {
        locale_guard guard;
        for (const char *name : {"de_DE.UTF-8", "de_DE.utf8", "en_DK.UTF-8", "en_DK.utf8", "German_Germany.1252"})
        {
            try
            {
                std::locale loc(name);
                if (std::setlocale(LC_ALL, name) == nullptr)
                    continue;
                std::locale::global(loc);
                used = name;
                break;
            }
            catch (const std::runtime_error &)
            {
            }
        }
        if (!used.empty())
        {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.1f", 1.5);
            decimal_point = buf;
            l = read_back(f_in.fn);
            write_data(f_out.fn, l, true);
            text_l = slurp(f_out.fn);
        }
    }
    if (used.empty())
        SKIP("no locale with a decimal comma is installed");

    QD_DIAG("\n# diag: conf_file_write locale '%s', 1.5 printed as '%s', texts equal %d\n", used.c_str(),
            decimal_point.c_str(), (int)(text_l == text_c));
    CHECK(decimal_point == "1,5"); // the locale is a decimal-comma locale
    CHECK(differences(c, l).empty());
    CHECK(text_l == text_c);
    CHECK(text_c.find("-6.28") != std::string::npos);
}
