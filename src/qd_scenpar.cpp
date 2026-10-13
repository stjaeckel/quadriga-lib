// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Validation classes of the scenario parameters
enum qd_scenpar_class : unsigned char
{
    QD_ANY,        // Any value except NaN
    QD_REAL,       // Any value except NaN; same rule as QD_ANY, kept apart to mirror the MATLAB reference
    QD_NONNEG,     // Finite and >= 0
    QD_POS,        // Finite and > 0
    QD_INT_POS,    // Integer and > 0
    QD_INT_NONNEG, // Integer and >= 0
    QD_CORR,       // In [-1, 1]
    QD_UNIT        // In [0, 1]
};

// One entry of the key table
struct qd_scenpar_key
{
    const char *name;     // Key name as written in the conf file
    double def;           // Default value
    qd_scenpar_class cls; // Validation class
    signed char row, col; // Position in the upper triangle of the LSP cross-correlation matrix, -1 = none
};

static const double QD_INF = std::numeric_limits<double>::infinity();

// Key table: the single source of truth for names, canonical order, defaults and validation classes
// - 138 numeric legacy keys in the order of QuaDRiGa's 'set_scenario_table.m', then the 27 keys for TR 38.901 v19.5.0
// - The legacy key 'GR_epsilon' is not supported: a real value has no use, the relative permittivity of the ground is complex
// - 'SubpathMethod' is the only string key and is not part of the table; its legacy position is field 107
// - LSP order of the cross-correlation matrix: DS, KF, SF, ASD, ASA, ESD, ESA, XPR
static const qd_scenpar_key qd_scenpar_keys[] = {
    {"NumClusters", 6.0, QD_INT_POS, -1, -1},
    {"NumSubPaths", 20.0, QD_INT_POS, -1, -1},
    {"DS_mu", -6.0, QD_REAL, -1, -1},
    {"DS_sigma", 0.5, QD_REAL, -1, -1},
    {"DS_lambda", 20.0, QD_NONNEG, -1, -1},
    {"DS_omega", 0.0, QD_ANY, -1, -1},
    {"DS_gamma", 0.0, QD_REAL, -1, -1},
    {"DS_epsilon", 0.0, QD_ANY, -1, -1},
    {"DS_zeta", 0.0, QD_ANY, -1, -1},
    {"DS_alpha", 0.0, QD_ANY, -1, -1},
    {"DS_delta", 0.0, QD_REAL, -1, -1},
    {"DS_kappa", 0.0, QD_ANY, -1, -1},
    {"DS_tau", 0.0, QD_ANY, -1, -1},
    {"DS_beta", 0.0, QD_ANY, -1, -1},
    {"AS_D_mu", 1.0, QD_REAL, -1, -1},
    {"AS_D_sigma", 0.2, QD_REAL, -1, -1},
    {"AS_D_lambda", 20.0, QD_NONNEG, -1, -1},
    {"AS_D_omega", 0.0, QD_ANY, -1, -1},
    {"AS_D_gamma", 0.0, QD_REAL, -1, -1},
    {"AS_D_epsilon", 0.0, QD_ANY, -1, -1},
    {"AS_D_zeta", 0.0, QD_ANY, -1, -1},
    {"AS_D_alpha", 0.0, QD_ANY, -1, -1},
    {"AS_D_delta", 0.0, QD_REAL, -1, -1},
    {"AS_D_kappa", 0.0, QD_ANY, -1, -1},
    {"AS_D_tau", 0.0, QD_ANY, -1, -1},
    {"AS_D_beta", 0.0, QD_ANY, -1, -1},
    {"AS_A_mu", 1.0, QD_REAL, -1, -1},
    {"AS_A_sigma", 0.2, QD_REAL, -1, -1},
    {"AS_A_lambda", 20.0, QD_NONNEG, -1, -1},
    {"AS_A_omega", 0.0, QD_ANY, -1, -1},
    {"AS_A_gamma", 0.0, QD_REAL, -1, -1},
    {"AS_A_epsilon", 0.0, QD_ANY, -1, -1},
    {"AS_A_zeta", 0.0, QD_ANY, -1, -1},
    {"AS_A_alpha", 0.0, QD_ANY, -1, -1},
    {"AS_A_delta", 0.0, QD_REAL, -1, -1},
    {"AS_A_kappa", 0.0, QD_ANY, -1, -1},
    {"AS_A_tau", 0.0, QD_ANY, -1, -1},
    {"AS_A_beta", 0.0, QD_ANY, -1, -1},
    {"ES_D_mu", 0.3, QD_REAL, -1, -1},
    {"ES_D_sigma", 0.2, QD_REAL, -1, -1},
    {"ES_D_lambda", 20.0, QD_NONNEG, -1, -1},
    {"ES_D_omega", 0.0, QD_ANY, -1, -1},
    {"ES_D_gamma", 0.0, QD_REAL, -1, -1},
    {"ES_D_epsilon", 0.0, QD_ANY, -1, -1},
    {"ES_D_zeta", 0.0, QD_ANY, -1, -1},
    {"ES_D_alpha", 0.0, QD_ANY, -1, -1},
    {"ES_D_delta", 0.0, QD_REAL, -1, -1},
    {"ES_D_kappa", 0.0, QD_ANY, -1, -1},
    {"ES_D_tau", 0.0, QD_ANY, -1, -1},
    {"ES_D_beta", 0.0, QD_ANY, -1, -1},
    {"ES_D_mu_A", 0.0, QD_REAL, -1, -1},
    {"ES_D_mu_min", -QD_INF, QD_REAL, -1, -1},
    {"ES_A_mu", 0.3, QD_REAL, -1, -1},
    {"ES_A_sigma", 0.2, QD_REAL, -1, -1},
    {"ES_A_lambda", 20.0, QD_NONNEG, -1, -1},
    {"ES_A_omega", 0.0, QD_ANY, -1, -1},
    {"ES_A_gamma", 0.0, QD_REAL, -1, -1},
    {"ES_A_epsilon", 0.0, QD_ANY, -1, -1},
    {"ES_A_zeta", 0.0, QD_ANY, -1, -1},
    {"ES_A_alpha", 0.0, QD_ANY, -1, -1},
    {"ES_A_delta", 0.0, QD_REAL, -1, -1},
    {"ES_A_kappa", 0.0, QD_ANY, -1, -1},
    {"ES_A_tau", 0.0, QD_ANY, -1, -1},
    {"ES_A_beta", 0.0, QD_ANY, -1, -1},
    {"SF_sigma", 8.0, QD_REAL, -1, -1},
    {"SF_lambda", 20.0, QD_NONNEG, -1, -1},
    {"SF_omega", 0.0, QD_ANY, -1, -1},
    {"SF_delta", 0.0, QD_REAL, -1, -1},
    {"SF_kappa", 0.0, QD_ANY, -1, -1},
    {"SF_tau", 0.0, QD_ANY, -1, -1},
    {"SF_beta", 0.0, QD_ANY, -1, -1},
    {"KF_mu", 0.0, QD_REAL, -1, -1},
    {"KF_sigma", 0.0, QD_REAL, -1, -1},
    {"KF_lambda", 20.0, QD_NONNEG, -1, -1},
    {"KF_omega", 0.0, QD_ANY, -1, -1},
    {"KF_gamma", 0.0, QD_REAL, -1, -1},
    {"KF_epsilon", 0.0, QD_ANY, -1, -1},
    {"KF_zeta", 0.0, QD_ANY, -1, -1},
    {"KF_alpha", 0.0, QD_ANY, -1, -1},
    {"KF_delta", 0.0, QD_REAL, -1, -1},
    {"KF_kappa", 0.0, QD_ANY, -1, -1},
    {"KF_tau", 0.0, QD_ANY, -1, -1},
    {"KF_beta", 0.0, QD_ANY, -1, -1},
    {"XPR_mu", 0.0, QD_REAL, -1, -1},
    {"XPR_sigma", 10.0, QD_REAL, -1, -1},
    {"XPR_lambda", 20.0, QD_NONNEG, -1, -1},
    {"XPR_omega", 0.0, QD_ANY, -1, -1},
    {"XPR_gamma", 0.0, QD_REAL, -1, -1},
    {"XPR_epsilon", 0.0, QD_ANY, -1, -1},
    {"XPR_zeta", 0.0, QD_ANY, -1, -1},
    {"XPR_alpha", 0.0, QD_ANY, -1, -1},
    {"XPR_delta", 0.0, QD_REAL, -1, -1},
    {"XPR_kappa", 0.0, QD_ANY, -1, -1},
    {"XPR_tau", 0.0, QD_ANY, -1, -1},
    {"XPR_beta", 0.0, QD_ANY, -1, -1},
    {"r_DS", 2.0, QD_POS, -1, -1},
    {"PerClusterDS", 0.0, QD_NONNEG, -1, -1},
    {"PerClusterDS_gamma", 0.0, QD_REAL, -1, -1},
    {"PerClusterDS_min", 0.0, QD_NONNEG, -1, -1},
    {"PerClusterAS_D", 0.0, QD_NONNEG, -1, -1},
    {"PerClusterAS_A", 0.0, QD_NONNEG, -1, -1},
    {"PerClusterES_D", 0.0, QD_NONNEG, -1, -1},
    {"PerClusterES_A", 0.0, QD_NONNEG, -1, -1},
    {"absTOA_mu", -100.0, QD_REAL, -1, -1},
    {"absTOA_sigma", 0.0, QD_REAL, -1, -1},
    {"absTOA_lambda", 0.0, QD_NONNEG, -1, -1},
    {"LOS_scatter_radius", 0.0, QD_NONNEG, -1, -1},
    {"SC_lambda", 0.0, QD_NONNEG, -1, -1},
    {"LNS_ksi", 3.0, QD_NONNEG, -1, -1},
    {"GR_enabled", 0.0, QD_NONNEG, -1, -1},
    {"asD_ds", 0.0, QD_CORR, 0, 3},
    {"asA_ds", 0.0, QD_CORR, 0, 4},
    {"asA_sf", 0.0, QD_CORR, 2, 4},
    {"asD_sf", 0.0, QD_CORR, 2, 3},
    {"ds_sf", 0.0, QD_CORR, 0, 2},
    {"asD_asA", 0.0, QD_CORR, 3, 4},
    {"asD_kf", 0.0, QD_CORR, 1, 3},
    {"asA_kf", 0.0, QD_CORR, 1, 4},
    {"ds_kf", 0.0, QD_CORR, 0, 1},
    {"sf_kf", 0.0, QD_CORR, 1, 2},
    {"esD_ds", 0.0, QD_CORR, 0, 5},
    {"esA_ds", 0.0, QD_CORR, 0, 6},
    {"esA_sf", 0.0, QD_CORR, 2, 6},
    {"esD_sf", 0.0, QD_CORR, 2, 5},
    {"esD_esA", 0.0, QD_CORR, 5, 6},
    {"esD_asD", 0.0, QD_CORR, 3, 5},
    {"esD_asA", 0.0, QD_CORR, 4, 5},
    {"esA_asD", 0.0, QD_CORR, 3, 6},
    {"esA_asA", 0.0, QD_CORR, 4, 6},
    {"esD_kf", 0.0, QD_CORR, 1, 5},
    {"esA_kf", 0.0, QD_CORR, 1, 6},
    {"xpr_ds", 0.0, QD_CORR, 0, 7},
    {"xpr_kf", 0.0, QD_CORR, 1, 7},
    {"xpr_sf", 0.0, QD_CORR, 2, 7},
    {"xpr_asd", 0.0, QD_CORR, 3, 7},
    {"xpr_asa", 0.0, QD_CORR, 4, 7},
    {"xpr_esd", 0.0, QD_CORR, 5, 7},
    {"xpr_esa", 0.0, QD_CORR, 6, 7},
    // New keys for 3GPP TR 38.901 v19.5.0
    {"LSP_fc_min", 0.0, QD_NONNEG, -1, -1},
    {"ES_D_mu_B", 0.0, QD_REAL, -1, -1},
    {"ES_D_mu_C", 0.0, QD_REAL, -1, -1},
    {"ES_D_mu_D", 0.0, QD_REAL, -1, -1},
    {"EoD_offset_e", 0.0, QD_REAL, -1, -1},
    {"EoD_offset_e_gamma", 0.0, QD_REAL, -1, -1},
    {"EoD_offset_a", 0.0, QD_REAL, -1, -1},
    {"EoD_offset_a_gamma", 0.0, QD_REAL, -1, -1},
    {"EoD_offset_b", 0.0, QD_NONNEG, -1, -1},
    {"EoD_offset_c", 0.0, QD_REAL, -1, -1},
    {"EoD_offset_c_gamma", 0.0, QD_REAL, -1, -1},
    {"EoD_offset_c_hUT", 0.0, QD_REAL, -1, -1},
    {"EoD_offset_hcl", 0.0, QD_NONNEG, -1, -1},
    {"PerClusterDS_fc_min", 0.0, QD_NONNEG, -1, -1},
    {"PerClusterES_D_scale", 0.0, QD_NONNEG, -1, -1},
    {"absTOA_Lmax", 0.0, QD_NONNEG, -1, -1},
    {"NumClusters_min", 0.0, QD_INT_NONNEG, -1, -1},
    {"NF_k1", 0.0, QD_INT_NONNEG, -1, -1},
    {"NF_alpha", 0.0, QD_NONNEG, -1, -1},
    {"NF_beta", 0.0, QD_NONNEG, -1, -1},
    {"SNS_mu", 0.0, QD_UNIT, -1, -1},
    {"SNS_sigma", 0.0, QD_NONNEG, -1, -1},
    {"SNS_A", 0.0, QD_NONNEG, -1, -1},
    {"SNS_B", 0.0, QD_NONNEG, -1, -1},
    {"SNS_R", 0.0, QD_NONNEG, -1, -1},
    {"SNS_delta_var", 0.0, QD_NONNEG, -1, -1},
    {"PolVar_sigma", 0.0, QD_NONNEG, -1, -1},
};

static const arma::uword qd_scenpar_n_key = sizeof(qd_scenpar_keys) / sizeof(qd_scenpar_keys[0]);

// Index of a key in the key table, qd_scenpar_n_key if the name is not known
static arma::uword qd_scenpar_find(const std::string &name)
{
    for (arma::uword i = 0; i < qd_scenpar_n_key; ++i)
        if (name == qd_scenpar_keys[i].name)
            return i;
    return qd_scenpar_n_key;
}

// Value check of one key; returns the rule as text if the value violates it, nullptr if it passes
static const char *qd_scenpar_check_value(qd_scenpar_class cls, double v)
{
    switch (cls)
    {
    case QD_ANY:
    case QD_REAL:
        return std::isnan(v) ? "must not be NaN" : nullptr;
    case QD_NONNEG:
        return (v >= 0.0 && std::isfinite(v)) ? nullptr : "must be finite and >= 0";
    case QD_POS:
        return (v > 0.0 && std::isfinite(v)) ? nullptr : "must be finite and > 0";
    case QD_INT_POS:
        return (v > 0.0 && std::isfinite(v) && std::floor(v) == v) ? nullptr : "must be an integer > 0";
    case QD_INT_NONNEG:
        return (v >= 0.0 && std::isfinite(v) && std::floor(v) == v) ? nullptr : "must be an integer >= 0";
    case QD_CORR:
        return (v >= -1.0 && v <= 1.0) ? nullptr : "must be in the range from -1 to 1";
    case QD_UNIT:
        return (v >= 0.0 && v <= 1.0) ? nullptr : "must be in the range from 0 to 1";
    }
    return nullptr;
}

// Value checks of a complete parameter set, 'val' has qd_scenpar_n_key elements in table order
// - 'fn' and 'line' (1-based line per key, 0 = not in the file) are used for the messages of the file reader
// - Throws std::invalid_argument at the first violation
static void qd_scenpar_check(const double *val, const std::string &subpath_method,
                             const std::string &fn = "", const arma::uword *line = nullptr, arma::uword subpath_line = 0)
{
    auto where = [&fn](arma::uword ln) -> std::string
    {
        if (fn.empty())
            return "";
        return ln == 0 ? fn + ": " : fn + ", line " + std::to_string(ln) + ": ";
    };

    for (arma::uword i = 0; i < qd_scenpar_n_key; ++i)
    {
        const char *rule = qd_scenpar_check_value(qd_scenpar_keys[i].cls, val[i]);
        if (rule != nullptr)
            throw std::invalid_argument(where(line == nullptr ? 0 : line[i]) + "Value of '" +
                                        qd_scenpar_keys[i].name + "' " + rule + ".");
    }

    if (subpath_method != "legacy" && subpath_method != "Laplacian" && subpath_method != "mmMAGIC")
        throw std::invalid_argument(where(subpath_line) +
                                    "Value of 'SubpathMethod' must be 'legacy', 'Laplacian' or 'mmMAGIC'.");

    // Checks that involve two keys
    const arma::uword i_ncl = qd_scenpar_find("NumClusters"), i_min = qd_scenpar_find("NumClusters_min");
    if (val[i_min] > val[i_ncl])
        throw std::invalid_argument(where(line == nullptr ? 0 : line[i_min]) +
                                    "Value of 'NumClusters_min' must not exceed 'NumClusters'.");

    const arma::uword i_alpha = qd_scenpar_find("NF_alpha"), i_beta = qd_scenpar_find("NF_beta");
    if ((val[i_alpha] > 0.0) != (val[i_beta] > 0.0))
        throw std::invalid_argument(where(line == nullptr ? 0 : line[val[i_alpha] > 0.0 ? i_alpha : i_beta]) +
                                    "Values of 'NF_alpha' and 'NF_beta' must be both 0 or both > 0.");
}

// LSP cross-correlation matrix from a complete parameter set, 'X' has 64 elements (8 x 8, symmetric)
// - Returns true if the matrix is positive definite: all 8 pivots of the Cholesky factorization are > 0
static bool qd_scenpar_xcorr(const double *val, double *X)
{
    for (int i = 0; i < 64; ++i)
        X[i] = (i % 9 == 0) ? 1.0 : 0.0;

    for (arma::uword i = 0; i < qd_scenpar_n_key; ++i)
        if (qd_scenpar_keys[i].row >= 0)
        {
            const int r = qd_scenpar_keys[i].row, c = qd_scenpar_keys[i].col;
            X[c * 8 + r] = val[i];
            X[r * 8 + c] = val[i];
        }

    // Cholesky-Banachiewicz, lower triangle; a NaN pivot fails the test
    double L[64];
    for (int j = 0; j < 8; ++j)
    {
        double d = X[j * 8 + j];
        for (int k = 0; k < j; ++k)
            d -= L[k * 8 + j] * L[k * 8 + j];
        if (!(d > 0.0))
            return false;
        d = std::sqrt(d);
        L[j * 8 + j] = d;
        for (int i = j + 1; i < 8; ++i)
        {
            double s = X[j * 8 + i];
            for (int k = 0; k < j; ++k)
                s -= L[k * 8 + i] * L[k * 8 + j];
            L[j * 8 + i] = s / d;
        }
    }
    return true;
}

// Write the optional outputs that both validators share
static void qd_scenpar_write(const double *X, arma::mat *lsp_xcorr, std::vector<std::string> *key_order)
{
    if (lsp_xcorr != nullptr)
    {
        if (lsp_xcorr->n_rows != 8 || lsp_xcorr->n_cols != 8)
            lsp_xcorr->set_size(8, 8);
        std::memcpy(lsp_xcorr->memptr(), X, 64 * sizeof(double));
    }
    if (key_order != nullptr)
    {
        key_order->resize(qd_scenpar_n_key);
        for (arma::uword i = 0; i < qd_scenpar_n_key; ++i)
            (*key_order)[i] = qd_scenpar_keys[i].name;
    }
}

static inline bool qd_conf_is_space(char c) { return c == ' ' || c == '\t'; }

static inline bool qd_conf_is_name_char(char c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

// True if 's' is one token of the characters A-Z, a-z, 0-9 and '_' (letters only if 'letters_only')
static bool qd_conf_is_name(const std::string &s, bool letters_only = false)
{
    if (s.empty())
        return false;
    for (char c : s)
        if (!qd_conf_is_name_char(c) || (letters_only && ((c >= '0' && c <= '9') || c == '_')))
            return false;
    return true;
}

static std::string qd_conf_trim(const std::string &s, size_t b, size_t e)
{
    while (b < e && qd_conf_is_space(s[b]))
        ++b;
    while (e > b && qd_conf_is_space(s[e - 1]))
        --e;
    return s.substr(b, e - b);
}

// Split a file into lines at '\n' and remove one trailing '\r' per line
static std::vector<std::string> qd_conf_read_lines(const std::string &fn)
{
    std::error_code ec;
    std::ifstream file(fn, std::ios::in | std::ios::binary);
    if (!file.is_open() || std::filesystem::is_directory(fn, ec))
        throw std::runtime_error("Cannot open file '" + fn + "'.");

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        lines.push_back(line);
    }
    return lines;
}

// Split one line into key and value
// - Returns 0 for a blank or comment line, 1 for an assignment, 2 if there is no '=', 3 if the text before the
//   '=' is not exactly one name
// - The comment (from the first '%') is removed first; 'value' is the trimmed text after the first '='
static int qd_conf_split_line(const std::string &line, std::string &key, std::string &value)
{
    size_t end = line.find('%');
    if (end == std::string::npos)
        end = line.size();

    const std::string text = qd_conf_trim(line, 0, end);
    if (text.empty())
        return 0;

    const size_t eq = text.find('=');
    if (eq == std::string::npos)
        return 2;

    key = qd_conf_trim(text, 0, eq);
    value = qd_conf_trim(text, eq + 1, text.size());
    return qd_conf_is_name(key) ? 1 : 3;
}

// Split a value at white space
static std::vector<std::string> qd_conf_tokens(const std::string &value)
{
    std::vector<std::string> tok;
    size_t i = 0;
    while (i < value.size())
    {
        while (i < value.size() && qd_conf_is_space(value[i]))
            ++i;
        size_t j = i;
        while (j < value.size() && !qd_conf_is_space(value[j]))
            ++j;
        if (j > i)
            tok.push_back(value.substr(i, j - i));
        i = j;
    }
    return tok;
}

// Parse one number token: [+-] digits [. digits] [e|E [+-] digits], with at least one digit before the
// exponent, or [+-] Inf in any case. Returns false for anything else (NaN, complex, hex, trailing text).
static bool qd_conf_parse_number(const std::string &s, double &v)
{
    const size_t n = s.size();
    size_t i = 0;
    bool negative = false;
    if (i < n && (s[i] == '+' || s[i] == '-'))
        negative = s[i++] == '-';

    if (n - i == 3 && (s[i] == 'i' || s[i] == 'I') && (s[i + 1] == 'n' || s[i + 1] == 'N') &&
        (s[i + 2] == 'f' || s[i + 2] == 'F'))
    {
        v = negative ? -QD_INF : QD_INF;
        return true;
    }

    size_t n_digit = 0;
    while (i < n && s[i] >= '0' && s[i] <= '9')
        ++i, ++n_digit;
    if (i < n && s[i] == '.')
    {
        ++i;
        while (i < n && s[i] >= '0' && s[i] <= '9')
            ++i, ++n_digit;
    }
    if (n_digit == 0)
        return false;

    if (i < n && (s[i] == 'e' || s[i] == 'E'))
    {
        ++i;
        if (i < n && (s[i] == '+' || s[i] == '-'))
            ++i;
        size_t n_exp = 0;
        while (i < n && s[i] >= '0' && s[i] <= '9')
            ++i, ++n_exp;
        if (n_exp == 0)
            return false;
    }
    if (i != n)
        return false;

    // The text is a plain decimal number; convert it independent of the global locale
    std::istringstream is(s);
    is.imbue(std::locale::classic());
    is >> v;
    return !is.fail();
}

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# conf_file_read
Read and validate a QuaDRiGa scenario configuration file

- A conf file is a text file with one `Key = value` assignment per line; `%` starts a comment; blank lines
  are allowed; line ends can be LF or CRLF; space and tab are white space
- Keys are case-sensitive names of the characters `A-Z`, `a-z`, `0-9` and `_`; there are four kinds:
  - Scenario parameters: `n_key = 165` numeric keys with one number each, returned in `scenpar`. Keys that
    are not in the file have their default value. The names and their order are returned in `key_order`.
  - `SubpathMethod`: one of the words `legacy`, `Laplacian`, `mmMAGIC`; default `legacy`
  - Path-loss parameters: every key that starts with `PL_`. `PL_model` has one name as value. All others
    have one or more numbers separated by white space and are returned in `plpar` without the prefix.
  - `ShortName`: an alias for the scenario, one name; can occur several times
- Any other key is ignored and reported in `ignored_keys`; its value is not examined
- A key that is stated more than once has its last value
- Numbers are decimal with an optional sign, an optional decimal point and an optional exponent (`e` or
  `E`), for example `-6.28`, `.5`, `6.2e-07`; `Inf`, `+Inf` and `-Inf` are accepted in any case
- Elevation angles are used throughout (+90 deg is up); no key refers to a zenith angle
- The 165 keys are 138 numeric keys of QuaDRiGa and 27 keys for 3GPP TR 38.901 v19.5.0. With their
  default of 0 the new keys have no effect.
- The QuaDRiGa key `GR_epsilon` (fixed relative permittivity of the ground) is not supported and is
  reported in `ignored_keys` like any other unknown key; the permittivity is complex and is set in the simulation
- The 28 correlation keys (`asD_ds` to `xpr_esa`) define the symmetric matrix `lsp_xcorr` with unit diagonal
- Throws `std::runtime_error` if the file cannot be opened
- Throws `std::invalid_argument` with file name and line number for a line that is not blank, not a comment
  and not a well-formed assignment: no `=`, more or less than one name before the `=`, a missing value, a
  word where a number is required or the reverse, a second value or other text after the value, `NaN`,
  complex numbers, a `PL_` key without a name
- With `validate = true`, a value that fails the checks of [[scenpar_validate]] is an error as well
- A correlation matrix that is not positive definite is not an error; it is reported by the return value
- The file is parsed and checked completely before any output is written; a failed call leaves all
  outputs untouched

## Declaration:
```
bool quadriga_lib::conf_file_read(
    const std::string &fn,
    std::unordered_map<std::string, double> &scenpar,
    std::string *subpath_method = nullptr,
    std::string *pl_model = nullptr,
    std::unordered_map<std::string, std::vector<double>> *plpar = nullptr,
    arma::mat *lsp_xcorr = nullptr,
    std::vector<std::string> *key_order = nullptr,
    std::vector<std::string> *pl_order = nullptr,
    std::vector<std::string> *short_names = nullptr,
    std::vector<std::string> *ignored_keys = nullptr,
    bool validate = true);
```

## Inputs:
- **`fn`** — Path to the conf file
- **`validate`** *(optional)* — `true`: apply the value checks of [[scenpar_validate]]; `false`: only the line
  grammar is checked

## Outputs:
- **`scenpar`** — Numeric scenario parameters; previous content is discarded; `n_key = 165` entries
- **`subpath_method`** *(optional)* — Value of `SubpathMethod`; `legacy` if not in the file
- **`pl_model`** *(optional)* — Value of `PL_model`; empty if not in the file
- **`plpar`** *(optional)* — Path-loss parameters; keys without the prefix `PL_`; `PL_model` is not included;
  empty if the file has no such key; `n_pl` entries with one or more values each
- **`lsp_xcorr`** *(optional)* — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `[8, 8]`
- **`key_order`** *(optional)* — Canonical order of the keys in `scenpar`: the QuaDRiGa order without
  `SubpathMethod` (its position is between elements 105 and 106, 0-based), then the 27 new keys; `[n_key]`
- **`pl_order`** *(optional)* — Keys of `plpar` in the order of their first occurrence in the file; `[n_pl]`
- **`short_names`** *(optional)* — Values of all `ShortName` lines in file order
- **`ignored_keys`** *(optional)* — Unknown keys, each once, in the order of their first occurrence

## Returns:
- `true` if `lsp_xcorr` is positive definite, i.e. correlated LSPs can be generated from it

## See also:
- [[scenpar_validate]] (value checks, defaults and the list of validation rules)
- [[conf_file_list]] (find the conf files in a set of directories)
MD!*/

bool quadriga_lib::conf_file_read(const std::string &fn,
                                  std::unordered_map<std::string, double> &scenpar,
                                  std::string *subpath_method,
                                  std::string *pl_model,
                                  std::unordered_map<std::string, std::vector<double>> *plpar,
                                  arma::mat *lsp_xcorr,
                                  std::vector<std::string> *key_order,
                                  std::vector<std::string> *pl_order,
                                  std::vector<std::string> *short_names,
                                  std::vector<std::string> *ignored_keys,
                                  bool validate)
{
    const std::vector<std::string> lines = qd_conf_read_lines(fn);

    // Parse into local variables
    std::vector<double> val(qd_scenpar_n_key);
    std::vector<arma::uword> val_line(qd_scenpar_n_key, 0);
    for (arma::uword i = 0; i < qd_scenpar_n_key; ++i)
        val[i] = qd_scenpar_keys[i].def;

    std::string l_subpath_method = "legacy", l_pl_model;
    arma::uword subpath_line = 0;
    std::unordered_map<std::string, std::vector<double>> l_plpar;
    std::vector<std::string> l_pl_order, l_short_names, l_ignored_keys;
    std::unordered_set<std::string> ignored_set;

    std::string key, value;
    for (arma::uword i_line = 0; i_line < lines.size(); ++i_line)
    {
        const std::string where = fn + ", line " + std::to_string(i_line + 1) + ": ";

        const int kind = qd_conf_split_line(lines[i_line], key, value);
        if (kind == 0)
            continue;
        if (kind == 2)
            throw std::invalid_argument(where + "Line is not a comment and has no '='.");
        if (kind == 3)
            throw std::invalid_argument(where + "Text before the '=' must be one key name.");

        if (key == "ShortName")
        {
            if (!qd_conf_is_name(value))
                throw std::invalid_argument(where + "Value of 'ShortName' must be one name.");
            l_short_names.push_back(value);
        }
        else if (key.compare(0, 3, "PL_") == 0)
        {
            const std::string name = key.substr(3);
            if (name.empty())
                throw std::invalid_argument(where + "Key 'PL_' has no parameter name.");

            if (name == "model")
            {
                if (!qd_conf_is_name(value))
                    throw std::invalid_argument(where + "Value of 'PL_model' must be one name.");
                l_pl_model = value;
                continue;
            }

            const std::vector<std::string> tok = qd_conf_tokens(value);
            if (tok.empty())
                throw std::invalid_argument(where + "Value of '" + key + "' is missing.");

            std::vector<double> v(tok.size());
            for (size_t i = 0; i < tok.size(); ++i)
                if (!qd_conf_parse_number(tok[i], v[i]))
                    throw std::invalid_argument(where + "Value of '" + key + "' is not a number.");

            if (l_plpar.count(name) == 0)
                l_pl_order.push_back(name);
            l_plpar[name] = v;
        }
        else if (key == "SubpathMethod")
        {
            if (!qd_conf_is_name(value, true))
                throw std::invalid_argument(where + "Value of 'SubpathMethod' must be one word.");
            l_subpath_method = value;
            subpath_line = i_line + 1;
        }
        else
        {
            const arma::uword ind = qd_scenpar_find(key);
            if (ind == qd_scenpar_n_key) // Unknown key, value is not examined
            {
                if (ignored_set.insert(key).second)
                    l_ignored_keys.push_back(key);
                continue;
            }
            if (value.empty())
                throw std::invalid_argument(where + "Value of '" + key + "' is missing.");
            if (!qd_conf_parse_number(value, val[ind]))
                throw std::invalid_argument(where + "Value of '" + key + "' is not a number.");
            val_line[ind] = i_line + 1;
        }
    }

    if (validate)
        qd_scenpar_check(val.data(), l_subpath_method, fn, val_line.data(), subpath_line);

    double X[64];
    const bool positive_definite = qd_scenpar_xcorr(val.data(), X);

    // All checks passed, write the outputs
    scenpar.clear();
    for (arma::uword i = 0; i < qd_scenpar_n_key; ++i)
        scenpar[qd_scenpar_keys[i].name] = val[i];

    if (subpath_method != nullptr)
        *subpath_method = l_subpath_method;
    if (pl_model != nullptr)
        *pl_model = l_pl_model;
    if (plpar != nullptr)
        *plpar = l_plpar;
    if (pl_order != nullptr)
        *pl_order = l_pl_order;
    if (short_names != nullptr)
        *short_names = l_short_names;
    if (ignored_keys != nullptr)
        *ignored_keys = l_ignored_keys;

    qd_scenpar_write(X, lsp_xcorr, key_order);

    return positive_definite;
}

/*!MD
# scenpar_validate
Validate scenario parameters and complete them with default values

- Checks a set of QuaDRiGa scenario parameters that did not come from a conf file and adds every missing key
  with its default value, so that `scenpar` has all `n_key = 165` entries on return
- An empty `scenpar` returns the complete default parameter set
- A set with the 138 numeric keys of QuaDRiGa is completed with the 27 keys for 3GPP TR 38.901 v19.5.0, which
  have no effect at their default of 0
- A name in `scenpar` that is not one of the 165 keys is an error, also with `validate = false`; this includes
  the QuaDRiGa key `GR_epsilon`, which is not supported
- Value checks with `validate = true`; `NaN` is an error for every key:
  - Integer and > 0: `NumClusters`, `NumSubPaths`
  - Finite and > 0: `r_DS`
  - Finite and >= 0: the 8 decorrelation distances (`DS_lambda` ... `XPR_lambda`), `SC_lambda`,
    `absTOA_lambda`, `PerClusterDS`, `PerClusterDS_min`, `PerClusterAS_D`, `PerClusterAS_A`, `PerClusterES_D`,
    `PerClusterES_A`, `LNS_ksi`, `LOS_scatter_radius`, `GR_enabled`, `LSP_fc_min`, `EoD_offset_b`,
    `EoD_offset_hcl`, `PerClusterDS_fc_min`, `PerClusterES_D_scale`, `absTOA_Lmax`, `NF_alpha`, `NF_beta`,
    `SNS_sigma`, `SNS_A`, `SNS_B`, `SNS_R`, `SNS_delta_var`, `PolVar_sigma`
  - Integer and >= 0: `NumClusters_min`, `NF_k1`
  - In the range from 0 to 1: `SNS_mu`
  - In the range from -1 to 1: the 28 correlations (`asD_ds` to `xpr_esa`)
  - All other keys accept any value including `Inf` and `-Inf`
  - `NumClusters_min` must not exceed `NumClusters`
  - `NF_alpha` and `NF_beta` must be both 0 or both > 0
  - `subpath_method` must be `legacy`, `Laplacian` or `mmMAGIC`
- With `validate = false` no value is checked: `NaN` and any `subpath_method` are accepted
- The symmetric matrix `lsp_xcorr` is built from the 28 correlation keys and has a unit diagonal. It is
  tested by a Cholesky factorization. A matrix that is not positive definite is not an error.
- Throws `std::invalid_argument`; a failed call leaves `scenpar` and the outputs untouched

## Declaration:
```
bool quadriga_lib::scenpar_validate(
    std::unordered_map<std::string, double> &scenpar,
    const std::string &subpath_method = "legacy",
    arma::mat *lsp_xcorr = nullptr,
    std::vector<std::string> *key_order = nullptr,
    bool validate = true);
```

## Inputs:
- **`scenpar`** — Numeric scenario parameters; updated in-place: given values are kept, missing keys are
  added with their default; `n_key = 165` entries on return
- **`subpath_method`** *(optional)* — Value of `SubpathMethod`, the only parameter that is not numeric
- **`validate`** *(optional)* — `false`: only complete with defaults, no value checks

## Outputs:
- **`lsp_xcorr`** *(optional)* — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `[8, 8]`
- **`key_order`** *(optional)* — Canonical order of the keys in `scenpar`: the QuaDRiGa order without
  `SubpathMethod` (its position is between elements 105 and 106, 0-based), then the 27 new keys; `[n_key]`

## Returns:
- `true` if `lsp_xcorr` is positive definite (all 8 pivots of the factorization are > 0); `false` also if it
  contains `NaN`

## See also:
- [[conf_file_read]] (read the parameters from a conf file)
MD!*/

bool quadriga_lib::scenpar_validate(std::unordered_map<std::string, double> &scenpar,
                                    const std::string &subpath_method,
                                    arma::mat *lsp_xcorr,
                                    std::vector<std::string> *key_order,
                                    bool validate)
{
    double val[sizeof(qd_scenpar_keys) / sizeof(qd_scenpar_keys[0])];
    bool given[sizeof(qd_scenpar_keys) / sizeof(qd_scenpar_keys[0])];
    for (arma::uword i = 0; i < qd_scenpar_n_key; ++i)
        val[i] = qd_scenpar_keys[i].def, given[i] = false;

    for (const auto &entry : scenpar)
    {
        const arma::uword ind = qd_scenpar_find(entry.first);
        if (ind == qd_scenpar_n_key)
            throw std::invalid_argument("Unknown key '" + entry.first + "' in 'scenpar'.");
        val[ind] = entry.second;
        given[ind] = true;
    }

    if (validate)
        qd_scenpar_check(val, subpath_method);

    double X[64];
    const bool positive_definite = qd_scenpar_xcorr(val, X);

    // All checks passed, write the outputs
    for (arma::uword i = 0; i < qd_scenpar_n_key; ++i)
        if (!given[i])
            scenpar[qd_scenpar_keys[i].name] = val[i];

    qd_scenpar_write(X, lsp_xcorr, key_order);

    return positive_definite;
}

/*!MD
# conf_file_list
List the scenarios defined by the conf files in a set of directories

- Lists the files whose name ends in `.conf` (case-sensitive) in each directory, not recursive. The
  scenario name is the file name without this extension.
- Directories are processed in the given order, the files of one directory in byte-wise sorted order of their
  names (digits before upper case before lower case)
- A file whose name was already found in an earlier directory is skipped completely, so the first directory
  wins. This allows a local folder to override the files of a shared folder.
- With `parse_shortnames = true` each file is read and one entry is added right after the file for every
  line `ShortName = name` that is not a comment. Other lines are not examined.
- Throws `std::invalid_argument` if a short name equals an earlier entry, if a file name equals an earlier
  short name, or if the value of a `ShortName` line is not one name of the characters `A-Z`, `a-z`, `0-9`, `_`
- Throws `std::runtime_error` if a listed file cannot be opened for reading the short names
- A directory that does not exist or is not a directory is skipped; an empty string is the current directory
- A failed call leaves the outputs untouched

## Declaration:
```
arma::uword quadriga_lib::conf_file_list(
    const std::vector<std::string> &dirs,
    std::vector<std::string> &scenarios,
    std::vector<std::string> *file_names = nullptr,
    std::vector<std::string> *file_dirs = nullptr,
    bool parse_shortnames = true);
```

## Inputs:
- **`dirs`** — Directories, searched in this order; can be empty
- **`parse_shortnames`** *(optional)* — `true`: read the files and add their short names; `false`: list the
  files only, which is faster

## Outputs:
- **`scenarios`** — Scenario names and short names; previous content is discarded; `[n_scen]`
- **`file_names`** *(optional)* — File name including `.conf` for each entry; `[n_scen]`
- **`file_dirs`** *(optional)* — Directory of the file for each entry, exactly as given in `dirs`; `[n_scen]`

## Returns:
- Number of entries, `n_scen`

## See also:
- [[conf_file_read]] (read one of the files)
MD!*/

arma::uword quadriga_lib::conf_file_list(const std::vector<std::string> &dirs,
                                         std::vector<std::string> &scenarios,
                                         std::vector<std::string> *file_names,
                                         std::vector<std::string> *file_dirs,
                                         bool parse_shortnames)
{
    std::vector<std::string> l_scenarios, l_file_names, l_file_dirs;
    std::unordered_set<std::string> file_set, short_set;

    for (const std::string &dir : dirs)
    {
        const std::filesystem::path path = dir.empty() ? std::filesystem::path(".") : std::filesystem::path(dir);

        std::error_code ec;
        if (!std::filesystem::is_directory(path, ec))
            continue;

        // Names of the regular files that end in ".conf", byte-wise sorted
        std::vector<std::string> names;
        for (std::filesystem::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec))
        {
            std::error_code ec_file;
            if (!it->is_regular_file(ec_file))
                continue;
            const std::string name = it->path().filename().string();
            if (name.size() > 5 && name.compare(name.size() - 5, 5, ".conf") == 0)
                names.push_back(name);
        }
        std::sort(names.begin(), names.end());

        for (const std::string &name : names)
        {
            const std::string scen = name.substr(0, name.size() - 5);
            if (file_set.count(scen) != 0) // Defined by an earlier directory
                continue;
            if (short_set.count(scen) != 0)
                throw std::invalid_argument("File name '" + name + "' is ambiguous, an earlier file defines '" +
                                            scen + "' as its short name.");

            file_set.insert(scen);
            l_scenarios.push_back(scen);
            l_file_names.push_back(name);
            l_file_dirs.push_back(dir);

            if (!parse_shortnames)
                continue;

            const std::string fn = (path / name).string();
            const std::vector<std::string> lines = qd_conf_read_lines(fn);
            std::string key, value;
            for (arma::uword i_line = 0; i_line < lines.size(); ++i_line)
            {
                if (qd_conf_split_line(lines[i_line], key, value) != 1 || key != "ShortName")
                    continue;

                const std::string where = fn + ", line " + std::to_string(i_line + 1) + ": ";
                if (!qd_conf_is_name(value))
                    throw std::invalid_argument(where + "Value of 'ShortName' must be one name.");
                if (file_set.count(value) != 0 || short_set.count(value) != 0)
                    throw std::invalid_argument(where + "ShortName '" + value + "' is ambiguous.");

                short_set.insert(value);
                l_scenarios.push_back(value);
                l_file_names.push_back(name);
                l_file_dirs.push_back(dir);
            }
        }
    }

    scenarios = l_scenarios;
    if (file_names != nullptr)
        *file_names = l_file_names;
    if (file_dirs != nullptr)
        *file_dirs = l_file_dirs;

    return (arma::uword)scenarios.size();
}
