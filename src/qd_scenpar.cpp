// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
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
    QD_INT_POS,    // Integer from 1 to 1000
    QD_INT_NONNEG, // Integer from 0 to 1000
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
    unsigned char sec;    // Section in a written conf file, see 'qd_conf_sections'
    unsigned char par;    // Paragraph within the section; written in table order, separated by a blank line
    const char *comment;  // Comment after the value in a written conf file, with the unit in brackets
};

static const double QD_INF = std::numeric_limits<double>::infinity();

// Key table: the single source of truth for names, canonical order, defaults and validation classes
// - 138 numeric legacy keys in the order of QuaDRiGa's 'set_scenario_table.m', then the 33 keys for TR 38.901 v19.5.0
// - The legacy key 'GR_epsilon' is not supported: a real value has no use, the relative permittivity of the ground is complex
// - 'SubpathMethod' is the only string key and is not part of the table; its legacy position is field 107
// - LSP order of the cross-correlation matrix: DS, KF, SF, ASD, ASA, ESD, ESA, XPR
static const qd_scenpar_key qd_scenpar_keys[] = {
    {"NumClusters", 6.0, QD_INT_POS, -1, -1, 3, 1, "number of clusters"},
    {"NumSubPaths", 20.0, QD_INT_POS, -1, -1, 3, 1, "number of paths per (NLOS) cluster"},
    {"DS_mu", -6.0, QD_REAL, -1, -1, 0, 1, "delay spread [log10(s)]"},
    {"DS_sigma", 0.5, QD_REAL, -1, -1, 0, 1, "delay spread STD [log10(s)]"},
    {"DS_lambda", 20.0, QD_NONNEG, -1, -1, 1, 1, "DS decorrelation distance [m]"},
    {"DS_omega", 0.0, QD_ANY, -1, -1, 0, 1, "reference frequency offset for the DS [GHz]"},
    {"DS_gamma", 0.0, QD_REAL, -1, -1, 0, 1, "freq.-dep. of DS [log10(s)/log10(GHz)]"},
    {"DS_epsilon", 0.0, QD_ANY, -1, -1, 0, 1, "TX-RX 2D dist.-dep. of DS [log10(s)/log10(m)]"},
    {"DS_zeta", 0.0, QD_ANY, -1, -1, 0, 1, "TX height-dep. of DS [log10(s)/log10(m)]"},
    {"DS_alpha", 0.0, QD_ANY, -1, -1, 0, 1, "elevation-dep. of DS [log10(s)/log10(rad)]"},
    {"DS_delta", 0.0, QD_REAL, -1, -1, 0, 1, "freq.-dep. of DS STD [log10(s)/log10(GHz)]"},
    {"DS_kappa", 0.0, QD_ANY, -1, -1, 0, 1, "TX-RX 2D dist.-dep. of DS STD [log10(s)/log10(m)]"},
    {"DS_tau", 0.0, QD_ANY, -1, -1, 0, 1, "TX height-dep. of DS STD [log10(s)/log10(m)]"},
    {"DS_beta", 0.0, QD_ANY, -1, -1, 0, 1, "elevation-dep. of DS STD [log10(s)/log10(rad)]"},
    {"AS_D_mu", 1.0, QD_REAL, -1, -1, 0, 4, "azimuth of departure angle spread [log10(deg)]"},
    {"AS_D_sigma", 0.2, QD_REAL, -1, -1, 0, 4, "azimuth of departure angle spread STD [log10(deg)]"},
    {"AS_D_lambda", 20.0, QD_NONNEG, -1, -1, 1, 1, "ASD decorrelation distance [m]"},
    {"AS_D_omega", 0.0, QD_ANY, -1, -1, 0, 4, "reference frequency offset for the ASD [GHz]"},
    {"AS_D_gamma", 0.0, QD_REAL, -1, -1, 0, 4, "freq.-dep. of ASD [log10(deg)/log10(GHz)]"},
    {"AS_D_epsilon", 0.0, QD_ANY, -1, -1, 0, 4, "TX-RX 2D dist.-dep. of ASD [log10(deg)/log10(m)]"},
    {"AS_D_zeta", 0.0, QD_ANY, -1, -1, 0, 4, "TX height-dep. of ASD [log10(deg)/log10(m)]"},
    {"AS_D_alpha", 0.0, QD_ANY, -1, -1, 0, 4, "elevation-dep. of ASD [log10(deg)/log10(rad)]"},
    {"AS_D_delta", 0.0, QD_REAL, -1, -1, 0, 4, "freq.-dep. of ASD STD [log10(deg)/log10(GHz)]"},
    {"AS_D_kappa", 0.0, QD_ANY, -1, -1, 0, 4, "TX-RX 2D dist.-dep. of ASD STD [log10(deg)/log10(m)]"},
    {"AS_D_tau", 0.0, QD_ANY, -1, -1, 0, 4, "TX height-dep. of ASD STD [log10(deg)/log10(m)]"},
    {"AS_D_beta", 0.0, QD_ANY, -1, -1, 0, 4, "elevation-dep. of ASD STD [log10(deg)/log10(rad)]"},
    {"AS_A_mu", 1.0, QD_REAL, -1, -1, 0, 5, "azimuth of arrival angle spread [log10(deg)]"},
    {"AS_A_sigma", 0.2, QD_REAL, -1, -1, 0, 5, "azimuth of arrival angle spread STD [log10(deg)]"},
    {"AS_A_lambda", 20.0, QD_NONNEG, -1, -1, 1, 1, "ASA decorrelation distance [m]"},
    {"AS_A_omega", 0.0, QD_ANY, -1, -1, 0, 5, "reference frequency offset for the ASA [GHz]"},
    {"AS_A_gamma", 0.0, QD_REAL, -1, -1, 0, 5, "freq.-dep. of ASA [log10(deg)/log10(GHz)]"},
    {"AS_A_epsilon", 0.0, QD_ANY, -1, -1, 0, 5, "TX-RX 2D dist.-dep. of ASA [log10(deg)/log10(m)]"},
    {"AS_A_zeta", 0.0, QD_ANY, -1, -1, 0, 5, "TX height-dep. of ASA [log10(deg)/log10(m)]"},
    {"AS_A_alpha", 0.0, QD_ANY, -1, -1, 0, 5, "elevation-dep. of ASA [log10(deg)/log10(rad)]"},
    {"AS_A_delta", 0.0, QD_REAL, -1, -1, 0, 5, "freq.-dep. of ASA STD [log10(deg)/log10(GHz)]"},
    {"AS_A_kappa", 0.0, QD_ANY, -1, -1, 0, 5, "TX-RX 2D dist.-dep. of ASA STD [log10(deg)/log10(m)]"},
    {"AS_A_tau", 0.0, QD_ANY, -1, -1, 0, 5, "TX height-dep. of ASA STD [log10(deg)/log10(m)]"},
    {"AS_A_beta", 0.0, QD_ANY, -1, -1, 0, 5, "elevation-dep. of ASA STD [log10(deg)/log10(rad)]"},
    {"ES_D_mu", 0.3, QD_REAL, -1, -1, 0, 6, "elevation of departure angle spread [log10(deg)]"},
    {"ES_D_sigma", 0.2, QD_REAL, -1, -1, 0, 6, "elevation of departure angle spread STD [log10(deg)]"},
    {"ES_D_lambda", 20.0, QD_NONNEG, -1, -1, 1, 1, "ESD decorrelation distance [m]"},
    {"ES_D_omega", 0.0, QD_ANY, -1, -1, 0, 6, "reference frequency offset for the ESD [GHz]"},
    {"ES_D_gamma", 0.0, QD_REAL, -1, -1, 0, 6, "freq.-dep. of ESD [log10(deg)/log10(GHz)]"},
    {"ES_D_epsilon", 0.0, QD_ANY, -1, -1, 0, 6, "TX-RX 2D dist.-dep. of ESD [log10(deg)/log10(m)]"},
    {"ES_D_zeta", 0.0, QD_ANY, -1, -1, 0, 6, "TX height-dep. of ESD [log10(deg)/log10(m)]"},
    {"ES_D_alpha", 0.0, QD_ANY, -1, -1, 0, 6, "elevation-dep. of ESD [log10(deg)/log10(rad)]"},
    {"ES_D_delta", 0.0, QD_REAL, -1, -1, 0, 6, "freq.-dep. of ESD STD [log10(deg)/log10(GHz)]"},
    {"ES_D_kappa", 0.0, QD_ANY, -1, -1, 0, 6, "TX-RX 2D dist.-dep. of ESD STD [log10(deg)/log10(m)]"},
    {"ES_D_tau", 0.0, QD_ANY, -1, -1, 0, 6, "TX height-dep. of ESD STD [log10(deg)/log10(m)]"},
    {"ES_D_beta", 0.0, QD_ANY, -1, -1, 0, 6, "elevation-dep. of ESD STD [log10(deg)/log10(rad)]"},
    {"ES_D_mu_A", 0.0, QD_REAL, -1, -1, 0, 6, "TX-RX 2D dist.-dep. of ESD [log10(deg)/km]"},
    {"ES_D_mu_min", -QD_INF, QD_REAL, -1, -1, 0, 6, "minimum ESD reference value [log10(deg)]"},
    {"ES_A_mu", 0.3, QD_REAL, -1, -1, 0, 7, "elevation of arrival angle spread [log10(deg)]"},
    {"ES_A_sigma", 0.2, QD_REAL, -1, -1, 0, 7, "elevation of arrival angle spread STD [log10(deg)]"},
    {"ES_A_lambda", 20.0, QD_NONNEG, -1, -1, 1, 1, "ESA decorrelation distance [m]"},
    {"ES_A_omega", 0.0, QD_ANY, -1, -1, 0, 7, "reference frequency offset for the ESA [GHz]"},
    {"ES_A_gamma", 0.0, QD_REAL, -1, -1, 0, 7, "freq.-dep. of ESA [log10(deg)/log10(GHz)]"},
    {"ES_A_epsilon", 0.0, QD_ANY, -1, -1, 0, 7, "TX-RX 2D dist.-dep. of ESA [log10(deg)/log10(m)]"},
    {"ES_A_zeta", 0.0, QD_ANY, -1, -1, 0, 7, "TX height-dep. of ESA [log10(deg)/log10(m)]"},
    {"ES_A_alpha", 0.0, QD_ANY, -1, -1, 0, 7, "elevation-dep. of ESA [log10(deg)/log10(rad)]"},
    {"ES_A_delta", 0.0, QD_REAL, -1, -1, 0, 7, "freq.-dep. of ESA STD [log10(deg)/log10(GHz)]"},
    {"ES_A_kappa", 0.0, QD_ANY, -1, -1, 0, 7, "TX-RX 2D dist.-dep. of ESA STD [log10(deg)/log10(m)]"},
    {"ES_A_tau", 0.0, QD_ANY, -1, -1, 0, 7, "TX height-dep. of ESA STD [log10(deg)/log10(m)]"},
    {"ES_A_beta", 0.0, QD_ANY, -1, -1, 0, 7, "elevation-dep. of ESA STD [log10(deg)/log10(rad)]"},
    {"SF_sigma", 8.0, QD_REAL, -1, -1, 0, 3, "Shadow Fading STD [dB]"},
    {"SF_lambda", 20.0, QD_NONNEG, -1, -1, 1, 1, "SF decorrelation distance [m]"},
    {"SF_omega", 0.0, QD_ANY, -1, -1, 0, 3, "reference frequency offset for the SF [GHz]"},
    {"SF_delta", 0.0, QD_REAL, -1, -1, 0, 3, "freq.-dep. of SF STD [dB/log10(GHz)]"},
    {"SF_kappa", 0.0, QD_ANY, -1, -1, 0, 3, "TX-RX 2D dist.-dep. of SF STD [dB/log10(m)]"},
    {"SF_tau", 0.0, QD_ANY, -1, -1, 0, 3, "TX height-dep. of SF STD [dB/log10(m)]"},
    {"SF_beta", 0.0, QD_ANY, -1, -1, 0, 3, "elevation-dep. of SF STD [dB/log10(rad)]"},
    {"KF_mu", 0.0, QD_REAL, -1, -1, 0, 2, "Ricean K-factor [dB]"},
    {"KF_sigma", 0.0, QD_REAL, -1, -1, 0, 2, "Ricean K-factor STD [dB]"},
    {"KF_lambda", 20.0, QD_NONNEG, -1, -1, 1, 1, "KF decorrelation distance [m]"},
    {"KF_omega", 0.0, QD_ANY, -1, -1, 0, 2, "reference frequency offset for the KF [GHz]"},
    {"KF_gamma", 0.0, QD_REAL, -1, -1, 0, 2, "freq.-dep. of KF [dB/log10(GHz)]"},
    {"KF_epsilon", 0.0, QD_ANY, -1, -1, 0, 2, "TX-RX 2D dist.-dep. of KF [dB/log10(m)]"},
    {"KF_zeta", 0.0, QD_ANY, -1, -1, 0, 2, "TX height-dep. of KF [dB/log10(m)]"},
    {"KF_alpha", 0.0, QD_ANY, -1, -1, 0, 2, "elevation-dep. of KF [dB/log10(rad)]"},
    {"KF_delta", 0.0, QD_REAL, -1, -1, 0, 2, "freq.-dep. of KF STD [dB/log10(GHz)]"},
    {"KF_kappa", 0.0, QD_ANY, -1, -1, 0, 2, "TX-RX 2D dist.-dep. of KF STD [dB/log10(m)]"},
    {"KF_tau", 0.0, QD_ANY, -1, -1, 0, 2, "TX height-dep. of KF STD [dB/log10(m)]"},
    {"KF_beta", 0.0, QD_ANY, -1, -1, 0, 2, "elevation-dep. of KF STD [dB/log10(rad)]"},
    {"XPR_mu", 0.0, QD_REAL, -1, -1, 0, 8, "cross-polarization ratio [dB]"},
    {"XPR_sigma", 10.0, QD_REAL, -1, -1, 0, 8, "cross-polarization ratio STD [dB]"},
    {"XPR_lambda", 20.0, QD_NONNEG, -1, -1, 1, 1, "XPR decorrelation distance [m]"},
    {"XPR_omega", 0.0, QD_ANY, -1, -1, 0, 8, "reference frequency offset for the XPR [GHz]"},
    {"XPR_gamma", 0.0, QD_REAL, -1, -1, 0, 8, "freq.-dep. of XPR [dB/log10(GHz)]"},
    {"XPR_epsilon", 0.0, QD_ANY, -1, -1, 0, 8, "TX-RX 2D dist.-dep. of XPR [dB/log10(m)]"},
    {"XPR_zeta", 0.0, QD_ANY, -1, -1, 0, 8, "TX height-dep. of XPR [dB/log10(m)]"},
    {"XPR_alpha", 0.0, QD_ANY, -1, -1, 0, 8, "elevation-dep. of XPR [dB/log10(rad)]"},
    {"XPR_delta", 0.0, QD_REAL, -1, -1, 0, 8, "freq.-dep. of XPR STD [dB/log10(GHz)]"},
    {"XPR_kappa", 0.0, QD_ANY, -1, -1, 0, 8, "TX-RX 2D dist.-dep. of XPR STD [dB/log10(m)]"},
    {"XPR_tau", 0.0, QD_ANY, -1, -1, 0, 8, "TX height-dep. of XPR STD [dB/log10(m)]"},
    {"XPR_beta", 0.0, QD_ANY, -1, -1, 0, 8, "elevation-dep. of XPR STD [dB/log10(rad)]"},
    {"r_DS", 2.0, QD_POS, -1, -1, 3, 2, "delay scaling factor"},
    {"PerClusterDS", 0.0, QD_NONNEG, -1, -1, 3, 3, "cluster delay spread [ns]"},
    {"PerClusterDS_gamma", 0.0, QD_REAL, -1, -1, 3, 3, "freq.-dep. of cluster delay spread [ns/log10(GHz)]"},
    {"PerClusterDS_min", 0.0, QD_NONNEG, -1, -1, 3, 3, "minimum cluster delay spread [ns]"},
    {"PerClusterAS_D", 0.0, QD_NONNEG, -1, -1, 3, 3, "cluster azimuth of departure angle spread [deg]"},
    {"PerClusterAS_A", 0.0, QD_NONNEG, -1, -1, 3, 3, "cluster azimuth of arrival angle spread [deg]"},
    {"PerClusterES_D", 0.0, QD_NONNEG, -1, -1, 3, 3, "cluster elevation of departure angle spread [deg]"},
    {"PerClusterES_A", 0.0, QD_NONNEG, -1, -1, 3, 3, "cluster elevation of arrival angle spread [deg]"},
    {"absTOA_mu", -100.0, QD_REAL, -1, -1, 6, 1, "absolute time of arrival offset reference value [log10(s)]"},
    {"absTOA_sigma", 0.0, QD_REAL, -1, -1, 6, 1, "absolute time of arrival offset reference STD [log10(s)]"},
    {"absTOA_lambda", 0.0, QD_NONNEG, -1, -1, 6, 1, "absolute time of arrival offset decorrelation distance [m]"},
    {"LOS_scatter_radius", 0.0, QD_NONNEG, -1, -1, 3, 1, "not used"},
    {"SC_lambda", 0.0, QD_NONNEG, -1, -1, 4, 1, "decorrelation distance [m]; 0 = disabled"},
    {"LNS_ksi", 3.0, QD_NONNEG, -1, -1, 3, 2, "per cluster shadowing STD [dB]"},
    {"GR_enabled", 0.0, QD_NONNEG, -1, -1, 5, 1, "Enables the explicit ground reflection model"},
    {"asD_ds", 0.0, QD_CORR, 0, 3, 2, 1, "DS vs. ASD"},
    {"asA_ds", 0.0, QD_CORR, 0, 4, 2, 1, "DS vs. ASA"},
    {"asA_sf", 0.0, QD_CORR, 2, 4, 2, 1, "SF vs. ASA"},
    {"asD_sf", 0.0, QD_CORR, 2, 3, 2, 1, "SF vs. ASD"},
    {"ds_sf", 0.0, QD_CORR, 0, 2, 2, 1, "DS vs. SF"},
    {"asD_asA", 0.0, QD_CORR, 3, 4, 2, 1, "ASD vs. ASA"},
    {"asD_kf", 0.0, QD_CORR, 1, 3, 2, 1, "KF vs. ASD"},
    {"asA_kf", 0.0, QD_CORR, 1, 4, 2, 1, "KF vs. ASA"},
    {"ds_kf", 0.0, QD_CORR, 0, 1, 2, 1, "DS vs. KF"},
    {"sf_kf", 0.0, QD_CORR, 1, 2, 2, 1, "KF vs. SF"},
    {"esD_ds", 0.0, QD_CORR, 0, 5, 2, 1, "DS vs. ESD"},
    {"esA_ds", 0.0, QD_CORR, 0, 6, 2, 1, "DS vs. ESA"},
    {"esA_sf", 0.0, QD_CORR, 2, 6, 2, 1, "SF vs. ESA"},
    {"esD_sf", 0.0, QD_CORR, 2, 5, 2, 1, "SF vs. ESD"},
    {"esD_esA", 0.0, QD_CORR, 5, 6, 2, 1, "ESD vs. ESA"},
    {"esD_asD", 0.0, QD_CORR, 3, 5, 2, 1, "ASD vs. ESD"},
    {"esD_asA", 0.0, QD_CORR, 4, 5, 2, 1, "ASA vs. ESD"},
    {"esA_asD", 0.0, QD_CORR, 3, 6, 2, 1, "ASD vs. ESA"},
    {"esA_asA", 0.0, QD_CORR, 4, 6, 2, 1, "ASA vs. ESA"},
    {"esD_kf", 0.0, QD_CORR, 1, 5, 2, 1, "KF vs. ESD"},
    {"esA_kf", 0.0, QD_CORR, 1, 6, 2, 1, "KF vs. ESA"},
    {"xpr_ds", 0.0, QD_CORR, 0, 7, 2, 1, "DS vs. XPR"},
    {"xpr_kf", 0.0, QD_CORR, 1, 7, 2, 1, "KF vs. XPR"},
    {"xpr_sf", 0.0, QD_CORR, 2, 7, 2, 1, "SF vs. XPR"},
    {"xpr_asd", 0.0, QD_CORR, 3, 7, 2, 1, "ASD vs. XPR"},
    {"xpr_asa", 0.0, QD_CORR, 4, 7, 2, 1, "ASA vs. XPR"},
    {"xpr_esd", 0.0, QD_CORR, 5, 7, 2, 1, "ESD vs. XPR"},
    {"xpr_esa", 0.0, QD_CORR, 6, 7, 2, 1, "ESA vs. XPR"},
    // New keys for 3GPP TR 38.901 v19.5.0
    {"LSP_fc_min", 0.0, QD_NONNEG, -1, -1, 0, 9, "lower limit of fc for the frequency-dependent terms [GHz]"},
    {"ES_D_mu_B", 0.0, QD_REAL, -1, -1, 0, 6, "RX height-dep. of ESD [log10(deg)/m]"},
    {"ES_D_mu_C", 0.0, QD_REAL, -1, -1, 0, 6, "dep. of ESD on abs(hUT - hBS) [log10(deg)/m]"},
    {"ES_D_mu_D", 0.0, QD_REAL, -1, -1, 0, 6, "dep. of ESD on max(hUT - hBS, 0) [log10(deg)/m]"},
    {"EoD_offset_e", 0.0, QD_REAL, -1, -1, 8, 1, "constant of the EoD offset [deg]; TR value with opposite sign"},
    {"EoD_offset_e_gamma", 0.0, QD_REAL, -1, -1, 8, 1, "freq.-dep. of the constant [deg/log10(GHz)]; TR value with opposite sign"},
    {"EoD_offset_a", 0.0, QD_REAL, -1, -1, 8, 1, "distance exponent"},
    {"EoD_offset_a_gamma", 0.0, QD_REAL, -1, -1, 8, 1, "freq.-dep. of the distance exponent [1/log10(GHz)]"},
    {"EoD_offset_b", 0.0, QD_NONNEG, -1, -1, 8, 1, "minimum 2D distance [m]; 0 = no power term"},
    {"EoD_offset_c", 0.0, QD_REAL, -1, -1, 8, 1, "constant of the exponent"},
    {"EoD_offset_c_gamma", 0.0, QD_REAL, -1, -1, 8, 1, "freq.-dep. of the exponent [1/log10(GHz)]"},
    {"EoD_offset_c_hUT", 0.0, QD_REAL, -1, -1, 8, 1, "RX height-dep. of the exponent [1/m]"},
    {"EoD_offset_hcl", 0.0, QD_NONNEG, -1, -1, 8, 1, "clutter height [m] of the geometric EoD offset; 0 = disabled"},
    {"PerClusterDS_fc_min", 0.0, QD_NONNEG, -1, -1, 3, 3, "lower limit of fc for the cluster delay spread [GHz]"},
    {"PerClusterES_D_scale", 0.0, QD_NONNEG, -1, -1, 3, 3, "cluster ESD as fraction of 10^(ESD mean); 0 = use PerClusterES_D"},
    {"absTOA_Lmax", 0.0, QD_NONNEG, -1, -1, 6, 1, "largest room dimension L [m]; the excess delay is bounded by 2L/c; 0 = no bound"},
    {"NumClusters_min", 0.0, QD_INT_NONNEG, -1, -1, 3, 1, "lower end of the cluster-count range, counted like NumClusters"},
    {"NF_k1", 0.0, QD_INT_NONNEG, -1, -1, 9, 1, "number of strongest clusters with scaling factor 1"},
    {"NF_alpha", 0.0, QD_NONNEG, -1, -1, 9, 1, "parameter alpha of the Beta distribution"},
    {"NF_beta", 0.0, QD_NONNEG, -1, -1, 9, 1, "parameter beta of the Beta distribution"},
    {"SNS_mu", 0.0, QD_UNIT, -1, -1, 9, 2, "mean of the probability Pr_sns"},
    {"SNS_sigma", 0.0, QD_NONNEG, -1, -1, 9, 2, "STD of the probability Pr_sns"},
    {"SNS_A", 0.0, QD_NONNEG, -1, -1, 9, 2, "visibility region parameter A"},
    {"SNS_B", 0.0, QD_NONNEG, -1, -1, 9, 2, "visibility region parameter B"},
    {"SNS_R", 0.0, QD_NONNEG, -1, -1, 9, 2, "visibility region parameter R [dB]"},
    {"SNS_delta_var", 0.0, QD_NONNEG, -1, -1, 9, 2, "variance of the visibility region parameter delta"},
    {"PolVar_sigma", 0.0, QD_NONNEG, -1, -1, 9, 3, "STD of the polarization variability powers [dB]"},
    // TR 38.901 v19.5.0: LOS/indoor state correlation distances, maximum indoor distance, InF delay-spread terms
    {"LOS_state_lambda", 0.0, QD_NONNEG, -1, -1, 4, 2, "decorrelation distance of the LOS/NLOS state [m], stated in the NLOS file of a scenario; 0 = not stated here"},
    {"IO_state_lambda", 0.0, QD_NONNEG, -1, -1, 4, 2, "decorrelation distance of the indoor/outdoor state [m], stated in the O2I files of a scenario; 0 = not stated here"},
    {"O2I_d2D_in_max", 0.0, QD_NONNEG, -1, -1, 3, 4, "maximum indoor 2D distance of O2I UTs [m]; 0 = not stated"},
    {"DS_VS", 0.0, QD_NONNEG, -1, -1, 0, 1, "volume-to-surface ratio V/S of the hall [m]; 0 = no V/S term"},
    {"DS_VS_a", 0.0, QD_NONNEG, -1, -1, 0, 1, "factor a of the DS term log10(a V/S + b) [1/m]"},
    {"DS_VS_b", 0.0, QD_NONNEG, -1, -1, 0, 1, "offset b of the DS term log10(a V/S + b)"},
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
        return (v >= 1.0 && v <= 1000.0 && std::floor(v) == v) ? nullptr : "must be an integer from 1 to 1000";
    case QD_INT_NONNEG:
        return (v >= 0.0 && v <= 1000.0 && std::floor(v) == v) ? nullptr : "must be an integer from 0 to 1000";
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
    // From the reader a failing key always has a line: defaults pass every check, the cross-key checks name a key
    // that was set in the file
    auto where = [&fn](arma::uword ln) -> std::string
    { return fn.empty() ? "" : fn + ", line " + std::to_string(ln) + ": "; };

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

    const arma::uword i_vs = qd_scenpar_find("DS_VS"), i_vsa = qd_scenpar_find("DS_VS_a"), i_vsb = qd_scenpar_find("DS_VS_b");
    if (val[i_vs] > 0.0 && val[i_vsa] == 0.0 && val[i_vsb] == 0.0)
        throw std::invalid_argument(where(line == nullptr ? 0 : line[i_vs]) +
                                    "Values of 'DS_VS_a' and 'DS_VS_b' must not both be 0 if 'DS_VS' is > 0.");
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

    // Cholesky factorization; a NaN pivot fails the test
    double L[64];
    return qd_gbsm_chol8(X, L);
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

// Split a file into lines at '\n' and remove one trailing '\r' per line; a UTF-8 byte order mark at the start of the
// file is removed
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
    if (!lines.empty() && lines[0].compare(0, 3, "\xEF\xBB\xBF") == 0)
        lines[0].erase(0, 3);
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

// Result of 'qd_conf_parse_number'
enum qd_conf_num : unsigned char
{
    QD_NUM_OK,      // A number; 'v' is set
    QD_NUM_INVALID, // Not a number by the grammar
    QD_NUM_RANGE    // Beyond the range of a double
};

// Parse one number token: [+-] digits [. digits] [e|E [+-] digits], with at least one digit before the
// exponent, or [+-] Inf in any case; anything else (NaN, complex, hex, trailing text) is QD_NUM_INVALID
// - A number beyond the range of a double is QD_NUM_RANGE; a number below the smallest subnormal is the 0 with its
//   sign, which is what strtod returns in the C locale
// - 'v' is written only for QD_NUM_OK
// - The conversion uses a stream with the classic locale, not std::from_chars: in libstdc++ the double overload of
//   from_chars needs GLIBCXX_3.4.29, which the libstdc++ shipped with MATLAB R2024a does not provide
static qd_conf_num qd_conf_parse_number(const std::string &s, double &v)
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
        return QD_NUM_OK;
    }

    // Decimal order of the first non-zero digit: digits before the point counted from it, minus the zeros after
    // the point before it; plus the exponent (saturated)
    long long order = 0;
    bool nonzero = false;
    size_t n_digit = 0;
    while (i < n && s[i] >= '0' && s[i] <= '9')
    {
        nonzero = nonzero || s[i] != '0';
        order += nonzero ? 1 : 0;
        ++i, ++n_digit;
    }
    if (i < n && s[i] == '.')
    {
        ++i;
        while (i < n && s[i] >= '0' && s[i] <= '9')
        {
            nonzero = nonzero || s[i] != '0';
            order -= nonzero ? 0 : 1;
            ++i, ++n_digit;
        }
    }
    if (n_digit == 0)
        return QD_NUM_INVALID;

    if (i < n && (s[i] == 'e' || s[i] == 'E'))
    {
        ++i;
        bool negative_exp = false;
        if (i < n && (s[i] == '+' || s[i] == '-'))
            negative_exp = s[i++] == '-';
        size_t n_exp = 0;
        long long exp = 0;
        while (i < n && s[i] >= '0' && s[i] <= '9')
        {
            exp = std::min(exp * 10 + (s[i] - '0'), 1000000000LL);
            ++i, ++n_exp;
        }
        if (n_exp == 0)
            return QD_NUM_INVALID;
        order += negative_exp ? -exp : exp;
    }
    if (i != n)
        return QD_NUM_INVALID;

    // The text is a plain decimal number; convert it independent of the global locale. The stream fails on
    // overflow (LWG 23) and returns the value of strtod on underflow.
    std::istringstream is(s);
    is.imbue(std::locale::classic());
    double w = 0.0;
    is >> w;
    if (!is.fail())
    {
        v = w;
        return QD_NUM_OK;
    }
    return (nonzero && order > 0) ? QD_NUM_RANGE : QD_NUM_INVALID;
}

// Values of a parameter set in table order; keys that are not in 'scenpar' have their default
// - 'given' (optional, qd_scenpar_n_key elements) marks the keys that are in 'scenpar'
// - Throws std::invalid_argument for a name that is not in the key table
static void qd_scenpar_values(const std::unordered_map<std::string, double> &scenpar, double *val, bool *given = nullptr)
{
    for (arma::uword i = 0; i < qd_scenpar_n_key; ++i)
    {
        val[i] = qd_scenpar_keys[i].def;
        if (given != nullptr)
            given[i] = false;
    }
    for (const auto &entry : scenpar)
    {
        const arma::uword ind = qd_scenpar_find(entry.first);
        if (ind == qd_scenpar_n_key)
            throw std::invalid_argument("Unknown key '" + entry.first + "' in 'scenpar'.");
        val[ind] = entry.second;
        if (given != nullptr)
            given[ind] = true;
    }
}

// Sections of a written conf file; the index is the position in 'ref_sec', the write order is 'qd_conf_section_order'
static const char *const qd_conf_sections[] = {
    "Large scale distributions",
    "Large-Scale fading decorrelation distances",
    "Inter-parameter correlations",
    "Model parameters",
    "Decorrelation distance for the small-scale fading spatial consistency",
    "Parameters for the ground reflection extension",
    "Absolute time of arrival offset",
    "Path-loss model",
    "EoD offset (elevation of departure, TR: ZOD offset)",
    "Optional modelling components"};

static const unsigned char qd_conf_section_order[] = {0, 1, 2, 3, 8, 4, 5, 6, 9, 7};
static const unsigned char QD_CONF_SEC_CORR = 2, QD_CONF_SEC_MODEL = 3, QD_CONF_SEC_PL = 7, QD_CONF_N_SEC = 10;

// LSPs in the order of the cross-correlation matrix: key prefix, label, name of the random variable
static const char *const qd_conf_lsp[8][3] = {{"DS", "DS", "Xds"}, {"KF", "KF", "Xkf"}, {"SF", "SF", "Xsf"},
                                              {"AS_D", "ASD", "Xasd"}, {"AS_A", "ASA", "Xasa"}, {"ES_D", "ESD", "Xesd"},
                                              {"ES_A", "ESA", "Xesa"}, {"XPR", "XPR", "Xxpr"}};

// Comments of the path-loss keys; the first row whose name matches and whose model is empty or equal to 'PL_model' wins
static const char *const qd_conf_pl_comments[][3] = {
    {"A", "constant", "Reference PL in [dB]"},
    {"A", "", "TX-RX 3D dist.-dep. of the PL [dB/log10(m)]"},
    {"A1", "nlos", "TX-RX 3D dist.-dep. of the LOS-PL before break-point [dB/log10(m)]"},
    {"A1", "", "TX-RX 3D dist.-dep. of the PL before break-point [dB/log10(m)]"},
    {"A2", "nlos", "TX-RX 3D dist.-dep. of the LOS-PL after break-point [dB/log10(m)]"},
    {"A2", "", "TX-RX 3D dist.-dep. of the PL after break-point [dB/log10(m)]"},
    {"A3", "", "TX-RX 3D dist.-dep. of the PL after second break-point [dB/log10(m)]"},
    {"An", "", "TX-RX 3D dist.-dep. of NLOS-PL [dB/log10(m)]"},
    {"B", "nlos", "Reference LOS-PL in [dB] @ 1 GHz, 1 m TX-RX dist."},
    {"B", "", "Reference PL in [dB] @ 1 GHz, 1 m TX-RX dist."},
    {"Bn", "", "Reference NLOS-PL in [dB] @ 1 GHz, 1 m TX-RX dist."},
    {"C", "nlos", "Freq.-dep. of the LOS-PL in [dB/log10(GHz)]"},
    {"C", "", "Freq.-dep. of the PL in [dB/log10(GHz)]"},
    {"Cn", "", "Freq.-dep. of the NLOS-PL in [dB/log10(GHz)]"},
    {"D", "nlos", "TX-RX 3D dist.-dep. of LOS-PL [dB/m]"},
    {"D", "satellite", "Elevation angle dep. of PL [dB/rad]"},
    {"D", "", "TX-RX 3D dist.-dep. of PL [dB/m]"},
    {"Dn", "", "TX height-dep. of the NLOS-PL in [dB/log10(m)]"},
    {"D1n", "", "TX height-dep. of the NLOS-PL in [dB/log10(m)/m]"},
    {"D2n", "", "TX height-dep. of the NLOS-PL in [dB/log10(m)/m^2]"},
    {"D3n", "", "TX height-dep. of the NLOS-PL in [dB/m]"},
    {"E", "", "Breakpoint scaling factor"},
    {"E1", "", "Breakpoint scaling factor for first BP in [s/m]"},
    {"E2", "", "Breakpoint scaling factor for second BP in [s/m]"},
    {"En", "", "RX height-dep. of the NLOS-PL in [dB/log10(m)]"},
    {"E1n", "", "RX height-dep. of the NLOS-PL in [dB/log10(m)/m]"},
    {"E2n", "", "RX height-dep. of the NLOS-PL in [dB/log10(m)/m^2]"},
    {"E3n", "", "RX height-dep. of the NLOS-PL in [dB/m]"},
    {"Fn", "", "Combined TX-height and freq.-dep. of the NLOS-PL in [dB/(log10(m)*log10(GHz))]"},
    {"G1n", "", "RX height-dep. of the NLOS-PL on the square of the MT height [dB/(log10(m))^2]"},
    {"G2n", "", "RX height-dep. factor"},
    {"hE", "", "Environment height in [m]"},
    {"hE1", "", "Environment height for first BP in [m]"},
    {"hE2", "", "Environment height for second BP in [m]"},
    {"sig1", "", "Shadow Fading STD before breakpoint [dB]"},
    {"sig2", "", "Shadow Fading STD after breakpoint [dB]"},
    {"sig3", "", "Shadow Fading STD after second breakpoint [dB]"},
    {"hE_rand", "", "1 = random environment height (TR 38.901 Table 7.4.1-1, note 1); 0 = fixed hE"},
    {"LOSmax", "", "0 = no maximum with the LOS-PL; 1 = PL = max( PL_LOS, PLn )"},
    {"Ax", "", "TX-RX 3D dist.-dep. of the additional PL term in the maximum [dB/log10(m)]"},
    {"Bx", "", "Reference PL of the additional term in [dB] @ 1 GHz, 1 m TX-RX dist."},
    {"Cx", "", "Freq.-dep. of the additional PL term in [dB/log10(GHz)]"},
    {"usePLa", "", "Enable/disable attenuation by atmospheric gases"}};

// Formula blocks of the path-loss section, written in this order; first entry: the models, each enclosed in '|'
static const char *const qd_conf_pl_formulas[][2] = {
    {"|logdist|",
     "% Formula for Hata pathloss model:\n"
     "% (Distance in meters, frequency in GHz)\n"
     "%\n"
     "%    PL = A * log10( d3D ) + B + C * log10( fGHz )\n"},
    {"|dual_slope|nlos|",
     "% Formula for dual-slope (LOS) pathloss model:\n"
     "% (Distance in meters, frequency in GHz)\n"
     "%\n"
     "%     PL = PL1 for d2D <= dBP | PL2 for d2D > dBP\n"
     "%    PL1 = A1 * log10( d3D ) + B + C * log10( fGHz ) + D * d3D\n"
     "%    PL2 = PL1( dBP ) + A2 * log10( d3D / dBP )\n"
     "%    dBP = E * ( hBS-hE ) * ( hMS-hE ) * fGHz\n"},
    {"|tripple_slope|",
     "% Formula for triple-slope (LOS+GR) pathloss model:\n"
     "% (Distance in meters, frequency in GHz)\n"
     "%\n"
     "%     PL = PL1 for d2D <= dBP1 | PL2 for dBP1 < d2D <= dBP2 | PL3 for d2D > dBP2\n"
     "%    PL1 = A1 * log10( d3D ) + B + C * log10( fGHz ) + D * d3D\n"
     "%    PL2 = PL1( dBP1 ) + A2 * log10( d3D / dBP1 )\n"
     "%    PL3 = PL2( dBP2 ) + A3 * log10( d3D / dBP2 )\n"
     "%   dBP1 = E1 * ( hBS-hE1 ) * ( hMS-hE1 ) * fGHz\n"
     "%   dBP2 = E2 * ( hBS-hE2 ) * ( hMS-hE2 ) * fGHz\n"},
    {"|nlos|",
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
     "%     PL = max( PL_LOS, PLn )\n"}};

// Number text that 'qd_conf_parse_number' reads back to the same double: the fewest of 15, 16, 17 significant digits
static std::string qd_conf_number(double v)
{
    if (std::isinf(v))
        return v < 0.0 ? "-Inf" : "Inf";
    std::string s;
    for (int prec = 15; prec <= 17; ++prec)
    {
        std::ostringstream os;
        os.imbue(std::locale::classic());
        os.precision(prec);
        os << v;
        s = os.str();
        double w;
        if (qd_conf_parse_number(s, w) == QD_NUM_OK && std::memcmp(&v, &w, sizeof(double)) == 0)
            break;
    }
    return s; // 17 digits always read back for a correctly rounding parser
}

// One assignment line: name padded to 19 characters, value padded to 13, comment
static std::string qd_conf_line(const std::string &name, const std::string &value, const std::string &comment)
{
    const std::string v = (!value.empty() && value[0] == '-') ? value : " " + value;
    std::string s = name + " = " + std::string(name.size() < 19 ? 19 - name.size() : 0, ' ') + v;
    if (!comment.empty())
        s += std::string(v.size() < 13 ? 13 - v.size() : 0, ' ') + " % " + comment;
    return s + "\n";
}

// True if 's' contains a line break
static bool qd_conf_has_newline(const std::string &s)
{
    return s.find('\n') != std::string::npos || s.find('\r') != std::string::npos;
}

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!SECTION_DESC
A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of environment, the
*scenario* (e.g. urban macro-cell with or without line of sight), instead of from the geometry of one site. The
functions of this section implement the large-scale part of the model of 3GPP TR 38.901, clause 7.5 steps 1 to 4:
path loss, shadow fading, LOS probability, O2I penetration loss and the large-scale parameters (LSPs) of each link,
i.e. the spreads of delays and angles, the K-factor and the cross-polarization ratio.<br><br>

The statistics of a scenario are read from a conf file by [[conf_file_read]]. The folder `data/config` holds 23
files for TR 38.901 v19.5.0; [[scenpar_validate]] completes a parameter set from another source,
[[conf_file_list]] finds the files and [[conf_file_write]] writes them.<br><br>

Per drop (layout): [[o2i_loss]] gives the penetration loss and the indoor distance of each indoor UT; then
[[los_probability]] evaluates the LOS probability at the outdoor distance `d2D - d2D-in`. The caller draws the LOS
state and the indoor state with [[sos_rand]] / [[sos_randn]] at the decorrelation distances of the keys
`LOS_state_lambda` and `IO_state_lambda`, and the O2I variables at the distances of TR 38.901 clause 7.6.3.3 (see
[[o2i_loss]]). The two state distances apply to all links of a scenario and are stated once in the TR 38.901 files:
`LOS_state_lambda` in the NLOS file, `IO_state_lambda` in the O2I files; 0 in the other files means "not stated
here". Per BS site and scenario (builder): [[lsf_init_phases]] draws the random
phases once; [[lsf_draw]] evaluates them at the links (columns 0 DS, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA, 7 XPR,
8 absolute TOA: Normal; 9 ground permittivity, 10 UMa environment height: Uniform; each at the decorrelation
distance of its key); [[path_loss]] returns the path loss and the SF standard deviation; [[lsp_generate]] turns the
random values into the LSPs of each link and frequency.<br><br>

Where TR 38.901 and QuaDRiGa differ, the TR behaviour is the default; QuaDRiGa's behaviour is available
through an argument where it is still wanted (3D correlation, symmetric square root of the LSP
cross-correlation matrix, no clipping of the spreads, outdoor path-loss distance of O2I links). Where
QuaDRiGa's behaviour was an error, only the corrected rule is implemented.<br><br>

Units: positions in m, frequencies in Hz, path loss in dB, DS in s, KF, SF and XPR as linear power ratios, angular
spreads in degrees.
SECTION_DESC!*/

/*!MD
# conf_file_read
Read and validate a QuaDRiGa scenario configuration file

- A conf file is a text file with one `Key = value` assignment per line; `%` starts a comment; blank lines
  are allowed; line ends can be LF or CRLF; space and tab are white space; a UTF-8 byte order mark at the start
  of the file is skipped
- Keys are case-sensitive names of the characters `A-Z`, `a-z`, `0-9` and `_`; there are four kinds:
  - Scenario parameters: `n_key = 171` numeric keys with one number each, returned in `scenpar`. Keys that
    are not in the file have their default value. The names and their order are returned in `key_order`.
  - `SubpathMethod`: one of the words `legacy`, `Laplacian`, `mmMAGIC`; default `legacy`
  - Path-loss parameters: every key that starts with `PL_`. `PL_model` has one name as value. All others
    have one or more numbers separated by white space and are returned in `plpar` without the prefix.
  - `ShortName`: an alias for the scenario, one name; can occur several times with different values
- Any other key is ignored and reported in `ignored_keys`; its value is not examined
- A key that is stated more than once has its last value
- Numbers are decimal with an optional sign, an optional decimal point and an optional exponent (`e` or
  `E`), for example `-6.28`, `.5`, `6.2e-07`; `Inf`, `+Inf` and `-Inf` are accepted in any case
- A number beyond the range of a double (`1e400`) is an error "out of range"; a number below the smallest
  subnormal reads as `0` (or `-0`)
- Numbers are read independent of the global locale (the decimal point is always `.`)
- Elevation angles are used throughout (+90 deg is up); no key refers to a zenith angle
- The 171 keys are 138 numeric keys of QuaDRiGa and 33 keys for 3GPP TR 38.901 v19.5.0. With their
  default of 0 the new keys have no effect.
- The QuaDRiGa key `GR_epsilon` (fixed relative permittivity of the ground) is not supported and is
  reported in `ignored_keys` like any other unknown key; the permittivity is complex and is set in the simulation
- The 28 correlation keys (`asD_ds` to `xpr_esa`) define the symmetric matrix `lsp_xcorr` with unit diagonal
- Throws `std::runtime_error` if the file cannot be opened or is a directory
- Throws `std::invalid_argument` with file name and line number for a line that is not blank, not a comment
  and not a well-formed assignment: no `=`, more or less than one name before the `=`, a missing value, a
  word where a number is required or the reverse, a second value or other text after the value, `NaN`,
  complex numbers, a number out of range, a `PL_` key without a name, a `ShortName` value stated twice
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
- **`scenpar`** — Numeric scenario parameters; previous content is discarded; `n_key = 171` entries
- **`subpath_method`** *(optional)* — Value of `SubpathMethod`; `legacy` if not in the file
- **`pl_model`** *(optional)* — Value of `PL_model`; empty if not in the file
- **`plpar`** *(optional)* — Path-loss parameters; keys without the prefix `PL_`; `PL_model` is not included;
  empty if the file has no such key; `n_pl` entries with one or more values each
- **`lsp_xcorr`** *(optional)* — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `[8, 8]`
- **`key_order`** *(optional)* — Canonical order of the keys in `scenpar`: the QuaDRiGa order without
  `SubpathMethod` (its position is between elements 105 and 106, 0-based), then the 33 new keys; `[n_key]`
- **`pl_order`** *(optional)* — Keys of `plpar` in the order of their first occurrence in the file; `[n_pl]`
- **`short_names`** *(optional)* — Values of all `ShortName` lines in file order; `[n_short]`
- **`ignored_keys`** *(optional)* — Unknown keys, each once, in the order of their first occurrence; `[n_ignored]`

## Returns:
- `true` if `lsp_xcorr` is positive definite, i.e. correlated LSPs can be generated from it

## See also:
- [[scenpar_validate]] (value checks, defaults and the list of validation rules)
- [[conf_file_list]] (find the conf files in a set of directories)
- [[conf_file_write]] (write a conf file)
- [[lsf_init_phases]] (first step of the large-scale model with the scenario parameters)
- [[path_loss]] (evaluates the path-loss model and its parameters)
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
    std::unordered_set<std::string> ignored_set, short_set;

    // Number of a key, with the error message for a text that is not a number or out of range
    auto number = [](const std::string &text, double &v, const std::string &where, const std::string &key)
    {
        const qd_conf_num res = qd_conf_parse_number(text, v);
        if (res == QD_NUM_RANGE)
            throw std::invalid_argument(where + "Value of '" + key + "' is out of range.");
        if (res != QD_NUM_OK)
            throw std::invalid_argument(where + "Value of '" + key + "' is not a number.");
    };

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
            if (!short_set.insert(value).second)
                throw std::invalid_argument(where + "ShortName '" + value + "' is stated twice.");
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
                number(tok[i], v[i], where, key);

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
            number(value, val[ind], where, key);
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
  with its default value, so that `scenpar` has all `n_key = 171` entries on return
- An empty `scenpar` returns the complete default parameter set
- A set with the 138 numeric keys of QuaDRiGa is completed with the 33 keys for 3GPP TR 38.901 v19.5.0, which
  have no effect at their default of 0
- A name in `scenpar` that is not one of the 171 keys is an error, also with `validate = false`; this includes
  the QuaDRiGa key `GR_epsilon`, which is not supported
- Value checks with `validate = true`; `NaN` is an error for every key:
  - Integer from 1 to 1000: `NumClusters`, `NumSubPaths`
  - Finite and > 0: `r_DS`
  - Finite and >= 0: the 8 decorrelation distances (`DS_lambda` ... `XPR_lambda`), `SC_lambda`,
    `absTOA_lambda`, `PerClusterDS`, `PerClusterDS_min`, `PerClusterAS_D`, `PerClusterAS_A`, `PerClusterES_D`,
    `PerClusterES_A`, `LNS_ksi`, `LOS_scatter_radius`, `GR_enabled`, `LSP_fc_min`, `EoD_offset_b`,
    `EoD_offset_hcl`, `PerClusterDS_fc_min`, `PerClusterES_D_scale`, `absTOA_Lmax`, `NF_alpha`, `NF_beta`,
    `SNS_sigma`, `SNS_A`, `SNS_B`, `SNS_R`, `SNS_delta_var`, `PolVar_sigma`, `LOS_state_lambda`,
    `IO_state_lambda`, `O2I_d2D_in_max`, `DS_VS`, `DS_VS_a`, `DS_VS_b`
  - Integer from 0 to 1000: `NumClusters_min`, `NF_k1`
  - In the range from 0 to 1: `SNS_mu`
  - In the range from -1 to 1: the 28 correlations (`asD_ds` to `xpr_esa`)
  - All other keys accept any value including `Inf` and `-Inf`
  - `NumClusters_min` must not exceed `NumClusters`
  - `NF_alpha` and `NF_beta` must be both 0 or both > 0
  - `DS_VS_a` and `DS_VS_b` must not both be 0 if `DS_VS` is > 0
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
  added with their default; `n_key = 171` entries on return
- **`subpath_method`** *(optional)* — Value of `SubpathMethod`, the only parameter that is not numeric
- **`validate`** *(optional)* — `false`: only complete with defaults, no value checks

## Outputs:
- **`lsp_xcorr`** *(optional)* — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `[8, 8]`
- **`key_order`** *(optional)* — Canonical order of the keys in `scenpar`: the QuaDRiGa order without
  `SubpathMethod` (its position is between elements 105 and 106, 0-based), then the 33 new keys; `[n_key]`

## Returns:
- `true` if `lsp_xcorr` is positive definite (all 8 pivots of the factorization are > 0); `false` also if it
  contains `NaN`

## See also:
- [[conf_file_read]] (read the parameters from a conf file)
- [[conf_file_write]] (applies the same checks before writing)
MD!*/

bool quadriga_lib::scenpar_validate(std::unordered_map<std::string, double> &scenpar,
                                    const std::string &subpath_method,
                                    arma::mat *lsp_xcorr,
                                    std::vector<std::string> *key_order,
                                    bool validate)
{
    double val[sizeof(qd_scenpar_keys) / sizeof(qd_scenpar_keys[0])];
    bool given[sizeof(qd_scenpar_keys) / sizeof(qd_scenpar_keys[0])];
    qd_scenpar_values(scenpar, val, given);

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

- Lists the regular files and symbolic links to regular files whose name ends in `.conf` (case-sensitive) and
  is longer than `.conf`, not recursive. The scenario name is the file name without this extension, whatever
  characters it has.
- Directories are processed in the given order, the files of one directory in byte-wise sorted order of their
  names (digits before upper case before lower case)
- A file whose name was already found in an earlier directory is skipped completely, so the first directory
  wins. This allows a local folder to override the files of a shared folder.
- With `parse_shortnames = true` each file is read and one entry is added right after the file for every
  line `ShortName = name` that is not a comment. Other lines are not examined; a UTF-8 byte order mark at the
  start of the file is skipped.
- Throws `std::invalid_argument` if a short name equals an earlier entry, if a file name equals an earlier
  short name, or if the value of a `ShortName` line is not one name of the characters `A-Z`, `a-z`, `0-9`, `_`
- Throws `std::runtime_error` if a listed file cannot be opened for reading the short names
- A directory that does not exist, is not a directory or cannot be listed is skipped; an empty string is the
  current directory
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

/*!MD
# conf_file_write
Write a QuaDRiGa scenario configuration file

- Writes scenario parameters, path-loss parameters and short names in the layout of the conf files shipped with
  QuaDRiGa, so that [[conf_file_read]] returns the same values bit for bit
- Header: `% Config file for scenario "name"` with the file name without the extension `.conf`, then the
  reference `ref`, the validity range and one line `ShortName = name` per short name
- Up to 10 sections follow, each with a header and, if the entry of `ref_sec` is not empty, a line
  `% See: ref; ref_sec`. Index in `ref_sec` and write order:
  - 0: Large scale distributions, with an overview formula per LSP; 1: Large-Scale fading decorrelation distances;
    2: Inter-parameter correlations, with the correlation matrix as comment; 3: Model parameters;
    8: EoD offset; 4: Decorrelation distance for the small-scale fading spatial consistency;
    5: Parameters for the ground reflection extension; 6: Absolute time of arrival offset;
    9: Optional modelling components; 7: Path-loss model
  - The indices 0 to 7 are those of QuaDRiGa, so its 8 section references can be passed unchanged
- Within a section, keys are grouped into paragraphs separated by a blank line; correlations are written in the
  row order of the matrix, all other keys in the order of `key_order`; `SubpathMethod` stands before
  `LOS_scatter_radius`
- The overview formula of an LSP shows its mean and STD with the frequency, distance, height and elevation terms;
  with `LSP_fc_min > 0` the frequency is written as `max( fGHz, fc_min )`; the ESD line also has the terms of
  `ES_D_mu_A` to `ES_D_mu_D` and, if `ES_D_mu_min` is not `-Inf`, the lower limit of the mean; the DS line has
  the term `log10( a · V/S + b )` when `DS_VS > 0` (a = `DS_VS_a`, b = `DS_VS_b`; `V/S` stays symbolic)
- Each line is `name = value % comment`, with the name padded to 19 and the value to 13 characters. The comment
  states the unit; for the mean and STD of an LSP it adds the reference point of the frequency, distance, height
  and elevation terms, e.g. `@ 1 GHz`.
- The path-loss section has the formula blocks of the model (`logdist`, `dual_slope`, `nlos`, `tripple_slope`),
  the line `PL_model`, then the keys of `plpar` with the prefix `PL_`; known keys have a comment that can depend on
  the model; several values of one key are separated by a space
- Numbers are written with up to 15 significant digits, or with 16 or 17 where 15 do not read back to the same
  double in [[conf_file_read]], independent of the global locale; `Inf` and `-Inf` are written as such, `-0` is
  kept
- With `write_defaults = false` a key is written only if its value differs from the default, compared bit for bit;
  `SubpathMethod` only if it is not `legacy`; a section without a written key is left out
- With `write_defaults = true` all 171 keys and `SubpathMethod` are written
- Validation before the file is opened: the file name has no line break; the value checks of [[scenpar_validate]]
  (unknown keys, `NaN`, classes, `subpath_method`); `pl_model` empty or one name; `plpar` keys are names other
  than `model`, each with at least one value and no `NaN`; `pl_order` names each key of `plpar` once; short names
  are names and distinct; `ref` and `ref_sec` have no line break; `ref_sec` has at most 10 entries; `valid_range`
  is empty or `[4, 2]`
- The text is written to a temporary file in the same directory, which must be writable, and then renamed to
  `fn`; a symbolic link `fn` is written through
- A failed call does not create a file and leaves an existing file unchanged; an existing file is replaced and
  keeps its permissions
- Throws `std::invalid_argument` for invalid input; `std::runtime_error` if the file cannot be opened or written,
  or if `fn` exists and is not a regular file that can be opened for writing
- Every file written passes [[conf_file_read]] with `validate = true` and without ignored keys

## Declaration:
```
void quadriga_lib::conf_file_write(
    const std::string &fn,
    const std::unordered_map<std::string, double> &scenpar,
    const std::string &subpath_method = "legacy",
    const std::string &pl_model = "",
    const std::unordered_map<std::string, std::vector<double>> *plpar = nullptr,
    const std::vector<std::string> *pl_order = nullptr,
    const std::vector<std::string> *short_names = nullptr,
    const std::string &ref = "",
    const std::vector<std::string> *ref_sec = nullptr,
    const arma::mat *valid_range = nullptr,
    bool write_defaults = false);
```

## Inputs:
- **`fn`** — Path of the conf file to write
- **`scenpar`** — Numeric scenario parameters; any subset of the `n_key = 171` keys of [[conf_file_read]]; missing
  keys have their default
- **`subpath_method`** *(optional)* — Value of `SubpathMethod`: `legacy`, `Laplacian` or `mmMAGIC`
- **`pl_model`** *(optional)* — Value of `PL_model`; one name; empty: not written
- **`plpar`** *(optional)* — Path-loss parameters; keys without the prefix `PL_`, one or more values each; `[n_pl]`
- **`pl_order`** *(optional)* — Order in which the keys of `plpar` are written; must name each key once; omitted:
  sorted by name; `[n_pl]`
- **`short_names`** *(optional)* — Values of the `ShortName` lines, one name each
- **`ref`** *(optional)* — Literature reference; written in the header and before each non-empty entry of
  `ref_sec`; one line
- **`ref_sec`** *(optional)* — Reference per section, indexed as listed above; one line each; an empty or missing
  entry writes no reference; up to `n_sec = 10` entries
- **`valid_range`** *(optional)* — Validity range, written as comment; rows: frequency in GHz, distance, BS
  height, MT height in m; columns: lower and upper end, one value is written if both are equal; a row with `NaN`
  is not written; empty: no validity range; `[4, 2]`
- **`write_defaults`** *(optional)* — `true`: write all keys; `false`: write only keys that differ from their default

## See also:
- [[conf_file_read]] (read the file back)
- [[scenpar_validate]] (value checks and defaults)
MD!*/

void quadriga_lib::conf_file_write(const std::string &fn,
                                   const std::unordered_map<std::string, double> &scenpar,
                                   const std::string &subpath_method,
                                   const std::string &pl_model,
                                   const std::unordered_map<std::string, std::vector<double>> *plpar,
                                   const std::vector<std::string> *pl_order,
                                   const std::vector<std::string> *short_names,
                                   const std::string &ref,
                                   const std::vector<std::string> *ref_sec,
                                   const arma::mat *valid_range,
                                   bool write_defaults)
{
    // Input validation
    if (fn.empty())
        throw std::invalid_argument("Input 'fn' cannot be empty.");

    // Scenario name in the header: the file name without the extension ".conf"
    std::string scenario = std::filesystem::path(fn).filename().string();
    if (scenario.size() > 5 && scenario.compare(scenario.size() - 5, 5, ".conf") == 0)
        scenario.resize(scenario.size() - 5);
    if (qd_conf_has_newline(scenario))
        throw std::invalid_argument("Input 'fn' must not contain a line break in the file name.");

    double val[sizeof(qd_scenpar_keys) / sizeof(qd_scenpar_keys[0])];
    qd_scenpar_values(scenpar, val);
    qd_scenpar_check(val, subpath_method);

    if (!pl_model.empty() && !qd_conf_is_name(pl_model))
        throw std::invalid_argument("Input 'pl_model' must be empty or one name.");

    std::vector<std::string> pl_keys; // Path-loss keys in write order
    if (plpar != nullptr)
        for (const auto &entry : *plpar)
        {
            if (!qd_conf_is_name(entry.first) || entry.first == "model")
                throw std::invalid_argument("Key '" + entry.first + "' in 'plpar' is not a valid parameter name.");
            if (entry.second.empty())
                throw std::invalid_argument("Key '" + entry.first + "' in 'plpar' has no value.");
            for (double v : entry.second)
                if (std::isnan(v))
                    throw std::invalid_argument("Value of '" + entry.first + "' in 'plpar' must not be NaN.");
            pl_keys.push_back(entry.first);
        }

    if (pl_order == nullptr)
        std::sort(pl_keys.begin(), pl_keys.end());
    else
    {
        std::unordered_set<std::string> seen;
        for (const auto &key : *pl_order)
            if (plpar == nullptr || plpar->count(key) == 0 || !seen.insert(key).second)
                throw std::invalid_argument("Input 'pl_order' must name each key of 'plpar' once.");
        if (seen.size() != pl_keys.size())
            throw std::invalid_argument("Input 'pl_order' must name each key of 'plpar' once.");
        pl_keys = *pl_order;
    }

    if (short_names != nullptr)
    {
        std::unordered_set<std::string> seen;
        for (const auto &name : *short_names)
        {
            if (!qd_conf_is_name(name))
                throw std::invalid_argument("Entries of 'short_names' must be one name each.");
            if (!seen.insert(name).second)
                throw std::invalid_argument("Entries of 'short_names' must be distinct.");
        }
    }

    if (qd_conf_has_newline(ref))
        throw std::invalid_argument("Input 'ref' must not contain a line break.");
    if (ref_sec != nullptr)
    {
        if (ref_sec->size() > QD_CONF_N_SEC)
            throw std::invalid_argument("Input 'ref_sec' can have at most 10 entries.");
        for (const auto &s : *ref_sec)
            if (qd_conf_has_newline(s))
                throw std::invalid_argument("Entries of 'ref_sec' must not contain a line break.");
    }

    if (valid_range != nullptr && !valid_range->empty() && (valid_range->n_rows != 4 || valid_range->n_cols != 2))
        throw std::invalid_argument("Input 'valid_range' must have size [4, 2].");

    // Header: scenario name, reference, validity range, short names
    std::string out = "% Config file for scenario \"" + scenario + "\"\n";
    if (!ref.empty())
        out += "% See: " + ref + "\n";

    if (valid_range != nullptr && !valid_range->empty())
    {
        const char *label[4][2] = {{"% Valid frequency range:        ", " GHz\n"},
                                   {"% Valid distance range:         ", " m\n"},
                                   {"% Valid BS antenna height:      ", " m\n"},
                                   {"% Valid MT antenna height:      ", " m\n"}};
        std::string block;
        for (arma::uword r = 0; r < 4; ++r)
        {
            const double lo = valid_range->at(r, 0), hi = valid_range->at(r, 1);
            if (!std::isnan(lo) && !std::isnan(hi))
                block += label[r][0] + qd_conf_number(lo) + (lo == hi ? "" : " to " + qd_conf_number(hi)) + label[r][1];
        }
        if (!block.empty())
            out += "\n" + block;
    }

    if (short_names != nullptr && !short_names->empty())
    {
        out += "\n";
        for (const auto &name : *short_names)
            out += qd_conf_line("ShortName", name, "You can use this short name to load the file");
    }

    // Value of a key, 0 if the name is not in the table (SF has no mean and no mean-related keys)
    auto get = [&val](const std::string &name)
    {
        const arma::uword i = qd_scenpar_find(name);
        return i == qd_scenpar_n_key ? 0.0 : val[i];
    };

    // Frequency in the overview formulas; with 'LSP_fc_min' > 0 lower frequencies are raised to it
    const double fc_min = get("LSP_fc_min");
    const std::string f_ghz = fc_min > 0.0 ? "max( fGHz, " + qd_conf_number(fc_min) + " )" : "fGHz";

    // Append the term ± abs(c) · var to 's'; nothing if c is 0
    auto add = [](std::string &s, double c, const std::string &var)
    {
        if (c != 0.0)
            s += (c < 0.0 ? " - " : " + ") + qd_conf_number(std::abs(c)) + " * " + var;
    };

    // Overview term: m ± g · log10( omega + fGHz ) ± e · log10( d2D ) ± z · log10( hBS ) ± a · log10( alpha_rad )
    auto term = [&add, &f_ghz](double m, double g, double omega, double e, double z, double a)
    {
        std::string s = qd_conf_number(m);
        add(s, g, "log10( " + (omega != 0.0 ? qd_conf_number(omega) + " + " : std::string()) + f_ghz + " )");
        add(s, e, "log10( d2D )"), add(s, z, "log10( hBS )"), add(s, a, "log10( alpha_rad )");
        return s;
    };

    // Reference point of an LSP mean or STD that depends on frequency, distance, height or elevation
    auto at_suffix = [&get](const std::string &name)
    {
        for (const auto &lsp : qd_conf_lsp)
        {
            const std::string p = lsp[0];
            const bool mu = name == p + "_mu";
            if (!mu && name != p + "_sigma")
                continue;
            std::string s;
            if (get(p + (mu ? "_gamma" : "_delta")) != 0.0)
                s += ", " + qd_conf_number(1.0 - get(p + "_omega")) + " GHz";
            if (get(p + (mu ? "_epsilon" : "_kappa")) != 0.0)
                s += ", 1 m TX-RX dist.";
            if (get(p + (mu ? "_zeta" : "_tau")) != 0.0)
                s += ", 1 m TX height";
            if (get(p + (mu ? "_alpha" : "_beta")) != 0.0)
                s += ", 57.3 deg elevation";
            return s.empty() ? s : " @" + s.substr(1);
        }
        return std::string();
    };

    // Sections: a key is written if its value differs from the default (bit-wise, so that -0 is kept)
    const bool write_subpath = write_defaults || subpath_method != "legacy";
    const arma::uword i_los = qd_scenpar_find("LOS_scatter_radius");
    double X[64];
    qd_scenpar_xcorr(val, X);

    for (unsigned char sec : qd_conf_section_order)
    {
        std::string body; // Section content after the header
        if (sec == QD_CONF_SEC_PL)
        {
            if (pl_model.empty() && pl_keys.empty())
                continue;
            for (const auto &f : qd_conf_pl_formulas)
                if (!pl_model.empty() && std::strstr(f[0], ("|" + pl_model + "|").c_str()) != nullptr)
                    body += std::string(f[1]) + "\n";
            if (!pl_model.empty())
                body += qd_conf_line("PL_model", pl_model, "");
            for (const auto &key : pl_keys)
            {
                std::string value, comment;
                for (double v : plpar->at(key))
                    value += (value.empty() ? "" : " ") + qd_conf_number(v);
                for (const auto &c : qd_conf_pl_comments)
                    if (key == c[0] && (c[1][0] == 0 || pl_model == c[1]))
                    {
                        comment = c[2];
                        break;
                    }
                body += qd_conf_line("PL_" + key, value, comment);
            }
        }
        else
        {
            // Keys of the section, sorted by paragraph, matrix position (correlations) and table order;
            // 'SubpathMethod' (index n_key) stands right before 'LOS_scatter_radius'
            struct entry
            {
                unsigned par, rc;
                arma::uword ord, ind;
            };
            std::vector<entry> list;
            for (arma::uword i = 0; i < qd_scenpar_n_key; ++i)
            {
                const qd_scenpar_key &k = qd_scenpar_keys[i];
                if (k.sec == sec && (write_defaults || std::memcmp(&val[i], &k.def, sizeof(double)) != 0))
                    list.push_back({k.par, k.row < 0 ? 0u : unsigned(k.row * 8 + k.col), 2 * i, i});
            }
            if (sec == QD_CONF_SEC_MODEL && write_subpath)
                list.push_back({qd_scenpar_keys[i_los].par, 0, 2 * i_los - 1, qd_scenpar_n_key});
            if (list.empty())
                continue;
            std::sort(list.begin(), list.end(), [](const entry &a, const entry &b)
                      { return a.par != b.par ? a.par < b.par : (a.rc != b.rc ? a.rc < b.rc : a.ord < b.ord); });

            if (sec == 0) // Overview of the LSP distributions
                for (const auto &lsp : qd_conf_lsp)
                {
                    const std::string p = lsp[0], label = lsp[1];
                    body += "% " + label + std::string(4 - label.size(), ' ') + "= ";
                    if (p != "SF")
                    {
                        std::string mean = term(get(p + "_mu"), get(p + "_gamma"), get(p + "_omega"),
                                                get(p + "_epsilon"), get(p + "_zeta"), get(p + "_alpha"));
                        if (p == "ES_D") // Distance and height terms and the lower limit of the ESD mean
                        {
                            add(mean, get("ES_D_mu_A"), "d2D / 1000"), add(mean, get("ES_D_mu_B"), "hUT");
                            add(mean, get("ES_D_mu_C"), "abs( hUT - hBS )");
                            add(mean, get("ES_D_mu_D"), "max( hUT - hBS, 0 )");
                            if (get("ES_D_mu_min") != -QD_INF)
                                mean = "max( " + qd_conf_number(get("ES_D_mu_min")) + ", " + mean + " )";
                        }
                        if (p == "DS" && get("DS_VS") > 0.0) // InF: volume-to-surface term before the mean
                        {
                            const double m = get("DS_mu");
                            mean = "log10( " + qd_conf_number(get("DS_VS_a")) + " * V/S + " +
                                   qd_conf_number(get("DS_VS_b")) + " )" + (std::signbit(m) ? " - " : " + ") +
                                   qd_conf_number(std::abs(m)) + mean.substr(qd_conf_number(m).size());
                        }
                        body += mean + " + ";
                    }
                    body += std::string(lsp[2]) + " * ( " +
                            term(get(p + "_sigma"), get(p + "_delta"), get(p + "_omega"), get(p + "_kappa"),
                                 get(p + "_tau"), get(p + "_beta")) + " )\n";
                }

            if (sec == QD_CONF_SEC_CORR) // Correlation matrix with 2 significant digits in columns of 7 characters
            {
                body += "%         DS     KF     SF     ASD    ASA    ESD    ESA    XPR\n";
                for (int r = 0; r < 8; ++r)
                {
                    body += r == 3 ? "% R = |  " : "%     |  ";
                    for (int c = 0; c < 8; ++c)
                    {
                        std::ostringstream os;
                        os.imbue(std::locale::classic());
                        os.precision(2);
                        os << X[c * 8 + r];
                        const std::string t = os.str()[0] == '-' ? os.str() : " " + os.str();
                        body += t + std::string(t.size() < 7 ? 7 - t.size() : 0, ' ');
                    }
                    body += "| " + std::string(qd_conf_lsp[r][1]) + "\n";
                }
            }

            for (size_t n = 0; n < list.size(); ++n)
            {
                if (n == 0 ? !body.empty() : list[n].par != list[n - 1].par)
                    body += "\n";
                const arma::uword i = list[n].ind;
                if (i == qd_scenpar_n_key)
                    body += qd_conf_line("SubpathMethod", subpath_method, "subpath mapping method (legacy, Laplacian or mmMAGIC)");
                else
                    body += qd_conf_line(qd_scenpar_keys[i].name, qd_conf_number(val[i]),
                                         qd_scenpar_keys[i].comment + at_suffix(qd_scenpar_keys[i].name));
            }
        }

        const std::string rule = "% " + std::string(98, '=') + "\n";
        out += "\n" + rule + "% " + qd_conf_sections[sec] + "\n";
        if (ref_sec != nullptr && sec < ref_sec->size() && !(*ref_sec)[sec].empty())
            out += "% See: " + (ref.empty() ? "" : ref + "; ") + (*ref_sec)[sec] + "\n";
        out += rule + "\n" + body;
    }

    // All checks passed and the text is complete. It is written to a temporary file in the same directory, which
    // replaces the target only after it was closed without error; a failed call leaves an existing file unchanged.
    std::error_code ec;
    std::filesystem::path target = fn;
    if (std::filesystem::is_symlink(target, ec)) // Write through the link
    {
        target = std::filesystem::weakly_canonical(target, ec);
        if (ec)
            throw std::runtime_error("Cannot open file '" + fn + "' for writing.");
    }

    // An existing target must be a regular file that can be opened for writing; opening it to append changes nothing
    const std::filesystem::file_status status = std::filesystem::status(target, ec);
    const bool exists = std::filesystem::exists(status);
    if (exists && (!std::filesystem::is_regular_file(status) ||
                   !std::ofstream(target, std::ios::out | std::ios::app).is_open()))
        throw std::runtime_error("Cannot open file '" + fn + "' for writing.");

    // Temporary name: the target name, ".tmp" and a hex token of the time and the thread
    auto hex = [](unsigned long long x)
    {
        std::string h;
        do
            h.insert(h.begin(), "0123456789abcdef"[x & 15ULL]);
        while ((x >>= 4) != 0);
        return h;
    };
    const unsigned long long t_now = (unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count();
    const unsigned long long t_id = (unsigned long long)std::hash<std::thread::id>()(std::this_thread::get_id());
    const std::filesystem::path tmp = target.string() + ".tmp" + hex(t_now) + hex(t_id);

    std::ofstream file(tmp, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file '" + fn + "' for writing.");
    file << out;
    file.close();
    if (file.fail())
    {
        std::filesystem::remove(tmp, ec);
        throw std::runtime_error("Cannot write file '" + fn + "'.");
    }

    if (exists) // Keep the permissions of the replaced file
        std::filesystem::permissions(tmp, status.permissions(), ec);

    std::filesystem::rename(tmp, target, ec);
    if (ec)
    {
        std::filesystem::remove(tmp, ec);
        throw std::runtime_error("Cannot write file '" + fn + "'.");
    }
}
