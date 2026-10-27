// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef quadriga_gbsm_H
#define quadriga_gbsm_H

#include <armadillo>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

// If arma::uword and size_t are not the same width (e.g. 64 bit), the compiler will throw an error here
// This allows the use of "uword", "size_t" and "unsigned long long" interchangeably
// This requires a 64 bit platform, but will compile on Linux, Windows and macOS
static_assert(sizeof(arma::uword) == sizeof(unsigned long long), "arma::uword and unsigned long long have different sizes");
static_assert(sizeof(size_t) == sizeof(unsigned long long), "size_t and unsigned long long have different sizes");

namespace quadriga_lib
{

    // ---- Scenario parameters ----

    // Read and validate a QuaDRiGa scenario configuration file
    // - Returns true if the LSP cross-correlation matrix is positive definite
    // - 'scenpar' always holds all n_key numeric keys; keys not in the file have their default value
    bool conf_file_read(const std::string &fn,                                                 // Path to the .conf file
                        std::unordered_map<std::string, double> &scenpar,                      // Output: Numeric scenario parameters, n_key entries
                        std::string *subpath_method = nullptr,                                 // Optional output: Value of 'SubpathMethod'
                        std::string *pl_model = nullptr,                                       // Optional output: Value of 'PL_model', empty if not in file
                        std::unordered_map<std::string, std::vector<double>> *plpar = nullptr, // Optional output: Path-loss parameters, keys without "PL_"
                        arma::mat *lsp_xcorr = nullptr,                                        // Optional output: LSP cross-correlation matrix, [8, 8]
                        std::vector<std::string> *key_order = nullptr,                         // Optional output: Canonical order of the scenpar keys, [n_key]
                        std::vector<std::string> *pl_order = nullptr,                          // Optional output: plpar keys in file order
                        std::vector<std::string> *short_names = nullptr,                       // Optional output: Values of 'ShortName' in file order
                        std::vector<std::string> *ignored_keys = nullptr,                      // Optional output: Keys in the file that are not known
                        bool validate = true);                                                 // Apply the value checks of 'scenpar_validate'

    // Validate scenario parameters and complete them with default values
    // - Returns true if the LSP cross-correlation matrix is positive definite
    // - An empty 'scenpar' returns the complete default parameter set
    bool scenpar_validate(std::unordered_map<std::string, double> &scenpar, // In/out: Missing keys are added with their default
                          const std::string &subpath_method = "legacy",     // Value of 'SubpathMethod'
                          arma::mat *lsp_xcorr = nullptr,                   // Optional output: LSP cross-correlation matrix, [8, 8]
                          std::vector<std::string> *key_order = nullptr,    // Optional output: Canonical order of the scenpar keys, [n_key]
                          bool validate = true);                            // false: only complete with defaults, no value checks

    // List the scenarios defined by the conf files in a set of directories
    // - Returns the number of entries, n_scen
    arma::uword conf_file_list(const std::vector<std::string> &dirs,           // Directories, searched in this order, not recursive
                               std::vector<std::string> &scenarios,            // Output: Scenario names and short names, [n_scen]
                               std::vector<std::string> *file_names = nullptr, // Optional output: File name incl. ".conf" per entry, [n_scen]
                               std::vector<std::string> *file_dirs = nullptr,  // Optional output: Directory per entry, as given in 'dirs', [n_scen]
                               bool parse_shortnames = true);                  // Read the files and add their short names

    // Write a QuaDRiGa scenario configuration file
    // - All inputs are validated and the complete text is composed before the file is opened; an existing file is replaced;
    //   a failed call leaves it unchanged
    // - Numbers are written with up to 15 significant digits, or with 16 or 17 where 15 do not read back to the same
    //   double in 'conf_file_read'; Inf and -Inf are written as such
    // - The scenario name in the header comment is the file name without the extension ".conf"
    void conf_file_write(const std::string &fn,                                                       // Path of the .conf file to write
                         const std::unordered_map<std::string, double> &scenpar,                      // Numeric scenario parameters, any subset of the n_key keys; missing keys have their default
                         const std::string &subpath_method = "legacy",                                // Value of 'SubpathMethod'
                         const std::string &pl_model = "",                                            // Value of 'PL_model', empty: not written
                         const std::unordered_map<std::string, std::vector<double>> *plpar = nullptr, // Optional input: Path-loss parameters, keys without "PL_", [n_pl]
                         const std::vector<std::string> *pl_order = nullptr,                          // Optional input: Order of the plpar keys, [n_pl]; nullptr: sorted by name
                         const std::vector<std::string> *short_names = nullptr,                       // Optional input: Values of 'ShortName'
                         const std::string &ref = "",                                                 // Literature reference for the header, empty: none
                         const std::vector<std::string> *ref_sec = nullptr,                           // Optional input: Reference per section, up to n_sec = 10 entries, empty entry: none
                         const arma::mat *valid_range = nullptr,                                      // Optional input: Validity range of frequency [GHz], distance, BS height, MT height [m], [4, 2]
                         bool write_defaults = false);                                                // true: write all keys; false: only keys whose value differs from the default

    // ---- Large-scale fading ----

    // Generators of the large-scale fading (LSF) phase set, column index of 'phase' and 'rnd':
    //   0 DS, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA, 7 XPR      Normal(0,1), decorrelation distance of the LSP
    //   8 absolute TOA                                           Normal(0,1), absTOA_lambda
    //   9 ground permittivity                                    Uniform(0,1), SC_lambda
    //  10 UMa environment height hE                              Uniform(0,1), SF_lambda
    // - Slice 1 (Tx side) of 'phase' is slice 0 (Rx side) of the mirror generator: ASD <-> ASA, ESD <-> ESA, all
    //   others mirror themselves; a decorrelation distance of 0 is used as 0.1 m
    // - The functions are documented in src/qd_gbsm_phases.cpp (lsf_init_phases, lsf_draw), src/path_loss.cpp,
    //   src/lsp_generate.cpp, src/los_probability.cpp, src/o2i_loss.cpp

    // Draw the random phases of the large-scale generators of one BS site and scenario
    // - Generator order and reciprocity rules: see the LSF generator index above
    // - Bias removal over the positions in 'rx_pos' (and 'tx_pos' for dual mobility) when given and n_link > 2
    // - TR 38.901 clause 7.4.4 literally: preset "Exp256", remove_bias = false
    void lsf_init_phases(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (decorrelation distances)
                         arma::cube &phase,                                      // Output: Phases in [rad], [n_coeff, 11, 2]; slice 0 = Rx side, slice 1 = Tx side
                         arma::sword seed = -1,                                  // Seed, -1 = system random device
                         const arma::mat *tx_pos = nullptr,                      // Optional: Tx positions, [1, 3] or [n_link, 3]
                         const arma::mat *rx_pos = nullptr,                      // Optional: Rx positions for the bias removal (2D: one floor), [n_link, 3]
                         bool dual_mobility = false,                             // Tx and Rx both move: the Tx side is a second device
                         bool remove_bias = true,                                // Remove the sample mean over the positions
                         const std::string &preset = "Comb256",                  // SOS preset, see sos_preset
                         bool horizontal = true);                                // true: 2D correlation in the x-y plane (TR); false: 3D (legacy)

    // Spatially correlated normalized random values of the large-scale generators at a set of links
    // - Columns 0 to 8 are Normal(0,1), columns 9 and 10 Uniform(0,1)
    // - Dual mobility: the Tx side uses the mirror generator, so ASD/ASA and ESD/ESA are correlated by the ACF
    //   at the Tx-Rx distance
    // - Links with different floor IDs are uncorrelated at equal horizontal positions; equal IDs share one field
    // - Dual mobility: both devices of a link are on the link's floor
    // - Floor IDs only in 2D mode: with horizontal = false the height separates the floors
    void lsf_draw(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (decorrelation distances)
                  const arma::cube &phase,                                 // Phases from lsf_init_phases, [n_coeff, 11, 2]
                  const arma::mat &tx_pos,                                 // Tx positions, [1, 3] or [n_link, 3]
                  const arma::mat &rx_pos,                                 // Rx positions, [n_link, 3]
                  arma::mat &rnd,                                          // Output: Random values, [n_link, 11]
                  const arma::uvec *floor_id = nullptr,                    // Optional: Floor index per link, [n_link]; nullptr = all on one floor; 2D only
                  bool dual_mobility = false,                              // As in lsf_init_phases
                  const std::string &preset = "Comb256",                   // Must match lsf_init_phases
                  bool horizontal = true);                                 // Must match lsf_init_phases

    // Path loss and shadow-fading standard deviation of a set of links
    // - Models: logdist, logdist_simple, constant, winner_los, winner_nlos, winner_pathloss, dual_slope,
    //   tripple_slope, nlos, satellite; parameters in 'plpar' as in the conf files (keys without "PL_")
    // - An empty 'pl_model' gives zero loss
    void path_loss(const std::string &pl_model,                                          // Path-loss model, value of 'PL_model'
                   const std::unordered_map<std::string, std::vector<double>> &plpar,    // Path-loss parameters
                   const std::unordered_map<std::string, double> &scenpar,               // Scenario parameters (SF std, LSP_fc_min)
                   const arma::vec &fc,                                                  // Carrier frequencies in [Hz], [n_freq]
                   const arma::mat &tx_pos,                                              // Tx positions, [1, 3] or [n_link, 3]
                   const arma::mat &rx_pos,                                              // Rx positions, [n_link, 3]
                   arma::mat &pl,                                                        // Output: Path loss in [dB], [n_link, n_freq]
                   arma::mat *sf_sigma = nullptr,                                        // Optional output: SF standard deviation in [dB], [n_link, n_freq]
                   const arma::vec *d2d_in = nullptr,                                    // Optional: Indoor 2D distance in [m], [n_link]; given = outdoor distance only (legacy); 0 = outdoor link, geometric distance
                   const arma::vec *rnd_he = nullptr);                                   // Optional: Uniform(0,1) per link for the random hE, column 10 of lsf_draw, [n_link]

    // Large-scale parameters of a set of links from normalized random values
    // - Mean and std per link and frequency from the scenario parameters (incl. LSP_fc_min, ES_D_mu_B/C/D);
    //   cross-correlation by the root of the LSP cross-correlation matrix; transformation to linear values
    // - Same random values for all frequencies (TR 7.6.5)
    void lsp_generate(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters
                      const arma::vec &fc,                                    // Carrier frequencies in [Hz], [n_freq]
                      const arma::mat &tx_pos,                                // Tx positions, [1, 3] or [n_link, 3]
                      const arma::mat &rx_pos,                                // Rx positions, [n_link, 3]
                      const arma::mat &rnd,                                   // Random values from lsf_draw, [n_link, 11]
                      arma::cube &lsp,                                        // Output: [n_link, n_freq, 8]: DS [s], KF, SF [linear], ASD, ASA, ESD, ESA [deg], XPR [linear]
                      const arma::mat *sf_sigma = nullptr,                    // Optional: SF std in [dB] from path_loss, replaces the scenpar value, [n_link, n_freq]
                      arma::vec *abs_toa = nullptr,                           // Optional output: Absolute TOA offset in [s], 0 if disabled, [n_link]
                      arma::mat *gr_eps_re = nullptr,                         // Optional output: Ground permittivity, real part, [n_link, n_freq]
                      arma::mat *gr_eps_im = nullptr,                         // Optional output: Ground permittivity, imaginary part, [n_link, n_freq]
                      arma::mat *eod_offset_deg = nullptr,                    // Optional output: EoD offset in [deg], positive = up, [n_link, n_freq]
                      arma::cube *lsp_mu = nullptr,                           // Optional output: Mean in the log domain (lg or dB), [n_link, n_freq, 8]
                      arma::cube *lsp_sigma = nullptr,                        // Optional output: Std in the log domain, [n_link, n_freq, 8]
                      bool cholesky = true,                                   // true: Cholesky in TR order; false: symmetric square root (legacy)
                      bool clip_spreads = true);                              // Clip AS to 104 deg, ES to 52 deg (TR step 4)

    // LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models
    // - Models: "38.901_RMa", "38.901_UMi", "38.901_UMa", "38.901_InH_Mixed", "38.901_InH_Open",
    //   "38.901_InF_SL", "38.901_InF_DL", "38.901_InF_SH", "38.901_InF_DH", "38.901_InF_HH", "38.901_SMa",
    //   "38.811_DenseUrban", "38.811_Urban", "38.811_Suburban", "38.811_Rural", "QuaDRiGa_Industrial",
    //   "LOS" (always 1), "NLOS" (always 0)
    // - Soft LOS (7.6-18) when 'rnd_g', 'fc' and 'los_soft' are given
    void los_probability(const std::string &model,          // Model name
                         const arma::vec &d2d_out,          // 2D outdoor distance in [m] (indoor scenarios: 2D distance), [n_link]
                         const arma::vec &h_bs,             // BS height in [m], [n_link] or [1]
                         const arma::vec &h_ut,             // UT height in [m], [n_link] or [1]
                         arma::vec &p_los,                  // Output: LOS probability, [n_link]
                         double d_clutter = NAN,            // InF, SMa: clutter size in [m]; NaN = TR default of the model
                         double r_clutter = NAN,            // InF: clutter density; SMa: vegetation density r_vegetation; NaN = default
                         double h_clutter = NAN,            // InF SH, DH: clutter height in [m]; NaN = default
                         const arma::vec *rnd_g = nullptr,  // Optional: Normal(0,1) per link for the soft LOS state, [n_link]
                         const arma::vec *fc = nullptr,     // Optional: Carrier frequencies in [Hz] for the soft LOS state, [n_freq]
                         arma::mat *los_soft = nullptr);    // Optional output: Soft LOS state in [0, 1], [n_link, n_freq]

    // O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3
    // - Models: "low", "high", "low_A" (Table 7.4.3-2), "legacy_6GHz" (Table 7.4.3-3), "car", "car_metal" (7.4.3.2)
    // - The model of each UT is chosen by comparing column 0 of 'rnd' with the cumulative 'weights'
    void o2i_loss(const arma::vec &fc,                       // Carrier frequencies in [Hz], [n_freq]
                  const arma::mat &rnd,                      // Random values per UT, [n_ut, 4]: U model, U d2D-in 1, U d2D-in 2, N deviation
                  const std::vector<std::string> &models,    // Models to mix, [n_model]
                  const arma::vec &weights,                  // Probability of each model, sum 1, [n_model]
                  arma::mat &loss,                           // Output: Penetration loss in [dB], [n_ut, n_freq]
                  double d2d_in_max = 25.0,                  // Maximum indoor 2D distance in [m] (key O2I_d2D_in_max)
                  arma::vec *d2d_in = nullptr,               // Optional output: Indoor 2D distance in [m], 0 for car models, [n_ut]
                  arma::uvec *model_index = nullptr);        // Optional output: Chosen model per UT, 0-based index into 'models', [n_ut]
}

#endif
