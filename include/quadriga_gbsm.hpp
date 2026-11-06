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
                         const std::string &pl_model = "",                                            // Value of 'PL_model', empty: not written
                         const std::unordered_map<std::string, std::vector<double>> *plpar = nullptr, // Optional input: Path-loss parameters, keys without "PL_", [n_pl]
                         const std::vector<std::string> *pl_order = nullptr,                          // Optional input: Order of the plpar keys, [n_pl]; nullptr: sorted by name
                         const std::vector<std::string> *short_names = nullptr,                       // Optional input: Values of 'ShortName'
                         const std::string &ref = "",                                                 // Literature reference for the header, empty: none
                         const std::vector<std::string> *ref_sec = nullptr,                           // Optional input: Reference per section, up to n_sec = 10 entries, empty entry: none
                         const arma::mat *valid_range = nullptr,                                      // Optional input: Validity range of frequency [GHz], distance, BS height, UT height [m], [4, 2]
                         bool write_defaults = false);                                                // true: write all keys; false: only keys whose value differs from the default

    // ---- Random sets ----

    // Draw the SOS phases of one random set of a BS site and scenario: the random phases of the sinusoids of its generators
    // - Set types: "lsf" large-scale generators, "cluster" cluster random values, "ray" ray random values; generators
    //   and columns: see gbsm_draw
    // - The SOS phases are a fixed function of seed, set_type and set_id, equal on all platforms; the sets are independent
    // - "lsf": bias removal over the positions in 'rx_pos' (and 'tx_pos' for dual mobility) when given and n_link > 2
    void gbsm_init_phases(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (decorrelation distances; NumClusters; NumSubPaths for "ray")
                          const std::string &set_type,                            // "lsf", "cluster" or "ray"
                          arma::cube &phase,                                      // Output: SOS phases in [rad], [n_coeff, n_gen, 2]; slice 0 = Rx side, slice 1 = Tx side
                          arma::sword seed = -1,                                  // Seed of the site and scenario, the same for all set types; -1 = system random device
                          arma::sword set_id = -1,                                // "cluster", "ray": -1 = shared set, k >= 0 = set of frequency k (0-based); ignored for "lsf"
                          const std::string &preset = "Comb256",                  // SOS preset, see sos_preset
                          const arma::mat *tx_pos = nullptr,                      // Optional: Tx positions for the bias removal, [1, 3] or [n_link, 3]; "lsf" only
                          const arma::mat *rx_pos = nullptr,                      // Optional: Rx positions for the bias removal (2D: one floor), [n_link, 3]; "lsf" only
                          bool dual_mobility = false,                             // "lsf": Tx and Rx both move, the Tx side is a second device
                          bool remove_bias = true,                                // "lsf": remove the sample mean over the positions
                          bool horizontal = true);                                // "lsf": 2D bias removal in the x-y plane (TR); false: 3D (legacy)

    // Spatially correlated normalized random values of one random set at a set of links
    // - Columns, distributions and decorrelation distances of each set type: see the set table in the gbsm_draw documentation
    // - Dual mobility: the Tx side uses the mirror generator; swapping 'tx_pos' and 'rx_pos' exchanges the mirror pairs
    // - Links with different floor IDs are uncorrelated at equal horizontal positions (2D mode only)
    void gbsm_draw(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (decorrelation distances, NumClusters, NumSubPaths, PerCluster* for "ray")
                   const std::string &set_type,                             // "lsf", "cluster" or "ray"; must match gbsm_init_phases
                   const arma::cube &phase,                                 // SOS phases from gbsm_init_phases, [n_coeff, n_gen, 2]
                   const arma::mat &tx_pos,                                 // Tx positions in [m], [1, 3] or [n_link, 3]
                   const arma::mat &rx_pos,                                 // Rx positions in [m], [n_link, 3]
                   arma::mat &rnd,                                          // Output: Random values, [n_link, n_gen]
                   arma::sword set_id = -1,                                 // As in gbsm_init_phases; ignored for "lsf"
                   const std::string &preset = "Comb256",                   // Must match gbsm_init_phases
                   const arma::uvec *floor_id = nullptr,                    // Optional: Floor index per link, [n_link]; nullptr = all on one floor; 2D only
                   const arma::vec *center_frequency = nullptr,             // Optional: Carrier frequencies in [Hz], [n_freq]; required by the shared "ray" set, same list as for ssf_rays
                   bool dual_mobility = false,                              // Tx and Rx both move: the Tx side is a second device
                   bool horizontal = true);                                 // true: 2D correlation in the x-y plane (TR); false: 3D (legacy)

    // ---- Large-scale fading ----
    // Large-scale generators ("lsf" set of gbsm_init_phases and gbsm_draw): see the set table in the gbsm_draw documentation

    // Path loss and shadow-fading standard deviation of a set of links
    // - Models: logdist, logdist_simple, constant, winner_los, winner_nlos, winner_pathloss, dual_slope,
    //   tripple_slope, nlos, satellite; parameters in 'plpar' as in the conf files (keys without "PL_")
    // - An empty 'pl_model' gives zero loss
    void path_loss(const std::string &pl_model,                                          // Path-loss model, value of 'PL_model'
                   const std::unordered_map<std::string, std::vector<double>> &plpar,    // Path-loss parameters
                   const std::unordered_map<std::string, double> &scenpar,               // Scenario parameters (SF std, LSP_fc_min)
                   const arma::vec &center_frequency,                                    // Carrier frequencies in [Hz], [n_freq]
                   const arma::mat &tx_pos,                                              // Tx positions, [1, 3] or [n_link, 3]
                   const arma::mat &rx_pos,                                              // Rx positions, [n_link, 3]
                   arma::mat &pl,                                                        // Output: Path loss in [dB], [n_link, n_freq]
                   arma::mat *sf_sigma = nullptr,                                        // Optional output: SF standard deviation in [dB], [n_link, n_freq]
                   const arma::vec *d2d_in = nullptr,                                    // Optional: Indoor 2D distance in [m], [n_link]; given = outdoor distance only (legacy); 0 = outdoor link, geometric distance
                   const arma::vec *rnd_he = nullptr);                                   // Optional: Uniform(0,1) per link for the random hE, column 10 of gbsm_draw ("lsf"), [n_link]

    // Large-scale parameters of a set of links from normalized random values
    // - Mean and std per link and frequency from the scenario parameters (incl. LSP_fc_min, ES_D_mu_B/C/D);
    //   cross-correlation by the root of the LSP cross-correlation matrix; transformation to linear values
    // - Same random values for all frequencies (TR 7.6.5)
    void lsp_generate(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters
                      const arma::vec &center_frequency,                      // Carrier frequencies in [Hz], [n_freq]
                      const arma::mat &tx_pos,                                // Tx positions, [1, 3] or [n_link, 3]
                      const arma::mat &rx_pos,                                // Rx positions, [n_link, 3]
                      const arma::mat &rnd,                                   // Random values from gbsm_draw ("lsf"), [n_link, 11]
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
    // - Soft LOS (7.6-18) when 'rnd_g', 'center_frequency' and 'los_soft' are given
    void los_probability(const std::string &model,          // Model name
                         const arma::vec &d2d_out,          // 2D outdoor distance in [m] (indoor scenarios: 2D distance), [n_link]
                         const arma::vec &h_bs,             // BS height in [m], [n_link] or [1]
                         const arma::vec &h_ut,             // UT height in [m], [n_link] or [1]
                         arma::vec &p_los,                  // Output: LOS probability, [n_link]
                         double d_clutter = NAN,            // InF, SMa: clutter size in [m]; NaN = TR default of the model
                         double r_clutter = NAN,            // InF: clutter density; SMa: vegetation density r_vegetation; NaN = default
                         double h_clutter = NAN,            // InF SH, DH: clutter height in [m]; NaN = default
                         const arma::vec *rnd_g = nullptr,  // Optional: Normal(0,1) per link for the soft LOS state, [n_link]
                         const arma::vec *center_frequency = nullptr,   // Optional: Carrier frequencies in [Hz] for the soft LOS state, [n_freq]
                         arma::mat *los_soft = nullptr);    // Optional output: Soft LOS state in [0, 1], [n_link, n_freq]

    // O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3
    // - Models: "low", "high", "low_A" (Table 7.4.3-2), "legacy_6GHz" (Table 7.4.3-3), "car", "car_metal" (7.4.3.2)
    // - The model of each UT is chosen by comparing column 0 of 'rnd' with the cumulative 'weights'
    void o2i_loss(const arma::vec &center_frequency,         // Carrier frequencies in [Hz], [n_freq]
                  const arma::mat &rnd,                      // Random values per UT, [n_ut, 4]: U model, U d2D-in 1, U d2D-in 2, N deviation
                  const std::vector<std::string> &models,    // Models to mix, [n_model]
                  const arma::vec &weights,                  // Probability of each model, sum 1, [n_model]
                  arma::mat &loss,                           // Output: Penetration loss in [dB], [n_ut, n_freq]
                  double d2d_in_max = 25.0,                  // Maximum indoor 2D distance in [m] (key O2I_d2D_in_max)
                  arma::vec *d2d_in = nullptr,               // Optional output: Indoor 2D distance in [m], 0 for car models, [n_ut]
                  arma::uvec *model_index = nullptr);        // Optional output: Chosen model per UT, 0-based index into 'models', [n_ut]

    // ---- Small-scale fading ----
    // Cluster level (ssf_clusters) and ray level (gbsm_rays, ssf_rays, ssf_paths) of TR 38.901 clause 7.5;
    // N = NumClusters - 1 NLOS clusters, M = NumSubPaths rays per cluster. The column layout of the random values:
    // see the set table in the gbsm_draw documentation

    // Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7)
    // - Methods "38.901", "38.901_MF" (clause 7.6.5.1), "quadriga"
    // - Slots: 0 = LOS path, 1 = ground reflection if GR_enabled, then the N NLOS clusters
    void ssf_clusters(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters
                      const arma::mat &tx_pos,                                // Tx positions in [m], [1, 3] or [n_link, 3]
                      const arma::mat &rx_pos,                                // Rx positions in [m], [n_link, 3]
                      const arma::cube &lsp,                                  // LSPs from lsp_generate, [n_link, n_freq, 8]
                      const arma::mat &rnd,                                   // Shared cluster set from gbsm_draw ("cluster", set_id = -1), [n_link, 9·N]
                      arma::cube &delay,                                      // Output: Delays in [s], [n_link, n_slot, n_freq] for 38.901_MF, else [n_link, n_slot, 1]
                      arma::cube &power,                                      // Output: Powers (linear), [n_link, n_slot, n_freq]
                      arma::cube &aod,                                        // Output: Azimuth of departure in [rad], [n_link, n_slot, n_freq] for 38.901_MF, else [n_link, n_slot, 1]
                      arma::cube &eod,                                        // Output: Elevation of departure in [rad], [n_link, n_slot, n_freq] for 38.901_MF, else [n_link, n_slot, 1]
                      arma::cube &aoa,                                        // Output: Azimuth of arrival in [rad], [n_link, n_slot, n_freq] for 38.901_MF, else [n_link, n_slot, 1]
                      arma::cube &eoa,                                        // Output: Elevation of arrival in [rad], [n_link, n_slot, n_freq] for 38.901_MF, else [n_link, n_slot, 1]
                      const std::string &method = "38.901",                   // "38.901", "38.901_MF" or "quadriga"
                      const arma::cube *rnd_freq = nullptr,                   // Optional: Per-frequency cluster sets from gbsm_draw ("cluster", set_id = k), [n_link, N, n_freq]; required by 38.901, 38.901_MF
                      const arma::cube *lsp_anchor = nullptr,                 // Optional: LSPs at the anchor frequency, [n_link, 1, 8]; required by 38.901_MF
                      const arma::mat *eod_offset_deg = nullptr,              // Optional: EoD offset in [deg] from lsp_generate, positive = up, [n_link, n_freq]; nullptr = 0
                      const arma::vec *abs_toa = nullptr,                     // Optional: Absolute TOA offset in [s] from lsp_generate, [n_link]; required by quadriga in NLOS if absTOA_mu > -30
                      const arma::mat *gr_eps_re = nullptr,                   // Optional: Ground permittivity, real part, [n_link, n_freq]; required if GR_enabled
                      const arma::mat *gr_eps_im = nullptr,                   // Optional: Ground permittivity, imaginary part, [n_link, n_freq]; required if GR_enabled
                      const std::vector<bool> *o2i = nullptr,                 // Optional: O2I flag per link (mean ZOA 90 deg in 38.901), [n_link]; nullptr = none
                      const arma::uvec *n_cluster = nullptr,                  // Optional: Number of NLOS clusters per link (TR clause 7.6.15), [n_link]; nullptr = N
                      arma::uvec *n_active = nullptr);                        // Optional output: Number of NLOS clusters with power > 0, [n_link]

    // Ray data of one link (geometry-based stochastic model)
    // - Output of ssf_rays and gbsm_rays_from_paths; solve_scatterers adds the scatterer positions
    // - Input of ssf_paths and gbsm_single_link_channel, which call is_valid() first
    // - Set the attributes by name: later versions append attributes
    class gbsm_rays
    {
    public:
        std::string method = "38.901"; // Method of the ray data: "38.901", "38.901_MF" or "quadriga"; sets the GR power law and the default mobility
        arma::vec tx_pos;               // Tx position the ray data refer to in [m], [3]
        arma::vec rx_pos;               // Rx position the ray data refer to in [m], [3]
        arma::vec center_frequency;     // Carrier frequencies in [Hz], [n_freq]
        arma::vec delay;                // Delays in [s] relative to the LOS delay |rx_pos - tx_pos| / c, [n_ray]
        arma::mat power;                // Coefficient powers (linear), without link gain; data from paths: path gain, [n_ray, n_freq]
        arma::vec aod;                  // Azimuth of departure in [rad], GCS, [n_ray]
        arma::vec eod;                  // Elevation of departure in [rad], GCS, [n_ray]
        arma::vec aoa;                  // Azimuth of arrival in [rad], GCS, [n_ray]
        arma::vec eoa;                  // Elevation of arrival in [rad], GCS, [n_ray]
        arma::cube M;                   // Polarization transfer matrices incl. initial phases, row order of get_channels_*, [8, n_ray, n_freq]
        arma::uvec path_type;           // Path type, 0 = LOS, 1 = GR, 2 = NLOS, [n_ray]
        arma::uvec tap;                 // Tap index, rays with equal index form one tap, [n_ray]
        arma::vec gr_eps_re;            // Ground permittivity, real part, [n_freq] or empty; needed where a GR ray is recomputed
        arma::vec gr_eps_im;            // Ground permittivity, imaginary part, [n_freq] or empty; needed where a GR ray is recomputed
        arma::mat fbs_pos;              // First-bounce scatterer positions in [m], [3, n_ray]; empty until solve_scatterers
        arma::mat lbs_pos;              // Last-bounce scatterer positions in [m], [3, n_ray]; empty until solve_scatterers
        arma::uvec fallback;            // 1 = lengthened by the solver fallback, [n_ray]; empty until solve_scatterers
        gbsm_rays() {};                 // Default constructor

        arma::uword n_ray() const;                     // Number of rays, length of 'delay'
        arma::uword n_freq() const;                    // Number of frequencies, length of 'center_frequency'
        bool empty() const;                            // true if the object holds no rays
        std::string is_valid() const;                  // Empty string if the object is valid, else an error message
        arma::vec b_med(arma::uword i_freq = 0) const; // b_med of the Tx (to FBS) and the Rx (to LBS) in [m], [2]; NaN without scatterers or NLOS power
        arma::vec b_eff(arma::uword i_freq = 0) const; // b_eff of the Tx and the Rx in [m], [2]; NaN without scatterers or power
        void solve_scatterers(double d_min = 2.0);     // Two-bounce solver: writes fbs_pos, lbs_pos and fallback; d_min in [m], > 0
    };

    // Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16)
    // - One geometry per call: "38.901" and "38.901_MF" compute frequency 'i_freq' (with slice i_freq of 'rnd_ray');
    //   "quadriga" all frequencies; all other inputs as ssf_clusters and lsp_generate return them
    // - A pure function of the clusters, the LSPs and the random values: no seed, batch-independent
    void ssf_rays(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (NumClusters, NumSubPaths, PerCluster*, GR_enabled, PolVar_sigma)
                  const arma::mat &tx_pos,                                // Tx positions in [m], [1, 3] or [n_link, 3]
                  const arma::mat &rx_pos,                                // Rx positions in [m], [n_link, 3]
                  const arma::vec &center_frequency,                      // Carrier frequencies in [Hz], [n_freq]
                  const arma::cube &lsp,                                  // LSPs from lsp_generate, [n_link, n_freq, 8]
                  const arma::cube &lsp_mu,                               // Log-domain means from lsp_generate, [n_link, n_freq, 8]
                  const arma::cube &lsp_sigma,                            // Log-domain std from lsp_generate, [n_link, n_freq, 8]
                  const arma::cube &delay,                                // Cluster delays from ssf_clusters, [n_link, n_slot, n_freq] ("38.901_MF") or [n_link, n_slot, 1]
                  const arma::cube &power,                                // Cluster powers from ssf_clusters, [n_link, n_slot, n_freq]
                  const arma::cube &aod,                                  // Cluster AOD from ssf_clusters, size as 'delay'
                  const arma::cube &eod,                                  // Cluster EOD from ssf_clusters, size as 'delay'
                  const arma::cube &aoa,                                  // Cluster AOA from ssf_clusters, size as 'delay'
                  const arma::cube &eoa,                                  // Cluster EOA from ssf_clusters, size as 'delay'
                  const arma::cube &rnd_ray,                              // "ray" sets from gbsm_draw, [n_link, n_gen, n_freq] (TR methods: slice k = set k) or [n_link, n_gen, 1] (quadriga: set -1)
                  std::vector<gbsm_rays> &rays,                           // Output: Ray data, n_link entries; fbs_pos, lbs_pos, fallback empty
                  const std::string &method = "38.901",                   // Method of ssf_clusters: "38.901", "38.901_MF" or "quadriga"
                  arma::uword i_freq = 0,                                 // TR methods: index into center_frequency and slice of rnd_ray, 0-based; quadriga: must be 0
                  const std::string &angle_model = "additive",            // "additive" (TR, zenith fold) or "rotation" (3-D rotation of the offsets)
                  const std::string &pol_model = "38.901",                // "38.901" (kappa, four phases) or "quadriga" (rotation model)
                  const arma::mat *gr_eps_re = nullptr,                   // Optional: Ground permittivity, real part, from lsp_generate, [n_link, n_freq]; required if GR_enabled
                  const arma::mat *gr_eps_im = nullptr,                   // Optional: Ground permittivity, imaginary part, from lsp_generate, [n_link, n_freq]; required if GR_enabled
                  bool pol_variability = false,                           // TR 7.6.16 polarisation power variability; pol_model "38.901" only
                  bool random_phase = true);                              // false: all initial phases 0

    // Antenna-free paths of one link at one position
    // - "phase": use the outputs with get_channels_planar and the arrays at the positions of 'rays' (the displacement
    //   phase is in M); "drift": inputs of get_channels_spherical / get_channels_multifreq and qrt_paths_from_scatterers
    // - Rays with zero power at all frequencies are dropped, LOS and GR rays are kept
    // - Validity: "phase" up to about sqrt(lambda * b_eff / 8) per moving end (see gbsm_rays::b_eff),
    //   "drift" up to about 0.2 * b_med per moving end
    void ssf_paths(const gbsm_rays &rays,                // Ray data of the link
                   const arma::vec &link_gain_dB,        // Link gain -PL - O2I + 10·lg(SF) at the position in [dB], [n_freq]; 0 for data from paths
                   arma::mat &path_gain,                 // Output: Path gains (linear), [n_path, n_freq]
                   arma::vec &path_length,               // Output: Path lengths in [m], [n_path]
                   arma::cube &M,                        // Output: Polarization transfer matrices, [8, n_path, n_freq]
                   const std::string &mobility = "auto", // "phase", "drift" or "auto" (phase for the TR methods, drift for quadriga)
                   const arma::vec *tx_pos = nullptr,    // Optional: Tx position in [m], [3]; nullptr = rays.tx_pos
                   const arma::vec *rx_pos = nullptr,    // Optional: Rx position in [m], [3]; nullptr = rays.rx_pos
                   const arma::vec *kf_ratio = nullptr,  // Optional: K(position) / K(drop), [n_freq]; nullptr = 1
                   arma::vec *aod = nullptr,             // Optional output: Azimuth of departure in [rad], [n_path]
                   arma::vec *eod = nullptr,             // Optional output: Elevation of departure in [rad], [n_path]
                   arma::vec *aoa = nullptr,             // Optional output: Azimuth of arrival in [rad], [n_path]
                   arma::vec *eoa = nullptr,             // Optional output: Elevation of arrival in [rad], [n_path]
                   arma::mat *fbs_pos = nullptr,         // Optional output: First-bounce scatterer positions in [m], [3, n_path]; "phase": [3, 0]
                   arma::mat *lbs_pos = nullptr,         // Optional output: Last-bounce scatterer positions in [m], [3, n_path]; "phase": [3, 0]
                   arma::uvec *path_type = nullptr,      // Optional output: Path type, 0 = LOS, 1 = GR, 2 = NLOS, [n_path]
                   arma::uvec *ray_ind = nullptr,        // Optional output: Index of the ray in 'rays', 0-based, [n_path]
                   bool oxygen = false);                 // TR 7.6.1 oxygen absorption from each path's own length

    // Ray data of one link from spherical path data (e.g. read with qrt_file_read)
    // - Delays from the path lengths, angles from the scatterer positions, power = path gain
    // - Fills a pre-initialised object in place
    void gbsm_rays_from_paths(const arma::vec &center_frequency,      // Frequencies in [Hz], [n_freq]
                              const arma::vec &tx_pos,                // Tx position in [m], [3]
                              const arma::vec &rx_pos,                // Rx position in [m], [3]
                              const arma::mat &fbs_pos,               // First-bounce scatterer positions in [m], [3, n_path]
                              const arma::mat &lbs_pos,               // Last-bounce scatterer positions in [m], [3, n_path]
                              const arma::mat &path_gain,             // Path gains (linear), [n_path, n_freq]
                              const arma::vec &path_length,           // Path lengths in [m], [n_path]
                              const arma::cube &M,                    // Polarization transfer matrices, [8, n_path, n_freq]
                              gbsm_rays &rays,                        // Output: Ray data, n_ray = n_path
                              const std::string &method = "quadriga", // Value of rays.method: GR power law and default mobility
                              const arma::uvec *path_type = nullptr,  // Optional: Path type, 0 = LOS, 1 = GR, 2 = NLOS, [n_path]; nullptr = LOS where FBS = LBS on the direct line, else NLOS
                              const arma::uvec *tap = nullptr,        // Optional: Tap index, [n_path]; nullptr = one tap per path
                              const arma::vec *gr_eps_re = nullptr,   // Optional: Ground permittivity, real part, [n_freq]; needed where a GR path is recomputed
                              const arma::vec *gr_eps_im = nullptr);  // Optional: Ground permittivity, imaginary part, [n_freq]; needed where a GR path is recomputed
}

#endif
