// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef quadriga_layout_H
#define quadriga_layout_H

#include <armadillo>
#include <string>
#include <unordered_map>
#include <vector>

#include "quadriga_gbsm.hpp"

namespace quadriga_lib
{

    // ---- Network layouts and mobility ----

    // One side of a layout (Tx or Rx); flat arrays, nodes in order of first appearance
    // - A node is one (name, sector) pair with one antenna and one track; its rows are consecutive, starting at offset
    // - Optional members may be empty, which has the meaning given in the comment
    class layout_nodes
    {
      public:
        std::vector<std::string> name;                    // Site or terminal name, [n_node]; no comma, semicolon, quote, line break, no blank at the ends
        arma::u32_vec sector;                             // Sector index, [n_node]
        arma::u32_vec i_antenna;                          // 1-based index into the antenna lists of the layout, 0 = generic xpol, [n_node]
        arma::uvec offset;                                // First row of each node, 0-based, [n_node]
        arma::mat pos;                                    // Positions in [m], [3, n_row]
        arma::mat orientation;                            // Bank, tilt, heading in [rad], [3, n_row]
        arma::mat direction;                              // Azimuth, elevation of travel in [rad], [2, n_row]; NaN azimuth = follow the orientation; empty = all NaN
        arma::vec speed;                                  // Speed in [m/s], applies from this row on, [n_row]; empty = 0
        arma::vec time;                                   // Time in [s], [n_row]; NaN on all rows of a node = not given; empty = none
        std::vector<std::string> scenario;                // Scenario family or conf name, "" = none, [n_row]; empty = none
        arma::s32_vec indoor;                             // -1 = draw, 0 = outdoor, 1 = building (commercial), 2 = car, 3 = residential building, [n_row]; empty = all -1
        arma::s32_vec floor;                              // Floor number from 1, -1 = derive from z, [n_row]; empty = all -1
        arma::vec d2d_in;                                 // Indoor 2D distance in [m], NaN = draw, [n_row]; empty = all NaN
        arma::mat clutter;                                // d_clutter in [m], r_clutter, h_clutter in [m], NaN = model default, [3, n_node]; empty = all NaN
        std::unordered_map<std::string, arma::vec> scenpar_override; // Scenario parameter overrides by key, NaN = none, each [n_node]
        arma::s64_vec seed;                               // Site seed, -1 = derived from the layout seed, [n_node]; empty = all -1
        arma::u32_vec relative;                           // 1 = orientation relative to the direction of motion, [n_node]; empty = all 0
        std::vector<std::string> extra_names;             // Names of unknown columns that were kept, [n_extra]
        std::vector<std::vector<std::string>> extra;      // Their cells as text, [n_extra][n_row]

        arma::uword n_node() const;                       // Number of nodes
        arma::uword n_row() const;                        // Number of rows of all nodes
        arma::uword n_row(arma::uword i_node) const;      // Number of rows of one node
        std::string is_valid() const;                     // Empty string if valid, else an error message
    };

    // State of the propagation links of one realisation; empty until it is set
    class layout_links
    {
      public:
        arma::u32_vec i_tx, i_rx, i_segment;              // Site, terminal and segment of each propagation link, 0-based, [n_link]
        arma::u32_vec image;                              // Wrap image, 0 = real site, [n_link]
        arma::u32_vec i_conf;                             // Index into conf_names, derived from scenario, LOS and indoor state, [n_link]
        std::vector<std::string> conf_names;              // Resolved conf names, [n_conf]
        arma::u32_vec los;                                // 1 = LOS, [n_link]
        arma::s32_vec indoor;                             // O2I state at the start of the segment, [n_link]
        arma::u32_vec floor_id;                           // Floor number - 1, 0 outdoors, [n_link]
        arma::vec d2d_in;                                 // Indoor 2D distance in [m], [n_link]
        arma::mat o2i_rnd;                                // Random inputs of o2i_loss per terminal, [n_terminal, 4]
        arma::u32_vec o2i_model;                          // Chosen O2I model, index into the list of the family, [n_terminal]
        arma::s64_vec gbsm_seed;                          // Seed of the random phases; equal for one site, image and conf, [n_link]
        arma::sword seed = -1;                            // Master seed
        std::string method, sos_preset;                   // Method and SOS preset as used
        double anchor_freq = 2.0e9;                       // Anchor frequency of the multi-frequency model in [Hz]
        bool pol_variability = false;                     // Polarization power variability
        std::vector<std::string> conf_files;              // Resolved paths of the conf files, [n_conf]
        unsigned long long layout_hash = 0;               // Hash of all layout members that determine the links

        arma::uword n_link() const;                       // Number of propagation links
        bool empty() const;                               // True if there are no links
        void reset();                                     // Empties all members and restores the defaults
        std::string is_valid() const;                     // Empty string if valid, else an error message
    };

    // A network layout: Tx and Rx nodes with tracks, scenarios and antennas
    class layout
    {
      public:
        std::string name;                                 // Name of the layout
        layout_nodes tx, rx;                              // BS/AP side and UT/STA side
        layout_links links;                               // Link states of the current realisation
        arma::sword seed = -1;                            // Layout seed, -1 = not set
        arma::vec center_freq;                            // Simulation frequencies in [Hz], strictly ascending, [n_freq]
        std::vector<std::vector<arrayant<double>>> antenna;      // Evaluated antennas, [n_antenna][0, 1 or n_freq]; empty = not evaluated
        std::vector<std::string> antenna_cols;                   // Names of the antenna columns, [n_col]
        std::vector<std::vector<std::string>> antenna_args;      // CSV lines of the antenna columns, [n_antenna][0, 1 or n_freq]; "" = evaluated array
        arma::mat wrap;                                   // Lattice vectors in [m], one per column, [2, 1] or [2, 2]; empty = none
        arma::umat pairing;                               // Tx site index and Rx terminal index of each linked pair, 0-based, [2, n_pair]; empty = all pairs

        arma::uword n_antenna() const;                    // Length of antenna_args, or of antenna if there are no arguments
        void eval_antennas(const arma::vec &center_freq = {}); // Generates 'antenna' from 'antenna_args'; a given list in [Hz] replaces center_freq
        std::string is_valid() const;                     // Empty string if valid, else an error message
    };

    // Read the Tx and Rx CSV files of a layout
    // - Either file name may be "" for no nodes on that side
    // - Antenna columns are composed into antenna_args, antenna_cols and i_antenna; equal descriptions share one entry
    // - With eval_antennas, the arrays are generated as by layout.eval_antennas()
    layout layout_csv_read(
        const std::string &fn_tx,                         // Tx CSV file, "" = no Tx nodes
        const std::string &fn_rx,                         // Rx CSV file, "" = no Rx nodes
        const arma::vec &center_freq = {},                // Simulation frequencies in [Hz]; empty = from the files
        bool eval_antennas = true,                        // true: generate the antennas; false: antenna stays empty
        std::vector<std::string> *extra_tx = nullptr,     // Optional output: names of unknown Tx columns
        std::vector<std::string> *extra_rx = nullptr);    // Optional output: names of unknown Rx columns

    // Write a layout to the Tx and Rx CSV files
    // - Either file name may be "" for not written; all files are composed and written to temporary files first, and
    //   only then are the existing files replaced
    // - Antenna bands that are evaluated arrays without text go to QDANT files <stem>_<k>.qdant next to the CSV file;
    //   such files that the new CSV files do not reference are deleted
    void layout_csv_write(
        const std::string &fn_tx,                         // Tx CSV file, "" = not written
        const std::string &fn_rx,                         // Rx CSV file, "" = not written
        const layout &layout,                             // Layout
        bool compact = true);                             // true: empty cells where the format allows; false: every cell written

    // Set the antenna of nodes from the arguments of generate_antenna; the layout stores them as an antenna line
    // - Argument names as in generate_antenna; a name that the type does not use throws
    // - With a center_freq, only that band is set, in a new antenna entry if others share the old one; a frequency
    //   that is not in layout.center_freq is added
    void set_antenna(
        layout &layout,                                                // Layout, updated in place
        const std::string &type,                                       // Antenna type of generate_antenna, or "none"
        const std::unordered_map<std::string, arma::mat> &args = {},   // Arguments by name, e.g. {"panel", {4, 4, 1, 2}}; flags 0 or 1
        bool rx_side = false,                                          // false: Tx nodes, true: Rx nodes
        const uint_list &i_node = {},                                  // Node indices, 0-based; empty = all nodes of the side
        double center_freq = NAN,                                      // Band in [Hz]; NaN = one antenna for all bands
        bool evaluate = true);                                         // true: generate all antennas that are not evaluated yet

    // Set the antenna of nodes from an evaluated array, e.g. a measurement read with qdant_read
    void set_antenna(
        layout &layout,                                                // Layout, updated in place
        const arrayant<double> &array,                                 // Single-frequency array
        bool rx_side = false,                                          // false: Tx nodes, true: Rx nodes
        const uint_list &i_node = {},                                  // Node indices, 0-based; empty = all nodes of the side
        double center_freq = NAN);                                     // Band in [Hz]; NaN = the same array for all bands

    // ---- Tracks: sampling, generation and orientation ----

    // Positions and orientations of one node at the snapshot times
    // - Without times, each row of the node is one snapshot; with times, a point drop moves along its direction with its speed, and a
    //   track is interpolated between its rows (linear position, shortest-way interpolation of each angle, ends held)
    // - Row times come from 'time', else from the arc length and 'speed'; a track with neither throws if times are given
    // - Orientations are in the GCS; for a node with relative = 1 they include the rotation by the direction of motion
    void sample_track(
        const layout_nodes &nodes,           // Nodes
        arma::uword i_node,                  // Node index, 0-based
        const arma::vec &times,              // Snapshot times in [s], [n_snap]; empty = one snapshot per row
        arma::mat &pos,                      // Output: positions in [m], [3, n_snap]
        arma::mat &orientation,              // Output: orientations (bank, tilt, heading) in [rad] in the GCS, [3, n_snap]
        arma::uvec *i_row = nullptr);        // Optional output: row (0-based, into the arrays of 'nodes') at or before each snapshot, [n_snap]

    // Replace the rows of nodes by a generated track or by a point drop
    // - Each node keeps its first position; the call applies to all nodes with the name of a given node (sectors, panels)
    // - 'point' keeps one row with speed and an explicit horizontal direction; the other types are tracks of equal speed whose
    //   orientation follows the motion (as set_orientation with force = false does), without direction and time
    void generate_track(
        layout_nodes &nodes,                 // Nodes, updated in place
        const uint_list &i_node,             // Node indices, 0-based
        const std::string &type,             // "point", "linear", "circular" or "street"
        double length = 0.0,                 // Length in [m]; 0 = type default; "point": unused
        double speed = NAN,                  // Speed in [m/s]; NaN = keep the speed of the first row if valid
        const arma::vec &direction = {},     // Start direction (azimuth) in [rad], [1] or one per index in i_node; empty = uniform random
        double spacing = NAN,                // Point spacing in [m]; NaN = type default
        const std::unordered_map<std::string, std::vector<double>> &options = {}, // Options of the type, e.g. {"turn_probability", {0.3}}
        arma::sword seed = -1);              // Seed of the random draws, -1 = system random device

    // Set the orientation of nodes from their motion, or to a given value
    void set_orientation(
        layout_nodes &nodes,                 // Nodes, updated in place
        const arma::vec &orientation = {},   // [bank, tilt, heading] in [rad]; empty = 0
        const uint_list &i_node = {},        // Node indices, 0-based; empty = all nodes
        bool force = false);                 // false: from the motion of tracks plus 'orientation' as offset; true: 'orientation' on all rows

    // ---- Layout generators ----

    // A network layout from a preset: BS sites (hexagonal grid with wrap-around, or indoor grid) and a drop of UTs, or
    // roads with vehicles
    // - Presets of 3GPP TR 38.901: "38.901_UMi", "38.901_UMa", "38.901_RMa", "38.901_SMa", "38.901_InH_Open",
    //   "38.901_InH_Mixed", "38.901_InF_SL", "38.901_InF_DL", "38.901_InF_SH", "38.901_InF_DH", "38.901_InF_HH"
    // - Presets of ITU-R M.2412: "M.2412_InH", "M.2412_DenseUrban", "M.2412_Rural", "M.2412_UrbanMacro"
    // - Presets of 3GPP TR 37.885: "37.885_Urban", "37.885_Highway" (n_ut must be 0)
    // - Tuple entries that are NaN or omitted take the preset value; an entry the preset does not use throws
    // - UTs are point drops with orientation, speed and direction; O2I presets also get indoor, floor and d2d_in
    layout generate_layout(
        const std::string &type,                // Preset
        arma::uword n_ut = 0,                   // UTs per cell (per_cell) or in total; 0 = no UTs
        const arma::vec &spacing = {},          // [d_x, d_y] in [m]; hexagonal: d_x = ISD
        const arma::vec &grid = {},             // [n_x, n_y, n_sector]; hexagonal: n_x = rings (0 = one site), n_y unused
        const arma::vec &area = {},             // [length, width, height] of the hall, floor or road area in [m]
        const arma::vec &bs = {},               // [height, tilt, sector azimuth offset] in [m], [deg], [deg]
        const arma::vec &ut = {},               // [height, min 2D distance, speed, bank, tilt, heading] in [m], [m], [m/s], [deg]
        const arma::vec &indoor = {},           // [indoor ratio, min floors, max floors, floor height in [m]]
        const std::unordered_map<std::string, std::vector<double>> &options = {}, // Scenario-specific values, e.g. {"p_residential", {0.8}}
        bool per_cell = true,                   // true: n_ut per cell; false: n_ut in total
        arma::sword seed = -1);                 // Seed of the drop, -1 = system random device; stored as layout.seed

    // Drop UTs into the cells of the Tx side of a layout, replacing the Rx side
    // - Cells are the Voronoi regions of the sites (with the wrap images), split into wedges around the sector headings
    // - The scenario family of each site gives the rules of the drop in its cells, as in generate_layout
    void drop_terminals(
        layout &layout,                         // Layout, updated in place: Rx side replaced, pairing and links emptied
        arma::uword n_ut,                       // UTs per cell (per_cell) or in total
        const arma::vec &bounds = {},           // [x_min, x_max, y_min, y_max] in [m]; required without two wrap vectors
        const arma::vec &ut = {},               // As in generate_layout
        const arma::vec &indoor = {},           // As in generate_layout
        const std::unordered_map<std::string, std::vector<double>> &options = {}, // As in generate_layout
        bool per_cell = true,                   // true: n_ut per cell; false: n_ut in total
        arma::sword seed = -1);                 // Seed, -1 = layout.seed if set, else the system random device

    // ---- Channels from a layout ----

    // Channel coefficients of a layout in one call: link states, rays and channels
    // - Uses layout.links if it is filled; else draws the link states on a local copy and leaves the layout unchanged
    // - One channel per Tx node, Rx node and frequency of the linked pairs: Tx node fastest, then Rx node, then frequency
    std::vector<channel<double>> get_channels_layout(
        const layout &layout,                                // Layout with frequencies; antennas evaluated or the generic antenna
        const arma::vec &times = {},                         // Snapshot times in [s], [n_snap]; empty = one snapshot per row
        const std::string &output = "taps",                  // "taps" or "rays"
        const std::string &method = "",                      // "38.901", "38.901_MF" or "quadriga"; "" = method of layout.links, else "38.901"
        arma::umat *index = nullptr,                         // Optional output: Tx node, Rx node and frequency index of each channel, 0-based, [3, n_channel]
        const std::string &mobility = "auto",                // "phase", "drift" or "auto" (phase for the TR methods, drift for quadriga)
        double overlap = 0.5,                                // Overlap of consecutive segments, 0 to 1
        arma::sword seed = -1,                               // Master seed if layout.links is empty; -1 = layout.seed, else the system random device
        const std::vector<std::string> &config_dirs = {});   // Folders of user conf files, searched before the shipped folder; unused with filled links

    // Stage 1: draw the discrete state of the propagation links (LOS, indoor state, scenario, wrap image, seeds)
    // - Writes layout.links and the master seed used into layout.seed; a failed call leaves the layout unchanged
    void init_link_states(
        layout &layout,                                      // Layout, links and seed updated in place
        const std::string &method = "38.901",                // "38.901", "38.901_MF" or "quadriga"
        const arma::vec &o2i_low = {},                       // Share of the low-loss building model, [1]; empty = family value
        bool pol_variability = false,                        // Polarization power variability (TR 38.901 clause 7.6.16); TR methods only
        double anchor_freq = 2.0e9,                          // Anchor frequency of "38.901_MF" in [Hz]
        const std::string &sos_preset = "",                  // SOS preset of the GBSM sets; "" = Exp256 for the TR methods, Comb256 for quadriga
        arma::sword seed = -1,                               // Master seed; -1 = layout.seed, else the system random device
        const std::vector<std::string> &config_dirs = {});   // Folders of user conf files, searched before the shipped folder

    // Stage 2: rays of propagation links at the start of their segment
    // - Returns [n_out][n_freq] objects for the TR methods (one frequency each), [n_out][1] for quadriga
    std::vector<std::vector<gbsm_rays>> generate_rays(
        const layout &layout,                                // Layout with links from init_link_states
        const uint_list &i_link = {},                        // Propagation links, 0-based, [n_out]; empty = all
        bool scatterers = false);                            // Solve the scatterers (needed for drift); always done for quadriga

    // Stage 3: channels of the channel links of the given propagation links
    // - Applies the antennas, the motion along the tracks and the link gain to the rays
    std::vector<channel<double>> get_gbsm_channels(
        const layout &layout,                                // Layout with links from init_link_states
        const std::vector<std::vector<gbsm_rays>> &rays,     // Rays from generate_rays, possibly edited, [n_out][1 or n_freq]
        const uint_list &i_link = {},                        // The propagation links of 'rays', 0-based, [n_out]; empty = all
        const arma::vec &times = {},                         // Snapshot times in [s], [n_snap]; empty = one snapshot per row
        const std::string &output = "taps",                  // "taps" or "rays"
        arma::umat *index = nullptr,                         // Optional output: Tx node, Rx node and frequency index of each channel, 0-based, [3, n_channel]
        const std::string &mobility = "auto",                // "phase", "drift" or "auto"; "drift" needs rays with scatterers
        bool oxygen = false,                                 // Oxygen absorption (TR 38.901 clause 7.6.1)
        bool use_absolute_delays = false,                    // Include the LOS delay in all delays
        double overlap = 0.5);                               // Overlap of consecutive segments, 0 to 1

    // Serving Tx node of each terminal by the large-scale gain or by RSRP; returns the Tx node per Rx terminal, 0-based, [n_rx_terminal]
    // - Evaluates all sites at the first position of each node and ignores layout.pairing; a terminal with several nodes
    //   takes the highest value of its nodes, a tie the lowest Tx node index
    arma::uvec attach_terminals(
        const layout &layout,                                    // Layout with links from init_link_states
        const std::string &rule = "large_scale",                 // "large_scale" or "rsrp"
        const std::vector<std::vector<gbsm_rays>> &rays = {},    // "rsrp": rays of all links from generate_rays, required; "large_scale": unused
        arma::uvec *site = nullptr,                              // Optional output: serving site per Rx terminal, 0-based, [n_rx_terminal]
        arma::mat *gain_dB = nullptr,                            // Optional output: coupling gain or RSRP in [dB], [n_rx_terminal, n_tx_node]
        arma::uword i_freq = 0,                                  // Frequency index into layout.center_freq, 0-based
        arma::uword tx_port = 0);                                // Port of the Tx antennas, 0-based
}

#endif
