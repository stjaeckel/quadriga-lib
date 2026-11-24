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
        std::vector<std::vector<arrayant<double>>> antenna;      // Evaluated antennas, [n_antenna][1 or n_freq]; empty = not evaluated
        std::vector<std::string> antenna_cols;                   // Names of the antenna columns, [n_col]
        std::vector<std::vector<std::string>> antenna_args;      // CSV lines of the antenna columns, [n_antenna][0, 1 or n_freq]
        arma::mat wrap;                                   // Lattice vectors in [m], one per column, [2, 1] or [2, 2]; empty = none
        arma::umat pairing;                               // Tx site index and Rx terminal index of each linked pair, 0-based, [2, n_pair]; empty = all pairs

        arma::uword n_antenna() const;                    // Length of antenna_args, or of antenna if there are no arguments
        std::string is_valid() const;                     // Empty string if valid, else an error message
    };

    // Read the Tx and Rx CSV files of a layout
    // - Either file name may be "" for no nodes on that side
    // - Antenna columns are composed into antenna_args, antenna_cols and i_antenna; equal descriptions share one entry
    // - The antennas are not evaluated yet: the argument 'eval_antennas' is accepted and ignored, antenna stays empty
    layout layout_csv_read(
        const std::string &fn_tx,                         // Tx CSV file, "" = no Tx nodes
        const std::string &fn_rx,                         // Rx CSV file, "" = no Rx nodes
        const arma::vec &center_freq = {},                // Simulation frequencies in [Hz]; empty = from the files
        bool eval_antennas = true,                        // Currently ignored
        std::vector<std::string> *extra_tx = nullptr,     // Optional output: names of unknown Tx columns
        std::vector<std::string> *extra_rx = nullptr);    // Optional output: names of unknown Rx columns

    // Write a layout to the Tx and Rx CSV files
    // - Either file name may be "" for not written; all files are composed and written to temporary files first, and
    //   only then are the existing files replaced
    // - Antennas that have no CSV description cannot be written yet
    void layout_csv_write(
        const std::string &fn_tx,                         // Tx CSV file, "" = not written
        const std::string &fn_rx,                         // Rx CSV file, "" = not written
        const layout &layout,                             // Layout
        bool compact = true);                             // true: empty cells where the format allows; false: every cell written

}

#endif
