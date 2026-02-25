// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (http://quadriga-lib.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "quadriga_tools.hpp"

#include <cmath>
#include <stdexcept>

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# calc_rician_k_factor
Calculate the Rician K-Factor from channel impulse response data

## Description:
- The Rician K-Factor (KF) is defined as the ratio of signal power in the dominant line-of-sight
  (LOS) path to the power in the scattered (non-line-of-sight, NLOS) paths.
- The LOS path is identified by matching the absolute path length with the direct distance between
  TX and RX positions (`dTR`).
- All paths arriving within `dTR + window_size` are considered LOS and their power is summed.
- Paths arriving after `dTR + window_size` are considered NLOS and their power is summed.
- If the total NLOS power is zero (i.e. no scattered paths), the K-Factor is set to infinity (`HUGE_VAL`).
- If the total LOS power is zero (i.e. no LOS paths), the K-Factor is set to zero.
- The transmitter and receiver positions can be fixed (size `[3, 1]`) or mobile (size `[3, n_cir]`).
  Fixed positions are reused for all channel snapshots.
- Optional output `pg` returns the total path gain (sum of all path powers) for each snapshot.

## Declaration:
```
template <typename dtype>
void quadriga_lib::calc_rician_k_factor(
    const std::vector<arma::Col<dtype>> &powers,
    const std::vector<arma::Col<dtype>> &path_length,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    arma::Col<dtype> *kf = nullptr,
    arma::Col<dtype> *pg = nullptr,
    dtype window_size = 0.01);
```

## Arguments:
- `const std::vector<arma::Col<dtype>> &**powers**` (input)<br>
  Path powers in Watts [W]. Vector of length `n_cir`, where each element is a column vector of
  length `n_path` (number of paths may vary per snapshot).

- `const std::vector<arma::Col<dtype>> &**path_length**` (input)<br>
  Absolute path lengths from TX to RX phase center in meters. Vector of length `n_cir`, where
  each element is a column vector of length `n_path` matching the corresponding entry in `powers`.

- `const arma::Mat<dtype> &**tx_pos**` (input)<br>
  Transmitter position in Cartesian coordinates [x; y; z]. Size `[3, 1]` for a fixed TX or
  `[3, n_cir]` for a mobile TX.

- `const arma::Mat<dtype> &**rx_pos**` (input)<br>
  Receiver position in Cartesian coordinates [x; y; z]. Size `[3, 1]` for a fixed RX or
  `[3, n_cir]` for a mobile RX.

- `arma::Col<dtype> ***kf** = nullptr` (optional output)<br>
  Rician K-Factor on linear scale. Length `[n_cir]`.

- `arma::Col<dtype> ***pg** = nullptr` (optional output)<br>
  Total path gain (sum of path powers). Length `[n_cir]`.

- `dtype **window_size** = 0.01` (input)<br>
  LOS window size in meters. Paths with length ≤ `dTR + window_size` are considered LOS.

## Example:
```
#include "quadriga_tools.hpp"

// Single snapshot with 3 paths
std::vector<arma::vec> powers(1), path_length(1);
powers[0] = {1.0, 0.5, 0.25};       // Path powers in W
path_length[0] = {10.0, 11.0, 12.0}; // Path lengths in m

arma::mat tx_pos(3, 1), rx_pos(3, 1);
tx_pos.col(0) = {0.0, 0.0, 0.0};
rx_pos.col(0) = {10.0, 0.0, 0.0};   // dTR = 10.0 m

arma::vec kf, pg;
quadriga_lib::calc_rician_k_factor(powers, path_length, tx_pos, rx_pos, &kf, &pg, 0.01);
// kf[0] = 1.0 / (0.5 + 0.25) = 1.333...
// pg[0] = 1.0 + 0.5 + 0.25 = 1.75
```
MD!*/

template <typename dtype>
void quadriga_lib::calc_rician_k_factor(const std::vector<arma::Col<dtype>> &powers,
                                        const std::vector<arma::Col<dtype>> &path_length,
                                        const arma::Mat<dtype> &tx_pos,
                                        const arma::Mat<dtype> &rx_pos,
                                        arma::Col<dtype> *kf,
                                        arma::Col<dtype> *pg,
                                        dtype window_size)
{
    // --- Input validation ---
    arma::uword n_cir = (arma::uword)powers.size();
    if (n_cir == 0)
        throw std::invalid_argument("Input 'powers' must not be empty.");

    if (path_length.size() != n_cir)
        throw std::invalid_argument("Inputs 'powers' and 'path_length' must have the same length.");

    if (tx_pos.n_rows != 3 || (tx_pos.n_cols != 1 && tx_pos.n_cols != n_cir))
        throw std::invalid_argument("Input 'tx_pos' must have 3 rows and either 1 or 'n_cir' columns.");

    if (rx_pos.n_rows != 3 || (rx_pos.n_cols != 1 && rx_pos.n_cols != n_cir))
        throw std::invalid_argument("Input 'rx_pos' must have 3 rows and either 1 or 'n_cir' columns.");

    if (window_size < (dtype)0)
        throw std::invalid_argument("Input 'window_size' must be non-negative.");

    // Check that each element of powers and path_length have matching lengths
    for (arma::uword i = 0; i < n_cir; ++i)
    {
        if (powers[i].n_elem != path_length[i].n_elem)
            throw std::invalid_argument("Each element of 'powers' and 'path_length' must have the same number of elements.");
    }

    // Early return if no outputs are requested
    if (kf == nullptr && pg == nullptr)
        return;

    // --- Determine TX/RX column access pattern ---

    bool tx_fixed = (tx_pos.n_cols == 1);
    bool rx_fixed = (rx_pos.n_cols == 1);

    // --- Allocate outputs ---

    if (kf != nullptr)
        kf->set_size(n_cir);

    if (pg != nullptr)
        pg->set_size(n_cir);

    // --- Compute K-Factor and path gain ---

    const dtype *tx_ptr = tx_pos.memptr();
    const dtype *rx_ptr = rx_pos.memptr();

    for (arma::uword i = 0; i < n_cir; ++i)
    {
        // Get TX and RX position for this snapshot
        const dtype *tx_col = tx_fixed ? tx_ptr : tx_ptr + i * 3;
        const dtype *rx_col = rx_fixed ? rx_ptr : rx_ptr + i * 3;

        // Compute direct distance between TX and RX
        dtype dx = tx_col[0] - rx_col[0];
        dtype dy = tx_col[1] - rx_col[1];
        dtype dz = tx_col[2] - rx_col[2];
        dtype dTR = std::sqrt(dx * dx + dy * dy + dz * dz);

        // LOS threshold
        dtype threshold = dTR + window_size;

        // Sum LOS and NLOS powers
        arma::uword n_path = powers[i].n_elem;
        const dtype *pw_ptr = powers[i].memptr();
        const dtype *pl_ptr = path_length[i].memptr();

        dtype los_power = (dtype)0;
        dtype nlos_power = (dtype)0;

        for (arma::uword p = 0; p < n_path; ++p)
        {
            if (pl_ptr[p] <= threshold)
                los_power += pw_ptr[p];
            else
                nlos_power += pw_ptr[p];
        }

        // Write outputs
        if (kf != nullptr)
        {
            if (nlos_power == (dtype)0)
                (*kf)(i) = (los_power > (dtype)0) ? (dtype)HUGE_VAL : (dtype)0;
            else
                (*kf)(i) = los_power / nlos_power;
        }

        if (pg != nullptr)
            (*pg)(i) = los_power + nlos_power;
    }
}

// --- Explicit template instantiation ---
template void quadriga_lib::calc_rician_k_factor(const std::vector<arma::Col<float>> &powers,
                                                 const std::vector<arma::Col<float>> &path_length,
                                                 const arma::Mat<float> &tx_pos,
                                                 const arma::Mat<float> &rx_pos,
                                                 arma::Col<float> *kf,
                                                 arma::Col<float> *pg,
                                                 float window_size);

template void quadriga_lib::calc_rician_k_factor(const std::vector<arma::Col<double>> &powers,
                                                 const std::vector<arma::Col<double>> &path_length,
                                                 const arma::Mat<double> &tx_pos,
                                                 const arma::Mat<double> &rx_pos,
                                                 arma::Col<double> *kf,
                                                 arma::Col<double> *pg,
                                                 double window_size);
