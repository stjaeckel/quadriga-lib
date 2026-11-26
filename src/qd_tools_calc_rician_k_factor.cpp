// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "qd_helpers_shared.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# calc_rician_k_factor
Calculate the Rician K-factor of each CIR, the ratio of the LOS to the NLOS power

- $K = P_\mathrm{LOS} / P_\mathrm{NLOS}$ with the sums of the path powers; LOS paths are those with a length
  `path_length` <= $d_\mathrm{TR}$ + `window_size`, $d_\mathrm{TR}$ the Tx-Rx distance, all others are NLOS
- Without NLOS power, $K$ is infinite, or 0 if the LOS power is zero too (also for a CIR without paths)

## Declaration:
```
void quadriga_lib::calc_rician_k_factor(
    const std::vector<arma::Col<dtype>> &powers,
    const std::vector<arma::Col<dtype>> &path_length,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    arma::Col<dtype> *kf = nullptr,
    arma::Col<dtype> *pg = nullptr,
    double window_size = 0.01);
```

## Inputs:
- **`powers`** — Path powers, linear; `[n_cir]` vectors `[n_path]`, at least one CIR; `n_path` may differ between the
  CIRs
- **`path_length`** — Path lengths in m; `[n_cir]` vectors `[n_path]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `[3, 1]` (fixed) or `[3, n_cir]`
- **`window_size`** *(optional)* — LOS window in m, >= 0

## Outputs:
- **`kf`** *(optional)* — K-factor, linear; `[n_cir]`
- **`pg`** *(optional)* — Sum of all path powers, linear; `[n_cir]`

## See also:
- [[calc_xpr]] (XPR with the same LOS window)
- [[calc_delay_spread]] (delay spread of the same paths)
MD!*/

template <typename dtype>
void quadriga_lib::calc_rician_k_factor(const std::vector<arma::Col<dtype>> &powers,
                                        const std::vector<arma::Col<dtype>> &path_length,
                                        const arma::Mat<dtype> &tx_pos,
                                        const arma::Mat<dtype> &rx_pos,
                                        arma::Col<dtype> *kf,
                                        arma::Col<dtype> *pg,
                                        double window_size_in)
{
    const dtype window_size = dtype(window_size_in);
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
            throw std::invalid_argument("Entry " + std::to_string(i) + ": Inputs 'powers' and 'path_length' must have the same number of elements.");
    }

    qd_check_alias(kf, "kf", &tx_pos, &rx_pos);
    qd_check_alias(pg, "pg", &tx_pos, &rx_pos);
    qd_check_alias_outputs(kf, "kf", pg, "pg");
    for (arma::uword i = 0; i < n_cir; ++i)
        qd_check_alias(kf, "kf", &powers[i], &path_length[i]),
            qd_check_alias(pg, "pg", &powers[i], &path_length[i]);

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
        dtype dTR = qd_calc_length(rx_col[0], rx_col[1], rx_col[2], tx_col[0], tx_col[1], tx_col[2]);

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

template void quadriga_lib::calc_rician_k_factor(const std::vector<arma::Col<float>> &powers,
                                                 const std::vector<arma::Col<float>> &path_length,
                                                 const arma::Mat<float> &tx_pos,
                                                 const arma::Mat<float> &rx_pos,
                                                 arma::Col<float> *kf,
                                                 arma::Col<float> *pg,
                                                 double window_size);

template void quadriga_lib::calc_rician_k_factor(const std::vector<arma::Col<double>> &powers,
                                                 const std::vector<arma::Col<double>> &path_length,
                                                 const arma::Mat<double> &tx_pos,
                                                 const arma::Mat<double> &rx_pos,
                                                 arma::Col<double> *kf,
                                                 arma::Col<double> *pg,
                                                 double window_size);
