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
#include <limits>

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

## Description:
- Computes the aggregate cross-polarization ratio (XPR) from the polarization transfer matrices
  of all channel impulse responses (CIRs) using the total-power-ratio method (Option B).
- For each CIR, the total co-polarized and cross-polarized received powers are accumulated
  across all qualifying paths, and the XPR is obtained as a single ratio of the totals.
- This method is physically meaningful: it corresponds to what a receiver antenna measures as
  the ratio of co-polarized to cross-polarized energy across the entire channel impulse response.
- In addition to the linear V/H basis, the XPR is also computed in the circular LHCP/RHCP basis
  by applying the unitary Jones matrix transformation M_circ = T * M_lin * T^-1.
- The LOS path is identified by comparing each path's absolute length against the direct
  TX-RX distance `dTR`. All paths with `path_length < dTR + window_size` are excluded from
  the XPR calculation by default (controlled by `include_los`).
- The polarization transfer matrix `M` is stored in column-major order with interleaved
  real/imaginary parts: rows = [Re(M_vv), Im(M_vv), Re(M_hv), Im(M_hv), Re(M_vh), Im(M_vh),
  Re(M_hh), Im(M_hh)], i.e., 8 rows per path.
- `M` may or may not be normalized. Normalization does not affect the XPR since it cancels
  in the ratio. However, it does affect the path gain output `pg`.
- If the total cross-polarized power is zero (perfect polarization isolation), the XPR is
  set to 0 (undefined). Check `pg` to distinguish this from a true zero-isolation channel.

## Declaration:
```
template <typename dtype>
void quadriga_lib::calc_cross_polarization_ratio(
    const std::vector<arma::Col<dtype>> &powers,
    const std::vector<arma::Mat<dtype>> &M,
    const std::vector<arma::Col<dtype>> &path_length,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    arma::Mat<dtype> *xpr = nullptr,
    arma::Col<dtype> *pg = nullptr,
    bool include_los = false,
    dtype window_size = 0.01);
```

## Arguments:
- `const std::vector<arma::Col<dtype>> &**powers**` (input)<br>
  Path powers in Watts. Vector of length `[n_cir]`, each element is a column vector of length `[n_path]`.

- `const std::vector<arma::Mat<dtype>> &**M**` (input)<br>
  Polarization transfer matrices. Vector of length `[n_cir]`, each element is a matrix of size `[8, n_path]`
  with interleaved real/imaginary parts in column-major order.

- `const std::vector<arma::Col<dtype>> &**path_length**` (input)<br>
  Absolute path length from TX to RX phase center in meters. Vector of length `[n_cir]`,
  each element is a column vector of length `[n_path]`.

- `const arma::Mat<dtype> &**tx_pos**` (input)<br>
  Transmitter position in Cartesian coordinates. Size `[3, 1]` (fixed TX) or `[3, n_cir]` (mobile TX).

- `const arma::Mat<dtype> &**rx_pos**` (input)<br>
  Receiver position in Cartesian coordinates. Size `[3, 1]` (fixed RX) or `[3, n_cir]` (mobile RX).

- `arma::Mat<dtype> ***xpr** = nullptr` (optional output)<br>
  Cross-polarization ratio in linear scale. Size `[n_cir, 6]` with columns:<br>
  0 = Aggregate linear XPR (total V+H co-pol / total V+H cross-pol),<br>
  1 = V-XPR (|M_vv|^2 / |M_hv|^2, power-summed over paths),<br>
  2 = H-XPR (|M_hh|^2 / |M_vh|^2, power-summed over paths),<br>
  3 = Aggregate circular XPR (total L+R co-pol / total L+R cross-pol),<br>
  4 = LHCP XPR (|M_LL|^2 / |M_RL|^2, power-summed over paths),<br>
  5 = RHCP XPR (|M_RR|^2 / |M_LR|^2, power-summed over paths).

- `arma::Col<dtype> ***pg** = nullptr` (optional output)<br>
  Total path gain computed over all paths (including LOS). Length `[n_cir]`.
  Calculated as the sum of `powers[p] * (|M_vv|^2 + |M_hv|^2 + |M_vh|^2 + |M_hh|^2)` over all paths.

- `bool **include_los** = false` (input)<br>
  If `true`, include LOS and near-LOS paths in the XPR calculation.
  If `false` (default), exclude paths with `path_length < dTR + window_size`.

- `dtype **window_size** = 0.01` (input)<br>
  LOS window size in meters. Paths within `dTR + window_size` of the direct path are excluded
  from the XPR calculation when `include_los` is `false`. Default is 0.01 m (1 cm).

## Example:
```
#include "quadriga_channel.hpp"

// Single CIR with 3 paths
arma::vec pw = {1.0, 0.5, 0.3};
arma::mat M(8, 3);
M.col(0) = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0}; // LOS: diagonal
M.col(1) = {0.9, 0.0, 0.1, 0.0, 0.1, 0.0, 0.8, 0.0};   // NLOS
M.col(2) = {0.7, 0.0, 0.2, 0.0, 0.15, 0.0, 0.6, 0.0};   // NLOS

arma::vec pl = {10.0, 12.0, 15.0};
arma::mat tx(3, 1); tx.col(0) = {0.0, 0.0, 0.0};
arma::mat rx(3, 1); rx.col(0) = {10.0, 0.0, 0.0};

std::vector<arma::vec> powers_vec = {pw};
std::vector<arma::mat> M_vec = {M};
std::vector<arma::vec> pl_vec = {pl};

arma::mat xpr;
arma::vec pg;

quadriga_lib::calc_cross_polarization_ratio(powers_vec, M_vec, pl_vec, tx, rx, &xpr, &pg);
// xpr has size [1, 6], pg has size [1]
```
MD!*/

template <typename dtype>
void quadriga_lib::calc_cross_polarization_ratio(
    const std::vector<arma::Col<dtype>> &powers,
    const std::vector<arma::Mat<dtype>> &M,
    const std::vector<arma::Col<dtype>> &path_length,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    arma::Mat<dtype> *xpr,
    arma::Col<dtype> *pg,
    bool include_los,
    dtype window_size)
{
    // --- Input validation ---
    arma::uword n_cir = (arma::uword)powers.size();
    if (n_cir == 0)
        throw std::invalid_argument("Input 'powers' must not be empty.");

    if ((arma::uword)M.size() != n_cir)
        throw std::invalid_argument("Input 'M' must have the same number of elements as 'powers'.");

    if ((arma::uword)path_length.size() != n_cir)
        throw std::invalid_argument("Input 'path_length' must have the same number of elements as 'powers'.");

    if (tx_pos.n_rows != 3)
        throw std::invalid_argument("Input 'tx_pos' must have 3 rows.");

    if (rx_pos.n_rows != 3)
        throw std::invalid_argument("Input 'rx_pos' must have 3 rows.");

    bool fixed_tx = (tx_pos.n_cols == 1);
    bool fixed_rx = (rx_pos.n_cols == 1);

    if (!fixed_tx && tx_pos.n_cols != n_cir)
        throw std::invalid_argument("Input 'tx_pos' must have 1 or 'n_cir' columns.");

    if (!fixed_rx && rx_pos.n_cols != n_cir)
        throw std::invalid_argument("Input 'rx_pos' must have 1 or 'n_cir' columns.");

    if (window_size < (dtype)0)
        throw std::invalid_argument("Input 'window_size' must be non-negative.");

    // --- Initialize outputs ---
    if (xpr != nullptr)
        xpr->zeros(n_cir, 6);

    if (pg != nullptr)
        pg->zeros(n_cir);

    if (xpr == nullptr && pg == nullptr)
        return;

    // --- Precompute fixed positions ---
    const dtype *pTX = tx_pos.memptr();
    const dtype *pRX = rx_pos.memptr();

    // --- Process each CIR ---
    for (arma::uword i = 0; i < n_cir; ++i)
    {
        arma::uword n_path = powers[i].n_elem;

        if (n_path == 0)
            continue;

        if (M[i].n_rows != 8 || M[i].n_cols != n_path)
            throw std::invalid_argument("Input 'M' element " + std::to_string(i) +
                                        " must have 8 rows and " + std::to_string(n_path) + " columns.");

        if (path_length[i].n_elem != n_path)
            throw std::invalid_argument("Input 'path_length' element " + std::to_string(i) +
                                        " must have " + std::to_string(n_path) + " elements.");

        // --- Calculate TX-RX distance ---
        arma::uword tx_col = fixed_tx ? 0 : i;
        arma::uword rx_col = fixed_rx ? 0 : i;
        dtype dx = pTX[tx_col * 3] - pRX[rx_col * 3];
        dtype dy = pTX[tx_col * 3 + 1] - pRX[rx_col * 3 + 1];
        dtype dz = pTX[tx_col * 3 + 2] - pRX[rx_col * 3 + 2];
        dtype dTR = std::sqrt(dx * dx + dy * dy + dz * dz);
        dtype threshold = dTR + window_size;

        // --- Accumulate polarization powers ---
        dtype P_vv = (dtype)0, P_hv = (dtype)0, P_vh = (dtype)0, P_hh = (dtype)0;
        dtype P_LL = (dtype)0, P_RL = (dtype)0, P_LR = (dtype)0, P_RR = (dtype)0;
        dtype P_total = (dtype)0;

        const dtype *pM = M[i].memptr();
        const dtype *pP = powers[i].memptr();
        const dtype *pL = path_length[i].memptr();

        for (arma::uword p = 0; p < n_path; ++p)
        {
            arma::uword offset = p * 8;
            dtype a = pM[offset];       // Re(M_vv)
            dtype b = pM[offset + 1];   // Im(M_vv)
            dtype c = pM[offset + 2];   // Re(M_hv)
            dtype d = pM[offset + 3];   // Im(M_hv)
            dtype e = pM[offset + 4];   // Re(M_vh)
            dtype f = pM[offset + 5];   // Im(M_vh)
            dtype g = pM[offset + 6];   // Re(M_hh)
            dtype h = pM[offset + 7];   // Im(M_hh)

            dtype abs2_vv = a * a + b * b;
            dtype abs2_hv = c * c + d * d;
            dtype abs2_vh = e * e + f * f;
            dtype abs2_hh = g * g + h * h;

            dtype w = pP[p];
            dtype path_power = w * (abs2_vv + abs2_hv + abs2_vh + abs2_hh);

            // Always accumulate total path gain (all paths including LOS)
            P_total += path_power;

            // Exclude LOS / near-LOS paths from XPR if requested
            if (!include_los && pL[p] < threshold)
                continue;

            // --- Linear basis ---
            P_vv += w * abs2_vv;
            P_hv += w * abs2_hv;
            P_vh += w * abs2_vh;
            P_hh += w * abs2_hh;

            // --- Circular basis ---
            // M_LL = (M_vv + M_hh + j*(M_hv - M_vh)) / 2
            dtype LL_re = a + g - d + f;
            dtype LL_im = b + h + c - e;
            P_LL += w * (LL_re * LL_re + LL_im * LL_im) / (dtype)4;

            // M_RL = (M_vv - M_hh - j*(M_hv + M_vh)) / 2
            dtype RL_re = a - g + d + f;
            dtype RL_im = b - h - c - e;
            P_RL += w * (RL_re * RL_re + RL_im * RL_im) / (dtype)4;

            // M_LR = (M_vv - M_hh + j*(M_hv + M_vh)) / 2
            dtype LR_re = a - g - d - f;
            dtype LR_im = b - h + c + e;
            P_LR += w * (LR_re * LR_re + LR_im * LR_im) / (dtype)4;

            // M_RR = (M_vv + M_hh - j*(M_hv - M_vh)) / 2
            dtype RR_re = a + g + d - f;
            dtype RR_im = b + h - c + e;
            P_RR += w * (RR_re * RR_re + RR_im * RR_im) / (dtype)4;
        }

        // --- Write outputs ---
        if (pg != nullptr)
            pg->at(i) = P_total;

        if (xpr != nullptr)
        {
            dtype *pX = xpr->memptr();
            arma::uword stride = n_cir; // column-major stride

            // Col 0: Aggregate linear XPR
            dtype cross_lin = P_hv + P_vh;
            pX[i] = (cross_lin > (dtype)0) ? (P_vv + P_hh) / cross_lin : (dtype)0;

            // Col 1: V-XPR
            pX[i + stride] = (P_hv > (dtype)0) ? P_vv / P_hv : (dtype)0;

            // Col 2: H-XPR
            pX[i + 2 * stride] = (P_vh > (dtype)0) ? P_hh / P_vh : (dtype)0;

            // Col 3: Aggregate circular XPR
            dtype cross_circ = P_RL + P_LR;
            pX[i + 3 * stride] = (cross_circ > (dtype)0) ? (P_LL + P_RR) / cross_circ : (dtype)0;

            // Col 4: LHCP XPR (transmit L, co=LL, cross=RL)
            pX[i + 4 * stride] = (P_RL > (dtype)0) ? P_LL / P_RL : (dtype)0;

            // Col 5: RHCP XPR (transmit R, co=RR, cross=LR)
            pX[i + 5 * stride] = (P_LR > (dtype)0) ? P_RR / P_LR : (dtype)0;
        }
    }
}

// --- Explicit template instantiation ---
template void quadriga_lib::calc_cross_polarization_ratio(
    const std::vector<arma::Col<float>> &powers,
    const std::vector<arma::Mat<float>> &M,
    const std::vector<arma::Col<float>> &path_length,
    const arma::Mat<float> &tx_pos,
    const arma::Mat<float> &rx_pos,
    arma::Mat<float> *xpr,
    arma::Col<float> *pg,
    bool include_los,
    float window_size);

template void quadriga_lib::calc_cross_polarization_ratio(
    const std::vector<arma::Col<double>> &powers,
    const std::vector<arma::Mat<double>> &M,
    const std::vector<arma::Col<double>> &path_length,
    const arma::Mat<double> &tx_pos,
    const arma::Mat<double> &rx_pos,
    arma::Mat<double> *xpr,
    arma::Col<double> *pg,
    bool include_los,
    double window_size);
