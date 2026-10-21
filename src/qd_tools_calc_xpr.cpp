// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>

/*!SECTION
Channel statistics
SECTION!*/

/*!MD
# calc_cross_polarization_ratio
Calculate the cross-polarization ratio (XPR) for linear and circular polarization bases

- Computes aggregate XPR from polarization transfer matrices using the total-power-ratio method:
  co-pol and cross-pol powers are summed across all qualifying paths per CIR, and XPR is their ratio.
- XPR is computed in both the linear V/H basis and the circular LHCP/RHCP basis.
- LOS paths are identified by comparing path length against direct TX-RX distance `dTR`; paths with
  `path_length <= dTR + window_size` are excluded by default (`include_los = false`).
- Polarization transfer matrix `M` is the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation
  V/H), stored column-major with interleaved real/imaginary parts, 8 rows per path: rows 0-1 = (V rx, V tx),
  rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx). Below, `M_<rx><tx>` names an entry by its
  receive and transmit polarisation: `M_vv` = rows 0-1, `M_hv` = rows 2-3, `M_vh` = rows 4-5, `M_hh` = rows 6-7
- V-XPR and H-XPR are ratios per receive polarisation: V-XPR is the power received in V from V transmission over
  the power received in V from H transmission; H-XPR likewise for the H receive polarisation
- The circular entries couple circular patterns through `M` as [[get_channels_spherical]] couples antenna patterns,
  `c(r, t) = r^T M t` (plain transpose), with the same circular vectors at both ends, each in that antenna's own
  (e_theta, e_phi) frame: L = (V, H) = (1, +j) / sqrt(2) (LHCP), R = (1, -j) / sqrt(2) (RHCP). The names follow the
  IEEE definition (rotation seen along the propagation direction) for the time convention exp(+j 2 pi f t), which the
  library's coefficients exp(-j 2 pi f tau) imply, and e_theta x e_phi = the outward direction of the pattern.
  `M_<rx><tx>`: `M_LL = (M_vv - M_hh + j(M_vh + M_hv)) / 2`, `M_RR = (M_vv - M_hh - j(M_vh + M_hv)) / 2`,
  `M_LR = (M_vv + M_hh + j(M_hv - M_vh)) / 2` (receive L, transmit R), `M_RL = (M_vv + M_hh + j(M_vh - M_hv)) / 2`.
  Co-polar means the same circular vector at both ends: the LOS matrix `diag(1, -1)` of [[get_channels_spherical]] is
  co-polar (handedness kept), `diag(1, 1)` (one mirrored component, as by a single reflection) is cross-polar.
- LHCP-XPR and RHCP-XPR are ratios per receive polarisation, like V-XPR and H-XPR.
- Normalization of `M` does not affect XPR (cancels in ratio) but does affect `pg`.
- If cross-pol power is zero and co-pol is positive, XPR is set to infinity; if both are zero, XPR is set to 0.
- TX/RX positions may be fixed `[3, 1]` or mobile `[3, n_cir]`.

## Declaration:
```
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

## Inputs:
- **`powers`** — Path powers in [W]; `[n_cir]` vector, each element of length `n_path`
- **`M`** — Polarization transfer matrices; `[n_cir]` vector, each element of size `[8, n_path]`
- **`path_length`** — Absolute TX-to-RX path lengths; same structure as `powers`
- **`tx_pos`** — Transmitter position [x; y; z]; `[3, 1]` or `[3, n_cir]`
- **`rx_pos`** — Receiver position [x; y; z]; `[3, 1]` or `[3, n_cir]`
- **`include_los`** *(optional)* — If true, includes LOS and near-LOS paths in the XPR calculation
- **`window_size`** *(optional)* — LOS exclusion window; paths with `path_length <= dTR + window_size` are excluded
  when `include_los = false`

## Outputs:
- **`xpr`** *(optional)* — XPR on linear scale; `[n_cir, 6]`; columns:<br><br>
   | Col | Description                                                     |
   | --- | --------------------------------------------------------------- |
   | 0   | Aggregate linear XPR (total V+H co-pol / total V+H cross-pol)   |
   | 1   | V-XPR: sum(abs(M_vv)^2) / sum(abs(M_vh)^2) (receive V)          |
   | 2   | H-XPR: sum(abs(M_hh)^2) / sum(abs(M_hv)^2) (receive H)          |
   | 3   | Aggregate circular XPR (total L+R co-pol / total L+R cross-pol) |
   | 4   | LHCP XPR: sum(abs(M_LL)^2) / sum(abs(M_LR)^2) (receive L)       |
   | 5   | RHCP XPR: sum(abs(M_RR)^2) / sum(abs(M_RL)^2) (receive R)       |
- **`pg`** *(optional)* — Total path gain summed over all paths (including LOS) as
  `0.5 * sum(powers * (abs(M_vv)^2 + abs(M_hv)^2 + abs(M_vh)^2 + abs(M_hh)^2))`; `[n_cir]`
MD!*/

template <typename dtype>
void quadriga_lib::calc_cross_polarization_ratio(const std::vector<arma::Col<dtype>> &powers,
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
            dtype a = pM[offset];     // Re(M_vv)
            dtype b = pM[offset + 1]; // Im(M_vv)
            dtype c = pM[offset + 2]; // Re(M_hv), H rx, V tx
            dtype d = pM[offset + 3]; // Im(M_hv)
            dtype e = pM[offset + 4]; // Re(M_vh), V rx, H tx
            dtype f = pM[offset + 5]; // Im(M_vh)
            dtype g = pM[offset + 6]; // Re(M_hh)
            dtype h = pM[offset + 7]; // Im(M_hh)

            dtype abs2_vv = a * a + b * b;
            // Local names keep the old labels: abs2_vh / P_vh = rows 2-3 (H rx, V tx) = M_hv of the docs,
            // abs2_hv / P_hv = rows 4-5 (V rx, H tx) = M_vh of the docs
            dtype abs2_vh = c * c + d * d;
            dtype abs2_hv = e * e + f * f;
            dtype abs2_hh = g * g + h * h;

            dtype w = pP[p];
            dtype path_power = dtype(0.5) * w * (abs2_vv + abs2_hv + abs2_vh + abs2_hh);

            // Always accumulate total path gain (all paths including LOS)
            P_total += path_power;

            // Exclude LOS / near-LOS paths from XPR if requested
            if (!include_los && pL[p] <= threshold)
                continue;

            // --- Linear basis ---
            P_vv += w * abs2_vv;
            P_vh += w * abs2_vh;
            P_hv += w * abs2_hv;
            P_hh += w * abs2_hh;

            // --- Circular basis ---
            // c(r, t) = r^T M t with L = (1, +j)/sqrt(2), R = (1, -j)/sqrt(2) in (V, H) at both ends; M_<rx><tx>
            // M_LL = (M_vv - M_hh + j*(M_vh + M_hv)) / 2
            dtype LL_re = a - g - d - f;
            dtype LL_im = b - h + c + e;
            P_LL += w * (LL_re * LL_re + LL_im * LL_im) / (dtype)4;

            // M_RR = (M_vv - M_hh - j*(M_vh + M_hv)) / 2
            dtype RR_re = a - g + d + f;
            dtype RR_im = b - h - c - e;
            P_RR += w * (RR_re * RR_re + RR_im * RR_im) / (dtype)4;

            // M_LR = (M_vv + M_hh + j*(M_hv - M_vh)) / 2, receive L, transmit R
            dtype LR_re = a + g - d + f;
            dtype LR_im = b + h + c - e;
            P_LR += w * (LR_re * LR_re + LR_im * LR_im) / (dtype)4;

            // M_RL = (M_vv + M_hh + j*(M_vh - M_hv)) / 2, receive R, transmit L
            dtype RL_re = a + g + d - f;
            dtype RL_im = b + h + e - c;
            P_RL += w * (RL_re * RL_re + RL_im * RL_im) / (dtype)4;
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
            dtype co_lin = P_vv + P_hh;
            pX[i] = (cross_lin > (dtype)0) ? co_lin / cross_lin
                    : (co_lin > (dtype)0)  ? std::numeric_limits<dtype>::infinity()
                                           : (dtype)0;

            // Col 1: V-XPR
            pX[i + stride] = (P_hv > (dtype)0)   ? P_vv / P_hv
                             : (P_vv > (dtype)0) ? std::numeric_limits<dtype>::infinity()
                                                 : (dtype)0;

            // Col 2: H-XPR
            pX[i + 2 * stride] = (P_vh > (dtype)0)   ? P_hh / P_vh
                                 : (P_hh > (dtype)0) ? std::numeric_limits<dtype>::infinity()
                                                     : (dtype)0;

            // Col 3: Aggregate circular XPR
            dtype cross_circ = P_LR + P_RL;
            dtype co_circ = P_LL + P_RR;
            pX[i + 3 * stride] = (cross_circ > (dtype)0) ? co_circ / cross_circ
                                 : (co_circ > (dtype)0)  ? std::numeric_limits<dtype>::infinity()
                                                         : (dtype)0;

            // Col 4: LHCP XPR (receive L, co = LL, cross = LR)
            pX[i + 4 * stride] = (P_LR > (dtype)0)   ? P_LL / P_LR
                                 : (P_LL > (dtype)0) ? std::numeric_limits<dtype>::infinity()
                                                     : (dtype)0;

            // Col 5: RHCP XPR (receive R, co = RR, cross = RL)
            pX[i + 5 * stride] = (P_RL > (dtype)0)   ? P_RR / P_RL
                                 : (P_RR > (dtype)0) ? std::numeric_limits<dtype>::infinity()
                                                     : (dtype)0;
        }
    }
}

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