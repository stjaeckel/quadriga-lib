// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "qd_helpers_shared.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>

/*!SECTION
Channel data and processing
SECTION!*/

/*!MD
# calc_xpr
Calculate the cross-polarization ratio (XPR) of each CIR in the linear and the circular polarization basis

- Total-power ratio: the co- and cross-polar powers of the paths of a CIR, weighted by the path powers $p_l$, are
  summed, and the XPR is their ratio; LOS paths are left out unless `include_los` is set
- Entries of the polarization transfer matrix $M$ of a path, first index receive, second transmit polarization:
  rows 0-1 of `polmat` = $M_{vv}$, 2-3 = $M_{hv}$ (H rx, V tx), 4-5 = $M_{vh}$ (V rx, H tx), 6-7 = $M_{hh}$, each as
  real and imaginary part
- Circular basis: $M_{rt} = \mathbf{r}^T M \, \mathbf{t}$ (plain transpose), as [[get_channels_spherical]]
  couples antenna patterns, with the same vectors at both ends, each in the $(e_\theta, e_\phi)$ frame of its antenna:
  $\mathbf{L} = (1, +j) / \sqrt{2}$ (LHCP) and $\mathbf{R} = (1, -j) / \sqrt{2}$ (RHCP) in (V, H), which gives

  $$M_{LL}, M_{RR} = \frac{1}{2} \left( M_{vv} - M_{hh} \pm j (M_{vh} + M_{hv}) \right)$$

  $$M_{LR}, M_{RL} = \frac{1}{2} \left( M_{vv} + M_{hh} \pm j (M_{hv} - M_{vh}) \right)$$

  (upper sign for the first entry)

- The names L and R follow the IEEE definition (rotation seen along the propagation direction) for the time
  convention $e^{+j 2\pi f t}$, which the coefficients $e^{-j 2\pi f \tau}$ of the library imply, with
  $e_\theta \times e_\phi$ the outward direction of the pattern
- Co-polar means the same circular vector at both ends: the LOS matrix $\mathrm{diag}(1, -1)$ of
  [[get_channels_spherical]] is co-polar (handedness kept), $\mathrm{diag}(1, 1)$ (one mirrored component, as
  by a single reflection) is cross-polar

## Declaration:
```
void quadriga_lib::calc_xpr(
    const std::vector<arma::Col<dtype>> &powers,
    const std::vector<arma::Mat<dtype>> &polmat,
    const std::vector<arma::Col<dtype>> &path_length,
    const arma::Mat<dtype> &tx_pos,
    const arma::Mat<dtype> &rx_pos,
    arma::Mat<dtype> *xpr = nullptr,
    arma::Col<dtype> *pg = nullptr,
    bool include_los = false,
    double window_size = 0.01);
```

## Inputs:
- **`powers`** — Path powers $p_l$, linear; `[n_cir]` vectors `[n_path]`, at least one; `n_path` may differ between
  the CIRs
- **`polmat`** — Polarization transfer matrices, magnitudes below about 1e18; `[n_cir]` matrices `[8, n_path]`
- **`path_length`** — Path lengths in m; `[n_cir]` vectors `[n_path]`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx positions in m; `[3, 1]` (fixed) or `[3, n_cir]`
- **`include_los`** *(optional)* — Include the LOS paths in the XPR
- **`window_size`** *(optional)* — LOS window in m, >= 0: paths with `path_length` <= $d_\mathrm{TR}$ +
  `window_size` are LOS, $d_\mathrm{TR}$ the Tx-Rx distance

## Outputs:
- **`xpr`** *(optional)* — XPR, linear; `[n_cir, 6]`; columns:

  | Column | XPR |
  | ------ | --- |
  | 0 | Linear: $(P_{vv} + P_{hh}) / (P_{vh} + P_{hv})$ with $P_{rt} = \sum_l p_l \lvert M_{rt} \rvert^2$ |
  | 1 | V (receive V): $P_{vv} / P_{vh}$ |
  | 2 | H (receive H): $P_{hh} / P_{hv}$ |
  | 3 | Circular: $(P_{LL} + P_{RR}) / (P_{LR} + P_{RL})$ |
  | 4 | LHCP (receive L): $P_{LL} / P_{LR}$ |
  | 5 | RHCP (receive R): $P_{RR} / P_{RL}$ |

- **`pg`** *(optional)* — Path gain of all paths, LOS included,
  $\frac{1}{2} \sum_l p_l \sum_{r,t \in \{v,h\}} \lvert M_{rt} \rvert^2$; `[n_cir]`

## Notes:
- A ratio with zero cross-polar power is infinite, or 0 if the co-polar power is zero too (also for a CIR without
  paths, which has `pg` = 0)
- A scaling of `polmat` does not change the XPR, but `pg`

## See also:
- [[calc_rician_k_factor]] (LOS and NLOS power with the same LOS window)
- [[get_channels_spherical]] (polarization transfer matrices of the paths)
MD!*/

template <typename dtype>
void quadriga_lib::calc_xpr(const std::vector<arma::Col<dtype>> &powers,
                            const std::vector<arma::Mat<dtype>> &polmat,
                            const std::vector<arma::Col<dtype>> &path_length,
                            const arma::Mat<dtype> &tx_pos,
                            const arma::Mat<dtype> &rx_pos,
                            arma::Mat<dtype> *xpr,
                            arma::Col<dtype> *pg,
                            bool include_los,
                            double window_size_in)
{
    const dtype window_size = dtype(window_size_in);
    // --- Input validation ---
    arma::uword n_cir = (arma::uword)powers.size();
    if (n_cir == 0)
        throw std::invalid_argument("Input 'powers' must not be empty.");

    if ((arma::uword)polmat.size() != n_cir)
        throw std::invalid_argument("Input 'polmat' must have the same number of elements as 'powers'.");

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

    qd_check_alias(xpr, "xpr", &tx_pos, &rx_pos);
    qd_check_alias(pg, "pg", &tx_pos, &rx_pos);
    qd_check_alias_outputs(xpr, "xpr", pg, "pg");
    for (arma::uword i = 0; i < n_cir; ++i)
        qd_check_alias(xpr, "xpr", &powers[i], &polmat[i], &path_length[i]),
            qd_check_alias(pg, "pg", &powers[i], &polmat[i], &path_length[i]);

    if (xpr == nullptr && pg == nullptr)
        return;

    // Per-entry checks, before any output is written
    for (arma::uword i = 0; i < n_cir; ++i)
    {
        arma::uword n_path = powers[i].n_elem;
        if (n_path == 0)
            continue;
        if (polmat[i].n_rows != 8 || polmat[i].n_cols != n_path)
            throw std::invalid_argument("Entry " + std::to_string(i) + ": Input 'polmat' must have 8 rows and " + std::to_string(n_path) + " columns.");
        if (path_length[i].n_elem != n_path)
            throw std::invalid_argument("Entry " + std::to_string(i) + ": Input 'path_length' must have " + std::to_string(n_path) + " elements.");
    }

    // --- Initialize outputs ---
    if (xpr != nullptr)
        xpr->zeros(n_cir, 6);

    if (pg != nullptr)
        pg->zeros(n_cir);

    // --- Precompute fixed positions ---
    const dtype *pTX = tx_pos.memptr();
    const dtype *pRX = rx_pos.memptr();

    // --- Process each CIR ---
    for (arma::uword i = 0; i < n_cir; ++i)
    {
        arma::uword n_path = powers[i].n_elem;

        if (n_path == 0)
            continue;

        // --- Calculate TX-RX distance ---
        arma::uword tx_col = fixed_tx ? 0 : i;
        arma::uword rx_col = fixed_rx ? 0 : i;
        const dtype *tx_c = &pTX[tx_col * 3], *rx_c = &pRX[rx_col * 3];
        dtype dTR = qd_calc_length(rx_c[0], rx_c[1], rx_c[2], tx_c[0], tx_c[1], tx_c[2]);
        dtype threshold = dTR + window_size;

        // --- Accumulate polarization powers ---
        dtype P_vv = (dtype)0, P_hv = (dtype)0, P_vh = (dtype)0, P_hh = (dtype)0;
        dtype P_LL = (dtype)0, P_RL = (dtype)0, P_LR = (dtype)0, P_RR = (dtype)0;
        dtype P_total = (dtype)0;

        const dtype *pM = polmat[i].memptr();
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
            // The local names are swapped against the docs: abs2_vh / P_vh = rows 2-3 (H rx, V tx) = M_hv of the docs,
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

template void quadriga_lib::calc_xpr(
    const std::vector<arma::Col<float>> &powers,
    const std::vector<arma::Mat<float>> &polmat,
    const std::vector<arma::Col<float>> &path_length,
    const arma::Mat<float> &tx_pos,
    const arma::Mat<float> &rx_pos,
    arma::Mat<float> *xpr,
    arma::Col<float> *pg,
    bool include_los,
    double window_size);

template void quadriga_lib::calc_xpr(
    const std::vector<arma::Col<double>> &powers,
    const std::vector<arma::Mat<double>> &polmat,
    const std::vector<arma::Col<double>> &path_length,
    const arma::Mat<double> &tx_pos,
    const arma::Mat<double> &rx_pos,
    arma::Mat<double> *xpr,
    arma::Col<double> *pg,
    bool include_los,
    double window_size);
