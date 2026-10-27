// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"
#include "quadriga_sos.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    // Bias removal of the generators with gen[g].bias (Normal fields), in place
    // - P0 [n_coeff, n_gen]: Rx-side phases; U [n_pos, 3], n_pos > 2: all positions where the fields are evaluated
    // - One sos_remove_bias call per frequency table with all corrected columns that use it; the offset is computed
    //   per column, so the result equals one call per generator
    void qd_gbsm_remove_bias(const std::vector<qd_gbsm_gen> &gen, const std::vector<arma::mat> &freq,
                             const std::vector<arma::uword> &tab, arma::mat &P0, const arma::mat &U)
    {
        for (arma::uword t = 0; t < (arma::uword)freq.size(); ++t)
        {
            std::vector<arma::uword> cols;
            for (arma::uword g = 0; g < (arma::uword)gen.size(); ++g)
                if (gen[g].bias && tab[g] == t)
                    cols.push_back(g);
            if (cols.empty())
                continue;

            arma::mat ph(P0.n_rows, cols.size());
            for (arma::uword i = 0; i < (arma::uword)cols.size(); ++i)
                ph.col(i) = P0.col(cols[i]);
            quadriga_lib::sos_remove_bias<double>(freq[t], ph, U);
            for (arma::uword i = 0; i < (arma::uword)cols.size(); ++i)
                P0.col(cols[i]) = ph.col(i);
        }
    }
}

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# lsf_init_phases
Draw the random phases of the large-scale generators of one BS site and scenario

- The large-scale fading (LSF) of a site is described by 11 spatially correlated sum-of-sinusoids generators, one
  column each: 0 DS, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA, 7 XPR, 8 absolute TOA (Normal fields), 9 ground
  permittivity, 10 UMa environment height hE (Uniform fields). Their decorrelation distances are the scenario
  parameters `DS_lambda`, `KF_lambda`, `SF_lambda`, `AS_D_lambda`, `AS_A_lambda`, `ES_D_lambda`, `ES_A_lambda`,
  `XPR_lambda`, `absTOA_lambda`, `SC_lambda` and again `SF_lambda` for hE; a decorrelation distance of 0 is used
  as 0.1 m. The generator of column 8 exists also when the absolute TOA is disabled
- This function draws the phases; [[lsf_draw]] evaluates them at the links. The caller stores `phase` and
  evaluates the same fields again along tracks or on maps
- `scenpar` is completed with the defaults of missing keys and validated by [[scenpar_validate]]; its messages
  (e.g. for an unknown key or a negative decorrelation distance) pass through unchanged. An invalid value of a key
  that is not used here throws as well
- Slice 0 (Rx side) holds the phases of the 11 generators. Without bias removal it equals
  `sos_init_phases(n_coeff, 11, seed)` of [[sos_init_phases]], with `n_coeff` the number of sinusoids of
  `preset`. The same seed gives the same phases
- Slice 1 (Tx side) is always the mirror of slice 0: `slice1.col(g) = slice0.col(m(g))` with the mirror
  generator `m(g)`: ASD and ASA (3, 4) and ESD and ESA (5, 6) are exchanged, all others mirror themselves. For dual
  mobility the departure spreads at the Tx are thus the arrival spreads of the Tx as a second device
- Bias removal (`remove_bias = true`, `rx_pos` given, `n_link > 2`): one realisation of a correlated field has a
  non-zero mean over a drop. For each Normal generator (columns 0 to 8) one common phase offset is applied with
  [[sos_remove_bias]], so that the sample mean of its field over the positions is exactly 0. Single mobility: over
  the Rx positions; `tx_pos` is not used. Dual mobility: over the Rx and Tx positions together (`2·n_link`
  positions), because the field of a generator is shared by both devices (the Tx side of ASD is the ASA field);
  `tx_pos` is then required. The Uniform generators 9 and 10 are never corrected
- The correction is applied to slice 0 and the Tx side is mirrored afterwards, so the mirror relation holds
  exactly with and without bias removal. Swapping `tx_pos` and `rx_pos` in dual mobility gives the same phases up to
  rounding (the sums over the positions are formed in another order)
- Consequences for [[lsf_draw]] at the same positions: in single mobility the link mean of columns 0 to 8 is 0; in
  dual mobility the link mean of DS, KF, SF, XPR and TOA is 0 when all links have the same Tx-Rx distance (the
  normalisation is then the same for all links), while ASD/ASA and ESD/ESA have a zero link mean only
  approximately (their two devices see different fields, each corrected over the union of the positions)
- The positions are used as in [[lsf_draw]]: projected on the x-y plane (z = 0) for `horizontal = true`, in 3D
  for `horizontal = false`. `preset` and `horizontal` must be the same in [[lsf_draw]]
- All positions in `rx_pos` are treated as one field (floor 0): one phase correction per generator cannot zero the
  means of several floor fields at once. For a drop with UTs on several floors pass the positions of one floor (e.g.
  the links of floor 0) or set `remove_bias = false`; the fields of the other floors keep their sample mean
- The correction removes variance: the field is conditioned on a zero mean over the positions, which costs little
  only when the positions spread over many decorrelation distances (see [[sos_remove_bias]])
- With fewer than 3 positions, without `rx_pos` or with `remove_bias = false` no correction is made
- Validation order: `seed`, `preset`, `scenpar`, then the positions (3 columns, rows of `tx_pos` 1 or `n_link`
  when both are given, finite values), then the requirement of `tx_pos` for the bias removal in dual mobility

## Declaration:
```
void quadriga_lib::lsf_init_phases(
    const std::unordered_map<std::string, double> &scenpar,
    arma::cube &phase,
    arma::sword seed = -1,
    const arma::mat *tx_pos = nullptr,
    const arma::mat *rx_pos = nullptr,
    bool dual_mobility = false,
    bool remove_bias = true,
    const std::string &preset = "Comb256",
    bool horizontal = true);
```

## Inputs:
- **`scenpar`** — Scenario parameters as returned by [[conf_file_read]] or [[scenpar_validate]]; missing keys
  have their default value; only the 10 decorrelation distances are used
- **`seed`** *(optional)* — Seed of the random phases, must be >= -1; -1 = system random device, a new set of
  phases on each call
- **`tx_pos`** *(optional)* — Tx positions in [m]; used only for the bias removal with dual mobility, required
  there; one row is used for all links; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** *(optional)* — Rx positions of the drop in [m] (one floor) for the bias removal; omitted or fewer
  than 3 rows = no bias removal; `[n_link, 3]`
- **`dual_mobility`** *(optional)* — `true`: Tx and Rx both move, the Tx is a second device on the same maps
- **`remove_bias`** *(optional)* — `true`: remove the sample mean of the Normal fields over the positions
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [[sos_preset]]; sets `n_coeff`
- **`horizontal`** *(optional)* — `true`: 2D correlation in the x-y plane (TR 38.901 behaviour);
  `false`: 3D correlation (legacy QuaDRiGa behaviour)

## Output:
- **`phase`** — Phases in [rad]; slice 0 = Rx side, slice 1 = Tx side (mirror of slice 0); all values in
  [-pi, pi); `[n_coeff, 11, 2]`

## See also:
- [[conf_file_read]] (the scenario parameters)
- [[lsf_draw]] (evaluates the phases at the links)
- [[sos_init_phases]] (the raw phases)
- [[sos_remove_bias]] (the bias removal of one generator)
MD!*/

void quadriga_lib::lsf_init_phases(const std::unordered_map<std::string, double> &scenpar,
                                   arma::cube &phase,
                                   arma::sword seed,
                                   const arma::mat *tx_pos,
                                   const arma::mat *rx_pos,
                                   bool dual_mobility,
                                   bool remove_bias,
                                   const std::string &preset,
                                   bool horizontal)
{
    // Validation, before any output is touched
    if (seed < -1)
        throw std::invalid_argument("Input 'seed' must be >= -1.");

    int type = 0;
    arma::uword n_coeff = 0;
    if (!qd_sos_preset_lookup(preset, type, n_coeff))
        throw std::invalid_argument("Input 'preset' is not a known preset.");

    std::unordered_map<std::string, double> sp = scenpar;
    quadriga_lib::scenpar_validate(sp, "legacy", nullptr, nullptr, true);

    if (rx_pos != nullptr && rx_pos->n_cols != 3)
        throw std::invalid_argument("Input 'rx_pos' must have 3 columns.");
    if (tx_pos != nullptr && tx_pos->n_cols != 3)
        throw std::invalid_argument("Input 'tx_pos' must have 3 columns.");
    if (rx_pos != nullptr && tx_pos != nullptr && tx_pos->n_rows != 1 && tx_pos->n_rows != rx_pos->n_rows)
        throw std::invalid_argument("Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'.");
    if (rx_pos != nullptr && !rx_pos->is_finite())
        throw std::invalid_argument("Input 'rx_pos' must be finite.");
    if (tx_pos != nullptr && !tx_pos->is_finite())
        throw std::invalid_argument("Input 'tx_pos' must be finite.");

    const arma::uword n_link = (rx_pos != nullptr) ? rx_pos->n_rows : 0;
    const bool do_bias = remove_bias && n_link > 2;
    if (do_bias && dual_mobility && tx_pos == nullptr)
        throw std::invalid_argument("Input 'tx_pos' is required for the bias removal with dual mobility.");

    // Raw phases of the Rx side
    const std::vector<qd_gbsm_gen> gen = qd_gbsm_lsf_generators(sp);
    const arma::uword n_gen = (arma::uword)gen.size();
    arma::mat P0;
    quadriga_lib::sos_init_phases<double>(n_coeff, n_gen, P0, seed);

    // Bias removal over all positions where the fields are evaluated
    if (do_bias)
    {
        std::vector<arma::mat> freq;
        std::vector<arma::vec> acf, dist;
        std::vector<arma::uword> tab;
        qd_gbsm_freq(preset, gen, freq, acf, dist, tab);

        arma::mat R, T;
        qd_gbsm_positions(*rx_pos, (tx_pos != nullptr) ? *tx_pos : *rx_pos, horizontal, R, T);
        if (dual_mobility)
            qd_gbsm_remove_bias(gen, freq, tab, P0, arma::join_cols(R, T));
        else
            qd_gbsm_remove_bias(gen, freq, tab, P0, R);
    }

    // Output: slice 0 = Rx side, slice 1 = mirror of the Rx side
    if (phase.n_rows != n_coeff || phase.n_cols != n_gen || phase.n_slices != 2)
        phase.set_size(n_coeff, n_gen, 2);
    for (arma::uword g = 0; g < n_gen; ++g)
    {
        phase.slice(0).col(g) = P0.col(g);
        phase.slice(1).col(g) = P0.col(gen[g].mirror);
    }
}
