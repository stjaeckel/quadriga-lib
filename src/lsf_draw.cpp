// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"
#include "quadriga_sos.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    // Phase offsets of floor q, [n_coeff]: 2·pi·(q mod N)·n/N + psi_(q div N)(n)
    // - N = n_coeff; psi_0 = 0; psi_b(n) = 2·pi·u_n for b >= 1, with u_n the n-th value of qd_sos_uniform of the
    //   stream qd_sos_rng(b)
    // - Two floors in the same block of N IDs have the cross-correlation (1/N)·sum_n cos(Delta_q(n) - Delta_q'(n)) = 0
    //   at zero horizontal lag; floors in different blocks a random one with standard deviation 1/sqrt(2N)
    arma::vec qd_gbsm_floor_offset(arma::uword n_coeff, arma::uword q)
    {
        const double two_pi = 6.283185307179586;
        const arma::uword k = q % n_coeff, b = q / n_coeff;
        arma::vec offset(n_coeff);
        for (arma::uword n = 0; n < n_coeff; ++n) // k·n is reduced modulo N, which keeps the argument below 2·pi
            offset.at(n) = two_pi * (double)((k * n) % n_coeff) / (double)n_coeff;
        if (b > 0)
        {
            std::mt19937_64 rng = qd_sos_rng((arma::sword)b);
            for (arma::uword n = 0; n < n_coeff; ++n)
                offset.at(n) += two_pi * qd_sos_uniform(rng);
        }
        return offset;
    }

    // Normalized values of a phase set at a set of links
    // - gen, freq, acf, dist, tab: generator table and frequency tables (qd_gbsm_lsf_generators, qd_gbsm_freq)
    // - phase [n_coeff, n_gen, 2]: slice 0 Rx side, slice 1 Tx side (mirror of slice 0)
    // - R, T [n_link, 3], n_link >= 1: positions from qd_gbsm_positions (T unused without dual mobility)
    // - floor_id: [n_link] or nullptr (all on floor 0)
    // - val [n_link, n_gen]: column g Normal(0,1), or Uniform(0,1) (map 0.5·erfc(-x/sqrt(2))) if gen[g].uniform
    // - Rx side of generator g: field of g at R (phases slice0.col(g), table tab[g]); Tx side (dual mobility): field
    //   of the mirror generator m(g) at T (phases slice1.col(g) = slice0.col(m(g)), table tab[m(g)])
    // - Dual mobility: self-mirrored generators (a + b) / sqrt(2 + 2·A_g(D)), pairs (a + b) / sqrt(2), with the
    //   Tx-Rx distance D and the ACF A_g of the table of g
    // - Floors: the offsets of qd_gbsm_floor_offset are added to all phases of both slices
    // - One sos_val call per floor, frequency table and side, with all phase columns that use the table
    void qd_gbsm_eval(const std::vector<qd_gbsm_gen> &gen, const std::vector<arma::mat> &freq,
                      const std::vector<arma::vec> &acf, const std::vector<arma::vec> &dist,
                      const std::vector<arma::uword> &tab, const arma::cube &phase,
                      const arma::mat &R, const arma::mat &T, const arma::uvec *floor_id,
                      bool dual_mobility, arma::mat &val)
    {
        const arma::uword n_link = R.n_rows, n_gen = (arma::uword)gen.size(), n_coeff = phase.n_rows;
        const arma::uword n_tab = (arma::uword)freq.size();

        // Fields of the Rx side (written to 'val') and of the Tx side ('b'), all evaluated as Normal
        if (val.n_rows != n_link || val.n_cols != n_gen)
            val.set_size(n_link, n_gen);
        arma::mat b;
        if (dual_mobility)
            b.set_size(n_link, n_gen);

        // Links grouped by floor ID (ordered map: the groups are processed in increasing order of the ID)
        std::map<arma::uword, std::vector<arma::uword>> floors;
        if (floor_id == nullptr)
            floors[0] = {};
        else
            for (arma::uword l = 0; l < n_link; ++l)
                floors[floor_id->at(l)].push_back(l);
        const bool one_group = floors.size() == 1; // all links in their order, no copies of the positions

        for (const auto &fl : floors)
        {
            arma::uvec L;
            arma::mat RL, TL;
            if (!one_group)
            {
                L = arma::conv_to<arma::uvec>::from(fl.second);
                RL = R.rows(L);
                if (dual_mobility)
                    TL = T.rows(L);
            }
            const arma::uword n_fl = one_group ? n_link : L.n_elem;

            arma::vec offset;
            if (fl.first != 0)
                offset = qd_gbsm_floor_offset(n_coeff, fl.first);

            for (int side = 0; side < (dual_mobility ? 2 : 1); ++side)
                for (arma::uword t = 0; t < n_tab; ++t)
                {
                    // Columns whose field on this side uses table t
                    std::vector<arma::uword> cols;
                    for (arma::uword g = 0; g < n_gen; ++g)
                        if ((side == 0 ? tab[g] : tab[gen[g].mirror]) == t)
                            cols.push_back(g);
                    if (cols.empty())
                        continue;

                    arma::mat ph(n_coeff, cols.size());
                    for (arma::uword i = 0; i < (arma::uword)cols.size(); ++i)
                    {
                        ph.col(i) = phase.slice((arma::uword)side).col(cols[i]);
                        if (fl.first != 0)
                            ph.col(i) += offset;
                    }

                    const arma::mat &pos = one_group ? (side == 0 ? R : T) : (side == 0 ? RL : TL);
                    arma::mat v;
                    quadriga_lib::sos_val<double>(freq[t], ph, pos, v, nullptr, nullptr, nullptr, nullptr, 1.0, false, 0);

                    arma::mat &out = (side == 0) ? val : b;
                    for (arma::uword i = 0; i < (arma::uword)cols.size(); ++i)
                    {
                        const double *pv = v.colptr(i);
                        double *po = out.colptr(cols[i]);
                        for (arma::uword j = 0; j < n_fl; ++j)
                            po[one_group ? j : L.at(j)] = pv[j];
                    }
                }
        }

        // Combination of the two devices
        if (dual_mobility)
        {
            const double rsqrt2 = std::sqrt(0.5);
            for (arma::uword g = 0; g < n_gen; ++g)
            {
                const double *pb = b.colptr(g);
                double *pv = val.colptr(g);
                if (gen[g].mirror != g)
                    for (arma::uword l = 0; l < n_link; ++l)
                        pv[l] = (pv[l] + pb[l]) * rsqrt2;
                else
                {
                    const arma::vec &A = acf[tab[g]], &D = dist[tab[g]];
                    for (arma::uword l = 0; l < n_link; ++l)
                    {
                        const double dx = R.at(l, 0) - T.at(l, 0), dy = R.at(l, 1) - T.at(l, 1), dz = R.at(l, 2) - T.at(l, 2);
                        const double c = qd_sos_acf_interp(A.memptr(), D.memptr(), A.n_elem, std::sqrt(dx * dx + dy * dy + dz * dz));
                        pv[l] = (pv[l] + pb[l]) / std::sqrt(2.0 + 2.0 * c);
                    }
                }
            }
        }

        // Uniform generators
        const double rsqrt2 = std::sqrt(0.5);
        for (arma::uword g = 0; g < n_gen; ++g)
            if (gen[g].uniform)
            {
                double *pv = val.colptr(g);
                for (arma::uword l = 0; l < n_link; ++l)
                    pv[l] = 0.5 * std::erfc(-pv[l] * rsqrt2);
            }
    }
}

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# lsf_draw
Spatially correlated normalized random values of the large-scale generators at a set of links

- Evaluates the 11 large-scale fading generators of [[lsf_init_phases]] at the links: column 0 DS, 1 KF, 2 SF,
  3 ASD, 4 ASA, 5 ESD, 6 ESA, 7 XPR, 8 absolute TOA (Normal(0,1)), 9 ground permittivity, 10 UMa environment height
  hE (Uniform(0,1)); each generator with the sum-of-sinusoids frequencies of `preset` at its own decorrelation
  distance (a distance of 0 is used as 0.1 m)
- `scenpar` is completed and validated by [[scenpar_validate]] as in [[lsf_init_phases]]; its messages pass
  through unchanged
- Single mobility (`dual_mobility = false`): column g is the field of generator g at the Rx position,
  `x_g = s_g(rx)` with `s_g(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )` and the phases of slice 0;
  the values do not depend on `tx_pos`
- Dual mobility: the Tx is a second device. Its side uses slice 1, i.e. the field of the mirror generator `m(g)`
  at the Tx position (ASD and ASA, ESD and ESA exchanged). Self-mirrored generators (DS, KF, SF, XPR, TOA, 9, 10)
  share one map: `x_g = ( s_g(rx) + s_g(tx) ) / sqrt( 2 + 2·A_g(D) )`, with the Tx-Rx distance `D` and the ACF
  `A_g` of the preset, linearly interpolated. The pairs are two maps: `x_g = ( s_g(rx) + s_m(tx) ) / sqrt(2)`.
  Columns 3 to 6 thus stay Normal(0,1), and ASD/ASA and ESD/ESA have the correlation
  `( A_g(D) + A_m(D) ) / 2` at the Tx-Rx distance, 1 at `D = 0` (then the two columns are equal)
- Swap reciprocity in dual mobility: swapping `tx_pos` and `rx_pos` leaves columns 0, 1, 2, 7, 8, 9, 10
  unchanged and swaps column 3 with 4 and column 5 with 6, exactly (the same evaluations in swapped roles), also
  for unequal decorrelation distances and with bias-removed phases, when both calls use the same phase cube
- Uniform columns 9 and 10: `0.5 · erfc( -x / sqrt(2) )` of the Normal value
- `horizontal = true`: the positions are projected on the x-y plane (z = 0) and the values do not depend on z;
  `false`: 3D positions. `D` is measured in the same space
- Floors: links with different floor IDs see different fields, links with equal IDs share one field. Floor `q`
  adds the offsets `2·pi · k · n / N + psi_b(n)` to the phases of all generators on both sides, with
  `N = n_coeff`, `k = q mod N`, `b = q div N`, `n = 0 ... N-1`, `psi_0 = 0` and `psi_b(n) = 2·pi · u_n` for
  `b >= 1`, where `u_n` are the uniform numbers of a `std::mt19937_64` seeded with `b` (the upper 53 bits of each
  raw output times 2^-53). Floor 0 and `floor_id = nullptr` give the same values
- Fields of two floors are uncorrelated at equal horizontal positions (TR 38.901 Table 7.5-6 note 3): exactly for
  two IDs in the same block of `N` IDs, with a random residual of standard deviation `1/sqrt(2N)` for IDs in
  different blocks (0.044 for 256 sinusoids). At other horizontal lags the cross-correlation of two floors is not
  zero (up to about 0.16 for 256 sinusoids), the same level as for random offsets. Mirror relations and swap
  reciprocity hold per floor
- `floor_id` is the caller's floor index of the UT: for the UT heights `hUT = 3·(n_fl - 1) + 1.5` of TR 38.901
  Table 7.2-1 it is `n_fl - 1`; outdoor UTs use 0
- The value of a link does not depend on the other links of the call
- Precision: the sinusoids are evaluated by [[sos_val]] with the AVX2 kernel where available (sine and cosine in
  single precision, accumulation in double); the values deviate from a double-precision evaluation by less than
  `4e-7 · sqrt(2 · n_coeff)` (9e-6 for 256 sinusoids)
- `preset` and `horizontal` must be those of [[lsf_init_phases]]. A wrong preset with the same number of sinusoids
  (e.g. `Exp256` for `Comb256`) cannot be detected
- `n_link = 0` returns `rnd` with size `[0, 11]`
- Validation order: positions (3 columns, rows of `tx_pos`, finite values), `preset`, `scenpar`, size of `phase`,
  finite `phase`, length of `floor_id`

## Declaration:
```
void quadriga_lib::lsf_draw(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::cube &phase,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    arma::mat &rnd,
    const arma::uvec *floor_id = nullptr,
    bool dual_mobility = false,
    const std::string &preset = "Comb256",
    bool horizontal = true);
```

## Inputs:
- **`scenpar`** — Scenario parameters as returned by [[conf_file_read]] or [[scenpar_validate]]; missing keys
  have their default value; only the 10 decorrelation distances are used
- **`phase`** — Phases in [rad] from [[lsf_init_phases]]; slice 0 = Rx side, slice 1 = Tx side; must be finite;
  `[n_coeff, 11, 2]`, `n_coeff` of `preset`
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; not used in single mobility (but validated);
  `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`floor_id`** *(optional)* — Floor index per link, 0-based; omitted = all links on floor 0; `[n_link]`
- **`dual_mobility`** *(optional)* — `true`: Tx and Rx both move, the Tx side is evaluated as a second device
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [[sos_preset]]; must match [[lsf_init_phases]]
- **`horizontal`** *(optional)* — `true`: 2D correlation in the x-y plane (TR 38.901 behaviour); `false`: 3D
  correlation (legacy QuaDRiGa behaviour); must match [[lsf_init_phases]]

## Output:
- **`rnd`** — Random values; columns 0 to 8 Normal(0,1), columns 9 and 10 Uniform(0,1); `[n_link, 11]`

## See also:
- [[lsf_init_phases]] (draws the phases)
- [[path_loss]] (column 10 as `rnd_he`)
- [[lsp_generate]] (the large-scale parameters from `rnd`)
- [[sos_val]] (evaluation of a sum-of-sinusoids generator)
MD!*/

void quadriga_lib::lsf_draw(const std::unordered_map<std::string, double> &scenpar,
                            const arma::cube &phase,
                            const arma::mat &tx_pos,
                            const arma::mat &rx_pos,
                            arma::mat &rnd,
                            const arma::uvec *floor_id,
                            bool dual_mobility,
                            const std::string &preset,
                            bool horizontal)
{
    // Validation, before any output is touched
    const arma::uword n_link = rx_pos.n_rows;
    if (rx_pos.n_cols != 3)
        throw std::invalid_argument("Input 'rx_pos' must have 3 columns.");
    if (tx_pos.n_cols != 3)
        throw std::invalid_argument("Input 'tx_pos' must have 3 columns.");
    if (tx_pos.n_rows != 1 && tx_pos.n_rows != n_link)
        throw std::invalid_argument("Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'.");
    if (!rx_pos.is_finite())
        throw std::invalid_argument("Input 'rx_pos' must be finite.");
    if (!tx_pos.is_finite())
        throw std::invalid_argument("Input 'tx_pos' must be finite.");

    int type = 0;
    arma::uword n_coeff = 0;
    if (!qd_sos_preset_lookup(preset, type, n_coeff))
        throw std::invalid_argument("Input 'preset' is not a known preset.");

    std::unordered_map<std::string, double> sp = scenpar;
    quadriga_lib::scenpar_validate(sp, "legacy", nullptr, nullptr, true);
    const std::vector<qd_gbsm_gen> gen = qd_gbsm_lsf_generators(sp);
    const arma::uword n_gen = (arma::uword)gen.size();

    if (phase.n_rows != n_coeff || phase.n_cols != n_gen || phase.n_slices != 2)
        throw std::invalid_argument("Input 'phase' must have size [" + std::to_string(n_coeff) + ", " + std::to_string(n_gen) +
                                    ", 2] for preset '" + preset + "'.");
    if (!phase.is_finite())
        throw std::invalid_argument("Input 'phase' must be finite.");
    if (floor_id != nullptr && floor_id->n_elem != n_link)
        throw std::invalid_argument("Number of elements in 'floor_id' does not match number of rows in 'rx_pos'.");

    if (n_link == 0)
    {
        if (rnd.n_rows != 0 || rnd.n_cols != n_gen)
            rnd.set_size(0, n_gen);
        return;
    }

    std::vector<arma::mat> freq;
    std::vector<arma::vec> acf, dist;
    std::vector<arma::uword> tab;
    qd_gbsm_freq(preset, gen, freq, acf, dist, tab);

    arma::mat R, T;
    qd_gbsm_positions(rx_pos, tx_pos, horizontal, R, T);

    qd_gbsm_eval(gen, freq, acf, dist, tab, phase, R, T, floor_id, dual_mobility, rnd);
}
