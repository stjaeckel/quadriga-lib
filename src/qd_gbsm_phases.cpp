// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

// Phase sets of the geometry-based stochastic model: the spatially correlated sum-of-sinusoids generators, their
// phases (lsf_init_phases) and their evaluation at the links (lsf_draw). The machinery in the anonymous namespace
// is generic over a generator table (qd_gbsm_gen): further phase sets are defined by their own table and are
// evaluated with the same functions (mirroring, floors, 2D / 3D, dual mobility, bias removal).
//
// LSF generator index (column of the phase cube and of the random values of lsf_draw):
//
//   g   Generator                  Distribution   Decorrelation distance   Mirror m(g)   Bias removal
//   0   DS                         Normal         DS_lambda                0             yes
//   1   KF                         Normal         KF_lambda                1             yes
//   2   SF                         Normal         SF_lambda                2             yes
//   3   ASD                        Normal         AS_D_lambda              4             yes
//   4   ASA                        Normal         AS_A_lambda              3             yes
//   5   ESD                        Normal         ES_D_lambda              6             yes
//   6   ESA                        Normal         ES_A_lambda              5             yes
//   7   XPR                        Normal         XPR_lambda               7             yes
//   8   absolute TOA               Normal         absTOA_lambda            8             yes
//   9   ground permittivity        Uniform        SC_lambda                9             no
//  10   UMa environment height hE  Uniform        SF_lambda                10            no
//
// A decorrelation distance of 0 is used as 0.1 m (as in sos_randn).
//
// Phase cube of a phase set, [n_coeff, n_gen, 2]: slice 0 = Rx side, slice 1 = Tx side. The Tx side is the mirror of
// the Rx side, slice1.col(g) = slice0.col(m(g)): at the Tx position, generator g evaluates the field of its mirror
// generator m(g) (e.g. the ASD of a link seen from the Tx is the ASA field of the Tx as a second device).
//
// Units: positions in [m], rows {x, y, z}; frequencies of the sum-of-sinusoids tables in [1/m]; decorrelation
// distances in [m].

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
    // One generator of a phase set: decorrelation distance in [m] (0 already replaced by 0.1 m), distribution,
    // mirror partner, bias removal on/off
    struct qd_gbsm_gen
    {
        double lambda;      // Decorrelation distance in [m], > 0
        bool uniform;       // false: Normal(0,1), true: Uniform(0,1)
        arma::uword mirror; // Index of the mirror generator (Tx side); equal to the own index for self-mirrored ones
        bool bias;          // Remove the sample mean in the bias removal
    };

    // Number of sinusoids of a preset; throws "Input 'preset' is not a known preset."
    arma::uword qd_gbsm_n_coeff(const std::string &preset)
    {
        int type = 0;
        arma::uword n_coeff = 0;
        if (!qd_sos_preset_lookup(preset, type, n_coeff))
            throw std::invalid_argument("Input 'preset' is not a known preset.");
        return n_coeff;
    }

    // The 11 LSF generators (index table above) from a completed scenpar
    // - 'sp' must hold all keys (output of scenpar_validate); a missing key throws std::out_of_range
    std::vector<qd_gbsm_gen> qd_gbsm_lsf_generators(const std::unordered_map<std::string, double> &sp)
    {
        const char *keys[11] = {"DS_lambda", "KF_lambda", "SF_lambda", "AS_D_lambda", "AS_A_lambda", "ES_D_lambda",
                                "ES_A_lambda", "XPR_lambda", "absTOA_lambda", "SC_lambda", "SF_lambda"};
        const arma::uword mirror[11] = {0, 1, 2, 4, 3, 6, 5, 7, 8, 9, 10};

        std::vector<qd_gbsm_gen> gen(11);
        for (arma::uword g = 0; g < 11; ++g)
        {
            const double lambda = sp.at(keys[g]);
            gen[g].lambda = (lambda > 0.0) ? lambda : 0.1;
            gen[g].uniform = g >= 9;
            gen[g].mirror = mirror[g];
            gen[g].bias = g < 9;
        }
        return gen;
    }

    // Frequency tables, one per distinct decorrelation distance of 'gen' (in order of first appearance):
    // sos_preset(preset, freq[t], &acf[t], &dist[t], lambda_t); tab[g] = table index of generator g
    // - 'preset' must be a known preset (checked by the caller with qd_gbsm_n_coeff)
    // - Distances are grouped by exact equality: equal input values give identical tables
    void qd_gbsm_freq(const std::string &preset, const std::vector<qd_gbsm_gen> &gen,
                      std::vector<arma::mat> &freq, std::vector<arma::vec> &acf,
                      std::vector<arma::vec> &dist, std::vector<arma::uword> &tab)
    {
        freq.clear(), acf.clear(), dist.clear();
        tab.assign(gen.size(), 0);
        std::vector<double> lambda;
        for (size_t g = 0; g < gen.size(); ++g)
        {
            size_t t = 0;
            while (t < lambda.size() && lambda[t] != gen[g].lambda)
                ++t;
            if (t == lambda.size())
            {
                lambda.push_back(gen[g].lambda);
                freq.emplace_back(), acf.emplace_back(), dist.emplace_back();
                quadriga_lib::sos_preset<double>(preset, freq[t], &acf[t], &dist[t], gen[g].lambda);
            }
            tab[g] = (arma::uword)t;
        }
    }

    // Positions for the evaluation: R = rx_pos, T = tx_pos broadcast to n_link rows; z = 0 in 2D mode
    // - 'rx_pos' [n_link, 3]; 'tx_pos' [1, 3] or [n_link, 3]; R and T [n_link, 3]
    void qd_gbsm_positions(const arma::mat &rx_pos, const arma::mat &tx_pos, bool horizontal,
                           arma::mat &R, arma::mat &T)
    {
        const arma::uword n_link = rx_pos.n_rows;
        R = rx_pos;
        if (tx_pos.n_rows == n_link)
            T = tx_pos;
        else
        {
            T.set_size(n_link, 3);
            for (arma::uword c = 0; c < 3; ++c)
                T.col(c).fill(tx_pos(0, c));
        }
        if (horizontal)
            R.col(2).zeros(), T.col(2).zeros();
    }

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
    // - Floors: the offsets of qd_gbsm_floor_offset of the link's floor are added to all phases of both slices, so
    //   in dual mobility both devices of a link are evaluated on the link's floor
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
        const double rsqrt2 = std::sqrt(0.5);
        if (dual_mobility)
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
                        const double c = qd_interp_linear(A.memptr(), D.memptr(), A.n_elem, std::sqrt(dx * dx + dy * dy + dz * dz));
                        pv[l] = (pv[l] + pb[l]) / std::sqrt(2.0 + 2.0 * c);
                    }
                }
            }

        // Uniform generators
        for (arma::uword g = 0; g < n_gen; ++g)
            if (gen[g].uniform)
            {
                double *pv = val.colptr(g);
                for (arma::uword l = 0; l < n_link; ++l)
                    pv[l] = 0.5 * std::erfc(-pv[l] * rsqrt2);
            }
    }

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
- `phase` must not share memory with `tx_pos` or `rx_pos`
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

    const arma::uword n_coeff = qd_gbsm_n_coeff(preset);

    std::unordered_map<std::string, double> sp = scenpar;
    quadriga_lib::scenpar_validate(sp, "legacy", nullptr, nullptr, true);

    qd_gbsm_check_links(tx_pos, rx_pos);

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
- Dual mobility with floors: `floor_id` is one value per link and its offsets are added on both sides, i.e. both
  devices of a link are on the link's floor. A Tx device on another floor than its Rx (e.g. a D2D link between two
  floors) is evaluated on the floor of the Rx, so the same Tx device sees different fields on links to Rx devices
  on different floors
- The value of a link does not depend on the other links of the call
- Precision: the sinusoids are evaluated by [[sos_val]] with the AVX2 kernel where available (sine and cosine in
  single precision, accumulation in double); the values deviate from a double-precision evaluation by less than
  `4e-7 · sqrt(2 · n_coeff)` (9e-6 for 256 sinusoids)
- `preset` and `horizontal` must be those of [[lsf_init_phases]]. A wrong preset with the same number of sinusoids
  (e.g. `Exp256` for `Comb256`) cannot be detected
- `n_link = 0` returns `rnd` with size `[0, 11]`. `rnd` may share memory with `tx_pos` or `rx_pos` (the positions
  are copied before `rnd` is written)
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
- **`floor_id`** *(optional)* — Floor index per link, 0-based; omitted = all links on floor 0; in dual mobility
  the floor of both devices; `[n_link]`
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
    qd_gbsm_check_links(&tx_pos, &rx_pos);
    const arma::uword n_coeff = qd_gbsm_n_coeff(preset);

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
