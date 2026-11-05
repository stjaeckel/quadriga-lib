// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"
#include "quadriga_sos.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Random sets of the geometry-based stochastic model: the spatially correlated sum-of-sinusoids generators, their
// phases (gbsm_init_phases) and their evaluation at the links (gbsm_draw). The machinery in the anonymous namespace
// is generic over a generator table (qd_gbsm_gen_lsf, qd_gbsm_gen_cluster): mirroring, floors, 2D / 3D, dual
// mobility and bias removal work the same for every set type. The column layouts are documented in the set table of
// the section description (qd_scenpar.cpp).
//
// Phase cube of a set, [n_coeff, n_gen, 2]: slice 0 = Rx side, slice 1 = Tx side, slice1.col(g) = slice0.col(m(g)):
// at the Tx position, generator g evaluates the field of its mirror generator m(g).
//
// Units: positions in [m], rows {x, y, z}; frequencies of the sum-of-sinusoids tables in [1/m]; decorrelation
// distances in [m], 0 used as 0.1 m (as in sos_randn).
//
// Seeds: the "lsf" set uses the seed directly; the "cluster" sets use SplitMix64 of the seed and a set tag (1 for
// the shared set, 2 + k for frequency k; see derive_seed), so the sets are independent streams, distinct from the
// "lsf" phases; seed -1 passes on (system random device). No "cluster" generator is bias-corrected; NumClusters = 1
// (N = 0) gives empty "cluster" sets without a sum-of-sinusoids call.
//
// A set never changes size: new random quantities come in new sets (a new set type or set id). sos_init_phases fills
// column g from one stream independently of n_gen, and gbsm_draw evaluates each column with its own phases only, so
// the values of a column depend neither on the other columns nor on the other links.

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

    // The 11 generators of the "lsf" set from a completed scenpar; column layout in the set table of the section
    // description (qd_scenpar.cpp)
    // - 'sp' must hold all keys (output of scenpar_validate); a missing key throws std::out_of_range
    std::vector<qd_gbsm_gen> qd_gbsm_gen_lsf(const std::unordered_map<std::string, double> &sp)
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

    // Generators of the "cluster" sets from a completed scenpar (column layout in the set table): set_id = -1 shared
    // set (9·N), >= 0 per-frequency set (N); N = NumClusters - 1, empty for N = 0
    std::vector<qd_gbsm_gen> qd_gbsm_gen_cluster(const std::unordered_map<std::string, double> &sp, arma::sword set_id)
    {
        const arma::uword mirror_b[9] = {0, 2, 1, 4, 3, 6, 5, 8, 7};
        const arma::uword N = (arma::uword)sp.at("NumClusters") - 1;
        const double lambda = sp.at("SC_lambda");
        const bool shared = set_id < 0;

        std::vector<qd_gbsm_gen> gen(shared ? 9 * N : N);
        for (arma::uword c = 0; c < (arma::uword)gen.size(); ++c)
        {
            const arma::uword b = c / N, n = c % N;
            gen[c].lambda = (lambda > 0.0) ? lambda : 0.1;
            gen[c].uniform = shared && b == 0;
            gen[c].mirror = shared ? mirror_b[b] * N + n : c;
            gen[c].bias = false;
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

    // Normalized values of a phase set at a set of links
    // - gen, freq, acf, dist, tab: generator table (qd_gbsm_gen_lsf or qd_gbsm_gen_cluster) and frequency tables
    //   (qd_gbsm_freq)
    // - phase [n_coeff, n_gen, 2]: slice 0 Rx side, slice 1 Tx side (mirror of slice 0)
    // - R, T [n_link, 3], n_link >= 1: positions from qd_gbsm_positions (T unused without dual mobility)
    // - floor_id: [n_link] or nullptr (all on floor 0)
    // - val [n_link, n_gen]: column g Normal(0,1), or Uniform(0,1) (map 0.5·erfc(-x/sqrt(2))) if gen[g].uniform
    // - Rx side of generator g: field of g at R (phases slice0.col(g), table tab[g]); Tx side (dual mobility): field
    //   of the mirror generator m(g) at T (phases slice1.col(g) = slice0.col(m(g)), table tab[m(g)])
    // - Dual mobility: self-mirrored generators (a + b) / sqrt(2 + 2·A_g(D)), pairs (a + b) / sqrt(2), with the
    //   Tx-Rx distance D and the ACF A_g of the table of g
    // - Floors: the phase offsets of the link's floor are added to all phases of both slices, so in dual mobility
    //   both devices of a link are evaluated on the link's floor
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

        // Phase offsets of floor q, [n_coeff]: 2·pi·(q mod N)·n/N + psi_(q div N)(n)
        // - N = n_coeff; psi_0 = 0; psi_b(n) = 2·pi·u_n for b >= 1, with u_n the n-th value of qd_sos_uniform of the
        //   stream qd_sos_rng(b)
        // - Two floors in the same block of N IDs have the cross-correlation (1/N)·sum_n cos(Delta_q(n) - Delta_q'(n))
        //   = 0 at zero horizontal lag; floors in different blocks a random one with standard deviation 1/sqrt(2N)
        // - Phase offsets instead of a coordinate offset per floor: they work in 2D and 3D mode, keep the mirror
        //   relations, need no large coordinates (which would push small decorrelation distances towards the
        //   precision limit of the kernels) and give exactly zero correlation at zero horizontal lag (UTs stacked
        //   above each other, the most relevant case) within a block of N floor IDs
        // - At other horizontal lags the cross-correlation of two floors is not zero (up to about 0.16 for 256
        //   sinusoids), the same level as for random offsets; mirror relations and swap reciprocity hold per floor
        // - Dual mobility: the offsets are added on both sides, so a Tx device on another floor than its Rx (a D2D
        //   link between two floors) is evaluated on the floor of the Rx
        auto floor_offset = [n_coeff](arma::uword q)
        {
            const double two_pi = 6.283185307179586;
            const arma::uword k = q % n_coeff, blk = q / n_coeff;
            arma::vec offset(n_coeff);
            for (arma::uword n = 0; n < n_coeff; ++n) // k·n is reduced modulo N, which keeps the argument below 2·pi
                offset.at(n) = two_pi * (double)((k * n) % n_coeff) / (double)n_coeff;
            if (blk > 0)
            {
                std::mt19937_64 rng = qd_sos_rng((arma::sword)blk);
                for (arma::uword n = 0; n < n_coeff; ++n)
                    offset.at(n) += two_pi * qd_sos_uniform(rng);
            }
            return offset;
        };

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
                offset = floor_offset(fl.first);

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
        // - Each side is evaluated with the table of its own field (instead of the two-device modes of sos_val), so
        //   swapping Tx and Rx repeats the same sos_val calls in swapped roles: swap reciprocity holds bit for bit,
        //   also for unequal decorrelation distances and with bias-removed phases
        // - The pairs need no further coupling: (f_g(R) + f_m(T))/sqrt(2) and (f_m(R) + f_g(T))/sqrt(2) already have
        //   the correlation (A_g(D) + A_m(D))/2. QuaDRiGa's gen_lsf_parameters multiplied the pair once more by
        //   sqrtm([1, A_g; A_m, 1]), which raises the variances above 1 (up to 2 at small D); that coupling is not
        //   taken, so that columns 3 to 6 stay Normal(0,1)
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
# gbsm_init_phases
Draw the random phases of one generator set of a BS site and scenario

- Draws the phases of one random set (set table in the section description) for [[gbsm_draw]]; store `phase` or the
  seed to evaluate the same fields again along tracks or on maps
- The phases are a fixed function of `seed`, `set_type` and `set_id`, equal on all platforms (bias-removed `"lsf"`
  phases within about 2e-14 rad); the sets are independent; each scenario of a site needs its own seed
- `"lsf"` bias removal (`remove_bias`, at least 3 rows in `rx_pos`): [[sos_remove_bias]] sets the sample mean of
  columns 0 to 8 over the positions to 0 (with `tx_pos` in dual mobility; 2D mode: one floor), removing variance
- TR 38.901 clause 7.4.4 literally: preset `Exp256` without bias removal; `NumClusters = 1`: empty `"cluster"` sets

## Declaration:
```
void quadriga_lib::gbsm_init_phases(
    const std::unordered_map<std::string, double> &scenpar,
    const std::string &set_type,
    arma::cube &phase,
    arma::sword seed = -1,
    arma::sword set_id = -1,
    const std::string &preset = "Comb256",
    const arma::mat *tx_pos = nullptr,
    const arma::mat *rx_pos = nullptr,
    bool dual_mobility = false,
    bool remove_bias = true,
    bool horizontal = true);
```

## Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances and `NumClusters` are used
- **`set_type`** — `"lsf"` or `"cluster"`; `"ray"` is not available yet
- **`seed`** *(optional)* — Seed of the site and scenario, >= -1; -1 = system random device, new phases on each call
- **`set_id`** *(optional)* — `"cluster"`: `-1` = shared set, `k >= 0` = set of frequency `k`; ignored for `"lsf"`
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [[sos_preset]]; sets `n_coeff`
- **`tx_pos`**, **`rx_pos`** *(optional)* — `"lsf"` only: positions in [m] for the bias removal, as in [[gbsm_draw]]
- **`dual_mobility`**, **`remove_bias`**, **`horizontal`** *(optional)* — `"lsf"`: Tx and Rx both move; remove the
  sample mean; 2D bias removal (TR 38.901), `false`: 3D (legacy QuaDRiGa)

## Output:
- **`phase`** — Phases in [rad], in [-pi, pi); slice 0 = Rx side, slice 1 = Tx side; `[n_coeff, n_gen, 2]`

## See also:
- [[gbsm_draw]] (evaluates the phases at the links)
- [[sos_remove_bias]] (the bias removal of one generator)
MD!*/

void quadriga_lib::gbsm_init_phases(const std::unordered_map<std::string, double> &scenpar,
                                    const std::string &set_type,
                                    arma::cube &phase,
                                    arma::sword seed,
                                    arma::sword set_id,
                                    const std::string &preset,
                                    const arma::mat *tx_pos,
                                    const arma::mat *rx_pos,
                                    bool dual_mobility,
                                    bool remove_bias,
                                    bool horizontal)
{
    // Validation, before any output is touched
    const bool lsf = set_type == "lsf";
    if (!lsf && set_type != "cluster" && set_type != "ray")
        throw std::invalid_argument("Input 'set_type' must be 'lsf', 'cluster' or 'ray'.");
    if (seed < -1)
        throw std::invalid_argument("Input 'seed' must be >= -1.");
    if (!lsf && set_id < -1)
        throw std::invalid_argument("Input 'set_id' must be >= -1.");

    const arma::uword n_coeff = qd_gbsm_n_coeff(preset);
    if (n_coeff == 0)
        throw std::invalid_argument("Input 'preset' is not a known preset.");

    std::unordered_map<std::string, double> sp = scenpar;
    quadriga_lib::scenpar_validate(sp, nullptr, nullptr, true);

    if (!lsf && (tx_pos != nullptr || rx_pos != nullptr))
        throw std::invalid_argument("Inputs 'tx_pos' and 'rx_pos' are used for set type 'lsf' only.");
    if (set_type == "ray")
        throw std::invalid_argument("Set type 'ray' is not available yet.");

    qd_gbsm_check_links(tx_pos, rx_pos);
    const arma::uword n_link = (rx_pos != nullptr) ? rx_pos->n_rows : 0;
    const bool do_bias = lsf && remove_bias && n_link > 2;
    if (do_bias && dual_mobility && tx_pos == nullptr)
        throw std::invalid_argument("Input 'tx_pos' is required for the bias removal with dual mobility.");

    const std::vector<qd_gbsm_gen> gen = lsf ? qd_gbsm_gen_lsf(sp) : qd_gbsm_gen_cluster(sp, set_id);
    const arma::uword n_gen = qd_gbsm_n_gen(sp, set_type, set_id);
    if (phase.n_rows != n_coeff || phase.n_cols != n_gen || phase.n_slices != 2)
        phase.set_size(n_coeff, n_gen, 2);
    if (n_gen == 0) // N = 0: empty "cluster" set, sos_init_phases rejects zero generators
        return;

    // Seed of a "cluster" set: SplitMix64 of the seed and the set tag; -1 passes on
    auto derive_seed = [](arma::sword seed, arma::sword set_id) -> arma::sword
    {
        if (seed == -1)
            return -1;
        auto splitmix64 = [](std::uint64_t x)
        {
            std::uint64_t z = x + 0x9E3779B97F4A7C15ULL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        };
        const std::uint64_t tag = (set_id < 0) ? 1ULL : 2ULL + (std::uint64_t)set_id;
        return (arma::sword)(splitmix64(splitmix64((std::uint64_t)seed) ^ tag) >> 1);
    };

    // Raw phases of the Rx side
    arma::mat P0;
    quadriga_lib::sos_init_phases<double>(n_coeff, n_gen, P0, lsf ? seed : derive_seed(seed, set_id));

    // "lsf" bias removal over all positions where the fields are evaluated, before the mirroring
    // - Each Normal field is corrected once, with its own frequencies, over the union of the positions where it is
    //   evaluated; the Tx side is then copied from the corrected phases. QuaDRiGa's init_sos corrected before the
    //   mirroring, so the ASD/ASA and ESD/ESA phases carried a correction computed with the partner's frequencies,
    //   which left a non-zero drop mean whenever the two decorrelation distances differ (also in single mobility)
    // - Consequences for gbsm_draw at the same positions: in single mobility the link mean of columns 0 to 8 is 0; in
    //   dual mobility the link mean of DS, KF, SF, XPR and TOA is 0 when all links have the same Tx-Rx distance (the
    //   normalisation is then the same for all links), while ASD/ASA and ESD/ESA have a zero link mean only
    //   approximately (their two devices see different fields, each corrected over the union of the positions)
    // - 2D mode: one offset per generator cannot zero the means of several floor fields at once, so all positions
    //   are corrected as floor 0; in 3D mode the correction covers all positions as one 3D field
    if (do_bias)
    {
        std::vector<arma::mat> freq;
        std::vector<arma::vec> acf, dist;
        std::vector<arma::uword> tab;
        qd_gbsm_freq(preset, gen, freq, acf, dist, tab);

        arma::mat R, T;
        qd_gbsm_positions(*rx_pos, (tx_pos != nullptr) ? *tx_pos : *rx_pos, horizontal, R, T);
        qd_gbsm_remove_bias(gen, freq, tab, P0, dual_mobility ? arma::mat(arma::join_cols(R, T)) : R);
    }

    // Output: slice 0 = Rx side, slice 1 = mirror of the Rx side
    for (arma::uword g = 0; g < n_gen; ++g)
    {
        phase.slice(0).col(g) = P0.col(g);
        phase.slice(1).col(g) = P0.col(gen[g].mirror);
    }
}

/*!MD
# gbsm_draw
Spatially correlated normalized random values of one generator set at a set of links

- Evaluates the phases of [[gbsm_init_phases]] at the links, one column per generator; columns, distributions and
  decorrelation distances: see the set table in the section description. The value of a link does not depend on the
  other links of the call; in single mobility it does not depend on `tx_pos`
- Dual mobility: the Tx is a second device on the same maps, its side of each generator uses the mirror generator.
  Every column keeps its distribution; swapping `tx_pos` and `rx_pos` exchanges the columns of each mirror pair and
  keeps the self-mirrored columns, exactly
- Floors (2D mode): links with equal `floor_id` share one field; fields of different floors are uncorrelated at
  equal horizontal positions, exactly within a block of `n_coeff` consecutive IDs, otherwise up to a random residual
  of standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both devices of a link are on the link's floor
- The values are within `4e-7 · sqrt(2·n_coeff)` of a double-precision evaluation (AVX2 kernel of [[sos_val]])
- A set never changes size: new random quantities come in new sets (a new `set_type` or `set_id`). A column depends
  neither on the other columns nor on the other links
- `rnd` may share memory with `tx_pos` or `rx_pos`; an empty set (`NumClusters = 1`) accepts any empty `phase`

## Declaration:
```
void quadriga_lib::gbsm_draw(
    const std::unordered_map<std::string, double> &scenpar,
    const std::string &set_type,
    const arma::cube &phase,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    arma::mat &rnd,
    arma::sword set_id = -1,
    const std::string &preset = "Comb256",
    const arma::uvec *floor_id = nullptr,
    const arma::vec *center_frequency = nullptr,
    bool dual_mobility = false,
    bool horizontal = true);
```

## Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances and `NumClusters` are used
- **`set_type`** — Set type of [[gbsm_init_phases]]
- **`phase`** — Phases from [[gbsm_init_phases]], finite; `[n_coeff, n_gen, 2]`
- **`tx_pos`** — Tx positions in [m]; used in dual mobility, always validated; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`, `n_link = 0` allowed
- **`set_id`** *(optional)* — Set of [[gbsm_init_phases]]; ignored for `"lsf"`
- **`preset`** *(optional)* — Preset of [[gbsm_init_phases]]; another preset with the same `n_coeff` is not detected
- **`floor_id`** *(optional)* — Floor index of the UT per link, 0-based (`n_fl - 1` for the UT heights of TR 38.901
  Table 7.2-1, 0 for outdoor UTs); omitted = floor 0; 2D mode only; `[n_link]`
- **`center_frequency`** *(optional)* — Carrier frequencies in [Hz]; ignored and not checked for the `"lsf"` and
  `"cluster"` sets; `[n_freq]`
- **`dual_mobility`** *(optional)* — `true`: Tx and Rx both move
- **`horizontal`** *(optional)* — `true`: 2D mode (TR 38.901); `false`: 3D mode (legacy QuaDRiGa)

## Output:
- **`rnd`** — Random values; `[n_link, n_gen]`

## See also:
- [[lsp_generate]] (the large-scale parameters from the `"lsf"` set)
- [[path_loss]] (column 10 of the `"lsf"` set as `rnd_he`)
- [[ssf_clusters]] (the clusters from the `"cluster"` sets)
MD!*/

void quadriga_lib::gbsm_draw(const std::unordered_map<std::string, double> &scenpar,
                             const std::string &set_type,
                             const arma::cube &phase,
                             const arma::mat &tx_pos,
                             const arma::mat &rx_pos,
                             arma::mat &rnd,
                             arma::sword set_id,
                             const std::string &preset,
                             const arma::uvec *floor_id,
                             const arma::vec *center_frequency,
                             bool dual_mobility,
                             bool horizontal)
{
    // Validation, before any output is touched; 'center_frequency' is used by the shared "ray" set only
    (void)center_frequency;
    const bool lsf = set_type == "lsf";
    if (!lsf && set_type != "cluster" && set_type != "ray")
        throw std::invalid_argument("Input 'set_type' must be 'lsf', 'cluster' or 'ray'.");
    const arma::uword n_link = rx_pos.n_rows;
    qd_gbsm_check_links(&tx_pos, &rx_pos);
    if (!lsf && set_id < -1)
        throw std::invalid_argument("Input 'set_id' must be >= -1.");
    const arma::uword n_coeff = qd_gbsm_n_coeff(preset);
    if (n_coeff == 0)
        throw std::invalid_argument("Input 'preset' is not a known preset.");

    std::unordered_map<std::string, double> sp = scenpar;
    quadriga_lib::scenpar_validate(sp, nullptr, nullptr, true);
    if (set_type == "ray")
        throw std::invalid_argument("Set type 'ray' is not available yet.");

    const std::vector<qd_gbsm_gen> gen = lsf ? qd_gbsm_gen_lsf(sp) : qd_gbsm_gen_cluster(sp, set_id);
    const arma::uword n_gen = qd_gbsm_n_gen(sp, set_type, set_id);

    // An empty set accepts any empty cube (MATLAB passes zeros(n_coeff, 0, 2) as 0x0x0)
    if (n_gen == 0 ? phase.n_elem != 0 : (phase.n_rows != n_coeff || phase.n_cols != n_gen || phase.n_slices != 2))
        throw std::invalid_argument("Input 'phase' must have size [" + std::to_string(n_coeff) + ", " + std::to_string(n_gen) +
                                    ", 2] for preset '" + preset + (lsf ? "'." : "' and this set."));
    if (!phase.is_finite())
        throw std::invalid_argument("Input 'phase' must be finite.");
    if (floor_id != nullptr && floor_id->n_elem != n_link)
        throw std::invalid_argument("Number of elements in 'floor_id' does not match number of rows in 'rx_pos'.");
    if (floor_id != nullptr && !horizontal)
        throw std::invalid_argument("Input 'floor_id' cannot be used with 'horizontal = false': in 3D mode the height separates the floors.");

    if (n_link == 0 || n_gen == 0)
    {
        if (rnd.n_rows != n_link || rnd.n_cols != n_gen)
            rnd.set_size(n_link, n_gen);
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
