// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"
#include "quadriga_lib_helper_functions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    // Parameters of the dual-slope model (also the LOS term of the model "nlos" and the first two slopes of
    // "tripple_slope", where E holds E1 and hE is not used)
    struct qd_pl_dual
    {
        double A1, A2, B, C, D, E, hE;
        bool hE_rand;          // Random environment height from 'rnd_he'
        bool has_sig1, has_sig2;
        double sig1, sig2;
    };

    // Additional parameters of the model "tripple_slope"
    struct qd_pl_triple
    {
        double A3, E2, hE1, hE2, sig3;
        bool has_sig3;
    };

    // Dual-slope path loss of one link at the frequency f in [GHz]
    // - d3, d2: 3D and 2D distance in [m] (clamped); hBS, hUT: raw heights in [m]
    // - hE: environment height in [m]; random_hE: no 1.1·hE clamp of the UT height
    // - Returns the loss in [dB]; 'beyond' is true beyond the breakpoint
    inline double qd_pl_dual_slope(const qd_pl_dual &p, double f, double d3, double d2, double hBS, double hUT,
                                   double hE, bool random_hE, bool &beyond)
    {
        double hb = hBS;
        if (hb < 1.1 * hE)
            hb = 1.1 * hE;
        if (hb < 0.1)
            hb = 0.1;
        double hm = hUT;
        if (!random_hE && hm < 1.1 * hE)
            hm = 1.1 * hE;
        if (hm < 0.1)
            hm = 0.1;

        const double dBP = p.E * f * (hb - hE) * (hm - hE);
        const double lf = std::log10(f);
        if (d2 <= dBP)
        {
            beyond = false;
            return p.A1 * std::log10(d3) + p.B + p.C * lf + p.D * d3;
        }
        beyond = true;
        const double dBP3 = std::sqrt(dBP * dBP + (hb - hm) * (hb - hm));
        return p.A1 * std::log10(dBP3) + p.B + p.C * lf + p.D * dBP3 + p.A2 * std::log10(d3 / dBP3);
    }

    // Zenith attenuation of the satellite model: frequencies in [GHz], attenuation in [dB]
    const double qd_pl_zenith_f[93] = {
        1, 1.99, 2.97, 3.96, 4.95, 5.92, 6.87, 7.87, 8.83, 9.77, 11.03, 12.32, 13.17, 14.44, 15.65, 16.37, 17.18, 18.03,
        18.8, 19.34, 19.91, 20.42, 20.71, 21.16, 21.49, 21.7, 22.09, 22.52, 22.99, 23.98, 24.93, 26.13, 27.38, 29.18,
        30.44, 32.18, 33.8, 36.09, 38.87, 40.59, 42.33, 44, 44.96, 47.2, 48.42, 49.42, 50.17, 50.96, 51.73, 52.62, 53.32,
        53.6, 53.75, 54.06, 54.38, 55.12, 55.73, 56.54, 56.97, 57.52, 58.01, 58.18, 58.33, 58.62, 58.99, 59.38, 59.57,
        59.84, 60.27, 60.59, 60.93, 61.22, 61.44, 61.93, 62.27, 63.34, 64.27, 65.42, 66.35, 67.19, 68.16, 69.22, 70.33,
        71.66, 73.53, 75.28, 77.48, 80.64, 83.77, 87.07, 92.1, 96.39, 100};
    const double qd_pl_zenith_att[93] = {
        0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.06, 0.06, 0.07, 0.08, 0.09, 0.11, 0.13, 0.16, 0.19,
        0.23, 0.28, 0.35, 0.4, 0.46, 0.51, 0.52, 0.52, 0.49, 0.44, 0.36, 0.3, 0.26, 0.24, 0.23, 0.24, 0.25, 0.27, 0.32,
        0.4, 0.46, 0.56, 0.68, 0.8, 1.16, 1.53, 2.18, 2.99, 4.47, 8.08, 15.49, 30.37, 39.94, 47.33, 60.66, 80.97, 102.6,
        120.55, 137.92, 144.01, 149.57, 153.68, 155.18, 156.63, 157.9, 158.81, 157.57, 156.04, 151.64, 141.33, 130.52,
        119.21, 97.42, 68, 52.74, 39.81, 17.77, 8.61, 4.2, 2.8, 2.19, 1.69, 1.42, 1.22, 1.07, 0.93, 0.85, 0.79, 0.75,
        0.74, 0.75, 0.81, 0.87, 0.97};
}

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# path_loss
Path loss and shadow-fading standard deviation of a set of links

- Evaluates one of the 10 parametric path-loss models of the QuaDRiGa conf files for every link and carrier
  frequency. `pl_model` and `plpar` are the values of `PL_model` and the `PL_` keys (without the prefix) as returned
  by [[conf_file_read]]; the TR 38.901 v19.5.0 conf files in `data/config` reproduce Table 7.4.1-1 with them
- Notation: `f = fc / 1e9` in GHz (not limited by `LSP_fc_min`); `d3`, `d2` the 3D and 2D distance in m; `hBS`,
  `hUT` the z coordinates of Tx and Rx (no ground model); `[X]` the value of key `X`, or its default if the key is
  missing (`X = v` in the lists below); `PL1(x) = A1·lg(x) + B + C·lg(f) + D·x`; `sf` the shadow-fading std
- Models, required keys and defaults:
  - `constant` (A): `pl = A`
  - `logdist` (A, B, C; SF optional): `pl = A·lg(d3) + B + C·lg(f)`; `sf = [SF]` if present
  - `logdist_simple` (A, B): `pl = A·lg(d3) + B`
  - `winner_los` (A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2): `hb = max(hBS, 1.5)`,
    `hm = max(hUT, 1.5)` of each link; `G = B1 + C1·lg f + D1·lg hb + E1·lg hm`,
    `H = B2 + C2·lg f + D2·lg hb + E2·lg hm`, breakpoint `bp = 10^((H - G)/(A1 - A2))`;
    `d2 <= bp`: `pl = A1·lg d2 + G + F1·hm`, `sf = sig1`; else `pl = A2·lg d2 + H + F2·hm`, `sf = sig2`
    (continuous at `bp` for `F1 = F2`); `A1` and `A2` must differ
  - `winner_nlos` (A1..A3, Ah1..Ah3, B1..B3, C1..C3, D1..D3, E1..E3, F1..F3): `hb`, `hm` as `winner_los`;
    `L_j = (A_j + Ah_j·lg hb)·lg d2 + B_j + C_j·lg f + D_j·lg hb + E_j·lg hm + F_j·hm`; `pl = L_1` for `f < 1.5`,
    `L_2` for `1.5 <= f < 2`, `L_3` for `f >= 2`
  - `winner_pathloss` (A, B, C, X; SF optional): `pl = A·lg d3 + B + C·lg(f/5) + X`; `sf = [SF]` if present
  - `dual_slope` (A1 or A, B, E; A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, sig1, sig2 optional):
    `hb = max(hBS, 1.1·hE, 0.1)`, `hm = max(hUT, 1.1·hE, 0.1)`; `dBP = E·f·(hb - hE)·(hm - hE)`;
    `d2 <= dBP`: `pl = PL1(d3)`, `sf = [sig1]`; `d2 > dBP`: `pl = PL1(dBP3) + A2·lg(d3/dBP3)`, `sf = [sig2]`
    (each only if present), with the 3D breakpoint distance `dBP3 = sqrt(dBP² + (hb - hm)²)`. This is Table 7.4.1-1
    for UMa, UMi, InH and InF; for RMa and SMa LOS the TR uses the 2D breakpoint distance in PL2, which differs by
    at most 0.034 dB over the TR ranges
  - `tripple_slope` (A1 or A, B, E1 or E; A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0, hE2 = 0, sig1..sig3
    optional): `hb = max(hBS, 1.1·hE1, 1.1·hE2, 0.1)`, `hm` the same; `dBPj = Ej·f·(hb - hEj)·(hm - hEj)`,
    `dBP1 = min(dBP1, dBP2)`, `dBPj3 = sqrt(dBPj² + (hb - hm)²)`; `LBP1 = PL1(dBP13)`,
    `LBP2 = LBP1 + A2·lg(dBP23/dBP13)`; `d2 <= dBP1`: `PL1(d3)`, `sf = [sig1]`; `dBP1 < d2 <= dBP2`:
    `LBP1 + A2·lg(d3/dBP13)`, `sf = [sig2]`; `d2 > dBP2`: `LBP2 + A3·lg(d3/dBP23)`, `sf = [sig3]`
  - `nlos` (An, Bn; the `dual_slope` keys if `LOSmax != 0`; Cn, Dn, D1n, D2n, D3n, En, E1n, E2n, E3n, Fn, G1n = 0,
    G2n = 1, LOSmax = 1, sig optional; Ax, Bx, Cx = 0): `hb = max(hBS, 0.1)`, `hm = max(hUT, 0.1)`;
    `PLn = An·lg d3 + Bn + Cn·lg f + Dn·lg hb + D1n·lg(hb)/hb + D2n·lg(hb)/hb² + D3n·hb + En·lg hm +
    E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm + Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²`; `pl = PLn`;
    if `[LOSmax] != 0`: `pl = max(pl, PL_LOS)` with the `dual_slope` loss of the same parameters (incl. the
    random hE); if `Ax`, `Bx` or `Cx` is present: `pl = max(pl, Ax·lg d3 + Bx + Cx·lg f)` (e.g. the InF-SL term of
    InF-DL); `sf = [sig]` if present
  - `satellite` (A = 20, B = 32.45, C = 20, D = 0, usePLa = 1): `sa = min(hBS / d3, 1)` with the absolute Tx height;
    `pl = +Inf` if `sa <= 0`; else `pl = A·lg d3 + B + C·lg f + D·lg(asin(sa)) + PLa` with `PLa = Z(f) / sa` if
    `usePLa != 0`, else 0; `Z(f)` is the zenith attenuation by atmospheric gases (93 values from 1 to 100 GHz,
    linear interpolation, constant beyond both ends)
  - empty `pl_model`: `pl = 0`
- New keys of the TR 38.901 v19 files: `hE_rand` (non-zero: random environment height of UMa), `LOSmax` (0: no
  maximum with the LOS term in `nlos`), `Ax`, `Bx`, `Cx` (additional term in the maximum of `nlos`)
- Random environment height (TR 38.901 Table 7.4.1-1, note 1), in `dual_slope` and the LOS term of `nlos` when
  `[hE_rand] != 0` and `rnd_he` is given, per link with `u = rnd_he(l)`, `d = d2`, raw `hUT`:
  `g = 0` for `d <= 18`, else `1.25·(d/100)³·exp(-d/150)`; `C' = 0` for `hUT < 13`, `((hUT - 13)/10)^1.5` for
  `hUT <= 23`, else 1; `C = C'·g`, `p1 = 1/(1 + C)`; `n` = size of the set {12, 15, ..., hUT - 1.5};
  `hE = 1` if `C = 0`, `n = 0`, `u < p1` or `p1` rounds to 1 (C below 1.1e-16, e.g. beyond about 7.5 km at
  `hUT = 20`), else `hE = 12 + 3·min(floor((u - p1)/(1 - p1)·n), n - 1)`.
  The UT height then gets no 1.1·hE clamp (`hm = max(hUT, 0.1)`), since `hUT - hE >= 1.5` holds for the TR set.
  Without `rnd_he` the fixed `[hE]` is used. `rnd_he` is column 10 of [[lsf_draw]], a spatially consistent
  uniform value: along a track hE switches between its discrete values where `u` crosses a threshold, and the
  loss beyond the breakpoint steps there; evaluate `rnd_he` once per track segment to avoid this
- Distances: `d3`, `d2` are the geometric distances (TR 38.901 eq. 7.4-1: for an indoor UT `d3D-out + d3D-in`).
  If `d2d_in` is given (legacy mode), a link with `d2d_in(l) > 0` uses the outdoor distance instead:
  `din = min(d2d_in, d2)`, `d3in = d3·din/d2` (`d3` if `d2 = 0`), limited to `d3`; if `d3 - d3in < 10`:
  `d3in = d3 - 10` (minimum outdoor distance 10 m), then `max(d3in, 0)`; `p = d3in/d3` (0 if `d3 = 0`);
  `d3 = d3 - d3in`, `d2 = d2·(1 - p)`. A link with `d2d_in(l) = 0` is an outdoor link and keeps the geometric
  distances (no 10 m rule). Then `d3 = max(d3, 0.1)`, `d2 = max(d2, 0.1)`. The O2I penetration loss is not added;
  add the output of [[o2i_loss]]
- Default shadow-fading std, where a model does not set it: the SF std of the LSP formula of [[lsp_generate]]
  with the geometric distance, `sf = max(0, max(0, SF_sigma + SF_delta·lg(SF_omega + f')) + SF_kappa·lg(d) +
  SF_tau·lg(h) + SF_beta·lg(atan(h/d)))` with `d = max(d2D, 1e-4)`, `h = max(hBS, 1e-4)`,
  `f' = max(f, LSP_fc_min)`; for scenario files without such terms this is `SF_sigma`. Passing `sf_sigma` to
  [[lsp_generate]] is then neutral. With `sf_sigma` requested, a frequency term with `SF_delta != 0` requires
  `SF_omega + f' > 0`
- Validation: positions, `fc`, the lengths of `d2d_in` and `rnd_he`, their values, the model name, the
  parameters of the model in the order the model reads them (a required key that is missing throws
  `Path-loss model '<model>' requires the parameter '<key>'.`, naming `A1` for "A1 or A" and `E1` for "E1 or E";
  every key that the model reads must have exactly one value; keys not read are ignored), the breakpoint factors
  `E`, `E1`, `E2` (> 0) and `A1 != A2` of `winner_los`, then `scenpar` ([[scenpar_validate]], its messages pass
  through), then the SF frequency term if `sf_sigma` is requested. All checks run before any output is written
- The models are evaluated as written for any positive frequency, also outside their validity range
- Outputs must not share memory with the inputs or with each other

## Declaration:
```
void quadriga_lib::path_loss(
    const std::string &pl_model,
    const std::unordered_map<std::string, std::vector<double>> &plpar,
    const std::unordered_map<std::string, double> &scenpar,
    const arma::vec &fc,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    arma::mat &pl,
    arma::mat *sf_sigma = nullptr,
    const arma::vec *d2d_in = nullptr,
    const arma::vec *rnd_he = nullptr);
```

## Inputs:
- **`pl_model`** — Path-loss model: `logdist`, `logdist_simple`, `constant`, `winner_los`, `winner_nlos`,
  `winner_pathloss`, `dual_slope`, `tripple_slope`, `nlos`, `satellite`, or empty (zero loss)
- **`plpar`** — Path-loss parameters, keys without "PL_", one value each, as returned by [[conf_file_read]]
- **`scenpar`** — Scenario parameters (SF std terms, `LSP_fc_min`) as returned by [[conf_file_read]] or
  [[scenpar_validate]]; missing keys have their default value
- **`fc`** — Carrier frequencies in [Hz], finite and > 0; `[n_freq]`
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`d2d_in`** *(optional)* — Indoor 2D distance in [m] of the UT of each link (e.g. from [[o2i_loss]]), finite
  and >= 0; given = legacy mode with the outdoor distance; 0 = outdoor link; `[n_link]`
- **`rnd_he`** *(optional)* — Uniform(0,1) value per link for the random environment height, column 10 of
  [[lsf_draw]], in [0, 1]; `[n_link]`

## Outputs:
- **`pl`** — Path loss in [dB]; `[n_link, n_freq]`
- **`sf_sigma`** *(optional)* — Shadow-fading standard deviation in [dB], >= 0 for the default; `[n_link, n_freq]`

## See also:
- [[conf_file_read]] (the model and its parameters)
- [[lsf_draw]] (the random values, column 10 for `rnd_he`)
- [[lsp_generate]] (takes `sf_sigma`)
- [[o2i_loss]] (penetration loss and indoor distance)
MD!*/

void quadriga_lib::path_loss(const std::string &pl_model,
                             const std::unordered_map<std::string, std::vector<double>> &plpar,
                             const std::unordered_map<std::string, double> &scenpar,
                             const arma::vec &fc,
                             const arma::mat &tx_pos,
                             const arma::mat &rx_pos,
                             arma::mat &pl,
                             arma::mat *sf_sigma,
                             const arma::vec *d2d_in,
                             const arma::vec *rnd_he)
{
    // Validation, before any output is touched
    qd_gbsm_check_links(&tx_pos, &rx_pos);
    qd_gbsm_check_fc(fc);
    const arma::uword n_link = rx_pos.n_rows, n_freq = fc.n_elem;
    if (d2d_in != nullptr && d2d_in->n_elem != n_link)
        throw std::invalid_argument("Number of elements in 'd2d_in' does not match number of rows in 'rx_pos'.");
    if (rnd_he != nullptr && rnd_he->n_elem != n_link)
        throw std::invalid_argument("Number of elements in 'rnd_he' does not match number of rows in 'rx_pos'.");
    if (d2d_in != nullptr && (!d2d_in->is_finite() || arma::any(*d2d_in < 0.0)))
        throw std::invalid_argument("Input 'd2d_in' must be finite and >= 0.");
    if (rnd_he != nullptr)
        for (arma::uword l = 0; l < n_link; ++l)
            if (!((*rnd_he)(l) >= 0.0 && (*rnd_he)(l) <= 1.0))
                throw std::invalid_argument("Input 'rnd_he' must be in the range from 0 to 1.");

    // Model
    enum model_t
    {
        m_none,
        m_constant,
        m_logdist,
        m_logdist_simple,
        m_winner_los,
        m_winner_nlos,
        m_winner_pathloss,
        m_dual_slope,
        m_tripple_slope,
        m_nlos,
        m_satellite
    };
    model_t model = m_none;
    if (pl_model.empty())
        model = m_none;
    else if (pl_model == "constant")
        model = m_constant;
    else if (pl_model == "logdist")
        model = m_logdist;
    else if (pl_model == "logdist_simple")
        model = m_logdist_simple;
    else if (pl_model == "winner_los")
        model = m_winner_los;
    else if (pl_model == "winner_nlos")
        model = m_winner_nlos;
    else if (pl_model == "winner_pathloss")
        model = m_winner_pathloss;
    else if (pl_model == "dual_slope")
        model = m_dual_slope;
    else if (pl_model == "tripple_slope")
        model = m_tripple_slope;
    else if (pl_model == "nlos")
        model = m_nlos;
    else if (pl_model == "satellite")
        model = m_satellite;
    else
        throw std::invalid_argument("Path-loss model '" + pl_model + "' is not supported.");

    // Parameters, in the order the model reads them: 'opt' returns false if the key is absent, 'req' throws
    auto opt = [&](const char *key, double &val) -> bool
    {
        auto it = plpar.find(key);
        if (it == plpar.end())
            return false;
        if (it->second.size() != 1)
            throw std::invalid_argument(std::string("Path-loss parameter '") + key + "' must have one value.");
        val = it->second[0];
        return true;
    };
    auto missing = [&](const char *key)
    { return std::invalid_argument("Path-loss model '" + pl_model + "' requires the parameter '" + key + "'."); };
    auto req = [&](const char *key, double &val)
    {
        if (!opt(key, val))
            throw missing(key);
    };
    auto positive = [](const char *key, double val)
    {
        if (!(val > 0.0))
            throw std::invalid_argument(std::string("Path-loss parameter '") + key + "' must be > 0.");
    };

    // Parameters of the dual-slope model; the breakpoint factor is "E1 or E" for 'tripple_slope', "E" otherwise
    auto read_dual = [&](qd_pl_dual &p, bool triple)
    {
        if (!opt("A1", p.A1) && !opt("A", p.A1))
            throw missing("A1");
        req("B", p.B);
        const char *e_key = "E";
        if (!triple)
            req("E", p.E);
        else if (opt("E1", p.E))
            e_key = "E1";
        else if (!opt("E", p.E))
            throw missing("E1");
        p.A2 = p.A1, p.C = 0.0, p.D = 0.0, p.hE = 0.0, p.sig1 = 0.0, p.sig2 = 0.0;
        opt("A2", p.A2), opt("C", p.C), opt("D", p.D);
        positive(e_key, p.E);
    };

    qd_pl_dual dual = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, false, false, 0.0, 0.0};
    double A = 0.0, B = 0.0, C = 0.0, D = 0.0, X = 0.0, SF = 0.0;
    bool has_SF = false;
    double w[21] = {}; // winner_los: A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2; winner_nlos: 21 keys
    qd_pl_triple tri = {0.0, 0.0, 0.0, 0.0, 0.0, false};
    double nl[13] = {}; // An, Bn, Cn, Dn, D1n, D2n, D3n, En, E1n, E2n, E3n, Fn, G1n
    double G2n = 1.0, LOSmax = 1.0, Ax = 0.0, Bx = 0.0, Cx = 0.0, sig = 0.0, usePLa = 1.0;
    bool has_x = false, has_sig = false;

    switch (model)
    {
    case m_none:
        break;
    case m_constant:
        req("A", A);
        break;
    case m_logdist:
        req("A", A), req("B", B), req("C", C);
        has_SF = opt("SF", SF);
        break;
    case m_logdist_simple:
        req("A", A), req("B", B);
        break;
    case m_winner_los:
    {
        const char *keys[14] = {"A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2", "E1", "E2", "F1", "F2", "sig1", "sig2"};
        for (int i = 0; i < 14; ++i)
            req(keys[i], w[i]);
        if (w[0] == w[1])
            throw std::invalid_argument("Path-loss parameters 'A1' and 'A2' must differ for model 'winner_los'.");
        break;
    }
    case m_winner_nlos:
    {
        const char *keys[21] = {"A1", "A2", "A3", "Ah1", "Ah2", "Ah3", "B1", "B2", "B3", "C1", "C2", "C3",
                                "D1", "D2", "D3", "E1", "E2", "E3", "F1", "F2", "F3"};
        for (int i = 0; i < 21; ++i)
            req(keys[i], w[i]);
        break;
    }
    case m_winner_pathloss:
        req("A", A), req("B", B), req("C", C), req("X", X);
        has_SF = opt("SF", SF);
        break;
    case m_dual_slope:
    {
        read_dual(dual, false);
        double v = 0.0;
        opt("hE", dual.hE);
        dual.hE_rand = opt("hE_rand", v) && v != 0.0;
        dual.has_sig1 = opt("sig1", dual.sig1);
        dual.has_sig2 = opt("sig2", dual.sig2);
        break;
    }
    case m_tripple_slope:
    {
        read_dual(dual, true);
        tri.A3 = dual.A2, tri.E2 = dual.E;
        opt("A3", tri.A3);
        if (opt("E2", tri.E2))
            positive("E2", tri.E2);
        opt("hE1", tri.hE1), opt("hE2", tri.hE2);
        dual.has_sig1 = opt("sig1", dual.sig1);
        dual.has_sig2 = opt("sig2", dual.sig2);
        tri.has_sig3 = opt("sig3", tri.sig3);
        break;
    }
    case m_nlos:
    {
        const char *keys[13] = {"An", "Bn", "Cn", "Dn", "D1n", "D2n", "D3n", "En", "E1n", "E2n", "E3n", "Fn", "G1n"};
        req(keys[0], nl[0]), req(keys[1], nl[1]);
        for (int i = 2; i < 13; ++i)
            opt(keys[i], nl[i]);
        opt("G2n", G2n);
        opt("LOSmax", LOSmax);
        if (LOSmax != 0.0)
        {
            read_dual(dual, false);
            double v = 0.0;
            opt("hE", dual.hE);
            dual.hE_rand = opt("hE_rand", v) && v != 0.0;
        }
        const bool hax = opt("Ax", Ax), hbx = opt("Bx", Bx), hcx = opt("Cx", Cx);
        has_x = hax || hbx || hcx;
        has_sig = opt("sig", sig);
        break;
    }
    case m_satellite:
        A = 20.0, B = 32.45, C = 20.0, D = 0.0;
        opt("A", A), opt("B", B), opt("C", C), opt("D", D), opt("usePLa", usePLa);
        break;
    }

    // Scenario parameters, completed and validated
    std::unordered_map<std::string, double> sp = scenpar;
    quadriga_lib::scenpar_validate(sp, "legacy", nullptr, nullptr, true);

    // Default shadow-fading std: the SF std of the LSP formula (only SF evaluated and checked)
    qd_gbsm_lsp_stats sf_stats(sp, (sf_sigma != nullptr) ? fc : arma::vec(), 1u << 2);

    // UMa environment height of TR 38.901 Table 7.4.1-1, note 1, from a Uniform(0,1) value u
    // - d: 2D distance in [m], hUT: raw UT height in [m]
    // - hE = 1 where p1 = 1/(1 + C) rounds to 1 (C below the rounding of 1 + C): the other heights then have a
    //   probability below 1e-16, and (u - p1)/(1 - p1) would be 0/0 for u = 1
    auto draw_hE = [](double u, double d, double hUT)
    {
        const double g = (d <= 18.0) ? 0.0 : 1.25 * std::pow(d / 100.0, 3.0) * std::exp(-d / 150.0);
        const double Cp = (hUT < 13.0) ? 0.0 : (hUT <= 23.0 ? std::pow((hUT - 13.0) / 10.0, 1.5) : 1.0);
        const double Cg = Cp * g; // C of the TR note
        const double p1 = 1.0 / (1.0 + Cg);
        const double n = (hUT - 1.5 >= 12.0) ? std::floor((hUT - 13.5) / 3.0 + 1e-9) + 1.0 : 0.0;
        if (Cg == 0.0 || n == 0.0 || u < p1 || p1 >= 1.0)
            return 1.0;
        const double j = std::min(std::floor((u - p1) / (1.0 - p1) * n), n - 1.0);
        return 12.0 + 3.0 * j;
    };

    // Outputs
    if (pl.n_rows != n_link || pl.n_cols != n_freq)
        pl.set_size(n_link, n_freq);
    if (sf_sigma != nullptr && (sf_sigma->n_rows != n_link || sf_sigma->n_cols != n_freq))
        sf_sigma->set_size(n_link, n_freq);

    const bool tx_single = tx_pos.n_rows != n_link;
    for (arma::uword l = 0; l < n_link; ++l)
    {
        // Distances
        const arma::uword t = tx_single ? 0 : l;
        const double hBS = tx_pos(t, 2), hUT = rx_pos(l, 2);
        const double dx = rx_pos(l, 0) - tx_pos(t, 0), dy = rx_pos(l, 1) - tx_pos(t, 1), dz = hUT - hBS;
        double d2 = std::sqrt(dx * dx + dy * dy), d3 = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (sf_sigma != nullptr)
            sf_stats.set_link(d2, hBS, hUT);
        if (d2d_in != nullptr && (*d2d_in)(l) > 0.0) // Legacy outdoor distance
        {
            const double din = std::min((*d2d_in)(l), d2);
            double d3in = (d2 > 0.0) ? d3 * din / d2 : d3;
            d3in = std::min(d3in, d3);
            if (d3 - d3in < 10.0)
                d3in = d3 - 10.0;
            d3in = std::max(d3in, 0.0);
            const double p = (d3 > 0.0) ? d3in / d3 : 0.0;
            d3 = d3 - d3in;
            d2 = d2 * (1.0 - p);
        }
        d3 = std::max(d3, 0.1), d2 = std::max(d2, 0.1);
        const double lg_d3 = std::log10(d3), lg_d2 = std::log10(d2);

        // Environment height of the dual-slope model
        const bool random_hE = dual.hE_rand && rnd_he != nullptr;
        const double hE = random_hE ? draw_hE((*rnd_he)(l), d2, hUT) : dual.hE;

        for (arma::uword k = 0; k < n_freq; ++k)
        {
            const double f = fc(k) / 1.0e9, lf = std::log10(f);
            double loss = 0.0, sf = 0.0;
            bool set_sf = false; // false: keep the default
            switch (model)
            {
            case m_none:
                break;
            case m_constant:
                loss = A;
                break;
            case m_logdist:
                loss = A * lg_d3 + B + C * lf;
                if (has_SF)
                    sf = SF, set_sf = true;
                break;
            case m_logdist_simple:
                loss = A * lg_d3 + B;
                break;
            case m_winner_los:
            {
                const double hb = std::max(hBS, 1.5), hm = std::max(hUT, 1.5);
                const double G = w[2] + w[4] * lf + w[6] * std::log10(hb) + w[8] * std::log10(hm);
                const double H = w[3] + w[5] * lf + w[7] * std::log10(hb) + w[9] * std::log10(hm);
                const double bp = std::pow(10.0, (H - G) / (w[0] - w[1]));
                if (d2 <= bp)
                    loss = w[0] * lg_d2 + G + w[10] * hm, sf = w[12], set_sf = true;
                else
                    loss = w[1] * lg_d2 + H + w[11] * hm, sf = w[13], set_sf = true;
                break;
            }
            case m_winner_nlos:
            {
                const double hb = std::max(hBS, 1.5), hm = std::max(hUT, 1.5), lhb = std::log10(hb);
                const int j = (f < 1.5) ? 0 : (f < 2.0 ? 1 : 2); // A, Ah, B, C, D, E, F at w[3·i + j]
                loss = (w[j] + w[3 + j] * lhb) * lg_d2 + w[6 + j] + w[9 + j] * lf + w[12 + j] * lhb +
                       w[15 + j] * std::log10(hm) + w[18 + j] * hm;
                break;
            }
            case m_winner_pathloss:
                loss = A * lg_d3 + B + C * std::log10(f / 5.0) + X;
                if (has_SF)
                    sf = SF, set_sf = true;
                break;
            case m_dual_slope:
            {
                bool beyond = false;
                loss = qd_pl_dual_slope(dual, f, d3, d2, hBS, hUT, hE, random_hE, beyond);
                if (!beyond && dual.has_sig1)
                    sf = dual.sig1, set_sf = true;
                if (beyond && dual.has_sig2)
                    sf = dual.sig2, set_sf = true;
                break;
            }
            case m_tripple_slope:
            {
                const double hb = std::max(std::max(std::max(hBS, 1.1 * tri.hE1), 1.1 * tri.hE2), 0.1);
                const double hm = std::max(std::max(std::max(hUT, 1.1 * tri.hE1), 1.1 * tri.hE2), 0.1);
                double dBP1 = dual.E * f * (hb - tri.hE1) * (hm - tri.hE1);
                const double dBP2 = tri.E2 * f * (hb - tri.hE2) * (hm - tri.hE2);
                dBP1 = std::min(dBP1, dBP2);
                const double dBP13 = std::sqrt(dBP1 * dBP1 + (hb - hm) * (hb - hm));
                const double dBP23 = std::sqrt(dBP2 * dBP2 + (hb - hm) * (hb - hm));
                const double LBP1 = dual.A1 * std::log10(dBP13) + dual.B + dual.C * lf + dual.D * dBP13;
                if (d2 <= dBP1)
                {
                    loss = dual.A1 * lg_d3 + dual.B + dual.C * lf + dual.D * d3;
                    if (dual.has_sig1)
                        sf = dual.sig1, set_sf = true;
                }
                else if (d2 <= dBP2)
                {
                    loss = LBP1 + dual.A2 * std::log10(d3 / dBP13);
                    if (dual.has_sig2)
                        sf = dual.sig2, set_sf = true;
                }
                else
                {
                    loss = LBP1 + dual.A2 * std::log10(dBP23 / dBP13) + tri.A3 * std::log10(d3 / dBP23);
                    if (tri.has_sig3)
                        sf = tri.sig3, set_sf = true;
                }
                break;
            }
            case m_nlos:
            {
                const double hb = std::max(hBS, 0.1), hm = std::max(hUT, 0.1);
                const double lhb = std::log10(hb), lhm = std::log10(hm), g = std::log10(G2n * hm);
                loss = nl[0] * lg_d3 + nl[1] + nl[2] * lf + nl[3] * lhb + nl[4] * lhb / hb + nl[5] * lhb / (hb * hb) +
                       nl[6] * hb + nl[7] * lhm + nl[8] * lhm / hm + nl[9] * lhm / (hm * hm) + nl[10] * hm +
                       nl[11] * lhb * lg_d3 + nl[12] * g * g;
                if (LOSmax != 0.0)
                {
                    bool beyond = false;
                    loss = std::max(loss, qd_pl_dual_slope(dual, f, d3, d2, hBS, hUT, hE, random_hE, beyond));
                }
                if (has_x)
                    loss = std::max(loss, Ax * lg_d3 + Bx + Cx * lf);
                if (has_sig)
                    sf = sig, set_sf = true;
                break;
            }
            case m_satellite:
            {
                const double sa = std::min(hBS / d3, 1.0);
                if (sa <= 0.0)
                    loss = std::numeric_limits<double>::infinity();
                else
                {
                    const double PLa = (usePLa != 0.0) ? qd_interp_linear(qd_pl_zenith_att, qd_pl_zenith_f, 93, f) / sa : 0.0;
                    loss = A * lg_d3 + B + C * lf + D * std::log10(std::asin(sa)) + PLa;
                }
                break;
            }
            }
            pl(l, k) = loss;
            if (sf_sigma != nullptr)
            {
                if (!set_sf)
                {
                    double mu = 0.0;
                    sf_stats.get(2, k, mu, sf);
                }
                (*sf_sigma)(l, k) = sf;
            }
        }
    }
}
