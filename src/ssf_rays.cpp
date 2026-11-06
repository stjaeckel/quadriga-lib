// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_lib.hpp"
#include "quadriga_gbsm_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Magnitudes a_k of the ray offsets of TR 38.901 Table 7.5-3 (TR ray 2k - 1: +a_k, ray 2k: -a_k), per unit rms
// cluster spread
static const double qd_table_7_5_3[10] = {0.0447, 0.1413, 0.2492, 0.3715, 0.5129,
                                          0.6797, 0.8844, 1.1481, 1.5195, 2.1551};

// Ray offsets alpha_m of M rays per unit cluster spread, ascending
// - M = 20: the 20 values of Table 7.5-3
// - M < 20: the first M values o of the reordered table of QuaDRiGa-NG, alpha = (o - mean(o)) / rms(o - mean(o));
//   mean 0 and rms 1, not symmetric; M = 1 gives 0
// - M > 20: Laplacian quantiles q_k = -ln(2·(1 - j_k / (M + 1))), j_k = M - K + k, k = 1 ... K = floor(M / 2), the
//   set {0 if M is odd, +-q_k} scaled to the rms of Table 7.5-3; symmetric, the largest offset grows with M
static std::vector<double> qd_ray_offsets(arma::uword M)
{
    const double ng[20] = {-0.8844, 1.1481, 0.6797, -1.1481, -0.6797, 1.5195, -0.0447, 0.3715, -1.5195, -0.2492,
                           -0.5129, 0.5129, 0.8844, -2.1551, -0.1413, 0.1413, 0.0447, 2.1551, -0.3715, 0.2492};
    auto rms = [](const std::vector<double> &v)
    {
        double s = 0.0;
        for (double x : v)
            s += x * x;
        return std::sqrt(s / (double)v.size());
    };
    std::vector<double> o;
    if (M == 20)
        for (double a : qd_table_7_5_3)
            o.push_back(a), o.push_back(-a);
    else if (M < 20)
    {
        double mean = 0.0;
        for (arma::uword m = 0; m < M; ++m)
            mean += ng[m] / (double)M;
        for (arma::uword m = 0; m < M; ++m)
            o.push_back(ng[m] - mean);
        const double r = rms(o);
        for (double &x : o)
            x = (r > 0.0) ? x / r : 0.0;
    }
    else
    {
        std::vector<double> table;
        for (double a : qd_table_7_5_3)
            table.push_back(a), table.push_back(-a);
        const arma::uword K = M / 2;
        if (M % 2 == 1)
            o.push_back(0.0);
        for (arma::uword k = 1; k <= K; ++k)
        {
            const double q = -std::log(2.0 * (1.0 - (double)(M - K + k) / (double)(M + 1)));
            o.push_back(q), o.push_back(-q);
        }
        const double scale = rms(table) / rms(o);
        for (double &x : o)
            x *= scale;
    }
    std::sort(o.begin(), o.end());
    return o;
}

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_rays
Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16)

- Splits the clusters of [[ssf_clusters]] into rays and writes one [[gbsm_rays]] object per link: delays, coefficient
  powers, angles, polarization transfer matrices with initial phases, path types and tap indices; `fbs_pos`, `lbs_pos`
  and `fallback` stay empty (see .[[solve_scatterers]])
- A pure function of the clusters, the LSPs and the random values, without a seed: a link does not depend on the other
  links of the call
- One geometry per call: the TR methods compute frequency `i_freq` with slice `i_freq` of `rnd_ray` (the output holds
  this frequency only; one call per frequency), `"quadriga"` all frequencies
- Ray layout per link, fixed per scenario: index 0 LOS, 1 GR (if `GR_enabled`), then ray `m` of NLOS cluster `n` at
  `1 + G + n·M + m` (`G` = 1 with GR, `M` = `NumSubPaths`); clusters at power 0 keep their rays at power 0
- LOS and GR rays: delay, angles and power of their slot; Jones matrices diag(1, -1) and diag(R_par, -R_per)/R of
  the ground reflection (TR eqs. (7.6-38) to (7.6-40), R² = (abs(R_par)² + abs(R_per)²)/2)
- NLOS rays: power P_n/M; offsets of Table 7.5-3 scaled by the cluster spreads (`PerClusterAS_D`, `PerClusterAS_A`,
  `PerClusterES_A`; ZSD of eq. (7.5-20)), the zenith angles wrapped as in step 7; XPR and four initial phases per ray
  (steps 9, 10)
- Coupling (step 8): within a cluster or sub-cluster, the ray with the k-th smallest value of a quantity's rank
  block gets the k-th smallest offset, separately for AOD, AOA, ZOD and ZOA
- Sub-clusters (step 11, `M` = 20 only): the two clusters of largest power at frequency `i_freq` (equal powers: lower
  index first) are split into three taps at the delays + 0, 1.28 and 2.56 c_DS (Table 7.5-6) with the rays at the
  positions 0-9, 10-15 and 16-19; every tap is a contiguous block of rays
- `M` other than 20: QuaDRiGa-NG's offsets below 20 (mean 0, rms 1), Laplacian quantiles at the rms of Table 7.5-3
  above 20; no sub-clusters
- `i_freq` indexes `center_frequency` here and `rays.center_frequency` in [[gbsm_single_link_channel]] (one entry
  for the TR methods, so 0 there)
- Not available yet: method `"quadriga"`, `angle_model = "rotation"`, `pol_model = "quadriga"`, `pol_variability`

## Declaration:
```
void quadriga_lib::ssf_rays(
    const std::unordered_map<std::string, double> &scenpar,
    const arma::mat &tx_pos,
    const arma::mat &rx_pos,
    const arma::vec &center_frequency,
    const arma::cube &lsp,
    const arma::cube &lsp_mu,
    const arma::cube &lsp_sigma,
    const arma::cube &delay,
    const arma::cube &power,
    const arma::cube &aod,
    const arma::cube &eod,
    const arma::cube &aoa,
    const arma::cube &eoa,
    const arma::cube &rnd_ray,
    std::vector<gbsm_rays> &rays,
    const std::string &method = "38.901",
    arma::uword i_freq = 0,
    const std::string &angle_model = "additive",
    const std::string &pol_model = "38.901",
    const arma::mat *gr_eps_re = nullptr,
    const arma::mat *gr_eps_im = nullptr,
    bool pol_variability = false,
    bool random_phase = true);
```

## Inputs:
- **`scenpar`** — Scenario parameters
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`center_frequency`** — Carrier frequencies in [Hz]; `[n_freq]`
- **`lsp`**, **`lsp_mu`**, **`lsp_sigma`** — LSPs, log-domain means and std from [[lsp_generate]]; used: the ESD
  mean and the XPR mean and std, the XPR LSP for `pol_model = "quadriga"`; `[n_link, n_freq, 8]`
- **`delay`**, **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Cluster delays and angles from [[ssf_clusters]]
  (`n_slot` slots); `[n_link, n_slot, n_freq]` for `"38.901_MF"`, else `[n_link, n_slot, 1]`
- **`power`** — Cluster powers from [[ssf_clusters]]; `[n_link, n_slot, n_freq]`
- **`rnd_ray`** — "ray" sets from [[gbsm_draw]], `n_gen` = 14·N·M + N with N = `NumClusters` - 1; TR methods:
  slice `k` = set `k`, `[n_link, n_gen, n_freq]`; `"quadriga"`: the shared set, `[n_link, n_gen, 1]`
- **`method`** *(optional)* — Method of [[ssf_clusters]]: `"38.901"`, `"38.901_MF"` or `"quadriga"`
- **`i_freq`** *(optional)* — TR methods: index into `center_frequency`, 0-based; `"quadriga"`: 0
- **`angle_model`** *(optional)* — `"additive"` (TR) or `"rotation"`
- **`pol_model`** *(optional)* — `"38.901"` (TR) or `"quadriga"`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity from [[lsp_generate]]; required if
  `GR_enabled`, else ignored; `[n_link, n_freq]`
- **`pol_variability`** *(optional)* — TR 7.6.16 polarisation power variability
- **`random_phase`** *(optional)* — `false`: all initial phases 0

## Output:
- **`rays`** — Ray data, one entry per link; entries with attributes of the right size are written in place;
  `[n_link]`

## See also:
- [[gbsm_rays]] (attributes of the output)
- [[ssf_paths]] (paths of one link)
MD!*/

void quadriga_lib::ssf_rays(const std::unordered_map<std::string, double> &scenpar,
                            const arma::mat &tx_pos,
                            const arma::mat &rx_pos,
                            const arma::vec &center_frequency,
                            const arma::cube &lsp,
                            const arma::cube &lsp_mu,
                            const arma::cube &lsp_sigma,
                            const arma::cube &delay,
                            const arma::cube &power,
                            const arma::cube &aod,
                            const arma::cube &eod,
                            const arma::cube &aoa,
                            const arma::cube &eoa,
                            const arma::cube &rnd_ray,
                            std::vector<gbsm_rays> &rays,
                            const std::string &method,
                            arma::uword i_freq,
                            const std::string &angle_model,
                            const std::string &pol_model,
                            const arma::mat *gr_eps_re,
                            const arma::mat *gr_eps_im,
                            bool pol_variability,
                            bool random_phase)
{
    // Validation, before the output is touched
    const bool m_tr = method == "38.901", m_mf = method == "38.901_MF", m_qd = method == "quadriga";
    if (!m_tr && !m_mf && !m_qd)
        throw std::invalid_argument("Input 'method' must be '38.901', '38.901_MF' or 'quadriga'.");
    if (angle_model != "additive" && angle_model != "rotation")
        throw std::invalid_argument("Input 'angle_model' must be 'additive' or 'rotation'.");
    if (pol_model != "38.901" && pol_model != "quadriga")
        throw std::invalid_argument("Input 'pol_model' must be '38.901' or 'quadriga'.");
    const bool pol_qd = pol_model == "quadriga";
    if (pol_variability && pol_qd)
        throw std::invalid_argument("Input 'pol_variability' needs pol_model '38.901'.");

    std::unordered_map<std::string, double> sp = scenpar;
    quadriga_lib::scenpar_validate(sp, nullptr, nullptr, true);
    arma::uword N, M, G, n_ray, n_gen;
    qd_gbsm_ray_layout(sp, N, M, G, n_ray, n_gen);
    qd_gbsm_check_links(&tx_pos, &rx_pos);
    qd_gbsm_check_fc(center_frequency);
    const arma::uword n_link = rx_pos.n_rows, F = center_frequency.n_elem, n_slot = 1 + G + N;
    if (m_tr && F > 1)
        throw std::invalid_argument(
            "Method '38.901' supports one frequency only; use '38.901_MF' or 'quadriga' for several frequencies.");
    if (!m_qd && i_freq >= F)
        throw std::invalid_argument("Input 'i_freq' is out of range for " + std::to_string(F) + " frequencies.");
    if (m_qd && i_freq != 0)
        throw std::invalid_argument("Input 'i_freq' must select the first frequency for method 'quadriga'.");

    auto has_size = [](const arma::cube &c, arma::uword r, arma::uword n_col, arma::uword n_sl)
    { return c.n_rows == r && c.n_cols == n_col && c.n_slices == n_sl; };
    const std::pair<const arma::cube *, const char *> stats[3] = {{&lsp, "lsp"}, {&lsp_mu, "lsp_mu"},
                                                                  {&lsp_sigma, "lsp_sigma"}};
    for (const auto &x : stats)
        if (!has_size(*x.first, n_link, F, 8))
            throw std::invalid_argument("Input '" + std::string(x.second) + "' must have size [n_link, n_freq, 8].");
    const std::pair<const arma::cube *, const char *> geo[5] = {
        {&delay, "delay"}, {&aod, "aod"}, {&eod, "eod"}, {&aoa, "aoa"}, {&eoa, "eoa"}};
    for (const auto &x : geo)
        if (!has_size(*x.first, n_link, n_slot, m_mf ? F : 1))
            throw std::invalid_argument("Input '" + std::string(x.second) +
                                        "' must have size [n_link, n_slot, n_freq] for method '38.901_MF', else "
                                        "[n_link, n_slot, 1].");
    if (!has_size(power, n_link, n_slot, F))
        throw std::invalid_argument("Input 'power' must have size [n_link, n_slot, n_freq].");
    if (n_gen == 0 ? rnd_ray.n_elem != 0 : !has_size(rnd_ray, n_link, n_gen, m_qd ? 1 : F))
        throw std::invalid_argument("Input 'rnd_ray' must have size [n_link, " + std::to_string(n_gen) +
                                    ", n_freq] for the TR methods or [n_link, " + std::to_string(n_gen) +
                                    ", 1] for 'quadriga'.");
    if (G)
    {
        if (gr_eps_re == nullptr || gr_eps_im == nullptr)
            throw std::invalid_argument("Inputs 'gr_eps_re' and 'gr_eps_im' are required when 'GR_enabled' is set.");
        if (gr_eps_re->n_rows != n_link || gr_eps_re->n_cols != F || gr_eps_im->n_rows != n_link ||
            gr_eps_im->n_cols != F)
            throw std::invalid_argument("Inputs 'gr_eps_re' and 'gr_eps_im' must have size [n_link, n_freq].");
        if (!gr_eps_re->is_finite() || !gr_eps_im->is_finite())
            throw std::invalid_argument("Inputs 'gr_eps_re' and 'gr_eps_im' must be finite.");
        if (arma::any(arma::vectorise(*gr_eps_re) < 1.0) || arma::any(arma::vectorise(*gr_eps_im) < 0.0))
            throw std::invalid_argument("Ground permittivity needs 'gr_eps_re' >= 1 and 'gr_eps_im' >= 0.");
        if (n_link > 0 && (arma::any(rx_pos.col(2) < 0.0) || arma::any(tx_pos.col(2) < 0.0)))
            throw std::invalid_argument("Ground reflection requires z >= 0 in 'tx_pos' and 'rx_pos'.");
    }

    // Values used: the TR methods read frequency i_freq (geometry slice i_freq for "38.901_MF", else 0), "quadriga"
    // all frequencies
    const arma::uword k0 = m_qd ? 0 : i_freq, k1 = m_qd ? F - 1 : i_freq, g = m_mf ? i_freq : 0;
    for (const auto &x : geo)
        if (!x.first->slice(g).is_finite())
            throw std::invalid_argument("Input '" + std::string(x.second) + "' must be finite.");
    for (arma::uword k = k0; k <= k1; ++k)
    {
        const arma::mat &p = power.slice(k);
        if (!p.is_finite() || arma::any(arma::vectorise(p) < 0.0))
            throw std::invalid_argument("Input 'power' must be finite and >= 0.");
    }
    for (arma::uword k = k0; k <= k1; ++k)
        for (arma::uword l = 0; l < n_link; ++l)
            if (!std::isfinite(lsp_mu(l, k, 5)) || !std::isfinite(lsp_mu(l, k, 7)))
                throw std::invalid_argument("Input 'lsp_mu' must be finite.");
    for (arma::uword k = k0; k <= k1; ++k)
        for (arma::uword l = 0; l < n_link; ++l)
            if (!(std::isfinite(lsp_sigma(l, k, 7)) && lsp_sigma(l, k, 7) >= 0.0))
                throw std::invalid_argument("Input 'lsp_sigma' must be finite and >= 0.");
    for (arma::uword k = k0; k <= k1 && pol_qd; ++k)
        for (arma::uword l = 0; l < n_link; ++l)
            if (!(std::isfinite(lsp(l, k, 7)) && lsp(l, k, 7) > 0.0))
                throw std::invalid_argument("Input 'lsp' must be finite and > 0.");
    if (n_gen > 0 && !rnd_ray.slice(m_qd ? 0 : i_freq).is_finite())
        throw std::invalid_argument("Input 'rnd_ray' must be finite.");
    const arma::uword NM = N * M, b14 = 13 * NM + N; // first column of block 14
    if (m_qd && n_gen > 0)
    {
        const arma::mat x = rnd_ray.slice(0).cols(b14, n_gen - 1);
        if (arma::any(arma::vectorise(x) < 0.0) || arma::any(arma::vectorise(x) > 1.0))
            throw std::invalid_argument("Values of block 14 of 'rnd_ray' must be in [0, 1].");
    }
    if (m_qd)
        throw std::invalid_argument("Option 'method = quadriga' is not available yet.");
    if (angle_model == "rotation")
        throw std::invalid_argument("Option 'angle_model = rotation' is not available yet.");
    if (pol_qd)
        throw std::invalid_argument("Option 'pol_model = quadriga' is not available yet.");
    if (pol_variability)
        throw std::invalid_argument("Option 'pol_variability' is not available yet.");

    if (rays.size() != n_link)
        rays.resize(n_link);
    if (n_link == 0)
        return;

    // Ray offsets per unit cluster spread, ascending: all M offsets for an unsplit cluster; at M = 20 the three
    // sub-cluster groups of Table 7.5-5 at the ray positions 0-9 (TR rays 1-8, 19, 20), 10-15 (TR rays 9-12, 17,
    // 18) and 16-19 (TR rays 13-16), from the magnitudes a_k of TR rays 2k - 1 and 2k
    const std::vector<double> off_all = qd_ray_offsets(M);
    std::vector<double> off_sub[3];
    if (M == 20)
    {
        const int grp[3][5] = {{0, 1, 2, 3, 9}, {4, 5, 8, -1, -1}, {6, 7, -1, -1, -1}};
        for (int i = 0; i < 3; ++i)
        {
            for (int a : grp[i])
                if (a >= 0)
                    off_sub[i].push_back(-qd_table_7_5_3[a]), off_sub[i].push_back(qd_table_7_5_3[a]);
            std::sort(off_sub[i].begin(), off_sub[i].end());
        }
    }

    const double pi = arma::datum::pi, d2r = pi / 180.0, sqrt_half = std::sqrt(0.5);
    const double c_asd = sp.at("PerClusterAS_D"), c_asa = sp.at("PerClusterAS_A"), c_zsa = sp.at("PerClusterES_A");
    const double zsd_scale = sp.at("PerClusterES_D_scale"), c_zsd0 = sp.at("PerClusterES_D");
    const double c_ds = qd_gbsm_cds(sp, center_frequency(i_freq));
    const bool tx_single = tx_pos.n_rows != n_link;

    // Azimuth wrapped to [-pi, pi); elevation in [rad] of the zenith angle 90 deg - el_c + off (in [deg]), wrapped
    // to [0, 360) and folded at 180 deg (TR 38.901 clause 7.5 step 7, used for ZOA and ZOD)
    auto wrap = [pi](double x) { return x - 2.0 * pi * std::floor((x + pi) / (2.0 * pi)); };
    auto fold = [d2r](double el_c, double off)
    {
        double t = 90.0 - el_c / d2r + off;
        t -= 360.0 * std::floor(t / 360.0);
        return (90.0 - (t >= 180.0 ? 360.0 - t : t)) * d2r;
    };

    auto one_link = [&](arma::uword l)
    {
        const arma::uword t = tx_single ? 0 : l, k = i_freq;
        gbsm_rays &r = rays[l];

        // Sizes; attributes of the right size are written in place
        r.method = method;
        auto fit = [](auto &x, arma::uword n)
        {
            if (x.n_elem != n)
                x.set_size(n);
        };
        fit(r.tx_pos, 3), fit(r.rx_pos, 3), fit(r.center_frequency, 1), fit(r.delay, n_ray);
        fit(r.aod, n_ray), fit(r.eod, n_ray), fit(r.aoa, n_ray), fit(r.eoa, n_ray), fit(r.path_type, n_ray);
        fit(r.tap, n_ray);
        if (r.power.n_rows != n_ray || r.power.n_cols != 1)
            r.power.set_size(n_ray, 1);
        if (r.M.n_rows != 8 || r.M.n_cols != n_ray || r.M.n_slices != 1)
            r.M.set_size(8, n_ray, 1);
        if (G)
        {
            fit(r.gr_eps_re, 1), fit(r.gr_eps_im, 1);
            r.gr_eps_re(0) = (*gr_eps_re)(l, k), r.gr_eps_im(0) = (*gr_eps_im)(l, k);
        }
        for (auto *x : {&r.gr_eps_re, &r.gr_eps_im})
            if (!G && !x->empty())
                x->reset();
        for (auto *x : {&r.fbs_pos, &r.lbs_pos})
            if (!x->empty())
                x->reset();
        if (!r.fallback.empty())
            r.fallback.reset();
        for (arma::uword i = 0; i < 3; ++i)
            r.tx_pos(i) = tx_pos(t, i), r.rx_pos(i) = rx_pos(l, i);
        r.center_frequency(0) = center_frequency(k);
        r.M.zeros();

        // LOS and GR rays: the slot values; Jones matrices diag(1, -1) (TR 38.901 eq. (7.5-29)) and
        // diag(R_par, -R_per) / R with R² = (abs(R_par)² + abs(R_per)²) / 2 (eqs. (7.6-38) to (7.6-40); R = 0 gives 0)
        for (arma::uword s = 0; s < 1 + G; ++s)
        {
            r.delay(s) = delay(l, s, g), r.power(s, 0) = power(l, s, k);
            r.aod(s) = aod(l, s, g), r.eod(s) = eod(l, s, g), r.aoa(s) = aoa(l, s, g), r.eoa(s) = eoa(l, s, g);
            r.path_type(s) = s, r.tap(s) = 3 * s;
        }
        r.M(0, 0, 0) = 1.0, r.M(6, 0, 0) = -1.0;
        if (G)
        {
            const double dx = rx_pos(l, 0) - tx_pos(t, 0), dy = rx_pos(l, 1) - tx_pos(t, 1);
            std::complex<double> R_par, R_per;
            qd_gbsm_fresnel((*gr_eps_re)(l, k), (*gr_eps_im)(l, k), tx_pos(t, 2), rx_pos(l, 2),
                            std::sqrt(dx * dx + dy * dy), R_par, R_per);
            const double R = std::sqrt(0.5 * (std::norm(R_par) + std::norm(R_per)));
            if (R > 0.0)
                r.M(0, 1, 0) = R_par.real() / R, r.M(1, 1, 0) = R_par.imag() / R, r.M(6, 1, 0) = -R_per.real() / R,
                          r.M(7, 1, 0) = -R_per.imag() / R;
        }
        if (N == 0)
            return;

        // Split (TR 38.901 clause 7.5 step 11, M = 20 only): the two clusters of the largest coefficient power > 0
        // at frequency k, equal powers in cluster order
        arma::uword split_a = N, split_b = N; // N = none
        for (arma::uword n = 0; n < N && M == 20; ++n)
        {
            const double p = power(l, 1 + G + n, k);
            if (!(p > 0.0))
                continue;
            if (split_a == N || p > power(l, 1 + G + split_a, k))
                split_b = split_a, split_a = n;
            else if (split_b == N || p > power(l, 1 + G + split_b, k))
                split_b = n;
        }

        // Spreads in [deg] per rank block (0 AOD, 1 AOA, 2 ZOD, 3 ZOA); c_ZSD of eq. (7.5-20)
        const double c_zsd = zsd_scale > 0.0 ? zsd_scale * std::pow(10.0, lsp_mu(l, k, 5)) : c_zsd0;
        const double c_spread[4] = {c_asd, c_asa, c_zsd, c_zsa};
        const double xpr_mu = lsp_mu(l, k, 7), xpr_sigma = lsp_sigma(l, k, 7);
        std::vector<double> xr(n_gen); // the values of the link, contiguous
        for (arma::uword c = 0; c < n_gen; ++c)
            xr[c] = rnd_ray.at(l, c, k);
        auto x = [&](arma::uword b, arma::uword n, arma::uword m) { return xr[b * NM + n * M + m]; };
        std::vector<arma::uword> order(M);
        std::vector<double> alpha(4 * M); // offset per rank block and ray position

        for (arma::uword n = 0; n < N; ++n)
        {
            const arma::uword s = 1 + G + n;
            const bool split = n == split_a || n == split_b;

            // Coupling (step 8): within each group of ray positions, the ray with the k-th smallest value of the
            // rank block gets the k-th smallest offset of the group (equal values: lower position first)
            const arma::uword n_grp = split ? 3 : 1, first[3] = {0, 10, 16}, size[3] = {split ? 10 : M, 6, 4};
            for (arma::uword q = 0; q < 4; ++q)
                for (arma::uword i = 0; i < n_grp; ++i)
                {
                    const std::vector<double> &A = split ? off_sub[i] : off_all;
                    for (arma::uword j = 0; j < size[i]; ++j)
                        order[j] = first[i] + j;
                    std::stable_sort(order.begin(), order.begin() + size[i],
                                     [&](arma::uword a, arma::uword b) { return x(q, n, a) < x(q, n, b); });
                    for (arma::uword j = 0; j < size[i]; ++j)
                        alpha[q * M + order[j]] = A[j];
                }

            for (arma::uword m = 0; m < M; ++m)
            {
                const arma::uword i = 1 + G + n * M + m, sub = split ? (m < 10 ? 0 : (m < 16 ? 1 : 2)) : 0;

                // Delay (eq. (7.5-26), Table 7.5-5), tap, power P_n / M
                r.delay(i) = delay(l, s, g) + (sub == 0 ? 0.0 : (sub == 1 ? 1.28 : 2.56)) * c_ds;
                r.tap(i) = 3 * s + sub, r.path_type(i) = 2, r.power(i, 0) = power(l, s, k) / (double)M;

                // Angles (step 7, eqs. (7.5-13), (7.5-18), (7.5-20)): offsets c·alpha_m in [deg]
                r.aod(i) = wrap(aod(l, s, g) + c_spread[0] * alpha[m] * d2r);
                r.aoa(i) = wrap(aoa(l, s, g) + c_spread[1] * alpha[M + m] * d2r);
                r.eod(i) = fold(eod(l, s, g), c_spread[2] * alpha[2 * M + m]);
                r.eoa(i) = fold(eoa(l, s, g), c_spread[3] * alpha[3 * M + m]);

                // Polarisation (steps 9, 10, eqs. (7.5-21), (7.5-28)): XPR kappa = 10^(X/10), X = mu + sigma·x in
                // [dB]; initial phases 2·pi·Phi(x) - pi with the standard normal CDF Phi; 8-row layout of M: Re/Im
                // of theta-theta, phi-theta (Rx phi, Tx theta), theta-phi, phi-phi
                const double cross = std::sqrt(std::pow(10.0, -(xpr_mu + xpr_sigma * x(4, n, m)) / 10.0));
                const double amp[4] = {1.0, cross, cross, 1.0};
                const arma::uword block[4] = {5, 7, 6, 8};
                for (int e = 0; e < 4; ++e)
                {
                    const double ph = random_phase ? pi * std::erfc(-x(block[e], n, m) * sqrt_half) - pi : 0.0;
                    r.M(2 * e, i, 0) = amp[e] * std::cos(ph), r.M(2 * e + 1, i, 0) = amp[e] * std::sin(ph);
                }
            }
        }
    };

    // Exceptions (only std::bad_alloc can occur) must not leave the parallel region. Limit: about 0.19 ms per link
    // (N = 19, M = 20); the default thread count is 2.2 times faster than serial from 4 links on (1.7 at 3)
    bool omp_failed = false;
    std::string omp_error;
#pragma omp parallel for if (n_link >= 4)
    for (long long l = 0; l < (long long)n_link; ++l)
    {
        OMP_SAFE_CALL(omp_failed, omp_error, one_link((arma::uword)l));
    }
    if (omp_failed)
        throw std::runtime_error(omp_error);
}
