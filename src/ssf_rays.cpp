// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_gbsm.hpp"
#include "quadriga_gbsm_helpers.hpp"

#include <string>
#include <unordered_map>
#include <vector>

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_rays
Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16)

- Splits the clusters of [[ssf_clusters]] into rays and writes one [[gbsm_rays]] object per link: ray delays,
  powers, angles, polarization transfer matrices with initial phases, path types and tap indices
- One geometry per call: the TR methods compute frequency `i_freq`, `"quadriga"` all frequencies
- Ray layout per link: index 0 LOS, 1 GR (if `GR_enabled`), then ray m of NLOS cluster n at `1 + G + n·M + m`

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
- **`lsp`**, **`lsp_mu`**, **`lsp_sigma`** — LSPs, log-domain means and std from [[lsp_generate]];
  `[n_link, n_freq, 8]`
- **`delay`**, **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Cluster delays and angles from [[ssf_clusters]];
  `[n_link, n_slot, n_freq]` for `"38.901_MF"`, else `[n_link, n_slot, 1]`
- **`power`** — Cluster powers from [[ssf_clusters]]; `[n_link, n_slot, n_freq]`
- **`rnd_ray`** — "ray" sets from [[gbsm_draw]]; `[n_link, n_gen, n_freq]` (TR methods) or `[n_link, n_gen, 1]`
- **`method`** *(optional)* — Method of [[ssf_clusters]]: `"38.901"`, `"38.901_MF"` or `"quadriga"`
- **`i_freq`** *(optional)* — TR methods: index into `center_frequency`, 0-based; `"quadriga"`: 0
- **`angle_model`** *(optional)* — `"additive"` or `"rotation"`
- **`pol_model`** *(optional)* — `"38.901"` or `"quadriga"`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity from [[lsp_generate]]; required if
  `GR_enabled`; `[n_link, n_freq]`
- **`pol_variability`** *(optional)* — TR 7.6.16 polarisation power variability
- **`random_phase`** *(optional)* — `false`: all initial phases 0

## Output:
- **`rays`** — Ray data, one entry per link; `fbs_pos`, `lbs_pos`, `fallback` empty; `[n_link]`

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
    // STUB: no computation yet; outputs sized with zeros
    (void)tx_pos, (void)lsp, (void)lsp_mu, (void)lsp_sigma, (void)delay, (void)power, (void)aod, (void)eod;
    (void)aoa, (void)eoa, (void)rnd_ray, (void)i_freq, (void)angle_model, (void)pol_model, (void)gr_eps_re;
    (void)gr_eps_im, (void)pol_variability, (void)random_phase;

    std::unordered_map<std::string, double> sp = scenpar;
    quadriga_lib::scenpar_validate(sp, nullptr, nullptr, true);
    arma::uword N, M, G, n_ray, n_gen;
    qd_gbsm_ray_layout(sp, N, M, G, n_ray, n_gen);

    const arma::uword n_link = rx_pos.n_rows, n_freq = (method == "quadriga") ? center_frequency.n_elem : 1;
    if (rays.size() != n_link)
        rays.resize(n_link);
    for (auto &r : rays)
    {
        r.method = method;
        r.tx_pos.zeros(3), r.rx_pos.zeros(3), r.center_frequency.zeros(n_freq);
        r.delay.zeros(n_ray), r.power.zeros(n_ray, n_freq);
        r.aod.zeros(n_ray), r.eod.zeros(n_ray), r.aoa.zeros(n_ray), r.eoa.zeros(n_ray);
        r.M.zeros(8, n_ray, n_freq), r.path_type.zeros(n_ray), r.tap.zeros(n_ray);
        r.gr_eps_re.zeros(G ? n_freq : 0), r.gr_eps_im.zeros(G ? n_freq : 0);
        r.fbs_pos.reset(), r.lbs_pos.reset(), r.fallback.reset();
    }
}
