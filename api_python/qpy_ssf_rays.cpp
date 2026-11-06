// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"
#include "quadriga_gbsm_helpers.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_rays
Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16)

- Splits the clusters of [[ssf_clusters]] into rays and returns one ray dict per link (keys below): delays, coefficient
  powers, angles, polarization transfer matrices with initial phases, path types and tap indices; `fbs_pos`, `lbs_pos`
  and `fallback` stay empty (see [[ssf_scatterers]])
- A pure function of the clusters, the LSPs and the random values, without a seed: a link does not depend on the other
  links of the call
- Ray layout per link, fixed per scenario: index 0 LOS, 1 GR (if `GR_enabled`), then ray `m` of NLOS cluster `n` at
  `1 + G + n·M + m` (`G` = 1 with GR, `M` = `NumSubPaths`); clusters at power 0 keep their rays at power 0
- LOS and GR rays: delay, angles and power of their slot; Jones matrices diag(1, -1) and diag(R_par, -R_perp)/R of
  the ground reflection (TR eqs. (7.6-38) to (7.6-40))
- NLOS rays: power P_n/M; offsets of Table 7.5-3 scaled by the cluster spreads; coupling (step 8) by the rank blocks
  of `rnd_ray`; `M` other than 20: QuaDRiGa-NG's offsets below 20, Laplacian quantiles above 20, no sub-clusters

| Method | `"38.901"`, `"38.901_MF"` | `"quadriga"` |
| --- | --- | --- |
| Frequencies per call | `i_freq` only; slice `i_freq` of `rnd_ray` | all; one geometry, per frequency powers, XPR and Jones matrices; the shared "ray" set |
| Ray delays | cluster delay; sub-clusters + 0, 1.28, 2.56 c_DS (Table 7.5-6) | cluster delay + 2·c_DS·(u - mean of u in the cluster), u of block 14; negative delays set to 0 |
| Taps | sub-cluster of the two strongest clusters (`M` = 20), else cluster | cluster |
| c_DS, ZSD | at frequency `i_freq` | mean over the frequencies |

| Option | Default (TR) | Alternative (QuaDRiGa-NG) |
| --- | --- | --- |
| `angle_model` | `"additive"`: offsets added to azimuth and zenith, zenith folded (step 7) | `"rotation"`: offset pattern rotated to the cluster direction; equal at elevation 0 |
| `pol_model` | `"38.901"`: XPR and four initial phases per ray (steps 9, 10) | `"quadriga"`: rotation polarisation model of QuaDRiGa, one phase, XPR per ray and per cluster |
| `pol_variability` | `False` | `True`: clause 7.6.16 on the NLOS rays with std `PolVar_sigma`; `pol_model = "38.901"` only |

- Sub-clusters (step 11): the two clusters of largest power at frequency `i_freq`; every tap is a contiguous block
- `"quadriga"` sets 0.2 to 1.4 % of the active NLOS rays to delay 0 (at most 2.9 % of the NLOS power)
- `pol_model = "quadriga"`: the matrix refers to the owner end, the end with the larger z, then x, then y (ends within
  1 mm count as equal; the Tx if all tie); the Rx as owner transposes it
- Swap reciprocity: with mirrored inputs (dual mobility in [[gbsm.draw]], mirrored LSPs and clusters,
  `PerClusterAS_D` = `PerClusterAS_A`, `PerClusterES_D` = `PerClusterES_A`, `PerClusterES_D_scale` = 0), swapping
  `tx_pos` and `rx_pos` exchanges departure and arrival per ray and transposes `M` (not for `pol_model = "quadriga"`
  with ends within 1 mm of each other, where the Tx stays the owner)
- Messages name the argument `center_freq` as `'center_frequency'` and the real and imaginary part of `gr_eps` as
  `gr_eps_re` and `gr_eps_im`; the flags take a bool or `numpy.bool_`, `None` selects the default, any other type
  raises `TypeError`

## Ray dict:
One link; all keys are present on output (empty where empty). As input, missing optional keys are empty, other missing
or unknown keys raise `ValueError`; integer keys accept any real dtype. Rays from several sources in one dict: at most
one LOS and one GR ray, with tap index 0 and 3; the other rays take any other tap indices; one `method`;
[[ssf_scatterers]] overwrites given scatterers, so solve rays without scatterers separately and concatenate the arrays.

| Key | Shape | Type | Meaning | Required |
| --- | --- | --- | --- | --- |
| `method` | — | `str` | `"38.901"`, `"38.901_MF"` or `"quadriga"`; GR power law and default mobility | yes |
| `tx_pos`, `rx_pos` | `(3,)` | float | Tx and Rx position the ray data refer to in m | yes |
| `center_frequency` | `(n_freq,)` | float | Carrier frequencies in Hz | yes |
| `delay` | `(n_ray,)` | float | Delays in s relative to the LOS delay d3D / c, d3D the Tx-Rx distance | yes |
| `power` | `(n_ray, n_freq)` | float | Coefficient powers (linear) without link gain; data from paths: path gain | yes |
| `aod`, `eod`, `aoa`, `eoa` | `(n_ray,)` | float | Departure and arrival azimuth and elevation in rad, GCS | yes |
| `M` | `(8, n_ray, n_freq)` | float | Polarization transfer matrices incl. initial phases, real, rows as in [[get_channels_planar]] | yes |
| `path_type` | `(n_ray,)` | uint32 | 0 = LOS, 1 = GR, 2 = NLOS | yes |
| `tap` | `(n_ray,)` | uint32 | Tap index; rays with equal index form one tap | yes |
| `gr_eps` | `(n_freq,)` or empty | complex | Ground permittivity, used as `real - j·imag`; needed where a GR ray is recomputed | no |
| `fbs_pos`, `lbs_pos` | `(3, n_ray)` or empty | float | First- and last-bounce scatterer positions in m | no |
| `fallback` | `(n_ray,)` or empty | uint32 | 1 = ray lengthened by the solver fallback | no |

## Usage:
```
rays = quadriga_lib.gbsm.ssf_rays( scenpar, tx_pos, rx_pos, center_freq, lsp, lsp_mu, lsp_sigma, delay, power, \
    aod, eod, aoa, eoa, rnd_ray, method, i_freq, angle_model, pol_model, gr_eps, pol_variability, random_phase )
```

## Inputs:
- **`scenpar`** — Scenario parameters
- **`tx_pos`** — Tx positions in m; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; `(n_link, 3)`
- **`center_freq`** — Carrier frequencies in Hz; `(n_freq,)`
- **`lsp`**, **`lsp_mu`**, **`lsp_sigma`** — LSPs, log-domain means and std from [[lsp_generate]]; `(n_link, n_freq, 8)`
- **`delay`**, **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Cluster delays and angles from [[ssf_clusters]];
  `(n_link, n_slot, n_freq)` for `"38.901_MF"`, else `(n_link, n_slot, 1)`
- **`power`** — Cluster powers from [[ssf_clusters]]; `(n_link, n_slot, n_freq)`
- **`rnd_ray`** — "ray" sets from [[gbsm.draw]], `n_gen` = 14·N·M + N; TR methods: slice `k` = set `k`, `(n_link,
  n_gen, n_freq)`; `"quadriga"`: the shared set (`set_id = -1`), `(n_link, n_gen, 1)` or `(n_link, n_gen)`
- **`method`** — Method of [[ssf_clusters]]: `"38.901"`, `"38.901_MF"` or `"quadriga"`; default: `"38.901"`
- **`i_freq`** — TR methods: index into `center_freq`, 0-based (`center_frequency` of the output then has one entry, so
  `i_freq` = 0 in [[gbsm.single_link_channel]]); `"quadriga"`: 0; default: 0
- **`angle_model`** — `"additive"` or `"rotation"`; default: `"additive"`
- **`pol_model`** — `"38.901"` or `"quadriga"`; default: `"38.901"`
- **`gr_eps`** — Ground permittivity as [[lsp_generate]] returns it (complex); required if `GR_enabled`, else
  ignored; `(n_link, n_freq)`; default: None
- **`pol_variability`** — TR 7.6.16 polarisation power variability; default: `False`
- **`random_phase`** — `False`: all initial phases 0; default: `True`

## Output:
- **`rays`** — Ray dicts, one per link; `list` of `n_link` dicts

## See also:
- [[ssf_scatterers]] (scatterer positions and validity scales of one link)
- [[ssf_paths]] (paths of one link)
- [[gbsm.single_link_channel]] (channel coefficients of one link)
MD!*/

py::list ssf_rays(py::handle scenpar,
                  const py::array_t<double> &tx_pos,
                  const py::array_t<double> &rx_pos,
                  const py::array_t<double> &center_freq,
                  const py::array_t<double> &lsp,
                  const py::array_t<double> &lsp_mu,
                  const py::array_t<double> &lsp_sigma,
                  const py::array_t<double> &delay,
                  const py::array_t<double> &power,
                  const py::array_t<double> &aod,
                  const py::array_t<double> &eod,
                  const py::array_t<double> &aoa,
                  const py::array_t<double> &eoa,
                  const py::array_t<double> &rnd_ray,
                  const std::string &method,
                  arma::uword i_freq,
                  const std::string &angle_model,
                  const std::string &pol_model,
                  py::handle gr_eps,
                  py::handle pol_variability,
                  py::handle random_phase)
{
    // Read inputs
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, "scenpar");
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto center_freq_a = qd_python_numpy2arma_Col<double>(center_freq, true);
    const auto lsp_a = qd_python_numpy2arma_Cube<double>(lsp, true);
    const auto lsp_mu_a = qd_python_numpy2arma_Cube<double>(lsp_mu, true);
    const auto lsp_sigma_a = qd_python_numpy2arma_Cube<double>(lsp_sigma, true);
    const auto delay_a = qd_python_numpy2arma_Cube<double>(delay, true);
    const auto power_a = qd_python_numpy2arma_Cube<double>(power, true);
    const auto aod_a = qd_python_numpy2arma_Cube<double>(aod, true);
    const auto eod_a = qd_python_numpy2arma_Cube<double>(eod, true);
    const auto aoa_a = qd_python_numpy2arma_Cube<double>(aoa, true);
    const auto eoa_a = qd_python_numpy2arma_Cube<double>(eoa, true);
    const auto rnd_ray_a = qd_python_numpy2arma_Cube<double>(rnd_ray, true);
    const bool pol_variability_b = qd_python_flag(pol_variability, false, "pol_variability");
    const bool random_phase_b = qd_python_flag(random_phase, true, "random_phase");

    // Complex permittivity -> real and imaginary part; a real array is cast to complex (imaginary part 0)
    arma::mat gr_eps_re, gr_eps_im;
    if (!gr_eps.is_none())
    {
        const auto gr_eps_c = py::cast<py::array_t<std::complex<double>>>(gr_eps);
        if (gr_eps_c.ndim() > 2)
            throw std::invalid_argument("Input 'gr_eps' must have size (n_link, n_freq).");
        arma::cube re, im;
        qd_python_copy2arma(gr_eps_c, re, im);
        gr_eps_re = arma::mat(re.memptr(), re.n_rows, re.n_cols * re.n_slices);
        gr_eps_im = arma::mat(im.memptr(), im.n_rows, im.n_cols * im.n_slices);
    }
    const arma::mat *p_gr_eps_re = gr_eps.is_none() ? nullptr : &gr_eps_re;
    const arma::mat *p_gr_eps_im = gr_eps.is_none() ? nullptr : &gr_eps_im;

    // Output dicts in numpy memory, sizes from a completed copy of 'scenpar'; with an invalid 'scenpar' the library
    // call throws its message before any output is needed
    std::vector<quadriga_lib::gbsm_rays> rays;
    py::list rays_py;
    auto sp = scenpar_a;
    bool known = true;
    try
    {
        quadriga_lib::scenpar_validate(sp);
    }
    catch (const std::exception &)
    {
        known = false;
    }
    arma::uword G = 0;
    if (known)
    {
        arma::uword N, M, n_ray, n_gen;
        qd_gbsm_ray_layout(sp, N, M, G, n_ray, n_gen);
        const arma::uword n_freq_out = (method == "quadriga") ? center_freq_a.n_elem : 1;
        rays_py = qd_python_init_gbsm_rays(rx_pos_a.n_rows, n_ray, n_freq_out, G == 1, false, rays);
    }

    quadriga_lib::ssf_rays(scenpar_a, tx_pos_a, rx_pos_a, center_freq_a, lsp_a, lsp_mu_a, lsp_sigma_a, delay_a,
                           power_a, aod_a, eod_a, aoa_a, eoa_a, rnd_ray_a, rays, method, i_freq, angle_model, pol_model,
                           p_gr_eps_re, p_gr_eps_im, pol_variability_b, random_phase_b);

    // The method string and the complex permittivity cannot be aliased
    for (size_t l = 0; l < rays.size(); ++l)
    {
        py::dict d = rays_py[l];
        d["method"] = rays[l].method;
        if (G == 1)
            d["gr_eps"] = qd_python_copy2numpy<double, std::complex<double>>(&rays[l].gr_eps_re, &rays[l].gr_eps_im);
    }
    return rays_py;
}

// pybind11 declaration:
// m.def("ssf_rays", &ssf_rays,
//       py::arg("scenpar"),
//       py::arg("tx_pos").none(false),
//       py::arg("rx_pos").none(false),
//       py::arg("center_freq").none(false),
//       py::arg("lsp").none(false),
//       py::arg("lsp_mu").none(false),
//       py::arg("lsp_sigma").none(false),
//       py::arg("delay").none(false),
//       py::arg("power").none(false),
//       py::arg("aod").none(false),
//       py::arg("eod").none(false),
//       py::arg("aoa").none(false),
//       py::arg("eoa").none(false),
//       py::arg("rnd_ray").none(false),
//       py::arg("method") = "38.901",
//       py::arg("i_freq") = 0,
//       py::arg("angle_model") = "additive",
//       py::arg("pol_model") = "38.901",
//       py::arg("gr_eps") = py::none(),
//       py::arg("pol_variability") = false,
//       py::arg("random_phase") = true);
