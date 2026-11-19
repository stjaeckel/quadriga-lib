// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"
#include "qd_helpers_gbsm.hpp"



/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_clusters
Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7)

- Turns the LSPs of [[lsp_generate]] and the cluster random values of [[gbsm.draw]] into the clusters of each link, by
  one of three methods:
  - `"38.901"` (default): TR 38.901 clause 7.5 steps 5 to 7 at one frequency, for drop-based calibration. Clusters
    25 dB below the strongest get power 0 (no renormalization); the K-factor in the scaling factors is
    limited to >= -5 dB. Under spatial consistency (clause 7.6.3.1) evaluate the blocks 1 to 8 of `rnd` at the drop
    or segment start and keep them along the track
  - `"38.901_MF"`: TR 38.901 clause 7.6.5.1 as written, with the anchor LSPs `lsp_anchor`; where the multi-frequency
    method of the TR is required
  - `"quadriga"`: S. Jaeckel et al., "A Spatially Consistent Geometric D2D Small-Scale Fading Model for Multiple
    Frequencies", 2019, with the LOS direction added to the cluster offsets; for tracks, several frequencies and
    device-to-device links; delays and angles depend on the frequency list, so keep it fixed to reproduce a seed

| Property | `"38.901"` | `"38.901_MF"` | `"quadriga"` |
| --- | --- | --- | --- |
| Continuous along tracks | no | NLOS only | yes |
| Same delays and angles at all frequencies | n/a | NLOS, or K the same at all frequencies | yes |
| Output spreads equal the input spreads | no | no | DS, and AS below the caps, at one frequency without ground reflection |
| Per-cluster shadowing | yes | yes, per frequency | no |

- `n_slot = NumClusters + G` slots per link (see Terms), `G = 1` if `GR_enabled > 0`. Slot `1 + G + n` belongs to
  column `n` of each block of `rnd` (no sorting). Clusters `n >= n_cluster[l]` (TR clause 7.6.15) get delay 0, power 0
  and the LOS angles. Identify the LOS path by its slot: NLOS clusters of the TR methods can have delay 0
- A scenario is LOS if `KF_mu > -30` dB. In an NLOS scenario the LOS and ground-reflection powers are 0 whatever `lsp`
  holds (`NumClusters > 1`); legacy QuaDRiGa "NLOS" files with `KF_mu > -30` run as LOS
- Delays in s relative to the LOS delay `d3D/c`; slot 0 holds the geometric LOS direction. Powers are the coefficient
  powers of TR eq. (7.5-30), LOS `K/(K+1)`, NLOS clusters `P_n/(K+1)`, without path loss and shadow fading
- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: `"38.901"`, `"38.901_MF"` add `abs_toa[l]` to the NLOS delays;
  `"quadriga"` places the first NLOS cluster at `abs_toa[l]` (where `absTOA_mu <= -30`: only where `abs_toa[l] > 0`)
- Ground reflection (TR clause 7.6.8) on the plane z = 0: slot power `K/(K+1)·(d3D/d_GR)²·R²` (TR methods, on top of
  the other slots) or `K/(K+1)·R²` (`"quadriga"`), with `R²` the mean of `abs(R_par)²` and `abs(R_perp)²` of TR eqs.
  (7.6-38), (7.6-39); the coefficient is `sqrt(power)·diag(R_par, -R_perp)/R`, without polarization
  `sqrt(power)·exp(j·arg(R_par))`
- `NumClusters = 1`: the LOS path only, power 1 in all scenarios, plus the ground reflection if enabled; `rnd` must be
  empty, `rnd_freq` and `lsp_anchor` are not used
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario; only their sizes are checked

## Usage:
```
delay, power, aod, eod, aoa, eoa, n_active = quadriga_lib.gbsm.ssf_clusters( scenpar, tx_pos, rx_pos, \
    lsp, rnd, *, method, rnd_freq, lsp_anchor, eod_offset_deg, abs_toa, gr_eps, o2i, n_cluster )
```

## Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumClusters_min`, `KF_mu`, `absTOA_mu`, `r_DS`, `LNS_ksi`,
  `GR_enabled`
- **`tx_pos`** — Tx positions in m; z >= 0 with ground reflection; `(3,)` or `(3, n_link)`
- **`rx_pos`** — Rx positions in m; z >= 0 with ground reflection; `(3, n_link)`
- **`lsp`** — LSPs from [[lsp_generate]], slices and units as there; DS, ASD, ASA, ESD, ESA (slices 0, 3 to 6) > 0,
  KF (slice 1) >= 0; `"38.901"`: one frequency, call once per frequency; `(n_link, n_freq, 8)`
- **`rnd`** — Shared cluster set of [[gbsm.draw]] (`"cluster"`, `set_id = -1`), columns 0 to N-1 in [0, 1];
  `(n_link, 9·N)`
- **`method`** — `"38.901"`, `"38.901_MF"` or `"quadriga"`; default: `"38.901"`
- **`rnd_freq`** — Per-frequency cluster sets of [[gbsm.draw]], slice `k` from `set_id = k` (`"38.901"`: one slice,
  the set of the frequency in `lsp`, also as `(n_link, N)`); required by the TR methods; `(n_link, N, n_freq)`;
  default: None
- **`lsp_anchor`** — LSPs at the anchor frequency (e.g. 2 GHz) from [[lsp_generate]]; required by `"38.901_MF"`;
  `(n_link, 1, 8)`; default: None
- **`eod_offset_deg`** — EoD offset in deg from [[lsp_generate]]; used by `"38.901"` and `"quadriga"` in NLOS
  scenarios; `None` = 0; `(n_link, n_freq)`; default: None
- **`abs_toa`** — Absolute TOA offset in s from [[lsp_generate]], >= 0; required by `"quadriga"` in NLOS scenarios
  with `absTOA_mu > -30`; `None` = no offset; `(n_link,)`; default: None
- **`gr_eps`** — Ground permittivity as [[lsp_generate]] returns it (complex, real part >= 1, imaginary part >= 0,
  used as `real - j·imag`; a real array = imaginary part 0); required if `GR_enabled > 0`; `(n_link, n_freq)`;
  default: None
- **`o2i`** — O2I flag per link (mean arrival elevation 0), bool or numeric array (nonzero = O2I); used by `"38.901"`
  in NLOS scenarios; `None` = none; `(n_link,)`; default: None
- **`n_cluster`** — Active NLOS clusters per link in `[min(max(1, NumClusters_min - 1), N), N]`, int or float
  array of non-negative integers; `None` = N; `(n_link,)`; default: None

## Outputs:
- **`delay`** — Delays in s; `(n_link, n_slot, n_freq)` for `"38.901_MF"`, else `(n_link, n_slot, 1)`
- **`power`** — Coefficient powers (linear); `(n_link, n_slot, n_freq)`
- **`aod`**, **`eod`** — Azimuth in [-pi, pi) and elevation in [-pi/2, pi/2] of departure in rad; shapes as `delay`
- **`aoa`**, **`eoa`** — Azimuth and elevation of arrival in rad, ranges as departure; shapes as `delay`
- **`n_active`** — Number of NLOS clusters with power > 0; uint32; `(n_link,)`

## See also:
- [[gbsm.draw]] (the random values `rnd` and `rnd_freq`)
- [[lsp_generate]] (the LSPs, EoD offset, absolute TOA and ground permittivity)
- [[ssf_rays]] (the rays of the clusters)
- [[calc_delay_spread]], [[calc_angular_spread]] (spreads of the output)
MD!*/

py::tuple ssf_clusters(py::handle scenpar,
                       const py::array_t<double> &tx_pos,
                       const py::array_t<double> &rx_pos,
                       const py::array_t<double> &lsp,
                       const py::array_t<double> &rnd,
                       const std::string &method,
                       py::handle rnd_freq,
                       py::handle lsp_anchor,
                       py::handle eod_offset_deg,
                       py::handle abs_toa,
                       py::handle gr_eps,
                       py::handle o2i,
                       py::handle n_cluster)
{
    // Read inputs
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, "scenpar");
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto lsp_a = qd_python_numpy2arma_Cube<double>(lsp, true);
    const auto rnd_a = qd_python_numpy2arma_Mat<double>(rnd, true);
    const auto rnd_freq_a = qd_python_numpy2arma_Cube<double>(rnd_freq, true);
    const auto lsp_anchor_a = qd_python_numpy2arma_Cube<double>(lsp_anchor, true);
    const auto eod_offset_deg_a = qd_python_numpy2arma_Mat<double>(eod_offset_deg, true);
    const auto abs_toa_a = qd_python_numpy2arma_Col<double>(abs_toa, true);
    const auto o2i_in = qd_python_numpy2arma_Col<double>(o2i, true);
    const arma::u32_vec n_cluster_a = qd_python_u32_entries(n_cluster, "n_cluster");

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

    // O2I flags: nonzero = true
    std::vector<bool> o2i_a(o2i_in.n_elem);
    for (arma::uword i = 0; i < o2i_in.n_elem; ++i)
        o2i_a[i] = o2i_in[i] != 0.0;


    // Output allocation in numpy memory, sizes from a completed copy of 'scenpar'; with an invalid 'scenpar' the call
    // throws before any output is needed
    arma::cube delay, power, aod, eod, aoa, eoa;
    arma::u32_vec n_active;
    py::array_t<double> delay_py, power_py, aod_py, eod_py, aoa_py, eoa_py;
    py::array_t<unsigned> n_active_py;
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
    if (known)
    {
        const arma::uword n_link = qd_gbsm_n_link(rx_pos_a), n_freq = lsp_a.n_cols;
        const arma::uword n_slot = (arma::uword)sp.at("NumClusters") + (sp.at("GR_enabled") > 0.0 ? 1 : 0);
        const arma::uword n_fd = (method == "38.901_MF") ? n_freq : 1;
        delay_py = qd_python_init_output(n_link, n_slot, n_fd, &delay);
        power_py = qd_python_init_output(n_link, n_slot, n_freq, &power);
        aod_py = qd_python_init_output(n_link, n_slot, n_fd, &aod);
        eod_py = qd_python_init_output(n_link, n_slot, n_fd, &eod);
        aoa_py = qd_python_init_output(n_link, n_slot, n_fd, &aoa);
        eoa_py = qd_python_init_output(n_link, n_slot, n_fd, &eoa);
        n_active_py = qd_python_init_output(n_link, &n_active);
    }

    // Call library function
    quadriga_lib::ssf_clusters(scenpar_a, tx_pos_a, rx_pos_a, lsp_a, rnd_a, delay, power, aod, eod, aoa, eoa, method,
                               rnd_freq_a, lsp_anchor_a, eod_offset_deg_a, abs_toa_a, gr_eps_re, gr_eps_im, o2i_a,
                               n_cluster_a, &n_active);

    return py::make_tuple(delay_py, power_py, aod_py, eod_py, aoa_py, eoa_py, n_active_py);
}

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
- LOS and GR rays: delay, angles and power of their slot; Jones matrix diag(1, -1) (LOS), GR as in [[ssf_clusters]]
- NLOS rays: power P_n / `NumSubPaths`; offsets of Table 7.5-3 scaled by the cluster spreads `PerClusterAS_D`,
  `PerClusterAS_A`, `PerClusterES_A` and the cluster ESD of eq. (7.5-20), `PerClusterES_D_scale`·10^(`lsp_mu` of
  ESD), or `PerClusterES_D` if the scale is 0; coupling (step 8): within a cluster or sub-cluster, the ray with the k-th
  smallest value of a quantity's rank block gets the k-th smallest offset, separately for AOD, AOA, EOD and EOA
- c_DS, the cluster DS of Table 7.5-6: `max(PerClusterDS_min, PerClusterDS + PerClusterDS_gamma·lg f')` ns with
  `f' = max(center_freq/1e9, PerClusterDS_fc_min)`; 3.91 ns where `PerClusterDS` = 0
- `NumSubPaths` other than 20: the offsets of QuaDRiGa-NG (the MATLAB QuaDRiGa fork that uses this library) below 20
  (mean 0, rms 1), Laplacian quantiles at the rms of Table 7.5-3 above 20; no sub-clusters

| Method | `"38.901"`, `"38.901_MF"` | `"quadriga"` |
| --- | --- | --- |
| Frequencies per call | `i_freq` only; slice `i_freq` of `rnd_ray`; `"38.901"` takes one frequency | all; one geometry, per frequency powers, XPR and Jones matrices; the shared `"ray"` set |
| Ray delays | cluster delay; sub-clusters + 0, 1.28, 2.56 c_DS (Table 7.5-5) | cluster delay + 2·c_DS·(u - mean of u in the cluster), u of block 14; negative delays set to 0 |
| Taps | sub-cluster of the two strongest clusters (`NumSubPaths` = 20), else cluster | cluster |
| c_DS, cluster ESD | at frequency `i_freq` | mean over the frequencies |

| Option | Default (TR) | Alternative (QuaDRiGa-NG) |
| --- | --- | --- |
| `angle_model` | `"additive"`: offsets added to azimuth and elevation, elevation folded at the poles (step 7) | `"rotation"`: offset pattern rotated to the cluster direction; equal at elevation 0 |
| `pol_model` | `"38.901"`: XPR and four initial phases per ray (steps 9, 10) | `"quadriga"`: rotation polarization model of QuaDRiGa, one phase, XPR per ray and per cluster |
| `pol_variability` | `False` | `True`: clause 7.6.16 on the NLOS rays with std `PolVar_sigma` (key default 0: no effect); `pol_model = "38.901"` only |

- Sub-clusters (step 11): the two clusters of largest power at frequency `i_freq` (equal powers: lower index first),
  rays at the positions 0-9, 10-15 and 16-19 (TR rays 1-8, 19, 20; 9-12, 17, 18; 13-16); every tap is a contiguous
  block of rays
- `pol_model = "quadriga"`: the matrix refers to the end with the larger z, then x, then y (coordinates within 1 mm
  count as equal; the Tx if all tie), so UTs dropped separately at equal heights may get transposed matrices

## Ray dict:
One link at one drop. All keys are present on output (empty where empty). As input, missing optional keys are empty,
other missing or unknown keys raise `ValueError`; integer keys accept any real dtype. The dict has scatterers when
`fbs_pos` has `n_ray` columns (a dict with no rays counts as having them). Rays added from another source take tap
indices the dict does not use; at most one LOS and one GR ray and one `method` for all rays. [[ssf_scatterers]]
overwrites given scatterers, so solve rays without scatterers separately and concatenate the arrays.

| Key | Shape | Type | Meaning | Required |
| --- | --- | --- | --- | --- |
| `method` | — | `str` | `"38.901"`, `"38.901_MF"` or `"quadriga"`; GR power law and default mobility | yes |
| `tx_pos`, `rx_pos` | `(3,)` | float | Tx and Rx position the ray data refer to in m | yes |
| `center_freq` | `(n_freq,)` | float | Carrier frequencies in Hz | yes |
| `delay` | `(n_ray,)` | float | Delays in s relative to the LOS delay d3D / c, d3D the Tx-Rx distance | yes |
| `power` | `(n_ray, n_freq)` | float | Coefficient powers (linear) without link gain; data from paths: path gain | yes |
| `aod`, `eod`, `aoa`, `eoa` | `(n_ray,)` | float | Departure and arrival azimuth and elevation in rad, GCS | yes |
| `polmat` | `(8, n_ray, n_freq)` | float | Polarization transfer matrices incl. initial phases, real, rows as in [[channel.planar]] | yes |
| `path_type` | `(n_ray,)` | uint32 | 0 = LOS, 1 = GR, 2 = NLOS | yes |
| `tap` | `(n_ray,)` | uint32 | Tap index; rays with equal index form one tap | yes |
| `gr_eps` | `(n_freq,)` or empty | complex | Ground permittivity, used as `real - j·imag`; needed where a GR ray is recomputed | no |
| `fbs_pos`, `lbs_pos` | `(3, n_ray)` or empty | float | First- and last-bounce scatterer positions in m | no |
| `fallback` | `(n_ray,)` or empty | uint32 | 1 = ray lengthened by the solver fallback | no |

## Usage:
```
rays = quadriga_lib.gbsm.ssf_rays( scenpar, tx_pos, rx_pos, center_freq, lsp, lsp_mu, lsp_sigma, \
    delay, power, aod, eod, aoa, eoa, rnd_ray, *, method, i_freq, angle_model, pol_model, gr_eps, \
    pol_variability, random_phase )
```

## Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumSubPaths`, the `PerCluster*` keys, `GR_enabled`,
  `PolVar_sigma`
- **`tx_pos`** — Tx positions in m; `(3,)` or `(3, n_link)`
- **`rx_pos`** — Rx positions in m; `(3, n_link)`
- **`center_freq`** — Carrier frequencies in Hz; `(n_freq,)`
- **`lsp`**, **`lsp_mu`**, **`lsp_sigma`** — LSPs, log-domain means and std from [[lsp_generate]], slices and units as
  there; used: the ESD mean and the XPR mean and std, the XPR LSP for `pol_model = "quadriga"`; `(n_link, n_freq, 8)`
- **`delay`**, **`power`**, **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Cluster delays, powers and angles, outputs 0 to
  5 of [[ssf_clusters]] in this order (`n_slot` slots); `power` `(n_link, n_slot, n_freq)`, the others
  `(n_link, n_slot, n_freq)` for `"38.901_MF"`, else `(n_link, n_slot, 1)`
- **`rnd_ray`** — `"ray"` sets from [[gbsm.draw]] (`n_gen` columns); TR methods: slice `k` = set `k`,
  `(n_link, n_gen, n_freq)`, one set also `(n_link, n_gen)`; `"quadriga"`: the shared set (`set_id = -1`),
  `(n_link, n_gen)` or `(n_link, n_gen, 1)`
- **`method`** — Method of [[ssf_clusters]]: `"38.901"`, `"38.901_MF"` or `"quadriga"`; default: `"38.901"`
- **`i_freq`** — TR methods: index into `center_freq`, 0-based; `"quadriga"`: must be 0; default: 0
- **`angle_model`**, **`pol_model`** — See the option table; default: `"additive"`, `"38.901"`
- **`gr_eps`** — Ground permittivity as [[lsp_generate]] returns it (complex); required if `GR_enabled`, else
  ignored; `(n_link, n_freq)`; default: None
- **`pol_variability`** — TR 7.6.16 polarization power variability, see the option table; default: `False`
- **`random_phase`** — `False`: all initial phases 0; default: `True`

## Output:
- **`rays`** — Ray dicts, one per link, TR methods at frequency `i_freq` only; `list` of `n_link` dicts

## See also:
- [[ssf_clusters]] (the clusters)
- [[ssf_scatterers]] (scatterer positions and validity scales of one link)
- [[ssf_paths]] (paths of one link)
- [[gbsm.single_link_channel]] (channel coefficients from a ray dict)
- [[gbsm.rays_from_paths]] (ray dict from path data)
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
        rays_py = qd_python_init_gbsm_rays(qd_gbsm_n_link(rx_pos_a), n_ray, n_freq_out, G == 1, false, rays);
    }

    quadriga_lib::ssf_rays(scenpar_a, tx_pos_a, rx_pos_a, center_freq_a, lsp_a, lsp_mu_a, lsp_sigma_a, delay_a,
                           power_a, aod_a, eod_a, aoa_a, eoa_a, rnd_ray_a, rays, method, i_freq, angle_model, pol_model,
                           gr_eps_re, gr_eps_im, pol_variability_b, random_phase_b);

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

/*!MD
# ssf_paths
Antenna-free paths of one link at one position

- Computes the antenna-free paths (gains, lengths, polarization matrices, angles and scatterers) of the kept rays of
  `rays` (see Terms) at new Tx and Rx positions (default: the ray-data positions), LOS and GR also at zero power; paths
  in ray order, `ray_ind` maps them to the rays (e.g. `rays["tap"][ray_ind]`)
- `"phase"`: use the outputs with [[channel.planar]] and the arrays at the positions of `rays`; the displacement
  phase of TR 38.901 eq. (7.5-22) is in `polmat`. With the arrays at the new positions the LOS path is no longer identified
  and the displacement phase counts twice, without an error
- `"drift"`: use the outputs at the new positions with [[channel.spherical]] or [[channel.multifreq]]; needs
  the scatterers of [[ssf_scatterers]]

| Quantity | `"phase"` | `"drift"` |
| --- | --- | --- |
| Path length | Tx-Rx distance of `rays` + c·delay | Polyline Tx, FBS, LBS, Rx; LOS direct line; GR via the specular point |
| Angles | Ray data | Geometry: towards FBS and LBS, LOS direct line, GR towards the image points |
| `polmat` | Ray `polmat` times the displacement phase | Ray `polmat`; GR recomputed when an end moved |
| `fbs_pos`, `lbs_pos` | `(3, 0)` | Scatterers of the rays; LOS midpoint of the direct line; GR specular point |

- `"drift"` with an end moved (any coordinate differs from `rays`): GR power by the power law of the method of `rays`
  (`(d3D / d_GR)²` for the TR methods, 1 for `"quadriga"`), GR `polmat` the scalar multiple of `diag(R_par, -R_perp)` (TR
  38.901 eqs. (7.6-38), (7.6-39)) that best fits the stored matrix, which replaces a matrix of another form (e.g. a
  conjugated one). Fallback rays have their longer polyline, also at the positions of `rays`
- Path gain per frequency: the coefficient power; with `kf_ratio` LOS and GR are rescaled against NLOS so that the
  K-factor of the coefficient powers changes by `kf_ratio`, LOS + NLOS keep their sum; times `10^(link_gain_dB / 10)`;
  with `oxygen` times the absorption of TR 38.901 clause 7.6.1 (Table 7.6.1-1 at each frequency) over each path's own
  length, without the delay offset τ_Δ of eq. (7.6-1) and at the carrier only (no in-band variation of eq. (7.6-2))
- Validity per moving end, with `b_eff` and `b_med` of [[ssf_scatterers]]: `"phase"` up to about `sqrt(λ · b_eff / 8)`,
  typically 0.1 to 1 m at the carrier, about 0.15 to 0.35 m at the edges of a 400 MHz band; `"drift"` up to about
  `0.2 · b_med`, typically 1.5 to 12 m, new segment from `0.5 · b_med`; in `"drift"` each path length changes by at most
  the displacement. [[gbsm.single_link_channel]] reports the displacement relative to the limit
- Raises `ValueError` for invalid input; in `"drift"` also without scatterers, and for a GR ray at moved ends without
  permittivity in `rays` or with an end below z = 0

## Usage:
```
path_gain, path_length, polmat, aod, eod, aoa, eoa, fbs_pos, lbs_pos, path_type, ray_ind = \
    quadriga_lib.gbsm.ssf_paths( rays, link_gain_dB, *, mobility, tx_pos, rx_pos, kf_ratio, oxygen )
```

## Inputs:
- **`rays`** — Ray dict of the link, see [[ssf_rays]]
- **`link_gain_dB`** — Link gain (see Terms) at the position in dB, per frequency; 0 for data from paths; `(n_freq,)`
- **`mobility`** — `"phase"`, `"drift"` or `"auto"` (see Terms; `"auto"` is `"drift"` for `"quadriga"` rays, which then
  need [[ssf_scatterers]]); default: `"auto"`
- **`tx_pos`**, **`rx_pos`** — Tx and Rx position in m; `None`: the positions of `rays`; `(3,)`; default: None
- **`kf_ratio`** — KF(position) / KF(drop), >= 0, per frequency; `None`: 1; `(n_freq,)`; default: None
- **`oxygen`** — Oxygen absorption of TR 38.901 clause 7.6.1; default: `False`

## Outputs:
- **`path_gain`** — Path gains (linear); `(n_path, n_freq)`
- **`path_length`** — Path lengths in m; `(n_path,)`
- **`polmat`** — Polarization transfer matrices, real, rows as in [[channel.planar]]; `(8, n_path, n_freq)`
- **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Departure and arrival angles in rad, GCS; `(n_path,)`
- **`fbs_pos`**, **`lbs_pos`** — Scatterer positions in m; `(3, n_path)`, `(3, 0)` in `"phase"`
- **`path_type`** — Path type, 0 = LOS, 1 = GR, 2 = NLOS; int64; `(n_path,)`
- **`ray_ind`** — Index of the ray in `rays`, 0-based; int64; `(n_path,)`

## See also:
- [[ssf_scatterers]] (scatterers for `"drift"`, validity scales `b_eff` and `b_med`)
- [[gbsm.single_link_channel]] (channel coefficients of one link over a segment)
- [[gbsm.rays_from_paths]] (ray data from path data, e.g. of a QRT file)
MD!*/

py::tuple ssf_paths(const py::dict &rays,
                    const py::array_t<double> &link_gain_dB,
                    const std::string &mobility,
                    py::handle tx_pos,
                    py::handle rx_pos,
                    py::handle kf_ratio,
                    py::handle oxygen)
{
    // Read inputs
    const auto rays_a = qd_python_dict2gbsm_rays(rays, "rays");
    const auto link_gain_dB_a = qd_python_numpy2arma_Col<double>(link_gain_dB, true);
    const auto tx_pos_a = qd_python_numpy2arma_Col<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Col<double>(rx_pos, true);
    const auto kf_ratio_a = qd_python_numpy2arma_Col<double>(kf_ratio, true);
    const bool oxygen_b = qd_python_flag(oxygen, false, "oxygen");

    // Outputs of unknown size (n_path), copied after the call
    arma::mat path_gain, fbs_pos, lbs_pos;
    arma::vec path_length, aod, eod, aoa, eoa;
    arma::cube polmat;
    arma::u32_vec path_type;
    arma::uvec ray_ind;
    quadriga_lib::ssf_paths(rays_a, link_gain_dB_a, path_gain, path_length, polmat, mobility,
                            tx_pos_a, rx_pos_a,
                            kf_ratio_a, &aod, &eod, &aoa, &eoa, &fbs_pos, &lbs_pos,
                            &path_type, &ray_ind, oxygen_b);

    return py::make_tuple(qd_python_copy2numpy(&path_gain), qd_python_copy2numpy(&path_length),
                          qd_python_copy2numpy(&polmat), qd_python_copy2numpy(&aod), qd_python_copy2numpy(&eod),
                          qd_python_copy2numpy(&aoa), qd_python_copy2numpy(&eoa), qd_python_copy2numpy(&fbs_pos),
                          qd_python_copy2numpy(&lbs_pos), qd_python_copy2numpy<unsigned, py::ssize_t>(&path_type),
                          qd_python_copy2numpy<arma::uword, py::ssize_t>(&ray_ind));
}

/*!MD
# ssf_scatterers
First- and last-bounce scatterers of the rays of one link (two-bounce solver)

- Places the scatterers that the mobility `drift` needs and returns the validity scales of [[ssf_paths]]
- NLOS rays with power > 0 at some frequency: FBS on the departure ray and LBS on the arrival ray at equal distances
  from their end, so that the polyline Tx, FBS, LBS, Rx has the length d3D + c·delay (d3D the Tx-Rx distance) and the
  angles are exact
- Where no such pair exists with both distances at least `d_min`, both scatterers are placed at `d_min` along the ray's
  own directions and `fallback` is 1: the polyline is longer than the ray, `delay` keeps its value. With `d_min` = 2 m
  this applies to up to about 23 % of the rays (most in the TR methods without absolute TOA) and lengthens them by up to
  25 ns (median below 5 ns), the DS of a link by up to 3 ns; `b_med` can then equal `d_min`
- LOS ray and NLOS rays with zero power at all frequencies: the midpoint of the direct line; GR ray: the specular point
  on z = 0, which requires both ends at z >= 0
- `rays_out` is a new dict with all keys; it shares the unchanged entries with `rays` as they are (no copy and no
  conversion: a list stays a list, and an in-place edit of `rays_out["delay"]` changes `rays["delay"]`) and has new
  arrays `fbs_pos`, `lbs_pos` and `fallback`; `rays` and its arrays are not changed, also when it already holds
  scatterers
- `b_med`, the scale of `drift`: the lower power-weighted median of the distances Tx to FBS and Rx to LBS over the NLOS
  rays with power > 0, fallback rays included; NaN without NLOS power. `b_eff`, the scale of `phase`:
  `sqrt(sum P / sum(P / b²))` over the rays with power `P > 0`, with `b` the end-to-scatterer distance (NLOS), d3D
  (LOS) or the GR path length (GR)
- One link per call: `rays[l], b_med, b_eff = quadriga_lib.gbsm.ssf_scatterers(rays[l])`

## Usage:
```
rays_out, b_med, b_eff = quadriga_lib.gbsm.ssf_scatterers( rays, *, d_min )
```

## Inputs:
- **`rays`** — Ray dict of one link, see [[ssf_rays]]
- **`d_min`** — Minimum distance of the scatterers to both ends in m, finite and > 0; default: 2.0

## Outputs:
- **`rays_out`** — Ray dict with `fbs_pos`, `lbs_pos` `(3, n_ray)` and `fallback` `(n_ray,)` (uint32)
- **`b_med`**, **`b_eff`** — Validity scales in m, rows Tx and Rx, column `k` for frequency `k`; `(2, n_freq)`

## See also:
- [[ssf_rays]] (ray data of a set of links)
- [[ssf_paths]] (mobility `drift` needs the scatterers)
- [[gbsm.single_link_channel]] (reports `b_med`, `b_eff` and the displacement ratios)
MD!*/

py::tuple ssf_scatterers(const py::dict &rays, double d_min)
{
    // The input is viewed (read-only); the solver writes into three new arrays
    auto rays_a = qd_python_dict2gbsm_rays(rays, "rays");
    const arma::uword n_ray = rays_a.n_ray(), n_freq = rays_a.n_freq(), n3 = 3, n2 = 2;
    auto fbs_pos_py = qd_python_init_output(n3, n_ray, &rays_a.fbs_pos);
    auto lbs_pos_py = qd_python_init_output(n3, n_ray, &rays_a.lbs_pos);
    auto fallback_py = qd_python_init_output(n_ray, &rays_a.fallback);
    rays_a.fbs_pos.zeros(), rays_a.lbs_pos.zeros(), rays_a.fallback.zeros(); // numpy memory is not initialised

    quadriga_lib::ssf_scatterers(rays_a, d_min);

    // Validity scales per frequency
    arma::mat b_med, b_eff;
    auto b_med_py = qd_python_init_output(n2, n_freq, &b_med);
    auto b_eff_py = qd_python_init_output(n2, n_freq, &b_eff);
    for (arma::uword k = 0; k < n_freq; ++k)
        b_med.col(k) = rays_a.b_med(k), b_eff.col(k) = rays_a.b_eff(k);

    // New dict with all keys: the input's objects for the unchanged keys (empty where missing), the new solver arrays
    py::dict rays_out;
    for (const char *k : qd_python_gbsm_rays_keys)
    {
        const std::string key = k;
        if (key == "fbs_pos")
            rays_out[k] = fbs_pos_py;
        else if (key == "lbs_pos")
            rays_out[k] = lbs_pos_py;
        else if (key == "fallback")
            rays_out[k] = fallback_py;
        else if (rays.contains(k) && !py::object(rays[k]).is_none())
            rays_out[k] = rays[k];
        else if (key == "gr_eps") // the only other optional key
            rays_out[k] = py::array_t<std::complex<double>>(0);
        else // a key given as None, valid for n_ray = 0: empty with the dtype of the key
            rays_out[k] = (key == "path_type" || key == "tap") ? py::array(py::array_t<unsigned>(0))
                                                               : py::array(py::array_t<double>(0));
    }
    return py::make_tuple(rays_out, b_med_py, b_eff_py);
}
