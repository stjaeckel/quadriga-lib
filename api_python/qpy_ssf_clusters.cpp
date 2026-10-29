// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# ssf_clusters
Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7)

- Turns the LSPs of [[lsp_generate]] and the random values of [[ssf_draw]] into the clusters of each link, by one of
  three methods (a method × property table is in the C++ documentation of `ssf_clusters`):
  - `"38.901"` (default): TR 38.901 clause 7.5 steps 5 to 7 at one frequency, for drop-based calibration. Clusters
    25 dB below the strongest get power 0 (no renormalisation); the K-factor in the scaling factors is limited to
    >= -5 dB. Under spatial consistency (clause 7.6.3.1) evaluate the blocks 1 to 8 of `rnd` at the drop or segment
    start and keep them along the track
  - `"38.901_MF"`: TR 38.901 clause 7.6.5.1 as written, with the anchor LSPs `lsp_anchor`; where the multi-frequency
    method of the TR is required
  - `"quadriga"`: S. Jaeckel et al., "A Spatially Consistent Geometric D2D Small-Scale Fading Model for Multiple
    Frequencies", 2019, with the LOS direction added to the cluster offsets; for tracks, several frequencies and
    device-to-device links
- Slots per link: 0 = LOS path, 1 = ground reflection if `GR_enabled > 0` (`G = 1`), then the `N = NumClusters - 1`
  NLOS clusters; `n_slot = NumClusters + G`. Slot `1 + G + n` belongs to column `n` of each block of `rnd` (no
  sorting). Clusters `n >= n_cluster(l)` (TR clause 7.6.15) get delay 0, power 0 and the LOS angles. Zero links give
  outputs with 0 rows. Identify the LOS path by its slot, never by delay or path length: NLOS clusters of the TR
  methods can have delay 0
- A scenario is LOS if `KF_mu > -30` dB. In an NLOS scenario the LOS and ground-reflection powers are 0 whatever
  `lsp` holds (`NumClusters > 1`); legacy QuaDRiGa "NLOS" files with `KF_mu > -30` run as LOS
- Delays in [s] relative to the LOS delay `d3D/c`; angles in [rad], GCS, elevation convention; slot 0 holds the
  geometric LOS direction. Powers are the coefficient powers of TR eq. (7.5-30), LOS `K/(K+1)`, NLOS clusters
  `P_n/(K+1)`, without path loss and shadow fading
- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: `"38.901"`, `"38.901_MF"` add `abs_toa(l)` to the NLOS
  delays; `"quadriga"` places the first NLOS cluster at `abs_toa(l)` (where `absTOA_mu <= -30`: only where
  `abs_toa(l) > 0`)
- Ground reflection (TR clause 7.6.8) on the plane z = 0. Slot power `K/(K+1)·(d3D/d_GR)²·R²` (TR methods, on top of
  the other slots) or `K/(K+1)·R²` (`"quadriga"`) with `R² = (abs(R_par)² + abs(R_perp)²)/2` of TR eqs. (7.6-38),
  (7.6-39). The coefficient is `sqrt(power)·diag(R_par, -R_perp)/R`, without polarisation
  `sqrt(power)·exp(j·arg(R_par))`
- `NumClusters = 1`: the LOS path only, power 1 in all scenarios (`K/(K+1) = 1`, also in the ground-reflection
  power), plus the ground reflection if enabled; `rnd`, `rnd_freq` and `lsp_anchor` may be empty
- Swap reciprocity: with mirrored random values (dual mobility in [[ssf_draw]]) and mirrored LSPs, swapping `tx_pos`
  and `rx_pos` exchanges the departure and arrival angles (`"38.901"`: `eod_offset_deg` = 0 and no O2I link;
  `"quadriga"`: `eod_offset_deg` = 0)
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario (only sizes are checked); `scenpar` is
  validated by [[scenpar_validate]]

## Usage:
```
delay, power, aod, eod, aoa, eoa, n_active = quadriga_lib.gbsm.ssf_clusters( scenpar, tx_pos, rx_pos, lsp, rnd, \
    method, rnd_freq, lsp_anchor, eod_offset_deg, abs_toa, gr_eps, o2i, n_cluster )
```

## Inputs:
- **`scenpar`** — Scenario parameters (dict, `SubpathMethod` ignored); used: `NumClusters`, `NumClusters_min`,
  `KF_mu`, `absTOA_mu`, `r_DS`, `LNS_ksi`, `GR_enabled`
- **`tx_pos`** — Tx positions in m, one row for all links; z >= 0 with ground reflection; `(1, 3)` or `(n_link, 3)`
- **`rx_pos`** — Rx positions in m; z >= 0 with ground reflection; `(n_link, 3)`
- **`lsp`** — LSPs from [[lsp_generate]]; DS, ASD, ASA, ESD, ESA (slices 0, 3 to 6) > 0, KF (slice 1) >= 0;
  `"38.901"`: one frequency; `(n_link, n_freq, 8)`
- **`rnd`** — Shared cluster set of [[ssf_draw]] (`set_id = -1`), columns 0 to N-1 in [0, 1]; `(n_link, 9·N)`
- **`method`** — `"38.901"`, `"38.901_MF"` or `"quadriga"`; default: `"38.901"`
- **`rnd_freq`** — Per-frequency sets of [[ssf_draw]], slice `k` from `set_id = k`; required by `"38.901"` and
  `"38.901_MF"`; `(n_link, N, n_freq)`; default: None
- **`lsp_anchor`** — LSPs at the anchor frequency (e.g. 2 GHz) from [[lsp_generate]]; required by `"38.901_MF"`;
  `(n_link, 1, 8)`; default: None
- **`eod_offset_deg`** — EoD offset in deg from [[lsp_generate]]; used by `"38.901"` and `"quadriga"` in NLOS
  scenarios; `None` = 0; `(n_link, n_freq)`; default: None
- **`abs_toa`** — Absolute TOA offset in s from [[lsp_generate]], >= 0; required by `"quadriga"` in NLOS scenarios
  with `absTOA_mu > -30`; `None` = no offset; `(n_link,)`; default: None
- **`gr_eps`** — Ground permittivity as [[lsp_generate]] returns it (complex, real part >= 1, imaginary part >= 0,
  used as `real - j·imag`; a real array = imaginary part 0); required if `GR_enabled > 0`; `(n_link, n_freq)`;
  default: None
- **`o2i`** — O2I flag per link (mean ZOA 90 deg), bool or numeric array (nonzero = O2I); used by `"38.901"` in
  NLOS scenarios; `None` = none; `(n_link,)`; default: None
- **`n_cluster`** — Active NLOS clusters per link in `[min(max(1, NumClusters_min - 1), N), N]`, int or float
  array of non-negative integers; `None` = N; `(n_link,)`; default: None

## Outputs:
- **`delay`** — Delays in s; `(n_link, n_slot, n_fd)`, `n_fd = n_freq` for `"38.901_MF"`, else 1
- **`power`** — Coefficient powers (linear); `(n_link, n_slot, n_freq)`
- **`aod`**, **`aoa`** — Azimuth of departure and arrival in rad, in [-pi, pi); `(n_link, n_slot, n_fd)`
- **`eod`**, **`eoa`** — Elevation of departure and arrival in rad, in [-pi/2, pi/2]; `(n_link, n_slot, n_fd)`
- **`n_active`** — Number of NLOS clusters with power > 0; uint64; `(n_link,)`

## See also:
- [[ssf_draw]] (the random values `rnd` and `rnd_freq`)
- [[lsp_generate]] (the LSPs, EoD offset, absolute TOA and ground permittivity)
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
    // Read inputs; the entry 'SubpathMethod' of the scenario dict is not used
    std::string subpath_method;
    const auto scenpar_a = qd_python_dict2scenpar(scenpar, subpath_method, "scenpar");
    const auto tx_pos_a = qd_python_numpy2arma_Mat<double>(tx_pos, true);
    const auto rx_pos_a = qd_python_numpy2arma_Mat<double>(rx_pos, true);
    const auto lsp_a = qd_python_numpy2arma_Cube<double>(lsp, true);
    const auto rnd_a = qd_python_numpy2arma_Mat<double>(rnd, true);
    const auto rnd_freq_a = qd_python_numpy2arma_Cube<double>(rnd_freq, true);
    const auto lsp_anchor_a = qd_python_numpy2arma_Cube<double>(lsp_anchor, true);
    const auto eod_offset_deg_a = qd_python_numpy2arma_Mat<double>(eod_offset_deg, true);
    const auto abs_toa_a = qd_python_numpy2arma_Col<double>(abs_toa, true);
    const auto o2i_in = qd_python_numpy2arma_Col<double>(o2i, true);
    const auto n_cluster_in = qd_python_numpy2arma_Col<double>(n_cluster, true);

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

    const arma::uvec n_cluster_a = qd_python_uint_entries(n_cluster_in, "n_cluster");

    // Wrap optional pointers; None selects the default
    const arma::cube *p_rnd_freq = rnd_freq.is_none() ? nullptr : &rnd_freq_a;
    const arma::cube *p_lsp_anchor = lsp_anchor.is_none() ? nullptr : &lsp_anchor_a;
    const arma::mat *p_eod_offset_deg = eod_offset_deg.is_none() ? nullptr : &eod_offset_deg_a;
    const arma::vec *p_abs_toa = abs_toa.is_none() ? nullptr : &abs_toa_a;
    const arma::mat *p_gr_eps_re = gr_eps.is_none() ? nullptr : &gr_eps_re;
    const arma::mat *p_gr_eps_im = gr_eps.is_none() ? nullptr : &gr_eps_im;
    const std::vector<bool> *p_o2i = o2i.is_none() ? nullptr : &o2i_a;
    const arma::uvec *p_n_cluster = n_cluster.is_none() ? nullptr : &n_cluster_a;

    // Call library function; the output sizes depend on 'scenpar' and 'method', so the outputs are copied afterwards
    arma::cube delay, power, aod, eod, aoa, eoa;
    arma::uvec n_active;
    quadriga_lib::ssf_clusters(scenpar_a, tx_pos_a, rx_pos_a, lsp_a, rnd_a, delay, power, aod, eod, aoa, eoa, method,
                               p_rnd_freq, p_lsp_anchor, p_eod_offset_deg, p_abs_toa, p_gr_eps_re, p_gr_eps_im, p_o2i,
                               p_n_cluster, &n_active);

    return py::make_tuple(qd_python_copy2numpy(&delay), qd_python_copy2numpy(&power), qd_python_copy2numpy(&aod),
                          qd_python_copy2numpy(&eod), qd_python_copy2numpy(&aoa), qd_python_copy2numpy(&eoa),
                          qd_python_copy2numpy(&n_active));
}

// pybind11 declaration:
// m.def("ssf_clusters", &ssf_clusters,
//       py::arg("scenpar"),
//       py::arg("tx_pos").none(false),
//       py::arg("rx_pos").none(false),
//       py::arg("lsp").none(false),
//       py::arg("rnd").none(false),
//       py::arg("method") = "38.901",
//       py::arg("rnd_freq") = py::none(),
//       py::arg("lsp_anchor") = py::none(),
//       py::arg("eod_offset_deg") = py::none(),
//       py::arg("abs_toa") = py::none(),
//       py::arg("gr_eps") = py::none(),
//       py::arg("o2i") = py::none(),
//       py::arg("n_cluster") = py::none());
