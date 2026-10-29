// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# SSF_CLUSTERS
Cluster delays, powers and angles of a set of links (TR 38.901 clause 7.5 steps 5 to 7)

- Turns the LSPs of [[lsp_generate]] and the random values of [[ssf_draw]] into the clusters of each link, by one of
  three methods (a method × property table is in the C++ documentation of `ssf_clusters`):
  - `'38.901'` (default): TR 38.901 clause 7.5 steps 5 to 7 at one frequency, for drop-based calibration. Clusters
    25 dB below the strongest get power 0 (no renormalisation); the K-factor in the scaling factors is limited to
    >= -5 dB. Under spatial consistency (clause 7.6.3.1) evaluate the blocks 1 to 8 of `rnd` (columns
    N + 1 to 9·N) at the drop or segment start and keep them along the track
  - `'38.901_MF'`: TR 38.901 clause 7.6.5.1 as written, with the anchor LSPs `lsp_anchor`; where the multi-frequency
    method of the TR is required
  - `'quadriga'`: S. Jaeckel et al., "A Spatially Consistent Geometric D2D Small-Scale Fading Model for Multiple
    Frequencies", 2019, with the LOS direction added to the cluster offsets; for tracks, several frequencies and
    device-to-device links
- Slots per link: 1 = LOS path, 2 = ground reflection if `GR_enabled > 0` (`G = 1`), then the `N = NumClusters - 1`
  NLOS clusters; `n_slot = NumClusters + G`. NLOS cluster `n = 1 ... N` is in slot `1 + G + n` and belongs to the
  columns `b·N + n` of `rnd` (no sorting). Clusters `n > n_cluster(l)` (TR clause 7.6.15) get delay 0, power 0 and
  the LOS angles. Identify the LOS path by its slot, never by delay or path length: NLOS clusters of the TR methods
  can have delay 0
- A scenario is LOS if `KF_mu > -30` dB. In an NLOS scenario the LOS and ground-reflection powers are 0 whatever
  `lsp` holds (`NumClusters > 1`); legacy QuaDRiGa "NLOS" files with `KF_mu > -30` run as LOS
- Delays in [s] relative to the LOS delay `d3D/c`; angles in [rad], GCS, elevation convention; slot 1 holds the
  geometric LOS direction. Powers are the coefficient powers of TR eq. (7.5-30), LOS `K/(K+1)`, NLOS clusters
  `P_n/(K+1)`, without path loss and shadow fading
- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: `'38.901'`, `'38.901_MF'` add `abs_toa(l)` to the NLOS
  delays; `'quadriga'` places the first NLOS cluster at `abs_toa(l)` (where `absTOA_mu <= -30`: only where
  `abs_toa(l) > 0`)
- Ground reflection (TR clause 7.6.8) on the plane z = 0. Slot power `K/(K+1)·(d3D/d_GR)²·R²` (TR methods, on top of
  the other slots) or `K/(K+1)·R²` (`'quadriga'`) with `R² = (abs(R_par)² + abs(R_perp)²)/2` of TR eqs. (7.6-38),
  (7.6-39). The coefficient is `sqrt(power)·diag(R_par, -R_perp)/R`, without polarisation
  `sqrt(power)·exp(j·arg(R_par))`
- `NumClusters = 1`: the LOS path only, power 1 in all scenarios (`K/(K+1) = 1`, also in the ground-reflection
  power), plus the ground reflection if enabled; `rnd`, `rnd_freq` and `lsp_anchor` may be empty
- Swap reciprocity: with mirrored random values (dual mobility in [[ssf_draw]]) and mirrored LSPs, swapping `tx_pos`
  and `rx_pos` exchanges the departure and arrival angles (`'38.901'`: `eod_offset_deg` = 0 and no O2I link;
  `'quadriga'`: `eod_offset_deg` = 0)
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario (only sizes are checked); `scenpar` is
  validated by [[scenpar_validate]]. At least one link is required

## Usage:
```
[ delay, power, aod, eod, aoa, eoa, n_active ] = quadriga_lib.ssf_clusters( scenpar, tx_pos, rx_pos, ...
    lsp, rnd, method, rnd_freq, lsp_anchor, eod_offset_deg, abs_toa, gr_eps_re, gr_eps_im, o2i, n_cluster );
```

## Inputs:
- **`scenpar`** — Scenario parameters (struct, `SubpathMethod` ignored); used: `NumClusters`, `NumClusters_min`,
  `KF_mu`, `absTOA_mu`, `r_DS`, `LNS_ksi`, `GR_enabled`
- **`tx_pos`** — Tx positions in [m], one row for all links; z >= 0 with ground reflection; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; z >= 0 with ground reflection; `[n_link, 3]`
- **`lsp`** — LSPs from [[lsp_generate]]; DS, ASD, ASA, ESD, ESA (slices 1, 4 to 7) > 0, KF (slice 2) >= 0;
  `'38.901'`: one frequency; `[n_link, n_freq, 8]`
- **`rnd`** — Shared cluster set of [[ssf_draw]] (`set_id = 0`), columns 1 to N in [0, 1]; `[n_link, 9·N]`
- **`method`** *(optional)* — `'38.901'` (default), `'38.901_MF'` or `'quadriga'`
- **`rnd_freq`** *(optional)* — Per-frequency sets of [[ssf_draw]], slice `k` from `set_id = k`; required by
  `'38.901'` and `'38.901_MF'`; `[n_link, N, n_freq]`
- **`lsp_anchor`** *(optional)* — LSPs at the anchor frequency (e.g. 2 GHz); required by `'38.901_MF'`;
  `[n_link, 1, 8]`
- **`eod_offset_deg`** *(optional)* — EoD offset in [deg] from [[lsp_generate]]; used by `'38.901'` and
  `'quadriga'` in NLOS scenarios; omitted or empty = 0; `[n_link, n_freq]`
- **`abs_toa`** *(optional)* — Absolute TOA offset in [s] from [[lsp_generate]], >= 0; required by `'quadriga'` in
  NLOS scenarios with `absTOA_mu > -30`; omitted or empty = no offset; `[n_link]`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity `gr_eps_re - j·gr_eps_im`, real part >= 1,
  imaginary part >= 0; required if `GR_enabled > 0`; `[n_link, n_freq]`
- **`o2i`** *(optional)* — O2I flag per link (mean ZOA 90 deg), logical or numeric; used by `'38.901'` in NLOS
  scenarios; omitted or empty = none; `[n_link]`
- **`n_cluster`** *(optional)* — Active NLOS clusters per link (a count) in
  `[min(max(1, NumClusters_min - 1), N), N]`; omitted or empty = N; `[n_link]`

## Outputs:
- **`delay`** — Delays in [s]; `[n_link, n_slot, n_fd]`, `n_fd = n_freq` for `'38.901_MF'`, else 1
- **`power`** — Coefficient powers (linear); `[n_link, n_slot, n_freq]`
- **`aod`**, **`aoa`** — Azimuth of departure and arrival in [rad], in [-pi, pi); `[n_link, n_slot, n_fd]`
- **`eod`**, **`eoa`** — Elevation of departure and arrival in [rad], in [-pi/2, pi/2]; `[n_link, n_slot, n_fd]`
- **`n_active`** — Number of NLOS clusters with power > 0; uint64; `[n_link, 1]`

## See also:
- [[ssf_draw]] (the random values `rnd` and `rnd_freq`)
- [[lsp_generate]] (the LSPs, EoD offset, absolute TOA and ground permittivity)
- [[calc_delay_spread]], [[calc_angular_spread]] (spreads of the output)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 5 || nrhs > 14)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs; the field 'SubpathMethod' of the scenario struct is not used
    std::string subpath_method;
    const auto scenpar = qd_mex_struct2scenpar(prhs[0], subpath_method, "scenpar");
    const arma::mat tx_pos = qd_mex_get_Mat<double>(prhs[1]);
    const arma::mat rx_pos = qd_mex_get_Mat<double>(prhs[2]);
    const arma::cube lsp = qd_mex_get_Cube<double>(prhs[3]);
    const arma::mat rnd = qd_mex_get_Mat<double>(prhs[4]);
    const std::string method = (nrhs < 6) ? std::string("38.901") : qd_mex_get_string(prhs[5], "38.901");
    const arma::cube rnd_freq = (nrhs < 7) ? arma::cube() : qd_mex_get_Cube<double>(prhs[6]);
    const arma::cube lsp_anchor = (nrhs < 8) ? arma::cube() : qd_mex_get_Cube<double>(prhs[7]);
    const arma::mat eod_offset_deg = (nrhs < 9) ? arma::mat() : qd_mex_get_Mat<double>(prhs[8]);
    const arma::vec abs_toa = (nrhs < 10) ? arma::vec() : qd_mex_get_Col<double>(prhs[9]);
    const arma::mat gr_eps_re = (nrhs < 11) ? arma::mat() : qd_mex_get_Mat<double>(prhs[10]);
    const arma::mat gr_eps_im = (nrhs < 12) ? arma::mat() : qd_mex_get_Mat<double>(prhs[11]);
    const std::vector<bool> o2i = (nrhs < 13) ? std::vector<bool>() : qd_mex_matlab2vector_Bool(prhs[12]);
    const arma::vec n_cluster_in = (nrhs < 14) ? arma::vec() : qd_mex_get_Col<double>(prhs[13], false, "n_cluster");

    // Cluster numbers are counts (no index shift)
    const arma::uvec n_cluster = qd_mex_uint_entries(n_cluster_in, "n_cluster");

    // Wrap optional pointers; empty arrays select the default
    const arma::cube *p_rnd_freq = rnd_freq.empty() ? nullptr : &rnd_freq;
    const arma::cube *p_lsp_anchor = lsp_anchor.empty() ? nullptr : &lsp_anchor;
    const arma::mat *p_eod_offset_deg = eod_offset_deg.empty() ? nullptr : &eod_offset_deg;
    const arma::vec *p_abs_toa = abs_toa.empty() ? nullptr : &abs_toa;
    const arma::mat *p_gr_eps_re = gr_eps_re.empty() ? nullptr : &gr_eps_re;
    const arma::mat *p_gr_eps_im = gr_eps_im.empty() ? nullptr : &gr_eps_im;
    const std::vector<bool> *p_o2i = o2i.empty() ? nullptr : &o2i;
    const arma::uvec *p_n_cluster = n_cluster.empty() ? nullptr : &n_cluster;

    // Call library function; the output sizes depend on 'scenpar' and 'method', so the outputs are copied afterwards
    arma::cube delay, power, aod, eod, aoa, eoa;
    arma::uvec n_active;
    arma::uvec *p_n_active = (nlhs > 6) ? &n_active : nullptr;
    CALL_QD(quadriga_lib::ssf_clusters(scenpar, tx_pos, rx_pos, lsp, rnd, delay, power, aod, eod, aoa, eoa, method,
                                       p_rnd_freq, p_lsp_anchor, p_eod_offset_deg, p_abs_toa, p_gr_eps_re,
                                       p_gr_eps_im, p_o2i, p_n_cluster, p_n_active));

    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&delay);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&power);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&aod);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&eod);
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&aoa);
    if (nlhs > 5)
        plhs[5] = qd_mex_copy2matlab(&eoa);
    if (nlhs > 6)
        plhs[6] = qd_mex_copy2matlab(&n_active);
}
