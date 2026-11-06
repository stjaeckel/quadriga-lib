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

- Turns the LSPs of [[lsp_generate]] and the cluster random values of [[gbsm_draw]] into the clusters of each link, by
  one of three methods:
  - `'38.901'` (default): TR 38.901 clause 7.5 steps 5 to 7 at one frequency, for drop-based calibration. Clusters
    25 dB below the strongest get power 0 (no renormalization); the K-factor in the scaling factors is
    limited to >= -5 dB. Under spatial consistency (clause 7.6.3.1) evaluate the blocks 1 to 8 of `rnd` (columns
    N + 1 to 9·N) at the drop or segment start and keep them along the track
  - `'38.901_MF'`: TR 38.901 clause 7.6.5.1 as written, with the anchor LSPs `lsp_anchor`; where the multi-frequency
    method of the TR is required
  - `'quadriga'`: S. Jaeckel et al., "A Spatially Consistent Geometric D2D Small-Scale Fading Model for Multiple
    Frequencies", 2019, with the LOS direction added to the cluster offsets; for tracks, several frequencies and
    device-to-device links

| Property | `'38.901'` | `'38.901_MF'` | `'quadriga'` |
| --- | --- | --- | --- |
| Continuous along tracks | no | NLOS only | yes |
| Same delays and angles at all frequencies | n/a | NLOS, or K the same at all frequencies | yes |
| Output spreads equal the input spreads | no | no | DS, and AS below the caps, at one frequency without ground reflection |
| Per-cluster shadowing | yes | yes, per frequency | no |

- `n_slot = NumClusters + G` slots per link (see Terms), `G = 1` if `GR_enabled > 0`. NLOS cluster `n` (from 1) is in
  slot `1 + G + n` and belongs to the columns `b·N + n` of `rnd` (no sorting). Clusters `n > n_cluster(l)` (TR clause
  7.6.15) get delay 0, power 0 and the LOS angles. Identify the LOS path by its slot: NLOS clusters of the TR methods
  can have delay 0
- A scenario is LOS if `KF_mu > -30` dB. In an NLOS scenario the LOS and ground-reflection powers are 0 whatever `lsp`
  holds (`NumClusters > 1`); legacy QuaDRiGa "NLOS" files with `KF_mu > -30` run as LOS
- Delays in [s] relative to the LOS delay `d3D/c`; slot 1 holds the geometric LOS direction. Powers are the coefficient
  powers of TR eq. (7.5-30), LOS `K/(K+1)`, NLOS clusters `P_n/(K+1)`, without path loss and shadow fading
- Absolute TOA (TR clause 7.6.9), NLOS scenarios only: `'38.901'`, `'38.901_MF'` add `abs_toa(l)` to the NLOS delays;
  `'quadriga'` places the first NLOS cluster at `abs_toa(l)` (where `absTOA_mu <= -30`: only where `abs_toa(l) > 0`)
- Ground reflection (TR clause 7.6.8) on the plane z = 0: slot power `K/(K+1)·(d3D/d_GR)²·R²` (TR methods, on top of
  the other slots) or `K/(K+1)·R²` (`'quadriga'`), with `R²` the mean of `abs(R_par)²` and `abs(R_perp)²` of TR eqs.
  (7.6-38), (7.6-39); the coefficient is `sqrt(power)·diag(R_par, -R_perp)/R`, without polarization
  `sqrt(power)·exp(j·arg(R_par))`
- `NumClusters = 1`: the LOS path only, power 1 in all scenarios, plus the ground reflection if enabled; `rnd` must be
  empty, `rnd_freq` and `lsp_anchor` are not used
- `scenpar`, `lsp`, `rnd` and `rnd_freq` must come from the same scenario; only their sizes are checked

## Usage:
```
[ delay, power, aod, eod, aoa, eoa, n_active ] = quadriga_lib.ssf_clusters( scenpar, tx_pos, ...
    rx_pos, lsp, rnd, method, rnd_freq, lsp_anchor, eod_offset_deg, abs_toa, gr_eps_re, gr_eps_im, ...
    o2i, n_cluster );
```

## Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumClusters_min`, `KF_mu`, `absTOA_mu`, `r_DS`, `LNS_ksi`,
  `GR_enabled`
- **`tx_pos`** — Tx positions in [m]; z >= 0 with ground reflection; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; z >= 0 with ground reflection; `[n_link, 3]`
- **`lsp`** — LSPs from [[lsp_generate]]; DS, ASD, ASA, ESD, ESA (slices 1, 4 to 7) > 0, KF (slice 2) >= 0;
  `'38.901'`: one frequency, call once per frequency; `[n_link, n_freq, 8]`
- **`rnd`** — Shared cluster set of [[gbsm_draw]] (`'cluster'`, `set_id = 0`), columns 1 to N in [0, 1];
  `[n_link, 9·N]`
- **`method`** — `'38.901'`, `'38.901_MF'` or `'quadriga'`; default: `'38.901'`
- **`rnd_freq`** *(optional)* — Per-frequency cluster sets of [[gbsm_draw]], slice `k` from `set_id = k` (`'38.901'`:
  one slice, the set of the frequency in `lsp`); required by the TR methods; `[n_link, N, n_freq]`
- **`lsp_anchor`** *(optional)* — LSPs at the anchor frequency (e.g. 2 GHz) from [[lsp_generate]]; required by
  `'38.901_MF'`; `[n_link, 1, 8]`
- **`eod_offset_deg`** — EoD offset in [deg] from [[lsp_generate]]; used by `'38.901'` and `'quadriga'` in NLOS
  scenarios; `[n_link, n_freq]`; default: 0
- **`abs_toa`** — Absolute TOA offset in [s] from [[lsp_generate]], >= 0; required by `'quadriga'` in NLOS scenarios
  with `absTOA_mu > -30`; `[n_link]`; default: no offset
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity `gr_eps_re - j·gr_eps_im` from [[lsp_generate]],
  real part >= 1, imaginary part >= 0; required if `GR_enabled > 0`; `[n_link, n_freq]`
- **`o2i`** — O2I flag per link (mean arrival elevation 0), logical or numeric; used by `'38.901'` in NLOS scenarios;
  `[n_link]`; default: no O2I link
- **`n_cluster`** — Active NLOS clusters per link (a count) in `[min(max(1, NumClusters_min - 1), N), N]`; `[n_link]`;
  default: N

## Outputs:
- **`delay`** — Delays in [s]; `[n_link, n_slot, n_freq]` for `'38.901_MF'`, else `[n_link, n_slot, 1]`
- **`power`** — Coefficient powers (linear); `[n_link, n_slot, n_freq]`
- **`aod`**, **`eod`** — Azimuth in [-pi, pi) and elevation in [-pi/2, pi/2] of departure in [rad]; sizes as `delay`
- **`aoa`**, **`eoa`** — Azimuth and elevation of arrival in [rad], ranges as departure; sizes as `delay`
- **`n_active`** — Number of NLOS clusters with power > 0; uint32; `[n_link, 1]`

## See also:
- [[gbsm_draw]] (the random values `rnd` and `rnd_freq`)
- [[lsp_generate]] (the LSPs, EoD offset, absolute TOA and ground permittivity)
- [[ssf_rays]] (the rays of the clusters)
- [[calc_delay_spread]], [[calc_angular_spread]] (spreads of the output)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 5 || nrhs > 14)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const auto scenpar = qd_mex_struct2scenpar(prhs[0], "scenpar");
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
    const std::vector<bool> o2i = (nrhs < 13) ? std::vector<bool>() : qd_mex_matlab2vector_Bool(prhs[12], "o2i");
    const arma::u32_vec n_cluster =
        (nrhs < 14) ? arma::u32_vec() : qd_mex_get_Col<unsigned>(prhs[13], false, "n_cluster");

    // Wrap optional pointers; an omitted or empty input selects the default; with zero links an empty input other
    // than [], e.g. [0, N], counts as given
    auto given = [&](int i)
    {
        if (nrhs <= i)
            return false;
        const bool shaped = mxGetM(prhs[i]) != 0 || mxGetN(prhs[i]) != 0; // not [] (0 x 0)
        return !mxIsEmpty(prhs[i]) || (rx_pos.n_rows == 0 && shaped);
    };
    const arma::cube *p_rnd_freq = given(6) ? &rnd_freq : nullptr;
    const arma::cube *p_lsp_anchor = given(7) ? &lsp_anchor : nullptr;
    const arma::mat *p_eod_offset_deg = given(8) ? &eod_offset_deg : nullptr;
    const arma::vec *p_abs_toa = given(9) ? &abs_toa : nullptr;
    const arma::mat *p_gr_eps_re = given(10) ? &gr_eps_re : nullptr;
    const arma::mat *p_gr_eps_im = given(11) ? &gr_eps_im : nullptr;
    const std::vector<bool> *p_o2i = given(12) ? &o2i : nullptr;
    const arma::u32_vec *p_n_cluster = given(13) ? &n_cluster : nullptr;

    // Output allocation in MATLAB memory, sizes from a completed copy of 'scenpar'; an invalid 'scenpar' allocates
    // nothing, the library call then throws its message
    arma::cube delay, power, aod, eod, aoa, eoa;
    arma::u32_vec n_active;
    auto sp = scenpar;
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
        const arma::uword n_link = rx_pos.n_rows, n_freq = lsp.n_cols;
        const arma::uword n_slot = (arma::uword)sp.at("NumClusters") + (sp.at("GR_enabled") > 0.0 ? 1 : 0);
        const arma::uword n_fd = (method == "38.901_MF") ? n_freq : 1;
        arma::cube *out[6] = {&delay, &power, &aod, &eod, &aoa, &eoa};
        for (int k = 0; k < 6 && k < nlhs; ++k)
            plhs[k] = qd_mex_init_output(out[k], n_link, n_slot, k == 1 ? n_freq : n_fd);
        if (nlhs > 6)
            plhs[6] = qd_mex_init_output(&n_active, n_link);
    }

    // Call library function
    arma::u32_vec *p_n_active = (nlhs > 6) ? &n_active : nullptr;
    CALL_QD(quadriga_lib::ssf_clusters(scenpar, tx_pos, rx_pos, lsp, rnd, delay, power, aod, eod, aoa, eoa, method,
                                       p_rnd_freq, p_lsp_anchor, p_eod_offset_deg, p_abs_toa, p_gr_eps_re,
                                       p_gr_eps_im, p_o2i, p_n_cluster, p_n_active));
}
