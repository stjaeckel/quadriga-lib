// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_quadriga_lib_functions.hpp"
#include "quadriga_gbsm_helpers.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# SSF_RAYS
Ray parameters of a set of links at the drop (TR 38.901 clause 7.5 steps 7 to 10, clause 7.6.16)

- Splits the clusters of [[ssf_clusters]] into rays and returns one ray struct per link (fields below): delays,
  coefficient powers, angles, polarization transfer matrices with initial phases, path types and tap indices;
  `fbs_pos`, `lbs_pos` and `fallback` stay empty (see [[ssf_scatterers]])
- A pure function of the clusters, the LSPs and the random values, without a seed: a link does not depend on the other
  links of the call
- Ray layout per link, fixed per scenario: ray 1 LOS, 2 GR (if `GR_enabled`), then ray `m` of NLOS cluster `n` (both
  from 1) at `1 + G + (n - 1)·M + m` (`G` = 1 with GR, `M` = `NumSubPaths`); clusters at power 0 keep their rays at
  power 0
- LOS and GR rays: delay, angles and power of their slot; Jones matrix diag(1, -1) (LOS), GR as in [[ssf_clusters]]
- NLOS rays: power P_n / `NumSubPaths`; offsets of Table 7.5-3 scaled by the cluster spreads `PerClusterAS_D`,
  `PerClusterAS_A`, `PerClusterES_A` and the cluster ESD of eq. (7.5-20), `PerClusterES_D_scale`·10^(`lsp_mu` of
  ESD), or `PerClusterES_D` if the scale is 0; coupling (step 8): within a cluster or sub-cluster, the ray with the k-th
  smallest value of a quantity's rank block gets the k-th smallest offset, separately for AOD, AOA, EOD and EOA
- c_DS, the cluster DS of Table 7.5-6: `max(PerClusterDS_min, PerClusterDS + PerClusterDS_gamma·lg f')` ns with
  `f' = max(center_freq/1e9, PerClusterDS_fc_min)`; 3.91 ns where `PerClusterDS` = 0
- `NumSubPaths` other than 20: the offsets of QuaDRiGa-NG (the MATLAB QuaDRiGa fork that uses this library) below 20
  (mean 0, rms 1), Laplacian quantiles at the rms of Table 7.5-3 above 20; no sub-clusters

| Method | `'38.901'`, `'38.901_MF'` | `'quadriga'` |
| --- | --- | --- |
| Frequencies per call | `i_freq` only; slice `i_freq` of `rnd_ray`; `'38.901'` takes one frequency | all; one geometry, per frequency powers, XPR and Jones matrices; the shared `'ray'` set |
| Ray delays | cluster delay; sub-clusters + 0, 1.28, 2.56 c_DS (Table 7.5-5) | cluster delay + 2·c_DS·(u - mean of u in the cluster), u of block 14; negative delays set to 0 |
| Taps | sub-cluster of the two strongest clusters (`NumSubPaths` = 20), else cluster | cluster |
| c_DS, cluster ESD | at frequency `i_freq` | mean over the frequencies |

| Option | Default (TR) | Alternative (QuaDRiGa-NG) |
| --- | --- | --- |
| `angle_model` | `'additive'`: offsets added to azimuth and elevation, elevation folded at the poles (step 7) | `'rotation'`: offset pattern rotated to the cluster direction; equal at elevation 0 |
| `pol_model` | `'38.901'`: XPR and four initial phases per ray (steps 9, 10) | `'quadriga'`: rotation polarization model of QuaDRiGa, one phase, XPR per ray and per cluster |
| `pol_variability` | `false` | `true`: clause 7.6.16 on the NLOS rays with std `PolVar_sigma` (field default 0: no effect); `pol_model = '38.901'` only |

- Sub-clusters (step 11): the two clusters of largest power at frequency `i_freq` (equal powers: lower index first),
  rays at the positions 1-10, 11-16 and 17-20 (TR rays 1-8, 19, 20; 9-12, 17, 18; 13-16); every tap is a contiguous
  block of rays
- `pol_model = 'quadriga'`: the matrix refers to the end with the larger z, then x, then y (coordinates within 1 mm
  count as equal; the Tx if all tie), so UTs dropped separately at equal heights may get transposed matrices

## Ray struct:
One link at one drop. All fields are present on output (empty where empty). As input, missing optional fields are empty,
other missing or unknown fields are an error; integer fields accept any real class. The struct has scatterers when
`fbs_pos` has `n_ray` columns (a struct with no rays counts as having them). Rays added from another source take tap
indices the struct does not use; at most one LOS and one GR ray and one `method` for all rays. [[ssf_scatterers]]
overwrites given scatterers, so solve rays without scatterers separately and concatenate the fields.

| Field | Size | Type | Meaning | Required |
| --- | --- | --- | --- | --- |
| `method` | string | char | `'38.901'`, `'38.901_MF'` or `'quadriga'`; GR power law and default mobility | yes |
| `tx_pos`, `rx_pos` | `[3, 1]` | double | Tx and Rx position the ray data refer to in [m] | yes |
| `center_frequency` | `[n_freq, 1]` | double | Carrier frequencies in [Hz] | yes |
| `delay` | `[n_ray, 1]` | double | Delays in [s] relative to the LOS delay d3D / c, d3D the Tx-Rx distance | yes |
| `power` | `[n_ray, n_freq]` | double | Coefficient powers (linear) without link gain; data from paths: path gain | yes |
| `aod`, `eod`, `aoa`, `eoa` | `[n_ray, 1]` | double | Departure and arrival azimuth and elevation in [rad], GCS | yes |
| `M` | `[8, n_ray, n_freq]` | double | Polarization transfer matrices incl. initial phases, rows as in [[get_channels_planar]] | yes |
| `path_type` | `[n_ray, 1]` | uint32 | 0 = LOS, 1 = GR, 2 = NLOS (a code, not shifted) | yes |
| `tap` | `[n_ray, 1]` | uint32 | Tap index (a code, not shifted); rays with equal index form one tap | yes |
| `gr_eps_re`, `gr_eps_im` | `[n_freq, 1]` or empty | double | Ground permittivity `gr_eps_re - j·gr_eps_im`; needed where a GR ray is recomputed | no |
| `fbs_pos`, `lbs_pos` | `[3, n_ray]` or empty | double | First- and last-bounce scatterer positions in [m] | no |
| `fallback` | `[n_ray, 1]` or empty | uint32 | 1 = ray lengthened by the solver fallback | no |

## Usage:
```
rays = quadriga_lib.ssf_rays( scenpar, tx_pos, rx_pos, center_freq, lsp, lsp_mu, lsp_sigma, delay, ...
    power, aod, eod, aoa, eoa, rnd_ray, method, i_freq, angle_model, pol_model, gr_eps_re, ...
    gr_eps_im, pol_variability, random_phase );
```

## Inputs:
- **`scenpar`** — Scenario parameters; used: `NumClusters`, `NumSubPaths`, the `PerCluster*` fields, `GR_enabled`,
  `PolVar_sigma`
- **`tx_pos`** — Tx positions in [m]; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`center_freq`** — Carrier frequencies in [Hz]; `[n_freq]`
- **`lsp`**, **`lsp_mu`**, **`lsp_sigma`** — LSPs, log-domain means and std from [[lsp_generate]]; used: the ESD mean
  and the XPR mean and std, the XPR LSP for `pol_model = 'quadriga'`; `[n_link, n_freq, 8]`
- **`delay`**, **`power`**, **`aod`**, **`eod`**, **`aoa`**, **`eoa`** — Cluster delays, powers and angles, outputs 1 to
  6 of [[ssf_clusters]] in this order (`n_slot` slots); `power` `[n_link, n_slot, n_freq]`, the others
  `[n_link, n_slot, n_freq]` for `'38.901_MF'`, else `[n_link, n_slot, 1]`
- **`rnd_ray`** — `'ray'` sets from [[gbsm_draw]] (`n_gen` columns); TR methods: slice `k` = set `k`,
  `[n_link, n_gen, n_freq]`; `'quadriga'`: the shared set (`set_id = 0`), `[n_link, n_gen]`
- **`method`** — Method of [[ssf_clusters]]: `'38.901'`, `'38.901_MF'` or `'quadriga'`; default: `'38.901'`
- **`i_freq`** — TR methods: index into `center_freq`, 1-based; `'quadriga'`: must be 1; default: 1
- **`angle_model`**, **`pol_model`** — See the option table; default: `'additive'`, `'38.901'`
- **`gr_eps_re`**, **`gr_eps_im`** *(optional)* — Ground permittivity from [[lsp_generate]]; required if
  `GR_enabled`, else ignored; `[n_link, n_freq]`
- **`pol_variability`** — TR 7.6.16 polarization power variability, see the option table; default: `false`
- **`random_phase`** — `false`: all initial phases 0; default: `true`

## Output:
- **`rays`** — Ray structs, one per link, TR methods at frequency `i_freq` only; `[1, n_link]` struct array

## See also:
- [[ssf_clusters]] (the clusters)
- [[ssf_scatterers]] (scatterer positions and validity scales of one link)
- [[ssf_paths]] (paths of one link)
- [[gbsm_single_link_channel]] (channel coefficients from a ray struct)
- [[gbsm_rays_from_paths]] (ray struct from path data)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 14 || nrhs > 22)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const auto scenpar = qd_mex_struct2scenpar(prhs[0], "scenpar");
    const arma::mat tx_pos = qd_mex_get_Mat<double>(prhs[1], false, "tx_pos");
    const arma::mat rx_pos = qd_mex_get_Mat<double>(prhs[2], false, "rx_pos");
    const arma::vec center_freq = qd_mex_get_Col<double>(prhs[3], false, "center_freq");
    const arma::cube lsp = qd_mex_get_Cube<double>(prhs[4], false, "lsp");
    const arma::cube lsp_mu = qd_mex_get_Cube<double>(prhs[5], false, "lsp_mu");
    const arma::cube lsp_sigma = qd_mex_get_Cube<double>(prhs[6], false, "lsp_sigma");
    const arma::cube delay = qd_mex_get_Cube<double>(prhs[7], false, "delay");
    const arma::cube power = qd_mex_get_Cube<double>(prhs[8], false, "power");
    const arma::cube aod = qd_mex_get_Cube<double>(prhs[9], false, "aod");
    const arma::cube eod = qd_mex_get_Cube<double>(prhs[10], false, "eod");
    const arma::cube aoa = qd_mex_get_Cube<double>(prhs[11], false, "aoa");
    const arma::cube eoa = qd_mex_get_Cube<double>(prhs[12], false, "eoa");
    const arma::cube rnd_ray = qd_mex_get_Cube<double>(prhs[13], false, "rnd_ray");
    const std::string method = (nrhs < 15) ? std::string("38.901") : qd_mex_get_string(prhs[14], "38.901");
    arma::uword i_freq = (nrhs < 16) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[15], "i_freq", 1);
    const std::string angle_model = (nrhs < 17) ? std::string("additive") : qd_mex_get_string(prhs[16], "additive");
    const std::string pol_model = (nrhs < 18) ? std::string("38.901") : qd_mex_get_string(prhs[17], "38.901");
    const arma::mat gr_eps_re = (nrhs < 19) ? arma::mat() : qd_mex_get_Mat<double>(prhs[18], false, "gr_eps_re");
    const arma::mat gr_eps_im = (nrhs < 20) ? arma::mat() : qd_mex_get_Mat<double>(prhs[19], false, "gr_eps_im");
    const bool pol_variability = (nrhs < 21) ? false : qd_mex_get_scalar<bool>(prhs[20], "pol_variability", false);
    const bool random_phase = (nrhs < 22) ? true : qd_mex_get_scalar<bool>(prhs[21], "random_phase", true);

    if (i_freq == 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'i_freq' cannot be 0 (1-based index).");
    i_freq -= 1; // 1-based (MATLAB) -> 0-based (C++)

    // An omitted or empty input selects the default; with zero links an empty input other than [] counts as given
    auto given = [&](int i)
    {
        if (nrhs <= i)
            return false;
        const bool shaped = mxGetM(prhs[i]) != 0 || mxGetN(prhs[i]) != 0; // not [] (0 x 0)
        return !mxIsEmpty(prhs[i]) || (rx_pos.n_rows == 0 && shaped);
    };
    const arma::mat *p_gr_eps_re = given(18) ? &gr_eps_re : nullptr;
    const arma::mat *p_gr_eps_im = given(19) ? &gr_eps_im : nullptr;

    // Output struct array in MATLAB memory, sizes from a completed copy of 'scenpar'; with an invalid 'scenpar' the
    // library call throws its message before any output is needed
    std::vector<quadriga_lib::gbsm_rays> rays;
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
        arma::uword N, M, G, n_ray, n_gen;
        qd_gbsm_ray_layout(sp, N, M, G, n_ray, n_gen);
        const arma::uword n_freq_out = (method == "quadriga") ? center_freq.n_elem : 1;
        plhs[0] = qd_mex_init_gbsm_rays(rx_pos.n_rows, n_ray, n_freq_out, G == 1, false, rays);
    }

    CALL_QD(quadriga_lib::ssf_rays(scenpar, tx_pos, rx_pos, center_freq, lsp, lsp_mu, lsp_sigma, delay, power, aod,
                                   eod, aoa, eoa, rnd_ray, rays, method, i_freq, angle_model, pol_model, p_gr_eps_re,
                                   p_gr_eps_im, pol_variability, random_phase));

    // The method string cannot be aliased
    for (size_t l = 0; l < rays.size(); ++l)
        mxSetField(plhs[0], (mwIndex)l, "method", mxCreateString(rays[l].method.c_str()));
}
