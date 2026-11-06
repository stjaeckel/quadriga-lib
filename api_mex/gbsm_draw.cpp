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
# GBSM_DRAW
Spatially correlated normalized random values of one random set at a set of links

- Evaluates the SOS phases of [[gbsm_init_phases]] at the links as [[sos_val]], one column per generator. The value of a
  link depends neither on the other links of the call nor on the other columns; with `dual_mobility = false` not on
  `tx_pos`
- `dual_mobility = true`: the Tx is a second device on the same fields; its side of each generator uses the mirror
  generator (e.g. departure and arrival quantities exchange), and every column keeps its distribution
- 2D mode (`horizontal = true`, TR): the values are correlated in the x-y plane; links with equal `floor_id` share one
  field, fields of different floors are uncorrelated at equal horizontal positions (TR 38.901 Table 7.5-6 note 3):
  exactly for IDs in the same group of `n_coeff` consecutive IDs (0 to `n_coeff - 1`, ...), otherwise up to a random
  residual of standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both devices of a link are on the link's floor
- 3D mode (legacy QuaDRiGa): the height separates the floors, so UTs on different floors are correlated by the ACF of
  their height difference. The per-link statistics are the same in both modes
- An empty set (`NumClusters = 1`) gives `[n_link, 0]` and accepts any empty `phase`

Random sets, with `N = NumClusters - 1`, `M = NumSubPaths`, columns from 1 and blocks from 0; a set never changes size
(new random quantities come in new sets), and a decorrelation distance of 0 is used as 0.1 m:

| `set_type`, `set_id` | `n_gen` | Columns | Decorrelation distance | Used by |
| --- | --- | --- | --- | --- |
| `'lsf'` | 11 | 1 to 8 DS, KF, SF, ASD, ASA, ESD, ESA, XPR and 9 absolute TOA, Normal(0,1); 10 ground permittivity and 11 UMa environment height hE, Uniform(0,1) | `DS_lambda` ... `XPR_lambda` (1 to 8), `absTOA_lambda` (9), `SC_lambda` (10), `SF_lambda` (11) | [[lsp_generate]] `rnd`, [[path_loss]] `rnd_he` (column 11) |
| `'cluster'`, 0 | `9·N` | column `b·N + n` for block `b` and NLOS cluster `n` (from 1): block 0 delay, Uniform(0,1); blocks 1 to 8 Normal(0,1): 1 to 4 sign of AOD, AOA, EOD, EOA, 5 to 8 offset of the same | `SC_lambda` | [[ssf_clusters]] `rnd` |
| `'cluster'`, `k` | `N` | cluster shadowing at frequency `k`, Normal(0,1) | `SC_lambda` | [[ssf_clusters]] `rnd_freq`, slice `k` |
| `'ray'`, 0 or `k` | `14·N·M + N` | blocks 0 to 14, below | `SC_lambda`; block 14 of set 0: 1.6·c·c_DS / `PerClusterAS_A` (in rad), within 0.1 m to 10 km | [[ssf_rays]] `rnd_ray`: 0 for `'quadriga'`, `k` as slice `k` for the TR methods |

A `'ray'` set holds block `b` up to 12 for ray `m` of NLOS cluster `n` (both from 1) in column
`b·N·M + (n - 1)·M + m`: 0 to 3 coupling ranks of AOD, AOA, EOD, EOA; 4 XPR; 5 to 8 initial phases θθ, θφ, φθ, φφ; 9 to
12 polarization power variability of the same. Block 13, the XPR per cluster, is in column `13·N·M + n`; block 14, the
ray delay of `'quadriga'`, in column `13·N·M + N + (n - 1)·M + m`. All blocks are Normal(0,1) except block 14,
Uniform(0,1).

## Usage:
```
rnd = quadriga_lib.gbsm_draw( scenpar, set_type, phase, tx_pos, rx_pos, set_id, preset, floor_id, ...
    center_freq, dual_mobility, horizontal );
```

## Inputs:
- **`scenpar`** — Scenario parameters; decorrelation distances, `NumClusters`, `NumSubPaths`, `PerClusterDS*` and
  `PerClusterAS_A`
- **`set_type`** — Set type of [[gbsm_init_phases]]; string
- **`phase`** — SOS phases from [[gbsm_init_phases]], finite; `[n_coeff, n_gen, 2]`
- **`tx_pos`** — Tx positions in [m]; used in dual mobility, always validated; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`, `n_link = 0` allowed
- **`set_id`** — Set of [[gbsm_init_phases]]: `0` = shared set, `k >= 1` = set of frequency `k`; ignored for `'lsf'`;
  another set with the same `n_gen` is not detected; default: 0
- **`preset`** — Preset of [[gbsm_init_phases]]; another preset with the same `n_coeff` is not detected; default:
  `'Comb256'`
- **`floor_id`** — Floor number of the UT per link, ground floor 0 (a label, not shifted; `n_fl - 1` for the UT heights
  of TR 38.901 Table 7.2-1, 0 for outdoor UTs); 2D mode only; `[n_link]`; default: all 0
- **`center_freq`** *(optional)* — Carrier frequencies in [Hz]; required by the shared `'ray'` set, whose block 14 uses
  their mean c_DS ([[ssf_rays]]): pass the same list as to [[ssf_rays]] (another list gives other `'quadriga'` rays);
  ignored and not checked for all other sets; `[n_freq]`
- **`dual_mobility`** — `true`: Tx and Rx both move; default: false
- **`horizontal`** — `true`: 2D mode; `false`: 3D mode; default: true

## Output:
- **`rnd`** — Random values; `[n_link, n_gen]`

## See also:
- [[gbsm_init_phases]] (the SOS phases)
- [[lsp_generate]] (the large-scale parameters from the `'lsf'` set)
- [[path_loss]] (column 11 of the `'lsf'` set as `rnd_he`)
- [[ssf_clusters]] (the clusters from the `'cluster'` sets)
- [[ssf_rays]] (the rays from the `'ray'` sets)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 5 || nrhs > 11)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const auto scenpar = qd_mex_struct2scenpar(prhs[0], "scenpar");
    const std::string set_type = qd_mex_get_string(prhs[1], "", "set_type");
    const arma::cube phase = qd_mex_get_Cube<double>(prhs[2], false, "phase");
    const arma::mat tx_pos = qd_mex_get_Mat<double>(prhs[3], false, "tx_pos");
    const arma::mat rx_pos = qd_mex_get_Mat<double>(prhs[4], false, "rx_pos");
    const double set_id_in = (nrhs < 6) ? 0.0 : qd_mex_get_scalar<double>(prhs[5], "set_id", 0.0);
    const std::string preset = (nrhs < 7) ? std::string("Comb256") : qd_mex_get_string(prhs[6], "Comb256", "preset");
    const arma::u32_vec floor_id = (nrhs < 8) ? arma::u32_vec() : qd_mex_get_Col<unsigned>(prhs[7], false, "floor_id");
    const arma::vec center_freq = (nrhs < 9) ? arma::vec() : qd_mex_get_Col<double>(prhs[8], false, "center_freq");
    const bool dual_mobility = (nrhs < 10) ? false : qd_mex_get_scalar<bool>(prhs[9], "dual_mobility", false);
    const bool horizontal = (nrhs < 11) ? true : qd_mex_get_scalar<bool>(prhs[10], "horizontal", true);

    // Set index: MATLAB 0 = shared set, k >= 1 = frequency k (1-based); C++ -1 and k - 1; ignored for "lsf"
    const bool lsf = set_type == "lsf";
    if (!lsf && !(set_id_in >= 0.0 && set_id_in < 9007199254740992.0 && set_id_in == std::floor(set_id_in)))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'set_id' must be 0 (shared set) or a frequency index >= 1.");
    const arma::sword set_id = lsf ? -1 : (arma::sword)set_id_in - 1;

    // Wrap optional pointers
    const arma::u32_vec *p_floor_id = floor_id.empty() ? nullptr : &floor_id;
    const arma::vec *p_center_freq = center_freq.empty() ? nullptr : &center_freq;

    // Output allocation in MATLAB memory: [n_link, n_gen], n_gen = columns of 'phase' (checked by the library)
    arma::mat rnd;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&rnd, rx_pos.n_rows, phase.n_cols);

    // Call library function
    CALL_QD(quadriga_lib::gbsm_draw(scenpar, set_type, phase, tx_pos, rx_pos, rnd, set_id, preset, p_floor_id,
                                    p_center_freq, dual_mobility, horizontal));
}
