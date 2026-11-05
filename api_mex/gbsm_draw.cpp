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
Spatially correlated normalized random values of one generator set at a set of links

- Evaluates the phases of [[gbsm_init_phases]] at the links, one column per generator; columns,
  distributions and decorrelation distances: see the set table in the section description. The value
  of a link does not depend on the other links of the call; in single mobility it does not depend on
  `tx_pos`
- Dual mobility: the Tx is a second device on the same maps, its side of each generator uses the
  mirror generator. Every column keeps its distribution; swapping `tx_pos` and `rx_pos` exchanges the
  columns of each mirror pair and keeps the self-mirrored columns, exactly
- Floors (2D mode): links with equal `floor_id` share one field; fields of different floors are
  uncorrelated at equal horizontal positions, exactly within a block of `n_coeff` consecutive IDs,
  otherwise up to a random residual of standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both
  devices of a link are on the link's floor
- The values are within `4e-7 · sqrt(2·n_coeff)` of a double-precision evaluation
- A set never changes size: new random quantities come in new sets (a new `set_type` or `set_id`). A
  column depends neither on the other columns nor on the other links
- An empty set (`NumClusters = 1`) gives `[n_link, 0]` and accepts any empty `phase`

## Usage:
```
rnd = quadriga_lib.gbsm_draw( scenpar, set_type, phase, tx_pos, rx_pos, set_id, preset, floor_id, ...
    center_freq, dual_mobility, horizontal );
```

## Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances and `NumClusters` are used
- **`set_type`** — Set type of [[gbsm_init_phases]]; string
- **`phase`** — Phases from [[gbsm_init_phases]], finite; `[n_coeff, n_gen, 2]`
- **`tx_pos`** — Tx positions in [m]; used in dual mobility, always validated; `[1, 3]` or
  `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`, `n_link = 0` allowed
- **`set_id`** — Set of [[gbsm_init_phases]]: `0` = shared set, `k >= 1` = set of frequency `k`;
  ignored for `'lsf'`; default: 0
- **`preset`** — Preset of [[gbsm_init_phases]]; another preset with the same `n_coeff` is not
  detected; default: `'Comb256'`
- **`floor_id`** *(optional)* — Floor index of the UT per link, 0-based (`n_fl - 1` for the UT
  heights of TR 38.901 Table 7.2-1, 0 for outdoor UTs); omitted or empty = floor 0; 2D mode only;
  `[n_link]`
- **`center_freq`** *(optional)* — Carrier frequencies in [Hz] (messages name it
  `center_frequency`); ignored and not checked for the `'lsf'` and `'cluster'` sets; `[n_freq]`
- **`dual_mobility`** — `true`: Tx and Rx both move; default: false
- **`horizontal`** — `true`: 2D mode (TR 38.901); `false`: 3D mode (legacy QuaDRiGa); default: true

## Output:
- **`rnd`** — Random values; `[n_link, n_gen]`

## See also:
- [[lsp_generate]] (the large-scale parameters from the `'lsf'` set)
- [[path_loss]] (column 11 of the `'lsf'` set as `rnd_he`)
- [[ssf_clusters]] (the clusters from the `'cluster'` sets)
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
    const std::string set_type = qd_mex_get_string(prhs[1]);
    const arma::cube phase = qd_mex_get_Cube<double>(prhs[2]);
    const arma::mat tx_pos = qd_mex_get_Mat<double>(prhs[3]);
    const arma::mat rx_pos = qd_mex_get_Mat<double>(prhs[4]);
    const double set_id_in = (nrhs < 6) ? 0.0 : qd_mex_get_scalar<double>(prhs[5], "set_id", 0.0);
    const std::string preset = (nrhs < 7) ? std::string("Comb256") : qd_mex_get_string(prhs[6], "Comb256");
    const arma::vec floor_id_in = (nrhs < 8) ? arma::vec() : qd_mex_get_Col<double>(prhs[7], false, "floor_id");
    const arma::vec center_freq = (nrhs < 9) ? arma::vec() : qd_mex_get_Col<double>(prhs[8], false, "center_freq");
    const bool dual_mobility = (nrhs < 10) ? false : qd_mex_get_scalar<bool>(prhs[9], "dual_mobility", false);
    const bool horizontal = (nrhs < 11) ? true : qd_mex_get_scalar<bool>(prhs[10], "horizontal", true);

    // Set index: MATLAB 0 = shared set, k >= 1 = frequency k (1-based); C++ -1 and k - 1; ignored for "lsf"
    const bool lsf = set_type == "lsf";
    if (!lsf && !(set_id_in >= 0.0 && set_id_in < 9007199254740992.0 && set_id_in == std::floor(set_id_in)))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'set_id' must be 0 (shared set) or a frequency index >= 1.");
    const arma::sword set_id = lsf ? -1 : (arma::sword)set_id_in - 1;

    const arma::uvec floor_id = qd_mex_uint_entries(floor_id_in, "floor_id");

    // Wrap optional pointers
    const arma::uvec *p_floor_id = floor_id.empty() ? nullptr : &floor_id;
    const arma::vec *p_center_freq = center_freq.empty() ? nullptr : &center_freq;

    // Output allocation in MATLAB memory: [n_link, n_gen], n_gen = columns of 'phase' (checked by the library)
    arma::mat rnd;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&rnd, rx_pos.n_rows, phase.n_cols);

    // Call library function
    CALL_QD(quadriga_lib::gbsm_draw(scenpar, set_type, phase, tx_pos, rx_pos, rnd, set_id, preset, p_floor_id,
                                    p_center_freq, dual_mobility, horizontal));
}
