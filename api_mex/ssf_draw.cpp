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
# SSF_DRAW
Spatially correlated normalized random values of one small-scale fading generator set at a set of links

- Evaluates a set of [[ssf_init_phases]] at the links with the rules of [[lsf_draw]] for single and dual mobility,
  floors, 2D/3D mode and precision; every generator uses the decorrelation distance `SC_lambda` (TR 38.901 Table
  7.6.3.1-2; 0 = spatial consistency off, used as 0.1 m)
- Columns, `N = NumClusters - 1`; block `b = 0 ... 8` holds the columns `b·N + 1` to `(b + 1)·N`, one per NLOS
  cluster:
  - shared set (`set_id = 0`, `9·N` columns): block 0 cluster delay, Uniform(0,1); blocks 1, 2 sign of AOD, AOA;
    3, 4 sign of ZOD, ZOA; 5, 6 angle offset of AOD, AOA; 7, 8 angle offset of ZOD, ZOA; Normal(0,1)
  - per-frequency set (`set_id = k >= 1`, `N` columns): cluster shadowing, Normal(0,1)
- Dual mobility: swapping `tx_pos` and `rx_pos` exchanges the blocks 1 and 2, 3 and 4, 5 and 6, 7 and 8 and keeps
  the delay and shadowing blocks, exactly (D2D reciprocity of S. Jaeckel et al., "A Spatially Consistent Geometric
  D2D Small-Scale Fading Model for Multiple Frequencies", 2019)
- A column depends neither on the other columns nor on the other links; later versions append blocks and keep the
  values of the existing columns
- `preset`, `horizontal` and the set type must match [[ssf_init_phases]]; at least one link is required

## Usage:
```
rnd = quadriga_lib.ssf_draw( scenpar, phase, tx_pos, rx_pos, set_id, floor_id, dual_mobility, ...
    preset, horizontal );
```

## Inputs:
- **`scenpar`** — Scenario parameters as in [[ssf_init_phases]]; `NumClusters` and `SC_lambda` are used
- **`phase`** — Phases from [[ssf_init_phases]]; `[n_coeff, n_gen, 2]`
- **`tx_pos`** — Tx positions in [m], one row for all links; used in dual mobility; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`set_id`** *(optional)* — `0` = shared cluster set; `k >= 1` = per-frequency set; default: 0
- **`floor_id`** *(optional)* — Floor label per link starting at 0 (no index shift) as in [[lsf_draw]], 2D mode
  only; omitted or empty = floor 0; `[n_link]`
- **`dual_mobility`** *(optional)* — `true`: the Tx is a second moving device; default: false
- **`preset`** *(optional)* — Sum-of-sinusoids preset, see [[sos_preset]]; default: `'Comb256'`
- **`horizontal`** *(optional)* — `true`: 2D correlation (TR 38.901); `false`: 3D (legacy); default: true

## Output:
- **`rnd`** — Random values; `[n_link, n_gen]`

## See also:
- [[ssf_init_phases]] (draws the phases)
- [[ssf_clusters]] (the clusters from the random values)
- [[lsf_draw]] (the large-scale values; mobility, floor and precision rules)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 4 || nrhs > 9)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs; the field 'SubpathMethod' of the scenario struct is not used
    std::string subpath_method;
    const auto scenpar = qd_mex_struct2scenpar(prhs[0], subpath_method, "scenpar");
    const arma::cube phase = qd_mex_get_Cube<double>(prhs[1]);
    const arma::mat tx_pos = qd_mex_get_Mat<double>(prhs[2]);
    const arma::mat rx_pos = qd_mex_get_Mat<double>(prhs[3]);
    const double set_id_in = (nrhs < 5) ? 0.0 : qd_mex_get_scalar<double>(prhs[4], "set_id", 0.0);
    const arma::vec floor_id_in = (nrhs < 6) ? arma::vec() : qd_mex_get_Col<double>(prhs[5], false, "floor_id");
    const bool dual_mobility = (nrhs < 7) ? false : qd_mex_get_scalar<bool>(prhs[6], "dual_mobility", false);
    const std::string preset = (nrhs < 8) ? std::string("Comb256") : qd_mex_get_string(prhs[7], "Comb256");
    const bool horizontal = (nrhs < 9) ? true : qd_mex_get_scalar<bool>(prhs[8], "horizontal", true);

    // Set index: MATLAB 0 = shared set, k >= 1 = frequency k (1-based); C++ -1 and k - 1
    if (!(set_id_in >= 0.0 && set_id_in < 9007199254740992.0 && set_id_in == std::floor(set_id_in)))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'set_id' must be 0 (shared set) or a frequency index >= 1.");
    const arma::sword set_id = (arma::sword)set_id_in - 1;

    const arma::uvec floor_id = qd_mex_uint_entries(floor_id_in, "floor_id");

    // Wrap optional pointers
    const arma::uvec *p_floor_id = floor_id.empty() ? nullptr : &floor_id;

    // Call library function; the number of columns depends on 'scenpar', so 'rnd' is copied afterwards
    arma::mat rnd;
    CALL_QD(quadriga_lib::ssf_draw(scenpar, phase, tx_pos, rx_pos, rnd, set_id, p_floor_id, dual_mobility, preset,
                                   horizontal));

    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&rnd);
}
