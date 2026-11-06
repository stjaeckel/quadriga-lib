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
# GBSM_INIT_PHASES
SOS phases of one random set of a BS site and scenario

- Draws the SOS phases of one random set (set table in [[gbsm_draw]]): the random phases of the sinusoids of its
  [sum-of-sinusoids (SOS)](mex_api.html#sec-sum-of-sinusoids-random-number-generator) generators, as
  [[sos_init_phases]]; [[gbsm_draw]] evaluates them at the links. Store `phase` or the seed to evaluate the same fields
  again along tracks or on maps
- The SOS phases are a fixed function of `seed`, `set_type` and `set_id`, equal on all platforms (bias-removed `'lsf'`
  sets within about 2e-14 rad); the sets are independent; each scenario of a site needs its own seed
- `'lsf'` bias removal (`remove_bias`): [[sos_remove_bias]] sets the sample mean of columns 1 to 9 over the positions
  (Rx, and Tx in dual mobility; in 2D mode all on floor 0) to 0, so these means do not vary from drop to drop
- `NumClusters = 1`: all sets but `'lsf'` are empty
- TR 38.901 clause 7.4.4 literally: preset `Exp256` without bias removal

## Usage:
```
phase = quadriga_lib.gbsm_init_phases( scenpar, set_type, seed, set_id, preset, tx_pos, rx_pos, ...
    dual_mobility, remove_bias, horizontal );
```

## Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances, `NumClusters` and `NumSubPaths` are used
- **`set_type`** — `'lsf'`, `'cluster'` or `'ray'`; string
- **`seed`** — Seed of the site and scenario; integer from -1 to 2^63-1; -1 = system random device, new SOS phases on
  each call; default: -1
- **`set_id`** — `0` = shared set, `k >= 1` = set of frequency `k`; ignored for `'lsf'`; default: 0
- **`preset`** — Sum-of-sinusoids preset, see [[sos_preset]]; sets `n_coeff`; default: `'Comb256'`
- **`tx_pos`**, **`rx_pos`** *(optional)* — Bias-removal positions in [m] as in [[gbsm_draw]]; `'lsf'` only, else an
  error
- **`dual_mobility`** — `'lsf'` with bias removal: Tx and Rx both move; as in [[gbsm_draw]]; default: false
- **`remove_bias`** — `'lsf'` only: remove the sample mean over the positions; needs `rx_pos` with at least 3 rows, else
  no effect; default: true
- **`horizontal`** — `'lsf'` with bias removal: `true` 2D, `false` 3D mode; as in [[gbsm_draw]]; default: true

## Output:
- **`phase`** — SOS phases in [rad], in [-pi, pi), row = sinusoid, column = generator; slice 1 = Rx side, slice 2 = Tx
  side; `[n_coeff, n_gen, 2]`

## See also:
- [[gbsm_draw]] (evaluates the SOS phases at the links)
- [[sos_init_phases]], [[sos_val]] (SOS phases and evaluation of one generator), [[sos_remove_bias]] (its bias removal)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 2 || nrhs > 10)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const auto scenpar = qd_mex_struct2scenpar(prhs[0], "scenpar");
    const std::string set_type = qd_mex_get_string(prhs[1]);
    const arma::sword seed = (nrhs < 3) ? -1 : qd_mex_get_seed(prhs[2]);
    const double set_id_in = (nrhs < 4) ? 0.0 : qd_mex_get_scalar<double>(prhs[3], "set_id", 0.0);
    const std::string preset = (nrhs < 5) ? std::string("Comb256") : qd_mex_get_string(prhs[4], "Comb256");
    const arma::mat tx_pos = (nrhs < 6) ? arma::mat() : qd_mex_get_Mat<double>(prhs[5]);
    const arma::mat rx_pos = (nrhs < 7) ? arma::mat() : qd_mex_get_Mat<double>(prhs[6]);
    const bool dual_mobility = (nrhs < 8) ? false : qd_mex_get_scalar<bool>(prhs[7], "dual_mobility", false);
    const bool remove_bias = (nrhs < 9) ? true : qd_mex_get_scalar<bool>(prhs[8], "remove_bias", true);
    const bool horizontal = (nrhs < 10) ? true : qd_mex_get_scalar<bool>(prhs[9], "horizontal", true);

    // Set index: MATLAB 0 = shared set, k >= 1 = frequency k (1-based); C++ -1 and k - 1; ignored for "lsf"
    const bool lsf = set_type == "lsf";
    if (!lsf && !(set_id_in >= 0.0 && set_id_in < 9007199254740992.0 && set_id_in == std::floor(set_id_in)))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'set_id' must be 0 (shared set) or a frequency index >= 1.");
    const arma::sword set_id = lsf ? -1 : (arma::sword)set_id_in - 1;

    // Wrap optional pointers
    const arma::mat *p_tx_pos = tx_pos.empty() ? nullptr : &tx_pos;
    const arma::mat *p_rx_pos = rx_pos.empty() ? nullptr : &rx_pos;

    // Output allocation in MATLAB memory: n_coeff of the preset, n_gen of the set from a completed copy of
    // 'scenpar'; with an unknown preset or set type or an invalid 'scenpar' the call throws before any output
    arma::cube phase;
    const arma::uword n_coeff = qd_gbsm_n_coeff(preset);
    bool known = nlhs > 0 && n_coeff > 0 && (lsf || set_type == "cluster" || set_type == "ray");
    auto sp = scenpar;
    try
    {
        if (known)
            quadriga_lib::scenpar_validate(sp);
    }
    catch (const std::exception &)
    {
        known = false;
    }
    if (known)
        plhs[0] = qd_mex_init_output(&phase, n_coeff, qd_gbsm_n_gen(sp, set_type, set_id), 2);

    // Call library function
    CALL_QD(quadriga_lib::gbsm_init_phases(scenpar, set_type, phase, seed, set_id, preset, p_tx_pos, p_rx_pos,
                                           dual_mobility, remove_bias, horizontal));

}
