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
# LSF_INIT_PHASES
Draw the random phases of the large-scale generators of one BS site and scenario

- Draws the phases of the 11 large-scale generators (see the section description); [[lsf_draw]]
  evaluates them at the links. The caller stores `phase` to evaluate the same fields again along
  tracks or on maps
- Without bias removal, slice 1 (Rx side) equals [[sos_init_phases]] with 11 generators and the
  `n_coeff` of `preset`; slice 2 (Tx side) is its mirror, with and without bias removal
- Bias removal (`remove_bias = true` and at least 3 rows in `rx_pos`): [[sos_remove_bias]] sets the
  sample mean of each Normal field (columns 1 to 9) over the positions to 0; in dual mobility over
  `rx_pos` and `tx_pos` together. It removes variance unless the positions spread over many
  decorrelation distances
- In 2D mode the bias removal treats all positions as one floor: for a drop with UTs on several
  floors pass the positions of one floor or set `remove_bias = false`
- TR 38.901 clause 7.4.4 literally (exponential ACF, no bias removal): `preset = 'Exp256'`,
  `remove_bias = false`. The default `'Comb256'` has the same decorrelation distance without the
  kink of the exponential ACF at zero distance, so the values evolve smoothly along a track

## Usage:
```
phase = quadriga_lib.lsf_init_phases( scenpar, seed, tx_pos, rx_pos, dual_mobility, ...
    remove_bias, preset, horizontal );
```

## Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances are used
- **`seed`** — Seed of the phases; integer from -1 to 2^63-1; -1 = system random device, new phases
  on each call; default: -1
- **`tx_pos`** *(optional)* — Tx positions in [m]; required for the bias removal in dual mobility,
  otherwise not used; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** *(optional)* — Rx positions of the drop in [m] for the bias removal; omitted or empty
  or fewer than 3 rows: no bias removal; `[n_link, 3]`
- **`dual_mobility`** — `true`: Tx and Rx both move; default: false
- **`remove_bias`** — `true`: remove the sample mean of the Normal fields over the positions;
  default: true
- **`preset`** — Sum-of-sinusoids preset, see [[sos_preset]]; sets `n_coeff`; default: `'Comb256'`
- **`horizontal`** — `true`: 2D mode (TR 38.901); `false`: 3D mode (legacy QuaDRiGa); default: true

## Output:
- **`phase`** — Phases in [rad], in [-pi, pi); slice 1 = Rx side, slice 2 = Tx side; `[n_coeff, 11,
  2]`

## See also:
- [[lsf_draw]] (evaluates the phases at the links)
- [[ssf_init_phases]] (the small-scale sets, seeds derived from this seed)
- [[sos_remove_bias]] (the bias removal of one generator)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs; the field 'SubpathMethod' of the scenario struct is not used
    std::string subpath_method;
    const auto scenpar = qd_mex_struct2scenpar(prhs[0], subpath_method, "scenpar");
    const arma::sword seed = (nrhs < 2) ? -1 : qd_mex_get_seed(prhs[1]);
    const arma::mat tx_pos = (nrhs < 3) ? arma::mat() : qd_mex_get_Mat<double>(prhs[2]);
    const arma::mat rx_pos = (nrhs < 4) ? arma::mat() : qd_mex_get_Mat<double>(prhs[3]);
    const bool dual_mobility = (nrhs < 5) ? false : qd_mex_get_scalar<bool>(prhs[4], "dual_mobility", false);
    const bool remove_bias = (nrhs < 6) ? true : qd_mex_get_scalar<bool>(prhs[5], "remove_bias", true);
    const std::string preset = (nrhs < 7) ? std::string("Comb256") : qd_mex_get_string(prhs[6], "Comb256");
    const bool horizontal = (nrhs < 8) ? true : qd_mex_get_scalar<bool>(prhs[7], "horizontal", true);

    // Wrap optional pointers
    const arma::mat *p_tx_pos = tx_pos.empty() ? nullptr : &tx_pos;
    const arma::mat *p_rx_pos = rx_pos.empty() ? nullptr : &rx_pos;

    // Call library function; the size of 'phase' depends on the preset, so it is copied afterwards
    arma::cube phase;
    CALL_QD(quadriga_lib::lsf_init_phases(scenpar, phase, seed, p_tx_pos, p_rx_pos, dual_mobility, remove_bias,
                                          preset, horizontal));

    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&phase);
}
