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
# LSF_DRAW
Spatially correlated normalized random values of the large-scale generators at a set of links

- Evaluates the phases of [[lsf_init_phases]] at the links, one column per generator (see the
  section description). The value of a link does not depend on the other links of the call; in
  single mobility it does not depend on `tx_pos`
- Dual mobility: the Tx is a second device on the same maps. Every column keeps its distribution;
  ASD and ASA, ESD and ESA of a link are correlated by the ACF at the Tx-Rx distance (equal at
  distance 0). Swapping `tx_pos` and `rx_pos` swaps the columns 4 and 5, 6 and 7 and leaves the
  others unchanged, exactly
- Floors (2D mode): links with equal `floor_id` share one field; fields of different floors are
  uncorrelated at equal horizontal positions, exactly within a block of `n_coeff` consecutive IDs,
  otherwise up to a random residual of standard deviation `1/sqrt(2·n_coeff)`. In dual mobility both
  devices of a link are on the link's floor
- The values are within `4e-7 · sqrt(2·n_coeff)` of a double-precision evaluation (AVX2 kernel of
  [[sos_val]])

## Usage:
```
rnd = quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, floor_id, dual_mobility, ...
    preset, horizontal );
```

## Inputs:
- **`scenpar`** — Scenario parameters; the decorrelation distances are used
- **`phase`** — Phases from [[lsf_init_phases]], finite; `[n_coeff, 11, 2]`
- **`tx_pos`** — Tx positions in [m]; used in dual mobility, always validated; `[1, 3]` or `[n_link,
  3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`floor_id`** *(optional)* — Floor index of the UT per link, a label starting at 0 (no index
  shift), non-negative integers below 2^53 (`n_fl - 1` for the UT heights of TR 38.901 Table 7.2-1,
  0 for outdoor UTs); omitted or empty = floor 0; 2D mode only; `[n_link]`
- **`dual_mobility`** — `true`: Tx and Rx both move; default: false
- **`preset`** — Sum-of-sinusoids preset of [[lsf_init_phases]], see [[sos_preset]]; a different
  preset with the same number of sinusoids is not detected; default: `'Comb256'`
- **`horizontal`** — Mode of [[lsf_init_phases]]; default: true

## Output:
- **`rnd`** — Random values, columns as in the section description; `[n_link, 11]`

## See also:
- [[lsf_init_phases]] (draws the phases)
- [[path_loss]] (column 11 as `rnd_he`)
- [[lsp_generate]] (the large-scale parameters from `rnd`)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 4 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs; the field 'SubpathMethod' of the scenario struct is not used
    std::string subpath_method;
    const auto scenpar = qd_mex_struct2scenpar(prhs[0], subpath_method, "scenpar");
    const arma::cube phase = qd_mex_get_Cube<double>(prhs[1]);
    const arma::mat tx_pos = qd_mex_get_Mat<double>(prhs[2]);
    const arma::mat rx_pos = qd_mex_get_Mat<double>(prhs[3]);
    const arma::vec floor_id_in = (nrhs < 5) ? arma::vec() : qd_mex_get_Col<double>(prhs[4], false, "floor_id");
    const bool dual_mobility = (nrhs < 6) ? false : qd_mex_get_scalar<bool>(prhs[5], "dual_mobility", false);
    const std::string preset = (nrhs < 7) ? std::string("Comb256") : qd_mex_get_string(prhs[6], "Comb256");
    const bool horizontal = (nrhs < 8) ? true : qd_mex_get_scalar<bool>(prhs[7], "horizontal", true);

    const arma::uvec floor_id = qd_mex_uint_entries(floor_id_in, "floor_id");

    // Output allocation
    arma::mat rnd;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&rnd, rx_pos.n_rows, 11);

    // Wrap optional pointers
    const arma::uvec *p_floor_id = floor_id.empty() ? nullptr : &floor_id;

    // Call library function
    CALL_QD(quadriga_lib::lsf_draw(scenpar, phase, tx_pos, rx_pos, rnd, p_floor_id, dual_mobility, preset, horizontal));
}
