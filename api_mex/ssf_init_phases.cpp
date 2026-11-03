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
# SSF_INIT_PHASES
Draw the random phases of one small-scale fading generator set of a BS site and scenario

- The cluster random values (TR 38.901 clause 7.6.3.1) come in one shared cluster set (`set_id = 0`,
  `9·N` generators, `N = NumClusters - 1` NLOS clusters) and one set per carrier frequency (`set_id
  = k`, `N` generators, `k` the position in the caller's frequency list, 1-based); [[ssf_draw]]
  evaluates them and lists the columns
- The phases are a fixed function of `seed` and `set_id`, equal on all platforms; the sets are
  independent of each other and of the large-scale phases. Keep an explicit seed (or the phase
  arrays) to reproduce the clusters
- Each scenario of a site (LOS, NLOS, O2I) needs its own large-scale seed: TR 38.901 clause 7.6.3.4
  makes the link types uncorrelated
- `NumClusters = 1`: both sets are empty

## Usage:
```
phase = quadriga_lib.ssf_init_phases( scenpar, seed, set_id, preset );
```

## Inputs:
- **`scenpar`** — Scenario parameters; `NumClusters` sets `N`
- **`seed`** — Seed of the large-scale set of the same site and scenario ([[lsf_init_phases]]);
  integer from -1 to 2^63-1; -1 = system random device, new phases on each call; default: -1
- **`set_id`** — `0` = shared cluster set; `k >= 1` = set of frequency `k`; default: 0
- **`preset`** — Sum-of-sinusoids preset, see [[sos_preset]]; sets `n_coeff`; default: `'Comb256'`

## Output:
- **`phase`** — Phases in [rad], in [-pi, pi); slice 1 = Rx side, slice 2 = Tx side; `[n_coeff,
  n_gen, 2]`, `n_gen = 9·N` (shared set) or `N`

## See also:
- [[ssf_draw]] (evaluates the phases at the links)
- [[lsf_init_phases]] (the large-scale set and its seed)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs; the field 'SubpathMethod' of the scenario struct is not used
    std::string subpath_method;
    const auto scenpar = qd_mex_struct2scenpar(prhs[0], subpath_method, "scenpar");
    const arma::sword seed = (nrhs < 2) ? -1 : qd_mex_get_seed(prhs[1]);
    const double set_id_in = (nrhs < 3) ? 0.0 : qd_mex_get_scalar<double>(prhs[2], "set_id", 0.0);
    const std::string preset = (nrhs < 4) ? std::string("Comb256") : qd_mex_get_string(prhs[3], "Comb256");

    // Set index: MATLAB 0 = shared set, k >= 1 = frequency k (1-based); C++ -1 and k - 1
    if (!(set_id_in >= 0.0 && set_id_in < 9007199254740992.0 && set_id_in == std::floor(set_id_in)))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'set_id' must be 0 (shared set) or a frequency index >= 1.");
    const arma::sword set_id = (arma::sword)set_id_in - 1;

    // Call library function; the size of 'phase' depends on 'scenpar' and 'preset', so it is copied afterwards
    arma::cube phase;
    CALL_QD(quadriga_lib::ssf_init_phases(scenpar, phase, seed, set_id, preset));

    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&phase);
}
