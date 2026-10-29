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

- The small-scale fading (SSF) random variables of the clusters (TR 38.901 clause 7.6.3.1) are
  spatially correlated sum-of-sinusoids generators in two kinds of sets, with `N = NumClusters - 1`
  NLOS clusters. Column `b·N + n + 1` belongs to block `b` and cluster `n` (`b`, `n` counted from 0)
- Shared cluster set (`set_id = 0`, frequency-independent), `9·N` generators: block 0 cluster delay
  (Uniform(0,1)); blocks 1, 2 sign of AOD, AOA; 3, 4 sign of ZOD, ZOA; 5, 6 angle offset of AOD,
  AOA; 7, 8 angle offset of ZOD, ZOA (all Normal(0,1))
- Per-frequency set (`set_id = k >= 1`), `N` generators: block 0 cluster shadowing (Normal(0,1)).
  `k` is the index of the frequency in the caller's frequency list (1-based); the list must have
  the same order in every call
- This function draws the phases; [[ssf_draw]] evaluates them at the links. The caller stores
  `phase` and evaluates the same fields again along tracks or on maps
- Slice 2 (Tx side) is always the mirror of slice 1: column `b·N + n + 1` of slice 2 is column
  `m(b)·N + n + 1` of slice 1, with `m` exchanging the blocks 1 and 2, 3 and 4, 5 and 6, 7 and 8
  (departure and arrival of the same cluster); the delay block and the shadowing block mirror
  themselves
- Seed of the set, derived from `seed`, the seed of the large-scale set of the same site and
  scenario (see [[lsf_init_phases]]): `seed_set = splitmix64( splitmix64(seed) ^ tag ) >> 1` with
  `tag = 1` for the shared set and `tag = 1 + k` for the set of frequency `k >= 1`, and
  `splitmix64(x)`: `z = x + 0x9E3779B97F4A7C15`, `z = (z ^ (z >> 30)) · 0xBF58476D1CE4E5B9`,
  `z = (z ^ (z >> 27)) · 0x94D049BB133111EB`, result `z ^ (z >> 31)` (unsigned 64-bit arithmetic,
  wrapping). Slice 1 equals `quadriga_lib.sos_init_phases( n_coeff, n_gen, seed_set )` of
  [[sos_init_phases]], with `n_coeff` the number of sinusoids of `preset`. The sets therefore never
  repeat the large-scale phases of the same seed, and the sets of different frequencies are
  independent streams
- The same seed gives the same phases on all platforms; the caller keeps an explicit seed (or the
  phase arrays) to reproduce the clusters. `seed = -1` draws unrelated random phases from the system
  random device. Seeds above 2^53 are exact only when given as `int64` or `uint64`
- Each scenario of a site (link type LOS, NLOS, O2I) needs its own large-scale seed: TR 38.901
  clause 7.6.3.4 makes the link types uncorrelated, and one seed would give the LOS and the NLOS
  scenario identical cluster fields
- `NumClusters = 1` (`N = 0`, LOS path only): both sets are empty, `phase` is empty
- Append-only: a later version that appends blocks to a set keeps the existing columns and their
  phases; [[sos_init_phases]] fills column `g` from one stream independently of the number of
  columns
- No bias removal (the function takes no positions). `SC_lambda`, the decorrelation distance of all
  generators, is used only by [[ssf_draw]]
- `scenpar` is completed with the defaults of missing fields and validated as in
  [[scenpar_validate]]; its messages pass through unchanged. An invalid value of a field that is
  not used here throws as well
- Validation order: `seed`, `set_id`, `preset`, `scenpar`

## Usage:
```
phase = quadriga_lib.ssf_init_phases( scenpar, seed, set_id, preset );
```

## Inputs:
- **`scenpar`** — Scenario parameters; struct as returned by [[conf_file_read]] or
  [[scenpar_validate]] (the field `SubpathMethod` is ignored), or any subset of its fields; missing
  fields have their default value; `[]` = all defaults; `NumClusters` sets the number of generators
- **`seed`** — Seed of the large-scale set of the same site and scenario; integer from -1 to
  2^63-1; -1 = system random device, a new set of phases on each call; default: -1
- **`set_id`** — `0` = shared cluster set; `k >= 1` = set of frequency `k` (1-based); default: 0
- **`preset`** — Sum-of-sinusoids preset, see [[sos_preset]]; sets `n_coeff`; default: `'Comb256'`

## Output:
- **`phase`** — Phases in [rad]; slice 1 = Rx side, slice 2 = Tx side (mirror of slice 1); all
  values in [-pi, pi); `[n_coeff, n_gen, 2]` with `n_gen = 9·N` (shared set) or `N`
  (per-frequency set)

## See also:
- [[ssf_draw]] (evaluates the phases at the links)
- [[ssf_clusters]] (the clusters from the random values)
- [[lsf_init_phases]] (the large-scale set and its seed)
- [[sos_init_phases]] (the raw phases)
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
