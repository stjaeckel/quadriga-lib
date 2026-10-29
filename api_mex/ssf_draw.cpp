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

- Evaluates the generators of a set of [[ssf_init_phases]] at the links. With
  `N = NumClusters - 1`, column `b·N + n + 1` belongs to block `b` and NLOS cluster `n` (`b`, `n`
  counted from 0). Shared cluster set (`set_id = 0`, `9·N` columns): block 0 cluster delay
  (Uniform(0,1)), blocks 1, 2 sign of AOD, AOA, 3, 4 sign of ZOD, ZOA, 5, 6 angle offset of AOD,
  AOA, 7, 8 angle offset of ZOD, ZOA (Normal(0,1)). Per-frequency set (`set_id = k >= 1`, `N`
  columns): cluster shadowing (Normal(0,1))
- All generators use the decorrelation distance `SC_lambda` (TR 38.901 Table 7.6.3.1-2);
  `SC_lambda = 0` (spatial consistency off) is used as 0.1 m, which gives fields that decorrelate
  within 0.1 m
- `scenpar` is completed and validated as in [[ssf_init_phases]]; the messages of
  [[scenpar_validate]] pass through unchanged
- Single mobility (`dual_mobility = false`): column `c` is the field of generator `c` at the Rx
  position, `x_c = s_c(rx)` with `s_c(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )` and
  the phases of slice 1; the values do not depend on `tx_pos`
- Dual mobility: the Tx is a second device. Its side uses slice 2, i.e. the field of the mirror
  generator at the Tx position (departure and arrival blocks exchanged). Self-mirrored columns
  (delay, shadowing) share one map: `x_c = ( s_c(rx) + s_c(tx) ) / sqrt( 2 + 2·A(D) )`, with the
  Tx-Rx distance `D` and the ACF `A` of the preset, linearly interpolated. The pair columns are two
  maps: `x_c = ( s_c(rx) + s_m(tx) ) / sqrt(2)`
- Swap reciprocity in dual mobility (S. Jaeckel et al., "A Spatially Consistent Geometric D2D
  Small-Scale Fading Model for Multiple Frequencies", 2019, eqs. (3)-(5)): swapping `tx_pos` and
  `rx_pos` leaves the delay and the shadowing blocks unchanged and exchanges block 1 with 2, 3 with
  4, 5 with 6 and 7 with 8, exactly (the same evaluations in swapped roles)
- Uniform block 0 of the shared set: `0.5 · erfc( -x / sqrt(2) )` of the Normal value
- `horizontal = true`: the positions are projected on the x-y plane (z = 0) and the values do not
  depend on z; `false`: 3D positions (legacy). `D` is measured in the same space
- Floors as in [[lsf_draw]] (same phase offsets per floor ID; TR 38.901 clause 7.6.3.3: SSF
  variables are uncorrelated across floors): pass the same `floor_id` as to [[lsf_draw]]; only in
  2D mode
- The value of a link does not depend on the other links of the call, and a column does not depend
  on the phases of the other columns (append-only: appended blocks leave the existing columns
  unchanged bit for bit)
- Clusters by TR 38.901 clause 7.5 under spatial consistency: clause 7.6.3.1 keeps the signs and the
  random offsets of the clusters unchanged per drop; the caller evaluates them at the drop or
  segment-start positions and reuses those values along the track
- Precision: the sinusoids are evaluated by [[sos_val]] with the AVX2 kernel where available (sine
  and cosine in single precision, accumulation in double); the values deviate from a
  double-precision evaluation by less than `4e-7 · sqrt(2 · n_coeff)` (9e-6 for 256 sinusoids)
- `preset`, `horizontal` and the set type (`set_id = 0` or `>= 1`) must match [[ssf_init_phases]]
- `NumClusters = 1` returns an empty `rnd` and accepts any empty `phase`. At least one link is
  required
- Validation order: positions (3 columns, rows of `tx_pos`, finite values), `set_id`, `preset`,
  `scenpar`, size of `phase`, finite `phase`, length of `floor_id`, `floor_id` with
  `horizontal = false`

## Usage:
```
rnd = quadriga_lib.ssf_draw( scenpar, phase, tx_pos, rx_pos, set_id, floor_id, dual_mobility, ...
    preset, horizontal );
```

## Inputs:
- **`scenpar`** — Scenario parameters; struct as returned by [[conf_file_read]] or
  [[scenpar_validate]] (the field `SubpathMethod` is ignored), or any subset of its fields; missing
  fields have their default value; `[]` = all defaults; `NumClusters` and `SC_lambda` are used
- **`phase`** — Phases in [rad] from [[ssf_init_phases]]; slice 1 = Rx side, slice 2 = Tx side;
  must be finite; `[n_coeff, n_gen, 2]`, `n_coeff` of `preset`, `n_gen` of the set type
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; not used in single mobility
  (but validated); `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`set_id`** — Set type as in [[ssf_init_phases]]: `0` = shared cluster set, `k >= 1` =
  per-frequency set; default: 0
- **`floor_id`** *(optional)* — Floor index per link, a label starting at 0 (no index shift);
  non-negative integers below 2^53; omitted or empty = all links on floor 0; in dual mobility the
  floor of both devices; only with `horizontal = true`; `[n_link]`
- **`dual_mobility`** — `true`: Tx and Rx both move, the Tx side is evaluated as a second device;
  default: false
- **`preset`** — Sum-of-sinusoids preset, see [[sos_preset]]; must match [[ssf_init_phases]];
  default: `'Comb256'`
- **`horizontal`** — `true`: 2D correlation in the x-y plane (TR 38.901 behaviour); `false`: 3D
  correlation (legacy QuaDRiGa behaviour); must match [[ssf_init_phases]]; default: true

## Output:
- **`rnd`** — Random values; shared set: columns 1 to N Uniform(0,1), the others Normal(0,1);
  per-frequency set: Normal(0,1); `[n_link, n_gen]`

## See also:
- [[ssf_init_phases]] (draws the phases)
- [[ssf_clusters]] (the clusters from the random values)
- [[lsf_draw]] (the large-scale values, same floors and mobility rules)
- [[sos_val]] (evaluation of a sum-of-sinusoids generator)
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

    // Floor IDs are labels (no index shift): read as double, checked, converted; integers from 2^53 on are not
    // exact in a double
    arma::uvec floor_id(floor_id_in.n_elem);
    for (arma::uword i = 0; i < floor_id_in.n_elem; ++i)
    {
        const double v = floor_id_in[i];
        if (!(v >= 0.0 && v < 9007199254740992.0 && v == std::floor(v)))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Entries in 'floor_id' must be non-negative integers below 2^53.");
        floor_id[i] = (arma::uword)v;
    }

    // Wrap optional pointers
    const arma::uvec *p_floor_id = floor_id.empty() ? nullptr : &floor_id;

    // Call library function; the number of columns depends on 'scenpar', so 'rnd' is copied afterwards
    arma::mat rnd;
    CALL_QD(quadriga_lib::ssf_draw(scenpar, phase, tx_pos, rx_pos, rnd, set_id, p_floor_id, dual_mobility, preset,
                                   horizontal));

    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&rnd);
}
