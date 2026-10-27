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

- The large-scale fading (LSF) of a site is described by 11 spatially correlated sum-of-sinusoids
  generators, one column each: 1 DS, 2 KF, 3 SF, 4 ASD, 5 ASA, 6 ESD, 7 ESA, 8 XPR, 9 absolute TOA
  (Normal fields), 10 ground permittivity, 11 UMa environment height hE (Uniform fields)
- Their decorrelation distances are the scenario parameters `DS_lambda`, `KF_lambda`, `SF_lambda`,
  `AS_D_lambda`, `AS_A_lambda`, `ES_D_lambda`, `ES_A_lambda`, `XPR_lambda`, `absTOA_lambda`,
  `SC_lambda` and again `SF_lambda` for hE; a decorrelation distance of 0 is used as 0.1 m. The
  generator of column 9 exists also when the absolute TOA is disabled
- This function draws the phases; [[lsf_draw]] evaluates them at the links. The caller stores
  `phase` and evaluates the same fields again along tracks or on maps
- `scenpar` is completed with the defaults of missing fields and validated as in
  [[scenpar_validate]]; its messages (e.g. for an unknown field or a negative decorrelation
  distance) pass through unchanged. An invalid value of a field that is not used here throws as well
- Slice 1 (Rx side) holds the phases of the 11 generators. Without bias removal it equals
  `quadriga_lib.sos_init_phases( n_coeff, 11, seed )` of [[sos_init_phases]], with `n_coeff` the
  number of sinusoids of `preset`. The same seed gives the same phases
- Slice 2 (Tx side) is always the mirror of slice 1:
  `phase(:, :, 2) = phase(:, [1, 2, 3, 5, 4, 7, 6, 8, 9, 10, 11], 1)`, i.e. ASD and ASA and ESD and
  ESA are exchanged, all others mirror themselves. For dual mobility the departure spreads at the Tx
  are thus the arrival spreads of the Tx as a second device
- Bias removal (`remove_bias = true`, `rx_pos` given, `n_link > 2`): one realisation of a correlated
  field has a non-zero mean over a drop. For each Normal generator (columns 1 to 9) one common phase
  offset is applied with [[sos_remove_bias]], so that the sample mean of its field over the
  positions is exactly 0. Single mobility: over the Rx positions; `tx_pos` is not used. Dual
  mobility: over the Rx and Tx positions together (`2·n_link` positions), because the field of a
  generator is shared by both devices (the Tx side of ASD is the ASA field); `tx_pos` is then
  required. The Uniform generators 10 and 11 are never corrected
- The correction is applied to slice 1 and the Tx side is mirrored afterwards, so the mirror
  relation holds exactly with and without bias removal. Swapping `tx_pos` and `rx_pos` in dual
  mobility gives the same phases up to rounding (the sums over the positions are formed in another
  order)
- Consequences for [[lsf_draw]] at the same positions: in single mobility the link mean of columns
  1 to 9 is 0; in dual mobility the link mean of DS, KF, SF, XPR and TOA is 0 when all links have
  the same Tx-Rx distance (the normalisation is then the same for all links), while ASD/ASA and
  ESD/ESA have a zero link mean only approximately (their two devices see different fields, each
  corrected over the union of the positions)
- The positions are used as in [[lsf_draw]]: projected on the x-y plane (z = 0) for
  `horizontal = true`, in 3D for `horizontal = false`. `preset` and `horizontal` must be the same in
  [[lsf_draw]]
- All positions in `rx_pos` are treated as one field (floor 0): one phase correction per generator
  cannot zero the means of several floor fields at once. For a drop with UTs on several floors pass
  the positions of one floor (e.g. the links of floor 0) or set `remove_bias = false`; the fields of
  the other floors keep their sample mean
- The correction removes variance: the field is conditioned on a zero mean over the positions, which
  costs little only when the positions spread over many decorrelation distances (see
  [[sos_remove_bias]])
- With fewer than 3 positions, without `rx_pos` or with `remove_bias = false` no correction is made
- Validation order: `seed`, `preset`, `scenpar`, then the positions (3 columns, rows of `tx_pos` 1
  or `n_link` when both are given, finite values), then the requirement of `tx_pos` for the bias
  removal in dual mobility

## Usage:
```
phase = quadriga_lib.lsf_init_phases( scenpar, seed, tx_pos, rx_pos, dual_mobility, ...
    remove_bias, preset, horizontal );
```

## Inputs:
- **`scenpar`** — Scenario parameters; struct as returned by [[conf_file_read]] or
  [[scenpar_validate]] (the field `SubpathMethod` is ignored), or any subset of its fields; missing
  fields have their default value; `[]` = all defaults; only the 10 decorrelation distances are used
- **`seed`** — Seed of the random phases; integer from -1 to 2^63-1; -1 = system random device, a
  new set of phases on each call; default: -1
- **`tx_pos`** *(optional)* — Tx positions in [m]; used only for the bias removal with dual
  mobility, required there; one row is used for all links; `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** *(optional)* — Rx positions of the drop in [m] (one floor) for the bias removal;
  omitted, empty or fewer than 3 rows = no bias removal; `[n_link, 3]`
- **`dual_mobility`** — `true`: Tx and Rx both move, the Tx is a second device on the same maps;
  default: false
- **`remove_bias`** — `true`: remove the sample mean of the Normal fields over the positions;
  default: true
- **`preset`** — Sum-of-sinusoids preset, see [[sos_preset]]; sets `n_coeff`; default: `'Comb256'`
- **`horizontal`** — `true`: 2D correlation in the x-y plane (TR 38.901 behaviour); `false`: 3D
  correlation (legacy QuaDRiGa behaviour); default: true

## Output:
- **`phase`** — Phases in [rad]; slice 1 = Rx side, slice 2 = Tx side (mirror of slice 1); all
  values in [-pi, pi); `[n_coeff, 11, 2]`

## See also:
- [[conf_file_read]] (the scenario parameters)
- [[lsf_draw]] (evaluates the phases at the links)
- [[sos_init_phases]] (the raw phases)
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
