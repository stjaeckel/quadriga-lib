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

- Evaluates the 11 large-scale fading generators of [[lsf_init_phases]] at the links: column 1 DS,
  2 KF, 3 SF, 4 ASD, 5 ASA, 6 ESD, 7 ESA, 8 XPR, 9 absolute TOA (Normal(0,1)), 10 ground
  permittivity, 11 UMa environment height hE (Uniform(0,1)); each generator with the
  sum-of-sinusoids frequencies of `preset` at its own decorrelation distance (a distance of 0 is
  used as 0.1 m)
- `scenpar` is completed and validated as in [[lsf_init_phases]]; the messages of
  [[scenpar_validate]] pass through unchanged
- Single mobility (`dual_mobility = false`): column g is the field of generator g at the Rx
  position, `x_g = s_g(rx)` with `s_g(x) = sqrt(2/n_coeff) · sum_n cos( 2·pi · f_n·x + phi_n )` and
  the phases of slice 1; the values do not depend on `tx_pos`
- Dual mobility: the Tx is a second device. Its side uses slice 2, i.e. the field of the mirror
  generator `m(g)` at the Tx position (ASD and ASA, ESD and ESA exchanged). Self-mirrored generators
  (DS, KF, SF, XPR, TOA, columns 10 and 11) share one map:
  `x_g = ( s_g(rx) + s_g(tx) ) / sqrt( 2 + 2·A_g(D) )`, with the Tx-Rx distance `D` and the ACF
  `A_g` of the preset, linearly interpolated. The pairs are two maps:
  `x_g = ( s_g(rx) + s_m(tx) ) / sqrt(2)`. Columns 4 to 7 thus stay Normal(0,1), and ASD/ASA and
  ESD/ESA have the correlation `( A_g(D) + A_m(D) ) / 2` at the Tx-Rx distance, 1 at `D = 0` (then
  the two columns are equal)
- Swap reciprocity in dual mobility: swapping `tx_pos` and `rx_pos` leaves columns 1, 2, 3, 8, 9,
  10, 11 unchanged and swaps column 4 with 5 and column 6 with 7, exactly (the same evaluations in
  swapped roles), also for unequal decorrelation distances and with bias-removed phases, when both
  calls use the same `phase`
- Uniform columns 10 and 11: `0.5 · erfc( -x / sqrt(2) )` of the Normal value
- `horizontal = true`: the positions are projected on the x-y plane (z = 0) and the values do not
  depend on z; `false`: 3D positions. `D` is measured in the same space
- Floors: links with different floor IDs see different fields, links with equal IDs share one
  field. Floor `q` adds the offsets `2·pi · k · n / N + psi_b(n)` to the phases of all generators
  on both sides, with `N = n_coeff`, `k = q mod N`, `b = q div N`, `n = 0 ... N-1`, `psi_0 = 0` and
  `psi_b(n) = 2·pi · u_n` for `b >= 1`, where `u_n` are the uniform numbers of a `std::mt19937_64`
  seeded with `b` (the upper 53 bits of each raw output times 2^-53). Floor 0 and an omitted
  `floor_id` give the same values
- Fields of two floors are uncorrelated at equal horizontal positions (TR 38.901 Table 7.5-6
  note 3): exactly for two IDs in the same block of `N` IDs, with a random residual of standard
  deviation `1/sqrt(2N)` for IDs in different blocks (0.044 for 256 sinusoids). At other horizontal
  lags the cross-correlation of two floors is not zero (up to about 0.16 for 256 sinusoids), the
  same level as for random offsets. Mirror relations and swap reciprocity hold per floor
- `floor_id` is the caller's floor index of the UT, a label (no index shift): for the UT heights
  `hUT = 3·(n_fl - 1) + 1.5` of TR 38.901 Table 7.2-1 it is `n_fl - 1`; outdoor UTs use 0
- The value of a link does not depend on the other links of the call
- Precision: the sinusoids are evaluated by [[sos_val]] with the AVX2 kernel where available (sine
  and cosine in single precision, accumulation in double); the values deviate from a
  double-precision evaluation by less than `4e-7 · sqrt(2 · n_coeff)` (9e-6 for 256 sinusoids)
- `preset` and `horizontal` must be those of [[lsf_init_phases]]. A wrong preset with the same
  number of sinusoids (e.g. `Exp256` for `Comb256`) cannot be detected
- Validation order: positions (3 columns, rows of `tx_pos`, finite values), `preset`, `scenpar`,
  size of `phase`, finite `phase`, length of `floor_id`

## Usage:
```
rnd = quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, floor_id, dual_mobility, ...
    preset, horizontal );
```

## Inputs:
- **`scenpar`** — Scenario parameters; struct as returned by [[conf_file_read]] or
  [[scenpar_validate]] (the field `SubpathMethod` is ignored), or any subset of its fields; missing
  fields have their default value; `[]` = all defaults; only the 10 decorrelation distances are used
- **`phase`** — Phases in [rad] from [[lsf_init_phases]]; slice 1 = Rx side, slice 2 = Tx side;
  must be finite; `[n_coeff, 11, 2]`, `n_coeff` of `preset`
- **`tx_pos`** — Tx positions in [m]; one row is used for all links; not used in single mobility
  (but validated); `[1, 3]` or `[n_link, 3]`
- **`rx_pos`** — Rx positions in [m]; `[n_link, 3]`
- **`floor_id`** *(optional)* — Floor index per link, a label starting at 0 (no index shift);
  non-negative integers; omitted or empty = all links on floor 0; `[n_link]`
- **`dual_mobility`** — `true`: Tx and Rx both move, the Tx side is evaluated as a second device;
  default: false
- **`preset`** — Sum-of-sinusoids preset, see [[sos_preset]]; must match [[lsf_init_phases]];
  default: `'Comb256'`
- **`horizontal`** — `true`: 2D correlation in the x-y plane (TR 38.901 behaviour); `false`: 3D
  correlation (legacy QuaDRiGa behaviour); must match [[lsf_init_phases]]; default: true

## Output:
- **`rnd`** — Random values; columns 1 to 9 Normal(0,1), columns 10 and 11 Uniform(0,1);
  `[n_link, 11]`

## See also:
- [[lsf_init_phases]] (draws the phases)
- [[path_loss]] (column 11 as `rnd_he`)
- [[lsp_generate]] (the large-scale parameters from `rnd`)
- [[sos_val]] (evaluation of a sum-of-sinusoids generator)
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

    // Floor IDs are labels (no index shift): read as double, checked, converted
    arma::uvec floor_id(floor_id_in.n_elem);
    for (arma::uword i = 0; i < floor_id_in.n_elem; ++i)
    {
        const double v = floor_id_in[i];
        if (!(v >= 0.0 && v < 18446744073709551616.0 && v == std::floor(v)))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Entries in 'floor_id' must be non-negative integers.");
        floor_id[i] = (arma::uword)v;
    }

    // Output allocation
    arma::mat rnd;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&rnd, rx_pos.n_rows, 11);

    // Wrap optional pointers
    const arma::uvec *p_floor_id = floor_id.empty() ? nullptr : &floor_id;

    // Call library function
    CALL_QD(quadriga_lib::lsf_draw(scenpar, phase, tx_pos, rx_pos, rnd, p_floor_id, dual_mobility, preset, horizontal));
}
