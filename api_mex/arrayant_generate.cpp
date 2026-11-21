// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_GENERATE
Generate an array antenna of a given type

- One function for all antenna types. The panel types take the parameters of 3GPP TR 38.901 clause 7.3 as tuples: trailing entries may be omitted; 0 (in `panel`, `ports`, `pol`) or NaN (in `spacing`, `steer`, `element`) selects the default of that entry; more entries than listed raise an error.
- An input that a type does not use raises an error, also `separate_beams` or `apply_weights` set to `true`. The result has the given `center_freq`.
- Panel arrays: elements ordered polarization fastest, then `m` (upwards along z), `n` (along +y), `mg`, `ng`, positions centred on the origin; ports in the same order with `m_p`, `n_p`. Each port feeds a sub-array of `M / M_P` × `N / N_P` elements with the weights of TR 38.901 eq. 7.3-1 (`tilt`, and by analogy along y `pan`), stored in the coupling matrix.
- `multibeam`: maximum-ratio-transmission weights for the directions in `beams`; one port per polarization, with `separate_beams` one per polarization and beam.

## Usage:
```
% omni, xpol, dipole, half-wave-dipole
ant = quadriga_lib.arrayant_generate( type, res, center_freq );

% Custom-shape element
ant = quadriga_lib.arrayant_generate( 'custom', res, center_freq, [], [], [], [], [], element );

% Uniform linear array (omni element) and TR 38.901 panel array
ant = quadriga_lib.arrayant_generate( 'ula', res, center_freq, panel, spacing, ports, pol, steer, [], ...
    pattern, [], [], apply_weights );
ant = quadriga_lib.arrayant_generate( '3GPP', res, center_freq, panel, spacing, ports, pol, steer, element, ...
    pattern, [], [], apply_weights );

% Multi-beam panel array
ant = quadriga_lib.arrayant_generate( 'multibeam', res, center_freq, panel, spacing, [], pol, [], element, ...
    [], beams, separate_beams, apply_weights );
```

## Inputs:
- **`type`** — Antenna type and the inputs it takes besides `res` and `center_freq`:<br><br>
   | `type`                                           | Inputs                                                                             |
   | ------------------------------------------------ | ---------------------------------------------------------------------------------- |
   | `omni`, `xpol`                                   | none                                                                               |
   | `dipole` (or `short-dipole`), `half-wave-dipole` | none                                                                               |
   | `custom`                                         | `element`                                                                          |
   | `ula`                                            | `panel`, `spacing`, `ports`, `pol`, `steer`, `pattern`, `apply_weights`            |
   | `3GPP` (or `3gpp`)                               | `panel`, `spacing`, `ports`, `pol`, `steer`, `element`, `pattern`, `apply_weights` |
   | `multibeam`                                      | `panel`, `spacing`, `pol`, `element`, `beams`, `separate_beams`, `apply_weights`  |
- **`res`** — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: default; default: 10 for `omni` and `xpol`, 1 otherwise
- **`center_freq`** — Center frequency in Hz; positive and finite; default: 299792458
- **`panel`** — `[M, N, Mg, Ng]`: elements per column, columns per panel, panels per column, panels per row; at most 1e7 elements per polarization; default: 1 (`ula`: `panel = 8` is a vertical column, `[1, 8]` a horizontal row)
- **`spacing`** — `[d_V, d_H, d_gV, d_gH]` in wavelengths: vertical and horizontal element spacing, vertical and horizontal panel spacing (centre to centre); >= 0; default: `[0.5, d_V, M d_V, N d_H]`
- **`ports`** — `[M_P, N_P]`: ports per column and per row of a panel and polarization; must divide `M` and `N`; default: `[M, N]` (one port per element)
- **`pol`** — `[polarization, model]`: polarization 1 (vertical), 2 (H/V) or 3 (±45°); model 1 or 2 (TR 38.901 Model-1, Model-2); default: `[1, 1]`
- **`steer`** — `[tilt, pan]` in degrees, -90 to 90: electrical downtilt (positive down) and pan (positive towards +y); `[1, 2]` for all panels or `[Mg Ng, 2]` with one row per panel; one column: pan 0; a non-zero tilt needs `M_P < M`, a non-zero pan `N_P < N`; default: 0
- **`element`** — Element parameters per type:
  - `custom` `[az_3dB, el_3dB, floor_dB]`: 3 dB beamwidths in degrees (> 0, Inf: no taper); floor of the power pattern as a positive attenuation in dB below the peak (10: 10 dB below the peak; Inf: no floor); default `[90, 90, Inf]`
  - `multibeam`: the same, default `[120, 120, Inf]`
  - `3GPP` `[phi_3dB, theta_3dB, SLA_V, A_max, G_max]` of TR 38.901 Table 7.3-1: beamwidths in degrees (> 0, Inf: no taper); SLA_V and A_max as positive attenuations in dB below the peak, as in the TR (>= 0, Inf: no limit); G_max in dBi; default `[65, 65, 30, 30, 8]`; entries left off at the end or set to NaN take their default, where `theta_3dB` defaults to `phi_3dB` and `A_max` to `SLA_V` (e.g. `[90]` gives `[90, 90, 30, 30, 8]`); `G_max` left off is 8 dBi, NaN sets the peak gain to the directivity
- **`pattern`** — Element struct to use instead of the default element: its field patterns (`e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` with their grids) are copied to every element position; its `element_pos`, `coupling` and `center_freq` are not used, the generator sets them; a one-element pattern gets the polarizations of `pol`, one with several elements is a polarization set used as given and needs `pol` [1, 1]; default: the element of the type
- **`beams`** — `[2 or 3, n_beams]`: azimuth in degrees; elevation in degrees; weight (linear; 2 rows: 1); default: `[0; 0; 1]`
- **`separate_beams`** — If `true`, one port per beam (weights ignored); default: `false`
- **`apply_weights`** — If `true`, combine each port into one pattern at its sub-array centre; default: `false`

## Outputs:
- **`ant`** — Struct with fields:<br><br>
   | Field            | Description                                               | Size                                   |
   | ---------------- | --------------------------------------------------------- | -------------------------------------- |
   | `e_theta_re`     | e-theta field component, real part                        | `[n_elevation, n_azimuth, n_elements]` |
   | `e_theta_im`     | e-theta field component, imaginary part                   | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_re`       | e-phi field component, real part                          | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_im`       | e-phi field component, imaginary part                     | `[n_elevation, n_azimuth, n_elements]` |
   | `azimuth_grid`   | Azimuth angles in rad, -π to π, strictly increasing       | `[1, n_azimuth]`                       |
   | `elevation_grid` | Elevation angles in rad, -π/2 to π/2, strictly increasing | `[1, n_elevation]`                     |
   | `element_pos`    | Element (x,y,z) positions in meters                       | `[3, n_elements]`                      |
   | `coupling_re`    | Coupling matrix, real part                                | `[n_elements, n_ports]`                |
   | `coupling_im`    | Coupling matrix, imaginary part                           | `[n_elements, n_ports]`                |
   | `center_freq`    | Center frequency in Hz                                    | scalar                                 |
   | `name`           | Name of the array antenna object                          | string                                 |

   If `ant` is used as an input to other functions, fields `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid` are mandatory; remaining fields are optional (defaults: unit coupling, zero positions, 299792458 Hz).
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 13)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Omitted or [] inputs are empty containers (not given); the C++ function fills the defaults per type
    const std::string type = qd_mex_get_string(prhs[0], "", "type");
    const double res = (nrhs < 2) ? NAN : qd_mex_get_scalar<double>(prhs[1], "res"); // [] gives NaN
    const double center_freq = (nrhs < 3) ? 299792458.0 : qd_mex_get_scalar<double>(prhs[2], "center_freq", 299792458.0);
    const arma::uvec panel = (nrhs < 4) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[3], false, "panel");
    const arma::vec spacing = (nrhs < 5) ? arma::vec() : qd_mex_get_Col<double>(prhs[4], false, "spacing");
    const arma::uvec ports = (nrhs < 6) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[5], false, "ports");
    const arma::uvec pol = (nrhs < 7) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[6], false, "pol");
    const arma::mat steer = (nrhs < 8) ? arma::mat() : qd_mex_get_Mat<double>(prhs[7], false, "steer");
    const arma::vec element = (nrhs < 9) ? arma::vec() : qd_mex_get_Col<double>(prhs[8], false, "element");
    const quadriga_lib::arrayant<double> pattern = (nrhs < 10 || mxIsEmpty(prhs[9]))
                                                       ? quadriga_lib::arrayant<double>()
                                                       : qd_mex_struct2arrayant(prhs[9], false, false, "pattern"); // Validated by the C++ function
    const arma::mat beams = (nrhs < 11) ? arma::mat() : qd_mex_get_Mat<double>(prhs[10], false, "beams");
    const bool separate_beams = (nrhs < 12) ? false : qd_mex_get_scalar<bool>(prhs[11], "separate_beams", false);
    const bool apply_weights = (nrhs < 13) ? false : qd_mex_get_scalar<bool>(prhs[12], "apply_weights", false);

    quadriga_lib::arrayant<double> ant;
    CALL_QD(ant = quadriga_lib::arrayant_generate<double>(type, res, center_freq, panel, spacing, ports, pol, steer, element,
                                                          pattern, beams, separate_beams, apply_weights));

    plhs[0] = qd_mex_arrayant2struct(ant);
}
