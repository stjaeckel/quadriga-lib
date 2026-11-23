// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_mex_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# GENERATE_ANTENNA
Generate an antenna array of a given type

- Creates every built-in antenna: single elements, uniform linear arrays, 3GPP TR 38.901 panel arrays, multibeam arrays,
  and the handheld UT and CPE devices. `type` selects the antenna; the other inputs set its parameters
- The result holds one pattern per element, the element positions and the port weights. Ports stay separate from the
  elements unless `apply_weights` is set; [[apply_port_weights]] combines them later
- Panel inputs use the TR notation: `P` polarizations, `M` rows, `N` columns, `Mg` and `Ng` panels, and `M_P`, `N_P`
  ports per panel and polarization
- Tuple inputs may be partial: entries left off at the end or set to NaN (0 in `panel`, `pol`, `ports`) take their
  default, e.g. `panel = [4, 8]` is one panel of 4 rows and 8 columns
- An input that the selected type does not take raises an error

## Usage:
```
% omni, xpol, dipole, half-wave-dipole
ant = quadriga_lib.generate_antenna( type, res, center_freq );

% Custom-shape element
ant = quadriga_lib.generate_antenna( 'custom', res, center_freq, element );

% Uniform linear array (omni element) and TR 38.901 panel array
ant = quadriga_lib.generate_antenna( 'ula', res, center_freq, [], panel, spacing, pol, ports, steer, ...
    [], [], [], pattern, [], apply_weights );
ant = quadriga_lib.generate_antenna( '3GPP', res, center_freq, element, panel, spacing, pol, ports, steer, ...
    [], [], [], pattern, [], apply_weights );

% Handheld UT and CPE
ant = quadriga_lib.generate_antenna( '3GPP-UT', res, center_freq, element, [], [], pol, [], [], ...
    [], device_size, locations, pattern );

% Multi-beam panel array
ant = quadriga_lib.generate_antenna( 'multibeam', res, center_freq, element, panel, spacing, pol, [], [], ...
    beams, [], [], [], separate_beams, apply_weights );
```

## Inputs:
- **`type`** — Antenna type:<br><br>
   | `type`                       | Antenna                                                                    | Inputs besides `res`, `center_freq`                                                |
   | ---------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
   | `omni`                       | Isotropic element, vertical polarization                                   | none                                                                               |
   | `xpol`                       | Two isotropic elements, vertical and horizontal polarization               | none                                                                               |
   | `dipole` (or `short-dipole`) | Short dipole, vertical polarization                                        | none                                                                               |
   | `half-wave-dipole`           | Half-wave dipole, vertical polarization                                    | none                                                                               |
   | `custom`                     | One element with given 3 dB beamwidths, peak gain equal to its directivity | `element`                                                                          |
   | `ula`                        | Uniform linear array of omni elements                                      | `panel`, `spacing`, `pol`, `ports`, `steer`, `pattern`, `apply_weights`            |
   | `3GPP` (or `3gpp`)           | Panel array of 3GPP TR 38.901 clause 7.3                                   | `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `pattern`, `apply_weights` |
   | `multibeam`                  | Panel array with beams toward given directions                             | `element`, `panel`, `spacing`, `pol`, `beams`, `separate_beams`, `apply_weights`   |
   | `3GPP-UT` (or `3gpp-ut`)     | Handheld UT of 3GPP TR 38.901 clause 7.3                                   | `element`, `pol`, `pattern`, `device_size`, `locations`                            |
   | `3GPP-CPE` (or `3gpp-cpe`)   | CPE of 3GPP TR 38.901 clause 7.3                                           | `element`, `pol`, `pattern`, `device_size`, `locations`                            |
- **`res`** — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; the grid has `floor(360 / res)` azimuth
  and `floor(180 / res)` elevation steps; NaN or <= 0: the default; ignored if `pattern` is given; default: 10 for
  `omni` and `xpol`, 1 otherwise
- **`center_freq`** — Center frequency in Hz; positive and finite; NaN: the default; default: 299792458 (wavelength 1 m)
- **`element`** — Element parameters per type:
  - `custom` `[az_3dB, el_3dB, floor_dB]`: 3 dB beamwidths in degrees (> 0, Inf: no taper); floor of the power pattern
    as a positive attenuation in dB below the peak (10: 10 dB below the peak; Inf: no floor); default `[90, 90, Inf]`
  - `multibeam`: the same, default `[120, 120, Inf]`
  - `3GPP` `[phi_3dB, theta_3dB, SLA_V, A_max, G_max]` of TR 38.901 Table 7.3-1: beamwidths in degrees (> 0, Inf: no
    taper); SLA_V and A_max as positive attenuations in dB below the peak, as in the TR (>= 0, Inf: no limit); G_max in
    dBi (finite or NaN); default `[65, 65, 30, 30, 8]`; not together with `pattern`
  - `3GPP-UT`: the same tuple, default `[125, 125, 22.5, 22.5, 5.3]` (Table 7.3-2); `3GPP-CPE`: default
    `[90, 90, 25, 25, 5]`; `[Inf, Inf, NaN, NaN, 0]` is isotropic
- **`panel`** — `[M, N, Mg, Ng]`: elements per column, columns per panel, panels per column, panels per row; at most 1e7
  elements per polarization (`ula`: `panel = 8` is a vertical column, `[1, 8]` a horizontal row); default: 1
- **`spacing`** — `[d_V, d_H, d_gV, d_gH]` in wavelengths: vertical and horizontal element spacing, vertical and
  horizontal panel spacing (center to center); finite and >= 0 (0 places elements or panels at the same position);
  default: `[0.5, d_V, M·d_V, N·d_H]`
- **`pol`** — `[polarization, model]`. Polarization: 1 = one element (ζ = 0°, vertical), 2 = two elements (ζ = 0° and
  90°), 3 = two elements (ζ = +45° and -45°). Model of TR 38.901 clause 7.3.2: 1 = the element pattern is turned by ζ
  about its boresight (Model-1, eq. 7.3-3), 2 = the power pattern is kept and the field is split by ζ (Model-2, eq.
  7.3-4, 7.3-5); default: `[1, 1]`; for `3GPP-UT` with polarization 2 or 3 the default model is 2, which gives the
  reference fields of the handheld model (eq. 7.3-6)
- **`ports`** — `[M_P, N_P]`: ports (transceiver units, TXRUs) per column and per row of a panel and polarization; must
  divide `M` and `N`; default: `[M, N]` (one port per element)
- **`steer`** — `[tilt, pan]` in degrees, -90 to 90: electrical downtilt and pan of the sub-array weights; `[1, 2]` for
  all panels or `[Mg·Ng, 2]` with one row per panel (panel order `mg`, `ng`); one column: pan 0; a non-zero tilt needs
  `M_P < M`, a non-zero pan `N_P < N`; default: 0
- **`beams`** — `[2, n_beams]` or `[3, n_beams]`: azimuth in degrees; elevation in degrees; weight (linear; 2 rows: 1);
  finite; default: `[0; 0; 1]`
- **`device_size`** — `[X, Y, Z]` in m (depth, width, height); NaN or left off: the default; default: `3GPP-UT`
  `[0.15, 0.07, 0]` with X, Y > 0 and Z = 0, `3GPP-CPE` `[0, 0.2, 0.2]` with X = 0 and Y, Z >= 0
- **`locations`** — TR location labels (`3GPP-UT` 1 to 8, `3GPP-CPE` 1 to 9), not array indices; any order, repeats
  allowed; default: all
- **`pattern`** — Element struct to use instead of the default element: its field patterns (`e_theta_re`, `e_theta_im`,
  `e_phi_re`, `e_phi_im` with their grids) are copied to every element position; its `element_pos`, port weights and
  `center_freq` are not used, the generator sets them; a one-element pattern gets the polarizations of `pol`, one with
  several elements is a polarization set used as given and needs `pol` `[1, 1]`; for `3GPP-UT`, a pattern on an
  irregular grid is first resampled to a regular one; default: `[]` (the element of the type)
- **`separate_beams`** — If `true`, one port per beam (weights ignored); default: `false`
- **`apply_weights`** — If `true`, combine each port into one pattern at its sub-array center; default: `false`

## Output:
- **`ant`** — Arrayant struct; panel arrays have `P·M·N·Mg·Ng` elements and `P·M_P·N_P·Mg·Ng` ports (`ula`, `3GPP`);
  with `apply_weights`, one element per port. Fields:<br><br>
   | Field             | Description                                               | Size                                   |
   | ----------------- | --------------------------------------------------------- | -------------------------------------- |
   | `e_theta_re`      | e-theta field component, real part                        | `[n_elevation, n_azimuth, n_elements]` |
   | `e_theta_im`      | e-theta field component, imaginary part                   | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_re`        | e-phi field component, real part                          | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_im`        | e-phi field component, imaginary part                     | `[n_elevation, n_azimuth, n_elements]` |
   | `azimuth_grid`    | Azimuth angles in rad, -π to π, strictly increasing       | `[1, n_azimuth]`                       |
   | `elevation_grid`  | Elevation angles in rad, -π/2 to π/2, strictly increasing | `[1, n_elevation]`                     |
   | `element_pos`     | Element (x,y,z) positions in meters                       | `[3, n_elements]`                      |
   | `port_weights_re` | Port weights, real part                                   | `[n_elements, n_ports]`                |
   | `port_weights_im` | Port weights, imaginary part                              | `[n_elements, n_ports]`                |
   | `center_freq`     | Center frequency in Hz                                    | scalar                                 |
   | `name`            | Name of the antenna array object                          | string                                 |

   As an input to other functions, the fields `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`,
   `elevation_grid` are mandatory; the others are optional (defaults: identity port weights, zero positions, 299792458
   Hz).

## Notes:
- **Element tuple** (`3GPP`, `3GPP-UT`, `3GPP-CPE`): entries left off at the end or set to NaN take their default, where
  `theta_3dB` defaults to `phi_3dB` and `A_max` to `SLA_V` (e.g. `3GPP` with `[90]` gives `[90, 90, 30, 30, 8]`);
  `G_max` left off takes the default of the type, NaN sets the peak gain to the directivity
- **Element and port order** (panel arrays): polarization fastest, then row `m` (upward along z), column `n` (along
  +y), panel row `mg`, panel column `ng`, centered on the origin; ports in the same order with `m_p`, `n_p`
- **Sub-array weights** (`ula`, `3GPP`; TR 38.901 eq. 7.3-1, along y by analogy): a port feeds `K = M / M_P` by
  `L = N / N_P` elements. With tilt θ_t and pan φ_p, element (k, l), counted from k = 0 and l = 0, gets

  `w(k,l) = exp(j 2π ((k - (K-1)/2) d_V sin θ_t - (l - (L-1)/2) d_H sin φ_p)) / √(K L)`

  A positive tilt points the beam down, a positive pan toward +y
- **Polarization** (TR 38.901 clause 7.3.2): `pol` 2 has the slants ζ = 0° and 90°, `pol` 3 ζ = +45° and -45°, in this
  element order. Model-1 turns the element by ζ about its boresight (eq. 7.3-3). Model-2 keeps the power pattern
  `A' = |F_θ|² + |F_φ|²` and sets `F_θ = √A' cos ζ`, `F_φ = √A' sin ζ` (eq. 7.3-4, 7.3-5)
- **`multibeam`**: maximum-ratio-transmission weights for the directions in `beams`, exact for one beam and approximate
  for several; only the ratios of the beam weights matter. One port per polarization, with `separate_beams` one per
  polarization and beam
- **`3GPP-UT`** (TR 38.901 clause 7.3): `P` elements at each entry of `locations` (Fig. 7.3-2), polarization fastest.
  Each element is rolled about its boresight by 90° (45° for two field patterns: `pol` 2 or a two-element `pattern`) and
  turned to the bearing `atan2(y, x)` of its location (Tables A.6-1, A.6-2). `pol` 1 and 2 with their default model are
  the TR model (eq. 7.3-6 to 7.3-8), other values are extensions. For an element symmetric under a 180° roll (every
  `element` tuple), the other z'' direction of Fig. 7.3-7 is the same antenna with a negated port weight
- **`3GPP-CPE`**: `P` elements at each entry of `locations` (Fig. 7.3-6 seen from the front, in the y-z plane), facing
  +x. TR 38.901 defines no CPE element; the default is the UE element of TR 38.803 Table 5.2.3.3-1

## See also:
- [[apply_port_weights]] (combines the ports into patterns)
- [[rotate_pattern]] (turns an antenna)
- [[calc_directivity]], [[calc_beamwidth]] (check the generated patterns)
- [[qdant_write]] (stores the antenna in a file)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 15)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Omitted or [] inputs are empty containers (not given); the C++ function fills the defaults per type
    const std::string type = qd_mex_get_string(prhs[0], "", "type");
    const double res = (nrhs < 2) ? NAN : qd_mex_get_scalar<double>(prhs[1], "res"); // [] gives NaN
    const double center_freq = (nrhs < 3) ? 299792458.0 : qd_mex_get_scalar<double>(prhs[2], "center_freq", 299792458.0);
    const arma::vec element = (nrhs < 4) ? arma::vec() : qd_mex_get_Col<double>(prhs[3], false, "element");
    const arma::uvec panel = (nrhs < 5) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[4], false, "panel");
    const arma::vec spacing = (nrhs < 6) ? arma::vec() : qd_mex_get_Col<double>(prhs[5], false, "spacing");
    const arma::uvec pol = (nrhs < 7) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[6], false, "pol");
    const arma::uvec ports = (nrhs < 8) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[7], false, "ports");
    const arma::mat steer = (nrhs < 9) ? arma::mat() : qd_mex_get_Mat<double>(prhs[8], false, "steer");
    const arma::mat beams = (nrhs < 10) ? arma::mat() : qd_mex_get_Mat<double>(prhs[9], false, "beams");
    const arma::vec device_size = (nrhs < 11) ? arma::vec() : qd_mex_get_Col<double>(prhs[10], false, "device_size");
    const arma::uvec locations = (nrhs < 12) ? arma::uvec() : qd_mex_get_Col<arma::uword>(prhs[11], false, "locations");
    const quadriga_lib::arrayant<double> pattern = (nrhs < 13 || mxIsEmpty(prhs[12])) ? quadriga_lib::arrayant<double>() : qd_mex_struct2arrayant(prhs[12], false, false, "pattern"); // Validated by the C++ function
    const bool separate_beams = (nrhs < 14) ? false : qd_mex_get_scalar<bool>(prhs[13], "separate_beams", false);
    const bool apply_weights = (nrhs < 15) ? false : qd_mex_get_scalar<bool>(prhs[14], "apply_weights", false);

    quadriga_lib::arrayant<double> ant;
    CALL_QD(ant = quadriga_lib::generate_antenna<double>(type, res, center_freq, element, panel, spacing, pol, ports, steer,
                                                         beams, device_size, locations, pattern, separate_beams, apply_weights));

    plhs[0] = qd_mex_arrayant2struct(ant);
}