// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# ARRAYANT_GENERATE
Generates predefined array antenna models

- Dispatches to one of several C++ generator functions based on the `type` string
- Supported types: `omni`, `dipole` (or `short-dipole`), `half-wave-dipole`, `xpol`, `custom`,
  `ula`, `3GPP` (or `3gpp`), `multibeam`, `multibeam_sep`
- All positional arguments after `type` are optional and type-specific; use `[]` to skip and fall
  back to defaults

## Usage:
```
% Simple antennas (v-pol)
ant = quadriga_lib.arrayant_generate('omni', res);
ant = quadriga_lib.arrayant_generate('dipole', res);
ant = quadriga_lib.arrayant_generate('half-wave-dipole', res);

% Cross-polarized isotropic
ant = quadriga_lib.arrayant_generate('xpol', res);

% Custom 3dB beamwidth
ant = quadriga_lib.arrayant_generate('custom', res, freq, az_3dB, el_3dB, rear_gain_lin);

% Uniform linear array (N horizontal elements, half-wavelength spacing by default)
ant = quadriga_lib.arrayant_generate('ula', res, freq, [], [], [], [], N, pol, [], spacing);

% Uniform linear array with custom per-element pattern struct
ant = quadriga_lib.arrayant_generate('ula', res, freq, [], [], [], [], N, [], [], spacing, ...
                                     [], [], [], [], pattern);

% 3GPP-NR array (default 3GPP element pattern)
ant = quadriga_lib.arrayant_generate('3GPP', res, freq, [], [], [], 
                                     M, N, pol, tilt, spacing, Mg, Ng, dgv, dgh);

% 3GPP-NR array with custom element beamwidth
ant = quadriga_lib.arrayant_generate('3GPP', res, freq, az_3dB, el_3dB, rear_gain_lin, ...
                                     M, N, pol, tilt, spacing, Mg, Ng, dgv, dgh);

% 3GPP-NR array with custom per-element pattern struct
ant = quadriga_lib.arrayant_generate('3GPP', res, freq, [], [], [], ...
                                     M, N, pol, tilt, spacing, Mg, Ng, dgv, dgh, pattern);

% Multi-beam M×N array (one combined beam / one beam per direction)
ant = quadriga_lib.arrayant_generate('multibeam', res, freq, az_3dB, el_3dB, rear_gain_lin, ...
                                     M, N, pol, beam_angles, spacing);
ant = quadriga_lib.arrayant_generate('multibeam_sep', res, freq, az_3dB, el_3dB, rear_gain_lin, ...
                                     M, N, pol, beam_angles, spacing);

% Separate outputs (must request exactly 11)
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, ...
    element_pos, coupling_re, coupling_im, center_freq, name] = quadriga_lib.arrayant_generate( ... );
```

## Inputs (common):
- **`type`** — Antenna model type; string (see Usage above)
- **`res`** *(optional)* — Pattern sampling grid resolution in degrees; default: 1
- **`freq`** *(optional)* — Center frequency in Hz; default: 299792458

## Inputs (type custom, 3GPP, multibeam, multibeam_sep):
- **`az_3dB`** *(optional)* — Azimuth 3dB beamwidth in degrees; default: 90 for `custom`, else
  triggers 3GPP / multibeam internal defaults
- **`el_3dB`** *(optional)* — Elevation 3dB beamwidth in degrees; default: 90 for `custom`, else
  triggers 3GPP / multibeam internal defaults
- **`rear_gain_lin`** *(optional)* — Front-to-back gain ratio (linear); default: 0

## Inputs (type ula, 3GPP, multibeam, multibeam_sep):
- **`M`** *(optional)* — Number of vertical elements per panel; default: 1; ignored for `ula`
- **`N`** *(optional)* — Number of horizontal elements per panel; default: 1
- **`pol`** *(optional)* — Polarization mode; default: 1:<br><br>
   | `pol` | Description                                              |
   | ----- | -------------------------------------------------------- |
   | 1     | Vertical polarization                                    |
   | 2     | H/V polarization (2NM elements)                          |
   | 3     | ±45° polarization (2NM elements)                         |
   | 4     | Vertical, vertical elements combined (N elements)        |
   | 5     | H/V, vertical elements combined (2N elements)            |
   | 6     | ±45°, vertical elements combined (2N elements)           |
- **`spacing`** *(optional)* — Inter-element spacing in wavelengths; default: 0.5

## Inputs (type 3GPP only):
- **`tilt`** *(optional)* — Electrical downtilt in degrees; applies to `pol` 4–6; default: 0
- **`Mg`** *(optional)* — Number of vertically stacked panels; default: 1
- **`Ng`** *(optional)* — Number of horizontally stacked panels; default: 1
- **`dgv`** *(optional)* — Panel spacing in vertical direction in wavelengths; default: 0.5
- **`dgh`** *(optional)* — Panel spacing in horizontal direction in wavelengths; default: 0.5

## Inputs (type ula, 3GPP):
- **`pattern`** *(optional)* — Custom per-element pattern struct used for 3GPP or ULA; same format as
  outputs; overrides default 3GPP/ULA element pattern; other struct fields if present are ignored

## Inputs (type multibeam, multibeam_sep):
- **`beam_angles`** — Beam steering angles, `[3, n_beams]`; rows are
  `[azimuth_deg; elevation_deg; weight]`; `multibeam` combines beams via MRT weighting,
  `multibeam_sep` produces one independent beam per column (weights ignored)

## Outputs:
- **`ant`** — Struct with fields:<br><br>
   | Field            | Description                                     | Size                                   |
   | ---------------- | ----------------------------------------------- | -------------------------------------- |
   | `e_theta_re`     | e-theta field component, real part              | `[n_elevation, n_azimuth, n_elements]` |
   | `e_theta_im`     | e-theta field component, imaginary part         | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_re`       | e-phi field component, real part                | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_im`       | e-phi field component, imaginary part           | `[n_elevation, n_azimuth, n_elements]` |
   | `azimuth_grid`   | Azimuth angles in rad, -π to π, sorted          | `[n_azimuth]`                          |
   | `elevation_grid` | Elevation angles in rad, -π/2 to π/2, sorted    | `[n_elevation]`                        |
   | `element_pos`    | Element (x,y,z) positions in meters             | `[3, n_elements]`                      |
   | `coupling_re`    | Coupling matrix, real part                      | `[n_elements, n_ports]`                |
   | `coupling_im`    | Coupling matrix, imaginary part                 | `[n_elements, n_ports]`                |
   | `center_freq`    | Center frequency in Hz                          | scalar                                 |
   | `name`           | Name of the array antenna object                | string                                 |
   If `ant` is used as an input to other fuctions, fields `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, 
   `elevation_grid` are mandatory; remaining fields are optional (defaults: unit coupling, zero positions, 299792458 Hz)."
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Arrayant type name missing.");

    if (nrhs > 16)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs == 0)
        return;

    std::string array_type = qd_mex_get_string(prhs[0]);
    double res = (nrhs < 2) ? 1.0 : qd_mex_get_scalar<double>(prhs[1], "res", 1.0);
    double freq = (nrhs < 3) ? 299792458.0 : qd_mex_get_scalar<double>(prhs[2], "freq", 299792458.0);
    double az_3dB = (nrhs < 4) ? 0.0 : qd_mex_get_scalar<double>(prhs[3], "az_3dB", 0.0);
    double el_3dB = (nrhs < 5) ? 0.0 : qd_mex_get_scalar<double>(prhs[4], "el_3dB", 0.0);
    double rear_gain_lin = (nrhs < 6) ? 0.0 : qd_mex_get_scalar<double>(prhs[5], "rear_gain_lin", 0.0);
    unsigned M = (nrhs < 7) ? 1 : qd_mex_get_scalar<unsigned>(prhs[6], "M", 1);
    unsigned N = (nrhs < 8) ? 1 : qd_mex_get_scalar<unsigned>(prhs[7], "N", 1);
    unsigned pol = (nrhs < 9) ? 1 : qd_mex_get_scalar<unsigned>(prhs[8], "pol", 1);
    double tilt = (nrhs < 10) ? 0.0 : qd_mex_get_scalar<double>(prhs[9], "tilt", 0.0);
    double spacing = (nrhs < 11) ? 0.5 : qd_mex_get_scalar<double>(prhs[10], "spacing", 0.5);
    unsigned Mg = (nrhs < 12) ? 1 : qd_mex_get_scalar<unsigned>(prhs[11], "Mg", 1);
    unsigned Ng = (nrhs < 13) ? 1 : qd_mex_get_scalar<unsigned>(prhs[12], "Ng", 1);
    double dgv = (nrhs < 14) ? 0.5 : qd_mex_get_scalar<double>(prhs[13], "dgv", 0.5);
    double dgh = (nrhs < 15) ? 0.5 : qd_mex_get_scalar<double>(prhs[14], "dgh", 0.5);

    quadriga_lib::arrayant<double> arrayant;

    if (array_type == "omni")
        CALL_QD(arrayant = quadriga_lib::generate_arrayant_omni<double>(res));
    else if (array_type == "dipole" || array_type == "short-dipole")
        CALL_QD(arrayant = quadriga_lib::generate_arrayant_dipole<double>(res));
    else if (array_type == "half-wave-dipole")
        CALL_QD(arrayant = quadriga_lib::generate_arrayant_half_wave_dipole<double>(res));
    else if (array_type == "xpol")
        CALL_QD(arrayant = quadriga_lib::generate_arrayant_xpol<double>(res));
    else if (array_type == "custom")
    {
        // Match C++ defaults (90°/90°) when user passed nothing or zero
        double az = (az_3dB > 0.0) ? az_3dB : 90.0;
        double el = (el_3dB > 0.0) ? el_3dB : 90.0;
        CALL_QD(arrayant = quadriga_lib::generate_arrayant_custom<double>(az, el, rear_gain_lin, res));
    }
    else if (array_type == "ula")
    {
        if (nrhs > 15) // Use custom pattern
        {
            auto custom_array = qd_mex_struct2arrayant(prhs[15], true);
            CALL_QD(arrayant = quadriga_lib::generate_arrayant_ula<double>(N, freq, spacing, &custom_array));
        }
        else if (pol == 2 || pol == 5 || pol == 3 || pol == 6) // Build dual-polarized base pattern and pass to ULA
        {
            quadriga_lib::arrayant<double> custom_array;
            if (pol == 2 || pol == 5)
            {
                CALL_QD(custom_array = quadriga_lib::generate_arrayant_xpol(res));
            }
            else if (pol == 3 || pol == 6)
            {
                CALL_QD(custom_array = quadriga_lib::generate_arrayant_omni(res));
                custom_array.copy_element(0, 1);
                custom_array.rotate_pattern(45.0, 0.0, 0.0, 2, 0);
                custom_array.rotate_pattern(-45.0, 0.0, 0.0, 2, 1);
            }
            CALL_QD(arrayant = quadriga_lib::generate_arrayant_ula<double>(N, freq, spacing, &custom_array));
        }
        else // Use isotropic
            CALL_QD(arrayant = quadriga_lib::generate_arrayant_ula<double>(N, freq, spacing, nullptr, res));
        arrayant.name = "ula";
    }
    else if (array_type == "3GPP" || array_type == "3gpp")
    {
        if (nrhs > 15) // Use custom pattern
        {
            auto custom_array = qd_mex_struct2arrayant(prhs[15], true);
            CALL_QD(arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, &custom_array));
        }
        else if (az_3dB > 0.0 && el_3dB > 0.0) // Use custom beam width
        {
            auto custom_array = quadriga_lib::generate_arrayant_custom<double>(az_3dB, el_3dB, rear_gain_lin, res);
            if (pol == 2 || pol == 5)
            {
                custom_array.copy_element(0, 1);
                custom_array.rotate_pattern(90.0, 0.0, 0.0, 2, 1);
            }
            else if (pol == 3 || pol == 6)
            {
                custom_array.copy_element(0, 1);
                custom_array.rotate_pattern(45.0, 0.0, 0.0, 2, 0);
                custom_array.rotate_pattern(-45.0, 0.0, 0.0, 2, 1);
            }
            CALL_QD(arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, &custom_array));
        }
        else // Use 3GPP default pattern
            CALL_QD(arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, nullptr, res));
    }
    else if (array_type == "multibeam" || array_type == "multibeam_sep")
    {
        arma::mat beam_angles(3, 1);
        if (nrhs > 9)
            beam_angles = qd_mex_get_Mat<double>(prhs[9]);
        else
            beam_angles(2, 0) = 1.0;

        if (beam_angles.n_rows < 2)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'beam_angles' must have at least 2 rows for a multi-beam antenna.");

        arma::vec az = beam_angles.row(0).as_col();
        arma::vec el = beam_angles.row(1).as_col();
        arma::vec weight = (beam_angles.n_rows > 2) ? beam_angles.row(2).as_col() : arma::vec(beam_angles.n_cols, arma::fill::ones);

        bool separate_beams = array_type == "multibeam_sep";
        CALL_QD(arrayant = quadriga_lib::generate_arrayant_multibeam<double>(M, N, az, el, weight, freq, pol, spacing,
                                                                             az_3dB, el_3dB, rear_gain_lin, res, separate_beams));
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Array type not supported!");

    // Set center frequency for all types
    arrayant.center_frequency = freq;

    if (nlhs == 1) // Output as struct
        plhs[0] = qd_mex_arrayant2struct(arrayant);
    else if (nlhs == 11) // Separate outputs
    {
        plhs[0] = qd_mex_copy2matlab(&arrayant.e_theta_re);
        plhs[1] = qd_mex_copy2matlab(&arrayant.e_theta_im);
        plhs[2] = qd_mex_copy2matlab(&arrayant.e_phi_re);
        plhs[3] = qd_mex_copy2matlab(&arrayant.e_phi_im);
        plhs[4] = qd_mex_copy2matlab(&arrayant.azimuth_grid, true);
        plhs[5] = qd_mex_copy2matlab(&arrayant.elevation_grid, true);
        plhs[6] = qd_mex_copy2matlab(&arrayant.element_pos);
        plhs[7] = qd_mex_copy2matlab(&arrayant.coupling_re);
        plhs[8] = qd_mex_copy2matlab(&arrayant.coupling_im);
        plhs[9] = qd_mex_copy2matlab(&arrayant.center_frequency);
        plhs[10] = mxCreateString(arrayant.name.c_str());
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");
}