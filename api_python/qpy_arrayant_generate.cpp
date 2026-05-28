// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# generate
Generates predefined array antenna models

- Dispatches to one of several C++ generator functions based on the `type` string
- Supported types: `omni`, `dipole` (or `short-dipole`), `half-wave-dipole`, `xpol`, `custom`,
  `ula`, `3gpp` (or `3GPP`), `multibeam`
- All arguments after `type` are keyword-only and type-specific; defaults are filled where omitted
- For `3gpp` and `ula`, a `pattern` dict can override the default per-element pattern
- `multibeam` combines beams via MRT weighting; set `separate_beams=True` for one beam per direction

## Usage:
```
# Simple antennas (v-pol)
ant = quadriga_lib.arrayant.generate('omni')
ant = quadriga_lib.arrayant.generate('dipole')
ant = quadriga_lib.arrayant.generate('half-wave-dipole')

# Cross-polarized isotropic
ant = quadriga_lib.arrayant.generate('xpol')

# Custom 3dB beamwidth
ant = quadriga_lib.arrayant.generate('custom', az_3dB=90.0, el_3dB=90.0, rear_gain_lin=0.0)

# Uniform linear array
ant = quadriga_lib.arrayant.generate('ula', N=4, freq=2.4e9, spacing=0.7, pol=1)

# 3GPP-NR array (default 3GPP element pattern)
ant = quadriga_lib.arrayant.generate('3gpp', M=2, N=2, freq=3.7e9, pol=1, spacing=0.7)

# 3GPP-NR array with custom per-element pattern dict
ant = quadriga_lib.arrayant.generate('3gpp', M=2, N=2, freq=3.7e9, pol=1, pattern=my_pattern)

# Multi-beam M×N array
ant = quadriga_lib.arrayant.generate('multibeam', M=6, N=6, freq=3.7e9, pol=1, \
    beam_az=[-30.0, 30.0], beam_el=[0.0, 0.0], separate_beams=False)
```

## Inputs (common):
- **`type`** — Antenna model type; str
- **`res`** — Pattern sampling grid resolution in degrees; default: 10 for `omni` and `xpol`, 1 otherwise
- **`freq`** — Center frequency in Hz; default: 299792458 (λ = 1 m)

## Inputs (custom, 3gpp, multibeam):
- **`az_3dB`** — Azimuth 3dB beamwidth in degrees; default: 90 for `custom`, 67 for `3gpp`, 120 for `multibeam`
- **`el_3dB`** — Elevation 3dB beamwidth in degrees; same defaults as `az_3dB`
- **`rear_gain_lin`** — Back-to-front gain ratio (linear scale); default: 0.0

## Inputs (ula, 3gpp, multibeam):
- **`M`** — Number of vertical elements per panel; default: 1; ignored for `ula`
- **`N`** — Number of horizontal elements per panel; default: 1
- **`pol`** — Polarization mode (1..6); default: 1<br><br>
   | `pol` | Description                                              |
   | :---: | -------------------------------------------------------- |
   | 1     | Vertical polarization                                    |
   | 2     | H/V polarization (2NM elements)                          |
   | 3     | ±45° polarization (2NM elements)                         |
   | 4     | Vertical, vertical elements combined (N elements)        |
   | 5     | H/V, vertical elements combined (2N elements)            |
   | 6     | ±45°, vertical elements combined (2N elements)           |
- **`spacing`** — Inter-element spacing in wavelengths; default: 0.5

## Inputs (3gpp only):
- **`tilt`** — Electrical downtilt in degrees; applies to `pol = 4/5/6`; default: 0.0
- **`Mg`** — Number of vertically stacked panels; default: 1
- **`Ng`** — Number of horizontally stacked panels; default: 1
- **`dgv`** — Panel spacing in vertical direction in wavelengths; default: 0.5
- **`dgh`** — Panel spacing in horizontal direction in wavelengths; default: 0.5

## Inputs (ula, 3gpp):
- **`pattern`** — Custom per-element pattern dict (same field layout as the output);
  overrides default 3GPP/ULA element pattern; default: empty dict (use built-in pattern)

## Inputs (multibeam only):
- **`beam_az`** — Azimuth beam angles in degrees; `(n_beams,)`; default: `[0.0]`
- **`beam_el`** — Elevation beam angles in degrees; `(n_beams,)`; default: `[0.0]`
- **`beam_weight`** — Per-beam scaling factor; normalized so the sum equals 1; same length as `beam_az`; default: `[1.0, 1.0, ...]` (all-ones)
- **`separate_beams`** — Produce one independent beam per angle pair (weights ignored); default: False
- **`apply_weights`** — Apply the beamforming weights to the output coupling matrix; default: False

## Outputs:
- **`ant`** — Dict with fields:<br><br>
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
MD!*/

py::dict arrayant_generate(const std::string &type,
                           double res,
                           double freq,
                           double az_3dB,
                           double el_3dB,
                           double rear_gain_lin,
                           arma::uword M,
                           arma::uword N,
                           unsigned pol,
                           double tilt,
                           double spacing,
                           arma::uword Mg,
                           arma::uword Ng,
                           double dgv,
                           double dgh,
                           py::handle beam_az,
                           py::handle beam_el,
                           py::handle beam_weight,
                           bool separate_beams,
                           bool apply_weights,
                           const py::dict &pattern)
{
    // Set default resolution by type when res == 0
    if (res == 0.0)
        res = (type == "omni" || type == "xpol") ? 10.0 : 1.0;

    quadriga_lib::arrayant<double> arrayant;

    if (type == "omni")
        arrayant = quadriga_lib::generate_arrayant_omni<double>(res);
    else if (type == "dipole" || type == "short-dipole")
        arrayant = quadriga_lib::generate_arrayant_dipole<double>(res);
    else if (type == "half-wave-dipole")
        arrayant = quadriga_lib::generate_arrayant_half_wave_dipole<double>(res);
    else if (type == "xpol")
        arrayant = quadriga_lib::generate_arrayant_xpol<double>(res);
    else if (type == "custom")
    {
        double az = (az_3dB > 0.0) ? az_3dB : 90.0;
        double el = (el_3dB > 0.0) ? el_3dB : 90.0;
        arrayant = quadriga_lib::generate_arrayant_custom<double>(az, el, rear_gain_lin, res);
    }
    else if (type == "ula")
    {
        if (pattern.size() != 0) // Use custom pattern dict
        {
            const auto custom_array = qd_python_dict2arrayant(pattern, true);
            arrayant = quadriga_lib::generate_arrayant_ula<double>(N, freq, spacing, &custom_array);
        }
        else if (pol == 2 || pol == 5 || pol == 3 || pol == 6) // Dual-polarized base pattern
        {
            quadriga_lib::arrayant<double> custom_array;
            if (pol == 2 || pol == 5)
                custom_array = quadriga_lib::generate_arrayant_xpol<double>(res);
            else // pol == 3 || pol == 6
            {
                custom_array = quadriga_lib::generate_arrayant_omni<double>(res);
                custom_array.copy_element(0, 1);
                custom_array.rotate_pattern(45.0, 0.0, 0.0, 2, 0);
                custom_array.rotate_pattern(-45.0, 0.0, 0.0, 2, 1);
            }
            arrayant = quadriga_lib::generate_arrayant_ula<double>(N, freq, spacing, &custom_array);
        }
        else // Isotropic per-element pattern
            arrayant = quadriga_lib::generate_arrayant_ula<double>(N, freq, spacing, nullptr, res);
        arrayant.name = "ula";
    }
    else if (type == "3GPP" || type == "3gpp")
    {
        if (pattern.size() != 0) // Use custom pattern dict
        {
            auto custom_array = qd_python_dict2arrayant(pattern, true);
            arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing,
                                                                    Mg, Ng, dgv, dgh, &custom_array);
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
            arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing,
                                                                    Mg, Ng, dgv, dgh, &custom_array);
        }
        else // Use 3GPP default pattern
            arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing,
                                                                    Mg, Ng, dgv, dgh, nullptr, res);
    }
    else if (type == "multibeam")
    {
        const auto az = qd_python_numpy2arma_Col<double>(beam_az, true);
        const auto el = qd_python_numpy2arma_Col<double>(beam_el, true);
        const auto weight = qd_python_numpy2arma_Col<double>(beam_weight, true);

        arma::Col<double> az_default = {0.0};
        arma::Col<double> el_default = {0.0};
        const arma::Col<double> &az_use = az.empty() ? az_default : az;
        const arma::Col<double> &el_use = el.empty() ? el_default : el;
        arma::Col<double> weight_use = weight.empty()
                                           ? arma::Col<double>(az_use.n_elem, arma::fill::ones)
                                           : weight;

        arrayant = quadriga_lib::generate_arrayant_multibeam<double>(M, N, az_use, el_use, weight_use,
                                                                     freq, pol, spacing,
                                                                     az_3dB, el_3dB, rear_gain_lin, res,
                                                                     separate_beams, apply_weights);
    }
    else
        throw std::invalid_argument("Array type not supported!");

    arrayant.center_frequency = freq;
    return qd_python_arrayant2dict(arrayant);
}

// pybind11 declaration:
// m.def("generate", &arrayant_generate,
//       py::arg("type"),
//       py::arg("res") = 0.0,
//       py::arg("freq") = 299792458.0,
//       py::arg("az_3dB") = 0.0,
//       py::arg("el_3dB") = 0.0,
//       py::arg("rear_gain_lin") = 0.0,
//       py::arg("M") = 1,
//       py::arg("N") = 1,
//       py::arg("pol") = 1,
//       py::arg("tilt") = 0.0,
//       py::arg("spacing") = 0.5,
//       py::arg("Mg") = 1,
//       py::arg("Ng") = 1,
//       py::arg("dgv") = 0.5,
//       py::arg("dgh") = 0.5,
//       py::arg("beam_az") = py::none(),
//       py::arg("beam_el") = py::none(),
//       py::arg("beam_weight") = py::none(),
//       py::arg("separate_beams") = false,
//       py::arg("apply_weights") = false,
//       py::arg("pattern") = py::dict());