// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"



/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# generate
Generates predefined array antenna models

- Dispatches to one of several C++ generator functions based on the `type` string
- Supported types: `omni`, `dipole` (or `short-dipole`), `half-wave-dipole`, `xpol`, `custom`,
  `ula`, `3gpp` (or `3GPP`), `multibeam`
- `res` applies to all types; the arguments after it are keyword-only and type-specific; defaults are filled
  where omitted
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
- **`res`** — Pattern sampling grid resolution in degrees, 0.1 to 90; default: 10 for `omni` and `xpol`, 1 otherwise
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
  overrides default 3GPP/ULA element pattern; only the radiation pattern is used: element positions,
  coupling and center frequency of the pattern are ignored; default: `None` (use built-in pattern)

## Inputs (multibeam only):
- **`beam_az`** — Azimuth beam angles in degrees; `(n_beams,)`; default: `[0.0]`
- **`beam_el`** — Elevation beam angles in degrees; `(n_beams,)`; default: `[0.0]`
- **`beam_weight`** — Per-beam scaling factor; not normalized; same length as `beam_az`; default: `[1.0, 1.0, ...]` (all-ones)
- **`separate_beams`** — Produce one independent beam per angle pair (weights ignored); default: False
- **`apply_weights`** — Apply the beamforming weights to the output coupling matrix; default: False

## Outputs:
- **`ant`** — Dict with fields:<br><br>
   | Field            | Description                                               | Size                                   |
   | ---------------- | --------------------------------------------------------- | -------------------------------------- |
   | `e_theta_re`     | e-theta field component, real part                        | `[n_elevation, n_azimuth, n_elements]` |
   | `e_theta_im`     | e-theta field component, imaginary part                   | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_re`       | e-phi field component, real part                          | `[n_elevation, n_azimuth, n_elements]` |
   | `e_phi_im`       | e-phi field component, imaginary part                     | `[n_elevation, n_azimuth, n_elements]` |
   | `azimuth_grid`   | Azimuth angles in rad, -π to π, strictly increasing       | `[n_azimuth]`                          |
   | `elevation_grid` | Elevation angles in rad, -π/2 to π/2, strictly increasing | `[n_elevation]`                        |
   | `element_pos`    | Element (x,y,z) positions in meters                       | `[3, n_elements]`                      |
   | `coupling_re`    | Coupling matrix, real part                                | `[n_elements, n_ports]`                |
   | `coupling_im`    | Coupling matrix, imaginary part                           | `[n_elements, n_ports]`                |
   | `center_freq`    | Center frequency in Hz; inputs accept `(1,)`              | scalar                                 |
   | `name`           | Name of the array antenna object                          | string                                 |
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
                           py::handle separate_beams,
                           py::handle apply_weights,
                           const py::handle &pattern)
{
    const bool separate_beams_b = qd_python_flag(separate_beams, false, "separate_beams");
    const bool apply_weights_b = qd_python_flag(apply_weights, false, "apply_weights");

    py::dict pattern_d;
    if (!pattern.is_none())
    {
        if (!py::isinstance<py::dict>(pattern))
            throw py::type_error("Input 'pattern' must be a dict or None.");
        pattern_d = py::reinterpret_borrow<py::dict>(pattern);
    }
    const arma::uvec el_0 = {0}, el_1 = {1}; // Elements turned by the polarization rotations of custom patterns

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
        if (!pattern_d.empty()) // Use custom pattern dict
        {
            const auto custom_array = qd_python_dict2arrayant(pattern_d, true);
            arrayant = quadriga_lib::generate_arrayant_ula<double>(N, freq, spacing, custom_array);
        }
        else if (pol == 2 || pol == 5 || pol == 3 || pol == 6) // Dual-polarized base pattern
        {
            quadriga_lib::arrayant<double> custom_array;
            if (pol == 2 || pol == 5)
                custom_array = quadriga_lib::generate_arrayant_xpol<double>(res);
            else // pol == 3 || pol == 6
            {
                custom_array = quadriga_lib::generate_arrayant_omni<double>(res);
                quadriga_lib::arrayant_copy_element(custom_array, 0, {1});
                quadriga_lib::arrayant_rotate_pattern(custom_array, arma::vec{arma::datum::pi / 4.0, 0.0, 0.0}, 2, el_0);
                quadriga_lib::arrayant_rotate_pattern(custom_array, arma::vec{-arma::datum::pi / 4.0, 0.0, 0.0}, 2, el_1);
            }
            arrayant = quadriga_lib::generate_arrayant_ula<double>(N, freq, spacing, custom_array);
        }
        else // Isotropic per-element pattern
            arrayant = quadriga_lib::generate_arrayant_ula<double>(N, freq, spacing, {}, res);
        arrayant.name = "ula";
    }
    else if (type == "3GPP" || type == "3gpp")
    {
        if (!pattern_d.empty()) // Use custom pattern dict
        {
            auto custom_array = qd_python_dict2arrayant(pattern_d, true);
            arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing,
                                                                    Mg, Ng, dgv, dgh, custom_array);
        }
        else if (az_3dB > 0.0 && el_3dB > 0.0) // Use custom beam width
        {
            auto custom_array = quadriga_lib::generate_arrayant_custom<double>(az_3dB, el_3dB, rear_gain_lin, res);
            if (pol == 2 || pol == 5)
            {
                quadriga_lib::arrayant_copy_element(custom_array, 0, {1});
                quadriga_lib::arrayant_rotate_pattern(custom_array, arma::vec{arma::datum::pi / 2.0, 0.0, 0.0}, 2, el_1);
            }
            else if (pol == 3 || pol == 6)
            {
                quadriga_lib::arrayant_copy_element(custom_array, 0, {1});
                quadriga_lib::arrayant_rotate_pattern(custom_array, arma::vec{arma::datum::pi / 4.0, 0.0, 0.0}, 2, el_0);
                quadriga_lib::arrayant_rotate_pattern(custom_array, arma::vec{-arma::datum::pi / 4.0, 0.0, 0.0}, 2, el_1);
            }
            arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing,
                                                                    Mg, Ng, dgv, dgh, custom_array);
        }
        else // Use 3GPP default pattern
            arrayant = quadriga_lib::generate_arrayant_3GPP<double>(M, N, freq, pol, tilt, spacing,
                                                                    Mg, Ng, dgv, dgh, {}, res);
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

        // Match the C++ default element beamwidth (120°/120°) when the user passed nothing or zero
        double az_bw = (az_3dB > 0.0) ? az_3dB : 120.0;
        double el_bw = (el_3dB > 0.0) ? el_3dB : 120.0;
        arrayant = quadriga_lib::generate_arrayant_multibeam<double>(M, N, az_use, el_use, weight_use,
                                                                     freq, pol, spacing,
                                                                     az_bw, el_bw, rear_gain_lin, res,
                                                                     separate_beams_b, apply_weights_b);
    }
    else
        throw std::invalid_argument("Array type not supported!");

    arrayant.center_freq = freq;
    return qd_python_arrayant2dict(arrayant);
}

/*!MD
# generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

- Returns a multi-frequency arrayant dict; pattern fields are 4D arrays where the 4th dim is frequency; for a
  single frequency the dict is a single-frequency arrayant dict (3D pattern fields, scalar `center_freq`)
- Directivity is stored as real-valued data in `e_theta_re`; dipole rear hemisphere is encoded with negative sign for 180° phase inversion
- Multi-driver systems (e.g. two-way) are built by calling this per driver and combining via [[concat]];
  crossover behavior emerges from overlapping bandpass responses
- Frequency response is a Butterworth-style bandpass: `H(f) = 1/sqrt(1+(f_low/f)^(2n)) · 1/sqrt(1+(f/f_high)^(2n))`,
  where `n = slope_dB_per_octave / 6`; -3 dB at the cutoff frequencies
- Sensitivity scales amplitude linearly relative to 85 dB SPL: `sens_lin = 10^((sensitivity_dB - 85) / 20)`
- If `center_freq` is None or empty, third-octave band centers are auto-generated from one band below
  `lower_cutoff` to one band above `upper_cutoff`, clipped to 20-20000 Hz
- Speed of sound assumed to be 343 m/s
- Driver models (`driver_type`):
  - `piston` - circular piston in baffle, `D(theta) = 2·J1(ka·sin theta)/(ka·sin theta)`, rotationally symmetric, narrows with increasing `ka`
  - `horn` - separable cosine-power `cos^n(angle)` with frequency-dependent blend toward omni below `horn_control_freq`
  - `omni` - frequency-independent omnidirectional pattern
- Enclosure models (`radiation_type`):
  - `monopole` - no modification
  - `hemisphere` - sealed box with baffle-step transition, `f_baffle = c/(pi·sqrt(W·H))`
  - `dipole` - figure-8, `R = abs(cos(theta_off))` with sign inversion in rear hemisphere
  - `cardioid` - `R = 0.5·(1+cos(theta_off))`
- For `horn`, `hor_coverage_deg`, `ver_coverage_deg` and `horn_control_freq` set to NaN (the default) or <= 0 select 90°, 60°
  and `f_ctrl = c/(2pi·radius)`

## Usage:
```
speaker = quadriga_lib.arrayant.generate_speaker( driver_type, *, radius, lower_cutoff, upper_cutoff, \
    lower_rolloff_slope_dB, upper_rolloff_slope_dB, sensitivity_dB, radiation_type, hor_coverage_deg, \
    ver_coverage_deg, horn_control_freq, baffle_width, baffle_height, center_freq, angular_resolution_deg )
```

## Inputs:
- **`driver_type`** — Driver directivity model: `piston`, `horn`, or `omni`; default: `piston`
- **`radius`** — Effective radiating radius in m, > 0; cone/dome radius for piston, mouth radius for horn;
  default: 0.05
- **`lower_cutoff`** — Lower -3 dB bandpass frequency in Hz, > 0; default: 80
- **`upper_cutoff`** — Upper -3 dB bandpass frequency in Hz, larger than `lower_cutoff`;
  default: 12000
- **`lower_rolloff_slope_dB`** — Low-frequency rolloff in dB/octave, > 0 (12 dB/oct = 2nd-order Butterworth);
  default: 12
- **`upper_rolloff_slope_dB`** — High-frequency rolloff in dB/octave, > 0; default: 12
- **`sensitivity_dB`** — On-axis sensitivity in dB SPL at 1W/1m; 85 dB gives unity amplitude; default: 85
- **`radiation_type`** — Enclosure radiation model: `monopole`, `hemisphere`, `dipole`, or `cardioid`;
  default: `hemisphere`
- **`hor_coverage_deg`** — Horn horizontal coverage angle in degrees; NaN or <= 0: default 90; horn only;
  default: NaN
- **`ver_coverage_deg`** — Horn vertical coverage angle in degrees; NaN or <= 0: default 60; horn only;
  default: NaN
- **`horn_control_freq`** — Horn pattern control frequency in Hz; NaN or <= 0: default `c/(2pi·radius)`;
  horn only; default: NaN
- **`baffle_width`** — Baffle width in m, > 0; used by `hemisphere` model; default: 0.15
- **`baffle_height`** — Baffle height in m, > 0; used by `hemisphere` model; default: 0.25
- **`center_freq`** — Frequency sample points in Hz; auto-generated third-octave bands if None;
  `(n_freq,)` or None; default: None
- **`angular_resolution_deg`** — Azimuth and elevation grid resolution in degrees, 0.1 to 90; default: 5

## Outputs:
- **`speaker`** — Multi-frequency arrayant dict with the fields below; for `n_freq = 1` the pattern fields
  are 3D `(n_elevation, n_azimuth, n_elements)` and `center_freq` is a float:
  - `e_theta_re` — e-theta field component, real part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_theta_im` — e-theta field component, imaginary part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_phi_re` — e-phi field component, real part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_phi_im` — e-phi field component, imaginary part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `azimuth_grid` — Azimuth angles in rad, -π to π, strictly increasing; `(n_azimuth,)`
  - `elevation_grid` — Elevation angles in rad, -π/2 to π/2, strictly increasing; `(n_elevation,)`
  - `element_pos` — Element (x,y,z) positions in meters; `(3, n_elements)`
  - `coupling_re` — Coupling matrix, real part;
    `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)` if it varies across frequencies
  - `coupling_im` — Coupling matrix, imaginary part;
    `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)` if it varies across frequencies
  - `center_freq` — Frequency samples in Hz; `(n_freq,)`
  - `name` — Name of the array antenna object; str
MD!*/

py::dict generate_speaker(const std::string &driver_type,
                          double radius,
                          double lower_cutoff,
                          double upper_cutoff,
                          double lower_rolloff_slope_dB,
                          double upper_rolloff_slope_dB,
                          double sensitivity_dB,
                          const std::string &radiation_type,
                          double hor_coverage_deg,
                          double ver_coverage_deg,
                          double horn_control_freq,
                          double baffle_width,
                          double baffle_height,
                          py::handle center_freq,
                          double angular_resolution_deg)
{
    // Empty arma::vec when center_freq is None or empty
    const auto freq_vec = qd_python_numpy2arma_Col<double>(center_freq, true);

    auto speaker = quadriga_lib::generate_speaker<double>(
        driver_type, radius, lower_cutoff, upper_cutoff,
        lower_rolloff_slope_dB, upper_rolloff_slope_dB, sensitivity_dB,
        radiation_type, hor_coverage_deg, ver_coverage_deg, horn_control_freq,
        baffle_width, baffle_height, freq_vec, angular_resolution_deg);

    return qd_python_arrayant2dict_multi(speaker);
}
