// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# generate_antenna
Generate an antenna array of a given type

- Tuples (`element`, `panel`, `spacing`, `pol`, `ports`, `steer`): entries left off at the end or set to NaN (0 in `panel`, `pol`, `ports`) take their default, e.g. `panel=[4, 8]` is one panel of 4 rows and 8 columns.
- Panel arrays: elements are ordered polarization fastest, then row `m` (upwards along z), column `n` (along +y), panel row `mg`, panel column `ng`, centred on the origin; ports in the same order with `m_p`, `n_p`. The port weights `[n_elements, n_ports]` hold the weights; `apply_weights` combines each port into one element at its sub-array centre.
- Sub-array weights (`ula`, `3GPP`; TR 38.901 eq. 7.3-1, along y by analogy): a port feeds `K = M / M_P` × `L = N / N_P` elements; element `(k, l)` gets `exp(j 2π ((k - (K-1)/2) d_V sin(tilt) - (l - (L-1)/2) d_H sin(pan))) / sqrt(K L)`; a positive tilt points the beam down, a positive pan towards +y.
- Polarization (TR 38.901 clause 7.3.2): `pol` 2 has the slants ζ = 0° and 90°, `pol` 3 ζ = +45° and -45° (element 0 first). Model-1 turns the element by ζ about its boresight (eq. 7.3-3); Model-2 sets F_θ = sqrt(A') cos ζ and F_φ = sqrt(A') sin ζ with A' = |F_θ|² + |F_φ|² of the element (eq. 7.3-4, 7.3-5).
- `multibeam`: maximum-ratio-transmission weights for the directions in `beams` (exact for one beam, approximate for several; only the ratios of the beam weights matter); one port per polarization, with `separate_beams` one per polarization and beam.
- Handheld UT (`3GPP-UT`, TR 38.901 clause 7.3): `P` elements at each entry of `locations` (Fig. 7.3-2), polarization fastest; each element is rolled about its boresight by 90° (45° for two field patterns: `pol` 2 or a two-element `pattern`) and turned to the bearing `atan2(y, x)` of its location (Tables A.6-1, A.6-2); a `pattern` on an irregular grid is first resampled to a regular one. `pol` 1 and 2 with their default model are the TR model (eq. 7.3-6 to 7.3-8), other values are extensions. For an element symmetric under a 180° roll (every `element` tuple), the other z'' direction of Fig. 7.3-7 is the same antenna with a negated port weight.
- CPE (`3GPP-CPE`): `P` elements at each entry of `locations` (Fig. 7.3-6 seen from the front, in the y-z plane), facing +x. TR 38.901 defines no CPE element; the default is the UE element of TR 38.803 Table 5.2.3.3-1.
- An input that the type does not take raises `ValueError`.

## Usage:
```
# omni, xpol, dipole, half-wave-dipole
ant = quadriga_lib.arrayant.generate_antenna( type, res, *, center_freq )

# Custom-shape element
ant = quadriga_lib.arrayant.generate_antenna( 'custom', res, *, center_freq, element )

# Uniform linear array (omni element) and TR 38.901 panel array
ant = quadriga_lib.arrayant.generate_antenna( 'ula', res, *, center_freq, panel, spacing, pol, ports, steer, \
    pattern, apply_weights )
ant = quadriga_lib.arrayant.generate_antenna( '3GPP', res, *, center_freq, element, panel, spacing, pol, ports, \
    steer, pattern, apply_weights )

# Handheld UT and CPE
ant = quadriga_lib.arrayant.generate_antenna( '3GPP-UT', res, *, center_freq, element, pol, device_size, \
    locations, pattern )

# Multi-beam panel array
ant = quadriga_lib.arrayant.generate_antenna( 'multibeam', res, *, center_freq, element, panel, spacing, pol, \
    beams, separate_beams, apply_weights )
```

## Inputs:
- **`type`** — Antenna type; str:<br><br>
   | `type`                       | Antenna                                                                    | Inputs besides `res`, `center_freq`                                                |
   | ---------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
   | `omni`                       | Isotropic element, vertical polarization                                   | none                                                                               |
   | `xpol`                       | Two isotropic elements, vertical and horizontal polarization               | none                                                                               |
   | `dipole` (or `short-dipole`) | Short dipole, vertical polarization                                        | none                                                                               |
   | `half-wave-dipole`           | Half-wave dipole, vertical polarization                                    | none                                                                               |
   | `custom`                     | One element with given 3 dB beamwidths, peak gain equal to its directivity | `element`                                                                          |
   | `ula`                        | Uniform linear array of omni elements                                      | `panel`, `spacing`, `pol`, `ports`, `steer`, `pattern`, `apply_weights`            |
   | `3GPP` (or `3gpp`)           | Panel array of 3GPP TR 38.901 clause 7.3                                   | `element`, `panel`, `spacing`, `pol`, `ports`, `steer`, `pattern`, `apply_weights` |
   | `multibeam`                  | Panel array with beams towards given directions                            | `element`, `panel`, `spacing`, `pol`, `beams`, `separate_beams`, `apply_weights`   |
   | `3GPP-UT` (or `3gpp-ut`)     | Handheld UT of 3GPP TR 38.901 clause 7.3                                   | `element`, `pol`, `pattern`, `device_size`, `locations`                            |
   | `3GPP-CPE` (or `3gpp-cpe`)   | CPE of 3GPP TR 38.901 clause 7.3                                           | `element`, `pol`, `pattern`, `device_size`, `locations`                            |
- **`res`** — Antenna pattern sampling grid resolution in degrees; 0.1 to 90; NaN or <= 0: default; default: 10 for `omni` and `xpol`, 1 otherwise
- **`center_freq`** — Center frequency in Hz; positive and finite; NaN: default; default: 299792458 (wavelength 1 m)
- **`element`** — Element parameters per type; default: `None` (the defaults below):
  - `custom` `[az_3dB, el_3dB, floor_dB]`: 3 dB beamwidths in degrees (> 0, Inf: no taper); floor of the power pattern as a positive attenuation in dB below the peak (10: 10 dB below the peak; Inf: no floor); default `[90, 90, Inf]`
  - `multibeam`: the same, default `[120, 120, Inf]`
  - `3GPP` `[phi_3dB, theta_3dB, SLA_V, A_max, G_max]` of TR 38.901 Table 7.3-1: beamwidths in degrees (> 0, Inf: no taper); SLA_V and A_max as positive attenuations in dB below the peak, as in the TR (>= 0, Inf: no limit); G_max in dBi; default `[65, 65, 30, 30, 8]`; entries left off at the end or set to NaN take their default, where `theta_3dB` defaults to `phi_3dB` and `A_max` to `SLA_V` (e.g. `[90]` gives `[90, 90, 30, 30, 8]`); `G_max` left off is 8 dBi, NaN sets the peak gain to the directivity
  - `3GPP-UT`: the same tuple, default `[125, 125, 22.5, 22.5, 5.3]` (Table 7.3-2); `3GPP-CPE`: default `[90, 90, 25, 25, 5]`; `[Inf, Inf, NaN, NaN, 0]` is isotropic
- **`panel`** — `[M, N, Mg, Ng]`: elements per column, columns per panel, panels per column, panels per row; int or `(n,)` of int; at most 1e7 elements per polarization; default: `None` (1; `ula`: `panel=8` is a vertical column, `[1, 8]` a horizontal row)
- **`spacing`** — `[d_V, d_H, d_gV, d_gH]` in wavelengths: vertical and horizontal element spacing, vertical and horizontal panel spacing (centre to centre); >= 0; default: `None` (`[0.5, d_V, M d_V, N d_H]`)
- **`pol`** — `[polarization, model]`. Polarization: 1 = one element (ζ = 0°, vertical), 2 = two elements (ζ = 0° and 90°), 3 = two elements (ζ = +45° and -45°). Model of TR 38.901 clause 7.3.2: 1 = the element pattern is turned by ζ about its boresight (Model-1, eq. 7.3-3), 2 = the power pattern is kept and the field is split by ζ (Model-2, eq. 7.3-4, 7.3-5). Int or `(n,)` of int; default: `None` (`[1, 1]`; for `3GPP-UT` with polarization 2 or 3 the default model is 2, which gives the reference fields of the handheld model, eq. 7.3-6)
- **`ports`** — `[M_P, N_P]`: ports per column and per row of a panel and polarization; int or `(n,)` of int; must divide `M` and `N`; default: `None` (`[M, N]`, one port per element)
- **`steer`** — `[tilt, pan]` in degrees, -90 to 90: electrical downtilt (positive down) and pan (positive towards +y); `(2,)` (a 1D input is one row) for all panels or `(Mg Ng, 2)` with one row per panel; one column: pan 0; a non-zero tilt needs `M_P < M`, a non-zero pan `N_P < N`; default: `None` (0)
- **`beams`** — `(2, n_beams)` or `(3, n_beams)`: azimuth in degrees; elevation in degrees; weight (linear; 2 rows: 1); a 1D input is one beam; default: `None` (`[[0], [0], [1]]`)
- **`device_size`** — `[X, Y, Z]` in m (depth, width, height); `(n,)`; NaN or left off: the default; `3GPP-UT` `[0.15, 0.07, 0]` with X, Y > 0 and Z = 0, `3GPP-CPE` `[0, 0.2, 0.2]` with X = 0 and Y, Z >= 0; default: `None`
- **`locations`** — TR location labels (`3GPP-UT` 1 to 8, `3GPP-CPE` 1 to 9), not 0-based indices; any order, repeats allowed; `(n,)` of int; default: `None` (all)
- **`pattern`** — Element dict (keys as the output) to use instead of the default element: its field patterns (`e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` with their grids) are copied to every element position; its `element_pos`, port weights and `center_freq` are not used, the generator sets them; a one-element pattern gets the polarizations of `pol`, one with several elements is a polarization set used as given and needs `pol` [1, 1]; default: `None` (the element of the type)
- **`separate_beams`** — If `True`, one port per beam (weights ignored); default: `False`
- **`apply_weights`** — If `True`, combine each port into one pattern at its sub-array centre; default: `False`

## Outputs:
- **`ant`** — Dict with keys:<br><br>
   | Key               | Description                                               | Shape                                  |
   | ----------------- | --------------------------------------------------------- | -------------------------------------- |
   | `e_theta_re`      | e-theta field component, real part                        | `(n_elevation, n_azimuth, n_elements)` |
   | `e_theta_im`      | e-theta field component, imaginary part                   | `(n_elevation, n_azimuth, n_elements)` |
   | `e_phi_re`        | e-phi field component, real part                          | `(n_elevation, n_azimuth, n_elements)` |
   | `e_phi_im`        | e-phi field component, imaginary part                     | `(n_elevation, n_azimuth, n_elements)` |
   | `azimuth_grid`    | Azimuth angles in rad, -π to π, strictly increasing       | `(n_azimuth,)`                         |
   | `elevation_grid`  | Elevation angles in rad, -π/2 to π/2, strictly increasing | `(n_elevation,)`                       |
   | `element_pos`     | Element (x,y,z) positions in meters                       | `(3, n_elements)`                      |
   | `port_weights_re` | Port weights, real part                                   | `(n_elements, n_ports)`                |
   | `port_weights_im` | Port weights, imaginary part                              | `(n_elements, n_ports)`                |
   | `center_freq`     | Center frequency in Hz; inputs accept `(1,)`              | scalar                                 |
   | `name`            | Name of the antenna array object                          | str                                    |
MD!*/

py::dict generate_antenna(const std::string &type,
                          double res,
                          double center_freq,
                          const py::handle &element,
                          const py::handle &panel,
                          const py::handle &spacing,
                          const py::handle &pol,
                          const py::handle &ports,
                          const py::handle &steer,
                          const py::handle &beams,
                          const py::handle &device_size,
                          const py::handle &locations,
                          const py::handle &pattern,
                          const py::handle &separate_beams,
                          const py::handle &apply_weights)
{
    // None is an empty container (not given); the C++ function fills the defaults per type
    const auto panel_a = qd_python_numpy2arma_Col<arma::uword>(panel, true, false, "panel");
    const auto spacing_a = qd_python_numpy2arma_Col<double>(spacing, true);
    const auto ports_a = qd_python_numpy2arma_Col<arma::uword>(ports, true, false, "ports");
    const auto pol_a = qd_python_numpy2arma_Col<arma::uword>(pol, true, false, "pol");
    const auto element_a = qd_python_numpy2arma_Col<double>(element, true);
    const auto beams_a = qd_python_numpy2arma_Mat<double>(beams, true);
    const auto device_size_a = qd_python_numpy2arma_Col<double>(device_size, true);
    const auto locations_a = qd_python_numpy2arma_Col<arma::uword>(locations, true, false, "locations");

    arma::mat steer_a = qd_python_numpy2arma_Mat<double>(steer, false); // Copy: a 1D input is turned into one row
    if (!steer.is_none())
    {
        const py::array steer_arr = py::array::ensure(steer);
        if (steer_arr && steer_arr.ndim() < 2)
            arma::inplace_trans(steer_a);
    }

    quadriga_lib::arrayant<double> pattern_a;
    if (!pattern.is_none())
    {
        if (!py::isinstance<py::dict>(pattern))
            throw py::type_error("Input 'pattern' must be a dict or None.");
        const auto pattern_d = py::reinterpret_borrow<py::dict>(pattern);
        if (!pattern_d.empty()) // An empty dict is no pattern, as [] in MATLAB
            pattern_a = qd_python_dict2arrayant(pattern_d, true); // Validated by the C++ function
    }

    const bool separate_beams_b = qd_python_flag(separate_beams, false, "separate_beams");
    const bool apply_weights_b = qd_python_flag(apply_weights, false, "apply_weights");

    const auto ant = quadriga_lib::generate_antenna<double>(type, res, center_freq, element_a, panel_a, spacing_a, pol_a, ports_a,
                                                             steer_a, beams_a, device_size_a, locations_a, pattern_a,
                                                             separate_beams_b, apply_weights_b);
    return qd_python_arrayant2dict(ant);
}

/*!MD
# generate_speaker
Generate a parametric frequency-dependent loudspeaker directivity model

- Returns a multi-frequency arrayant dict; pattern fields are 4D arrays where the 4th dim is frequency; for a
  single frequency the dict is a single-frequency arrayant dict (3D pattern fields, scalar `center_freq`)
- Directivity is stored as real-valued data in `e_theta_re`; dipole rear hemisphere is encoded with negative sign for 180° phase inversion
- Multi-driver systems (e.g. two-way) are built by calling this per driver and combining via [[join_arrays]];
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

## Output:
- **`speaker`** — Multi-frequency arrayant dict with the fields below; for `n_freq = 1` the pattern fields
  are 3D `(n_elevation, n_azimuth, n_elements)` and `center_freq` is a float:
  - `e_theta_re` — e-theta field component, real part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_theta_im` — e-theta field component, imaginary part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_phi_re` — e-phi field component, real part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `e_phi_im` — e-phi field component, imaginary part; `(n_elevation, n_azimuth, n_elements, n_freq)`
  - `azimuth_grid` — Azimuth angles in rad, -π to π, strictly increasing; `(n_azimuth,)`
  - `elevation_grid` — Elevation angles in rad, -π/2 to π/2, strictly increasing; `(n_elevation,)`
  - `element_pos` — Element (x,y,z) positions in meters; `(3, n_elements)`
  - `port_weights_re` — Port weights, real part;
    `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)` if it varies across frequencies
  - `port_weights_im` — Port weights, imaginary part;
    `(n_elements, n_ports)` or `(n_elements, n_ports, n_freq)` if it varies across frequencies
  - `center_freq` — Frequency samples in Hz; `(n_freq,)`
  - `name` — Name of the antenna array object; str
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
