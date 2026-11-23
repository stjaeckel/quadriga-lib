// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!MD
# calc_beamwidth
Calculate the beamwidth and pointing angles of antenna array elements in degrees

- Computes azimuth and elevation beamwidth at a given dB threshold (default 3 dB = FWHM)
- Also returns the azimuth and elevation pointing angles of the main beam
- Sub-grid resolution is achieved by bilinear interpolation of the field pattern
  (≈100x finer grid in each direction than the antenna sampling grid)
- The width is that of the main lobe, searched outwards from the maximum; side lobes above the threshold
  are not included
- Azimuth: cut at the elevation of the maximum, circular across ±180° when the grid covers the full
  circle; 360° when the whole cut is above the threshold (e.g. a beam pointing to a pole)
- Elevation: cut along the great circle through the maximum, so a lobe can extend across the poles;
  at most 180°
- Calculated per element, not per port; ignores the port weights
- Accepts both single-frequency (3D pattern fields) and multi-frequency (4D pattern fields)
  arrayant dicts; output dimensionality follows the input

## Usage:
```
beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg = \
    quadriga_lib.arrayant.calc_beamwidth( arrayant, *, i_element, threshold_dB )
```

## Inputs:
- **`arrayant`** — Arrayant dict; single-frequency or multi-frequency (4th pattern dim is frequency);
  only `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid` are used;
  see [[generate_antenna]] for the field layout
- **`i_element`** — Element indices, 0-based, duplicates allowed; empty: all elements; `(n_out,)`; default: None (all elements)
- **`threshold_dB`** — Threshold in dB, positive and finite (3 = FWHM); default: 3.0

## Outputs:
- **`beamwidth_az_deg`** — Azimuth beamwidth in degrees; `(n_out,)` for single-frequency input,
  `(n_out, n_freq)` for multi-frequency input; `n_out = n_elements` when `i_element` is None
- **`beamwidth_el_deg`** — Elevation beamwidth in degrees; same shape as `beamwidth_az_deg`
- **`az_point_ang_deg`** — Azimuth pointing angle of the main beam in degrees; same shape as `beamwidth_az_deg`
- **`el_point_ang_deg`** — Elevation pointing angle of the main beam in degrees; same shape as `beamwidth_az_deg`

## See also:
- [[apply_port_weights]] (apply the port weights before calculating beamwidth)
- [[calc_directivity]] (directivity in dBi of antenna array elements)
MD!*/

py::tuple calc_beamwidth(const py::dict &arrayant,
                         py::handle i_element,
                         double threshold_dB)
{
    // Pattern arrays as views; 3D single-frequency and 4D multi-frequency arrays; validated by the C++ function
    const auto ant_multi = qd_python_dict2arrayant_multi(arrayant, true, false, false, false);
    const arma::uvec i_el = qd_python_numpy2arma_Col<arma::uword>(i_element, true, false, "i_element");

    if (ant_multi.size() == 1) // Single frequency: 1D outputs (n_out,)
    {
        arma::vec bw_az, bw_el, az_pt, el_pt;
        quadriga_lib::calc_beamwidth(ant_multi[0], i_el, threshold_dB, &bw_az, &bw_el, &az_pt, &el_pt);
        return py::make_tuple(qd_python_copy2numpy(&bw_az), qd_python_copy2numpy(&bw_el),
                              qd_python_copy2numpy(&az_pt), qd_python_copy2numpy(&el_pt));
    }

    // Multi-frequency: 2D outputs (n_out, n_freq)
    arma::mat bw_az, bw_el, az_pt, el_pt;
    quadriga_lib::calc_beamwidth(ant_multi, i_el, threshold_dB, &bw_az, &bw_el, &az_pt, &el_pt);
    return py::make_tuple(qd_python_copy2numpy(&bw_az), qd_python_copy2numpy(&bw_el),
                          qd_python_copy2numpy(&az_pt), qd_python_copy2numpy(&el_pt));
}

/*!MD
# calc_directivity
Calculates the directivity in dBi of antenna array elements

- Directivity = 10·log10(peak radiation intensity / mean over 4π); isotropic radiator = 0 dBi
- Calculated per element, not per port; ignores the port weights
- Independent of the scale of the pattern; an element whose pattern is zero everywhere gives 0 dBi
- Accepts both single-frequency (3D pattern fields) and multi-frequency (4D pattern fields)
  arrayant dicts; output dimensionality follows the input

## Usage:
```
directivity_dBi = quadriga_lib.arrayant.calc_directivity( arrayant, *, i_element )
```

## Inputs:
- **`arrayant`** — Arrayant dict; single-frequency or multi-frequency (4th pattern dim is frequency);
  only `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im`, `azimuth_grid`, `elevation_grid` are used;
  see [[generate_antenna]] for the field layout
- **`i_element`** — Element indices, 0-based, duplicates allowed; empty: all elements; `(n_out,)`; default: None (all elements)

## Output:
- **`directivity_dBi`** — Directivity in dBi; `(n_out,)` for a single-frequency input,
  `(n_out, n_freq)` for a multi-frequency input; `n_out = n_elements` when `i_element` is None

## See also:
- [[apply_port_weights]] (apply the port weights before calculating directivity)
- [[calc_beamwidth]] (calculates the beam width of antenna arrays)
MD!*/

py::array_t<double> calc_directivity(const py::dict &arrayant,
                                     py::handle i_element)
{
    // Pattern arrays as views; 3D single-frequency and 4D multi-frequency arrays; validated by the C++ function
    const auto ant_multi = qd_python_dict2arrayant_multi(arrayant, true, false, false, false);
    const arma::uvec i_el = qd_python_numpy2arma_Col<arma::uword>(i_element, true, false, "i_element");

    if (ant_multi.size() == 1) // Single frequency: 1D output (n_out,)
    {
        const arma::vec directivity_dBi = quadriga_lib::calc_directivity(ant_multi[0], i_el);
        return qd_python_copy2numpy(&directivity_dBi);
    }

    // Multi-frequency: 2D output (n_out, n_freq)
    const arma::mat directivity_dBi = quadriga_lib::calc_directivity(ant_multi, i_el);
    return qd_python_copy2numpy(&directivity_dBi);
}

/*!MD
# apply_port_weights
Apply the port weights of an antenna array: one pattern per port

- Evaluates the element patterns on the output grid, applies the phase shift of each element position (which
  moves every pattern to the array origin) and sums the elements with the port weights
- The result has one element per port, identity port weights and all element positions at the origin
- Speeds up MIMO channel computation; useful for beamforming in 5G systems and network planning
- Accepts a frequency-dependent antenna (4D pattern fields); the pattern is then combined per frequency
- `freq_new` may recompute (interpolate) the combined pattern at one or more requested frequencies
- The element positions add a phase that depends on frequency and `propagation_speed`; the default speed of light
  applies to radio waves, use about 343 m/s for acoustic arrays (e.g. from [[generate_speaker]])

## Usage:
```
# Single frequency
arrayant_out = quadriga_lib.arrayant.apply_port_weights( arrayant )
arrayant_out = quadriga_lib.arrayant.apply_port_weights( arrayant, *, freq_new, azimuth_grid, \
    elevation_grid )

# Multiple frequencies (freq_new as a 1D array, or a frequency-dependent input antenna)
arrayant_out = quadriga_lib.arrayant.apply_port_weights( arrayant, *, freq_new, azimuth_grid, \
    elevation_grid )

# Acoustic array (speed of sound)
arrayant_out = quadriga_lib.arrayant.apply_port_weights( arrayant, propagation_speed=343.0 )
```

## Inputs:
- **`arrayant`** — Dict with the antenna array data; keys as documented in [[generate_antenna]]; 4D pattern fields
  require `center_freq` with one value per frequency entry `(n_freq,)`
- **`freq_new`** — Alternative center frequency in Hz; scalar or 1D array; if given, the pattern is
  recomputed at each value; if `None`, each input entry's `center_freq` is used; a scalar `<= 0`
  is treated as not given; default: `None`
- **`azimuth_grid`** — Alternative output azimuth grid in rad; -pi to pi; strictly increasing; `(n_azimuth_out,)`;
  if `None`, the input grid is used; default: `None`
- **`elevation_grid`** — Alternative output elevation grid in rad; -pi/2 to pi/2; strictly increasing;
  `(n_elevation_out,)`; if `None`, the input grid is used; default: `None`
- **`propagation_speed`** — Wave propagation speed in m/s; positive and finite; use about 343 for
  acoustics; default: `299792458.0`

## Output:
- **`arrayant_out`** — Dict with the combined antenna array data; keys as in [[generate_antenna]]

## See also:
- [[generate_antenna]] (for field layout in the arrayant struct)
- [[rotate_pattern]] (for changing the orientation of elements before combining)
- [[calc_beamwidth]] (calculates the beam width of antenna arrays)
- [[calc_directivity]] (directivity in dBi of antenna array elements)
MD!*/

py::dict apply_port_weights(const py::dict &arrayant,
                            py::handle freq_new,
                            py::handle azimuth_grid,
                            py::handle elevation_grid,
                            double propagation_speed)
{
    // Optional output grids (empty Col = keep the input grid)
    const auto azimuth_grid_a = qd_python_numpy2arma_Col<double>(azimuth_grid, true);
    const auto elevation_grid_a = qd_python_numpy2arma_Col<double>(elevation_grid, true);

    // A scalar freq_new <= 0 is no override
    arma::vec freq_new_a = qd_python_numpy2arma_Col<double>(freq_new, false); // Copy: may be reset below
    if (freq_new_a.n_elem == 1 && freq_new_a[0] <= 0.0)
        freq_new_a.reset();

    // Length 1 for a single-frequency dict
    auto ant = qd_python_dict2arrayant_multi(arrayant, true, false, true);

    // Multi-frequency branch: frequency-dependent input, or a freq_new vector with > 1 entry
    if (ant.size() > 1 || freq_new_a.n_elem > 1)
    {
        auto out = quadriga_lib::apply_port_weights(ant, azimuth_grid_a, elevation_grid_a, freq_new_a, propagation_speed);
        return qd_python_arrayant2dict_multi(out);
    }

    if (freq_new_a.n_elem == 1)
        ant[0].center_freq = freq_new_a[0];

    auto arrayant_out = quadriga_lib::apply_port_weights(ant[0], azimuth_grid_a, elevation_grid_a, propagation_speed);
    return qd_python_arrayant2dict(arrayant_out);
}

/*!MD
# join_arrays
Concatenate two antenna arrays into a single one

- Appends all elements of `arrayant2` onto `arrayant1` along the element dimension; the `element_pos`
  matrices are joined horizontally
- Both inputs must share identical azimuth and elevation sampling grids
- Port weights are assembled block-diagonally: `arrayant1`'s elements connect only to its own ports,
  `arrayant2`'s elements only to its own ports
- `center_freq` and `name` are taken from `arrayant1`
- Frequency-dependent input (4D pattern fields): both inputs must have the same number of
  frequency entries with matching `center_freq` at each index; concatenation is done per entry
- Output format matches the input: single-frequency dict for 3D input, frequency-dependent dict
  for 4D input

## Usage:
```
arrayant_out = quadriga_lib.arrayant.join_arrays( arrayant1, arrayant2 )
```

## Inputs:
- **`arrayant1`** — Dict with the first antenna array; keys as in [[generate_antenna]]; pattern fields
  may be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`arrayant2`** — Dict with the second antenna array; must use the same azimuth and elevation
  grids as `arrayant1`, and for 4D input the same number of frequency entries with matching `center_freq`

## Output:
- **`arrayant_out`** — Dict with the concatenated antenna array; same keys and layout as the
  inputs; single-frequency dict for 3D input, frequency-dependent dict for 4D input
MD!*/

py::dict join_arrays(const py::dict &arrayant1,
                     const py::dict &arrayant2)
{
    auto ant1 = qd_python_dict2arrayant_multi(arrayant1, true, false, true, false);
    auto ant2 = qd_python_dict2arrayant_multi(arrayant2, true, false, true, false);

    if (ant1.empty() || ant2.empty())
        throw std::invalid_argument("Inputs 'arrayant1' and 'arrayant2' cannot be empty.");

    if (ant1.size() != ant2.size())
        throw std::invalid_argument("Inputs 'arrayant1' and 'arrayant2' must have the same number of frequency entries.");

    // Frequency-dependent: concatenate per entry (requires matching center_freq at each index)
    if (ant1.size() > 1)
    {
        auto out = quadriga_lib::join_arrays<double>(ant1, ant2);
        return qd_python_arrayant2dict_multi(out);
    }

    auto out = quadriga_lib::join_arrays(ant1[0], ant2[0]);
    return qd_python_arrayant2dict(out);
}

/*!MD
# copy_element
Create copies of antenna array elements

- Copies a source element to one or more destination slots within an arrayant
- The antenna is resized when a destination index exceeds the current number of elements
- Port weights of newly added elements are set to identity; existing port weights are kept
- Works on single-frequency dicts (3D pattern fields) and frequency-dependent dicts (4D pattern
  fields); the same copy is applied to every frequency and a matching dict is returned
- If `i_source` has one entry, that element is copied to every index in `i_target`
- If `i_source` has several entries, `i_target` must have the same length and copies
  are performed pairwise as `i_source[i]` to `i_target[i]`

## Usage:
```
# Copy element 0 to position 1
arrayant_out = quadriga_lib.arrayant.copy_element( arrayant, 0, 1 )

# Copy element 0 to several positions
arrayant_out = quadriga_lib.arrayant.copy_element( arrayant, 0, [2, 3] )

# Pairwise copy — i_source and i_target must have equal length
arrayant_out = quadriga_lib.arrayant.copy_element( arrayant, [0, 1], [2, 3] )
```

## Inputs:
- **`arrayant`** — Dict with the antenna array data; keys as in [[generate_antenna]]; pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`i_source`** — Index of the source element(s); scalar int or 1D list or array of int; `(n_copy,)`
- **`i_target`** — Index of the destination element(s); scalar int or 1D list or array of int; `(n_copy,)`;
  must have the same length as `i_source` unless `i_source` is a single index

## Output:
- **`arrayant_out`** — Dict with the modified antenna array data; same keys and layout as
  `arrayant`; single-frequency dict for 3D input, frequency-dependent dict for 4D input
MD!*/

py::dict copy_element(const py::dict &arrayant,
                      const py::handle &i_source,
                      const py::handle &i_target)
{
    // Index vectors (0-based)
    const auto source = qd_python_numpy2arma_Col<arma::uword>(i_source, true, false, "i_source");
    const auto dest = qd_python_numpy2arma_Col<arma::uword>(i_target, true, false, "i_target");

    if (source.n_elem == 0 || dest.n_elem == 0)
        throw std::invalid_argument("Inputs 'i_source' and 'i_target' cannot be empty.");

    if (source.n_elem > 1 && source.n_elem != dest.n_elem)
        throw std::invalid_argument("Inputs 'i_source' and 'i_target' must have the same length when 'i_source' has more than one entry.");

    auto ant = qd_python_dict2arrayant_multi(arrayant, false, false, true, false);

    // Apply the copy to every frequency entry
    if (source.n_elem == 1)
        quadriga_lib::copy_element(ant, source.at(0), dest);
    else
        for (arma::uword i = 0; i < source.n_elem; ++i)
            quadriga_lib::copy_element(ant, source.at(i), arma::uvec{dest.at(i)});

    return qd_python_arrayant2dict_multi(ant);
}

/*!MD
# rotate_pattern
Rotate antenna radiation patterns using Euler rotations

- Rotates the pattern and/or polarization by the orientation (bank, tilt, heading); the rotated pattern equals what
  [[interpolate_pattern]] returns for the unrotated pattern with the same `orientation`
- A positive tilt turns the main beam down (downtilt), as in 3GPP TR 38.901
- Element positions are rotated with the pattern (usage 0, 1, 3 and 4); port weights, `center_freq` and `name` are kept
- Single-frequency input (3D pattern fields): `usage` 0 and 1 adjust the sampling grid for non-uniformly sampled
  antennas; this needs all elements, so a subset in `i_element` raises an error when the grid must be adjusted
- Frequency-dependent input (4D pattern fields): the rotation is applied to every entry and the grid is never adjusted,
  since all entries must share one grid
- For scalar acoustic fields (pressure stored in `e_theta_re` only) use `usage = 1` to avoid spurious polarization effects

## Usage:
```
arrayant_out = quadriga_lib.arrayant.rotate_pattern( arrayant, orientation, *, usage, i_element )
```

## Inputs:
- **`arrayant`** — Dict with the antenna array data; keys as in [[generate_antenna]]; pattern fields may
  be 3D `(n_elevation, n_azimuth, n_elements)` or 4D `(n_elevation, n_azimuth, n_elements, n_freq)`
- **`orientation`** — Euler angles (bank, tilt, heading) in rad; 3 elements
- **`usage`** — Rotation mode; default: 0<br><br>
   | Mode | Pattern | Polarization | Grid adj. |
   | :--: | :-----: | :----------: | :-------: |
   | 0    | Yes     | Yes          | Yes       |
   | 1    | Yes     | No           | Yes       |
   | 2    | No      | Yes          | No        |
   | 3    | Yes     | Yes          | No        |
   | 4    | Yes     | No           | No        |

   for 4D input the grid is never adjusted, so `0`/`3` and `1`/`4` are equivalent
- **`i_element`** — Element indices to rotate, 0-based, no duplicates; `None` or empty rotates all elements;
  default: `None`

## Output:
- **`arrayant_out`** — Dict with the rotated antenna array data; same keys and layout as
  `arrayant`; single-frequency dict for 3D input, frequency-dependent dict for 4D input
MD!*/

py::dict rotate_pattern(const py::dict &arrayant,
                        const py::handle &orientation,
                        unsigned usage,
                        const py::handle &i_element)
{
    const arma::vec ori = qd_python_numpy2arma_Col<double>(orientation, true); // 3 elements, checked by C++

    // Element indices (0-based); empty = rotate all elements
    const arma::uvec i_el = qd_python_numpy2arma_Col<arma::uword>(i_element, true, false, "i_element");

    // A copy: the caller's arrays are not changed
    auto ant_vec = qd_python_dict2arrayant_multi(arrayant, false, false, false, false);

    if (ant_vec.size() > 1) // Frequency-dependent: rotate every entry
    {
        const unsigned usage_multi = (usage == 3) ? 0 : ((usage == 4) ? 1 : usage);
        quadriga_lib::rotate_pattern(ant_vec, ori, usage_multi, i_el);
    }
    else // Single-frequency: usage 0/1 adjust the sampling grid
        quadriga_lib::rotate_pattern(ant_vec[0], ori, usage, i_el);

    return qd_python_arrayant2dict_multi(ant_vec);
}
