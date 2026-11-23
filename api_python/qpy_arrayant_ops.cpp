// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python_types.hpp"

/*!SECTION
Antennas, transducers and patterns
SECTION!*/

/*!SECTION_DESC
The functions of this section create, inspect, transform and store antenna and acoustic arrays. They share one data
structure, the arrayant dict (keys in [[generate_antenna]]): complex polarimetric patterns of the elements on an azimuth
and elevation grid, the element positions and the port weights. The channel functions take it as transmit and receive
antenna.<br><br>

**Terms** (full list in the [Glossary](cpp_api.html#glossary)):

- **Element**: one radiator or sensor (antenna element, loudspeaker driver, microphone capsule) with a pattern, a
  position and one row of the port weights
- **Grid**: azimuth $\phi$ in $[-\pi, \pi]$, measured from +x toward +y, and elevation $\theta$ in $[-\pi/2, \pi/2]$,
  measured from the x-y plane (TR 38.901 uses the zenith angle $90° - \theta$), both in rad and strictly increasing.
  Boresight is +x
- **Pattern**: the direction-dependent far field of an element: $F_\theta$ (`e_theta_re/im`) along
  $\hat\theta = (\sin\theta\cos\phi, \sin\theta\sin\phi, -\cos\theta)$, the zenith-angle unit vector of TR 38.901, which
  points toward decreasing elevation, and $F_\phi$ (`e_phi_re/im`) along $\hat\phi = (-\sin\phi, \cos\phi, 0)$. For an
  antenna, $\lvert F_\theta \rvert^2 + \lvert F_\phi \rvert^2$ is the linear gain over an isotropic radiator
- **Port**: one signal of the array, one column of the port weights
- **Port weights**: the fixed complex feed network $w_{i,p}$ from element $i$ to port $p$, `(n_elements, n_ports)`. It
  is not the mutual coupling between elements. Empty means identity (one port per element)
- **Element position**: local Cartesian coordinates in m, `(3, n_elements)`; empty means all at the origin
- **Multi-frequency array**: a dict whose pattern arrays have the frequency as fourth axis,
  `(n_elevation, n_azimuth, n_elements, n_freq)`, with `center_freq` `(n_freq,)`, one grid and one set of element
  positions, and port weights `(n_elements, n_ports)` or, if they differ between frequencies,
  `(n_elements, n_ports, n_freq)`. [[interpolate_pattern]] and [[apply_port_weights]] interpolate over frequency between
  the entries

The field of port $p$ in a direction is the weighted sum of the element fields, each with the phase of its position,

$$F_p = \sum_i w_{i,p} \, F_i \, e^{-j 2\pi f d_i / c}$$

where $d_i = -\hat{r} \cdot x_i$ is the distance of element $i$ at position $x_i$ from the wavefront plane through the
origin, negative for elements closer to the source, $\hat{r}$ the unit vector toward the source, $f$ the frequency and
$c$ the propagation speed.<br><br>

**Functions by task:**

| Task | Functions |
| ---- | --------- |
| Create | [[generate_antenna]], [[generate_speaker]], [[qdant_read]] |
| Evaluate | [[interpolate_pattern]], [[calc_directivity]], [[calc_beamwidth]] |
| Transform | [[apply_port_weights]], [[rotate_pattern]] |
| Edit and combine | [[join_arrays]], [[copy_element]] |
| Store, show | [[qdant_write]], [[export_pattern_obj]] |

Arrays are stored in the [QDANT file format](formats.html#qdant).
SECTION_DESC!*/

/*!MD
# calc_beamwidth
Calculate the beamwidth and pointing angles of the elements of an antenna array in degrees

- Measures the width of the main lobe of each element in azimuth and elevation at a threshold below its peak, and the
  direction of the peak
- Calculated per element, not per port: the port weights are ignored

## Usage:
```
beamwidth_az_deg, beamwidth_el_deg, az_point_ang_deg, el_point_ang_deg = \
    quadriga_lib.arrayant.calc_beamwidth( arrayant, *, i_element, threshold_dB )
```

## Inputs:
- **`arrayant`** — Arrayant dict ([[generate_antenna]]); 4D pattern arrays hold a multi-frequency array; only the
  pattern arrays and the grids are needed
- **`i_element`** — Element indices, 0-based, duplicates allowed; `(n_out,)`; default: `None` (all elements)
- **`threshold_dB`** — Threshold below the peak in dB, positive and finite; 3 dB is the half-power beamwidth;
  loudspeakers state the coverage angle between the -6 dB points, use 6; default: 3.0

## Outputs:
- **`beamwidth_az_deg`** — Azimuth beamwidth in degrees; `(n_out,)`, 4D input `(n_out, n_freq)`
- **`beamwidth_el_deg`** — Elevation beamwidth in degrees; same shape
- **`az_point_ang_deg`** — Azimuth pointing angle of the main beam in degrees; same shape
- **`el_point_ang_deg`** — Elevation pointing angle of the main beam in degrees; same shape

## Notes:
- **Main lobe**: starting at the maximum, the search goes outward until the power drops below the threshold; side lobes
  above the threshold are not included
- **Resolution**: the cuts are interpolated as in [[interpolate_pattern]], about 100 times finer than the sampling grid
- **Azimuth cut**: at the elevation of the maximum, circular across ±180° when the grid covers the full circle; 360°
  when the whole cut is above the threshold (e.g. omni patterns or a beam pointing to a pole, where the cut is a point)
- **Elevation cut**: along the great circle through the maximum, so a lobe can extend across the poles (requires samples
  at ±90°); the result is at most 180°

## See also:
- [[calc_directivity]] (directivity in dBi of the elements)
- [[apply_port_weights]] (combines the elements into ports, to measure the beams of the ports)
- [[generate_antenna]] (creates elements with a given beamwidth)
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
Calculate the directivity in dBi of the elements of an antenna array

- Measures how strongly an element concentrates its power: $D = 10 \log_{10}(P_\mathrm{max} / \bar{P})$ with the peak
  radiation intensity $P_\mathrm{max}$ and its mean $\bar{P}$ over the sphere; an isotropic radiator has 0 dBi
- Calculated per element, not per port: the port weights are ignored

## Usage:
```
directivity_dBi = quadriga_lib.arrayant.calc_directivity( arrayant, *, i_element )
```

## Inputs:
- **`arrayant`** — Arrayant dict ([[generate_antenna]]); 4D pattern arrays hold a multi-frequency array; only the
  pattern arrays and the grids are needed
- **`i_element`** — Element indices, 0-based, duplicates allowed; `(n_out,)`; default: `None` (all elements)

## Output:
- **`directivity_dBi`** — Directivity in dBi; `(n_out,)`, 4D input `(n_out, n_freq)`

## Notes:
- **Scale**: the result does not depend on the scale of the pattern; an element whose pattern is zero everywhere gives 0
  dBi

## See also:
- [[calc_beamwidth]] (beamwidth and pointing angles of the elements)
- [[apply_port_weights]] (combines the elements into ports, to measure the directivity of the ports)
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

- Combines the elements of an array into one pattern per port, for beamforming and to speed up MIMO channel computation
- Each port pattern is the weighted sum of the element fields from the section overview, evaluated on the output grid
  and including the phase of each element position
- The result has one element per port, identity port weights and all element positions at the origin

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
- **`arrayant`** — Arrayant dict ([[generate_antenna]]); 4D pattern arrays hold a multi-frequency array
- **`freq_new`** — Output frequencies in Hz, positive and finite; a scalar <= 0 counts as not given; a 3D input has one
  pattern, so that only the phase of the element positions follows `freq_new`; scalar or `(n_freq_new,)`; default:
  `None` (the `center_freq` of the entries, one output per entry)
- **`azimuth_grid`** — Output azimuth grid in rad; `(n_azimuth_out,)`; default: `None` (the input grid)
- **`elevation_grid`** — Output elevation grid in rad; `(n_elevation_out,)`; default: `None` (the input grid)
- **`propagation_speed`** — Wave propagation speed $c$ in m/s, positive and finite; the default is the speed of light
  for radio waves, use about 343 for acoustic arrays (e.g. from [[generate_speaker]]); default: 299792458.0

## Output:
- **`arrayant_out`** — Arrayant dict with `n_ports` elements, each holding the pattern of that port; `name` is kept,
  `center_freq` is the output frequency; 3D pattern arrays for one output frequency, otherwise 4D with one entry per
  output frequency

## Notes:
- **Frequency interpolation** (4D input or several output frequencies): the fields are interpolated over frequency as in
  [[interpolate_pattern]], the port weights spherically between the two entries that bracket the output frequency.
  Output frequencies outside the entries take the nearest entry. The position phase uses the output frequency

## See also:
- [[generate_antenna]] (creates arrays with port weights)
- [[interpolate_pattern]] (evaluates the element patterns without the port weights)
- [[calc_directivity]], [[calc_beamwidth]] (work per element, so they measure the port patterns of the result)
- [[rotate_pattern]] (turns the combined pattern)
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
Join the elements of two antenna arrays into a new array

- Combines two arrays with identical sampling grids, e.g. the drivers of a loudspeaker; `name` and `center_freq` are
  taken from the first array
- The ports stay separate: elements of the first array connect only to its ports, and likewise for the second:<br><br>
   | Element \ Port        | P1…Pp1 (first) | Pp1+1…Pp1+p2 (second) |
   | --------------------- | :------------: | :-------------------: |
   | E1…En1 (first)        |    W1 block    |           0           |
   | En1+1…En1+n2 (second) |       0        |       W2 block        |

## Usage:
```
arrayant_out = quadriga_lib.arrayant.join_arrays( arrayant1, arrayant2 )
```

## Inputs:
- **`arrayant1`** — First array; arrayant dict ([[generate_antenna]]), 3D or 4D (multi-frequency) pattern arrays; its
  elements come first in the output
- **`arrayant2`** — Second array with the same sampling grids; 4D input needs as many frequency entries as the first and
  matching `center_freq` values at each index

## Output:
- **`arrayant_out`** — New arrayant dict with `n_elem1 + n_elem2` elements and `n_ports1 + n_ports2` ports, in the form
  of the inputs (3D or 4D)

## See also:
- [[copy_element]] (duplicates an element within one array)
- [[generate_speaker]] (creates the driver models to join)
- [[qdant_write]] (stores the joined model)
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
Copy a single element of an antenna array to one or more destination slots

- Copies the pattern and the position of an element, e.g. to build an array from a single element; the port weights of
  existing elements do not change
- A destination beyond the last element enlarges the array. Each added element gets its own new port with weight 1;
  added elements that are not a destination have a zero pattern and position

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
- **`arrayant`** — Arrayant dict ([[generate_antenna]]); 4D pattern arrays hold a multi-frequency array; the same copy
  is made at every frequency
- **`i_source`** — Index of the element to copy, 0-based, below `n_elements`; int, or list or array `(n_copy,)` for
  pairwise copies from `i_source[k]` to `i_target[k]`
- **`i_target`** — Destination indices, 0-based, not empty; int, or list or array `(n_copy,)`; a single `i_source` is
  copied to each of them, several need the same length

## Output:
- **`arrayant_out`** — Arrayant dict with the copies, in the form of `arrayant` (3D or 4D)

## See also:
- [[join_arrays]] (joins two arrays)
- [[generate_antenna]] (creates the element to copy)
- [[rotate_pattern]] (turns single copies, through `i_element`)
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
Rotate the patterns of an antenna array using Euler rotations

- Turns the patterns, the polarization and the element positions of an array, e.g. for a mechanical downtilt. The
  rotated pattern equals, at the grid points, what [[interpolate_pattern]] returns for the unrotated pattern with the
  same `orientation`
- Port weights, `center_freq` and `name` are kept

## Usage:
```
arrayant_out = quadriga_lib.arrayant.rotate_pattern( arrayant, orientation, *, usage, i_element )
```

## Inputs:
- **`arrayant`** — Arrayant dict ([[generate_antenna]]); 4D pattern arrays hold a multi-frequency array
- **`orientation`** — Euler angles (bank, tilt, heading) in rad, finite; a positive tilt turns the main beam down, as in
  3GPP TR 38.901; 3 elements
- **`usage`** — Rotation mode; default: 0. Modes 0 and 1 adjust the sampling grid of non-uniformly sampled patterns,
  which needs all elements: for such a pattern, a subset in `i_element` raises an error. 4D input never adjusts the
  grid, so that all entries keep the same grid: there 0 equals 3, and 1 equals 4. Element positions turn in all modes
  except 2. For scalar acoustic fields (pressure in `e_theta_re` only), use 1 to avoid spurious polarization
  effects:<br><br>
   | Mode | Pattern | Polarization | Grid adjustment |
   | ---- | ------- | ------------ | --------------- |
   | 0    | Yes     | Yes          | Yes             |
   | 1    | Yes     | No           | Yes             |
   | 2    | No      | Yes          | No              |
   | 3    | Yes     | Yes          | No              |
   | 4    | Yes     | No           | No              |
- **`i_element`** — Element indices to rotate, 0-based, no duplicates; default: `None` (all elements)

## Output:
- **`arrayant_out`** — Rotated arrayant dict, in the form of `arrayant` (3D or 4D)

## See also:
- [[interpolate_pattern]] (evaluates a pattern for an orientation without changing it)
- [[calc_rotation_matrix]] (rotation matrix of the orientation)
- [[generate_antenna]] (electrical tilt through the port weights)
- [[join_arrays]] (joins arrays that are then rotated together)
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