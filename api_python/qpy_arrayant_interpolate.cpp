// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "python_quadriga_adapter.hpp"

/*!SECTION
Array antenna functions
SECTION!*/

/*!MD
# interpolate
Interpolate array antenna field patterns

## Description:
- This function interpolates polarimetric antenna field patterns for a given set of azimuth and
  elevation angles. It supports both single-frequency arrayants (3D pattern fields) and multi-frequency
  arrayants (4D pattern fields). The function auto-detects the input format by inspecting the
  dimensionality of the `e_theta_re` field (3D = single-frequency, 4D = multi-frequency).
- For multi-frequency inputs, the function interpolates both spatially (azimuth/elevation) and across
  frequency, producing output fields with an additional frequency dimension. Target frequencies for
  the multi-frequency path are specified via the `frequency` parameter. The options `dist` and
  `local_angles` are only available for the single-frequency path without `frequency`.
- If a single-frequency arrayant (3D patterns) is used together with the `frequency` parameter, the
  function performs spatial interpolation once and replicates the 2D result across all requested
  frequencies, yielding 3D output of shape `(n_out, n_ang, n_freq_out)`. This allows uniform output
  shapes regardless of whether the underlying arrayant is frequency-dependent or not.

## Usage:

```
from quadriga_lib import arrayant

# Minimal example (single-frequency, 3D patterns)
vr,vi,hr,hi = arrayant.interpolate(arrayant, azimuth, elevation)

# Output as complex type
v,h = arrayant.interpolate(arrayant, azimuth, elevation, complex=1)

# Generate projected distance (single-frequency only)
vr,vi,hr,hi,dist = arrayant.interpolate(arrayant, azimuth, elevation, dist=1)
v,h,dist = arrayant.interpolate(arrayant, azimuth, elevation, complex=1, dist=1)

# Additional inputs
vr,vi,hr,hi = arrayant.interpolate(arrayant, azimuth, elevation, element, orientation, element_pos)

# Output angles in antenna-local coordinates (single-frequency only)
vr,vi,hr,hi,az_local,el_local,gamma = arrayant.interpolate(arrayant, azimuth, elevation, orientation=ori, local_angles=1)

# Multi-frequency interpolation (4D patterns)
vr,vi,hr,hi = arrayant.interpolate(speaker, azimuth, elevation, frequency=freqs)
v,h = arrayant.interpolate(speaker, azimuth, elevation, frequency=freqs, complex=1)

# Single-frequency arrayant with frequency duplication (output is 3D, duplicated across freqs)
vr,vi,hr,hi = arrayant.interpolate(ant, azimuth, elevation, frequency=freqs)
```

## Input Arguments:
- **`arrayant`** (required)<br>
  Dictionary containing array antenna data. Pattern fields may be 3D (single-frequency) or
  4D (multi-frequency, 4th dimension = frequency). The following keys are expected:
  `e_theta_re`     | Real part of e-theta field component             | Shape: `(n_elevation, n_azimuth, n_elements)` or `(n_elevation, n_azimuth, n_elements, n_freq)`
  `e_theta_im`     | Imaginary part of e-theta field component        | Shape: `(n_elevation, n_azimuth, n_elements)` or `(n_elevation, n_azimuth, n_elements, n_freq)`
  `e_phi_re`       | Real part of e-phi field component               | Shape: `(n_elevation, n_azimuth, n_elements)` or `(n_elevation, n_azimuth, n_elements, n_freq)`
  `e_phi_im`       | Imaginary part of e-phi field component          | Shape: `(n_elevation, n_azimuth, n_elements)` or `(n_elevation, n_azimuth, n_elements, n_freq)`
  `azimuth_grid`   | Azimuth angles in [rad] -pi to pi, sorted        | Shape: `(n_azimuth)`
  `elevation_grid` | Elevation angles in [rad], -pi/2 to pi/2, sorted | Shape: `(n_elevation)`
  `element_pos`    | Antenna element (x,y,z) positions, optional      | Shape: `(3, n_elements)`
  `center_freq`    | Center frequency in [Hz], optional               | Scalar or 1D array `(n_freq)`

- **`azimuth`** (required)<br>
  Azimuth angles in [rad] for which the field pattern should be interpolated. Values must be between -pi and pi.
  Option 1:  | Use the same angles for all antenna elements (planar wave approximation)
             | Shape: `(1, n_ang)`
  Option 2:  | Provide different angles for each array element (e.g. for spherical waves)
             | Shape: `(n_out, n_ang)`

- **`elevation`** (required)<br>
  Elevation angles in [rad] for which the field pattern should be interpolated. Values must be between -pi/2 and pi/2.
  Option 1:  | Use the same angles for all antenna elements (planar wave approximation)
             | Shape: `(1, n_ang)`
  Option 2:  | Provide different angles for each array element (e.g. for spherical waves)
             | Shape: `[n_out, n_ang)`

- **`element`** (optional)<br>
  The element indices for which the interpolation should be done. Optional parameter. Values must
  be between 0 and n_elements-1. It is possible to duplicate elements, i.e. by passing `[1,1,2]`.
  If this parameter is not provided (or an empty array is passed), `i_element` is initialized
  to `[0:n_elements-1]`. In this case, `n_out = n_elements`.
  Shape: `(1, n_out)` or `(n_out, 1)` or empty `()`

- **`orientation`** (optional)<br>
  This (optional) 3-element vector describes the orientation of the array antenna or of individual
  array elements using Euler angles in [rad].
  Shape: `(3, 1)` or `(3, n_out)` or `(3, 1, n_ang)` or `(3, n_out, n_ang)` or empty `()`

- **`element_pos`** (optional)<br>
  Alternative positions of the array antenna elements in local cartesian coordinates (using units of [m]).
  If this parameter is not given, element positions from `arrayant` are used. If the `arrayant` has no
  positions, they are initialized to [0,0,0]. For example, when duplicating the first element by setting
  `element = [1,1]`, different element positions can be set for the two elements in the output.
  Shape: `(3, n_out)` or empty `()`

- **`frequency`** (optional)<br>
  Target frequencies in [Hz] for multi-frequency interpolation. When the input arrayant has 4D
  pattern fields, each requested frequency is interpolated between the two bracketing entries;
  out-of-range frequencies are clamped to the nearest entry. When the input arrayant has 3D
  pattern fields, the spatial interpolation is performed once and the result is duplicated across
  all requested frequencies. In both cases, the output shape gains a frequency dimension
  `(n_out, n_ang, n_freq_out)`. If empty, no frequency dimension is added (single-frequency
  path returns 2D outputs). The options `dist` and `local_angles` are not available when
  `frequency` is provided.
  Shape: `(n_freq_out)` or empty `()`

- **`complex`** (optional flag)<br>
  If set to 1, output is returned in complex notation. This reduces performance due to additional
  copies of the data in memory. Default: 0, false

- **`dist`** (optional flag)<br>
  Switch to calculate the effective distances for phase calculation. Only available for single-frequency
  arrayants (3D patterns). Default: 0, false

- **`local_angles`** (optional flag)<br>
  Switch to return the angles in antenna-local coordinates. These differ from the input when the
  orientation of the antenna is adjusted. Only available for single-frequency arrayants (3D patterns).
  Default: 0, false

- **`fast_access`** (optional flag)<br>
  If arrayant data is provided as numpy.ndarray of type double in Fortran-contiguous (column-major)
  order, `arrayant_interpolate` can access the Python memory directly without a conversion of the
  data. This will increase performance and is done by default. If the data is not in the correct
  format, a conversion is done in the background. Setting `fast_access` to 1 will skip the conversion
  and throw an error if the arrayant data is not correctly formatted. Only applies to the single-
  frequency path. Default: 0, false (convert)

## Derived inputs:
  `n_azimuth`      | Number of azimuth angles in the field pattern
  `n_elevation`    | Number of elevation angles in the field pattern
  `n_elements`     | Number of antenna elements in the field pattern of the array antenna
  `n_ang`          | Number of interpolation angles
  `n_out`          | Number of antenna elements in the generated output (may differ from n_elements)
  `n_freq`         | Number of frequency entries in the multi-frequency arrayant (4D input only)
  `n_freq_out`     | Number of target frequencies (multi-frequency path only)

## Output Arguments (single-frequency path):
- **`vr`**<br>
  Real part of the interpolated e-theta (vertical) field component. Shape `(n_out, n_ang)`

- **`vi`**<br>
  Imaginary part of the interpolated e-theta (vertical) field component. Shape `(n_out, n_ang)`

- **`hr`**<br>
  Real part of the interpolated e-phi (horizontal) field component. Shape `(n_out, n_ang)`

- **`hi`**<br>
  Imaginary part of the interpolated e-phi (horizontal) field component. Shape `(n_out, n_ang)`

- **`dist`** (optional)<br>
  The effective distances between the antenna elements when seen from the direction of the
  incident path. The distance is calculated by a projection of the array positions on the normal
  plane of the incident path. This is needed for calculating the phase of the antenna response.
  Only returned when `dist` flag is set to 1. Shape `(n_out, n_ang)`

- **`azimuth_loc`** (optional)<br>
  The azimuth angles in [rad] for the local antenna coordinate system, i.e., after applying the
  'orientation'. If no orientation vector is given, these angles are identical to the input
  azimuth angles. Only returned when `local_angles` flag is set to 1. Shape `(n_out, n_ang)`

- **`elevation_loc`** (optional)<br>
  The elevation angles in [rad] for the local antenna coordinate system, i.e., after applying the
  'orientation'. If no orientation vector is given, these angles are identical to the input
  elevation angles. Only returned when `local_angles` flag is set to 1. Shape `(n_out, n_ang)`

- **`gamma`** (optional)<br>
  Polarization rotation angles in [rad]. Only returned when `local_angles` flag is set to 1.
  Shape `(n_out, n_ang)`

## Output Arguments (multi-frequency path):
- **`vr`**<br>
  Real part of the interpolated e-theta (vertical) field component. Shape `(n_out, n_ang, n_freq_out)`

- **`vi`**<br>
  Imaginary part of the interpolated e-theta (vertical) field component. Shape `(n_out, n_ang, n_freq_out)`

- **`hr`**<br>
  Real part of the interpolated e-phi (horizontal) field component. Shape `(n_out, n_ang, n_freq_out)`

- **`hi`**<br>
  Imaginary part of the interpolated e-phi (horizontal) field component. Shape `(n_out, n_ang, n_freq_out)`

- **`v`** (complex mode)<br>
  Complex-valued interpolated e-theta (vertical) field component. Shape `(n_out, n_ang, n_freq_out)`

- **`h`** (complex mode)<br>
  Complex-valued interpolated e-phi (horizontal) field component. Shape `(n_out, n_ang, n_freq_out)`
MD!*/

py::tuple arrayant_interpolate(const py::dict &arrayant,                // Array antenna data
                               const py::array_t<double> &azimuth,      // Azimuth angles in [rad], Shape: `[1, n_ang]` or `[n_out, n_ang]`
                               const py::array_t<double> &elevation,    // Elevation angles in [rad], Shape: `[1, n_ang]` or `[n_out, n_ang]`
                               const py::array_t<arma::uword> &element, // Antenna element indices, 0-based
                               const py::array_t<double> &orientation,  // Euler angles
                               const py::array_t<double> &element_pos,  // Alternative positions of the array antenna elements
                               const py::array_t<double> &frequency,    // Target frequencies in [Hz] for multi-frequency interpolation
                               bool complex,                            // Switch to return output in complex form or separate Re/Im
                               bool dist,                               // Switch to calculate the effective distances
                               bool local_angles,                       // Switch to calculate the antenna-local angles (az, el, gamma)
                               bool fast_access)                        // Enforces fast memory access
{
    // Detect single vs multi-frequency from e_theta_re dimensionality
    py::array e_theta_re_arr = py::cast<py::array>(arrayant["e_theta_re"]);
    int nd = (int)e_theta_re_arr.request().ndim;

    // Parse shared inputs
    const auto az = qd_python_numpy2arma_Mat(azimuth, true);
    const auto el = qd_python_numpy2arma_Mat(elevation, true);
    const arma::uvec element_ind = qd_python_numpy2arma_Col(element, true);
    const auto ori = qd_python_numpy2arma_Cube(orientation, true);
    const auto elpos = qd_python_numpy2arma_Mat(element_pos, true);

    if (az.n_elem == 0)
        throw std::invalid_argument("Azimuth angles cannot be empty.");

    arma::uword n_ang = az.n_cols;

    if (nd == 4) // Multi-frequency path
    {
        if (dist)
            throw std::invalid_argument("Interpolate: 'dist' output is not supported for multi-frequency arrayants.");
        if (local_angles)
            throw std::invalid_argument("Interpolate: 'local_angles' output is not supported for multi-frequency arrayants.");

        auto ant_vec = qd_python_dict2arrayant_multi(arrayant, false);

        // Validate multi-frequency consistency
        std::string err = quadriga_lib::arrayant_is_valid_multi(ant_vec, true);
        if (!err.empty())
            throw std::invalid_argument(err);

        // Parse target frequencies
        const arma::vec freq = qd_python_numpy2arma_Col(frequency, true);
        if (freq.n_elem == 0)
            throw std::invalid_argument("Interpolate: 'frequency' must be provided for multi-frequency arrayants.");

        arma::uword n_freq_out = freq.n_elem;
        arma::uword n_out = (element_ind.n_elem == 0) ? ant_vec[0].n_elements() : element_ind.n_elem;

        // Allocate output cubes [n_out, n_ang, n_freq_out]
        arma::cube V_re, V_im, H_re, H_im;
        py::array_t<double> V_re_py, V_im_py, H_re_py, H_im_py;

        if (!complex)
        {
            V_re_py = qd_python_init_output(n_out, n_ang, n_freq_out, &V_re);
            V_im_py = qd_python_init_output(n_out, n_ang, n_freq_out, &V_im);
            H_re_py = qd_python_init_output(n_out, n_ang, n_freq_out, &H_re);
            H_im_py = qd_python_init_output(n_out, n_ang, n_freq_out, &H_im);
        }
        else
        {
            V_re.set_size(n_out, n_ang, n_freq_out);
            V_im.set_size(n_out, n_ang, n_freq_out);
            H_re.set_size(n_out, n_ang, n_freq_out);
            H_im.set_size(n_out, n_ang, n_freq_out);
        }

        // Call multi-frequency interpolation (validation already done above)
        quadriga_lib::arrayant_interpolate_multi(ant_vec, &az, &el, &freq, &V_re, &V_im, &H_re, &H_im,
                                                 element_ind, &ori, &elpos, false);

        // Assemble output
        ssize_t output_size = complex ? 2 : 4;
        py::tuple output(output_size);
        ssize_t ind = 0;

        if (complex)
        {
            output[ind++] = qd_python_copy2numpy(V_re, V_im);
            output[ind++] = qd_python_copy2numpy(H_re, H_im);
        }
        else
        {
            output[ind++] = std::move(V_re_py);
            output[ind++] = std::move(V_im_py);
            output[ind++] = std::move(H_re_py);
            output[ind++] = std::move(H_im_py);
        }

        return output;
    }
    else // Single-frequency path
    {
        const auto ant = qd_python_dict2arrayant(arrayant, true, fast_access);

        // Check if frequency duplication is requested
        const arma::vec freq = qd_python_numpy2arma_Col(frequency, true);
        arma::uword n_freq_out = freq.n_elem;

        arma::uword n_out = (element_ind.n_elem == 0) ? ant.n_elements() : element_ind.n_elem;

        if (n_freq_out > 0) // Duplicate 2D results into 3D cubes [n_out, n_ang, n_freq_out]
        {
            if (dist)
                throw std::invalid_argument("Interpolate: 'dist' output is not supported when 'frequency' is provided.");
            if (local_angles)
                throw std::invalid_argument("Interpolate: 'local_angles' output is not supported when 'frequency' is provided.");

            // Allocate 3D output and interpolate directly into the first slice (zero-copy)
            arma::cube V_re_3d, V_im_3d, H_re_3d, H_im_3d;
            py::array_t<double> V_re_py, V_im_py, H_re_py, H_im_py;

            if (!complex)
            {
                V_re_py = qd_python_init_output(n_out, n_ang, n_freq_out, &V_re_3d);
                V_im_py = qd_python_init_output(n_out, n_ang, n_freq_out, &V_im_3d);
                H_re_py = qd_python_init_output(n_out, n_ang, n_freq_out, &H_re_3d);
                H_im_py = qd_python_init_output(n_out, n_ang, n_freq_out, &H_im_3d);
            }
            else
            {
                V_re_3d.set_size(n_out, n_ang, n_freq_out);
                V_im_3d.set_size(n_out, n_ang, n_freq_out);
                H_re_3d.set_size(n_out, n_ang, n_freq_out);
                H_im_3d.set_size(n_out, n_ang, n_freq_out);
            }

            // Wrap first slice as Mat for direct interpolation (no copy)
            arma::mat V_re_s0(V_re_3d.slice_memptr(0), n_out, n_ang, false, true);
            arma::mat V_im_s0(V_im_3d.slice_memptr(0), n_out, n_ang, false, true);
            arma::mat H_re_s0(H_re_3d.slice_memptr(0), n_out, n_ang, false, true);
            arma::mat H_im_s0(H_im_3d.slice_memptr(0), n_out, n_ang, false, true);

            ant.interpolate(&az, &el, &V_re_s0, &V_im_s0, &H_re_s0, &H_im_s0, element_ind, &ori, &elpos);

            // Duplicate first slice to remaining slices via memcpy
            const size_t slice_bytes = (size_t)n_out * (size_t)n_ang * sizeof(double);
            for (arma::uword f = 1; f < n_freq_out; ++f)
            {
                std::memcpy(V_re_3d.slice_memptr(f), V_re_3d.slice_memptr(0), slice_bytes);
                std::memcpy(V_im_3d.slice_memptr(f), V_im_3d.slice_memptr(0), slice_bytes);
                std::memcpy(H_re_3d.slice_memptr(f), H_re_3d.slice_memptr(0), slice_bytes);
                std::memcpy(H_im_3d.slice_memptr(f), H_im_3d.slice_memptr(0), slice_bytes);
            }

            py::tuple output(complex ? 2 : 4);
            ssize_t ind = 0;
            if (complex)
            {
                output[ind++] = qd_python_copy2numpy(V_re_3d, V_im_3d);
                output[ind++] = qd_python_copy2numpy(H_re_3d, H_im_3d);
            }
            else
            {
                output[ind++] = std::move(V_re_py);
                output[ind++] = std::move(V_im_py);
                output[ind++] = std::move(H_re_py);
                output[ind++] = std::move(H_im_py);
            }
            return output;
        }

        // Standard 2D output (no frequency parameter) — zero-copy via init_output
        arma::mat V_re, V_im, H_re, H_im, dist_proj, azimuth_loc, elevation_loc, gamma;
        py::array_t<double> V_re_py, V_im_py, H_re_py, H_im_py, dist_proj_py, azimuth_loc_py, elevation_loc_py, gamma_py;

        if (!complex)
        {
            V_re_py = qd_python_init_output(n_out, n_ang, &V_re);
            V_im_py = qd_python_init_output(n_out, n_ang, &V_im);
            H_re_py = qd_python_init_output(n_out, n_ang, &H_re);
            H_im_py = qd_python_init_output(n_out, n_ang, &H_im);
        }
        if (dist)
            dist_proj_py = qd_python_init_output(n_out, n_ang, &dist_proj);
        if (local_angles)
        {
            azimuth_loc_py = qd_python_init_output(n_out, n_ang, &azimuth_loc);
            elevation_loc_py = qd_python_init_output(n_out, n_ang, &elevation_loc);
            gamma_py = qd_python_init_output(n_out, n_ang, &gamma);
        }

        if (dist && local_angles)
            ant.interpolate(&az, &el, &V_re, &V_im, &H_re, &H_im, element_ind, &ori, &elpos, &dist_proj, &azimuth_loc, &elevation_loc, &gamma);
        else if (dist)
            ant.interpolate(&az, &el, &V_re, &V_im, &H_re, &H_im, element_ind, &ori, &elpos, &dist_proj);
        else
            ant.interpolate(&az, &el, &V_re, &V_im, &H_re, &H_im, element_ind, &ori, &elpos);

        ssize_t output_size = 0;
        output_size += complex ? 2 : 4;
        output_size += dist ? 1 : 0;
        output_size += local_angles ? 3 : 0;

        py::tuple output(output_size);
        ssize_t ind = 0;
        if (complex)
        {
            output[ind++] = qd_python_copy2numpy(V_re, V_im);
            output[ind++] = qd_python_copy2numpy(H_re, H_im);
        }
        else
        {
            output[ind++] = std::move(V_re_py);
            output[ind++] = std::move(V_im_py);
            output[ind++] = std::move(H_re_py);
            output[ind++] = std::move(H_im_py);
        }
        if (dist)
            output[ind++] = std::move(dist_proj_py);
        if (local_angles)
        {
            output[ind++] = std::move(azimuth_loc_py);
            output[ind++] = std::move(elevation_loc_py);
            output[ind++] = std::move(gamma_py);
        }

        return output;
    }
}